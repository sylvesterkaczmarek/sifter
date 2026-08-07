"""Build dependency graph for Sifter CLI."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

from sifter.models import BuildSpec


class BuildAction(Enum):
    """Action to take for a build node."""

    BUILD = "build"  # Build from scratch
    TAG_CACHED = "tag_cached"  # Tag existing cached container
    PULL_REMOTE = "pull_remote"  # Pull from remote registry


@dataclass
class DAGNode:
    """A node in the build dependency graph.

    Each node represents a container that needs to be built or acquired.
    Nodes are linked by dependencies (edges in the DAG).

    Attributes:
        build_spec: Build specification for this container
        action: What action to take (build, tag_cached, pull_remote)
        depends_on: List of nodes this node depends on
        dependents: List of nodes that depend on this node
        existing_filename: Filename of existing container (for tag_cached/pull_remote)
    """

    build_spec: BuildSpec
    action: BuildAction
    depends_on: list[DAGNode] = field(default_factory=list)
    dependents: list[DAGNode] = field(default_factory=list)
    existing_filename: str | None = None

    @property
    def build_name(self) -> str:
        """Build name (e.g., 'vllm-0.14.0')."""
        return self.build_spec.build.name

    @property
    def full_name(self) -> str:
        """Full name without .sif (e.g., 'vllm-0.14.0_v0.0.5')."""
        return self.build_spec.full_name

    @property
    def output_filename(self) -> str:
        """Output SIF filename."""
        return self.build_spec.output_filename


class DAGError(Exception):
    """Error in DAG construction or traversal."""


@dataclass
class BuildDAG:
    """Directed acyclic graph of build tasks.

    The DAG captures dependencies between container builds, allowing
    for correct ordering and SLURM job chaining.

    Attributes:
        nodes: Mapping of full_name to DAGNode
    """

    nodes: dict[str, DAGNode] = field(default_factory=dict)

    def add_node(
        self,
        build_spec: BuildSpec,
        action: BuildAction,
        dependency_full_names: list[str] | None = None,
        existing_filename: str | None = None,
    ) -> DAGNode:
        """Add a build task to the DAG.

        Args:
            build_spec: Build specification
            action: Action to take for this node
            dependency_full_names: Full names of dependency nodes
            existing_filename: Filename of existing container (for use_local/pull_remote)

        Returns:
            The created DAGNode

        Raises:
            DAGError: If a dependency doesn't exist in the DAG
        """
        full_name = build_spec.full_name
        if full_name in self.nodes:
            return self.nodes[full_name]

        node = DAGNode(
            build_spec=build_spec,
            action=action,
            existing_filename=existing_filename,
        )

        # Link dependencies
        if dependency_full_names:
            for dep_name in dependency_full_names:
                if dep_name not in self.nodes:
                    raise DAGError(
                        f"Dependency '{dep_name}' not found in DAG. "
                        f"Build dependencies must be added before dependents."
                    )
                dep_node = self.nodes[dep_name]
                node.depends_on.append(dep_node)
                dep_node.dependents.append(node)

        self.nodes[full_name] = node
        return node

    def get_node(self, full_name: str) -> DAGNode | None:
        """Get a node by full name.

        Args:
            full_name: Full name like 'vllm-0.14.0_v0.0.5'

        Returns:
            DAGNode if found, None otherwise
        """
        return self.nodes.get(full_name)

    def roots(self) -> list[DAGNode]:
        """Get nodes with no dependencies (can start immediately).

        Returns:
            List of root nodes
        """
        return [n for n in self.nodes.values() if not n.depends_on]

    def topological_order(self) -> Iterator[DAGNode]:
        """Yield nodes in topological order (dependencies first).

        Uses Kahn's algorithm for topological sorting.

        Yields:
            Nodes in order where dependencies come before dependents

        Raises:
            DAGError: If the graph contains a cycle
        """
        # Count incoming edges for each node
        in_degree: dict[str, int] = {
            name: len(node.depends_on) for name, node in self.nodes.items()
        }

        # Start with nodes that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        visited = 0

        while queue:
            name = queue.pop(0)
            node = self.nodes[name]
            visited += 1
            yield node

            # Reduce in-degree of dependents
            for dependent in node.dependents:
                dep_name = dependent.full_name
                in_degree[dep_name] -= 1
                if in_degree[dep_name] == 0:
                    queue.append(dep_name)

        if visited != len(self.nodes):
            raise DAGError("DAG contains a cycle - cannot determine build order")

    def builds_required(self) -> list[DAGNode]:
        """Get nodes that require building (BUILD action).

        Returns:
            List of nodes with BUILD action in topological order
        """
        return [n for n in self.topological_order() if n.action == BuildAction.BUILD]

    def pulls_required(self) -> list[DAGNode]:
        """Get nodes that require pulling from S3.

        Returns:
            List of nodes with PULL_REMOTE action
        """
        return [n for n in self.nodes.values() if n.action == BuildAction.PULL_REMOTE]

    def __len__(self) -> int:
        """Number of nodes in the DAG."""
        return len(self.nodes)

    def __bool__(self) -> bool:
        """True if DAG has any nodes."""
        return bool(self.nodes)
