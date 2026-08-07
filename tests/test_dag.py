"""Tests for sifter.dag module."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sifter.dag import BuildAction, BuildDAG, DAGError, DAGNode
from sifter.models import Build, BuildSpec, Step, cache_filename


def make_build_spec(
    name: str,
    version: str = "0.0.5",
    base_image: str | None = None,
) -> BuildSpec:
    """Helper to create a BuildSpec for testing."""
    build = Build(
        name=name,
        version=version,
        steps=[Step(path=Path(f"definitions/{name}/{name}.def"), args={})],
    )
    content_hash = "g" + hashlib.sha256(name.encode()).hexdigest()[:12]
    return BuildSpec(
        build=build,
        definition_path=build.steps[0].path,
        args_dict={},
        output_filename=cache_filename(content_hash),
        base_image=base_image,
        content_hash=content_hash,
        should_tag=True,
        registry_tag=build.tag,
    )


class TestDAGNode:
    """Tests for DAGNode class."""

    def test_build_name(self) -> None:
        """build_name returns the build name."""
        spec = make_build_spec("pytorch-2.9.1-cu126")
        node = DAGNode(build_spec=spec, action=BuildAction.BUILD)

        assert node.build_name == "pytorch-2.9.1-cu126"

    def test_full_name(self) -> None:
        """full_name returns full name without .sif."""
        spec = make_build_spec("pytorch-2.9.1-cu126", version="0.0.5")
        node = DAGNode(build_spec=spec, action=BuildAction.BUILD)

        assert node.full_name == spec.full_name

    def test_output_filename(self) -> None:
        """output_filename returns the SIF filename."""
        spec = make_build_spec("pytorch-2.9.1-cu126")
        node = DAGNode(build_spec=spec, action=BuildAction.BUILD)

        assert node.output_filename == spec.output_filename


class TestBuildDAG:
    """Tests for BuildDAG class."""

    def test_add_single_node(self) -> None:
        """Can add a single node without dependencies."""
        dag = BuildDAG()
        spec = make_build_spec("pytorch-2.9.1-cu126")

        node = dag.add_node(spec, BuildAction.BUILD)

        assert len(dag) == 1
        assert dag.get_node(spec.full_name) is node
        assert node.depends_on == []
        assert node.dependents == []

    def test_add_node_with_dependency(self) -> None:
        """Can add a node that depends on another."""
        dag = BuildDAG()
        pytorch_spec = make_build_spec("pytorch-2.9.1-cu126")
        vllm_spec = make_build_spec("vllm-0.14.0", base_image="pytorch-2.9.1-cu126_0.0.5.sif")

        pytorch_node = dag.add_node(pytorch_spec, BuildAction.BUILD)
        vllm_node = dag.add_node(
            vllm_spec,
            BuildAction.BUILD,
            dependency_full_names=[pytorch_spec.full_name],
        )

        assert len(dag) == 2
        assert vllm_node.depends_on == [pytorch_node]
        assert pytorch_node.dependents == [vllm_node]

    def test_add_node_missing_dependency_raises(self) -> None:
        """Adding node with missing dependency raises error."""
        dag = BuildDAG()
        vllm_spec = make_build_spec("vllm-0.14.0")

        with pytest.raises(DAGError, match=r"Dependency.*not found"):
            dag.add_node(
                vllm_spec,
                BuildAction.BUILD,
                dependency_full_names=[vllm_spec.full_name],
            )

    def test_add_duplicate_node_returns_existing(self) -> None:
        """Adding same node twice returns existing node."""
        dag = BuildDAG()
        spec = make_build_spec("pytorch-2.9.1-cu126")

        node1 = dag.add_node(spec, BuildAction.BUILD)
        node2 = dag.add_node(spec, BuildAction.BUILD)

        assert node1 is node2
        assert len(dag) == 1

    def test_roots_single_node(self) -> None:
        """Single node with no deps is a root."""
        dag = BuildDAG()
        spec = make_build_spec("pytorch-2.9.1-cu126")
        dag.add_node(spec, BuildAction.BUILD)

        roots = dag.roots()

        assert len(roots) == 1
        assert roots[0].build_name == "pytorch-2.9.1-cu126"

    def test_roots_with_chain(self) -> None:
        """Only base of dependency chain is root."""
        dag = BuildDAG()
        pytorch_spec = make_build_spec("pytorch-2.9.1-cu126")
        vllm_spec = make_build_spec("vllm-0.14.0")

        dag.add_node(pytorch_spec, BuildAction.BUILD)
        dag.add_node(
            vllm_spec,
            BuildAction.BUILD,
            dependency_full_names=[pytorch_spec.full_name],
        )

        roots = dag.roots()

        assert len(roots) == 1
        assert roots[0].build_name == "pytorch-2.9.1-cu126"

    def test_topological_order_single(self) -> None:
        """Single node yields correctly."""
        dag = BuildDAG()
        spec = make_build_spec("pytorch-2.9.1-cu126")
        dag.add_node(spec, BuildAction.BUILD)

        order = list(dag.topological_order())

        assert len(order) == 1
        assert order[0].build_name == "pytorch-2.9.1-cu126"

    def test_topological_order_chain(self) -> None:
        """Dependencies come before dependents."""
        dag = BuildDAG()
        pytorch_spec = make_build_spec("pytorch-2.9.1-cu126")
        vllm_spec = make_build_spec("vllm-0.14.0")

        dag.add_node(pytorch_spec, BuildAction.BUILD)
        dag.add_node(
            vllm_spec,
            BuildAction.BUILD,
            dependency_full_names=[pytorch_spec.full_name],
        )

        order = list(dag.topological_order())

        assert len(order) == 2
        # pytorch must come before vllm
        assert order[0].build_name == "pytorch-2.9.1-cu126"
        assert order[1].build_name == "vllm-0.14.0"

    def test_topological_order_diamond(self) -> None:
        """Diamond dependency handled correctly."""
        dag = BuildDAG()
        base_spec = make_build_spec("base-1.0")
        left_spec = make_build_spec("left-1.0")
        right_spec = make_build_spec("right-1.0")
        top_spec = make_build_spec("top-1.0")

        dag.add_node(base_spec, BuildAction.BUILD)
        dag.add_node(left_spec, BuildAction.BUILD, dependency_full_names=[base_spec.full_name])
        dag.add_node(right_spec, BuildAction.BUILD, dependency_full_names=[base_spec.full_name])
        dag.add_node(
            top_spec,
            BuildAction.BUILD,
            dependency_full_names=[left_spec.full_name, right_spec.full_name],
        )

        order = list(dag.topological_order())

        assert len(order) == 4
        # base must come first
        assert order[0].build_name == "base-1.0"
        # top must come last
        assert order[3].build_name == "top-1.0"
        # left and right must come before top (order between them doesn't matter)
        middle_names = {order[1].build_name, order[2].build_name}
        assert middle_names == {"left-1.0", "right-1.0"}

    def test_builds_required(self) -> None:
        """builds_required returns only BUILD action nodes."""
        dag = BuildDAG()
        pytorch_spec = make_build_spec("pytorch-2.9.1-cu126")
        vllm_spec = make_build_spec("vllm-0.14.0")

        dag.add_node(pytorch_spec, BuildAction.TAG_CACHED)
        dag.add_node(
            vllm_spec,
            BuildAction.BUILD,
            dependency_full_names=[pytorch_spec.full_name],
        )

        builds = dag.builds_required()

        assert len(builds) == 1
        assert builds[0].build_name == "vllm-0.14.0"

    def test_pulls_required(self) -> None:
        """pulls_required returns only PULL_REMOTE action nodes."""
        dag = BuildDAG()
        pytorch_spec = make_build_spec("pytorch-2.9.1-cu126")
        vllm_spec = make_build_spec("vllm-0.14.0")

        dag.add_node(pytorch_spec, BuildAction.PULL_REMOTE)
        dag.add_node(
            vllm_spec,
            BuildAction.BUILD,
            dependency_full_names=[pytorch_spec.full_name],
        )

        pulls = dag.pulls_required()

        assert len(pulls) == 1
        assert pulls[0].build_name == "pytorch-2.9.1-cu126"

    def test_len(self) -> None:
        """len() returns number of nodes."""
        dag = BuildDAG()
        assert len(dag) == 0

        dag.add_node(make_build_spec("pytorch-2.9.1-cu126"), BuildAction.BUILD)
        assert len(dag) == 1

    def test_bool_empty(self) -> None:
        """Empty DAG is falsy."""
        dag = BuildDAG()
        assert not dag

    def test_bool_nonempty(self) -> None:
        """Non-empty DAG is truthy."""
        dag = BuildDAG()
        dag.add_node(make_build_spec("pytorch-2.9.1-cu126"), BuildAction.BUILD)
        assert dag


class TestBuildAction:
    """Tests for BuildAction enum."""

    def test_action_values(self) -> None:
        """All expected action values exist."""
        assert BuildAction.BUILD.value == "build"
        assert BuildAction.TAG_CACHED.value == "tag_cached"
        assert BuildAction.PULL_REMOTE.value == "pull_remote"
