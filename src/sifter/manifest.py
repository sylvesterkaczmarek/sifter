"""Manifest parsing and validation for Sifter CLI.

The manifest uses a `builds:` format with tags (name_version)
and multi-step build definitions.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from sifter.models import Build


class ManifestError(Exception):
    """Raised when the manifest is invalid or cannot be parsed."""


class Manifest:
    """Parsed sifter.yaml manifest.

    The manifest defines builds using tags (name_version).
    Each build consists of one or more steps (definition files) that
    are built in sequence.

    Example:
        builds:
          pytorch-base_0.0.1:
            steps:
              - path: definitions/base.def
                args:
                  PYTHON_VERSION: "3.12"

          vllm-0.14.0_0.0.1:
            base: pytorch-base_0.0.1
            steps:
              - path: definitions/vllm.def
                args:
                  VLLM_VERSION: "0.14.0"

    Attributes:
        builds: Mapping of tag to Build
        path: Path to the manifest file
    """

    def __init__(
        self,
        builds: dict[str, Build],
        path: Path | None = None,
    ):
        """Initialize manifest with builds.

        Args:
            builds: Mapping of tag to Build
            path: Optional path to manifest file (for error messages)
        """
        self._builds = builds
        self.path = path

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Load and validate a manifest from a YAML file.

        Args:
            path: Path to sifter.yaml

        Returns:
            Parsed Manifest instance

        Raises:
            ManifestError: If the file cannot be read or is invalid
        """
        try:
            with path.open() as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise ManifestError(f"Manifest not found: {path}") from None
        except yaml.YAMLError as e:
            raise ManifestError(f"Invalid YAML in {path}: {e}") from e

        if not isinstance(data, dict):
            raise ManifestError(f"Manifest must be a YAML mapping, got {type(data)}")

        if "builds" not in data:
            raise ManifestError("Manifest must have a 'builds' mapping at the top level")

        builds_data = data.get("builds")
        if not isinstance(builds_data, dict):
            raise ManifestError("'builds' must be a mapping")

        if not builds_data:
            raise ManifestError("No builds defined in manifest")

        builds: dict[str, Build] = {}
        for key, build_data in builds_data.items():
            try:
                build = Build.from_dict(key, build_data)
                builds[build.tag] = build
            except (KeyError, ValueError, TypeError) as e:
                raise ManifestError(f"Invalid build {key!r}: {e}") from e

        manifest = cls(builds=builds, path=path)
        manifest._validate_bases()
        return manifest

    def _validate_bases(self) -> None:
        """Validate that all base references point to valid builds and detect cycles.

        Raises:
            ManifestError: If any base reference is invalid or forms a cycle
        """
        for tag, build in self._builds.items():
            if build.base and build.base not in self._builds:
                raise ManifestError(
                    f"Build '{tag}' has base '{build.base}' which is not defined in the manifest"
                )

        # Detect cycles in base references
        for start_tag in self._builds:
            visited: set[str] = set()
            current = start_tag
            while current:
                if current in visited:
                    raise ManifestError(
                        f"Circular dependency detected: '{start_tag}' has a cycle through base references"
                    )
                visited.add(current)
                build = self._builds.get(current)
                current = build.base if build else None

    @property
    def builds(self) -> dict[str, Build]:
        """All builds in the manifest."""
        return self._builds.copy()

    def get_build_by_name(self, name: str) -> Build | None:
        """Find a build by name (first match).

        Args:
            name: Build name (e.g., "vllm-0.14.0")

        Returns:
            Build if found, None otherwise
        """
        for build in self._builds.values():
            if build.name == name:
                return build
        return None

    def get_build_by_tag(self, tag: str) -> Build | None:
        """Find a build by its full tag (name_version)."""
        return self._builds.get(tag)
