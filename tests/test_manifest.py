"""Tests for sifter.manifest module."""

from __future__ import annotations

from pathlib import Path

import pytest

from sifter.manifest import Manifest, ManifestError

from .conftest import load_manifest


class TestManifest:
    """Tests for Manifest class."""

    def test_load_valid_manifest(self, sample_repo: Path) -> None:
        """Manifest.load parses valid manifest."""
        manifest = Manifest.load(sample_repo / "sifter.yaml")

        assert len(manifest.builds) == 3
        assert "pytorch-2.9.1-cu126_0.0.5" in manifest.builds
        assert "vllm-0.14.0_0.0.5" in manifest.builds
        assert "vllm-0.14.1_0.0.5" in manifest.builds

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Manifest.load raises error for missing file."""
        with pytest.raises(ManifestError, match="not found"):
            Manifest.load(tmp_path / "nonexistent.yaml")

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        """Manifest.load raises error for invalid YAML."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("builds:\n  - this is not valid: ][")

        with pytest.raises(ManifestError, match="Invalid YAML"):
            Manifest.load(bad_yaml)

    def test_load_missing_builds_key(self, tmp_path: Path) -> None:
        """Manifest.load raises error for missing builds key."""
        manifest_path = tmp_path / "sifter.yaml"
        manifest_path.write_text("version: 1.0")

        with pytest.raises(ManifestError, match="'builds'"):
            Manifest.load(manifest_path)

    def test_load_invalid_base_reference(self, tmp_path: Path) -> None:
        """Manifest.load raises error for invalid base reference."""
        manifest_path = tmp_path / "sifter.yaml"
        manifest_path.write_text(load_manifest("undefined_base"))

        with pytest.raises(ManifestError, match=r"base.*not defined"):
            Manifest.load(manifest_path)


class TestBuildsParsing:
    """Tests for builds: manifest format parsing."""

    def test_simple_build(self, tmp_path: Path) -> None:
        """Single step build parses correctly."""
        manifest_path = tmp_path / "sifter.yaml"
        manifest_path.write_text(load_manifest("simple_build"))

        manifest = Manifest.load(manifest_path)

        assert len(manifest.builds) == 1
        build = manifest.builds["pytorch-base_0.0.1"]
        assert build.name == "pytorch-base"
        assert build.version == "0.0.1"
        assert len(build.steps) == 1
        assert build.steps[0].path == Path("definitions/pytorch/pytorch.def")
        assert build.steps[0].args == {"PYTORCH_VERSION": "2.9.1"}

    def test_multi_step_build(self, tmp_path: Path) -> None:
        """Multi-step build parses correctly."""
        manifest_path = tmp_path / "sifter.yaml"
        manifest_path.write_text(load_manifest("multi_step_build"))

        manifest = Manifest.load(manifest_path)
        build = manifest.builds["vllm-0.14.0_0.0.1"]

        assert len(build.steps) == 2
        assert build.steps[0].path.name == "pytorch.def"
        assert build.steps[1].path.name == "vllm.def"

    def test_build_with_base(self, tmp_path: Path) -> None:
        """Build with base image parses correctly."""
        manifest_path = tmp_path / "sifter.yaml"
        manifest_path.write_text(load_manifest("build_with_base"))

        manifest = Manifest.load(manifest_path)
        overlay = manifest.builds["overlay_1.0.0"]

        assert overlay.base == "pytorch-base_0.0.1"

    def test_shorthand_step_syntax(self, tmp_path: Path) -> None:
        """Step can be just a path string (no args)."""
        manifest_path = tmp_path / "sifter.yaml"
        manifest_path.write_text(load_manifest("shorthand_step"))

        manifest = Manifest.load(manifest_path)
        build = manifest.builds["myapp_1.0.0"]

        assert len(build.steps) == 1
        assert build.steps[0].path == Path("definitions/app.def")
        assert build.steps[0].args == {}

    def test_empty_builds_raises_error(self, tmp_path: Path) -> None:
        """Empty builds section raises error."""
        manifest_path = tmp_path / "sifter.yaml"
        manifest_path.write_text("builds: {}")

        with pytest.raises(ManifestError, match="No builds defined"):
            Manifest.load(manifest_path)

    def test_invalid_tag_format(self, tmp_path: Path) -> None:
        """Build key without underscore separator raises error."""
        manifest_path = tmp_path / "sifter.yaml"
        manifest_path.write_text("""
builds:
  invalid-no-version:
    steps:
      - definitions/app.def
""")

        with pytest.raises(ManifestError, match="name_version"):
            Manifest.load(manifest_path)

    def test_build_without_steps(self, tmp_path: Path) -> None:
        """Build without steps raises error."""
        manifest_path = tmp_path / "sifter.yaml"
        manifest_path.write_text("""
builds:
  myapp_1.0.0:
    base: something_1.0.0
""")

        with pytest.raises(ManifestError, match="at least one step"):
            Manifest.load(manifest_path)


class TestFlexibleTags:
    """Tests for non-digit tags in manifest."""

    def test_alpha_tag_parses(self, tmp_path: Path) -> None:
        """Builds with alphabetic tags like 'testing' parse correctly."""
        manifest_path = tmp_path / "sifter.yaml"
        manifest_path.write_text("""
builds:
  mybase_testing:
    steps:
      - definitions/base.def
""")

        manifest = Manifest.load(manifest_path)

        assert "mybase_testing" in manifest.builds
        build = manifest.builds["mybase_testing"]
        assert build.name == "mybase"
        assert build.version == "testing"

    def test_get_build_by_tag_with_alpha_tag(self, tmp_path: Path) -> None:
        """get_build_by_tag works with non-digit tags."""
        manifest_path = tmp_path / "sifter.yaml"
        manifest_path.write_text("""
builds:
  mybase_testing:
    steps:
      - definitions/base.def
""")

        manifest = Manifest.load(manifest_path)

        build = manifest.get_build_by_tag("mybase_testing")
        assert build is not None
        assert build.name == "mybase"

    def test_stable_dev_prod_tags(self, tmp_path: Path) -> None:
        """Common alpha tags like 'stable', 'dev', 'prod' work."""
        manifest_path = tmp_path / "sifter.yaml"
        manifest_path.write_text(load_manifest("alpha_tags"))

        manifest = Manifest.load(manifest_path)

        assert "myapp_stable" in manifest.builds
        assert "myapp_dev" in manifest.builds
        assert "myapp_prod" in manifest.builds


class TestManifestBaseValidation:
    """Tests for base reference validation in manifest."""

    def test_circular_dependency_via_base_raises_error(self, tmp_path: Path) -> None:
        """Circular dependency A→B→A via base references raises ManifestError."""
        (tmp_path / "sifter.yaml").write_text(load_manifest("circular_dependency"))
        with pytest.raises(ManifestError, match=r"[Cc]ircular|base.*not defined"):
            Manifest.load(tmp_path / "sifter.yaml")

    def test_self_referencing_base_raises_error(self, tmp_path: Path) -> None:
        """Build referencing itself as base raises ManifestError."""
        (tmp_path / "sifter.yaml").write_text(load_manifest("self_reference"))
        with pytest.raises(ManifestError):
            Manifest.load(tmp_path / "sifter.yaml")

    def test_base_referencing_undefined_build_raises_error(self, tmp_path: Path) -> None:
        """Base referencing a build not yet defined raises ManifestError."""
        (tmp_path / "sifter.yaml").write_text(load_manifest("undefined_base"))
        with pytest.raises(ManifestError, match=r"base.*not defined"):
            Manifest.load(tmp_path / "sifter.yaml")

    def test_diamond_dependency_pattern_allowed(self, tmp_path: Path) -> None:
        """Diamond pattern (D→B,C→A) is valid, not circular."""
        (tmp_path / "sifter.yaml").write_text(load_manifest("diamond_dependency"))
        manifest = Manifest.load(tmp_path / "sifter.yaml")
        assert manifest is not None

    def test_long_dependency_chain_allowed(self, tmp_path: Path) -> None:
        """Linear chain A→B→C→D (no cycle) is valid."""
        (tmp_path / "sifter.yaml").write_text(load_manifest("long_chain"))
        manifest = Manifest.load(tmp_path / "sifter.yaml")
        assert manifest is not None


class TestManifestEmptyValidation:
    """Tests for empty/null manifest validation."""

    def test_null_builds_section_raises_error(self, tmp_path: Path) -> None:
        """builds: null raises ManifestError."""
        (tmp_path / "sifter.yaml").write_text("builds: null")
        with pytest.raises(ManifestError, match="'builds' must be a mapping"):
            Manifest.load(tmp_path / "sifter.yaml")

    def test_build_with_empty_steps_raises_error(self, tmp_path: Path) -> None:
        """Build with steps: [] raises ManifestError."""
        (tmp_path / "sifter.yaml").write_text(load_manifest("empty_steps"))
        with pytest.raises(ManifestError, match="at least one step"):
            Manifest.load(tmp_path / "sifter.yaml")


class TestManifestMultipleBuilds:
    """Tests for manifests with multiple builds."""

    def test_multiple_unique_builds_allowed(self, tmp_path: Path) -> None:
        """Multiple builds with unique tags are allowed."""
        (tmp_path / "sifter.yaml").write_text(load_manifest("multiple_unique_builds"))
        manifest = Manifest.load(tmp_path / "sifter.yaml")
        assert len(manifest.builds) == 3
