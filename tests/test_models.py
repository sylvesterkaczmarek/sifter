"""Tests for sifter.models module."""

from __future__ import annotations

from pathlib import Path

import pytest

from sifter.models import (
    Build,
    BuildJobResult,
    BuildResult,
    BuildSpec,
    CacheInfo,
    ContainerInfo,
    RunResult,
    Step,
    TransferResult,
    cache_filename,
    is_dev_filename,
    parse_filename,
)


class TestCacheFilename:
    """Tests for cache filename functions."""

    def test_cache_filename(self) -> None:
        """cache_filename generates correct format."""
        assert cache_filename("g7354f89abc1") == "g7354f89abc1.sif"


class TestBuildSpec:
    """Tests for BuildSpec dataclass."""

    def test_definition_file_path(self) -> None:
        """definition_file returns correct path."""
        build = Build(
            name="vllm-0.14.0",
            version="0.0.5",
            steps=[Step(path=Path("definitions/vllm/vllm.def"), args={})],
        )
        spec = BuildSpec(
            build=build,
            definition_path=build.steps[0].path,
            args_dict={},
            output_filename="vllm-0.14.0_0.0.5.sif",
            base_image=None,
            content_hash="g7354f89abc1",
            should_tag=True,
        )
        assert str(spec.definition_file) == "definitions/vllm/vllm.def"

    def test_full_name(self) -> None:
        """full_name removes .sif extension."""
        build = Build(
            name="vllm-0.14.0",
            version="0.0.5",
            steps=[Step(path=Path("definitions/vllm/vllm.def"), args={})],
        )
        spec = BuildSpec(
            build=build,
            definition_path=build.steps[0].path,
            args_dict={},
            output_filename="vllm-0.14.0_0.0.5.sif",
            base_image=None,
            content_hash="g7354f89abc1",
            should_tag=True,
        )
        assert spec.full_name == "vllm-0.14.0_0.0.5"


class TestParseFilename:
    """Tests for parse_filename function."""

    def test_parse_release_filename(self) -> None:
        """Parses release filename correctly."""
        name, tag, legacy_hash = parse_filename("vllm-0.14.0_0.0.5.sif")
        assert name == "vllm-0.14.0"
        assert tag == "0.0.5"
        assert legacy_hash is None

    def test_parse_filename_no_sif_extension(self) -> None:
        """Raises error for filename without .sif."""
        with pytest.raises(ValueError, match=r"must end with \.sif"):
            parse_filename("vllm-0.14.0_0.0.5")

    def test_parse_filename_no_tag_separator(self) -> None:
        """Raises error for filename without tag separator."""
        with pytest.raises(ValueError, match="must contain '_' separator"):
            parse_filename("vllm-0.14.0.sif")


class TestParseFilenameFlexibleTags:
    """Tests for parse_filename with flexible (non-digit) tags."""

    def test_parse_alpha_tag(self) -> None:
        """Parses filename with alphabetic tag like 'testing'."""
        name, tag, legacy_hash = parse_filename("danbase_testing.sif")
        assert name == "danbase"
        assert tag == "testing"
        assert legacy_hash is None

    def test_parse_dev_tag(self) -> None:
        """Parses filename with 'dev' tag."""
        name, tag, _ = parse_filename("myapp_dev.sif")
        assert (name, tag) == ("myapp", "dev")

    def test_parse_stable_tag(self) -> None:
        """Parses filename with 'stable' tag."""
        name, tag, _ = parse_filename("pytorch-base_stable.sif")
        assert (name, tag) == ("pytorch-base", "stable")

    def test_parse_numeric_version_still_works(self) -> None:
        """Backward compatibility with numeric versions."""
        name, tag, _ = parse_filename("vllm-0.14.0_0.0.5.sif")
        assert (name, tag) == ("vllm-0.14.0", "0.0.5")

    def test_parse_mixed_alphanumeric_tag(self) -> None:
        """Parses filename with alphanumeric tag like 'v1-beta'."""
        name, tag, _ = parse_filename("mycontainer_v1-beta.sif")
        assert (name, tag) == ("mycontainer", "v1-beta")


class TestIsDevFilename:
    """Tests for is_dev_filename function."""

    def test_release_is_not_dev(self) -> None:
        """Release filename returns False."""
        assert is_dev_filename("vllm-0.14.0_0.0.5.sif") is False

    def test_dev_is_dev(self) -> None:
        """Dev filename returns True."""
        assert is_dev_filename("vllm-0.14.0_0.0.5+devg7354f89abc1.sif") is True


class TestContainerInfo:
    """Tests for ContainerInfo dataclass."""

    def test_basic_construction(self) -> None:
        """ContainerInfo stores parsed container metadata."""
        info = ContainerInfo(
            name="vllm-0.14.0",
            tag="0.0.5",
            filename="vllm-0.14.0_0.0.5.sif",
            location="local",
        )
        assert info.name == "vllm-0.14.0"
        assert info.tag == "0.0.5"
        assert info.filename == "vllm-0.14.0_0.0.5.sif"
        assert info.location == "local"
        assert info.is_dev is False
        assert info.size_bytes is None
        assert info.modified is None

    def test_frozen(self) -> None:
        """ContainerInfo is immutable."""
        info = ContainerInfo(
            name="myapp",
            tag="testing",
            filename="myapp_testing.sif",
            location="remote",
        )
        with pytest.raises(AttributeError):
            info.name = "other"  # ty: ignore[invalid-assignment]

    def test_remote_location(self) -> None:
        """ContainerInfo can represent remote containers."""
        info = ContainerInfo(
            name="vllm-0.14.0",
            tag="0.0.5",
            filename="vllm-0.14.0_0.0.5.sif",
            location="remote",
            size_bytes=1024,
        )
        assert info.location == "remote"
        assert info.size_bytes == 1024


class TestCacheInfo:
    """Tests for CacheInfo dataclass."""

    def test_basic_construction(self) -> None:
        info = CacheInfo(path=Path("/tmp/cache"), file_count=5, total_bytes=1024)
        assert info.file_count == 5
        assert info.total_bytes == 1024

    def test_frozen(self) -> None:
        info = CacheInfo(path=Path("/tmp"), file_count=0, total_bytes=0)
        with pytest.raises(AttributeError):
            info.file_count = 1  # ty: ignore[invalid-assignment]


class TestTransferResult:
    """Tests for TransferResult dataclass."""

    def test_submitted(self) -> None:
        r = TransferResult(filename="test.sif", job_id="12345", skipped=False)
        assert r.job_id == "12345"
        assert r.skipped is False

    def test_skipped(self) -> None:
        r = TransferResult(filename="test.sif", job_id=None, skipped=True)
        assert r.job_id is None
        assert r.skipped is True


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_local_result(self) -> None:
        r = RunResult(exit_code=0)
        assert r.exit_code == 0
        assert r.job_id is None

    def test_slurm_result(self) -> None:
        r = RunResult(job_id="12345")
        assert r.exit_code is None
        assert r.job_id == "12345"


class TestBuildResult:
    """Tests for BuildResult and BuildJobResult dataclasses."""

    def test_empty_build_result(self) -> None:
        r = BuildResult()
        assert r.jobs == []
        assert r.dag is None

    def test_build_job_result(self) -> None:
        j = BuildJobResult(name="vllm", job_id="123", action="build")
        assert j.name == "vllm"
        assert j.action == "build"


# ============================================================================
# New models for builds: manifest format
# ============================================================================


class TestStep:
    """Tests for Step dataclass (new builds: format)."""

    def test_step_from_string_path(self) -> None:
        """Step can be created from string path (shorthand)."""
        step = Step.from_dict("definitions/pytorch/pytorch.def")
        assert step.path == Path("definitions/pytorch/pytorch.def")
        assert step.args == {}

    def test_step_from_dict_with_args(self) -> None:
        """Step can be created from dict with path and args."""
        data = {
            "path": "definitions/vllm/vllm.def",
            "args": {"VLLM_VERSION": "0.14.0", "CUDA_VERSION": "12.6"},
        }
        step = Step.from_dict(data)
        assert step.path == Path("definitions/vllm/vllm.def")
        assert step.args == {"VLLM_VERSION": "0.14.0", "CUDA_VERSION": "12.6"}

    def test_step_from_dict_minimal(self) -> None:
        """Step from dict with only path (no args)."""
        data = {"path": "definitions/base/base.def"}
        step = Step.from_dict(data)
        assert step.path == Path("definitions/base/base.def")
        assert step.args == {}

    def test_step_equality(self) -> None:
        """Two steps with same path and args are equal."""
        step1 = Step(path=Path("test.def"), args={"A": "1"})
        step2 = Step(path=Path("test.def"), args={"A": "1"})
        assert step1 == step2


class TestBuild:
    """Tests for Build dataclass (new builds: format)."""

    def test_build_from_manifest_entry(self) -> None:
        """Build parses 'name_version' key and steps."""
        key = "vllm-0.14.0_0.0.1"
        data = {
            "steps": [
                {"path": "definitions/pytorch/pytorch.def", "args": {"PYTORCH_VERSION": "2.9.1"}},
                {"path": "definitions/vllm/vllm.def", "args": {"VLLM_VERSION": "0.14.0"}},
            ]
        }
        build = Build.from_dict(key, data)

        assert build.name == "vllm-0.14.0"
        assert build.version == "0.0.1"
        assert build.tag == "vllm-0.14.0_0.0.1"
        assert len(build.steps) == 2

    def test_build_name_is_tag(self) -> None:
        """Build.tag combines name and version."""
        build = Build(
            name="myapp",
            version="1.0.0",
            steps=[Step(path=Path("app.def"), args={})],
        )
        assert build.tag == "myapp_1.0.0"

    def test_build_output_filename(self) -> None:
        """Build.output_filename returns tag.sif."""
        build = Build(
            name="myapp",
            version="1.0.0",
            steps=[Step(path=Path("app.def"), args={})],
        )
        assert build.output_filename == "myapp_1.0.0.sif"

    def test_build_with_base(self) -> None:
        """Build can have optional base image."""
        key = "overlay_1.0"
        data = {
            "base": "pytorch-base_0.0.1",
            "steps": [{"path": "definitions/overlay.def"}],
        }
        build = Build.from_dict(key, data)

        assert build.base == "pytorch-base_0.0.1"

    def test_build_single_step_shorthand(self) -> None:
        """Build can have single step as shorthand string."""
        key = "simple_1.0"
        data = {"steps": ["definitions/simple.def"]}
        build = Build.from_dict(key, data)

        assert len(build.steps) == 1
        assert build.steps[0].path == Path("definitions/simple.def")

    def test_build_invalid_key_format(self) -> None:
        """Build raises error for invalid key format (missing colon)."""
        with pytest.raises(ValueError, match="must be in format 'name_version'"):
            Build.from_dict("invalid-key", {"steps": ["test.def"]})

    @pytest.mark.parametrize(
        "key",
        [
            "../../outside_1.0",
            "/tmp/outside_1.0",
            r"..\outside_1.0",
        ],
    )
    def test_build_key_cannot_escape_the_registry_directory(self, key: str) -> None:
        with pytest.raises(ValueError, match="single filename component"):
            Build.from_dict(key, {"steps": ["test.def"]})

    def test_build_no_steps(self) -> None:
        """Build raises error when no steps provided."""
        with pytest.raises(ValueError, match="must have at least one step"):
            Build.from_dict("test_1.0", {"steps": []})
