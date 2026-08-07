"""Tests for sifter.config module."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

from sifter.config import (
    ConfigError,
    FilesystemRegistryConfig,
    S3RegistryConfig,
    SifterConfig,
    print_config_summary,
)


class TestSifterConfig:
    """Tests for SifterConfig class (pydantic BaseSettings)."""

    def test_from_env_with_scratch(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """Config loads from SCRATCHDIR environment variable."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        # Clear any SIFTER_ env vars that might interfere
        monkeypatch.delenv("SIFTER_DIST", raising=False)
        monkeypatch.delenv("SIFTER_LOGS", raising=False)
        monkeypatch.delenv("SIFTER_CACHE", raising=False)
        monkeypatch.delenv("SIFTER_STAGING", raising=False)
        # repo_dir defaults to cwd
        monkeypatch.chdir(sample_repo)

        config = SifterConfig.from_env()

        assert config.dist_dir == scratch / "sifter" / "registry"
        assert config.logs_dir == scratch / "sifter" / "logs"
        assert config.repo_dir == sample_repo

    def test_from_env_with_custom_dist(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """Config respects SIFTER_DIST override."""
        custom_dist = sample_repo / "custom_dist"
        monkeypatch.setenv("SIFTER_DIST", str(custom_dist))
        monkeypatch.setenv("SCRATCHDIR", str(sample_repo / "scratch"))

        config = SifterConfig.from_env()

        assert config.dist_dir == custom_dist

    def test_from_env_with_custom_logs(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """Config respects SIFTER_LOGS override."""
        custom_logs = sample_repo / "custom_logs"
        monkeypatch.setenv("SIFTER_LOGS", str(custom_logs))
        monkeypatch.setenv("SCRATCHDIR", str(sample_repo / "scratch"))

        config = SifterConfig.from_env()

        assert config.logs_dir == custom_logs

    def test_from_env_missing_scratch_and_dist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config raises error if SCRATCHDIR missing and no override."""
        monkeypatch.delenv("SCRATCHDIR", raising=False)
        monkeypatch.delenv("SCRATCH", raising=False)
        monkeypatch.delenv("SIFTER_DIST", raising=False)

        with pytest.raises(ConfigError, match="Cannot determine dist directory"):
            SifterConfig.from_env()

    @pytest.mark.parametrize(
        ("skip_var", "expected"),
        [
            ("SIFTER_DIST", "Set SIFTER_DIST, SCRATCHDIR, or SCRATCH environment variable."),
            ("SIFTER_LOGS", "Set SIFTER_LOGS, SCRATCHDIR, or SCRATCH environment variable."),
            ("SIFTER_CACHE", "Set SIFTER_CACHE, SCRATCHDIR, or SCRATCH environment variable."),
            (
                "SIFTER_STAGING",
                "Set SIFTER_STAGING, SCRATCHDIR, or SCRATCH environment variable.",
            ),
        ],
    )
    def test_missing_scratch_error_mentions_scratch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        skip_var: str,
        expected: str,
    ) -> None:
        """Errors name every variable that can supply the directory, including SCRATCH."""
        monkeypatch.delenv("SCRATCHDIR", raising=False)
        monkeypatch.delenv("SCRATCH", raising=False)
        # Satisfy every directory except the one under test, so its branch is reached.
        for var in ("SIFTER_DIST", "SIFTER_LOGS", "SIFTER_CACHE", "SIFTER_STAGING"):
            if var == skip_var:
                monkeypatch.delenv(var, raising=False)
            else:
                monkeypatch.setenv(var, str(tmp_path / var.lower()))

        with pytest.raises(ConfigError, match=re.escape(expected)):
            SifterConfig.from_env()

    def test_slurm_defaults(self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> None:
        """SLURM resource defaults are tuned for Isambard-class GH200 nodes."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        config = SifterConfig.from_env()

        assert config.slurm_partition == "workq"
        assert config.slurm_cpus == 144
        assert config.slurm_gpus == 1
        assert config.slurm_mem == "0"

    def test_slurm_overrides_from_env(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """SIFTER_SLURM_* env vars override the SLURM resource defaults."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.setenv("SIFTER_SLURM_PARTITION", "gpu")
        monkeypatch.setenv("SIFTER_SLURM_CPUS", "32")
        monkeypatch.setenv("SIFTER_SLURM_GPUS", "4")
        monkeypatch.setenv("SIFTER_SLURM_MEM", "128G")
        monkeypatch.chdir(sample_repo)

        config = SifterConfig.from_env()

        assert config.slurm_partition == "gpu"
        assert config.slurm_cpus == 32
        assert config.slurm_gpus == 4
        assert config.slurm_mem == "128G"

    def test_manifest_path(self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> None:
        """Config.manifest_path returns correct path."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        config = SifterConfig.from_env()

        assert config.manifest_path == sample_repo / "sifter.yaml"

    def test_from_env_cache_dir(self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> None:
        """Config creates cache_dir from SCRATCHDIR."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.delenv("SIFTER_CACHE", raising=False)

        config = SifterConfig.from_env()

        assert config.cache_dir == scratch / "sifter" / "cache"

    def test_from_env_staging_dir(self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> None:
        """Config creates staging_dir from SCRATCHDIR."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.delenv("SIFTER_STAGING", raising=False)

        config = SifterConfig.from_env()

        assert config.staging_dir == scratch / "sifter" / "staging"

    def test_from_env_custom_cache_dir(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """Config respects SIFTER_CACHE override."""
        custom_cache = sample_repo / "custom_cache"
        monkeypatch.setenv("SIFTER_CACHE", str(custom_cache))
        monkeypatch.setenv("SCRATCHDIR", str(sample_repo / "scratch"))

        config = SifterConfig.from_env()

        assert config.cache_dir == custom_cache

    def test_from_env_custom_staging_dir(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """Config respects SIFTER_STAGING override."""
        custom_staging = sample_repo / "custom_staging"
        monkeypatch.setenv("SIFTER_STAGING", str(custom_staging))
        monkeypatch.setenv("SCRATCHDIR", str(sample_repo / "scratch"))

        config = SifterConfig.from_env()

        assert config.staging_dir == custom_staging


# Backwards compatibility alias test
class TestConfigAlias:
    """Test that Config is aliased to SifterConfig for backwards compatibility."""

    def test_config_alias_exists(self) -> None:
        """Config should be an alias for SifterConfig."""
        from sifter.config import Config, SifterConfig

        assert Config is SifterConfig


class TestRegistryConfig:
    """Tests for registry configuration models."""

    def test_s3_registry_config_effective_prefixes(self) -> None:
        config = S3RegistryConfig(
            registry_type="s3",
            s3_bucket="my-bucket",
            s3_prefix="containers/sifter",
        )
        assert config.effective_cache_prefix == "containers/sifter/cache"
        assert config.effective_registry_prefix == "containers/sifter/registry"

    def test_s3_registry_config_custom_prefixes(self) -> None:
        config = S3RegistryConfig(
            registry_type="s3",
            s3_bucket="my-bucket",
            s3_prefix="base/prefix",
            s3_cache_prefix="custom/cache",
            s3_registry_prefix="custom/registry",
        )
        assert config.effective_cache_prefix == "custom/cache"
        assert config.effective_registry_prefix == "custom/registry"

    def test_filesystem_registry_config_effective_cache_path(self, tmp_path: Path) -> None:
        config = FilesystemRegistryConfig(
            registry_type="filesystem",
            registry_path=tmp_path / "registry",
        )
        assert config.effective_cache_path == tmp_path / "cache"

    def test_filesystem_registry_config_custom_cache_path(self, tmp_path: Path) -> None:
        config = FilesystemRegistryConfig(
            registry_type="filesystem",
            registry_path=tmp_path / "registry",
            registry_cache_path=tmp_path / "custom_cache",
        )
        assert config.effective_cache_path == tmp_path / "custom_cache"

    def test_create_remote_cache_s3(self) -> None:
        from sifter.registry import S3Registry

        config = SifterConfig(
            dist_dir=Path("/tmp/dist"),
            logs_dir=Path("/tmp/logs"),
            cache_dir=Path("/tmp/cache"),
            staging_dir=Path("/tmp/staging"),
            repo_dir=Path("/tmp/repo"),
            registry=S3RegistryConfig(
                registry_type="s3", s3_bucket="my-bucket", s3_prefix="sifter"
            ),
        )
        remote_cache = config.create_remote_cache()
        assert remote_cache is not None
        assert isinstance(remote_cache, S3Registry)

    def test_create_remote_registry_filesystem(self, tmp_path: Path) -> None:
        from sifter.registry import FilesystemRegistry

        config = SifterConfig(
            dist_dir=Path("/tmp/dist"),
            logs_dir=Path("/tmp/logs"),
            cache_dir=Path("/tmp/cache"),
            staging_dir=Path("/tmp/staging"),
            repo_dir=Path("/tmp/repo"),
            registry=FilesystemRegistryConfig(
                registry_type="filesystem", registry_path=tmp_path / "registry"
            ),
        )
        remote_registry = config.create_remote_registry()
        assert remote_registry is not None
        assert isinstance(remote_registry, FilesystemRegistry)

    def test_create_remote_returns_none_when_no_registry(self) -> None:
        config = SifterConfig(
            dist_dir=Path("/tmp/dist"),
            logs_dir=Path("/tmp/logs"),
            cache_dir=Path("/tmp/cache"),
            staging_dir=Path("/tmp/staging"),
            repo_dir=Path("/tmp/repo"),
            registry=None,
        )
        assert config.create_remote_cache() is None
        assert config.create_remote_registry() is None
        assert config.remote_description == "not configured"

    def test_from_env_no_registry(self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> None:
        """Config works without registry configuration (local-only)."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(sample_repo / "nonexistent"))
        monkeypatch.delenv("SIFTER_REGISTRY", raising=False)

        config = SifterConfig.from_env()

        assert config.registry is None
        assert config.create_remote_cache() is None
        assert config.create_remote_registry() is None


class TestConfigSummaryWarnings:
    """The summary every CLI command prints must not imply signing cover it lacks."""

    @staticmethod
    def _config(registry: S3RegistryConfig | FilesystemRegistryConfig | None) -> SifterConfig:
        return SifterConfig(
            dist_dir=Path("/tmp/dist"),
            logs_dir=Path("/tmp/logs"),
            cache_dir=Path("/tmp/cache"),
            staging_dir=Path("/tmp/staging"),
            repo_dir=Path("/tmp/repo"),
            registry=registry,
        )

    @staticmethod
    def _unwrapped_output(capsys: pytest.CaptureFixture[str]) -> str:
        """Undo the console's line wrapping so a sentence can be matched."""
        return " ".join(capsys.readouterr().out.split())

    def test_filesystem_backend_is_reported_as_unsigned(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = self._config(
            FilesystemRegistryConfig(registry_type="filesystem", registry_path=tmp_path)
        )

        print_config_summary(config)

        assert "SIFTER_SIGN and SIFTER_VERIFY have no effect" in self._unwrapped_output(capsys)

    def test_the_unsigned_notice_is_not_repeated(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], recwarn: pytest.WarningsRecorder
    ) -> None:
        # Saying the same thing twice in one command teaches people to skip it.
        config = self._config(
            FilesystemRegistryConfig(registry_type="filesystem", registry_path=tmp_path)
        )

        print_config_summary(config)
        config.create_remote_registry()

        assert self._unwrapped_output(capsys).count("shared-filesystem registry backend") == 1
        assert not [w for w in recwarn if "shared-filesystem" in str(w.message)]

    def test_describing_the_remote_does_not_construct_it(self) -> None:
        # A label is not a reason to build a backend: constructing the S3 one raises a
        # deprecation warning, which under -W error kills a caller that only wanted a name.
        config = self._config(
            S3RegistryConfig(registry_type="s3", s3_bucket="b", s3_prefix="sifter")
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert config.remote_description == "s3://b/sifter/registry"

    @pytest.mark.parametrize(
        "registry",
        [
            S3RegistryConfig(registry_type="s3", s3_bucket="b", s3_prefix="sifter"),
            S3RegistryConfig(
                registry_type="s3", s3_bucket="b", s3_prefix="sifter", s3_registry_prefix="dist/"
            ),
            FilesystemRegistryConfig(registry_type="filesystem", registry_path=Path("/mnt/shared")),
        ],
    )
    def test_the_summary_label_matches_what_the_backend_calls_itself(
        self, registry: S3RegistryConfig | FilesystemRegistryConfig
    ) -> None:
        # remote_description derives the label from config instead of asking a
        # backend, so this is what stops the two wordings drifting apart.
        config = self._config(registry)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            backend = config.create_remote_registry()

        assert backend is not None
        assert config.remote_description == backend.description

    def test_the_oci_summary_label_matches_what_the_backend_calls_itself(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The OCI branch of the label is built from config too, and a registry may
        # be written with a scheme or a trailing slash the backend then drops.
        monkeypatch.setenv("SIFTER_REGISTRIES", "https://reg.example.com/team/")
        monkeypatch.setenv("SCRATCHDIR", str(tmp_path))
        monkeypatch.chdir(tmp_path)

        config = SifterConfig.from_env()
        backend = config.create_remote_registry()

        assert backend is not None
        assert config.remote_description == backend.description

    def test_s3_backend_is_reported_as_deprecated(self, capsys: pytest.CaptureFixture[str]) -> None:
        config = self._config(
            S3RegistryConfig(registry_type="s3", s3_bucket="b", s3_prefix="sifter")
        )

        print_config_summary(config)

        assert "deprecated" in self._unwrapped_output(capsys).lower()

    def test_no_remote_is_not_flagged_as_unsigned(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_config_summary(self._config(None))

        assert "cannot sign or verify" not in self._unwrapped_output(capsys)


class TestYamlConfigLoading:
    """Tests for YAML config file loading and registry resolution."""

    @pytest.fixture(autouse=True)
    def _clean_registry_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clear SIFTER_REGISTRY so real env doesn't interfere."""
        monkeypatch.delenv("SIFTER_REGISTRY", raising=False)

    def test_from_env_reads_registry_from_project_yaml(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(sample_repo / "nonexistent"))

        sifter_yaml = sample_repo / "sifter.yaml"
        content = sifter_yaml.read_text()
        content += "\nregistry:\n  registry_type: s3\n  s3_bucket: project-bucket\n  s3_prefix: project/prefix\n"
        sifter_yaml.write_text(content)

        config = SifterConfig.from_env()

        assert config.registry is not None
        assert isinstance(config.registry, S3RegistryConfig)
        assert config.registry.s3_bucket == "project-bucket"

    def test_from_env_reads_registry_from_user_config(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        user_config_dir = sample_repo / "fake_config" / "sifter"
        user_config_dir.mkdir(parents=True)
        user_config = user_config_dir / "config.yaml"
        user_config.write_text(
            "registry:\n  registry_type: s3\n  s3_bucket: user-bucket\n  s3_prefix: user/prefix\n"
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(sample_repo / "fake_config"))

        config = SifterConfig.from_env()

        assert config.registry is not None
        assert isinstance(config.registry, S3RegistryConfig)
        assert config.registry.s3_bucket == "user-bucket"

    def test_project_config_merges_with_user_config(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """Project config merges over user config, preserving unset keys."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        user_config_dir = sample_repo / "fake_config" / "sifter"
        user_config_dir.mkdir(parents=True)
        user_config = user_config_dir / "config.yaml"
        user_config.write_text(
            "registry:\n  registry_type: s3\n  s3_bucket: user-bucket\n  s3_prefix: user/prefix\n"
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(sample_repo / "fake_config"))

        sifter_yaml = sample_repo / "sifter.yaml"
        content = sifter_yaml.read_text()
        content += "\nregistry:\n  s3_bucket: project-bucket\n  s3_prefix: project/prefix\n"
        sifter_yaml.write_text(content)

        config = SifterConfig.from_env()

        assert config.registry is not None
        assert isinstance(config.registry, S3RegistryConfig)
        assert config.registry.s3_bucket == "project-bucket"
        assert config.registry.s3_prefix == "project/prefix"

    def test_env_var_overrides_yaml(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """SIFTER_REGISTRY env var (JSON) overrides YAML config."""
        import json

        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        sifter_yaml = sample_repo / "sifter.yaml"
        content = sifter_yaml.read_text()
        content += "\nregistry:\n  registry_type: s3\n  s3_bucket: yaml-bucket\n"
        sifter_yaml.write_text(content)

        monkeypatch.setenv(
            "SIFTER_REGISTRY",
            json.dumps(
                {"registry_type": "s3", "s3_bucket": "env-bucket", "s3_prefix": "env/prefix"}
            ),
        )

        config = SifterConfig.from_env()

        assert config.registry is not None
        assert isinstance(config.registry, S3RegistryConfig)
        assert config.registry.s3_bucket == "env-bucket"

    def test_filesystem_registry_from_yaml(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(sample_repo / "nonexistent"))

        sifter_yaml = sample_repo / "sifter.yaml"
        content = sifter_yaml.read_text()
        content += "\nregistry:\n  registry_type: filesystem\n  registry_path: /shared/containers/registry\n"
        sifter_yaml.write_text(content)

        config = SifterConfig.from_env()

        assert config.registry is not None
        assert isinstance(config.registry, FilesystemRegistryConfig)
        assert config.registry.registry_path == Path("/shared/containers/registry")

    def test_invalid_yaml_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(sample_repo / "nonexistent"))

        sifter_yaml = sample_repo / "sifter.yaml"
        sifter_yaml.write_text("registry:\n  bad: [unclosed\n")

        with pytest.raises(ConfigError, match="Failed to parse"):
            SifterConfig.from_env()
