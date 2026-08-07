"""Tests for sifter.api module (public convenience API)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sifter.api import (
    _stable_re,
    cache_info,
    cache_purge,
    find_latest_container,
    list_containers,
    remove_containers,
)
from sifter.models import CacheInfo, ContainerInfo
from sifter.storage import StorageError

# ---------------------------------------------------------------------------
# _stable_re regex tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "vllm-ext-0.16.0_0.1.0.sif",
        "vllm-ext-0.16.0_0.4.0.sif",
        "vllm-ext-1.0.0_2.3.4.sif",
    ],
)
def test_stable_re_prefixed_matches(name: str) -> None:
    """A prefixed regex matches stable release container names."""
    pat = _stable_re("vllm-ext-")
    assert pat.match(name)


@pytest.mark.parametrize(
    "name",
    [
        "vllm-ext-0.16.1rc0_0.1.0.sif",
        "vllm-ext-nightly-0bfa229_0.1.0.sif",
        "vllm-0.16.0_0.2.0.sif",
        "vllm-ext-0.16.0_0.1.0.tar",
        "vllm-ext-0.16.0_0.1.0.sif.bak",
    ],
)
def test_stable_re_prefixed_rejects(name: str) -> None:
    """A prefixed regex rejects non-stable container names."""
    pat = _stable_re("vllm-ext-")
    assert not pat.match(name)


@pytest.mark.parametrize(
    "name",
    [
        "vllm-ext-0.16.0_0.1.0.sif",
        "pytorch-2.5.0_0.1.0.sif",
        "some-tool-1.0.0_2.3.4.sif",
    ],
)
def test_stable_re_empty_prefix_matches_any_name(name: str) -> None:
    """The empty (default) prefix matches stable containers of any name."""
    pat = _stable_re("")
    assert pat.match(name)


@pytest.mark.parametrize(
    "name",
    [
        "vllm-ext-0.16.1rc0_0.1.0.sif",
        "pytorch-nightly-0bfa229_0.1.0.sif",
        "pytorch-2.5.0_0.1.0.tar",
    ],
)
def test_stable_re_empty_prefix_rejects_non_stable(name: str) -> None:
    """The empty prefix still rejects RC/nightly/non-.sif names."""
    pat = _stable_re("")
    assert not pat.match(name)


def test_stable_re_vllm_prefix_matches_plain_vllm() -> None:
    """_stable_re with 'vllm-' prefix matches plain vllm containers."""
    pat = _stable_re("vllm-")
    assert pat.match("vllm-0.16.0_0.2.0.sif")


def test_stable_re_vllm_prefix_excludes_vllm_lens() -> None:
    """_stable_re with 'vllm-' prefix does NOT match vllm-ext containers."""
    pat = _stable_re("vllm-")
    assert not pat.match("vllm-ext-0.16.0_0.4.0.sif")


# ---------------------------------------------------------------------------
# find_latest_container tests (filesystem-based)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_registry(tmp_path: Path) -> Path:
    """Create a fake sifter registry directory with sample .sif files."""
    files = [
        "vllm-ext-0.16.0_0.1.0.sif",
        "vllm-ext-0.16.0_0.2.0.sif",
        "vllm-ext-0.16.0_0.4.0.sif",
        "vllm-ext-0.16.1rc0_0.1.0.sif",
        "vllm-ext-0.16.1rc0_0.2.0.sif",
        "vllm-ext-nightly-0bfa229_0.1.0.sif",
        "vllm-0.16.0_0.2.0.sif",
    ]
    for name in files:
        (tmp_path / name).touch()
    return tmp_path


def _patch_registry(registry: Path):
    """Patch get_config to return a config pointing at the fake registry."""
    from unittest.mock import MagicMock

    mock_config = MagicMock()
    mock_config.dist_dir = registry
    return patch("sifter.api.get_config", return_value=mock_config)


def test_default_prefix_matches_all_stable(tmp_path: Path) -> None:
    """The bare default (empty prefix) matches stable containers of any name."""
    for name in (
        "pytorch-2.5.0_0.1.0.sif",
        "pytorch-2.5.0_0.2.0.sif",
        "zlib-1.3.1_0.1.0.sif",
        "zlib-1.3.1rc1_0.9.0.sif",
    ):
        (tmp_path / name).touch()
    with _patch_registry(tmp_path):
        result = find_latest_container()
    # Latest by sort order across every name, RCs excluded.
    assert Path(result).name == "zlib-1.3.1_0.1.0.sif"


def test_default_prefix_with_version_matches_any_name(tmp_path: Path) -> None:
    """The bare default plus --version narrows on version, not on name."""
    (tmp_path / "pytorch-2.5.0_0.1.0.sif").touch()
    (tmp_path / "pytorch-2.6.0_0.1.0.sif").touch()
    with _patch_registry(tmp_path):
        result = find_latest_container(version="2.5.0")
    assert Path(result).name == "pytorch-2.5.0_0.1.0.sif"


def test_prefixed_picks_latest_stable(fake_registry: Path) -> None:
    """A prefixed call (no version) returns the latest stable container."""
    with _patch_registry(fake_registry):
        result = find_latest_container(prefix="vllm-ext-")
    assert Path(result).name == "vllm-ext-0.16.0_0.4.0.sif"


def test_prefixed_excludes_rc_and_nightly(fake_registry: Path) -> None:
    """A prefixed call does not return RC or nightly containers."""
    with _patch_registry(fake_registry):
        result = find_latest_container(prefix="vllm-ext-")
    name = Path(result).name
    assert "rc" not in name
    assert "nightly" not in name


def test_version_picks_latest_for_version(fake_registry: Path) -> None:
    """version narrows to that version and picks the latest build."""
    with _patch_registry(fake_registry):
        result = find_latest_container(prefix="vllm-ext-", version="0.16.1rc0")
    assert Path(result).name == "vllm-ext-0.16.1rc0_0.2.0.sif"


def test_version_no_match_raises(fake_registry: Path) -> None:
    """RuntimeError when no container matches the requested version."""
    with (
        _patch_registry(fake_registry),
        pytest.raises(RuntimeError, match="No container"),
    ):
        find_latest_container(prefix="vllm-ext-", version="0.99.0")


def test_empty_registry_raises(tmp_path: Path) -> None:
    """RuntimeError when registry has no matching containers at all."""
    with _patch_registry(tmp_path), pytest.raises(RuntimeError, match="No container"):
        find_latest_container()


def test_version_ignores_stable(fake_registry: Path) -> None:
    """version="0.16.1rc0" does not accidentally match stable 0.16.0 containers."""
    with _patch_registry(fake_registry):
        result = find_latest_container(prefix="vllm-ext-", version="0.16.1rc0")
    assert "0.16.0_" not in Path(result).name


def test_stable_version_via_version(fake_registry: Path) -> None:
    """version can also select a stable version explicitly."""
    with _patch_registry(fake_registry):
        result = find_latest_container(prefix="vllm-ext-", version="0.16.0")
    assert Path(result).name == "vllm-ext-0.16.0_0.4.0.sif"


def test_vllm_prefix_default_picks_plain_vllm(fake_registry: Path) -> None:
    """prefix='vllm-' without version returns the plain vllm container."""
    with _patch_registry(fake_registry):
        result = find_latest_container(prefix="vllm-")
    name = Path(result).name
    assert name == "vllm-0.16.0_0.2.0.sif"
    assert "vllm-ext" not in name


def test_vllm_prefix_with_version_picks_plain_vllm(fake_registry: Path) -> None:
    """prefix='vllm-' with version='0.16.0' returns the plain vllm container."""
    with _patch_registry(fake_registry):
        result = find_latest_container(prefix="vllm-", version="0.16.0")
    name = Path(result).name
    assert name == "vllm-0.16.0_0.2.0.sif"
    assert "vllm-ext" not in name


def test_vllm_prefix_no_match_raises(fake_registry: Path) -> None:
    """prefix='vllm-' with non-existent version raises RuntimeError."""
    with (
        _patch_registry(fake_registry),
        pytest.raises(RuntimeError, match="No container"),
    ):
        find_latest_container(prefix="vllm-", version="0.99.0")


# ---------------------------------------------------------------------------
# list_containers tests
# ---------------------------------------------------------------------------


def _patch_config_for_list(registry: Path):
    """Patch get_config for list_containers."""
    mock_config = MagicMock()
    mock_config.dist_dir = registry
    mock_config.create_remote_registry.return_value = None
    return patch("sifter.api.get_config", return_value=mock_config)


@pytest.fixture
def local_registry(tmp_path: Path) -> Path:
    """Create a local registry with a mix of parseable and unparseable files."""
    files = [
        "vllm-0.14.0_0.0.5.sif",
        "vllm-0.14.0_0.0.6.sif",
        "myapp_testing.sif",
        "g7354f89abc1.sif",  # cache file, unparseable
    ]
    for name in files:
        (tmp_path / name).touch()
    return tmp_path


class TestListContainers:
    """Tests for list_containers function."""

    def test_lists_local_containers(self, local_registry: Path) -> None:
        """Returns ContainerInfo for parseable local files."""
        with _patch_config_for_list(local_registry):
            result = list_containers()

        assert len(result) == 3
        assert all(isinstance(c, ContainerInfo) for c in result)
        assert all(c.location == "local" for c in result)

    def test_skips_unparseable_filenames(self, local_registry: Path) -> None:
        """Cache files (hash-only names) are silently skipped."""
        with _patch_config_for_list(local_registry):
            result = list_containers()

        filenames = [c.filename for c in result]
        assert "g7354f89abc1.sif" not in filenames

    def test_name_filter(self, local_registry: Path) -> None:
        """Name filter returns only matching containers."""
        with _patch_config_for_list(local_registry):
            result = list_containers(name="vllm-0.14.0")

        assert len(result) == 2
        assert all(c.name == "vllm-0.14.0" for c in result)

    def test_tag_filter(self, local_registry: Path) -> None:
        """Tag filter returns only matching containers."""
        with _patch_config_for_list(local_registry):
            result = list_containers(tag="testing")

        assert len(result) == 1
        assert result[0].tag == "testing"
        assert result[0].name == "myapp"

    def test_name_and_tag_filter(self, local_registry: Path) -> None:
        """Both filters applied together."""
        with _patch_config_for_list(local_registry):
            result = list_containers(name="vllm-0.14.0", tag="0.0.5")

        assert len(result) == 1
        assert result[0].filename == "vllm-0.14.0_0.0.5.sif"

    def test_no_match_returns_empty(self, local_registry: Path) -> None:
        """Filter with no matches returns empty list."""
        with _patch_config_for_list(local_registry):
            result = list_containers(name="nonexistent")

        assert result == []

    def test_empty_registry(self, tmp_path: Path) -> None:
        """Empty registry returns empty list."""
        with _patch_config_for_list(tmp_path):
            result = list_containers()

        assert result == []

    def test_sorted_by_filename(self, local_registry: Path) -> None:
        """Results are sorted by filename."""
        with _patch_config_for_list(local_registry):
            result = list_containers()

        filenames = [c.filename for c in result]
        assert filenames == sorted(filenames)

    def test_remote_raises_when_no_remote_configured(self, tmp_path: Path) -> None:
        """remote=True raises StorageError when no remote registry is configured."""
        with _patch_config_for_list(tmp_path), pytest.raises(StorageError):
            list_containers(remote=True)

    def test_parses_fields_correctly(self, local_registry: Path) -> None:
        """ContainerInfo fields are populated from parsed filename."""
        with _patch_config_for_list(local_registry):
            result = list_containers(name="vllm-0.14.0", tag="0.0.5")

        info = result[0]
        assert info.name == "vllm-0.14.0"
        assert info.tag == "0.0.5"
        assert info.filename == "vllm-0.14.0_0.0.5.sif"
        assert info.is_dev is False


# ---------------------------------------------------------------------------
# cache_info / cache_purge tests
# ---------------------------------------------------------------------------


def _patch_config_for_cache(cache_dir, dist_dir):
    """Patch get_config for cache operations."""
    mock_config = MagicMock()
    mock_config.cache_dir = cache_dir
    mock_config.dist_dir = dist_dir
    mock_config.create_remote_registry.return_value = None
    return patch("sifter.api.get_config", return_value=mock_config)


class TestCacheInfo:
    """Tests for cache_info function."""

    def test_empty_cache(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        with _patch_config_for_cache(cache_dir, tmp_path):
            info = cache_info()
        assert info.file_count == 0
        assert info.total_bytes == 0

    def test_nonexistent_cache(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "nonexistent"
        with _patch_config_for_cache(cache_dir, tmp_path):
            info = cache_info()
        assert info.file_count == 0

    def test_cache_with_files(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "abc123.sif").write_bytes(b"x" * 100)
        (cache_dir / "def456.sif").write_bytes(b"y" * 200)
        with _patch_config_for_cache(cache_dir, tmp_path):
            info = cache_info()
        assert info.file_count == 2
        assert info.total_bytes == 300
        assert isinstance(info, CacheInfo)


class TestCachePurge:
    """Tests for cache_purge function."""

    def test_purge_deletes_files(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "abc123.sif").touch()
        (cache_dir / "def456.sif").touch()
        with _patch_config_for_cache(cache_dir, tmp_path):
            deleted = cache_purge()
        assert deleted == 2
        assert len(list(cache_dir.glob("*.sif"))) == 0

    def test_purge_empty_cache(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        with _patch_config_for_cache(cache_dir, tmp_path):
            deleted = cache_purge()
        assert deleted == 0


# ---------------------------------------------------------------------------
# remove_containers tests
# ---------------------------------------------------------------------------


class TestRemoveContainers:
    """Tests for remove_containers function."""

    def test_remove_by_prefix(self, tmp_path: Path) -> None:
        (tmp_path / "vllm-0.14.0_0.0.5.sif").touch()
        (tmp_path / "pytorch_0.0.1.sif").touch()
        with _patch_config_for_list(tmp_path):
            deleted = remove_containers(name="vllm")
        assert "vllm-0.14.0_0.0.5.sif" in deleted
        assert "pytorch_0.0.1.sif" not in deleted

    def test_remove_all(self, tmp_path: Path) -> None:
        (tmp_path / "vllm-0.14.0_0.0.5.sif").touch()
        (tmp_path / "pytorch_0.0.1.sif").touch()
        with _patch_config_for_list(tmp_path):
            deleted = remove_containers(name=None)
        assert len(deleted) == 2

    def test_remove_empty(self, tmp_path: Path) -> None:
        with _patch_config_for_list(tmp_path):
            deleted = remove_containers(name="nonexistent")
        assert deleted == []
