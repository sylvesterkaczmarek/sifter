"""Tests for sifter.storage module."""

from __future__ import annotations

from pathlib import Path

import pytest

from sifter.storage import ContainerFile, Storage, StorageError


class TestContainerFile:
    """Tests for ContainerFile dataclass."""

    def test_create_release_file(self) -> None:
        """Can create release ContainerFile."""
        cf = ContainerFile(
            filename="vllm-0.14.0_0.0.5.sif",
            is_dev=False,
            size_bytes=1024 * 1024 * 500,
        )
        assert cf.filename == "vllm-0.14.0_0.0.5.sif"
        assert cf.is_dev is False
        assert cf.size_bytes == 1024 * 1024 * 500

    def test_create_dev_file(self) -> None:
        """Can create dev ContainerFile."""
        cf = ContainerFile(
            filename="vllm-0.14.0_0.0.5+devg1234567890ab.sif",
            is_dev=True,
        )
        assert cf.is_dev is True


class TestStorageLocal:
    """Tests for Storage local operations."""

    def test_local_path(self, tmp_path: Path) -> None:
        """local_path returns correct path."""
        storage = Storage(local_dir=tmp_path)

        path = storage.local_path("test.sif")

        assert path == tmp_path / "test.sif"

    @pytest.mark.parametrize("filename", ["../outside.sif", "/tmp/outside.sif", "a/b.sif"])
    def test_local_path_refuses_to_escape_its_root(self, tmp_path: Path, filename: str) -> None:
        storage = Storage(local_dir=tmp_path)

        with pytest.raises(StorageError, match="single filename component"):
            storage.local_path(filename)

    def test_local_exists_true(self, tmp_path: Path) -> None:
        """local_exists returns True when file exists."""
        storage = Storage(local_dir=tmp_path)
        (tmp_path / "test.sif").write_text("content")

        assert storage.local_exists("test.sif") is True

    def test_local_exists_false(self, tmp_path: Path) -> None:
        """local_exists returns False when file doesn't exist."""
        storage = Storage(local_dir=tmp_path)

        assert storage.local_exists("nonexistent.sif") is False

    def test_list_local_empty(self, tmp_path: Path) -> None:
        """list_local returns empty list for empty directory."""
        storage = Storage(local_dir=tmp_path)

        result = storage.list_local()

        assert result == []

    def test_list_local_with_files(self, tmp_path: Path) -> None:
        """list_local returns all .sif files."""
        storage = Storage(local_dir=tmp_path)
        (tmp_path / "a.sif").write_text("a")
        (tmp_path / "b.sif").write_text("b")
        (tmp_path / "not-sif.txt").write_text("c")

        result = storage.list_local()

        assert len(result) == 2
        assert result[0].filename == "a.sif"
        assert result[1].filename == "b.sif"

    def test_list_local_with_prefix(self, tmp_path: Path) -> None:
        """list_local filters by prefix."""
        storage = Storage(local_dir=tmp_path)
        (tmp_path / "vllm-0.14.0_0.0.5.sif").write_text("a")
        (tmp_path / "pytorch-2.9.1_0.0.5.sif").write_text("b")

        result = storage.list_local(prefix="vllm")

        assert len(result) == 1
        assert result[0].filename == "vllm-0.14.0_0.0.5.sif"

    def test_list_local_detects_dev(self, tmp_path: Path) -> None:
        """list_local correctly identifies dev builds."""
        storage = Storage(local_dir=tmp_path)
        (tmp_path / "vllm-0.14.0_0.0.5.sif").write_text("a")
        (tmp_path / "vllm-0.14.0_0.0.5+devg123.sif").write_text("b")

        result = storage.list_local()

        release = next(f for f in result if "+dev" not in f.filename)
        dev = next(f for f in result if "+dev" in f.filename)
        assert release.is_dev is False
        assert dev.is_dev is True

    def test_list_local_nonexistent_dir(self, tmp_path: Path) -> None:
        """list_local returns empty for nonexistent directory."""
        storage = Storage(local_dir=tmp_path / "nonexistent")

        result = storage.list_local()

        assert result == []

    def test_ensure_local_dir(self, tmp_path: Path) -> None:
        """ensure_local_dir creates directory."""
        new_dir = tmp_path / "new" / "nested" / "dir"
        storage = Storage(local_dir=new_dir)

        storage.ensure_local_dir()

        assert new_dir.exists()
