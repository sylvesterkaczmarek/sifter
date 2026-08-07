"""Tests for sifter.cache module."""

from __future__ import annotations

from pathlib import Path

from sifter.cache import hash_definition


class TestHashDefinition:
    """Tests for hash_definition function."""

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        """Identical definition + args = identical hash."""
        def_file = tmp_path / "test.def"
        def_file.write_text("Bootstrap: docker\nFrom: ubuntu:22.04")

        hash1 = hash_definition(def_file, args={"KEY": "value"})
        hash2 = hash_definition(def_file, args={"KEY": "value"})

        assert hash1 == hash2

    def test_different_args_different_hash(self, tmp_path: Path) -> None:
        """Same definition, different args = different hash."""
        def_file = tmp_path / "test.def"
        def_file.write_text("Bootstrap: docker\nFrom: ubuntu:22.04")

        hash1 = hash_definition(def_file, args={"VERSION": "1.0"})
        hash2 = hash_definition(def_file, args={"VERSION": "2.0"})

        assert hash1 != hash2

    def test_args_order_independent(self, tmp_path: Path) -> None:
        """Args are sorted, so order doesn't matter."""
        def_file = tmp_path / "test.def"
        def_file.write_text("Bootstrap: docker\nFrom: ubuntu:22.04")

        hash1 = hash_definition(def_file, args={"A": "1", "B": "2"})
        hash2 = hash_definition(def_file, args={"B": "2", "A": "1"})

        assert hash1 == hash2

    def test_base_hash_affects_output_hash(self, tmp_path: Path) -> None:
        """Building on different base = different hash."""
        def_file = tmp_path / "test.def"
        def_file.write_text("Bootstrap: localimage\nFrom: {{ BASE_IMAGE }}")

        hash1 = hash_definition(def_file, base_image_hash="abc123")
        hash2 = hash_definition(def_file, base_image_hash="def456")

        assert hash1 != hash2

    def test_no_args_no_base(self, tmp_path: Path) -> None:
        """Hash works without args or base."""
        def_file = tmp_path / "test.def"
        def_file.write_text("Bootstrap: docker\nFrom: ubuntu:22.04")

        result = hash_definition(def_file)

        assert len(result) == 16
