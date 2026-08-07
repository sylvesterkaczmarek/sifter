"""Tests for sifter.hasher module."""

from __future__ import annotations

from pathlib import Path

import pytest

from sifter.hasher import HashCache, compute_content_hash
from sifter.models import Build, Step


class TestComputeContentHash:
    """Tests for compute_content_hash function."""

    def test_basic_hash(self, sample_repo: Path) -> None:
        """compute_content_hash returns valid hash format."""
        build = Build(
            name="pytorch",
            version="0.0.5",
            steps=[
                Step(
                    path=Path("definitions/pytorch/pytorch.def"),
                    args={"CUDA_VERSION": "12.6", "PYTORCH_VERSION": "2.9.1"},
                )
            ],
        )

        result = compute_content_hash(sample_repo, build)

        assert result.startswith("g")
        assert len(result) == 13  # 'g' + 12 hex chars
        assert all(c in "0123456789abcdef" for c in result[1:])

    def test_same_content_same_hash(self, sample_repo: Path) -> None:
        """Same content produces same hash."""
        build = Build(
            name="pytorch",
            version="0.0.5",
            steps=[
                Step(
                    path=Path("definitions/pytorch/pytorch.def"),
                    args={"CUDA_VERSION": "12.6", "PYTORCH_VERSION": "2.9.1"},
                )
            ],
        )

        hash1 = compute_content_hash(sample_repo, build)
        hash2 = compute_content_hash(sample_repo, build)

        assert hash1 == hash2

    def test_different_args_different_hash(self, sample_repo: Path) -> None:
        """Different args produce different hash."""
        build1 = Build(
            name="pytorch",
            version="0.0.5",
            steps=[
                Step(
                    path=Path("definitions/pytorch/pytorch.def"),
                    args={"CUDA_VERSION": "12.6", "PYTORCH_VERSION": "2.9.1"},
                )
            ],
        )
        build2 = Build(
            name="pytorch",
            version="0.0.5",
            steps=[
                Step(
                    path=Path("definitions/pytorch/pytorch.def"),
                    args={"CUDA_VERSION": "12.4", "PYTORCH_VERSION": "2.9.1"},
                )
            ],
        )

        hash1 = compute_content_hash(sample_repo, build1)
        hash2 = compute_content_hash(sample_repo, build2)

        assert hash1 != hash2

    def test_different_version_different_hash(self, sample_repo: Path) -> None:
        """Different container version produces different hash."""
        build1 = Build(
            name="pytorch",
            version="0.0.5",
            steps=[
                Step(
                    path=Path("definitions/pytorch/pytorch.def"),
                    args={"CUDA_VERSION": "12.6", "PYTORCH_VERSION": "2.9.1"},
                )
            ],
        )
        build2 = Build(
            name="pytorch",
            version="0.0.6",
            steps=[
                Step(
                    path=Path("definitions/pytorch/pytorch.def"),
                    args={"CUDA_VERSION": "12.6", "PYTORCH_VERSION": "2.9.1"},
                )
            ],
        )

        hash1 = compute_content_hash(sample_repo, build1)
        hash2 = compute_content_hash(sample_repo, build2)

        assert hash1 != hash2

    def test_different_def_file_different_hash(self, sample_repo: Path) -> None:
        """Modified .def file produces different hash."""
        build = Build(
            name="pytorch",
            version="0.0.5",
            steps=[
                Step(
                    path=Path("definitions/pytorch/pytorch.def"),
                    args={"CUDA_VERSION": "12.6"},
                )
            ],
        )

        hash1 = compute_content_hash(sample_repo, build)

        # Modify .def file
        def_file = sample_repo / "definitions" / "pytorch" / "pytorch.def"
        def_file.write_text(def_file.read_text() + "\n# modified")

        hash2 = compute_content_hash(sample_repo, build)

        assert hash1 != hash2

    def test_missing_def_file_raises(self, sample_repo: Path) -> None:
        """Missing .def file raises FileNotFoundError."""
        build = Build(
            name="pytorch",
            version="0.0.5",
            steps=[Step(path=Path("definitions/pytorch/pytorch.def"), args={})],
        )

        # Remove .def file
        (sample_repo / "definitions" / "pytorch" / "pytorch.def").unlink()

        with pytest.raises(FileNotFoundError):
            compute_content_hash(sample_repo, build)


class TestHashCache:
    """Tests for HashCache class."""

    def test_caches_result(self, sample_repo: Path) -> None:
        """HashCache caches computed hashes."""
        cache = HashCache(sample_repo)
        build = Build(
            name="pytorch",
            version="0.0.5",
            steps=[
                Step(path=Path("definitions/pytorch/pytorch.def"), args={"CUDA_VERSION": "12.6"})
            ],
        )

        hash1 = cache.get_hash(build)
        hash2 = cache.get_hash(build)

        assert hash1 == hash2
        assert len(cache._cache) == 1

    def test_clear_cache(self, sample_repo: Path) -> None:
        """clear() empties the cache."""
        cache = HashCache(sample_repo)
        build = Build(
            name="pytorch",
            version="0.0.5",
            steps=[
                Step(path=Path("definitions/pytorch/pytorch.def"), args={"CUDA_VERSION": "12.6"})
            ],
        )

        cache.get_hash(build)
        assert len(cache._cache) == 1

        cache.clear()
        assert len(cache._cache) == 0


class TestFlexibleTagsInDependencies:
    """Tests for dependencies with non-digit tags."""

    def test_dependency_with_alpha_tag(self, tmp_path: Path) -> None:
        """Dependencies with alphabetic tags like 'testing' should work."""
        from sifter.manifest import Manifest

        # Create container definition files
        base_dir = tmp_path / "definitions" / "base"
        base_dir.mkdir(parents=True)
        (base_dir / "base.def").write_text("Bootstrap: docker\nFrom: ubuntu")

        app_dir = tmp_path / "definitions" / "app"
        app_dir.mkdir(parents=True)
        (app_dir / "app.def").write_text("Bootstrap: localimage\nFrom: base.sif")

        # Create manifest with base at tag 'testing'
        manifest_content = """
builds:
  mybase_testing:
    steps:
      - path: definitions/base/base.def

  myapp_stable:
    base: mybase_testing
    steps:
      - path: definitions/app/app.def
"""
        (tmp_path / "sifter.yaml").write_text(manifest_content)
        manifest = Manifest.load(tmp_path / "sifter.yaml")

        # Get app container/variant
        app_build = manifest.get_build_by_name("myapp")
        assert app_build is not None

        # HashCache should handle the non-digit tag correctly
        hash_cache = HashCache(tmp_path, manifest)

        # Should not raise - the hash should compute successfully
        app_hash = hash_cache.get_hash(app_build)
        assert app_hash.startswith("g")
        assert len(app_hash) == 13

    def test_hash_cascades_for_same_alpha_tag(self, tmp_path: Path) -> None:
        """Hash should cascade when dependencies have same alpha tag.

        When myapp_testing depends on mybase_testing, changes to mybase
        should affect myapp's hash (same-tag cascading).
        """
        from sifter.manifest import Manifest

        # Create container definition files
        base_dir = tmp_path / "definitions" / "base"
        base_dir.mkdir(parents=True)
        (base_dir / "base.def").write_text("Bootstrap: docker\nFrom: ubuntu")

        app_dir = tmp_path / "definitions" / "app"
        app_dir.mkdir(parents=True)
        (app_dir / "app.def").write_text("Bootstrap: localimage\nFrom: base.sif")

        # Both at 'testing' tag
        manifest_content = """
builds:
  mybase_testing:
    steps:
      - path: definitions/base/base.def

  myapp_testing:
    base: mybase_testing
    steps:
      - path: definitions/app/app.def
"""
        (tmp_path / "sifter.yaml").write_text(manifest_content)
        manifest = Manifest.load(tmp_path / "sifter.yaml")

        app_build = manifest.get_build_by_name("myapp")
        assert app_build is not None

        hash_cache = HashCache(tmp_path, manifest)

        # Compute initial hash
        app_hash_v1 = hash_cache.get_hash(app_build)

        # Modify base's .def file
        (base_dir / "base.def").write_text("Bootstrap: docker\nFrom: ubuntu:24.04")
        hash_cache.clear()

        # Recompute hash - should change due to cascading
        app_hash_v2 = hash_cache.get_hash(app_build)

        assert app_hash_v1 != app_hash_v2, (
            "myapp hash should change when same-tag dependency mybase changes"
        )


class TestDependencyHashCascading:
    """Tests for dependency hash cascading (Bug 9)."""

    def test_hash_changes_when_same_version_dependency_changes(self, tmp_path: Path) -> None:
        """Container hash should change when same-version dependency hash changes.

        If pytorch's hash changes, vllm's hash should also change,
        even though vllm's own files haven't changed.
        """
        from sifter.manifest import Manifest

        # Create container definition files
        pytorch_dir = tmp_path / "definitions" / "pytorch"
        pytorch_dir.mkdir(parents=True)
        (pytorch_dir / "pytorch.def").write_text("Bootstrap: docker\nFrom: ubuntu")

        vllm_dir = tmp_path / "definitions" / "vllm"
        vllm_dir.mkdir(parents=True)
        (vllm_dir / "vllm.def").write_text("Bootstrap: localimage\nFrom: pytorch.sif")

        # Create manifest using new builds: format
        manifest_content = """
builds:
  pytorch-2.9_0.0.5:
    steps:
      - path: definitions/pytorch/pytorch.def

  vllm-0.14_0.0.5:
    base: pytorch-2.9_0.0.5
    steps:
      - path: definitions/vllm/vllm.def
"""
        (tmp_path / "sifter.yaml").write_text(manifest_content)
        manifest = Manifest.load(tmp_path / "sifter.yaml")

        # Get container/variant pairs via compatibility shim
        pytorch_build = manifest.get_build_by_name("pytorch-2.9")
        vllm_build = manifest.get_build_by_name("vllm-0.14")
        assert pytorch_build is not None
        assert vllm_build is not None

        hash_cache = HashCache(tmp_path, manifest)

        # Compute initial hashes
        vllm_hash_v1 = hash_cache.get_hash(vllm_build)

        # Modify pytorch's .def file
        (pytorch_dir / "pytorch.def").write_text("Bootstrap: docker\nFrom: ubuntu:24.04")
        hash_cache.clear()

        # Recompute vllm's hash
        vllm_hash_v2 = hash_cache.get_hash(vllm_build)

        # vllm's hash should change because its same-version dependency changed
        assert vllm_hash_v1 != vllm_hash_v2, (
            "vllm hash should change when same-version dependency changes"
        )

    def test_hash_not_affected_by_different_version_dependency(self, tmp_path: Path) -> None:
        """Hash should NOT change when modifying a dependency at a different version.

        If vllm depends on pytorch v0.0.4 (old release), changing pytorch
        at v0.0.5 locally should NOT affect vllm's hash. We should never
        compute the hash of a dependency at a different version.

        This is important because:
        - vllm will use pytorch v0.0.4 from S3 (immutable release)
        - Local pytorch v0.0.5 changes are irrelevant to this vllm build
        """
        from sifter.manifest import Manifest

        # Create container definition files
        pytorch_dir = tmp_path / "definitions" / "pytorch"
        pytorch_dir.mkdir(parents=True)
        (pytorch_dir / "pytorch.def").write_text("Bootstrap: docker\nFrom: ubuntu")

        vllm_dir = tmp_path / "definitions" / "vllm"
        vllm_dir.mkdir(parents=True)
        (vllm_dir / "vllm.def").write_text("Bootstrap: localimage\nFrom: pytorch.sif")

        # Create manifest with pytorch at v0.0.5, but vllm depending on OLD v0.0.4
        # Since v0.0.4 doesn't exist in manifest, we test that changing v0.0.5
        # doesn't affect the hash of vllm when it references a different version
        manifest_content = """
builds:
  pytorch-2.9_0.0.5:
    steps:
      - path: definitions/pytorch/pytorch.def
"""
        (tmp_path / "sifter.yaml").write_text(manifest_content)
        manifest = Manifest.load(tmp_path / "sifter.yaml")

        # vllm at v0.0.5 depends on pytorch at v0.0.4 (OLD version that's not in manifest)
        # This simulates the scenario where vllm would pull from S3
        vllm_build = Build(
            name="vllm-0.14",
            version="0.0.5",
            base="pytorch-2.9_0.0.4",
            steps=[Step(path=Path("definitions/vllm/vllm.def"), args={})],
        )

        hash_cache = HashCache(tmp_path, manifest)

        # Compute vllm's hash
        vllm_hash_v1 = hash_cache.get_hash(vllm_build)

        # Modify local pytorch .def file (at v0.0.5)
        (pytorch_dir / "pytorch.def").write_text("Bootstrap: docker\nFrom: ubuntu:24.04")
        hash_cache.clear()

        # Recompute vllm's hash
        vllm_hash_v2 = hash_cache.get_hash(vllm_build)

        # vllm's hash should NOT change because it depends on v0.0.4, not v0.0.5
        assert vllm_hash_v1 == vllm_hash_v2, (
            "vllm hash should not change when modifying pytorch at a different version"
        )

    def test_hash_changes_when_dependency_version_changes(self, tmp_path: Path) -> None:
        """Hash should change when the dependency version changes in manifest.

        Even though the dependency content might be the same, changing from
        pytorch v0.0.4 to v0.0.5 should change vllm's hash because it's now
        using a different base image.
        """
        # Create container definition files
        vllm_dir = tmp_path / "definitions" / "vllm"
        vllm_dir.mkdir(parents=True)
        (vllm_dir / "vllm.def").write_text("Bootstrap: localimage\nFrom: pytorch.sif")

        # vllm with dependency on pytorch v0.0.4
        vllm_build_v1 = Build(
            name="vllm-0.14",
            version="0.0.5",
            base="pytorch-2.9_0.0.4",
            steps=[Step(path=Path("definitions/vllm/vllm.def"), args={})],
        )

        # Compute hash with old dependency
        hash_v1 = compute_content_hash(
            tmp_path,
            vllm_build_v1,
            dependency_info=[("pytorch-2.9_v0.0.4", None)],
        )

        # Now update to depend on pytorch v0.0.5
        vllm_build_v2 = Build(
            name="vllm-0.14",
            version="0.0.5",
            base="pytorch-2.9_0.0.5",
            steps=[Step(path=Path("definitions/vllm/vllm.def"), args={})],
        )

        # Compute hash with new dependency
        hash_v2 = compute_content_hash(
            tmp_path,
            vllm_build_v2,
            dependency_info=[("pytorch-2.9_0.0.5", None)],
        )

        # Hashes should differ because dependency full names are different
        assert hash_v1 != hash_v2, "Hash should change when dependency version changes in manifest"

    def test_dependency_info_includes_content_hash(self, tmp_path: Path) -> None:
        """Dependency content hash affects the container hash."""
        # Create container definition files
        vllm_dir = tmp_path / "definitions" / "vllm"
        vllm_dir.mkdir(parents=True)
        (vllm_dir / "vllm.def").write_text("Bootstrap: localimage\nFrom: pytorch.sif")

        vllm_build = Build(
            name="vllm-0.14",
            version="0.0.5",
            base="pytorch-2.9_0.0.5",
            steps=[Step(path=Path("definitions/vllm/vllm.def"), args={})],
        )

        # Compute hash with one content hash
        hash_v1 = compute_content_hash(
            tmp_path,
            vllm_build,
            dependency_info=[("pytorch-2.9_0.0.5", "g111111111111")],
        )

        # Compute hash with different content hash
        hash_v2 = compute_content_hash(
            tmp_path,
            vllm_build,
            dependency_info=[("pytorch-2.9_0.0.5", "g222222222222")],
        )

        # Hashes should differ because dependency content hashes differ
        assert hash_v1 != hash_v2, "Hash should change when dependency content hash changes"
