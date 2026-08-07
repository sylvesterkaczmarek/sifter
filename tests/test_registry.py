"""Tests for sifter.registry module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sifter.registry import FilesystemRegistry, RemoteRegistry, S3Registry


class TestRemoteRegistryProtocol:
    """Tests for the RemoteRegistry protocol."""

    def test_s3_registry_satisfies_protocol(self) -> None:
        reg = S3Registry(bucket="b", prefix="p")
        assert isinstance(reg, RemoteRegistry)

    def test_filesystem_registry_satisfies_protocol(self, tmp_path: Path) -> None:
        reg = FilesystemRegistry(path=tmp_path)
        assert isinstance(reg, RemoteRegistry)


class TestS3Registry:
    """Tests for S3Registry."""

    def test_construction_warns_s3_is_deprecated_and_unsigned(self) -> None:
        # S3 cannot cosign; constructing it must steer users to OCI/ECR.
        with pytest.warns(DeprecationWarning, match="sign"):
            S3Registry(bucket="b", prefix="p")

    def test_exists_true(self) -> None:
        reg = S3Registry(bucket="mybucket", prefix="sifter/cache")
        mock_client = MagicMock()
        reg._client = mock_client

        result = reg.exists("abc123.sif")

        assert result is True
        mock_client.head_object.assert_called_once_with(
            Bucket="mybucket", Key="sifter/cache/abc123.sif"
        )

    def test_exists_false_on_error(self) -> None:
        reg = S3Registry(bucket="mybucket", prefix="sifter/cache")
        mock_client = MagicMock()
        mock_client.head_object.side_effect = Exception("Not found")
        reg._client = mock_client

        assert reg.exists("missing.sif") is False

    def test_uri(self) -> None:
        reg = S3Registry(bucket="mybucket", prefix="containers/sifter")
        assert reg.uri("test.sif") == "s3://mybucket/containers/sifter/test.sif"

    def test_generate_pull_command(self) -> None:
        reg = S3Registry(bucket="mybucket", prefix="sifter/cache")
        cmd = reg.generate_pull_command("test.sif", Path("/local/test.sif"))
        assert cmd == "aws s3 cp s3://mybucket/sifter/cache/test.sif /local/test.sif"

    def test_generate_push_command(self) -> None:
        reg = S3Registry(bucket="mybucket", prefix="sifter/registry")
        cmd = reg.generate_push_command(Path("/local/test.sif"), "test.sif")
        assert cmd == "aws s3 cp /local/test.sif s3://mybucket/sifter/registry/test.sif"

    def test_description(self) -> None:
        reg = S3Registry(bucket="mybucket", prefix="containers/sifter")
        assert reg.description == "s3://mybucket/containers/sifter"

    def test_prefix_trailing_slash_stripped(self) -> None:
        reg = S3Registry(bucket="b", prefix="prefix/")
        assert reg.prefix == "prefix"
        assert reg.uri("f.sif") == "s3://b/prefix/f.sif"

    def test_list_files(self) -> None:
        reg = S3Registry(bucket="mybucket", prefix="sifter/registry")
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "sifter/registry/pytorch_0.0.5.sif", "Size": 1024},
                    {"Key": "sifter/registry/vllm_0.0.5.sif", "Size": 2048},
                    {"Key": "sifter/registry/test_0.0.5+devg123.sif", "Size": 512},
                    {"Key": "sifter/registry/not-a-sif.txt", "Size": 100},
                ]
            }
        ]
        mock_client.get_paginator.return_value = mock_paginator
        reg._client = mock_client

        results = reg.list_files()

        assert len(results) == 2
        assert results[0].filename == "pytorch_0.0.5.sif"
        assert results[1].filename == "vllm_0.0.5.sif"

    def test_list_files_include_dev(self) -> None:
        reg = S3Registry(bucket="mybucket", prefix="sifter/registry")
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "sifter/registry/pytorch_0.0.5.sif", "Size": 1024},
                    {"Key": "sifter/registry/test_0.0.5+devg123.sif", "Size": 512},
                ]
            }
        ]
        mock_client.get_paginator.return_value = mock_paginator
        reg._client = mock_client

        results = reg.list_files(include_dev=True)

        assert len(results) == 2

    def test_lazy_imports_boto3(self) -> None:
        reg = S3Registry(bucket="b", prefix="p")
        assert reg._client is None

        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            client = reg._get_client()
            assert client is not None

    def test_raises_import_error_without_boto3(self) -> None:
        reg = S3Registry(bucket="b", prefix="p")

        with (
            patch.dict("sys.modules", {"boto3": None}),
            pytest.raises(ImportError, match="boto3 is required"),
        ):
            reg._get_client()


class TestFilesystemRegistry:
    """Tests for FilesystemRegistry."""

    def test_exists_true(self, tmp_path: Path) -> None:
        (tmp_path / "test.sif").write_bytes(b"content")
        reg = FilesystemRegistry(path=tmp_path)
        assert reg.exists("test.sif") is True

    def test_exists_false(self, tmp_path: Path) -> None:
        reg = FilesystemRegistry(path=tmp_path)
        assert reg.exists("missing.sif") is False

    def test_list_files(self, tmp_path: Path) -> None:
        (tmp_path / "pytorch_0.0.5.sif").write_bytes(b"a")
        (tmp_path / "vllm_0.0.5.sif").write_bytes(b"bb")
        (tmp_path / "not-a-sif.txt").write_bytes(b"ccc")

        reg = FilesystemRegistry(path=tmp_path)
        results = reg.list_files()

        assert len(results) == 2
        assert results[0].filename == "pytorch_0.0.5.sif"
        assert results[1].filename == "vllm_0.0.5.sif"

    def test_list_files_excludes_dev_by_default(self, tmp_path: Path) -> None:
        (tmp_path / "pytorch_0.0.5.sif").write_bytes(b"a")
        (tmp_path / "test_0.0.5+devg123.sif").write_bytes(b"b")

        reg = FilesystemRegistry(path=tmp_path)
        results = reg.list_files()

        assert len(results) == 1
        assert results[0].filename == "pytorch_0.0.5.sif"

    def test_list_files_include_dev(self, tmp_path: Path) -> None:
        (tmp_path / "pytorch_0.0.5.sif").write_bytes(b"a")
        (tmp_path / "test_0.0.5+devg123.sif").write_bytes(b"b")

        reg = FilesystemRegistry(path=tmp_path)
        results = reg.list_files(include_dev=True)

        assert len(results) == 2

    def test_list_files_with_prefix(self, tmp_path: Path) -> None:
        (tmp_path / "pytorch_0.0.5.sif").write_bytes(b"a")
        (tmp_path / "vllm_0.0.5.sif").write_bytes(b"b")

        reg = FilesystemRegistry(path=tmp_path)
        results = reg.list_files(prefix="pytorch")

        assert len(results) == 1
        assert results[0].filename == "pytorch_0.0.5.sif"

    def test_list_files_empty_dir(self, tmp_path: Path) -> None:
        reg = FilesystemRegistry(path=tmp_path)
        assert reg.list_files() == []

    def test_list_files_nonexistent_dir(self, tmp_path: Path) -> None:
        reg = FilesystemRegistry(path=tmp_path / "nonexistent")
        assert reg.list_files() == []

    def test_uri(self, tmp_path: Path) -> None:
        reg = FilesystemRegistry(path=tmp_path)
        assert reg.uri("test.sif") == str(tmp_path / "test.sif")

    def test_generate_pull_command(self, tmp_path: Path) -> None:
        reg = FilesystemRegistry(path=tmp_path)
        cmd = reg.generate_pull_command("test.sif", Path("/local/test.sif"))
        assert cmd == f"cp {tmp_path / 'test.sif'} /local/test.sif"

    def test_generate_push_command(self, tmp_path: Path) -> None:
        reg = FilesystemRegistry(path=tmp_path)
        cmd = reg.generate_push_command(Path("/local/test.sif"), "test.sif")
        assert cmd == f"mkdir -p {tmp_path} && cp /local/test.sif {tmp_path / 'test.sif'}"

    def test_description(self, tmp_path: Path) -> None:
        reg = FilesystemRegistry(path=tmp_path)
        assert reg.description == str(tmp_path)

    def test_list_files_has_size(self, tmp_path: Path) -> None:
        (tmp_path / "test.sif").write_bytes(b"x" * 1024)
        reg = FilesystemRegistry(path=tmp_path)
        results = reg.list_files()
        assert len(results) == 1
        assert results[0].size_bytes == 1024
