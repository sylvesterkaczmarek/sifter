"""Pluggable remote registry backends for Sifter.

Provides a RemoteRegistry protocol with S3 and filesystem implementations.
When no remote is configured, the registry is None — callers guard with
``if remote is not None:``.
"""

from __future__ import annotations

import shlex
import warnings
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from sifter.storage import ContainerFile

S3_DEPRECATION_MESSAGE = (
    "The S3 registry backend is deprecated and cannot sign or verify images. "
    "Migrate to an OCI/ECR registry (set SIFTER_REGISTRIES) for signed, "
    "verifiable containers."
)

# The sign/verify gates are enforced in the OCI path only, so this backend is
# exempt from them no matter how they are set — say so rather than imply cover.
FILESYSTEM_UNSIGNED_MESSAGE = (
    "The shared-filesystem registry backend cannot sign or verify images: "
    "SIFTER_SIGN and SIFTER_VERIFY have no effect on it, and images are trusted "
    "on the basis of filesystem permissions alone. Use an OCI/ECR registry "
    "(set SIFTER_REGISTRIES) for signed, verifiable containers."
)


@runtime_checkable
class RemoteRegistry(Protocol):
    """Protocol for remote container registries.

    Implementations provide access to a remote store of .sif files.
    If you have an instance, the backend is configured. ``None`` means no remote.
    """

    def exists(self, filename: str) -> bool:
        """Check if a file exists in the remote registry."""
        ...

    def list_files(
        self, prefix: str | None = None, include_dev: bool = False
    ) -> list[ContainerFile]:
        """List .sif files in the remote registry."""
        ...

    def uri(self, filename: str) -> str:
        """Return a human-readable URI for a file."""
        ...

    def generate_pull_command(self, filename: str, local_path: Path) -> str:
        """Generate a shell command to pull a file from remote to local_path."""
        ...

    def generate_push_command(self, local_path: Path, filename: str) -> str:
        """Generate a shell command to push a local file to remote."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of this registry (e.g. 's3://bucket/prefix')."""
        ...


class S3Registry:
    """S3-backed remote registry.

    Lazy-imports boto3 so that it remains optional for non-S3 users.
    """

    def __init__(self, bucket: str, prefix: str) -> None:
        warnings.warn(S3_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as e:
                raise ImportError(
                    "boto3 is required for S3 registry support. "
                    "Install it with: pip install sifter[s3]"
                ) from e
            self._client = boto3.client("s3")
        return self._client

    def _key(self, filename: str) -> str:
        return f"{self.prefix}/{filename}"

    def exists(self, filename: str) -> bool:
        try:
            self._get_client().head_object(Bucket=self.bucket, Key=self._key(filename))
            return True
        except Exception:
            return False

    def list_files(
        self, prefix: str | None = None, include_dev: bool = False
    ) -> list[ContainerFile]:
        search_prefix = self.prefix + "/"
        if prefix:
            search_prefix += prefix

        results: list[ContainerFile] = []
        try:
            paginator = self._get_client().get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=search_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    filename = key.removeprefix(self.prefix + "/")

                    if not filename.endswith(".sif"):
                        continue

                    is_dev = "+dev" in filename
                    if is_dev and not include_dev:
                        continue

                    results.append(
                        ContainerFile(
                            filename=filename,
                            is_dev=is_dev,
                            size_bytes=obj.get("Size"),
                            modified=obj.get("LastModified"),
                        )
                    )
        except Exception as e:
            from sifter.storage import StorageError

            raise StorageError(f"Failed to list S3: {e}") from e

        return sorted(results, key=lambda f: f.filename)

    def uri(self, filename: str) -> str:
        return f"s3://{self.bucket}/{self._key(filename)}"

    def generate_pull_command(self, filename: str, local_path: Path) -> str:
        return f"aws s3 cp {shlex.quote(self.uri(filename))} {shlex.quote(str(local_path))}"

    def generate_push_command(self, local_path: Path, filename: str) -> str:
        return f"aws s3 cp {shlex.quote(str(local_path))} {shlex.quote(self.uri(filename))}"

    @property
    def description(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"


class FilesystemRegistry:
    """Shared-filesystem remote registry (e.g. NFS mount)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def exists(self, filename: str) -> bool:
        return (self.path / filename).exists()

    def list_files(
        self, prefix: str | None = None, include_dev: bool = False
    ) -> list[ContainerFile]:
        if not self.path.exists():
            return []

        results: list[ContainerFile] = []
        for p in self.path.glob("*.sif"):
            if prefix and not p.name.startswith(prefix):
                continue

            is_dev = "+dev" in p.name
            if is_dev and not include_dev:
                continue

            stat = p.stat()
            results.append(
                ContainerFile(
                    filename=p.name,
                    is_dev=is_dev,
                    size_bytes=stat.st_size,
                    modified=None,
                )
            )
        return sorted(results, key=lambda f: f.filename)

    def uri(self, filename: str) -> str:
        return str(self.path / filename)

    def generate_pull_command(self, filename: str, local_path: Path) -> str:
        return f"cp {shlex.quote(str(self.path / filename))} {shlex.quote(str(local_path))}"

    def generate_push_command(self, local_path: Path, filename: str) -> str:
        dest = self.path / filename
        return f"mkdir -p {shlex.quote(str(dest.parent))} && cp {shlex.quote(str(local_path))} {shlex.quote(str(dest))}"

    @property
    def description(self) -> str:
        return str(self.path)
