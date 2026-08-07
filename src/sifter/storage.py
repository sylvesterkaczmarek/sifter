"""Local filesystem operations for Sifter CLI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sifter.text import reject_path_component


@dataclass
class ContainerFile:
    """Information about a container file.

    Attributes:
        filename: SIF filename (e.g., "vllm-0.14.0_v0.0.5.sif")
        is_dev: Whether this is a dev build
        size_bytes: File size in bytes (if known)
        modified: Last modified time (if known)
    """

    filename: str
    is_dev: bool
    size_bytes: int | None = None
    modified: datetime | None = None


class StorageError(Exception):
    """Error in storage operations."""


class Storage:
    """Local filesystem operations for container files."""

    def __init__(self, local_dir: Path):
        """Initialize storage.

        Args:
            local_dir: Local directory for container files
        """
        self.local_dir = local_dir

    def local_path(self, filename: str) -> Path:
        """Get local path for a filename."""
        try:
            reject_path_component("container filename", filename)
        except ValueError as e:
            raise StorageError(str(e)) from e
        return self.local_dir / filename

    def local_exists(self, filename: str) -> bool:
        """Check if a file exists locally."""
        return self.local_path(filename).exists()

    def list_local(self, prefix: str | None = None) -> list[ContainerFile]:
        """List local .sif files.

        Args:
            prefix: Optional prefix to filter filenames

        Returns:
            List of ContainerFile objects
        """
        if not self.local_dir.exists():
            return []

        results: list[ContainerFile] = []
        for path in self.local_dir.glob("*.sif"):
            if prefix and not path.name.startswith(prefix):
                continue
            stat = path.stat()
            results.append(
                ContainerFile(
                    filename=path.name,
                    is_dev="+dev" in path.name,
                    size_bytes=stat.st_size,
                    modified=None,
                )
            )
        return sorted(results, key=lambda f: f.filename)

    def ensure_local_dir(self) -> None:
        """Create local directory if it doesn't exist."""
        self.local_dir.mkdir(parents=True, exist_ok=True)
