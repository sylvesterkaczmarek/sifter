"""Sifter: declarative Apptainer/Singularity container builds on SLURM/HPC."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Distribution name (see pyproject); the import package stays `sifter`.
    __version__ = version("sifter-build")
except PackageNotFoundError:  # pragma: no cover — not installed (e.g. source tree)
    __version__ = "0.0.0+unknown"

from sifter.api import (
    build,
    cache_info,
    cache_purge,
    find_latest_container,
    get_config,
    get_job_log,
    get_jobs,
    list_containers,
    pull_container,
    push_container,
    push_release,
    registry_path,
    remove_containers,
    resolve_container,
    run_container,
    submit_build,
)
from sifter.config import FilesystemRegistryConfig, S3RegistryConfig, SifterConfig
from sifter.manifest import ManifestError
from sifter.models import (
    BuildJobResult,
    BuildResult,
    CacheInfo,
    ContainerInfo,
    RunResult,
    TransferResult,
)
from sifter.registry import FilesystemRegistry, RemoteRegistry, S3Registry
from sifter.slurm import SLURMError, SLURMJob
from sifter.storage import ContainerFile, Storage, StorageError

__all__ = [
    "BuildJobResult",
    "BuildResult",
    "CacheInfo",
    "ContainerFile",
    "ContainerInfo",
    "FilesystemRegistry",
    "FilesystemRegistryConfig",
    "ManifestError",
    "RemoteRegistry",
    "RunResult",
    "S3Registry",
    "S3RegistryConfig",
    "SLURMError",
    "SLURMJob",
    "SifterConfig",
    "Storage",
    "StorageError",
    "TransferResult",
    "build",
    "cache_info",
    "cache_purge",
    "find_latest_container",
    "get_config",
    "get_job_log",
    "get_jobs",
    "list_containers",
    "pull_container",
    "push_container",
    "push_release",
    "registry_path",
    "remove_containers",
    "resolve_container",
    "run_container",
    "submit_build",
]
