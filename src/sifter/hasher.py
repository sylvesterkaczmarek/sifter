"""Content-based hashing for build caching in Sifter CLI."""

from __future__ import annotations

import hashlib
from pathlib import Path

from sifter.manifest import Manifest
from sifter.models import Build


def compute_content_hash(
    repo_dir: Path,
    build: Build,
    dependency_info: list[tuple[str, str | None]] | None = None,
) -> str:
    """Compute a content hash for cache invalidation.

    The hash captures:
    - Definition file content (for each step, in order)
    - Build args per step (sorted for determinism)
    - Container version from manifest
    - Dependency full names (always included)
    - Dependency content hashes (only for same-version deps)

    Args:
        repo_dir: Repository root directory
        build: Build definition from the manifest
        dependency_info: List of (full_name, content_hash_or_none) tuples.
            - full_name is ALWAYS included in hash (ensures version changes trigger rebuild)
            - content_hash is included only if not None (same-version deps)

    Returns:
        Hash string with 'g' prefix and 12 hex chars (e.g., "g7354f89abc12")

    Raises:
        FileNotFoundError: If definition file doesn't exist
    """
    hasher = hashlib.sha256()

    # Hash definition files and args in step order
    for step in build.steps:
        def_file = repo_dir / step.path
        hasher.update(def_file.read_bytes())
        for key, value in sorted(step.args.items()):
            hasher.update(f"{key}={value}".encode())

    # Hash build version (version bump = forced rebuild)
    hasher.update(f"version={build.version}".encode())

    # Hash dependency info (sorted by full_name for determinism)
    if dependency_info:
        for full_name, content_hash in sorted(dependency_info):
            # Always include full name (ensures hash changes when dep version changes)
            hasher.update(f"dep_name={full_name}".encode())
            # Include content hash only if available (same-version dependencies)
            if content_hash:
                hasher.update(f"dep_hash={content_hash}".encode())

    # Return 'g' prefix + 12 hex chars
    return "g" + hasher.hexdigest()[:12]


class HashCache:
    """Cache for computed content hashes with version-aware dependency resolution.

    Avoids recomputing hashes for the same container/variant pairs.
    Handles dependency hash cascading for same-version dependencies only.
    """

    def __init__(self, repo_dir: Path, manifest: Manifest | None = None):
        """Initialize hash cache.

        Args:
            repo_dir: Repository root directory
            manifest: Manifest for looking up same-version dependencies.
                     If None, dependency hashes are not computed.
        """
        self.repo_dir = repo_dir
        self.manifest = manifest
        self._cache: dict[str, str] = {}

    def get_hash(self, build: Build) -> str:
        """Get content hash for a build, using cache.

        For same-version dependencies: recursively computes their content hashes
        For different-version dependencies: only includes the full name (immutable)

        Args:
            build: Build definition

        Returns:
            Content hash string
        """
        cache_key = build.tag
        if cache_key not in self._cache:
            # Compute dependency info (full_name, hash_or_none)
            dep_info = self._compute_dependency_info(build)

            self._cache[cache_key] = compute_content_hash(
                self.repo_dir, build, dep_info if dep_info else None
            )
        return self._cache[cache_key]

    def _compute_dependency_info(self, build: Build) -> list[tuple[str, str | None]]:
        """Compute dependency info for hash computation.

        For each dependency:
        - Always include the full name (ensures hash changes when dep version changes)
        - Only compute content hash if dep is at the SAME version (same-version cascading)
        - For different-version deps (old releases), content hash is None (immutable)

        Args:
            build: Build whose dependencies to process

        Returns:
            List of (full_name, content_hash_or_none) tuples
        """
        if not build.base:
            return []

        if self.manifest is None:
            # No manifest provided - include dependency name only
            return [(build.base, None)]

        dep_full_name = build.base
        if "_" not in dep_full_name:
            return [(dep_full_name, None)]

        _dep_name, dep_tag = dep_full_name.rsplit("_", 1)
        if not dep_tag:
            return [(dep_full_name, None)]

        if dep_tag == build.version:
            dep_build = self.manifest.get_build_by_tag(dep_full_name)
            if dep_build:
                dep_hash = self.get_hash(dep_build)
                return [(dep_full_name, dep_hash)]
            return [(dep_full_name, None)]

        return [(dep_full_name, None)]

    def clear(self) -> None:
        """Clear the hash cache."""
        self._cache.clear()
