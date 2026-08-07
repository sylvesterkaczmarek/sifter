"""Content hashing utilities for Sifter builds."""

from __future__ import annotations

import hashlib
from pathlib import Path


def hash_definition(
    def_path: Path,
    base_image_hash: str | None = None,
    args: dict[str, str] | None = None,
) -> str:
    """Compute content hash for a definition file and its build context.

    The hash captures:
    - Definition file content
    - Base image hash (if building on another container)
    - Build arguments (sorted for determinism)

    Args:
        def_path: Path to the .def file.
        base_image_hash: Hash of the base image (for chained builds).
        args: Build arguments to pass to the definition.

    Returns:
        First 16 characters of SHA256 hex digest.
    """
    hasher = hashlib.sha256()

    # Hash definition file content
    hasher.update(def_path.read_bytes())

    # Hash base image if provided
    if base_image_hash:
        hasher.update(f"base={base_image_hash}".encode())

    # Hash args (sorted for determinism)
    if args:
        for key, value in sorted(args.items()):
            hasher.update(f"{key}={value}".encode())

    return hasher.hexdigest()[:16]
