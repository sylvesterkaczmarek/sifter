"""Shared test fixtures for sifter tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Rich freezes its width when a Console is constructed, and sifter builds its
# consoles at import, so a fixture would be too late to stop assertions on
# wrapped output passing or failing on whatever terminal the suite runs in.
os.environ["COLUMNS"] = "80"

# Path to test resources directory
RESOURCES_DIR = Path(__file__).parent / "resources"
MANIFESTS_DIR = RESOURCES_DIR / "manifests"


def load_manifest(name: str) -> str:
    """Load a manifest YAML file from the resources directory.

    Args:
        name: Manifest filename (with or without .yaml extension)

    Returns:
        Contents of the manifest file
    """
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    return (MANIFESTS_DIR / name).read_text()


@pytest.fixture(autouse=True)
def _isolate_sifter_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Clear all ``SIFTER_*`` vars so tests don't inherit the deployed VM env.

    A real dev VM exports SIFTER_REGISTRIES / SIFTER_SIGNING_KEY / etc.; without
    this a test that assumes "no remote configured" silently sees the ambient
    ECR chain. Tests set exactly the vars they exercise.

    Also points ``XDG_CONFIG_HOME`` at an empty tmp dir so ``resolve_oci_config``
    never reads a real ``~/.config/sifter/config.yaml`` on the dev machine (tests
    that need a user config set their own path, which overrides this).
    """
    for var in (
        "SIFTER_REGISTRIES",
        "SIFTER_REGISTRY",
        "SIFTER_SIGNING_KEY",
        "SIFTER_VERIFY_KEY",
        "SIFTER_SIGN",
        "SIFTER_VERIFY",
        "SIFTER_ORAS_BIN",
        "SIFTER_PROBE_TIMEOUT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))


@pytest.fixture
def sample_manifest_yaml() -> str:
    """Sample sifter.yaml content for testing (builds format)."""
    return load_manifest("valid_with_dependencies")


@pytest.fixture
def sample_repo(tmp_path: Path, sample_manifest_yaml: str) -> Path:
    """Create a sample repository structure for testing.

    Creates:
    - sifter.yaml with sample content
    - definitions/pytorch/ with pytorch.def and build.sh
    - definitions/vllm/ with vllm.def and build.sh

    Returns:
        Path to temporary repository root
    """
    # Write manifest
    manifest_path = tmp_path / "sifter.yaml"
    manifest_path.write_text(sample_manifest_yaml)

    # Create pytorch definition
    pytorch_dir = tmp_path / "definitions" / "pytorch"
    pytorch_dir.mkdir(parents=True)
    (pytorch_dir / "pytorch.def").write_text(
        """Bootstrap: docker
From: nvcr.io/nvidia/cuda:{{ CUDA_VERSION }}.3-cudnn-devel-ubuntu24.04

%arguments
    PYTORCH_VERSION=2.9.1
    CUDA_VERSION=12.6

%post
    pip install torch=={{ PYTORCH_VERSION }}
"""
    )
    (pytorch_dir / "build.sh").write_text(
        """#!/bin/bash
#SBATCH --job-name=build_pytorch
echo "Building pytorch"
"""
    )

    # Create vllm definition
    vllm_dir = tmp_path / "definitions" / "vllm"
    vllm_dir.mkdir(parents=True)
    (vllm_dir / "vllm.def").write_text(
        """Bootstrap: localimage
From: {{ BASE_IMAGE }}

%arguments
    BASE_IMAGE=pytorch.sif
    VLLM_VERSION=0.14.0

%post
    pip install vllm=={{ VLLM_VERSION }}
"""
    )
    (vllm_dir / "build.sh").write_text(
        """#!/bin/bash
#SBATCH --job-name=build_vllm
echo "Building vllm"
"""
    )

    return tmp_path
