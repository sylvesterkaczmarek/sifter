"""Tests for sifter.scripts module."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


class TestScripts:
    """Tests for bundled build scripts."""

    def test_build_script_exists(self) -> None:
        """build.sh script exists."""
        script = Path(str(files("sifter.scripts").joinpath("build.sh")))
        assert script.exists()
        assert script.name == "build.sh"

    def test_build_script_is_executable(self) -> None:
        """build.sh is executable."""
        import os

        script = Path(str(files("sifter.scripts").joinpath("build.sh")))
        assert os.access(script, os.X_OK)

    def test_build_script_has_shebang(self) -> None:
        """build.sh starts with bash shebang."""
        script = Path(str(files("sifter.scripts").joinpath("build.sh")))
        content = script.read_text()
        assert content.startswith("#!/bin/bash")

    def test_build_script_has_sbatch_directives(self) -> None:
        """build.sh contains SBATCH directives."""
        script = Path(str(files("sifter.scripts").joinpath("build.sh")))
        content = script.read_text()
        assert "#SBATCH" in content
        assert "--job-name" in content
