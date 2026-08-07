# Contributing to Sifter

This guide covers development setup, testing, and contribution guidelines.

Found a security problem? Don't open an issue or PR — follow
[SECURITY.md](SECURITY.md) instead.

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for package management

The unit tests mock SLURM and the filesystem, so they need none of sifter's
runtime tooling. To exercise the real build/run/registry paths you'll also need
the runtime prerequisites from the [README](README.md#prerequisites): the
`singularity` command (Apptainer or SingularityCE), SLURM, and — for a signed
OCI/ECR registry — `oras`, `cosign` ≥ 3.0.0, and the `aws` CLI (ECR).

### Installation

```bash
git clone https://github.com/AI-Safety-Institute/sifter.git
cd sifter
uv sync --all-extras
```

This installs all dependencies including dev tools (pytest, ruff, ty, prek) and
the optional S3 backend, which the type checker needs to resolve its lazy
`boto3` import.

## Code Quality

### Type Checking with ty

[ty](https://github.com/astral-sh/ty) is a fast type checker for Python.

```bash
# Check all source files
uv run ty check src/

# Check specific file
uv run ty check src/sifter/cli.py
```

### Linting and Formatting with Ruff

[Ruff](https://docs.astral.sh/ruff/) handles both linting and formatting.

```bash
# Check for lint errors
uv run ruff check src/ tests/

# Auto-fix lint errors
uv run ruff check --fix src/ tests/

# Format code
uv run ruff format src/ tests/

# Check formatting without changing files
uv run ruff format --check src/ tests/
```

### Pre-commit Hooks with prek

[prek](https://github.com/astral-sh/prek) runs checks before each commit.

```bash
# Install pre-commit hooks
uv run prek install

# Run all checks manually
uv run prek run

# Run all checks across the entire repo (not just staged changes)
uv run prek run --all-files

# Run specific check
uv run prek run ruff
```

The hooks are configured in `.pre-commit-config.yaml` and run:
- `ruff check` - linting
- `ruff format --check` - formatting
- `ty check` - type checking

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v
```

### Test Structure

```
tests/
├── conftest.py         # Shared fixtures
├── test_cli.py         # CLI command tests
├── test_config.py      # Configuration tests
├── test_dag.py         # Build DAG tests
├── test_hasher.py      # Hash computation tests
├── test_manifest.py    # Manifest parsing tests
├── test_models.py      # Data model tests
├── test_slurm.py       # SLURM client tests
└── test_storage.py     # Storage operations tests
```

### Writing Tests

- Use pytest fixtures from `conftest.py` for common setup
- Mock external dependencies (SLURM, S3, filesystem)
- Test both success and error cases
- Use descriptive test names

Example:

```python
def test_build_computes_correct_hash(tmp_path, mock_slurm):
    """Build command should compute content hash from definition and args."""
    # Arrange
    manifest = create_test_manifest(tmp_path)

    # Act
    result = runner.invoke(app, ["build", "myapp", "--dry-run"])

    # Assert
    assert result.exit_code == 0
    assert "g" in result.output  # Hash prefix
```

## Code Style

- Use Google-style docstrings
- Type hints for all function signatures
- Keep functions focused and small
- Prefer explicit over implicit

## Project Structure

```
src/sifter/
├── __init__.py       # Public exports + version (importlib.metadata)
├── api.py            # Public Python API — all business logic
├── cli.py            # Typer CLI commands (thin wrappers over api.py)
├── config.py         # Env vars, YAML config, sign/verify gates
├── manifest.py       # sifter.yaml parsing (builds: → steps: + base:)
├── models.py         # Build/Step/BuildSpec + API result dataclasses
├── dag.py            # Build dependency graph (BuildDAG, BuildAction)
├── hasher.py         # Content-based hash computation (manifest builds)
├── cache.py          # Ad-hoc build hash computation
├── slurm.py          # SLURM client (sbatch, squeue, sacct, scontrol)
├── storage.py        # Local filesystem operations
├── registry.py       # RemoteRegistry protocol + Filesystem / S3 (legacy)
├── oci.py            # Signed OCI/ECR backends (oras client)
├── signing.py        # cosign sign/verify shell fragments
├── text.py           # What repo-derived text may reach a terminal
└── scripts/          # Shell scripts for SLURM jobs (build/run/transfer)
```

See [CLAUDE.md](CLAUDE.md) for the full architecture guide (module-by-module,
the build-decision flow, and the behaviours that bite).

## Making Changes

1. Create a feature branch from `main`
2. Make your changes
3. Run tests and linting
4. Update `CHANGELOG.md` with your changes under an `[Unreleased]` section
5. Commit with descriptive message
6. Open a pull request

### Changelog

We use [Keep a Changelog](https://keepachangelog.com/) format. Add your changes under the appropriate heading:

- **Added** - new features
- **Changed** - changes to existing functionality
- **Deprecated** - soon-to-be removed features
- **Removed** - removed features
- **Fixed** - bug fixes
- **Security** - vulnerability fixes

## Releasing

When releasing a new version:

1. Update the version in `pyproject.toml`
2. Move `[Unreleased]` changes in `CHANGELOG.md` to a new version section with the date
3. Commit the release preparation and merge it to `main`
4. Tag the merged commit: `git tag X.Y.Z`
5. Push that tag: `git push origin X.Y.Z`
6. Create a GitHub Release named `X.Y.Z` using the matching changelog section
