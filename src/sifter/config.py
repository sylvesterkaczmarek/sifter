"""Configuration and environment handling for Sifter CLI.

Uses pydantic-settings for environment variable parsing and validation.
Settings come from ``./sifter.yaml``, ``$XDG_CONFIG_HOME/sifter/config.yaml``
and ``SIFTER_*`` environment variables, with env winning everywhere.

The project file is **untrusted** — it arrives with a cloned repo — and the two
YAML sections treat it differently:

* ``oci:`` — the project may only set the keys named by :data:`PROJECT_OCI_KEYS`,
  and wins on those, one key at a time. It cannot choose the binary sifter
  executes, the signing gates, or the keys signatures are checked against.
* ``registry:`` — the project wins, every key is project-settable, and the merge
  is recursive. These backends have no signing to weaken, but a repo does choose
  the filesystem path or S3 bucket sifter reads and writes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    Discriminator,
    Field,
    PrivateAttr,
    ValidationError,
    ValidationInfo,
    computed_field,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.text import Text

from sifter.oci import (
    DEFAULT_ORAS_BIN,
    DEFAULT_PROBE_TIMEOUT,
    OCIRegistry,
    normalise_registry,
    registry_for,
)
from sifter.registry import (
    FILESYSTEM_UNSIGNED_MESSAGE,
    S3_DEPRECATION_MESSAGE,
    FilesystemRegistry,
    RemoteRegistry,
    S3Registry,
)
from sifter.text import bound, reject_control_characters

# Marker file that identifies the repo root
REPO_MARKER = "sifter.yaml"


def _to_path(value: str | Path | None) -> Path | None:
    """Convert string to Path."""
    if value is None:
        return None
    return Path(value)


PathField = Annotated[Path, BeforeValidator(_to_path)]


class S3RegistryConfig(BaseModel):
    """S3 registry configuration."""

    registry_type: Literal["s3"]
    s3_bucket: str
    s3_prefix: str = "sifter"
    s3_cache_prefix: str | None = None
    s3_registry_prefix: str | None = None

    # Every one of these is project-settable and printed back as the remote's name.
    @field_validator("s3_bucket", "s3_prefix", "s3_cache_prefix", "s3_registry_prefix")
    @classmethod
    def _printable(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        reject_control_characters(str(info.field_name), value)
        return value

    @property
    def effective_cache_prefix(self) -> str:
        if self.s3_cache_prefix:
            return self.s3_cache_prefix
        return f"{self.s3_prefix}/cache"

    @property
    def effective_registry_prefix(self) -> str:
        if self.s3_registry_prefix:
            return self.s3_registry_prefix
        return f"{self.s3_prefix}/registry"


class FilesystemRegistryConfig(BaseModel):
    """Shared filesystem registry configuration."""

    registry_type: Literal["filesystem"]
    registry_path: Path
    registry_cache_path: Path | None = None

    @field_validator("registry_path", "registry_cache_path")
    @classmethod
    def _printable(cls, value: Path | None, info: ValidationInfo) -> Path | None:
        if value is None:
            return None
        reject_control_characters(str(info.field_name), str(value))
        return value

    @property
    def effective_cache_path(self) -> Path:
        if self.registry_cache_path:
            return self.registry_cache_path
        return self.registry_path.parent / "cache"


RegistryConfig = Annotated[
    S3RegistryConfig | FilesystemRegistryConfig,
    Discriminator("registry_type"),
]


def _user_config_path() -> Path:
    """Path to the user's own sifter config — the trusted half of the YAML sources."""
    config_home = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(config_home) / "sifter" / "config.yaml"


def _load_section(path: Path, key: str) -> dict[str, Any]:
    """Read one top-level mapping from a YAML config file, or ``{}`` if absent.

    Raises:
        ConfigError: if the file cannot be read, is not usable YAML, or holds
            something other than settings where settings belong. A project file
            arrives with a cloned repo, so every way it can be malformed — bytes
            that are not UTF-8, nesting deep enough to exhaust the parser's own
            stack — has to land as a configuration error rather than a traceback.
    """
    if not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse {path}: {e}") from e
    except RecursionError as e:
        # The parser descends once per nesting level, and the file chooses how many
        # levels it has, so running out of stack is one of its available outcomes.
        raise ConfigError(f"Failed to parse {path}: nesting is too deep") from e
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"Failed to read {path}: {e}") from e
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must be a mapping of settings, not {type(loaded).__name__}")
    section = loaded.get(key)
    if section is None:
        return {}
    if not isinstance(section, dict):
        # Reading it as "no settings" would be indistinguishable from a file that
        # really has none, so nobody learns why their settings stopped applying.
        raise ConfigError(
            f"{key!r} in {path} must be a mapping of settings, not {type(section).__name__}"
        )
    return section


def resolve_registry_env() -> RegistryConfig | None:
    """Resolve registry config from SIFTER_REGISTRY env var (JSON string).

    Returns None if env var is not set.
    Raises ConfigError if the value is invalid JSON or doesn't match the schema.
    """
    raw = os.environ.get("SIFTER_REGISTRY")
    if not raw:
        return None

    try:
        from pydantic import TypeAdapter

        adapter = TypeAdapter(RegistryConfig)
        return adapter.validate_python(json.loads(raw))
    except (json.JSONDecodeError, Exception) as e:
        raise ConfigError(f"Invalid SIFTER_REGISTRY environment variable: {e}") from e


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base. Override values win for non-dict leaves."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_registry_config(repo_dir: Path) -> RegistryConfig | None:
    """Resolve registry config from YAML files with deep merge precedence.

    Search order (later wins, deep-merged):
      1. ~/.config/sifter/config.yaml  (user/site defaults)
      2. <repo_dir>/sifter.yaml        (project overrides)

    Returns None if no registry section found in any file.
    """
    user_data = _load_section(_user_config_path(), "registry")

    # Project config deep-merges over user — keys override, missing keys preserved
    project_data = _load_section(repo_dir / REPO_MARKER, "registry")
    registry_data = _deep_merge(user_data, project_data)

    if not registry_data:
        return None

    # Parse via pydantic discriminated union
    from pydantic import TypeAdapter

    adapter = TypeAdapter(RegistryConfig | None)
    try:
        return adapter.validate_python(registry_data)
    except ValidationError as e:
        # Every key here is project-settable, so a cloned repo decides whether this
        # validates; a rejected value is something to go and change, not a traceback.
        raise ConfigError(f"invalid registry configuration: {e}") from e


def _parse_registry(value: Any) -> Any:
    """Parse SIFTER_REGISTRY env var from JSON string if needed."""
    if isinstance(value, str):
        return json.loads(value)
    return value


_FALSY = frozenset({"0", "false", "no", "off"})


def _gate(raw: str | None, *, default: bool = True) -> bool:
    """Fail-safe boolean gate: only an explicit falsy value disables.

    Unset, empty, or unrecognised keeps ``default`` (True for the sign/verify
    gates), so a typo can never silently turn a security guard off.
    """
    if raw is None:
        return default
    normalised = raw.strip().lower()
    if normalised in _FALSY:
        return False
    return default


def _resolve_gate(
    env_value: str | None, yaml_value: object, *, key: str, default: bool = True
) -> bool:
    """Combine env (wins) and the trusted user YAML for a fail-safe-ON gate.

    Env is parsed by :func:`_gate`. A YAML boolean is taken verbatim; a YAML
    string goes through :func:`_gate`; a YAML null leaves the gate at its default
    (never silently off).

    Raises:
        ConfigError: if the YAML value is neither a boolean, a string, nor null.
            ``sign: 0`` is written by someone who means off, and YAML makes it an
            integer — better to say so than to do the opposite of what a trusted
            config asked for.
    """
    if env_value is not None:
        return _gate(env_value, default=default)
    if yaml_value is None:
        return default
    if isinstance(yaml_value, bool):
        return yaml_value
    if isinstance(yaml_value, str):
        return _gate(yaml_value, default=default)
    raise ConfigError(f"{key} must be true or false, not {yaml_value!r}")


class OCIConfig(BaseModel):
    """OCI/ECR registry configuration plus cosign signing.

    Distinct from the legacy single-registry :data:`RegistryConfig` (S3 /
    filesystem). The first ``registries`` entry is the push destination and the
    only one consulted; multi-registry pull failover is not yet implemented.
    """

    registries: Annotated[list[str], Field(min_length=1)]
    signing_key: str | None = None
    verify_key: str | None = None
    sign: bool = True
    verify: bool = True
    oras_bin: str = DEFAULT_ORAS_BIN
    # Bounded for every source, trusted included: a huge value lets a cloned repo hang
    # every probe run in it, and a zero one makes them all fail outright.
    probe_timeout: Annotated[int, Field(gt=0, le=600)] = DEFAULT_PROBE_TIMEOUT

    @field_validator("probe_timeout", mode="before")
    @classmethod
    def _reject_boolean_timeout(cls, value: Any) -> Any:
        """Keep ``true`` from arriving as a one-second timeout via bool-to-int coercion."""
        if isinstance(value, bool):
            raise ValueError("probe_timeout is a number of seconds, not a boolean")
        return value

    @field_validator("registries")
    @classmethod
    def _normalise_registries(cls, registries: list[str]) -> list[str]:
        """Store registries in the form the backends use, rejecting unusable names.

        Normalising here rather than at backend construction is what lets a name a
        project file supplied be validated against the string that actually reaches
        argv, and keeps the configuration summary from drifting from the backend it
        describes.
        """
        return [normalise_registry(url) for url in registries]

    def build_chain(self) -> list[OCIRegistry]:
        """Instantiate the registries into OCI/ECR backends, in chain order."""
        return [
            registry_for(
                url,
                signing_key=self.signing_key,
                verify_key=self.verify_key,
                sign=self.sign,
                verify=self.verify,
                oras_bin=self.oras_bin,
                probe_timeout=self.probe_timeout,
            )
            for url in self.registries
        ]


# The only `oci:` settings a cloned repo may decide for itself. An allowlist, so a
# field added to OCIConfig later stays refused until someone weighs it against this.
PROJECT_OCI_KEYS = frozenset({"registries", "probe_timeout"})

# Named first when a refusal is too long to print whole: a project file chooses how
# many keys it asks for, so alphabetical order lets it flood the ones that matter off.
_SECURITY_OCI_KEYS = frozenset({"oras_bin", "signing_key", "verify_key", "sign", "verify"})


def _project_oci_settings(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Split a project ``oci:`` section into what it may set and what it may not.

    Returns the settings kept from the project file, plus the names of every key
    it asked for and did not get, so a caller can report the refusal.
    """
    section = _load_section(path, "oci")
    allowed = {k: v for k, v in section.items() if k in PROJECT_OCI_KEYS}
    # Being allowed to set `registries` is not permission to erase it: an emptied
    # value drops the signed OCI path onto the unsigned `registry:` section below.
    if "registries" in allowed and not allowed["registries"]:
        del allowed["registries"]
    # YAML keys are not all strings (a bare `on:` parses as a boolean), so sort the
    # names rather than the keys themselves.
    return allowed, sorted(str(k) for k in section.keys() - allowed.keys())


_NOTICE_NAMES = 8
_NOTICE_NAME_CHARS = 80


def _names(keys: list[str]) -> str:
    """Name settings in a notice, bounded — a project file writes these strings.

    Unbounded, a file with thousands of refused keys scrolls the notice's own point
    off the screen, which is the suppression the notice exists to prevent.
    """
    shown = [repr(bound(name, _NOTICE_NAME_CHARS)) for name in keys[:_NOTICE_NAMES]]
    if len(keys) > _NOTICE_NAMES:
        shown.append(f"and {len(keys) - _NOTICE_NAMES} more")
    return ", ".join(shown)


class ResolvedOCIConfig(BaseModel):
    """An ``oci:`` resolution: the config, and what it silently did not honour.

    The notices travel with the config rather than through the warnings module,
    where a filter set far from here decides whether a repo's own notice is
    silenced or turned into a traceback in place of the run.
    """

    config: OCIConfig | None = None
    refused_project_keys: tuple[str, ...] = ()
    project_path: Path | None = None

    @property
    def notices(self) -> tuple[str, ...]:
        """Every setting this resolution did not honour, and why, in reading order."""
        return tuple(part for part in (self._refusal, self._unused_registries) if part)

    @property
    def _refusal(self) -> str:
        if not self.refused_project_keys:
            return ""
        # A project-settable key can only be here by having been emptied, so the
        # allowlist tells the two reasons apart without a second field to keep in step.
        emptied = [k for k in self.refused_project_keys if k in PROJECT_OCI_KEYS]
        forbidden = sorted(
            (k for k in self.refused_project_keys if k not in PROJECT_OCI_KEYS),
            key=lambda k: (k not in _SECURITY_OCI_KEYS, k),
        )
        parts = []
        if forbidden:
            parts.append(
                f"ignoring {_names(forbidden)} from {self.project_path}: a cloned repo "
                "cannot make these choices for you — set them in your own sifter config "
                "or the matching SIFTER_* environment variable instead."
            )
        if emptied:
            parts.append(
                f"ignoring an empty {_names(emptied)} in {self.project_path}: emptying it "
                "would drop sifter onto a registry backend that cannot verify a "
                "signature, so it reads as unset."
            )
        return " ".join(parts)

    @property
    def _unused_registries(self) -> str:
        # Only the first registry is ever used, so extra ones are accepted and then
        # silently ignored — worth a line, since a push will not reach them.
        if self.config is None or len(self.config.registries) < 2:
            return ""
        return (
            f"only the first of {len(self.config.registries)} configured registries "
            f"({self.config.registries[0]!r}) is used — multi-registry support is not "
            f"implemented yet, so the rest are ignored: {_names(self.config.registries[1:])}."
        )


def resolve_oci_config(repo_dir: Path) -> ResolvedOCIConfig:
    """Resolve OCI/ECR config from user YAML, then project YAML, then env.

    A project's ``sifter.yaml`` may only set the keys named by
    :data:`PROJECT_OCI_KEYS` — it ships with the repo, so trusting the rest would
    hand its author the binary sifter executes, the signing gates, and the keys
    signatures are checked against. Within that allowlist the project wins, as it
    does for ``registry:``; precedence only ever arbitrates between keys the
    allowlist has already cleared. Reversing it would buy no security property and
    would silently publish every clone to whatever registry a user config defaults to.

    The resolved ``config`` is None when no registries are configured — the legacy
    S3/filesystem path (``SIFTER_REGISTRY``) is untouched and never consulted here.
    The refusal is reported either way: a repo asking to choose the binary sifter
    runs is the security-relevant event, whether or not it also named a registry.
    """
    project_path = repo_dir / REPO_MARKER
    project, refused = _project_oci_settings(project_path)
    user = _load_section(_user_config_path(), "oci")
    # Shallow on purpose, unlike `registry:`: a recursive merge would let a project
    # reach inside a value the allowlist admitted whole.
    data: dict[str, Any] = {**user, **project}

    raw_registries = os.environ.get("SIFTER_REGISTRIES")
    if raw_registries is not None:
        data["registries"] = [r for r in raw_registries.replace(",", " ").split() if r]
    for env_key, field in (
        ("SIFTER_SIGNING_KEY", "signing_key"),
        ("SIFTER_VERIFY_KEY", "verify_key"),
        ("SIFTER_ORAS_BIN", "oras_bin"),
    ):
        if (value := os.environ.get(env_key)) is not None:
            data[field] = value
    if (raw_timeout := os.environ.get("SIFTER_PROBE_TIMEOUT")) is not None:
        try:
            data["probe_timeout"] = int(raw_timeout)
        except ValueError as e:
            raise ConfigError(f"SIFTER_PROBE_TIMEOUT must be an integer: {raw_timeout!r}") from e

    resolved = ResolvedOCIConfig(
        refused_project_keys=tuple(refused),
        project_path=project_path,
    )
    if not data.get("registries"):
        return resolved

    data["sign"] = _resolve_gate(os.environ.get("SIFTER_SIGN"), data.get("sign"), key="sign")
    data["verify"] = _resolve_gate(
        os.environ.get("SIFTER_VERIFY"), data.get("verify"), key="verify"
    )
    try:
        config = OCIConfig.model_validate(data)
    except ValidationError as e:
        # A rejected setting is something the user has to go and change, so it belongs
        # with this module's other configuration errors rather than as a traceback.
        raise ConfigError(f"invalid oci configuration: {e}") from e

    return resolved.model_copy(update={"config": config})


class SifterConfig(BaseSettings):
    """Runtime configuration from environment variables and YAML config files.

    Resolution order for both YAML sections (later wins): user config, project
    config, then ``SIFTER_*``. The difference is what the project is allowed to
    contribute — every ``registry:`` key, but only the ``oci:`` keys named by
    :data:`PROJECT_OCI_KEYS`. See :func:`resolve_oci_config`.

    Attributes:
        dist_dir: Local directory for built .sif files (registry)
        logs_dir: Directory for build logs
        cache_dir: Directory for content-addressed cache
        staging_dir: Directory for staging build definitions
        registry: Remote registry configuration (None = local-only)
        repo_dir: Repository root containing sifter.yaml
        slurm_partition: SLURM partition to submit jobs to
        slurm_cpus: CPUs per task requested for build jobs
        slurm_gpus: Default GPU count for ``sifter run --slurm``
        slurm_mem: Memory request for build jobs ("0" = all node memory)
    """

    model_config = SettingsConfigDict(
        env_prefix="SIFTER_",
        extra="ignore",
    )

    # Core directories - can be set via SIFTER_DIST, SIFTER_LOGS, etc.
    dist_dir: PathField
    logs_dir: PathField
    cache_dir: PathField
    staging_dir: PathField

    # Registry configuration (None = local cache only, no remote registry)
    # BeforeValidator handles SIFTER_REGISTRY env var as JSON string
    registry: Annotated[RegistryConfig | None, BeforeValidator(_parse_registry)] = None

    # Repository root (set programmatically, not from env)
    repo_dir: PathField

    # SLURM resource requests - SIFTER_SLURM_PARTITION, SIFTER_SLURM_CPUS, etc.
    # Defaults are tuned for Isambard-class GH200 nodes (4 GPUs, 72 ARM cores
    # x2 SMT); override them for other clusters.
    slurm_partition: str = "workq"
    slurm_cpus: int = 144
    slurm_gpus: int = 1
    slurm_mem: str = "0"

    # Set programmatically in from_env rather than parsed as a field: a SIFTER_OCI
    # would otherwise inject a whole config, signing gates included.
    _resolved_oci: ResolvedOCIConfig = PrivateAttr(default_factory=ResolvedOCIConfig)

    @property
    def oci(self) -> OCIConfig | None:
        """OCI config; takes precedence over the legacy `registry` when set."""
        return self._resolved_oci.config

    @property
    def config_notices(self) -> tuple[str, ...]:
        """Notices about ``oci:`` settings that were asked for and not honoured."""
        return self._resolved_oci.notices

    @classmethod
    def from_env(cls, repo_dir: Path | None = None) -> SifterConfig:
        """Create configuration from environment variables.

        This is the primary way to construct a SifterConfig. It handles
        the SCRATCHDIR fallback logic before passing to pydantic.

        Args:
            repo_dir: Repository whose ``sifter.yaml`` supplies project
                settings. Defaults to the current working directory.

        Environment Variables:
            SIFTER_DIST: Override local registry dir (default: $SCRATCH/sifter/registry)
            SIFTER_LOGS: Override logs dir (default: $SCRATCH/sifter/logs)
            SIFTER_CACHE: Override cache dir (default: $SCRATCH/sifter/cache)
            SIFTER_STAGING: Override staging dir (default: $SCRATCH/sifter/staging)
            SIFTER_REGISTRY: Registry config as JSON (overrides YAML)
            SCRATCHDIR: HPC scratch directory (or SCRATCH; required if defaults used)

        Raises:
            ConfigError: If required environment variables are missing
        """
        scratch = os.environ.get("SCRATCHDIR") or os.environ.get("SCRATCH")

        # Dist/registry directory
        if dist_str := os.environ.get("SIFTER_DIST"):
            dist_dir = Path(dist_str)
        elif scratch:
            dist_dir = Path(scratch) / "sifter" / "registry"
        else:
            raise ConfigError(
                "Cannot determine dist directory. "
                "Set SIFTER_DIST, SCRATCHDIR, or SCRATCH environment variable."
            )

        # Logs directory
        if logs_str := os.environ.get("SIFTER_LOGS"):
            logs_dir = Path(logs_str)
        elif scratch:
            logs_dir = Path(scratch) / "sifter" / "logs"
        else:
            raise ConfigError(
                "Cannot determine logs directory. "
                "Set SIFTER_LOGS, SCRATCHDIR, or SCRATCH environment variable."
            )

        # Cache directory
        if cache_str := os.environ.get("SIFTER_CACHE"):
            cache_dir = Path(cache_str)
        elif scratch:
            cache_dir = Path(scratch) / "sifter" / "cache"
        else:
            raise ConfigError(
                "Cannot determine cache directory. "
                "Set SIFTER_CACHE, SCRATCHDIR, or SCRATCH environment variable."
            )

        # Staging directory
        if staging_str := os.environ.get("SIFTER_STAGING"):
            staging_dir = Path(staging_str)
        elif scratch:
            staging_dir = Path(scratch) / "sifter" / "staging"
        else:
            raise ConfigError(
                "Cannot determine staging directory. "
                "Set SIFTER_STAGING, SCRATCHDIR, or SCRATCH environment variable."
            )

        # Repository directory - default to cwd. Callers with a custom manifest
        # must provide its parent before either YAML section is resolved; changing
        # only the field afterwards leaves the cwd's private OCI config attached.
        repo_dir = Path.cwd() if repo_dir is None else Path(repo_dir)

        # Resolve registry: env var (JSON) > project yaml > user yaml
        registry = resolve_registry_env() or resolve_registry_config(repo_dir)
        # OCI/ECR pull chain + signing; independent of the legacy
        # S3/filesystem `registry` above.
        resolved_oci = resolve_oci_config(repo_dir)

        config = cls(
            dist_dir=dist_dir,
            logs_dir=logs_dir,
            cache_dir=cache_dir,
            staging_dir=staging_dir,
            registry=registry,
            repo_dir=repo_dir,
        )
        config._resolved_oci = resolved_oci
        return config

    @computed_field
    @property
    def manifest_path(self) -> Path:
        """Path to sifter.yaml manifest file."""
        return self.repo_dir / REPO_MARKER

    def create_remote_cache(self) -> RemoteRegistry | None:
        """Create a RemoteRegistry for the cache, or None if not configured."""
        if self.oci is not None:
            # The content-hash cache (g<hash>.sif) has no OCI ref mapping yet;
            # with an OCI registry the cache is local-only. Callers guard on None.
            return None
        match self.registry:
            case None:
                return None
            case S3RegistryConfig():
                return S3Registry(
                    bucket=self.registry.s3_bucket,
                    prefix=self.registry.effective_cache_prefix,
                )
            case FilesystemRegistryConfig():
                return FilesystemRegistry(path=self.registry.effective_cache_path)

    def create_remote_registry(self) -> RemoteRegistry | None:
        """Create a RemoteRegistry for the registry, or None if not configured.

        When an OCI pull chain is configured it wins over the legacy
        S3/filesystem registry; the first chain entry is the push destination.
        """
        if self.oci is not None:
            return self.oci.build_chain()[0]
        match self.registry:
            case None:
                return None
            case S3RegistryConfig():
                return S3Registry(
                    bucket=self.registry.s3_bucket,
                    prefix=self.registry.effective_registry_prefix,
                )
            case FilesystemRegistryConfig():
                return FilesystemRegistry(path=self.registry.registry_path)

    @property
    def remote_description(self) -> str:
        """Human-readable description of the remote registry.

        Read off the configuration rather than a constructed backend: naming the
        remote should not open clients to it, nor warn about using it.
        """
        if self.oci is not None:
            return f"oras://{self.oci.registries[0]}"
        match self.registry:
            case None:
                return "not configured"
            case S3RegistryConfig():
                prefix = self.registry.effective_registry_prefix.rstrip("/")
                return f"s3://{self.registry.s3_bucket}/{prefix}"
            case FilesystemRegistryConfig():
                return str(self.registry.registry_path)


# Backwards compatibility alias
Config = SifterConfig


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""


def print_config_summary(config: SifterConfig) -> None:
    """Print configuration summary for CLI commands."""
    console = Console()
    # Text, not markup: a remote name comes from a repo, and the whole point of
    # printing it is that it is the name that reaches oras and cosign.
    console.print(Text.assemble(("Local:  ", "dim"), str(config.dist_dir)))
    # Bounded here rather than in the description itself, which callers read as data:
    # a name long enough to wrap for pages scrolls the notice below off the screen.
    console.print(Text.assemble(("Remote: ", "dim"), bound(config.remote_description)))
    # Only the active backend is worth flagging, and an OCI chain supersedes both
    # of these — it is the one path the sign/verify gates actually cover.
    if config.oci is not None:
        return
    match config.registry:
        case S3RegistryConfig():
            console.print(f"[yellow]⚠ {S3_DEPRECATION_MESSAGE}[/yellow]", style="yellow")
        case FilesystemRegistryConfig():
            console.print(f"[yellow]⚠ {FILESYSTEM_UNSIGNED_MESSAGE}[/yellow]", style="yellow")
