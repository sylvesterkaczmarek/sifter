# Sifter codebase guide for AI agents

## Design philosophy: declarative manifest, executor codebase

Sifter is a **declarative tool**. `sifter.yaml` is the source of truth for what
gets built and how; everything in `src/sifter/` is the executor that turns that
manifest into `singularity`/`sbatch` subprocesses. Four principles hold the
codebase together — keep them intact:

- **`api.py` holds the logic; `cli.py` is a thin skin.** Every operation lives
  as a function in `api.py` that returns a structured dataclass; the Typer
  commands in `cli.py` parse args, call one API function, catch its typed
  exceptions, and render Rich output. Business logic never lives in `cli.py`.
- **Builds are content-addressed.** A build's inputs (definition bytes, sorted
  args, version, dependency hashes) hash to a `g…` id; the image is cached as
  `<hash>.sif`. This is the core mechanism — dedup, reproducibility, and the
  build/skip decision all fall out of it. (This is the deliberate opposite of a
  pure existence check — see [What NOT to do](#what-not-to-do).)
- **The whole plan is a DAG, submitted to SLURM in one shot.** `sifter build`
  never blocks on a running job. It plans a `BuildDAG`, then submits every node
  at once with `--dependency=afterok` chaining and exits; the scheduler runs the
  chain.
- **Publishing is always explicit.** A build writes to the local cache and tags
  the local registry — it never pushes to a remote. Releasing an image is a
  separate `sifter push`.

## Architecture in one diagram

```
                         ┌───────────────────────────┐
   CLI user  ───────────▶│ cli.py — Typer commands   │  (thin display wrappers,
                         │           (Rich output)   │   catch + render errors)
                         └─────────────┬─────────────┘
                                       │
   Python consumer ────────────────────┤   both call the same layer
                                       ▼
                         ┌───────────────────────────┐
                         │ api.py — all business      │  returns dataclasses,
                         │ logic (mirrors every cmd)  │  raises typed exceptions
                         └─┬───────┬───────┬───────┬──┘
                           │       │       │       │
              ┌────────────▼─┐ ┌───▼────┐ ┌▼──────┐ ┌▼──────────────┐
              │ manifest.py  │ │ dag.py │ │hasher │ │ slurm.py       │
              │ (parse YAML) │ │ (plan) │ │cache  │ │ (sbatch/squeue │
              │ models.py    │ │        │ │(hash) │ │  /sacct/scon.) │
              └──────────────┘ └────────┘ └───────┘ └────────────────┘
                           │                             │
              ┌────────────▼─────────────┐   ┌───────────▼───────────┐
              │ config.py                │   │ storage.py (local FS) │
              │ (env + YAML + gates)     │   │ scripts/{build,run,   │
              │                          │   │  transfer}.sh         │
              └────────────┬─────────────┘   └───────────────────────┘
                           │
     ┌─────────────────────▼──────────────────────────────────┐
     │ registry.py  RemoteRegistry protocol + S3 (legacy) /    │
     │              Filesystem backends                        │
     │ oci.py       OCIRegistry / ECRRegistry (oras client)    │
     │ signing.py   cosign sign/verify shell fragments         │
     │ text.py      rules for repo text going to a terminal    │
     └─────────────────────────────────────────────────────────┘
```

`cli.py` is the only entry point users see; `api.py` is the one downstream code
imports. Everything below them is internal plumbing.

## Module-by-module

### `manifest.py` — YAML → validated `Manifest`

Loads `sifter.yaml`. The schema is `builds:` → a mapping keyed by the full tag
`<name>_<version>`, each entry carrying `steps:` (a list of `.def` paths + `args`)
and an optional `base:` (the full tag of another build).

- `Manifest.load(path)` — parse + validate; raises `ManifestError`. Validates
  that every `base:` points at a defined build and that base references form no
  cycle.
- `get_build_by_name(name)` — first build whose **name** matches (what `sifter
  build <name>` resolves against).
- `get_build_by_tag(tag)` — exact `<name>_<version>` lookup (what `base:`
  resolves against).

### `models.py` — core dataclasses + filename grammar

Build inputs: `Build` (name, version, `steps`, optional `base`; `tag` and
`output_filename` are computed), `Step` (a `.def` path + `args`), `BuildSpec`
(one DAG node's frozen inputs — build, definition path, flattened args, output
`<hash>.sif`, base image, content hash, `should_tag`, `registry_tag`).
Build keys are single filename components: neither they nor the local-storage
boundary accept absolute, nested or `..` paths.

API result types: `ContainerInfo`, `CacheInfo`, `TransferResult`, `RunResult`,
`BuildJobResult`, `BuildResult` (the plan + submitted jobs + summary counts).

Filename grammar: `parse_filename()` splits `<name>_<version>.sif` on the **last**
`_` (name may contain `-` and `.`, never `_`); `cache_filename(hash)` →
`<hash>.sif`; `is_dev_filename()` flags a legacy `+`-suffixed build.

### `dag.py` — the build plan

`BuildDAG` is a graph of `DAGNode`s keyed by `full_name` (the content hash). Each
node carries a `BuildSpec` and a `BuildAction`:

- `BuildAction.BUILD` — build from scratch.
- `BuildAction.TAG_CACHED` — the hash already exists in the local cache; just
  copy it to the registry under its tag.
- `BuildAction.PULL_REMOTE` — the hash exists in the *remote content cache*; pull
  it into the local cache (see the OCI caveat below).

`topological_order()` is Kahn's algorithm (dependencies first; raises on a
cycle); `roots()`, `builds_required()`, `pulls_required()` are convenience views.

### `hasher.py` — manifest content hashing

`HashCache.get_hash(build)` computes and memoises a build's content hash:
`SHA256(def bytes + sorted args, per step in order) + version + dependency info)`,
returned as `g` + 12 hex chars. Dependency handling is the subtle part
(`_compute_dependency_info`): a dependency's **full name is always** folded in
(so a dep version bump changes the hash), but its **content hash cascades only
when the dep is pinned at the same version** as the dependent — different-version
(older, frozen) deps contribute name only. So editing a base `.def` cascades new
hashes to same-version dependents, but not to builds pinned at an older base tag.

### `cache.py` — ad-hoc build hashing

`hash_definition(def_path, base_image_hash, args)` — a simpler SHA256 for
manifest-less builds, returning **16 hex chars with no `g` prefix** (distinct
from `hasher.py`'s `g…`). Ad-hoc builds don't take `args`.

### `dag` + `api` build decision (where it all comes together)

`api._add_build_to_dag()` is the planner. For each target (recursively, deps
first) it hashes the build, then picks the action by **existence of that hash**:
local cache present → `TAG_CACHED`; else remote *content* cache present →
`PULL_REMOTE`; else `BUILD`. `api._execute_build_dag()` walks the DAG in
topological order and submits one SLURM job per node, threading `afterok`
dependencies between them. `submit_build()` re-runs execution on an
already-planned `BuildResult` (so the CLI can plan → show → confirm → execute
without recomputing).

### `slurm.py` — SLURM client

`SLURMClient` shells out to `sbatch`/`squeue`/`sacct`/`scontrol`.

- `stage_build()` copies the definition's **whole sibling folder** into a staging
  dir keyed by content hash and writes `build_args.env` (args never go on argv).
- `submit_staged_build()` dispatches `scripts/build.sh` with the staging dir,
  hash, cache/registry dirs, optional base-image hash, tag, and remote-cache push
  command as environment.
- `submit_transfer_job()` / `submit_run_job()` dispatch `transfer.sh` / `run.sh`.
- Job names: `sifter-build-<tag>` (or `-<hash[:12]>` when untagged),
  `sifter-pull-<name>`, `sifter-push-<name>`, `sifter-run-<name>`; logs land at
  `<job_name>_<jobid>.log`. `sifter status` filters jobs on `^sifter-`.
- Chaining: `--dependency=afterok:<ids>` with `--kill-on-invalid-dep=yes`, so a
  failed dependency cancels its dependents rather than building on a missing base.

### `storage.py` — local filesystem only

`Storage` handles the **local** side: `list_local()`, `local_exists()`,
`local_path()`, `ensure_local_dir()`. `ContainerFile` is the raw listing record;
`StorageError` is the shared local/remote storage exception. No remote logic
lives here. `local_path()` refuses anything other than a single filename
component even when its caller has already validated the manifest.

### `registry.py` — remote registry protocol + legacy backends

`RemoteRegistry` is a `runtime_checkable` `Protocol`
(`exists`/`list_files`/`uri`/`generate_pull_command`/`generate_push_command`/
`description`). Two backends live here:

- `FilesystemRegistry` — a shared filesystem (NFS) tree; push/pull are `cp`.
- `S3Registry` — **legacy, deprecated**. Lazy-imports `boto3` (the `s3` extra);
  constructing it emits a `DeprecationWarning`. It **cannot sign or verify** and
  is kept only for existing installs — don't build on it.

Transfer backends return **shell commands** (not in-process transfers) because
the actual copy runs inside a SLURM job.

### `oci.py` — signed OCI/ECR backends (the recommended path)

`OCIRegistry` implements the same protocol over the standalone
[`oras`](https://oras.land) client, storing each SIF as an OCI artifact.
`ECRRegistry(OCIRegistry)` adds AWS ECR's non-interactive login and repo
auto-create. `registry_for(url, …)` dispatches ECR-host → `ECRRegistry`, else
`OCIRegistry`.

- **Filename bridge:** `<name>_<version>.sif` ↔ OCI ref `<registry>/<name>:<version>`
  (`repo_and_tag()` reuses the build-tag grammar so a SIF and its ref never
  disagree).
- **Push signs, pull verifies.** `generate_push_command()` chains `oras push`
  with `cosign sign`; `generate_pull_command()` chains `oras pull` with a
  fail-closed `cosign verify` that `rm`s the SIF on a bad/missing signature.
- **`exists()` fails loud.** A cheap unauthenticated HEAD (trusted only on 200),
  then an authenticated `oras manifest fetch`. A structured absence
  (`NAME_UNKNOWN`/`MANIFEST_UNKNOWN`/"not found") returns `False`; an
  auth/network/timeout failure **raises `RegistryError`** rather than being read
  as "absent" — because a false "absent" would rebuild a multi-hour image that
  actually exists, or clobber a push. This probe runs on the **invoking host**,
  not the compute node (ECR self-authenticates the probe).
- **ECR detection is an exact boundary.** Only canonical private-ECR hostnames
  get the automatic `aws ecr get-login-password | oras login` path. A project
  controls its registry name, so broad `*.amazonaws.com` matching leaks a fresh
  token to attacker-owned AWS endpoints.
- `list_files()` **raises** — an OCI registry is addressed by `name:tag`, so
  there is no whole-registry listing (hence `sifter ls --remote` works only for
  the filesystem/S3 backends).

### `signing.py` — cosign primitives (as shell fragments)

`sign_command(ref, key)` / `verify_command(ref, key)` return cosign shell
fragments embedded in the transfer job (signing runs on the compute node, not
in-process). Each fragment first enforces a **cosign ≥ 3.0.0 floor**: sifter uses
OCI-1.1 *referrer* signatures (the cosign v3 default), so a too-old cosign fails
loudly instead of silently producing signatures the verify step can't see. Both fragments raise `SigningError` if
their key is missing (with the `SIFTER_SIGN=0`/`SIFTER_VERIFY=0` opt-out hint).

### `config.py` — env + YAML config, and the sign/verify gates

`SifterConfig` (pydantic-settings) resolves local dirs and the registry. Local
paths come from `SIFTER_DIST`/`CACHE`/`LOGS`/`STAGING`, defaulting under
`$SCRATCH/sifter/` (or `$SCRATCHDIR`).

Two independent registry configs:
- **OCI/ECR (recommended)** — `OCIConfig`, resolved from the `oci:` YAML section
  and/or env (`SIFTER_REGISTRIES` list, `SIFTER_SIGNING_KEY`,
  `SIFTER_VERIFY_KEY`, `SIFTER_ORAS_BIN`, `SIFTER_PROBE_TIMEOUT`). The **first**
  registry is the push+pull target — multi-registry pull failover is *not yet
  implemented*. When present, it wins over the legacy `registry`.
- **Legacy** — `registry: RegistryConfig | None`, a discriminated union on
  `registry_type` (`S3RegistryConfig` / `FilesystemRegistryConfig`), from the
  `registry:` YAML section or `SIFTER_REGISTRY` (JSON).

Config layering (later wins, deep-merged): `~/.config/sifter/config.yaml` →
`./sifter.yaml` → env vars. The `oci:` section does not layer that way: only the
allowlisted keys of a project `sifter.yaml` reach the merge at all (see **Don't
trust `./sifter.yaml`** below).

The **sign/verify gates** (`SIFTER_SIGN`/`SIFTER_VERIFY`) go through `_gate` /
`_resolve_gate`, which are **fail-safe ON**: only an explicit falsy value
(`0/false/no/off`) disables; unset/empty/unrecognised keeps the default, so a
typo can never silently disable a security guard. `create_remote_cache()` /
`create_remote_registry()` are the factories that turn config into backends.

### `text.py` — the rules for repo-derived text

`reject_control_characters(label, value)` refuses a name a terminal would act on
rather than show, `show_control_characters(value)` makes such bytes visible in
log output, `reject_path_component(label, value)` enforces local path
containment, and `bound(value)` truncates one long enough to scroll the rest of
the output away. Config, models and the OCI backend all validate through here so
the checks cannot drift apart.

### `api.py` — the public surface

Every CLI command has a matching function returning a dataclass:

| Function | CLI | Returns |
|----------|-----|---------|
| `list_containers()` | `sifter ls` | `list[ContainerInfo]` |
| `find_latest_container()` | `sifter latest` | `str` (path) |
| `build()` | `sifter build` | `BuildResult` (plan; `dry_run=True` to plan only) |
| `submit_build()` | — (execute a plan) | `BuildResult` |
| `get_jobs()` | `sifter status` | `list[SLURMJob]` |
| `get_job_log()` | `sifter logs` | `str` |
| `push_container()` | `sifter push --name` | `str` (job id) |
| `push_release()` | `sifter push --release` | `list[TransferResult]` |
| `pull_container()` | `sifter pull` | `str` (job id) |
| `run_container()` | `sifter run` | `RunResult` (exit_code or job_id) |
| `remove_containers()` | `sifter rm` | `list[str]` |
| `cache_info()` / `cache_purge()` | `sifter cache` [`--purge`] | `CacheInfo` / `int` |
| `get_config()` / `registry_path()` / `resolve_container()` | — | config helpers |

CLI and API share the primitives in `dag`/`hasher`/`slurm`/`registry` — when you
change planner or execution semantics, both layers move together.

### `scripts/` — the shell that runs on compute nodes

- `build.sh` — staged into node-local scratch (`$SIFTER_BUILD_TMP`, else
  `$LOCALDIR` if the scheduler sets it, else `/tmp`) to dodge `--fakeroot` issues
  on networked filesystems (e.g. Lustre); substitutes `{{ BASE_IMAGE }}` with the base SIF's
  path, reads `build_args.env`, runs `singularity build --fakeroot`, caches the
  output as `<hash>.sif`, **best-effort** pushes to the remote content cache
  (a failure only warns), and copies to the local registry when tagged. It does
  **not** publish to a remote registry.
- `run.sh` / `transfer.sh` — the run and push/pull job bodies.

## Subtle behaviours that bite

- **`sifter build` takes the build *name*, not the tag.** `build <name>` resolves
  via `get_build_by_name` (the part before the last `_`); passing the full
  `name_version` tag errors with "not found in manifest". `base:` is the
  opposite — it takes the full tag.
- **Under OCI, there is no remote content cache.** `create_remote_cache()`
  returns `None` for an OCI/ECR registry (the `g…` cache has no OCI-ref mapping),
  so `PULL_REMOTE` and the build-time remote-cache push only happen with the
  legacy S3/filesystem backends. With OCI, the cache is local-only and a build
  **never** auto-pulls a dependency from the registry — you `sifter pull` a
  release explicitly.
- **A build never publishes.** `build.sh` writes the cache and (when tagged) the
  local registry, and stops. Releasing is `sifter push` (which signs).
- **The presence probe fails loud, not absent.** An auth/network/timeout error in
  `OCIRegistry.exists()` raises — never treat it as "not there". A false absence
  would rebuild an existing multi-hour image or overwrite a pushed tag.
- **cosign ≥ 3.0.0 is mandatory for OCI signing.** It uses OCI-1.1 referrer
  signatures (the v3 default); the floor is enforced in the generated shell.
- **`_resolved_oci` is a `PrivateAttr`, set in `from_env`, not an env-parsed
  field.** This is deliberate: it stops untrusted env from injecting a whole OCI
  config (and its gates) via a `SIFTER_OCI`-style variable. It carries the
  notices naming what a project file was refused, alongside the config itself.
- **Ad-hoc hashes differ from manifest hashes.** `cache.hash_definition` → 16 hex,
  no `g`; `hasher.HashCache` → `g` + 12 hex. Don't cross-compare them.
- **Staging is keyed by content hash**, so a source edit mid-flight can't race a
  running build.

## Testing

Tests live in `tests/` (`pytest`), with a comprehensive per-module suite
(`test_api`, `test_cli`, `test_dag`, `test_hasher`, `test_manifest`, `test_oci`,
`test_oci_config`, `test_registry`, `test_signing`, `test_slurm`, …).

Fixtures and conventions (`tests/conftest.py`):
- **`_isolate_sifter_env` (autouse)** clears every `SIFTER_*` var, so a test never
  inherits the deployed VM's ambient ECR chain / signing keys. Set exactly the
  vars a test exercises.
- **`sample_repo`** writes a `sifter.yaml` plus `pytorch`/`vllm` definition dirs
  into `tmp_path`; **`sample_manifest_yaml`** / the `load_manifest()` helper pull
  fixtures from `tests/resources/manifests/`.
- **Mock at `sifter.api.*`, not `sifter.cli.*`.** Build/push/pull tests patch
  `sifter.api.SLURMClient` (also `sifter.api.Storage` / `get_config` /
  `SifterConfig`), because that's where the logic imports them — patching the CLI
  module misses the code path.

Run: `uv run pytest -q` (fast; no HPC needed — SLURM and the filesystem are
mocked).

## Lint, format, type-check

`ruff` lints + formats and `ty` (Astral's type checker — **not mypy**) type-checks:

```bash
uv run ruff format src tests          # format
uv run ruff check --fix src tests     # lint (autofix)
uv run ty check src                    # type-check
```

`prek` (a fast `pre-commit` reimplementation, shipped as a dev dep) mirrors CI's
lint + type-check locally from `.pre-commit-config.yaml` (hygiene hooks,
`uv-lock` staleness, `ruff-check --fix`, `ruff-format`, `ty`). It does **not** run
the tests, so:

```bash
uv run prek install                    # git hook
uv run prek run --all-files            # after big changes
uv run pytest -q                       # then the tests
```

A clean `prek run --all-files` plus a green `pytest` is the pre-push bar.

## When making changes

- **Adding a CLI command** — add the API function (with a structured return type)
  in `api.py`, a thin Typer wrapper in `cli.py`, export from `__init__.py`, and
  test against both surfaces (`test_api.py`, `test_cli.py`).
- **Changing build logic** — planning is `api._add_build_to_dag()`, execution is
  `api._execute_build_dag()`, hashing is `hasher.py`, submission is `slurm.py`.
- **Changing the manifest schema** — update `manifest.py`, `models.py`,
  `examples/sifter.yaml`, and `tests/test_manifest.py` together.
- **Adding a registry backend** — implement the `RemoteRegistry` protocol; wire it
  into `config.create_remote_registry()` / `create_remote_cache()`. OCI-family
  backends belong in `oci.py`.
- **Any change that alters documented behaviour** — update the docs in the same
  PR. README.md, this file, and CONTRIBUTING.md describe commands, flags,
  defaults, and layout in detail; detailed docs rot fast if changes don't carry
  them along.

## What NOT to do

- **Don't make `sifter build` publish.** A build tags locally and stops;
  publishing is an explicit `sifter push`. Don't reintroduce auto-push (it once
  falsely surprised users with released images).
- **Don't read a probe/transfer auth or network failure as "absent".**
  `OCIRegistry.exists()` must raise on an inconclusive result — a false absence
  rebuilds or clobbers. Keep it fail-loud.
- **Don't weaken signing.** Keep `SIFTER_SIGN` and `SIFTER_VERIFY` independent and
  fail-safe-ON; keep the cosign ≥ 3.0.0 floor. The gates cover the OCI path only —
  the filesystem and S3 backends ignore them, and must keep saying so
  (`FILESYSTEM_UNSIGNED_MESSAGE` in the config summary, `S3_DEPRECATION_MESSAGE`
  on construction) until they either enforce the gates or go away.
- **Don't trust `./sifter.yaml`.** It arrives with a cloned repo. `PROJECT_OCI_KEYS`
  is an allowlist of the `oci:` keys it may set; any other key it sets there is
  dropped, and the refusal is rendered by the CLI rather than sent through
  `warnings` — a security notice must survive both `PYTHONWARNINGS=ignore` and
  `=error`. Don't add a key naming a binary or a signing decision to the allowlist
  — `oras_bin` is executed, so honouring it was RCE, and `sign`/`verify`/
  `verify_key` are the gates themselves. Within the allowlist the project wins over
  the user config, deliberately: precedence only arbitrates keys the allowlist has
  already cleared, so don't reach for it as a security control. Being on the
  allowlist is not permission to *erase* a key: an emptied `registries` is treated
  as unset, since deleting the OCI config falls through to the unsigned
  `registry:` section. A custom `--manifest` parent must be passed into
  `from_env()` before resolution; changing `repo_dir` on an already-resolved
  model carries the wrong repository's private OCI configuration with it.
- **Don't broaden ECR hostname detection.** Only a canonical private-ECR host
  may receive an automatically minted ECR login token; other AWS service
  endpoints can be attacker-owned.
- **Keep the content-addressed model.** Content hashing (`hasher.py`, `cache.py`)
  and the `BuildDAG` (`dag.py`) are load-bearing — don't remove or route around
  them. The manifest schema is exactly `builds:` → `steps:` (+ optional `base:`);
  keep it that way.
- **Don't build on S3.** It's deprecated and can't sign/verify; new work targets
  OCI/ECR. Don't add S3-specific features.
- **Don't parse `_resolved_oci` straight from an env field.** It's a
  `PrivateAttr` set in `from_env` on purpose — keep that boundary.
