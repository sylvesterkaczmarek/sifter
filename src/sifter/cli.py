"""Sifter CLI: declarative Apptainer/Singularity container builds on SLURM/HPC."""

from __future__ import annotations

import re
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Annotated

import humanize
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from sifter import __version__
from sifter.api import (
    build as api_build,
)
from sifter.api import (
    cache_info as api_cache_info,
)
from sifter.api import (
    cache_purge as api_cache_purge,
)
from sifter.api import (
    find_latest_container,
    get_job_log,
    get_jobs,
    list_containers,
    pull_container,
    push_container,
    remove_containers,
    run_container,
    submit_build,
)
from sifter.api import (
    push_release as api_push_release,
)
from sifter.config import Config, ConfigError, print_config_summary
from sifter.dag import BuildAction, BuildDAG, DAGNode
from sifter.manifest import Manifest, ManifestError
from sifter.models import ContainerInfo, is_dev_filename
from sifter.slurm import SLURMClient, SLURMError, SLURMJob
from sifter.storage import Storage, StorageError
from sifter.text import show_control_characters

app = typer.Typer(
    name="sifter",
    help="Declarative Apptainer/Singularity container builds on SLURM/HPC",
    no_args_is_help=True,
)
console = Console(highlight=False)
# Refusals and configuration errors go here, so `singularity exec $(sifter latest)`
# still receives a path and `sifter run` still pipes only the container's output.
err_console = Console(stderr=True, highlight=False)
SIFTER_GIT_URL = "git+https://github.com/AI-Safety-Institute/sifter.git"


def _format_filename_with_tag(filename: str, style: str = "") -> Text:
    """Highlight the tag half of a filename, as text rather than as markup.

    A filename carries a repo's own build key, so it is displayed verbatim.
    """
    if "_" in filename:
        name, tag_part = filename.rsplit("_", 1)
        return Text.assemble(f"{name}_", (tag_part, "bold cyan"), style=style)
    return Text(filename, style=style)


def _get_config(manifest_path: Path | None = None) -> Config:
    """Get configuration, raising nice error on failure.

    Every command that reads configuration goes through here, so that a project
    file's refused settings are reported wherever it is read.

    Args:
        manifest_path: Optional custom manifest path. If provided,
            the repo_dir will be derived from this path's parent directory.
    """
    try:
        config = Config.from_env(
            repo_dir=manifest_path.parent if manifest_path is not None else None
        )
    except ConfigError as e:
        err_console.print(Text.assemble(("Configuration error: ", "red"), str(e)))
        raise typer.Exit(1) from None
    # Hearing nothing about a refused setting is indistinguishable from having been
    # obeyed. Text, not markup: both messages quote the repo's own words back.
    for notice in config.config_notices:
        err_console.print(Text(f"⚠ {notice}", style="yellow"))
    return config


# --- ls Command ---


@app.command("ls")
def ls(
    name: Annotated[
        str | None, typer.Option("--name", "-n", help="Filter by container name")
    ] = None,
    tag: Annotated[str | None, typer.Option("--tag", "-t", help="Filter by tag")] = None,
    remote: Annotated[
        bool, typer.Option("--remote", help="Show remote containers only (no dev)")
    ] = False,
) -> None:
    """List containers in the registry.

    Shows local containers grouped by name.
    Use --remote to show containers in the remote registry instead.

    Examples:
        sifter ls                    # List local containers
        sifter ls --remote           # List remote registry
        sifter ls --name vllm        # Filter by container name
    """
    config = _get_config()
    print_config_summary(config)
    console.print()

    if remote:
        _list_remote(name, tag)
    else:
        _list_local(name, tag)


def _list_local(
    name_filter: str | None,
    tag_filter: str | None,
) -> None:
    """List local containers."""

    items = list_containers(name=name_filter, tag=tag_filter, remote=False)

    if not items:
        console.print("[dim]No containers found[/dim]")
        return

    # Group by name for tree display
    groups: dict[str, list[ContainerInfo]] = {}
    for info in items:
        groups.setdefault(info.name, []).append(info)

    for container_name in sorted(groups):
        tree = Tree(Text(container_name, style="bold"))
        for info in sorted(groups[container_name], key=lambda i: i.filename):
            dev = "dim" if is_dev_filename(info.filename) else ""
            tree.add(_format_filename_with_tag(info.filename, dev))
        console.print(tree)
        console.print()


def _list_remote(
    name_filter: str | None,
    tag_filter: str | None,
) -> None:
    """List remote containers (release only)."""

    try:
        items = list_containers(name=name_filter, tag=tag_filter, remote=True)
    except StorageError as e:
        err_console.print(Text(str(e), style="red"))
        return

    if not items:
        console.print("[dim]No containers found in remote registry[/dim]")
        return

    # Group by name for tree display
    groups: dict[str, list[ContainerInfo]] = {}
    for info in items:
        groups.setdefault(info.name, []).append(info)

    for container_name in sorted(groups):
        tree = Tree(Text(container_name, style="bold"))
        for info in sorted(groups[container_name], key=lambda i: i.filename):
            tree.add(_format_filename_with_tag(info.filename, "green"))
        console.print(tree)
        console.print()


@app.command("latest")
def latest_cmd(
    prefix: Annotated[
        str,
        typer.Option("--prefix", "-p", help="Container name prefix to match (default: any name)"),
    ] = "",
    version: Annotated[
        str | None, typer.Option("--version", "-v", help="Specific upstream version")
    ] = None,
) -> None:
    """Find the latest container matching a prefix.

    By default matches stable releases (X.Y.Z_X.Y.Z) of any container name.
    Use --version to select a specific upstream version (including RC/nightly).

    Outputs a bare path for shell composability:

        singularity exec $(sifter latest) /app/run.sh

    Examples:
        sifter latest                              # Latest stable container
        sifter latest --prefix pytorch-            # Latest stable pytorch
        sifter latest -p pytorch- -v 2.6.0rc1      # Latest RC build
    """

    _get_config()
    try:
        result = find_latest_container(prefix=prefix, version=version)
    except RuntimeError as e:
        err_console.print(Text.assemble(("Error: ", "red"), str(e)))
        raise typer.Exit(1) from None
    print(result)


# --- Build Command ---


@app.command("build")
def build(
    name: Annotated[
        str | None, typer.Argument(help="Build name to build (e.g., 'vllm-0.14.0')")
    ] = None,
    all_: Annotated[bool, typer.Option("--all", help="Build all containers in manifest")] = False,
    base: Annotated[
        str | None,
        typer.Option("--base", help="Override base image (skip dependency resolution)"),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", "-n", help="Show plan without executing")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
    tag: Annotated[
        str | None,
        typer.Option("--tag", "-t", help="Output tag (e.g., 'testing', '0.0.5')"),
    ] = None,
    steps: Annotated[
        str | None,
        typer.Option(
            "--steps",
            help="Comma-separated .def files to build in order (e.g., 'base.def,app.def')",
        ),
    ] = None,
    ad_hoc_name: Annotated[
        str | None,
        typer.Option(
            "--ad-hoc-name", help="Container name for ad-hoc builds (default: folder name)"
        ),
    ] = None,
    manifest: Annotated[
        Path | None,
        typer.Option("--manifest", "-m", help="Path to sifter.yaml manifest file"),
    ] = None,
    reservation: Annotated[
        str | None,
        typer.Option("--reservation", help="SLURM reservation for submitted jobs"),
    ] = None,
) -> None:
    """Build container(s) with automatic dependency resolution.

    Builds containers defined in sifter.yaml, or ad-hoc builds from .def files.
    Uses content hashing to avoid redundant builds - checks if a matching container
    is already building, exists locally, or exists in the remote registry before building.

    All builds go to the cache as <hash>.sif. Requested targets are also tagged
    in the registry as <name>_<version>.sif.

    Ad-hoc builds (using .def files directly) do not support build arguments.
    Use the manifest workflow if you need to pass args to your definition files.

    Examples:
        sifter build vllm-0.14.0                  # Build from manifest
        sifter build vllm-0.14.0 --dry-run        # Preview build plan
        sifter build --all                        # Build all containers in manifest
        sifter build pytorch.def --tag testing    # Ad-hoc build (no args)
        sifter build --steps base.def,app.def --tag myapp  # Multi-step ad-hoc
    """

    config = _get_config(manifest_path=manifest)
    print_config_summary(config)
    console.print()

    try:
        plan = api_build(
            name=name,
            all=all_,
            manifest_path=manifest,
            base=base,
            tag=tag,
            steps=steps,
            ad_hoc_name=ad_hoc_name,
            dry_run=True,
            reservation=reservation,
        )
    except (ValueError, FileNotFoundError, ManifestError) as e:
        err_console.print(Text.assemble(("Error: ", "red"), str(e)))
        raise typer.Exit(1) from None

    if not plan.dag or not plan.dag.nodes:
        console.print("[green]Nothing to build - all containers already exist[/green]")
        return

    # Show build plan
    dag = plan.dag
    console.print(Panel("[bold]Build Plan[/bold]", expand=False))
    _print_dag_tree(dag)
    console.print()

    # Check for registry overwrites
    registry = Storage(local_dir=config.dist_dir)
    overwrites = [
        node.build_spec.registry_tag
        for node in dag.nodes.values()
        if node.build_spec.registry_tag
        and registry.local_exists(f"{node.build_spec.registry_tag}.sif")
    ]
    if overwrites:
        console.print("[yellow]Warning:[/yellow] The following tags will be overwritten:")
        for ow in overwrites:
            console.print(Text(f"  {ow}", style="dim"))
        console.print()

    # Summary (use pre-computed counts from BuildResult)
    if plan.builds_count:
        console.print(f"[yellow]Build:[/yellow] {plan.builds_count} container(s)")
    if plan.pulls_count:
        console.print(f"[blue]Pull:[/blue] {plan.pulls_count} container(s) from remote")
    if plan.tags_count:
        console.print(f"[green]Tag:[/green] {plan.tags_count} container(s) from cache")

    if dry_run:
        console.print("\n[dim]Dry run - no jobs submitted[/dim]")
        return

    if not yes:
        console.print()
        if not typer.confirm("Submit build jobs?"):
            console.print("[dim]Cancelled[/dim]")
            raise typer.Exit(0)

    # Execute the pre-computed plan (no double DAG computation)
    try:
        result = submit_build(plan, reservation=reservation)
    except (SLURMError, FileNotFoundError) as e:
        err_console.print(Text.assemble(("Error: ", "red"), str(e)))
        raise typer.Exit(1) from None

    console.print("\n[bold]Submitting jobs...[/bold]\n")
    for job in result.jobs:
        action_colors = {"build": "yellow", "pull": "blue", "tag": "green"}
        color = action_colors.get(job.action, "white")
        console.print(
            Text.assemble((f"{job.action.title()}: ", color), job.name, f" (job {job.job_id})")
        )

    console.print("\n[green]All jobs submitted![/green]")
    console.print("Use 'sifter status' to check progress")


def _print_dag_tree(dag: BuildDAG) -> None:
    """Print DAG as a tree with build order (roots at top, dependents below)."""
    printed: set[str] = set()

    def add_node_to_tree(node: DAGNode, tree: Tree) -> None:
        if node.full_name in printed:
            return
        printed.add(node.full_name)

        action_colors = {
            BuildAction.BUILD: "yellow",
            BuildAction.TAG_CACHED: "green",
            BuildAction.PULL_REMOTE: "blue",
        }
        color = action_colors.get(node.action, "white")
        action_label = node.action.value.upper()

        tag_info = ""
        if node.build_spec.registry_tag:
            tag_info = f" → {node.build_spec.registry_tag}"
        # Text, not markup: the build name and tag are the repo's own words.
        branch = tree.add(
            Text.assemble(
                (f"[{action_label}]", color),
                f" {node.build_spec.build.name} ({node.output_filename}){tag_info}",
            )
        )

        for dep in node.dependents:
            add_node_to_tree(dep, branch)

    for root in dag.roots():
        tree = Tree(Text(root.build_spec.build.name, style="bold"))
        add_node_to_tree(root, tree)
        console.print(tree)


# --- Status Command ---


@app.command("status")
def status(
    job_id: Annotated[str | None, typer.Option("--id", help="Show specific job by ID")] = None,
    all_: Annotated[bool, typer.Option("--all", help="Show all jobs (not just active)")] = False,
    since: Annotated[
        str | None, typer.Option("--since", help="Time window (e.g., '24h', '7d')")
    ] = None,
) -> None:
    """Show SLURM job status for sifter builds.

    By default shows only active jobs (pending, running). Use --all to include
    completed and failed jobs from the last 7 days, or --since for a custom window.

    Examples:
        sifter status              # Show active jobs
        sifter status --all        # Show all jobs (last 7 days)
        sifter status --since 24h  # Show jobs from last 24 hours
        sifter status --id 12345   # Show details for specific job
    """

    config = _get_config()
    print_config_summary(config)
    console.print()

    since_td: timedelta | None = None
    if since:
        since_td = _parse_duration(since)
    elif all_:
        since_td = timedelta(days=7)

    jobs = get_jobs(job_id=job_id, since=since_td)

    if job_id:
        if not jobs:
            err_console.print(Text(f"Job {job_id} not found", style="red"))
            raise typer.Exit(1)
        _show_single_job(jobs[0], config)
    else:
        _show_all_jobs(jobs, all_, config)


def _show_single_job(job: SLURMJob, config: Config) -> None:
    """Show details for a single job."""
    slurm = SLURMClient(logs_dir=config.logs_dir)

    table = Table(title=f"Job {job.job_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    # A job name is a build key the repository chose, and it reaches the log path too;
    # a table cell goes through the markup parser like anything else printed.
    table.add_row("Name", Text(job.name))
    table.add_row("State", _colorize_state(job.state))
    table.add_row("Node", job.node or "-")
    table.add_row("Elapsed", str(job.elapsed).split(".")[0])

    if job.time_remaining is not None:
        table.add_row("Time Remaining", str(job.time_remaining).split(".")[0])

    if job.dependency:
        table.add_row("Depends on", job.dependency)

    log_path = slurm.log_path(job.name, job.job_id)
    table.add_row("Log", Text(str(log_path)))

    console.print(table)


def _show_all_jobs(jobs: list, all_jobs: bool, config: Config) -> None:
    """Show all sifter jobs."""
    if not jobs:
        console.print("[dim]No sifter jobs found[/dim]")
        return

    slurm = SLURMClient(logs_dir=config.logs_dir)

    active = [j for j in jobs if j.is_active]
    completed = [j for j in jobs if j.state == "COMPLETED"]
    failed = [j for j in jobs if j.is_failed]

    if active:
        console.print("[bold]Active Jobs[/bold]")
        _print_job_table(active, slurm)
        console.print()

    if all_jobs or not active:
        if completed:
            console.print("[bold]Completed Jobs[/bold]")
            _print_job_table(completed, slurm)
            console.print()

        if failed:
            console.print("[bold]Failed Jobs[/bold]")
            _print_job_table(failed, slurm)
            console.print()


def _print_job_table(jobs: list, slurm: SLURMClient) -> None:
    """Print a table of jobs."""
    # Check if any job has time_remaining to decide whether to show column
    has_remaining = any(job.time_remaining is not None for job in jobs)
    # Check if any job has ended to show "Ended" column
    has_ended = any(not job.is_active and job.end_time for job in jobs)

    table = Table()
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("State")
    table.add_column("Elapsed")
    if has_remaining:
        table.add_column("Remaining")
    if has_ended:
        table.add_column("Ended")
    table.add_column("Log")

    for job in jobs:
        log_path = slurm.log_path(job.name, job.job_id)
        row: list[str | Text] = [
            job.job_id,
            Text(job.name),
            _colorize_state(job.state),
            str(job.elapsed).split(".")[0],
        ]
        if has_remaining:
            remaining = str(job.time_remaining).split(".")[0] if job.time_remaining else "-"
            row.append(remaining)
        if has_ended:
            # Show relative time for completed jobs (e.g., "2 hours ago")
            if not job.is_active and job.end_time:
                row.append(humanize.naturaltime(job.end_time))
            else:
                row.append("-")
        row.append(Text(log_path.name))
        table.add_row(*row)

    console.print(table)


def _colorize_state(state: str) -> str:
    """Add color to job state."""
    colors = {
        "PENDING": "cyan",
        "RUNNING": "yellow",
        "COMPLETED": "green",
        "FAILED": "red",
        "CANCELLED": "red",
        "TIMEOUT": "red",
    }
    color = colors.get(state, "white")
    return f"[{color}]{state}[/{color}]"


def _parse_duration(s: str) -> timedelta:
    """Parse duration string like '24h', '7d', '30m'."""
    match = re.match(r"(\d+)([hmd])", s.lower())
    if not match:
        raise typer.BadParameter(f"Invalid duration: {s}")

    value = int(match.group(1))
    unit = match.group(2)

    if unit == "h":
        return timedelta(hours=value)
    elif unit == "d":
        return timedelta(days=value)
    elif unit == "m":
        return timedelta(minutes=value)
    else:
        raise typer.BadParameter(f"Invalid duration unit: {unit}")


# --- Logs Command ---


@app.command("logs")
def logs(
    job_id: Annotated[str, typer.Argument(help="Job ID to view logs for")],
    follow: Annotated[bool, typer.Option("-f", "--follow", help="Follow log output")] = False,
    tail: Annotated[int, typer.Option("-n", "--tail", help="Show last N lines")] = 100,
) -> None:
    """View build logs for a SLURM job.

    Shows the last 100 lines by default. Use -f to follow output in real-time.

    Examples:
        sifter logs 12345          # Show last 100 lines
        sifter logs 12345 -f       # Follow log output
        sifter logs 12345 -n 500   # Show last 500 lines
    """
    config = _get_config()
    if follow:
        # Follow mode needs the log path directly (subprocess tail -f)
        slurm = SLURMClient(logs_dir=config.logs_dir)
        jobs = get_jobs(job_id=job_id)
        if jobs:
            log_path = slurm.log_path(jobs[0].name, job_id)
        else:
            log_files = list(config.logs_dir.glob(f"*_{job_id}.log"))
            if not log_files:
                err_console.print(Text(f"No logs found for job {job_id}", style="red"))
                raise typer.Exit(1)
            log_path = log_files[0]
        if not log_path.exists():
            err_console.print(Text.assemble(("Log file not found: ", "red"), str(log_path)))
            raise typer.Exit(1)
        subprocess.run(["tail", "-f", str(log_path)])
    else:
        try:
            content = get_job_log(job_id, tail=tail)
        except FileNotFoundError as e:
            err_console.print(Text(str(e), style="red"))
            raise typer.Exit(1) from None
        # A build log is whatever the repo's own definition files printed.
        console.print(Text(show_control_characters(content)))


# --- Push Command ---


@app.command("push")
def push(
    filename: Annotated[
        str | None,
        typer.Argument(help="SIF filename to push (e.g. myapp_0.0.1.sif)"),
    ] = None,
    all_: Annotated[
        bool,
        typer.Option(
            "--all",
            "--release",
            help="Push all containers defined in the manifest (--release is a deprecated alias)",
        ),
    ] = False,
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Deprecated: pass the SIF filename positionally instead"),
    ] = None,
    manifest_path: Annotated[
        Path | None,
        typer.Option("--manifest", "-m", help="Path to sifter.yaml (for --all)"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing files in remote registry"),
    ] = False,
) -> None:
    """Push local container(s) to the remote registry.

    Push a single image by filename (positional), or use --all to push every
    container defined in the manifest. Submits as a SLURM job for large transfers.

    The remote registry is immutable by default - use --force to overwrite.

    Examples:
        sifter push myapp_0.0.1.sif       # Push a single container
        sifter push --all                 # Push all manifest containers
        sifter push --all -m path/to/sifter.yaml  # Explicit manifest
        sifter push myapp.sif --force     # Overwrite existing
    """
    target = filename or name
    if filename and name:
        console.print("[red]Error:[/red] Pass the filename positionally or with --name, not both")
        raise typer.Exit(1)
    if target and all_:
        console.print("[red]Error:[/red] Cannot push a single image and --all together")
        raise typer.Exit(1)
    if not target and not all_:
        console.print("[red]Error:[/red] Specify a SIF filename or --all")
        raise typer.Exit(1)
    if name:
        console.print("[yellow]--name is deprecated; pass the filename positionally.[/yellow]")

    config = _get_config(manifest_path=manifest_path)
    print_config_summary(config)
    console.print()

    if all_:
        try:
            # Load manifest to show missing containers diagnostic
            mpath = manifest_path if manifest_path is not None else config.manifest_path
            loaded_manifest = Manifest.load(mpath)
            manifest_filenames = {b.output_filename for b in loaded_manifest.builds.values()}
            local_filenames = {c.filename for c in list_containers()}
            missing = manifest_filenames - local_filenames

            results = api_push_release(manifest_path=manifest_path, force=force)
        except (StorageError, ManifestError) as e:
            err_console.print(Text.assemble(("Error: ", "red"), str(e)))
            raise typer.Exit(1) from None

        if not results:
            console.print("[yellow]No manifest containers found locally[/yellow]")
            console.print(
                Text.assemble(("Expected: ", "dim"), (", ".join(sorted(manifest_filenames)), "dim"))
            )
            return

        if missing:
            console.print(f"[yellow]Missing locally ({len(missing)}):[/yellow]")
            for fn in sorted(missing):
                console.print(Text(f"  - {fn}"))
            console.print()

        for r in results:
            if r.skipped:
                console.print(
                    Text.assemble(("Skipping (already in remote registry): ", "dim"), r.filename)
                )
            else:
                console.print(
                    Text.assemble(
                        ("Push job submitted: ", "green"), r.filename, f" (job {r.job_id})"
                    )
                )

        submitted = sum(1 for r in results if not r.skipped)
        if submitted > 0:
            console.print(f"\n[green]{submitted} push job(s) submitted[/green]")
        elif results:
            console.print("[dim]All containers already in remote registry[/dim]")
    else:
        assert target is not None
        try:
            job_id = push_container(target, force=force)
            console.print(Text.assemble(("Push job submitted: ", "green"), job_id))
        except FileNotFoundError as e:
            err_console.print(Text.assemble(("Error: ", "red"), str(e)))
            raise typer.Exit(1) from None
        except FileExistsError as e:
            err_console.print(Text(str(e), style="yellow"))
            err_console.print("[dim]Use --force to overwrite[/dim]")
            raise typer.Exit(0) from None
        except (StorageError, SLURMError) as e:
            err_console.print(Text.assemble(("Error: ", "red"), str(e)))
            raise typer.Exit(1) from None


# --- Pull Command ---


@app.command("pull")
def pull(
    filename: Annotated[str, typer.Argument(help="SIF filename to pull (e.g., myapp_0.0.1.sif)")],
) -> None:
    """Pull a container from the remote registry.

    Downloads the specified container from the remote registry to local.
    Submits as a SLURM job for large transfers.

    Examples:
        sifter pull vllm-0.14.0_0.0.5.sif  # Pull a release container
        sifter pull myapp.sif              # .sif extension is optional
    """

    config = _get_config()
    print_config_summary(config)
    console.print()

    try:
        job_id = pull_container(filename)
        console.print(Text.assemble(("Pull job submitted: ", "green"), job_id))
    except (StorageError, SLURMError) as e:
        err_console.print(Text.assemble(("Error: ", "red"), str(e)))
        raise typer.Exit(1) from None


# --- Run Command ---


@app.command("run")
def run(
    container: Annotated[str, typer.Argument(help="Container name or path to SIF file")],
    command: Annotated[
        list[str] | None, typer.Argument(help="Command to run inside container")
    ] = None,
    interactive: Annotated[
        bool, typer.Option("-i", "--interactive", help="Start interactive shell")
    ] = False,
    env: Annotated[
        list[str] | None, typer.Option("-e", "--env", help="Environment variable (KEY=value)")
    ] = None,
    slurm: Annotated[
        bool, typer.Option("--slurm", help="Submit as SLURM job instead of running locally")
    ] = False,
    time: Annotated[
        str, typer.Option("--time", help="SLURM time limit (only with --slurm)")
    ] = "01:00:00",
    gpus: Annotated[
        int | None,
        typer.Option("--gpus", help="Number of GPUs (only with --slurm)"),
    ] = None,
) -> None:
    """Run a container interactively or execute a command.

    Runs containers with GPU support (--nv) and a clean environment (--cleanenv).
    Automatically bind-mounts the SCRATCH directory.

    Can run locally (default) or submit as a SLURM job with --slurm.

    Examples:
        sifter run myapp_1.0.0 python script.py        # Run locally
        sifter run -i myapp_1.0.0                      # Interactive shell
        sifter run --slurm myapp_1.0.0 python train.py # Submit as SLURM job
        sifter run -e MY_VAR=value myapp_1.0.0 bash    # With environment variable
    """
    if not command and not interactive:
        console.print("[red]Error:[/red] Specify a command or use -i for interactive shell")
        console.print("[dim]Example: sifter run myapp_1.0.0 python script.py[/dim]")
        console.print("[dim]Example: sifter run -i myapp_1.0.0[/dim]")
        raise typer.Exit(1)

    # Running a container resolves a config without displaying one, so resolve it here
    # too: otherwise a refusal goes unsaid and a bad file arrives as a traceback.
    _get_config()

    try:
        result = run_container(
            container=container,
            command=command,
            interactive=interactive,
            env=env,
            slurm=slurm,
            time=time,
            gpus=gpus,
        )
    except FileNotFoundError:
        err_console.print(Text.assemble(("Error: ", "red"), f"Container not found: {container}"))
        err_console.print("\nUse 'sifter ls' to see available containers")
        raise typer.Exit(1) from None
    except (SLURMError, ValueError) as e:
        err_console.print(Text.assemble(("Error: ", "red"), str(e)))
        raise typer.Exit(1) from None

    if result.job_id:
        console.print(Text.assemble(("Submitted job: ", "green"), result.job_id))
        console.print(Text(f"View status: sifter status --id {result.job_id}", style="dim"))
        console.print(Text(f"View logs: sifter logs {result.job_id}", style="dim"))
    elif result.exit_code is not None:
        raise typer.Exit(result.exit_code)


# --- Cache Command ---


@app.command("cache")
def cache(
    purge: Annotated[bool, typer.Option("--purge", help="Delete all cached containers")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Manage the local container cache.

    The cache stores containers by content hash (e.g., g7354f89abc1.sif) for
    deduplication across builds. Tagged containers in the registry are separate.

    Without options, shows cache statistics. Use --purge to clear the cache.

    Examples:
        sifter cache             # Show cache status
        sifter cache --purge     # Delete all cached containers
    """
    config = _get_config()

    print_config_summary(config)
    console.print()

    if purge:
        info = api_cache_info()
        if info.file_count == 0:
            console.print("[dim]Cache is empty[/dim]")
            return

        console.print("[bold]Cache contents:[/bold]")
        console.print(f"  {info.file_count} container(s)")
        console.print(f"  {_format_size(info.total_bytes)} total")
        console.print()

        if not yes and not typer.confirm("Delete entire cache?"):
            console.print("[dim]Cancelled[/dim]")
            raise typer.Exit(0)

        deleted = api_cache_purge()
        console.print(f"\n[green]Deleted {deleted} cached container(s)[/green]")
    else:
        info = api_cache_info()
        console.print(Text.assemble(("Cache directory: ", "bold"), str(info.path)))
        console.print(f"[bold]Cached containers:[/bold] {info.file_count}")
        console.print(f"[bold]Total size:[/bold] {_format_size(info.total_bytes)}")
        if info.file_count > 0:
            console.print()
            console.print("[dim]Use --purge to delete all cached containers[/dim]")


def _format_size(size_bytes: int) -> str:
    """Format size in human-readable form."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


# --- Remove Command ---


@app.command("rm")
def rm(
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Delete containers matching this prefix"),
    ] = None,
    all_: Annotated[bool, typer.Option("--all", help="Delete all local containers")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Remove local container files from the registry.

    Delete containers by prefix or remove all. Does not affect the cache or remote registry.

    Examples:
        sifter rm --name vllm          # Delete all containers starting with 'vllm'
        sifter rm --all                # Delete all local containers
        sifter rm --name myapp -y      # Skip confirmation
    """
    if not name and not all_:
        console.print("[red]Error:[/red] Specify --name <prefix> or --all")
        raise typer.Exit(1)

    config = _get_config()
    storage = Storage(local_dir=config.dist_dir)
    print_config_summary(config)
    console.print()

    # Preview what will be deleted (prefix match on filename)
    prefix = name if not all_ else None
    containers = storage.list_local(prefix=prefix)
    if not containers:
        console.print("[dim]No containers found to delete[/dim]")
        return

    console.print("[bold]Containers to delete:[/bold]")
    for cf in containers:
        console.print(Text(f"  {cf.filename}"))
    console.print()

    if not yes and not typer.confirm(f"Delete {len(containers)} container(s)?"):
        console.print("[dim]Cancelled[/dim]")
        raise typer.Exit(0)

    deleted = remove_containers(name=prefix)
    for fn in deleted:
        console.print(Text.assemble(("Deleted: ", "red"), fn))
    console.print(f"\n[green]Deleted {len(deleted)} container(s)[/green]")


# --- Version Command ---


@app.command("update")
def update() -> None:
    """Update sifter from the latest git commit using uv tool install."""
    uv = shutil.which("uv")
    if not uv:
        console.print("[red]uv not found in PATH.[/red] Install uv to update sifter.")
        raise typer.Exit(1)

    # Preserve [s3] extra if boto3 is currently available
    import importlib.util

    install_spec = SIFTER_GIT_URL
    if importlib.util.find_spec("boto3") is not None:
        install_spec = f"sifter-build[s3] @ {SIFTER_GIT_URL}"

    console.print(f"Updating sifter via uv: {install_spec}")
    try:
        subprocess.run([uv, "tool", "install", "--force", install_spec], check=True)
    except subprocess.CalledProcessError as e:
        err_console.print(Text.assemble(("Update failed. ", "red"), str(e)))
        raise typer.Exit(e.returncode) from None

    console.print("[green]Update complete.[/green] Re-run your command to use the new version.")


@app.command("version")
def version() -> None:
    """Show sifter version."""
    console.print(f"sifter {__version__}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
