"""SLURM client for job submission and management in Sifter CLI."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib.resources import files
from pathlib import Path
from typing import Literal

# SLURM job states
JobState = Literal[
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "CONFIGURING",
    "COMPLETING",
    "PREEMPTED",
    "SUSPENDED",
    "UNKNOWN",  # Fallback for unrecognized states
]


@dataclass
class SLURMJob:
    """Information about a SLURM job.

    Attributes:
        job_id: SLURM job ID
        name: Job name
        state: Current job state
        submit_time: When job was submitted
        start_time: When job started (None if not started)
        end_time: When job ended (None if not ended)
        deadline: When job will timeout (for running jobs)
        node: Node(s) running the job (compressed list, e.g. "nid[001020-001022]")
        batch_host: Head node where the batch script runs (for multi-node jobs)
        dependency: Job IDs this depends on
    """

    job_id: str
    name: str
    state: JobState
    submit_time: datetime | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    deadline: datetime | None = None
    node: str | None = None
    batch_host: str | None = None
    dependency: str | None = None

    @property
    def elapsed(self) -> timedelta:
        """Time elapsed since start (or since submit if not started)."""
        now = datetime.now(timezone.utc)
        if self.start_time is None:
            if self.submit_time is None:
                return timedelta(0)
            return now - self.submit_time

        # For active jobs, calculate from start to now (not to deadline)
        if self.is_active:
            return now - self.start_time

        # For completed jobs, use actual end_time
        end = self.end_time or now
        return end - self.start_time

    @property
    def time_remaining(self) -> timedelta | None:
        """Time remaining until deadline (for running jobs)."""
        if self.deadline and self.is_active:
            remaining = self.deadline - datetime.now(timezone.utc)
            return remaining if remaining.total_seconds() > 0 else timedelta(0)
        return None

    @property
    def is_active(self) -> bool:
        """Whether the job is still active (not finished)."""
        return self.state in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING")

    @property
    def is_failed(self) -> bool:
        """Whether the job failed."""
        return self.state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED")


class SLURMError(Exception):
    """Error in SLURM operations."""


# Standard job name prefixes for sifter jobs
BUILD_JOB_PREFIX = "sifter-build-"
PULL_JOB_PREFIX = "sifter-pull-"
PUSH_JOB_PREFIX = "sifter-push-"
RUN_JOB_PREFIX = "sifter-run-"

# Default SLURM resource requests. These mirror the #SBATCH directives in
# sifter/scripts/*.sh and are tuned for Isambard-class GH200 nodes
# (4 GPUs, 72 ARM cores x2 SMT). Override via SIFTER_SLURM_* env vars.
DEFAULT_PARTITION = "workq"
DEFAULT_CPUS = 144
DEFAULT_GPUS = 1
DEFAULT_MEM = "0"


class SLURMClient:
    """Client for SLURM job submission and management."""

    def __init__(
        self,
        logs_dir: Path,
        staging_dir: Path | None = None,
        partition: str = DEFAULT_PARTITION,
        cpus: int = DEFAULT_CPUS,
        gpus: int = DEFAULT_GPUS,
        mem: str = DEFAULT_MEM,
    ):
        """Initialize SLURM client.

        Resource requests are passed to ``sbatch`` as command-line flags, which
        override the ``#SBATCH`` directives baked into the job scripts. The
        defaults reproduce those directives exactly.

        Args:
            logs_dir: Directory for job log files
            staging_dir: Directory for staging build definitions (optional)
            partition: SLURM partition for all submitted jobs
            cpus: CPUs per task for build jobs
            gpus: Default GPU count for run jobs
            mem: Memory request for build jobs ("0" = all node memory)
        """
        self.logs_dir = logs_dir
        self.staging_dir = staging_dir
        self.partition = partition
        self.cpus = cpus
        self.gpus = gpus
        self.mem = mem

    def ensure_logs_dir(self) -> None:
        """Create logs directory if it doesn't exist."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def ensure_staging_dir(self) -> None:
        """Create staging directory if it doesn't exist."""
        if self.staging_dir:
            self.staging_dir.mkdir(parents=True, exist_ok=True)

    def log_path(self, job_name: str, job_id: str) -> Path:
        """Get log file path for a job."""
        return self.logs_dir / f"{job_name}_{job_id}.log"

    def stage_build(
        self,
        content_hash: str,
        definition_path: Path,
        args: dict[str, str],
        repo_dir: Path,
    ) -> Path:
        """Stage definition files for a build.

        Creates a staging directory keyed by content hash, copies definition
        files, and writes build_args.env.

        Args:
            content_hash: Content hash for this build step
            definition_path: Path to .def file (relative to repo)
            args: Build arguments to write to env file
            repo_dir: Repository root directory

        Returns:
            Path to staging directory for this build

        Raises:
            SLURMError: If staging fails
        """
        if not self.staging_dir:
            raise SLURMError("staging_dir not configured")

        self.ensure_staging_dir()

        # Create staging dir keyed by hash (prevents race conditions)
        stage_dir = self.staging_dir / content_hash
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Copy entire definition directory (includes %files companions)
        src_def = repo_dir / definition_path
        if not src_def.exists():
            raise SLURMError(f"Definition file not found: {src_def}")

        dest_def = stage_dir / definition_path
        shutil.copytree(src_def.parent, dest_def.parent, dirs_exist_ok=True)

        # Write build_args.env
        args_file = stage_dir / "build_args.env"
        with args_file.open("w") as f:
            for key, value in sorted(args.items()):
                f.write(f"{key}={value}\n")

        return stage_dir

    def submit_staged_build(
        self,
        build_script: Path,
        staging_dir: Path,
        definition_path: Path,
        output_hash: str,
        cache_dir: Path,
        tag: str | None = None,
        registry_dir: Path | None = None,
        base_image_hash: str | None = None,
        remote_push_cache_cmd: str | None = None,
        dependency_job_ids: list[str] | None = None,
        reservation: str | None = None,
    ) -> str:
        """Submit a build job using generic build script and staging.

        Args:
            build_script: Path to generic build.sh
            staging_dir: Staging directory for this build
            definition_path: Relative path to .def in staging
            output_hash: Content hash for output filename
            cache_dir: Local cache directory
            tag: Final tag name (if final step)
            registry_dir: Local registry directory
            base_image_hash: Hash of base image (for chained builds)
            remote_push_cache_cmd: Shell command to push to remote cache
            dependency_job_ids: SLURM job IDs to wait for
            reservation: SLURM reservation name

        Returns:
            SLURM job ID

        Raises:
            SLURMError: If submission fails
        """
        # Build environment variables for the generic build script
        env: dict[str, str] = {
            "STAGING_DIR": str(staging_dir),
            "DEFINITION_PATH": str(definition_path),
            "OUTPUT_HASH": output_hash,
            "CACHE_DIR": str(cache_dir),
        }

        if base_image_hash:
            env["BASE_IMAGE_HASH"] = base_image_hash

        if tag and registry_dir:
            env["TAG_NAME"] = tag
            env["REGISTRY_DIR"] = str(registry_dir)

        if remote_push_cache_cmd:
            env["REMOTE_PUSH_CACHE_CMD"] = remote_push_cache_cmd

        # Job name based on tag or hash
        if tag:
            job_name = f"{BUILD_JOB_PREFIX}{tag.replace(':', '-')}"
        else:
            job_name = f"{BUILD_JOB_PREFIX}{output_hash[:12]}"

        sbatch_args = [
            "--partition",
            self.partition,
            "--cpus-per-task",
            str(self.cpus),
            "--mem",
            self.mem,
        ]
        if reservation:
            sbatch_args += ["--reservation", reservation]

        return self._submit_job(
            script_path=build_script,
            job_name=job_name,
            env=env,
            dependency_job_ids=dependency_job_ids,
            sbatch_args=sbatch_args,
        )

    def submit_transfer_job(
        self,
        command: str,
        filename: str,
        is_pull: bool,
        dependency_job_ids: list[str] | None = None,
        reservation: str | None = None,
    ) -> str:
        """Submit a transfer (pull/push) job.

        Args:
            command: AWS CLI command to execute
            filename: Container filename being transferred
            is_pull: Whether this is a pull (True) or push (False)
            dependency_job_ids: SLURM job IDs to wait for
            reservation: SLURM reservation name

        Returns:
            SLURM job ID

        Raises:
            SLURMError: If submission fails
        """
        prefix = PULL_JOB_PREFIX if is_pull else PUSH_JOB_PREFIX
        # Extract variant name from filename for job naming
        # Version starts with a digit after the last underscore
        if "_" in filename:
            parts = filename.rsplit("_", 1)
            version_part = parts[1].removesuffix(".sif")
            if version_part and version_part[0].isdigit():
                variant_name = parts[0]
            else:
                variant_name = filename.removesuffix(".sif")
        else:
            variant_name = filename.removesuffix(".sif")
        job_name = f"{prefix}{variant_name}"

        self.ensure_logs_dir()

        script_path = Path(str(files("sifter.scripts").joinpath("transfer.sh")))
        env = {
            "TRANSFER_COMMAND": command,
            "TRANSFER_FILENAME": filename,
            "TRANSFER_DIRECTION": "pull" if is_pull else "push",
        }

        # Transfers keep transfer.sh's own (deliberately small) cpu/memory
        # request — only the partition is configurable.
        sbatch_args = ["--partition", self.partition]
        if reservation:
            sbatch_args += ["--reservation", reservation]

        return self._submit_job(
            script_path=script_path,
            job_name=job_name,
            env=env,
            dependency_job_ids=dependency_job_ids,
            sbatch_args=sbatch_args,
        )

    def submit_run_job(
        self,
        container_path: Path,
        singularity_cmd: list[str],
        time: str = "01:00:00",
        gpus: int | None = None,
    ) -> str:
        """Submit a container run job via SLURM.

        Args:
            container_path: Path to the container SIF file
            singularity_cmd: Full singularity command to execute
            time: SLURM time limit (default: 1 hour)
            gpus: Number of GPUs to request (default: the client's configured
                ``gpus``)

        Returns:
            SLURM job ID

        Raises:
            SLURMError: If submission fails
        """
        # Extract container name from path for job naming
        # e.g., /path/to/vllm-0.14.0_0.0.5.sif -> vllm-0.14.0
        container_name = container_path.stem  # Remove .sif
        if "_" in container_name:
            # Remove tag suffix: vllm-0.14.0_0.0.5 -> vllm-0.14.0
            container_name = container_name.rsplit("_", 1)[0]

        job_name = f"{RUN_JOB_PREFIX}{container_name}"

        self.ensure_logs_dir()

        # Build the command string for the script
        cmd_str = shlex.join(singularity_cmd)
        script_path = Path(str(files("sifter.scripts").joinpath("run.sh")))
        env = {
            "RUN_COMMAND": cmd_str,
            "CONTAINER_PATH": str(container_path),
        }

        return self._submit_job(
            script_path=script_path,
            job_name=job_name,
            env=env,
            sbatch_args=[
                "--partition",
                self.partition,
                "--time",
                time,
                "--gpus",
                str(self.gpus if gpus is None else gpus),
            ],
        )

    def _submit_job(
        self,
        script_path: Path,
        job_name: str,
        env: dict[str, str],
        dependency_job_ids: list[str] | None = None,
        sbatch_args: list[str] | None = None,
    ) -> str:
        """Submit a job via sbatch.

        Args:
            script_path: Path to job script
            job_name: SLURM job name
            env: Environment variables
            dependency_job_ids: Job IDs to wait for

        Returns:
            SLURM job ID
        """
        self.ensure_logs_dir()

        args = [
            "sbatch",
            "--parsable",
            "--job-name",
            job_name,
            "--output",
            str(self.logs_dir / f"{job_name}_%j.log"),
            "--error",
            str(self.logs_dir / f"{job_name}_%j.log"),
        ]
        if sbatch_args:
            args.extend(sbatch_args)

        if dependency_job_ids:
            deps = ":".join(dependency_job_ids)
            args.extend(["--dependency", f"afterok:{deps}", "--kill-on-invalid-dep=yes"])

        args.append(str(script_path))

        # Merge environment
        run_env = {**os.environ, **env}

        try:
            result = subprocess.run(
                args,
                env=run_env,
                capture_output=True,
                text=True,
                check=True,
            )
            job_id = result.stdout.strip().split(";")[0]  # Handle cluster output format
            return job_id
        except subprocess.CalledProcessError as e:
            raise SLURMError(f"sbatch failed: {e.stderr}") from e

    def get_job(self, job_id: str) -> SLURMJob | None:
        """Get job information by ID.

        Args:
            job_id: SLURM job ID

        Returns:
            SLURMJob if found, None otherwise
        """
        try:
            result = subprocess.run(
                ["scontrol", "show", "job", job_id, "--json"],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(result.stdout)
            jobs = data.get("jobs", [])
            if not jobs:
                return None
            return self._parse_job(jobs[0])
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return None

    def get_user_jobs(
        self,
        user: str | None = None,
        name_pattern: str | None = None,
        since: timedelta | None = None,
    ) -> list[SLURMJob]:
        """Get jobs for a user.

        Args:
            user: Username (default: current user)
            name_pattern: Filter by job name pattern
            since: Only include jobs from this duration ago (uses sacct for history)

        Returns:
            List of SLURMJob objects
        """
        if user is None:
            user = os.environ.get("USER", "")

        # Use sacct for historical jobs (when since is specified)
        # Use squeue for current/active jobs only
        if since:
            # Get historical jobs from sacct AND active jobs from squeue
            sacct_jobs = self._get_jobs_from_sacct(user, name_pattern, since)
            squeue_jobs = self._get_jobs_from_squeue(user, name_pattern)

            # Merge, avoiding duplicates (prefer sacct data for running jobs)
            seen_ids = {j.job_id for j in sacct_jobs}
            for job in squeue_jobs:
                if job.job_id not in seen_ids:
                    sacct_jobs.append(job)
            return sacct_jobs
        else:
            return self._get_jobs_from_squeue(user, name_pattern)

    def _get_jobs_from_squeue(
        self,
        user: str,
        name_pattern: str | None,
    ) -> list[SLURMJob]:
        """Get active jobs from squeue."""
        args = ["squeue", "--user", user, "--json"]

        try:
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return []

        jobs: list[SLURMJob] = []
        for job_data in data.get("jobs", []):
            job = self._parse_job(job_data)
            if job is None:
                continue

            # Filter by pattern
            if name_pattern and not re.match(name_pattern, job.name):
                continue

            jobs.append(job)

        return jobs

    def _get_jobs_from_sacct(
        self,
        user: str,
        name_pattern: str | None,
        since: timedelta,
    ) -> list[SLURMJob]:
        """Get historical jobs from sacct."""
        now = datetime.now(timezone.utc)
        start_date = (now - since).strftime("%Y-%m-%d")
        args = ["sacct", "--user", user, "--starttime", start_date, "--json"]

        try:
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return []

        jobs: list[SLURMJob] = []
        cutoff = now - since

        for job_data in data.get("jobs", []):
            # Skip job steps (e.g., "12345.batch", "12345.extern")
            # sacct may have a "step" field - check for that to filter out steps
            step_info = job_data.get("step", {})
            if isinstance(step_info, dict) and step_info.get("id"):
                # This is a job step, skip it
                continue

            job = self._parse_sacct_job(job_data)
            if job is None:
                continue

            # Filter by pattern
            if name_pattern and not re.match(name_pattern, job.name):
                continue

            # Filter by time
            if job.submit_time and job.submit_time < cutoff:
                continue

            jobs.append(job)

        return jobs

    def _parse_sacct_job(self, data: dict) -> SLURMJob | None:
        """Parse job data from sacct JSON output (different format from squeue)."""
        try:
            job_id = str(data.get("job_id", ""))
            name = data.get("name", "")

            # sacct uses state.current as a list
            state_data = data.get("state", {})
            state = state_data.get("current", ["UNKNOWN"])
            if isinstance(state, list):
                state = state[0] if state else "UNKNOWN"

            # sacct uses time.submission, time.start, time.end as direct integers
            time_data = data.get("time", {})
            submit_time = self._parse_timestamp(time_data.get("submission"))
            start_time = self._parse_timestamp(time_data.get("start"))
            end_time_val = self._parse_timestamp(time_data.get("end"))

            # For active jobs, end_time is actually the deadline
            # For completed jobs, end_time is the actual end time
            # Note: sacct usually only returns completed jobs, but handle both cases
            if state in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING"):
                deadline = end_time_val
                end_time = None
            else:
                deadline = None
                end_time = end_time_val

            # sacct uses nodes.list for node list
            nodes_data = data.get("nodes", {})
            if isinstance(nodes_data, dict):
                node_list = nodes_data.get("list", [])
                node = node_list[0] if node_list else None
            else:
                node = nodes_data

            dependency = data.get("dependency")

            return SLURMJob(
                job_id=job_id,
                name=name,
                state=state,
                submit_time=submit_time,
                start_time=start_time,
                end_time=end_time,
                deadline=deadline,
                node=node,
                dependency=dependency,
            )
        except Exception:
            return None

    def _parse_job(self, data: dict) -> SLURMJob | None:
        """Parse job data from SLURM JSON output."""
        try:
            job_id = str(data.get("job_id", ""))
            name = data.get("name", "")

            # Handle job_state as string or list (newer SLURM versions use list)
            state = data.get("job_state", "UNKNOWN")
            if isinstance(state, list):
                state = state[0] if state else "UNKNOWN"

            # Parse timestamps (SLURM uses epoch seconds)
            submit_time = self._parse_timestamp(data.get("submit_time", {}).get("number"))
            start_time = self._parse_timestamp(data.get("start_time", {}).get("number"))
            end_time_val = self._parse_timestamp(data.get("end_time", {}).get("number"))

            # For active jobs, end_time is actually the deadline (start + time_limit)
            # For completed jobs, end_time is the actual end time
            if state in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING"):
                deadline = end_time_val
                end_time = None
            else:
                deadline = None
                end_time = end_time_val

            node = data.get("nodes")
            batch_host = data.get("batch_host")
            dependency = data.get("dependency")

            return SLURMJob(
                job_id=job_id,
                name=name,
                state=state,
                submit_time=submit_time,
                start_time=start_time,
                end_time=end_time,
                deadline=deadline,
                node=node,
                batch_host=batch_host,
                dependency=dependency,
            )
        except Exception:
            return None

    def _parse_timestamp(self, epoch: int | None) -> datetime | None:
        """Parse SLURM epoch timestamp as UTC-aware datetime."""
        if epoch is None or epoch == 0:
            return None
        return datetime.fromtimestamp(epoch, tz=timezone.utc)

    def read_log(self, job_name: str, job_id: str, tail: int | None = None) -> str:
        """Read log file for a job.

        Args:
            job_name: Job name
            job_id: SLURM job ID
            tail: Only return last N lines (None for all)

        Returns:
            Log file contents
        """
        log_path = self.log_path(job_name, job_id)
        if not log_path.exists():
            # Try pattern match
            for path in self.logs_dir.glob(f"{job_name}_{job_id}*.log"):
                log_path = path
                break

        if not log_path.exists():
            return f"Log file not found: {log_path}"

        content = log_path.read_text()
        if tail:
            lines = content.splitlines()
            content = "\n".join(lines[-tail:])
        return content
