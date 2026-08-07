"""Tests for sifter.slurm module."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sifter.slurm import (
    RUN_JOB_PREFIX,
    JobState,
    SLURMClient,
    SLURMError,
    SLURMJob,
)


class TestSLURMJob:
    """Tests for SLURMJob dataclass."""

    def test_is_active_pending(self) -> None:
        """PENDING job is active."""
        job = SLURMJob(job_id="123", name="test", state="PENDING")
        assert job.is_active is True

    def test_is_active_running(self) -> None:
        """RUNNING job is active."""
        job = SLURMJob(job_id="123", name="test", state="RUNNING")
        assert job.is_active is True

    def test_is_active_completed(self) -> None:
        """COMPLETED job is not active."""
        job = SLURMJob(job_id="123", name="test", state="COMPLETED")
        assert job.is_active is False

    def test_is_active_failed(self) -> None:
        """FAILED job is not active."""
        job = SLURMJob(job_id="123", name="test", state="FAILED")
        assert job.is_active is False

    def test_is_failed(self) -> None:
        """is_failed returns True for failure states."""
        failed_states: list[JobState] = ["FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]
        for state in failed_states:
            job = SLURMJob(job_id="123", name="test", state=state)
            assert job.is_failed is True

        job = SLURMJob(job_id="123", name="test", state="COMPLETED")
        assert job.is_failed is False

    def test_elapsed_not_started(self) -> None:
        """elapsed returns 0 for job not started."""
        job = SLURMJob(job_id="123", name="test", state="PENDING")
        assert job.elapsed == timedelta(0)

    def test_elapsed_running(self) -> None:
        """elapsed calculates from start_time."""
        start = datetime.now(timezone.utc) - timedelta(minutes=10)
        job = SLURMJob(job_id="123", name="test", state="RUNNING", start_time=start)
        elapsed = job.elapsed
        # Allow some tolerance
        assert timedelta(minutes=9) < elapsed < timedelta(minutes=11)

    def test_elapsed_completed(self) -> None:
        """elapsed calculates from start to end."""
        start = datetime.now(timezone.utc) - timedelta(minutes=20)
        end = datetime.now(timezone.utc) - timedelta(minutes=10)
        job = SLURMJob(
            job_id="123",
            name="test",
            state="COMPLETED",
            start_time=start,
            end_time=end,
        )
        elapsed = job.elapsed
        # Should be about 10 minutes
        assert timedelta(minutes=9) < elapsed < timedelta(minutes=11)


class TestSLURMClient:
    """Tests for SLURMClient class."""

    def test_ensure_logs_dir(self, tmp_path: Path) -> None:
        """ensure_logs_dir creates directory."""
        logs_dir = tmp_path / "logs"
        client = SLURMClient(logs_dir=logs_dir)

        client.ensure_logs_dir()

        assert logs_dir.exists()

    def test_log_path(self, tmp_path: Path) -> None:
        """log_path returns correct path."""
        client = SLURMClient(logs_dir=tmp_path)

        path = client.log_path("test-job", "12345")

        assert path == tmp_path / "test-job_12345.log"

    def test_submit_transfer_job(self, tmp_path: Path) -> None:
        """submit_transfer_job submits the shared transfer script."""
        client = SLURMClient(logs_dir=tmp_path)

        mock_result = MagicMock()
        mock_result.stdout = "12345\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            job_id = client.submit_transfer_job(
                command="aws s3 cp s3://bucket/file.sif /local/file.sif",
                filename="file.sif",
                is_pull=True,
            )

            assert job_id == "12345"
            call_args = mock_run.call_args[0][0]
            assert call_args[-1].endswith("sifter/scripts/transfer.sh")

    def test_submit_transfer_job_with_reservation(self, tmp_path: Path) -> None:
        """submit_transfer_job passes --reservation to sbatch."""
        client = SLURMClient(logs_dir=tmp_path)

        mock_result = MagicMock()
        mock_result.stdout = "12345\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            job_id = client.submit_transfer_job(
                command="aws s3 cp s3://bucket/file.sif /local/file.sif",
                filename="file.sif",
                is_pull=True,
                reservation="my-reservation",
            )

            assert job_id == "12345"
            call_args = mock_run.call_args[0][0]
            assert "--reservation" in call_args
            res_idx = call_args.index("--reservation")
            assert call_args[res_idx + 1] == "my-reservation"

    def test_get_job_success(self, tmp_path: Path) -> None:
        """get_job returns SLURMJob on success."""
        client = SLURMClient(logs_dir=tmp_path)

        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                "jobs": [
                    {
                        "job_id": 12345,
                        "name": "test-job",
                        "job_state": "RUNNING",
                        "submit_time": {"number": 1700000000},
                        "start_time": {"number": 1700000100},
                        "end_time": {"number": 0},
                        "nodes": "node01",
                    }
                ]
            }
        )

        with patch("subprocess.run", return_value=mock_result):
            job = client.get_job("12345")

        assert job is not None
        assert job.job_id == "12345"
        assert job.name == "test-job"
        assert job.state == "RUNNING"

    def test_get_job_not_found(self, tmp_path: Path) -> None:
        """get_job returns None when job not found."""
        client = SLURMClient(logs_dir=tmp_path)

        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"jobs": []})

        with patch("subprocess.run", return_value=mock_result):
            job = client.get_job("99999")

        assert job is None

    def test_get_user_jobs(self, tmp_path: Path) -> None:
        """get_user_jobs returns list of jobs."""
        client = SLURMClient(logs_dir=tmp_path)

        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                "jobs": [
                    {
                        "job_id": 123,
                        "name": "sifter-build-test",
                        "job_state": "RUNNING",
                        "submit_time": {"number": 1700000000},
                        "start_time": {"number": 0},
                        "end_time": {"number": 0},
                    },
                    {
                        "job_id": 124,
                        "name": "other-job",
                        "job_state": "PENDING",
                        "submit_time": {"number": 1700000000},
                        "start_time": {"number": 0},
                        "end_time": {"number": 0},
                    },
                ]
            }
        )

        with patch("subprocess.run", return_value=mock_result):
            jobs = client.get_user_jobs()

        assert len(jobs) == 2

    def test_get_user_jobs_with_pattern(self, tmp_path: Path) -> None:
        """get_user_jobs filters by name pattern."""
        client = SLURMClient(logs_dir=tmp_path)

        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                "jobs": [
                    {
                        "job_id": 123,
                        "name": "sifter-build-test",
                        "job_state": "RUNNING",
                        "submit_time": {"number": 1700000000},
                        "start_time": {"number": 0},
                        "end_time": {"number": 0},
                    },
                    {
                        "job_id": 124,
                        "name": "other-job",
                        "job_state": "PENDING",
                        "submit_time": {"number": 1700000000},
                        "start_time": {"number": 0},
                        "end_time": {"number": 0},
                    },
                ]
            }
        )

        with patch("subprocess.run", return_value=mock_result):
            jobs = client.get_user_jobs(name_pattern=r"^sifter-")

        assert len(jobs) == 1
        assert jobs[0].name == "sifter-build-test"

    def test_read_log(self, tmp_path: Path) -> None:
        """read_log returns log contents."""
        client = SLURMClient(logs_dir=tmp_path)

        # Create log file
        log_file = tmp_path / "test-job_12345.log"
        log_file.write_text("line1\nline2\nline3\n")

        content = client.read_log("test-job", "12345")

        assert "line1" in content
        assert "line3" in content

    def test_read_log_tail(self, tmp_path: Path) -> None:
        """read_log with tail returns last N lines."""
        client = SLURMClient(logs_dir=tmp_path)

        log_file = tmp_path / "test-job_12345.log"
        log_file.write_text("line1\nline2\nline3\nline4\nline5\n")

        content = client.read_log("test-job", "12345", tail=2)

        assert "line4" in content
        assert "line5" in content
        assert "line1" not in content

    def test_read_log_not_found(self, tmp_path: Path) -> None:
        """read_log returns message for missing file."""
        client = SLURMClient(logs_dir=tmp_path)

        content = client.read_log("nonexistent", "99999")

        assert "not found" in content.lower()

    def test_get_job_handles_list_state(self, tmp_path: Path) -> None:
        """get_job handles job_state as list (newer SLURM versions)."""
        client = SLURMClient(logs_dir=tmp_path)

        # Newer SLURM versions return job_state as a list
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                "jobs": [
                    {
                        "job_id": 12345,
                        "name": "sifter-build-vllm",
                        "job_state": ["RUNNING"],  # List format!
                        "submit_time": {"number": 1700000000},
                        "start_time": {"number": 1700000100},
                        "end_time": {"number": 0},
                        "nodes": "node01",
                    }
                ]
            }
        )

        with patch("subprocess.run", return_value=mock_result):
            job = client.get_job("12345")

        assert job is not None
        assert job.state == "RUNNING"  # Should extract from list
        assert job.is_active is True

    def test_get_user_jobs_handles_list_state(self, tmp_path: Path) -> None:
        """get_user_jobs handles job_state as list."""
        client = SLURMClient(logs_dir=tmp_path)

        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                "jobs": [
                    {
                        "job_id": 123,
                        "name": "sifter-build-test",
                        "job_state": ["PENDING"],  # List format
                        "submit_time": {"number": 1700000000},
                        "start_time": {"number": 0},
                        "end_time": {"number": 0},
                    },
                ]
            }
        )

        with patch("subprocess.run", return_value=mock_result):
            jobs = client.get_user_jobs(name_pattern=r"^sifter-")

        assert len(jobs) == 1
        assert jobs[0].state == "PENDING"
        assert jobs[0].is_active is True

    def test_elapsed_for_recently_started_job(self, tmp_path: Path) -> None:
        """Elapsed time should be accurate for recently started jobs."""
        client = SLURMClient(logs_dir=tmp_path)

        # Simulate a job that started 2 minutes ago
        now_epoch = int(datetime.now().timestamp())
        start_epoch = now_epoch - 120  # 2 minutes ago

        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                "jobs": [
                    {
                        "job_id": 12345,
                        "name": "sifter-build-test",
                        "job_state": ["RUNNING"],
                        "submit_time": {"set": True, "infinite": False, "number": start_epoch - 60},
                        "start_time": {"set": True, "infinite": False, "number": start_epoch},
                        "end_time": {"set": True, "infinite": False, "number": 0},
                        "nodes": "node01",
                    }
                ]
            }
        )

        with patch("subprocess.run", return_value=mock_result):
            job = client.get_job("12345")

        assert job is not None
        assert job.state == "RUNNING"
        assert job.start_time is not None

        # Elapsed should be approximately 2 minutes, not 4 hours
        elapsed_seconds = job.elapsed.total_seconds()
        assert 100 < elapsed_seconds < 200, (
            f"Elapsed should be ~2 minutes (120s), got {elapsed_seconds}s"
        )

    def test_get_user_jobs_with_since_uses_sacct(self, tmp_path: Path) -> None:
        """get_user_jobs with since parameter uses sacct for historical jobs."""
        client = SLURMClient(logs_dir=tmp_path)

        now_epoch = int(datetime.now().timestamp())

        # Mock sacct response (different format from squeue)
        sacct_response = json.dumps(
            {
                "jobs": [
                    {
                        "job_id": 12345,
                        "name": "sifter-build-vllm",
                        "state": {"current": ["COMPLETED"], "reason": "None"},
                        "time": {
                            "submission": now_epoch - 3600,
                            "start": now_epoch - 3600,
                            "end": now_epoch - 3000,
                            "elapsed": 600,
                        },
                        "nodes": {"list": ["node01"]},
                    },
                    {
                        "job_id": 12346,
                        "name": "sifter-build-pytorch",
                        "state": {"current": ["FAILED"], "reason": "None"},
                        "time": {
                            "submission": now_epoch - 7200,
                            "start": now_epoch - 7200,
                            "end": now_epoch - 7000,
                            "elapsed": 200,
                        },
                        "nodes": {"list": ["node02"]},
                    },
                ]
            }
        )

        # Mock squeue response (empty - no active jobs)
        squeue_response = json.dumps({"jobs": []})

        def mock_run(args, **_kwargs):
            result = MagicMock()
            if "sacct" in args:
                result.stdout = sacct_response
            else:  # squeue
                result.stdout = squeue_response
            return result

        with patch("subprocess.run", side_effect=mock_run) as mock_run_obj:
            jobs = client.get_user_jobs(
                name_pattern=r"^sifter-",
                since=timedelta(days=7),
            )

        # Should have called both sacct and squeue
        all_calls = [str(call) for call in mock_run_obj.call_args_list]
        assert any("sacct" in call for call in all_calls), f"Expected sacct call, got: {all_calls}"

        assert len(jobs) == 2
        assert jobs[0].state == "COMPLETED"
        assert jobs[1].state == "FAILED"

    def test_running_job_has_elapsed_and_deadline(self, tmp_path: Path) -> None:
        """Running job has accurate elapsed time and deadline.

        SLURM sets end_time to the deadline (start + time_limit) for running jobs.
        We should parse this as deadline and calculate elapsed from start to now.
        """
        client = SLURMClient(logs_dir=tmp_path)

        now_epoch = int(datetime.now().timestamp())
        start_epoch = now_epoch - 120  # Started 2 minutes ago
        deadline_epoch = start_epoch + 14400  # 4 hour time limit

        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                "jobs": [
                    {
                        "job_id": 178971,
                        "name": "sifter-build-test",
                        "job_state": ["RUNNING"],
                        "submit_time": {"set": True, "infinite": False, "number": start_epoch},
                        "start_time": {"set": True, "infinite": False, "number": start_epoch},
                        "end_time": {"set": True, "infinite": False, "number": deadline_epoch},
                        "nodes": "node01",
                    }
                ]
            }
        )

        with patch("subprocess.run", return_value=mock_result):
            job = client.get_job("178971")

        assert job is not None
        assert job.state == "RUNNING"

        # Elapsed should be ~2 minutes, NOT 4 hours
        elapsed_seconds = job.elapsed.total_seconds()
        assert elapsed_seconds < 300, f"Elapsed should be ~2 mins, got {elapsed_seconds}s"

        # Deadline should be set
        assert job.deadline is not None

        # Time remaining should be ~4 hours minus 2 minutes
        assert job.time_remaining is not None
        remaining_seconds = job.time_remaining.total_seconds()
        assert 14000 < remaining_seconds < 14400, (
            f"Time remaining should be ~4h, got {remaining_seconds}s"
        )

    def test_completed_job_has_no_deadline(self, tmp_path: Path) -> None:
        """Completed job uses end_time as actual end, not deadline."""
        client = SLURMClient(logs_dir=tmp_path)

        now_epoch = int(datetime.now().timestamp())
        start_epoch = now_epoch - 3600  # Started 1 hour ago
        end_epoch = now_epoch - 1800  # Ended 30 mins ago

        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                "jobs": [
                    {
                        "job_id": 12345,
                        "name": "sifter-build-test",
                        "job_state": ["COMPLETED"],
                        "submit_time": {"number": start_epoch - 60},
                        "start_time": {"number": start_epoch},
                        "end_time": {"number": end_epoch},
                        "nodes": "node01",
                    }
                ]
            }
        )

        with patch("subprocess.run", return_value=mock_result):
            job = client.get_job("12345")

        assert job is not None
        assert job.state == "COMPLETED"
        assert job.deadline is None
        assert job.time_remaining is None

        # Elapsed should be ~30 minutes (end - start)
        elapsed_seconds = job.elapsed.total_seconds()
        assert 1700 < elapsed_seconds < 1900, f"Elapsed should be ~30 mins, got {elapsed_seconds}s"

    def test_get_user_jobs_with_since_includes_pending(self, tmp_path: Path) -> None:
        """get_user_jobs with since includes PENDING jobs from squeue."""
        client = SLURMClient(logs_dir=tmp_path)

        now_epoch = int(datetime.now().timestamp())

        # sacct returns completed jobs only
        sacct_response = json.dumps(
            {
                "jobs": [
                    {
                        "job_id": 12345,
                        "name": "sifter-build-old",
                        "state": {"current": ["COMPLETED"]},
                        "time": {
                            "submission": now_epoch - 3600,
                            "start": now_epoch - 3600,
                            "end": now_epoch - 3000,
                        },
                        "nodes": {"list": ["node01"]},
                    }
                ]
            }
        )

        # squeue returns pending jobs
        squeue_response = json.dumps(
            {
                "jobs": [
                    {
                        "job_id": 12346,
                        "name": "sifter-build-new",
                        "job_state": ["PENDING"],
                        "submit_time": {"number": now_epoch - 60},
                        "start_time": {"number": 0},
                        "end_time": {"number": 0},
                    }
                ]
            }
        )

        def mock_run(args, **_kwargs):
            result = MagicMock()
            if "sacct" in args:
                result.stdout = sacct_response
            else:  # squeue
                result.stdout = squeue_response
            return result

        with patch("subprocess.run", side_effect=mock_run):
            jobs = client.get_user_jobs(name_pattern=r"^sifter-", since=timedelta(days=7))

        # Should include both completed (from sacct) and pending (from squeue)
        job_ids = {j.job_id for j in jobs}
        assert "12345" in job_ids, "Should include completed job from sacct"
        assert "12346" in job_ids, "Should include pending job from squeue"


class TestStageBuild:
    """Tests for stage_build method (new builds: format)."""

    def test_stage_build_creates_staging_dir(self, tmp_path: Path) -> None:
        """stage_build creates staging directory keyed by hash."""
        logs_dir = tmp_path / "logs"
        staging_dir = tmp_path / "staging"
        repo_dir = tmp_path / "repo"

        # Create definition file
        def_path = repo_dir / "definitions" / "test" / "test.def"
        def_path.parent.mkdir(parents=True)
        def_path.write_text("Bootstrap: docker\nFrom: ubuntu:22.04")

        client = SLURMClient(logs_dir=logs_dir, staging_dir=staging_dir)

        result = client.stage_build(
            content_hash="abc123def456",
            definition_path=Path("definitions/test/test.def"),
            args={"KEY": "value"},
            repo_dir=repo_dir,
        )

        assert result == staging_dir / "abc123def456"
        assert result.exists()

    def test_stage_build_copies_definition(self, tmp_path: Path) -> None:
        """stage_build copies definition file to staging."""
        logs_dir = tmp_path / "logs"
        staging_dir = tmp_path / "staging"
        repo_dir = tmp_path / "repo"

        def_path = repo_dir / "definitions" / "test" / "test.def"
        def_path.parent.mkdir(parents=True)
        def_path.write_text("Bootstrap: docker\nFrom: ubuntu:22.04")

        client = SLURMClient(logs_dir=logs_dir, staging_dir=staging_dir)

        result = client.stage_build(
            content_hash="abc123def456",
            definition_path=Path("definitions/test/test.def"),
            args={},
            repo_dir=repo_dir,
        )

        staged_def = result / "definitions" / "test" / "test.def"
        assert staged_def.exists()
        assert staged_def.read_text() == "Bootstrap: docker\nFrom: ubuntu:22.04"

    def test_stage_build_writes_args_env(self, tmp_path: Path) -> None:
        """stage_build writes build_args.env file."""
        logs_dir = tmp_path / "logs"
        staging_dir = tmp_path / "staging"
        repo_dir = tmp_path / "repo"

        def_path = repo_dir / "definitions" / "test" / "test.def"
        def_path.parent.mkdir(parents=True)
        def_path.write_text("Bootstrap: docker")

        client = SLURMClient(logs_dir=logs_dir, staging_dir=staging_dir)

        result = client.stage_build(
            content_hash="abc123def456",
            definition_path=Path("definitions/test/test.def"),
            args={"VERSION": "1.0", "DEBUG": "true"},
            repo_dir=repo_dir,
        )

        args_file = result / "build_args.env"
        assert args_file.exists()
        content = args_file.read_text()
        # Args should be sorted
        assert "DEBUG=true\n" in content
        assert "VERSION=1.0\n" in content

    def test_stage_build_requires_staging_dir(self, tmp_path: Path) -> None:
        """stage_build raises error if staging_dir not configured."""
        client = SLURMClient(logs_dir=tmp_path)  # No staging_dir

        with pytest.raises(SLURMError, match="staging_dir not configured"):
            client.stage_build(
                content_hash="abc123",
                definition_path=Path("test.def"),
                args={},
                repo_dir=tmp_path,
            )

    def test_stage_build_raises_for_missing_def(self, tmp_path: Path) -> None:
        """stage_build raises error for missing definition file."""
        client = SLURMClient(logs_dir=tmp_path, staging_dir=tmp_path / "staging")

        with pytest.raises(SLURMError, match="Definition file not found"):
            client.stage_build(
                content_hash="abc123",
                definition_path=Path("nonexistent.def"),
                args={},
                repo_dir=tmp_path,
            )


class TestSubmitStagedBuild:
    """Tests for submit_staged_build method."""

    def test_submit_staged_build_sets_env_vars(self, tmp_path: Path) -> None:
        """submit_staged_build passes correct env vars."""
        client = SLURMClient(logs_dir=tmp_path)

        script = tmp_path / "build.sh"
        script.write_text("#!/bin/bash\necho test")

        staging = tmp_path / "staging" / "abc123"
        staging.mkdir(parents=True)

        mock_result = MagicMock()
        mock_result.stdout = "12345\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.submit_staged_build(
                build_script=script,
                staging_dir=staging,
                definition_path=Path("definitions/test.def"),
                output_hash="abc123def456",
                cache_dir=tmp_path / "cache",
                tag="myapp_1.0",
                registry_dir=tmp_path / "dist",
                base_image_hash="xyz789",
                remote_push_cache_cmd="aws s3 cp /tmp/cache.sif s3://bucket/cache.sif",
            )

            # Check environment passed to subprocess
            call_kwargs = mock_run.call_args[1]
            env = call_kwargs["env"]
            assert env["STAGING_DIR"] == str(staging)
            assert env["DEFINITION_PATH"] == "definitions/test.def"
            assert env["OUTPUT_HASH"] == "abc123def456"
            assert env["CACHE_DIR"] == str(tmp_path / "cache")
            assert env["TAG_NAME"] == "myapp_1.0"
            assert env["REGISTRY_DIR"] == str(tmp_path / "dist")
            assert env["BASE_IMAGE_HASH"] == "xyz789"
            assert env["REMOTE_PUSH_CACHE_CMD"] == "aws s3 cp /tmp/cache.sif s3://bucket/cache.sif"
            # Builds never publish to the registry — that's an explicit `sifter push`.
            assert "REMOTE_PUSH_REGISTRY_CMD" not in env

    def test_submit_staged_build_with_reservation(self, tmp_path: Path) -> None:
        """submit_staged_build passes --reservation to sbatch."""
        client = SLURMClient(logs_dir=tmp_path)

        script = tmp_path / "build.sh"
        script.write_text("#!/bin/bash\necho test")

        staging = tmp_path / "staging" / "abc123"
        staging.mkdir(parents=True)

        mock_result = MagicMock()
        mock_result.stdout = "12345\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.submit_staged_build(
                build_script=script,
                staging_dir=staging,
                definition_path=Path("definitions/test.def"),
                output_hash="abc123def456",
                cache_dir=tmp_path / "cache",
                reservation="my-reservation",
            )

            call_args = mock_run.call_args[0][0]
            assert "--reservation" in call_args
            res_idx = call_args.index("--reservation")
            assert call_args[res_idx + 1] == "my-reservation"

    def test_submit_staged_build_without_reservation(self, tmp_path: Path) -> None:
        """submit_staged_build omits --reservation when not set."""
        client = SLURMClient(logs_dir=tmp_path)

        script = tmp_path / "build.sh"
        script.write_text("#!/bin/bash\necho test")

        staging = tmp_path / "staging" / "abc123"
        staging.mkdir(parents=True)

        mock_result = MagicMock()
        mock_result.stdout = "12345\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.submit_staged_build(
                build_script=script,
                staging_dir=staging,
                definition_path=Path("definitions/test.def"),
                output_hash="abc123def456",
                cache_dir=tmp_path / "cache",
            )

            call_args = mock_run.call_args[0][0]
            assert "--reservation" not in call_args


class TestSubmitRunJob:
    """Tests for submit_run_job method (sifter run --slurm)."""

    def test_submit_run_job_creates_script(self, tmp_path: Path) -> None:
        """submit_run_job submits the shared run script."""
        client = SLURMClient(logs_dir=tmp_path)

        mock_result = MagicMock()
        mock_result.stdout = "12345\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            job_id = client.submit_run_job(
                container_path=tmp_path / "myapp_1.0.sif",
                singularity_cmd=[
                    "singularity",
                    "exec",
                    "--nv",
                    str(tmp_path / "myapp_1.0.sif"),
                    "python",
                    "train.py",
                ],
                time="02:00:00",
                gpus=4,
            )

            assert job_id == "12345"
            call_args = mock_run.call_args[0][0]
            assert call_args[-1].endswith("sifter/scripts/run.sh")

    def test_submit_run_job_uses_correct_prefix(self, tmp_path: Path) -> None:
        """submit_run_job uses sifter-run- prefix for job names."""
        client = SLURMClient(logs_dir=tmp_path)

        mock_result = MagicMock()
        mock_result.stdout = "12345\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.submit_run_job(
                container_path=tmp_path / "vllm-0.14.0_0.0.5.sif",
                singularity_cmd=["singularity", "exec", str(tmp_path / "vllm-0.14.0_0.0.5.sif")],
                time="01:00:00",
                gpus=1,
            )

            call_args = mock_run.call_args[0][0]
            # Job name should start with sifter-run- prefix
            job_name_args = [
                arg for i, arg in enumerate(call_args) if i > 0 and call_args[i - 1] == "--job-name"
            ]
            assert len(job_name_args) == 1
            assert job_name_args[0].startswith(RUN_JOB_PREFIX)

    def test_submit_run_job_includes_time_and_gpus(self, tmp_path: Path) -> None:
        """submit_run_job includes time limit and GPU count in sbatch args."""
        client = SLURMClient(logs_dir=tmp_path)

        mock_result = MagicMock()
        mock_result.stdout = "12345\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.submit_run_job(
                container_path=tmp_path / "myapp.sif",
                singularity_cmd=[
                    "singularity",
                    "exec",
                    str(tmp_path / "myapp.sif"),
                    "python",
                    "train.py",
                ],
                time="04:00:00",
                gpus=2,
            )

            call_args = mock_run.call_args[0][0]
            assert "--time" in call_args
            assert "04:00:00" in call_args
            assert "--gpus" in call_args
            assert "2" in call_args

    def test_submit_run_job_extracts_container_name(self, tmp_path: Path) -> None:
        """submit_run_job extracts container name from path for job name."""
        client = SLURMClient(logs_dir=tmp_path)

        mock_result = MagicMock()
        mock_result.stdout = "12345\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.submit_run_job(
                container_path=tmp_path / "pytorch-2.9.1-cu126_0.0.5.sif",
                singularity_cmd=[
                    "singularity",
                    "exec",
                    str(tmp_path / "pytorch-2.9.1-cu126_0.0.5.sif"),
                ],
                time="01:00:00",
                gpus=1,
            )

            call_args = mock_run.call_args[0][0]
            job_name_args = [
                arg for i, arg in enumerate(call_args) if i > 0 and call_args[i - 1] == "--job-name"
            ]
            # Job name should be based on container name
            assert "pytorch-2.9.1-cu126" in job_name_args[0]


class TestGetRunJobs:
    """Tests for detecting sifter run jobs in status."""

    def test_status_pattern_matches_run_jobs(self) -> None:
        """The ^sifter- pattern used by status matches run jobs."""
        import re

        pattern = r"^sifter-"

        # These should all match
        assert re.match(pattern, "sifter-run-myapp")
        assert re.match(pattern, "sifter-run-pytorch-2.9.1")
        assert re.match(pattern, "sifter-build-vllm")
        assert re.match(pattern, "sifter-pull-container")
        assert re.match(pattern, "sifter-push-container")

        # These should not match
        assert not re.match(pattern, "other-job")
        assert not re.match(pattern, "my-sifter-job")

    def test_get_user_jobs_returns_run_jobs(self, tmp_path: Path) -> None:
        """get_user_jobs with ^sifter- pattern returns run jobs."""
        client = SLURMClient(logs_dir=tmp_path)

        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                "jobs": [
                    {
                        "job_id": 123,
                        "name": "sifter-run-myapp",
                        "job_state": "RUNNING",
                        "submit_time": {"number": 1700000000},
                        "start_time": {"number": 1700000100},
                        "end_time": {"number": 0},
                    },
                    {
                        "job_id": 124,
                        "name": "sifter-build-pytorch",
                        "job_state": "RUNNING",
                        "submit_time": {"number": 1700000000},
                        "start_time": {"number": 0},
                        "end_time": {"number": 0},
                    },
                ]
            }
        )

        with patch("subprocess.run", return_value=mock_result):
            jobs = client.get_user_jobs(name_pattern=r"^sifter-")

        assert len(jobs) == 2
        names = {j.name for j in jobs}
        assert "sifter-run-myapp" in names
        assert "sifter-build-pytorch" in names


class TestSLURMJobTimezoneAwareness:
    """Tests for timezone-aware timestamps in SLURM jobs."""

    def test_job_timestamps_are_utc_aware(self, tmp_path: Path) -> None:
        """SLURMJob timestamps should be timezone-aware (UTC)."""
        client = SLURMClient(logs_dir=tmp_path)
        now_epoch = int(datetime.now(timezone.utc).timestamp())

        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                "jobs": [
                    {
                        "job_id": 12345,
                        "name": "test-job",
                        "job_state": "RUNNING",
                        "submit_time": {"number": now_epoch - 3600},
                        "start_time": {"number": now_epoch - 120},
                        "end_time": {"number": now_epoch + 3600},
                    }
                ]
            }
        )
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            job = client.get_job("12345")

        assert job is not None
        assert job.start_time is not None
        assert job.start_time.tzinfo is not None
        assert job.submit_time is not None
        assert job.submit_time.tzinfo is not None

    def test_elapsed_time_correct_with_utc_timestamps(self, tmp_path: Path) -> None:
        """Elapsed time calculation works correctly with UTC timestamps."""
        client = SLURMClient(logs_dir=tmp_path)
        now_epoch = int(datetime.now(timezone.utc).timestamp())
        start_epoch = now_epoch - 300  # 5 minutes ago

        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                "jobs": [
                    {
                        "job_id": 12345,
                        "name": "test-job",
                        "job_state": "RUNNING",
                        "submit_time": {"number": start_epoch - 60},
                        "start_time": {"number": start_epoch},
                        "end_time": {"number": now_epoch + 3600},
                    }
                ]
            }
        )
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            job = client.get_job("12345")

        assert job is not None
        elapsed_seconds = job.elapsed.total_seconds()
        # Allow 2 second tolerance for test execution time
        assert 298 <= elapsed_seconds <= 310

    def test_time_remaining_correct_with_utc_deadline(self, tmp_path: Path) -> None:
        """Time remaining calculation works with UTC deadline."""
        client = SLURMClient(logs_dir=tmp_path)
        now_epoch = int(datetime.now(timezone.utc).timestamp())
        deadline_epoch = now_epoch + 1800  # 30 minutes from now

        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                "jobs": [
                    {
                        "job_id": 12345,
                        "name": "test-job",
                        "job_state": "RUNNING",
                        "submit_time": {"number": now_epoch - 3600},
                        "start_time": {"number": now_epoch - 600},
                        "end_time": {"number": deadline_epoch},
                    }
                ]
            }
        )
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            job = client.get_job("12345")

        assert job is not None
        assert job.time_remaining is not None
        remaining_seconds = job.time_remaining.total_seconds()
        # Should be approximately 30 minutes (1800 seconds)
        assert 1790 <= remaining_seconds <= 1810


class TestSLURMResourceFlags:
    """Tests for parametrised SLURM resource directives (SIFTER_SLURM_*)."""

    @staticmethod
    def _flag(args: list[str], name: str) -> str | None:
        """Return the value passed for *name* in an sbatch argv, or None."""
        return args[args.index(name) + 1] if name in args else None

    def _submit_build(self, client: SLURMClient, tmp_path: Path) -> list[str]:
        script = tmp_path / "build.sh"
        script.write_text("#!/bin/bash\necho test")
        staging = tmp_path / "staging" / "abc123"
        staging.mkdir(parents=True)

        mock_result = MagicMock()
        mock_result.stdout = "12345\n"
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.submit_staged_build(
                build_script=script,
                staging_dir=staging,
                definition_path=Path("definitions/test.def"),
                output_hash="abc123def456",
                cache_dir=tmp_path / "cache",
            )
        args: list[str] = mock_run.call_args[0][0]
        return args

    def test_client_defaults_match_script_directives(self, tmp_path: Path) -> None:
        """Default resources preserve the historical #SBATCH values in build.sh."""
        client = SLURMClient(logs_dir=tmp_path)

        args = self._submit_build(client, tmp_path)

        assert self._flag(args, "--partition") == "workq"
        assert self._flag(args, "--cpus-per-task") == "144"
        assert self._flag(args, "--mem") == "0"

    def test_build_job_uses_configured_resources(self, tmp_path: Path) -> None:
        """Build jobs pass the configured partition/cpus/mem as sbatch flags."""
        client = SLURMClient(logs_dir=tmp_path, partition="gpu", cpus=32, mem="128G")

        args = self._submit_build(client, tmp_path)

        assert self._flag(args, "--partition") == "gpu"
        assert self._flag(args, "--cpus-per-task") == "32"
        assert self._flag(args, "--mem") == "128G"

    def test_transfer_job_sets_partition_only(self, tmp_path: Path) -> None:
        """Transfer jobs take the partition but keep transfer.sh's own cpus/mem."""
        client = SLURMClient(logs_dir=tmp_path, partition="gpu", cpus=32, mem="128G")

        mock_result = MagicMock()
        mock_result.stdout = "12345\n"
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.submit_transfer_job(
                command="aws s3 cp s3://bucket/file.sif /local/file.sif",
                filename="file.sif",
                is_pull=True,
            )

        args: list[str] = mock_run.call_args[0][0]
        assert self._flag(args, "--partition") == "gpu"
        assert "--cpus-per-task" not in args
        assert "--mem" not in args

    def test_run_job_uses_configured_partition_and_gpus(self, tmp_path: Path) -> None:
        """Run jobs take the configured partition and default GPU count."""
        client = SLURMClient(logs_dir=tmp_path, partition="gpu", gpus=4)

        mock_result = MagicMock()
        mock_result.stdout = "12345\n"
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.submit_run_job(
                container_path=tmp_path / "myapp_1.0.sif",
                singularity_cmd=["singularity", "exec", str(tmp_path / "myapp_1.0.sif")],
            )

        args: list[str] = mock_run.call_args[0][0]
        assert self._flag(args, "--partition") == "gpu"
        assert self._flag(args, "--gpus") == "4"

    def test_run_job_gpus_argument_overrides_config(self, tmp_path: Path) -> None:
        """An explicit gpus argument still wins over the configured default."""
        client = SLURMClient(logs_dir=tmp_path, gpus=4)

        mock_result = MagicMock()
        mock_result.stdout = "12345\n"
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.submit_run_job(
                container_path=tmp_path / "myapp_1.0.sif",
                singularity_cmd=["singularity", "exec", str(tmp_path / "myapp_1.0.sif")],
                gpus=2,
            )

        args: list[str] = mock_run.call_args[0][0]
        assert self._flag(args, "--gpus") == "2"
