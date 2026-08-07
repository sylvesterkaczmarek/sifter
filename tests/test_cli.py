"""Tests for sifter.cli module."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from sifter.cli import app
from sifter.slurm import SLURMJob

runner = CliRunner()


class TestBuildTreeRendering:
    """Tests for build tree visualization."""

    def test_build_tree_shows_dependency_as_root(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """Build tree shows pytorch (dependency) as root, not vllm.

        When building vllm-0.14.0 which depends on pytorch-2.9.1-cu126:
        - pytorch should appear as the root of the tree (no indentation)
        - vllm should appear as a child/nested under pytorch
        """
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(app, ["build", "vllm-0.14.0", "--dry-run"])

        assert result.exit_code == 0
        assert "Build Plan" in result.output

        # The tree should show pytorch as the root (first node in tree)
        # and vllm as a dependent/child.
        # Find tree lines - pytorch should appear BEFORE vllm in tree output
        lines = result.output.split("\n")

        # Find the line positions
        pytorch_line_idx = None
        vllm_line_idx = None
        for i, line in enumerate(lines):
            if "pytorch-2.9.1-cu126" in line:
                pytorch_line_idx = i
            if "vllm-0.14.0" in line:
                vllm_line_idx = i

        assert pytorch_line_idx is not None, "pytorch should appear in output"
        assert vllm_line_idx is not None, "vllm should appear in output"

        # pytorch (the dependency) should appear BEFORE vllm in the tree
        # because it's the root of the build order
        assert pytorch_line_idx < vllm_line_idx, (
            f"pytorch (dependency) should appear before vllm in tree. "
            f"Got pytorch at line {pytorch_line_idx}, vllm at {vllm_line_idx}"
        )


class TestVersion:
    """Tests for version command."""

    def test_version(self) -> None:
        """version command shows version."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "sifter" in result.output


class TestConfigErrors:
    """A bad setting is a user's mistake to correct, so it must read as one."""

    def test_an_unusable_registry_name_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # Validation moved into the config model, so its rejections now surface
        # through the same path as any other bad setting rather than as a traceback.
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.setenv("SIFTER_REGISTRIES", "http://localhost:5000/team")
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["ls"])

        assert result.exit_code == 1
        assert "Configuration error" in result.output
        assert "plain-HTTP" in result.output

    def test_a_rejected_setting_is_quoted_back_as_the_repo_wrote_it(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # The error names the value it rejected, so the value chooses part of the
        # message. Read as console input it renders as something else entirely.
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        with (sample_repo / "sifter.yaml").open("a") as manifest:
            manifest.write('oci:\n  registries: ["-:skull:.example.com/team"]\n')
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["ls"])

        assert result.exit_code == 1
        assert "Configuration error" in result.output
        assert ":skull:" in result.output

    @pytest.mark.parametrize("registry", ["\\e[2K\\e[Areg.example.com/team", "a\\0b.example.com"])
    def test_a_registry_name_carrying_a_control_character_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path, registry: str
    ) -> None:
        # The name reaches a terminal and an argv. A terminal acts on these bytes, so
        # a name can rewrite what the operator was told; execve rejects NUL outright.
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        with (sample_repo / "sifter.yaml").open("a") as manifest:
            manifest.write(f'oci:\n  registries: ["{registry}"]\n')
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["ls"])

        assert result.exit_code == 1
        assert "Configuration error" in result.output
        assert "\x1b" not in result.output
        assert "\0" not in result.output

    def test_a_bad_setting_reads_as_a_mistake_when_running_a_container(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # Running a container resolves a config like every other command does, so a
        # bad one is the user's to correct here too rather than a traceback to report.
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        with (sample_repo / "sifter.yaml").open("a") as manifest:
            manifest.write("oci:\n  registries: [http://localhost:5000/team]\n")
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["run", "-i", "nosuch_1.0.0"])

        assert result.exit_code == 1
        assert "Configuration error" in result.output


class TestARefusedProjectSettingIsVisible:
    """The repo asked for something and did not get it; that has to reach the user."""

    def test_a_refused_setting_is_named_in_the_command_output(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # Hearing nothing is indistinguishable from having been obeyed. Warnings are
        # fatal here: a notice sent through them would abort instead of informing.
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        with (sample_repo / "sifter.yaml").open("a") as manifest:
            manifest.write("oci:\n  registries: [reg.example.com/team]\n  oras_bin: ./payload\n")
        monkeypatch.chdir(sample_repo)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = runner.invoke(app, ["ls"])

        assert result.exit_code == 0
        assert "oras_bin" in result.output
        assert "sifter.yaml" in result.output

    @pytest.mark.parametrize("key", ["oras_bin[/red]", ":heavy_check_mark: oras_bin"])
    def test_a_refused_key_is_reported_as_written(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path, key: str
    ) -> None:
        # The notice quotes the repo's own text back, so anything the console reads as
        # an instruction hands the repo the notice: it can restyle it, delete
        # characters from it, or make printing it the thing that fails.
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        with (sample_repo / "sifter.yaml").open("a") as manifest:
            manifest.write(f'oci:\n  registries: [reg.example.com/team]\n  "{key}": x\n')
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["ls"])

        assert result.exit_code == 0
        assert key in result.output

    def test_a_refused_setting_is_named_when_running_a_container(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # The run fails for its own reasons here; the point is that the refusal was
        # said out loud first, on a command that resolves a config without showing one.
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        with (sample_repo / "sifter.yaml").open("a") as manifest:
            manifest.write("oci:\n  registries: [reg.example.com/team]\n  oras_bin: ./payload\n")
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["run", "-i", "nosuch_1.0.0"])

        assert "oras_bin" in result.output

    def test_a_registry_is_shown_as_the_repo_spelled_it(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # The displayed name is how anyone checks where a push is going, so it has
        # to be the name that reaches oras, not a prettier rendering of it.
        registry = "reg.example.com/[red]team[/red]"
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        with (sample_repo / "sifter.yaml").open("a") as manifest:
            manifest.write(f"oci:\n  registries: ['{registry}']\n")
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["ls"])

        assert result.exit_code == 0
        assert registry in result.output


class TestExtraRegistriesAreReportedAsUnused:
    """Only the first registry is consulted; the rest must not pass unmentioned."""

    def test_the_unused_registries_are_named_without_going_through_warnings(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # A repo picks the registry list, so a repo decides whether this message is
        # emitted. As a warning it either vanishes or aborts the command, on filters
        # set far from here — and aborting takes the refusal notice down with it.
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        with (sample_repo / "sifter.yaml").open("a") as manifest:
            manifest.write("oci:\n  registries: [reg.a/team, reg.b/team]\n  oras_bin: ./payload\n")
        monkeypatch.chdir(sample_repo)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = runner.invoke(app, ["ls"])

        assert result.exit_code == 0
        assert "reg.b/team" in result.output
        assert "oras_bin" in result.output


class TestEveryCommandReportsWhatItRefused:
    """A command that resolves a configuration has to say what it did not honour."""

    def test_a_refused_setting_is_named_when_reading_a_build_log(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # Reading a log resolves a full configuration, so a repo asking for the binary
        # sifter runs is the security-relevant event here as much as anywhere else.
        scratch = sample_repo / "scratch"
        logs_dir = scratch / "sifter" / "logs"
        logs_dir.mkdir(parents=True)
        (logs_dir / "demo_12345.log").write_text("build finished\n")
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        with (sample_repo / "sifter.yaml").open("a") as manifest:
            manifest.write("oci:\n  registries: [reg.example.com/team]\n  oras_bin: ./payload\n")
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient.get_job", return_value=None):
            result = runner.invoke(app, ["logs", "12345"])

        assert result.exit_code == 0
        assert "build finished" in result.output
        assert "oras_bin" in result.output

    def test_a_bad_setting_reads_as_a_mistake_when_reading_a_build_log(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        scratch = sample_repo / "scratch"
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        with (sample_repo / "sifter.yaml").open("a") as manifest:
            manifest.write("oci:\n  registries: [http://localhost:5000/team]\n")
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["logs", "12345"])

        assert result.exit_code == 1
        assert "Configuration error" in result.output

    def test_a_bad_setting_reads_as_a_mistake_when_asking_for_the_latest_container(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        scratch = sample_repo / "scratch"
        (scratch / "sifter" / "registry").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        with (sample_repo / "sifter.yaml").open("a") as manifest:
            manifest.write("oci:\n  registries: [http://localhost:5000/team]\n")
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["latest"])

        assert result.exit_code == 1
        assert "Configuration error" in result.output

    def test_a_notice_leaves_a_composable_path_alone(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # `singularity exec $(sifter latest)` has to keep working in a repo whose
        # settings were refused, so the notice cannot be reported on stdout.
        scratch = sample_repo / "scratch"
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)
        (dist_dir / "demo-1.0.0_0.1.0.sif").touch()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        with (sample_repo / "sifter.yaml").open("a") as manifest:
            manifest.write("oci:\n  registries: [reg.example.com/team]\n  oras_bin: ./payload\n")
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["latest"])

        assert result.exit_code == 0
        assert result.stdout.strip() == str(dist_dir / "demo-1.0.0_0.1.0.sif")
        assert "oras_bin" in result.stderr

    def test_a_project_file_cannot_bury_the_notice_in_its_own_output(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # The repo writes the keys the notice quotes back, so an unbounded notice lets
        # it scroll the line that matters off the screen instead of suppressing it.
        scratch = sample_repo / "scratch"
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        flood = "".join(f"  k{i}: x\n" for i in range(5000))
        with (sample_repo / "sifter.yaml").open("a") as manifest:
            manifest.write(f"oci:\n  registries: [reg.example.com/team]\n{flood}")
            manifest.write(f"  {'w' * 1000}: x\n")
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["ls"])

        assert result.exit_code == 0
        assert "cannot make these choices for you" in " ".join(result.stderr.split())
        assert len(result.stderr) < 2000


class TestRepoTextCannotRewriteTheScreen:
    """A refusal the repo can erase from the terminal is a refusal it can suppress."""

    @pytest.mark.parametrize(
        "section",
        [
            'registry:\n  registry_type: filesystem\n  registry_path: "/srv/\\e[8A\\e[2K\\e[J"\n',
            "registry:\n  registry_type: filesystem\n  registry_path: /srv/reg\n"
            '  registry_cache_path: "/srv/\\e[8A\\e[2K\\e[J"\n',
            'registry:\n  registry_type: s3\n  s3_bucket: "buck\\e[8A\\e[2K\\e[Jet"\n',
            'registry:\n  registry_type: s3\n  s3_bucket: buck\n  s3_prefix: "\\e[2K\\e[J"\n',
            'registry:\n  registry_type: s3\n  s3_bucket: buck\n  s3_cache_prefix: "\\e[2K\\e[J"\n',
            "registry:\n  registry_type: s3\n  s3_bucket: buck\n"
            '  s3_registry_prefix: "\\e[2K\\e[J"\n',
        ],
    )
    def test_a_remote_named_with_a_terminal_escape_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path, section: str
    ) -> None:
        # The summary names the active backend, and a terminal acts on these bytes: a
        # name beginning with a cursor-up-and-erase deletes the lines that named it.
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        with (sample_repo / "sifter.yaml").open("a") as manifest:
            manifest.write(section)
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["ls"])

        assert result.exit_code == 1
        assert "Configuration error" in result.output
        assert "\x1b" not in result.output

    def test_a_long_remote_name_cannot_scroll_the_notice_away(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # A registry name is one of the two settings a repo may still choose, and the
        # summary prints it next to the notice — unbounded, it pushes it off the screen.
        scratch = sample_repo / "scratch"
        (scratch / "sifter" / "registry").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        with (sample_repo / "sifter.yaml").open("a") as manifest:
            manifest.write(f'oci:\n  registries: ["{"a" * 40000}.example.com/team"]\n')
            manifest.write("  oras_bin: ./payload\n")
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["push", "--all"])

        assert "oras_bin" in result.stderr
        assert len(result.output) < 4000

    def test_a_build_named_with_a_terminal_escape_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # A build name is displayed and becomes a filename, and the repo writes it.
        scratch = sample_repo / "scratch"
        (scratch / "sifter" / "registry").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        manifest_path = sample_repo / "sifter.yaml"
        manifest_path.write_text(
            'builds:\n  "demo\\e[8A\\e[2K\\e[J_1.0.0":\n'
            "    steps:\n      - path: definitions/pytorch/pytorch.def\n"
        )
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["build", "--all", "--dry-run"])

        assert result.exit_code == 1
        assert "\x1b" not in result.output

    def test_a_build_name_is_displayed_as_the_repo_wrote_it(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # Console markup in a build name otherwise either hides half the name or, when
        # it does not balance, replaces the build plan with a traceback.
        scratch = sample_repo / "scratch"
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "cache").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        manifest_path = sample_repo / "sifter.yaml"
        manifest_path.write_text(
            'builds:\n  "demo[red]x_1.0.0":\n'
            "    steps:\n      - path: definitions/pytorch/pytorch.def\n"
        )
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(app, ["build", "--all", "--dry-run"])

        assert result.exit_code == 0
        assert "demo[red]x" in result.output

    def test_a_base_image_named_with_a_terminal_escape_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # A base is quoted back when it cannot be resolved, and the escape below
        # scrolls up and erases the lines that named it.
        scratch = sample_repo / "scratch"
        (scratch / "sifter" / "registry").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        manifest_path = sample_repo / "sifter.yaml"
        manifest_path.write_text(
            'builds:\n  "demo_1.0.0":\n    base: "\\e[8A\\e[2K\\e[Jelsewhere_9.9.9"\n'
            "    steps:\n      - path: definitions/pytorch/pytorch.def\n"
        )
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["build", "--all", "--dry-run"])

        assert result.exit_code == 1
        assert "\x1b" not in result.output

    def test_a_build_name_is_displayed_as_the_repo_wrote_it_when_publishing(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # The name of a container about to be published is how anyone checks what is
        # going out, so it has to survive to the screen as the repo spelled it.
        scratch = sample_repo / "scratch"
        registry = scratch / "sifter" / "registry"
        registry.mkdir(parents=True)
        (registry / "boom[dim]_2.0.0.sif").write_text("sif")
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        manifest_path = sample_repo / "sifter.yaml"
        manifest_path.write_text(
            'builds:\n  "boom[dim]_2.0.0":\n'
            "    steps:\n      - path: definitions/pytorch/pytorch.def\n"
        )
        monkeypatch.chdir(sample_repo)
        remote = MagicMock(exists=MagicMock(return_value=False))

        with (
            patch("sifter.api.SifterConfig.create_remote_registry", return_value=remote),
            patch("sifter.api.SLURMClient.submit_transfer_job", return_value="99"),
        ):
            result = runner.invoke(app, ["push", "--all"])

        assert result.exit_code == 0
        assert "Push job submitted" in result.output
        assert "boom[dim]_2.0.0.sif" in result.output

    def test_a_container_is_deleted_under_the_name_it_was_shown_as(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # A build key becomes a filename, so the deletion list is repo text too. A
        # name shown with half of itself hidden is a confirmation of the wrong thing.
        scratch = sample_repo / "scratch"
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)
        (dist_dir / "app[red]hidden_1.0.0.sif").write_text("sif")
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["rm", "--all", "--yes"])

        assert result.exit_code == 0
        assert result.output.count("app[red]hidden_1.0.0.sif") == 2

    def test_a_container_listing_shows_both_halves_of_the_name(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # The listing splits a filename into a tree root and a leaf, and both halves
        # are repo text; either one parsed as markup hides part of what is installed.
        scratch = sample_repo / "scratch"
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)
        (dist_dir / "app[bold]gone_1.0[dim]0.sif").write_text("sif")
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["ls"])

        assert result.exit_code == 0
        assert "app[bold]gone" in result.output
        assert "1.0[dim]0.sif" in result.output

    def test_a_build_log_is_shown_rather_than_interpreted(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # A build log is whatever the repo's own definition files printed while
        # building, so reading one back must not be a way to fail or restyle it.
        scratch = sample_repo / "scratch"
        logs_dir = scratch / "sifter" / "logs"
        logs_dir.mkdir(parents=True)
        (logs_dir / "demo_12345.log").write_text(
            "FATAL [/red] at :skull: stage\x1b[2JOVERWRITTEN\n"
        )
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient.get_job", return_value=None):
            result = runner.invoke(app, ["logs", "12345"])

        assert result.exit_code == 0
        assert "FATAL [/red] at :skull: stage" in result.output
        assert r"\x1b[2JOVERWRITTEN" in result.output
        assert "\x1b[2J" not in result.output

    @pytest.mark.parametrize("args", [["status"], ["status", "--id", "12345"]])
    def test_a_job_is_listed_under_the_name_the_repo_gave_it(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path, args: list[str]
    ) -> None:
        # A job name is the build key the repo wrote, and a table cell is parsed as
        # console markup like any other string, so half the name would go missing.
        scratch = sample_repo / "scratch"
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)
        job = SLURMJob(job_id="12345", name="sifter-build-demo[/red]x", state="RUNNING")

        with patch("sifter.cli.get_jobs", return_value=[job]):
            result = runner.invoke(app, args)

        assert result.exit_code == 0
        assert "demo[/red]x" in result.output

    def test_a_missing_container_does_not_arrive_as_a_path(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        # `singularity exec $(sifter latest)` would otherwise be handed the failure
        # message as its argument.
        scratch = sample_repo / "scratch"
        (scratch / "sifter" / "registry").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["latest"])

        assert result.exit_code == 1
        assert result.stdout.strip() == ""


class TestLs:
    """Tests for ls command (renamed from list)."""

    def test_ls_works(self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> None:
        """ls command works."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(app, ["ls"])

        assert result.exit_code == 0
        assert "No containers found" in result.output

    def test_list_alias_works(self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> None:
        """list command still works as alias."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(app, ["ls"])

        assert result.exit_code == 0


class TestList:
    """Tests for list command."""

    def test_list_local_empty(self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> None:
        """list shows empty when no containers."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        # Mock SLURM to avoid calling real squeue
        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(app, ["ls"])

        assert result.exit_code == 0
        assert "No containers found" in result.output

    def test_list_local_with_containers(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """list shows local containers."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)

        # Create some containers
        (dist_dir / "vllm-0.14.0_0.0.5.sif").write_text("content")
        (dist_dir / "pytorch-2.9.1-cu126_0.0.5.sif").write_text("content")

        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(app, ["ls"])

        assert result.exit_code == 0
        assert "vllm" in result.output
        assert "pytorch" in result.output

    def test_list_with_name_filter(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """list --name filters by name."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)

        (dist_dir / "vllm-0.14.0_0.0.5.sif").write_text("content")
        (dist_dir / "pytorch-2.9.1-cu126_0.0.5.sif").write_text("content")

        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(app, ["ls", "--name", "vllm-0.14.0"])

        assert result.exit_code == 0
        assert "vllm" in result.output
        # pytorch should not be shown
        lines = result.output.split("\n")
        assert not any("pytorch-2.9.1" in line for line in lines)


class TestBuild:
    """Tests for build command."""

    def test_build_requires_name_or_all(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """build requires build name or --all."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["build"])

        assert result.exit_code == 1
        assert "Specify" in result.output

    def test_build_unknown_variant(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """build with unknown build shows error."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(app, ["build", "nonexistent-variant"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_build_dry_run(self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> None:
        """build --dry-run shows plan without submitting."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(app, ["build", "pytorch-2.9.1-cu126", "--dry-run"])

        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert "Build Plan" in result.output

    def test_build_all_tags_all_containers(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """build --all should tag ALL containers in the manifest.

        This tests the fix for the issue where dependencies added first with
        should_tag=False wouldn't get updated when processed as top-level targets.
        """
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "cache").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(app, ["build", "--all", "--dry-run"])

        assert result.exit_code == 0

        # All three containers should have registry tags shown in the output
        # The tree output shows: variant (cache_file) → registry_tag
        assert "pytorch-2.9.1-cu126_0.0.5" in result.output
        assert "vllm-0.14.0_0.0.5" in result.output
        assert "vllm-0.14.1_0.0.5" in result.output

    def test_build_single_cached_target_skips_dependencies(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """When target exists in local cache, dependencies should NOT be processed.

        If vllm-0.14.0 exists in cache, we don't need to build/pull/check pytorch.
        The dependency is redundant because we already have the final artifact.
        """
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        cache_dir = scratch / "sifter" / "cache"
        cache_dir.mkdir(parents=True)
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        with (
            patch("sifter.api.SLURMClient"),
            patch("sifter.api.Storage.local_exists", return_value=True),
        ):
            result = runner.invoke(app, ["build", "vllm-0.14.0", "--dry-run"])

        assert result.exit_code == 0
        # vllm should show as TAG_CACHED
        assert "TAG_CACHED" in result.output
        assert "vllm-0.14.0" in result.output
        # pytorch (the dependency) should NOT appear because vllm is already cached
        assert "pytorch" not in result.output

    def test_build_single_target_needs_build_includes_dependencies(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """When target needs building, dependencies SHOULD be processed.

        If vllm-0.14.0 is not in cache, we need to check/build pytorch first.
        """
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "cache").mkdir(parents=True)
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient"):
            # No cache mocking - nothing is cached, so builds are needed
            result = runner.invoke(app, ["build", "vllm-0.14.0", "--dry-run"])

        assert result.exit_code == 0
        # Both should appear because vllm needs building
        assert "vllm-0.14.0" in result.output
        assert "pytorch-2.9.1-cu126" in result.output
        # Both should show as BUILD
        assert "BUILD" in result.output

    def test_build_single_target_remote_cached_skips_dependencies(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """When target exists in remote cache, dependencies should NOT be processed.

        If vllm-0.14.0 exists in remote cache, we pull it but don't need pytorch.
        """
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        cache_dir = scratch / "sifter" / "cache"
        cache_dir.mkdir(parents=True)
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        mock_remote = MagicMock()
        mock_remote.exists.return_value = True

        with (
            patch("sifter.api.SLURMClient"),
            patch("sifter.api.Storage.local_exists", return_value=False),
            patch("sifter.api.SifterConfig.create_remote_cache", return_value=mock_remote),
        ):
            result = runner.invoke(app, ["build", "vllm-0.14.0", "--dry-run"])

        assert result.exit_code == 0
        # vllm should show as PULL_REMOTE
        assert "PULL_REMOTE" in result.output
        assert "vllm-0.14.0" in result.output
        # pytorch (the dependency) should NOT appear
        assert "pytorch" not in result.output

    def test_build_warns_when_overwriting_existing_tag(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """Build should warn when a tagged container already exists in registry.

        If pytorch-2.9.1-cu126_0.0.5.sif already exists in the local registry,
        warn that it will be overwritten.
        """
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        cache_dir = scratch / "sifter" / "cache"
        cache_dir.mkdir(parents=True)
        registry_dir = scratch / "sifter" / "registry"
        registry_dir.mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        # Create an existing tagged container in the registry
        existing_tag = registry_dir / "pytorch-2.9.1-cu126_0.0.5.sif"
        existing_tag.write_bytes(b"fake container")

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(app, ["build", "pytorch-2.9.1-cu126", "--dry-run"])

        assert result.exit_code == 0
        # Should show a warning about overwriting
        assert (
            "will be overwritten" in result.output.lower() or "overwrite" in result.output.lower()
        )


class TestStatus:
    """Tests for status command."""

    def test_status_no_jobs(self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> None:
        """status shows message when no jobs."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient") as mock_slurm:
            mock_slurm.return_value.get_user_jobs.return_value = []

            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "No sifter jobs found" in result.output


class TestPush:
    """Tests for push command."""

    def test_push_requires_filename_or_all(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """push requires a positional filename or --all."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["push"])

        assert result.exit_code == 1
        assert "--all" in result.output

    def test_push_single_and_all_mutually_exclusive(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """push cannot take a single image and --all together."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["push", "test.sif", "--all"])

        assert result.exit_code == 1
        assert "--all together" in result.output

    def test_push_no_remote_configured(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """push --name with no remote registry shows error."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)
        monkeypatch.delenv("SIFTER_REGISTRY", raising=False)

        result = runner.invoke(app, ["push", "--name", "nonexistent.sif"])

        assert result.exit_code == 1
        assert "No remote registry configured" in result.output

    def test_push_release_no_files(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """push --all with no manifest containers locally shows message."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)
        # Create only dev files (not in manifest)
        (dist_dir / "test_0.0.5+devg123456789012.sif").touch()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        mock_remote = MagicMock()
        with patch("sifter.api.SifterConfig.create_remote_registry", return_value=mock_remote):
            result = runner.invoke(app, ["push", "--all"])

        assert result.exit_code == 0
        assert "No manifest containers found" in result.output

    def test_push_accepts_positional_filename(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """push takes the SIF filename positionally, mirroring pull."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)
        (dist_dir / "pytorch-2.9.1-cu126_0.0.5.sif").touch()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        mock_remote = MagicMock()
        mock_remote.exists.return_value = False
        mock_remote.generate_push_command.return_value = "echo push"
        with (
            patch("sifter.api.SifterConfig.create_remote_registry", return_value=mock_remote),
            patch("sifter.api.SLURMClient.submit_transfer_job", return_value="99"),
        ):
            result = runner.invoke(app, ["push", "pytorch-2.9.1-cu126_0.0.5.sif"])

        assert result.exit_code == 0
        assert "Push job submitted" in result.output

    def test_push_release_only_pushes_manifest_containers(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """push --release only pushes containers defined in manifest."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)
        # Create a manifest-defined container AND an extra one
        (dist_dir / "pytorch-2.9.1-cu126_0.0.5.sif").touch()
        (dist_dir / "other-container_1.0.0.sif").touch()  # Not in manifest
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        mock_remote = MagicMock()
        mock_remote.exists.return_value = False
        mock_remote.generate_push_command.return_value = "echo push"

        with (
            patch("sifter.api.SifterConfig.create_remote_registry", return_value=mock_remote),
            patch("sifter.api.SLURMClient.submit_transfer_job") as mock_submit,
        ):
            mock_submit.return_value = "12345"
            result = runner.invoke(app, ["push", "--release"])

        assert result.exit_code == 0
        # Should push manifest container
        assert "pytorch-2.9.1-cu126_0.0.5" in result.output
        # Should NOT push other container
        assert "other-container_1.0.0" not in result.output
        # Should only call submit once (for the manifest container)
        assert mock_submit.call_count == 1


class TestRm:
    """Tests for rm command."""

    def test_rm_requires_name_or_all(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """rm requires --name or --all."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["rm"])

        assert result.exit_code == 1
        assert "--name" in result.output or "--all" in result.output

    def test_rm_by_name_deletes_matching(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """rm --name deletes matching containers."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)

        # Create some containers
        (dist_dir / "vllm-0.14.0_0.0.5.sif").write_text("content")
        (dist_dir / "vllm-0.14.0_0.0.5.sif").write_text("content")
        (dist_dir / "pytorch-2.9.1-cu126_0.0.5.sif").write_text("content")

        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["rm", "--name", "vllm", "--yes"])

        assert result.exit_code == 0
        # vllm containers should be deleted
        assert not (dist_dir / "vllm-0.14.0_0.0.5.sif").exists()
        assert not (dist_dir / "vllm-0.14.0_0.0.5.sif").exists()
        # pytorch should still exist
        assert (dist_dir / "pytorch-2.9.1-cu126_0.0.5.sif").exists()

    def test_rm_all_deletes_all(self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> None:
        """rm --all deletes all local containers."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)

        # Create some containers
        (dist_dir / "vllm-0.14.0_0.0.5.sif").write_text("content")
        (dist_dir / "pytorch-2.9.1-cu126_0.0.5.sif").write_text("content")

        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["rm", "--all", "--yes"])

        assert result.exit_code == 0
        # All containers should be deleted
        assert not (dist_dir / "vllm-0.14.0_0.0.5.sif").exists()
        assert not (dist_dir / "pytorch-2.9.1-cu126_0.0.5.sif").exists()

    def test_rm_no_match_shows_message(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """rm --name with no matches shows appropriate message."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)

        (dist_dir / "pytorch-2.9.1-cu126_0.0.5.sif").write_text("content")

        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["rm", "--name", "nonexistent", "--yes"])

        assert result.exit_code == 0
        assert "No containers" in result.output or "nothing" in result.output.lower()


class TestCache:
    """Tests for cache command."""

    def test_cache_shows_status(self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> None:
        """cache command shows cache status by default."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        cache_dir = scratch / "sifter" / "cache"
        cache_dir.mkdir(parents=True)

        # Create some cached files
        (cache_dir / "g1234567890ab.sif").write_bytes(b"x" * 1024)
        (cache_dir / "g9876543210ab.sif").write_bytes(b"y" * 2048)

        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["cache"])

        assert result.exit_code == 0
        assert "2" in result.output  # 2 cached containers
        assert "cache" in result.output.lower()

    def test_cache_purge_deletes_all(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """cache --purge deletes all cached containers."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        cache_dir = scratch / "sifter" / "cache"
        cache_dir.mkdir(parents=True)

        # Create some cached files
        file1 = cache_dir / "g1234567890ab.sif"
        file2 = cache_dir / "g9876543210ab.sif"
        file1.write_bytes(b"content1")
        file2.write_bytes(b"content2")

        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["cache", "--purge", "--yes"])

        assert result.exit_code == 0
        assert not file1.exists()
        assert not file2.exists()
        assert "Deleted" in result.output

    def test_cache_purge_empty_cache(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """cache --purge on empty cache shows appropriate message."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        cache_dir = scratch / "sifter" / "cache"
        cache_dir.mkdir(parents=True)

        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["cache", "--purge", "--yes"])

        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_cache_purge_preserves_registry(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """cache --purge does not affect registry containers."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        cache_dir = scratch / "sifter" / "cache"
        dist_dir = scratch / "sifter" / "registry"
        cache_dir.mkdir(parents=True)
        dist_dir.mkdir(parents=True)

        # Create cached file and registry file
        cache_file = cache_dir / "g1234567890ab.sif"
        registry_file = dist_dir / "pytorch-2.9.1-cu126_0.0.5.sif"
        cache_file.write_bytes(b"cached")
        registry_file.write_bytes(b"registry")

        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["cache", "--purge", "--yes"])

        assert result.exit_code == 0
        assert not cache_file.exists()
        assert registry_file.exists()  # Registry should be preserved


class TestPull:
    """Tests for pull command."""

    def test_pull_no_remote_configured(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """pull with no remote registry shows error."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.delenv("SIFTER_REGISTRY", raising=False)
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["pull", "test.sif"])

        assert result.exit_code == 1
        assert "No remote registry configured" in result.output


class TestExecuteDagStagedBuilds:
    """Tests for _execute_dag using generic build script with staging."""

    def test_execute_dag_calls_staged_build(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """_execute_dag uses submit_staged_build with correct parameters.

        Builds are staged and submitted using the generic build script.
        """
        from unittest.mock import MagicMock

        from sifter.config import Config
        from sifter.dag import BuildAction, BuildDAG
        from sifter.models import Build, BuildSpec, Step

        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        cache_dir = scratch / "sifter" / "cache"
        logs_dir = scratch / "sifter" / "logs"
        staging_dir = scratch / "sifter" / "staging"
        dist_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        staging_dir.mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        # Create a build spec - output goes to cache as hash.sif
        build = Build(
            name="pytorch-2.9.1-cu126",
            version="0.0.5",
            steps=[Step(path=Path("definitions/pytorch/pytorch.def"), args={})],
        )
        content_hash = "g7354f89abc1"
        build_spec = BuildSpec(
            build=build,
            definition_path=build.steps[0].path,
            args_dict={},
            output_filename=f"{content_hash}.sif",  # Cache filename
            base_image=None,
            content_hash=content_hash,
            should_tag=False,
        )

        # Create DAG with one BUILD node
        dag = BuildDAG()
        dag.add_node(build_spec, BuildAction.BUILD)

        # Mock the SLURM client
        mock_slurm = MagicMock()
        mock_slurm.staging_dir = staging_dir
        mock_slurm.stage_build.return_value = staging_dir / content_hash
        mock_slurm.submit_staged_build.return_value = "12345"

        # Import and call _execute_dag
        from sifter.api import _execute_build_dag
        from sifter.storage import Storage

        config = Config(
            repo_dir=sample_repo,
            dist_dir=dist_dir,
            logs_dir=logs_dir,
            cache_dir=cache_dir,
            staging_dir=staging_dir,
        )
        cache_storage = Storage(local_dir=cache_dir)
        registry_storage = Storage(local_dir=dist_dir)

        _execute_build_dag(dag, config, cache_storage, registry_storage, mock_slurm)

        # Verify stage_build was called
        mock_slurm.stage_build.assert_called_once()
        stage_call = mock_slurm.stage_build.call_args
        assert stage_call.kwargs["content_hash"] == content_hash
        assert stage_call.kwargs["definition_path"] == Path("definitions/pytorch/pytorch.def")

        # Verify submit_staged_build was called with correct parameters
        mock_slurm.submit_staged_build.assert_called_once()
        submit_call = mock_slurm.submit_staged_build.call_args
        assert submit_call.kwargs["output_hash"] == content_hash
        assert submit_call.kwargs["cache_dir"] == cache_dir
        assert submit_call.kwargs["tag"] is None  # Non-release build

    def test_execute_dag_passes_registry_tag_for_release(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """_execute_dag passes registry_tag for release builds."""
        from unittest.mock import MagicMock

        from sifter.config import Config
        from sifter.dag import BuildAction, BuildDAG
        from sifter.models import Build, BuildSpec, Step

        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        cache_dir = scratch / "sifter" / "cache"
        logs_dir = scratch / "sifter" / "logs"
        staging_dir = scratch / "sifter" / "staging"
        dist_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        staging_dir.mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        # Create a build spec with registry_tag (release build)
        build = Build(
            name="pytorch-2.9.1-cu126",
            version="0.0.5",
            steps=[Step(path=Path("definitions/pytorch/pytorch.def"), args={})],
        )
        content_hash = "g7354f89abc1"
        registry_tag = "pytorch-2.9.1-cu126_0.0.5"
        build_spec = BuildSpec(
            build=build,
            definition_path=build.steps[0].path,
            args_dict={},
            output_filename=f"{content_hash}.sif",
            base_image=None,
            content_hash=content_hash,
            should_tag=True,
            registry_tag=registry_tag,
        )

        dag = BuildDAG()
        dag.add_node(build_spec, BuildAction.BUILD)

        mock_slurm = MagicMock()
        mock_slurm.staging_dir = staging_dir
        mock_slurm.stage_build.return_value = staging_dir / content_hash
        mock_slurm.submit_staged_build.return_value = "12345"

        from sifter.api import _execute_build_dag
        from sifter.storage import Storage

        config = Config(
            repo_dir=sample_repo,
            dist_dir=dist_dir,
            logs_dir=logs_dir,
            cache_dir=cache_dir,
            staging_dir=staging_dir,
        )
        cache_storage = Storage(local_dir=cache_dir)
        registry_storage = Storage(local_dir=dist_dir)

        _execute_build_dag(dag, config, cache_storage, registry_storage, mock_slurm)

        # Verify submit_staged_build was called with tag (local tagging only).
        mock_slurm.submit_staged_build.assert_called_once()
        submit_call = mock_slurm.submit_staged_build.call_args
        assert submit_call.kwargs["tag"] == registry_tag
        assert submit_call.kwargs["registry_dir"] == dist_dir
        # A build never publishes: no registry-push command is passed to the job.
        assert "remote_push_registry_cmd" not in submit_call.kwargs


class TestUseLocalTagging:
    """Tests for tagging behavior when container already exists in cache."""

    def test_execute_dag_copies_cached_to_registry_when_tagged(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """When TAG_CACHED with registry_tag, should copy from cache to registry."""
        from unittest.mock import MagicMock

        from sifter.config import Config
        from sifter.dag import BuildAction, BuildDAG
        from sifter.models import Build, BuildSpec, Step

        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        logs_dir = scratch / "sifter" / "logs"
        cache_dir = scratch / "sifter" / "cache"
        staging_dir = scratch / "sifter" / "staging"
        dist_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)
        staging_dir.mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        # Create a cached container file
        content_hash = "g7354f89abc1"
        cached_file = cache_dir / f"{content_hash}.sif"
        cached_file.write_bytes(b"cached container content")

        # Create a build spec with registry_tag
        build = Build(
            name="pytorch-2.9.1-cu126",
            version="0.0.5",
            steps=[Step(path=Path("definitions/pytorch/pytorch.def"), args={})],
        )
        registry_tag = "pytorch-2.9.1-cu126_0.0.5"
        build_spec = BuildSpec(
            build=build,
            definition_path=build.steps[0].path,
            args_dict={},
            output_filename=f"{content_hash}.sif",
            base_image=None,
            content_hash=content_hash,
            should_tag=True,
            registry_tag=registry_tag,
        )

        # Create DAG with TAG_CACHED action
        dag = BuildDAG()
        dag.add_node(build_spec, BuildAction.TAG_CACHED)

        mock_slurm = MagicMock()

        from sifter.api import _execute_build_dag
        from sifter.storage import Storage

        config = Config(
            repo_dir=sample_repo,
            dist_dir=dist_dir,
            logs_dir=logs_dir,
            cache_dir=cache_dir,
            staging_dir=staging_dir,
        )
        cache_storage = Storage(local_dir=cache_dir)
        registry_storage = Storage(local_dir=dist_dir)

        _execute_build_dag(dag, config, cache_storage, registry_storage, mock_slurm)

        # Verify the registry file was created
        registry_file = dist_dir / f"{registry_tag}.sif"
        assert registry_file.exists(), f"Registry file should be created: {registry_file}"
        assert registry_file.read_bytes() == b"cached container content"

        # No SLURM jobs should have been submitted for TAG_CACHED
        mock_slurm.submit_staged_build.assert_not_called()
        mock_slurm.submit_transfer_job.assert_not_called()

    def test_execute_dag_skips_copy_if_registry_already_exists(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """When TAG_CACHED and registry file already exists, should not overwrite."""
        from unittest.mock import MagicMock

        from sifter.config import Config
        from sifter.dag import BuildAction, BuildDAG
        from sifter.models import Build, BuildSpec, Step

        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        logs_dir = scratch / "sifter" / "logs"
        cache_dir = scratch / "sifter" / "cache"
        staging_dir = scratch / "sifter" / "staging"
        dist_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)
        staging_dir.mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        # Create cached container and registry file (both already exist)
        content_hash = "g7354f89abc1"
        cached_file = cache_dir / f"{content_hash}.sif"
        cached_file.write_bytes(b"cached container content")

        registry_tag = "pytorch-2.9.1-cu126_0.0.5"
        registry_file = dist_dir / f"{registry_tag}.sif"
        registry_file.write_bytes(b"existing registry content")

        build_spec = BuildSpec(
            build=Build(
                name="pytorch-2.9.1-cu126",
                version="0.0.5",
                steps=[Step(path=Path("definitions/pytorch/pytorch.def"), args={})],
            ),
            definition_path=Path("definitions/pytorch/pytorch.def"),
            args_dict={},
            output_filename=f"{content_hash}.sif",
            base_image=None,
            content_hash=content_hash,
            should_tag=True,
            registry_tag=registry_tag,
        )

        dag = BuildDAG()
        dag.add_node(build_spec, BuildAction.TAG_CACHED)

        mock_slurm = MagicMock()

        from sifter.api import _execute_build_dag
        from sifter.storage import Storage

        config = Config(
            repo_dir=sample_repo,
            dist_dir=dist_dir,
            logs_dir=logs_dir,
            cache_dir=cache_dir,
            staging_dir=staging_dir,
        )
        cache_storage = Storage(local_dir=cache_dir)
        registry_storage = Storage(local_dir=dist_dir)

        _execute_build_dag(dag, config, cache_storage, registry_storage, mock_slurm)

        # Registry file should still have original content (not overwritten)
        assert registry_file.read_bytes() == b"existing registry content"


class TestBaseImagePropagation:
    """Tests for base image propagation from cache to dependent builds."""

    def test_add_to_dag_uses_cache_filename_for_use_local(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """Base image should use cache filename (hash.sif) for TAG_CACHED actions.

        When pytorch exists in cache with a hash-based filename, vllm's base_image
        should use that hash-based filename.

        We mock the hasher to return a known hash so we can create a matching cache file.
        """
        from unittest.mock import MagicMock

        from sifter.api import _add_build_to_dag
        from sifter.config import Config
        from sifter.dag import BuildAction, BuildDAG
        from sifter.manifest import Manifest
        from sifter.storage import Storage

        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        cache_dir = scratch / "sifter" / "cache"
        logs_dir = scratch / "sifter" / "logs"
        dist_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        # Create a cache file with hash-based naming (new cache format)
        known_hash = "g7354f89abc1"
        cache_filename = f"{known_hash}.sif"
        (cache_dir / cache_filename).write_text("pytorch container")

        # Load manifest from sample_repo (uses new builds: format)
        manifest = Manifest.load(sample_repo / "sifter.yaml")
        cache = Storage(local_dir=cache_dir)
        registry = Storage(local_dir=dist_dir)
        config = Config(
            repo_dir=sample_repo,
            dist_dir=dist_dir,
            logs_dir=logs_dir,
            cache_dir=cache_dir,
            staging_dir=scratch / "sifter" / "staging",
        )

        # Mock hasher to return known hash for pytorch, different hash for vllm
        mock_hasher = MagicMock()

        def get_hash_side_effect(build):
            if build.name == "pytorch-2.9.1-cu126":
                return known_hash  # Matches existing cache file
            return "gabc123def456"  # Different hash for vllm

        mock_hasher.get_hash.side_effect = get_hash_side_effect

        # Create DAG and add pytorch first (will get TAG_CACHED action due to matching hash)
        dag = BuildDAG()

        # First add pytorch - it should get TAG_CACHED because cache file exists
        pytorch_build = manifest.get_build_by_name("pytorch-2.9.1-cu126")
        assert pytorch_build is not None
        _add_build_to_dag(
            dag=dag,
            config=config,
            manifest=manifest,
            cache=cache,
            registry=registry,
            hasher=mock_hasher,
            target=pytorch_build,
            should_tag=False,
            base_override=None,
        )

        # Find the pytorch node
        pytorch_node = None
        for node in dag.nodes.values():
            if node.build_spec.build.name == "pytorch-2.9.1-cu126":
                pytorch_node = node
                break

        assert pytorch_node is not None, "pytorch node should exist"
        assert pytorch_node.action == BuildAction.TAG_CACHED, (
            f"pytorch should have TAG_CACHED action but got {pytorch_node.action}"
        )
        # Cache filename is hash-based
        assert pytorch_node.output_filename == cache_filename, (
            f"pytorch output_filename should be '{cache_filename}' "
            f"but got '{pytorch_node.output_filename}'"
        )

        # Now add vllm - it should pick up pytorch's output_filename as base_image
        vllm_build = manifest.get_build_by_name("vllm-0.14.0")
        assert vllm_build is not None
        _add_build_to_dag(
            dag=dag,
            config=config,
            manifest=manifest,
            cache=cache,
            registry=registry,
            hasher=mock_hasher,
            target=vllm_build,
            should_tag=False,
            base_override=None,
        )

        # Find the vllm node
        vllm_node = None
        for node in dag.nodes.values():
            if node.build_spec.build.name == "vllm-0.14.0":
                vllm_node = node
                break

        assert vllm_node is not None, "vllm node should exist"

        # THE KEY ASSERTION: vllm's base_image should be pytorch's cache filename
        assert vllm_node.build_spec.base_image == cache_filename, (
            f"vllm base_image should be '{cache_filename}' "
            f"(pytorch's cache filename) but got '{vllm_node.build_spec.base_image}'"
        )


class TestBug12TimeAgoDisplay:
    """Tests for Bug 12: Status --all should show time ago for completed jobs."""

    def test_print_job_table_shows_ended_column_for_completed_jobs(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Completed jobs table should show when they ended (time ago).

        The table should include an "Ended" column with relative time
        like "2 hours ago" for completed jobs.
        """
        from datetime import datetime, timedelta
        from unittest.mock import MagicMock

        from sifter.slurm import SLURMClient, SLURMJob

        scratch = sample_repo / "scratch"
        scratch.mkdir()
        logs_dir = scratch / "sifter" / "logs"
        logs_dir.mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))

        # Create a mock SLURM client
        mock_slurm = MagicMock(spec=SLURMClient)
        mock_slurm.logs_dir = logs_dir
        mock_slurm.log_path.return_value = logs_dir / "test_12345.log"

        # Create a completed job that ended 2 hours ago
        completed_job = SLURMJob(
            job_id="12345",
            name="sifter-build-pytorch",
            state="COMPLETED",
            submit_time=datetime.now() - timedelta(hours=5),
            start_time=datetime.now() - timedelta(hours=5),
            end_time=datetime.now() - timedelta(hours=2),
        )

        # Import and call _print_job_table
        from sifter.cli import _print_job_table

        _print_job_table([completed_job], mock_slurm)

        # Capture output
        output = capsys.readouterr().out

        # The output should contain relative time indication
        # humanize.naturaltime returns things like "2 hours ago"
        assert "ago" in output.lower() or "Ended" in output, (
            f"Output should show relative time for completed jobs. Got:\n{output}"
        )


class TestRun:
    """Tests for run command."""

    def test_run_shows_help(self) -> None:
        """run --help shows usage."""
        result = runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0
        assert "container" in result.output.lower()

    def test_run_requires_container(self) -> None:
        """run requires a container argument."""
        result = runner.invoke(app, ["run"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "container" in result.output.lower()

    def test_run_resolves_registry_name(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """run should resolve container names from registry."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        registry_dir = scratch / "sifter" / "registry"
        registry_dir.mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        # Create a container in the registry
        (registry_dir / "myapp_testing.sif").write_bytes(b"fake container")

        # Run with just the name (no .sif)
        result = runner.invoke(app, ["run", "myapp_testing", "--help"])

        # Should not error about container not found
        # (--help exits before actually running the container)
        assert "not found" not in result.output.lower()

    def test_run_resolves_registry_name_with_sif(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """run should resolve container.sif names from registry."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        registry_dir = scratch / "sifter" / "registry"
        registry_dir.mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        # Create a container in the registry
        (registry_dir / "myapp_testing.sif").write_bytes(b"fake container")

        # Run with .sif suffix - should still check registry
        result = runner.invoke(app, ["run", "myapp_testing.sif", "--help"])

        # Should not error about container not found
        assert "not found" not in result.output.lower()

    def test_run_not_found_shows_helpful_message(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """run with nonexistent container shows helpful error."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["run", "nonexistent", "echo", "hello"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()
        assert "registry" in result.output.lower() or "sifter ls" in result.output.lower()

    def test_run_requires_command_or_interactive(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """run without command or -i shows error."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["run", "myapp_1.0.0"])

        assert result.exit_code == 1
        assert "specify a command or use -i" in result.output.lower()

    def test_run_slurm_submits_job(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """run --slurm submits a SLURM job."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        registry_dir = scratch / "sifter" / "registry"
        registry_dir.mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        # Create a container in the registry
        (registry_dir / "myapp_testing.sif").write_bytes(b"fake container")

        with patch("sifter.api.SLURMClient") as mock_slurm:
            mock_slurm.return_value.submit_run_job.return_value = "12345"

            result = runner.invoke(app, ["run", "myapp_testing", "--slurm", "python", "train.py"])

        # Should show submitted job ID
        assert result.exit_code == 0
        assert "12345" in result.output
        assert "Submitted" in result.output or "submitted" in result.output.lower()

        # Should have called submit_run_job
        mock_slurm.return_value.submit_run_job.assert_called_once()

    def test_run_slurm_shows_status_hint(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """run --slurm shows how to check status."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        registry_dir = scratch / "sifter" / "registry"
        registry_dir.mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        (registry_dir / "myapp_testing.sif").write_bytes(b"fake container")

        with patch("sifter.api.SLURMClient") as mock_slurm:
            mock_slurm.return_value.submit_run_job.return_value = "12345"

            result = runner.invoke(app, ["run", "myapp_testing", "--slurm", "python", "train.py"])

        # Should show hints for status and logs
        assert "sifter status" in result.output
        assert "sifter logs" in result.output


class TestLsSimplifiedGrouping:
    """Tests for simplified ls grouping (1-level instead of 2-level)."""

    def test_ls_groups_by_full_name_not_prefix(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """pytorch-2.9.1-cu126 and pytorch-2.8.0-cu118 are separate groups.

        With the new 1-level grouping, we no longer have a parent 'pytorch' group.
        Each unique name becomes its own top-level group.
        """
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)

        # Create containers with similar prefixes but different names
        (dist_dir / "pytorch-2.9.1-cu126_0.0.5.sif").write_text("content")
        (dist_dir / "pytorch-2.8.0-cu118_0.0.4.sif").write_text("content")

        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(app, ["ls"])

        assert result.exit_code == 0
        # Both should appear as separate groups
        assert "pytorch-2.9.1-cu126" in result.output
        assert "pytorch-2.8.0-cu118" in result.output

    def test_ls_groups_tags_under_name(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """myapp_0.0.5, myapp_testing, myapp_stable all under 'myapp'."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)

        # Create containers with same name but different tags
        (dist_dir / "myapp_0.0.5.sif").write_text("content")
        (dist_dir / "myapp_testing.sif").write_text("content")
        (dist_dir / "myapp_stable.sif").write_text("content")

        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(app, ["ls"])

        assert result.exit_code == 0
        # All three should appear under 'myapp' group
        assert "myapp" in result.output
        assert "0.0.5" in result.output
        assert "testing" in result.output
        assert "stable" in result.output

    def test_ls_name_flag_replaces_container(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """--name flag filters by name (replaces --container)."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)

        (dist_dir / "vllm-0.14.0_0.0.5.sif").write_text("content")
        (dist_dir / "pytorch-2.9.1-cu126_0.0.5.sif").write_text("content")

        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(app, ["ls", "--name", "vllm-0.14.0"])

        assert result.exit_code == 0
        assert "vllm-0.14.0" in result.output
        # pytorch should not be shown
        lines = result.output.split("\n")
        assert not any("pytorch" in line for line in lines)

    def test_ls_tag_flag_filters_by_tag(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """--tag flag filters by tag."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)

        (dist_dir / "myapp_0.0.5.sif").write_text("content")
        (dist_dir / "myapp_testing.sif").write_text("content")

        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(app, ["ls", "--tag", "testing"])

        assert result.exit_code == 0
        assert "testing" in result.output
        # 0.0.5 tag should not be shown
        assert "0.0.5" not in result.output


class TestBuildTagFlag:
    """Tests for build --tag flag (Goal 1.3)."""

    def test_build_with_tag_dry_run(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """build --tag produces tagged output filename.

        When using --tag myapp:1.0.0, the output should be myapp:1.0.0.sif
        (Docker-style naming), not using the manifest's variant naming.
        """
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        (scratch / "sifter" / "cache").mkdir(parents=True)
        (scratch / "sifter" / "staging").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(
                app, ["build", "pytorch-2.9.1-cu126", "--tag", "myapp_1.0.0", "--dry-run"]
            )

        assert result.exit_code == 0
        # Should show the custom tag in output
        assert "myapp_1.0.0" in result.output or "myapp" in result.output


class TestBuildStepsFlag:
    """Tests for build --steps flag (Goal 4 - decouple from manifest)."""

    def test_build_def_file_auto_adhoc(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """build with .def file argument auto-detects ad-hoc mode.

        Users should be able to run:
        sifter build pytorch.def --tag testing

        This is shorthand for:
        sifter build --steps pytorch.def --tag testing
        """
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        (scratch / "sifter" / "cache").mkdir(parents=True)
        (scratch / "sifter" / "staging").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(tmp_path)

        # Create def file (no sifter.yaml)
        (tmp_path / "pytorch.def").write_text("Bootstrap: docker\nFrom: ubuntu:22.04")

        with patch("sifter.api.SLURMClient"):
            # No --steps flag needed - auto-detected from .def extension
            result = runner.invoke(app, ["build", "pytorch.def", "--tag", "testing", "--dry-run"])

        # Should work without manifest, using ad-hoc mode
        assert result.exit_code == 0, (
            f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
        )
        assert "pytorch_testing" in result.output  # Name derived from file stem

    def test_build_def_file_path_auto_adhoc(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """build with path/to/file.def auto-detects ad-hoc mode."""
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        (scratch / "sifter" / "cache").mkdir(parents=True)
        (scratch / "sifter" / "staging").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(tmp_path)

        # Create def file in subdirectory
        defs_dir = tmp_path / "definitions" / "base"
        defs_dir.mkdir(parents=True)
        (defs_dir / "base.def").write_text("Bootstrap: docker\nFrom: ubuntu:22.04")

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(
                app, ["build", "definitions/base/base.def", "--tag", "testing", "--dry-run"]
            )

        assert result.exit_code == 0, (
            f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
        )
        assert "base_testing" in result.output

    def test_build_with_steps_no_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """build --steps works without a manifest file.

        Users should be able to run:
        sifter build --steps base.def,app.def --tag myapp

        This allows ad-hoc builds without setting up sifter.yaml.
        """
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        (scratch / "sifter" / "cache").mkdir(parents=True)
        (scratch / "sifter" / "staging").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(tmp_path)

        # Create def files (no sifter.yaml)
        (tmp_path / "base.def").write_text("Bootstrap: docker\nFrom: ubuntu:22.04")
        (tmp_path / "app.def").write_text("Bootstrap: localimage\nFrom: {{ BASE_IMAGE }}")

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(
                app, ["build", "--steps", "base.def,app.def", "--tag", "testing", "--dry-run"]
            )

        # Should work without manifest
        assert result.exit_code == 0, (
            f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
        )

    def test_build_steps_with_custom_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--steps with --ad-hoc-name produces correct output: name_tag.sif.

        e.g., --steps base.def --ad-hoc-name danbase --tag testing → danbase_testing.sif
        """
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        (scratch / "sifter" / "cache").mkdir(parents=True)
        (scratch / "sifter" / "staging").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(tmp_path)

        # Create def file
        (tmp_path / "base.def").write_text("Bootstrap: docker\nFrom: ubuntu:22.04")

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(
                app,
                [
                    "build",
                    "--steps",
                    "base.def",
                    "--ad-hoc-name",
                    "danbase",
                    "--tag",
                    "testing",
                    "--dry-run",
                ],
            )

        assert result.exit_code == 0
        # Should show danbase_testing as output
        assert "danbase_testing" in result.output

    def test_build_steps_name_defaults_to_file_stem(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--steps defaults name to file stem (not folder name).

        e.g., --steps mydir/base.def --tag testing → base_testing.sif
        """
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        (scratch / "sifter" / "cache").mkdir(parents=True)
        (scratch / "sifter" / "staging").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(tmp_path)

        # Create def file in subdirectory
        def_dir = tmp_path / "mydir"
        def_dir.mkdir()
        (def_dir / "base.def").write_text("Bootstrap: docker\nFrom: ubuntu:22.04")

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(
                app, ["build", "--steps", "mydir/base.def", "--tag", "testing", "--dry-run"]
            )

        assert result.exit_code == 0
        # Should show base_testing as output (file stem + tag, not folder name)
        assert "base_testing" in result.output

    def test_build_steps_current_dir_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--steps with file in current directory uses file stem.

        e.g., --steps container.def --tag testing → container_testing.sif
        """
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        (scratch / "sifter" / "cache").mkdir(parents=True)
        (scratch / "sifter" / "staging").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(tmp_path)

        # Create def file in current directory (no subdirectory)
        (tmp_path / "container.def").write_text("Bootstrap: docker\nFrom: ubuntu:22.04")

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(
                app, ["build", "--steps", "container.def", "--tag", "testing", "--dry-run"]
            )

        assert result.exit_code == 0
        # Should show container_testing as output
        assert "container_testing" in result.output


class TestBuildStepsBaseImage:
    """Tests for --base flag in ad-hoc builds."""

    def test_adhoc_build_base_checks_registry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--base should check registry as well as cache."""
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        registry_dir = scratch / "sifter" / "registry"
        registry_dir.mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        (scratch / "sifter" / "cache").mkdir(parents=True)
        (scratch / "sifter" / "staging").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(tmp_path)

        # Create def file
        (tmp_path / "app.def").write_text("Bootstrap: localimage\nFrom: {{ BASE_IMAGE }}")

        # Create base image in REGISTRY (not cache)
        (registry_dir / "mybase_stable.sif").write_bytes(b"base image content")

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(
                app,
                [
                    "build",
                    "--steps",
                    "app.def",
                    "--base",
                    "mybase_stable",
                    "--tag",
                    "testing",
                    "--dry-run",
                ],
            )

        # Should succeed - base found in registry
        assert result.exit_code == 0, (
            f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
        )

    def test_adhoc_build_base_not_found_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--base with nonexistent image should show error."""
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        (scratch / "sifter" / "cache").mkdir(parents=True)
        (scratch / "sifter" / "staging").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(tmp_path)

        # Create def file
        (tmp_path / "app.def").write_text("Bootstrap: localimage\nFrom: {{ BASE_IMAGE }}")

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(
                app,
                [
                    "build",
                    "--steps",
                    "app.def",
                    "--base",
                    "nonexistent",
                    "--tag",
                    "testing",
                    "--dry-run",
                ],
            )

        # Should fail with helpful error
        assert result.exit_code == 1
        assert "not found" in result.output.lower()


class TestBuildStepsUsesDAG:
    """Tests for ad-hoc builds using the same DAG infrastructure as manifest builds."""

    def test_adhoc_build_shows_tree_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Ad-hoc builds should show tree output like manifest builds."""
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        (scratch / "sifter" / "cache").mkdir(parents=True)
        (scratch / "sifter" / "staging").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(tmp_path)

        (tmp_path / "base.def").write_text("Bootstrap: docker\nFrom: ubuntu:22.04")

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(
                app, ["build", "--steps", "base.def", "--tag", "testing", "--dry-run"]
            )

        assert result.exit_code == 0
        # Should show BUILD action like manifest builds
        assert "BUILD" in result.output

    def test_adhoc_build_checks_local_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Ad-hoc builds should check local cache and skip if exists."""
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        cache_dir = scratch / "sifter" / "cache"
        cache_dir.mkdir(parents=True)
        (scratch / "sifter" / "staging").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(tmp_path)

        # Create def file
        (tmp_path / "base.def").write_text("Bootstrap: docker\nFrom: ubuntu:22.04")

        # Pre-compute the hash and create a cached file
        from sifter.cache import hash_definition

        content_hash = hash_definition(tmp_path / "base.def")
        (cache_dir / f"{content_hash}.sif").write_bytes(b"cached content")

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(
                app, ["build", "--steps", "base.def", "--tag", "testing", "--dry-run"]
            )

        assert result.exit_code == 0
        # Should show TAG_CACHED since file exists in local cache
        assert "TAG_CACHED" in result.output
        # Should NOT show BUILD
        assert "BUILD" not in result.output

    def test_adhoc_multi_step_shows_dependency_tree(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Multi-step ad-hoc builds should show dependency tree."""
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        (scratch / "sifter" / "cache").mkdir(parents=True)
        (scratch / "sifter" / "staging").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(tmp_path)

        # Create two def files
        (tmp_path / "base.def").write_text("Bootstrap: docker\nFrom: ubuntu:22.04")
        (tmp_path / "app.def").write_text("Bootstrap: localimage\nFrom: {{ BASE_IMAGE }}")

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(
                app, ["build", "--steps", "base.def,app.def", "--tag", "testing", "--dry-run"]
            )

        assert result.exit_code == 0
        # Should show both step names in output (not filenames, step names are stems)
        assert "base" in result.output
        assert "app" in result.output
        # Both should be BUILD actions
        assert result.output.count("BUILD") == 2


class TestBuildManifestFlag:
    """Tests for build --manifest flag (Goal 4 - optional manifest path)."""

    def test_build_with_manifest_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """build --manifest allows using a different manifest location.

        Users can specify --manifest path/to/sifter.yaml to use
        a manifest in a non-default location.
        """
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "sifter" / "registry").mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)
        (scratch / "sifter" / "cache").mkdir(parents=True)
        (scratch / "sifter" / "staging").mkdir(parents=True)
        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(tmp_path)

        # Create a custom manifest in a non-default location
        custom_manifest_dir = tmp_path / "custom"
        custom_manifest_dir.mkdir()
        custom_manifest = custom_manifest_dir / "sifter.yaml"
        custom_manifest.write_text("""
builds:
  test-variant_1.0.0:
    steps:
      - path: definitions/test/test.def
        args: {}
oci:
  verify: false
""")

        # Create the definition directory RELATIVE to the manifest's parent
        # (definition paths in manifest are relative to repo_dir, which is manifest.parent)
        def_dir = custom_manifest_dir / "definitions" / "test"
        def_dir.mkdir(parents=True)
        (def_dir / "test.def").write_text("Bootstrap: docker\nFrom: ubuntu:22.04")
        (def_dir / "build.sh").write_text("#!/bin/bash\necho build")

        with patch("sifter.api.SLURMClient"):
            result = runner.invoke(
                app,
                ["build", "--manifest", str(custom_manifest), "test-variant", "--dry-run"],
            )

        # Should be able to find the variant from the custom manifest
        # If this fails with "not found", it means the manifest flag isn't working
        assert result.exit_code == 0, (
            f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"
        )
        assert "ignoring 'verify'" in result.output
        assert str(custom_manifest) in result.output.replace("\n", "")


class TestLatestCommand:
    """Tests for 'sifter latest' CLI command."""

    def test_latest_returns_path(self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> None:
        """sifter latest prints the path to the latest matching container."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)

        (dist_dir / "vllm-ext-0.16.0_0.1.0.sif").touch()
        (dist_dir / "vllm-ext-0.16.0_0.2.0.sif").touch()

        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["latest", "--prefix", "vllm-ext-"])

        assert result.exit_code == 0
        assert "vllm-ext-0.16.0_0.2.0.sif" in result.output

    def test_latest_bare_matches_any_container(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """Bare `sifter latest` (no --prefix) matches stable containers of any name."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)

        (dist_dir / "pytorch-2.5.0_0.1.0.sif").touch()
        (dist_dir / "pytorch-2.5.0_0.2.0.sif").touch()

        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["latest"])

        assert result.exit_code == 0
        assert "pytorch-2.5.0_0.2.0.sif" in result.output

    def test_latest_no_match_exits_1(
        self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path
    ) -> None:
        """sifter latest exits with code 1 when no containers match."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)

        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["latest", "--prefix", "nonexistent-"])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_latest_with_version(self, monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> None:
        """sifter latest --version narrows to a specific upstream version."""
        scratch = sample_repo / "scratch"
        scratch.mkdir()
        dist_dir = scratch / "sifter" / "registry"
        dist_dir.mkdir(parents=True)
        (scratch / "sifter" / "logs").mkdir(parents=True)

        (dist_dir / "vllm-ext-0.16.0_0.1.0.sif").touch()
        (dist_dir / "vllm-ext-0.16.1rc0_0.1.0.sif").touch()
        (dist_dir / "vllm-ext-0.16.1rc0_0.2.0.sif").touch()

        monkeypatch.setenv("SCRATCHDIR", str(scratch))
        monkeypatch.chdir(sample_repo)

        result = runner.invoke(app, ["latest", "--prefix", "vllm-ext-", "--version", "0.16.1rc0"])

        assert result.exit_code == 0
        assert "vllm-ext-0.16.1rc0_0.2.0.sif" in result.output
