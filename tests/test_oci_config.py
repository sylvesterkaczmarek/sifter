"""Tests for OCI/ECR registry configuration (SIFTER_REGISTRIES + signing gates)."""

from __future__ import annotations

import os
import shlex
import urllib.error
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from sifter.config import (
    ConfigError,
    OCIConfig,
    SifterConfig,
    resolve_oci_config,
    resolve_registry_config,
)
from sifter.oci import ECRRegistry, OCIRegistry
from sifter.signing import SigningError


def _resolved(tmp_path: Path) -> OCIConfig:
    cfg = resolve_oci_config(tmp_path).config
    assert cfg is not None
    return cfg


def _notices(repo_dir: Path) -> str:
    """Everything a resolution has to tell the operator, as one searchable block."""
    return "\n".join(resolve_oci_config(repo_dir).notices)


def _write_user_config(body: str) -> None:
    """Write the trusted user config (conftest points XDG_CONFIG_HOME at a tmp dir)."""
    path = Path(os.environ["XDG_CONFIG_HOME"]) / "sifter" / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def _write_project_config(repo_dir: Path, body: str) -> None:
    """Write the untrusted project config that arrives with a cloned repo."""
    (repo_dir / "sifter.yaml").write_text(body)


def _config_in(monkeypatch: pytest.MonkeyPatch, repo_dir: Path) -> SifterConfig:
    """Resolve a full config as a shell sitting in *repo_dir* would."""
    for var in ("SIFTER_DIST", "SIFTER_LOGS", "SIFTER_CACHE", "SIFTER_STAGING"):
        monkeypatch.setenv(var, str(repo_dir / var.lower()))
    monkeypatch.chdir(repo_dir)
    return SifterConfig.from_env()


def test_no_registries_env_yields_no_oci_config(tmp_path: Path) -> None:
    assert resolve_oci_config(tmp_path).config is None


def test_empty_registries_is_rejected_at_construction() -> None:
    # The model must self-validate: an empty chain fails fast here, rather than
    # surfacing later as a bare IndexError when the push target is selected.
    with pytest.raises(ValidationError):
        OCIConfig(registries=[])


@pytest.mark.parametrize("timeout", [0, -1, 86_400])
def test_a_probe_timeout_no_probe_would_survive_is_rejected(timeout: int) -> None:
    # A project file may set this one, so an unbounded value is a repo's licence to
    # hang every sifter run in its clone, and a zero one to fail them all.
    with pytest.raises(ValidationError):
        OCIConfig(registries=["reg/x"], probe_timeout=timeout)


def test_a_boolean_probe_timeout_is_rejected_rather_than_read_as_one_second() -> None:
    # `probe_timeout: true` is a mistake, not a one-second timeout — and a repo may
    # set this key, so coercing it silently hands one a way to fail every probe.
    with pytest.raises(ValidationError):
        OCIConfig(registries=["reg/x"], probe_timeout=True)


def test_plain_http_registry_is_rejected_at_construction() -> None:
    # http:// is otherwise accepted-then-unusable: the HEAD probe and every
    # oras/cosign command assume https. Reject it up front with a clear error.
    with pytest.raises(ValidationError, match="plain-HTTP"):
        OCIConfig(registries=["http://localhost:5000/team"])


@pytest.mark.parametrize(
    "registry",
    [
        "--registry-config=/repo/oras.json",
        " --registry-config=/repo/oras.json",
        "oras://--registry-config=/repo/oras.json",
        "https://--registry-config=/repo/oras.json",
        "-x/team",
        "oras://-x/team",
    ],
)
def test_registry_that_would_become_a_command_line_flag_is_rejected(registry: str) -> None:
    # A repo may name its own registries, and the name lands in oras's argv. One
    # starting with a dash stops being a positional and starts being a flag —
    # including when the dash is hidden behind a scheme that is stripped later.
    with pytest.raises(ValidationError, match="cannot start with"):
        OCIConfig(registries=[registry])


def test_a_registry_name_that_is_nothing_but_a_scheme_is_rejected() -> None:
    # `oras://` normalises away to nothing, and an empty name reaches oras as a ref
    # with no host — a push that silently goes nowhere rather than a refusal.
    with pytest.raises(ValidationError, match="empty"):
        OCIConfig(registries=["oras://"])


@pytest.mark.parametrize(
    ("registry", "message"),
    [
        ("http://localhost:5000/team", "plain-HTTP"),
        ("oras://", "empty"),
        ("-x/team", "cannot start with"),
        ("reg.example.com /team", "whitespace"),
    ],
)
def test_a_backend_built_directly_refuses_what_the_config_model_refuses(
    registry: str, message: str
) -> None:
    # A programmatic caller builds a backend without going through the config model,
    # so the name has to be checked where it is used and not only where it is written.
    with pytest.raises(ValueError, match=message):
        OCIRegistry(registry)


def test_https_registry_is_accepted_with_the_scheme_normalised_away() -> None:
    # The https:// sibling must still pass (guards against over-broad matching),
    # and it is stored in the form the backends and the config summary use.
    cfg = OCIConfig(registries=["https://reg.example.com/team/"])
    assert cfg.registries == ["reg.example.com/team"]


@pytest.mark.parametrize(
    "registry",
    ["oras://reg.example.com/team/", "https://reg.example.com/team", "reg.example.com/team"],
)
def test_a_registry_is_stored_as_the_backend_will_use_it(registry: str) -> None:
    # The stored value is what gets reported to users and what reaches oras; if
    # the two disagree, one of them is a lie.
    cfg = OCIConfig(registries=[registry])
    assert cfg.build_chain()[0].url == cfg.registries[0]


@pytest.mark.parametrize("registry", [" reg.example.com/team ", "\treg.example.com/team\n"])
def test_a_registry_name_is_stored_without_the_whitespace_around_it(registry: str) -> None:
    # YAML and comma-separated env lists both leave padding behind, and a padded
    # name reaches oras as a different name than the one printed back.
    cfg = OCIConfig(registries=[registry])
    assert cfg.registries == ["reg.example.com/team"]
    assert cfg.build_chain()[0].url == cfg.registries[0]


@pytest.mark.parametrize("registry", ["reg.example.com /team", "reg.example.com\t/team"])
def test_a_registry_name_split_by_whitespace_is_rejected(registry: str) -> None:
    with pytest.raises(ValidationError, match="whitespace"):
        OCIConfig(registries=[registry])


@pytest.mark.parametrize("registry", ["oras://https://reg.example.com/team", "https://oras://x/y"])
def test_a_registry_behind_two_schemes_is_rejected(registry: str) -> None:
    # Config load and backend construction both normalise, so a name that survives
    # normalisation has to survive it again unchanged. Stacked schemes are the one
    # input where that failed: the config kept a name the backend then refused.
    with pytest.raises(ValidationError, match="scheme"):
        OCIConfig(registries=[registry])


def test_registries_split_on_comma_and_whitespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SIFTER_REGISTRIES", "reg.a/team, reg.b/team reg.c/team")
    assert _resolved(tmp_path).registries == ["reg.a/team", "reg.b/team", "reg.c/team"]


def test_the_registries_beyond_the_first_are_reported_as_unused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Silently consulting one of three is worse than not accepting three at all.
    monkeypatch.setenv("SIFTER_REGISTRIES", "reg.a/team reg.b/team reg.c/team")
    notices = _notices(tmp_path)
    assert "reg.b/team" in notices and "reg.c/team" in notices


def test_a_single_registry_is_not_reported_as_partly_unused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SIFTER_REGISTRIES", "reg.only/team")
    assert resolve_oci_config(tmp_path).notices == ()


def test_resolution_never_speaks_through_the_warnings_module(tmp_path: Path) -> None:
    # Filters set far from here decide whether a warning is silent or fatal, and a
    # repo picks the registry list, so a repo would decide which of the two it got.
    _write_project_config(tmp_path, "oci:\n  registries: [reg.a/x, reg.b/x]\n  oras_bin: ./p\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        notices = "\n".join(resolve_oci_config(tmp_path).notices)
    assert "oras_bin" in notices
    assert "reg.b/x" in notices


def test_ecr_hosts_become_ecr_registries_others_oci(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(
        "SIFTER_REGISTRIES",
        "123456789012.dkr.ecr.eu-west-2.amazonaws.com/org/team,reg.example.com/team",
    )
    chain = _resolved(tmp_path).build_chain()
    assert isinstance(chain[0], ECRRegistry)
    assert isinstance(chain[1], OCIRegistry) and not isinstance(chain[1], ECRRegistry)


class TestGatesFailSafeOn:
    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", "Off"])
    def test_explicit_falsy_disables(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str
    ) -> None:
        monkeypatch.setenv("SIFTER_REGISTRIES", "reg/team")
        monkeypatch.setenv("SIFTER_SIGN", value)
        assert _resolved(tmp_path).sign is False

    @pytest.mark.parametrize("value", ["", "1", "true", "yes", "garbage", "maybe"])
    def test_unset_empty_or_unrecognised_stays_on(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str
    ) -> None:
        monkeypatch.setenv("SIFTER_REGISTRIES", "reg/team")
        if value:
            monkeypatch.setenv("SIFTER_VERIFY", value)
        assert _resolved(tmp_path).verify is True


def test_signing_keys_and_oras_bin_flow_into_the_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SIFTER_REGISTRIES", "reg.example.com/team")
    monkeypatch.setenv("SIFTER_SIGNING_KEY", "awskms:///alias/example-key")
    monkeypatch.setenv("SIFTER_VERIFY_KEY", "awskms:///alias/example-key")
    monkeypatch.setenv("SIFTER_ORAS_BIN", "/opt/oras")
    reg = _resolved(tmp_path).build_chain()[0]
    assert reg.signing_key == "awskms:///alias/example-key"
    assert reg.verify_key == "awskms:///alias/example-key"
    assert reg.oras_bin == "/opt/oras"


def test_legacy_s3_registry_env_does_not_produce_oci(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(
        "SIFTER_REGISTRY",
        '{"registry_type": "s3", "s3_bucket": "b", "s3_prefix": "p"}',
    )
    # SIFTER_REGISTRY (singular, legacy S3) must not be read by the OCI path.
    assert resolve_oci_config(tmp_path).config is None


class TestConfigWiring:
    """SifterConfig factory behaviour when an OCI chain is configured."""

    def _config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        registries: str = "reg.a/team,reg.b/team",
    ) -> SifterConfig:
        monkeypatch.setenv("SIFTER_REGISTRIES", registries)
        return _config_in(monkeypatch, tmp_path)

    def test_remote_cache_is_none_so_content_hash_files_never_hit_oci(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Cache files are g<hash>.sif (no <name>_<tag>); with OCI the cache is
        # local-only, so build planning never asks OCI to map a cache filename.
        assert self._config(monkeypatch, tmp_path).create_remote_cache() is None

    def test_push_destination_is_the_first_chain_entry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        remote = self._config(monkeypatch, tmp_path).create_remote_registry()
        assert remote is not None
        assert remote.uri("base_1.sif") == "reg.a/team/base:1"

    def test_remote_description_reflects_oci_not_not_configured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert self._config(monkeypatch, tmp_path).remote_description == "oras://reg.a/team"


def test_sifter_oci_env_cannot_smuggle_a_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `oci` is a private attr, not an env-parsed field, so untrusted env can't
    # inject a whole config (bypassing the gate normalisation) via SIFTER_OCI.
    for var in ("SIFTER_DIST", "SIFTER_LOGS", "SIFTER_CACHE", "SIFTER_STAGING"):
        monkeypatch.setenv(var, str(tmp_path / var.lower()))
    monkeypatch.setenv("SIFTER_OCI", '{"registries": ["smuggled/x"], "sign": false}')
    monkeypatch.chdir(tmp_path)
    assert SifterConfig.from_env().oci is None


class TestYamlGateEdge:
    def test_empty_user_gate_value_does_not_disable_signing(self, tmp_path: Path) -> None:
        # A null/empty gate in YAML must leave the fail-safe-ON default intact.
        _write_user_config("oci:\n  registries: [reg/x]\n  sign:\n")
        assert _resolved(tmp_path).sign is True

    @pytest.mark.parametrize("value", ["0", "1", "[]"])
    def test_a_gate_that_is_not_a_boolean_is_a_configuration_error(
        self, tmp_path: Path, value: str
    ) -> None:
        # `sign: 0` is written by someone who means off. YAML makes it an integer,
        # which no longer resembles a gate value — better to say so than to quietly
        # do the opposite of what a trusted config asked for.
        _write_user_config(f"oci:\n  registries: [reg/x]\n  sign: {value}\n")
        with pytest.raises(ConfigError, match="sign"):
            resolve_oci_config(tmp_path)

    def test_user_yaml_false_disables_and_env_overrides_yaml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_user_config("oci:\n  registries: [reg/x]\n  verify: false\n")
        assert _resolved(tmp_path).verify is False
        monkeypatch.setenv("SIFTER_VERIFY", "1")
        assert _resolved(tmp_path).verify is True


class TestProjectConfigCannotWeakenSecurity:
    """``sifter.yaml`` arrives with the cloned repo, so it is attacker-controlled
    whenever the repo is: it must not pick the binary sifter runs, nor open the gates."""

    def test_project_oras_bin_is_not_executed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The presence probe execs oras_bin, so a repo-supplied path is RCE on
        # anyone who runs sifter in a clone of that repo.
        attacker_ran = tmp_path / "attacker-ran"
        trusted_ran = tmp_path / "trusted-ran"
        hostile = tmp_path / "hostile-oras"
        hostile.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(attacker_ran))}\n")
        hostile.chmod(0o755)
        bindir = tmp_path / "bin"
        bindir.mkdir()
        (bindir / "oras").write_text(f"#!/bin/sh\ntouch {shlex.quote(str(trusted_ran))}\n")
        (bindir / "oras").chmod(0o755)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
        _write_project_config(
            tmp_path, f"oci:\n  registries: [reg.example.com/team]\n  oras_bin: {hostile}\n"
        )

        registry = _resolved(tmp_path).build_chain()[0]
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
            registry.exists("base_1.sif")

        assert not attacker_ran.exists()
        assert trusted_ran.exists()

    @pytest.mark.parametrize("gate", ["sign", "verify"])
    def test_project_yaml_cannot_disable_a_gate(self, tmp_path: Path, gate: str) -> None:
        _write_project_config(tmp_path, f"oci:\n  registries: [reg/x]\n  {gate}: false\n")
        assert getattr(_resolved(tmp_path), gate) is True

    @pytest.mark.parametrize(
        ("setting", "value"),
        [
            ("sign", "false"),
            ("verify", "false"),
            ("oras_bin", "/tmp/hostile"),
            ("verify_key", "/repo/attacker.pub"),
            ("signing_key", "/repo/attacker.key"),
        ],
    )
    def test_ignored_project_security_settings_are_reported(
        self, tmp_path: Path, setting: str, value: str
    ) -> None:
        # Silently dropping a setting the repo asked for would look like a bug to
        # whoever wrote it; say which key was refused and where to set it instead.
        _write_project_config(tmp_path, f"oci:\n  registries: [reg/x]\n  {setting}: {value}\n")
        resolved = resolve_oci_config(tmp_path)
        assert resolved.refused_project_keys == (setting,)
        assert setting in _notices(tmp_path)

    def test_a_refusal_is_reported_even_with_no_registry_configured(self, tmp_path: Path) -> None:
        # Nothing was at risk without a registry, but the repo still asked; hearing
        # nothing back is indistinguishable from having been obeyed.
        _write_project_config(tmp_path, "oci:\n  oras_bin: ./payload\n")
        assert resolve_oci_config(tmp_path).config is None
        assert "oras_bin" in _notices(tmp_path)

    def test_a_refusal_names_the_file_that_asked_and_where_to_ask_instead(
        self, tmp_path: Path
    ) -> None:
        # A notice that does not say which file was refused, or where the setting
        # legitimately lives, leaves the reader with a mystery rather than a fix.
        _write_project_config(tmp_path, "oci:\n  registries: [reg/x]\n  oras_bin: ./payload\n")
        notices = _notices(tmp_path)
        assert str(tmp_path / "sifter.yaml") in notices
        assert "SIFTER_" in notices

    def test_an_emptied_registries_is_reported_as_empty_not_as_forbidden(
        self, tmp_path: Path
    ) -> None:
        # `registries` is project-settable, so telling the reader a repo may not set
        # it sends them hunting a permission problem that does not exist.
        _write_project_config(tmp_path, "oci:\n  registries: []\n")
        notices = _notices(tmp_path)
        assert "registries" in notices
        assert "empty" in notices

    def test_a_project_file_asking_for_nothing_is_not_refused_anything(
        self, tmp_path: Path
    ) -> None:
        _write_project_config(tmp_path, "oci:\n  registries: [reg/x]\n")
        resolved = resolve_oci_config(tmp_path)
        assert resolved.refused_project_keys == ()
        assert resolved.notices == ()

    def test_project_cannot_replace_a_user_verify_key(self, tmp_path: Path) -> None:
        _write_user_config("oci:\n  registries: [reg/x]\n  verify_key: /keys/trusted.pub\n")
        _write_project_config(tmp_path, "oci:\n  verify_key: /repo/attacker.pub\n")
        assert _resolved(tmp_path).verify_key == "/keys/trusted.pub"

    def test_a_repo_supplied_key_cannot_stand_in_for_a_missing_user_key(
        self, tmp_path: Path
    ) -> None:
        # With no trusted key, verification must stay fail-closed. Accepting the
        # repo's key instead turns that hard failure into a green verification
        # against a key the repo's author holds.
        _write_project_config(
            tmp_path, "oci:\n  registries: [reg/x]\n  verify_key: /repo/attacker.pub\n"
        )
        registry = _resolved(tmp_path).build_chain()[0]
        with pytest.raises(SigningError, match="verify key"):
            registry.generate_pull_command("base_1.sif", tmp_path / "base_1.sif")

    def test_a_repo_cannot_choose_the_key_its_own_images_are_signed_with(
        self, tmp_path: Path
    ) -> None:
        _write_project_config(
            tmp_path, "oci:\n  registries: [reg/x]\n  signing_key: /repo/attacker.key\n"
        )
        registry = _resolved(tmp_path).build_chain()[0]
        with pytest.raises(SigningError, match="signing key"):
            registry.generate_push_command(tmp_path / "base_1.sif", "base_1.sif")

    def test_project_registries_still_apply(self, tmp_path: Path) -> None:
        # The repo legitimately owns where its images live; only security-relevant
        # settings are refused.
        _write_project_config(tmp_path, "oci:\n  registries: [reg.project.example/team]\n")
        assert _resolved(tmp_path).registries == ["reg.project.example/team"]

    def test_a_project_may_set_the_probe_timeout(self, tmp_path: Path) -> None:
        # Nothing about a bounded network timeout is a security decision, so the
        # allowlist is not just `registries` by accident.
        _write_project_config(tmp_path, "oci:\n  registries: [reg/x]\n  probe_timeout: 45\n")
        assert _resolved(tmp_path).probe_timeout == 45

    def test_a_project_registry_beats_one_in_the_user_config(self, tmp_path: Path) -> None:
        # The allowlist, not the merge order, is what keeps a repo away from the
        # security settings. Once a key is on it the repo really does own it —
        # otherwise anyone with a default registry in their own config silently
        # publishes every project to it.
        _write_user_config("oci:\n  registries: [reg.mine.example/me]\n  probe_timeout: 5\n")
        _write_project_config(
            tmp_path, "oci:\n  registries: [reg.project.example/team]\n  probe_timeout: 45\n"
        )
        resolved = _resolved(tmp_path)
        assert resolved.registries == ["reg.project.example/team"]
        assert resolved.probe_timeout == 45

    def test_the_user_config_supplies_what_the_project_leaves_out(self, tmp_path: Path) -> None:
        _write_user_config("oci:\n  registries: [reg.mine.example/me]\n  probe_timeout: 5\n")
        _write_project_config(tmp_path, "oci:\n  oras_bin: ./payload\n")
        resolved = _resolved(tmp_path)
        assert resolved.registries == ["reg.mine.example/me"]
        assert resolved.probe_timeout == 5

    def test_a_refused_key_yaml_does_not_read_as_a_string_is_reported(self, tmp_path: Path) -> None:
        # YAML reads a bare `on:` as a boolean, so the refused keys are not all
        # strings — naming them must not be what breaks, or a repo gets to turn the
        # refusal itself into a crash.
        _write_project_config(
            tmp_path, "oci:\n  registries: [reg/x]\n  on: 1\n  oras_bin: ./payload\n"
        )
        resolved = resolve_oci_config(tmp_path)
        assert "oras_bin" in _notices(tmp_path)
        assert resolved.config is not None and resolved.config.registries == ["reg/x"]

    def test_a_single_refused_name_cannot_fill_the_notice(self, tmp_path: Path) -> None:
        # One key is enough to scroll the notice away: YAML's explicit-key syntax
        # lifts the length limit a plain `name:` is held to.
        name = "z" * 5000
        _write_project_config(tmp_path, f"oci:\n  oras_bin: ./payload\n  ? {name}\n  : 1\n")

        notices = _notices(tmp_path)

        assert "oras_bin" in notices
        assert name not in notices
        assert len(notices) < len(name)

    def test_the_binary_a_project_asked_to_choose_is_named_among_a_flood_of_keys(
        self, tmp_path: Path
    ) -> None:
        # The notice is bounded, so a repo can pad the refused list with harmless
        # keys until the one that matters falls off the end of it.
        padding = "".join(f"  aaa{n}: 1\n" for n in range(400))
        _write_project_config(tmp_path, f"oci:\n  oras_bin: ./payload\n{padding}")
        assert "oras_bin" in _notices(tmp_path)


class TestAProjectCannotDeleteTheOCIConfigItInherits:
    """Being allowed to set ``registries`` is not permission to erase it.

    An emptied value drops the signed OCI path entirely, and what waits below it
    is the ``registry:`` section — fully project-settable, and unable to verify a
    signature. A repo may say where its own images live, not that there are none.
    """

    @pytest.mark.parametrize("emptied", ["[]", "{}", "''", "0", ""])
    def test_an_emptied_project_registries_leaves_the_user_registries_standing(
        self, tmp_path: Path, emptied: str
    ) -> None:
        _write_user_config("oci:\n  registries: [reg.mine.example/me]\n")
        _write_project_config(tmp_path, f"oci:\n  registries: {emptied}\n")
        assert _resolved(tmp_path).registries == ["reg.mine.example/me"]

    def test_the_signed_remote_survives_a_project_offering_an_unsigned_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Emptying `registries` and supplying a `registry:` section is one move:
        # it silences OCI and lands the caller on a backend that cannot verify.
        _write_user_config("oci:\n  registries: [reg.mine.example/me]\n")
        _write_project_config(
            tmp_path,
            "oci:\n  registries: []\n"
            f"registry:\n  registry_type: filesystem\n  registry_path: {tmp_path / 'attacker'}\n",
        )
        config = _config_in(monkeypatch, tmp_path)
        assert isinstance(config.create_remote_registry(), OCIRegistry)
        assert config.remote_description == "oras://reg.mine.example/me"


class TestAnUnusableProjectFileIsReportedNotRaised:
    """A cloned repo decides what is in its ``sifter.yaml``.

    Every way that file can be unusable therefore has to surface as a
    configuration error, or a repo's author picks which sifter commands end in a
    traceback inside their clone.
    """

    def test_a_file_that_is_not_utf8_is_a_configuration_error(self, tmp_path: Path) -> None:
        (tmp_path / "sifter.yaml").write_bytes(b"oci:\n  registries: [\xff\xfe.example.com]\n")
        with pytest.raises(ConfigError, match=r"sifter\.yaml"):
            resolve_oci_config(tmp_path)

    def test_registries_given_as_a_bare_string_is_a_configuration_error(
        self, tmp_path: Path
    ) -> None:
        # It must be reported as the wrong type and nothing else: measuring the
        # string instead once announced one registry per character.
        _write_project_config(tmp_path, "oci:\n  registries: reg.example.com/team\n")
        with pytest.raises(ConfigError, match="invalid oci configuration"):
            resolve_oci_config(tmp_path)

    def test_a_file_nested_too_deeply_to_parse_is_a_configuration_error(
        self, tmp_path: Path
    ) -> None:
        # The parser recurses per nesting level, and a repo chooses how many levels
        # its file has, so exhausting the stack must not be one of its options.
        _write_project_config(tmp_path, "oci:\n  registries: " + "[" * 1000 + "]" * 1000)
        with pytest.raises(ConfigError, match=r"sifter\.yaml"):
            resolve_oci_config(tmp_path)

    @pytest.mark.parametrize("body", ["- one\n- two\n", "just a string\n"])
    def test_a_file_that_is_not_a_mapping_is_a_configuration_error(
        self, tmp_path: Path, body: str
    ) -> None:
        # Treating it as "no settings" would read identically to a file that really
        # has none, so nobody finds out why their settings stopped applying.
        _write_project_config(tmp_path, body)
        with pytest.raises(ConfigError, match=r"sifter\.yaml"):
            resolve_oci_config(tmp_path)

    def test_an_oci_section_that_is_not_a_mapping_is_a_configuration_error(
        self, tmp_path: Path
    ) -> None:
        _write_project_config(tmp_path, "oci: registries\n")
        with pytest.raises(ConfigError, match="oci"):
            resolve_oci_config(tmp_path)

    def test_an_empty_section_is_simply_empty(self, tmp_path: Path) -> None:
        # `oci:` with nothing under it is a real thing to write, not a broken file.
        _write_project_config(tmp_path, "oci:\n")
        assert resolve_oci_config(tmp_path).config is None

    def test_an_unknown_registry_backend_is_a_configuration_error(self, tmp_path: Path) -> None:
        # The whole `registry:` section is project-settable by design, so a repo can
        # reach this validator with anything it likes.
        _write_project_config(tmp_path, "registry:\n  registry_type: bogus\n")
        with pytest.raises(ConfigError, match="invalid registry configuration"):
            resolve_registry_config(tmp_path)
