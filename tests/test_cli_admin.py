from __future__ import annotations

import json
import os
import types

import bn.cli
import pytest

from _cli_helpers import *  # noqa: F401,F403


@pytest.fixture(autouse=True)
def _isolate_omp_skill_destination(monkeypatch, tmp_path):
    config_root = tmp_path / "absent-omp-config"
    agent_dir = config_root / "agent"
    monkeypatch.setattr(bn.cli, "omp_config_root", lambda: config_root, raising=False)
    monkeypatch.setattr(bn.cli, "omp_agent_dir", lambda: agent_dir, raising=False)
    monkeypatch.setattr(
        bn.cli, "omp_skills_dir", lambda: agent_dir / "skills", raising=False
    )


def test_skills_source_dir_prefers_repo_then_falls_back_to_prefix(monkeypatch, tmp_path):
    # #83: editable checkout uses repo skills/; a wheel install (no repo skills/)
    # falls back to the install prefix where the data files land.
    import sys as _sys

    import bn.paths as paths

    (tmp_path / "skills").mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    assert paths.skills_source_dir() == tmp_path / "skills"

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: empty)  # no skills/ here
    assert paths.skills_source_dir() == paths.Path(_sys.prefix)


def test_plugin_source_dir_prefers_repo_then_falls_back_to_installed_module(monkeypatch, tmp_path):
    # #83/#406: editable checkout uses repo src/<name>; a wheel install resolves
    # the bridge packaged into site-packages via find_spec.
    import bn.paths as paths

    repo_plugin = tmp_path / "src" / paths.PLUGIN_NAME
    repo_plugin.mkdir(parents=True)
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    assert paths.plugin_source_dir() == repo_plugin

    empty = tmp_path / "empty"
    empty.mkdir()
    installed = tmp_path / "site" / paths.PLUGIN_NAME
    installed.mkdir(parents=True)
    fake_spec = types.SimpleNamespace(origin=str(installed / "__init__.py"))
    monkeypatch.setattr(paths, "repo_root", lambda: empty)
    monkeypatch.setattr(paths.importlib.util, "find_spec", lambda name: fake_spec)
    assert paths.plugin_source_dir() == installed


def test_plugin_install_copy_mode(tmp_path):
    destination = tmp_path / "plugin-copy"
    rc = bn.cli.main(
        [
            "plugin",
            "install",
            "--mode",
            "copy",
            "--dest",
            str(destination),
        ]
    )
    assert rc == 0
    assert (destination / "bridge.py").exists()


def test_skill_install_copy_mode(tmp_path):
    destination = tmp_path / "skill-copy"
    rc = bn.cli.main(
        [
            "skill",
            "install",
            "--mode",
            "copy",
            "--dest",
            str(destination),
        ]
    )
    assert rc == 0
    assert (destination / "bn" / "SKILL.md").exists()
    assert (destination / "bn" / "agents" / "openai.yaml").exists()
    assert (destination / "bn-re" / "SKILL.md").exists()
    assert (destination / "bn-vr" / "SKILL.md").exists()
    assert (destination / "bn-kernel" / "SKILL.md").exists()


def test_skill_install_copy_omits_python_bytecode(tmp_path, monkeypatch):
    source_root = tmp_path / "bundled-skills"
    skill = source_root / "sample"
    cache = skill / "__pycache__"
    cache.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: sample\n---\n")
    (skill / "adapter.py").write_text("VALUE = 1\n")
    (cache / "adapter.cpython-314.pyc").write_bytes(b"stale bytecode")
    (skill / "adapter.pyo").write_bytes(b"stale bytecode")
    destination = tmp_path / "installed"
    monkeypatch.setattr(bn.cli, "skills_source_dir", lambda: source_root)

    rc = bn.cli.main(
        ["skill", "install", "--mode", "copy", "--dest", str(destination)]
    )

    assert rc == 0
    assert (destination / "sample" / "adapter.py").exists()
    assert not (destination / "sample" / "__pycache__").exists()
    assert not (destination / "sample" / "adapter.pyo").exists()


def test_skill_install_defaults_to_claude_only_without_codex_home(tmp_path, monkeypatch):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    codex_root = codex_home / "skills"
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)
    monkeypatch.setattr(bn.cli, "codex_skills_dir", lambda: codex_root)

    rc = bn.cli.main(["skill", "install", "--mode", "copy"])

    assert rc == 0
    assert (claude_root / "bn" / "SKILL.md").exists()
    assert not codex_root.exists()


def test_skill_install_defaults_to_claude_and_codex_when_codex_home_exists(tmp_path, monkeypatch):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    codex_root = codex_home / "skills"
    codex_home.mkdir()
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)
    monkeypatch.setattr(bn.cli, "codex_skills_dir", lambda: codex_root)

    rc = bn.cli.main(["skill", "install", "--mode", "copy"])

    assert rc == 0
    assert (claude_root / "bn" / "SKILL.md").exists()
    assert (codex_root / "bn" / "SKILL.md").exists()
    assert (codex_root / "bn-re" / "SKILL.md").exists()
    assert (codex_root / "bn-vr" / "SKILL.md").exists()


def test_skill_install_defaults_skip_existing_destinations(tmp_path, monkeypatch):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    codex_root = codex_home / "skills"
    codex_home.mkdir()
    (claude_root / "bn").mkdir(parents=True)
    (claude_root / "bn-re").mkdir()
    (claude_root / "bn-vr").mkdir()
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)
    monkeypatch.setattr(bn.cli, "codex_skills_dir", lambda: codex_root)

    rc = bn.cli.main(["skill", "install", "--mode", "copy"])

    assert rc == 0
    assert (codex_root / "bn" / "SKILL.md").exists()
    assert (codex_root / "bn-re" / "SKILL.md").exists()
    assert (codex_root / "bn-vr" / "SKILL.md").exists()


def test_skill_install_default_output_is_text(tmp_path, monkeypatch, capsys):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)

    rc = bn.cli.main(["skill", "install", "--mode", "copy"])

    assert rc == 0
    output = capsys.readouterr().out
    assert output.startswith("Installed skills (copy):\n")
    assert "- " + str(claude_root / "bn") in output
    assert '"installed"' not in output


def test_skill_install_json_output_remains_available(tmp_path, monkeypatch, capsys):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)

    rc = bn.cli.main(["skill", "install", "--mode", "copy", "--format", "json"])

    assert rc == 0
    output = capsys.readouterr().out
    assert '"installed":true' in output          # compact json (#215)
    assert '"installed_destinations"' in output


def test_skill_install_all_skipped_reports_installed_false(tmp_path, monkeypatch, capsys):
    # #620(b): when every destination already exists and is skipped, nothing
    # was actually installed -- the result must say so instead of always
    # claiming success.
    source_root = tmp_path / "bundled-skills"
    skill = source_root / "sample"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: sample\n---\n")
    monkeypatch.setattr(bn.cli, "skills_source_dir", lambda: source_root)

    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    (claude_root / "sample").mkdir(parents=True)
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)

    rc = bn.cli.main(["skill", "install", "--mode", "copy", "--format", "json"])

    assert rc == 0
    output = capsys.readouterr().out
    assert '"installed":false' in output
    assert '"installed_destinations":[]' in output
    assert str(claude_root / "sample") in output  # skipped_destinations explains why


def test_skill_install_custom_dest_still_fails_when_destination_exists(tmp_path):
    destination = tmp_path / "skill-copy"
    (destination / "bn").mkdir(parents=True)

    rc = bn.cli.main(["skill", "install", "--mode", "copy", "--dest", str(destination)])

    assert rc == 2



def test_omp_path_resolution_follows_profiles_and_agent_override(monkeypatch, tmp_path):
    import bn.paths as paths

    monkeypatch.setenv("HOME", str(tmp_path))
    for name in (
        "PI_CONFIG_DIR",
        "OMP_PROFILE",
        "PI_PROFILE",
        "PI_CODING_AGENT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    assert paths.omp_config_root() == tmp_path / ".omp"
    assert paths.omp_agent_dir() == tmp_path / ".omp" / "agent"
    assert paths.omp_skills_dir() == tmp_path / ".omp" / "agent" / "skills"

    monkeypatch.setenv("PI_CONFIG_DIR", ".custom-omp")
    monkeypatch.setenv("PI_PROFILE", "fallback")
    monkeypatch.setenv("OMP_PROFILE", "  preferred  ")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", "~/ignored-agent")
    assert paths.omp_config_root() == tmp_path / ".custom-omp"
    assert paths.omp_agent_dir() == (
        tmp_path / ".custom-omp" / "profiles" / "preferred" / "agent"
    )

    monkeypatch.setenv("OMP_PROFILE", "  ")
    assert paths.omp_agent_dir() == tmp_path / "ignored-agent"

    monkeypatch.setenv("OMP_PROFILE", "default")
    assert paths.omp_agent_dir() == tmp_path / "ignored-agent"


def test_skill_install_adds_default_omp_profile_when_root_exists(
    tmp_path, monkeypatch
):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    omp_config = tmp_path / ".omp"
    omp_agent = omp_config / "agent"
    omp_root = omp_agent / "skills"
    omp_config.mkdir()
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)
    monkeypatch.setattr(bn.cli, "omp_config_root", lambda: omp_config)
    monkeypatch.setattr(bn.cli, "omp_agent_dir", lambda: omp_agent)
    monkeypatch.setattr(bn.cli, "omp_skills_dir", lambda: omp_root)

    rc = bn.cli.main(["skill", "install", "--mode", "copy"])

    assert rc == 0
    assert (omp_root / "bn-kernel" / "SKILL.md").exists()
    assert (omp_root / "bn" / "SKILL.md").exists()


def test_skill_install_adds_omp_when_only_agent_directory_exists(
    tmp_path, monkeypatch
):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    omp_config = tmp_path / "missing-config"
    omp_agent = tmp_path / "custom-agent"
    omp_root = omp_agent / "skills"
    omp_agent.mkdir()
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)
    monkeypatch.setattr(bn.cli, "omp_config_root", lambda: omp_config)
    monkeypatch.setattr(bn.cli, "omp_agent_dir", lambda: omp_agent)
    monkeypatch.setattr(bn.cli, "omp_skills_dir", lambda: omp_root)

    assert bn.cli.main(["skill", "install", "--mode", "copy"]) == 0
    assert (omp_root / "bn-kernel" / "SKILL.md").exists()


def test_skill_install_explicit_destination_never_writes_omp(
    tmp_path, monkeypatch
):
    explicit = tmp_path / "explicit"
    omp_config = tmp_path / ".omp"
    omp_agent = omp_config / "agent"
    omp_root = omp_agent / "skills"
    omp_config.mkdir()
    monkeypatch.setattr(bn.cli, "omp_config_root", lambda: omp_config)
    monkeypatch.setattr(bn.cli, "omp_agent_dir", lambda: omp_agent)
    monkeypatch.setattr(bn.cli, "omp_skills_dir", lambda: omp_root)

    rc = bn.cli.main(
        ["skill", "install", "--mode", "copy", "--dest", str(explicit)]
    )

    assert rc == 0
    assert (explicit / "bn-kernel" / "SKILL.md").exists()
    assert not omp_root.exists()

def test_version_flag_prints_version(capsys):
    # `bn --version` is a real affordance, not "unrecognized arguments" (#49).
    from bn.version import VERSION
    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["--version"])
    assert exc.value.code == 0
    out, _ = capsys.readouterr()
    assert out.strip() == f"bn {VERSION}"


def test_version_is_single_sourced_from_pyproject():
    # The version literal lives only in pyproject.toml; version.py derives it,
    # so CLI/bridge never drift and a bump touches one file (#82).
    import tomllib
    from pathlib import Path

    import bn.version

    repo_root = Path(bn.version.__file__).resolve().parents[2]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    canonical = pyproject["project"]["version"]

    assert canonical == "0.20.0"  # the reset target
    assert bn.version.VERSION == canonical
    # No stray literal: the old number must be gone from the version module.
    assert "0.12.2" not in Path(bn.version.__file__).read_text(encoding="utf-8")


def test_resolve_version_falls_back_to_dist_metadata(monkeypatch):
    # When pyproject is unreachable (installed wheel), VERSION resolves from the
    # installed distribution metadata rather than crashing (#82).
    import tomllib

    import bn.version

    monkeypatch.setattr(
        tomllib, "loads",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no pyproject here")),
    )
    # Editable install metadata reports the same canonical version.
    assert bn.version._resolve_version() == bn.version.VERSION


def test_bridge_plugin_json_carries_no_version_literal():
    # plugin.json is BN-manager metadata that cannot import Python; it must not
    # duplicate the version (single-source invariant, #82).
    import json
    from pathlib import Path

    import bn.version

    repo_root = Path(bn.version.__file__).resolve().parents[2]
    manifest = json.loads(
        (repo_root / "src" / "bn_agent_bridge" / "plugin.json").read_text(encoding="utf-8")
    )
    assert "version" not in manifest


def test_doctor_reports_stale_loaded_plugin(monkeypatch, tmp_path, capsys):
    install_dir = tmp_path / "install"
    source_dir = tmp_path / "source"
    install_dir.mkdir()
    source_dir.mkdir()
    (install_dir / "bridge.py").write_text("print('new build')\n", encoding="utf-8")
    (source_dir / "bridge.py").write_text("print('new build')\n", encoding="utf-8")

    fake_instance = type(
        "FakeInstance",
        (),
        {
            "pid": 123,
            "socket_path": tmp_path / "bridge.sock",
            "plugin_version": "0.4.0",
            "started_at": "2026-03-09T00:00:00+00:00",
        },
    )()

    monkeypatch.setattr(bn.cli, "list_instances", lambda: [fake_instance])
    monkeypatch.setattr(bn.cli, "plugin_install_dir", lambda: install_dir)
    monkeypatch.setattr(bn.cli, "plugin_source_dir", lambda: source_dir)
    monkeypatch.setattr(
        bn.cli,
        "_send_request_to_instance",
        lambda instance, op, params=None, target=None: {
            "ok": True,
            "result": {
                "plugin_name": "bn_agent_bridge",
                "plugin_version": "0.4.0",
                "plugin_build_id": "oldbuild123456",
                "pid": 123,
                "socket_path": str(tmp_path / "bridge.sock"),
                "targets": [],
            },
        },
    )

    rc = bn.cli.main(["doctor", "--format", "json"])

    # #620(c): stale_plugin_code makes the instance unhealthy -> nonzero exit.
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["cli_version"] == bn.cli.VERSION
    assert payload["plugin_install_build_id"]
    assert payload["instances"][0]["stale_plugin_version"] is True
    assert payload["instances"][0]["stale_plugin_code"] is True


def test_doctor_flags_stale_engine(monkeypatch, tmp_path, capsys):
    # #161: doctor reports a per-instance engine fingerprint and flags
    # stale_engine when the loaded engine package diverges from on-disk.
    install_dir = tmp_path / "install"
    source_dir = tmp_path / "source"
    install_dir.mkdir()
    source_dir.mkdir()
    for d in (install_dir, source_dir):
        (d / "bridge.py").write_text("print('bridge')\n", encoding="utf-8")
        (d / "taint_engine.py").write_text("X = 1\n", encoding="utf-8")

    fake_instance = type("FakeInstance", (), {
        "instance_id": "abc123", "pid": 123, "socket_path": tmp_path / "bridge.sock",
        "plugin_version": bn.cli.VERSION, "started_at": "2026-03-09T00:00:00+00:00",
    })()
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [fake_instance])
    monkeypatch.setattr(bn.cli, "plugin_install_dir", lambda: install_dir)
    monkeypatch.setattr(bn.cli, "plugin_source_dir", lambda: source_dir)
    monkeypatch.setattr(
        bn.cli, "_send_request_to_instance",
        lambda instance, op, params=None, target=None: {"ok": True, "result": {
            "plugin_name": "bn_agent_bridge", "plugin_version": bn.cli.VERSION,
            "plugin_build_id": bn.cli.build_id_for_file(install_dir / "bridge.py"),
            # Loaded engine fingerprint differs from on-disk -> stale_engine.
            "engine_build_id": "staleengine00",
            "pid": 123, "socket_path": str(tmp_path / "bridge.sock"), "targets": [],
        }},
    )

    rc = bn.cli.main(["doctor", "--format", "json"])
    # #620(c): stale_engine makes the instance unhealthy -> nonzero exit.
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    inst = payload["instances"][0]
    assert inst["stale_engine"] is True
    assert inst["stale_plugin_code"] is False  # bridge.py matches; only the engine is stale
    assert payload["engine_install_build_id"]


def test_session_restart_respawns_and_reloads_targets(monkeypatch, capsys):
    from bn.transport import BridgeInstance
    old = type("FakeInstance", (), {
        "instance_id": "keep-me", "pid": 500,
        "socket_path": __import__("pathlib").Path("/tmp/old.sock"),
        # Restart reads the registry payload to recover private project
        # associations; this instance owned none.
        "meta": {},
    })()
    new = BridgeInstance(
        pid=999, socket_path=__import__("pathlib").Path("/tmp/new.sock"),
        registry_path=__import__("pathlib").Path("/tmp/new.json"),
        plugin_name="bn_agent_bridge", plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z", meta={}, instance_id="keep-me",
    )
    calls = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        calls.append((op, instance_id, params))
        return {"ok": True, "result": {"path": (params or {}).get("path")}}

    monkeypatch.setattr(bn.cli, "list_instances", lambda **kw: [old])
    monkeypatch.setattr(bn.cli, "find_lifecycle_instance", lambda target: old)
    monkeypatch.setattr(bn.cli, "instance_selector", lambda i: getattr(i, "instance_id", ""))
    monkeypatch.setattr(
        bn.cli, "_send_request_to_instance",
        lambda instance, op, params=None, target=None: {"ok": True, "result": [
            {"filename": "/fw/svc_a", "analysis_state": "full"},
        ]},
    )
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, timeout=5.0: True)
    spawned = {}
    def fake_spawn(instance_id=None):
        spawned["id"] = instance_id
        return new
    monkeypatch.setattr(bn.cli, "spawn_instance", fake_spawn)
    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "restart", "keep-me", "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["restarted"] is True
    assert payload["instance_id"] == "keep-me"
    assert spawned["id"] == "keep-me"             # respawned under the same id
    ops = [c[0] for c in calls]
    assert "shutdown" in ops and "load_binary" in ops   # stopped, then reloaded the target
    assert any(c[0] == "load_binary" and (c[2] or {}).get("path") == "/fw/svc_a" for c in calls)


def test_session_restart_warns_when_list_targets_fails_but_still_restarts(monkeypatch, capsys):
    # #620(a): a `list_targets` failure while capturing the pre-restart state
    # must not be silently swallowed -- it is reported on stderr, and the
    # restart still proceeds (just with no targets to reload).
    from bn.transport import BridgeInstance
    old = type("FakeInstance", (), {
        "instance_id": "keep-me", "pid": 500,
        "socket_path": __import__("pathlib").Path("/tmp/old.sock"),
        "meta": {},
    })()
    new = BridgeInstance(
        pid=999, socket_path=__import__("pathlib").Path("/tmp/new.sock"),
        registry_path=__import__("pathlib").Path("/tmp/new.json"),
        plugin_name="bn_agent_bridge", plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z", meta={}, instance_id="keep-me",
    )

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {}}

    monkeypatch.setattr(bn.cli, "list_instances", lambda **kw: [old])
    monkeypatch.setattr(bn.cli, "find_lifecycle_instance", lambda target: old)
    monkeypatch.setattr(bn.cli, "instance_selector", lambda i: getattr(i, "instance_id", ""))

    def failing_send_to_instance(instance, op, params=None, target=None):
        raise OSError("connection refused")

    monkeypatch.setattr(bn.cli, "_send_request_to_instance", failing_send_to_instance)
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, timeout=5.0: True)
    spawned = {}
    def fake_spawn(instance_id=None):
        spawned["id"] = instance_id
        return new
    monkeypatch.setattr(bn.cli, "spawn_instance", fake_spawn)
    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "restart", "keep-me", "--format", "json"])

    assert rc == 0
    assert spawned["id"] == "keep-me"   # restart still happened
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["restarted"] is True
    assert payload["loaded"] == []      # nothing captured to reload
    assert "could not enumerate targets to reload after restart" in captured.err


def test_doctor_text_marks_healthy_instance_ok(monkeypatch, tmp_path, capsys):
    install_dir = tmp_path / "install"
    source_dir = tmp_path / "source"
    install_dir.mkdir()
    source_dir.mkdir()
    (install_dir / "bridge.py").write_text("print('new build')\n", encoding="utf-8")
    (source_dir / "bridge.py").write_text("print('new build')\n", encoding="utf-8")

    fake_instance = type(
        "FakeInstance",
        (),
        {
            "pid": 123,
            "socket_path": tmp_path / "bridge.sock",
            "plugin_version": bn.cli.VERSION,
            "started_at": "2026-03-09T00:00:00+00:00",
        },
    )()

    monkeypatch.setattr(bn.cli, "list_instances", lambda: [fake_instance])
    monkeypatch.setattr(bn.cli, "plugin_install_dir", lambda: install_dir)
    monkeypatch.setattr(bn.cli, "plugin_source_dir", lambda: source_dir)
    # Genuinely healthy: the loaded plugin_build_id matches what's on disk (the
    # doctor exit code now reflects staleness, so a mismatched hardcoded id
    # here would make this "healthy" fixture unhealthy).
    matching_build_id = bn.cli.build_id_for_file(install_dir / "bridge.py")
    monkeypatch.setattr(
        bn.cli,
        "_send_request_to_instance",
        lambda instance, op, params=None, target=None: {
            "ok": True,
            "result": {
                "plugin_name": "bn_agent_bridge",
                "plugin_version": bn.cli.VERSION,
                "plugin_build_id": matching_build_id,
                "pid": 123,
                "socket_path": str(tmp_path / "bridge.sock"),
                "targets": [],
            },
        },
    )

    rc = bn.cli.main(["doctor"])

    assert rc == 0
    output = capsys.readouterr().out
    assert f"pid=123 plugin={bn.cli.VERSION} status=ok" in output
    assert "status=error" not in output


def test_doctor_json_carries_reachable_and_status(monkeypatch, tmp_path, capsys):
    """doctor --format json must carry the same health signal the text mode shows
    (reachable / status), so a scripted JSON health check can read it directly
    instead of re-deriving reachability from the absence of doctor.error. (L16)"""
    install_dir = tmp_path / "install"
    source_dir = tmp_path / "source"
    install_dir.mkdir()
    source_dir.mkdir()
    (install_dir / "bridge.py").write_text("print('b')\n", encoding="utf-8")
    (source_dir / "bridge.py").write_text("print('b')\n", encoding="utf-8")

    def _inst(pid, name):
        return type("FakeInstance", (), {
            "pid": pid, "socket_path": tmp_path / f"{name}.sock",
            "plugin_version": bn.cli.VERSION, "started_at": "2026-03-09T00:00:00+00:00",
            "instance_id": name,
        })()

    ok_inst, bad_inst = _inst(1, "ok"), _inst(2, "bad")
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [ok_inst, bad_inst])
    monkeypatch.setattr(bn.cli, "plugin_install_dir", lambda: install_dir)
    monkeypatch.setattr(bn.cli, "plugin_source_dir", lambda: source_dir)

    def fake_send(instance, op, params=None, target=None):
        if instance is ok_inst:
            return {"ok": True, "result": {
                "plugin_version": bn.cli.VERSION, "plugin_build_id": "b", "targets": []}}
        raise OSError("connection refused")

    monkeypatch.setattr(bn.cli, "_send_request_to_instance", fake_send)

    rc = bn.cli.main(["doctor", "--format", "json"])
    # #620(c): an unreachable instance makes the overall doctor exit nonzero,
    # so a scripted health check can trust the exit code alone.
    assert rc == 1
    data = json.loads(capsys.readouterr().out)
    by_pid = {i["pid"]: i for i in data["instances"]}
    assert by_pid[1]["reachable"] is True and by_pid[1]["status"] == "ok"
    assert by_pid[2]["reachable"] is False and by_pid[2]["status"] == "error"


def test_instance_flag_passed_to_send_request(monkeypatch, capsys):
    captured_instance_ids = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured_instance_ids.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "test.bndb"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["--instance", "abc123", "function", "list"])

    assert "abc123" in captured_instance_ids


def test_instance_flag_on_subcommand(monkeypatch, capsys):
    captured_instance_ids = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured_instance_ids.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "test.bndb"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["function", "list", "--instance", "abc123"])

    assert "abc123" in captured_instance_ids


def test_instance_flag_from_env(monkeypatch, capsys):
    captured_instance_ids = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured_instance_ids.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "test.bndb"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setenv("BN_INSTANCE", "env_inst")

    bn.cli.main(["function", "list"])

    assert "env_inst" in captured_instance_ids


def test_session_list_shows_instances(monkeypatch, capsys):
    from bn.transport import BridgeInstance

    fake_instances = [
        BridgeInstance(
            pid=111,
            socket_path=__import__("pathlib").Path("/tmp/a.sock"),
            registry_path=__import__("pathlib").Path("/tmp/a.json"),
            plugin_name="bn_agent_bridge",
            plugin_version="0.1.0",
            started_at="2026-01-01T00:00:00Z",
            meta={"project_roots": ["/workspace/project-a"]},
            instance_id="aaaa1111",
        ),
        BridgeInstance(
            pid=222,
            socket_path=__import__("pathlib").Path("/tmp/b.sock"),
            registry_path=__import__("pathlib").Path("/tmp/b.json"),
            plugin_name="bn_agent_bridge",
            plugin_version="0.1.0",
            started_at="2026-01-01T00:01:00Z",
            meta={},
            instance_id="bbbb2222",
        ),
    ]
    monkeypatch.setattr(bn.cli, "list_instances", lambda: fake_instances)

    rc = bn.cli.main(["session", "list", "--format", "json"])

    assert rc == 0
    stdout = capsys.readouterr().out
    parsed = json.loads(stdout)
    # #358: session/instance list now uses the {kind, items} envelope (with
    # total_rss_mb kept as an extra field).
    assert parsed["kind"] == "instances"
    assert len(parsed["items"]) == 2
    assert parsed["items"][0]["selector"] == "aaaa1111"
    assert parsed["items"][0]["instance_id"] == "aaaa1111"
    assert parsed["items"][0]["project_roots"] == ["/workspace/project-a"]
    assert parsed["items"][1]["instance_id"] == "bbbb2222"
    assert "rss_mb" in parsed["items"][0]
    assert "total_rss_mb" in parsed

    rc = bn.cli.main(
        ["session", "list", "-i", "bbbb2222", "--format", "json"]
    )
    assert rc == 0
    filtered = json.loads(capsys.readouterr().out)
    assert [item["instance_id"] for item in filtered["items"]] == ["bbbb2222"]


def test_session_status_polls_detached_load_job(monkeypatch, capsys):
    calls = []

    def fake_send_request(op, **kwargs):
        calls.append((op, kwargs))
        return {
            "ok": True,
            "result": {
                "kind": "load_jobs",
                "items": [
                    {
                        "job_id": "job123",
                        "state": "running",
                        "path": "/tmp/sample.bndb",
                    }
                ],
                "count": 1,
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(
        [
            "session",
            "status",
            "job123",
            "-i",
            "worker",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    assert calls[0][0] == "load_status"
    assert calls[0][1]["params"] == {"job_id": "job123"}
    payload = json.loads(capsys.readouterr().out)
    assert payload["items"][0]["state"] == "running"


def test_session_status_job_passes_through_machine_fields(monkeypatch, capsys):
    # The CLI must not re-derive or flatten the bridge's job verdict; whatever
    # top-level state/terminal/succeeded contract the bridge publishes is what
    # `--format json` hands the polling agent.
    def fake_send_request(op, **kwargs):
        return {
            "ok": True,
            "result": {
                "kind": "load_job",
                "job_id": "job123",
                "state": "complete",
                "terminal": True,
                "succeeded": True,
                "job": {"job_id": "job123", "state": "complete", "path": "/tmp/s.bndb"},
                "items": [{"job_id": "job123", "state": "complete", "path": "/tmp/s.bndb"}],
                "count": 1,
                "status_command": "bn -i worker session status job123",
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(
        ["session", "status", "job123", "-i", "worker", "--format", "json"]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["job_id"] == "job123"
    assert payload["state"] == "complete"
    assert payload["terminal"] is True
    assert payload["succeeded"] is True
    assert payload["job"]["path"] == "/tmp/s.bndb"


def test_session_status_unknown_job_fails_loudly(monkeypatch, capsys):
    # A bad job id is an error, not an empty poll result: silently returning
    # "no jobs" would make a status loop spin until its own deadline.
    def fake_send_request(op, **kwargs):
        raise bn.transport.BridgeError("Unknown load job: nope")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "status", "nope", "-i", "worker"])

    assert rc != 0
    assert "Unknown load job: nope" in capsys.readouterr().err




@pytest.mark.parametrize("flag", ["--instance", "--instance-id"])
def test_session_stop_accepts_instance_flag(monkeypatch, capsys, flag):
    # #456: after threading --instance through every command, cleanup naturally
    # tries `session stop --instance <id>`; accept it as an alias for the positional.
    seen = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                          instance_id=None, spawn_missing_named=False):
        seen["op"] = op
        seen["instance_id"] = instance_id
        return {"ok": True, "result": {}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [])

    rc = bn.cli.main(["session", "stop", flag, "abc123", "--format", "json"])
    assert rc == 0
    assert seen["op"] == "shutdown" and seen["instance_id"] == "abc123"


def test_session_stop_requires_an_instance_id(monkeypatch, capsys):
    # Neither positional nor --instance -> a clean error, not a crash.
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [])
    rc = bn.cli.main(["session", "stop"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "instance id" in err.lower()


def test_session_stop_sends_shutdown(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "shutdown"
        assert instance_id == "abc123"
        return {"ok": True, "result": {"shutting_down": True}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "stop", "abc123", "--format", "json"])

    assert rc == 0
    stdout = capsys.readouterr().out
    parsed = json.loads(stdout)
    assert parsed["stopped"] is True
    assert parsed["instance_id"] == "abc123"


def test_session_start_rejects_global_instance_selector(monkeypatch, capsys):
    monkeypatch.setattr(
        bn.cli,
        "spawn_instance",
        lambda *args, **kwargs: pytest.fail("session start spawned random instance"),
    )

    rc = bn.cli.main(["-i", "named", "session", "start", "/bin/ls"])

    assert rc == 2
    error = capsys.readouterr().err
    assert "--instance-id named" in error
    assert "does not name" in error


def test_session_start_spawns_instance(monkeypatch, capsys):
    from bn.transport import BridgeInstance

    fake_inst = BridgeInstance(
        pid=999,
        socket_path=__import__("pathlib").Path("/tmp/test.sock"),
        registry_path=__import__("pathlib").Path("/tmp/test.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={},
        instance_id="test1234",
    )
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: fake_inst)
    monkeypatch.setattr(
        bn.cli,
        "send_request",
        lambda op, **kwargs: {
            "ok": True,
            "result": {
                "instance_id": "test1234",
                "associated": [os.getcwd()],
                "skipped": [],
            },
        },
    )

    rc = bn.cli.main(["session", "start", "--format", "json"])

    assert rc == 0
    stdout = capsys.readouterr().out
    parsed = json.loads(stdout)
    assert parsed["instance_id"] == "test1234"
    assert parsed["pid"] == 999


def _fake_instance(instance_id):
    from bn.transport import BridgeInstance
    import pathlib
    return BridgeInstance(
        pid=999,
        socket_path=pathlib.Path("/tmp/test.sock"),
        registry_path=pathlib.Path("/tmp/test.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={},
        instance_id=instance_id,
    )


def test_session_start_associates_project_and_forwards_workdir(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(
        bn.cli, "spawn_instance", lambda instance_id=None: _fake_instance("m1")
    )
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                          instance_id=None, spawn_missing_named=False):
        calls.append((op, params))
        if op == "associate_project_roots":
            return {
                "ok": True,
                "result": {
                    "instance_id": "m1",
                    "associated": [str(tmp_path)],
                    "skipped": [],
                },
            }
        return {
            "ok": True,
            "result": {
                "path": params.get("path"),
                "loaded": True,
                "targets": [{"selector": "x.bndb"}],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main([
        "session", "start", str(tmp_path / "x.bndb"), "--format", "json"
    ])

    assert rc == 0
    assert calls[0] == ("associate_project_roots", {"roots": [str(tmp_path)]})
    load = next(params for op, params in calls if op == "load_binary")
    assert load["workdir"] == str(tmp_path)
    assert "no_marker" not in load
    assert json.loads(capsys.readouterr().out)["project_roots"] == [str(tmp_path)]


def test_session_start_text_mode_shows_associated_projects(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(
        bn.cli, "spawn_instance", lambda instance_id=None: _fake_instance("m1")
    )
    monkeypatch.chdir(tmp_path)

    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                          instance_id=None, spawn_missing_named=False):
        if op == "associate_project_roots":
            return {
                "ok": True,
                "result": {
                    "instance_id": "m1",
                    "associated": [str(tmp_path)],
                    "skipped": [],
                },
            }
        return {
            "ok": True,
            "result": {
                "path": params.get("path"),
                "loaded": True,
                "targets": [{"selector": "x.bndb"}],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "start", str(tmp_path / "x.bndb")])

    assert rc == 0
    assert f"projects: {tmp_path}" in capsys.readouterr().out


def test_session_start_text_mode_shows_association_error(monkeypatch, capsys):
    monkeypatch.setattr(
        bn.cli, "spawn_instance", lambda instance_id=None: _fake_instance("m2")
    )

    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                          instance_id=None, spawn_missing_named=False):
        if op == "associate_project_roots":
            raise bn.cli.BridgeError("bridge is shutting down")
        raise AssertionError(f"unexpected op {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "start"])

    assert rc == 1
    out = capsys.readouterr().out
    assert "project association error: bridge is shutting down" in out
    assert "pass -i m2" in out


def test_session_start_partial_failure_keeps_bridge_but_exits_nonzero(monkeypatch, capsys):
    from bn.transport import BridgeInstance

    fake_inst = BridgeInstance(
        pid=999,
        socket_path=__import__("pathlib").Path("/tmp/test.sock"),
        registry_path=__import__("pathlib").Path("/tmp/test.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={},
        instance_id="half",
    )
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: fake_inst)

    ops = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        ops.append(op)
        if op == "associate_project_roots":
            return {
                "ok": True,
                "result": {"instance_id": "half", "associated": [], "skipped": []},
            }
        if op == "load_binary":
            if "good" in params["path"]:
                return {"ok": True, "result": {"path": params["path"], "loaded": True, "targets": [{"selector": "good.so"}]}}
            # Malformed success (no open target) -- NOT a transport-level
            # BridgeError -- this is the branch whose error text used to
            # predict teardown the bridge never actually performs.
            return {"ok": True, "result": {"path": params["path"], "loaded": True, "targets": []}}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "start", "/tmp/good.so", "/tmp/bad.so", "--format", "json"])

    # One binary loaded, so the bridge stays up, but the failure still surfaces.
    assert rc == 1
    assert "shutdown" not in ops
    parsed = json.loads(capsys.readouterr().out)
    assert "stopped" not in parsed
    bad_error = next(
        item["error"] for item in parsed["loaded"] if item["path"].endswith("bad.so")
    )
    # The bridge survives partial success -- the per-binary error text must
    # not predict a teardown that never happens.
    assert "will be stopped" not in bad_error
    assert "stopped" not in bad_error


def test_close_ignores_sticky_target_pin(fake_transport, monkeypatch, capsys):
    # A sticky pin must NOT pick which target a bare `close` tears down, and a
    # stale pin must not make cleanup fail. The pin is dropped and the bare
    # selector then resolves like any target-required command: with a single
    # open target it closes that one (#664 -- it no longer means close-all).
    calls = fake_transport({
        "list_targets": {
            "ok": True,
            "result": [{"target_id": "123:1:7", "selector": "foo.bndb"}],
        },
        "close_binary": {"ok": True, "result": {"closed": []}},
    })
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {"target": "stale_pin"})

    rc = bn.cli.main(["close", "--format", "text"])

    assert rc == 0
    assert [c["op"] for c in calls] == ["list_targets", "close_binary"]
    assert calls[-1]["target"] == "123:1:7"  # pin dropped, single target pinned by id
    assert "all" not in (calls[-1]["params"] or {})


def test_close_with_sticky_pin_under_multiple_targets_refuses(fake_transport, monkeypatch, capsys):
    # #664: the sticky pin is still dropped for `close`, but under multiple open
    # targets that bare selector now REFUSES with the --target hint instead of
    # falling through to a close-all. Nothing is closed.
    calls = fake_transport({
        "list_targets": {
            "ok": True,
            "result": [
                {"target_id": "123:1:7", "selector": "alpha.so", "view_id": "1"},
                {"target_id": "123:2:9", "selector": "beta.so", "view_id": "2"},
            ],
        },
        "close_binary": {"ok": True, "result": {"closed": []}},
    })
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {"target": "alpha.so"})

    rc = bn.cli.main(["close", "--format", "text"])

    assert rc == 2
    assert [c["op"] for c in calls] == ["list_targets"]
    err = capsys.readouterr().err
    assert "requires --target when multiple targets are open" in err
    assert "alpha.so" in err and "beta.so" in err


def test_instance_use_writes_state(tmp_session, monkeypatch, capsys):
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [_fake_bridge_instance("abc123")])

    rc = bn.cli.main(["instance", "use", "abc123"])

    assert rc == 0
    state = bn.session_state.read()
    assert state["instance_id"] == "abc123"
    assert capsys.readouterr().out.strip() == "instance: abc123"


def test_instance_use_clears_stale_target_pin_on_switch(tmp_session, monkeypatch, capsys):
    # #368 facet 3: switching to a DIFFERENT instance clears the target pin (it
    # belonged to the old instance) so a coincidentally matching selector in the
    # new instance can't silently resolve a bare command to a different target.
    bn.session_state.update(instance_id="old", target="recv_daemon")
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [_fake_bridge_instance("new1")])

    rc = bn.cli.main(["instance", "use", "new1"])

    assert rc == 0
    state = bn.session_state.read()
    assert state["instance_id"] == "new1"
    assert state.get("target") is None          # stale pin cleared
    assert "cleared stale target pin" in capsys.readouterr().out


def test_instance_use_keeps_target_pin_on_same_instance(tmp_session, monkeypatch, capsys):
    # Re-pinning the SAME instance keeps the target pin (not a switch).
    bn.session_state.update(instance_id="same1", target="recv_daemon")
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [_fake_bridge_instance("same1")])

    rc = bn.cli.main(["instance", "use", "same1"])

    assert rc == 0
    assert bn.session_state.read().get("target") == "recv_daemon"


def test_instance_use_keeps_target_pin_when_no_prior_instance(tmp_session, monkeypatch, capsys):
    # #368 review (MED): with a target pin but NO prior instance pin (prev_instance
    # is None), pinning an instance must NOT clear the target pin -- that is a first
    # pin, not a switch FROM a different instance. (None != resolved would wrongly
    # trip the stale-pin clear.)
    bn.session_state.update(target="recv_daemon")          # target pin, no instance pin
    assert bn.session_state.read().get("instance_id") is None
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [_fake_bridge_instance("first1")])

    rc = bn.cli.main(["instance", "use", "first1"])

    assert rc == 0
    state = bn.session_state.read()
    assert state["instance_id"] == "first1"
    assert state.get("target") == "recv_daemon"            # pin kept (not a switch)
    assert "cleared stale target pin" not in capsys.readouterr().out


def test_instance_use_default_pins_gui_bridge(tmp_session, monkeypatch, capsys):
    # The fixed GUI bridge has instance_id=None and selector "default". Storing
    # the raw None made session_state.update() DELETE the pin, so the pin
    # silently vanished. `bn instance use default` must persist "default" so
    # later bare commands resolve to the GUI bridge (#93).
    gui = _fake_bridge_instance("gui")
    object.__setattr__(gui, "instance_id", None)  # GUI bridge: id is None
    named = _fake_bridge_instance("headless1")
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [gui, named])

    rc = bn.cli.main(["instance", "use", "default"])

    assert rc == 0
    state = bn.session_state.read()
    assert state.get("instance_id") == "default"  # pin persisted, not deleted
    assert capsys.readouterr().out.strip() == "instance: default"


def test_instance_use_rejects_unknown_id(tmp_session, monkeypatch, capsys):
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [_fake_bridge_instance("abc123")])

    rc = bn.cli.main(["instance", "use", "not-running"])

    assert rc == 2
    assert "No running bridge instance" in capsys.readouterr().err
    assert bn.session_state.read() == {}


def test_instance_clear_removes_state(tmp_session, monkeypatch, capsys):
    bn.session_state.update(instance_id="abc123")
    assert bn.session_state.read()["instance_id"] == "abc123"

    rc = bn.cli.main(["instance", "clear"])

    assert rc == 0
    assert "instance_id" not in bn.session_state.read()
    assert capsys.readouterr().out.strip() == "cleared"


def test_sticky_instance_fills_when_flag_absent(tmp_session, monkeypatch):
    bn.session_state.update(instance_id="sticky_inst")

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1", "selector": "x"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["function", "list"])

    assert "sticky_inst" in captured


def test_cli_instance_flag_overrides_sticky(tmp_session, monkeypatch):
    bn.session_state.update(instance_id="sticky_inst")

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1", "selector": "x"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["--instance", "explicit", "function", "list"])

    assert "explicit" in captured
    assert "sticky_inst" not in captured


def test_env_var_overrides_sticky_instance(tmp_session, monkeypatch):
    bn.session_state.update(instance_id="sticky_inst")
    monkeypatch.setenv("BN_INSTANCE", "env_inst")

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1", "selector": "x"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["function", "list"])

    assert "env_inst" in captured
    assert "sticky_inst" not in captured


def test_sticky_target_fills_when_flag_absent(tmp_session, monkeypatch):
    bn.session_state.update(target="pam_qnx.so.2")

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured.append(target)
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["function", "list"])

    assert "pam_qnx.so.2" in captured


def test_cli_target_flag_overrides_sticky(tmp_session, monkeypatch):
    bn.session_state.update(target="sticky_tgt")

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured.append(target)
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["function", "list", "-t", "explicit_tgt"])

    assert "explicit_tgt" in captured
    assert "sticky_tgt" not in captured


def test_session_state_survives_subdir_navigation(tmp_session, monkeypatch):
    # Mark tmp_session as a project root via .git, then descend into subdirs.
    (tmp_session / ".git").mkdir()
    bn.session_state.update(target="pam_qnx.so.2")

    sub = tmp_session / "src" / "deep"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)

    assert bn.session_state.read()["target"] == "pam_qnx.so.2"


def test_malformed_session_state_treated_as_empty(tmp_session):
    from bn.paths import session_state_path, sessions_dir

    sessions_dir().mkdir(parents=True, exist_ok=True)
    session_state_path().write_text("{not json")

    assert bn.session_state.read() == {}


def test_session_list_marks_sticky(tmp_session, monkeypatch, capsys):
    monkeypatch.setattr(
        bn.cli, "list_instances",
        lambda: [_fake_bridge_instance("aaaa1111"), _fake_bridge_instance("bbbb2222", pid=222)],
    )
    bn.session_state.update(instance_id="aaaa1111")

    rc = bn.cli.main(["session", "list", "--format", "json"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    by_id = {entry["instance_id"]: entry for entry in parsed["items"]}
    assert by_id["aaaa1111"].get("sticky") is True
    assert "sticky" not in by_id["bbbb2222"]


def test_target_list_marks_sticky(tmp_session, fake_transport, capsys):
    fake_transport({
        "list_targets": {
            "ok": True,
            "result": [
                {"target_id": "1", "selector": "foo.so", "filename": "/p/foo.so"},
                {"target_id": "2", "selector": "bar.so", "filename": "/p/bar.so"},
            ],
        }
    })
    bn.session_state.update(target="foo.so")

    rc = bn.cli.main(["target", "list", "--format", "json"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    # #358: target list now uses the {kind, items} envelope.
    assert parsed["kind"] == "targets"
    by_sel = {entry["selector"]: entry for entry in parsed["items"]}
    assert by_sel["foo.so"].get("sticky") is True
    assert "sticky" not in by_sel["bar.so"]


def test_stale_sticky_instance_emits_hint(tmp_session, monkeypatch, capsys):
    bn.session_state.update(instance_id="dead_inst")

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        from bn.transport import BridgeError as _BE
        raise _BE(f"No bridge instance found with id: {instance_id}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "list"])
    err = capsys.readouterr().err

    assert rc == 2
    assert "No bridge instance found with id: dead_inst" in err
    assert "bn instance clear" in err


def test_sticky_hint_on_failed_contact(tmp_session, monkeypatch, capsys):
    """Bridge stopped mid-flight surfaces a transport error, not a registry miss."""
    bn.session_state.update(instance_id="dying_inst")

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        from bn.transport import BridgeError as _BE
        raise _BE(
            "Failed to contact Binary Ninja bridge pid 17881 at /tmp/x.sock: "
            "[Errno 104] Connection reset by peer"
        )

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["target", "list"])
    err = capsys.readouterr().err

    assert rc == 2
    assert "Failed to contact" in err
    assert "bn instance clear" in err


def test_sticky_hint_on_bridge_timeout(tmp_session, monkeypatch, capsys):
    bn.session_state.update(instance_id="slow_inst")

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        from bn.transport import BridgeError as _BE
        raise _BE(
            "Timed out waiting for Binary Ninja bridge pid 9999 at /tmp/x.sock after 30.0s"
        )

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["target", "list"])
    err = capsys.readouterr().err

    assert rc == 2
    assert "Timed out" in err
    assert "bn instance clear" in err


def test_sticky_hint_skipped_for_unrelated_errors(tmp_session, monkeypatch, capsys):
    """Bridge-side analysis errors must not get the sticky-clear hint."""
    bn.session_state.update(instance_id="alive_inst")

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        from bn.transport import BridgeError as _BE
        raise _BE("Function not found: nonexistent_symbol")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "info", "nonexistent_symbol"])
    err = capsys.readouterr().err

    assert rc == 2
    assert "Function not found" in err
    assert "bn instance clear" not in err


def test_session_start_no_bndb_propagates_to_each_load(monkeypatch, tmp_path):
    from bn.transport import BridgeInstance
    import pathlib

    a = tmp_path / "a"
    a.write_bytes(b"")
    b = tmp_path / "b"
    b.write_bytes(b"")

    fake_inst = BridgeInstance(
        pid=999,
        socket_path=pathlib.Path("/tmp/test.sock"),
        registry_path=pathlib.Path("/tmp/test.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={},
        instance_id="test1234",
    )
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: fake_inst)

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                          instance_id=None, spawn_missing_named=False):
        if op == "associate_project_roots":
            return {
                "ok": True,
                "result": {
                    "instance_id": "test1234",
                    "associated": [str(tmp_path)],
                    "skipped": [],
                },
            }
        captured.append(dict(params or {}))
        return {
            "ok": True,
            "result": {
                "loaded": True,
                "path": params["path"],
                "notes": [],
                "targets": [{"selector": pathlib.Path(params["path"]).name}],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["session", "start", "--no-bndb", str(a), str(b)])

    assert rc == 0
    assert len(captured) == 2
    assert all(item["prefer_bndb"] is False for item in captured)
    assert {item["path"] for item in captured} == {str(a), str(b)}


def test_class_list_invokes_op(monkeypatch):
    captured = {}

    def fake_call(args, op, params, **kwargs):
        captured["op"] = op
        captured["params"] = params
        return 0

    import bn.commands.cpp_class as cpp_class
    monkeypatch.setattr(cpp_class, "_call", fake_call)
    from bn.cli import build_parser
    args = build_parser().parse_args(["class", "list", "--all", "--query", "Session"])
    assert args.handler(args) == 0
    assert captured["op"] == "class_list"
    assert captured["params"]["include_all"] is True
    assert captured["params"]["query"] == "Session"
    assert "no_stl" not in captured["params"]   # flag absent unless passed

    args = build_parser().parse_args(["class", "list", "--no-stl"])
    assert args.handler(args) == 0
    assert captured["params"]["no_stl"] is True


def test_class_show_invokes_op(monkeypatch):
    captured = {}

    def fake_call(args, op, params, **kwargs):
        captured["op"] = op
        captured["params"] = params
        return 0

    import bn.commands.cpp_class as cpp_class
    monkeypatch.setattr(cpp_class, "_call", fake_call)
    from bn.cli import build_parser
    args = build_parser().parse_args(["class", "show", "net::Session"])
    assert args.handler(args) == 0
    assert captured["op"] == "class_show"
    assert captured["params"]["name"] == "net::Session"


def test_admin_text_renderer_failure_becomes_clean_error(monkeypatch, capsys):
    # Admin commands build their result locally and render it WITHOUT going
    # through _call, so they must apply the SAME malformed-result guard _call has
    # (#101): a text renderer that raises must surface a clean BridgeError
    # (exit 2) pointing at --format json, never a raw traceback.
    import bn.commands.admin as admin

    monkeypatch.setattr(bn.cli, "list_instances", lambda: [])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    def _boom(_value):
        raise ValueError("simulated malformed bridge result")

    monkeypatch.setattr(admin, "_render_session_list_text", _boom)

    rc = bn.cli.main(["session", "list"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--format json" in err




def test_instance_gc_json_carries_counts(monkeypatch, capsys):
    summary = {
        "live_instances": 0, "registries_purged": 0,
        "logs_removed": 0, "sockets_removed": 0, "removed": [],
    }
    monkeypatch.setattr(bn.cli, "gc_instances", lambda: summary)

    rc = bn.cli.main(["instance", "gc", "--format", "json"])

    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["logs_removed"] == 0
    assert data["live_instances"] == 0

def test_instance_gc_reports_summary_text(monkeypatch, capsys):
    # #80: `bn instance gc` reaps dead-instance cache litter and reports counts.
    monkeypatch.setattr(bn.cli, "gc_instances", lambda: {
        "live_instances": 2, "registries_purged": 1,
        "logs_removed": 147, "sockets_removed": 3, "removed": ["x"],
    })

    rc = bn.cli.main(["instance", "gc"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "147" in out          # logs reaped (the headline pain)
    assert "2" in out            # live instances kept
    assert "Traceback" not in out


# --- #276 Option 2: machine-readable capability index -----------------------

def test_capabilities_json_index_is_registry_derived(capsys):
    # A structured, registry-derived command->purpose->prefer-when index an
    # agent reads once to route. Local command -- no bridge/target required.
    rc = bn.cli.main(["capabilities", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)

    assert data["kind"] == "capabilities"
    items = data["items"]
    assert items and data["count"] == len(items)
    for it in items:
        assert {"command", "group", "help", "requires_target",
                "default_format", "prefer_when", "see_also"} <= set(it.keys())

    by_cmd = {it["command"]: it for it in items}
    # the overlaps the issue calls out are present and cross-linked
    assert "exact" in by_cmd["callsites"]["prefer_when"].lower()
    assert "xrefs" in by_cmd["callsites"]["see_also"]
    assert "callsites" in by_cmd["xrefs"]["see_also"]
    assert "function search" in by_cmd["function list"]["see_also"]
    assert "function list" in by_cmd["function search"]["see_also"]


def test_capabilities_see_also_references_are_valid_commands(capsys):
    # Integrity: every see_also points at a real registered command (the index
    # is registry-derived, so a stale/typo'd cross-link must fail loudly).
    rc = bn.cli.main(["capabilities", "--format", "json"])
    assert rc == 0
    items = json.loads(capsys.readouterr().out)["items"]
    commands = {it["command"] for it in items}
    for it in items:
        for ref in it["see_also"]:
            assert ref in commands, f"{it['command']} see_also -> unknown command {ref!r}"


def test_capabilities_text_groups_commands_with_routing_hints(capsys):
    rc = bn.cli.main(["capabilities"])  # text is the default
    assert rc == 0
    out = capsys.readouterr().out
    assert "callsites" in out and "xrefs" in out
    assert "prefer when:" in out
    assert "see also:" in out


def test_help_command_advertises_machine_catalog(capsys):
    rc = bn.cli.main(["help", "--format", "json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "help"
    assert "function" in payload["groups"]
    assert payload["capabilities_command"] == "bn capabilities --format json"


def test_help_command_filters_one_family(capsys):
    rc = bn.cli.main(["help", "function"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "function list:" in output
    assert "function search:" in output
    assert "machine-readable catalog" not in output


def _inst_with_binaries(binaries):
    from pathlib import Path as _P
    from bn.transport import BridgeInstance
    return BridgeInstance(
        pid=111, socket_path=_P("/tmp/x.sock"), registry_path=_P("/tmp/x.json"),
        plugin_name="bn_agent_bridge", plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={"binaries": list(binaries)}, instance_id="abc123")


def test_instance_list_shows_open_binaries(monkeypatch, capsys):
    # #80: `bn instance list` surfaces each instance's open binaries from the
    # registry, so "which instance has libfoo.so?" needs no per-instance round-trip.
    inst = _inst_with_binaries(["/fw/lib64/libfoo.so", "/fw/bin/daemon"])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "libfoo.so" in out and "daemon" in out


def test_instance_list_json_includes_binaries(monkeypatch, capsys):
    inst = _inst_with_binaries(["/fw/lib64/libfoo.so"])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "list", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["items"][0]["binaries"] == ["/fw/lib64/libfoo.so"]


def test_instance_list_no_binaries_key_when_empty(monkeypatch, capsys):
    # An instance with nothing open (or an older registry without the field) renders
    # cleanly without a binaries line.
    inst = _inst_with_binaries([])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "binaries" not in out


def test_instance_find_locates_binary_by_basename(monkeypatch, capsys):
    # #80: `bn instance find <name>` answers "which instance has this binary?"
    # from the registry (no per-instance round-trip), matching by basename.
    inst = _inst_with_binaries(["/fw/lib64/libfoo.so", "/fw/bin/daemon"])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "find", "libfoo.so"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "abc123" in out and "libfoo.so" in out


def test_instance_find_by_exact_path(monkeypatch, capsys):
    inst = _inst_with_binaries(["/fw/lib64/libfoo.so"])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "find", "/fw/lib64/libfoo.so", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["count"] == 1
    assert data["items"][0]["instance_id"] == "abc123"
    assert data["items"][0]["binary"] == "/fw/lib64/libfoo.so"


def test_instance_find_no_match(monkeypatch, capsys):
    inst = _inst_with_binaries(["/fw/lib64/libfoo.so"])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "find", "nope.so"])
    assert rc == 0
    assert "no instance" in capsys.readouterr().out.lower()


def test_instance_find_substring_of_basename(monkeypatch, capsys):
    # a bare query is a basename substring, so "libfoo" finds "libfoo.so.1.2.11"
    inst = _inst_with_binaries(["/fw/lib64/libfoo.so.1.2.11"])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "find", "libfoo"])
    assert rc == 0
    assert "abc123" in capsys.readouterr().out


def test_instance_find_path_suffix_is_component_aligned(monkeypatch, capsys):
    # A path-form query matches as a component-aligned suffix: `lib64/libfoo.so`
    # matches `/fw/lib64/libfoo.so` but a mid-component byte suffix must NOT
    # (`bar/libfoo.so` must not match `/foobar/libfoo.so`) (#80 review M1).
    inst = _inst_with_binaries(["/fw/lib64/libfoo.so", "/foobar/libqux.so"])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "find", "lib64/libfoo.so", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert [i["binary"] for i in data["items"]] == ["/fw/lib64/libfoo.so"]
    # mid-component suffix does not match
    rc = bn.cli.main(["instance", "find", "bar/libqux.so", "--format", "json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["count"] == 0


def test_instance_find_empty_query_matches_nothing(monkeypatch, capsys):
    inst = _inst_with_binaries(["/fw/lib64/libfoo.so"])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "find", "", "--format", "json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["count"] == 0


def test_instance_find_across_multiple_instances_and_old_bridge(monkeypatch, capsys):
    # A query matching binaries in TWO instances lists both; an older bridge whose
    # registry has no `binaries` key is skipped without error.
    a = _inst_with_binaries(["/fw/lib64/libfoo.so"]); a.instance_id = "inst_a"
    b = _inst_with_binaries(["/other/libfoo.so"]); b.instance_id = "inst_b"
    old = _inst_with_binaries([]); old.instance_id = "old"; old.meta.pop("binaries", None)
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [a, b, old])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "find", "libfoo.so", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert {i["instance_id"] for i in data["items"]} == {"inst_a", "inst_b"}
    assert data["count"] == 2


def test_explicit_empty_instance_is_rejected_not_pin_filled(fake_transport, monkeypatch, capsys):
    # #690 r3: `-i "$INST"` with $INST unset must not silently route the
    # command to the pinned instance (the same unset-shell-var doctrine the
    # r2 close guards follow for -t and the path).
    calls = fake_transport({})
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {"instance_id": "pinned-inst"})

    rc = bn.cli.main(["function", "list", "-i", ""])

    assert rc == 2
    assert calls == []
    assert "--instance is empty" in capsys.readouterr().err


def test_session_stop_rejects_explicit_empty_positional(fake_transport, monkeypatch, capsys):
    # #690 r4: `bn session stop "$ID"` with $ID unset must NOT fall through to
    # the sticky-pin-filled -i and shut down the PINNED bridge.
    calls = fake_transport({})
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {"instance_id": "pinned-inst"})

    rc = bn.cli.main(["session", "stop", ""])

    assert rc == 2
    assert calls == []
    err = capsys.readouterr().err
    assert "instance id is empty" in err


def test_session_stop_bare_does_not_stop_sticky_pinned_instance(fake_transport, monkeypatch, capsys):
    # #588: a bare `session stop` (no positional, no -i/--instance at all) must
    # NOT fall through to a sticky-pinned instance and stop it -- only a
    # genuinely explicit -i on THIS invocation may supply the target.
    calls = fake_transport({})
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {"instance_id": "pinned-inst"})

    rc = bn.cli.main(["session", "stop"])

    assert rc == 2
    assert calls == []
    err = capsys.readouterr().err
    assert "instance id" in err.lower()


def test_session_stop_explicit_instance_flag_still_resolves_sticky_slot(fake_transport, monkeypatch, capsys):
    # #588: an EXPLICIT -i/--instance on the `session stop` invocation itself
    # (as opposed to a silently-injected sticky pin) must still work.
    calls = fake_transport({"shutdown": {"ok": True, "result": {}}})
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {"instance_id": "pinned-inst"})

    rc = bn.cli.main(["session", "stop", "-i", "pinned-inst", "--format", "json"])

    assert rc == 0
    assert calls[0]["op"] == "shutdown"


def test_close_empty_instance_gets_the_actionable_message(fake_transport, monkeypatch, capsys):
    # #690 r4: close peeks list_targets BEFORE _call, so the empty-instance
    # rejection must fire on the peek path too -- with the same actionable
    # message every other command gets, not transport's generic one.
    calls = fake_transport({})
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    rc = bn.cli.main(["close", "-i", ""])

    assert rc == 2
    assert calls == []
    assert "--instance is empty" in capsys.readouterr().err


# --------------------------------------------------------------------------
# #694: stop/restart must prove the pid is still the bridge before signalling
# --------------------------------------------------------------------------


class _FakeSignaller:
    """Stand-in for transport.BridgeProcessSignal: records every send.

    Signalling is atomic now (pin the pid, verify identity through the pin, send
    through the same pin), so the CLI's remaining job is routing refusals and
    escalations. That routing is what these tests pin down; the pin itself is
    covered in tests/test_transport.py.
    """

    instances: list["_FakeSignaller"] = []

    def __init__(self, instance, refusal=None):
        self.instance = instance
        self.refusal = refusal
        self.sent: list[int] = []
        self.closed = False
        _FakeSignaller.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        self.closed = True

    def send(self, sig):
        if self.refusal is not None:
            return self.refusal
        self.sent.append(sig)
        return None


def _fake_signaller(monkeypatch, refusal=None):
    """Install the fake signaller; returns the list of created signallers."""
    _FakeSignaller.instances = []
    monkeypatch.setattr(
        bn.cli,
        "BridgeProcessSignal",
        lambda instance: _FakeSignaller(instance, refusal=refusal),
    )
    return _FakeSignaller.instances


def _unreachable_fake_instance(instance_id="abc123", pid=111):
    inst = _fake_bridge_instance(instance_id, pid=pid)  # noqa: F405
    inst.unreachable = True
    return inst


def test_session_stop_sigterm_fallback_sends_through_the_verified_pin(monkeypatch, capsys):
    import signal as signal_mod

    from bn.transport import BridgeError

    def fail_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, **kwargs):
        raise BridgeError("bridge unreachable")

    monkeypatch.setattr(bn.cli, "send_request", fail_send_request)
    monkeypatch.setattr(
        bn.cli, "find_lifecycle_instance", lambda target: _fake_bridge_instance("abc123")  # noqa: F405
    )
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, **kw: True)
    signallers = _fake_signaller(monkeypatch)
    monkeypatch.setattr("os.kill", lambda pid, sig: pytest.fail("raw os.kill is banned"))

    rc = bn.cli.main(["session", "stop", "abc123", "--format", "json"])

    assert rc == 0
    assert len(signallers) == 1                      # one pin for the whole stop
    assert signallers[0].sent == [signal_mod.SIGTERM]
    assert signallers[0].closed is True
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["stopped"] is True and parsed["method"] == "sigterm"


def test_session_stop_escalates_sigkill_through_the_same_pin(monkeypatch, capsys):
    import signal as signal_mod

    monkeypatch.setattr(
        bn.cli,
        "send_request",
        lambda op, **kwargs: {"ok": True, "result": {"shutting_down": True}},
    )
    monkeypatch.setattr(
        bn.cli, "find_lifecycle_instance", lambda target: _fake_bridge_instance("abc123")  # noqa: F405
    )
    converged = iter([False, True])
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, **kw: next(converged))
    signallers = _fake_signaller(monkeypatch)

    rc = bn.cli.main(["session", "stop", "abc123", "--format", "json"])

    assert rc == 0
    assert len(signallers) == 1                      # SAME pin, never reopened
    assert signallers[0].sent == [signal_mod.SIGKILL]
    assert json.loads(capsys.readouterr().out)["method"] == "sigkill"


def test_session_stop_reports_a_refused_sigterm_and_signals_nothing(monkeypatch, capsys):
    from bn.transport import BridgeError

    def fail_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, **kwargs):
        raise BridgeError("bridge unreachable")

    monkeypatch.setattr(bn.cli, "send_request", fail_send_request)
    monkeypatch.setattr(
        bn.cli, "find_lifecycle_instance", lambda target: _fake_bridge_instance("abc123")  # noqa: F405
    )
    signallers = _fake_signaller(
        monkeypatch,
        refusal=(
            "refusing to signal pid 111 for bridge instance 'abc123': the identity "
            "recorded at startup (boot id plus process start time) does not match "
            "the pinned process, so the bridge exited and its pid was reused"
        ),
    )

    rc = bn.cli.main(["session", "stop", "abc123"])

    assert rc == 1
    assert signallers[0].sent == []
    err = capsys.readouterr().err
    assert "refusing to signal pid 111" in err
    assert "reused" in err


def test_session_stop_reports_a_refused_sigkill_escalation(monkeypatch, capsys):
    monkeypatch.setattr(
        bn.cli,
        "send_request",
        lambda op, **kwargs: {"ok": True, "result": {"shutting_down": True}},
    )
    monkeypatch.setattr(
        bn.cli, "find_lifecycle_instance", lambda target: _fake_bridge_instance("abc123")  # noqa: F405
    )
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, **kw: False)
    signallers = _fake_signaller(
        monkeypatch, refusal="refusing to signal pid 111: no verifiable process identity"
    )

    rc = bn.cli.main(["session", "stop", "abc123", "--format", "json"])

    assert rc == 1
    assert signallers[0].sent == []
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["stopped"] is False
    assert "did not fully tear down" in parsed["error"]
    assert "no verifiable process identity" in parsed["error"]


def test_session_stop_resolves_an_unreachable_bridge(monkeypatch, capsys):
    # A socket-less bridge is hidden from `session list` (nothing can be
    # dispatched to it), but stopping the live process it names is exactly what a
    # user needs -- so stop resolves it through the lifecycle lookup (#694).
    import signal as signal_mod

    from bn.transport import BridgeError

    lookups: list[str] = []

    def fail_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, **kwargs):
        raise BridgeError("No bridge instance found with id: abc123")

    monkeypatch.setattr(bn.cli, "send_request", fail_send_request)
    monkeypatch.setattr(
        bn.cli, "list_instances", lambda **kw: pytest.fail("must use the lifecycle lookup")
    )
    monkeypatch.setattr(
        bn.cli,
        "find_lifecycle_instance",
        lambda target: lookups.append(target) or _unreachable_fake_instance("abc123"),
    )
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, **kw: True)
    signallers = _fake_signaller(monkeypatch)

    rc = bn.cli.main(["session", "stop", "abc123", "--format", "json"])

    assert rc == 0
    assert lookups == ["abc123"]
    assert signallers[0].sent == [signal_mod.SIGTERM]
    assert json.loads(capsys.readouterr().out)["method"] == "sigterm"


def test_session_restart_refuses_when_the_signal_is_not_delivered(monkeypatch, capsys):
    from bn.transport import BridgeError

    def fail_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, **kwargs):
        raise BridgeError("bridge unreachable")

    monkeypatch.setattr(bn.cli, "send_request", fail_send_request)
    monkeypatch.setattr(
        bn.cli, "find_lifecycle_instance", lambda target: _fake_bridge_instance("res1")  # noqa: F405
    )
    monkeypatch.setattr(
        bn.cli, "_send_request_to_instance", lambda *a, **k: {"ok": True, "result": []}
    )
    monkeypatch.setattr(
        bn.cli,
        "spawn_instance",
        lambda instance_id=None: pytest.fail("must not respawn after refusing"),
    )
    signallers = _fake_signaller(
        monkeypatch,
        refusal="refusing to signal pid 111 for bridge instance 'res1': its pid was reused",
    )

    rc = bn.cli.main(["session", "restart", "res1"])

    assert rc == 2
    assert signallers[0].sent == []
    err = capsys.readouterr().err
    assert "refusing to signal pid 111" in err
    assert "left as it is" in err


def test_session_restart_escalates_through_one_pin(monkeypatch, capsys):
    import pathlib
    import signal as signal_mod

    from bn.transport import BridgeError, BridgeInstance

    def send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, **kwargs):
        if op == "shutdown":
            raise BridgeError("bridge unreachable")
        return {"ok": True, "result": {"loaded": True, "path": "/tmp/app.bin", "targets": []}}

    new_instance = BridgeInstance(
        pid=5151,
        socket_path=pathlib.Path("/tmp/res1.sock"),
        registry_path=pathlib.Path("/tmp/res1.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta={},
        instance_id="res1",
    )
    monkeypatch.setattr(bn.cli, "send_request", send_request)
    monkeypatch.setattr(
        bn.cli, "find_lifecycle_instance", lambda target: _fake_bridge_instance("res1")  # noqa: F405
    )
    monkeypatch.setattr(
        bn.cli, "_send_request_to_instance", lambda *a, **k: {"ok": True, "result": []}
    )
    converged = iter([False, True])
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, **kw: next(converged))
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: new_instance)
    signallers = _fake_signaller(monkeypatch)

    assert bn.cli.main(["session", "restart", "res1", "--format", "json"]) == 0
    assert len(signallers) == 1
    assert signallers[0].sent == [signal_mod.SIGTERM, signal_mod.SIGKILL]
    assert signallers[0].closed is True


def _failing_start_instance():
    import pathlib

    from bn.transport import BridgeInstance

    return BridgeInstance(
        pid=7777,
        socket_path=pathlib.Path("/tmp/start1.sock"),
        registry_path=pathlib.Path("/tmp/start1.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta={},
        instance_id="start1",
    )


def _failing_start_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, **kwargs):
    from bn.transport import BridgeError

    raise BridgeError("load failed" if op.startswith("load_binary") else "shutdown refused")


def test_session_start_cleanup_signals_through_the_verified_pin(monkeypatch, capsys, tmp_path):
    # The all-preloads-failed cleanup is a fallback signalling path too: it used
    # raw os.kill and could terminate a recycled pid (#694).
    import signal as signal_mod

    binary = tmp_path / "app.bin"
    binary.write_bytes(b"\x7fELF")
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: _failing_start_instance())
    monkeypatch.setattr(bn.cli, "send_request", _failing_start_send_request)
    converged = iter([False, True])
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, **kw: next(converged))
    signallers = _fake_signaller(monkeypatch)
    monkeypatch.setattr("os.kill", lambda pid, sig: pytest.fail("raw os.kill is banned"))

    rc = bn.cli.main(["session", "start", str(binary), "--format", "json"])

    assert rc == 1
    assert len(signallers) == 1                      # one pin for TERM + KILL
    assert signallers[0].sent == [signal_mod.SIGTERM, signal_mod.SIGKILL]
    assert signallers[0].closed is True
    assert json.loads(capsys.readouterr().out)["stopped"] is True


def test_session_start_cleanup_reports_a_refused_signal(monkeypatch, capsys, tmp_path):
    binary = tmp_path / "app.bin"
    binary.write_bytes(b"\x7fELF")
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: _failing_start_instance())
    monkeypatch.setattr(bn.cli, "send_request", _failing_start_send_request)
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, **kw: False)
    signallers = _fake_signaller(
        monkeypatch, refusal="refusing to signal pid 7777: no verifiable process identity"
    )

    rc = bn.cli.main(["session", "start", str(binary), "--format", "json"])

    assert rc == 1
    assert signallers[0].sent == []
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["stopped"] is False
    errors = " ".join(parsed["cleanup_errors"])
    assert "SIGTERM not sent" in errors and "SIGKILL not sent" in errors
    assert "no verifiable process identity" in errors


# --------------------------------------------------------------------------
# #694: restart preserves private project associations
# --------------------------------------------------------------------------


def _restart_bridge_stubs(
    monkeypatch, project_roots, *, restore_result=None, restore_error=None
):
    import pathlib

    from bn.transport import BridgeError, BridgeInstance

    old = _fake_bridge_instance("keep1", pid=4242)  # noqa: F405
    old.meta["project_roots"] = [str(path) for path in project_roots]
    new = BridgeInstance(
        pid=5151,
        socket_path=pathlib.Path("/tmp/keep1.sock"),
        registry_path=pathlib.Path("/tmp/keep1.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={},
        instance_id="keep1",
    )
    calls: list[tuple[str, dict]] = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                          instance_id=None, **kwargs):
        calls.append((op, dict(params or {})))
        if op == "shutdown":
            return {"ok": True, "result": {"shutting_down": True}}
        if op == "load_binary":
            return {
                "ok": True,
                "result": {
                    "loaded": True,
                    "path": params["path"],
                    "notes": [],
                    "targets": [],
                },
            }
        if op == "associate_project_roots":
            if restore_error is not None:
                raise BridgeError(restore_error)
            return {
                "ok": True,
                "result": restore_result
                if restore_result is not None
                else {
                    "instance_id": "keep1",
                    "associated": list(params["roots"]),
                    "skipped": [],
                },
            }
        raise AssertionError(f"unexpected op {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli, "list_instances", lambda **kw: [old])
    monkeypatch.setattr(bn.cli, "find_lifecycle_instance", lambda target: old)
    monkeypatch.setattr(
        bn.cli,
        "_send_request_to_instance",
        lambda inst, op, **kwargs: {
            "ok": True,
            "result": [{"filename": "/tmp/app.bin", "analysis_state": "full"}],
        },
    )
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, **kw: True)
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: new)
    return calls


def test_session_restart_restores_original_project_roots(
    monkeypatch, capsys, tmp_path
):
    project = tmp_path / "project"
    project.mkdir()
    calls = _restart_bridge_stubs(monkeypatch, [project])

    rc = bn.cli.main(["session", "restart", "keep1", "--format", "json"])

    assert rc == 0
    restore = [params for op, params in calls if op == "associate_project_roots"]
    assert restore == [{"roots": [str(project)]}]
    load = next(params for op, params in calls if op == "load_binary")
    assert "workdir" not in load
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["project_roots"] == [str(project)]
    assert "project_association_error" not in parsed
    assert [op for op, _ in calls].index("associate_project_roots") > [
        op for op, _ in calls
    ].index("load_binary")


def test_session_restart_reports_failed_association_restore_and_fails(
    monkeypatch, capsys, tmp_path
):
    project = tmp_path / "project"
    project.mkdir()
    _restart_bridge_stubs(
        monkeypatch, [project], restore_error="associate operation unavailable"
    )

    rc = bn.cli.main(["session", "restart", "keep1", "--format", "json"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "project association failed" in captured.err
    parsed = json.loads(captured.out)
    assert parsed["project_association_error"] == "associate operation unavailable"
    assert "project_roots" not in parsed


def test_session_restart_reports_failed_association_restore_in_text_mode(
    monkeypatch, capsys, tmp_path
):
    project = tmp_path / "project"
    project.mkdir()
    _restart_bridge_stubs(
        monkeypatch, [project], restore_error="associate operation unavailable"
    )

    rc = bn.cli.main(["session", "restart", "keep1"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "project association error: associate operation unavailable" in captured.out
    assert "pass -i keep1" in captured.out


def test_session_restart_warns_about_each_skipped_association(
    monkeypatch, capsys, tmp_path
):
    project = tmp_path / "project"
    project.mkdir()
    _restart_bridge_stubs(
        monkeypatch,
        [project],
        restore_result={
            "instance_id": "keep1",
            "associated": [],
            "skipped": [
                {"path": str(project), "reason": "project directory does not exist"}
            ],
        },
    )

    assert bn.cli.main(["session", "restart", "keep1", "--format", "json"]) == 0
    assert "project directory does not exist" in capsys.readouterr().err


# --------------------------------------------------------------------------
# #694: an explicit empty instance selector is an error, never "list them all"
# --------------------------------------------------------------------------


def test_session_list_rejects_an_explicit_empty_instance(monkeypatch, capsys):
    # The selector filter is truthiness-based, so `-i ''` silently disabled it and
    # listed EVERY session -- the opposite of what an explicit selector asks for.
    listed = []
    monkeypatch.setattr(
        bn.cli,
        "list_instances",
        lambda: listed.append(True) or [_fake_bridge_instance("abc123")],  # noqa: F405
    )
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    rc = bn.cli.main(["session", "list", "-i", ""])

    assert rc == 2
    assert listed == []
    assert "--instance is empty" in capsys.readouterr().err


def test_instance_list_rejects_an_explicit_empty_instance(monkeypatch, capsys):
    monkeypatch.setattr(
        bn.cli, "list_instances", lambda: pytest.fail("must not enumerate instances")
    )
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    assert bn.cli.main(["instance", "list", "-i", ""]) == 2
    assert "--instance is empty" in capsys.readouterr().err


def test_session_list_with_an_explicit_selector_still_filters(monkeypatch, capsys):
    monkeypatch.setattr(
        bn.cli,
        "list_instances",
        lambda: [_fake_bridge_instance("abc123"), _fake_bridge_instance("zz9999")],  # noqa: F405
    )
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    assert bn.cli.main(["session", "list", "-i", "zz9999", "--format", "json"]) == 0

    items = json.loads(capsys.readouterr().out)["items"]
    assert [item["instance_id"] for item in items] == ["zz9999"]
