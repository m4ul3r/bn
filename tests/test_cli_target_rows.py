"""#1033 items 1, 2, 4, 5, 7, #1035 item 1, #1036 item 7: target rows, instance
routing and printed commands.

Every printed command here is run as printed through a real `sh -c`, in a
directory holding an executable named `aaa` with `.` on PATH, against a
recording stub `bn`. A glob or an unquoted metacharacter in the pasted text
would run `aaa` (which leaves a trace file) instead of the stub, and the argv
the stub received is parsed back with the real CLI parser.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

import bn.cli
from bn.paths import instances_dir
from bn.target_hint import format_multi_target_hint, target_row
from bn.transport import BridgeError, BridgeInstance


def _paste(line: str, tmp_path: Path) -> list[str] | None:
    """Run *line* through `sh -c`; return the argv the stub `bn` received, or
    None when the stub never ran. Fails if the glob trap `aaa` ran."""
    stub_dir = tmp_path / "stub"
    trap_dir = tmp_path / "populated"
    record = tmp_path / "argv.bin"
    trap = tmp_path / "trap-ran"
    if not stub_dir.exists():
        stub_dir.mkdir()
        trap_dir.mkdir()
        (stub_dir / "bn").write_text(f'#!/bin/sh\nprintf "%s\\0" "$@" > {record}\n')
        (stub_dir / "bn").chmod(0o755)
        (trap_dir / "aaa").write_text(f"#!/bin/sh\ntouch {trap}\n")
        (trap_dir / "aaa").chmod(0o755)
    record.unlink(missing_ok=True)
    env = dict(os.environ, PATH=f"{stub_dir}:.:{os.environ.get('PATH', '')}")
    subprocess.run(["sh", "-c", line], check=False, cwd=trap_dir, env=env,
                   capture_output=True, text=True, timeout=10)
    assert not trap.exists(), f"pasting {line!r} ran a file from the cwd"
    if not record.exists():
        return None
    return record.read_bytes().decode().split("\0")[:-1]


def _parse(argv: list[str]):
    return bn.cli.build_parser().parse_args(argv)


_TARGETS = [
    {"selector": "-diagnostics", "view_id": "v1", "target_id": "1:1:1",
     "filename": "/p/-diagnostics", "active": True},
    {"selector": "hello world", "view_id": "v2", "target_id": "1:2:1",
     "filename": "/p/hello world"},
]


def _listing() -> str:
    return format_multi_target_hint(
        "No active BinaryView is selected and multiple targets are open.", _TARGETS)


def _commands(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.startswith("  bn ")]


# --- item 2: a pasted raw row runs nothing ------------------------------------


@pytest.mark.parametrize("target", _TARGETS)
def test_a_pasted_raw_listing_row_runs_nothing_1033_2(tmp_path, target):
    """The active row used to start with `*`, which a populated directory
    expanded into a file name the shell then ran."""
    row = target_row(target)
    assert _paste(row, tmp_path) is None
    assert row.endswith("(active)") is bool(target.get("active"))


# --- item 1: fan-out and structured batch errors are complete commands --------


@pytest.mark.parametrize("instance", ["a", "-demo"])
def test_fanout_error_rows_paste_as_that_instances_call_1033_1(
    monkeypatch, capsys, tmp_path, instance,
):
    insts = [types.SimpleNamespace(instance_id=instance)]
    monkeypatch.setattr(bn.cli, "list_instances", lambda: insts)
    monkeypatch.setattr(bn.cli, "instance_selector", lambda i: i.instance_id)
    monkeypatch.setattr(bn.cli.session_state, "read", dict)

    def fake_send(op, *, params=None, target=None, instance_id=None, **k):
        if op == "list_targets":
            raise BridgeError("peek failed")
        raise BridgeError(_listing())

    monkeypatch.setattr(bn.cli, "send_request", fake_send)
    monkeypatch.setattr(bn.cli, "_resolve_target", lambda args, **k: None)

    bn.cli.main(["function", "list", "--all-instances", "--limit", "5", "--format", "json"])
    (row,) = json.loads(capsys.readouterr().out)["instances"]
    commands = _commands(row["error"])
    assert len(commands) == len(_TARGETS)
    for command, target in zip(commands, _TARGETS):
        parsed = _parse(_paste(command, tmp_path))
        assert parsed._command_path == ("function", "list")
        assert (parsed.instance, parsed.target, parsed.limit) == (instance, target["selector"], 5)
        assert not getattr(parsed, "all_instances", False)


@pytest.mark.parametrize("fmt", ["json", "ndjson", "text"])
def test_batch_decompile_error_rows_paste_in_every_format_1033_1(
    fake_transport, capsys, tmp_path, fmt,
):
    fake_transport({"decompile_batch": {"ok": True, "result": {
        "kind": "decompile_batch", "requested": 1, "resolved": 0,
        "functions": [{"identifier": "main", "ok": False, "error": _listing()}],
    }}})
    bn.cli.main(["--instance=-demo", "-t", "x", "decompile", "main", "other", "--format", fmt])
    out = capsys.readouterr().out
    if fmt == "json":
        out = json.loads(out)["functions"][0]["error"]
    elif fmt == "ndjson":
        records = [json.loads(line) for line in out.splitlines()]
        out = next(r["error"] for r in records if isinstance(r, dict) and "error" in r)
    commands = _commands(out)
    assert len(commands) == len(_TARGETS)
    for command, target in zip(commands, _TARGETS):
        parsed = _parse(_paste(command, tmp_path))
        assert parsed._command_path == ("decompile",)
        assert (parsed.instance, parsed.target, parsed.identifier) == (
            "-demo", target["selector"], ["main", "other"])


def test_batch_decompile_target_refusal_fails_the_request_once_1033_1(monkeypatch):
    """The bridge resolves the target before the loop: a refusal is the
    request's error (rendered with complete rows by the CLI), not N copies."""
    from bn_agent_bridge import read_decompile

    class _Ctx:
        def _resolve_view(self, selector):
            raise RuntimeError(_listing())

        def _find_function(self, *a, **k):
            raise AssertionError("no identifier is attempted after a refusal")

    with pytest.raises(RuntimeError, match="Open targets:"):
        read_decompile._decompile_batch(_Ctx(), None, ["a", "b"])


# --- item 4: an instance id starting with `-` ---------------------------------


def test_dash_instance_id_spawns_and_is_advised_attached_1033_4(monkeypatch, tmp_path):
    import bn.headless
    from bn import transport

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path / "cache"))
    instances_dir().mkdir(parents=True, exist_ok=True)
    spawned: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        spawned.append(cmd)
        raise AssertionError("stop here")

    with monkeypatch.context() as patch:
        patch.setattr(transport.subprocess, "Popen", fake_popen)
        patch.setattr(transport, "_find_bn_agent", lambda: ["bn-agent"])
        with pytest.raises(AssertionError, match="stop here"):
            transport._spawn_instance_unlocked("-demo", timeout=5.0)
    started: dict = {}
    monkeypatch.setitem(sys.modules, "bn_agent_bridge.bridge", types.SimpleNamespace(
        start_headless=lambda binaries, **kw: started.update(kw)))
    assert bn.headless.main(spawned[0][1:]) == 0
    assert started["instance_id"] == "-demo"

    with pytest.raises(BridgeError) as excinfo:
        transport.choose_instance("-demo", auto_start=False)
    advice = re.search(r"Start one with: (bn [^(]*?)\s*(?:\(|$)", str(excinfo.value)).group(1)
    parsed = _parse(_paste(advice.replace("/path/to/binary", "./x"), tmp_path))
    assert (parsed._command_path, parsed.instance_id) == (("session", "start"), "-demo")


# --- item 5: stop/restart "Save first" ----------------------------------------


@pytest.mark.parametrize("verb", ["stop", "restart"])
def test_save_first_advice_pastes_with_dash_values_1033_5(tmp_path, verb):
    from bn.commands import admin

    rows = [{"selector": "-diagnostics", "filename": "/p/-diagnostics", "unsaved": True}]
    with pytest.raises(BridgeError) as excinfo:
        admin._unsaved_targets_or_refuse(verb, "-demo", rows, discard=False)
    (command,) = _commands(str(excinfo.value))
    parsed = _parse(_paste(command, tmp_path))
    assert (parsed._command_path, parsed.instance, parsed.target) == (
        ("save",), "-demo", "-diagnostics")


# --- item 7: a locator starting with -t/-i is not routing ---------------------


@pytest.mark.parametrize("hint", [
    "bn il --view llil --ssa -- '-th x'",
    "bn il '-th x' --view llil --ssa",
    "bn il '-ifoo' --view llil --ssa",
])
def test_dash_locator_is_not_mistaken_for_routing_1033_7(hint):
    value = {"error": f"Inspect `{hint}`."}
    bn.cli._contextualize_diagnostic_hints(value, target="sel", instance="X1")
    assert value["error"] == f"Inspect `bn -i X1 -t sel {hint[3:]}`."


def test_the_trace_il_hint_runs_for_a_dash_function_name_1033_7(tmp_path):
    value = {"error": "Inspect `bn il --view llil --ssa -- '-th x'`."}
    bn.cli._contextualize_diagnostic_hints(value, target="-local", instance="-demo")
    command = re.search(r"`(bn [^`]+)`", value["error"]).group(1)
    parsed = _parse(_paste(command, tmp_path))
    assert (parsed.instance, parsed.target, parsed.identifier, parsed.view) == (
        "-demo", "-local", "-th x", "llil")


# --- #1035 item 1: a deliberate stop is not reported as a crash ---------------


def _handle(registry_path: Path, *, token: str = "t", pid: int = 4242,
            meta: dict | None = None) -> BridgeInstance:
    return BridgeInstance(
        pid=pid, socket_path=registry_path.with_suffix(".sock"), registry_path=registry_path,
        plugin_name="bn_agent_bridge", plugin_version="0", started_at=None,
        meta={"bind_token": "b", **(meta or {})}, instance_token=token,
        instance_id=registry_path.stem,
    )


def _record(**over) -> dict:
    """A well-formed `stopped` record for `_handle()`'s bind."""
    return {"instance_id": "x", "instance_token": "t", "bind_token": "b", "pid": 4242,
            "reason": "shutdown_request", "state": "stopped",
            "began_at": "2026-10-16T09:00:00+00:00",
            "completed_at": "2026-10-16T09:00:02+00:00",
            "requests_drained": False, "unfinished_requests": 1, **over}


def _diagnose(handle: BridgeInstance, record: object = None) -> str:
    from bn import transport
    from bn.paths import stop_record_path

    if record is not None:
        stop_record_path(handle.registry_path).write_text(
            record if isinstance(record, str) else json.dumps(record))
    return str(transport._empty_response_error(handle, "refresh"))


_CLAIMS = ("crash", "OOM", "fault", "shutting down", "clean", "completed", "began")


@pytest.mark.parametrize(("record", "found"), [
    (None, "no stop record was found"),
    ("{not json", "could not be read as one"),
    ("[]", "could not be read as one"),
    (_record(instance_token="successor"), "is not this bridge's"),
    (_record(bind_token="earlier-bind"), "is not this bridge's"),
    (_record(pid=9999), "is not this bridge's"),
])
@pytest.mark.parametrize("running", [True, False])
def test_without_this_binds_stop_record_no_cause_is_claimed_1035_1(
    monkeypatch, tmp_path, record, found, running,
):
    """Registry absence, a corrupt record or another bridge's or bind's record
    is not evidence of why the bridge closed the connection."""
    from bn import transport

    monkeypatch.setattr(transport, "_process_running", lambda pid: running)
    message = _diagnose(_handle(tmp_path / "x.json"), record)
    assert found in message and "reason it closed the connection is unknown" in message
    assert ("still running" if running else "no longer running") in message
    for claim in _CLAIMS:
        assert claim not in message


@pytest.mark.parametrize("record", [
    {key: value for key, value in _record().items() if key != "bind_token"},
    _record(reason=[]), _record(reason="made_up"), _record(began_at={}),
    _record(began_at="yesterday"), _record(pid="bad"), _record(pid=True),
    _record(state="done"), _record(completed_at=None), _record(unfinished_requests=True),
    _record(requests_drained="no"), _record(instance_id=None),
])
def test_a_malformed_record_with_this_binds_token_is_not_a_witness_1035_1(
    monkeypatch, tmp_path, record,
):
    """Review round 2: only the exact shape stop() writes counts."""
    from bn import transport

    monkeypatch.setattr(transport, "_process_running", lambda pid: False)
    message = _diagnose(_handle(tmp_path / "x.json"), record)
    assert "could not be read as one" in message
    for claim in _CLAIMS:
        assert claim not in message


def test_only_a_stopped_record_reports_a_completed_teardown_1035_1(monkeypatch, tmp_path):
    """Review round 2: `stopping` proves a stop began, not that it completed."""
    from bn import transport

    monkeypatch.setattr(transport, "_process_running", lambda pid: False)
    handle = _handle(tmp_path / "x.json")
    done = _diagnose(handle, _record())
    assert "shows a completed teardown (a shutdown request" in done
    assert "1 request still in flight" in done
    begun = _diagnose(handle, {key: value for key, value in _record(state="stopping").items()
                               if key not in ("completed_at", "requests_drained",
                                              "unfinished_requests")})
    assert "began at 2026-10-16T09:00:00+00:00, but not that the teardown completed" in begun
    assert "completed teardown" not in begun and "clean" not in begun


@pytest.mark.parametrize("kind", ["fifo", "directory", "symlink"])
def test_a_non_regular_record_path_is_unreadable_without_blocking_1035_1(tmp_path, kind):
    """Review round 2: a FIFO at the record path must not hang the diagnosis."""
    import threading

    from bn.paths import stop_record_path

    handle = _handle(tmp_path / "x.json")
    path = stop_record_path(handle.registry_path)
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    else:
        (tmp_path / "real.json").write_text(json.dumps(_record()))
        path.symlink_to(tmp_path / "real.json")
    out: list[str] = []
    worker = threading.Thread(target=lambda: out.append(_diagnose(handle)), daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert out, "the diagnosis blocked on the record path"
    assert "could not be read as one" in out[0]


def test_distinct_ids_get_distinct_record_paths_1035_1(tmp_path):
    """Review round 2: `...` and `....json` are both valid ids."""
    from bn.paths import stop_record_path

    assert (stop_record_path(tmp_path / "....json")
            != stop_record_path(tmp_path / "....json.json"))


def _started(module, instance_id: str):
    bridge = module.BinaryNinjaBridge(instance_id=instance_id)
    bridge.start()
    meta = json.loads(bridge.registry_path.read_text())
    handle = BridgeInstance(
        pid=meta["pid"], socket_path=bridge.socket_path, registry_path=bridge.registry_path,
        plugin_name="bn_agent_bridge", plugin_version="0", started_at=None, meta=meta,
        instance_token=meta["instance_token"], instance_id=instance_id,
    )
    return bridge, handle


def test_a_bridge_stop_writes_the_record_the_diagnosis_reads_1035_1(monkeypatch, tmp_path):
    """The real teardown writes the witness in two phases; the CLI matches it to
    the handle discovery produced from the real registry."""
    from _bridge_fakes import _load_bridge

    from bn import transport
    from bn.paths import stop_record_path

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)
    bridge, ours = _started(module, "sx")
    try:
        assert "no stop record was found" in str(transport._empty_response_error(ours, "x"))
        module._bind_shutdown(bridge, {}, None)
    finally:
        bridge.stop()
    record = json.loads(stop_record_path(bridge.registry_path).read_text())
    assert (record["state"], record["reason"], record["bind_token"]) == (
        "stopped", "shutdown_request", ours.meta["bind_token"])
    assert "completed teardown (a shutdown request" in str(
        transport._empty_response_error(ours, "refresh"))
    stale = _handle(bridge.registry_path, token="another-bridge", pid=os.getpid())
    assert "is not this bridge's" in str(transport._empty_response_error(stale, "x"))
    # `bn instance gc` reaps the record once no registry of that name remains.
    assert transport.gc_instances()["stop_records_removed"] == 1
    assert not stop_record_path(bridge.registry_path).exists()


def test_a_teardown_interrupted_after_it_began_claims_no_completion_1035_1(
    monkeypatch, tmp_path,
):
    """Review round 2 blocker: a bridge that dies between the phases leaves a
    `stopping` record, which reports only that a stop began."""
    from _bridge_fakes import _load_bridge

    from bn import transport
    from bn.paths import stop_record_path

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)
    bridge, ours = _started(module, "sx")
    module._bind_shutdown(bridge, {}, None)
    with monkeypatch.context() as patch:
        def interrupted():
            raise RuntimeError("died during teardown")

        patch.setattr(bridge, "_release_own_instance_state_locked", interrupted)
        with pytest.raises(RuntimeError, match="died during teardown"):
            bridge.stop()
    try:
        assert json.loads(stop_record_path(bridge.registry_path).read_text())["state"] == (
            "stopping")
        message = str(transport._empty_response_error(ours, "refresh"))
        assert "but not that the teardown completed" in message
        assert "completed teardown" not in message
    finally:
        bridge.stop()


def test_a_rebind_does_not_inherit_the_previous_binds_record_1035_1(monkeypatch, tmp_path):
    """Review round 2 blocker: `start(); stop(); start()` on one object."""
    from _bridge_fakes import _load_bridge

    from bn import transport
    from bn.paths import stop_record_path

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)
    bridge, first = _started(module, "sx")
    bridge.stop()
    earlier = stop_record_path(bridge.registry_path).read_text()
    bridge.start()
    try:
        second = dataclasses.replace(first, meta={**first.meta, "bind_token": bridge.bind_token})
        assert first.meta["bind_token"] != bridge.bind_token
        # The new bind removed the earlier record ...
        for handle in (first, second):
            assert "no stop record was found" in str(
                transport._empty_response_error(handle, "x"))
        # ... and one put back is not this bind's.
        stop_record_path(bridge.registry_path).write_text(earlier)
        assert "is not this bridge's" in str(transport._empty_response_error(second, "x"))
    finally:
        bridge.stop()


def test_the_gui_bridge_writes_no_record_outside_the_gc_scan_1035_1(monkeypatch, tmp_path):
    """Review round 2 nit: the fixed pair is not in the instances directory."""
    from _bridge_fakes import _load_bridge

    from bn.paths import stop_record_path

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)
    gui = module.BinaryNinjaBridge()
    gui.bind_token = "b"
    assert gui._write_stop_record_locked("stopping") is False
    assert not stop_record_path(gui.registry_path).exists()


def test_a_reused_pid_is_not_reported_as_the_running_bridge_1035_1(tmp_path):
    """Review round 1: liveness is the recorded identity, not the pid number."""
    from bn import transport
    from bn.proc_identity import identity_payload

    live = identity_payload()
    if not live:
        pytest.skip("this host records no process identity")
    proven = _handle(tmp_path / "x.json", pid=os.getpid(), meta=live)
    reused = _handle(tmp_path / "x.json", pid=os.getpid(),
                     meta={**live, "pid_start_ticks": live["pid_start_ticks"] + 1})
    assert "still running" in str(transport._empty_response_error(proven, "x"))
    assert "no longer running" in str(transport._empty_response_error(reused, "x"))


def test_unreachable_instance_stop_advice_pastes_for_a_dash_id_1033_4(tmp_path):
    """Review round 1: the lifecycle-only refusal's `session stop` must parse."""
    import dataclasses

    from bn import transport

    handle = dataclasses.replace(_handle(tmp_path / "-t0116-parent.json", pid=os.getpid()),
                                 unreachable=True)
    with pytest.raises(BridgeError) as excinfo:
        transport._send_request_to_instance(handle, "ping", timeout=1.0)
    command = re.search(r"stop it with `(bn [^`]+)`", str(excinfo.value)).group(1)
    parsed = _parse(_paste(command, tmp_path))
    assert (parsed._command_path, parsed.instance_id) == (("session", "stop"), "-t0116-parent")


# --- #1036 item 7: project routing is disclosed; instance list --mine ---------


def _registered(instance_id: str, roots: list[str]) -> BridgeInstance:
    return BridgeInstance(
        pid=1, socket_path=Path(f"/nonexistent/{instance_id}.sock"),
        registry_path=Path(f"/nonexistent/{instance_id}.json"),
        plugin_name="bn_agent_bridge", plugin_version="0", started_at=None,
        meta={"project_roots": roots}, instance_token="t", instance_id=instance_id,
    )


def test_project_association_pick_is_disclosed_once_1036_7(monkeypatch, capsys, tmp_path):
    from bn import transport

    root = str(tmp_path)
    monkeypatch.setattr(transport, "project_root", lambda: Path(root))
    fleet = [_registered("-mine", [root]), _registered("other1", ["/elsewhere"]),
             _registered("other2", [])]
    monkeypatch.setattr(transport, "list_instances", lambda **k: fleet)
    monkeypatch.setattr(transport, "_associated_noted", set())
    assert transport.choose_instance().instance_id == "-mine"
    assert transport.choose_instance().instance_id == "-mine"
    err = capsys.readouterr().err
    assert err.count("note:") == 1
    assert "2 other instances are running" in err
    pasted = re.search(r"Pass (\S+) to make", err).group(1)
    assert _parse([pasted, "target", "list"]).instance == "-mine"


def test_association_pick_under_the_spawn_lock_is_disclosed_1036_7(
    monkeypatch, capsys, tmp_path,
):
    """Review round 1: the auto-spawn recheck selects by association too."""
    from bn import transport

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path / "cache"))
    instances_dir().mkdir(parents=True, exist_ok=True)
    root = str(tmp_path)
    monkeypatch.setattr(transport, "project_root", lambda: Path(root))
    fleet = [_registered("mine", [root]), _registered("other", [])]
    monkeypatch.setattr(transport, "list_instances", lambda **k: fleet)
    monkeypatch.setattr(transport, "_associated_noted", set())
    assert transport._auto_spawn_locked(timeout=5.0).instance_id == "mine"
    assert "note: using instance mine, associated with" in capsys.readouterr().err


def test_instance_list_mine_keeps_associated_and_pinned_1036_7(monkeypatch, capsys, tmp_path):
    from bn.commands import admin

    root = str(tmp_path)
    monkeypatch.setattr(admin, "project_root", lambda: Path(root))
    fleet = [_registered("mine", [root]), _registered("pinned", []),
             _registered("theirs", ["/elsewhere"])]
    monkeypatch.setattr(bn.cli, "list_instances", lambda: fleet)
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {"instance_id": "pinned"})
    assert bn.cli.main(["instance", "list", "--mine", "--format", "json"]) == 0
    listed = [row["instance_id"] for row in json.loads(capsys.readouterr().out)["items"]]
    assert listed == ["mine", "pinned"]
    assert bn.cli.main(["instance", "list", "--format", "json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["items"]) == 3
