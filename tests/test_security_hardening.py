"""Trust-boundary hardening for the bridge Unix socket (#612).

Covers the four independently-valuable defenses added in #612:

* ``paths.ensure_private_dir`` -- cache/instance/session/spill dirs are 0o700.
* Spill/registry/session files land at 0o600 regardless of umask.
* The bound Unix socket is chmod'd to 0o600.
* ``_check_peer_credentials`` rejects a foreign-uid peer (Linux SO_PEERCRED)
  before any op dispatch.

``py_exec`` runs arbitrary code unconditionally -- the socket's owner-only
trust boundary (peercred + 0o600 / 0o700) is the only gate.

No real BN, no dogfood binaries: all fakes + tmp paths.
"""
from __future__ import annotations

import io
import json
import logging
import os
import socket
import stat
import struct
import sys
from pathlib import Path

import pytest

from _bridge_fakes import _FakeBV, _load_bridge  # noqa: F401

import bn.paths as paths
import bn.output as output
import bn.session_state as session_state


# --------------------------------------------------------------------------
# ensure_private_dir: directory mode enforcement
# --------------------------------------------------------------------------

@pytest.fixture
def _restore_umask():
    old = os.umask(0o000)
    try:
        yield
    finally:
        os.umask(old)


def test_ensure_private_dir_creates_0o700_under_permissive_umask(tmp_path, _restore_umask):
    target = tmp_path / "cache" / "instances"
    result = paths.ensure_private_dir(target)
    assert result == target
    assert target.is_dir()
    # umask 0o000 would normally leave a fresh dir at 0o777; the helper forces 0o700.
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


def test_ensure_private_dir_tightens_existing_0o755_dir(tmp_path, _restore_umask):
    target = tmp_path / "pre-existing"
    target.mkdir(mode=0o755)
    os.chmod(target, 0o755)  # defeat umask masking so we truly start at 0o755
    assert stat.S_IMODE(target.stat().st_mode) == 0o755

    paths.ensure_private_dir(target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


def test_ensure_private_dir_is_idempotent_and_preserves_contents(tmp_path, _restore_umask):
    # Negative control: tightening an existing dir must not wipe its contents or
    # fail on a second call (exist_ok semantics preserved).
    target = tmp_path / "d"
    paths.ensure_private_dir(target)
    (target / "keep.txt").write_text("data")
    paths.ensure_private_dir(target)
    assert (target / "keep.txt").read_text() == "data"
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


def test_ensure_private_dir_warns_when_chmod_fails(tmp_path, monkeypatch, caplog):
    """#612 follow-up: a chmod failure on a sensitive dir must NOT fail open
    silently. On a filesystem/ACL that allows creation but rejects mode changes
    the dir stays group/world-accessible; that outcome has to be surfaced (a
    clear warning naming the path), not swallowed. The helper still keeps
    operating (returns the dir) -- it just no longer hides the weakened mode."""
    # A pre-existing 0o755 dir forces the tighten-to-0o700 chmod (a freshly
    # mkdir'd dir already lands at 0o700, so only this path exercises chmod).
    target = tmp_path / "pre-existing"
    target.mkdir(mode=0o755)
    os.chmod(target, 0o755)

    real_chmod = os.chmod

    def _refuse(path, mode, *a, **k):
        if Path(path) == target:
            raise OSError("operation not permitted")
        return real_chmod(path, mode, *a, **k)

    monkeypatch.setattr(paths.os, "chmod", _refuse)

    with caplog.at_level(logging.WARNING):
        result = paths.ensure_private_dir(target)

    assert result == target
    assert target.is_dir()  # keeps operating despite the failed tighten
    messages = [r.getMessage() for r in caplog.records]
    assert any(str(target) in m for m in messages), messages
    assert any("0o700" in m or "permission" in m.lower() for m in messages), messages


# --------------------------------------------------------------------------
# spill / registry / session file modes
# --------------------------------------------------------------------------

def test_spill_file_is_0o600_under_permissive_umask(tmp_path, monkeypatch, _restore_umask):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path / "cache"))
    # Force a spill: tiny limit so any output overflows to disk.
    big = {"items": ["x" * 100 for _ in range(50)]}
    result = output.write_output_result(
        big, fmt="json", out_path=None, stem="functions", spill_token_limit=1
    )
    assert result.spilled is True
    spill_path = Path(result.artifact["artifact_path"])
    assert spill_path.exists()
    # Under umask 0o000 a naive write_bytes would leave 0o666 (world-readable).
    assert stat.S_IMODE(spill_path.stat().st_mode) == 0o600
    # And the day dir itself is private.
    assert stat.S_IMODE(spill_path.parent.stat().st_mode) == 0o700


def test_session_state_file_is_0o600_under_umask_022(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path / "cache"))
    old = os.umask(0o022)
    try:
        session_state.update(instance_id="abc", target="demo")
    finally:
        os.umask(old)
    path = paths.session_state_path()
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_registry_file_is_0o600_under_umask_022(tmp_path, monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    instance.registry_path = tmp_path / "inst.json"
    old = os.umask(0o022)
    try:
        instance._write_registry()
    finally:
        os.umask(old)
    assert instance.registry_path.exists()
    assert stat.S_IMODE(instance.registry_path.stat().st_mode) == 0o600


# --------------------------------------------------------------------------
# socket mode after bind
# --------------------------------------------------------------------------

def test_socket_is_0o600_after_start(tmp_path, monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    sock_path = tmp_path / "bridge.sock"
    instance.socket_path = sock_path
    instance.registry_path = tmp_path / "bridge.json"

    old = os.umask(0o000)  # a permissive umask must not widen the socket
    try:
        instance.start()
        assert sock_path.exists()
        assert stat.S_IMODE(os.stat(sock_path).st_mode) == 0o600
    finally:
        os.umask(old)
        instance.stop()


def test_socket_chmod_failure_is_surfaced_not_silent(tmp_path, monkeypatch):
    """#612 follow-up: if tightening the bound Unix socket to 0o600 fails, the
    bridge must warn (naming the socket) rather than silently serve a
    world-accessible socket -- on non-Linux, where the SO_PEERCRED peer check is
    skipped, this socket mode is the PRIMARY cross-user boundary. The bridge
    keeps operating; it just no longer hides the weakened mode."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    sock_path = tmp_path / "bridge.sock"
    instance.socket_path = sock_path
    instance.registry_path = tmp_path / "bridge.json"

    warnings: list[str] = []
    monkeypatch.setattr(bridge.bn, "log_warn", lambda msg, *a, **k: warnings.append(str(msg)))

    real_chmod = os.chmod

    def _refuse(path, mode, *a, **k):
        if Path(path) == sock_path:
            raise OSError("permission denied")
        return real_chmod(path, mode, *a, **k)

    monkeypatch.setattr(bridge.os, "chmod", _refuse)

    try:
        instance.start()
        assert sock_path.exists()  # still serving
        assert any(str(sock_path) in w for w in warnings), warnings
    finally:
        instance.stop()


# --------------------------------------------------------------------------
# SO_PEERCRED peer authentication
# --------------------------------------------------------------------------

class _FakeConn:
    """Minimal stand-in for an accepted socket exposing getsockopt(SO_PEERCRED)."""

    def __init__(self, uid: int, *, pid: int = 4242, gid: int = 0, raise_os: bool = False):
        self._packed = struct.pack("3i", pid, uid, gid)
        self._raise = raise_os

    def getsockopt(self, level, optname, buflen):
        if self._raise:
            raise OSError("SO_PEERCRED unavailable")
        return self._packed[:buflen]


def test_check_peer_credentials_accepts_same_uid(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    assert bridge._check_peer_credentials(_FakeConn(os.getuid())) is None


def test_check_peer_credentials_rejects_foreign_uid(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    err = bridge._check_peer_credentials(_FakeConn(os.getuid() + 1))
    assert err is not None
    assert "uid" in err.lower()


def test_check_peer_credentials_fails_closed_when_peercred_unavailable(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    err = bridge._check_peer_credentials(_FakeConn(os.getuid(), raise_os=True))
    assert err is not None  # can't verify -> reject, never wave through


def test_check_peer_credentials_skipped_on_non_linux(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    # Even a mismatched uid is allowed (None) because the check is Linux-only.
    assert bridge._check_peer_credentials(_FakeConn(os.getuid() + 1)) is None


def test_check_peer_credentials_explicit_env_opt_out(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    monkeypatch.setenv("BN_SOCKET_ALLOW_ANY_UID", "1")
    assert bridge._check_peer_credentials(_FakeConn(os.getuid() + 1)) is None


class _RecordingBridge:
    def __init__(self):
        self.dispatch_calls = []

    def dispatch(self, payload):
        self.dispatch_calls.append(payload)
        return {"ok": True, "result": "pong"}

    def _begin_request(self, request_id):
        pass

    def _end_request(self, request_id):
        pass

    def _is_request_cancelled(self, request_id):
        return False


def _make_handler(bridge, connection, raw: bytes):
    """Build a BridgeHandler without socketserver setup, wired for handle()."""
    handler = bridge.BridgeHandler.__new__(bridge.BridgeHandler)
    handler.connection = connection
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()

    class _Server:
        pass

    server = _Server()
    server.bridge = _RecordingBridge()
    handler.server = server
    return handler


def test_handle_rejects_foreign_peer_before_dispatch(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    handler = _make_handler(
        bridge, _FakeConn(os.getuid() + 1), b'{"op": "ping"}\n'
    )
    handler.handle()

    # dispatch must never be reached for a rejected peer.
    assert handler.server.bridge.dispatch_calls == []
    response = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert response["ok"] is False
    assert "uid" in response["error"].lower()


def test_handle_allows_same_uid_peer_to_dispatch(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    handler = _make_handler(
        bridge, _FakeConn(os.getuid()), b'{"op": "ping"}\n'
    )
    handler.handle()

    # A matching uid proceeds exactly as before -- dispatch runs.
    assert handler.server.bridge.dispatch_calls == [{"op": "ping"}]


# --------------------------------------------------------------------------
# py_exec runs unconditionally (no env gate)
# --------------------------------------------------------------------------

def test_py_exec_runs_regardless_of_env(monkeypatch):
    # There is no BN_ALLOW_PY_EXEC gate: whoever reaches the owner-only socket
    # (peercred + 0o600 socket / 0o700 dir) is trusted to run arbitrary code.
    monkeypatch.delenv("BN_ALLOW_PY_EXEC", raising=False)
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._py_exec("active", "result = 21 * 2")
    assert result["result"] == 42
