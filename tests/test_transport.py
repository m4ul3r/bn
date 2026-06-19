from __future__ import annotations

import errno
import json
import os
import socket
import socketserver
import threading
import uuid
from pathlib import Path

import pytest

from bn.paths import bridge_registry_path, instances_dir
from bn.transport import (
    BridgeError,
    choose_instance,
    list_instances,
    send_request,
    spawn_instance,
    validate_instance_id,
    wait_for_teardown,
)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        raw = self.rfile.readline()
        if not raw:
            return
        payload = json.loads(raw.decode("utf-8"))
        response = {
            "ok": True,
            "result": {
                "op": payload["op"],
                "target": payload.get("target"),
                "params": payload.get("params"),
            },
        }
        self.wfile.write(json.dumps(response).encode("utf-8"))


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def test_send_request_uses_registry_and_socket(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    pid = os.getpid()
    socket_path = Path("/tmp") / f"bn-test-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
    registry_path = bridge_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    server = _Server(str(socket_path), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    registry_path.write_text(
        json.dumps(
            {
                "pid": pid,
                "socket_path": str(socket_path),
                "plugin_name": "bn_agent_bridge",
                "plugin_version": "0.1.0",
            }
        ),
        encoding="utf-8",
    )

    try:
        instances = list_instances()
        assert len(instances) == 1
        instance = choose_instance()
        assert instance.pid == pid

        response = send_request("ping", params={"hello": "world"}, target=f"{pid}:1:999")
        assert response["result"]["op"] == "ping"
        assert response["result"]["params"] == {"hello": "world"}
    finally:
        server.shutdown()
        server.server_close()


def test_list_instances_prunes_stale_registry_and_socket(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    registry_path = bridge_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    stale_socket_path = Path("/tmp") / f"bn-stale-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
    stale_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale_server.bind(str(stale_socket_path))
    stale_server.listen(1)
    stale_server.close()

    registry_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "socket_path": str(stale_socket_path),
                "plugin_name": "bn_agent_bridge",
                "plugin_version": "0.1.0",
            }
        ),
        encoding="utf-8",
    )

    assert stale_socket_path.exists()

    instances = list_instances()

    assert instances == []
    assert not registry_path.exists()
    # The dead socket file is swept too (a SIGKILL/crash leaves it behind).
    assert not stale_socket_path.exists()


def test_send_request_wraps_socket_errors(tmp_path, monkeypatch):
    from bn.transport import BridgeError, BridgeInstance

    instance = BridgeInstance(
        pid=999,
        socket_path=tmp_path / "missing.sock",
        registry_path=tmp_path / "missing.json",
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta={},
    )
    monkeypatch.setattr("bn.transport.choose_instance", lambda instance_id=None, **kw: instance)

    with pytest.raises(BridgeError, match="Failed to contact Binary Ninja bridge pid 999"):
        send_request("doctor")


def test_purge_keeps_log_sibling_for_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    inst_dir = instances_dir()
    inst_dir.mkdir(parents=True, exist_ok=True)

    registry_path = inst_dir / "ghost.json"
    socket_path = inst_dir / "ghost.sock"
    log_path = inst_dir / "ghost.log"

    # Bind+close leaves the socket file on disk but refuses connections,
    # mimicking a process that died without unlinking it.
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(socket_path))
    dead.listen(1)
    dead.close()

    log_path.write_text("native crash output\n", encoding="utf-8")
    registry_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "socket_path": str(socket_path),
                "instance_id": "ghost",
                "plugin_name": "bn_agent_bridge",
                "plugin_version": "0.1.0",
            }
        ),
        encoding="utf-8",
    )

    instances = list_instances()

    assert instances == []
    assert not registry_path.exists()
    assert not socket_path.exists()  # orphan socket swept
    assert log_path.exists()  # log preserved for diagnosis
    assert log_path.read_text(encoding="utf-8") == "native crash output\n"


def _empty_reply_socket():
    class _FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def settimeout(self, timeout):
            pass

        def connect(self, path):
            pass

        def sendall(self, payload):
            pass

        def shutdown(self, how):
            pass

        def recv(self, size):
            return b""

    return _FakeSocket()


def test_send_request_empty_response_reports_dead_process(tmp_path, monkeypatch):
    from bn.transport import BridgeInstance

    log_path = tmp_path / "ghost.log"
    log_path.write_text("segfault\n", encoding="utf-8")
    instance = BridgeInstance(
        pid=4242,
        socket_path=tmp_path / "bridge.sock",
        registry_path=tmp_path / "ghost.json",
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta={},
        instance_id="ghost",
    )
    monkeypatch.setattr("bn.transport.choose_instance", lambda instance_id=None, **kw: instance)
    monkeypatch.setattr("bn.transport._process_alive", lambda pid: False)
    monkeypatch.setattr("bn.transport.socket.socket", lambda *a, **k: _empty_reply_socket())

    with pytest.raises(BridgeError) as excinfo:
        send_request("decompile")

    msg = str(excinfo.value)
    assert "empty response" in msg
    assert "op 'decompile'" in msg
    assert "pid 4242" in msg
    assert "ghost" in msg
    assert "no longer running" in msg
    assert str(log_path) in msg


def test_send_request_empty_response_reports_live_process(tmp_path, monkeypatch):
    from bn.transport import BridgeInstance

    instance = BridgeInstance(
        pid=4242,
        socket_path=tmp_path / "bridge.sock",
        registry_path=tmp_path / "ghost.json",
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta={},
        instance_id="ghost",
    )
    monkeypatch.setattr("bn.transport.choose_instance", lambda instance_id=None, **kw: instance)
    monkeypatch.setattr("bn.transport._process_alive", lambda pid: True)
    monkeypatch.setattr("bn.transport.socket.socket", lambda *a, **k: _empty_reply_socket())

    with pytest.raises(BridgeError) as excinfo:
        send_request("decompile")

    msg = str(excinfo.value)
    assert "still running" in msg
    assert "pid 4242" in msg


def test_send_request_retries_transient_connect_failures(tmp_path, monkeypatch):
    from bn.transport import BridgeInstance

    instance = BridgeInstance(
        pid=999,
        socket_path=tmp_path / "bridge.sock",
        registry_path=tmp_path / "bridge.json",
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta={},
    )
    monkeypatch.setattr("bn.transport.choose_instance", lambda instance_id=None, **kw: instance)

    class _FakeSocket:
        attempts = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def settimeout(self, timeout):
            self.timeout = timeout

        def connect(self, path):
            type(self).attempts += 1
            if type(self).attempts == 1:
                raise ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")

        def sendall(self, payload):
            self.payload = payload

        def shutdown(self, how):
            self.how = how

        def recv(self, size):
            if not hasattr(self, "_sent"):
                self._sent = True
                return json.dumps({"ok": True, "result": {"pong": True}}).encode("utf-8")
            return b""

    monkeypatch.setattr("bn.transport.socket.socket", lambda *args, **kwargs: _FakeSocket())

    response = send_request("ping")

    assert response["result"]["pong"] is True
    assert _FakeSocket.attempts == 2


def _make_timeout_probe_socket():
    class _FakeSocket:
        timeouts: list[float] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def settimeout(self, timeout):
            type(self).timeouts.append(timeout)
            self.timeout = timeout

        def connect(self, path):
            self.path = path

        def sendall(self, payload):
            self.payload = payload

        def shutdown(self, how):
            self.how = how

        def recv(self, size):
            if not hasattr(self, "_sent"):
                self._sent = True
                return json.dumps({"ok": True, "result": {"pong": True}}).encode("utf-8")
            return b""

    return _FakeSocket


def _make_instance(tmp_path):
    from bn.transport import BridgeInstance

    return BridgeInstance(
        pid=999,
        socket_path=tmp_path / "bridge.sock",
        registry_path=tmp_path / "bridge.json",
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta={},
    )


def test_send_request_applies_default_timeout(tmp_path, monkeypatch):
    from bn.transport import DEFAULT_REQUEST_TIMEOUT

    monkeypatch.delenv("BN_REQUEST_TIMEOUT", raising=False)
    instance = _make_instance(tmp_path)
    monkeypatch.setattr("bn.transport.choose_instance", lambda instance_id=None, **kw: instance)
    fake_socket = _make_timeout_probe_socket()
    monkeypatch.setattr("bn.transport.socket.socket", lambda *args, **kwargs: fake_socket())

    response = send_request("ping")

    assert response["result"]["pong"] is True
    assert fake_socket.timeouts == [DEFAULT_REQUEST_TIMEOUT]


def test_send_request_timeout_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "42.5")
    instance = _make_instance(tmp_path)
    monkeypatch.setattr("bn.transport.choose_instance", lambda instance_id=None, **kw: instance)
    fake_socket = _make_timeout_probe_socket()
    monkeypatch.setattr("bn.transport.socket.socket", lambda *args, **kwargs: fake_socket())

    send_request("ping")

    assert fake_socket.timeouts == [42.5]


def test_send_request_timeout_env_zero_disables(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "0")
    instance = _make_instance(tmp_path)
    monkeypatch.setattr("bn.transport.choose_instance", lambda instance_id=None, **kw: instance)
    fake_socket = _make_timeout_probe_socket()
    monkeypatch.setattr("bn.transport.socket.socket", lambda *args, **kwargs: fake_socket())

    send_request("ping")

    assert fake_socket.timeouts == []


def test_resolve_timeout_rejects_non_numeric(monkeypatch):
    """A typo'd BN_REQUEST_TIMEOUT must fail loud, not silently fall back to the
    600s default (the user believes they set a short timeout but didn't) (#255)."""
    from bn.transport import _resolve_timeout, BridgeError

    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "abc")
    with pytest.raises(BridgeError) as exc:
        _resolve_timeout(None)
    assert "BN_REQUEST_TIMEOUT" in str(exc.value)


def test_resolve_timeout_rejects_negative(monkeypatch):
    """A negative value is not a valid socket timeout -- reject it instead of
    passing -1.0 through unvalidated (#255)."""
    from bn.transport import _resolve_timeout, BridgeError

    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "-1")
    with pytest.raises(BridgeError) as exc:
        _resolve_timeout(None)
    assert "BN_REQUEST_TIMEOUT" in str(exc.value)


def test_resolve_timeout_rejects_non_finite(monkeypatch):
    """inf/nan parse as floats but aren't valid socket timeouts -- reject them
    like negatives rather than passing them to settimeout (#255)."""
    from bn.transport import _resolve_timeout, BridgeError

    for val in ("inf", "-inf", "nan"):
        monkeypatch.setenv("BN_REQUEST_TIMEOUT", val)
        with pytest.raises(BridgeError):
            _resolve_timeout(None)


def test_resolve_timeout_zero_and_sentinels_disable(monkeypatch):
    """0 / 0.0 / none / off / empty all disable the timeout (return None)."""
    from bn.transport import _resolve_timeout

    for val in ("0", "0.0", "none", "off", "", "  Off  "):
        monkeypatch.setenv("BN_REQUEST_TIMEOUT", val)
        assert _resolve_timeout(None) is None, val


def test_resolve_timeout_rejects_float_underflow(monkeypatch):
    """A tiny magnitude that underflows to +/-0.0 (e.g. 1e-325, -1e-325) is NOT a
    real 0/disable request: a positive value the user set shouldn't silently turn
    the timeout off, and a negative one must be rejected like any other negative.
    `value < 0` misses -0.0, and `value or None` collapses +0.0 to disable, so
    both slipped through before (#255 review)."""
    from bn.transport import _resolve_timeout, BridgeError

    for val in ("1e-325", "-1e-325", "-0.0"):
        monkeypatch.setenv("BN_REQUEST_TIMEOUT", val)
        with pytest.raises(BridgeError):
            _resolve_timeout(None)


def test_resolve_timeout_positive_default_and_explicit(monkeypatch):
    from bn.transport import _resolve_timeout, DEFAULT_REQUEST_TIMEOUT

    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "42.5")
    assert _resolve_timeout(None) == 42.5
    monkeypatch.delenv("BN_REQUEST_TIMEOUT", raising=False)
    assert _resolve_timeout(None) == DEFAULT_REQUEST_TIMEOUT
    # an explicit timeout arg always wins and is never re-validated against the env
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "abc")
    assert _resolve_timeout(12.0) == 12.0


def test_invalid_timeout_is_rejected_before_choosing_an_instance(monkeypatch, tmp_path):
    # #265 review: an invalid BN_REQUEST_TIMEOUT must be rejected BEFORE
    # send_request() calls choose_instance() -- which auto-spawns a headless
    # bridge. Otherwise `BN_REQUEST_TIMEOUT=abc bn target list` errors out but
    # leaves a stray random instance behind in a fresh cache.
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "abc")

    def _must_not_spawn(*a, **k):
        raise AssertionError("choose_instance reached before timeout validation")

    monkeypatch.setattr("bn.transport.choose_instance", _must_not_spawn)

    with pytest.raises(BridgeError, match="BN_REQUEST_TIMEOUT"):
        send_request("list_targets")

    # Fresh cache stays empty: nothing was spawned.
    assert list_instances() == []


def test_send_request_partial_response_reports_real_error(tmp_path, monkeypatch):
    from bn.transport import BridgeError

    instance = _make_instance(tmp_path)
    monkeypatch.setattr("bn.transport.choose_instance", lambda instance_id=None, **kw: instance)

    class _FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def settimeout(self, timeout):
            self.timeout = timeout

        def connect(self, path):
            self.path = path

        def sendall(self, payload):
            self.payload = payload

        def shutdown(self, how):
            self.how = how

        def recv(self, size):
            if not hasattr(self, "_sent"):
                self._sent = True
                return b'{"ok": true, "resu'
            raise socket.timeout("timed out")

    monkeypatch.setattr("bn.transport.socket.socket", lambda *args, **kwargs: _FakeSocket())

    with pytest.raises(BridgeError, match="failed mid-response for op 'ping' after 18 bytes"):
        send_request("ping", timeout=5.0)


def test_send_request_reports_timeout_waiting_for_response(tmp_path, monkeypatch):
    from bn.transport import BridgeError, BridgeInstance

    instance = BridgeInstance(
        pid=999,
        socket_path=tmp_path / "bridge.sock",
        registry_path=tmp_path / "bridge.json",
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta={},
    )
    monkeypatch.setattr("bn.transport.choose_instance", lambda instance_id=None, **kw: instance)

    class _FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def settimeout(self, timeout):
            self.timeout = timeout

        def connect(self, path):
            self.path = path

        def sendall(self, payload):
            self.payload = payload

        def shutdown(self, how):
            self.how = how

        def recv(self, size):
            raise socket.timeout("timed out")

    monkeypatch.setattr("bn.transport.socket.socket", lambda *args, **kwargs: _FakeSocket())

    with pytest.raises(BridgeError, match="Timed out waiting for Binary Ninja bridge pid 999"):
        send_request("ping", timeout=12.5)


def test_list_instances_trusts_live_socket_even_with_stale_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    registry_path = bridge_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    socket_path = Path("/tmp") / f"bn-live-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
    server = _Server(str(socket_path), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    registry_path.write_text(
        json.dumps(
            {
                "pid": 111,
                "socket_path": str(socket_path),
                "plugin_name": "bn_agent_bridge",
                "plugin_version": "0.1.0",
            }
        ),
        encoding="utf-8",
    )

    try:
        instances = list_instances()

        assert len(instances) == 1
        assert instances[0].pid == 111
        assert registry_path.exists()
    finally:
        server.shutdown()
        server.server_close()


def test_list_instances_reads_fixed_registry_path(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    pid = os.getpid()
    socket_path = Path("/tmp") / f"bn-fixed-{pid}-{uuid.uuid4().hex[:8]}.sock"
    server = _Server(str(socket_path), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    registry_path = bridge_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "pid": pid,
                "socket_path": str(socket_path),
                "plugin_name": "bn_agent_bridge",
                "plugin_version": "0.1.0",
            }
        ),
        encoding="utf-8",
    )

    try:
        instances = list_instances()

        assert len(instances) == 1
        assert instances[0].pid == pid
        assert instances[0].registry_path == registry_path
    finally:
        server.shutdown()
        server.server_close()


def _create_live_instance(tmp_path, instance_id, *, subdir="instances"):
    """Helper: start a mock server and write a registry file, return server."""
    inst_dir = tmp_path / subdir
    inst_dir.mkdir(parents=True, exist_ok=True)
    socket_path = Path("/tmp") / f"bn-inst-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
    server = _Server(str(socket_path), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    registry_path = inst_dir / f"{instance_id}.json"
    registry_path.write_text(
        json.dumps({
            "pid": os.getpid(),
            "socket_path": str(socket_path),
            "plugin_name": "bn_agent_bridge",
            "plugin_version": "0.1.0",
            "instance_id": instance_id,
        }),
        encoding="utf-8",
    )
    return server


def test_list_instances_discovers_instance_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    srv_a = _create_live_instance(tmp_path, "aaaa1111")
    srv_b = _create_live_instance(tmp_path, "bbbb2222")
    try:
        instances = list_instances()
        ids = {inst.instance_id for inst in instances}
        assert "aaaa1111" in ids
        assert "bbbb2222" in ids
        assert len(instances) >= 2
    finally:
        srv_a.shutdown()
        srv_a.server_close()
        srv_b.shutdown()
        srv_b.server_close()


def test_choose_instance_by_id(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    srv_a = _create_live_instance(tmp_path, "aaaa1111")
    srv_b = _create_live_instance(tmp_path, "bbbb2222")
    try:
        inst = choose_instance("bbbb2222", auto_start=False)
        assert inst.instance_id == "bbbb2222"

        inst = choose_instance("aaaa1111", auto_start=False)
        assert inst.instance_id == "aaaa1111"
    finally:
        srv_a.shutdown()
        srv_a.server_close()
        srv_b.shutdown()
        srv_b.server_close()


def test_choose_instance_by_default_selects_fixed_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    pid = os.getpid()
    socket_path = Path("/tmp") / f"bn-default-{pid}-{uuid.uuid4().hex[:8]}.sock"
    server = _Server(str(socket_path), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    registry_path = bridge_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "pid": pid,
                "socket_path": str(socket_path),
                "plugin_name": "bn_agent_bridge",
                "plugin_version": "0.1.0",
            }
        ),
        encoding="utf-8",
    )

    srv = _create_live_instance(tmp_path, "aaaa1111")
    try:
        inst = choose_instance("default", auto_start=False)
        assert inst.instance_id is None
        assert inst.registry_path == registry_path
    finally:
        server.shutdown()
        server.server_close()
        srv.shutdown()
        srv.server_close()


def test_choose_instance_requires_id_when_multiple_instances_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    srv_a = _create_live_instance(tmp_path, "aaaa1111")
    srv_b = _create_live_instance(tmp_path, "bbbb2222")
    try:
        with pytest.raises(BridgeError, match="Multiple Binary Ninja bridge instances are running") as exc:
            choose_instance(auto_start=False)
        message = str(exc.value)
        assert "--instance <id>" in message
        assert "aaaa1111" in message
        assert "bbbb2222" in message
    finally:
        srv_a.shutdown()
        srv_a.server_close()
        srv_b.shutdown()
        srv_b.server_close()


def test_choose_instance_no_match_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    srv = _create_live_instance(tmp_path, "aaaa1111")
    try:
        with pytest.raises(BridgeError, match="No bridge instance found with id: missing"):
            choose_instance("missing", auto_start=False)
    finally:
        srv.shutdown()
        srv.server_close()


def test_list_instances_prunes_stale_in_instances_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    inst_dir = instances_dir()
    inst_dir.mkdir(parents=True, exist_ok=True)

    stale_socket = Path("/tmp") / f"bn-stale-inst-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
    stale_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale_server.bind(str(stale_socket))
    stale_server.listen(1)
    stale_server.close()

    registry_path = inst_dir / "deadbeef.json"
    registry_path.write_text(
        json.dumps({
            "pid": os.getpid(),
            "socket_path": str(stale_socket),
            "plugin_name": "bn_agent_bridge",
            "plugin_version": "0.1.0",
            "instance_id": "deadbeef",
        }),
        encoding="utf-8",
    )

    instances = list_instances()
    assert not any(inst.instance_id == "deadbeef" for inst in instances)
    assert not registry_path.exists()


def test_choose_instance_no_auto_start_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    with pytest.raises(BridgeError, match="No running Binary Ninja bridge instances found"):
        choose_instance(auto_start=False)


def test_spawn_instance_rejects_duplicate_id(monkeypatch, tmp_path):
    from bn.transport import BridgeInstance

    existing = BridgeInstance(
        pid=123,
        socket_path=tmp_path / "existing.sock",
        registry_path=tmp_path / "existing.json",
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta={},
        instance_id="aaaa1111",
    )
    monkeypatch.setattr("bn.transport.list_instances", lambda: [existing])

    with pytest.raises(BridgeError, match="Bridge instance already exists with id: aaaa1111"):
        spawn_instance("aaaa1111")


def test_spawn_instance_starts_new_instance_when_other_instances_exist(monkeypatch, tmp_path):
    from bn.transport import BridgeInstance

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    existing = BridgeInstance(
        pid=123,
        socket_path=tmp_path / "existing.sock",
        registry_path=tmp_path / "existing.json",
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta={},
        instance_id="aaaa1111",
    )
    created = BridgeInstance(
        pid=456,
        socket_path=tmp_path / "new.sock",
        registry_path=bridge_registry_path("newid"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta={},
        instance_id="newid",
    )
    bridge_registry_path("newid").parent.mkdir(parents=True, exist_ok=True)
    bridge_registry_path("newid").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("bn.transport.list_instances", lambda: [existing])
    monkeypatch.setattr("bn.transport._find_bn_agent", lambda: ["bn-agent"])
    monkeypatch.setattr("bn.transport._load_instance", lambda path: created)

    popen_calls = []

    class _FakePopen:
        pid = 456

        def __init__(self, cmd, **kwargs):
            popen_calls.append({"cmd": cmd, **kwargs})

    monkeypatch.setattr("bn.transport.subprocess.Popen", _FakePopen)

    inst = spawn_instance("newid")

    assert inst.instance_id == "newid"
    assert popen_calls[0]["cmd"] == ["bn-agent", "--instance-id", "newid"]


def test_spawn_instance_reports_exit_code_and_log_when_child_dies(monkeypatch, tmp_path):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("bn.transport.list_instances", lambda: [])
    monkeypatch.setattr("bn.transport._find_bn_agent", lambda: ["bn-agent"])

    class _FakePopen:
        pid = 456

        def __init__(self, cmd, **kwargs):
            # Write to the log file handle just like a crashing child would.
            kwargs["stdout"].write(
                "Traceback (most recent call last):\nImportError: no module named binaryninja\n"
            )

        def poll(self):
            return 3

    monkeypatch.setattr("bn.transport.subprocess.Popen", _FakePopen)

    with pytest.raises(BridgeError) as excinfo:
        spawn_instance("deadkid")

    msg = str(excinfo.value)
    assert "exited with code 3" in msg
    assert "ImportError: no module named binaryninja" in msg
    assert str(instances_dir() / "deadkid.log") in msg


def test_spawn_instance_terminates_child_on_registration_timeout(monkeypatch, tmp_path):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("bn.transport.list_instances", lambda: [])
    monkeypatch.setattr("bn.transport._find_bn_agent", lambda: ["bn-agent"])

    lifecycle: list[str] = []

    class _FakePopen:
        pid = 456

        def __init__(self, cmd, **kwargs):
            pass

        def poll(self):
            return None

        def terminate(self):
            lifecycle.append("terminate")

        def wait(self, timeout=None):
            lifecycle.append("wait")
            return 0

        def kill(self):  # pragma: no cover - only reached if wait() raised
            lifecycle.append("kill")

    monkeypatch.setattr("bn.transport.subprocess.Popen", _FakePopen)

    with pytest.raises(BridgeError, match="did not register within .* and was terminated"):
        spawn_instance("slowkid", timeout=0.05, poll_interval=0.01)

    assert lifecycle[:2] == ["terminate", "wait"]


def test_choose_instance_spawn_missing_named_spawns_new(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    spawned = {}
    sentinel = object()

    def fake_spawn(instance_id):
        spawned["id"] = instance_id
        return sentinel

    monkeypatch.setattr("bn.transport.spawn_instance", fake_spawn)

    # opt-in (the `bn load` path): a missing named id is spawned, not an error.
    result = choose_instance("brandnew", spawn_missing_named=True)
    assert result is sentinel
    assert spawned["id"] == "brandnew"

    # default: a missing named id still fails fast, and the error points the way.
    with pytest.raises(BridgeError, match="session start --instance-id brandnew") as exc:
        choose_instance("brandnew")
    assert "No bridge instance found with id: brandnew" in str(exc.value)


# ---------------------------------------------------------------------------
# #84 — instance-id validation (path traversal)
# ---------------------------------------------------------------------------


def test_validate_instance_id_accepts_safe_basenames():
    for good in ("abc123", "goal-v1", "my_inst.2", "A.B-C_9", "default"):
        assert validate_instance_id(good) == good


@pytest.mark.parametrize(
    "bad",
    [
        "../evil",
        "../../tmp/evil",
        "/abs/path",
        "a/b",
        "a\\b",
        ".",
        "..",
        "",
        "has space",
        "weird;id",
    ],
)
def test_validate_instance_id_rejects_traversal_and_separators(bad):
    with pytest.raises(BridgeError, match="Invalid instance id|non-empty"):
        validate_instance_id(bad)


def test_spawn_instance_rejects_traversal_id_before_any_fs(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    # Must raise before spawning anything; no files outside instances_dir().
    popen_called = {"n": 0}
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: popen_called.__setitem__("n", popen_called["n"] + 1))
    with pytest.raises(BridgeError, match="Invalid instance id"):
        spawn_instance("../../tmp/evil")
    assert popen_called["n"] == 0
    # No escaped registry/socket/log created anywhere under tmp_path's parent.
    assert not (tmp_path.parent / "tmp" / "evil.json").exists()


def test_choose_instance_rejects_traversal_id(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    with pytest.raises(BridgeError, match="Invalid instance id"):
        choose_instance("../escape")


# ---------------------------------------------------------------------------
# #92 — spawn pid verification + stop-teardown convergence
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, pid, *, on_start=None):
        self.pid = pid
        self._terminated = False
        if on_start is not None:
            on_start()

    def poll(self):
        return None

    def terminate(self):
        self._terminated = True

    def kill(self):
        self._terminated = True

    def wait(self, timeout=None):
        return 0


def test_spawn_rejects_registry_owned_by_other_pid(tmp_path, monkeypatch):
    # The registry that appears is owned by a DIFFERENT pid than our child:
    # spawn must reap our child and refuse, not return a stranger's bridge (#92).
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("bn.transport._find_bn_agent", lambda: ["/bin/true"])

    inst_dir = instances_dir()
    inst_dir.mkdir(parents=True, exist_ok=True)
    socket_path = Path("/tmp") / f"bn-pidtest-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
    server = _Server(str(socket_path), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    other_pid = os.getpid()  # a live pid that is NOT our spawned child's pid
    our_child_pid = 2_000_000_000  # implausible, definitely != other_pid

    def write_registry():
        (inst_dir / "racer.json").write_text(
            json.dumps({
                "pid": other_pid,
                "socket_path": str(socket_path),
                "plugin_name": "bn_agent_bridge",
                "instance_id": "racer",
            }),
            encoding="utf-8",
        )

    # Construct the fake child LAZILY (when Popen is called inside spawn), so the
    # racer registry is written DURING the spawn wait, not at test-setup time.
    holder = {}

    def fake_popen(*a, **k):
        holder["proc"] = _FakeProc(our_child_pid, on_start=write_registry)
        return holder["proc"]

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    try:
        with pytest.raises(BridgeError, match="already owned by another process"):
            spawn_instance("racer")
        assert holder["proc"]._terminated  # our orphan child was reaped
    finally:
        server.shutdown()
        server.server_close()


def test_wait_for_teardown_converges_when_process_and_registry_gone(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    from bn.transport import BridgeInstance

    inst = BridgeInstance(
        pid=2_000_000_001,  # not a live pid
        socket_path=tmp_path / "gone.sock",
        registry_path=tmp_path / "gone.json",  # never created -> _load_instance None
        plugin_name="bn_agent_bridge",
        plugin_version="0",
        started_at=None,
        meta={},
        instance_id="gone",
    )
    assert wait_for_teardown(inst, timeout=1.0) is True


def test_wait_for_teardown_times_out_while_live(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    from bn.transport import BridgeInstance

    socket_path = Path("/tmp") / f"bn-live-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
    server = _Server(str(socket_path), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    reg = tmp_path / "live.json"
    reg.write_text(
        json.dumps({"pid": os.getpid(), "socket_path": str(socket_path), "instance_id": "live"}),
        encoding="utf-8",
    )
    inst = BridgeInstance(
        pid=os.getpid(),  # this test process is alive
        socket_path=socket_path,
        registry_path=reg,
        plugin_name="bn_agent_bridge",
        plugin_version="0",
        started_at=None,
        meta={},
        instance_id="live",
    )
    try:
        assert wait_for_teardown(inst, timeout=0.3) is False
    finally:
        server.shutdown()
        server.server_close()
