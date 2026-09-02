from __future__ import annotations

import errno
import json
import os
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from bn.paths import bridge_registry_path, instances_dir
from bn.transport import (
    BridgeError,
    choose_instance,
    gc_instances,
    list_instances,
    send_request,
    spawn_instance,
    validate_instance_id,
    wait_for_teardown,
)


def test_choose_instance_multiple_with_no_selector_errors_before_target(monkeypatch):
    # #368 facet 2: with several live instances and NO selector or project
    # association, resolution must surface workspace ambiguity before target
    # resolution runs.
    import types
    import bn.transport as t
    insts = [types.SimpleNamespace(instance_id="aa11", pid=11, socket_path="/s/aa11", started_at=None),
             types.SimpleNamespace(instance_id="bb22", pid=22, socket_path="/s/bb22", started_at=None)]
    monkeypatch.setattr(t, "list_instances", lambda **kwargs: insts)
    monkeypatch.setattr(t, "_resolve_from_project_roots", lambda instances: None)
    with pytest.raises(BridgeError) as exc:
        choose_instance(auto_start=False)
    assert "Multiple" in str(exc.value) and "instance" in str(exc.value).lower()


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        raw = self.rfile.readline()
        if not raw:
            return
        payload = json.loads(raw.decode("utf-8"))
        self.server.requests.append(payload)
        response = {
            "ok": True,
            "result": {
                "op": payload["op"],
                "target": payload.get("target"),
                "params": payload.get("params"),
            },
            "bridge_identity": getattr(
                self.server, "bridge_identity", payload.get("_bridge_identity")
            ),
        }
        self.wfile.write(json.dumps(response).encode("utf-8"))


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    def __init__(self, *args, **kwargs):
        self.requests = []
        self.bridge_identity = None
        super().__init__(*args, **kwargs)

    def server_close(self):
        # Closing an AF_UNIX socket does not remove its filesystem pathname;
        # unlink it so tests never leave stale *.sock files behind (#563).
        super().server_close()
        Path(self.server_address).unlink(missing_ok=True)


def test_send_request_uses_registry_and_socket(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    pid = os.getpid()
    socket_path = tmp_path / "bn-test.sock"
    registry_path = bridge_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    server = _Server(str(socket_path), _Handler)
    token = "test-instance-token"
    server.bridge_identity = {
        "instance_id": None,
        "pid": pid,
        "token": token,
    }
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    registry_path.write_text(
        json.dumps(
            {
                "pid": pid,
                "socket_path": str(socket_path),
                "plugin_name": "bn_agent_bridge",
                "plugin_version": "0.1.0",
                "instance_token": token,
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



def test_send_request_rejects_foreign_response_identity(tmp_path, monkeypatch):
    import bn.transport as transport

    socket_path = tmp_path / "foreign.sock"
    server = _Server(str(socket_path), _Handler)
    server.bridge_identity = {
        "instance_id": "foreign",
        "pid": os.getpid(),
        "token": "foreign-token",
    }
    threading.Thread(target=server.serve_forever, daemon=True).start()
    instance = transport.BridgeInstance(
        pid=os.getpid(),
        socket_path=socket_path,
        registry_path=tmp_path / "expected.json",
        plugin_name="bn_agent_bridge",
        plugin_version="1",
        started_at=None,
        meta={},
        instance_id="expected",
        instance_token="expected-token",
    )
    monkeypatch.setattr(transport, "choose_instance", lambda *args, **kwargs: instance)

    try:
        with pytest.raises(BridgeError, match="bridge identity mismatch"):
            send_request("target_info", instance_id="expected")
    finally:
        server.shutdown()
        server.server_close()


def test_send_request_rejects_tokenless_registry_before_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    socket_path = tmp_path / "legacy.sock"
    server = _Server(str(socket_path), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    registry = bridge_registry_path("legacy")
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "socket_path": str(socket_path),
                "instance_id": "legacy",
            }
        ),
        encoding="utf-8",
    )

    try:
        with pytest.raises(BridgeError, match="identity token"):
            send_request("target_info", instance_id="legacy")
        assert server.requests == []
    finally:
        server.shutdown()
        server.server_close()


def test_send_request_rejects_foreign_socket_pid_before_dispatch(tmp_path, monkeypatch):
    import bn.transport as transport

    socket_path = tmp_path / "foreign-pid.sock"
    server = _Server(str(socket_path), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    instance = transport.BridgeInstance(
        pid=os.getpid() + 100000,
        socket_path=socket_path,
        registry_path=tmp_path / "expected.json",
        plugin_name="bn_agent_bridge",
        plugin_version="1",
        started_at=None,
        meta={},
        instance_id="expected",
        instance_token="expected-token",
    )
    monkeypatch.setattr(transport, "choose_instance", lambda *args, **kwargs: instance)

    try:
        with pytest.raises(BridgeError, match="socket peer pid"):
            send_request("load_binary", instance_id="expected")
        assert server.requests == []
    finally:
        server.shutdown()
        server.server_close()


def test_list_instances_rejects_registry_filename_identity_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    socket_path = tmp_path / "foreign.sock"
    server = _Server(str(socket_path), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    registry = instances_dir() / "expected.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "socket_path": str(socket_path),
                "instance_id": "foreign",
                "instance_token": "foreign-token",
            }
        ),
        encoding="utf-8",
    )

    try:
        assert list_instances() == []
        assert not registry.exists()
        assert socket_path.exists()
    finally:
        server.shutdown()
        server.server_close()

def test_list_instances_prunes_stale_registry_and_socket(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("bn.transport._process_alive", lambda pid: False)
    registry_path = bridge_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    stale_socket_path = tmp_path / "bn-stale.sock"
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

    try:
        assert stale_socket_path.exists()

        instances = list_instances()

        assert instances == []
        assert not registry_path.exists()
        # The dead socket file is swept too (a SIGKILL/crash leaves it behind).
        assert not stale_socket_path.exists()
    finally:
        # list_instances() is expected to sweep the stale socket; if the test
        # fails before that, don't leave the pathname behind.
        stale_socket_path.unlink(missing_ok=True)


def test_list_instances_preserves_live_stopped_bridge_when_probe_fails(
    tmp_path, monkeypatch
):
    import bn.transport as transport

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    registry_path = bridge_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path = tmp_path / "stopped.sock"
    socket_path.touch()
    registry_path.write_text(
        json.dumps(
            {
                "pid": 4242,
                "socket_path": str(socket_path),
                "plugin_name": "bn_agent_bridge",
                "plugin_version": "0.1.0",
                "instance_token": "stopped-token",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(transport, "_socket_is_live", lambda *args, **kwargs: False)
    monkeypatch.setattr(transport, "_process_alive", lambda pid: True)
    monkeypatch.setattr(transport, "_process_state", lambda pid: "T")

    instances = list_instances()

    assert len(instances) == 1
    assert instances[0].pid == 4242
    assert registry_path.exists()
    assert socket_path.exists()


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
        instance_token="test-token",
    )
    monkeypatch.setattr("bn.transport.choose_instance", lambda instance_id=None, **kw: instance)

    with pytest.raises(BridgeError, match="Failed to contact Binary Ninja bridge pid 999"):
        send_request("doctor")


def test_purge_keeps_log_sibling_for_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("bn.transport._process_alive", lambda pid: False)
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


def test_gc_instances_removes_dead_logs_and_orphans_keeps_live(tmp_path, monkeypatch):
    # #80 cache hygiene: the lazy purge keeps dead instances' .log breadcrumbs
    monkeypatch.setattr("bn.transport._process_alive", lambda pid: False)
    # forever (147 zero-byte logs measured on a real host). `gc_instances()`
    # reaps logs + orphan sockets of dead/gone instances while leaving a live
    # instance and the spawn lock untouched.
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    inst_dir = instances_dir()
    inst_dir.mkdir(parents=True, exist_ok=True)

    # Dead instance: registry+socket present but the socket refuses connections.
    dead_reg = inst_dir / "dead.json"
    dead_sock = inst_dir / "dead.sock"
    dead_log = inst_dir / "dead.log"
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(dead_sock)); dead.listen(1); dead.close()
    dead_log.write_text("crash output\n", encoding="utf-8")
    dead_reg.write_text(json.dumps({
        "pid": os.getpid(), "socket_path": str(dead_sock), "instance_id": "dead",
        "plugin_name": "bn_agent_bridge", "plugin_version": "0",
    }), encoding="utf-8")

    # Orphans with no registry at all (the long-dead zero-byte log + stale sock).
    orphan_log = inst_dir / "longgone.log"
    orphan_log.write_text("", encoding="utf-8")
    orphan_sock = inst_dir / "longgone.sock"
    orphan_sock.write_text("", encoding="utf-8")

    # Live instance: a real listening socket kept open across the gc call.
    live_reg = inst_dir / "live.json"
    live_sock = inst_dir / "live.sock"
    live_log = inst_dir / "live.log"
    live_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    live_server.bind(str(live_sock)); live_server.listen(1)
    live_log.write_text("serving\n", encoding="utf-8")
    live_reg.write_text(json.dumps({
        "pid": os.getpid(), "socket_path": str(live_sock), "instance_id": "live",
        "plugin_name": "bn_agent_bridge", "plugin_version": "0",
    }), encoding="utf-8")

    # The active spawn lock must never be reaped.
    spawn_lock = inst_dir / ".spawn.lock"
    spawn_lock.write_text("", encoding="utf-8")

    try:
        result = gc_instances()
    finally:
        live_server.close()

    # Live instance fully preserved.
    assert live_reg.exists() and live_sock.exists() and live_log.exists()
    assert live_log.read_text(encoding="utf-8") == "serving\n"
    # Dead instance fully reaped (registry+socket by the liveness sweep, log by gc).
    assert not dead_reg.exists() and not dead_sock.exists() and not dead_log.exists()
    # Registry-less orphans reaped.
    assert not orphan_log.exists() and not orphan_sock.exists()
    # Spawn lock preserved.
    assert spawn_lock.exists()
    # Honest report.
    assert result["live_instances"] == 1
    assert result["registries_purged"] == 1          # dead.json
    assert result["logs_removed"] == 2               # dead.log + longgone.log
    assert result["sockets_removed"] == 1            # longgone.sock (dead.sock swept by liveness)


def test_gc_instances_on_empty_or_missing_dir_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path / "fresh"))
    result = gc_instances()
    assert result == {
        "live_instances": 0, "registries_purged": 0, "logs_removed": 0,
        "sockets_removed": 0, "removed": [],
    }


def test_gc_instances_keeps_siblings_of_unparseable_registry(tmp_path, monkeypatch):
    # A registry that fails to parse is NOT purged by the liveness sweep, so its
    # stem stays "live"; gc must keep its .log/.sock (err toward keeping rather
    # than reaping an instance we simply couldn't read).
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    inst_dir = instances_dir()
    inst_dir.mkdir(parents=True, exist_ok=True)
    bad_reg = inst_dir / "weird.json"
    bad_reg.write_text("{ not valid json", encoding="utf-8")
    bad_log = inst_dir / "weird.log"
    bad_log.write_text("x", encoding="utf-8")
    bad_sock = inst_dir / "weird.sock"
    bad_sock.write_text("", encoding="utf-8")

    result = gc_instances()

    assert bad_reg.exists() and bad_log.exists() and bad_sock.exists()
    assert result["logs_removed"] == 0 and result["sockets_removed"] == 0


def test_gc_holds_spawn_lock_so_it_cannot_reap_an_in_flight_spawn(tmp_path, monkeypatch):
    # #80 review: a spawn creates <id>.log + <id>.sock BEFORE it writes the
    # registry. gc must hold _spawn_lock so it can't reap those live files
    # mid-spawn. Hold the lock (simulating an in-flight spawn) and assert gc
    # blocks and leaves the in-flight files untouched while the lock is held.
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    from bn.transport import _spawn_lock

    inst_dir = instances_dir()
    inst_dir.mkdir(parents=True, exist_ok=True)
    spawning_log = inst_dir / "spawning.log"
    spawning_log.write_text("", encoding="utf-8")
    spawning_sock = inst_dir / "spawning.sock"
    spawning_sock.write_text("", encoding="utf-8")

    box: dict = {}

    def run_gc():
        box["result"] = gc_instances()

    worker = threading.Thread(target=run_gc)
    with _spawn_lock():
        worker.start()
        worker.join(timeout=0.5)
        # gc is blocked on the spawn lock -> the in-flight files survive.
        assert worker.is_alive()
        assert spawning_log.exists() and spawning_sock.exists()
    # Lock released: gc proceeds and completes (now safely reaping the orphans,
    # since no spawn is in flight once the lock is free).
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert "result" in box


def test_spawn_lock_timeout_is_bounded_and_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    from bn.transport import _spawn_lock

    errors = []

    def contend():
        try:
            with _spawn_lock(timeout=0.05):
                raise AssertionError("contender unexpectedly acquired lock")
        except BaseException as exc:
            errors.append(exc)

    with _spawn_lock():
        worker = threading.Thread(target=contend)
        worker.start()
        worker.join(timeout=1)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], BridgeError)
    assert "spawn lock" in str(errors[0]).lower()
    assert "0.05s" in str(errors[0])


def test_send_request_budget_covers_spawn_lock_contention(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    from bn.transport import _spawn_lock

    started = time.monotonic()
    with _spawn_lock():
        with pytest.raises(BridgeError, match="Timed out"):
            send_request("ping", timeout=0.01)

    assert time.monotonic() - started < 0.1


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
        instance_token="test-token",
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
        instance_token="test-token",
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
        instance_token="test-token",
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
                request = json.loads(self.payload.decode("utf-8"))
                return json.dumps(
                    {
                        "ok": True,
                        "result": {"pong": True},
                        "bridge_identity": request["_bridge_identity"],
                    }
                ).encode("utf-8")
            return b""

    monkeypatch.setattr("bn.transport.socket.socket", lambda *args, **kwargs: _FakeSocket())

    response = send_request("ping")

    assert response["result"]["pong"] is True
    assert _FakeSocket.attempts == 2


def test_transient_connect_retries_obey_end_to_end_deadline(
    tmp_path, monkeypatch
):
    from bn.transport import BridgeInstance

    instance = BridgeInstance(
        pid=999,
        socket_path=tmp_path / "bridge.sock",
        registry_path=tmp_path / "bridge.json",
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta={},
        instance_token="test-token",
    )
    monkeypatch.setattr(
        "bn.transport.choose_instance", lambda instance_id=None, **kw: instance
    )

    class RefusingSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def settimeout(self, timeout):
            self.timeout = timeout

        def connect(self, path):
            raise ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")

    monkeypatch.setattr(
        "bn.transport.socket.socket", lambda *args, **kwargs: RefusingSocket()
    )
    started = time.monotonic()

    with pytest.raises(BridgeError, match="Timed out"):
        send_request("ping", timeout=0.01)

    assert time.monotonic() - started < 0.1


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
                request = json.loads(self.payload.decode("utf-8"))
                return json.dumps(
                    {
                        "ok": True,
                        "result": {"pong": True},
                        "bridge_identity": request["_bridge_identity"],
                    }
                ).encode("utf-8")
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
        instance_token="test-token",
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
    assert fake_socket.timeouts == pytest.approx([DEFAULT_REQUEST_TIMEOUT], rel=1e-5)


def test_send_request_timeout_budget_includes_instance_selection(
    tmp_path, monkeypatch
):
    import bn.transport as transport

    instance = _make_instance(tmp_path)
    captured = {}

    def choose(*args, **kwargs):
        time.sleep(0.03)
        return instance

    def send(instance, op, **kwargs):
        captured["timeout"] = kwargs["timeout"]
        return {
            "ok": True,
            "result": {},
            "bridge_identity": {
                "instance_id": instance.instance_id,
                "pid": instance.pid,
                "token": instance.instance_token,
            },
        }

    monkeypatch.setattr(transport, "choose_instance", choose)
    monkeypatch.setattr(transport, "_send_request_to_instance", send)

    transport.send_request("ping", timeout=0.1)

    assert 0 < captured["timeout"] < 0.09


def test_send_request_timeout_can_expire_during_instance_selection(
    monkeypatch,
):
    import bn.transport as transport

    def choose(*args, **kwargs):
        time.sleep(0.02)
        return object()

    monkeypatch.setattr(transport, "choose_instance", choose)
    monkeypatch.setattr(
        transport,
        "_send_request_to_instance",
        lambda *args, **kwargs: pytest.fail("request sent after deadline"),
    )

    with pytest.raises(BridgeError, match="selecting a bridge instance"):
        transport.send_request("ping", timeout=0.001)


def test_send_request_timeout_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "42.5")
    instance = _make_instance(tmp_path)
    monkeypatch.setattr("bn.transport.choose_instance", lambda instance_id=None, **kw: instance)
    fake_socket = _make_timeout_probe_socket()
    monkeypatch.setattr("bn.transport.socket.socket", lambda *args, **kwargs: fake_socket())

    send_request("ping")

    assert fake_socket.timeouts == pytest.approx([42.5], rel=1e-5)


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


def test_resolve_timeout_environment_is_validated_and_overrides_explicit(monkeypatch):
    from bn.transport import _resolve_timeout, DEFAULT_REQUEST_TIMEOUT

    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "42.5")
    assert _resolve_timeout(None) == 42.5
    assert _resolve_timeout(12.0) == 42.5
    monkeypatch.delenv("BN_REQUEST_TIMEOUT", raising=False)
    assert _resolve_timeout(None) == DEFAULT_REQUEST_TIMEOUT
    assert _resolve_timeout(12.0) == 12.0
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "abc")
    with pytest.raises(BridgeError, match="BN_REQUEST_TIMEOUT"):
        _resolve_timeout(12.0)


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



def test_stopped_bridge_fails_before_opening_socket(monkeypatch, tmp_path):
    import bn.transport as transport

    instance = _make_instance(tmp_path)
    monkeypatch.setattr(transport, "_process_state", lambda pid: "T")
    monkeypatch.setattr(
        transport.socket,
        "socket",
        lambda *args, **kwargs: pytest.fail("socket opened for stopped bridge"),
    )

    with pytest.raises(BridgeError, match="bridge_stopped.*kill -CONT"):
        transport._send_request_to_instance(instance, "target_info", timeout=30)


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



def _fake_socket_returning(payload_bytes):
    """A fake connected socket whose first recv() returns *payload_bytes*."""

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
            if not hasattr(self, "_sent"):
                self._sent = True
                return payload_bytes
            return b""

    return _FakeSocket


def test_send_request_ok_without_result_raises_bridge_error(tmp_path, monkeypatch):
    # #617: an "ok": true reply with no "result" key used to reach _run_one's
    # bare response["result"] index, raising KeyError instead of BridgeError
    # and crashing a whole fan-out survey. The transport layer must reject it.
    import bn.transport as transport

    instance = _make_instance(tmp_path)

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
            self.payload = payload

        def shutdown(self, how):
            pass

        def recv(self, size):
            if not hasattr(self, "_sent"):
                self._sent = True
                request = json.loads(self.payload.decode("utf-8"))
                return json.dumps(
                    {"ok": True, "bridge_identity": request["_bridge_identity"]}
                ).encode("utf-8")
            return b""

    monkeypatch.setattr(transport.socket, "socket", lambda *a, **k: _FakeSocket())

    with pytest.raises(BridgeError, match="without a result field") as excinfo:
        transport._send_request_to_instance(instance, "ping", timeout=5.0)
    assert not isinstance(excinfo.value, KeyError)


def test_send_request_fan_out_isolates_missing_result_from_other_instances(
    tmp_path, monkeypatch
):
    # #617 fan-out isolation: one instance replying "ok" without "result" must
    # raise BridgeError for that instance alone -- a sibling instance's normal
    # reply must still succeed.
    import bn.transport as transport

    broken = _make_instance(tmp_path)
    healthy = _make_instance(tmp_path)

    def _identity_echoing_socket(*, include_result):
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
                self.payload = payload

            def shutdown(self, how):
                pass

            def recv(self, size):
                if not hasattr(self, "_sent"):
                    self._sent = True
                    request = json.loads(self.payload.decode("utf-8"))
                    body = {
                        "ok": True,
                        "bridge_identity": request["_bridge_identity"],
                    }
                    if include_result:
                        body["result"] = {"pong": True}
                    return json.dumps(body).encode("utf-8")
                return b""

        return _FakeSocket

    broken_socket = _identity_echoing_socket(include_result=False)
    healthy_socket = _identity_echoing_socket(include_result=True)
    sockets = {id(broken): broken_socket, id(healthy): healthy_socket}

    _current = [id(broken)]

    def _fake_socket_ctor(*args, **kwargs):
        # Route by which instance is currently being dialed via a closure
        # variable set just before each call below.
        return sockets[_current[0]]()

    monkeypatch.setattr(transport.socket, "socket", _fake_socket_ctor)

    with pytest.raises(BridgeError, match="without a result field"):
        transport._send_request_to_instance(broken, "ping", timeout=5.0)

    _current[0] = id(healthy)
    response = transport._send_request_to_instance(healthy, "ping", timeout=5.0)
    assert response["result"]["pong"] is True


def test_send_request_non_utf8_reply_raises_bridge_error(tmp_path, monkeypatch):
    # #601: a non-UTF-8 byte stream used to raise a bare UnicodeDecodeError,
    # an exception type no caller handles. It must surface as BridgeError.
    import bn.transport as transport

    instance = _make_instance(tmp_path)
    fake_socket = _fake_socket_returning(b"\xff\xfe\x00not valid utf-8")
    monkeypatch.setattr(transport.socket, "socket", lambda *a, **k: fake_socket())

    with pytest.raises(BridgeError, match="non-UTF-8"):
        transport._send_request_to_instance(instance, "ping", timeout=5.0)


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
        instance_token="test-token",
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


def test_send_request_timeout_sends_cancel_request(tmp_path, monkeypatch):
    from bn.transport import BridgeError, BridgeInstance

    instance = BridgeInstance(
        pid=999,
        socket_path=tmp_path / "bridge.sock",
        registry_path=tmp_path / "bridge.json",
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta={},
        instance_token="test-token",
    )
    monkeypatch.setattr("bn.transport.choose_instance", lambda instance_id=None, **kw: instance)
    sent_payloads = []

    class _FakeSocket:
        def __init__(self):
            self.index = len(sent_payloads)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def settimeout(self, timeout):
            self.timeout = timeout

        def connect(self, path):
            self.path = path

        def sendall(self, payload):
            sent_payloads.append(json.loads(payload.decode("utf-8")))

        def shutdown(self, how):
            self.how = how

        def recv(self, size):
            if self.index == 0:
                raise socket.timeout("timed out")
            if not hasattr(self, "_sent"):
                self._sent = True
                return b'{"ok": true, "result": {"cancelled": true}}'
            return b""

    monkeypatch.setattr("bn.transport.socket.socket", lambda *args, **kwargs: _FakeSocket())

    with pytest.raises(BridgeError, match="Timed out waiting for Binary Ninja bridge pid 999"):
        send_request("ping", timeout=12.5)

    assert [payload["op"] for payload in sent_payloads] == ["ping", "cancel_request"]
    assert sent_payloads[1]["params"]["request_id"] == sent_payloads[0]["id"]


def test_timeout_message_shows_subsecond_value(tmp_path, monkeypatch):
    """A sub-second BN_REQUEST_TIMEOUT must render its real value, not round to
    'after 0.0s' (#370.3)."""
    from bn.transport import BridgeError, BridgeInstance, send_request

    instance = BridgeInstance(
        pid=999,
        socket_path=tmp_path / "bridge.sock",
        registry_path=tmp_path / "bridge.json",
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta={},
        instance_token="test-token",
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

    with pytest.raises(BridgeError, match=r"after 0\.01s"):
        send_request("ping", timeout=0.01)


def test_list_instances_trusts_live_socket_even_with_stale_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    registry_path = bridge_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    socket_path = tmp_path / "bn-live.sock"
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
    socket_path = tmp_path / "bn-fixed.sock"
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
    """Helper: start a mock server and write a registry file, return server.

    The socket lives at tmp_path root (NOT under instances_dir()) so gc's
    orphan sweep can't touch it; server_close() unlinks it on teardown.
    """
    inst_dir = tmp_path / subdir
    inst_dir.mkdir(parents=True, exist_ok=True)
    socket_path = tmp_path / f"bn-inst-{instance_id}.sock"
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
    socket_path = tmp_path / "bn-default.sock"
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
        assert "-i/--instance <id>" in message
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
    monkeypatch.setattr("bn.transport._process_alive", lambda pid: False)
    inst_dir = instances_dir()
    inst_dir.mkdir(parents=True, exist_ok=True)

    stale_socket = inst_dir / "bn-stale-inst.sock"
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

    try:
        instances = list_instances()
        assert not any(inst.instance_id == "deadbeef" for inst in instances)
        assert not registry_path.exists()
        # The dead socket is swept alongside its registry.
        assert not stale_socket.exists()
    finally:
        stale_socket.unlink(missing_ok=True)


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
    monkeypatch.setattr("bn.transport.list_instances", lambda **kwargs: [existing])

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
    monkeypatch.setattr("bn.transport.list_instances", lambda **kwargs: [existing])
    monkeypatch.setattr("bn.transport._find_bn_agent", lambda: ["bn-agent"])
    monkeypatch.setattr("bn.transport._load_instance", lambda path, **kwargs: created)

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
    monkeypatch.setattr("bn.transport.list_instances", lambda **kwargs: [])
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
    monkeypatch.setattr("bn.transport.list_instances", lambda **kwargs: [])
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

    def fake_spawn(instance_id, **kwargs):
        spawned["id"] = instance_id
        return sentinel

    monkeypatch.setattr("bn.transport.spawn_instance", fake_spawn)

    # opt-in (the `bn load` path): a missing named id is spawned, not an error.
    result = choose_instance("brandnew", spawn_missing_named=True)
    assert result is sentinel
    assert spawned["id"] == "brandnew"

    # default: a missing named id still fails fast, and the error points the way.
    with pytest.raises(
        BridgeError,
        match="session start /path/to/binary --instance-id brandnew",
    ) as exc:
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


# --- #523: bridge path boundary validates instance ids too ------------------
# The CLI validates ids before spawning, but a bridge started directly
# (bn-agent --instance-id) reaches bridge_socket_path/bridge_registry_path with
# a raw id. Those helpers must reject a traversal/absolute/empty id before it is
# joined into a filesystem path, so no bridge files land outside instances_dir().


@pytest.mark.parametrize("bad", ["../evil", "../../tmp/evil", "/abs/id", "a/b", "a\\b", ".", "..", "", "safe\n", "\nsafe"])
def test_bridge_path_helpers_reject_unsafe_instance_id(bad, tmp_path, monkeypatch):
    from bn.paths import bridge_registry_path as reg, bridge_socket_path as sock

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="Invalid instance id|non-empty"):
        reg(bad)
    with pytest.raises(ValueError, match="Invalid instance id|non-empty"):
        sock(bad)


def test_bridge_path_helpers_accept_generated_and_legacy_ids(tmp_path, monkeypatch):
    import secrets

    from bn.paths import (
        bridge_registry_path as reg,
        bridge_socket_path as sock,
        cache_home,
        instances_dir,
    )

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    # A normal generated id (secrets.token_hex) must pass and stay inside instances_dir().
    gen = secrets.token_hex(4)
    assert reg(gen) == instances_dir() / f"{gen}.json"
    assert sock(gen) == instances_dir() / f"{gen}.sock"
    # The legacy GUI fixed pair (instance_id=None) is exempt from validation.
    assert reg(None) == cache_home() / "bn_agent_bridge.json"
    assert sock(None) == cache_home() / "bn_agent_bridge.sock"


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
    socket_path = tmp_path / "bn-pidtest.sock"
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

    socket_path = tmp_path / "bn-live.sock"
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


def test_resolve_timeout_per_op_default_applies_only_without_env(monkeypatch):
    """#321: load/refresh raise their no-env default client timeout so a long
    one-time analysis isn't abandoned at the 600s read-op default -- but an
    explicit BN_REQUEST_TIMEOUT still wins, and the disable sentinels still
    disable."""
    from bn.transport import _resolve_timeout, REFRESH_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT

    # no env -> the per-op default applies (the larger refresh value), not 600s
    monkeypatch.delenv("BN_REQUEST_TIMEOUT", raising=False)
    assert _resolve_timeout(None, default=REFRESH_REQUEST_TIMEOUT) == REFRESH_REQUEST_TIMEOUT
    assert _resolve_timeout(None) == DEFAULT_REQUEST_TIMEOUT          # unchanged for normal ops
    # explicit env wins over the per-op default
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "42.5")
    assert _resolve_timeout(None, default=REFRESH_REQUEST_TIMEOUT) == 42.5
    # disable sentinel still disables, regardless of the per-op default
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "0")
    assert _resolve_timeout(None, default=REFRESH_REQUEST_TIMEOUT) is None
    # an explicit per-call timeout overrides everything
    monkeypatch.delenv("BN_REQUEST_TIMEOUT", raising=False)
    assert _resolve_timeout(5.0, default=REFRESH_REQUEST_TIMEOUT) == 5.0


def test_send_request_load_refresh_use_larger_default(tmp_path, monkeypatch):
    """#321: send_request with a larger default_timeout (the load/refresh path)
    applies it to the socket when BN_REQUEST_TIMEOUT is unset."""
    from bn.transport import REFRESH_REQUEST_TIMEOUT

    monkeypatch.delenv("BN_REQUEST_TIMEOUT", raising=False)
    instance = _make_instance(tmp_path)
    monkeypatch.setattr("bn.transport.choose_instance", lambda instance_id=None, **kw: instance)
    fake_socket = _make_timeout_probe_socket()
    monkeypatch.setattr("bn.transport.socket.socket", lambda *args, **kwargs: fake_socket())

    send_request("refresh", default_timeout=REFRESH_REQUEST_TIMEOUT)
    assert fake_socket.timeouts == pytest.approx([REFRESH_REQUEST_TIMEOUT], rel=1e-5)


# -- private project associations -----------------------------------------
import bn.transport as _t
from pathlib import Path as _Path


def _mk_inst(iid, *, roots=None, meta=None):
    payload = dict(meta or {})
    if roots is not None:
        payload["project_roots"] = roots
    return _t.BridgeInstance(
        pid=1,
        socket_path=_Path("/x.sock"),
        registry_path=_Path("/x.json"),
        plugin_name="p",
        plugin_version="v",
        started_at=None,
        meta=payload,
        instance_id=iid,
    )


def test_resolve_from_project_roots_picks_unique_live_instance(monkeypatch, tmp_path):
    root = tmp_path / "project"
    (root / ".git").mkdir(parents=True)
    monkeypatch.chdir(root)
    live = _mk_inst("abcd", roots=[str(root)])

    assert _t._resolve_from_project_roots([_mk_inst("ef01"), live]) is live


def test_resolve_from_project_roots_ignores_missing_or_malformed_metadata(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)

    assert _t._resolve_from_project_roots([
        _mk_inst("none1"),
        _mk_inst("bad1", meta={"project_roots": "not-a-list"}),
        _mk_inst("bad2", roots=[str(tmp_path), 7]),
    ]) is None


def test_resolve_from_project_roots_fails_closed_on_multiple_matches(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    first = _mk_inst("aaaa", roots=[str(tmp_path)])
    second = _mk_inst("bbbb", roots=[str(tmp_path)])

    with pytest.raises(_t.BridgeError, match="Multiple.*associated"):
        _t._resolve_from_project_roots([first, second])


def test_resolve_from_project_roots_keeps_nested_repositories_isolated(
    monkeypatch, tmp_path
):
    outer = tmp_path / "outer"
    inner = outer / "vendor" / "inner"
    (outer / ".git").mkdir(parents=True)
    (inner / ".git").mkdir(parents=True)
    monkeypatch.chdir(inner)

    assert _t._resolve_from_project_roots([
        _mk_inst("outer1", roots=[str(outer)]),
        _mk_inst("inner1", roots=[str(inner)]),
    ]).instance_id == "inner1"


def test_send_request_still_lets_the_env_override_win_for_a_single_request(
    monkeypatch, tmp_path
):
    import bn.transport as transport

    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "5")
    instance = _make_instance(tmp_path)
    seen: list[float | None] = []
    monkeypatch.setattr(transport, "choose_instance", lambda *a, **k: instance)
    monkeypatch.setattr(
        transport,
        "_send_request_to_instance",
        lambda inst, op, **kwargs: seen.append(kwargs["timeout"])
        or {"ok": True, "result": {}},
    )

    send_request("target_info", timeout=0.5)

    assert seen and 4.0 < seen[0] <= 5.0


def test_send_request_forwards_a_preresolved_budget_without_re_expanding_it(
    monkeypatch, tmp_path
):
    """A paginating caller resolves BN_REQUEST_TIMEOUT once, then hands down the
    shrinking remainder. Re-resolving here restores the full env value and lets a
    multi-page collection overrun its own declared end-to-end budget."""
    import bn.transport as transport

    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "5")
    instance = _make_instance(tmp_path)
    seen: list[float | None] = []
    selection: list[float | None] = []

    def choose(instance_id=None, **kwargs):
        selection.append(kwargs.get("timeout"))
        return instance

    monkeypatch.setattr(transport, "choose_instance", choose)
    monkeypatch.setattr(
        transport,
        "_send_request_to_instance",
        lambda inst, op, **kwargs: seen.append(kwargs["timeout"])
        or {"ok": True, "result": {}},
    )

    send_request("target_info", timeout=0.5, resolved=True)

    assert seen and 0 < seen[0] <= 0.5
    assert selection and 0 < selection[0] <= 0.5


def test_send_request_to_instance_does_not_re_resolve_a_preresolved_budget(
    monkeypatch, tmp_path
):
    import bn.transport as transport

    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "600")
    instance = _make_instance(tmp_path)
    monkeypatch.setattr(transport, "_process_state", lambda pid: "S")
    fake_socket = _make_timeout_probe_socket()
    monkeypatch.setattr(
        transport.socket, "socket", lambda *args, **kwargs: fake_socket()
    )

    transport._send_request_to_instance(
        instance, "ping", timeout=0.25, resolved=True, connect_retries=1
    )

    assert fake_socket.timeouts == pytest.approx([0.25], rel=1e-5)


# --------------------------------------------------------------------------
# #694: the spawn budget is separate from the request budget
# --------------------------------------------------------------------------


def test_auto_spawn_uses_the_spawn_budget_not_the_request_budget(monkeypatch):
    # choose_instance() used to hand the resolved REQUEST timeout to the spawn
    # path, so a child that never registers held an ordinary request for 600s
    # (3600s for load/refresh) and BN_SPAWN_TIMEOUT was ignored entirely.
    import bn.transport as transport

    monkeypatch.setenv("BN_SPAWN_TIMEOUT", "1.5")
    monkeypatch.setattr(transport, "list_instances", lambda **kwargs: [])
    seen: list[float | None] = []

    def fake_auto_spawn(timeout=None):
        seen.append(timeout)
        return "instance"

    monkeypatch.setattr(transport, "_auto_spawn_locked", fake_auto_spawn)

    assert choose_instance(timeout=600.0) == "instance"
    assert seen == [pytest.approx(1.5, rel=1e-3)]


def test_spawn_budget_is_capped_by_the_remaining_request_deadline(monkeypatch):
    # Both bounds are real: the spawn must not outlive the caller's end-to-end
    # request budget either.
    import bn.transport as transport

    monkeypatch.setenv("BN_SPAWN_TIMEOUT", "60")
    monkeypatch.setattr(transport, "list_instances", lambda **kwargs: [])
    seen: list[float | None] = []
    monkeypatch.setattr(
        transport,
        "_auto_spawn_locked",
        lambda timeout=None: seen.append(timeout) or "instance",
    )

    choose_instance(timeout=2.0)

    assert seen and 0 < seen[0] <= 2.0


def test_named_spawn_uses_the_spawn_budget(monkeypatch):
    import bn.transport as transport

    monkeypatch.setenv("BN_SPAWN_TIMEOUT", "2")
    monkeypatch.setattr(transport, "list_instances", lambda **kwargs: [])
    seen: list[float | None] = []
    monkeypatch.setattr(
        transport,
        "spawn_instance",
        lambda instance_id=None, timeout=None: seen.append(timeout) or "named",
    )

    assert choose_instance("wanted", spawn_missing_named=True, timeout=3600.0) == "named"
    assert seen == [pytest.approx(2.0, rel=1e-3)]


def test_malformed_spawn_timeout_is_rejected_before_spawning(monkeypatch):
    import bn.transport as transport

    monkeypatch.setenv("BN_SPAWN_TIMEOUT", "soon")
    monkeypatch.setattr(transport, "list_instances", lambda **kwargs: [])
    monkeypatch.setattr(
        transport,
        "_auto_spawn_locked",
        lambda timeout=None: pytest.fail("must not spawn on a malformed budget"),
    )

    with pytest.raises(BridgeError, match="BN_SPAWN_TIMEOUT"):
        choose_instance(timeout=600.0)


def test_spawn_instance_still_resolves_its_own_env_budget(monkeypatch):
    import bn.transport as transport

    monkeypatch.setenv("BN_SPAWN_TIMEOUT", "3.25")
    seen: list[float] = []

    class _NullLock:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(transport, "_spawn_lock", lambda timeout=None: _NullLock())
    monkeypatch.setattr(
        transport,
        "_spawn_instance_unlocked",
        lambda instance_id=None, timeout=None, poll_interval=0.2: seen.append(timeout)
        or "spawned",
    )

    assert spawn_instance("named1") == "spawned"
    assert seen and 0 < seen[0] <= 3.25


# --------------------------------------------------------------------------
# #694: durable identity + ATOMIC signalling
#
# A registry pid is not an identity, and a /proc check followed by os.kill is not
# atomic: the verified process can exit and its pid be recycled in between. The
# bridge records (boot id, pid, start ticks) -- boot id because start ticks are
# unique only within one boot while registries live in a persistent cache -- and
# the CLI pins the pid with a pidfd, verifies identity THROUGH the pin, and sends
# every signal of a teardown through that same pin.
# --------------------------------------------------------------------------


def _pidfd_available() -> bool:
    from bn.proc_identity import PIDFD_AVAILABLE

    return PIDFD_AVAILABLE


def _identity(*, ticks_delta=0, boot=None, omit_boot=False, omit_ticks=False):
    """A recorded identity for THIS process, optionally tampered with."""
    from bn.proc_identity import boot_id, process_start_ticks

    payload = {}
    if not omit_ticks:
        payload["pid_start_ticks"] = process_start_ticks(os.getpid()) + ticks_delta
    if not omit_boot:
        payload["boot_id"] = boot if boot is not None else boot_id()
    return payload


def _registry_payload(socket_path, *, pid, identity=None, instance_id=None):
    payload = {
        "pid": pid,
        "socket_path": str(socket_path),
        "plugin_name": "bn_agent_bridge",
        "plugin_version": "0.1.0",
        "instance_token": "identity-token",
    }
    payload.update(identity or {})
    if instance_id is not None:
        payload["instance_id"] = instance_id
    return payload


def _sleeper(*, ignore_sigterm=False):
    """A real child process to pin and signal."""
    program = "import time; time.sleep(30)"
    if ignore_sigterm:
        program = (
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "time.sleep(30)\n"
        )
    return subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_identity_payload_records_boot_id_and_start_ticks():
    from bn.proc_identity import boot_id, identity_payload, process_start_ticks

    assert identity_payload() == {
        "pid_start_ticks": process_start_ticks(os.getpid()),
        "boot_id": boot_id(),
    }


def test_identity_verdict_requires_both_boot_id_and_start_ticks():
    from bn.proc_identity import identity_verdict

    live = os.getpid()

    assert identity_verdict(_identity(), live) == "proven"
    assert identity_verdict(_identity(ticks_delta=1), live) == "mismatch"
    # A registry that predates this boot is positively stale: start ticks count
    # from boot, so an old (pid, ticks) pair can collide with a fresh process --
    # and registries live in a persistent cache dir across reboots.
    assert (
        identity_verdict(_identity(boot="00000000-0000-4000-8000-000000000000"), live)
        == "mismatch"
    )
    # Half an identity proves nothing, so older registries stay unproven.
    assert identity_verdict(_identity(omit_boot=True), live) == "unrecorded"
    assert identity_verdict(_identity(omit_ticks=True), live) == "unrecorded"
    assert identity_verdict({}, live) == "unrecorded"
    for bad_ticks in (True, "12345", -1, None):
        assert (
            identity_verdict({"pid_start_ticks": bad_ticks, "boot_id": "b"}, live)
            == "unrecorded"
        )
    for bad_boot in (True, 7, "", "   ", None):
        assert (
            identity_verdict({"pid_start_ticks": 1, "boot_id": bad_boot}, live)
            == "unrecorded"
        )


class _FakePin:
    """Stand-in for a pidfd pin, so the signalling policy is testable anywhere.

    The pidfd wrappers are a property of the interpreter build (the
    python-build-standalone runtime uv installs has none), and the safety policy
    -- verify once, send every escalation signal through THAT pin, send nothing
    when unproven -- must be exercised regardless. One native integration test
    below covers the real syscall path where it exists.
    """

    def __init__(self, pid, verdict="proven", error=None):
        self.pid = pid
        self.fd = 4242
        self._verdict = verdict
        self._error = error
        self.sent: list[int] = []
        self.closed = False

    def verdict(self, payload):
        assert not self.closed, "verdict() after close(): the pin must stay open"
        return self._verdict

    def send(self, sig):
        assert not self.closed, "send() after close(): the pin must stay open"
        if self._error is not None:
            raise self._error
        self.sent.append(sig)

    def close(self):
        self.closed = True


def _fake_pin(monkeypatch, **kwargs):
    """Install a fake pin factory; returns (pins_created, call_pids)."""
    import bn.transport as transport

    pins: list[_FakePin] = []
    pids: list[int] = []

    def factory(pid):
        pids.append(pid)
        pin = _FakePin(pid, **kwargs)
        pins.append(pin)
        return pin

    monkeypatch.setattr(transport, "pin_process", factory)
    return pins, pids


def _instance(tmp_path, *, pid=4321, meta=None):
    from bn.transport import BridgeInstance

    return BridgeInstance(
        pid=pid,
        socket_path=tmp_path / "child.sock",
        registry_path=tmp_path / "child.json",
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta=_identity() if meta is None else meta,
        instance_id="child1",
        instance_token="t",
    )


def test_bridge_signal_sends_term_then_kill_through_one_verified_pin(
    tmp_path, monkeypatch
):
    # The escalation must hold ONE pin: reopening between SIGTERM and SIGKILL is a
    # second check-then-act window, and after SIGTERM the process can be a zombie
    # whose identity re-read would look unprovable.
    pins, pids = _fake_pin(monkeypatch)

    with transport_module().BridgeProcessSignal(_instance(tmp_path)) as signaller:
        assert signaller.refusal is None
        assert signaller.send(signal.SIGTERM) is None
        assert signaller.send(signal.SIGKILL) is None

    assert pids == [4321]                     # pinned exactly once
    assert len(pins) == 1
    assert pins[0].sent == [signal.SIGTERM, signal.SIGKILL]
    assert pins[0].closed is True             # released with the context manager


@pytest.mark.parametrize(
    "verdict, expected",
    [
        ("mismatch", "does not match the pinned process"),
        ("unrecorded", "no verifiable process identity"),
    ],
)
def test_bridge_signal_refuses_and_sends_nothing_when_unproven(
    tmp_path, monkeypatch, verdict, expected
):
    pins, _ = _fake_pin(monkeypatch, verdict=verdict)

    with transport_module().BridgeProcessSignal(_instance(tmp_path)) as signaller:
        assert signaller.refusal is not None and expected in signaller.refusal
        assert expected in (signaller.send(signal.SIGTERM) or "")
        assert expected in (signaller.send(signal.SIGKILL) or "")

    assert pins[0].sent == []                 # nothing was signalled at all
    assert pins[0].closed is True             # and the fd was not leaked


def test_bridge_signal_refuses_when_the_pid_cannot_be_pinned(tmp_path, monkeypatch):
    # Covers both "pid is gone" and "no pidfd on this interpreter": with no atomic
    # pin, falling back to os.kill would reintroduce the reuse race, so nothing is
    # signalled.
    import bn.transport as transport
    from bn.proc_identity import PinUnavailable

    def refuse(pid):
        raise PinUnavailable(f"pid {pid} is not running, so there is nothing to signal")

    monkeypatch.setattr(transport, "pin_process", refuse)

    with transport.BridgeProcessSignal(_instance(tmp_path)) as signaller:
        assert signaller.refusal is not None
        assert "refusing to signal pid 4321" in signaller.refusal
        assert "not running" in signaller.refusal
        assert signaller.send(signal.SIGKILL) == signaller.refusal


def test_bridge_signal_refuses_without_an_atomic_primitive(tmp_path, monkeypatch):
    import bn.proc_identity as proc_identity
    import bn.transport as transport

    monkeypatch.setattr(proc_identity, "PIDFD_AVAILABLE", False)

    with transport.BridgeProcessSignal(_instance(tmp_path)) as signaller:
        assert signaller.refusal is not None
        assert "provides no pidfd" in signaller.refusal
        assert signaller.send(signal.SIGTERM) is not None


def test_bridge_signal_reports_a_process_that_exited_before_delivery(
    tmp_path, monkeypatch
):
    _fake_pin(monkeypatch, error=ProcessLookupError())

    with transport_module().BridgeProcessSignal(_instance(tmp_path)) as signaller:
        assert signaller.refusal is None
        reason = signaller.send(signal.SIGTERM)

    assert reason is not None and "exited before the signal was delivered" in reason


def test_bridge_signal_reports_a_permission_failure(tmp_path, monkeypatch):
    _fake_pin(monkeypatch, error=PermissionError(1, "Operation not permitted"))

    with transport_module().BridgeProcessSignal(_instance(tmp_path)) as signaller:
        reason = signaller.send(signal.SIGKILL)

    assert reason is not None and "failed to signal pid 4321" in reason


def test_bridge_signal_refuses_after_its_pin_is_released(tmp_path, monkeypatch):
    pins, _ = _fake_pin(monkeypatch)
    signaller = transport_module().BridgeProcessSignal(_instance(tmp_path))
    signaller.close()

    reason = signaller.send(signal.SIGTERM)

    assert reason is not None and "already released" in reason
    assert pins[0].sent == []


def transport_module():
    import bn.transport as transport

    return transport


@pytest.mark.skipif(
    not _pidfd_available(),
    reason="this interpreter exposes no pidfd wrappers (os.pidfd_open)",
)
def test_bridge_signal_native_pidfd_delivers_and_reuses_the_same_fd(tmp_path):
    # The one integration test over the real syscalls: everything above proves the
    # policy, this proves the primitive is wired correctly where it exists.
    from bn.proc_identity import boot_id, process_start_ticks
    from bn.transport import BridgeProcessSignal

    proc = _sleeper(ignore_sigterm=True)
    try:
        instance = _instance(
            tmp_path,
            pid=proc.pid,
            meta={
                "pid_start_ticks": process_start_ticks(proc.pid),
                "boot_id": boot_id(),
            },
        )
        with BridgeProcessSignal(instance) as signaller:
            assert signaller.refusal is None
            pinned_fd = signaller._pin._fd
            assert signaller.send(signal.SIGTERM) is None
            time.sleep(0.2)
            assert proc.poll() is None            # SIGTERM ignored on purpose
            assert signaller.send(signal.SIGKILL) is None
            assert signaller._pin._fd == pinned_fd
        assert proc.wait(timeout=5) == -signal.SIGKILL
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_wait_for_teardown_treats_a_reused_pid_as_gone(tmp_path, monkeypatch):
    # `session stop` escalates when teardown does not converge. With bare
    # liveness a recycled pid never converges, so the escalation fired against a
    # stranger; identity makes the recycled pid read as gone.
    import bn.transport as transport
    from bn.transport import BridgeInstance

    instance = BridgeInstance(
        pid=os.getpid(),
        socket_path=tmp_path / "t.sock",
        registry_path=tmp_path / "t.json",
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta=_identity(ticks_delta=11),
        instance_id="tear1",
        instance_token="t",
    )
    monkeypatch.setattr(transport, "_load_instance", lambda *a, **k: None)

    assert wait_for_teardown(instance, timeout=0.05) is True


# --------------------------------------------------------------------------
# #694 finding 10: a socket-less registry never reaches normal discovery
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "identity",
    [
        _identity,
        lambda: _identity(ticks_delta=5),
        lambda: _identity(omit_boot=True),
        dict,
    ],
    ids=["proven", "reused-pid", "no-boot-id", "no-identity"],
)
def test_socketless_registry_is_never_listed(tmp_path, monkeypatch, identity):
    # start() binds the socket BEFORE writing the registry, so "registry, no
    # socket" is never a startup window -- such a bridge is irrecoverably
    # unreachable and must not be advertised, whatever its identity says.
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    registry_path = bridge_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            _registry_payload(
                tmp_path / "gone.sock", pid=os.getpid(), identity=identity()
            )
        ),
        encoding="utf-8",
    )

    assert list_instances() == []


@pytest.mark.parametrize(
    "identity",
    [lambda: _identity(ticks_delta=5), lambda: _identity(omit_boot=True), dict],
    ids=["reused-pid", "no-boot-id", "no-identity"],
)
def test_socketless_registry_without_proof_self_heals(tmp_path, monkeypatch, identity):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    registry_path = bridge_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            _registry_payload(
                tmp_path / "gone.sock", pid=os.getpid(), identity=identity()
            )
        ),
        encoding="utf-8",
    )

    assert list_instances() == []
    assert not registry_path.exists()


def test_unresponsive_socket_with_identity_mismatch_is_swept(tmp_path, monkeypatch):
    # A crash leaves the socket FILE behind (only a clean stop unlinks it). A
    # failing probe alone stays non-destructive -- a busy bridge may not accept in
    # time -- but a recorded identity that no longer matches is proof the bridge
    # exited and its pid was reused, so the entry goes.
    import bn.transport as transport

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    registry_path = bridge_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path = tmp_path / "stale.sock"
    socket_path.touch()
    registry_path.write_text(
        json.dumps(
            _registry_payload(
                socket_path, pid=os.getpid(), identity=_identity(ticks_delta=7)
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(transport, "_socket_is_live", lambda *a, **k: False)

    assert list_instances() == []
    assert not registry_path.exists()
    assert not socket_path.exists()


def test_unresponsive_socket_of_a_proven_bridge_is_preserved(tmp_path, monkeypatch):
    # The other half of that rule: a live, proven bridge whose accept backlog is
    # full must never be swept by discovery.
    import bn.transport as transport

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    registry_path = bridge_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path = tmp_path / "busy.sock"
    socket_path.touch()
    registry_path.write_text(
        json.dumps(
            _registry_payload(socket_path, pid=os.getpid(), identity=_identity())
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(transport, "_socket_is_live", lambda *a, **k: False)

    instances = list_instances()

    assert [inst.pid for inst in instances] == [os.getpid()]
    assert instances[0].unreachable is False
    assert registry_path.exists() and socket_path.exists()


def test_socketless_registry_with_proven_owner_is_lifecycle_only(tmp_path, monkeypatch):
    # Hidden from every routing path, but `bn session stop` must still be able to
    # name the live process that is holding memory -- so the record survives for
    # the lifecycle lookup and is marked unreachable.
    import bn.transport as transport

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    registry_path = instances_dir() / "hidden1.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            _registry_payload(
                tmp_path / "gone.sock",
                pid=os.getpid(),
                identity=_identity(),
                instance_id="hidden1",
            )
        ),
        encoding="utf-8",
    )

    assert list_instances() == []
    assert registry_path.exists()               # not purged while its owner lives

    admin = list_instances(include_unreachable=True)
    assert [inst.instance_id for inst in admin] == ["hidden1"]
    assert admin[0].unreachable is True

    found = transport.find_lifecycle_instance("hidden1")
    assert found is not None and found.unreachable is True
    assert transport.find_lifecycle_instance("nope") is None


def test_socketless_registry_is_purged_once_its_owner_exits(tmp_path, monkeypatch):
    from bn.proc_identity import boot_id, process_start_ticks

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    proc = _sleeper()
    registry_path = instances_dir() / "gone1.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            _registry_payload(
                tmp_path / "gone.sock",
                pid=proc.pid,
                identity={
                    "pid_start_ticks": process_start_ticks(proc.pid),
                    "boot_id": boot_id(),
                },
                instance_id="gone1",
            )
        ),
        encoding="utf-8",
    )
    assert list_instances(include_unreachable=True)[0].instance_id == "gone1"

    proc.kill()
    proc.wait(timeout=5)

    assert list_instances(include_unreachable=True) == []
    assert not registry_path.exists()


def test_spawn_refuses_to_reuse_a_hidden_unreachable_instance_id(tmp_path, monkeypatch):
    # The dangerous interaction of hiding unreachable records: a spawn that cannot
    # see one would bind its socket path, overwrite its registry, and orphan the
    # live process with no record left to stop it.
    import bn.transport as transport

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    registry_path = instances_dir() / "busy1.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            _registry_payload(
                tmp_path / "gone.sock",
                pid=os.getpid(),
                identity=_identity(),
                instance_id="busy1",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        transport.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("a duplicate id must be refused before spawning"),
    )

    with pytest.raises(BridgeError, match="already exists with id: busy1"):
        transport._spawn_instance_unlocked("busy1", timeout=5.0)

    assert registry_path.exists()


def test_auto_spawn_never_picks_a_hidden_instance_id(tmp_path, monkeypatch):
    import bn.transport as transport

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    registry_path = instances_dir() / "aaaa1111.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            _registry_payload(
                tmp_path / "gone.sock",
                pid=os.getpid(),
                identity=_identity(),
                instance_id="aaaa1111",
            )
        ),
        encoding="utf-8",
    )
    # Force the random id generator to collide first, then yield a free one.
    candidates = iter(["aaaa1111", "bbbb2222"])
    monkeypatch.setattr(transport.secrets, "token_hex", lambda n: next(candidates))
    spawned: list[str] = []

    def fake_popen(cmd, **kwargs):
        spawned.append(cmd[-1])
        raise AssertionError("stop here: the chosen id is what matters")

    monkeypatch.setattr(transport.subprocess, "Popen", fake_popen)

    with pytest.raises(AssertionError, match="stop here"):
        transport._spawn_instance_unlocked(None, timeout=5.0)

    assert spawned == ["bbbb2222"]
