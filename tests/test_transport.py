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
from bn.transport import BridgeError, choose_instance, list_instances, send_request, spawn_instance


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
    assert stale_socket_path.exists()


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
    monkeypatch.setattr("bn.transport.choose_instance", lambda instance_id=None: instance)

    with pytest.raises(BridgeError, match="Failed to contact Binary Ninja bridge pid 999"):
        send_request("doctor")


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
    monkeypatch.setattr("bn.transport.choose_instance", lambda instance_id=None: instance)

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


def test_send_request_uses_blocking_socket_by_default(tmp_path, monkeypatch):
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
        timeout_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def settimeout(self, timeout):
            type(self).timeout_calls += 1
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

    monkeypatch.setattr("bn.transport.socket.socket", lambda *args, **kwargs: _FakeSocket())

    response = send_request("ping")

    assert response["result"]["pong"] is True
    assert _FakeSocket.timeout_calls == 0


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
    monkeypatch.setattr("bn.transport.choose_instance", lambda instance_id=None: instance)

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
