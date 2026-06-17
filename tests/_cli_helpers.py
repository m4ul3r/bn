from __future__ import annotations

import json
import types

import bn.cli
import pytest


def _spill_artifact_namespace(path: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        rendered=f"ok: true\nspilled: true\npath: {path}\n",
        spilled=True,
        artifact={
            "artifact_path": path,
            "bytes": 4321,
            "format": "text",
            "sha256": "feedface",
            "spilled": True,
            "summary": {"kind": "string", "chars": 99},
            "tokenizer": "estimate",
            "tokens": 34567,
        },
    )


def _zero_function_search(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
    return {"ok": True, "result": {"functions": [], "total": 0, "offset": 0,
                                   "limit": None, "returned": 0, "has_more": False}}


def _empty_xrefs(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
    return {"ok": True, "result": {"address": "0x308", "code_refs": [], "data_refs": []}}


# --- Sticky instance/target ---


@pytest.fixture
def tmp_session(tmp_path, monkeypatch):
    """Isolate session-state file per test by redirecting BN_CACHE_DIR and cwd."""
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _fake_bridge_instance(instance_id="abc123", pid=111):
    from pathlib import Path as _Path

    from bn.transport import BridgeInstance

    return BridgeInstance(
        pid=pid,
        socket_path=_Path(f"/tmp/{instance_id}.sock"),
        registry_path=_Path(f"/tmp/{instance_id}.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={},
        instance_id=instance_id,
    )


def _load_capture(monkeypatch, raw, analyzed):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "load_binary"
        captured.update(params)
        return {"ok": True, "result": {
            "loaded": True, "path": str(raw), "analyzed": analyzed,
            "notes": ([] if analyzed else ["loaded without analysis (--quick): run `bn refresh`"]),
            "targets": [],
        }}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    return captured


# --- Review fixes: display-only flags, xrefs mutex, session stop, spill hints ---


def _assert_no_bridge_call(monkeypatch):
    def fail_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        raise AssertionError(f"bridge should not be called (got op {op!r})")

    monkeypatch.setattr(bn.cli, "send_request", fail_send_request)


def _capture_xrefs_call(monkeypatch):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params or {}
        return {"ok": True, "result": {
            "address": "0x401000", "target_context": {}, "code_refs": [], "data_refs": [],
            "code_ref_count": 0, "data_ref_count": 0, "caller_function_count": 0,
            "items": [], "total": 0, "offset": 0, "limit": 3, "returned": 0, "has_more": False,
        }}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    return captured


__all__ = ['_spill_artifact_namespace', '_zero_function_search', '_empty_xrefs', 'tmp_session', '_fake_bridge_instance', '_load_capture', '_assert_no_bridge_call', '_capture_xrefs_call']
