from __future__ import annotations

import importlib
import importlib.util
import io
import json
import socket
import sys
import threading
import time
import types
import weakref
from pathlib import Path

import pytest

from _bridge_fakes import *  # noqa: F401,F403


def test_function_create_at_executable_address_returns_verified(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_create(None, "0x1000", False)

    assert result["success"] is True
    assert result["committed"] is True
    assert bv.added == [0x1000]
    assert "refresh" in bv.events
    assert ("commit", "state") in bv.events
    res = result["results"][0]
    assert res["op"] == "function_create"
    assert res["status"] == "verified"
    assert res["address"] == "0x1000"
    assert res["function"] == "sub_1000"
    assert result["affected_functions"][0]["after_name"] == "sub_1000"


def test_function_create_preview_reverts_without_committing(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_create(None, "0x1000", True)

    assert result["success"] is True
    assert result["committed"] is False
    assert result["results"][0]["status"] == "verified"
    assert ("revert", "state") in bv.events
    assert ("commit", "state") not in bv.events


def test_function_create_existing_function_is_noop(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        functions=[_FakeFunction(0x1000, "player_update")],
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_create(None, "0x1000", False)

    assert result["success"] is True
    assert result["committed"] is False
    assert bv.added == []
    res = result["results"][0]
    assert res["status"] == "noop"
    assert res["function"] == "player_update"


def test_function_create_unmapped_address_is_rejected(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(RuntimeError, match="0xdead.*not mapped"):
        instance._function_create(None, "0xdead", False)

    assert bv.added == []


def test_function_create_non_executable_address_is_rejected(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x5000: _FakeSegment(readable=True, writable=True, executable=False)},
        memory={0x5000: b"\x01\x02\x03\x04"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(RuntimeError, match="0x5000.*not inside an executable segment"):
        instance._function_create(None, "0x5000", False)

    assert bv.added == []


def test_function_create_preview_actually_removes_function(monkeypatch):
    """--preview must leave NO trace. add_function is not journaled, so the
    preview revert has to explicitly remove the created function and read back
    that it is gone -- reporting 'reverted' while the function persists in the
    view is the bug (#117)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_create(None, "0x1000", True)

    assert result["preview"] is True
    assert result["committed"] is False
    assert result["success"] is True
    assert result["rolled_back"] is True
    assert result["results"][0]["status"] == "verified"
    # The revert uses the non-poisoning remove_function (#304), not
    # remove_user_function (which would suppress the address).
    assert ("remove_function", 0x1000) in bv.events
    assert ("remove_user_function", 0x1000) not in bv.events
    # The crux: no function may persist at the address after a preview.
    assert bv.get_function_at(0x1000) is None


def test_function_create_preview_does_not_poison_subsequent_live_create(monkeypatch):
    """#304: a --preview must not sabotage the follow-up live `function create`
    at the same address. The old revert used remove_user_function, which records
    a persistent user "no function here" override -- so the preview reported
    `verified` but the live commit then reported `verification_failed`. The
    non-poisoning remove_function makes preview and live agree."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    preview = instance._function_create(None, "0x1000", True)
    assert preview["success"] is True
    assert preview["results"][0]["status"] == "verified"
    assert bv.get_function_at(0x1000) is None  # reverted

    # The follow-up live create at the SAME address must still succeed -- the
    # preview's revert must not have suppressed the address.
    live = instance._function_create(None, "0x1000", False)
    assert live["committed"] is True, live
    assert live["results"][0]["status"] == "verified", live


def test_function_create_preview_revert_failure_is_not_success(monkeypatch):
    """If removing the created function on preview-revert fails, the view is
    left modified -- report success:false / rolled_back:false, never a clean
    'reverted'. Honesty over an unverified revert claim (#117)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    # BOTH removal paths silently do nothing, so the function persists past the
    # revert (remove_function is tried first now, with remove_user_function as
    # the fallback).
    bv.remove_function = lambda fn: bv.events.append(("remove_attempt", int(fn.start)))
    bv.remove_user_function = lambda fn: bv.events.append(("remove_attempt_user", int(fn.start)))
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_create(None, "0x1000", True)

    assert result["success"] is False
    assert result["rolled_back"] is False
    assert "may be left modified" in result["message"]
    assert bv.get_function_at(0x1000) is not None
    # The per-op status must not keep claiming 'verified' when the revert failed
    # and the function persists -- route it to 'failed:' like the batch engine.
    assert result["results"][0]["status"] == "rollback_failed"
    from bn.formatters import FAILED_MUTATION_STATUSES
    assert "rollback_failed" in FAILED_MUTATION_STATUSES


def test_batch_op_function_create_verified_and_restore_no_poison(monkeypatch):
    # #308: function_create is a batch op. It creates+verifies and registers a
    # restore that removes the function on revert via the non-poisoning
    # remove_function (#304), so a later create still works.
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    restores = []
    res = me._op_function_create(ctx, bv, {"op": "function_create", "address": "0x1000"}, restores)
    assert res["status"] == "verified"
    assert res["function"] == "sub_1000"
    assert bv.get_function_at(0x1000) is not None
    assert len(restores) == 1
    restores[0]()                                   # batch revert
    assert bv.get_function_at(0x1000) is None        # removed
    assert ("remove_function", 0x1000) in bv.events  # non-poisoning removal
    # re-create after the revert still works (not poisoned)
    res2 = me._op_function_create(ctx, bv, {"op": "function_create", "address": "0x1000"}, [])
    assert res2["status"] == "verified"
    assert bv.get_function_at(0x1000) is not None


def test_batch_op_function_create_existing_is_noop(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx
    bv = _FakeFunctionCreateBV(
        functions=[_FakeFunction(0x1000, "already_here")],
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    res = me._op_function_create(ctx, bv, {"op": "function_create", "address": "0x1000"}, [])
    assert res["status"] == "noop"
    assert res["function"] == "already_here"
    assert bv.added == []


def test_batch_op_function_create_rejects_non_executable(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx
    bv = _FakeFunctionCreateBV(
        segments={0x5000: _FakeSegment(readable=True, writable=True, executable=False)},
        memory={0x5000: b"\x01\x02\x03\x04"},
    )
    with pytest.raises(bridge.OperationFailure) as exc:
        me._op_function_create(ctx, bv, {"op": "function_create", "address": "0x5000"}, [])
    assert exc.value.status == "invalid_request"
    assert "executable" in str(exc.value)
    assert bv.added == []


def test_apply_operation_dispatches_function_create(monkeypatch):
    # Teeth for the _apply_operation dispatch ARM (#308): a function_create op
    # must route to _op_function_create, not the "unsupported" fallthrough.
    # (Reverting the dispatch line makes this fail.)
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    restores = []
    res = me._apply_operation(ctx, bv, {"op": "function_create", "address": "0x1000"}, restores)
    assert res["op"] == "function_create"
    assert res["status"] == "verified"
    assert len(restores) == 1  # the remove-on-revert restore was registered


def test_verify_operation_dispatches_function_create(monkeypatch):
    # Teeth for the _verify_operation ARM: a function_create result must route to
    # _verify_function_create, not the "Unsupported verification path" raise (which
    # _verify_operation would catch and stamp status="unsupported").
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx
    bv = _FakeFunctionCreateBV(
        functions=[_FakeFunction(0x1000, "sub_1000")],
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    out = me._verify_operation(
        ctx, bv,
        {"op": "function_create", "address": "0x1000", "status": "verified", "function": "sub_1000"})
    assert out["status"] == "verified"  # would be "unsupported" without the arm
