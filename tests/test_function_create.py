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
