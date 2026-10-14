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
    assert _has_event(bv, "commit")
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
    assert _has_event(bv, "revert")
    assert not _has_event(bv, "commit")


def test_function_create_reanalysis_runs_under_gate_not_exclusive_lock(monkeypatch):
    """#628: function_create self-manages locking. The create itself is exclusive,
    but the post-create update_analysis_and_wait() runs under the write gate ONLY
    (so concurrent reads stay live), and the readback/commit phase is exclusive
    again. Same protocol as the batch mutation engine."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    lock = instance._target_lock
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    states: dict = {}
    create = bv.create_user_function

    def create_user_function(addr):
        states["create_writer"] = lock._writer
        states["create_gate"] = instance._write_gate.locked()
        return create(addr)

    bv.create_user_function = create_user_function
    original = bv.update_analysis_and_wait
    waits: list = []

    def analyze():
        waits.append((lock._writer, instance._write_gate.locked()))
        original()

    bv.update_analysis_and_wait = analyze

    result = instance._function_create(None, "0x1000", False)

    assert result["committed"] is True
    assert states["create_writer"] is True   # create phase holds the exclusive lock
    assert states["create_gate"] is True     # ...and is serialized as a writer
    assert waits == [(False, True)]          # post-create reanalysis: gate only
    assert lock._writer is False             # both released at the end
    assert instance._write_gate.locked() is False


def test_function_create_preview_revert_settle_still_runs_under_exclusive_lock(monkeypatch):
    """#628: on a --preview the post-create reanalysis is relaxed, but the revert's
    own reanalysis (removing the created function, a non-journaled creation) stays
    under the exclusive lock -- a reader must never observe a half-reverted view."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    lock = instance._target_lock
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    original = bv.update_analysis_and_wait
    waits: list = []

    def analyze():
        waits.append((lock._writer, instance._write_gate.locked()))
        original()

    bv.update_analysis_and_wait = analyze

    result = instance._function_create(None, "0x1000", True)

    assert result["rolled_back"] is True
    assert waits == [(False, True), (True, True)]


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
    """M-24: the standalone refusal raised a bare RuntimeError (exit 2) where
    the batch op raised `invalid_request` (exit 3) for the same address. A
    bridge mutation refusal is 3 on both surfaces."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure, match="0xdead.*not mapped") as exc:
        instance._function_create(None, "0xdead", False)

    assert exc.value.status == "invalid_request"
    assert exc.value.requested == {"op": "function_create", "address": "0xdead"}
    assert bv.added == []


def test_function_create_unmapped_address_refuses_alike_standalone_and_batch(monkeypatch):
    """The two surfaces must classify the refusal identically, or the exit
    code depends on which one the agent happened to use (M-24)."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure) as standalone:
        instance._function_create(None, "0xdead", False)
    with pytest.raises(bridge.OperationFailure) as batch:
        me._op_function_create(instance.ctx, bv,
                               {"op": "function_create", "address": "0xdead"}, [])

    assert standalone.value.status == batch.value.status == "invalid_request"
    assert standalone.value.message == batch.value.message


def test_function_create_non_executable_address_is_rejected(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x5000: _FakeSegment(readable=True, writable=True, executable=False)},
        memory={0x5000: b"\x01\x02\x03\x04"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure,
                       match="0x5000.*not inside an executable segment") as exc:
        instance._function_create(None, "0x5000", False)

    assert exc.value.status == "invalid_request"
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


def test_function_create_refused_on_quick_view_without_analysis(monkeypatch):
    """#479: on a --quick-loaded view, function create must fail fast (pointing at
    `bn refresh`) instead of triggering full analysis under the write lock -- which
    wedged the instance so even later `target info` reads hung. It must NOT start
    analysis or create anything. #927 review: the refusal is the batch op's
    `OperationFailure("invalid_request")` (exit 3 on both surfaces), not a bare
    RuntimeError that put the standalone command at exit 2 -- M-24's twin."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    bridge._quick_loaded_views.add(bv)

    with pytest.raises(bridge.OperationFailure, match="--quick") as exc:
        instance._function_create(None, "0x1000", False)
    assert exc.value.status == "invalid_request"
    assert exc.value.requested == {"op": "function_create", "address": "0x1000"}

    # No analysis kicked off, nothing created -> no wedge, no side effects.
    assert "refresh" not in bv.events
    assert bv.added == []
    bridge._quick_loaded_views.discard(bv)


def test_function_create_preview_refused_on_quick_view(monkeypatch):
    """#479: the --preview path must also refuse on a quick view -- a preview that
    wedges the instance defeats the point of a bounded, revertible probe."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    bridge._quick_loaded_views.add(bv)

    with pytest.raises(bridge.OperationFailure, match="bn refresh") as exc:
        instance._function_create(None, "0x1000", True)
    assert exc.value.status == "invalid_request"

    assert "refresh" not in bv.events
    assert bv.added == []
    bridge._quick_loaded_views.discard(bv)


def test_op_function_create_refused_on_quick_view(monkeypatch):
    """#479: the batch function_create op must also refuse on a quick view with a
    structured invalid_request, not run analysis under the write lock."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    bridge._quick_loaded_views.add(bv)

    with pytest.raises(bridge.OperationFailure) as exc:
        me._op_function_create(ctx, bv, {"op": "function_create", "address": "0x1000"}, [])
    assert exc.value.status == "invalid_request"
    assert "--quick" in str(exc.value)
    assert "refresh" not in bv.events
    assert bv.added == []
    bridge._quick_loaded_views.discard(bv)


def test_op_function_create_rejects_unaligned_on_fixed_width_isa(monkeypatch):
    """Forced create_user_function lands a function even on non-code (#386). An
    unaligned start on a fixed-width ISA (aarch64/MIPS) can't be a real function
    start -- reject + remove (non-poisoning) instead of reporting it verified."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx
    bv = _FakeFunctionCreateBV(
        segments={0x1001: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x00\x01\x02\x03\x04\x05\x06\x07"},
        arch=_FakeArch(name="aarch64", instr_alignment=4),
    )
    with pytest.raises(bridge.OperationFailure) as exc:
        me._op_function_create(ctx, bv, {"op": "function_create", "address": "0x1001"}, [])
    assert exc.value.status == "verification_failed"
    assert bv.get_function_at(0x1001) is None              # junk removed
    assert ("remove_function", 0x1001) in bv.events        # non-poisoning removal


def test_op_function_create_rejects_undecodable_start(monkeypatch):
    """An in-segment, aligned address BN can't decode an instruction at
    (get_instruction_length == 0) is not code -- reject, don't fabricate."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx
    bv = _FakeFunctionCreateBV(
        segments={0x2000: _FakeSegment(readable=True, executable=True)},
        memory={0x2000: b"\xff\xff\xff\xff"},
        arch=_FakeArch(name="aarch64", instr_alignment=4),
        instruction_lengths={0x2000: 0},
    )
    with pytest.raises(bridge.OperationFailure) as exc:
        me._op_function_create(ctx, bv, {"op": "function_create", "address": "0x2000"}, [])
    assert exc.value.status == "verification_failed"
    assert bv.get_function_at(0x2000) is None


def test_op_function_create_rejects_empty_body(monkeypatch):
    """A created function with an empty body (total_bytes == 0) is not code."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx
    bv = _FakeFunctionCreateBV(
        segments={0x3000: _FakeSegment(readable=True, executable=True)},
        memory={0x3000: b"\x00\x00\x00\x00"},
        created_total_bytes=0,
    )
    with pytest.raises(bridge.OperationFailure) as exc:
        me._op_function_create(ctx, bv, {"op": "function_create", "address": "0x3000"}, [])
    assert exc.value.status == "verification_failed"
    assert bv.get_function_at(0x3000) is None


def test_op_function_create_accepts_aligned_decodable_code(monkeypatch):
    """The guard must NOT false-positive on real, aligned, decodable code -- a
    legitimate missed-handler recovery (the #360 use-case) still verifies."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx
    bv = _FakeFunctionCreateBV(
        segments={0x4000: _FakeSegment(readable=True, executable=True)},
        memory={0x4000: b"\x55\x48\x89\xe5"},
        arch=_FakeArch(name="aarch64", instr_alignment=4),
    )
    res = me._op_function_create(ctx, bv, {"op": "function_create", "address": "0x4000"}, [])
    assert res["status"] == "verified"
    assert bv.get_function_at(0x4000) is not None


def test_function_create_standalone_rejects_unaligned_address(monkeypatch):
    """The standalone (non-batch) path applies the same #386 guard: reject and
    roll back instead of returning success:true for a fabricated function."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1001: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x00\x01\x02\x03\x04\x05\x06\x07"},
        arch=_FakeArch(name="aarch64", instr_alignment=4),
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_create(None, "0x1001", False)

    assert result["success"] is False
    assert result["committed"] is False
    assert result["results"][0]["status"] == "verification_failed"
    assert bv.get_function_at(0x1001) is None


def test_function_create_commit_marks_view_dirty(monkeypatch):
    """#519: a committed standalone function create must mark the target dirty so
    `bn close` warns about unsaved changes. This path bypasses the generic
    _mutation() dirty-marking shim, and BN's bv.file.modified never flips True for
    our verified mutations -- so without the mark, close reports unsaved=false and
    silently drops the created function."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    dirtied: list = []
    monkeypatch.setattr(instance.ctx.targets, "mark_dirty", lambda b: dirtied.append(b))

    result = instance._function_create(None, "0x1000", False)

    assert result["committed"] is True
    assert result["results"][0]["status"] == "verified"
    assert dirtied == [bv]


def test_function_create_preview_does_not_mark_view_dirty(monkeypatch):
    """#519: a --preview create reverts, so it must NOT dirty the view."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    dirtied: list = []
    monkeypatch.setattr(instance.ctx.targets, "mark_dirty", lambda b: dirtied.append(b))

    result = instance._function_create(None, "0x1000", True)

    assert result["committed"] is False
    assert dirtied == []


def test_function_create_noop_does_not_mark_view_dirty(monkeypatch):
    """#519: creating a function where one already exists is a no-op and changes
    nothing, so it must NOT dirty the view."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        functions=[_FakeFunction(0x1000, "already_here")],
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    dirtied: list = []
    monkeypatch.setattr(instance.ctx.targets, "mark_dirty", lambda b: dirtied.append(b))

    result = instance._function_create(None, "0x1000", False)

    assert result["results"][0]["status"] == "noop"
    assert dirtied == []


def test_function_create_preview_revert_failure_marks_view_dirty(monkeypatch):
    """#545: a --preview create whose revert FAILS leaves the fabricated function
    live in the view, yet BN's bv.file.modified never flips True for our create --
    so without a dirty mark `bn close` would compute unsaved=false and never warn
    about the leftover. The op still reports failure, but the view must be dirtied
    so close warns. (A clean preview revert stays non-dirtying -- see the sibling
    test.)"""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    # BOTH removal paths silently do nothing, so the function persists past the
    # revert -- the rollback_failed condition.
    bv.remove_function = lambda fn: bv.events.append(("remove_attempt", int(fn.start)))
    bv.remove_user_function = lambda fn: bv.events.append(("remove_attempt_user", int(fn.start)))
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    dirtied: list = []
    monkeypatch.setattr(instance.ctx.targets, "mark_dirty", lambda b: dirtied.append(b))

    result = instance._function_create(None, "0x1000", True)

    assert result["success"] is False
    assert result["results"][0]["status"] == "rollback_failed"
    assert bv.get_function_at(0x1000) is not None
    # The leftover function must dirty the view so `bn close` still warns.
    assert dirtied == [bv]


def test_op_function_create_guard_rejection_with_failed_removal_is_not_clean_rollback(monkeypatch):
    """#520: when the code guard rejects a just-created function AND removing it
    fails (create_user_function is not reliably undone), the batch must NOT report
    a clean rollback. The op registers the cleanup as a restore BEFORE the guard so
    the standard batch accounting covers it, and that restore RAISES on removal
    failure so _run_local_restores marks restore_ok=False -> rolled_back=false."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx
    bv = _FakeFunctionCreateBV(
        segments={0x1001: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x00\x01\x02\x03\x04\x05\x06\x07"},
        arch=_FakeArch(name="aarch64", instr_alignment=4),
    )
    # Both removal paths silently do nothing, so the fabricated function persists
    # past cleanup -> _remove_created_function returns False.
    bv.remove_function = lambda fn: bv.events.append(("remove_attempt", int(fn.start)))
    bv.remove_user_function = lambda fn: bv.events.append(("remove_attempt_user", int(fn.start)))

    restores: list = []
    with pytest.raises(bridge.OperationFailure) as exc:
        me._op_function_create(ctx, bv, {"op": "function_create", "address": "0x1001"}, restores)

    assert exc.value.status == "verification_failed"
    # A restore was registered DESPITE the guard rejection (before-guard registration).
    assert len(restores) == 1
    # Removal failed, so the fabricated function is still present.
    assert bv.get_function_at(0x1001) is not None
    # The batch rollback restores must therefore report FAILURE -- the batch may
    # not claim rolled_back=true while the fabricated function persists.
    assert me._run_local_restores(ctx, bv, restores) is False
    assert bv.get_function_at(0x1001) is not None


def test_op_function_create_guard_rejection_with_successful_removal_is_clean_rollback(monkeypatch):
    """#520 companion: when the guard rejects a just-created function AND the
    inline removal SUCCEEDS, the restore closure registered before the guard must
    be a harmless no-op on batch rollback (the function is already gone) and must
    NOT raise -- otherwise the double-removal would falsely flip a clean rollback
    to rolled_back=false. Locks down the inline-removal + restore ordering."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx
    bv = _FakeFunctionCreateBV(
        segments={0x1001: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x00\x01\x02\x03\x04\x05\x06\x07"},
        arch=_FakeArch(name="aarch64", instr_alignment=4),
    )
    # remove_function is left at its default -- it actually removes the function,
    # so the inline cleanup on guard rejection succeeds.
    restores: list = []
    with pytest.raises(bridge.OperationFailure) as exc:
        me._op_function_create(ctx, bv, {"op": "function_create", "address": "0x1001"}, restores)

    assert exc.value.status == "verification_failed"
    # A restore was registered before the guard, even though inline removal ran.
    assert len(restores) == 1
    # The inline removal already dropped the fabricated function.
    assert bv.get_function_at(0x1001) is None
    # The registered restore is therefore a no-op: it must succeed (idempotent,
    # fn already gone) and NOT raise -> the batch reports a clean rollback.
    assert me._run_local_restores(ctx, bv, restores) is True
    assert bv.get_function_at(0x1001) is None


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


def test_function_create_rejects_unmapped_address_with_no_memory_seeded_at_all(monkeypatch):
    """#616 regression: previously, a _FakeFunctionCreateBV with an entirely
    empty memory map (no `memory=` kwarg) made bv.read() return fabricated
    \\x90 filler for ANY address, so the create_comments.py mappedness guard
    (`len(bytes(bv.read(addr, 1))) == 0`) never fired and this production
    branch went untested. With a faithful empty-read default, the guard now
    correctly rejects the address as unmapped."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
    )  # no memory= at all -- every address is unmapped
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(RuntimeError, match="0x1000.*not mapped"):
        instance._function_create(None, "0x1000", False)

    assert bv.added == []


class _A64Arch:
    """A fixed-width ISA: every word decodes as the instruction the table names,
    with the branch kinds BN's InstructionInfo would report for it."""
    instr_alignment = 4
    max_instr_length = 4

    def __init__(self, program):
        self.program = program          # address -> (mnemonic, [branch kind names])

    def get_instruction_info(self, data, address):
        entry = self.program.get(int(address))
        if entry is None:
            return None
        return types.SimpleNamespace(length=4, branches=[
            types.SimpleNamespace(type=types.SimpleNamespace(name=kind)) for kind in entry[1]])

    def get_instruction_text(self, data, address):
        entry = self.program.get(int(address))
        return ([entry[0]], 4) if entry else None


@pytest.mark.parametrize(
    ("previous", "expected"),
    [
        pytest.param(("add", []), 0x402ffc, id="straight-line-falls-through"),
        pytest.param(("b.ne", ["TrueBranch", "FalseBranch"]), 0x402ffc,
                     id="conditional-branch-not-taken-falls-through"),
        pytest.param(("ret", ["FunctionReturn"]), None, id="after-ret-is-an-entry"),
        pytest.param(("b", ["UnconditionalBranch"]), None, id="after-b-is-an-entry"),
        pytest.param(("bl", ["CallDestination"]), None, id="after-a-call-may-be-noreturn"),
        pytest.param(("nop", []), None, id="padding-is-not-a-body"),
        pytest.param(None, None, id="undecodable-says-nothing"),
    ],
)
def test_fallthrough_source_names_the_instruction_running_into_the_address_997(
        monkeypatch, previous, expected):
    """#997 item 2: `function create` accepted an `adrp` page base in the middle of
    a loop body. On a fixed-width ISA the instruction before the address is known,
    and if it neither returns nor branches away, execution runs straight in: the
    address is mid-body. Anything the bytes do not settle answers None."""
    bridge = _load_bridge(monkeypatch)
    program = {0x402ffc: previous} if previous else {}
    bv = types.SimpleNamespace(arch=_A64Arch(program), read=lambda a, n: b"\x00" * n)
    assert bridge.create_comments._create_fallthrough_source(bv, 0x403000) == expected


def test_fallthrough_source_is_silent_on_a_variable_length_isa(monkeypatch):
    """x86: any byte before the address may start an instruction, so no single
    predecessor exists to name."""
    bridge = _load_bridge(monkeypatch)
    arch = _A64Arch({0x402fff: ("add", [])})
    arch.instr_alignment, arch.max_instr_length = 1, 15
    bv = types.SimpleNamespace(arch=arch, read=lambda a, n: b"\x00" * n)
    assert bridge.create_comments._create_fallthrough_source(bv, 0x403000) is None


def test_fallthrough_create_is_disclosed_on_the_compact_status_997():
    """The warning rides on the line every loop reads, not only the verbose row."""
    from bn.formatters import _mutation_summary, _render_mutation_summary_text
    row = {"op": "function_create", "status": "verified", "address": "0x403000",
           "fallthrough_from": "0x402ffc", "note": "..."}
    summary = _mutation_summary({"success": True, "committed": False, "preview": True,
                                 "rolled_back": True, "results": [row]})
    assert summary["fallthrough_entries"] == [
        {"address": "0x403000", "fallthrough_from": "0x402ffc"}]
    assert ("warning: 0x403000 is reached by fall-through from 0x402ffc; likely "
            "mid-function, not an entry") in _render_mutation_summary_text(summary)
    plain = _mutation_summary({"success": True, "committed": True, "results": [
        {"op": "function_create", "status": "verified", "address": "0x403110"}]})
    assert "fallthrough_entries" not in plain


def test_fallthrough_probe_ignores_literal_pools_and_mixed_arm_thumb_views_997(monkeypatch):
    """#1006 review: an ARMv7 literal-pool word before a real entry decodes as
    `andeq` with no branches and read as a fall-through. A word BN holds as a
    data variable after a `ret` is a literal pool, not an instruction running
    into the address -- but one that a straight-line instruction runs into is
    code that something merely loads (round 2), and still warns. An ARM view
    mixes ARM and Thumb code, so the instruction before an address is not
    knowable from the view's architecture at all."""
    bridge = _load_bridge(monkeypatch)
    probe = bridge.create_comments._create_fallthrough_source
    program = {0x402ffc: ("add", [])}

    def view(arch, data_vars=()):
        return types.SimpleNamespace(arch=arch, read=lambda a, n: b"\x00" * n,
                                     get_data_var_at=lambda a: object() if a in data_vars else None)

    assert probe(view(_A64Arch(program)), 0x403000) == 0x402ffc
    after_ret = {0x402ff8: ("ret", ["FunctionReturn"]), **program}
    assert probe(view(_A64Arch(after_ret), data_vars={0x402ffc}), 0x403000) is None
    pool_run = {0x402ff8: ("add", []), **program}
    assert probe(view(_A64Arch(pool_run), data_vars={0x402ff8, 0x402ffc}), 0x403000) is None
    loaded_code = {0x402ff8: ("add", []), **program}
    assert probe(view(_A64Arch(loaded_code), data_vars={0x402ffc}), 0x403000) == 0x402ffc
    arm = _A64Arch(program)
    arm.name = "armv7"
    assert probe(view(arm), 0x403000) is None
