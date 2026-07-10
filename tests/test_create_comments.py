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


class _FunctionCreateBV(_FakeBV):
    """Models the BN reality #360 hinges on: ``add_function`` is an advisory
    auto-analysis hint that DECLINES an address auto-analysis already chose to
    skip (a data-table / missed-handler entry), while ``create_user_function``
    forces a user-defined function there. Only the forced path creates."""

    def __init__(self, addr):
        super().__init__()
        self._addr = int(addr)
        self._created = False
        self.add_function_called = False
        self.create_user_function_called = False

    def read(self, addr, length):
        return b"\x90" * length  # mapped, returns bytes

    def get_function_at(self, addr):
        if self._created and int(addr) == self._addr:
            return _FakeFunction(self._addr, f"sub_{self._addr:x}")
        return None

    def add_function(self, addr, *a, **k):
        # advisory hint -- a no-op on a skipped address (the #360 failure mode)
        self.add_function_called = True

    def create_user_function(self, addr, *a, **k):
        self.create_user_function_called = True
        self._created = True
        return _FakeFunction(self._addr, f"sub_{self._addr:x}")

    def begin_undo_actions(self):
        return "state"

    def commit_undo_actions(self, state):
        pass

    def revert_undo_actions(self, state):
        pass

    def update_analysis_and_wait(self):
        pass


def test_function_create_uses_forced_create_on_skipped_address(monkeypatch):
    """function create must succeed on its documented use-case: an address
    auto-analysis skipped (data-table / missed-handler entry). It must use the
    forced ``create_user_function`` -- ``add_function`` is only an advisory hint
    and declines exactly those addresses, so the op returned verification_failed
    on the very case it exists for (#360)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    addr = 0x401abc
    bv = _FunctionCreateBV(addr)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.read_misc, "_is_executable_address",
                        lambda ctx, _bv, _addr: True)

    result = bridge.create_comments._function_create(instance.ctx, None, hex(addr), False)

    assert result["results"][0]["status"] == "verified"
    assert result["committed"] is True
    assert bv.create_user_function_called is True
    assert bv.add_function_called is False  # the advisory hint must NOT be the path


def test_batch_comment_ops_require_one_locator(monkeypatch):
    """set_comment / delete_comment target a function OR an address; a manifest
    op with neither is rejected with a clear invalid_request, not silently
    no-op'd against address 0 (#173, locator parity with #67/#94)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()

    for op in (
        {"op": "set_comment", "comment": "hi"},   # no function, no address
        {"op": "delete_comment"},                 # no function, no address
    ):
        with pytest.raises(bridge.OperationFailure) as exc:
            instance._apply_operation(bv, op)
        assert exc.value.status == "invalid_request"
        assert "one of" in exc.value.message
        assert "function" in exc.value.message and "address" in exc.value.message


def test_batch_comment_ops_reject_both_locators(monkeypatch):
    """function and address target DIFFERENT locations; passing both is
    ambiguous and rejected, rather than silently honoring one and dropping the
    other (#94 parity, now asserted for the batch path) (#173)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()

    for op in (
        {"op": "set_comment", "comment": "hi", "function": "f", "address": "0x1000"},
        {"op": "delete_comment", "function": "f", "address": "0x1000"},
    ):
        with pytest.raises(bridge.OperationFailure) as exc:
            instance._apply_operation(bv, op)
        assert exc.value.status == "invalid_request"
        assert "not both" in exc.value.message


def test_diff_snapshots_marks_comment_only_change(monkeypatch):
    """A comment set/delete changes no HLIL body text, so a text-only snapshot
    reports changed:false with an empty diff. The diff/changed signal must also
    reflect comment state (#121)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    diffs = instance._diff_snapshots(
        {0x401000: {"name": "f", "address": "0x401000", "text": "return 7;", "comments": {}, "locals": {}}},
        {0x401000: {"name": "f", "address": "0x401000", "text": "return 7;",
                    "comments": {"0x401010": "decryption key"}, "locals": {}}},
    )

    assert len(diffs) == 1
    assert diffs[0]["changed"] is True
    assert diffs[0]["diff"]
    assert "decryption key" in diffs[0]["diff"]


def test_capture_function_snapshots_reads_global_comment_store(monkeypatch):
    """Comment ops write to BN's GLOBAL comment store (bv.set_comment_at /
    bv.address_comments), which is a DIFFERENT store from Function.comments.
    The snapshot must read the global store filtered to the function -- reading
    Function.comments (as a first cut did) sees nothing the op wrote and the
    comment --preview still shows changed:false (#121)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(bridge.il_format, "_function_text", lambda bv, fn, view="hlil": "body")
    fn = _FakeFunction(0x401000, "f")
    fn.basic_blocks = [_FakeBasicBlock(0x401000, 0x401020)]
    # Function.comments is the WRONG (function-local) store -- it must be ignored.
    fn.comments = {0x401000: "stale-local-store"}
    bv = _FakeMutationBV()
    bv.functions = [fn]
    # Where bv.set_comment_at actually lands: in-function + an out-of-function one.
    bv.address_comments = {0x401004: "decryption key", 0x500000: "outside"}

    snaps = instance._capture_function_snapshots(bv, [fn])

    assert snaps[0x401000]["comments"] == {"0x401004": "decryption key"}
    assert snaps[0x401000]["locals"] == {}


def test_apply_operation_comment_function_only_form_accepted(monkeypatch):
    # The documented function-only comment form (no `address`) must pass
    # required-field validation, not be rejected as missing `address` (#67).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(bridge.mutation_engine, "_op_set_comment", lambda ctx, bv, op: {"ok": "set"})
    monkeypatch.setattr(bridge.mutation_engine, "_op_delete_comment", lambda ctx, bv, op: {"ok": "del"})
    assert instance._apply_operation(
        None, {"op": "set_comment", "function": "main", "comment": "hi"}) == {"ok": "set"}
    assert instance._apply_operation(
        None, {"op": "delete_comment", "function": "main"}) == {"ok": "del"}


def test_apply_operation_comment_address_form_still_accepted(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(bridge.mutation_engine, "_op_set_comment", lambda ctx, bv, op: {"ok": "set"})
    assert instance._apply_operation(
        None, {"op": "set_comment", "address": "0x1000", "comment": "hi"}) == {"ok": "set"}


def test_apply_operation_comment_requires_function_or_address(monkeypatch):
    # Neither locator field present -> precise invalid_request naming both options.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._apply_operation(None, {"op": "set_comment", "comment": "hi"})
    assert exc.value.status == "invalid_request"
    assert "function" in str(exc.value) and "address" in str(exc.value)


def test_apply_operation_set_comment_missing_comment_still_rejected(monkeypatch):
    # The genuinely-required field is still enforced precisely.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._apply_operation(None, {"op": "set_comment", "function": "main"})
    assert exc.value.status == "invalid_request"
    assert "comment" in str(exc.value)


# ---------------------------------------------------------------------------
# Batch 5: bridge-side validation (#94 comment guard, #100 count validation)
# ---------------------------------------------------------------------------


def test_get_comment_rejects_both_locators(monkeypatch):
    # #94: a raw socket client sending both function and address must be rejected
    # (the CLI mutex group doesn't protect raw clients).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: _FakeBV())
    with pytest.raises(RuntimeError, match="not both"):
        instance._get_comment("active", "0x1000", "main")


def test_get_comment_rejects_unmapped_address(monkeypatch):
    """comment get on an unmapped address must reject (exit 2) like read/decompile,
    not return a false 'no comment' (exit 0) for a typo'd/stale address (#374)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    bv.is_valid_offset = lambda addr: False
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    with pytest.raises(RuntimeError, match="not mapped"):
        instance._get_comment("active", "0xdeadbeef", None)


def test_get_comment_unmapped_but_commented_address_returns_comment(monkeypatch):
    """A real comment must never be suppressed: if the address carries a comment,
    return it even when is_valid_offset says unmapped -- only a genuinely-empty
    AND unmapped address is rejected (#374 follow-up, mirrors the xrefs refs
    gate)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(comments={0x1000: "real comment here"})
    bv.is_valid_offset = lambda addr: False
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._get_comment("active", "0x1000", None)
    assert result["has_comment"] is True
    assert result["comment"] == "real comment here"


def test_get_comment_mapped_address_without_comment_stays_clean(monkeypatch):
    """A MAPPED address with no comment is a clean has_comment:false / exit 0 --
    only the unmapped case is rejected (#374)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(comments={})
    bv.is_valid_offset = lambda addr: True
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._get_comment("active", "0x1000", None)
    assert result["has_comment"] is False
    assert result["comment"] == ""


def test_get_comment_function_aggregates_body_comments(monkeypatch):
    """`comment get --function` must aggregate ALL comments within the function's
    address range (matching `comment list`'s attribution), not just the
    entry-address comment -- a function with body comments but no entry comment
    previously reported (no comment), contradicting `comment list` (#203)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    fn = _FakeFunction(0x40a810, "ns::Cls::rwBuffer")
    fn.basic_blocks = [_FakeBasicBlock(0x40a810, 0x410900)]  # covers the body addrs

    class _CommentBV(_FakeBV):
        def get_comment_at(self, addr):
            return self.address_comments.get(int(addr), "")

    bv = _CommentBV(functions=[fn])
    # body comments only -- nothing at the entry address 0x40a810
    bv.address_comments = {0x4105f4: "validate length here", 0x410858: "off-by-one risk"}
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.ctx, "_find_function", lambda _bv, ident: fn)

    result = instance._get_comment(None, None, "ns::Cls::rwBuffer")
    assert result["has_comment"] is True
    assert result["function"] == "ns::Cls::rwBuffer"
    assert [c["address"] for c in result["comments"]] == ["0x4105f4", "0x410858"]
    assert [c["comment"] for c in result["comments"]] == [
        "validate length here", "off-by-one risk"
    ]


def test_set_comment_function_targets_fn_comment_not_address(monkeypatch):
    """`comment set --function` writes BN's real whole-function documentation
    property (fn.comment), NOT an address comment at the function's entry --
    the two are different stores (function-doc vs. address comment) and this
    is the deliberate repurposing of the --function locator (Task 8)."""
    bridge = _load_bridge(monkeypatch)
    mutation_engine = bridge.mutation_engine

    fn = _FakeFunction(0x1000, "sub_1000")
    fn.basic_blocks = [_FakeBasicBlock(0x1000, 0x1040)]
    fn.comment = ""
    bv = _FakeCommentMutationBV()
    bv.functions = [fn]

    class _C:
        def _resolve_view(self, s): return bv
        def _find_function(self, b, ident): return fn

    op = {"op": "set_comment", "function": "sub_1000", "comment": "documents the parser"}
    result = mutation_engine._op_set_comment(_C(), bv, op)
    assert fn.comment == "documents the parser"
    # the address store at the entry must NOT be touched
    assert bv.get_comment_at(0x1000) == ""
    assert result.get("scope") == "function_doc"

    verified = mutation_engine._verify_set_comment(_C(), bv, result)
    assert verified["status"] == "verified"


def test_delete_comment_function_clears_fn_comment(monkeypatch):
    """`comment delete --function` clears fn.comment (sets it to ""), the same
    function-doc store `comment set --function` now writes."""
    bridge = _load_bridge(monkeypatch)
    mutation_engine = bridge.mutation_engine

    fn = _FakeFunction(0x1000, "sub_1000")
    fn.comment = "old doc"
    bv = _FakeCommentMutationBV()
    bv.functions = [fn]

    class _C:
        def _resolve_view(self, s): return bv
        def _find_function(self, b, ident): return fn

    op = {"op": "delete_comment", "function": "sub_1000"}
    result = mutation_engine._op_delete_comment(_C(), bv, op)
    assert fn.comment == ""
    verified = mutation_engine._verify_delete_comment(_C(), bv, result)
    assert verified["status"] == "verified"


def test_op_set_comment_rejects_both_locators(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._op_set_comment(_FakeBV(), {"op": "set_comment", "function": "main", "address": "0x1000", "comment": "x"})
    assert exc.value.status == "invalid_request"


def test_op_delete_comment_rejects_both_locators(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._op_delete_comment(_FakeBV(), {"op": "delete_comment", "function": "main", "address": "0x1000"})
    assert exc.value.status == "invalid_request"


def test_list_comments_rejects_negative_count(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    bv.address_comments = {}
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._list_comments("active", limit=-3)
    assert exc.value.status == "invalid_request"


def test_list_comments_returns_paging_envelope(monkeypatch):
    # #131: comment list returns the {items,total,offset,limit,returned,has_more}
    # envelope (parity with strings/imports/sections), not a bare list.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    bv.address_comments = {0x1000: "first", 0x2000: "second", 0x3000: "third"}
    bv.get_functions_containing = lambda addr: []
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    res = instance._list_comments("active", offset=0, limit=2)
    assert res["total"] == 3
    assert res["returned"] == 2
    assert res["has_more"] is True
    assert [i["comment"] for i in res["items"]] == ["first", "second"]


def test_bind_list_comments_tolerates_none_limit(monkeypatch):
    """The CLI sends `limit: None` (the --limit default) for a bare `comment
    list`, so the key is present with value None. The binder must read that as
    "no limit" -- guarding on `params.get("limit") is not None`, like every
    sibling list binder -- not on key presence, which does int(None) and crashes
    the command with a raw `int() argument must be ... not 'NoneType'` TypeError
    (regression from the #131 paging-envelope adoption)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    bv.address_comments = {0x1000: "first", 0x2000: "second"}
    bv.get_functions_containing = lambda addr: []
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # Exactly what the CLI forwards for `bn comment list` with no --limit/--offset.
    res = bridge._bind_list_comments(instance, {"query": None, "offset": 0, "limit": None}, "active")
    assert res["total"] == 2
    assert res["returned"] == 2
    assert res["limit"] is None       # None means "no limit", parity with siblings
    assert res["has_more"] is False

    # An explicit --limit still pages.
    res2 = bridge._bind_list_comments(instance, {"query": None, "offset": 0, "limit": 1}, "active")
    assert res2["returned"] == 1
    assert res2["has_more"] is True
