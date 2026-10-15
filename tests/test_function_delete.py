from __future__ import annotations

from contextlib import contextmanager

import pytest

from _bridge_fakes import (
    _FakeBasicBlock, _FakeFunction, _FakeFunctionCreateBV, _has_event, _load_bridge,
)


def _setup(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    entry = _FakeFunction(0x1010, "false_tail")
    host = _FakeFunction(0x1000, "host")
    host.basic_blocks = [_FakeBasicBlock(0x1000, 0x1010)]
    bv = _FakeFunctionCreateBV(functions=[host, entry])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    return bridge, instance, bv, host, entry


def test_delete_observes_containment_without_claiming_full_tail_merge(monkeypatch):
    _, instance, bv, host, _ = _setup(monkeypatch)
    states = []

    def analyze():
        states.append((instance._target_lock._writer, instance._write_gate.locked()))
        bv.events.append("refresh")
        # Only the entry byte becomes owned; the original tail is not all merged.
        host.basic_blocks = [_FakeBasicBlock(0x1000, 0x1011)]

    monkeypatch.setattr(bv, "update_analysis_and_wait", analyze)
    result = instance._mutation(None, False, [{"op": "function_delete", "address": "0x1010"}])

    assert result["success"] is True
    assert result["committed"] is True
    assert result["rolled_back"] is False
    row = result["results"][0]
    assert row["status"] == "verified"
    assert row["op"] == "function_delete"
    assert row["observed"] == {
        "function_removed": True,
        "function_at_entry": None,
        "bodies_containing_deleted_entry": [{"address": "0x1000", "name": "host"}],
        "reanalysis_completed": True,
        "clean_undo_available": False,
    }
    assert ("remove_user_function", 0x1010) in bv.events
    assert ("remove_function", 0x1010) not in bv.events
    assert _has_event(bv, "begin") and _has_event(bv, "commit")
    assert not _has_event(bv, "revert")
    assert states == [(False, True)]
    assert "no clean undo" in result["message"].lower()


@pytest.mark.parametrize("operations,preview", [
    ([{"op": "function_delete", "address": "0x1010"}], True),
    ([{"op": "set_comment", "address": "0x1000", "comment": "must not write"},
      {"op": "function_delete", "address": "0x1010"}], False),
    ([{"op": "function_delete", "address": "0x1010"},
      {"op": "set_comment", "address": "0x1000", "comment": "must not write"}], False),
    ([{"op": "function_delete", "address": "0x1010"},
      {"op": "function_delete", "address": "0x1000"}], False),
])
def test_delete_refusals_precede_every_write(monkeypatch, operations, preview):
    bridge, instance, bv, _, entry = _setup(monkeypatch)
    with pytest.raises(bridge.OperationFailure) as failure:
        instance._mutation(None, preview, operations)
    assert failure.value.status == "unsupported"
    assert bv.events == []
    assert bv.get_function_at(0x1010) is entry
    assert bv.get_comment_at(0x1000) == ""


def test_delete_preview_refuses_before_journaling(monkeypatch):
    bridge, instance, bv, _, entry = _setup(monkeypatch)
    with pytest.raises(bridge.OperationFailure) as failure:
        instance._mutation(None, True, [{"op": "function_delete", "address": "0x1010"}])
    assert failure.value.status == "unsupported"
    assert bv.events == []
    assert bv.get_function_at(0x1010) is entry


@pytest.mark.parametrize("failure_mode", ["ignored", "raised", "recreated", "analysis"])
def test_delete_failure_never_claims_verified_or_clean_rollback(monkeypatch, failure_mode):
    _, instance, bv, _, entry = _setup(monkeypatch)
    if failure_mode == "ignored":
        monkeypatch.setattr(bv, "remove_user_function", lambda fn: None)
    elif failure_mode == "raised":
        def remove(fn):
            raise RuntimeError("removal failed")
        monkeypatch.setattr(bv, "remove_user_function", remove)
    elif failure_mode == "recreated":
        monkeypatch.setattr(bv, "update_analysis_and_wait", lambda: bv.functions.append(entry))
    else:
        def analyze():
            raise RuntimeError("analysis failed")
        monkeypatch.setattr(bv, "update_analysis_and_wait", analyze)

    result = instance._mutation(None, False, [{"op": "function_delete", "address": "0x1010"}])

    assert result["success"] is False
    assert result["committed"] is True
    assert result["rolled_back"] is False
    row = result["results"][0]
    assert row["status"] == "verification_failed"
    observed = row["observed"]
    assert observed["clean_undo_available"] is False
    assert observed["function_removed"] is (None if failure_mode == "analysis" else False)
    assert observed["function_at_entry"] == (None if failure_mode == "analysis" else "false_tail")
    if failure_mode == "analysis":
        assert observed["bodies_containing_deleted_entry"] is None
        assert bv.get_function_at(0x1010) is None
    assert observed["reanalysis_completed"] is (failure_mode != "analysis")
    if failure_mode == "raised":
        assert observed["apply_error"] == "removal failed"
    if failure_mode == "analysis":
        assert observed["analysis_error"] == "analysis failed"
    assert _has_event(bv, "commit")
    assert not _has_event(bv, "revert")


def test_delete_analysis_failure_compact_summary_reports_dirty_commit(monkeypatch):
    from bn.cli import _mutation_exit_code
    from bn.formatters import _mutation_summary, _render_mutation_summary_text

    _, instance, bv, _, _ = _setup(monkeypatch)

    def analyze():
        raise RuntimeError("analysis failed")

    monkeypatch.setattr(bv, "update_analysis_and_wait", analyze)
    result = instance._mutation(None, False, [{"op": "function_delete", "address": "0x1010"}])

    assert bv.get_function_at(0x1010) is None
    assert result["results"][0]["observed"]["analysis_error"] == "analysis failed"
    summary = _mutation_summary(result)
    assert summary["success"] is False
    assert summary["committed"] is True
    assert summary["rolled_back"] is False
    assert summary["changed_count"] == 1
    assert summary["dirty_after"] is True
    assert summary["failed_count"] == 1
    assert summary["verified_count"] == 0
    assert _mutation_exit_code(result, _mutation_summary) == 3
    rendered = _render_mutation_summary_text(summary)
    for field in ("changed=1", "dirty_after=True", "failed=1", "verified=0"):
        assert field in rendered, rendered


@pytest.mark.parametrize("failure_point", [
    "mark_dirty",
    "commit_before",
    "commit_after",
    "predecessor_lookup",
    "reanalyze",
    "analysis",
    "entry_readback",
    "entry_name",
    "containing_lookup",
    "containing_iteration",
    "containing_start",
    "containing_name",
    "scope_exit",
    "readback_scope_enter",
    "readback_scope_exit",
])
def test_delete_post_removal_exceptions_preserve_irreversible_result(monkeypatch, failure_point):
    from bn.cli import _mutation_exit_code
    from bn.formatters import _mutation_summary, _render_mutation_summary_text

    _, instance, bv, host, _ = _setup(monkeypatch)
    error = f"injected post-removal failure: {failure_point}"
    removed = False
    marks = []
    commits = []
    readback_reached = False
    targets = instance.ctx.targets
    view_id = "post-removal-test-view"
    targets._ids_by_object[bv] = view_id
    original_remove = bv.remove_user_function
    original_mark = targets.mark_dirty
    original_commit = bv.commit_undo_actions
    original_lookup = bv.get_function_at
    original_containing = bv.get_functions_containing

    def fail(*args, **kwargs):
        raise RuntimeError(error)

    def remove(fn):
        nonlocal removed
        original_remove(fn)
        removed = True

    def mark(view):
        assert not removed, "dirty tracking must precede the irreversible write"
        marks.append(view)
        if failure_point == "mark_dirty":
            fail()
        original_mark(view)

    def commit(state):
        commits.append(state)
        assert marks == [bv], "dirty tracking must precede even a failing commit"
        assert view_id in targets._dirty_view_ids
        if failure_point == "commit_before":
            fail()
        original_commit(state)
        if failure_point == "commit_after":
            fail()

    class BrokenReadback:
        @property
        def start(self):
            if failure_point == "containing_start":
                fail()
            return host.start

        @property
        def name(self):
            fail()

    def lookup(address):
        if removed:
            if failure_point == "predecessor_lookup" and address == host.start:
                fail()
            if address == 0x1010:
                if failure_point == "entry_readback":
                    fail()
                if failure_point == "entry_name":
                    return BrokenReadback()
        return original_lookup(address)

    def broken_iteration():
        # A partial observation must not become a complete containing-body list.
        yield host
        fail()

    def containing(address):
        nonlocal readback_reached
        readback_reached = True
        if failure_point == "containing_lookup":
            fail()
        if failure_point == "containing_iteration":
            return broken_iteration()
        if failure_point in {"containing_start", "containing_name"}:
            return [BrokenReadback()]
        return original_containing(address)

    monkeypatch.setattr(bv, "remove_user_function", remove)
    monkeypatch.setattr(targets, "mark_dirty", mark)
    monkeypatch.setattr(bv, "commit_undo_actions", commit)
    monkeypatch.setattr(bv, "get_function_at", lookup)
    monkeypatch.setattr(bv, "get_functions_containing", containing)
    if failure_point == "reanalyze":
        monkeypatch.setattr(host, "reanalyze", fail)
    if failure_point == "analysis":
        monkeypatch.setattr(bv, "update_analysis_and_wait", fail)
    if failure_point in {"scope_exit", "readback_scope_enter", "readback_scope_exit"}:
        original_write = instance._target_lock.write
        scope_failed = False

        @contextmanager
        def write_scope():
            nonlocal scope_failed
            with original_write():
                if removed and failure_point == "readback_scope_enter":
                    fail()
                yield
            if removed and not scope_failed and (
                failure_point == "scope_exit"
                or (failure_point == "readback_scope_exit" and readback_reached)
            ):
                scope_failed = True
                fail()

        monkeypatch.setattr(instance._target_lock, "write", write_scope)

    result = instance._mutation(None, False, [{"op": "function_delete", "address": "0x1010"}])

    assert removed is (failure_point != "mark_dirty")
    assert marks == [bv]
    assert (view_id in targets._dirty_view_ids) is (failure_point != "mark_dirty")
    if failure_point == "mark_dirty":
        assert original_lookup(0x1010) is not None
        assert not commits
    else:
        assert commits
    if failure_point == "commit_before":
        assert not _has_event(bv, "commit")
    if failure_point == "commit_after":
        assert _has_event(bv, "commit")
    assert not _has_event(bv, "revert")
    assert (0x1010 in bv._suppressed) is (failure_point != "mark_dirty")
    row = result["results"][0]
    assert row["status"] == "verification_failed"
    assert row["irreversible_change"] is True
    observed = row["observed"]
    assert observed["error"] == error
    assert isinstance(observed["error_stage"], str) and observed["error_stage"]
    if failure_point in {"scope_exit", "readback_scope_exit"}:
        assert observed["error_stage"] == "scope_exit"
        assert "apply_error" not in observed and "analysis_error" not in observed
    if failure_point == "readback_scope_enter":
        assert observed["error_stage"] == "scope_enter"
        assert "analysis_error" not in observed
    assert observed["clean_undo_available"] is False
    if failure_point in {"entry_readback", "entry_name"}:
        assert observed["function_at_entry"] is None
    if failure_point == "entry_readback":
        assert observed["function_removed"] is None
    if failure_point.startswith("containing_"):
        assert observed["bodies_containing_deleted_entry"] is None
    if failure_point in {"mark_dirty", "commit_before", "commit_after"}:
        assert observed["function_removed"] is None
        assert observed["function_at_entry"] is None
        assert observed["bodies_containing_deleted_entry"] is None
        assert observed["reanalysis_completed"] is False
    if failure_point in {"predecessor_lookup", "reanalyze", "analysis"}:
        assert observed["function_removed"] is None
        assert observed["function_at_entry"] is None
        assert observed["bodies_containing_deleted_entry"] is None
    if failure_point == "analysis":
        assert observed["analysis_error"] == error

    summary = _mutation_summary(result)
    for payload in (result, summary):
        assert payload["success"] is False
        assert payload["committed"] is True
        assert payload["rolled_back"] is False
    assert summary["changed_count"] == 1
    assert summary["dirty_after"] is True
    assert summary["failed_count"] == 1
    assert summary["verified_count"] == 0
    assert _mutation_exit_code(result, _mutation_summary) == 3
    rendered = _render_mutation_summary_text(summary)
    for field in ("changed=1", "dirty_after=True", "failed=1", "verified=0"):
        assert field in rendered, rendered


def test_single_delete_batch_uses_observed_mutation_result(monkeypatch):
    _, instance, bv, _, _ = _setup(monkeypatch)
    result = instance._mutation(None, False, [{"op": "function_delete", "address": "0x1010"}])
    assert result["success"] is True
    assert result["results"][0]["observed"]["bodies_containing_deleted_entry"] == []
    assert bv.get_function_at(0x1010) is None


def test_singleton_delete_preserves_result_when_post_apply_lock_raises(monkeypatch):
    from bn.formatters import _mutation_summary

    _, instance, bv, _, _ = _setup(monkeypatch)
    targets = instance.ctx.targets
    targets._ids_by_object[bv] = "post-apply-delete"
    original_write = instance._target_lock.write

    @contextmanager
    def post_apply_failure():
        with original_write():
            if 0x1010 in bv._suppressed:
                raise RuntimeError("post-apply exclusive scope unavailable")
            yield

    monkeypatch.setattr(instance._target_lock, "write", post_apply_failure)
    result = instance._mutation(None, False, [
        {"op": "function_delete", "address": "0x1010"},
    ])
    assert result["committed"] is True and result["rolled_back"] is False
    row = result["results"][0]
    assert row["irreversible_change"] is True
    assert row["observed"]["error_stage"] == "scope_enter"
    assert row["observed"]["error"] == "post-apply exclusive scope unavailable"
    assert bv.get_function_at(0x1010) is None
    assert "post-apply-delete" in targets._dirty_view_ids
    summary = _mutation_summary(result)
    assert summary["dirty_after"] is True and summary["changed_count"] == 1


def test_delete_requires_exact_entry_not_containing_body(monkeypatch):
    bridge, instance, bv, _, _ = _setup(monkeypatch)
    with pytest.raises(bridge.OperationFailure) as failure:
        instance._mutation(None, False, [{"op": "function_delete", "address": "0x1004"}])
    assert failure.value.status == "invalid_request"
    assert bv.events == []


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit, BaseException])
@pytest.mark.parametrize("phase", ["remove", "analysis", "readback"])
def test_delete_interruption_propagates_with_real_dirty_ledger(monkeypatch, exception_type, phase):
    _, instance, bv, _, _ = _setup(monkeypatch)
    targets = instance.ctx.targets
    targets._ids_by_object[bv] = "interrupted-delete"
    original_remove = bv.remove_user_function
    interruption = exception_type("injected deletion interruption")

    def fail(*args, **kwargs):
        raise interruption

    def remove(fn):
        assert "interrupted-delete" in targets._dirty_view_ids
        original_remove(fn)
        if phase == "remove":
            fail()

    monkeypatch.setattr(bv, "remove_user_function", remove)
    if phase == "analysis":
        monkeypatch.setattr(bv, "update_analysis_and_wait", fail)
    if phase == "readback":
        monkeypatch.setattr(bv, "get_functions_containing", fail)
    with pytest.raises(exception_type) as caught:
        instance._mutation(None, False, [{"op": "function_delete", "address": "0x1010"}])
    assert caught.value is interruption
    assert bv.get_function_at(0x1010) is None
    assert "interrupted-delete" in targets._dirty_view_ids
