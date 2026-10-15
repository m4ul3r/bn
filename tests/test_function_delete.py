from __future__ import annotations

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
    assert observed["function_removed"] is (failure_mode == "analysis")
    assert observed["function_at_entry"] == (None if failure_mode == "analysis" else "false_tail")
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


def test_single_delete_batch_uses_observed_mutation_result(monkeypatch):
    _, instance, bv, _, _ = _setup(monkeypatch)
    result = instance._mutation(None, False, [{"op": "function_delete", "address": "0x1010"}])
    assert result["success"] is True
    assert result["results"][0]["observed"]["bodies_containing_deleted_entry"] == []
    assert bv.get_function_at(0x1010) is None


def test_delete_requires_exact_entry_not_containing_body(monkeypatch):
    bridge, instance, bv, _, _ = _setup(monkeypatch)
    with pytest.raises(bridge.OperationFailure) as failure:
        instance._mutation(None, False, [{"op": "function_delete", "address": "0x1004"}])
    assert failure.value.status == "invalid_request"
    assert bv.events == []
