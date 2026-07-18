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


def test_find_function_stays_ambiguous_for_two_real_bodies_with_kinds(monkeypatch):
    """Two genuine same-named bodies (the A/B-duplicate firmware case) stay
    ambiguous -- auto-pick must NOT guess -- and the error now names each
    candidate's symbol kind so the collision self-documents (#122)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    a = _FakeFunction(0x401000, "dup")
    a.symbol = _FakeSymbol("FunctionSymbol")
    b = _FakeFunction(0x402000, "dup")
    b.symbol = _FakeSymbol("FunctionSymbol")
    bv = _FakeBV(functions=[a, b])

    with pytest.raises(RuntimeError, match="Ambiguous function identifier") as excinfo:
        instance._find_function(bv, "dup")
    assert "[FunctionSymbol]" in str(excinfo.value)


def test_verify_rename_symbol_reports_noop(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x401000, "player_update")])

    result = instance._verify_operation(
        bv,
        {
            "op": "rename_symbol",
            "kind": "function",
            "address": "0x401000",
            "before_name": "player_update",
            "new_name": "player_update",
            "requested": {
                "op": "rename_symbol",
                "identifier": "player_update",
                "new_name": "player_update",
            },
        },
    )

    assert result["status"] == "noop"
    assert result["observed"]["name"] == "player_update"


@pytest.mark.parametrize("bad_name", ["", "   ", "\t", None])
def test_op_rename_symbol_rejects_empty_new_name(monkeypatch, bad_name):
    """The bridge rejects an empty/whitespace-only/null new name as
    invalid_request, so a batch apply or raw-socket rename_symbol op cannot
    create a degenerate unnamed function (#363) -- the CLI guard is not the only
    line of defense. JSON ``null`` must NOT stringify to the literal "None" and
    slip through. Rejection happens before target resolution, so no view state
    is touched."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x401000, "mput")])
    op = {"op": "rename_symbol", "identifier": "mput", "new_name": bad_name}

    with pytest.raises(bridge.mutation_engine.OperationFailure) as excinfo:
        bridge.mutation_engine._op_rename_symbol(instance.ctx, bv, op)
    assert excinfo.value.status == "invalid_request"
    assert "non-empty" in excinfo.value.message


def test_mutation_refused_on_quick_view_before_any_analysis(monkeypatch):
    """#479 (sibling): every write op except function_create routes through
    _mutation, whose success/revert paths call update_analysis_and_wait(). On a
    --quick view that wedges the instance under the write lock. The whole batch --
    including --preview -- must be refused fast (bn type/comment/rename/proto/...),
    before touching the view."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    bridge._quick_loaded_views.add(bv)

    for preview in (False, True):
        with pytest.raises(bridge.OperationFailure) as exc:
            instance._mutation("active", preview, [{"op": "set_comment", "comment": "x"}])
        assert exc.value.status == "invalid_request"
        assert "--quick" in str(exc.value)
    # Refused before any undo bracket / analysis was started -> no wedge.
    assert "refresh" not in bv.events
    assert not _has_event(bv, "begin")
    bridge._quick_loaded_views.discard(bv)


def test_mutation_reverts_on_verification_failure(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    # _mutation moved to mutation_engine and calls these peers module-locally;
    # patch the seam helper on instance.ctx and the mutation peers on the module.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.mutation_engine, "_guess_affected_functions", lambda ctx, bv, operations: [])
    monkeypatch.setattr(bridge.mutation_engine, "_capture_function_snapshots", lambda ctx, bv, functions: {})
    monkeypatch.setattr(bridge.mutation_engine, "_capture_type_snapshots", lambda ctx, bv, operations: {})
    monkeypatch.setattr(bridge.mutation_engine, "_diff_snapshots", lambda ctx, before, after: [])
    monkeypatch.setattr(bridge.mutation_engine, "_diff_type_snapshots", lambda ctx, before, after: [])
    monkeypatch.setattr(
        bridge.mutation_engine,
        "_apply_operation",
        lambda ctx, bv, op, restores=None: {
            "op": "rename_symbol",
            "kind": "function",
            "address": "0x401000",
            "new_name": "player_update",
            "requested": {"identifier": "sub_401000", "new_name": "player_update"},
        },
    )
    monkeypatch.setattr(
        bridge.mutation_engine,
        "_verify_operation",
        lambda ctx, bv, result: {
            **result,
            "status": "verification_failed",
            "message": "Live rename verification failed at 0x401000",
        },
    )

    result = instance._mutation("active", False, [{"op": "rename_symbol"}])

    assert result["success"] is False
    assert result["committed"] is False
    assert _has_event(bv, "revert")
    assert not _has_event(bv, "commit")


def test_run_local_restores_runs_reverse_and_reports_failure(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    order: list[int] = []

    def mk(n, *, fail=False):
        def _restore():
            order.append(n)
            if fail:
                raise RuntimeError("boom")
        return _restore

    settled = []
    bv = types.SimpleNamespace(update_analysis_and_wait=lambda: settled.append(True))

    # A failing restore must not stop the others, and the result is False.
    ok = instance._run_local_restores(bv, [mk(1), mk(2, fail=True), mk(3)])
    assert order == [3, 2, 1]  # reverse of apply order
    assert ok is False
    assert settled == [True]  # view re-settled so the restore materializes

    order.clear()
    settled.clear()
    assert instance._run_local_restores(bv, [mk(1), mk(2)]) is True
    assert order == [2, 1]
    assert settled == [True]

    # Empty restore list is a no-op: no reanalysis triggered.
    settled.clear()
    assert instance._run_local_restores(bv, []) is True
    assert settled == []


def test_apply_failure_runs_restores_even_when_undo_revert_fails(monkeypatch):
    """An apply failure must run the explicit non-journaled restores even when
    the undo revert fails — `and` short-circuit would skip them and leave
    local/prototype changes applied (#88)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    calls = {"restores": 0}

    def apply(bv_, op, restores=None):
        if op.get("op") == "boom":
            raise bridge.OperationFailure("unsupported", "nope", requested={})
        restores.append(lambda: None)
        return {"op": "local_rename", "requested": {}}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply)
    monkeypatch.setattr(bridge.mutation_engine, "_revert_undo_safely", lambda ctx, bv_, state: False)

    def run_restores(ctx, bv_, restores):
        calls["restores"] += 1
        assert len(restores) == 1
        return True

    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", run_restores)

    result = instance._mutation("active", False, [{"op": "local_rename"}, {"op": "boom"}])

    assert calls["restores"] == 1  # restores ran despite the failed undo revert
    assert result["success"] is False
    assert result["rolled_back"] is False  # undo revert failed, so not fully reverted
    assert "rollback itself failed" in result["message"]


def test_preview_restore_failure_is_not_success(monkeypatch):
    """A preview whose non-journaled restore failed left the view modified;
    it must not report success:true / exit 0 to automation (#88)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None):
        restores.append(lambda: None)
        return {"op": "local_rename", "requested": {}}

    _mutation_with_stubs(
        monkeypatch, bridge, instance, bv,
        apply=apply,
        verify=lambda bv_, result: {**result, "status": "verified"},
    )
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, restores: False)

    result = instance._mutation("active", True, [{"op": "local_rename"}])

    assert result["preview"] is True
    assert result["success"] is False
    assert result["committed"] is False
    assert result["rolled_back"] is False
    assert "failed" in result["message"]
    assert _has_event(bv, "revert")
    assert not _has_event(bv, "commit")


def test_preview_drift_restore_failure_is_not_success(monkeypatch):
    """If reverting BN's propagation onto aliased siblings (the var-drift
    restore) fails, the preview left the view modified and must report
    success:false / rolled_back:false (#88)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None):
        return {"op": "local_rename", "requested": {}}

    _mutation_with_stubs(
        monkeypatch, bridge, instance, bv,
        apply=apply,
        verify=lambda bv_, result: {**result, "status": "verified"},
    )
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, restores: True)
    # Force a non-empty var snapshot and a failing drift restore.
    monkeypatch.setattr(bridge.mutation_engine, "_capture_local_var_snapshots", lambda ctx, bv_, fns: {0x1: {1: ("a", "int")}})
    monkeypatch.setattr(bridge.mutation_engine, "_restore_local_var_drift", lambda ctx, bv_, snap: False)

    result = instance._mutation("active", True, [{"op": "local_rename"}])

    assert result["preview"] is True
    assert result["success"] is False
    assert result["rolled_back"] is False


def test_preview_with_successful_restore_still_succeeds(monkeypatch):
    """The restored-success coupling must not regress the normal preview path."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None):
        restores.append(lambda: None)
        return {"op": "local_rename", "requested": {}}

    _mutation_with_stubs(
        monkeypatch, bridge, instance, bv,
        apply=apply,
        verify=lambda bv_, result: {**result, "status": "verified"},
    )
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, restores: True)

    result = instance._mutation("active", True, [{"op": "local_rename"}])

    assert result["preview"] is True
    assert result["success"] is True
    assert result["committed"] is False
    assert result["rolled_back"] is True


def test_rolled_back_sibling_op_reports_reverted_not_unsupported(monkeypatch):
    """When a later op fails mid-batch and the batch is reverted, an op that
    ALREADY succeeded must be reported as 'reverted', not 'unsupported' -- it
    was supported and applied; a sibling failed. 'reverted' is not a failure
    status (#118)."""
    from bn.formatters import FAILED_MUTATION_STATUSES

    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None):
        if op.get("op") == "boom":
            raise bridge.OperationFailure("unsupported", "Function not found: x", requested={})
        return {"op": "rename_symbol", "status": "applied", "requested": {}}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply)
    monkeypatch.setattr(bridge.mutation_engine, "_revert_undo_safely", lambda ctx, bv_, state: True)
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, restores: True)

    result = instance._mutation("active", False, [{"op": "rename_symbol"}, {"op": "boom"}])

    assert result["success"] is False
    assert result["rolled_back"] is True
    statuses = [r["status"] for r in result["results"]]
    assert statuses[0] == "reverted"          # succeeded-then-reverted, honestly
    assert statuses[1] == "unsupported"        # the real failing op keeps its status
    assert "reverted" not in FAILED_MUTATION_STATUSES


def test_rolled_back_sibling_reports_rollback_failed_when_revert_fails(monkeypatch):
    """If the rollback itself fails, a preceding op may STILL be applied -- it
    must not be labeled 'reverted'. Use a distinct failed status so exit codes
    and rendering treat the left-modified view as the failure it is (#118)."""
    from bn.formatters import FAILED_MUTATION_STATUSES

    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None):
        if op.get("op") == "boom":
            raise bridge.OperationFailure("unsupported", "nope", requested={})
        return {"op": "rename_symbol", "status": "applied", "requested": {}}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply)
    monkeypatch.setattr(bridge.mutation_engine, "_revert_undo_safely", lambda ctx, bv_, state: False)
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, restores: True)

    result = instance._mutation("active", False, [{"op": "rename_symbol"}, {"op": "boom"}])

    assert result["success"] is False
    assert result["rolled_back"] is False
    assert result["results"][0]["status"] == "rollback_failed"
    assert "rollback_failed" in FAILED_MUTATION_STATUSES


def test_restore_local_var_drift_unpins_propagated_auto_sibling(monkeypatch):
    """BN's create_user_var propagates a USER override onto aliased siblings
    (naming a stack var also renames the aliased register), and that propagation
    is NOT journaled -- it survives the undo. The drift mop-up must DROP that
    override (delete_user_var) so the AUTO sibling comes back AUTO, NOT re-pin it
    with create_user_var (which restores the displayed name but leaves the var
    USER while claiming a clean revert -- #630)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    target = _FakeVariable(name="var_8", storage=-8, var_type="int32_t", identifier=10)
    sibling = _FakeVariable(name="r2", storage=2, var_type="int32_t", identifier=20,
                            source_type="RegisterVariableSourceType")
    fn = _FakeFunction(0x11744, "f")
    fn.stack_layout = [target]
    fn.hlil = types.SimpleNamespace(vars=[sibling])
    bv = _FakeMutationBV(functions=[fn])
    fn.view = bv

    # Snapshot BEFORE apply: both locals are AUTO.
    before = instance._capture_local_var_snapshots(bv, [fn])
    assert before[0x11744][10] == ("var_8", "int32_t", False)
    assert before[0x11744][20] == ("r2", "int32_t", False)

    # The targeted var was already un-pinned by the per-op restore; the aliased
    # sibling was left USER-pinned + renamed by BN's non-journaled propagation.
    fn.create_user_var(sibling, "int32_t", "Q8_1")
    assert fn.is_var_user_defined(sibling) is True

    ok = instance._restore_local_var_drift(bv, before)
    assert ok is True
    # Provenance restored to AUTO -- NOT re-pinned USER (the #630 bug).
    assert fn.is_var_user_defined(sibling) is False
    assert sibling.name == "r2"


def test_restore_local_var_drift_replays_genuine_user_sibling(monkeypatch):
    """Negative control: a sibling that ALREADY had a USER definition must be
    RE-ASSERTED with create_user_var if it drifted -- restoring a genuine user
    override is correct, not residue (#630)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    var = _FakeVariable(name="var_8", storage=-8, var_type="int32_t", identifier=10)
    fn = _FakeFunction(0x11744, "f")
    fn.stack_layout = [var]
    bv = _FakeMutationBV(functions=[fn])
    fn.view = bv

    fn.create_user_var(var, "int32_t", "kept")  # genuine prior USER definition
    before = instance._capture_local_var_snapshots(bv, [fn])
    assert before[0x11744][10] == ("kept", "int32_t", True)

    var.name = "drifted"  # a later op perturbed the displayed name
    ok = instance._restore_local_var_drift(bv, before)
    assert ok is True
    assert var.name == "kept"
    assert fn.is_var_user_defined(var) is True  # still a user var


def test_restore_local_var_drift_leaves_reanalyzed_auto_name_alone(monkeypatch):
    """An AUTO local that is STILL auto but whose name BN re-derived differently
    is phantom drift (#581): the mop-up must not touch it (no create_user_var,
    no delete_user_var) -- doing so would pin an AUTO var USER (#630)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    var = _FakeVariable(name="var_8", storage=-8, var_type="int32_t", identifier=10)
    fn = _FakeFunction(0x11744, "f")
    fn.stack_layout = [var]
    bv = _FakeMutationBV(functions=[fn])
    fn.view = bv

    before = instance._capture_local_var_snapshots(bv, [fn])
    var.name = "var_10"  # BN re-derived a different AUTO name; still AUTO
    assert fn.is_var_user_defined(var) is False

    ok = instance._restore_local_var_drift(bv, before)
    assert ok is True
    assert fn.is_var_user_defined(var) is False  # untouched, still AUTO
    assert "refresh" not in bv.events  # nothing pinned/dropped -> no reanalysis


def test_restore_local_var_drift_noop_when_nothing_changed(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    var = _FakeVariable(name="keep", storage=-8, var_type="int32_t", identifier=10)
    fn = _FakeFunction(0x11744, "f")
    fn.stack_layout = [var]
    settled: list[bool] = []
    bv = _FakeBV(functions=[fn])
    bv.update_analysis_and_wait = lambda: settled.append(True)

    before = instance._capture_local_var_snapshots(bv, [fn])
    assert instance._restore_local_var_drift(bv, before) is True
    assert settled == []  # nothing drifted -> no reanalysis


def test_restore_local_var_drift_reports_failure_on_missing_function(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # Snapshot references a function the view can no longer resolve.
    snapshots = {0xdead: {10: ("v", "int32_t")}}
    bv = _FakeBV(functions=[])
    assert instance._restore_local_var_drift(bv, snapshots) is False


def test_op_local_rename_registers_restore_that_undoes_the_rename(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    var = _FakeVariable(
        name="r2_1", storage=2, var_type="uint32_t", identifier=123,
        source_type="RegisterVariableSourceType",
    )

    class _RecordingFunc:
        def __init__(self):
            self.start = 0x1000
            self.name = "f"
            self.calls: list[tuple] = []

        def create_user_var(self, v, type_obj, name):
            self.calls.append((name, str(type_obj)))
            v.name = name
            v.type = type_obj

    fn = _RecordingFunc()
    bv = types.SimpleNamespace(get_function_at=lambda addr: fn)
    # _op_local_rename moved to mutation_engine; it resolves _find_function via
    # the seam (instance.ctx), the variable helpers via vars_mod, and
    # _find_var_for_restore module-locally (now ctx-first).
    monkeypatch.setattr(instance.ctx, "_find_function", lambda _bv, ident: fn)
    monkeypatch.setattr(bridge.vars_mod, "_find_variable_selector", lambda _f, sel: (var, False))
    monkeypatch.setattr(bridge.mutation_engine, "_find_var_for_restore", lambda ctx, _f, identifier, storage, is_parameter: var)
    monkeypatch.setattr(bridge.vars_mod, "_local_id", lambda _f, _v, is_parameter: "lid")

    restores: list = []
    result = instance._op_local_rename(
        bv, {"op": "local_rename", "function": "f", "variable": "r2_1", "new_name": "tbl_count"}, restores
    )

    # before_name is the OLD name, the rename applied, and a restore was registered.
    assert result["before_name"] == "r2_1"
    assert result["new_name"] == "tbl_count"
    assert var.name == "tbl_count"
    assert len(restores) == 1

    # Replaying the restore puts the local back to its original name+type.
    restores[0]()
    assert var.name == "r2_1"
    assert str(var.type) == "uint32_t"
    assert fn.calls[-1] == ("r2_1", "uint32_t")


def test_op_local_rename_noop_registers_no_restore(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    var = _FakeVariable(name="keep", storage=2, var_type="int32_t", identifier=1)
    fn = types.SimpleNamespace(start=0x1000, name="f", create_user_var=lambda *a: None)
    monkeypatch.setattr(instance.ctx, "_find_function", lambda _bv, ident: fn)
    monkeypatch.setattr(bridge.vars_mod, "_find_variable_selector", lambda _f, sel: (var, False))
    monkeypatch.setattr(bridge.vars_mod, "_local_id", lambda _f, _v, is_parameter: "lid")

    restores: list = []
    instance._op_local_rename(bv := object(), {"op": "local_rename", "function": "f", "variable": "keep", "new_name": "keep"}, restores)
    assert restores == []  # renaming to the same name mutates nothing, so nothing to revert


def test_op_set_prototype_uses_string_user_type_for_bn_compat(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _SetterFunction(_FakeFunction):
        def __init__(self):
            super().__init__(0x43F200, "update_garbage_hazard", "void* __fastcall(void* arg1)")
            self.user_type_calls = []

        def set_user_type(self, value):
            self.user_type_calls.append(value)
            if isinstance(value, str):
                super().set_user_type(value)

    class _PrototypeBV(_FakeBV):
        def parse_type_string(self, declaration):
            return _FakeType("void* __thiscall(struct GarbageHazardRuntime* self)", type_class="FunctionTypeClass"), None

    fn = _SetterFunction()
    bv = _PrototypeBV(functions=[fn])

    result = instance._op_set_prototype(
        bv,
        {
            "op": "set_prototype",
            "identifier": "update_garbage_hazard",
            "prototype": "void* __thiscall update_garbage_hazard(struct GarbageHazardRuntime* self)",
        },
    )

    assert fn.user_type_calls == ["void* __thiscall(struct GarbageHazardRuntime* self)"]
    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert verified["observed"]["prototype"] == "void* __thiscall(struct GarbageHazardRuntime* self)"


def test_apply_operation_user_error_message_has_no_class_name(monkeypatch):
    """A handler raising a user-facing RuntimeError (e.g. a mistyped function
    name -> 'Function not found') must surface a clean, actionable message --
    not 'unsupported: RuntimeError: ...' that reads like an internal crash
    (#122)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    me = bridge.mutation_engine

    def boom_user(ctx, bv, op):
        raise RuntimeError("Function not found: ghost")

    monkeypatch.setattr(me, "_op_set_comment", boom_user)
    bv = _FakeBV()

    with pytest.raises(bridge.OperationFailure) as excinfo:
        instance._apply_operation(bv, {"op": "set_comment", "comment": "x", "function": "ghost"})

    assert excinfo.value.status == "unsupported"
    assert excinfo.value.message == "Function not found: ghost"
    assert "RuntimeError" not in excinfo.value.message


def test_apply_operation_unexpected_error_gets_internal_error_status(monkeypatch):
    """A genuinely UNEXPECTED internal error gets the distinct 'internal_error'
    status (kept in FAILED_MUTATION_STATUSES so exit codes still flag it) and
    keeps the class name for debugging -- not the misleading 'unsupported'
    (#122)."""
    from bn.formatters import FAILED_MUTATION_STATUSES

    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    me = bridge.mutation_engine

    def boom_internal(ctx, bv, op):
        raise KeyError("unexpected")

    monkeypatch.setattr(me, "_op_set_comment", boom_internal)
    bv = _FakeBV()

    with pytest.raises(bridge.OperationFailure) as excinfo:
        instance._apply_operation(bv, {"op": "set_comment", "comment": "x", "function": "g"})

    assert excinfo.value.status == "internal_error"
    assert "KeyError" in excinfo.value.message
    assert "internal_error" in FAILED_MUTATION_STATUSES


def test_op_set_prototype_hints_to_declare_unknown_struct(monkeypatch):
    """A prototype that references a not-yet-defined struct makes
    parse_type_string fail. Surface a clear invalid_request that hints to
    declare the type first -- not a raw 'unsupported: SyntaxError: ...' that
    leaks the Python exception class and gives no next step (#122)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _UnknownTypeBV(_FakeBV):
        def parse_type_string(self, declaration):
            raise SyntaxError("unexpected token 'GarbageHazardRuntime'")

    fn = _FakeFunction(0x401000, "handler")
    bv = _UnknownTypeBV(functions=[fn])

    with pytest.raises(bridge.OperationFailure) as excinfo:
        instance._op_set_prototype(
            bv,
            {
                "op": "set_prototype",
                "identifier": "handler",
                "prototype": "int handler(struct GarbageHazardRuntime* self)",
            },
        )

    assert excinfo.value.status == "invalid_request"
    message = excinfo.value.message.lower()
    assert "declare" in message            # actionable next step
    assert "syntaxerror" not in message    # raw exception class must not leak


def test_batch_struct_field_accepts_type_name_alias(monkeypatch):
    """A struct_field_* batch op may use `type_name` (the key the output /
    affected_types surface uses, and an analyst's natural reflex) as an alias
    for the canonical `struct_name`, instead of failing validation with
    'missing required field struct_name' (M12)."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine

    # the alias is normalized in place for every struct_field_* kind
    for kind in ("struct_field_set", "struct_field_rename", "struct_field_delete"):
        op = {"op": kind, "type_name": "Elf64_Sym"}
        me._normalize_struct_alias(op)
        assert op["struct_name"] == "Elf64_Sym", kind

    # an explicit struct_name always wins (alias never clobbers it)
    op = {"op": "struct_field_rename", "struct_name": "A", "type_name": "B"}
    me._normalize_struct_alias(op)
    assert op["struct_name"] == "A"

    # non-struct ops are left untouched
    op = {"op": "rename_symbol", "type_name": "X"}
    me._normalize_struct_alias(op)
    assert "struct_name" not in op

    # end-to-end through _apply_operation: validation no longer rejects a
    # type_name-only struct op, and the handler receives struct_name
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[])
    seen = {}

    def _stub_rename(ctx, bv_, op_):
        seen["op"] = op_
        return {"status": "verified"}

    monkeypatch.setattr(me, "_op_struct_field_rename", _stub_rename)
    result = me._apply_operation(
        instance.ctx, bv,
        {"op": "struct_field_rename", "type_name": "Elf64_Sym",
         "old_name": "st_info", "new_name": "sym_info"},
    )
    assert result == {"status": "verified"}
    assert seen["op"]["struct_name"] == "Elf64_Sym"


def test_batch_op_parity_table_locked_to_validators(monkeypatch):
    """The checked-in parity table must match the live validators exactly, and
    cover every dispatched op kind -- so the audit can't rot as ops are added or
    their field sets change (#173)."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine

    for kind, row in _BATCH_OP_PARITY.items():
        assert me.REQUIRED_FIELDS.get(kind, ()) == row["required"], kind
        assert me.REQUIRED_ONE_OF.get(kind, ()) == row["one_of"], kind
        assert me.ENUM_FIELDS.get(kind, {}) == row["enum"], kind

    # No validator names an op the table forgot, and vice versa.
    assert set(me.REQUIRED_FIELDS) == set(_BATCH_OP_PARITY)
    assert set(me.REQUIRED_ONE_OF) <= set(_BATCH_OP_PARITY)
    assert set(me.ENUM_FIELDS) <= set(_BATCH_OP_PARITY)


def test_batch_missing_required_field_rejected_per_op(monkeypatch):
    """Dropping any single required field from any of the 10 ops yields a clean
    invalid_request that NAMES the field -- never a raw KeyError mislabeled
    internal_error/unsupported from a handler hard-reading op[field] (#173)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()

    for kind, row in _BATCH_OP_PARITY.items():
        for field in row["required"]:
            op = _minimal_valid_op(kind)
            del op[field]
            with pytest.raises(bridge.OperationFailure) as exc:
                instance._apply_operation(bv, op)
            assert exc.value.status == "invalid_request", (kind, field)
            assert field in exc.value.message, (kind, field)


def test_batch_rename_rejects_invalid_kind(monkeypatch):
    """An out-of-set rename `kind` must be rejected the way interactive
    `bn rename --kind` rejects it via argparse choices. The batch path has no
    argparse layer, and the unguarded handler SILENTLY treated an unknown kind
    as a (failing) data-symbol lookup -- so `kind: "garbage"` against a function
    that plainly exists produced a misleading "Symbol not found" instead of a
    clear "kind must be one of ..." (#173)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x1000, "foo")])

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._apply_operation(
            bv, {"op": "rename_symbol", "identifier": "foo", "new_name": "bar", "kind": "garbage"}
        )
    assert exc.value.status == "invalid_request"
    msg = exc.value.message
    assert "kind" in msg and "garbage" in msg
    assert "auto" in msg and "function" in msg and "data" in msg

    # The guard must NOT over-reject a valid kind: kind="function" still resolves
    # (here a noop, since new_name == current name) without raising.
    result = instance._apply_operation(
        bv, {"op": "rename_symbol", "identifier": "foo", "new_name": "foo", "kind": "function"}
    )
    assert result["op"] == "rename_symbol"


def test_batch_invalid_op_rolls_back_prior_applied_op(monkeypatch):
    """A manifest whose 2nd op is malformed (missing a required field) fails the
    WHOLE batch: the 1st op's already-applied change is reverted (no partial
    apply -- the undo state is reverted, never committed) and the failing op is
    reported as a clean invalid_request (#173)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeCommentMutationBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._mutation(
        "active",
        False,
        [
            {"op": "set_comment", "address": "0x1000", "comment": "first op applied"},
            {"op": "set_comment", "address": "0x2000"},  # missing required 'comment'
        ],
    )

    assert result["success"] is False
    assert result["committed"] is False
    assert result["rolled_back"] is True
    assert _has_event(bv, "revert")
    assert not _has_event(bv, "commit")

    statuses = [r.get("status") for r in result["results"]]
    assert statuses[0] == "reverted"          # 1st op applied, then rolled back
    assert statuses[1] == "invalid_request"   # 2nd op rejected before apply
    assert "comment" in result["results"][1].get("message", "")


def test_preview_diff_truncated_to_stay_inline(monkeypatch):
    """A previewed mutation's per-function `diff` is capped so a single rename /
    proto preview on a large function stays inline instead of tripping the 10k
    spill threshold; the full diff stays available via --out. (M14)"""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine

    # a short diff passes through untouched
    short = "\n".join(f"line {i}" for i in range(10))
    assert me._truncate_preview_diff(short) == short

    # a long diff is capped to max_lines + a marker pointing at --out
    long = "\n".join(f"line {i}" for i in range(me.PREVIEW_DIFF_MAX_LINES + 500))
    out = me._truncate_preview_diff(long)
    body, _, marker = out.rpartition("\n")
    assert len(body.splitlines()) == me.PREVIEW_DIFF_MAX_LINES
    assert "diff truncated" in marker and "500 more" in marker and "--out" in marker

    # integration: a whole-body change yields a bounded diff, changed=True, and
    # the focused snippet excerpt is still present for the glance
    ctx = bridge.BinaryNinjaBridge().ctx
    big_before = "\n".join(f"old {i}" for i in range(2000))
    big_after = "\n".join(f"new {i}" for i in range(2000))
    diffs = me._diff_snapshots(
        ctx,
        {0x1000: {"text": big_before, "name": "f"}},
        {0x1000: {"text": big_after, "name": "f"}},
    )
    d = diffs[0]
    assert d["changed"] is True
    assert len(d["diff"].splitlines()) <= me.PREVIEW_DIFF_MAX_LINES + 1
    assert "before_excerpt" in d


def test_op_set_prototype_registers_restore_for_preview(monkeypatch):
    # set_user_type is NOT journaled by BN's undo buffer, so --preview/rollback
    # must register an explicit restore that puts the prototype back, else the
    # previewed prototype silently persists in the view (#51). The restore uses
    # the .type property setter (a clean revert, no convention re-pinning).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _SetterFunction(_FakeFunction):
        def __init__(self):
            super().__init__(0x1000, "f", "int32_t(int32_t* arg1)")

        def set_user_type(self, value):
            super().set_user_type(value if isinstance(value, str) else str(value))

    class _PrototypeBV(_FakeBV):
        def parse_type_string(self, declaration):
            return _FakeType("void(uint32_t* p)", type_class="FunctionTypeClass"), None

    fn = _SetterFunction()
    bv = _PrototypeBV(functions=[fn])
    baseline = fn.type
    restores: list = []
    instance._op_set_prototype(
        bv, {"op": "set_prototype", "identifier": "f", "prototype": "void f(uint32_t* p)"}, restores
    )
    # the prototype was applied ...
    assert fn.type == "void(uint32_t* p)"
    # ... and exactly one restore was registered for the preview/rollback path
    assert len(restores) == 1
    # running it (as the preview path does) puts the original prototype back
    restores[0]()
    assert fn.type == baseline


def test_op_set_prototype_no_restore_when_unchanged(monkeypatch):
    # Setting the same prototype mutates nothing, so no restore is queued (the
    # revert path stays a true no-op).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _SetterFunction(_FakeFunction):
        def __init__(self):
            super().__init__(0x1000, "f", "int32_t(int32_t* arg1)")

        def set_user_type(self, value):
            super().set_user_type(value if isinstance(value, str) else str(value))

    class _PrototypeBV(_FakeBV):
        def parse_type_string(self, declaration):
            return _FakeType("int32_t(int32_t* arg1)", type_class="FunctionTypeClass"), None

    fn = _SetterFunction()
    bv = _PrototypeBV(functions=[fn])
    restores: list = []
    instance._op_set_prototype(
        bv, {"op": "set_prototype", "identifier": "f", "prototype": "int32_t f(int32_t* arg1)"}, restores
    )
    assert restores == []


def test_find_function_suggests_close_match_when_not_found(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "player_update"),
            _FakeFunction(0x402000, "player_render"),
        ]
    )

    with pytest.raises(RuntimeError) as exc_info:
        instance._find_function(bv, "player_updaet")

    message = str(exc_info.value)
    assert message.startswith("Function not found: player_updaet")
    assert "Did you mean: player_update" in message


def test_find_function_not_found_without_close_match(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x401000, "player_update")])

    with pytest.raises(RuntimeError) as exc_info:
        instance._find_function(bv, "zzzzzzzz")

    assert str(exc_info.value) == "Function not found: zzzzzzzz"


def test_parse_address_reports_friendly_error_for_garbage(monkeypatch):
    bridge = _load_bridge(monkeypatch)

    with pytest.raises(ValueError) as exc_info:
        bridge._parse_address("not_an_address")

    message = str(exc_info.value)
    assert "not a valid address" in message
    # The raw int() ValueError must not leak through.
    assert "invalid literal for int" not in message


def test_find_function_invalid_hex_reports_address_not_missing_function(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x401000, "player_update")])

    with pytest.raises(RuntimeError) as exc_info:
        instance._find_function(bv, "0xGGGG")

    message = str(exc_info.value)
    assert "Invalid address" in message
    assert "0xGGGG" in message
    assert "Function not found" not in message


def test_find_function_valid_address_with_no_function(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x401000, "player_update")])

    with pytest.raises(RuntimeError) as exc_info:
        instance._find_function(bv, "0x999999")

    assert str(exc_info.value) == "No function found at address 0x999999"


def test_bridge_ops_reject_out_of_range_count_params(monkeypatch):
    # Non-CLI callers (py exec / raw socket) reach the op handlers directly, so
    # the bridge must re-enforce the count/offset contract the CLI argparse
    # layer applies -- a negative/zero limit must not silently drop the tail or
    # return a degenerate empty-but-"truncated" result (#28).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # validation happens before _resolve_view, so no fake view is needed
    with pytest.raises(bridge.OperationFailure) as e:
        instance._message_lens("active", "x", limit=0)
    assert e.value.status == "invalid_request"
    with pytest.raises(bridge.OperationFailure):
        instance._list_functions("active", limit=-1)
    with pytest.raises(bridge.OperationFailure):
        instance._search_functions("active", "q", limit=-5)
    with pytest.raises(bridge.OperationFailure):
        instance._types("active", query=None, offset=0, limit=0)
    with pytest.raises(bridge.OperationFailure):
        instance._strings("active", query=None, offset=-1, limit=10)


def test_diff_snapshots_marks_name_only_changes(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    diffs = instance._diff_snapshots(
        {
            0x401000: {
                "name": "sub_401000",
                "address": "0x401000",
                "text": "return 7;",
            }
        },
        {
            0x401000: {
                "name": "player_update",
                "address": "0x401000",
                "text": "return 7;",
            }
        },
    )

    assert len(diffs) == 1
    assert diffs[0]["changed"] is True
    assert diffs[0]["before_name"] == "sub_401000"
    assert diffs[0]["after_name"] == "player_update"
    assert diffs[0]["diff"] == "--- before:sub_401000\n+++ after:player_update"
    assert "before_excerpt" not in diffs[0]


def test_diff_snapshots_marks_local_only_change(monkeypatch):
    """A local rename/retype of a variable not rendered in the HLIL body leaves
    the body text identical; the diff/changed signal must reflect local
    name/type state too (#121)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    diffs = instance._diff_snapshots(
        {0x401000: {"name": "f", "address": "0x401000", "text": "return arg1;",
                    "comments": {}, "locals": {"1": "arg1:int32_t"}}},
        {0x401000: {"name": "f", "address": "0x401000", "text": "return arg1;",
                    "comments": {}, "locals": {"1": "session_id:int32_t"}}},
    )

    assert len(diffs) == 1
    assert diffs[0]["changed"] is True
    assert diffs[0]["diff"]
    assert "session_id" in diffs[0]["diff"]


# ---------------------------------------------------------------------------
# Verification: local rename with SSA-style variable reconstruction
# ---------------------------------------------------------------------------


def test_verify_local_rename_passes_when_auto_name_persists_but_user_name_on_alt_var(monkeypatch):
    """After analysis BN may reconstruct variable objects at the same storage
    offset.  If the primary variable still reports its auto name but a second
    variable at the same offset carries the user-assigned name, verification
    should succeed.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    # Two variables at the same storage offset — simulates post-analysis state
    # where BN keeps both the auto-named and user-named entries.
    auto_var = _FakeVariable(name="var_48", storage=-72, var_type="int32_t", identifier=3001)
    user_var = _FakeVariable(name="wIndex", storage=-72, var_type="int32_t", identifier=3001)

    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [auto_var, user_var]

    bv = _FakeBV(functions=[fn])

    # Build a result dict as _op_local_rename would produce.
    result = {
        "op": "local_rename",
        "function": "process_usb",
        "address": "0x401000",
        "variable": "var_48",
        "local_id": "0x401000:local:stack:-72:0:3001",
        "storage": -72,
        "identifier": 3001,
        "source_type": "StackVariableSourceType",
        "is_parameter": False,
        "before_name": "var_48",
        "new_name": "wIndex",
        "requested": {"variable": "var_48", "new_name": "wIndex"},
    }

    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert verified["observed"]["variable"] == "wIndex"


def test_verify_local_rename_uses_identifier_lookup(monkeypatch):
    """Verification should prefer identifier-based lookup over raw storage
    matching so it finds the correct variable after analysis rebuilds the
    stack layout."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    # Variable at same storage but different identifier — should NOT be matched.
    other_var = _FakeVariable(name="var_48", storage=-72, var_type="int32_t", identifier=9999)
    renamed_var = _FakeVariable(name="wIndex", storage=-72, var_type="int32_t", identifier=3001)

    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [other_var, renamed_var]

    bv = _FakeBV(functions=[fn])

    result = {
        "op": "local_rename",
        "function": "process_usb",
        "address": "0x401000",
        "variable": "var_48",
        "local_id": "0x401000:local:stack:-72:0:3001",
        "storage": -72,
        "identifier": 3001,
        "source_type": "StackVariableSourceType",
        "is_parameter": False,
        "before_name": "var_48",
        "new_name": "wIndex",
        "requested": {"variable": "var_48", "new_name": "wIndex"},
    }

    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert verified["observed"]["variable"] == "wIndex"


def test_verify_local_rename_fails_when_name_truly_missing(monkeypatch):
    """If no variable at the storage offset has the expected name, verification
    should still fail."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    wrong_var = _FakeVariable(name="var_48", storage=-72, var_type="int32_t", identifier=3001)

    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [wrong_var]

    bv = _FakeBV(functions=[fn])

    result = {
        "op": "local_rename",
        "function": "process_usb",
        "address": "0x401000",
        "variable": "var_48",
        "local_id": "0x401000:local:stack:-72:0:3001",
        "storage": -72,
        "identifier": 3001,
        "source_type": "StackVariableSourceType",
        "is_parameter": False,
        "before_name": "var_48",
        "new_name": "wIndex",
        "requested": {"variable": "var_48", "new_name": "wIndex"},
    }

    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verification_failed"


def test_verify_local_retype_uses_identifier_for_register_locals(monkeypatch):
    """A retyped register/HLIL-visible local lives in neither parameter_vars
    nor stack_layout, so storage-only resolution cannot see it and a change
    that actually landed would fail verification and roll back (#87).
    Identifier-based lookup over the canonical set must find it."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    reg_var = _FakeVariable(
        name="r2_1", storage=2, var_type="char*", identifier=3001,
        source_type="RegisterVariableSourceType",
    )
    fn = _FakeFunction(0x401000, "process_usb")
    fn.hlil = types.SimpleNamespace(vars=[reg_var])
    bv = _FakeBV(functions=[fn])

    result = _local_retype_result(variable="r2_1", storage=2)
    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert verified["observed"]["type"] == "char*"


def test_verify_local_retype_rejects_same_storage_different_identifier(monkeypatch):
    """A different variable at the same storage offset whose type happens to
    match the expected type must not count as success — verification would be
    reading the wrong variable (#87)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    # Neighbor listed FIRST so storage-only resolution would pick it.
    neighbor = _FakeVariable(name="other", storage=-72, var_type="char*", identifier=9999)
    actual = _FakeVariable(name="var_48", storage=-72, var_type="int32_t", identifier=3001)
    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [neighbor, actual]
    bv = _FakeBV(functions=[fn])

    verified = instance._verify_operation(bv, _local_retype_result())
    assert verified["status"] == "verification_failed"
    assert verified["observed"]["variable"] == "var_48"
    assert verified["observed"]["type"] == "int32_t"


def test_verify_local_retype_passes_when_new_type_on_alt_entry_same_identifier(monkeypatch):
    """After analysis BN may keep both an auto and a user entry at the same
    storage offset. If the primary entry still shows the old type but the
    alternate entry (same identifier) carries the expected type, verification
    should succeed — mirroring the rename path."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    stale = _FakeVariable(name="var_48", storage=-72, var_type="int32_t", identifier=3001)
    fresh = _FakeVariable(name="var_48", storage=-72, var_type="char*", identifier=3001)
    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [stale, fresh]
    bv = _FakeBV(functions=[fn])

    verified = instance._verify_operation(bv, _local_retype_result())
    assert verified["status"] == "verified"
    assert verified["observed"]["type"] == "char*"


def test_verify_local_retype_falls_back_to_storage_without_identifier(monkeypatch):
    """When no identifier was recorded, storage resolution remains the only
    handle and must still work."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    var = _FakeVariable(name="var_48", storage=-72, var_type="char*", identifier=3001)
    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [var]
    bv = _FakeBV(functions=[fn])

    verified = instance._verify_operation(bv, _local_retype_result(identifier=None))
    assert verified["status"] == "verified"


def test_verify_local_retype_fails_when_identifier_vanished(monkeypatch):
    """If the recorded identifier no longer resolves, verification must report
    failure rather than silently verifying a same-storage stranger."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    stranger = _FakeVariable(name="other", storage=-72, var_type="char*", identifier=9999)
    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [stranger]
    bv = _FakeBV(functions=[fn])

    verified = instance._verify_operation(bv, _local_retype_result())
    assert verified["status"] == "verification_failed"
    assert verified["observed"]["variable"] is None


def test_verify_local_retype_relocates_register_local_dropped_from_hlil(monkeypatch):
    """Narrowing a register-backed local (u32 -> u8) can drop it out of
    hlil.vars even though func.vars still carries it correctly narrowed, so the
    canonical (param/stack/hlil) scan misses it. Verification must relocate it
    by its stable identifier across the full func.vars set and report
    `verified`, not a cry-wolf `verification_failed` (#156)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    narrowed = _FakeVariable(
        name="x0_1", storage=34, var_type="uint8_t", identifier=3001,
        source_type="RegisterVariableSourceType",
    )
    fn = _FakeFunction(0x401000, "process_usb")
    fn.hlil = types.SimpleNamespace(vars=[])  # dropped out of HLIL after narrow
    fn.vars = [narrowed]                       # but still in the complete set
    bv = _FakeBV(functions=[fn])

    result = _local_retype_result(
        variable="x0_1", storage=34, identifier=3001,
        source_type="RegisterVariableSourceType",
        before_type="int32_t", expected_type="uint8_t",
    )
    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert verified["observed"]["type"] == "uint8_t"
    assert verified["observed"]["variable"] == "x0_1"


def test_verify_local_retype_relocates_register_local_narrowed_u16(monkeypatch):
    """Same relocation, u32 -> u16 (the other narrowing in #156's AC)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    narrowed = _FakeVariable(
        name="x0_1", storage=34, var_type="uint16_t", identifier=3001,
        source_type="RegisterVariableSourceType",
    )
    fn = _FakeFunction(0x401000, "process_usb")
    fn.hlil = types.SimpleNamespace(vars=[])
    fn.vars = [narrowed]
    bv = _FakeBV(functions=[fn])

    result = _local_retype_result(
        variable="x0_1", storage=34, identifier=3001,
        source_type="RegisterVariableSourceType",
        before_type="int32_t", expected_type="uint16_t",
    )
    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert verified["observed"]["type"] == "uint16_t"


def test_verify_local_retype_funcvars_match_is_identifier_exact(monkeypatch):
    """The func.vars fallback matches on the unique identifier only: a
    same-storage stranger with the expected type but a different identifier
    must not be accepted (mirrors the canonical-scan safety guarantee)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    stranger = _FakeVariable(
        name="other", storage=34, var_type="uint8_t", identifier=9999,
        source_type="RegisterVariableSourceType",
    )
    fn = _FakeFunction(0x401000, "process_usb")
    fn.hlil = types.SimpleNamespace(vars=[])
    fn.vars = [stranger]  # id 3001 truly gone
    bv = _FakeBV(functions=[fn])

    result = _local_retype_result(
        variable="x0_1", storage=34, identifier=3001,
        source_type="RegisterVariableSourceType",
        before_type="int32_t", expected_type="uint8_t",
    )
    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verification_failed"
    assert verified["observed"]["variable"] is None


# ---------------------------------------------------------------------------
# Verification: prototype with implicit calling convention
# ---------------------------------------------------------------------------


def test_verify_prototype_passes_with_implicit_calling_convention(monkeypatch):
    """BN analysis may add __convention("cdecl") to the function type after
    set_user_type.  Verification should normalise calling conventions before
    comparing."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _ConventionFunction(_FakeFunction):
        def __init__(self):
            # After set_user_type + analysis, BN reports the type WITH
            # the implicit convention annotation.
            super().__init__(
                0x43F200,
                "parse_config",
                'int32_t __convention("cdecl")(char const* path)',
            )

        def set_user_type(self, value):
            # Store with convention added by analysis.
            super().set_user_type('int32_t __convention("cdecl")(char const* path)')

    class _ConventionBV(_FakeBV):
        def parse_type_string(self, declaration):
            # parse_type_string returns WITHOUT convention.
            return _FakeType("int32_t(char const* path)", type_class="FunctionTypeClass"), None

    fn = _ConventionFunction()
    bv = _ConventionBV(functions=[fn])

    result = instance._op_set_prototype(
        bv,
        {
            "op": "set_prototype",
            "identifier": "parse_config",
            "prototype": "int32_t parse_config(char const* path)",
        },
    )

    # expected_prototype comes from str(parse_type_string(...)): no convention
    assert result["expected_prototype"] == "int32_t(char const* path)"
    # observed will be the fn.type string WITH __convention("cdecl")
    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert '__convention("cdecl")' in verified["observed"]["prototype"]


def test_verify_prototype_still_fails_on_real_mismatch(monkeypatch):
    """When the actual return type or params differ, verification must still
    fail even after convention normalisation."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _MismatchFunction(_FakeFunction):
        def __init__(self):
            super().__init__(0x43F200, "parse_config", "void*(int32_t x)")

        def set_user_type(self, value):
            # Analysis "corrected" the type to something different.
            super().set_user_type("void*(int32_t x)")

    class _MismatchBV(_FakeBV):
        def parse_type_string(self, declaration):
            return _FakeType("int32_t(char const* path)", type_class="FunctionTypeClass"), None

    fn = _MismatchFunction()
    bv = _MismatchBV(functions=[fn])

    result = instance._op_set_prototype(
        bv,
        {
            "op": "set_prototype",
            "identifier": "parse_config",
            "prototype": "int32_t parse_config(char const* path)",
        },
    )

    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verification_failed"


def test_verify_prototype_passes_when_bn_infers_pure_attribute(monkeypatch):
    """BN may re-infer a __pure / __noreturn attribute suffix after
    set_user_type (common on accessors). The requested type lacked it but is
    semantically identical, so the readback must normalise the attribute and
    report `verified` -- not verification_failed + revert the valid edit (#199)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _PureFunction(_FakeFunction):
        def __init__(self):
            # BN auto-typed this accessor int64_t() __pure before the edit.
            super().__init__(0x405250, "reset", "int64_t() __pure")

        def set_user_type(self, value):
            # After set_user_type + analysis BN re-adds the __pure suffix that
            # the requested prototype did not carry.
            super().set_user_type("void(void* self) __pure")

    class _PureBV(_FakeBV):
        def parse_type_string(self, declaration):
            # parse_type_string returns the requested type WITHOUT __pure.
            return _FakeType("void(void* self)", type_class="FunctionTypeClass"), None

    fn = _PureFunction()
    bv = _PureBV(functions=[fn])

    result = instance._op_set_prototype(
        bv,
        {
            "op": "set_prototype",
            "identifier": "reset",
            "prototype": "void reset(void* self)",
        },
    )

    assert result["expected_prototype"] == "void(void* self)"
    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert "__pure" in verified["observed"]["prototype"]


def test_batch_apply_binder_rejects_nonboolean_preview(monkeypatch):
    """A raw/manifest client sending {"preview": "false"} must be rejected, not
    silently coerced to truthy preview mode (#128)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    with pytest.raises(bridge.OperationFailure) as exc:
        bridge._bind_batch_apply(instance, {"preview": "false", "ops": []}, None)
    assert exc.value.status == "invalid_request"


def test_mutation_binders_reject_nonboolean_preview(monkeypatch):
    """Every single-mutation binder validates its `preview` flag as a real JSON
    boolean before dispatching to _mutation (#128)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    binders = [
        bridge._bind_function_create,
        bridge._bind_rename_symbol,
        bridge._bind_set_comment,
        bridge._bind_delete_comment,
        bridge._bind_set_prototype,
        bridge._bind_local_rename,
        bridge._bind_local_retype,
        bridge._bind_struct_field_set,
        bridge._bind_struct_field_rename,
        bridge._bind_struct_field_delete,
        bridge._bind_types_declare,
    ]
    # `address` satisfies _bind_function_create's params["address"] lookup, which
    # is evaluated before the preview arg; harmless for the other binders.
    for binder in binders:
        with pytest.raises(bridge.OperationFailure) as exc:
            binder(instance, {"preview": "false", "address": "0x1000"}, None)
        assert exc.value.status == "invalid_request", binder.__name__


def test_struct_field_set_rejects_nonboolean_overwrite_existing(monkeypatch):
    """`overwrite_existing` is a documented boolean op field; a string must be
    rejected rather than coerced to True and silently overwriting (#128)."""
    bridge, instance, builder, bv = _struct_set_instance(monkeypatch, [(0, "x")])
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._op_struct_field_set(bv, {
            "struct_name": "S", "offset": "0x8", "field_name": "newf",
            "field_type": "int32_t", "overwrite_existing": "false"})
    assert exc.value.status == "invalid_request"
    assert builder.added == []  # never reached add_member_at_offset


def test_parse_concrete_type_rejects_inline_bitfield(monkeypatch):
    """An inline struct with a `:N` bitfield must be rejected, not silently
    accepted: BN's headless parser drops the width and mis-lays-out the members
    yet the verify path (applied-vs-applied) would report it verified -- the same
    root as #322, but on the retype/field-set ops that lacked the guard (#367)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _BV:
        def parse_type_string(self, s):
            raise AssertionError("bitfield must be rejected before parse")

    with pytest.raises(bridge.OperationFailure) as exc:
        bridge.mutation_engine._parse_concrete_type(
            instance.ctx, _BV(), {"op": "local_retype"},
            "struct{unsigned a:3;}", label="type")
    assert exc.value.status == "invalid_request"
    assert "bitfield" in exc.value.message.lower()


def test_parse_concrete_type_rejects_dropped_pointer(monkeypatch):
    """A pointer-to-inline-anonymous-aggregate (`struct{...}*`) whose trailing `*`
    BN silently drops -- parsing it to a struct VALUE -- must be rejected, not
    reported verified against the coerced value type (#367)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _BV:
        def parse_type_string(self, s):
            return _FakeType("struct", type_class="StructureTypeClass"), None

    with pytest.raises(bridge.OperationFailure) as exc:
        bridge.mutation_engine._parse_concrete_type(
            instance.ctx, _BV(), {"op": "local_retype"},
            "struct{int a;int b;}*", label="type")
    assert exc.value.status == "invalid_request"
    assert "pointer" in exc.value.message.lower()


def test_parse_concrete_type_accepts_real_pointer(monkeypatch):
    """A genuine pointer type (the parser keeps pointer-ness) passes unchanged --
    only a SILENTLY-coerced pointer is rejected."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _BV:
        def parse_type_string(self, s):
            return _FakeType("char*", type_class="PointerTypeClass"), None

    parsed, _ = bridge.mutation_engine._parse_concrete_type(
        instance.ctx, _BV(), {"op": "local_retype"}, "char*", label="type")
    assert str(parsed) == "char*"


def test_struct_field_set_rejects_inline_bitfield_type(monkeypatch):
    """struct field set must reject a bitfield field type -- the #322 guard was
    absent on this op, so a mis-laid-out bitfield struct was applied + reported
    verified (#367)."""
    bridge, instance, builder, bv = _struct_set_instance(monkeypatch, [(0, "x")])
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._op_struct_field_set(bv, {
            "struct_name": "S", "offset": "0x8", "field_name": "f",
            "field_type": "struct{unsigned a:3;}"})
    assert exc.value.status == "invalid_request"
    assert builder.added == []


def test_struct_field_set_rejects_negative_offset(monkeypatch):
    """A negative offset must give a clear 'offset must be >= 0' error, not the
    misleading 'No effective change detected' verification_failed that BN's silent
    add_member_at_offset no-op produced (#369)."""
    bridge, instance, builder, bv = _struct_set_instance(monkeypatch, [(0, "x")])
    # both decimal (-8, parses negative) and hex (-0x8, which _parse_address
    # rejects outright) must hit the same actionable ">= 0" message, and an
    # int -8 from a raw-socket client too.
    for bad in ("-8", "-0x8", -8):
        with pytest.raises(bridge.OperationFailure) as exc:
            instance._op_struct_field_set(bv, {
                "struct_name": "S", "offset": bad, "field_name": "f",
                "field_type": "int"})
        assert exc.value.status == "invalid_request", bad
        assert ">= 0" in exc.value.message, bad
    assert builder.added == []


def test_struct_field_set_rejects_absurd_offset(monkeypatch):
    """An offset that would explode the struct into a multi-GB sparse type is a
    likely typo and must be soft-rejected, not applied + reported verified with a
    4GB+ struct width (#369)."""
    bridge, instance, builder, bv = _struct_set_instance(monkeypatch, [(0, "x")])
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._op_struct_field_set(bv, {
            "struct_name": "S", "offset": "0xFFFFFFFF", "field_name": "f",
            "field_type": "int"})
    assert exc.value.status == "invalid_request"
    assert builder.added == []


def test_struct_field_set_accepts_normal_offset(monkeypatch):
    """A sane in-range offset still applies -- the new bounds only catch the
    degenerate negative / absurd cases (#369)."""
    bridge, instance, builder, bv = _struct_set_instance(monkeypatch, [(0, "x")])
    instance._op_struct_field_set(bv, {
        "struct_name": "S", "offset": "0x10", "field_name": "f",
        "field_type": "int"})
    assert len(builder.added) == 1


# ---------------------------------------------------------------------------
# batch_apply: missing target must stay None, not become "None"
# ---------------------------------------------------------------------------


def test_batch_apply_passes_none_target_through(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    captured: dict = {}

    def fake_mutation(selector, preview, operations):
        captured["selector"] = selector
        captured["preview"] = preview
        captured["operations"] = operations
        return {"success": True}

    monkeypatch.setattr(instance, "_mutation", fake_mutation)

    instance._dispatch_on_main("batch_apply", {"ops": [{"op": "rename_symbol"}]}, None)
    # Both manifest target and request target are absent -> the single-open-
    # target default must still apply, so the selector stays None (not "None").
    assert captured["selector"] is None

    instance._dispatch_on_main("batch_apply", {"ops": []}, "alpha.bndb")
    assert captured["selector"] == "alpha.bndb"

    instance._dispatch_on_main(
        "batch_apply", {"ops": [], "target": "beta.bndb"}, "alpha.bndb"
    )
    assert captured["selector"] == "beta.bndb"


# ---------------------------------------------------------------------------
# Verification: fallback must not accept an unrelated same-named variable
# ---------------------------------------------------------------------------


def test_verify_local_rename_rejects_unrelated_var_with_target_name(monkeypatch):
    """Two variables share a storage slot. The OTHER one (different identifier)
    already carries the requested name; the renamed variable still shows its
    auto name, i.e. the rename did not land. Verification must fail instead of
    crediting the neighbor's name."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    renamed = _FakeVariable(name="var_48", storage=-72, var_type="int32_t", identifier=3001)
    other = _FakeVariable(name="wIndex", storage=-72, var_type="int32_t", identifier=9999)

    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [renamed, other]

    bv = _FakeBV(functions=[fn])

    result = {
        "op": "local_rename",
        "function": "process_usb",
        "address": "0x401000",
        "variable": "var_48",
        "local_id": "0x401000:local:stack:-72:0:3001",
        "storage": -72,
        "identifier": 3001,
        "source_type": "StackVariableSourceType",
        "is_parameter": False,
        "before_name": "var_48",
        "new_name": "wIndex",
        "requested": {"variable": "var_48", "new_name": "wIndex"},
    }

    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verification_failed"


def test_apply_operation_missing_op_key_is_invalid_request(monkeypatch):
    # A manifest op without an `op` key must be invalid_request naming `op`, NOT
    # silently dispatched as a rename_symbol (which risks a wrong mutation) (#48).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._apply_operation(None, {"identifier": "x", "new_name": "y"})
    assert exc.value.status == "invalid_request"
    assert "'op'" in str(exc.value)


def test_operation_failure_result_missing_op_is_honest(monkeypatch):
    # The per-op failure echo for an op missing `op` must not claim rename_symbol.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    exc = bridge.OperationFailure("invalid_request", "missing op")
    result = instance._operation_failure_result({"identifier": "x"}, exc)
    assert result["op"] == "<missing>"


def test_apply_operation_non_object_op_is_invalid_request(monkeypatch):
    # A non-object manifest op element (e.g. "ops": ["foo"]) must be a clean
    # invalid_request, not an AttributeError (#48).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._apply_operation(None, "not_an_object")
    assert exc.value.status == "invalid_request"
    # the failure-result/echo helpers must tolerate the non-dict op too
    assert instance._operation_requested("not_an_object") == {}
    assert instance._operation_failure_result("not_an_object", exc.value)["op"] == "<non-object>"


def test_apply_operation_missing_field_is_invalid_request(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    # rename_symbol requires 'identifier' (and 'new_name'); omitting it is a
    # malformed request, not an unsupported operation.
    with pytest.raises(bridge.OperationFailure) as excinfo:
        instance._apply_operation(bv, {"op": "rename_symbol"})
    assert excinfo.value.status == "invalid_request"
    assert "identifier" in str(excinfo.value)


def test_invalid_request_counts_as_a_failed_mutation_status():
    from bn.formatters import FAILED_MUTATION_STATUSES
    assert "invalid_request" in FAILED_MUTATION_STATUSES


def test_struct_field_rename_accepts_offset(monkeypatch):
    bridge, instance, builder = _struct_instance(
        monkeypatch, [_FakeStructMember(0, "a"), _FakeStructMember(8, "b")])
    res = instance._op_struct_field_rename(None, {"struct_name": "S", "old_name": "0x8", "new_name": "bb"})
    assert res["old_name"] == "b"  # offset 0x8 resolved to field 'b'
    assert builder.members[1].name == "bb"


def test_struct_field_rename_still_accepts_name(monkeypatch):
    bridge, instance, builder = _struct_instance(
        monkeypatch, [_FakeStructMember(0, "a"), _FakeStructMember(8, "b")])
    res = instance._op_struct_field_rename(None, {"struct_name": "S", "old_name": "a", "new_name": "aa"})
    assert res["old_name"] == "a"
    assert builder.members[0].name == "aa"


def test_struct_field_delete_accepts_offset(monkeypatch):
    bridge, instance, builder = _struct_instance(
        monkeypatch, [_FakeStructMember(0, "a"), _FakeStructMember(8, "b")])
    res = instance._op_struct_field_delete(None, {"struct_name": "S", "field_name": "0x8"})
    assert res["field_name"] == "b"
    assert [m.name for m in builder.members] == ["a"]


def test_struct_field_unknown_locator_is_invalid_request(monkeypatch):
    bridge, instance, builder = _struct_instance(monkeypatch, [_FakeStructMember(0, "a")])
    with pytest.raises(bridge.OperationFailure) as excinfo:
        instance._op_struct_field_rename(None, {"struct_name": "S", "old_name": "0x99", "new_name": "x"})
    assert excinfo.value.status == "invalid_request"


def test_struct_field_offset_delete_targets_right_field_on_duplicate_names(monkeypatch):
    # Two members share the name 'dup' at offsets 0x0 and 0x8. Deleting by
    # offset 0x8 must remove the SECOND one. A name round-trip would resolve via
    # index_by_name's first match (0x0) and silently delete the wrong field (#25).
    bridge, instance, builder = _struct_instance(
        monkeypatch, [_FakeStructMember(0, "dup"), _FakeStructMember(8, "dup")])
    res = instance._op_struct_field_delete(None, {"struct_name": "S", "field_name": "0x8"})
    assert res["field_name"] == "dup"
    assert [(m.offset, m.name) for m in builder.members] == [(0, "dup")]  # 0x8 gone, 0x0 kept


def test_struct_field_delete_trailing_shrinks_struct_width(monkeypatch):
    # #320: removing the field that reaches the struct end must shrink the width.
    # BN's builder.remove() leaves the width stale, so the delete would otherwise
    # keep phantom trailing bytes (size still 0x1c) and still report verified.
    pad = _FakeStructMember(0x0, "pad", types.SimpleNamespace(width=24))
    extra = _FakeStructMember(0x18, "extra", types.SimpleNamespace(width=4))
    bridge, instance, builder = _struct_instance(monkeypatch, [pad, extra])
    builder.width = 0x1c
    instance._op_struct_field_delete(None, {"struct_name": "S", "field_name": "extra"})
    assert [m.name for m in builder.members] == ["pad"]
    assert builder.width == 0x18  # shrank to the end of the new last field


def test_struct_field_delete_only_member_zeroes_width(monkeypatch):
    # Deleting the sole member leaves an empty struct of width 0 (not the stale
    # old width).
    only = _FakeStructMember(0x0, "pad", types.SimpleNamespace(width=24))
    bridge, instance, builder = _struct_instance(monkeypatch, [only])
    builder.width = 0x18
    instance._op_struct_field_delete(None, {"struct_name": "S", "field_name": "pad"})
    assert builder.members == []
    assert builder.width == 0


def test_struct_field_delete_trailing_tie_keeps_width(monkeypatch):
    # Two members both reach the struct end (an overlay at 0x0): deleting one must
    # NOT shrink, because the other still defines the end. Guards the
    # `new_end < old_width` boundary.
    a = _FakeStructMember(0x0, "a", types.SimpleNamespace(width=8))
    b = _FakeStructMember(0x0, "b", types.SimpleNamespace(width=8))
    bridge, instance, builder = _struct_instance(monkeypatch, [a, b])
    builder.width = 0x8
    res = instance._op_struct_field_delete(None, {"struct_name": "S", "field_name": "a"})
    assert builder.width == 0x8           # 'b' still reaches the end
    assert res["expected_width"] is None  # no shrink intended


def test_struct_field_delete_overlapping_trailing_shrinks_to_remaining(monkeypatch):
    # A big member at 0x0 overlapping a smaller later one defines the end; deleting
    # it shrinks the width down to the remaining member's end (not to 0).
    big = _FakeStructMember(0x0, "big", types.SimpleNamespace(width=0x20))
    small = _FakeStructMember(0x4, "small", types.SimpleNamespace(width=4))
    bridge, instance, builder = _struct_instance(monkeypatch, [big, small])
    builder.width = 0x20
    res = instance._op_struct_field_delete(None, {"struct_name": "S", "field_name": "big"})
    assert builder.width == 0x8          # 'small' ends at 0x8
    assert res["expected_width"] == 0x8


def test_verify_struct_field_delete_flags_stale_width(monkeypatch):
    # The verifier must REJECT a delete that left the width stale (the op intended
    # a shrink to 0x18 but the live width is still 0x1c) instead of reporting
    # verified -- the #320 false-positive class. This is the closed loop: even if
    # the width assignment silently failed, the readback is checked.
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    live_type = types.SimpleNamespace(width=0x1c, members=[
        types.SimpleNamespace(offset=0x0, name="pad", type=types.SimpleNamespace(width=0x18)),
    ])
    bv = types.SimpleNamespace(get_type_by_name=lambda n: live_type)
    ctx = bridge.BinaryNinjaBridge().ctx
    item = {"struct_name": "S", "field_name": "extra", "member_offset": 0x18, "expected_width": 0x18}
    with pytest.raises(bridge.OperationFailure) as exc:
        me._verify_struct_field_delete(ctx, bv, item)
    assert exc.value.status == "verification_failed"
    assert "width" in str(exc.value).lower()


def test_verify_struct_field_delete_passes_when_width_shrank(monkeypatch):
    # The contrast: when the live width matches the intended shrink, the delete
    # verifies cleanly.
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    live_type = types.SimpleNamespace(width=0x18, members=[
        types.SimpleNamespace(offset=0x0, name="pad", type=types.SimpleNamespace(width=0x18)),
    ])
    bv = types.SimpleNamespace(get_type_by_name=lambda n: live_type)
    ctx = bridge.BinaryNinjaBridge().ctx
    item = {"struct_name": "S", "field_name": "extra", "member_offset": 0x18, "expected_width": 0x18}
    out = me._verify_struct_field_delete(ctx, bv, item)
    assert out["status"] == "verified"
    assert out["observed"]["width"] == 0x18


def test_struct_field_delete_interior_keeps_width(monkeypatch):
    # A field that does NOT reach the struct end must not collapse the width:
    # in a partially-recovered struct sized larger than its mapped fields, the
    # explicit width is intentional and deleting an early field must preserve it.
    a = _FakeStructMember(0x0, "a", types.SimpleNamespace(width=4))
    b = _FakeStructMember(0x8, "b", types.SimpleNamespace(width=4))
    bridge, instance, builder = _struct_instance(monkeypatch, [a, b])
    builder.width = 0x100  # 256-byte struct, only two fields mapped so far
    instance._op_struct_field_delete(None, {"struct_name": "S", "field_name": "b"})
    assert [m.name for m in builder.members] == ["a"]
    assert builder.width == 0x100  # 'b' didn't reach the end -> width unchanged


def test_struct_field_offset_rename_targets_right_field_on_duplicate_names(monkeypatch):
    bridge, instance, builder = _struct_instance(
        monkeypatch, [_FakeStructMember(0, "dup"), _FakeStructMember(8, "dup")])
    instance._op_struct_field_rename(None, {"struct_name": "S", "old_name": "0x8", "new_name": "renamed"})
    # only the member at offset 0x8 is renamed; offset 0x0 is untouched
    assert [(m.offset, m.name) for m in builder.members] == [(0, "dup"), (8, "renamed")]


def test_struct_field_set_no_overwrite_at_occupied_offset_refuses(monkeypatch):
    # --no-overwrite at an occupied offset must REFUSE, not append an overlapping
    # member (BN's add_member_at_offset(overwrite=False) silently overlaps) (#56).
    bridge, instance, builder, bv = _struct_set_instance(monkeypatch, [(0, "x")])
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._op_struct_field_set(bv, {
            "struct_name": "S", "offset": "0x0", "field_name": "dupfld",
            "field_type": "int32_t", "overwrite_existing": False})
    assert exc.value.status == "invalid_request"
    assert "x" in str(exc.value)              # names the existing member
    assert builder.added == []                # never reached add_member_at_offset


def test_struct_field_set_no_overwrite_at_free_offset_adds(monkeypatch):
    # The contrast case: --no-overwrite at a FREE offset still adds.
    bridge, instance, builder, bv = _struct_set_instance(monkeypatch, [(0, "x")])
    res = instance._op_struct_field_set(bv, {
        "struct_name": "S", "offset": "0x8", "field_name": "newf",
        "field_type": "int32_t", "overwrite_existing": False})
    assert res["before_member"] is None       # 0x8 is free
    assert builder.added and builder.added[0][0] == "newf"


def test_struct_field_set_no_overwrite_refuses_interior_overlap(monkeypatch):
    # An offset that lands INSIDE a wider member (0x4 within an 8-byte member at
    # 0x0) overlaps just as much as an exact-start collision -- must refuse (#56).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    builder = _AddableStructBuilder()
    big = types.SimpleNamespace(offset=0, name="big", type=types.SimpleNamespace(width=8))
    occupied_type = types.SimpleNamespace(members=[big])

    class _BV:
        def parse_type_string(self, s):
            return types.SimpleNamespace(width=4), None   # a 4-byte field

        def get_type_by_name(self, n):
            return occupied_type

    monkeypatch.setattr(bridge.mutation_engine, "_struct_builder", lambda ctx, bv, name: ("S", builder))
    monkeypatch.setattr(bridge.mutation_engine, "_commit_struct_builder", lambda *a, **k: None)
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._op_struct_field_set(_BV(), {
            "struct_name": "S", "offset": "0x4", "field_name": "mid",
            "field_type": "int32_t", "overwrite_existing": False})
    assert exc.value.status == "invalid_request"
    assert "big" in str(exc.value)            # names the spanned member
    assert builder.added == []


def test_struct_field_set_overwrite_at_occupied_offset_replaces(monkeypatch):
    # The other contrast case: default overwrite at an occupied offset still
    # applies (it replaces, not refuses).
    bridge, instance, builder, bv = _struct_set_instance(monkeypatch, [(0, "x")])
    res = instance._op_struct_field_set(bv, {
        "struct_name": "S", "offset": "0x0", "field_name": "replfld",
        "field_type": "int32_t", "overwrite_existing": True})
    assert res["before_member"]["field_name"] == "x"
    assert builder.added and builder.added[0] == ("replfld", 0, True)


def test_types_declare_rejects_bitfield(monkeypatch):
    # #322: BN's parser silently drops `:N` bit widths and lays each bitfield out
    # as a full-width integer at the byte offset of its bit position -> overlapping,
    # oversized members reported as `verified`. Reject the declaration up front
    # (before anything is parsed/applied) with a clear invalid_request.
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx
    op = {"op": "types_declare",
          "declaration": "struct BF { unsigned a:3; unsigned b:5; unsigned c:1; unsigned d:23; };"}
    with pytest.raises(bridge.OperationFailure) as exc:
        me._op_types_declare(ctx, object(), op)
    assert exc.value.status == "invalid_request"
    assert "bitfield" in str(exc.value).lower()


def test_declaration_has_bitfield_classifies_correctly(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    detect = [
        "struct BF { unsigned a:3; unsigned b:5; };",
        "struct R { uint32_t lo : 16, hi : 16; };",   # comma-separated bitfields
        "struct Z { int x:0; int y:1; };",            # zero-width aligner
        "struct N { struct { unsigned a:3; } bits; };",  # nested
        "struct H { unsigned a:0x3; unsigned b:0x5; };",  # HEX widths (overlap-only)
        "struct U2 { unsigned : 3; unsigned b : 5; };",   # anonymous padding bitfield
        "struct C { // /*\n unsigned y:3; /* real */ };",  # bitfield after a //-hidden /*
    ]
    skip = [
        "struct Normal { int a; char b; long c; };",
        "union U { int a; long c; };",
        "enum E { A = 1, B = 2 };",
        "struct Arr { int data[16]; };",
        "struct Cmt { int x; /* width:32 reserved */ char y; };",  # colon-num in comment
        "struct Px { int a; }; // note: 3 fields",                  # colon-num in line comment
        "class D : public Base { int x; };",                       # C++ inheritance
        "struct Ptr { void *fn; int (*cb)(int); };",
        "enum E8 { A = 5, B = A ? A : 3 };",                        # ternary enum value
        "enum E9 { X = (1 > 0) ? 5 : 3 };",                        # ternary, parenthesized
    ]
    for s in detect:
        assert me._declaration_has_bitfield(s), s
    for s in skip:
        assert not me._declaration_has_bitfield(s), s


def test_struct_overflow_member_flags_only_overflow(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine

    def mk(width, members, cls="StructureTypeClass"):
        return types.SimpleNamespace(
            width=width, type_class=cls,
            members=[types.SimpleNamespace(offset=o, name=n, type=types.SimpleNamespace(width=w))
                     for o, n, w in members])

    # union-shape (members overlap at 0x0, none past width): legitimate -> None
    assert me._struct_overflow_member(mk(8, [(0, "a", 4), (0, "c", 8)])) is None
    # normal struct: no member past width -> None
    assert me._struct_overflow_member(mk(16, [(0, "a", 4), (4, "b", 1), (8, "c", 8)])) is None
    # corrupt: member at 0x1 width 4 ends at 0x5, past width 4 -> flagged
    bad = me._struct_overflow_member(mk(4, [(0, "a", 4), (1, "c", 4)]))
    assert bad is not None and bad.name == "c"
    # opaque / forward-declared (width 0): nothing to check
    assert me._struct_overflow_member(mk(0, [])) is None
    # non-structure type: never flagged
    assert me._struct_overflow_member(
        types.SimpleNamespace(width=4, type_class="EnumerationTypeClass", members=[])) is None


def test_types_declare_rejects_overflowing_parsed_struct(monkeypatch):
    # The non-bitfield backstop: if the parser ever emits a struct whose member
    # extends past the type width, refuse to apply it (would otherwise report
    # `verified` on a corrupt layout). Drives _op_types_declare with a stubbed
    # parser returning such a type.
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx
    corrupt = types.SimpleNamespace(
        width=4, type_class="StructureTypeClass",
        members=[types.SimpleNamespace(offset=1, name="c", type=types.SimpleNamespace(width=4))])
    monkeypatch.setattr(me, "_parse_declaration_source",
                        lambda *a, **k: {"types": [("Corrupt", corrupt)], "variables": [], "functions": []})
    op = {"op": "types_declare", "declaration": "struct Corrupt { /* opaque */ };"}
    with pytest.raises(bridge.OperationFailure) as exc:
        me._op_types_declare(ctx, object(), op)
    assert exc.value.status == "invalid_request"
    assert "corrupt layout" in str(exc.value).lower()


def test_struct_field_offset_grammar_matches_set(monkeypatch):
    # A zero-padded offset that `struct field set` accepts (_parse_address) must
    # also resolve in rename/delete; int(text, 0) rejected leading zeros (#25).
    bridge, instance, builder = _struct_instance(
        monkeypatch, [_FakeStructMember(0, "a"), _FakeStructMember(8, "b")])
    res = instance._op_struct_field_rename(
        None, {"struct_name": "S", "old_name": "0008", "new_name": "bb"})
    assert res["old_name"] == "b"
    assert builder.members[1].name == "bb"


def test_single_mutation_missing_field_message_is_neutral(monkeypatch):
    # A single mutation missing a field must NOT be described as a "batch
    # operation" -- it names the op kind and field neutrally (#30).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    with pytest.raises(bridge.OperationFailure) as e:
        instance._apply_operation(bv, {"op": "local_rename", "function": "f", "variable": "v"})
    assert e.value.status == "invalid_request"
    assert "new_name" in str(e.value)
    assert "batch" not in str(e.value).lower()


def test_internal_keyerror_not_mislabeled_as_missing_field(monkeypatch):
    # A KeyError raised deeper than request-field reads (e.g. BN internals) must
    # NOT be reported as a missing request field now that fields are validated
    # up front (#30).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def boom(ctx, b, o):
        raise KeyError("some_internal_key")

    monkeypatch.setattr(bridge.mutation_engine, "_op_rename_symbol", boom)
    with pytest.raises(bridge.OperationFailure) as e:
        instance._apply_operation(bv, {"op": "rename_symbol", "identifier": "x", "new_name": "y"})
    assert "missing required field" not in str(e.value)
    assert "KeyError" in str(e.value)


def test_unsupported_op_kind_uses_neutral_wording(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    with pytest.raises(bridge.OperationFailure) as e:
        instance._apply_operation(bv, {"op": "nonsuch_op"})
    assert e.value.status == "unsupported"
    assert "batch" not in str(e.value).lower()
    # with no close match, the error lists the valid op names so the caller can
    # pick one (#361).
    assert "set_prototype" in str(e.value) and "function_create" in str(e.value)


def test_unsupported_op_suggests_close_cli_verb(monkeypatch):
    """A batch op named with the CLI verb (proto_set) instead of the batch op name
    (set_prototype) must get a did-you-mean suggestion, not a bare unsupported,
    so an agent reusing CLI verbs doesn't silently waste a batch (#361)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    for guessed, real in [("proto_set", "set_prototype"),
                          ("rename_local", "local_rename"),
                          ("retype_local", "local_retype")]:
        with pytest.raises(bridge.OperationFailure) as e:
            instance._apply_operation(bv, {"op": guessed})
        assert e.value.status == "unsupported", guessed
        msg = str(e.value).lower()
        # the SUGGESTION itself must be the right op (not just present in the
        # always-appended op list) -- a misleading suggestion is worse than none.
        assert f"did you mean '{real}'" in msg, (guessed, real, msg)
        assert "batch" not in msg


def test_mutation_mixed_batch_scopes_blast_radius_and_tags_direct(monkeypatch):
    """A mixed batch (types_declare + set_prototype) scopes the blast radius to
    the TYPE op and tags the direct op's affected function `direct`, so the
    set_prototype target is excluded from the type's referenced/reflowed counts
    and the formatter can keep the two apart (Codex review on #240)."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    uses_ep = _FakeFunction(0x10, "uses_ep", "void()")      # references the type
    handler = _FakeFunction(0x401000, "handler", "void()")  # set_prototype target

    def _fns_for_op(ctx, b, op, *, type_limit):
        return [uses_ep] if me._is_type_op(op) else [handler]

    diffs = [
        {"address": "0x10", "before_name": "uses_ep", "after_name": "uses_ep", "changed": True, "diff": ""},
        {"address": "0x401000", "before_name": "handler", "after_name": "handler", "changed": True, "diff": ""},
    ]

    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(me, "_functions_for_op", _fns_for_op)
    monkeypatch.setattr(me, "_guess_affected_functions", lambda ctx, b, ops: [])
    monkeypatch.setattr(me, "_capture_function_snapshots", lambda ctx, b, fns: {})
    monkeypatch.setattr(me, "_capture_type_snapshots", lambda ctx, b, ops: {})
    monkeypatch.setattr(me, "_diff_snapshots", lambda ctx, before, after: [dict(d) for d in diffs])
    monkeypatch.setattr(me, "_diff_type_snapshots", lambda ctx, before, after: [{"type_name": "Ep", "changed": True}])
    monkeypatch.setattr(me, "_apply_operation", lambda ctx, b, op, restores=None: {"op": op.get("op")})
    monkeypatch.setattr(me, "_verify_operation", lambda ctx, b, result: {**result, "status": "verified"})
    monkeypatch.setattr(me, "_annotate_operation_results", lambda ctx, results, type_diffs: results)

    result = instance._mutation("active", False,
                                [{"op": "types_declare"}, {"op": "set_prototype"}])

    assert result["success"] is True
    assert _has_event(bv, "commit")
    # Blast radius counts the type's reach only -- handler (direct) is excluded.
    assert result["affected_summary"] == {"referenced": 1, "reflowed": 1}
    tags = {d["address"]: d["direct"] for d in result["affected_functions"]}
    assert tags == {"0x10": False, "0x401000": True}


def test_count_referenced_functions_is_uncapped_past_snapshot_cap(monkeypatch):
    """affected_functions is capped at 10 for snapshotting, but the reported
    blast radius (affected_summary.referenced) must be the true total -- a struct
    used by 200 functions previously surfaced as "10" with no hint of real scope."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx
    funcs = [_FakeFunction(0x1000 + i * 4, f"f{i}", "void(struct Widget* w)") for i in range(15)]
    bv = _FakeBV(functions=funcs)
    # Sidestep the C parser: the type resolution is exercised elsewhere.
    monkeypatch.setattr(me, "_operation_type_names", lambda c, b, op: ["Widget"])
    ops = [{"op": "types_declare", "declaration": "struct Widget { int x; };"}]

    assert len(me._guess_affected_functions(ctx, bv, ops)) == 10  # snapshot set, capped
    assert me._count_referenced_functions(ctx, bv, ops, fallback=10) == 15  # true total


def test_count_referenced_functions_falls_back_on_scan_error(monkeypatch):
    """A stubbed/odd view must never crash a mutation: the count degrades to the
    capped fallback rather than raising."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx

    def _boom(*a, **k):
        raise RuntimeError("view scan blew up")

    monkeypatch.setattr(me, "_functions_for_op", _boom)
    assert me._count_referenced_functions(ctx, _FakeBV(), [{"op": "set_prototype"}], fallback=4) == 4


def test_diff_snapshots_omits_excerpt_when_full_diff_fits(monkeypatch):
    """A small real body change: the full unified diff fits inline, so the focused
    before/after_excerpt would only duplicate it. The excerpt is reserved for the
    large-function case where the diff gets truncated (see the M14 test above)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    diffs = instance._diff_snapshots(
        {0x1000: {"text": "x = 1;\ny = prev;\nz = 3;", "name": "f"}},
        {0x1000: {"text": "x = 1;\ny = next;\nz = 3;", "name": "f"}},
    )
    d = diffs[0]
    assert d["changed"] is True
    assert "prev" in d["diff"] and "next" in d["diff"]  # change visible inline
    assert "before_excerpt" not in d and "after_excerpt" not in d


def _typedef_alias_bv(*, alias_name="AliasRec", tag="InnerRec"):
    """A fake BV mirroring real BN's typedef-of-struct shape (#246): *alias_name*
    resolves to a NamedTypeReference whose ``.target(bv)`` is the underlying
    registered struct (`tag`), and ``get_type_by_name(tag)`` returns that struct.
    The alias's own ``mutable_copy()`` raises -- exactly as BN's
    NamedTypeReferenceBuilder does on ``add_member_at_offset`` -- so a fix that
    fails to follow the reference reproduces the original crash."""
    underlying = _FakeType(f"struct {tag}", type_class="StructureTypeClass",
                           members=[_FakeMember(0, "x", "uint32_t")])
    builder = _AddableStructBuilder()
    underlying.mutable_copy = lambda: builder
    underlying.registered_name = types.SimpleNamespace(name=tag)

    alias = _FakeType(f"struct {tag} {alias_name}", type_class="NamedTypeReferenceClass")

    def _alias_mutable_copy():
        raise AttributeError(
            "'NamedTypeReferenceBuilder' object has no attribute 'add_member_at_offset'")

    alias.mutable_copy = _alias_mutable_copy
    alias.target = lambda _bv: underlying

    class _BV:
        def __init__(self):
            self.defined = []

        def get_type_by_name(self, n):
            return {alias_name: alias, tag: underlying}.get(n)

        def parse_type_string(self, s):
            return _FakeType("uint32_t", width=4), None

        def define_user_type(self, name, b):
            self.defined.append((name, b))

    return alias, underlying, builder, _BV()


def test_struct_builder_follows_typedef_alias_to_underlying_struct(monkeypatch):
    """`struct field set/rename/delete` on a typedef (the idiomatic
    `typedef struct {..} X;`) must follow the alias (a NamedTypeReference) to the
    underlying registered struct tag and build from THAT -- not crash calling
    add_member_at_offset on a NamedTypeReferenceBuilder (#246). All three field
    ops route through _struct_builder, so fixing it here fixes set/rename/delete."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _alias, _under, builder, bv = _typedef_alias_bv()
    resolved_name, got = bridge.mutation_engine._struct_builder(instance, bv, "AliasRec")
    assert resolved_name == "InnerRec"    # commit to the tag, not the alias
    assert got is builder                 # a real StructureBuilder, not an NTR builder


def test_struct_builder_follows_anonymous_typedef_struct(monkeypatch):
    """`typedef struct {..} AnonRec;` registers the body under an auto-named tag
    (`_AnonRec`); the alias is a NamedTypeReference to it. The follow must land on
    `_AnonRec` -- the most common C struct idiom, not an edge case (#246)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _alias, _under, builder, bv = _typedef_alias_bv(alias_name="AnonRec", tag="_AnonRec")
    resolved_name, got = bridge.mutation_engine._struct_builder(instance, bv, "AnonRec")
    assert resolved_name == "_AnonRec"
    assert got is builder


def test_struct_builder_typedef_to_nonstruct_is_invalid_request(monkeypatch):
    """A name that resolves to a non-aggregate (`typedef uint32_t Foo;` resolves
    straight to the integer type in BN) must raise a clean invalid_request, not a
    raw AttributeError from add_member_at_offset (#246)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    prim = _FakeType("uint32_t", type_class="IntegerTypeClass")

    class _BV:
        def get_type_by_name(self, n):
            return prim if n == "MyU32" else None

    with pytest.raises(bridge.OperationFailure) as exc:
        bridge.mutation_engine._struct_builder(instance, _BV(), "MyU32")
    assert exc.value.status == "invalid_request"


def test_op_struct_field_set_through_typedef_alias_commits_to_tag(monkeypatch):
    """End-to-end through the op handler: a set on the typedef alias must add the
    field to the underlying struct builder and commit it to the TAG name, with the
    result reporting the tag (so the caller learns where the body lives) (#246)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _alias, _under, builder, bv = _typedef_alias_bv()
    res = instance._op_struct_field_set(bv, {
        "struct_name": "AliasRec", "offset": "0x4", "field_name": "added",
        "field_type": "uint32_t", "overwrite_existing": True})
    assert res["struct_name"] == "InnerRec"          # reported as the tag
    assert builder.added and builder.added[0][0] == "added"
    assert ("InnerRec", builder) in bv.defined       # committed to the tag




def test_prototype_matches_ignoring_param_names(monkeypatch):
    """The proto-set verifier accepts a name-omitted prototype whose parameter
    TYPES and return type match BN's auto-named readback, but still rejects a real
    type / arity / return mismatch, and falls back (returns False) when the types
    lack the BN function-type shape or the expected string won't parse (#254)."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine

    def fn_type(ret, *param_types):
        return types.SimpleNamespace(
            return_value=ret,
            parameters=[types.SimpleNamespace(type=t, name=f"arg{i + 1}")
                        for i, t in enumerate(param_types)])

    observed = fn_type("void", "int32_t", "char**", "char**")  # BN auto-named readback

    def bv_returning(expected):
        return types.SimpleNamespace(parse_type_string=lambda s: (expected, None))

    # identical types, names differ/absent -> match (the #254 case)
    assert me._prototype_matches_ignoring_param_names(
        bv_returning(fn_type("void", "int32_t", "char**", "char**")),
        observed, "void(int32_t, char**, char**)")
    # a wrong param type -> reject
    assert not me._prototype_matches_ignoring_param_names(
        bv_returning(fn_type("void", "int32_t", "char*", "char**")),
        observed, "void(int32_t, char*, char**)")
    # wrong arity -> reject
    assert not me._prototype_matches_ignoring_param_names(
        bv_returning(fn_type("void", "int32_t", "char**")),
        observed, "void(int32_t, char**)")
    # wrong return type -> reject
    assert not me._prototype_matches_ignoring_param_names(
        bv_returning(fn_type("int32_t", "int32_t", "char**", "char**")),
        observed, "int32_t(int32_t, char**, char**)")
    # varargs mismatch -> reject (a non-vararg readback must not match a `...`
    # request just because the fixed params line up)
    observed_va = fn_type("int32_t", "char const*")
    observed_va.has_variable_arguments = False
    expected_va = fn_type("int32_t", "char const*")
    expected_va.has_variable_arguments = True
    assert not me._prototype_matches_ignoring_param_names(
        bv_returning(expected_va), observed_va, "int32_t(char const*, ...)")
    # unparseable expected -> False (caller falls back to the string compare)
    def _boom(s):
        raise ValueError("bad")
    assert not me._prototype_matches_ignoring_param_names(
        types.SimpleNamespace(parse_type_string=_boom), observed, "garbage")
    # observed lacking the BN function-type shape (a mocked string type) -> False,
    # so the existing string-compare tests keep their behavior unchanged
    assert not me._prototype_matches_ignoring_param_names(
        bv_returning(fn_type("void", "int32_t")), "void(int32_t arg1)", "void(int32_t)")

def test_prototype_matches_rejects_named_param_that_did_not_land(monkeypatch):
    """#263 review: the name-insensitive acceptance must only tolerate names the
    request OMITTED (BN auto-names those arg1/arg2 on readback -- the #254 case).
    When the request EXPLICITLY named a param and BN read it back as arg1, the
    name did NOT land -- a partial application that must not be reported verified."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine

    def fn_type(ret, *params):
        # params are (type, name) pairs so the test can set explicit names
        return types.SimpleNamespace(
            return_value=ret,
            parameters=[types.SimpleNamespace(type=t, name=n) for t, n in params])

    def bv_returning(expected):
        return types.SimpleNamespace(parse_type_string=lambda s: (expected, None))

    observed = fn_type("int32_t", ("int32_t", "arg1"))  # BN readback: requested name absent

    # request explicitly named the param `fd` -> name did not land -> reject
    assert not me._prototype_matches_ignoring_param_names(
        bv_returning(fn_type("int32_t", ("int32_t", "fd"))),
        observed, "int32_t(int32_t fd)")

    # request OMITTED the name (parses to empty) -> BN auto-named it -> tolerate (#254)
    assert me._prototype_matches_ignoring_param_names(
        bv_returning(fn_type("int32_t", ("int32_t", ""))),
        observed, "int32_t(int32_t)")


# ===========================================================================
# #598 -- post-verify/preview rollback must use _revert_undo_safely and fold
# journaled-undo failure into `restored` (like the apply-failure path).
# ===========================================================================

def test_preview_journaled_undo_failure_is_not_success(monkeypatch):
    """A clean preview whose JOURNALED undo (revert_undo_actions) failed left
    the view modified. It must report success/committed/rolled_back False and a
    structured envelope -- even when the local/drift restores both succeeded
    (#598). Pre-fix the post-verify block called bare bv.revert_undo_actions and
    ignored its outcome, so a failed undo still reported rolled_back True."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None):
        restores.append(lambda: None)
        return {"op": "rename_symbol", "requested": {}}

    _mutation_with_stubs(
        monkeypatch, bridge, instance, bv,
        apply=apply,
        verify=lambda bv_, result: {**result, "status": "verified"},
    )
    # local/drift restores succeed; only the journaled undo fails.
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, r: True)
    monkeypatch.setattr(bridge.mutation_engine, "_revert_undo_safely", lambda ctx, bv_, s: False)

    result = instance._mutation("active", True, [{"op": "rename_symbol"}])

    assert isinstance(result, dict)
    assert result["preview"] is True
    assert result["success"] is False
    assert result["committed"] is False
    assert result["rolled_back"] is False
    assert not _has_event(bv, "commit")  # commit_undo_actions never called on preview


def test_verify_fail_journaled_undo_failure_returns_structured_result(monkeypatch):
    """A live (non-preview) verification-failed batch whose journaled undo fails
    must return a structured dict (success/rolled_back False), NOT a bare
    RuntimeError from the outer handler (#598)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None):
        return {"op": "rename_symbol", "requested": {}}

    _mutation_with_stubs(
        monkeypatch, bridge, instance, bv,
        apply=apply,
        verify=lambda bv_, result: {**result, "status": "verification_failed", "message": "nope"},
    )
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, r: True)
    monkeypatch.setattr(bridge.mutation_engine, "_revert_undo_safely", lambda ctx, bv_, s: False)

    result = instance._mutation("active", False, [{"op": "rename_symbol"}])

    assert isinstance(result, dict)
    assert result["success"] is False
    assert result["rolled_back"] is False


def test_preview_revert_raise_absorbed_into_structured_result(monkeypatch):
    """bv.revert_undo_actions raising on the preview path must be absorbed by
    the real _revert_undo_safely helper into a structured result
    (rolled_back False), not escape as a generic bridge/transport error (#598).
    Uses the REAL helper (unpatched) to prove the path calls it, not bare."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def boom(state):
        raise RuntimeError("undo boom")

    bv.revert_undo_actions = boom

    def apply(bv_, op, restores=None):
        restores.append(lambda: None)
        return {"op": "rename_symbol", "requested": {}}

    _mutation_with_stubs(
        monkeypatch, bridge, instance, bv,
        apply=apply,
        verify=lambda bv_, result: {**result, "status": "verified"},
    )
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, r: True)

    result = instance._mutation("active", True, [{"op": "rename_symbol"}])

    assert isinstance(result, dict)  # no raise
    assert result["success"] is False
    assert result["rolled_back"] is False


# ===========================================================================
# #602 -- sibling status honesty on verification-failure rollback.
# ===========================================================================

def test_verify_fail_restamps_verified_siblings_as_reverted(monkeypatch):
    """Live batch, opA verifies, opB verification_failed, undo+restores succeed:
    opA must report 'reverted' (NOT 'verified'), opB keeps 'verification_failed'
    with its message intact, and 'reverted' stays out of FAILED_MUTATION_STATUSES
    (#602)."""
    from bn.formatters import FAILED_MUTATION_STATUSES

    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None):
        return {"op": op["op"], "requested": {}}

    calls = {"n": 0}

    def verify(bv_, result):
        calls["n"] += 1
        if calls["n"] == 1:
            return {**result, "status": "verified"}
        return {**result, "status": "verification_failed", "message": "prototype did not land"}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply, verify=verify)
    monkeypatch.setattr(bridge.mutation_engine, "_revert_undo_safely", lambda ctx, bv_, s: True)
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, r: True)

    result = instance._mutation("active", False, [{"op": "rename_symbol"}, {"op": "set_prototype"}])

    assert result["success"] is False
    assert result["committed"] is False
    assert result["rolled_back"] is True
    statuses = [r["status"] for r in result["results"]]
    assert statuses[0] == "reverted"
    assert statuses[1] == "verification_failed"
    assert result["results"][1]["message"] == "prototype did not land"
    assert "reverted" not in FAILED_MUTATION_STATUSES


def test_verify_fail_restamps_siblings_rollback_failed_when_revert_fails(monkeypatch):
    """Same batch, but undo/restores fail: sibling A reports 'rollback_failed'
    and top-level rolled_back is False (#602)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None):
        return {"op": op["op"], "requested": {}}

    calls = {"n": 0}

    def verify(bv_, result):
        calls["n"] += 1
        if calls["n"] == 1:
            return {**result, "status": "verified"}
        return {**result, "status": "verification_failed", "message": "did not land"}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply, verify=verify)
    monkeypatch.setattr(bridge.mutation_engine, "_revert_undo_safely", lambda ctx, bv_, s: False)
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, r: True)

    result = instance._mutation("active", False, [{"op": "rename_symbol"}, {"op": "set_prototype"}])

    assert result["rolled_back"] is False
    assert result["results"][0]["status"] == "rollback_failed"
    assert result["results"][1]["status"] == "verification_failed"


def test_verify_fail_all_ops_failed_none_restamped_reverted(monkeypatch):
    """All-ops-failed batch: every op keeps 'verification_failed'; none is
    overwritten to 'reverted' (#602)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None):
        return {"op": op["op"], "requested": {}}

    _mutation_with_stubs(
        monkeypatch, bridge, instance, bv, apply=apply,
        verify=lambda bv_, result: {**result, "status": "verification_failed", "message": "no"},
    )
    monkeypatch.setattr(bridge.mutation_engine, "_revert_undo_safely", lambda ctx, bv_, s: True)
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, r: True)

    result = instance._mutation("active", False, [{"op": "rename_symbol"}, {"op": "set_prototype"}])

    assert [r["status"] for r in result["results"]] == ["verification_failed", "verification_failed"]


def test_noop_op_retains_noop_status_through_rollback(monkeypatch):
    """A genuine no-op op (its requested state was already satisfied) changed
    nothing, so a batch rollback triggered by a SIBLING's verification failure
    must leave it 'noop', not restamp it 'reverted'/'rollback_failed' -- that
    would corrupt the per-op status and the summary counts (#630)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None):
        return {"op": op["op"], "requested": {}}

    calls = {"n": 0}

    def verify(bv_, result):
        calls["n"] += 1
        if calls["n"] == 1:
            return {**result, "status": "noop"}   # already satisfied
        return {**result, "status": "verification_failed", "message": "did not land"}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply, verify=verify)
    monkeypatch.setattr(bridge.mutation_engine, "_revert_undo_safely", lambda ctx, bv_, s: True)
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, r: True)

    result = instance._mutation("active", False, [{"op": "rename_symbol"}, {"op": "set_prototype"}])

    assert result["results"][0]["status"] == "noop"   # unchanged, NOT reverted
    assert result["results"][1]["status"] == "verification_failed"


# ===========================================================================
# #606 -- bridge._mutation marks the view dirty when rolled_back is False.
# ===========================================================================

def _dirty_probe(monkeypatch, bridge, instance, return_value):
    marked = []
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: "BV")
    monkeypatch.setattr(instance.targets, "mark_dirty", lambda bv: marked.append(bv))
    monkeypatch.setattr(bridge.mutation_engine, "_mutation", lambda ctx, *a, **k: return_value)
    instance._mutation("active", return_value.get("preview", False), [{"op": "rename_symbol"}])
    return marked


def test_mutation_dirty_on_rolled_back_false_preview(monkeypatch):
    """A failed preview rollback (rolled_back False) leaves partial state live;
    the bridge must mark dirty so `bn close` warns (#606)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    marked = _dirty_probe(monkeypatch, bridge, instance, {
        "committed": False, "preview": True, "rolled_back": False, "success": False,
        "results": [{"status": "rollback_failed"}],
    })
    assert marked == ["BV"]


def test_mutation_dirty_on_rolled_back_false_live_verify_fail(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    marked = _dirty_probe(monkeypatch, bridge, instance, {
        "committed": False, "preview": False, "rolled_back": False, "success": False,
        "results": [{"status": "verification_failed"}],
    })
    assert marked == ["BV"]


def test_mutation_not_dirty_on_clean_preview_rollback(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    marked = _dirty_probe(monkeypatch, bridge, instance, {
        "committed": False, "preview": True, "rolled_back": True, "success": True,
        "results": [{"status": "verified"}],
    })
    assert marked == []


def test_mutation_not_dirty_on_clean_apply_failure_rollback(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    marked = _dirty_probe(monkeypatch, bridge, instance, {
        "committed": False, "preview": False, "rolled_back": True, "success": False,
        "results": [{"status": "reverted"}],
    })
    assert marked == []


def test_mutation_dirty_on_committed_verified(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    marked = _dirty_probe(monkeypatch, bridge, instance, {
        "committed": True, "preview": False,
        "results": [{"status": "verified"}],
    })
    assert marked == ["BV"]


def test_mutation_not_dirty_on_pure_noop(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    marked = _dirty_probe(monkeypatch, bridge, instance, {
        "committed": True, "preview": False,
        "results": [{"status": "noop"}],
    })
    assert marked == []


def test_mutation_dirty_on_prototype_user_type_residue(monkeypatch):
    """An unclearable has_user_type override left behind must mark the view dirty
    so `bn close` warns, even though the prototype VALUE round-tripped (#630)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    marked = _dirty_probe(monkeypatch, bridge, instance, {
        "committed": False, "preview": False, "rolled_back": False, "success": False,
        "prototype_user_type_residue": True,
        "results": [{"status": "rollback_failed"}],
    })
    assert marked == ["BV"]


def test_end_to_end_clean_preview_leaves_view_unchanged_and_not_dirty(monkeypatch):
    """A clean preview of a local rename, driven through the REAL _mutation path
    (apply -> verify -> undo -> restores -> drift), must leave the variable back
    at its original AUTO name/provenance AND not mark the view dirty (#630). Not a
    fabricated envelope -- it proves the view is actually unchanged."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    me = bridge.mutation_engine

    var = _FakeVariable(name="var_8", storage=-8, var_type="int32_t", identifier=10)
    fn = _FakeFunction(0x401000, "f")
    fn.stack_layout = [var]
    bv = _FakeMutationBV(functions=[fn])
    fn.view = bv

    marked: list = []
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "mark_dirty", lambda b: marked.append(b))
    # Wire the seam/peers so REAL _op_local_rename + restores + drift run; only the
    # snapshot/diff machinery (irrelevant to this assertion) is stubbed out.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.ctx, "_find_function", lambda _bv, ident: fn)
    monkeypatch.setattr(me, "_guess_affected_functions", lambda ctx, b, ops: [fn])
    monkeypatch.setattr(me, "_capture_function_snapshots", lambda ctx, b, fns: {})
    monkeypatch.setattr(me, "_capture_type_snapshots", lambda ctx, b, ops: {})
    monkeypatch.setattr(me, "_diff_snapshots", lambda ctx, before, after: [])
    monkeypatch.setattr(me, "_diff_type_snapshots", lambda ctx, before, after: [])
    monkeypatch.setattr(bridge.vars_mod, "_find_variable_selector", lambda _f, sel: (var, False))
    monkeypatch.setattr(bridge.vars_mod, "_local_id", lambda _f, _v, is_parameter: "lid")
    monkeypatch.setattr(me, "_find_var_for_restore",
                        lambda ctx, _f, identifier, storage, is_parameter: var)
    monkeypatch.setattr(me, "_verify_operation",
                        lambda ctx, b, result: {**result, "status": "verified"})

    result = instance._mutation(
        "active", True,
        [{"op": "local_rename", "function": "f", "variable": "var_8", "new_name": "probe"}],
    )

    assert result["success"] is True
    assert result["rolled_back"] is True
    assert "prototype_user_type_residue" not in result
    # The view is actually unchanged: AUTO provenance and original name restored.
    assert var.name == "var_8"
    assert fn.is_var_user_defined(var) is False
    assert marked == []  # a clean preview does not dirty the view


# ===========================================================================
# #621 -- function-scoped comment/tag verify must resolve by stored start
# address, so a same-batch rename does not false-fail verification.
# ===========================================================================

def _tagged_bv(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    me = bridge.mutation_engine
    fn = _FakeFunction(0x401000, "handle_request")
    fn.basic_blocks = [_FakeBasicBlock(0x401000, 0x401100)]
    bv = _FakeMutationBV(functions=[fn])
    fn.view = bv  # _FakeFunction.add_tag resolves the tag type through the view
    bv.create_tag_type("Bug", "🐞")
    return bridge, instance, me, fn, bv


def test_verify_function_doc_comment_survives_same_batch_rename(monkeypatch):
    bridge, instance, me, fn, bv = _tagged_bv(monkeypatch)
    result = me._op_set_comment(
        instance.ctx, bv,
        {"op": "set_comment", "function": "handle_request", "comment": "entry HTTP handler"},
    )
    assert result["address"] == "0x401000"
    fn.name = fn.raw_name = "handle_http_request"  # a later op renamed it (updates the symbol)
    verified = me._verify_set_comment(instance.ctx, bv, result)
    assert verified["status"] == "verified"


def test_verify_function_doc_delete_comment_survives_rename(monkeypatch):
    bridge, instance, me, fn, bv = _tagged_bv(monkeypatch)
    fn.comment = "old note"
    result = me._op_delete_comment(instance.ctx, bv, {"op": "delete_comment", "function": "handle_request"})
    fn.name = fn.raw_name = "renamed"
    verified = me._verify_delete_comment(instance.ctx, bv, result)
    assert verified["status"] == "verified"


def test_verify_function_tag_add_survives_rename(monkeypatch):
    bridge, instance, me, fn, bv = _tagged_bv(monkeypatch)
    result = me._op_tag_add(instance.ctx, bv, {"op": "tag_add", "function": "handle_request", "type": "Bug"})
    assert result["scope"] == "function"
    fn.name = fn.raw_name = "renamed"
    verified = me._verify_tag_add(instance.ctx, bv, result)
    assert verified["status"] == "verified"


def test_verify_address_tag_add_survives_rename(monkeypatch):
    bridge, instance, me, fn, bv = _tagged_bv(monkeypatch)
    result = me._op_tag_add(instance.ctx, bv, {"op": "tag_add", "address": "0x401010", "type": "Bug"})
    assert result["scope"] == "address"
    fn.name = fn.raw_name = "renamed"
    verified = me._verify_tag_add(instance.ctx, bv, result)
    assert verified["status"] == "verified"


def test_verify_function_tag_remove_records_address_and_survives_rename(monkeypatch):
    bridge, instance, me, fn, bv = _tagged_bv(monkeypatch)
    add = me._op_tag_add(instance.ctx, bv, {"op": "tag_add", "function": "handle_request", "type": "Bug"})
    tag_id = add["tag_id"]
    result = me._op_tag_remove(instance.ctx, bv, {"op": "tag_remove", "tag_id": tag_id})
    # function-scope remove targets now carry the stable start address.
    assert any(t["scope"] == "function" and t.get("address") == "0x401000" for t in result["targets"])
    fn.name = fn.raw_name = "renamed"
    verified = me._verify_tag_remove(instance.ctx, bv, result)
    assert verified["status"] == "verified"


def test_verify_function_doc_comment_missing_fn_raises_verification_failed(monkeypatch):
    """A missing function at the stored address is an honest verification_failed,
    not a crash (#621)."""
    bridge, instance, me, fn, bv = _tagged_bv(monkeypatch)
    result = me._op_set_comment(
        instance.ctx, bv,
        {"op": "set_comment", "function": "handle_request", "comment": "x"},
    )
    bv.functions.clear()  # function gone at the stored address
    with pytest.raises(bridge.OperationFailure) as exc:
        me._verify_set_comment(instance.ctx, bv, result)
    assert exc.value.status == "verification_failed"


# ===========================================================================
# #630 -- tag verification must check the EXACT target function/address, not
# just "any function containing the address" / a name fallback.
# ===========================================================================

def test_address_tag_add_records_function_start(monkeypatch):
    """An address-scope tag add records the EXACT function it landed on (stable
    start address) so verification can check that function, not any containing
    one (#630)."""
    bridge, instance, me, fn, bv = _tagged_bv(monkeypatch)
    result = me._op_tag_add(instance.ctx, bv, {"op": "tag_add", "address": "0x401010", "type": "Bug"})
    assert result["scope"] == "address"
    assert result["function_start"] == "0x401000"


def test_verify_address_tag_add_checks_exact_function_not_any_containing(monkeypatch):
    """A matching tag on a DIFFERENT function that also contains the address must
    NOT satisfy verification: the tag was added to a specific function (#630)."""
    bridge, instance, me, fn, bv = _tagged_bv(monkeypatch)
    # A second function that also contains 0x401010 but is NOT where the tag went.
    other = _FakeFunction(0x401008, "sibling")
    other.basic_blocks = [_FakeBasicBlock(0x401008, 0x401100)]
    other.view = bv
    bv.functions.append(other)

    result = me._op_tag_add(instance.ctx, bv, {"op": "tag_add", "address": "0x401010", "type": "Bug"})
    assert result["function_start"] == "0x401000"
    # Put a matching tag on the OTHER function and remove it from the real target.
    other.add_tag("Bug", "", 0x401010)
    fn._address_tags[0x401010] = []
    with pytest.raises(bridge.OperationFailure) as exc:
        me._verify_tag_add(instance.ctx, bv, result)
    assert exc.value.status == "verification_failed"


def test_verify_function_tag_remove_treats_missing_function_as_removed(monkeypatch):
    """A function-scope tag remove whose recorded start address no longer
    resolves to a function verifies as removed (its tags are gone with it) --
    it must NOT fall back to a name lookup, which would raise 'function not
    found' (or match a DIFFERENT function) instead of cleanly verifying (#630)."""
    bridge, instance, me, fn, bv = _tagged_bv(monkeypatch)
    add = me._op_tag_add(instance.ctx, bv, {"op": "tag_add", "function": "handle_request", "type": "Bug"})
    result = me._op_tag_remove(instance.ctx, bv, {"op": "tag_remove", "tag_id": add["tag_id"]})
    bv.functions.clear()  # the function was deleted after the removal
    verified = me._verify_tag_remove(instance.ctx, bv, result)
    assert verified["status"] == "verified"  # gone -> its function tag cannot remain


# ===========================================================================
# #581 -- preview of a local rename/retype on an AUTO variable must NOT pin it
# as USER: the restore uses delete_user_var, not create_user_var.
# ===========================================================================

def _local_rename_engine(monkeypatch, fn, var):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    me = bridge.mutation_engine
    bv = _FakeMutationBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_find_function", lambda _bv, ident: fn)
    monkeypatch.setattr(bridge.vars_mod, "_find_variable_selector", lambda _f, sel: (var, False))
    monkeypatch.setattr(bridge.mutation_engine, "_find_var_for_restore",
                        lambda ctx, _f, identifier, storage, is_parameter: var)
    monkeypatch.setattr(bridge.vars_mod, "_local_id", lambda _f, _v, is_parameter: "lid")
    return bridge, instance, me, bv


def test_preview_local_rename_restores_auto_var_via_delete_user_var(monkeypatch):
    """An AUTO local previewed-renamed then reverted must return to is_user
    False (delete_user_var), not stay pinned USER by a create_user_var replay
    (#581)."""
    var = _FakeVariable(name="var_8", storage=-8, var_type="int32_t", identifier=10)
    fn = _FakeFunction(0x401000, "f")
    fn.stack_layout = [var]
    bridge, instance, me, bv = _local_rename_engine(monkeypatch, fn, var)

    assert fn.is_var_user_defined(var) is False
    restores: list = []
    me._op_local_rename(
        instance.ctx, bv,
        {"op": "local_rename", "function": "f", "variable": "var_8", "new_name": "probe"},
        restores,
    )
    assert fn.is_var_user_defined(var) is True   # apply pinned USER
    assert var.name == "probe"

    assert me._run_local_restores(instance.ctx, bv, restores) is True
    # Provenance back to AUTO -- NOT pinned USER (the #581 bug).
    assert fn.is_var_user_defined(var) is False
    assert var.name == "var_8"


def test_preview_local_rename_preserves_genuine_user_var(monkeypatch):
    """Negative control: a variable that ALREADY had a USER definition must be
    replayed with create_user_var and survive the preview as USER (#581 must not
    over-correct genuine user vars)."""
    var = _FakeVariable(name="var_8", storage=-8, var_type="int32_t", identifier=10)
    fn = _FakeFunction(0x401000, "f")
    fn.stack_layout = [var]
    bridge, instance, me, bv = _local_rename_engine(monkeypatch, fn, var)

    fn.create_user_var(var, var.type, "myvar")  # genuine prior USER definition
    assert fn.is_var_user_defined(var) is True

    restores: list = []
    me._op_local_rename(
        instance.ctx, bv,
        {"op": "local_rename", "function": "f", "variable": "myvar", "new_name": "probe"},
        restores,
    )
    assert var.name == "probe"
    assert me._run_local_restores(instance.ctx, bv, restores) is True
    assert fn.is_var_user_defined(var) is True   # still a user var
    assert var.name == "myvar"


# ===========================================================================
# #630 -- a proto set pins has_user_type, and Binary Ninja exposes NO API to
# clear it (verified live on BN 5.4: set_auto_type and revert_undo_actions both
# leave the flag set, has_user_type has no setter, there is no delete_user_type).
# has_user_type is behaviorally meaningful (once set, analysis will not re-derive
# the signature), so it is NOT value-neutral metadata. The honest contract:
#   * a --preview of a proto set on an AUTO function is REFUSED before any
#     mutation (the view stays pristine) -- a preview that could not be cleanly
#     reverted must not be performed and reported as a clean rollback; and
#   * an INVOLUNTARY rollback of a live batch (a sibling op failed, or an apply
#     failed) that had already pinned has_user_type reports success:false /
#     rolled_back:false, discloses prototype_user_type_residue, and is dirty.
# (Supersedes the earlier #582 "disclose but still report success" behavior,
# which masked a real rollback failure.)
# ===========================================================================

def test_preview_proto_set_on_auto_function_is_refused_pristine(monkeypatch):
    """A --preview of a prototype change on a function with no user type is
    REFUSED before any mutation: BN cannot clear the has_user_type it would pin,
    so the preview could not be cleanly reverted. The view is left PRISTINE --
    no undo transaction is even opened -- rather than applied and reported as a
    false clean rollback (#630)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "f")  # has_user_type False, type "int32_t()"
    bv = _FakeMutationBV(functions=[fn])
    fn.view = bv
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.ctx, "_find_function", lambda _bv, ident: fn)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._mutation(
            "active", True,
            [{"op": "set_prototype", "identifier": "f", "prototype": "uint64_t f(int32_t a)"}],
        )

    assert exc.value.status == "unsupported"
    assert "has_user_type" in exc.value.message
    # The view is untouched: no mutation, no undo transaction opened.
    assert fn.has_user_type is False
    assert str(fn.type) == "int32_t()"
    assert not _has_event(bv, "begin")


def test_unrevertible_preview_prototypes_flags_only_auto_changing_ops(monkeypatch):
    """The preflight flags a proto set that would pin has_user_type on an AUTO
    function, and ONLY that -- a function with a prior user type, and a no-op set
    (requested prototype already the current one), are both safe to preview (#630)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    me = bridge.mutation_engine

    auto_fn = _FakeFunction(0x401000, "auto_fn")            # AUTO, "int32_t()"
    user_fn = _FakeFunction(0x402000, "user_fn")
    user_fn.set_user_type("int32_t()")                     # genuine prior user type
    noop_fn = _FakeFunction(0x403000, "noop_fn")           # AUTO, already "int32_t()"
    bv = _FakeMutationBV(functions=[auto_fn, user_fn, noop_fn])
    for fn in (auto_fn, user_fn, noop_fn):
        fn.view = bv
    monkeypatch.setattr(instance.ctx, "_find_function",
                        lambda _bv, ident: {"auto_fn": auto_fn, "user_fn": user_fn,
                                            "noop_fn": noop_fn}[ident])

    flagged = me._unrevertible_preview_prototypes(
        instance.ctx, bv,
        [
            {"op": "set_prototype", "identifier": "auto_fn", "prototype": "uint64_t auto_fn(int32_t a)"},
            {"op": "set_prototype", "identifier": "user_fn", "prototype": "uint64_t user_fn(int32_t a)"},
            {"op": "set_prototype", "identifier": "noop_fn", "prototype": "int32_t noop_fn()"},
        ],
    )
    assert flagged == ["auto_fn"]


def test_live_verify_fail_with_proto_on_auto_reports_residue_and_fails(monkeypatch):
    """An INVOLUNTARY rollback -- a live batch where a proto-set-on-AUTO applied
    but a sibling op failed verification -- must report success:false /
    rolled_back:false and disclose the unclearable has_user_type residue, never a
    false clean rollback (#630)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "f")  # has_user_type starts False
    bv = _FakeMutationBV(functions=[fn])

    def apply(bv_, op, restores=None):
        if op["op"] == "set_prototype":
            fn.set_user_type("uint64_t f(int32_t a)")  # pins has_user_type True
            restores.append(lambda: fn.set_auto_type("int32_t()"))  # value only; flag stays
            return {"op": "set_prototype", "address": "0x401000",
                    "before_has_user_type": False, "requested": {}}
        return {"op": "rename_symbol", "address": "0x401000", "requested": {}}

    def verify(bv_, result):
        if result["op"] == "set_prototype":
            return {**result, "status": "verified"}
        return {**result, "status": "verification_failed", "message": "rename did not land"}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply, verify=verify)

    result = instance._mutation("active", False, [{"op": "set_prototype"}, {"op": "rename_symbol"}])

    assert fn.has_user_type is True  # BN could not clear it
    assert result["success"] is False
    assert result["rolled_back"] is False
    assert result["prototype_user_type_residue"] is True
    assert "has_user_type" in result["message"]


def test_apply_failure_with_proto_on_auto_discloses_residue(monkeypatch):
    """The apply-failure path (a later op raised during apply) must ALSO disclose
    proto residue and treat it as a failed rollback -- it previously reported
    status 'reverted'/rolled_back True with no disclosure at all (#630)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "f")
    bv = _FakeMutationBV(functions=[fn])

    def apply(bv_, op, restores=None):
        if op["op"] == "boom":
            raise bridge.OperationFailure("unsupported", "nope", requested={})
        fn.set_user_type("uint64_t f(int32_t a)")
        restores.append(lambda: fn.set_auto_type("int32_t()"))
        return {"op": "set_prototype", "address": "0x401000",
                "before_has_user_type": False, "requested": {}}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply)

    result = instance._mutation("active", False, [{"op": "set_prototype"}, {"op": "boom"}])

    assert fn.has_user_type is True
    assert result["success"] is False
    assert result["rolled_back"] is False
    assert result["prototype_user_type_residue"] is True
    assert result["results"][0]["status"] == "rollback_failed"
    assert "has_user_type" in result["message"]


def test_preview_locals_only_batch_reports_no_proto_residue(monkeypatch):
    """Independence from #582: a locals-only preview must not be dragged into a
    proto-residue false-failure -- it reverts cleanly (proves the two fixes are
    separate)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None):
        restores.append(lambda: None)
        return {"op": "local_rename", "requested": {}}

    _mutation_with_stubs(
        monkeypatch, bridge, instance, bv,
        apply=apply,
        verify=lambda bv_, result: {**result, "status": "verified"},
    )
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, r: True)

    result = instance._mutation("active", True, [{"op": "local_rename"}])
    assert result["success"] is True
    assert result["message"] == "Preview verified and reverted."
