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


@pytest.mark.parametrize("control", ["\0", "\t", "\n", "\r", "\x1b", "\x1f", "\x7f", "\x80", "\x85", "\x9f"])
@pytest.mark.parametrize("kind", ["function", "data"])
def test_symbol_rename_control_name_refused_before_resolution(monkeypatch, control, kind):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "parse_record")
    bv = _FakeMutationBV(functions=[fn])

    def unexpected_resolution(*args):
        pytest.fail("invalid symbol name reached target resolution")

    monkeypatch.setattr(instance.ctx, "_resolve_rename_target", unexpected_resolution)
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._op_rename_symbol(bv, {
            "op": "rename_symbol", "kind": kind, "identifier": hex(fn.start),
            "new_name": f"bad{control}name",
        })
    assert exc.value.status == "invalid_request"
    assert fn.name == "parse_record"
    assert bv.events == []


@pytest.mark.parametrize("bad_name", ["", "   ", None, "bad\nname", "bad\0name", "bad\x7fname", "bad\x9fname"])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("preview", [False, True])
def test_symbol_rename_invalid_name_refuses_raw_and_batch_before_writes(
    monkeypatch, bad_name, batch, preview,
):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "parse_record")
    bv = _FakeMutationBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    rename = {"op": "rename_symbol", "identifier": hex(fn.start), "new_name": bad_name}
    # A valid earlier sibling must not apply even temporarily.
    params = {"ops": [
        {"op": "set_comment", "function": hex(fn.start), "comment": "must not land"},
        rename,
    ]} if batch else rename
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._dispatch_on_main(
            "batch_apply" if batch else "rename_symbol", {**params, "preview": preview}, "fixture",
        )
    assert exc.value.status == "invalid_request"
    assert exc.value.requested["new_name"] == bad_name
    if batch:
        assert "operation 1 " in exc.value.message
    assert fn.name == "parse_record"
    assert fn.comment == ""
    assert bv.events == []  # No transaction, analysis, commit, or revert.


@pytest.mark.parametrize("name", ["''", '""', "operator[]", "ns::parse_record", "café", "two words"])
def test_symbol_rename_preserves_printable_quotes_and_punctuation(monkeypatch, name):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "parse_record")
    bv = _FakeBV(functions=[fn])
    op = {"op": "rename_symbol", "identifier": hex(fn.start), "new_name": name}
    result = instance._apply_operation(bv, op)
    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert verified["observed"]["name"] == name
    assert fn.name == name


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
        lambda ctx, bv, op, restores=None, **kwargs: {
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

    result = instance._mutation("active", False, [
        {"op": "rename_symbol", "identifier": "sub_401000", "new_name": "player_update"}])

    assert result["success"] is False
    assert result["committed"] is False
    assert _has_event(bv, "revert")
    assert not _has_event(bv, "commit")


@pytest.mark.parametrize("addrs", [("0x1000",), ("0x1000", "0x2000")])
def test_batch_function_create_reanalysis_runs_under_gate_not_exclusive_lock(monkeypatch, addrs):
    """#780 (residual of #628): _mutation holds the exclusive target lock across the
    whole apply loop, and a function_create op reanalyzed inside itself
    (update_analysis_and_wait) -- so a batch of N creates held the exclusive lock
    across N full analyses while every concurrent read (which takes
    _target_lock.read() and waits on _writer) blocked for all of them. The op's
    reanalysis is deferred to the batch's own post-apply reanalysis, which #628
    already runs under the write gate alone: NO reanalysis happens with the
    exclusive lock held, and the end state is unchanged (both functions
    created, verified and visible; a 1-op batch still reanalyzes).

    #927 P2 adds one gate-only analysis BETWEEN two creates, so the second
    create's interior guard sees the first one's ranges -- the lock
    discipline is the invariant, the count of gate-only analyses is not: a
    1-create batch analyzes once, an N-create batch N times, never exclusively."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    lock = instance._target_lock
    bv = _FakeFunctionCreateBV(
        segments={int(a, 16): _FakeSegment(readable=True, executable=True) for a in addrs},
        memory={int(a, 16): b"\x55\x48\x89\xe5" for a in addrs},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    original = bv.update_analysis_and_wait
    waits: list = []

    def analyze():
        # (exclusive writer held?, write gate held?)
        waits.append((lock._writer, instance._write_gate.locked()))
        original()

    bv.update_analysis_and_wait = analyze

    result = instance._mutation(
        "active", False,
        [{"op": "function_create", "address": a} for a in addrs],
    )

    # The core assertion: no reanalysis at all ran while the exclusive writer lock
    # was held; every analysis the batch ran was gate-only (#628), one per create
    # (#927 P2: the inter-create one plus the post-apply one).
    assert waits == [(False, True)] * len(addrs)
    assert lock._writer is False
    assert "refresh" in bv.events
    # End state unchanged: every op verified and every function live.
    assert result["success"] is True
    assert result["committed"] is True
    assert [r["status"] for r in result["results"]] == ["verified"] * len(addrs)
    for a in addrs:
        fn = bv.get_function_at(int(a, 16))
        assert fn is not None, f"function at {a} missing after the batch"
        assert fn.name == f"sub_{int(a, 16):x}"


def test_batch_function_create_junk_address_still_fails_and_rolls_back(monkeypatch):
    """#780: the #386 code guard moved from the (now reanalysis-free) per-op apply
    to the batch's post-analysis verification, so a forced create on non-code must
    still fail its op, roll the whole batch back, and leave no fabricated function
    behind -- and it must not reintroduce an exclusive-lock reanalysis."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    lock = instance._target_lock
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True),
                  0x3000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5", 0x3000: b"\xff\xff\xff\xff"},
        arch=_FakeArch(name="aarch64", instr_alignment=4),
        instruction_lengths={0x3000: 0},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    original = bv.update_analysis_and_wait
    waits: list = []

    def analyze():
        waits.append((lock._writer, instance._write_gate.locked()))
        original()

    bv.update_analysis_and_wait = analyze

    result = instance._mutation("active", False, [
        {"op": "function_create", "address": "0x1000"},
        {"op": "function_create", "address": "0x3000"},
    ])

    assert waits[0] == (False, True)  # the batch reanalysis is still gate-only
    assert result["success"] is False
    assert result["committed"] is False
    assert result["rolled_back"] is True
    assert {r["address"]: r["status"] for r in result["results"]} == {
        "0x1000": "reverted",             # applied, then rolled back with the batch
        "0x3000": "verification_failed",  # the junk create itself
    }
    assert bv.get_function_at(0x1000) is None  # sibling reverted
    assert bv.get_function_at(0x3000) is None  # fabricated function dropped


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

    def apply(bv_, op, restores=None, **kwargs):
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

    def apply(bv_, op, restores=None, **kwargs):
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


@pytest.mark.parametrize("revert_fault,residue_text", [
    (None, None),
    ("values", "reverting a non-journaled change"),
    ("hosts", "a function a create split is not back"),
    ("proto", "has_user_type override could NOT be cleared"),
])
def test_preview_verification_failure_never_claims_verified_937(
        monkeypatch, revert_fault, residue_text):
    """#937: a preview whose readback FAILED kept its exit/status/rollback
    contract but said "Preview verified and reverted." -- on the clean-revert
    branch and on every incomplete-revert branch alike. The message must name
    the failed verification, and still name which part of the revert did not
    hold."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None, **kwargs):
        restores.append(lambda: None)
        return {"op": "local_rename", "requested": {}}

    _mutation_with_stubs(
        monkeypatch, bridge, instance, bv,
        apply=apply,
        verify=lambda bv_, result: {**result, "status": "verification_failed"},
    )
    monkeypatch.setattr(me, "_run_local_restores",
                        lambda ctx, bv_, restores: revert_fault != "values")
    monkeypatch.setattr(bridge.create_comments, "_split_hosts_at_baseline",
                        lambda bv_, baseline, what: revert_fault != "hosts")
    monkeypatch.setattr(me, "_prototype_user_type_residue",
                        lambda ctx, bv_, results: revert_fault == "proto")

    result = instance._mutation("active", True, [{"op": "local_rename"}])

    assert result["preview"] is True
    assert result["success"] is False
    assert result["committed"] is False
    assert result["rolled_back"] is (revert_fault is None)
    assert result["results"][0]["status"] == "verification_failed"
    assert "verified" not in result["message"].replace("verification failed", "")
    if revert_fault is None:
        assert result["message"] == "Preview verification failed; the change was reverted."
    else:
        assert result["message"].startswith("Preview verification failed, and ")
        assert residue_text in result["message"]


def test_a_name_renamed_earlier_in_the_batch_is_explained_975(monkeypatch):
    """#975: a later op naming a function by the spelling an earlier op renamed
    away failed with the resolver's lexically-close but unrelated suggestions.
    The failure must name the rename. It must NOT fire when the name still
    resolves -- a later rename reused it for another function -- and the real
    failure is a different one: there the hint would point the agent at the
    WRONG function. (A chain through the name is refused before apply, #995.)"""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    names: dict[str, int] = {}

    def resolve(name):
        if name not in names:
            raise bridge.OperationFailure(
                "unsupported", f"Function not found: {name}. Did you mean: sub_4025dc")
        return names[name]

    def apply(bv_, op, restores=None, **kwargs):
        if op["op"] == "rename_symbol":
            before = op["identifier"]
            address = resolve(before)
            del names[before]
            names[op["new_name"]] = address
            return {"op": "rename_symbol", "kind": "function", "address": hex(address),
                    "before_name": before, "new_name": op["new_name"], "requested": {}}
        resolve(op["function"])
        if op["op"] == "local_rename":
            raise bridge.OperationFailure(
                "unsupported", f"Variable not found: {op['variable']}")
        return {"op": op["op"], "requested": {}}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply)

    def run(*ops):
        names.clear()
        names.update({"sub_401d10": 0x401d10, "sub_401e20": 0x401e20})
        return instance._mutation("active", True, list(ops))

    def rename(old, new):
        return {"op": "rename_symbol", "identifier": old, "new_name": new}

    comment = {"op": "set_comment", "function": "sub_401d10", "comment": "entry"}

    row = run(rename("sub_401d10", "handle_request"), comment)["results"][1]
    assert row["status"] == "unsupported"
    assert "Did you mean" not in row["message"]
    assert row["message"].startswith(
        "sub_401d10 not found: operation 0 of this batch renamed sub_401d10 "
        "(at 0x401d10) to handle_request.")

    # A miss on a name no rename touched keeps the resolver's message.
    unrelated = {"op": "set_comment", "function": "sub_missing", "comment": "entry"}
    row = run(rename("sub_401d10", "handle_request"), unrelated)["results"][1]
    assert row["message"] == "Function not found: sub_missing. Did you mean: sub_4025dc"

    # Reuse: `sub_401d10` moved off 0x401d10 and was then given to 0x401e20. It
    # resolves -- to 0x401e20 -- so no hint may send the agent back to 0x401d10;
    # the real failure is the variable, and that is what must be reported.
    bad_local = {"op": "local_rename", "function": "sub_401d10",
                 "variable": "no_such_var", "new_name": "x"}
    row = run(rename("sub_401d10", "alpha"), rename("sub_401e20", "sub_401d10"),
              bad_local)["results"][2]
    assert row["message"] == "Variable not found: no_such_var"


def _rename_rows_batch(monkeypatch, landed):
    """Run a live batch of renames whose apply LANDED where *landed* says:
    ``[(address, before_name, new_name), ...]``, one per op -- the address the
    real resolver chose, which is all the #995 chain check reads."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    rows = iter(landed)

    def apply(bv_, op, restores=None, **kwargs):
        address, before, new = next(rows)
        return {"op": "rename_symbol", "kind": "function", "address": hex(address),
                "before_name": before, "new_name": new, "requested": dict(op)}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply,
                         verify=lambda bv_, result: {**result, "status": "verified"})
    ops = [{"op": "rename_symbol", "identifier": ident, "new_name": new}
           for ident, (_, _, new) in zip(("add", "foo", "both"), landed)]
    return instance._mutation("active", False, ops), bv


@pytest.mark.parametrize("landed,failed_index,first", [
    # `add -> foo`, then `foo -> bar` (or its address, a case variant, a data
    # rename at the function's address): the second landed on the same symbol.
    ([(0x401129, "add", "foo"), (0x401129, "foo", "bar")], 1, 0),
    # An unrelated rename between the two does not hide the chain.
    ([(0x401129, "add", "both"), (0x404010, "datum", "both"),
      (0x401129, "both", "lastadd")], 2, 0),
])
def test_a_rename_landing_on_a_symbol_already_renamed_fails_the_batch_995(
        monkeypatch, landed, failed_index, first):
    """#995: renaming one symbol twice in a batch always failed verification on
    the first rename (every op is verified against the END state) and rolled
    back as `Live rename verification failed at <addr>`, which reads as a BN
    problem. The second rename is now refused where it lands, naming both ops
    and the one rename that would do, and the batch is rolled back."""
    result, bv = _rename_rows_batch(monkeypatch, landed)

    row = result["results"][failed_index]
    assert row["status"] == "invalid_request"
    assert row["requested"]["conflicting_op_indices"] == [first, failed_index]
    assert f"renames the symbol at {hex(landed[first][0])} again" in row["message"]
    assert (f"Rename the symbol at {hex(landed[first][0])} once, by that address, to "
            f"{landed[failed_index][2]}") in row["message"]
    assert result["rolled_back"] is True and result["success"] is False
    assert all(r["status"] == "reverted" for r in result["results"][:failed_index])
    assert _has_event(bv, "revert")


@pytest.mark.parametrize("landed", [
    # The review's false refusals: the later rename LANDED on a different symbol
    # (a case-folded function beside a data symbol named by the earlier rename;
    # a data symbol holding a name a function rename also took; an existing
    # function sharing the name). None of them is a chain.
    [(0x40401c, "datum", "ADD"), (0x401136, "add", "newadd")],
    [(0x401129, "add", "__dso_handle"), (0x404008, "__dso_handle", "marker")],
    [(0x401300, "x", "foo"), (0x401200, "foo", "bar")],
    # Same symbol, same final name: both renames verify, so nothing to refuse.
    [(0x401129, "add", "foo"), (0x401129, "foo", "foo")],
])
def test_renames_that_land_on_different_symbols_are_not_refused_995(monkeypatch, landed):
    """#995 review rounds 1-2: an up-front simulation of the resolver refused
    batches like these and advised renaming the wrong symbol. The check reads
    only where each rename actually landed, so they apply."""
    result, _ = _rename_rows_batch(monkeypatch, landed)

    assert result["success"] is True, result["results"]
    assert [r["status"] for r in result["results"]] == ["verified"] * len(landed)


def test_preview_drift_restore_failure_is_not_success(monkeypatch):
    """If reverting BN's propagation onto aliased siblings (the var-drift
    restore) fails, the preview left the view modified and must report
    success:false / rolled_back:false (#88)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None, **kwargs):
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


def test_apply_failure_and_exception_revert_settle_before_var_drift_restore(monkeypatch):
    """#657: the apply-failure and generic-exception revert paths must settle the
    view before reading var drift, exactly like the preview path (#581). A stale
    read sees phantom drift and re-pins an AUTO local as USER. Observed
    behaviourally: the settle must run BEFORE _restore_local_var_drift on both
    paths, and be skipped when the snapshot is empty (nothing to drift-check =
    no pointless reanalysis)."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine

    def drive(*, apply_exc=None, verify_exc=None, snapshots):
        instance = bridge.BinaryNinjaBridge()
        bv = _FakeMutationBV()
        order: list[str] = []

        def apply(bv_, op, restores=None, **kwargs):
            if apply_exc is not None:
                raise apply_exc
            return {"op": "local_rename", "requested": {}}

        def verify(bv_, result):
            if verify_exc is not None:
                raise verify_exc
            return {**result, "status": "verified"}

        # Record the two operations whose relative order is the contract.
        monkeypatch.setattr(bv, "update_analysis_and_wait", lambda: order.append("settle"))
        monkeypatch.setattr(me, "_capture_local_var_snapshots", lambda ctx, bv_, fns: snapshots)
        monkeypatch.setattr(me, "_run_local_restores", lambda ctx, bv_, restores: True)
        monkeypatch.setattr(
            me, "_restore_local_var_drift", lambda ctx, bv_, snap: order.append("drift") or True
        )
        _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply, verify=verify)
        return instance, order

    snapshot = {0x1: {1: ("a", "int32_t")}}

    # Apply-failure path: settle, then read drift.
    instance, order = drive(
        apply_exc=me.OperationFailure("unsupported", "nope", requested={}), snapshots=snapshot
    )
    result = instance._mutation("active", False, [{"op": "local_rename"}])
    assert result["rolled_back"] is True
    assert order == ["settle", "drift"]

    # Generic-exception path (here: post-apply verification raises): the settle
    # must immediately precede the drift read, and the error still propagates.
    instance, order = drive(verify_exc=ValueError("boom"), snapshots=snapshot)
    with pytest.raises(ValueError, match="boom"):
        instance._mutation("active", False, [{"op": "local_rename"}])
    assert order[-2:] == ["settle", "drift"]

    # Empty snapshot -> no drift to read, so the revert path skips the
    # reanalysis entirely (same guard as the preview path's settle).
    instance, order = drive(
        apply_exc=me.OperationFailure("unsupported", "nope", requested={}), snapshots={}
    )
    instance._mutation("active", False, [{"op": "local_rename"}])
    assert order == ["drift"]


def test_preview_with_successful_restore_still_succeeds(monkeypatch):
    """The restored-success coupling must not regress the normal preview path."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None, **kwargs):
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

    def apply(bv_, op, restores=None, **kwargs):
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

    def apply(bv_, op, restores=None, **kwargs):
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


def test_op_set_prototype_hands_bn_the_parsed_type_999(monkeypatch):
    """#999 item 1: the parsed Type goes to set_user_type, never `str(Type)`.
    BN renders a `::`-qualified type without the backticks it needs to parse
    (`struct ns::Cls*`), so re-parsing the string failed with an internal
    SyntaxError on a prototype that had just parsed fine."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    parsed = _FakeType("void*(struct ns::Cls* self, void* p)", type_class="FunctionTypeClass")

    class _SetterFunction(_FakeFunction):
        def __init__(self):
            super().__init__(0x40116A, "_ZN2ns3Cls6methodEPv", "void* __fastcall(void* arg1)")
            self.user_type_calls = []

        def set_user_type(self, value):
            if isinstance(value, str):
                raise SyntaxError("error: string.hpp:1:7 expected unqualified-id")
            self.user_type_calls.append(value)
            super().set_user_type(str(value))

    class _PrototypeBV(_FakeBV):
        def parse_type_string(self, declaration):
            return parsed, None

    fn = _SetterFunction()
    bv = _PrototypeBV(functions=[fn])

    result = instance._op_set_prototype(
        bv,
        {"op": "set_prototype", "identifier": "_ZN2ns3Cls6methodEPv",
         "prototype": "void* method(struct `ns::Cls`* self, void* p)"},
    )

    assert fn.user_type_calls == [parsed]
    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert verified["observed"]["prototype"] == "void*(struct ns::Cls* self, void* p)"


def test_apply_operation_user_error_message_has_no_class_name(monkeypatch):
    """A handler raising a user-facing RuntimeError (e.g. a mistyped function
    name -> 'Function not found') must surface a clean, actionable message --
    not 'unsupported: RuntimeError: ...' that reads like an internal crash
    (#122)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    me = bridge.mutation_engine

    # #825 item 1: raise the way a REAL handler does -- through bridge code.
    # A bare `raise` in this file is attributed to a library and correctly
    # triaged with an `internal error:` prefix, so it could no longer stand
    # in for the handler this test is about. Delegating to a genuine bridge
    # rejection is what an actual `_op_*` does on bad input.
    from bn_agent_bridge import _shared
    user_facing = "'ghost' is not a valid address; expected a decimal or 0x-prefixed hex value"

    def boom_user(ctx, bv, op):
        _shared._parse_address("ghost")

    monkeypatch.setattr(me, "_op_set_comment", boom_user)
    bv = _FakeBV()

    with pytest.raises(bridge.OperationFailure) as excinfo:
        instance._apply_operation(bv, {"op": "set_comment", "comment": "x", "function": "ghost"})

    assert excinfo.value.status == "unsupported"
    assert excinfo.value.message == user_facing
    assert "ValueError" not in excinfo.value.message
    assert "internal error" not in excinfo.value.message


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

    # The alias is normalized for every struct_field_* kind. #825: the helper
    # returns a normalized COPY, so the contract is read off the RETURN value --
    # asserting it on the input dict was asserting the in-place mechanism, which
    # is the externally-observable side effect #825 removes.
    for kind in ("struct_field_set", "struct_field_rename", "struct_field_delete"):
        op = {"op": kind, "type_name": "Elf64_Sym"}
        assert me._normalize_struct_alias(op)["struct_name"] == "Elf64_Sym", kind
        # ...and the caller's own dict is left exactly as it was sent.
        assert "struct_name" not in op, kind

    # an explicit struct_name always wins (alias never clobbers it)
    op = {"op": "struct_field_rename", "struct_name": "A", "type_name": "B"}
    assert me._normalize_struct_alias(op)["struct_name"] == "A"

    # non-struct ops are left untouched, and returned as the same object since
    # there is nothing to normalize (no needless copy on the common path)
    op = {"op": "rename_symbol", "type_name": "X"}
    assert me._normalize_struct_alias(op) is op
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


def test_batch_invalid_op_is_rejected_before_anything_is_applied(monkeypatch):
    """A manifest whose 2nd op is malformed (missing a required field) is rejected
    UP FRONT as invalid_request, naming the op index and the field, with nothing
    applied and nothing to roll back (#650, superseding #173's rollback contract:
    one agent lost 12 good ops to a field name)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeCommentMutationBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._mutation(
            "active",
            False,
            [
                {"op": "set_comment", "address": "0x1000", "comment": "first op applied"},
                {"op": "set_comment", "address": "0x2000"},  # missing required 'comment'
            ],
        )

    assert exc.value.status == "invalid_request"
    assert "operation 1" in exc.value.message
    assert "comment" in exc.value.message
    # No transaction was ever opened, so there is no partial apply to revert.
    assert bv.get_comment_at(0x1000) == ""
    assert not _has_event(bv, "begin")
    assert not _has_event(bv, "commit")


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


@pytest.mark.parametrize("bad_name", ["", "   ", "\t", None])
def test_op_local_rename_rejects_empty_new_name(monkeypatch, bad_name):
    """Mirrors test_op_rename_symbol_rejects_empty_new_name (#605): an empty,
    whitespace-only, or null new_name must be rejected before the variable is
    touched -- str(None) must not slip through as the literal "None"."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    var = _FakeVariable(name="v1", storage=8, var_type="int32_t", identifier=1)
    fn = _FakeFunction(0x401000, "f")
    fn.stack_layout = [var]
    bv = types.SimpleNamespace(get_function_at=lambda addr: fn)
    monkeypatch.setattr(instance.ctx, "_find_function", lambda _bv, ident: fn)

    op = {"op": "local_rename", "function": "f", "variable": "v1", "new_name": bad_name}
    with pytest.raises(bridge.mutation_engine.OperationFailure) as excinfo:
        bridge.mutation_engine._op_local_rename(instance.ctx, bv, op)
    assert excinfo.value.status == "invalid_request"
    assert "non-empty" in excinfo.value.message
    assert var.name == "v1"  # rejected before any mutation landed


@pytest.mark.parametrize("bad_name", ["", "   ", "\t", None])
def test_op_struct_field_rename_rejects_empty_new_name(monkeypatch, bad_name):
    """Mirrors test_op_rename_symbol_rejects_empty_new_name (#605) for struct
    field rename."""
    bridge, instance, builder = _struct_instance(monkeypatch, [_FakeStructMember(0, "a")])
    with pytest.raises(bridge.OperationFailure) as excinfo:
        instance._op_struct_field_rename(None, {"struct_name": "S", "old_name": "a", "new_name": bad_name})
    assert excinfo.value.status == "invalid_request"
    assert "non-empty" in excinfo.value.message
    assert builder.members[0].name == "a"  # rejected before any mutation landed


def test_op_struct_field_rename_rejects_duplicate_name(monkeypatch):
    """Renaming a field to a name already held by ANOTHER member must be
    rejected (#666): otherwise the rename "verifies" cleanly and a later
    struct_field_delete resolving that name picks an arbitrary member."""
    bridge, instance, builder = _struct_instance(
        monkeypatch, [_FakeStructMember(0, "a"), _FakeStructMember(8, "b")])
    with pytest.raises(bridge.OperationFailure) as excinfo:
        instance._op_struct_field_rename(None, {"struct_name": "S", "old_name": "a", "new_name": "b"})
    assert excinfo.value.status == "invalid_request"
    assert "already has a member named" in excinfo.value.message
    # neither member was touched
    assert [(m.offset, m.name) for m in builder.members] == [(0, "a"), (8, "b")]


def test_op_struct_field_rename_unique_name_still_succeeds(monkeypatch):
    """Regression guard: a unique, non-empty rename still succeeds and
    verifies (not over-corrected by the #605/#666 guards)."""
    bridge, instance, builder = _struct_instance(
        monkeypatch, [_FakeStructMember(0, "a"), _FakeStructMember(8, "b")])
    res = instance._op_struct_field_rename(None, {"struct_name": "S", "old_name": "a", "new_name": "aa"})
    assert res["old_name"] == "a"
    assert res["new_name"] == "aa"
    assert builder.members[0].name == "aa"


def test_struct_field_offset_delete_targets_right_field_on_duplicate_names(monkeypatch):
    # Two members share the name 'dup' at offsets 0x0 and 0x8. Deleting by
    # offset 0x8 must remove the SECOND one. A name round-trip would resolve via
    # index_by_name's first match (0x0) and silently delete the wrong field (#25).
    bridge, instance, builder = _struct_instance(
        monkeypatch, [_FakeStructMember(0, "dup"), _FakeStructMember(8, "dup")])
    res = instance._op_struct_field_delete(None, {"struct_name": "S", "field_name": "0x8"})
    assert res["field_name"] == "dup"
    assert [(m.offset, m.name) for m in builder.members] == [(0, "dup")]  # 0x8 gone, 0x0 kept


def test_struct_field_delete_by_ambiguous_name_is_invalid_request(monkeypatch):
    # #666: a name locator that matches more than one member must be rejected
    # as ambiguous rather than silently deleting whichever member the
    # first-match scan happens to hit -- the exact "coin flip that still
    # verifies" scenario the issue describes.
    bridge, instance, builder = _struct_instance(
        monkeypatch, [_FakeStructMember(0, "dup"), _FakeStructMember(8, "dup")])
    with pytest.raises(bridge.OperationFailure) as excinfo:
        instance._op_struct_field_delete(None, {"struct_name": "S", "field_name": "dup"})
    assert excinfo.value.status == "invalid_request"
    assert "0x0" in excinfo.value.message
    assert "0x8" in excinfo.value.message
    # nothing was deleted
    assert [(m.offset, m.name) for m in builder.members] == [(0, "dup"), (8, "dup")]


def test_struct_field_rename_by_ambiguous_name_is_invalid_request(monkeypatch):
    # Same ambiguity, via the rename op's old_name locator.
    bridge, instance, builder = _struct_instance(
        monkeypatch, [_FakeStructMember(0, "dup"), _FakeStructMember(8, "dup")])
    with pytest.raises(bridge.OperationFailure) as excinfo:
        instance._op_struct_field_rename(None, {"struct_name": "S", "old_name": "dup", "new_name": "renamed"})
    assert excinfo.value.status == "invalid_request"
    assert "0x0" in excinfo.value.message
    assert "0x8" in excinfo.value.message
    # neither member was renamed
    assert [(m.offset, m.name) for m in builder.members] == [(0, "dup"), (8, "dup")]


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


def test_struct_field_set_overwrite_narrower_retype_reusing_removed_name_succeeds(monkeypatch):
    """#666 round 3 follow-up: `add_member_at_offset(overwrite=True)` removes
    EVERY member overlapping the new byte range, so retyping a wide field
    narrower at a different offset but the SAME name creates no duplicate --
    the guard must not false-reject it. `wide` occupies [0x0, 0x8); the new
    4-byte field at 0x4 overlaps it (removes it), and the requested name
    `wide` is exactly the member being removed."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    builder = _AddableStructBuilder()
    wide = types.SimpleNamespace(offset=0, name="wide", type=types.SimpleNamespace(width=8))
    occupied_type = types.SimpleNamespace(members=[wide])

    class _BV:
        def parse_type_string(self, s):
            return types.SimpleNamespace(width=4), None   # a 4-byte field

        def get_type_by_name(self, n):
            return occupied_type

    monkeypatch.setattr(bridge.mutation_engine, "_struct_builder", lambda ctx, bv, name: ("S", builder))
    monkeypatch.setattr(bridge.mutation_engine, "_commit_struct_builder", lambda *a, **k: None)
    res = instance._op_struct_field_set(_BV(), {
        "struct_name": "S", "offset": "0x4", "field_name": "wide",
        "field_type": "int32_t", "overwrite_existing": True})
    assert res["before_member"] is None       # 0x4 itself wasn't a member start
    assert builder.added and builder.added[0] == ("wide", 4, True)


def test_struct_field_set_overwrite_minting_surviving_duplicate_refused(monkeypatch):
    """#666 round 3 follow-up, the genuine case: `a` at 0x0 and `b` at 0x8 do
    NOT overlap a 4-byte write at 0x0 (only `a` is removed), so naming the new
    field `b` would leave TWO members named `b` -- must be refused, and never
    reach add_member_at_offset."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    builder = _AddableStructBuilder()
    a = types.SimpleNamespace(offset=0, name="a", type=types.SimpleNamespace(width=4))
    b = types.SimpleNamespace(offset=8, name="b", type=types.SimpleNamespace(width=4))
    occupied_type = types.SimpleNamespace(members=[a, b])

    class _BV:
        def parse_type_string(self, s):
            return types.SimpleNamespace(width=4), None

        def get_type_by_name(self, n):
            return occupied_type

    monkeypatch.setattr(bridge.mutation_engine, "_struct_builder", lambda ctx, bv, name: ("S", builder))
    monkeypatch.setattr(bridge.mutation_engine, "_commit_struct_builder", lambda *a, **k: None)
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._op_struct_field_set(_BV(), {
            "struct_name": "S", "offset": "0x0", "field_name": "b",
            "field_type": "int32_t", "overwrite_existing": True})
    assert exc.value.status == "invalid_request"
    assert "b" in str(exc.value)
    assert "0x8" in str(exc.value)             # names the surviving member's offset
    assert builder.added == []                 # never reached add_member_at_offset


def test_struct_field_set_retype_under_own_name_succeeds_with_fresh_member_objects(monkeypatch):
    """#961: BN hands out NEW member objects on every `.members` access, so a
    guard that matched the members a write removes by `id()` across two reads
    refused a same-name retype (`vtable` at 0x0 -> `void**`) as a duplicate of
    itself. The type below mints fresh members per access, as BN does."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    builder = _AddableStructBuilder()

    class _FreshMembersType:
        @property
        def members(self):
            return [types.SimpleNamespace(offset=0, name="vtable", type=types.SimpleNamespace(width=8)),
                    types.SimpleNamespace(offset=8, name="field", type=types.SimpleNamespace(width=4))]

    class _BV:
        def parse_type_string(self, s):
            return types.SimpleNamespace(width=8), None

        def get_type_by_name(self, n):
            return _FreshMembersType()

    monkeypatch.setattr(bridge.mutation_engine, "_struct_builder", lambda ctx, bv, name: ("Probe", builder))
    monkeypatch.setattr(bridge.mutation_engine, "_commit_struct_builder", lambda *a, **k: None)
    res = instance._op_struct_field_set(_BV(), {
        "struct_name": "Probe", "offset": "0x0", "field_name": "vtable",
        "field_type": "int64_t", "overwrite_existing": True})
    assert res["before_member"]["field_name"] == "vtable"
    assert builder.added and builder.added[0] == ("vtable", 0, True)
    # The surviving member still guards its name.
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._op_struct_field_set(_BV(), {
            "struct_name": "Probe", "offset": "0x0", "field_name": "field",
            "field_type": "int64_t", "overwrite_existing": True})
    assert "0x8" in str(exc.value)



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
    monkeypatch.setattr(me, "_apply_operation", lambda ctx, b, op, restores=None, **kwargs: {"op": op.get("op")})
    monkeypatch.setattr(me, "_verify_operation", lambda ctx, b, result: {**result, "status": "verified"})
    monkeypatch.setattr(me, "_annotate_operation_results", lambda ctx, results, type_diffs: results)

    result = instance._mutation("active", False, [
        {"op": "types_declare", "declaration": "struct ep { int a; };"},
        {"op": "set_prototype", "identifier": "handler", "prototype": "void handler()"}])

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

    def apply(bv_, op, restores=None, **kwargs):
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

    def apply(bv_, op, restores=None, **kwargs):
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

    def apply(bv_, op, restores=None, **kwargs):
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

    def apply(bv_, op, restores=None, **kwargs):
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

    def apply(bv_, op, restores=None, **kwargs):
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

    def apply(bv_, op, restores=None, **kwargs):
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

    def apply(bv_, op, restores=None, **kwargs):
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


# ===========================================================================
# #656 -- go_rename marks the view dirty when applied renames survive a
# failed rollback, not just on a clean commit.
# ===========================================================================


class _GoRenameFn:
    def __init__(self, name):
        self.name = name


class _GoRenameRollbackFailsBV:
    """A single function whose rename applies cleanly but whose rollback
    lookup (the SECOND get_function_at call for that address) reports the
    function missing, so _rollback_go_renames returns False."""

    def __init__(self, fn):
        self._fn = fn
        self._calls = 0

    def get_function_at(self, addr):
        self._calls += 1
        if self._calls > 1:
            return None
        return self._fn


def test_go_rename_marks_dirty_when_preview_rollback_fails(monkeypatch):
    """A preview go_rename applies a rename, then its rollback FAILS -- the
    view is left with a live, uncommitted rename. That must mark the view
    dirty so `bn close` warns, even though committed/verified_count gate is
    False on a preview (#656)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _GoRenameFn("sub_401000")
    bv = _GoRenameRollbackFailsBV(fn)
    marked: list[object] = []
    monkeypatch.setattr(instance.targets, "mark_dirty", lambda b: marked.append(b))

    candidate = {"address": 0x401000, "before_name": "sub_401000", "new_name": "main.foo"}
    result = instance._apply_go_renames_chunked(
        bv, [candidate], preview=True, skipped_user_named=0, defined_count=1
    )

    assert result["rolled_back"] is False
    assert fn.name == "main.foo"  # the rename really is still live
    assert marked == [bv]


def test_go_rename_not_dirty_when_preview_rollback_succeeds(monkeypatch):
    """The negative control: a clean preview rollback must NOT mark dirty."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _GoRenameFn("sub_401000")

    class _CleanRollbackBV:
        def get_function_at(self, addr):
            return fn

    bv = _CleanRollbackBV()
    marked: list[object] = []
    monkeypatch.setattr(instance.targets, "mark_dirty", lambda b: marked.append(b))

    candidate = {"address": 0x401000, "before_name": "sub_401000", "new_name": "main.foo"}
    result = instance._apply_go_renames_chunked(
        bv, [candidate], preview=True, skipped_user_named=0, defined_count=1
    )

    assert result["rolled_back"] is True
    assert fn.name == "sub_401000"  # reverted
    assert marked == []


class _GoRenameCancelledRollbackFailsBV:
    """The function is already gone by the time the cancellation path's own
    rollback looks it up, so _rollback_go_renames reports failure on its
    very first (and only, for this call) get_function_at lookup -- unlike
    _GoRenameRollbackFailsBV, which only fails from the SECOND call onward
    and so rolls back cleanly when invoked directly (not preceded by an
    apply-loop lookup)."""

    def get_function_at(self, addr):
        return None


def test_go_rename_cancelled_marks_dirty_when_rollback_fails(monkeypatch):
    """A cancelled go_rename whose rollback FAILS must mark the view dirty
    before raising -- otherwise `bn close` sees unsaved=false over live,
    un-rolled-back renames (#656, the cancellation path)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _GoRenameCancelledRollbackFailsBV()
    marked: list[object] = []
    monkeypatch.setattr(instance.targets, "mark_dirty", lambda b: marked.append(b))

    applied = [{"address": 0x401000, "before_name": "sub_401000", "new_name": "main.foo"}]

    with pytest.raises(RuntimeError, match="rollback failed"):
        instance._go_rename_cancelled(bv, applied)

    assert marked == [bv]


def test_go_rename_cancelled_not_dirty_when_rollback_succeeds(monkeypatch):
    """Negative control: a cancelled go_rename whose rollback SUCCEEDS must
    not mark dirty."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _GoRenameFn("sub_401000")

    class _CleanRollbackBV:
        def get_function_at(self, addr):
            return fn

    bv = _CleanRollbackBV()
    marked: list[object] = []
    monkeypatch.setattr(instance.targets, "mark_dirty", lambda b: marked.append(b))

    applied = [{"address": 0x401000, "before_name": "sub_401000", "new_name": "main.foo"}]

    with pytest.raises(RuntimeError) as exc_info:
        instance._go_rename_cancelled(bv, applied)
    assert "rollback failed" not in str(exc_info.value)
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


def test_end_to_end_batch_with_invalid_local_rename_leaves_view_untouched(monkeypatch):
    """#605 acceptance criterion 5: a batch containing an invalid `local_rename`
    after a valid one is atomic. Driven through the REAL _mutation path (real
    _apply_operation dispatch to _op_local_rename, real restore replay, real
    undo revert) -- not the stubbed-apply helper, which would bypass the very
    validation this test defends. The first op actually renames a variable;
    the second's blank new_name is rejected by _require_nonempty_name only
    AFTER the first has applied, so the batch must roll the first back too."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    me = bridge.mutation_engine

    var8 = _FakeVariable(name="var_8", storage=-8, var_type="int32_t", identifier=10)
    var12 = _FakeVariable(name="var_c", storage=-12, var_type="int32_t", identifier=11)
    fn = _FakeFunction(0x401000, "f")
    fn.stack_layout = [var8, var12]
    bv = _FakeMutationBV(functions=[fn])
    fn.view = bv

    marked: list = []
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "mark_dirty", lambda b: marked.append(b))
    # Wire the seam/peers so REAL _op_local_rename + restores + drift run; only
    # the snapshot/diff machinery (irrelevant to this assertion) is stubbed out.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.ctx, "_find_function", lambda _bv, ident: fn)
    monkeypatch.setattr(me, "_guess_affected_functions", lambda ctx, b, ops: [fn])
    monkeypatch.setattr(me, "_capture_function_snapshots", lambda ctx, b, fns: {})
    monkeypatch.setattr(me, "_capture_type_snapshots", lambda ctx, b, ops: {})
    monkeypatch.setattr(me, "_diff_snapshots", lambda ctx, before, after: [])
    monkeypatch.setattr(me, "_diff_type_snapshots", lambda ctx, before, after: [])
    by_selector = {"var_8": (var8, False), "var_c": (var12, False)}
    monkeypatch.setattr(bridge.vars_mod, "_find_variable_selector",
                        lambda _f, sel: by_selector[sel])
    monkeypatch.setattr(bridge.vars_mod, "_local_id", lambda _f, _v, is_parameter: "lid")
    by_storage = {int(var8.storage): var8, int(var12.storage): var12}
    monkeypatch.setattr(me, "_find_var_for_restore",
                        lambda ctx, _f, identifier, storage, is_parameter: by_storage[int(storage)])

    result = instance._mutation(
        "active", False,
        [
            {"op": "local_rename", "function": "f", "variable": "var_8", "new_name": "renamed_ok"},
            {"op": "local_rename", "function": "f", "variable": "var_c", "new_name": ""},
        ],
    )

    assert result["success"] is False
    assert result["committed"] is False
    assert result["rolled_back"] is True
    assert result["results"][-1]["status"] == "invalid_request"
    # The view is actually unchanged: the first op's rename was rolled back,
    # not just reported as failed -- the batch is provably atomic.
    assert var8.name == "var_8"
    assert fn.is_var_user_defined(var8) is False
    assert var12.name == "var_c"
    assert marked == []


def test_end_to_end_propagated_auto_sibling_drift_repaired_by_mutation(monkeypatch):
    """#630 round 2, FINDING 3: a REAL propagated AUTO-sibling override, created
    during a local-rename preview driven through the WHOLE _mutation path
    (apply -> verify -> undo -> per-op restore -> drift mop-up), must be un-pinned
    by _restore_local_var_drift's delete_user_var. Models BN 5.4: naming a stack
    var ALSO pins the aliased register sibling (USER propagation), and that
    propagation is NON-journaled -- it survives revert_undo_actions -- so the
    per-op restore (which only touches the target) leaves it, and ONLY the drift
    mop-up can drop it. Reverting that mop-up's delete_user_var to create_user_var
    re-pins the sibling and FAILS this end-to-end test (not just the helper one)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    me = bridge.mutation_engine

    target = _FakeVariable(name="var_8", storage=-8, var_type="int32_t", identifier=10)
    sibling = _FakeVariable(name="r2", storage=2, var_type="int32_t", identifier=20,
                            source_type="RegisterVariableSourceType")
    fn = _FakeFunction(0x401000, "f")
    fn.stack_layout = [target]
    fn.hlil = types.SimpleNamespace(vars=[sibling])
    bv = _FakeMutationBV(functions=[fn])
    fn.view = bv

    # Model BN's aliased USER propagation as NON-journaled: create_user_var on the
    # stack target ALSO pins the aliased register sibling, and revert_undo_actions
    # does NOT clear that propagated pin (so the drift mop-up is the only thing
    # that can). real_create is the ORIGINAL bound method (no recursion).
    real_create = fn.create_user_var
    propagated = {"hit": False}

    def propagating_create_user_var(var, var_type, name, ignore_disjoint_uses=False):
        real_create(var, var_type, name)
        if var is target:
            real_create(sibling, sibling.type, name + "_1")  # aliased propagation
            propagated["hit"] = True
    fn.create_user_var = propagating_create_user_var

    real_restore = fn.provenance_restore

    def restore_preserving_propagation(snapshot):
        real_restore(snapshot)                     # journaled state: both AUTO again
        if propagated["hit"]:
            real_create(sibling, sibling.type, "probe_1")  # non-journaled: survives undo
    fn.provenance_restore = restore_preserving_propagation

    marked: list = []
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "mark_dirty", lambda b: marked.append(b))
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.ctx, "_find_function", lambda _bv, ident: fn)
    monkeypatch.setattr(me, "_guess_affected_functions", lambda ctx, b, ops: [fn])
    monkeypatch.setattr(me, "_capture_function_snapshots", lambda ctx, b, fns: {})
    monkeypatch.setattr(me, "_capture_type_snapshots", lambda ctx, b, ops: {})
    monkeypatch.setattr(me, "_diff_snapshots", lambda ctx, before, after: [])
    monkeypatch.setattr(me, "_diff_type_snapshots", lambda ctx, before, after: [])
    monkeypatch.setattr(bridge.vars_mod, "_find_variable_selector", lambda _f, sel: (target, False))
    monkeypatch.setattr(bridge.vars_mod, "_local_id", lambda _f, _v, is_parameter: "lid")
    monkeypatch.setattr(me, "_find_var_for_restore",
                        lambda ctx, _f, identifier, storage, is_parameter: target)
    monkeypatch.setattr(me, "_verify_operation",
                        lambda ctx, b, result: {**result, "status": "verified"})

    result = instance._mutation(
        "active", True,
        [{"op": "local_rename", "function": "f", "variable": "var_8", "new_name": "probe"}],
    )

    assert result["success"] is True
    assert result["rolled_back"] is True
    # The propagated sibling actually drifted through the real path and the drift
    # mop-up repaired it: un-pinned back to AUTO with its re-derived name.
    assert propagated["hit"] is True
    assert fn.is_var_user_defined(sibling) is False
    assert sibling.name == "r2"
    # Target also back to AUTO, and a clean preview does not dirty the view.
    assert fn.is_var_user_defined(target) is False
    assert target.name == "var_8"
    assert marked == []


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
    assert flagged == [(0, "auto_fn")]


def _skip_preview_fixture(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    auto_fn = _FakeFunction(0x4011AA, "touch")              # AUTO, "int32_t()"
    other_fn = _FakeFunction(0x401200, "other")
    bv = _FakeMutationBV(functions=[auto_fn, other_fn])
    for fn in (auto_fn, other_fn):
        fn.view = bv
    applied: list[dict] = []

    def apply(bv_, op, restores=None, **kwargs):
        applied.append(op)
        return {"op": op["op"], "requested": dict(op)}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply,
                         verify=lambda bv_, result: {**result, "status": "verified"})
    lookup = {"touch": auto_fn, "0x4011aa": auto_fn, "other": other_fn}

    def find(_bv, ident, **kwargs):
        if ident not in lookup:
            raise bridge.OperationFailure("unsupported", f"Function not found: {ident}")
        return lookup[ident]

    monkeypatch.setattr(instance.ctx, "_find_function", find)
    monkeypatch.setattr(instance.ctx, "_resolve_rename_target",
                        lambda _bv, ident, kind: {"kind": "function", "address": 0x401200,
                                                  "before_name": "other"})
    return bridge, instance, bv, auto_fn, applied


def test_preview_skips_an_unrevertible_prototype_and_its_dependents_999(monkeypatch):
    """#999 item 5: one proto set on an AUTO function used to refuse the whole
    --preview, so the comments beside it could not be previewed either. It is
    now left out with the local op on the same function, the rest is previewed,
    and the result names what was skipped -- the skipped ops never touch the
    view, so has_user_type is never pinned."""
    bridge, instance, bv, auto_fn, applied = _skip_preview_fixture(monkeypatch)
    ops = [
        {"op": "set_prototype", "identifier": "0x4011aa", "prototype": "int touch(void* b)"},
        {"op": "local_rename", "function": "touch", "variable": "arg1", "new_name": "big"},
        {"op": "set_comment", "address": "0x4011aa", "comment": "hello"},
        {"op": "local_rename", "function": "other", "variable": "arg1", "new_name": "x"},
    ]

    result = instance._mutation("active", True, ops)

    assert applied == [ops[2], ops[3]]
    assert [row["status"] for row in result["results"]] == [
        "not_previewed", "not_previewed", "verified", "verified"]
    assert [row["index"] for row in result["preview_skipped"]] == [0, 1]
    assert "touch has no user prototype yet" in result["preview_skipped"][0]["reason"]
    assert "depends on operation 0" in result["preview_skipped"][1]["reason"]
    assert result["message"].startswith("2 of 4 operations were not previewed")
    assert result["success"] is True and result["rolled_back"] is True
    assert auto_fn.has_user_type is False


def test_preview_skips_every_prototype_beside_a_rename_999(monkeypatch):
    """A rename in the same --preview batch means no prototype target can be
    proven from the pre-mutation view (#630 round 5). Every set_prototype, and
    every local op, is left out; the rename and comment are still previewed."""
    bridge, instance, bv, auto_fn, applied = _skip_preview_fixture(monkeypatch)
    ops = [
        {"op": "set_prototype", "identifier": "touch", "prototype": "int touch(void* b)"},
        {"op": "local_retype", "function": "other", "variable": "arg1", "new_type": "int"},
        {"op": "rename_symbol", "identifier": "other", "new_name": "renamed"},
        {"op": "set_comment", "address": "0x401200", "comment": "hi"},
    ]

    result = instance._mutation("active", True, ops)

    assert applied == [ops[2], ops[3]]
    assert [row["index"] for row in result["preview_skipped"]] == [0, 1]
    assert "rename_symbol" in result["preview_skipped"][0]["reason"]


def test_preview_with_nothing_left_to_preview_is_still_refused_999(monkeypatch):
    bridge, instance, bv, auto_fn, applied = _skip_preview_fixture(monkeypatch)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._mutation("active", True, [
            {"op": "set_prototype", "identifier": "touch", "prototype": "int touch(void* b)"},
            {"op": "local_rename", "function": "touch", "variable": "arg1", "new_name": "b"},
        ])

    assert exc.value.status == "unsupported"
    assert exc.value.message.startswith("Cannot --preview a prototype change on touch:")
    assert applied == [] and not _has_event(bv, "begin")


def test_preview_reports_an_unparseable_prototype_as_a_parse_error_999(monkeypatch):
    """#999 item 1: a prototype that does not parse used to be reported as "cannot
    preview", so every attempt to find the bad spelling was a live one."""
    bridge, instance, bv, auto_fn, applied = _skip_preview_fixture(monkeypatch)

    def reject(text):
        raise SyntaxError("error: string.hpp:1:11 unknown type name 'nosuch'")

    monkeypatch.setattr(bv, "parse_type_string", reject, raising=False)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._mutation("active", True, [
            {"op": "set_prototype", "identifier": "touch", "prototype": "int touch(nosuch* b)"},
            {"op": "set_comment", "address": "0x4011aa", "comment": "hello"},
        ])

    assert exc.value.status == "invalid_request"
    assert "could not parse prototype" in exc.value.message
    assert applied == []


def test_preview_reports_an_unresolvable_prototype_target_999(monkeypatch):
    """#999 review: a set_prototype whose function does not exist was skipped as
    if it had found one ("has no user prototype yet") and the preview exited 0,
    while the same batch live fails on it. The preview reports the miss."""
    bridge, instance, bv, auto_fn, applied = _skip_preview_fixture(monkeypatch)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._mutation("active", True, [
            {"op": "set_prototype", "identifier": "nosuchfn", "prototype": "int f(void)"},
            {"op": "set_comment", "address": "0x4011aa", "comment": "hello"},
        ])

    assert exc.value.message == "Function not found: nosuchfn"
    assert applied == []


def test_preview_parse_checks_a_prototype_beside_a_rename_999(monkeypatch):
    """#999 review: the rename branch skipped every prototype without checking
    that it parses, so a typo previewed as exit 0 and failed only on commit."""
    bridge, instance, bv, auto_fn, applied = _skip_preview_fixture(monkeypatch)

    def reject(text):
        raise SyntaxError("error: string.hpp:1:11 unknown type name 'nosuch'")

    monkeypatch.setattr(bv, "parse_type_string", reject, raising=False)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._mutation("active", True, [
            {"op": "set_prototype", "identifier": "touch", "prototype": "int f(struct nosuch* p)"},
            {"op": "rename_symbol", "identifier": "other", "new_name": "other2"},
        ])

    assert exc.value.status == "invalid_request"
    assert exc.value.message.startswith(
        "could not parse prototype 'int f(struct nosuch* p)'")
    assert applied == []


def test_preview_skip_reason_names_a_cause_that_is_not_a_rename_999(monkeypatch):
    """An op that is not a rename but is not on the resolution-safe whitelist
    either (`data_retype`) is named as such in the skip reason -- never blamed
    as a rename (#630 follow-up wording, kept through #999 item 5)."""
    bridge, instance, bv, auto_fn, applied = _skip_preview_fixture(monkeypatch)

    result = instance._mutation("active", True, [
        {"op": "set_prototype", "identifier": "touch", "prototype": "int touch(void* b)"},
        {"op": "data_retype", "address": "0x404000", "new_type": "int32_t"},
    ])

    reason = result["preview_skipped"][0]["reason"]
    assert "the operation 'data_retype', which may change how a symbol resolves" in reason
    assert "rename" not in reason


def test_preview_misspelled_prototype_field_gets_the_field_hint_999(monkeypatch):
    """#999 review: the preview preflights read `prototype` before the #650 shape
    check ran, so a misspelled field was `internal error: KeyError` under
    --preview and the field-name hint only without it."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x4011AA, "touch")
    bv = _FakeMutationBV(functions=[fn])
    fn.view = bv
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._mutation("active", True, [
            {"op": "set_prototype", "identifier": "touch", "proto": "int touch(void* b)"},
            {"op": "set_comment", "address": "0x4011aa", "comment": "hello"},
        ])

    assert exc.value.status == "invalid_request"
    assert "did you mean 'prototype'" in exc.value.message
    assert not _has_event(bv, "begin")


@pytest.mark.parametrize("later", [
    {"op": "rename_symbol", "identifier": "other", "new_name": "nosuchfn"},
    {"op": "function_create", "address": "0x401300"},
])
def test_preview_fails_an_unresolvable_prototype_no_earlier_op_can_supply_999(
        monkeypatch, later):
    """#999 review rounds 2-3: an op AFTER the prototype cannot supply its target
    -- even a rename to exactly that name -- so the miss fails the preview the
    way it fails the live apply."""
    bridge, instance, bv, auto_fn, applied = _skip_preview_fixture(monkeypatch)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._mutation("active", True, [
            {"op": "set_prototype", "identifier": "nosuchfn", "prototype": "int f(void)"},
            later,
        ])

    assert exc.value.message == "Function not found: nosuchfn"
    assert applied == []


@pytest.mark.parametrize("earlier", [
    {"op": "rename_symbol", "identifier": "other", "new_name": "unrelated"},
    {"op": "function_create", "address": "0x401300"},
])
def test_preview_leaves_out_an_unresolvable_prototype_after_a_resolution_op_999(
        monkeypatch, earlier):
    """#999 review round 3: whether an earlier op supplies the target is not
    predicted (a follow-up). The prototype is left out with a reason that says
    its target was NOT checked -- never reported as checked."""
    bridge, instance, bv, auto_fn, applied = _skip_preview_fixture(monkeypatch)

    result = instance._mutation("active", True, [
        earlier,
        {"op": "set_prototype", "identifier": "newfn", "prototype": "int f(void)"},
    ])

    (skip,) = result["preview_skipped"]
    assert skip["index"] == 1
    assert "does not resolve yet (Function not found: newfn)" in skip["reason"]
    assert "only checked when the batch is applied" in skip["reason"]
    assert result["results"][1]["status"] == "not_previewed"
    assert applied == [earlier]


@pytest.mark.parametrize("bn_column,typed_column", [
    (32, 30),   # `this`, after the quoted name: both backticks before it
    (6, 5),     # inside the name: only the opening backtick before it
    (1, 1),     # before the name: none
])
def test_a_prototype_parse_error_quotes_what_was_typed_999(monkeypatch, bn_column,
                                                            typed_column):
    """The declarator quoting is internal: the error shows the user's text, and
    BN's column -- which counts the quoted rewrite -- points into that text."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def reject(text):
        raise SyntaxError(f"error: string.hpp:1:{bn_column} invalid parameter name: "
                          "'this' is a keyword")

    monkeypatch.setattr(bv, "parse_type_string", reject, raising=False)
    typed = "int ns::Cls::run(struct Cls* this)"
    with pytest.raises(bridge.OperationFailure) as exc:
        bridge.mutation_engine._parse_prototype(
            instance.ctx, bv, {"op": "set_prototype", "prototype": typed})
    assert exc.value.message.startswith(f"could not parse prototype {typed!r}")
    assert "`ns::Cls::run`" not in exc.value.message
    assert f"string.hpp:1:{typed_column} " in exc.value.message


def test_a_parse_error_column_on_the_name_line_is_translated_999(monkeypatch):
    """A prototype split over lines: only the name's own line moved."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def reject(text):
        raise SyntaxError("error: string.hpp:2:19 expected ')' "
                          "error: string.hpp:1:3 note")

    monkeypatch.setattr(bv, "parse_type_string", reject, raising=False)
    typed = "int\n ns::Cls::ref(int i"
    with pytest.raises(bridge.OperationFailure) as exc:
        bridge.mutation_engine._parse_prototype(
            instance.ctx, bv, {"op": "set_prototype", "prototype": typed})
    assert "string.hpp:2:17 " in exc.value.message
    assert "string.hpp:1:3 " in exc.value.message


@pytest.mark.parametrize("text,expected", [
    ("int ns::Cls::run(int x)", "int `ns::Cls::run`(int x)"),
    ("struct `ns::Cls`* Cls::get(void)", "struct `ns::Cls`* `Cls::get`(void)"),
    ("int `Cls::run`(int x)", "int `Cls::run`(int x)"),       # already quoted
    ("int run(struct ns::Cls* self)", "int run(struct ns::Cls* self)"),  # only the name
    ("ns::Cls(int x)", "ns::Cls(int x)"),                     # no return type: not a name
    ("void *ns::Cls::method(void* p)", "void *`ns::Cls::method`(void* p)"),
    ("int &ns::Cls::ref(void)", "int &`ns::Cls::ref`(void)"),
    ("void * ns::Cls::method(void* p)", "void * `ns::Cls::method`(void* p)"),
])
def test_a_qualified_prototype_name_is_quoted_999(monkeypatch, text, expected):
    """#999 item 2: the name in a prototype is ignored, but BN's parser reads an
    unquoted `A::b` there as an out-of-line member definition and refuses it."""
    bridge = _load_bridge(monkeypatch)
    assert bridge.mutation_engine._quote_qualified_declarator(text) == expected


@pytest.mark.parametrize("detail,hint", [
    ("error: string.hpp:1:32 invalid parameter name: 'this' is a keyword",
     "bn local retype <function> this"),
    ("error: string.hpp:1:10 out-of-line declaration of 'run' does not match any "
     "declaration in 'Cls'", "must be quoted with backticks"),
    ("error: string.hpp:1:11 unknown type name 'nosuch'", "declare it first"),
])
def test_prototype_parse_failures_get_the_hint_for_their_cause_999(monkeypatch, detail, hint):
    """#999 item 2: `this` and an unquoted qualified name are spelling problems;
    the generic "declare it first" sent agents re-declaring types they had."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def reject(text):
        raise SyntaxError(detail)

    monkeypatch.setattr(bv, "parse_type_string", reject, raising=False)
    with pytest.raises(bridge.OperationFailure) as exc:
        bridge.mutation_engine._parse_type_or_hint(
            instance.ctx, bv, {"op": "set_prototype"}, "int f(x)", label="prototype")
    assert hint in exc.value.message
    if "declare it first" not in hint:
        assert "declare it first" not in exc.value.message



# ---------------------------------------------------------------------------
# #802: the OTHER half of the preview-proto revert -- a function that ALREADY
# carries a genuine USER prototype, so _register_prototype_restore takes the
# had_user_type=True branch. The #630 preflight only lets such a function be
# previewed, so this is the branch every real `--preview` proto set takes; a
# regression there would leave the previewed prototype committed to the view.
# ---------------------------------------------------------------------------

class _UserPrototypePathFunction(_FakeFunction):
    """A function that records which side of BN's prototype provenance each write
    took: the USER path (`set_user_type`, also reached by the `.type` setter) or
    the AUTO one (`set_auto_type`). Per the fakes' provenance contract
    (tests/test_fakes_provenance.py) only the user path pins BNFunctionHasUserType,
    so the two are not interchangeable -- but for a baseline that was ALREADY
    user-typed the apply has pinned that flag and BN offers no API to clear it
    (#630), so both paths leave the same (value, flag) pair. The write path is
    therefore the only observable that tells the two revert branches apart.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prototype_writes: list[str] = []

    def set_user_type(self, value):
        self.prototype_writes.append("user")
        super().set_user_type(value)

    def set_auto_type(self, value):
        self.prototype_writes.append("auto")
        super().set_auto_type(value)


def test_preview_proto_re_asserts_prior_user_prototype_802(monkeypatch):
    """A `--preview` prototype change on a function that ALREADY carries a
    user-declared prototype must revert to that PRIOR USER prototype -- not to an
    AUTO baseline, not to the previewed one -- and report rolled_back. Driven
    through the REAL apply/verify/revert path, so _op_set_prototype really
    registers and replays the restore with had_user_type=True; the neighbouring
    test_preview_proto_plus_safe_ops_on_user_typed_function_allowed stubs apply,
    so it never registers a restore at all (#802)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    me = bridge.mutation_engine

    fn = _UserPrototypePathFunction(0x402000, "sub_402000")
    # A genuine prior user prototype, as BN hands one back: a Type object, no
    # declarator name.
    fn.set_user_type(_FakeType("uint64_t(int32_t arg1)", type_class="FunctionTypeClass"))
    assert fn.has_user_type is True
    bv = _FakeMutationBV(functions=[fn])
    fn.view = bv

    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.ctx, "_find_function", lambda _bv, ident: fn)
    # Only the snapshot/diff machinery is stubbed: it takes no part in the
    # register/restore path under test.
    monkeypatch.setattr(me, "_guess_affected_functions", lambda ctx, b, ops: [fn])
    monkeypatch.setattr(me, "_capture_function_snapshots", lambda ctx, b, fns: {})
    monkeypatch.setattr(me, "_capture_type_snapshots", lambda ctx, b, ops: {})
    monkeypatch.setattr(me, "_diff_snapshots", lambda ctx, before, after: [])
    monkeypatch.setattr(me, "_diff_type_snapshots", lambda ctx, before, after: [])

    result = instance._mutation(
        "active", True,
        [{"op": "set_prototype", "identifier": "sub_402000",
          "prototype": "void sub_402000(uint32_t* p)"}],
    )

    assert result["success"] is True
    assert result["rolled_back"] is True
    assert result["message"] == "Preview verified and reverted."
    assert "prototype_user_type_residue" not in result
    # The user's PRIOR prototype is back -- not the previewed one, and not an AUTO
    # baseline -- and the function is still USER-typed.
    assert str(fn.type) == "uint64_t(int32_t arg1)"
    assert fn.has_user_type is True
    # ...and the revert re-asserted it through the USER path: pin, apply, revert.
    # A revert that "treated the baseline as AUTO" writes the auto side of BN's
    # provenance instead, which is exactly the regression this branch forbids
    # (#582) -- see the class docstring for why the path is the observable.
    assert fn.prototype_writes == ["user", "user", "user"]


def test_live_verify_fail_with_proto_on_auto_reports_residue_and_fails(monkeypatch):
    """An INVOLUNTARY rollback -- a live batch where a proto-set-on-AUTO applied
    but a sibling op failed verification -- must report success:false /
    rolled_back:false and disclose the unclearable has_user_type residue, never a
    false clean rollback (#630)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "f")  # has_user_type starts False
    bv = _FakeMutationBV(functions=[fn])

    def apply(bv_, op, restores=None, **kwargs):
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

    def apply(bv_, op, restores=None, **kwargs):
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


def _auto_function_pair():
    impl = _FakeFunction(0x401000, "process")
    impl.set_user_type("int32_t()")  # genuine prior user type
    stub = _FakeFunction(0x401800, "process")  # AUTO
    stub.symbol = types.SimpleNamespace(
        type=types.SimpleNamespace(name="ImportedFunctionSymbol"))
    return [impl, stub]


@pytest.mark.parametrize("functions,ops", [
    # #630 round 2: rename an AUTO function, then set its prototype BY THE NEW NAME.
    (lambda: [_FakeFunction(0x401000, "orig_fini")],
     [{"op": "rename_symbol", "identifier": "orig_fini", "new_name": "renamed_fini"},
      {"op": "set_prototype", "identifier": "renamed_fini",
       "prototype": "void renamed_fini(int32_t value)"}]),
    # A case-folding rename, then the prototype by the folded name.
    (lambda: [_FakeFunction(0x401000, "greet")],
     [{"op": "rename_symbol", "identifier": "greet", "new_name": "Round3Case"},
      {"op": "set_prototype", "identifier": "round3case",
       "prototype": "uint64_t round3case(int32_t a)"}]),
    # Future-rename hijack: a later rename hands the prototype's name to another function.
    (lambda: [_FakeFunction(0x401000, "shared"), _FakeFunction(0x402000, "impl")],
     [{"op": "set_prototype", "identifier": "shared", "prototype": "uint64_t shared(int32_t a)"},
      {"op": "rename_symbol", "identifier": "impl", "new_name": "shared"}]),
    # Round 4: a data symbol created at an AUTO function's address, then used as the target.
    (lambda: [_FakeFunction(0x401000, "sub_401000")],
     [{"op": "rename_symbol", "kind": "data", "identifier": "0x401000", "new_name": "g_alias"},
      {"op": "set_prototype", "identifier": "g_alias",
       "prototype": "uint64_t g_alias(int32_t a)"}]),
    # Round 4: a USER-typed impl renamed away, so its shared name falls to the AUTO stub.
    (_auto_function_pair,
     [{"op": "rename_symbol", "identifier": "process", "new_name": "process_impl"},
      {"op": "set_prototype", "identifier": "process",
       "prototype": "uint64_t process(int32_t a)"}]),
])
def test_preview_never_applies_a_prototype_beside_a_rename(monkeypatch, functions, ops):
    """#630 round 5: a rename or symbol create in the same --preview batch means
    the function a set_prototype lands on cannot be proven from the pre-mutation
    view, so it could pin has_user_type on an AUTO function. Every one of the
    review rounds' bypasses must leave the prototype out -- since #999 item 5 by
    skipping it (`not_previewed`, naming the rename) while the rest is previewed,
    no longer by refusing the batch -- and leave every AUTO function unpinned.
    The rest of the batch must preview cleanly too, or this would pass on a
    rename that never ran; and the reverted preview leaves no symbol behind."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fns = functions()
    bv = _FakeMutationBV(functions=fns)
    for fn in fns:
        fn.view = bv
    before = {fn.start: (fn.name, fn.has_user_type, str(fn.type)) for fn in fns}
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._mutation("active", True, ops)

    proto_index = next(i for i, op in enumerate(ops) if op["op"] == "set_prototype")
    row = result["results"][proto_index]
    assert row["status"] == "not_previewed"
    # Either reason is honest: the rename is named, or the target was not
    # checked because an earlier op may create it -- never "checked".
    assert ("rename_symbol" in row["message"]
            or "only checked when the batch is applied" in row["message"]), row["message"]
    assert [item["index"] for item in result["preview_skipped"]] == [proto_index]
    assert {fn.start: (fn.name, fn.has_user_type, str(fn.type)) for fn in fns} == before
    others = [r["status"] for i, r in enumerate(result["results"]) if i != proto_index]
    assert others == ["verified"] * len(others), result["results"]
    assert result["success"] is True and result["rolled_back"] is True, result["message"]
    assert bv.user_symbols == {}
    assert bv.get_symbol_at(0x401000) is None


# ---------------------------------------------------------------------------
# #630 round 5: the AUTO-prototype preview guard STOPS simulating BN's
# apply-time name resolver. Four rounds of review showed the simulation always
# loses to some symbol-resolution edge (case-fold, future-rename hijack, raw
# FunctionSymbol aliases, in-batch data symbols at a function address, stale
# impl/stub name forms). The conservative, sound-by-construction replacement:
#   * if ANY op in the same --preview batch could move function/symbol
#     name-or-address resolution (rename, symbol/function create -- anything not
#     on the known-safe whitelist), every set_prototype is left out of the
#     preview (#999 item 5; it used to refuse the whole batch); and
#   * otherwise current-view resolution == apply-time resolution, so each
#     set_prototype target is resolved with BN's REAL lookup (ctx._find_function,
#     incl. the raw-symbol-at-address fallback) -- no simulation -- and refused
#     when it resolves to an AUTO function / is ambiguous / is unresolvable.
# Over-refusing legitimate batches is acceptable; an irreversible AUTO-prototype
# pin under --preview is not.
# ---------------------------------------------------------------------------

def test_batch_changes_symbol_resolution_whitelist(monkeypatch):
    """The whitelist predicate: rename_symbol (function OR data), function_create,
    and any unknown/future op could move name-or-address resolution; comments,
    tags, local-var edits, struct-field edits, type declarations and set_prototype
    itself provably cannot. A missing/unknown op defaults to the unsafe side."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine

    # Resolution-affecting (True):
    assert me._batch_changes_symbol_resolution([{"op": "rename_symbol"}]) is True
    assert me._batch_changes_symbol_resolution(
        [{"op": "rename_symbol", "kind": "data"}]) is True
    assert me._batch_changes_symbol_resolution([{"op": "function_create"}]) is True
    assert me._batch_changes_symbol_resolution([{"op": "brand_new_op"}]) is True
    assert me._batch_changes_symbol_resolution([{"op": "set_comment"},
                                                {"op": "rename_symbol"}]) is True
    assert me._batch_changes_symbol_resolution(["not-a-dict"]) is True

    # Provably safe to coexist with a set_prototype (False):
    for safe in ("set_comment", "delete_comment", "set_prototype", "local_rename",
                 "local_retype", "struct_field_set", "struct_field_rename",
                 "struct_field_delete", "types_declare", "tag_add", "tag_remove",
                 "tag_type_create", "tag_type_remove"):
        assert me._batch_changes_symbol_resolution([{"op": safe}]) is False, safe
    assert me._batch_changes_symbol_resolution(
        [{"op": "set_prototype"}, {"op": "set_comment"}, {"op": "local_rename"}]
    ) is False


def test_unrevertible_preview_prototypes_uses_real_lookup_no_simulation(monkeypatch):
    """Helper level: on a resolution-safe batch the guard resolves each
    set_prototype target with the REAL ctx._find_function (the same path apply
    uses), not a simulated name map -- so an AUTO function reached by a raw
    FunctionSymbol alias (identifier != the function's display name) is flagged."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    me = bridge.mutation_engine

    fn = _FakeFunction(0x401000, "sub_401000")  # AUTO, display name only
    # A secondary raw FunctionSymbol alias at the same address: resolves via the
    # raw-symbol-at-address fallback in ctx._find_function (seam.py ~129), NOT via
    # the function's name forms.
    alias = types.SimpleNamespace(
        raw_name="raw_alias_sym", name="alias_sym", address=0x401000,
        type=_FakeSymbol("FunctionSymbol").type,
    )
    bv = _FakeMutationBV(functions=[fn], symbols=[alias])
    fn.view = bv

    flagged = me._unrevertible_preview_prototypes(
        instance.ctx, bv,
        [{"op": "set_prototype", "identifier": "raw_alias_sym",
          "prototype": "uint64_t raw_alias_sym(int32_t a)"}],
    )
    assert flagged == [(0, "sub_401000")]


def test_post_apply_exception_with_proto_on_auto_marks_dirty(monkeypatch):
    """#630 round 2, FINDING 2: a NON-OperationFailure exception raised AFTER the
    apply loop (e.g. post-apply reanalysis/snapshot blew up) restores the
    prototype VALUE but leaves has_user_type pinned. That residue leaves the view
    modified, so the exception path must disclose it and the bridge must mark the
    view dirty -- otherwise `close` reads bv.file.modified == false and reports no
    unsaved state. The original exception must still propagate unchanged."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    me = bridge.mutation_engine
    fn = _FakeFunction(0x401000, "f")  # has_user_type starts False
    bv = _FakeMutationBV(functions=[fn])
    fn.view = bv

    marked: list = []
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "mark_dirty", lambda b: marked.append(b))

    def apply(bv_, op, restores=None, **kwargs):
        fn.set_user_type("uint64_t f(int32_t a)")           # pins has_user_type True
        restores.append(lambda: fn.set_auto_type("int32_t()"))  # value only; flag stays
        return {"op": "set_prototype", "address": "0x401000",
                "before_has_user_type": False, "requested": {}}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply)

    # Blow up on the POST-apply snapshot only (the pre-apply `before` capture is
    # the first call and must succeed so we actually reach the apply loop).
    calls = {"n": 0}

    def snap(ctx, b, fns):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("post-apply snapshot exploded")
        return {}

    monkeypatch.setattr(me, "_capture_function_snapshots", snap)

    with pytest.raises(RuntimeError, match="post-apply snapshot exploded"):
        instance._mutation("active", False, [{"op": "set_prototype"}])

    assert fn.has_user_type is True    # BN could not clear the flag
    assert marked == [bv]              # view marked dirty despite the raise


def test_preview_locals_only_batch_reports_no_proto_residue(monkeypatch):
    """Independence from #582: a locals-only preview must not be dragged into a
    proto-residue false-failure -- it reverts cleanly (proves the two fixes are
    separate)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None, **kwargs):
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


# ---------------------------------------------------------------------------
# #630 round 5 end-to-end: the conservative rule closes codex's round-4 bypasses
# and still allows a legitimate proto-only preview on a non-AUTO function.
# ---------------------------------------------------------------------------

def test_preview_raw_alias_proto_on_auto_refused_via_real_lookup(monkeypatch):
    """Round-4 bypass (raw FunctionSymbol alias): a proto-only --preview whose
    identifier is a raw FunctionSymbol alias resolving to an AUTO function is
    REFUSED. Nothing else in the batch moves resolution, so the guard resolves the
    target with the REAL ctx._find_function (raw-symbol-at-address fallback) -- the
    same path apply uses -- and refuses because it lands on an AUTO function.
    Pristine, exit 2."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "sub_401000")  # AUTO, display name only
    alias = types.SimpleNamespace(
        raw_name="raw_alias_sym", name="alias_sym", address=0x401000,
        type=_FakeSymbol("FunctionSymbol").type,
    )
    bv = _FakeMutationBV(functions=[fn], symbols=[alias])
    fn.view = bv
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    # NOTE: ctx._find_function is NOT monkeypatched -- the real resolver runs.

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._mutation(
            "active", True,
            [{"op": "set_prototype", "identifier": "raw_alias_sym",
              "prototype": "uint64_t raw_alias_sym(int32_t a)"}],
        )

    assert exc.value.status == "unsupported"
    assert "has_user_type" in exc.value.message
    assert fn.has_user_type is False
    assert str(fn.type) == "int32_t()"
    assert not _has_event(bv, "begin")


def test_preview_proto_plus_safe_ops_on_user_typed_function_allowed(monkeypatch):
    """A --preview batch of a set_prototype on a NON-AUTO (user-typed) function,
    alongside only resolution-SAFE ops (a comment), is ALLOWED: nothing moves
    resolution, the target already carries a user type (re-asserting it on revert
    is correct), so the preview verifies and reverts cleanly (exit 0)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x402000, "user_fn")
    fn.set_user_type("int32_t()")  # genuine prior user type
    bv = _FakeMutationBV(functions=[fn])
    fn.view = bv
    # Resolution-safe batch -> the guard resolves via the REAL lookup; point it at
    # the user-typed function.
    monkeypatch.setattr(instance.ctx, "_find_function", lambda _bv, ident: fn)

    def apply(bv_, op, restores=None, **kwargs):
        restores.append(lambda: None)
        return {"op": op["op"], "address": "0x402000",
                "before_has_user_type": True, "requested": {}}

    _mutation_with_stubs(
        monkeypatch, bridge, instance, bv,
        apply=apply,
        verify=lambda bv_, result: {**result, "status": "verified"},
    )

    result = instance._mutation(
        "active", True,
        [
            {"op": "set_prototype", "identifier": "user_fn",
             "prototype": "uint64_t user_fn(int32_t a)"},
            {"op": "set_comment", "function": "user_fn", "comment": "note"},
        ],
    )
    assert result["success"] is True
    assert result["rolled_back"] is True
    assert result["message"] == "Preview verified and reverted."
    assert "prototype_user_type_residue" not in result


# ===========================================================================
# #652 -- batch apply output/validation contract.
# ===========================================================================

def test_rolled_back_present_on_the_success_path_652(monkeypatch):
    """#652: `rolled_back` was present in every case EXCEPT success, so a parser
    written against a preview or a failure -- the cases you develop against --
    raised KeyError on the happy path. It is now always present (False when
    committed), matching the read-command contract where paging keys are present
    regardless of outcome."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None, **kwargs):
        return {"op": op["op"], "requested": {}}

    def verify(bv_, result):
        return {**result, "status": "verified"}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply, verify=verify)

    live = instance._mutation("active", False, [{"op": "rename_symbol"}])
    assert live["committed"] is True and live["success"] is True
    assert live["rolled_back"] is False          # present, not absent

    monkeypatch.setattr(bridge.mutation_engine, "_revert_undo_safely", lambda ctx, bv_, s: True)
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, r: True)
    prev = instance._mutation("active", True, [{"op": "rename_symbol"}])
    assert prev["rolled_back"] is True and prev["committed"] is False


def test_duplicate_write_key_rejected_up_front_652(monkeypatch):
    """#652: a manifest writing the same key twice can NEVER succeed -- every op is
    verified against the batch's END state, so under last-write-wins op 0 is judged
    against op 1's value, fails, and takes the whole atomic batch down. Easy to hit
    when a manifest is machine-generated from two passes (the recommended bulk
    workflow). Reject before applying anything, naming BOTH op indices."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    applied = []

    def apply(bv_, op, restores=None, **kwargs):
        applied.append(op)
        return {"op": op["op"], "requested": {}}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply,
                         verify=lambda bv_, r: {**r, "status": "verified"})

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._mutation("active", False, [
            {"op": "set_comment", "function": "handle_request", "comment": "first doc"},
            {"op": "set_comment", "function": "handle_request", "comment": "second doc"},
        ])
    assert exc.value.status == "invalid_request"
    assert "operations 0 and 1" in str(exc.value)
    assert exc.value.requested["conflicting_op_indices"] == [0, 1]
    assert applied == []          # nothing was applied, so nothing to roll back


def test_duplicate_write_key_normalizes_addresses_652(monkeypatch):
    """#652: `0x401120` and its decimal spelling are the same key."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    _mutation_with_stubs(monkeypatch, bridge, instance, bv,
                         apply=lambda bv_, op, restores=None, **kwargs: {"op": op["op"], "requested": {}},
                         verify=lambda bv_, r: {**r, "status": "verified"})

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._mutation("active", False, [
            {"op": "set_comment", "address": "0x401120", "comment": "a"},
            {"op": "delete_comment", "address": str(0x401120)},
        ])
    assert exc.value.status == "invalid_request"
    assert "0x401120" in str(exc.value)


def test_distinct_and_accumulative_keys_still_apply_652(monkeypatch):
    """#652 negative control -- the assertion that keeps the check from rejecting
    legitimate manifests. Different addresses, different attributes of the same
    variable (local_rename vs local_retype), and accumulative tag ops at one
    address are all fine."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    _mutation_with_stubs(monkeypatch, bridge, instance, bv,
                         apply=lambda bv_, op, restores=None, **kwargs: {"op": op["op"], "requested": {}},
                         verify=lambda bv_, r: {**r, "status": "verified"})

    result = instance._mutation("active", False, [
        {"op": "set_comment", "address": "0x401120", "comment": "a"},
        {"op": "set_comment", "address": "0x401180", "comment": "b"},
        {"op": "local_rename", "function": "f", "variable": "var_8", "new_name": "len"},
        {"op": "local_retype", "function": "f", "variable": "var_8", "new_type": "int32_t"},
        {"op": "tag_add", "address": "0x401120", "type": "Bookmarks"},
        {"op": "tag_add", "address": "0x401120", "type": "Bugs"},
    ])
    assert result["success"] is True and len(result["results"]) == 6


# ===========================================================================
# #649 -- `data retype`: a verified mutation path for typing a data variable.
# ===========================================================================

class _FakeDataVar:
    def __init__(self, type_text, *, auto_discovered=True):
        self.type = _FakeType(type_text)
        self.auto_discovered = auto_discovered

    def __str__(self):
        return f"<var {self.type}>"


class _DataVarBV(_FakeMutationBV):
    """A view modelling BN's data-variable surface, including the journaling
    behaviour verified live: `define_user_data_var` IS captured by the undo
    buffer, so `revert_undo_actions` restores the prior (auto) type."""

    def __init__(self, *, data_vars=None, mapped=True, **kwargs):
        super().__init__(**kwargs)
        self._data_vars = dict(data_vars or {})
        self._mapped = mapped
        self._journaled_data: list[tuple[str, dict]] = []

    def is_valid_offset(self, address):
        return self._mapped

    def parse_type_string(self, text):
        return _FakeType(str(text)), None

    def get_data_var_at(self, address):
        return self._data_vars.get(int(address))

    def define_user_data_var(self, address, type_obj):
        self._data_vars[int(address)] = _FakeDataVar(str(type_obj), auto_discovered=False)

    def begin_undo_actions(self):
        state = super().begin_undo_actions()
        self._journaled_data.append((state, dict(self._data_vars)))
        return state

    def _pop_data_journal(self, state):
        for i in range(len(self._journaled_data) - 1, -1, -1):
            if self._journaled_data[i][0] == state:
                snapshot = self._journaled_data[i][1]
                del self._journaled_data[i:]
                return snapshot
        return None

    def revert_undo_actions(self, state):
        snapshot = self._pop_data_journal(state)
        if snapshot is not None:
            self._data_vars = dict(snapshot)
        super().revert_undo_actions(state)

    def commit_undo_actions(self, state):
        self._pop_data_journal(state)
        super().commit_undo_actions(state)


def _data_retype_instance(monkeypatch, bv):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    return bridge, instance


def test_data_retype_applies_and_verifies_649(monkeypatch):
    """#649: struct-typing a recovered global table is a routine RE move with no
    verified path -- `types declare` defines the struct but cannot apply it,
    `symbol rename --kind data` renames without typing, and `struct field set`
    edits a type, not a variable's binding. The only way through was `py exec`:
    no --preview, no readback, no batch atomicity, no audit trail."""
    bv = _DataVarBV(data_vars={0x460000: _FakeDataVar("void")})
    bridge, instance = _data_retype_instance(monkeypatch, bv)

    result = instance._mutation("active", False, [
        {"op": "data_retype", "address": "0x460000", "new_type": "cmd_help_entry[257]"}])

    assert result["success"] is True and result["committed"] is True
    op = result["results"][0]
    assert op["status"] == "verified"
    assert op["before_type"] == "void"
    assert op["expected_type"] == "cmd_help_entry[257]"
    # readback, not a claim: the view really carries the type now
    assert str(bv.get_data_var_at(0x460000).type) == "cmd_help_entry[257]"
    assert bv.get_data_var_at(0x460000).auto_discovered is False


def test_data_retype_preview_applies_and_reverts_649(monkeypatch):
    bv = _DataVarBV(data_vars={0x460000: _FakeDataVar("void")})
    bridge, instance = _data_retype_instance(monkeypatch, bv)

    result = instance._mutation("active", True, [
        {"op": "data_retype", "address": "0x460000", "new_type": "uint32_t[4]"}])

    assert result["preview"] is True and result["committed"] is False
    assert result["rolled_back"] is True
    assert result["results"][0]["status"] == "verified"   # it DID land, then reverted
    # define_user_data_var is journaled (verified live), so the revert restores it
    assert str(bv.get_data_var_at(0x460000).type) == "void"


def test_data_retype_same_type_is_noop_649(monkeypatch):
    bv = _DataVarBV(data_vars={0x460000: _FakeDataVar("uint32_t[4]")})
    bridge, instance = _data_retype_instance(monkeypatch, bv)

    result = instance._mutation("active", False, [
        {"op": "data_retype", "address": "0x460000", "new_type": "uint32_t[4]"}])
    assert result["results"][0]["status"] == "noop"
    assert result["success"] is True


def test_data_retype_unmapped_address_is_invalid_request_649(monkeypatch):
    """A typo'd address must be a clean invalid_request, not a data var defined
    into nowhere and then reported `verified` against itself."""
    bv = _DataVarBV(data_vars={}, mapped=False)
    bridge, instance = _data_retype_instance(monkeypatch, bv)

    result = instance._mutation("active", False, [
        {"op": "data_retype", "address": "0xdeadbeef", "new_type": "uint32_t"}])
    assert result["success"] is False
    assert result["results"][-1]["status"] == "invalid_request"
    assert "not mapped" in result["results"][-1]["message"]


def test_data_retype_undeclared_type_is_clean_error_649(monkeypatch):
    """An undeclared type name is a clean invalid_request pointing at `types
    declare`, not a traceback."""
    bv = _DataVarBV(data_vars={0x460000: _FakeDataVar("void")})

    def _boom(text):
        raise RuntimeError("error: unknown type name 'cmd_help_entry'")

    bv.parse_type_string = _boom
    bridge, instance = _data_retype_instance(monkeypatch, bv)

    result = instance._mutation("active", False, [
        {"op": "data_retype", "address": "0x460000", "new_type": "cmd_help_entry[257]"}])
    assert result["results"][-1]["status"] == "invalid_request"
    message = result["results"][-1]["message"]
    assert "could not parse type" in message and "types declare" in message
    assert "Traceback" not in message


def test_data_retype_verification_failure_rolls_back_649(monkeypatch):
    """Readback is a real gate: if the type does not land, the batch fails and
    reverts rather than reporting a false `verified`."""
    bv = _DataVarBV(data_vars={0x460000: _FakeDataVar("void")})
    bv.define_user_data_var = lambda address, type_obj: None   # silently ignores the write
    bridge, instance = _data_retype_instance(monkeypatch, bv)

    result = instance._mutation("active", False, [
        {"op": "data_retype", "address": "0x460000", "new_type": "uint32_t[4]"}])
    assert result["success"] is False and result["committed"] is False
    assert result["results"][0]["status"] == "verification_failed"
    assert result["results"][0]["observed"]["type"] == "void"


def test_data_retype_composes_with_types_declare_in_one_batch_649(monkeypatch):
    """#649: the natural pairing -- declare the struct and bind it to the address
    in ONE atomic manifest."""
    bv = _DataVarBV(data_vars={0x460000: _FakeDataVar("void")})
    bridge, instance = _data_retype_instance(monkeypatch, bv)
    me = bridge.mutation_engine
    # Keep the types_declare half stubbed: this asserts COMPOSITION (both ops run
    # under one transaction and one verification pass), not BN's C parser.
    monkeypatch.setattr(me, "_op_types_declare",
                        lambda ctx, bv_, op, *a, **k: {"op": "types_declare",
                                                       "requested": {}, "defined_types": {}})
    monkeypatch.setattr(me, "_verify_declared_types",
                        lambda ctx, bv_, result: {**result, "status": "verified"})

    result = instance._mutation("active", False, [
        {"op": "types_declare", "declaration": "struct cmd_help_entry { char* desc; };"},
        {"op": "data_retype", "address": "0x460000", "new_type": "cmd_help_entry[257]"},
    ])
    assert result["success"] is True and result["committed"] is True
    assert [r["status"] for r in result["results"]] == ["verified", "verified"]


def test_failing_data_retype_rolls_back_the_types_declare_649(monkeypatch):
    """#649: the batch is atomic in both directions -- a failing data_retype must
    revert the types_declare that preceded it."""
    bv = _DataVarBV(data_vars={0x460000: _FakeDataVar("void")}, mapped=False)
    bridge, instance = _data_retype_instance(monkeypatch, bv)
    me = bridge.mutation_engine
    declared = {}
    monkeypatch.setattr(me, "_op_types_declare",
                        lambda ctx, bv_, op, *a, **k: (declared.setdefault("applied", True),
                                                       {"op": "types_declare", "requested": {},
                                                        "defined_types": {}})[1])

    result = instance._mutation("active", False, [
        {"op": "types_declare", "declaration": "struct cmd_help_entry { char* desc; };"},
        {"op": "data_retype", "address": "0xdeadbeef", "new_type": "cmd_help_entry[257]"},
    ])
    assert result["success"] is False and result["committed"] is False
    assert result["rolled_back"] is True
    assert declared["applied"] is True          # it HAD been applied...
    assert _has_event(bv, "revert")             # ...and the transaction was reverted


# ===========================================================================
# #650 -- up-front batch op/field validation with did-you-mean.
# ===========================================================================

def test_batch_guessed_op_name_rejected_up_front_650(monkeypatch):
    """#650: the batch path has no argparse layer, so a guessed op name failed only
    at APPLY time -- and, the batch being atomic, took every good op with it. One
    agent lost 2 rounds guessing `retype_local` vs `local_retype`. Reject before
    anything is applied, with the did-you-mean #361 added."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeCommentMutationBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._mutation("active", False, [
            {"op": "set_comment", "address": "0x1000", "comment": "good op"},
            {"op": "retype_local", "function": "f", "variable": "v", "new_type": "int32_t"},
        ])
    assert exc.value.status == "unsupported"
    assert "operation 1" in exc.value.message or "retype_local" in exc.value.message
    assert "'local_retype'" in exc.value.message          # did you mean
    assert bv.get_comment_at(0x1000) == ""                # the good op never ran
    assert not _has_event(bv, "begin")


def test_batch_guessed_field_name_hints_the_right_one_650(monkeypatch):
    """#650: per-op required fields are not mutually consistent (`local_retype`
    takes `variable` where `rename_symbol` takes `identifier`), so hint from the
    keys actually provided instead of only naming the missing one."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeCommentMutationBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._mutation("active", False, [
            {"op": "local_retype", "function": "f", "variables": "v", "new_type": "int32_t"}])
    assert exc.value.status == "invalid_request"
    assert "'variable'" in exc.value.message
    assert "'variables'" in exc.value.message       # names what WAS provided


def test_batch_validation_covers_every_op_before_the_first_apply_650(monkeypatch):
    """The pass is over the WHOLE manifest: a bad op at index 13 is caught with the
    twelve good ops still unapplied."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeCommentMutationBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    ops = [{"op": "set_comment", "address": hex(0x1000 + i * 4), "comment": f"note {i}"}
           for i in range(12)]
    ops.append({"op": "set_comment", "address": "0x2000"})     # index 12: no comment
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._mutation("active", False, ops)
    assert "operation 12" in exc.value.message
    assert all(bv.get_comment_at(0x1000 + i * 4) == "" for i in range(12))


# ===========================================================================
# #624 -- the comment fake must be undo-honest. `_FakeCommentMutationBV` used
# to keep a SECOND dict for `get_comment_at`/`set_comment_at` while the
# inherited `address_comments` read `_comments`, and the undo journal snapshotted
# only function provenance -- so a "reverted" comment survived in the store the
# snapshot/list paths read, and a rollback test could stay green with the view
# state wrong.
# ===========================================================================

def test_fake_comment_store_is_unified_624():
    """Real BN exposes ONE global address-comment store: what `set_comment_at`
    writes is what `get_comment_at` reads and what `address_comments` hands back.
    A second dict on the subclass broke that, so an apply landed in a store the
    snapshot/list paths never saw."""
    bv = _FakeCommentMutationBV()

    bv.set_comment_at(0x1000, "x")
    assert bv.get_comment_at(0x1000) == "x"
    assert bv.address_comments[0x1000] == "x"

    # An empty/None comment REMOVES the key (real BN semantics), in both views.
    bv.set_comment_at(0x1000, "")
    assert bv.get_comment_at(0x1000) == ""
    assert 0x1000 not in bv.address_comments


def test_fake_mutation_undo_restores_address_comments_624():
    """Real BN journals global address comments, so a reverted transaction drops
    the comment it applied; a commit keeps it. The fake journal used to carry
    only function provenance, so a revert left the comment behind."""
    bv = _FakeCommentMutationBV()
    state = bv.begin_undo_actions()
    bv.set_comment_at(0x1000, "x")
    bv.revert_undo_actions(state)
    assert bv.get_comment_at(0x1000) == ""
    assert bv.address_comments == {}

    bv = _FakeCommentMutationBV()
    state = bv.begin_undo_actions()
    bv.set_comment_at(0x1000, "x")
    bv.commit_undo_actions(state)
    assert bv.get_comment_at(0x1000) == "x"

    # Nested transactions keep the existing stack semantics: reverting the inner
    # transaction restores the comment state at ITS begin, the outer one at its
    # own, and closing the outer drops any snapshot left nested inside it.
    bv = _FakeCommentMutationBV()
    outer = bv.begin_undo_actions()
    bv.set_comment_at(0x1000, "outer")
    inner = bv.begin_undo_actions()
    bv.set_comment_at(0x1000, "inner")
    bv.revert_undo_actions(inner)
    assert bv.get_comment_at(0x1000) == "outer"
    bv.revert_undo_actions(outer)
    assert bv.get_comment_at(0x1000) == ""


def test_preview_set_comment_revert_clears_the_view_624(monkeypatch):
    """A preview's promise is that the view is left unmodified. For a comment op
    the revert must really clear the address comment it applied -- the rollback
    tests only asserted undo events and status strings, so a revert that skipped
    the comment store shipped green (#173's rollback contract, #624)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeCommentMutationBV(memory={0x1000: b"\x00"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._mutation("active", True, [
        {"op": "set_comment", "address": "0x1000", "comment": "reverted note"}])

    assert result["preview"] is True and result["committed"] is False
    assert result["rolled_back"] is True
    assert result["results"][0]["status"] == "verified"   # it DID land, then reverted
    # readback, not a claim: the view really carries no comment now
    assert bv.get_comment_at(0x1000) == ""
    assert 0x1000 not in bv.address_comments


def test_batch_invalid_op_rolls_back_prior_applied_op(monkeypatch):
    """#624 AC3, on the LIVE (non-preview) batch path: when a later op fails, the
    comment an earlier op already applied must be gone from the VIEW, not merely
    reported as 'reverted'. The old test of this name asserted only undo events
    and status strings, so a revert that skipped the comment store shipped green;
    #624's dual-store/undo fixes are what make the readback below meaningful."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeCommentMutationBV(memory={0x1000: b"\x00"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._mutation("active", False, [
        {"op": "set_comment", "address": "0x1000", "comment": "doomed note"},
        {"op": "local_rename", "function": "0x2000", "variable": "v", "new_name": "w"},
    ])

    assert result["committed"] is False
    assert result["rolled_back"] is True
    statuses = [r["status"] for r in result["results"]]
    assert statuses[0] == "reverted"          # applied, then undone by the sibling
    assert statuses[1] != "verified"          # the op that actually failed
    # readback, not a claim: the view really carries no comment now
    assert bv.get_comment_at(0x1000) == ""
    assert 0x1000 not in bv.address_comments


# ===========================================================================
# #782 -- the tag/comment half of the undo journal. BN's undo buffer carries
# more than function provenance and the global address-comment store: it also
# carries `Function.comment` (the whole-function doc comment), the function tag
# collections, the function ADDRESS tag collections, the view-level data tags
# and the tag types -- all verified live on BN 5.4, where a reverted transaction
# comes back with the prior comment and the prior tags, including tags it
# removed and a tag type it removed. The fake journaled none of those, so a
# `comment set --function` / `tag add` --preview reported rolled_back:true while
# the view still held the write, and a rollback of those two ops could not be
# asserted at all.
# ===========================================================================

_DOC_FN_START = 0x401000
_DOC_FN_END = _DOC_FN_START + 0x100


def _tag_comment_bv(*, comment: str = ""):
    """One function plus one tag type, ready for a function-scoped
    comment/tag preview through the real _mutation path."""
    fn = _FakeFunction(_DOC_FN_START, "handle_request")
    fn.basic_blocks = [_FakeBasicBlock(_DOC_FN_START, _DOC_FN_END)]
    bv = _FakeMutationBV()
    bv.functions = [fn]
    fn.view = bv
    bv.create_tag_type("Bug", "!")
    fn.comment = comment
    return bv, fn


def _preview_mutation(monkeypatch, instance, bv, ops):
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    return instance._mutation("active", True, ops)


def _fn_tags(fn):
    return [(t.type.name, t.data) for t in fn.get_function_tags()]


def test_fake_undo_restores_function_doc_tags_and_tag_types_782():
    """The journal must carry the doc comment, both function tag collections and
    the view tag types, restoring the PRIOR values -- a comment and a tag the
    transaction removed come back, and a tag type it removed does too."""
    bv = _FakeMutationBV()
    fn = _FakeFunction(_DOC_FN_START, "handle_request")
    fn.view = bv
    bv.functions = [fn]
    bv.create_tag_type("Bug", "!")
    addr = _DOC_FN_START + 4
    fn.comment = "previous doc"
    fn.add_tag("Bug", "existing function tag", None)
    fn.add_tag("Bug", "existing address tag", addr, arch=bv.arch)
    bv.add_tag(_DOC_FN_START + 8, "Bug", "existing data tag")
    before = fn.tag_comment_snapshot()

    state = bv.begin_undo_actions()
    fn.comment = "replacement doc"
    fn.remove_user_function_tag(fn.get_function_tags()[0])
    fn.add_tag("Bug", "fresh function tag", None)
    fn.remove_user_address_tag(addr, fn.get_tags_at(addr)[0], arch=bv.arch)
    bv.remove_tag_type("Bug")
    bv.create_tag_type("Scratch", "?")

    bv.revert_undo_actions(state)

    assert fn.comment == "previous doc"
    assert _fn_tags(fn) == [("Bug", "existing function tag")]
    assert [(t.type.name, t.data) for t in fn.get_tags_at(addr)] == [("Bug", "existing address tag")]
    assert fn.tag_comment_snapshot() == before        # exact prior state, not just "not empty"
    assert sorted(bv.tag_types) == ["Bug"]
    assert [(hex(a), t.data) for a, t in bv.get_tags()] == [(hex(_DOC_FN_START + 8), "existing data tag")]


def test_preview_function_doc_comment_revert_restores_previous_782(monkeypatch):
    """A --preview of `comment set --function` must leave fn.comment EXACTLY as
    it was: empty when the function had no doc comment, and the previous text
    when it had one. Pre-fix the preview reported rolled_back:true with the new
    comment still in place."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    bv, fn = _tag_comment_bv()
    result = _preview_mutation(monkeypatch, instance, bv, [
        {"op": "set_comment", "function": "handle_request", "comment": "fresh doc"}])
    assert result["preview"] is True and result["committed"] is False
    assert result["rolled_back"] is True
    assert result["results"][0]["status"] == "verified"   # it DID land, then reverted
    assert fn.comment == ""

    bv, fn = _tag_comment_bv(comment="previous doc")
    result = _preview_mutation(monkeypatch, instance, bv, [
        {"op": "set_comment", "function": "handle_request", "comment": "fresh doc"}])
    assert result["rolled_back"] is True
    assert fn.comment == "previous doc"

    # `comment delete --function` writes the SAME journaled property, so its
    # preview must put the doc comment back too.
    bv, fn = _tag_comment_bv(comment="previous doc")
    result = _preview_mutation(monkeypatch, instance, bv, [
        {"op": "delete_comment", "function": "handle_request"}])
    assert result["rolled_back"] is True
    assert fn.comment == "previous doc"


def test_preview_tag_add_revert_restores_the_tag_list_782(monkeypatch):
    """A --preview of `tag add --function` must leave the function's tag list as
    it was: empty when the function carried no tag, and exactly the prior tag
    when it carried one. Pre-fix the preview reported rolled_back:true with the
    new tag still attached."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    bv, fn = _tag_comment_bv()
    result = _preview_mutation(monkeypatch, instance, bv, [
        {"op": "tag_add", "function": "handle_request", "type": "Bug", "data": "look here"}])
    assert result["preview"] is True and result["committed"] is False
    assert result["rolled_back"] is True
    assert result["results"][0]["status"] == "verified"
    assert fn.get_function_tags() == []

    bv, fn = _tag_comment_bv()
    fn.add_tag("Bug", "pre-existing", None)
    result = _preview_mutation(monkeypatch, instance, bv, [
        {"op": "tag_add", "function": "handle_request", "type": "Bug", "data": "look here"}])
    assert result["rolled_back"] is True
    assert _fn_tags(fn) == [("Bug", "pre-existing")]


def test_preview_tag_remove_revert_restores_the_removed_tag_782(monkeypatch):
    """The other direction of the same journal: a --preview that REMOVES a tag
    must put it back, not leave the function tagless."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    bv, fn = _tag_comment_bv()
    fn.add_tag("Bug", "doomed", None)
    tag_id = str(fn.get_function_tags()[0].id)

    result = _preview_mutation(monkeypatch, instance, bv, [
        {"op": "tag_remove", "tag_id": tag_id}])
    assert result["rolled_back"] is True
    assert result["results"][0]["status"] == "verified"
    assert _fn_tags(fn) == [("Bug", "doomed")]


def test_preview_data_tag_and_tag_type_revert_restore_view_state_782(monkeypatch):
    """The view-level half of the journal: a --preview of a data-scope tag add
    drops the tag it added, and a tag-type create/remove preview leaves the type
    set unchanged."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    bv, _fn = _tag_comment_bv()
    result = _preview_mutation(monkeypatch, instance, bv, [
        {"op": "tag_add", "address": hex(_DOC_FN_START + 0x40), "type": "Bug",
         "data": "view tag", "force_data": True}])
    assert result["rolled_back"] is True
    assert bv.get_tags() == []

    bv, _fn = _tag_comment_bv()
    result = _preview_mutation(monkeypatch, instance, bv, [
        {"op": "tag_type_create", "name": "Scratch", "icon": "?"}])
    assert result["rolled_back"] is True
    assert sorted(bv.tag_types) == ["Bug"]

    bv, _fn = _tag_comment_bv()
    bv.create_tag_type("Scratch", "?")
    result = _preview_mutation(monkeypatch, instance, bv, [
        {"op": "tag_type_remove", "name": "Scratch"}])
    assert result["rolled_back"] is True
    assert sorted(bv.tag_types) == ["Bug", "Scratch"]


# --- #675 item 15: a doomed batch must say what it never attempted ---------


def test_a_failed_batch_returns_a_row_for_every_submitted_op_675(monkeypatch):
    """#675 item 15, driven through `_mutation` -- the path production uses.

    The first cut of this test called `_unattempted_results` directly and
    stayed GREEN with the wiring removed: the right logic through the wrong
    entry point, which is the form that had just been named against a sibling
    fix. The helper is not the contract; the batch response is.

    The failure path returned rows only up to and including the op that
    failed, so a 3-op batch whose first op failed answered with ONE row -- and
    the response could not express "not attempted". A consumer diffing the
    manifest against the results reads the missing ops as absent from the
    REQUEST; a consumer counting reads 1 where it submitted 3.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None, **kwargs):
        if op.get("op") == "boom":
            raise bridge.OperationFailure("unsupported", "Symbol not found: GONE",
                                          requested={})
        return {"op": op.get("op"), "status": "applied", "requested": {}}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply)
    monkeypatch.setattr(bridge.mutation_engine, "_revert_undo_safely",
                        lambda ctx, bv_, state: True)
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores",
                        lambda ctx, bv_, restores: True)

    ops = [{"op": "boom"},
           {"op": "set_comment", "address": "0x1000", "comment": "a"},
           {"op": "set_comment", "address": "0x2000", "comment": "b"}]
    result = instance._mutation("active", False, ops)

    rows = result["results"]
    assert len(rows) == len(ops), "one row per SUBMITTED op, or the count lies"
    assert rows[0]["status"] == "unsupported"
    assert [r["status"] for r in rows[1:]] == ["not_attempted", "not_attempted"]
    for row, op in zip(rows[1:], ops[1:]):
        assert "never ran" in row["message"]
        assert row["requested"]["address"] == op["address"]


def test_the_unattempted_helper_shape_675():
    """#675 item 15: the failure path returned rows only up to and including
    the op that failed, so a 3-op batch whose first op failed answered with
    ONE row.

    The response then could not express "not attempted": a consumer diffing
    the manifest against the results reads the missing ops as absent from the
    REQUEST, and a consumer counting reads 1 where it submitted 3. Unlike the
    renderer cases in this family it is not a wording problem -- no field
    carried the fact at all.
    """
    from bn_agent_bridge import mutation_engine as me
    ops = [
        {"op": "rename_symbol", "identifier": "GONE", "new_name": "x"},
        {"op": "set_comment", "address": "0x1000", "comment": "a"},
        {"op": "set_comment", "address": "0x2000", "comment": "b"},
    ]
    rows = me._unattempted_results(None, ops[1:])

    assert len(rows) == 2
    for row, op in zip(rows, ops[1:]):
        assert row["status"] == "not_attempted"
        assert row["op"] == op["op"]
        assert "never ran" in row["message"]
        # The row echoes its own request, so the reconciliation holds without
        # the consumer re-reading the manifest it sent.
        assert row["requested"]["address"] == op["address"]


def test_not_attempted_is_not_a_failure_status_675():
    """THE property that keeps the fix from making things worse: these ops did
    not fail, they never ran. Stamping them as failures would turn one bad op
    into N, inflate the failure count a control loop reads, and change nothing
    about what actually went wrong.

    Pinned against the shared status set rather than a literal, so a future
    edit that adds `not_attempted` to the failure set fails here.
    """
    from bn.formatters import FAILED_MUTATION_STATUSES
    assert "not_attempted" not in FAILED_MUTATION_STATUSES
    # `reverted` is the existing precedent for an honest non-failure status on
    # a doomed batch (#118); the new one sits beside it.
    assert "reverted" not in FAILED_MUTATION_STATUSES


def test_a_batch_that_reaches_every_op_grows_no_unattempted_rows_675():
    """Must-not-fire twin: nothing is unattempted when nothing failed, so an
    ordinary batch's result set is unchanged -- the helper returns no rows for
    an empty remainder, which is the case every successful batch hits."""
    from bn_agent_bridge import mutation_engine as me
    assert me._unattempted_results(None, []) == []


def _documented_per_op_statuses() -> set[str]:
    """The status names `skills/bn/reference/mutating.md` enumerates.

    Parsed from the file's own "Per-op statuses:" bullet list rather than
    from a copy kept here, so the guard reads the list an agent reads.
    """
    import re
    from pathlib import Path

    doc = (Path(__file__).resolve().parents[1]
           / "skills/bn/reference/mutating.md").read_text(encoding="utf-8")
    _, _, tail = doc.partition("Per-op statuses:")
    assert tail, "mutating.md no longer enumerates per-op statuses"
    names: set[str] = set()
    for line in tail.splitlines():
        if not line.strip():
            continue
        match = re.match(r"- `([a-z_]+)` +\u2014", line)
        if not match:
            break          # the list ends at the first non-bullet line
        names.add(match[1])
    return names


def test_every_status_a_doomed_batch_emits_is_in_the_reference_status_list_675(
        monkeypatch):
    """The reference's per-op status list is a CLOSED enumeration an agent
    reads to decide what a row means, so a status the bridge emits and the
    list omits reads as an unknown failure mode.

    Round 1 major: `not_attempted` shipped on every failed batch and was never
    added there. The population is taken from a real `_mutation` response
    rather than from a hand-listed set, so the guard cannot drift from what
    the engine actually emits -- and one batch with a MIDDLE op failing is
    enough to produce all three of the doomed-batch statuses at once: the op
    that already applied (re-stamped), the op that failed, and the ops never
    reached.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None, **kwargs):
        if op.get("op") == "boom":
            raise bridge.OperationFailure("unsupported", "Symbol not found: GONE",
                                          requested={})
        return {"op": op.get("op"), "status": "applied", "requested": {}}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply)
    monkeypatch.setattr(bridge.mutation_engine, "_revert_undo_safely",
                        lambda ctx, bv_, state: True)
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores",
                        lambda ctx, bv_, restores: True)

    result = instance._mutation("active", False, [
        {"op": "set_comment", "address": "0x1000", "comment": "a"},
        {"op": "boom"},
        {"op": "set_comment", "address": "0x2000", "comment": "b"},
    ])
    emitted = {row["status"] for row in result["results"]}
    # The premise: this really is the doomed-batch shape, not one status.
    assert len(emitted) == 3, emitted

    documented = _documented_per_op_statuses()
    assert emitted <= documented, (
        "mutating.md's per-op status list omits statuses the bridge emits: "
        f"{sorted(emitted - documented)}")


# --- #675 item 14 / M-03: a create inside an existing function ------------


_HOST = {"name": "main", "address": "0x401339", "offset": "0x8"}


def test_a_create_inside_an_existing_function_discloses_the_overlap_675():
    """#675 item 14: `function create` at a mid-function address returned
    `verified` with a row carrying only address/function/op/requested/status
    -- a split host and no trace of it in the result. The row builder is
    reached only by a deliberate `allow_overlap` create (M-03 refuses the
    rest), and must still name the host.
    """
    from bn_agent_bridge import create_comments as cc
    row = cc._with_overlap_note(
        {"op": "function_create", "status": "verified", "address": "0x401341"},
        [_HOST])

    assert row["overlaps"] == [_HOST]
    assert "main @ 0x401339" in row["note"]
    # It must say which reading is which, or the note is just an alarm.
    assert "legitimate" in row["note"] and "mistake" in row["note"]


def test_a_create_at_a_free_address_grows_no_overlap_keys_675():
    """Must-not-fire twin: the ordinary create is the common path. A row that
    always carried `overlaps` would make the disclosure noise, and a consumer
    testing `"overlaps" in row` would see it every time."""
    from bn_agent_bridge import create_comments as cc
    row = {"op": "function_create", "status": "verified", "address": "0x401020"}
    assert cc._with_overlap_note(dict(row), []) == row


class _Range:
    def __init__(self, start, end):
        self.start, self.end = start, end


class _HostFunction(_FakeFunction):
    """A function with the shape the interior guard and its revert read:
    `address_ranges`, blocks (`get_basic_block_at`) and a `symbol` with
    provenance. Ranges are the one source of truth; blocks mirror them so
    `_function_span` (the stock containment rule) agrees."""

    def __init__(self, start, name, ranges, *, auto_symbol=True):
        super().__init__(start, name)
        self.set_ranges(ranges)
        self.symbol = types.SimpleNamespace(auto=auto_symbol, name=name, address=start,
                                            type="FunctionSymbol")

    def set_ranges(self, ranges):
        self.ranges = [tuple(r) for r in ranges]
        self.basic_blocks = [types.SimpleNamespace(start=a, end=b, arch=None)
                             for a, b in self.ranges]

    @property
    def address_ranges(self):
        return [_Range(a, b) for a, b in self.ranges]

    def get_basic_block_at(self, addr):
        for block in self.basic_blocks:
            if block.start <= addr < block.end:
                return block
        return None

    def covers(self, addr):
        return any(a <= addr < b for a, b in self.ranges)


class _OverlappingCreateBV(_FakeFunctionCreateBV):
    """A create view that replays what real BN 6.1 does to a function a
    create lands inside (#927 review, verified on `parser_x86_64`):

      * at an INSTRUCTION BOUNDARY the host is truncated at the new start
        and the new function owns the rest;
      * at a MID-INSTRUCTION address the host is left intact and the new
        function overlaps it, decoding a different stream;
      * removing the created function lets a truncated host re-extend.

    Instructions are a fixed 4 bytes here, so "boundary" is a multiple of
    four from the block start. The stock fake's containment never answers,
    so every other create drive takes the EMPTY path and cannot observe the
    guard at all; the first cut of this stub answered `main` for every
    address and modelled no truncation, which is how a revert that did not
    restore the host passed its tests."""

    def __init__(self, *, hosts, instruction_length=4, **kw):
        super().__init__(**kw)
        self.functions = list(hosts)
        self._insn = instruction_length
        self._pending: list = []

    def get_instruction_length(self, addr, arch=None):
        return self._insn

    def get_functions_containing(self, addr):
        return [fn for fn in self.functions
                if isinstance(fn, _HostFunction) and fn.covers(int(addr))]

    def get_next_function_start_after(self, addr):
        later = [int(fn.start) for fn in self.functions if int(fn.start) > int(addr)]
        return min(later) if later else 0xFFFFFFFF

    def create_user_function(self, addr):
        self.events.append(("create_user_function", addr))
        self.added.append(int(addr))
        self._suppressed.discard(int(addr))
        fn = _HostFunction(int(addr), f"sub_{addr:x}", [], auto_symbol=True)
        fn.arch = self.arch
        fn.total_bytes = self._created_total_bytes
        self.functions.append(fn)
        self._pending.append(fn)
        return fn

    def _analyze_created(self, fn):
        """What analysis does for a function created at an interior address."""
        hosts = [h for h in self.get_functions_containing(fn.start) if h is not fn]
        end = max([b for h in hosts for a, b in h.ranges if a <= fn.start < b]
                  or [fn.start + self._created_total_bytes])
        fn.set_ranges([(fn.start, end)])
        for host in hosts:
            block = host.get_basic_block_at(fn.start)
            if (fn.start - block.start) % self._insn == 0:
                host.set_ranges([(a, fn.start if a <= fn.start < b else b)
                                 for a, b in host.ranges])

    def _settle_hosts(self):
        """A truncated host re-extends once nothing starts where it was cut."""
        for host in self.functions:
            if not isinstance(host, _HostFunction) or not hasattr(host, "_cut_from"):
                continue
            cut_at, old_end = host._cut_from
            if self.get_function_at(cut_at) is None:
                host.set_ranges([(a, old_end if b == cut_at else b) for a, b in host.ranges])
                del host._cut_from

    def update_analysis_and_wait(self):
        super().update_analysis_and_wait()
        for fn in self._pending:
            before = {h: list(h.ranges) for h in self.functions
                      if isinstance(h, _HostFunction) and h.ranges}
            self._analyze_created(fn)
            for host, ranges in before.items():
                if host.ranges != ranges:
                    host._cut_from = (fn.start, max(b for a, b in ranges))
        self._pending = []
        self._settle_hosts()


def _overlap_bv(addr, *, offset=8, hosts=None, **kw):
    """`main` covering [addr-offset, addr-offset+0x40) with the create at
    +offset -- a boundary when offset is a multiple of 4."""
    start = addr - offset
    if hosts is None:
        hosts = [_HostFunction(start, "main", [(start, start + 0x40)])]
    return _OverlappingCreateBV(
        hosts=hosts,
        segments={addr: _FakeSegment(readable=True, executable=True)},
        memory={addr: b"\x55\x48\x89\xe5"},
        **kw,
    )


def _assert_interior_refusal(exc, addr):
    """The refusal names the host, its start, the offset and which of BN's
    two outcomes the address gets -- the facts that tell a mistyped address
    from a deliberate split -- and says how to ask for it. Status
    `invalid_request` is what puts the run at exit 3 on both surfaces."""
    assert exc.status == "invalid_request"
    assert "main" in exc.message
    assert hex(addr - 8) in exc.message
    assert "+0x8" in exc.message
    assert "allow" in exc.message and "overlap" in exc.message
    assert exc.observed["overlaps"] == [
        {"name": "main", "address": hex(addr - 8), "offset": "0x8",
         "instruction_boundary": True}]


def test_the_single_command_create_inside_a_function_is_refused_m03(monkeypatch):
    """M-03: `bn function create <host+8>` committed `changed=1 verified=1`
    and cut the host down to 8 bytes; the only trace was the verbose row
    note. Refused by default, before anything is created or analysed."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    addr = 0x401341
    bv = _overlap_bv(addr)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._function_create(None, hex(addr), False)

    _assert_interior_refusal(exc.value, addr)
    assert "--allow-overlap" in exc.value.message
    assert bv.added == [] and "refresh" not in bv.events


def test_the_refusal_says_a_boundary_address_truncates_the_host_927():
    """The refusal must not promise ONE outcome (#927 review): at an
    instruction boundary BN truncates the host, and the text says so."""
    from bn_agent_bridge import create_comments as cc
    exc = cc._overlap_refusal(
        0x401341, [{**_HOST, "instruction_boundary": True}], {}, flag="--allow-overlap")
    assert "truncates the host" in exc.message
    assert "instruction boundary" in exc.message
    assert "intact" not in exc.message


def test_the_refusal_says_a_mid_instruction_address_overlaps_the_host_927():
    """...and at a mid-instruction address -- the typo-shaped case -- BN
    leaves the host intact beside a function decoding a bogus stream, which
    the old "splits the host at the new start" wording got exactly wrong."""
    from bn_agent_bridge import create_comments as cc
    exc = cc._overlap_refusal(
        0x401343, [{**_HOST, "offset": "0xa", "instruction_boundary": False}], {},
        flag="--allow-overlap")
    assert "not an instruction boundary" in exc.message
    assert "leaves the host intact" in exc.message
    assert "mid-instruction" in exc.message
    assert "truncates the host" not in exc.message


def test_the_refusal_covers_both_outcomes_when_the_boundary_is_unknown_927():
    """A host whose blocks cannot be walked gets no verdict, and the refusal
    then states both outcomes rather than guessing one."""
    from bn_agent_bridge import create_comments as cc
    exc = cc._overlap_refusal(0x401341, [_HOST], {}, flag="--allow-overlap")
    assert "either truncates the host" in exc.message
    assert "leaves the host intact" in exc.message
    assert "boundary unknown" in exc.message


def test_the_boundary_probe_walks_the_hosts_block_with_bn_instruction_lengths_927():
    from bn_agent_bridge import create_comments as cc
    host = _HostFunction(0x401339, "main", [(0x401339, 0x401379)])
    bv = _overlap_bv(0x401341, hosts=[host])
    assert cc._instruction_boundary(bv, host, 0x401341) is True     # +8
    assert cc._instruction_boundary(bv, host, 0x401343) is False    # +10
    assert cc._instruction_boundary(bv, host, 0x401339) is True     # the start
    assert cc._instruction_boundary(bv, host, 0x401400) is None     # outside
    # A function with no blocks to walk -- the stock fake -- is "unknown".
    assert cc._instruction_boundary(bv, _FakeFunction(0x401339, "main"), 0x401341) is None


def test_the_single_command_preview_obeys_the_interior_guard_m03(monkeypatch):
    """A preview that said `verified` for a create the live command refuses
    would be a plan the loop cannot execute."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    addr = 0x401341
    bv = _overlap_bv(addr)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._function_create(None, hex(addr), True)

    assert exc.value.status == "invalid_request"
    assert bv.added == []


_SPLIT_HOST = {**_HOST, "instruction_boundary": True, "truncated": True}


def test_the_single_command_create_path_returns_the_overlap_with_the_flag_675(monkeypatch):
    """The deliberate split: `allow_overlap` goes ahead and the row discloses
    the host and what became of it, asserted on the OBSERVED row rather than
    on the call site -- and the truncated host is an affected function too
    (#927 review), not only an entry under the overlap list."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    addr = 0x401341
    bv = _overlap_bv(addr)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_create(None, hex(addr), False, allow_overlap=True)

    assert result["committed"] is True
    row = result["results"][0]
    assert row["status"] == "verified"
    assert row["overlaps"] == [_SPLIT_HOST]
    assert "main @ 0x401339 was truncated at 0x401341" in row["note"]
    assert "legitimate" in row["note"] and "mistake" in row["note"]
    assert row["requested"]["allow_overlap"] is True
    # The fake did what BN does: the host now ends at the new start.
    assert bv.get_function_at(0x401339).ranges == [(0x401339, 0x401341)]
    assert result["affected_functions"] == [
        {"address": hex(addr), "before_name": None, "after_name": f"sub_{addr:x}",
         "changed": True},
        {"address": "0x401339", "before_name": "main", "after_name": "main",
         "changed": True},
    ]


def test_a_mid_instruction_create_reports_the_host_intact_927(monkeypatch):
    """BN's other outcome: the host keeps every byte and the new function
    overlaps it. The row must say `intact`, and the host audit row reports
    no change -- a status that claimed a split here would be the inverse
    of the M-03 lie."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    addr = 0x401343
    bv = _overlap_bv(addr, offset=10)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_create(None, hex(addr), False, allow_overlap=True)

    row = result["results"][0]
    assert row["status"] == "verified"
    assert row["overlaps"] == [{"name": "main", "address": "0x401339", "offset": "0xa",
                                "instruction_boundary": False, "truncated": False}]
    assert "main @ 0x401339 is intact and now overlaps the new function" in row["note"]
    assert bv.get_function_at(0x401339).ranges == [(0x401339, 0x401379)]
    assert result["affected_functions"][1] == {
        "address": "0x401339", "before_name": "main", "after_name": "main", "changed": False}


def test_the_batch_create_inside_a_function_is_refused_m03(monkeypatch):
    """The other entry point refuses in the same words, and the batch is
    reverted like any other `invalid_request`."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    addr = 0x401341
    bv = _overlap_bv(addr)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._mutation(
        "active", False, [{"op": "function_create", "address": hex(addr)}])

    assert result["success"] is False and result["committed"] is False
    row = result["results"][0]
    assert row["status"] == "invalid_request"
    assert "main" in row["message"] and hex(addr - 8) in row["message"]
    assert "+0x8" in row["message"]
    assert "allow_overlap" in row["message"]
    assert "truncates the host" in row["message"]
    assert bv.get_function_at(addr) is None


def test_the_batch_create_path_returns_the_same_overlap_with_the_flag_675(monkeypatch):
    """With `allow_overlap: true` the batch op proceeds and its row carries
    the same disclosure the standalone row does -- including the observed
    fate of the host, stamped after the batch's deferred analysis."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    addr = 0x401341
    bv = _overlap_bv(addr)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._mutation(
        "active", False,
        [{"op": "function_create", "address": hex(addr), "allow_overlap": True}])

    row = result["results"][0]
    assert row["status"] == "verified"
    assert row["overlaps"] == [_SPLIT_HOST]
    assert "was truncated at 0x401341" in row["note"]
    assert "legitimate" in row["note"] and "mistake" in row["note"]


def test_the_batch_allow_overlap_field_must_be_a_json_boolean_m03(monkeypatch):
    """`"allow_overlap": "false"` must not read as permission (#91's rule)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    addr = 0x401341
    bv = _overlap_bv(addr)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._mutation(
        "active", False,
        [{"op": "function_create", "address": hex(addr), "allow_overlap": "false"}])

    row = result["results"][0]
    assert row["status"] == "invalid_request"
    assert "allow_overlap must be a JSON boolean" in row["message"]
    assert bv.get_function_at(addr) is None


def test_a_malformed_allow_overlap_is_refused_even_at_an_existing_start_927(monkeypatch):
    """#927 review P3: the type check sat below the existing-start noop, so
    `"allow_overlap": "false"` was a noop at a function start and a refusal
    one byte in. A malformed field is malformed wherever it points."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _overlap_bv(0x401341)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._mutation(
        "active", True,
        [{"op": "function_create", "address": "0x401339", "allow_overlap": "false"}])

    row = result["results"][0]
    assert row["status"] == "invalid_request", row
    assert "allow_overlap must be a JSON boolean" in row["message"]


def test_a_view_that_cannot_answer_containment_refuses_the_create_927():
    """#927 review P3: the probe swallowed every exception and returned `[]`,
    so a view that could not say whether the address was inside a function
    let the create proceed -- and truncate -- silently: the M-03 harm, on
    the one view nobody could see it coming. A guard fails CLOSED."""
    from bn_agent_bridge import create_comments as cc

    class _Hostile:
        def get_functions_containing(self, addr):
            raise RuntimeError("analysis unavailable")

    with pytest.raises(cc.OperationFailure) as exc:
        cc._containing_function_rows(_Hostile(), 0x401341, requested={"op": "function_create"})
    assert exc.value.status == "internal_error"
    assert "Could not determine whether 0x401341 lies inside" in exc.value.message
    assert exc.value.requested == {"op": "function_create"}


def test_an_unreadable_containment_creates_nothing_on_either_path_927(monkeypatch):
    """Driven through both entry points: nothing is created or analysed when
    the guard cannot read its evidence."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    addr = 0x401341
    bv = _overlap_bv(addr)

    def _unreadable(a):
        raise RuntimeError("analysis unavailable")
    bv.get_functions_containing = _unreadable
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._function_create(None, hex(addr), False, allow_overlap=True)
    assert exc.value.status == "internal_error"
    assert bv.added == [] and "refresh" not in bv.events

    result = instance._mutation(
        "active", False, [{"op": "function_create", "address": hex(addr)}])
    assert result["results"][0]["status"] == "internal_error"
    assert bv.added == []


def test_a_function_starting_at_the_address_is_not_an_overlap_675():
    """The exact-start case is a `noop`, handled before this; it must not
    also be reported as overlapping itself."""
    from bn_agent_bridge import create_comments as cc

    class _Fn:
        def __init__(self, start, name):
            self.start, self.name = start, name

    class _BV:
        def get_functions_containing(self, addr):
            return [_Fn(0x401341, "sub_401341"), _Fn(0x401339, "main")]

    rows = cc._containing_function_rows(_BV(), 0x401341, requested={})
    assert rows == [_HOST]


# --- #927 P1: a revert must put the hosts back, not just drop the create ---


class _SharedTailCreateBV(_OverlappingCreateBV):
    """Replays the real-BN 6.1 geometry the #927 review reproduced: a
    discontiguous host A = [0x1000,0x1002)+[0x1010,0x1019) and a host
    B = [0x1004,0x1019) sharing the tail. A create at 0x1012:

      * cuts A to [0x1000,0x1002) (and analysis renames it `j_sub_1010`,
        a jump thunk), cuts B to [0x1004,0x1010), and spawns an
        ANALYSIS-CREATED function at 0x1010;
      * removing the created function leaves all of that in place -- the
        extra function simply grows over the tail;
      * `remove_function` on the extra does not stick (analysis re-creates
        it), `remove_user_function` does, and only then does reanalyzing
        the hosts give them their tail back; the auto name comes back only
        when the thunk symbol is undefined and the recorded one redefined.

    The symbols live in the view's LISTING (`get_symbols`), as on real BN,
    where a loader or `j_*` auto symbol is registered -- an implicit
    `sub_*` is not, and `_auto_symbols_at` reads the listing only (#927
    round 4).

    Every step above was observed, not designed; the stub exists so the
    revert's verification and repair have something to fail against."""

    TAIL, CUT, END = 0x1010, 0x1012, 0x1019

    def __init__(self, *, repairable=True):
        self.a = _HostFunction(0x1000, "entry_a", [(0x1000, 0x1002), (0x1010, 0x1019)])
        self.b = _HostFunction(0x1004, "entry_b", [(0x1004, 0x1019)])
        super().__init__(
            hosts=[self.a, self.b], instruction_length=1,
            segments={a: _FakeSegment(readable=True, executable=True)
                      for a in range(0x1000, 0x1019)},
            memory={a: b"\x90" for a in range(0x1000, 0x1019)},
        )
        self._symbols = [self.a.symbol, self.b.symbol]
        self._repairable = repairable
        self._split = False

    def _spawn_extra(self):
        extra = _HostFunction(self.TAIL, f"sub_{self.TAIL:x}", [(self.TAIL, self.END)])
        extra.auto = True
        self.functions.append(extra)
        return extra

    def undefine_auto_symbol(self, symbol):
        self.events.append(("undefine_auto_symbol", symbol.name))
        self._symbols.remove(symbol)

    def define_auto_symbol(self, symbol):
        self.events.append(("define_auto_symbol", symbol.name))
        self._symbols.append(symbol)
        fn = self.get_function_at(symbol.address)
        if fn is not None:
            fn.name = symbol.name
            fn.symbol = symbol

    def update_analysis_and_wait(self):
        _FakeMutationBV.update_analysis_and_wait(self)
        created = self.get_function_at(self.CUT)
        extra = self.get_function_at(self.TAIL)
        if created is not None and not self._split:
            self._split = True
            self._pending = []
            created.set_ranges([(self.CUT, self.END)])
            self.a.set_ranges([(0x1000, 0x1002)])
            self.a.name = f"j_sub_{self.TAIL:x}"
            self.a.symbol = types.SimpleNamespace(auto=True, name=self.a.name,
                                                  address=0x1000, type="FunctionSymbol")
            self._symbols = [s for s in self._symbols if s.address != 0x1000] + [self.a.symbol]
            self.b.set_ranges([(0x1004, self.TAIL)])
            self._spawn_extra().set_ranges([(self.TAIL, self.CUT)])
            return
        if not self._split:
            return
        if created is None and extra is not None:
            extra.set_ranges([(self.TAIL, self.END)])
        if created is None and extra is None:
            if self.TAIL not in self._suppressed or not self._repairable:
                self._spawn_extra()   # analysis brings it straight back
            elif self.a.reanalyzed and self.b.reanalyzed:
                self.a.set_ranges([(0x1000, 0x1002), (0x1010, 0x1019)])
                self.b.set_ranges([(0x1004, 0x1019)])
                self._split = False


def _shared_tail_snapshot(bv):
    return sorted((fn.start, fn.name, list(fn.ranges)) for fn in bv.functions)


def test_a_preview_inside_shared_tail_hosts_restores_them_or_says_so_927(monkeypatch):
    """#927 P1 (standalone): the preview reported `rolled_back: true` while
    both hosts stayed shortened and an analysis-created function kept the
    tail. The revert now verifies the hosts against their snapshot, repairs
    them the way BN allows, and only then claims a clean revert."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _SharedTailCreateBV()
    before = _shared_tail_snapshot(bv)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_create(None, hex(bv.CUT), True, allow_overlap=True)

    assert result["success"] is True and result["rolled_back"] is True
    assert result["results"][0]["status"] == "verified"
    # BOTH hosts were disclosed, as truncated, with the fate observed before
    # the revert put them back.
    assert [(o["name"], o["truncated"]) for o in result["results"][0]["overlaps"]] == [
        ("entry_a", True), ("entry_b", True)]
    assert _shared_tail_snapshot(bv) == before, "the hosts are not back as they were"
    # The repair escalated exactly as documented: non-poisoning first, then
    # the suppressing removal of the ANALYSIS-created function, never of a
    # host or of a function that was there before.
    assert ("remove_function", bv.TAIL) in bv.events
    assert ("remove_user_function", bv.TAIL) in bv.events
    assert ("remove_user_function", 0x1000) not in bv.events
    assert ("remove_user_function", 0x1004) not in bv.events
    assert ("define_auto_symbol", "entry_a") in bv.events
    assert bv.a.name == "entry_a"


def test_a_preview_whose_hosts_cannot_be_restored_reports_rollback_failed_927(monkeypatch):
    """When BN will not give the hosts back, the preview must say so: a
    failed rollback, not a success, and the view marked dirty so `close`
    warns -- `rolled_back: true` over shortened hosts is the P1 lie."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _SharedTailCreateBV(repairable=False)
    marked = []
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "mark_dirty", lambda view: marked.append(view))

    result = instance._function_create(None, hex(bv.CUT), True, allow_overlap=True)

    assert result["success"] is False and result["rolled_back"] is False
    row = result["results"][0]
    assert row["status"] == "rollback_failed"
    assert row["overlaps"], "the overlap is still live and must stay disclosed"
    assert "a function it split" in result["message"]
    assert marked == [bv]
    assert bv.a.ranges == [(0x1000, 0x1002)]


def test_a_batch_preview_inside_shared_tail_hosts_restores_them_927(monkeypatch):
    """#927 P1 (batch): the same geometry through `batch apply --preview`,
    whose restore is the op's registered closure."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _SharedTailCreateBV()
    before = _shared_tail_snapshot(bv)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._mutation(
        "active", True,
        [{"op": "function_create", "address": hex(bv.CUT), "allow_overlap": True}])

    assert result["success"] is True and result["rolled_back"] is True
    assert result["results"][0]["status"] == "verified"
    assert [(o["name"], o["truncated"]) for o in result["results"][0]["overlaps"]] == [
        ("entry_a", True), ("entry_b", True)]
    assert _shared_tail_snapshot(bv) == before


def test_a_batch_whose_hosts_cannot_be_restored_reports_rollback_failed_927(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _SharedTailCreateBV(repairable=False)
    marked = []
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "mark_dirty", lambda view: marked.append(view))

    result = instance._mutation(
        "active", True,
        [{"op": "function_create", "address": hex(bv.CUT), "allow_overlap": True}])

    assert result["success"] is False and result["rolled_back"] is False
    assert "may be left modified" in result["message"]
    assert marked == [bv]


def test_the_restore_never_removes_a_function_that_predated_the_create_927(monkeypatch):
    """The repair removes only starts the snapshot did not record: a
    function that already sat inside a host's range is not analysis
    residue, and a repair that dropped it would trade one loss for another."""
    cc = _load_bridge(monkeypatch).create_comments
    host = _HostFunction(0x2000, "outer", [(0x2000, 0x2040)])
    inner = _HostFunction(0x2020, "inner", [(0x2020, 0x2040)])
    bv = _overlap_bv(0x2008, hosts=[host, inner])
    snapshot = cc._snapshot_split_hosts(bv, [{"name": "outer", "address": "0x2000", "offset": "0x8"}])
    assert snapshot["starts"] == {0x2000, 0x2020}
    # Shorten the host by hand and let the restore try: the only candidate
    # start inside the span is `inner`, which predates the create.
    host.set_ranges([(0x2000, 0x2008)])
    assert cc._restore_split_hosts(None, bv, 0x2008, snapshot) is False
    assert bv.get_function_at(0x2020) is inner
    assert not any(e[0].startswith("remove") for e in bv.events if isinstance(e, tuple))

# --- #927 round 3: a sibling rename before an allowed create --------------


def _renamed_host_batch(addr, *extra_ops):
    """`[rename host, create inside it]`, the ordering whose rollback
    resurrected the preview-only name (#927 round 3)."""
    return [
        {"op": "rename_symbol", "identifier": hex(addr - 8), "new_name": "preview_only_name"},
        {"op": "function_create", "address": hex(addr), "allow_overlap": True},
        *extra_ops,
    ]


def test_a_batch_preview_does_not_resurrect_a_siblings_rename_of_the_host_927(monkeypatch):
    """#927 round 3: `[rename host, create inside host] --preview` reported
    `rolled_back: true` with the host still carrying the preview-only name.
    The create's host snapshot was taken AFTER the rename applied; the
    batch undo put the original name back, and the repair then wrote the
    snapshot's (renamed) name over it. The repair owns ranges and auto
    symbols only; the name is the journal's, and the batch verifies it
    against the pre-transaction baseline."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    addr = 0x401341
    bv = _overlap_bv(addr)
    host = bv.get_function_at(addr - 8)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._mutation("active", True, _renamed_host_batch(addr))

    assert result["success"] is True and result["rolled_back"] is True, result["message"]
    assert [r["status"] for r in result["results"]] == ["verified", "verified"]
    assert host.name == "main", "the preview-only name survived the rollback"
    assert host.ranges == [(addr - 8, addr - 8 + 0x40)]
    assert bv.get_function_at(addr) is None


def test_a_failed_live_batch_really_reverts_a_rename_before_an_allowed_create_927(monkeypatch):
    """The live twin: a later op fails, every applied op is rolled back, and
    the rename row's `reverted` must be true of the view -- not a stamp over
    a host the create's repair renamed again."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    addr = 0x401341
    bv = _overlap_bv(addr)
    host = bv.get_function_at(addr - 8)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._mutation("active", False, _renamed_host_batch(
        addr, {"op": "function_create", "address": "0x10"}))

    assert result["success"] is False and result["rolled_back"] is True, result["message"]
    assert [r["status"] for r in result["results"]] == [
        "reverted", "reverted", "invalid_request"]
    assert host.name == "main"
    assert host.ranges == [(addr - 8, addr - 8 + 0x40)]


class _NameStickingBV(_OverlappingCreateBV):
    """A view whose undo does NOT give a renamed host its name back -- the
    shape of a journal that silently failed, which the baseline check must
    catch rather than trust `undo_ok`."""

    def revert_undo_actions(self, state):
        names = {fn: fn.name for fn in self.functions}
        super().revert_undo_actions(state)
        for fn, name in names.items():
            fn.name = name


def test_a_host_left_renamed_after_the_rollback_is_a_failed_rollback_927(monkeypatch):
    """The final check is against the PRE-BATCH baseline: a host whose name
    is not back -- whatever put it there -- is `rollback_failed` with the
    view marked dirty, never `rolled_back: true`."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    addr = 0x401341
    start = addr - 8
    bv = _NameStickingBV(
        hosts=[_HostFunction(start, "main", [(start, start + 0x40)])],
        segments={addr: _FakeSegment(readable=True, executable=True)},
        memory={addr: b"\x55\x48\x89\xe5"},
    )
    marked = []
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "mark_dirty", lambda view: marked.append(view))

    result = instance._mutation("active", True, _renamed_host_batch(addr))

    assert result["success"] is False and result["rolled_back"] is False
    assert "a function a create split is not back" in result["message"]
    assert marked == [bv]
    # The repair did not paper over it either: no name was written.
    assert bv.get_function_at(start).name == "preview_only_name"


class _ImplicitNameHost(_HostFunction):
    """A host named by analysis alone: `fn.symbol` is SYNTHESIZED from the
    current name -- auto for the implicit `sub_*`, user once renamed -- and
    nothing is registered in the view's listing, the shape of every
    function on a stripped target."""

    @property
    def symbol(self):
        return types.SimpleNamespace(auto=self.name.startswith("sub_"), name=self.name,
                                     address=self.start, type="FunctionSymbol")

    @symbol.setter
    def symbol(self, value):
        pass


def test_a_preview_rename_before_create_on_an_implicit_named_host_rolls_back_clean_927(monkeypatch):
    """#927 round 4: on a host named only `sub_<addr>` (no registered
    symbol), `[rename host, create inside it] --preview` reported
    `rollback_failed` over a perfectly restored view. Under the rename the
    listing held only the user symbol (auto set: none); after the undo it
    was empty and the fallback read `fn.symbol`'s synthesized `sub_*` as an
    auto symbol analysis ADDED -- one `undefine_auto_symbol` cannot remove.
    The auto set is the REGISTERED listing on both reads, and the implicit
    name is the baseline name check's business."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    addr = 0x401341
    start = addr - 8
    host = _ImplicitNameHost(start, f"sub_{start:x}", [(start, start + 0x40)])
    bv = _overlap_bv(addr, hosts=[host])
    bv.undefine_auto_symbol = lambda symbol: bv.events.append(("undefine_auto_symbol", symbol.name))
    bv.define_auto_symbol = lambda symbol: bv.events.append(("define_auto_symbol", symbol.name))
    marked = []
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "mark_dirty", lambda view: marked.append(view))

    result = instance._mutation("active", True, _renamed_host_batch(addr))

    assert result["success"] is True and result["rolled_back"] is True, result["message"]
    assert host.name == f"sub_{start:x}"
    assert host.ranges == [(start, start + 0x40)]
    assert bv.get_function_at(addr) is None
    assert not any(e[0].endswith("auto_symbol") for e in bv.events if isinstance(e, tuple))
    assert marked == []


def test_an_exception_path_rollback_that_leaves_a_host_renamed_is_a_structured_failure_927(monkeypatch):
    """#927 round 4: when the post-analysis snapshot raises and the undo
    leaves the host renamed, the baseline check saw it but the engine raised
    a bare RuntimeError -- no status, so the CLI exited 2 as a generic
    error, and the bridge shim (keyed on the prototype residue attribute
    alone) never marked the view dirty. It is the same failed rollback the
    other revert paths report: status `rollback_failed`, `observed` saying
    the view is dirty, and the view marked so `close` warns."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    instance = bridge.BinaryNinjaBridge()
    addr = 0x401341
    start = addr - 8
    bv = _NameStickingBV(
        hosts=[_HostFunction(start, "main", [(start, start + 0x40)])],
        segments={addr: _FakeSegment(readable=True, executable=True)},
        memory={addr: b"\x55\x48\x89\xe5"},
    )
    marked = []
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "mark_dirty", lambda view: marked.append(view))
    real_capture = me._capture_function_snapshots
    calls = []

    def _capture_then_fault(ctx, view, affected):
        calls.append(affected)
        if len(calls) == 2:   # the post-analysis snapshot, after every op applied
            raise RuntimeError("snapshot fault injected after apply")
        return real_capture(ctx, view, affected)

    monkeypatch.setattr(me, "_capture_function_snapshots", _capture_then_fault)

    with pytest.raises(bridge.OperationFailure) as exc_info:
        instance._mutation("active", True, _renamed_host_batch(addr))

    exc = exc_info.value
    assert exc.status == "rollback_failed"
    assert "snapshot fault injected" in exc.message and "rollback failed" in exc.message
    assert exc.observed == {"rolled_back": False, "dirty_after": True}
    assert marked == [bv]
    assert bv.get_function_at(start).name == "preview_only_name"


def test_an_exception_path_rollback_that_restores_the_hosts_is_not_a_failed_rollback_927(monkeypatch):
    """The twin that must NOT fire: the same fault with a working undo
    leaves the view as it was, so the original error escapes untouched --
    no `rollback_failed` status, no dirty mark."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    instance = bridge.BinaryNinjaBridge()
    addr = 0x401341
    bv = _overlap_bv(addr)
    marked = []
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "mark_dirty", lambda view: marked.append(view))
    real_capture = me._capture_function_snapshots
    calls = []

    def _capture_then_fault(ctx, view, affected):
        calls.append(affected)
        if len(calls) == 2:
            raise RuntimeError("snapshot fault injected after apply")
        return real_capture(ctx, view, affected)

    monkeypatch.setattr(me, "_capture_function_snapshots", _capture_then_fault)

    with pytest.raises(RuntimeError, match="snapshot fault injected") as exc_info:
        instance._mutation("active", True, _renamed_host_batch(addr))

    assert not isinstance(exc_info.value, bridge.OperationFailure)
    assert marked == []
    assert bv.get_function_at(addr - 8).name == "main"
    assert bv.get_function_at(addr) is None


def test_the_repair_never_writes_a_user_name_927(monkeypatch):
    """Driven directly: a host whose user name differs from the snapshot
    after the revert is the journal's business. The repair restores ranges
    and reports the ranges as back; it neither assigns the name nor defines
    a symbol, and the standalone baseline check is what fails."""
    cc = _load_bridge(monkeypatch).create_comments
    host = _HostFunction(0x2000, "renamed_by_sibling", [(0x2000, 0x2040)], auto_symbol=False)
    bv = _overlap_bv(0x2008, hosts=[host])
    snapshot = cc._snapshot_split_hosts(bv, [{"name": "renamed_by_sibling", "address": "0x2000",
                                             "offset": "0x8"}])
    assert snapshot["hosts"][0x2000]["auto_symbols"] == {}
    host.name = "original"   # what the undo put back
    assert cc._restore_split_hosts(None, bv, 0x2008, snapshot) is True
    assert host.name == "original"
    assert not any(e[0].endswith("auto_symbol") for e in bv.events if isinstance(e, tuple))
    assert cc._split_hosts_at_baseline(bv, snapshot["hosts"], what="a create") is False
    assert cc._split_hosts_at_baseline(
        bv, {0x2000: {"name": "original", "ranges": ((0x2000, 0x2040),)}}, what="a create") is True


def test_the_repair_restores_the_auto_symbol_set_under_a_user_name_927(monkeypatch):
    """The analysis-owned half: the auto symbols a thunk cut added are
    undefined and the recorded ones redefined -- read from the view's
    listing, not `fn.symbol`, because a user name on top hides them."""
    cc = _load_bridge(monkeypatch).create_comments
    host = _HostFunction(0x2000, "sub_2000", [(0x2000, 0x2040)])
    bv = _overlap_bv(0x2008, hosts=[host])
    loader = types.SimpleNamespace(auto=True, name="sub_2000", address=0x2000, type="FunctionSymbol")
    bv._symbols = [loader]
    events = []
    bv.undefine_auto_symbol = lambda symbol: (events.append(("undefine", symbol.name)),
                                              bv._symbols.remove(symbol))
    bv.define_auto_symbol = lambda symbol: (events.append(("define", symbol.name)),
                                            bv._symbols.append(symbol))
    snapshot = cc._snapshot_split_hosts(bv, [{"name": "sub_2000", "address": "0x2000",
                                             "offset": "0x8"}])
    assert list(snapshot["hosts"][0x2000]["auto_symbols"]) == ["sub_2000"]
    # What the cut did underneath a sibling's user rename: a `j_*` auto
    # symbol replaced the loader one.
    thunk = types.SimpleNamespace(auto=True, name="j_sub_2010", address=0x2000, type="FunctionSymbol")
    bv._symbols = [types.SimpleNamespace(auto=False, name="user_x", address=0x2000,
                                         type="FunctionSymbol"), thunk]

    assert cc._restore_split_hosts(None, bv, 0x2008, snapshot) is True

    assert events == [("undefine", "j_sub_2010"), ("define", "sub_2000")]
    assert [s.name for s in bv._symbols] == ["user_x", "sub_2000"]


# --- #927 P2: a batch create inside an EARLIER batch create ---------------


def test_a_batch_create_inside_an_earlier_create_is_refused_927(monkeypatch):
    """#927 P2: `[create X, create X+4]` without `allow_overlap` exited 0 with
    both rows verified and no overlap -- the second op's guard ran before
    the first create had analyzed ranges, and the post-analysis verifier
    checks existence and code only. The batch now analyzes between two
    creates (gate-only), so the second sees the first and is refused."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    first, second = 0x4011d8, 0x4011dc
    bv = _OverlappingCreateBV(
        hosts=[],
        segments={a: _FakeSegment(readable=True, executable=True) for a in (first, second)},
        memory={a: b"\x55\x48\x89\xe5" for a in (first, second)},
        created_total_bytes=0x60,
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._mutation("active", False, [
        {"op": "function_create", "address": hex(first)},
        {"op": "function_create", "address": hex(second)},
    ])

    assert result["success"] is False and result["committed"] is False
    assert [r["status"] for r in result["results"]] == ["reverted", "invalid_request"]
    refusal = result["results"][1]["message"]
    assert f"sub_{first:x}" in refusal and hex(first) in refusal and "+0x4" in refusal
    assert "allow_overlap" in refusal
    # The whole batch is undone: neither function is live.
    assert bv.get_function_at(first) is None and bv.get_function_at(second) is None


def test_a_batch_create_inside_an_earlier_create_proceeds_with_the_flag_927(monkeypatch):
    """The must-fire twin's permission path: with `allow_overlap` on the
    second op the batch commits and the second row names the FIRST create
    as the host it truncated."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    first, second = 0x4011d8, 0x4011dc
    bv = _OverlappingCreateBV(
        hosts=[],
        segments={a: _FakeSegment(readable=True, executable=True) for a in (first, second)},
        memory={a: b"\x55\x48\x89\xe5" for a in (first, second)},
        created_total_bytes=0x60,
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._mutation("active", False, [
        {"op": "function_create", "address": hex(first)},
        {"op": "function_create", "address": hex(second), "allow_overlap": True},
    ])

    assert result["success"] is True and result["committed"] is True
    assert [r["status"] for r in result["results"]] == ["verified", "verified"]
    assert result["results"][1]["overlaps"] == [
        {"name": f"sub_{first:x}", "address": hex(first), "offset": "0x4",
         "instruction_boundary": True, "truncated": True}]
    assert "overlaps" not in result["results"][0]


def test_a_batch_with_creates_followed_only_by_other_ops_analyzes_once_927(monkeypatch):
    """Must-not-fire twin of the inter-create analysis: only a LATER create
    needs an earlier one's ranges, so a create followed by a comment keeps
    #780's single post-apply analysis."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ops = [{"op": "function_create", "address": "0x1000"},
           {"op": "set_comment", "address": "0x1000", "comment": "x"}]
    assert me._later_create_needs_ranges(ops, 0) is False
    assert me._later_create_needs_ranges(ops, 1) is False
    two = [{"op": "function_create", "address": "0x1000"},
           {"op": "set_comment", "address": "0x1000", "comment": "x"},
           {"op": "function_create", "address": "0x2000"}]
    assert [me._later_create_needs_ranges(two, i) for i in range(3)] == [True, False, False]
    assert me._later_create_needs_ranges(["not-a-dict", {"op": "function_create"}], 0) is False


def _commit_mutation(monkeypatch, instance, bv, ops):
    """Drive the real commit and dirty-mark path against the same fake view."""
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    return instance._mutation("active", False, ops)


def test_all_noop_committed_batch_does_not_dirty_the_view_772(monkeypatch):
    # A committed all-noop batch rolled nothing back and changed nothing; its
    # result must not make close warn about unsaved work.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, fn = _tag_comment_bv()
    fn.comment = "already here"
    marked: list[object] = []
    monkeypatch.setattr(instance.targets, "mark_dirty", lambda b: marked.append(b))

    result = _commit_mutation(monkeypatch, instance, bv, [
        {"op": "set_comment", "function": "handle_request", "comment": "already here"}])

    assert result["committed"] is True
    assert [r["status"] for r in result["results"]] == ["noop"]
    assert marked == [], "an all-noop commit changed nothing and must not dirty the view"


def test_committed_batch_with_a_real_change_still_dirties_the_view_772(monkeypatch):
    # The must-not-fire twin: a genuine committed change must still warn on
    # close if it was not saved.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, fn = _tag_comment_bv()
    fn.comment = "before"
    marked: list[object] = []
    monkeypatch.setattr(instance.targets, "mark_dirty", lambda b: marked.append(b))

    result = _commit_mutation(monkeypatch, instance, bv, [
        {"op": "set_comment", "function": "handle_request", "comment": "after"}])

    assert result["committed"] is True
    assert [r["status"] for r in result["results"]] == ["verified"]
    assert marked, "a committed batch that really changed state must dirty the view"


def test_idempotent_tag_add_reports_noop_and_does_not_duplicate_779(monkeypatch):
    # #779: BN's add_tag does not dedupe -- `_find_new_tag` exists precisely
    # because every call mints a fresh Tag id -- so running the same add twice
    # genuinely produced two identical tags and reported `verified` both times.
    # The second add must now write nothing and report `noop`.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, fn = _tag_comment_bv()

    op = {"op": "tag_add", "function": "handle_request", "type": "Bug", "data": "look here"}
    first = _commit_mutation(monkeypatch, instance, bv, [dict(op)])
    assert first["results"][0]["status"] == "verified", first["results"][0]
    assert _fn_tags(fn) == [("Bug", "look here")]

    second = _commit_mutation(monkeypatch, instance, bv, [dict(op)])
    assert second["results"][0]["status"] == "noop", second["results"][0]
    # THE defect: the tag list must not have grown.
    assert _fn_tags(fn) == [("Bug", "look here")]


def test_tag_add_with_different_data_is_still_a_real_add_779(monkeypatch):
    # The must-not-fire twin: only an exact (type, data) match is a noop. A
    # second tag of the same TYPE carrying different data is a genuine add, or
    # the dedupe would swallow real annotations.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, fn = _tag_comment_bv()
    fn.add_tag("Bug", "pre-existing", None)

    result = _commit_mutation(monkeypatch, instance, bv, [
        {"op": "tag_add", "function": "handle_request", "type": "Bug", "data": "look here"}])
    assert result["results"][0]["status"] == "verified", result["results"][0]
    assert sorted(_fn_tags(fn)) == [("Bug", "look here"), ("Bug", "pre-existing")]


def test_set_comment_refuses_an_unmapped_address_781(monkeypatch):
    # #781: an unmapped address used to be accepted here -- the comment was
    # written into nowhere and then reported `verified` against its own
    # readback -- while `comment get` and `data retype` both reject the
    # identical address. `invalid_request` matches data_retype's precedent for
    # the same condition: a bad request, not an unsupported operation.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _fn = _tag_comment_bv()
    bv.is_valid_offset = lambda addr: False

    result = _commit_mutation(monkeypatch, instance, bv, [
        {"op": "set_comment", "address": "0xdead0000", "comment": "note"}])
    row = result["results"][0]
    assert row["status"] == "invalid_request", row
    assert "not mapped" in row.get("message", "")


def test_set_comment_on_an_indeterminate_view_still_works_781(monkeypatch):
    # Only an AFFIRMATIVE "unmapped" is refused. A view that cannot answer --
    # no `is_valid_offset` at all -- is indeterminate, not invalid, and must
    # keep working; otherwise the guard breaks every reduced/fake shape.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _fn = _tag_comment_bv()
    monkeypatch.delattr(_FakeBV, "is_valid_offset")
    assert not hasattr(bv, "is_valid_offset")

    result = _commit_mutation(monkeypatch, instance, bv, [
        {"op": "set_comment", "address": "0x1000", "comment": "note"}])
    assert result["results"][0]["status"] == "verified", result["results"][0]


def test_unmapped_address_guard_is_one_helper_serving_both_mutations_890():
    # #890 obs 1: the comment guard shipped as a SECOND TEXTUAL COPY of the
    # data-retype guard -- the exact drift shape #777 fixed elsewhere in this
    # same branch. Both callsites now route through one helper, so the
    # tri-state cannot be hardened on one path and left alone on the other.
    import inspect
    from bn_agent_bridge import mutation_engine as me

    source = inspect.getsource(me)
    # The probe is read in exactly one place: the helper.
    assert source.count('getattr(bv, "is_valid_offset", None)') == 1, (
        "the affirmative-unmapped probe is duplicated again; route the new "
        "callsite through _refuse_unmapped_mutation_address instead")
    for fn in (me._op_set_comment, me._op_data_retype):
        assert "_refuse_unmapped_mutation_address" in inspect.getsource(fn), fn


def test_data_retype_refuses_an_unmapped_address_through_the_shared_guard_890(monkeypatch):
    # The data-retype half of the shared guard, exercised for real rather than
    # inferred from the comment half sharing a helper with it.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _fn = _tag_comment_bv()
    bv.is_valid_offset = lambda addr: False

    result = _commit_mutation(monkeypatch, instance, bv, [
        {"op": "data_retype", "address": "0xdead0000", "new_type": "int32_t"}])
    row = result["results"][0]
    assert row["status"] == "invalid_request", row
    assert "not mapped" in row.get("message", "")
    # The consequence clause is the callsite's, not the helper's.
    assert "data variable" in row.get("message", "")


def test_a_raising_mapped_probe_is_indeterminate_not_invalid_890(monkeypatch):
    # The third state, and the one no test covered: a probe that RAISES is
    # indeterminate, exactly like a view that does not implement it. Treating
    # a raising probe as "unmapped" would refuse good requests on any view
    # whose offset check throws.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _fn = _tag_comment_bv()

    def _boom(addr):
        raise RuntimeError("offset probe unavailable")

    bv.is_valid_offset = _boom

    result = _commit_mutation(monkeypatch, instance, bv, [
        {"op": "set_comment", "address": "0x1000", "comment": "note"}])
    assert result["results"][0]["status"] == "verified", result["results"][0]


def test_batch_revert_gate_reads_the_shared_failure_status_set_777():
    # #777: `_has_failed_results` carried its own inline
    # {"unsupported", "verification_failed"} literal -- a strict subset of the
    # canonical five -- so a verifier raising `invalid_request`,
    # `rollback_failed` or `internal_error` would have left a FAILED batch
    # un-reverted while the CLI still reported exit 3. Both sides now read one
    # definition, so the drift cannot reopen.
    from pathlib import Path

    from bn import mutation_statuses as cli_mod
    from bn_agent_bridge import mutation_engine as me
    from bn_agent_bridge import mutation_statuses as bridge_mod
    from bn_agent_bridge.mutation_statuses import FAILED_MUTATION_STATUSES
    from bn.formatters import FAILED_MUTATION_STATUSES as cli_set

    # ONE source file, reached from both packages. Not an `is` check: the
    # symlink shares the SOURCE, so each package imports its own module object
    # (exactly as paths.py/version.py already do) -- resolving __file__ is what
    # actually proves there is no second copy to drift.
    assert Path(cli_mod.__file__).resolve() == Path(bridge_mod.__file__).resolve()
    assert cli_set == FAILED_MUTATION_STATUSES
    # ...and `formatters` re-exports rather than redeclaring, so the documented
    # public name cannot fall behind the definition.
    assert cli_set is cli_mod.FAILED_MUTATION_STATUSES
    for status in sorted(FAILED_MUTATION_STATUSES):
        assert me._has_failed_results(None, [{"status": status}]) is True, status
    for status in ("verified", "noop", "reverted"):
        assert me._has_failed_results(None, [{"status": status}]) is False, status
