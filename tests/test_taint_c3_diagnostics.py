"""C3 closure: diagnostics and refusal text that must say what the result shows.

- V4-02: ``diagnostics.propagated`` is false when the only frontier leaves are
  kinds the seed step emits on its own (a seed nothing reads, ...). Every leaf
  kind the engine names is classified here, so a new kind that is not fails.
- V4-04: the call-seed refusal's ``--resolve-map`` hint is the flag's real
  form: a FILE holding ``{"<call address>": ["TARGET"]}``.
- W07-05: the process entry stub handing ``main`` to an imported C-runtime
  start routine is a ``crt_entry`` caller row, not an ``indirect_reference``,
  and it adds no dispatching-caller advice: ``next_action`` is the run's other
  frontier advice.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
from test_taint_backward_coverage import (
    _ENTRY_NAMES,
    _LIBC_START_SLOT,
    _NAMES,
    _advice_commands,
    _backward,
    _caller_leaves,
    _entry,
    _gate,
    _main_fn,
)
from test_taint_caller_advice import _outer_program
from test_taint_engine import FBV, FExpr, FFunc, FInstr, FSSAFunc, _indirect_sink_program, te

from bn.taint_vocabulary import BLOCKING_LEAF_KINDS
from bn_agent_bridge import taint_result
from bn_agent_bridge.taint_result import (
    backward_diagnostics,
    forward_zero_diagnostics,
)

_SRC = Path(__file__).resolve().parent.parent / "src" / "bn_agent_bridge"


# --------------------------------------------------------------------------
# V4-02: which leaf kinds show the seed reached something
# --------------------------------------------------------------------------

# Every blocking leaf kind, classified: "reached" -- the leaf exists only
# because tainted data got there; "seed" -- the seed step (or a scan that runs
# whether or not taint arrived) can emit it on its own, so it proves no
# propagation; "backward" -- a backward-only kind, never in a forward result.
_LEAF_CLASS = {
    "coarse_memory_store": "reached",
    "gated_sink_reached": "reached",
    "pointer_escape": "reached",
    "indirect_call_unresolved": "reached",
    "unmodeled_callee": "reached",
    "arg_under_recovered": "reached",
    "unlifted_instruction_reached": "reached",
    "model_destination_unrecovered": "reached",
    "global_memory_continuation": "reached",
    "unread_callee_body": "reached",
    "field_load_unresolved": "reached",
    "seed_unused": "seed",
    "source_seed_misanchored": "seed",
    "indirect_call_unchecked": "seed",
    "format_destination_unrecovered": "seed",
    "scanf_arity_residual": "seed",
    "format_unchecked": "seed",
    "abi_provenance_budget": "seed",
    "analysis_truncated": "seed",
    "caller_sites_truncated": "backward",
    "address_of_local_unresolved": "backward",
    "pointer_object_unresolved": "backward",
    "object_extent_unknown": "backward",
    "region_store_proof_not_certified": "backward",
    "pointer_demand_not_certified": "backward",
    "sink_role_unknown": "backward",
    "sink_unseeded": "backward",
    "abi_mapping_unbound": "backward",
}

# `"kind": "<x>"` literals the engine uses for things that are not leaves:
# origins, objects a pointer demand may read, slice steps and model rows.
_NON_LEAF_KINDS = {
    "call", "constant", "entry", "global", "got_or_extern", "indirect_call",
    "loaded_pointer", "local", "param", "parameter", "returned", "source",
    "unknown", "unresolved",
}


def _kind_values(expr: ast.AST, consts: dict[str, str]) -> set[str]:
    """The strings a `kind` value expression can evaluate to: a literal, a
    module constant, either arm of a conditional or boolean -- never the
    strings a condition merely compares against."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return {expr.value}
    if isinstance(expr, ast.Name):
        return {consts[expr.id]} if expr.id in consts else set()
    if isinstance(expr, ast.IfExp):
        return _kind_values(expr.body, consts) | _kind_values(expr.orelse, consts)
    if isinstance(expr, ast.BoolOp):
        return set().union(*(_kind_values(v, consts) for v in expr.values))
    return set()


def _engine_kind_literals() -> set[str]:
    """Every kind the engine CONSTRUCTS: the value of a `"kind": ...` dict
    entry or a `kind=` argument, through module constants
    (`_EXTENT_LEAF = ...`) and conditional arms (`"a" if p else "b"`).
    A kind named only in a classification set is not constructed (R4-05)."""
    found: set[str] = set()
    for name in ("taint_engine.py", "taint_result.py"):
        tree = ast.parse((_SRC / name).read_text())
        consts = {node.targets[0].id: node.value.value for node in tree.body
                  if isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name)
                  and re.fullmatch(r"_[A-Z_]+", node.targets[0].id)
                  and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                values = [v for k, v in zip(node.keys, node.values)
                          if isinstance(k, ast.Constant) and k.value == "kind"]
            elif isinstance(node, ast.keyword) and node.arg == "kind":
                values = [node.value]
            else:
                continue
            for value in values:
                found |= _kind_values(value, consts)
    return found


def test_every_engine_leaf_kind_is_classified_for_propagation():
    literals = _engine_kind_literals()
    unregistered = sorted(literals - BLOCKING_LEAF_KINDS - _NON_LEAF_KINDS)
    assert unregistered == [], unregistered
    # R4-05: and every registered kind has a constructor -- a kind kept only in
    # the classification sets (and so in the docs the drift test holds to
    # them) is a frontier no run can produce.
    unconstructed = sorted(BLOCKING_LEAF_KINDS - literals)
    assert unconstructed == [], unconstructed
    assert set(_LEAF_CLASS) == set(BLOCKING_LEAF_KINDS), (
        sorted(set(BLOCKING_LEAF_KINDS) ^ set(_LEAF_CLASS)))
    seed_side = {k for k, c in _LEAF_CLASS.items() if c == "seed"}
    assert set(taint_result._NO_PROPAGATION_KINDS) == seed_side


@pytest.mark.parametrize("kind", sorted(k for k, c in _LEAF_CLASS.items() if c != "backward"))
def test_propagated_is_read_off_the_leaf_kind(kind):
    # A finished walk that produced only the seed value and recorded no use:
    # the leaf alone decides whether the result shows propagation.
    leaf = {"kind": kind, "address": "0x401000",
            "global": {"address": "0x404040", "label": "g_state"}}
    sub = {"diag": {"tainted_values": 1, "last_use": None},
           "leaves": [leaf], "assumptions": []}
    diag = forward_zero_diagnostics(sub, seed_callsites=1)
    assert diag["propagated"] is (_LEAF_CLASS[kind] == "reached"), kind
    assert diag["frontier"]["by_kind"] == {kind: 1}
    assert diag["safe_to_report_all_clear"] is False


def test_dead_seed_run_does_not_claim_propagation():
    # handler(a, b) reads only b; seeding a follows nothing.
    from test_taint_claim_gaps import FGlobalBV, _forward, _two_params

    result = _forward(FGlobalBV({}), _two_params(read_index=1), te.load_models(),
                      source="param:0")
    diag = result["diagnostics"]
    assert [lf["kind"] for lf in result["leaves"]] == ["seed_unused"]
    assert (diag["tainted_values"], diag["last_use"]) == (1, None)
    assert diag["propagated"] is False
    assert diag["safe_to_report_all_clear"] is False


# --------------------------------------------------------------------------
# V4-04: the --resolve-map hint names a FILE and prints its document
# --------------------------------------------------------------------------

def _resolve_map_hint(message: str) -> dict:
    # Every `--resolve-map` the refusal prints is followed by the FILE
    # placeholder, never an inline `<call>=<target>` pair the flag refuses.
    operands = re.findall(r"--resolve-map (\S+)", message)
    assert "FILE," in operands, operands
    assert not any(("=" in op or "<" in op or op.startswith("0x")) for op in operands), operands
    docs = [json.loads(m) for m in re.findall(r"\{[^{}]*\}", message)]
    assert len(docs) == 1, message
    return docs[0]


def test_refusal_with_only_indirect_calls_prints_the_resolve_map_document():
    func = _indirect_sink_program(with_pvs=False)
    with pytest.raises(te.TaintError) as err:
        te.TaintEngine(FBV({}), te.load_models()).backward(func, [te.parse_locator("arg:send:1")])
    doc = _resolve_map_hint(str(err.value))
    assert doc == {"0x14": ["TARGET"]}
    # The document is what the flag reads: filled in, the engine loads it and
    # the pinned call seeds.
    filled = {call: ["0x900"] for call in doc}
    bv = FBV({0x900: "send"})
    result = te.TaintEngine(bv, te.load_models(), resolve_map=filled).backward(
        func, [te.parse_locator("arg:send:1")])
    assert result["slices"]


def test_refusal_naming_direct_callers_prints_the_resolve_map_document():
    bv, outer = _outer_program()
    models = dict(te.load_models(), pull={"sources": [{"to": "*arg:1"}, {"to": "ret"}]})
    with pytest.raises(te.TaintError) as err:
        te.TaintEngine(bv, models).forward(outer, [te.parse_locator("call:pull")])
    assert re.findall(r"-f (0x[0-9a-f]+)", str(err.value)) == ["0x2000", "0x1500"]
    assert _resolve_map_hint(str(err.value)) == {"0x6004": ["TARGET"]}



# --------------------------------------------------------------------------
# W07-05: the C runtime entry stub's handoff of main is `crt_entry`
# --------------------------------------------------------------------------

def _start_handoff(**kw):
    start = _entry("_start", [(_LIBC_START_SLOT, True)])
    return _backward(_main_fn(41), {**_NAMES, **_ENTRY_NAMES, 0x800: "main"},
                     code_refs={0x800: [(0x41e, start)]}, entry=start, **kw)


def test_entry_stub_handing_main_to_the_imported_runtime_is_a_crt_entry_row():
    result = _start_handoff()
    (leaf,) = _caller_leaves(result)
    assert leaf["reasons"] == {"crt_entry": 1}
    assert leaf["unfollowed"] == [{"address": "0x41e", "caller": "_start",
                                  "param": 0, "reason": "crt_entry"}]
    assert (leaf["callers_total"], leaf["callers_dropped"]) == (1, 1)
    # Still blocking: what the runtime passes is not proven.
    assert _gate(result) is False
    assert [(o["kind"], o.get("index")) for sl in result["slices"] for o in sl["origins"]] == [
        ("parameter", 0)]
    # No caller in the binary to slice from: no xrefs/callsites advice, and
    # next_action is what the same result says without the caller row.
    advice = result["diagnostics"]["next_action"]
    assert _advice_commands(advice, "xrefs") == [] and _advice_commands(advice, "callsites") == []
    rest = [lf for lf in result["leaves"] if lf["kind"] != "caller_sites_truncated"]
    alone = backward_diagnostics(rest, result["assumptions"], sinks_seeded=1,
                                 slices=len(result["slices"]))
    assert advice == alone["next_action"]


@pytest.mark.parametrize("variant", ["in-binary", "second-use", "operand-1", "not-entry"])
def test_unproven_entry_handoffs_stay_indirect_references(variant):
    calls = {"in-binary": [(_LIBC_START_SLOT, False)],
             "operand-1": [(_LIBC_START_SLOT, True, 1)]}.get(variant, [(_LIBC_START_SLOT, True)])
    start = _entry("_start", calls)
    if variant == "second-use":
        start.mlil.ssa_form.instructions.append(FInstr(
            1, 0x420, "MLIL_STORE_SSA", "save callback",
            dest=FExpr("MLIL_CONST_PTR", "slot", constant=0x4000),
            src=FExpr("MLIL_CONST_PTR", "main", constant=0x800)))
    view_entry = FFunc("other_entry", 0x300, FSSAFunc([])) if variant == "not-entry" else start
    result = _backward(_main_fn(41), {**_NAMES, **_ENTRY_NAMES, 0x800: "main"},
                       code_refs={0x800: [(0x41e, start)]}, entry=view_entry)
    (leaf,) = _caller_leaves(result)
    assert leaf["reasons"] == {"indirect_reference": 1}
    assert _gate(result) is False
    assert [c[c.index("xrefs") + 1] for c in _advice_commands(
        result["diagnostics"]["next_action"], "xrefs")] == ["main"]


def _caller_leaf(name, address, reasons):
    n = sum(reasons.values())
    return {"kind": "caller_sites_truncated", "address": address,
            "function": {"name": name, "display_name": name, "address": address},
            "callers_total": n, "callers_fully_followed": 0, "callers_partially_followed": 0,
            "callers_unfollowed": n, "callers_dropped": n, "reasons": reasons}


def test_crt_entry_rows_add_no_caller_advice_beside_other_reasons():
    frontier = [{"kind": "field_load_unresolved", "address": "0x4010"},
                {"kind": "pointer_demand_not_certified", "address": "0x4020",
                 "function": {"name": "main", "address": "0x4000"}}]
    base = backward_diagnostics(frontier, [], sinks_seeded=1, slices=1)
    crt = backward_diagnostics([*frontier, _caller_leaf("main", "0x4000", {"crt_entry": 1})],
                               [], sinks_seeded=1, slices=1)
    assert crt["next_action"] == base["next_action"]
    assert crt["safe_to_report_complete_slice"] is False
    assert crt["frontier"]["dropped_callers"] == 1

    # Nothing else left: the origins-reached advice, still not complete.
    only = backward_diagnostics([_caller_leaf("main", "0x4000", {"crt_entry": 1})], [],
                                sinks_seeded=1, slices=1)
    assert only["next_action"] == backward_diagnostics([], [], sinks_seeded=1,
                                                       slices=1)["next_action"]
    assert only["safe_to_report_complete_slice"] is False

    # Another function's indirect reference keeps its own xrefs advice; the
    # crt_entry function is never named in it.
    mixed = backward_diagnostics(
        [_caller_leaf("main", "0x4000", {"crt_entry": 1}),
         _caller_leaf("handler", "0x4100", {"indirect_reference": 1})],
        [], sinks_seeded=1, slices=1)
    named = [c[c.index("xrefs") + 1] for c in _advice_commands(mixed["next_action"], "xrefs")]
    assert named == ["handler"]
    assert _advice_commands(mixed["next_action"], "callsites") == []
