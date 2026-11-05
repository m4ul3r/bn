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
from test_taint_truncation_vocabulary import _bound_names, _Crawler, _Unresolved

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
    "call", "constant", "constant_address", "entry", "global", "got_or_extern",
    "indirect_call", "literal", "loaded_pointer", "local", "param", "parameter",
    "recurrence_initializer",
    "returned", "source", "unknown", "unresolved",
}


class _KindCrawler(_Crawler):
    """The truncation crawler's reduction (literals, conditionals, names
    bound to them, dict tables) for the value a ``"kind"`` entry or a
    ``kind=`` argument holds, plus the shapes leaf constructors use:

    * either operand of a boolean (`a or "fallback"`);
    * a parameter: its default and every argument a call to its function in
      the module passes it (by keyword, or by position when it may be
      positional) -- `_buffer_leaf(..., kind="...")`;
    * a value read out of another object (`w[0]`, `sink.get("kind")`): a
      COPY of a kind constructed elsewhere, which this crawl reads there.

    Anything else raises :class:`_Unresolved`, which the guard reports with
    the call site, as the truncation crawler does -- a constructor skipped
    silently is exactly how an unregistered kind escapes (D5c drift)."""

    def __init__(self, tree: ast.AST) -> None:
        super().__init__(tree)
        self.tree = tree
        # (function, parameter) pairs being reduced: a function forwarding
        # its own parameter to itself adds nothing new.
        self.reducing: set[tuple[int, str]] = set()

    def strings(self, expr: ast.expr) -> set[str]:
        if isinstance(expr, ast.BoolOp):
            return set().union(*(self.strings(v) for v in expr.values))
        if isinstance(expr, ast.Name):
            fn = self.parameter_of(expr)
            if fn is not None:
                return self.arguments(fn, expr.id)
        if isinstance(expr, ast.Subscript) and not self.is_table(expr.value):
            return set()
        if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute)
                and expr.func.attr == "get" and not self.is_table(expr.func.value)):
            return set()
        return super().strings(expr)

    def is_table(self, expr: ast.expr) -> bool:
        try:
            self.dict_literal(expr)
        except _Unresolved:
            return False
        return True

    def parameter_of(self, name: ast.Name) -> ast.AST | None:
        """The function whose parameter *name* is, when the nearest scope
        that binds it binds it as one."""
        for scope in self.scopes(name):
            args = getattr(scope, "args", None)
            if args is not None and name.id in {
                    a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}:
                return scope if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
            if any(isinstance(n, (ast.Assign, ast.For, ast.comprehension))
                   and self.owner(n) is scope
                   and name.id in _bound_names(n.targets[0] if isinstance(n, ast.Assign)
                                               else n.target)
                   for n in ast.walk(scope)):
                return None
        return None

    def arguments(self, fn: ast.AST, param: str) -> set[str]:
        key = (id(fn), param)
        if key in self.reducing:
            return set()
        self.reducing.add(key)
        args = fn.args
        positional = [a.arg for a in (*args.posonlyargs, *args.args)]
        defaults = dict(zip(positional[len(positional) - len(args.defaults):], args.defaults))
        defaults.update({a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults) if d})
        out = self.strings(defaults[param]) if param in defaults else set()
        index = positional.index(param) if param in positional else None
        for call in ast.walk(self.tree):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == fn.name):
                continue
            for kw in call.keywords:
                if kw.arg == param:
                    out |= self.strings(kw.value)
            if index is not None and index < len(call.args):
                out |= self.strings(call.args[index])
        return out


def _kind_literals(path: Path) -> set[str]:
    """Every kind *path* CONSTRUCTS: the value of a `"kind": ...` dict entry or
    a `kind=` argument. A value the crawler cannot reduce fails naming its
    site; a kind named only in a classification set is not constructed
    (R4-05)."""
    tree = ast.parse(path.read_text())
    crawler = _KindCrawler(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            values = [v for k, v in zip(node.keys, node.values)
                      if isinstance(k, ast.Constant) and k.value == "kind"]
        elif isinstance(node, ast.keyword) and node.arg == "kind":
            values = [node.value]
        else:
            continue
        for value in values:
            try:
                found |= crawler.strings(value)
            except _Unresolved as exc:
                pytest.fail(f"{path.name}:{value.lineno}: cannot tell which kind "
                            f"`{ast.unparse(value)}` constructs ({exc}); pass a literal or a "
                            "name bound to literals, or teach this crawler the shape")
    return found


def _engine_kind_literals() -> set[str]:
    return _kind_literals(_SRC / "taint_engine.py") | _kind_literals(_SRC / "taint_result.py")


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


_CONSTRUCTORS = '''
_EXTENT = "from_module_constant"
def build(at, *, kind="from_default"):
    return {"kind": kind, "address": at}
def run(p, rows):
    build(1, kind="from_keyword")
    local = "from_local" if p else "from_arm"
    leaves = [{"kind": local}, {"kind": _EXTENT}, {"kind": p or "from_bool"}]
    for row in rows:
        leaves.append({"kind": row["kind"]})
    return leaves
'''


def test_the_kind_crawler_follows_every_constructor_shape(tmp_path):
    module = tmp_path / "engine.py"
    module.write_text(_CONSTRUCTORS)
    assert _kind_literals(module) == {
        "from_module_constant", "from_default", "from_keyword", "from_local", "from_arm",
        "from_bool"}


def test_the_kind_crawler_refuses_a_kind_it_cannot_follow_naming_the_site(tmp_path):
    # A kind computed by a call silently contributed nothing, so a new leaf
    # kind built that way escaped both the registry and the docs.
    module = tmp_path / "engine.py"
    module.write_text("def run(self):\n    return {'address': 1,\n"
                      "            'kind': self.pick_kind()}\n")
    with pytest.raises(pytest.fail.Exception, match=r"engine\.py:3: .*pick_kind"):
        _kind_literals(module)


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
    # next_action names the runtime boundary instead (R2).
    advice = result["diagnostics"]["next_action"]
    assert _advice_commands(advice, "xrefs") == [] and _advice_commands(advice, "callsites") == []
    assert "main" in advice and "C runtime" in advice, advice


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

    # Nothing else left: the slice reached its origins, but `main` is entered
    # from the C runtime, so the advice names that boundary -- not the
    # complete-run wording of a slice with no frontier at all.
    only = backward_diagnostics([_caller_leaf("main", "0x4000", {"crt_entry": 1})], [],
                                sinks_seeded=1, slices=1)
    plain = backward_diagnostics([], [], sinks_seeded=1, slices=1)["next_action"]
    assert only["next_action"] != plain
    assert "main" in only["next_action"] and "C runtime" in only["next_action"]
    assert _advice_commands(only["next_action"], "callsites") == []
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
