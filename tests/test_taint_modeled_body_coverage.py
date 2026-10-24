"""A model never certifies a callee body Binary Ninja did not read (IC05).

Forward taint descends a call when the target is an in-binary BODY, and
applies a model when one matches the name -- two independent decisions
(#807). The argument-descent path kept a third, hidden dependency between
them: the accounting for an in-binary body Binary Ninja MEASURED and declined
to analyse (an oversized function past ``analysis.limits.maxFunctionSize``)
lived inside the branch that runs only when nothing in the call chain is
modeled. So a matching entry -- a truthful ``arg:0 -> ret``, an empty ``{}``,
a reflexive rule that states nothing at all -- erased
``stats.analysis_incomplete`` and turned a withheld all-clear into a clean
one: accurate positive evidence made the answer LESS conservative (BN6-C008;
BN21-REACH-01 settled the same question for the tainted-GLOBAL continuation
in IC03, which left this path alone).

Body coverage is a property of the binary. These tests pin it as independent
of the model DB on every route a call takes to a skipped body -- direct, a
veneer chain, an analysed wrapper, an agent-pinned indirect site -- while the
model's own propagation keeps working, an IMPORT (which has no in-binary
implementation) stays complete, and a fully analysed modeled body is still
descended.
"""
from __future__ import annotations

import re
import shlex
import types

import pytest
from test_taint_engine import FBV, FExpr, FFunc, FInstr, FPVS, FSSA, FSSAFunc, FVar, te

from bn import formatters
from bn_agent_bridge.taint_result import BLOCKING_LEAF_KINDS

KIND = "unread_callee_body"

ENTRY = 0x401000
CALL = 0x401004
SINK_CALL = 0x401008
BODY = 0x402000          # the oversized body BN measured and skipped
VENEER = 0x403000        # j_unread_body
VENEER2 = 0x403100       # j_j_unread_body
WRAPPER = 0x404000       # an analysed wrapper that calls the body
ANALYSED = 0x405000      # a fully analysed body that reaches a sink
IMPORTED = 0x406000      # a genuine import: no in-binary implementation
SYSTEM = 0x401070
MEMCPY = 0x401080

PARTIAL = {"propagates": [{"from": "arg:0", "to": "ret"}]}
IRRELEVANT = {"propagates": [{"from": "arg:0", "to": "arg:0"}]}
OVERLAYS = {
    "no-model": None,
    "empty-entry": {},
    "partial": PARTIAL,
    "irrelevant": IRRELEVANT,
}


@pytest.fixture
def models():
    return te.load_models()


def _ptr(addr):
    return FExpr("MLIL_CONST_PTR", hex(addr), constant=addr)


def _val(ssa_var):
    return FExpr("MLIL_VAR_SSA", str(ssa_var), reads=[ssa_var], src=ssa_var)


def _skipped(name, addr, reason="ExceedFunctionSizeSkipReason"):
    """A real in-binary FunctionSymbol BN measured and refused to analyse.

    Binary Ninja reports the skip itself: the function exists, its MLIL does
    not. This is the shape of the retained 70,004-byte public fixture
    (``analysis_skipped`` with ``ExceedFunctionSizeSkipReason``), not an
    import and not an address with no function at all.
    """
    fn = FFunc(name, addr, FSSAFunc([]), params=[FVar("q", ident=90)],
               symbol_type="FunctionSymbol")
    fn.analysis_skipped = True
    fn.analysis_skip_reason = types.SimpleNamespace(name=reason)
    return fn


def _veneer(name, addr, target):
    """A one-instruction tail-call thunk onto *target* -- what BN names
    ``j_<callee>`` for a jump stub."""
    fn = FFunc(name, addr, FSSAFunc([
        FInstr(0, addr, "MLIL_TAILCALL_SSA", f"jump {hex(target)}",
               reads=[], writes=[], dest=_ptr(target), params=[])]))
    fn.is_thunk = True
    return fn


def _calls(name, addr, target, *, param_ident=91):
    """An ordinary analysed body that forwards its parameter to *target*."""
    q = FVar("q", ident=param_ident)
    q0 = FSSA(q, 0)
    return FFunc(name, addr, FSSAFunc([
        FInstr(0, addr + 4, "MLIL_CALL_SSA", f"{hex(target)}(q#0)",
               reads=[q0], writes=[], dest=_ptr(target), params=[_val(q0)])]),
        params=[q])


def _program(route, overlay, *, consume_return=False):
    """``call_unread(p)`` reaching an unread body by *route*.

    Returns ``(entry, bv, models, resolve_map)``. With ``consume_return`` the
    caller hands the call's RESULT to a modeled sink, which is how a model's
    own return semantics are observed to survive the disclosure.
    """
    p, result = FVar("p", ident=40), FVar("result", ident=41)
    p0, result1 = FSSA(p, 0), FSSA(result, 1)
    body = _skipped("unread_body", BODY)
    funcs = {BODY: body}
    names = {BODY: "unread_body", SYSTEM: "system", MEMCPY: "memcpy"}
    resolve_map: dict = {}
    target = BODY
    dest = _ptr(BODY)

    if route in ("thunk", "multihop"):
        funcs[VENEER] = _veneer("j_unread_body", VENEER, BODY)
        names[VENEER] = "j_unread_body"
        target = VENEER
    if route == "multihop":
        funcs[VENEER2] = _veneer("j_j_unread_body", VENEER2, VENEER)
        names[VENEER2] = "j_j_unread_body"
        target = VENEER2
    if route == "wrapper":
        funcs[WRAPPER] = _calls("wrap_unread", WRAPPER, BODY)
        names[WRAPPER] = "wrap_unread"
        target = WRAPPER
    if route == "import":
        funcs[IMPORTED] = FFunc("memset", IMPORTED, FSSAFunc([]),
                                symbol_type="ImportedFunctionSymbol")
        names[IMPORTED] = "memset"
        target = IMPORTED
    if route == "analysed":
        q = FVar("q", ident=92)
        q0 = FSSA(q, 0)
        funcs[ANALYSED] = FFunc("analysed_body", ANALYSED, FSSAFunc([
            FInstr(0, ANALYSED + 4, "MLIL_CALL_SSA", f"{hex(SYSTEM)}(q#0)",
                   reads=[q0], writes=[], dest=_ptr(SYSTEM), params=[_val(q0)])]),
            params=[q])
        names[ANALYSED] = "analysed_body"
        target = ANALYSED
    if route == "pinned":
        # An indirect site the value set cannot pin, resolved by --resolve-map.
        dest = FExpr("MLIL_VAR_SSA", "hook#1", reads=[], possible_values=FPVS("Undetermined"))
        resolve_map = {hex(CALL): [hex(BODY)]}
        target = None
    else:
        dest = _ptr(target)

    instrs = [FInstr(0, CALL, "MLIL_CALL_SSA", "result = callee(p#0)",
                     reads=[p0], writes=[result1], dest=dest, params=[_val(p0)])]
    if consume_return:
        instrs.append(FInstr(
            1, SINK_CALL, "MLIL_CALL_SSA", "memcpy(dst, src, result#1)",
            reads=[result1], writes=[], dest=_ptr(MEMCPY),
            params=[_ptr(0x407000), _ptr(0x408000), _val(result1)]))
    entry = FFunc("call_unread", ENTRY, FSSAFunc(instrs), params=[p])
    catalog = te.load_models({"unread_body": overlay} if overlay is not None else None)
    return entry, FBV(names, funcs=funcs), catalog, resolve_map


def _forward(route, overlay, *, consume_return=False, max_depth=8, **engine_kwargs):
    entry, bv, catalog, resolve_map = _program(route, overlay,
                                               consume_return=consume_return)
    engine = te.TaintEngine(bv, catalog, resolve_map=resolve_map, **engine_kwargs)
    return engine.forward(entry, [te.parse_locator("param:0")], max_depth=max_depth)


def _boundaries(result):
    return [lf for lf in result["leaves"] if lf.get("kind") == KIND]


# --------------------------------------------------------------------------
# the defect: a matching model used to buy off the unread body
# --------------------------------------------------------------------------

@pytest.mark.parametrize("route", ["direct", "thunk", "multihop", "wrapper", "pinned"])
@pytest.mark.parametrize("overlay", list(OVERLAYS), ids=list(OVERLAYS))
def test_a_matching_model_never_certifies_an_unread_body(route, overlay):
    # BN6-C008: with ANY matching entry the run reported
    # analysis_incomplete=false, zero leaves and safe_to_report_all_clear=true
    # -- on the identical binary whose unmodeled run withheld the claim. The
    # overlay states one relation; it never establishes that the rest of the
    # body was read.
    result = _forward(route, OVERLAYS[overlay])
    stats, diag = result["stats"], result["diagnostics"]

    assert stats["analysis_incomplete"] is True, (route, overlay, stats)
    # The name is the BODY's, on every route: a veneer is not the function an
    # agent has to analyse.
    assert stats["analysis_incomplete_functions"] == ["unread_body"], stats
    assert diag["analysis_incomplete"] is True, diag
    assert diag["safe_to_report_all_clear"] is False, diag

    [leaf] = _boundaries(result)
    assert leaf["callee"]["name"] == "unread_body", leaf
    assert leaf["callee"]["address"] == hex(BODY), leaf
    # The call site is the one that enters the unread body: for the wrapper
    # route that is the wrapper's own call, not the entry's.
    assert leaf["address"] == hex(WRAPPER + 4 if route == "wrapper" else CALL), leaf
    assert leaf["skip_reason"] == "analysis_skipped:ExceedFunctionSizeSkipReason", leaf
    assert leaf["tainted_args"] == [0], leaf
    assert leaf["modeled"] is (OVERLAYS[overlay] is not None), leaf


@pytest.mark.parametrize("route, veneer", [("thunk", "j_unread_body"),
                                           ("multihop", "j_j_unread_body")])
def test_a_veneer_is_named_as_the_route_not_as_the_unread_body(route, veneer):
    # The call site names the stub; the hole is the body behind it. Reporting
    # the stub would send `bn decompile` at a jump instruction, and reporting
    # nothing (which is what the unmodeled thunk route did) loses the body
    # entirely.
    [leaf] = _boundaries(_forward(route, PARTIAL))
    assert leaf["via"] == veneer, leaf
    assert leaf["callee"]["name"] == "unread_body", leaf
    assert hex(BODY) in leaf["detail"], leaf


def test_the_boundary_is_a_blocking_frontier_kind():
    # The gate keys on the registry, not on this kind by name.
    assert KIND in BLOCKING_LEAF_KINDS


# --------------------------------------------------------------------------
# controls: what must NOT acquire the disclosure
# --------------------------------------------------------------------------

@pytest.mark.parametrize("overlay", [None, PARTIAL], ids=["no-model", "partial"])
def test_a_genuine_import_is_not_an_unread_body(overlay):
    # An import has no in-binary implementation to read, so there is nothing
    # incomplete about not reading one: marking it would make every modeled
    # libc call a permanent frontier (IC05 falsification criterion 2).
    result = _forward("import", overlay)
    assert result["stats"]["analysis_incomplete"] is False, result["stats"]
    assert _boundaries(result) == [], result["leaves"]


def test_a_fully_analysed_modeled_body_is_still_descended_and_additive():
    # #807 in the other direction: the model must not stop the descent, and
    # the descent must not acquire an unread-body frontier. The body reaches
    # `system`, which only a descent can find.
    result = _forward("analysed", PARTIAL)
    assert _boundaries(result) == [], result["leaves"]
    assert result["stats"]["analysis_incomplete"] is False, result["stats"]
    assert any(s["sink"]["callee"] == "system" for s in result["reached_sinks"]), result


def test_an_unmodeled_run_keeps_its_in_binary_caveat_and_conservative_return():
    # The pre-existing unmodeled disclosure (#1001 item 3 / #811) is not
    # replaced by the structured leaf -- both are present, and the
    # conservative return is untouched.
    result = _forward("direct", None, consume_return=True)
    assert any("in-binary unread_body was not analysed" in a and "has no model" in a
               for a in result["assumptions"]), result["assumptions"]
    assert any(s["sink"]["class"] == "overflow_len" for s in result["reached_sinks"]), result


def test_a_models_own_return_semantics_survive_the_disclosure():
    # "Do not drop the partial model's valid flow": `arg:0 -> ret` still
    # taints the result, which still reaches the length argument of memcpy.
    result = _forward("direct", PARTIAL, consume_return=True)
    assert any(s["sink"]["class"] == "overflow_len" for s in result["reached_sinks"]), result
    assert _boundaries(result), result["leaves"]


# --------------------------------------------------------------------------
# the machine-readable gate, counts and advice
# --------------------------------------------------------------------------

def test_the_gate_names_the_blocking_leaf_and_counts_it_as_unresolved():
    diag = _forward("direct", PARTIAL)["diagnostics"]
    assert diag["frontier"]["by_kind"] == {KIND: 1}, diag
    # A resolved call whose body nobody read is an unresolved frontier...
    assert diag["frontier"]["unresolved"] == 1, diag
    assert diag["frontier"]["coarse_memory"] == 0, diag
    # ...but it is NOT "an unmodeled call was reached": a model does describe
    # this call. The two signals say different things and must not be fused.
    assert diag["unmodeled_calls_reached"] is False, diag
    assert KIND in diag["all_clear_reason"], diag


def test_the_advice_prints_a_command_that_runs_as_printed():
    diag = _forward("direct", PARTIAL)["diagnostics"]
    action = diag["next_action"]
    assert "unread_body" in action, action
    # The remedy is the one that changes the next run's answer.
    [command] = re.findall(r"`(bn [^`]+)`", action)
    assert shlex.split(command) == ["bn", "decompile", hex(BODY), "--force-analysis"], command
    # A model is named as what it is, so the reader is not told the overlay
    # settled the question.
    assert "model" in action, action
    assert "model" not in _forward("direct", None)["diagnostics"]["next_action"]


def test_text_mode_shows_the_note_the_frontier_row_and_the_modeled_marker():
    result = _forward("direct", PARTIAL)
    text = formatters._render_taint_text(result)
    assert ("NOTE: analysis incomplete -- 1 callee body/bodies could not be read "
            "(unread_body)" in text), text
    row = next(line for line in text.splitlines() if line.strip().startswith(KIND))
    assert hex(CALL) in row and hex(BODY) in row, row
    assert "analysis_skipped:ExceedFunctionSizeSkipReason" in row, row
    assert "modeled" in row, row
    assert "safe_to_report_all_clear: false" in text, text


def test_one_call_to_one_body_renders_as_one_row():
    # The fixpoint can reach a call site from several tainted-argument
    # contexts; that is one hole, not a wall of near-identical rows.
    leaf = _boundaries(_forward("direct", PARTIAL))[0]
    other = {**leaf, "tainted_args": [0, 1]}
    rows = formatters._render_grouped_leaves([leaf, other])
    assert rows[0] == "frontiers (2 in 1 group(s)):", rows
    assert len(rows) == 2, rows


# --------------------------------------------------------------------------
# bounds: the disclosure must survive the walk's own limits, not duplicate
# --------------------------------------------------------------------------

def test_a_recursive_wrapper_discloses_the_body_once():
    # `recurse()` calls itself and the skipped body. The cycle stops the
    # descent; the body behind it is still one boundary, and the walk
    # terminates.
    p = FVar("p", ident=40)
    p0 = FSSA(p, 0)
    q = FVar("q", ident=93)
    q0 = FSSA(q, 0)
    recurse = FFunc("recurse", WRAPPER, FSSAFunc([
        FInstr(0, WRAPPER + 4, "MLIL_CALL_SSA", f"{hex(WRAPPER)}(q#0)", reads=[q0],
               writes=[], dest=_ptr(WRAPPER), params=[_val(q0)]),
        FInstr(1, WRAPPER + 8, "MLIL_CALL_SSA", f"{hex(BODY)}(q#0)", reads=[q0],
               writes=[], dest=_ptr(BODY), params=[_val(q0)])]), params=[q])
    body = _skipped("unread_body", BODY)
    entry = FFunc("call_unread", ENTRY, FSSAFunc([
        FInstr(0, CALL, "MLIL_CALL_SSA", f"{hex(WRAPPER)}(p#0)", reads=[p0],
               writes=[], dest=_ptr(WRAPPER), params=[_val(p0)])]), params=[p])
    bv = FBV({BODY: "unread_body", WRAPPER: "recurse"},
             funcs={BODY: body, WRAPPER: recurse})
    result = te.TaintEngine(bv, te.load_models({"unread_body": PARTIAL})).forward(
        entry, [te.parse_locator("param:0")])

    assert len(_boundaries(result)) == 1, result["leaves"]
    assert result["stats"]["analysis_incomplete_functions"] == ["unread_body"]


def test_a_depth_bound_short_of_the_wrapper_claims_nothing_about_the_body():
    # Honest in both directions: with no depth left the wrapper itself is not
    # entered, so the run must not name a body it never reached -- and must
    # not report an all-clear either.
    shallow = _forward("wrapper", PARTIAL, max_depth=0)
    assert _boundaries(shallow) == [], shallow["leaves"]
    assert shallow["diagnostics"]["safe_to_report_all_clear"] is False, shallow
    deep = _forward("wrapper", PARTIAL, max_depth=4)
    assert [lf["callee"]["name"] for lf in _boundaries(deep)] == ["unread_body"]


@pytest.mark.parametrize("policy", ["conservative", "stop"])
def test_both_unknown_call_policies_disclose_the_unread_body(policy):
    # `--unknown-call stop` already emitted its own propagation-policy leaf
    # for an unmodeled callee. Body coverage is a different claim and is
    # reported under both policies, modeled or not.
    result = _forward("direct", PARTIAL, unknown_call_policy=policy)
    assert [lf["callee"]["name"] for lf in _boundaries(result)] == ["unread_body"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


# --------------------------------------------------------------------------
# IC03's global-continuation path is a different boundary and stays its own
# --------------------------------------------------------------------------

@pytest.mark.parametrize("hops", [0, 1, 2])
def test_the_global_continuation_boundary_is_not_duplicated(models, hops):
    # The IC03 shape: tainted bytes in a global, and a call with NO tainted
    # argument. The argument path has nothing to disclose there, so the
    # global leaf stays the single boundary and keeps its own cause.
    from test_taint_forward_global_frontier import _forward as _gforward
    from test_taint_forward_global_frontier import _unread_program

    func, bv = _unread_program(hops)
    overlaid = dict(models)
    overlaid["opaque_stage"] = PARTIAL
    for catalog in (models, overlaid):
        result = _gforward(bv, func, catalog)
        kinds = [lf["kind"] for lf in result["leaves"]]
        assert kinds == ["global_memory_continuation"], result["leaves"]
        assert result["leaves"][0]["reachability"]["cause"] == KIND
        assert result["stats"]["analysis_incomplete"] is True
