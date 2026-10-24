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


def _no_mlil(name, addr):
    """A DEFINED in-binary function whose implementation MLIL is unavailable.

    Binary Ninja names it a `FunctionSymbol` of this binary -- so the
    implementation exists here -- and produced no MLIL for it (a lift that
    failed). The hole is the same as a measured skip: an implementation this
    binary has, that this walk did not read.
    """
    return FFunc(name, addr, FSSAFunc([]), params=[FVar("q", ident=90)],
                 symbol_type="FunctionSymbol")


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


def _program(route, overlay, *, consume_return=False, body_shape="skipped"):
    """``call_unread(p)`` reaching an unread body by *route*.

    Returns ``(entry, bv, models, resolve_map)``. With ``consume_return`` the
    caller hands the call's RESULT to a modeled sink, which is how a model's
    own return semantics are observed to survive the disclosure.
    ``body_shape`` picks which way the implementation is unavailable: measured
    and skipped, or defined with no MLIL at all.
    """
    p, result = FVar("p", ident=40), FVar("result", ident=41)
    p0, result1 = FSSA(p, 0), FSSA(result, 1)
    body = (_skipped("unread_body", BODY) if body_shape == "skipped"
            else _no_mlil("unread_body", BODY))
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


def _forward(route, overlay, *, consume_return=False, max_depth=8,
             body_shape="skipped", **engine_kwargs):
    entry, bv, catalog, resolve_map = _program(route, overlay, body_shape=body_shape,
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
@pytest.mark.parametrize("body_shape, reason", [
    ("skipped", "analysis_skipped:ExceedFunctionSizeSkipReason"),
    ("no-mlil", "no_mlil"),
], ids=["measured-skip", "no-mlil"])
def test_a_matching_model_never_certifies_an_unread_body(route, overlay,
                                                         body_shape, reason):
    # BN6-C008: with ANY matching entry the run reported
    # analysis_incomplete=false, zero leaves and safe_to_report_all_clear=true
    # -- on the identical binary whose unmodeled run withheld the claim. The
    # overlay states one relation; it never establishes that the rest of the
    # body was read. Both ways an implementation can be unavailable count:
    # BN measured the body and skipped it, or it produced no MLIL at all.
    result = _forward(route, OVERLAYS[overlay], body_shape=body_shape)
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
    assert leaf["skip_reason"] == reason, leaf
    assert leaf["tainted_args"] == [0], leaf
    assert leaf["abi_recovered_args"] == [], leaf
    assert leaf["modeled"] is (OVERLAYS[overlay] is not None), leaf


def test_a_bodiless_record_with_no_defined_symbol_is_not_an_unread_body():
    # The external machinery's degenerate shape: an address a NAME is known at,
    # with no function Binary Ninja defines behind it. There is no in-binary
    # implementation to read, so there is nothing incomplete about not reading
    # one -- and marking it would make every modeled external call a permanent
    # frontier.
    p = FVar("p", ident=40)
    p0 = FSSA(p, 0)
    call = FInstr(0, CALL, "MLIL_CALL_SSA", "memcpy(p#0, s, n)", reads=[p0], writes=[],
                  dest=_ptr(MEMCPY),
                  params=[_val(p0), _ptr(0x407000), _ptr(0x408000)])
    entry = FFunc("call_unread", ENTRY, FSSAFunc([call]), params=[p])
    result = te.TaintEngine(FBV({MEMCPY: "memcpy"}), te.load_models()).forward(
        entry, [te.parse_locator("param:0")])
    assert _boundaries(result) == [], result["leaves"]
    assert result["stats"]["analysis_incomplete"] is False, result["stats"]


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


def _two_context_program(overlay):
    """``contexts(p)``: ``nested(p, 0)`` and ``nested(0, p)`` -- ONE call to the
    unread body, reached from two tainted-argument contexts."""
    p = FVar("p", ident=40)
    p0 = FSSA(p, 0)
    a, b = FVar("a", ident=94), FVar("b", ident=95)
    a0, b0 = FSSA(a, 0), FSSA(b, 0)
    r0, r1, r2 = (FSSA(FVar(n, ident=i), 1)
                  for n, i in (("rn", 98), ("r0", 99), ("r1", 100)))
    nested = FFunc("nested", WRAPPER, FSSAFunc([
        FInstr(0, WRAPPER + 4, "MLIL_CALL_SSA", "rn#1 = unread_body(a#0, b#0)",
               reads=[a0, b0], writes=[r0], dest=_ptr(BODY),
               params=[_val(a0), _val(b0)])]), params=[a, b])
    zero = FExpr("MLIL_CONST", "0", constant=0)
    entry = FFunc("contexts", ENTRY, FSSAFunc([
        FInstr(0, CALL, "MLIL_CALL_SSA", "r0#1 = nested(p#0, 0)", reads=[p0],
               writes=[r1], dest=_ptr(WRAPPER), params=[_val(p0), zero]),
        FInstr(1, SINK_CALL, "MLIL_CALL_SSA", "r1#1 = nested(0, p#0)", reads=[p0],
               writes=[r2], dest=_ptr(WRAPPER), params=[zero, _val(p0)]),
    ]), params=[p])
    body = _skipped("unread_body", BODY)
    body.parameter_vars = [FVar("x", ident=96), FVar("y", ident=97)]
    bv = FBV({BODY: "unread_body", WRAPPER: "nested"},
             funcs={BODY: body, WRAPPER: nested})
    return entry, bv, te.load_models({"unread_body": overlay} if overlay else None)


@pytest.mark.parametrize("overlay", [None, PARTIAL], ids=["no-model", "partial"])
def test_one_call_to_one_body_is_one_boundary_with_every_argument(overlay):
    # Review F2: the same call, reached from two tainted-argument contexts,
    # produced TWO leaves -- the unresolved frontier counted 2, and the text
    # row printed the first context's arguments followed by `[2 leaves]`,
    # hiding the second's. One unread body at one call site is one hole.
    entry, bv, catalog = _two_context_program(overlay)
    result = te.TaintEngine(bv, catalog).forward(entry, [te.parse_locator("param:0")])

    [leaf] = _boundaries(result)
    assert leaf["address"] == hex(WRAPPER + 4), leaf
    assert leaf["tainted_args"] == [0, 1], leaf
    diag = result["diagnostics"]
    assert diag["frontier"]["by_kind"][KIND] == 1, diag
    assert diag["frontier"]["unresolved"] == 1, diag

    row = next(line for line in formatters._render_taint_text(result).splitlines()
               if line.strip().startswith(KIND))
    assert "[0, 1]" in row, row
    assert "leaves]" not in row, row


def test_two_calls_to_one_body_stay_two_boundaries():
    # The must-not-overreach twin: distinct call sites are distinct holes, and
    # merging them would hide one of the two places to look.
    leaf = _boundaries(_forward("direct", PARTIAL))[0]
    elsewhere = {**leaf, "address": hex(SINK_CALL)}
    leaves: list = []
    te._add_leaf(leaves, leaf)
    te._add_leaf(leaves, elsewhere)
    assert [lf["address"] for lf in leaves] == [hex(CALL), hex(SINK_CALL)]


def test_merging_a_boundary_never_mutates_the_leaf_it_merges_into():
    # The merged copy must not be written back into an object a cached callee
    # summary (or another callsite's by_source row) may still hold.
    first = _boundaries(_forward("direct", PARTIAL))[0]
    leaves: list = []
    te._add_leaf(leaves, first)
    te._add_leaf(leaves, {**first, "tainted_args": [2]})
    assert leaves[0]["tainted_args"] == [0, 2], leaves
    assert first["tainted_args"] == [0], first


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


# --------------------------------------------------------------------------
# F3: an argument the MLIL call does not carry, proven by the ABI register
# --------------------------------------------------------------------------

class _LBlock(list):
    """An LLIL-SSA basic block: a list of instructions plus CFG edges."""

    def __init__(self, instrs, incoming=()):
        super().__init__(instrs)
        self.incoming_edges = [types.SimpleNamespace(source=b) for b in incoming]
        self.immediate_dominator = None
        for ins in instrs:
            ins.il_basic_block = self


_CALL_OP = "LLIL_CALL_SSA_OP"
_SET_OP = "LLIL_SET_REG_SSA_OP"
_FAKE_BN = types.SimpleNamespace(LowLevelILOperation=types.SimpleNamespace(
    LLIL_CALL_SSA=_CALL_OP, LLIL_SET_REG_SSA=_SET_OP, LLIL_REG_PHI="LLIL_REG_PHI_OP",
    LLIL_SYSCALL_SSA="LLIL_SYSCALL_SSA_OP", LLIL_INTRINSIC_SSA="LLIL_INTRINSIC_SSA_OP",
    LLIL_TAILCALL_SSA="LLIL_TAILCALL_SSA_OP"))


def _lins(index, address, operation, *, reg=None, mlil_writes=None):
    ins = types.SimpleNamespace(instr_index=index, address=address,
                                operation=operation, output=[])
    if reg is not None:
        ins.dest = types.SimpleNamespace(reg=reg)
    if mlil_writes is not None:
        ins.mlil = types.SimpleNamespace(
            ssa_form=types.SimpleNamespace(vars_written=list(mlil_writes)))
    return ins


def _llil_func(blocks):
    return types.SimpleNamespace(llil=types.SimpleNamespace(ssa_form=list(blocks)))


def test_an_unwritten_argument_register_carries_the_entry_value():
    # `f(p) { unread_body(/* rdi still p */); }` -- nothing defines rdi, and
    # nothing that can run first may have clobbered it, so the register still
    # holds what the function was entered with.
    call = _lins(0, CALL, _CALL_OP)
    _LBlock([call])
    kind, seeds = te.argument_register_provenance(
        _llil_func([[call]]), CALL, "rdi", _FAKE_BN)
    assert (kind, seeds) == ("entry", [])


def test_a_defining_instruction_is_bridged_to_its_mlil_value():
    v = FSSA(FVar("token", ident=50), 1)
    setter = _lins(0, CALL - 4, _SET_OP, reg="rdi", mlil_writes=[v])
    call = _lins(1, CALL, _CALL_OP)
    _LBlock([setter, call])
    kind, seeds = te.argument_register_provenance(
        _llil_func([[setter, call]]), CALL, "rdi", _FAKE_BN)
    assert kind == "defined"
    assert [var for var, _ in seeds] == [v]


def test_a_call_that_may_have_clobbered_the_register_proves_nothing():
    # A callee may clobber any caller-saved register and no definition in this
    # function records it, so the entry value is NOT claimed. Nothing is
    # inferred from the convention's clobber list.
    earlier = _lins(0, CALL - 8, _CALL_OP)
    call = _lins(1, CALL, _CALL_OP)
    _LBlock([earlier, call])
    kind, seeds = te.argument_register_provenance(
        _llil_func([[earlier, call]]), CALL, "rdi", _FAKE_BN)
    assert (kind, seeds) == ("unproven", [])


def test_a_call_later_in_a_loop_body_also_refuses_the_entry_value():
    # On the second iteration the LATER call runs first. Instruction order is
    # not execution order, so the CFG decides.
    call = _lins(0, CALL, _CALL_OP)
    later = _lins(1, CALL + 8, _CALL_OP)
    block = _LBlock([call, later])
    block.incoming_edges = [types.SimpleNamespace(source=block)]
    kind, _ = te.argument_register_provenance(
        _llil_func([block]), CALL, "rdi", _FAKE_BN)
    assert kind == "unproven"


def test_a_call_on_a_reaching_branch_refuses_the_entry_value():
    # The clobbering call is not on the dominator chain, but its block reaches
    # the call's block, so it may still have run first.
    other = _lins(0, CALL - 16, _CALL_OP)
    branch = _LBlock([other])
    call = _lins(1, CALL, _CALL_OP)
    _LBlock([call], incoming=[branch])
    kind, _ = te.argument_register_provenance(
        _llil_func([branch, [call]]), CALL, "rdi", _FAKE_BN)
    assert kind == "unproven"


def test_a_view_with_no_llil_claims_nothing():
    assert te.argument_register_provenance(
        types.SimpleNamespace(llil=None), CALL, "rdi", _FAKE_BN) == ("unproven", [])


def _dropped_arg_program(*, overlay=None, extra_instr=None, n_args=1):
    """`call_unread(p)`: a call to the unread body whose MLIL carries NO
    parameters -- the shape an optimised build leaves when BN has no prototype
    for the callee to recover arguments from."""
    p, q = FVar("p", ident=40), FVar("q", ident=42)
    p0, q0 = FSSA(p, 0), FSSA(q, 0)
    params = [p, q][:n_args]
    call = FInstr(0, CALL, "MLIL_CALL_SSA", "unread_body()", reads=[], writes=[],
                  dest=_ptr(BODY), params=[])
    instrs = [call] + list(extra_instr or [])
    entry = FFunc("call_unread", ENTRY, FSSAFunc(instrs), params=params)
    entry.calling_convention = types.SimpleNamespace(int_arg_regs=["rdi", "rsi"])
    body = _skipped("unread_body", BODY)
    body.parameter_vars = [FVar("x", ident=96), FVar("y", ident=97)]
    bv = FBV({BODY: "unread_body", MEMCPY: "memcpy"}, funcs={BODY: body})
    catalog = te.load_models({"unread_body": overlay} if overlay else None)
    return entry, bv, catalog, (p0, q0), call


def _with_recovered(engine, recovered):
    """Stand in for the LLIL bridge: `recovered` maps an argument index to the
    MLIL SSA value its register is proven to carry."""
    engine._reaching_arg_seeds_via_reg = (
        lambda f, sites, idx: (["rdi", "rsi"][idx] if idx < 2 else None,
                               [(recovered[idx], sites[0])] if idx in recovered else []))


@pytest.mark.parametrize("overlay", [None, PARTIAL], ids=["no-model", "partial"])
def test_an_argument_the_call_does_not_carry_still_reaches_the_unread_body(overlay):
    # The confirmed ABI-provenance gap: BN recovers a call's arguments from the
    # CALLEE's prototype, and a callee it never analysed has none -- so an
    # optimised build left `unread_body()` with no arguments and the walk
    # reported an all-clear while the machine code passed the seeded value in
    # rdi. The register's own evidence is enough to cross the boundary.
    entry, bv, catalog, (p0, _q0), _call = _dropped_arg_program(overlay=overlay)
    engine = te.TaintEngine(bv, catalog)
    _with_recovered(engine, {0: p0})
    result = engine.forward(entry, [te.parse_locator("param:0")])

    [leaf] = _boundaries(result)
    assert leaf["tainted_args"] == [0], leaf
    assert leaf["abi_recovered_args"] == [0], leaf
    assert result["stats"]["analysis_incomplete_functions"] == ["unread_body"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    assert any("calling convention's register(s) rdi" in a
               for a in result["assumptions"]), result["assumptions"]


def test_several_dropped_arguments_are_each_recovered_on_their_own_register():
    entry, bv, catalog, (p0, q0), _call = _dropped_arg_program(n_args=2)
    engine = te.TaintEngine(bv, catalog)
    _with_recovered(engine, {0: p0, 1: q0})
    result = engine.forward(entry, [te.parse_locator("param:0"),
                                    te.parse_locator("param:1")])
    [leaf] = _boundaries(result)
    assert leaf["tainted_args"] == [0, 1], leaf
    assert leaf["abi_recovered_args"] == [0, 1], leaf


def test_an_untainted_register_recovers_nothing():
    # The must-not-fire twin: the register is proven, and what it carries is
    # simply not tainted. No boundary, no caveat, and the run stays clean.
    entry, bv, catalog, _seeds, _call = _dropped_arg_program()
    engine = te.TaintEngine(bv, catalog)
    _with_recovered(engine, {0: FSSA(FVar("unrelated", ident=60), 1)})
    result = engine.forward(entry, [te.parse_locator("param:0")])
    assert _boundaries(result) == [], result["leaves"]
    assert result["stats"]["analysis_incomplete"] is False, result["stats"]


def test_nothing_is_recovered_when_the_register_is_unproven():
    # No LLIL to read (the default for these fakes): the gap is left as it is
    # rather than guessed at.
    entry, bv, catalog, _seeds, _call = _dropped_arg_program()
    result = te.TaintEngine(bv, catalog).forward(entry, [te.parse_locator("param:0")])
    assert _boundaries(result) == [], result["leaves"]
    assert result["stats"]["analysis_incomplete"] is False, result["stats"]


def test_a_recovered_argument_feeds_the_models_own_propagation():
    # "Seed model application from that evidence": `arg:0 -> ret` fires on an
    # argument the call itself does not carry, so the tainted return still
    # reaches the downstream sink.
    result_var = FVar("result", ident=43)
    r1 = FSSA(result_var, 1)
    sink = FInstr(1, SINK_CALL, "MLIL_CALL_SSA", "memcpy(dst, src, result#1)",
                  reads=[r1], writes=[], dest=_ptr(MEMCPY),
                  params=[_ptr(0x407000), _ptr(0x408000), _val(r1)])
    entry, bv, catalog, (p0, _q0), call = _dropped_arg_program(
        overlay=PARTIAL, extra_instr=[sink])
    call.vars_written = [r1]
    engine = te.TaintEngine(bv, catalog)
    _with_recovered(engine, {0: p0})
    result = engine.forward(entry, [te.parse_locator("param:0")])
    assert any(s["sink"]["class"] == "overflow_len" for s in result["reached_sinks"]), result
    assert _boundaries(result), result["leaves"]


def test_the_recovered_argument_is_disclosed_in_text():
    entry, bv, catalog, (p0, _q0), _call = _dropped_arg_program(overlay=PARTIAL)
    engine = te.TaintEngine(bv, catalog)
    _with_recovered(engine, {0: p0})
    text = formatters._render_taint_text(engine.forward(
        entry, [te.parse_locator("param:0")]), full=True)
    row = next(line for line in text.splitlines() if line.strip().startswith(KIND))
    assert "ABI-recovered arg(s) [0]" in row, row


def test_a_clobbering_call_is_never_read_as_a_definition():
    # BN lists the registers a call clobbers among its LLIL outputs, and the
    # MLIL mapping of that call writes its RESULT variable -- the RETURN
    # register's value, which says nothing about an argument register.
    # Bridging to it handed the walk the wrong value entirely.
    v = FSSA(FVar("result", ident=51), 2)
    earlier = _lins(0, CALL - 8, _CALL_OP, mlil_writes=[v])
    earlier.output = [types.SimpleNamespace(reg="rdx")]
    call = _lins(1, CALL, _CALL_OP)
    _LBlock([earlier, call])
    kind, seeds = te.argument_register_provenance(
        _llil_func([[earlier, call]]), CALL, "rdx", _FAKE_BN)
    assert (kind, seeds) == ("unproven", [])


def test_a_call_whose_arguments_binary_ninja_recovered_is_not_second_guessed():
    # The arity gate: a recovered parameter list IS Binary Ninja's arity
    # determination. A register past it holds dead scratch as often as an
    # argument -- at -O0 a compiler stages the second argument through `rdx`
    # before moving it to `rsi` -- so reading one as "argument 2" of a
    # two-argument call invents an argument that is not passed.
    p = FVar("p", ident=40)
    p0 = FSSA(p, 0)
    call = FInstr(0, CALL, "MLIL_CALL_SSA", "unread_body(p#0)", reads=[p0],
                  writes=[], dest=_ptr(BODY), params=[_val(p0)])
    entry = FFunc("call_unread", ENTRY, FSSAFunc([call]), params=[p])
    entry.calling_convention = types.SimpleNamespace(int_arg_regs=["rdi", "rsi"])
    body = _skipped("unread_body", BODY)
    body.parameter_vars = [FVar("x", ident=96), FVar("y", ident=97)]
    engine = te.TaintEngine(FBV({BODY: "unread_body"}, funcs={BODY: body}),
                            te.load_models())
    _with_recovered(engine, {1: p0})   # rsi would answer, if it were asked
    [leaf] = _boundaries(engine.forward(entry, [te.parse_locator("param:0")]))
    assert leaf["tainted_args"] == [0], leaf
    assert leaf["abi_recovered_args"] == [], leaf
