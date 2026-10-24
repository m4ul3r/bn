"""Forward taint: a reached model rule whose declared destination is unrecovered.

A model propagation rule such as ``arg:0 -> *arg:1`` states that the callee
writes its input through its second argument. When Binary Ninja recovered only
the first argument for a call, ``_apply_to_token`` had nothing to write and the
unkeyed-store disclosure nothing to name: both returned early on the missing
index, so the declared output vanished and the run reported no leaf and
``safe_to_report_all_clear: true`` (BN6-C009).

The fix discloses the write as a blocking ``model_destination_unrecovered``
leaf -- callee, call site, declared destination and the reached rule source --
without guessing a pointer for it. It fires only for a REACHED rule whose
destination index is beyond the call's recovered arguments: a recovered
destination keeps its existing outcome (keyed, already tainted, or
``coarse_memory_store``), and a rule whose source never carried taint stays
silent.
"""
from __future__ import annotations

import pytest
from test_taint_engine import FBV, FExpr, FFunc, FInstr, FSSA, FSSAFunc, FVar, te

from bn import formatters
from bn_agent_bridge.taint_result import BLOCKING_LEAF_KINDS

KIND = "model_destination_unrecovered"

EMIT = 0x401030        # external `model_emit`, the modeled callee
SYSTEM = 0x401040      # external `system`, a modeled sink
GETBUF = 0x401050      # external `get_buf`, an opaque pointer producer
OTHER = 0x401060       # external `other_emit`, modeled but never called
F_ADDR = 0x4018af      # call_model_emit(input)
G_ADDR = 0x401900      # outer(input) -> call_model_emit(input)
SITE = 0x4018c6        # the model_emit call inside call_model_emit


@pytest.fixture
def models():
    return te.load_models()


def _var(name: str, version: int = 0):
    v = FVar(name)
    return v, FSSA(v, version)


def _const(addr: int) -> FExpr:
    return FExpr("MLIL_CONST_PTR", hex(addr), constant=addr)


def _call(index: int, addr: int, target: int, callee: str, args: list[FExpr],
          writes: list | None = None) -> FInstr:
    text = f"{callee}({', '.join(str(a) for a in args)})"
    if writes:
        text = f"{writes[0].var.name}#{writes[0].version} = {text}"
    return FInstr(index, addr, "MLIL_CALL_SSA", text,
                  reads=[r for a in args for r in a.vars_read], writes=list(writes or []),
                  dest=_const(target), params=list(args))


def _ret(index: int, addr: int, reads=()) -> FInstr:
    return FInstr(index, addr, "MLIL_RET", "return", reads=list(reads))


def _bv(funcs=None) -> FBV:
    return FBV({EMIT: "model_emit", SYSTEM: "system", GETBUF: "get_buf",
                OTHER: "other_emit"}, funcs)


def _one_arg_emit_func(name: str = "call_model_emit", start: int = F_ADDR,
                       site: int = SITE, extra: list | None = None) -> FFunc:
    """``name(input) { model_emit(input); }`` -- BN recovered ONE argument."""
    inp, i0 = _var("input")
    instrs = [_call(0, site, EMIT, "model_emit", [FExpr("MLIL_VAR_SSA", "input#0", reads=[i0])])]
    instrs += list(extra or [])
    instrs.append(_ret(len(instrs), site + 5))
    return FFunc(name, start, FSSAFunc(instrs), params=[inp])


def _forward(models, func, source="param:0", funcs=None):
    return te.TaintEngine(_bv(funcs), models).forward(func, [te.parse_locator(source)])


def _missing(result) -> list[dict]:
    return [lf for lf in result["leaves"] if lf.get("kind") == KIND]


def _rules(*pairs: tuple[str, str]) -> dict:
    return {"propagates": [{"from": f, "to": t} for f, t in pairs]}


# --------------------------------------------------------------------------
# the defect: a reached rule into an unrecovered destination
# --------------------------------------------------------------------------

def test_root_call_reached_rule_into_unrecovered_pointee_blocks_all_clear(models):
    models["model_emit"] = _rules(("arg:0", "*arg:1"))
    result = _forward(models, _one_arg_emit_func())

    [leaf] = _missing(result)
    assert leaf["address"] == hex(SITE)
    assert leaf["callee"] == {"name": "model_emit", "display_name": "model_emit",
                              "address": hex(EMIT)}
    assert leaf["function"]["address"] == hex(F_ADDR)
    assert leaf["destination"] == "*arg:1" and leaf["destination_index"] == 1
    assert leaf["recovered_params"] == 1
    assert leaf["sources"] == [{"from": "arg:0", "tainted": "input"}]
    # No pointer was fabricated for the missing argument: nothing was keyed
    # or tainted for it, and no other frontier stands in for it.
    assert [lf["kind"] for lf in result["leaves"]] == [KIND]
    assert result["reached_sinks"] == []

    diag = result["diagnostics"]
    assert diag["safe_to_report_all_clear"] is False
    assert KIND in diag["all_clear_reason"]
    assert diag["frontier"]["by_kind"] == {KIND: 1}
    assert diag["frontier"]["unresolved"] == 1
    assert diag["frontier"]["coarse_memory"] == 0
    # The callee is modeled: an unresolved write is not an unmodeled call.
    assert diag["unmodeled_calls_reached"] is False
    assert KIND in BLOCKING_LEAF_KINDS


def test_scalar_output_destination_unrecovered_is_disclosed_too(models):
    models["model_emit"] = _rules(("arg:0", "arg:1"))
    [leaf] = _missing(_forward(models, _one_arg_emit_func()))
    assert leaf["destination"] == "arg:1" and leaf["destination_index"] == 1


def test_descended_call_discloses_unrecovered_destination(models):
    models["model_emit"] = _rules(("arg:0", "*arg:1"))
    inner = _one_arg_emit_func()
    # `return 0`: a valued return, so the hand-off back adds no frontier of its own.
    inner.mlil.ssa_form.instructions[-1].src = [FExpr("MLIL_CONST", "0", constant=0)]
    x, x0 = _var("x")
    y, y1 = _var("y", 1)
    outer = FFunc("outer", G_ADDR, FSSAFunc([
        _call(0, G_ADDR + 0x10, F_ADDR, "call_model_emit",
              [FExpr("MLIL_VAR_SSA", "x#0", reads=[x0])], writes=[y1]),
        _ret(1, G_ADDR + 0x15),
    ]), params=[x])

    result = _forward(models, outer, funcs={F_ADDR: inner})

    [leaf] = _missing(result)
    assert leaf["address"] == hex(SITE)
    assert leaf["function"] == {"name": "call_model_emit", "display_name": "call_model_emit",
                                "address": hex(F_ADDR)}
    assert leaf["sources"] == [{"from": "arg:0", "tainted": "input"}]
    diag = result["diagnostics"]
    assert diag["safe_to_report_all_clear"] is False
    assert [lf["kind"] for lf in result["leaves"]] == [KIND]
    assert f"`bn disasm {hex(F_ADDR)}`" in diag["next_action"]


def test_distinct_missing_outputs_kept_identical_rules_and_sources_merged(models):
    models["model_emit"] = _rules(
        ("arg:0", "*arg:1"),
        ("arg:0", "*arg:1"),      # an identical rule: the same write, once
        ("*arg:0", "*arg:1"),     # another reached source of the same write
        ("arg:0", "*arg:2"),      # a second missing output
        ("arg:0", "arg:3"),       # a third, scalar
    )
    result = _forward(models, _one_arg_emit_func())

    leaves = _missing(result)
    assert sorted(lf["destination"] for lf in leaves) == ["*arg:1", "*arg:2", "arg:3"]
    one = next(lf for lf in leaves if lf["destination"] == "*arg:1")
    assert one["sources"] == [{"from": "arg:0", "tainted": "input"},
                              {"from": "*arg:0", "tainted": "input"}]
    assert result["diagnostics"]["frontier"]["by_kind"] == {KIND: 3}
    assert result["diagnostics"]["frontier"]["unresolved"] == 3

    # Text keeps the three writes apart rather than one "[3 leaves]" row.
    text = formatters._render_taint_text(result)
    rows = [ln for ln in text.splitlines() if ln.startswith(f"  {KIND} @")]
    assert len(rows) == 3, text
    assert "[3 leaves]" not in text
    for dest in ("*arg:1", "*arg:2", "arg:3"):
        assert any(f"declared destination {dest} unavailable" in r for r in rows), rows


def test_repeated_call_sites_each_disclosed(models):
    models["model_emit"] = _rules(("arg:0", "*arg:1"))
    _, i0 = _var("input")
    second = _call(1, SITE + 0x10, EMIT, "model_emit",
                   [FExpr("MLIL_VAR_SSA", "input#0", reads=[i0])])
    func = _one_arg_emit_func()
    func.mlil.ssa_form.instructions.insert(1, second)
    # Both calls read the same tainted input variable, so the second is reached.
    second.vars_read = func.mlil.ssa_form.instructions[0].vars_read
    second.params = func.mlil.ssa_form.instructions[0].params

    result = _forward(models, func)
    assert sorted(lf["address"] for lf in _missing(result)) == [hex(SITE), hex(SITE + 0x10)]
    text = formatters._render_taint_text(result)
    assert "[2 leaves]" in text, text   # one destination, two sites: one grouped row


def test_varargs_destination_unrecovered_is_disclosed(models):
    models["model_emit"] = {"varargs": {"first_index": 0, "to": "*arg:2"}}
    [leaf] = _missing(_forward(models, _one_arg_emit_func()))
    assert leaf["destination"] == "*arg:2"
    assert leaf["sources"] == [{"from": "arg:0", "tainted": "input"}]


# --------------------------------------------------------------------------
# must not fire
# --------------------------------------------------------------------------

def test_unreached_rule_source_stays_silent(models):
    # The rule's source is the unrecovered arg 2: it never carried taint.
    models["model_emit"] = _rules(("arg:2", "*arg:1"))
    result = _forward(models, _one_arg_emit_func())
    assert result["leaves"] == []
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


def test_rule_source_recovered_but_untainted_stays_silent(models):
    # outer(a, b) { model_emit(b); } seeded at param:0: arg 0 of the call is clean.
    models["model_emit"] = _rules(("arg:0", "*arg:1"))
    a, _ = _var("a")
    b, b0 = _var("b")
    func = FFunc("two", F_ADDR, FSSAFunc([
        _call(0, SITE, EMIT, "model_emit", [FExpr("MLIL_VAR_SSA", "b#0", reads=[b0])]),
        _ret(1, SITE + 5),
    ]), params=[a, b])
    result = _forward(models, func, "param:0")
    assert _missing(result) == []
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


def _two_arg_emit_func(dest: FExpr, pre: list | None = None, post: list | None = None,
                       extra_params: list | None = None) -> FFunc:
    inp, i0 = _var("input")
    instrs = list(pre or [])
    instrs.append(_call(len(instrs), SITE, EMIT, "model_emit",
                        [FExpr("MLIL_VAR_SSA", "input#0", reads=[i0]), dest]))
    instrs += list(post or [])
    instrs.append(_ret(len(instrs), SITE + 0x40))
    for n, ins in enumerate(instrs):
        ins.instr_index = n
    return FFunc("call_model_emit", F_ADDR, FSSAFunc(instrs), params=[inp, *(extra_params or [])])


def test_recovered_keyable_destination_propagates_without_false_leaf(models):
    models["model_emit"] = _rules(("arg:0", "*arg:1"))
    buf = FVar("buf")
    func = _two_arg_emit_func(
        FExpr("MLIL_ADDRESS_OF", "&buf", src=buf),
        post=[_call(0, SITE + 0x10, SYSTEM, "system", [FExpr("MLIL_ADDRESS_OF", "&buf", src=buf)])])
    result = _forward(models, func)
    assert _missing(result) == []
    assert result["leaves"] == []
    # The recovered output still carries the bytes to the downstream sink.
    assert [s["sink"]["callee"] for s in result["reached_sinks"]] == ["system"], result


def test_recovered_destination_already_tainted_is_not_a_missing_output(models):
    models["model_emit"] = _rules(("arg:0", "*arg:1"))
    buf = FVar("buf")
    _, i0 = _var("input")
    first = _call(0, SITE - 0x10, EMIT, "model_emit",
                  [FExpr("MLIL_VAR_SSA", "input#0", reads=[i0]),
                   FExpr("MLIL_ADDRESS_OF", "&buf", src=buf)])
    func = _two_arg_emit_func(FExpr("MLIL_ADDRESS_OF", "&buf", src=buf), pre=[first])
    first.vars_read = func.mlil.ssa_form.instructions[1].vars_read
    first.params[0] = func.mlil.ssa_form.instructions[1].params[0]
    result = _forward(models, func)
    # The second write lands in a keyed buffer the first already tainted:
    # benign, neither a missing output nor a coarse store.
    assert result["leaves"] == []


def test_recovered_unkeyable_destination_stays_coarse_memory_store(models):
    models["model_emit"] = _rules(("arg:0", "*arg:1"))
    p, p1 = _var("p", 1)
    getbuf = _call(0, SITE - 0x10, GETBUF, "get_buf", [], writes=[p1])
    func = _two_arg_emit_func(FExpr("MLIL_VAR_SSA", "p#1", reads=[p1]), pre=[getbuf])
    result = _forward(models, func)
    assert _missing(result) == []
    assert "coarse_memory_store" in [lf["kind"] for lf in result["leaves"]]


def test_return_plus_missing_output(models):
    models["model_emit"] = _rules(("arg:0", "ret"), ("arg:0", "*arg:1"))
    inp, i0 = _var("input")
    r, r1 = _var("r", 1)
    func = FFunc("call_model_emit", F_ADDR, FSSAFunc([
        _call(0, SITE, EMIT, "model_emit", [FExpr("MLIL_VAR_SSA", "input#0", reads=[i0])],
              writes=[r1]),
        _ret(1, SITE + 5),
    ]), params=[inp])
    result = _forward(models, func)
    # The return channel propagated as before ...
    assert result["diagnostics"]["last_use"]["label"] == "r#1"
    # ... and the declared output it could not carry is still disclosed.
    assert [lf["destination"] for lf in _missing(result)] == ["*arg:1"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # With the return reaching a sink, the run has findings AND the leaf.
    s = _call(1, SITE + 0x10, SYSTEM, "system", [FExpr("MLIL_VAR_SSA", "r#1", reads=[r1])])
    func.mlil.ssa_form.instructions.insert(1, s)
    func.mlil.ssa_form.instructions[2].instr_index = 2
    result = _forward(models, func)
    assert [f["sink"]["callee"] for f in result["reached_sinks"]] == ["system"]
    assert [lf["destination"] for lf in _missing(result)] == ["*arg:1"]


@pytest.mark.parametrize("to", ["*argv:1", "buf", "*ret:1", ""])
def test_non_argument_destination_token_is_not_a_missing_output(models, to):
    models["model_emit"] = _rules(("arg:0", to))
    assert _missing(_forward(models, _one_arg_emit_func())) == []


def test_model_for_an_absent_callee_is_irrelevant(models):
    models["other_emit"] = _rules(("arg:0", "*arg:1"))
    result = _forward(models, _one_arg_emit_func())
    assert _missing(result) == []


# --------------------------------------------------------------------------
# advice and text
# --------------------------------------------------------------------------

def test_next_action_and_leaf_advice_name_runnable_reads(models):
    models["model_emit"] = _rules(("arg:0", "*arg:1"))
    result = _forward(models, _one_arg_emit_func())
    [leaf] = _missing(result)
    action = result["diagnostics"]["next_action"]
    assert "*arg:1 of model_emit at " + hex(SITE) in action
    assert f"`bn proto get {hex(EMIT)}`" in action
    assert f"`bn disasm {hex(F_ADDR)}`" in action
    assert "template" not in action      # every command runs as printed
    # The view cannot say whether model_emit is variadic: the advice reads first.
    assert leaf["callee_variadic"] is None
    assert f"`bn proto get {hex(EMIT)}`" in leaf["recovery_advice"]
    assert "undetermined" in leaf["recovery_advice"]


def test_text_names_callee_destination_source_and_gate(models):
    models["model_emit"] = _rules(("arg:0", "*arg:1"))
    text = formatters._render_taint_text(_forward(models, _one_arg_emit_func()))
    assert "frontier: 1 unresolved, 0 coarse-memory" in text
    assert "safe_to_report_all_clear: false" in text
    assert "unmodeled call(s) reached: no" in text
    row = next(ln for ln in text.splitlines() if ln.startswith(f"  {KIND} @ {hex(SITE)}"))
    assert f"-> model_emit @ {hex(EMIT)}" in row
    assert "declared destination *arg:1 unavailable" in row
    assert "recovered 1 arg(s)" in row
    assert "reached source(s) arg:0=input" in row
