"""scanf-family seeds: both ISO spellings, and destinations BN did not recover
(P2-12).

A `call:`/`ret:`/`arg:` seed on a scanf-family call whose constant format
writes more destinations than Binary Ninja recovered used to be refused ("no
taint sources resolved") or come back as a clean all-clear; the frontier that
names the lost destinations was built and thrown away. And `call:scanf` found
the C23 `__isoc23_scanf` but not the C99 `__isoc99_scanf`.
"""
from __future__ import annotations

import pytest

from test_taint_engine import FBVStr, FExpr, FFunc, FInstr, FSSA, FSSAFunc, FVar, _ext_call, te

SCANF, FMT = 0x900, 0x700


def _scan_forwarded(callee: str, fmt: str, *, tail: bool = False, used: bool = True):
    """scan_fwd(unused, a, b, c): r = <callee>(fmt) -- the destinations are
    forwarded incoming pointers BN left in their registers, so the MLIL call
    carries the format alone (arity 1)."""
    r1 = FSSA(FVar("r"), 1)
    params = [FExpr("MLIL_CONST_PTR", repr(fmt), constant=FMT)]
    if tail:
        call = FInstr(0, 0x401221, "MLIL_TAILCALL_SSA", f"return {callee}(fmt) __tailcall",
                      dest=FExpr("MLIL_CONST_PTR", hex(SCANF), constant=SCANF), params=params)
    else:
        call = _ext_call(0, 0x401221, f"r#1 = {callee}(fmt)", SCANF, params,
                         writes=[r1] if used else [])
    instrs = [call, FInstr(1, 0x401230, "MLIL_RET", "return r#1", reads=[r1])]
    func = FFunc("scan_fwd", 0x401210, FSSAFunc(instrs),
                 params=[FVar("unused"), FVar("a"), FVar("b"), FVar("c")])
    return func, FBVStr({SCANF: callee}, strings={FMT: fmt})


@pytest.mark.parametrize("callee", ["__isoc99_scanf", "__isoc23_scanf", "scanf"])
@pytest.mark.parametrize(("locator", "tail"), [
    ("call:scanf", False), ("call:{callee}", False), ("ret:{callee}", False),
    # A tail call has no result, so ret: has nothing there to name.
    ("call:scanf", True), ("call:{callee}", True),
])
def test_a_seed_on_a_call_with_unrecovered_destinations_reports_them(callee, locator, tail):
    func, bv = _scan_forwarded(callee, "%15s %15s %15s", tail=tail)
    result = te.TaintEngine(bv, te.load_models()).forward(
        func, [te.parse_locator(locator.format(callee=callee))])
    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "format_destination_unrecovered"]
    assert (leaf["address"], leaf["destinations"], leaf["recovered"], leaf["missing"]) == \
        ("0x401221", 3, 0, 3), leaf
    # `callee` is the symbol the call site calls; the model it was judged by
    # rides beside it as `model` when that is another name.
    assert (leaf["callee"], leaf.get("model")) == \
        (callee, None if callee == "scanf" else "scanf"), leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_a_call_seed_with_nothing_to_seed_and_no_frontier_is_still_refused():
    # The control: a format that writes nothing names no lost destination, so
    # a seed that keys nothing is still a bad locator.
    func, bv = _scan_forwarded("__isoc99_scanf", "literal text only")
    with pytest.raises(te.TaintError, match="no taint sources resolved"):
        te.TaintEngine(bv, te.load_models()).forward(func, [te.parse_locator("call:scanf")])


@pytest.mark.parametrize("std", ["99", "23"])
@pytest.mark.parametrize(("base", "stream"), [("scanf", False), ("fscanf", True)])
def test_a_plain_call_locator_finds_either_iso_spelling(std, base, stream):
    # W04-04 / W07-15: `call:fscanf` and `call:scanf` name the call whichever
    # ISO-C spelling the compiler emitted, and seed its destination.
    callee = f"__isoc{std}_{base}"
    d = FVar("d", typ="char[0x10]")
    args = ([FExpr("MLIL_VAR_SSA", "f#0", reads=[FSSA(FVar("f"), 0)])] if stream else [])
    args += [FExpr("MLIL_CONST_PTR", "fmt", constant=FMT), FExpr("MLIL_ADDRESS_OF", "&d", src=d)]
    SYSTEM = 0x901
    instrs = [_ext_call(0, 0x10, f"{callee}(fmt, &d)", SCANF, args),
              _ext_call(1, 0x14, "system(&d)", SYSTEM, [FExpr("MLIL_ADDRESS_OF", "&d", src=d)]),
              FInstr(2, 0x18, "MLIL_RET", "return")]
    func = FFunc("read_in", 0x10, FSSAFunc(instrs), params=[])
    bv = FBVStr({SCANF: callee, SYSTEM: "system"}, strings={FMT: "%15s"})
    result = te.TaintEngine(bv, te.load_models()).forward(
        func, [te.parse_locator(f"call:{base}")])
    assert [(s["sink"]["callee"], s["sink"]["class"]) for s in result["reached_sinks"]] == \
        [("system", "command_injection")], result["reached_sinks"]


class _SymBV(FBVStr):
    """FBVStr whose import symbols resolve by raw name, as a real view's do."""
    def get_symbol_by_raw_name(self, name):
        if name in self._names.values():
            return type("Sym", (), {"name": name, "short_name": name})()
        return None


@pytest.mark.parametrize("std", ["99", "23"])
def test_a_finding_on_an_iso_alias_names_the_symbol_the_binary_calls(std):
    # `callee` is the spelling the binary imports, as it was while the C99
    # family had duplicated models of its own; the plain model the finding was
    # judged by is `model`.
    callee = f"__isoc{std}_sscanf"
    s, d = FVar("s"), FVar("d", typ="char[0x10]")
    s0 = FSSA(s, 0)
    instrs = [_ext_call(0, 0x10, f"{callee}(s#0, fmt, &d)", SCANF,
                        [FExpr("MLIL_VAR_SSA", "s#0", reads=[s0]),
                         FExpr("MLIL_CONST_PTR", "fmt", constant=FMT),
                         FExpr("MLIL_ADDRESS_OF", "&d", src=d)], reads=[s0]),
              FInstr(1, 0x14, "MLIL_RET", "return")]
    func = FFunc("parse", 0x10, FSSAFunc(instrs), params=[s])
    result = te.TaintEngine(_SymBV({SCANF: callee}, strings={FMT: "%s"}), te.load_models()).forward(
        func, [te.parse_locator("param:0")])
    assert [(f["sink"]["callee"], f["sink"]["callee_symbol"], f["sink"].get("model"))
            for f in result["reached_sinks"]] == [(callee, callee, "sscanf")], result["reached_sinks"]
