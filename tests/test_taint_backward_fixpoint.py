from __future__ import annotations

import pytest

from test_taint_engine import (
    FVar, FExpr, _BufProgram, _addr_of, _cstr, _bw, _store_proof_only,
    _B_STRCPY, _B_READ, _B_MEMSET, _B_NAMES, te,
)


@pytest.mark.parametrize("writer", [_B_MEMSET, _B_STRCPY])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("initialized", [False, True])
def test_loop_exit_coverage_950(reverse, initialized, writer):
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    if initialized:
        pg.whole_store(buf)
    entry = pg.mem
    header = pg.phi(entry, entry + 3)
    params = ([_addr_of(buf), FExpr("MLIL_CONST", "0", constant=0),
               FExpr("MLIL_CONST", "64", constant=64)] if writer == _B_MEMSET
              else [_addr_of(buf), _cstr(0x700)])
    fill = pg.call(writer, params)
    back = pg.call(_B_STRCPY, [_addr_of(FVar("other", ident=99)), _cstr(0x700)])
    back.src_memory = header.dest_memory
    arms = [fill.dest_memory, back.dest_memory]
    pg.phi(*(reversed(arms) if reverse else arms))
    pg.sink(buf)
    result, _, leaves = _bw(pg.func())
    # Initialized by a store: the best a buffer reaches is the uncertified
    # store proof (#931 round 3).
    assert _store_proof_only(result) is initialized
    assert bool(leaves) is not initialized


@pytest.mark.parametrize("escape", [False, True])
def test_loop_carried_escape_951(escape):
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    pg.whole_store(buf)     # a call handed &buf would itself start an escape (#1053 r4)
    header = pg.phi(1, 4)
    # Use an existing unknown target name, with no pointer argument.
    unknown = next(k for k, v in _B_NAMES.items() if v == "compose")
    pg.call(unknown, [])
    pg.store(_cstr(0x404000), _addr_of(buf) if escape else FExpr("MLIL_CONST", "0", constant=0))
    sink = pg.sink(buf)
    sink.src_memory = header.dest_memory
    result, reasons, leaves = _bw(pg.func())
    assert _store_proof_only(result) is not escape
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    assert any(l.get("callee") == "compose" for l in leaves) is escape


@pytest.mark.parametrize("writer", [_B_MEMSET, _B_READ])
@pytest.mark.parametrize("length", [0, 1, 63, 64, 65])
def test_call_fill_never_proves_initialisation_931(writer, length):
    # A requested length is not the bytes a call delivered (a short read, a
    # failing call): even a full-size memset or read at offset 0 leaves the
    # buffer unproven, disclosed with the call and the consumed region.
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    n = FExpr("MLIL_CONST", str(length), constant=length)
    pg.call(writer, [_addr_of(buf), FExpr("MLIL_CONST", "0", constant=0), n]
            if writer == _B_MEMSET else [FExpr("MLIL_CONST", "0", constant=0), _addr_of(buf), n])
    pg.sink(buf)
    result, reasons, leaves = _bw(pg.func())
    name = _B_NAMES[writer]
    assert any(f"written by {name}" in r for r in reasons)
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    [leaf] = leaves
    assert leaf["calls"] == [name] and leaf["region"] == {"offset": 0, "size": 64}, leaf
    assert leaf["unproven"] == [[0, 64]], leaf


@pytest.mark.parametrize("budget", [1, 5, 12])
def test_fixed_point_budget_never_certifies_partial_proof(monkeypatch, budget):
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    pg.call(_B_STRCPY, [_addr_of(buf), _cstr(0x700)])
    header = pg.phi(1, 4)
    pg.call(next(k for k, v in _B_NAMES.items() if v == "compose"), [])
    pg.store(_cstr(0x404000), _addr_of(buf))
    sink = pg.sink(buf)
    sink.src_memory = header.dest_memory
    monkeypatch.setattr(te, "MEM_CHAIN_BUDGET", budget)
    result, _, _ = _bw(pg.func())
    assert result["stats"]["truncated"]
    assert not result["diagnostics"]["safe_to_report_complete_slice"]


@pytest.mark.parametrize("length", [0, 1, 63])
def test_short_source_preserves_source_without_coverage(length):
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    pg.call(_B_READ, [FExpr("MLIL_CONST", "0", constant=0), _addr_of(buf),
                     FExpr("MLIL_CONST", str(length), constant=length)])
    pg.sink(buf)
    result, reasons, leaves = _bw(pg.func())
    assert (result["slices"][0]["origin"]["kind"], result["slices"][0]["origin"]["callee"]) == ("source", "read")
    assert leaves and not result["diagnostics"]["safe_to_report_complete_slice"]


def test_loop_carried_stash_is_not_a_definite_destination():
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    header = pg.phi(0, 3)
    reload = FExpr("MLIL_LOAD_SSA", "[stash]", src=_cstr(0x404000),
                   src_memory=header.dest_memory)
    reload.size = 8
    pg.call(_B_STRCPY, [reload, _cstr(0x700)])
    pg.store(_cstr(0x404000), _addr_of(buf))
    pg.sink(buf)
    result, _, leaves = _bw(pg.func())
    assert leaves and not result["diagnostics"]["safe_to_report_complete_slice"]


@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_nested_loop_escape_reaches_inner_call(escape, reverse):
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    pg.whole_store(buf)
    pg.phi(*([6, 1] if reverse else [1, 6]))
    pg.phi(*([5, 2] if reverse else [2, 5]))
    pg.call(next(k for k, v in _B_NAMES.items() if v == "compose"), [])
    pg.store(_cstr(0x404008), FExpr("MLIL_CONST", "0", constant=0))
    pg.store(_cstr(0x404000), _addr_of(buf) if escape else FExpr("MLIL_CONST", "0", constant=0))
    sink = pg.sink(buf)
    sink.src_memory = 2
    result, _, leaves = _bw(pg.func())
    assert _store_proof_only(result) is not escape
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    assert any(l.get("callee") == "compose" for l in leaves) is escape
    assert not result["stats"]["truncated"]


def test_future_stash_does_not_invent_an_earlier_buffer_writer():
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    pg.whole_store(buf)     # a call handed &buf would itself start an escape (#1053 r4)
    reload = FExpr("MLIL_LOAD_SSA", "[stash]", src=_cstr(0x404000), src_memory=pg.mem)
    pg.call(_B_STRCPY, [reload, _cstr(0x700)])
    pg.store(_cstr(0x404000), _addr_of(buf))
    pg.sink(buf)
    result, reasons, leaves = _bw(pg.func())
    assert not any("written by strcpy" in reason for reason in reasons)
    assert _store_proof_only(result), result["leaves"]
    assert not leaves


def test_discovery_cut_keeps_nearby_escaped_call_frontier():
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    unknown = next(k for k, v in _B_NAMES.items() if v == "compose")
    for _ in range(300):
        pg.call(unknown, [])
    pg.call(_B_MEMSET, [_addr_of(buf), FExpr("MLIL_CONST", "0", constant=0),
                       FExpr("MLIL_CONST", "64", constant=64)])
    pg.store(_cstr(0x404000), _addr_of(buf))
    near = pg.call(unknown, [])
    pg.sink(buf)
    result, _, leaves = _bw(pg.func())
    assert result["stats"]["truncated"]
    assert not result["diagnostics"]["safe_to_report_complete_slice"]
    assert any(l.get("callee") == "compose" and l["address"] == hex(near.address)
               for l in leaves)


# --------------------------------------------------------------------------
# IC08: a loop counter's initializer is not a runtime constant
# --------------------------------------------------------------------------

_MEMCPY = 0xc40


def _counter_program(shape):
    """`count(dst, src)`: a length `k` in one of four shapes, then
    `memcpy(dst, src, k)`.

    * ``counter``: `k = 0; loop { if (..) k = k + 1; }` -- the -O0 lift of a
      conditional count, `k#2 = phi(k#1, k#4)`, `k#4 = phi(k#2, k#3)`;
    * ``doubling``: `k = 1; loop { k = k * 2; }` -- a recurrence with no
      constant per-trip increment;
    * ``choice``: `k = c ? 0 : 5` -- a phi of two literals, no cycle;
    * ``literal``: `k = 16`."""
    from test_taint_engine import FBV, FFunc, FInstr, FSSA, FSSAFunc

    dst, src = FVar("dst", ident=1, typ="char*"), FVar("src", ident=2, typ="char*")
    kv = FVar("k", ident=3)
    k = [FSSA(kv, i) for i in range(5)]

    def var(s):
        return FExpr("MLIL_VAR_SSA", f"k#{s.version}", reads=[s], src=s)

    def const(n):
        return FExpr("MLIL_CONST", str(n), constant=n)

    def set_(i, dest, src_expr, reads=()):
        return FInstr(i, 0x100 + 4 * i, "MLIL_SET_VAR_SSA", f"k#{dest.version} = {src_expr}",
                      reads=list(reads), writes=[dest], dest=dest, src=src_expr)

    def phi(i, dest, *arms):
        ins = FInstr(i, 0x100 + 4 * i, "MLIL_VAR_PHI",
                     f"k#{dest.version} = phi({', '.join(f'k#{a.version}' for a in arms)})",
                     reads=list(arms), writes=[dest])
        ins.dest, ins.src = dest, list(arms)
        return ins

    if shape == "counter":
        body = [set_(0, k[1], const(0)), phi(1, k[2], k[1], k[4]),
                set_(2, k[3], FExpr("MLIL_ADD", "k#2 + 1", reads=[k[2]], left=var(k[2]),
                                    right=const(1)), reads=[k[2]]),
                phi(3, k[4], k[2], k[3])]
        length = k[2]
    elif shape == "doubling":
        body = [set_(0, k[1], const(1)), phi(1, k[2], k[1], k[3]),
                set_(2, k[3], FExpr("MLIL_MUL", "k#2 * 2", reads=[k[2]], left=var(k[2]),
                                    right=const(2)), reads=[k[2]])]
        length = k[2]
    elif shape == "choice":
        body = [set_(0, k[1], const(0)), set_(1, k[2], const(5)), phi(2, k[3], k[1], k[2])]
        length = k[3]
    else:
        body = [set_(0, k[1], const(16))]
        length = k[1]
    d0, s0 = FSSA(dst, 0), FSSA(src, 0)
    body.append(FInstr(len(body), 0x200, "MLIL_CALL_SSA", f"memcpy(dst#0, src#0, k#{length.version})",
                       reads=[d0, s0, length],
                       dest=FExpr("MLIL_CONST_PTR", hex(_MEMCPY), constant=_MEMCPY),
                       params=[FExpr("MLIL_VAR_SSA", "dst#0", reads=[d0]),
                               FExpr("MLIL_VAR_SSA", "src#0", reads=[s0]), var(length)]))
    func = FFunc("count", 0x100, FSSAFunc(body), params=[dst, src])
    return te.TaintEngine(FBV({_MEMCPY: "memcpy"}), te.load_models()).backward(
        func, [te.parse_locator("arg:memcpy:2")])


def _origins(result):
    return [o for sl in result["slices"] for o in sl["origins"]]


@pytest.mark.parametrize("shape, value, increments", [
    ("counter", 0, [0, 1]),
    ("doubling", 1, None),
])
def test_a_loop_counters_initializer_is_no_runtime_constant(shape, value, increments):
    # IC08: `k = 0; for (..) if (..) k++; memcpy(dst, src, k)`. The literal
    # is only where the count STARTS; how far it climbs is decided by the
    # loop's control, which a value slice does not follow. It is a
    # `recurrence_initializer` carrying the value and the per-trip
    # increments, never a `constant` origin of a complete slice.
    result = _counter_program(shape)
    [origin] = _origins(result)
    assert origin["kind"] == "recurrence_initializer", origin
    assert (origin["value"], origin["var"], origin["increments"]) == (value, "k#1", increments)
    assert origin["phi"] == "k#2"
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    # The full slice keeps the phi and the increment it walked.
    steps = {st["address"] for sl in result["slices"] for st in sl["slice"]}
    assert {"0x100", "0x104", "0x108"} <= steps
    # The compact text names the kind, never a `constant` origin.
    from bn.formatters import _render_taint_text
    lines = _render_taint_text(result).splitlines()
    assert any("recurrence_initializer → memcpy" in line for line in lines), lines
    assert [line.split()[:2] for line in lines if line.strip().startswith("origin:")] == [
        ["origin:", "recurrence_initializer"]]


@pytest.mark.parametrize("shape, values", [("choice", {0, 5}), ("literal", {16})])
def test_an_acyclic_constant_stays_a_literal_origin(shape, values):
    # A phi of literals with no cycle, and a plain literal, ARE the value:
    # constant origins (a choice reports its representative literal), a
    # complete slice, no recurrence.
    result = _counter_program(shape)
    origins = _origins(result)
    assert origins and {o["kind"] for o in origins} == {"constant"}, origins
    assert {o["value"] for o in origins} <= values
    assert result["diagnostics"]["safe_to_report_complete_slice"] is True
