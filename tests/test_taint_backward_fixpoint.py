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

_MEMCPY, _TRANSFORM, _RESET = 0xc40, 0xc50, 0xc60


def _counter_program(shape, copies=64):
    """`count(dst, src)`: a length `k` in one of these shapes, then
    `memcpy(dst, src, k)`.

    * ``counter``: `k = 0; loop { if (..) k = k + 1; }` -- the -O0 lift of a
      conditional count, `k#2 = phi(k#1, k#4)`, `k#4 = phi(k#2, k#3)`;
    * ``doubling``: `k = 1; loop { k = k * 2; }` -- a recurrence with no
      constant per-trip increment;
    * ``copied``: `k = 0; loop { k = k + 1; }` with *copies* register copies
      on the back arm between the increment and the phi;
    * ``nested``: `k = 0; loop { loop { k = k + 1; } }`, the inner phi
      reached twice per outer trip;
    * ``call``: `k = 0; loop { k = transform(k); }`;
    * ``call_return``: the same loop with the length read straight off the
      call's return (the -O2 lift: `memcpy(.., transform(k))`);
    * ``call_of_choice``: `k = c ? 0 : 5; n = transform(k)` -- a call handed
      a phi of literals, no loop;
    * ``mixed``: `k = 0; loop { if (..) k = k + 1; else k = reset(); }` -- a
      recurrence beside a stronger call origin on another back edge;
    * ``unrelated``: `k = 0; n = k; loop { j = k..; j++ }` with the length `n`
      -- the literal also starts a loop the slice never walks;
    * ``choice``: `k = c ? 0 : 5` -- a phi of two literals, no cycle;
    * ``carried``: `k = 7; loop { k = k; }` -- a cycle that never changes it;
    * ``literal``: `k = 16`."""
    from test_taint_engine import FBV, FFunc, FInstr, FSSA, FSSAFunc

    dst, src = FVar("dst", ident=1, typ="char*"), FVar("src", ident=2, typ="char*")
    kv = FVar("k", ident=3)
    k = [FSSA(kv, i) for i in range(copies + 8)]

    def var(s):
        return FExpr("MLIL_VAR_SSA", f"k#{s.version}", reads=[s], src=s)

    def const(n):
        return FExpr("MLIL_CONST", str(n), constant=n)

    def set_(i, dest, src_expr, reads=()):
        return FInstr(i, 0x100 + 4 * i, "MLIL_SET_VAR_SSA", f"k#{dest.version} = {src_expr}",
                      reads=list(reads), writes=[dest], dest=dest, src=src_expr)

    def inc(i, dest, of):
        return set_(i, dest, FExpr("MLIL_ADD", f"k#{of.version} + 1", reads=[of],
                                   left=var(of), right=const(1)), reads=[of])

    def phi(i, dest, *arms):
        ins = FInstr(i, 0x100 + 4 * i, "MLIL_VAR_PHI",
                     f"k#{dest.version} = phi({', '.join(f'k#{a.version}' for a in arms)})",
                     reads=list(arms), writes=[dest])
        ins.dest, ins.src = dest, list(arms)
        return ins

    def call(i, dest, target, args):
        return FInstr(i, 0x100 + 4 * i, "MLIL_CALL_SSA", f"k#{dest.version} = {target:#x}(..)",
                      reads=list(args), writes=[dest],
                      dest=FExpr("MLIL_CONST_PTR", hex(target), constant=target),
                      params=[var(a) for a in args])

    if shape == "counter":
        body = [set_(0, k[1], const(0)), phi(1, k[2], k[1], k[4]), inc(2, k[3], k[2]),
                phi(3, k[4], k[2], k[3])]
        length = k[2]
    elif shape == "doubling":
        body = [set_(0, k[1], const(1)), phi(1, k[2], k[1], k[3]),
                set_(2, k[3], FExpr("MLIL_MUL", "k#2 * 2", reads=[k[2]], left=var(k[2]),
                                    right=const(2)), reads=[k[2]])]
        length = k[2]
    elif shape == "copied":
        last = k[3 + copies]
        body = [set_(0, k[1], const(0)), phi(1, k[2], k[1], last), inc(2, k[3], k[2])]
        body += [set_(3 + c, k[4 + c], var(k[3 + c]), reads=[k[3 + c]]) for c in range(copies)]
        length = k[2]
    elif shape == "nested":
        body = [set_(0, k[1], const(0)), phi(1, k[2], k[1], k[5]), phi(2, k[3], k[2], k[4]),
                inc(3, k[4], k[3]), set_(4, k[5], var(k[3]), reads=[k[3]])]
        length = k[2]
    elif shape == "call":
        body = [set_(0, k[1], const(0)), phi(1, k[2], k[1], k[3]),
                call(2, k[3], _TRANSFORM, [k[2]])]
        length = k[2]
    elif shape == "call_return":
        body = [set_(0, k[1], const(0)), phi(1, k[2], k[1], k[3]),
                call(2, k[3], _TRANSFORM, [k[2]])]
        length = k[3]
    elif shape == "call_of_choice":
        body = [set_(0, k[1], const(0)), set_(1, k[2], const(5)), phi(2, k[3], k[1], k[2]),
                call(3, k[4], _TRANSFORM, [k[3]])]
        length = k[4]
    elif shape == "mixed":
        body = [set_(0, k[1], const(0)), phi(1, k[2], k[1], k[3], k[4]), inc(2, k[3], k[2]),
                call(3, k[4], _RESET, [])]
        length = k[2]
    elif shape == "unrelated":
        body = [set_(0, k[1], const(0)), set_(1, k[2], var(k[1]), reads=[k[1]]),
                phi(2, k[3], k[1], k[4]), inc(3, k[4], k[3])]
        length = k[2]
    elif shape == "choice":
        body = [set_(0, k[1], const(0)), set_(1, k[2], const(5)), phi(2, k[3], k[1], k[2])]
        length = k[3]
    elif shape == "carried":
        body = [set_(0, k[1], const(7)), phi(1, k[2], k[1], k[3]),
                set_(2, k[3], var(k[2]), reads=[k[2]])]
        length = k[2]
    else:
        body = [set_(0, k[1], const(16))]
        length = k[1]
    d0, s0 = FSSA(dst, 0), FSSA(src, 0)
    body.append(FInstr(len(body), 0x400, "MLIL_CALL_SSA", f"memcpy(dst#0, src#0, k#{length.version})",
                       reads=[d0, s0, length],
                       dest=FExpr("MLIL_CONST_PTR", hex(_MEMCPY), constant=_MEMCPY),
                       params=[FExpr("MLIL_VAR_SSA", "dst#0", reads=[d0]),
                               FExpr("MLIL_VAR_SSA", "src#0", reads=[s0]), var(length)]))
    func = FFunc("count", 0x100, FSSAFunc(body), params=[dst, src])
    names = {_MEMCPY: "memcpy", _TRANSFORM: "transform", _RESET: "reset"}
    return te.TaintEngine(FBV(names), te.load_models()).backward(
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


@pytest.mark.parametrize("shape, values", [
    ("choice", {0, 5}), ("literal", {16}), ("carried", {7}), ("unrelated", {0})])
def test_a_literal_that_is_the_value_stays_a_constant_origin(shape, values):
    # A phi of literals with no cycle, a loop that only carries the literal,
    # and a plain literal ARE the value: constant origins (a choice reports
    # its representative literal), a complete slice, no recurrence. So is a
    # literal reaching the sink by an acyclic copy, though it also starts a
    # loop the slice never walks (E2c: every use of the literal was scanned).
    result = _counter_program(shape)
    origins = _origins(result)
    assert origins and {o["kind"] for o in origins} == {"constant"}, origins
    assert {o["value"] for o in origins} <= values
    assert result["diagnostics"]["safe_to_report_complete_slice"] is True


def test_a_call_handed_a_choice_of_literals_is_only_the_call():
    # Control for V7-03: the call's argument is a phi with no back edge, so
    # its return is just the call's -- one `call` origin, a complete slice.
    result = _counter_program("call_of_choice")
    assert [(o["kind"], o.get("callee")) for o in _origins(result)] == [("call", "transform")]
    assert result["diagnostics"]["safe_to_report_complete_slice"] is True


def _recurrence(result):
    return [o for o in _origins(result) if o["kind"] == "recurrence_initializer"]


@pytest.mark.parametrize("shape, increments, calls, others", [
    # A back arm through ten copies: well within the search.
    pytest.param("copied", [1], None, set(), id="copied"),
    # An inner loop reaching the outer phi twice per trip: its trip count
    # makes the outer increment unknown -- never a fresh search state per
    # inner trip (E2c: it exhausted the budget and came back `constant`).
    pytest.param("nested", None, None, set(), id="nested"),
    # `k = transform(k)`: the update is a call, named (E2c: skipped).
    pytest.param("call", None, ["transform"], {"call"}, id="call"),
    # A reset call on another back edge outranks the literal, yet the
    # recurrence stays an origin of its own (E2c: the call replaced it).
    pytest.param("mixed", [1], None, {"call"}, id="mixed"),
    # The slice stops at the call whose return the sink reads, but the value
    # it was handed is the loop's phi (V7-03: only the call, complete).
    pytest.param("call_return", None, ["transform"], {"call"}, id="call-return"),
])
def test_every_recurrence_shape_keeps_its_initializer(shape, increments, calls, others):
    result = _counter_program(shape, copies=10)
    [origin] = _recurrence(result)
    assert (origin["value"], origin["var"], origin["phi"]) == (0, "k#1", "k#2"), origin
    assert origin["increments"] == increments, origin
    assert origin.get("calls") == calls, origin
    assert "search_exhausted" not in origin
    assert {o["kind"] for o in _origins(result)} - {"recurrence_initializer"} == others
    assert "constant" not in {o["kind"] for o in _origins(result)}
    assert result["stats"]["truncation_cause"] == []
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    assert _names_the_initializer(result)


@pytest.mark.parametrize("copies, exhausted", [(62, False), (63, True)])
def test_a_recurrence_search_that_runs_out_is_a_disclosed_cut(copies, exhausted):
    # The search expands the literal arm, the increment and each copy on the
    # back arm once; with 63 copies the phi lies past its 64 expansions. A
    # search that ran out has not ruled the loop out (E2c: it returned "no
    # cycle", so `constant 0`, complete); one copy fewer it proves the +1.
    result = _counter_program("copied", copies=copies)
    [origin] = _origins(result)
    assert origin["kind"] == "recurrence_initializer", origin
    assert (origin["increments"], origin.get("search_exhausted", False)) == (
        None if exhausted else [1], exhausted)
    assert result["stats"]["truncated"] is exhausted
    assert result["stats"]["truncation_cause"] == (
        ["recurrence_search_budget"] if exhausted else [])
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    assert _names_the_initializer(result)


def _names_the_initializer(result):
    """Whether the text's origin row lists `k#1 = 0` as a recurrence
    initializer -- kind, variable and value -- never as a constant."""
    from bn.formatters import _render_taint_text
    rows = [line for line in _render_taint_text(result).splitlines()
            if line.strip().startswith(("origin:", "origins ("))]
    return (any("recurrence_initializer k#1 = 0x0" in row for row in rows)
            and not any("constant 0x0" in row for row in rows))
