"""A sink reached only through an object tainted at an unknown offset says so (V6-01).

A tainted write at an index nothing makes constant (`rec.tag[i] = c`) taints
the whole record, so a later read of a DIFFERENT field (`rec.note`) is a
may-flow: kept, because the byte could have landed there, but not shown to.
The sink carries ``object_taint`` naming the object, the write and the field
offset read, so a consumer can rank it below a field-precise flow; a flow some
write at a stated offset backs carries none.
"""
from __future__ import annotations

import pytest
from test_taint_engine import FExpr, FInstr, FSSA, FSSAFunc, FVar, te
from test_taint_forward_global_frontier import FExtentBV, GFunc

from bn.formatters import _render_flow_line

REC, OUT = 0x404800, 0x404c00     # a 64-byte record {tag[32], note[32]}; a destination
NOTE = 32
STRCPY = 0x401040
STAMP, HELPER, CALLER = 0x401100, 0x401200, 0x401300


@pytest.fixture
def models():
    return te.load_models()


def _ptr(addr):
    return FExpr("MLIL_CONST_PTR", hex(addr), constant=addr)


def _var(ssa):
    return FExpr("MLIL_VAR_SSA", f"{ssa.var.name}#{ssa.version}", reads=[ssa])


def _call(index, addr, target, params, *, writes=()):
    return FInstr(index, addr, "MLIL_CALL_SSA", f"call {hex(target)}",
                  reads=[r for p in params for r in getattr(p, "vars_read", [])],
                  writes=list(writes), dest=_ptr(target), params=list(params))


def _indexed_store(start, v, i):
    """`rec.tag[i] = v[0]`: the address `rec + i`, then a one-byte store."""
    d = FSSA(FVar("d"), 1)
    store = FInstr(1, start + 0x08, "MLIL_STORE_SSA", "[d#1].b = v#0", reads=[v],
                   dest=_var(d), src=_var(v))
    store.size = 1
    return [FInstr(0, start + 0x04, "MLIL_SET_VAR_SSA", "d#1 = rec + i#0",
                   reads=[i], writes=[d],
                   src=FExpr("MLIL_ADD", "rec + i#0", reads=[i],
                             left=_ptr(REC), right=_var(i))),
            store]


def _bv(*funcs):
    return FExtentBV({STRCPY: "strcpy"}, funcs={f.start: f for f in funcs},
                     objects={REC: 64, OUT: 64}, authoritative=[REC, OUT],
                     section=(0x404000, 0x405000))


def _sinks(result):
    return {f["sink"]["address"]: f["sink"] for f in result["reached_sinks"]}


def _forward(models, func, bv):
    return te.TaintEngine(bv, models).forward(func, [te.parse_locator("param:0")])


def test_a_field_read_after_an_unknown_offset_write_is_object_wide(models):
    v, i = FSSA(FVar("v"), 0), FSSA(FVar("i"), 0)
    func = GFunc("object_wide", STAMP, FSSAFunc([
        *_indexed_store(STAMP, v, i),
        _call(2, STAMP + 0x10, STRCPY, [_ptr(OUT), _ptr(REC + NOTE)]),
    ]), params=[v.var, i.var])
    sink = _sinks(_forward(models, func, _bv(func)))[hex(STAMP + 0x10)]
    assert sink["object_taint"] == {"kind": "unknown_offset", "object": hex(REC),
                                    "store": hex(STAMP + 0x08), "field_read": NOTE}


def test_a_field_a_stated_offset_write_tainted_is_field_precise(models):
    # The same read, but the tainted string is copied INTO that field first.
    v = FSSA(FVar("v"), 0)
    func = GFunc("field_precise", STAMP, FSSAFunc([
        _call(0, STAMP + 0x04, STRCPY, [_ptr(REC + NOTE), _var(v)]),
        _call(1, STAMP + 0x10, STRCPY, [_ptr(OUT), _ptr(REC + NOTE)]),
    ]), params=[v.var])
    sinks = _sinks(_forward(models, func, _bv(func)))
    assert sorted(sinks) == [hex(STAMP + 0x04), hex(STAMP + 0x10)]
    assert [s.get("object_taint") for s in sinks.values()] == [None, None]


@pytest.mark.parametrize("field, wide", [(NOTE, False), (0, True)])
def test_a_stated_offset_write_backs_only_the_field_it_covers(models, field, wide):
    # Both writes happen; the copy into `note` covers [32, end) only, so a read
    # of `note` is field-precise and a read of `tag` is still object-wide.
    v, i = FSSA(FVar("v"), 0), FSSA(FVar("i"), 0)
    func = GFunc("mixed", STAMP, FSSAFunc([
        *_indexed_store(STAMP, v, i),
        _call(2, STAMP + 0x0c, STRCPY, [_ptr(REC + NOTE), _var(v)]),
        _call(3, STAMP + 0x10, STRCPY, [_ptr(OUT), _ptr(REC + field)]),
    ]), params=[v.var, i.var])
    sink = _sinks(_forward(models, func, _bv(func)))[hex(STAMP + 0x10)]
    assert (sink.get("object_taint") or {}).get("field_read") == (field if wide else None)


def test_a_field_pointer_returned_after_an_unknown_offset_write_is_object_wide(models):
    # `strcpy(out, note_of(v, i))` where `note_of` stamps the record and returns
    # `rec.note`: the caller's sink reads what the callee's object taint reached.
    v, i = FSSA(FVar("v"), 0), FSSA(FVar("i"), 0)
    helper = GFunc("note_of", HELPER, FSSAFunc([
        *_indexed_store(HELPER, v, i),
        FInstr(2, HELPER + 0x10, "MLIL_RET", "return rec.note", src=[_ptr(REC + NOTE)]),
    ]), params=[v.var, i.var])
    a, n, r = FSSA(FVar("a"), 0), FSSA(FVar("n"), 0), FSSA(FVar("r"), 1)
    caller = GFunc("caller", CALLER, FSSAFunc([
        _call(0, CALLER + 0x04, HELPER, [_var(a), _var(n)], writes=[r]),
        _call(1, CALLER + 0x10, STRCPY, [_ptr(OUT), _var(r)]),
        FInstr(2, CALLER + 0x14, "MLIL_RET", "return", src=[]),
    ]), params=[a.var, n.var], callees=[helper])
    sink = _sinks(_forward(models, caller, _bv(helper, caller)))[hex(CALLER + 0x10)]
    assert sink["object_taint"] == {"kind": "unknown_offset", "object": hex(REC),
                                    "store": hex(HELPER + 0x08), "field_read": NOTE}


def test_a_callee_sink_fed_an_object_wide_argument_is_object_wide(models):
    # The caller stamps the record and hands `rec.note` to a helper whose own
    # walk seeds that parameter: the helper's sink is the same may-flow.
    v, i = FSSA(FVar("v"), 0), FSSA(FVar("i"), 0)
    p = FSSA(FVar("p"), 0)
    helper = GFunc("show", HELPER, FSSAFunc([
        _call(0, HELPER + 0x04, STRCPY, [_ptr(OUT), _var(p)]),
        FInstr(1, HELPER + 0x08, "MLIL_RET", "return", src=[]),
    ]), params=[p.var])
    caller = GFunc("caller", CALLER, FSSAFunc([
        *_indexed_store(CALLER, v, i),
        _call(2, CALLER + 0x10, HELPER, [_ptr(REC + NOTE)]),
        FInstr(3, CALLER + 0x14, "MLIL_RET", "return", src=[]),
    ]), params=[v.var, i.var], callees=[helper])
    sink = _sinks(_forward(models, caller, _bv(helper, caller)))[hex(HELPER + 0x04)]
    assert sink["object_taint"]["store"] == hex(CALLER + 0x08)
    assert sink["object_taint"]["field_read"] == NOTE


def test_the_flow_line_tags_only_the_object_wide_flow(models):
    v, i = FSSA(FVar("v"), 0), FSSA(FVar("i"), 0)
    func = GFunc("mixed", STAMP, FSSAFunc([
        *_indexed_store(STAMP, v, i),
        _call(2, STAMP + 0x0c, STRCPY, [_ptr(REC + NOTE), _var(v)]),
        _call(3, STAMP + 0x10, STRCPY, [_ptr(OUT), _ptr(REC)]),
    ]), params=[v.var, i.var])
    result = _forward(models, func, _bv(func))
    tagged = {f["sink"]["address"]: "[object-wide]" in _render_flow_line(f).split()
              for f in result["reached_sinks"]}
    assert tagged == {hex(STAMP + 0x0c): False, hex(STAMP + 0x10): True}
