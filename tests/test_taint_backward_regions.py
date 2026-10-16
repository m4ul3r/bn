"""Backward coverage keyed to the consumed region (#933), proven only by plain
stores (#931), plus the K-review alias follow-ups S1/S2."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_taint_engine import (
    _B_NAMES,
    _B_POPEN,
    _B_STRCPY,
    FSSA,
    FExpr,
    FInstr,
    FOp,
    FVar,
    _addr_of,
    _BufProgram,
    _bw,
    _cstr,
)

_B_BUILTIN_MEMSET, _B_STRSEP = 0x980, 0x981
_NAMES = {_B_BUILTIN_MEMSET: "__builtin_memset", _B_STRSEP: "strsep"}
_B_NAMES.update(_NAMES)    # as test_taint_engine registers its own extra targets


def _field_addr(v, off):
    e = FExpr("MLIL_ADDRESS_OF_FIELD", f"&{v.name}:{off:#x}", src=v)
    e.offset = off
    return e


def _sink_at(pg, ptr):
    """`p#1 = <ptr>; popen(p#1, "r")`."""
    p1 = FSSA(FVar("p"), 1)
    i, a = pg._next()
    pg.instrs.append(FInstr(i, a, "MLIL_SET_VAR_SSA", f"p#1 = {ptr}", writes=[p1], src=ptr))
    return pg.call(_B_POPEN, [FExpr("MLIL_VAR_SSA", "p#1", reads=[p1]), _cstr(0x710)], reads=[p1])


def _qword_stores(pg, buf, offsets, value=0x41):
    for off in offsets:
        pg.aliased_store(buf, off, value).size = 8


@pytest.mark.parametrize("gap, sink_off, unproven", [
    (None, 4, None),             # every byte from the member to the end: complete
    (0x24, 4, [[0x24, 0x2c]]),   # one qword missing inside the member
    (None, 0, [[0, 4]]),         # the whole struct: the leading int was never written
])
def test_store_proven_region_from_the_sink_offset_933(gap, sink_off, unproven):
    # `struct { int id; char cmd[64]; } s;` filled by qword stores into cmd
    # only, `popen(s.cmd)`: coverage is keyed to the bytes from the sink's
    # offset to the end of the object, not to the enclosing aggregate.
    s = FVar("s", typ="char[0x44]")
    pg = _BufProgram()
    _qword_stores(pg, s, [o for o in range(4, 0x44, 8) if o != gap])
    _sink_at(pg, _field_addr(s, sink_off) if sink_off else _addr_of(s))
    result, _, aol = _bw(pg.func())
    complete = result["diagnostics"]["safe_to_report_complete_slice"]
    if unproven is None:
        assert complete is True and aol == [], result["leaves"]
        assert result["slices"][0]["origin"]["kind"] == "constant"
        assert any("offset 0x4 of local buffer s" in a and "(#933)" in a
                   for a in result["assumptions"]), result["assumptions"]
    else:
        assert complete is False
        [leaf] = aol
        assert leaf["unproven"] == unproven, leaf
        assert leaf["region"] == {"offset": sink_off, "size": 0x44 - sink_off}, leaf
        assert "calls" not in leaf, leaf


def test_trailing_member_is_part_of_the_consumed_region_933():
    # A forward reader may run past an unterminated member into the next one:
    # `struct { char cmd[8]; char tail[56]; }` with only cmd stored is NOT
    # complete for `popen(s.cmd)` -- cmd at offset 0 is the whole object.
    s = FVar("s", typ="char[0x40]")
    pg = _BufProgram()
    _qword_stores(pg, s, [0])
    _sink_at(pg, _addr_of(s))
    result, _, aol = _bw(pg.func())
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    assert aol[0]["unproven"] == [[8, 0x40]], aol


@pytest.mark.parametrize("via_pointer", [False, True])
def test_store_proven_load_region_931(via_pointer):
    # `*(int *)&buf[8] = n; malloc(*(int *)&buf[8])`: the load reads [8, 12),
    # and the 4-byte store there proves exactly those bytes -- the rest of the
    # buffer is not consumed. A 2-byte store leaves [10, 12) unproven. Through
    # `p = &buf; p->len` the pointer is the address, not data: walking it as a
    # value must not demand the whole buffer again.
    for width, complete in ((4, True), (2, False)):
        n = FVar("n", ident=31); n0 = FSSA(n, 0)
        buf = FVar("buf", typ="char[0x40]")
        pg = _BufProgram(params=[n])
        st = pg.store(_field_addr(buf, 8), FExpr("MLIL_VAR_SSA", "n#0", reads=[n0]), reads=[n0])
        st.size = width
        x1 = FSSA(FVar("x"), 1)
        addr = _field_addr(buf, 8)
        if via_pointer:
            q1 = FSSA(FVar("q"), 1)
            i, a = pg._next()
            pg.instrs.append(FInstr(i, a, "MLIL_SET_VAR_SSA", "q#1 = &buf", writes=[q1],
                                    src=_addr_of(buf)))
            qv = FExpr("MLIL_VAR_SSA", "q#1", reads=[q1])
            qv.size = 8
            addr = FExpr("MLIL_ADD", "q#1 + 8", reads=[q1], left=qv,
                         right=FExpr("MLIL_CONST", "8", constant=8))
        ld = FExpr("MLIL_LOAD_SSA", "[addr].d", src=addr, src_memory=pg.mem,
                   reads=list(getattr(addr, "vars_read", [])))
        ld.size = 4
        i, a = pg._next()
        pg.instrs.append(FInstr(i, a, "MLIL_SET_VAR_SSA", "x#1 = [&buf:0x8].d", writes=[x1], src=ld))
        pg.call(_B_POPEN, [FExpr("MLIL_VAR_SSA", "x#1", reads=[x1]), _cstr(0x710)], reads=[x1])
        result, _, aol = _bw(pg.func())
        assert result["diagnostics"]["safe_to_report_complete_slice"] is complete, result["leaves"]
        assert result["slices"][0]["origin"]["kind"] == "parameter", result["slices"][0]["origin"]
        if not complete:
            assert aol[0]["unproven"] == [[10, 12]], aol


def _llil_store(base, off, size):
    reg = FExpr("LLIL_REG_SSA", base)
    dest = reg if off == 0 else FExpr(
        "LLIL_ADD" if off > 0 else "LLIL_SUB", f"{base} + {off}", left=reg,
        right=FExpr("LLIL_CONST", str(abs(off)), constant=abs(off)))
    return SimpleNamespace(operation=FOp("LLIL_STORE_SSA"), dest=dest, size=size)


def _builtin_memset_program(stores, length=0x40):
    buf = FVar("buf", typ="char[0x40]")
    pg = _BufProgram()
    call = pg.call(_B_BUILTIN_MEMSET, [_addr_of(buf), FExpr("MLIL_CONST", "0", constant=0),
                                       FExpr("MLIL_CONST", hex(length), constant=length)])
    call.llils = stores
    _sink_at(pg, _addr_of(buf))
    return pg


_RUN = [_llil_store("rbp#1", -0x40 + 8 * k, 8) for k in range(8)]


@pytest.mark.parametrize("stores, length, covered", [
    (_RUN, 0x40, True),                                   # rbp-0x40 .. rbp: 64 bytes
    (_RUN[:3] + _RUN[4:], 0x40, False),                   # a gap in the run
    (_RUN, 0x48, False),                                  # stated length disagrees
    (_RUN[:-1] + [_llil_store("rax#1", 0, 8)], 0x40, False),  # two bases
    (_RUN + [SimpleNamespace(operation=FOp("LLIL_CALL_SSA"))], 0x40, False),  # a real call
    ([], 0x40, False),                                    # no stores at all
])
def test_builtin_fill_counts_only_when_its_llil_stores_say_so_931(stores, length, covered):
    # BN outlines inline stores as `__builtin_memset(&buf, 0, 0x40)`. Its
    # length is trusted only when the call's own LLIL is one contiguous store
    # run of exactly that length from one base -- what BN actually lifted.
    result, reasons, aol = _bw(_builtin_memset_program(stores, length).func(),
                               extra_names=_NAMES)
    assert any("written by __builtin_memset" in r for r in reasons), reasons
    assert result["diagnostics"]["safe_to_report_complete_slice"] is covered, result["leaves"]
    if not covered:
        assert aol[0]["calls"] == ["__builtin_memset"], aol
        # No `bn callsites` for a synthetic builtin: BN has no symbol to list.
        assert "callsites" not in aol[0]["detail"], aol[0]["detail"]


def test_struct_field_pointer_is_a_may_alias_s1():
    # S1: `struct { long n; char *p; } s; s.p = buf; strcpy(s.p, host)` with s
    # held in registers (MLIL_SET_VAR_SSA_FIELD): the pointer read back out of
    # the field was "unrelated" before any escape, so strcpy was dropped and
    # the store-initialized buffer read as complete with origin constant.
    host = FVar("host", ident=11); host0 = FSSA(host, 0)
    buf = FVar("buf", typ="char[0x40]")
    s = FVar("s", ident=40); s0, s1 = FSSA(s, 0), FSSA(s, 1)
    q2 = FSSA(FVar("q"), 2)
    pg = _BufProgram(params=[host])
    pg.whole_store(buf)
    i, a = pg._next()
    pg.instrs.append(FInstr(i, a, "MLIL_SET_VAR_SSA_FIELD", "s#1:8 = &buf", writes=[s1],
                            reads=[s0], dest=s1, src=_addr_of(buf), prev=s0, offset=8))
    i, a = pg._next()
    pg.instrs.append(FInstr(i, a, "MLIL_SET_VAR_SSA", "q#2 = s#1:8.q", writes=[q2],
                            src=FExpr("MLIL_VAR_SSA_FIELD", "s#1:8.q", reads=[s1])))
    pg.call(_B_STRCPY, [FExpr("MLIL_VAR_SSA", "q#2", reads=[q2]),
                        FExpr("MLIL_VAR_SSA", "host#0", reads=[host0])], reads=[q2, host0])
    _sink_at(pg, _addr_of(buf))
    result, _, aol = _bw(pg.func())
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    assert any(l.get("callee") == "strcpy" for l in aol), aol


@pytest.mark.parametrize("held_buf", [True, False])
def test_pointee_to_return_rule_after_an_escape_is_a_may_alias_s2(held_buf):
    # S2: `char *p = buf; char *t = strsep(&p, ","); strcpy(t, host)` -- the
    # `*arg:0 -> ret` rule was judged by whether `&p` aliases buf (it does
    # not), so strcpy was dropped. strsep returns what p HELD; once &buf has
    # escaped (here into p), that may be a pointer into buf. With p holding
    # another local nothing escaped and the return stays unrelated.
    host = FVar("host", ident=11); host0 = FSSA(host, 0)
    buf = FVar("buf", typ="char[0x40]")
    other = FVar("other", typ="char[0x40]")
    p = FVar("p", ident=41)
    pg = _BufProgram(params=[host])
    pg.whole_store(buf)
    ptr_store = pg.whole_store(p)
    ptr_store.src = _addr_of(buf if held_buf else other)
    t1 = FSSA(FVar("t"), 1)
    pg.call(_B_STRSEP, [_addr_of(p), _cstr(0x720)], writes=[t1])
    pg.call(_B_STRCPY, [FExpr("MLIL_VAR_SSA", "t#1", reads=[t1]),
                        FExpr("MLIL_VAR_SSA", "host#0", reads=[host0])], reads=[t1, host0])
    _sink_at(pg, _addr_of(buf))
    result, _, aol = _bw(pg.func(), extra_names=_NAMES)
    assert any(l.get("callee") == "strcpy" for l in aol) is held_buf, aol
    assert result["diagnostics"]["safe_to_report_complete_slice"] is not held_buf
