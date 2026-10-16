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
    _vphi,
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
        assert any("bytes [0x4, 0x44) of s" in a and "before offset 0x4" in a
                   and "NUL terminator" in a for a in result["assumptions"]), \
            result["assumptions"]
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


# --- round 1: the consumer's demand, returned pointers, pointer objects -----

_B_FGETS, _B_MALLOC = 0x982, 0x983
_B_NAMES.update({_B_FGETS: "fgets", _B_MALLOC: "malloc"})


def _set_ptr(pg, dst, text, src, reads=()):
    i, a = pg._next()
    pg.instrs.append(FInstr(i, a, "MLIL_SET_VAR_SSA", text, writes=[dst], reads=list(reads), src=src))


def _ssa_ref(v):
    e = FExpr("MLIL_VAR_SSA", f"{v.var.name}#{v.version}", reads=[v])
    e.size = 8
    return e


def _const(n):
    return FExpr("MLIL_CONST", hex(n), constant=n)


def _sink_ssa(pg, v):
    return pg.call(_B_POPEN, [FExpr("MLIL_VAR_SSA", f"{v.var.name}#{v.version}", reads=[v]),
                              _cstr(0x710)], reads=[v])


@pytest.mark.parametrize("shape", ["minus", "phi", "loop"])
def test_demand_is_the_consumers_not_the_definitions_r1(shape):
    # `p = b + 8; popen(p - 8)`: the sink reads from offset 0, and only
    # [8, 16) is stored -- incomplete. Rebuilding the region from `p`'s own
    # definition narrowed it to [8, 16) and claimed a complete slice. A phi
    # whose arms disagree (`k ? &b[8] : &b[0]`) and a loop cursor moving back
    # from &b[8] are no single offset either: the whole object is consumed.
    # Each definition the walk reaches alone points into the stored bytes.
    b = FVar("b", typ="char[0x10]")
    pg = _BufProgram()
    pg.aliased_store(b, 8, 0x41).size = 8
    p1, p2, p3 = FSSA(FVar("p"), 1), FSSA(FVar("p"), 2), FSSA(FVar("p"), 3)
    _set_ptr(pg, p1, "p#1 = &b:8", _field_addr(b, 8))
    if shape == "minus":
        _set_ptr(pg, p2, "p#2 = p#1 - 8",
                 FExpr("MLIL_SUB", "p#1 - 8", reads=[p1], left=_ssa_ref(p1), right=_const(8)),
                 reads=[p1])
        sink_v = p2
    elif shape == "phi":
        # p#3 = phi(&b[8], &b[0xc]); p#4 = p#3 - 4; p#5 = phi(p#3, p#4):
        # each definition alone points into the stored [8, 16).
        _set_ptr(pg, p2, "p#2 = &b:0xc", _field_addr(b, 0xc))
        p3, phi = _vphi(FVar("p"), 3, [p1, p2], *pg._next())
        pg.instrs.append(phi)
        p4 = FSSA(FVar("p"), 4)
        _set_ptr(pg, p4, "p#4 = p#3 - 4",
                 FExpr("MLIL_SUB", "p#3 - 4", reads=[p3], left=_ssa_ref(p3), right=_const(4)),
                 reads=[p3])
        sink_v, phi2 = _vphi(FVar("p"), 5, [p3, p4], *pg._next())
        pg.instrs.append(phi2)
    else:
        # p#2 = phi(p#1, p#3); p#3 = p#2 - 1 (the back edge)
        p2, phi = _vphi(FVar("p"), 2, [p1, p3], *pg._next())
        pg.instrs.append(phi)
        _set_ptr(pg, p3, "p#3 = p#2 - 1",
                 FExpr("MLIL_SUB", "p#2 - 1", reads=[p2], left=_ssa_ref(p2), right=_const(1)),
                 reads=[p2])
        sink_v = p2
    _sink_ssa(pg, sink_v)
    result, _, aol = _bw(pg.func())
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    [leaf] = aol
    assert leaf["region"] == {"offset": 0, "size": 0x10} and leaf["unproven"] == [[0, 8]], leaf


def test_demand_narrows_through_arithmetic_on_a_whole_object_pointer_r1():
    # The inverse: `p = (char *)&s; popen(p + 4)` reads from offset 4, so the
    # stores proving [4, 0x44) are enough -- the leading `id` is not demanded.
    s = FVar("s", typ="char[0x44]")
    pg = _BufProgram()
    _qword_stores(pg, s, range(4, 0x44, 8))
    p1, p2 = FSSA(FVar("p"), 1), FSSA(FVar("p"), 2)
    _set_ptr(pg, p1, "p#1 = &s", _addr_of(s))
    _set_ptr(pg, p2, "p#2 = p#1 + 4",
             FExpr("MLIL_ADD", "p#1 + 4", reads=[p1], left=_ssa_ref(p1), right=_const(4)),
             reads=[p1])
    _sink_ssa(pg, p2)
    result, _, aol = _bw(pg.func())
    assert result["diagnostics"]["safe_to_report_complete_slice"] is True, result["leaves"]
    assert any("bytes [0x4, 0x44) of s" in a for a in result["assumptions"]), result["assumptions"]


def test_offset_zero_proof_discloses_the_object_boundary_r1():
    # A whole-object proof at offset 0 rests on the same end-of-object
    # assumption as a member's: it is disclosed, and says the bytes do not
    # establish a terminator.
    buf = FVar("buf", typ="char[0x40]")
    pg = _BufProgram()
    pg.whole_store(buf, 0x41)
    pg.sink(buf)
    result, _, aol = _bw(pg.func())
    assert result["diagnostics"]["safe_to_report_complete_slice"] is True
    [a] = [a for a in result["assumptions"] if "(#933)" in a]
    assert "bytes [0x0, 0x40) of buf" in a and "NUL terminator" in a and "before offset" not in a, a


def test_pointer_returned_into_a_local_checks_the_local_r1():
    # `p = fgets(c, 64, f); write(1, p, 64)`: fgets returns `c`, so the
    # consumer reads c -- whose bytes no call proves. The modeled source stays
    # the origin; the slice is incomplete with c's leaf naming fgets.
    f = FVar("f", ident=51); f0 = FSSA(f, 0)
    c = FVar("c", typ="char[0x40]")
    p1 = FSSA(FVar("p", typ="char*"), 1)
    pg = _BufProgram(params=[f])
    pg.call(_B_FGETS, [_addr_of(c), _const(0x40), FExpr("MLIL_VAR_SSA", "f#0", reads=[f0])],
            reads=[f0], writes=[p1])
    _sink_ssa(pg, p1)
    result, _, aol = _bw(pg.func())
    assert result["slices"][0]["origin"] == {"kind": "source", "callee": "fgets"}
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    [leaf] = aol
    assert leaf["buffer"] == "c" and leaf["calls"] == ["fgets"], leaf

    # An integer return (a count) is not read through: nothing changes.
    n1 = FSSA(FVar("n", typ="int64_t"), 1)
    pg2 = _BufProgram(params=[f])
    pg2.call(_B_FGETS, [_addr_of(c), _const(0x40), FExpr("MLIL_VAR_SSA", "f#0", reads=[f0])],
             reads=[f0], writes=[n1])
    _sink_ssa(pg2, n1)
    result2, _, aol2 = _bw(pg2.func())
    assert aol2 == [] and result2["diagnostics"]["safe_to_report_complete_slice"] is True


@pytest.mark.parametrize("root", ["malloc", "parameter"])
def test_pointer_object_walks_its_writers_and_fails_closed_r1(root):
    # `b = malloc(64); strcpy(b, host); popen(b)` -- and the same with `b` a
    # parameter: no local is behind `b`, so the object's extent and earlier
    # bytes are unknown. Its writers are walked (strcpy and `host` stay in
    # the slice), and a `pointer_object_unresolved` leaf blocks completeness.
    host = FVar("host", ident=11); host0 = FSSA(host, 0)
    b = FVar("b", ident=12, typ="char*")
    params = [host] if root == "malloc" else [host, b]
    pg = _BufProgram(params=params)
    if root == "malloc":
        bv = FSSA(b, 1)
        pg.call(_B_MALLOC, [_const(0x40)], writes=[bv])
    else:
        bv = FSSA(b, 0)
    pg.call(_B_STRCPY, [FExpr("MLIL_VAR_SSA", f"b#{bv.version}", reads=[bv]),
                        FExpr("MLIL_VAR_SSA", "host#0", reads=[host0])], reads=[bv, host0])
    _sink_ssa(pg, bv)
    result, reasons, _ = _bw(pg.func())
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    ptr = [lf for lf in result["leaves"] if lf["kind"] == "pointer_object_unresolved"]
    assert [lf.get("calls") for lf in ptr] == [["strcpy"]], ptr
    assert any("the object b#" in r and "written by strcpy" in r for r in reasons), reasons
    # The writer's input is walked: `host` is a parameter origin (for the
    # parameter root, `b` itself is reached first and ranks equal).
    want = "host#0" if root == "malloc" else "b#0"
    assert result["slices"][0]["origin"]["var"] == want, result["slices"][0]["origin"]
    assert any(st.get("il_text", "").startswith("strcpy") or "strcpy" in st.get("reason", "")
               for st in result["slices"][0]["slice"]), result["slices"][0]["slice"]
    assert "`bn callsites strcpy --within 0x10`" in ptr[0]["detail"], ptr[0]["detail"]


# --- round 2: the selected operand, outside-object demands, object extents ---

def test_demand_is_the_selected_operand_not_another_argument_r2():
    # `strcpy(p + 8, p)` with one SSA `p = &b`, only b[8..16) stored: the
    # query selects the SOURCE (arg 1), which reads from offset 0 -- [0, 8)
    # is unproven. Anchoring the demand on the first argument that reads `p`
    # (the destination, at offset 8) claimed a complete slice. The
    # destination query reads [8, 16) and is complete.
    b = FVar("b", typ="char[16]")
    p1 = FSSA(FVar("p", typ="char*"), 1)
    pg = _BufProgram()
    pg.aliased_store(b, 8, 0x4141).size = 8
    _set_ptr(pg, p1, "p#1 = &b", _addr_of(b))
    ref = _ssa_ref(p1)
    dest = FExpr("MLIL_ADD", "p#1 + 8", reads=[p1], left=ref, right=_const(8))
    dest.size = 8
    pg.call(_B_STRCPY, [dest, ref], reads=[p1])
    src, _, aol = _bw(pg.func(), sink="arg:strcpy:1")
    assert src["diagnostics"]["safe_to_report_complete_slice"] is False, src
    assert [lf["unproven"] for lf in aol] == [[[0, 8]]], aol
    dst, _, aol2 = _bw(pg.func(), sink="arg:strcpy:0")
    assert dst["diagnostics"]["safe_to_report_complete_slice"] is True, dst["leaves"]


@pytest.mark.parametrize("delta, where", [(-12, "0x8 bytes before it"),
                                          (0x10, "at offset 0x14, past its 0x14-byte end")])
def test_demand_outside_the_object_fails_closed_r2(delta, where):
    # A fully stored `struct { int id; char cmd[16]; } s` (0x14 bytes), and a
    # member pointer moved 12 bytes back -- 8 bytes before `s` -- or to its
    # end. The consumer reads bytes outside the object: never clamped back
    # into it (round 0/1 reported complete), an `object_extent_unknown` leaf.
    s = FVar("s", typ="char[0x14]")
    pg = _BufProgram()
    pg.whole_store(s, 0x41)
    p1, p2 = FSSA(FVar("p", typ="char*"), 1), FSSA(FVar("p", typ="char*"), 2)
    _set_ptr(pg, p1, "p#1 = &s:4", _field_addr(s, 4))
    op = "MLIL_SUB" if delta < 0 else "MLIL_ADD"
    _set_ptr(pg, p2, f"p#2 = p#1 {'-' if delta < 0 else '+'} {abs(delta)}",
             FExpr(op, "p#1 +- n", reads=[p1], left=_ssa_ref(p1), right=_const(abs(delta))),
             reads=[p1])
    _sink_ssa(pg, p2)
    result, _, _ = _bw(pg.func())
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "object_extent_unknown"]
    assert leaf["cause"] == "outside_object" and where in leaf["detail"], leaf
    assert leaf["region"]["offset"] == 4 + delta, leaf


def test_inferred_extent_is_no_proof_r2():
    # The same whole-object stores, but BN recovered the local itself (no
    # debug info, no user type): it may be a fragment of the object the
    # consumer reads. The stores are reported as facts; nothing is proven.
    buf = FVar("buf", typ="char[0x40]")
    pg = _BufProgram()
    pg.whole_store(buf, 0x41)
    pg.sink(buf)
    result, _, _ = _bw(pg.func(inferred_vars=[buf]))
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    [leaf] = result["leaves"]
    assert leaf["kind"] == "object_extent_unknown" and leaf["cause"] == "inferred_extent", leaf
    assert leaf["extent"] == "inferred" and leaf["unproven"] == [], leaf
    assert not any("(#933)" in a for a in result["assumptions"]), result["assumptions"]


def test_unknown_offset_is_no_proof_r2():
    # A loop cursor into a fully stored, debug-typed buffer: the consumer's
    # offset is not one constant, so it may leave the object -- widening to
    # the object cannot certify it.
    b = FVar("b", typ="char[0x10]")
    pg = _BufProgram()
    pg.whole_store(b, 0x41)
    p1, p3 = FSSA(FVar("p"), 1), FSSA(FVar("p"), 3)
    _set_ptr(pg, p1, "p#1 = &b:8", _field_addr(b, 8))
    p2, phi = _vphi(FVar("p"), 2, [p1, p3], *pg._next())
    pg.instrs.append(phi)
    _set_ptr(pg, p3, "p#3 = p#2 - 1",
             FExpr("MLIL_SUB", "p#2 - 1", reads=[p2], left=_ssa_ref(p2), right=_const(1)),
             reads=[p2])
    _sink_ssa(pg, p2)
    result, _, _ = _bw(pg.func())
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    [leaf] = result["leaves"]
    assert leaf["kind"] == "object_extent_unknown" and leaf["cause"] == "offset_unknown", leaf


@pytest.mark.parametrize("root", ["malloc", "parameter"])
def test_returned_pointer_over_an_external_object_r2(root):
    # `p = fgets(b, 64, f); write(1, p, 64)` with `b` a heap allocation or a
    # pointer parameter: the return is `b`'s object, which no local is behind.
    # The source stays the origin; the object is a `pointer_object_unresolved`
    # frontier, as when the consumer reads `b` directly.
    f = FVar("f", ident=51); f0 = FSSA(f, 0)
    b = FVar("b", ident=52, typ="char*")
    p1 = FSSA(FVar("p", typ="char*"), 1)
    pg = _BufProgram(params=[f] if root == "malloc" else [f, b])
    if root == "malloc":
        bv = FSSA(b, 1)
        pg.call(_B_MALLOC, [_const(0x40)], writes=[bv])
    else:
        bv = FSSA(b, 0)
    pg.call(_B_FGETS, [FExpr("MLIL_VAR_SSA", f"b#{bv.version}", reads=[bv]), _const(0x40),
                       FExpr("MLIL_VAR_SSA", "f#0", reads=[f0])], reads=[bv, f0], writes=[p1])
    _sink_ssa(pg, p1)
    result, _, _ = _bw(pg.func())
    assert result["slices"][0]["origin"] == {"kind": "source", "callee": "fgets"}, result["slices"]
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    ptr = [lf for lf in result["leaves"] if lf["kind"] == "pointer_object_unresolved"]
    assert any(lf["pointer"] == f"b#{bv.version}" for lf in ptr), result["leaves"]
