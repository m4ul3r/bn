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
    _object_leaves,
    _store_proof_only,
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
        # Every consumed byte is stored: a fact, never certification (#931
        # round 3). The member's region is reported; `id` is not demanded.
        assert _store_proof_only(result) and aol == [], result["leaves"]
        assert result["leaves"][0]["stores_cover"] == [4, 0x44], result["leaves"]
        assert result["slices"][0]["origin"]["kind"] == "constant"
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
    # and the 4-byte store there covers exactly those bytes (a fact: no store
    # proof certifies a slice, #931 round 3) -- the rest of the
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
        assert _store_proof_only(result) is complete, result["leaves"]
        assert result["diagnostics"]["safe_to_report_complete_slice"] is False
        assert result["slices"][0]["origin"]["kind"] == "parameter", result["slices"][0]["origin"]
        if complete:
            assert result["leaves"][0]["stores_cover"] == [8, 12], result["leaves"]
        else:
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
    # length counts as stores only when the call's own LLIL is one contiguous
    # store run of exactly that length from one base -- what BN actually
    # lifted. Covered bytes are a fact; no store proof certifies (round 3).
    result, reasons, aol = _bw(_builtin_memset_program(stores, length).func(),
                               extra_names=_NAMES)
    assert any("written by __builtin_memset" in r for r in reasons), reasons
    assert _store_proof_only(result) is covered, result["leaves"]
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    if covered:
        assert result["leaves"][0]["stores_cover"] == [0, 0x40], result["leaves"]
    else:
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
    assert _store_proof_only(result) is not held_buf, result["leaves"]


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
    # stores covering [4, 0x44) cover the demand -- the leading `id` is not
    # demanded. A fact, not a certified proof (round 3).
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
    assert _store_proof_only(result), result["leaves"]
    assert result["leaves"][0]["stores_cover"] == [4, 0x44], result["leaves"]


def test_store_proof_is_a_fact_never_certification_r3():
    # #931 round 3: stores the walk can see write every byte of the whole,
    # debug-typed buffer -- the region is reported (`stores_cover`), but a
    # region proof from stores never certifies the slice: it rests on BN's
    # recovery of the object (a name-only rename made an inferred fragment
    # look authoritative). The writer's origin stays; `next_action` says how
    # to check by hand, with a command that names the function.
    buf = FVar("buf", typ="char[0x40]")
    pg = _BufProgram()
    pg.whole_store(buf, 0x41)
    pg.sink(buf)
    result, _, aol = _bw(pg.func())
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    [leaf] = _object_leaves(result)
    assert leaf["kind"] == "region_store_proof_not_certified" and aol == [], leaf
    assert leaf["stores_cover"] == [0, 0x40] and leaf["unproven"] == [], leaf
    assert result["slices"][0]["origin"]["kind"] == "constant", result["slices"][0]["origin"]
    assert "`bn il 0x10 --view mlil`" in result["diagnostics"]["next_action"], \
        result["diagnostics"]["next_action"]
    assert not any("(#933)" in a for a in result["assumptions"]), result["assumptions"]


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
    assert (result["slices"][0]["origin"]["kind"], result["slices"][0]["origin"]["callee"]) == ("source", "fgets")
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    [leaf] = aol
    assert leaf["buffer"] == "c" and leaf["calls"] == ["fgets"], leaf

    # An integer return (a count) is not walked as a pointer object: no
    # local leaf. popen still reads through its argument 0 (its model
    # says so, whatever BN typed the value), so only the policy leaf blocks.
    n1 = FSSA(FVar("n", typ="int64_t"), 1)
    pg2 = _BufProgram(params=[f])
    pg2.call(_B_FGETS, [_addr_of(c), _const(0x40), FExpr("MLIL_VAR_SSA", "f#0", reads=[f0])],
             reads=[f0], writes=[n1])
    _sink_ssa(pg2, n1)
    result2, _, aol2 = _bw(pg2.func())
    assert aol2 == [] and result2["diagnostics"]["safe_to_report_complete_slice"] is False
    assert [lf["kind"] for lf in result2["leaves"]] == ["pointer_demand_not_certified"], result2


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
    # destination query reads [8, 16), which the stores cover.
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
    assert _store_proof_only(dst), dst["leaves"]
    assert dst["leaves"][0]["stores_cover"] == [8, 16], dst["leaves"]


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
    # The policy leaf's object carries the CONSUMER's displacement, the same
    # one the region leaf reports -- never the definition's `&s:4` (t-0148).
    [demand] = [lf for lf in result["leaves"] if lf["kind"] == "pointer_demand_not_certified"]
    assert demand["objects"] == [{"kind": "local", "object": "s", "offset": 4 + delta}], demand


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
    [leaf] = _object_leaves(result)
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
    assert (result["slices"][0]["origin"]["kind"], result["slices"][0]["origin"]["callee"]) == ("source", "fgets"), result["slices"]
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    ptr = [lf for lf in result["leaves"] if lf["kind"] == "pointer_object_unresolved"]
    assert any(lf["pointer"] == f"b#{bv.version}" for lf in ptr), result["leaves"]


# --- round 3: returned pointers over writable globals -----------------------

@pytest.mark.parametrize("addr", [0x404040, 0x404064])     # a buffer, a struct member
def test_returned_pointer_over_a_writable_global_r3(addr):
    # `p = fgets(g_buf, 64, f); write(1, p, 64)` -- and the same through
    # `g_req.cmd`: the argument is a constant address with no SSA reads, so
    # the returned-object walk followed nothing and the slice was complete
    # with `source:fgets`. It is the same `global` frontier as consuming the
    # global directly; the source stays the origin.
    from test_taint_engine import FBVStorage, te
    f = FVar("f", ident=51); f0 = FSSA(f, 0)
    p1 = FSSA(FVar("p", typ="char*"), 1)
    pg = _BufProgram(params=[f])
    pg.call(_B_FGETS, [_cstr(addr), _const(0x40), FExpr("MLIL_VAR_SSA", "f#0", reads=[f0])],
            reads=[f0], writes=[p1])
    _sink_ssa(pg, p1)
    bv = FBVStorage(_B_NAMES, ro=[(0x700, 0x800)], rw=[(0x404000, 0x405000)])
    result = te.TaintEngine(bv, te.load_models()).backward(pg.func(), [te.parse_locator("arg:popen:0")])
    assert (result["slices"][0]["origin"]["kind"], result["slices"][0]["origin"]["callee"]) == ("source", "fgets"), result["slices"]
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    glob = [lf for lf in result["leaves"] if lf.get("global")]
    assert [lf["global"] for lf in glob] == [hex(addr)], result["leaves"]
    assert f"`bn xrefs {hex(addr)}`" in glob[0]["detail"], glob[0]["detail"]


# --- re-dogfood of round 3: default deny on the returned-pointer path -------

def _import_ref(name, slot):
    e = FExpr("MLIL_IMPORT", name, constant=slot)
    return e


def test_returned_pointer_over_a_got_loaded_global_is_a_frontier():
    # `popen(fgets(global_buf, 0x40, f), "r")` in a PIC shared object: the
    # preemptible global's address is loaded from the GOT (`rdi#1 =
    # global_buf`, MLIL_IMPORT). Nothing resolved it to a local or a named
    # writable constant, so the slice was complete with `source:fgets`. The
    # object is a `pointer_object_unresolved` frontier naming the GOT slot.
    f = FVar("f", ident=51); f0 = FSSA(f, 0)
    r1 = FSSA(FVar("rdi", typ="char*"), 1)
    p1 = FSSA(FVar("p", typ="char*"), 1)
    pg = _BufProgram(params=[f])
    _set_ptr(pg, r1, "rdi#1 = global_buf", _import_ref("global_buf", 0x403fb8))
    pg.call(_B_FGETS, [_ssa_ref(r1), _const(0x40), FExpr("MLIL_VAR_SSA", "f#0", reads=[f0])],
            reads=[r1, f0], writes=[p1])
    _sink_ssa(pg, p1)
    result, _, _ = _bw(pg.func())
    assert (result["slices"][0]["origin"]["kind"], result["slices"][0]["origin"]["callee"]) == ("source", "fgets")
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "pointer_object_unresolved"]
    assert leaf["pointer"] == "global_buf" and "GOT slot 0x403fb8" in leaf["detail"], leaf


def test_returned_pointer_over_an_unresolvable_object_is_denied_by_default():
    # The object the call was handed reads no SSA value and names no local
    # or writable global (`fgets([0x404000].q, ..)`: a pointer loaded from a
    # constant slot). Whatever it is, it must end in a frontier.
    f = FVar("f", ident=51); f0 = FSSA(f, 0)
    p1 = FSSA(FVar("p", typ="char*"), 1)
    pg = _BufProgram(params=[f])
    loaded = FExpr("MLIL_LOAD_SSA", "[0x404000].q", src=_cstr(0x404000), src_memory=0)
    loaded.size = 8
    pg.call(_B_FGETS, [loaded, _const(0x40), FExpr("MLIL_VAR_SSA", "f#0", reads=[f0])],
            reads=[f0], writes=[p1])
    _sink_ssa(pg, p1)
    result, _, _ = _bw(pg.func())
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    assert any(lf["kind"] == "pointer_object_unresolved" and lf["pointer"] == "[0x404000].q"
               for lf in result["leaves"]), result["leaves"]


def test_direct_read_through_a_got_loaded_global_is_a_frontier():
    # The direct path: `p = flag ? global_buf : arg; popen(p)` -- the
    # parameter arm supplies an origin, so the GOT arm must not vanish as an
    # unresolved definition the parameter outranks.
    a = FVar("a", ident=61, typ="int64_t"); a0 = FSSA(a, 0)
    r1 = FSSA(FVar("p", typ="char*"), 1)
    pg = _BufProgram(params=[a])
    _set_ptr(pg, r1, "p#1 = global_buf", _import_ref("global_buf", 0x403fb8))
    p2, phi = _vphi(FVar("p", typ="char*"), 2, [r1, a0], *pg._next())
    pg.instrs.append(phi)
    _sink_ssa(pg, p2)
    result, _, _ = _bw(pg.func())
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    assert any(lf["kind"] == "pointer_object_unresolved" and lf["pointer"] == "global_buf"
               for lf in result["leaves"]), result["leaves"]


# --- round 5: the policy, not an object classification ----------------------
# A sink that reads through a pointer is complete only when that pointer is a
# constant address of read-only data. Every other pointer demand carries a
# `pointer_demand_not_certified` leaf listing each object it may read; the
# object descriptions are descriptive only.

def _engine_result(pg, **storage):
    from test_taint_engine import FBVStorage, te
    bv = FBVStorage(_B_NAMES, **storage)
    return te.TaintEngine(bv, te.load_models()).backward(pg.func(),
                                                          [te.parse_locator("arg:popen:0")])


def _demand_leaf(result):
    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "pointer_demand_not_certified"]
    return leaf


@pytest.mark.parametrize("segment_writable", [False, True])
def test_literal_sink_is_never_complete_r6(segment_writable):
    # `p = "ls -l"; popen(p, "r")`: round 6 drops the literal exception -- a
    # sink that reads through a pointer is never complete. The literal is a
    # FACT in the leaf, with its segment's writability. The twin is a
    # `.rodata` section inside a writable PT_LOAD (section semantics say
    # read-only, the mapping says writable): the round-5 exception trusted
    # the section and completed it.
    from test_taint_engine import FBVStorage, te

    class View(FBVStorage):
        def is_offset_writable(self, addr):
            return segment_writable or super().is_offset_writable(addr)

    p1 = FSSA(FVar("p", typ="char*"), 1)
    pg = _BufProgram(params=[])
    _set_ptr(pg, p1, "p#1 = \"ls -l\"", _cstr(0x720))
    _sink_ssa(pg, p1)
    bv = View(_B_NAMES, ro=[(0x700, 0x800)])
    result = te.TaintEngine(bv, te.load_models()).backward(pg.func(),
                                                          [te.parse_locator("arg:popen:0")])
    assert result["slices"][0]["origin"]["kind"] == "constant", result["slices"]
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    [leaf] = result["leaves"]
    assert leaf["kind"] == "pointer_demand_not_certified", leaf
    assert leaf["objects"] == [{"kind": "literal", "object": "0x720", "address": "0x720",
                                "offset": 0, "segment_writable": segment_writable}], leaf


@pytest.mark.parametrize("index", ["parameter", "constant"])
def test_got_global_plus_an_index_is_not_complete_r5(index):
    # `popen(global_buf + i)` / `popen(g_req.cmd)` through the GOT: the
    # import is nested under arithmetic. The objects name the GOT global.
    i = FVar("i", ident=71, typ="int64_t"); i0 = FSSA(i, 0)
    g1 = FSSA(FVar("rax", typ="char*"), 1)
    p2 = FSSA(FVar("p", typ="char*"), 2)
    pg = _BufProgram(params=[i])
    _set_ptr(pg, g1, "rax#1 = global_buf", _import_ref("global_buf", 0x403fb8))
    rhs = (FExpr("MLIL_VAR_SSA", "i#0", reads=[i0]) if index == "parameter" else _const(0x10))
    _set_ptr(pg, p2, "p#2 = rax#1 + ...",
             FExpr("MLIL_ADD", "rax#1 + x", left=_ssa_ref(g1), right=rhs,
                   reads=[g1] + ([i0] if index == "parameter" else [])),
             reads=[g1] + ([i0] if index == "parameter" else []))
    _sink_ssa(pg, p2)
    result = _engine_result(pg)
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    objs = _demand_leaf(result)["objects"]
    assert objs == [{"kind": "got_or_extern", "object": "global_buf", "slot": "0x403fb8",
                     "offset": None if index == "parameter" else 0x10}], objs


def test_call_recovered_with_no_destination_is_not_complete_r5():
    # BN recovers `fgets()` with no arguments (a reduced prototype): no
    # argument names the object its return points into, so nothing was
    # walked and the slice was complete with `source:fgets`.
    p1 = FSSA(FVar("p", typ="char*"), 1)
    pg = _BufProgram(params=[])
    pg.call(_B_FGETS, [], reads=[], writes=[p1])
    _sink_ssa(pg, p1)
    result = _engine_result(pg)
    assert (result["slices"][0]["origin"]["kind"], result["slices"][0]["origin"]["callee"]) == ("source", "fgets")
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    [obj] = _demand_leaf(result)["objects"]
    assert obj["kind"] == "returned" and obj["callee"] == "fgets", obj
    assert obj["note"] == "BN recovered no argument the return may point into", obj


def test_every_object_of_a_phi_is_disclosed_r5():
    # `p = flag ? cmd : g_req.cmd (through the GOT); popen(p)`: the local's
    # leaf must not stand in for the GOT member.
    cmd = FVar("cmd", typ="char[0x40]")
    a1 = FSSA(FVar("a", typ="char*"), 1)
    g1 = FSSA(FVar("g", typ="char*"), 1)
    b1 = FSSA(FVar("b", typ="char*"), 1)
    pg = _BufProgram(params=[])
    _set_ptr(pg, a1, "a#1 = &cmd", _addr_of(cmd))
    _set_ptr(pg, g1, "g#1 = g_req", _import_ref("g_req", 0x403fc0))
    _set_ptr(pg, b1, "b#1 = g#1 + 4",
             FExpr("MLIL_ADD", "g#1 + 4", left=_ssa_ref(g1), right=_const(4), reads=[g1]),
             reads=[g1])
    p2, phi = _vphi(FVar("p", typ="char*"), 2, [a1, b1], *pg._next())
    pg.instrs.append(phi)
    _sink_ssa(pg, p2)
    result = _engine_result(pg)
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    kinds = {(o["kind"], o["object"]) for o in _demand_leaf(result)["objects"]}
    assert kinds == {("local", "cmd"), ("got_or_extern", "g_req")}, kinds


# --- round 6: the sink model's role decides pointer vs value ----------------

_B_MEMCPY6, _B_SYSTEM6, _B_HELPER6 = 0x9a0, 0x9a1, 0x9a2
_B_NAMES.update({_B_MEMCPY6: "memcpy", _B_SYSTEM6: "system", _B_HELPER6: "emit_line"})


def _demand_leaves(result):
    return [lf for lf in result["leaves"] if lf["kind"] == "pointer_demand_not_certified"]


def test_object_offset_is_the_consumers_displacement_r6():
    # t-0148: `p = b + 8; popen(p - 8)` reads b at 0 -- the region leaf said
    # 0, the object said 8 (the definition's own offset).
    b = FVar("b", typ="char[0x10]")
    pg = _BufProgram()
    p1, p2 = FSSA(FVar("p", typ="char*"), 1), FSSA(FVar("p", typ="char*"), 2)
    _set_ptr(pg, p1, "p#1 = &b:8", _field_addr(b, 8))
    _set_ptr(pg, p2, "p#2 = p#1 - 8",
             FExpr("MLIL_SUB", "p#1 - 8", reads=[p1], left=_ssa_ref(p1), right=_const(8)),
             reads=[p1])
    _sink_ssa(pg, p2)
    result, _, _ = _bw(pg.func())
    [leaf] = _demand_leaves(result)
    assert leaf["objects"] == [{"kind": "local", "object": "b", "offset": 0}], leaf
    # A variable step: the displacement is unknown, said as such.
    i = FVar("i", ident=81, typ="int64_t"); i0 = FSSA(i, 0)
    pg2 = _BufProgram(params=[i])
    q1, q2 = FSSA(FVar("q", typ="char*"), 1), FSSA(FVar("q", typ="char*"), 2)
    _set_ptr(pg2, q1, "q#1 = &b:8", _field_addr(b, 8))
    _set_ptr(pg2, q2, "q#2 = q#1 + i#0",
             FExpr("MLIL_ADD", "q#1 + i#0", reads=[q1, i0], left=_ssa_ref(q1),
                   right=FExpr("MLIL_VAR_SSA", "i#0", reads=[i0])), reads=[q1, i0])
    _sink_ssa(pg2, q2)
    result2, _, _ = _bw(pg2.func())
    [leaf2] = _demand_leaves(result2)
    assert leaf2["objects"] == [{"kind": "local", "object": "b", "offset": None}], leaf2


def test_untyped_modeled_sink_argument_is_a_pointer_demand_r6():
    # popen's model says its argument 0 is a command it reads through, so an
    # untyped value (an import BN gave no prototype, a stripped register)
    # carries the policy leaf: the role is the model's, not BN's typing.
    n = FVar("n", ident=82); n0 = FSSA(n, 0)
    pg = _BufProgram(params=[n])
    p1 = FSSA(FVar("p"), 1)
    _set_ptr(pg, p1, "p#1 = n#0", FExpr("MLIL_VAR_SSA", "n#0", reads=[n0]), reads=[n0])
    _sink_ssa(pg, p1)
    result, _, _ = _bw(pg.func())
    assert result["slices"][0]["origin"]["kind"] == "parameter", result["slices"]
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    [leaf] = _demand_leaves(result)
    assert leaf["objects"][0]["kind"] == "parameter" and leaf["objects"][0]["object"] == "n#0"


def test_register_recovered_pointer_argument_is_a_pointer_demand_r6(monkeypatch):
    # BN recovered popen with no parameters; the backward seed reads arg 0
    # from its calling-convention register. The policy ran only for an
    # operand in the recovered parameter list, so this completed.
    from test_taint_engine import FBV, te
    n = FVar("n", ident=83); n0 = FSSA(n, 0)
    rdi1 = FSSA(FVar("rdi", typ="char*"), 1)
    pg = _BufProgram(params=[n])
    _set_ptr(pg, rdi1, "rdi#1 = n#0", FExpr("MLIL_VAR_SSA", "n#0", reads=[n0]), reads=[n0])
    call = pg.call(_B_POPEN, [], reads=[])
    func = pg.func()
    engine = te.TaintEngine(FBV(_B_NAMES), te.load_models())
    monkeypatch.setattr(engine, "_reaching_arg_seeds_via_reg",
                        lambda f, sites, idx: ("rdi", [(rdi1, sites[0])]), raising=False)
    result = engine.backward(func, [te.parse_locator("arg:popen:0")])
    assert any("recovered from calling-convention register rdi" in a
               for a in result["assumptions"]), result["assumptions"]
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    [leaf] = _demand_leaves(result)
    assert leaf["recovered_from_register"] is True and leaf["operand"] == "rdi#1", leaf
    assert leaf["objects"][0]["kind"] == "parameter", leaf
    assert call is not None


def test_resolve_map_pinned_sink_uses_the_pinned_models_role_r6():
    # `call [slot](p)` pinned to system by --resolve-map: the locator's callee
    # is the pinned target, so its model decides. The round-5 gate asked
    # resolve_call_target, saw no callee and no type, and completed.
    from test_taint_engine import FBV, FPVS, FFunc, FSSAFunc, te  # noqa: F401
    n = FVar("n", ident=84); slot = FVar("slot")
    n0, slot1 = FSSA(n, 0), FSSA(slot, 1)
    instrs = [FInstr(0, 0x14, "MLIL_CALL_SSA", "[slot#1](n#0)", reads=[slot1, n0],
                     dest=FExpr("MLIL_VAR_SSA", "slot#1", reads=[slot1]),
                     params=[FExpr("MLIL_VAR_SSA", "n#0", reads=[n0])])]
    func = FFunc("dispatch", 0x10, FSSAFunc(instrs), params=[n])
    engine = te.TaintEngine(FBV(_B_NAMES), te.load_models(),
                            resolve_map={"0x14": [hex(_B_SYSTEM6)]})
    result = engine.backward(func, [te.parse_locator("arg:system:0")])
    assert result["slices"][0]["origin"]["kind"] == "parameter", result["slices"]
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    assert len(_demand_leaves(result)) == 1, result["leaves"]


@pytest.mark.parametrize("sink, idx", [("malloc", 0), ("memcpy", 2)])
def test_pointer_typed_scalar_into_a_size_is_a_value_r6(sink, idx):
    # `malloc((size_t)p)` / `memcpy(d, s, (size_t)p)` with `p` typed char*:
    # the model says a size / a length -- a value. No pointer leaf, no
    # pointer object: the slice completes at the parameter, as on base.
    from test_taint_engine import FBV, te
    p = FVar("p", ident=85, typ="char*"); p0 = FSSA(p, 0)
    v1 = FSSA(FVar("v", typ="char*"), 1)
    pg = _BufProgram(params=[p])
    _set_ptr(pg, v1, "v#1 = p#0", FExpr("MLIL_VAR_SSA", "p#0", reads=[p0]), reads=[p0])
    ref = FExpr("MLIL_VAR_SSA", "v#1", reads=[v1])
    if sink == "malloc":
        pg.call(_B_MALLOC, [ref], reads=[v1])
    else:
        pg.call(_B_MEMCPY6, [_cstr(0x404000), _cstr(0x405000), ref], reads=[v1])
    engine = te.TaintEngine(FBV(_B_NAMES), te.load_models())
    result = engine.backward(pg.func(), [te.parse_locator(f"arg:{sink}:{idx}")])
    assert result["leaves"] == [], result["leaves"]
    assert (result["slices"][0]["origin"]["kind"], result["slices"][0]["origin"]["index"], result["slices"][0]["origin"]["var"]) == ("parameter", 0, "p#0")
    assert result["diagnostics"]["safe_to_report_complete_slice"] is True, result


@pytest.mark.parametrize("carrier", ["int64_t", "struct pair", "union word", "double"])
def test_unmodeled_sink_argument_has_no_role_r7(carrier):
    # An ad hoc `arg:emit_line:0`: no model gives the argument a role. Round 6
    # fell back to BN's types and called every KNOWN type a value -- by-value
    # structs, unions and floats included -- so the slice completed. An
    # argument is a value only when a model says so: whatever the carrier,
    # the slice is incomplete with a `sink_role_unknown` leaf that prints the
    # overlay declaring a value role.
    from test_taint_engine import FBV, te
    n = FVar("n", ident=86, typ=carrier); n0 = FSSA(n, 0)
    pg = _BufProgram(params=[n])
    pg.call(_B_HELPER6, [FExpr("MLIL_VAR_SSA", "n#0", reads=[n0])], reads=[n0])
    result = te.TaintEngine(FBV(_B_NAMES), te.load_models()).backward(
        pg.func(), [te.parse_locator("arg:emit_line:0")])
    assert result["slices"][0]["origin"]["kind"] == "parameter", result["slices"]
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    [leaf] = result["leaves"]
    assert leaf["kind"] == "sink_role_unknown", leaf
    assert (leaf["callee"], leaf["arg_index"]) == ("emit_line", 0), leaf
    # Round 8: guidance, not a generated command. A TEMPLATE entry as data,
    # UPPER-CASE placeholders, no redirection, no rebuilt re-run.
    decl = leaf["declare_value_role"]
    assert decl["template"] is True, decl
    assert decl["entry"] == {"emit_line": {
        "sink": {"class": "VALUE_CLASS", "tainted_args": [0]},
        "OTHER_ROLES": "COPY_FROM_THE_CURRENT_MODEL"}}, decl
    assert decl["placeholders"]["VALUE_CLASS"] == ["alloc_size", "fortified_overflow",
                                                   "overflow_len"], decl
    advice = result["diagnostics"]["next_action"]
    assert ">" not in advice and "printf" not in advice, advice
    assert "bn taint backward" not in advice and "add `--models FILE` to the same command" in advice
    assert "`bn taint models --format json`" in advice, advice


def test_declared_value_role_completes_the_ad_hoc_sink_r7():
    # The printed overlay is the explicit claim: with it the argument is a
    # value, it carries no demand, and the slice completes at its parameter.
    from test_taint_engine import FBV, te
    n = FVar("n", ident=86, typ="int64_t"); n0 = FSSA(n, 0)
    pg = _BufProgram(params=[n])
    pg.call(_B_HELPER6, [FExpr("MLIL_VAR_SSA", "n#0", reads=[n0])], reads=[n0])
    engine = te.TaintEngine(FBV(_B_NAMES), te.load_models())
    entry = engine.backward(pg.func(), [te.parse_locator("arg:emit_line:0")])["leaves"][0][
        "declare_value_role"]["entry"]
    # Fill the template: a value class; emit_line has no current model, so
    # there are no other roles to copy.
    entry["emit_line"]["sink"]["class"] = "alloc_size"
    del entry["emit_line"]["OTHER_ROLES"]
    declared = te.TaintEngine(FBV(_B_NAMES), te.load_models(entry)).backward(
        pg.func(), [te.parse_locator("arg:emit_line:0")])
    assert declared["leaves"] == [], declared["leaves"]
    assert declared["diagnostics"]["safe_to_report_complete_slice"] is True, declared


def test_direct_literal_command_is_incomplete_not_refused_r7():
    # `popen("ls", "r")`: the operand reads no variable. It was a refusal
    # (exit 3, "reads no variable"); popen reads through it, so it is an
    # incomplete answer whose leaf names the literal.
    from test_taint_engine import FBVStorage, te
    pg = _BufProgram(params=[])
    pg.call(_B_POPEN, [_cstr(0x720), _cstr(0x710)])
    bv = FBVStorage(_B_NAMES, ro=[(0x700, 0x800)])
    result = te.TaintEngine(bv, te.load_models()).backward(pg.func(),
                                                          [te.parse_locator("arg:popen:0")])
    assert result["slices"] == []
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    [leaf] = result["leaves"]
    assert leaf["kind"] == "pointer_demand_not_certified", leaf
    assert leaf["objects"] == [{"kind": "literal", "object": "0x720", "address": "0x720",
                                "offset": 0, "segment_writable": False}], leaf


@pytest.mark.parametrize("other", ["unseeded_pointer", "uncalled_sink"])
def test_multi_sink_is_complete_only_when_every_operand_is_r7(other):
    # The reviewer's case: a fixed address to popen (no seed) beside a
    # parameter to malloc's size (a complete value slice). The popen operand
    # was refused and skipped, and the malloc slice made the whole result
    # complete with no leaf. An unseeded operand -- or a sink the query
    # selected but the function never calls -- counts, with its leaf.
    from test_taint_engine import FBV, te
    n = FVar("n", ident=89, typ="int64_t"); n0 = FSSA(n, 0)
    pg = _BufProgram(params=[n])
    pg.call(_B_POPEN, [FExpr("MLIL_CONST", "0x4000", constant=0x4000), _cstr(0x710)])
    pg.call(_B_MALLOC, [FExpr("MLIL_VAR_SSA", "n#0", reads=[n0])], reads=[n0])
    sinks = ["arg:malloc:0", "arg:popen:0" if other == "unseeded_pointer" else "arg:system:0"]
    result = te.TaintEngine(FBV(_B_NAMES), te.load_models()).backward(
        pg.func(), [te.parse_locator(s) for s in sinks])
    assert result["slices"][0]["origin"]["kind"] == "parameter", result["slices"]
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result
    want = "pointer_demand_not_certified" if other == "unseeded_pointer" else "sink_unseeded"
    assert [lf["kind"] for lf in result["leaves"]] == [want], result["leaves"]
    # The malloc slice alone is still complete.
    alone = te.TaintEngine(FBV(_B_NAMES), te.load_models()).backward(
        pg.func(), [te.parse_locator("arg:malloc:0")])
    assert alone["diagnostics"]["safe_to_report_complete_slice"] is True, alone


def test_mixed_sink_names_its_buffer_in_pointer_args_r6():
    # `app_send(fd, buf, len)` as a net_write overlay: the buffer it names in
    # `pointer_args` is a pointer demand, the length a value. A malformed
    # `pointer_args` is refused at load.
    from test_taint_engine import FBV, te
    _B_NAMES[0x9a3] = "app_send"
    overlay = {"app_send": {"sink": {"class": "net_write", "tainted_args": [1, 2],
                                     "pointer_args": [1]}}}
    b = FVar("b", ident=87); n = FVar("n", ident=88, typ="int64_t")
    b0, n0 = FSSA(b, 0), FSSA(n, 0)
    pg = _BufProgram(params=[b, n])
    pg.call(0x9a3, [_const(1), FExpr("MLIL_VAR_SSA", "b#0", reads=[b0]),
                    FExpr("MLIL_VAR_SSA", "n#0", reads=[n0])], reads=[b0, n0])
    engine = te.TaintEngine(FBV(_B_NAMES), te.load_models(overlay))
    buf = engine.backward(pg.func(), [te.parse_locator("arg:app_send:1")])
    assert buf["diagnostics"]["safe_to_report_complete_slice"] is False, buf
    assert len(_demand_leaves(buf)) == 1, buf["leaves"]
    length = engine.backward(pg.func(), [te.parse_locator("arg:app_send:2")])
    assert length["leaves"] == [] and length["diagnostics"]["safe_to_report_complete_slice"]
    with pytest.raises(te.TaintError, match="pointer_args"):
        te.load_models({"app_send": {"sink": {"class": "net_write", "tainted_args": [1],
                                              "pointer_args": ["1"]}}})


def test_role_advice_survives_a_backtick_in_the_name_and_rebuilds_no_query_r8():
    # A callee named with a backtick must not unbalance the leaf's inline
    # code (nor feed the `bn ...` router a fake command), and a two-sink
    # query gets no rebuilt one-sink re-run.
    from test_taint_engine import FBV, te
    _B_NAMES[0x9a4] = "emit`line"
    n = FVar("n", ident=90, typ="int64_t"); n0 = FSSA(n, 0)
    pg = _BufProgram(params=[n])
    pg.call(0x9a4, [FExpr("MLIL_VAR_SSA", "n#0", reads=[n0])], reads=[n0])
    pg.call(_B_MALLOC, [FExpr("MLIL_VAR_SSA", "n#0", reads=[n0])], reads=[n0])
    result = te.TaintEngine(FBV(_B_NAMES), te.load_models()).backward(
        pg.func(), [te.parse_locator("arg:emit`line:0"), te.parse_locator("arg:malloc:0")])
    [leaf] = result["leaves"]
    assert leaf["callee"] == "emit`line" and '"emit`line"' in leaf["detail"], leaf["detail"]
    for text in (leaf["detail"], result["diagnostics"]["next_action"]):
        stripped = text.replace('"emit`line"', "")
        assert stripped.count("`") % 2 == 0, text
        assert "bn taint backward" not in text and "--models value_roles" not in text, text
