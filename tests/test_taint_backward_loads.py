"""Backward taint through memory loads and stashed pointers (#1041 item 2,
#1034 items 1-3): a loaded value comes from the writers of the loaded bytes on
every path, never from the pointer it was loaded through, and a stashed
pointer is the buffer only when one store definitely put it there."""
from __future__ import annotations

import pytest
from test_taint_engine import (
    _B_MEMCPY,
    _B_MEMSET,
    _B_NAMES,
    _B_STRCPY,
    FBV,
    FSSA,
    FExpr,
    FFunc,
    FInstr,
    FSSAFunc,
    FVar,
    _addr_of,
    _BufProgram,
    _bw,
    _cstr,
    te,
)

_READ_U32, _MEMCPY, _RECV = 0xa10, 0xa20, 0xa30
_NAMES = {_READ_U32: "read_u32", _MEMCPY: "memcpy", _RECV: "recv"}


class _FieldProgram:
    """`f(m, n, src)`: `t#1 = read_u32(src)`, then memory-defining stores and
    phis numbered like BN's memory SSA, then ``x#1 = <value>`` and
    ``memcpy(d, s, x#1)``. ``m``/``n`` are pointer parameters."""

    def __init__(self):
        self.m, self.n, self.src = FVar("m", ident=1), FVar("n", ident=2), FVar("src", ident=3)
        self.m0, self.n0, self.src0 = FSSA(self.m, 0), FSSA(self.n, 0), FSSA(self.src, 0)
        self.t1 = FSSA(FVar("t", ident=4), 1)
        self.instrs = [FInstr(0, 0x10, "MLIL_CALL_SSA", "t#1 = read_u32(src#0)",
                              reads=[self.src0], writes=[self.t1],
                              dest=FExpr("MLIL_CONST_PTR", hex(_READ_U32), constant=_READ_U32),
                              params=[FExpr("MLIL_VAR_SSA", "src#0", reads=[self.src0])])]
        self.mem_defs: dict = {}
        self.mem = 0

    def _at(self):
        return len(self.instrs), 0x10 + 4 * len(self.instrs)

    def addr(self, base, off):
        """`[base + off]` as BN writes it: an ADD of the pointer and a constant."""
        a = FExpr("MLIL_ADD", f"{base.var.name} + {off:#x}", reads=[base])
        a.left = FExpr("MLIL_VAR_SSA", base.var.name, reads=[base])
        a.right = FExpr("MLIL_CONST", hex(off), constant=off)
        return a

    def store(self, dest, size, value=None, *, op="MLIL_STORE_SSA", offset=None):
        i, a = self._at()
        value = value or FExpr("MLIL_VAR_SSA", "t#1", reads=[self.t1])
        ins = FInstr(i, a, op, f"[{dest}] = {value}", reads=list(dest.vars_read) + list(value.vars_read),
                     dest=dest, src=value, offset=offset)
        ins.size = size
        ins.src_memory = self.mem
        self.mem += 1
        ins.dest_memory = self.mem
        self.mem_defs[self.mem] = ins
        self.instrs.append(ins)
        return ins

    def phi(self, *versions):
        i, a = self._at()
        ins = FInstr(i, a, "MLIL_MEM_PHI", "phi")
        ins.src_memory = list(versions)
        self.mem += 1
        ins.dest_memory = self.mem
        self.mem_defs[self.mem] = ins
        self.instrs.append(ins)
        return ins

    def call(self, target, params):
        """A memory-defining call, e.g. ``recv(fd, <dest>, <len>, 0)``."""
        i, a = self._at()
        reads = [r for p in params for r in p.vars_read]
        ins = FInstr(i, a, "MLIL_CALL_SSA", f"{_NAMES[target]}(...)", reads=reads,
                     dest=FExpr("MLIL_CONST_PTR", hex(target), constant=target), params=params)
        ins.src_memory = self.mem
        self.mem += 1
        ins.dest_memory = self.mem
        self.mem_defs[self.mem] = ins
        self.instrs.append(ins)
        return ins

    def load(self, src, size, *, op="MLIL_LOAD_SSA", offset=None):
        ld = FExpr(op, f"[{src}].{size}", reads=list(src.vars_read), src=src, src_memory=self.mem)
        ld.size = size
        if offset is not None:
            ld.offset = offset
        return ld

    def sink(self, value):
        x1 = FSSA(FVar("x", ident=5), 1)
        i, a = self._at()
        self.instrs.append(FInstr(i, a, "MLIL_SET_VAR_SSA", f"x#1 = {value}",
                                  reads=list(value.vars_read), writes=[x1], src=value))
        i, a = self._at()
        self.instrs.append(FInstr(i, a, "MLIL_CALL_SSA", "memcpy(d, s, x#1)", reads=[x1],
                                  dest=FExpr("MLIL_CONST_PTR", hex(_MEMCPY), constant=_MEMCPY),
                                  params=[FExpr("MLIL_VAR_SSA", "d", reads=[]),
                                          FExpr("MLIL_VAR_SSA", "s", reads=[]),
                                          FExpr("MLIL_VAR_SSA", "x#1", reads=[x1])]))

    def run(self):
        func = FFunc("f", 0x10, FSSAFunc(self.instrs, mem_defs=self.mem_defs),
                     params=[self.m, self.n, self.src])
        result = te.TaintEngine(FBV(_NAMES), te.load_models()).backward(
            func, [te.parse_locator("arg:memcpy:2")])
        origins = [sl["origin"] for sl in result["slices"]]
        reasons = [st.get("reason", "") for sl in result["slices"] for st in sl["slice"]]
        fields = [lf for lf in result["leaves"] if lf["kind"] == "field_load_unresolved"]
        return result, origins, reasons, fields


def _sx(inner):
    return FExpr("MLIL_SX", f"sx.q({inner})", reads=list(inner.vars_read), src=inner)


@pytest.mark.parametrize("stored", [True, False])
def test_sign_extended_upper_half_load_follows_the_bytes_not_the_pointer(stored):
    # `m->lens = t` (8 bytes at +8), then `x = sx.q([m + 0xc].d)`: the length is
    # the upper half of what was stored. Walking the load's ADDRESS named the
    # object pointer -- here parameter 0, on the dogfood target the caller's
    # allocator -- as the origin of a complete slice.
    p = _FieldProgram()
    if stored:
        p.store(p.addr(p.m0, 8), 8)
    p.sink(_sx(p.load(p.addr(p.m0, 0xc), 4)))
    result, origins, reasons, fields = p.run()
    assert {"kind": "parameter", "index": 0, "var": "m#0"} not in origins
    if stored:
        assert {"kind": "call", "callee": "read_u32"} in origins, origins
        assert any("reaching store to field" in r for r in reasons), reasons
        assert not fields
    else:
        assert fields and fields[0]["offset"] == "0xc" and fields[0]["width"] == 4, fields
        assert result["diagnostics"]["safe_to_report_complete_slice"] is False


def test_field_stored_on_one_path_only_is_a_frontier():
    # `if (c) m->n = t; memcpy(.., m->n)`: the store reaches on one arm; on the
    # other the bytes are what the caller left -- not a complete slice.
    p = _FieldProgram()
    p.store(p.addr(p.m0, 8), 4)
    p.phi(0, 1)
    p.sink(p.load(p.addr(p.m0, 8), 4))
    result, origins, reasons, fields = p.run()
    assert {"kind": "call", "callee": "read_u32"} in origins or any(
        "reaching store" in r for r in reasons)
    assert fields and "stored before this function ran" in fields[0]["detail"], fields
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


@pytest.mark.parametrize("second", ["other_pointer", "same_pointer_other_field"])
def test_store_through_another_pointer_may_write_the_loaded_field(second):
    # `m->n = t; n->n = 0; memcpy(.., m->n)`: `n` may alias `m`, so the zero
    # may be what is loaded. A store to another field of `m` cannot be.
    p = _FieldProgram()
    p.store(p.addr(p.m0, 8), 4)
    base, off = (p.n0, 8) if second == "other_pointer" else (p.m0, 0x10)
    p.store(p.addr(base, off), 4, FExpr("MLIL_CONST", "0", constant=0))
    p.sink(p.load(p.addr(p.m0, 8), 4))
    result, origins, _, fields = p.run()
    assert {"kind": "call", "callee": "read_u32"} in origins or fields
    if second == "other_pointer":
        assert fields and "cannot prove disjoint" in fields[0]["detail"], fields
        assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    else:
        assert not fields
        assert result["diagnostics"]["safe_to_report_complete_slice"] is True


@pytest.mark.parametrize("store_offset, covers", [(0, False), (8, True)])
def test_struct_field_offsets_key_the_loaded_slot(store_offset, covers):
    # BN's `m->a = t` / `m->b` forms keep the field offset in `.offset` beside a
    # bare base. Keying both on the base alone made a store to one field the
    # writer of every other field.
    p = _FieldProgram()
    p.store(FExpr("MLIL_VAR_SSA", "m#0", reads=[p.m0]), 4,
            op="MLIL_STORE_STRUCT_SSA", offset=store_offset)
    p.sink(p.load(FExpr("MLIL_VAR_SSA", "m#0", reads=[p.m0]), 4,
                  op="MLIL_LOAD_STRUCT_SSA", offset=8))
    result, origins, _, fields = p.run()
    assert ({"kind": "call", "callee": "read_u32"} in origins) is covers, origins
    assert bool(fields) is not covers
    assert result["diagnostics"]["safe_to_report_complete_slice"] is covers


@pytest.mark.parametrize("dest_off, length, effect", [
    (0, 8, "disjoint"),      # recv(fd, p->hdr, 8): bytes [0,8); the field is [8,12)
    (8, 4, "cover"),
    (4, 8, "cover"),
    (10, 4, "partial"),      # [10,14) writes the field's upper half only
    (0, None, "may"),        # an unknown length may reach the field
])
def test_source_fill_writes_a_field_only_over_its_extent(dest_off, length, effect):
    # #1053 review 2: a source that fills ANOTHER part of the same object is
    # not the loaded field's producer -- comparing only the base named recv as
    # the origin of a complete slice.
    p = _FieldProgram()
    n = FVar("len", ident=9)
    size = (FExpr("MLIL_CONST", hex(length), constant=length) if length is not None
            else FExpr("MLIL_VAR_SSA", "len#0", reads=[FSSA(n, 0)]))
    dest = p.addr(p.m0, dest_off) if dest_off else FExpr("MLIL_VAR_SSA", "m#0", reads=[p.m0])
    p.call(_RECV, [FExpr("MLIL_VAR_SSA", "fd", reads=[]), dest, size,
                   FExpr("MLIL_CONST", "0", constant=0)])
    p.sink(p.load(p.addr(p.m0, 8), 4))
    result, origins, reasons, fields = p.run()
    named = {"kind": "source", "callee": "recv"} in origins or any("source recv" in r for r in reasons)
    assert named is (effect != "disjoint"), (origins, reasons)
    assert bool(fields) is (effect != "cover"), fields
    assert result["diagnostics"]["safe_to_report_complete_slice"] is (effect == "cover")


def test_disjoint_source_fill_keeps_the_earlier_field_store():
    # `p->len = t; recv(fd, p->hdr, 8); memcpy(.., p->len)`: the length is t.
    p = _FieldProgram()
    p.store(p.addr(p.m0, 8), 4)
    p.call(_RECV, [FExpr("MLIL_VAR_SSA", "fd", reads=[]),
                   FExpr("MLIL_VAR_SSA", "m#0", reads=[p.m0]),
                   FExpr("MLIL_CONST", "0x8", constant=8), FExpr("MLIL_CONST", "0", constant=0)])
    p.sink(p.load(p.addr(p.m0, 8), 4))
    result, origins, _, fields = p.run()
    assert origins == [{"kind": "call", "callee": "read_u32"}], origins
    assert not fields
    assert result["diagnostics"]["safe_to_report_complete_slice"] is True


# --- stash identity (#1034 items 1 and 2) -------------------------------------

_SLOT, _SLOT2, _OTHER = 0x404000, 0x404008, 0x405000


def _reload(slot, mem, size=None):
    ld = FExpr("MLIL_LOAD_SSA", f"[{slot:#x}]", src=_cstr(slot), src_memory=mem)
    if size is not None:
        ld.size = size
    return ld


def _memset_buf(pg, buf):
    pg.call(_B_MEMSET, [_addr_of(buf), FExpr("MLIL_CONST", "0", constant=0),
                       FExpr("MLIL_CONST", "64", constant=64)])


@pytest.mark.parametrize("narrowed", [False, True])
def test_stash_through_a_narrowed_slot_address_is_not_the_slot(narrowed):
    # #1053 review 1: `*(char **)(uint32_t)slot = buf; strcpy(*slot, ..)`
    # stores through `slot#0.edi` and reloads `[slot#0]`. Equal value widths
    # do not make the addresses equal; the strcpy need not write buf.
    slot = FVar("slot", ident=21)
    slot0 = FSSA(slot, 0)
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram(params=[slot])
    dest = (FExpr("MLIL_VAR_SSA_FIELD", "slot#0.edi", reads=[slot0]) if narrowed
            else FExpr("MLIL_VAR_SSA", "slot#0", reads=[slot0]))
    dest.size = 8
    pg.store(dest, _addr_of(buf), reads=[slot0]).size = 8
    reload = FExpr("MLIL_LOAD_SSA", "[slot#0].q", reads=[slot0],
                   src=FExpr("MLIL_VAR_SSA", "slot#0", reads=[slot0]), src_memory=pg.mem)
    reload.size = 8
    pg.call(_B_STRCPY, [reload, _cstr(0x700)])
    pg.sink(buf)
    result, reasons, leaves = _bw(pg.func())
    assert any("buf written by strcpy" in r for r in reasons) is not narrowed, reasons
    assert result["diagnostics"]["safe_to_report_complete_slice"] is not narrowed
    assert bool(leaves) is narrowed


@pytest.mark.parametrize("between, kept", [
    ("memcpy_over_slot", False),
    ("memcpy_elsewhere", True),
    ("unmodeled_call", False),
    ("store_through_pointer", False),
])
def test_remembered_stash_dies_at_every_possible_overwrite(between, kept):
    # #1053 review 3: `stash = buf; memcpy(&stash, &otherptr, 8); strcpy(stash,
    # ..)` -- the remembered slot survived the modeled overwrite and named
    # strcpy a writer (and getenv the source) of buf. Any write that may reach
    # the slot forgets it; a write provably elsewhere does not.
    buf = FVar("buf", typ="char[64]")
    ptr = FVar("ptr", ident=22)
    pg = _BufProgram(params=[ptr])
    _memset_buf(pg, buf)
    pg.store(_cstr(_SLOT), _addr_of(buf))
    if between.startswith("memcpy"):
        pg.call(_B_MEMCPY, [_cstr(_SLOT if between == "memcpy_over_slot" else _SLOT2),
                            _cstr(_OTHER), FExpr("MLIL_CONST", "8", constant=8)])
    elif between == "unmodeled_call":
        pg.call(next(k for k, v in _B_NAMES.items() if v == "run"), [])
    else:
        p0 = FSSA(ptr, 0)
        pg.store(FExpr("MLIL_VAR_SSA", "ptr#0", reads=[p0]), FExpr("MLIL_CONST", "0", constant=0),
                 reads=[p0])
    pg.call(_B_STRCPY, [_reload(_SLOT, pg.mem, 8), _cstr(0x700)])
    pg.sink(buf)
    result, reasons, leaves = _bw(pg.func())
    assert any("buf written by strcpy" in r for r in reasons) is kept, reasons
    assert any(lf.get("callee") == "strcpy" for lf in leaves) is not kept, leaves
    if kept:
        assert result["diagnostics"]["safe_to_report_complete_slice"] is True
    else:
        assert result["diagnostics"]["safe_to_report_complete_slice"] is False


@pytest.mark.parametrize("shape", ["held_copy", "overwrite", "restore"])
def test_stale_stash_never_becomes_a_writer_of_the_buffer(shape):
    # `stash` held another pointer before `&buf` was stored to it. A copy of
    # that OLD value (loaded before the escape), an overwritten slot and a
    # restored slot are not `&buf`: the strcpy through them must not be a
    # definite writer of `buf`, let alone a source of a complete slice.
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    _memset_buf(pg, buf)
    old = _reload(_SLOT, pg.mem)
    pg.store(_cstr(_SLOT), _addr_of(buf))
    if shape == "held_copy":
        pg.store(_cstr(_SLOT2), old)
        dest = _reload(_SLOT2, pg.mem)
    elif shape == "overwrite":
        pg.store(_cstr(_SLOT), _cstr(_OTHER))
        dest = _reload(_SLOT, pg.mem)
    else:
        pg.store(_cstr(_SLOT), old)
        dest = _reload(_SLOT, pg.mem)
    pg.call(_B_STRCPY, [dest, _cstr(0x700)])
    pg.sink(buf)
    result, reasons, _ = _bw(pg.func())
    assert not any("buf written by strcpy" in r for r in reasons), reasons
    assert result["diagnostics"]["safe_to_report_complete_slice"] is True
    assert result["slices"][0]["origin"] == {"kind": "call", "callee": "memset"}


@pytest.mark.parametrize("reload_size, covered", [(8, True), (4, False)])
def test_stash_reload_is_the_buffer_only_at_pointer_width(reload_size, covered):
    # `stash = buf; strcpy((char *)(uint32_t)stash, ...)`: a 4-byte reload of
    # the 8-byte stash is not the pointer -- the write may land anywhere.
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    pg.store(_cstr(_SLOT), _addr_of(buf))
    pg.call(_B_STRCPY, [_reload(_SLOT, pg.mem, reload_size), _cstr(0x700)])
    pg.sink(buf)
    result, reasons, leaves = _bw(pg.func())
    assert any("buf written by strcpy" in r for r in reasons) is covered, reasons
    assert result["diagnostics"]["safe_to_report_complete_slice"] is covered
    assert bool(leaves) is not covered


def test_zero_extended_stash_reload_is_a_may_alias_store():
    # `p = zx.q([stash].d); *p = 0`: the converted pointer may point at `buf`
    # (the stash held it) but is not `&buf`: a disclosed may-store, not a
    # store that was dropped as unrelated.
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    pg.store(_cstr(_SLOT), _addr_of(buf))
    narrow = _reload(_SLOT, pg.mem, 4)
    pg.store(FExpr("MLIL_ZX", "zx.q([stash].d)", src=narrow), FExpr("MLIL_CONST", "0", constant=0))
    pg.sink(buf)
    result, reasons, _ = _bw(pg.func())
    assert any("store that may write local buffer buf" in r for r in reasons), reasons
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


def test_discovery_cut_keeps_loop_carried_escaped_call_frontier():
    # 300 unrelated calls exhaust the memory-chain proof; the loop
    # `while (n--) { compose(); stash = buf; }` runs compose after the escape
    # on its second trip. The cut must keep naming compose, not only the
    # generic proof-cut leaf (#1034 item 3).
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    unknown = next(k for k, v in _B_NAMES.items() if v == "run")
    for _ in range(300):
        pg.call(unknown, [])
    _memset_buf(pg, buf)
    before = pg.mem
    header = pg.phi(before, before + 3)
    compose = pg.call(next(k for k, v in _B_NAMES.items() if v == "compose"), [])
    pg.store(_cstr(_SLOT), _addr_of(buf))
    sink = pg.sink(buf)
    sink.src_memory = header.dest_memory
    result, _, leaves = _bw(pg.func())
    assert result["stats"]["truncated"]
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    assert any(lf.get("callee") == "compose" and lf["address"] == hex(compose.address)
               for lf in leaves), leaves


@pytest.mark.parametrize("leaf, command", [
    ({"buffer": "buf", "function": "0x401136"},
     "`bn taint forward -f 0x401136 --source var:buf`"),
    ({"buffer": "line buf", "function": "0x401136"},
     "`bn taint forward -f 0x401136 --source 'var:line buf'`"),
    ({"buffer": "rdi#1", "function": "0x401136", "global": "0x404060"}, None),
    ({"buffer": "buf"}, None),
])
def test_backward_next_action_prints_only_a_complete_forward_command(leaf, command):
    # #1033 item 6: the aggregate advice names the function and the buffer a
    # leaf recorded, shell-quoted; with no such leaf it prints no command
    # rather than a `var:<buf>` template the shell or the parser rejects.
    from bn_agent_bridge import taint_result as tr

    diagnostics = tr.backward_diagnostics(
        [{"kind": "address_of_local_unresolved", **leaf}], [],
        sinks_seeded=1, slices=1, unresolved_origins=1)
    text = diagnostics["next_action"]
    if command is None:
        assert "`bn " not in text, text
    else:
        assert command in text, text
