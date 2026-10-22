"""Backward taint through memory loads and stashed pointers (#1041 item 2,
#1034 items 1-3): a loaded value comes from the writers of the loaded bytes on
every path, never from the pointer it was loaded through, and a stashed
pointer is the buffer only when one store definitely put it there."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_taint_engine import (
    _B_MEMCPY,
    _B_MEMSET,
    _B_NAMES,
    _B_READ,
    _B_SNPRINTF,
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
    _store_proof_only,
    te,
)

_READ_U32, _MEMCPY, _RECV, _STRTOK, _STAT, _STRNCAT, _STRNCPY, _GETENV = (
    0xa10, 0xa20, 0xa30, 0xa40, 0xa50, 0xa60, 0xa70, 0xa80)
_NAMES = {_READ_U32: "read_u32", _MEMCPY: "memcpy", _RECV: "recv", _STRTOK: "strtok",
          _STAT: "stat", _STRNCAT: "strncat", _STRNCPY: "strncpy", _GETENV: "getenv"}


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

    def run(self, *, writable=None, segment_writable=None):
        """*writable*: ``bv.is_offset_writable``'s answer. *segment_writable*:
        a BN segment over 0x402000..0x403000 with that write flag (None: no
        segment information)."""
        func = FFunc("f", 0x10, FSSAFunc(self.instrs, mem_defs=self.mem_defs),
                     params=[self.m, self.n, self.src])
        bv = FBV(_NAMES)
        if writable is not None:
            bv.is_offset_writable = lambda addr: writable
        if segment_writable is not None:
            seg = SimpleNamespace(start=0x402000, end=0x403000, writable=segment_writable)
            bv.get_segment_at = lambda addr: seg if seg.start <= addr < seg.end else None
        result = te.TaintEngine(bv, te.load_models()).backward(
            func, [te.parse_locator("arg:memcpy:2")])
        origins = [sl["origin"] for sl in result["slices"]]
        reasons = [st.get("reason", "") for sl in result["slices"] for st in sl["slice"]]
        fields = [lf for lf in result["leaves"] if lf["kind"] == "field_load_unresolved"]
        return result, origins, reasons, fields


@pytest.mark.parametrize("writable", [False, True])
def test_constant_table_at_a_scaled_index_is_a_frontier_with_the_index(writable):
    # `malloc(readonly_sizes[i & 3])` lifts to `rdx#1 = n#0 << 3;
    # [rdx#1 + 0x402020].q`, which keys as an exact slot on the scaled index.
    # The index picks the bytes and stays in the slice (#1053 review 4). The
    # bytes are never the binary's own: even a bounded load wholly inside a
    # non-writable segment may read entries the loader wrote (an IFUNC symbol
    # relocation under TEXTREL, an IRELATIVE entry) -- #1053 review 6 removed
    # that claim. A read-only table is a `field_load_unresolved` frontier; a
    # writable one a `global` frontier.
    p = _FieldProgram()
    rdx1 = FSSA(FVar("rdx", ident=6), 1)
    scaled = FExpr("MLIL_LSL", "n#0 << 3", reads=[p.n0])
    scaled.left = FExpr("MLIL_VAR_SSA", "n#0", reads=[p.n0])
    scaled.right = FExpr("MLIL_CONST", "3", constant=3)
    i, a = p._at()
    p.instrs.append(FInstr(i, a, "MLIL_SET_VAR_SSA", "rdx#1 = n#0 << 3", reads=[p.n0],
                           writes=[rdx1], src=scaled))
    addr = FExpr("MLIL_ADD", "rdx#1 + 0x402020", reads=[rdx1])
    addr.left, addr.right = FExpr("MLIL_VAR_SSA", "rdx#1", reads=[rdx1]), _cstr(0x402020)
    # BN bounds the address to the four 8-byte entries.
    addr.possible_values = SimpleNamespace(
        type=SimpleNamespace(name="UnsignedRangeValue"),
        ranges=[SimpleNamespace(start=0x402020, end=0x402038, step=8)])
    p.sink(p.load(addr, 8))
    result, origins, _, fields = p.run(writable=writable, segment_writable=writable)
    assert any((o["kind"], o.get("index"), o.get("var")) == ("parameter", 1, "n#0") for o in origins), origins
    if writable:
        assert not fields, fields
        assert any(lf.get("global") == "0x402020" for lf in result["leaves"]), result["leaves"]
    else:
        assert fields and "constant table" in fields[0]["detail"], result["leaves"]
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


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
    assert not any(o["kind"] == "parameter" and o.get("index") == 0 and o.get("var") == "m#0"
                   for o in origins)
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


@pytest.mark.parametrize("between", [
    "recv_disjoint",          # recv(fd, p->hdr, 8): payload [0,8), the field is [8,12)
    "recv_covering",          # recv(fd, p, 12)
    "recv_zero_count",        # a zero-count read still sets errno, which p may be (round 3)
    "recv_pointer_unrecovered",  # BN recovered `recv(fd)` only (round 2)
    "getenv",                 # ISO C allows getenv to copy into an internal buffer (round 3)
    "strtok",                 # rewrites the delimiter; its model names no output
    "stat",                   # sink-only model, yet it fills *arg:1
    "strncat",                # appends at dst + strlen(dst)
    "strncpy",                # writes [dst, dst+3), but no model proves its complete effect
])
def test_no_call_is_proof_that_a_field_was_not_written(between):
    # #1053 review rounds 1-3: `p->len = t; <call>; memcpy(.., p->len)`. No
    # model enumerates every memory write its callee performs, so every call
    # may have written the field: the earlier store is walked (t stays a
    # candidate), a modeled source is named, and the slice is not complete.
    p = _FieldProgram()
    p.store(p.addr(p.m0, 8), 4)
    m = FExpr("MLIL_VAR_SSA", "m#0", reads=[p.m0])
    src = FExpr("MLIL_VAR_SSA", "src#0", reads=[p.src0])
    fd = FExpr("MLIL_VAR_SSA", "fd", reads=[])

    def const(n):
        return FExpr("MLIL_CONST", hex(n), constant=n)

    target, params = {
        "recv_disjoint": (_RECV, [fd, m, const(8), const(0)]),
        "recv_covering": (_RECV, [fd, m, const(12), const(0)]),
        "recv_zero_count": (_RECV, [fd, p.addr(p.m0, 8), const(0), const(0)]),
        "recv_pointer_unrecovered": (_RECV, [fd]),
        "getenv": (_GETENV, [_cstr(0x700)]),
        "strtok": (_STRTOK, [m, _cstr(0x700)]),
        "stat": (_STAT, [_cstr(0x700), m]),
        "strncat": (_STRNCAT, [m, src, const(3)]),
        "strncpy": (_STRNCPY, [m, src, const(3)]),
    }[between]
    p.call(target, params)
    p.sink(p.load(p.addr(p.m0, 8), 4))
    result, _, reasons, fields = p.run()
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result["diagnostics"]
    assert fields, result["leaves"]
    if target == _RECV:
        assert any("field may be filled by source recv" in r for r in reasons), reasons
        assert any("reaching store to field" in r for r in reasons), reasons


# --- stash identity (#1034 items 1 and 2) -------------------------------------

_SLOT, _SLOT2, _OTHER = 0x404000, 0x404008, 0x405000


def _reload(slot, mem, size=None):
    ld = FExpr("MLIL_LOAD_SSA", f"[{slot:#x}]", src=_cstr(slot), src_memory=mem)
    if size is not None:
        ld.size = size
    return ld


def _init_buf(pg, buf):
    """Initialize the whole buffer by a STORE: a call handed &buf would itself
    start an escape epoch (#1053 review 4) and change what these tests probe."""
    pg.whole_store(buf)


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
    # A definite writer is a named fill; no call's extent proves the bytes
    # (#931), so the slice is incomplete either way -- but only the narrowed
    # stash leaves strcpy an opaque possible writer.
    assert any(lf.get("callee") == "strcpy" for lf in leaves) is narrowed, leaves
    assert any(lf.get("calls") == ["strcpy"] for lf in leaves) is not narrowed, leaves
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


@pytest.mark.parametrize("between, kept", [
    ("memcpy_over_slot", False),
    ("memcpy_elsewhere", False),           # no model proves a call's complete effect
    ("memset_through_proven_buf", False),  # round 2's framed pass, reverted in round 3
    ("read_zero_count", False),            # a failing read still writes errno (round 3)
    ("read_pointer_unrecovered", False),
    ("snprintf_elsewhere", False),
    ("unmodeled_call", False),
    ("store_through_pointer", False),
    ("store_elsewhere", True),             # a STORE to another exact address
    ("store_through_proven_buf", True),    # a STORE through a reload proven to be &buf
])
def test_remembered_stash_dies_at_every_possible_overwrite(between, kept):
    # #1053 review rounds 1-3: `stash = buf; memcpy(&stash, &otherptr, 8);
    # strcpy(stash, ..)` -- the remembered slot survived the overwrite and
    # named strcpy a writer (and getenv the source) of buf. No model proves
    # what a call does not write, so every call forgets the slot; a store is
    # judged by its exact address and width.
    buf = FVar("buf", typ="char[64]")
    ptr = FVar("ptr", ident=22)
    pg = _BufProgram(params=[ptr])
    _init_buf(pg, buf)
    pg.store(_cstr(_SLOT), _addr_of(buf))
    zero, eight = FExpr("MLIL_CONST", "0", constant=0), FExpr("MLIL_CONST", "8", constant=8)
    if between.startswith("memcpy"):
        pg.call(_B_MEMCPY, [_cstr(_SLOT if between == "memcpy_over_slot" else _SLOT2),
                            _cstr(_OTHER), eight])
    elif between == "memset_through_proven_buf":
        pg.call(_B_MEMSET, [_reload(_SLOT, pg.mem, 8), zero,
                            FExpr("MLIL_CONST", "64", constant=64)])
    elif between == "read_zero_count":
        pg.call(_B_READ, [zero, _cstr(_SLOT2), zero])
    elif between == "read_pointer_unrecovered":
        pg.call(_B_READ, [zero])
    elif between == "snprintf_elsewhere":
        pg.call(_B_SNPRINTF, [_cstr(_SLOT2), eight, _cstr(0x700)])
    elif between == "unmodeled_call":
        pg.call(next(k for k, v in _B_NAMES.items() if v == "run"), [])
    elif between == "store_through_pointer":
        p0 = FSSA(ptr, 0)
        pg.store(FExpr("MLIL_VAR_SSA", "ptr#0", reads=[p0]), zero, reads=[p0])
    elif between == "store_elsewhere":
        pg.store(_cstr(_SLOT2), zero)
    else:
        pg.store(_reload(_SLOT, pg.mem, 8), zero).size = 1
    pg.call(_B_STRCPY, [_reload(_SLOT, pg.mem, 8), _cstr(0x700)])
    pg.sink(buf)
    result, reasons, leaves = _bw(pg.func())
    assert any("buf written by strcpy" in r for r in reasons) is kept, reasons
    assert any(lf.get("callee") == "strcpy" for lf in leaves) is not kept, leaves
    assert _store_proof_only(result) is kept, result["leaves"]
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


@pytest.mark.parametrize("shape", ["held_copy", "overwrite", "restore"])
def test_stale_stash_never_becomes_a_writer_of_the_buffer(shape):
    # `stash` held another pointer before `&buf` was stored to it. A copy of
    # that OLD value (loaded before the escape), an overwritten slot and a
    # restored slot are not `&buf`: the strcpy through them must not be a
    # definite writer of `buf`, let alone a source of a complete slice.
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    _init_buf(pg, buf)
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
    assert _store_proof_only(result), result["leaves"]
    assert result["slices"][0]["origin"]["kind"] == "constant", result["slices"][0]["origin"]


@pytest.mark.parametrize("explicit_escape", [False, True])
def test_call_handed_the_buffer_may_plant_a_pointer_to_it(explicit_escape):
    # #1053 review 4: `strtok_r(buf, ",", &g_save)` saves buf + 2; `p = g_save;
    # [stash = buf;] strcpy(p, host); popen(buf)`. A call handed &buf may
    # retain or plant it, so a later reload is a may-alias of buf and the
    # strcpy through it a frontier -- never an unrelated pointer.
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    pg.call(_B_STRCPY, [_addr_of(buf), _cstr(0x700)])
    planted = _reload(_SLOT2, pg.mem, 8)
    if explicit_escape:
        pg.store(_cstr(_SLOT), _addr_of(buf))
    pg.call(_B_STRCPY, [planted, _cstr(0x704)])
    pg.sink(buf)
    result, reasons, leaves = _bw(pg.func())
    assert any(lf.get("callee") == "strcpy" for lf in leaves), (leaves, reasons)
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


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
    assert any(lf.get("callee") == "strcpy" for lf in leaves) is not covered, leaves
    assert any(lf.get("calls") == ["strcpy"] for lf in leaves) is covered, leaves
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


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
    _init_buf(pg, buf)
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
