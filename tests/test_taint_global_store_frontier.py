"""Forward taint: a tainted global's frontier and the evidence behind its rows.

Two defects the real-binary dogfood of the global-memory continuation found:

* P1-04 -- keying a direct store into a global turned the blocking
  ``coarse_memory_store`` it used to be into a non-blocking ``global_approx``
  assumption. The continuation only speaks at a CALL made after the write, so
  a writer with no later call that leaves the global tainted for a reader
  elsewhere answered ``safe_to_report_all_clear: true``.
* P1-05 -- the continuation's "reads it" rows were mostly loads of OTHER
  globals: an unbounded use of a preceding neighbour was cited as a proven
  read, declared object sizes were ignored so a typed scalar's extent ran
  through every following symbol, a stripped object ran to the end of its
  section, a static ``.bss`` store folded onto a synthetic header symbol and
  was never keyed, and a ``cmp [g]; jcc`` read was cited at the branch.
"""
from __future__ import annotations

import pytest
from test_taint_engine import FExpr, FInstr, FSSA, FSSAFunc, FVar, te
from test_taint_forward_global_frontier import (
    G_DIR,
    G_OTHER,
    G_TAIL,
    MEMCPY,
    STRNCPY,
    SYSTEM,
    FDataVar,
    FExtentBV,
    FGlobalBV,
    FRawView,
    FRef,
    GFunc,
    _const,
    _copy_then_call,
    _elf_with_object_sizes,
    _forward,
    _gptr,
    _interval_program,
    _leaves,
    _reader,
    _unbounded_reader,
    _var,
)

COARSE = "coarse_memory_store"


@pytest.fixture
def models():
    return te.load_models()


# --------------------------------------------------------------------------
# P1-05: an unbounded use of a preceding neighbour is a candidate, not a read
# --------------------------------------------------------------------------

def test_an_unbounded_neighbour_use_is_an_unverified_candidate(models):
    # `system(packet)` reads an unstated number of bytes from BEFORE the
    # tainted tail: not ruled out, so it still blocks -- but it is not shown
    # to reach those bytes, so it is never a proven read.
    func, bv = _interval_program(_unbounded_reader("string", 0x401200, G_DIR))
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    [row] = leaf["references"]
    assert row["access"] == "overlap_unverified", row
    assert row["offset_known"] is False, row
    assert leaf["reads_proven"] is False, leaf
    assert leaf["unverified_overlaps"] == 1, leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_the_unverified_candidate_advice_targets_the_neighbour(models):
    func, bv = _interval_program(_unbounded_reader("string", 0x401200, G_DIR))
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["reachability"]["cause"] == "object_extent_unproven", leaf
    assert f"bn data retype {hex(G_DIR)} char[SIZE]" in leaf["detail"], leaf["detail"]
    assert f"bn data retype {hex(G_TAIL)}" not in leaf["detail"], leaf["detail"]


class FDeclaredExtentBV(FExtentBV):
    """An extent view whose image or debug info DECLARES object sizes."""

    def __init__(self, *args, declared=None, source="symtab", **kw):
        super().__init__(*args, **kw)
        declared = declared or {}
        if source == "symtab":
            self.parent_view = FRawView(_elf_with_object_sizes(declared))
        else:
            sized = {addr: size for addr, size in declared.values()}
            dtype = lambda w: type("T", (), {"width": w})()  # noqa: E731
            self.debug_info = type("DI", (), {
                "get_data_variables_by_address":
                    lambda _self, a: [("DWARF", "v", dtype(sized[a]))] if a in sized else []})()


def test_a_declared_neighbour_is_separate_storage(models):
    # `packet` declares 64 bytes, so it ends exactly where `packet_tail`
    # begins: its unbounded read is of its own storage, not of the tail.
    consumer = _unbounded_reader("string", 0x401200, G_DIR)
    func, _argv = _copy_then_call(
        0x401104, MEMCPY, _gptr(G_TAIL),
        call_sites=[(0x401108, _gptr(0x401200), [])], params_extra=[_const(32)])
    bv = FDeclaredExtentBV(
        {MEMCPY: "memcpy", SYSTEM: "system", 0x401200: "string"},
        funcs={0x401200: consumer},
        refs={G_DIR: [FRef(consumer, 0x401210)], G_TAIL: [FRef(func, 0x401104)]},
        objects={G_DIR: 0, G_TAIL: 0, G_OTHER: 0}, section=(0x404000, 0x405000),
        declared={"packet": (G_DIR, 64), "packet_tail": (G_TAIL, 32)})
    result = _forward(bv, func, models)
    assert result["leaves"] == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


# --------------------------------------------------------------------------
# P1-05: a declared width ends the object, even after a span-less write
# --------------------------------------------------------------------------

def _spanless_write_program(bv_factory, reader_at):
    """`memcpy(g, argv, n)` with a runtime `n`, then a reader of *reader_at*."""
    reader = _reader("next_reader", 0x401200, reader_at, 0x401210)
    func, _argv = _copy_then_call(
        0x401104, MEMCPY, G_DIR,
        call_sites=[(0x401108, _gptr(0x401200), [])],
        params_extra=[_var(FSSA(FVar("n"), 0))])
    bv = bv_factory(
        {MEMCPY: "memcpy", SYSTEM: "system", 0x401200: "next_reader"},
        funcs={0x401200: reader},
        refs={reader_at: [FRef(reader, 0x401210)], G_DIR: [FRef(func, 0x401104)]})
    return func, bv


@pytest.mark.parametrize("source", ["symtab", "debug_info"])
def test_a_declared_scalar_ends_where_it_is_declared_to(models, source):
    # A 4-byte scalar BN typed by inference, declared 4 bytes by the image
    # (or its debug info), filled by a write of unstated length: the next
    # object's reader is not a reader of this one.
    nxt = G_DIR + 4

    def factory(names, *, funcs, refs):
        return FDeclaredExtentBV(
            names, funcs=funcs, refs=refs, objects={G_DIR: 4, nxt: 4, G_OTHER: 4},
            section=(0x404000, 0x405000), source=source,
            declared={"g": (G_DIR, 4), "g_next": (nxt, 4)})

    func, bv = _spanless_write_program(factory, nxt)
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


class FStrippedBV(FExtentBV):
    """Every data variable is anonymous: a stripped binary."""

    def get_symbol_at(self, addr):
        return None


def test_a_symbol_less_object_ends_at_the_next_referenced_address(models):
    # No symbol anywhere, so no named neighbour to stop at: the next data
    # variable BN recorded bounds the object, not the end of `.bss`.
    far = G_DIR + 0x100

    def factory(names, *, funcs, refs):
        return FStrippedBV(names, funcs=funcs, refs=refs,
                           objects={G_DIR: 4, G_DIR + 8: 0, far: 0},
                           section=(0x404000, 0x405000))

    func, bv = _spanless_write_program(factory, far)
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


# --------------------------------------------------------------------------
# P1-05: a constant-address `.bss` store on a static stripped binary is keyed
# --------------------------------------------------------------------------

HEADERS = 0x400040          # BN's synthetic `__elf_program_headers`, no section
BUILTINS = 0x405100         # `.synthetic_builtins`, past `.bss`


class FStaticBV(FExtentBV):
    """Stripped static image: the only symbols are BN's synthetic ones."""

    def get_symbol_at(self, addr):
        if int(addr) in (HEADERS, BUILTINS):
            return type("Sym", (), {"name": "synthetic", "address": int(addr)})()
        return None


def test_a_static_bss_store_is_keyed_to_its_own_object(models):
    reader = _reader("consume", 0x401200, G_DIR, 0x401210)
    n = FVar("n")
    n0 = FSSA(n, 0)
    store = FInstr(0, 0x401104, "MLIL_STORE_SSA", f"[{hex(G_DIR)}].d = n#0",
                   reads=[n0], dest=_gptr(G_DIR))
    store.size = 4
    call = FInstr(1, 0x401108, "MLIL_CALL_SSA", "consume()",
                  dest=_gptr(0x401200), params=[])
    func = GFunc("handler", 0x401100, FSSAFunc([store, call]), params=[n])
    bv = FStaticBV({SYSTEM: "system", 0x401200: "consume"},
                   funcs={0x401200: reader},
                   refs={G_DIR: [FRef(reader, 0x401210), FRef(func, 0x401104)]},
                   objects={HEADERS: 0x1c0, G_DIR: 4, BUILTINS: 8},
                   section=(0x404000, 0x405000))
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["global"]["address"] == hex(G_DIR), leaf
    assert [r["display_name"] for r in leaf["references"]] == ["consume"], leaf
    assert _leaves(result, COARSE) == [], result["leaves"]


# --------------------------------------------------------------------------
# P1-05: a read is cited where it happens
# --------------------------------------------------------------------------

def test_a_compared_load_is_cited_at_the_compare(models):
    # `cmp dword [g], 0; jne` lifts to `if ([g].d != 0)` at the BRANCH; the
    # load inside it -- and BN's code reference -- is at the compare.
    load = FExpr("MLIL_LOAD_SSA", f"[{hex(G_DIR)}].d", src=_gptr(G_DIR))
    load.address, load.size = 0x401218, 4
    cond = FExpr("MLIL_CMP_NE", f"[{hex(G_DIR)}].d != 0", left=load, right=_const(0))
    branch = FInstr(0, 0x40121f, "MLIL_IF", f"if ([{hex(G_DIR)}].d != 0)")
    branch.operands = [cond]
    checker = GFunc("checker", 0x401200, FSSAFunc([branch]), params=[])
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR, call_sites=[(0x401108, _gptr(0x401200), [])],
        params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system", 0x401200: "checker"},
                   funcs={0x401200: checker},
                   refs={G_DIR: [FRef(checker, 0x401218)]})
    [leaf] = _leaves(_forward(bv, func, models))
    [row] = leaf["references"]
    assert (row["at"], row["access"]) == (hex(0x401218), "load"), row


def test_the_declared_table_follows_the_view_s_load_base(models):
    # A position-independent image declares `st_value` link-time addresses;
    # the view mapped it at 0x400000, so the declaration moves with it.
    import struct

    image = bytearray(_elf_with_object_sizes({"g": (0x4060, 4)}))
    struct.pack_into("<Q", image, 0x20, 0x40)              # e_phoff
    struct.pack_into("<HH", image, 0x36, 56, 1)            # e_phentsize, e_phnum
    phdr = struct.pack("<IIQQQQQQ", 1, 5, 0, 0, 0, 0x1000, 0x1000, 0x1000)
    image[0x40:0x40 + len(phdr)] = phdr
    bv = FGlobalBV({})
    bv.parent_view = FRawView(bytes(image))
    bv.start = 0x400000
    eng = te.TaintEngine(bv, models)
    assert eng._authoritative_width(0x404060) == 4
    assert eng._authoritative_width(0x4060) == 0


def test_a_user_type_still_outranks_a_declaration(models):
    bv = FDeclaredExtentBV({}, objects={G_DIR: 16}, authoritative=[G_DIR],
                           section=(0x404000, 0x405000),
                           declared={"g": (G_DIR, 4)})
    assert FDataVar(G_DIR, 16, auto=False).auto_discovered is False
    eng = te.TaintEngine(bv, models)
    assert eng._global_extent(G_DIR) == 16
