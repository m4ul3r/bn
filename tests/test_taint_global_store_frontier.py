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
    _spanning_reader,
    _unbounded_reader,
    _var,
)

COARSE = "coarse_memory_store"


@pytest.fixture
def models():
    return te.load_models()


# --------------------------------------------------------------------------
# P1-04: a tainted global the walk does not follow stays a blocking frontier
# --------------------------------------------------------------------------

def _store_fn(name, start):
    """`void name(int n) { g = n; }` -- the tainted value stored DIRECTLY into
    the global, then return: no call after the write, so no continuation can
    be asked."""
    n = FVar("n")
    n0 = FSSA(n, 0)
    store = FInstr(0, start + 4, "MLIL_STORE_SSA", f"[{hex(G_DIR)}].d = n#0",
                   reads=[n0], dest=_gptr(G_DIR))
    store.size = 4
    ret = FInstr(1, start + 8, "MLIL_RET", "return")
    return GFunc(name, start, FSSAFunc([store, ret]), params=[n])


def _fill_program(*, other_reader=True, via_helper=False, handback=False):
    """`fill(n)` stores into the global itself, or hands `n` to a helper that
    does; `use()` -- never called by the walk -- reads it."""
    use = _reader("use", 0x401200, G_DIR, 0x401210)
    writer = _store_fn("fill_helper" if via_helper else "fill",
                       0x401300 if via_helper else 0x401100)
    names = {MEMCPY: "memcpy", SYSTEM: "system", 0x401200: "use"}
    funcs = {0x401200: use}
    func = writer
    if via_helper:
        n = FVar("n")
        n0 = FSSA(n, 0)
        call = FInstr(0, 0x401104, "MLIL_CALL_SSA", "fill_helper(n#0)", reads=[n0],
                      dest=_gptr(0x401300), params=[_var(n0)])
        func = GFunc("fill", 0x401100,
                     FSSAFunc([call, FInstr(1, 0x401108, "MLIL_RET", "return")]),
                     params=[n], callees=[writer])
        names[0x401300] = "fill_helper"
        funcs[0x401300] = writer
    refs = {G_DIR: [FRef(writer, int(writer.start) + 4)]}
    if handback:
        # `char *handback(void) { return g; }`, laid out BEFORE the reader:
        # it hands the address on but reads none of its bytes.
        back = GFunc("handback", 0x401180, FSSAFunc([
            FInstr(0, 0x401184, "MLIL_RET", f"return {hex(G_DIR)}",
                   src=[_gptr(G_DIR)])]), params=[])
        names[0x401180] = "handback"
        funcs[0x401180] = back
        refs[G_DIR].append(FRef(back, 0x401184))
    if other_reader:
        refs[G_DIR].append(FRef(use, 0x401210))
    return func, FGlobalBV(names, funcs=funcs, refs=refs)


@pytest.mark.parametrize("via_helper", [False, True], ids=["seeded", "descended"])
def test_a_global_left_tainted_for_an_unwalked_reader_blocks(models, via_helper):
    func, bv = _fill_program(via_helper=via_helper)
    result = _forward(bv, func, models, source="param:0")
    assert result["reached_sinks"] == []
    [leaf] = _leaves(result, COARSE)
    # At the store itself -- inside the helper when the helper wrote it.
    assert leaf["address"] == hex(0x401304 if via_helper else 0x401104), leaf
    assert leaf["global"]["address"] == hex(G_DIR), leaf
    assert [(r["display_name"], r["at"]) for r in leaf["references"]] == \
        [("use", hex(0x401210))], leaf
    assert leaf["reads_proven"] is True, leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_a_global_no_other_function_reads_stays_clear(models):
    # The precision boundary: only the writer itself references the object,
    # so nothing past its return can read those bytes.
    func, bv = _fill_program(other_reader=False)
    result = _forward(bv, func, models, source="param:0")
    assert result["leaves"] == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


def test_a_view_that_cannot_list_references_invents_no_frontier(models):
    func, bv = _fill_program()
    bv.get_code_refs = None
    result = _forward(bv, func, models, source="param:0")
    assert _leaves(result, COARSE) == [], result["leaves"]


def test_the_store_leaf_cites_the_function_that_reads_it(models):
    # A function that only hands the address back comes first in the
    # layout; the row must name the one that READS the bytes, and the
    # other stays listed at BN's own reference to it.
    func, bv = _fill_program(handback=True)
    result = _forward(bv, func, models, source="param:0")
    [leaf] = _leaves(result, COARSE)
    assert [(r["display_name"], r["access"]) for r in leaf["references"]] == \
        [("use", "call_arg")], leaf
    assert leaf["reads_proven"] is True, leaf
    assert [(o["name"], o["at"]) for o in leaf["unexamined_reference_sites"]] == \
        [("handback", hex(0x401184))], leaf
    assert leaf["unexamined_references_total"] == 1, leaf


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


def _declared_neighbour_program(consumer):
    """`packet` (64 bytes) then `packet_tail` (32), both DECLARED by the
    image; the copy taints the tail, then *consumer* runs."""
    func, _argv = _copy_then_call(
        0x401104, MEMCPY, _gptr(G_TAIL),
        call_sites=[(0x401108, _gptr(0x401200), [])], params_extra=[_const(32)])
    return func, FDeclaredExtentBV(
        {MEMCPY: "memcpy", SYSTEM: "system", 0x401200: consumer.name},
        funcs={0x401200: consumer},
        refs={G_DIR: [FRef(consumer, 0x401210)], G_TAIL: [FRef(func, 0x401104)]},
        objects={G_DIR: 0, G_TAIL: 0, G_OTHER: 0}, section=(0x404000, 0x405000),
        declared={"packet": (G_DIR, 64), "packet_tail": (G_TAIL, 32)})


def test_a_declared_neighbour_is_separate_storage(models):
    # `packet` declares 64 bytes, so it ends exactly where `packet_tail`
    # begins: its unbounded read is of its own storage, not of the tail.
    func, bv = _declared_neighbour_program(
        _unbounded_reader("string", 0x401200, G_DIR))
    result = _forward(bv, func, models)
    assert result["leaves"] == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


def test_a_stated_read_past_a_declared_end_still_reads_the_tail(models):
    # ...but `memcpy(out, packet, 112)` STATES that it reads 48 bytes past
    # that end, into the tainted tail: the declaration separates storage,
    # it does not hide an over-read of it.
    func, bv = _declared_neighbour_program(
        _spanning_reader("over_read", 0x401200, G_DIR, 112))
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert [(r["display_name"], r["access"]) for r in leaf["references"]] == \
        [("over_read", "call_arg")], leaf
    assert leaf["reads_proven"] is True, leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


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


# --------------------------------------------------------------------------
# V3-01: the seeded function's own read of a static, before its store
# --------------------------------------------------------------------------

class _Block:
    """One IL basic block, as `_reaches_avoiding` reads it."""

    def __init__(self, start, end):
        self.start, self.end = start, end
        self.outgoing_edges = []


def _static_index_fn(*, read_first):
    """`int next_tok(int n) { x = n; c = g; g = n; return c; }`: the load of
    the static `g` either precedes the tainting store (its NEXT activation
    reads this call's input) or follows it (it reads this call's own value,
    which the walk followed)."""
    n, c, x = FVar("n"), FVar("c"), FVar("x")
    n0, c1, x1 = FSSA(n, 0), FSSA(c, 1), FSSA(x, 1)
    start = 0x401100
    load = FExpr("MLIL_LOAD_SSA", f"[{hex(G_DIR)}].d", src=_gptr(G_DIR))
    load.size = 4
    copy = FInstr(0, start + 4, "MLIL_SET_VAR_SSA", "x#1 = n#0", reads=[n0], writes=[x1],
                  src=_var(n0))
    read = FInstr(0, 0, "MLIL_SET_VAR_SSA", f"c#1 = [{hex(G_DIR)}].d", writes=[c1], src=load)
    store = FInstr(0, 0, "MLIL_STORE_SSA", f"[{hex(G_DIR)}].d = x#1",
                   reads=[x1], dest=_gptr(G_DIR), src=_var(x1))
    store.size = 4
    body = [copy, read, store] if read_first else [copy, store, read]
    ret = FInstr(3, start + 16, "MLIL_RET", "return c#1", reads=[c1], src=[_var(c1)])
    instrs = [*body, ret]
    block = _Block(0, len(instrs))
    for i, ins in enumerate(instrs):
        ins.instr_index = i
        ins.address = start + 4 * (i + 1)
        ins.il_basic_block = block
    func = GFunc("next_tok", start, FSSAFunc(instrs), params=[n])
    refs = {G_DIR: [FRef(func, int(read.address)), FRef(func, int(store.address))]}
    return func, read, store, FGlobalBV({}, funcs={start: func}, refs=refs)


def test_a_static_the_function_reads_before_storing_blocks(models):
    func, read, store, bv = _static_index_fn(read_first=True)
    result = _forward(bv, func, models, source="param:0")
    [leaf] = _leaves(result, COARSE)
    assert leaf["address"] == hex(int(store.address)), leaf
    assert leaf["self_reread"] is True, leaf
    assert [(r["name"], r["at"], r["access"], r["offset"], r["offset_known"])
            for r in leaf["references"]] == [
        ("next_tok", hex(int(read.address)), "load", 0, True)], leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_a_static_read_only_after_its_store_stays_clear(models):
    # The precision boundary: every read follows the tainting store, so it
    # reads this activation's own value -- which the walk followed.
    func, _read, _store, bv = _static_index_fn(read_first=False)
    result = _forward(bv, func, models, source="param:0")
    assert _leaves(result, COARSE) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


# --------------------------------------------------------------------------
# V3-02: a code reference no lifted instruction names is no reader
# --------------------------------------------------------------------------

def test_a_reference_the_lifted_body_never_names_is_not_a_reader(models):
    # On ARM a PC-relative base (`add r3, pc`) lands inside the tainted
    # static while the load it feeds reads another object; BN records the
    # base as a code reference and the MLIL folds it away. That reference
    # was cited as an `unclassified` read at offset 0 of the static.
    func, bv = _fill_program(other_reader=False)
    flag = FExpr("MLIL_LOAD_SSA", f"[{hex(G_OTHER)}].d", src=_gptr(G_OTHER))
    v1 = FSSA(FVar("v"), 1)
    pic_user = GFunc("log_flagged", 0x401200, FSSAFunc([
        FInstr(0, 0x401212, "MLIL_SET_VAR_SSA", f"v#1 = [{hex(G_OTHER)}].d",
               writes=[v1], src=flag),
        FInstr(1, 0x401216, "MLIL_RET", "return v#1", reads=[v1], src=[_var(v1)]),
    ]), params=[])
    bv._names[0x401200] = "log_flagged"
    bv._funcs[0x401200] = pic_user
    bv._refs[G_DIR].append(FRef(pic_user, 0x401210))
    result = _forward(bv, func, models, source="param:0")
    assert _leaves(result, COARSE) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True
