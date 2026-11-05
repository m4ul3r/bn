"""Forward taint: a tainted global's continuation across several calls, its
pointer aliases, and the references its exit frontier lists.

* R1-01 -- one global crossing two calls to an unbounded reader of its
  neighbour made two continuation leaves carrying ``unverified_overlaps``,
  and coalescing them iterated that count, so the whole forward op failed;
  the merged ``written_offsets`` were also sorted as text.
* R1-03 -- pointer-slot alias discovery kept the first 64 data references,
  so a reader of the 65th slot was never swept and the run cleared.
* R1-04 -- once the exit frontier had cited a reader, every further function
  that named only a declared preceding object was listed as referencing the
  tainted one.
"""
from __future__ import annotations

import pytest
from test_taint_claim_gaps import G, G_SIZE, FAliasBV, _all_clear, _copy_then_calls, _slot_reader
from test_taint_engine import FInstr, FSSA, FSSAFunc, FVar, te
from test_taint_forward_global_frontier import (
    G_DIR,
    G_OTHER,
    G_TAIL,
    MEMCPY,
    SYSTEM,
    FExtentBV,
    FRef,
    GFunc,
    _const,
    _copy_then_call,
    _forward,
    _gptr,
    _leaves,
    _reader,
    _unbounded_reader,
    _var,
)
from test_taint_global_store_frontier import FDeclaredExtentBV


@pytest.fixture
def models():
    return te.load_models()


# --------------------------------------------------------------------------
# R1-01: several call sites, each with an unverified overlap, coalesce
# --------------------------------------------------------------------------

def _two_crossings_of_a_neighbour_reader():
    """Taint the tail symbol, then call `string()` -- an unbounded reader of
    the PRECEDING symbol -- twice."""
    consumer = _unbounded_reader("string", 0x401200, G_DIR)
    func, _argv = _copy_then_call(
        0x401104, MEMCPY, _gptr(G_TAIL),
        call_sites=[(0x401108, _gptr(0x401200), []), (0x40110c, _gptr(0x401200), [])],
        params_extra=[_const(32)])
    bv = FExtentBV({MEMCPY: "memcpy", SYSTEM: "system", 0x401200: "string"},
                   funcs={0x401200: consumer},
                   refs={G_DIR: [FRef(consumer, 0x401210)],
                         G_TAIL: [FRef(func, 0x401104)]},
                   objects={G_DIR: 0, G_TAIL: 0, G_OTHER: 0},
                   section=(0x404000, 0x405000))
    return func, bv


def test_two_sites_with_unverified_overlaps_coalesce_into_one_leaf(models):
    func, bv = _two_crossings_of_a_neighbour_reader()
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert [s["address"] for s in leaf["sites"]] == [hex(0x401108), hex(0x40110c)], leaf
    assert [(r["display_name"], r["access"]) for r in leaf["references"]] == \
        [("string", "overlap_unverified")], leaf
    # The count is of the MERGED rows: one candidate, seen from two calls.
    assert leaf["unverified_overlaps"] == 1, leaf
    assert leaf["reads_proven"] is False, leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_coalesced_written_offsets_stay_in_numeric_order(models):
    # `memcpy(g + 8, s, 8); reader(); memcpy(g + 0x10, s, 8); reader();` --
    # each site's offsets are numbers rendered as hex; merged, they must
    # still sort as numbers, not as text ("0x10" before "0x8").
    reader = _reader("reader", 0x401200, G_DIR, 0x401210)
    argv0 = FSSA(FVar("argv"), 0)

    def copy(i, at, dest):
        return FInstr(i, at, "MLIL_CALL_SSA", f"memcpy({hex(dest)}, argv#0, 8)",
                      reads=[argv0], dest=_gptr(MEMCPY),
                      params=[_gptr(dest), _var(argv0), _const(8)])

    def call(i, at):
        return FInstr(i, at, "MLIL_CALL_SSA", "reader()", dest=_gptr(0x401200), params=[])

    func = GFunc("main", 0x401100, FSSAFunc([
        copy(0, 0x401104, G_DIR + 8), call(1, 0x401108),
        copy(2, 0x40110c, G_DIR + 0x10), call(3, 0x401110)]), params=[argv0.var])
    bv = FExtentBV({MEMCPY: "memcpy", SYSTEM: "system", 0x401200: "reader"},
                   funcs={0x401200: reader},
                   refs={G_DIR: [FRef(reader, 0x401210)]},
                   objects={G_DIR: 0x80, G_OTHER: 0x80},
                   interior=[G_DIR + 8, G_DIR + 0x10],
                   section=(0x404000, 0x405000))
    [leaf] = _leaves(_forward(bv, func, models))
    assert [s["address"] for s in leaf["sites"]] == [hex(0x401108), hex(0x401110)], leaf
    assert leaf["global"]["written_offsets"] == ["0x8", "0x10"], leaf


# --------------------------------------------------------------------------
# R1-03: every pointer slot initialised to the object is an alias
# --------------------------------------------------------------------------

SLOTS = 0x404200        # `char *P[65]`-worth of separate pointer globals


def test_the_only_reader_behind_the_sixty_fifth_pointer_slot_blocks(models):
    # 65 writable pointer globals, each initialised to G; only the reader of
    # the LAST one runs. Its body names that slot, never G.
    slots = [SLOTS + 8 * k for k in range(65)]
    reader = _slot_reader("via_last_slot", 0x401200, slots[-1], 0x401210)
    func = _copy_then_calls(reader)
    bv = FAliasBV({MEMCPY: "memcpy", SYSTEM: "system", reader.start: reader.name},
                  funcs={reader.start: reader},
                  refs={slots[-1]: [FRef(reader, 0x40120c)]},
                  objects={G: G_SIZE, **{s: 8 for s in slots}},
                  pointers={s: G for s in slots})
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["global"]["address"] == hex(G), leaf
    assert [(r["display_name"], r["at"]) for r in leaf["references"]] == \
        [("via_last_slot", hex(0x401210))], leaf
    assert _all_clear(result) is False


# --------------------------------------------------------------------------
# R1-04: the exit frontier lists only functions that reference the object
# --------------------------------------------------------------------------

def test_a_declared_neighbour_reader_is_not_listed_beside_the_cited_reader(models):
    # `packet` (64 bytes) and `packet_tail` (32) are both declared; `fill(n)`
    # stores into the tail and returns. `reader` loads the tail; `scan`
    # hands `packet` to `system` -- unbounded, but of storage the
    # declaration ends exactly where the tail begins. Only `reader` refers
    # to the tainted object; `scan` must not be listed as referencing it.
    n0 = FSSA(FVar("n"), 0)
    store = FInstr(0, 0x401104, "MLIL_STORE_SSA", f"[{hex(G_TAIL)}].d = n#0",
                   reads=[n0], dest=_gptr(G_TAIL))
    store.size = 4
    func = GFunc("fill", 0x401100, FSSAFunc(
        [store, FInstr(1, 0x401108, "MLIL_RET", "return")]), params=[n0.var])
    reader = _reader("reader", 0x401200, G_TAIL, 0x401210)
    scan = _unbounded_reader("scan", 0x401300, G_DIR, at=0x401310)
    bv = FDeclaredExtentBV(
        {MEMCPY: "memcpy", SYSTEM: "system", 0x401200: "reader", 0x401300: "scan"},
        funcs={0x401200: reader, 0x401300: scan},
        refs={G_TAIL: [FRef(func, 0x401104), FRef(reader, 0x401210)],
              G_DIR: [FRef(scan, 0x401310)]},
        objects={G_DIR: 0, G_TAIL: 0, G_OTHER: 0}, section=(0x404000, 0x405000),
        declared={"packet": (G_DIR, 64), "packet_tail": (G_TAIL, 32)})
    result = _forward(bv, func, models)
    [leaf] = _leaves(result, "coarse_memory_store")
    assert leaf["global"]["address"] == hex(G_TAIL), leaf
    assert [(r["display_name"], r["at"]) for r in leaf["references"]] == \
        [("reader", hex(0x401210))], leaf
    assert leaf["unexamined_reference_sites"] == [], leaf
    assert leaf["unexamined_references_total"] == 0, leaf
    assert _all_clear(result) is False


def test_a_second_reader_of_the_object_itself_is_still_listed(models):
    # The control: a function that DOES reference the tainted object is
    # listed, at BN's reference, beside the reader cited.
    n0 = FSSA(FVar("n"), 0)
    store = FInstr(0, 0x401104, "MLIL_STORE_SSA", f"[{hex(G_TAIL)}].d = n#0",
                   reads=[n0], dest=_gptr(G_TAIL))
    store.size = 4
    func = GFunc("fill", 0x401100, FSSAFunc(
        [store, FInstr(1, 0x401108, "MLIL_RET", "return")]), params=[n0.var])
    reader = _reader("reader", 0x401200, G_TAIL, 0x401210)
    again = _reader("again", 0x401300, G_TAIL, 0x401310)
    bv = FDeclaredExtentBV(
        {MEMCPY: "memcpy", SYSTEM: "system", 0x401200: "reader", 0x401300: "again"},
        funcs={0x401200: reader, 0x401300: again},
        refs={G_TAIL: [FRef(func, 0x401104), FRef(reader, 0x401210),
                       FRef(again, 0x401310)]},
        objects={G_DIR: 0, G_TAIL: 0, G_OTHER: 0}, section=(0x404000, 0x405000),
        declared={"packet": (G_DIR, 64), "packet_tail": (G_TAIL, 32)})
    [leaf] = _leaves(_forward(bv, func, models), "coarse_memory_store")
    assert [(o["name"], o["at"]) for o in leaf["unexamined_reference_sites"]] == \
        [("again", hex(0x401310))], leaf
    assert leaf["unexamined_references_total"] == 1, leaf
