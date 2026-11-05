"""Backward origin through a reused stack slot (W06-08): a field store that
overwrites the whole slot ends the slot's def-chain, so the previous version
-- and the slot's entry version behind it -- is no input of the reload."""
from __future__ import annotations

import pytest
from test_taint_engine import (
    _B_NAMES,
    FSSA,
    FExpr,
    FInstr,
    FVar,
    _BufProgram,
    _bw,
)

_B_MALLOC, _B_READ_ROWS = 0x9F0, 0x9F1
_B_NAMES.update({_B_MALLOC: "malloc", _B_READ_ROWS: "read_rows"})    # as test_taint_engine does


class _Width:
    """A BN ``Type`` as far as the walk reads it: its byte width."""

    def __init__(self, width):
        self.width = width

    def __str__(self):
        return f"int{self.width * 8}_t"


def _reused_slot_program(slot_width, store_size, store_offset):
    """`alloc_rows()`: `n#1 = read_rows(); slot#1:<off>.<size> = n#1 @ slot#0;
    x#1 = slot#1:0.q; malloc(x#1)` -- the spill slot is (re)defined in the
    function before the reload; its entry version slot#0 is what was there
    before the function ran."""
    n1 = FSSA(FVar("n"), 1)
    slot = FVar("rows_1", ident=50, typ=_Width(slot_width))
    slot0, slot1 = FSSA(slot, 0), FSSA(slot, 1)
    x1 = FSSA(FVar("x0_4"), 1)
    pg = _BufProgram(name="alloc_rows")
    pg.call(_B_READ_ROWS, [], writes=[n1])
    src = FExpr("MLIL_VAR_SSA", "n#1", reads=[n1])
    src.size = store_size
    i, a = pg._next()
    pg.instrs.append(FInstr(i, a, "MLIL_SET_VAR_SSA_FIELD",
                            f"rows_1#1:{store_offset}.{store_size} = n#1 @ rows_1#0",
                            writes=[slot1], reads=[slot0, n1], dest=slot1, src=src,
                            prev=slot0, offset=store_offset))
    i, a = pg._next()
    pg.instrs.append(FInstr(i, a, "MLIL_SET_VAR_SSA", "x0_4#1 = rows_1#1:0.q", writes=[x1],
                            reads=[slot1],
                            src=FExpr("MLIL_VAR_SSA_FIELD", "rows_1#1:0.q", reads=[slot1])))
    pg.call(_B_MALLOC, [FExpr("MLIL_VAR_SSA", "x0_4#1", reads=[x1])], reads=[x1])
    return pg.func()


def _origins(result):
    return [(o["kind"], o.get("callee"), o.get("var"))
            for sl in result["slices"] for o in (sl.get("origins") or [sl["origin"]])]


@pytest.mark.parametrize("slot_width, store_size", [(4, 8), (8, 8)])
def test_whole_slot_store_has_no_entry_origin_w06_08(slot_width, store_size):
    # A 4-byte local BN typed on an 8-byte spill slot, rewritten in full
    # before the reload: the reload reads only the store, so the origin is
    # the call whose return it stored -- never the slot's entry value.
    result, _, _ = _bw(_reused_slot_program(slot_width, store_size, 0), sink="arg:malloc:0")
    assert [sl["sink"]["address"] for sl in result["slices"]] == ["0x1c"], result["slices"]
    assert _origins(result) == [("call", "read_rows", None)], result["slices"]
    assert [st["address"] for st in result["slices"][0]["slice"]] == [
        "0x10", "0x14", "0x18"], result["slices"]


@pytest.mark.parametrize("slot_width, store_size, store_offset", [
    (8, 4, 0),       # a narrower store keeps the slot's upper bytes
    (8, 4, 4),       # a store past offset 0 keeps the lower bytes
])
def test_partial_slot_store_keeps_the_previous_version_w06_08(slot_width, store_size,
                                                              store_offset):
    # Bytes the store does not cover still come from the previous version:
    # its entry value stays an origin beside the stored call return.
    result, _, _ = _bw(_reused_slot_program(slot_width, store_size, store_offset),
                       sink="arg:malloc:0")
    assert sorted(_origins(result)) == [("call", "read_rows", None),
                                        ("entry", None, "rows_1#0")], result["slices"]
