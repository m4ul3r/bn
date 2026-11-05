"""A stack probe is not a may-writer of anything (W09-07).

`alloca` with a run-time size makes the compiler touch each page the frame
grows by (`or qword [rsp+rax-8], 0`), which BN lifts as a store that writes
back the bytes it loads from the same stack address. The backward walk used to
read it as a store through an unpinnable pointer that MAY write the object the
sink consumes, so every slice through the frame gained a blocking leaf and a
row of constant origins it never had."""
from __future__ import annotations

import pytest
from test_taint_engine import (
    _B_NAMES,
    _B_POPEN,
    _B_STRCPY,
    FBV,
    FExpr,
    FInstr,
    FSSA,
    FVar,
    _BufProgram,
    te,
)


def _probe_slice(value_kind: str):
    """`strcpy(dst, host); <store into the probed stack slot>; popen(dst)` where
    `dst` is a pointer parameter, so the object behind it is unresolved and
    every store the walk cannot pin is a candidate writer of it."""
    host = FVar("host", ident=11); host0 = FSSA(host, 0)
    dst = FVar("dst", ident=12, typ="char*"); dst0 = FSSA(dst, 0)
    rsp3 = FSSA(FVar("rsp"), 3)
    size2 = FSSA(FVar("size"), 2)
    pg = _BufProgram(params=[host, dst])
    pg.call(_B_STRCPY, [FExpr("MLIL_VAR_SSA", "dst#0", reads=[dst0]),
                        FExpr("MLIL_VAR_SSA", "host#0", reads=[host0])], reads=[dst0, host0])
    # The alloca size is reloaded from memory after `dst` was handed to a
    # call, so the walk cannot pin the address the store below uses.
    i, a = pg._next()
    load = FExpr("MLIL_LOAD_SSA", "[0x404060].q", src=FExpr("MLIL_CONST_PTR", "0x404060",
                                                              constant=0x404060))
    load.src_memory = pg.mem
    pg.instrs.append(FInstr(i, a, "MLIL_SET_VAR_SSA", "size#2 = [0x404060].q",
                            writes=[size2], src=load))
    base = FExpr("MLIL_ADD", "rsp#3 + size#2", reads=[rsp3, size2],
                 left=FExpr("MLIL_VAR_SSA", "rsp#3", reads=[rsp3]),
                 right=FExpr("MLIL_VAR_SSA", "size#2", reads=[size2]))
    slot = FExpr("MLIL_SUB", "rsp#3 + size#2 - 8", reads=[rsp3, size2],
                 left=base, right=FExpr("MLIL_CONST", "8", constant=8))
    if value_kind == "probe":
        value = FExpr("MLIL_LOAD_SSA", "[rsp#3 + size#2 - 8].q", reads=[rsp3, size2], src=slot)
        value.size = 8
    else:
        value = FExpr("MLIL_VAR_SSA", "host#0", reads=[host0])
    store = pg.store(slot, value, reads=[rsp3, size2] + ([host0] if value_kind != "probe" else []))
    store.size = 8
    pg.call(_B_POPEN, [FExpr("MLIL_VAR_SSA", "dst#0", reads=[dst0])], reads=[dst0])
    bv = FBV(dict(_B_NAMES))
    result = te.TaintEngine(bv, te.load_models()).backward(
        pg.func(), [te.parse_locator("arg:popen:0")])
    rows = [(st.get("address"), st.get("reason", ""))
            for sl in result["slices"] for st in sl["slice"]]
    return result, rows, hex(int(store.address))


@pytest.mark.parametrize("value_kind, may_write", [
    ("probe", False),       # writes back what it read: no object changes
    ("data", True),         # the same stack-addressed store of real data
])
def test_stack_probe_store_is_not_a_may_writer(value_kind, may_write):
    result, rows, store_addr = _probe_slice(value_kind)
    at_store = [reason for addr, reason in rows if addr == store_addr]
    assert any("may write" in r for r in at_store) is may_write, rows
    origins = {sl["origin"]["kind"] for sl in result["slices"]}
    # The probe contributes no origin of its own; strcpy's input does.
    assert "parameter" in origins, result["slices"]
