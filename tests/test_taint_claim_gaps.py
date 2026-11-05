"""Forward-taint claim gaps: shapes where an empty forward result used to be
read as an all-clear (or a finding was lost) although the walk had NOT ruled
the flow out.

Each test pins the consumer-visible contract -- leaf ``kind``/fields, sink
classes, ``diagnostics.safe_to_report_all_clear`` -- of one fix:

* P2-10: a taint-armed opt-in receive sink, an unread ``param:`` seed and an
  ``arg:`` seed on an unmodeled callee are blocking frontiers, not silence.
* P2-11: a tainted global live at the seeded function's exit is a frontier
  unless a constant fill kills it; a pointer global initialised to the object
  reads it; a callee filling the caller's global through its pointer
  parameter taints that global; a heap block kept in a global slot carries
  its taint through the reload.
* P2-08: a struct field store taints that field only; a load through a
  member pointing into the same struct reads the field it lands on, or is a
  ``field_load_unresolved`` frontier when that place is unknown (V1-01).
* P2-04: one continuation leaf per tainted global, every call in ``sites``.
* R3 sweep economy: bodies that cannot see the global are not scanned, and
  bodies BN's call edges show no reader beyond are not lifted (V2-02).

All names and addresses are invented.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_taint_engine import FExpr, FInstr, FSSA, FSSAFunc, FVar, te
from test_taint_forward_global_frontier import (
    MEMCPY,
    MEMSET,
    STRNCPY,
    SYSTEM,
    FBlock,
    FExtentBV,
    FGlobalBV,
    FRef,
    GFunc,
    _const,
    _fanout,
    _forward,
    _gptr,
    _leaves,
    _reader,
    _var,
)

from bn.taint_vocabulary import BLOCKING_LEAF_KINDS

READ = 0x4010b0
HELPER = 0x401300
G = 0x404100            # a writable global object
G_SIZE = 0x40


@pytest.fixture
def models():
    return te.load_models()


def _all_clear(result):
    return result["diagnostics"]["safe_to_report_all_clear"]


def _kinds(result):
    return [lf["kind"] for lf in result["leaves"]]


def _in_one_block(func):
    """Give every instruction of *func* one IL basic block, so path questions
    (`is the write killed before the return`) have a CFG to answer from."""
    instrs = func.mlil.ssa_form.instructions
    block = FBlock(0, len(instrs))
    for i, ins in enumerate(instrs):
        ins.instr_index = i
        ins.il_basic_block = block
    func.mlil.ssa_form.basic_blocks = [block]
    return func


def _ret(index, addr):
    return FInstr(index, addr, "MLIL_RET", "return")


# --------------------------------------------------------------------------
# P2-10: a taint-armed opt-in receive sink is disclosed, not dropped
# --------------------------------------------------------------------------

def _read_with_length(length_expr, reads, *, index=0, ret_write=None):
    """`[ret_write =] read(fd, &buf, <length_expr>)` at 0x401110."""
    fd = FSSA(FVar("fd"), 0)
    buf = FVar("buf", typ="char[0x40]")
    call = FInstr(index, 0x401110, "MLIL_CALL_SSA", f"read(fd#0, &buf, {length_expr})",
                  reads=[fd, *reads], writes=[ret_write] if ret_write else [],
                  dest=_gptr(READ),
                  params=[_var(fd), FExpr("MLIL_ADDRESS_OF", "&buf", src=buf), length_expr])
    return fd, call


def test_tainted_receive_length_is_a_gated_frontier_until_opted_in(models):
    # `read(fd, buf, n)` with `n` the seeded parameter: the receive-length
    # sink is opt-in (`recv_overflow`), so a default run does not REPORT it --
    # but taint armed it, so the run is not an all-clear either.
    n0 = FSSA(FVar("n"), 0)
    fd, call = _read_with_length(_var(n0), [n0])
    func = GFunc("pump", 0x401100, FSSAFunc([call, _ret(1, 0x401118)]),
                 params=[fd.var, n0.var])
    bv = _read_bv()

    result = _forward(bv, func, models, source="param:1")
    [gated] = [lf for lf in result["leaves"] if lf["kind"] == "gated_sink_reached"]
    assert (gated["callee"], gated["class"], gated["gate"], gated["tainted_args"],
            gated["address"]) == ("read", "overflow_len", "recv_overflow", [2], hex(0x401110))
    assert result["reached_sinks"] == []
    assert _all_clear(result) is False
    assert "gated_sink_reached" in BLOCKING_LEAF_KINDS

    # Opted in, the same arming is an ordinary finding and no longer a leaf.
    opted = _forward(bv, func, models, source="param:1",
                     enabled_sink_classes={"recv_overflow"})
    assert [s["sink"]["class"] for s in opted["reached_sinks"]] == ["overflow_len"]
    assert "gated_sink_reached" not in _kinds(opted)


def _fill_loop(cap_from_param):
    """`fill(fd, n)`: `got = 0; loop { r = read(fd, buf, <cap> - got); got += r; }`
    where <cap> is 0x40, or the parameter `n` when *cap_from_param*."""
    got = FVar("got")
    got1, got2, got3 = FSSA(got, 1), FSSA(got, 2), FSSA(got, 3)
    r1, n0 = FSSA(FVar("r"), 1), FSSA(FVar("n"), 0)
    cap = _var(n0) if cap_from_param else _const(0x40)
    length = FExpr("MLIL_SUB", f"{cap} - got#2", reads=[*cap.vars_read, got2],
                   left=cap, right=_var(got2))
    fd, call = _read_with_length(length, list(length.vars_read), index=2, ret_write=r1)
    return GFunc("fill", 0x401100, FSSAFunc([
        FInstr(0, 0x401104, "MLIL_SET_VAR_SSA", "got#1 = 0", writes=[got1], src=_const(0)),
        FInstr(1, 0x401108, "MLIL_VAR_PHI", "got#2 = phi(got#1, got#3)",
               reads=[got1, got3], writes=[got2], src=[got1, got3]),
        call,
        FInstr(3, 0x401114, "MLIL_SET_VAR_SSA", "got#3 = got#2 + r#1", reads=[got2, r1],
               writes=[got3], src=FExpr("MLIL_ADD", "got#2 + r#1", reads=[got2, r1],
                                        left=_var(got2), right=_var(r1))),
        _ret(4, 0x401118),
    ]), params=[fd.var, n0.var])


def test_receive_length_rearmed_only_by_its_own_return_stays_silent(models):
    # The #499 fill loop: the only taint on the length is this same read's own
    # return round the loop -- the idiom the gate exists for.
    bv = _read_bv()
    func = _fill_loop(cap_from_param=False)
    # Taint DOES reach the length (opted in, it is reported)...
    opted = _forward(bv, func, models, source="ret:read", enabled_sink_classes={"recv_overflow"})
    assert [s["sink"]["class"] for s in opted["reached_sinks"]] == ["overflow_len"]
    # ...but by default the self-re-arming loop raises no gated frontier.
    result = _forward(bv, func, models, source="ret:read")
    assert "gated_sink_reached" not in _kinds(result), result["leaves"]
    assert _all_clear(result) is True

    # The same loop whose cap is ALSO a tainted parameter is armed from
    # outside the loop.
    capped = te.TaintEngine(bv, models).forward(
        _fill_loop(cap_from_param=True), [te.parse_locator("ret:read"),
                                          te.parse_locator("param:1")])
    [gated] = [lf for lf in capped["leaves"] if lf["kind"] == "gated_sink_reached"]
    assert gated["tainted_args"] == [2]


def _read_bv():
    return FGlobalBV({READ: "read"})


# --------------------------------------------------------------------------
# P2-10: a seed nothing reads, and an `arg:` seed on an unmodeled callee
# --------------------------------------------------------------------------

def _two_params(read_index):
    """`handler(a, b)` whose body reads only the parameter at *read_index*."""
    a, b = FVar("a"), FVar("b")
    used = FSSA((a, b)[read_index], 0)
    x1 = FSSA(FVar("x"), 1)
    return GFunc("handler", 0x401100, FSSAFunc([
        FInstr(0, 0x401104, "MLIL_SET_VAR_SSA", f"x#1 = {used.var.name}#0", reads=[used],
               writes=[x1], src=_var(used)),
        _ret(1, 0x401108),
    ]), params=[a, b])


def test_a_seed_the_body_never_reads_is_not_an_all_clear(models):
    result = _forward(FGlobalBV({}), _two_params(read_index=1), models, source="param:0")
    [unused] = [lf for lf in result["leaves"] if lf["kind"] == "seed_unused"]
    assert (unused["source"], unused["variable"], unused["address"]) == (
        "param:0", "a", hex(0x401100))
    assert _kinds(result) == ["seed_unused"]
    assert _all_clear(result) is False
    assert "seed_unused" in BLOCKING_LEAF_KINDS

    # Control: the seed IS read (into a value that reaches nothing).
    read = _forward(FGlobalBV({}), _two_params(read_index=0), models, source="param:0")
    assert read["leaves"] == []
    assert _all_clear(read) is True


def _calls_helper_with(arg_kind):
    """`main(p)`: `n = stage_input(0, <arg>)` where `stage_input` is an
    in-binary body with no taint model that returns a constant. *arg_kind*
    "pointer" passes the plain pointer parameter `p` (no buffer to key);
    "local" passes `&local`."""
    p0 = FSSA(FVar("p", typ="char*"), 0)
    n1 = FSSA(FVar("n"), 1)
    if arg_kind == "pointer":
        arg, reads = _var(p0), [p0]
    else:
        arg, reads = FExpr("MLIL_ADDRESS_OF", "&local", src=FVar("local", typ="char[0x40]")), []
    helper = GFunc("stage_input", HELPER, FSSAFunc([
        FInstr(0, HELPER + 4, "MLIL_RET", "return 0", src=[_const(0)])]),
        params=[FVar("fd"), FVar("dst", typ="char*")])
    func = GFunc("main", 0x401100, FSSAFunc([
        FInstr(0, 0x401108, "MLIL_CALL_SSA", f"n#1 = stage_input(0, {arg})", reads=reads,
               writes=[n1], dest=_gptr(HELPER), params=[_const(0), arg]),
        _ret(1, 0x401110),
    ]), params=[p0.var])
    bv = FGlobalBV({HELPER: "stage_input"}, funcs={HELPER: helper})
    return func, bv


def test_arg_seed_on_an_unmodeled_callee_with_an_unkeyable_pointer_is_a_frontier(models):
    # `arg:stage_input:1` where arg 1 is a plain pointer the engine cannot key:
    # only the pointer VALUE is seeded, not what the callee writes through it.
    func, bv = _calls_helper_with("pointer")
    result = _forward(bv, func, models, source="arg:stage_input:1")
    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "unmodeled_callee"]
    assert leaf["reason"] == "seed_callee_unmodeled"
    assert leaf["address"] == hex(0x401108)
    assert leaf["callee"]["address"] == hex(HELPER)
    assert leaf["tainted_args"] == [1]
    assert _all_clear(result) is False

    # Control: a keyable buffer (`&local`) seeds the bytes themselves.
    func, bv = _calls_helper_with("local")
    keyed = _forward(bv, func, models, source="arg:stage_input:1")
    assert not [lf for lf in keyed["leaves"] if lf.get("reason") == "seed_callee_unmodeled"]


# --------------------------------------------------------------------------
# P2-11: a tainted global live at the seeded function's exit
# --------------------------------------------------------------------------

def _exit_program(write, *, clear=None):
    """`main(argv)`: taint G (*write*: "copy" = memcpy, "store" = a direct
    store), optionally `memset(G, 0, <clear>)`, and return. `reader`, which
    `main` never calls, hands G to `system`."""
    argv0 = FSSA(FVar("argv"), 0)
    if write == "copy":
        first = FInstr(0, 0x401104, "MLIL_CALL_SSA", "memcpy(G, argv#0, 0x40)",
                       reads=[argv0], dest=_gptr(MEMCPY),
                       params=[_gptr(G), _var(argv0), _const(G_SIZE)])
    else:
        first = FInstr(0, 0x401104, "MLIL_STORE_SSA", "[G].q = argv#0", reads=[argv0],
                       dest=_gptr(G), src=_var(argv0))
        first.size = 8
    instrs = [first]
    if clear is not None:
        instrs.append(FInstr(1, 0x401108, "MLIL_CALL_SSA", f"memset(G, 0, {hex(clear)})",
                             dest=_gptr(MEMSET), params=[_gptr(G), _const(0), _const(clear)]))
    instrs.append(_ret(len(instrs), 0x401110))
    func = _in_one_block(GFunc("main", 0x401100, FSSAFunc(instrs), params=[argv0.var]))
    reader = _reader("reader", 0x401200, G, 0x401210)
    bv = FExtentBV({MEMCPY: "memcpy", MEMSET: "memset", SYSTEM: "system",
                    0x401200: "reader"},
                   funcs={0x401200: reader}, refs={G: [FRef(reader, 0x401210)]},
                   objects={G: G_SIZE})
    return func, bv


@pytest.mark.parametrize("write", ["copy", "store"])
def test_a_full_constant_fill_before_return_kills_the_exit_frontier(models, write):
    func, bv = _exit_program(write, clear=G_SIZE)
    result = _forward(bv, func, models)
    assert [lf for lf in result["leaves"] if lf["kind"] in BLOCKING_LEAF_KINDS] == [], \
        result["leaves"]
    assert _all_clear(result) is True

    # Control: a fill that misses part of the object kills nothing -- the
    # write stays the exit frontier, naming the reader the walk never ran.
    func, bv = _exit_program(write, clear=4)
    partial = _forward(bv, func, models)
    [store] = _leaves(partial, "coarse_memory_store")
    assert (store["address"], store["global"]["address"]) == (hex(0x401104), hex(G))
    assert [r["display_name"] for r in store["references"]] == ["reader"]
    assert _all_clear(partial) is False


# --------------------------------------------------------------------------
# P2-11: a pointer global initialised to the object reads it
# --------------------------------------------------------------------------

P_SLOT = 0x404200       # `char *P = &G;` -- a writable pointer global
G_ELSE = 0x404300       # an unrelated object


class FAliasBV(FExtentBV):
    """``FExtentBV`` plus BN's DATA cross-references and pointer reads, so a
    pointer global's initial value can be read back."""

    def __init__(self, *args, pointers=None, **kw):
        super().__init__(*args, **kw)
        self._pointers = dict(pointers or {})

    def get_data_refs(self, addr, length=None):
        lo, hi = int(addr), int(addr) + (int(length) if length else 1)
        return [slot for slot, value in sorted(self._pointers.items()) if lo <= value < hi]

    def read_pointer(self, addr):
        return self._pointers[int(addr)]


def _load(addr_expr, size=8):
    load = FExpr("MLIL_LOAD_SSA", f"[{addr_expr}].q", src=addr_expr)
    load.size = size
    return load


def _slot_reader(name, start, slot, at):
    """`system(*P)`: the body names only the pointer slot, never G."""
    p1 = FSSA(FVar("p", typ="char*"), 1)
    return GFunc(name, start, FSSAFunc([
        FInstr(0, at - 4, "MLIL_SET_VAR_SSA", f"p#1 = [{hex(slot)}].q", writes=[p1],
               src=_load(_gptr(slot))),
        FInstr(1, at, "MLIL_CALL_SSA", "system(p#1)", reads=[p1],
               dest=_gptr(SYSTEM), params=[_var(p1)]),
    ]), params=[])


def _copy_then_calls(*callees, copy_at=0x401104):
    """`main(argv)`: `memcpy(G, argv, 0x40)`, then call each of *callees*
    (no arguments) in order."""
    argv0 = FSSA(FVar("argv"), 0)
    instrs = [FInstr(0, copy_at, "MLIL_CALL_SSA", "memcpy(G, argv#0, 0x40)", reads=[argv0],
                     dest=_gptr(MEMCPY), params=[_gptr(G), _var(argv0), _const(G_SIZE)])]
    for i, fn in enumerate(callees, start=1):
        instrs.append(FInstr(i, copy_at + 4 * i, "MLIL_CALL_SSA", f"{fn.name}()",
                             dest=_gptr(fn.start), params=[]))
    return GFunc("main", 0x401100, FSSAFunc(instrs), params=[argv0.var])


def test_a_pointer_global_initialised_to_the_object_is_a_reader(models):
    reader = _slot_reader("via_slot", 0x401200, P_SLOT, 0x401210)
    func = _copy_then_calls(reader)

    def run(points_at):
        bv = FAliasBV({MEMCPY: "memcpy", SYSTEM: "system", reader.start: reader.name},
                      funcs={reader.start: reader},
                      refs={P_SLOT: [FRef(reader, 0x40120c)]},
                      objects={G: G_SIZE, P_SLOT: 8, G_ELSE: G_SIZE},
                      pointers={P_SLOT: points_at})
        return _forward(bv, func, models)

    result = run(G)
    [leaf] = _leaves(result)
    assert leaf["global"]["address"] == hex(G)
    assert leaf["address"] == hex(0x401108)
    assert [r["display_name"] for r in leaf["references"]] == ["via_slot"]
    assert _all_clear(result) is False

    # Control: the slot points at another object -- nothing names G's reader.
    other = run(G_ELSE)
    assert not [lf for lf in other["leaves"]
                if any(r.get("display_name") == "via_slot" for r in lf.get("references") or ())]


# --------------------------------------------------------------------------
# P2-11: a callee fills the caller's global through its pointer parameter
# --------------------------------------------------------------------------

FILL = 0x401400
LOOKUP = 0x4010f8


def _fill_program(dest_kind):
    """`main(argv)`: `fill_name(<dest>, argv)`, then `reader()`; `fill_name(dst,
    src)` is `strncpy(dst, src, 0x40); return 0;`. *dest_kind* "global" passes
    G; "opaque" passes a pointer an unmodeled `lookup_slot()` returned."""
    dst0 = FSSA(FVar("dst", ident=0x101, typ="char*"), 0)
    src0 = FSSA(FVar("src", ident=0x102, typ="char*"), 0)
    fill = GFunc("fill_name", FILL, FSSAFunc([
        FInstr(0, FILL + 4, "MLIL_CALL_SSA", "strncpy(dst#0, src#0, 0x40)", reads=[dst0, src0],
               dest=_gptr(STRNCPY), params=[_var(dst0), _var(src0), _const(0x40)]),
        FInstr(1, FILL + 8, "MLIL_RET", "return 0", src=[_const(0)]),
    ]), params=[dst0.var, src0.var])
    reader = _reader("reader", 0x401200, G, 0x401210)
    argv0, r1 = FSSA(FVar("argv"), 0), FSSA(FVar("r"), 1)
    pre = []
    if dest_kind == "global":
        dest, dreads = _gptr(G), []
    else:
        q1 = FSSA(FVar("q", typ="char*"), 1)
        pre = [FInstr(0, 0x401102, "MLIL_CALL_SSA", "q#1 = lookup_slot()", writes=[q1],
                      dest=_gptr(LOOKUP), params=[])]
        dest, dreads = _var(q1), [q1]
    instrs = [*pre,
              FInstr(0, 0x401104, "MLIL_CALL_SSA", f"r#1 = fill_name({dest}, argv#0)",
                     reads=[*dreads, argv0], writes=[r1], dest=_gptr(FILL),
                     params=[dest, _var(argv0)]),
              FInstr(0, 0x401108, "MLIL_CALL_SSA", "reader()", dest=_gptr(reader.start),
                     params=[]),
              _ret(0, 0x401110)]
    for i, ins in enumerate(instrs):
        ins.instr_index = i
    func = GFunc("main", 0x401100, FSSAFunc(instrs), params=[argv0.var])
    bv = FExtentBV({STRNCPY: "strncpy", SYSTEM: "system", LOOKUP: "lookup_slot",
                    FILL: "fill_name", reader.start: "reader"},
                   funcs={FILL: fill, reader.start: reader},
                   refs={G: [FRef(reader, 0x401210)]}, objects={G: G_SIZE})
    return func, bv


def test_a_callee_filling_the_callers_global_taints_it_there(models):
    func, bv = _fill_program("global")
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert (leaf["global"]["address"], leaf["address"]) == (hex(G), hex(0x401108))
    assert [r["display_name"] for r in leaf["references"]] == ["reader"]
    # The strncpy inside `fill_name` writes ITS parameter: the caller keyed
    # it, so it is not an uncorrelated store.
    assert _leaves(result, "coarse_memory_store") == [], result["leaves"]


def test_a_callee_filling_an_unkeyable_caller_pointer_is_reported_at_the_call(models):
    func, bv = _fill_program("opaque")
    result = _forward(bv, func, models)
    [store] = _leaves(result, "coarse_memory_store")
    assert store["address"] == hex(0x401104)
    assert store["dest_expr"] == "q#1"
    assert _all_clear(result) is False


# --------------------------------------------------------------------------
# P2-11: a heap block kept in a global slot carries its taint through reloads
# --------------------------------------------------------------------------

MALLOC, STRLEN = 0x401140, 0x401150
SLOT, SLOT_OTHER = 0x404400, 0x404408


def _heap_slot_program(reload_from):
    """`main(argv)`: `p = malloc(0x40); *SLOT = p; q = *SLOT; memcpy(q, argv,
    0x40); r = *<reload_from>; n = strlen(r); malloc(n + 1)`. A second block
    sits in SLOT_OTHER."""
    argv0 = FSSA(FVar("argv"), 0)
    p1, p2 = FSSA(FVar("p", typ="char*"), 1), FSSA(FVar("p2", typ="char*"), 1)
    q1, r1 = FSSA(FVar("q", typ="char*"), 1), FSSA(FVar("r", typ="char*"), 1)
    n1, m1, x1 = FSSA(FVar("n"), 1), FSSA(FVar("m"), 1), FSSA(FVar("x", typ="char*"), 1)
    instrs = [
        FInstr(0, 0x401104, "MLIL_CALL_SSA", "p#1 = malloc(0x40)", writes=[p1],
               dest=_gptr(MALLOC), params=[_const(0x40)]),
        FInstr(1, 0x401108, "MLIL_STORE_SSA", "[SLOT].q = p#1", reads=[p1],
               dest=_gptr(SLOT), src=_var(p1)),
        FInstr(2, 0x40110c, "MLIL_CALL_SSA", "p2#1 = malloc(0x40)", writes=[p2],
               dest=_gptr(MALLOC), params=[_const(0x40)]),
        FInstr(3, 0x401110, "MLIL_STORE_SSA", "[SLOT_OTHER].q = p2#1", reads=[p2],
               dest=_gptr(SLOT_OTHER), src=_var(p2)),
        FInstr(4, 0x401114, "MLIL_SET_VAR_SSA", "q#1 = [SLOT].q", writes=[q1],
               src=_load(_gptr(SLOT))),
        FInstr(5, 0x401118, "MLIL_CALL_SSA", "memcpy(q#1, argv#0, 0x40)", reads=[q1, argv0],
               dest=_gptr(MEMCPY), params=[_var(q1), _var(argv0), _const(0x40)]),
        FInstr(6, 0x40111c, "MLIL_SET_VAR_SSA", "r#1 = [..].q", writes=[r1],
               src=_load(_gptr(reload_from))),
        FInstr(7, 0x401120, "MLIL_CALL_SSA", "n#1 = strlen(r#1)", reads=[r1], writes=[n1],
               dest=_gptr(STRLEN), params=[_var(r1)]),
        FInstr(8, 0x401124, "MLIL_SET_VAR_SSA", "m#1 = n#1 + 1", reads=[n1], writes=[m1],
               src=FExpr("MLIL_ADD", "n#1 + 1", reads=[n1], left=_var(n1), right=_const(1))),
        FInstr(9, 0x401128, "MLIL_CALL_SSA", "x#1 = malloc(m#1)", reads=[m1], writes=[x1],
               dest=_gptr(MALLOC), params=[_var(m1)]),
        _ret(10, 0x401130),
    ]
    func = GFunc("main", 0x401100, FSSAFunc(instrs), params=[argv0.var])
    bv = FGlobalBV({MALLOC: "malloc", STRLEN: "strlen", MEMCPY: "memcpy"})
    return func, bv


def test_a_heap_block_reloaded_from_a_global_slot_stays_tainted(models):
    func, bv = _heap_slot_program(SLOT)
    result = _forward(bv, func, models)
    assert [(s["sink"]["class"], s["sink"]["address"]) for s in result["reached_sinks"]] == [
        ("alloc_size", hex(0x401128))]

    # Control: the other slot holds a clean block -- its length is not tainted.
    func, bv = _heap_slot_program(SLOT_OTHER)
    assert _forward(bv, func, models)["reached_sinks"] == []


# --------------------------------------------------------------------------
# P2-08: a field store taints that field, not the whole struct
# --------------------------------------------------------------------------

def _field(var_ssa, offset, size=8):
    """`<struct>.<offset>` read as a value."""
    e = FExpr("MLIL_VAR_ALIASED_FIELD", f"{var_ssa.var.name}#{var_ssa.version}.{offset:#x}",
              reads=[var_ssa], src=var_ssa)
    e.offset, e.size = offset, size
    return e


def _field_store(index, addr, struct, prev_version, value_ssa, offset, size=8):
    """`struct.<offset> = value` (no vars_written, as BN exposes it)."""
    dest = FSSA(struct, prev_version + 1)
    value = _var(value_ssa)
    value.size = size
    return FInstr(index, addr, "MLIL_SET_VAR_ALIASED_FIELD",
                  f"{struct.name}#{dest.version}.{offset:#x} = {value}", reads=[value_ssa],
                  dest=dest, src=value, prev=FSSA(struct, prev_version), offset=offset), dest


def test_a_local_struct_field_store_taints_only_that_field(models):
    # `options.len = atoi(arg)`; `malloc(options.flags)` reads another field,
    # `malloc(options.len)` the tainted one.
    arg0 = FSSA(FVar("arg"), 0)
    options = FVar("options", typ="struct opts")
    store, opt2 = _field_store(0, 0x401104, options, 1, arg0, offset=16)
    a1, b1 = FSSA(FVar("a"), 1), FSSA(FVar("b"), 1)
    func = GFunc("main", 0x401100, FSSAFunc([
        store,
        FInstr(1, 0x401108, "MLIL_SET_VAR_SSA", "a#1 = options#2.0", reads=[opt2], writes=[a1],
               src=_field(opt2, 0)),
        FInstr(2, 0x40110c, "MLIL_CALL_SSA", "malloc(a#1)", reads=[a1],
               dest=_gptr(MALLOC), params=[_var(a1)]),
        FInstr(3, 0x401110, "MLIL_SET_VAR_SSA", "b#1 = options#2.0x10", reads=[opt2],
               writes=[b1], src=_field(opt2, 16)),
        FInstr(4, 0x401114, "MLIL_CALL_SSA", "malloc(b#1)", reads=[b1],
               dest=_gptr(MALLOC), params=[_var(b1)]),
        _ret(5, 0x401118),
    ]), params=[arg0.var])
    result = _forward(FGlobalBV({MALLOC: "malloc"}), func, models)
    assert [(s["sink"]["class"], s["sink"]["address"]) for s in result["reached_sinks"]] == [
        ("alloc_size", hex(0x401114))]


def _field_consumer(name, start, offset):
    """`consume(struct opts *o)`: `memcpy(dst, src, o-><offset>)`; *offset*
    None reads `o[i]` at an index the walk cannot resolve."""
    o0, i0 = FSSA(FVar("o", typ="struct opts*"), 0), FSSA(FVar("i"), 0)
    n1 = FSSA(FVar("n"), 1)
    if offset is None:
        addr = FExpr("MLIL_ADD", "o#0 + i#0", reads=[o0, i0], left=_var(o0), right=_var(i0))
    else:
        addr = FExpr("MLIL_ADD", f"o#0 + {offset:#x}", reads=[o0], left=_var(o0),
                     right=_const(offset))
    load = _load(addr)
    load.vars_read = list(addr.vars_read)
    dst = FExpr("MLIL_ADDRESS_OF", "&dst", src=FVar("dst", typ="char[0x100]"))
    return GFunc(name, start, FSSAFunc([
        FInstr(0, start + 4, "MLIL_SET_VAR_SSA", f"n#1 = [{addr}].q", reads=list(addr.vars_read),
               writes=[n1], src=load),
        FInstr(1, start + 8, "MLIL_CALL_SSA", "memcpy(&dst, o#0, n#1)", reads=[o0, n1],
               dest=_gptr(MEMCPY), params=[dst, _var(o0), _var(n1)]),
        FInstr(2, start + 12, "MLIL_RET", "return 0", src=[_const(0)]),
    ]), params=[o0.var, i0.var])


def test_a_struct_passed_by_address_carries_only_its_tainted_field(models):
    # `options.len = atoi(arg)` at +16, then `options` handed by address to
    # three readers: of +0x48 (clean), of +16 (tainted), of +i (unknown).
    arg0 = FSSA(FVar("arg"), 0)
    options = FVar("options", typ="struct opts")
    store, _opt2 = _field_store(0, 0x401104, options, 1, arg0, offset=16)
    far = _field_consumer("use_flags", 0x401500, 0x48)
    hit = _field_consumer("use_len", 0x401600, 16)
    anyf = _field_consumer("use_indexed", 0x401700, None)
    instrs = [store]
    for i, fn in enumerate((far, hit, anyf), start=1):
        r = FSSA(FVar(f"r{i}"), 1)
        instrs.append(FInstr(i, 0x401104 + 4 * i, "MLIL_CALL_SSA",
                             f"r{i}#1 = {fn.name}(&options, 0)", writes=[r],
                             dest=_gptr(fn.start),
                             params=[FExpr("MLIL_ADDRESS_OF", "&options", src=options),
                                     _const(0)]))
    instrs.append(_ret(4, 0x401120))
    func = GFunc("main", 0x401100, FSSAFunc(instrs), params=[arg0.var])
    bv = FGlobalBV({MEMCPY: "memcpy", far.start: far.name, hit.start: hit.name,
                    anyf.start: anyf.name},
                   funcs={far.start: far, hit.start: hit, anyf.start: anyf})
    result = _forward(bv, func, models)
    assert [(s["sink"]["class"], s["sink"]["address"]) for s in result["reached_sinks"]] == [
        ("overflow_len", hex(0x401608))]
    [unknown] = [lf for lf in result["leaves"] if lf["kind"] == "field_load_unresolved"]
    assert unknown["address"] == hex(0x401704)


def test_a_global_struct_field_store_taints_only_that_field(models):
    # `g_opts.len = arg` (a direct store at G+16), then `malloc(g_opts.flags)`
    # (G+0, clean) and `malloc(g_opts.len)` (tainted).
    arg0 = FSSA(FVar("arg"), 0)
    a1, b1 = FSSA(FVar("a"), 1), FSSA(FVar("b"), 1)
    store = FInstr(0, 0x401104, "MLIL_STORE_SSA", "[G + 0x10].q = arg#0", reads=[arg0],
                   dest=_gptr(G + 16), src=_var(arg0))
    store.size = 8
    func = GFunc("main", 0x401100, FSSAFunc([
        store,
        FInstr(1, 0x401108, "MLIL_SET_VAR_SSA", "a#1 = [G].q", writes=[a1], src=_load(_gptr(G))),
        FInstr(2, 0x40110c, "MLIL_CALL_SSA", "malloc(a#1)", reads=[a1],
               dest=_gptr(MALLOC), params=[_var(a1)]),
        FInstr(3, 0x401110, "MLIL_SET_VAR_SSA", "b#1 = [G + 0x10].q", writes=[b1],
               src=_load(_gptr(G + 16))),
        FInstr(4, 0x401114, "MLIL_CALL_SSA", "malloc(b#1)", reads=[b1],
               dest=_gptr(MALLOC), params=[_var(b1)]),
        _ret(5, 0x401118),
    ]), params=[arg0.var])
    bv = FExtentBV({MALLOC: "malloc"}, objects={G: G_SIZE})
    result = _forward(bv, func, models)
    assert [(s["sink"]["class"], s["sink"]["address"]) for s in result["reached_sinks"]] == [
        ("alloc_size", hex(0x401114))]


# V1-01 / V2-01: a member pointing into the same partially tainted struct

def _through_member(name, start, member, field):
    """`use(struct session *o)`: `c = o-><member>; memcpy(dst, o, c-><field>)`
    -- the length is loaded through a pointer the struct holds."""
    o0 = FSSA(FVar("o", typ="struct session*"), 0)
    c1, n1 = FSSA(FVar("c"), 1), FSSA(FVar("n"), 1)
    at_member = FExpr("MLIL_ADD", f"o#0 + {member:#x}", reads=[o0], left=_var(o0),
                      right=_const(member))
    at_field = FExpr("MLIL_ADD", f"c#1 + {field:#x}", reads=[c1], left=_var(c1),
                     right=_const(field))
    dst = FExpr("MLIL_ADDRESS_OF", "&dst", src=FVar("dst", typ="char[0x100]"))
    return GFunc(name, start, FSSAFunc([
        FInstr(0, start + 4, "MLIL_SET_VAR_SSA", f"c#1 = [{at_member}].q", reads=[o0],
               writes=[c1], src=_load(at_member)),
        FInstr(1, start + 8, "MLIL_SET_VAR_SSA", f"n#1 = [{at_field}].q", reads=[c1],
               writes=[n1], src=_load(at_field)),
        FInstr(2, start + 12, "MLIL_CALL_SSA", "memcpy(&dst, o#0, n#1)", reads=[o0, n1],
               dest=_gptr(MEMCPY), params=[dst, _var(o0), _var(n1)]),
        FInstr(3, start + 16, "MLIL_RET", "return 0", src=[_const(0)]),
    ]), params=[o0.var])


def _self_pointer_program(known_slot):
    """`main(arg, k)`: `s.active = &s.slots[0]` (or `&s.slots[k]`) at +0x10,
    `s.slots[0].len = arg` at +0x28, then `s` handed by address to a reader
    of `s.active->len` (+8 into the slot) and one of `s.active->spare`
    (+0x10, clean)."""
    arg0, k0 = FSSA(FVar("arg"), 0), FSSA(FVar("k"), 0)
    s = FVar("s", typ="struct session")
    slot = FExpr("MLIL_ADDRESS_OF_FIELD", "&s.0x20", src=s)
    slot.offset = 0x20
    if not known_slot:
        slot = FExpr("MLIL_ADD", "&s.0x20 + k#0", reads=[k0], left=slot, right=_var(k0))
    point = FInstr(0, 0x401104, "MLIL_SET_VAR_ALIASED_FIELD", f"s#2.0x10 = {slot}",
                   reads=list(getattr(slot, "vars_read", []) or []), dest=FSSA(s, 2),
                   src=slot, prev=FSSA(s, 1), offset=0x10)
    point.size = 8
    fill, _s3 = _field_store(1, 0x401108, s, 2, arg0, offset=0x28)
    hit = _through_member("use_len", 0x401600, 0x10, 0x8)
    clean = _through_member("use_spare", 0x401700, 0x10, 0x10)
    instrs = [point, fill]
    for i, fn in enumerate((hit, clean), start=2):
        r = FSSA(FVar(f"r{i}"), 1)
        instrs.append(FInstr(i, 0x401104 + 4 * i, "MLIL_CALL_SSA", f"r{i}#1 = {fn.name}(&s)",
                             writes=[r], dest=_gptr(fn.start),
                             params=[FExpr("MLIL_ADDRESS_OF", "&s", src=s)]))
    instrs.append(_ret(4, 0x401120))
    func = GFunc("main", 0x401100, FSSAFunc(instrs), params=[arg0.var, k0.var])
    bv = FGlobalBV({MEMCPY: "memcpy", hit.start: hit.name, clean.start: clean.name},
                   funcs={hit.start: hit, clean.start: clean})
    return func, bv


def test_a_load_through_a_member_pointing_into_the_struct_reads_its_field(models):
    func, bv = _self_pointer_program(known_slot=True)
    result = _forward(bv, func, models)
    assert [(s["sink"]["class"], s["sink"]["address"]) for s in result["reached_sinks"]] == [
        ("overflow_len", hex(0x40160c))]
    assert not [lf for lf in result["leaves"] if lf["kind"] == "field_load_unresolved"], \
        result["leaves"]


def test_a_member_pointing_at_an_unknown_place_in_the_struct_is_a_frontier(models):
    func, bv = _self_pointer_program(known_slot=False)
    result = _forward(bv, func, models)
    assert result["reached_sinks"] == [], result["reached_sinks"]
    unresolved = {lf["address"] for lf in result["leaves"]
                  if lf["kind"] == "field_load_unresolved"}
    assert hex(0x401608) in unresolved and hex(0x401708) in unresolved, result["leaves"]
    assert _all_clear(result) is False


# --------------------------------------------------------------------------
# P2-04: one continuation leaf per tainted global
# --------------------------------------------------------------------------

def test_one_continuation_leaf_per_global_names_every_call_and_reader(models):
    first = _reader("reader_a", 0x401200, G, 0x401210)
    second = _reader("reader_b", 0x401300, G, 0x401310)
    func = _copy_then_calls(first, second)
    bv = FExtentBV({MEMCPY: "memcpy", SYSTEM: "system", first.start: first.name,
                    second.start: second.name},
                   funcs={first.start: first, second.start: second},
                   refs={G: [FRef(first, 0x401210), FRef(second, 0x401310)]},
                   objects={G: G_SIZE})
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert [(s["address"], s["callee"]["name"], s["descended"]) for s in leaf["sites"]] == [
        (hex(0x401108), "reader_a", False), (hex(0x40110c), "reader_b", False)]
    assert sorted(r["display_name"] for r in leaf["references"]) == ["reader_a", "reader_b"]
    assert result["stats"]["leaves"] == len(result["leaves"])


# --------------------------------------------------------------------------
# R3 sweep economy: bodies that cannot see the global are not scanned
# --------------------------------------------------------------------------

def test_bodies_that_cannot_see_the_global_do_not_spend_the_sweep(models, monkeypatch):
    # `main` fills the global and calls `fanout`, which calls 40 bodies that
    # neither name the global nor return a pointer, then the one reader. An
    # allowance that classifying all 40 would exhaust (the base engine
    # reports this search `truncated` with no reader) still finds the reader.
    monkeypatch.setattr(te.TaintEngine, "_SWEEP_UNITS", 20)
    func, bv = _fanout(41)
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["reachability"]["status"] == "complete", leaf["reachability"]
    assert [r["display_name"] for r in leaf["references"]] == ["consumer"]


class FEdgeBV(FExtentBV):
    """``FExtentBV`` plus BN's recorded call edges: each body's ``call_sites``
    and ``callers``, and the resolved targets of a call site."""

    def __init__(self, *args, edges=None, **kw):
        super().__init__(*args, **kw)
        self._edges = dict(edges or {})

    def get_callees(self, addr, func=None):
        return list(self._edges.get(int(addr), ()))


def _unrelated_breadth(with_edges, n=40):
    """`main` fills G, then calls `hub`, whose *n* callees touch no global; the
    only reader of G is `consumer`, called from `elsewhere` -- nowhere `main`
    reaches."""
    consumer = _reader("consumer", 0x402000, G, 0x402010)
    elsewhere = GFunc("elsewhere", 0x402100, FSSAFunc([
        FInstr(0, 0x402104, "MLIL_CALL_SSA", "consumer()", dest=_gptr(0x402000), params=[])]),
        params=[], callees=[consumer])
    quiet = [GFunc(f"q{i}", 0x410000 + i * 0x10,
                   FSSAFunc([FInstr(0, 0x410000 + i * 0x10, "MLIL_RET", "return 0")]),
                   params=[]) for i in range(n)]
    hub = GFunc("hub", 0x401300, FSSAFunc([
        FInstr(i, 0x401304 + 4 * i, "MLIL_CALL_SSA", f"{q.name}()", dest=_gptr(q.start),
               params=[]) for i, q in enumerate(quiet)]), params=[], callees=quiet)
    func = _copy_then_calls(hub)
    edges = {}
    if with_edges:
        for body, callees in ((hub, quiet), (elsewhere, [consumer]), (func, [hub])):
            body.call_sites = []
            for c in callees:
                site = next(int(i.address) for i in body.mlil.ssa_form.instructions
                            if "CALL" in i.operation.name
                            and int(getattr(i.dest, "constant", -1)) == c.start)
                body.call_sites.append(SimpleNamespace(address=site))
                edges[site] = [c.start]
                c.callers = [*getattr(c, "callers", []), body]
        for q in quiet:
            q.call_sites = []
        consumer.call_sites, elsewhere.callers = [], []
    names = {MEMCPY: "memcpy", SYSTEM: "system", hub.start: "hub", consumer.start: "consumer",
             elsewhere.start: "elsewhere", **{q.start: q.name for q in quiet}}
    funcs = {hub.start: hub, consumer.start: consumer, elsewhere.start: elsewhere,
             **{q.start: q for q in quiet}}
    bv = FEdgeBV(names, funcs=funcs, refs={G: [FRef(consumer, 0x402010)]},
                 objects={G: G_SIZE}, edges=edges)
    return func, bv, quiet


def test_bodies_no_reader_lies_beyond_are_settled_on_call_edges(models, monkeypatch):
    # V2-02: when BN's call edges show no possible reader of the global below
    # a call, the walk answers from the edges -- the same answer as lifting
    # every body on the way -- and never classifies the unrelated bodies.
    lifted = []
    original = te.TaintEngine._not_internal_reason

    def spy(self, fn):
        lifted.append(getattr(fn, "name", None))
        return original(self, fn)

    func, bv, _quiet = _unrelated_breadth(with_edges=False)
    baseline = _forward(bv, func, models)
    monkeypatch.setattr(te.TaintEngine, "_not_internal_reason", spy)
    func, bv, quiet = _unrelated_breadth(with_edges=True)
    result = _forward(bv, func, models)
    assert result["leaves"] == baseline["leaves"]
    assert result["reached_sinks"] == baseline["reached_sinks"]
    assert _all_clear(result) == _all_clear(baseline)
    assert not {q.name for q in quiet} & set(lifted), lifted
