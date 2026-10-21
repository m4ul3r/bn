"""Backward caller/demand coverage (IC01): a backward slice that bottoms out at a
formal parameter accounts for EVERY formal demand at EVERY known caller -- a
site it could not follow, a function reached only through a stored address, a
caller the configured ``--max-depth`` cut off, a demand a recursive call remaps
to another formal, and a def chain the engine's step budget cut are each
structured incompleteness, whichever origin the slice happens to name as its
representative. A chain that WAS walked to its origins (a caller's literal, a
formal with no known caller) stays a complete slice."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import FixtureBuildError
from test_taint_backward_loads import _FieldProgram
from test_taint_engine import FBV, FSSA, FExpr, FFunc, FInstr, FSite, FSSAFunc, FVar, te

_MEMCPY, _MALLOC, _RECV, _GETPID = 0x940, 0x950, 0x960, 0x970
_NAMES = {_MEMCPY: "memcpy", _MALLOC: "malloc", _RECV: "recv", _GETPID: "getpid"}


class _BV(FBV):
    """FBV with BN's reverse reference queries: *data_refs* maps a function
    start to the data addresses that store it (a dispatch table), *code_refs*
    to ``(address, referencing function)`` code references (a call site, or
    `mov edi, handler` handing it on); *entry* is the view's entry function."""

    def __init__(self, names, data_refs=None, code_refs=None, entry=None):
        super().__init__(names)
        self._data_refs = data_refs or {}
        self._code_refs = code_refs or {}
        self.entry_function = entry

    def get_data_refs(self, addr):
        return list(self._data_refs.get(int(addr), []))

    def get_code_refs(self, addr):
        return [FSite(fn, a) for a, fn in self._code_refs.get(int(addr), [])]


def _var(e):
    return FExpr("MLIL_VAR_SSA", f"{e.var.name}#{e.version}", reads=[e])


def _const(c):
    return FExpr("MLIL_CONST", hex(c), constant=c)


def _call(index, addr, target, params, writes=()):
    reads = [r for p in params for r in p.vars_read]
    return FInstr(index, addr, "MLIL_CALL_SSA", f"{target:#x}(...)", reads=reads,
                  writes=list(writes), dest=FExpr("MLIL_CONST_PTR", hex(target), constant=target),
                  params=params)


def _length_sink(name, start, ident):
    """``name(dst, src, n) { memcpy(dst, src, n); }`` -- the length is the formal n."""
    dst, src, n = FVar("dst", ident=ident), FVar("src", ident=ident + 1), FVar("n", ident=ident + 2)
    n0 = FSSA(n, 0)
    fn = FFunc(name, start, FSSAFunc([
        _call(0, start + 4, _MEMCPY, [FExpr("MLIL_VAR_SSA", "dst#0"),
                                      FExpr("MLIL_VAR_SSA", "src#0"), _var(n0)]),
    ]), params=[dst, src, n])
    return fn


def _caller(name, start, call_addr, callee_start, arg, params=()):
    """``name(..) { callee(a, b, <arg>); }`` with *arg* the length operand."""
    return FFunc(name, start, FSSAFunc([
        _call(0, call_addr, callee_start, [FExpr("MLIL_VAR_SSA", "a"),
                                           FExpr("MLIL_VAR_SSA", "b"), arg]),
    ]), params=list(params))


def _backward(func, names, sink="arg:memcpy:2", *, max_depth=8, engine_depth=None, data_refs=None,
              code_refs=None, entry=None):
    bv = _BV(names, data_refs, code_refs, entry)
    kw = {} if engine_depth is None else {"max_depth": engine_depth}
    return te.TaintEngine(bv, te.load_models(), **kw).backward(
        func, [te.parse_locator(sink)], max_depth=max_depth)


def _gate(result):
    return result["diagnostics"]["safe_to_report_complete_slice"]


def _caller_leaves(result):
    return [lf for lf in result["leaves"] if lf.get("kind") == "caller_sites_truncated"]


def _origins(result):
    return [sl["origin"] for sl in result["slices"]]


# --------------------------------------------------------------------------
# TT-PILOT-HOST02-01: below-cap caller sites the ascent could not follow
# --------------------------------------------------------------------------

def _put_record(third_arg):
    """``put_record(dst, src, n)`` with three direct callers: one forwards its
    own formal, one passes the literal 64, one passes *third_arg*."""
    put = _length_sink("put_record", 0x800, 20)
    x = FVar("argc", ident=40)
    x0 = FSSA(x, 0)
    main = _caller("main", 0xa00, 0xa10, 0x800, _var(x0), params=[x])
    lit = _caller("from_literal", 0xb00, 0xb10, 0x800, _const(64))
    third = _caller("from_config", 0xc00, 0xc10, 0x800, third_arg)
    put.caller_sites = [FSite(main, 0xa10), FSite(lit, 0xb10), FSite(third, 0xc10)]
    return put


def test_constant_address_caller_argument_is_an_unfollowed_site_not_a_complete_slice():
    # `put_record(g_out, src, &g_rec_len)`-style: a constant ADDRESS has no
    # def chain the ascent can walk, so that caller's contribution is not
    # accounted for -- named per site, never a silent caveat beside a
    # complete gate.
    put = _put_record(FExpr("MLIL_CONST_PTR", "0x404028", constant=0x404028))
    result = _backward(put, {**_NAMES, 0x800: "put_record"})

    assert _gate(result) is False, result["diagnostics"]
    (leaf,) = _caller_leaves(result)
    assert leaf["callers_total"] == 3
    assert leaf["callers_followed"] == 2
    assert leaf["callers_dropped"] == 1
    assert leaf["unfollowed"] == [{"address": "0xc10", "caller": "from_config",
                                   "param": 2, "reason": "constant_address"}]
    assert leaf["reasons"] == {"constant_address": 1}
    # the followed formal and the caller's literal are both origins
    assert {"kind": "parameter", "index": 0, "var": "argc#0"} in _origins(result)
    assert {"kind": "constant", "value": 64} in _origins(result)


def test_caller_site_with_no_call_instruction_below_the_cap_blocks_the_gate():
    # Below the 16-site cap, a site whose recorded address carries no call
    # instruction used to `continue` silently: two of three callers followed
    # and the slice certified complete.
    put = _put_record(FExpr("MLIL_VAR_SSA", "a"))
    put.caller_sites[2].address = 0xc18
    result = _backward(put, {**_NAMES, 0x800: "put_record"})

    assert _gate(result) is False, result["diagnostics"]
    (leaf,) = _caller_leaves(result)
    assert (leaf["callers_total"], leaf["callers_followed"], leaf["callers_dropped"]) == (3, 2, 1)
    assert leaf["unfollowed"] == [{"address": "0xc18", "caller": "from_config",
                                   "param": 2, "reason": "no_call_instruction"}]
    assert result["stats"]["truncated"] is False  # not the cap: nothing was budgeted away


def test_literal_caller_arguments_are_constant_origins_of_a_complete_slice():
    # Control (`put_fixed`): every caller passes a literal length. Each is an
    # origin the walk reached -- a complete slice, not an unfollowed caller.
    put = _length_sink("put_fixed", 0x800, 20)
    a = _caller("from_config", 0xb00, 0xb10, 0x800, _const(200))
    b = _caller("from_literal", 0xc00, 0xc10, 0x800, _const(32))
    put.caller_sites = [FSite(a, 0xb10), FSite(b, 0xc10)]
    result = _backward(put, {**_NAMES, 0x800: "put_fixed"})

    assert _gate(result) is True, result["diagnostics"]
    assert _caller_leaves(result) == []
    assert sorted(o["value"] for o in _origins(result)) == [32, 200]
    assert all(sl["crossed_functions"] == ["put_fixed"] for sl in result["slices"])


# --------------------------------------------------------------------------
# TT-PILOT-FX03-01: callers known only through a stored address
# --------------------------------------------------------------------------

def test_function_reached_only_through_a_stored_address_is_not_a_complete_slice():
    # `h_copy` sits in a dispatch table: no direct call site, one data
    # reference. Its formal is not where the length comes from -- the
    # indirect callers are, and they were not followed.
    h_copy = _length_sink("h_copy", 0x800, 20)
    result = _backward(h_copy, {**_NAMES, 0x800: "h_copy"}, data_refs={0x800: [0x403df0]})

    assert _gate(result) is False, result["diagnostics"]
    (leaf,) = _caller_leaves(result)
    assert leaf["reasons"] == {"indirect_reference": 1}
    assert leaf["unfollowed"] == [{"address": "0x403df0", "caller": None,
                                   "param": 2, "reason": "indirect_reference"}]
    assert (leaf["callers_followed"], leaf["callers_dropped"]) == (0, 1)
    assert "indirect" in result["diagnostics"]["next_action"]


def test_function_with_no_known_caller_stays_a_complete_parameter_slice():
    # Control: no call site and no stored address -- the formal IS the origin
    # (an exported entry point), exactly as before.
    h_copy = _length_sink("h_copy", 0x800, 20)
    result = _backward(h_copy, {**_NAMES, 0x800: "h_copy"})

    assert _gate(result) is True, result["diagnostics"]
    assert _caller_leaves(result) == []
    assert _origins(result) == [{"kind": "parameter", "index": 2, "var": "n#0"}]


def _copy_handler():
    """``copy_handler(.., n) { memcpy(.., n); }`` with no direct call site."""
    return _length_sink("copy_handler", 0x800, 20)


def test_callback_handed_on_in_code_is_an_indirect_caller_not_a_complete_slice():
    # `dispatch(copy_handler, n)`: main takes the handler's address in code
    # (`mov edi, copy_handler`) and dispatch calls it through the pointer. No
    # data reference and no direct call site, so the formal was certified as
    # the origin; the caller that supplies `n` was never followed.
    main = FFunc("main", 0xa00, FSSAFunc([]))
    start = FFunc("_start", 0x400, FSSAFunc([]))
    result = _backward(_copy_handler(), {**_NAMES, 0x800: "copy_handler"},
                       code_refs={0x800: [(0xa1e, main)]}, entry=start)

    assert _gate(result) is False, result["diagnostics"]
    (leaf,) = _caller_leaves(result)
    assert leaf["unfollowed"] == [{"address": "0xa1e", "caller": "main",
                                   "param": 2, "reason": "indirect_reference"}]
    assert "handed on as a value" in result["diagnostics"]["next_action"]


def test_callback_the_entry_function_hands_on_is_an_indirect_caller():
    # A custom entry (`-nostartfiles -e entry_point`) calling
    # `dispatch(copy_handler, 17)`: the entry function's reference is not the
    # runtime's handoff of `main`, so the handler's caller is unfollowed.
    entry = FFunc("entry_point", 0x400, FSSAFunc([]))
    result = _backward(_copy_handler(), {**_NAMES, 0x800: "copy_handler"},
                       code_refs={0x800: [(0x40d, entry), (0x414, entry)]}, entry=entry)

    assert _gate(result) is False, result["diagnostics"]
    (leaf,) = _caller_leaves(result)
    assert [(r["address"], r["caller"], r["reason"]) for r in leaf["unfollowed"]] == [
        ("0x40d", "entry_point", "indirect_reference"),
        ("0x414", "entry_point", "indirect_reference")]


_LIBC_START_SLOT, _DISPATCH = 0x990, 0x980
_ENTRY_NAMES = {_LIBC_START_SLOT: "__libc_start_main", _DISPATCH: "dispatch"}


def _main_fn(ident):
    """``main(argc) { memcpy(.., argc); }`` at 0x800."""
    argc = FVar("argc", ident=ident)
    return FFunc("main", 0x800, FSSAFunc([
        _call(0, 0x804, _MEMCPY, [FExpr("MLIL_VAR_SSA", "d"), FExpr("MLIL_VAR_SSA", "s"),
                                  _var(FSSA(argc, 0))]),
    ]), params=[argc])


def _entry(name, calls):
    """An entry function whose calls hand on the address 0x800: *calls* is
    ``[(callee, is_import)]`` (0x800 as operand 0) or ``[(callee, is_import,
    operand)]``."""
    instrs = []
    for i, (callee, imported, *where) in enumerate(calls):
        dest = (FExpr("MLIL_IMPORT", "slot", constant=callee) if imported
                else FExpr("MLIL_CONST_PTR", hex(callee), constant=callee))
        params = [_const(17), _const(0), _const(0), _const(0)]
        params[where[0] if where else 0] = FExpr("MLIL_CONST_PTR", "0x800", constant=0x800)
        instrs.append(FInstr(i, 0x410 + 8 * i, "MLIL_CALL_SSA", "call", dest=dest,
                             params=params))
    return FFunc(name, 0x400, FSSAFunc(instrs))


def test_main_handed_on_by_an_ordinary_function_is_an_indirect_caller():
    # Only the entry function's runtime handoff of `main` is exempt; another
    # function taking `main`'s address is a callback like any other.
    start = _entry("_start", [(_LIBC_START_SLOT, True)])
    hooks = FFunc("register_hooks", 0xb00, FSSAFunc([]))
    result = _backward(_main_fn(42), {**_NAMES, **_ENTRY_NAMES, 0x800: "main"},
                       code_refs={0x800: [(0x41e, start), (0xb0c, hooks)]}, entry=start)

    assert _gate(result) is False, result["diagnostics"]
    (leaf,) = _caller_leaves(result)
    assert [(r["address"], r["caller"]) for r in leaf["unfollowed"]] == [("0xb0c", "register_hooks")]


@pytest.mark.parametrize("calls", [
    [(_DISPATCH, False)],                              # entry_point: dispatch(main, 17)
    [(_LIBC_START_SLOT, True), (_DISPATCH, False)],    # the handoff AND an in-binary dispatch
    [],                                                 # an entry reference with no handoff evidence
    [(_LIBC_START_SLOT, True, 3)],                      # handed to the runtime, but not as main
], ids=["dispatch-only", "handoff-and-dispatch", "no-evidence", "not-the-main-operand"])
def test_callback_named_main_the_entry_dispatches_is_an_indirect_caller(calls):
    # The exemption is evidence, not a name: a custom entry handing `main` to
    # its own dispatcher is an in-binary indirect caller of `main`.
    entry = _entry("entry_point", calls)
    result = _backward(_main_fn(43), {**_NAMES, **_ENTRY_NAMES, 0x800: "main"},
                       code_refs={0x800: [(0x40d, entry)]}, entry=entry)

    assert _gate(result) is False, result["diagnostics"]
    (leaf,) = _caller_leaves(result)
    assert [(r["address"], r["caller"]) for r in leaf["unfollowed"]] == [("0x40d", "entry_point")]


@pytest.mark.parametrize("imported", [True, False], ids=["plt-import", "static-direct"])
def test_process_entry_handoff_of_main_and_direct_call_refs_are_not_indirect_callers(imported):
    # Controls: `_start` handing `main` to `__libc_start_main` (through the
    # GOT, or a direct call in a static binary) is no in-binary caller -- its
    # formals are what the loader supplied -- and the code reference a
    # direct call site makes is that site, already followed.
    start = _entry("_start", [(_LIBC_START_SLOT, imported)])
    handoff = _backward(_main_fn(41), {**_NAMES, **_ENTRY_NAMES, 0x800: "main"},
                        code_refs={0x800: [(0x41e, start)]}, entry=start)
    assert _gate(handoff) is True, handoff["diagnostics"]
    assert _caller_leaves(handoff) == []
    assert _origins(handoff) == [{"kind": "parameter", "index": 0, "var": "argc#0"}]

    put = _length_sink("put", 0x800, 20)
    x = FVar("argc", ident=40)
    main = _caller("main", 0xa00, 0xa10, 0x800, _var(FSSA(x, 0)), params=[x])
    put.caller_sites = [FSite(main, 0xa10)]
    direct = _backward(put, {**_NAMES, 0x800: "put"},
                       code_refs={0x800: [(0xa10, main)]}, entry=start)
    assert _gate(direct) is True, direct["diagnostics"]
    assert _caller_leaves(direct) == []
    assert {"kind": "parameter", "index": 0, "var": "argc#0"} in _origins(direct)


# --------------------------------------------------------------------------
# BN6-C003: the configured caller depth cuts a known caller
# --------------------------------------------------------------------------

def _depth_chain():
    """``main(argc) -> middle_copy(argc) -> sink_copy(n) -> memcpy(.., n)``,
    and a second caller of sink_copy that passes the literal 7."""
    sink = _length_sink("sink_copy", 0x800, 20)
    m = FVar("n", ident=30)
    m0 = FSSA(m, 0)
    middle = _caller("middle_copy", 0x900, 0x910, 0x800, _var(m0), params=[FVar("a"), FVar("b"), m])
    argc = FVar("argc", ident=31)
    main = _caller("main", 0xa00, 0xa10, 0x900, _var(FSSA(argc, 0)), params=[argc])
    seven = _caller("fixed_seven", 0xb00, 0xb10, 0x800, _const(7))
    sink.caller_sites = [FSite(middle, 0x910), FSite(seven, 0xb10)]
    middle.caller_sites = [FSite(main, 0xa10)]
    return sink, {**_NAMES, 0x800: "sink_copy", 0x900: "middle_copy"}


@pytest.mark.parametrize("depth,cut_in", [(0, "sink_copy"), (1, "middle_copy")])
def test_caller_depth_cut_is_structured_incompleteness_beside_a_resolved_sibling(depth, cut_in):
    sink, names = _depth_chain()
    result = _backward(sink, names, max_depth=depth)

    assert _gate(result) is False, result["diagnostics"]
    (leaf,) = _caller_leaves(result)
    assert leaf["function"]["name"] == cut_in
    assert leaf["reasons"] == {"max_depth": len(leaf["unfollowed"])}
    assert leaf["callers_followed"] == 0
    assert "--max-depth" in result["diagnostics"]["next_action"]
    if depth == 1:
        # the literal sibling was resolved; the cut is still disclosed
        assert {"kind": "constant", "value": 7} in _origins(result)


def test_larger_caller_depth_reaches_the_root_and_removes_the_boundary():
    sink, names = _depth_chain()
    result = _backward(sink, names, max_depth=8)

    assert _gate(result) is True, result["diagnostics"]
    assert _caller_leaves(result) == []
    rooted = [sl for sl in result["slices"] if sl["origin"].get("var") == "argc#0"]
    assert rooted and rooted[0]["crossed_functions"] == ["sink_copy", "middle_copy"]


# --------------------------------------------------------------------------
# BN6-C004: a recursive call remaps the demand to another formal
# --------------------------------------------------------------------------

def test_recursive_remap_to_another_formal_is_followed_to_the_external_caller():
    # recursive_copy(first, second) { memcpy(.., first); recursive_copy(second, first); }
    # start_recursive(external) { recursive_copy(getpid(), external); }
    # The self-call demands `second`, a formal the first visit never asked
    # about: it must reach `external`, not stop at `second` as solved.
    first, second = FVar("first", ident=50), FVar("second", ident=51)
    f0, s0 = FSSA(first, 0), FSSA(second, 0)
    rc = FFunc("recursive_copy", 0x800, FSSAFunc([
        _call(0, 0x804, _MEMCPY, [FExpr("MLIL_VAR_SSA", "g"), FExpr("MLIL_VAR_SSA", "s"), _var(f0)]),
        _call(1, 0x810, 0x800, [_var(s0), _var(f0)]),
    ]), params=[first, second])
    ext = FVar("external", ident=52)
    e0, g1 = FSSA(ext, 0), FSSA(FVar("pid", ident=53), 1)
    start = FFunc("start_recursive", 0x900, FSSAFunc([
        _call(0, 0x904, _GETPID, [], writes=[g1]),
        _call(1, 0x910, 0x800, [_var(g1), _var(e0)]),
    ]), params=[ext])
    rc.caller_sites = [FSite(rc, 0x810), FSite(start, 0x910)]
    result = _backward(rc, {**_NAMES, 0x800: "recursive_copy", 0x900: "start_recursive"})

    origins = _origins(result)
    assert {"kind": "parameter", "index": 0, "var": "external#0"} in origins, origins
    reached = [sl for sl in result["slices"] if sl["origin"].get("var") == "external#0"]
    assert reached[0]["crossed_functions"] == ["recursive_copy", "recursive_copy"]
    assert any(o.get("kind") == "call" and o.get("callee") == "getpid" for o in origins)
    # no formal is left as an unexpanded demand, and the self-call's back
    # edge to the demand already being solved (`first`) is not an origin
    assert not any(o.get("kind") == "parameter" and o.get("var") in ("second#0", "first#0")
                   for o in origins), origins


def _recursive_producer(update_callee):
    """``recursive_producer(n, rem) { malloc(n); recursive_producer(n + u(), rem); }``
    and ``start_producer() { recursive_producer(16, 1); }`` -- the back edge
    adds the return of ``u`` (``getpid``: unmodeled; ``recv``: a source)."""
    n, rem = FVar("n", ident=110), FVar("remaining", ident=111)
    n0, u1, t1 = FSSA(n, 0), FSSA(FVar("u", ident=112), 1), FSSA(FVar("t", ident=113), 1)
    rp = FFunc("recursive_producer", 0x800, FSSAFunc([
        _call(0, 0x804, _MALLOC, [_var(n0)]),
        _call(1, 0x808, update_callee, [], writes=[u1]),
        FInstr(2, 0x80c, "MLIL_SET_VAR_SSA", "t#1 = n#0 + u#1", reads=[n0, u1], writes=[t1],
               src=FExpr("MLIL_ADD", "n#0 + u#1", reads=[n0, u1])),
        _call(3, 0x810, 0x800, [_var(t1), FExpr("MLIL_VAR_SSA", "remaining#0")]),
    ]), params=[n, rem])
    start = FFunc("start_producer", 0x900, FSSAFunc([
        _call(0, 0x910, 0x800, [_const(16), _const(1)]),
    ]))
    rp.caller_sites = [FSite(rp, 0x810), FSite(start, 0x910)]
    return _backward(rp, {**_NAMES, 0x800: "recursive_producer", 0x900: "start_producer"},
                     "arg:malloc:0")


@pytest.mark.parametrize("callee,origin", [
    (_GETPID, {"kind": "call", "callee": "getpid"}),
    (_RECV, {"kind": "source", "callee": "recv"}),
], ids=["getpid", "recv-source"])
def test_recursive_back_edge_keeps_the_producer_it_introduces(callee, origin):
    # The self-call's back edge stops the ascent (its `n` is answered by the
    # enclosing frame) but the `u()` it adds on the way round is a root no
    # other path reaches: it keeps its own slice, with the definitions that
    # lead to it, beside the start_producer literal -- never a constant-only
    # slice certified complete.
    result = _recursive_producer(callee)

    origins = _origins(result)
    assert {"kind": "constant", "value": 16} in origins, origins
    assert origin in origins, origins
    produced = next(sl for sl in result["slices"] if sl["origin"] == origin)
    assert produced["crossed_functions"] == ["recursive_producer"]
    assert {"0x808", "0x80c", "0x810"} <= {st["address"] for st in produced["slice"]}
    assert not any(o.get("kind") == "parameter" for o in origins), origins



def _remap_pair(mutual, external_caller=True):
    """The reviewer's remap_cycle shape: ``remap_a(first, second) {
    malloc(first); remap_b(first, second); }``, ``remap_b(x, y) {
    remap_a(y, x + getpid()); }`` and ``start_remap(external) { remap_a(7,
    external); }`` -- or, ``mutual=False``, the same remap as a direct
    self-call ``direct_remap(second, first + getpid())``."""
    first, second = FVar("first", ident=120), FVar("second", ident=121)
    f0, s0 = FSSA(first, 0), FSSA(second, 0)
    a = FFunc("remap_a" if mutual else "direct_remap", 0x800, FSSAFunc([]), params=[first, second])
    names = {**_NAMES, 0x800: a.name}
    g1, t1 = FSSA(FVar("g", ident=122), 1), FSSA(FVar("t", ident=123), 1)

    def _remap_body(index0, x0, y0, back_to):
        return [_call(index0, 0x904 if mutual else 0x808, _GETPID, [], writes=[g1]),
                FInstr(index0 + 1, 0x908 if mutual else 0x80c, "MLIL_SET_VAR_SSA",
                       "t#1 = x + g#1", reads=[x0, g1], writes=[t1],
                       src=FExpr("MLIL_ADD", "x + g#1", reads=[x0, g1])),
                _call(index0 + 2, 0x910 if mutual else 0x810, back_to, [_var(y0), _var(t1)])]

    sites = []
    if mutual:
        x, y = FVar("x", ident=124), FVar("y", ident=125)
        b = FFunc("remap_b", 0x900, FSSAFunc(_remap_body(0, FSSA(x, 0), FSSA(y, 0), 0x800)),
                  params=[x, y])
        names[0x900] = "remap_b"
        a.mlil = FFunc("_", 0, FSSAFunc([
            _call(0, 0x804, _MALLOC, [_var(f0)]),
            _call(1, 0x810, 0x900, [_var(f0), _var(s0)]),
        ])).mlil
        b.caller_sites = [FSite(a, 0x810)]
        sites.append(FSite(b, 0x910))
    else:
        a.mlil = FFunc("_", 0, FSSAFunc(
            [_call(0, 0x804, _MALLOC, [_var(f0)])] + _remap_body(1, f0, s0, 0x800))).mlil
        sites.append(FSite(a, 0x810))
    if external_caller:
        ext = FVar("external", ident=126)
        start = FFunc("start_remap", 0xa00, FSSAFunc([
            _call(0, 0xa10, 0x800, [_const(7), _var(FSSA(ext, 0))]),
        ]), params=[ext])
        sites.append(FSite(start, 0xa10))
    a.caller_sites = sites
    return _backward(a, names, "arg:malloc:0")


@pytest.mark.parametrize("mutual", [True, False], ids=["mutual", "direct"])
def test_recursive_remap_keeps_the_back_edge_producer_not_an_internal_formal(mutual):
    # first <- (b) second <- (a) second <- (b) x + getpid() <- (a) first: the
    # walk closes the cycle on `first`, already being solved by the outer
    # frame. The literal 7, `external` and the getpid() the cycle adds are
    # the roots; no internal formal stands in for the getpid producer, in
    # mutual recursion exactly as in the direct self-call.
    result = _remap_pair(mutual)

    origins = _origins(result)
    assert {"kind": "constant", "value": 7} in origins, origins
    assert {"kind": "call", "callee": "getpid"} in origins, origins
    params = [o for o in origins if o["kind"] == "parameter"]
    assert params == [{"kind": "parameter", "index": 0, "var": "external#0"}], origins
    assert _gate(result) is True, result["diagnostics"]


@pytest.mark.parametrize("mutual", [True, False], ids=["mutual", "direct"])
def test_pure_recursive_cycle_keeps_its_own_formal_as_the_origin(mutual):
    # Control: no caller outside the cycle. What closes on the sink's own
    # demand is answered by nothing else, so its formal stays the origin
    # (beside the producer the cycle adds) -- never an empty slice list.
    result = _remap_pair(mutual, external_caller=False)

    origins = _origins(result)
    assert {"kind": "parameter", "index": 0, "var": "first#0"} in origins, origins
    assert {"kind": "call", "callee": "getpid"} in origins, origins
    assert not any(o.get("var") in ("x#0", "y#0", "second#0") for o in origins), origins


# --------------------------------------------------------------------------
# BN6-C005: the engine's def-chain step budget cuts one operand
# --------------------------------------------------------------------------

def _long_chain(links):
    """``long_chain(root, other) { v0 = other; v1 = v0 + 1; ...; malloc(root + vN); }``"""
    root, other = FVar("root", ident=60), FVar("other", ident=61)
    r0, o0 = FSSA(root, 0), FSSA(other, 0)
    instrs, prev = [], o0
    for i in range(links):
        v = FSSA(FVar(f"v{i}", ident=100 + i), 1)
        src = FExpr("MLIL_ADD", f"{prev.var.name} + {i}", reads=[prev])
        instrs.append(FInstr(i, 0x10 + 4 * i, "MLIL_SET_VAR_SSA", f"v{i}#1 = ...",
                             reads=[prev], writes=[v], src=src))
        prev = v
    size = FSSA(FVar("size", ident=99), 1)
    k = len(instrs)
    instrs.append(FInstr(k, 0x10 + 4 * k, "MLIL_SET_VAR_SSA", "size#1 = root#0 + v",
                         reads=[r0, prev], writes=[size],
                         src=FExpr("MLIL_ADD", "root#0 + v", reads=[r0, prev])))
    instrs.append(_call(k + 1, 0x10 + 4 * (k + 1), _MALLOC, [_var(size)]))
    return FFunc("long_chain", 0x400, FSSAFunc(instrs), params=[root, other])


def test_def_chain_budget_cut_beside_a_resolved_formal_is_not_complete():
    result = _backward(_long_chain(12), _NAMES, "arg:malloc:0", engine_depth=6)

    assert _gate(result) is False, result["diagnostics"]
    cuts = [o for o in _origins(result) if o.get("cause") == "def_chain_budget"]
    assert cuts and cuts[0]["kind"] == "unresolved" and cuts[0]["budget"] == 6, _origins(result)
    assert {"kind": "parameter", "index": 0, "var": "root#0"} in _origins(result)


def test_larger_def_chain_budget_finds_the_cut_root_and_drops_the_boundary():
    result = _backward(_long_chain(12), _NAMES, "arg:malloc:0", engine_depth=64)

    assert _gate(result) is True, result["diagnostics"]
    assert not any(o.get("cause") == "def_chain_budget" for o in _origins(result))
    # every formal demand is an origin, not just the representative one
    params = {o.get("var") for o in _origins(result) if o.get("kind") == "parameter"}
    assert params == {"root#0", "other#0"}


# --------------------------------------------------------------------------
# BN6-C012: a modeled source beside a formal must not suppress the ascent
# --------------------------------------------------------------------------

@pytest.mark.parametrize("formal_first", [True, False])
def test_source_plus_formal_keeps_both_roots_in_either_operand_order(formal_first):
    # mixed_origin(upstream, fd) { r = recv(fd, ..); malloc(upstream + r); }
    # start_mixed(external) { mixed_origin(external, 0); }
    up, fd = FVar("upstream", ident=70), FVar("fd", ident=71)
    u0, r1, size = FSSA(up, 0), FSSA(FVar("r", ident=72), 1), FSSA(FVar("size", ident=73), 1)
    reads = [u0, r1] if formal_first else [r1, u0]
    mixed = FFunc("mixed_origin", 0x800, FSSAFunc([
        _call(0, 0x804, _RECV, [FExpr("MLIL_VAR_SSA", "fd#0"), FExpr("MLIL_VAR_SSA", "buf"),
                                _const(8), _const(0)], writes=[r1]),
        FInstr(1, 0x808, "MLIL_SET_VAR_SSA", "size#1 = upstream#0 + r#1", reads=reads,
               writes=[size], src=FExpr("MLIL_ADD", "upstream#0 + r#1", reads=reads)),
        _call(2, 0x80c, _MALLOC, [_var(size)]),
    ]), params=[up, fd])
    ext = FVar("external", ident=74)
    start = FFunc("start_mixed", 0x900, FSSAFunc([
        _call(0, 0x910, 0x800, [_var(FSSA(ext, 0)), _const(0)]),
    ]), params=[ext])
    mixed.caller_sites = [FSite(start, 0x910)]
    result = _backward(mixed, {**_NAMES, 0x800: "mixed_origin"}, "arg:malloc:0")

    origins = _origins(result)
    assert {"kind": "source", "callee": "recv"} in origins, origins
    up_root = [sl for sl in result["slices"] if sl["origin"].get("var") == "external#0"]
    assert up_root and up_root[0]["crossed_functions"] == ["mixed_origin"], origins
    assert _gate(result) is True, result["diagnostics"]


# --------------------------------------------------------------------------
# BN6-C005 (load traversal): the same budget, cut inside the memory walk
# --------------------------------------------------------------------------


def _load_chain(budget):
    """``f(m, n, src)``: ``m->c = 5; m->b = m->c; m->a = m->b;
    memcpy(.., m->a + src)`` -- each stored value is itself a load, one
    traversal step deeper than the last; the formal ``src`` is a sibling
    resolved at the first step."""
    p = _FieldProgram()
    p.store(p.addr(p.m0, 24), 4, value=_const(5))
    p.store(p.addr(p.m0, 16), 4, value=p.load(p.addr(p.m0, 24), 4))
    p.store(p.addr(p.m0, 8), 4, value=p.load(p.addr(p.m0, 16), 4))
    loaded = p.load(p.addr(p.m0, 8), 4)
    p.sink(FExpr("MLIL_ADD", "[m + 8].d + src#0", reads=list(loaded.vars_read) + [p.src0],
                 left=loaded, right=FExpr("MLIL_VAR_SSA", "src#0", reads=[p.src0])))
    func = FFunc("f", 0x10, FSSAFunc(p.instrs, mem_defs=p.mem_defs), params=[p.m, p.n, p.src])
    return te.TaintEngine(FBV({0xa20: "memcpy", 0xa10: "read_u32"}), te.load_models(),
                          max_depth=budget).backward(func, [te.parse_locator("arg:memcpy:2")])


def test_load_traversal_budget_cut_beside_a_resolved_formal_is_not_complete():
    result = _load_chain(1)

    assert any("def-chain walk truncated" in a for a in result["assumptions"]), result
    assert _gate(result) is False, result["diagnostics"]
    assert any(o.get("cause") == "def_chain_budget" for o in _origins(result)), _origins(result)
    assert {"kind": "parameter", "index": 2, "var": "src#0"} in _origins(result)


def test_larger_load_traversal_budget_walks_to_the_stored_literal_and_drops_the_boundary():
    result = _load_chain(64)

    assert not any("def-chain walk truncated" in a for a in result["assumptions"]), result
    assert not any(o.get("cause") == "def_chain_budget" for o in _origins(result))
    texts = [st.get("il_text", "") for sl in result["slices"] for st in sl["slice"]]
    assert any("0x18] = 0x5" in t for t in texts), texts
    assert _gate(result) is True, result["diagnostics"]


# --------------------------------------------------------------------------
# review round 1: a cut must be a value NO path walked, and each caller's
# cut stays its own row
# --------------------------------------------------------------------------

def _shared_dag(formal_first, links=8):
    """``resolved_dag(root) { c = root; a0 = c + 0; ...; malloc(c + a_last); }``
    -- the chain reaches the shared copy ``c`` one step past an engine budget
    of ``links + 1``, the other operand reaches it at the second step."""
    root = FVar("root", ident=80)
    r0, c1 = FSSA(root, 0), FSSA(FVar("c", ident=81), 1)
    instrs = [FInstr(0, 0x10, "MLIL_SET_VAR_SSA", "c#1 = root#0", reads=[r0], writes=[c1],
                     src=_var(r0))]
    prev = c1
    for i in range(links):
        v = FSSA(FVar(f"a{i}", ident=90 + i), 1)
        instrs.append(FInstr(i + 1, 0x14 + 4 * i, "MLIL_SET_VAR_SSA", f"a{i}#1 = ...",
                             reads=[prev], writes=[v],
                             src=FExpr("MLIL_ADD", f"{prev.var.name} + {i}", reads=[prev])))
        prev = v
    size = FSSA(FVar("size", ident=99), 1)
    reads = [c1, prev] if formal_first else [prev, c1]
    k = len(instrs)
    instrs.append(FInstr(k, 0x80, "MLIL_SET_VAR_SSA", "size#1 = c#1 + a", reads=reads,
                         writes=[size], src=FExpr("MLIL_ADD", "c#1 + a", reads=reads)))
    instrs.append(_call(k + 1, 0x84, _MALLOC, [_var(size)]))
    return FFunc("resolved_dag", 0x400, FSSAFunc(instrs), params=[root])


@pytest.mark.parametrize("formal_first", [True, False])
@pytest.mark.parametrize("links", [8, 9])
def test_value_cut_on_one_path_but_walked_on_another_is_no_budget_cut(formal_first, links):
    # The chain reaches past the budget either at the shared copy `c` itself
    # (9 links) or at `root` behind it (8 links); the other operand walks `c`
    # and `root` within it, whichever operand the walk takes first -- also
    # when the chain walked `c` first, deeper, and cut `root` beneath it.
    # Every dependency was followed: complete, with no cut origin or prose.
    result = _backward(_shared_dag(formal_first, links), _NAMES, "arg:malloc:0",
                       engine_depth=9)

    assert _origins(result) == [{"kind": "parameter", "index": 0, "var": "root#0"}]
    assert not any("def-chain walk truncated" in a for a in result["assumptions"]), result
    assert _gate(result) is True, result["diagnostics"]


@pytest.mark.parametrize("formal_first", [True, False])
def test_value_cut_on_every_path_stays_a_budget_cut(formal_first):
    # Control: with a smaller budget the chain is cut at a link only the
    # chain reaches -- that cut stands, beside the resolved formal.
    result = _backward(_shared_dag(formal_first), _NAMES, "arg:malloc:0", engine_depth=5)

    cuts = [o for o in _origins(result) if o.get("cause") == "def_chain_budget"]
    assert [c["unwalked"] for c in cuts] == [["a2#1"]], _origins(result)
    assert {"kind": "parameter", "index": 0, "var": "root#0"} in _origins(result)
    assert _gate(result) is False, result["diagnostics"]


def _chain_caller(name, start, call_addr, ident, links):
    """``name(x) { v0 = x + 0; ...; accept_size(v_last); }`` -- the chain's
    variables are named alike in every caller."""
    x = FVar("x", ident=ident)
    prev, instrs = FSSA(x, 0), []
    for i in range(links):
        v = FSSA(FVar(f"v{i}", ident=ident + 1 + i), 1)
        instrs.append(FInstr(i, start + 4 + 4 * i, "MLIL_SET_VAR_SSA", f"v{i}#1 = ...",
                             reads=[prev], writes=[v],
                             src=FExpr("MLIL_ADD", f"{prev.var.name} + {i}", reads=[prev])))
        prev = v
    instrs.append(_call(links, call_addr, 0x800, [_var(prev)]))
    return FFunc(name, start, FSSAFunc(instrs), params=[x])


def test_each_callers_budget_cut_is_its_own_slice():
    # accept_size(n) { malloc(n); } with two callers whose chains are cut by
    # the same engine budget at the same variable NAME: two frontiers in two
    # functions, never one row "reached via 2 call sites".
    accept = FFunc("accept_size", 0x800, FSSAFunc([
        _call(0, 0x804, _MALLOC, [_var(FSSA(FVar("n", ident=70), 0))]),
    ]), params=[FVar("n", ident=70)])
    first = _chain_caller("first_chain", 0x3000, 0x3080, 200, 8)
    second = _chain_caller("second_chain", 0x3100, 0x3180, 300, 8)
    accept.caller_sites = [FSite(first, 0x3080), FSite(second, 0x3180)]
    result = _backward(accept, {**_NAMES, 0x800: "accept_size"}, "arg:malloc:0",
                       engine_depth=4)

    cuts = [sl for sl in result["slices"] if sl["origin"].get("cause") == "def_chain_budget"]
    assert sorted(sl["origin"]["function"] for sl in cuts) == ["0x3000", "0x3100"], cuts
    assert all(sl["origin"]["unwalked"] == ["v2#1"] for sl in cuts), cuts
    assert all("reached_via_call_sites" not in sl for sl in cuts), cuts
    assert _gate(result) is False, result["diagnostics"]


def _shared_load(formal_first, budget, links=6):
    """``f(m, n, src) { m->len = src + 1; c = m->len; a0 = c + 0; ...;
    memcpy(.., a_last + c); }`` -- the chain enters the load of ``m->len``
    one step within an engine budget of ``links + 1`` and reaches ``src``
    beneath it past the budget; the other operand enters the same load at
    the first step."""
    p = _FieldProgram()
    p.store(p.addr(p.m0, 8), 8,
            value=FExpr("MLIL_ADD", "src#0 + 1", reads=[p.src0],
                        left=FExpr("MLIL_VAR_SSA", "src#0", reads=[p.src0]), right=_const(1)))
    c1 = FSSA(FVar("c", ident=40), 1)
    loaded = p.load(p.addr(p.m0, 8), 8)
    i, a = p._at()
    p.instrs.append(FInstr(i, a, "MLIL_SET_VAR_SSA", "c#1 = [m + 8].q", reads=[p.m0],
                           writes=[c1], src=loaded))
    prev = c1
    for k in range(links):
        v = FSSA(FVar(f"a{k}", ident=41 + k), 1)
        i, a = p._at()
        p.instrs.append(FInstr(i, a, "MLIL_SET_VAR_SSA", f"a{k}#1 = ...", reads=[prev],
                               writes=[v], src=FExpr("MLIL_ADD", f"{prev.var.name} + {k}",
                                                     reads=[prev])))
        prev = v
    reads = [c1, prev] if formal_first else [prev, c1]
    p.sink(FExpr("MLIL_ADD", "c#1 + a", reads=reads))
    func = FFunc("f", 0x10, FSSAFunc(p.instrs, mem_defs=p.mem_defs), params=[p.m, p.n, p.src])
    return te.TaintEngine(FBV({0xa20: "memcpy", 0xa10: "read_u32"}), te.load_models(),
                          max_depth=budget).backward(func, [te.parse_locator("arg:memcpy:2")])


@pytest.mark.parametrize("formal_first", [True, False])
@pytest.mark.parametrize("budget", [7, 8, 64])
def test_shared_load_entered_deep_then_shallow_keeps_its_root(formal_first, budget):
    # The chain enters the load first (when it is the first operand) and the
    # budget cuts `src` beneath it; the shallow operand reaches the same load
    # within budget. The root must survive -- not a lone cut -- in either
    # operand order and at every budget that reaches the load at all.
    result = _shared_load(formal_first, budget)

    assert {"kind": "parameter", "index": 2, "var": "src#0"} in _origins(result), _origins(result)
    assert not any(o.get("cause") == "def_chain_budget" for o in _origins(result)), _origins(result)
    assert _gate(result) is True, result["diagnostics"]


@pytest.mark.parametrize("formal_first", [True, False])
def test_shared_load_cut_on_every_path_stays_a_budget_cut(formal_first):
    # Control: a budget the shallow path cannot reach `src` within either
    # leaves the cut standing, and no root is invented.
    result = _shared_load(formal_first, 1)

    assert any(o.get("cause") == "def_chain_budget" for o in _origins(result)), _origins(result)
    assert _gate(result) is False, result["diagnostics"]


# --------------------------------------------------------------------------
# the diagnostics vocabulary: per-reason remediation, one leaf kind
# --------------------------------------------------------------------------

def test_next_action_names_each_unfollowed_reason_and_keeps_the_cap_advice():
    from bn_agent_bridge.taint_result import backward_diagnostics

    leaf = {"kind": "caller_sites_truncated",
            "reasons": {"caller_cap": 4, "max_depth": 1, "no_call_instruction": 1}}
    diag = backward_diagnostics([leaf], [], sinks_seeded=1, slices=1)
    nxt = diag["next_action"]
    assert "caller ascent was capped" in nxt
    assert "--max-depth" in nxt
    assert "bn callsites" in nxt
    assert diag["frontier"]["dropped_callers"] == 1
    assert diag["safe_to_report_complete_slice"] is False


# --------------------------------------------------------------------------
# real Binary Ninja: the same shapes on compiled public synthetic programs
# --------------------------------------------------------------------------

_ENTRY_CALLBACK_C = r"""
#include <stddef.h>
#include <string.h>
static char out[128];
typedef void (*handler)(size_t);
__attribute__((noinline)) void copy_handler(size_t n) { memcpy(out, "public", n); }
__attribute__((noinline)) void dispatch(handler h, size_t n) { h(n); }
__attribute__((noinline,noreturn)) void entry_point(void) {
    dispatch(copy_handler, 17);
    for (;;) __asm__ volatile ("pause");
}
"""

_MAIN_CALLBACK_C = r"""
#include <stddef.h>
#include <string.h>
static char out[128];
typedef int (*handler)(int, char **);
__attribute__((noinline)) int main(int n, char **args) {
    (void)args;
    memcpy(out, "public", (size_t)n);
    return out[0];
}
__attribute__((noinline)) int dispatch(handler h, int n) { return h(n, 0); }
__attribute__((noinline,noreturn)) void entry_point(void) {
    (void)dispatch(main, 17);
    for (;;) __asm__ volatile ("pause");
}
"""

_STARTUP_C = r"""
#include <stddef.h>
#include <string.h>
typedef void (*handler)(size_t);
static char out[128];
__attribute__((noinline)) void put_len(size_t n) { memcpy(out, "public", n); }
__attribute__((noinline)) void copy_handler(size_t n) { memcpy(out, "public", n); }
__attribute__((noinline)) void dispatch(handler h, size_t n) { h(n); }
int main(int argc, char **argv) {
    (void)argv;
    put_len((size_t)argc);
    dispatch(copy_handler, (size_t)argc);
    return out[0];
}
"""

_BUDGET_C = r"""
#include <stddef.h>
#include <stdlib.h>
size_t observed;
#define CHAIN8(x) x=x+1; x=x+2; x=x+3; x=x+4; x=x+5; x=x+6; x=x+7; x=x+8;
#define CHAIN64(x) CHAIN8(x) CHAIN8(x) CHAIN8(x) CHAIN8(x) CHAIN8(x) CHAIN8(x) CHAIN8(x) CHAIN8(x)
#define LINKS(c) size_t a0=c+1; size_t a1=a0+2; size_t a2=a1+3; size_t a3=a2+4; \
    size_t a4=a3+5; size_t a5=a4+6; size_t a6=a5+7; size_t a7=a6+8; size_t a8=a7+9; \
    size_t a9=a8+10; size_t a10=a9+11; size_t a11=a10+12; size_t a12=a11+13; \
    size_t a13=a12+14; size_t a14=a13+15; size_t a15=a14+16; size_t a16=a15+17; \
    size_t a17=a16+18; size_t a18=a17+19; size_t a19=a18+20;
__attribute__((noinline)) void accept_size(size_t n) { observed = (size_t)malloc(n); }
__attribute__((noinline)) void first_chain(size_t alpha) { CHAIN64(alpha) CHAIN64(alpha) accept_size(alpha); }
__attribute__((noinline)) void second_chain(size_t beta) { CHAIN64(beta) CHAIN64(beta) CHAIN8(beta) accept_size(beta); }
__attribute__((noinline)) void *dag(size_t root) { size_t c = root; LINKS(c) return malloc(a19 + c); }
__attribute__((noinline)) void *dag_reverse(size_t root) { size_t c = root; LINKS(c) return malloc(c + a19); }
__attribute__((noinline)) void *shared_load(size_t *m, size_t root) { m[0] = root + 1; size_t c = m[0]; LINKS(c) return malloc(a19 + c); }
__attribute__((noinline)) void *shared_load_reverse(size_t *m, size_t root) { m[0] = root + 1; size_t c = m[0]; LINKS(c) return malloc(c + a19); }
"""

_RECURSIVE_C = r"""
#include <stddef.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>
size_t observed;
__attribute__((noinline)) void pid_producer(size_t n, unsigned remaining) {
    observed = (size_t)malloc(n);
    if (remaining) pid_producer(n + (size_t)getpid(), remaining - 1);
}
__attribute__((noinline)) void start_pid(void) { pid_producer(16, 1); }
__attribute__((noinline)) void recv_producer(size_t n, unsigned remaining) {
    char bytes[8];
    observed = (size_t)malloc(n);
    if (remaining) recv_producer(n + (size_t)recv(0, bytes, sizeof(bytes), 0), remaining - 1);
}
__attribute__((noinline)) void start_recv(void) { recv_producer(16, 1); }
__attribute__((noinline)) void remap_copy(size_t first, size_t second, unsigned remaining) {
    observed = (size_t)malloc(first);
    if (remaining) remap_copy(second, first, remaining - 1);
}
__attribute__((noinline)) void start_remap(size_t external) { remap_copy(7, external, 1); }
"""

# The reviewer's remap_cycle fixture, verbatim, plus the same remaps with no
# caller outside the cycle (`pure_*`).
_REMAP_CYCLE_C = r"""
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
volatile size_t observed;
#define NO __attribute__((noinline))
NO void remap_b(size_t, size_t, unsigned);
NO void remap_a(size_t first, size_t second, unsigned r) {
    observed = (size_t)malloc(first);
    if (r) remap_b(first, second, r - 1);
}
NO void remap_b(size_t first, size_t second, unsigned r) {
    if (r) remap_a(second, first + (size_t)getpid(), r - 1);
}
NO void start_remap(size_t external) { remap_a(7, external, 5); }
NO void direct_remap(size_t first, size_t second, unsigned r) {
    observed = (size_t)malloc(first);
    if (r) direct_remap(second, first + (size_t)getpid(), r - 1);
}
NO void start_direct(size_t external) { direct_remap(7, external, 5); }
NO void pure_b(size_t, size_t, unsigned);
NO void pure_a(size_t first, size_t second, unsigned r) {
    observed = (size_t)malloc(first);
    if (r) pure_b(first, second, r - 1);
}
NO void pure_b(size_t first, size_t second, unsigned r) {
    if (r) pure_a(second, first + (size_t)getpid(), r - 1);
}
"""


def _compile(tmp_path: Path, name: str, source: str, flags: list[str]) -> Path:
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        raise FixtureBuildError("no C compiler; the real-BN coverage lane cannot build its programs")
    src = tmp_path / f"{name}.c"
    src.write_text(source)
    out = tmp_path / name
    built = subprocess.run([cc, "-O0", "-g", "-fno-builtin", *flags, str(src), "-o", str(out)],
                           capture_output=True, text=True, timeout=120.0)
    if built.returncode:
        raise FixtureBuildError(f"{name} failed to build: {built.stderr}")
    return out


def _bn_backward(bridge, target: str, fn: str, sink: str, *extra: str):
    return bridge.json("-t", target, "taint", "backward", "-f", fn, "--sink", sink, *extra,
                       "--verbose", timeout=180.0)


@pytest.mark.real_bn
def test_real_bn_custom_entry_callback_is_an_indirect_caller(tmp_path, shared_bn):
    # `-nostartfiles -e entry_point`: the entry function hands copy_handler to
    # dispatch. That is a callback, not the runtime's handoff of `main`.
    binary = _compile(tmp_path, "entry_callback", _ENTRY_CALLBACK_C,
                      ["-no-pie", "-nostartfiles", "-Wl,-e,entry_point"])
    target = shared_bn.load(binary, copy=False)
    result = _bn_backward(shared_bn, target, "copy_handler", "arg:memcpy:2")

    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result["diagnostics"]
    (leaf,) = _caller_leaves(result)
    assert leaf["reasons"] == {"indirect_reference": len(leaf["unfollowed"])}
    assert {r["caller"] for r in leaf["unfollowed"]} == {"entry_point"}


@pytest.mark.real_bn
def test_real_bn_custom_entry_dispatching_main_is_an_indirect_caller(tmp_path, shared_bn):
    # A callback NAMED main that the custom entry hands to its own
    # dispatcher: no C-runtime handoff, so the name earns no exemption.
    binary = _compile(tmp_path, "main_callback", _MAIN_CALLBACK_C,
                      ["-no-pie", "-nostartfiles", "-Wl,-e,entry_point"])
    target = shared_bn.load(binary, copy=False)
    result = _bn_backward(shared_bn, target, "main", "arg:memcpy:2")

    assert result["diagnostics"]["safe_to_report_complete_slice"] is False, result["diagnostics"]
    (leaf,) = _caller_leaves(result)
    assert leaf["reasons"] == {"indirect_reference": len(leaf["unfollowed"])}
    assert {r["caller"] for r in leaf["unfollowed"]} == {"entry_point"}


@pytest.mark.real_bn
def test_real_bn_main_startup_stays_an_origin_beside_a_real_callback(tmp_path, shared_bn):
    # Normal `_start -> __libc_start_main(main)` startup: main's argc is the
    # origin of a complete slice. In the same program a handler main hands to
    # dispatch is an indirect callee.
    binary = _compile(tmp_path, "startup", _STARTUP_C, ["-no-pie"])
    target = shared_bn.load(binary, copy=False)

    direct = _bn_backward(shared_bn, target, "put_len", "arg:memcpy:2")
    assert direct["diagnostics"]["safe_to_report_complete_slice"] is True, direct["diagnostics"]
    assert _caller_leaves(direct) == []
    assert [(sl["origin"]["kind"], sl["origin"].get("index"), sl["crossed_functions"])
            for sl in direct["slices"]] == [("parameter", 0, ["put_len"])]

    callback = _bn_backward(shared_bn, target, "copy_handler", "arg:memcpy:2")
    assert callback["diagnostics"]["safe_to_report_complete_slice"] is False
    (leaf,) = _caller_leaves(callback)
    assert {r["caller"] for r in leaf["unfollowed"]} == {"main"}


@pytest.mark.real_bn
def test_real_bn_budget_cuts_are_per_path_and_per_caller(tmp_path, shared_bn):
    binary = _compile(tmp_path, "budget.so", _BUDGET_C, ["-shared", "-fPIC", "-Wl,-Bsymbolic"])
    target = shared_bn.load(binary, copy=False)

    # A shared value / shared load the long operand reaches past the budget
    # but the short one walks: complete, the formal root, no cut -- in either
    # operand order.
    for fn, root_index in [("dag", 0), ("dag_reverse", 0),
                           ("shared_load", 1), ("shared_load_reverse", 1)]:
        result = _bn_backward(shared_bn, target, fn, "arg:malloc:0")
        origins = [sl["origin"] for sl in result["slices"]]
        assert [(o["kind"], o.get("index")) for o in origins] == [("parameter", root_index)], (fn, origins)
        assert result["diagnostics"]["safe_to_report_complete_slice"] is True, (fn, result["diagnostics"])

    # Two callers whose chains the budget really cuts: two frontiers, one per caller.
    result = _bn_backward(shared_bn, target, "accept_size", "arg:malloc:0")
    cuts = [sl["origin"] for sl in result["slices"]
            if sl["origin"].get("cause") == "def_chain_budget"]
    assert len({c["function"] for c in cuts}) == 2, cuts
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


@pytest.mark.real_bn
def test_real_bn_recursive_back_edge_keeps_its_producers(tmp_path, shared_bn):
    # `f(n + getpid())` around a self-call: the literal from the outside
    # caller AND the getpid() the back edge adds -- in JSON and in text. A
    # modeled source on the back edge (`recv`) is the control, and a pure
    # remap back edge still adds no pseudo-origin.
    binary = _compile(tmp_path, "recursive.so", _RECURSIVE_C,
                      ["-fno-optimize-sibling-calls", "-shared", "-fPIC", "-Wl,-Bsymbolic"])
    target = shared_bn.load(binary, copy=False)

    for fn, root in [("pid_producer", {"kind": "call", "callee": "getpid"}),
                     ("recv_producer", {"kind": "source", "callee": "recv"})]:
        result = _bn_backward(shared_bn, target, fn, "arg:malloc:0")
        origins = [sl["origin"] for sl in result["slices"]]
        assert {"kind": "constant", "value": 16} in origins, (fn, origins)
        assert root in origins, (fn, origins)
        assert not any(o.get("kind") == "parameter" for o in origins), (fn, origins)

    text = shared_bn.run("-t", target, "taint", "backward", "-f", "pid_producer",
                         "--sink", "arg:malloc:0", timeout=180.0)
    assert text.returncode == 0, text.stderr
    assert "getpid" in text.stdout and "constant" in text.stdout, text.stdout

    remap = _bn_backward(shared_bn, target, "remap_copy", "arg:malloc:0")
    origins = [(sl["origin"].get("kind"), sl["origin"].get("value"), sl["crossed_functions"])
               for sl in remap["slices"]]
    assert ("constant", 7, ["remap_copy"]) in origins, origins
    assert any(k == "parameter" and crossed == ["remap_copy", "remap_copy"]
               for k, _, crossed in origins), origins
    assert sum(1 for k, _, _ in origins if k == "parameter") == 1, origins


@pytest.mark.real_bn
def test_real_bn_mutual_recursion_keeps_the_back_edge_producer(tmp_path, shared_bn):
    # remap_a -> remap_b -> remap_a(second, first + getpid()): the literal 7,
    # `external` and getpid() are the roots, in mutual recursion exactly as
    # in the equivalent direct self-call -- no internal formal stands in for
    # getpid. With no caller outside the cycle, the sink's own formal is the
    # cycle's root beside getpid.
    binary = _compile(tmp_path, "remap_cycle.so", _REMAP_CYCLE_C,
                      ["-fno-optimize-sibling-calls", "-shared", "-fPIC", "-Wl,-Bsymbolic"])
    target = shared_bn.load(binary, copy=False)
    getpid = {"kind": "call", "callee": "getpid"}

    for fn in ("remap_a", "direct_remap"):
        result = _bn_backward(shared_bn, target, fn, "arg:malloc:0")
        origins = [sl["origin"] for sl in result["slices"]]
        assert {"kind": "constant", "value": 7} in origins, (fn, origins)
        assert getpid in origins, (fn, origins)
        params = [o for o in origins if o["kind"] == "parameter"]
        assert len(params) == 1 and params[0]["index"] == 0, (fn, origins)
        crossed = {tuple(sl["crossed_functions"]) for sl in result["slices"]
                   if sl["origin"]["kind"] == "parameter"}
        assert all(c and c[-1] == fn for c in crossed), (fn, crossed)  # start_*'s formal
        assert result["diagnostics"]["safe_to_report_complete_slice"] is True

    pure = _bn_backward(shared_bn, target, "pure_a", "arg:malloc:0")
    origins = [(sl["origin"].get("kind"), sl["origin"].get("callee"), sl["crossed_functions"])
               for sl in pure["slices"]]
    assert any(k == "call" and callee == "getpid" for k, callee, _ in origins), origins
    # the sink's own formal, not a pure_b one: the cycle's root
    assert [c for k, _, c in origins if k == "parameter"] == [[]], origins

    text = shared_bn.run("-t", target, "taint", "backward", "-f", "remap_a",
                         "--sink", "arg:malloc:0", timeout=180.0)
    assert text.returncode == 0, text.stderr
    assert "getpid" in text.stdout, text.stdout
