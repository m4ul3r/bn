"""Backward caller/demand coverage (IC01): a backward slice that bottoms out at a
formal parameter accounts for EVERY formal demand at EVERY known caller -- a
site it could not follow, a function reached only through a stored address, a
caller the configured ``--max-depth`` cut off, a demand a recursive call remaps
to another formal, and a def chain the engine's step budget cut are each
structured incompleteness, whichever origin the slice happens to name as its
representative. A chain that WAS walked to its origins (a caller's literal, a
formal with no known caller) stays a complete slice."""
from __future__ import annotations

import pytest
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


def test_process_entry_handoff_and_direct_call_refs_are_not_indirect_callers():
    # Controls: `_start` handing `main` to the C runtime is no in-binary
    # caller (its formals are what the loader supplied), and the code
    # reference a direct call site makes is that site, already followed.
    start = FFunc("_start", 0x400, FSSAFunc([]))
    handoff = _backward(_copy_handler(), {**_NAMES, 0x800: "copy_handler"},
                        code_refs={0x800: [(0x41e, start)]}, entry=start)
    assert _gate(handoff) is True, handoff["diagnostics"]
    assert _caller_leaves(handoff) == []

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
