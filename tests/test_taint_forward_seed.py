"""Forward seeding honesty and the advice a zero-sink forward run prints.

#1041 item 1: a `call:` preset whose modeled `*arg:N` output cannot be keyed
must disclose it the way an `arg:` seed does, and neither may send the agent to
the other locator, which keys the buffer through the same pointer.
#1033 item 6: every command the forward advice prints in backticks runs as
printed or is marked as a template.
"""
from __future__ import annotations

import re
import shlex

import pytest
from test_taint_engine import (
    FBV,
    FPVS,
    FSSA,
    FBlock,
    FBVStr,
    FExpr,
    FFunc,
    FInstr,
    FSSAFunc,
    FVar,
    _addr_of,
    _cstr,
    _ext_call,
    _FEdge,
    _slot_recv_func,
    te,
)

from bn_agent_bridge import il_format
from bn_agent_bridge.read_taint_models import build_catalog
from bn_agent_bridge.taint_models import _validate_model_interior


@pytest.fixture
def models():
    return te.load_models()


def _var(ssa, text=None):
    return FExpr("MLIL_VAR_SSA", text or f"{ssa.var.name}#{ssa.version}", reads=[ssa])


def _const(value):
    return FExpr("MLIL_CONST", hex(value), constant=value)


def _field_load(base, off):
    addr = FExpr("MLIL_ADD", f"c#0 + {hex(off)}", left=_var(base), right=_const(off))
    return FExpr("MLIL_LOAD_SSA", f"[c#0 + {hex(off)}]", reads=[base], src=addr)


def _member_buffer_pump(*, dest="field_plus_offset"):
    """`recv(c->fd, c->buf + have, ...)`, then a length re-read through a
    fresh load of `c->buf` sizes a memcpy (the #1041 fixture's `pump`)."""
    c, fd, buf, p, n, again, size = (FVar(s) for s in ("c", "fd", "buf", "p", "n", "again", "size"))
    have = FVar("have")
    c0, fd0, have0 = FSSA(c, 0), FSSA(fd, 0), FSSA(have, 0)
    buf1, p1, n1, again1, size1 = FSSA(buf, 1), FSSA(p, 1), FSSA(n, 1), FSSA(again, 1), FSSA(size, 1)
    if dest == "null":
        pointer = _const(0)
        setup = []
    else:
        pointer = _var(p1)
        setup = [
            FInstr(0, 0x1000, "MLIL_SET_VAR_SSA", "buf#1 = [c#0 + 8]", reads=[c0],
                   writes=[buf1], src=_field_load(c0, 8)),
            FInstr(1, 0x1004, "MLIL_SET_VAR_SSA", "p#1 = buf#1 + have#0", reads=[buf1, have0],
                   writes=[p1], src=FExpr("MLIL_ADD", "buf#1 + have#0", reads=[buf1, have0],
                                          left=_var(buf1), right=_var(have0))),
        ]
    instrs = [
        *setup,
        FInstr(2, 0x1008, "MLIL_CALL_SSA", "n#1 = recv(fd#0, p, 0x100, 0)", writes=[n1],
               dest=FExpr("MLIL_CONST_PTR", "0x900", constant=0x900),
               params=[_var(fd0), pointer, _const(0x100), _const(0)]),
        FInstr(3, 0x100c, "MLIL_SET_VAR_SSA", "again#1 = [c#0 + 8]", reads=[c0],
               writes=[again1], src=_field_load(c0, 8)),
        FInstr(4, 0x1010, "MLIL_SET_VAR_SSA", "size#1 = [again#1 + 4]", reads=[again1],
               writes=[size1], src=FExpr("MLIL_LOAD_SSA", "[again#1 + 4]", reads=[again1])),
        FInstr(5, 0x1014, "MLIL_CALL_SSA", "memcpy(0x6000, 0x7000, size#1)", reads=[size1],
               dest=FExpr("MLIL_CONST_PTR", "0x901", constant=0x901),
               params=[_const(0x6000), _const(0x7000), _var(size1)]),
    ]
    return FFunc("pump", 0x1000, FSSAFunc(instrs), params=[c, fd, have])


def _engine(models):
    return te.TaintEngine(FBV({0x900: "recv", 0x901: "memcpy"}), models)


def _misanchored(result):
    return [lf for lf in result["leaves"] if lf.get("kind") == "source_seed_misanchored"]


def test_call_preset_discloses_an_unkeyed_member_buffer_1041(models):
    result = _engine(models).forward(_member_buffer_pump(), [te.parse_locator("call:recv")])

    assert result["reached_sinks"] == []
    [leaf] = _misanchored(result)
    assert (leaf["seed"], leaf["reason"], leaf["arg_index"]) == ("call", "buffer_not_keyed", 1)
    assert leaf["suggested_source"] is None
    diag = result["diagnostics"]
    assert diag["frontier"]["seed_misanchored"] == 1
    assert diag["safe_to_report_all_clear"] is False


@pytest.mark.parametrize("source", ["call:recv", "arg:recv:1"])
def test_unkeyed_buffer_advice_never_sends_the_agent_to_the_other_locator_1041(models, source):
    """`arg:` and `call:` resolve the buffer through the same pointer, so each
    one's advice recommending the other is a loop, not a fix."""
    result = _engine(models).forward(_member_buffer_pump(), [te.parse_locator(source)])

    advice = result["diagnostics"]["next_action"]
    for command in re.findall(r"`(bn [^`]+)`", advice):
        assert not any(word.startswith(("call:", "arg:")) for word in shlex.split(command)), advice
    for leaf in _misanchored(result):
        assert "call:" not in leaf["note"] or "the same way" in leaf["note"], leaf


def test_call_preset_with_a_correlating_slot_keeps_its_flow_and_no_seed_leaf_1041(models):
    """The #193 slot correlation owns an indirect-load pointer: the `call:`
    preset must not add an unkeyed-buffer leaf on top of a slot that DID
    correlate to the re-load that reaches the sink."""
    engine = te.TaintEngine(FBV({0x900: "recvfrom", 0x901: "system"}), models)
    result = engine.forward(_slot_recv_func(), [te.parse_locator("call:recvfrom")])

    assert [s["sink"]["class"] for s in result["reached_sinks"]] == ["command_injection"]
    assert _misanchored(result) == []


def test_call_preset_null_destination_receives_nothing_1041(models):
    result = _engine(models).forward(_member_buffer_pump(dest="null"), [te.parse_locator("call:recv")])

    assert _misanchored(result) == []


def _callback_program(*, order="after", use="callback"):
    """`p#1 = buf#1 + have`, `x1 = p#1; recv(fd, x1, ..)`, and a second copy
    `x0 = p#1` handed to `fp(x0)` (``use="callback"``) or `system(x0)`
    (``use="system"``) -- two register copies of one unkeyed pointer (the
    #1050 review finding 6 AArch64 shape). ``order``: the use runs ``after``
    the receive, ``before`` it (#1050 round 2 `sys_before`), or before it in a
    ``loop`` body whose back-edge runs it again after the receive."""
    c, fd, have, buf, p, x1, x0 = (FVar(s) for s in ("c", "fd", "have", "buf", "p", "x1", "x0"))
    c0, fd0, have0 = FSSA(c, 0), FSSA(fd, 0), FSSA(have, 0)
    buf1, p1, x11, x01 = FSSA(buf, 1), FSSA(p, 1), FSSA(x1, 1), FSSA(x0, 1)
    prefix = [
        FInstr(0, 0x1000, "MLIL_SET_VAR_SSA", "buf#1 = [c#0 + 8]", reads=[c0], writes=[buf1],
               src=_field_load(c0, 8)),
        FInstr(0, 0x1004, "MLIL_SET_VAR_SSA", "p#1 = buf#1 + have#0", reads=[buf1, have0],
               writes=[p1], src=FExpr("MLIL_ADD", "buf#1 + have#0", reads=[buf1, have0],
                                      left=_var(buf1), right=_var(have0))),
    ]
    receive = [
        FInstr(0, 0x1008, "MLIL_SET_VAR_SSA", "x1#1 = p#1", reads=[p1], writes=[x11],
               src=_var(p1)),
        FInstr(0, 0x100c, "MLIL_CALL_SSA", "recv(fd#0, x1#1, 0x40, 0)", reads=[x11],
               dest=FExpr("MLIL_CONST_PTR", "0x900", constant=0x900),
               params=[_var(fd0), _var(x11), _const(0x40), _const(0)]),
    ]
    target = (FExpr("MLIL_VAR_SSA", "fp#1") if use == "callback"
              else FExpr("MLIL_CONST_PTR", "0x902", constant=0x902))
    handoff = [
        FInstr(0, 0x1010, "MLIL_SET_VAR_SSA", "x0#1 = p#1", reads=[p1], writes=[x01],
               src=_var(p1)),
        FInstr(0, 0x1014, "MLIL_CALL_SSA", f"{'fp' if use == 'callback' else 'system'}(x0#1)",
               reads=[x01], dest=target, params=[_var(x01)]),
    ]
    body = receive + handoff if order == "after" else handoff + receive
    ret = FInstr(0, 0x1020, "MLIL_RET", "return")
    ordered = prefix + body + [ret]
    for k, ins in enumerate(ordered):
        ins.instr_index = k
    entry, loop, exit_ = FBlock(0, 2), FBlock(2, 6), FBlock(6, 7)
    entry.outgoing_edges = [_FEdge(loop)]
    loop.outgoing_edges = ([_FEdge(loop), _FEdge(exit_)] if order == "loop"
                           else [_FEdge(exit_)])
    exit_.outgoing_edges = []
    for ins in ordered:
        ins.il_basic_block = entry if ins.instr_index < 2 else (
            loop if ins.instr_index < 6 else exit_)
    return FFunc("pump_cb", 0x1000, FSSAFunc(ordered), params=[c, fd, have])


def _cb_engine(models):
    return te.TaintEngine(FBV({0x900: "recv", 0x902: "system"}), models)


@pytest.mark.parametrize("order", ["after", "loop", "before"])
@pytest.mark.parametrize("source", ["call:recv", "arg:recv:1"])
def test_an_unkeyed_buffer_discloses_unfollowed_indirect_calls_1041(models, source, order):
    """#1041 item 1, narrowed after #1050 review round 3: whether a callback
    receives the same pointer is not decided from SSA copies (an alias made
    before the receive defeated every gating rule tried). The unkeyed-buffer
    leaf states it unconditionally, and no callback frontier is claimed."""
    result = _cb_engine(models).forward(_callback_program(order=order),
                                        [te.parse_locator(source)])

    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "source_seed_misanchored"]
    disclosure = ("indirect calls that may receive this buffer are not followed -- "
                  "their callees are unchecked")
    assert disclosure in leaf["detail"], leaf
    assert "indirect calls that may receive this buffer are not followed" in (
        result["diagnostics"]["next_action"])
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


@pytest.mark.parametrize("use", ["system", "callback"])
@pytest.mark.parametrize("source", ["call:recv", "arg:recv:1"])
def test_a_use_that_only_runs_before_the_receive_stays_clean_1050(models, source, use):
    """#1050 round 2 `sys_before`: `x0 = p; system(x0)` runs before
    `x1 = p; recv(fd, x1, ..)` and no path leads back, so the bytes received
    cannot reach it -- no sink, no callback frontier."""
    result = _cb_engine(models).forward(_callback_program(order="before", use=use),
                                        [te.parse_locator(source)])

    assert result["reached_sinks"] == []
    assert not any(lf["kind"] == "indirect_call_unresolved" for lf in result["leaves"])


def _field_cb_copy(*, callback="after", resolved=False):
    """The #1050 review round 4 `field_cb_copy` shape: `p = c->buf`, `recv(fd,
    p, ..)`, the field reloaded for a harmless `puts(c->buf)` (which correlates
    with the receive's slot), then the saved `p` handed to `fp(p)`.
    ``callback``: ``after`` the receive, ``before`` it, ``none``, or
    ``syscall`` (a process-ID system call after the receive instead).
    ``resolved``: the indirect call carries a value set pinning its target."""
    c, fd, p, q = FVar("c"), FVar("fd"), FVar("p"), FVar("q")
    c0, fd0, p1, q1 = FSSA(c, 0), FSSA(fd, 0), FSSA(p, 1), FSSA(q, 1)
    pvs = FPVS("ConstantPointerValue", value=0x3000) if resolved else None
    cb = FInstr(0, 0x1014, "MLIL_CALL_SSA", "fp(p#1)", reads=[p1],
                dest=FExpr("MLIL_VAR_SSA", "fp#1", possible_values=pvs), params=[_var(p1)])
    if callback == "syscall":
        cb = FInstr(0, 0x1014, "MLIL_SYSCALL_SSA", "syscall(sys_getpid)",
                    params=[_const(0x27)])
    body = [
        FInstr(0, 0x1000, "MLIL_SET_VAR_SSA", "p#1 = [c#0 + 8]", reads=[c0], writes=[p1],
               src=_field_load(c0, 8)),
        FInstr(0, 0x1004, "MLIL_CALL_SSA", "recv(fd#0, p#1, 0x40, 0)", reads=[p1],
               dest=FExpr("MLIL_CONST_PTR", "0x900", constant=0x900),
               params=[_var(fd0), _var(p1), _const(0x40), _const(0)]),
        FInstr(0, 0x1008, "MLIL_SET_VAR_SSA", "q#1 = [c#0 + 8]", reads=[c0], writes=[q1],
               src=_field_load(c0, 8)),
        FInstr(0, 0x100c, "MLIL_CALL_SSA", "puts(q#1)", reads=[q1],
               dest=FExpr("MLIL_CONST_PTR", "0x903", constant=0x903), params=[_var(q1)]),
    ]
    if callback in ("after", "syscall"):
        body.append(cb)
    elif callback == "before":
        body.insert(0, cb)
    body.append(FInstr(0, 0x1020, "MLIL_RET", "return"))
    block = FBlock(0, len(body))
    block.outgoing_edges = []
    for k, ins in enumerate(body):
        ins.instr_index = k
        ins.il_basic_block = block
    func = FFunc("field_cb_copy", 0x1000, FSSAFunc(body), params=[c, fd])
    target = FFunc("handler", 0x3000, FSSAFunc([FInstr(0, 0x3000, "MLIL_RET", "return")]),
                   params=[FVar("b")])
    return func, FBV({0x900: "recv", 0x903: "puts"}, funcs={0x3000: target})


def _unchecked(result):
    return [lf for lf in result["leaves"] if lf["kind"] == "indirect_call_unchecked"]


@pytest.mark.parametrize("source", ["call:recv", "arg:recv:1"])
def test_an_unresolved_callback_after_a_receive_withholds_the_all_clear_1050(models, source):
    """#1050 review round 4: a correlated slot reload once let `call:recv`
    report an all-clear while the saved pointer went to an unchecked callback.
    The gate claims nothing about aliasing: an unresolved indirect call
    reachable from the receive is a blocking leaf, keyed buffer or not."""
    func, bv = _field_cb_copy()
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator(source)])

    [leaf] = _unchecked(result)
    assert leaf["address"] == "0x1014"
    assert ("indirect calls that may receive this buffer are not followed -- their "
            "callees are unchecked") in leaf["detail"]
    # #1050 review round 5: pinning a target follows the call, not the buffer.
    assert "it does not show that the target receives this buffer" in leaf["detail"]
    assert "to follow it" not in leaf["detail"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


@pytest.mark.parametrize("variant", [
    {"callback": "none"},                    # the correlated slot control
    {"callback": "before"},                  # runs before the receive, no path back
    {"callback": "after", "resolved": True},  # its target is pinned and followed
    {"callback": "syscall"},                 # a system call is no function pointer
])
def test_no_unchecked_callback_keeps_the_correlated_slot_all_clear_1050(models, variant):
    func, bv = _field_cb_copy(**variant)
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("call:recv")])

    assert _unchecked(result) == []
    if variant["callback"] in ("none", "syscall"):
        assert result["diagnostics"]["safe_to_report_all_clear"] is True, result["diagnostics"]


def test_a_receive_family_all_clear_says_what_it_does_not_cover_1050(models):
    """#1050 review round 5: the gate covers the source function only, so an
    all-clear from a buffer-filling seed carries the caveat, in JSON and text."""
    from bn.formatters import _render_forward_diagnostics
    func, bv = _field_cb_copy(callback="none")
    diag = te.TaintEngine(bv, models).forward(func, [te.parse_locator("call:recv")])["diagnostics"]

    assert diag["safe_to_report_all_clear"] is True
    assert diag["all_clear_caveats"] == [te._taint_result_mod.RECEIVE_ALL_CLEAR_CAVEAT]
    assert any(line.strip().startswith("not covered: the all-clear does not cover other "
                                       "copies of the receive buffer's pointer")
               for line in _render_forward_diagnostics(diag))


@pytest.mark.parametrize("source", ["arg:recv:0", "param:1"])
def test_a_scalar_seed_all_clear_carries_no_receive_caveat_1050(models, source):
    """`arg:recv:0` seeds the descriptor, not the filled buffer; `param:1` is
    no receive at all."""
    func, bv = _field_cb_copy(callback="none")
    diag = te.TaintEngine(bv, models).forward(func, [te.parse_locator(source)])["diagnostics"]

    assert diag["safe_to_report_all_clear"] is True, diag
    assert diag["all_clear_caveats"] == []


# --------------------------------------------------------------------------
# #1041 item 3: --resolve-map is validated when it is loaded
# --------------------------------------------------------------------------

def _dispatch_program():
    """`dispatch(buf)` calls `fp(buf)` at 0x70c with no value set; `run_sys`
    at 0x500 passes its argument to system()."""
    buf, p = FVar("buf"), FVar("p")
    buf0, p0 = FSSA(buf, 0), FSSA(p, 0)
    run_sys = FFunc("run_sys", 0x500, FSSAFunc([
        FInstr(0, 0x504, "MLIL_CALL_SSA", "0x900(p#0)", reads=[p0],
               dest=FExpr("MLIL_CONST_PTR", "0x900", constant=0x900), params=[_var(p0)]),
    ]), params=[p])
    dispatch = FFunc("dispatch", 0x700, FSSAFunc([
        FInstr(0, 0x70c, "MLIL_CALL_SSA", "fp(buf#0)", reads=[buf0],
               dest=FExpr("MLIL_VAR_SSA", "fp#1"), params=[_var(buf0)]),
    ]), params=[buf])
    return dispatch, FBV({0x900: "system"}, funcs={0x500: run_sys})


def _find(name):
    if name == "run_sys":
        return 0x500
    raise RuntimeError(f"Function not found: {name}")


@pytest.mark.parametrize("rmap", [{"0x70c": ["run_sys"]}, {"0X70C": [0x500]}])
def test_resolve_map_target_names_and_any_hex_case_resolve_1041(models, rmap):
    dispatch, bv = _dispatch_program()
    engine = te.TaintEngine(bv, models, resolve_map=rmap, find_function=_find)

    result = engine.forward(dispatch, [te.parse_locator("param:0")])

    assert [s["sink"]["class"] for s in result["reached_sinks"]] == ["command_injection"]


@pytest.mark.parametrize(("rmap", "named"), [
    ({"0x70c": ["main"]}, "'main'"),             # a name nobody can resolve
    ({"0x70c": ["401176"]}, "'401176'"),         # bare digits: hex or decimal?
    ({"main": ["0x500"]}, "'main'"),             # a key must be a call address
    ({"0x70c": []}, "0x70c"),
    (["0x70c"], "JSON object"),
])
def test_resolve_map_refusal_names_the_file_and_entry_before_the_walk_1041(models, rmap, named):
    _dispatch, bv = _dispatch_program()

    with pytest.raises(te.TaintError) as refused:
        te.TaintEngine(bv, models, resolve_map=rmap, find_function=_find,
                       resolve_map_origin="--resolve-map maps/dispatch.json")

    message = str(refused.value)
    assert message.startswith("--resolve-map maps/dispatch.json: "), message
    assert named in message, message


def test_resolve_map_name_without_a_resolver_is_refused_not_guessed_1041(models):
    _dispatch, bv = _dispatch_program()
    with pytest.raises(te.TaintError, match="must be addresses"):
        te.TaintEngine(bv, models, resolve_map={"0x70c": ["run_sys"]})


# --------------------------------------------------------------------------
# #1041 item 4: width-less scanf string conversions are sinks
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("fmt", "unbounded"), [
    ("%[^\n]", [(0, "%[")]),
    ("%63[^\n]", []),               # a width bounds the token
    ("%d %s", [(1, "%s")]),         # the offset counts the %d destination
    ("%*s %s", [(0, "%s")]),        # a suppressed conversion has no destination
    ("%ms", []),                    # the callee allocates the destination
    ("%%s %c %s", [(1, "%s")]),     # %% is literal; %c is not a string read
    ("%1$s", None),                 # positions do not consume in order
])
def test_unbounded_scanf_strings_names_the_destination_offset_1041(fmt, unbounded):
    assert il_format.unbounded_scanf_strings(fmt) == unbounded


def _sscanf_program(fmt, *, input_tainted=True):
    """`n = read(fd, &msg, ..)` (or not), then `sscanf(&msg, fmt, &code, &name)`."""
    fd, msg, code, name, n = (FVar(s) for s in ("fd", "msg", "code", "name", "n"))
    n1 = FSSA(n, 1)
    fmt_arg = (FExpr("MLIL_CONST_PTR", repr(fmt), constant=0x5000) if fmt is not None
               else FExpr("MLIL_VAR_SSA", "fmt#0", reads=[FSSA(FVar("fmt"), 0)]))
    instrs = [
        FInstr(0, 0x100, "MLIL_CALL_SSA", "n#1 = read(fd#0, &msg, 0x40)", writes=[n1],
               dest=FExpr("MLIL_CONST_PTR", "0x900", constant=0x900),
               params=[_var(FSSA(fd, 0)), FExpr("MLIL_ADDRESS_OF", "&msg", src=msg),
                       _const(0x40)]),
        FInstr(1, 0x104, "MLIL_CALL_SSA", "sscanf(&msg, fmt, &code, &name)",
               dest=FExpr("MLIL_CONST_PTR", "0x901", constant=0x901),
               params=[FExpr("MLIL_ADDRESS_OF", "&msg", src=msg if input_tainted else FVar("other")),
                       fmt_arg,
                       FExpr("MLIL_ADDRESS_OF", "&code", src=code),
                       FExpr("MLIL_ADDRESS_OF", "&name", src=name)]),
    ]
    bv = FBVStr({0x900: "read", 0x901: "sscanf"}, strings={0x5000: fmt} if fmt else {})
    return FFunc("parse_reply", 0x100, FSSAFunc(instrs), params=[fd]), bv


def _format_sinks(result):
    return [s["sink"] for s in result["reached_sinks"] if s["sink"].get("format_conversions")]


def test_sscanf_width_less_string_into_a_tainted_input_is_a_sink_1041(models):
    func, bv = _sscanf_program("%d %s")
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("call:read")])

    [sink] = _format_sinks(result)
    assert (sink["class"], sink["tainted_arg_index"]) == ("overflow_unbounded", 0)
    assert sink["format_conversions"] == [{"conversion": "%s", "dest_arg": 3}]


@pytest.mark.parametrize(("fmt", "input_tainted"), [
    ("%d %15s", True),     # bounded by a width
    ("%d %s", False),      # the input string is not the tainted buffer
])
def test_sscanf_bounded_or_untainted_is_not_a_sink_1041(models, fmt, input_tainted):
    func, bv = _sscanf_program(fmt, input_tainted=input_tainted)
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("call:read")])

    assert _format_sinks(result) == []


class _WritableStrings(FBVStr):
    """The format string sits in writable `.data` (`char g_fmt[] = "%s"`)."""

    def is_offset_writable(self, addr):
        return True


@pytest.mark.parametrize(("fmt", "writable", "reason"), [
    (None, False, "not_constant"),       # a format passed in
    ("%1$s", False, "positional"),       # positions do not consume in order
    ("%5s", True, "not_constant"),       # initial bytes of a mutable global
    ("%s", True, "not_constant"),        # ... in either direction
])
def test_unchecked_format_is_a_blocking_frontier_not_a_guess_1050(models, fmt, writable, reason):
    """#1050 review findings 1 and 4: a format bn cannot read as a read-only
    constant is never guessed -- not as bounded (no all-clear) and not as
    `%s` (no finding) -- and withholds the all-clear."""
    func, bv = _sscanf_program(fmt)
    if writable:
        bv = _WritableStrings(bv._names, strings=bv._strings)
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("call:read")])

    assert _format_sinks(result) == []
    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "format_unchecked"]
    assert (leaf["reason"], leaf["format_arg"]) == (reason, 1)
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_format_gated_sink_is_listed_with_its_condition_1041():
    [entry] = [e for e in build_catalog(te.load_models(), role="sink")["sinks_by_class"]
               ["overflow_unbounded"] if e["symbol"] == "sscanf"]
    assert "constant format (argument 1)" in entry["model_description"]
    assert "argument 0 is tainted" in entry["model_description"]


def test_format_arg_without_first_dest_is_refused_1041():
    with pytest.raises(te.TaintError, match="first_dest"):
        _validate_model_interior("my_scanf", {"sink": {"class": "overflow_unbounded",
                                                       "format_arg": 0}}, "--models x.json")


# --------------------------------------------------------------------------
# #932: discovering tokenizer state is part of fixpoint convergence
# --------------------------------------------------------------------------

def _continuation_before_initializer(*, back_edge: bool, undetermined: bool = False):
    """tok(line#0): { x = strtok(NULL, " "); system(x) } then `strtok(&line, " ")`
    with its return unused -- the IL lists the continuation FIRST. With a
    back-edge from the initializer's block to the continuation's, the
    initializer arms it on the next iteration; without one it never can.
    ``undetermined``: the initializer's block ends in a register-indirect jump
    BN could not resolve -- no enumerated edges, `has_undetermined_outgoing_edges`."""
    line = FVar("line", typ="char[0x40]")
    x = FSSA(FVar("x"), 1)
    STRTOK, SYS = 0x900, 0x901
    cont = _ext_call(0, 0x10, 'x#1 = strtok(nullptr, " ")', STRTOK,
                     [FExpr("MLIL_CONST", "0", constant=0), _cstr(0x700)], writes=[x])
    sink = _ext_call(1, 0x14, "system(x#1)", SYS, [FExpr("MLIL_VAR_SSA", "x#1", reads=[x])],
                     reads=[x])
    init = _ext_call(2, 0x18, 'strtok(&line, " ")', STRTOK, [_addr_of(line), _cstr(0x700)])
    ret = FInstr(3, 0x1c, "MLIL_RET", "return")
    head, tail, exit_ = FBlock(0, 2), FBlock(2, 3), FBlock(3, 4)
    head.outgoing_edges = [_FEdge(tail)]
    tail.outgoing_edges = [_FEdge(head), _FEdge(exit_)] if back_edge else [_FEdge(exit_)]
    if undetermined:
        tail.outgoing_edges = []
        tail.has_undetermined_outgoing_edges = True
    exit_.outgoing_edges = []
    for ins, block in ((cont, head), (sink, head), (init, tail), (ret, exit_)):
        ins.il_basic_block = block
    func = FFunc("tok", 0x10, FSSAFunc([cont, sink, init, ret]), params=[line])
    return func, FBV({STRTOK: "strtok", SYS: "system"})


def test_undetermined_successors_never_prove_disconnection_1050(models):
    """#1050 review finding 2: a computed jump after the tainted strtok may
    resume at the continuation; BN enumerates no edges for it, which is not
    proof that it cannot."""
    func, bv = _continuation_before_initializer(back_edge=False, undetermined=True)
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("param:0")])

    sinks = [(s["sink"]["callee"], s["sink"]["class"]) for s in result["reached_sinks"]]
    assert sinks == [("system", "command_injection")], result["assumptions"]


def test_a_clean_call_before_the_tainted_one_is_not_named_a_reset_1050(models):
    """#1050 review round 3: straight-line `strtok(clean)`, `strtok(line)`,
    `system(strtok(NULL))`. The clean call runs before the tainted one, so it
    is on no path from it to the continuation or to the return; neither caveat
    may call it a partial reset."""
    line, clean = FVar("line", typ="char[0x40]"), FVar("clean", typ="char[0x10]")
    x = FSSA(FVar("x"), 1)
    STRTOK, SYS = 0x900, 0x901
    instrs = [
        _ext_call(0, 0x10, 'strtok(&clean, " ")', STRTOK, [_addr_of(clean), _cstr(0x700)]),
        _ext_call(1, 0x14, 'strtok(&line, " ")', STRTOK, [_addr_of(line), _cstr(0x700)]),
        _ext_call(2, 0x18, 'x#1 = strtok(nullptr, " ")', STRTOK,
                  [FExpr("MLIL_CONST", "0", constant=0), _cstr(0x700)], writes=[x]),
        _ext_call(3, 0x1c, "system(x#1)", SYS, [FExpr("MLIL_VAR_SSA", "x#1", reads=[x])],
                  reads=[x]),
        FInstr(4, 0x20, "MLIL_RET", "return"),
    ]
    block = FBlock(0, 5)
    block.outgoing_edges = []
    for ins in instrs:
        ins.il_basic_block = block
    func = FFunc("tok", 0x10, FSSAFunc(instrs), params=[line])
    result = te.TaintEngine(FBV({STRTOK: "strtok", SYS: "system"}), models).forward(
        func, [te.parse_locator("param:0")])

    assert [s["sink"]["callee"] for s in result["reached_sinks"]] == ["system"]
    tokenizer = [a for a in result["assumptions"] if "strtok" in a]
    assert tokenizer and not any("clean strtok @ 0x10" in a for a in tokenizer), tokenizer


def test_reset_disclosure_is_written_from_converged_taint_1050(models):
    """#1050 review finding 7: `strtok(&line)`, `x = strtok(NULL)`, then
    `memcpy(&other, &line)` taints `other`, then `strtok(&other)`. On the first
    pass `strtok(&other)` still looked clean; the final caveats must not name
    it a clean reset, nor repeat the continuation with and without that clause."""
    line, other = FVar("line", typ="char[0x40]"), FVar("other", typ="char[0x40]")
    x = FSSA(FVar("x"), 1)
    STRTOK, SYS, MEMCPY = 0x900, 0x901, 0x902
    instrs = [
        _ext_call(0, 0x10, 'strtok(&line, " ")', STRTOK, [_addr_of(line), _cstr(0x700)]),
        _ext_call(1, 0x14, 'x#1 = strtok(nullptr, " ")', STRTOK,
                  [FExpr("MLIL_CONST", "0", constant=0), _cstr(0x700)], writes=[x]),
        _ext_call(2, 0x18, "system(x#1)", SYS, [FExpr("MLIL_VAR_SSA", "x#1", reads=[x])],
                  reads=[x]),
        _ext_call(3, 0x1c, "memcpy(&other, &line, 0x40)", MEMCPY,
                  [_addr_of(other), _addr_of(line), _const(0x40)]),
        _ext_call(4, 0x20, 'strtok(&other, " ")', STRTOK, [_addr_of(other), _cstr(0x700)]),
    ]
    func = FFunc("tok", 0x10, FSSAFunc(instrs), params=[line])
    bv = FBV({STRTOK: "strtok", SYS: "system", MEMCPY: "memcpy"})
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("param:0")])

    cont = [a for a in result["assumptions"] if "continues the tokenizer state" in a]
    # One caveat per arming site (0x10, and 0x20 once `other` is tainted), and
    # none still calls 0x20 a clean reset.
    assert sorted(a.split(" call @ ")[1].split(",")[0] for a in cont) == ["0x10", "0x20"], cont
    assert not any("clean strtok @ 0x20" in a for a in cont), cont


def test_state_discovered_after_its_continuation_is_revisited_932(models):
    """The initializer's return is unused, so arming the state changes nothing
    else in the pass; the discovery itself must force the revisit."""
    func, bv = _continuation_before_initializer(back_edge=True)
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("param:0")])

    sinks = [(s["sink"]["callee"], s["sink"]["class"]) for s in result["reached_sinks"]]
    assert sinks == [("system", "command_injection")], result["assumptions"]


def test_state_that_cannot_reach_the_continuation_does_not_arm_it_932(models):
    """Every tainted site is state now, not only the latest in IL order, so the
    reachability test runs even with no clean reset: an initializer that runs
    only AFTER the continuation must not arm it."""
    func, bv = _continuation_before_initializer(back_edge=False)
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("param:0")])

    assert result["reached_sinks"] == []
    assert any(a.startswith("tokenizer_state_escapes") for a in result["assumptions"])


# --------------------------------------------------------------------------
# #1041 item 6: each flow of a several-source run names its own source
# --------------------------------------------------------------------------

def test_each_flow_names_the_source_it_starts_from_1041(models):
    """`f(a, b)`: `memcpy(.., .., a)` and `memcpy(.., .., b)`. Run with both
    params as sources, each flow used to say `multiple`."""
    a, b = FVar("a"), FVar("b")
    a0, b0 = FSSA(a, 0), FSSA(b, 0)
    MEMCPY = 0x901

    def copy(i, addr, length):
        return _ext_call(i, addr, f"memcpy(0x6000, 0x7000, {length.var.name}#0)", MEMCPY,
                         [_const(0x6000), _const(0x7000), _var(length)], reads=[length])

    func = FFunc("f", 0x10, FSSAFunc([copy(0, 0x10, a0), copy(1, 0x14, b0)]), params=[a, b])
    result = te.TaintEngine(FBV({MEMCPY: "memcpy"}), models).forward(
        func, [te.parse_locator("param:0"), te.parse_locator("param:1")])

    named = {s["sink"]["address"]: s["signature"]["source"] for s in result["reached_sinks"]}
    assert named == {"0x10": "param:0", "0x14": "param:1"}


def _two_sources_program():
    """`two_sources(first, second)` calls `helper(first, second)`; helper stores
    `first` to a global and uses only `second` as a memcpy length (#1050 review
    finding 3). `both(a)` passes `a` as both lengths of one memcpy pair."""
    q, r = FVar("q"), FVar("r")
    q0, r0 = FSSA(q, 0), FSSA(r, 0)
    MEMCPY, HELPER = 0x901, 0x2000
    helper = FFunc("helper", HELPER, FSSAFunc([
        FInstr(0, 0x2000, "MLIL_STORE_SSA", "[0x6100].q = q#0", reads=[q0],
               dest=FExpr("MLIL_CONST_PTR", "0x6100", constant=0x6100), src=_var(q0)),
        _ext_call(1, 0x2004, "memcpy(0x6000, 0x7000, r#0)", MEMCPY,
                  [_const(0x6000), _const(0x7000), _var(r0)], reads=[r0]),
    ]), params=[q, r])
    first, second = FVar("first"), FVar("second")
    f0, s0 = FSSA(first, 0), FSSA(second, 0)
    caller = FFunc("two_sources", 0x1000, FSSAFunc([
        FInstr(0, 0x1000, "MLIL_CALL_SSA", "helper(first#0, second#0)", reads=[f0, s0],
               dest=FExpr("MLIL_CONST_PTR", hex(HELPER), constant=HELPER),
               params=[_var(f0), _var(s0)]),
    ]), params=[first, second])
    return caller, FBV({MEMCPY: "memcpy"}, funcs={HELPER: helper})


def test_a_callee_flow_names_only_the_source_that_reaches_it_1050(models):
    caller, bv = _two_sources_program()
    result = te.TaintEngine(bv, models).forward(
        caller, [te.parse_locator("param:0"), te.parse_locator("param:1")])

    [flow] = result["reached_sinks"]
    assert flow["signature"]["source"] == "param:1"
    assert flow["source_candidates"] == ["param:1"]


def test_a_flow_each_source_reaches_alone_stays_multiple_1050(models):
    """`n = a + b; memcpy(.., .., n)`: either source alone reaches the sink, so
    neither is THE source; both are listed as candidates."""
    a, b, n = FVar("a"), FVar("b"), FVar("n")
    a0, b0, n1 = FSSA(a, 0), FSSA(b, 0), FSSA(n, 1)
    MEMCPY = 0x901
    func = FFunc("f", 0x10, FSSAFunc([
        FInstr(0, 0x10, "MLIL_SET_VAR_SSA", "n#1 = a#0 + b#0", reads=[a0, b0], writes=[n1]),
        _ext_call(1, 0x14, "memcpy(0x6000, 0x7000, n#1)", MEMCPY,
                  [_const(0x6000), _const(0x7000), _var(n1)], reads=[n1]),
    ]), params=[a, b])
    result = te.TaintEngine(FBV({MEMCPY: "memcpy"}), models).forward(
        func, [te.parse_locator("param:0"), te.parse_locator("param:1")])

    [flow] = result["reached_sinks"]
    assert flow["signature"]["source"] == "multiple"
    assert flow["source_candidates"] == ["param:0", "param:1"]


@pytest.mark.parametrize("doc", [[], False, 0, ""])
def test_a_false_y_resolve_map_document_is_refused_1050(models, doc):
    """#1050 review finding 8: only an absent map defaults to empty."""
    _dispatch, bv = _dispatch_program()
    with pytest.raises(te.TaintError, match="--resolve-map m.json"):
        te.TaintEngine(bv, models, resolve_map=doc, resolve_map_origin="--resolve-map m.json")


def test_a_supplied_null_resolve_map_is_refused_1050(models):
    """#1050 review round 2: JSON `null` is a supplied document, not an omission."""
    _dispatch, bv = _dispatch_program()
    with pytest.raises(te.TaintError, match="--resolve-map m.json: expected a JSON object"):
        te.TaintEngine(bv, models, resolve_map=None, resolve_map_origin="--resolve-map m.json")


def _scanf_stale_arg(fmt):
    """`scanf(fmt, &b, x2#1)`: BN recovered a third argument from a stale
    register (the stack-guard address on AArch64) that `"%s"` never writes."""
    b, x2 = FVar("b", typ="char[0x10]"), FSSA(FVar("x2"), 1)
    fmt_arg = (FExpr("MLIL_CONST_PTR", repr(fmt), constant=0x5000) if fmt is not None
               else FExpr("MLIL_VAR_SSA", "fmt#0", reads=[FSSA(FVar("fmt"), 0)]))
    call = _ext_call(0, 0x10, "scanf(fmt, &b, x2#1)", 0x900,
                     [fmt_arg, _addr_of(b), FExpr("MLIL_VAR_SSA", "x2#1", reads=[x2])])
    func = FFunc("scan_s", 0x10, FSSAFunc([call]), params=[])
    return func, FBVStr({0x900: "scanf"}, strings={0x5000: fmt} if fmt else {})


@pytest.mark.parametrize("fmt", ["%s", None], ids=["constant-format", "unreadable-format"])
def test_a_stale_argument_is_still_seeded_by_the_modeled_range_1050(models, fmt):
    """#1050 review round 9 removed destination trimming: the call: preset
    seeds the model's whole *arg:1..4 run, so BN's stale arg 2 (the stack-
    guard address) keeps its conservative `source_seed_misanchored` leaf --
    a withheld all-clear, never a dropped destination."""
    func, bv = _scanf_stale_arg(fmt)
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("call:scanf")])

    assert [lf["arg_index"] for lf in result["leaves"]
            if lf["kind"] == "source_seed_misanchored"] == [2]


def test_a_format_needing_more_destinations_than_recovered_withholds_the_all_clear_1050(models):
    """#1050 review round 8: `scanf("%15s %15s %15s", a, b, c)` where `b` and
    `c` are forwarded incoming pointers left in their argument registers, and
    BN's MLIL lists only the format and `a`. The two lost destinations are
    disclosed by count -- never guessed -- and the all-clear withheld."""
    a = FVar("a", typ="char[0x10]")
    call = _ext_call(0, 0x10, "scanf(fmt, &a)", 0x900,
                     [FExpr("MLIL_CONST_PTR", "fmt", constant=0x5000), _addr_of(a)])
    func = FFunc("fwd", 0x10, FSSAFunc([call]), params=[])
    bv = FBVStr({0x900: "scanf"}, strings={0x5000: "%15s %15s %15s"})
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("call:scanf")])

    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "format_destination_unrecovered"]
    assert (leaf["destinations"], leaf["recovered"], leaf["missing"]) == (3, 1, 2)
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


class _PrefixStringView(FBVStr):
    """BN's ASCII string at the format stops at a non-ASCII literal byte; the
    raw bytes run on to the real terminator."""

    def __init__(self, addr_names, blob):
        super().__init__(addr_names, strings={})
        self._blob = blob

    def get_ascii_string_at(self, addr, min_length=1):
        if int(addr) != 0x5000:
            return None
        return type("S", (), {"value": self._blob.split(b"\xe9", 1)[0].decode()})()

    def read(self, addr, length):
        off = int(addr) - 0x5000
        return self._blob[off:off + length] if 0 <= off < len(self._blob) else b""


@pytest.mark.parametrize("pad", [600, 5000], ids=["nul-in-window", "no-nul-in-window"])
def test_a_format_prefix_without_its_terminator_never_trims_destinations_1050(models, pad):
    """#1050 review round 8: two bounded conversions separated by a non-ASCII
    literal and padding. A prefix that did not reach the NUL is not the format:
    the second destination stays seeded, and its command sink is found."""
    a, b = FVar("a", typ="char[0x10]"), FVar("b", typ="char[0x10]")
    blob = b"%15s\xe9" + b"A" * pad + b"%15s\x00"
    scan = _ext_call(0, 0x10, "scanf(fmt, &a, &b)", 0x900,
                     [FExpr("MLIL_CONST_PTR", "fmt", constant=0x5000), _addr_of(a), _addr_of(b)])
    run = _ext_call(1, 0x14, "system(&b)", 0x901, [_addr_of(b)])
    func = FFunc("two", 0x10, FSSAFunc([scan, run]), params=[])
    bv = _PrefixStringView({0x900: "scanf", 0x901: "system"}, blob)
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("call:scanf")])

    assert [s["sink"]["class"] for s in result["reached_sinks"]] == ["command_injection"]
    unchecked = [lf for lf in result["leaves"] if lf["kind"] == "format_unchecked"]
    assert bool(unchecked) == (pad == 5000)   # unterminated within the read: unreadable


def test_a_format_with_a_writable_suffix_never_drops_a_real_destination_1050(models):
    """#1050 review round 9: the format starts in read-only storage and runs
    into an adjacent writable segment, where the program rewrites it to add
    a third conversion. The stored text names two destinations; the modeled
    range is seeded regardless, so the third buffer's command sink is found."""
    a, b, c = (FVar(n, typ="char[0x10]") for n in "abc")
    scan = _ext_call(0, 0x10, "scanf(fmt, &a, &b, &c)", 0x900,
                     [FExpr("MLIL_CONST_PTR", "fmt", constant=0x5000),
                      _addr_of(a), _addr_of(b), _addr_of(c)])
    run = _ext_call(1, 0x14, "system(&c)", 0x901, [_addr_of(c)])
    func = FFunc("seg", 0x10, FSSAFunc([scan, run]), params=[])
    bv = FBVStr({0x900: "scanf", 0x901: "system"}, strings={0x5000: "%15s %15s"})
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("call:scanf")])

    assert [s["sink"]["class"] for s in result["reached_sinks"]] == ["command_injection"]
