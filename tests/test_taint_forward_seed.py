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

from test_taint_engine import (  # noqa: F401 -- `models` is a fixture
    FBV, FExpr, FFunc, FInstr, FSSA, FSSAFunc, FVar, _slot_recv_func, models, te,
)


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
    dispatch, bv = _dispatch_program()

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
