"""One leaf shape, and advice a reader can run as printed (W07-11, W07-12,
W05-04, W06-10).

Every leaf carries its human sentence under ``detail`` and names its function
as a ``{name, display_name, address}`` object; the two kinds that said ``note``
and the backward leaves that said a bare hex address made every consumer
branch on the kind. A flag a leaf tells the reader to raise must exist, and a
disassembly it points at must be the call, in a bounded window.
"""
from __future__ import annotations

import re
import shlex

import pytest

import bn.cli
from test_taint_engine import (
    _caller_fan_in_program, _frontier_depth_program, _popen_of_local_program,
    _variadic_logger_program, te,
)
from test_taint_tail_return import _dropped_result_program, _program


def _assert_shape(leaves: list[dict]) -> None:
    assert leaves
    for leaf in leaves:
        assert "note" not in leaf, leaf
        if "function" in leaf:
            fn = leaf["function"]
            assert isinstance(fn, dict) and fn.get("address", "").startswith("0x"), leaf
            assert fn.get("name"), leaf


def test_forward_leaves_carry_detail_never_note():
    func, bv = _frontier_depth_program()
    depth_cut = te.TaintEngine(bv, te.load_models()).forward(
        func, [te.parse_locator("arg:recv:1")], max_depth=0)
    stopped = te.TaintEngine(bv, te.load_models(), unknown_call_policy="stop").forward(
        func, [te.parse_locator("arg:recv:1")], max_depth=0)
    handler, vbv, _, _ = _variadic_logger_program(True)
    varargs = te.TaintEngine(vbv, te.load_models()).forward(
        handler, [te.parse_locator("param:0")])
    for result in (depth_cut, stopped, varargs):
        _assert_shape(result["leaves"])
        assert all(leaf.get("detail") for leaf in result["leaves"]
                   if leaf["kind"] in ("unmodeled_callee", "arg_under_recovered"))


def test_backward_leaves_name_their_function_as_an_object():
    func, original = _popen_of_local_program(fill="compose")
    buffer_run = te.TaintEngine(original, te.load_models()).backward(
        func, [te.parse_locator("arg:popen:0")])
    use_len, bv = _caller_fan_in_program(20)
    capped = te.TaintEngine(bv, te.load_models()).backward(
        use_len, [te.parse_locator("arg:memcpy:2")])
    for result in (buffer_run, capped):
        _assert_shape(result["leaves"])
    assert any("function" in leaf for leaf in buffer_run["leaves"])


def test_backward_advice_runs_against_the_leaf_function_address():
    func, original = _popen_of_local_program(fill="compose")
    result = te.TaintEngine(original, te.load_models()).backward(
        func, [te.parse_locator("arg:popen:0")])
    advice = result["diagnostics"]["next_action"]
    addresses = {leaf["function"]["address"] for leaf in result["leaves"] if "function" in leaf}
    commands = re.findall(r"`bn ([^`]+)`", advice)
    assert commands, advice
    # Every function selector the advice prints is a leaf's function address.
    for command in commands:
        words = shlex.split(command)
        for flag in ("-f", "--within"):
            if flag in words:
                assert words[words.index(flag) + 1] in addresses, (command, addresses)


def test_seed_misanchored_leaves_carry_one_detail():
    for leaf in (te.misanchored_recv_leaf(callee="recv", arg_index=1, reason="buffer_not_keyed"),
                 te.misanchored_recv_leaf(callee="recvmsg", arg_index=1, reason="msghdr_not_payload"),
                 te.indirect_pointer_slot_leaf(callee="recvfrom", arg_index=1)):
        _assert_shape([leaf])
        assert leaf["detail"]


def test_a_depth_cut_names_only_flags_taint_forward_accepts():
    func, bv = _frontier_depth_program()
    result = te.TaintEngine(bv, te.load_models()).forward(
        func, [te.parse_locator("arg:recv:1")], max_depth=0)
    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "unmodeled_callee"]
    sub = bn.cli._selected_parser_for_argv(bn.cli.build_parser(), ["taint", "forward"])
    accepted = bn.cli._known_option_strings(sub)
    flags = re.findall(r"--[a-z][a-z-]*", leaf["detail"])
    assert flags, leaf["detail"]
    for flag in flags:
        assert flag in accepted, (flag, leaf["detail"])


def _disasm_commands(text: str) -> list[list[str]]:
    return [shlex.split(c)[1:] for c in re.findall(r"`(bn disasm [^`]+)`", text)]


@pytest.mark.parametrize("callee_returns_value", [True, False])
def test_dropped_result_advice_reads_a_window_at_the_call(callee_returns_value):
    main, bv, models = _dropped_result_program(callee_returns_value=callee_returns_value)
    result = te.TaintEngine(bv, models).forward(main, [te.parse_locator("param:0")])
    [leaf] = [lf for lf in result["leaves"]
              if lf.get("cause") == "return_value_unrecovered" and lf["address"] == "0x1010"]
    [command] = _disasm_commands(leaf["detail"])
    assert command[1] == leaf["address"]                    # the call, not the caller's entry
    assert command[2] == "--linear" and int(command[3]) > 0  # a bounded window


def test_valueless_return_advice_reads_bounded_windows_including_the_call():
    main, bv, models = _program("valueless", tail=False, wrapper_returns_value=False)
    result = te.TaintEngine(bv, models).forward(main, [te.parse_locator("param:0")])
    [leaf] = [lf for lf in result["leaves"] if lf.get("cause") == "return_value_unrecovered"]
    commands = _disasm_commands(leaf["detail"])
    assert commands
    assert all(c[2] == "--linear" for c in commands), commands
    assert leaf["address"] in {c[1] for c in commands}, commands
