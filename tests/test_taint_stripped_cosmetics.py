"""Stripped targets (W12-09): every function is an auto ``sub_…`` name, and text
must still name functions and call targets the way ``bn disasm`` does.

- a backward origin's function prints as its ``sub_…`` display name, not a
  bare address;
- a dropped-result frontier points its ``disasm`` advice at the call site,
  not the caller's entry, when caller and callee are both ``sub_…``;
- a callsite row's one-line disassembly names an import's PLT target
  (``call 0x… <name>``) as ``bn disasm`` does.
"""
from __future__ import annotations

import re
import shlex
from types import SimpleNamespace

from bn.formatters import _render_taint_text
from bn_agent_bridge import read_taint_slice
from test_taint_backward_coverage import _BV, _NAMES, _caller, _const, _length_sink
from test_taint_engine import FBV, FExpr, FFunc, FInstr, FSite, FSSA, FSSAFunc, FVar, te


def test_backward_origin_function_prints_its_sub_name():
    sink = _length_sink("sub_800", 0x800, 10)
    caller = _caller("sub_a00", 0xa00, 0xa10, sink.start, _const(0x40))
    sink.caller_sites = [FSite(caller, 0xa10)]
    # No symbol for either function: the view knows them only as functions,
    # under BN's auto `sub_…` names.
    bv = _BV(dict(_NAMES))
    bv._funcs.update({sink.start: sink, caller.start: caller})
    result = te.TaintEngine(bv, te.load_models()).backward(
        sink, [te.parse_locator("arg:memcpy:2")], max_depth=8)
    [row] = result["slices"]
    assert row["origin"]["function"] == "0xa00"
    assert row["origin_names"]["0xa00"]["display_name"] == "sub_a00"
    [line] = [ln for ln in _render_taint_text(result).splitlines()
              if ln.lstrip().startswith("origin")]
    assert line.endswith(" in sub_a00"), line
    assert "0xa00" not in line


def _value(var):
    return FExpr("MLIL_VAR_SSA", str(var), reads=[var], src=var)


def test_dropped_result_advice_on_sub_functions_reads_the_call_site():
    n, p = FVar("n"), FVar("p")
    ns, ps = FSSA(n, 0), FSSA(p, 0)
    callee = FFunc("sub_2000", 0x2000, FSSAFunc([
        FInstr(0, 0x2004, "MLIL_RET", "return", reads=[ps], src=[_value(ps)])]), params=[p])
    caller = FFunc("sub_1000", 0x1000, FSSAFunc([
        FInstr(0, 0x1010, "MLIL_CALL_SSA", "sub_2000(n)", reads=[ns], writes=[],
               params=[_value(ns)], dest=FExpr("MLIL_CONST_PTR", "0x2000", constant=0x2000))]),
        params=[n])
    bv = FBV({}, {0x1000: caller, 0x2000: callee})
    result = te.TaintEngine(bv, te.load_models()).forward(caller, [te.parse_locator("param:0")])
    [leaf] = [lf for lf in result["leaves"] if lf.get("cause") == "return_value_unrecovered"]
    assert leaf["address"] == "0x1010"
    commands = [shlex.split(c)[1:] for c in re.findall(r"`(bn disasm [^`]+)`", leaf["detail"])]
    assert commands == [["disasm", "0x1010", "--linear", commands[0][3]]]
    assert int(commands[0][3]) > 0
    assert "0x1000" not in leaf["detail"]


class _Token:
    def __init__(self, text, kind="TextToken", value=None):
        self.text, self.value = text, value
        self.type = SimpleNamespace(name=kind)

    def __str__(self):
        return self.text


class _Arch:
    max_instr_length = 16

    def get_instruction_text(self, data, address):
        return [_Token("call"), _Token("    "),
                _Token("0x3030", "PossibleAddressToken", 0x3030)], 5

    def get_instruction_info(self, data, address):
        return SimpleNamespace(length=5, branches=[SimpleNamespace(target=0x3030)])


class _View:
    arch = _Arch()

    def read(self, address, length):
        return b"\xe8" + b"\x00" * (length - 1)

    def get_disassembly(self, address, arch=None):
        return "call    0x3030"

    def get_symbol_at(self, address):
        return SimpleNamespace(short_name="read", name="read") if address == 0x3030 else None

    def get_function_at(self, address):
        return None

    def get_instruction_length(self, address, arch=None):
        return 5


def test_callsite_row_disasm_names_the_import_target():
    assert read_taint_slice._disasm_at(_View(), 0x1010) == "call    0x3030 <read>"


def test_callsite_row_disasm_degrades_to_empty_on_an_unreadable_view():
    assert read_taint_slice._disasm_at(SimpleNamespace(), 0x1010) == ""
