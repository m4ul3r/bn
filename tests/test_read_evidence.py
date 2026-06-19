from __future__ import annotations

import importlib
import importlib.util
import io
import json
import socket
import sys
import threading
import time
import types
import weakref
from pathlib import Path

import pytest

from _bridge_fakes import *  # noqa: F401,F403


def test_evidence_mlil_drops_clobber_lhs(monkeypatch):
    """evidence function's per-call `mlil` shows the call + its inputs, not the
    full caller-saved clobber set BN renders as the assignment LHS. (E17)"""
    bridge = _load_bridge(monkeypatch)
    render = bridge.read_evidence._mlil_call_text
    clobber = "arg1, arg2, x2, x3, lr, v0, v31 = call(0x471d60, arg1, arg2, x2, stack = &fp)"
    assert render(clobber) == "call(0x471d60, arg1, arg2, x2, stack = &fp)"
    assert render("call(0x401000, arg1)") == "call(0x401000, arg1)"  # no output: unchanged
    assert render(None) is None

    class _M:
        def __str__(self):
            return clobber

    assert render(_M()) == "call(0x471d60, arg1, arg2, x2, stack = &fp)"


def test_stack_var_span_annotation(monkeypatch):
    """Stack vars carry span_to_next (bytes to the next stack slot = the slot's
    capacity); register/flag locals (non-negative storage) do not. (F20)"""
    bridge = _load_bridge(monkeypatch)
    entries = [
        {"name": "buf", "storage": -1016},
        {"name": "x", "storage": -8},
        {"name": "reg", "storage": 53},     # register var -> no span
        {"name": "buf2", "storage": -1024},
    ]
    bridge.vars_mod._annotate_stack_spans(entries)
    by = {e["name"]: e for e in entries}
    # sorted stack: -1024 (buf2) -> -1016 (buf) -> -8 (x) -> 0 (frame base)
    assert by["buf2"]["span_to_next"] == 8       # -1016 - (-1024)
    assert by["buf"]["span_to_next"] == 1008     # -8 - (-1016)
    assert by["x"]["span_to_next"] == 8          # 0 - (-8)
    assert "span_to_next" not in by["reg"]       # register var untouched


def test_segment_entries_for_verbose_target_info(monkeypatch):
    """target info --verbose builds a segment map with r/w/x flags + length;
    a bv with no segments yields an empty list, not an error. (F21)"""
    bridge = _load_bridge(monkeypatch)
    seg = type("S", (), {"start": 0x1000, "end": 0x2000,
                         "readable": True, "writable": False, "executable": True})()
    bv = type("BV", (), {"segments": [seg]})()
    assert bridge._segment_entries(bv) == [{
        "start": "0x1000", "end": "0x2000", "length": 0x1000,
        "readable": True, "writable": False, "executable": True,
    }]
    assert bridge._segment_entries(type("BV", (), {})()) == []


def test_function_name_summary_counts_named_vs_auto(monkeypatch):
    """target info needs a function-count summary every agent reaches for.
    Auto-named functions are BN's sub_<addr> / j_sub_<addr> defaults; named are
    everything else EXCEPT import/extern stubs (whose names come from
    relocations), which get their own bucket so they don't inflate "named" on a
    stripped binary (#122)."""
    bridge = _load_bridge(monkeypatch)

    class _Sym:
        def __init__(self, type_name):
            self.type = type("_SymType", (), {"name": type_name})()

    imported = _FakeFunction(0x401400, "puts")
    imported.symbol = _Sym("ImportedFunctionSymbol")

    bv = _FakeBV(functions=[
        _FakeFunction(0x401000, "main"),
        _FakeFunction(0x401100, "parse_header"),
        _FakeFunction(0x401200, "sub_401200"),
        _FakeFunction(0x401300, "j_sub_401300"),
        imported,
    ])

    summary = bridge._function_name_summary(bv)

    assert summary["function_count"] == 5
    assert summary["named_function_count"] == 2       # main, parse_header
    assert summary["unnamed_function_count"] == 2      # sub_401200, j_sub_401300
    assert summary["imported_function_count"] == 1     # puts (PLT stub), not "named"


def test_callsites_returns_local_hlil_assignment_and_pre_branch_condition(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    branch = _FakeHLILInstruction(
        "if (result == 2)",
        class_name="HighLevelILIf",
        condition="result == 2",
        expr_index=40,
        instr_index=40,
    )
    first_statement = _FakeHLILInstruction(
        "edx_1:eax_1 = sx.q(crt_rand())",
        class_name="HighLevelILVarInit",
        expr_index=32,
        instr_index=32,
    )
    first_sx = _FakeHLILInstruction(
        "sx.q(crt_rand())",
        class_name="HighLevelILSx",
        parent=first_statement,
        expr_index=31,
        instr_index=31,
    )
    first_call = _FakeHLILInstruction(
        "crt_rand()",
        class_name="HighLevelILCall",
        parent=first_sx,
        expr_index=30,
        instr_index=30,
    )
    second_statement = _FakeHLILInstruction(
        "eax_3, edx_2 = crt_rand()",
        class_name="HighLevelILVarInit",
        parent=branch,
        expr_index=42,
        instr_index=42,
    )
    second_call = _FakeHLILInstruction(
        "crt_rand()",
        class_name="HighLevelILCall",
        parent=second_statement,
        expr_index=41,
        instr_index=41,
    )
    callee = _FakeFunction(0x461746, "crt_rand")
    fn = _FakeFunction(0x412470, "bonus_pick_random_type")
    fn.basic_blocks = [_FakeBasicBlock(0x41249C, 0x4124D8)]
    fn.low_level_il = [
        [
            _FakeLLILInstruction(0x4124A0, _FakeConstPtr(0x461746), hlils=[first_call]),
            _FakeLLILInstruction(0x4124D1, _FakeConstPtr(0x461746), hlils=[second_call]),
        ]
    ]
    bv = _FakeBV(
        functions=[callee, fn],
        instruction_lengths={
            0x41249C: 2,
            0x41249E: 2,
            0x4124A0: 5,
            0x4124A5: 3,
            0x4124D1: 5,
            0x4124D6: 2,
        },
        disassembly={
            0x41249C: "mov eax, 0",
            0x41249E: "mov ebx, 0",
            0x4124A0: "call crt_rand",
            0x4124A5: "cmp eax, 0xd",
            0x4124D1: "call crt_rand",
            0x4124D6: "test al, 0x3f",
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance,
        "active",
        "crt_rand",
        within_identifiers=["bonus_pick_random_type"],
        context=2,
    )

    assert [row["caller_static"] for row in rows] == ["0x4124a5", "0x4124d6"]
    assert rows[0]["call_addr"] == "0x4124a0"
    assert rows[0]["instruction_length"] == 5
    assert rows[0]["call_index"] == 0
    assert rows[0]["within_query"] == "bonus_pick_random_type"
    assert rows[0]["hlil_statement"] == "edx_1:eax_1 = sx.q(crt_rand())"
    assert rows[0]["pre_branch_condition"] is None
    assert rows[1]["call_index"] == 1
    assert rows[1]["hlil_statement"] == "eax_3, edx_2 = crt_rand()"
    assert rows[1]["pre_branch_condition"] == "result == 2"
    assert [item["address"] for item in rows[0]["previous_instructions"]] == ["0x41249c", "0x41249e"]
    assert rows[0]["call_instruction"]["text"] == "call crt_rand"
    assert [item["address"] for item in rows[0]["next_instructions"][:1]] == ["0x4124a5"]


def test_callsites_finds_register_dest_call_via_code_ref_db(monkeypatch):
    # On stripped/kernel/MIPS targets a call's LLIL dest is a register/computed
    # value BN resolved via analysis and recorded in the code-ref DB (what xrefs
    # reads), NOT a literal const. callsites must agree with xrefs, not silently
    # drop the edge. Here the only call dest is a register, so the literal-const
    # match misses it and only the code-ref DB resolves it.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "target_fn")
    fn = _FakeFunction(0x412470, "caller_fn")
    fn.basic_blocks = [_FakeBasicBlock(0x4124A0, 0x4124A4)]
    fn.low_level_il = [[_FakeLLILInstruction(0x4124A0, _FakeReg("x8"))]]
    bv = _FakeBV(
        functions=[callee, fn],
        instruction_lengths={0x4124A0: 4},
        disassembly={0x4124A0: "blr x8"},
        code_refs={0x461746: [_FakeCodeRef(0x4124A0, fn)]},  # BN's DB knows the edge
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance, "active", "target_fn", within_identifiers=["caller_fn"], context=1)
    assert len(rows) == 1
    assert rows[0]["call_addr"] == "0x4124a0"
    assert rows[0]["caller_static"] == "0x4124a4"


def test_callsites_prefers_local_expression_over_broad_enclosing_hlil(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    branch = _FakeHLILInstruction(
        "if (config_fx_toggle != 0)",
        class_name="HighLevelILIf",
        condition="config_fx_toggle != 0",
        expr_index=100,
        instr_index=100,
    )
    broad_statement = _FakeHLILInstruction(
        "if (config_fx_toggle != 0)\nlong expression blob\nreturn",
        class_name="HighLevelILVarInit",
        parent=branch,
        expr_index=99,
        instr_index=99,
    )
    add_expr = _FakeHLILInstruction(
        "float.t(crt_rand() & 0xf) * 0.01 + 0.84",
        class_name="HighLevelILAdd",
        parent=broad_statement,
        expr_index=35,
        instr_index=9,
    )
    mul_expr = _FakeHLILInstruction(
        "float.t(crt_rand() & 0xf) * 0.01",
        class_name="HighLevelILMul",
        parent=add_expr,
        expr_index=34,
        instr_index=9,
    )
    cast_expr = _FakeHLILInstruction(
        "float.t(crt_rand() & 0xf)",
        class_name="HighLevelILIntToFloat",
        parent=mul_expr,
        expr_index=33,
        instr_index=9,
    )
    and_expr = _FakeHLILInstruction(
        "crt_rand() & 0xf",
        class_name="HighLevelILAnd",
        parent=cast_expr,
        expr_index=32,
        instr_index=9,
    )
    call_expr = _FakeHLILInstruction(
        "crt_rand()",
        class_name="HighLevelILCall",
        parent=and_expr,
        expr_index=31,
        instr_index=9,
    )
    callee = _FakeFunction(0x461746, "crt_rand")
    fn = _FakeFunction(0x427700, "fx_queue_add_random")
    fn.basic_blocks = [_FakeBasicBlock(0x427753, 0x427768)]
    fn.low_level_il = [[_FakeLLILInstruction(0x42775B, _FakeConstPtr(0x461746), hlils=[broad_statement, call_expr])]]
    bv = _FakeBV(
        functions=[callee, fn],
        instruction_lengths={
            0x427753: 5,
            0x427758: 3,
            0x42775B: 5,
            0x427760: 3,
        },
        disassembly={
            0x427753: "call helper",
            0x427758: "add esp, 0x4",
            0x42775B: "call crt_rand",
            0x427760: "and eax, 0xf",
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance,
        "active",
        "crt_rand",
        within_identifiers=["fx_queue_add_random"],
        context=2,
    )

    assert len(rows) == 1
    assert rows[0]["hlil_statement"] == "float.t(crt_rand() & 0xf) * 0.01 + 0.84"
    assert rows[0]["pre_branch_condition"] == "config_fx_toggle != 0"
    assert rows[0]["call_index"] == 0
    assert rows[0]["within_query"] == "fx_queue_add_random"


def test_callsites_within_file_scope_preserves_file_order_and_dedupes(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "crt_rand")
    alpha = _FakeFunction(0x401000, "alpha")
    alpha.basic_blocks = [_FakeBasicBlock(0x401010, 0x401016)]
    alpha.low_level_il = [[_FakeLLILInstruction(0x401010, _FakeConstPtr(0x461746))]]
    beta = _FakeFunction(0x402000, "beta")
    beta.basic_blocks = [_FakeBasicBlock(0x402020, 0x402026)]
    beta.low_level_il = [[_FakeLLILInstruction(0x402020, _FakeConstPtr(0x461746))]]
    bv = _FakeBV(
        functions=[callee, alpha, beta],
        instruction_lengths={0x401010: 5, 0x402020: 5},
        disassembly={0x401010: "call crt_rand", 0x402020: "call crt_rand"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance,
        "active",
        "crt_rand",
        within_identifiers=["beta", "alpha", "beta"],
        context=0,
    )

    assert [row["containing_function"]["name"] for row in rows] == ["beta", "alpha"]
    assert [row["caller_static"] for row in rows] == ["0x402025", "0x401015"]
    assert [row["within_query"] for row in rows] == ["beta", "alpha"]
    assert [row["call_index"] for row in rows] == [0, 0]


def test_callsites_ignores_indirect_calls_and_returns_null_context_when_unmapped(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "crt_rand")
    fn = _FakeFunction(0x500000, "fx_queue_add_random")
    fn.basic_blocks = [_FakeBasicBlock(0x500010, 0x50001A)]
    fn.low_level_il = [
        [
            _FakeLLILInstruction(0x500010, _FakeReg("eax")),
            _FakeLLILInstruction(0x500015, _FakeConstPtr(0x461746)),
        ]
    ]
    bv = _FakeBV(
        functions=[callee, fn],
        instruction_lengths={0x500010: 5, 0x500015: 5},
        disassembly={0x500010: "call eax", 0x500015: "call crt_rand"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance,
        "active",
        "crt_rand",
        within_identifiers=["fx_queue_add_random"],
        context=1,
    )

    assert len(rows) == 1
    assert rows[0]["call_addr"] == "0x500015"
    assert rows[0]["hlil_statement"] is None
    assert rows[0]["pre_branch_condition"] is None


def test_callsites_counts_tailcall_into_target(monkeypatch):
    # A tail-branch into the target (`return <addr>(...) __tailcall`, e.g. a
    # j_memcpy veneer) must be reported as a callsite -- xrefs and taint already
    # treat it as a call, so callsites must agree (#47).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "memcpy")
    fn = _FakeFunction(0x700000, "j_memcpy")
    fn.basic_blocks = [_FakeBasicBlock(0x700010, 0x700014)]
    fn.low_level_il = [[
        _FakeLLILInstruction(0x700010, _FakeConstPtr(0x461746), operation="LLIL_TAILCALL"),
    ]]
    bv = _FakeBV(
        functions=[callee, fn],
        instruction_lengths={0x700010: 4},
        disassembly={0x700010: "b #memcpy"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance,"active", "memcpy", within_identifiers=["j_memcpy"], context=1)

    assert len(rows) == 1
    assert rows[0]["call_addr"] == "0x700010"
    assert rows[0]["call_kind"] == "tailcall"


def test_callsites_marks_regular_call_kind(monkeypatch):
    # A normal bl/blx call is reported with call_kind 'call' (not tailcall).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "memcpy")
    fn = _FakeFunction(0x700000, "caller")
    fn.basic_blocks = [_FakeBasicBlock(0x700010, 0x700014)]
    fn.low_level_il = [[_FakeLLILInstruction(0x700010, _FakeConstPtr(0x461746))]]  # default LLIL_CALL
    bv = _FakeBV(
        functions=[callee, fn],
        instruction_lengths={0x700010: 4},
        disassembly={0x700010: "bl #memcpy"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance,"active", "memcpy", within_identifiers=["caller"], context=1)

    assert len(rows) == 1
    assert rows[0]["call_kind"] == "call"


def test_callsites_returns_null_for_coarse_only_hlil(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "crt_rand")
    broad_statement = _FakeHLILInstruction(
        "if (x)\nwhole function blob\nreturn",
        class_name="HighLevelILVarInit",
        expr_index=10,
        instr_index=10,
    )
    fn = _FakeFunction(0x600000, "coarse")
    fn.basic_blocks = [_FakeBasicBlock(0x600010, 0x600016)]
    fn.low_level_il = [[_FakeLLILInstruction(0x600010, _FakeConstPtr(0x461746), hlils=[broad_statement])]]
    bv = _FakeBV(
        functions=[callee, fn],
        instruction_lengths={0x600010: 5},
        disassembly={0x600010: "call crt_rand"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance,
        "active",
        "crt_rand",
        within_identifiers=["coarse"],
        context=1,
    )

    assert len(rows) == 1
    assert rows[0]["hlil_statement"] is None
    assert rows[0]["pre_branch_condition"] is None


def test_callsites_filters_placeholder_pre_branch_condition(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    branch = _FakeHLILInstruction(
        "do while (not(cond:0_1))",
        class_name="HighLevelILDoWhile",
        condition="not(cond:0_1)",
        expr_index=50,
        instr_index=50,
    )
    statement = _FakeHLILInstruction(
        "eax_1 = crt_rand()",
        class_name="HighLevelILVarInit",
        parent=branch,
        expr_index=51,
        instr_index=51,
    )
    call = _FakeHLILInstruction(
        "crt_rand()",
        class_name="HighLevelILCall",
        parent=statement,
        expr_index=52,
        instr_index=52,
    )
    callee = _FakeFunction(0x461746, "crt_rand")
    fn = _FakeFunction(0x700000, "placeholder_cond")
    fn.basic_blocks = [_FakeBasicBlock(0x700010, 0x700016)]
    fn.low_level_il = [[_FakeLLILInstruction(0x700010, _FakeConstPtr(0x461746), hlils=[call])]]
    bv = _FakeBV(
        functions=[callee, fn],
        instruction_lengths={0x700010: 5},
        disassembly={0x700010: "call crt_rand"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance,
        "active",
        "crt_rand",
        within_identifiers=["placeholder_cond"],
        context=1,
    )

    assert rows[0]["pre_branch_condition"] is None


def test_function_evidence_reports_calls_arguments_and_thunk_candidate(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "send_message")
    caller = _FakeFunction(0x412470, "build_response")
    call_expr = _FakeHLILInstruction(
        "send_message(6, &response)",
        class_name="HighLevelILCall",
        expr_index=30,
        instr_index=30,
    )
    call_expr.params = [6, "&response"]
    call_insn = _FakeLLILInstruction(0x4124A0, _FakeConstPtr(0x461746), hlils=[call_expr])
    call_insn.params = [_FakeReg("r0"), _FakeConstPtr(6), _FakeReg("r2")]
    caller.basic_blocks = [_FakeBasicBlock(0x41249C, 0x4124A8)]
    caller.low_level_il = [[call_insn]]
    thunk = _FakeFunction(0x500000, "j_send_message")
    thunk.basic_blocks = [_FakeBasicBlock(0x500000, 0x500004)]
    thunk.low_level_il = [[_FakeLLILInstruction(0x500000, _FakeConstPtr(0x461746), operation="LLIL_JUMP")]]
    bv = _FakeBV(
        functions=[callee, caller, thunk],
        instruction_lengths={0x41249C: 2, 0x41249E: 2, 0x4124A0: 4, 0x4124A4: 4},
        disassembly={
            0x41249C: "mov r1, #6",
            0x41249E: "mov r2, response",
            0x4124A0: "bl send_message",
            0x4124A4: "pop {pc}",
            0x500000: "b send_message",
        },
    )
    # _function_evidence now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_evidence("active", "build_response", context=1)
    call = result["calls"][0]

    assert call["direct"] is True
    assert call["target"]["function"]["name"] == "send_message"
    # primary args come from the single matched HLIL call; no merged/source-tagged noise
    assert call["argument_source"] == "hlil"
    assert [arg["text"] for arg in call["arguments"]] == ["6", "&response"]
    # other IL layers are quarantined as candidates, JSON-only
    assert any(c["source"] == "llil" for c in call["argument_candidates"])
    assert call["previous_instructions"][0]["text"] == "mov r2, response"

    thunk_result = instance._function_evidence("active", "j_send_message", context=0)
    assert thunk_result["thunk"]["is_candidate"] is True
    assert thunk_result["thunk"]["target"]["function"]["name"] == "send_message"


def test_function_evidence_resolves_pointer_constant_arguments(monkeypatch):
    # ILX #2: append(&var, 0x2a4f4) should annotate the constant with "4" [.rodata].
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x154F4, "append")
    caller = _FakeFunction(0x1A36C, "createBTService")
    call_expr = _FakeHLILInstruction(
        "append(&var_38, 0x2a4f4)", class_name="HighLevelILCall", expr_index=10, instr_index=10
    )
    call_expr.params = ["&var_38", "0x2a4f4"]
    call_insn = _FakeLLILInstruction(0x1A38E, _FakeConstPtr(0x154F4), hlils=[call_expr])
    caller.basic_blocks = [_FakeBasicBlock(0x1A38E, 0x1A392)]
    caller.low_level_il = [[call_insn]]
    bv = _FakeBV(
        functions=[callee, caller],
        instruction_lengths={0x1A38E: 4},
        disassembly={0x1A38E: "blx append"},
        sections={".rodata": _FakeSection(".rodata", 0x2A000, 0x2B000)},
        segments={0x2A4F4: _FakeSegment(readable=True)},
        memory={0x2A4F4: b"4\x00"},
    )
    # _function_evidence now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_evidence("active", "createBTService", context=0)
    call = result["calls"][0]
    constant = next(arg for arg in call["arguments"] if arg["text"] == "0x2a4f4")
    assert constant["resolved"]["string"] == "4"
    assert constant["resolved"]["section"] == ".rodata"


def test_function_evidence_does_not_merge_unrelated_hlil_call_args(monkeypatch):
    # ILX #3: one LLIL call mapping to two HLIL calls must not borrow the other's args.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x16520, "getopt_long")
    caller = _FakeFunction(0x17140, "main")
    real = _FakeHLILInstruction(
        'getopt_long(argc, argv, "hb:d:")', class_name="HighLevelILCall", expr_index=5, instr_index=5
    )
    real.params = ["argc", "argv", '"hb:d:"']
    real.address = 0x17188
    unrelated = _FakeHLILInstruction(
        '__android_log_print(4, "aa accessory", x)',
        class_name="HighLevelILCall",
        expr_index=9,
        instr_index=9,
    )
    unrelated.params = ["4", '"aa accessory"', "x"]
    unrelated.address = 0x172A0
    call_insn = _FakeLLILInstruction(0x17188, _FakeConstPtr(0x16520), hlils=[real, unrelated])
    caller.basic_blocks = [_FakeBasicBlock(0x17140, 0x17200)]
    caller.low_level_il = [[call_insn]]
    bv = _FakeBV(
        functions=[callee, caller],
        instruction_lengths={0x17188: 4},
        disassembly={0x17188: "blx getopt_long"},
    )
    # _function_evidence now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_evidence("active", "main", context=0)
    call = result["calls"][0]
    assert call["argument_source"] == "hlil"
    assert [arg["text"] for arg in call["arguments"]] == ["argc", "argv", '"hb:d:"']
    assert all("aa accessory" not in arg["text"] for arg in call["arguments"])
    assert any(c["text"] == '"aa accessory"' for c in call["argument_candidates"])


def test_pointer_table_normalizes_thumb_function_pointers(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    target = _FakeFunction(0x401000, "handler")
    table = (0x401001).to_bytes(4, "little") + (0x402000).to_bytes(4, "little")

    class _ThumbTolerantBV(_FakeBV):
        def get_function_at(self, address: int):
            if int(address) == 0x401001:
                return target
            return super().get_function_at(address)

    bv = _ThumbTolerantBV(functions=[target], arch=_FakeArch(name="armv7"), memory={0x3000: table})
    # _pointer_table now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._pointer_table("active", "0x3000", entries=2)

    assert result["entries"][0]["value"] == "0x401001"
    assert result["entries"][0]["target"]["normalized"] == "0x401000"
    assert result["entries"][0]["target"]["thumb_adjusted"] is True
    assert result["entries"][0]["target"]["function"]["name"] == "handler"
    assert result["entries"][0]["target"]["function"]["exact_start"] is True
    assert result["entries"][0]["target"]["context"]["address"] == "0x401000"
    assert result["entries"][1]["target"]["function"] is None
    assert result["entries"][1]["target"]["plausible"] is False


def test_pointer_table_does_not_thumb_normalize_non_arm_pointers(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    target = _FakeFunction(0x401000, "handler")
    table = (0x401001).to_bytes(4, "little")

    class _OddTolerantBV(_FakeBV):
        def get_function_at(self, address: int):
            if int(address) == 0x401001:
                return target
            return super().get_function_at(address)

    bv = _OddTolerantBV(functions=[target], arch=_FakeArch(name="x86"), memory={0x3000: table})
    # _pointer_table now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._pointer_table("active", "0x3000", entries=1)
    target_info = result["entries"][0]["target"]

    assert target_info["raw"] == "0x401001"
    assert target_info["normalized"] == "0x401001"
    assert target_info["thumb_adjusted"] is False
    assert target_info["function"]["name"] == "handler"
    assert target_info["function"]["exact_start"] is False
    assert target_info["function"]["offset"] == "0x1"
    assert target_info["context"]["address"] == "0x401001"
    assert any("inside functions" in warning for warning in result["warnings"])


def test_pointer_table_downgrades_inline_scalar_fields(monkeypatch):
    """A mixed record {function ptr, uint8 flag, ptr} read at a fixed stride must
    not count the inline scalar as a failed pointer resolution; only genuine
    pointer slots feed the 'do not resolve' warning (#170)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    handler = _FakeFunction(0x401000, "handler")
    # entry0: function pointer (plausible); entry1: a uint8 flag = 5 read as a
    # pointer-sized slot (inline scalar); entry2: a large unmapped value (a
    # genuine failed pointer slot that SHOULD still be counted).
    table = (
        (0x401000).to_bytes(4, "little")
        + (5).to_bytes(4, "little")
        + (0xDEADBEEF).to_bytes(4, "little")
    )
    bv = _FakeBV(functions=[handler], arch=_FakeArch(name="x86", address_size=4),
                 memory={0x3000: table})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._pointer_table("active", "0x3000", entries=3)
    rows = result["entries"]
    assert rows[0]["plausible"] is True
    assert rows[1]["likely_scalar"] is True and rows[1]["plausible"] is False
    assert rows[2]["likely_scalar"] is False and rows[2]["plausible"] is False
    warnings = " ".join(result["warnings"])
    assert "1 non-null entries do not resolve to mapped addresses" in warnings  # only entry2
    assert "inline scalar fields" in warnings                                   # entry1 noted



def test_function_evidence_marks_plt_stubs_as_thunk_candidates(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    plt = _FakeFunction(0x404020, "puts@plt")
    plt.basic_blocks = [_FakeBasicBlock(0x404020, 0x404026)]
    plt.low_level_il = [[_FakeLLILInstruction(0x404020, _FakeReg("rax"), operation="LLIL_JUMP")]]
    bv = _FakeBV(
        functions=[plt],
        sections={".plt.got": _FakeSection(".plt.got", 0x404020, 0x404100)},
        disassembly={0x404020: "jmp qword ptr [rip+0x2000]"},
    )
    # _function_evidence now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_evidence("active", "puts@plt", context=0)

    assert result["thunk"]["is_candidate"] is True
    assert "PLT" in result["thunk"]["reason"]


def test_pointer_table_warns_when_start_looks_like_code_not_table(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    memory = {0x64EA0: b"\xfb\x6b\xdb\xb2\x00\x2b\x00\xf0"}
    bv = _FakeBV(
        sections={".text": _FakeSection(".text", 0x64000, 0x65000)},
        segments={0x64EA0: _FakeSegment(readable=True, executable=True)},
        memory=memory,
    )
    # _pointer_table now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._pointer_table("active", "0x64ea0", entries=2)

    assert all(entry["plausible"] is False for entry in result["entries"])
    assert any("executable segment" in warning for warning in result["warnings"])
    assert any("low confidence" in warning for warning in result["warnings"])


def test_pointer_table_errors_on_unmapped_base(monkeypatch):
    """`evidence table` at an unmapped address must error like `bn read`, not
    return exit 0 with 16 fabricated readable:false slots and empty warnings
    (#119)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # Memory mapped elsewhere (so reads model real BN: b"" outside it); 0xdeadbeef
    # has no segment/section and is unreadable -> genuinely unmapped.
    bv = _FakeBV(memory={0x1000: b"\x00\x00\x00\x00"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(RuntimeError, match="0xdeadbeef.*not mapped"):
        instance._pointer_table("active", "0xdeadbeef", entries=16)


def test_pointer_table_for_view_warns_on_unmapped_base_without_erroring(monkeypatch):
    """The shared helper (used by message-lens / init-array windows) must FLAG
    an unmapped base instead of silently fabricating slots, but it must not
    abort the surrounding scan -- only the top-level command errors (#119)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(memory={0x1000: b"\x00\x00\x00\x00"})  # 0xdeadbeef unreadable -> unmapped
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    table = bridge.read_evidence._pointer_table_for_view(
        instance.ctx, bv, 0xDEADBEEF, entries=4, stride_size=4,
    )

    assert table["context"]["kind"] == "unmapped"
    assert any("unmapped" in warning for warning in table["warnings"])


def test_message_lens_reports_true_total_and_flags_truncation(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # 5 strings match the needle; with limit=2 only 2 rich matches come back,
    # but the reported total must be the honest 5 with truncated=True (issue #13).
    strings = [_FakeStringRef(0x1000 + i * 0x20, 9, f"Evt{i}_token") for i in range(5)]
    bv = _FakeBV(arch=_FakeArch(name="armv7"), strings=strings)
    # _message_lens now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._message_lens("active", "token", limit=2)

    assert result["count"] == 2          # only `limit` rich matches returned
    assert len(result["matches"]) == 2
    assert result["total"] == 5          # but the count reported is honest
    assert result["truncated"] is True


def test_message_lens_not_truncated_when_all_matches_fit(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    strings = [_FakeStringRef(0x1000 + i * 0x20, 9, f"Evt{i}_token") for i in range(3)]
    bv = _FakeBV(arch=_FakeArch(name="armv7"), strings=strings)
    # _message_lens now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._message_lens("active", "token", limit=20)

    assert result["count"] == result["total"] == 3
    assert result["truncated"] is False


def test_message_lens_metadata_window_stops_at_obvious_non_pointer(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    builder = _FakeFunction(0x586A2, "build_type_name")
    # First pointer resolves to code, then ASCII bytes "N6co" as little-endian
    # integer. The lens should include the bad entry as evidence and stop.
    memory = {
        0x6000: (
            (0x586A3).to_bytes(4, "little")
            + int.from_bytes(b"N6co", "little").to_bytes(4, "little")
            + int.from_bytes(b"mmon", "little").to_bytes(4, "little")
        )
    }
    bv = _FakeBV(
        functions=[builder],
        arch=_FakeArch(name="armv7"),
        strings=[_FakeStringRef(0x175BE4, 23, "N6common12HeadUnitInfoE")],
        data_refs={0x175BE4: [0x6008]},
        memory=memory,
    )
    # _message_lens now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._message_lens("active", "HeadUnitInfo", limit=5, table_entries=8)
    table = result["matches"][0]["metadata_table_windows"][0]

    assert len(table["entries"]) == 2
    assert table["entries"][1]["target"]["status"] == "unmapped"
    assert any("stopped after" in warning for warning in table["warnings"])


def test_init_arrays_summarizes_constructor_pointer_sections(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    ctor = _FakeFunction(0x401000, "global_ctor")
    table = (0x401001).to_bytes(4, "little") + (0x402000).to_bytes(4, "little")
    bv = _FakeBV(
        functions=[ctor],
        arch=_FakeArch(name="armv7"),
        sections={
            ".init_array": _FakeSection(".init_array", 0x5000, 0x5008),
            ".data": _FakeSection(".data", 0x6000, 0x6010),
        },
        memory={0x5000: table},
    )
    # _init_arrays now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._init_arrays("active", limit=4)

    assert result["pointer_size"] == 4
    assert len(result["sections"]) == 1
    section = result["sections"][0]
    assert section["name"] == ".init_array"
    assert section["total_entries"] == 2
    assert section["table"]["entries"][0]["target"]["function"]["name"] == "global_ctor"


def test_scan_for_calls_to_finds_llil_calls(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    func_sym = sys.modules["binaryninja"].Symbol(
        sys.modules["binaryninja"].SymbolType.FunctionSymbol, 0x10000, "my_func"
    )
    func_sym.short_name = "my_func"

    insn_call = _FakeLLILInstruction(0x10010, _FakeConstPtr(0x20000))
    insn_tailcall = _FakeLLILInstruction(
        0x10020, _FakeConstPtr(0x20000), operation="LLIL_TAILCALL"
    )
    insn_other = _FakeLLILInstruction(0x10030, _FakeConstPtr(0x30000))

    fn = _FakeFunction(0x10000, "my_func")
    fn.low_level_il = [[insn_call, insn_tailcall, insn_other]]

    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._scan_for_calls_to(bv, 0x20000)

    assert len(result) == 2
    addresses = [int(r["address"], 16) for r in result]
    assert 0x10010 in addresses
    assert 0x10020 in addresses
    assert 0x10030 not in addresses


def test_scan_for_calls_to_deduplicates_same_address(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    insn = _FakeLLILInstruction(0x10010, _FakeConstPtr(0x20000))
    fn = _FakeFunction(0x10000, "my_func")
    fn.low_level_il = [[insn, insn]]

    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._scan_for_calls_to(bv, 0x20000)

    assert len(result) == 1


def test_callsites_requires_refresh_when_quick_loaded(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # Quick-loaded: "Function not found" would misattribute missing analysis
    # to a typo. Refuse with a directive instead.
    bridge._quick_loaded_views.add(bv)
    with pytest.raises(RuntimeError, match="loaded with --quick"):
        _callsites_items(instance,None, "strcpy", within_identifiers=["main"])
    bridge._quick_loaded_views.discard(bv)


def test_pointer_table_read_width_tracks_substride(monkeypatch):
    """`evidence table --stride 4` on 8-byte-aligned data must read 4 bytes wide,
    so odd slots read the (zero) high half instead of an 8-byte window overlapping
    the next pointer -> 0x40018000000000 garbage flagged [implausible] (#225)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    table = ((0x40c370).to_bytes(8, "little")
             + (0x400180).to_bytes(8, "little")
             + (0x40ca80).to_bytes(8, "little"))
    bv = _FakeBV(arch=_FakeArch(name="x86_64", address_size=8), memory={0x40b580: table})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._pointer_table("active", "0x40b580", entries=6, stride="4")
    assert result["read_width"] == 4
    vals = [e.get("value") for e in result["entries"]]
    assert vals[0] == "0x40c370" and vals[2] == "0x400180" and vals[4] == "0x40ca80"
    assert vals[1] == "0x0" and vals[3] == "0x0"            # zero high halves, not garbage
    assert "0x40018000000000" not in vals                   # no overlapping read

    # default (stride == pointer size) still reads 8-byte pointers
    r8 = instance._pointer_table("active", "0x40b580", entries=3, stride="8")
    assert r8["read_width"] == 8
    assert [e.get("value") for e in r8["entries"]] == ["0x40c370", "0x400180", "0x40ca80"]

    # explicit --width overrides the stride-derived width
    rw = instance._pointer_table("active", "0x40b580", entries=2, stride="8", width="4")
    assert rw["read_width"] == 4
    assert rw["entries"][0]["value"] == "0x40c370"


def test_message_lens_excludes_dynstr_and_resolves_rtti(monkeypatch):
    """On a symbol-retaining binary, evidence message must exclude the noisy
    .dynstr symbol-name matches and instead resolve the real RTTI data symbols
    (_ZTV/_ZTI/_ZTS<type>) with xrefs + a hint (#194)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    # the only string match is a .dynstr symbol-name string (noise)
    dynstr_str = _FakeStringRef(0x9000, 16, "_ZTVN5TCLAP3ArgE")
    # the real RTTI vtable data symbol the lens should surface
    vtable_sym = fake_bn.Symbol(fake_bn.SymbolType.DataSymbol, 0xA000, "_ZTVN5TCLAP3ArgE")
    vtable_sym.raw_name = "_ZTVN5TCLAP3ArgE"
    user = _FakeFunction(0x401000, "user")
    bv = _FakeBV(
        functions=[user],
        strings=[dynstr_str],
        symbols=[vtable_sym],
        sections={".dynstr": _FakeSection(".dynstr", 0x9000, 0x9100),
                  ".data.rel.ro": _FakeSection(".data.rel.ro", 0xA000, 0xB000)},
        code_refs={0xA000: [_FakeCodeRef(0x401010, user)]},
        segments={0x401010: _FakeSegment(readable=True, executable=True),
                  0xA000: _FakeSegment(readable=True, writable=True)},
        memory={0xA000: (0).to_bytes(8, "little") + (0xB100).to_bytes(8, "little")},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._message_lens("active", "N5TCLAP3ArgE", limit=5, table_entries=2)
    assert result["dynstr_excluded"] == 1
    assert result["count"] == 0                       # the lone .dynstr match was excluded
    rtti = result["rtti_symbols"]
    assert any(s["kind"] == "vtable" and s["symbol"] == "_ZTVN5TCLAP3ArgE" for s in rtti)
    vt = next(s for s in rtti if s["kind"] == "vtable")
    assert vt["address"] == "0xa000"
    assert len(vt["xrefs"]["code_refs"]) == 1
    assert result["hints"]                            # dynstr + rtti hints present


