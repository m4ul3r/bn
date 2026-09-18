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


def test_function_name_summary_counts_callable_got_slots_478(monkeypatch):
    """#478: callable GOT slots (JUMP_SLOT-relocated) with no recovered PLT-stub
    function object still count toward imported_function_count -- otherwise a target
    whose PLT-stub recovery failed reports zero imports and reads as stripped/static.
    A slot whose name DOES have an imported function object is unioned by name (not
    double-counted); GLOB_DAT data slots do not count; the bv.functions partition is
    unaffected."""
    bridge = _load_bridge(monkeypatch)
    fake_bn = sys.modules["binaryninja"]
    JS = fake_bn.RelocationType.ELFJumpSlotRelocationType
    GD = fake_bn.RelocationType.ELFGlobalRelocationType

    class _ImpSym:  # faithful imported-function symbol: .type.name + .raw_name
        def __init__(self, raw):
            self.type = type("_T", (), {"name": "ImportedFunctionSymbol"})()
            self.raw_name = raw

    # memcpy has BOTH a recovered PLT-stub function object AND its JUMP_SLOT slot ->
    # must count once. strcpy/recv are symptom slots (JUMP_SLOT, no function object)
    # -> count. stdout is a GLOB_DAT data slot -> never counts.
    memcpy_fn = _FakeFunction(0x1200, "memcpy")
    memcpy_fn.symbol = _ImpSym("memcpy")
    memcpy = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x3000, "memcpy")
    strcpy = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x3008, "strcpy")
    recv = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x3018, "recv")
    stdout = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x3010, "stdout")
    bv = _FakeBV(
        functions=[_FakeFunction(0x1000, "main"), _FakeFunction(0x1100, "sub_1100"), memcpy_fn],
        symbols=[memcpy, strcpy, recv, stdout],
        relocations={
            0x3000: [_FakeReloc(JS, memcpy)],
            0x3008: [_FakeReloc(JS, strcpy)],
            0x3018: [_FakeReloc(JS, recv)],
            0x3010: [_FakeReloc(GD, stdout)],
        },
    )
    summary = bridge._function_name_summary(bv)
    assert summary["function_count"] == 3
    assert summary["named_function_count"] == 1        # main (memcpy_fn is imported, not named)
    assert summary["unnamed_function_count"] == 1      # sub_1100
    # memcpy (object + slot) counts once; strcpy + recv add 2; stdout (GLOB_DAT) excluded.
    assert summary["imported_function_count"] == 3


def test_callable_import_slot_names_excludes_self_defined_478(monkeypatch):
    """#478 defense-in-depth: a PIC .so's own exported function can also carry a
    JUMP_SLOT GOT slot; that self-reference must NOT inflate the callable-import
    set, mirroring the imports-listing #202 self-reference filter."""
    bridge = _load_bridge(monkeypatch)
    read_misc = bridge.read_misc
    fake_bn = sys.modules["binaryninja"]
    JS = fake_bn.RelocationType.ELFJumpSlotRelocationType

    own_def = fake_bn.Symbol(fake_bn.SymbolType.FunctionSymbol, 0x1500, "own_api")
    own_slot = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x3000, "own_api")
    memcpy_slot = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x3008, "memcpy")
    bv = _FakeBV(symbols=[own_def, own_slot, memcpy_slot], relocations={
        0x3000: [_FakeReloc(JS, own_slot)],
        0x3008: [_FakeReloc(JS, memcpy_slot)],
    })
    assert read_misc._callable_import_slot_names(bv) == {"memcpy"}


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


def test_callsites_keeps_call_absent_from_disasm_index_with_a_reason(monkeypatch):
    """#816: a call the LLIL/code-ref union confirms but this function's structured
    disassembly walk never decoded (a decode hole) used to be DROPPED -- no row, no
    reason -- so `callsites` reported fewer sites than the `xrefs` evidence it is
    meant to agree with, and an agent could not tell a hole from "not called". The
    site's identity is known whatever the disassembly sweep managed, so emit the row
    with a null context plus a machine-readable reason (the `hlil_statement_reason`
    shape)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "target_fn")
    fn = _FakeFunction(0x412470, "caller_fn")
    fn.basic_blocks = [_FakeBasicBlock(0x4124A0, 0x4124AE)]
    fn.low_level_il = [[
        _FakeLLILInstruction(0x4124A0, _FakeConstPtr(0x461746)),
        _FakeLLILInstruction(0x4124A6, _FakeConstPtr(0x461746)),
    ]]
    bv = _FakeBV(
        functions=[callee, fn],
        instruction_lengths={0x4124A0: 5, 0x4124A5: 1, 0x4124A6: 5, 0x4124AB: 3},
        disassembly={
            0x4124A0: "call target_fn",
            0x4124A5: "nop",
            # 0x4124A6 is the hole: the walk decodes nothing here, so it emits no entry.
            0x4124AB: "test al, 0x3f",
        },
        code_refs={0x461746: [
            _FakeCodeRef(0x4124A0, fn),
            _FakeCodeRef(0x4124A6, fn),
        ]},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._callsites(None, "target_fn", within_identifiers=["caller_fn"], context=1)

    assert [row["call_addr"] for row in result["items"]] == ["0x4124a0", "0x4124a6"]
    kept = result["items"][1]
    # The identity of the site stays usable even though its context is missing.
    assert kept["callee"] == {"name": "target_fn", "address": "0x461746"}
    assert kept["containing_function"] == {"name": "caller_fn", "address": "0x412470"}
    assert kept["call_kind"] == "call"
    assert kept["instruction_length"] == 5
    assert kept["caller_static"] == "0x4124ab"
    assert kept["call_index"] == 1
    # Context is null by evidence, and the row says why.
    assert kept["call_instruction"] is None
    assert kept["previous_instructions"] == []
    assert kept["next_instructions"] == []
    assert kept["disasm_context_reason"] == "no_structured_disasm_entry"
    # A localized row is untouched: real context, no reason.
    assert result["items"][0]["disasm_context_reason"] is None
    assert result["items"][0]["call_instruction"] == {"address": "0x4124a0", "text": "call target_fn"}
    assert result["items"][0]["next_instructions"] == [{"address": "0x4124a5", "text": "nop"}]
    # Counts stay honest: callsites must not report fewer sites than xrefs for the
    # same callee (the pre-fix drop made this 2 vs 1).
    assert result["total"] == 2
    assert instance._xrefs(None, "target_fn")["code_ref_count"] == result["total"]


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


def test_callsites_empty_scope_discovers_all_callers_and_dedupes(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x403120, "checksum_update")
    alpha = _FakeFunction(0x401000, "alpha")
    alpha.basic_blocks = [_FakeBasicBlock(0x401010, 0x401016)]
    alpha.low_level_il = [[_FakeLLILInstruction(0x401010, _FakeConstPtr(0x403120))]]
    beta = _FakeFunction(0x402000, "beta")
    beta.basic_blocks = [_FakeBasicBlock(0x402020, 0x402026)]
    beta.low_level_il = [[_FakeLLILInstruction(0x402020, _FakeConstPtr(0x403120))]]
    bv = _FakeBV(
        functions=[callee, beta, alpha],
        instruction_lengths={0x401010: 5, 0x402020: 5},
        disassembly={
            0x401010: "call checksum_update",
            0x402020: "call checksum_update",
        },
        code_refs={
            0x403120: [
                _FakeCodeRef(0x402020, beta),
                _FakeCodeRef(0x401010, alpha),
                _FakeCodeRef(0x401011, alpha),
            ]
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(
        instance,
        "active",
        "checksum_update",
        within_identifiers=[],
        context=0,
    )

    assert [row["containing_function"]["name"] for row in rows] == ["alpha", "beta"]
    assert [row["within_query"] for row in rows] == ["alpha", "beta"]



def test_callsites_stops_after_page_lookahead_on_high_fan_in(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x5000, "sink")
    callers = [
        (f"caller_{index}", _FakeFunction(0x1000 + index * 0x10, f"caller_{index}"))
        for index in range(10)
    ]
    bv = _FakeBV(functions=[callee, *(function for _, function in callers)])
    visited = []

    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.ctx, "_find_function", lambda *args, **kwargs: callee)
    monkeypatch.setattr(instance.ctx, "_same_name_stub_functions", lambda *args: [])
    monkeypatch.setattr(
        bridge.read_listing,
        "_all_caller_functions",
        lambda *args: callers,
    )

    def one_callsite(ctx, view, target, function, **kwargs):
        visited.append(function.name)
        return [{"call_addr": hex(function.start)}]

    monkeypatch.setattr(
        bridge.read_listing, "_callsites_within_function", one_callsite
    )

    result = instance._callsites(
        "active", "sink", within_identifiers=[], context=0, limit=2
    )

    assert visited == ["caller_0", "caller_1", "caller_2"]
    assert len(result["items"]) == 2
    assert result["has_more"] is True
    assert result["total"] is None
    assert result["total_lower_bound"] == 3
    assert result["scan_truncated"] is True
    assert result["callers_scanned"] == 3
    assert result["caller_total"] == 10


def test_callsites_returns_empty_for_unreferenced_import_symbol(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]
    symbol = fake_bn.Symbol(
        fake_bn.SymbolType.ImportAddressSymbol, 0x3000, "receive_record"
    )
    bv = _FakeBV(symbols=[symbol], code_refs={})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._callsites(
        "active", "receive_record", within_identifiers=[], context=0, limit=10
    )

    assert result["items"] == []
    assert result["total"] == 0
    assert result["callee_symbol_only"] is True


def test_callsites_enumerates_import_callers_absent_from_the_code_ref_db(monkeypatch):
    """#816: `xrefs` covers an import BN recorded no code ref for with a bounded
    LLIL call scan (#622), but callsites enumerated its callers from the code-ref
    DB alone -- so the same callee came back "no callers" here while `xrefs`
    reported a confirmed call. That is the silent drop this op is not allowed to
    have, on the path where it is hardest to notice: an empty list."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]
    symbol = fake_bn.Symbol(
        fake_bn.SymbolType.ImportAddressSymbol, 0x461746, "receive_record"
    )
    symbol.short_name = "receive_record"
    caller = _FakeFunction(0x412470, "handle_packet")
    caller.basic_blocks = [_FakeBasicBlock(0x4124A0, 0x4124AE)]
    caller.low_level_il = [[
        _FakeLLILInstruction(0x4124A0, _FakeConstPtr(0x461746)),
        _FakeLLILInstruction(0x4124A6, _FakeConstPtr(0x461746)),
    ]]
    bv = _FakeBV(
        functions=[caller],
        symbols=[symbol],
        code_refs={},
        instruction_lengths={0x4124A0: 5, 0x4124A5: 1, 0x4124A6: 5},
        disassembly={
            0x4124A0: "call receive_record",
            0x4124A5: "nop",
            0x4124A6: "call receive_record",
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._callsites(
        "active", "receive_record", within_identifiers=[], context=1, limit=10
    )

    assert [row["call_addr"] for row in result["items"]] == ["0x4124a0", "0x4124a6"]
    assert {row["containing_function"]["name"] for row in result["items"]} == {"handle_packet"}
    # Both ops answer the same question on the same view, and now agree.
    assert instance._xrefs(None, "receive_record")["code_ref_count"] == result["total"] == 2
    assert result["callee_symbol_only"] is True
    # A complete scan claims nothing it cannot back.
    assert result["caller_scan_truncated"] is False
    assert result["caller_scan_note"] is None


def test_callsites_discloses_a_capped_caller_scan(monkeypatch):
    """#816/#622: when the caller ENUMERATION is itself partial, every count under
    it is a lower bound. Both consumers must say so -- JSON via
    `caller_scan_truncated`/`caller_scan_note` and a `null` total, text via the
    reason -- because a capped scan returning no rows is exactly the case that
    would otherwise read as proof the callee is never called."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]
    symbol = fake_bn.Symbol(
        fake_bn.SymbolType.ImportAddressSymbol, 0x461746, "receive_record"
    )
    symbol.short_name = "receive_record"
    unrelated = _FakeFunction(0x400000, "unrelated")
    unrelated.low_level_il = [[_FakeLLILInstruction(0x400010, _FakeConstPtr(0x999999))]]
    caller = _FakeFunction(0x412470, "handle_packet")
    caller.low_level_il = [[_FakeLLILInstruction(0x4124A0, _FakeConstPtr(0x461746))]]
    bv = _FakeBV(functions=[unrelated, caller], symbols=[symbol], code_refs={})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.read_xrefs, "SCAN_CALLS_MAX_FUNCS", 1)
    from bn.formatters import _render_callsites_text

    result = instance._callsites(
        "active", "receive_record", within_identifiers=[], context=1, limit=10
    )

    assert result["items"] == []
    assert result["has_more"] is False
    assert result["total"] is None
    assert result["total_lower_bound"] == 0
    # The row scan itself was not capped: the caller list below it was, and the
    # two are separately disclosed.
    assert result["scan_truncated"] is False
    assert result["caller_scan_truncated"] is True
    assert result["caller_total"] is None
    assert isinstance(result["caller_scan_note"], str) and result["caller_scan_note"]

    text = _render_callsites_text(result)
    assert "the caller scan was incomplete" in text
    assert "absence is not established" in text


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
    # #673: the thunk/veneer guard now only flags EXTERNAL branch targets --
    # mark this callee as an import so `j_send_message` below still reads as a
    # legitimate veneer, not a local tail-call.
    callee.symbol = _FakeSymbol("ImportedFunctionSymbol")
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


def test_function_evidence_resolves_mid_function_address(monkeypatch):
    # #626: a sink address reported by taint/trace lands mid-callee. `evidence
    # function` must resolve an interior address to the containing function --
    # the same #193 Part 4 contract decompile/info/il already honor -- so the
    # sink feeds straight into the next command instead of hitting a dead end.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "parse_packet")
    fn.basic_blocks = [_FakeBasicBlock(0x401000, 0x401040)]  # spans 0x401000..0x401040
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_evidence("active", "0x401010", context=0)

    assert result["function"]["name"] == "parse_packet"
    assert result["function"]["address"] == "0x401000"
    assert result["resolved_from"] == {"requested_address": "0x401010", "offset": "+0x10"}

    # Exact start (and, by construction, a name identifier) carries no annotation.
    exact = instance._function_evidence("active", "0x401000", context=0)
    assert exact["function"]["address"] == "0x401000"
    assert "resolved_from" not in exact


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
    # #476: the unrelated NEIGHBOR call (address 0x172A0 != this callsite 0x17188) must
    # NOT leak into the candidate list either -- previously it was quarantined there,
    # which let an auditor mis-attribute another call's value to this one.
    assert all("aa accessory" not in c["text"] for c in call["argument_candidates"])


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

    assert result["kind"] == "pointer_table" and result["total"] == 2  # #275
    assert result["items"][0]["value"] == "0x401001"
    assert result["items"][0]["target"]["normalized"] == "0x401000"
    assert result["items"][0]["target"]["thumb_adjusted"] is True
    assert result["items"][0]["target"]["function"]["name"] == "handler"
    assert result["items"][0]["target"]["function"]["exact_start"] is True
    assert result["items"][0]["target"]["context"]["address"] == "0x401000"
    assert result["items"][1]["target"]["function"] is None
    assert result["items"][1]["target"]["plausible"] is False


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
    target_info = result["items"][0]["target"]

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
    rows = result["items"]
    assert rows[0]["plausible"] is True
    assert rows[1]["likely_scalar"] is True and rows[1]["plausible"] is False
    assert rows[2]["likely_scalar"] is False and rows[2]["plausible"] is False
    # #480: the documented per-slot discriminator is present at the ROW level (not
    # only nested under row["target"]), so scripts can key on it uniformly.
    assert rows[0]["status"] == "function"
    assert all("status" in r for r in rows)
    assert rows[0]["status"] == rows[0]["target"]["status"]  # row mirrors target
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

    assert all(entry["plausible"] is False for entry in result["items"])
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

    assert result["kind"] == "messages"  # #275
    assert result["count"] == 2          # only `limit` rich matches returned
    assert len(result["items"]) == 2
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
    table = result["items"][0]["metadata_table_windows"][0]

    assert table["kind"] == "pointer_table"  # #275: embedded table is canonical
    assert len(table["items"]) == 2
    assert table["items"][1]["target"]["status"] == "unmapped"
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

    assert result["kind"] == "init_arrays" and result["total"] == 1  # #275
    assert result["pointer_size"] == 4
    assert len(result["items"]) == 1
    section = result["items"][0]
    assert section["name"] == ".init_array"
    assert section["total_entries"] == 2
    assert section["table"]["kind"] == "pointer_table"  # #275: embedded table canonical
    assert section["table"]["items"][0]["target"]["function"]["name"] == "global_ctor"


def test_init_arrays_surfaces_pe_tls_callbacks(monkeypatch):
    # #380: a PE's pre-entry TLS callbacks (IMAGE_TLS_DIRECTORY.AddressOfCallBacks)
    # must show in evidence init as pointer-table evidence, not be silently omitted.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # codex review: HIGH callback VAs (> 0xFFFFFFFF) so a 4-byte misread would
    # truncate them to their low dword. _FakeBV reports a 4-byte pointer size, so
    # without an explicit 8-byte read_width the PE32+ pointer table truncates.
    cb1, cb2 = 0x1_4000_1000, 0x1_4000_2000
    f1 = _FakeFunction(cb1, "tls_cb1"); f2 = _FakeFunction(cb2, "tls_cb2")
    # crafted PE32+ headers at base 0: MZ -> e_lfanew -> "PE\0\0" -> opt magic 0x20b
    # -> data directory[9] (TLS) RVA -> IMAGE_TLS_DIRECTORY -> AddressOfCallBacks.
    hdr = bytearray(0x158)
    hdr[0:2] = b"MZ"
    hdr[0x3C:0x40] = (0x80).to_bytes(4, "little")          # e_lfanew
    hdr[0x80:0x84] = b"PE\x00\x00"
    hdr[0x98:0x9A] = (0x20B).to_bytes(2, "little")         # PE32+ magic (opt = 0x80+24)
    hdr[0x104:0x108] = (16).to_bytes(4, "little")          # NumberOfRvaAndSizes (opt+108)
    hdr[0x150:0x154] = (0x2000).to_bytes(4, "little")      # data dir[9] RVA (opt+112+72)
    hdr[0x154:0x158] = (40).to_bytes(4, "little")          # size
    tlsdir = bytearray(32)
    tlsdir[24:32] = (0x3000).to_bytes(8, "little")         # AddressOfCallBacks VA
    cbarray = cb1.to_bytes(8, "little") + cb2.to_bytes(8, "little") + (0).to_bytes(8, "little")
    bv = _FakeBV(functions=[f1, f2], sections={},
                 memory={0: bytes(hdr), 0x2000: bytes(tlsdir), 0x3000: cbarray})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._init_arrays("active", limit=16)
    tls = [it for it in result["items"] if "TLS callbacks" in it["name"]]
    assert len(tls) == 1
    assert tls[0]["total_entries"] == 2     # null-terminated array of 2
    names = [r["target"]["function"]["name"] for r in tls[0]["table"]["items"]]
    assert names == ["tls_cb1", "tls_cb2"]
    # the full 8-byte VAs must survive (a 4-byte misread would mangle them)
    targets = [int(r["target"]["normalized"], 16) for r in tls[0]["table"]["items"]]
    assert targets == [cb1, cb2]


def test_init_arrays_no_tls_item_for_non_pe(monkeypatch):
    # A non-PE target must NOT grow a spurious TLS item. Give the MZ gate direct
    # teeth: a nonzero e_lfanew @0x3C pointing at a NON-"PE" signature, so only the
    # "MZ" magic check (not a coincidentally-falsy e_lfanew) stops the parse.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    blob = bytearray(b"\x7fELF" + b"\x00" * 0x200)
    blob[0x3C:0x40] = (0x40).to_bytes(4, "little")     # nonzero e_lfanew
    blob[0x40:0x44] = b"ELF\x00"                        # NOT "PE\0\0"
    bv = _FakeBV(functions=[], sections={}, memory={0: bytes(blob)})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._init_arrays("active")
    assert not any("TLS callbacks" in it["name"] for it in result["items"])


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

    refs, truncated, note = instance._scan_for_calls_to(bv, 0x20000)

    assert truncated is False          # complete under budget -> NOT flagged (#622)
    assert note is None                # ...and carries no partial-scan note either
    assert len(refs) == 2
    addresses = [int(r["address"], 16) for r in refs]
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

    refs, _truncated, _note = instance._scan_for_calls_to(bv, 0x20000)

    assert len(refs) == 1


def _scan_call_fn(start: int, name: str, dest: int):
    fn = _FakeFunction(start, name)
    fn.low_level_il = [[_FakeLLILInstruction(start + 0x10, _FakeConstPtr(dest))]]
    return fn


def test_scan_for_calls_to_stops_at_function_budget_and_reports_it(monkeypatch):
    """#622: the import-xref fallback scan is budgeted by function count so it can
    no longer walk every function's LLIL on a large view. A capped scan returns
    what it found AND reports that it stopped early -- an agent must never read
    the partial set as "no callers"."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_scan_call_fn(0x1000, "a", 0x20000),
                            _scan_call_fn(0x2000, "b", 0x20000)])
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.read_xrefs, "SCAN_CALLS_MAX_FUNCS", 1)

    refs, truncated, note = instance._scan_for_calls_to(bv, 0x20000)

    assert truncated is True
    assert note and "budget" in note          # names the actual reason (#622 review)
    assert [int(r["address"], 16) for r in refs] == [0x1010]   # partial, not all


def test_scan_for_calls_to_stops_at_instruction_budget_and_reports_it(monkeypatch):
    """#622: bounded in BOTH dimensions -- a function's LLIL must not blow the
    instruction budget either."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x1000, "a")
    fn.low_level_il = [[
        _FakeLLILInstruction(0x1010, _FakeConstPtr(0x20000)),
        _FakeLLILInstruction(0x1020, _FakeConstPtr(0x20000)),
    ]]
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.read_xrefs, "SCAN_CALLS_MAX_FUNCS", 100)
    monkeypatch.setattr(bridge.read_xrefs, "SCAN_CALLS_MAX_INSNS", 1)

    refs, truncated, note = instance._scan_for_calls_to(bv, 0x20000)

    assert truncated is True
    assert note and "budget" in note
    assert [int(r["address"], 16) for r in refs] == [0x1010]   # 1 of 2 examined


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
    vals = [e.get("value") for e in result["items"]]
    assert vals[0] == "0x40c370" and vals[2] == "0x400180" and vals[4] == "0x40ca80"
    assert vals[1] == "0x0" and vals[3] == "0x0"            # zero high halves, not garbage
    assert "0x40018000000000" not in vals                   # no overlapping read

    # default (stride == pointer size) still reads 8-byte pointers
    r8 = instance._pointer_table("active", "0x40b580", entries=3, stride="8")
    assert r8["read_width"] == 8
    assert [e.get("value") for e in r8["items"]] == ["0x40c370", "0x400180", "0x40ca80"]

    # explicit --width overrides the stride-derived width
    rw = instance._pointer_table("active", "0x40b580", entries=2, stride="8", width="4")
    assert rw["read_width"] == 4
    assert rw["items"][0]["value"] == "0x40c370"


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




def test_pointer_table_refuses_got_alias_import_address_slot(monkeypatch):
    # #313: evidence table on a .got/ImportAddressSymbol slot must REFUSE (never
    # fabricate adjacent unrelated GOT entries as bogus vtable slots) and name the
    # real table at *slot[0].
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]
    got_slot = 0x4a000
    real_vtable = 0x402000
    sym = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, got_slot, "_ZTV3Foo")
    bv = _FakeBV(symbols=[sym], arch=_FakeArch(name="x86_64", address_size=8),
                 memory={got_slot: real_vtable.to_bytes(8, "little") + b"\xaa" * 24})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._pointer_table("active", hex(got_slot), entries=8)
    assert exc.value.status == "got_alias"
    msg = str(exc.value)
    assert "GOT" in msg
    assert hex(real_vtable) in msg          # names the real table (*slot[0])
    assert "_ZTV3Foo" in msg                # names the alias symbol
    assert "evidence table" in msg          # actionable next command


def test_pointer_table_normal_table_unaffected_by_got_guard(monkeypatch):
    # The contrast: a normal (non-GOT) pointer table with no ImportAddressSymbol
    # still walks normally -- the guard only fires on a GOT-alias slot.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    target = _FakeFunction(0x401000, "handler")
    table = (0x401000).to_bytes(8, "little") + (0x401000).to_bytes(8, "little")
    bv = _FakeBV(functions=[target], arch=_FakeArch(name="x86_64", address_size=8),
                 memory={0x3000: table})  # no symbol at 0x3000
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._pointer_table("active", "0x3000", entries=2)
    assert result["kind"] == "pointer_table" and result["total"] == 2


def test_pointer_table_got_alias_unreadable_slot0(monkeypatch):
    # The slot0-unreadable branch: an IAT slot whose pointer can't be read still
    # refuses (got_alias) with the "unreadable" wording (#313).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]
    addr = 0x4b000
    sym = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, addr, "_ZTV3Bar")
    # memory set elsewhere so read(addr, 8) returns b"" -> _read_pointer_value None
    bv = _FakeBV(symbols=[sym], arch=_FakeArch(name="x86_64", address_size=8),
                 memory={0x99999: b"\x00" * 8})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._pointer_table("active", hex(addr), entries=8)
    assert exc.value.status == "got_alias"
    assert "unreadable" in str(exc.value)


def test_itanium_typeinfo_fragment_mangles_qualified_names(monkeypatch):
    # #305 defect 1: a demangled C++ name -> its Itanium typeinfo-string fragment.
    bridge = _load_bridge(monkeypatch)
    f = bridge.read_evidence._itanium_typeinfo_fragment
    assert f("media::codec::JsonCodec") == "N5media5codec9JsonCodecE"
    assert f("Codec") == "5Codec"
    assert f("a::B") == "N1a1BE"
    assert f("Foo<int>") is None       # templates out of scope (return None)
    assert f("operator++") is None     # operators out of scope
    assert f("") is None


def test_message_lens_matches_demangled_qualified_name(monkeypatch):
    # #305 defect 1: the DEMANGLED fully-qualified name `class list`/`class show`
    # print matches the RTTI typeinfo string -- not only the hand-stripped leaf.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    ts = _FakeStringRef(0x402020, 24, "N5media5codec9JsonCodecE")
    bv = _FakeBV(
        strings=[ts],
        sections={".rodata": _FakeSection(".rodata", 0x402000, 0x403000)},
        segments={0x402020: _FakeSegment(readable=True)},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._message_lens("active", "media::codec::JsonCodec", limit=5, table_entries=0)
    assert result["count"] == 1
    assert result["items"][0]["type_string"]["value"] == "N5media5codec9JsonCodecE"


def test_best_rtti_symbol_prefers_definition_over_got_alias(monkeypatch):
    # #305 defect 2: an _ZTV<T> name resolving to BOTH a .data.rel.ro definition
    # and a .got import alias must pick the DEFINITION (the real vtable object),
    # not the alias (whose neighbors would render as bogus slots).
    bridge = _load_bridge(monkeypatch)
    fake_bn = sys.modules["binaryninja"]
    name = "_ZTVN5media5codec9JsonCodecE"
    definition = fake_bn.Symbol(fake_bn.SymbolType.DataSymbol, 0x403dc8, name)
    alias = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x403fe0, name)
    bv = _FakeBV(
        symbols=[alias, definition],   # alias FIRST -> proves it's not order luck
        sections={".data.rel.ro": _FakeSection(".data.rel.ro", 0x403d00, 0x404000),
                  ".got": _FakeSection(".got", 0x403f00, 0x404100)},
    )
    best = bridge.read_evidence._best_rtti_symbol(bv, name)
    assert int(best.address) == 0x403dc8   # the .data.rel.ro definition

    # alias-only: falls back to it (the caller then refuses the window honestly)
    bv2 = _FakeBV(symbols=[alias], sections={".got": _FakeSection(".got", 0x403f00, 0x404100)})
    assert int(bridge.read_evidence._best_rtti_symbol(bv2, name).address) == 0x403fe0


def test_message_lens_fragment_hint_only_on_qualified_match(monkeypatch):
    # #305 review #5: the "matched via Itanium fragment" hint must only fire when
    # the query is ::-qualified (the fragment is genuinely the matcher) AND
    # something matched -- not on a bare-leaf or zero-match query.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    ts = _FakeStringRef(0x402020, 24, "N5media5codec9JsonCodecE")
    bv = _FakeBV(strings=[ts],
                 sections={".rodata": _FakeSection(".rodata", 0x402000, 0x403000)},
                 segments={0x402020: _FakeSegment(readable=True)})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    frag_hint = "Itanium typeinfo fragment"
    r1 = instance._message_lens("active", "media::codec::JsonCodec", limit=5, table_entries=0)
    assert any(frag_hint in h for h in r1["hints"])          # qualified + matched
    r2 = instance._message_lens("active", "media::codec::Nope", limit=5, table_entries=0)
    assert not any(frag_hint in h for h in r2["hints"])      # qualified but 0 matches
    r3 = instance._message_lens("active", "JsonCodec", limit=5, table_entries=0)
    assert not any(frag_hint in h for h in r3["hints"])      # bare leaf (plain needle matched)


def test_orient_digest_composes_subreads_and_handles_quick(monkeypatch):
    # #169 L2: the orient digest assembles target + imports + strings + function
    # count + sections, and degrades the strings sample honestly on a --quick view
    # instead of erroring the whole digest.
    bridge = _load_bridge(monkeypatch)
    inst = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(inst, "_target_info",
                        lambda sel: {"basename": "x", "analyzed": True, "analysis_state": "full"})
    monkeypatch.setattr(bridge.read_misc, "_imports",
                        lambda ctx, sel, **k: {"kind": "imports_summary", "total_symbols": 3,
                                               "by_kind": {"function": 3}})
    monkeypatch.setattr(bridge.read_misc, "_strings",
                        lambda ctx, sel, **k: {"kind": "strings",
                                               "items": [{"address": "0x1", "value": "hi"}], "total": 1})
    monkeypatch.setattr(bridge.read_misc, "_sections",
                        lambda ctx, sel, **k: {"items": [{"name": ".text"}], "total": 1})
    monkeypatch.setattr(bridge.read_listing, "_list_functions",
                        lambda ctx, sel, **k: {"total": 42})

    d = inst._orient_digest(None)
    assert d["kind"] == "orient_digest"
    assert d["function_count"] == 42
    assert d["imports_summary"]["total_symbols"] == 3
    assert d["strings_sample"]["items"][0]["value"] == "hi"
    # orient samples strings at a higher min-length (6) than `bn strings` (BN
    # default ~4), so its total diverges; disclose the filter so the gap is
    # explained, not a mystery (#357).
    assert d["strings_min_length"] == 6

    # --quick: no strings call, honest unavailable marker
    monkeypatch.setattr(inst, "_target_info",
                        lambda sel: {"basename": "x", "analyzed": False, "analysis_state": "quick"})
    d2 = inst._orient_digest(None)
    assert "unavailable" in d2["strings_sample"]

    # analyzed but strings refuses (RuntimeError) -> caught, not propagated
    monkeypatch.setattr(inst, "_target_info",
                        lambda sel: {"basename": "x", "analyzed": True, "analysis_state": "full"})
    def _raise(ctx, sel, **k):
        raise RuntimeError("strings not available")
    monkeypatch.setattr(bridge.read_misc, "_strings", _raise)
    d3 = inst._orient_digest(None)
    assert "unavailable" in d3["strings_sample"]


class _StringsSection:
    def __init__(self, name, start, end):
        self.name = name; self.start = start; self.end = end


class _FakeString:
    def __init__(self, start, value):
        self.start = start; self.value = value; self.length = len(value); self.type = 0


class _StringsBV:
    """The #646 ELF shape: the LOWEST string addresses are loader/linker metadata
    (`.interp`, then `.dynstr` import names), and the domain literals live much
    higher up in `.rodata`."""
    def __init__(self, *, rodata=True):
        self.strings = [
            _FakeString(0x400240, "/lib/ld-linux-aarch64.so.1"),
            _FakeString(0x400a10, "libwidget.so.1"),
            _FakeString(0x400a20, "_ITM_deregisterTMCloneTable"),
            _FakeString(0x400a3c, "__gmon_start__"),
            _FakeString(0x400a65, "WIDGET_UpdateCertificate"),
        ]
        self.sections = {
            ".interp": _StringsSection(".interp", 0x400240, 0x400260),
            ".dynstr": _StringsSection(".dynstr", 0x400a00, 0x400b00),
        }
        if rodata:
            self.strings += [
                _FakeString(0x452100, "config parse failed: %s"),
                _FakeString(0x452180, "Set Channel Index"),
            ]
            self.sections[".rodata"] = _StringsSection(".rodata", 0x452000, 0x460000)

    def get_sections_at(self, addr):
        addr = int(addr)
        return [s for s in self.sections.values() if s.start <= addr < s.end]


def _strings_ctx(bv):
    class _Ctx:
        def _resolve_view(self, s): return bv
    return _Ctx()


def test_strings_domain_sections_only_skips_elf_metadata_646(monkeypatch):
    """#646: orient's strings sample took the first N strings in ADDRESS order, and
    on an ELF the lowest string addresses are always `.interp` / `.dynstr`, so the
    sample was deterministically the loader path, `__gmon_start__`, and imported
    symbol names -- observed on 4 of 4 ELF targets, "strictly worse than random".
    min_length=6 cannot help: `_ITM_deregisterTMCloneTable` is 27 characters."""
    bridge = _load_bridge(monkeypatch)
    read_misc = bridge.read_misc
    bv = _StringsBV()

    out = read_misc._strings(_strings_ctx(bv), None, query=None, offset=0, limit=3,
                            min_length=6, domain_sections_only=True)
    values = [i["value"] for i in out["items"]]
    assert values == ["config parse failed: %s", "Set Channel Index"]
    assert not any("gmon" in v or "ld-linux" in v or "ITM_" in v for v in values)

    # Unfiltered, the head is exactly the metadata the bug reported.
    head = read_misc._strings(_strings_ctx(bv), None, query=None, offset=0, limit=3,
                              min_length=6)
    assert head["items"][0]["value"] == "/lib/ld-linux-aarch64.so.1"


def test_orient_prefers_rodata_and_discloses_sections_646(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    inst = bridge.BinaryNinjaBridge()
    bv = _StringsBV()
    monkeypatch.setattr(inst, "_target_info",
                        lambda sel: {"basename": "x", "analyzed": True, "analysis_state": "full"})
    monkeypatch.setattr(bridge.read_misc, "_imports", lambda ctx, sel, **k: {"total_symbols": 0})
    monkeypatch.setattr(bridge.read_misc, "_sections", lambda ctx, sel, **k: {"items": [], "total": 0})
    monkeypatch.setattr(bridge.read_listing, "_list_functions", lambda ctx, sel, **k: {"total": 3})
    monkeypatch.setattr(inst, "_resolve_view", lambda sel: bv)
    monkeypatch.setattr(inst.ctx, "_resolve_view", lambda sel: bv)

    sample = inst._orient_digest(None)["strings_sample"]
    assert [i["value"] for i in sample["items"]] == [
        "config parse failed: %s", "Set Channel Index"]
    # option (3): the sample is attributable.
    assert sample["sample_sections"] == [".rodata"]


def test_orient_strings_degrades_when_no_rodata_646(monkeypatch):
    """#646 negative control: a view whose only strings ARE metadata still gets a
    non-empty sample -- degrade, don't error (and don't report a false 'no strings')."""
    bridge = _load_bridge(monkeypatch)
    inst = bridge.BinaryNinjaBridge()
    bv = _StringsBV(rodata=False)
    monkeypatch.setattr(inst, "_target_info",
                        lambda sel: {"basename": "x", "analyzed": True, "analysis_state": "full"})
    monkeypatch.setattr(bridge.read_misc, "_imports", lambda ctx, sel, **k: {"total_symbols": 0})
    monkeypatch.setattr(bridge.read_misc, "_sections", lambda ctx, sel, **k: {"items": [], "total": 0})
    monkeypatch.setattr(bridge.read_listing, "_list_functions", lambda ctx, sel, **k: {"total": 3})
    monkeypatch.setattr(inst, "_resolve_view", lambda sel: bv)
    monkeypatch.setattr(inst.ctx, "_resolve_view", lambda sel: bv)

    sample = inst._orient_digest(None)["strings_sample"]
    assert sample["items"], "must fall back rather than report an empty sample"
    assert sample["items"][0]["value"] == "/lib/ld-linux-aarch64.so.1"


def test_strings_domain_filter_keeps_sectionless_views_646(monkeypatch):
    """#646: a raw/monolithic firmware image with no named sections must not be
    blinded by the filter."""
    bridge = _load_bridge(monkeypatch)
    read_misc = bridge.read_misc

    class _RawBV:
        strings = [_FakeString(0x80000000, "vxWorks boot")]
        def get_sections_at(self, addr):
            return []

    out = read_misc._strings(_strings_ctx(_RawBV()), None, query=None, offset=0,
                             limit=5, min_length=6, domain_sections_only=True)
    assert [i["value"] for i in out["items"]] == ["vxWorks boot"]


def test_render_orient_text_card(monkeypatch):
    from bn.formatters import _render_orient_text
    d = {"kind": "orient_digest", "target": {"basename": "foo"}, "analyzed": True,
         "analysis_state": "full", "function_count": 42,
         "imports_summary": {"total_symbols": 3, "by_kind": {"function": 3}},
         "sections": {"items": [{"name": ".text"}], "total": 1},
         "strings_min_length": 6,
         "strings_sample": {"items": [{"address": "0x1", "value": "hello"}], "total": 1}}
    out = _render_orient_text(d)
    assert "foo" in out and "analysis: full" in out and "functions: 42" in out and "hello" in out
    # the strings line discloses the min-length filter so its total can be
    # reconciled with `bn strings` (#357).
    assert "min-length 6" in out
    # --quick view: the warning fires and strings shows unavailable
    d2 = {**d, "analyzed": False, "analysis_state": "quick",
          "strings_sample": {"unavailable": "run refresh"}}
    out2 = _render_orient_text(d2)
    assert "--quick" in out2 and "unavailable" in out2


# --- #455: evidence table record-aware (mixed-record) mode ---

class _RecBV:
    """A bv whose read() serves inline scalar bytes for record fields."""
    def __init__(self, mem):
        self._mem = mem  # (addr, size) -> bytes

    def read(self, addr, n):
        return self._mem.get((int(addr), int(n)), b"\x00" * int(n))


class _RecCtx:
    """ctx stub for _record_table_for_view: pointer reads + classification are
    injected so the record-scan logic is tested without live BN."""
    def __init__(self, ptrs, targets):
        self._ptrs = ptrs        # addr -> pointer value (or None = unreadable)
        self._targets = targets  # value -> normalized target dict

    def _pointer_size(self, bv):
        return 8

    def _byteorder(self, bv):
        return "little"

    def _read_pointer_value(self, bv, addr, *, size=None):
        return self._ptrs.get(int(addr))

    def _normalize_code_pointer(self, bv, value):
        return self._targets[value]


def _fn_target(addr, name):
    return {"status": "function", "normalized": hex(addr), "function": {"name": name}, "context": {}}


def _data_target(addr, preview):
    return {"status": "mapped", "normalized": hex(addr), "function": None,
            "context": {"string": {"value": preview}}}


def test_record_table_classifies_mixed_fields():
    # A dispatch descriptor: opcode/flags (8B scalar) + handler (fn ptr) + name
    # (data ptr -> string). A plain pointer-stride scan would render the opcode as
    # a noisy unmapped slot; record mode labels each field.
    read_evidence = importlib.import_module("bn_agent_bridge.read_evidence")
    start, rec = 0x500000, 0x18
    ptrs = {
        start + 8: 0x401234, start + 0x10: 0x600100,        # record 0
        start + rec + 8: 0x401300, start + rec + 0x10: 0x600200,  # record 1
    }
    targets = {
        0x401234: _fn_target(0x401234, "handle_foo"),
        0x600100: _data_target(0x600100, "CMD_FOO"),
        0x401300: _fn_target(0x401300, "handle_bar"),
        0x600200: _data_target(0x600200, "CMD_BAR"),
    }
    mem = {(start, 8): (0x12).to_bytes(8, "little"), (start + rec, 8): (0x34).to_bytes(8, "little")}
    out = read_evidence._record_table_for_view(
        _RecCtx(ptrs, targets), _RecBV(mem), start,
        entries=2, record_size=rec, ptr_fields=[8, 0x10])

    assert out["kind"] == "record_table" and out["total"] == 2
    assert out["ptr_fields"] == ["0x8", "0x10"]
    r0 = out["items"][0]
    assert r0["base"] == hex(start)
    fields = {f["offset"]: f for f in r0["fields"]}
    assert set(fields) == {0, 8, 0x10}  # scalar gap + 2 declared pointers, no phantom slots
    assert fields[0]["kind"] == "scalar" and fields[0]["value"] == "0x12" and fields[0]["size"] == 8
    assert fields[8]["kind"] == "function_pointer" and fields[8]["name"] == "handle_foo"
    assert fields[0x10]["kind"] == "data_pointer" and fields[0x10]["preview"] == "CMD_FOO"
    assert not out["warnings"]

    # The text renderer labels each field.
    from bn.formatters import _render_pointer_table_text
    text = _render_pointer_table_text(out)
    assert "record table @ 0x500000" in text
    assert "fn      0x401234  handle_foo" in text
    assert 'data    0x600100  "CMD_FOO"' in text
    assert "scalar  0x12" in text


def test_record_table_warns_on_unmapped_pointer_field():
    # A declared pointer field that doesn't resolve is a distinct, warned case --
    # not silently a scalar.
    read_evidence = importlib.import_module("bn_agent_bridge.read_evidence")
    start = 0x500000
    ptrs = {start + 8: 0x33}  # unmapped-ish value
    targets = {0x33: {"status": "unmapped", "normalized": "0x33", "function": None, "context": {}}}
    out = read_evidence._record_table_for_view(
        _RecCtx(ptrs, targets), _RecBV({}), start,
        entries=1, record_size=0x10, ptr_fields=[8])
    field = {f["offset"]: f for f in out["items"][0]["fields"]}[8]
    assert field["kind"] == "unmapped"
    assert out["warnings"] and "did not resolve" in out["warnings"][0]


def test_record_table_ptr_field_out_of_range_errors():
    read_evidence = importlib.import_module("bn_agent_bridge.read_evidence")
    with pytest.raises(Exception) as exc:
        read_evidence._record_table_for_view(
            _RecCtx({}, {}), _RecBV({}), 0x500000,
            entries=1, record_size=0x10, ptr_fields=[0xc])  # 0xc + 8 > 0x10
    assert "exceeds record-size" in str(exc.value)


# --- G3 evidence/callsite rendering fidelity (#475/#476/#482) -----------------

def test_hlil_statement_scopes_folded_calls_to_callsite_address_475(monkeypatch):
    """#475: BN folds adjacent/nested calls into ONE LLIL instruction that maps to
    multiple HighLevelILCall roots. The statement selector must return the statement
    for THIS call's address, not a neighbor's (the neighbor could be listed first)."""
    bridge = _load_bridge(monkeypatch)
    il_format = bridge.il_format
    stmt_a = _FakeHLILInstruction("x = sinkA(d, s)", class_name="HighLevelILVarInit",
                                  address=0xA, expr_index=10, instr_index=10)
    call_a = _FakeHLILInstruction("sinkA(d, s)", class_name="HighLevelILCall",
                                  parent=stmt_a, address=0xA, expr_index=11, instr_index=11)
    stmt_b = _FakeHLILInstruction("y = sinkB(d)", class_name="HighLevelILVarInit",
                                  address=0xB, expr_index=20, instr_index=20)
    call_b = _FakeHLILInstruction("sinkB(d)", class_name="HighLevelILCall",
                                  parent=stmt_b, address=0xB, expr_index=21, instr_index=21)
    insn = _FakeLLILInstruction(0xA, _FakeConstPtr(0x1000), hlils=[call_b, call_a])  # neighbor first
    assert il_format._hlil_statement_text(insn) == "x = sinkA(d, s)"


def test_hlil_statement_null_when_no_folded_root_matches_475(monkeypatch):
    """#475: if NO folded root sits at this call's address (ambiguous fold), return
    None -> null statement, rather than describing another call."""
    bridge = _load_bridge(monkeypatch)
    il_format = bridge.il_format
    stmt_b = _FakeHLILInstruction("y = sinkB(d)", class_name="HighLevelILVarInit",
                                  address=0xB, expr_index=20, instr_index=20)
    call_b = _FakeHLILInstruction("sinkB(d)", class_name="HighLevelILCall",
                                  parent=stmt_b, address=0xB, expr_index=21, instr_index=21)
    call_c = _FakeHLILInstruction("sinkC()", class_name="HighLevelILCall",
                                  address=0xC, expr_index=31, instr_index=31)
    insn = _FakeLLILInstruction(0xA, _FakeConstPtr(0x1000), hlils=[call_b, call_c])  # neither at 0xA
    assert il_format._hlil_statement_text(insn) is None


def test_hlil_statement_recovers_enclosing_stmt_for_nested_cast_call_490(monkeypatch):
    """#490: for `outer(cast(inner()))` the INNER call's only non-trivial ancestor is
    the outer call across a trivial width cast (sx.q/zx.q). #475 correctly stopped
    attaching a NEIGHBOR call's statement, but over-nulled this shape. After the #475
    address filter every ancestor of the matched root provably CONTAINS this call, so
    the enclosing statement must be returned, not None. Public repro:
    `char *str = itos(getpid());` -> the getpid() callsite renders the assignment."""
    bridge = _load_bridge(monkeypatch)
    il_format = bridge.il_format
    stmt = _FakeHLILInstruction("str = itos(sx.q(getpid()))", class_name="HighLevelILVarInit",
                                parent=None, address=0xA, expr_index=10, instr_index=10)
    itos_call = _FakeHLILInstruction("itos(sx.q(getpid()))", class_name="HighLevelILCall",
                                     parent=stmt, address=0xA, expr_index=11, instr_index=11)
    cast = _FakeHLILInstruction("sx.q(getpid())", class_name="HighLevelILSx",
                                parent=itos_call, address=0xA, expr_index=12, instr_index=12)
    getpid_call = _FakeHLILInstruction("getpid()", class_name="HighLevelILCall",
                                       parent=cast, address=0xA, expr_index=13, instr_index=13)
    insn = _FakeLLILInstruction(0xA, _FakeConstPtr(0x1000), hlils=[getpid_call])
    assert il_format._hlil_statement_text(insn) == "str = itos(sx.q(getpid()))"


def test_hlil_statement_recovers_outer_call_for_bare_nested_call_490(monkeypatch):
    """#490: `foo(bar(inner()))` as a bare expression statement -- the enclosing
    statement is the outermost call, which contains the inner call. Walk through the
    folded ancestor calls to it rather than nulling."""
    bridge = _load_bridge(monkeypatch)
    il_format = bridge.il_format
    block = _FakeHLILInstruction("{...}", class_name="HighLevelILBlock",
                                 parent=None, address=0xA, expr_index=9, instr_index=9)
    outer = _FakeHLILInstruction("foo(bar(getpid()))", class_name="HighLevelILCall",
                                 parent=block, address=0xA, expr_index=10, instr_index=10)
    bar = _FakeHLILInstruction("bar(getpid())", class_name="HighLevelILCall",
                               parent=outer, address=0xA, expr_index=11, instr_index=11)
    getpid_call = _FakeHLILInstruction("getpid()", class_name="HighLevelILCall",
                                       parent=bar, address=0xA, expr_index=12, instr_index=12)
    insn = _FakeLLILInstruction(0xA, _FakeConstPtr(0x1000), hlils=[getpid_call])
    assert il_format._hlil_statement_text(insn) == "foo(bar(getpid()))"


def test_bare_void_call_statement_resolves_to_the_call_itself_644(monkeypatch):
    """#644: a call whose return value is discarded (`memset(&buf, 0, n)`) IS the
    statement -- its parent is the enclosing HighLevelILBlock, which the ancestor walk
    treats as a hard boundary. Before the root fallback that nulled hlil_statement for
    every bare call statement (the most common call shape: memcpy/strcpy/free/...)."""
    bridge = _load_bridge(monkeypatch)
    il_format = bridge.il_format
    block = _FakeHLILInstruction("sink(d, s)\nbreak", class_name="HighLevelILBlock",
                                 address=0xA, expr_index=9, instr_index=9)
    call = _FakeHLILInstruction("sink(d, s)", class_name="HighLevelILCall",
                                parent=block, address=0xA, expr_index=11, instr_index=11)
    insn = _FakeLLILInstruction(0xA, _FakeConstPtr(0x1000), hlils=[call])
    assert il_format._hlil_statement_text(insn) == "sink(d, s)"
    assert il_format._hlil_statement_localization(insn) == ("sink(d, s)", None)


def test_bare_call_statement_with_nonlocal_text_stays_null_644(monkeypatch):
    """#644 negative control: the root fallback must not smuggle a non-local blob
    through. A root whose own rendered text fails the localness filter still yields
    None -- and reports `statement_not_local`, distinct from `no_local_statement`."""
    bridge = _load_bridge(monkeypatch)
    il_format = bridge.il_format
    block = _FakeHLILInstruction("{...}", class_name="HighLevelILBlock",
                                 address=0xA, expr_index=9, instr_index=9)
    call = _FakeHLILInstruction("sink(\n" + "x" * 300 + "\n)", class_name="HighLevelILCall",
                                parent=block, address=0xA, expr_index=11, instr_index=11)
    insn = _FakeLLILInstruction(0xA, _FakeConstPtr(0x1000), hlils=[call])
    assert il_format._hlil_statement_localization(insn) == (None, "statement_not_local")


def test_call_arguments_candidates_scoped_to_callsite_476(monkeypatch):
    """#476: a folded NEIGHBOR call's HLIL args must not leak into THIS call's
    candidate list (the primary vector; the mapped-MLIL under-recovery vector is
    BN-core and stays JSON-only)."""
    bridge = _load_bridge(monkeypatch)
    read_evidence = bridge.read_evidence
    instance = bridge.BinaryNinjaBridge()
    call_a = _FakeHLILInstruction("sinkA(b, 16)", class_name="HighLevelILCall",
                                  address=0xA, expr_index=11, instr_index=11)
    call_a.params = ["b", "16"]
    call_b = _FakeHLILInstruction("sinkB(a, n)", class_name="HighLevelILCall",
                                  address=0xB, expr_index=21, instr_index=21)
    call_b.params = ["a", "n"]
    insn = _FakeLLILInstruction(0xA, _FakeConstPtr(0x1000), hlils=[call_a, call_b])
    bv = _FakeBV(functions=[])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    source, primary, candidates = read_evidence._call_arguments(instance.ctx, bv, insn, 0xA)
    assert source == "hlil"
    assert [e["text"] for e in primary] == ["b", "16"]        # THIS call's args
    cand_texts = {c["text"] for c in candidates}
    assert "a" not in cand_texts and "n" not in cand_texts    # neighbor args excluded


def test_cpp_method_this_caveat_482(monkeypatch):
    """#482: a demangled C++ instance method whose recovered first param is a
    non-pointer scalar AND is used as a pointer base (this mis-typed, no DWARF) gets
    an advisory caveat. A pointer first param, a non-method name, or a first formal
    used only as a plain scalar (static method / namespaced free function) does not."""
    bridge = _load_bridge(monkeypatch)
    caveat = bridge.read_evidence._cpp_method_this_caveat

    scalar_this = _FakeFunction(0x401000, "Owner::dispatch")
    scalar_this.parameter_vars = [_FakeVariable(name="arg1", storage=0, var_type="int32_t", identifier=1)]
    body = "void Owner::dispatch(int32_t arg1)\n{ return *(uint64_t*)(arg1 + 8); }"
    note = caveat(scalar_this, body)
    assert note is not None and "proto set" in note and "this" in note

    # #482 FP audit: same demangled shape but the scalar first formal is used only as
    # a value (a static method / namespaced free fn) -> NO caveat.
    assert caveat(scalar_this, "int32_t Owner::make(int32_t arg1) { return arg1 + 1; }") is None

    ptr_this = _FakeFunction(0x401100, "Owner::run")
    ptr_this.parameter_vars = [_FakeVariable(name="this", storage=0, var_type="Owner*", identifier=1)]
    assert caveat(ptr_this, "*(uint64_t*)(this + 8)") is None  # pointer this -> no caveat

    free_fn = _FakeFunction(0x401200, "plain_handler")
    free_fn.parameter_vars = [_FakeVariable(name="arg1", storage=0, var_type="int32_t", identifier=1)]
    assert caveat(free_fn, "*(uint64_t*)(arg1 + 8)") is None   # no '::' -> not a method


def test_function_evidence_slicing_471(monkeypatch):
    # #471: --offset/--limit/--address-window slice the call-evidence set so a large
    # dispatch function can be inspected in bounded chunks. Monkeypatch the call
    # builder with a synthetic 5-call set to test the slicing deterministically.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_calls = [{"address": hex(0x402000 + i * 0x10), "callee": f"c{i}"} for i in range(5)]
    monkeypatch.setattr(bridge.read_evidence, "_function_call_evidence",
                        lambda ctx, bv, func, context: [dict(c) for c in fake_calls])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda sel: _FakeBV(functions=[_FakeFunction(0x402000, "dispatch")]))
    monkeypatch.setattr(instance.ctx, "_find_function", lambda bv, ident, **kw: _FakeFunction(0x402000, "dispatch"))
    monkeypatch.setattr(bridge.il_format, "_decompile_text", lambda bv, f: "")
    monkeypatch.setattr(bridge.il_format, "_render_warnings", lambda t: [])
    monkeypatch.setattr(bridge.il_format, "_function_metadata", lambda f: {})

    full = instance._function_evidence("active", "dispatch", context=1)
    assert full["total_calls"] == 5 and full["returned"] == 5 and full["has_more"] is False

    page = instance._function_evidence("active", "dispatch", context=1, offset=1, limit=2)
    assert [c["callee"] for c in page["calls"]] == ["c1", "c2"]
    assert page["returned"] == 2 and page["matched_calls"] == 5 and page["has_more"] is True

    win = instance._function_evidence("active", "dispatch", context=1,
                                      address_window=(0x402010, 0x402030))
    assert [c["callee"] for c in win["calls"]] == ["c1", "c2"]   # 0x402010, 0x402020
    assert win["matched_calls"] == 2 and win["total_calls"] == 5

    # invalid slicing args are clean errors
    with pytest.raises(bridge.OperationFailure):
        instance._function_evidence("active", "dispatch", limit=0)


def test_function_evidence_paged_read_defers_decompile_622(monkeypatch):
    """#622: a paged read returns the call PAGE without paying for the full
    Pseudo-C decompile, and says so -- `decompile_deferred` + a warning line, so
    the missing decompiler warnings / C++ `this` caveat are disclosed rather than
    silently absent. An unsliced read is unchanged (full fidelity)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_calls = [{"address": hex(0x402000 + i * 0x10), "callee": f"c{i}"} for i in range(5)]
    monkeypatch.setattr(bridge.read_evidence, "_function_call_evidence",
                        lambda ctx, bv, func, context: [dict(c) for c in fake_calls])
    monkeypatch.setattr(instance.ctx, "_resolve_view",
                        lambda sel: _FakeBV(functions=[_FakeFunction(0x402000, "dispatch")]))
    monkeypatch.setattr(instance.ctx, "_find_function",
                        lambda bv, ident, **kw: _FakeFunction(0x402000, "dispatch"))
    monkeypatch.setattr(bridge.il_format, "_function_metadata", lambda f: {})
    monkeypatch.setattr(bridge.il_format, "_render_warnings", lambda t: [])
    decompiles: list = []
    monkeypatch.setattr(bridge.il_format, "_decompile_text",
                        lambda bv, f: (decompiles.append(1), "")[1])

    for kwargs in ({"limit": 2}, {"offset": 3}, {"address_window": (0x402010, 0x402030)}):
        decompiles.clear()
        page = instance._function_evidence("active", "dispatch", context=1, **kwargs)
        assert decompiles == [], f"decompile ran for a paged read: {kwargs}"
        assert page["decompile_deferred"] is True
        assert any("defer" in w.lower() for w in page["warnings"])
        # #471 paging semantics are untouched by the deferral: totals always
        # come off the FULL call set, never off the returned page.
        assert page["total_calls"] == 5
        assert page["offset"] == kwargs.get("offset", 0)

    page = instance._function_evidence("active", "dispatch", context=1, limit=2)
    assert [c["callee"] for c in page["calls"]] == ["c0", "c1"]
    assert page["matched_calls"] == 5
    assert page["returned"] == 2 and page["has_more"] is True

    win = instance._function_evidence("active", "dispatch", context=1,
                                      address_window=(0x402010, 0x402030))
    assert [c["callee"] for c in win["calls"]] == ["c1", "c2"]
    assert win["matched_calls"] == 2 and win["total_calls"] == 5

    # unsliced -> identical behaviour to before the deferral (decompile + its
    # warnings collected), and no honesty field is invented for it.
    decompiles.clear()
    full = instance._function_evidence("active", "dispatch", context=1)
    assert decompiles == [1]
    assert "decompile_deferred" not in full
    assert full["warnings"] == []


def test_parse_field_spec_467(monkeypatch):
    # #467: name:type@offset typed field specs (scalars + char[N]).
    bridge = _load_bridge(monkeypatch)
    p = bridge.read_evidence._parse_field_spec
    assert p("command:u32@0") == {"name": "command", "kind": "scalar", "width": 4, "signed": False, "offset": 0, "type": "u32"}
    assert p("delta:i16@0x4") == {"name": "delta", "kind": "scalar", "width": 2, "signed": True, "offset": 4, "type": "i16"}
    assert p("name:char[16]@8") == {"name": "name", "kind": "char_array", "width": 16, "offset": 8, "type": "char[16]"}
    for bad in ("noatsign", "n:u32", "n:weird@0", "n:u32@notanint", "n:u128@0", "n:char[0]@0"):
        with pytest.raises(bridge.OperationFailure):
            p(bad)


def test_record_table_typed_fields_and_zero_pointers_467():
    # #467: a scalar/string-only record (ZERO pointer fields) decoded from typed
    # field specs -- u32/u16 scalars + an inline char[16].
    read_evidence = importlib.import_module("bn_agent_bridge.read_evidence")
    start, rec = 0x500000, 24
    mem = {
        (start + 0, 4): (0x1101).to_bytes(4, "little"),
        (start + 4, 2): (2).to_bytes(2, "little"),
        (start + 8, 16): b"get_status\x00\x00\x00\x00\x00\x00",
    }
    specs = [read_evidence._parse_field_spec(s)
             for s in ["command:u32@0", "set_args:u16@4", "name:char[16]@8"]]
    out = read_evidence._record_table_for_view(
        _RecCtx({}, {}), _RecBV(mem), start,
        entries=1, record_size=rec, ptr_fields=[], field_specs=specs)  # zero pointer fields
    assert out["kind"] == "record_table" and out["total"] == 1
    fields = {f["name"]: f for f in out["items"][0]["fields"] if f.get("name")}
    assert fields["command"]["value"] == 0x1101 and fields["command"]["hex"] == "0x1101"
    assert fields["set_args"]["value"] == 2
    assert fields["name"]["kind"] == "char_array" and fields["name"]["value"] == "get_status"
    assert [f["name"] for f in out["fields"]] == ["command", "set_args", "name"]
    # a field exceeding the record size is a clean error (no false decode).
    with pytest.raises(read_evidence.OperationFailure):
        read_evidence._record_table_for_view(
            _RecCtx({}, {}), _RecBV({}), start, entries=1, record_size=8, ptr_fields=[],
            field_specs=[read_evidence._parse_field_spec("x:char[16]@0")])


def test_record_table_text_shows_signed_scalar_467():
    # #467: a signed (i*) typed field renders its decimal value in text, not the
    # unsigned hex (so -2 isn't shown as 0xfffe).
    from bn.formatters import _render_record_table_text
    value = {
        "kind": "record_table", "address": "0x1000", "record_size": 8, "ptr_fields": [],
        "fields": [{"name": "delta", "type": "i16", "offset": "0x0"}],
        "items": [{"row": 0, "base": "0x1000", "fields": [
            {"name": "delta", "offset": 0, "kind": "scalar", "size": 2, "type": "i16",
             "value": -2, "hex": "0xfffe"},
            {"name": "flags", "offset": 4, "kind": "scalar", "size": 4, "type": "u32",
             "value": 5, "hex": "0x5"},
        ]}],
        "warnings": [],
    }
    out = _render_record_table_text(value)
    assert "delta  -2 (0xfffe)" in out       # signed: decimal + hex
    assert "flags  0x5" in out                # unsigned: hex


class _FakeExpr:
    def __init__(self, op, **kw):
        self._op = op
        for k, v in kw.items():
            setattr(self, k, v)
    @property
    def operation(self):
        return type("Op", (), {"name": self._op})


class _DescBV:
    def __init__(self, fns=None, syms=None):
        self._fns = fns or {}       # addr -> name
        self._syms = syms or {}     # addr -> short_name
    def get_function_at(self, a):
        n = self._fns.get(int(a))
        return type("F", (), {"name": n, "symbol": type("S", (), {"short_name": n})}) if n else None
    def get_symbol_at(self, a):
        n = self._syms.get(int(a))
        return type("S", (), {"short_name": n, "name": n}) if n else None


def test_parse_field_spec_ptr_469(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    p = bridge.read_evidence._parse_field_spec("callback:ptr@0x10")
    assert p == {"name": "callback", "kind": "ptr", "width": None, "offset": 16, "type": "ptr"}


def test_decode_descriptor_field_469(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    dec = bridge.read_evidence._decode_descriptor_field
    bv = _DescBV(fns={0x401139: "status_rsp"}, syms={0x600100: "g_table"})
    # ptr resolving to a function symbol
    ptr_spec = {"name": "callback", "kind": "ptr", "offset": 16, "type": "ptr", "width": 8}
    f = dec(bv, ptr_spec, (0x402100, 16, _FakeExpr("MLIL_CONST_PTR", constant=0x401139), False), 8)
    assert f["status"] == "resolved" and f["value"] == "0x401139" and f["symbol"] == "status_rsp"
    # ptr resolving to a data symbol
    f = dec(bv, {**ptr_spec, "name": "ctx"}, (0x402108, 16, _FakeExpr("MLIL_CONST_PTR", constant=0x600100), False), 8)
    assert f["symbol"] == "g_table"
    # signed scalar
    s_spec = {"name": "delta", "kind": "scalar", "offset": 4, "type": "i16", "width": 2, "signed": True}
    f = dec(bv, s_spec, (0x402110, 4, _FakeExpr("MLIL_CONST", constant=0xfffe), False), 8)
    assert f["value"] == -2
    # unsigned scalar -> hex
    u_spec = {"name": "cmd", "kind": "scalar", "offset": 0, "type": "u16", "width": 2, "signed": False}
    assert dec(bv, u_spec, (0x402114, 0, _FakeExpr("MLIL_CONST", constant=0x1101), False), 8)["value"] == "0x1101"
    # non-constant src -> computed (not dropped)
    f = dec(bv, u_spec, (0x402118, 0, _FakeExpr("MLIL_VAR", src="x"), False), 8)
    assert f["status"] == "computed"
    # no write -> unknown
    assert dec(bv, u_spec, None, 8)["status"] == "unknown"
    # source address recorded for a resolved field
    assert dec(bv, u_spec, (0x402114, 0, _FakeExpr("MLIL_CONST", constant=1), False), 8)["source_address"] == "0x402114"


def test_store_offset_into_469(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    soi = bridge.read_evidence._store_offset_into
    var = object()
    assert soi(_FakeExpr("MLIL_ADDRESS_OF", src=var), var) == 0
    add = _FakeExpr("MLIL_ADD", left=_FakeExpr("MLIL_ADDRESS_OF", src=var),
                    right=_FakeExpr("MLIL_CONST", constant=16))
    assert soi(add, var) == 16
    assert soi(_FakeExpr("MLIL_ADDRESS_OF", src=object()), var) is None  # different var


def test_decode_descriptor_field_slice_and_sibling_469(monkeypatch):
    # #469 audit P1/P2: a write-combined wide store is SLICED per field; a sibling-slot
    # recovery is marked `via: sibling_slot` (lower confidence).
    bridge = _load_bridge(monkeypatch)
    dec = bridge.read_evidence._decode_descriptor_field
    bv = _DescBV()
    # one 8-byte store at offset 0 holds cmd@0(u16)=0x1101, type@2(u8)=0xed, flags@4(u32)=0x27
    blob = 0x27 << 32 | 0xed << 16 | 0x1101
    cmd = {"name": "cmd", "kind": "scalar", "offset": 0, "type": "u16", "width": 2, "signed": False}
    typ = {"name": "type", "kind": "scalar", "offset": 2, "type": "u8", "width": 1, "signed": False}
    flg = {"name": "flags", "kind": "scalar", "offset": 4, "type": "u32", "width": 4, "signed": False}
    w = (0x401000, 0, _FakeExpr("MLIL_CONST", constant=blob), False)  # write_off=0, covers all three
    assert dec(bv, cmd, w, 8)["value"] == "0x1101"
    assert dec(bv, typ, w, 8)["value"] == "0xed"
    assert dec(bv, flg, w, 8)["value"] == "0x27"
    # sibling-slot marker
    sib = dec(bv, cmd, (0x401000, 0, _FakeExpr("MLIL_CONST", constant=0x1101), True), 8)
    assert sib["via"] == "sibling_slot"
    # a ptr SLICED out of a scalar block is NOT resolved to a symbol (only exact stores)
    pspec = {"name": "cb", "kind": "ptr", "offset": 2, "type": "ptr", "width": 8}
    sliced = dec(bv, pspec, (0x401000, 0, _FakeExpr("MLIL_CONST_PTR", constant=0x401139), False), 8)
    assert "symbol" not in sliced


class _SurfSeg:
    executable = True
    readable = True
    def __init__(self, start=0, end=0):
        self.start = start; self.end = end


class _SurfArch:
    def get_instruction_info(self, data, addr):
        return type("_Info", (), {"length": 4})()   # fixed 4-byte insns (RISC)


class _SurfSection:
    def __init__(self, name, start, end, semantics=None):
        self.name = name; self.start = start; self.end = end
        # BN's SectionSemantics is an IntEnum whose str() is the NUMBER, so the
        # production code reads `.name` -- model that, not a bare string (#647).
        self.semantics = types.SimpleNamespace(name=semantics) if semantics else None


class _SurfBV:
    """A bv exposing one .data.rel.ro table of 4 code-pointers: 2 resolve to functions,
    and 2 don't -- 0x2020 decodes a long clean run (code-likely), 0x4000 hits an
    undefined instruction at once (data). Then a null terminator ends the run."""
    address_size = 8
    start = 0x1000
    end = 0x6000
    _VALS = [0x2000, 0x2010, 0x2020, 0x4000, 0]

    def __init__(self):
        self.sections = {
            ".data.rel.ro": _SurfSection(".data.rel.ro", 0x1000, 0x1000 + 8 * len(self._VALS),
                                         "ReadOnlyDataSectionSemantics"),
            # The pointer TARGETS live in code. Modelled explicitly because #647 tests
            # the target's SECTION, not just its segment perms.
            ".text": _SurfSection(".text", 0x2000, 0x6000, "ReadOnlyCodeSectionSemantics"),
        }
        self.arch = _SurfArch()
        self._fns = {0x2000, 0x2010}

    def read(self, addr, n):
        if int(addr) == 0x1000:
            return b"".join(v.to_bytes(8, "little") for v in self._VALS)[:n]
        return b"\x00" * int(n)

    def get_functions_containing(self, a):
        return [object()] if int(a) in self._fns else []

    def get_segment_at(self, a):
        return _SurfSeg(0x2000, 0x6000) if 0x2000 <= int(a) < 0x6000 else None

    def get_disassembly(self, a):
        a = int(a)
        return "mov x0, x0" if 0x2020 <= a < 0x2100 else "undefined"   # code region vs data

    def get_sections_at(self, a):
        a = int(a)
        return [s for s in self.sections.values() if s.start <= a < s.end]


def test_hidden_surface_scan_503(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_init_arrays", lambda ctx, sel, **kw: {"items": []})
    bv = _SurfBV()

    class _Ctx:
        def _resolve_view(self, s): return bv
        def _pointer_size(self, b): return 8
        def _byteorder(self, b): return "little"

    out = re._hidden_surface(_Ctx(), None)
    assert out["kind"] == "hidden_surface"
    # the 4 consecutive code pointers form one candidate table (>= default min-run 3)
    assert len(out["candidate_tables"]) == 1
    t = out["candidate_tables"][0]
    assert t["entries"] == 4 and t["resolved_functions"] == 2 and t["missing_functions"] == 2
    # the two functionless targets surface as candidates; decode_depth discriminates
    # 0x2020 (long clean run -> code_likely) from 0x4000 (undefined at once -> data).
    cands = {c["address"]: c for c in out["missing_function_candidates"]}
    assert set(cands) == {"0x2020", "0x4000"}
    assert cands["0x2020"]["aligned"] is True and cands["0x2020"]["decode_depth"] >= 8
    assert cands["0x2020"]["code_likely"] is True
    assert cands["0x4000"]["decode_depth"] == 0 and cands["0x4000"]["code_likely"] is False
    assert out["summary"]["missing_function_candidates"] == 2
    assert out["summary"]["code_likely_candidates"] == 1
    # a run shorter than table-min-run is not reported
    assert re._hidden_surface(_Ctx(), None, table_min_run=5)["candidate_tables"] == []
    with pytest.raises(re.OperationFailure):
        re._hidden_surface(_Ctx(), None, table_min_run=1)


def test_hidden_surface_segment_fallback_503(monkeypatch):
    # #503 (firmware): a raw/monolithic image with NO named data sections must fall
    # back to scanning readable SEGMENTS, else it finds nothing on its primary target
    # class (e.g. a VxWorks kernel loaded as one r-x segment).
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_init_arrays", lambda ctx, sel, **kw: {"items": []})

    class _Seg:
        readable = True
        executable = True
        start = 0x1000
        end = 0x1000 + 8 * len(_SurfBV._VALS)

    bv = _SurfBV()
    bv.sections = {}                 # no named sections at all
    bv.segments = [_Seg()]

    class _Ctx:
        def _resolve_view(self, s): return bv
        def _pointer_size(self, b): return 8
        def _byteorder(self, b): return "little"

    out = re._hidden_surface(_Ctx(), None)
    assert len(out["candidate_tables"]) == 1                # the table found via the segment
    assert out["candidate_tables"][0]["section"].startswith("segment@")
    assert any("no named data sections" in w for w in out["warnings"])   # fallback disclosed
    assert out["summary"]["missing_function_candidates"] == 2


def test_hidden_surface_x86_warns_decode_depth_weak_503(monkeypatch):
    # #503 audit: on a variable-length ISA the code_likely/decode_depth signal is weak;
    # the warning must ride on the OUTPUT (not just the skill doc).
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_init_arrays", lambda ctx, sel, **kw: {"items": []})
    bv = _SurfBV()
    bv.arch = type("_A", (_SurfArch,), {"name": "x86_64"})()

    class _Ctx:
        def _resolve_view(self, s): return bv
        def _pointer_size(self, b): return 8
        def _byteorder(self, b): return "little"

    out = re._hidden_surface(_Ctx(), None)
    assert out["summary"]["missing_function_candidates"] == 2
    assert any("variable-length ISA" in w and "code_likely" in w for w in out["warnings"])


class _PieSurfBV:
    """The #647 shape: a single-`LOAD` aarch64 PIE where the WHOLE image -- `.text`
    and `.rodata` alike -- is mapped r-x, so a segment-perms-only test passes every
    pointer into read-only DATA. The table here is a `{char *desc; char *usage;}`
    help-row array whose slots point at string bodies in `.rodata`."""
    address_size = 8
    start = 0x400000
    end = 0x500000
    _VALS = [0x452100, 0x452180, 0x452200, 0x452280]   # all .rodata string bodies

    def __init__(self, *, code_targets=False):
        table_end = 0x460000 + 8 * len(self._VALS)
        self.sections = {
            ".text": _SurfSection(".text", 0x400000, 0x451000, "ReadOnlyCodeSectionSemantics"),
            ".rodata": _SurfSection(".rodata", 0x451000, 0x460000, "ReadOnlyDataSectionSemantics"),
            ".data": _SurfSection(".data", 0x460000, table_end, "ReadWriteDataSectionSemantics"),
        }
        self.arch = _SurfArch()
        self._code_targets = code_targets

    def read(self, addr, n):
        if int(addr) == 0x460000:
            vals = [v - 0x52000 for v in self._VALS] if self._code_targets else self._VALS
            return b"".join(v.to_bytes(8, "little") for v in vals)[:n]
        return b"\x00" * int(n)

    def get_functions_containing(self, a):
        return []

    def get_segment_at(self, a):
        # ONE r-x LOAD spanning the entire image -- the whole point of the bug.
        return _SurfSeg(0x400000, 0x500000) if 0x400000 <= int(a) < 0x500000 else None

    def get_disassembly(self, a):
        return "mov x0, x0"          # decodes cleanly everywhere: no help from decode_depth

    def get_sections_at(self, a):
        a = int(a)
        return [s for s in self.sections.values() if s.start <= a < s.end]


def _pie_ctx(bv, *, strings=None):
    class _Ctx:
        def _resolve_view(self, s): return bv
        def _pointer_size(self, b): return 8
        def _byteorder(self, b): return "little"
        def _address_context(self, b, addr):
            value = (strings or {}).get(int(addr))
            return {"string": {"value": value}} if value else {}
    return _Ctx()


def test_hidden_surface_rodata_pointers_are_not_code_647(monkeypatch):
    """#647: `exec_target` tested SEGMENT perms only. On a single-`LOAD` aarch64 PIE
    `.rodata` is mapped r-x, so a 514-row help-string table was reported as a
    514-entry dispatch table with 514 missing functions -- a confident false positive
    on exactly the stripped-firmware targets this command exists for. The `evidence
    table` path in this same file already consulted section semantics."""
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_init_arrays", lambda ctx, sel, **kw: {"items": []})
    bv = _PieSurfBV()

    out = re._hidden_surface(_pie_ctx(bv), None)
    assert out["candidate_tables"] == []
    assert out["missing_function_candidates"] == []
    assert out["summary"]["missing_function_candidates"] == 0


def test_hidden_surface_pie_code_pointers_still_reported_647(monkeypatch):
    """#647 positive control: the fix must not blind the scan on PIE binaries. The
    same single-`LOAD` layout with a run of pointers into `.text` is still reported."""
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_init_arrays", lambda ctx, sel, **kw: {"items": []})
    bv = _PieSurfBV(code_targets=True)

    out = re._hidden_surface(_pie_ctx(bv), None)
    assert len(out["candidate_tables"]) == 1
    t = out["candidate_tables"][0]
    assert t["entries"] == 4 and t["missing_functions"] == 4
    assert out["summary"]["missing_function_candidates"] == 4


def test_hidden_surface_candidate_string_preview_647(monkeypatch):
    """#647 defence in depth: when a candidate resolves to a printable string, inline
    it -- a slot rendering `-> "Set Channel Index"` is self-refuting where a bare
    address costs a second command to disprove."""
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_init_arrays", lambda ctx, sel, **kw: {"items": []})
    bv = _PieSurfBV(code_targets=True)
    ctx = _pie_ctx(bv, strings={0x400100: "----- RF TEST MENU -----"})

    cands = {c["address"]: c for c in re._hidden_surface(ctx, None)["missing_function_candidates"]}
    assert cands["0x400100"]["string"] == "----- RF TEST MENU -----"
    assert "string" not in cands["0x400180"]


def test_hidden_surface_cap_warnings_disclose_totals_653(monkeypatch):
    """#653.5: `capped at 128` gave no basis for deciding whether raising the cap was
    worthwhile. Disclose the pre-cap total (and carry it in the summary)."""
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_init_arrays", lambda ctx, sel, **kw: {"items": []})
    bv = _PieSurfBV(code_targets=True)

    out = re._hidden_surface(_pie_ctx(bv), None, max_candidates=2)
    assert out["summary"]["missing_function_candidates"] == 2
    assert out["summary"]["missing_function_candidates_total"] == 4
    assert any("capped at 2 of 4" in w for w in out["warnings"])
    # Uncapped runs still report a total equal to the shown count.
    full = re._hidden_surface(_pie_ctx(bv), None)
    assert full["summary"]["missing_function_candidates_total"] == 4
    assert not any("capped at" in w for w in full["warnings"])


# --- #466 cross-target virtual-call slot extraction --------------------------

def _vc_expr(opname, **kw):
    return types.SimpleNamespace(operation=types.SimpleNamespace(name=opname), **kw)


def _vc_var_expr(name, ident):
    var = types.SimpleNamespace(identifier=ident, name=name)
    return _vc_expr("MLIL_VAR", src=var), var


def test_virtual_call_slot_extraction_offset_466(monkeypatch):
    """#466: `[vtable + 0x18](...)` -> slot offset 0x18 extracted from the call's
    LOAD dest. The factory trace is best-effort (None with no def context here)."""
    bridge = _load_bridge(monkeypatch)
    re_mod = bridge.read_evidence
    base_expr, _ = _vc_var_expr("rax_1", 1)
    add = _vc_expr("MLIL_ADD", left=base_expr,
                   right=_vc_expr("MLIL_CONST", constant=0x18))
    dest = _vc_expr("MLIL_LOAD", src=add)
    call = _vc_expr("MLIL_CALL", dest=dest, output=[], address=0x40115d)
    caller = types.SimpleNamespace(mlil=types.SimpleNamespace(instructions=[call]), view=None)
    off, factory = re_mod._vc_slot_and_factory(caller, call, 8)
    assert off == 0x18
    assert factory is None


def test_virtual_call_slot_extraction_slot0_466(monkeypatch):
    """#466: `[vtable](...)` (slot 0, no offset) -> slot offset 0."""
    bridge = _load_bridge(monkeypatch)
    re_mod = bridge.read_evidence
    base_expr, _ = _vc_var_expr("rax_1", 1)
    dest = _vc_expr("MLIL_LOAD", src=base_expr)
    call = _vc_expr("MLIL_CALL", dest=dest, output=[], address=0x1000)
    caller = types.SimpleNamespace(mlil=types.SimpleNamespace(instructions=[call]), view=None)
    off, _ = re_mod._vc_slot_and_factory(caller, call, 8)
    assert off == 0


def test_virtual_call_not_vtable_dispatch_466(monkeypatch):
    """#466: a direct call (dest is a CONST_PTR, not a LOAD) has no vtable slot ->
    the extractor returns None so the handler reports not_virtual."""
    bridge = _load_bridge(monkeypatch)
    re_mod = bridge.read_evidence
    call = _vc_expr("MLIL_CALL", dest=_vc_expr("MLIL_CONST_PTR", constant=0x401050),
                    output=[], address=0x1000)
    caller = types.SimpleNamespace(mlil=types.SimpleNamespace(instructions=[call]), view=None)
    assert re_mod._vc_slot_and_factory(caller, call, 8) is None


def test_virtual_call_slot_extraction_aarch64_shape_544(monkeypatch):
    """#544: on a non-folded ISA (aarch64) the dispatch is two instructions --
    `xN = [vtable + off]` (a SET_VAR whose src is a LOAD) then `CALL xN` -- so the
    call `dest` is an MLIL_VAR, not a LOAD. The resolver must follow the call-dest
    var one reaching-def hop to the LOAD and extract the slot offset. Under the OLD
    code this same input returned None (dest op is MLIL_VAR, rejected by the LOAD
    guard) -- asserted below so the test is meaningful."""
    bridge = _load_bridge(monkeypatch)
    re_mod = bridge.read_evidence
    vt_expr, _ = _vc_var_expr("x8", 2)                       # vtable pointer register
    add = _vc_expr("MLIL_ADD", left=vt_expr,
                   right=_vc_expr("MLIL_CONST", constant=0x18))
    load = _vc_expr("MLIL_LOAD", src=add)
    call_dest, call_var = _vc_var_expr("x9", 3)              # xN holding the slot target
    set_var = _vc_expr("MLIL_SET_VAR", dest=call_var, src=load, address=0x1000)
    call = _vc_expr("MLIL_CALL", dest=call_dest, output=[], address=0x1004)
    caller = types.SimpleNamespace(
        mlil=types.SimpleNamespace(instructions=[set_var, call]), view=None)
    # The OLD fast-path guard (`"LOAD" not in _op(dest)`) rejected this exact input:
    assert re_mod._op(call.dest) == "MLIL_VAR"
    off, factory = re_mod._vc_slot_and_factory(caller, call, 8)
    assert off == 0x18
    assert factory is None


def test_virtual_call_register_indirect_not_misresolved_544(monkeypatch):
    """#544: a plain register-indirect call whose reaching def is NOT a vtable-slot
    load (`xN = &func` -- a function-pointer constant) must still return None. The
    def-hop follows the var but its src is a CONST_PTR, not a LOAD, so the LOAD guard
    still cleanly rejects it -- no misresolve to a bogus slot."""
    bridge = _load_bridge(monkeypatch)
    re_mod = bridge.read_evidence
    call_dest, call_var = _vc_var_expr("x9", 3)
    set_var = _vc_expr("MLIL_SET_VAR", dest=call_var,
                       src=_vc_expr("MLIL_CONST_PTR", constant=0x401050),
                       address=0x1000)
    call = _vc_expr("MLIL_CALL", dest=call_dest, output=[], address=0x1004)
    caller = types.SimpleNamespace(
        mlil=types.SimpleNamespace(instructions=[set_var, call]), view=None)
    assert re_mod._vc_slot_and_factory(caller, call, 8) is None


def test_virtual_call_slot_offset_in_its_own_instruction_790(monkeypatch):
    """#790: g++ -O0 does not fold `vptr + off` into the dispatch load. The real
    shape is three instructions:

        0x401126  rdx = [rax].q        (vptr)
        0x401129  rdx_1 = rdx + 0x10   (the slot offset, computed SEPARATELY)
        0x40112d  rdx_2 = [rdx_1].q    (the dispatch load)
        0x401133  rax_1 = rdx_2(rdi)   (the call)

    so the load's src is an MLIL_VAR and the offset read as 0 -- every -O0
    dispatch resolved to slot 0 while still reporting `resolved: true`, i.e. a
    silently WRONG provider method (m0 instead of m2 on the fixture this was
    found on; the -O2 build of the same source folds the ADD and was correct).
    The offset must come from the address var's own reaching def, one hop
    further out than #544's call-dest hop."""
    bridge = _load_bridge(monkeypatch)
    re_mod = bridge.read_evidence
    vptr_expr, _ = _vc_var_expr("rdx", 1)
    add = _vc_expr("MLIL_ADD", left=vptr_expr,
                   right=_vc_expr("MLIL_CONST", constant=0x10))
    _, slot_addr_var = _vc_var_expr("rdx_1", 2)
    set_add = _vc_expr("MLIL_SET_VAR", dest=slot_addr_var, src=add, address=0x401129)
    slot_load = _vc_expr("MLIL_LOAD", src=_vc_var_expr("rdx_1", 2)[0])
    _, slot_target_var = _vc_var_expr("rdx_2", 3)
    set_load = _vc_expr("MLIL_SET_VAR", dest=slot_target_var, src=slot_load,
                        address=0x40112d)
    call = _vc_expr("MLIL_CALL", dest=_vc_var_expr("rdx_2", 3)[0], output=[],
                    address=0x401133)
    caller = types.SimpleNamespace(
        mlil=types.SimpleNamespace(instructions=[set_add, set_load, call]), view=None)

    off, factory = re_mod._vc_slot_and_factory(caller, call, 8)

    assert off == 0x10, "the slot offset lives in its own ADD instruction"
    assert factory is None


def test_virtual_call_slot_offset_own_instruction_commutative_790(monkeypatch):
    """#790: the def-walk must keep the folded case's commutativity handling --
    `vptr + const` and `const + vptr` are the same slot, wherever the ADD lives."""
    bridge = _load_bridge(monkeypatch)
    re_mod = bridge.read_evidence
    vptr_expr, _ = _vc_var_expr("rdx", 1)
    add = _vc_expr("MLIL_ADD", left=_vc_expr("MLIL_CONST", constant=0x18),
                   right=vptr_expr)
    _, slot_addr_var = _vc_var_expr("rdx_1", 2)
    set_add = _vc_expr("MLIL_SET_VAR", dest=slot_addr_var, src=add, address=0x401129)
    slot_load = _vc_expr("MLIL_LOAD", src=_vc_var_expr("rdx_1", 2)[0])
    _, slot_target_var = _vc_var_expr("rdx_2", 3)
    set_load = _vc_expr("MLIL_SET_VAR", dest=slot_target_var, src=slot_load,
                        address=0x40112d)
    call = _vc_expr("MLIL_CALL", dest=_vc_var_expr("rdx_2", 3)[0], output=[],
                    address=0x401133)
    caller = types.SimpleNamespace(
        mlil=types.SimpleNamespace(instructions=[set_add, set_load, call]), view=None)

    off, _ = re_mod._vc_slot_and_factory(caller, call, 8)

    assert off == 0x18


def test_virtual_call_folded_shape_keeps_factory_790(monkeypatch):
    """#790: the def-walk must not run PAST the vtable pointer. On the folded
    (-O2-style) shape the walk still visits the vtable var -- `[vtable + 0x18]`
    with `vtable = [obj]` and `obj = factory()` -- so `base` must stay the
    vtable VAR; ending on the vtable LOAD instead makes the factory trace below
    read `_vc_var(<load>)`, get None, and drop the factory from a dispatch that
    resolved it before #790. This is the half of the change that is meant to be
    unchanged, hence asserted here rather than left to the slot-offset tests."""
    bridge = _load_bridge(monkeypatch)
    re_mod = bridge.read_evidence
    _, obj_var = _vc_var_expr("rax", 1)
    get_obj = _vc_expr("MLIL_CALL", dest=_vc_expr("MLIL_CONST_PTR", constant=0x4200),
                       output=[obj_var], address=0x1000)
    vt_expr, vt_var = _vc_var_expr("rcx", 2)
    set_vt = _vc_expr("MLIL_SET_VAR", dest=vt_var,
                      src=_vc_expr("MLIL_LOAD", src=_vc_var_expr("rax", 1)[0]),
                      address=0x1004)
    dest = _vc_expr("MLIL_LOAD", src=_vc_expr("MLIL_ADD", left=vt_expr,
                                              right=_vc_expr("MLIL_CONST", constant=0x18)))
    call = _vc_expr("MLIL_CALL", dest=dest, output=[], address=0x1008)
    bv = types.SimpleNamespace(
        get_symbol_at=lambda a: types.SimpleNamespace(short_name="get_instance",
                                                      name="get_instance"))
    caller = types.SimpleNamespace(
        mlil=types.SimpleNamespace(instructions=[get_obj, set_vt, call]), view=bv)

    off, factory = re_mod._vc_slot_and_factory(caller, call, 8)

    assert off == 0x18
    assert factory == "get_instance", "the folded path's factory trace must survive the walk"


def test_virtual_call_stacked_addends_resolve_790(monkeypatch):
    """#790: stacked constant addends. `-O0` emits a vtable-base adjustment and
    the slot offset as SEPARATE ADDs --

        0x101000  a = vtable + 8     (base adjustment)
        0x101004  b = a + 8          (base adjustment)
        0x101008  c = b + 0x10       (the slot offset)
        0x10100c  t = [c].q
        0x101010  t(...)

    so the slot really is at 0x20 (index 4). A walk bounded to a fixed handful of
    hops sums only the tail, hands back a too-small offset, and the resolver then
    names the provider method at the WRONG slot while still reporting
    `resolved: true` -- the silently-wrong-answer class #790 is about. Asserted
    through the resolver, because the wrong slot index is what a caller sees."""
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    vt_expr, vt_var = _vc_var_expr("rcx", 2)
    a_expr, a_var = _vc_var_expr("a", 3)
    set_a = _vc_expr("MLIL_SET_VAR", dest=a_var,
                     src=_vc_expr("MLIL_ADD", left=vt_expr,
                                  right=_vc_expr("MLIL_CONST", constant=8)),
                     address=0x101000)
    b_expr, b_var = _vc_var_expr("b", 4)
    set_b = _vc_expr("MLIL_SET_VAR", dest=b_var,
                     src=_vc_expr("MLIL_ADD", left=a_expr,
                                  right=_vc_expr("MLIL_CONST", constant=8)),
                     address=0x101004)
    c_expr, c_var = _vc_var_expr("c", 5)
    set_c = _vc_expr("MLIL_SET_VAR", dest=c_var,
                     src=_vc_expr("MLIL_ADD", left=b_expr,
                                  right=_vc_expr("MLIL_CONST", constant=0x10)),
                     address=0x101008)
    _, t_var = _vc_var_expr("t", 6)
    set_t = _vc_expr("MLIL_SET_VAR", dest=t_var, src=_vc_expr("MLIL_LOAD", src=c_expr),
                     address=0x10100c)
    call = _vc_expr("MLIL_CALL", dest=_vc_var_expr("t", 6)[0], output=[], address=0x101010)
    caller = types.SimpleNamespace(
        mlil=types.SimpleNamespace(instructions=[set_a, set_b, set_c, set_t, call]),
        view=None)

    class _Ctx:
        def _resolve_view(self, s): return object()
        def _pointer_size(self, b): return 8
        def _find_function(self, b, addr, contained=True): return caller

    monkeypatch.setattr(re, "_mlil_call_at", lambda c, a: call)
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps",
                        lambda pv: {"Provider": {"vtable": types.SimpleNamespace(address=0x9000)}})
    monkeypatch.setattr(bridge.read_class, "_vtable_layout",
                        lambda ctx, pv, addr: {"slots": [
                            {"index": 4, "method": {"name": "doWork", "address": "0x4200"}}]})

    out = re._resolve_virtual_call(_Ctx(), None, "0x101010")

    assert out["slot_offset"] == "0x20", "all stacked addends belong to the slot offset"
    assert out["slot_index"] == 4
    assert out["resolved"] is True
    assert out["candidates"][0]["method"] == "doWork"


# --- #530 Thumb-pointer miss count normalization -----------------------------

class _ThumbSurfBV:
    """ARM/Thumb bv: a table of 3 Thumb function pointers (stored as addr|1). The
    real functions live at the EVEN entries. Before #530, the candidate-table miss
    count checked the raw odd value and counted every Thumb slot as missing."""
    address_size = 4
    start = 0x1000
    end = 0x6000
    _VALS = [0x2001, 0x2011, 0x2021]         # addr | 1 (Thumb tag)

    def __init__(self):
        end = 0x1000 + 4 * len(self._VALS)
        self.sections = {
            ".rodata": _SurfSection(".rodata", 0x1000, end, "ReadOnlyDataSectionSemantics"),
            ".text": _SurfSection(".text", 0x2000, 0x6000, "ReadOnlyCodeSectionSemantics"),
        }
        self.arch = _SurfArch()
        self._fns = {0x2000, 0x2010, 0x2020}  # functions at the EVEN entries

    def read(self, addr, n):
        if int(addr) == 0x1000:
            return b"".join(v.to_bytes(4, "little") for v in self._VALS)[:n]
        return b"\x00" * int(n)

    def get_functions_containing(self, a):
        return [object()] if int(a) in self._fns else []

    def get_segment_at(self, a):
        return _SurfSeg(0x2000, 0x6000) if 0x2000 <= int(a) < 0x6000 else None

    def get_disassembly(self, a):
        return "mov r0, r0"

    def get_sections_at(self, a):
        a = int(a)
        return [s for s in self.sections.values() if s.start <= a < s.end]


def test_hidden_surface_thumb_miss_count_normalized_530(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_init_arrays", lambda ctx, sel, **kw: {"items": []})
    bv = _ThumbSurfBV()

    class _Ctx:
        def _resolve_view(self, s): return bv
        def _pointer_size(self, b): return 4
        def _byteorder(self, b): return "little"
        def _supports_thumb_pointer_tags(self, b): return True

    out = re._hidden_surface(_Ctx(), None, table_min_run=3)
    assert len(out["candidate_tables"]) == 1
    t = out["candidate_tables"][0]
    # All three Thumb pointers normalize to real functions -> 0 missing, 3 resolved.
    assert t["entries"] == 3
    assert t["missing_functions"] == 0
    assert t["resolved_functions"] == 3
    # And no functionless candidates surface (they all normalize to real fns).
    assert out["summary"]["missing_function_candidates"] == 0


def test_hidden_surface_non_arm_miss_count_unchanged_530(monkeypatch):
    # Guard: a non-Thumb target must be unaffected (norm_ptr is a no-op) -- the
    # original _SurfBV still reports 2 resolved / 2 missing.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_init_arrays", lambda ctx, sel, **kw: {"items": []})
    bv = _SurfBV()

    class _Ctx:
        def _resolve_view(self, s): return bv
        def _pointer_size(self, b): return 8
        def _byteorder(self, b): return "little"

    t = re._hidden_surface(_Ctx(), None)["candidate_tables"][0]
    assert t["resolved_functions"] == 2 and t["missing_functions"] == 2


# --- #531 virtual-call slot alignment validation -----------------------------

def _vc_resolve_ctx(bv, ptr=8):
    class _Ctx:
        def _resolve_view(self, s): return bv
        def _pointer_size(self, b): return ptr
        def _find_function(self, b, addr, contained=True):
            return types.SimpleNamespace(name="consumer", view=b)
    return _Ctx()


def test_virtual_call_unaligned_slot_offset_skipped_531(monkeypatch):
    # #531: an unaligned slot offset (ptr=8, slot_off=12) must NOT floor to slot 1
    # and name that provider's method -- it is reported unresolved with a reason.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (12, None))

    def _boom(*a, **k):
        raise AssertionError("must not build candidates for an unaligned slot")
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps", _boom)

    out = re._resolve_virtual_call(_vc_resolve_ctx(object()), None, "0x1000")
    assert out["resolved"] is False
    assert out["candidates"] == []
    assert out["slot_index"] is None
    assert "unresolved_reason" in out


def test_virtual_call_aligned_slot_offset_resolves_531(monkeypatch):
    # An aligned offset (ptr=8, slot_off=16 -> index 2) still resolves normally.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (16, None))
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps",
                        lambda pv: {"Provider": {"vtable": types.SimpleNamespace(address=0x9000)}})
    monkeypatch.setattr(bridge.read_class, "_vtable_layout",
                        lambda ctx, pv, addr: {"slots": [
                            {"index": 2, "method": {"name": "doWork", "address": "0x4100"}}]})

    out = re._resolve_virtual_call(_vc_resolve_ctx(object()), None, "0x1000")
    assert out["slot_index"] == 2
    assert out["resolved"] is True
    assert out["candidates"][0]["class"] == "Provider"
    assert out["candidates"][0]["method"] == "doWork"


def _vc_code_row(index):
    """A readable CODE row of the provider's pointer table."""
    value = 0x410000 + index * 8
    return {"index": index, "entry_address": hex(0x9010 + index * 8),
            "value": hex(value), "readable": True, "plausible": True,
            "target": {"status": "function", "normalized": hex(value),
                       "function": {"name": f"m{index}", "address": hex(value)}}} 


class _VcTableCtx:
    """ctx for the #822 vcall tests: the provider's vtable is a REAL pointer
    table, so `_vtable_layout` (whose display window stops at 64) and the
    targeted slot probe both run for real. That disagreement between "what the
    listing shows" and "what the table holds" IS the behaviour under test, so a
    monkeypatched layout cannot express it."""
    _BASE = 0x9010                      # vtable_addr + 2*ptr for the 0x9000 vtable

    def __init__(self, rows):
        self._rows = rows               # absolute row index -> row dict

    def _resolve_view(self, s):
        return object()

    def _pointer_size(self, b):
        return 8

    def _find_function(self, b, addr, contained=True):
        return types.SimpleNamespace(name="consumer", view=b)

    def _read_pointer_value(self, b, addr, *, size=None):
        return 0x9100                   # word[1] -> a valid typeinfo

    def _typeinfo_name_at(self, b, addr):
        return "prov::Klass"

    def _pointer_table_layout(self, bv, start, *, entries, stride):
        first = (start - self._BASE) // stride
        out = []
        for i in range(first, first + entries):
            row = self._rows.get(i)
            out.append({**row, "index": i} if row is not None
                       else {"index": i, "value": None, "readable": False})
        return {"kind": "pointer_table", "items": out}


def test_virtual_call_beyond_display_cap_resolves_the_requested_slot(monkeypatch):
    # #822 (g-584-3): this test REPLACED
    # `test_virtual_call_beyond_truncated_cap_reports_reason`, which pinned the
    # opposite -- slot 70 of a valid 80-slot table came back unresolved with a
    # "beyond the recovered window" reason, because the DISPLAY scan stops at
    # 64. #584's acceptance is that a valid slot 70 RESOLVES anyway, while the
    # display listing stays capped at 64. Deliberate contract change.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 70, None))
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps",
                        lambda pv: {"Provider": {"vtable": types.SimpleNamespace(address=0x9000)}})
    ctx = _VcTableCtx({i: _vc_code_row(i) for i in range(80)})

    out = re._resolve_virtual_call(ctx, None, "0x1000")
    assert out["slot_index"] == 70
    assert out["resolved"] is True
    assert out["ambiguous"] is False
    assert len(out["candidates"]) == 1
    cand = out["candidates"][0]
    assert cand["class"] == "Provider"
    assert cand["method"] == "m70"
    assert cand["method_address"] == hex(0x410000 + 70 * 8)
    # Itanium: vtable + 2-word header + index*ptr -- the SLOT's address, distinct
    # from the method address it stores.
    assert cand["vtable_entry"] == hex(0x9000 + 2 * 8 + 70 * 8)
    assert "unresolved_reason" not in out
    assert "unresolved_reason_code" not in out

    # ...and the DISPLAY listing for that same table is still capped: resolving
    # the requested slot must not have uncapped `class show`.
    layout = bridge.read_class._vtable_layout(ctx, None, 0x9000)
    assert len(layout["slots"]) == 64
    assert layout["truncated"] is True
    assert layout["total_lower_bound"] == 65


def test_virtual_call_reports_a_genuinely_absent_slot_as_absent(monkeypatch):
    # #822 (g-584-5): a provider whose table ENDS before the requested index is
    # an absence, and the machine-readable discriminator must say so -- this is
    # the shape that must never be confused with the capped-scan one below.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 5, None))
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps",
                        lambda pv: {"Provider": {"vtable": types.SimpleNamespace(address=0x9000)}})
    # rows 0..2 are methods, row 3 is the next object's typeinfo: the table ends
    # there, so index 5 does not exist and the scan was NOT capped.
    rows = {0: _vc_code_row(0), 1: _vc_code_row(1), 2: _vc_code_row(2),
            3: {"index": 3, "value": "0xA100", "readable": True, "plausible": True,
                "target": {"status": "mapped", "function": None,
                           "context": {"kind": "data", "segment": {"executable": False}}}}}
    ctx = _VcTableCtx(rows)

    out = re._resolve_virtual_call(ctx, None, "0x1000")
    assert out["resolved"] is False
    assert out["candidates"] == []
    assert out["unresolved_reason_code"] == "slot_not_present"
    assert "unresolved_reason" not in out          # no prose: nothing was censored


def test_virtual_call_reports_truncated_reason_when_the_probe_cannot_decide(monkeypatch):
    # #822: when the window is capped AND the targeted read fails, the slot is
    # UNKNOWN -- the truncated code must survive instead of degrading into a
    # confident "slot_not_present".
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 70, None))
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps",
                        lambda pv: {"Provider": {"vtable": types.SimpleNamespace(address=0x9000)}})
    monkeypatch.setattr(bridge.read_class, "_vtable_layout",
                        lambda ctx, pv, addr: {
                            "slots": [{"index": i, "method": {"name": f"m{i}"}} for i in range(64)],
                            "truncated": True,
                            "max_slots": 64,
                            "scanned": 64,
                            "total": None,
                            "total_lower_bound": 65,
                            "slots_truncated": True,
                            "scan_truncated": True,
                            "truncated_reason": "scan_capped",
                        })

    class _UnreadableCtx:
        def _resolve_view(self, s): return object()
        def _pointer_size(self, b): return 8
        def _find_function(self, b, addr, contained=True):
            return types.SimpleNamespace(name="consumer", view=b)
        def _pointer_table_layout(self, bv, start, *, entries, stride):
            raise RuntimeError("provider vtable unreadable")

    out = re._resolve_virtual_call(_UnreadableCtx(), None, "0x1000")
    assert out["resolved"] is False
    assert out["candidates"] == []
    assert out["unresolved_reason_code"] == "vtable_scan_truncated"
    assert "70" in out["unresolved_reason"]
    assert "64" in out["unresolved_reason"]


def test_virtual_call_keeps_the_slot_unknown_when_an_unreadable_row_blocks_the_probe(monkeypatch):
    # #822 review (round 1): the probe walks the rows between the display window
    # and the requested slot, and a row it cannot READ does not prove the table
    # ends there. Reporting `slot_not_present` turned a failed read into a
    # confident absence -- the inverse of g-584-5's discriminator (the capped
    # scan's "may exist past" disclosure must survive).
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 70, None))
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps",
                        lambda pv: {"Provider": {"vtable": types.SimpleNamespace(address=0x9000)}})
    rows = {i: _vc_code_row(i) for i in range(80)}
    rows[66] = {"index": 66, "entry_address": hex(0x9010 + 66 * 8), "value": None,
                "readable": False}
    ctx = _VcTableCtx(rows)

    out = re._resolve_virtual_call(ctx, None, "0x1000")
    assert out["resolved"] is False
    assert out["candidates"] == []
    assert out["unresolved_reason_code"] == "vtable_scan_truncated"
    assert "70" in out["unresolved_reason"]
    assert "64" in out["unresolved_reason"]


def test_virtual_call_names_an_unreadable_stop_instead_of_the_cap(monkeypatch):
    # The other unreadable shape: the provider's WINDOW scan stopped early on a
    # row it could not read, so no cap was involved. The disclosure must name
    # that stop -- reusing the cap sentence would blame a bound nothing hit.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 70, None))
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps",
                        lambda pv: {"Provider": {"vtable": types.SimpleNamespace(address=0x9000)}})
    rows = {i: _vc_code_row(i) for i in range(80)}
    rows[3] = {"index": 3, "entry_address": hex(0x9010 + 3 * 8), "value": None,
               "readable": False}
    ctx = _VcTableCtx(rows)

    out = re._resolve_virtual_call(ctx, None, "0x1000")
    assert out["resolved"] is False
    assert out["candidates"] == []
    assert out["unresolved_reason_code"] == "vtable_scan_truncated"
    assert "70" in out["unresolved_reason"]
    assert "could not be read" in out["unresolved_reason"]
    assert "scan capped" not in out["unresolved_reason"]


def test_virtual_call_reports_an_undecodable_provider_body_as_unknown(monkeypatch):
    # #822 review (round 2): a provider whose vtable SYMBOL resolves but whose
    # body this command refuses to decode (word[1] is not a typeinfo -- the
    # import/GOT and relocated-to-zero shapes) was never SEARCHED for the slot.
    # `slot_not_present` ("the provider's table genuinely ends before the index")
    # therefore claimed an absence about a table this command never read. The
    # base SHA emitted no reason code here at all; the wrong claim arrived with
    # the code.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence

    class _NoBodyCtx(_VcTableCtx):
        def _typeinfo_name_at(self, b, addr):
            return None

    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 5, None))
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps",
                        lambda pv: {"Provider": {"vtable": types.SimpleNamespace(address=0x9000)}})
    ctx = _NoBodyCtx({i: _vc_code_row(i) for i in range(8)})

    out = re._resolve_virtual_call(ctx, None, "0x1000")
    assert out["resolved"] is False
    assert out["candidates"] == []
    assert out["unresolved_reason_code"] == "vtable_body_unresolved"
    assert "5" in out["unresolved_reason"]


def test_virtual_call_reports_an_unreadable_provider_layout_as_unknown(monkeypatch):
    # The other undecodable shape: reading the provider's layout RAISED. That was
    # swallowed, and the swallowed failure read as a confirmed absence to
    # anything branching on the reason code.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence

    class _RaisingCtx(_VcTableCtx):
        def _pointer_table_layout(self, bv, start, *, entries, stride):
            raise RuntimeError("provider vtable unreadable")

    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 5, None))
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps",
                        lambda pv: {"Provider": {"vtable": types.SimpleNamespace(address=0x9000)}})

    out = re._resolve_virtual_call(_RaisingCtx({}), None, "0x1000")
    assert out["resolved"] is False
    assert out["candidates"] == []
    assert out["unresolved_reason_code"] == "vtable_body_unresolved"
    assert "5" in out["unresolved_reason"]


def test_virtual_call_warns_when_another_provider_body_could_not_be_decoded(monkeypatch):
    # A resolved set must not imply every provider was consulted: a provider whose
    # body could not be decoded may implement the slot, and that provider is
    # silently absent from `candidates` otherwise.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence

    class _MixedBodyCtx(_VcTableCtx):
        """0x9000 decodes; 0xA000's word[1] is not a typeinfo, so it is refused."""
        def _read_pointer_value(self, b, addr, *, size=None):
            return 0x9100 if addr == 0x9008 else 0

    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 5, None))
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps", lambda pv: {
        "Provider": {"vtable": types.SimpleNamespace(address=0x9000)},
        "Opaque": {"vtable": types.SimpleNamespace(address=0xA000)},
    })
    ctx = _MixedBodyCtx({i: _vc_code_row(i) for i in range(8)})

    out = re._resolve_virtual_call(ctx, None, "0x1000")
    assert out["resolved"] is True
    assert out["ambiguous"] is False
    assert len(out["candidates"]) == 1
    assert "unresolved_reason" not in out
    assert len(out["warnings"]) == 1
    assert "may be incomplete" in out["warnings"][0]
    assert "not fully scanned" not in out["warnings"][0]


def test_virtual_call_types_a_present_index_that_holds_no_method(monkeypatch):
    # #822 review (round 2) major: the index IS in the table -- `class show`
    # lists it as a slot -- but it holds no callable target (a null/pure-virtual
    # placeholder). Typing that `slot_not_present` said the index was past the
    # table's end, contradicting the listing the same command shows.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 2, None))
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps",
                        lambda pv: {"Provider": {"vtable": types.SimpleNamespace(address=0x9000)}})
    # index 2 is an INTERIOR null placeholder (a trailing one would be trimmed as
    # the next object's padding, and then it genuinely is not a listed slot).
    rows = {0: _vc_code_row(0), 1: _vc_code_row(1),
            2: {"index": 2, "entry_address": hex(0x9010 + 2 * 8), "value": "0x0",
                "readable": True, "plausible": True,
                "target": {"status": "null", "context": {"kind": "null"}}},
            3: _vc_code_row(3),
            4: {"index": 4, "entry_address": hex(0x9010 + 4 * 8), "value": "0xA100",
                "readable": True, "plausible": True,
                "target": {"status": "mapped", "function": None,
                           "context": {"kind": "data", "segment": {"executable": False}}}}}
    ctx = _VcTableCtx(rows)

    out = re._resolve_virtual_call(ctx, None, "0x1000")
    assert out["resolved"] is False
    assert out["candidates"] == []
    assert out["unresolved_reason_code"] == "slot_present_no_method"
    assert "2" in out["unresolved_reason"]
    # the index really is listed by the layout the same call reads
    layout = bridge.read_class._vtable_layout(ctx, None, 0x9000)
    assert 2 in [s["index"] for s in layout["slots"]]


def test_virtual_call_types_a_present_beyond_cap_index_that_holds_no_method(monkeypatch):
    # ...and the same holds for an index the targeted probe reaches past the
    # display window: resolving a placeholder is not resolving a method.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 70, None))
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps",
                        lambda pv: {"Provider": {"vtable": types.SimpleNamespace(address=0x9000)}})
    rows = {i: _vc_code_row(i) for i in range(80)}
    rows[70] = {"index": 70, "entry_address": hex(0x9010 + 70 * 8), "value": "0x0",
                "readable": True, "plausible": True,
                "target": {"status": "null", "context": {"kind": "null"}}}
    ctx = _VcTableCtx(rows)

    out = re._resolve_virtual_call(ctx, None, "0x1000")
    assert out["slot_index"] == 70
    assert out["resolved"] is False
    assert out["candidates"] == []
    assert out["unresolved_reason_code"] == "slot_present_no_method"
    assert "70" in out["unresolved_reason"]


def test_virtual_call_reports_a_view_with_no_provider_table_to_search(monkeypatch):
    # #822 review (round 3): NOTHING was searched -- this view yields no RTTI
    # class with a vtable at all (stripped/imported RTTI) -- so
    # `slot_not_present` ("a provider's table was READ and genuinely ends before
    # the index") claimed an absence the command never established. The
    # "unsearched is unknown" rule has to hold for every trigger, not only the
    # round-2 named pair.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 5, None))

    class _BareView:
        def get_symbols(self):
            return []

    class _BareCtx(_VcTableCtx):
        def _resolve_view(self, s):
            return _BareView()

    out = re._resolve_virtual_call(_BareCtx({}), None, "0x1000")   # real discovery: {}
    assert out["resolved"] is False
    assert out["candidates"] == []
    assert out["unresolved_reason_code"] == "vtable_body_unresolved"
    assert "5" in out["unresolved_reason"]


def test_virtual_call_types_a_provider_class_without_a_located_vtable_as_unknown(monkeypatch):
    # The other unsearched trigger: a provider class that is IN the view's map
    # but has no vtable symbol to read (RTTI only). Its table was never
    # searched, so nothing there proves the slot absent.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 5, None))
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps", lambda pv: {
        "TypeinfoOnly": {"typeinfo": types.SimpleNamespace(address=0xB000)},
    })

    out = re._resolve_virtual_call(_VcTableCtx({}), None, "0x1000")
    assert out["resolved"] is False
    assert out["candidates"] == []
    assert out["unresolved_reason_code"] == "vtable_body_unresolved"
    assert "5" in out["unresolved_reason"]


def test_virtual_call_warns_about_a_provider_class_without_a_located_vtable(monkeypatch):
    # ...and a resolved set must say so too: the unsearched class could hold a
    # competing implementation, so `resolved: true` must not imply that every
    # provider was consulted.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 5, None))
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps", lambda pv: {
        "Provider": {"vtable": types.SimpleNamespace(address=0x9000)},
        "TypeinfoOnly": {"typeinfo": types.SimpleNamespace(address=0xB000)},
    })
    ctx = _VcTableCtx({i: _vc_code_row(i) for i in range(8)})

    out = re._resolve_virtual_call(ctx, None, "0x1000")
    assert out["resolved"] is True
    assert len(out["candidates"]) == 1
    assert "unresolved_reason" not in out
    assert len(out["warnings"]) == 1
    assert "may be incomplete" in out["warnings"][0]


def test_virtual_call_reports_probe_limit_rather_than_a_false_absence(monkeypatch):
    # #822: the requested index comes from an MLIL constant. When it is so far
    # past the window that the targeted walk would exceed its bound, the result
    # says WHICH bound stopped it -- a garbage offset must not read as a
    # confirmed absent slot.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    limit = bridge.read_class._VTABLE_SLOT_PROBE_MAX_ROWS
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * (limit + 100), None))
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps",
                        lambda pv: {"Provider": {"vtable": types.SimpleNamespace(address=0x9000)}})
    ctx = _VcTableCtx({i: _vc_code_row(i) for i in range(limit + 200)})

    out = re._resolve_virtual_call(ctx, None, "0x1000")
    assert out["slot_index"] == limit + 100
    assert out["resolved"] is False
    assert out["candidates"] == []
    assert out["unresolved_reason_code"] == "vtable_slot_probe_limit"
    assert str(limit) in out["unresolved_reason"]


def test_virtual_call_unaligned_code_is_typed_822(monkeypatch):
    # #822 (g-584-5): the #531 unaligned-offset reason carries its typed code
    # too, so every non-resolution shape is machine-readable, not just some.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (12, None))

    out = re._resolve_virtual_call(_vc_resolve_ctx(object()), None, "0x1000")
    assert out["unresolved_reason_code"] == "slot_offset_not_aligned"


def test_virtual_call_unresolved_without_truncation_omits_reason(monkeypatch):
    # Pins that `unresolved_reason` is never spuriously attached when the
    # provider's scan was not truncated -- a genuinely nonexistent slot stays
    # a plain unresolved result. (#822: still no PROSE; the typed code carries
    # the machine-readable half, asserted in the absent-slot test above.)
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 5, None))
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps",
                        lambda pv: {"Provider": {"vtable": types.SimpleNamespace(address=0x9000)}})
    monkeypatch.setattr(bridge.read_class, "_vtable_layout",
                        lambda ctx, pv, addr: {
                            "slots": [{"index": i, "method": {"name": f"m{i}"}} for i in range(3)],
                            "truncated": False,
                            "max_slots": 64,
                            "scan_truncated": False,
                            "truncated_reason": None,
                        })

    out = re._resolve_virtual_call(_vc_resolve_ctx(object()), None, "0x1000")
    assert out["candidates"] == []
    assert "unresolved_reason" not in out


def test_virtual_call_resolved_flags_uncertainty_from_other_truncated_provider(monkeypatch):
    # Round-2 finding 9 / #706 follow-up: `Provider` resolves slot 2 cleanly,
    # but `OtherProvider` -- a DIFFERENT class implementing the same slot
    # shape -- had its own vtable scan capped before it ever reached slot 2
    # (its window only covers indices 0-1). That provider was never actually
    # checked for this slot, so it could supply another candidate this result
    # doesn't include. `resolved: true` / `ambiguous: false` must not imply
    # every provider was consulted -- the uncertainty must be surfaced
    # alongside the resolved candidate, not silently dropped.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 2, None))
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps", lambda pv: {
        "Provider": {"vtable": types.SimpleNamespace(address=0x9000)},
        "OtherProvider": {"vtable": types.SimpleNamespace(address=0xA000)},
    })

    def _fake_layout(ctx, pv, addr):
        if addr == 0x9000:
            return {"slots": [{"index": 2, "method": {"name": "doWork", "address": "0x4100"}}],
                    "truncated": False, "max_slots": 64,
                    "scan_truncated": False, "truncated_reason": None}
        return {"slots": [{"index": i, "method": {"name": f"o{i}"}} for i in range(2)],
                "truncated": True, "max_slots": 2, "total": None, "total_lower_bound": 3,
                "slots_truncated": True, "scan_truncated": True,
                "truncated_reason": "scan_capped"}

    monkeypatch.setattr(bridge.read_class, "_vtable_layout", _fake_layout)

    out = re._resolve_virtual_call(_vc_resolve_ctx(object()), None, "0x1000")
    assert out["resolved"] is True
    assert out["ambiguous"] is False
    assert len(out["candidates"]) == 1
    assert out["candidates"][0]["class"] == "Provider"
    assert "unresolved_reason" not in out          # not the empty-candidates shape
    assert len(out.get("warnings") or []) == 1
    warning = out["warnings"][0]
    assert "2" in warning and "not fully scanned" in warning


def test_virtual_call_resolved_omits_warnings_when_no_provider_truncated(monkeypatch):
    # Guard: `warnings` is never spuriously attached when every provider's
    # scan actually covered the slot.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 2, None))
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps",
                        lambda pv: {"Provider": {"vtable": types.SimpleNamespace(address=0x9000)}})
    monkeypatch.setattr(bridge.read_class, "_vtable_layout",
                        lambda ctx, pv, addr: {
                            "slots": [{"index": 2, "method": {"name": "doWork", "address": "0x4100"}}],
                            "truncated": False, "max_slots": 64,
                        })

    out = re._resolve_virtual_call(_vc_resolve_ctx(object()), None, "0x1000")
    assert out["resolved"] is True
    assert "warnings" not in out


# --- #813: class-axis budget for the provider-class sweep --------------------

def _vc_budget(re):
    """#813 class budget, read from the module so the test tracks the documented
    constant instead of hardcoding 64. `getattr` keeps the BASE tree (no constant)
    on the same scenario: there it must fail on the missing truncation signal, not
    on a missing attribute name."""
    return getattr(re, "_VC_MAX_CLASSES", 64)


def _vc_class_maps(n_classes, first_addr=0x9000):
    """`n_classes` provider classes, each with a vtable address."""
    return {f"Provider{i}": {"vtable": types.SimpleNamespace(address=first_addr + i * 0x100)}
            for i in range(n_classes)}


def test_virtual_call_class_sweep_budget_reports_truncation_813(monkeypatch):
    # #813: the sweep over provider RTTI classes was unbounded, and a result that
    # stopped early looked identical to one that consulted every class. With more
    # classes than the budget, a candidate found INSIDE the scanned prefix must
    # still carry the class-axis uncertainty -- `resolved: true` must not imply
    # every class was consulted.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 2, None))
    budget = _vc_budget(re)
    maps = _vc_class_maps(budget + 1)          # exactly one class past the budget
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps", lambda pv: maps)
    decoded: list[int] = []

    def _fake_layout(ctx, pv, addr):
        decoded.append(addr)
        if addr == 0x9000:                     # the first class implements slot 2
            return {"slots": [{"index": 2, "method": {"name": "doWork", "address": "0x4100"}}],
                    "truncated": False, "max_slots": 64}
        return {"slots": [], "truncated": False, "max_slots": 64}

    monkeypatch.setattr(bridge.read_class, "_vtable_layout", _fake_layout)

    out = re._resolve_virtual_call(_vc_resolve_ctx(object()), None, "0x1000")
    # The candidate inside the scanned prefix is still resolved ...
    assert out["resolved"] is True
    assert out["candidates"][0]["class"] == "Provider0"
    # ... but the class axis reports the cut sweep machine-readably, with counts.
    assert out["classes_truncated"] is True
    assert out["classes_scanned"] == budget
    assert out["classes_total"] == budget + 1
    assert len(out["warnings"]) == 1
    assert f"{budget} of {budget + 1}" in out["warnings"][0]
    # ... and the budget really stopped the sweep: no class past it was decoded.
    assert len(decoded) == budget
    # Text mode must disclose it too: the note has to live in a field the text
    # renderer actually prints (not only in the machine-readable counts).
    from bn.formatters import _render_virtual_call_text
    text = _render_virtual_call_text(out)
    assert f"{budget} of {budget + 1}" in text and "class budget" in text


def test_virtual_call_under_class_budget_reports_no_class_signal_813(monkeypatch):
    # Guard: a provider view with fewer classes than the budget reports nothing
    # new -- no spurious class-axis truncation, and no reason invented from it.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 2, None))
    maps = _vc_class_maps(_vc_budget(re) - 1)   # one class short of the budget
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps", lambda pv: maps)
    monkeypatch.setattr(bridge.read_class, "_vtable_layout",
                        lambda ctx, pv, addr: {
                            "slots": [{"index": 2, "method": {"name": "doWork", "address": "0x4100"}}],
                            "truncated": False, "max_slots": 64,
                        })

    out = re._resolve_virtual_call(_vc_resolve_ctx(object()), None, "0x1000")
    # every class was swept: each implements slot 2, so the result is ambiguous
    assert out["ambiguous"] is True
    assert len(out["candidates"]) == _vc_budget(re) - 1
    assert "classes_truncated" not in out
    assert "classes_scanned" not in out and "classes_total" not in out
    assert "warnings" not in out
    assert "unresolved_reason" not in out


def test_virtual_call_class_budget_unresolved_reason_joins_table_reason_813(monkeypatch):
    # Either axis can truncate alone; when BOTH do and no candidate is found, the
    # reason must name both -- the class-axis note must not clobber the #584
    # per-table reason, and the counts stay machine-readable.
    bridge = _load_bridge(monkeypatch)
    re = bridge.read_evidence
    monkeypatch.setattr(re, "_mlil_call_at", lambda caller, a: object())
    monkeypatch.setattr(re, "_vc_slot_and_factory", lambda caller, call, ptr: (8 * 70, None))
    budget = _vc_budget(re)
    monkeypatch.setattr(bridge.read_class, "_rtti_symbol_maps",
                        lambda pv: _vc_class_maps(budget + 1))
    monkeypatch.setattr(bridge.read_class, "_vtable_layout",
                        lambda ctx, pv, addr: {
                            "slots": [{"index": i, "method": {"name": f"m{i}"}} for i in range(64)],
                            # #822 split the two questions this fixture used to
                            # answer with one flag: `truncated` is "the LISTING is
                            # a prefix", `scan_truncated` is "the total is not
                            # exact". The read consults `scan_truncated` to decide
                            # the slot is undecided, so a stub carrying only
                            # `truncated` describes a layout the real
                            # `_vtable_layout` never returns for a capped scan --
                            # and silently loses the #584 table axis this test
                            # exists to check survives alongside #813's.
                            "truncated": True, "slots_truncated": True,
                            "scan_truncated": True,
                            "truncated_reason": "scan_capped", "max_slots": 64,
                        })

    out = re._resolve_virtual_call(_vc_resolve_ctx(object()), None, "0x1000")
    assert out["candidates"] == []
    reason = out["unresolved_reason"]
    assert "70" in reason and "64 slots" in reason           # #584 table axis kept
    assert f"{budget} of {budget + 1}" in reason             # #813 class axis added
    assert out["classes_truncated"] is True
    assert out["classes_scanned"] == budget
    assert out["classes_total"] == budget + 1
    # Both axes reach text mode on the empty-candidate path.
    from bn.formatters import _render_virtual_call_text
    text = _render_virtual_call_text(out)
    assert "64 slots" in text and f"{budget} of {budget + 1}" in text


# --- #557: machine-readable reason codes for a null hlil_statement ---------


def test_hlil_statement_localization_reason_codes(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    il_format = bridge.il_format

    # no HLIL mapping at all -> no_hlil_mapping
    bare = _FakeLLILInstruction(0xA, _FakeConstPtr(0x1000))
    assert il_format._hlil_statement_localization(bare) == (None, "no_hlil_mapping")

    # HLIL exists but folded into a non-call statement -> hlil_not_call_shaped
    var_init = _FakeHLILInstruction("if (x)\nwhole blob\nreturn",
                                    class_name="HighLevelILVarInit", expr_index=10, instr_index=10)
    coarse = _FakeLLILInstruction(0xA, _FakeConstPtr(0x1000), hlils=[var_init])
    assert il_format._hlil_statement_localization(coarse) == (None, "hlil_not_call_shaped")

    # multiple folded call roots, none at this address -> ambiguous_fold
    stmt_b = _FakeHLILInstruction("y = sinkB(d)", class_name="HighLevelILVarInit",
                                  address=0xB, expr_index=20, instr_index=20)
    call_b = _FakeHLILInstruction("sinkB(d)", class_name="HighLevelILCall",
                                  parent=stmt_b, address=0xB, expr_index=21, instr_index=21)
    call_c = _FakeHLILInstruction("sinkC()", class_name="HighLevelILCall",
                                  address=0xC, expr_index=31, instr_index=31)
    ambiguous = _FakeLLILInstruction(0xA, _FakeConstPtr(0x1000), hlils=[call_b, call_c])
    assert il_format._hlil_statement_localization(ambiguous) == (None, "ambiguous_fold")

    # a matched, localizable statement -> (text, None)
    stmt_a = _FakeHLILInstruction("x = sinkA(d, s)", class_name="HighLevelILVarInit",
                                  address=0xA, expr_index=10, instr_index=10)
    call_a = _FakeHLILInstruction("sinkA(d, s)", class_name="HighLevelILCall",
                                  parent=stmt_a, address=0xA, expr_index=11, instr_index=11)
    good = _FakeLLILInstruction(0xA, _FakeConstPtr(0x1000), hlils=[call_a])
    assert il_format._hlil_statement_localization(good) == ("x = sinkA(d, s)", None)


def test_callsites_null_hlil_carries_reason_code(monkeypatch):
    # #557: an indirect/unmapped callsite whose hlil_statement is null now also
    # reports WHY it's null instead of a bare null.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "crt_rand")
    fn = _FakeFunction(0x500000, "caller")
    fn.basic_blocks = [_FakeBasicBlock(0x500010, 0x500015)]
    fn.low_level_il = [[_FakeLLILInstruction(0x500010, _FakeConstPtr(0x461746))]]
    bv = _FakeBV(functions=[callee, fn], instruction_lengths={0x500010: 5},
                 disassembly={0x500010: "call crt_rand"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    rows = _callsites_items(instance, "active", "crt_rand", within_identifiers=["caller"], context=1)
    assert rows[0]["hlil_statement"] is None
    assert rows[0]["hlil_statement_reason"] == "no_hlil_mapping"


# --- #549: authoritative arguments vs low-confidence candidates ------------


def test_function_evidence_marks_argument_confidence(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "send_message")
    # A recovered 2-parameter prototype: what EARNS the `authoritative` stamp. Without
    # it the callee's arity is unknown and #648 correctly demotes to `inferred`.
    callee.parameter_vars = [
        _FakeVariable(name="code", storage=0, var_type="int32_t", identifier=1),
        _FakeVariable(name="out", storage=1, var_type="void*", identifier=2),
    ]
    caller = _FakeFunction(0x412470, "build_response")
    stmt = _FakeHLILInstruction("rc = send_message(6, &response)", class_name="HighLevelILVarInit",
                                address=0x4124A0, expr_index=29, instr_index=29)
    call_expr = _FakeHLILInstruction("send_message(6, &response)", class_name="HighLevelILCall",
                                     parent=stmt, address=0x4124A0, expr_index=30, instr_index=30)
    call_expr.params = ["6", "&response"]
    call_insn = _FakeLLILInstruction(0x4124A0, _FakeConstPtr(0x461746), hlils=[call_expr])
    call_insn.params = [_FakeReg("r0"), _FakeConstPtr(6)]
    caller.basic_blocks = [_FakeBasicBlock(0x4124A0, 0x4124A4)]
    caller.low_level_il = [[call_insn]]
    bv = _FakeBV(functions=[callee, caller], instruction_lengths={0x4124A0: 4},
                 disassembly={0x4124A0: "bl send_message"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    call = instance._function_evidence("active", "build_response", context=0)["calls"][0]
    # canonical HLIL args -> authoritative; the reason code is null (statement present)
    assert call["argument_source"] == "hlil"
    assert call["argument_confidence"] == "authoritative"
    assert call["hlil_statement_reason"] is None
    # every candidate is explicitly low-confidence with a provenance source
    assert call["argument_candidates"]
    for cand in call["argument_candidates"]:
        assert cand["confidence"] == "low"
        assert cand["source"] in ("llil", "mlil", "hlil")


def _arity_bv(monkeypatch, instance, *, callee_params, arg_texts, arg_regs=8,
              user_type=False):
    """A direct call with a structured prototype and independently rendered args."""
    callee = _FakeFunction(0x401100, "hw_get_version")
    callee.parameter_vars = [
        _FakeVariable(name=f"a{i}", storage=i, var_type="int64_t", identifier=i + 1)
        for i in range(callee_params)
    ]
    prototype = types.SimpleNamespace(
        parameters=list(callee.parameter_vars), has_variable_arguments=False)
    if user_type:
        callee.set_user_type(prototype)
    else:
        callee.set_auto_type(prototype)
    callee.calling_convention = types.SimpleNamespace(
        int_arg_regs=[f"x{i}" for i in range(arg_regs)])
    caller = _FakeFunction(0x401400, "probe_device")
    rendered = f"int32_t r = hw_get_version({', '.join(arg_texts)})"
    stmt = _FakeHLILInstruction(rendered, class_name="HighLevelILVarInit",
                                address=0x401400, expr_index=10, instr_index=10)
    call_expr = _FakeHLILInstruction(f"hw_get_version({', '.join(arg_texts)})",
                                     class_name="HighLevelILCall", parent=stmt,
                                     address=0x401400, expr_index=11, instr_index=11)
    call_expr.params = list(arg_texts)
    call_insn = _FakeLLILInstruction(0x401400, _FakeConstPtr(0x401100), hlils=[call_expr])
    call_insn.params = [_FakeReg("x0")]
    caller.basic_blocks = [_FakeBasicBlock(0x401400, 0x401404)]
    caller.low_level_il = [[call_insn]]
    bv = _FakeBV(functions=[callee, caller], instruction_lengths={0x401400: 4},
                 disassembly={0x401400: "bl hw_get_version"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    return bv


def test_argument_confidence_demoted_on_unknown_arity_callee_648(monkeypatch):
    """#648: with no recovered prototype BN assumes every argument register is live,
    so HLIL renders the NEIGHBOURING call's staging -- a log string and the stack
    canary -- as arguments 2..5 of a 1-parameter vendor API. `authoritative` meant
    "HLIL produced a list", not "the list is right"; #549 left this residual on the
    canonical field. Confirmed wrong against upstream source on a second target."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _arity_bv(monkeypatch, instance, callee_params=0, arg_regs=6,
              arg_texts=["&var_20", "1", '"get_version"', '"<<<< ENTER >>>>"',
                         "__stack_chk_guard", "0"])

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["argument_source"] == "hlil"
    assert call["argument_confidence"] != "authoritative"
    assert call["argument_confidence"] == "inferred"
    assert call["arity_unknown"] is True
    # 6 recovered args on a 6-register ABI: BN is enumerating registers.
    assert call["abi_register_saturated"] is True


def test_argument_confidence_kept_for_known_prototype_648(monkeypatch):
    """#648 negative control -- the assertion that keeps the fix from blanket-demoting
    everything. `memset` has a bundled 3-parameter prototype, so its arguments really
    ARE authoritative (verified live: BN reports 3 declared params for memset and 0
    for an unprototyped stub)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _arity_bv(monkeypatch, instance, callee_params=3,
              arg_texts=["&buf", "0", "0x100"])

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["argument_confidence"] == "authoritative"
    assert call["arity_unknown"] is False
    assert "abi_register_saturated" not in call


def _with_type_library(bv, *, symbol, params, variadic=False,
                       lib_name="libc_x86_64.so.6"):
    """Attach a type library that declares *symbol* with *params* parameters.

    Mirrors BN's real surface: `bv.type_libraries` of objects answering
    `get_named_object(name)` with a function type (or None). A `_FakeBV` has no
    `type_libraries` at all, which is why every other test in this file keeps its
    pre-#759 behaviour -- no library, no claim.
    """
    proto = types.SimpleNamespace(
        parameters=[types.SimpleNamespace(name=f"p{i}") for i in range(params)],
        has_variable_arguments=variadic,
    )
    bv.type_libraries = [types.SimpleNamespace(
        name=lib_name,
        get_named_object=lambda n, _s=symbol, _p=proto: _p if str(n) == _s else None,
    )]
    return bv


def _as_import(bv, name="hw_get_version"):
    """Give the callee an IMPORTED function symbol, the provenance that makes a
    library signature evidence about it. Measured shape: 174 of 175
    library-name-matched callees on a dynamically linked target were imports."""
    callee = next(f for f in bv.functions if f.name == name)
    # `_FakeSymbol` is the shared stand-in whose `.type.name` is what
    # `is_imported_function` reads -- the fake `SymbolType` members are plain
    # strings with no `.name`, which is #593's divergence and would silently
    # report "not an import" here.
    callee.symbol = _FakeSymbol("ImportedFunctionSymbol")
    return bv



def test_argument_confidence_demoted_when_a_library_contradicts_the_prototype_759(
    monkeypatch,
):
    """#759: the #742 guard compares the rendered list against the callee's own
    RECOVERED prototype, so an under-recovered callee agrees with itself and keeps
    `authoritative` -- vacuous by construction.

    Measured on a real target: `__popcountdi2()` rendered ZERO arguments with
    `argument_confidence: authoritative` and `arity_mismatch` absent, while the
    attached type library declares one parameter. The library is an INDEPENDENT
    witness, so the row must stop asserting authority and say why."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=0, arg_texts=[])
    _with_type_library(bv, symbol="hw_get_version", params=3)
    _as_import(bv)

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]

    assert call["argument_confidence"] == "inferred"
    assert call["prototype_unverified"] is True
    assert call["library_arity"] == 3
    assert call["library_source"] == "libc_x86_64.so.6"


def test_argument_confidence_kept_when_the_library_agrees_759(monkeypatch):
    """Negative control: an agreeing library must not disturb an earned
    `authoritative` (the #648 `memset` precedent)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=3,
                   arg_texts=["&buf", "0", "0x100"])
    _with_type_library(bv, symbol="hw_get_version", params=3)
    _as_import(bv)

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]

    assert call["argument_confidence"] == "authoritative"
    assert "prototype_unverified" not in call


def test_library_cross_check_skips_mangled_cxx_names_759(monkeypatch):
    """The measured false-positive class. A mangled C++ name carries IMPLICIT
    parameters a count comparison cannot see -- `this` on a method, an sret
    return-slot pointer on a by-value class return -- so either side can differ by
    one with nothing wrong. On a C++-heavy target 3 of 4 raw firings were exactly
    that shape; restricting to unmangled names took the rate to 0."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=2, arg_texts=["a", "b"])
    bv.functions[0].name = "_ZNSt6vectorIiSaIiEE9push_backERKi"
    _with_type_library(bv, symbol="_ZNSt6vectorIiSaIiEE9push_backERKi", params=3)

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]

    assert "prototype_unverified" not in call
    assert call["argument_confidence"] == "authoritative"


def test_library_cross_check_makes_no_claim_for_a_variadic_signature_759(monkeypatch):
    """A variadic library signature states only its FIXED count, so a count
    comparison against it is not evidence of under-recovery."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=3,
                   arg_texts=["fmt", "a", "b"])
    _with_type_library(bv, symbol="hw_get_version", params=1, variadic=True)
    _as_import(bv)

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]

    assert "prototype_unverified" not in call
    assert call["argument_confidence"] == "authoritative"


# --- #865: the callee-side read witness -----------------------------------
#
# Where no library names the callee, `_library_param_count` refuses (#862) and
# `arity_mismatch` is silent -- the rendered list and the recovered prototype are
# the SAME under-recovery, so `declared_count == len(arguments)` compares the
# recovery against itself. The witness these tests drive settles that shape from
# inside the binary: a body that reads an argument register its recovered
# prototype does not declare takes more arguments than the recovery admits.


def _with_llil(bv, *instructions):
    """Give the callee at 0x401100 a body: one block, in address order."""
    callee = bv.get_function_at(0x401100)
    callee.low_level_il = [list(instructions)]
    return callee


def _read_into(address, dest, source_reg):
    """``dest = <expr reading source_reg>`` -- a SET_REG with a register SOURCE."""
    insn = _FakeLLILInstruction(
        address, types.SimpleNamespace(name=dest), operation="LLIL_SET_REG")
    insn.src = _FakeReg(source_reg)
    return insn


def _write_reg(address, dest):
    """``dest = <constant>`` -- a SET_REG that reads no register at all."""
    insn = _FakeLLILInstruction(
        address, types.SimpleNamespace(name=dest), operation="LLIL_SET_REG")
    insn.src = _FakeConstPtr(0x4000)
    return insn


def test_argument_confidence_demoted_when_the_callee_body_reads_past_its_prototype_865(
    monkeypatch,
):
    """#865 AC1, the shape neither shipped witness settles: an ordinary-named
    local function no attached library names, whose recovered prototype declares
    2 parameters and which HLIL rendered exactly those 2 -- so the #742 guard has
    nothing to compare (#862's measured residual, `declared_count ==
    len(arguments)`) and #862 has no library to consult. Its own body reads x2, a
    third argument register, which is a positive reason to distrust the recovery:
    the row stops claiming authority and carries the two numbers that disagree."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=2, arg_texts=["a", "b"])
    _with_llil(
        bv,
        _read_into(0x401100, "sp_0", "x0"),
        _read_into(0x401104, "sp_8", "x2"),   # a THIRD argument register
    )

    card = instance._function_evidence("active", "probe_device", context=0)
    call = card["calls"][0]

    assert call["argument_source"] == "hlil"
    assert call["argument_confidence"] == "inferred"
    assert call["callee_under_recovered"] is True
    assert call["callee_read_arity"] == 3
    assert call["declared_arity"] == 2
    # No library named the callee, so the #862 witness stayed silent: this is the
    # residual itself being answered, not an override of a library verdict.
    assert "prototype_unverified" not in call
    assert "arity_mismatch" not in call
    # The reason reaches TEXT mode too (hoisted into the function-level warnings),
    # so the demotion is never silent on the card.
    assert any("reads 3 argument register(s)" in w for w in card["warnings"])


def test_argument_confidence_kept_when_the_callee_body_reads_only_its_declared_args_865(
    monkeypatch,
):
    """#865 AC4, matching arity: reading exactly the registers the prototype
    declares is what a CORRECT recovery looks like and must stay authoritative."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=3, arg_texts=["a", "b", "c"])
    _with_llil(
        bv,
        _read_into(0x401100, "sp_0", "x0"),
        _read_into(0x401104, "sp_8", "x1"),
        _read_into(0x401108, "sp_16", "x2"),
    )

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["argument_confidence"] == "authoritative"
    assert "callee_under_recovered" not in call


def test_argument_confidence_kept_for_a_void_callee_reading_no_argument_register_865(
    monkeypatch,
):
    """#865 AC4, the genuinely-void callee: a body that reads no argument register
    agrees with its recovered 0-parameter prototype, so it keeps `authoritative`
    -- demoting here would flag every `f()` call in a stripped binary for nothing.
    A body that WRITES an argument register as scratch is not a read."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=0, arg_texts=[])
    _with_llil(bv, _write_reg(0x401100, "x0"))

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["argument_confidence"] == "authoritative"
    assert "callee_under_recovered" not in call


def test_argument_confidence_kept_for_a_variadic_callee_reading_every_register_865(
    monkeypatch,
):
    """#865 AC4, variadic: a variadic body reaches its arguments through the
    register save area / `va_list`, so an argument-register read is not an arity
    claim at all -- the body below reads every integer register and the row is
    left alone."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=1, arg_texts=["fmt"], user_type=True)
    bv.get_function_at(0x401100).type.has_variable_arguments = True
    _with_llil(bv, *[_read_into(0x401100 + i * 4, f"sp_{i * 8}", f"x{i}") for i in range(6)])

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["argument_confidence"] == "authoritative"
    assert "callee_under_recovered" not in call


def test_argument_confidence_kept_when_the_callee_body_reads_after_writing_865(monkeypatch):
    """The witness's own false-positive guard. A compiler routinely reuses a
    caller-saved argument register as scratch, and a CALL clobbers all of them --
    only a read of a register the body has not already written says anything about
    the INCOMING arguments."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=1, arg_texts=["a"])
    _with_llil(
        bv,
        _write_reg(0x401100, "x2"),           # x2 reused as scratch...
        _read_into(0x401104, "sp_8", "x2"),   # ...then read: not an argument
        _FakeLLILInstruction(0x401108, None, operation="LLIL_CALL"),
        _read_into(0x40110C, "sp_16", "x3"),  # read after a call: a result, not setup
    )

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["argument_confidence"] == "authoritative"
    assert "callee_under_recovered" not in call


def test_argument_confidence_undemoted_when_the_callee_has_no_body_to_read_865(monkeypatch):
    """#865's stated intersection, pinned as a KNOWN gap rather than papered over:
    both witnesses are blind to an import whose body is not in the image, so the
    row keeps its claim and no witness field is invented for it."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=1, arg_texts=["a"])
    _as_import(bv)   # an imported symbol, no body in this image

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["argument_confidence"] == "authoritative"
    assert "callee_under_recovered" not in call


def test_argument_confidence_fires_on_a_sub_register_argument_read_865(monkeypatch):
    """The REAL-IL shape, which a name-only match missed entirely: BN names a
    register by the WIDTH the code touched, so a SysV argument read renders as
    `esi` while `int_arg_regs` lists `rsi`. Measured on a compiled probe, a
    3-parameter body reported zero argument-register reads and the witness never
    fired outside the fakes. The scan canonicalizes both sides now."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=2, arg_texts=["a", "b"])
    bv.get_function_at(0x401100).calling_convention = types.SimpleNamespace(
        int_arg_regs=["rdi", "rsi", "rdx", "rcx", "r8", "r9"])
    _with_llil(
        bv,
        _read_into(0x401100, "sp_0", "esi"),      # the 32-bit form of rsi...
        _read_into(0x401104, "sp_8", "edx"),      # ...and of rdx: a THIRD argument
    )

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["argument_confidence"] == "inferred"
    assert call["callee_read_arity"] == 3 and call["declared_arity"] == 2


def test_argument_confidence_kept_when_a_sub_register_write_retires_the_argument_865(
    monkeypatch,
):
    """The same canonicalization on the WRITE side: a compiler that reuses the
    32-bit half of an argument register as scratch (`esi = eax`) retires the whole
    register, so a later read of either half is not evidence about the incoming
    argument. x86-64 and AArch64's w-form are both covered."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=2, arg_texts=["a", "b"])
    bv.get_function_at(0x401100).calling_convention = types.SimpleNamespace(
        int_arg_regs=["rdi", "rsi", "rdx", "rcx", "r8", "r9"])
    _with_llil(
        bv,
        _read_into(0x401100, "sp_0", "edi"),      # arg 0: legitimate, index 0
        _write_reg(0x401104, "esi"),              # rsi (arg 1) reused as scratch...
        _read_into(0x401108, "sp_8", "rsi"),      # ...so neither the 64-bit form...
        _read_into(0x40110c, "sp_16", "esi"),     # ...nor the 32-bit half is an argument
    )

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["argument_confidence"] == "authoritative"
    assert "callee_under_recovered" not in call


def test_argument_confidence_fires_on_an_aarch64_w_register_argument_read_865(monkeypatch):
    """AArch64's other half of the same rule: `w2` IS `x2`, so a body reading it
    demonstrates a third argument register (`_arity_bv`'s default ABI list)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=2, arg_texts=["a", "b"])
    _with_llil(
        bv,
        _read_into(0x401100, "sp_0", "w0"),
        _read_into(0x401104, "sp_8", "w2"),
    )

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["callee_read_arity"] == 3 and call["callee_under_recovered"] is True


def test_argument_confidence_kept_for_a_decorated_callee_865(monkeypatch):
    """#865's second shape, refused rather than guessed at. A decorated name can
    carry IMPLICIT parameters -- a method's `this`, a by-value class return's sret
    slot -- which are argument REGISTERS the body legitimately reads and the
    parameter count never mentions, so a body-reads-more-than-declared comparison
    is meaningless there. Same refusal `_library_param_count` makes, same
    measured reason (3 of 4 raw #862 firings on a C++-heavy target were exactly
    this). Here the callee's body reads three registers against a 2-parameter
    prototype -- indistinguishable from the true positive above except for the
    name, which is the whole point."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=2, arg_texts=["a", "b"])
    callee = bv.get_function_at(0x401100)
    callee.name = "_ZN4Impl7combineEii"          # an Itanium-mangled method
    callee.raw_name = callee.name
    _with_llil(
        bv,
        _read_into(0x401100, "sp_0", "x0"),      # `this`
        _read_into(0x401104, "sp_8", "x1"),
        _read_into(0x401108, "sp_16", "x2"),
    )

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["argument_confidence"] == "authoritative"
    assert "callee_under_recovered" not in call


def test_undecorated_name_refusal_is_shared_with_the_library_cross_check_865(monkeypatch):
    """One rule, two callers: the witness and `_library_param_count` must agree
    about which names are trustworthy, or a name could be refused by one and
    trusted by the other."""
    bridge = _load_bridge(monkeypatch)
    mod = bridge.read_evidence
    for plain in ("memcpy", "hw_get_version", "__popcountdi2", "sub_401199"):
        assert mod._undecorated_name(plain) is True, plain
    for decorated in ("", "_ZN4Impl7combineEii", "_Rfoo", "_Dbar", "_Tbaz",
                      "?name@@YAXXZ", "$s4main3fooyyF", "memcpy.cold", "sym@GLIBC_2.2.5"):
        assert mod._undecorated_name(decorated) is False, decorated


def test_argument_confidence_zero_args_on_unknown_arity_not_demoted_648(monkeypatch):
    """#648: a genuinely void callee rendering NO arguments agrees with its recovered
    prototype -- nothing was invented, so it keeps `authoritative`. Demoting here
    would flag every `f()` call in a stripped binary for no gain."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _arity_bv(monkeypatch, instance, callee_params=0, arg_texts=[])

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["arity_unknown"] is False
    assert call["argument_confidence"] == "authoritative"


def test_argument_confidence_unknown_arity_below_abi_width_648(monkeypatch):
    """#648: the phantom-argument case that motivated the second confirmation --
    `sock_process(a, b, a)` where upstream has TWO parameters. The count does NOT
    saturate the ABI registers, so `abi_register_saturated` is absent, but the arity
    is still unknown and the confidence must still be demoted."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _arity_bv(monkeypatch, instance, callee_params=0, arg_texts=["a", "b", "a"])

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["arity_unknown"] is True
    assert call["argument_confidence"] == "inferred"
    assert "abi_register_saturated" not in call


@pytest.mark.parametrize(
    "declared, arguments, confidence, mismatch",
    [
        pytest.param(3, ["1"], "inferred", True, id="under-recovered"),
        pytest.param(3, ["1", "2", "3", "4"], "inferred", True, id="extra"),
        pytest.param(3, ["1", "2", "3"], "authoritative", False, id="matching"),
        pytest.param(3, [], "inferred", True, id="zero-rendered"),
        pytest.param(0, [], "authoritative", False, id="user-void"),
        pytest.param(0, ["1"], "inferred", True, id="user-void-extra"),
    ],
)
def test_argument_confidence_checks_user_prototype_742(
    monkeypatch, declared, arguments, confidence, mismatch,
):
    """A user prototype establishes the expected count, not the rendered count."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _arity_bv(monkeypatch, instance, callee_params=declared,
              arg_texts=arguments, user_type=True)

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["argument_source"] == "hlil"
    assert call["arity_unknown"] is False
    assert call["argument_confidence"] == confidence
    if mismatch:
        assert call["arity_mismatch"] is True
        assert call["declared_arity"] == declared
    else:
        assert "arity_mismatch" not in call
    assert "abi_register_saturated" not in call


@pytest.mark.parametrize("arguments", [[], ["1"]], ids=["zero-rendered", "nonzero-rendered"])
def test_argument_confidence_unavailable_user_prototype_742(monkeypatch, arguments):
    """User provenance alone cannot establish a count when neither API supplies it."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=0,
                   arg_texts=arguments, user_type=True)
    callee = bv.get_function_at(0x401100)
    callee.type.parameters = None
    callee.parameter_vars = None

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["arity_unknown"] is True
    assert call["argument_confidence"] == "inferred"
    assert "arity_mismatch" not in call


@pytest.mark.parametrize("fixed_count", [0, 1], ids=["no-fixed-parameters", "fixed-parameter"])
def test_argument_confidence_user_variadic_has_no_exact_count_742(monkeypatch, fixed_count):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=fixed_count,
                   arg_texts=["1", "2"], user_type=True)
    bv.get_function_at(0x401100).type.has_variable_arguments = True

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["arity_unknown"] is False
    assert call["argument_confidence"] == "authoritative"
    assert "arity_mismatch" not in call


def test_argument_confidence_arity_mismatch_more_args_than_declared_648(monkeypatch):
    """#648: a callee WITH a recovered prototype (2 declared params) where HLIL
    rendered 3 arguments -- an invented ABI-register arg riding along with the
    real ones. The arity itself is known (not `arity_unknown`), but the extra
    argument means the canonical list is not what the prototype declares, so it
    must demote and flag `arity_mismatch`."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _arity_bv(monkeypatch, instance, callee_params=2, arg_texts=["1", "2", "0x100"])

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["argument_confidence"] == "inferred"
    assert call["arity_mismatch"] is True
    assert call["arity_unknown"] is False


def test_argument_confidence_arity_mismatch_fewer_args_than_declared_648(monkeypatch):
    """#648/#704: the mirror case of the more-args test -- a callee WITH a
    recovered prototype (3 declared params) where HLIL rendered only 2
    arguments. The rendered list demonstrably does not match the declared
    signature (under-recovery), so this must demote and flag
    `arity_mismatch` just as the extra-argument direction does."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _arity_bv(monkeypatch, instance, callee_params=3, arg_texts=["1", "2"])

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["argument_confidence"] == "inferred"
    assert call["arity_mismatch"] is True
    assert call["arity_unknown"] is False


def test_argument_confidence_negative_control_matching_arity_stays_authoritative_648(monkeypatch):
    """#648 negative control (like a `memcpy`-shaped prototype): 3 declared params
    and exactly 3 rendered arguments -- nothing invented, stays `authoritative`
    and carries neither `arity_mismatch` nor `arity_unknown`."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _arity_bv(monkeypatch, instance, callee_params=3, arg_texts=["&dst", "&src", "0x40"])

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["argument_confidence"] == "authoritative"
    assert "arity_mismatch" not in call
    assert call["arity_unknown"] is False


def test_argument_confidence_arity_mismatch_note_names_true_provenance_704(monkeypatch):
    """#704 round 3: when the HLIL fold at this call's address is AMBIGUOUS (>1
    root, none at this address), `_call_arguments` falls back to MLIL (#661).
    On that fallback the rendered list is the CALLER's ABI registers, not the
    callee's operands -- exactly what #661 exists to stop presenting as callee
    arguments. `arity_mismatch` must not fire off that fallback list at all:
    firing it (and the old note hard-coding "HLIL rendered") would attribute a
    provenance the tool never established, reintroducing #661's defect class
    in prose."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x401100, "hw_get_version")
    callee.parameter_vars = [
        _FakeVariable(name=f"a{i}", storage=i, var_type="int64_t", identifier=i + 1)
        for i in range(2)
    ]
    callee.calling_convention = types.SimpleNamespace(
        int_arg_regs=[f"x{i}" for i in range(8)])
    caller = _FakeFunction(0x401400, "probe_device")
    # Two folded HLIL call roots, NEITHER at this call's address (0x401400) --
    # an ambiguous fold (#475/#476) `_call_arguments` cannot scope to a single
    # root, so it falls back to MLIL rather than guessing which root is ours.
    root_a = _FakeHLILInstruction("hw_get_version(x0, x1, x2, x3)",
                                  class_name="HighLevelILCall",
                                  address=0x401404, expr_index=10, instr_index=10)
    root_a.params = ["x0", "x1", "x2", "x3"]
    root_b = _FakeHLILInstruction("other_call(y0)", class_name="HighLevelILCall",
                                  address=0x401408, expr_index=11, instr_index=11)
    root_b.params = ["y0"]
    call_insn = _FakeLLILInstruction(0x401400, _FakeConstPtr(0x401100),
                                     hlils=[root_a, root_b])

    class _MappedMLIL:
        # The mapped/coalesced MLIL form -- names the CALLER's ABI registers,
        # not the callee's real operands (#661).
        params = ["arg1", "arg2", "arg3", "arg4"]

        def __str__(self):
            return "call(0x401100, arg1, arg2, arg3, arg4)"

    call_insn.mlil = _MappedMLIL()
    caller.basic_blocks = [_FakeBasicBlock(0x401400, 0x401404)]
    caller.low_level_il = [[call_insn]]
    bv = _FakeBV(functions=[callee, caller], instruction_lengths={0x401400: 4},
                 disassembly={0x401400: "bl hw_get_version"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    # The 2-parameter callee vs. the 4-argument fallback list is precisely the
    # shape that used to mis-render as an "HLIL" mismatch note (#704 round 3).
    assert call["argument_source"] == "mlil"
    assert len(call["arguments"]) == 4
    assert "arity_mismatch" not in call
    assert call["arity_unknown"] is False

    from bn.formatters import _render_function_evidence_text
    out = _render_function_evidence_text({
        "function": {"name": "probe_device", "address": "0x401400"},
        "prototype": "int32_t probe_device()", "calling_convention": "__cdecl",
        "thunk": {"is_candidate": False},
        "total_calls": 1, "matched_calls": 1, "offset": 0, "limit": None,
        "calls": [call],
    })
    assert "arguments: (mlil" in out
    assert "arity: MISMATCH" not in out
    assert "HLIL rendered" not in out


def test_argument_confidence_empty_hlil_list_is_a_count_742(monkeypatch):
    """An actual params=[] contradicts a positive prototype; it is not missing IL."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _arity_bv(monkeypatch, instance, callee_params=3, arg_texts=[])

    evidence = instance._function_evidence("active", "probe_device", context=0)
    call = evidence["calls"][0]
    assert call["argument_source"] == "hlil"
    assert call["arguments"] == []
    assert call["arity_mismatch"] is True
    assert call["declared_arity"] == 3
    assert call["arity_unknown"] is False
    assert call["argument_confidence"] == "inferred"

    from bn.formatters import _render_function_evidence_text
    out = _render_function_evidence_text(evidence)
    assert "arguments: (hlil inferred)" in out
    assert "arity: MISMATCH" in out
    assert "HLIL rendered 0 argument(s)" in out
    assert "prototype declares 3" in out


@pytest.mark.parametrize(
    "argument_fields",
    [
        pytest.param({}, id="absent"),
        pytest.param({"arguments": None}, id="null"),
        pytest.param({"arguments": {}}, id="wrong-shape"),
        pytest.param({"arguments": [None]}, id="invalid-member"),
        pytest.param({"arguments": [{"text": "1"}, None]}, id="partly-renderable"),
    ],
)
def test_argument_mismatch_text_does_not_invent_unavailable_count_742(
    monkeypatch, argument_fields,
):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _arity_bv(monkeypatch, instance, callee_params=3, arg_texts=[])
    evidence = instance._function_evidence("active", "probe_device", context=0)
    call = evidence["calls"][0]
    del call["arguments"]
    call.update(argument_fields)

    from bn.formatters import _render_function_evidence_text
    out = _render_function_evidence_text(evidence)
    assert "arguments: (hlil inferred)" in out
    assert "arity: MISMATCH" in out
    assert "rendered a different argument count" in out
    assert "argument(s) but" not in out


@pytest.mark.parametrize("malformed_first", [True, False])
def test_argument_mismatch_text_counts_are_row_local_742(monkeypatch, malformed_first):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _arity_bv(monkeypatch, instance, callee_params=3, arg_texts=[])
    evidence = instance._function_evidence("active", "probe_device", context=0)
    genuine_empty = evidence["calls"][0]
    malformed = {**genuine_empty, "address": "0x401404", "arguments": {}}
    evidence["calls"] = (
        [malformed, genuine_empty] if malformed_first else [genuine_empty, malformed]
    )

    from bn.formatters import _render_function_evidence_text
    out = _render_function_evidence_text(evidence)
    assert "HLIL rendered 0 argument(s)" in out
    assert "prototype declares 3" in out
    assert "rendered a different argument count" in out


def test_argument_confidence_missing_hlil_list_uses_mlil_provenance_742(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=3, arg_texts=[], user_type=True)
    insn = bv.get_function_at(0x401400).low_level_il[0][0]
    insn.hlils[0].params = None
    insn.mlil = types.SimpleNamespace(params=["1", "2"])

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["argument_source"] == "mlil"
    assert [arg["text"] for arg in call["arguments"]] == ["1", "2"]
    assert call["argument_confidence"] == "heuristic"
    assert call["arity_unknown"] is False
    assert "arity_mismatch" not in call


def test_argument_confidence_unavailable_argument_lists_are_heuristic_742(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=3, arg_texts=[], user_type=True)
    insn = bv.get_function_at(0x401400).low_level_il[0][0]
    del insn.hlils[0].params
    del insn.params

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert call["argument_source"] == "llil"
    assert call["arguments"] == []
    assert call["argument_confidence"] == "heuristic"
    assert call["arity_unknown"] is False
    assert "arity_mismatch" not in call


def test_argument_confidence_indirect_call_never_authoritative_648(monkeypatch):
    """#648: an indirect call has no resolvable callee, so its arity is
    unknowable even when HLIL produced a clean-looking argument list. The
    canonical field must never report `authoritative` for it."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x401800, "dispatch")
    call_expr = _FakeHLILInstruction("(*fp)(a, b)", class_name="HighLevelILCall",
                                     address=0x401800, expr_index=5, instr_index=5)
    call_expr.params = ["a", "b"]
    call_insn = _FakeLLILInstruction(0x401800, _FakeReg("r3"), hlils=[call_expr])
    caller.basic_blocks = [_FakeBasicBlock(0x401800, 0x401804)]
    caller.low_level_il = [[call_insn]]
    bv = _FakeBV(functions=[caller], instruction_lengths={0x401800: 4},
                 disassembly={0x401800: "blx r3"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    call = instance._function_evidence("active", "dispatch", context=0)["calls"][0]
    assert call["direct"] is False
    assert call["argument_source"] == "hlil"
    assert call["indirect_call"] is True
    assert call["callee_unresolved"] is True
    assert call["argument_confidence"] == "heuristic"


def test_argument_confidence_direct_call_to_unresolved_target_not_marked_indirect_704(monkeypatch):
    """#704: a DIRECT call whose destination is a resolved constant that
    matches no function (e.g. a mid-function branch target, or an address
    with no covering function) must NOT be marked `indirect_call` -- that
    would contradict the sibling `direct: true` in the same record. It is
    still `callee_unresolved` (the callee's arity is unknowable either way),
    so `argument_confidence` is demoted off `authoritative` regardless."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x401800, "dispatch")
    call_expr = _FakeHLILInstruction("fn_405000(a, b)", class_name="HighLevelILCall",
                                     address=0x401800, expr_index=5, instr_index=5)
    call_expr.params = ["a", "b"]
    # 0x405000 resolves to a real constant (a DIRECT call) but no function in
    # this view starts there or contains it -- exactly the case the dead
    # `fn_entry.get("start")` lookup used to (fail to) handle.
    call_insn = _FakeLLILInstruction(0x401800, _FakeConstPtr(0x405000), hlils=[call_expr])
    caller.basic_blocks = [_FakeBasicBlock(0x401800, 0x401804)]
    caller.low_level_il = [[call_insn]]
    bv = _FakeBV(functions=[caller], instruction_lengths={0x401800: 4},
                 disassembly={0x401800: "bl 0x405000"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    call = instance._function_evidence("active", "dispatch", context=0)["calls"][0]
    assert call["direct"] is True
    assert call.get("indirect_call") is not True
    assert call["callee_unresolved"] is True
    assert call["argument_confidence"] != "authoritative"


# --- #661: mlil: line / MLIL candidates come from the TRUE per-instruction MLIL --


def test_function_evidence_mlil_uses_true_per_instruction_form_661(monkeypatch):
    """#661: `mlil:` and the MLIL argument_candidates must come from the LLIL
    instruction's TRUE per-instruction MLIL (`insn.mlil`), which renders the
    callee's real operands (`0x401156(rdi, 3)`), not the mapped/coalesced form
    (`call(0x401156, arg1, arg2, ...)`) which names the CALLER's ABI registers."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x401156, "rotl8")
    caller = _FakeFunction(0x40118B, "encrypt")
    call_expr = _FakeHLILInstruction("rotl8(edi, 3)", class_name="HighLevelILCall",
                                     address=0x4011F2, expr_index=7, instr_index=7)
    call_expr.params = ["edi", "3"]
    call_insn = _FakeLLILInstruction(0x4011F2, _FakeConstPtr(0x401156), hlils=[call_expr])

    class _TrueMLIL:
        def __init__(self):
            self.params = ["rdi", "3"]

        def __str__(self):
            return "0x401156(rdi, 3)"

    call_insn.mlil = _TrueMLIL()
    # A stale mapped-MLIL form the fix must NOT prefer -- if the accessor
    # regresses to this it names the CALLER's registers instead of the callee's.
    call_insn.mapped_medium_level_il = "arg1, arg2 = call(0x401156, arg1, arg2)"
    caller.basic_blocks = [_FakeBasicBlock(0x4011F2, 0x4011F6)]
    caller.low_level_il = [[call_insn]]
    bv = _FakeBV(functions=[callee, caller], instruction_lengths={0x4011F2: 4},
                 disassembly={0x4011F2: "call rotl8"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    call = instance._function_evidence("active", "encrypt", context=0)["calls"][0]
    assert call["mlil"] == "0x401156(rdi, 3)"
    assert "arg1, arg2" not in call["mlil"]
    assert any(c["source"] == "mlil" and c["text"] == "3" for c in call["argument_candidates"])


def test_function_evidence_mlil_falls_back_to_mapped_form_when_true_mlil_absent_661(monkeypatch):
    """#661 fallback: when `insn.mlil` is unavailable, the mapped form is still
    used (better than nothing) rather than emitting no `mlil` line at all."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "send_message")
    caller = _FakeFunction(0x412470, "build_response")
    call_insn = _FakeLLILInstruction(0x4124A0, _FakeConstPtr(0x461746))
    call_insn.mapped_medium_level_il = "call(0x461746, arg1)"
    caller.basic_blocks = [_FakeBasicBlock(0x4124A0, 0x4124A4)]
    caller.low_level_il = [[call_insn]]
    bv = _FakeBV(functions=[callee, caller], instruction_lengths={0x4124A0: 4},
                 disassembly={0x4124A0: "bl send_message"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    call = instance._function_evidence("active", "build_response", context=0)["calls"][0]
    assert call["mlil"] == "call(0x461746, arg1)"


def test_function_evidence_mlil_falls_back_when_true_mlil_accessor_raises_661(monkeypatch):
    """#704 (hardening #661's fallback chain): BN's real `.mlil` property can
    itself raise (e.g. an internal `assert result is not None, "MLIL not
    present"`) rather than raise `AttributeError` -- `getattr(insn, "mlil",
    None)` only suppresses the latter, so a raising accessor used to
    propagate out of the whole `evidence function` op instead of falling
    back to the mapped form. This is pre-existing fragility symmetric with
    the mapped accessor (not a regression from preferring `.mlil`); wrapping
    it lets the documented fallback chain actually run."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _RaisingMLILInstruction(_FakeLLILInstruction):
        @property
        def mlil(self):
            raise AssertionError("MLIL not present")

    callee = _FakeFunction(0x461746, "send_message")
    caller = _FakeFunction(0x412470, "build_response")
    call_insn = _RaisingMLILInstruction(0x4124A0, _FakeConstPtr(0x461746))
    call_insn.mapped_medium_level_il = "call(0x461746, arg1)"
    caller.basic_blocks = [_FakeBasicBlock(0x4124A0, 0x4124A4)]
    caller.low_level_il = [[call_insn]]
    bv = _FakeBV(functions=[callee, caller], instruction_lengths={0x4124A0: 4},
                 disassembly={0x4124A0: "bl send_message"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    call = instance._function_evidence("active", "build_response", context=0)["calls"][0]
    assert call["mlil"] == "call(0x461746, arg1)"


def test_function_evidence_mlil_absent_when_no_mlil_form_available_661(monkeypatch):
    """#661: when neither `insn.mlil` nor `insn.mapped_medium_level_il` exists,
    emit NO `mlil` line rather than fabricate one."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "send_message")
    caller = _FakeFunction(0x412470, "build_response")
    call_insn = _FakeLLILInstruction(0x4124A0, _FakeConstPtr(0x461746))
    caller.basic_blocks = [_FakeBasicBlock(0x4124A0, 0x4124A4)]
    caller.low_level_il = [[call_insn]]
    bv = _FakeBV(functions=[callee, caller], instruction_lengths={0x4124A0: 4},
                 disassembly={0x4124A0: "bl send_message"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    call = instance._function_evidence("active", "build_response", context=0)["calls"][0]
    assert call["mlil"] is None


# --- #667: --field/--ptr-fields without --record-size must error, not silently
# drop the record-aware fields and run a plain scalar pointer scan -----------


def test_pointer_table_field_without_record_size_errors_667(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    target = _FakeFunction(0x401000, "handler")
    bv = _FakeBV(functions=[target], memory={0x3000: (0x401000).to_bytes(8, "little")})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._pointer_table("active", "0x3000", entries=1, ptr_fields=["0x8"])
    assert exc.value.status == "invalid_request"

    with pytest.raises(bridge.OperationFailure) as exc2:
        instance._pointer_table("active", "0x3000", entries=1, fields=["flag:u32@0"])
    assert exc2.value.status == "invalid_request"


def test_pointer_table_field_with_record_size_still_works_667(monkeypatch):
    """Negative control: --record-size alongside --ptr-fields/--field is
    unaffected by the new guard and still scans as a record table."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    target = _FakeFunction(0x401000, "handler")
    table = (0x401000).to_bytes(4, "little") + (5).to_bytes(4, "little")
    bv = _FakeBV(functions=[target], arch=_FakeArch(name="x86", address_size=4),
                 memory={0x3000: table})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._pointer_table("active", "0x3000", entries=1,
                                      record_size="0x8", ptr_fields=["0x0"])
    assert result["kind"] == "record_table"


# --- #673: thunk/veneer candidates must be EXTERNAL targets only ------------


def test_function_evidence_flags_tailcall_to_import_as_thunk_673(monkeypatch):
    """#673 positive control: a small function whose tail branch resolves to an
    IMPORTED function (not a PLT-named section, e.g. a GOT-indirected veneer) is
    still a legitimate thunk/veneer candidate."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "log_write")
    callee.symbol = _FakeSymbol("ImportedFunctionSymbol")
    veneer = _FakeFunction(0x500000, "j_log_write")
    veneer.basic_blocks = [_FakeBasicBlock(0x500000, 0x500004)]
    veneer.low_level_il = [[_FakeLLILInstruction(0x500000, _FakeConstPtr(0x461746), operation="LLIL_JUMP")]]
    bv = _FakeBV(
        functions=[callee, veneer],
        disassembly={0x500000: "b log_write"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_evidence("active", "j_log_write", context=0)
    assert result["thunk"]["is_candidate"] is True


def test_function_evidence_does_not_flag_local_tailcall_as_thunk_673(monkeypatch):
    """#673: a small function tail-calling a LOCAL defined function (e.g. an
    .init_array constructor tail-calling a local helper) is not a thunk/veneer FOR
    that helper -- it has its own real body and must not be masked behind a
    'go to X' pointer."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    helper = _FakeFunction(0x461746, "init_helper")  # no .symbol -> not imported
    ctor = _FakeFunction(0x500000, "init_array_0")
    ctor.basic_blocks = [_FakeBasicBlock(0x500000, 0x500004)]
    ctor.low_level_il = [[_FakeLLILInstruction(0x500000, _FakeConstPtr(0x461746), operation="LLIL_TAILCALL")]]
    bv = _FakeBV(
        functions=[helper, ctor],
        disassembly={0x500000: "b init_helper"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_evidence("active", "init_array_0", context=0)
    assert result["thunk"]["is_candidate"] is False


def test_function_evidence_local_tailcall_reports_target_without_thunk_claim_704(monkeypatch):
    """#704 round 3 follow-up: F1 correctly stopped flagging a local tail call
    as a thunk candidate (#673), but a bare `is_candidate: False` made a
    genuine local `j_`-style veneer invisible -- a false positive traded for a
    false negative. The resolved LOCAL target must still be surfaced (so
    `evidence function` shows the forwarding) WITHOUT asserting the
    thunk/veneer claim the tool never established for a local destination."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    helper = _FakeFunction(0x461746, "init_helper")  # no .symbol -> not imported
    ctor = _FakeFunction(0x500000, "init_array_0")
    ctor.basic_blocks = [_FakeBasicBlock(0x500000, 0x500004)]
    ctor.low_level_il = [[_FakeLLILInstruction(0x500000, _FakeConstPtr(0x461746), operation="LLIL_TAILCALL")]]
    bv = _FakeBV(
        functions=[helper, ctor],
        disassembly={0x500000: "b init_helper"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_evidence("active", "init_array_0", context=0)
    thunk = result["thunk"]
    assert thunk["is_candidate"] is False
    assert thunk["target"]["function"]["name"] == "init_helper"


def test_function_evidence_pseudo_c_fallback_does_not_flag_local_tailcall_673(monkeypatch):
    """#673/#704: the pseudo-C `/* tailcall */` fallback must not re-flag a
    function whose LLIL loop already positively identified a LOCAL,
    non-imported branch target. Before the fix, the loop's per-instruction
    `continue` (no `return`) let control fall through unconditionally to the
    unguarded fallback below, which set `is_candidate: True` with
    `target: None` for exactly this local-tailcall case whenever BN's
    pseudo-C happened to contain the `/* tailcall */` marker -- which the
    decompile-path caller (`read_decompile._thunk_veneer_warning`) already
    guarantees before ever calling in."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    helper = _FakeFunction(0x461746, "init_helper")  # no .symbol -> not imported
    ctor = _FakeFunction(0x500000, "init_array_0")
    ctor.basic_blocks = [_FakeBasicBlock(0x500000, 0x500004)]
    ctor.low_level_il = [[_FakeLLILInstruction(0x500000, _FakeConstPtr(0x461746), operation="LLIL_TAILCALL")]]
    bv = _FakeBV(
        functions=[helper, ctor],
        disassembly={0x500000: "b init_helper"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    # Force the pseudo-C decompile path to render a tailcall marker, exactly
    # as BN does for a real small-function tailcall -- the condition the
    # unguarded fallback used to re-flag regardless of what the LLIL loop
    # above already determined.
    monkeypatch.setattr(bridge.il_format, "_decompile_text",
                         lambda bv, f: "return init_helper() /* tailcall */")

    result = instance._function_evidence("active", "init_array_0", context=0)
    assert result["thunk"]["is_candidate"] is False


# --- #558: variadic (scanf-family) under-recovery warning + recovery -------


def test_variadic_format_helpers(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    il_format = bridge.il_format
    assert il_format._variadic_format_family("sscanf") == (1, True)
    assert il_format._variadic_format_family("__isoc99_sscanf") == (1, True)
    assert il_format._variadic_format_family("printf@plt") == (0, False)
    assert il_format._variadic_format_family("memcpy") is None
    # conversion counting: %% skipped, scanf %* suppressed, scanset handled
    assert il_format._count_format_conversions("%d %31s", is_scanf=True) == 2
    assert il_format._count_format_conversions("%d%%", is_scanf=True) == 1
    assert il_format._count_format_conversions("%*d %d", is_scanf=True) == 1
    assert il_format._count_format_conversions("%[^,],%d", is_scanf=True) == 2


def _variadic_caller_bv(monkeypatch, instance, *, params):
    callee = _FakeFunction(0x461746, "sscanf")
    caller = _FakeFunction(0x412470, "parse_line")
    call_expr = _FakeHLILInstruction("sscanf(...)", class_name="HighLevelILCall",
                                     address=0x4124A0, expr_index=30, instr_index=30)
    call_expr.params = params
    call_insn = _FakeLLILInstruction(0x4124A0, _FakeConstPtr(0x461746), hlils=[call_expr])
    caller.basic_blocks = [_FakeBasicBlock(0x4124A0, 0x4124A4)]
    caller.low_level_il = [[call_insn]]
    bv = _FakeBV(functions=[callee, caller], instruction_lengths={0x4124A0: 4},
                 disassembly={0x4124A0: "bl sscanf"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    return bv


def test_evidence_warns_variadic_under_recovery(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _variadic_caller_bv(monkeypatch, instance, params=["input"])   # only the fixed arg
    result = instance._function_evidence("active", "parse_line", context=0)
    call = result["calls"][0]
    variadic = call["variadic"]
    assert variadic["is_variadic"] is True and variadic["family"] == "scanf"
    assert variadic["under_recovered"] is True
    assert variadic["recovered_arg_count"] == 1
    assert "under-recovered" in variadic["warning"]
    # hoisted to the function-level warnings, address-tagged
    assert any("under-recovered" in w and "0x4124a0" in w.lower() for w in result["warnings"])


def test_evidence_variadic_recovers_format_and_is_not_under_when_complete(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _variadic_caller_bv(monkeypatch, instance,
                        params=["input", '"%d %d"', "&a", "&b"])   # fully recovered
    call = instance._function_evidence("active", "parse_line", context=0)["calls"][0]
    variadic = call["variadic"]
    assert variadic["format_string"] == "%d %d"
    assert variadic["format_conversions"] == 2
    assert variadic["expected_min_arg_count"] == 4
    assert variadic["under_recovered"] is False


def test_callsites_variadic_callee_hint(monkeypatch):
    # #558: callsites to an imported variadic callee carry a steer to the
    # argument-recovery views on every row.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "sscanf")
    fn = _FakeFunction(0x500000, "parse_line")
    fn.basic_blocks = [_FakeBasicBlock(0x500010, 0x500015)]
    fn.low_level_il = [[_FakeLLILInstruction(0x500010, _FakeConstPtr(0x461746))]]
    bv = _FakeBV(functions=[callee, fn], instruction_lengths={0x500010: 5},
                 disassembly={0x500010: "bl sscanf"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    rows = _callsites_items(instance, "active", "sscanf", within_identifiers=["parse_line"], context=1)
    hint = rows[0]["callee_variadic"]
    assert hint["is_variadic"] is True and hint["name"] == "sscanf" and hint["family"] == "scanf"


def test_callsites_no_variadic_hint_for_ordinary_callee(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "memcpy")
    fn = _FakeFunction(0x500000, "caller")
    fn.basic_blocks = [_FakeBasicBlock(0x500010, 0x500015)]
    fn.low_level_il = [[_FakeLLILInstruction(0x500010, _FakeConstPtr(0x461746))]]
    bv = _FakeBV(functions=[callee, fn], instruction_lengths={0x500010: 5},
                 disassembly={0x500010: "bl memcpy"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    rows = _callsites_items(instance, "active", "memcpy", within_identifiers=["caller"], context=1)
    assert "callee_variadic" not in rows[0]


# --- #561: existing-annotation counts in the orient digest -----------------


def test_annotation_summary_counts(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _AutoSym:
        def __init__(self, auto):
            self.auto = auto

    documented = _FakeFunction(0x1000, "documented")
    documented.comment = "prior-run note"
    documented.comments = {0x1004: "function-local address note"}
    plain = _FakeFunction(0x2000, "plain")
    bv = _FakeBV(functions=[documented, plain],
                 symbols=[_AutoSym(False), _AutoSym(False), _AutoSym(True)],
                 comments={0x1000: "an address comment", 0x1004: "another"})
    summary = bridge.read_listing._annotation_summary(instance.ctx, bv)
    assert summary["comments"] == 3
    assert summary["function_comments"] == 1
    assert summary["user_symbols"] == 2
    assert [item["address"] for item in summary["comment_locations"]] == [
        "0x1000",
        "0x1004",
        "0x1004",
    ]
    assert summary["comment_locations"][-1] == {
        "name": "documented",
        "address": "0x1004",
        "comment": "function-local address note",
    }
    assert summary["function_comment_locations"] == [
        {
            "name": "documented",
            "address": "0x1000",
            "comment": "prior-run note",
        }
    ]
    assert summary["locations_truncated"] is False
    # #733 F2: an `_AutoSym` carries no `name` at all, which resolves to `""`.
    # An empty name is NOT a placeholder shape (the conservative direction), so
    # both non-auto symbols here count as analyst work. Pinned, not incidental.
    assert summary["analyst_symbols"] == 2
    assert summary["placeholder_symbols"] == 0


def test_annotation_summary_does_not_hide_an_unreadable_local_comment_store(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class UnreadableComments(_FakeFunction):
        @property
        def comments(self):
            raise RuntimeError("function comments unavailable")

    bv = _FakeBV(functions=[UnreadableComments(0x1000, "parse_record")])
    with pytest.raises(RuntimeError, match="function comments unavailable"):
        bridge.read_listing._annotation_summary(instance.ctx, bv)


def test_annotation_summary_splits_loader_placeholders_from_analyst_symbols(monkeypatch):
    """#733 F2: `auto=False` is not a measure of analyst work.

    BN's loaders synthesize named symbols and mark them non-auto, so a freshly
    loaded binary with zero analyst effort reported hundreds of "user symbols".
    The raw count stays (lossless); the split is what orientation and the
    contamination gate key on.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _NamedSym:
        def __init__(self, name, address):
            self.auto = False
            self.name = name
            self.address = address

    placeholders = [
        _NamedSym("func_10a4c0", 0x10A4C0),
        _NamedSym("sub_10b220", 0x10B220),
        _NamedSym("start_routine_10c000", 0x10C000),
        _NamedSym("init_routine", 0x10C100),
        _NamedSym("_init", 0x10C200),
        _NamedSym("_fini", 0x10C300),
        # Measured, not assumed: a freshly loaded stripped dynamic ELF reports
        # exactly one non-auto symbol, `main`, synthesized by BN's loader from
        # the `__libc_start_main` argument. Without this row every fresh ELF
        # hinted "1 analyst symbol(s)" and the benchmark gate refused it.
        _NamedSym("main", 0x10C400),
    ]
    analyst = _NamedSym("parse_header", 0x10D000)
    bv = _FakeBV(functions=[], symbols=[*placeholders, analyst])

    summary = bridge.read_listing._annotation_summary(instance.ctx, bv)

    assert summary["user_symbols"] == 8          # raw non-auto count, unchanged
    assert summary["placeholder_symbols"] == 7
    assert summary["analyst_symbols"] == 1
    assert summary["analyst_symbol_locations"] == [
        {"name": "parse_header", "address": "0x10d000"}
    ]


def test_annotation_summary_discloses_every_loader_helper_exclusion(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    names = ["init", "fini", "dest", "destr", "destr_1a2b", "compar", "compar_1A2B"]
    # Exceed the legacy location sample: every exclusion still needs a reason.
    symbols = [
        types.SimpleNamespace(
            auto=False, name=name, address=0x1000 + index * 0x10,
            namespace="BNINTERNALNAMESPACE",
        )
        for index, name in enumerate(names * 4)
    ]
    analyst = types.SimpleNamespace(
        auto=False, name="parse_record", address=0x2000,
        namespace="BNINTERNALNAMESPACE",
    )
    bv = _FakeBV(functions=[], symbols=[*symbols, analyst])

    summary = bridge.read_listing._annotation_summary(instance.ctx, bv)

    assert summary["user_symbols"] == 29
    assert summary["placeholder_symbols"] == 28
    assert summary["analyst_symbols"] == 1
    assert summary["analyst_symbol_locations"] == [
        {"name": "parse_record", "address": "0x2000"}
    ]
    assert summary["symbol_exclusions"] == [
        {"name": symbol.name, "address": hex(symbol.address), "reason": "name_shape"}
        for symbol in symbols
    ]
    assert summary["locations_truncated"] is True


def test_annotation_summary_counts_survive_an_unreadable_symbol(monkeypatch):
    """One bad symbol must not fabricate a pristine view.

    `assert_unannotated` now REFUSES on `analyst_symbols`, so the inverse of a
    fabricated refusal is a fabricated clean certification: a summary that
    collapses every count to zero because one symbol out of hundreds has an
    unreadable `address` reports contaminated benchmark data as untouched. The
    counts are therefore exact and only the sample ROW degrades.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _NamedSym:
        def __init__(self, name, address):
            self.auto = False
            self.name = name
            self.address = address

    class _BadAddressSym:
        auto = False
        name = "parse_trailer"

        @property
        def address(self):
            raise RuntimeError("symbol address unavailable")

    class _BadAddressPlaceholder(_BadAddressSym):
        name = "init"

    class _BadProvenanceSym:
        name = "dispatch_entry"
        address = 0x10E100

        @property
        def auto(self):
            raise RuntimeError("symbol provenance unavailable")

    bv = _FakeBV(functions=[], symbols=[
        _NamedSym("parse_header", 0x10D000),
        _BadAddressSym(),
        _NamedSym("_init", 0x10D100),
        _BadProvenanceSym(),
        _BadAddressPlaceholder(),
    ])

    summary = bridge.read_listing._annotation_summary(instance.ctx, bv)

    # Every symbol is counted, even if its address or provenance is unreadable.
    assert summary["user_symbols"] == 5
    assert summary["placeholder_symbols"] == 2
    assert summary["analyst_symbols"] == 3
    # Only the row whose address is unreadable is missing from the samples.
    assert [row["name"] for row in summary["analyst_symbol_locations"]] == [
        "parse_header", "dispatch_entry"
    ]
    assert summary["locations_truncated"] is True
    assert summary["symbol_exclusions"] == [
        {"name": "_init", "address": "0x10d100", "reason": "name_shape"},
        {"name": "init", "address": None, "reason": "name_shape"},
    ]


def test_an_unreadable_symbol_enumeration_refuses_instead_of_reporting_zero(monkeypatch):
    """The other half: a view whose symbols cannot be ENUMERATED must not be
    published as pristine.

    The summary used to answer an enumeration failure with five zeros, and
    `assert_unannotated` refuses on `analyst_symbols` -- so a generator that
    raises after yielding two inherited renames turned contaminated benchmark
    data into a clean certification. It now propagates, and `_orient_digest`
    degrades the block to the `unavailable` marker the kernel's digest contract
    refuses (#733 F2 review).
    """
    bridge = _load_bridge(monkeypatch)
    inst = bridge.BinaryNinjaBridge()

    class _NamedSym:
        def __init__(self, name, address):
            self.auto = False
            self.name = name
            self.address = address

    def _hostile_symbols():
        yield _NamedSym("parse_header", 0x401000)
        yield _NamedSym("emit_record", 0x401100)
        raise RuntimeError("symbol table unavailable")

    class _BV:
        functions: list = []
        address_comments: dict = {}

        def get_symbols(self):
            return _hostile_symbols()

    bv = _BV()
    with pytest.raises(RuntimeError, match="symbol table unavailable"):
        bridge.read_listing._annotation_summary(inst.ctx, bv)

    monkeypatch.setattr(inst, "_target_info",
                        lambda sel: {"basename": "netsvcd", "filename": "/opt/netsvcd",
                                     "analyzed": True, "analysis_state": "full"})
    monkeypatch.setattr(bridge.read_misc, "_imports", lambda ctx, sel, **k: {"total_symbols": 0})
    monkeypatch.setattr(bridge.read_misc, "_strings",
                        lambda ctx, sel, **k: {"items": [], "total": 0})
    monkeypatch.setattr(bridge.read_misc, "_sections",
                        lambda ctx, sel, **k: {"items": [], "total": 0})
    monkeypatch.setattr(bridge.read_listing, "_list_functions", lambda ctx, sel, **k: {"total": 0})
    monkeypatch.setattr(inst, "_resolve_view", lambda sel: bv)

    ea = inst._orient_digest(None)["existing_annotations"]
    assert "annotation counts unavailable" in ea["unavailable"]
    assert "analyst_symbols" not in ea      # nothing is claimed, so nothing passes


def test_annotation_summary_credits_debug_info_names_to_the_binary(monkeypatch):
    """Names the DWARF importer recovered are the binary's, not an analyst's.

    BN marks a plain symtab/dynsym name `auto=True` (the summary never sees
    those), but a DWARF-recovered name arrives `auto=False` -- measured: a
    `cc -g -O0` build of a three-function program reported two "analyst
    symbols" on a view nobody had touched, and the contamination gate refused
    it. The match is keyed on the name AND the importer's address, so an
    analyst rename still counts -- including one that REUSES a name the debug
    info supplied for a different function, which a bare-name match credited
    to the loader (#733 F2 review).
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _NamedSym:
        def __init__(self, name, address):
            self.auto = False
            self.name = name
            self.address = address

    class _DebugFn:
        def __init__(self, short_name, address):
            self.short_name = short_name
            self.full_name = short_name
            self.address = address

    bv = _FakeBV(functions=[], symbols=[
        _NamedSym("parse_header", 0x401149),
        _NamedSym("emit_record", 0x40115B),
        _NamedSym("parse_trailer", 0x401179),     # the analyst's own rename
        # The collision: renamed to a name the debug info gave a DIFFERENT
        # function, at an address the debug info never named.
        _NamedSym("parse_header", 0x4020A0),
        _NamedSym("init", 0x403000),             # debug provenance beats name shape
    ])
    bv.debug_info = types.SimpleNamespace(functions=[
        _DebugFn("parse_header", 0x401149),
        _DebugFn("emit_record", 0x40115B),
        _DebugFn("init", 0x403000),
        _DebugFn("printf", 0),
    ])

    summary = bridge.read_listing._annotation_summary(instance.ctx, bv)

    assert summary["user_symbols"] == 5
    assert summary["placeholder_symbols"] == 3
    assert summary["analyst_symbols"] == 2
    assert [(row["name"], row["address"]) for row
            in summary["analyst_symbol_locations"]] == [
        ("parse_trailer", "0x401179"), ("parse_header", "0x4020a0")
    ]
    assert summary["symbol_exclusions"] == [
        {"name": "parse_header", "address": "0x401149", "reason": "debug_info"},
        {"name": "emit_record", "address": "0x40115b", "reason": "debug_info"},
        {"name": "init", "address": "0x403000", "reason": "debug_info"},
    ]


def test_annotation_summary_fails_closed_on_unreadable_debug_info(monkeypatch):
    """A debug-info source that cannot be read must not reclassify analyst work
    as the binary's own: no names means no exclusions, so the symbols count as
    analyst work and the gate refuses rather than certifying (#733 F2 review)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _NamedSym:
        def __init__(self, name, address):
            self.auto = False
            self.name = name
            self.address = address

    class _Hostile:
        @property
        def functions(self):
            raise RuntimeError("debug info unavailable")

    bv = _FakeBV(functions=[], symbols=[_NamedSym("parse_header", 0x401149)])
    bv.debug_info = _Hostile()

    summary = bridge.read_listing._annotation_summary(instance.ctx, bv)

    assert summary["analyst_symbols"] == 1
    assert summary["placeholder_symbols"] == 0
    assert bridge.read_listing._debug_info_symbols(bv) == frozenset()


def test_a_trailing_newline_name_is_not_a_placeholder(monkeypatch):
    """The pattern is anchored with `fullmatch`, not a trailing `$`.

    `$` also matches immediately before a trailing newline, and an ELF string
    table is NUL- not newline-terminated, so `"main\\n"` is a legal symbol name
    that `$` classified as a loader placeholder -- excluding it from the
    counter `assert_unannotated` refuses on, which is a fail-OPEN on the gate
    (#733 F2 review).
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _NamedSym:
        def __init__(self, name, address):
            self.auto = False
            self.name = name
            self.address = address

    bv = _FakeBV(functions=[], symbols=[
        _NamedSym("main", 0x401000),        # the placeholder shape
        _NamedSym("main\n", 0x401010),      # NOT the placeholder shape
        _NamedSym("sub_401020\n", 0x401020),
    ])

    summary = bridge.read_listing._annotation_summary(instance.ctx, bv)

    assert summary["placeholder_symbols"] == 1
    assert summary["analyst_symbols"] == 2


def test_orient_hint_is_null_when_only_loader_placeholders_are_present(monkeypatch):
    """#733 F2: the provenance hint must not fire on the loader's own names.

    A hint that tells the reader to discount current-run analysis on a view
    whose annotations are ALL current-run is the defect; one analyst rename is
    the boundary where it must fire.
    """
    bridge = _load_bridge(monkeypatch)
    inst = bridge.BinaryNinjaBridge()
    # A non-`.bndb` filename, so `analysis_cache_restored` cannot drive the hint.
    monkeypatch.setattr(inst, "_target_info",
                        lambda sel: {"basename": "netsvcd", "filename": "/opt/netsvcd",
                                     "analyzed": True, "analysis_state": "full"})
    monkeypatch.setattr(bridge.read_misc, "_imports", lambda ctx, sel, **k: {"total_symbols": 0})
    monkeypatch.setattr(bridge.read_misc, "_strings",
                        lambda ctx, sel, **k: {"items": [], "total": 0})
    monkeypatch.setattr(bridge.read_misc, "_sections",
                        lambda ctx, sel, **k: {"items": [], "total": 0})
    monkeypatch.setattr(bridge.read_listing, "_list_functions", lambda ctx, sel, **k: {"total": 0})

    class _NamedSym:
        def __init__(self, name, address):
            self.auto = False
            self.name = name
            self.address = address

    bv = _FakeBV(functions=[], symbols=[_NamedSym("func_401000", 0x401000),
                                        _NamedSym("_init", 0x401100)])
    monkeypatch.setattr(inst, "_resolve_view", lambda sel: bv)

    ea = inst._orient_digest(None)["existing_annotations"]
    assert ea["user_symbols"] == 2
    assert ea["analyst_symbols"] == 0
    assert ea["placeholder_symbols"] == 2
    assert ea["analysis_cache_restored"] is False
    assert ea["provenance_hint"] is None

    bv._symbols = [*bv._symbols, _NamedSym("parse_header", 0x401200)]
    ea = inst._orient_digest(None)["existing_annotations"]
    assert ea["analyst_symbols"] == 1
    assert "1 analyst symbol(s)" in ea["provenance_hint"]
    assert "2 loader placeholder(s) excluded" in ea["provenance_hint"]



def test_orient_surfaces_existing_annotations(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    inst = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(inst, "_target_info",
                        lambda sel: {"basename": "x.bndb", "filename": "/c/x.bndb",
                                     "analyzed": True, "analysis_state": "full"})
    monkeypatch.setattr(bridge.read_misc, "_imports", lambda ctx, sel, **k: {"total_symbols": 0})
    monkeypatch.setattr(bridge.read_misc, "_strings",
                        lambda ctx, sel, **k: {"items": [], "total": 0})
    monkeypatch.setattr(bridge.read_misc, "_sections", lambda ctx, sel, **k: {"items": [], "total": 0})
    monkeypatch.setattr(bridge.read_listing, "_list_functions", lambda ctx, sel, **k: {"total": 0})
    bv = _FakeBV(functions=[], comments={0x1: "c"})
    monkeypatch.setattr(inst, "_resolve_view", lambda sel: bv)

    d = inst._orient_digest(None)
    ea = d["existing_annotations"]
    assert ea["comments"] == 1
    assert ea["analysis_cache_restored"] is True
    assert "predate this run" in ea["provenance_hint"]


def test_class_list_zero_result_reports_its_inputs_653(monkeypatch):
    """#653.6: `classes: 0 shown of 0` is correct on a C target and IDENTICAL to
    what a clustering failure would print, so an agent could not tell "this target
    is C" from "the lens failed" -- one spent two extra calls proving RTTI absence
    by hand (`strings --regex '_ZTV|_ZTI'`). Report the empty inputs instead."""
    bridge = _load_bridge(monkeypatch)
    read_class = bridge.read_class

    class _CBV:
        functions = [_FakeFunction(0x401000, "parse_header"),
                     _FakeFunction(0x401100, "main")]
        def get_symbols(self):
            return []

    bv = _CBV()

    class _Ctx:
        def _resolve_view(self, s): return bv
        def _bases_for(self, b, rec): return []

    out = read_class._class_list(_Ctx(), None)
    assert out["total"] == 0
    assert out["inputs"] == {"demangled_cxx_methods": 0, "rtti_typeinfo_symbols": 0,
                             "rtti_vtable_symbols": 0}

    count = read_class._class_list(_Ctx(), None, count_only=True)
    assert count["count"] == 0 and count["inputs"]["rtti_vtable_symbols"] == 0

    from bn.formatters import _render_class_list_text
    text = _render_class_list_text(out)
    assert "demangled C++ symbols: 0" in text
    assert "no C++ type evidence" in text


# --- #469 call-descriptor evidence (`evidence calls`) ------------------------
#
# The op reads a stack descriptor built by each caller and reports it FIELD BY
# FIELD, so the whole answer is a mapping from a write's offset to a declared
# field's name. A mapping that slips by one slot reports the wrong value under
# the right name -- an undetectable wrong answer, not a crash -- which is what
# these three tests exist to make loud. The op had no functional test at all
# before this: the only mentions of it were a name in the op-registry
# membership pin and a fake-CLI page loop that never reaches the handler.

def _desc_caller(name, start, end, instructions):
    caller = _FakeFunction(start, name)
    caller.basic_blocks = [_FakeBasicBlock(start, end)]
    caller.mlil = types.SimpleNamespace(instructions=instructions)
    return caller


def _desc_stack_var(name, storage, identifier):
    return _FakeVariable(name=name, storage=storage, var_type="void*",
                         identifier=identifier)


def _desc_call(address, *params):
    return _vc_expr("MLIL_CALL", address=address, params=list(params), output=[])


def test_call_descriptors_maps_each_write_to_the_field_at_its_own_offset(monkeypatch):
    """Each declared field reports the constant written AT ITS OWN OFFSET, and a
    `ptr` field resolves the callback it points at.

    The two scalars are deliberately distinguishable (7 vs 0x11) and adjacent:
    a mapping that walks the declared fields against the writes in order, or
    slips a slot, swaps them and this fails. The pointer field proves the other
    half of the contract -- a descriptor slot is only useful once the callback
    it holds is named -- and it is reached through a register copy of `&desc`,
    which is how a real caller passes it.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    desc = _desc_stack_var("desc", -0x30, 1)
    argreg = _FakeVariable(name="rsi", storage=0, var_type="void*", identifier=2,
                           source_type="RegisterVariableSourceType")
    callback = _FakeFunction(0x401500, "on_event_cb")
    callee = _FakeFunction(0x401200, "register_handler")
    caller = _desc_caller("install_handlers", 0x401000, 0x401030, [
        _vc_expr("MLIL_SET_VAR_FIELD", dest=desc, offset=0, size=4,
                 src=_vc_expr("MLIL_CONST", constant=7), address=0x401010),
        _vc_expr("MLIL_SET_VAR_FIELD", dest=desc, offset=4, size=4,
                 src=_vc_expr("MLIL_CONST", constant=0x11), address=0x401014),
        _vc_expr("MLIL_SET_VAR_FIELD", dest=desc, offset=8, size=8,
                 src=_vc_expr("MLIL_CONST_PTR", constant=0x401500), address=0x401018),
        _vc_expr("MLIL_SET_VAR", dest=argreg, size=8,
                 src=_vc_expr("MLIL_ADDRESS_OF", src=desc), address=0x40101C),
        _desc_call(0x401020, _vc_expr("MLIL_CONST", constant=0),
                   _vc_expr("MLIL_VAR", src=argreg)),
    ])
    bv = _FakeBV(functions=[callee, caller, callback],
                 arch=_FakeArch(name="x86_64", address_size=8),
                 code_refs={0x401200: [_FakeCodeRef(0x401020)]})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._call_descriptor_evidence(
        None, "register_handler", arg_index=1,
        field_specs=["version:u32@0", "flags:u32@4", "on_event:ptr@0x8"])

    assert result["kind"] == "call_descriptors"
    assert result["total"] == 1 and not result["warnings"]
    row = result["items"][0]
    assert row["status"] == "ok"
    assert row["caller"] == "install_handlers" and row["call_address"] == "0x401020"
    fields = {f["name"]: f for f in row["fields"]}
    assert fields["version"]["value"] == "0x7", row["fields"]
    assert fields["flags"]["value"] == "0x11", row["fields"]
    assert fields["on_event"]["value"] == "0x401500"
    assert fields["on_event"]["symbol"] == "on_event_cb"
    # Each value is attributed to the instruction that wrote it, so the claim can
    # be checked in the listing rather than taken on trust.
    assert fields["version"]["source_address"] == "0x401010"
    assert fields["flags"]["source_address"] == "0x401014"
    assert all(f["status"] == "resolved" for f in row["fields"])


def test_call_descriptors_slices_a_write_combined_store_per_field(monkeypatch):
    """A single wide store covering two fields is SLICED to each field's bytes.

    An optimizer merges `version = 7; flags = 0x11` into one 8-byte store of
    0x11_00000007. Reporting the whole store under both names -- or shifting by
    the wrong end -- is the same wrong-value-under-the-right-name failure as a
    slot slip, so both halves are asserted, little-endian: the low word is
    `version`, the high word is `flags`.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    desc = _desc_stack_var("desc", -0x20, 1)
    callee = _FakeFunction(0x402200, "register_handler")
    caller = _desc_caller("install_one", 0x402000, 0x402030, [
        _vc_expr("MLIL_SET_VAR_FIELD", dest=desc, offset=0, size=8,
                 src=_vc_expr("MLIL_CONST", constant=(0x11 << 32) | 7),
                 address=0x402010),
        _desc_call(0x402020, _vc_expr("MLIL_ADDRESS_OF", src=desc)),
    ])
    bv = _FakeBV(functions=[callee, caller],
                 arch=_FakeArch(name="x86_64", address_size=8),
                 code_refs={0x402200: [_FakeCodeRef(0x402020)]})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._call_descriptor_evidence(
        None, "register_handler", arg_index=0,
        field_specs=["version:u32@0", "flags:u32@4"])

    fields = {f["name"]: f for f in result["items"][0]["fields"]}
    assert fields["version"]["value"] == "0x7", fields
    assert fields["flags"]["value"] == "0x11", fields


def test_call_descriptors_reports_a_callsite_it_could_not_resolve(monkeypatch):
    """A callsite whose argument is not a local descriptor is REPORTED as such.

    Dropping it would turn "I could not read this one" into "this one has no
    fields", which is the shape #469 was filed against: the row keeps its
    caller and address, carries the reason, and the count reaches the warning
    so a reader knows the sweep was incomplete.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    desc = _desc_stack_var("desc", -0x30, 1)
    callee = _FakeFunction(0x403300, "register_handler")
    resolved = _desc_caller("install_local", 0x403000, 0x403030, [
        _vc_expr("MLIL_SET_VAR_FIELD", dest=desc, offset=0, size=4,
                 src=_vc_expr("MLIL_CONST", constant=3), address=0x403010),
        _desc_call(0x403020, _vc_expr("MLIL_ADDRESS_OF", src=desc)),
    ])
    # A static descriptor in .data: the argument is a constant pointer, not the
    # address of a stack local, so nothing can be recovered from the caller.
    static = _desc_caller("install_static", 0x403100, 0x403130, [
        _desc_call(0x403120, _vc_expr("MLIL_CONST_PTR", constant=0x500000)),
    ])
    bv = _FakeBV(functions=[callee, resolved, static],
                 arch=_FakeArch(name="x86_64", address_size=8),
                 code_refs={0x403300: [_FakeCodeRef(0x403020), _FakeCodeRef(0x403120)]})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._call_descriptor_evidence(
        None, "register_handler", arg_index=0, field_specs=["version:u32@0"])

    rows = {row["caller"]: row for row in result["items"]}
    assert rows["install_local"]["status"] == "ok"
    assert rows["install_static"]["status"] == "not_a_local_descriptor"
    assert rows["install_static"]["call_address"] == "0x403120"
    assert rows["install_static"]["fields"] == []
    assert result["total"] == 2
    assert any("1 callsite(s) did not resolve arg 0" in w for w in result["warnings"])


def test_call_descriptors_refuses_a_run_with_no_declared_field(monkeypatch):
    """No `--field` means no question was asked. Returning an empty-field row per
    callsite would read as "this descriptor has nothing in it", so the op refuses
    up front instead."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(instance.ctx, "_resolve_view",
                        lambda selector: pytest.fail("refusal must precede view resolution"))

    with pytest.raises(bridge.OperationFailure) as excinfo:
        instance._call_descriptor_evidence(None, "register_handler", arg_index=0,
                                           field_specs=[])
    assert excinfo.value.status == "invalid_request"


def test_function_evidence_discloses_quick_analysis_state(monkeypatch):
    """#820: `evidence function` answers on a --quick view (matrix: partial), so
    the card carries the view's analysis state -- the call/ABI read is real, its
    fidelity is not."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x412470, "build_response")
    bv = _FakeBV(functions=[caller])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    _install_fake_pseudo_c(
        monkeypatch, bridge, caller, [[(0x412470, "int32_t build_response()")]]
    )

    full = instance._function_evidence("active", "build_response", context=0)
    assert full["analysis_state"] == "full"
    assert full["partial"] is False

    bridge._quick_loaded_views.add(bv)
    try:
        quick = instance._function_evidence("active", "build_response", context=0)
    finally:
        bridge._quick_loaded_views.discard(bv)

    assert quick["analysis_state"] == "quick"
    assert quick["partial"] is True


def test_render_function_evidence_text_warns_when_quick_loaded():
    """#820: the warning leads the card -- including the empty-call-set card,
    which returns early, because a quick-loaded read with no calls is exactly
    what a reader would take for a complete answer."""
    from bn.formatters import _render_function_evidence_text
    value = {
        "function": {"name": "build_response", "address": "0x412470"},
        "prototype": "int32_t build_response()", "calling_convention": "__cdecl",
        "thunk": {"is_candidate": False},
        "total_calls": 0, "matched_calls": 0, "offset": 0, "limit": None,
        "calls": [], "warnings": [],
        "analysis_state": "quick", "partial": True,
    }
    out = _render_function_evidence_text(value)
    lines = out.splitlines()
    assert lines[0] == "WARNING: target is quick-loaded; function evidence is partial. "                        "Run `bn refresh` for full analysis."
    assert lines[1] == "build_response @ 0x412470"

    full = _render_function_evidence_text({**value, "analysis_state": "full", "partial": False})
    assert "WARNING" not in full
    assert full.splitlines()[0] == "build_response @ 0x412470"


def test_library_cross_check_refuses_a_locally_defined_name_collision_759(monkeypatch):
    """#862 review blocker: the lookup keyed on the callee's NAME alone, so a
    statically linked image defining its OWN ordinary-named function under a name
    an attached library also carries got EVERY call row to it demoted -- a false
    demotion on a collision that says nothing about the recovery.

    No import symbol and an ordinary identifier means the library is describing
    something else, so it is not evidence here."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=2, arg_texts=["a", "b"])
    _with_type_library(bv, symbol="hw_get_version", params=3)   # no _as_import

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]

    assert "prototype_unverified" not in call
    assert call["argument_confidence"] == "authoritative"


def test_library_cross_check_applies_to_a_reserved_identifier_759(monkeypatch):
    """The other side of that gate, and the shape the whole change rests on.
    `__popcountdi2` is statically linked from libgcc, so it is NOT an import --
    an import-only gate would have thrown away the only true positive measured.
    C11 7.1.3 reserves a leading `__` to the implementation, so a conforming
    program cannot define it and the library still describes this callee."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=0, arg_texts=[])
    next(f for f in bv.functions if f.name == "hw_get_version").name = "__popcountdi2"
    _with_type_library(bv, symbol="__popcountdi2", params=1,
                       lib_name="libgcc_s_x86_64.so.1")

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]

    assert call["argument_confidence"] == "inferred"
    assert call["prototype_unverified"] is True
    assert call["library_arity"] == 1
    assert call["library_source"] == "libgcc_s_x86_64.so.1"


def test_a_library_contradiction_demotes_even_with_a_user_prototype_759(monkeypatch):
    """#862 DOGFOOD finding, and a deliberate reversal of the round-1 precedence
    fix. Suppressing the cross-check on `has_user_type` turned the whole feature
    off wherever it matters: BN sets that flag on almost every function and never
    clears it (`proto set --preview` is refused precisely because there is no API
    to clear it), so after a `bn save` + reopen it reads True for essentially
    everything -- measured 104/104 imports and 853/853 locals on a reopened view,
    957/959 on a corpus database. Against a saved `.bndb` -- the normal case --
    the gate was a no-op versus base, including for the `__popcountdi2` callee it
    was built for.

    So the demotion is no longer suppressed. The round-1 concern was that it must
    not SILENTLY overrule an analyst's own statement, and that is met by
    disclosure rather than by privilege: a pinned prototype is demoted like any
    other, while the row names both counts and the library and the text line
    states that the contradiction is REPORTED, not judged -- so an analyst who
    pinned one sees the claim, confirms with `bn proto get`, and can conclude the
    library is wrong for this binary.

    This is the shape the pre-dogfood tests never exercised: a callee whose type
    reads as user-set, which is what every function on a reopened database looks
    like."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=2,
                   arg_texts=["a", "b"], user_type=True)
    _with_type_library(bv, symbol="hw_get_version", params=3)
    _as_import(bv)

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]

    assert call["argument_confidence"] == "inferred"
    assert call["prototype_unverified"] is True
    assert call["declared_arity"] == 2
    assert call["library_arity"] == 3
    # The analyst-facing half of the trade: the contradiction is fully attributed,
    # so a pinned prototype is checkable rather than quietly overruled.
    assert call["library_source"] == "libc_x86_64.so.6"


def test_an_agreeing_library_leaves_a_user_prototype_alone_759(monkeypatch):
    """The control for the reversal: dropping the suppression must not demote a
    user prototype the library AGREES with -- the demotion follows a real
    contradiction, not the mere presence of a library entry."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=3,
                   arg_texts=["a", "b", "c"], user_type=True)
    _with_type_library(bv, symbol="hw_get_version", params=3)
    _as_import(bv)

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]

    assert call["argument_confidence"] == "authoritative"
    assert "prototype_unverified" not in call


@pytest.mark.parametrize(
    "mangled",
    ["_ZNSt6vectorIiSaIiEE9push_backERKi",
     "_RNvCs1234_4core3fmt5write",
     "_D3std5stdio6writeln",
     "_TtC4main6Widget"],
    ids=["itanium", "rust-v0", "dlang", "swift-old"],
)
def test_library_cross_check_refuses_every_decorated_scheme_759(monkeypatch, mangled):
    """#862 review round 2 MAJOR: the reserved-identifier arm was `_` plus an
    uppercase letter, which is exactly where the mangling prefixes live -- every
    Rust-v0 `_R...` symbol satisfied it, so a LOCAL Rust-mangled function
    colliding with a library entry was still demoted. That is the round-1
    collision class, narrowed to the one scheme the `_Z` refusal did not name.

    A decorated name's implicit parameters (`this`, an sret return slot) make the
    count comparison meaningless whatever its provenance, so every such scheme is
    refused -- here even WITH import provenance, which is the stronger claim.

    RESTORED after round 5 found it deleted: a bad end-boundary in an edit that
    replaced the neighbouring test swallowed these four cases, leaving the
    mangling-prefix refusal -- the fix this very finding produced -- with no
    coverage at all. A repair must not remove its own coverage.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=2, arg_texts=["a", "b"])
    next(f for f in bv.functions if f.name == "hw_get_version").name = mangled
    _with_type_library(bv, symbol=mangled, params=3)
    _as_import(bv, name=mangled)

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]

    assert "prototype_unverified" not in call
    assert call["argument_confidence"] == "authoritative"


def test_reserved_identifier_arm_requires_a_double_underscore_759(monkeypatch):
    """The reserved arm's own test, which round 1 lacked. `__`-prefixed is the
    implementation namespace a statically linked library function lives in; a
    single underscore plus an uppercase letter is NOT admitted, because that is
    the mangling-prefix space."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    def confidence(local_name):
        bv = _arity_bv(monkeypatch, instance, callee_params=0, arg_texts=[])
        next(f for f in bv.functions if f.name == "hw_get_version").name = local_name
        _with_type_library(bv, symbol=local_name, params=1)
        return instance._function_evidence(
            "active", "probe_device", context=0)["calls"][0]["argument_confidence"]

    # A local `__`-reserved name IS described by the library (the `__popcountdi2`
    # shape, statically linked from libgcc and not an import).
    assert confidence("__popcountdi2") == "inferred"
    # A local `_X` name is not, so no demotion.
    assert confidence("_Exit") == "authoritative"


def test_library_cross_check_refuses_when_two_libraries_disagree_759(monkeypatch):
    """#862 review round 2 minor (a): first-library-wins made both the verdict and
    the reported `library_source` depend on `bv.type_libraries` order, and a
    later library that AGREED with the recovery was never consulted. Libraries
    that contradict each other cannot settle anything."""
    import types as _t

    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=3,
                   arg_texts=["a", "b", "c"])
    _as_import(bv)

    def lib(name, params):
        proto = _t.SimpleNamespace(
            parameters=[_t.SimpleNamespace(name=f"p{i}") for i in range(params)],
            has_variable_arguments=False)
        return _t.SimpleNamespace(
            name=name,
            get_named_object=lambda n, _p=proto: _p if str(n) == "hw_get_version" else None)

    # One says 1, the other agrees with the recovered 3: no usable claim either
    # way, and in particular no demotion decided by ordering.
    bv.type_libraries = [lib("libfirst.so", 1), lib("libsecond.so", 3)]
    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert "prototype_unverified" not in call
    assert call["argument_confidence"] == "authoritative"

    # Reversed order, same answer -- the point of the fix.
    bv.type_libraries = [lib("libsecond.so", 3), lib("libfirst.so", 1)]
    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]
    assert "prototype_unverified" not in call


@pytest.mark.parametrize(
    "decorated",
    ["?hw_get_version@@YAHXZ",
     "$s4main13hwGetVersionSiyF",
     "hw_get_version.part.0",
     "hw_get_version@GLIBC_2.14"],
    ids=["msvc", "swift-current", "gcc-clone-suffix", "versioned-symbol"],
)
def test_library_cross_check_refuses_punctuation_decorated_names_759(monkeypatch, decorated):
    """#862 review round 6 MAJOR: the IDENTIFIER half of the refusal had no test
    anywhere. Deleting `or not _C_IDENTIFIER_RE.match(name)` left every changed
    test file green while IMPORTED punctuation-decorated callees flipped from
    `authoritative` to `inferred` + `prototype_unverified` -- the measured
    false-demotion class the agent-facing paragraph claims that rule excludes.

    These are the schemes the prefix tuple does NOT catch, because they are not
    valid C identifiers at all: MSVC `?name@@...`, current Swift `$s...`, a GCC
    clone suffix, and a versioned symbol. Import provenance is deliberately
    granted here, so the row can only stay authoritative because the NAME was
    refused -- which is what makes this fail when the identifier clause goes."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=2, arg_texts=["a", "b"])
    next(f for f in bv.functions if f.name == "hw_get_version").name = decorated
    _with_type_library(bv, symbol=decorated, params=3)
    _as_import(bv, name=decorated)

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]

    assert "prototype_unverified" not in call
    assert call["argument_confidence"] == "authoritative"


def test_a_plain_c_identifier_is_still_admitted_759(monkeypatch):
    """The control that keeps the identifier rule from becoming a blanket refusal:
    an ordinary C name with import provenance and a contradicting library MUST
    still demote, or the rule above would be "refuse everything" and the two
    tests would pass together while the feature did nothing."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _arity_bv(monkeypatch, instance, callee_params=2, arg_texts=["a", "b"])
    _with_type_library(bv, symbol="hw_get_version", params=3)
    _as_import(bv)

    call = instance._function_evidence("active", "probe_device", context=0)["calls"][0]

    assert call["prototype_unverified"] is True
    assert call["argument_confidence"] == "inferred"
