from __future__ import annotations

import importlib
import importlib.util
import io
import itertools
import json
import socket
import sys
import threading
import time
import types
import weakref
from collections.abc import Mapping
from enum import IntEnum
from pathlib import Path

import pytest

from _bridge_fakes import *  # noqa: F401,F403

read_decompile = importlib.import_module("bn_agent_bridge.read_decompile")
read_evidence = importlib.import_module("bn_agent_bridge.read_evidence")


def test_thunk_veneer_warning_names_target_and_is_quiet_for_non_thunks(monkeypatch):
    # #446: a PLT/GOT veneer decompiles as apparent self-recursion; the warning
    # names the real trampoline target (or a generic note) instead.
    # #704 round 4: the mocked summary uses the shape `_function_thunk_summary`
    # actually returns -- name/address live under `target["function"]`, not
    # flat on `target`. The flat shape used here previously made the
    # named-target branch dead against real data (see the round-4 tests below
    # that drive this through the real, un-mocked producer).
    monkeypatch.setattr(read_evidence, "_function_thunk_summary",
                        lambda ctx, bv, func: {"is_candidate": True,
                                               "target": {"function": {"name": "memcpy", "address": "0x1000",
                                                                        "exact_start": True}}})
    w = read_decompile._thunk_veneer_warning(None, None, None)["detail"]
    assert "thunk/veneer -> memcpy @ 0x1000" in w and "self-recursive" in w

    monkeypatch.setattr(read_evidence, "_function_thunk_summary",
                        lambda ctx, bv, func: {"is_candidate": True, "target": None})
    w2 = read_decompile._thunk_veneer_warning(None, None, None)["detail"]
    assert "thunk/veneer" in w2 and "trampoline" in w2

    monkeypatch.setattr(read_evidence, "_function_thunk_summary",
                        lambda ctx, bv, func: {"is_candidate": False})
    assert read_decompile._thunk_veneer_warning(None, None, None) is None


def test_thunk_veneer_warning_mid_function_target_not_confirmed_704(monkeypatch):
    """#704 round 4: a resolved target whose `exact_start` is False lands
    mid-function, not at a trampoline entry point -- the warning must say the
    thunk/veneer verdict is not confirmed, while still naming the resolved
    containing function (closes round-3 follow-up 3(b))."""
    monkeypatch.setattr(
        read_evidence, "_function_thunk_summary",
        lambda ctx, bv, func: {
            "is_candidate": True,
            "reason": "small function with llil_jump to another address",
            "target": {"function": {"name": "big_fn", "address": "0x405000",
                                     "exact_start": False, "offset": "0x10"}},
        })
    w = read_decompile._thunk_veneer_warning(None, None, None)["detail"]
    assert "big_fn @ 0x405000" in w
    assert "not confirmed" in w
    assert "not a self-recursive function" not in w  # the confident wording only


def test_thunk_veneer_warning_names_resolved_exact_start_target_through_real_producer_704(monkeypatch):
    """#704 round 4 regression: the named-target branch tested
    `target.get("name")`/`target.get("address")` directly, but the real
    producer's target is `ctx._normalize_code_pointer(...)`, whose name and
    address live under `target["function"]`. That mismatch made the branch
    dead against real data -- EVERY resolved candidate, including an import
    veneer whose target the tool DID resolve, fell through into the
    "not resolved" fallback. Drive the warning through the real, un-mocked
    producer (`_function_thunk_summary`) to prove the branch fires."""
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

    w = read_decompile._thunk_veneer_warning(instance.ctx, bv, veneer)

    assert w is not None
    assert "log_write @ 0x461746" in w["detail"]
    assert "PLT/GOT trampoline" in w["detail"]
    assert "not resolved" not in w["detail"]
    assert "not confirmed" not in w["detail"]


def test_thunk_veneer_warning_mid_function_target_through_real_producer_704(monkeypatch):
    """#704 round 4: same regression, exercised for the `exact_start is
    False` case through the real producer -- a branch landing inside a
    function's body (not at its entry) must not be confirmed as a PLT/GOT
    thunk, while still naming what the tool resolved."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    big_fn = _FakeFunction(0x405000, "big_fn")
    big_fn.basic_blocks = [_FakeBasicBlock(0x405000, 0x405020)]
    stub = _FakeFunction(0x500000, "stub")
    stub.basic_blocks = [_FakeBasicBlock(0x500000, 0x500004)]
    stub.low_level_il = [[_FakeLLILInstruction(0x500000, _FakeConstPtr(0x405010), operation="LLIL_JUMP")]]
    bv = _FakeBV(
        functions=[big_fn, stub],
        disassembly={0x500000: "b 0x405010"},
    )

    w = read_decompile._thunk_veneer_warning(instance.ctx, bv, stub)

    assert w is not None
    assert "big_fn" in w["detail"]
    assert "not confirmed" in w["detail"]
    assert "not a self-recursive function" not in w["detail"]


def test_thunk_veneer_warning_target_less_wording_states_only_what_is_known_704(monkeypatch):
    """#704 round 3: when `_function_thunk_summary` flagged the candidate but
    resolved no target (the PLT-section case, or the pseudo-C `/* tailcall */`
    fallback), the warning must not assert "PLT/GOT trampoline ... the
    apparent self-call is the resolved target" -- no target was ever
    resolved, so nothing was established about "the resolved target". It must
    instead name the actual reason the candidate was flagged and say plainly
    that the branch target was not resolved."""
    monkeypatch.setattr(
        read_evidence, "_function_thunk_summary",
        lambda ctx, bv, func: {
            "is_candidate": True,
            "reason": "small function rendered as a pseudo-C tailcall",
            "target": None,
        })
    w = read_decompile._thunk_veneer_warning(None, None, None)["detail"]
    assert "small function rendered as a pseudo-C tailcall" in w
    assert "not resolved" in w
    assert "the apparent self-call is the resolved target" not in w

    monkeypatch.setattr(
        read_evidence, "_function_thunk_summary",
        lambda ctx, bv, func: {
            "is_candidate": True,
            "reason": "function starts in a PLT/import trampoline section",
            "target": None,
        })
    w2 = read_decompile._thunk_veneer_warning(None, None, None)["detail"]
    assert "function starts in a PLT/import trampoline section" in w2
    assert "the apparent self-call is the resolved target" not in w2


def test_list_locals_returns_stable_ids(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "player_update", "int32_t player_update(int32_t arg1)")
    fn.parameter_vars = [
        _FakeVariable(name="arg1", storage=4, var_type="int32_t", identifier=1001, index=0)
    ]
    fn.stack_layout = [
        _FakeVariable(name="var_4", storage=-4, var_type="float", identifier=2001, index=1)
    ]
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._list_locals_for_function("active", "player_update")

    assert result["function"]["name"] == "player_update"
    assert len(result["locals"]) == 2
    assert result["locals"][0]["local_id"].startswith("0x401000:param:")
    assert result["locals"][1]["local_id"].startswith("0x401000:local:")


def test_function_disasm_is_address_ordered_and_0x_prefixed(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    function = _FakeFunction(0x1000, "target")
    function.basic_blocks = [
        _FakeBasicBlock(0x2000, 0x2002),
        _FakeBasicBlock(0x1000, 0x1002),
        _FakeBasicBlock(0x3000, 0x3001),
    ]
    bv = _FakeBV(
        memory={
            0x1000: b"\x90\x90",
            0x2000: b"\x90\x90",
            0x3000: b"\xff",
        },
        disassembly={0x1000: "first", 0x2000: "second", 0x3000: ""},
        instruction_lengths={0x1000: 2, 0x2000: 2, 0x3000: 1},
    )

    text = bridge.il_format._disasm_text(bv, function)
    addresses = [line.split()[0] for line in text.splitlines()]

    assert addresses == ["0x1000", "0x2000", "0x3000"]
    assert ".byte 0xff" in text


def test_function_disasm_reports_unreadable_instead_of_bare_byte_directive(monkeypatch):
    # #616 follow-up: a block address with no decode text AND no bytes (a gap
    # inside an otherwise-mapped block, or a stale/partial mapping) must not
    # render `.byte ` with nothing after it -- syntactically broken and reads
    # as "decoded to an empty instruction" rather than "could not read this".
    bridge = _load_bridge(monkeypatch)
    function = _FakeFunction(0x1000, "target")
    function.basic_blocks = [_FakeBasicBlock(0x1000, 0x1001)]
    bv = _FakeBV(memory={}, disassembly={0x1000: ""})

    text = bridge.il_format._disasm_text(bv, function)

    assert ".byte " not in text
    assert "<unreadable" in text
    assert text.startswith("0x1000")


def _thumb_it_disasm_bv_741():
    # Physical text lengths differ from the IT instruction-info span. The
    # 32-bit move also rules out a fixed two-byte stepping workaround.
    rows = [
        (0x1000, bytes.fromhex("14bf"), "ite ne"),
        (0x1002, bytes.fromhex("4ff00203"), "mov.w r3, #2"),
        (0x1006, bytes.fromhex("0023"), "movs r3, #0"),
        (0x1008, bytes.fromhex("7047"), "bx lr"),
    ]
    arch = _FakeArch(
        {0x1000: 8, 0x1002: 4, 0x1006: 2, 0x1008: 2}, name="thumb2"
    )

    def get_instruction_text(data, address):
        for addr, raw, text in rows:
            if address == addr:
                return [text], len(raw)
        return [], 0

    arch.get_instruction_text = get_instruction_text
    function = _FakeFunction(0x1000, "thumb_guard", arch=arch)
    function.basic_blocks = [
        _FakeBasicBlock(0x1008, 0x100A),
        _FakeBasicBlock(0x1000, 0x1008),
    ]
    bv = _FakeBV(
        functions=[function],
        arch=arch,
        memory={0x1000: b"".join(raw for _, raw, _ in rows) + b"\xff"},
    )
    return bv, function, rows


def test_function_disasm_thumb_it_physical_rows_and_slice_741(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, function, rows = _thumb_it_disasm_bv_741()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._disasm(None, function.name)
    expected = [
        [hex(addr), *raw.hex(" ").split(), *text.split()]
        for addr, raw, text in rows
    ]
    assert [line.split() for line in result["text"].splitlines()] == expected
    assert result["total_lines"] == result["returned_lines"] == 4

    sliced = instance._disasm(
        None, function.name, line_start=2, line_end=3, strict_range=True
    )
    assert [line.split() for line in sliced["text"].splitlines()] == expected[1:3]
    assert sliced["total_lines"] == 4
    assert sliced["returned_lines"] == 2


def test_structured_disasm_thumb_it_keeps_predicated_rows_741(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    bv, function, rows = _thumb_it_disasm_bv_741()

    entries = bridge.il_format._structured_disasm_entries(bv, function)
    assert entries == [
        {"address": hex(addr), "text": text, "_address_int": addr}
        for addr, _, text in rows
    ]
    # Evidence callers receive the same two-field entry, not internal length
    # metadata or a changed tuple return type.
    assert bridge.il_format._disasm_entry(bv, 0x1000, arch=function.arch) == {
        "address": "0x1000", "text": "ite ne",
    }


@pytest.mark.parametrize("mode", [None, "thumb"])
def test_disasm_linear_thumb_it_count_stops_inside_block_741(monkeypatch, mode):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, function, _ = _thumb_it_disasm_bv_741()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(
        bridge.read_decompile.bn, "Architecture", {"thumb2": function.arch}, raising=False
    )

    result = instance._disasm(None, "0x1000", linear=2, mode=mode)
    assert result["instruction_count"] == result["requested_count"] == 2
    assert result["instructions"] == [
        {"address": "0x1000", "bytes": "14 bf", "length": 2, "text": "ite ne"},
        {"address": "0x1002", "bytes": "4f f0 02 03", "length": 4, "text": "mov.w r3, #2"},
    ]


def test_disasm_linear_thumb_it_byte_fallback_count_741(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _, rows = _thumb_it_disasm_bv_741()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._disasm(None, "0x1000", linear=5)
    assert result["instruction_count"] == 5
    assert result["instructions"] == [
        {"address": hex(addr), "bytes": raw.hex(" "), "length": len(raw), "text": text}
        for addr, raw, text in rows
    ] + [{"address": "0x100a", "bytes": "ff", "length": 1, "text": ".byte 0xff"}]
    # A byte directive consumes one count unit; it is not a decoded instruction.
    assert ".byte" in result["note"] and "unit" in result["note"]


def test_disasm_linear_thumb_it_boundaries_and_snap_741(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _, _ = _thumb_it_disasm_bv_741()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    at_boundary = instance._disasm(None, "0x1002", linear=1, snap_to_instruction=True)
    assert at_boundary["boundary_warning"] is None
    assert at_boundary["snapped_from"] is None
    assert at_boundary["instructions"][0]["text"] == "mov.w r3, #2"

    interior = instance._disasm(None, "0x1004", linear=1)
    assert interior["boundary_warning"]["nearest_start_at_or_below"] == "0x1002"
    assert interior["boundary_warning"]["nearest_start_above"] == "0x1006"

    snapped = instance._disasm(None, "0x1004", linear=1, snap_to_instruction=True)
    assert snapped["address"] == "0x1002"
    assert snapped["snapped_from"] == "0x1004"
    assert snapped["boundary_warning"] is None
    assert snapped["instructions"][0]["bytes"] == "4f f0 02 03"


@pytest.mark.parametrize("raises", [False, True], ids=["no-decode", "decoder-error"])
def test_disasm_decoder_failure_preserves_mode_fallbacks_741(monkeypatch, raises):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    arch = _FakeArch(name="armv7")

    def failed_decode(data, address):
        if raises:
            raise ValueError("synthetic decoder failure")
        return None

    arch.get_instruction_info = failed_decode
    arch.get_instruction_text = failed_decode
    bv = _FakeBV(
        arch=arch, memory={0x1000: bytes.fromhex("00f020e3")},
        disassembly={0x1000: "nop"}, instruction_lengths={0x1000: 4},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(
        bridge.read_decompile.bn, "Architecture", {"armv7": arch}, raising=False
    )

    lenient = instance._disasm(None, "0x1000", linear=1)
    assert lenient["instructions"] == [
        {"address": "0x1000", "bytes": "00 f0 20 e3", "length": 4, "text": "nop"},
    ]
    forced = instance._disasm(None, "0x1000", linear=1, mode="arm")
    assert forced["instructions"] == [
        {"address": "0x1000", "bytes": "00", "length": 1, "text": ".byte 0x00"},
    ]


def test_function_disasm_rejects_out_of_range_line_window(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    function = _FakeFunction(0x1000, "target")
    function.basic_blocks = [_FakeBasicBlock(0x1000, 0x1002)]
    bv = _FakeBV(
        functions=[function],
        memory={0x1000: b"\x90\x90"},
        disassembly={0x1000: "nop"},
        instruction_lengths={0x1000: 2},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(Exception, match="1-indexed line numbers, not addresses"):
        instance._disasm(
            "active",
            "target",
            line_start=0x1000,
            line_end=0x1001,
            strict_range=True,
        )


def test_list_locals_skips_stack_aliases_for_parameters(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "player_update")
    parameter = _FakeVariable(name="arg1", storage=4, var_type="int32_t", identifier=1001)
    alias = _FakeVariable(name="arg1", storage=4, var_type="int32_t", identifier=1001)
    local = _FakeVariable(name="var_4", storage=-4, var_type="float", identifier=2001)
    fn.parameter_vars = [parameter]
    fn.stack_layout = [alias, local]

    locals_list = instance._list_locals(fn)

    assert len(locals_list) == 2
    assert [item["local_id"] for item in locals_list] == [
        "0x401000:param:stack:4:0:1001",
        "0x401000:local:stack:-4:0:2001",
    ]


def test_list_locals_surfaces_hlil_register_and_flag_vars(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401230, "keychecker_step", "int32_t keychecker_step(void* arg1, char arg2)")
    arg1 = _FakeVariable(name="arg1", storage=105, var_type="void*", identifier=5001,
                         source_type="RegisterVariableSourceType")
    arg2 = _FakeVariable(name="arg2", storage=104, var_type="char", identifier=5002,
                         source_type="RegisterVariableSourceType")
    ret = _FakeVariable(name="__return_addr", storage=0, var_type="void*", identifier=6001,
                        source_type="StackVariableSourceType")
    fn.parameter_vars = [arg1, arg2]
    fn.stack_layout = [ret]
    # Register/flag locals only visible through HLIL; arg1/arg2 reappear here
    # (same Variable identity) and must dedupe against the parameter entries.
    rsi_1 = _FakeVariable(name="rsi_1", storage=104, var_type="char", identifier=5011, index=11,
                          source_type="RegisterVariableSourceType")
    rdx_3 = _FakeVariable(name="rdx_3", storage=100, var_type="int32_t", identifier=5032, index=32,
                          source_type="RegisterVariableSourceType")
    cond = _FakeVariable(name="cond:0", storage=2147483648, var_type="bool", identifier=7000, index=15,
                         source_type="FlagVariableSourceType")
    fn.hlil = types.SimpleNamespace(vars=[arg1, arg2, rsi_1, rdx_3, cond])

    locals_list = instance._list_locals(fn)
    by_name = {item["name"]: item for item in locals_list}

    # params + stack + the 3 HLIL-only vars, with no duplicate arg1/arg2
    assert [item["name"] for item in locals_list].count("arg1") == 1
    assert [item["name"] for item in locals_list].count("arg2") == 1
    assert {"rsi_1", "rdx_3", "cond:0"} <= set(by_name)
    assert by_name["rsi_1"]["local_id"] == "0x401230:local:reg:104:11:5011"
    assert by_name["cond:0"]["local_id"] == "0x401230:local:flag:2147483648:15:7000"

    # The point of the fix: a register var is now resolvable for rename/retype,
    # by both its local_id and its name.
    found, is_param = instance._find_variable_selector(fn, by_name["rsi_1"]["local_id"])
    assert found is rsi_1 and is_param is False
    found_by_name, _ = instance._find_variable_selector(fn, "rdx_3")
    assert found_by_name is rdx_3


def test_list_locals_without_hlil_falls_back_to_param_and_stack(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # _FakeFunction has no `.hlil` attribute -> graceful fallback, no crash.
    fn = _FakeFunction(0x401000, "f", "int32_t f(int32_t arg1)")
    fn.parameter_vars = [_FakeVariable(name="arg1", storage=4, var_type="int32_t", identifier=1001)]
    fn.stack_layout = [_FakeVariable(name="var_4", storage=-4, var_type="int32_t", identifier=2001)]

    locals_list = instance._list_locals(fn)

    assert [item["name"] for item in locals_list] == ["arg1", "var_4"]


def _mlil(operation: str, **fields):
    # BN 6.1's `vars_written`: a plain assignment names its `dest`, a call or
    # split passes its outputs explicitly, and a field write (`rax.al = ...`) or
    # typed syscall reports NOTHING -- the gap `_vars_written` covers.
    fields.setdefault("vars_written", [fields["dest"]] if operation == "MLIL_SET_VAR" else [])
    fields.setdefault("size", 8)
    return types.SimpleNamespace(operation=types.SimpleNamespace(name=operation), **fields)


def _aliased_frame_function():
    """An AArch64-shaped frame: four SSA versions of one word, three of the
    next, a saved-register pair, a 48-byte buffer recovered as six 8-byte slots,
    the canary, and the caller's argument area. The prologue copies the guard
    through a GOT-style symbol load into x8, then into var_28."""
    fn = _FakeFunction(0x40c3a0, "format_report")
    fn.arch = types.SimpleNamespace(name="aarch64", address_size=8)
    ident = iter(range(3000, 4000))

    def stack(name, storage, var_type="int64_t", index=0):
        return _FakeVariable(name=name, storage=storage, var_type=var_type, identifier=next(ident), index=index)

    fn.stack_layout = [
        *(stack(f"var_a0{suffix}", -160, "uint16_t", i) for i, suffix in enumerate(("", "_1", "_2", "_3"))),
        *(stack(f"var_98{suffix}", -152, "uint16_t", i) for i, suffix in enumerate(("", "_1", "_2"))),
        stack("__saved_fp", -112),
        stack("__saved_lr", -104),
        *(stack(f"var_{-offset:x}", offset) for offset in (-88, -80, -72, -64, -56, -48)),
        stack("var_28", -40),
        stack("arg_0", 0, "void"),
    ]
    x8 = _FakeVariable(name="x8", storage=8, var_type="int64_t", identifier=9008,
                       source_type="RegisterVariableSourceType")
    canary = next(v for v in fn.stack_layout if v.name == "var_28")
    fn.mlil = types.SimpleNamespace(basic_blocks=[[
        _mlil("MLIL_SET_VAR", dest=x8, src=_mlil("MLIL_LOAD", src=_mlil("MLIL_CONST_PTR", constant=0x42e110))),
        _mlil("MLIL_SET_VAR", dest=canary, src=_mlil("MLIL_VAR", src=x8)),
    ]])
    guard = types.SimpleNamespace(name="__stack_chk_guard")
    fn.view = types.SimpleNamespace(get_symbol_at=lambda address: guard if address == 0x42e110 else None)
    return fn


def test_frame_collapses_aliases_to_one_row_per_distinct_offset(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _aliased_frame_function()

    frame = bridge.vars_mod._list_frame(fn)
    slots = frame["slots"]

    distinct = sorted({int(v.storage) for v in fn.stack_layout})
    assert [slot["storage"] for slot in slots] == distinct
    assert frame["slot_count"] == len(distinct) < frame["variable_count"] == len(fn.stack_layout)
    by_storage = {slot["storage"]: slot for slot in slots}
    assert by_storage[-160]["names"] == ["var_a0", "var_a0_1", "var_a0_2", "var_a0_3"]
    assert by_storage[-160]["types"] == ["uint16_t"]
    assert by_storage[-160]["distance_to_next_slot"] == 8
    assert by_storage[-152]["distance_to_next_slot"] == 40   # to __saved_fp at -112
    # The buffer BN split into 8-byte slots: pitch says 8, the canary says 48.
    assert by_storage[-88]["distance_to_next_slot"] == 8
    assert by_storage[-88]["to_canary"] == 48
    assert frame["canary_storage"] == -40
    assert {s: slot["kind"] for s, slot in by_storage.items() if slot["kind"] != "local"} == {
        -112: "saved_register", -104: "saved_register", -40: "canary", 0: "caller_frame",
    }
    assert by_storage[-40]["to_canary"] is None and by_storage[0]["distance_to_next_slot"] is None

    # The listing keeps every alias (the frame collapses them, the resolver does
    # not), and every local_id the frame prints still resolves for rename/retype.
    assert len(instance._list_locals(fn)) == len(fn.stack_layout)
    for slot in slots:
        for item in slot["variables"]:
            found, is_parameter = instance._find_variable_selector(fn, item["local_id"])
            assert (str(found.name), int(found.storage), is_parameter) == (item["name"], slot["storage"], False)


def test_frame_detects_the_x86_tls_canary_and_ignores_a_clobbered_guard_register(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    fn = _FakeFunction(0x401189, "copy_name")
    buf = _FakeVariable(name="str", storage=-56, var_type="char[0x28]", identifier=1)
    slot = _FakeVariable(name="var_10", storage=-16, var_type="int64_t", identifier=2)
    spill = _FakeVariable(name="var_18", storage=-24, var_type="int64_t", identifier=3)
    fn.stack_layout = [buf, spill, slot]
    rax = _FakeVariable(name="rax", storage=0, var_type="int64_t", identifier=10,
                        source_type="RegisterVariableSourceType")
    fsbase = _FakeVariable(name="fsbase", storage=1, var_type="int64_t", identifier=11,
                           source_type="RegisterVariableSourceType")
    fn.arch = types.SimpleNamespace(
        name="x86_64", address_size=8, get_reg_name={0: "rax", 1: "fsbase"}.get)
    tls_guard = _mlil("MLIL_LOAD", src=_mlil(
        "MLIL_ADD", left=_mlil("MLIL_VAR", src=fsbase), right=_mlil("MLIL_CONST", constant=0x28)))
    fn.view = types.SimpleNamespace(get_symbol_at=lambda address: None)
    fn.mlil = types.SimpleNamespace(basic_blocks=[[
        _mlil("MLIL_SET_VAR", dest=rax, src=tls_guard),
        _mlil("MLIL_SET_VAR", dest=rax, src=_mlil("MLIL_CONST", constant=0)),
        _mlil("MLIL_SET_VAR", dest=spill, src=_mlil("MLIL_VAR", src=rax)),   # rax no longer holds the guard
        _mlil("MLIL_SET_VAR", dest=rax, src=tls_guard),
        _mlil("MLIL_SET_VAR", dest=slot, src=_mlil("MLIL_VAR", src=rax)),
    ]])

    frame = bridge.vars_mod._list_frame(fn)

    assert frame["canary_storage"] == -16
    assert [(s["storage"], s["kind"], s["to_canary"]) for s in frame["slots"]] == [
        (-56, "local", 40), (-24, "local", 8), (-16, "canary", None),
    ]


def _tls_guard_frame(*clobber):
    """``rax = fs:[0x28]; <clobber>; var_10 = rax`` -- the store is the canary
    only when no instruction in between wrote rax."""
    fn = _FakeFunction(0x401189, "copy_name")
    slot = _FakeVariable(name="var_10", storage=-16, var_type="int64_t", identifier=2)
    fn.stack_layout = [_FakeVariable(name="str", storage=-56, var_type="char[0x28]", identifier=1), slot]
    regs = {name: _FakeVariable(name=name, storage=i, var_type="int64_t", identifier=10 + i,
                                source_type="RegisterVariableSourceType")
            for i, name in enumerate(("rax", "fsbase", "rcx", "rdx"))}
    fn.arch = types.SimpleNamespace(
        name="x86_64", address_size=8,
        get_reg_name={var.storage: name for name, var in regs.items()}.get)
    tls_guard = _mlil("MLIL_LOAD", src=_mlil(
        "MLIL_ADD", left=_mlil("MLIL_VAR", src=regs["fsbase"]), right=_mlil("MLIL_CONST", constant=0x28)))
    fn.view = types.SimpleNamespace(get_symbol_at=lambda address: None)
    fn.mlil = types.SimpleNamespace(basic_blocks=[[
        _mlil("MLIL_SET_VAR", dest=regs["rax"], src=tls_guard),
        *(build(regs) for build in clobber),
        _mlil("MLIL_SET_VAR", dest=slot, src=_mlil("MLIL_VAR", src=regs["rax"])),
    ]])
    return fn


@pytest.mark.parametrize("writer", [
    lambda r: _mlil("MLIL_CALL", output=[r["rax"]], vars_written=[r["rax"]],
                    dest=_mlil("MLIL_CONST_PTR", constant=0x401000), params=[]),
    lambda r: _mlil("MLIL_SET_VAR_SPLIT", high=r["rdx"], low=r["rax"], vars_written=[r["rdx"], r["rax"]],
                    src=_mlil("MLIL_CONST", constant=0)),
    lambda r: _mlil("MLIL_SET_VAR_FIELD", dest=r["rax"], offset=0, src=_mlil("MLIL_CONST", constant=0)),
    lambda r: _mlil("MLIL_SYSCALL", output=[r["rax"]], params=[]),
], ids=["call_output", "set_var_split", "set_var_field", "syscall_output"])
def test_frame_canary_ignores_a_guard_register_any_instruction_overwrote(monkeypatch, writer):
    bridge = _load_bridge(monkeypatch)

    frame = bridge.vars_mod._list_frame(_tls_guard_frame(writer))

    assert frame["canary_storage"] is None
    assert all(s["kind"] == "local" and s["to_canary"] is None for s in frame["slots"])


def test_frame_canary_keeps_a_guard_copy_that_survives_unrelated_writes(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    fn = _tls_guard_frame(
        # rcx = rax copies the guard; a call writing only rdx and a self-copy of
        # rax leave rax holding it.
        lambda r: _mlil("MLIL_SET_VAR", dest=r["rcx"], src=_mlil("MLIL_VAR", src=r["rax"])),
        lambda r: _mlil("MLIL_CALL", output=[r["rdx"]], vars_written=[r["rdx"]],
                        dest=_mlil("MLIL_CONST_PTR", constant=0x401000), params=[]),
        lambda r: _mlil("MLIL_SET_VAR", dest=r["rax"], src=_mlil("MLIL_VAR", src=r["rax"])),
    )

    assert bridge.vars_mod._list_frame(fn)["canary_storage"] == -16


@pytest.mark.parametrize("architecture,register,source,offset,display,expected", [
    ("x86_64", "fsbase", "RegisterVariableSourceType", 0x28, "renamed_tls", -16),
    ("x86", "gsbase", "RegisterVariableSourceType", 0x14, "renamed_tls", -16),
    ("x86_64", "rdi", "RegisterVariableSourceType", 0x28, "fsbase", None),
    ("x86_64", "fsbase", "StackVariableSourceType", 0x28, "fsbase", None),
    ("x86_64", "gsbase", "RegisterVariableSourceType", 0x14, "gsbase", None),
])
def test_frame_tls_canary_uses_architectural_identity(
    monkeypatch, architecture, register, source, offset, display, expected,
):
    bridge = _load_bridge(monkeypatch)
    fn = _tls_guard_frame()
    address = fn.mlil.basic_blocks[0][0].src.src
    base = address.left.src
    base.name = display
    base.source_type = types.SimpleNamespace(name=source)
    address.right.constant = offset
    width = 4 if architecture == "x86" else 8
    for instruction in fn.mlil.basic_blocks[0]:
        instruction.size = instruction.src.size = width
    fn.arch = types.SimpleNamespace(
        name=architecture, address_size=width,
        get_reg_name=lambda storage: register if storage == 1 else "rax")

    assert bridge.vars_mod._list_frame(fn)["canary_storage"] == expected


@pytest.mark.parametrize("load,register_write,stack_write,stack_read", [
    (1, 8, 8, 8),
    (8, 1, 8, 8),
    (8, 8, 1, 8),
    (8, 8, 8, 1),
])
def test_frame_canary_requires_a_full_guard_copy(
    monkeypatch, load, register_write, stack_write, stack_read,
):
    bridge = _load_bridge(monkeypatch)
    fn = _tls_guard_frame()
    loaded, stored = fn.mlil.basic_blocks[0]
    loaded.size, loaded.src.size = register_write, load
    stored.size, stored.src.size = stack_write, stack_read

    frame = bridge.vars_mod._list_frame(fn)
    assert frame["canary_storage"] is None
    assert all(slot["kind"] == "local" and slot["to_canary"] is None
               for slot in frame["slots"])


def test_frame_dereferencing_the_guard_is_not_copying_the_guard(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    fn = _tls_guard_frame()
    loaded = fn.mlil.basic_blocks[0][0]
    loaded.src = _mlil("MLIL_LOAD", src=loaded.src)

    assert bridge.vars_mod._list_frame(fn)["canary_storage"] is None


def test_frame_without_canary_evidence_marks_no_canary(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    fn = _FakeFunction(0x401000, "leaf")   # no `.mlil`: no evidence, no guess
    fn.stack_layout = [
        _FakeVariable(name="var_18", storage=-24, var_type="int64_t", identifier=1),
        _FakeVariable(name="var_8", storage=-8, var_type="int64_t", identifier=2),
    ]

    frame = bridge.vars_mod._list_frame(fn)

    assert frame["canary_storage"] is None
    assert [(s["kind"], s["distance_to_next_slot"], s["to_canary"]) for s in frame["slots"]] == [
        ("local", 16, None), ("local", 8, None),
    ]


def test_frame_negative_slots_stop_at_entry_pointer_without_a_zero_slot(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "frame_boundary")
    fn.stack_layout = [
        _FakeVariable(name="first", storage=-16, var_type="int64_t", identifier=1),
        _FakeVariable(name="alias", storage=-16, var_type="int64_t", identifier=2),
        _FakeVariable(name="last", storage=-8, var_type="int64_t", identifier=3),
        _FakeVariable(name="arg_8", storage=8, var_type="int64_t", identifier=4),
        _FakeVariable(name="arg_18", storage=24, var_type="int64_t", identifier=5),
    ]

    frame = bridge.vars_mod._list_frame(fn)
    assert [(slot["storage"], slot["distance_to_next_slot"]) for slot in frame["slots"]] == [
        (-16, 8), (-8, 8), (8, 16), (24, None),
    ]
    assert [
        (item["storage"], item["distance_to_next_slot"])
        for item in instance._list_locals(fn) if item["storage"] < 0
    ] == [(-16, 8), (-16, 8), (-8, 8)]


def test_find_variable_selector_prefers_local_id(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "player_update")
    shared = _FakeVariable(name="tmp", storage=-4, var_type="int32_t", identifier=2001)
    duplicate = _FakeVariable(name="tmp", storage=-8, var_type="int32_t", identifier=2002)
    fn.stack_layout = [shared, duplicate]

    local_id = instance._local_id(fn, duplicate, is_parameter=False)
    found, is_parameter = instance._find_variable_selector(fn, local_id)

    assert found is duplicate
    assert is_parameter is False


def test_function_info_includes_metadata(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "player_update", "int32_t player_update(int32_t arg1)")
    fn.parameter_vars = [
        _FakeVariable(name="arg1", storage=4, var_type="int32_t", identifier=1001, index=0)
    ]
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_info("active", "player_update")

    assert result["prototype"] == "int32_t player_update(int32_t arg1)"
    assert result["return_type"] == "int32_t"
    assert result["calling_convention"] == "__cdecl"
    assert result["size"] is None


def test_disasm_linear_walks_arbitrary_non_function_address(monkeypatch):
    # #314: linear disasm reads N instructions from a MAPPED address that BN never
    # made part of a function (a missed handler / vtable slot left as data), which
    # the function-scoped path refuses.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        memory={0x1000: b"\x90" * 16},  # mapped, no function here
        disassembly={0x1000: "nop", 0x1002: "nop", 0x1004: "nop"},
        instruction_lengths={0x1000: 2, 0x1002: 2, 0x1004: 2},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    res = instance._disasm(None, "0x1000", linear=3)
    assert res["linear"] is True
    assert res["function"] is None  # not function-bounded
    assert res["instruction_count"] == 3
    assert [e["address"] for e in res["instructions"]] == ["0x1000", "0x1002", "0x1004"]
    assert all(e["text"] == "nop" for e in res["instructions"])
    assert "0x1000" in res["text"] and "nop" in res["text"]
    assert "not function-bounded" in res["note"]


def test_disasm_linear_stops_at_unmapped_tail(monkeypatch):
    # The walk stops when it runs off the end of the mapped region instead of
    # reading garbage / raising.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        memory={0x1000: b"\x90" * 4},  # only two 2-byte instructions fit
        disassembly={0x1000: "nop", 0x1002: "nop"},
        instruction_lengths={0x1000: 2, 0x1002: 2},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    res = instance._disasm(None, "0x1000", linear=10)
    assert res["instruction_count"] == 2  # stopped at the mapped tail


def test_disasm_linear_unmapped_start_errors(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(memory={0x1000: b"\x90" * 4})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    with pytest.raises(Exception) as exc:
        instance._disasm(None, "0x9999", linear=4)
    assert "not mapped" in str(exc.value)


def test_disasm_linear_resolves_function_name(monkeypatch):
    # --linear accepts a function/symbol name too, anchoring at its start.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x1000, "handler")
    bv = _FakeBV(
        functions=[fn],
        memory={0x1000: b"\x90" * 8},
        disassembly={0x1000: "nop", 0x1002: "nop"},
        instruction_lengths={0x1000: 2, 0x1002: 2},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    res = instance._disasm(None, "handler", linear=2)
    assert res["address"] == "0x1000"
    assert res["instruction_count"] == 2


def test_disasm_linear_byte_fallback_on_undecodable(monkeypatch):
    # A mapped address with no decodable instruction (data / invalid opcode) must
    # surface a `.byte 0xNN` and advance one byte, not silently stop -- pointing
    # --linear at data is a primary use.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # No disassembly entries -> _disasm_instruction returns empty text (the fake _FakeArch has
    # no get_instruction_text), so every byte trips the .byte fallback.
    bv = _FakeBV(memory={0x1000: b"\xff\xfe\xfd"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    res = instance._disasm(None, "0x1000", linear=3)
    assert res["instruction_count"] == 3
    assert [e["text"] for e in res["instructions"]] == [".byte 0xff", ".byte 0xfe", ".byte 0xfd"]
    assert [e["length"] for e in res["instructions"]] == [1, 1, 1]


def test_disasm_linear_caps_requested_count(monkeypatch):
    # A request beyond the cap is clamped, but requested_count reports the
    # ORIGINAL request and the result is flagged capped.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(bridge.read_decompile, "_LINEAR_DISASM_MAX", 2)
    bv = _FakeBV(
        memory={0x1000: b"\x90" * 32},
        disassembly={a: "nop" for a in range(0x1000, 0x1020)},
        instruction_lengths={a: 1 for a in range(0x1000, 0x1020)},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    res = instance._disasm(None, "0x1000", linear=10)
    assert res["instruction_count"] == 2          # clamped to the cap
    assert res["requested_count"] == 10           # original request preserved
    assert res["capped"] is True
    assert "capped" in res["note"]


def test_linear_decode_arch_honors_function_and_forces_mode(monkeypatch):
    # #382: BN defaults a whole ARM binary to one mode (e.g. thumb2), so an ARM
    # region decodes wrong. _linear_decode_arch honors the containing function's
    # arch by default and lets --mode force ARM/Thumb for the stripped/missed case.
    bridge = _load_bridge(monkeypatch)
    rd = bridge.read_decompile
    armv7 = type("A", (), {"name": "armv7"})()
    thumb2 = type("A", (), {"name": "thumb2"})()

    class _Ctx:
        def _functions_containing(self, bv, addr):
            return [type("F", (), {"arch": armv7})()] if addr == 0x1000 else []

    bv = type("BV", (), {"arch": thumb2})()
    ctx = _Ctx()
    # default: inside an armv7 function -> armv7 (NOT the thumb2 bv default)
    assert rd._linear_decode_arch(ctx, bv, 0x1000, None).name == "armv7"
    # default: not in a function -> bv default (thumb2)
    assert rd._linear_decode_arch(ctx, bv, 0x9999, None).name == "thumb2"
    # --mode on a non-ARM target is rejected
    bv_x86 = type("BV", (), {"arch": type("A", (), {"name": "x86_64"})()})()
    with pytest.raises(ValueError):
        rd._linear_decode_arch(_Ctx(), bv_x86, 0x1, "arm")
    # --mode forces the architecture (endianness from the bv arch), overriding fn arch
    monkeypatch.setattr(rd.bn, "Architecture", {"armv7": armv7, "thumb2": thumb2}, raising=False)
    assert rd._linear_decode_arch(ctx, bv, 0x9999, "arm").name == "armv7"
    assert rd._linear_decode_arch(ctx, bv, 0x1000, "thumb").name == "thumb2"


def test_disasm_mode_requires_linear(monkeypatch):
    # #382: --mode is meaningful only for --linear; without it, a clear error.
    import bn.cli
    rc = bn.cli.main(["disasm", "sub_1", "--mode", "arm", "--target", "active"])
    assert rc == 2


def test_disasm_non_function_address_hints_at_linear(monkeypatch):
    # Without --linear, a bare address not in a function still errors -- but the
    # disasm-specific message now points at --linear (the generic _find_function
    # error other commands share is untouched).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(memory={0x1000: b"\x90" * 4})  # mapped but no function
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    with pytest.raises(Exception) as exc:
        instance._disasm(None, "0x1000")
    assert "--linear" in str(exc.value)


class _ArmModeArch:
    """An ARM/Thumb arch that NEVER decodes (its forced mode can't model these
    bytes), so the strict forced-mode path must surface `.byte` instead of
    silently falling back to the BV-default decode (#382 review Finding 1)."""

    def __init__(self, name: str = "armv7"):
        self.name = name
        self.address_size = 4
        self.max_instr_length = 4

    def __str__(self):
        return self.name

    def get_instruction_info(self, data, address):
        return None  # the forced arch can't decode -> length 0

    def get_instruction_text(self, data, address):
        return ([], 0)  # the forced arch can't decode -> no tokens


def test_disasm_linear_forced_mode_no_bv_fallback(monkeypatch):
    # #382 review Finding 1: under an explicit --mode the decode is FORCED to that
    # arch. When the forced arch can't decode the bytes, the output must show the
    # honest `.byte 0x..` form, NOT a BV-default-arch disassembly string that
    # would contradict the "decoded as armv7 (forced via --mode)" note.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    forced = _ArmModeArch("armv7")
    # The BV default arch DOES "decode" (get_disassembly returns wrong-mode text);
    # strict-mode must not reach for it.
    bv = _FakeBV(
        arch=_FakeArch(name="thumb2"),
        memory={0x1000: b"\xff\xfe\xfd"},
        disassembly={0x1000: "BV_DEFAULT", 0x1001: "BV_DEFAULT", 0x1002: "BV_DEFAULT"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(
        bridge.read_decompile, "_linear_decode_arch", lambda ctx, bv, addr, mode: forced
    )
    res = instance._disasm(None, "0x1000", linear=3, mode="arm")
    assert [e["text"] for e in res["instructions"]] == [".byte 0xff", ".byte 0xfe", ".byte 0xfd"]
    assert "BV_DEFAULT" not in res["text"]
    assert "forced via --mode" in res["note"]


def test_disasm_linear_normalizes_thumb_tagged_pointer(monkeypatch):
    # #382 review Finding 2: an ARM code pointer commonly carries bit 0 as the
    # Thumb tag. On an ARM/Thumb target, an ODD resolved linear address is masked
    # to even before decoding, with a disclosure note -- otherwise the walk starts
    # one byte into the instruction at 0x1000.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        arch=_FakeArch(name="thumb2"),
        memory={0x1000: b"\x90" * 8},
        disassembly={0x1000: "nop", 0x1002: "nop"},
        instruction_lengths={0x1000: 2, 0x1002: 2},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    res = instance._disasm(None, "0x1001", linear=2)
    assert res["address"] == "0x1000"  # masked even
    assert res["instructions"][0]["address"] == "0x1000"
    assert "Thumb" in res["note"] and "0x1001" in res["note"]


def test_disasm_linear_no_thumb_mask_on_non_arm(monkeypatch):
    # The bit-0 normalization is ARM-only: an odd address on a non-ARM target is
    # left untouched (odd code addresses are legitimate there).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        arch=_FakeArch(name="x86_64"),
        memory={0x1000: b"\x90" * 8},
        disassembly={a: "nop" for a in range(0x1000, 0x1008)},
        instruction_lengths={a: 1 for a in range(0x1000, 0x1008)},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    res = instance._disasm(None, "0x1001", linear=1)
    assert res["address"] == "0x1001"  # NOT masked
    assert "Thumb" not in res["note"]


def test_linear_decode_arch_rejects_non_arm_mode(monkeypatch):
    # #382 review Finding 3: a raw JSON caller can send {"mode":"mips"}; the bridge
    # must reject it with a clean ValueError, not a KeyError on _ARM_MODE_ARCHES.
    bridge = _load_bridge(monkeypatch)
    rd = bridge.read_decompile

    class _Ctx:
        def _functions_containing(self, bv, addr):
            return []

    bv = type("BV", (), {"arch": type("A", (), {"name": "armv7"})()})()
    with pytest.raises(ValueError):
        rd._linear_decode_arch(_Ctx(), bv, 0x1000, "mips")


def test_disasm_linear_no_thumb_mask_on_aarch64(monkeypatch):
    # #600 parity check for BN's "aarch64" spelling: an odd linear start on an
    # AArch64 target is NOT a Thumb pointer tag (AArch64 has no Thumb mode) and
    # must be left untouched. NOTE this spelling alone does NOT catch a reversion
    # of the #600 fix -- "aarch64".startswith("arm") is False, so the old raw
    # startswith("arm")/("thumb") gate already left "aarch64" unmasked and this
    # test passes identically with the fix reverted. The genuine regression guard
    # is the sibling test using BN's "arm64" spelling, which DOES start with
    # "arm" and so distinguishes the fixed gate from the broken one.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        arch=_FakeArch(name="aarch64"),
        memory={0x401000: b"\x90" * 8},
        disassembly={a: "nop" for a in range(0x401000, 0x401008)},
        instruction_lengths={a: 4 for a in range(0x401000, 0x401008)},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    res = instance._disasm(None, "0x401001", linear=1)
    assert res["address"] == "0x401001"  # NOT masked
    assert "Thumb" not in res["note"]


def test_disasm_linear_no_thumb_mask_on_arm64(monkeypatch):
    # BN's alternate AArch64 spelling "arm64" -- the GENUINE #600 regression
    # guard: "arm64".startswith("arm") is True, so the pre-fix raw gate wrongly
    # masked bit 0 as a Thumb pointer tag here. Reverting the fix fails THIS test
    # (the "aarch64"-spelled sibling stays green, so it can't guard the fix).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        arch=_FakeArch(name="arm64"),
        memory={0x401000: b"\x90" * 8},
        disassembly={a: "nop" for a in range(0x401000, 0x401008)},
        instruction_lengths={a: 4 for a in range(0x401000, 0x401008)},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    res = instance._disasm(None, "0x401001", linear=1)
    assert res["address"] == "0x401001"  # NOT masked
    assert "Thumb" not in res["note"]


def test_linear_decode_arch_rejects_mode_on_arm64(monkeypatch):
    # #600: --mode arm|thumb must not force armv7/thumb2 decode on an AArch64
    # target spelled "arm64" -- the wrong ISA entirely. Pre-fix this was a live
    # bug: the old gate was a raw `cur.startswith("arm")`, and "arm64" DOES
    # start with "arm", so the old code wrongly let --mode through and forced
    # an armv7/thumb2 decode of an AArch64 target. (The "aarch64" spelling
    # never exercised this bug -- "aarch64" does not start with "arm" -- so a
    # test using that spelling alone passes identically with the fix reverted;
    # this is the arch spelling that actually distinguishes fixed from broken.)
    # The error must name the actual arch.
    bridge = _load_bridge(monkeypatch)
    rd = bridge.read_decompile

    class _Ctx:
        def _functions_containing(self, bv, addr):
            return []

    bv = type("BV", (), {"arch": type("A", (), {"name": "arm64"})()})()
    with pytest.raises(ValueError, match="arm64"):
        rd._linear_decode_arch(_Ctx(), bv, 0x401000, "arm")


def test_linear_decode_arch_rejects_mode_on_aarch64(monkeypatch):
    # Parity check for the other AArch64 spelling BN uses. Kept alongside the
    # arm64 test above since both spellings must be covered, even though this
    # one alone would not catch a reversion of the #600 fix.
    bridge = _load_bridge(monkeypatch)
    rd = bridge.read_decompile

    class _Ctx:
        def _functions_containing(self, bv, addr):
            return []

    bv = type("BV", (), {"arch": type("A", (), {"name": "aarch64"})()})()
    with pytest.raises(ValueError, match="aarch64"):
        rd._linear_decode_arch(_Ctx(), bv, 0x401000, "arm")
    with pytest.raises(ValueError, match="aarch64"):
        rd._linear_decode_arch(_Ctx(), bv, 0x401000, "thumb")


def _struct_assist_layout(name: str = "cfg", *, flags_width: int = 8) -> _FakeType:
    def integer(width: int) -> _FakeType:
        return _FakeType(
            f"uint{width * 8}_t", width=width, type_class="IntegerTypeClass")
    layout = _FakeType(f"struct {name}", width=0x38, members=[
        _FakeMember(0x28, "flags", integer(flags_width)),
        _FakeMember(0x30, "depth", integer(4)),
    ])
    layout.type = types.SimpleNamespace(name="StructStructureType")
    layout.base_structures = []
    return layout


_STRUCT_ASSIST_EXPRS = itertools.count()


def _struct_assist_node(operation: str, **operands) -> _FakeMLILInsn:
    node = _FakeMLILInsn(0x401004, operation=operation, **operands)
    node.expr_index = next(_STRUCT_ASSIST_EXPRS)
    return node


def _struct_assist_read(var: _FakeVariable, version: int = 0) -> _FakeMLILInsn:
    ssa_var = types.SimpleNamespace(var=var, version=version)
    return _struct_assist_node("MLIL_VAR_SSA", src=ssa_var, vars_read=[ssa_var])


def _struct_assist_access(
    var: _FakeVariable, offset: int, width: int, *,
    operation: str = "MLIL_LOAD_SSA", version: int = 0,
) -> _FakeMLILInsn:
    base = _struct_assist_read(var, version)
    address = base
    if "_STRUCT_" not in operation:
        address = _struct_assist_node(
            "MLIL_ADD", left=base,
            right=_struct_assist_node("MLIL_CONST", constant=offset),
            vars_read=base.vars_read)
    operands = {"src" if operation.startswith("MLIL_LOAD") else "dest": address}
    ins = _struct_assist_node(operation, size=width, **operands)
    if "_STRUCT_" in operation:
        ins.offset = offset
    return ins


def _struct_assist_walk(node, callback):
    """Pre-order expression walk, like MLIL ``traverse`` over nested operands."""
    yield callback(node)
    for name in ("dest", "src", "left", "right"):
        child = getattr(node, name, None)
        if isinstance(child, _FakeMLILInsn):
            yield from _struct_assist_walk(child, callback)
    for child in node.params:
        yield from _struct_assist_walk(child, callback)


def _struct_assist_case() -> tuple[_FakeBV, _FakeFunction, _FakeVariable, list[_FakeMLILInsn]]:
    pointer = _FakeType("void *", width=8, type_class="PointerTypeClass")
    pointer.target = _FakeType("void", type_class="VoidTypeClass")
    var = _FakeVariable(name="c", storage=5, var_type=pointer, identifier=1001)
    fn = _FakeFunction(0x401000, "handle", "int handle(void *c)")
    fn.parameter_vars = [var]
    nodes = [_struct_assist_access(var, 0x28, 8), _struct_assist_access(var, 0x30, 4)]
    fn.mlil = types.SimpleNamespace(ssa_form=types.SimpleNamespace(
        traverse=lambda callback: (
            result for node in nodes for result in _struct_assist_walk(node, callback))))
    bv = _FakeBV(functions=[fn], types_={"cfg": _struct_assist_layout()})
    return bv, fn, var, nodes


def test_decompile_struct_assist_witness_and_no_mutation(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, fn, var, _ = _struct_assist_case()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    _install_fake_pseudo_c(monkeypatch, bridge, fn, [[
        (0x401004, "return *(uint64_t*)((char*)c + 0x28) + *(uint32_t*)((char*)c + 0x30);")
    ]])
    before_type, before_pointer = fn.type, var.type

    result = instance._decompile("active", "handle")

    assert len(result["warnings"]) == 1
    hint = result["warnings"][0]
    assert result["warning_rows"][0]["detail"] == hint
    assert result["warning_rows"][0]["code"] == "layout"
    # These are the actionable evidence, not a pin of the explanatory prose.
    for evidence in ("cfg", "c", "+0x28/8", "flags", "+0x30/4", "depth",
                     "bn struct show cfg", "bn proto get 0x401000",
                     "bn proto set 0x401000", "struct cfg *",
                     # Excluded alternatives are disclosed, not claimed ruled out.
                     "nested struct, union, array or base class"):
        assert evidence in hint
    assert "(char*)c + 0x28" in result["text"]
    assert fn.type is before_type and var.type is before_pointer
    assert fn.has_user_type is False and fn.reanalyzed is False


@pytest.mark.parametrize("operation", [
    "MLIL_LOAD_SSA", "MLIL_STORE_SSA", "MLIL_LOAD_STRUCT_SSA", "MLIL_STORE_STRUCT_SSA",
])
def test_struct_assist_supports_direct_loads_and_stores(operation):
    bv, fn, var, nodes = _struct_assist_case()
    nodes[:] = [_struct_assist_access(var, off, width, operation=operation)
                for off, width in ((0x28, 8), (0x30, 4))]
    assert len(read_decompile._struct_layout_warnings(bv, fn)) == 1


def test_struct_assist_ignores_layout_alias_but_not_distinct_matching_struct():
    bv, fn, _, _ = _struct_assist_case()
    bv.types["cfg_alias"] = _FakeType("cfg", type_class="NamedTypeReferenceClass")
    assert len(read_decompile._struct_layout_warnings(bv, fn)) == 1
    bv.types["other_cfg"] = _struct_assist_layout("other_cfg")
    assert read_decompile._struct_layout_warnings(bv, fn) == []


@pytest.mark.parametrize("other_layout", [False, True], ids=["sole-layout", "ambiguous"])
def test_struct_assist_resolves_scalar_member_typedef_before_ambiguity(monkeypatch, other_layout):
    _load_bridge(monkeypatch)
    class TypeClass(IntEnum):
        IntegerTypeClass = 2
        NamedTypeReferenceClass = 11

    def reference(name: str, width: int) -> _FakeType:
        alias = _FakeType(name, width=width, type_class=TypeClass.NamedTypeReferenceClass)
        alias.name = name
        alias.type_id = ""
        return alias

    bv, fn, _, _ = _struct_assist_case()
    if not other_layout:
        bv.types.clear()
    # Model two member typedef hops through the view's registered type table,
    # with numeric enums rather than permissive string-only type classes.
    bv.types["word_t"] = _FakeType(
        "unsigned long long", width=8, type_class=TypeClass.IntegerTypeClass)
    bv.types["flags_t"] = reference("word_t", 8)
    wrapped = _struct_assist_layout("wrapped_cfg")
    wrapped.members[0].type = reference("flags_t", 8)
    bv.types["wrapped_cfg"] = wrapped
    # A whole-struct alias must not create another concrete candidate.
    bv.types["wrapped_alias"] = reference("wrapped_cfg", wrapped.width)

    warnings = [w["detail"] for w in read_decompile._struct_layout_warnings(bv, fn)]
    if other_layout:
        assert warnings == []
    else:
        assert len(warnings) == 1
        for witness in ("struct wrapped_cfg *", "+0x28/8", "flags", "+0x30/4", "depth"):
            assert witness in warnings[0]


@pytest.mark.parametrize("direct_layout", [False, True], ids=["sole-layout", "competitor"])
@pytest.mark.parametrize("case, expected_lookups, sole_hints, competing_hints", [
    ("16-hops", 16, 1, 0), ("17-hops", 16, 0, 0), ("64-hops", 16, 0, 0),
    ("cycle", 2, 0, 0), ("unresolved", 1, 0, 0),
    ("array-terminal", 1, 0, 1), ("nested-terminal", 1, 0, 1),
])
def test_struct_assist_bounds_member_typedef_lookups(
    monkeypatch, direct_layout, case, expected_lookups, sole_hints, competing_hints,
):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, fn, _, _ = _struct_assist_case()
    scalar = bv.types["cfg"].members[0].type

    def reference(index):
        alias = _FakeType("word_t", width=8, type_class="NamedTypeReferenceClass")
        alias.name = "word_t"
        alias.type_id = f"ref-{index}"
        return alias

    lookups = []

    def lookup(type_id):
        lookups.append(type_id)
        if case == "unresolved":
            return None
        if case == "array-terminal":
            return _FakeType("char[8]", width=8, type_class="ArrayTypeClass")
        if case == "nested-terminal":
            return _FakeType("struct inner", width=8, members=[
                _FakeMember(0, "value", scalar)])
        index = int(type_id.rsplit("-", 1)[1])
        if case == "cycle":
            return reference((index + 1) % 2)
        return scalar if index + 1 == int(case.split("-")[0]) else reference(index + 1)

    bv.get_type_by_id = lookup
    if not direct_layout:
        bv.types.clear()
    wrapped = _struct_assist_layout("wrapped_cfg")
    wrapped.members[0].type = reference(0)
    bv.types["wrapped_cfg"] = wrapped
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    _install_fake_pseudo_c(monkeypatch, bridge, fn, [[
        (0x401004, "return *(uint64_t*)((char*)c + 0x28) + *(uint32_t*)((char*)c + 0x30);")
    ]])

    result = instance._decompile("active", "handle")

    expected_hints = competing_hints if direct_layout else sole_hints
    assert len(result["warnings"]) == expected_hints
    if expected_hints:
        candidate = "cfg" if direct_layout else "wrapped_cfg"
        for witness in (f"struct {candidate} *", "+0x28/8", "flags", "+0x30/4", "depth"):
            assert witness in result["warnings"][0]
    assert len(lookups) == expected_lookups


@pytest.mark.parametrize("case", [
    "wrong-offset", "wrong-width", "non-scalar-member",
    "unobserved-opaque", "unrelated-opaque",
])
def test_decompile_struct_assist_opaque_members_do_not_hide_supported_layout(monkeypatch, case):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, fn, _, _ = _struct_assist_case()
    opaque = _FakeType("missing_t", width=8, type_class="NamedTypeReferenceClass")
    opaque.name = "missing_t"
    opaque.type_id = "missing-id"
    bv.get_type_by_id = lambda type_id: None
    wrapped = _struct_assist_layout("wrapped_cfg")
    wrapped.members[0].type = opaque
    if case == "wrong-offset":
        # Even with one compatible-but-opaque access, another access rules out
        # the competitor; uncertainty does not override proven incompatibility.
        wrapped.members[1].offset = 0x34
    elif case == "wrong-width":
        wrapped.members[1].type.width = 2
    elif case == "non-scalar-member":
        wrapped.members[1].type = _FakeType(
            "char[4]", width=4, type_class="ArrayTypeClass")
    elif case == "unobserved-opaque":
        bv.types["cfg"].members.insert(0, _FakeMember(0x20, "opaque", opaque))
    elif case == "unrelated-opaque":
        bv.types["missing_alias"] = opaque
    if case not in {"unobserved-opaque", "unrelated-opaque"}:
        bv.types["wrapped_cfg"] = wrapped
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    _install_fake_pseudo_c(monkeypatch, bridge, fn, [[
        (0x401004, "return *(uint64_t*)((char*)c + 0x28) + *(uint32_t*)((char*)c + 0x30);")
    ]])

    result = instance._decompile("active", "handle")

    assert len(result["warnings"]) == 1
    for witness in ("struct cfg *", "+0x28/8", "flags", "+0x30/4", "depth"):
        assert witness in result["warnings"][0]


@pytest.mark.parametrize("case", [
    "no-layout", "wrong-width", "wrong-offset", "array-member", "bitfield",
    "union", "inherited", "overlapping", "out-of-bounds", "single-offset",
    "conflicting-width", "computed-offset", "reassigned-parameter", "arithmetic-only",
    "no-ssa", "different-parameter",
])
def test_struct_assist_requires_consistent_supported_witness(case):
    bv, fn, var, nodes = _struct_assist_case()
    layout = bv.types["cfg"]
    if case == "no-layout":
        bv.types.clear()
    elif case == "wrong-width":
        bv.types["cfg"] = _struct_assist_layout(flags_width=4)
    elif case == "wrong-offset":
        layout.members[0].offset = 0x20
    elif case == "array-member":
        layout.members[0].type.type_class = "ArrayTypeClass"
    elif case == "bitfield":
        layout.members[0].bit_width = 3
    elif case == "union":
        layout.type = types.SimpleNamespace(name="UnionStructureType")
    elif case == "inherited":
        layout.base_structures = [object()]
    elif case == "overlapping":
        layout.members.append(_FakeMember(0x2c, "overlap", layout.members[1].type))
    elif case == "out-of-bounds":
        layout.width = 0x30
    elif case == "single-offset":
        nodes[1] = _struct_assist_access(var, 0x28, 8)
    elif case == "conflicting-width":
        nodes.append(_struct_assist_access(var, 0x28, 4))
    elif case == "computed-offset":
        extra = _struct_assist_access(var, 0x38, 4)
        extra.src.right = extra.src.left  # *(c + c), not a constant offset
        nodes.append(extra)
    elif case == "reassigned-parameter":
        nodes.append(_struct_assist_access(var, 0x38, 4, version=1))
    elif case == "arithmetic-only":
        nodes[:] = [node.src for node in nodes]
    elif case == "no-ssa":
        fn.mlil.ssa_form = None
    elif case == "different-parameter":
        other = _FakeVariable(name="other", storage=6, var_type=var.type, identifier=1002)
        fn.parameter_vars.append(other)
        nodes[1] = _struct_assist_access(other, 0x30, 4)
    assert read_decompile._struct_layout_warnings(bv, fn) == []


@pytest.mark.parametrize("case, hints", [
    ("copy", 0), ("phi", 0), ("call-argument", 0), ("derived-pointer", 0),
    ("stored-value", 0), ("address-taken", 0), ("null-check", 1),
])
def test_struct_assist_parameter_value_must_not_escape_witness(case, hints):
    bv, fn, var, nodes = _struct_assist_case()
    alias = _FakeVariable(name="rbx", storage=7, var_type=var.type, identifier=1003)
    alias_ssa = types.SimpleNamespace(var=alias, version=1)
    if case == "copy":
        # rbx#1 = c#0, then an access through the copy outside cfg's layout.
        nodes.append(_struct_assist_node(
            "MLIL_SET_VAR_SSA", dest=alias_ssa, src=_struct_assist_read(var)))
        nodes.append(_struct_assist_access(alias, 0x80, 4, version=1))
    elif case == "phi":
        nodes.append(_struct_assist_node(
            "MLIL_VAR_PHI", dest=alias_ssa,
            src=[types.SimpleNamespace(var=var, version=0), alias_ssa]))
    elif case == "call-argument":
        nodes.append(_struct_assist_node(
            "MLIL_CALL_SSA", params=[_struct_assist_read(var)]))
    elif case == "derived-pointer":
        nodes.append(_struct_assist_node("MLIL_SET_VAR_SSA", dest=alias_ssa, src=(
            _struct_assist_node(
                "MLIL_ADD", left=_struct_assist_read(var),
                right=_struct_assist_node("MLIL_CONST", constant=0x10)))))
    elif case == "stored-value":
        nodes.append(_struct_assist_node(
            "MLIL_STORE_SSA", size=8,
            dest=_struct_assist_node("MLIL_CONST_PTR", constant=0x404000),
            src=_struct_assist_read(var)))
    elif case == "address-taken":
        nodes.append(_struct_assist_node("MLIL_ADDRESS_OF", src=var))
    elif case == "null-check":
        # A comparison result cannot carry the pointer to unseen accesses.
        nodes.append(_struct_assist_node(
            "MLIL_CMP_E", left=_struct_assist_read(var),
            right=_struct_assist_node("MLIL_CONST", constant=0)))
    assert len(read_decompile._struct_layout_warnings(bv, fn)) == hints


@pytest.mark.parametrize("case, hints", [
    ("bitfield-unit", 0), ("split-bitfields", 0), ("overlapping-members", 0),
    ("unobserved-bitfield", 1), ("class-competitor", 0), ("sole-class", 0),
    ("flexible-tail-competitor", 0), ("unresolved-tail-competitor", 0),
    ("zero-width-before-access", 0), ("own-flexible-tail", 1),
    ("overrun-competitor", 0), ("sole-overrun", 0),
])
def test_struct_assist_potentially_compatible_competitor_blocks_uniqueness(case, hints):
    bv, fn, _, _ = _struct_assist_case()
    other = _struct_assist_layout("other_cfg")
    unit = other.members[1].type

    def bitfield(offset: int, name: str, bits: int) -> _FakeMember:
        member = _FakeMember(offset, name, unit)
        member.bit_width = bits
        return member

    if case == "bitfield-unit":
        other.members[1] = bitfield(0x30, "lo", 20)
        other.width = 0x33  # BN sizes a trailing bitfield to the bit
    elif case == "split-bitfields":
        # BN reports the second field at its first byte with the unit's width.
        other.members[1:] = [bitfield(0x30, "lo", 16), bitfield(0x32, "hi", 16)]
        other.width = 0x38
    elif case == "overlapping-members":
        other.members.append(_FakeMember(0x2c, "overlap", unit))
    elif case in {"class-competitor", "sole-class"}:
        # A base-less class is as concrete a layout as a struct, but is never
        # itself suggested as `struct K *`.
        other.type = types.SimpleNamespace(name="ClassStructureType")
        if case == "sole-class":
            del bv.types["cfg"]
    elif case in {"flexible-tail-competitor", "own-flexible-tail"}:
        if case == "own-flexible-tail":
            other = bv.types["cfg"]
        other.members.append(_FakeMember(0x38, "tail", _FakeType(
            "char[0]", width=0, type_class="ArrayTypeClass")))
    elif case == "unresolved-tail-competitor":
        other.members.append(_FakeMember(0x38, "ext", _FakeType(
            "missing_t", width=0, type_class="NamedTypeReferenceClass")))
    elif case in {"overrun-competitor", "sole-overrun"}:
        # A redeclared embedded type can outgrow the outer struct's stored
        # width; the unobserved stale tail must not drop the competitor.
        other.members.append(_FakeMember(0x38, "tail", _FakeType(
            "struct ext", width=16, type_class="StructureTypeClass")))
        if case == "sole-overrun":
            del bv.types["cfg"]
    elif case == "zero-width-before-access":
        # An unknown-width member may extend over every later byte.
        other.members.insert(0, _FakeMember(0x20, "ext", _FakeType(
            "missing_t", width=0, type_class="NamedTypeReferenceClass")))
    elif case == "unobserved-bitfield":
        # Unclassifiable storage away from every access does not block cfg.
        other = bv.types["cfg"]
        other.members.append(bitfield(0x34, "spare", 3))
    bv.types[str(other).removeprefix("struct ")] = other
    warnings = [w["detail"] for w in read_decompile._struct_layout_warnings(bv, fn)]
    assert len(warnings) == hints
    if hints:
        assert "struct cfg *" in warnings[0]


def test_struct_assist_resolves_only_observed_member_typedefs():
    bv, fn, _, _ = _struct_assist_case()
    lookups = []
    scalar = _FakeType("uint32_t", width=4, type_class="IntegerTypeClass")

    def lookup(type_id):
        lookups.append(type_id)
        return scalar

    def reference(index: int) -> _FakeType:
        alias = _FakeType("u32_t", width=4, type_class="NamedTypeReferenceClass")
        alias.name = "u32_t"
        alias.type_id = f"ref-{index}"
        return alias

    bv.get_type_by_id = lookup
    for index in range(200):
        decoy = _FakeType(f"struct decoy{index}", width=0x40, members=[
            _FakeMember(offset, f"f{offset:x}", reference(index))
            for offset in (*range(0, 0x28, 4), 0x34, 0x38, 0x3c)
        ])
        decoy.type = types.SimpleNamespace(name="StructStructureType")
        decoy.base_structures = []
        bv.types[f"decoy{index}"] = decoy
    wrapped = _struct_assist_layout("wrapped_cfg")
    wrapped.members[1].type = reference(-1)
    wrapped.members.insert(0, _FakeMember(0x20, "unobserved", reference(-2)))
    bv.types["wrapped_cfg"] = wrapped
    del bv.types["cfg"]

    warnings = [w["detail"] for w in read_decompile._struct_layout_warnings(bv, fn)]

    assert len(warnings) == 1 and "struct wrapped_cfg *" in warnings[0]
    assert lookups == ["ref--1"]


@pytest.mark.parametrize("pointee", ["StructureTypeClass", "IntegerTypeClass"])
def test_struct_assist_leaves_typed_parameters_and_rendering_alone(monkeypatch, pointee):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, fn, var, _ = _struct_assist_case()
    var.type.target.type_class = pointee
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    _install_fake_pseudo_c(
        monkeypatch, bridge, fn, [[(0x401004, "return c->flags + c->depth;")]])
    result = instance._decompile("active", "handle")
    assert result["text"] == "return c->flags + c->depth;"
    assert result["warnings"] == []


def test_struct_assist_preserves_pointee_qualifiers_in_suggestion():
    bv, fn, var, _ = _struct_assist_case()
    var.type.target.const = True
    var.type.target.volatile = True
    assert "const volatile struct cfg *" in read_decompile._struct_layout_warnings(bv, fn)[0]["detail"]


def test_decompile_renders_pseudo_c(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "player_update")
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    _install_fake_pseudo_c(
        monkeypatch,
        bridge,
        fn,
        [
            # BN indents the signature line by 2 spaces; we left-justify it.
            [(0x401000, "  int32_t player_update(int32_t arg1)")],
            [(0x401000, "{")],
            [(0x401004, "    return arg1 + 1;")],
            [(0x401008, "}")],
        ],
    )

    result = instance._decompile("active", "player_update")

    assert result["function"] == {"name": "player_update", "address": "0x401000"}
    assert result["text"] == (
        "int32_t player_update(int32_t arg1)\n{\n    return arg1 + 1;\n}"
    )


def test_decompile_pseudo_c_with_address_gutter(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "player_update")
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    _install_fake_pseudo_c(
        monkeypatch,
        bridge,
        fn,
        [
            [(0x401000, "")],  # leading blank separator -> trimmed, no orphan address
            [(0x401000, "  int32_t player_update(int32_t arg1)")],  # 2-space indent stripped
            [(0x401000, "")],  # internal blank -> empty line, not "00401000"
            [(0x401004, "    return arg1 + 1;")],
            [(0x401004, "")],  # trailing blank separator -> trimmed
        ],
    )

    result = instance._decompile("active", "player_update", addresses=True)

    assert result["text"] == (
        "0x401000        int32_t player_update(int32_t arg1)\n"
        "\n"
        "0x401004            return arg1 + 1;"
    )


def test_decompile_redacts_annotation_bodies_unless_explicitly_included(
    monkeypatch
):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    function = _FakeFunction(0x401000, "parse_record")
    function.basic_blocks = [_FakeBasicBlock(0x401000, 0x401004)]
    function.comment = "inherited function note"
    function.comments = {0x401002: "function-local address note"}
    bv = _FakeBV(
        functions=[function],
        # A grouped instruction span must not skip a rendered comment within it.
        instruction_lengths={0x401000: 4},
        comments={0x401002: "inherited address note", 0x401004: "outside function"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    text = (
        "void parse_record() {  // inherited function note\n"
        "    // inherited address note\n"
        "    return; // function-local address note\n"
        "}"
    )
    monkeypatch.setattr(
        bridge.il_format, "_decompile_text", lambda *args, **kwargs: text
    )

    redacted = instance._decompile("active", "parse_record")
    included = instance._decompile(
        "active", "parse_record", include_annotations=True
    )

    assert redacted["text"] == (
        "void parse_record() {  // <annotation redacted>\n"
        "    // <annotation redacted>\n"
        "    return; // <annotation redacted>\n"
        "}"
    )
    assert redacted["comments"] == {}
    assert redacted["annotation_summary"] == {
        "comment_count": 3,
        "redacted": True,
        "hidden": 3,
        "session_shown": 0,
    }
    assert included["text"] == text
    # The legacy map stays global-only; local bodies are collected independently.
    assert included["comments"] == {"0x401002": "inherited address note"}


def test_bundle_function_redacts_annotations_like_decompile_752(monkeypatch):
    """#752: `bundle function` rendered its text through a second call path
    (`il_format._decompile_text` directly), bypassing the #740 redaction that
    `decompile` applies by default -- so the artifact carried annotation bodies in
    full while the identically-rendered `decompile` output stripped them. The
    bundle is presented as a shareable artifact, so its decompilation must agree
    with what `decompile` shows, and the same flag must open both."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    function = _FakeFunction(0x401000, "parse_record")
    function.basic_blocks = [_FakeBasicBlock(0x401000, 0x401004)]
    function.comment = "inherited function note"
    function.comments = {0x401002: "function-local address note"}
    bv = _FakeBV(
        functions=[function],
        instruction_lengths={0x401000: 4},
        comments={0x401002: "inherited address note"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    text = (
        "void parse_record() {  // inherited function note\n"
        "    // inherited address note\n"
        "    return; // function-local address note\n"
        "}"
    )
    monkeypatch.setattr(
        bridge.il_format, "_decompile_text", lambda *args, **kwargs: text
    )
    # Not under test here: the bundle's other sections.
    monkeypatch.setattr(instance, "_target_info", lambda selector, **k: {})
    monkeypatch.setattr(instance, "_function_text", lambda *a, **k: "")
    monkeypatch.setattr(instance, "_disasm_text", lambda *a, **k: "")
    monkeypatch.setattr(instance, "_list_locals", lambda *a, **k: [])
    monkeypatch.setattr(instance, "_xrefs_to_address", lambda *a, **k: [])

    bundle = instance._bundle_function("active", "parse_record", None)
    expected = instance._decompile("active", "parse_record")["text"]

    assert "inherited function note" not in bundle["decompile"]
    assert "inherited address note" not in bundle["decompile"]
    assert "function-local address note" not in bundle["decompile"]
    assert bundle["decompile"].count("// <annotation redacted>") == 3
    assert bundle["decompile"] == expected

    included = instance._bundle_function(
        "active", "parse_record", None, include_annotations=True
    )
    assert included["decompile"] == text


@pytest.mark.parametrize("store", ["global", "local"])
def test_decompile_does_not_hide_an_unreadable_comment_store(monkeypatch, store):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class UnreadableLocalComments(_FakeFunction):
        @property
        def comments(self):
            raise RuntimeError("local comment store unavailable")

    class UnreadableGlobalComments(_FakeBV):
        @property
        def address_comments(self):
            raise RuntimeError("global comment store unavailable")

    function_type = UnreadableLocalComments if store == "local" else _FakeFunction
    view_type = UnreadableGlobalComments if store == "global" else _FakeBV
    function = function_type(0x401000, "parse_record")
    bv = view_type(functions=[function])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(RuntimeError, match=f"{store} comment store unavailable"):
        instance._decompile("active", "parse_record")


@pytest.mark.parametrize("body,code", [
    ("1", "value = 1;"),
    ("buf", "char* buf = input;"),
    ("end", 'char* label = "weekend";'),
])
def test_decompile_redacts_only_rendered_comment_lines(monkeypatch, body, code):
    # A short comment body like "1", "buf", or "end" is a substring of
    # unrelated code (`value = 1;`, `char* buf`, `"weekend"`); redaction must
    # rewrite only the rendered comment LINE, never bare-substring the body
    # out of surrounding code.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    function = _FakeFunction(0x401000, "parse_record")
    function.basic_blocks = [_FakeBasicBlock(0x401000, 0x401002)]
    bv = _FakeBV(functions=[function], comments={0x401000: body})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    text = f"void parse_record() {{\n    // {body}\n    {code}\n}}"
    monkeypatch.setattr(
        bridge.il_format, "_decompile_text", lambda *args, **kwargs: text
    )

    result = instance._decompile("active", "parse_record")

    assert f"// {body}" not in result["text"]
    assert "// <annotation redacted>" in result["text"]
    assert code in result["text"]


def test_decompile_redacts_every_line_of_multiline_annotation(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    function = _FakeFunction(0x401000, "parse_record")
    function.basic_blocks = [_FakeBasicBlock(0x401000, 0x401002)]
    function.comment = "first line\nsecond line"
    bv = _FakeBV(functions=[function])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    text = (
        "void parse_record() {\n"
        "    // first line\n"
        "    do_thing(); // second line\n"
        "}"
    )
    monkeypatch.setattr(
        bridge.il_format, "_decompile_text", lambda *args, **kwargs: text
    )

    result = instance._decompile("active", "parse_record")
    included = instance._decompile("active", "parse_record", include_annotations=True)

    assert "first line" not in result["text"]
    assert "second line" not in result["text"]
    assert result["text"].count("// <annotation redacted>") == 2
    assert "do_thing();" in result["text"]
    assert included["text"] == text


def test_redact_rendered_annotations_preserves_address_gutter():
    # The `--addresses` gutter form: a 0x-prefixed address, whitespace, then
    # the rendered line. Redaction must preserve the gutter and the `//`
    # prefix, rewriting only the comment body.
    text = (
        "0x401000        // secret note\n"
        "0x401004            value = 1;"
    )
    redacted, _hidden, _shown = read_decompile._redact_rendered_annotations(
        text, ["secret note"]
    )
    assert redacted == (
        "0x401000        // <annotation redacted>\n"
        "0x401004            value = 1;"
    )


def test_redact_rendered_annotations_preserves_literals_and_noncomment_spans():
    lines = [
        '0x401000    const char* url = "https://host/buf";',
        r'0x401004    const char* quoted = "\"// buf"; // buf',
        r"0x401008    char quote = '\''; char slash = '/';",
        "0x40100c    buf = end / 1 + count; /* // buf */",
        "0x401010    /* block comment",
        "0x401014       // buf",
        "0x401018    */ buf++;",
        '0x40101c    puts("buf"); // unrelated note',
        '0x401020    puts("// buf"); /* // buf */ return; // \tbuf  ',
        "0x401024    // buf\t",
    ]
    text = "\r\n".join(lines)
    expected = "\r\n".join([
        lines[0],
        r'0x401004    const char* quoted = "\"// buf"; // <annotation redacted>',
        *lines[2:-2],
        '0x401020    puts("// buf"); /* // buf */ return; // \t<annotation redacted>  ',
        "0x401024    // <annotation redacted>\t",
    ])

    assert read_decompile._redact_rendered_annotations(text, ["buf"])[0] == expected


def test_decompile_falls_back_to_hlil_when_pseudo_c_unavailable(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "player_update", "int32_t player_update()")
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    # No fake lineardisassembly module installed -> _pseudo_c_text raises and we
    # fall back to wrapped HLIL produced by _function_text. The renderer now
    # lives in il_format and _decompile_text calls it module-locally, so stub it
    # there (patching instance._function_text no longer intercepts that call).
    monkeypatch.setattr(bridge.il_format, "_function_text", lambda bv, func, **kw: "    return 1;")

    result = instance._decompile("active", "player_update")

    # The pseudo-C failure is surfaced via an explicit marker line instead of
    # silently presenting the HLIL fallback as a successful decompilation.
    lines = result["text"].splitlines()
    assert lines[0].startswith("// bn: decompilation failed (")
    assert "\n".join(lines[1:]) == "int32_t player_update()\n{\n    return 1;\n}"


def test_decompile_warns_on_skipped_analysis(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "big_fn")
    fn.analysis_skipped = True
    fn.analysis_skip_reason = "ExceedFunctionSize"
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    # Body has no telltale text -> warning must fire on the analysis_skipped flag alone.
    _install_fake_pseudo_c(
        monkeypatch, bridge, fn,
        [[(0x401000, "int32_t big_fn()")], [(0x401000, "{")], [(0x401000, "}")]],
    )

    result = instance._decompile("active", "big_fn")

    assert result["analysis_skipped"] is True
    assert result["analysis_forced"] is False
    assert result["analysis_force_requested"] is False
    assert fn.reanalyzed is False  # warn-only must NOT reanalyze
    assert any("big_fn" in w and "ExceedFunctionSize" in w and "--force-analysis" in w for w in result["warnings"])


def test_decompile_warns_on_placeholder_text_when_flag_clear(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "big_fn")  # analysis_skipped defaults False
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    _install_fake_pseudo_c(
        monkeypatch, bridge, fn,
        [
            [(0x401000, "int32_t big_fn()")],
            [(0x401000, "{")],
            [(0x401000, "    // This function is taking too long to analyze")],
            [(0x401000, "    // Loading...")],
            [(0x401000, "}")],
        ],
    )

    result = instance._decompile("active", "big_fn")

    assert result["analysis_skipped"] is False
    assert any("incomplete stub" in w for w in result["warnings"])


def test_analysis_stub_warning_ignores_legitimate_phrase_in_program_text(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    fn = _FakeFunction(0x401000, "real_fn")
    text = 'int real_fn() { puts("This function is taking too long to analyze"); }'

    assert bridge.il_format._analysis_stub_warning(fn, text) is None


def test_decompile_force_analysis_reanalyzes_and_clears_warning(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "big_fn")
    fn.analysis_skipped = True
    fn.analysis_skip_reason = "ExceedFunctionSize"
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    _install_fake_pseudo_c(
        monkeypatch, bridge, fn,
        [
            [(0x401000, "int32_t big_fn()")],
            [(0x401000, "{")],
            [(0x401004, "    return 1;")],
            [(0x401008, "}")],
        ],
    )

    result = instance._decompile("active", "big_fn", force_analysis=True)

    assert fn.reanalyzed is True                       # reanalysis was triggered
    assert getattr(bv, "analysis_updated", False) is True
    assert fn.analysis_skipped is False               # skip override cleared
    assert result["analysis_forced"] is True
    assert result["analysis_force_requested"] is True
    assert result["analysis_skipped"] is False
    assert result["text"] == "int32_t big_fn()\n{\n    return 1;\n}"
    assert not any("stub" in w.lower() or "skipped analysis" in w for w in result["warnings"])


def test_decompile_force_analysis_warns_on_likely_data_region(monkeypatch):
    # #371.1: forcing analysis on an oversized, 0-inbound-ref capped "function"
    # is exactly the data-as-code trap -- BN tentatively made a function on a
    # string/pointer table or packed data; forced decode grows it unboundedly.
    # Emit a hedged verify-nudge (not a false "this is data" verdict).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x80586324, "sub_80586324", total_bytes=65540)
    fn.analysis_skipped = True
    fn.analysis_skip_reason = "ExceedFunctionSize"
    bv = _FakeBV(functions=[fn])  # no code_refs registered -> 0 inbound refs
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    _install_fake_pseudo_c(
        monkeypatch, bridge, fn,
        [[(0x80586324, "int32_t sub_80586324()")], [(0x80586324, "{")], [(0x80586324, "}")]],
    )

    result = instance._decompile("active", "sub_80586324", force_analysis=True)

    assert result["analysis_forced"] is True
    assert any("data region" in w.lower() for w in result["warnings"])
    assert any("0 inbound code refs" in w for w in result["warnings"])


def test_decompile_force_analysis_no_data_warning_when_referenced(monkeypatch):
    # A forced oversized function WITH inbound callers is real code, not a
    # tentatively-typed data region -- no data-vs-code nudge.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "big_real_fn", total_bytes=65540)
    fn.analysis_skipped = True
    fn.analysis_skip_reason = "ExceedFunctionSize"
    bv = _FakeBV(functions=[fn], code_refs={0x401000: [object()]})  # a real caller
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    _install_fake_pseudo_c(
        monkeypatch, bridge, fn,
        [[(0x401000, "int32_t big_real_fn()")], [(0x401000, "{")], [(0x401000, "}")]],
    )

    result = instance._decompile("active", "big_real_fn", force_analysis=True)

    assert result["analysis_forced"] is True
    assert not any("data region" in w.lower() for w in result["warnings"])


def test_decompile_force_analysis_no_data_warning_for_init_array_initializer(monkeypatch):
    # #371.1 false-positive guard (found in dogfood): a large `.init_array`/ctor
    # initializer is real code with 0 direct CALLERS but a DATA ref from the init
    # table pointing at its start. Requiring 0 data refs too keeps it quiet while
    # still firing on a truly unreferenced data region (0 code AND 0 data refs).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x402000, "init_array_ctor", total_bytes=46080)
    fn.analysis_skipped = True
    fn.analysis_skip_reason = "ExceedFunctionSize"
    # 0 code refs, but the init_array entry is a DATA ref to the function start.
    bv = _FakeBV(functions=[fn], data_refs={0x402000: [object()]})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    _install_fake_pseudo_c(
        monkeypatch, bridge, fn,
        [[(0x402000, "void init_array_ctor()")], [(0x402000, "{")], [(0x402000, "}")]],
    )

    result = instance._decompile("active", "init_array_ctor", force_analysis=True)

    assert result["analysis_forced"] is True
    assert not any("data region" in w.lower() for w in result["warnings"])


def test_function_list_defers_display_projection_to_the_returned_page(monkeypatch):
    # Perf: display_name (a per-function symbol lookup) and size must be computed
    # ONLY for the returned page, not the whole filtered set -- the same rule #411
    # applied to basic_block_count. Otherwise a `function list --limit 1` over a
    # 24k-function binary pays a full-set projection to return one row. We assert
    # the projection helpers fire exactly `returned` times, and the page still
    # carries all three display fields.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    funcs = [_FakeFunction(0x401000 + i * 0x1000, f"sub_{i}", total_bytes=16 * (i + 1))
             for i in range(20)]
    bv = _FakeBV(functions=funcs)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    disp_calls = {"n": 0}
    size_calls = {"n": 0}
    real_disp = bridge.il_format._display_name
    real_size = bridge.il_format._function_size
    monkeypatch.setattr(bridge.il_format, "_display_name",
                        lambda fn: (disp_calls.__setitem__("n", disp_calls["n"] + 1), real_disp(fn))[1])
    monkeypatch.setattr(bridge.il_format, "_function_size",
                        lambda fn: (size_calls.__setitem__("n", size_calls["n"] + 1), real_size(fn))[1])

    result = instance._list_functions("active", limit=3)

    assert result["returned"] == 3 and result["total"] == 20
    # display_name + size projected for the 3 returned rows only, not all 20.
    assert disp_calls["n"] == 3
    assert size_calls["n"] == 3
    for it in result["items"]:
        assert "display_name" in it and "size" in it and "basic_block_count" in it
        assert "_fn" not in it


def test_function_list_sort_by_size_still_ranks_the_full_set(monkeypatch):
    # `--sort size` needs size for the WHOLE set to rank it, so the deferral must
    # not break ascending size ordering regardless of which page a row lands on.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    small = _FakeFunction(0x401000, "small_fn", total_bytes=16)
    huge = _FakeFunction(0x402000, "huge_fn", total_bytes=65536)
    mid = _FakeFunction(0x403000, "mid_fn", total_bytes=4096)
    bv = _FakeBV(functions=[small, huge, mid])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._list_functions("active", sort="size", limit=1)

    # Natural ordering is ascending, like address/name and Python's sorted().
    assert result["returned"] == 1
    assert result["items"][0]["address"] == "0x401000"
    assert result["items"][0]["size"] == 16


def test_list_functions_is_sorted_by_address(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x402000, "sub_402000"),
            _FakeFunction(0x401000, "sub_401000"),
        ]
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._list_functions("active")

    assert [item["address"] for item in result["items"]] == ["0x401000", "0x402000"]
    assert result["total"] == 2 and result["has_more"] is False


def test_function_list_rows_carry_basic_block_count(monkeypatch):
    # #411: `size` is a raw address span -- agents misread it as complexity. Each
    # list row carries basic_block_count, a real triage metric, computed for the
    # returned page only.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # BN exposes the count as len(fn.basic_blocks), not a basic_block_count attr.
    f1 = _FakeFunction(0x401000, "sub_401000"); f1.basic_blocks = [object()]
    f2 = _FakeFunction(0x402000, "parse_loop"); f2.basic_blocks = [object()] * 42
    bv = _FakeBV(functions=[f1, f2])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._list_functions("active")
    by = {it["address"]: it for it in result["items"]}
    assert by["0x401000"]["basic_block_count"] == 1
    assert by["0x402000"]["basic_block_count"] == 42
    assert "_fn" not in by["0x401000"]   # transient enrich key is dropped


def test_function_search_rows_carry_basic_block_count(monkeypatch):
    # #411 review: `function search` emits the same triage row shape as
    # `function list`, so its rows must carry the same page-only basic_block_count
    # (else search-driven triage falls back to the misleading byte span).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    f1 = _FakeFunction(0x401000, "parse_header"); f1.basic_blocks = [object()] * 7
    f2 = _FakeFunction(0x402000, "parse_body"); f2.basic_blocks = [object()] * 19
    other = _FakeFunction(0x403000, "unrelated"); other.basic_blocks = [object()]
    bv = _FakeBV(functions=[f1, f2, other])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._search_functions("active", "parse")
    by = {it["address"]: it for it in result["items"]}
    assert set(by) == {"0x401000", "0x402000"}     # only the matches
    assert by["0x401000"]["basic_block_count"] == 7
    assert by["0x402000"]["basic_block_count"] == 19
    assert "_fn" not in by["0x401000"]             # transient enrich key is dropped


def test_function_list_basic_block_count_guards_bad_function(monkeypatch):
    # #411 review: len(fn.basic_blocks) is unguarded -- a single problematic
    # function on the returned page would otherwise fail the whole list request.
    # A function whose basic_blocks access RAISES yields basic_block_count: None
    # and the request still succeeds (mirrors il_format._function_size's guard).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _ExplodingFunction(_FakeFunction):
        # Reading basic_blocks raises (analysis artifact). Modeled via
        # __getattribute__ so __init__'s `self.basic_blocks = []` (a __setattr__)
        # still succeeds while every READ of the attribute blows up.
        def __getattribute__(self, attr):
            if attr == "basic_blocks":
                raise RuntimeError("analysis artifact: basic_blocks unavailable")
            return super().__getattribute__(attr)

    good = _FakeFunction(0x401000, "ok_fn")
    good.basic_blocks = [object()] * 3
    good.total_bytes = 32
    bad = _ExplodingFunction(0x402000, "bad_fn")
    bv = _FakeBV(functions=[good, bad])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._list_functions("active")
    by = {it["address"]: it for it in result["items"]}
    assert by["0x401000"]["basic_block_count"] == 3
    assert by["0x402000"]["basic_block_count"] is None   # guarded, not a crash
    assert by["0x401000"]["size"] == 32
    assert by["0x401000"]["size_known"] is True
    assert by["0x402000"]["size"] == 0
    assert by["0x402000"]["size_known"] is False
    assert result["total"] == 2                          # whole request succeeded


def test_function_list_envelope_kind_and_no_functions_alias(monkeypatch):
    # #275: the canonical envelope carries a `kind` discriminator and drops the
    # deprecated `functions` alias (items is the universal container).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x401000, "sub_401000")])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    listed = instance._list_functions("active")
    assert listed["kind"] == "functions"
    assert "functions" not in listed
    assert isinstance(listed["items"], list)

    searched = instance._search_functions("active", "sub")
    assert searched["kind"] == "functions"
    assert "functions" not in searched

    counted = instance._list_functions("active", count_only=True)
    assert counted["kind"] == "functions" and counted["count"] == 1


def test_function_list_rows_carry_size_and_sort_by_size(monkeypatch):
    # Expose size on every row and keep the default sort direction consistent
    # with address/name; --reverse/--desc requests largest-first.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    small = _FakeFunction(0x1000, "small_fn"); small.total_bytes = 16
    big = _FakeFunction(0x2000, "big_fn"); big.total_bytes = 4096
    mid = _FakeFunction(0x3000, "mid_fn"); mid.total_bytes = 256
    bv = _FakeBV(functions=[small, big, mid])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    res = instance._list_functions("active")
    by_name = {r["name"]: r for r in res["items"]}
    assert by_name["small_fn"]["size"] == 16
    assert by_name["big_fn"]["size"] == 4096

    ranked = instance._list_functions("active", sort="size")
    assert [r["name"] for r in ranked["items"]] == ["small_fn", "mid_fn", "big_fn"]


def test_function_search_rows_carry_size(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x2000, "parse_packet"); fn.total_bytes = 512
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    res = instance._search_functions("active", "parse")
    assert res["items"][0]["size"] == 512


def test_list_functions_binder_forwards_sort(monkeypatch):
    # Regression: the op binder must FORWARD `sort` to the handler. A unit test
    # that calls the handler directly misses a binder that drops the param --
    # the live re-use gate caught exactly this, so guard it through the binder.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    small = _FakeFunction(0x1000, "small_fn"); small.total_bytes = 8
    big = _FakeFunction(0x2000, "big_fn"); big.total_bytes = 9000
    bv = _FakeBV(functions=[small, big])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    res = bridge._bind_list_functions(instance, {"sort": "size"}, "active")
    assert [r["name"] for r in res["items"]] == ["small_fn", "big_fn"]
    res2 = bridge._bind_search_functions(instance, {"query": "_fn", "sort": "size"}, "active")
    assert [r["name"] for r in res2["items"]] == ["small_fn", "big_fn"]


def test_function_binders_tolerate_none_limit(monkeypatch):
    """A raw-protocol / py-exec / batch caller can send `limit: None` (key
    present, null value) to list_functions / search_functions -- the CLI omits
    the key when None, but the bridge protocol accepts arbitrary params. The
    binder must read None as "no limit", not int(None) -- the same key-presence
    vs value-not-None guard bug that crashed `comment list`."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x1000, "alpha"), _FakeFunction(0x2000, "beta")])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    res = bridge._bind_list_functions(instance, {"limit": None, "offset": 0}, "active")
    assert len(res["items"]) == 2
    res2 = bridge._bind_search_functions(instance, {"query": "alpha", "limit": None, "offset": 0}, "active")
    assert len(res2["items"]) == 1


def test_function_list_envelope_uses_items_and_count_total(monkeypatch):
    # JSON-consistency (#275): function list/search expose the universal `items`
    # key (no `functions` alias) so `data["items"]` works across every list
    # command; --count carries `total` to match the list envelope's key.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x1000, "a"), _FakeFunction(0x2000, "b")])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    res = instance._list_functions("active")
    assert len(res["items"]) == 2 and "functions" not in res
    sres = instance._search_functions("active", "a")
    assert isinstance(sres["items"], list) and "functions" not in sres
    counted = instance._list_functions("active", count_only=True)
    assert counted["count"] == 2 and counted["total"] == 2

    count = instance._list_functions("active", count_only=True)
    assert count["count"] == 2 and count["total"] == 2


def test_list_functions_can_filter_by_address_range(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "sub_401000"),
            _FakeFunction(0x402000, "sub_402000"),
            _FakeFunction(0x403000, "sub_403000"),
        ]
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._list_functions("active", min_address="0x401800", max_address="0x402fff")

    assert [item["address"] for item in result["items"]] == ["0x402000"]
    assert result["total"] == 1


def test_search_functions_supports_regex(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "load_attachment"),
            _FakeFunction(0x402000, "detach_player"),
            _FakeFunction(0x403000, "update_camera"),
        ]
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._search_functions("active", "attach|detach", regex=True)

    assert [item["name"] for item in result["items"]] == ["load_attachment", "detach_player"]


def test_search_functions_rejects_invalid_regex(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x401000, "load_attachment")])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure, match="Invalid function regex"):
        instance._search_functions("active", "(", regex=True)


def test_find_var_for_restore_relocates_register_local_via_func_vars(monkeypatch):
    """On revert, the non-journaled restore must also relocate a register local
    that dropped out of the canonical set; otherwise the closure raises and the
    clean preview falsely reports 'the view may be left modified' (#156)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    narrowed = _FakeVariable(
        name="x0_1", storage=34, var_type="uint8_t", identifier=3001,
        source_type="RegisterVariableSourceType",
    )
    fn = _FakeFunction(0x401000, "process_usb")
    fn.hlil = types.SimpleNamespace(vars=[])
    fn.vars = [narrowed]

    found = bridge.mutation_engine._find_var_for_restore(
        instance, fn, 3001, 34, False
    )
    assert found is narrowed


def test_find_var_for_restore_rejects_same_storage_stranger_after_identifier_miss(monkeypatch):
    """#521: when a captured identifier no longer resolves, the restore must NOT
    fall back to a storage-only match -- a DIFFERENT variable now at the same
    storage would match and the restore closure would stamp the old name/type
    onto the wrong logical variable. Return None instead."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    # A stranger occupies storage -72 now, but with a different identifier than
    # the one we captured (3001). The captured variable is gone.
    stranger = _FakeVariable(name="other", storage=-72, var_type="int32_t", identifier=9999)
    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [stranger]

    found = bridge.mutation_engine._find_var_for_restore(
        instance, fn, 3001, -72, False
    )
    assert found is None  # must not clobber the same-storage stranger


def test_find_var_for_restore_falls_back_to_storage_when_no_identifier(monkeypatch):
    """#521: the storage fallback is still the legitimate recovery when BN never
    gave an identifier -- the same logical variable recreated with no identifier."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    var = _FakeVariable(name="var_48", storage=-72, var_type="int32_t", identifier=3001)
    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [var]

    found = bridge.mutation_engine._find_var_for_restore(
        instance, fn, None, -72, False
    )
    assert found is var


def test_load_binary_quick_skips_analysis(monkeypatch, tmp_path):
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")

    result = instance._load_binary(str(raw), quick=True)

    assert result["analyzed"] is False
    assert any("--quick" in note for note in result["notes"])
    assert bridge._headless_views[-1].analysis_updated is False  # heavy phase skipped
    bridge._headless_views.clear()


def test_load_binary_quick_is_noop_for_bndb(monkeypatch, tmp_path):
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    bndb = tmp_path / "foo.so.bndb"
    bndb.write_bytes(b"")

    result = instance._load_binary(str(bndb), quick=True)

    # A .bndb already carries its saved analysis: --quick is a no-op there.
    assert result["analyzed"] is True
    assert bridge._headless_views[-1].analysis_updated is True
    bridge._headless_views.clear()


def test_preload_binary_marks_quick_views_for_honesty(monkeypatch, tmp_path):
    # Headless `bn-agent --quick` preload must record the view in
    # _quick_loaded_views so target_info/strings stay honest (#90).
    bridge = _load_bridge(monkeypatch)
    bridge._headless_views.clear()
    bridge._quick_loaded_views.clear()
    binaryninja = sys.modules["binaryninja"]
    binaryninja.load = lambda path, update_analysis=True: _LoadBV()

    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bv = bridge._preload_binary(str(raw), quick=True)

    assert bv in bridge._quick_loaded_views          # marked quick
    assert bv.analysis_updated is False              # heavy phase skipped
    assert bv in bridge._headless_views

    # target_info and strings now tell the truth about the quick view.
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    # #775: `target_info` takes ONE snapshot now (view + listing from the
    # same refresh), so the composed seam needs stubbing too.
    monkeypatch.setattr(instance.targets, "resolve_with_snapshot",
                        lambda selector: (bv, []), raising=False)
    monkeypatch.setattr(instance.targets, "refresh", lambda: [])
    info = instance._target_info("active")
    assert info["analyzed"] is False and info["analysis_state"] == "quick"
    with pytest.raises(RuntimeError, match="bn refresh"):
        instance._strings("active", query=None, offset=0, limit=10)
    bridge._headless_views.clear()
    bridge._quick_loaded_views.clear()


def test_preload_binary_closes_view_when_analysis_fails(monkeypatch, tmp_path):
    """#609 parity: if a preload's update_analysis_and_wait() raises after load(),
    the opened view is closed and not left registered -- mirroring the runtime
    `bn load` close-on-failure guard."""
    bridge = _load_bridge(monkeypatch)
    bridge._headless_views.clear()
    bridge._quick_loaded_views.clear()

    closed: list[bool] = []

    class _FailingBV:
        def __init__(self, path):
            self.functions = [object()]
            self.view_type = "ELF"
            self.file = types.SimpleNamespace(
                filename=path,
                close=lambda: closed.append(True),
            )

        def update_analysis_and_wait(self):
            raise RuntimeError("analysis OOM")

    binaryninja = sys.modules["binaryninja"]
    binaryninja.load = lambda path, update_analysis=True: _FailingBV(path)

    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")

    with pytest.raises(RuntimeError, match="analysis OOM"):
        bridge._preload_binary(str(raw), quick=False)

    assert closed == [True], "preload did not close the view on analysis failure"
    assert bridge._headless_views == []


def test_preload_binary_full_analysis_not_marked_quick(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    bridge._headless_views.clear()
    bridge._quick_loaded_views.clear()
    binaryninja = sys.modules["binaryninja"]
    binaryninja.load = lambda path, update_analysis=True: _LoadBV()

    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bv = bridge._preload_binary(str(raw), quick=False)

    assert bv not in bridge._quick_loaded_views
    assert bv.analysis_updated is True
    bridge._headless_views.clear()


def test_preload_binary_quick_is_noop_for_sibling_bndb(monkeypatch, tmp_path):
    # When preload resolves to the sidecar .bndb, --quick is a no-op there (the
    # .bndb already carries its analysis), so the view is fully analyzed and not
    # marked quick -- same contract as `bn load --quick` on a .bndb (#178/#90).
    bridge = _load_bridge(monkeypatch)
    bridge._headless_views.clear()
    bridge._quick_loaded_views.clear()
    binaryninja = sys.modules["binaryninja"]
    binaryninja.load = lambda path, update_analysis=True: _LoadBV()

    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bndb = tmp_path / "foo.so.bndb"
    bndb.write_bytes(b"")

    bv = bridge._preload_binary(str(raw), quick=True)

    assert bv not in bridge._quick_loaded_views
    assert bv.analysis_updated is True
    bridge._headless_views.clear()
    bridge._quick_loaded_views.clear()


def test_preload_binary_bndb_recovers_analyzed_view(monkeypatch, tmp_path):
    """#458 parity: `bn-agent <file>.bndb` preloading a .bndb whose load() defaults
    to the raw container view must recover the analyzed view, not preload a
    no-symbol target."""
    bridge = _load_bridge(monkeypatch)
    bridge._headless_views.clear()
    bridge._quick_loaded_views.clear()

    bndb = tmp_path / "image.bndb"
    bndb.write_bytes(b"")
    analyzed = _LoadBV(filename=str(bndb), view_type="ELF",
                       functions=[object(), object()])
    raw = _LoadBV(filename=str(bndb), view_type="Raw", functions=[],
                  existing_views=["ELF", "Raw"], db_views={"ELF": analyzed})
    sys.modules["binaryninja"].load = lambda path, update_analysis=True: raw

    bv = bridge._preload_binary(str(bndb), quick=False)

    assert bv is analyzed                    # published the analyzed view
    assert bridge._headless_views == [analyzed]
    bridge._headless_views.clear()


def test_dispatch_rejects_non_boolean_quick(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._dispatch_on_main("load_binary", {"path": str(tmp_path / "x"), "quick": "false"}, None)
    assert exc.value.status == "invalid_request"


def test_list_functions_count_only_returns_count(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[
        _FakeFunction(0x1000, "a"),
        _FakeFunction(0x2000, "b"),
        _FakeFunction(0x3000, "c"),
    ])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # count_only carries `kind` + `total` (matching the list envelope's key)
    # alongside the back-compat `count` (#275), plus the analysis-state signal (#437).
    assert instance._list_functions(None, count_only=True) == {
        "kind": "functions", "count": 3, "total": 3,
        "analysis_state": "full", "partial": False}
    # count must match the full listing's reported total
    listing = instance._list_functions(None)
    assert listing["total"] == 3 and listing["returned"] == 3 and len(listing["items"]) == 3
    assert "functions" not in listing  # no legacy alias


def test_backward_slice_simple_chain(monkeypatch):
    """Trace a variable through one SET_VAR."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    var_r0 = _FakeSSAVariable("r0#1")
    var_r1 = _FakeSSAVariable("r1#2")

    call_insn = _FakeMLILInsn(
        0x10010,
        operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA", vars_read=[var_r0])],
        vars_read=[var_r0],
    )
    def_insn = _FakeMLILInsn(
        0x10008,
        operation="MLIL_SET_VAR_SSA",
        vars_read=[var_r1],
    )

    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(
        instructions=[call_insn],
        definitions={var_r0: def_insn},
    )
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._backward_slice("active", "test_func", "0x10010", arg_index=0)

    assert result["function"] == "test_func"
    assert result["function_address"] == "0x10000"
    assert result["target_address"] == "0x10010"
    assert result["arg_index"] == 0
    assert result["step_count"] == 2
    assert result["truncated"] is False
    assert result["trace"][0]["ssa_var"] == "r0#1"
    assert result["trace"][0]["terminates"] is False
    assert result["trace"][1]["ssa_var"] == "r1#2"
    assert result["trace"][1]["terminates"] is True
    # No reaching def and no parameter info available -> neutral terminal,
    # not a false "function parameter" claim.
    assert result["trace"][1]["reason"] == "undefined_or_global"


def test_backward_slice_undefined_var(monkeypatch):
    """Variable with no definition should terminate immediately."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    var_param = _FakeSSAVariable("arg1#0")

    call_insn = _FakeMLILInsn(
        0x10020,
        operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10020, operation="MLIL_VAR_SSA", vars_read=[var_param])],
        vars_read=[var_param],
    )

    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(instructions=[call_insn])
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._backward_slice("active", "test_func", "0x10020", arg_index=0)

    assert result["step_count"] == 1
    assert result["trace"][0]["ssa_var"] == "arg1#0"
    assert result["trace"][0]["terminates"] is True
    # The fake exposes no parameter_vars, so an undefined terminal is reported
    # neutrally rather than asserted to be a parameter.
    assert result["trace"][0]["reason"] == "undefined_or_global"


def test_backward_slice_labels_true_parameter(monkeypatch):
    """An undefined terminal that IS a formal parameter is labeled as such."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    var_param = _FakeSSAVariable("arg1#0")
    call_insn = _FakeMLILInsn(
        0x10020,
        operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10020, operation="MLIL_VAR_SSA", vars_read=[var_param])],
        vars_read=[var_param],
    )
    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(instructions=[call_insn])
    # Wire parameter info so the undefined terminal resolves to a real parameter
    # rather than the neutral "undefined" label.
    fn.parameter_vars = [_FakeSSAVariable("arg1")]
    fn.medium_level_il.ssa_form.source_function = fn
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._backward_slice("active", "test_func", "0x10020", arg_index=0)

    assert result["trace"][0]["ssa_var"] == "arg1#0"
    assert result["trace"][0]["terminates"] is True
    assert result["trace"][0]["reason"] == "function_parameter"


def test_backward_slice_depth_is_def_use_distance(monkeypatch):
    """`depth` is the real graph distance from the seed: operands of one
    definition are siblings sharing a depth, not a sequential append index."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    x = _FakeSSAVariable("x#3")
    y = _FakeSSAVariable("y#1")
    z = _FakeSSAVariable("z#2")
    # x = y <op> z : one definition reading two operands.
    def_x = _FakeMLILInsn(0x2000, operation="MLIL_SET_VAR_SSA", vars_read=[y, z], dest=x)
    call_insn = _FakeMLILInsn(
        0x2010,
        operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x2010, operation="MLIL_VAR_SSA", vars_read=[x])],
        vars_read=[x],
    )
    fn = _FakeFunction(0x2000, "f")
    fn.medium_level_il = _FakeMLILFunction(instructions=[def_x, call_insn], definitions={x: def_x})
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._backward_slice("active", "f", "0x2010", arg_index=0)
    by_var = {s["ssa_var"]: s for s in result["trace"]}

    assert by_var["x#3"]["depth"] == 0
    assert by_var["y#1"]["depth"] == 1
    assert by_var["z#2"]["depth"] == 1  # sibling of y#1, same depth (not 2)


def test_backward_slice_steps_carry_ssa_label_and_definition_reason(monkeypatch):
    """Every step gets a stable ssa_label, and an ordinary definition step now
    reports reason `definition` instead of null (#162)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    r0 = _FakeSSAVariable("r0#1")
    r1 = _FakeSSAVariable("r1#2")
    call_insn = _FakeMLILInsn(
        0x10010, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA", vars_read=[r0])], vars_read=[r0])
    def_insn = _FakeMLILInsn(0x10008, operation="MLIL_SET_VAR_SSA", vars_read=[r1])
    fn = _FakeFunction(0x10000, "f")
    fn.medium_level_il = _FakeMLILFunction(instructions=[call_insn], definitions={r0: def_insn})
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._backward_slice("active", "f", "0x10010", arg_index=0)
    assert result["trace"][0]["ssa_label"] == "r0#1"
    assert result["trace"][0]["reason"] == "definition"
    assert result["trace"][1]["ssa_label"] == "r1#2"


def test_backward_slice_field_load_carries_base_offset_width(monkeypatch):
    """A `len = [obj + 8]` field load reports reason `field_load` with structured
    base/offset/width metadata (#162)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    length = _FakeSSAVariable("len#3")
    obj = _FakeSSAVariable("obj#1")
    addr_expr = _FakeMLILInsn(
        0x2000, operation="MLIL_ADD",
        left=_FakeMLILInsn(0x2000, operation="MLIL_VAR_SSA", vars_read=[obj]),
        right=_FakeMLILInsn(0x2000, operation="MLIL_CONST", constant=8))
    load_expr = _FakeMLILInsn(0x2000, operation="MLIL_LOAD_SSA", src=addr_expr, size=4, vars_read=[obj])
    load_def = _FakeMLILInsn(0x2000, operation="MLIL_SET_VAR_SSA", src=load_expr, vars_read=[obj])
    call_insn = _FakeMLILInsn(
        0x2010, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x2010, operation="MLIL_VAR_SSA", vars_read=[length])], vars_read=[length])
    fn = _FakeFunction(0x2000, "f")
    fn.medium_level_il = _FakeMLILFunction([load_def, call_insn], definitions={length: load_def})
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._backward_slice("active", "f", "0x2010", arg_index=0)
    step = result["trace"][0]
    assert step["ssa_label"] == "len#3"
    assert step["reason"] == "field_load"
    assert step["base"] == "obj#1"
    assert step["offset"] == "0x8"
    assert step["width"] == 4


def _out_param_fixture():
    """A caller that fills a local struct via an out-pointer (`parse_record(input,
    &rec)`), then loads a field out of it and passes it to a sink. Shared by the
    #416 out-param caveat tests."""
    rec = _FakeSSAVariable("rec")            # local struct; &rec is the out-pointer
    length = _FakeSSAVariable("len#3")
    addr_of_rec = _FakeMLILInsn(0x2010, operation="MLIL_ADDRESS_OF", src=rec)
    parse_call = _FakeMLILInsn(
        0x2010, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x2010, operation="MLIL_VAR_SSA", vars_read=[]), addr_of_rec])
    addr_expr = _FakeMLILInsn(
        0x2020, operation="MLIL_ADD",
        left=_FakeMLILInsn(0x2020, operation="MLIL_ADDRESS_OF", src=rec),
        right=_FakeMLILInsn(0x2020, operation="MLIL_CONST", constant=8))
    load_expr = _FakeMLILInsn(0x2020, operation="MLIL_LOAD_SSA", src=addr_expr, size=4, vars_read=[])
    len_def = _FakeMLILInsn(0x2020, operation="MLIL_SET_VAR_SSA", src=load_expr, vars_read=[])
    sink = _FakeMLILInsn(
        0x2030, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x2030, operation="MLIL_VAR_SSA", vars_read=[]),
                _FakeMLILInsn(0x2030, operation="MLIL_VAR_SSA", vars_read=[length])],
        vars_read=[length])
    fn = _FakeFunction(0x2000, "caller")
    fn.medium_level_il = _FakeMLILFunction([parse_call, len_def, sink], definitions={length: len_def})
    return fn


def test_backward_slice_interprocedural_out_param_caveat(monkeypatch):
    """#416: a backward trace from a value loaded out of a local whose address was
    passed into a call emits an out-param boundary reason naming the callee, instead
    of silently bottoming out at the local's address (interprocedural follows only
    return values, not out-parameters)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _out_param_fixture()
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.read_taint_slice, "_callee_display_name",
                        lambda ctx, bv, ins: "parse_record")

    result = instance._backward_slice("active", "caller", "0x2030", arg_index=1, interprocedural=True)

    step = result["trace"][0]
    assert step["reason"] == "interprocedural_out_param_not_followed"
    assert step["out_param_callee"] == "parse_record"
    assert step["terminates"] is True


def test_backward_slice_out_param_caveat_stack_struct_one_hop(monkeypatch):
    """#416, the real-world shape: BN models the stack struct as a variable, so the
    slice bottoms out at the local `rec` (undefined), and the address-of is one hop
    before the call (`p#1 = &rec; parse_record(input, p#1)`). The caveat must still
    fire and name the callee."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    rec = _FakeSSAVariable("rec#2")
    addr_holder = _FakeSSAVariable("p#1")
    length = _FakeSSAVariable("len#3")
    # p#1 = &rec
    take_addr = _FakeMLILInsn(
        0x2008, operation="MLIL_SET_VAR_SSA",
        src=_FakeMLILInsn(0x2008, operation="MLIL_ADDRESS_OF", src=rec),
        vars_written=[addr_holder])
    # parse_record(input, p#1)  -- the call passes the SSA var holding &rec
    parse_call = _FakeMLILInsn(
        0x2010, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x2010, operation="MLIL_VAR_SSA", vars_read=[]),
                _FakeMLILInsn(0x2010, operation="MLIL_VAR_SSA", vars_read=[addr_holder])])
    # len = rec.len  -- BN reads the field straight off the local variable
    len_def = _FakeMLILInsn(0x2020, operation="MLIL_SET_VAR_SSA", vars_read=[rec])
    sink = _FakeMLILInsn(
        0x2030, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x2030, operation="MLIL_VAR_SSA", vars_read=[length])],
        vars_read=[length])
    fn = _FakeFunction(0x2000, "caller")
    fn.medium_level_il = _FakeMLILFunction(
        [take_addr, parse_call, len_def, sink], definitions={length: len_def})
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.read_taint_slice, "_callee_display_name",
                        lambda ctx, bv, ins: "parse_record")

    result = instance._backward_slice("active", "caller", "0x2030", arg_index=0, interprocedural=True)

    term = result["trace"][-1]
    assert term["reason"] == "interprocedural_out_param_not_followed"
    assert term["out_param_callee"] == "parse_record"


def test_backward_slice_out_param_caveat_requires_call_before_read(monkeypatch):
    """#416 ordering: a call that takes `&local` AFTER the value was read is not its
    source, so no caveat fires (guards the `use(rec.len); audit(&rec)` and
    `sink(rec.len, &rec)` false positives)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    rec = _FakeSSAVariable("rec#2")
    length = _FakeSSAVariable("len#3")
    len_def = _FakeMLILInsn(0x2020, operation="MLIL_SET_VAR_SSA", vars_read=[rec])
    sink = _FakeMLILInsn(
        0x2030, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x2030, operation="MLIL_VAR_SSA", vars_read=[length])],
        vars_read=[length])
    audit = _FakeMLILInsn(  # takes &rec, but AFTER the read at 0x2020
        0x2040, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x2040, operation="MLIL_ADDRESS_OF", src=rec)])
    fn = _FakeFunction(0x2000, "caller")
    fn.medium_level_il = _FakeMLILFunction([len_def, sink, audit], definitions={length: len_def})
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.read_taint_slice, "_callee_display_name",
                        lambda ctx, bv, ins: "audit")

    result = instance._backward_slice("active", "caller", "0x2030", arg_index=0, interprocedural=True)

    assert result["trace"][-1]["reason"] != "interprocedural_out_param_not_followed"

    # #672 (round-3 follow-up): the ordering guard is load-bearing in intra
    # mode too, since it now also gates `out_param_not_followed`.
    intra_result = instance._backward_slice("active", "caller", "0x2030", arg_index=0)
    assert intra_result["trace"][-1]["reason"] != "out_param_not_followed"


def test_backward_slice_out_param_caveat_fires_in_intra_mode(monkeypatch):
    """#672: WITHOUT --interprocedural, the out-param caveat still fires -- the
    load is named `out_param_not_followed` (the intra-mode reason string,
    distinct from `interprocedural_out_param_not_followed`) rather than reading
    as a plain, misleading `memory_load`, and the caveat surfaces in
    `assumptions` too. This is the LOAD-branch counterpart of
    `test_backward_slice_assumptions_nonempty_for_unfollowed_out_param`
    (test_read_taint_slice.py), which only drives the `def_insn is None`
    branch; `_out_param_fixture` is the only fixture that exercises the
    rewritten LOAD branch (read_taint_slice.py:894-910) end-to-end."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _out_param_fixture()
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.read_taint_slice, "_callee_display_name",
                        lambda ctx, bv, ins: "parse_record")

    result = instance._backward_slice("active", "caller", "0x2030", arg_index=1)

    step = result["trace"][0]
    assert step["reason"] == "out_param_not_followed"
    assert step["out_param_callee"] == "parse_record"
    assert step["terminates"] is True
    assert len(result["assumptions"]) == 1
    assert "parse_record" in result["assumptions"][0]


def test_backward_slice_phi_step_reason(monkeypatch):
    """A phi definition reports reason `phi_source` (#162)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    merged = _FakeSSAVariable("v#3")
    a = _FakeSSAVariable("v#1")
    b = _FakeSSAVariable("v#2")
    phi_def = _FakeMLILInsn(0x3000, operation="MLIL_VAR_PHI", vars_read=[a, b])
    call_insn = _FakeMLILInsn(
        0x3010, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x3010, operation="MLIL_VAR_SSA", vars_read=[merged])], vars_read=[merged])
    fn = _FakeFunction(0x3000, "f")
    fn.medium_level_il = _FakeMLILFunction([phi_def, call_insn], definitions={merged: phi_def})
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._backward_slice("active", "f", "0x3010", arg_index=0)
    assert result["trace"][0]["reason"] == "phi_source"


def test_backward_slice_arg_label_and_output_pointer_hint(monkeypatch):
    """An address-of arg with no value reads yields the calling-convention
    register label plus an output-pointer dead-end hint, not a bare empty trace
    (#166)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    addr_of = _FakeMLILInsn(0x4010, operation="MLIL_ADDRESS_OF", vars_read=[])
    other = _FakeMLILInsn(0x4010, operation="MLIL_VAR_SSA", vars_read=[])
    call_insn = _FakeMLILInsn(0x4010, operation="MLIL_CALL_SSA", params=[other, addr_of], vars_read=[])
    fn = _FakeFunction(0x4000, "f")
    fn.calling_convention = type("CC", (), {"int_arg_regs": ["x0", "x1", "x2"]})()
    fn.medium_level_il = _FakeMLILFunction([call_insn])
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._backward_slice("active", "f", "0x4010", arg_index=1)
    assert result["arg_label"]["index"] == 1
    assert result["arg_label"]["register"] == "x1"
    assert result["hints"]
    assert "pointer" in result["hints"][0]


def _addr_of_local_call_fixture(monkeypatch, *, callee):
    """run_tool: x1 = &var_68; <callee>(path, x1) -- `&local` arriving through a
    register copy, as on AArch64. `callee` names the call target, or None for an
    unresolved (indirect) call."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    var_68 = types.SimpleNamespace(name="var_68")
    x1 = _FakeSSAVariable("x1_9#20")
    addr_def = _FakeMLILInsn(
        0x4017f0, operation="MLIL_SET_VAR_SSA", vars_written=[x1],
        src=_FakeMLILInsn(0x4017f0, operation="MLIL_ADDRESS_OF", src=var_68))
    call_insn = _FakeMLILInsn(
        0x401804, operation="MLIL_CALL_SSA", vars_read=[x1],
        dest=_FakeConstPtr(0x401100) if callee else None,
        params=[_FakeMLILInsn(0x401804, operation="MLIL_CONST_PTR", constant=0x401d30),
                _FakeMLILInsn(0x401804, operation="MLIL_VAR_SSA", vars_read=[x1])])
    fn = _FakeFunction(0x401660, "run_tool")
    fn.medium_level_il = _FakeMLILFunction([addr_def, call_insn], definitions={x1: addr_def})
    functions = [fn] + ([_FakeFunction(0x401100, callee)] if callee else [])
    bv = _FakeBV(functions=functions)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    return instance._backward_slice("active", "run_tool", "0x401804", arg_index=1)


@pytest.mark.parametrize("callee, source", [
    ("execv", "arg:execv:1"),       # no model sources: `call:execv` would exit 2
    ("read", "call:read"),          # a source model: `call:` seeds its outputs
])
def test_backward_slice_address_of_local_via_register_copy_1001(monkeypatch, callee, source):
    """#1001 item 2: on AArch64 `&local` reaches the call through a register
    copy (`x1 = &var_68; execv(path, x1)`), so the #166 check on the call
    parameter never fired and the slice stopped after one plain definition with
    no frontier, caveat or hint. The definition is now the frontier, named, and
    the hint names commands that run as printed (#1001 review)."""
    from bn import formatters
    result = _addr_of_local_call_fixture(monkeypatch, callee=callee)

    step = result["trace"][0]
    assert (step["reason"], step["terminates"], step["local"]) == ("address_of_local", True, "var_68")
    assert [(g["reason"], g["count"], g["examples"][0]["local"]) for g in result["frontiers"]] == [
        ("address_of_local", 1, "var_68")]
    assert any("address of local var_68" in a for a in result["assumptions"]), result["assumptions"]
    hint = next(h for h in result["hints"] if "var_68" in h)
    assert f"`bn taint backward -f 0x401660 --sink arg:{callee}:1`" in hint
    assert f"`bn taint forward -f 0x401660 --source {source}`" in hint
    text = formatters._render_trace_text(result)
    assert "address of local (stores into it not followed)  [1 step] (var_68)" in text


def test_address_of_local_hint_prints_no_command_for_an_unresolved_callee_1001(monkeypatch):
    """An indirect call has no name an `arg:` locator could use; a printed
    `--sink arg:<callee>:N` fails, so the hint prints no command at all."""
    result = _addr_of_local_call_fixture(monkeypatch, callee=None)
    hint = next(h for h in result["hints"] if "var_68" in h)
    assert "did not resolve" in hint
    assert "`bn taint" not in hint and "<callee>" not in hint
    # The one command it does print runs as printed (a bare `bn decompile` exits 2).
    assert "`bn decompile 0x401660`" in hint


@pytest.mark.parametrize("arg_index, source", [
    (1, "call:recvfrom"),          # the model sources *arg:1, the payload buffer
    (5, "arg:recvfrom:5"),         # addrlen: `call:recvfrom` would not seed it
])
def test_address_of_local_hint_uses_call_only_for_a_modeled_output_1001(monkeypatch, arg_index, source):
    """#1001 review round 2: `call:<callee>` seeds what the callee's model
    declares it writes, so it is the forward source only for an argument the
    model writes; any other address-of argument is seeded as itself."""
    rts = _load_bridge(monkeypatch).read_taint_slice
    hint = rts._address_of_local_hint(arg_index, "addr", {"callee": "recvfrom"},
                                      types.SimpleNamespace(start=0x4007c0))
    assert f"`bn taint forward -f 0x4007c0 --source {source}`" in hint


def test_value_loaded_from_a_local_is_not_called_its_address_1001(monkeypatch):
    """#1001 review: `p = c ? &a : &b; puts(p[1])` -- the argument is a value
    computed FROM the locals, not the address of either. The frontier still
    names both locals; the "arg is the address of local" hint must not fire."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    pa, pb = _FakeSSAVariable("x0_1#2"), _FakeSSAVariable("x0_1#3")
    p, x0 = _FakeSSAVariable("x0_1#4"), _FakeSSAVariable("x0_2#5")
    defs = {
        pa: _FakeMLILInsn(0x4008c0, operation="MLIL_SET_VAR_SSA", vars_written=[pa],
                          src=_FakeMLILInsn(0x4008c0, operation="MLIL_ADDRESS_OF",
                                            src=types.SimpleNamespace(name="var_28"))),
        pb: _FakeMLILInsn(0x4008c8, operation="MLIL_SET_VAR_SSA", vars_written=[pb],
                          src=_FakeMLILInsn(0x4008c8, operation="MLIL_ADDRESS_OF",
                                            src=types.SimpleNamespace(name="var_38"))),
        p: _FakeMLILInsn(0x4008d0, operation="MLIL_VAR_PHI", vars_read=[pa, pb]),
        x0: _FakeMLILInsn(0x4008e8, operation="MLIL_SET_VAR_SSA", vars_written=[x0],
                          vars_read=[p],
                          src=_FakeMLILInsn(0x4008e8, operation="MLIL_ADD", vars_read=[p])),
    }
    call_insn = _FakeMLILInsn(
        0x4008f0, operation="MLIL_CALL_SSA", vars_read=[x0],
        params=[_FakeMLILInsn(0x4008f0, operation="MLIL_VAR_SSA", vars_read=[x0])])
    fn = _FakeFunction(0x4008a0, "sel")
    fn.medium_level_il = _FakeMLILFunction(list(defs.values()) + [call_insn], definitions=defs)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: _FakeBV(functions=[fn]))

    result = instance._backward_slice("active", "sel", "0x4008f0", arg_index=0)

    assert sorted(s["local"] for s in result["trace"] if s.get("reason") == "address_of_local") == [
        "var_28", "var_38"]
    assert not any("is the address of" in h for h in result["hints"]), result["hints"]


def test_backward_slice_plain_copy_is_not_an_address_of_local_1001(monkeypatch):
    """Must-not-fire twin: an ordinary register copy stays a non-terminal
    `definition` step and adds no address-of hint or caveat."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    x1 = _FakeSSAVariable("x1#2")
    x8 = _FakeSSAVariable("x8#1")
    copy_def = _FakeMLILInsn(0x4010, operation="MLIL_SET_VAR_SSA", vars_written=[x1],
                             vars_read=[x8],
                             src=_FakeMLILInsn(0x4010, operation="MLIL_VAR_SSA", vars_read=[x8]))
    call_insn = _FakeMLILInsn(
        0x4020, operation="MLIL_CALL_SSA", vars_read=[x1],
        params=[_FakeMLILInsn(0x4020, operation="MLIL_VAR_SSA", vars_read=[x1])])
    fn = _FakeFunction(0x4000, "f")
    fn.medium_level_il = _FakeMLILFunction([copy_def, call_insn], definitions={x1: copy_def})
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._backward_slice("active", "f", "0x4020", arg_index=0)

    assert (result["trace"][0]["reason"], result["trace"][0]["terminates"]) == ("definition", False)
    assert not any("address of" in h for h in result["hints"]), result["hints"]
    assert not any("address of local" in a for a in result["assumptions"]), result["assumptions"]


def test_backward_slice_no_call_at_address(monkeypatch):
    """Address with no call instruction should raise."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(instructions=[])
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure, match="No call instruction"):
        instance._backward_slice("active", "test_func", "0x99999", arg_index=0)


def test_backward_slice_bad_arg_index(monkeypatch):
    """Out-of-range arg index should raise."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    call_insn = _FakeMLILInsn(
        0x10010,
        operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA")],
    )

    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(instructions=[call_insn])
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure, match="out of range"):
        instance._backward_slice("active", "test_func", "0x10010", arg_index=5)


def test_backward_slice_no_mlil(monkeypatch):
    """Function with no MLIL should raise."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = None
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure, match="has no mlil"):
        instance._backward_slice("active", "test_func", "0x10010", arg_index=0)


def test_backward_slice_interprocedural_follows_callee(monkeypatch):
    """Interprocedural trace crosses into a callee when the traced arg is a call return value."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    # Callee: returns callee_ret_var, which is a copy of a parameter (callee_def_var).
    callee_ret_var = _FakeSSAVariable("result#1")
    callee_def_var = _FakeSSAVariable("tmp#2")
    callee_ret_insn = _FakeMLILInsn(
        0x20010, operation="MLIL_RET", vars_read=[callee_ret_var],
    )
    callee_def_insn = _FakeMLILInsn(
        0x20008, operation="MLIL_SET_VAR_SSA", vars_read=[callee_def_var],
    )
    callee = _FakeFunction(0x20000, "callee_fn")
    callee.medium_level_il = _FakeMLILFunction(
        instructions=[callee_ret_insn, callee_def_insn],
        definitions={callee_ret_var: callee_def_insn},
    )

    # Caller: arg 0 of the traced call (0x10010) is `ret_var`, and `ret_var` is
    # defined by an *inner* call to callee_fn (0x1000c). The slice must therefore
    # cross the call boundary into callee_fn.
    ret_var = _FakeSSAVariable("r0#3")
    # dest is a const-ptr expression (like real MLIL), so _resolve_callee exercises
    # the int(dest)->TypeError->`.constant` fallback rather than the raw-int fast path.
    inner_call_insn = _FakeMLILInsn(
        0x1000c, operation="MLIL_CALL_SSA", dest=_FakeConstPtr(0x20000),
    )
    target_call_insn = _FakeMLILInsn(
        0x10010,
        operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA", vars_read=[ret_var])],
        vars_read=[ret_var],
    )
    caller = _FakeFunction(0x10000, "caller_fn")
    caller.medium_level_il = _FakeMLILFunction(
        instructions=[inner_call_insn, target_call_insn],
        definitions={ret_var: inner_call_insn},
    )
    # Register both functions so _resolve_callee can find callee by address.
    bv = _FakeBV(functions=[caller, callee])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._backward_slice(
        "active", "caller_fn", "0x10010", arg_index=0,
        interprocedural=True, ip_depth=2,
    )

    assert result["interprocedural"] is True
    trace = result["trace"]
    # Exactly one boundary crossing, into callee_fn.
    cross = [s for s in trace if s.get("cross_function")]
    assert len(cross) == 1, f"expected one cross-function step, got {trace}"
    assert cross[0]["callee"] == "callee_fn"
    assert cross[0]["reason"] == "cross_function"
    assert cross[0]["terminates"] is False
    # Recursion actually entered the callee body (steps tagged with its context)...
    assert any(s.get("function_context") == "callee_fn" for s in trace)
    # ...and bottomed out at an undefined terminal in the callee (its
    # parameter; the fake exposes no parameter_vars to confirm that, so it is
    # reported neutrally).
    assert trace[-1]["terminates"] is True
    assert trace[-1]["reason"] == "undefined_or_global"


def test_backward_slice_ip_rejects_llil(monkeypatch):
    """Interprocedural mode should still reject LLIL (no get_ssa_var_definition)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    call_insn = _FakeMLILInsn(
        0x10010, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA")],
    )
    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(instructions=[call_insn])
    fn.low_level_il = _FakeMLILFunction(instructions=[call_insn])
    # Patch low_level_il.ssa_form to be a bare object without get_ssa_var_definition
    class _NoSsaDefs:
        basic_blocks = []
    fn.low_level_il.ssa_form = _NoSsaDefs()
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    with pytest.raises(bridge.OperationFailure, match="SSA form does not support"):
        instance._backward_slice("active", "test_func", "0x10010", arg_index=0,
                                 view="llil", interprocedural=True)


# --- function search --exact ---


def test_search_functions_exact_match(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "system"),
            _FakeFunction(0x402000, "QAudioSystemPlugin"),
            _FakeFunction(0x403000, "sprintf"),
        ]
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._search_functions("active", "system", exact=True)

    assert result["returned"] == 1 and result["total"] == 1
    assert result["items"][0]["name"] == "system"
    assert result["items"][0]["address"] == "0x401000"


def test_search_functions_exact_case_insensitive(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "System"),
            _FakeFunction(0x402000, "QSystemPlugin"),
        ]
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._search_functions("active", "system", exact=True)

    assert result["returned"] == 1
    assert result["items"][0]["name"] == "System"


def test_search_functions_exact_no_match(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "system_ex"),
            _FakeFunction(0x402000, "_system"),
        ]
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._search_functions("active", "system", exact=True)

    assert result["items"] == [] and result["total"] == 0


# ---------------------------------------------------------------------------
# Visible degradation markers for IL / pseudo-C rendering failures
# ---------------------------------------------------------------------------


def test_function_text_marks_il_failure_instead_of_silent_prototype(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    warnings: list[str] = []
    monkeypatch.setattr(bridge.bn, "log_warn", lambda message: warnings.append(message))

    fn = _FakeFunction(0x401000, "player_update")  # has no .hlil attribute

    text = instance._function_text(None, fn, view="hlil")

    assert text.startswith("// bn: IL rendering failed (")
    assert "showing prototype only" in text.splitlines()[0]
    assert warnings  # failure was logged, not swallowed


def test_function_text_rejects_unknown_view(monkeypatch):
    # #527: an unrecognized view must raise, not silently fall back to HLIL and
    # then get mislabeled with the raw requested view string by the caller.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "player_update")
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._function_text(None, fn, view="pseudo")
    assert exc.value.status == "unsupported"
    assert "pseudo" in exc.value.message


def test_function_text_accepts_valid_views(monkeypatch):
    # Valid views still render (no raise); llil is a legitimate view.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "player_update")
    for view in ("hlil", "mlil", "llil"):
        text = instance._function_text(None, fn, view=view)
        assert isinstance(text, str)


def test_function_text_renders_the_hlil_tree_for_a_string_operation_827(monkeypatch):
    # #827 item 5: `_format_hlil_tree` read `ins.operation.name` DIRECTLY at three
    # sites, so an instruction whose `operation` is a bare STRING raised
    # AttributeError mid-tree -- and `_function_text` swallows that into the flat
    # `il.instructions` listing, so an entire tree silently rendered as the
    # inferior listing instead of the tree the view actually had. The defensive
    # `_il_op_name` exists for exactly this shape; each site is driven below, one
    # root per site, so a fix that guards only the first one still fails.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _Op:
        def __init__(self, name):
            self.name = name

    class _StrNode:
        """An IL instruction whose `operation` is a STRING rather than an object
        with `.name` (a unit/double shape, and a documented possibility)."""

        def __init__(self, op_name, text, **attrs):
            self.operation = op_name
            self.text = text
            for key, value in attrs.items():
                setattr(self, key, value)

        def __str__(self):
            return self.text

    class _StrBlock(_StrNode):
        def __init__(self, children):
            super().__init__("HLIL_BLOCK", "")
            self.children = children

        def __iter__(self):
            return iter(self.children)

    class _OpNode(_StrNode):
        """The normal shape: `operation` is an object carrying `.name`."""

        def __init__(self, op_name, text, **attrs):
            super().__init__(_Op(op_name), text, **attrs)

    class _FlatFallback:
        def __str__(self):
            return "FLAT-FALLBACK-LINE"

    def render(root):
        hlil = types.SimpleNamespace(root=root, instructions=[_FlatFallback()])
        fn = types.SimpleNamespace(name="widget_poll", start=0x401000, hlil=hlil)
        return instance._function_text(None, fn, view="hlil")

    # 1. the root/statement op (pre-fix: AttributeError on the root's own op).
    text = render(_StrBlock([_StrNode("HLIL_ASSIGN", "x = 1", address=0x401000)]))
    assert "x = 1" in text and "FLAT-FALLBACK-LINE" not in text

    # 2. the HLIL_IF false-branch op.
    branch = _OpNode("HLIL_IF", "if (x)", condition="x")
    branch.true = _StrBlock([_StrNode("HLIL_ASSIGN", "y = 2")])
    branch.false = _StrNode("HLIL_NOP", "empty else")
    text = render(branch)
    assert "if (x)" in text and "y = 2" in text
    assert "FLAT-FALLBACK-LINE" not in text

    # 3. the HLIL_SWITCH default op.
    switch = _OpNode("HLIL_SWITCH", "switch (x)", condition="x")
    switch.cases = [_StrNode("HLIL_CASE", "case 1:", values=[1],
                             body=_StrNode("HLIL_ASSIGN", "y = 3"))]
    switch.default = _StrNode("HLIL_NOP", "empty default")
    text = render(switch)
    assert "switch (x)" in text and "case 1:" in text and "y = 3" in text
    # A NOP default renders no `default:` arm -- the comparison itself is the
    # site, so this pins that it was made through the guarded helper.
    assert "default:" not in text
    assert "FLAT-FALLBACK-LINE" not in text


def test_il_function_for_rejects_unknown_view(monkeypatch):
    # #527: the structured-IL boundary must reject an unknown view rather than
    # silently substituting MLIL (which the caller then labels as requested).
    bridge = _load_bridge(monkeypatch)
    func = types.SimpleNamespace(name="f", start=0x1000)
    with pytest.raises(bridge.OperationFailure) as exc:
        bridge.il_format._il_function_for(func, "garbage", False)
    assert exc.value.status == "unsupported"
    assert "garbage" in exc.value.message


def test_decompile_force_requested_but_not_skipped_echoes_flag(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "small_fn")  # analysis_skipped defaults False
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    _install_fake_pseudo_c(
        monkeypatch, bridge, fn,
        [[(0x401000, "int32_t small_fn()")], [(0x401000, "{")], [(0x401000, "}")]],
    )

    result = instance._decompile("active", "small_fn", force_analysis=True)

    # Nothing was skipped, so no reanalysis ran ...
    assert result["analysis_forced"] is False
    assert fn.reanalyzed is False
    # ... but the echo confirms --force-analysis was honored, not silently ignored.
    assert result["analysis_force_requested"] is True


def test_function_info_requires_refresh_when_quick_loaded(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # Quick-loaded: size/xref/signature fields are bogus until analysis runs.
    bridge._quick_loaded_views.add(bv)
    with pytest.raises(RuntimeError, match="loaded with --quick"):
        instance._function_info(None, "main")
    bridge._quick_loaded_views.discard(bv)


def test_taint_requires_refresh_when_quick_loaded(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # Quick-loaded: "no call to <sink> found" would misdiagnose missing
    # analysis. Refuse with a directive instead.
    bridge._quick_loaded_views.add(bv)
    with pytest.raises(RuntimeError, match="loaded with --quick"):
        instance._taint(None, {"function": "main", "direction": "backward", "sinks": ["system"]})
    bridge._quick_loaded_views.discard(bv)


def test_target_info_reports_quick_analysis_state(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    # #775: `target_info` takes ONE snapshot now (view + listing from the
    # same refresh), so the composed seam needs stubbing too.
    monkeypatch.setattr(instance.targets, "resolve_with_snapshot",
                        lambda selector: (bv, []), raising=False)
    monkeypatch.setattr(instance.targets, "refresh", lambda: [])

    bridge._quick_loaded_views.add(bv)
    info = instance._target_info("active")
    assert info["analyzed"] is False
    assert info["analysis_state"] == "quick"

    bridge._quick_loaded_views.discard(bv)
    info2 = instance._target_info("active")
    assert info2["analyzed"] is True
    assert info2["analysis_state"] == "full"


def test_target_info_reports_unanalyzed_state_for_raw_bndb(monkeypatch):
    """#458: a .bndb that restored a raw container with no product view is tracked
    in _unanalyzed_views; target info must report analysis_state=unanalyzed (not
    full), so a JSON consumer isn't told a 0-function raw view is fully analyzed."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    # #775: `target_info` takes ONE snapshot now (view + listing from the
    # same refresh), so the composed seam needs stubbing too.
    monkeypatch.setattr(instance.targets, "resolve_with_snapshot",
                        lambda selector: (bv, []), raising=False)
    monkeypatch.setattr(instance.targets, "refresh", lambda: [])

    bridge._unanalyzed_views.add(bv)
    info = instance._target_info("active")
    assert info["analyzed"] is False
    assert info["analysis_state"] == "unanalyzed"
    bridge._unanalyzed_views.discard(bv)


def test_load_quick_marks_view_full_load_does_not(monkeypatch, tmp_path):
    bridge, instance, _ = _setup_load_test(monkeypatch)

    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    result = instance._load_binary(str(raw), quick=True)
    assert result["analyzed"] is False
    quick_bv = bridge._headless_views[-1]
    assert quick_bv in bridge._quick_loaded_views

    raw2 = tmp_path / "bar.so"
    raw2.write_bytes(b"")
    full = instance._load_binary(str(raw2), quick=False)
    assert full["analyzed"] is True
    full_bv = bridge._headless_views[-1]
    assert full_bv not in bridge._quick_loaded_views

    bridge._headless_views.clear()


def test_possible_values_uses_source_when_instruction_undetermined(monkeypatch):
    # BN leaves a SET_VAR instruction's value-set undetermined while the SOURCE
    # expression carries the real const; report the source's value-set (#52).
    src = types.SimpleNamespace(possible_values=_pvs("ConstantValue", value=0xc48))
    ins = types.SimpleNamespace(
        address=0x1000, operation=types.SimpleNamespace(name="MLIL_SET_VAR_SSA"),
        possible_values=_pvs("UndeterminedValue"), src=src, dest="value#1")
    bridge, instance = _dataflow_values_instance(monkeypatch, ins)
    res = instance._possible_values(None, "f", "0x1000")
    assert res["value_basis"] == "source_expression"
    assert res["possible_values"]["type"] == "ConstantValue"
    assert res["possible_values"]["value"] == 0xc48
    assert "call_destination" not in res


def test_possible_values_keeps_instruction_set_when_determined(monkeypatch):
    # When the instruction itself has a determined value-set, keep it (don't
    # blindly prefer the source).
    src = types.SimpleNamespace(possible_values=_pvs("UndeterminedValue"))
    ins = types.SimpleNamespace(address=0x1000, possible_values=_pvs("ConstantValue", value=7), src=src)
    bridge, instance = _dataflow_values_instance(monkeypatch, ins)
    res = instance._possible_values(None, "f", "0x1000")
    assert res["value_basis"] == "instruction"
    assert res["possible_values"]["value"] == 7


def test_possible_values_no_source_uses_instruction(monkeypatch):
    # An instruction with no .src (e.g. not an assignment) falls back to its own
    # value-set.
    ins = types.SimpleNamespace(address=0x1000, possible_values=_pvs("ConstantValue", value=3))
    bridge, instance = _dataflow_values_instance(monkeypatch, ins)
    res = instance._possible_values(None, "f", "0x1000")
    assert res["value_basis"] == "instruction"
    assert res["possible_values"]["value"] == 3


class _ValuesExpression(types.SimpleNamespace):
    def __str__(self):
        return self.text


@pytest.mark.parametrize("operation", [
    "MLIL_CALL", "MLIL_CALL_SSA", "MLIL_CALL_UNTYPED", "MLIL_CALL_UNTYPED_SSA",
    "MLIL_TAILCALL", "MLIL_TAILCALL_SSA", "MLIL_TAILCALL_UNTYPED", "MLIL_TAILCALL_UNTYPED_SSA",
])
@pytest.mark.parametrize("instruction_value", [None, 7])
def test_possible_values_call_destination_is_separate(monkeypatch, operation, instruction_value):
    dest = _ValuesExpression(
        text="handler#2", possible_values=_pvs("InSetOfValues", values={0x402000, 0x403000}))
    instr_pvs = (_pvs("UndeterminedValue") if instruction_value is None
                 else _pvs("ConstantValue", value=instruction_value))
    ins = _ValuesExpression(
        text="handler#2(request#0)", address=0x401120,
        operation=types.SimpleNamespace(name=operation), possible_values=instr_pvs, dest=dest)
    bridge, instance = _dataflow_values_instance(monkeypatch, ins)
    res = instance._possible_values(None, "dispatch_request", "0x401120")
    assert res["value_basis"] == "instruction"
    assert res["expression"] == "handler#2(request#0)"
    assert res["source_expression"] is None
    assert res["possible_values"] == bridge.il_format._serialize_pvs(instr_pvs)
    assert "values" not in res["possible_values"]
    assert res["call_destination"] == {
        "expression": "handler#2",
        "value_basis": "destination_expression",
        "possible_values": bridge.il_format._serialize_pvs(dest.possible_values),
    }
    assert res["call_destination"]["possible_values"]["values"] == [0x402000, 0x403000]
    # The same native operand is what callgraph's resolution reads.
    targets = instance._pvs_targets(object(), dest.possible_values)
    assert sorted(row["address"] for row in targets) == ["0x402000", "0x403000"]
    # The JSON surface keeps the sets in distinct objects, not just in text.
    encoded = json.loads(json.dumps(res))
    assert encoded["call_destination"]["value_basis"] == "destination_expression"
    assert "values" not in encoded["possible_values"]


@pytest.mark.parametrize("destination_state", ["undetermined", "no_values", "no_operand"])
def test_possible_values_unknown_call_destination_stays_unknown(monkeypatch, destination_state):
    ins = _ValuesExpression(
        text="handler#2(request#0)", address=0x401120,
        operation=types.SimpleNamespace(name="MLIL_CALL_SSA"),
        possible_values=_pvs("UndeterminedValue"))
    if destination_state != "no_operand":
        ins.dest = _ValuesExpression(text="handler#2")
        if destination_state == "undetermined":
            ins.dest.possible_values = _pvs("UndeterminedValue")
    bridge, instance = _dataflow_values_instance(monkeypatch, ins)
    res = instance._possible_values(None, "dispatch_request", "0x401120")
    assert res["possible_values"]["type"] == "UndeterminedValue"
    destination = res["call_destination"]
    assert destination["value_basis"] == "destination_expression"
    assert destination["expression"] == (None if destination_state == "no_operand" else "handler#2")
    if destination_state == "undetermined":
        assert destination["possible_values"]["type"] == "UndeterminedValue"
        assert "values" not in destination["possible_values"]
    else:
        assert destination["possible_values"] is None


@pytest.mark.parametrize("phi_op", ["MLIL_VAR_PHI", "MLIL_MEM_PHI"])
def test_possible_values_call_at_phi_address_reads_call_operand(monkeypatch, phi_op):
    phi = _ValuesExpression(
        text="handler#2 = phi(handler#0, handler#1)", address=0x401120,
        operation=types.SimpleNamespace(name=phi_op),
        possible_values=_pvs("UndeterminedValue"))
    call = _ValuesExpression(
        text="handler#2(request#0)", address=0x401120,
        operation=types.SimpleNamespace(name="MLIL_CALL_SSA"),
        possible_values=_pvs("UndeterminedValue"),
        dest=_ValuesExpression(
            text="handler#2", possible_values=_pvs("InSetOfValues", values={0x402000, 0x403000})))
    bridge, instance = _dataflow_values_instance(monkeypatch, call)
    monkeypatch.setattr(bridge.il_format, "_il_function_for",
                        lambda *args: types.SimpleNamespace(instructions=[phi, call]))
    result = instance._possible_values(None, "dispatch_request", "0x401120")
    assert result["expression"] == str(call)
    assert result["possible_values"]["type"] == "UndeterminedValue"
    assert result["call_destination"]["possible_values"]["values"] == [0x402000, 0x403000]


def test_possible_values_call_destination_does_not_rescue_invalid_at(monkeypatch):
    ins = _ValuesExpression(
        text="handler#2(request#0)", address=0x401120,
        operation=types.SimpleNamespace(name="MLIL_CALL_SSA"),
        possible_values=_pvs("UndeterminedValue"),
        dest=_ValuesExpression(
            text="handler#2", possible_values=_pvs("InSetOfValues", values={0x402000, 0x403000})))
    bridge, instance = _dataflow_values_instance(monkeypatch, ins)
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._possible_values(None, "dispatch_request", "0x401121")
    assert exc.value.status == "no_instruction"
    assert exc.value.requested["at"] == "0x401121"



def test_possible_values_no_instruction_at_address_raises(monkeypatch):
    # #526: when no MLIL instruction begins at --at, refuse instead of returning a
    # success dict with expression:None that makes a bogus address look real.
    ins = types.SimpleNamespace(address=0x1000, possible_values=_pvs("ConstantValue", value=3))
    bridge, instance = _dataflow_values_instance(monkeypatch, ins)
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._possible_values(None, "f", "0x2000")   # no instruction at 0x2000
    assert exc.value.status == "no_instruction"


def test_pvs_determined_helper(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    assert instance._pvs_determined(_pvs("ConstantValue", value=1)) is True
    assert instance._pvs_determined(_pvs("UndeterminedValue")) is False
    assert instance._pvs_determined(None) is False


def test_address_context_disasm_uses_target_function_arch(monkeypatch):
    # target_context disasm must decode with the TARGET function's arch, not the
    # bv default -- a THUMB2 target must not be ARM-misdecoded into garbage (#53).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    thumb_arch = object()
    fn = types.SimpleNamespace(start=0x12e74, name="thumb_fn", arch=thumb_arch)

    class _BV:
        def get_function_at(self, a):
            return fn if a == 0x12e74 else None

    _stub_code_context(monkeypatch, instance, {"name": "thumb_fn"})
    recorded = {}

    def fake_safe(bv_, address, arch=None):
        recorded["arch"] = arch
        return "bx pc" if arch is thumb_arch else "udf #0xd478"

    monkeypatch.setattr(instance.ctx, "_safe_disassembly", fake_safe)
    ctx = instance._address_context(_BV(), 0x12e74, include_disasm=True)
    assert recorded["arch"] is thumb_arch           # used the target function's arch
    assert ctx["disasm"] == "bx pc"                 # not the ARM misdecode


def test_address_context_disasm_respects_explicit_arch(monkeypatch):
    # An explicitly-passed arch (the caller's arch for a code-ref site) is used
    # as-is, not overridden by the function-at-address derivation.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    explicit = object()
    _stub_code_context(monkeypatch, instance, None)
    recorded = {}

    def fake_safe(bv_, address, arch=None):
        recorded["arch"] = arch
        return "x"

    monkeypatch.setattr(instance.ctx, "_safe_disassembly", fake_safe)
    instance._address_context(object(), 0x1000, include_disasm=True, arch=explicit, assume_code=True)
    assert recorded["arch"] is explicit


def test_function_info_reports_unimplemented_instructions(monkeypatch):
    """function info aggregates instructions BN's lifter could not model so an
    FP-heavy function isn't mistaken for fully analyzed (#206)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x405000, "transform")
    # Two unlifted FP instructions (e.g. AArch64 fnmsub) surface as LLIL_UNIMPL.
    fn.low_level_il = [_FakeBlock([
        _FakeMLILInsn(0x4056f8, operation="LLIL_UNIMPL"),
        _FakeMLILInsn(0x4056fc, operation="LLIL_UNIMPL"),
        _FakeMLILInsn(0x405700, operation="LLIL_SET_REG"),
    ])]
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_info("active", "transform")

    ui = result["unimplemented_instructions"]
    assert ui["count"] == 2
    assert ui["addresses"] == ["0x4056f8", "0x4056fc"]
    assert ui["truncated"] is False


def test_function_list_carries_demangled_display_name(monkeypatch):
    """function list entries carry a demangled display_name (#196)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "_ZN3foo3bar4recvEi")
    sym = _FakeSymbol("FunctionSymbol")
    sym.short_name = "foo::bar::recv"
    fn.symbol = sym
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._list_functions(None)
    item = result["items"][0]
    assert item["name"] == "_ZN3foo3bar4recvEi"
    assert item["display_name"] == "foo::bar::recv"


def _cxx_function(start: int, mangled: str, demangled: str) -> _FakeFunction:
    fn = _FakeFunction(start, mangled)
    fn.symbol = _FakeSymbol("FunctionSymbol")
    fn.symbol.short_name = demangled
    return fn


def test_function_search_matches_the_display_name_by_default_978(monkeypatch):
    """#978: a short class-name query matched any function whose MANGLED name
    happened to contain the letters. Default search matches the demangled
    display name the text prints; `mangled` matches raw spellings instead, and
    a default search reports what only a mangled spelling would have matched."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[
        _cxx_function(0x401000, "_ZN4demo6Widget6handleEi", "demo::Widget::handle"),
        _cxx_function(0x402000, "_ZN4demo5Gizmo3runEv", "demo::Gizmo::run"),
        _FakeFunction(0x403000, "main"),
    ])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    def names(result):
        return [item["display_name"] for item in result["items"]]

    shown = instance._search_functions(None, "Widget::handle")
    assert names(shown) == ["demo::Widget::handle"]
    assert "mangled_only_matches" not in shown
    assert instance._search_functions(None, "Widget::handle", mangled=True)["items"] == []

    # `6Widget` occurs only inside the mangled spelling.
    raw_query = instance._search_functions(None, "6Widget")
    assert raw_query["items"] == [] and raw_query["mangled_only_matches"] == 1
    counted = instance._search_functions(None, "6Widget", count_only=True)
    assert counted["count"] == 0 and counted["mangled_only_matches"] == 1
    raw_hit = instance._search_functions(None, "6Widget", mangled=True)
    assert [item["name"] for item in raw_hit["items"]] == ["_ZN4demo6Widget6handleEi"]
    assert "mangled_only_matches" not in raw_hit

    # An unmangled name is the same spelling either way.
    for mangled in (False, True):
        assert names(instance._search_functions(None, "main", mangled=mangled)) == ["main"]


def _sort_population():
    return [
        _cxx_function(0x401000, "_Z10pump_queueiPc", "pump_queue"),
        _cxx_function(0x402000, "_Z9copy_tailPcPKcim", "copy_tail"),
        _FakeFunction(0x403000, "main"),
    ]


def test_sort_name_orders_by_the_name_the_text_shows_978(monkeypatch):
    """#978 review: text shows display names, so `--sort name` sorting by the
    mangled `fn.name` printed pump_queue before copy_tail. Sort by the display
    name by default and by the raw name under `mangled`, on list and search."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=_sort_population())
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    for read in (lambda **k: instance._list_functions(None, **k),
                 lambda **k: instance._search_functions(None, ".", regex=True, **k)):
        shown = read(sort="name")["items"]
        assert [item["name"] for item in shown] == [
            "_Z9copy_tailPcPKcim", "main", "_Z10pump_queueiPc"]
        raw = read(sort="name", mangled=True)["items"]
        assert [item["name"] for item in raw] == [
            "_Z10pump_queueiPc", "_Z9copy_tailPcPKcim", "main"]
        assert [item["name"] for item in read(sort="name", reverse=True)["items"]] == [
            "_Z10pump_queueiPc", "main", "_Z9copy_tailPcPKcim"]


def test_implicit_callsite_scope_keeps_raw_names_across_overloads_978(monkeypatch):
    """#986 review: with no --within the bridge fills `within_query` itself.
    JSON keeps the RAW caller name there (#978's contract): three `ns::put`
    overloads share one display name, which would merge three callers and
    cannot be passed back as `--within`, while each raw name resolves to its
    own function."""
    bridge = _load_bridge(monkeypatch)
    from bn_agent_bridge import read_listing
    instance = bridge.BinaryNinjaBridge()
    callers = [
        _cxx_function(0x400a68, "_ZN2ns3putEPcPKci", "ns::put"),
        _cxx_function(0x400aa0, "_ZN2ns3putEPci", "ns::put"),
        _cxx_function(0x400acc, "_ZN2ns3putEPcl", "ns::put"),
    ]
    bv = _FakeBV(functions=callers)
    bv.get_code_refs = lambda address: [types.SimpleNamespace(function=fn) for fn in callers]
    scope = read_listing._all_caller_functions(bv, {0x400100})
    queries = [query for query, _fn in scope]
    assert queries == ["_ZN2ns3putEPcPKci", "_ZN2ns3putEPci", "_ZN2ns3putEPcl"]
    for query, fn in scope:
        [(_echo, resolved)] = instance.ctx._resolve_scope_functions(bv, [query])
        assert resolved is fn
    with pytest.raises(Exception, match="(?i)ambiguous"):
        instance.ctx._resolve_scope_functions(bv, ["ns::put"])


def test_backward_slice_constant_arg_reports_value_hint(monkeypatch):
    """A constant/immediate arg (e.g. read(fd, buf, 0x1fff)'s count) has no SSA
    definition to trace. Instead of a renderer-only "constant or immediate" line
    with no value and an empty JSON `hints`, the bridge surfaces a structured
    hint naming the constant -- so text AND JSON consumers both see it."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    const_arg = _FakeMLILInsn(0x4010, operation="MLIL_CONST", vars_read=[], constant=0x1fff)
    other = _FakeMLILInsn(0x4010, operation="MLIL_VAR_SSA", vars_read=[])
    call_insn = _FakeMLILInsn(0x4010, operation="MLIL_CALL_SSA", params=[other, const_arg], vars_read=[])
    fn = _FakeFunction(0x4000, "f")
    fn.medium_level_il = _FakeMLILFunction([call_insn])
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._backward_slice("active", "f", "0x4010", arg_index=1)
    assert result["trace"] == []
    assert result["hints"]
    assert "0x1fff" in result["hints"][0]
    assert "constant" in result["hints"][0].lower()


def test_backward_slice_arg_index_message_states_mlil_convention(monkeypatch):
    """An out-of-range --arg names the MLIL count and the 0-based/MLIL
    convention so a user reading pseudo-C doesn't reach for the wrong index (#226)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    # A call the decompiler may render with one visible argument: exactly one
    # MLIL param, so only --arg 0 is valid.
    call_insn = _FakeMLILInsn(
        0x10010,
        operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA")],
    )
    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(instructions=[call_insn])
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._backward_slice("active", "test_func", "0x10010", arg_index=1)
    msg = str(exc.value)
    assert "this call has 1 MLIL argument(s)" in msg
    assert "(index 0)" in msg
    assert "0-based" in msg and "MLIL" in msg


def test_backward_slice_call_boundary_names_callee(monkeypatch):
    """A value originating at a (non-interprocedural) call boundary names the
    resolved callee symbol instead of just terminating at the raw target (#193)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    # Callee body so the resolver returns a real function with a name.
    callee = _FakeFunction(0x20000, "strlen")
    callee.medium_level_il = _FakeMLILFunction(instructions=[])

    # Caller: arg 0 of the traced call is `ret_var`, defined by an inner call to
    # `strlen`. Default (non-interprocedural) mode terminates at the boundary.
    ret_var = _FakeSSAVariable("r0#3")
    inner_call_insn = _FakeMLILInsn(
        0x1000c, operation="MLIL_CALL_SSA", dest=_FakeConstPtr(0x20000),
    )
    target_call_insn = _FakeMLILInsn(
        0x10010,
        operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA", vars_read=[ret_var])],
        vars_read=[ret_var],
    )
    caller = _FakeFunction(0x10000, "caller_fn")
    caller.medium_level_il = _FakeMLILFunction(
        instructions=[inner_call_insn, target_call_insn],
        definitions={ret_var: inner_call_insn},
    )
    bv = _FakeBV(functions=[caller, callee])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._backward_slice("active", "caller_fn", "0x10010", arg_index=0)

    boundary = [s for s in result["trace"] if s.get("reason") == "call_or_jump_boundary"]
    assert len(boundary) == 1, f"expected one call boundary, got {result['trace']}"
    assert boundary[0]["terminates"] is True
    assert boundary[0]["callee"] == "strlen"


# --- imports --summary ---




def test_search_functions_count_only_binder_forwards(monkeypatch):
    # The op binder must forward count_only so the CLI's `--count` reaches the
    # handler (regression guard against a CLI flag that never wires through).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x1000, "parse_a"), _FakeFunction(0x2000, "parse_b")])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    res = bridge._bind_search_functions(
        instance, {"query": "parse", "count_only": True}, "active",
    )

    assert res["count"] == 2 and res["total"] == 2

def test_search_functions_count_only_honors_address_filter(monkeypatch):
    # count_only reflects --min-address/--max-address: the count is computed
    # after match+address filtering, before paging -- parity with the listing.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[
        _FakeFunction(0x1000, "parse_a"),
        _FakeFunction(0x2000, "parse_b"),
        _FakeFunction(0x3000, "parse_c"),
    ])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    res = instance._search_functions("active", "parse", min_address="0x2000", count_only=True)

    assert res["count"] == 2 and res["total"] == 2

def test_search_functions_count_only_returns_total(monkeypatch):
    # Parity with `list_functions` count_only (#252): `search_functions` returns
    # just the match total, not the (paged) list, so an agent can size a query.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[
        _FakeFunction(0x1000, "parse_a"),
        _FakeFunction(0x2000, "parse_b"),
        _FakeFunction(0x3000, "unrelated"),
    ])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    res = instance._search_functions("active", "parse", count_only=True)

    assert res["count"] == 2 and res["total"] == 2
    assert "functions" not in res and "items" not in res


# --- #193 Part 4: resolve a mid-function (contained) address ----------------
#
# taint/trace report sinks at instruction addresses, frequently mid-callee. The
# function-scoped READ verbs must resolve such an address to its containing
# function so the sink feeds straight back into the next command -- while the
# strict (mutation) path keeps erroring, so a stray address can't rename/retype
# the wrong function.

def _mid_function_bv():
    fn = _FakeFunction(0x401000, "parse_packet")
    fn.basic_blocks = [_FakeBasicBlock(0x401000, 0x401040)]  # spans 0x401000..0x401040
    return _FakeBV(functions=[fn], memory={0x401000: b"\x90" * 0x40}), fn


def test_find_function_resolves_contained_address_only_when_opted_in(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, fn = _mid_function_bv()

    # Strict by default (the mutation contract): a mid-function address errors.
    with pytest.raises(RuntimeError, match="No function found at address 0x401010"):
        instance._find_function(bv, "0x401010")

    # contained=True (the read contract): resolves to the containing function.
    resolved = instance._find_function(bv, "0x401010", contained=True)
    assert resolved is fn

    # An exact start still resolves with contained=True (no behavior change).
    assert instance._find_function(bv, "0x401000", contained=True) is fn


def test_find_function_resolves_decimal_contained_address(monkeypatch):
    # #626 review (Finding 3): a decimal-spelled interior address (a documented
    # address format) must resolve via containment exactly like the 0x-hex
    # spelling -- not skip the containment branch and hard-error "No function
    # found". 4198416 == 0x401010, an interior address of parse_packet.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, fn = _mid_function_bv()

    assert 4198416 == 0x401010
    resolved = instance._find_function(bv, "4198416", contained=True)
    assert resolved is fn

    # A decimal exact start resolves too (no regression).
    assert instance._find_function(bv, "4198400", contained=True) is fn

    # A decimal interior address WITHOUT contained stays strict, like hex.
    with pytest.raises(RuntimeError, match="No function found at address 0x401010"):
        instance._find_function(bv, "4198416")


def test_find_function_contained_address_outside_any_function_still_errors(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _ = _mid_function_bv()

    # An address in no function (e.g. .data) errors even with contained=True --
    # "nothing found" must stay distinct from "resolved to a container".
    with pytest.raises(RuntimeError, match="No function found at address 0xdeadbeef"):
        instance._find_function(bv, "0xdeadbeef", contained=True)


@pytest.mark.parametrize("call", [
    pytest.param(lambda inst: inst._decompile("active", "0x401010"), id="decompile"),
    pytest.param(lambda inst: inst._function_info("active", "0x401010"), id="function_info"),
    pytest.param(lambda inst: inst._il("active", "0x401010", "mlil", False), id="il"),
    pytest.param(lambda inst: inst._disasm("active", "0x401010"), id="disasm"),
    # proto get and local list describe the same function as `function info`
    # (proto is a strict subset), so they tolerate an interior address too.
    pytest.param(lambda inst: inst._get_prototype("active", "0x401010"), id="proto_get"),
    pytest.param(lambda inst: inst._list_locals_for_function("active", "0x401010"), id="local_list"),
])
def test_read_verbs_resolve_mid_function_address(monkeypatch, call):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _ = _mid_function_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = call(instance)

    # Resolved to the container, reported by its real start (not the typed addr).
    assert result["function"]["name"] == "parse_packet"
    assert result["function"]["address"] == "0x401000"
    # ...and annotated so the agent knows the sink was mid-function, not the start.
    assert result["resolved_from"] == {"requested_address": "0x401010", "offset": "+0x10"}


def test_read_verb_exact_start_has_no_resolved_from(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _ = _mid_function_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._decompile("active", "0x401000")

    assert result["function"]["address"] == "0x401000"
    assert "resolved_from" not in result


def test_containment_meta_decimal_matches_hex(monkeypatch):
    # #626 review round 2 (Finding 1): _containment_meta gated the disclosure on a
    # 0x prefix, so a DECIMAL interior address -- which _find_function already
    # resolves to its container (hex OR decimal via _parse_address) -- silently
    # dropped resolved_from/offset. The metadata for the decimal spelling must be
    # IDENTICAL to the equivalent hex spelling.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _, fn = _mid_function_bv()
    assert 4198416 == 0x401010  # interior of parse_packet (0x401000..0x401040)

    hex_meta = instance.ctx._containment_meta("0x401010", fn)
    dec_meta = instance.ctx._containment_meta("4198416", fn)
    assert hex_meta == {"requested_address": "0x401010", "offset": "+0x10"}
    assert dec_meta == {
        "requested_address": "0x401010",
        "offset": "+0x10",
        "input_format": "decimal",
    }

    # An exact bare-decimal address is disclosed so it cannot be mistaken for a
    # digit-only symbol name; exact 0x input remains unremarkable.
    assert instance.ctx._containment_meta("4198400", fn) == {
        "requested_address": "0x401000",
        "offset": "+0x0",
        "input_format": "decimal",
    }
    assert instance.ctx._containment_meta("0x401000", fn) is None
    # A plain name is not an address -> no disclosure.
    assert instance.ctx._containment_meta("parse_packet", fn) is None


@pytest.mark.parametrize("call", [
    pytest.param(lambda inst: inst._decompile("active", "4198416"), id="decompile"),
    pytest.param(lambda inst: inst._function_info("active", "4198416"), id="function_info"),
    pytest.param(lambda inst: inst._il("active", "4198416", "mlil", False), id="il"),
    pytest.param(lambda inst: inst._disasm("active", "4198416"), id="disasm"),
    pytest.param(lambda inst: inst._get_prototype("active", "4198416"), id="proto_get"),
    pytest.param(lambda inst: inst._list_locals_for_function("active", "4198416"), id="local_list"),
])
def test_read_verbs_decimal_mid_address_discloses_like_hex(monkeypatch, call):
    # #626 review round 2 (Finding 1): the INTERACTION the original tests missed --
    # a DECIMAL interior-address request to a containment-enabled read must surface
    # the SAME resolved_from as the equivalent hex request (4198416 == 0x401010),
    # not resolve to the container while omitting the disclosure.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _ = _mid_function_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    assert 4198416 == 0x401010

    result = call(instance)

    assert result["function"]["name"] == "parse_packet"
    assert result["function"]["address"] == "0x401000"
    # requested_address is normalized to hex even though the request was decimal.
    assert result["resolved_from"] == {
        "requested_address": "0x401010",
        "offset": "+0x10",
        "input_format": "decimal",
    }


# --- #626: extend the mid-function (contained) contract to the evidence /
# dataflow READ verbs (#193 Part 4 shipped only decompile/info/il/disasm/etc.).
# A sink address reported by taint/trace lands mid-callee, and these verbs must
# resolve it to the containing function the same way decompile already does.

@pytest.mark.parametrize("call", [
    pytest.param(lambda inst: inst._structured_il("active", "0x401010"), id="structured_il"),
    pytest.param(lambda inst: inst._resolved_calls("active", "0x401010"), id="resolved_calls"),
])
def test_dataflow_reads_resolve_mid_function_address(monkeypatch, call):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _ = _mid_function_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    # The IL body is irrelevant here -- the point under test is that
    # _find_function resolves the interior address (contained=True) and the
    # result is annotated. Stub the IL so the handler runs to completion.
    monkeypatch.setattr(
        bridge.il_format, "_il_function_for",
        lambda fn, view, ssa: types.SimpleNamespace(instructions=[]),
    )

    result = call(instance)

    assert result["function"]["name"] == "parse_packet"
    assert result["function"]["address"] == "0x401000"
    assert result["resolved_from"] == {"requested_address": "0x401010", "offset": "+0x10"}


@pytest.mark.parametrize("call", [
    pytest.param(lambda inst: inst._structured_il("active", "0x401000"), id="structured_il"),
    pytest.param(lambda inst: inst._resolved_calls("active", "0x401000"), id="resolved_calls"),
])
def test_dataflow_reads_exact_start_has_no_resolved_from(monkeypatch, call):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _ = _mid_function_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(
        bridge.il_format, "_il_function_for",
        lambda fn, view, ssa: types.SimpleNamespace(instructions=[]),
    )

    result = call(instance)

    assert result["function"]["address"] == "0x401000"
    assert "resolved_from" not in result


def test_possible_values_resolves_mid_function_address(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _ = _mid_function_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    # An MLIL instruction begins at the interior address so the value-set read
    # reaches its result (not an early no_instruction refusal): this proves the
    # containment resolution happens before, not instead of, the real work.
    ins = types.SimpleNamespace(
        address=0x401010, possible_values=_pvs("ConstantValue", value=7), src=None,
    )
    monkeypatch.setattr(
        bridge.il_format, "_il_function_for",
        lambda fn, view, ssa: types.SimpleNamespace(instructions=[ins]),
    )

    result = instance._possible_values("active", "0x401010", "0x401010")

    assert result["function"]["name"] == "parse_packet"
    assert result["function"]["address"] == "0x401000"
    assert result["possible_values"]["value"] == 7
    assert result["resolved_from"] == {"requested_address": "0x401010", "offset": "+0x10"}


def test_possible_values_exact_start_has_no_resolved_from(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _ = _mid_function_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    ins = types.SimpleNamespace(
        address=0x401000, possible_values=_pvs("ConstantValue", value=7), src=None,
    )
    monkeypatch.setattr(
        bridge.il_format, "_il_function_for",
        lambda fn, view, ssa: types.SimpleNamespace(instructions=[ins]),
    )

    result = instance._possible_values("active", "0x401000", "0x401000")

    assert result["function"]["address"] == "0x401000"
    assert "resolved_from" not in result


def test_defuse_resolves_mid_function_address(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _ = _mid_function_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    # Reach the full def/use body so resolved_from is only set after the
    # variable lookup succeeds -- not a vacuous pass that skips var resolution.
    il = types.SimpleNamespace(
        instructions=[],
        get_ssa_var_definition=lambda v: None,
        get_ssa_var_uses=lambda v: [],
    )
    monkeypatch.setattr(bridge.il_format, "_il_function_for", lambda fn, view, ssa: il)
    ssa_var = types.SimpleNamespace(var=types.SimpleNamespace(name="arg1", type="int"), version=0)
    monkeypatch.setattr(
        bridge.il_format, "_resolve_ssa_variable",
        lambda func, il_, sel: (ssa_var, []),
    )
    monkeypatch.setattr(bridge.il_format, "_ssa_var_entry", lambda v: {"ssa": "arg1#0"})

    result = instance._defuse("active", "0x401010", "arg1#0")

    assert result["function"]["name"] == "parse_packet"
    assert result["function"]["address"] == "0x401000"
    assert result["resolved_from"] == {"requested_address": "0x401010", "offset": "+0x10"}


def test_defuse_exact_start_has_no_resolved_from(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _ = _mid_function_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    il = types.SimpleNamespace(
        instructions=[],
        get_ssa_var_definition=lambda v: None,
        get_ssa_var_uses=lambda v: [],
    )
    monkeypatch.setattr(bridge.il_format, "_il_function_for", lambda fn, view, ssa: il)
    ssa_var = types.SimpleNamespace(var=types.SimpleNamespace(name="arg1", type="int"), version=0)
    monkeypatch.setattr(
        bridge.il_format, "_resolve_ssa_variable",
        lambda func, il_, sel: (ssa_var, []),
    )
    monkeypatch.setattr(bridge.il_format, "_ssa_var_entry", lambda v: {"ssa": "arg1#0"})

    result = instance._defuse("active", "0x401000", "arg1#0")

    assert result["function"]["address"] == "0x401000"
    assert "resolved_from" not in result


def test_backward_slice_out_of_range_notes_stack_passed_varargs_324(monkeypatch):
    # #324: an --arg at/beyond the calling convention's integer-arg registers is
    # likely STACK-passed (BN's MLIL/HLIL call model omits those); the
    # out-of-range error must say so and point at the LLIL view, not silently
    # treat the arg as absent. An out-of-range index BELOW the register count
    # (a dropped register arg, a different gap) must NOT get the stack note.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    call_insn = _FakeMLILInsn(
        0x10010, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA"),
                _FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA")])
    fn = _FakeFunction(0x10000, "logger_caller")
    fn.medium_level_il = _FakeMLILFunction(instructions=[call_insn])
    fn.calling_convention = type("CC", (), {"int_arg_regs": ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]})()
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._backward_slice("active", "logger_caller", "0x10010", arg_index=7)
    msg = str(exc.value)
    assert "STACK" in msg and "likely passed on" in msg
    assert "llil" in msg and "#324" in msg

    # arg 2 is exactly the FIRST register slot BN did not recover (n=2 params): a
    # register arg likely dropped by a narrow prototype, not a stack arg. It gets
    # the (hedged) register-drop note pointing at LLIL, never the STACK note.
    with pytest.raises(bridge.OperationFailure) as exc2:
        instance._backward_slice("active", "logger_caller", "0x10010", arg_index=2)
    msg2 = str(exc2.value)
    assert "STACK" not in msg2
    assert "register-passed arg BN dropped" in msg2 and "#324" in msg2 and "llil" in msg2

    # arg 5 is below the 6 register args but BEYOND the first-missing slot -- a
    # genuinely low-arity call where the index is simply out of range. No note
    # (gating on arg_index == n avoids nagging on every low-arity call; #324).
    with pytest.raises(bridge.OperationFailure) as exc3:
        instance._backward_slice("active", "logger_caller", "0x10010", arg_index=5)
    msg3 = str(exc3.value)
    assert "STACK" not in msg3 and "register-passed arg BN dropped" not in msg3


def test_backward_slice_out_of_range_stack_note_fires_on_i386_cdecl_324(monkeypatch):
    # #324: on a pure stack-argument ABI (i386 cdecl, int_arg_regs=[]) EVERY
    # out-of-range arg is stack-passed. Previously an empty int_arg_regs collapsed to
    # a None reg-count and the STACK caveat was silently suppressed -- the exact
    # pure-stack-ABI case the ticket names. It must now fire.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    call_insn = _FakeMLILInsn(
        0x10010, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA")])
    fn = _FakeFunction(0x10000, "cdecl_caller")
    fn.medium_level_il = _FakeMLILFunction(instructions=[call_insn])
    fn.calling_convention = type("CC", (), {"int_arg_regs": []})()  # pure stack ABI
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._backward_slice("active", "cdecl_caller", "0x10010", arg_index=3)
    msg = str(exc.value)
    assert "STACK" in msg and "#324" in msg
    assert "cdecl" in msg  # names the pure-stack-ABI case explicitly


def test_callgraph_result_carries_kind_envelope(monkeypatch):
    """dataflow callgraph JSON must carry kind:'callgraph' so a consumer of the
    {kind, ...} family can identify it, instead of a bare {function, callees,
    callers} off-envelope object (#371.2)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    func = types.SimpleNamespace(name="f", start=0x1000, caller_sites=[])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda sel: object())
    monkeypatch.setattr(instance.ctx, "_find_function", lambda bv, ident, **kw: func)

    result = bridge.read_decompile._resolved_calls(
        instance.ctx, None, "f", direction="callers")
    assert result["kind"] == "callgraph"
    assert result["function"]["name"] == "f"
    assert "callers" in result


# --- #550: linear disasm warns / snaps on a non-instruction-boundary start ---


def _boundary_bv():
    fn = _FakeFunction(0x1000, "known")
    fn.basic_blocks = [_FakeBasicBlock(0x1000, 0x1006)]  # starts at 0x1000/0x1002/0x1004
    return _FakeBV(
        functions=[fn],
        memory={0x1000: b"\x90" * 16},
        disassembly={0x1000: "insA", 0x1002: "insB", 0x1003: "junkB",
                     0x1004: "insC", 0x1005: "junkC"},
        instruction_lengths={0x1000: 2, 0x1002: 2, 0x1004: 2, 0x1003: 1, 0x1005: 1},
    )


def test_disasm_linear_warns_on_non_boundary_start(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _boundary_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    res = instance._disasm(None, "0x1003", linear=2)   # inside `known`, mid-instruction
    bw = res["boundary_warning"]
    assert bw is not None
    assert bw["in_function"]["name"] == "known"
    assert bw["nearest_start_at_or_below"] == "0x1002"
    assert bw["nearest_start_above"] == "0x1004"
    assert "NOT a recovered instruction boundary" in res["note"]
    assert res["snapped_from"] is None
    assert res["order"] == "address-linear"
    assert res["address"] == "0x1003"           # not moved without --snap


def test_disasm_linear_snaps_to_instruction(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _boundary_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    res = instance._disasm(None, "0x1003", linear=2, snap_to_instruction=True)
    assert res["address"] == "0x1002"           # snapped down to the enclosing start
    assert res["snapped_from"] == "0x1003"
    assert res["boundary_warning"] is None       # landed on a real boundary
    assert "snapped 0x1003" in res["note"]


def test_disasm_linear_boundary_start_is_silent(monkeypatch):
    # Starting exactly on a recovered instruction boundary produces no warning.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _boundary_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    res = instance._disasm(None, "0x1002", linear=2)
    assert res["boundary_warning"] is None
    assert "NOT a recovered instruction boundary" not in res["note"]


def test_disasm_linear_no_boundary_warning_outside_function(monkeypatch):
    # A start not inside any function (the classic stripped/data case) is not a
    # boundary concern -- no warning.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(memory={0x2000: b"\x90" * 8},
                 disassembly={0x2000: "nop", 0x2002: "nop"},
                 instruction_lengths={0x2000: 2, 0x2002: 2})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    res = instance._disasm(None, "0x2001", linear=1)
    assert res["boundary_warning"] is None


# --- cfg: first-class CFG read op (promoted out of py_exec for bn-lens) ------


def _cfg_asm_bv():
    fn = _FakeFunction(0x401000, "process_packet", "int32_t process_packet(char* buf)")
    b2 = _FakeCFGBlock(0x401010, lines=[_FakeCFGLine(0x401010, "ret")])
    b1 = _FakeCFGBlock(
        0x401000,
        lines=[_FakeCFGLine(0x401000, "cmp eax, 0x0"),
               _FakeCFGLine(0x401004, "je 0x401010")],
        edges=[_FakeCFGEdge(b2, "TrueBranch"),
               _FakeCFGEdge(None, "IndirectBranch")],  # unresolved: disclosed (#682)
    )
    fn.basic_blocks = [b1, b2]
    return _FakeBV(functions=[fn]), fn


def test_cfg_asm_blocks_lines_and_edges(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _fn = _cfg_asm_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._cfg(None, "process_packet", view="asm")

    assert result["kind"] == "cfg"
    assert result["view"] == "asm"
    assert result["function"] == {"name": "process_packet", "address": "0x401000"}
    blocks = result["blocks"]
    assert [b["start"] for b in blocks] == ["0x401000", "0x401010"]
    assert blocks[0]["insns"] == [
        {"address": "0x401000", "text": "cmp eax, 0x0"},
        {"address": "0x401004", "text": "je 0x401010"},
    ]
    # #682 item 3: a null-target edge is NOT emitted as a row. It would have
    # to carry `to: null`, and bn-tui types that field `pub to: String` with
    # no serde default -- so a null fails the decode, and because the failure
    # is inside `Vec<CfgEdge>` the whole CFG parse fails and the TUI shows an
    # EMPTY view with no error. The shape is also unreachable on BN 6.1.
    # The unresolved target is reported on the BLOCK instead, where the
    # disclosure is additive and no consumer breaks.
    assert blocks[0]["edges"] == [{"to": "0x401010", "branch_type": "TrueBranch"}]
    assert blocks[0]["undetermined_edges"] is True
    assert blocks[1]["edges"] == []
    assert "undetermined_edges" not in blocks[1]


def test_cfg_edge_rows_stay_exactly_two_keys_for_a_strict_consumer_682(monkeypatch):
    # Must-not-fire twin for #682 item 3, repointed at the contract that
    # actually matters. A strict consumer (bn-tui) decodes an edge as
    # `{to: String, <kind>: String}` with `to` non-optional, so the edge row
    # must never grow a key that changes its shape and must never carry a
    # non-string `to` -- the whole reason the unresolved case moved to the
    # block. Assert the row's exact key set rather than the absence of one
    # name a later change could rename: #682 item 2 renamed the kind key to
    # `branch_type`, which is a change this assertion is meant to notice.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _fn = _cfg_asm_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._cfg(None, "process_packet", view="asm")

    for block in result["blocks"]:
        for edge in block["edges"]:
            assert set(edge) == {"to", "branch_type"}, edge
            assert isinstance(edge["to"], str), edge


def test_cfg_asm_blocks_are_sorted_by_start_not_bn_iteration_order(monkeypatch):
    # #682 item 4: BN yields asm basic blocks in its own order, which is not
    # address order. Feed them deliberately interleaved -- the shape observed
    # on a real target -- and require address order out.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401090, "handler", "void handler(void)")
    b_90 = _FakeCFGBlock(0x401090, lines=[_FakeCFGLine(0x401090, "push rbp")])
    b_b0 = _FakeCFGBlock(0x4010B0, lines=[_FakeCFGLine(0x4010B0, "ret")])
    b_9d = _FakeCFGBlock(0x40109D, lines=[_FakeCFGLine(0x40109D, "test eax, eax")])
    b_a7 = _FakeCFGBlock(0x4010A7, lines=[_FakeCFGLine(0x4010A7, "jmp 0x4010b0")])
    fn.basic_blocks = [b_90, b_b0, b_9d, b_a7]
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._cfg(None, "handler", view="asm")

    assert [b["start"] for b in result["blocks"]] == [
        "0x401090", "0x40109d", "0x4010a7", "0x4010b0",
    ]


def test_cfg_discloses_a_block_whose_successors_bn_could_not_resolve_682(monkeypatch):
    # #682 item 3, the LIVE half. A cross-dogfood sweep of 12,191 functions x
    # 3 IL levels over four targets found ZERO edges with `target is None`:
    # `BasicBlock._make_edges` asserts the reference is non-None and wraps it
    # through `_create_instance`, which cannot return None. What a real
    # unresolvable indirect jump (`jmp rax`) produces instead is NO EDGES AT
    # ALL -- so it rendered byte-identically to a block with no successor,
    # the same lost distinction, in the shape that actually occurs.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "dispatch", "void dispatch(void)")
    blk = _FakeCFGBlock(0x401000,
                        lines=[_FakeCFGLine(0x401000, "jmp rax")],
                        edges=[], undetermined=True)
    fn.basic_blocks = [blk]
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._cfg(None, "dispatch", view="asm")

    block = result["blocks"][0]
    assert block["edges"] == []
    assert block["undetermined_edges"] is True


def test_cfg_a_genuine_dead_end_block_is_not_marked_undetermined_682(monkeypatch):
    # Must-not-fire twin, and the whole point of the disclosure: a `ret`
    # block really has no successors. If both shapes carried the marker the
    # distinction would be lost in the other direction.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "leaf", "void leaf(void)")
    fn.basic_blocks = [_FakeCFGBlock(0x401000,
                                     lines=[_FakeCFGLine(0x401000, "ret")],
                                     edges=[])]
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._cfg(None, "leaf", view="asm")

    block = result["blocks"][0]
    assert block["edges"] == []
    assert "undetermined_edges" not in block


def test_cfg_a_block_that_cannot_answer_the_probe_claims_neither_682(monkeypatch):
    # An IL block object or a reduced view need not implement the property.
    # Absent -- or raising -- is INDETERMINATE: it must not be reported as
    # undetermined (a fabricated claim) and must not raise out of the op.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "reduced", "void reduced(void)")
    blk = _FakeCFGBlock(0x401000, lines=[_FakeCFGLine(0x401000, "ret")], edges=[])
    del blk.has_undetermined_outgoing_edges
    fn.basic_blocks = [blk]
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._cfg(None, "reduced", view="asm")

    assert "undetermined_edges" not in result["blocks"][0]



def test_cfg_a_probe_that_raises_claims_neither_682(monkeypatch):
    # #889b review: the comment promises "absent OR raising" is indeterminate,
    # but only the ABSENT variant drove the except path -- the raising half of
    # a defensive read was unexercised. The whole fix hinges on reading BN's
    # answer defensively, so the path that exists for a throwing probe needs a
    # probe that throws.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "hostile", "void hostile(void)")
    blk = _FakeCFGBlock(0x401000, lines=[_FakeCFGLine(0x401000, "ret")], edges=[])

    # A plain `__get__`-only class is a NON-data descriptor, so the instance
    # attribute `__init__` sets would shadow it and the probe would never
    # raise -- a vacuous test that passes against code with no except clause
    # at all. Use a subclass whose property really raises, on an instance
    # that has no shadowing entry in its __dict__.
    class _HostileBlock(type(blk)):
        @property
        def has_undetermined_outgoing_edges(self):
            raise RuntimeError("core refused the undetermined-edges query")

    blk.__class__ = _HostileBlock
    del blk.__dict__["has_undetermined_outgoing_edges"]
    fn.basic_blocks = [blk]
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._cfg(None, "hostile", view="asm")

    assert "undetermined_edges" not in result["blocks"][0]


def test_cfg_il_levels_emit_il_instruction_indexes_not_addresses(monkeypatch):
    # THE load-bearing contract for bn-lens: one assembly instruction can expand
    # to several IL blocks whose first lines share the SAME address, so block
    # `start` / edge `to` must be the IL instruction INDEX (hex), which is
    # unique -- the lens keys block identity and edge routing on parse_hex(start).
    # Per-line `a` stays a real address at every level.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "checked_div", "int32_t checked_div(int32_t a, int32_t b)")
    ilb2 = _FakeCFGBlock(2, lines=[_FakeCFGLine(0x401000, "temp0 = a / b")])
    ilb1 = _FakeCFGBlock(
        0,
        lines=[_FakeCFGLine(0x401000, "if (b == 0) trap")],
        edges=[_FakeCFGEdge(ilb2, "FalseBranch")],
    )
    fn.mlil = _FakeILCFGFunction([ilb1, ilb2])
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._cfg(None, "0x401000", view="mlil")

    blocks = result["blocks"]
    # Both blocks' first lines sit at 0x401000; starts stay distinct IL indexes.
    assert [b["start"] for b in blocks] == ["0x0", "0x2"]
    assert blocks[0]["edges"] == [{"to": "0x2", "branch_type": "FalseBranch"}]
    assert blocks[0]["insns"][0]["address"] == "0x401000"
    assert blocks[1]["insns"][0]["address"] == "0x401000"


def test_cfg_hlil_view_uses_hlil_function(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "main", "int32_t main()")
    fn.hlil = _FakeILCFGFunction([_FakeCFGBlock(0, lines=[_FakeCFGLine(0x401000, "return 0")])])
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._cfg(None, "main", view="hlil")

    assert result["view"] == "hlil"
    assert result["blocks"][0]["start"] == "0x0"
    assert result["blocks"][0]["insns"] == [{"address": "0x401000", "text": "return 0"}]


def test_cfg_il_unavailable_degrades_to_empty_blocks_with_warning(monkeypatch):
    # Mirrors the proven py_exec behavior: IL not materialized -> empty CFG, not
    # an error -- but now with an explicit warning instead of a silent [].
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "stub", "void stub()")  # no .mlil attribute at all
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._cfg(None, "stub", view="mlil")

    assert result["blocks"] == []
    assert any("mlil" in w for w in result["warnings"])


def test_cfg_rejects_unknown_view(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, _fn = _cfg_asm_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._cfg(None, "process_packet", view="llil")
    assert "view" in str(exc.value)


# --- Cross-surface decimal-disclosure parity ------------------------------
#
# The skill promises that a bare-decimal address is accepted everywhere a
# function identifier is accepted, and disclosed with ONE envelope shape. Agents
# reported observing different keys on different surfaces, so this enumerates
# EVERY containment-enabled read and drives the same four spellings through each
# one. It is behavioral: each entry runs the real handler and reads the real
# payload, so a new read that forgets `_annotate_containment` (or annotates with
# its own key set) fails here rather than being caught by a source grep.

def _containment_surface_calls():
    """(id, callable(instance, identifier)) for every containment-enabled read."""
    return [
        ("decompile", lambda inst, ident: inst._decompile("active", ident)),
        ("function_info", lambda inst, ident: inst._function_info("active", ident)),
        ("proto_get", lambda inst, ident: inst._get_prototype("active", ident)),
        ("local_list", lambda inst, ident: inst._list_locals_for_function("active", ident)),
        ("il", lambda inst, ident: inst._il("active", ident, "mlil", False)),
        ("cfg", lambda inst, ident: inst._cfg("active", ident)),
        ("disasm", lambda inst, ident: inst._disasm("active", ident)),
        ("structured_il", lambda inst, ident: inst._structured_il("active", ident)),
        ("resolved_calls", lambda inst, ident: inst._resolved_calls("active", ident)),
        ("defuse", lambda inst, ident: inst._defuse("active", ident, "arg1#0")),
        # `at` names the instruction to inspect and is always an address; the
        # identifier under test is the FUNCTION selector, which is what carries
        # the containment disclosure.
        ("possible_values", lambda inst, ident: inst._possible_values("active", ident, "0x401000")),
        ("evidence_function", lambda inst, ident: inst._function_evidence("active", ident, context=0)),
    ]


def _containment_instance(monkeypatch):
    """A bridge bound to parse_packet @ 0x401000..0x401040 with just enough IL
    stubbing that every surface reaches its result instead of refusing early."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # A block that satisfies BOTH the containment span (`end`) and `cfg`'s
    # disassembly-line reader, so one fixture drives every surface.
    block = _FakeCFGBlock(
        0x401000,
        lines=[_FakeCFGLine(0x401000, "push rbp"), _FakeCFGLine(0x401010, "ret")],
    )
    block.end = 0x401040
    fn = _FakeFunction(0x401000, "parse_packet")
    fn.basic_blocks = [block]
    bv = _FakeBV(functions=[fn], memory={0x401000: b"\x90" * 0x40})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # An MLIL instruction at each candidate address so `possible_values` resolves
    # a real value-set rather than short-circuiting on "no instruction here".
    instructions = [
        types.SimpleNamespace(
            address=address,
            possible_values=_pvs("ConstantValue", value=7),
            src=None,
        )
        for address in (0x401000, 0x401010)
    ]
    il = types.SimpleNamespace(
        instructions=instructions,
        get_ssa_var_definition=lambda v: None,
        get_ssa_var_uses=lambda v: [],
    )
    monkeypatch.setattr(bridge.il_format, "_il_function_for", lambda fn, view, ssa: il)
    ssa_var = types.SimpleNamespace(
        var=types.SimpleNamespace(name="arg1", type="int"), version=0
    )
    monkeypatch.setattr(
        bridge.il_format, "_resolve_ssa_variable", lambda func, il_, sel: (ssa_var, [])
    )
    monkeypatch.setattr(bridge.il_format, "_ssa_var_entry", lambda v: {"ssa": "arg1#0"})
    return instance


assert 4198416 == 0x401010 and 4198400 == 0x401000


@pytest.mark.parametrize(
    "surface,call", _containment_surface_calls(), ids=[c[0] for c in _containment_surface_calls()]
)
def test_every_containment_read_discloses_decimal_interior_identically(monkeypatch, surface, call):
    instance = _containment_instance(monkeypatch)

    hex_result = call(instance, "0x401010")
    dec_result = call(instance, "4198416")

    assert hex_result["function"]["address"] == "0x401000", surface
    assert dec_result["function"]["address"] == "0x401000", surface
    # ONE documented shape, requested_address normalized to hex on both, and the
    # decimal spelling is the only difference between them.
    assert hex_result["resolved_from"] == {
        "requested_address": "0x401010",
        "offset": "+0x10",
    }, surface
    assert dec_result["resolved_from"] == {
        "requested_address": "0x401010",
        "offset": "+0x10",
        "input_format": "decimal",
    }, surface


@pytest.mark.parametrize(
    "surface,call", _containment_surface_calls(), ids=[c[0] for c in _containment_surface_calls()]
)
def test_every_containment_read_discloses_decimal_exact_start_identically(monkeypatch, surface, call):
    # A digit-only token that lands exactly on the start is still disclosed, so
    # it can never be silently mistaken for a symbol named "4198400" -- with a
    # zero offset, which is what tells a reader it was NOT a containment hit.
    instance = _containment_instance(monkeypatch)

    result = call(instance, "4198400")

    assert result["function"]["address"] == "0x401000", surface
    assert result["resolved_from"] == {
        "requested_address": "0x401000",
        "offset": "+0x0",
        "input_format": "decimal",
    }, surface


@pytest.mark.parametrize(
    "surface,call", _containment_surface_calls(), ids=[c[0] for c in _containment_surface_calls()]
)
@pytest.mark.parametrize("identifier", ["0x401000", "parse_packet"])
def test_every_containment_read_leaves_exact_hex_and_names_unannotated(
    monkeypatch, surface, call, identifier
):
    instance = _containment_instance(monkeypatch)

    result = call(instance, identifier)

    assert result["function"]["address"] == "0x401000", surface
    assert "resolved_from" not in result, surface


def test_decompile_discloses_quick_analysis_state(monkeypatch):
    """#820: `decompile` ANSWERS on a --quick view (matrix: partial), so the
    envelope must carry the VIEW's analysis state. `analysis_skipped` cannot
    stand in for it: that is a per-FUNCTION "BN declined to analyze this one"
    flag, and it is False on a quick view whose body is unresolved-name Pseudo-C
    with an empty `warnings` list -- exactly the answer that read as analyzed."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "small_fn")
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    _install_fake_pseudo_c(
        monkeypatch, bridge, fn,
        [[(0x401000, "int32_t small_fn()")], [(0x401000, "{")], [(0x401000, "}")]],
    )

    full = instance._decompile("active", "small_fn")
    assert full["analysis_state"] == "full"
    assert full["partial"] is False

    bridge._quick_loaded_views.add(bv)
    try:
        quick = instance._decompile("active", "small_fn")
    finally:
        bridge._quick_loaded_views.discard(bv)

    assert quick["analysis_state"] == "quick"
    assert quick["partial"] is True
    # The per-function flag is NOT the view state -- it stays False here, which is
    # why it could never carry this disclosure.
    assert quick["analysis_skipped"] is False


def test_comment_map_tolerates_dict_mutation_during_iteration_850():
    # #850: _comment_map iterated bv.address_comments.items() without first
    # snapshotting the dict; on a quick-loaded view whose analysis is still
    # settling, BN can add an entry while that walk is in flight, and a live view
    # then raises exactly what CPython's dict views raise:
    # ``RuntimeError: dictionary changed size during iteration``.
    #
    # The store below reproduces THAT mechanism -- the walk itself mutates the
    # dict being walked -- so the pre-fix code dies of the production error.
    # A fake that mutates *before* handing back the view proves nothing: the
    # pre-fix code then only trips over a synthetic type error, and the fixed
    # ``dict(...)`` never calls ``items()`` at all.
    il_format = importlib.import_module("bn_agent_bridge.il_format")

    class _LiveCommentStore(Mapping):
        """address_comments as the bridge sees it: a live map that analysis may
        add to at any moment, including mid-walk."""

        def __init__(self, entries: dict[int, str]) -> None:
            self._entries = dict(entries)
            self.injected = False

        def __getitem__(self, key: int) -> str:
            return self._entries[key]

        def __iter__(self):
            return iter(self._entries)

        def __len__(self) -> int:
            return len(self._entries)

        def items(self):
            for index, (address, text) in enumerate(self._entries.items()):
                if index == 0 and not self.injected:
                    self.injected = True
                    self._entries[0x2000] = "settled mid-walk"
                yield address, text

    class _FakeBV:
        def __init__(self, store) -> None:
            self.address_comments = store

    class _FakeBlock:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class _FakeFunc:
        basic_blocks = [_FakeBlock(0x1000, 0x1010)]

    store = _LiveCommentStore({0x1000: "a comment"})
    # Pre-fix this raises the live-view RuntimeError (red-first verified by
    # reverting the source); post-fix the map is materialised before the walk,
    # so the answer is the one entry that existed when the call started.
    assert il_format._comment_map(_FakeBV(store), _FakeFunc()) == {"0x1000": "a comment"}


def test_annotation_entries_tolerate_dict_mutation_during_iteration_861():
    # #861: the SAME live-collection shape #850 fixed in `_comment_map`, at the
    # two sites that fix did not reach. Here it is `_annotation_entries` (formerly
    # `_annotation_bodies`), which unpacked `func.comments` -- a view of BN's per-function map, which
    # the analysis threads can add to while the walk is in flight. The store
    # below mutates DURING that walk (the #850 store's mechanism, walked through
    # `values()` because this site unpacked the value view), so the pre-fix code
    # dies of the production error and the snapshot returns what was there when
    # the call started.
    read_decompile = importlib.import_module("bn_agent_bridge.read_decompile")

    class _LiveFunctionCommentStore(Mapping):
        """`func.comments` as the bridge sees it: a live map analysis may add to
        at any moment, including mid-walk."""

        def __init__(self, entries: dict[int, str]) -> None:
            self._entries = dict(entries)
            self.injected = False

        def __getitem__(self, key: int) -> str:
            return self._entries[key]

        def __iter__(self):
            return iter(self._entries)

        def __len__(self) -> int:
            return len(self._entries)

        def values(self):
            for index, text in enumerate(self._entries.values()):
                if index == 0 and not self.injected:
                    self.injected = True
                    self._entries[0x2000] = "settled mid-walk"
                yield text

    class _FakeFunc:
        comment = ""

        def __init__(self, store) -> None:
            self.comments = store

    store = _LiveFunctionCommentStore({0x1000: "a local note"})
    # Pre-fix: RuntimeError: dictionary changed size during iteration.
    entries = read_decompile._annotation_entries(_FakeFunc(store), {})
    assert [body for _key, body in entries] == ["a local note"]

# ---------------------------------------------------------------------------
# #675 item 5: return width inferred from the full register, disclosed
# ---------------------------------------------------------------------------


class _RWType:
    def __init__(self, width):
        self.width = width


class _RWParam:
    def __init__(self, width):
        self.type = _RWType(width)


class _RWFunc:
    def __init__(self, ret_width=8, params=(4, 4), has_user_type=False):
        self.return_type = _RWType(ret_width)
        self.parameter_vars = [_RWParam(w) for w in params]
        self.has_user_type = has_user_type


class _RWArch:
    def __init__(self, name):
        self.name = name


class _RWBV:
    def __init__(self, arch):
        self.arch = _RWArch(arch)


def _note(arch, **kw):
    from bn_agent_bridge import read_decompile

    return read_decompile._return_width_inference_note(_RWBV(arch), _RWFunc(**kw))


@pytest.mark.parametrize("arch,reg", [("x86_64", "eax"), ("aarch64", "w0")])
def test_return_width_note_names_the_mechanism_for_each_arch_in_the_set(arch, reg):
    """The note must explain the arch's OWN register relation, not x86's.

    Both members of the set share one mechanism -- a 32-bit write zero-extends
    the 64-bit return register -- but they do not share register names. A note
    that said `eax` on AArch64 would be a confident claim about a register the
    target does not have.
    """
    note = _note(arch)
    assert note is not None
    assert arch in note
    assert reg in note


def test_return_width_note_is_silent_on_an_arch_outside_the_set():
    """The must-not-fire twin: ARM32/thumb2 has no widening to disclose.

    Its registers are natively 32-bit, so the ambiguity this note explains
    does not exist there -- a measured cross-build finds ZERO over-wide
    functions. A gate that leaked past its scope would print an x86/AArch64
    register relation over ARM firmware, which is worse than printing nothing:
    silence is the correct output where there is no evidence.
    """
    for arch in ("thumb2", "armv7", "mipsel32", "riscv32"):
        assert _note(arch) is None, f"note leaked onto {arch}"


def test_return_width_note_defers_to_an_analyst_set_prototype():
    """A user-set type is not an inference, so there is nothing to disclose."""
    assert _note("x86_64", has_user_type=True) is None


def test_return_width_note_requires_the_ambiguous_shape():
    """Selectivity is the whole design: a loud note is skipped within a week.

    Only a pointer-width return with all-narrower declared parameters can be
    the zero-extension artefact. A genuinely 64-bit return (64-bit params), a
    return that is already 32-bit, and a parameterless function are all
    ordinary and must stay quiet.
    """
    assert _note("x86_64", params=(8, 8)) is None
    assert _note("x86_64", ret_width=4) is None
    assert _note("x86_64", params=()) is None


def test_return_width_note_warns_about_the_second_surface():
    """The harm the issue reported is at the CALLER, not the prototype.

    A caller returning -1 renders `return 0xffffffff;`, which is the reading
    that actually misleads, so the note has to reach past its own output.
    """
    note = _note("x86_64")
    assert "0xffffffff" in note


# ---------------------------------------------------------------------------
# #676 item 12: HLIL trust debt annotated inline, not only in the skill
# ---------------------------------------------------------------------------


class _GuardBlock:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _GuardFunc:
    def __init__(self, span=8):
        self.arch = "probe-arch"
        self.basic_blocks = [_GuardBlock(0, span)]


def _guard_note(monkeypatch, mnemonics, span=None):
    """Drive the detector over a synthetic 4-byte-per-instruction listing."""
    from bn_agent_bridge import il_format, read_decompile

    listing = list(mnemonics)
    monkeypatch.setattr(
        il_format, "_disasm_instruction",
        lambda bv, addr, arch=None: (
            (listing[addr // 4], 4) if addr // 4 < len(listing) else ("", 4)
        ),
    )
    func = _GuardFunc(span if span is not None else 4 * len(listing))
    note = read_decompile._flattened_guard_warning(object(), func)
    return None if note is None else note["detail"]


def test_a_branch_free_select_is_named_in_the_output(monkeypatch):
    """The skill already documents this lie; the note reaches the reader who
    did not read the skill, which is every reader at the moment they are wrong.

    The mnemonic is NAMED rather than described, because "this function has a
    flattened guard" sends the reader looking for something they cannot
    identify in the listing.
    """
    note = _guard_note(monkeypatch, ["ccmp x0, x1, #0, ge", "csel x0, x1, x2, lt"])
    assert note is not None
    assert "ccmp" in note and "csel" in note
    assert "bn disasm" in note


def test_the_note_is_silent_on_a_function_with_no_select(monkeypatch):
    """A note on every function is a note nobody reads.

    Selectivity is the feature: the whole point of annotating inline is that
    its PRESENCE means something, which requires its absence to mean something
    too.
    """
    assert _guard_note(monkeypatch, ["add x0, x1, x2", "ret"]) is None


def test_the_same_note_fires_across_architectures_without_an_arch_gate(monkeypatch):
    """Keyed on the MNEMONIC, not the target -- the instruction IS the evidence.

    This is why item 12 is cheap where the return-width note (#675 item 5)
    needed a measured arch set: `csel` in a listing means a selected value in
    THIS function whatever the target is, so there is no per-architecture fact
    that could be inherited wrongly onto a target that does not have it.
    """
    for mnemonic in ("csel x0, x1, x2, lt", "cmovg %eax, %edx", "ite ne"):
        assert _guard_note(monkeypatch, [mnemonic]) is not None, mnemonic


def test_the_note_claims_nothing_about_the_lies_it_cannot_detect(monkeypatch):
    """The skill documents three HLIL lies; only one has a mnemonic.

    A hoisted loop-invariant bound aliased to a moving pointer, and a dropped
    `<< 4` in a size accumulator, are SEMANTIC properties with nothing to key
    on. A note that mentioned them would imply this function had been checked
    for them, which is worse than no note: it converts an unexamined property
    into an apparently cleared one.
    """
    note = _guard_note(monkeypatch, ["csel x0, x1, x2, lt"])
    lowered = note.lower()
    assert "hoist" not in lowered
    assert "<<" not in note
    assert "accumulator" not in lowered


def test_each_mnemonic_is_named_once_however_often_it_appears(monkeypatch):
    """A guard that repeats a mnemonic forty times must not print it forty
    times -- a note long enough to scroll is a note that gets skipped.

    The mechanism is a MEMBERSHIP check while collecting, not the `sorted()`
    that renders the names: breaking the sort leaves this test green, which
    cost one wasted sabotage round to discover. Break `mnemonic not in seen`
    to see this fail.
    """
    note = _guard_note(monkeypatch, ["csel x0, x1, x2, lt"] * 40 + ["ccmp x0, x1, #0, ge"])
    assert note.count("csel") == 1
    assert note.count("ccmp") == 1


def test_an_unreadable_listing_does_not_sink_the_decompile(monkeypatch):
    """The note is ADDITIVE, so a listing it cannot walk must cost nothing.

    Decompilation succeeding is the caller's actual request; an annotation
    that could turn a working read into an error would be a worse trade than
    the annotation is worth.
    """
    from bn_agent_bridge import il_format, read_decompile

    def _boom(bv, addr, arch=None):
        raise RuntimeError("no listing here")

    monkeypatch.setattr(il_format, "_disasm_instruction", _boom)
    assert read_decompile._flattened_guard_warning(object(), _GuardFunc()) is None


# --- #676 item 5 / #675: the multi-identifier producer, driven directly -----
#
# Round 1 major: `_decompile_batch` had no direct test at all. Failure
# isolation and the requested/resolved counters were asserted only against a
# hand-authored mock reply in the CLI renderer test -- which proves the
# renderer reads those keys, and nothing at all about the producer that is
# supposed to emit them.


def _install_per_function_pseudo_c(monkeypatch, bridge, bodies):
    """Like `_install_fake_pseudo_c`, but dispatches on the FUNCTION.

    The shared helper asserts `fn is func` for one function, so it cannot
    stand up a batch: a batch that decompiled the same function twice would
    not exercise ordering or per-row identity at all.
    """
    class _FakeContents:
        def __init__(self, address, text):
            self.address, self._text = address, text

        def __str__(self):
            return self._text

    class _FakeLine:
        def __init__(self, address, text):
            self.contents = _FakeContents(address, text)

    class _FakeViewObject:
        def __init__(self, batches):
            self.batches = [[_FakeLine(a, t) for (a, t) in b] for b in batches]

    class _FakeCursor:
        def __init__(self, view_obj):
            self._batches, self._i = view_obj.batches, 0

        def seek_to_begin(self):
            self._i = 0

        @property
        def lines(self):
            return self._batches[self._i] if self._i < len(self._batches) else []

        def next(self):
            self._i += 1
            return self._i < len(self._batches)

    class _FakeLinearViewObject:
        @staticmethod
        def single_function_language_representation(fn, settings=None,
                                                    language="Pseudo C"):
            # The shared helper asserts this and the first cut of this fake
            # dropped it, which made a local copy MORE FORGIVING than the
            # helper it names as its model: a producer asking for a different
            # representation left both batch tests green.
            assert language == "Pseudo C"
            return _FakeViewObject(bodies[fn.name])

    fake_mod = types.ModuleType("binaryninja.lineardisassembly")
    fake_mod.LinearViewObject = _FakeLinearViewObject
    fake_mod.LinearViewCursor = _FakeCursor
    monkeypatch.setattr(bridge.bn, "lineardisassembly", fake_mod, raising=False)

    class _FakeDisassemblySettings:
        def set_option(self, option, state=True):
            return None

    class _FakeDisassemblyOption:
        ShowAddress = 0
        ShowTypeCasts = 10
        WaitForIL = 66
        DisableLineFormatting = 68

    monkeypatch.setattr(bridge.bn, "DisassemblySettings", _FakeDisassemblySettings,
                        raising=False)
    monkeypatch.setattr(bridge.bn, "DisassemblyOption", _FakeDisassemblyOption,
                        raising=False)


def _batch_bridge(monkeypatch):
    """A bridge whose view holds two real functions with distinct bodies."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    handle = _FakeFunction(0x401000, "handle")
    parse = _FakeFunction(0x402000, "parse_file")
    bv = _FakeBV(functions=[handle, parse])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv_, func: {})
    _install_per_function_pseudo_c(monkeypatch, bridge, {
        "handle": [[(0x401000, "int32_t handle(void)")], [(0x401004, "{ }")]],
        "parse_file": [[(0x402000, "int32_t parse_file(void)")],
                       [(0x402004, "{ return 1; }")]],
    })
    return bridge, instance


def test_decompile_batch_keeps_a_miss_in_place_and_counts_it_676(monkeypatch):
    """A typo in the third name must not discard the two that resolved.

    That is the entire reason to batch: the caller already paid the analysis
    cost for the functions that worked, and a shell `for` loop under `set -e`
    throws them away. So the miss becomes a FAILED ROW in the caller's own
    order, carrying its error text, and the counters state the shortfall.
    """
    bridge, instance = _batch_bridge(monkeypatch)

    result = instance._decompile_batch(
        "active", ["handle", "nosuchfn", "parse_file"])

    assert result["kind"] == "decompile_batch"
    assert [row["identifier"] for row in result["functions"]] == [
        "handle", "nosuchfn", "parse_file"], "a miss keeps its SLOT"
    assert [row["ok"] for row in result["functions"]] == [True, False, True]

    # The counters are the shortfall a caller reads, not a row count.
    assert result["requested"] == 3
    assert result["resolved"] == 2

    # The two that resolved carry their own bodies, so the rows are not
    # interchangeable placeholders.
    assert result["functions"][0]["decompiled"]["text"] == (
        "int32_t handle(void)\n{ }")
    assert result["functions"][2]["decompiled"]["text"] == (
        "int32_t parse_file(void)\n{ return 1; }")

    # The miss says WHY, per identifier, rather than collapsing into one
    # failure for the whole request.
    miss = result["functions"][1]
    assert "decompiled" not in miss
    assert "nosuchfn" in miss["error"]


def test_decompile_batch_that_resolves_everything_reports_no_shortfall_676(
        monkeypatch):
    """Must-not-fire twin: `resolved` must track the successes, not be a
    constant that happens to be right on the failure path."""
    bridge, instance = _batch_bridge(monkeypatch)

    result = instance._decompile_batch("active", ["handle", "parse_file"])

    assert result["requested"] == result["resolved"] == 2
    assert all(row["ok"] for row in result["functions"])
    assert all("error" not in row for row in result["functions"])
def test_the_two_mapped_address_guards_diverge_on_an_indeterminate_view_827():
    """#827 item 3 / #888: `_require_mapped_address` and `_address_is_mapped`
    are NOT interchangeable, and the difference is only visible on a view that
    cannot answer.

    This exists because the divergence was real and UNTESTED: substituting one
    policy for the other left 145 tests passing across the three touched
    modules, so the only thing standing between a future "these two are
    duplicates" cleanup and a behaviour flip was a comment. A comment is not a
    test.

    Both policies are pinned here, on the SAME object, so a merge in either
    direction goes red:
      * no `is_valid_offset` at all, and
      * an `is_valid_offset` that raises
    are the two shapes of "the view cannot answer".

    #888's stated observable is one layer up -- the read OP answers a clean
    zero rather than raising -- so `tag get` is driven over the same two views
    as well: a caller-side change that starts rejecting an indeterminate probe
    would otherwise break the #374 contract with the helper-level pins still
    green.
    """
    import importlib

    _shared = importlib.import_module("bn_agent_bridge._shared")
    read_decompile = importlib.import_module("bn_agent_bridge.read_decompile")
    read_tags = importlib.import_module("bn_agent_bridge.read_tags")

    class _NoAnswer:
        """A view with no `is_valid_offset`; its 1-byte read also fails."""
        def read(self, addr, length):
            raise RuntimeError("cannot read")

        def get_functions_containing(self, addr):
            return []

        def get_tags_at(self, addr, auto=False):
            return []

    class _RaisingAnswer(_NoAnswer):
        def is_valid_offset(self, addr):
            raise RuntimeError("core error")

    class _Ctx:
        def __init__(self, bv):
            self._bv = bv

        def _resolve_view(self, selector):
            return self._bv

    for bv in (_NoAnswer(), _RaisingAnswer()):
        label = type(bv).__name__
        # PERMISSIVE: returns quietly, so its caller proceeds. #374 depends on
        # this -- a mapped address with zero refs must stay a clean exit 0, and
        # rejecting on "cannot tell" would turn that into an error.
        assert _shared._require_mapped_address(bv, 0x1000) is None, label
        # STRICT: answers False, i.e. treats "cannot tell" as "not mapped".
        assert read_decompile._address_is_mapped(bv, 0x1000) is False, label
        # The op the permissive policy exists for: a clean zero, not a raise.
        tags = read_tags._get_tags(_Ctx(bv), None, "0x1000", None)
        assert tags["address"] == "0x1000", label
        assert tags["tags"] == [], label
        assert tags["count"] == 0, label


def test_the_arm_predicates_disagree_on_every_bn_platform_name_827():
    """#827 item 3: the other pair deliberately left un-merged.

    `_is_classic_arm_or_thumb_arch` matches by PREFIX; the seam's
    `_supports_thumb_pointer_tags` matches by SUBSTRING. Measured against the
    installed BN API, they agree on every ARM *arch* name and disagree on every
    ARM *platform* name, because BN spells platforms `<os>-<arch>`. The names
    are pinned as literals so this test needs no BN: it is guarding the RULE,
    not BN's catalogue.

    The seam side calls the REAL predicate through a minimal view. The first
    cut re-declared the substring rule as a local closure, and a test that
    asserts a COPY of the rule is green whatever the rule does: rewriting the
    seam's last line to `joined.startswith(...)` -- the exact merge this pair
    exists to forbid, and the one #600 already paid for once -- left it
    passing. The whole point of the test is that that mutation goes red.
    """
    import importlib

    read_decompile = importlib.import_module("bn_agent_bridge.read_decompile")
    seam = importlib.import_module("bn_agent_bridge.seam")
    prefix = read_decompile._is_classic_arm_or_thumb_arch
    ctx = seam.BridgeContext(None)

    class _Named:
        """An arch/platform object shaped the way the seam reads one: it takes
        `.name`, and also `str()`, which BN's objects answer with the name."""

        def __init__(self, name: str) -> None:
            self.name = name

        def __str__(self) -> str:
            return self.name

    class _View:
        """The minimum the seam predicate reads. `arch` is absent on purpose:
        the seam joins arch AND platform names, so an ARM arch name sitting
        beside the platform name would satisfy either rule and hide the very
        divergence under test."""

        address_size = 4
        arch = None

        def __init__(self, name: str) -> None:
            self.platform = _Named(name)

    def substring(name: str) -> bool:
        return ctx._supports_thumb_pointer_tags(_View(name))

    # Arch names: the two rules agree, which is why the divergence is invisible
    # on an ordinary ARM binary (the seam joins arch AND platform, and the arch
    # name alone already matches under either rule).
    for arch in ("armv7", "armv7eb", "thumb2", "thumb2eb"):
        assert prefix(arch) is substring(arch) is True, arch

    # Platform names: every one is `<os>-<arch>`, so NONE starts with
    # arm/thumb and the two rules disagree on all of them.
    for platform in ("linux-armv7", "linux-armv7eb", "linux-thumb2",
                     "ios-armv7", "ios-kernel-thumb2", "mac-armv7",
                     "windows-thumb2", "efi-armv7", "freebsd-thumb2"):
        assert prefix(platform) is False, platform
        assert substring(platform) is True, platform

    # The #600 exclusion is the part they DO share, in both spellings.
    for arm64 in ("aarch64", "linux-aarch64", "arm64", "mac-arm64"):
        assert prefix(arm64) is substring(arm64) is False, arm64

    # The seam's 4-byte gate is part of the rule it is being pinned against: a
    # 64-bit view is never Thumb-tagged whatever its names say, so a future
    # "just use the arch predicate" merge cannot claim equivalence here either.
    class _WideView(_View):
        address_size = 8

    assert ctx._supports_thumb_pointer_tags(_WideView("linux-armv7")) is False


def test_variadic_text_marks_the_count_with_the_stated_confidence_886():
    """#827 item 6 / #886: the variadic count's confidence must reach TEXT, and
    must read as a claim about the COUNT.

    `read_evidence` stamps `confidence: heuristic` / `provenance:
    abi-format-heuristic` on the diagnostic, and the text renderer printed
    `expected >= N argument(s)` in the same voice as the recovered facts beside
    it -- JSON-only disclosure, the #883 shape. #886 asks specifically that the
    marker read as "this count is a heuristic" rather than as a hedge on the
    whole finding, which is why it is spelled `count: <confidence>` and sits
    directly after the verdict token it qualifies.

    Pinned here, because the marker shipped with NONE of it -- reverting the
    render site left every selected variadic test green, so deleting the
    marker was invisible to CI:
      * the EXACT line, so neither the `count:` spelling nor the marker's
        placement can drift away from the wording #886 asked for;
      * a FIRM word renders too, and is never omitted: `count: authoritative`
        states the count is authoritative, which is not a hedge. An earlier
        cut omitted it, which made a payload that calls the count firm render
        byte-identically to one that said nothing about it;
      * a payload that stated NOTHING renders no marker, and must not collide
        with either of the above;
      * a present-but-UNREADABLE confidence is disclosed by the render
        boundary rather than dropped, which is the whole reason the field is
        read through `_text_value` and not an inline isinstance (#619);
      * a whitespace-only word is `_text_value`'s PRESENT-AND-EMPTY case --
        no word was stated -- so it renders no marker rather than an empty
        `[count:    ]`.
    """
    from bn.formatters import _render_function_evidence_text

    def render(variadic: dict) -> str:
        return _render_function_evidence_text(
            {"calls": [{"address": "0x1000",
                        "variadic": {"is_variadic": True, **variadic}}]}
        )

    def variadic_line(variadic: dict) -> str:
        return [ln for ln in render(variadic).splitlines()
                if ln.startswith("  variadic: ")][0]

    warning = ("imported variadic call `f` under-recovered in HLIL: "
               "recovered 1 of an expected >= 3 argument(s)")
    under = {"under_recovered": True, "warning": warning}
    fmt = {"callee": "f", "format_string": "%s%d", "format_conversions": 2}

    # The exact line, both branches: `count:` names the subject, and the marker
    # sits on the verdict/callee token, not trailing the sentence after it.
    assert variadic_line({**under, "confidence": "heuristic"}) == (
        f"  variadic: UNDER-RECOVERED [count: heuristic] — {warning}")
    assert variadic_line({**fmt, "confidence": "heuristic"}) == (
        "  variadic: f [count: heuristic] format='%s%d' conversions=2")

    # A firm word is rendered, not omitted -- and the three states stay
    # distinguishable, which omitting the firm one destroyed.
    for payload in (under, fmt):
        firm = variadic_line({**payload, "confidence": "authoritative"})
        assert "[count: authoritative]" in firm, firm
        silent = variadic_line(payload)
        assert "[count:" not in silent, silent
        assert firm != silent != variadic_line({**payload, "confidence": "heuristic"})

    # An unrecognised word is the payload's to state; the renderer relays it.
    assert "[count: medium]" in variadic_line({**under, "confidence": "medium"})

    # Present but unreadable: no marker, but the boundary says the field was
    # there and could not be used.
    skewed = render({**under, "confidence": {"level": "heuristic"}})
    assert "[count:" not in skewed and "malformed confidence field" in skewed, skewed

    # Whitespace-only states no word: no empty marker, and no skew claimed.
    blank = render({**under, "confidence": "   "})
    assert "[count:" not in blank and "malformed" not in blank, blank

# --- #797: `dataflow defuse` discloses a dropped call argument --------------
# The defuse half of #489. BN clamps a direct call's MLIL parameters to the
# callee's recovered arity, so a variadic callee auto-typed fixed-arity leaves
# its STACK-passed arguments behind as `[sp+N].d = <var>` stores that nothing in
# the function reads back -- and `uses` then lists the argument set-up for the
# defused variable as if it were an ordinary use. `trace` already discloses that
# state (#489); defuse now states the same thing, over the same calls, from the
# same helper, so the two ops cannot disagree.

def _d797_op(name):
    return types.SimpleNamespace(operation=types.SimpleNamespace(name=name))


def _d797_reg(name):
    expr = _d797_op("LLIL_REG")
    expr.src = types.SimpleNamespace(name=name)
    return expr


def _d797_const(value):
    expr = _d797_op("LLIL_CONST")
    expr.constant = value
    return expr


def _d797_add(left, right):
    expr = _d797_op("LLIL_ADD")
    expr.left, expr.right = left, right
    return expr


def _d797_store(dest, address):
    expr = _d797_op("LLIL_STORE")
    expr.dest, expr.address = dest, address
    return expr


def _d797_call(address):
    expr = _d797_op("LLIL_CALL")
    expr.address = address
    return expr


def _d797_ins(name, address, *, params=None, instr_index=0):
    return types.SimpleNamespace(
        address=address, instr_index=instr_index,
        operation=types.SimpleNamespace(name=name),
        params=list(params or []), src=None, vars_read=[], vars_written=[],
    )


def _d797_thunk_resolver(thunk: str, real: str):
    """A `resolve_call_target` that actually HONORS `follow_thunks`, so the two
    resolutions differ the way they do on a real PLT stub / GCC veneer: the
    unfollowed answer is the thunk's own name, the followed one is the real
    callee."""
    def resolve(bv_, ins, follow_thunks=False):
        return types.SimpleNamespace(
            address=0x402100,
            function=types.SimpleNamespace(name=real if follow_thunks else thunk))
    return resolve


def _defuse_under_recovered_call(monkeypatch, *, resolve=None, use_at="before",
                                 defined=False):
    """`_defuse` over a function whose call at 0x401030 was recovered with ONE
    arg (the caller's format string) while the LLIL hands it two outgoing
    stack-arg stores -- the #489 shape, standing in for an auto-typed variadic.

    ``use_at`` places the defused variable's single use: ``"before"`` (the
    default) makes it the argument set-up store feeding that call, ``"after"``
    moves it past the call, and ``"none"`` gives the variable no uses at all.

    ``defined`` gives the variable a DEFINITION ahead of the call, which the
    real API supplies whenever the value is computed in this function.
    Hardcoding it absent meant every case ran with an empty row set, so the
    scoping the tests below assert was only ever measured on its
    nothing-to-scope short-circuit (#797 round-4 review)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, func = _mid_function_bv()
    bv._memory[0x402000] = b"%d %d\x00"
    func.arch = types.SimpleNamespace(stack_pointer="sp", address_size=4)
    func.calling_convention = types.SimpleNamespace(int_arg_regs=["r0", "r1", "r2", "r3"])
    call_addr = 0x401030
    func.low_level_il = [[
        _d797_store(_d797_reg("sp"), 0x401020),
        _d797_store(_d797_add(_d797_reg("sp"), _d797_const(4)), 0x401024),
        _d797_call(call_addr),
    ]]
    fmt = _d797_op("MLIL_CONST_PTR")
    fmt.constant = 0x402000
    call = _d797_ins("MLIL_CALL_SSA", call_addr, params=[fmt], instr_index=2)
    store = _d797_ins("MLIL_STORE_SSA", 0x401024, instr_index=1)
    trailing = _d797_ins("MLIL_STORE_SSA", 0x401040, instr_index=3)
    # The variable's own definition, AHEAD of the call the way a value computed
    # in this function is: `arg1#0 = ...` at 0x401010.
    defn = _d797_ins("MLIL_SET_VAR_SSA", 0x401010, instr_index=0)
    uses = {"before": [store], "after": [trailing], "none": []}[use_at]
    il = types.SimpleNamespace(
        instructions=([defn] if defined else []) + [store, call, trailing],
        get_ssa_var_definition=lambda v: defn if defined else None,
        get_ssa_var_uses=lambda v: list(uses),
    )
    monkeypatch.setattr(bridge.il_format, "_il_function_for", lambda fn, view, ssa: il)
    ssa_var = types.SimpleNamespace(var=types.SimpleNamespace(name="arg1", type="int"), version=0)
    monkeypatch.setattr(bridge.il_format, "_resolve_ssa_variable",
                        lambda f, i, sel: (ssa_var, []))
    monkeypatch.setattr(bridge.il_format, "_ssa_var_entry", lambda v: {"ssa": "arg1#0"})
    # The loaded bridge package is its own module COPY (`bn_test_bridge.*`), so
    # the patch has to land on the alias `_defuse` resolves through, not on the
    # same-named module this file imports for direct helper tests.
    monkeypatch.setattr(
        bridge.read_decompile._taint, "resolve_call_target",
        resolve or (lambda bv_, ins, follow_thunks=False: types.SimpleNamespace(
            address=0x402100,
            function=types.SimpleNamespace(name="my_logger"))))
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    return instance, il


def test_defuse_discloses_a_dropped_stack_arg_797(monkeypatch):
    """#797: the result names the call whose model dropped the stack args.

    Before this, the stack-arg store appeared in `uses` with nothing saying it
    feeds a call whose recovered parameters do not include it -- so a def-use
    answer was read as the complete set of things that touch the variable.
    """
    instance, _il = _defuse_under_recovered_call(monkeypatch)

    result = instance._defuse("active", "0x401000", "arg1#0")

    hints = result["hints"]
    assert len(hints) == 1, hints
    hint = hints[0]
    assert "call 0x401030" in hint
    assert "call-model truncation" in hint
    assert "my_logger" in hint
    assert "sp+0x0" in hint and "sp+0x4" in hint     # the dropped stores, named
    assert "proto set my_logger" in hint             # the runnable remedy
    # The use itself is unchanged: the disclosure is additive, so an existing
    # consumer of `uses` reads exactly what it read before.
    assert [u["address"] for u in result["uses"]] == ["0x401024"]


def test_defuse_stays_quiet_when_the_call_model_is_complete_797(monkeypatch):
    """The no-false-positive direction, inherited from the #489 gate: a call
    whose recovered args are NOT (all) register-passed -- here five args on a
    4-register convention -- is not a truncated model, so nothing is disclosed
    and the caller sees no invented caveat."""
    instance, il = _defuse_under_recovered_call(monkeypatch)
    il.instructions[1].params = il.instructions[1].params * 5

    result = instance._defuse("active", "0x401000", "arg1#0")

    assert result["hints"] == []


def test_defuse_resolves_the_callee_through_thunks_like_its_sibling_797(monkeypatch):
    """#797 review: `defuse` resolved the callee WITHOUT following thunks while
    `trace` follows them, and the shared note keys its two safety gates on that
    name -- so the claim that "the two ops cannot disagree" was false.

    The gate that matters here is the known-fixed-arity denylist: a PLT stub /
    GCC veneer in front of `strlen` resolves unfollowed to `j_strlen`, which is
    in no denylist, so the shared note fired on a call `trace` is silent about.
    That is exactly the residual false positive the #489 denylist was added to
    close, re-opened on the second op.
    """
    instance, _il = _defuse_under_recovered_call(
        monkeypatch, resolve=_d797_thunk_resolver("j_strlen", "strlen"))

    result = instance._defuse("active", "0x401000", "arg1#0")

    assert result["hints"] == [], result["hints"]


def test_defuse_names_the_real_callee_in_the_truncation_remedy_797(monkeypatch):
    """The other half of the same disagreement: when the note DOES fire through
    a thunk, its `proto set` remedy must name the callee whose prototype is
    actually wrong. `bn proto set j_my_logger ...` retypes the veneer and leaves
    the model truncated, so a copy-pasted remedy silently does nothing."""
    instance, _il = _defuse_under_recovered_call(
        monkeypatch, resolve=_d797_thunk_resolver("j_my_logger", "my_logger"))

    result = instance._defuse("active", "0x401000", "arg1#0")

    assert len(result["hints"]) == 1, result["hints"]
    hint = result["hints"][0]
    assert "proto set my_logger" in hint
    assert "j_my_logger" not in hint


def test_defuse_discloses_only_the_calls_the_variable_feeds_797(monkeypatch):
    """#797 review: the disclosure is about a USE, so it has to be scoped to one.

    The hint list was built over every call in the FUNCTION, which is not what
    the reference, `--help` and the renderer all say it is: a def-use of a
    variable with no uses at all still printed "call 0x...: call-model
    truncation" immediately above `uses (0):`, telling a reader that an empty
    listing is incomplete because of a call the variable never touches. A use
    that lands AFTER the call is the same error one step subtler -- it is not
    argument set-up for it, so nothing about that call explains it.

    Round-4 review: (a) and (b) were both green for the wrong reason -- the
    fixture hardcoded the variable as having NO definition, so every case ran
    with an empty row set and only the nothing-to-scope short-circuit was
    measured. Give the variable the definition the real API supplies and the
    scope re-opened: a definition ahead of the call made the call "fed", so
    the hint came back above `uses (0):`. A DEFINITION is not evidence that
    the value reaches a call -- it is where the value is produced -- so the
    scope is the variable's USES, and (c)/(d) are (a)/(b) with the definition
    present.
    """
    # (a) No uses at all: there is no listing for a disclosure to qualify.
    instance, _il = _defuse_under_recovered_call(monkeypatch, use_at="none")
    empty = instance._defuse("active", "0x401000", "arg1#0")
    assert empty["uses"] == []
    assert empty["hints"] == [], empty["hints"]

    # (b) The variable's only use sits past the call, so it is not part of the
    # outgoing-argument run the dropped stack stores belong to.
    instance, _il = _defuse_under_recovered_call(monkeypatch, use_at="after")
    downstream = instance._defuse("active", "0x401000", "arg1#0")
    assert [u["address"] for u in downstream["uses"]] == ["0x401040"]
    assert downstream["hints"] == [], downstream["hints"]

    # (c) The same empty listing for a variable that HAS a definition ahead of
    # the call -- the ordinary case, and the one the fixture never built.
    instance, _il = _defuse_under_recovered_call(monkeypatch, use_at="none",
                                                 defined=True)
    defined_empty = instance._defuse("active", "0x401000", "arg1#0")
    assert defined_empty["definition"]["address"] == "0x401010"
    assert defined_empty["uses"] == []
    assert defined_empty["hints"] == [], defined_empty["hints"]

    # (d) ...and the downstream use, with that definition present: the call
    # still explains no row in this listing.
    instance, _il = _defuse_under_recovered_call(monkeypatch, use_at="after",
                                                 defined=True)
    defined_downstream = instance._defuse("active", "0x401000", "arg1#0")
    assert defined_downstream["definition"]["address"] == "0x401010"
    assert [u["address"] for u in defined_downstream["uses"]] == ["0x401040"]
    assert defined_downstream["hints"] == [], defined_downstream["hints"]

    # ...while the use that IS the call's argument set-up still gets it, so the
    # scope narrowed to the right thing rather than to nothing.
    instance, _il = _defuse_under_recovered_call(monkeypatch, use_at="before",
                                                 defined=True)
    feeding = instance._defuse("active", "0x401000", "arg1#0")
    assert len(feeding["hints"]) == 1, feeding["hints"]
    assert "call 0x401030" in feeding["hints"][0]


# ---------------------------------------------------------------------------
# #955 call arity, #966 split bodies, #970 raw string pointers, #977 session
# annotations. Small SimpleNamespace IL fakes: these detectors read only the
# LLIL/MLIL fields named below.
# ---------------------------------------------------------------------------


def _op(name, **fields):
    return types.SimpleNamespace(operation=types.SimpleNamespace(name=name),
                                 operands=list(fields.values()), **fields)


def _reg(name):
    return _op("LLIL_REG", src=types.SimpleNamespace(name=name))


def _const(value):
    return _op("LLIL_CONST", constant=value)


def _set_reg(name, src):
    return _op("LLIL_SET_REG", dest=types.SimpleNamespace(name=name), src=src)


class _Llil(list):
    """One LLIL basic block holding every instruction, in order."""

    @property
    def instructions(self):
        return list(self)

    @property
    def basic_blocks(self):
        return [list(self)]


def _aarch64_arch():
    regs = {f"x{i}": types.SimpleNamespace(full_width_reg=f"x{i}", size=8) for i in range(31)}
    regs.update({f"w{i}": types.SimpleNamespace(full_width_reg=f"x{i}", size=4)
                 for i in range(31)})
    regs["fp"] = types.SimpleNamespace(full_width_reg="fp", size=8)
    regs["lr"] = types.SimpleNamespace(full_width_reg="lr", size=8)
    regs["sp"] = types.SimpleNamespace(full_width_reg="sp", size=8)
    return types.SimpleNamespace(regs=regs, link_reg="lr", name="aarch64")


_ARG_REGS = [f"x{i}" for i in range(8)]


def _arity_case(monkeypatch, setup, *, rendered, raw_name, kind="ImportedFunctionSymbol",
                signature=None, call_op="LLIL_CALL", calls=1):
    """*calls* identical calls at 0x401040.. to an import at 0x400e00, the first
    preceded in its block by *setup* LLIL instructions. *signature* is the
    demangled (params, variadic)."""
    llil = _Llil(setup)
    for n in range(calls):
        call = _op(call_op, dest=_op("LLIL_CONST_PTR", constant=0x400e00))
        call.address = 0x401040 + 8 * n
        call.mlil = types.SimpleNamespace(params=[object()] * rendered)
        llil.append(call)
    func = types.SimpleNamespace(
        start=0x401000, llil=llil,
        calling_convention=types.SimpleNamespace(int_arg_regs=_ARG_REGS),
    )
    symbol = types.SimpleNamespace(type=types.SimpleNamespace(name=kind),
                                   raw_name=raw_name,
                                   short_name=raw_name.removeprefix("_Z") + "(...)")
    bv = types.SimpleNamespace(
        arch=_aarch64_arch(),
        get_symbol_at=lambda a: symbol if a == 0x400e00 else None,
        get_function_at=lambda a: types.SimpleNamespace(name=raw_name, is_thunk=True,
                                                        start=a),
    )

    def demangle(arch, name):
        if signature is None:
            raise ValueError("not mangled")
        params, variadic = signature
        return (types.SimpleNamespace(
            parameters=[types.SimpleNamespace(name=p) for p in params],
            has_variable_arguments=variadic), [name])

    monkeypatch.setattr(read_decompile, "bn", types.SimpleNamespace(demangle_gnu3=demangle))
    return bv, func


def test_call_arity_names_a_call_rendering_fewer_arguments_than_its_signature_955(monkeypatch):
    # f.open(path, mode): `this`, the path and the mode are all set up, but the
    # type library's prototype for the import declares one parameter.
    bv, func = _arity_case(
        monkeypatch,
        [_set_reg("w2", _const(0x11)), _set_reg("x1", _reg("x23")),
         _set_reg("x0", _reg("sp"))],
        rendered=1, raw_name="_ZN4demo7Archive4openEPKci",
        signature=(["this", "", ""], False),
    )
    warning = read_decompile._call_arity_warning(bv, func)["detail"]
    assert warning is not None
    assert "0x401040" in warning and "renders 1 of 3 arguments" in warning
    assert "x0, x1, x2" in warning
    assert "--within 0x401000" in warning


def test_call_arity_does_not_count_the_stack_protector_scrub_955(monkeypatch):
    # GCC loads the canary into x1, stores it, then zeroes x1: that write is not
    # an argument. An untyped import with one rendered argument stays silent
    # instead of reporting the invented extra argument the issue saw.
    store = _op("LLIL_STORE", dest=_reg("sp"), src=_reg("x1"))
    bv, func = _arity_case(
        monkeypatch,
        [_set_reg("x1", _op("LLIL_LOAD", src=_reg("x0"))), store,
         _set_reg("x1", _const(0)), _set_reg("x0", _reg("sp"))],
        rendered=1, raw_name="widget_init",
    )
    assert read_decompile._call_arity_warning(bv, func) is None
    # ...whereas the same zero WITHOUT the store before it is a real NULL argument.
    bv, func = _arity_case(
        monkeypatch, [_set_reg("x1", _const(0)), _set_reg("x0", _reg("sp"))],
        rendered=1, raw_name="widget_init",
    )
    assert "renders 1 of 2 arguments" in read_decompile._call_arity_warning(bv, func)["detail"]


def test_call_arity_scrub_lookback_stops_at_the_previous_call_955(monkeypatch):
    # The canary store sits before an EARLIER call in the same LLIL block (LLIL
    # blocks do not end at calls). It cannot explain the `x1 = 0` set up for the
    # later call: that is a real NULL argument (`basic_ios::init(nullptr)`).
    earlier = _op("LLIL_CALL", dest=_op("LLIL_CONST_PTR", constant=0x400f00))
    earlier.address = 0x401020
    earlier.mlil = None
    bv, func = _arity_case(
        monkeypatch,
        [_set_reg("x1", _op("LLIL_LOAD", src=_reg("x0"))),
         _op("LLIL_STORE", dest=_reg("sp"), src=_reg("x1")),
         _set_reg("x1", _const(0)), _set_reg("x0", _reg("sp")), earlier,
         _set_reg("x1", _const(0)), _set_reg("x0", _reg("sp"))],
        rendered=1, raw_name="widget_init",
    )
    assert "renders 1 of 2 arguments" in read_decompile._call_arity_warning(bv, func)["detail"]


def test_call_arity_skips_sized_operator_delete_955(monkeypatch):
    # BN's type library declares `operator delete(void*)`; the size every sized
    # delete also passes is allocator bookkeeping, not data, and it flooded the
    # warning on ordinary C++.
    bv, func = _arity_case(
        monkeypatch, [_set_reg("x1", _const(0x18)), _set_reg("x0", _reg("x19"))],
        rendered=1, raw_name="_ZdlPvm", signature=(["", ""], False),
    )
    assert read_decompile._call_arity_warning(bv, func) is None


def test_call_arity_groups_repeated_calls_and_sees_stack_adjusting_calls_955(monkeypatch):
    bv, func = _arity_case(
        monkeypatch, [], rendered=1, raw_name="_ZN4demo7Archive4openEPKci",
        signature=(["this", "", ""], False), call_op="LLIL_CALL_STACK_ADJUST", calls=4,
    )
    warning = read_decompile._call_arity_warning(bv, func)["detail"]
    assert warning.startswith("call arity: 4 call(s)")
    assert "(4 calls at 0x401040, 0x401048, 0x401050 and 1 more) renders 1 of 3" in warning
    assert warning.count("renders 1 of 3") == 1


def test_call_arity_ignores_a_register_read_before_the_call_955(monkeypatch):
    # x1 is a temporary consumed by the x0 computation, not an argument.
    bv, func = _arity_case(
        monkeypatch,
        [_set_reg("x1", _reg("x20")), _set_reg("x0", _op("LLIL_ADD", left=_reg("x1"),
                                                          right=_const(8)))],
        rendered=1, raw_name="widget_free",
    )
    assert read_decompile._call_arity_warning(bv, func) is None


def test_call_arity_trusts_the_signature_over_a_static_members_assumed_this_955(monkeypatch):
    # BN's demangler adds `this` to every nested name, static members included.
    # One explicit parameter, rendered once, nothing else set up: silent.
    bv, func = _arity_case(
        monkeypatch, [_set_reg("x0", _const(5))],
        rendered=1, raw_name="_ZN4demo5Clock5ticksEi",
        signature=(["this", ""], False),
    )
    assert read_decompile._call_arity_warning(bv, func) is None


def test_call_arity_reports_an_argument_the_signature_does_not_take_955(monkeypatch):
    bv, func = _arity_case(
        monkeypatch, [], rendered=3, raw_name="_ZN4demo6Widget5resetEv",
        signature=(["this"], False),
    )
    assert "renders 3 arguments but the demangled signature takes only 1" in \
        read_decompile._call_arity_warning(bv, func)["detail"]
    bv, func = _arity_case(
        monkeypatch, [], rendered=3, raw_name="_Z6logfmtPKcz",
        signature=([""], True),
    )
    assert read_decompile._call_arity_warning(bv, func) is None


def _param_var(name, reg, width=None):
    return types.SimpleNamespace(
        name=name, storage=reg, type=types.SimpleNamespace(width=width),
        source_type=types.SimpleNamespace(name="RegisterVariableSourceType"))


def _ref(owner_op, start):
    """A code reference from a function at *start* whose instruction there lifts
    as *owner_op*."""
    owner = types.SimpleNamespace(start=start, get_llil_at=lambda a: _op(owner_op))
    return types.SimpleNamespace(function=owner, address=start + 0x10)


def _split_view(*, tail_mnemonics, tail_returns=True, tail_refs=None, tail_data_refs=()):
    """build_widget @ 0x4000dc ends at 0x4000f8, where sub_4000f8 begins. Its
    last instructions are *tail_mnemonics* (4 bytes each). By default the tail
    returns and reset_widget tail-jumps into it -- an ordinary shared tail."""
    arch = _aarch64_arch()
    arch.get_reg_name = lambda storage: storage
    arch.max_instr_length = 4
    cc = types.SimpleNamespace(int_arg_regs=_ARG_REGS, float_arg_regs=[],
                               callee_saved_regs=[f"x{i}" for i in range(19, 29)] + ["fp"])
    tail = types.SimpleNamespace(start=0x4000f8, name="sub_4000f8",
                                 symbol=None, calling_convention=cc,
                                 parameter_vars=[_param_var("arg1", "x19")],
                                 type=types.SimpleNamespace(parameters=[]),
                                 can_return=tail_returns, basic_blocks=[])
    first = 0x4000f8 - 4 * len(tail_mnemonics)
    mnemonics = {first + 4 * i: m for i, m in enumerate(tail_mnemonics)}
    block = types.SimpleNamespace(start=first, end=0x4000f8)
    head = types.SimpleNamespace(
        start=0x4000dc, name="build_widget", symbol=None, calling_convention=cc,
        parameter_vars=[_param_var("arg1", "x0"), _param_var("arg3", "x19"),
                        _param_var("arg5", "fp")],
        type=types.SimpleNamespace(parameters=[]), can_return=True,
        basic_blocks=[block],
        get_llil_at=lambda a: _op("LLIL_CALL" if mnemonics.get(a) == "blr" else "LLIL_STORE"),
    )
    block.function = head
    branches = {"ret": "FunctionReturn", "b": "UnconditionalBranch"}

    def info(data, address):
        kind = branches.get(mnemonics.get(address))
        return types.SimpleNamespace(branches=[types.SimpleNamespace(
            type=types.SimpleNamespace(name=kind), target=0)] if kind else [])

    refs = {
        0x4000f8: [_ref("LLIL_TAILCALL", 0x400108)] if tail_refs is None else tail_refs,
        0x4000dc: [_ref("LLIL_CALL", 0x400120)],
    }
    arch.get_instruction_info = info
    bv = types.SimpleNamespace(
        arch=arch,
        read=lambda a, n: b"\0" * n,
        get_instruction_length=lambda a: 4,
        get_disassembly=lambda a: f"{mnemonics.get(a, 'mov')}  x0, x1",
        get_function_at=lambda a: {0x4000f8: tail, 0x4000dc: head}.get(a),
        get_basic_blocks_at=lambda a: [block] if first <= a < 0x4000f8 else [],
        get_code_refs=lambda a: refs.get(a, []),
        get_data_refs=lambda a: list(tail_data_refs) if a == 0x4000f8 else [],
    )
    return bv, head, tail


def test_split_body_notes_name_the_fall_through_and_saved_register_params_966():
    bv, head, tail = _split_view(tail_mnemonics=["str"])
    head_notes = [n["detail"] for n in read_decompile._split_body_notes(bv, head)]
    assert any("the block ending at 0x4000f4 runs straight into sub_4000f8 @ 0x4000f8" in n
               for n in head_notes)
    assert any("(arg3 @ x19, arg5 @ fp)" in n for n in head_notes)
    # The argument register x0 is never reported.
    assert not any("arg1 @ x0" in n for n in head_notes)
    tail_notes = [n["detail"] for n in read_decompile._split_body_notes(bv, tail)]
    assert any("build_widget @ 0x4000dc runs straight into this function's entry" in n
               for n in tail_notes)


@pytest.mark.parametrize("tail", [["ret"], ["b"], ["blr", "nop", "nop"]])
def test_split_body_notes_are_silent_on_an_ordinary_function_end_966(tail):
    # A return, a branch away, or a (noreturn) call followed only by alignment
    # padding ends the function: the next function's entry is not reached.
    bv, head, _ = _split_view(tail_mnemonics=tail)
    head.parameter_vars = []
    assert read_decompile._split_body_notes(bv, head) == []
    assert read_decompile._fallthrough_successor(bv, head) is None


def test_disasm_discloses_where_execution_continues_966(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, head, _ = _split_view(tail_mnemonics=["str"])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.ctx, "_find_function", lambda *a, **k: head)
    monkeypatch.setattr(bridge.il_format, "_disasm_text", lambda *a, **k: "0x4000f4  str")
    result = instance._disasm("active", "build_widget")
    assert result["continues_into"]["address"] == "0x4000f8"
    assert result["continues_into"]["from"] == "0x4000f4"
    assert "bn disasm 0x4000f8" in result["continues_into"]["note"]


@pytest.mark.parametrize("tail_refs,tail_returns", [
    # A landing pad: nothing references it (the unwinder enters it via the LSDA).
    ([], True),
    # Cleanup reached only by a jump from its parent, and it never returns.
    ([_ref("LLIL_GOTO", 0x400200)], False),
])
def test_eh_fragment_gets_one_note_and_no_proto_set_advice_966(tail_refs, tail_returns):
    # The fragment's x19 is its parent's live register: dropping it with
    # `proto set` would break the frame accesses (#985 review).
    bv, head, tail = _split_view(tail_mnemonics=["str"], tail_returns=tail_returns,
                                 tail_refs=tail_refs)
    notes = [n["detail"] for n in read_decompile._split_body_notes(bv, tail)]
    assert len(notes) == 1 and notes[0].startswith("EH/cold fragment:")
    assert "(arg1 @ x19)" in notes[0] and "proto set" not in notes[0]
    # Its parent running into it is not a shared-tail split either.
    assert read_decompile._fallthrough_successor(bv, head) is None
    assert not any("runs straight into" in n["detail"] for n in read_decompile._split_body_notes(bv, head))


def test_callee_note_skips_an_eh_fragment_but_not_a_shared_tail_966():
    bv, _head, tail = _split_view(tail_mnemonics=["ret"])
    call = _op("LLIL_CALL", dest=_op("LLIL_CONST_PTR", constant=0x4000f8))
    caller = types.SimpleNamespace(start=0x400300, llil=_Llil([call]))
    assert len(read_decompile._callee_split_notes(bv, caller)) == 1
    tail.type = types.SimpleNamespace(parameters=[types.SimpleNamespace(
        type="struct _Unwind_Exception*")])
    assert read_decompile._callee_split_notes(bv, caller) == []


def test_jump_only_tail_gets_no_proto_set_advice_966():
    # The tail is entered only by reset_widget's tail jump and build_widget's
    # fall-through: its x19 is live in that code, so `proto set` advice is wrong.
    bv, head, tail = _split_view(tail_mnemonics=["str"])
    tail_notes = [n["detail"] for n in read_decompile._split_body_notes(bv, tail)]
    saved = [n for n in tail_notes if "callee-saved/link registers" in n]
    assert len(saved) == 1 and "nothing calls this function" in saved[0]
    assert "proto set" not in saved[0]
    # The head IS called (from main), so its stray parameters keep the advice.
    head_saved = [n["detail"] for n in read_decompile._split_body_notes(bv, head)
                  if "callee-saved/link registers" in n["detail"]]
    assert "drop them with `bn proto set`" in head_saved[0]


def test_eh_fragment_chain_longer_than_three_links_966():
    # BN chains cleanup fragments through synthetic calls; each never returns,
    # and the chain's root is entered only by a jump.
    chain = [types.SimpleNamespace(start=0x405600 + 0xc * i, name=f"sub_{i}",
                                   type=types.SimpleNamespace(parameters=[]),
                                   can_return=False) for i in range(8)]
    for fn in chain:
        fn.get_llil_at = lambda a: _op("LLIL_CALL")
    refs = {chain[0].start: [_ref("LLIL_GOTO", 0x414f00)]}
    for prev, nxt in itertools.pairwise(chain):
        refs[nxt.start] = [types.SimpleNamespace(function=prev, address=prev.start + 4)]
    bv = types.SimpleNamespace(get_code_refs=lambda a: refs.get(a, []),
                               get_data_refs=lambda a: [])
    assert read_decompile._is_eh_fragment(bv, chain[-1])
    # ...while the same chain under a root that returns is ordinary code.
    chain[0].can_return = True
    refs[chain[0].start] = [_ref("LLIL_CALL", 0x400100)]
    assert not read_decompile._is_eh_fragment(bv, chain[-1])


@pytest.mark.parametrize("returns", [True, False])
@pytest.mark.parametrize("data_ref", [
    0x43b488,  # a vtable slot in .data.rel.ro
    0x44f010,  # a callback table entry
])
def test_an_address_taken_function_is_called_not_a_body_piece_966(returns, data_ref):
    # Reached only through a vtable or a function pointer: no direct call site
    # and no jump, yet an ordinary function. Its callee-saved "parameter" is a
    # register it saves itself, so the `proto set` advice is right, and it is
    # neither a fragment nor "a piece of another function's body" (#985 re-review).
    bv, _head, tail = _split_view(tail_mnemonics=["ret"], tail_returns=returns,
                                  tail_refs=[], tail_data_refs=[data_ref])
    assert not read_decompile._is_eh_fragment(bv, tail)
    notes = [n["detail"] for n in read_decompile._split_body_notes(bv, tail)]
    assert len(notes) == 1 and "drop them with `bn proto set`" in notes[0]
    assert "nothing calls this function" not in notes[0]


def test_call_arity_group_counts_each_kind_of_evidence_955(monkeypatch):
    # Two calls to one callee: the first has its set-up registers in view; the
    # second follows it directly, so the first call ends its set-up window. The
    # group must not describe both by the first.
    bv, func = _arity_case(
        monkeypatch,
        [_set_reg("x2", _const(1)), _set_reg("x1", _const(2)), _set_reg("x0", _reg("sp"))],
        rendered=1, raw_name="_ZN4demo7Archive4openEPKci",
        signature=(["this", "", ""], False), calls=1,
    )
    second = _op("LLIL_CALL", dest=_op("LLIL_CONST_PTR", constant=0x400e00))
    second.address = 0x401080
    second.mlil = types.SimpleNamespace(params=[object()])
    func.llil.extend([second])
    warning = read_decompile._call_arity_warning(bv, func)["detail"]
    assert "(2 calls at 0x401040, 0x401080) renders 1 of 3 arguments (" in warning
    assert ("1 of 2: the demangled signature takes 3 with `this`; the call site writes "
            "x0, x1, x2 just before it / 1 of 2: the demangled signature takes 3 with "
            "`this`") in warning


def test_a_named_unreferenced_function_is_not_an_eh_fragment_966():
    # An exported method nothing in the binary calls has no references either;
    # only BN's placeholder `sub_` name can be a carved-out fragment.
    bv, _head, tail = _split_view(tail_mnemonics=["ret"], tail_refs=[])
    assert read_decompile._is_eh_fragment(bv, tail)
    tail.name = "_ZNSi8readsomeEPcl"
    assert not read_decompile._is_eh_fragment(bv, tail)


def test_running_into_a_noreturn_piece_is_not_a_split_966():
    # BN outlines a bare `call __stack_chk_fail` as a noreturn function of its
    # own; the block that falls into it hides no return path.
    bv, head, _ = _split_view(tail_mnemonics=["str"], tail_returns=False,
                              tail_refs=[_ref("LLIL_CALL", 0x400400)])
    assert read_decompile._fallthrough_successor(bv, head) is None


def _string_view(memory: bytes, *, semantics="ReadOnlyDataSectionSemantics", string=None,
                 param_type="void*"):
    """A call `put(x0, 0x402ae0)` to a callee at 0x401100 whose second
    parameter is declared *param_type*, with *memory* at 0x402ae0."""
    section = types.SimpleNamespace(semantics=types.SimpleNamespace(name=semantics))
    callee = types.SimpleNamespace(has_user_type=True, type=types.SimpleNamespace(parameters=[
        types.SimpleNamespace(type="void*"), types.SimpleNamespace(type=param_type)]))
    bv = types.SimpleNamespace(
        get_string_at=lambda a: string,
        get_data_var_at=lambda a: None,
        get_sections_at=lambda a: [section],
        read=lambda a, n: memory[:n],
        get_function_at=lambda a: callee if a == 0x401100 else None,
        get_symbol_at=lambda a: None,
    )
    call = _op("MLIL_CALL", dest=_op("MLIL_CONST_PTR", constant=0x401100),
               params=[_op("MLIL_VAR"), _op("MLIL_CONST_PTR", constant=0x402ae0)])
    call.address = 0x4023dc
    func = types.SimpleNamespace(mlil=types.SimpleNamespace(instructions=[call]))
    return bv, func


def test_raw_string_pointer_note_names_an_undefined_short_literal_970():
    bv, func = _string_view(b" -n\0\x01\x02", param_type="char const*")
    text = "    std::__ostream_insert<char>(x0, 0x402ae0);"
    note = read_decompile._raw_string_pointer_note(bv, func, text)
    assert note is not None
    assert '0x402ae0 = " -n" (call at 0x4023dc)' in note["detail"]
    assert "bn data retype 0x402ae0 'char[4]'" in note["detail"]


@pytest.mark.parametrize("param_type", ["char const*", "const char*", "char*"])
def test_raw_string_pointer_note_takes_a_short_run_passed_as_char_pointer_970(param_type):
    bv, func = _string_view(b"/\0\0\0", param_type=param_type)
    note = read_decompile._raw_string_pointer_note(bv, func, "    put(x0, 0x402ae0);")
    assert note is not None and '0x402ae0 = "/"' in note["detail"]


@pytest.mark.parametrize("memory,kwargs,text", [
    # Already rendered as a literal: the address is not in the text.
    (b" -n\0", {}, '    std::__ostream_insert<char>(x0, " -n");'),
    # BN defined a string there.
    (b" -n\0", {"string": object()}, "    f(x0, 0x402ae0);"),
    # Writable data is a buffer, not a literal.
    (b" -n\0", {"semantics": "ReadWriteDataSectionSemantics"}, "    f(x0, 0x402ae0);"),
    # Not printable, or not terminated within the window.
    (b"\x01\x02\0", {}, "    f(x0, 0x402ae0);"),
    (b"A" * 80, {}, "    f(x0, 0x402ae0);"),
    # Printable-then-NUL data that is not a string, passed where the callee does
    # not take a `char *` (#985 review): an int[] {0x41, ...}, a UTF-16 "hi", a
    # struct {0x2a, 0}, and a two-byte run.
    (b"A\0\0\0B\0\0\0", {}, "    sum_ints(0x402ae0);"),
    (b"h\0i\0\0\0", {"param_type": "uint16_t const*"}, "    wlen(0x402ae0);"),
    (b"*\0\0\0\0\0\0\0", {"param_type": "void const*"}, "    cfg_id(0x402ae0);"),
    (b"ok\0", {}, "    f(x0, 0x402ae0);"),
    # A wide or double-pointer parameter is not narrow-string evidence.
    (b"h\0i\0\0\0", {"param_type": "wchar_t const*"}, "    f(x0, 0x402ae0);"),
    (b"A\0", {"param_type": "char**"}, "    f(x0, 0x402ae0);"),
])
def test_raw_string_pointer_note_is_silent_without_an_undefined_literal_970(memory, kwargs, text):
    bv, func = _string_view(memory, **kwargs)
    assert read_decompile._raw_string_pointer_note(bv, func, text) is None


def test_decompile_shows_this_sessions_comments_and_counts_hidden_ones_977(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    function = _FakeFunction(0x401000, "parse_record")
    function.basic_blocks = [_FakeBasicBlock(0x401000, 0x401010)]
    function.comment = "parses one record"
    bv = _FakeBV(
        functions=[function],
        comments={0x401004: "dispatch on message type", 0x401008: "planted note"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    text = (
        "void parse_record() {  // parses one record\n"
        "    dispatch();  // dispatch on message type\n"
        "    // planted note\n"
        "}"
    )
    monkeypatch.setattr(bridge.il_format, "_decompile_text", lambda *a, **k: text)
    instance.targets._ids_by_object[bv] = "v1"
    instance.targets.note_session_annotations(bv, [
        {"op": "set_comment", "status": "verified", "address": "0x401004",
         "requested": {"comment": "dispatch on message type"}},
        {"op": "set_comment", "status": "verified", "scope": "function_doc",
         "address": "0x401000", "requested": {"comment": "parses one record"}},
        # A noop set means the body was already stored: not this session's.
        {"op": "set_comment", "status": "noop", "address": "0x401008",
         "requested": {"comment": "planted note"}},
    ])

    result = instance._decompile("active", "parse_record")

    assert result["text"] == (
        "void parse_record() {  // [this session] parses one record\n"
        "    dispatch();  // [this session] dispatch on message type\n"
        "    // <annotation redacted>\n"
        "}"
    )
    assert result["annotation_summary"] == {
        "comment_count": 3, "redacted": True, "hidden": 1, "session_shown": 2,
    }

    # A verified delete forgets the location; a later inherited body there is
    # redacted again.
    instance.targets.note_session_annotations(bv, [
        {"op": "delete_comment", "status": "verified", "address": "0x401004"},
    ])
    again = instance._decompile("active", "parse_record")
    assert "// <annotation redacted>" in again["text"].splitlines()[1]
    assert again["annotation_summary"]["hidden"] == 2


def test_session_comment_body_also_stored_elsewhere_stays_redacted_977():
    # The rendered line cannot say which store it came from, so a body that is
    # BOTH this session's and inherited elsewhere is treated as inherited.
    text, hidden, shown = read_decompile._redact_rendered_annotations(
        "    a();  // same words\n    b();  // same words", ["same words"], ["same words"])
    assert text.count("<annotation redacted>") == 2
    assert (hidden, shown) == (2, 0)


# ---------------------------------------------------------------------------
# #1000 decompile warnings: dropped arguments marked on the call line, an
# unresolved virtual tail call, `__offset` base types; #994 item 2 short
# literals passed to untyped C library imports.
# ---------------------------------------------------------------------------


class _Expr(types.SimpleNamespace):
    """An IL expression that prints the way BN renders it."""

    def __str__(self):
        return self.text


def _hop(name, text, **fields):
    return _Expr(operation=types.SimpleNamespace(name=name), text=text,
                 operands=list(fields.values()), **fields)


class _Block(list):
    def __init__(self, start, items, incoming=()):
        super().__init__(items)
        self.start = start
        self.incoming_edges = [types.SimpleNamespace(source=b) for b in incoming]
        self.outgoing_edges = []


def _at(address, ins):
    ins.address = address
    return ins


def _select_then_call_case(monkeypatch):
    """`write(this, buf, n)` where `n = len > 0x100 ? 0x100 : len` is one `csel`
    right before the call: BN lifts the csel into a diamond, so the call opens
    its own LLIL block with nothing in it."""
    head = _Block(0, [
        _at(0x4015b8, _set_reg("w2", _op("LLIL_LOAD", src=_reg("x21")))),
        _at(0x4015bc, _set_reg("w3", _const(0x100))),
        _at(0x4015c0, _set_reg("x1", _reg("x21"))),
        _at(0x4015c4, _set_reg("x0", _reg("x20"))),
        _at(0x4015cc, _op("LLIL_IF", condition=_reg("w2"))),
    ])
    keep = _Block(5, [_at(0x4015cc, _set_reg("w2", _reg("w2"))),
                      _at(0x4015cc, _op("LLIL_GOTO"))], [head])
    clamp = _Block(7, [_at(0x4015cc, _set_reg("w2", _reg("w3"))),
                       _at(0x4015cc, _op("LLIL_GOTO"))], [head])
    call = _at(0x4015d0, _op("LLIL_CALL", dest=_op("LLIL_CONST_PTR", constant=0x400e00)))
    call.mlil = types.SimpleNamespace(params=[object(), object()])
    join = _Block(9, [call], [keep, clamp])
    bv, func = _arity_case(monkeypatch, [], rendered=2, raw_name="_ZNSo5writeEPKcl",
                           signature=(["this", "", ""], False))
    func.llil = types.SimpleNamespace(basic_blocks=[head, keep, clamp, join])
    bv.get_disassembly = lambda a: "csel    w2, w2, w3, ls" if a == 0x4015cc else "mov"
    return bv, func


def test_call_arity_sees_an_argument_a_select_computes_right_before_the_call_1000(monkeypatch):
    bv, func = _select_then_call_case(monkeypatch)
    [row] = read_decompile._call_arity_findings(bv, func)
    assert (row["rendered"], row["expected"]) == (2, 3)
    # Both arms are read: `w3` feeds the clamp arm, so it is no argument.
    assert row["evidence"][-1] == "the call site writes x0, x1, x2 just before it"
    assert row["missing"] == [("x2", "csel @ 0x4015cc")]
    assert read_decompile._arity_line_note(row) == \
        "/* bn: +1 arg not rendered: x2 = csel @ 0x4015cc */"


def test_registers_only_the_select_condition_reads_are_not_arguments_1000(monkeypatch):
    # `ldr x6, [x0]; ldr x7, [x0, 8]; mov x0, x1; mov x1, #4; cmp x6, x7;
    # csel x2, x1, x0, ls; bl ext`: x6 and x7 feed the compare, not `ext`.
    head = _Block(0, [
        _at(0x4007a0, _set_reg("x6", _op("LLIL_LOAD", src=_reg("x0")))),
        _at(0x4007a4, _set_reg("x7", _op("LLIL_LOAD", src=_reg("x0")))),
        _at(0x4007a8, _set_reg("x0", _reg("x1"))),
        _at(0x4007ac, _set_reg("x1", _const(4))),
        _at(0x4007b0, _op("LLIL_IF", condition=_op("LLIL_CMP_ULE", left=_reg("x6"),
                                                  right=_reg("x7")))),
    ])
    low = _Block(5, [_at(0x4007b0, _set_reg("x2", _reg("x1"))),
                     _at(0x4007b0, _op("LLIL_GOTO"))], [head])
    high = _Block(7, [_at(0x4007b0, _set_reg("x2", _reg("x0"))),
                      _at(0x4007b0, _op("LLIL_GOTO"))], [head])
    call = _at(0x4007b4, _op("LLIL_CALL", dest=_op("LLIL_CONST_PTR", constant=0x400e00)))
    call.mlil = types.SimpleNamespace(params=[object()] * 3)
    bv, func = _arity_case(monkeypatch, [], rendered=3, raw_name="ext")
    func.llil = types.SimpleNamespace(
        basic_blocks=[head, low, high, _Block(9, [call], [low, high])])
    assert read_decompile._call_arity_findings(bv, func) == []


def test_call_arity_note_names_each_dropped_register_and_its_value_1000(monkeypatch):
    bv, func = _arity_case(
        monkeypatch,
        [_set_reg("w2", _hop("LLIL_CONST", "0x30", constant=0x30)),
         _set_reg("x1", _hop("LLIL_LOAD", "[sp {__path}].q", src=_reg("sp"))),
         _set_reg("x0", _reg("x27"))],
        rendered=1, raw_name="_ZN4demo7Archive4openEPKci",
        signature=(["this", "", ""], False),
    )
    [row] = read_decompile._call_arity_findings(bv, func)
    assert read_decompile._arity_line_note(row) == \
        "/* bn: +2 args not rendered: x1 = [sp].q, x2 = 0x30 */"
    # An excess names the counts instead: there is no register to point at.
    bv, func = _arity_case(monkeypatch, [], rendered=3, raw_name="_ZN4demo6Widget5resetEv",
                           signature=(["this"], False))
    [row] = read_decompile._call_arity_findings(bv, func)
    assert read_decompile._arity_line_note(row) == "/* bn: renders 3 args, signature takes 1 */"


def test_call_arity_note_never_names_a_register_overwritten_before_the_call_1000(monkeypatch):
    # `mov x2, x0` copies strlen's result, then `x0 = &s`: "x2 = x0" would read
    # as "the length is &s". The value is unknown here, so only x2 is named.
    bv, func = _arity_case(
        monkeypatch,
        [_set_reg("x2", _reg("x0")),
         _set_reg("x1", _hop("LLIL_ADD", "x0 + 8", left=_reg("x0"), right=_const(8))),
         _set_reg("x0", _reg("sp"))],
        rendered=1, raw_name="_ZN4demo6String6appendEPKcm",
        signature=(["this", "", ""], False),
    )
    [row] = read_decompile._call_arity_findings(bv, func)
    assert row["missing"] == [("x1", None), ("x2", None)]
    assert read_decompile._arity_line_note(row) == "/* bn: +2 args not rendered: x1, x2 */"


def test_window_values_follow_copies_and_name_parameters_1000():
    # `w4 = w2; w2 = w3; w3 = w4` swaps x2 and x3. BN's dataflow names what
    # each held when the window began: this function's arg3 and arg4.
    arch = _aarch64_arch()
    arch.get_reg_name = lambda storage: storage
    window = [_set_reg("w4", _reg("w2")), _set_reg("w2", _reg("w3")),
              _set_reg("w3", _reg("w4"))]
    window[0].address = 0x400574
    entry = {"x2": "x2", "x3": "x3"}

    def reg_value(address, reg):
        assert address == 0x400574
        return types.SimpleNamespace(type=types.SimpleNamespace(name="EntryValue"),
                                     reg=entry[reg]) if reg in entry else \
            types.SimpleNamespace(type=types.SimpleNamespace(name="UndeterminedValue"))

    func = types.SimpleNamespace(
        get_reg_value_at=reg_value,
        parameter_vars=[_param_var(n, r, 4) for n, r in (("arg3", "x2"), ("arg4", "x3"))])
    bv = types.SimpleNamespace(arch=arch)
    values = read_decompile._window_values(bv, func, window, set())
    assert (values["x2"], values["x3"]) == ("arg4", "arg3")
    # Without dataflow, a register the window itself overwrites is not a value.
    func.get_reg_value_at = lambda address, reg: None
    values = read_decompile._window_values(bv, func, window, set())
    assert (values["x2"], values["x3"]) == (None, None)


def _entry_func(entry, params):
    """A function whose dataflow says each register in *entry* still holds the
    entry value of the named register; everything else is undetermined."""
    def reg_value(address, reg):
        if reg in entry:
            return types.SimpleNamespace(type=types.SimpleNamespace(name="EntryValue"),
                                         reg=entry[reg])
        return types.SimpleNamespace(type=types.SimpleNamespace(name="UndeterminedValue"))
    return types.SimpleNamespace(get_reg_value_at=reg_value, parameter_vars=params)


def test_window_values_keep_the_width_a_narrow_copy_reads_1000():
    # `return o->cb((unsigned int)v)` with v a 64-bit arg2: `mov w0, w1`
    # hands over only the low 32 bits.
    arch = _aarch64_arch()
    arch.get_reg_name = lambda storage: storage
    window = [_set_reg("w0", _reg("w1")), _set_reg("x2", _reg("x1"))]
    window[0].address = 0x400850
    func = _entry_func({"x1": "x1"}, [_param_var("arg2", "x1", 8)])
    bv = types.SimpleNamespace(arch=arch)
    values = read_decompile._window_values(bv, func, window, set())
    assert values["x0"] == "(uint32_t)arg2"
    assert values["x2"] == "arg2"
    # A 32-bit parameter read through w1 is the whole value.
    func = _entry_func({"x1": "x1"}, [_param_var("arg2", "x1", 4)])
    assert read_decompile._window_values(bv, func, window, set())["x0"] == "arg2"


def test_window_values_treat_split_and_intrinsic_writes_as_clobbers_1000():
    # aarch64 `mov x1, x5; mrs x5, tpidr_el0; mrs x0, tpidr_el0; mov x2, x0`
    # and x86 `mul rsi` (rdx:rax): neither write is a SET_REG, and a note
    # that missed them named a value the register no longer held.
    arch = _aarch64_arch()
    arch.get_reg_name = lambda storage: storage
    window = [
        _set_reg("x1", _reg("x5")),
        _op("LLIL_INTRINSIC", output=[types.SimpleNamespace(name="x5")]),
        _op("LLIL_INTRINSIC", output=[types.SimpleNamespace(name="x0")]),
        _set_reg("x2", _reg("x0")),
        _set_reg("x3", _reg("x4")),
        _op("LLIL_SET_REG_SPLIT", hi=types.SimpleNamespace(name="x4"),
            lo=types.SimpleNamespace(name="x6"), src=_reg("x7")),
    ]
    window[0].address = 0x400900
    bv = types.SimpleNamespace(arch=arch)
    func = _entry_func({"x0": "x0"}, [_param_var("arg1", "x0", 8)])
    values = read_decompile._window_values(bv, func, window, set())
    # x5 has no name and is rewritten later: unknown, not `x5`.
    assert values["x1"] is None
    assert values["x2"] is None  # x0 holds the intrinsic's result, not arg1
    assert values["x3"] is None  # x4 is the split's high half by the jump
    assert values["x4"] is None and values["x6"] is None


def test_call_arity_summary_says_which_calls_carry_the_inline_note_1000(monkeypatch):
    bv, func = _arity_case(monkeypatch, [], rendered=1, raw_name="_ZN4demo7Archive4openEPKci",
                           signature=(["this", "", ""], False), calls=2)
    findings = read_decompile._call_arity_findings(bv, func)
    full = read_decompile._call_arity_warning(bv, func, findings, {0x401040, 0x401048})
    assert full["code"] == "arity"
    assert full["summary"] == ("2 call(s) render the wrong argument count, marked inline: "
                               "N4demo7Archive4openEPKci 1/3 @ 0x401040, 0x401048")
    assert [c["inline"] for c in full["calls"]] == [True, True]
    assert [c["missing"] for c in full["calls"]] == [
        [{"register": "x1", "value": None}, {"register": "x2", "value": None}]] * 2
    # A note BN gave no line to attach to is not claimed as placed.
    part = read_decompile._call_arity_warning(bv, func, findings, {0x401040})
    assert ", 1 marked inline:" in part["summary"]
    assert "1 of the 2 calls are marked inline" in part["detail"]
    none = read_decompile._call_arity_warning(bv, func, findings)
    assert "inline" not in none["summary"] and "inline" not in none["detail"]


def test_pseudo_c_appends_a_note_to_the_first_statement_line_at_its_address_1000(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    func = _FakeFunction(0x401420, "save_record")
    _install_fake_pseudo_c(monkeypatch, bridge, func, [[
        (0x401420, "  void save_record()"),
        (0x401420, "{"),
        (0x401598, "    int64_t r;"),
        (0x401598, "    // parse(f) comes next"),
        (0x401598, "    int64_t r = open(&f);  // opens the profile"),
        (0x401598, "    if (r)"),
        (0x401600, "❓    /* jump -> x7 */"),
        (0x401620, "}"),
    ]])
    placed: set[int] = set()
    text = bridge.il_format._pseudo_c_text(
        func, line_notes={0x401598: ["/* bn: +2 args not rendered: x1, x2 */"],
                          0x401600: ["/* bn: indirect tail call */"],
                          0x401700: ["/* bn: never rendered */"]},
        placed=placed)
    lines = text.splitlines()
    # Ahead of BN's trailing comment, never inside it; a comment-only line
    # carrying a `(` is no statement.
    assert lines[3] == "    // parse(f) comes next"
    assert lines[4] == ("    int64_t r = open(&f);  /* bn: +2 args not rendered: x1, x2 */"
                        "  // opens the profile")
    assert lines[6] == "❓    /* jump -> x7 */  /* bn: indirect tail call */"
    assert text.count("/* bn:") == 2
    assert placed == {0x401598, 0x401600}


def test_a_note_leaves_stored_comments_redacted_and_session_ones_marked_1000(monkeypatch):
    # The note used to land INSIDE the trailing comment, so the stored body
    # stopped matching: an inherited comment printed in full and this
    # session's lost its `[this session]` mark.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    function = _FakeFunction(0x401000, "save_record")
    function.basic_blocks = [_FakeBasicBlock(0x401000, 0x401010)]
    bv = _FakeBV(functions=[function],
                 comments={0x401004: "secret path note", 0x401008: "session wrote this"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    _install_fake_pseudo_c(monkeypatch, bridge, function, [[
        (0x401000, "  void save_record()"),
        (0x401000, "{"),
        (0x401004, "    open(&f);  // secret path note"),
        (0x401008, "    init(&s);  // session wrote this"),
        (0x40100c, "}"),
    ]])
    rows = [{"address": a, "target": 0x400e00, "label": "open @ 0x400e00", "rendered": 1,
             "expected": 2, "evidence": ("the call site writes x0, x1 just before it",),
             "missing": [("x1", None)]} for a in (0x401004, 0x401008)]
    monkeypatch.setattr(bridge.read_decompile, "_call_arity_findings", lambda bv, func: rows)
    instance.targets._ids_by_object[bv] = "v1"
    instance.targets.note_session_annotations(bv, [
        {"op": "set_comment", "status": "verified", "address": "0x401008",
         "requested": {"comment": "session wrote this"}},
    ])

    result = instance._decompile("active", "save_record")

    lines = result["text"].splitlines()
    assert lines[2] == "    open(&f);  /* bn: +1 arg not rendered: x1 */  // <annotation redacted>"
    assert lines[3] == ("    init(&s);  /* bn: +1 arg not rendered: x1 */  "
                        "// [this session] session wrote this")
    assert result["annotation_summary"]["hidden"] == 1
    assert result["annotation_summary"]["session_shown"] == 1
    assert [row["code"] for row in result["warning_rows"]] == ["arity"]
    # `warnings` keeps the sentence shape every other read op uses.
    assert result["warnings"] == [result["warning_rows"][0]["detail"]]


def _var(name):
    return _hop("HLIL_VAR", name, var=types.SimpleNamespace(name=name))


def _hconst(value, text=None):
    return _hop("HLIL_CONST", text or hex(value), constant=value)


def _plus(left, right):
    return _hop("HLIL_ADD", f"{left} + {right}", left=left, right=right)


def _load(src):
    return _hop("HLIL_DEREF", f"*({src})", src=src)


def _jump_case(*, exits=True, dest=None, symbol_kind="FunctionSymbol", section=".text",
               thunk=False, copies=1):
    """One machine jump at 0x40058c, the last instruction of *func*'s only
    block, that HLIL shows *copies* times (BN repeats it per switch case)."""
    if dest is None:
        dest = _load(_plus(_load(_load(_plus(_var("this"), _hconst(0x18)))), _hconst(0x10)))
    jumps = [_at(0x40058c, _hop("HLIL_JUMP", "jump(...)", dest=dest)) for _ in range(copies)]
    llil_jump = _op("LLIL_JUMP", dest=_reg("x16"))
    block = _Block(0, [llil_jump])
    if not exits:
        block.outgoing_edges = [types.SimpleNamespace(target=None)]
    llil_jump.il_basic_block = block
    func = types.SimpleNamespace(
        start=0x400570, is_thunk=thunk,
        hlil=types.SimpleNamespace(instructions=jumps),
        calling_convention=types.SimpleNamespace(int_arg_regs=["x0", "x1", "x2"]),
        get_llil_at=lambda a: llil_jump if a == 0x40058c else None,
    )
    bv = types.SimpleNamespace(
        get_symbol_at=lambda a: types.SimpleNamespace(type=types.SimpleNamespace(name=symbol_kind)),
        get_sections_at=lambda a: [types.SimpleNamespace(name=section)],
    )
    return bv, func


_TABLE = _hop("HLIL_CONST_PTR", "&tbl", constant=0x420010)


@pytest.mark.parametrize("dest", [
    # `return inner->dispatch(msg, b, a)` with Dispatcher untyped.
    None,
    # `return fn(x + 1)`: also the shape of a computed-goto dispatch BN split
    # into a function of its own -- which is why nothing here says which.
    _var("arg2"),
    # `goto *tbl[op & 3]`.
    _load(_plus(_TABLE, _hop("HLIL_LSL", "(op & 3) << 3"))),
], ids=["vtable-shape", "parameter", "computed-goto"])
def test_every_unresolved_exit_jump_gets_one_neutral_warning_1000(dest):
    # Three review rounds could not tell a tail call from a computed goto by
    # the IL; every shape gets the same row, naming both and claiming neither.
    bv, func = _jump_case(dest=dest)
    [row] = read_decompile._unresolved_jump_findings(bv, func)
    warning = read_decompile._unresolved_jump_warning(func, row, True)
    assert warning["code"] == "jump" and warning["address"] == "0x40058c"
    assert warning["target"] == str(row["target"])
    detail = warning["detail"]
    assert f"to `{row['target']}`" in detail and "❓ marks control flow" in detail
    assert "argument registers (x0, x1, x2)" in detail and "bn proto set" in detail
    assert "computed goto or jump table" in detail
    for claim in ("virtual", "slot", "through the function pointer", "is an indirect tail call"):
        assert claim not in detail and claim not in warning["summary"]
    # Neither reading is stated as fact: leaving the function is the tail
    # call's, staying in it the jump table's.
    assert "leaving the function" in detail and "staying in this function" in detail
    # The jump-table half promises nothing: no table here, so only the
    # decompile's own stores are offered, as a "may".
    assert "None of these steps is sure to find the targets" in detail
    assert "may name the targets" in detail and "`bn xrefs 0x" not in detail
    note = read_decompile._JUMP_LINE_NOTE
    assert note.startswith("/* bn:") and note.endswith("*/")
    assert "tail call" in note and "computed goto" in note and "virtual" not in note


def test_a_jump_reading_a_data_table_names_the_steps_and_their_limits_1000():
    # A `.bss` table filled at run time (`jt`), a static absolute table
    # (`idxjump`) and a relative one (ninja) all read from one data address.
    table = _hop("HLIL_CONST_PTR", "&tbl", constant=0x420010)
    bv, func = _jump_case(dest=_load(_plus(table, _hop("HLIL_LSL", "(op & 3) << 3"))))
    bv.is_valid_offset = lambda value: 0x400000 <= value < 0x430000
    bv.get_functions_containing = lambda value: [func] if 0x400570 <= value < 0x400590 else []
    [row] = read_decompile._unresolved_jump_findings(bv, func)
    detail = read_decompile._unresolved_jump_warning(func, row, False)["detail"]
    assert row["table"] == 0x420010
    # xrefs: the writes of a table filled at run time.
    assert "`bn xrefs 0x420010`" in detail and "filled at run time" in detail
    assert "base address only" in detail
    # evidence table: only a static table, and it says what relative entries mean.
    assert "`bn evidence table 0x420010`" in detail
    assert "added to the table base 0x420010" in detail and "reads as zeros" in detail
    assert "`bn disasm <target> --linear 16`" in detail
    # A target with no data address (a stack array, a register) names none.
    bv, func = _jump_case(dest=_var("arg2"))
    [row] = read_decompile._unresolved_jump_findings(bv, func)
    assert row["table"] is None


def test_an_operand_that_raises_names_no_table_and_keeps_the_row_1000():
    # `_jump_data_address` walks HLIL operands; one that raises used to escape
    # the per-jump `try` and abort the whole read. It costs only the table.
    class _Raising(_Expr):
        @property
        def operands(self):
            raise RuntimeError("operand list unavailable")

    bad = _Raising(operation=types.SimpleNamespace(name="HLIL_DEREF"), text="*(?)")
    bv, func = _jump_case(dest=bad)
    bv.is_valid_offset = lambda value: True
    bv.get_functions_containing = lambda value: []
    [row] = read_decompile._unresolved_jump_findings(bv, func)
    assert row["address"] == 0x40058c and row["table"] is None


def test_a_jump_bn_repeats_in_switch_cases_is_one_row_with_a_note_on_every_copy_1000(
        monkeypatch):
    # `sw_then_cb`: BN rendered one `br` in two switch cases; it got two rows,
    # both notes on the first copy, and none on the second.
    bv, func = _jump_case(copies=2)
    assert [row["address"] for row in read_decompile._unresolved_jump_findings(bv, func)] \
        == [0x40058c]
    bridge = _load_bridge(monkeypatch)
    fake = _FakeFunction(0x400570, "sw_then_cb")
    _install_fake_pseudo_c(monkeypatch, bridge, fake, [[
        (0x400570, "  int64_t sw_then_cb(int64_t arg1, int32_t arg2)"),
        (0x400570, "{"),
        (0x400580, "    case 4:"),
        (0x40058c, "❓        /* jump -> *(uint64_t*)(arg1 + 8) */"),
        (0x400584, "    case 5:"),
        (0x40058c, "❓        /* jump -> *(uint64_t*)(arg1 + 8) */  // shared exit"),
        (0x400590, "}"),
    ]])
    placed: set[int] = set()
    text = bridge.il_format._pseudo_c_text(
        fake, jump_notes={0x40058c: "/* bn: J */"}, placed=placed)
    lines = text.splitlines()
    assert lines[3] == "❓        /* jump -> *(uint64_t*)(arg1 + 8) */  /* bn: J */"
    assert lines[5] == "❓        /* jump -> *(uint64_t*)(arg1 + 8) */  /* bn: J */  // shared exit"
    assert text.count("/* bn: J */") == 2 and placed == {0x40058c}


@pytest.mark.parametrize("kwargs", [
    # BN resolved the target (a switch), or the jump does not leave the function.
    {"dest": _hop("HLIL_CONST_PTR", "0x400800", constant=0x400800)},
    {"exits": False},
    # An import stub, a thunk, and the PLT header (`push [..]; jmp [GOT+16]`).
    {"symbol_kind": "ImportedFunctionSymbol"},
    {"thunk": True},
    {"section": ".plt"},
    {"section": ".plt.sec"},
], ids=["resolved", "has-successor", "import", "thunk", "plt0", "plt-sec"])
def test_resolved_jumps_and_plt_stubs_get_no_jump_warning_1000(kwargs):
    assert read_decompile._unresolved_jump_findings(*_jump_case(**kwargs)) == []


def test_a_note_never_lands_on_the_signature_line_1000(monkeypatch):
    # A jump at a function's first instruction shares the entry address with
    # the prototype line; the note goes on the jump line.
    bridge = _load_bridge(monkeypatch)
    func = _FakeFunction(0x401000, "tcfirst")
    _install_fake_pseudo_c(monkeypatch, bridge, func, [[
        (0x401000, "  int64_t tcfirst(int64_t arg1)"),
        (0x401000, "{"),
        (0x401000, "❓    /* jump -> *(uint64_t*)(arg1 + 8) */"),
        (0x401000, "}"),
    ]])
    placed: set[int] = set()
    text = bridge.il_format._pseudo_c_text(
        func, line_notes={0x401000: ["/* bn: note */"]}, placed=placed)
    assert text.splitlines() == [
        "int64_t tcfirst(int64_t arg1)", "{",
        "❓    /* jump -> *(uint64_t*)(arg1 + 8) */  /* bn: note */", "}"]
    assert placed == {0x401000}


def _offset_func(this_type="demo::Store"):
    target = _Expr(text=this_type)
    this = types.SimpleNamespace(name="this", type=types.SimpleNamespace(target=target))
    return types.SimpleNamespace(start=0x401964, parameter_vars=[this])


_OFFSET_TEXT = ("    void* __offset(Req, 0x40) p = &r->body;\n"
                "    *this = __offset(vtable_for_demo::Store, 0x10);")


def test_offset_warning_says_which_base_types_have_no_definition_1000():
    bv = types.SimpleNamespace(get_type_by_name=lambda name: None)
    warning = read_decompile._offset_warning(bv, _offset_func(), _OFFSET_TEXT)
    assert warning["code"] == "offset"
    assert warning["types"] == {"defined": [],
                                "missing": ["Req", "vtable_for_demo::Store", "demo::Store"]}
    detail = warning["detail"]
    # The old note sent the reader to `types show` for types that do not exist.
    assert "authoritative" not in detail
    assert "No definition: Req, demo::Store" in detail and "bn types declare" in detail
    assert "vtable_for_demo::Store names a class's vtable OBJECT" in detail
    assert "`this` points at demo::Store, which has no definition either." in detail


def test_offset_warning_names_defined_types_and_the_vtable_route_1000():
    known = {"Req": object(), "demo::Store::VTable": object()}
    bv = types.SimpleNamespace(get_type_by_name=known.get)
    warning = read_decompile._offset_warning(bv, _offset_func(), _OFFSET_TEXT)
    assert warning["types"]["defined"] == ["Req"]
    assert "Defined here: Req" in warning["detail"]
    assert "`bn struct show demo::Store` prints the declaration" in warning["detail"]
    assert "struct show` will not" not in warning["detail"]
    assert read_decompile._offset_warning(bv, _offset_func(), "    return 0;") is None


def _libc_string_view(memory, *, name, kind="ImportedFunctionSymbol", params=(), library=None,
                      user_typed=False):
    """`if (!name(arg2, 0x661bd8))` where *name* is a callee at 0x401100 whose
    prototype declares *params*, and 0x661bd8 holds *memory* in .rodata."""
    section = types.SimpleNamespace(
        semantics=types.SimpleNamespace(name="ReadOnlyDataSectionSemantics"))
    callee = types.SimpleNamespace(
        has_user_type=user_typed,
        type=types.SimpleNamespace(parameters=[types.SimpleNamespace(type=t) for t in params]))
    symbol = types.SimpleNamespace(raw_name=name, type=types.SimpleNamespace(name=kind))
    libraries = [] if library is None else [types.SimpleNamespace(
        get_named_object=lambda n: library if n == name else None)]
    bv = types.SimpleNamespace(
        get_string_at=lambda a: None,
        get_data_var_at=lambda a: None,
        get_sections_at=lambda a: [section],
        read=lambda a, n: memory[:n],
        get_function_at=lambda a: callee if a == 0x401100 else None,
        get_symbol_at=lambda a: symbol if a == 0x401100 else None,
        platform=types.SimpleNamespace(type_libraries=libraries),
        type_libraries=[],
    )
    call = _op("MLIL_CALL", dest=_op("MLIL_CONST_PTR", constant=0x401100),
               params=[_op("MLIL_VAR"), _op("MLIL_CONST_PTR", constant=0x661bd8)])
    call.address = 0x51c884
    func = types.SimpleNamespace(mlil=types.SimpleNamespace(instructions=[call]))
    return bv, func


_LIBC_TEXT = "    if (!strcmp(arg2, 0x661bd8))"
_INT_PAIR = types.SimpleNamespace(parameters=[
    types.SimpleNamespace(type="int32_t"), types.SimpleNamespace(type="int64_t*")])


@pytest.mark.parametrize("kwargs", [
    {"name": "strcmp"}, {"name": "strcmp@plt"},
    # The format argument comes from the format-family table.
    {"name": "__isoc99_sscanf"}, {"name": "fprintf"},
    # A curated entry outranks a type library that types it wrongly (BN's
    # aarch64 libc declares `setlocale(int32_t, int64_t*)`).
    {"name": "setlocale", "params": ("int32_t", "int64_t*"), "library": _INT_PAIR},
], ids=["strcmp", "plt", "sscanf-format", "fprintf-format", "setlocale"])
def test_a_short_literal_to_an_untyped_libc_import_is_named_994(kwargs):
    # #994 item 2: `strcmp` with no prototype passed "C"; the typed import's
    # rendering showed `strcmp(arg2, "C")`, the untyped one a bare address.
    bv, func = _libc_string_view(b"C\0", **kwargs)
    note = read_decompile._raw_string_pointer_note(bv, func, _LIBC_TEXT)
    assert note is not None and '0x661bd8 = "C" (call at 0x51c884)' in note["detail"]
    assert note["strings"] == [{"address": "0x661bd8", "value": "C", "call": "0x51c884"}]


@pytest.mark.parametrize("kwargs", [
    # Not a C library name: no type evidence for a short literal.
    {"name": "frob_lookup"},
    # A body here whose own prototype covers the argument: BN's type stands.
    {"name": "strcmp", "kind": "FunctionSymbol", "params": ("char const*", "int64_t"),
     "user_typed": True},
    # A prototype the analyst set on the import (`bn proto set`) wins.
    {"name": "strcmp", "params": ("int64_t", "int64_t"), "user_typed": True},
    # A name only a type library knows, typed there as no `char *`.
    {"name": "locale_hook", "library": _INT_PAIR},
], ids=["unknown", "typed-body", "user-prototype", "library-int"])
def test_a_short_literal_needs_a_char_pointer_from_somewhere_994(kwargs):
    bv, func = _libc_string_view(b"C\0", **kwargs)
    assert read_decompile._raw_string_pointer_note(bv, func, _LIBC_TEXT) is None


def test_library_character_parameter_alone_is_not_string_evidence():
    library = types.SimpleNamespace(parameters=[
        types.SimpleNamespace(type="int32_t"), types.SimpleNamespace(type="char const*")])
    bv, func = _libc_string_view(b"C\0", name="locale_hook", library=library)
    note = read_decompile._raw_string_pointer_note(bv, func, _LIBC_TEXT)
    assert note is None


def test_a_literal_bn_renders_as_a_data_placeholder_is_named_994():
    # BN's aarch64 libc types `setlocale(int32_t, int64_t*)`, so the argument
    # renders `&data_661bd8`, not `0x661bd8`; the string there is just as
    # undefined.
    bv, func = _libc_string_view(b"C\0", name="setlocale", params=("int32_t", "int64_t*"))
    note = read_decompile._raw_string_pointer_note(
        bv, func, "    return setlocale(6, &data_661bd8);")
    assert note is not None and note["strings"][0]["value"] == "C"


@pytest.mark.parametrize("argument", ["0x661bd8", "&data_661bd8"])
@pytest.mark.parametrize("data_type", [None, "int32_t[4]", "struct Record", "char*"])
def test_memcmp_bytes_are_not_claimed_as_strings_1019(argument, data_type):
    # int k[4] = {0x00434241, 0, 0, 0} starts with printable "ABC".
    bv, func = _libc_string_view(
        b"ABC\0" + b"\0" * 12, name="memcmp", params=("void const*", "void const*"))
    bv.get_data_var_at = lambda a: (
        None if data_type is None else types.SimpleNamespace(type=data_type))
    note = read_decompile._raw_string_pointer_note(
        bv, func, f"    return memcmp(arg1, {argument}, 16);")
    assert note is not None and note["code"] == "bytes"
    assert note["address"] == "0x51c884"
    assert 'bytes at 0x661bd8 read as "ABC"' in note["detail"]
    assert note["strings"] == []
    assert note["bytes"] == [{"address": "0x661bd8", "value": "ABC", "call": "0x51c884"}]
    assert "bn data retype" not in note["summary"] + note["detail"]


@pytest.mark.parametrize("data_type", ["char[3]", "char[0x3]", "const char[0x3]"])
def test_declared_character_array_suppresses_retype_advice(data_type):
    bv, func = _libc_string_view("٬".encode() + b"\0", name="strcmp")
    bv.get_data_var_at = lambda a: types.SimpleNamespace(type=data_type)
    note = read_decompile._raw_string_pointer_note(bv, func, _LIBC_TEXT)
    assert note["strings"][0]["value"] == "٬"
    assert "bn data retype" not in note["detail"] + note["summary"]


@pytest.mark.parametrize("user_typed", [False, True])
def test_inferred_char_pointer_is_not_string_evidence(user_typed):
    bv, func = _libc_string_view(
        b"\xe2\x80\x99\0" + b"\1\0\0\0\2\0\0\0\3\0\0\0",
        name="opaque_hash", kind="FunctionSymbol",
        params=("int32_t", "char*"), user_typed=user_typed)
    note = read_decompile._raw_string_pointer_note(bv, func, _LIBC_TEXT)
    assert note["code"] == ("string" if user_typed else "bytes")
    if not user_typed:
        assert note["strings"] == []
        assert note["bytes"][0]["value"] == "’"
        assert "bn data retype" not in note["detail"] + note["summary"]


@pytest.mark.parametrize("kind", ["FunctionSymbol", "ImportedFunctionSymbol", "ExternalSymbol"])
def test_string_api_identity_requires_external_provenance(kind):
    bv, func = _libc_string_view(
        b"\xe2\x80\x99\0" + b"\1\0\0\0" * 3, name="strcmp", kind=kind)
    note = read_decompile._raw_string_pointer_note(bv, func, _LIBC_TEXT)
    assert note["code"] == ("bytes" if kind == "FunctionSymbol" else "string")
    if kind == "FunctionSymbol":
        assert "bn data retype" not in note["detail"]


@pytest.mark.parametrize("explicit", [False, True])
def test_per_parameter_user_type_is_string_evidence(explicit):
    bv, func = _libc_string_view(
        "’".encode() + b"\0", name="consume_void", kind="FunctionSymbol")
    callee = bv.get_function_at(0x401100)
    var = types.SimpleNamespace(type="char const*")
    callee.parameter_vars = [types.SimpleNamespace(type="int"), var]
    callee.is_var_user_defined = lambda candidate: explicit and candidate is var
    note = read_decompile._raw_string_pointer_note(bv, func, _LIBC_TEXT)
    assert note["code"] == ("string" if explicit else "bytes")


@pytest.mark.parametrize("data_type", ["int32_t[4]", "char[1]", "struct Record"])
def test_user_data_type_is_never_advised_over(data_type):
    bv, func = _libc_string_view("’".encode() + b"\0", name="strcmp")
    bv.get_data_var_at = lambda address: types.SimpleNamespace(
        type=data_type, auto_discovered=False)
    note = read_decompile._raw_string_pointer_note(bv, func, _LIBC_TEXT)
    assert note["strings"][0]["value"] == "’"
    assert "bn data retype" not in note["summary"] + note["detail"]


@pytest.mark.parametrize("value", [" ", "’", "٬", "é", "😀"])
@pytest.mark.parametrize("argument", ["0x661bd8", "&data_661bd8"])
def test_strcmp_names_short_utf8_with_byte_sized_retype_1017(value, argument):
    bv, func = _libc_string_view(value.encode("utf-8") + b"\0", name="strcmp")
    note = read_decompile._raw_string_pointer_note(
        bv, func, f"return strcmp(arg1, {argument});")
    assert note is not None and note["strings"][0]["value"] == value
    assert note["address"] == "0x51c884"
    assert f"'char[{len(value.encode('utf-8')) + 1}]'" in note["detail"]


@pytest.mark.parametrize("memory", [b"\xff\0", b"\xc3\0", b"\xc0\xaf\0", b"\x7f\0",
                                    b"\x1b\0", "\u202e".encode() + b"\0"])
def test_strcmp_does_not_present_invalid_utf8_or_controls_as_literals_1017(memory):
    bv, func = _libc_string_view(memory, name="strcmp")
    assert read_decompile._raw_string_pointer_note(bv, func, _LIBC_TEXT) is None


def test_offset_types_resolve_by_rendered_name_like_types_show_1000():
    # BN reads `get_type_by_name("std::_Rb_tree_node_base")` as one name part
    # and misses the type, which `types show` finds by its rendered name; the
    # note then called a defined type missing (#1000 review round 8).
    types_by_name = {"std::_Rb_tree_node_base": object(),
                     "std::map<std::string, int>::VTable": object()}
    bv = types.SimpleNamespace(get_type_by_name=lambda name: None,
                               types=types_by_name)
    text = ("    void* __offset(std::_Rb_tree_node_base, 0x10) a;\n"
            "    void* __offset(std::map<std::string, int>, 0x8) b;")
    warning = read_decompile._offset_warning(bv, _offset_func("demo::Store"), text)
    assert warning["types"]["defined"] == ["std::_Rb_tree_node_base"]
    assert warning["types"]["missing"] == ["std::map<std::string, int>", "demo::Store"]
    assert "bn struct show 'std::map<std::string, int>'" in warning["detail"]


@pytest.mark.parametrize("names", [
    ["demo::" + "Template" * 30],
    ["demo::" + "Template" * 20, "vtable_for_demo::" + "Template" * 20,
     "known::" + "Template" * 20],
    [f"demo::Type{index}" for index in range(100)],
])
def test_offset_summary_caps_total_length_without_truncating_names_1019(names):
    bv = types.SimpleNamespace(
        get_type_by_name=lambda name: object() if name.startswith("known::") else None,
        types={})
    text = "\n".join(f"__offset({name}, 8)" for name in names)
    row = read_decompile._offset_warning(
        bv, types.SimpleNamespace(start=0x401000, parameter_vars=[]), text)
    assert len(row["summary"]) <= 240
    assert row["address"] == "0x401000"
    assert set(row["types"]["defined"] + row["types"]["missing"]) == set(names)
    for name in names:
        assert name in row["detail"]
        if name not in row["summary"]:
            assert name[:min(20, len(name))] not in row["summary"]
    assert "more" in row["summary"]


def test_resolved_branch_facts_include_separate_starts_not_guessed_targets_1019():
    bv, func = _jump_case()
    other = types.SimpleNamespace(start=0x400800, name="dispatch_case")
    bv.get_function_at = lambda address: other if address == other.start else None
    func.indirect_branches = [
        types.SimpleNamespace(source_addr=0x40058c, dest_addr=address)
        for address in (0x400580, 0x400800, 0x400800)]
    [row] = read_decompile._resolved_branch_warnings(bv, func)
    assert row["address"] == "0x40058c"
    assert row["targets"] == [
        {"address": "0x400580"},
        {"address": "0x400800",
         "function": {"address": "0x400800", "name": "dispatch_case"}},
    ]
    assert "0x400800" in row["summary"] and "dispatch_case" in row["detail"]
    func.indirect_branches = []
    assert read_decompile._resolved_branch_warnings(bv, func) == []


def test_warning_address_uses_first_site_and_preserves_all_sites_1019():
    select = read_decompile._warning_row(
        "select", "summary", "detail", addresses=["0x401024", "0x401030"])
    arity = read_decompile._warning_row(
        "arity", "summary", "detail",
        calls=[{"address": "0x401040"}, {"address": "0x401050"}])
    assert select["address"] == select["addresses"][0] == "0x401024"
    assert arity["address"] == arity["calls"][0]["address"] == "0x401040"


def test_jump_table_guidance_uses_generic_symbol_and_address_placeholders_1019():
    import re

    bv, func = _jump_case()
    [finding] = read_decompile._unresolved_jump_findings(bv, func)
    detail = read_decompile._unresolved_jump_warning(func, finding, False)["detail"]
    assert "bn xrefs <symbol>" in detail
    assert "bn evidence table <address>" in detail
    assert not re.search(r"\b(?:data_[0-9a-f]+|a\.1)\b", detail)
