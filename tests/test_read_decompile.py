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

read_decompile = importlib.import_module("bn_agent_bridge.read_decompile")
read_evidence = importlib.import_module("bn_agent_bridge.read_evidence")


def test_thunk_veneer_warning_names_target_and_is_quiet_for_non_thunks(monkeypatch):
    # #446: a PLT/GOT veneer decompiles as apparent self-recursion; the warning
    # names the real trampoline target (or a generic note) instead.
    monkeypatch.setattr(read_evidence, "_function_thunk_summary",
                        lambda ctx, bv, func: {"is_candidate": True,
                                               "target": {"name": "memcpy", "address": "0x1000"}})
    w = read_decompile._thunk_veneer_warning(None, None, None)
    assert "thunk/veneer -> memcpy @ 0x1000" in w and "self-recursive" in w

    monkeypatch.setattr(read_evidence, "_function_thunk_summary",
                        lambda ctx, bv, func: {"is_candidate": True, "target": None})
    w2 = read_decompile._thunk_veneer_warning(None, None, None)
    assert "thunk/veneer" in w2 and "trampoline" in w2

    monkeypatch.setattr(read_evidence, "_function_thunk_summary",
                        lambda ctx, bv, func: {"is_candidate": False})
    assert read_decompile._thunk_veneer_warning(None, None, None) is None


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
    # No disassembly entries -> _disasm_entry returns "" (the fake _FakeArch has
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
    function.basic_blocks = [_FakeBasicBlock(0x401000, 0x401002)]
    function.comment = "inherited function note"
    bv = _FakeBV(
        functions=[function],
        comments={0x401000: "inherited address note"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(
        bridge.il_format,
        "_decompile_text",
        lambda *args, **kwargs: (
            "void parse_record() {\n"
            "    // inherited function note\n"
            "    // inherited address note\n"
            "}"
        ),
    )

    redacted = instance._decompile("active", "parse_record")
    included = instance._decompile(
        "active", "parse_record", include_annotations=True
    )

    assert "inherited function note" not in redacted["text"]
    assert "inherited address note" not in redacted["text"]
    assert redacted["comments"] == {}
    assert redacted["annotation_summary"] == {
        "comment_count": 2,
        "redacted": True,
    }
    assert "inherited function note" in included["text"]
    assert "inherited address note" in included["text"]
    assert included["comments"] == {"0x401000": "inherited address note"}


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
        "    // second line\n"
        "    do_thing();\n"
        "}"
    )
    monkeypatch.setattr(
        bridge.il_format, "_decompile_text", lambda *args, **kwargs: text
    )

    result = instance._decompile("active", "parse_record")

    assert "first line" not in result["text"]
    assert "second line" not in result["text"]
    assert result["text"].count("// <annotation redacted>") == 2
    assert "do_thing();" in result["text"]


def test_redact_rendered_annotations_preserves_address_gutter():
    # The `--addresses` gutter form: a 0x-prefixed address, whitespace, then
    # the rendered line. Redaction must preserve the gutter and the `//`
    # prefix, rewriting only the comment body.
    text = (
        "0x401000        // secret note\n"
        "0x401004            value = 1;"
    )
    redacted = read_decompile._redact_rendered_annotations(text, ["secret note"])
    assert redacted == (
        "0x401000        // <annotation redacted>\n"
        "0x401004            value = 1;"
    )


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


def test_backward_slice_out_param_caveat_only_interprocedural(monkeypatch):
    """Without --interprocedural the out-param caveat does not fire; the load reads
    as a plain memory_load, preserving prior behavior."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _out_param_fixture()
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.read_taint_slice, "_callee_display_name",
                        lambda ctx, bv, ins: "parse_record")

    result = instance._backward_slice("active", "caller", "0x2030", arg_index=1)

    assert result["trace"][0]["reason"] != "interprocedural_out_param_not_followed"


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
    ins = types.SimpleNamespace(address=0x1000, possible_values=_pvs("UndeterminedValue"), src=src)
    bridge, instance = _dataflow_values_instance(monkeypatch, ins)
    res = instance._possible_values(None, "f", "0x1000")
    assert res["value_basis"] == "source_expression"
    assert res["possible_values"]["type"] == "ConstantValue"
    assert res["possible_values"]["value"] == 0xc48


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
    return _FakeBV(functions=[fn]), fn


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
               _FakeCFGEdge(None, "IndirectBranch")],  # unresolved: must be dropped
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
        {"a": "0x401000", "t": "cmp eax, 0x0"},
        {"a": "0x401004", "t": "je 0x401010"},
    ]
    # The edge whose target is None (indirect/unresolved) is dropped, not rendered.
    assert blocks[0]["edges"] == [{"to": "0x401010", "k": "TrueBranch"}]
    assert blocks[1]["edges"] == []


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
    assert blocks[0]["edges"] == [{"to": "0x2", "k": "FalseBranch"}]
    assert blocks[0]["insns"][0]["a"] == "0x401000"
    assert blocks[1]["insns"][0]["a"] == "0x401000"


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
    assert result["blocks"][0]["insns"] == [{"a": "0x401000", "t": "return 0"}]


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
    bv = _FakeBV(functions=[fn])
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
