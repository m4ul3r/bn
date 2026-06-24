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
    assert "00001000" in res["text"] and "nop" in res["text"]
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
        "00401000        int32_t player_update(int32_t arg1)\n"
        "\n"
        "00401004            return arg1 + 1;"
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
            [(0x401000, "}")],
        ],
    )

    result = instance._decompile("active", "big_fn")

    assert result["analysis_skipped"] is False
    assert any("incomplete stub" in w for w in result["warnings"])


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
    # The dogfood's most-repeated friction: no size field forces per-function
    # info loops / write-locked py exec to find large functions. Expose `size`
    # on every row and a `--sort size` that ranks largest-first.
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
    assert [r["name"] for r in ranked["items"]] == ["big_fn", "mid_fn", "small_fn"]


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
    assert [r["name"] for r in res["items"]] == ["big_fn", "small_fn"]
    res2 = bridge._bind_search_functions(instance, {"query": "_fn", "sort": "size"}, "active")
    assert [r["name"] for r in res2["items"]] == ["big_fn", "small_fn"]


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
    # alongside the back-compat `count` (#275).
    assert instance._list_functions(None, count_only=True) == {
        "kind": "functions", "count": 3, "total": 3}
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

    with pytest.raises(bridge.OperationFailure) as exc2:
        instance._backward_slice("active", "logger_caller", "0x10010", arg_index=5)
    assert "STACK" not in str(exc2.value)   # arg 5 < 6 regs -> not stack-passed, no note


def test_callgraph_result_carries_kind_envelope(monkeypatch):
    """dataflow callgraph JSON must carry kind:'callgraph' so a consumer of the
    {kind, ...} family can identify it, instead of a bare {function, callees,
    callers} off-envelope object (#371.2)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    func = types.SimpleNamespace(name="f", start=0x1000, caller_sites=[])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda sel: object())
    monkeypatch.setattr(instance.ctx, "_find_function", lambda bv, ident: func)

    result = bridge.read_decompile._resolved_calls(
        instance.ctx, None, "f", direction="callers")
    assert result["kind"] == "callgraph"
    assert result["function"]["name"] == "f"
    assert "callers" in result
