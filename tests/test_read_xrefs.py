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


def test_xrefs_include_address_context(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x401000, "caller")
    target = _FakeFunction(0x402000, "target")
    bv = _FakeBV(
        functions=[caller, target],
        symbols=[bridge.bn.Symbol(bridge.bn.SymbolType.DataSymbol, 0x5000, "type_name")],
        code_refs={0x5000: [_FakeCodeRef(0x401010, caller)]},
        data_refs={0x5000: [0x6000]},
        disassembly={0x401010: "ldr r0, =type_name"},
        sections={
            ".text": _FakeSection(".text", 0x400000, 0x410000),
            ".rodata": _FakeSection(".rodata", 0x5000, 0x7000),
        },
        segments={
            0x401010: _FakeSegment(readable=True, executable=True),
            0x5000: _FakeSegment(readable=True),
            0x6000: _FakeSegment(readable=True, writable=True),
        },
    )

    result = instance._xrefs_to_address(bv, 0x5000)

    assert result["target_context"]["symbol"]["name"] == "type_name"
    assert result["code_refs"][0]["context"]["disasm"] == "ldr r0, =type_name"
    assert result["code_refs"][0]["context"]["sections"][0]["name"] == ".text"
    assert result["data_refs"][0]["context"]["sections"][0]["name"] == ".rodata"
    # JSON carries the same summary counts the text header shows, so an agent
    # can size/triage without materializing the (spilling) code_refs[] array.
    assert result["code_ref_count"] == 1
    assert result["data_ref_count"] == 1
    assert result["caller_function_count"] == 1


def test_xrefs_to_address_emits_paging_envelope(monkeypatch):
    # #164: xrefs adopts the canonical {items,total,offset,limit,returned,has_more}
    # envelope (items = code refs then data refs, each keeping its kind), pages on
    # offset/limit, and keeps the #140 summary counts + the deprecated dual shape.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x401000, "caller")
    bv = _FakeBV(
        functions=[caller],
        code_refs={0x5000: [_FakeCodeRef(0x401010, caller), _FakeCodeRef(0x401020, caller)]},
        data_refs={0x5000: [0x6000]},
        sections={".text": _FakeSection(".text", 0x400000, 0x410000),
                  ".rodata": _FakeSection(".rodata", 0x5000, 0x7000)},
        segments={0x401010: _FakeSegment(readable=True, executable=True),
                  0x401020: _FakeSegment(readable=True, executable=True),
                  0x6000: _FakeSegment(readable=True, writable=True)},
    )
    full = instance._xrefs_to_address(bv, 0x5000)
    assert full["total"] == 3
    assert full["returned"] == 3
    assert full["has_more"] is False
    assert [it["kind"] for it in full["items"]] == ["code", "code", "data"]
    assert full["code_ref_count"] == 2 and full["data_ref_count"] == 1
    # deprecated dual shape stays full (function-info embeds it unpaged)
    assert len(full["code_refs"]) == 2 and len(full["data_refs"]) == 1

    page = instance._xrefs_to_address(bv, 0x5000, offset=0, limit=2)
    assert page["returned"] == 2 and page["has_more"] is True
    assert [it["kind"] for it in page["items"]] == ["code", "code"]
    assert page["total"] == 3
    # summary counts + dual shape reflect the FULL set regardless of paging
    assert page["code_ref_count"] == 2 and len(page["code_refs"]) == 2


def test_xrefs_op_drops_deprecated_arrays(monkeypatch):
    # #184: the `xrefs` OP response must NOT carry the full code_refs/data_refs
    # arrays -- they rode unbounded past --offset/--limit and spilled the JSON on
    # high-fanout symbols. Keep the full-set summary counts (#140) + the paged
    # `items`. The lower-level _xrefs_to_address still produces the dual shape,
    # which `function info` and evidence message-lensing embed directly (locked by
    # test_xrefs_to_address_emits_paging_envelope above).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x401000, "caller")
    bv = _FakeBV(
        functions=[caller],
        code_refs={0x5000: [_FakeCodeRef(0x401010, caller), _FakeCodeRef(0x401020, caller)]},
        data_refs={0x5000: [0x6000]},
        sections={".text": _FakeSection(".text", 0x400000, 0x410000),
                  ".rodata": _FakeSection(".rodata", 0x5000, 0x7000)},
        segments={0x401010: _FakeSegment(readable=True, executable=True),
                  0x401020: _FakeSegment(readable=True, executable=True),
                  0x6000: _FakeSegment(readable=True, writable=True)},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._xrefs(None, "0x5000", limit=1)

    # deprecated dual arrays are gone -> --limit truly bounds the payload
    assert "code_refs" not in result
    assert "data_refs" not in result
    # full-set summary counts survive paging (the triage signal)
    assert result["code_ref_count"] == 2
    assert result["data_ref_count"] == 1
    assert result["caller_function_count"] == 1
    assert result["total"] == 3
    # items is bounded by --limit
    assert result["returned"] == 1 and len(result["items"]) == 1
    assert result["has_more"] is True
    assert result["items"][0]["kind"] == "code"


def test_xrefs_suppress_disasm_for_data_targets(monkeypatch):
    # ILX #1: a .rodata string target must not be disassembled into garbage,
    # even though firmware ELFs map .rodata into the r-x load segment.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x1A000, "caller")
    caller.basic_blocks = [_FakeBasicBlock(0x1A000, 0x1A100)]
    message = "basic_string::_M_construct null not valid"
    bv = _FakeBV(
        functions=[caller],
        code_refs={0x2A07C: [_FakeCodeRef(0x1A050, caller)]},
        disassembly={0x1A050: "ldr r0, =message"},
        sections={
            ".text": _FakeSection(".text", 0x10000, 0x20000),
            ".rodata": _FakeSection(".rodata", 0x2A000, 0x2B000),
        },
        segments={
            0x1A050: _FakeSegment(readable=True, executable=True),
            0x2A07C: _FakeSegment(readable=True, executable=True),  # rodata shares the r-x segment
        },
        memory={0x2A07C: message.encode() + b"\x00"},
    )

    result = instance._xrefs_to_address(bv, 0x2A07C)

    target = result["target_context"]
    assert target["kind"] == "string"
    assert target["string"]["value"] == message
    assert target["disasm"] is None
    assert target["notes"]
    # the referencing instruction is genuine code, so its disasm is kept
    assert result["code_refs"][0]["context"]["disasm"] == "ldr r0, =message"


def test_xrefs_resolve_multiline_strings_and_mark_truncation(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x401000, "usage")
    caller.basic_blocks = [_FakeBasicBlock(0x401000, 0x401100)]
    message = "Usage: %s [OPTION]... PATTERNS [FILE]...\n" + ("A" * 120)
    bv = _FakeBV(
        functions=[caller],
        code_refs={0x427840: [_FakeCodeRef(0x40EA7C, caller)]},
        disassembly={0x40EA7C: "lea rsi, [rel 0x427840]"},
        sections={
            ".text": _FakeSection(".text", 0x401000, 0x402000),
            ".rodata": _FakeSection(".rodata", 0x427000, 0x428000),
        },
        segments={
            0x40EA7C: _FakeSegment(readable=True, executable=True),
            0x427840: _FakeSegment(readable=True),
        },
        memory={0x427840: message.encode() + b"\x00"},
    )

    result = instance._xrefs_to_address(bv, 0x427840)

    target = result["target_context"]
    assert target["kind"] == "string"
    assert target["string"]["value"] == message[:96]
    assert "\n" in target["string"]["value"]
    assert target["string"]["truncated"] is True
    assert target["disasm"] is None
    assert result["code_refs"][0]["context"]["disasm"] == "lea rsi, [rel 0x427840]"


def test_message_lens_summarizes_type_string_xrefs_and_metadata_window(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    builder = _FakeFunction(0x586A2, "build_type_name")
    memory = {0x6000: (0x586A3).to_bytes(4, "little") + (0x7000).to_bytes(4, "little")}
    bv = _FakeBV(
        functions=[builder],
        arch=_FakeArch(name="armv7"),
        strings=[_FakeStringRef(0x175B20, 19, "common.HeadUnitInfo")],
        code_refs={0x175B20: [_FakeCodeRef(0x586C0, builder)]},
        data_refs={0x175B20: [0x6008]},
        disassembly={0x586C0: "adr r1, common.HeadUnitInfo"},
        memory=memory,
    )
    # _message_lens now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._message_lens("active", "HeadUnitInfo", limit=5, table_entries=2)

    assert result["count"] == 1
    match = result["matches"][0]
    assert match["type_string"]["value"] == "common.HeadUnitInfo"
    assert match["xrefs"]["code_refs"][0]["function"] == "build_type_name"
    assert match["metadata_table_windows"][0]["address"] == "0x6000"
    assert match["metadata_table_windows"][0]["entries"][0]["target"]["thumb_adjusted"] is True
    # single match under the limit: honest total, not truncated
    assert result["total"] == 1
    assert result["truncated"] is False


# --- xrefs import symbol resolution ---


def test_xrefs_falls_back_to_import_symbol_when_function_not_found(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    malloc_sym = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x20000, "malloc")
    malloc_sym.short_name = "malloc"
    malloc_sym.namespace = "libc"

    bv = _FakeBV(
        functions=[_FakeFunction(0x10000, "main")],
        symbols=[malloc_sym],
    )
    # _xrefs now resolves the view through the BridgeContext seam (read_xrefs).
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._xrefs(None, "malloc")

    assert result["import_resolved"] is True
    assert result["import_name"] == "malloc"
    assert result["address"] == "0x20000"


def test_xrefs_demangled_name_resolves_to_definition_not_veneer(monkeypatch):
    """A demangled C++ name matches an import veneer (PLT stub) via short_name,
    but the same symbol is also DEFINED in this module. xrefs must resolve to the
    real definition, not the stub, so the call-graph matches `xrefs <mangled>` /
    decompile rather than silently returning the veneer's refs (#201)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    MANGLED = "_ZN5proto3Msg6handleEv"
    DEMANGLED = "proto::Msg::handle"
    # the PLT import veneer (matched by the demangled short_name)
    veneer = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x403380, MANGLED)
    veneer.short_name = DEMANGLED
    veneer.raw_name = MANGLED
    veneer.namespace = "BNINTERNALNAMESPACE"
    # the real function body, defined in this module
    impl = _FakeFunction(0x405250, MANGLED)
    impl.symbol = _FakeSymbol("FunctionSymbol")

    bv = _FakeBV(functions=[impl], symbols=[veneer])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._xrefs(None, DEMANGLED)
    assert result["address"] == "0x405250"               # the definition, not 0x403380
    assert result["resolved_to_definition"] == "0x405250"
    assert result["import_resolved"] is True


def test_xrefs_import_symbol_raises_for_unknown_symbol(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x10000, "main")])
    # _xrefs now resolves the view through the BridgeContext seam (read_xrefs).
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(RuntimeError, match="Function not found: nonexistent"):
        instance._xrefs(None, "nonexistent")


# ---------------------------------------------------------------------------
# xrefs: ambiguous function identifiers must not degrade to "not found"
# ---------------------------------------------------------------------------


def test_xrefs_reraises_ambiguous_function_identifier(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "duplicate_name"),
            _FakeFunction(0x402000, "duplicate_name"),
        ]
    )
    # _xrefs now resolves the view through the BridgeContext seam (read_xrefs).
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(RuntimeError, match="Ambiguous function identifier"):
        instance._xrefs(None, "duplicate_name")


def test_field_xrefs_resolves_data_var_type(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    caller = _FakeFunction(0x1000, "use_field")
    code_ref = types.SimpleNamespace(func=caller, address=0x1010, size=4, incomingType="int32_t")
    bv = _FieldRefBV(
        code_refs={("Foo", 4): [code_ref]},
        data_refs={("Foo", 4): [0x2000, 0x3000]},
        symbols={0x2000: types.SimpleNamespace(name="g_foo")},
        # 0x2000 has a data var (type resolves); 0x3000 has none (type -> None).
        data_vars={0x2000: types.SimpleNamespace(type="struct Foo")},
        disassembly={0x1010: "ldr r0, [r1, #4]"},
    )

    # _field_xrefs now resolves the view through the BridgeContext seam and calls
    # the module-level _resolve_type_field directly (read_xrefs), so patch both
    # where the moved free function reaches them.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(
        bridge.read_xrefs,
        "_resolve_type_field",
        lambda ctx, view, spec: {"type_name": "Foo", "offset": 4, "field_name": "bar"},
    )

    # Must not raise (the old get_type_at call would AttributeError here).
    result = instance._field_xrefs("active", "Foo.bar")

    assert result["code_refs"][0]["function"] == "use_field"
    assert result["code_refs"][0]["disasm"] == "ldr r0, [r1, #4]"
    assert result["data_refs"] == [
        {"address": "0x2000", "symbol": "g_foo", "type": "struct Foo"},
        {"address": "0x3000", "symbol": None, "type": None},
    ]


def test_xrefs_requires_refresh_when_quick_loaded(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # Quick-loaded: code-ref analysis hasn't run, so a 0/0 result reads as
    # "no xrefs" rather than "not analyzed". Refuse with a directive instead.
    bridge._quick_loaded_views.add(bv)
    with pytest.raises(RuntimeError, match="loaded with --quick"):
        instance._xrefs(None, "main")
    bridge._quick_loaded_views.discard(bv)


def test_xrefs_any_marks_ambiguous_symbol_present(monkeypatch):
    """In a sink sweep an AMBIGUOUS symbol (resolves to >=2 bodies) must be
    reported present (it exists), not absent -- otherwise a real sink reads as
    unlinked (#218 review)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x401000, "dup"), _FakeFunction(0x402000, "dup")])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    res = instance._xrefs_any(None, ["dup", "nope"])
    syms = {s["symbol"]: s for s in res["symbols"]}
    assert syms["dup"]["present"] is True and syms["dup"].get("ambiguous") is True
    assert syms["nope"]["present"] is False
    assert res["present"] == 1


def test_xrefs_thunk_real_collision_surfaces_ambiguity_and_picks_hot(monkeypatch):
    """A bare name that resolves to a 16-byte thunk AND the real body must not
    silently pick the zero-caller member: surface both under `ambiguous_symbol`
    and report xrefs for the member carrying the call traffic (#220)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x500000, "caller")
    thunk = _FakeFunction(0x440030, "util_free")    # PLT-style thunk: hot
    thunk.is_thunk = True
    real = _FakeFunction(0x4d2e70, "util_free")     # real body: zero direct callers
    real.symbol = _FakeSymbol("FunctionSymbol")
    bv = _FakeBV(
        functions=[caller, thunk, real],
        code_refs={0x440030: [_FakeCodeRef(0x500010, caller), _FakeCodeRef(0x500020, caller)],
                   0x4d2e70: []},
        sections={".text": _FakeSection(".text", 0x400000, 0x500000)},
        segments={0x500010: _FakeSegment(readable=True, executable=True),
                  0x500020: _FakeSegment(readable=True, executable=True)},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._xrefs(None, "util_free")
    amb = result["ambiguous_symbol"]
    assert amb["resolved_to"] == "0x440030"                       # the hot member
    assert {m["address"] for m in amb["members"]} == {"0x440030", "0x4d2e70"}
    assert result["address"] == "0x440030"
    assert result["code_ref_count"] == 2


def test_xrefs_demangled_collision_prefers_definition_over_import_veneer(monkeypatch):
    """The #201 ⊕ #220 intersection: a demangled name matches BOTH the real body
    (FunctionSymbol) and a PIC import veneer (ImportedFunctionSymbol, is_thunk) --
    both present in bv.functions with the demangled short_name. xrefs must resolve
    to the DEFINITION, not the ref-carrying stub (the #220 ref-count tiebreak must
    not regress #201)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    DEMANGLED = "proto::Msg::handle"
    caller = _FakeFunction(0x500000, "caller")

    veneer = _FakeFunction(0x401050, "_ZN5proto3Msg6handleEv")   # PLT veneer: hot
    veneer.is_thunk = True
    vsym = _FakeSymbol("ImportedFunctionSymbol")
    vsym.short_name = DEMANGLED
    veneer.symbol = vsym

    impl = _FakeFunction(0x40114a, "_ZN5proto3Msg6handleEv")     # real body: 0 direct callers
    isym = _FakeSymbol("FunctionSymbol")
    isym.short_name = DEMANGLED
    impl.symbol = isym

    bv = _FakeBV(
        functions=[caller, veneer, impl],
        code_refs={0x401050: [_FakeCodeRef(0x500010, caller)], 0x40114a: []},
        sections={".text": _FakeSection(".text", 0x400000, 0x500000)},
        segments={0x500010: _FakeSegment(readable=True, executable=True)},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._xrefs(None, DEMANGLED)
    assert result["address"] == "0x40114a"               # the definition, NOT the stub
    assert result["resolved_to_definition"] == "0x40114a"
    assert "ambiguous_symbol" not in result              # stub-vs-impl, not a thunk/real collision


def test_find_function_resolves_demangled_via_symbol_short_name(monkeypatch):
    """A function whose `fn.name` BN kept mangled resolves by its demangled
    `symbol.short_name`/`full_name`, so callsites/decompile/xrefs all accept the
    same C++ name (#224a)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x405250, "_ZN3foo3bar4recvEi")   # BN kept fn.name mangled
    sym = _FakeSymbol("FunctionSymbol")
    sym.short_name = "foo::bar::recv"
    sym.full_name = "foo::bar::recv(int32_t)"
    fn.symbol = sym
    bv = _FakeBV(functions=[fn])

    assert int(instance._find_function(bv, "foo::bar::recv").start) == 0x405250
    assert int(instance._find_function(bv, "foo::bar::recv(int32_t)").start) == 0x405250
    assert int(instance._find_function(bv, "_ZN3foo3bar4recvEi").start) == 0x405250


def test_xrefs_resolves_data_symbol_by_name(monkeypatch):
    """`xrefs <data-symbol>` resolves a non-function symbol (a global table) to
    its address instead of failing with a misleading import-only error (#224b)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]
    data_sym = fake_bn.Symbol(fake_bn.SymbolType.DataSymbol, 0x56b688, "g_state_table")
    caller = _FakeFunction(0x401000, "user")
    bv = _FakeBV(
        functions=[caller],
        symbols=[data_sym],
        code_refs={0x56b688: [_FakeCodeRef(0x401010, caller)]},
        sections={".text": _FakeSection(".text", 0x400000, 0x410000)},
        segments={0x401010: _FakeSegment(readable=True, executable=True)},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._xrefs(None, "g_state_table")
    assert result["address"] == "0x56b688"
    assert result["resolved_symbol"]["kind"] == "data"
    assert result["code_ref_count"] == 1


