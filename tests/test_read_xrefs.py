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


def test_xrefs_rejects_unmapped_raw_address(monkeypatch):
    """A raw address that isn't mapped is a typo/stale value, not a real
    '0 callers' result; reject it (like read/decompile, exit 2) instead of
    returning a false-negative empty xref set with exit 0 (#374)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x401000, "caller")])
    bv.is_valid_offset = lambda addr: False
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    with pytest.raises(RuntimeError, match="not mapped"):
        instance._xrefs(None, "0xdeadbeef")


def test_xrefs_unmapped_but_referenced_address_returns_refs(monkeypatch):
    """An address that is unmapped (is_valid_offset False) but that BN holds real
    refs FOR must still return those refs, never be rejected as 'not mapped'
    (#374 follow-up). The canonical case is 0x0, the placeholder BN records for
    unresolved indirect-call sites -- rejecting it would discard the real
    'where are the unresolved indirect calls' answer."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x401000, "caller")
    bv = _FakeBV(
        functions=[caller],
        code_refs={0x0: [_FakeCodeRef(0x401010, caller)]},
        sections={".text": _FakeSection(".text", 0x400000, 0x410000)},
        segments={0x401010: _FakeSegment(readable=True, executable=True)},
    )
    bv.is_valid_offset = lambda addr: False  # 0x0 is never a valid offset
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._xrefs(None, "0x0")
    assert result["kind"] == "xrefs"
    assert result["code_ref_count"] == 1
    assert result["total"] == 1


def test_xrefs_mapped_address_with_no_refs_stays_clean(monkeypatch):
    """A MAPPED address with zero refs must remain a clean total:0 result -- only
    the genuinely-unmapped case is rejected, never a mapped-but-unreferenced
    address (#374)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[_FakeFunction(0x401000, "caller")],
        code_refs={}, data_refs={},
        sections={".rodata": _FakeSection(".rodata", 0x5000, 0x7000)},
        segments={0x5000: _FakeSegment(readable=True)},
    )
    bv.is_valid_offset = lambda addr: True
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._xrefs(None, "0x5000")
    assert result["kind"] == "xrefs"
    assert result["total"] == 0


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

    assert result["kind"] == "xrefs"
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
    assert result["kind"] == "messages"  # #275
    match = result["items"][0]
    assert match["type_string"]["value"] == "common.HeadUnitInfo"
    assert match["xrefs"]["code_refs"][0]["function"] == "build_type_name"
    assert match["metadata_table_windows"][0]["address"] == "0x6000"
    assert match["metadata_table_windows"][0]["kind"] == "pointer_table"  # #275: embedded table canonical
    assert match["metadata_table_windows"][0]["items"][0]["target"]["thumb_adjusted"] is True
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

    # #275: unified items envelope (each ref tagged code|data); legacy
    # code_refs/data_refs arrays dropped; `field` metadata retained.
    assert result["kind"] == "field_xrefs"
    assert "code_refs" not in result and "data_refs" not in result
    code_items = [it for it in result["items"] if it["kind"] == "code"]
    data_items = [it for it in result["items"] if it["kind"] == "data"]
    assert code_items[0]["function"] == "use_field"
    assert code_items[0]["disasm"] == "ldr r0, [r1, #4]"
    assert [{"address": d["address"], "symbol": d["symbol"], "type": d["type"]} for d in data_items] == [
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
    assert res["kind"] == "symbol_presence" and "symbols" not in res
    syms = {s["symbol"]: s for s in res["items"]}
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


# ===================================================================
# #284: adrp page-base over-report filter (AArch64)
# ===================================================================
#
# On AArch64 `adrp xN, <page>` materializes a 4 KB page base. When a function
# starts at a page-aligned address A, BN records every such adrp as a code ref
# to A even though the real target is A + <add/ldr offset>. The filter drops an
# adrp ref iff its paired in-page offset is nonzero; calls/branches/data refs
# and genuine function-pointer takes (offset 0) are always kept.


class _LOp:
    """Minimal LLIL expression node: an operation name + operands."""
    def __init__(self, op, operands=(), **kw):
        self.operation = types.SimpleNamespace(name=op)
        self.operands = list(operands)
        for k, v in kw.items():
            setattr(self, k, v)


def _reg(name):
    return _LOp("LLIL_REG", name=name)


def _const(v):
    return _LOp("LLIL_CONST", constant=v)


def _const_ptr(v):
    return _LOp("LLIL_CONST_PTR", constant=v)


def _set_reg(dest, src, idx):
    n = _LOp("LLIL_SET_REG", operands=[types.SimpleNamespace(name=dest), src],
             instr_index=idx)
    n.dest = types.SimpleNamespace(name=dest)
    n.src = src
    return n


def _adrp(dest, page_base, idx=0):
    return _set_reg(dest, _const_ptr(page_base), idx)


def _spurious(monkeypatch, adrp_il, following, page_base):
    bridge = _load_bridge(monkeypatch)
    return bridge.read_xrefs._adrp_pagebase_is_spurious(adrp_il, following, page_base)


def test_adrp_pagebase_add_nonzero_offset_is_spurious(monkeypatch):
    A = 0x438000
    adrp = _adrp("x0", A, 0)
    add = _set_reg("x0", _LOp("LLIL_ADD", [_reg("x0"), _const(0x350)]), 1)
    assert _spurious(monkeypatch, adrp, [add], A) is True


def test_adrp_pagebase_add_zero_offset_is_genuine(monkeypatch):
    A = 0x438000
    adrp = _adrp("x0", A, 0)
    add = _set_reg("x0", _LOp("LLIL_ADD", [_reg("x0"), _const(0)]), 1)
    assert _spurious(monkeypatch, adrp, [add], A) is False


def test_adrp_pagebase_used_directly_is_genuine(monkeypatch):
    # `adrp x0, A` then the pointer is used as-is (e.g. stored / passed) -> &fn.
    A = 0x438000
    adrp = _adrp("x0", A, 0)
    use = _set_reg("x1", _reg("x0"), 1)
    assert _spurious(monkeypatch, adrp, [use], A) is False


def test_adrp_pagebase_redefined_before_use_is_genuine(monkeypatch):
    A = 0x438000
    adrp = _adrp("x0", A, 0)
    redef = _set_reg("x0", _reg("x5"), 1)
    assert _spurious(monkeypatch, adrp, [redef], A) is False


def test_adrp_pagebase_register_offset_addend_is_genuine(monkeypatch):
    # `adrp x3, A` then `add x3, x3, x4` (a register, not a const) computes a
    # dynamic in-page target (a table index); the offset can't be resolved
    # statically, so it is conservatively KEPT -- never a false-negative drop.
    A = 0x438000
    adrp = _adrp("x3", A, 0)
    add = _set_reg("x3", _LOp("LLIL_ADD", [_reg("x3"), _reg("x4")]), 1)
    assert _spurious(monkeypatch, adrp, [add], A) is False


def test_adrp_pagebase_load_with_offset_is_spurious(monkeypatch):
    # `adrp x0, A` then `ldr x1, [x0, #0x40]` -> reads A+0x40, not A.
    A = 0x438000
    adrp = _adrp("x0", A, 0)
    ld = _set_reg("x1", _LOp("LLIL_LOAD", [_LOp("LLIL_ADD", [_reg("x0"), _const(0x40)])]), 1)
    assert _spurious(monkeypatch, adrp, [ld], A) is True


def test_adrp_pagebase_offset_after_unrelated_instr_is_spurious(monkeypatch):
    # The paired add can be a couple instructions later (an unrelated mov between).
    A = 0x438000
    adrp = _adrp("x3", A, 0)
    mov = _set_reg("x4", _reg("x22"), 1)
    add = _set_reg("x3", _LOp("LLIL_ADD", [_reg("x3"), _const(0x350)]), 2)
    assert _spurious(monkeypatch, adrp, [mov, add], A) is True


def test_non_setreg_ref_is_never_spurious(monkeypatch):
    # A bl/call to a page-aligned address is a genuine reference, not an adrp.
    A = 0x438000
    call = _LOp("LLIL_CALL", [_const_ptr(A)], instr_index=0)
    assert _spurious(monkeypatch, call, [], A) is False


def test_setreg_const_not_pagebase_is_not_spurious(monkeypatch):
    # SET_REG to a constant that isn't the queried page base -> not our pattern.
    adrp = _adrp("x0", 0x439000, 0)
    add = _set_reg("x0", _LOp("LLIL_ADD", [_reg("x0"), _const(0x350)]), 1)
    assert _spurious(monkeypatch, adrp, [add], 0x438000) is False


def _adrp_caller_fn(adrp_addr, add_addr, page_base, call_addr):
    """A fake function exposing get_low_level_il_at for one adrp+add pair (a
    spurious page-base ref) and one direct call (a genuine ref)."""
    adrp = _adrp("x0", page_base, 0)
    adrp.address = adrp_addr
    add = _set_reg("x0", _LOp("LLIL_ADD", [_reg("x0"), _const(0xc00)]), 1)
    add.address = add_addr
    adrp.il_basic_block = [adrp, add]
    add.il_basic_block = [adrp, add]
    call = _LOp("LLIL_CALL", [_const_ptr(page_base)], instr_index=0)
    call.address = call_addr
    call.il_basic_block = [call]
    by_addr = {adrp_addr: adrp, add_addr: add, call_addr: call}

    class _Fn:
        start = 0x43f000
        name = "caller"
        def get_low_level_il_at(self, addr):
            return by_addr.get(int(addr))

    return _Fn()


def test_xrefs_to_address_drops_spurious_adrp_pagebase(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    A = 0x438000  # page-aligned function start
    fn = _adrp_caller_fn(0x43f1a0, 0x43f1a4, A, 0x440e28)
    bv = _FakeBV(
        code_refs={A: [_FakeCodeRef(0x43f1a0, fn), _FakeCodeRef(0x440e28, fn)]},
        disassembly={0x43f1a0: "adrp    x0, 0x438000",
                     0x440e28: "bl      0x438000",
                     A: "stp     x29, x30, [sp, #-0x10]!"},
        segments={0x43f1a0: _FakeSegment(readable=True, executable=True),
                  0x440e28: _FakeSegment(readable=True, executable=True),
                  A: _FakeSegment(readable=True, executable=True)},
    )
    result = instance._xrefs_to_address(bv, A)
    # the spurious adrp page-base ref is dropped; only the real bl call survives
    assert result["code_ref_count"] == 1
    addrs = [r["address"] for r in result["code_refs"]]
    assert addrs == ["0x440e28"]


def test_xrefs_to_address_no_filter_when_not_page_aligned(monkeypatch):
    # A non-page-aligned target can't be an adrp page base -> no filtering runs,
    # even for an adrp-disassembled ref.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    A = 0x438010  # NOT page-aligned
    bv = _FakeBV(
        code_refs={A: [_FakeCodeRef(0x43f1a0, None)]},
        disassembly={0x43f1a0: "adrp    x0, 0x438000", A: "nop"},
        segments={0x43f1a0: _FakeSegment(readable=True, executable=True),
                  A: _FakeSegment(readable=True)},
    )
    result = instance._xrefs_to_address(bv, A)
    assert result["code_ref_count"] == 1


# ===================================================================
# #286: union same-name PLT/extern stub callers into the real body's xrefs
# ===================================================================
#
# For an exported function in a shared object, intra-library calls route through
# a same-name PLT stub (an ImportedFunctionSymbol) while the real body (a
# FunctionSymbol) shows zero code callers. xrefs of the body must union the
# stub's callers. The stub is identified by symbol type -- the stable signal
# _resolve_impl_over_stub already trusts (BN's is_thunk flag is analysis-timing
# dependent and unreliable).


def _impl_stub_bv(*, impl_ref_addrs=(), stub_ref_addrs=()):
    caller = _FakeFunction(0x500000, "caller")
    stub = _FakeFunction(0x40f1a0, "get_param")
    stub.symbol = _FakeSymbol("ImportedFunctionSymbol")
    impl = _FakeFunction(0x5c40, "get_param")
    impl.symbol = _FakeSymbol("FunctionSymbol")
    bv = _FakeBV(
        functions=[caller, stub, impl],
        code_refs={0x40f1a0: [_FakeCodeRef(a, caller) for a in stub_ref_addrs],
                   0x5c40: [_FakeCodeRef(a, caller) for a in impl_ref_addrs]},
        segments={0x500010: _FakeSegment(readable=True, executable=True),
                  0x500020: _FakeSegment(readable=True, executable=True)},
    )
    return bv, caller, stub, impl


def test_same_name_stub_functions_identifies_import_stub(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, caller, stub, impl = _impl_stub_bv()
    stubs = instance.ctx._same_name_stub_functions(bv, impl)
    assert [int(f.start) for f in stubs] == [0x40f1a0]
    # querying the stub itself yields no stub-typed sibling (impl is FunctionSymbol)
    assert instance.ctx._same_name_stub_functions(bv, stub) == []


def test_same_name_stub_functions_skips_ambiguous_multi_impl(monkeypatch):
    # Two real bodies share a name plus an import stub: the stub's target is
    # ambiguous, so neither body should absorb the stub's callers (the existing
    # ambiguous-symbol disclosure handles the collision instead).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    stub = _FakeFunction(0x40f1a0, "init"); stub.symbol = _FakeSymbol("ImportedFunctionSymbol")
    a = _FakeFunction(0x5000, "init"); a.symbol = _FakeSymbol("FunctionSymbol")
    b = _FakeFunction(0x6000, "init"); b.symbol = _FakeSymbol("FunctionSymbol")
    bv = _FakeBV(functions=[stub, a, b])
    assert instance.ctx._same_name_stub_functions(bv, a) == []
    assert instance.ctx._same_name_stub_functions(bv, b) == []


def test_xrefs_by_name_unions_stub_callers(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, caller, stub, impl = _impl_stub_bv(stub_ref_addrs=[0x500010])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._xrefs(None, "get_param")
    assert result["address"] == "0x5c40"                  # resolves to the body
    assert result["resolved_to_definition"] == "0x5c40"
    assert result["code_ref_count"] == 1                  # the stub-routed caller
    assert "0x40f1a0" in result.get("stub_callers_via", [])
    assert [it["address"] for it in result["items"] if it["kind"] == "code"] == ["0x500010"]


def test_xrefs_by_body_address_unions_stub_callers(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, caller, stub, impl = _impl_stub_bv(stub_ref_addrs=[0x500010])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._xrefs(None, "0x5c40")              # by body address
    assert result["code_ref_count"] == 1
    assert "0x40f1a0" in result.get("stub_callers_via", [])


def test_xrefs_dedups_when_caller_hits_both_body_and_stub(monkeypatch):
    # A caller that references both the body directly and the stub is counted once.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, caller, stub, impl = _impl_stub_bv(
        impl_ref_addrs=[0x500010], stub_ref_addrs=[0x500010, 0x500020])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._xrefs(None, "0x5c40")
    addrs = sorted(it["address"] for it in result["items"] if it["kind"] == "code")
    assert addrs == ["0x500010", "0x500020"]             # 0x500010 not double-counted


def test_xrefs_no_stub_union_for_plain_function(monkeypatch):
    # A normal function with no same-name stub is completely unaffected.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x500000, "caller")
    fn = _FakeFunction(0x6000, "solo")
    fn.symbol = _FakeSymbol("FunctionSymbol")
    bv = _FakeBV(functions=[caller, fn],
                 code_refs={0x6000: [_FakeCodeRef(0x500010, caller)]},
                 segments={0x500010: _FakeSegment(readable=True, executable=True)})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._xrefs(None, "0x6000")
    assert result["code_ref_count"] == 1
    assert "stub_callers_via" not in result


# ===================================================================
# #286 (callsites half): callsites see through a same-name PLT stub
# ===================================================================


def _callsites_caller_with_stub_call():
    """A caller function whose single call targets the stub at 0x40f1a0."""
    caller = _FakeFunction(0x500000, "caller")
    caller.basic_blocks = [_FakeBasicBlock(0x500010, 0x500014)]
    caller.arch = _FakeArch(lengths={0x500010: 4})
    call = _FakeLLILInstruction(0x500010, _FakeConstPtr(0x40f1a0), operation="LLIL_CALL")
    caller.low_level_il = [[call]]
    return caller


def test_callsites_within_function_matches_stub_target(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    caller = _callsites_caller_with_stub_call()
    impl = _FakeFunction(0x5c40, "get_param"); impl.symbol = _FakeSymbol("FunctionSymbol")
    stub = _FakeFunction(0x40f1a0, "get_param"); stub.symbol = _FakeSymbol("ImportedFunctionSymbol")
    bv = _FakeBV(
        functions=[caller, impl, stub],
        disassembly={0x500010: "bl 0x40f1a0"},
        code_refs={0x5c40: [], 0x40f1a0: [_FakeCodeRef(0x500010, caller)]},
    )
    rows = bridge.read_listing._callsites_within_function(
        None, bv, impl, caller, context=1, stub_addrs={0x40f1a0})
    assert [r["call_addr"] for r in rows] == ["0x500010"]


def test_callsites_within_function_misses_stub_without_union(monkeypatch):
    # Baseline (the #286 bug): with no stub addresses, a call that targets the
    # stub is not matched against the body -> no callsites found.
    bridge = _load_bridge(monkeypatch)
    caller = _callsites_caller_with_stub_call()
    impl = _FakeFunction(0x5c40, "get_param"); impl.symbol = _FakeSymbol("FunctionSymbol")
    bv = _FakeBV(
        functions=[caller, impl],
        disassembly={0x500010: "bl 0x40f1a0"},
        code_refs={0x5c40: []},
    )
    rows = bridge.read_listing._callsites_within_function(None, bv, impl, caller, context=1)
    assert rows == []


def _stub_call_fn(start, name, call_addr, target):
    fn = _FakeFunction(start, name)
    fn.basic_blocks = [_FakeBasicBlock(call_addr, call_addr + 4)]
    fn.arch = _FakeArch(lengths={call_addr: 4})
    fn.low_level_il = [[_FakeLLILInstruction(call_addr, _FakeConstPtr(target), operation="LLIL_CALL")]]
    return fn


def test_callsites_full_path_sees_through_stub_and_no_cross_stub_fp(monkeypatch):
    # End-to-end through _callsites: the wiring (_same_name_stub_functions ->
    # stub_addrs) must find a stub-routed call AND must not match a call that
    # targets a DIFFERENT exported function's stub (the critical no-false-positive
    # property -- #286 review).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    impl = _FakeFunction(0x5c40, "get_param"); impl.symbol = _FakeSymbol("FunctionSymbol")
    stub = _FakeFunction(0x40f1a0, "get_param"); stub.symbol = _FakeSymbol("ImportedFunctionSymbol")
    impl2 = _FakeFunction(0x6c40, "other_fn"); impl2.symbol = _FakeSymbol("FunctionSymbol")
    stub2 = _FakeFunction(0x40f1f0, "other_fn"); stub2.symbol = _FakeSymbol("ImportedFunctionSymbol")
    caller = _stub_call_fn(0x500000, "caller", 0x500010, 0x40f1a0)    # calls get_param's stub
    caller2 = _stub_call_fn(0x600000, "caller2", 0x600010, 0x40f1f0)  # calls other_fn's stub
    bv = _FakeBV(
        functions=[impl, stub, impl2, stub2, caller, caller2],
        disassembly={0x500010: "bl 0x40f1a0", 0x600010: "bl 0x40f1f0"},
        code_refs={0x5c40: [], 0x6c40: [],
                   0x40f1a0: [_FakeCodeRef(0x500010, caller)],
                   0x40f1f0: [_FakeCodeRef(0x600010, caller2)]},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    # caller calls get_param (via its stub) -> found
    hit = instance._callsites(None, "get_param", within_identifiers=["caller"])
    assert [r["call_addr"] for r in hit["items"]] == ["0x500010"]
    # caller2 calls a DIFFERENT function's stub -> no match for get_param
    miss = instance._callsites(None, "get_param", within_identifiers=["caller2"])
    assert miss["items"] == []


def test_function_pointer_data_refs_alignment_and_dedup(monkeypatch):
    # #323 core scan: finds a pointer-aligned stored function pointer, skips an
    # unaligned coincidental byte run, and dedups against already-known slots.
    bridge = _load_bridge(monkeypatch)
    rx = bridge.read_xrefs
    ctx = bridge.BinaryNinjaBridge().ctx
    func_addr = 0x401000
    blob = bytearray(0x100)
    blob[0x40:0x48] = func_addr.to_bytes(8, "little")   # aligned slot -> found
    blob[0x51:0x59] = func_addr.to_bytes(8, "little")   # unaligned -> skipped
    bv = _FakeBV(arch=_FakeArch(name="x86_64", address_size=8),
                 sections={".data": _FakeSection(".data", 0x420000, 0x420100)},
                 memory={0x420000: bytes(blob)})
    refs, truncated = rx._function_pointer_data_refs(ctx, bv, func_addr, set())
    slots = [s for s, _name, _thumb in refs]
    assert truncated is False
    assert 0x420040 in slots
    assert 0x420051 not in slots
    refs2, _ = rx._function_pointer_data_refs(ctx, bv, func_addr, {0x420040})
    assert 0x420040 not in [s for s, _n, _t in refs2]   # dedup vs known


def test_evidence_xrefs_backlinks_stored_function_pointer(monkeypatch):
    # #323: a function reached ONLY via a stored function pointer (a data-table
    # slot BN didn't model as a data ref) is back-linked by the fn-pointer scan
    # (fn_pointer_scan=True), so a callback-only function isn't reported as dead.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    func_addr = 0x401000
    target = _FakeFunction(func_addr, "callback_only")
    target.basic_blocks = [_FakeBasicBlock(func_addr, func_addr + 0x10)]  # real body
    blob = bytearray(0x100)
    blob[0x40:0x48] = func_addr.to_bytes(8, "little")
    bv = _FakeBV(
        functions=[target],
        arch=_FakeArch(name="x86_64", address_size=8),
        code_refs={func_addr: []},
        data_refs={func_addr: []},        # BN modeled NO ref -> looks dead
        sections={".data.rel.ro": _FakeSection(".data.rel.ro", 0x420000, 0x420100)},
        segments={0x420040: _FakeSegment(readable=True)},
        memory={0x420000: bytes(blob)},
    )
    plain = instance._xrefs_to_address(bv, func_addr)
    assert plain["data_ref_count"] == 0                 # the bug: looks dead

    scanned = instance._xrefs_to_address(bv, func_addr, fn_pointer_scan=True)
    fp = [r for r in scanned["data_refs"] if r.get("function_pointer")]
    assert len(fp) == 1
    assert fp[0]["address"] == "0x420040"
    assert fp[0]["kind"] == "data"
    assert fp[0]["context"]["sections"][0]["name"] == ".data.rel.ro"


def test_function_pointer_scan_skips_image_base_pseudo_function(monkeypatch):
    # #323 review (FP): the image-base / body-less pseudo-function (e.g. an ELF
    # header BN models as a function at bv.start) must NOT be scanned -- its
    # address is a common .rodata constant (the load base) that would produce
    # only false positives, never a real callback-table slot.
    bridge = _load_bridge(monkeypatch)
    rx = bridge.read_xrefs
    ctx = bridge.BinaryNinjaBridge().ctx
    base = 0x400000
    blob = bytearray(0x80)
    blob[0x10:0x18] = base.to_bytes(8, "little")  # the base word recurs in data
    bv = _FakeBV(arch=_FakeArch(name="x86_64", address_size=8),
                 sections={".rodata": _FakeSection(".rodata", 0x420000, 0x420080)},
                 memory={0x420000: bytes(blob)})
    bv.start = base
    refs, _trunc = rx._function_pointer_data_refs(ctx, bv, base, set())
    assert refs == []  # image-base needle skipped, no FPs


def test_data_section_ranges_skips_metadata_and_bss(monkeypatch):
    # #323 review (LOW): relocation/symbol/unwind metadata and .bss are not
    # pointer-table homes -- skip them (firmware-friendly deny-list, not an
    # allow-list, so custom data section names are still scanned).
    bridge = _load_bridge(monkeypatch)
    rx = bridge.read_xrefs
    bv = _FakeBV(sections={
        ".data.rel.ro": _FakeSection(".data.rel.ro", 0x1000, 0x1100),
        ".rela.dyn": _FakeSection(".rela.dyn", 0x2000, 0x2100),
        ".eh_frame": _FakeSection(".eh_frame", 0x3000, 0x3100),
        ".bss": _FakeSection(".bss", 0x4000, 0x4100),
        ".dynsym": _FakeSection(".dynsym", 0x5000, 0x5100),
        "fw_table": _FakeSection("fw_table", 0x6000, 0x6100),  # custom -> kept
    })
    names = {n for n, _s, _l in rx._data_section_ranges(bv)}
    assert names == {".data.rel.ro", "fw_table"}


def _paging_field_bv(n_code):
    # n_code code refs at 0x1000, 0x1004, ... plus 2 data refs.
    code = {("Hot", 0): [
        types.SimpleNamespace(func=_FakeFunction(0x1000 + 4 * i, f"use_{i}"),
                              address=0x1000 + 4 * i, size=4, incomingType="Hot*")
        for i in range(n_code)]}
    return _FieldRefBV(
        code_refs=code,
        data_refs={("Hot", 0): [0x8000, 0x8008]},
        symbols={},
        data_vars={},
        disassembly={},
    )


def test_field_xrefs_pages_with_limit_and_offset_532(monkeypatch):
    # #532: field xrefs now honor offset/limit and return the canonical paging
    # envelope (offset/limit/returned/has_more/total) like every other xref path,
    # instead of dumping the whole ref set and spilling.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _paging_field_bv(10)   # 10 code + 2 data = 12 total
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(
        bridge.read_xrefs, "_resolve_type_field",
        lambda ctx, view, spec: {"type_name": "Hot", "offset": 0, "field_name": "f"},
    )

    page = instance._field_xrefs("active", "Hot.f", offset=0, limit=5)
    assert page["total"] == 12
    assert page["returned"] == 5
    assert len(page["items"]) == 5
    assert page["offset"] == 0 and page["limit"] == 5
    assert page["has_more"] is True

    # offset skips into the list; last page has no more.
    tail = instance._field_xrefs("active", "Hot.f", offset=10, limit=5)
    assert tail["returned"] == 2          # only the 2 data refs remain
    assert tail["has_more"] is False
    assert [it["kind"] for it in tail["items"]] == ["data", "data"]

    # no limit -> whole set, has_more False.
    full = instance._field_xrefs("active", "Hot.f")
    assert full["returned"] == 12 and full["has_more"] is False


def test_field_xrefs_rejects_invalid_paging_532(monkeypatch):
    # #532: a raw-socket / py-exec caller must not slip a negative offset or a
    # limit<=0 past the op and get Python slice semantics -- same contract as
    # every other paged op (_validate_count).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: object())
    monkeypatch.setattr(
        bridge.read_xrefs, "_resolve_type_field",
        lambda ctx, view, spec: {"type_name": "Hot", "offset": 0, "field_name": "f"},
    )
    import pytest
    with pytest.raises(bridge.OperationFailure):
        instance._field_xrefs("active", "Hot.f", offset=-1)
    with pytest.raises(bridge.OperationFailure):
        instance._field_xrefs("active", "Hot.f", limit=0)
