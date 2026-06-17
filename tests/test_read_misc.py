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


# --- I2: strings filtering ---


def test_strings_min_length_excludes_short_strings(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(strings=[
        _FakeStringRef(0x1000, 2, "ab"),
        _FakeStringRef(0x2000, 5, "hello"),
        _FakeStringRef(0x3000, 10, "helloworld"),
    ])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query=None, offset=0, limit=100, min_length=4)

    items = result["items"]
    assert len(items) == 2
    assert result["total"] == 2
    assert items[0]["value"] == "hello"
    assert items[1]["value"] == "helloworld"


def test_strings_section_filter_keeps_only_matching_section(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        strings=[
            _FakeStringRef(0x1000, 4, "code"),
            _FakeStringRef(0x5000, 6, "rodata"),
        ],
        sections={
            ".text": _FakeSection(".text", 0x1000, 0x2000),
            ".rodata": _FakeSection(".rodata", 0x5000, 0x6000),
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query=None, offset=0, limit=100, section=".rodata")

    items = result["items"]
    assert len(items) == 1
    assert items[0]["value"] == "rodata"


def test_strings_no_crt_excludes_locale_and_text_section(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        strings=[
            _FakeStringRef(0x1000, 2, "en"),           # locale code
            _FakeStringRef(0x2000, 5, "en-US"),         # locale code
            _FakeStringRef(0x3000, 3, "Mon"),           # day abbreviation
            _FakeStringRef(0x4000, 5, "UTF-8"),         # encoding name
            _FakeStringRef(0x5000, 6, "player"),        # real string
            _FakeStringRef(0x6000, 4, "data"),          # in .text section
        ],
        sections={
            ".text": _FakeSection(".text", 0x6000, 0x7000),
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query=None, offset=0, limit=100, no_crt=True)

    items = result["items"]
    assert len(items) == 1
    assert items[0]["value"] == "player"


def test_strings_filters_combine(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        strings=[
            _FakeStringRef(0x5000, 2, "ab"),            # too short
            _FakeStringRef(0x5001, 6, "player"),        # passes all
            _FakeStringRef(0x1000, 6, "system"),        # wrong section
            _FakeStringRef(0x5002, 5, "en-US"),         # CRT locale
        ],
        sections={
            ".text": _FakeSection(".text", 0x1000, 0x2000),
            ".rodata": _FakeSection(".rodata", 0x5000, 0x6000),
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query=None, offset=0, limit=100,
                               min_length=4, section=".rodata", no_crt=True)

    items = result["items"]
    assert len(items) == 1
    assert items[0]["value"] == "player"


def test_strings_regex_search_matches_or_patterns(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        strings=[
            _FakeStringRef(0x5000, 7, "Vehicle"),
            _FakeStringRef(0x5010, 12, "HeadUnitInfo"),
            _FakeStringRef(0x5020, 4, "skip"),
        ]
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query="vehicle|headunit", offset=0, limit=100, regex=True)

    assert [item["value"] for item in result["items"]] == ["Vehicle", "HeadUnitInfo"]


def test_strings_invalid_regex_is_actionable(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: _FakeBV())

    with pytest.raises(bridge.OperationFailure, match="Invalid string regex"):
        instance._strings(None, query="(", offset=0, limit=100, regex=True)


# --- I5: sections ---


def test_sections_returns_all_sections_with_permissions(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        sections={
            ".text": _FakeSection(".text", 0x1000, 0x5000, semantics=1),
            ".data": _FakeSection(".data", 0x5000, 0x6000, semantics=3),
            ".rodata": _FakeSection(".rodata", 0x6000, 0x7000, semantics=2),
        },
        segments={
            0x1000: _FakeSegment(readable=True, writable=False, executable=True),
            0x5000: _FakeSegment(readable=True, writable=True, executable=False),
            0x6000: _FakeSegment(readable=True, writable=False, executable=False),
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._sections(None)

    items = result["items"]
    assert len(items) == 3
    assert result["total"] == 3
    text_sec = items[0]
    assert text_sec["name"] == ".text"
    assert text_sec["start"] == "0x1000"
    assert text_sec["end"] == "0x5000"
    assert text_sec["length"] == 0x4000
    assert text_sec["semantics"] == "ReadOnlyCode"
    assert text_sec["readable"] is True
    assert text_sec["writable"] is False
    assert text_sec["executable"] is True

    data_sec = items[1]
    assert data_sec["name"] == ".data"
    assert data_sec["semantics"] == "ReadWriteData"
    assert data_sec["writable"] is True

    rodata_sec = items[2]
    assert rodata_sec["name"] == ".rodata"
    assert rodata_sec["semantics"] == "ReadOnlyData"
    assert rodata_sec["executable"] is False


def test_sections_query_filters_by_name(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        sections={
            ".text": _FakeSection(".text", 0x1000, 0x5000),
            ".rodata": _FakeSection(".rodata", 0x5000, 0x6000),
            ".data": _FakeSection(".data", 0x6000, 0x7000),
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._sections(None, query="data")

    items = result["items"]
    assert len(items) == 2
    names = [s["name"] for s in items]
    assert ".rodata" in names
    assert ".data" in names


def test_sections_null_segment_omits_rwx(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        sections={".bss": _FakeSection(".bss", 0x9000, 0xa000)},
        segments={0x1000: _FakeSegment(readable=True, writable=False, executable=True)},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._sections(None)

    items = result["items"]
    assert len(items) == 1
    assert "readable" not in items[0]


def test_sections_without_segments_omits_rwx(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _BareView:
        def __init__(self):
            self.sections = {".text": _FakeSection(".text", 0x1000, 0x2000)}

    bv = _BareView()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._sections(None)

    items = result["items"]
    assert len(items) == 1
    assert "readable" not in items[0]
    assert "writable" not in items[0]
    assert "executable" not in items[0]


# --- I8: enhanced imports ---


def test_imports_includes_function_data_and_address_symbols(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    func_sym = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x1000, "printf")
    func_sym.short_name = "printf"
    func_sym.namespace = "libc"

    data_sym = fake_bn.Symbol(fake_bn.SymbolType.ImportedDataSymbol, 0x2000, "__stdout")
    data_sym.short_name = "__stdout"
    data_sym.namespace = "libc"

    addr_sym = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x3000, "iat_entry")
    addr_sym.short_name = "iat_entry"
    addr_sym.namespace = ""

    bv = _FakeBV(symbols=[func_sym, data_sym, addr_sym])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)

    items = result["items"]
    assert len(items) == 3
    assert result["total"] == 3
    kinds = {item["name"]: item["kind"] for item in items}
    assert kinds["printf"] == "function"
    assert kinds["__stdout"] == "data"
    assert kinds["iat_entry"] == "address"


def test_imports_excludes_pic_self_defined_exports(monkeypatch):
    """On a PIC .so BN models the lib's own defined+exported functions as import
    veneers (ImportedFunctionSymbol) plus their GOT slots (ImportAddressSymbol),
    even though they're defined in the same module. These self-references must be
    excluded from the imports survey (and counted, not silently dropped), leaving
    only genuine external dependencies (#202)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    NAME = "_ZN5boost6system15system_categoryEv"
    # the real, DEFINED export (function body in this module)
    defined = fake_bn.Symbol(fake_bn.SymbolType.FunctionSymbol, 0x401c90, NAME)
    defined.short_name = NAME
    # BN's PIC self-references: an import veneer + a GOT slot, both library:null
    veneer = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x401980, NAME)
    veneer.short_name = NAME
    veneer.namespace = "BNINTERNALNAMESPACE"
    got = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x413f58, NAME)
    got.short_name = NAME
    got.namespace = "BNINTERNALNAMESPACE"
    # a genuine external import (no in-module definition)
    ext = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x401000, "memcpy")
    ext.short_name = "memcpy"
    ext.namespace = ""

    bv = _FakeBV(symbols=[defined, veneer, got, ext])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)
    names = [it["name"] for it in result["items"]]
    assert names == ["memcpy"]                    # only the genuine import remains
    assert result["total"] == 1
    assert result["self_defined_excluded"] == 2   # veneer + GOT slot, visibly dropped

    # the summary path excludes them too, and surfaces the count
    summary = instance._imports(None, summary=True)
    assert summary["total_symbols"] == 1
    assert summary["self_defined_excluded"] == 2


def test_imports_sorts_function_kind_first_then_library_name(monkeypatch):
    # Imports order by kind usefulness (function -> data -> address) first, then
    # library/name (#07). The data symbol here has the alphabetically-EARLIER
    # library, yet the function still sorts first -- kind dominates library.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    sym_b = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x2000, "zebra")
    sym_b.short_name = "zebra"
    sym_b.namespace = "libz"

    sym_a = fake_bn.Symbol(fake_bn.SymbolType.ImportedDataSymbol, 0x1000, "alpha")
    sym_a.short_name = "alpha"
    sym_a.namespace = "liba"

    bv = _FakeBV(symbols=[sym_b, sym_a])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)

    items = result["items"]
    assert items[0]["name"] == "zebra" and items[0]["kind"] == "function"
    assert items[0]["library"] == "libz"
    assert items[1]["name"] == "alpha" and items[1]["kind"] == "data"
    assert items[1]["library"] == "liba"


def test_imports_bn_sentinel_namespace_is_not_surfaced_as_library(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    sym = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x1000, "memcpy")
    sym.short_name = "memcpy"
    sym.namespace = "BNINTERNALNAMESPACE"

    bv = _FakeBV(symbols=[sym])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)

    item = result["items"][0]
    # The meaningless sentinel must not masquerade as a real library...
    assert item["library"] is None
    # ...but stays available under an honestly-named field.
    assert item["namespace"] == "BNINTERNALNAMESPACE"


def test_imports_summary_includes_needed_libraries(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    sym = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x1000, "memcpy")
    sym.short_name = "memcpy"
    sym.namespace = "BNEXTERNALNAMESPACE"

    bv = _FakeBV(symbols=[sym])
    bv.libraries = ["libssl.so.1.1", "libc.so.6"]
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None, summary=True)

    # DT_NEEDED is the real dependency signal, sorted and de-duped.
    assert result["needed_libraries"] == ["libc.so.6", "libssl.so.1.1"]
    # namespace grouping still reflects the raw BN namespace.
    assert result["namespaces"] == {"BNEXTERNALNAMESPACE": 1}


# --- read: raw bytes at an address ---


def test_read_returns_hex_and_ascii_for_mapped_address(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(memory={0x1000: b"\x48\x65\x6c\x6c\x6f\x00\x90\xff"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._read(None, "0x1000", 8)

    assert result["address"] == "0x1000"
    assert result["length"] == 8
    assert result["hex"] == "48656c6c6f0090ff"
    assert result["ascii"] == "Hello..."
    assert "short_read" not in result
    assert "note" not in result


def test_read_accepts_decimal_address(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(memory={0x1000: b"\x41\x42\x43\x44"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._read(None, "4096", 4)

    assert result["address"] == "0x1000"
    assert result["hex"] == "41424344"
    assert result["ascii"] == "ABCD"


def test_read_unmapped_address_raises_naming_the_address(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(memory={0x1000: b"\x41\x42\x43\x44"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(RuntimeError, match="0xdead.*not mapped"):
        instance._read(None, "0xdead", 16)


def test_read_short_read_returns_mapped_bytes_with_note(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(memory={0x1000: b"\x01\x02\x03\x04"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._read(None, "0x1000", 16)

    assert result["length"] == 4
    assert result["hex"] == "01020304"
    assert result["short_read"] is True
    assert result["requested_length"] == 16
    assert "short read" in result["note"]
    assert "0x1000" in result["note"]


# --- imports --summary ---


def test_imports_summary_aggregates_counts(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    sym_a = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x1000, "printf")
    sym_a.short_name = "printf"
    sym_a.namespace = "libc"

    sym_b = fake_bn.Symbol(fake_bn.SymbolType.ImportedDataSymbol, 0x2000, "__stdout")
    sym_b.short_name = "__stdout"
    sym_b.namespace = "libc"

    sym_c = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x3000, "read")
    sym_c.short_name = "read"
    sym_c.namespace = "libc"

    sym_d = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x4000, "foobar")
    sym_d.short_name = "foobar"
    sym_d.namespace = "libfoo"

    bv = _FakeBV(symbols=[sym_a, sym_b, sym_c, sym_d])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None, summary=True)

    assert result["total_symbols"] == 4
    assert result["namespaces"] == {"libc": 3, "libfoo": 1}
    assert result["by_kind"] == {"function": 3, "data": 1}


def test_imports_summary_empty(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None, summary=True)

    assert result["total_symbols"] == 0
    assert result["namespaces"] == {}
    assert result["by_kind"] == {}


def test_imports_default_sort_surfaces_function_imports_first(monkeypatch):
    # #07: address-kind internals must not bury function/libc imports in the
    # default listing -- a `head`/page read should show the function imports.
    # The address symbol is named to sort FIRST alphabetically, so only a
    # kind-priority sort (not a name sort) can surface the function import.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]
    addr_sym = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x1000, "aaa_internal")
    addr_sym.short_name = "aaa_internal"
    fn_sym = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x2000, "memcpy")
    fn_sym.short_name = "memcpy"
    bv = _FakeBV(symbols=[addr_sym, fn_sym])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    items = instance._imports(None)["items"]
    assert items[0]["kind"] == "function" and items[0]["name"] == "memcpy"
    kinds = [it["kind"] for it in items]
    assert kinds.index("function") < kinds.index("address")


# --- papercuts: pagination, clean errors, force-analysis echo -----------------


def test_imports_pagination_slices_offset_and_limit(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]
    syms = []
    for i in range(5):
        s = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x1000 + i, f"fn{i}")
        s.short_name = f"fn{i}"
        s.namespace = "lib"
        syms.append(s)
    bv = _FakeBV(symbols=syms)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    full = instance._imports(None)
    full_items = full["items"]
    assert [item["name"] for item in full_items] == ["fn0", "fn1", "fn2", "fn3", "fn4"]
    assert full["total"] == 5 and full["has_more"] is False
    page = instance._imports(None, offset=1, limit=2)
    # The page carries the slice AND the honest total/remainder metadata.
    assert page["items"] == full_items[1:3]
    assert page["total"] == 5
    assert page["offset"] == 1 and page["limit"] == 2 and page["returned"] == 2
    assert page["has_more"] is True   # offset 1 + 2 returned < 5 total
    # summary aggregates the whole set regardless of offset/limit.
    summary = instance._imports(None, summary=True, offset=1, limit=2)
    assert summary["total_symbols"] == 5


def test_imports_rejects_negative_offset_and_limit(monkeypatch):
    # Non-CLI callers (raw socket / py exec) must hit the same paging guard the
    # sibling list ops enforce, not a silent negative-index slice (#68).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(symbols=[])
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    with pytest.raises(bridge.OperationFailure) as e1:
        instance._imports(None, offset=-1)
    assert e1.value.status == "invalid_request"
    with pytest.raises(bridge.OperationFailure) as e2:
        instance._imports(None, limit=-5)
    assert e2.value.status == "invalid_request"
    with pytest.raises(bridge.OperationFailure) as e3:
        instance._imports(None, limit=0)   # zero limit is degenerate too
    assert e3.value.status == "invalid_request"


def test_sections_pagination_slices_offset_and_limit(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    secs = {
        ".a": _FakeSection(".a", 0x1000, 0x1100),
        ".b": _FakeSection(".b", 0x2000, 0x2100),
        ".c": _FakeSection(".c", 0x3000, 0x3100),
    }
    bv = _FakeBV(sections=secs)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    full = instance._sections(None)
    assert [s["name"] for s in full["items"]] == [".a", ".b", ".c"]
    assert full["total"] == 3 and full["has_more"] is False
    page = instance._sections(None, offset=1, limit=1)
    assert [s["name"] for s in page["items"]] == [".b"]
    # The truncated page still reports the true total + remainder.
    assert page["total"] == 3 and page["returned"] == 1 and page["has_more"] is True


# --- --quick honesty: don't return a misleading empty result on an unanalyzed view


def test_strings_requires_refresh_when_quick_loaded(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(strings=[])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # Quick-loaded: strings analysis hasn't run, so refuse rather than return [].
    bridge._quick_loaded_views.add(bv)
    with pytest.raises(RuntimeError, match="loaded with --quick"):
        instance._strings(None, query=None, offset=0, limit=100)

    # Once analysis lands, strings answers normally (here: genuinely empty).
    bridge._quick_loaded_views.discard(bv)
    result = instance._strings(None, query=None, offset=0, limit=100)
    assert result["items"] == [] and result["total"] == 0


def test_refresh_clears_quick_state_and_enables_strings(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(strings=[])
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "refresh", lambda: [])

    bridge._quick_loaded_views.add(bv)
    instance._refresh(None)  # runs analysis and clears the quick flag

    assert bv not in bridge._quick_loaded_views
    assert getattr(bv, "analysis_updated", False) is True
    result = instance._strings(None, query=None, offset=0, limit=100)
    assert result["items"] == [] and result["total"] == 0


def test_sections_rejects_negative_count(monkeypatch):
    # #100: _sections re-enforces the count contract for raw callers.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: _FakeBV(sections={}))
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._sections("active", offset=-1)
    assert exc.value.status == "invalid_request"
