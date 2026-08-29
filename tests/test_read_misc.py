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


def test_list_envelopes_carry_kind(monkeypatch):
    # #275: every generic-list read carries a `kind` discriminator (strings shown
    # here; the registry-driven conformance test covers the rest).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(strings=[_FakeStringRef(0x1000, 5, "hello")])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    listed = instance._strings(None, query=None, offset=0, limit=100, min_length=4)
    assert listed["kind"] == "strings" and isinstance(listed["items"], list)

    counted = instance._strings(None, query=None, offset=0, limit=100, count_only=True)
    assert counted["kind"] == "strings" and "count" in counted


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


def test_strings_max_length_excludes_long_strings(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(strings=[
        _FakeStringRef(0x1000, 4, "abcd"),
        _FakeStringRef(0x2000, 12, "a long blobby"),
    ])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query=None, offset=0, limit=100, max_length=8)

    assert [item["value"] for item in result["items"]] == ["abcd"]


def test_strings_probable_format_rejects_blob_noise_keeps_real_format(monkeypatch):
    # The whole point (#554): a raw `%s|%d` regex matches accidental
    # percent-substrings in resource/font/blob data. The directive-grammar mode
    # keeps only strings whose EVERY '%' is a valid directive (or %%), rejecting
    # stray percents -- and rejects prose like "100% done" and bare "%%".
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(strings=[
        _FakeStringRef(0x1000, 7, "%s: %d\n"),      # real format string
        _FakeStringRef(0x1010, 8, "hp=%d/%d"),       # real format string
        _FakeStringRef(0x1020, 9, "100% done"),      # prose, NOT a directive
        _FakeStringRef(0x1030, 6, "50%% off"),       # only escaped %% -> no arg
        _FakeStringRef(0x1040, 7, "x%zzq%1"),        # blob noise: malformed %
        _FakeStringRef(0x1050, 5, "plain"),          # no percents at all
    ])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query=None, offset=0, limit=100,
                               probable_format_strings=True)

    assert [item["value"] for item in result["items"]] == ["%s: %d\n", "hp=%d/%d"]


def test_strings_probable_format_annotates_directives_and_code_refs(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        strings=[_FakeStringRef(0x1000, 12, "user %s: %d%%")],
        code_refs={0x1000: [_FakeCodeRef(0x400100), _FakeCodeRef(0x400200)]},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query=None, offset=0, limit=100,
                               probable_format_strings=True)

    items = result["items"]
    assert len(items) == 1
    entry = items[0]
    # %% is an escaped literal, not an argument-consuming directive, so it is
    # excluded from the directive list.
    assert entry["format_directives"] == ["%s", "%d"]
    assert entry["directive_count"] == 2
    assert entry["code_refs"] == 2


def test_strings_probable_format_recovers_flags_width_precision_length(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(strings=[
        _FakeStringRef(0x1000, 20, "%-10s %08x %5.2f %ld %p %n"),
    ])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query=None, offset=0, limit=100,
                               probable_format_strings=True)

    entry = result["items"][0]
    assert entry["format_directives"] == ["%-10s", "%08x", "%5.2f", "%ld", "%p", "%n"]


def test_strings_probable_format_keeps_indirect_zero_xref_candidate(monkeypatch):
    # code_refs is enrichment, NOT a hard filter: a format string reached
    # indirectly can have zero direct xrefs. Dropping it would be a false
    # negative, so it is kept and annotated code_refs=0.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(strings=[_FakeStringRef(0x2000, 6, "err %d")])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query=None, offset=0, limit=100,
                               probable_format_strings=True)

    entry = result["items"][0]
    assert entry["format_directives"] == ["%d"]
    assert entry["code_refs"] == 0


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


def test_sections_writable_executable_verdict(monkeypatch):
    # #453: `executable` is segment-derived, so r-x-mapped .rodata reads
    # executable=true. Give a direct W+X verdict (writable AND executable) so data
    # in an r-x segment is NOT reported as executable attack surface.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        sections={
            ".text": _FakeSection(".text", 0x1000, 0x2000, semantics=1),
            ".rodata": _FakeSection(".rodata", 0x2000, 0x3000, semantics=2),
            ".jit": _FakeSection(".jit", 0x3000, 0x4000, semantics=3),
        },
        segments={
            0x1000: _FakeSegment(readable=True, writable=False, executable=True),
            # .rodata mapped into an r-x load segment: executable=true, not writable.
            0x2000: _FakeSegment(readable=True, writable=False, executable=True),
            # genuinely writable + executable.
            0x3000: _FakeSegment(readable=True, writable=True, executable=True),
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._sections(None)
    by = {it["name"]: it for it in result["items"]}
    # The acceptance case: executable from the segment, but not W+X (not writable).
    assert by[".rodata"]["executable"] is True
    assert by[".rodata"]["writable_executable"] is False
    assert by[".rodata"]["permission_source"] == "segment"
    assert by[".text"]["writable_executable"] is False
    assert by[".jit"]["writable_executable"] is True
    # Top-level verdict counts only the genuinely W+X section.
    assert result["writable_executable_count"] == 1
    assert result["writable_executable_items"] == [".jit"]
    assert result["wx_verdict"] == "wx_sections_present"       # #461

    # Text verdict line.
    from bn.formatters import _render_sections_text
    assert _render_sections_text(result).splitlines()[0] == "w+x: 1 section(s): .jit"
    clean = {**result, "wx_verdict": "no_wx_sections_observed",
             "writable_executable_count": 0, "writable_executable_items": []}
    assert _render_sections_text(clean).splitlines()[0] == "w+x: none observed"


def test_sections_wx_verdict_unknown_on_mapped_view_without_perms_461(monkeypatch):
    # #461: a mapped/raw embedded view whose sections carry only synthetic/external
    # metadata (no segment perms) must report an explicit "unknown" verdict, NOT a
    # silent empty W+X set that reads as an all-clear.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        sections={
            # ExternalSection (semantics=4), mapped with all-false perms -- the exact
            # shape from the ticket: readable/writable/executable all false.
            ".synthetic_builtins": _FakeSection(".synthetic_builtins", 0x500000, 0x500018, semantics=4),
        },
        segments={
            0x500000: _FakeSegment(readable=False, writable=False, executable=False),
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._sections(None)

    assert result["wx_verdict"] == "unknown_insufficient_metadata"
    assert "writable_executable_count" not in result       # nothing to count on
    from bn.formatters import _render_sections_text
    first = _render_sections_text(result).splitlines()[0]
    assert first.startswith("w+x: unknown") and "NOT an all-clear" in first


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


def test_imports_callable_jump_slot_reclassified_as_function_478(monkeypatch):
    """#478: on a target where BN failed to recover PLT-stub ImportedFunctionSymbols,
    callable sinks survive only as ImportAddressSymbol GOT slots. A slot carrying a
    JUMP_SLOT (.rela.plt) relocation is a callable function import -- reclassify it as
    function-kind so a sink-rich target isn't misread as stripped/static. A GLOB_DAT
    data slot stays address-kind."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]
    JS = fake_bn.RelocationType.ELFJumpSlotRelocationType
    GD = fake_bn.RelocationType.ELFGlobalRelocationType

    memcpy = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x3000, "memcpy")
    strcpy = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x3008, "strcpy")
    stdout = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x3010, "stdout")
    bv = _FakeBV(symbols=[memcpy, strcpy, stdout], relocations={
        0x3000: [_FakeReloc(JS, memcpy)],
        0x3008: [_FakeReloc(JS, strcpy)],
        0x3010: [_FakeReloc(GD, stdout)],
    })
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    kinds = {it["name"]: it["kind"] for it in instance._imports(None)["items"]}
    assert kinds["memcpy"] == "function"
    assert kinds["strcpy"] == "function"
    assert kinds["stdout"] == "address"
    assert instance._imports(None, count_only=True)["count"] == 3


def test_imports_healthy_jump_slot_not_reclassified_478(monkeypatch):
    """#478 safety: when BN DID recover the PLT-stub ImportedFunctionSymbol, the
    matching JUMP_SLOT GOT slot must NOT be reclassified/double-counted. By default
    it collapses against the function entry; under --include-got it stays kind=address
    (a GOT view of an already-listed function), never a second function row."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]
    JS = fake_bn.RelocationType.ELFJumpSlotRelocationType

    plt = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x1000, "memcpy")
    got = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x3000, "memcpy")
    bv = _FakeBV(symbols=[plt, got], relocations={0x3000: [_FakeReloc(JS, got)]})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    default = instance._imports(None)
    assert [(it["name"], it["kind"]) for it in default["items"]] == [("memcpy", "function")]
    assert default["got_collapsed"] == 1
    got_kinds = sorted(it["kind"] for it in instance._imports(None, include_got=True)["items"])
    assert got_kinds == ["address", "function"]


def test_imports_recovers_relocation_only_external_symbol(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]
    jump_slot = fake_bn.RelocationType.ELFJumpSlotRelocationType
    symbol = fake_bn.Symbol(
        fake_bn.SymbolType.ExternalSymbol, 0, "dispatch_record"
    )
    relocation = _FakeReloc(jump_slot, symbol)
    relocation.address = 0x3000
    bv = _FakeBV()
    bv.relocations = [relocation]
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)

    assert result["items"] == [
        {
            "name": "dispatch_record",
            "address": "0x3000",
            "library": None,
            "namespace": "",
            "raw_name": "dispatch_record",
            "kind": "function",
            "provenance": "relocation",
        }
    ]
    assert result["relocation_recovered"] == 1


def test_imports_query_and_regex_filter(monkeypatch):
    # #450: filter the import survey so a sink sweep doesn't need full paging + jq.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    def imp(addr, name):
        s = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, addr, name)
        s.short_name = name
        s.namespace = "libc"
        return s

    bv = _FakeBV(symbols=[imp(0x1000, "system"), imp(0x2000, "execve"),
                          imp(0x3000, "recv"), imp(0x4000, "memcpy")])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    def names(res):
        return sorted(it["name"] for it in res["items"])

    assert names(instance._imports(None, query="system")) == ["system"]
    assert names(instance._imports(None, query="system|execve|popen", regex=True)) == \
        ["execve", "system"]
    assert instance._imports(None, query="recv", count_only=True)["count"] == 1
    # No match -> a clean empty result (total 0), not an error.
    assert instance._imports(None, query="nosuchimport")["items"] == []
    assert instance._imports(None, query="nosuchimport")["total"] == 0


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


def test_imports_keeps_import_when_thunk_shares_its_address(monkeypatch):
    """#379: on PE64 BN co-names the IAT jump-thunk (a FunctionSymbol) with the
    ImportedFunctionSymbol at the SAME address. That thunk is an import veneer, not
    a real local definition, so it must NOT suppress the genuine import -- otherwise
    the #202 self-export filter empties a PE's whole import list."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    imp = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x140007a48, "CreateFileA")
    imp.short_name = "CreateFileA"; imp.namespace = ""
    thunk = fake_bn.Symbol(fake_bn.SymbolType.FunctionSymbol, 0x140007a48, "CreateFileA")  # same addr -> thunk
    thunk.short_name = "CreateFileA"
    # codex review: BN also models a PE DATA import (ImportedDataSymbol) and
    # co-names a DataSymbol at its very IAT address -- that DataSymbol is a data
    # veneer, not a real def, so it must NOT suppress the genuine data import.
    data_imp = fake_bn.Symbol(fake_bn.SymbolType.ImportedDataSymbol, 0x140008100, "__imp___C_specific_handler")
    data_imp.short_name = "__imp___C_specific_handler"; data_imp.namespace = ""
    data_veneer = fake_bn.Symbol(fake_bn.SymbolType.DataSymbol, 0x140008100, "__imp___C_specific_handler")  # same addr
    data_veneer.short_name = "__imp___C_specific_handler"
    # a REAL local definition still suppresses its self-export veneer (#202): the
    # def sits at its own .text address, distinct from the veneer.
    defined = fake_bn.Symbol(fake_bn.SymbolType.FunctionSymbol, 0x401c90, "local_helper")
    defined.short_name = "local_helper"
    veneer = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x401980, "local_helper")
    veneer.short_name = "local_helper"; veneer.namespace = "BNINTERNALNAMESPACE"

    bv = _FakeBV(symbols=[imp, thunk, data_imp, data_veneer, defined, veneer])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)
    names = [it["name"] for it in result["items"]]
    assert "CreateFileA" in names              # genuine function import kept (#379)
    assert "__imp___C_specific_handler" in names  # genuine data import kept (codex review)
    assert "local_helper" not in names         # self-export veneer still dropped (#202)
    assert result.get("self_defined_excluded", 0) == 1   # only the real-def veneer


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


def test_function_list_envelope_discloses_quick_analysis_state(monkeypatch):
    # #437: a --quick-loaded target's function count is PARTIAL, but the
    # `functions` envelope looked complete ({count, total} with no state). Thread
    # the analysis_state the bridge already derives into list/search (count + full).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    bridge._quick_loaded_views.add(bv)
    for env in (
        instance._list_functions("active"),
        instance._list_functions("active", count_only=True),
        instance._search_functions("active", ""),
        instance._search_functions("active", "", count_only=True),
    ):
        assert env["analysis_state"] == "quick", env
        assert env["partial"] is True, env

    bridge._quick_loaded_views.discard(bv)
    for env in (
        instance._list_functions("active"),
        instance._list_functions("active", count_only=True),
        instance._search_functions("active", ""),
        instance._search_functions("active", "", count_only=True),
    ):
        assert env["analysis_state"] == "full", env
        assert env["partial"] is False, env


def test_function_list_search_min_size_drops_thunk_veneers(monkeypatch):
    # #446: --min-size drops the tiny PLT/GOT thunk veneers (typically <= 16 bytes)
    # that otherwise list/search under the same name as the real body.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[
        _FakeFunction(0x1000, "RFCOMM_Recv", total_bytes=16),    # veneer
        _FakeFunction(0x2000, "RFCOMM_Recv", total_bytes=880),   # real body
        _FakeFunction(0x3000, "helper", total_bytes=40),
    ])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    full = instance._list_functions("active")
    assert full["total"] == 3
    filtered = instance._list_functions("active", min_size=100)
    assert [it["address"] for it in filtered["items"]] == ["0x2000"]
    assert filtered["total"] == 1
    assert instance._list_functions("active", min_size=100, count_only=True)["count"] == 1

    hits = instance._search_functions("active", "RFCOMM", min_size=100)
    assert [it["address"] for it in hits["items"]] == ["0x2000"]
    assert instance._search_functions("active", "RFCOMM", min_size=100, count_only=True)["count"] == 1


def test_function_search_word_boundary_excludes_substring_fps(monkeypatch):
    # #457: plain (substring) `function search popen` matches unrelated symbols like
    # zipOpenArchive (which contains "popen") -- a sink-survey false positive.
    # --word matches the query only as a whole identifier token; --exact is stricter.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[
        _FakeFunction(0x1000, "popen"),
        _FakeFunction(0x2000, "popen@plt"),
        _FakeFunction(0x3000, "zipOpenArchive"),   # contains "popen" as a substring
        _FakeFunction(0x4000, "my_popen_wrapper"),
    ])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    def names(res):
        return sorted(it["name"] for it in res["items"])

    # Plain substring: all four match (the trap).
    assert names(instance._search_functions("active", "popen")) == \
        ["my_popen_wrapper", "popen", "popen@plt", "zipOpenArchive"]
    # --word: only the whole-token matches; the substring FPs are excluded.
    assert names(instance._search_functions("active", "popen", word=True)) == \
        ["popen", "popen@plt"]
    # --exact: only the exact name.
    assert names(instance._search_functions("active", "popen", exact=True)) == ["popen"]
    assert instance._search_functions("active", "popen", word=True, count_only=True)["count"] == 2


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


def test_imports_collapses_got_slot_duplicates(monkeypatch):
    """Each import appears as a function/data PLT entry AND an (address) GOT slot
    of the same name -- ~half the list is dups. Collapse the GOT slots by default;
    --include-got shows them (#212)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    fn = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x1000, "memcpy")
    fn.short_name = "memcpy"; fn.namespace = ""
    got = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x2000, "memcpy")  # GOT slot dup
    got.short_name = "memcpy"; got.namespace = ""
    uniq = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x3000, "__gmon_start__")
    uniq.short_name = "__gmon_start__"; uniq.namespace = ""
    bv = _FakeBV(symbols=[fn, got, uniq])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)
    names = [(it["name"], it["kind"]) for it in result["items"]]
    assert ("memcpy", "function") in names
    assert ("memcpy", "address") not in names         # GOT dup collapsed
    assert ("__gmon_start__", "address") in names     # genuinely-unique address kept
    assert result["got_collapsed"] == 1

    full = instance._imports(None, include_got=True)
    assert ("memcpy", "address") in [(it["name"], it["kind"]) for it in full["items"]]
    assert "got_collapsed" not in full


def test_imports_surfaces_et_rel_external_symbols(monkeypatch):
    """An ET_REL .ko has no Imported* symbols; its kernel-API refs are
    ExternalSymbols (.extern). `bn imports` must surface them (#213)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    printk = fake_bn.Symbol(fake_bn.SymbolType.ExternalSymbol, 0x0, "printk")
    printk.short_name = "printk"; printk.namespace = ""
    kmalloc = fake_bn.Symbol(fake_bn.SymbolType.ExternalSymbol, 0x8, "__kmalloc")
    kmalloc.short_name = "__kmalloc"; kmalloc.namespace = ""
    bv = _FakeBV(symbols=[printk, kmalloc])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)
    names = {it["name"]: it["kind"] for it in result["items"]}
    assert names == {"printk": "external", "__kmalloc": "external"}


def test_imports_external_dedups_against_got_only_imports(monkeypatch):
    """On a standard ELF most imports are GOT-only (ImportAddressSymbol, no PLT
    function entry) and the SAME names reappear as ExternalSymbol. The external
    dedup must skip them so they aren't double-listed (address + external) --
    reintroducing #212. The dedup is against ALL emitted names, not just
    function/data (#213 review)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    # memcpy: GOT-only import (address kind, no function twin) + an external dup
    got = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x1000, "memcpy")
    got.short_name = "memcpy"; got.namespace = ""
    ext = fake_bn.Symbol(fake_bn.SymbolType.ExternalSymbol, 0x0, "memcpy")
    ext.short_name = "memcpy"; ext.namespace = ""
    # a genuinely .ko-style external with no import twin -> must still appear
    only_ext = fake_bn.Symbol(fake_bn.SymbolType.ExternalSymbol, 0x8, "printk")
    only_ext.short_name = "printk"; only_ext.namespace = ""
    bv = _FakeBV(symbols=[got, ext, only_ext])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)
    rows = [(it["name"], it["kind"]) for it in result["items"]]
    assert ("memcpy", "address") in rows
    assert ("memcpy", "external") not in rows         # deduped against the address entry
    assert ("printk", "external") in rows             # unique external still listed
    assert result["total"] == 2                        # not 3 (no double-list)


def test_exports_lists_global_weak_definitions_only(monkeypatch):
    """`bn exports` lists the global/weak DEFINITIONS (the public API); local
    definitions are internal and excluded (#198)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    pub = fake_bn.Symbol(fake_bn.SymbolType.FunctionSymbol, 0x1000, "_ZN3foo3bar4recvEi",
                         binding=fake_bn.SymbolBinding.GlobalBinding)
    pub.short_name = "foo::bar::recv"
    weak = fake_bn.Symbol(fake_bn.SymbolType.DataSymbol, 0x2000, "g_table",
                          binding=fake_bn.SymbolBinding.WeakBinding)
    weak.short_name = "g_table"
    local = fake_bn.Symbol(fake_bn.SymbolType.FunctionSymbol, 0x3000, "helper",
                           binding=fake_bn.SymbolBinding.LocalBinding)
    local.short_name = "helper"
    bv = _FakeBV(symbols=[pub, weak, local])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._exports(None)
    names = {it["name"]: it for it in result["items"]}
    assert set(names) == {"_ZN3foo3bar4recvEi", "g_table"}       # local excluded
    assert names["_ZN3foo3bar4recvEi"]["display_name"] == "foo::bar::recv"  # demangled
    assert names["_ZN3foo3bar4recvEi"]["kind"] == "function"
    assert names["g_table"]["kind"] == "data"
    assert instance._exports(None, count_only=True)["count"] == 2




def test_sections_query_count_only_reflects_semantics_match(monkeypatch):
    # The count_only path must reflect the broader name-or-semantics match, not
    # the old name-only filter.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(sections={
        ".text": _FakeSection(".text", 0x1000, 0x5000, semantics=1),     # ReadOnlyCode
        ".rodata": _FakeSection(".rodata", 0x5000, 0x6000, semantics=2),  # ReadOnlyData
    })
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._sections(None, query="code", count_only=True)

    assert result["count"] == 1 and result["total"] == 1

def test_sections_query_matches_semantics_not_just_name(monkeypatch):
    # #257: --query should match the semantics label too, so `--query code`
    # finds executable sections (.text = ReadOnlyCode) even though "code" is
    # not in the section name. Match is case-insensitive.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        sections={
            ".text": _FakeSection(".text", 0x1000, 0x5000, semantics=1),     # ReadOnlyCode
            ".rodata": _FakeSection(".rodata", 0x5000, 0x6000, semantics=2),  # ReadOnlyData
            ".data": _FakeSection(".data", 0x6000, 0x7000, semantics=3),      # ReadWriteData
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._sections(None, query="code")

    names = [s["name"] for s in result["items"]]
    assert names == [".text"]

def test_sections_query_semantics_broadens_beyond_name(monkeypatch):
    # Intentional #257 behavior change (pinned so it isn't read as a no-side-
    # effect bugfix): `--query data` now matches any section whose SEMANTICS is
    # ReadOnlyData/ReadWriteData even when "data" is absent from the name.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(sections={
        ".text": _FakeSection(".text", 0x1000, 0x5000, semantics=1),    # ReadOnlyCode
        ".roseg": _FakeSection(".roseg", 0x5000, 0x6000, semantics=2),  # ReadOnlyData, no "data" in name
        ".rwseg": _FakeSection(".rwseg", 0x6000, 0x7000, semantics=3),  # ReadWriteData, no "data" in name
    })
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._sections(None, query="data")

    assert sorted(s["name"] for s in result["items"]) == [".roseg", ".rwseg"]

def test_sections_query_semantics_match_is_case_insensitive(monkeypatch):
    # An uppercase query still matches the (CamelCase) semantics label.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(sections={".text": _FakeSection(".text", 0x1000, 0x5000, semantics=1)})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._sections(None, query="CODE")

    assert [s["name"] for s in result["items"]] == [".text"]


def test_sections_wx_verdict_is_query_independent_461(monkeypatch):
    """#461 audit P1: the W+X verdict is computed over the FULL section set, so
    scoping `sections --query` to unrelated sections cannot flip it to a false
    'no W+X' all-clear (or a spurious 'unknown') on a genuinely W+X image."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        sections={
            ".text": _FakeSection(".text", 0x1000, 0x2000, semantics=1),
            ".jitcode": _FakeSection(".jitcode", 0x2000, 0x3000, semantics=3),   # W+X
            ".rodata": _FakeSection(".rodata", 0x3000, 0x4000, semantics=2),
        },
        segments={
            0x1000: _FakeSegment(readable=True, writable=False, executable=True),
            0x2000: _FakeSegment(readable=True, writable=True, executable=True),  # W+X
            0x3000: _FakeSegment(readable=True, writable=False, executable=False),
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    assert instance._sections(None)["wx_verdict"] == "wx_sections_present"

    # Query excludes the W+X section: the PAGE is filtered, but the verdict still
    # reflects the whole binary (and still lists the full-set W+X section).
    filtered = instance._sections(None, query="rodata")
    assert [it["name"] for it in filtered["items"]] == [".rodata"]
    assert filtered["wx_verdict"] == "wx_sections_present"
    assert filtered["writable_executable_items"] == [".jitcode"]

    # A no-match query on a fully-permissioned binary is NOT spuriously "unknown".
    nomatch = instance._sections(None, query="zzznomatch")
    assert nomatch["items"] == []
    assert nomatch["wx_verdict"] == "wx_sections_present"


# --- data_vars: windowed typed-data read op (promoted out of py_exec) --------


def _data_window_bv():
    ptr_t = _FakeType("char*", width=4, type_class="PointerTypeClass")
    # A pointer ARRAY: its rendered text carries a '*' and (for the 1-element
    # case) its width equals one pointer, so only the type_class distinguishes
    # it from a real pointer.
    arr_t = _FakeType("void* [4]", width=16, type_class="ArrayTypeClass")
    arr1_t = _FakeType("void* [1]", width=4, type_class="ArrayTypeClass")
    int_t = _FakeType("int32_t", width=4, type_class="IntegerTypeClass", signed=True)
    big_t = _FakeType("struct config", width=64)  # too wide for a scalar value
    bv = _FakeBV(
        arch=_FakeArch(name="x86", address_size=4),
        data_vars={
            0x1000: _FakeDataVariable(0x1000, int_t),     # before window
            0x2000: _FakeDataVariable(0x2000, int_t),     # == lo: included
            0x2004: _FakeDataVariable(0x2004, ptr_t),     # -> named symbol
            0x2008: _FakeDataVariable(0x2008, ptr_t),     # -> ascii string
            0x2010: _FakeDataVariable(0x2010, arr_t),
            0x2018: _FakeDataVariable(0x2018, arr1_t),
            0x2020: _FakeDataVariable(0x2020, big_t),
            0x3000: _FakeDataVariable(0x3000, int_t),     # == hi: excluded
        },
        symbols=[
            types.SimpleNamespace(address=0x2004, name="g_handler"),
            types.SimpleNamespace(address=0x5000, name="on_message"),
        ],
        sections={".data": _FakeSection(".data", 0x2000, 0x2800)},
        memory={
            0x2000: (b"\x2a\x00\x00\x00"          # 0x2000: value 42
                     b"\x00\x50\x00\x00"          # 0x2004: -> 0x5000 (on_message)
                     b"\x00\x60\x00\x00"),        # 0x2008: -> 0x6000 ("hello")
            # Readable on purpose: if the 1-element pointer array were treated
            # as a pointer it WOULD decode to 0x5000/on_message, so the row
            # staying undecorated is real evidence, not an unmapped-read artifact.
            0x2018: b"\x00\x50\x00\x00",
            0x6000: b"hello\x00",
        },
    )
    return bv


def test_data_vars_window_rows_carry_typed_fields(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _data_window_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._data_vars(None, start="0x2000", end="0x3000")

    assert result["kind"] == "data_vars"
    assert result["has_more"] is False
    rows = {row["a"]: row for row in result["items"]}
    # Half-open window: 0x1000 (before) and 0x3000 (== end) excluded, lo included.
    assert sorted(rows) == ["0x2000", "0x2004", "0x2008", "0x2010", "0x2018", "0x2020"]

    scalar = rows["0x2000"]
    assert scalar["t"] == "int32_t" and scalar["w"] == 4 and scalar["v"] == 42
    assert scalar["sec"] == ".data"

    to_sym = rows["0x2004"]
    assert to_sym["n"] == "g_handler"
    assert to_sym["p"] == "0x5000" and to_sym["ps"] == "on_message"

    to_str = rows["0x2008"]
    assert to_str["p"] == "0x6000" and to_str["pstr"] == "hello"
    assert "ps" not in to_str

    # A pointer ARRAY contains '*' but is not a pointer: it must keep all
    # elements visible (no p/ps/v collapse to the first slot).
    arr = rows["0x2010"]
    assert arr["w"] == 16 and "p" not in arr and "v" not in arr

    # The sharp case: a ONE-element pointer array is pointer-WIDE and its text
    # carries a '*', so a width+text heuristic collapses it to its first
    # element and silently hides that it is an array. Only type_class separates
    # them. Its slot is mapped and would decode to 0x5000/on_message if the
    # decode were still text-driven.
    arr1 = rows["0x2018"]
    assert arr1["w"] == 4 and "p" not in arr1 and "ps" not in arr1 and "v" not in arr1

    wide = rows["0x2020"]
    assert "v" not in wide


def test_data_vars_seeks_window_instead_of_scanning_all_vars(monkeypatch):
    # The py_exec predecessor iterated sorted(bv.data_vars) across the WHOLE
    # view; the op must seek via get_next_data_var_after and never visit
    # addresses at/after the window end.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _data_window_bv()
    visited = []
    original = bv.get_next_data_var_after

    def _tracked(address):
        visited.append(int(address))
        return original(address)

    monkeypatch.setattr(bv, "get_next_data_var_after", _tracked)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    instance._data_vars(None, start="0x2000", end="0x3000")

    assert visited, "op did not use the seek API at all"
    assert min(visited) >= 0x2000 - 1, "seek started before the window"
    assert max(visited) < 0x3000, "seek walked past the window end"


def test_data_vars_has_more_is_honest_at_the_cap(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _data_window_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    capped = instance._data_vars(None, start="0x2000", end="0x3000", limit=2)
    assert [r["a"] for r in capped["items"]] == ["0x2000", "0x2004"]
    assert capped["has_more"] is True

    # Exactly limit rows left in the window: nothing was truncated.
    exact = instance._data_vars(None, start="0x2000", end="0x3000", limit=6)
    assert len(exact["items"]) == 6
    assert exact["has_more"] is False


def test_data_vars_rejects_empty_window_and_bad_limit(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _data_window_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure):
        instance._data_vars(None, start="0x3000", end="0x2000")
    with pytest.raises(bridge.OperationFailure):
        instance._data_vars(None, start="0x2000", end="0x3000", limit=0)


def test_data_vars_unreadable_pointer_row_survives(monkeypatch):
    # A pointer var whose slot bytes are unmapped must still produce a row
    # (address/type/width), just without the p/ps/pstr decoration.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    ptr_t = _FakeType("char*", width=4, type_class="PointerTypeClass")
    bv = _FakeBV(arch=_FakeArch(name="x86", address_size=4),
                 data_vars={0x2000: _FakeDataVariable(0x2000, ptr_t)},
                 memory={0x9000: b"\x00"})  # 0x2000 unmapped
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._data_vars(None, start="0x2000", end="0x2100")

    assert [r["a"] for r in result["items"]] == ["0x2000"]
    assert "p" not in result["items"][0]


def test_data_vars_scalar_signedness_follows_the_declared_type(monkeypatch):
    # bv.read_int defaults to sign=True, so an unsigned global whose top bit is
    # set was rendered as a negative number (`uint32_t` 0xf0000000 -> the value
    # -268435456). The decoded `v` must follow the TYPE's signedness, not
    # read_int's default.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        arch=_FakeArch(name="x86", address_size=4),
        data_vars={
            0x2000: _FakeDataVariable(0x2000, _FakeType("uint32_t", width=4,
                                                type_class="IntegerTypeClass", signed=False)),
            0x2004: _FakeDataVariable(0x2004, _FakeType("int32_t", width=4,
                                                type_class="IntegerTypeClass", signed=True)),
            # An integer the core left without a signedness flag: unsigned is
            # the honest reading of the raw bytes.
            0x2008: _FakeDataVariable(0x2008, _FakeType("flags_t", width=4,
                                                type_class="IntegerTypeClass")),
        },
        memory={0x2000: b"\x00\x00\x00\xf0" b"\xff\xff\xff\xff" b"\x00\x00\x00\xf0"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = {row["a"]: row for row in instance._data_vars(None, start="0x2000", end="0x2100")["items"]}

    assert rows["0x2000"]["v"] == 0xF0000000   # NOT -268435456
    assert rows["0x2004"]["v"] == -1           # genuinely signed: -1, not 0xffffffff
    assert rows["0x2008"]["v"] == 0xF0000000


# --- data_symbols: named DataSymbol listing (promoted out of py_exec) --------


def test_data_symbols_lists_named_data_symbols_only(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]
    bv = _FakeBV(symbols=[
        fake_bn.Symbol(fake_bn.SymbolType.DataSymbol, 0x2000, "g_state"),
        fake_bn.Symbol(fake_bn.SymbolType.DataSymbol, 0x2010, "g_table"),
        fake_bn.Symbol(fake_bn.SymbolType.DataSymbol, 0x2020, ""),          # unnamed
        fake_bn.Symbol(fake_bn.SymbolType.FunctionSymbol, 0x4000, "main"),  # not data
    ])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._data_symbols(None)

    assert result["kind"] == "data_symbols"
    assert result["items"] == [
        {"a": "0x2000", "n": "g_state"},
        {"a": "0x2010", "n": "g_table"},
    ]
    # Paging is opt-in: the default call returns the WHOLE set, because the
    # consumer builds a goto/search index in one shot and a silent default cap
    # would drop exactly the renamed globals this read exists to keep visible.
    assert result["limit"] is None
    assert result["total"] == 2 and result["returned"] == 2
    assert result["has_more"] is False


def test_data_symbols_pages_on_demand_with_an_honest_total(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]
    bv = _FakeBV(symbols=[
        fake_bn.Symbol(fake_bn.SymbolType.DataSymbol, 0x2000 + i * 8, f"g_{i}")
        for i in range(5)
    ])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    page = instance._data_symbols(None, offset=1, limit=2)

    assert [s["n"] for s in page["items"]] == ["g_1", "g_2"]
    assert page["total"] == 5          # the true total, not the page size
    assert page["offset"] == 1 and page["limit"] == 2 and page["returned"] == 2
    assert page["has_more"] is True

    tail = instance._data_symbols(None, offset=3, limit=2)
    assert [s["n"] for s in tail["items"]] == ["g_3", "g_4"]
    assert tail["has_more"] is False

    with pytest.raises(bridge.OperationFailure):
        instance._data_symbols(None, limit=0)
    with pytest.raises(bridge.OperationFailure):
        instance._data_symbols(None, offset=-1)


def test_data_symbols_surfaces_a_lookup_failure_instead_of_reporting_none(monkeypatch):
    # A blanket `except Exception: symbols = []` made a real BN failure
    # indistinguishable from "this binary has no data symbols" -- the caller
    # would drop every data hotspot and never learn why.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(symbols=[])

    def _boom(sym_type, *a, **k):
        raise RuntimeError("core symbol table unavailable")

    monkeypatch.setattr(bv, "get_symbols_of_type", _boom)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(RuntimeError, match="core symbol table unavailable"):
        instance._data_symbols(None)
