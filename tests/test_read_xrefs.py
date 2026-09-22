from __future__ import annotations

import difflib
import gc
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


def test_xrefs_raw_address_reads_the_ref_lists_once_815(monkeypatch):
    """#815: the raw-address path probed both ref lists for emptiness and then the
    builder re-read them for the response, so a high-fan-in symbol's ref set was
    materialised twice per call. The #374 mapped-address guard now runs on the
    lists the builder already read, so each list is read exactly once."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x401010, "caller")
    bv = _FakeBV(
        functions=[caller],
        code_refs={0x402000: [_FakeCodeRef(0x401010, caller)]},
        data_refs={0x402000: [0x401100]},
        sections={".rodata": _FakeSection(".rodata", 0x402000, 0x403000)},
        segments={0x401010: _FakeSegment(readable=True, executable=True)},
    )
    code_reads: list[int] = []
    data_reads: list[int] = []
    real_code_refs = bv.get_code_refs
    real_data_refs = bv.get_data_refs

    def counting_code_refs(address):
        code_reads.append(int(address))
        return real_code_refs(address)

    def counting_data_refs(address):
        data_reads.append(int(address))
        return real_data_refs(address)

    bv.get_code_refs = counting_code_refs
    bv.get_data_refs = counting_data_refs
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._xrefs(None, "0x402000")

    # The answer itself is unchanged...
    assert result["code_ref_count"] == 1 and result["data_ref_count"] == 1
    # ...and each ref list was read ONCE (twice before the fix, for code).
    assert code_reads == [0x402000], code_reads
    assert data_reads == [0x402000], data_reads


def test_xrefs_guard_keys_on_bn_refs_not_on_the_284_filtered_list_815(monkeypatch):
    """The #374 guard rejects an address BN holds NO ref for; it must not reject
    one whose refs the #284 adrp filter merely declined to RENDER.

    A page-aligned address is exactly where the two populations differ: BN records
    every `adrp xN, <page>` as a code ref to the page base, and #284 drops the ones
    whose paired offset is nonzero. Such an address is one BN holds refs for -- the
    pre-#815 probe (`bool(list(get_code_refs(...)) or ...)`) saw them and answered a
    clean `0`. Keying the guard on the FILTERED list instead turns that same read
    into `Address ... is not mapped`, which is a different answer to the same
    question (#815 must not change #374's semantics)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    page_base = 0x438000
    adrp = _adrp("x0", page_base, 0)
    # `adrp x0, 0x438000` / `add x0, x0, #0x350` -> the real referent is
    # 0x438350, so the ref to the page base is spurious and #284 drops it.
    adrp.il_basic_block = [adrp, _set_reg("x0", _LOp("LLIL_ADD", [_reg("x0"), _const(0x350)]), 1)]
    caller = _FakeFunction(0x401010, "caller")
    caller.get_low_level_il_at = lambda address: adrp
    bv = _FakeBV(
        functions=[caller],
        code_refs={page_base: [_FakeCodeRef(0x401014, caller)]},
        data_refs={},
        disassembly={0x401014: "adrp x0, #0x438000"},
        sections={".text": _FakeSection(".text", 0x400000, 0x410000)},
        segments={0x401014: _FakeSegment(readable=True, executable=True)},
    )
    bv.is_valid_offset = lambda address: False
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # Precondition: BN really does hold a code ref for this address, and #284
    # really does filter it out -- otherwise the test proves nothing.
    assert len(list(bv.get_code_refs(page_base))) == 1
    assert bridge.read_xrefs._genuine_code_refs(bv, page_base) == []

    result = instance._xrefs(None, hex(page_base))

    assert result["kind"] == "xrefs"
    assert result["code_ref_count"] == 0 and result["total"] == 0


def test_xrefs_literal_address_never_turns_a_failed_ref_read_into_zero_callers_815(monkeypatch):
    """A ref enumeration that FAILED is not evidence of "no refs".

    #374 exists to stop a read answering a false-negative `0 callers`, and the
    guard decides on BOTH ref lists: if the read that produced one of them was
    swallowed into `[]`, a MAPPED address answers a confident `total: 0` for a
    binary BN could not enumerate. The probe #815 removed read both lists
    unguarded on this path, so the failure surfaced as an error -- that must
    survive the fold into the builder, for each list and for each way a view can
    be unable to answer (the reader raising, and no reader at all)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    def _raise(address):
        raise RuntimeError("BN ref enumeration failed")

    def _use(reader: str, mode: str, *, code_refs=None):
        bv = _FakeBV(
            functions=[_FakeFunction(0x401000, "caller")],
            code_refs=code_refs or {},
            sections={".text": _FakeSection(".text", 0x400000, 0x410000)},
            segments={0x401234: _FakeSegment(readable=True, executable=True)},
        )
        # MAPPED, so the #374 guard is not what would save this read.
        bv.is_valid_offset = lambda address: True
        setattr(bv, reader, _raise if mode == "raises" else None)
        monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # No code refs, so each list in turn is the evidence the guard decides on.
    for reader, kind in (("get_code_refs", "code"), ("get_data_refs", "data")):
        _use(reader, "raises")
        with pytest.raises(RuntimeError, match="enumeration failed"):
            instance._xrefs(None, "0x401234")

        _use(reader, "absent")
        with pytest.raises(RuntimeError, match=f"cannot enumerate {kind} references"):
            instance._xrefs(None, "0x401234")


def test_xrefs_unreadable_data_refs_still_answer_an_address_with_code_refs_815(monkeypatch):
    """The mirror of the rule above, and its limit: a list the guard never
    consults must not be able to refuse the read.

    The probe #815 removed was `bool(list(code_refs) or list(data_refs))`, whose
    `or` short-circuits: with code refs in hand it never touched the data reader,
    so a view that cannot enumerate data refs still answered. Propagating a data
    read failure in that branch would invent a refusal base did not have -- and
    would split the answer by how the identifier was spelled, since the name path
    is unguarded."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x401010, "caller")
    bv = _FakeBV(
        functions=[caller],
        code_refs={0x401234: [_FakeCodeRef(0x401010, caller)]},
        sections={".text": _FakeSection(".text", 0x400000, 0x410000)},
        segments={0x401010: _FakeSegment(readable=True, executable=True)},
    )
    bv.is_valid_offset = lambda address: True
    bv.get_data_refs = None
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._xrefs(None, "0x401234")

    assert result["code_ref_count"] == 1 and result["data_ref_count"] == 0
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


def _import_scan_bv(name: str, address: int, caller_starts: list[int]):
    fake_bn = sys.modules["binaryninja"]
    sym = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, address, name)
    sym.short_name = name
    callers = []
    for index, start in enumerate(caller_starts):
        fn = _FakeFunction(start, f"caller_{index}")
        fn.low_level_il = [[_FakeLLILInstruction(start + 0x10, _FakeConstPtr(address))]]
        callers.append(fn)
    return _FakeBV(functions=callers, symbols=[sym])


class _UnreadableBlock:
    """A basic block whose LLIL iteration raises, i.e. a read failure."""

    def __iter__(self):
        raise RuntimeError("LLIL unavailable")


def test_xrefs_import_scan_flags_only_a_partial_scan(monkeypatch):
    """#622: when BN reports no code refs for an import, the fallback LLIL scan is
    budgeted, and ONLY a scan that stopped early may claim truncation. Both phases
    run against the same view: under the default budget the two-caller scan is
    complete and carries no flag or note, then the same scan under a one-function
    budget hands back its partial caller list, flagged, so a truncated list is
    never read as "no callers found"."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _import_scan_bv("plt_target", 0x20000, [0x1000, 0x2000])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    complete = instance._xrefs(None, "plt_target")
    assert complete["code_refs_scanned"] is True
    assert complete.get("truncated") is not True
    assert "scan_note" not in complete
    assert complete["code_ref_count"] == 2
    assert complete["returned"] == 2

    monkeypatch.setattr(bridge.read_xrefs, "SCAN_CALLS_MAX_FUNCS", 1)
    capped = instance._xrefs(None, "plt_target")
    assert capped["code_refs_scanned"] is True
    assert capped["truncated"] is True
    assert "budget" in capped["scan_note"]
    assert capped["code_ref_count"] == 1          # partial, never empty
    assert capped["returned"] == 1
    assert int(capped["items"][0]["address"], 16) == 0x1010


def test_xrefs_import_scan_flags_unreadable_llil(monkeypatch):
    """#622 review: a block whose LLIL cannot be lifted was skipped without a
    trace, so a caller list truncated by a read failure was reported as complete.
    The failure must reach the envelope: `truncated: true` plus a `scan_note`
    naming the unreadable LLIL -- not the budget, which was never hit. The note
    counts FUNCTIONS, not blocks: the same function with TWO unreadable blocks is
    one truncated function, reported once."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _import_scan_bv("plt_target", 0x20000, [0x1000, 0x2000])
    bv.functions[1].low_level_il = [_UnreadableBlock(), _UnreadableBlock()]
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._xrefs(None, "plt_target")

    assert result["code_refs_scanned"] is True
    assert result["truncated"] is True
    assert "LLIL" in result["scan_note"]
    assert "budget" not in result["scan_note"]    # the budget was never hit
    assert "1 function(s)" in result["scan_note"]
    assert result["code_ref_count"] == 1          # the readable caller survives


class _UnIterableIL:
    """An IL container whose iteration raises and which exposes no basic blocks,
    i.e. a function whose LLIL cannot be read at all."""

    def __iter__(self):
        raise RuntimeError("LLIL unavailable")


def test_xrefs_import_scan_flags_a_function_with_no_llil(monkeypatch):
    """#622 review (round-3 blocker): a function BN could not lift at all --
    ``low_level_il`` is None (BN documents that as "an error occurred while
    loading the IL"), or an IL container that cannot be iterated and offers no
    basic blocks -- was skipped WITHOUT a trace, so its body never entered the
    scan while the envelope still reported a complete one.

    That is worse than the base revision it replaced: base asserted nothing about
    completeness, whereas the absence of ``truncated`` now positively claims the
    caller list is whole. With every scanned function unlifted the answer was a
    clean, flagless "no callers" -- the exact false negative the budget
    disclosure exists to prevent."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _import_scan_bv("plt_target", 0x20000, [0x1000, 0x2000])
    bv.functions[1].low_level_il = None          # BN never lifted this one
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    partial = instance._xrefs(None, "plt_target")
    assert partial["code_refs_scanned"] is True
    assert partial["truncated"] is True
    assert "LLIL" in partial["scan_note"]
    assert "budget" not in partial["scan_note"]   # the budget was never hit
    assert "1 function(s)" in partial["scan_note"]
    assert partial["code_ref_count"] == 1         # the readable caller survives

    # An IL object that cannot be iterated and has no basic_blocks fallback is
    # the same failure by another route, and must disclose the same way.
    bv.functions[1].low_level_il = _UnIterableIL()
    assert instance._xrefs(None, "plt_target")["truncated"] is True

    # Nothing readable at all: the answer is EMPTY, which is precisely when the
    # flag has to be there -- an unflagged empty caller list reads as "no callers".
    for fn in bv.functions:
        fn.low_level_il = None
    blind = instance._xrefs(None, "plt_target")
    assert blind["code_ref_count"] == 0
    assert blind["truncated"] is True, "an empty scan of unlifted functions is not 'no callers'"
    assert "2 function(s)" in blind["scan_note"]


class _HostileIL:
    """An IL container whose iteration raises AND whose ``basic_blocks`` access
    raises -- a read failure no enumeration of known shapes anticipated."""

    def __iter__(self):
        raise RuntimeError("LLIL unavailable")

    @property
    def basic_blocks(self):
        raise RuntimeError("LLIL unavailable")


def test_xrefs_import_scan_discloses_an_unanticipated_read_failure(monkeypatch):
    """#622 review (round-3 blocker, class not instance): the scan's completeness
    disclosure must be a CHOKE POINT, not a list of the read failures someone
    thought of. Two shapes slipped through silently once already; this is a shape
    neither of those fixes enumerated -- an IL object that raises on iteration AND
    on the ``basic_blocks`` fallback -- and it must be recorded, not escape as an
    exception out of the whole xrefs op (nor, worse, as a flagless empty scan).

    A function whose IL read fails in any way has not been examined, so the
    envelope says so."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _import_scan_bv("plt_target", 0x20000, [0x1000, 0x2000])
    bv.functions[1].low_level_il = _HostileIL()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._xrefs(None, "plt_target")

    assert result["truncated"] is True
    assert "LLIL" in result["scan_note"]
    assert "1 function(s)" in result["scan_note"]
    assert result["code_ref_count"] == 1          # the readable caller survives



def test_find_function_exact_hit_walks_once_and_then_never_again(monkeypatch):
    """#622(b) for the SINGLE-identifier path: an exact name hit pays the one
    index build and every later hit enumerates nothing.

    BN's own name index deliberately does NOT answer here even though it could:
    it is a strict subset of the walk (#224a), and a unique non-stub hit from it
    suppressed the ambiguous-identifier error for two real bodies sharing a
    spelling (#122, pinned by
    `test_find_function_reports_ambiguity_the_native_index_cannot_see`). The
    per-view index delivers the same zero-enumeration hit soundly, from the
    second lookup on."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _named_fn(0x401000, "big_dispatch")
    bv = _NotifyingBV(functions=[fn])
    bv.functions = _CountingFunctions([fn])
    bv.get_functions_by_name = lambda name: [fn] if name == "big_dispatch" else []

    assert int(instance._find_function(bv, "big_dispatch").start) == 0x401000
    assert bv.functions.enumerations == 1, "the cold hit builds the index once"
    for _ in range(3):
        assert int(instance._find_function(bv, "big_dispatch").start) == 0x401000
    assert bv.functions.enumerations == 1, (
        "a warm exact hit must not enumerate the view again"
    )


def test_find_function_reports_ambiguity_the_native_index_cannot_see(monkeypatch):
    """#122: two REAL bodies sharing one spelling must raise the ambiguous-name
    error, never be auto-picked.

    BN's own name index is a strict SUBSET of the walk -- it does not carry the
    demangled short/full spellings BN keeps only on the symbol (#224a) -- so a
    unique non-stub hit from it is NO evidence that the spelling is unique. Here a
    C body is literally named `handle` while a C++ body carries `handle` only as
    its demangled short name, exactly the mixed-language shape of a real target;
    the index resolves only the C body. Accepting that hit resolved a genuine
    two-implementation ambiguity silently.

    REGRESSION test: this passes at the base revision (which always walked) and
    fails only on the revisions that let the native index answer -- it pins
    behaviour that must be RESTORED, so being green on base is the point, not
    missing evidence."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    c_body = _named_fn(0x401000, "handle")
    cpp_body = _named_fn(0x402000, "_Z6handlev", short_name="handle")
    bv = _FakeBV(functions=[c_body, cpp_body])
    bv.get_functions_by_name = lambda name: [c_body] if name == "handle" else []

    with pytest.raises(RuntimeError) as exc_info:
        instance._find_function(bv, "handle")
    message = str(exc_info.value)
    assert "Ambiguous function identifier" in message, message
    assert "0x00401000" in message and "0x00402000" in message, message


def test_find_function_index_miss_still_walks_for_the_case_exact_spelling(monkeypatch):
    """INVARIANT GUARD -- green at the base revision by construction (base always
    walked), so it is not regression evidence for this PR; it is what fails if any
    later change lets an EMPTY result from BN's own name index end the lookup.

    That result is never authoritative: the index does not carry the demangled
    spellings BN keeps only on the symbol (#224a), so the walk must answer the
    exact-case query. Two functions whose demangled spellings differ only in CASE
    expose a fallback that stops at the empty index: folding them matches BOTH and
    raises the ambiguous-name error instead of returning the spelling the caller
    asked for."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    def _fn(start, raw, short):
        fn = _FakeFunction(start, raw)
        fn.symbol = _FakeSymbol("FunctionSymbol")
        fn.symbol.short_name = short
        return fn

    bv = _FakeBV(functions=[
        _fn(0x406000, "_ZN3pkg5ThingEv", "pkg::Thing"),
        _fn(0x407000, "_ZN3pkg5thingEv", "pkg::thing"),
    ])
    bv.get_functions_by_name = lambda name: []

    assert int(instance._find_function(bv, "pkg::Thing").start) == 0x406000


def _suggestion_spellings(fns) -> list[str]:
    """Every spelling of *fns* in the order the miss corpus collects them: the
    function's own name/raw_name plus the symbol's demangled short/full name,
    de-duplicated per function. Recomputed here (not imported from the seam) so
    the expectation below is independent of the code under test."""
    out: list[str] = []
    for fn in fns:
        sym = getattr(fn, "symbol", None)
        forms: list[str] = []
        for value in (
            getattr(fn, "name", None),
            getattr(fn, "raw_name", None),
            getattr(sym, "short_name", None),
            getattr(sym, "full_name", None),
        ):
            if value and str(value) not in forms:
                forms.append(str(value))
        out.extend(forms)
    return out


def _named_method_bv():
    """A synthetic C++-style view: one intended method whose demangled name lives
    only on the symbol (#224a), plus an unrelated helper. Synthetic names only."""
    def _fn(start, raw, short, full):
        fn = _FakeFunction(start, raw)
        fn.symbol = _FakeSymbol("FunctionSymbol")
        fn.symbol.short_name = short
        fn.symbol.full_name = full
        return fn

    return [
        _fn(0x401000, "_ZN3net7Session6onDataEi",
            "net::Session::onData", "net::Session::onData(int32_t)"),
        _fn(0x402000, "_ZN3net12ZlibChecksumEj",
            "net::ZlibChecksum", "net::ZlibChecksum(uint32_t)"),
    ]


def _miss_suggestions(instance, bv, query: str) -> list[str]:
    with pytest.raises(RuntimeError) as exc_info:
        instance._find_function(bv, query)
    message = str(exc_info.value)
    assert "Did you mean: " in message, message
    return message.split("Did you mean: ", 1)[1].split(", ")


class _CountingFunctions(list):
    """A view's function list that counts how many times it is enumerated."""

    def __init__(self, functions):
        super().__init__(functions)
        self.enumerations = 0

    def __iter__(self):
        self.enumerations += 1
        return super().__iter__()


class _MapLookupBV(_FakeBV):
    """A view double whose ``get_function_at`` is a MAP lookup, as real BN's is.

    ``BinaryView.get_function_at`` is a map lookup in the core; the base double
    instead scans ``self.functions``, which is the ``_CountingFunctions`` list the
    tests wrap -- so a lookup that resolves a cached bucket would be counted as a
    full walk. Iterating the plain list storage keeps ``enumerations`` meaning
    exactly "a full ``bv.functions`` walk".
    """

    def get_function_at(self, address: int):
        for fn in list.__iter__(self.functions):
            if int(fn.start) == int(address):
                return fn
        return None


class _NotifyingBV(_MapLookupBV):
    """A view double that supports BN's view notifications.

    ``fire`` stands in for BN's own callbacks: it invokes ``event`` on every
    registered notifier, synchronously, exactly as the core does inside the
    mutating call -- which is what makes a generation bump happen-before the next
    read. A test that models a change the core does NOT report simply mutates the
    double and never calls ``fire``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._notifiers: list = []

    def register_notification(self, notifier):
        self._notifiers.append(notifier)

    def unregister_notification(self, notifier):
        self._notifiers.remove(notifier)

    def fire(self, event, *args):
        for notifier in list(self._notifiers):
            getattr(notifier, event)(self, *args)


def _named_fn(start, name, short_name=None):
    """A function whose spellings are its name plus a symbol short name."""
    fn = _FakeFunction(start, name)
    fn.symbol = _FakeSymbol("FunctionSymbol")
    fn.symbol.short_name = short_name or name
    return fn


def test_name_lookup_reuses_the_per_view_index_and_a_change_invalidates_it(monkeypatch):
    """#622(b): the per-view index is built once and reused while BN reports no
    change (a warm lookup enumerates nothing), and BN's own change notification
    invalidates it -- the rename is visible on the very next lookup, for exactly
    the one enumeration the rebuild costs."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _named_fn(0x401000, "alpha")
    other = _named_fn(0x402000, "beta")
    bv = _NotifyingBV(functions=[fn, other])
    bv.functions = _CountingFunctions([fn, other])

    def lookup(text):
        return [
            int(f.start)
            for f in instance.ctx._find_functions_by_name(
                bv, text, case_sensitive=True)
        ]

    assert lookup("alpha") == [0x401000]
    assert bv.functions.enumerations == 1
    assert lookup("alpha") == [0x401000]
    assert bv.functions.enumerations == 1, "a warm lookup must not enumerate the view"

    # A rename, reported by BN's own notification.
    fn.name = "gamma"
    fn.raw_name = "gamma"
    fn.symbol.short_name = "gamma"
    bv.fire("symbol_updated", fn)

    assert lookup("gamma") == [0x401000]
    assert bv.functions.enumerations == 2, "the change must force exactly one rebuild"
    assert lookup("alpha") == []
    assert bv.functions.enumerations == 2, "a miss on the fresh index must not rebuild"
    assert lookup("gamma") == [0x401000]
    assert bv.functions.enumerations == 2


def test_name_lookup_never_caches_a_view_without_notification_support(monkeypatch):
    """NEGATIVE CONTROL for the safe-fallback rule -- this passes at the base by
    construction, so it is NOT regression evidence for the cache itself: a view
    with no notification surface has no sound invalidation signal, so it is never
    cached and two identical lookups each enumerate the view, i.e. the walk
    behaviour that predates the index. The cache -- reuse while BN reports no
    change, invalidation by BN's own notification -- is pinned by
    `test_name_lookup_reuses_the_per_view_index_and_a_change_invalidates_it`."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _named_fn(0x401000, "alpha")

    def lookup(bv):
        return [
            int(f.start)
            for f in instance.ctx._find_functions_by_name(
                bv, "alpha", case_sensitive=True)
        ]

    bv = _MapLookupBV(functions=[fn])
    bv.functions = _CountingFunctions([fn])
    assert lookup(bv) == [0x401000]
    assert lookup(bv) == [0x401000]
    assert bv.functions.enumerations == 2, (
        "a view that cannot report changes must walk on every lookup"
    )


def test_name_index_drops_a_stale_bucket_after_an_unnotified_rename(monkeypatch):
    """A change BN does NOT report leaves the generation counter untouched, so the
    cached bucket must be re-verified against the live view: the old spelling stops
    resolving (the stale member no longer carries it) and the new one resolves from
    the rebuild, at the cost of that one rebuild."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _named_fn(0x401000, "alpha")
    bv = _NotifyingBV(functions=[fn])
    bv.functions = _CountingFunctions([fn])

    def lookup(text):
        return [
            int(f.start)
            for f in instance.ctx._find_functions_by_name(
                bv, text, case_sensitive=True)
        ]

    assert lookup("alpha") == [0x401000]
    assert bv.functions.enumerations == 1

    # Renamed behind BN's back: no notification is fired.
    fn.name = "gamma"
    fn.raw_name = "gamma"
    fn.symbol.short_name = "gamma"

    assert lookup("alpha") == [], "the stale bucket must not answer"
    assert bv.functions.enumerations == 2, "the live guard must rebuild exactly once"
    assert lookup("gamma") == [0x401000]
    assert lookup("gamma") == [0x401000]
    assert bv.functions.enumerations == 2, "the rebuilt index must be reused"


def test_name_index_rebuilds_when_the_native_index_witnesses_an_unknown_spelling(monkeypatch):
    """A stale NEGATIVE is caught by BN's own name index: it witnesses that a
    function carrying the queried spelling exists even though the cached index has
    no bucket for it. The witness only forces a WALK-backed rebuild -- it never
    becomes the answer -- so a spelling carried by several functions must come back
    as the complete group even though the witness itself returned a member subset."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _named_fn(0x401000, "alpha")
    bv = _NotifyingBV(functions=[fn])
    bv.functions = _CountingFunctions([fn])

    def lookup(text):
        return [
            int(f.start)
            for f in instance.ctx._find_functions_by_name(
                bv, text, case_sensitive=True)
        ]

    assert lookup("alpha") == [0x401000]
    assert bv.functions.enumerations == 1

    # An unnotified late addition: the first function gains the spelling "shared"
    # and a second function carrying it appears in the view. The native index
    # resolves the spelling, but only to the new member.
    twin = _named_fn(0x403000, "shared")
    fn.symbol.short_name = "shared"
    bv.functions.append(twin)
    bv.get_functions_by_name = lambda name: [twin] if name == "shared" else []

    assert lookup("shared") == [0x401000, 0x403000], (
        "the witness must trigger the walk-backed rebuild, not supply the answer"
    )
    assert bv.functions.enumerations == 2
    assert lookup("shared") == [0x401000, 0x403000]
    assert bv.functions.enumerations == 2


def test_cached_bucket_with_a_missing_member_still_rebuilds(monkeypatch):
    """#622 review (blocker B): the completeness check is SYMMETRIC. A cached
    NON-empty bucket that lacks a member BN's own name index witnesses for the
    queried spelling must be rebuilt, so an unnotified same-name addition cannot be
    served as an incomplete group (which is how a veneer caller silently drops out
    of the xrefs union). The witness NEVER supplies the answer -- the walk-backed
    rebuild returns the COMPLETE group, in view order."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _named_fn(0x401000, "alpha")
    bv = _NotifyingBV(functions=[fn])
    bv.functions = _CountingFunctions([fn])

    def lookup(text):
        return [
            int(f.start)
            for f in instance.ctx._find_functions_by_name(
                bv, text, case_sensitive=True)
        ]

    assert lookup("alpha") == [0x401000]
    assert bv.functions.enumerations == 1

    # An unnotified SAME-NAME addition: a second function carrying the spelling
    # appears in the view, no notification fires, and BN's own name index resolves
    # the spelling -- to the new member only. The cached bucket is NON-empty, so
    # only the symmetric check can notice that the group is incomplete.
    twin = _named_fn(0x403000, "alpha")
    bv.functions.append(twin)
    bv.get_functions_by_name = lambda name: [twin] if name == "alpha" else []

    assert lookup("alpha") == [0x401000, 0x403000], (
        "an incomplete cached same-name group must not be served"
    )
    assert bv.functions.enumerations == 2, (
        "the completeness witness must force exactly one rebuild"
    )
    assert lookup("alpha") == [0x401000, 0x403000]
    assert bv.functions.enumerations == 2, "the rebuilt index must be reused"


def test_cached_folded_group_with_a_missing_member_still_rebuilds(monkeypatch):
    """#622 review (round-4 blocker): the completeness witness must be fetched in
    the VIEW's own casing, not the queried one. BN's name index is case-SENSITIVE,
    so witnessing a case-insensitive lookup under the QUERIED spelling witnesses
    nothing at all whenever the view spells the name differently -- which left
    every memo-served folded group with no completeness check, and an unnotified
    same-name addition served as an incomplete group (the veneer-caller drop-out
    class, #286). The witness still only forces the walk-backed rebuild; the
    rebuild supplies the COMPLETE group, in view order."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _named_fn(0x401000, "Alpha")
    bv = _NotifyingBV(functions=[fn])
    bv.functions = _CountingFunctions([fn])

    def lookup(text):
        return [
            int(f.start)
            for f in instance.ctx._find_functions_by_name(
                bv, text, case_sensitive=False)
        ]

    assert lookup("alpha") == [0x401000], "the folded bucket must answer the miss"
    assert bv.functions.enumerations == 1
    assert lookup("alpha") == [0x401000]
    assert bv.functions.enumerations == 1, (
        "a warm case-insensitive lookup must not enumerate the view"
    )

    # An unnotified SAME-NAME addition, spelled the way the VIEW spells it. BN's
    # own index resolves "Alpha" and can never be asked for "alpha", so only a
    # witness fetched in the view's casing can see the cached group is incomplete.
    twin = _named_fn(0x403000, "Alpha")
    bv.functions.append(twin)
    bv.get_functions_by_name = lambda name: [twin] if name == "Alpha" else []

    assert lookup("alpha") == [0x401000, 0x403000], (
        "an incomplete cached case-insensitive group must not be served"
    )
    assert bv.functions.enumerations == 2, (
        "the completeness witness must force exactly one rebuild"
    )
    assert lookup("alpha") == [0x401000, 0x403000]
    assert bv.functions.enumerations == 2, "the rebuilt index must be reused"


def test_an_addition_in_an_unwitnessable_casing_still_invalidates_the_group(monkeypatch):
    """#622 review (round-5 major): BN's own name index is case-SENSITIVE, so a
    same-name addition spelled in a casing NOTHING asked for and NO cached member
    carries cannot be witnessed by name at all -- the queried casing witnesses
    only itself, and the cached members' casings witness only theirs.

    The view's function COUNT is the case-independent witness of last resort: the
    addition changes it, so the cached group is rebuilt once and comes back
    COMPLETE rather than serving one of two same-name members."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _named_fn(0x401000, "alpha")
    bv = _NotifyingBV(functions=[fn])
    bv.functions = _CountingFunctions([fn])

    def lookup(text):
        return [
            int(f.start)
            for f in instance.ctx._find_functions_by_name(
                bv, text, case_sensitive=False)
        ]

    assert lookup("alpha") == [0x401000]
    assert bv.functions.enumerations == 1
    assert lookup("alpha") == [0x401000]
    assert bv.functions.enumerations == 1, "the memo must really be serving it"

    # A THIRD casing: not the queried "alpha", not the cached member's "alpha".
    twin = _named_fn(0x403000, "ALPHA")
    bv.functions.append(twin)
    bv.get_functions_by_name = lambda name: [twin] if name == "ALPHA" else []

    assert lookup("alpha") == [0x401000, 0x403000], (
        "an addition no name witness can see must still invalidate the group"
    )
    assert bv.functions.enumerations == 2, "exactly one rebuild"
    assert lookup("alpha") == [0x401000, 0x403000]
    assert bv.functions.enumerations == 2, "the rebuilt index must be reused"


def test_a_vanished_cached_start_invalidates_the_group(monkeypatch):
    """#622 review (round-6 major): the count witness is blind to a change that
    PRESERVES the count -- an unnotified ADDITION paired with a REMOVAL. When the
    removed function was in the cached group, the start the index recorded stops
    resolving, and that is proof the view changed: no spelling, no casing and no
    count difference needed.

    Without it the group is quietly served as the surviving members, which reads
    as a complete same-name group -- the exact shape that drops a veneer from the
    xrefs caller union (#286). Here nothing else can catch it: the count is
    unchanged and BN's name index is not answering at all."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    first = _named_fn(0x401000, "alpha")
    second = _named_fn(0x402000, "alpha")
    bv = _NotifyingBV(functions=[first, second])
    bv.functions = _CountingFunctions([first, second])

    def lookup(text):
        return [
            int(f.start)
            for f in instance.ctx._find_functions_by_name(
                bv, text, case_sensitive=True)
        ]

    assert lookup("alpha") == [0x401000, 0x402000]
    assert bv.functions.enumerations == 1
    assert lookup("alpha") == [0x401000, 0x402000]
    assert bv.functions.enumerations == 1, "the group must be memo-served"

    # Unnotified, and COUNT-PRESERVING: one group member goes away, another
    # same-name function appears. BN's name index answers nothing here, so the
    # vanished start is the only remaining evidence.
    bv.functions.remove(second)
    bv.functions.append(_named_fn(0x403000, "alpha"))
    assert len(bv.functions) == 2, "the count must be unchanged for this to bite"

    assert lookup("alpha") == [0x401000, 0x403000], (
        "a cached start the view no longer has must invalidate, not shrink, the group"
    )
    assert bv.functions.enumerations == 2, "exactly one rebuild"
    assert lookup("alpha") == [0x401000, 0x403000]
    assert bv.functions.enumerations == 2, "the rebuilt index must be reused"


def test_an_empty_cached_group_is_still_rechecked_against_the_view(monkeypatch):
    """#622 review (round-5 major): an EMPTY cached bucket has no member whose
    casing could be asked for, so a case-insensitive lookup has nothing to witness
    with -- BN's index can only be asked for the queried casing, and the view
    spells the name differently. The count witness covers exactly that: the
    unnotified addition changes the view's function count, so the empty group is
    rebuilt instead of being served as "no such function"."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    unrelated = _named_fn(0x401000, "zeta")
    bv = _NotifyingBV(functions=[unrelated])
    bv.functions = _CountingFunctions([unrelated])

    def lookup(text):
        return [
            int(f.start)
            for f in instance.ctx._find_functions_by_name(
                bv, text, case_sensitive=False)
        ]

    assert lookup("alpha") == []
    assert bv.functions.enumerations == 1
    assert lookup("alpha") == []
    assert bv.functions.enumerations == 1, "the empty group must be memo-served"

    twin = _named_fn(0x403000, "Alpha")
    bv.functions.append(twin)
    bv.get_functions_by_name = lambda name: [twin] if name == "Alpha" else []

    assert lookup("alpha") == [0x403000], (
        "an empty cached group with no witness must be rechecked, not served"
    )
    assert bv.functions.enumerations == 2, "exactly one rebuild"
    assert lookup("alpha") == [0x403000]
    assert bv.functions.enumerations == 2, "the rebuilt index must be reused"


def test_name_lookup_walks_a_view_that_cannot_resolve_a_start_address(monkeypatch):
    """The index stores start ADDRESSES and resolves them at lookup time (storing
    a Function would pin its view), so it can only answer a view that maps an
    address back to a function. A view without ``get_function_at`` must therefore
    be WALKED -- answering it from the index yields the empty group resolution
    produces, i.e. a silent "not found" for a function the view has.

    REGRESSION test: green at the base revision (which always walked); it fails
    only while the index path answers such a view from an unresolvable bucket."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _named_fn(0x401000, "alpha")

    class _NoResolveBV(_NotifyingBV):
        get_function_at = None          # a BinaryView-shaped double without it

    bv = _NoResolveBV(functions=[fn])
    bv.functions = _CountingFunctions([fn])

    def lookup(text, *, case_sensitive):
        return [
            int(f.start)
            for f in instance.ctx._find_functions_by_name(
                bv, text, case_sensitive=case_sensitive)
        ]

    assert lookup("alpha", case_sensitive=True) == [0x401000]
    assert lookup("ALPHA", case_sensitive=False) == [0x401000]
    assert bv.functions.enumerations == 2, (
        "a view the index cannot serve must walk on every lookup"
    )



def test_name_index_does_not_retain_a_closed_target(monkeypatch):
    """#622 review (blocker A): the memo must hold NOTHING that reaches the view
    back. A real BN Function strongly references its own view (``fn.view is bv``),
    so a bucket of Functions would keep the ``WeakKeyDictionary`` weak key alive and
    retain a closed target's full spelling corpus for the process lifetime. With
    the buckets holding start addresses only -- resolved through
    ``bv.get_function_at`` at lookup time -- dropping the view drops the index."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    def warm_and_release():
        fns = [_named_fn(0x401000, "alpha"), _named_fn(0x402000, "beta")]
        bv = _NotifyingBV(functions=fns)
        bv.functions = _CountingFunctions(fns)
        for fn in fns:
            fn.view = bv  # exactly as a real BN Function does

        def lookup(text):
            return [
                int(f.start)
                for f in instance.ctx._find_functions_by_name(
                    bv, text, case_sensitive=True)
            ]

        assert lookup("alpha") == [0x401000]
        assert bv.functions.enumerations == 1
        assert lookup("alpha") == [0x401000]
        assert bv.functions.enumerations == 1, "the index must really be warm"
        return weakref.ref(bv)

    ref = warm_and_release()
    gc.collect()
    assert ref() is None, (
        "the cached name index retained the closed target: no memo value may "
        "reference the view it is keyed on"
    )


def test_find_function_miss_hints_are_unfiltered_and_cost_one_enumeration(monkeypatch):
    """#622(b): a miss suggests over every spelling within the budget -- no prefix
    filter, no sampling, no length prefilter, so a FIRST-character typo still finds
    the intended name -- and costs exactly ONE `bv.functions` enumeration. Base
    walked the view three times per miss (exact, casefold, then the suggestion
    corpus). The hint set must stay difflib's over that corpus, in view order, for
    both typo shapes. (The budget itself is pinned by
    `test_find_function_miss_hints_are_bounded_and_the_budget_is_disclosed`.)"""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fns = _named_method_bv()
    bv = _FakeBV(functions=fns)
    bv.functions = _CountingFunctions(fns)
    bv.get_functions_by_name = lambda name: []
    spellings = _suggestion_spellings(fns)

    for typo in ("uet::Session::onData", "net::Sessoin::onData"):
        bv.functions.enumerations = 0
        expected = difflib.get_close_matches(typo, spellings, n=5, cutoff=0.5)
        assert "net::Session::onData" in expected     # in range for both typos

        hints = _miss_suggestions(instance, bv, typo)

        assert hints == expected
        assert "net::Session::onData" in hints
        assert not any("Zlib" in hint for hint in hints)
        assert bv.functions.enumerations == 1, (
            f"a miss must enumerate the view once, not "
            f"{bv.functions.enumerations} times"
        )


def test_find_function_miss_hints_are_bounded_and_the_budget_is_disclosed(monkeypatch):
    """#622(b) as a LATENCY bound: the hint search is O(all spellings) -- difflib
    gates every candidate it is handed -- so the number of candidates is capped,
    and the cap is DISCLOSED in the message instead of silently changing the hint
    ("prefer bounded latency + honest truncation over silent incompleteness").

    The budget is exercised through a patched constant rather than a 20k-function
    view, so what is pinned is the behaviour (at most `SUGGESTION_CORPUS_MAX`
    candidates, and the message naming how many of how many spellings were
    searched), not the constant's value."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    seam = sys.modules[type(instance.ctx).__module__]
    fns = [_named_fn(0x401000 + i * 0x100, f"handler_{i:03d}") for i in range(8)]
    bv = _NotifyingBV(functions=fns)
    bv.functions = _CountingFunctions(fns)
    bv.get_functions_by_name = lambda name: []

    searched: list[int] = []
    real_matcher = difflib.get_close_matches

    def _spy(word, possibilities, **kwargs):
        searched.append(len(possibilities))
        return real_matcher(word, possibilities, **kwargs)

    monkeypatch.setattr(seam, "SUGGESTION_CORPUS_MAX", 3)
    monkeypatch.setattr(seam.difflib, "get_close_matches", _spy)

    with pytest.raises(RuntimeError) as exc_info:
        instance._find_function(bv, "handler_00X")
    message = str(exc_info.value)

    assert searched == [3], (
        f"the hint search must see at most the budget, saw {searched}"
    )
    assert "the first 3 of 8 spellings" in message, message
    assert "suggestion budget" in message, message
    # Under the budget the message is unchanged -- no note, nothing to disclose.
    monkeypatch.setattr(seam, "SUGGESTION_CORPUS_MAX", 8)
    with pytest.raises(RuntimeError) as exc_info:
        instance._find_function(bv, "handler_00X")
    assert "suggestion budget" not in str(exc_info.value), str(exc_info.value)
    assert searched == [3, 8]


def test_find_function_stub_does_not_shadow_the_implementation(monkeypatch):
    """INVARIANT GUARD -- green at base by construction (base always walked), so
    it is not regression evidence for this PR; it fails if any later change lets a
    unique native-index hit answer a single identifier.

    An import stub can shadow a same-name real body (#122/#286), and only the
    walk's full match set lets the impl-over-stub resolver pick the body. BN's own
    name index resolving just the stub must never become the answer -- the real
    body is returned."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    def _fn(start, type_name):
        fn = _FakeFunction(start, "shared_entry")
        fn.symbol = _FakeSymbol(type_name)
        fn.symbol.short_name = "shared_entry"
        return fn

    stub = _fn(0x400000, "ImportedFunctionSymbol")
    body = _fn(0x401000, "FunctionSymbol")
    bv = _FakeBV(functions=[stub, body])
    bv.get_functions_by_name = lambda name: [stub]

    resolved = instance._find_function(bv, "shared_entry")

    assert int(resolved.start) == 0x401000


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


def test_adrp_pagebase_nonzero_then_zero_offset_is_genuine(monkeypatch):
    # `adrp x8, A` then `add x1, x8, #0x40` (nonzero) then `add x2, x8, #0`
    # (zero) in the same block -- the zero-offset consumer later in the scan
    # window is decisive genuine evidence; must NOT drop despite the earlier
    # nonzero consumer (#583, the literal repro: `add x1,x8,#0x40` then
    # `add x2,x8,#0`).
    A = 0x438000
    adrp = _adrp("x8", A, 0)
    add_nonzero = _set_reg("x1", _LOp("LLIL_ADD", [_reg("x8"), _const(0x40)]), 1)
    add_zero = _set_reg("x2", _LOp("LLIL_ADD", [_reg("x8"), _const(0)]), 2)
    assert _spurious(monkeypatch, adrp, [add_nonzero, add_zero], A) is False


def test_adrp_pagebase_zero_then_nonzero_offset_is_genuine(monkeypatch):
    # Order reversed from the case above -- already worked before the fix,
    # must still retain.
    A = 0x438000
    adrp = _adrp("x8", A, 0)
    add_zero = _set_reg("x2", _LOp("LLIL_ADD", [_reg("x8"), _const(0)]), 1)
    add_nonzero = _set_reg("x1", _LOp("LLIL_ADD", [_reg("x8"), _const(0x40)]), 2)
    assert _spurious(monkeypatch, adrp, [add_zero, add_nonzero], A) is False


def test_adrp_pagebase_two_nonzero_offsets_no_zero_is_spurious(monkeypatch):
    # Two nonzero-offset consumers, no zero-offset consumer anywhere -- must
    # still drop (confirms the fix didn't become overly conservative).
    A = 0x438000
    adrp = _adrp("x8", A, 0)
    add1 = _set_reg("x1", _LOp("LLIL_ADD", [_reg("x8"), _const(0x40)]), 1)
    add2 = _set_reg("x2", _LOp("LLIL_ADD", [_reg("x8"), _const(0x60)]), 2)
    assert _spurious(monkeypatch, adrp, [add1, add2], A) is True


def test_adrp_pagebase_nonzero_then_redefinition_no_zero_is_spurious(monkeypatch):
    # A nonzero-offset consumer followed by a redefinition of the page-base
    # register with no zero-offset consumer before the redefinition -- must
    # still drop.
    A = 0x438000
    adrp = _adrp("x8", A, 0)
    add_nonzero = _set_reg("x1", _LOp("LLIL_ADD", [_reg("x8"), _const(0x40)]), 1)
    redef = _set_reg("x8", _reg("x5"), 2)
    assert _spurious(monkeypatch, adrp, [add_nonzero, redef], A) is True


def test_adrp_pagebase_selfredefine_then_zero_offset_is_spurious(monkeypatch):
    # `adrp x8, A` then `add x8, x8, #0x40` (REDEFINES x8 to A+0x40) then
    # `add x2, x8, #0` (zero-offset use of the REDEFINED x8). The zero-offset
    # consumer references A+0x40, NOT the original page base A, so the adrp
    # page-base xref to a function starting at A is spurious. The middle
    # instruction both consumes x8 (nonzero) AND redefines it, so tracking of
    # the page-base identity must stop after it -- the later zero offset must
    # not be credited to the stale page base.
    A = 0x438000
    adrp = _adrp("x8", A, 0)
    add_redef = _set_reg("x8", _LOp("LLIL_ADD", [_reg("x8"), _const(0x40)]), 1)
    add_zero = _set_reg("x2", _LOp("LLIL_ADD", [_reg("x8"), _const(0)]), 2)
    assert _spurious(monkeypatch, adrp, [add_redef, add_zero], A) is True


def test_adrp_pagebase_selfredefine_then_direct_use_is_spurious(monkeypatch):
    # `adrp x8, A` then `add x8, x8, #0x40` (REDEFINES x8) then `mov x1, x8`
    # (direct use of REDEFINED x8). The direct use takes &(A+0x40), not &A, so
    # the page-base xref to A is spurious -- tracking must stop at the redefine.
    A = 0x438000
    adrp = _adrp("x8", A, 0)
    add_redef = _set_reg("x8", _LOp("LLIL_ADD", [_reg("x8"), _const(0x40)]), 1)
    direct = _set_reg("x1", _reg("x8"), 2)
    assert _spurious(monkeypatch, adrp, [add_redef, direct], A) is True


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


def test_xrefs_keeps_stub_callers_when_the_native_index_answers_a_subset(monkeypatch):
    """INVARIANT GUARD for the round-3 review BLOCKER -- green at base by
    construction (base always walked), and proven non-vacuous by mutation: making
    the group lookup answer from the native index turns it RED.

    The same-name group callers need EVERY member, but BN's own name index can
    only ever return a strict SUBSET of the authoritative walk -- it does not carry
    the demangled short/full spellings BN keeps only on the symbol (#224a). With an
    index that resolves just the real body, `xrefs <name>` must STILL union the
    veneer's callers: answering from the subset silently drops them and reads a hot
    function as zero-caller with no flag."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, caller, stub, impl = _impl_stub_bv(stub_ref_addrs=[0x500010])
    bv.get_functions_by_name = lambda name: [impl]      # subset: the veneer is absent
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._xrefs(None, "get_param")
    assert result["code_ref_count"] == 1
    assert "0x40f1a0" in result.get("stub_callers_via", [])


def test_same_name_stub_union_survives_a_subset_native_index(monkeypatch):
    """INVARIANT GUARD, sibling of the test above -- green at base by construction
    and proven non-vacuous by the same mutation.

    The group lookup is walk-backed even when the native index answers: it must
    return the COMPLETE same-name group (both members, view order) so the stub union
    sees the veneer, and the stub set must be derived from that full group rather
    than from the index's subset."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, caller, stub, impl = _impl_stub_bv()
    bv.get_functions_by_name = lambda name: [impl]
    assert [
        int(f.start)
        for f in instance.ctx._find_functions_by_name(bv, "get_param", case_sensitive=True)
    ] == [0x40f1a0, 0x5c40]
    assert [int(f.start) for f in instance.ctx._same_name_stub_functions(bv, impl)] == [0x40f1a0]


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
    # Unscoped/all-callers mode must derive callers from the exported symbol's
    # same-name stub references without admitting a different export's stub.
    all_callers = instance._callsites(
        None, "get_param", within_identifiers=[]
    )
    assert [row["containing_function"]["name"] for row in all_callers["items"]] == [
        "caller"
    ]


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
