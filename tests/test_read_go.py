from __future__ import annotations

import importlib
import struct

import pytest

from _bridge_fakes import _load_bridge


def _build_pclntab(*, magic=0xFFFFFFF1, ptr_size=8, text_start=0x400000):
    """A minimal Go 1.20-format .gopclntab with two functions:
    main.foo @ text_start+0x1000, main.bar @ text_start+0x2000."""
    names = b"main.foo\x00main.bar\x00"
    funcname_off = 72
    pcln_off = 96
    func0_off, func1_off = 128, 136          # relative to pcln_off
    blob = bytearray(300)
    struct.pack_into("<I", blob, 0, magic)
    blob[6] = 1                               # minLC
    blob[7] = ptr_size
    struct.pack_into("<Q", blob, 8, 2)        # nfunc
    struct.pack_into("<Q", blob, 24, text_start)
    struct.pack_into("<Q", blob, 32, funcname_off)
    struct.pack_into("<Q", blob, 64, pcln_off)
    blob[funcname_off:funcname_off + len(names)] = names
    struct.pack_into("<I", blob, pcln_off + 4, func0_off)    # functab entry0 funcoff
    struct.pack_into("<I", blob, pcln_off + 12, func1_off)   # functab entry1 funcoff
    struct.pack_into("<I", blob, pcln_off + func0_off, 0x1000)     # _func0 entryoff
    struct.pack_into("<i", blob, pcln_off + func0_off + 4, 0)      # nameoff -> main.foo
    struct.pack_into("<I", blob, pcln_off + func1_off, 0x2000)     # _func1 entryoff
    struct.pack_into("<i", blob, pcln_off + func1_off + 4, 9)      # nameoff -> main.bar
    return bytes(blob)


class _GoBV:
    def __init__(self, blob, *, base=0x500000, defined=(), text_start=0x400000):
        self._blob = blob or b""
        self._base = base
        self._defined = set(defined)
        self._text_start = text_start
        self.sections = {}

    def get_section_by_name(self, name):
        if name == ".gopclntab" and self._blob:
            return type("S", (), {"start": self._base, "length": len(self._blob),
                                  "end": self._base + len(self._blob)})()
        if name == ".text":
            return type("S", (), {"start": self._text_start})()
        return None

    def read(self, addr, size):
        o = addr - self._base
        return self._blob[o:o + size] if 0 <= o else b""

    def get_function_at(self, addr):
        return object() if addr in self._defined else None


def _ctx(monkeypatch, bv):
    bridge = _load_bridge(monkeypatch)
    inst = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(inst.ctx, "_resolve_view", lambda sel: bv)
    monkeypatch.setattr(inst.ctx, "_byteorder", lambda _bv: "little")
    return bridge, inst


def test_go_functions_recovers_names_and_addresses(monkeypatch):
    blob = _build_pclntab()
    bv = _GoBV(blob, defined={0x401000})        # only main.foo is a BN function
    bridge, inst = _ctx(monkeypatch, bv)
    out = inst._go_functions(None)
    assert out["kind"] == "go_functions" and out["go_version"] == "go1.20"
    by_name = {i["name"]: i for i in out["items"]}
    assert by_name["main.foo"]["address"] == hex(0x401000)
    assert by_name["main.foo"]["defined"] is True
    assert by_name["main.bar"]["address"] == hex(0x402000)
    assert by_name["main.bar"]["defined"] is False
    assert out["total"] == 2 and out["defined_count"] == 1


def test_go_functions_rebase_note_when_nothing_maps(monkeypatch):
    # #217: when no recovered address maps to a BN function AND the pcln textStart
    # differs from BN's .text start (PIE), disclose the rebase rather than emitting
    # silently-wrong addresses.
    bv = _GoBV(_build_pclntab(text_start=0x400000), defined=set(), text_start=0x800000)
    bridge, inst = _ctx(monkeypatch, bv)
    out = inst._go_functions(None)
    assert out["defined_count"] == 0
    assert "PIE" in out["note"] and "rebase" in out["note"].lower()


def test_go_functions_incomplete_analysis_note_when_text_matches(monkeypatch):
    # #217 review: 0 mapped but text starts MATCH -> attribute to incomplete
    # analysis / rebase ambiguity, NOT confidently to PIE.
    bv = _GoBV(_build_pclntab(text_start=0x400000), defined=set(), text_start=0x400000)
    bridge, inst = _ctx(monkeypatch, bv)
    out = inst._go_functions(None)
    assert out["defined_count"] == 0
    assert "PIE" not in out["note"] and "refresh" in out["note"]


def test_go_functions_short_header_is_honest(monkeypatch):
    # #217 review (MEDIUM): a short section with a valid magic must NOT throw an
    # opaque struct.error from the unbounded header reads -- decline honestly.
    import struct as _s
    blob = bytearray(40)
    _s.pack_into("<I", blob, 0, 0xFFFFFFF1)   # valid magic, but only 40 bytes
    blob[7] = 8
    bv = _GoBV(bytes(blob))
    bridge, inst = _ctx(monkeypatch, bv)
    with pytest.raises(bridge.OperationFailure) as exc:
        inst._go_functions(None)
    assert exc.value.status == "short_gopclntab"


def test_go_functions_no_gopclntab_is_honest(monkeypatch):
    bv = _GoBV(b"")                               # no .gopclntab
    bridge, inst = _ctx(monkeypatch, bv)
    with pytest.raises(bridge.OperationFailure) as exc:
        inst._go_functions(None)
    assert exc.value.status == "no_gopclntab"


def test_go_functions_declines_old_format(monkeypatch):
    bv = _GoBV(_build_pclntab(magic=0xFFFFFFFA))  # Go 1.16
    bridge, inst = _ctx(monkeypatch, bv)
    with pytest.raises(bridge.OperationFailure) as exc:
        inst._go_functions(None)
    assert exc.value.status == "unsupported_pclntab_version"


def test_go_functions_declines_32bit(monkeypatch):
    bv = _GoBV(_build_pclntab(ptr_size=4))
    bridge, inst = _ctx(monkeypatch, bv)
    with pytest.raises(bridge.OperationFailure) as exc:
        inst._go_functions(None)
    assert exc.value.status == "unsupported_ptr_size"
