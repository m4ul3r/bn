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


class _FakeFn:
    def __init__(self, name):
        self.name = name


def _fake_functions(names: dict[int, str]):
    fns = {addr: _FakeFn(name) for addr, name in names.items()}
    return fns, lambda addr: fns.get(addr)


def test_go_rename_applies_recovered_names_auto_only(monkeypatch):
    # #217: go rename applies recovered Go names to AUTO-named (sub_) functions
    # only -- never clobbering a user/symbol name -- and compacts the bulk result.
    blob = _build_pclntab()                       # main.foo @0x401000, main.bar @0x402000
    bv = _GoBV(blob, defined={0x401000, 0x402000})
    fns, get_function_at = _fake_functions({
        0x401000: "sub_401000",
        0x402000: "MyHandler",
    })
    monkeypatch.setattr(bv, "get_function_at", get_function_at)
    bridge, inst = _ctx(monkeypatch, bv)
    monkeypatch.setattr(inst, "_mutation",
                        lambda *a, **k: pytest.fail("go rename must not use generic mutation"))

    out = inst._go_rename(None)
    assert out["kind"] == "go_rename"
    assert out["go_renamed_candidates"] == 1 and out["skipped_user_named"] == 1
    assert fns[0x401000].name == "main.foo"
    assert fns[0x402000].name == "MyHandler"
    # compacted: blast radius dropped, failures-only results, counts present
    assert "affected_functions" not in out
    assert out["go_verified_count"] == 1 and out["go_failed_count"] == 0
    assert out["go_committed_count"] == 1
    assert out["results"] == []


def test_go_rename_noop_when_already_named(monkeypatch):
    blob = _build_pclntab()
    bv = _GoBV(blob, defined={0x401000})
    _fns, get_function_at = _fake_functions({0x401000: "main.foo"})
    monkeypatch.setattr(bv, "get_function_at", get_function_at)
    bridge, inst = _ctx(monkeypatch, bv)
    monkeypatch.setattr(inst, "_mutation",
                        lambda *a, **k: pytest.fail("mutation must not run on a noop"))
    out = inst._go_rename(None)
    assert out["go_renamed_candidates"] == 0 and out["success"] is True and out["results"] == []


def test_go_rename_skips_undefined_pcln_addresses(monkeypatch):
    blob = _build_pclntab()
    bv = _GoBV(blob, defined=set())                # no BN function at any pcln address
    bridge, inst = _ctx(monkeypatch, bv)
    monkeypatch.setattr(inst, "_mutation",
                        lambda *a, **k: pytest.fail("must not rename undefined addresses"))
    out = inst._go_rename(None)
    assert out["go_renamed_candidates"] == 0


def test_render_go_rename_text_is_compact():
    from bn.formatters import _render_go_rename_text
    assert "nothing to do" in _render_go_rename_text(
        {"go_renamed_candidates": 0, "defined_count": 10, "skipped_user_named": 3})
    # bulk success is ONE summary line, never a per-success wall
    out = _render_go_rename_text({"go_renamed_candidates": 1782, "go_verified_count": 1782,
                                  "skipped_user_named": 1, "results": [], "committed": True})
    assert out.count("\n") == 0 and "1782 renamed" in out and "0 failed" in out
    # #217 review: a failure reverts the WHOLE batch (committed=False), so the
    # output must NOT claim the readback-passing rows as "renamed" -- nothing
    # landed. Honest wording: 0 renamed, N would have, M failed (listed).
    out2 = _render_go_rename_text({"go_renamed_candidates": 1789, "go_verified_count": 1788,
                                   "go_committed_count": 0, "skipped_user_named": 1,
                                   "success": False, "committed": False, "rolled_back": True,
                                   "results": [{"new_name": "main.x", "address": "0x1",
                                                "status": "verification_failed"}]})
    assert "0 renamed" in out2 and "1788 would have" in out2 and "main.x" in out2
    assert "1788 renamed" not in out2          # the dishonest claim is gone
    assert "NOTHING was committed" in out2

    # preview: "would rename", nothing committed
    pv = _render_go_rename_text({"go_renamed_candidates": 5, "go_verified_count": 5,
                                 "skipped_user_named": 0, "preview": True, "committed": False,
                                 "results": []})
    assert "would rename" in pv and "reverted" in pv


def test_go_rename_guard_is_exact_not_prefix(monkeypatch):
    # #217 review (paramount): the guard matches BN's EXACT `sub_<addr>` form, not a
    # `sub_` PREFIX -- so a user name like `sub_handler` is NOT clobbered.
    blob = _build_pclntab()                       # main.foo @0x401000, main.bar @0x402000
    bv = _GoBV(blob, defined={0x401000, 0x402000})
    fns, get_function_at = _fake_functions({
        0x401000: "sub_401000",
        0x402000: "sub_handler",
    })
    monkeypatch.setattr(bv, "get_function_at", get_function_at)
    bridge, inst = _ctx(monkeypatch, bv)
    monkeypatch.setattr(inst, "_mutation",
                        lambda *a, **k: pytest.fail("go rename must not use generic mutation"))
    out = inst._go_rename(None)
    assert out["go_renamed_candidates"] == 1            # only exact sub_401000
    assert out["skipped_user_named"] == 1               # sub_handler NOT clobbered
    assert fns[0x401000].name == "main.foo"
    assert fns[0x402000].name == "sub_handler"
    assert out["go_committed_count"] == 1               # committed -> landed count == verified


def test_go_rename_preview_applies_and_reverts(monkeypatch):
    blob = _build_pclntab()
    bv = _GoBV(blob, defined={0x401000})
    fns, get_function_at = _fake_functions({0x401000: "sub_401000"})
    monkeypatch.setattr(bv, "get_function_at", get_function_at)
    bridge, inst = _ctx(monkeypatch, bv)
    out = inst._go_rename(None, preview=True)
    assert out["success"] is True
    assert out["preview"] is True
    assert out["committed"] is False
    assert out["rolled_back"] is True
    assert out["go_verified_count"] == 1
    assert out["go_committed_count"] == 0
    assert fns[0x401000].name == "sub_401000"


def test_go_rename_readback_failure_rolls_back(monkeypatch):
    class _RejectingFn:
        def __init__(self):
            self._name = "sub_401000"

        @property
        def name(self):
            return self._name

        @name.setter
        def name(self, value):
            self._name = "renamed_elsewhere" if value == "main.foo" else value

    blob = _build_pclntab()
    bv = _GoBV(blob, defined={0x401000})
    fn = _RejectingFn()
    monkeypatch.setattr(bv, "get_function_at", lambda addr: fn if addr == 0x401000 else None)
    bridge, inst = _ctx(monkeypatch, bv)

    out = inst._go_rename(None)
    assert out["success"] is False
    assert out["committed"] is False
    assert out["rolled_back"] is True
    assert out["go_verified_count"] == 0
    assert out["go_failed_count"] == 1
    assert out["go_committed_count"] == 0
    assert fn.name == "sub_401000"


def test_go_rename_cancel_rolls_back(monkeypatch):
    blob = _build_pclntab()
    bv = _GoBV(blob, defined={0x401000, 0x402000})
    fns, get_function_at = _fake_functions({
        0x401000: "sub_401000",
        0x402000: "sub_402000",
    })
    monkeypatch.setattr(bv, "get_function_at", get_function_at)
    bridge, inst = _ctx(monkeypatch, bv)
    calls = {"count": 0}

    def fake_cancelled():
        calls["count"] += 1
        return calls["count"] > 1

    monkeypatch.setattr(bridge, "_request_cancelled", fake_cancelled)
    monkeypatch.setattr(bridge, "GO_RENAME_CHUNK_SIZE", 1)

    with pytest.raises(RuntimeError, match="request cancelled"):
        inst._go_rename(None)

    assert fns[0x401000].name == "sub_401000"
    assert fns[0x402000].name == "sub_402000"
