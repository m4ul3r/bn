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


def _build_pclntab_many(count, *, text_start=0x400000, entry_step=0x1000):
    """A Go 1.20-format .gopclntab declaring *count* functions at distinct starts
    (`main.f0` @ text_start+entry_step = 0x401000, `main.f1` @ +2*entry_step, ...).

    Same layout as `_build_pclntab`, parameterized on the population so a test can
    drive a view where a RATIO decides the answer (#883 item 1) instead of the
    two-row fixture where any share is 0%, 50% or 100%.
    """
    funcname_off = 72
    names = bytearray()
    name_offsets: list[int] = []
    for i in range(count):
        name_offsets.append(len(names))
        names += f"main.f{i}\x00".encode()
    pcln_off = funcname_off + len(names)
    func_off = count * 8                      # the _func table follows the functab
    blob = bytearray(pcln_off + func_off + count * 8)
    struct.pack_into("<I", blob, 0, 0xFFFFFFF1)
    blob[6] = 1                               # minLC
    blob[7] = 8                               # ptrSize
    struct.pack_into("<Q", blob, 8, count)    # nfunc
    struct.pack_into("<Q", blob, 24, text_start)
    struct.pack_into("<Q", blob, 32, funcname_off)
    struct.pack_into("<Q", blob, 64, pcln_off)
    blob[funcname_off:funcname_off + len(names)] = names
    for i in range(count):
        struct.pack_into("<I", blob, pcln_off + i * 8 + 4, func_off + i * 8)
        struct.pack_into("<I", blob, pcln_off + func_off + i * 8,
                         entry_step * (i + 1))     # _func entryoff
        struct.pack_into("<i", blob, pcln_off + func_off + i * 8 + 4,
                         name_offsets[i])          # nameoff
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


def test_go_functions_defined_via_containment_for_an_interior_pc_818(monkeypatch):
    # #818: `defined` was read off `get_function_at` alone, which is START-only.
    # A pcln entry whose prolog is a few bytes off -- or whose `_func` entryoff is
    # an interior PC -- therefore read `defined: false`, and on a table where EVERY
    # row missed that way the 0-match NOTE fired (PIE rebase / incomplete analysis)
    # on a view that had in fact resolved each address to a function. Containment
    # is the relation the row is asking about; the fallback is the same
    # first-containing-function lookup the sibling reads use.
    class _InteriorGoBV(_GoBV):
        """BN recovers both pcln addresses as interior PCs of a function, so the
        START-only accessor misses them and only containment finds them."""

        _CONTAINERS = {0x401000: "sub_401000", 0x402000: "sub_402000"}

        def get_function_at(self, addr):
            return None

        def get_functions_containing(self, addr):
            name = self._CONTAINERS.get(int(addr))
            return [type("F", (), {"name": name})()] if name else []

    bridge, inst = _ctx(monkeypatch, _InteriorGoBV(_build_pclntab()))
    out = inst._go_functions(None)
    by_name = {i["name"]: i for i in out["items"]}

    assert by_name["main.foo"]["defined"] is True
    assert by_name["main.bar"]["defined"] is True
    assert out["defined_count"] == 2
    # #818 review: `defined` is satisfied by containment, so the note can no longer
    # be gated on it. Gated there, a table whose every row lands on an interior PC
    # (a constant rebase delta over a dense .text does exactly this) reported
    # `defined: true` everywhere with the rebase warning suppressed. It is gated on
    # START matches now, and its wording says which relation matched -- so a reader
    # is told the rows are off-prolog, not sent to rebase good addresses blindly.
    assert out["start_match_count"] == 0
    assert "note" in out
    assert "START" in out["note"] and "interior PC" in out["note"]


def test_go_functions_count_only_skips_the_list(monkeypatch):
    # #414: --count returns the recovered count without the full items list.
    bv = _GoBV(_build_pclntab(), defined={0x401000})
    bridge, inst = _ctx(monkeypatch, bv)
    out = inst._go_functions(None, count_only=True)
    assert out["kind"] == "go_functions"
    assert out["count"] == 2 and out["total"] == 2
    assert "items" not in out


def test_go_functions_summary_counts_recovered_defined_renamable(monkeypatch):
    # #414: summary gives recovered/defined/undefined + renamable (what `go rename`
    # would touch: defined fns still carrying an auto sub_<hex> name).
    class _NamedGoBV(_GoBV):
        def get_function_at(self, addr):
            if addr not in self._defined:
                return None
            # main.foo @0x401000 still auto-named sub_401000 -> renamable;
            return type("F", (), {"name": f"sub_{addr:x}"})()
    bv = _NamedGoBV(_build_pclntab(), defined={0x401000})
    bridge, inst = _ctx(monkeypatch, bv)
    out = inst._go_functions(None, summary=True)
    assert out["kind"] == "go_functions_summary"
    assert out["recovered"] == 2 and out["defined"] == 1 and out["undefined"] == 1
    assert out["renamable"] == 1
    assert "items" not in out


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


def test_go_functions_partial_walk_is_disclosed(monkeypatch):
    # #528: a functab that declares more functions than are recoverable (an entry
    # whose funcInfo runs off the section) must report recovered < expected with the
    # truncation signal set -- not len(items) masquerading as a complete count.
    blob = bytearray(_build_pclntab())            # nfunc=2, both recoverable
    pcln_off = 96
    struct.pack_into("<Q", blob, 8, 3)            # bump declared nfunc to 3
    # entry2's funcInfo offset points past the end of the section -> skipped.
    struct.pack_into("<I", blob, pcln_off + 20, 250)
    bv = _GoBV(bytes(blob), defined={0x401000})
    bridge, inst = _ctx(monkeypatch, bv)

    out = inst._go_functions(None)
    assert out["expected"] == 3
    assert out["recovered"] == 2
    assert out["skipped"] == 1
    assert out["truncated"] is True
    # the recovered items themselves are unchanged
    assert {i["name"] for i in out["items"]} == {"main.foo", "main.bar"}

    # count_only and summary paths carry the same honest disclosure
    co = inst._go_functions(None, count_only=True)
    assert co["count"] == 2 and co["expected"] == 3 and co["skipped"] == 1 and co["truncated"] is True
    sm = inst._go_functions(None, summary=True)
    assert sm["recovered"] == 2 and sm["expected"] == 3 and sm["skipped"] == 1 and sm["truncated"] is True


def test_go_functions_complete_walk_not_truncated(monkeypatch):
    # A fully-recovered table reports recovered == expected and truncated False.
    bv = _GoBV(_build_pclntab(), defined={0x401000})
    bridge, inst = _ctx(monkeypatch, bv)
    out = inst._go_functions(None)
    assert out["expected"] == 2 and out["recovered"] == 2
    assert out["skipped"] == 0 and out["truncated"] is False


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
class _ContainmentOnlyGoBV(_GoBV):
    """A view where the pcln addresses are INTERIOR PCs of an already-recovered
    body, so `get_function_at` (START-only) misses every one of them and only
    `get_functions_containing` answers (#818's relation, which `go rename` still
    re-resolved with the START-only accessor).

    *starts* names the addresses BN does have a function START at; the rest are
    answered by containment alone, under the containing body's own name.
    """

    _CONTAINER_NAME = "sub_400000"

    def __init__(self, blob, *, starts=(), **kw):
        super().__init__(blob, **kw)
        self._starts = dict(starts)

    def get_function_at(self, addr):
        name = self._starts.get(int(addr))
        return type("F", (), {"name": name})() if name else None

    def get_functions_containing(self, addr):
        return [type("F", (), {"name": self._CONTAINER_NAME})()]


def test_go_rename_accounts_for_containment_only_rows_818(monkeypatch):
    """#818 review: the two views must state ONE population.

    `go functions` counts `defined` by CONTAINMENT, so on a view whose pcln
    addresses resolve only as interior PCs it reports `defined 1848` -- while
    `go rename` re-resolved every candidate with the START-only accessor, matched
    nothing, and answered `defined_count: 1848` beside `go_renamed_candidates: 0`
    with `success: true` and `results: []`. Rows #818 promoted to `defined: true`
    landed in no bucket at all: not a candidate, not `skipped_user_named`, not a
    failure row, so nothing in the envelope reconciled the two totals.

    The row is DISCLOSED as unrenamable rather than renamed: the recovered name
    belongs to the function that starts at the pcln entryoff, and the containing
    body starts elsewhere -- applying it there would mislabel it.
    """
    blob = _build_pclntab()          # main.foo @0x401000, main.bar @0x402000
    bridge, inst = _ctx(monkeypatch, _ContainmentOnlyGoBV(blob))
    monkeypatch.setattr(inst, "_mutation",
                        lambda *a, **k: pytest.fail("go rename must not use generic mutation"))

    listed = inst._go_functions(None, summary=True)
    rename = inst._go_rename(None, preview=True)

    # Both views agree that two rows are defined...
    assert listed["defined"] == 2 and listed["start_match_count"] == 0
    assert rename["defined_count"] == 2
    # ...and the rename side accounts for both of them instead of dropping them.
    assert rename["go_renamed_candidates"] == 0 and rename["results"] == []
    assert rename["skipped_interior_pc"] == 2
    assert (rename["go_renamed_candidates"] + rename["skipped_user_named"]
            + rename["skipped_interior_pc"]) == rename["defined_count"]

    # The chunked/apply path carries the same accounting: one auto-named START
    # candidate beside one containment-only row.
    mixed, get_function_at = _fake_functions({0x401000: "sub_401000"})
    mixed_bv = _ContainmentOnlyGoBV(blob, starts={0x401000: "sub_401000"})
    monkeypatch.setattr(mixed_bv, "get_function_at", get_function_at)
    _bridge, mixed_inst = _ctx(monkeypatch, mixed_bv)

    applied = mixed_inst._go_rename(None, preview=True)

    assert applied["go_renamed_candidates"] == 1
    assert applied["go_verified_count"] == 1 and applied["skipped_interior_pc"] == 1
    assert applied["defined_count"] == 2
    assert mixed[0x401000].name == "sub_401000"          # preview reverted it


def test_go_functions_summary_carries_the_note_and_the_start_matches_818(monkeypatch):
    """#818 review: the go/no-go view is the one that must not lose the warning.

    The note was attached after the summary branch returned, so `--summary` could
    never carry it, and the counter it is gated on (`start_match_count`) was
    JSON-only: the text renderer did not print it at all. Pre-#818 that view said
    `defined 0 / undefined 1848` (loud and wrong); once `defined` could be
    satisfied by containment the same view says `defined 1848 / undefined 0` --
    the headline a caller decides `go rename` on -- with nothing saying that no
    row matches a function START.
    """
    from bn.formatters import _render_go_functions_summary_text

    blob = _build_pclntab()
    bridge, inst = _ctx(monkeypatch, _ContainmentOnlyGoBV(blob))

    summary = inst._go_functions(None, summary=True)

    assert summary["defined"] == 2 and summary["undefined"] == 0
    assert summary["start_match_count"] == 0
    assert "note" in summary and "interior PC" in summary["note"]

    text = _render_go_functions_summary_text(summary)
    assert "start_matches: 0" in text
    assert "interior PC" in text


def test_go_functions_rebase_note_survives_a_single_start_match_883(monkeypatch):
    """#883 item 1: a binary gate on `start_match_count` is not a gate on the
    question the note asks.

    Forced with exactly ONE matching address on a view whose other rows resolve
    only as interior PCs: `defined 10`, `start_match_count 1`, and before this the
    note was suppressed -- so the summary read clean while 9 of the 10 resolved
    addresses were off-prolog, which is the shape a constant rebase delta
    produces. A START match is evidence about THAT row, not about the table.

    The bucket partition stays self-consistent while the gate changes: the note
    states `defined - start_match` interior rows and invents no counter, so
    `start_match + interior == defined` still reconciles both views.
    """
    from bn.formatters import _render_go_functions_summary_text, _render_go_functions_text

    blob = _build_pclntab_many(10)               # starts at 0x401000, 0x402000, ...
    bridge, inst = _ctx(monkeypatch, _ContainmentOnlyGoBV(blob, starts={0x401000: "sub_401000"}))

    listed = inst._go_functions(None)

    assert listed["defined_count"] == 10 and listed["start_match_count"] == 1
    assert listed["defined_count"] - listed["start_match_count"] == 9   # the interior share
    assert "note" in listed, "9 of 10 resolved rows being interior PCs must not read clean"
    assert "9 of the 10" in listed["note"] and "1 matched a START" in listed["note"]
    assert "interior PC" in listed["note"]
    # The text face carries it, and carries it on a SLICE: the note lives on the
    # envelope, not on a row, so `--offset`/`--limit` pages cannot lose it.
    sliced = inst._go_functions(None, offset=2, limit=3)
    assert "note" in sliced and "9 of the 10" in sliced["note"]
    assert "interior PC" in _render_go_functions_text(sliced)

    summary = inst._go_functions(None, summary=True)
    assert summary["defined"] == 10 and summary["start_match_count"] == 1
    assert "note" in summary and "interior PC" in summary["note"]
    text = _render_go_functions_summary_text(summary)
    assert "start_matches: 1" in text and "interior PC" in text


def test_go_functions_rebase_note_ratio_boundary_883(monkeypatch):
    """The other half of the ratio: it must not turn a well-based table into an
    alarm. 2 interior rows in 10 (20%) stays quiet -- the counters still state the
    split -- while exactly half is where the note starts firing."""
    from bn.formatters import _render_go_functions_text

    blob = _build_pclntab_many(10)
    every_start = {0x400000 + 0x1000 * (i + 1): f"sub_{0x400000 + 0x1000 * (i + 1):x}"
                   for i in range(10)}

    bridge, inst = _ctx(monkeypatch, _ContainmentOnlyGoBV(blob, starts=every_start))
    quiet = inst._go_functions(None)
    assert quiet["defined_count"] == 10 and quiet["start_match_count"] == 10
    assert "note" not in quiet

    # Eight of ten matching at their start: two interior rows, below the gate.
    bridge, inst = _ctx(monkeypatch, _ContainmentOnlyGoBV(
        blob, starts={a: n for a, n in every_start.items() if a < 0x409000}))
    below = inst._go_functions(None)
    assert below["defined_count"] == 10 and below["start_match_count"] == 8
    assert "note" not in below
    assert "interior PC" not in _render_go_functions_text(below)

    # Exactly half -- the boundary the constant names.
    bridge, inst = _ctx(monkeypatch, _ContainmentOnlyGoBV(
        blob, starts={a: n for a, n in every_start.items() if a < 0x406000}))
    half = inst._go_functions(None)
    assert half["defined_count"] == 10 and half["start_match_count"] == 5
    assert "note" in half and "5 of the 10" in half["note"]
