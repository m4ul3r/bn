"""Go metadata lens: recover function names from ``.gopclntab`` (#217).

A Go-compiled binary ships a ``.gopclntab`` (the pc->line table) whose function
table maps every Go function's PC to its full name (``pkg.Func`` /
``(*T).Method``). BN's default analysis does not consume it, so a Go target is a
wall of ``sub_*`` auto-names. This module parses the modern (Go 1.18 / 1.20+)
``pcHeader`` + functab + funcname table and returns ``{name, address}`` for every
function the table describes -- a read-only lens, no mutation. Older formats
(Go 1.2 / 1.16) and 32-bit ``ptrSize`` are declined with an honest error rather
than mis-parsed.

Free functions taking the ``BridgeContext`` seam (``ctx``), mirroring the other
``read_*`` modules; ``BinaryNinjaBridge`` keeps a thin ``_go_functions`` shim.
Import direction is one-way (imports ``read_misc`` for the shared #275 paging
envelope and ``_shared``; never ``bridge``/``seam``).
"""
from __future__ import annotations

import struct
from typing import Any

try:
    import binaryninja as bn  # noqa: F401  (parity with sibling read_* modules)
except ModuleNotFoundError:  # importable without the Binary Ninja runtime (tests, tooling)
    bn = None  # type: ignore[assignment]

from ._shared import OperationFailure, _validate_count
from .read_misc import _paged_list_result

# Go pcHeader magics. 1.20 and 1.18 share the same field layout this parser reads
# (uint32 functab entries, the funcInfo entryoff/nameoff prefix); 1.16 (0xFA) and
# 1.2 (0xFB) use older layouts we decline rather than mis-parse.
_PCLNTAB_MAGICS = {0xFFFFFFF1: "go1.20", 0xFFFFFFF0: "go1.18"}
_OLD_MAGICS = {0xFFFFFFFA: "go1.16", 0xFFFFFFFB: "go1.2"}


def resolve_pcln_function(bv, addr: int):
    """The BN Function for a pcln address *addr*, plus whether it matched at its START.

    #818: `get_function_at` is START-only, so a pcln entry whose prolog is a few
    bytes off -- or a `_func` entryoff that is an interior PC -- missed and the
    row read `defined: false`, and on a table where EVERY row missed that way the
    0-match note fired (PIE rebase / incomplete analysis) on a view that had
    resolved all of them. Containment is the relation the row is really asking
    about, so fall back to `get_functions_containing` -- the same
    first-containing-function lookup the sibling reads use for an interior
    address. `defined` stays None only when the view exposes NEITHER accessor
    (a unit fake): "unknown", never a confident false.

    The START/containment distinction is load-bearing for two different readers,
    which is why it is returned rather than folded into the record:

    * the rebase note (#818 review) -- an address that resolves only as an
      INTERIOR PC is not evidence that the table is based correctly, and a
      constant rebase delta over a dense .text resolves every row to *some* body
      while matching no start;
    * `go rename` (bridge `_go_rename`) -- the recovered name belongs to the
      function that STARTS at the entryoff, so a containing function is never
      renamed and the row is disclosed as an unrenamable bucket instead.
    """
    get_fn = getattr(bv, "get_function_at", None)
    if callable(get_fn):
        fn = get_fn(addr)
        if fn is not None:
            return fn, True
    get_containing = getattr(bv, "get_functions_containing", None)
    if callable(get_containing):
        try:
            containers = list(get_containing(addr) or [])
        except Exception:
            containers = []
        if containers:
            return containers[0], False
    return None, False


def _rebase_note(items: list[Any], *, start_match_count: int, defined_count: int,
                 text_start: int, text_sec: int | None) -> str | None:
    """The #217/#818 rebase-or-incomplete-analysis note, or None when none applies.

    ONE builder for the two views that publish it: `go functions` and its
    `--summary` form. It is keyed on START matches, never on `defined`: once
    `defined` can be satisfied by containment, a table whose every row lands on an
    interior PC of *some* body would report `defined: true` everywhere and
    suppress the warning this note exists to give. The wording says which
    relation matched, so a reader is not sent to rebase addresses that are merely
    off-prolog.
    """
    if not items or start_match_count:
        return None
    if text_sec is not None and text_sec != text_start:
        return (
            "None of the recovered addresses match a BN function START, and the "
            "pcln table's textStart != BN's .text start: the binary is loaded at a "
            "different base (PIE). Every address resolves at best to an interior PC, "
            "so rebase each by (text_start_bv - text_start) before use."
        )
    if defined_count == 0:
        return (
            "0 of the recovered addresses match a BN function: BN analysis may "
            "be incomplete (run `bn refresh`), or the binary is rebased -- "
            "compare text_start vs text_start_bv before trusting the addresses."
        )
    return (
        "None of the recovered addresses match a BN function START -- they "
        "resolve only as interior PCs. BN analysis may be incomplete (run "
        "`bn refresh`), or the table is rebased by a constant delta: compare "
        "text_start vs text_start_bv before trusting the addresses."
    )


def _gopclntab_section(bv):
    """The ``.gopclntab`` section object, else None (by name, then by any section
    whose name ends in ``gopclntab`` for the rare renamed/embedded case)."""
    getter = getattr(bv, "get_section_by_name", None)
    if callable(getter):
        sec = getter(".gopclntab")
        if sec is not None:
            return sec
    for name, sec in (getattr(bv, "sections", {}) or {}).items():
        if str(name).endswith("gopclntab"):
            return sec
    return None


def _go_functions(ctx, selector: str | None, *, offset: int = 0, limit: int | None = None,
                  count_only: bool = False, summary: bool = False):
    """Parse ``.gopclntab`` and return ``{name, address, defined}`` per Go
    function (#217). ``defined`` flags whether BN already has a function at the
    pclntab-derived address; when it is mostly false the binary is loaded at a
    different base than the table's ``textStart`` (PIE) and the addresses need
    rebasing, which the result discloses via ``text_start`` / ``text_start_bv``."""
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    bv = ctx._resolve_view(selector)

    sec = _gopclntab_section(bv)
    if sec is None:
        raise OperationFailure(
            "no_gopclntab",
            "No .gopclntab section: this target does not look like a Go binary "
            "(or its pcln table was stripped/renamed).",
        )
    base = int(getattr(sec, "start", 0))
    length = int(getattr(sec, "length", 0) or (int(getattr(sec, "end", 0)) - base))
    if length <= 0:
        raise OperationFailure("empty_gopclntab", "The .gopclntab section is empty.")

    raw = bytes(bv.read(base, length) or b"")
    # The 64-bit pcHeader spans bytes 0..71 (the last field read, pclnOffset, is at
    # @64); require the whole header so the uptr() reads below can't throw an
    # opaque struct.error on a short/corrupt/mis-identified section (#217 review).
    if len(raw) < 72:
        raise OperationFailure("short_gopclntab", "The .gopclntab section is too short to hold a pcln header.")

    order = "<" if str(ctx._byteorder(bv)) == "little" else ">"
    magic = struct.unpack_from(order + "I", raw, 0)[0]
    if magic not in _PCLNTAB_MAGICS:
        if magic in _OLD_MAGICS:
            raise OperationFailure(
                "unsupported_pclntab_version",
                f"This .gopclntab is the older {_OLD_MAGICS[magic]} format "
                f"(magic {hex(magic)}); only Go 1.18/1.20+ (the modern layout) is "
                f"parsed. File an issue with the target's Go version if you need it.",
            )
        raise OperationFailure(
            "unrecognized_pclntab",
            f"Unrecognized .gopclntab magic {hex(magic)} -- not a Go pcln table this "
            f"lens understands (Go 1.18/1.20+).",
        )
    go_version = _PCLNTAB_MAGICS[magic]
    ptr_size = raw[7]
    if ptr_size != 8:
        raise OperationFailure(
            "unsupported_ptr_size",
            f"This lens currently parses 64-bit Go pcln tables only (ptrSize={ptr_size}); "
            f"32-bit Go targets aren't supported yet.",
        )

    def uptr(o: int) -> int:
        return struct.unpack_from(order + "Q", raw, o)[0]

    def u32(o: int) -> int:
        return struct.unpack_from(order + "I", raw, o)[0]

    def i32(o: int) -> int:
        return struct.unpack_from(order + "i", raw, o)[0]

    # pcHeader (ptrSize==8): nfunc@8, textStart@24, funcnameOffset@32, pclnOffset@64.
    nfunc = uptr(8)
    text_start = uptr(24)
    funcname_off = uptr(32)
    pcln_off = uptr(64)
    # Sanity-bound the table offsets against the section so a malformed/misread
    # header can't drive an out-of-range walk.
    if not (0 < pcln_off < length and 0 <= funcname_off < length) or nfunc <= 0 or nfunc > (length // 8):
        raise OperationFailure(
            "malformed_pclntab",
            f"The .gopclntab header is inconsistent (nfunc={nfunc}, "
            f"functab@{hex(pcln_off)}, funcname@{hex(funcname_off)}) -- refusing to "
            f"walk it rather than emit garbage.",
        )

    # The resolution relation (start vs containment) is shared with `go rename`,
    # which must know which of the two matched before it may apply a name; see
    # `resolve_pcln_function`.
    can_resolve = (
        callable(getattr(bv, "get_function_at", None))
        or callable(getattr(bv, "get_functions_containing", None))
    )

    def resolve_function(addr: int):
        return resolve_pcln_function(bv, addr)

    def cstr(o: int) -> str:
        end = raw.find(b"\x00", o)
        if end < 0:
            end = len(raw)
        return raw[o:end].decode("utf-8", "replace")

    items: list[dict[str, Any]] = []
    defined_count = 0
    start_match_count = 0  # #818 review: rows matching a function START, vs only by containment
    renamable_count = 0  # #414: defined fns whose current BN name `go rename` would replace
    skipped_count = 0  # #528: functab entries that ran off the section or held no name
    for i in range(nfunc):
        ent = pcln_off + i * 8
        if ent + 8 > length:
            # The functab itself runs off the end of the section: every remaining
            # declared entry is unrecoverable, so count them as skipped (#528).
            skipped_count += nfunc - i
            break
        func_off = u32(ent + 4)
        fo = pcln_off + func_off
        if fo + 8 > length:
            skipped_count += 1
            continue
        entryoff = u32(fo)
        nameoff = i32(fo + 4)
        npos = funcname_off + nameoff
        if npos < 0 or npos >= length:
            skipped_count += 1
            continue
        name = cstr(npos)
        if not name:
            skipped_count += 1
            continue
        addr = text_start + entryoff
        fn_obj, start_matched = resolve_function(addr) if can_resolve else (None, False)
        defined = bool(fn_obj) if can_resolve else None
        if start_matched:
            start_match_count += 1
        if defined:
            defined_count += 1
            cur = str(getattr(fn_obj, "name", "") or "")
            # mirror go_rename's auto-name predicate (sub_<hex> / nullsub_*): only
            # those get renamed, and only when the Go name actually differs.
            if (cur == f"sub_{addr:x}" or cur.startswith("nullsub_")) and name != cur:
                renamable_count += 1
        items.append({"name": name, "address": hex(addr), "defined": defined})

    # #528: a partial walk (entries that ran off the section or held no name) must
    # be disclosed -- otherwise len(items) reads as a complete count. `expected` is
    # the header's declared nfunc, `recovered` the entries we actually parsed, and
    # `skipped`/`truncated` flag the shortfall.
    recovered = len(items)
    truncated = recovered < nfunc
    if count_only:
        # #414: cheap sizing primitive -- recovered count without the full list.
        return {"kind": "go_functions", "count": recovered, "total": recovered,
                "expected": nfunc, "recovered": recovered, "skipped": skipped_count,
                "truncated": truncated, "go_version": go_version}
    text_sec = _gopclntab_text_start(bv)
    # #818 review: the note is built BEFORE the summary branch returns, so the
    # go/no-go view carries it too. It used to be attached only to the listing
    # view, and `--summary` is the view an agent reads to DECIDE whether to run
    # `go rename` -- the exact decision the note exists to inform.
    note = _rebase_note(
        items,
        start_match_count=start_match_count,
        defined_count=defined_count,
        text_start=text_start,
        text_sec=text_sec,
    )
    if summary:
        # #414: enough signal to decide whether to run `go rename`.
        result = {"kind": "go_functions_summary", "go_version": go_version,
                  "recovered": recovered, "defined": defined_count,
                  "start_match_count": start_match_count,
                  "undefined": recovered - defined_count, "renamable": renamable_count,
                  "expected": nfunc, "skipped": skipped_count, "truncated": truncated,
                  "text_start": hex(text_start),
                  "text_start_bv": hex(text_sec) if text_sec is not None else None,
                  "pclntab": True}
        if note:
            result["note"] = note
        return result

    items.sort(key=lambda it: int(it["address"], 16))
    result = _paged_list_result(items, offset=offset, limit=limit, kind="go_functions")
    result["go_version"] = go_version
    result["expected"] = nfunc
    result["recovered"] = recovered
    result["skipped"] = skipped_count
    result["truncated"] = truncated
    result["text_start"] = hex(text_start)
    # Disclose a likely PIE/rebase mismatch so the addresses aren't trusted blindly:
    # when almost nothing resolves to a BN function, the table's textStart differs
    # from where BN loaded the text (rebase by bv_text - text_start). (text_sec is
    # computed once above.)
    if text_sec is not None:
        result["text_start_bv"] = hex(text_sec)
    result["defined_count"] = defined_count
    # #818 review: the note is gated on START matches, not on `defined` -- see
    # `_rebase_note`, which the `--summary` view shares.
    result["start_match_count"] = start_match_count
    if note:
        result["note"] = note
    return result


def _gopclntab_text_start(bv) -> int | None:
    getter = getattr(bv, "get_section_by_name", None)
    if callable(getter):
        sec = getter(".text")
        if sec is not None:
            return int(getattr(sec, "start", 0))
    return None
