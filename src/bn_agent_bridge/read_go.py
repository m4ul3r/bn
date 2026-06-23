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

import binaryninja as bn  # noqa: F401  (parity with sibling read_* modules)

from ._shared import OperationFailure, _validate_count
from .read_misc import _paged_list_result

# Go pcHeader magics. 1.20 and 1.18 share the same field layout this parser reads
# (uint32 functab entries, the funcInfo entryoff/nameoff prefix); 1.16 (0xFA) and
# 1.2 (0xFB) use older layouts we decline rather than mis-parse.
_PCLNTAB_MAGICS = {0xFFFFFFF1: "go1.20", 0xFFFFFFF0: "go1.18"}
_OLD_MAGICS = {0xFFFFFFFA: "go1.16", 0xFFFFFFFB: "go1.2"}


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


def _go_functions(ctx, selector: str | None, *, offset: int = 0, limit: int | None = None):
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

    get_fn = getattr(bv, "get_function_at", None)

    def cstr(o: int) -> str:
        end = raw.find(b"\x00", o)
        if end < 0:
            end = len(raw)
        return raw[o:end].decode("utf-8", "replace")

    items: list[dict[str, Any]] = []
    defined_count = 0
    for i in range(nfunc):
        ent = pcln_off + i * 8
        if ent + 8 > length:
            break
        func_off = u32(ent + 4)
        fo = pcln_off + func_off
        if fo + 8 > length:
            continue
        entryoff = u32(fo)
        nameoff = i32(fo + 4)
        npos = funcname_off + nameoff
        if npos < 0 or npos >= length:
            continue
        name = cstr(npos)
        if not name:
            continue
        addr = text_start + entryoff
        defined = bool(get_fn(addr)) if callable(get_fn) else None
        if defined:
            defined_count += 1
        items.append({"name": name, "address": hex(addr), "defined": defined})

    items.sort(key=lambda it: int(it["address"], 16))
    result = _paged_list_result(items, offset=offset, limit=limit, kind="go_functions")
    result["go_version"] = go_version
    result["text_start"] = hex(text_start)
    # Disclose a likely PIE/rebase mismatch so the addresses aren't trusted blindly:
    # when almost nothing resolves to a BN function, the table's textStart differs
    # from where BN loaded the text (rebase by bv_text - text_start).
    text_sec = _gopclntab_text_start(bv)
    if text_sec is not None:
        result["text_start_bv"] = hex(text_sec)
    result["defined_count"] = defined_count
    if items and defined_count == 0:
        if text_sec is not None and text_sec != text_start:
            result["note"] = (
                "0 of the recovered addresses match a BN function and the pcln "
                "table's textStart != BN's .text start: the binary is loaded at a "
                "different base (PIE). Rebase each address by "
                "(text_start_bv - text_start) before use."
            )
        else:
            result["note"] = (
                "0 of the recovered addresses match a BN function: BN analysis may "
                "be incomplete (run `bn refresh`), or the binary is rebased -- "
                "compare text_start vs text_start_bv before trusting the addresses."
            )
    return result


def _gopclntab_text_start(bv) -> int | None:
    getter = getattr(bv, "get_section_by_name", None)
    if callable(getter):
        sec = getter(".text")
        if sec is not None:
            return int(getattr(sec, "start", 0))
    return None
