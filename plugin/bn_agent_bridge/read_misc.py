"""Strings, imports/needed-libraries, sections, and raw memory reads.

The strings/imports/sections/read cluster that used to live on
``BinaryNinjaBridge`` moves here as module-level free functions, each taking the
``BridgeContext`` seam (``ctx``) in place of ``self``. ``BinaryNinjaBridge``
keeps a thin delegating shim for every name the test suite / op binders
reference (``_strings``, ``_imports``, ``_imports_build_summary``,
``_needed_libraries``, ``_sections``, ``_read``, ``_ascii_render``,
``_is_executable_address``).

Outbound calls resolve through:
  * ``ctx`` -- resolution helpers relocated to the seam (``_resolve_view``);
  * ``read_xrefs`` -- ``_import_symbol_name`` (the preferred display name for an
    import symbol, used by the imports op);
  * ``bridge_state`` -- the ``_quick_loaded_views`` WeakSet (the strings op
    consults it to refuse with a directive when analysis hasn't run); imported
    as the SAME shared object so ``bv in _quick_loaded_views`` stays correct;
  * ``_shared`` -- module-free helpers (``_parse_address``, ``_validate_count``,
    ``OperationFailure``).

Import direction is one-way: this module imports ``read_xrefs``,
``bridge_state``, and ``_shared`` (plus stdlib + binaryninja). It NEVER imports
``bridge`` or ``seam`` -- those import THIS module one-way (design spec §3.2).
"""
from __future__ import annotations

import re
from typing import Any

import binaryninja as bn

from . import read_xrefs
from .bridge_state import _quick_loaded_views
from ._shared import OperationFailure, _parse_address, _validate_count

_STRING_TYPE_NAMES: dict[int, str] = {
    0: "ascii",
    1: "utf16",
    2: "utf32",
}

_NO_CRT_PATTERNS = re.compile(
    r"^(?:"
    r"[A-Za-z]$"                                      # single letters
    r"|[a-z]{2}(?:-[A-Z]{2})?$"                        # locale codes: en, en-US
    r"|[A-Z]{2,3}$"                                    # short uppercase tokens
    r"|(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)$"                # day abbreviations
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$"  # month abbreviations
    r"|(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)$"
    r"|(?:January|February|March|April|June|July|August|September|October|November|December)$"
    r"|(?:AM|PM|am|pm)$"
    r"|(?:UTF-?(?:7|8|16|32)|(?:us-)?ascii|iso-\d{4}.*|euc-\w+|big5|gb\d+|shift_jis|windows-\d+|cp\d+)$"
    r")",
    re.IGNORECASE,
)

_IMPORT_SYMBOL_TYPES: list[tuple[str, str]] = [
    ("ImportedFunctionSymbol", "function"),
    ("ImportedDataSymbol", "data"),
    ("ImportAddressSymbol", "address"),
]

# BN tags standard-ELF import symbols with these namespace sentinels rather
# than a real shared-object name (the dynamic linker only resolves the actual
# provider at runtime). Treat them as "no known library".
_BN_SENTINEL_NAMESPACES: frozenset[str] = frozenset(
    {"", "BNINTERNALNAMESPACE", "BNEXTERNALNAMESPACE"}
)

# Default `imports` ordering: surface the useful function/libc imports first,
# then data, then address-kind internals (often the bulk on real targets) -- so
# a `head`/first-page read isn't dominated by address symbols.
_IMPORT_KIND_ORDER: dict[str, int] = {"function": 0, "data": 1, "address": 2}

_SECTION_SEMANTICS_NAMES: dict[int, str] = {
    0: "DefaultSection",
    1: "ReadOnlyCode",
    2: "ReadOnlyData",
    3: "ReadWriteData",
    4: "ExternalSection",
}


def _paged_list_result(items: list[dict[str, Any]], *, offset: int,
                       limit: int | None, kind: str) -> dict[str, Any]:
    """Return a list page WITH paging metadata (the strings/imports/sections
    envelope).

    Mirrors ``read_listing._paged_function_result`` so the simple list ops
    expose the same honest paging contract -- the true total plus the remainder
    -- as ``function list``/``function search`` (#122). `items` is the universal
    data container and `kind` the envelope discriminator (#275); every caller
    passes its own `kind` (required, so a new list read can't omit it). The CLI
    can't compute the true total itself (it asks for a bounded page), so the
    bridge, which has the full filtered set, returns total/offset/limit/
    returned/has_more alongside the page."""
    total = len(items)
    page = items[offset:]
    if limit is not None:
        page = page[:limit]
    return {
        "kind": kind,
        "items": page,
        "total": total,
        "offset": offset,
        "limit": limit,
        "returned": len(page),
        "has_more": (offset + len(page)) < total,
    }


def _strings(ctx, selector: str | None, *, query, offset: int, limit: int | None,
             min_length: int | None = None, section: str | None = None,
             no_crt: bool = False, regex: bool = False, count_only: bool = False):
    offset = _validate_count(offset, label="offset", minimum=0)
    # allow_none mirrors the sibling list ops (imports/sections): limit=None
    # means "no limit", so a raw-socket / py exec caller can fetch every string.
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    bv = ctx._resolve_view(selector)
    if bv in _quick_loaded_views:
        # In --quick mode string analysis hasn't run, so bv.strings is empty
        # and `[]` would be indistinguishable from "this binary has none".
        # Refuse with a directive instead of misleading the caller.
        raise RuntimeError(
            "Strings are not available: this target was loaded with --quick (no analysis). "
            "Run `bn refresh` to build the full string set first."
        )
    items = []
    needle = str(query) if query else None
    pattern = None
    if needle and regex:
        try:
            pattern = re.compile(needle, re.IGNORECASE)
        except re.error as exc:
            raise OperationFailure("invalid_regex", f"Invalid string regex: {exc}") from exc
    elif needle:
        needle = needle.lower()
    for item in list(getattr(bv, "strings", [])):
        value = str(getattr(item, "value", ""))
        length = int(getattr(item, "length", 0))
        address = int(getattr(item, "start", 0))
        raw_type = getattr(item, "type", "")
        try:
            string_type = _STRING_TYPE_NAMES.get(int(raw_type), str(raw_type))
        except (TypeError, ValueError):
            string_type = str(raw_type)

        if pattern is not None:
            if not pattern.search(value):
                continue
        elif needle and needle not in value.lower():
            continue
        if min_length is not None and length < min_length:
            continue
        if section:
            secs = bv.get_sections_at(address) if hasattr(bv, "get_sections_at") else []
            if not any(getattr(s, "name", "") == section for s in secs):
                continue
        if no_crt:
            if _NO_CRT_PATTERNS.match(value):
                continue
            if len(value) >= 2 and len(set(value)) == 1:
                continue
            secs = bv.get_sections_at(address) if hasattr(bv, "get_sections_at") else []
            if any(getattr(s, "name", "") == ".text" for s in secs):
                continue

        entry = {
            "address": hex(address),
            "length": length,
            "chars": len(value),
            "type": string_type,
            "value": value,
        }
        items.append(entry)
    if count_only:
        # `total` mirrors the list envelope key for the same number (#165).
        return {"kind": "strings", "count": len(items), "total": len(items)}
    items.sort(key=lambda item: (int(item["address"], 16), item["value"]))
    return _paged_list_result(items, offset=offset, limit=limit, kind="strings")


def _needed_libraries(bv) -> list[str]:
    """DT_NEEDED shared objects this binary links against, if BN exposes them."""
    try:
        return sorted({str(lib) for lib in (getattr(bv, "libraries", None) or [])})
    except Exception:
        return []


def _defined_symbol_names(bv) -> set[str]:
    """Raw names of symbols DEFINED in this module (function/data definitions).

    Used to recognize a PIC shared object's own exports that BN also models as
    import veneers / GOT slots, so those self-references can be dropped from the
    imports survey (#202). Keyed on ``raw_name`` because that's what the import
    veneer and GOT slot carry too."""
    names: set[str] = set()
    for attr_name in ("FunctionSymbol", "DataSymbol"):
        sym_type = getattr(bn.SymbolType, attr_name, None)
        if sym_type is None:
            continue
        for sym in list(bv.get_symbols_of_type(sym_type)):
            names.add(str(getattr(sym, "raw_name", sym.name)))
    return names


# ExternalSymbol surfaces (a) statically-linked-but-undefined helpers and (b)
# the kernel-symbol refs of an ET_REL `.ko` (which BN models in .extern, not as
# Imported*Symbol) -- enumerating it makes a kernel module's API surface visible
# via `bn imports` (#213). Appended AFTER the Imported* kinds so a regular ELF's
# externals that also appear as PLT imports dedup against them.
_IMPORT_PLUS_EXTERNAL: list[tuple[str, str]] = _IMPORT_SYMBOL_TYPES + [("ExternalSymbol", "external")]


def _imports(ctx, selector: str | None, *, summary: bool = False,
             offset: int = 0, limit: int | None = None, count_only: bool = False,
             include_got: bool = False):
    # Guard paging the same way the sibling list ops do, so a raw-socket /
    # py exec caller passing a negative offset/limit gets a clean
    # invalid_request instead of a silent Python negative-index slice (#68).
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    bv = ctx._resolve_view(selector)
    needed_libraries = _needed_libraries(bv)
    defined_names = _defined_symbol_names(bv)
    items = []
    self_defined_excluded = 0
    got_collapsed = 0
    func_data_names: set[str] = set()
    emitted_names: set[str] = set()
    for attr_name, kind in _IMPORT_PLUS_EXTERNAL:
        sym_type = getattr(bn.SymbolType, attr_name, None)
        if sym_type is None:
            continue
        for sym in list(bv.get_symbols_of_type(sym_type)):
            name = read_xrefs._import_symbol_name(sym)
            raw_name = str(getattr(sym, "raw_name", sym.name))
            # On a PIC shared object BN models the library's OWN defined+exported
            # symbols as import veneers (ImportedFunctionSymbol) plus GOT/GLOB_DAT
            # slots (ImportAddressSymbol) -- internal self-references, not genuine
            # external dependencies. When the same name is defined in this module,
            # drop the import entry (counted below) so the survey isn't ~2x
            # bloated and real deps aren't buried (#202).
            if raw_name in defined_names:
                self_defined_excluded += 1
                continue
            # #212: an address-kind entry is the GOT slot for an import already
            # listed as its function/data PLT entry -- ~half of a standard ELF's
            # import list is these duplicates. Collapse them by default; show with
            # --include-got. A genuinely unique address symbol is still listed.
            if kind == "address" and not include_got and raw_name in func_data_names:
                got_collapsed += 1
                continue
            # An ET_REL external that ALSO appeared as an import on a regular ELF
            # is the same dependency. Dedup against EVERY already-emitted import
            # name -- not just function/data -- because on a standard ELF most
            # imports are GOT-only (ImportAddressSymbol, kind=address) with no PLT
            # function entry, and those same names reappear as ExternalSymbol;
            # checking only func_data_names would double-list them (#213 review).
            if kind == "external" and raw_name in emitted_names:
                continue
            namespace = str(getattr(sym, "namespace", "") or "")
            # Only surface `library` when it's a real per-library namespace;
            # BN's sentinels become None so agents don't read them as a
            # dependency. `namespace` keeps the raw value under an honest name.
            library = namespace if namespace not in _BN_SENTINEL_NAMESPACES else None
            items.append(
                {
                    "name": name,
                    "address": hex(sym.address),
                    "library": library,
                    "namespace": namespace,
                    "raw_name": raw_name,
                    "kind": kind,
                }
            )
            emitted_names.add(raw_name)
            if kind in ("function", "data"):
                func_data_names.add(raw_name)
    if count_only:
        result = {"kind": "imports", "count": len(items), "total": len(items)}
        if self_defined_excluded:
            result["self_defined_excluded"] = self_defined_excluded
        if got_collapsed:
            result["got_collapsed"] = got_collapsed
        return result
    if summary:
        # Summary aggregates the whole import set; paging would distort the
        # counts, so it always reflects every symbol regardless of offset/limit.
        result = _imports_build_summary(items, needed_libraries)
        if self_defined_excluded:
            result["self_defined_excluded"] = self_defined_excluded
        if got_collapsed:
            result["got_collapsed"] = got_collapsed
        return result
    # Order by kind USEFULNESS first (function -> data -> address), not the old
    # alphabetical kind sort which put "address"-kind internals ahead of
    # everything and buried the function/libc imports -- a `head`/first-page read
    # then concluded "0 function imports" on targets with many address symbols.
    # Library/name/address are tie-breakers within a kind.
    items.sort(key=lambda item: (
        _IMPORT_KIND_ORDER.get(item["kind"], 99),
        item["library"] or "",
        item["name"],
        int(item["address"], 16),
    ))
    result = _paged_list_result(items, offset=offset, limit=limit, kind="imports")
    # Only present when there's something to report, so the common case keeps the
    # standard paged-list envelope shared with strings/sections (#202).
    if self_defined_excluded:
        result["self_defined_excluded"] = self_defined_excluded
    if got_collapsed:
        result["got_collapsed"] = got_collapsed
    return result


def _exports(ctx, selector: str | None, *, offset: int = 0, limit: int | None = None,
             count_only: bool = False):
    """List a binary's EXPORTED symbols -- its public API. Exports are the
    globally/weakly-bound DEFINITIONS (the dynsym entries `nm -D --defined-only`
    shows); local/no-binding symbols are internal and excluded. Mirrors the
    `imports` envelope (name/address/kind + demangled display_name) (#198)."""
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    bv = ctx._resolve_view(selector)
    export_bindings = set()
    for bname in ("GlobalBinding", "WeakBinding"):
        b = getattr(getattr(bn, "SymbolBinding", None), bname, None)
        if b is not None:
            export_bindings.add(b)
    items = []
    for attr_name, kind in (("FunctionSymbol", "function"), ("DataSymbol", "data")):
        sym_type = getattr(bn.SymbolType, attr_name, None)
        if sym_type is None:
            continue
        for sym in list(bv.get_symbols_of_type(sym_type)):
            binding = getattr(sym, "binding", None)
            # Keep only global/weak definitions; LocalBinding/NoBinding are
            # internal. When binding info is unavailable (test fakes), keep the
            # symbol so callers still get the definition list.
            if export_bindings and binding is not None and binding not in export_bindings:
                continue
            short = getattr(sym, "short_name", None)
            items.append(
                {
                    "name": str(getattr(sym, "name", "")),
                    "display_name": str(short or getattr(sym, "name", "")),
                    "raw_name": str(getattr(sym, "raw_name", getattr(sym, "name", ""))),
                    "address": hex(int(sym.address)),
                    "kind": kind,
                    "binding": str(getattr(binding, "name", binding)) if binding is not None else None,
                    "ordinal": (int(getattr(sym, "ordinal", 0)) or None),
                }
            )
    if count_only:
        return {"kind": "exports", "count": len(items), "total": len(items)}
    items.sort(key=lambda item: (
        0 if item["kind"] == "function" else 1,
        item["name"],
        int(item["address"], 16),
    ))
    return _paged_list_result(items, offset=offset, limit=limit, kind="exports")


def _imports_build_summary(
    items: list[dict], needed_libraries: list[str] | None = None
) -> dict[str, Any]:
    # "namespaces" groups BN's symbol namespace (sentinels on standard ELF),
    # not a per-shared-object breakdown. The real dependency list is
    # "needed_libraries" (DT_NEEDED), which is what agents actually want.
    namespaces: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for item in items:
        ns = str(item.get("namespace", "") or "") or "(none)"
        namespaces[ns] = namespaces.get(ns, 0) + 1
        kind = str(item.get("kind", "unknown"))
        by_kind[kind] = by_kind.get(kind, 0) + 1
    return {
        # #275: a keyed aggregate, not a flat collection -- keeps its namespaces/
        # by_kind maps but carries a `kind` discriminator like every other read.
        "kind": "imports_summary",
        "total_symbols": len(items),
        "needed_libraries": needed_libraries or [],
        "namespaces": dict(sorted(namespaces.items(), key=lambda x: -x[1])),
        "by_kind": dict(sorted(by_kind.items(), key=lambda x: -x[1])),
    }


def _sections(ctx, selector: str | None, *, query: str | None = None,
              offset: int = 0, limit: int | None = None, count_only: bool = False):
    # Re-enforce the count contract for raw socket / py exec callers: a
    # negative offset/limit must be a clean invalid_request, not Python
    # negative-slice behavior returning a silently-wrong subset (#100).
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    bv = ctx._resolve_view(selector)
    items = []
    sections = getattr(bv, "sections", {})
    needle = str(query).lower() if query else None
    for name, sec in sections.items():
        start = int(getattr(sec, "start", 0))
        end = int(getattr(sec, "end", 0))
        length = end - start

        raw_semantics = getattr(sec, "semantics", 0)
        try:
            semantics_int = int(raw_semantics)
        except (TypeError, ValueError):
            semantics_int = 0
        semantics = _SECTION_SEMANTICS_NAMES.get(semantics_int, str(raw_semantics))

        # Match the query against the section name OR its semantics label, so
        # `--query code` finds executable sections (.text = ReadOnlyCode) even
        # though "code" is not in the name (#257).
        if needle and needle not in name.lower() and needle not in semantics.lower():
            continue

        entry: dict[str, Any] = {
            "name": name,
            "start": hex(start),
            "end": hex(end),
            "length": length,
            "semantics": semantics,
        }

        if hasattr(bv, "get_segment_at"):
            seg = bv.get_segment_at(start)
            if seg is not None:
                entry["readable"] = bool(getattr(seg, "readable", None))
                entry["writable"] = bool(getattr(seg, "writable", None))
                entry["executable"] = bool(getattr(seg, "executable", None))

        items.append(entry)
    if count_only:
        return {"kind": "sections", "count": len(items), "total": len(items)}
    items.sort(key=lambda item: int(item["start"], 16))
    return _paged_list_result(items, offset=offset, limit=limit, kind="sections")


def _ascii_render(data: bytes) -> str:
    return "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in data)


def _read(ctx, selector: str | None, address, length: int):
    bv = ctx._resolve_view(selector)
    addr = _parse_address(address)
    if length < 0:
        raise RuntimeError(f"read length must be non-negative, got {length}")

    data = bytes(bv.read(addr, length))
    if length > 0 and not data:
        raise RuntimeError(f"Address 0x{addr:x} is not mapped (no bytes available)")

    result: dict[str, Any] = {
        "address": hex(addr),
        "length": len(data),
        "hex": data.hex(),
        "ascii": _ascii_render(data),
    }
    if len(data) < length:
        result["requested_length"] = length
        result["short_read"] = True
        result["note"] = (
            f"short read: requested {length} bytes, only {len(data)} mapped from 0x{addr:x}"
        )
    return result


def _is_executable_address(ctx, bv, addr: int) -> bool:
    is_offset_executable = getattr(bv, "is_offset_executable", None)
    if callable(is_offset_executable):
        return bool(is_offset_executable(addr))
    get_segment_at = getattr(bv, "get_segment_at", None)
    if callable(get_segment_at):
        seg = get_segment_at(addr)
        if seg is not None:
            return bool(getattr(seg, "executable", False))
    return False
