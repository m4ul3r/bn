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
  * ``bridge_state`` -- ``analysis_unavailable``, which reads the shared
    ``_quick_loaded_views`` / ``_refreshing_views`` state (the strings op
    refuses with a directive when analysis hasn't run, or says "retry" while a
    ``bn refresh`` is running, #996);
  * ``_shared`` -- module-free helpers (``_parse_address``, ``_validate_count``,
    ``OperationFailure``).

Import direction is one-way: this module imports ``read_xrefs``,
``bridge_state``, and ``_shared`` (plus stdlib + binaryninja). It NEVER imports
``bridge`` or ``seam`` -- those import THIS module one-way (design spec §3.2).
"""
from __future__ import annotations

import os
import re
from bisect import bisect_right
from typing import Any

try:
    import binaryninja as bn
except ModuleNotFoundError:  # importable without the Binary Ninja runtime (tests, tooling)
    bn = None  # type: ignore[assignment]

from . import il_format
from . import read_xrefs
from .bridge_state import analysis_unavailable
from ._shared import OperationFailure, _parse_address, _validate_count
from .elf_dynamic import (
    ElfDynamicInfo,
    NeededExports,
    library_search_dirs,
    needed_export_libraries,
    parse_elf_dynamic,
)

_STRING_TYPE_NAMES: dict[int, str] = {
    0: "ascii",
    1: "utf16",
    2: "utf32",
}

# C printf directive grammar: %[flags][width][.precision][length]conversion, plus
# the escaped literal %%. The SPACE flag is deliberately excluded: "% d" is legal
# but vanishingly rare in real code, and admitting it turns ordinary prose like
# "100% done" or "50% off" into false format-string hits. Dropping it forfeits the
# uncommon space-flag directive but removes the dominant natural-language false
# positive.
_C_FORMAT_DIRECTIVE = re.compile(
    r"%(?:"
    r"%"                                                     # literal %%
    r"|[-+#0]*(?:\d+|\*)?(?:\.(?:\d+|\*))?(?:hh|ll|[hlLjztq])?"
    r"[diouxXeEfFgGaAcspn]"                                  # conversion
    r")"
)


def _probable_format_directives(value: str) -> list[str] | None:
    """Return the real (argument-consuming) printf conversion directives in
    *value* when it plausibly IS a C format string, else ``None``.

    A string qualifies only when EVERY ``%`` is either an escaped ``%%`` or a
    valid directive (no stray/malformed percents -- the signature of accidental
    percent-substrings in resource/font/blob data) AND at least one of those
    directives is a real conversion (a bare ``%%`` string consumes no argument
    and is uninteresting for a format-token survey).

    This is a labeling heuristic, NOT a vulnerability verdict: it reports "this
    looks like a C format string" and which directives it carries -- never "this
    is a format-string bug". A ``%n`` in the list is provenance for the model to
    weigh, not a finding the engine asserts."""
    directives: list[str] = []
    i = 0
    n = len(value)
    while i < n:
        if value[i] != "%":
            i += 1
            continue
        m = _C_FORMAT_DIRECTIVE.match(value, i)
        if m is None:
            # A '%' that does not begin a valid directive -> not a plausible C
            # format string. This is exactly the raw-regex noise the mode exists
            # to reject, so drop the whole string.
            return None
        token = m.group(0)
        if token != "%%":
            directives.append(token)
        i = m.end()
    return directives or None


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

# #827 item 3: read from the one owner rather than keeping a verbatim twin.
# Re-bound at module scope so every existing reference in this file (and any
# `read_misc._IMPORT_SYMBOL_TYPES` reader) keeps working unchanged; the list is
# defined once, in the module this one already imports.
_IMPORT_SYMBOL_TYPES: list[tuple[str, str]] = read_xrefs._IMPORT_SYMBOL_TYPES

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


def _page_window(total: int, *, offset: int, limit: int | None) -> tuple[int, int]:
    """The ``[start, stop)`` window of a filtered population that this request
    must materialize ROWS for (#814).

    The paging contract reports the exact size of the filtered population plus
    the remainder, so a read still has to *scan* the population (and know its
    order) before it can answer -- but it never has to build a row, project a
    display name, or count xrefs for anything outside this window. `stop` clamps
    to `total`, so an offset past the end yields an empty window (`has_more`
    False) exactly as the slicing envelope did."""
    start = min(offset, total)
    stop = total if limit is None else min(start + limit, total)
    return start, stop


def _paged_envelope(*, kind: str, items: list[dict[str, Any]], total: int,
                    offset: int, limit: int | None) -> dict[str, Any]:
    """The ONE paging envelope every list read returns: ``items`` is the returned
    PAGE and ``total`` the size of the whole filtered population it was drawn
    from (#122).

    `items` is the universal data container and `kind` the envelope discriminator
    (#275); every caller passes its own `kind` (required, so a new list read
    can't omit it). The CLI can't compute the true total itself (it asks for a
    bounded page), so the bridge -- which has the filtered population -- returns
    total/offset/limit/returned/has_more alongside the page. Callers that
    already hold the page (the #814 paging paths build rows for the window only)
    share this shape with the ones that still slice a materialized list."""
    return {
        "kind": kind,
        "items": items,
        "total": total,
        "offset": offset,
        "limit": limit,
        "returned": len(items),
        "has_more": (offset + len(items)) < total,
    }


def _paged_list_result(items: list[dict[str, Any]], *, offset: int,
                       limit: int | None, kind: str) -> dict[str, Any]:
    """Return a list page WITH paging metadata (the strings/imports/sections
    envelope) from an already-materialized *items* list.

    Mirrors ``read_listing._paged_function_result`` so the simple list ops
    expose the same honest paging contract -- the true total plus the remainder
    -- as ``function list``/``function search`` (#122). Slices the requested
    window here, then wraps it; a caller whose per-row work is expensive should
    instead page its population first (``_page_window``) and build rows for that
    window only (#814)."""
    start, stop = _page_window(len(items), offset=offset, limit=limit)
    return _paged_envelope(kind=kind, items=items[start:stop], total=len(items),
                           offset=offset, limit=limit)


# Sections that hold loader/linker METADATA strings rather than domain literals
# (#646), matched as name PREFIXES on top of the semantics test below: BN
# classifies `.dynstr` as ReadOnlyData exactly like `.rodata` (verified against
# a live view), so semantics alone cannot tell them apart -- and `.dynstr` /
# `.interp` sit at the LOWEST string addresses on every ELF, so they own the
# head of any address-ordered sample. Prefixes, not substrings: `.data.rel.ro`
# is data, not `.rel.dyn`. `.relr.` is the packed-relative form (`-z
# pack-relative-relocs`) that neither `.rel.` nor `.rela` covers.
# `.text`/`.init`/`.fini` are here for a view whose code sections carry no
# semantics (ReadOnlyCode is the normal signal).
_METADATA_STRING_SECTIONS = (
    ".interp", ".dynstr", ".dynsym", ".gnu.", ".hash", ".eh_frame", ".rela",
    ".rel.", ".relr.", ".dynamic", ".strtab", ".symtab", ".shstrtab", ".comment",
    ".note", ".plt", ".got", ".debug", ".text", ".init", ".fini",
)

# String ranking tiers (M-12). A string's tier is the tier of the section that
# holds it: `.rodata`-like (ReadOnlyData) sections carry the domain literals,
# `.data`-like sections sometimes do, and executable code (whose "strings" are
# instruction bytes that happen to decode as printable ASCII) and linker
# metadata are noise. An address no section claims is judged by its SEGMENT,
# and only on a view where BN has marked code at all: BN 6.1 never produces a
# sectionless view (every loader adds `.synthetic_builtins`, an
# ExternalSection), so "no section" is the ordinary state of every string on a
# raw / monolithic firmware image or a section-header-stripped ELF -- there it
# is the literals, not noise.
_STRING_TIER_RODATA = 0
_STRING_TIER_DATA = 1
_STRING_TIER_NOISE = 2
_STRING_NO_SECTION = "(no section)"


def _section_semantics_name(sec) -> str:
    raw_semantics = getattr(sec, "semantics", 0)
    try:
        return _SECTION_SEMANTICS_NAMES.get(int(raw_semantics), str(raw_semantics))
    except (TypeError, ValueError):
        return str(raw_semantics)


def _string_section_tier(sec) -> int | None:
    """Which ranking tier a section's strings belong to, by BN semantics first
    (`ReadOnlyCode` is noise regardless of name, `ReadOnlyData` is preferred)
    and the metadata name list second -- or ``None`` for a section that cannot
    claim a string at all: an `ExternalSection` (BN's synthetic import /
    builtin table, present on EVERY view) and a NOBITS section (`.bss`,
    `.tbss`), which has no file bytes, so its address range -- `.tbss` overlaps
    the head of `.data.rel.ro` on static AArch64 -- belongs to whatever real
    section shares it."""
    semantics = _section_semantics_name(sec)
    if semantics == "ExternalSection":
        return None
    if str(getattr(sec, "type", "") or "").upper() == "NOBITS":
        return None
    name = str(getattr(sec, "name", "") or "").lower()
    if semantics == "ReadOnlyCode" or name.startswith(_METADATA_STRING_SECTIONS):
        return _STRING_TIER_NOISE
    if semantics == "ReadOnlyData":
        return _STRING_TIER_RODATA
    return _STRING_TIER_DATA


class _StringSectionMap:
    """The section -> tier table one `strings` read classifies against (M-12),
    built once per read so a 50k-string scan pays one bisect per string rather
    than a walk over every section.

    ``starts`` / ``ends`` / ``tiers`` / ``names`` are parallel, address-sorted
    and NON-overlapping: where sections overlap (a ReadOnlyData container with a
    ReadOnlyCode `.text` inside it; a user section laid over a loader one) the
    overlap is resolved HERE, once, by an explicit precedence rather than by
    whichever section `bv.sections` happened to list first -- the narrowest
    claiming section wins, so a broad data section can never hide the code or
    metadata nested inside it, and between equal claims the noisier tier wins.
    ``has_code`` records whether BN marked any ReadOnlyCode section at all,
    which is the precondition for calling an unclaimed address noise."""

    __slots__ = ("bv", "starts", "ends", "tiers", "names", "has_code")

    def __init__(self, bv, claims: list[tuple[int, int, int, str]], *, has_code: bool):
        self.bv = bv
        self.has_code = has_code
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.tiers: list[int] = []
        self.names: list[str] = []
        # Sweep the section boundaries in address order, keeping the set of
        # sections open at each point; every elementary interval between two
        # consecutive boundaries has one owner, chosen by the precedence above.
        events: list[tuple[int, int, int]] = []  # (position, +1 open / -1 close, claim index)
        for index, (start, end, _tier, _name) in enumerate(claims):
            events.append((start, 1, index))
            events.append((end, -1, index))
        events.sort(key=lambda e: (e[0], e[1]))  # closes before opens at a shared boundary
        open_claims: set[int] = set()
        position = 0
        for at, kind, index in events:
            if open_claims and at > position:
                start, end, tier, name = min(
                    (claims[i] for i in open_claims),
                    key=lambda c: (c[1] - c[0], -c[2], c[3]))
                if self.names and self.names[-1] == name and self.ends[-1] == position:
                    self.ends[-1] = at
                else:
                    self.starts.append(position)
                    self.ends.append(at)
                    self.tiers.append(tier)
                    self.names.append(name)
            position = at
            if kind > 0:
                open_claims.add(index)
            else:
                open_claims.discard(index)


def _string_section_ranges(bv) -> _StringSectionMap:
    """The per-view classification table for `_strings` (see `_StringSectionMap`)."""
    claims: list[tuple[int, int, int, str]] = []
    has_code = False
    try:
        for name, sec in (getattr(bv, "sections", None) or {}).items():
            tier = _string_section_tier(sec)
            if tier is None:
                continue
            start, end = int(getattr(sec, "start", 0)), int(getattr(sec, "end", 0))
            if end <= start:
                continue
            claims.append((start, end, tier, str(name)))
            has_code = has_code or _section_semantics_name(sec) == "ReadOnlyCode"
    except Exception:
        claims, has_code = [], False
    return _StringSectionMap(bv, claims, has_code=has_code)


def _string_tier_at(ranges: _StringSectionMap, address: int) -> tuple[int, str]:
    """The tier and section name of *address*.

    An address no section claims is noise only when BN marked code somewhere
    (``has_code``) AND the segment holding the address is executable -- the ELF
    header, which decodes as a string on every AArch64 ELF, sits unclaimed in
    the r-x load segment of a view that has a `.text`. Without a ReadOnlyCode
    section the view is a raw image or a section-header-stripped ELF, where the
    unclaimed addresses ARE the literals (and on a static ELF `.rodata` shares
    the r-x segment with `.text`, so segment permissions alone cannot rank them):
    everything stays in the data tier, which is plain address order."""
    i = bisect_right(ranges.starts, address) - 1
    if i >= 0 and address < ranges.ends[i]:
        return ranges.tiers[i], ranges.names[i]
    if ranges.has_code and hasattr(ranges.bv, "get_segment_at"):
        try:
            segment = ranges.bv.get_segment_at(address)
        except Exception:
            segment = None
        if segment is not None and bool(getattr(segment, "executable", False)):
            return _STRING_TIER_NOISE, _STRING_NO_SECTION
    return _STRING_TIER_DATA, _STRING_NO_SECTION


def _strings(ctx, selector: str | None, *, query, offset: int, limit: int | None,
             min_length: int | None = None, max_length: int | None = None,
             section: str | None = None, no_crt: bool = False, regex: bool = False,
             probable_format_strings: bool = False, count_only: bool = False,
             domain_sections_only: bool = False, with_xrefs: bool = False,
             include_noise: bool = False):
    offset = _validate_count(offset, label="offset", minimum=0)
    # allow_none mirrors the sibling list ops (imports/sections): limit=None
    # means "no limit", so a raw-socket / py exec caller can fetch every string.
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    bv = ctx._resolve_view(selector)
    # In --quick mode string analysis hasn't run, so bv.strings is empty and `[]`
    # would be indistinguishable from "this binary has none". Refuse with a
    # directive instead of misleading the caller -- "retry" while a refresh is
    # already building them (#996).
    unavailable = analysis_unavailable(
        bv, "Strings", "Run `bn refresh` to build the full string set first.",
        plural=True)
    if unavailable is not None:
        raise RuntimeError(unavailable)
    needle = str(query) if query else None
    pattern = None
    if needle and regex:
        try:
            pattern = re.compile(needle, re.IGNORECASE)
        except re.error as exc:
            raise OperationFailure("invalid_regex", f"Invalid string regex: {exc}") from exc
    elif needle:
        needle = needle.lower()
    # #814: the scanning pass below (query/length/section/no-crt/format filters)
    # has to see every string -- the honest `total` is the size of the filtered
    # population, and the (address, value) order is only known once the scan is
    # done. What it must NOT do is build the enrichment the caller will never be
    # shown, so a survivor is retained as a lightweight tuple: the parsed fields
    # the order needs plus the source ref, with the `type` decode and -- in
    # --probable-format-strings mode -- the per-survivor code-xref count deferred
    # to `_string_entry`, which runs for the returned page only. Before #814 a
    # `strings --limit 20` paid both for every survivor of a 50k-string scan.
    survivors: list[tuple[int, int, str, int, Any, list[str] | None, str]] = []
    # M-12: every survivor is ranked by the tier of its section, computed once
    # per read. The default dump orders `.rodata`-like strings first and the
    # instruction-byte / linker-metadata "strings" last. `--section` pins the
    # population to one section, and THAT population stays in plain address
    # order whatever tier each address resolves to (another section may overlap
    # the pinned one), so the rank every survivor sorts by is a constant there.
    ranges = _string_section_ranges(bv)
    noise_by_section: dict[str, int] = {}
    # #1037 item 2: a `--query` is a search for domain literals, and the
    # linker-metadata / code-byte tier answered it with import names and
    # mangled symbols (`--query sock` listed `socket` from `.dynstr`; one regex
    # sweep grew to 1066 rows). Under a query those rows are dropped unless the
    # caller pinned a `--section` or asked for them with `--include-noise`, and
    # the drop is COUNTED per section so the listing says what it left out.
    drop_noise = bool(needle) and not section and not include_noise
    noise_dropped: dict[str, int] = {}
    data_by_section: dict[str, tuple[int, int]] = {}  # name -> (tier, strings)
    # #795: the candidate list is materialized so the read can disclose HOW MUCH
    # the active filters dropped. Without it the only way to the denominator was a
    # second, unfiltered invocation (`strings --count` said 1359 where
    # `--probable-format-strings --count` said 30, and nothing named the 1329 in
    # between). Counting candidates - survivors is the whole disclosure: it is the
    # same measurement `total` already is, taken one step earlier in the pipeline,
    # so it cannot disagree with the page about what the filter kept.
    candidates = list(getattr(bv, "strings", []))
    for item in candidates:
        value = str(getattr(item, "value", ""))
        length = int(getattr(item, "length", 0))
        address = int(getattr(item, "start", 0))

        if pattern is not None:
            if not pattern.search(value):
                continue
        elif needle and needle not in value.lower():
            continue
        # #946: the length filters are documented in CHARACTERS (decoded Unicode
        # code points), so they compare the decoded value's length. BN's
        # `StringReference.length` is the encoded byte span -- 4x the character
        # count for UTF-32 -- and stays the row's `length` (bytes) metadata.
        if min_length is not None and len(value) < min_length:
            continue
        if max_length is not None and len(value) > max_length:
            continue
        if section:
            secs = bv.get_sections_at(address) if hasattr(bv, "get_sections_at") else []
            if not any(getattr(s, "name", "") == section for s in secs):
                continue
        tier, section_name = _string_tier_at(ranges, address)
        # #646: drop code / loader / linker metadata strings (`.interp`, `.dynstr`
        # import names, `__gmon_start__`, `.text` bytes that decode as ASCII) so a
        # head sample carries domain literals instead of the ELF preamble.
        if domain_sections_only and tier == _STRING_TIER_NOISE:
            continue
        noise_match = drop_noise and tier == _STRING_TIER_NOISE
        if no_crt:
            if _NO_CRT_PATTERNS.match(value):
                continue
            if len(value) >= 2 and len(set(value)) == 1:
                continue
            secs = bv.get_sections_at(address) if hasattr(bv, "get_sections_at") else []
            if any(getattr(s, "name", "") == ".text" for s in secs):
                continue

        directives: list[str] | None = None
        if probable_format_strings:
            directives = _probable_format_directives(value)
            if directives is None:
                continue
        if noise_match:
            # Counted after every other filter, so the number is "rows this
            # query would have listed", not "noise strings the query touched".
            noise_dropped[section_name] = noise_dropped.get(section_name, 0) + 1
            continue

        if not section:
            if tier == _STRING_TIER_NOISE:
                noise_by_section[section_name] = noise_by_section.get(section_name, 0) + 1
            elif section_name != _STRING_NO_SECTION:
                seen = data_by_section.get(section_name, (tier, 0))
                data_by_section[section_name] = (tier, seen[1] + 1)
        survivors.append((0 if section else tier, address, value, length, item, directives,
                          section_name))
    # #795: how many candidates the ACTIVE filters dropped. An int on EVERY
    # result (0 when nothing filtered), so `jq '.filtered'` is a stable read
    # rather than a key that appears only sometimes -- the same reason `total`
    # and `has_more` are always present.
    filtered = len(candidates) - len(survivors)
    noise_dropped_total = sum(noise_dropped.values())
    if count_only:
        # `total` mirrors the list envelope key for the same number (#165).
        result = {"kind": "strings", "count": len(survivors), "total": len(survivors),
                  "filtered": filtered}
        if drop_noise:
            result["noise_dropped"] = noise_dropped_total
        return result
    # Preferred tier first, then the pre-#814 (address, value) order within a
    # tier, stable against the scan order for ties.
    survivors.sort(key=lambda row: (row[0], row[1], row[2]))
    start, stop = _page_window(len(survivors), offset=offset, limit=limit)
    page = _paged_envelope(
        kind="strings",
        items=[_string_entry(row, bv, with_xrefs=with_xrefs) for row in survivors[start:stop]],
        total=len(survivors),
        offset=offset,
        limit=limit,
    )
    page["filtered"] = filtered
    # M-12: which sections contributed code / metadata noise to this listing, so
    # the CLI tip can name them (and stay silent when there are none) instead of
    # blaming `.dynsym/.hash/.symtab` on a static file that has none -- and which
    # DATA sections the listing drew on, preferred tier first and most strings
    # first within it, so the tip can name a `--section` that exists on THIS
    # view (`.rdata` on a PE) rather than assume `.rodata`. Both are empty when
    # `--section` pinned the population: that noise is what the caller asked for.
    page["noise_strings"] = sum(noise_by_section.values())
    page["noise_sections"] = sorted(noise_by_section, key=lambda n: (-noise_by_section[n], n))
    page["data_sections"] = sorted(
        data_by_section, key=lambda n: (data_by_section[n][0], -data_by_section[n][1], n))
    if drop_noise:
        # Present whenever the drop was ACTIVE (0 when it removed nothing), so a
        # reader can tell "nothing matched in metadata" from "not dropped".
        page["noise_dropped"] = noise_dropped_total
        page["noise_dropped_sections"] = sorted(
            noise_dropped, key=lambda n: (-noise_dropped[n], n))
    if with_xrefs:
        _note_unreferenced_strings(page, bv)
    return page


def _string_entry(row: tuple[int, int, str, int, Any, list[str] | None, str], bv, *,
                  with_xrefs: bool = False) -> dict[str, Any]:
    """Build ONE `strings` row from a retained survivor tuple (#814).

    Split out so a row's per-row work -- the string-`type` decode and, in
    --probable-format-strings mode, the code-xref count plus the directive
    provenance -- is paid for the rows actually returned, never for the whole
    filtered population. (The directives themselves were already computed as the
    mode's FILTER and are handed through, so this cannot disagree with the
    filter that admitted the string.)"""
    _tier, address, value, length, item, directives, section_name = row
    raw_type = getattr(item, "type", "")
    try:
        string_type = _STRING_TYPE_NAMES.get(int(raw_type), str(raw_type))
    except (TypeError, ValueError):
        string_type = str(raw_type)
    entry = {
        "address": hex(address),
        "length": length,
        "chars": len(value),
        "type": string_type,
        "value": value,
        # #1037 item 2: the section the ranking already resolved for this row
        # (null when no section claims the address), so a JSON reader can
        # filter by it instead of joining against `bn sections` by hand.
        "section": None if section_name == _STRING_NO_SECTION else section_name,
    }
    if directives is not None:
        # Provenance for the model: WHICH printf directives the candidate
        # carries (a `%n`/`%s` here is a signal to weigh, not a verdict), and
        # the code-xref count -- a plausible format string is usually
        # referenced by code. Reuse the existing xref helper rather than
        # recompute, and only for the returned page so the mode stays cheap.
        # The count stays enrichment, NOT a hard filter: a format
        # string reached indirectly can have zero direct xrefs, and dropping
        # it would be a false negative. It carries the same name as the
        # `--with-xrefs` count it equals (#1037 item 3).
        entry["format_directives"] = directives
        entry["directive_count"] = len(directives)
        entry["code_ref_count"] = read_xrefs._code_ref_count(bv, address)
    if with_xrefs:
        entry.update(_string_xref_functions(bv, address))
    return entry


def _note_unreferenced_strings(page: dict[str, Any], bv) -> None:
    """#1037 item 11: say why `0 functions` may not mean "unused".

    A string used only from a C++ catch block is reached through a landing pad
    that `.gcc_except_table` alone points at; BN makes no function for such a
    pad, so the string shows no reference at all. When a returned row has no
    code and no data reference and the binary carries that table, the page
    counts those rows and states the limit. It claims nothing about any one
    string: it does not look for landing pads."""
    unreferenced = sum(
        1 for row in page.get("items") or []
        if row.get("code_ref_count") == 0 and row.get("data_ref_count") == 0)
    note = read_xrefs._landing_pad_note(bv) if unreferenced else None
    if note:
        page["unreferenced_strings"] = unreferenced
        page["unreferenced_note"] = note


# #968: the most referencing functions one `strings --with-xrefs` row names. The
# row states the full distinct count beside the capped list, so a string
# referenced from many functions reads as "N more", never as a short list.
_STRING_XREF_FUNCTION_CAP = 8


def _string_xref_functions(bv, address: int) -> dict[str, Any]:
    """Which functions reference the string at *address* (#968).

    Turns "string -> referencing functions" into one call: an agent recovering
    names from `__func__`-style logger strings used to pay one `xrefs` call per
    string. Code refs (the #284-filtered population `xrefs` renders) give the
    functions; data refs are counted, not resolved, because a pointer table
    holding the string is not a function that uses it."""
    code_refs = read_xrefs._genuine_code_refs(bv, address)
    functions: dict[int, Any] = {}
    for ref in code_refs:
        fn = getattr(ref, "function", None)
        if fn is None:
            continue
        functions.setdefault(int(fn.start), fn)
    entry: dict[str, Any] = {"code_ref_count": len(code_refs)}
    try:
        entry["data_ref_count"] = len(read_xrefs._data_refs_once(bv, address))
    except Exception as exc:  # noqa: BLE001 - one unreadable row must not fail the page
        # Unmeasured, not zero: a 0 here would read as "no pointer table holds
        # this string", which nothing established (#619/#715).
        entry["data_ref_count"] = None
        entry["data_ref_note"] = f"data refs could not be read: {type(exc).__name__}: {exc}"
    ordered = sorted(functions.items())
    entry["xref_function_count"] = len(ordered)
    # #978: raw `name` plus the demangled `display_name` the text shows, read
    # for the capped rows only.
    entry["xref_functions"] = [
        il_format._function_identity(fn)
        for _start, fn in ordered[:_STRING_XREF_FUNCTION_CAP]
    ]
    return entry


def _needed_libraries(bv, elf_info: ElfDynamicInfo | None = None) -> list[str]:
    """DT_NEEDED shared objects this binary links against: BN's list, else the
    one the file's own dynamic section names."""
    try:
        libraries = sorted({str(lib) for lib in (getattr(bv, "libraries", None) or [])})
    except Exception:
        libraries = []
    if not libraries and elf_info is not None:
        libraries = sorted(set(elf_info.needed))
    return libraries


def _elf_dynamic_info(bv) -> ElfDynamicInfo | None:
    """The ELF dynamic-section facts BN does not expose (#969), read from the
    ORIGINAL file bytes (the view's raw parent), or None for a non-ELF / static
    view or one with no raw parent.

    The raw view's length bounds every read the parser issues: BN's
    ``BinaryView.read`` segfaults on a hostile size, so a view whose length is
    unknown is not parsed at all. A parser failure is a bug, not "not an ELF":
    it is logged and surfaced as a note instead of reading as an absent
    dynamic section."""
    raw = getattr(bv, "parent_view", None)
    read = getattr(raw, "read", None)
    if not callable(read):
        return None
    try:
        base = int(getattr(raw, "start", 0) or 0)
        length = getattr(raw, "length", None)
        if length is None:
            return None
        return parse_elf_dynamic(lambda offset, size: read(base + offset, size), int(length))
    except Exception as exc:  # noqa: BLE001 - surfaced below, never swallowed
        message = f"the ELF dynamic section could not be parsed: {type(exc).__name__}: {exc}"
        if bn is not None and callable(getattr(bn, "log_warn", None)):
            bn.log_warn(f"BN Agent Bridge: {message}")
        return ElfDynamicInfo(notes=[message])


def _dynamic_link_info(bv) -> dict[str, Any]:
    """`target info`'s dynamic-linking block (#969): DT_NEEDED, DT_SONAME,
    DT_RPATH and DT_RUNPATH, plus `dynamic_section_notes` naming any part of
    the dynamic section that was out of range and skipped. Every key is always
    present (an empty list / null when the target has none), so a reader never
    has to tell "absent" from "not reported"."""
    elf_info = _elf_dynamic_info(bv)
    return {
        "needed_libraries": _needed_libraries(bv, elf_info),
        "soname": elf_info.soname if elf_info else None,
        "rpath": elf_info.rpath if elf_info else None,
        "runpath": elf_info.runpath if elf_info else None,
        "dynamic_section_notes": list(elf_info.notes) if elf_info else [],
    }


def _defined_symbol_names(bv) -> set[str]:
    """Raw names of symbols DEFINED in this module (real function/data definitions).

    Used to recognize a PIC shared object's own exports that BN also models as
    import veneers / GOT slots, so those self-references can be dropped from the
    imports survey (#202). Keyed on ``raw_name`` because that's what the import
    veneer and GOT slot carry too.

    A Function/Data symbol that sits AT an import address (an ImportedFunction /
    ImportedData / ImportAddress symbol address) is an import VENEER -- on PE64 BN
    co-names the IAT jump-thunk (FunctionSymbol) AND a data veneer
    (DataSymbol, sharing an ImportedDataSymbol address) with the import at its
    very address -- NOT a real local
    definition, so it is excluded here (#379). Otherwise the #202 filter would
    drop the genuine import and leave a near-empty PE import list. A real #202
    self-export def sits at its own .text/.data address, distinct from the
    veneer, so it is kept."""
    import_addrs: set[int] = set()
    # #827 item 3 / #888: the THIRD copy of the same symbol-kind family, inline
    # rather than as a constant, which is why the original bullet did not name
    # it. Read from the one owner too -- the kind names are the part that must
    # not drift; this loop needs only the names, not the (name, kind) pairing.
    for attr, _kind in read_xrefs._IMPORT_SYMBOL_TYPES:
        sym_type = getattr(bn.SymbolType, attr, None)
        if sym_type is None:
            continue
        for sym in list(bv.get_symbols_of_type(sym_type)):
            try:
                import_addrs.add(int(sym.address))
            except (TypeError, ValueError):
                pass
    names: set[str] = set()
    for attr_name in ("FunctionSymbol", "DataSymbol"):
        sym_type = getattr(bn.SymbolType, attr_name, None)
        if sym_type is None:
            continue
        for sym in list(bv.get_symbols_of_type(sym_type)):
            try:
                if int(sym.address) in import_addrs:
                    continue  # import veneer at an IAT address, not a real def (#379)
            except (TypeError, ValueError):
                pass
            names.add(str(getattr(sym, "raw_name", sym.name)))
    return names


# ExternalSymbol surfaces (a) statically-linked-but-undefined helpers and (b)
# the kernel-symbol refs of an ET_REL `.ko` (which BN models in .extern, not as
# Imported*Symbol) -- enumerating it makes a kernel module's API surface visible
# via `bn imports` (#213). Appended AFTER the Imported* kinds so a regular ELF's
# externals that also appear as PLT imports dedup against them.
_IMPORT_PLUS_EXTERNAL: list[tuple[str, str]] = _IMPORT_SYMBOL_TYPES + [("ExternalSymbol", "external")]


def _slot_carries_jump_slot_reloc(bv, addr: int) -> bool:
    """True when the GOT slot at *addr* carries an ELF JUMP_SLOT relocation -- a
    callable function import from ``.rela.plt``, even when BN recovered no PLT-stub
    ImportedFunctionSymbol for it. Analysis-independent (relocations are parsed at
    load), so it holds on the exact target where PLT-stub recovery failed. False for
    GLOB_DAT data slots and for raw/non-ELF views with no relocation table (#478)."""
    jump_slot = getattr(getattr(bn, "RelocationType", None), "ELFJumpSlotRelocationType", None)
    reloc_at = getattr(bv, "relocations_at", None)
    if jump_slot is None or reloc_at is None:
        return False
    try:
        relocs = reloc_at(int(addr)) or []
    except Exception:
        return False
    for r in relocs:
        info = getattr(r, "info", None)
        rtype = getattr(info, "type", None) if info is not None else getattr(r, "type", None)
        if rtype == jump_slot:
            return True
    return False


def _callable_import_slot_names(bv) -> set[str]:
    """Raw names of callable function-import GOT slots -- ImportAddressSymbols whose
    slot carries a JUMP_SLOT relocation. Keeps `imports` classification and
    `imported_function_count` honest when BN failed to recover PLT-stub function
    symbols (#478). Empty on raw/non-ELF views."""
    names: set[str] = set()
    sym_type = getattr(bn.SymbolType, "ImportAddressSymbol", None)
    if sym_type is None:
        return names
    try:
        syms = list(bv.get_symbols_of_type(sym_type))
    except Exception:
        return names
    # Mirror the `_imports` self-reference filter (#202): a PIC .so models its OWN
    # exported functions as GOT slots too; those are self-references, not genuine
    # imports, so exclude them here as well to keep imported_function_count
    # consistent with the `imports` listing (review defense-in-depth).
    try:
        defined = _defined_symbol_names(bv)
    except Exception:
        defined = set()
    for sym in syms:
        if _slot_carries_jump_slot_reloc(bv, getattr(sym, "address", 0)):
            raw = str(getattr(sym, "raw_name", getattr(sym, "name", "")) or "")
            if raw and raw not in defined:
                names.add(raw)
    return names


def _imports(ctx, selector: str | None, *, summary: bool = False,
             query: str | None = None, regex: bool = False,
             offset: int = 0, limit: int | None = None, count_only: bool = False,
             include_got: bool = False, with_callers: bool = False,
             sysroot: str | None = None):
    # Guard paging the same way the sibling list ops do, so a raw-socket /
    # py exec caller passing a negative offset/limit gets a clean
    # invalid_request instead of a silent Python negative-index slice (#68).
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    bv = ctx._resolve_view(selector)
    elf_info = _elf_dynamic_info(bv)
    needed_libraries = _needed_libraries(bv, elf_info)
    versioned = elf_info.import_libraries if elf_info is not None else {}

    def _library_for(namespace: str, raw_name: str) -> str | None:
        # A real per-library BN namespace wins; BN's sentinels are not a
        # dependency. Otherwise the loader's own binding: the verneed library
        # the symbol's version names (#969). An unversioned import stays None
        # rather than being guessed from the DT_NEEDED list.
        if namespace and namespace not in _BN_SENTINEL_NAMESPACES:
            return namespace
        return versioned.get(raw_name) or versioned.get(raw_name.split("@", 1)[0])

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
            # Per-symbol kind: `kind` is the OUTER loop variable (shared across every
            # symbol of this type), so never mutate it -- the #478 reclassification
            # below would otherwise leak into the next symbol's iteration.
            entry_kind = kind
            # #212: an address-kind entry is the GOT slot for an import already
            # listed as its function/data PLT entry -- ~half of a standard ELF's
            # import list is these duplicates. Collapse them by default; show with
            # --include-got. A genuinely unique address symbol is still listed.
            if entry_kind == "address" and not include_got and raw_name in func_data_names:
                got_collapsed += 1
                continue
            # An ET_REL external that ALSO appeared as an import on a regular ELF
            # is the same dependency. Dedup against EVERY already-emitted import
            # name -- not just function/data -- because on a standard ELF most
            # imports are GOT-only (ImportAddressSymbol, kind=address) with no PLT
            # function entry, and those same names reappear as ExternalSymbol;
            # checking only func_data_names would double-list them (#213 review).
            if entry_kind == "external" and raw_name in emitted_names:
                continue
            # #478: a standalone address-kind GOT slot carrying a JUMP_SLOT
            # relocation is a callable function import BN failed to recover a
            # PLT-stub symbol for -- reclassify it function-kind so a sink-rich
            # target isn't misread as stripped/static. Guarded to NON-dup slots
            # (raw_name absent from the function/data imports emitted above), so a
            # well-analyzed binary -- where the slot collapses against its PLT
            # function entry, or is shown as its GOT view under --include-got -- is
            # unaffected and never double-counted.
            if (entry_kind == "address" and raw_name not in func_data_names
                    and _slot_carries_jump_slot_reloc(bv, sym.address)):
                entry_kind = "function"
            namespace = str(getattr(sym, "namespace", "") or "")
            # A real per-library namespace, else the verneed binding (#969);
            # BN's sentinels never surface as a dependency. `namespace` keeps
            # the raw value under an honest name.
            library = _library_for(namespace, raw_name)
            items.append(
                {
                    "name": name,
                    "address": hex(sym.address),
                    "library": library,
                    "namespace": namespace,
                    "raw_name": raw_name,
                    "kind": entry_kind,
                }
            )
            emitted_names.add(raw_name)
            if entry_kind in ("function", "data"):
                func_data_names.add(raw_name)
    relocation_recovered = 0
    jump_slot_type = getattr(
        getattr(bn, "RelocationType", None),
        "ELFJumpSlotRelocationType",
        None,
    )
    try:
        relocations = list(getattr(bv, "relocations", []) or [])
    except Exception:
        relocations = []
    for relocation in relocations:
        symbol = getattr(relocation, "symbol", None)
        if symbol is None:
            continue
        raw_name = str(
            getattr(symbol, "raw_name", "")
            or getattr(symbol, "name", "")
        )
        if not raw_name or raw_name in defined_names or raw_name in emitted_names:
            continue
        info = getattr(relocation, "info", None)
        relocation_type = (
            getattr(info, "type", None)
            if info is not None
            else getattr(relocation, "type", None)
        )
        entry_kind = "function" if relocation_type == jump_slot_type else "external"
        address = int(
            getattr(
                relocation,
                "address",
                getattr(symbol, "address", 0),
            )
        )
        namespace = str(getattr(symbol, "namespace", "") or "")
        items.append(
            {
                "name": read_xrefs._import_symbol_name(symbol),
                "address": hex(address),
                "library": _library_for(namespace, raw_name),
                "namespace": namespace,
                "raw_name": raw_name,
                "kind": entry_kind,
                "provenance": "relocation",
            }
        )
        emitted_names.add(raw_name)
        relocation_recovered += 1
    # #1037 item 6: say where every attributed library came from, and give an
    # unversioned import the library the loader would bind it to when the
    # needed libraries can be read from disk.
    needed_exports = _attribute_from_needed_exports(bv, items, elf_info, sysroot)
    # #450: filter the survey to matching imports so a sink/source sweep
    # (`--query 'system|execve|popen' --regex`) doesn't need full paging + external
    # jq. Matches name / raw_name / library / namespace, mirroring strings/sections.
    if query:
        if regex:
            try:
                pattern = re.compile(query, re.IGNORECASE)
            except re.error as exc:
                raise OperationFailure("invalid_regex", f"Invalid import regex: {exc}") from exc

            def _hit(text: str) -> bool:
                return bool(pattern.search(text))
        else:
            needle = query.lower()

            def _hit(text: str) -> bool:
                return needle in text.lower()

        items = [
            it for it in items
            if any(_hit(str(it.get(f) or "")) for f in ("name", "raw_name", "library", "namespace"))
        ]
    if count_only:
        result = {"kind": "imports", "count": len(items), "total": len(items)}
        if self_defined_excluded:
            result["self_defined_excluded"] = self_defined_excluded
        if got_collapsed:
            result["got_collapsed"] = got_collapsed
        if relocation_recovered:
            result["relocation_recovered"] = relocation_recovered
        return result
    if summary:
        # Summary aggregates the whole import set; paging would distort the
        # counts, so it always reflects every symbol regardless of offset/limit.
        result = _imports_build_summary(items, needed_libraries)
        if elf_info is not None and elf_info.notes:
            result["dynamic_section_notes"] = list(elf_info.notes)
        if self_defined_excluded:
            result["self_defined_excluded"] = self_defined_excluded
        if got_collapsed:
            result["got_collapsed"] = got_collapsed
        if relocation_recovered:
            result["relocation_recovered"] = relocation_recovered
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
    if relocation_recovered:
        result["relocation_recovered"] = relocation_recovered
    # #969: rows whose `library` could not be attributed (an unversioned import,
    # or a target with no symbol versioning at all), counted over the whole
    # listing, with the DT_NEEDED list beside it -- so `library: null` points at
    # the answer instead of reading as "no dependency".
    unattributed = sum(1 for item in items if not item["library"])
    if unattributed and needed_libraries:
        result["library_unattributed"] = unattributed
        result["needed_libraries"] = needed_libraries
    if needed_exports is not None:
        # Which file each needed library was read from (null: not found or not
        # readable), so an attribution, or the lack of one, can be checked.
        result["needed_library_files"] = [
            {"library": library, "path": path} for library, path in needed_exports.files]
    # Why attribution may be partial: a dynamic-section structure that was out
    # of range for the file and skipped rather than read (#987 review).
    if elf_info is not None and elf_info.notes:
        result["dynamic_section_notes"] = list(elf_info.notes)
    if with_callers:
        _attach_import_callers(bv, result)
    return result


def _view_binary_path(bv) -> str | None:
    """The on-disk path of the binary *bv* was loaded from, or None. A view
    opened from a database names the `.bndb`; its original file is what the
    loader's `$ORIGIN` means."""
    file = getattr(bv, "file", None)
    for attr in ("original_filename", "filename"):
        try:
            path = getattr(file, attr, None)
        except Exception:  # noqa: BLE001 - an unreadable attribute is no path
            path = None
        if isinstance(path, str) and path and not path.endswith(".bndb") \
                and os.path.isfile(path):
            return path
    return None


def _attribute_from_needed_exports(bv, items: list[dict[str, Any]],
                                   elf_info: ElfDynamicInfo | None,
                                   sysroot: str | None) -> NeededExports | None:
    """Stamp `library_source` on attributed rows, and attribute the rest from
    the needed libraries' export tables (#1037 item 6).

    Sources: `namespace` (BN's own per-library namespace), `symbol-version`
    (the verneed entry the symbol's version names, #969) and `needed-exports`
    (the first needed library, in load order, whose file was read and defines
    the name; see `elf_dynamic.needed_export_libraries`). Returns what was read,
    or None when no row needed it or the target has no DT_NEEDED list."""
    unattributed = []
    for item in items:
        namespace = item.get("namespace") or ""
        if item.get("library"):
            item["library_source"] = ("namespace" if namespace
                                      and namespace not in _BN_SENTINEL_NAMESPACES
                                      else "symbol-version")
        else:
            unattributed.append(item)
    if not unattributed or elf_info is None or not elf_info.needed:
        return None
    search_dirs = library_search_dirs(
        binary_path=_view_binary_path(bv), runpath=elf_info.runpath,
        rpath=elf_info.rpath, sysroot=sysroot)
    found = needed_export_libraries(elf_info.needed, search_dirs,
                                    machine=elf_info.machine, sysroot=sysroot)
    for item in unattributed:
        raw_name = str(item.get("raw_name") or "")
        library = found.libraries.get(raw_name) or found.libraries.get(raw_name.split("@", 1)[0])
        if library:
            item["library"] = library
            item["library_source"] = "needed-exports"
    return found


# #1037 item 5: the most calling functions one `imports --with-callers` row
# names, as for `strings --with-xrefs` (#968); the row states the full count.
_IMPORT_CALLER_CAP = 8


def _attach_import_callers(bv, page: dict[str, Any]) -> None:
    """Name the functions that call each import on the returned page.

    One call instead of one `xrefs` per import. Counts BN's recorded code
    references to the import's address (the #284-filtered population `xrefs`
    renders); a row with none is counted on the envelope, because `bn xrefs
    <import>` also scans function LLIL for calls BN did not record and this
    sweep does not."""
    unrecorded = 0
    for item in page.get("items") or []:
        try:
            address = int(item["address"], 16)
            refs = read_xrefs._genuine_code_refs(bv, address)
        except Exception as exc:  # noqa: BLE001 - one unreadable row must not fail the page
            item["code_ref_count"] = None
            item["caller_count"] = None
            item["callers"] = []
            item["callers_note"] = f"code refs could not be read: {type(exc).__name__}: {exc}"
            continue
        functions: dict[int, Any] = {}
        for ref in refs:
            fn = getattr(ref, "function", None)
            if fn is not None:
                functions.setdefault(int(fn.start), fn)
        ordered = sorted(functions.items())
        item["code_ref_count"] = len(refs)
        item["caller_count"] = len(ordered)
        item["callers"] = [il_format._function_identity(fn)
                           for _start, fn in ordered[:_IMPORT_CALLER_CAP]]
        if not refs:
            unrecorded += 1
    page["callers_cap"] = _IMPORT_CALLER_CAP
    page["callers_unrecorded"] = unrecorded


def _exports(ctx, selector: str | None, *, offset: int = 0, limit: int | None = None,
             count_only: bool = False):
    """List global/weak definitions from any symbol table, not loader-visible exports.

    Select FunctionSymbol/DataSymbol definitions from BN's combined symbols by
    binding, without checking their source table or dynamic-loader visibility.
    On ELF, this can include .symtab definitions absent from .dynsym, so it is
    NOT equivalent to ``nm -D --defined-only``. Known local/no-binding symbols
    are excluded; definitions without binding metadata are retained.
    ``count_only`` counts the same population as the listing. Mirrors the
    ``imports`` envelope (name/address/kind + demangled display_name) (#198).
    """
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
    by_library: dict[str, int] = {}
    for item in items:
        ns = str(item.get("namespace", "") or "") or "(none)"
        namespaces[ns] = namespaces.get(ns, 0) + 1
        kind = str(item.get("kind", "unknown"))
        by_kind[kind] = by_kind.get(kind, 0) + 1
        # #969: the per-row `library` attribution, aggregated; an import no
        # version requirement binds is counted, not dropped.
        library = str(item.get("library") or "") or "(unattributed)"
        by_library[library] = by_library.get(library, 0) + 1
    return {
        # #275: a keyed aggregate, not a flat collection -- keeps its namespaces/
        # by_kind maps but carries a `kind` discriminator like every other read.
        "kind": "imports_summary",
        "total_symbols": len(items),
        "needed_libraries": needed_libraries or [],
        "by_library": dict(sorted(by_library.items(), key=lambda x: (-x[1], x[0]))),
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
    all_entries: list[dict[str, Any]] = []
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
                writable = bool(getattr(seg, "writable", None))
                executable = bool(getattr(seg, "executable", None))
                entry["readable"] = bool(getattr(seg, "readable", None))
                entry["writable"] = writable
                entry["executable"] = executable
                # #453: a direct W+X verdict. `executable` is SEGMENT-derived, so
                # read-only metadata (`.rodata` in an r-x load segment) reads
                # executable=true -- but it is not W+X unless also writable. Give
                # the canonical answer (writable AND executable) + where the perms
                # came from, so agents/fan-out summaries don't re-derive it and
                # mis-flag data as executable attack surface.
                entry["writable_executable"] = writable and executable
                entry["permission_source"] = "segment"

        all_entries.append(entry)
        # The query filters the RETURNED page only. The W+X verdict below is computed
        # over the FULL section set (all_entries), so scoping `sections --query x` to
        # unrelated sections can't flip the verdict to a false "no W+X" all-clear on a
        # genuinely W+X image (#461 audit P1).
        # Match the query against the section name OR its semantics label, so
        # `--query code` finds executable sections (.text = ReadOnlyCode) even
        # though "code" is not in the name (#257).
        if needle and needle not in name.lower() and needle not in semantics.lower():
            continue
        items.append(entry)
    # W+X verdict over the FULL section set (query-independent), computed before
    # paging so a truncated page still reports the true verdict/count.
    wx_items = [e["name"] for e in all_entries if e.get("writable_executable")]
    # "Have perms" means at least one section carries a REAL permission bit. A
    # synthetic/external section with all-false perms (no backing segment, or a
    # segment mapped r/w/x = false) is insufficient metadata, not a verified
    # not-W+X -- so it must NOT satisfy have_perms (#461).
    have_perms = any(
        e.get("readable") or e.get("writable") or e.get("executable")
        for e in all_entries
    )
    # #461: an EXPLICIT top-level verdict, always present. On a mapped/raw embedded
    # view whose sections carry only synthetic/external metadata (no segment perms),
    # the writable_executable set is empty -- but that proves the metadata is
    # insufficient, NOT that there are no executable-writable regions. Say
    # "unknown_insufficient_metadata" so an agent doesn't read the empty set as a
    # W+X all-clear. When perms ARE present, distinguish present vs not-observed.
    if not have_perms:
        wx_verdict = "unknown_insufficient_metadata"
    elif wx_items:
        wx_verdict = "wx_sections_present"
    else:
        wx_verdict = "no_wx_sections_observed"
    wx_summary: dict[str, Any] = {"wx_verdict": wx_verdict}
    if have_perms:
        wx_summary["writable_executable_count"] = len(wx_items)
        wx_summary["writable_executable_items"] = wx_items
    if count_only:
        return {"kind": "sections", "count": len(items), "total": len(items), **wx_summary}
    items.sort(key=lambda item: int(item["start"], 16))
    result = _paged_list_result(items, offset=offset, limit=limit, kind="sections")
    result.update(wx_summary)
    return result


_DATA_VARS_DEFAULT_LIMIT = 400
# The standard on-screen page every sibling list read defaults to (`cli.py`'s
# `_effective_limit`). Applied HERE, not only on the CLI, so the op's own
# default bounds a direct bridge caller too (#682 item 1).
_DATA_SYMBOLS_DEFAULT_LIMIT = 100
# ...and the explicit way past it. A TOKEN, deliberately not a number: a
# caller-visible `limit=0` already means "the schema, not the rows" across
# this repo's collection helpers (skills/bn-kernel/SKILL.md), which the client
# layers turn into a one-row wire probe precisely because every op refuses a
# zero limit. Spelling "everything" as the same value that means "as close to
# nothing as a page can get" is a trap, and no numeric limit below 1 is free
# of it. A word also cannot be produced by int() coercion, so no malformed
# limit falls into the unbounded read.
_DATA_SYMBOLS_ALL = "all"


def _is_pointer_type(type_, type_text: str) -> bool:
    """Whether *type_* is a single pointer, for the data-var decode.

    BN's ``type_class`` is authoritative whenever it is reachable, and the
    reason is the DENIAL it makes: a one-element pointer array --
    ``void (*[0x1])()`` -- renders with a ``*`` and its width equals one
    pointer, so the old ``"*" in text and width == psz`` test collapsed it to
    its first element and hid that it was an array at all. The enum class is
    ``ArrayTypeClass`` there, so it says no.

    What it does NOT do (#682 item 4, correcting this docstring): confirm a
    typedef'd pointer whose declaration text carries no ``*``. A typedef is
    modeled as a ``NamedTypeReferenceType``, whose ``type_class`` is
    ``NamedTypeReferenceClass`` -- not ``PointerTypeClass`` -- so wherever the
    enum decides, it DENIES the typedef.

    The enum decides only when BOTH halves of the check below are in hand:
    ``bn.TypeClass.PointerTypeClass`` resolves (the whole attribute chain --
    a ``TypeClass`` namespace lacking that member raises and counts as
    missing, which is what the ``try`` is for) AND this type object exposes a
    ``type_class``. Missing EITHER one drops through to the rendered text,
    and on that path a typedef is judged by its text like anything else --
    as is the one-element pointer array, so the denial above is unavailable
    there too. On a build where both halves are in hand (the normal case) the
    typedef is denied, the var is listed with its address, name, type text
    and width, and it simply carries no decoded ``pointer``: the intended
    conservative behaviour, not an accident of the check.

    Note this is NOT the composition ``read_evidence`` uses: there the
    rendered text is the default and ``type_class`` may only ADD a pointer
    verdict, so a typedef'd pointer can never be denied by the enum. Here the
    enum wins outright, which is what makes the array denial possible.
    """
    try:
        pointer_class = bn.TypeClass.PointerTypeClass
    except Exception:
        pointer_class = None
    observed = getattr(type_, "type_class", None)
    if pointer_class is not None and observed is not None:
        return observed == pointer_class
    return "*" in type_text


# Type classes whose bytes an integer read renders faithfully. Deliberately a
# whitelist: an aggregate (array/struct) must show as itself rather than as its
# first machine word, and a float's bit pattern read as an integer is a
# meaningless number rather than the value the type promises.
_SCALAR_TYPE_CLASS_NAMES = ("BoolTypeClass", "IntegerTypeClass",
                            "EnumerationTypeClass", "WideCharTypeClass")


def _is_scalar_type(type_, type_text: str) -> bool:
    """Whether *type_*'s bytes can be reported as one integer `v`alue.

    Uses BN's ``type_class`` when reachable; falls back to the rendered text,
    where a ``*`` or a ``[`` marks the type as a pointer or an aggregate.
    """
    type_class = getattr(bn, "TypeClass", None)
    observed = getattr(type_, "type_class", None)
    if type_class is not None and observed is not None:
        allowed = [getattr(type_class, name, None) for name in _SCALAR_TYPE_CLASS_NAMES]
        return any(known is not None and observed == known for known in allowed)
    return "*" not in type_text and "[" not in type_text


def _type_is_signed(type_) -> bool:
    """Whether *type_* is declared signed.

    BN exposes ``Type.signed`` as a ``BoolWithConfidence`` (``bool()`` yields
    the value) and leaves it None for non-integers. Anything without a usable
    flag reads as UNSIGNED, which is the honest default here: ``read_int``
    itself defaults to ``sign=True``, so a ``uint32_t`` holding 0xf0000000
    would otherwise be reported as -268435456.
    """
    try:
        signed = type_.signed
    except Exception:
        return False
    if signed is None:
        return False
    try:
        return bool(signed)
    except Exception:
        return False


def _data_var_row(bv, dv, psz: int) -> dict[str, Any]:
    """One typed data variable as a row: `address`, `name` (symbol), `type`,
    `width`, plus a decoded scalar `value` or -- for a single pointer-sized
    pointer -- the `pointer` target (with `pointer_symbol`, or the
    `pointer_string` ASCII preview when the target has no symbol) and the
    containing `section`.

    Key spelling (#682 item 2): spelled out rather than the old one- and
    two-letter wire form (`a`/`n`/`t`/`w`/`v`/`p`/`ps`/`pstr`/`sec`), which is
    opaque to the models that read this JSON without the reference open.
    """
    addr = int(dv.address)
    type_ = dv.type
    try:
        width = int(type_.width)
    except Exception:
        width = 0
    # #682 item 4: `get_symbol_at` / `get_sections_at` used to sit OUTSIDE the
    # try below, so one throwing accessor killed the whole windowed read --
    # contradicting this function's own guarantee that a row survives
    # undecorated. They decorate the row; they do not define it, so a failure
    # costs the decoration and nothing else.
    try:
        sym = bv.get_symbol_at(addr)
    except Exception:  # noqa: BLE001 - a throwing accessor must not drop the row
        sym = None
    type_text = str(type_)
    row: dict[str, Any] = {
        "address": hex(addr),
        "name": sym.name if sym else "",
        "type": type_text,
        "width": width,
    }
    try:
        secs = bv.get_sections_at(addr)
    except Exception:  # noqa: BLE001 - same: the section is decoration
        secs = None
    if secs:
        row["section"] = secs[0].name
    is_pointer = _is_pointer_type(type_, type_text)
    try:
        # Every read below is explicitly signed/unsigned: bv.read_int defaults to
        # sign=True, so leaving it implicit renders unsigned data as negative.
        if is_pointer and width == psz:
            row["pointer"] = hex(bv.read_int(addr, psz, sign=False))
            tsym = bv.get_symbol_at(int(row["pointer"], 16))
            if tsym:
                row["pointer_symbol"] = tsym.name
            else:
                preview = bv.get_ascii_string_at(int(row["pointer"], 16), 4)
                if preview:
                    row["pointer_string"] = preview.value[:48]
        elif not is_pointer and 0 < width <= 8 and _is_scalar_type(type_, type_text):
            row["value"] = bv.read_int(addr, width, sign=_type_is_signed(type_))
    except Exception:
        pass  # unmapped slot: the row still lists the var, just undecorated
    return row


def _data_vars(ctx, selector: str | None, *, start, end, limit=None):
    """Typed data variables in the half-open address window [start, end).

    Seeks the window via ``get_next_data_var_after`` instead of scanning every
    data var in the view (the py_exec predecessor was O(all data vars) per
    call), and reports an honest ``has_more`` when the row cap truncated the
    window -- page by re-issuing with ``start`` = last row's address + 1.
    """
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    if limit is None:
        limit = _DATA_VARS_DEFAULT_LIMIT
    bv = ctx._resolve_view(selector)
    lo = _parse_address(start)
    hi = _parse_address(end)
    if hi <= lo:
        raise OperationFailure(
            "invalid_request",
            f"empty window: end ({hex(hi)}) must be greater than start ({hex(lo)})",
        )
    psz = int(bv.address_size)
    if lo > 0:
        dv = bv.get_next_data_var_after(lo - 1)
    else:
        dv = bv.get_data_var_at(0)
        if dv is None:
            dv = bv.get_next_data_var_after(0)
    rows: list[dict[str, Any]] = []
    has_more = False
    while dv is not None and int(dv.address) < hi:
        if len(rows) >= limit:
            has_more = True
            break
        rows.append(_data_var_row(bv, dv, psz))
        dv = bv.get_next_data_var_after(int(dv.address))
    # #275: `items` is the universal data container, `kind` the discriminator.
    return {"kind": "data_vars", "items": rows, "has_more": has_more}


def _data_symbols(ctx, selector: str | None, *, offset: int = 0, limit=None):
    """Every *named* DataSymbol as address + name -- includes internal symbols
    the exports list omits, so a renamed data global stays addressable.

    Paged by DEFAULT, at this op and not only on the CLI (#682 item 1): an
    omitted `limit` is a 100-row page, exactly like every sibling list read.
    The issue's lock half is about the caller it names -- the out-of-tree lens
    calls this op with no params at all -- so a whole-set default left that
    consumer building and serializing a row for every data global while
    holding the read lock, the inversion of the thesis the read-locked lens
    ops were added under. A caller that genuinely wants the whole set now asks
    for it: `limit="all"` is the explicit uncap, and the envelope echoes it as
    the established no-cap `limit: null`. The token is deliberately NOT `0`
    -- a caller-visible zero already means "the schema, not the rows" across
    this repo's collection helpers, which the client layers turn into a
    one-row wire probe precisely because the ops refuse a zero limit -- and
    being a word, it is also unreachable by int() coercion, so a malformed
    `"0"`/`0.4`/`false` stays refused instead of falling into an unbounded
    read. A page larger than the population is the other honest way to the
    whole set; the token is the way that needs no guess.

    The out-of-tree index build must move to that request, and until it does
    it sees the first page -- a consumer break registered in
    `tests/test_wire_consumer_contract.py`'s KNOWN_BREAKS as
    `("data_symbols", "request:limit=all")`, where the contract test reads
    the registered request back and exercises it.

    An OVERSIZED page is not an error: the window clamps to the population, so
    `limit` > total returns the whole set and an `offset` past the end returns
    an empty page with the true `total` still visible (#682 item 4).

    Rows are built for the returned WINDOW only (#682 item 1): a page no longer
    constructs a dict and serializes it for every symbol in the view. The
    population still has to be scanned for the honest `total` -- BN hands back
    the symbol list in one call and there is no count-only API -- so the
    remaining in-lock cost on the page path is that scan, not the build.
    """
    offset = _validate_count(offset, label="offset", minimum=0)
    if limit is None:
        limit = _DATA_SYMBOLS_DEFAULT_LIMIT
    elif isinstance(limit, str) and not limit.lstrip("+-").isdigit():
        # A word, so it is meant as the token rather than as a count: say
        # which word, instead of letting `_validate_count` answer "must be an
        # integer" to a caller who was reaching for the documented spelling.
        if limit != _DATA_SYMBOLS_ALL:
            raise OperationFailure(
                "invalid_request",
                f'limit must be a count >= 1 or "{_DATA_SYMBOLS_ALL}" for the '
                f"whole set, got {limit!r}",
            )
        limit = None
    else:
        limit = _validate_count(limit, label="limit", minimum=1)
    bv = ctx._resolve_view(selector)
    sym_type = getattr(getattr(bn, "SymbolType", None), "DataSymbol", None)
    if sym_type is None:
        raise OperationFailure(
            "unsupported",
            "this Binary Ninja build does not expose SymbolType.DataSymbol",
        )
    # No blanket try/except here: a failing get_symbols_of_type used to be
    # flattened into an empty list, making a real BN error indistinguishable
    # from "this binary has no data symbols" -- the silent-empty failure mode.
    symbols = bv.get_symbols_of_type(sym_type)
    named = [sym for sym in symbols if getattr(sym, "name", "")]
    start, stop = _page_window(len(named), offset=offset, limit=limit)
    syms = [
        # `a`/`n` stay terse on purpose: #682 item 2 names the `cfg` line/edge
        # keys and the `data vars` row keys, and this row pair is a separate,
        # older contract the out-of-tree lens decodes by name.
        {"a": hex(int(sym.address)), "n": sym.name} for sym in named[start:stop]
    ]
    # #275: `items` is the universal data container and `kind` the discriminator.
    # This used to hand-roll a byte-identical envelope under the name `syms`,
    # which every generic consumer (and the paging footer) has to special-case.
    return _paged_envelope(kind="data_symbols", items=syms, total=len(named),
                           offset=offset, limit=limit)


def _ascii_render(data: bytes) -> str:
    return "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in data)


# Hard ceiling on ONE raw read, mirroring the linear-disassembly cap
# (`read_decompile._LINEAR_DISASM_MAX`): a request over it is served CAPPED and
# reports that it was capped (#827 item 4). High enough that an ordinary
# whole-region dump still fits in one call -- the CLI spills the response, so a
# bound this size is a guard against a pathological length, not a limit on work.
_READ_MAX = 100_000


def _read(ctx, selector: str | None, address, length: int):
    bv = ctx._resolve_view(selector)
    addr = _parse_address(address)
    if length < 0:
        raise RuntimeError(f"read length must be non-negative, got {length}")
    requested_length = length
    capped = length > _READ_MAX
    # #827 item 4: cap the request, exactly as the linear-disassembly sibling caps
    # at `read_decompile._LINEAR_DISASM_MAX`. The only other gate was `length < 0`,
    # so a raw client (or `py exec`) could ask for the whole address space and have
    # the bridge allocate it before anyone could see the size. The cap REPORTS
    # itself (`capped`/`requested_length`/a note) rather than silently truncating:
    # a bounded dump that reads as the whole window is the defect this file's
    # sibling notes exist to prevent. The CLI spills large output, so the ceiling
    # is high enough to be a dumping primitive, not a papercut.
    effective = _READ_MAX if capped else length

    data = bytes(bv.read(addr, effective))
    if effective > 0 and not data:
        raise RuntimeError(f"Address 0x{addr:x} is not mapped (no bytes available)")

    result: dict[str, Any] = {
        "address": hex(addr),
        "length": len(data),
        "hex": data.hex(),
        "ascii": _ascii_render(data),
    }
    notes: list[str] = []
    if capped:
        result["capped"] = True
        notes.append(f"capped at {_READ_MAX} bytes (requested {requested_length})")
    if len(data) < effective:
        result["short_read"] = True
        notes.append(
            f"short read: requested {effective} bytes, only {len(data)} mapped "
            f"from 0x{addr:x}"
        )
    if notes:
        # The ORIGINAL request, so a capped read and a short read report the same
        # key for the same question ("what did the caller ask for?").
        result["requested_length"] = requested_length
        result["note"] = "; ".join(notes)
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
