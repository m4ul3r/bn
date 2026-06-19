"""Raw-ABI call evidence, pointer tables, message lensing, .init_array walking.

The evidence/pointer-table cluster that used to live on ``BinaryNinjaBridge``
moves here as module-level free functions, each taking the ``BridgeContext``
seam (``ctx``) in place of ``self``. ``BinaryNinjaBridge`` keeps a thin
delegating shim for every name the test suite / op binders reference
(``_function_evidence``, ``_pointer_table``, ``_pointer_table_for_view``,
``_message_lens``, ``_init_arrays``, ...).

Outbound calls resolve through:
  * ``ctx`` -- resolution / address-context / ABI helpers relocated to the seam
    (``_resolve_view``, ``_find_function``, ``_address_context``,
    ``_pointer_size``, ``_read_pointer_value``, ``_normalize_code_pointer``,
    ``_sections_at``);
  * ``il_format`` -- the state-free IL/disasm helpers used by the call scan
    (``_iter_llil_instructions``, ``_il_op_name``, ``_structured_disasm_entries``,
    ``_disasm_entry``, ``_hlil_call_roots``, ``_hlil_statement_text``,
    ``_hlil_pre_branch_condition``, ``_decompile_text``, ``_function_metadata``,
    ``_render_warnings``, ``_llil_constant_value``);
  * ``read_xrefs`` -- ``_xrefs_to_address`` (used by message lensing);
  * ``_shared`` -- module-free helpers (``_parse_address``, ``_validate_count``,
    ``OperationFailure``).

Import direction is one-way: this module imports ``il_format``, ``read_xrefs``,
and ``_shared`` (plus stdlib + binaryninja). It NEVER imports ``bridge`` or
``seam`` -- those import THIS module one-way (design spec §3.2). It imports
``read_xrefs`` but ``read_xrefs`` NEVER imports this module.
"""
from __future__ import annotations

import re
from typing import Any

import binaryninja as bn  # noqa: F401  (kept for parity with sibling read_* modules)

from . import il_format
from . import read_xrefs
from ._shared import OperationFailure, _parse_address, _validate_count


def _call_destination_value(ctx, insn) -> int | None:
    return il_format._llil_constant_value(getattr(insn, "dest", None))


def _target_entry_for_call(ctx, bv, value: int | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return ctx._normalize_code_pointer(bv, value)


def _il_argument_texts(ctx, node) -> list[str]:
    for attr in ("params", "parameters"):
        params = getattr(node, attr, None)
        if params is None:
            continue
        try:
            return [str(item) for item in list(params)]
        except Exception:
            return [str(params)]
    return []


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_ARG_CONSTANT_RE = re.compile(r"0x[0-9a-fA-F]+")


def _resolve_argument_value(ctx, bv, text: str) -> dict[str, Any] | None:
    """Annotate a pointer-constant argument with what it points at.

    Generic: fixes std::string::append literals, log format strings, RTTI
    names, and service identifiers in one place. Returns None for arguments
    that are not a bare hex pointer or that resolve to nothing useful.
    """
    match = _ARG_CONSTANT_RE.fullmatch(text.strip())
    if match is None:
        return None
    address = _safe_int(int(match.group(0), 16))
    if not address:
        return None
    context = ctx._address_context(bv, address)
    resolved: dict[str, Any] = {"address": hex(address), "kind": context.get("kind")}
    string = context.get("string")
    if string:
        resolved["string"] = string.get("value")
        if string.get("encoding") and string.get("encoding") != "ascii":
            resolved["encoding"] = string["encoding"]
        if string.get("truncated"):
            resolved["truncated"] = True
    symbol = context.get("symbol")
    if symbol and symbol.get("name"):
        resolved["symbol"] = symbol["name"]
    function = context.get("function")
    if function and function.get("name"):
        resolved["function"] = function["name"]
    sections = context.get("sections")
    if sections:
        resolved["section"] = sections[0].get("name")
    if not any(key in resolved for key in ("string", "symbol", "function")):
        return None
    return resolved


def _call_arguments(ctx, bv, insn, call_addr: int) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Pick one primary argument source and quarantine uncertain extras.

    One LLIL call can map to several HLIL call expressions (BN folds adjacent
    statements); blindly merging their params attributes another call's
    arguments to this one. Prefer the single HLIL call whose address matches
    this call site; if that is ambiguous fall back to MLIL, then LLIL. Other
    candidates are returned separately (JSON-only, not shown in text).
    """
    roots = il_format._hlil_call_roots(insn)
    chosen = None
    matched = [r for r in roots if _safe_int(getattr(r, "address", None)) == int(call_addr)]
    if len(matched) == 1:
        chosen = matched[0]
    elif len(roots) == 1:
        chosen = roots[0]

    mlil = getattr(insn, "mapped_medium_level_il", None)
    if chosen is not None:
        source, texts = "hlil", _il_argument_texts(ctx, chosen)
    elif mlil is not None:
        source, texts = "mlil", _il_argument_texts(ctx, mlil)
    else:
        source, texts = "llil", _il_argument_texts(ctx, insn)

    primary: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        entry: dict[str, Any] = {"index": index, "text": text}
        resolved = _resolve_argument_value(ctx, bv, text)
        if resolved is not None:
            entry["resolved"] = resolved
        primary.append(entry)

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = {(source, e["index"], e["text"]) for e in primary}

    def add_candidates(candidate_source: str, candidate_texts: list[str]) -> None:
        for index, text in enumerate(candidate_texts):
            marker = (candidate_source, index, text)
            if marker in seen:
                continue
            seen.add(marker)
            candidates.append({"source": candidate_source, "index": index, "text": text})

    add_candidates("llil", _il_argument_texts(ctx, insn))
    if mlil is not None:
        add_candidates("mlil", _il_argument_texts(ctx, mlil))
    for root in roots:
        if root is chosen:
            continue
        add_candidates("hlil", _il_argument_texts(ctx, root))
    return source, primary, candidates


def _mlil_call_text(mlil) -> str | None:
    """Render a mapped-MLIL call WITHOUT its clobber-LHS.

    BN renders a mapped-MLIL call as "<written regs> = call(dest, args...)". For
    a varargs / full-clobber callee the LHS is the entire caller-saved register
    set (~44 regs on aarch64: arg1, arg2, x2..x18, lr, v0..v31), which buries the
    one thing the field is for -- the call and its inputs. Drop the assignment
    LHS so the line mirrors the concise `arguments:` block; the full instruction
    (with outputs) is still available in the sibling `llil` field. (E17)
    """
    if mlil is None:
        return None
    text = str(mlil)
    marker = " = call("
    idx = text.find(marker)
    if idx != -1:
        return text[idx + len(" = "):]
    return text


def _function_call_evidence(ctx, bv, func, *, context: int) -> list[dict[str, Any]]:
    disasm_entries = il_format._structured_disasm_entries(bv, func)
    index_by_addr = {
        int(item["_address_int"]): index for index, item in enumerate(disasm_entries)
    }
    calls = []
    for insn in il_format._iter_llil_instructions(func):
        op_name = il_format._il_op_name(insn)
        if op_name not in {
            "LLIL_CALL",
            "LLIL_CALL_STACK_ADJUST",
            "LLIL_TAILCALL",
        }:
            continue
        call_addr = int(getattr(insn, "address", 0))
        disasm_index = index_by_addr.get(call_addr)
        previous: list[dict[str, Any]] = []
        next_instructions: list[dict[str, Any]] = []
        call_instruction = il_format._disasm_entry(bv, call_addr, arch=getattr(func, "arch", None))
        if disasm_index is not None:
            previous = [
                {"address": item["address"], "text": item["text"]}
                for item in disasm_entries[max(0, disasm_index - context) : disasm_index]
            ]
            next_instructions = [
                {"address": item["address"], "text": item["text"]}
                for item in disasm_entries[disasm_index + 1 : disasm_index + 1 + context]
            ]
            call_instruction = {
                "address": disasm_entries[disasm_index]["address"],
                "text": disasm_entries[disasm_index]["text"],
            }

        mlil = getattr(insn, "mapped_medium_level_il", None)
        dest_value = _call_destination_value(ctx, insn)
        target = _target_entry_for_call(ctx, bv, dest_value)
        arg_source, arguments, argument_candidates = _call_arguments(ctx, bv, insn, call_addr)
        calls.append(
            {
                "address": hex(call_addr),
                "operation": op_name,
                "direct": dest_value is not None,
                "target": target,
                "llil": str(insn),
                "mlil": _mlil_call_text(mlil),
                "hlil_statement": il_format._hlil_statement_text(insn),
                "pre_branch_condition": il_format._hlil_pre_branch_condition(insn),
                "argument_source": arg_source,
                "arguments": arguments,
                "argument_candidates": argument_candidates,
                "call_instruction": call_instruction,
                "previous_instructions": previous,
                "next_instructions": next_instructions,
            }
        )
    return calls


def _function_thunk_summary(ctx, bv, func) -> dict[str, Any]:
    sections = ctx._sections_at(bv, int(func.start))
    if any("plt" in str(section.get("name", "")).lower() for section in sections):
        return {
            "is_candidate": True,
            "reason": "function starts in a PLT/import trampoline section",
            "target": None,
            "sections": sections,
        }

    llil = [
        insn
        for insn in il_format._iter_llil_instructions(func)
        if il_format._il_op_name(insn) not in {"LLIL_NOP", "LLIL_UNDEF"}
    ]
    result: dict[str, Any] = {
        "is_candidate": False,
        "reason": None,
        "target": None,
        "sections": sections,
    }
    if not llil or len(llil) > 3:
        return result
    for insn in llil:
        op_name = il_format._il_op_name(insn)
        if op_name not in {"LLIL_JUMP", "LLIL_TAILCALL", "LLIL_CALL", "LLIL_CALL_STACK_ADJUST"}:
            continue
        target = _target_entry_for_call(ctx, bv, _call_destination_value(ctx, insn))
        if target is None:
            continue
        result.update(
            {
                "is_candidate": True,
                "reason": f"small function with {op_name.lower()} to another address",
                "target": target,
            }
        )
        return result

    try:
        text = il_format._decompile_text(bv, func)
    except Exception:
        text = ""
    if "/* tailcall */" in text and len(llil) <= 3:
        result.update(
            {
                "is_candidate": True,
                "reason": "small function rendered as a pseudo-C tailcall",
            }
        )
    return result


def _function_evidence(ctx, selector: str | None, identifier, *, context: int = 2):
    if context < 0:
        raise OperationFailure("invalid_context", f"Invalid evidence context size: {context}")
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier)
    text = il_format._decompile_text(bv, func)
    return {
        "function": {
            "name": func.name,
            "address": hex(func.start),
            "raw_name": getattr(func, "raw_name", func.name),
        },
        **il_format._function_metadata(func),
        "thunk": _function_thunk_summary(ctx, bv, func),
        "calls": _function_call_evidence(ctx, bv, func, context=context),
        "warnings": il_format._render_warnings(text),
    }


def _pointer_table_for_view(
    ctx,
    bv,
    start: int,
    *,
    entries: int,
    stride_size: int,
    read_width: int | None = None,
    stop_after_invalid: int | None = None,
    error_on_unmapped: bool = False,
) -> dict[str, Any]:
    # Consult the address context up front: a table whose BASE is unmapped is
    # not backed by any segment, so every row would be a fabricated
    # readable:false slot. The top-level `evidence table` command errors like
    # `bn read`; internal reuse (message-lens / init-array windows) flags it as
    # a warning instead of aborting the surrounding scan (#119).
    table_context = ctx._address_context(bv, start)
    try:
        base_readable = len(bytes(bv.read(start, 1) or b"")) > 0
    except Exception:
        base_readable = False
    # "unmapped" only when the context says so AND we genuinely cannot read the
    # base. A readable base (even one without segment metadata) is not the
    # fabricated-readable:false-slots bug this guards against.
    base_unmapped = table_context.get("kind") == "unmapped" and not base_readable
    if base_unmapped and error_on_unmapped:
        raise RuntimeError(f"Address 0x{start:x} is not mapped (no bytes available)")
    pointer_size = ctx._pointer_size(bv)
    # Read width tracks the stride for sub-pointer strides: at `--stride 4` a
    # uint32[] table must be read 4 bytes wide, else an 8-byte read overlaps the
    # next slot and every entry decodes to garbage flagged [implausible] (#225).
    # Default = min(stride, pointer_size): stride 4 -> 4-byte reads; stride 8 ->
    # 8; stride 16 (a record with an 8-byte pointer per slot) -> 8. An explicit
    # read_width overrides.
    if read_width is None:
        read_width = min(stride_size, pointer_size) if stride_size > 0 else pointer_size
    read_width = max(1, int(read_width))
    # A non-null value below the lowest mapped address can't be a pointer into
    # the image -- at a fixed --stride it's almost always an inline scalar field
    # (a uint8/uint16 flag/enum in a mixed record), not a failed pointer slot.
    # Tag those so they don't inflate the "do not resolve" warning. (#170)
    mapped_floor = max(int(getattr(bv, "start", 0) or 0), 0x1000)
    rows = []
    warnings = []
    if base_unmapped:
        warnings.append(
            f"table base {hex(start)} is unmapped; entries are not backed by any segment"
        )
    invalid_run = 0
    for index in range(entries):
        entry_address = start + index * stride_size
        value = ctx._read_pointer_value(bv, entry_address, size=read_width)
        if value is None:
            rows.append(
                {
                    "index": index,
                    "entry_address": hex(entry_address),
                    "value": None,
                    "readable": False,
                }
            )
            invalid_run += 1
            if stop_after_invalid is not None and invalid_run >= stop_after_invalid:
                warnings.append(
                    f"stopped after {invalid_run} unreadable/implausible entries at {hex(entry_address)}"
                )
                break
            continue
        target = ctx._normalize_code_pointer(bv, value)
        likely_scalar = target["status"] == "unmapped" and 0 < value < mapped_floor
        # A legitimate inline scalar field is not an "invalid" run member either,
        # so it must not trip stop_after_invalid in interior windows.
        if target["plausible"] or target["status"] == "null" or likely_scalar:
            invalid_run = 0
        else:
            invalid_run += 1
        rows.append(
            {
                "index": index,
                "entry_address": hex(entry_address),
                "value": hex(value),
                "readable": True,
                "plausible": bool(target["plausible"]),
                "likely_scalar": bool(likely_scalar),
                "target": target,
            }
        )
        if stop_after_invalid is not None and invalid_run >= stop_after_invalid:
            warnings.append(
                f"stopped after {invalid_run} unreadable/implausible entries at {hex(entry_address)}"
            )
            break

    segment = table_context.get("segment")
    section_names = [
        str(section.get("name", "")).lower()
        for section in list(table_context.get("sections") or [])
        if isinstance(section, dict)
    ]
    code_like_section = any(
        name in {".text", "__text"} or name.startswith(".plt")
        for name in section_names
    )
    data_like_section = any(
        marker in name
        for name in section_names
        for marker in ("data", "rodata", "got", "rdata", "bss", "init_array", "fini_array", "ctors", "dtors")
    )
    if isinstance(segment, dict) and segment.get("executable") and (code_like_section or not data_like_section):
        warnings.append("table start is in an executable segment; this may be code, not a pointer table")
    non_null_rows = [
        row for row in rows
        if row.get("readable") and row.get("value") not in {None, "0x0"}
    ]
    plausible_rows = [row for row in non_null_rows if row.get("plausible")]
    scalar_rows = [row for row in non_null_rows if row.get("likely_scalar")]
    # Genuine pointer slots that failed to resolve -- excludes inline scalars.
    unresolved_rows = [
        row for row in non_null_rows
        if not row.get("plausible") and not row.get("likely_scalar")
    ]
    if non_null_rows and not plausible_rows and unresolved_rows:
        warnings.append("no non-null entries resolve to mapped addresses; low confidence pointer table")
    elif unresolved_rows:
        warnings.append(
            f"{len(unresolved_rows)} non-null entries do not resolve to mapped addresses"
        )
    if scalar_rows:
        warnings.append(
            f"{len(scalar_rows)} non-null entries look like inline scalar fields, not pointers "
            "(small values below the lowest mapped address)"
        )
    interior_function_rows = [
        row for row in non_null_rows
        if isinstance(row.get("target"), dict)
        and isinstance(row["target"].get("function"), dict)
        and row["target"]["function"].get("exact_start") is False
    ]
    if interior_function_rows:
        warnings.append(
            f"{len(interior_function_rows)} entries resolve inside functions but not at function starts"
        )
    return {
        "address": hex(start),
        "pointer_size": pointer_size,
        "stride": stride_size,
        "read_width": read_width,
        "context": table_context,
        "entries": rows,
        "warnings": warnings,
    }


def _pointer_table(ctx, selector: str | None, address, *, entries: int = 16, stride=None, width=None):
    if entries < 0:
        raise OperationFailure("invalid_entries", f"Invalid table entry count: {entries}")
    bv = ctx._resolve_view(selector)
    start = _parse_address(address)
    pointer_size = ctx._pointer_size(bv)
    stride_size = _parse_address(stride) if stride not in (None, "") else pointer_size
    if stride_size <= 0:
        raise OperationFailure("invalid_stride", f"Invalid table stride: {stride_size}")
    # Explicit --width overrides the stride-derived read width (#225).
    read_width = _parse_address(width) if width not in (None, "") else None
    if read_width is not None and read_width <= 0:
        raise OperationFailure("invalid_width", f"Invalid read width: {read_width}")
    # #275: present the canonical collection envelope at the op level (items +
    # kind), while the reusable _pointer_table_for_view helper keeps its `entries`
    # key for the nested table windows embedded by message-lens / init-arrays.
    view = _pointer_table_for_view(
        ctx,
        bv,
        start,
        entries=entries,
        stride_size=stride_size,
        read_width=read_width,
        error_on_unmapped=True,
    )
    rows = view.pop("entries", [])
    return {"kind": "pointer_table", **view, "items": rows, "total": len(rows)}


def _section_names_at(context) -> set[str]:
    return {
        str(s.get("name", "")).lower()
        for s in (context.get("sections") or [])
        if isinstance(s, dict) and s.get("name")
    }


def _symbol_by_any_name(bv, name: str):
    """A symbol matching *name* by raw (mangled) name, then by display name."""
    graw = getattr(bv, "get_symbol_by_raw_name", None)
    if callable(graw):
        try:
            s = graw(name)
            if s is not None:
                return s
        except Exception:
            pass
    gbn = getattr(bv, "get_symbols_by_name", None)
    if callable(gbn):
        try:
            ss = list(gbn(name) or [])
            if ss:
                return ss[0]
        except Exception:
            pass
    return None


# RTTI data-symbol tags (Itanium ABI): vtable / typeinfo / typeinfo-name. For a
# mangled type fragment `N5TCLAP3ArgE`, the symbols are `_ZTVN5TCLAP3ArgE`, etc.
_RTTI_PREFIXES = (("_ZTV", "vtable"), ("_ZTI", "typeinfo"), ("_ZTS", "typeinfo-name"))


def _resolve_rtti_symbols(ctx, bv, query: str, table_entries: int) -> list[dict[str, Any]]:
    """Resolve a (mangled) type-name to its RTTI DATA symbols -- the vtable /
    typeinfo / typeinfo-name objects that actually carry the metadata, in
    .rodata/.data.rel.ro with real xrefs -- which is what the lens is meant to
    find but a .dynstr name-string match never reaches (#194). Best-effort: only
    fires when the query is the mangled fragment (`_ZTV`+query resolves)."""
    out: list[dict[str, Any]] = []
    q = query.strip()
    if not q:
        return out
    for prefix, kind in _RTTI_PREFIXES:
        sym = _symbol_by_any_name(bv, prefix + q)
        if sym is None or getattr(sym, "address", None) is None:
            continue
        addr = int(sym.address)
        entry: dict[str, Any] = {
            "kind": kind,
            "symbol": prefix + q,
            "address": hex(addr),
            "xrefs": read_xrefs._xrefs_to_address(ctx, bv, addr),
        }
        # The vtable's slots (typeinfo pointer + virtual methods) are the payload;
        # show the table window so the lens directly surfaces them.
        if kind == "vtable" and table_entries:
            entry["table_window"] = _pointer_table_for_view(
                ctx, bv, addr, entries=table_entries,
                stride_size=ctx._pointer_size(bv), stop_after_invalid=2,
            )
        out.append(entry)
    return out


def _message_lens(ctx, selector: str | None, query: str, *, limit: int = 20, table_entries: int = 6):
    limit = _validate_count(limit, label="limit", minimum=1)
    table_entries = _validate_count(table_entries, label="table_entries", minimum=0)
    bv = ctx._resolve_view(selector)
    needle = query.lower()
    matches = []
    total_matched = 0
    dynstr_excluded = 0
    for item in list(getattr(bv, "strings", [])):
        value = str(getattr(item, "value", ""))
        if needle and needle not in value.lower():
            continue
        address = int(getattr(item, "start", 0))
        context = ctx._address_context(bv, address)
        # `.dynstr` matches are mangled SYMBOL-NAME strings, never the RTTI
        # metadata this lens targets; on a symbol-retaining binary they drown the
        # real result in 0-xref noise. Exclude them (a stripped binary has no
        # .dynstr, so this is safe there too) -- count for an honest total + hint
        # (#194).
        if ".dynstr" in _section_names_at(context):
            dynstr_excluded += 1
            continue
        # Count every (non-.dynstr) match so the reported total is honest, but
        # only build the expensive per-match evidence for the first `limit`.
        total_matched += 1
        if len(matches) >= limit:
            continue
        xrefs = read_xrefs._xrefs_to_address(ctx, bv, address)
        metadata_tables = []
        for ref in list(xrefs.get("data_refs") or [])[:3]:
            try:
                ref_addr = _parse_address(ref["address"])
            except Exception:
                continue
            start = max(0, ref_addr - ctx._pointer_size(bv) * 2)
            metadata_tables.append(
                _pointer_table_for_view(
                    ctx,
                    bv,
                    start,
                    entries=table_entries,
                    stride_size=ctx._pointer_size(bv),
                    stop_after_invalid=1,
                )
            )

        matches.append(
            {
                "type_string": {
                    "address": hex(address),
                    "value": value,
                    "length": int(getattr(item, "length", len(value))),
                    "context": context,
                },
                "xrefs": xrefs,
                "metadata_table_windows": metadata_tables,
            }
        )

    # Surface the real RTTI metadata directly (vtable/typeinfo/typeinfo-name data
    # symbols), the structures a .dynstr name match can never reach (#194).
    rtti_symbols = _resolve_rtti_symbols(ctx, bv, query, table_entries)

    hints: list[str] = []
    if dynstr_excluded:
        hints.append(
            f"excluded {dynstr_excluded} match(es) in .dynstr (mangled symbol-name "
            f"strings, never RTTI metadata). This binary retains its symbol table; "
            f"resolve the _ZTV/_ZTI/_ZTS<type> data symbols directly, or run "
            f"`bn evidence table <vtable-addr>`."
        )
    if rtti_symbols:
        hints.append(
            f"resolved {len(rtti_symbols)} RTTI data symbol(s) for the type "
            f"(vtable/typeinfo/typeinfo-name) -- see rtti_symbols."
        )

    return {
        "kind": "messages",
        "query": query,
        "items": matches,
        "count": len(matches),
        "total": total_matched,
        "truncated": total_matched > len(matches),
        "dynstr_excluded": dynstr_excluded,
        "rtti_symbols": rtti_symbols,
        "hints": hints,
    }


_INIT_SECTION_HINTS = (
    "init_array",
    "preinit_array",
    "fini_array",
    ".ctors",
    ".dtors",
    "__mod_init_func",
    "__mod_term_func",
)


def _init_arrays(ctx, selector: str | None, *, limit: int = 64):
    if limit < 0:
        raise OperationFailure("invalid_limit", f"Invalid init-array limit: {limit}")
    bv = ctx._resolve_view(selector)
    pointer_size = ctx._pointer_size(bv)
    sections = []
    for name, sec in getattr(bv, "sections", {}).items():
        lowered = str(name).lower()
        if not any(hint in lowered for hint in _INIT_SECTION_HINTS):
            continue
        start = int(getattr(sec, "start", 0))
        end = int(getattr(sec, "end", 0))
        total_entries = max(0, (end - start) // pointer_size)
        shown_entries = min(total_entries, limit)
        table = _pointer_table_for_view(
            ctx,
            bv,
            start,
            entries=shown_entries,
            stride_size=pointer_size,
        )
        sections.append(
            {
                "name": str(name),
                "start": hex(start),
                "end": hex(end),
                "total_entries": total_entries,
                "shown_entries": shown_entries,
                "truncated": total_entries > shown_entries,
                "table": table,
            }
        )
    sections.sort(key=lambda item: int(item["start"], 16))
    # #275: `items` are the init/ctor sections (each retains its nested `entries`
    # table); `kind` discriminates the envelope.
    return {
        "kind": "init_arrays",
        "pointer_size": pointer_size,
        "items": sections,
        "total": len(sections),
    }
