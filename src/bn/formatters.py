from __future__ import annotations

import json
from typing import Any, Callable

# "rollback_failed" = an op succeeded but the batch revert that should have
# undone it failed, so the view may be left modified -- a real failure. A
# cleanly rolled-back sibling ("reverted") is NOT a failure and is omitted (#118).
# "internal_error" = an unexpected engine bug (distinct from an unsupported
# request); still a failure, so exit codes/rendering flag it (#122).
FAILED_MUTATION_STATUSES = {"unsupported", "verification_failed", "invalid_request", "rollback_failed", "internal_error"}


def _render_fallback_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True)


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce a nested field to a dict for safe ``.get()`` chains.

    A renderer that does ``value.get("function") or {}`` still crashes when the
    field is present but a NON-dict (a string/list from a malformed or future
    bridge result), because the non-dict is truthy and reaches ``.get()``. This
    returns ``{}`` for anything that isn't a dict so the renderer degrades to
    placeholder text instead of an AttributeError (#101)."""
    return value if isinstance(value, dict) else {}


def _render_string_literal(value: Any, *, truncated: bool = False) -> str:
    text = json.dumps(value, ensure_ascii=True)
    if truncated:
        text += " [truncated]"
    return text


def _format_local_entry(item: dict[str, Any]) -> str:
    name = str(item.get("name", "<unknown>"))
    type_str = str(item.get("type", "<unknown>"))
    line = f"  {name:<20} {type_str}"
    # local_id is the stable handle `local rename` / `local retype` take; show it
    # so the text view is self-sufficient and doesn't force a --format json
    # round-trip just to drive those commands (#122). Other internal fields
    # (storage / source / identifier) stay out of the slim view.
    local_id = item.get("local_id")
    if local_id:
        line += f"  [id: {local_id}]"
    return line


def _text_field(field: str) -> Callable[[Any], str]:
    def render(value: Any) -> str:
        if isinstance(value, dict):
            text = value.get(field)
            if isinstance(text, str):
                return text
        return _render_fallback_text(value)

    return render


def _slice_text_lines(
    text: str, lines_range: tuple[int, int] | None, *, marker: str = "//"
) -> str:
    """Return only lines START..END (1-indexed, inclusive) with a count header.

    Shared by `decompile`, `il`, and `disasm` so every line-oriented view slices
    the same way. Slicing happens before the spill check, so `--lines` also keeps
    large functions inline.
    """
    if lines_range is None:
        return text
    all_lines = text.splitlines()
    total = len(all_lines)
    start, end = lines_range
    if start > total:
        return f"{marker} lines 0 of {total} (start {start} is beyond the last line)"
    sliced = all_lines[start - 1 : end]
    header = f"{marker} lines {start}-{min(end, total)} of {total}"
    return header + "\n" + "\n".join(sliced)


def _render_function_info_text(value: Any, verbose: bool = False) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    function = _as_dict(value.get("function"))
    lines = [
        f"{function.get('name', '<unknown>')} @ {function.get('address', '<unknown>')}",
        str(value.get("prototype", "")),
        f"calling convention: {value.get('calling_convention', '<unknown>')}",
        f"size: {value.get('size', '<unknown>')}",
        f"xrefs: {value.get('xref_count', 0)}",
    ]

    locals_only = list(value.get("locals") or [])
    if locals_only:
        lines.append(f"locals: {len(locals_only)} variables")

    if verbose:
        parameters = list(value.get("parameters") or [])
        if parameters:
            lines.append("")
            lines.append("parameters:")
            for item in parameters:
                lines.append(_format_local_entry(item))
        lines.append("")
        if locals_only:
            lines.append("locals:")
            for item in locals_only:
                lines.append(_format_local_entry(item))
        else:
            lines.append("locals: none")

    return "\n".join(lines)


def _render_proto_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    prototype = value.get("prototype")
    if isinstance(prototype, str):
        return prototype
    return _render_fallback_text(value)


def _render_local_list_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    function = value.get("function") or {}
    all_items = list(value.get("locals") or [])
    params = [item for item in all_items if item.get("is_parameter")]
    locals_only = [item for item in all_items if not item.get("is_parameter")]

    header = f"{function.get('name', '<unknown>')} @ {function.get('address', '<unknown>')}"
    header += f" ({len(params)} params, {len(locals_only)} locals)"
    lines = [header]

    if params:
        lines.extend(["", "params:"])
        for item in params:
            lines.append(_format_local_entry(item))
    if locals_only:
        lines.extend(["", "locals:"])
        for item in locals_only:
            lines.append(_format_local_entry(item))
    if not params and not locals_only:
        lines.extend(["", "no locals"])
    return "\n".join(lines)


def _render_type_info_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    layout = value.get("layout")
    if isinstance(layout, str) and layout:
        return layout
    decl = value.get("decl")
    if isinstance(decl, str) and decl:
        return decl
    return _render_fallback_text(value)


def _render_field_xrefs_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    field = value.get("field") or {}
    lines = [
        f"{field.get('type_name', '<unknown>')}.{field.get('field_name', '<unknown>')} @ +0x{int(field.get('offset', 0)):x}",
        f"type: {field.get('field_type', '<unknown>')}",
        "",
        "code refs:",
    ]
    code_refs = list(value.get("code_refs") or [])
    if code_refs:
        for ref in code_refs:
            details = [ref.get("address", "<unknown>")]
            if ref.get("function"):
                details.append(ref["function"])
            if ref.get("incoming_type"):
                details.append(f"type={ref['incoming_type']}")
            if ref.get("disasm"):
                details.append(ref["disasm"])
            lines.append("- " + " | ".join(details))
    else:
        lines.append("- none")

    lines.extend(["", "data refs:"])
    data_refs = list(value.get("data_refs") or [])
    if data_refs:
        for ref in data_refs:
            details = [ref.get("address", "<unknown>")]
            if ref.get("symbol"):
                details.append(ref["symbol"])
            if ref.get("type"):
                details.append(f"type={ref['type']}")
            lines.append("- " + " | ".join(details))
    else:
        lines.append("- none")

    return "\n".join(lines)


def _render_comment_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    comment = value.get("comment")
    if isinstance(comment, str):
        return comment if comment else "(no comment)"
    return _render_fallback_text(value)


def _render_comment_list_text(value: Any) -> str:
    # Paged envelope ({items,total,...}) -> render the page + the shared footer;
    # a bare list falls through to the per-item body below (back-compat) (#131).
    if isinstance(value, dict) and "items" in value:
        return _render_paged_list_text(value, "items", _render_comment_list_text)
    if not isinstance(value, list):
        return _render_fallback_text(value)
    if not value:
        return "none"
    lines = []
    for item in value:
        if not isinstance(item, dict):
            lines.append(_render_fallback_text(item))
            continue
        address = item.get("address", "<unknown>")
        func = item.get("function") or "<global>"
        comment = item.get("comment", "")
        lines.append(f"{address}  {func}  {comment}")
    return "\n".join(lines)


def _render_refresh_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    target = value.get("target")
    if isinstance(target, dict):
        return f"refreshed: true\n\n{_render_target_summary(target)}"
    return _render_fallback_text(value)


def _render_load_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    suffix = "  [not analyzed]" if value.get("analyzed") is False else ""
    lines = [f"loaded: {value.get('path', '<unknown>')}{suffix}"]
    for note in value.get("notes") or []:
        lines.append(f"note: {note}")
    targets = list(value.get("targets") or [])
    if targets:
        lines.append("")
        lines.append("targets:")
        for t in targets:
            if isinstance(t, dict):
                lines.append("- " + (t.get("selector") or t.get("basename") or "<unknown>"))
            else:
                lines.append("- " + _render_fallback_text(t))
    return "\n".join(lines)


def _render_close_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    closed = list(value.get("closed") or [])
    if not closed:
        return "no binaries closed"

    def _row(entry: Any) -> tuple[str, bool]:
        if isinstance(entry, dict):
            return str(entry.get("path", "")), bool(entry.get("unsaved"))
        return str(entry), False

    rows = [_row(e) for e in closed]
    unsaved_any = any(unsaved for _, unsaved in rows)

    if len(rows) == 1:
        path, unsaved = rows[0]
        lines = [f"closed: {path}"]
    else:
        lines = ["closed:"]
        for path, unsaved in rows:
            marker = "  [unsaved changes discarded]" if unsaved else ""
            lines.append(f"- {path}{marker}")

    if unsaved_any:
        lines.append("")
        lines.append(
            "warning: unsaved mutations were discarded. "
            "use `bn save` before `bn close` to persist them."
        )
    return "\n".join(lines)


def _render_save_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    return f"saved: {value.get('path', '<unknown>')}"


def _render_session_start_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    lines = [
        f"instance: {value.get('instance_id', '<unknown>')}",
        f"pid: {value.get('pid', '<unknown>')}",
        f"socket: {value.get('socket_path', '<unknown>')}",
    ]
    loaded = list(value.get("loaded") or [])
    if loaded:
        lines.append("")
        lines.append("loaded:")
        for item in loaded:
            if isinstance(item, dict):
                error = item.get("error")
                if error:
                    lines.append(f"- {item.get('path', '<unknown>')} [error: {error}]")
                else:
                    mark = "  [not analyzed]" if item.get("analyzed") is False else ""
                    lines.append(f"- {item.get('path', '<unknown>')}{mark}")
                for note in item.get("notes") or []:
                    lines.append(f"  note: {note}")
            else:
                lines.append(f"- {_render_fallback_text(item)}")
    if value.get("stopped"):
        lines.append("")
        lines.append("session stopped: no binaries loaded successfully")
    return "\n".join(lines)


def _render_session_stop_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    line = f"stopped: {value.get('instance_id', '<unknown>')}"
    method = value.get("method")
    if method:
        line += f" ({method})"
    return line


def _render_session_list_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    instances = list(value.get("instances") or [])
    if not instances:
        return "no sessions"
    lines = []
    for item in instances:
        if not isinstance(item, dict):
            lines.append(_render_fallback_text(item))
            continue
        head = str(item.get("selector") or item.get("instance_id") or "<unknown>")
        if item.get("sticky"):
            head += " [sticky]"
        parts = [head, f"pid={item.get('pid', '<unknown>')}"]
        rss = item.get("rss_mb")
        if rss is not None:
            parts.append(f"rss={rss}MB")
        if item.get("started_at"):
            parts.append(f"started={item['started_at']}")
        lines.append("  ".join(parts))
        if item.get("socket_path"):
            lines.append(f"  socket: {item['socket_path']}")
    total_rss = value.get("total_rss_mb")
    if total_rss is not None and instances:
        lines.append("")
        lines.append(f"total rss: {total_rss}MB")
    return "\n".join(lines)


def _render_target_summary(value: dict[str, Any]) -> str:
    view_id = value.get("view_id")
    label = value.get("selector") or value.get("target_id") or "<unknown>"
    prefix = f"[{view_id}] " if view_id is not None else ""
    lines = [f"{prefix}{label}"]
    if value.get("active"):
        lines[0] += " [active]"
    if value.get("sticky"):
        lines[0] += " [sticky]"
    # Flag a --quick (unanalyzed) view in text, not just JSON, so a cold agent
    # doesn't trust an empty/partial result from a view whose analysis is pending.
    if value.get("analyzed") is False:
        lines[0] += " [not analyzed]"

    details = [
        ("target", value.get("target_id")),
        ("view", value.get("view_id")),
        ("kind", value.get("view_name")),
        ("file", value.get("filename")),
        ("arch", value.get("arch")),
        ("platform", value.get("platform")),
        ("entry", value.get("entry_point")),
    ]
    for key, item in details:
        if item not in (None, ""):
            lines.append(f"\t{key}: {item}")
    # Function-count + named-vs-auto-named summary that every agent reaches for
    # first (#122). Counts reflect the current analysis state (a --quick view
    # reports what it has so far; analysis_state already flags that).
    fn_count = value.get("function_count")
    if fn_count is not None:
        summary = f"{fn_count} functions"
        named = value.get("named_function_count")
        unnamed = value.get("unnamed_function_count")
        if named is not None and unnamed is not None:
            parts = [f"{named} named", f"{unnamed} auto-named"]
            imported = value.get("imported_function_count")
            if imported:
                parts.append(f"{imported} imported")
            summary += f" ({', '.join(parts)})"
        lines.append(f"\tfunctions: {summary}")
    # Segment-level detail only when present -- target info --verbose adds it;
    # target list rows do not, so this stays out of the list view. (F21)
    segments = value.get("segments")
    if isinstance(segments, list) and segments:
        lines.append("\tsegments:")
        for seg in segments:
            perms = "".join(
                flag if seg.get(name) else "-"
                for name, flag in (("readable", "r"), ("writable", "w"), ("executable", "x"))
            )
            lines.append(
                f"\t\t{seg.get('start')}-{seg.get('end')} {perms} ({seg.get('length')} bytes)"
            )
    return "\n".join(lines)


def _render_target_list_text(value: Any) -> str:
    if not isinstance(value, list):
        return _render_fallback_text(value)
    if not value:
        return "no targets"
    return "\n\n".join(
        _render_target_summary(item) if isinstance(item, dict) else _render_fallback_text(item)
        for item in value
    )


def _render_target_info_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    return _render_target_summary(value)


def _render_target_choice(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    label = str(value.get("selector") or value.get("target_id") or "<unknown>")
    if value.get("active"):
        label += " [active]"

    target_id = value.get("target_id")
    if target_id not in (None, "", value.get("selector")):
        label += f" (target_id: {target_id})"
    return label


def _render_target_choices(value: Any) -> str:
    if not isinstance(value, list):
        return _render_fallback_text(value)
    if not value:
        return "none"
    return "\n".join(f"- {_render_target_choice(item)}" for item in value)


def _render_instance_use_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    return f"instance: {value.get('instance_id', '<unknown>')}"


def _render_target_use_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    return f"target: {value.get('target', '<unknown>')}"


def _render_pin_clear_text(value: Any) -> str:
    """Render `instance clear` / `target clear` confirmations."""
    return "cleared"


def _render_name_address_rows(value: Any) -> str:
    """Render a BARE list of name/address rows (imports, function pages)."""
    if not isinstance(value, list):
        return _render_fallback_text(value)
    if not value:
        return "none"

    lines = []
    for item in value:
        if not isinstance(item, dict):
            lines.append(_render_fallback_text(item))
            continue
        address = item.get("address", "<unknown>")
        name = item.get("name") or item.get("function") or "<unknown>"
        line = f"{address}  {name}"
        kind = item.get("kind")
        if kind and kind != "function":
            line += f" ({kind})"
        library = item.get("library")
        if library:
            line += f" [{library}]"
        raw_name = item.get("raw_name")
        if raw_name and raw_name != name:
            line += f" (raw: {raw_name})"
        size = item.get("size")
        if size is not None:
            line += f"  ({size} bytes)"
        lines.append(line)
    return "\n".join(lines)


def _render_name_address_list_text(value: Any) -> str:
    """Render imports: the paged {items, total, ...} envelope (with a footer),
    or a bare list for back-compat / internal callers (#122)."""
    return _render_paged_list_text(value, "items", _render_name_address_rows)


def _paging_footer(value: dict[str, Any], items: list[Any]) -> str | None:
    """Build the "// showing N of TOTAL" footer for a paged-list envelope.

    Shared by every paged list renderer (function list/search, strings, imports,
    sections) so the honest-total convention reads identically across them (#59,
    #122). Returns None when the page IS the whole set (no paging happened) or
    the envelope lacks an integer total to report against."""
    total = value.get("total")
    returned = value.get("returned", len(items) if isinstance(items, list) else 0)
    offset = value.get("offset", 0) or 0
    if not isinstance(total, int):
        return None
    if value.get("has_more"):
        remaining = total - (offset + returned)
        next_offset = offset + returned
        return (
            f"// showing {returned} of {total} ({remaining} more); "
            f"rerun with --offset {next_offset} or a larger --limit"
        )
    if offset or returned != total:
        return f"// showing {returned} of {total}"
    return None


def _render_paged_list_text(
    value: Any, page_key: str, item_renderer: Callable[[Any], str]
) -> str:
    """Render a paged-list envelope ({<page_key>, total, ...}) with a footer.

    Delegates the body to *item_renderer* (which renders a bare list) and
    appends the shared paging footer. Falls back to rendering *value* as a bare
    list when it isn't an envelope, so internal callers or older bridges that
    still hand over a plain list keep working (#122)."""
    if not isinstance(value, dict) or page_key not in value:
        return item_renderer(value)  # back-compat / fallback for a bare list
    items = value.get(page_key) or []
    body = item_renderer(items)
    footer = _paging_footer(value, items)
    if footer is None:
        return body
    return footer if body == "none" else f"{body}\n\n{footer}"


def _render_function_list_text(value: Any) -> str:
    """Render a paged function listing (the {functions, total, ...} envelope),
    with a footer stating the true total and remainder (#59)."""
    return _render_paged_list_text(value, "functions", _render_name_address_rows)


def _group_refs_by_caller(refs: list[Any]) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[str, Any]] = {}
    order: list[tuple] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        caller = ref.get("caller_function") if isinstance(ref.get("caller_function"), dict) else None
        key: tuple
        if caller is not None:
            key = ("fn", caller.get("address"), caller.get("name"))
            caller_address = caller.get("address")
            caller_name = caller.get("name")
        else:
            # No containing function: group by the ref's own resolved label
            # (symbol, else section) so refs under DIFFERENT symbols/sections do
            # not collapse into one group that gets stamped with the first
            # ref's label (which mislabeled the others). A label-less ref is
            # never coalesced -- it keys on its own address so each renders as a
            # distinct "<unknown>" line with its own context.
            label = _unknown_ref_label(ref.get("context")) or ref.get("function")
            if label:
                key = ("label", label)
            else:
                key = ("site", str(ref.get("address", "<unknown>")))
            caller_address = None
            caller_name = label or None
        if key not in groups:
            groups[key] = {
                "caller_address": caller_address,
                "caller_name": caller_name,
                "sites": [],
                "context": ref.get("context"),
            }
            order.append(key)
        groups[key]["sites"].append(str(ref.get("address", "<unknown>")))
    return [groups[k] for k in order]


def _unknown_ref_label(context: Any) -> str:
    """A concise fallback label (symbol name, else section name) for a ref that
    has no containing function, so it isn't rendered as a bare "<unknown>"."""
    if not isinstance(context, dict):
        return ""
    symbol = context.get("symbol")
    if isinstance(symbol, dict) and symbol.get("name"):
        return str(symbol["name"])
    sections = context.get("sections")
    if isinstance(sections, list):
        for item in sections:
            if isinstance(item, dict) and item.get("name"):
                return str(item["name"])
    return ""


def _render_xrefs_text(value: Any, limit: int | None = None) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    code_refs = list(value.get("code_refs") or [])
    data_refs = list(value.get("data_refs") or [])
    total_code = len(code_refs)
    total_data = len(data_refs)

    import_label = ""
    if value.get("import_resolved"):
        scanned = " (scanned)" if value.get("code_refs_scanned") else ""
        import_label = f"import: {value.get('import_name', '<unknown>')}{scanned}"

    def _render_group(refs: list[Any], total: int, label: str) -> list[str]:
        groups = _group_refs_by_caller(refs)
        if not groups:
            return [f"{label}:", "- none"]
        site_word = "site" if total == 1 else "sites"
        # The groups are distinct CALLERS, which may be functions OR function-less
        # locations (a data ref whose caller_function is null). Only call them
        # "functions" when every group really is one; otherwise use the neutral
        # "locations" so a function-less bucket isn't miscounted as a function (#49).
        # A function group carries a caller_address; a function-less one has None
        # (its caller_name holds a fallback symbol/section label, so that field
        # can't be the discriminator).
        all_functions = all(group.get("caller_address") for group in groups)
        grp_singular = "function" if all_functions else "location"
        grp_plural = "functions" if all_functions else "locations"
        grp_word = grp_singular if len(groups) == 1 else grp_plural
        header = f"{label}: {total} {site_word} across {len(groups)} {grp_word}"
        shown = groups[:limit] if limit else groups
        rendered = [header]
        for group in shown:
            caller_addr = group["caller_address"] or "<unknown>"
            caller_name = (
                group["caller_name"]
                or _unknown_ref_label(group.get("context"))
                or "<unknown>"
            )
            sites = group["sites"]
            if len(sites) == 1:
                suffix = f"(1 site: {sites[0]})"
            else:
                suffix = f"({len(sites)} sites: {', '.join(sites)})"
            rendered.append(f"  {caller_addr}  {caller_name}  {suffix}")
        if limit and len(groups) > limit:
            rendered.append(
                f"  ... {len(groups) - limit} more {grp_plural} "
                "(increase --limit or use --format json for all)"
            )
        return rendered

    lines = [f"xrefs to {value.get('address', '<unknown>')} ({total_code} code, {total_data} data)", ""]
    if import_label:
        lines.insert(0, import_label)
        lines.insert(1, "")
    lines.extend(_render_group(code_refs, total_code, "code refs"))
    lines.append("")
    lines.extend(_render_group(data_refs, total_data, "data refs"))
    return "\n".join(lines)


def _context_suffix(context: Any) -> str:
    if not isinstance(context, dict):
        return ""
    parts = []
    sections = context.get("sections")
    if isinstance(sections, list) and sections:
        names = [str(item.get("name", "")) for item in sections if isinstance(item, dict) and item.get("name")]
        if names:
            parts.append("section=" + ",".join(names))
    segment = context.get("segment")
    if isinstance(segment, dict):
        perms = (
            ("r" if segment.get("readable") else "-")
            + ("w" if segment.get("writable") else "-")
            + ("x" if segment.get("executable") else "-")
        )
        parts.append(f"seg={perms}")
    symbol = context.get("symbol")
    if isinstance(symbol, dict) and symbol.get("name"):
        sym_type = symbol.get("type")
        if sym_type:
            parts.append(f"symbol={symbol['name']}[{sym_type}]")
        else:
            parts.append(f"symbol={symbol['name']}")
    string = context.get("string")
    if isinstance(string, dict) and string.get("value"):
        enc = string.get("encoding")
        label = "string" if enc in (None, "ascii") else f"string({enc})"
        parts.append(
            f"{label}={_render_string_literal(string['value'], truncated=bool(string.get('truncated')))}"
        )
    disasm = context.get("disasm")
    if isinstance(disasm, str) and disasm:
        parts.append(f"disasm={disasm}")
    return " | " + " | ".join(parts) if parts else ""


def _render_evidence_xrefs_text(value: Any, limit: int | None = None) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    lines = [f"xrefs to {value.get('address', '<unknown>')}"]
    target_context = value.get("target_context")
    suffix = _context_suffix(target_context)
    if suffix:
        lines.append(f"target{suffix}")

    for label in ("code_refs", "data_refs"):
        refs = list(value.get(label) or [])
        total = len(refs)
        shown = refs[:limit] if limit else refs
        nice = label.replace("_", " ")
        lines.append("")
        # Report the true total and a truncation marker when capped, matching
        # the honesty convention used by strings/function list/evidence message
        # -- a bare cap with no "N more" would hide that refs exist (#31).
        if limit and total > len(shown):
            lines.append(f"{nice}: {total} total, showing first {len(shown)}")
        else:
            lines.append(f"{nice}:")
        if not shown:
            lines.append("- none")
            continue
        for ref in shown:
            if not isinstance(ref, dict):
                lines.append("- " + _render_fallback_text(ref))
                continue
            address = ref.get("address", "<unknown>")
            function = ref.get("function") or "<unknown>"
            kind = ref.get("kind") or label.removesuffix("_refs")
            lines.append(
                f"- {address}  {kind}  {function}{_context_suffix(ref.get('context'))}"
            )
    return "\n".join(lines)


def _render_target_line(target: Any) -> str:
    if not isinstance(target, dict):
        return "<unknown>"
    if target.get("status") == "unmapped":
        raw = target.get("raw") or "<unknown>"
        return f"{raw} [unmapped/non-pointer]"
    if target.get("status") == "null":
        return "0x0 [null]"
    raw = target.get("raw")
    normalized = target.get("normalized")
    fn = target.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        fn_address = fn.get("address", normalized or raw or "<unknown>")
        base = f"{fn.get('name')} @ {fn_address}"
        if fn.get("exact_start") is False:
            offset = fn.get("offset")
            if offset:
                base += str(offset) if str(offset).startswith("-") else f"+{offset}"
            actual = normalized or raw
            if actual and actual != fn_address:
                base += f" (target {actual}, not start)"
            else:
                base += " (not start)"
    else:
        addr = normalized or raw or "<unknown>"
        context = target.get("context") if isinstance(target.get("context"), dict) else {}
        symbol = context.get("symbol")
        string = context.get("string")
        sections = context.get("sections")
        section_name = None
        if isinstance(sections, list) and sections and isinstance(sections[0], dict):
            section_name = sections[0].get("name")
        if isinstance(symbol, dict) and symbol.get("name"):
            base = f"{symbol['name']} @ {addr}"
            annot = [a for a in (section_name, symbol.get("type")) if a]
            if annot:
                base += f" [{', '.join(annot)}]"
        elif isinstance(string, dict) and string.get("value"):
            enc = string.get("encoding")
            base = json.dumps(string["value"], ensure_ascii=True)
            annot = [
                a
                for a in (
                    section_name,
                    (enc if enc and enc != "ascii" else None),
                    ("truncated" if string.get("truncated") else None),
                )
                if a
            ]
            if annot:
                base += f" [{', '.join(annot)}]"
        else:
            base = str(addr)
    if raw and normalized and raw != normalized:
        base += f" (raw {raw})"
    if target.get("thumb_adjusted"):
        base += " [thumb-adjusted]"
    if target.get("plausible") is False:
        base += " [low-confidence]"
    return base


def _render_function_evidence_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    function = value.get("function") if isinstance(value.get("function"), dict) else {}
    lines = [
        f"{function.get('name', '<unknown>')} @ {function.get('address', '<unknown>')}",
        f"prototype: {value.get('prototype', '<unknown>')}",
        f"calling convention: {value.get('calling_convention', '<unknown>')}",
    ]
    thunk = value.get("thunk") if isinstance(value.get("thunk"), dict) else {}
    if thunk.get("is_candidate"):
        lines.append(f"thunk: candidate ({thunk.get('reason', 'no reason recorded')})")
        if thunk.get("target"):
            lines.append(f"  target: {_render_target_line(thunk['target'])}")
    else:
        lines.append("thunk: no")

    calls = list(value.get("calls") or [])
    lines.append("")
    lines.append(f"calls: {len(calls)}")
    if not calls:
        return "\n".join(lines)

    for call in calls:
        if not isinstance(call, dict):
            lines.append("- " + _render_fallback_text(call))
            continue
        call_addr = call.get("address", "<unknown>")
        operation = call.get("operation", "<unknown>")
        direct = "direct" if call.get("direct") else "indirect"
        lines.append(f"- {call_addr}  {operation}  {direct}")
        target = call.get("target")
        if target:
            lines.append(f"  target: {_render_target_line(target)}")
        if call.get("call_instruction"):
            instr = call["call_instruction"]
            if isinstance(instr, dict):
                lines.append(f"  instruction: {instr.get('address', call_addr)}  {instr.get('text', '')}".rstrip())
        if call.get("hlil_statement"):
            lines.append(f"  hlil: {call['hlil_statement']}")
        if call.get("mlil"):
            lines.append(f"  mlil: {call['mlil']}")
        if call.get("llil"):
            lines.append(f"  llil: {call['llil']}")
        args = [arg for arg in list(call.get("arguments") or []) if isinstance(arg, dict)]
        if args:
            source = call.get("argument_source")
            lines.append("  arguments:" + (f" ({source})" if source else ""))
            for arg in args:
                lines.append(f"    {arg.get('text', '')}{_render_resolved_arg(arg.get('resolved'))}")
    return "\n".join(lines)


def _render_resolved_arg(resolved: Any) -> str:
    if not isinstance(resolved, dict):
        return ""
    section = resolved.get("section")
    suffix = f" [{section}]" if section else ""
    if resolved.get("string") is not None:
        value = json.dumps(resolved["string"], ensure_ascii=True)
        encoding = resolved.get("encoding")
        if encoding:
            value += f"({encoding})"
        if resolved.get("truncated"):
            value += " [truncated]"
        return f" -> {value}{suffix}"
    if resolved.get("symbol"):
        return f" -> {resolved['symbol']}{suffix}"
    if resolved.get("function"):
        return f" -> {resolved['function']}{suffix}"
    return ""


def _render_pointer_table_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    lines = [
        f"pointer table @ {value.get('address', '<unknown>')}",
        f"pointer-size: {value.get('pointer_size', '<unknown>')}  stride: {value.get('stride', '<unknown>')}",
    ]
    suffix = _context_suffix(value.get("context"))
    if suffix:
        lines.append(f"context{suffix}")
    for warning in list(value.get("warnings") or []):
        lines.append(f"warning: {warning}")
    lines.append("")
    for item in list(value.get("entries") or []):
        if not isinstance(item, dict):
            lines.append(_render_fallback_text(item))
            continue
        prefix = f"[{item.get('index', '?'):>2}] {item.get('entry_address', '<unknown>')}"
        if not item.get("readable", True):
            lines.append(f"{prefix}  <unreadable>")
            continue
        plausibility = "" if item.get("plausible", True) else "  [implausible]"
        lines.append(f"{prefix}  {item.get('value', '<unknown>')} -> {_render_target_line(item.get('target'))}{plausibility}")
    return "\n".join(lines)


def _render_message_lens_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    shown = value.get("count", 0)
    total = value.get("total", shown)
    header = f"message lens: {value.get('query', '<unknown>')} ({total} matches"
    if value.get("truncated"):
        header += f"; showing first {shown}, increase --limit for the rest"
    header += ")"
    lines = [header]
    for match in list(value.get("matches") or []):
        if not isinstance(match, dict):
            lines.append(_render_fallback_text(match))
            continue
        type_string = match.get("type_string") if isinstance(match.get("type_string"), dict) else {}
        lines.append("")
        lines.append(f"{type_string.get('address', '<unknown>')}  {json.dumps(type_string.get('value', ''), ensure_ascii=True)}")
        suffix = _context_suffix(type_string.get("context"))
        if suffix:
            lines.append(f"  context{suffix}")
        xrefs = match.get("xrefs") if isinstance(match.get("xrefs"), dict) else {}
        code_count = len(list(xrefs.get("code_refs") or []))
        data_count = len(list(xrefs.get("data_refs") or []))
        lines.append(f"  xrefs: {code_count} code, {data_count} data")
        for ref in list(xrefs.get("code_refs") or [])[:3]:
            if isinstance(ref, dict):
                lines.append(f"    code {ref.get('address', '<unknown>')}  {ref.get('function') or '<unknown>'}{_context_suffix(ref.get('context'))}")
        for ref in list(xrefs.get("data_refs") or [])[:3]:
            if isinstance(ref, dict):
                lines.append(f"    data {ref.get('address', '<unknown>')}{_context_suffix(ref.get('context'))}")
        table_windows = list(match.get("metadata_table_windows") or [])
        if table_windows:
            lines.append(f"  metadata table windows: {len(table_windows)}")
            for table in table_windows[:2]:
                if isinstance(table, dict):
                    lines.append(f"    table @ {table.get('address', '<unknown>')}")
                    for warning in list(table.get("warnings") or [])[:2]:
                        lines.append(f"      warning: {warning}")
    return "\n".join(lines)


def _render_init_arrays_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    sections = list(value.get("sections") or [])
    if not sections:
        return "init arrays: none"
    lines = [f"init arrays: {len(sections)} section(s), pointer-size={value.get('pointer_size', '<unknown>')}"]
    for section in sections:
        if not isinstance(section, dict):
            lines.append(_render_fallback_text(section))
            continue
        lines.append("")
        lines.append(
            f"{section.get('name', '<unknown>')} "
            f"{section.get('start', '<unknown>')}-{section.get('end', '<unknown>')} "
            f"entries={section.get('total_entries', '?')}"
        )
        if section.get("truncated"):
            lines.append(f"  showing first {section.get('shown_entries', '?')} entries")
        table = section.get("table") if isinstance(section.get("table"), dict) else {}
        for warning in list(table.get("warnings") or []):
            lines.append(f"  warning: {warning}")
        for item in list(table.get("entries") or []):
            if not isinstance(item, dict):
                continue
            prefix = f"  [{item.get('index', '?'):>2}] {item.get('entry_address', '<unknown>')}"
            if not item.get("readable", True):
                lines.append(f"{prefix}  <unreadable>")
                continue
            lines.append(f"{prefix}  {item.get('value', '<unknown>')} -> {_render_target_line(item.get('target'))}")
    return "\n".join(lines)


def _render_callsites_text(value: Any, *, prefer_caller_static: bool = False) -> str:
    # callsites now returns the {items,total,...} envelope (#131 / item 11);
    # unwrap to the row list. has_more is always false (no bridge-side paging),
    # so no footer -- the --limit cap below stays a text-only renderer feature.
    if isinstance(value, dict) and "items" in value:
        value = value.get("items") or []
    if not isinstance(value, list):
        return _render_fallback_text(value)
    if not value:
        return "no callsites found"

    blocks = []
    for row in value:
        if not isinstance(row, dict):
            blocks.append(_render_fallback_text(row))
            continue

        callee = row.get("callee") if isinstance(row.get("callee"), dict) else {}
        containing = row.get("containing_function") if isinstance(row.get("containing_function"), dict) else {}
        call_addr = row.get("call_addr", "<unknown>")
        caller_static = row.get("caller_static", "<unknown>")
        call_index = row.get("call_index")
        # A tailcall (tail-branch into the target) has no real return site, so flag
        # it -- its caller_static is the byte after the branch, not a return addr (#47).
        kind_tag = "  [tailcall]" if row.get("call_kind") == "tailcall" else ""
        primary = (
            f"caller_static {caller_static} | call {call_addr}{kind_tag}"
            if prefer_caller_static
            else f"call {call_addr} | caller_static {caller_static}{kind_tag}"
        )
        lines = [
            primary,
            (
                f"within: {containing.get('name', '<unknown>')} @ "
                f"{containing.get('address', '<unknown>')}"
            ),
            f"callee: {callee.get('name', '<unknown>')} @ {callee.get('address', '<unknown>')}",
        ]
        if call_index is not None:
            lines.append(f"call-index: {call_index}")
        if row.get("within_query"):
            lines.append(f"within-query: {row['within_query']}")
        if row.get("hlil_statement"):
            lines.append(f"hlil: {row['hlil_statement']}")
        if row.get("pre_branch_condition"):
            lines.append(f"pre-branch: {row['pre_branch_condition']}")

        call_instruction = row.get("call_instruction") if isinstance(row.get("call_instruction"), dict) else {}
        previous = list(row.get("previous_instructions") or [])
        next_instructions = list(row.get("next_instructions") or [])
        lines.append("context:")
        for item in previous:
            if isinstance(item, dict):
                lines.append(f"  {item.get('address', '<unknown>')}  {item.get('text', '')}".rstrip())
        lines.append(
            f"> {call_instruction.get('address', '<unknown>')}  {call_instruction.get('text', '')}".rstrip()
        )
        for item in next_instructions:
            if isinstance(item, dict):
                lines.append(f"  {item.get('address', '<unknown>')}  {item.get('text', '')}".rstrip())
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _render_structured_il_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    fn = _as_dict(value.get("function"))
    form = "ssa" if value.get("ssa") else "non-ssa"
    lines = [f"{fn.get('name', '<unknown>')} @ {fn.get('address', '<unknown>')}  ({value.get('view', 'mlil')} {form})"]
    for ins in list(value.get("instructions") or []):
        # A malformed list element (non-dict) must render as fallback text, not
        # crash the whole listing with an AttributeError (#101).
        if not isinstance(ins, dict):
            lines.append(f"  {ins}")
            continue
        reads = ",".join(_as_dict(v).get("ssa", _as_dict(v).get("name", "?")) for v in (ins.get("vars_read") or []))
        writes = ",".join(_as_dict(v).get("ssa", _as_dict(v).get("name", "?")) for v in (ins.get("vars_written") or []))
        head = f"  [{ins.get('il_index')}] {ins.get('address')}  {ins.get('op')}  {ins.get('text', '')}".rstrip()
        lines.append(head)
        if reads or writes:
            lines.append(f"        r:[{reads}]  w:[{writes}]")
    return "\n".join(lines)


def _render_defuse_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    fn = value.get("function") or {}
    var = value.get("variable") or {}
    lines = [
        f"{fn.get('name', '<unknown>')} @ {fn.get('address', '<unknown>')}",
        f"variable: {var.get('ssa', var.get('name', '?'))}  ({var.get('type', '?')})",
    ]
    definition = value.get("definition")
    if definition:
        lines.append(f"def: {definition.get('address')}  {definition.get('op')}  {definition.get('text', '')}".rstrip())
    else:
        lines.append("def: <none (parameter/entry/aliased)>")
    if value.get("is_phi"):
        srcs = ", ".join(s.get("ssa", s.get("name", "?")) for s in (value.get("phi_sources") or []))
        lines.append(f"phi sources: {srcs}")
    uses = list(value.get("uses") or [])
    lines.append(f"uses ({len(uses)}):")
    for u in uses:
        lines.append(f"  {u.get('address')}  {u.get('op')}  {u.get('text', '')}".rstrip())
    others = value.get("other_versions") or []
    if others:
        lines.append(f"other versions of {var.get('name', '?')}: {others}")
    return "\n".join(lines)


def _render_callgraph_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    fn = value.get("function") or {}
    lines = [f"{fn.get('name', '<unknown>')} @ {fn.get('address', '<unknown>')}"]
    if "callees" in value:
        callees = list(value.get("callees") or [])
        lines.append(f"callees ({len(callees)}):")
        for c in callees:
            if c.get("kind") == "direct":
                tgt = c.get("target") or {}
                lines.append(f"  {c.get('call_addr')}  direct -> {tgt.get('name', '<unknown>')} @ {tgt.get('address')}")
            else:
                resolved = c.get("resolved") or []
                if resolved:
                    tgts = ", ".join(f"{r.get('name', '?')}@{r.get('address')}" for r in resolved)
                    suffix = f"resolved: {tgts}"
                else:
                    suffix = f"UNRESOLVED ({c.get('resolution_detail', 'indirect')})"
                lines.append(f"  {c.get('call_addr')}  indirect [{c.get('dest_expr', '')}]  {suffix}")
    if "callers" in value:
        callers = list(value.get("callers") or [])
        lines.append(f"callers ({len(callers)}):")
        for c in callers:
            caller = c.get("caller") or {}
            site = f"{c.get('call_addr')}  " if c.get("call_addr") else ""
            lines.append(f"  {site}{caller.get('name', '<unknown>')} @ {caller.get('address', '?')}")
    return "\n".join(lines)


def _render_values_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    fn = value.get("function") or {}
    lines = [f"{fn.get('name', '<unknown>')} @ {fn.get('address', '<unknown>')}"]
    lines.append(f"at {value.get('at')}: {value.get('expression', '<no instruction at address>')}")
    pvs = value.get("possible_values")
    if not pvs:
        lines.append("possible values: <unavailable>")
        return "\n".join(lines)
    summary = pvs.get("type", "?")
    if "value" in pvs:
        summary += f"  value={pvs['value']:#x}" if isinstance(pvs["value"], int) else f"  value={pvs['value']}"
    if pvs.get("values"):
        summary += f"  values={pvs['values']}"
    if pvs.get("ranges"):
        summary += f"  ranges={pvs['ranges']}"
    lines.append(f"possible values: {summary}")
    lines.append(f"  raw: {pvs.get('raw', '')}")
    if value.get("value_basis") == "source_expression" and value.get("source_expression"):
        lines.append(f"  of source expression: {value['source_expression']}")
    return "\n".join(lines)


def _render_leaf_line(leaf: dict[str, Any]) -> str:
    """One text line for a single unresolved-frontier leaf."""
    kind = leaf.get("kind")
    if kind == "unmodeled_callee":
        cal = leaf.get("callee") or {}
        args = leaf.get("tainted_args") or []
        return (
            f"  unmodeled_callee @ {leaf.get('address')}"
            f"  -> {cal.get('name', '?')} @ {cal.get('address', '?')}"
            f"  (tainted arg(s) {args})"
            + (f"  -- {leaf.get('note')}" if leaf.get("note") else "")
        )
    if kind == "field_load_unresolved":
        bits = []
        if leaf.get("base") is not None:
            bits.append(f"base={leaf['base']}")
        if leaf.get("offset") is not None:
            bits.append(f"offset={leaf['offset']}")
        if leaf.get("width") is not None:
            bits.append(f"width={leaf['width']}")
        meta = ("  " + " ".join(bits)) if bits else ""
        return f"  field_load_unresolved @ {leaf.get('address')}{meta}"
    return (
        f"  {kind} @ {leaf.get('address')}  [{leaf.get('dest_expr', leaf.get('il_text', ''))}]"
        + (f"  -- {leaf.get('detail')}" if leaf.get("detail") else "")
    )


def _leaf_group_key(leaf: dict[str, Any]) -> tuple:
    """Collapse near-identical frontier leaves: one group per callee for
    unmodeled calls, per (base, offset) for field loads, per kind otherwise."""
    kind = leaf.get("kind")
    if kind == "unmodeled_callee":
        return (kind, (leaf.get("callee") or {}).get("name", "?"))
    if kind == "field_load_unresolved":
        return (kind, leaf.get("base"), leaf.get("offset"))
    return (kind,)


def _render_grouped_leaves(leaves: list[dict[str, Any]], *, top_n: int = 12) -> list[str]:
    """Render unresolved leaves grouped by kind/callee with counts and a top-N
    cap (full detail stays in --format json) so real binaries don't flood the
    text output with a wall of near-identical leaves (#160)."""
    groups: dict[tuple, dict[str, Any]] = {}
    order: list[tuple] = []
    for leaf in leaves:
        gk = _leaf_group_key(leaf)
        g = groups.get(gk)
        if g is None:
            groups[gk] = {"rep": leaf, "count": 1}
            order.append(gk)
        else:
            g["count"] += 1
    order.sort(key=lambda gk: groups[gk]["count"], reverse=True)
    total = len(leaves)
    ngroups = len(order)
    hdr = f"UNRESOLVED LEAVES ({total}"
    if ngroups != total:
        hdr += f" in {ngroups} group(s)"
    hdr += "):"
    out = [hdr]
    for gk in order[:top_n]:
        g = groups[gk]
        line = _render_leaf_line(g["rep"])
        if g["count"] > 1:
            line += f"  (x{g['count']})"
        out.append(line)
    if ngroups > top_n:
        hidden = order[top_n:]
        hidden_leaves = sum(groups[gk]["count"] for gk in hidden)
        out.append(
            f"  ... and {len(hidden)} more group(s) ({hidden_leaves} leaf(s)); "
            "see --format json for the full list")
    return out


def _render_taint_path(steps: list[Any]) -> list[str]:
    out = []
    last = len(steps) - 1
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        marker = ">" if i == last else " "
        reason = step.get("reason")
        line = f"  {marker} {step.get('address')}  {step.get('op')}  {step.get('il_text', '')}".rstrip()
        out.append(line)
        if reason:
            out.append(f"        <- {reason}")
    return out


def _render_taint_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    fn = value.get("function") or {}
    direction = value.get("direction", "forward")
    lines = [f"{direction} taint in {fn.get('name', '<unknown>')} @ {fn.get('address', '<unknown>')}"]

    if direction == "forward":
        srcs = value.get("sources") or []
        lines.append("sources: " + (", ".join(_describe_loc(s) for s in srcs) or "<none>"))
        findings = list(value.get("reached_sinks") or [])
        lines.append("")
        if not findings:
            # Distinguish "no flow at all" from "flow stopped at an unmodeled
            # frontier" -- a bare "no sinks reached" over a non-empty leaves[]
            # is a dangerous false all-clear (#8).
            fwd_leaves = list(value.get("leaves") or [])
            if fwd_leaves:
                lines.append(
                    f"no modeled sink reached; {len(fwd_leaves)} tainted frontier(s) -- see leaves")
            else:
                lines.append("no sinks reached by tainted data")
        else:
            lines.append(f"reached {len(findings)} sink(s):")
            for f in findings:
                sink = f.get("sink") or {}
                lines.append("")
                lines.append(
                    f"[{sink.get('class', '?')}] {sink.get('callee', '?')} @ {sink.get('address')} "
                    f"(arg {sink.get('tainted_arg_index')}) -- {sink.get('detail', '')}".rstrip()
                )
                lines.extend(_render_taint_path(f.get("path") or []))
    else:
        sinks = value.get("sinks") or []
        lines.append("sinks: " + (", ".join(_describe_loc(s) for s in sinks) or "<none>"))
        slices = list(value.get("slices") or [])
        for sl in slices:
            sink = sl.get("sink") or {}
            origin = sl.get("origin") or {}
            lines.append("")
            lines.append(
                f"slice for {sink.get('callee') or sink.get('kind') or '?'} @ {sink.get('address')} (seed {sink.get('seed', '?')}):"
            )
            _ok = origin.get("kind")
            if _ok == "constant" and origin.get("value") is not None:
                _val = origin["value"]
                _vs = f"{_val:#x}" if isinstance(_val, int) else str(_val)
                _extra = _vs + (f" ({origin['var']})" if origin.get("var") else "")
            else:
                _extra = origin.get("callee") or origin.get("var") or ""
            lines.append(f"  origin: {_ok} {_extra}".rstrip())
            crossed = sl.get("crossed_functions") or []
            if crossed:
                lines.append(f"  crosses: {' <- '.join(crossed)}")
            for step in sl.get("slice") or []:
                if isinstance(step, dict):
                    lines.append(f"  {step.get('address')}  {step.get('op')}  {step.get('il_text', '')}".rstrip())
        unseeded = [s for s in (value.get("sink_status") or []) if not s.get("seeded", True)]
        if unseeded:
            lines.append("")
            lines.append(f"UNSEEDED SINKS ({len(unseeded)}):")
            for s in unseeded:
                lines.append(f"  {_describe_loc(s)} -- {s.get('note', 'could not seed')}")

    by_source = value.get("by_source")
    if direction == "forward" and isinstance(by_source, dict) and by_source:
        lines.append("")
        lines.append(f"PER-SOURCE ({len(by_source)} call site(s)):")
        for addr, br in by_source.items():
            bsinks = br.get("reached_sinks") or []
            bleaves = br.get("leaves") or []
            if bsinks:
                desc = ", ".join(
                    f"{(s.get('sink') or {}).get('class', '?')} {(s.get('sink') or {}).get('callee', '?')}"
                    for s in bsinks)
            else:
                desc = "no sinks"
            nfront = sum(1 for l in bleaves if l.get("kind") == "unmodeled_callee")
            if bleaves:
                desc += f"; {len(bleaves)} leaf(s)" + (f" ({nfront} frontier)" if nfront else "")
            lines.append(f"  {addr}: {desc}")

    leaves = list(value.get("leaves") or [])
    if leaves:
        lines.append("")
        lines.extend(_render_grouped_leaves(leaves))
    assumptions = list(value.get("assumptions") or [])
    if assumptions:
        lines.append("")
        lines.append("ASSUMPTIONS:")
        for a in assumptions:
            lines.append(f"  - {a}")
    if value.get("soundness"):
        lines.append("")
        lines.append(f"soundness: {value['soundness']}")
    return "\n".join(lines)


def _describe_loc(loc: Any) -> str:
    if not isinstance(loc, dict):
        return str(loc)
    kind = loc.get("kind")
    if kind == "param":
        return f"param:{loc.get('index')}"
    if kind == "var":
        return f"var:{loc.get('selector')}"
    if kind == "ret":
        return f"ret:{loc.get('callee')}"
    if kind == "arg":
        return f"arg:{loc.get('callee')}:{loc.get('index')}"
    return str(kind)


def _render_type_list_text(value: Any) -> str:
    # Paged envelope ({items,total,...}) -> render the page + the shared footer;
    # a bare list falls through to the per-item body below (back-compat) (#131).
    if isinstance(value, dict) and "items" in value:
        return _render_paged_list_text(value, "items", _render_type_list_text)
    if not isinstance(value, list):
        return _render_fallback_text(value)
    if not value:
        return "none"

    lines = []
    for item in value:
        if not isinstance(item, dict):
            lines.append(_render_fallback_text(item))
            continue
        name = item.get("name", "<unknown>")
        kind = item.get("kind", "<unknown>")
        decl = item.get("decl")
        line = f"{name} | {kind}"
        if decl:
            line += f" | {decl}"
        lines.append(line)
    return "\n".join(lines)


def _render_imports_summary_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    total = value.get("total_symbols", 0)
    # Label matches the JSON key (`total_symbols`) instead of drifting to
    # "total imports".
    lines = [f"total symbols: {total}"]
    needed = value.get("needed_libraries") or []
    if needed:
        lines.append("")
        lines.append("needed libraries (DT_NEEDED):")
        for lib in needed:
            lines.append(f"  {lib}")
    # Skip the breakdown sections entirely when empty (e.g. a 0-import target),
    # rather than printing dangling "by namespace:"/"by kind:" headers.
    namespaces = value.get("namespaces") or {}
    if namespaces:
        lines.append("")
        lines.append("by namespace:")
        for ns, count in sorted(namespaces.items(), key=lambda x: -x[1]):
            lines.append(f"  {count:>5}  {ns if ns else '(unnamed)'}")
    by_kind = value.get("by_kind") or {}
    if by_kind:
        lines.append("")
        lines.append("by kind:")
        for kind, count in sorted(by_kind.items(), key=lambda x: -x[1]):
            lines.append(f"  {count:>5}  {kind}")
    return "\n".join(lines)


def _render_strings_rows(value: Any) -> str:
    """Render a BARE list of string rows."""
    if not isinstance(value, list):
        return _render_fallback_text(value)
    if not value:
        return "none"

    lines = []
    for item in value:
        if not isinstance(item, dict):
            lines.append(_render_fallback_text(item))
            continue
        address = item.get("address", "<unknown>")
        length = item.get("length", "?")
        chars = item.get("chars")
        string_type = item.get("type", "")
        rendered = json.dumps(item.get("value", ""), ensure_ascii=True)
        if chars is not None and isinstance(length, int) and chars != length:
            size = f"chars={chars} bytes={length}"
        elif chars is not None:
            size = f"chars={chars}"
        else:
            size = f"len={length}"
        lines.append(f"{address}  {size}  {string_type}  {rendered}".rstrip())
    return "\n".join(lines)


def _render_strings_text(value: Any) -> str:
    """Render strings: the paged {items, total, ...} envelope (with a footer),
    or a bare list for back-compat / internal callers (#122)."""
    return _render_paged_list_text(value, "items", _render_strings_rows)


def _render_sections_rows(value: Any) -> str:
    """Render a BARE list of section rows."""
    if not isinstance(value, list):
        return _render_fallback_text(value)
    if not value:
        return "none"

    lines = []
    for item in value:
        if not isinstance(item, dict):
            lines.append(_render_fallback_text(item))
            continue
        name = item.get("name", "<unknown>")
        start = item.get("start", "?")
        end = item.get("end", "?")
        length = item.get("length", "?")
        semantics = item.get("semantics", "")
        perms = ""
        if "readable" in item:
            perms = ("r" if item["readable"] else "-") + ("w" if item.get("writable") else "-") + ("x" if item.get("executable") else "-")
        line = f"{start}-{end}  {length:>8}  {perms:>3}  {semantics:<20}  {name}"
        lines.append(line.rstrip())
    return "\n".join(lines)


def _render_sections_text(value: Any) -> str:
    """Render sections: the paged {items, total, ...} envelope (with a footer),
    or a bare list for back-compat / internal callers (#122)."""
    return _render_paged_list_text(value, "items", _render_sections_rows)


def _render_read_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    hex_str = value.get("hex")
    if not isinstance(hex_str, str):
        return _render_fallback_text(value)

    try:
        data = bytes.fromhex(hex_str)
    except ValueError:
        return _render_fallback_text(value)

    address = value.get("address", "0x0")
    try:
        base = int(str(address), 16) if str(address).lower().startswith("0x") else int(address)
    except (TypeError, ValueError):
        base = 0

    lines: list[str] = []
    width = 16
    for offset in range(0, len(data), width):
        chunk = data[offset:offset + width]
        hex_bytes = " ".join(f"{b:02x}" for b in chunk)
        hex_bytes = f"{hex_bytes:<{width * 3 - 1}}"
        ascii_chunk = "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in chunk)
        lines.append(f"{base + offset:08x}: {hex_bytes}  {ascii_chunk}")

    if not lines:
        lines.append(f"{base:08x}: (no bytes)")

    note = value.get("note")
    if isinstance(note, str) and note:
        lines.append("")
        lines.append(f"note: {note}")

    return "\n".join(lines)


def _render_doctor_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    lines = [
        f"cli version: {value.get('cli_version', '<unknown>')}",
        f"plugin source: {value.get('plugin_source_dir', '<unknown>')}",
        f"plugin install: {value.get('plugin_install_dir', '<unknown>')}",
        f"plugin source build: {value.get('plugin_source_build_id', '<unknown>')}",
        f"plugin install build: {value.get('plugin_install_build_id', '<unknown>')}",
        "",
        "instances:",
    ]
    instances = list(value.get("instances") or [])
    if not instances:
        lines.append("- none")
        return "\n".join(lines)

    for item in instances:
        if not isinstance(item, dict):
            lines.append("- " + _render_fallback_text(item))
            continue
        doctor = item.get("doctor") if isinstance(item.get("doctor"), dict) else {}
        # Prefer the status the JSON carries (L16) so text and JSON can't drift;
        # fall back to deriving it for any caller that built the dict the old way.
        status = item.get("status") or ("ok" if doctor and not doctor.get("error") else "error")
        lines.append(
            "- "
            + f"pid={item.get('pid', '<unknown>')} plugin={item.get('plugin_version', '<unknown>')} status={status}"
        )
        build_id = item.get("plugin_build_id")
        if build_id:
            lines.append(f"  build: {build_id}")
        if item.get("stale_plugin_version"):
            lines.append("  stale: loaded plugin version differs from CLI version")
        if item.get("stale_plugin_code"):
            lines.append("  stale: loaded plugin code does not match installed plugin file")
        if item.get("started_at"):
            lines.append(f"  started: {item['started_at']}")
        if item.get("socket_path"):
            lines.append(f"  socket: {item['socket_path']}")
        error = doctor.get("error")
        if error:
            lines.append(f"  error: {error}")
    return "\n".join(lines)


def _format_operation_result(item: dict[str, Any]) -> str:
    op = item.get("op", "<unknown>")
    requested = item.get("requested") or {}

    def _get(key: str, default: str = "<unknown>") -> str:
        return item.get(key) or requested.get(key, default)

    if op == "function_create":
        name = item.get("function")
        suffix = f" ({name})" if name else ""
        return f"function_create {_get('address')}{suffix}"
    if op == "rename_symbol":
        return f"rename_symbol {_get('kind', 'auto')} {_get('address')} -> {_get('new_name')}"
    if op == "set_comment":
        target = item.get("function") or requested.get("function") or _get("address")
        return f"set_comment {target}"
    if op == "delete_comment":
        target = item.get("function") or requested.get("function") or _get("address")
        return f"delete_comment {target}"
    if op == "set_prototype":
        return f"set_prototype {_get('function')} @ {_get('address')}"
    if op in {"local_rename", "local_retype"}:
        target = item.get("local_id") or item.get("variable") or requested.get("variable", "<unknown>")
        return f"{op} {_get('function')}::{target}"
    if op == "struct_field_set":
        return (
            f"struct_field_set {_get('struct_name')} "
            f"{_get('offset')} {_get('field_name')} {_get('field_type')}"
        )
    if op == "struct_field_rename":
        return (
            f"struct_field_rename {_get('struct_name')} "
            f"{_get('old_name')} -> {_get('new_name')}"
        )
    if op == "struct_field_delete":
        return f"struct_field_delete {_get('struct_name')}::{_get('field_name')}"
    if op == "types_declare":
        return (
            f"types_declare {item.get('count', 0)} types"
            f" (parsed functions={item.get('parsed_function_count', len(item.get('parsed_functions') or []))},"
            f" variables={item.get('parsed_variable_count', len(item.get('parsed_variables') or []))})"
        )
    return _render_fallback_text(item)


def _format_op_summary(item: dict[str, Any]) -> str:
    summary = _format_operation_result(item)
    if item.get("status"):
        summary += f" [{item['status']}]"
    if item.get("changed") is False and item.get("status") not in (None, "noop"):
        summary += " [no change]"
    if item.get("message"):
        summary += f" ({item['message']})"
    return summary


def _render_mutation_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    preview = bool(value.get("preview"))
    success = bool(value.get("success", True))
    committed = bool(value.get("committed", False))
    results = [r for r in (value.get("results") or []) if isinstance(r, dict)]
    failed = [r for r in results if r.get("status") in FAILED_MUTATION_STATUSES]

    lines: list[str] = []

    if not success or failed:
        if not committed:
            # Only claim a rollback we actually performed. When rolled_back is
            # explicitly False the revert failed and the view may be modified --
            # saying "rolled back" there contradicts the honest message and
            # re-states the very symptom #117 set out to fix.
            if value.get("rolled_back") is False:
                lines.append("rollback failed: the view may be left modified")
            else:
                lines.append("rolled back: live verification failed")
        if value.get("message"):
            lines.append(value["message"])
        for item in failed:
            lines.append("failed: " + _format_op_summary(item))
            if item.get("requested"):
                lines.append("  requested: " + json.dumps(item["requested"], sort_keys=True))
            if item.get("observed"):
                lines.append("  observed: " + json.dumps(item["observed"], sort_keys=True))
        lines.append("")
    elif preview:
        lines.append("preview: change applied + reverted")
        if value.get("message"):
            lines.append(value["message"])
        lines.append("")

    if results:
        if len(results) == 1 and success and not failed and not preview:
            lines.append(_format_op_summary(results[0]))
        else:
            lines.append(f"results ({len(results)}):")
            for item in results:
                lines.append("- " + _format_op_summary(item))

    affected_functions = [a for a in (value.get("affected_functions") or []) if isinstance(a, dict)]
    changed_functions = [a for a in affected_functions if a.get("changed")]
    if changed_functions:
        lines.extend(["", f"affected functions ({len(changed_functions)}):"])
        for item in changed_functions:
            before_name = item.get("before_name") or item.get("after_name") or "<unknown>"
            after_name = item.get("after_name") or before_name
            summary = f"{item.get('address', '<unknown>')} {before_name}"
            if after_name != before_name:
                summary += f" -> {after_name}"
            lines.append("- " + summary)
            if preview and item.get("diff"):
                lines.append(str(item["diff"]))

    affected_types = [a for a in (value.get("affected_types") or []) if isinstance(a, dict)]
    changed_types = [a for a in affected_types if a.get("changed")]
    if changed_types:
        lines.extend(["", f"affected types ({len(changed_types)}):"])
        for item in changed_types:
            summary = item.get("type_name", "<unknown>")
            if item.get("message"):
                summary += f" ({item['message']})"
            lines.append("- " + summary)
            if preview and item.get("layout_diff"):
                lines.append(str(item["layout_diff"]))

    return "\n".join(lines).rstrip() + "\n"


def _render_py_exec_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    parts: list[str] = []
    stdout = value.get("stdout")
    if isinstance(stdout, str) and stdout:
        parts.append(stdout.rstrip("\n"))

    result = value.get("result")
    if result is not None:
        body = result if isinstance(result, str) else json.dumps(result, indent=2, sort_keys=True)
        parts.append("result:\n" + body)

    warnings = list(value.get("warnings") or [])
    if warnings:
        parts.append("warnings:\n" + "\n".join(f"- {warning}" for warning in warnings))

    artifact = value.get("artifact")
    if isinstance(artifact, dict) and artifact.get("artifact_path"):
        parts.append(f"artifact: {artifact['artifact_path']}")

    if not parts:
        return ""
    return "\n\n".join(parts)


def _render_skill_install_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    installed = value.get("installed_destinations")
    skipped = value.get("skipped_destinations")
    lines = []

    if isinstance(installed, list) and installed:
        lines.append(f"Installed skills ({value.get('mode', 'unknown')}):")
        lines.extend(f"- {dest}" for dest in installed)
    else:
        lines.append("Skills already installed.")

    if isinstance(skipped, list) and skipped:
        lines.append("Skipped existing destinations:")
        lines.extend(f"- {dest}" for dest in skipped)

    return "\n".join(lines) + "\n"


_TRACE_REASON_LABELS: dict[str, str] = {
    "function_parameter": "function parameter",
    # A terminal with no reaching SSA definition that is NOT a confirmed
    # parameter: an undefined local or a global. Don't claim "function
    # parameter" — that misleads provenance/attacker-source slices.
    "undefined_or_global": "undefined / no reaching definition",
    # Legacy combined reason (still accepted on input): stay neutral.
    "function_parameter_or_global": "function parameter, global, or undefined",
    "memory_load": "memory load",
    "call_or_jump_boundary": "call boundary",
}


def _render_trace_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    fn_name = value.get("function", "<unknown>")
    fn_addr = value.get("function_address", "<unknown>")
    target_addr = value.get("target_address", "<unknown>")
    arg_index = value.get("arg_index", 0)
    trace = list(value.get("trace") or [])

    header = (
        f"backward trace of arg[{arg_index}] in {fn_name} @ {target_addr}"
    )
    step_word = "step" if len(trace) == 1 else "steps"
    info = f"  {fn_name} @ {fn_addr}  •  {len(trace)} {step_word}"
    if value.get("truncated"):
        info += "  •  truncated"

    if not trace:
        return f"{header}\n{info}\n\n  constant or immediate — no SSA trace"

    lines = [header, info, ""]
    current_fn: str | None = None
    for step in trace:
        if not isinstance(step, dict):
            lines.append(f"  {step}")
            continue

        fn_ctx = step.get("function_context")
        cross_fn = step.get("cross_function")
        ssa_var = step.get("ssa_var", "")
        addr = step.get("address")
        il_text = step.get("il_text") or ""
        reason = step.get("reason") or ""
        terminates = bool(step.get("terminates"))

        if cross_fn:
            callee = step.get("callee", "")
            lines.append(f"  ── enters {callee} ──")
            current_fn = callee
            continue

        if fn_ctx and fn_ctx != current_fn:
            lines.append(f"  ── in {fn_ctx} ──")
            current_fn = fn_ctx

        if terminates:
            label = _TRACE_REASON_LABELS.get(reason, reason.replace("_", " "))
            line = f"  {ssa_var}  —  {label}"
            if il_text:
                line += f"  @ {addr}  {il_text}" if addr else f"  {il_text}"
            lines.append(line)
        else:
            if addr:
                lines.append(f"  {addr}  {il_text}")
            else:
                lines.append(f"  {il_text}")

    return "\n".join(lines)
