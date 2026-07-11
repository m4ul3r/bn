from __future__ import annotations

import json
import re
from typing import Any, Callable

from .transport import BridgeError

# "rollback_failed" = an op succeeded but the batch revert that should have
# undone it failed, so the view may be left modified -- a real failure. A
# cleanly rolled-back sibling ("reverted") is NOT a failure and is omitted (#118).
# "internal_error" = an unexpected engine bug (distinct from an unsupported
# request); still a failure, so exit codes/rendering flag it (#122).
FAILED_MUTATION_STATUSES = {"unsupported", "verification_failed", "invalid_request", "rollback_failed", "internal_error"}

# Control chars (C0 minus the ones we name, plus DEL) in a symbol name would
# break a --format text row across lines or corrupt the terminal. Escape them so
# the row stays on one line and the name is still readable (#370.1). JSON output
# is untouched -- it round-trips the raw name faithfully.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _escape_control_chars(text: Any) -> str:
    s = str(text)
    s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return _CONTROL_CHAR_RE.sub(lambda m: f"\\x{ord(m.group()):02x}", s)


def _render_fallback_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True)


def _render_function_bundle_text(value: Any) -> str:
    """`bundle function` is a composite JSON artifact (function info + decompile +
    IL + xrefs + types) with no compact text form. Without a renderer the explicit
    --format text path emitted a single line of escaped JSON; pretty-print it with
    a note instead so a text-defaulting agent gets something readable (#362)."""
    if not isinstance(value, (dict, list)):
        return _render_fallback_text(value)
    try:
        body = json.dumps(value, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        return _render_fallback_text(value)
    return ("# function bundle: composite JSON (use --format json for machine "
            "consumption)\n" + body)


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
    text: str, lines_range: tuple[int, int] | None, *, marker: str = "//", flag: str = "--lines"
) -> str:
    """Return only lines START..END (1-indexed, inclusive) with a count header.

    Shared by `decompile`, `il`, `disasm`, and `structured-il` so every
    line-oriented view slices the same way. Slicing happens before the spill
    check, so `--lines` also keeps large functions inline. *flag* names the
    caller's flag in the out-of-range error so a `disasm --count N` failure
    reads `--count`, not `--lines` (#291.2 review).
    """
    if lines_range is None:
        return text
    all_lines = text.splitlines()
    total = len(all_lines)
    start, end = lines_range
    if start > total:
        # A start past the last line is a user error, not a result. Raising keeps
        # it from rendering as a `//` line (mistakable for code) and exiting 0:
        # BridgeError propagates to main() -> stderr diagnostic, non-zero exit, no
        # stdout a scripted consumer could read as a real slice (#253).
        raise BridgeError(
            f"{flag} start {start} is beyond the last line "
            f"(output has {total} line{'s' if total != 1 else ''}); "
            f"omit {flag} or choose a start within range"
        )
    sliced = all_lines[start - 1 : end]
    header = f"{marker} lines {start}-{min(end, total)} of {total}"
    return header + "\n" + "\n".join(sliced)


def _resolution_note(value: Any) -> str:
    """A leading note when a function-scoped read resolved an interior address
    to its containing function (#193 Part 4).

    Without it, text-mode output for a mid-function address (e.g. a taint/trace
    sink) silently shows a function whose start differs from what was asked,
    which reads like the wrong answer. Returns '' when not applicable.
    """
    if not isinstance(value, dict):
        return ""
    resolved_from = value.get("resolved_from")
    if not isinstance(resolved_from, dict):
        return ""
    function = _as_dict(value.get("function"))
    name = function.get("name", "?")
    address = function.get("address", "?")
    return (
        f"// bn: {resolved_from.get('requested_address')} is inside {name} "
        f"@ {address} ({resolved_from.get('offset')}); showing the containing function\n"
    )


def _disasm_linear_steer_note(value: Any, *, sliced: bool) -> str:
    """A disasm-only note when a mid-function address was sliced (#371.3).

    `disasm <mid-addr> --count N` (or `--lines`) slices from the function
    PROLOGUE, not the requested address, so an agent inspecting a call site via
    an xref address silently gets the prologue. Point at `--linear`, which
    decodes N instructions from the exact address regardless of function
    membership. Fires only when a slice is active AND the address resolved
    mid-function -- an exact start or a whole-function dump has no trap.
    """
    if not sliced or not isinstance(value, dict):
        return ""
    resolved_from = value.get("resolved_from")
    if not isinstance(resolved_from, dict):
        return ""
    addr = resolved_from.get("requested_address", "?")
    return (
        f"// bn: --count/--lines slices from the function start, not {addr}; "
        f"to disassemble from {addr} itself use `disasm {addr} --linear N`\n"
    )


def _render_disasm_linear_text(value: Any) -> str:
    """Render a linear (non-function-bounded) disassembly: a leading `// bn:` note
    so it's clearly NOT a function listing, then the address/bytes/mnemonic lines
    (#314)."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    note = value.get("note")
    body = _text_field("text")(value)
    if note:
        return f"// bn: {note}\n{body}" if body else f"// bn: {note}"
    return body


def _render_capabilities_text(value: Any) -> str:
    """Render the #276 capability index as a grouped, scannable catalog: each
    top-level group, its commands with one-line help, and the prefer-when /
    see-also routing hints where a command overlaps a neighbor."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    items = value.get("items") or []
    lines: list[str] = []
    current_group: str | None = None
    for item in items:
        group = item.get("group", "")
        if group != current_group:
            if lines:
                lines.append("")
            lines.append(f"{group}:")
            current_group = group
        command = item.get("command", "")
        help_text = item.get("help", "")
        lines.append(f"  {command}  --  {help_text}" if help_text else f"  {command}")
        if item.get("prefer_when"):
            lines.append(f"      prefer when: {item['prefer_when']}")
        if item.get("see_also"):
            lines.append(f"      see also: {', '.join(item['see_also'])}")
    return "\n".join(lines)


def _render_function_info_text(value: Any, verbose: bool = False, demangle: bool = False) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    function = _as_dict(value.get("function"))
    header_name = function.get("name", "<unknown>")
    if demangle and function.get("display_name"):
        header_name = function["display_name"]
    lines = [
        f"{_escape_control_chars(header_name)} @ {function.get('address', '<unknown>')}",
        str(value.get("prototype", "")),
        f"calling convention: {value.get('calling_convention', '<unknown>')}",
        f"size: {value.get('size', '<unknown>')}",
        f"xrefs: {value.get('xref_count', 0)}",
    ]

    locals_only = list(value.get("locals") or [])
    if locals_only:
        lines.append(f"locals: {len(locals_only)} variables")

    # Surface unlifted instructions (#206) -- a function whose computation BN
    # couldn't model otherwise reads as fully analyzed.
    unimpl = value.get("unimplemented_instructions")
    if isinstance(unimpl, dict) and unimpl.get("count"):
        addrs = list(unimpl.get("addresses") or [])
        shown = ", ".join(addrs)
        if unimpl.get("truncated"):
            shown += ", …"
        suffix = f" (e.g. {shown})" if shown else ""
        lines.append(
            f"unlifted instructions: {unimpl['count']} — BN could not model these; "
            f"dataflow through them is not tracked{suffix}")

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

    return _resolution_note(value) + "\n".join(lines)


def _render_proto_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    prototype = value.get("prototype")
    if not isinstance(prototype, str):
        return _render_fallback_text(value)
    # BN renders the prototype anonymously (`uint64_t (int32_t arg1)`); splice in
    # the function name so the output is a copy-pasteable C declaration (#222).
    note = _resolution_note(value)
    fn = value.get("function")
    name = fn.get("name") if isinstance(fn, dict) else None
    head, sep, rest = prototype.partition("(")
    if name and sep:
        head = head.rstrip()
        # Skip only when the return-type already ENDS with the name as its own
        # declarator token (an already-named prototype) -- a naive `name in head`
        # substring test wrongly skipped when the name was a substring of the
        # return type (e.g. name "t" in "uint64_t") (#222 review).
        already_named = head.split()[-1:] == [name] if head else False
        if not already_named:
            return note + f"{head} {name}({rest}"
    return note + prototype


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
    return _resolution_note(value) + "\n".join(lines)


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
    # #275: refs come as a unified `items` list, each tagged with its `kind`.
    items = list(value.get("items") or [])
    code_refs = [it for it in items if it.get("kind") == "code"]
    data_refs = [it for it in items if it.get("kind") == "data"]
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

    # #532: field xrefs now page like every other xref path. Surface the paging
    # metadata whenever the page isn't the whole ref set -- either more pages remain
    # (has_more) or an --offset skipped earlier refs -- so a partial view (including
    # the last, has_more=False page of an --offset run) isn't read as the full set.
    total = value.get("total")
    returned = value.get("returned", len(items))
    offset = value.get("offset", 0) or 0
    has_more = bool(value.get("has_more"))
    if isinstance(total, int) and (has_more or offset or returned != total):
        note = f"showing {returned} of {total} refs (offset {offset})"
        if has_more:
            note += "; more available -- raise --limit or use --offset"
        lines.extend(["", note])

    return "\n".join(lines)


def _render_comment_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    # `comment get --function` aggregates all in-function comments as a list (#203),
    # and, alongside them, the whole-function documentation (`fn.comment`) as
    # `function_doc` -- surfaced above the address comments so the two stores read
    # as one coherent view instead of the doc silently going missing from `get`.
    comments = value.get("comments")
    if isinstance(comments, list):
        lines = []
        doc = value.get("function_doc")
        if doc:
            lines.append(f"[doc] {doc}")
        if not comments and not doc:
            return "(no comment)"
        lines.extend(
            f"{c.get('address', '?')}  {c.get('comment', '')}"
            for c in comments if isinstance(c, dict)
        )
        return "\n".join(lines)
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


def _render_tag_types_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    types = value.get("tag_types")
    if not isinstance(types, list) or not types:
        return "none"
    lines = []
    for t in types:
        if not isinstance(t, dict):
            lines.append(_render_fallback_text(t))
            continue
        builtin = "  [builtin]" if t.get("is_builtin") else ""
        lines.append(f"{t.get('icon', '')}  {t.get('name', '<unknown>')}{builtin}")
    return "\n".join(lines)


def _render_tag_get_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    tags = value.get("tags")
    if not isinstance(tags, list) or not tags:
        return "(no tags)"
    return "\n".join(_render_tag_row(t) for t in tags if isinstance(t, dict))


def _render_tag_row(t: dict) -> str:
    # A function-scope tag has no address; show the function name it belongs to
    # instead of a bare placeholder. The JSON already carries `function`, so the
    # text renderer just surfaces it (address scope keeps the address, which is
    # the more precise locator when both are present).
    loc = t.get("address") or t.get("function") or "<function>"
    return f"{loc}  [{t.get('scope', '?')}]  {t.get('icon', '')} {t.get('type', '')}  {t.get('data', '')}"


def _render_tag_list_text(value: Any) -> str:
    if isinstance(value, dict) and "items" in value:
        return _render_paged_list_text(value, "items", _render_tag_list_text)
    if not isinstance(value, list):
        return _render_fallback_text(value)
    if not value:
        return "none"
    return "\n".join(_render_tag_row(t) for t in value if isinstance(t, dict))


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
    line = f"saved: {value.get('path', '<unknown>')}"
    if value.get("fallback"):
        # The default path was unwritable (e.g. a read-only firmware mount); the
        # database landed in the writable cache instead (#214).
        line += (f"\nnote: {value.get('requested_path')} was not writable; "
                 f"saved to the cache instead")
    return line


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
    removed = value.get("marker_removed")
    if removed:
        line += f"\nremoved marker(s): {', '.join(str(p) for p in removed)}"
    return line


def _render_session_list_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    instances = list(value.get("items") or value.get("instances") or [])
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
        binaries = item.get("binaries")
        if binaries:
            lines.append(f"  open: {', '.join(str(b) for b in binaries)}")
    total_rss = value.get("total_rss_mb")
    if total_rss is not None and instances:
        lines.append("")
        lines.append(f"total rss: {total_rss}MB")
    return "\n".join(lines)


def _render_instance_find_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    items = list(value.get("items") or [])
    if not items:
        return f"no instance has a binary matching {value.get('query')!r}"
    lines = []
    for item in items:
        selector = str(item.get("selector") or item.get("instance_id") or "<unknown>")
        lines.append(f"{selector}  (instance {item.get('instance_id')})")
        lines.append(f"  {item.get('binary')}")
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
        # Surface analysis_state (full/quick) in text too -- the bn-re methodology
        # tells agents to gate their survey on it, and on a quick-loaded view it
        # explains an apparently-empty result rather than "empty binary" (#378).
        ("analysis", value.get("analysis_state")),
        ("file", value.get("filename")),
        ("arch", value.get("arch")),
        ("platform", value.get("platform")),
        # Preferred/image base BN loaded at (#564) -- for a PIE binary this is the
        # rebase anchor a debugger handoff needs, so surface it beside entry.
        ("image base", value.get("image_base")),
        ("entry", value.get("entry_point")),
    ]
    for key, item in details:
        if item not in (None, ""):
            lines.append(f"\t{key}: {item}")
    # Live analysis phase/counts while a long `bn refresh` runs (#321), so a text
    # user can watch a large-target analysis on another connection instead of
    # guessing it's wedged. Shown for any *active* phase -- some phases (Discovery,
    # ExtendedAnalyze) legitimately report 0/0 -- and hidden only when idle/not
    # started, so the line doesn't flicker away mid-analysis. Counts appended when
    # a meaningful total is known.
    prog = value.get("analysis_progress")
    if isinstance(prog, dict):
        state = prog.get("state")
        if state and state not in ("IdleState", "InitialState"):
            total = prog.get("total") or 0
            counts = f" {prog.get('count')}/{total}" if total else ""
            lines.append(f"\tanalysis progress: {state}{counts}")
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
    # Accept both the {kind, items} envelope (#358) and a bare list (older
    # callers / raw socket clients).
    if isinstance(value, dict):
        items = value.get("items")
    else:
        items = value
    if not isinstance(items, list):
        return _render_fallback_text(value)
    if not items:
        return "no targets"
    return "\n\n".join(
        _render_target_summary(item) if isinstance(item, dict) else _render_fallback_text(item)
        for item in items
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
    line = f"instance: {value.get('instance_id', '<unknown>')}"
    cleared = value.get("cleared_target_pin")
    if cleared:
        line += f"\ncleared stale target pin {cleared!r} (belonged to the previous instance)"
    return line


def _render_target_use_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    return f"target: {value.get('target', '<unknown>')}"


def _render_pin_clear_text(value: Any) -> str:
    """Render `instance clear` / `target clear` confirmations."""
    return "cleared"


def _render_instance_gc_text(value: Any) -> str:
    """Render the `instance gc` cache-cleanup summary."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    logs = value.get("logs_removed", 0)
    socks = value.get("sockets_removed", 0)
    regs = value.get("registries_purged", 0)
    live = value.get("live_instances", 0)
    reaped = logs + socks + regs
    if reaped == 0:
        return f"gc: nothing to reap ({live} live instance{'' if live == 1 else 's'})"
    return (
        f"gc: reaped {logs} log{'' if logs == 1 else 's'}, "
        f"{socks} orphan socket{'' if socks == 1 else 's'}, "
        f"{regs} dead registr{'y' if regs == 1 else 'ies'} "
        f"({live} live instance{'' if live == 1 else 's'} kept)"
    )


def _render_name_address_rows(value: Any, *, demangle: bool = False) -> str:
    """Render a BARE list of name/address rows (imports, function pages). With
    ``demangle``, show the demangled ``display_name`` instead of the raw name so
    a C++ listing is greppable/clusterable without c++filt (#196)."""
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
        if demangle and item.get("display_name"):
            name = item["display_name"]
        line = f"{address}  {_escape_control_chars(name)}"
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
            # #411: surface basic_block_count (a real complexity metric) here too,
            # since text is the DEFAULT read output -- otherwise agents only ever
            # see the misleading byte span. Omit the blocks clause when the count
            # is absent/None (e.g. an older bridge, or a guarded bad function).
            blocks = item.get("basic_block_count")
            if blocks is not None:
                line += f"  ({size} bytes, {blocks} blocks)"
            else:
                line += f"  ({size} bytes)"
        lines.append(line)
    return "\n".join(lines)


def _render_name_address_list_text(value: Any) -> str:
    """Render imports: the paged {items, total, ...} envelope (with a footer),
    or a bare list for back-compat / internal callers (#122)."""
    body = _render_paged_list_text(value, "items", _render_name_address_rows)
    # Surface PIC self-references dropped from the survey so the exclusion isn't
    # silent (#202).
    excluded = value.get("self_defined_excluded") if isinstance(value, dict) else None
    if isinstance(excluded, int) and excluded > 0:
        note = (
            f"// {excluded} self-defined export(s) excluded "
            "(this module's own symbols modeled as import veneers / GOT slots)"
        )
        body = note if body == "none" else f"{body}\n{note}"
    return body


def _render_go_rename_text(value: Any) -> str:
    """Render `go rename` (#217): a compact summary (verified / failed / skipped
    counts) with the preview/rollback banner and a capped FAILED list -- never a
    per-success wall, since a bulk auto-name renames hundreds/thousands at once.
    `results` carries only failures; the applied names are in the database."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    skipped = value.get("skipped_user_named", 0)
    targeted = value.get("go_renamed_candidates", 0)
    if not targeted:
        return ("go rename: nothing to do — no auto-named (sub_*) Go functions to rename "
                f"({value.get('defined_count', 0)} defined at pcln addresses, "
                f"{skipped} already user-named)")
    failed = [r for r in (value.get("results") or []) if isinstance(r, dict)]
    verified = value.get("go_verified_count", targeted - len(failed))
    preview = bool(value.get("preview"))
    committed = bool(value.get("committed", True))
    lines: list[str] = []
    if preview:
        lines.append("preview: renames applied + reverted (nothing committed)")
        lines.append(f"go rename (preview): {verified} would rename, {len(failed)} failed, "
                     f"{skipped} skipped (already user-named)")
    elif not committed and (value.get("success") is False or failed):
        # All-or-nothing: a failure reverted the WHOLE batch, so NOTHING landed --
        # don't claim "N renamed" for rows that passed readback before the revert.
        if value.get("rolled_back") is False:
            lines.append("rollback failed: the view may be left modified")
        else:
            lines.append("rolled back: the batch was reverted because a rename failed — "
                         "NOTHING was committed")
        lines.append(f"go rename: 0 renamed ({verified} would have, {len(failed)} failed, "
                     f"{skipped} skipped); fix the failure(s) below and re-run")
    else:
        lines.append(f"go rename: {verified} renamed, {len(failed)} failed, "
                     f"{skipped} skipped (already user-named)")
    for r in failed[:50]:
        lines.append(f"  failed: {r.get('new_name', '?')} @ {r.get('address', '?')} "
                     f"({r.get('status', '?')})")
    if len(failed) > 50:
        lines.append(f"  ... and {len(failed) - 50} more failed")
    return "\n".join(lines)


def _render_go_functions_text(value: Any) -> str:
    """Render the Go pcln function lens (#217): a header with the detected Go
    version + how many recovered addresses already map to a BN function, the
    optional PIE-rebase note, then the name/address rows + paging footer."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    head = "go functions"
    if value.get("go_version"):
        head += f" ({value['go_version']})"
    parts = []
    if isinstance(value.get("total"), int):
        parts.append(f"{value['total']} recovered")
    if isinstance(value.get("defined_count"), int):
        parts.append(f"{value['defined_count']} mapped to a BN function")
    if parts:
        head += ": " + ", ".join(parts)
    lines = [head]
    if value.get("truncated"):
        # #528: a partial walk must not read as a complete count.
        lines.append(
            f"warning: partial pcln walk -- {value.get('recovered')} of "
            f"{value.get('expected')} declared functions recovered "
            f"({value.get('skipped')} skipped/truncated); the table is malformed or truncated."
        )
    if value.get("note"):
        lines.append(f"note: {value['note']}")
    lines.append(_render_paged_list_text(value, "items", _render_name_address_rows))
    return "\n".join(lines)


def _render_go_functions_summary_text(value: Any) -> str:
    """#414: compact go-metadata summary -- enough to decide whether to run
    `go rename` without listing every function."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    head = "go functions summary"
    if value.get("go_version"):
        head += f" ({value['go_version']})"
    lines = [head]
    for label, key in (("recovered", "recovered"), ("defined", "defined"),
                       ("undefined", "undefined"), ("renamable", "renamable")):
        if isinstance(value.get(key), int):
            lines.append(f"  {label}: {value[key]}")
    if value.get("truncated"):
        # #528: disclose that the declared table was only partially recovered.
        lines.append(
            f"  expected: {value.get('expected')} (partial walk -- "
            f"{value.get('skipped')} skipped/truncated)"
        )
    ts, tsb = value.get("text_start"), value.get("text_start_bv")
    if ts is not None:
        rebase = "" if (tsb is None or tsb == ts) else f"  (BN text {tsb} -- rebase needed)"
        lines.append(f"  text_start: {ts}{rebase}")
    return "\n".join(lines)


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


_QUICK_PARTIAL_WARNING = (
    "WARNING: target is quick-loaded; function list/count is partial. "
    "Run `bn refresh` for full analysis."
)


def _quick_partial_prefix(value: Any) -> str:
    """A leading warning line when the functions envelope is quick-loaded/partial
    (#437), so a text reader doesn't trust a partial count as the whole binary.
    Empty for a fully-analyzed view."""
    if isinstance(value, dict) and value.get("partial"):
        return _QUICK_PARTIAL_WARNING + "\n"
    return ""


def _render_function_count_text(value: Any) -> str:
    """Render a `function list/search --count` result, prefixing the quick-load
    partiality warning when the count is partial (#437)."""
    count = value.get("count", 0) if isinstance(value, dict) else 0
    return f"{_quick_partial_prefix(value)}Total functions: {count}"


def _render_function_list_text(value: Any, *, demangle: bool = False) -> str:
    """Render a paged function listing, with a footer stating the true total and
    remainder (#59). Prefers the canonical `items` key (every other list command
    uses it), falling back to the deprecated byte-identical `functions` alias for
    an older bridge that emits only the latter (#223). ``demangle`` shows the
    demangled display_name (#196). A quick-loaded (partial) listing is prefixed
    with a warning so the page isn't mistaken for the whole binary (#437)."""
    page_key = "items" if isinstance(value, dict) and "items" in value else "functions"
    return _quick_partial_prefix(value) + _render_paged_list_text(
        value, page_key, lambda items: _render_name_address_rows(items, demangle=demangle))


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


def _xref_buckets(value: dict[str, Any]) -> tuple[list[Any], list[Any], int, int]:
    """Split an xrefs response into ``(code_refs, data_refs, total_code, total_data)``.

    Tolerates both shapes: the deprecated dual arrays (still embedded by
    ``function info`` and emitted by field xrefs) and the #184 items-only ``xrefs``
    op response (reconstruct the buckets by splitting ``items`` on ``kind``).
    Totals come from the full-set summary counts (``code_ref_count`` /
    ``data_ref_count``) when present, so the header stays honest even though the op
    no longer ships the full arrays."""
    code_refs = value.get("code_refs")
    data_refs = value.get("data_refs")
    if code_refs is None and data_refs is None:
        items = list(value.get("items") or [])
        code_refs = [r for r in items if isinstance(r, dict) and r.get("kind") == "code"]
        data_refs = [r for r in items if isinstance(r, dict) and r.get("kind") == "data"]
    code_refs = list(code_refs or [])
    data_refs = list(data_refs or [])
    total_code = value.get("code_ref_count")
    total_code = len(code_refs) if total_code is None else total_code
    total_data = value.get("data_ref_count")
    total_data = len(data_refs) if total_data is None else total_data
    return code_refs, data_refs, total_code, total_data


def _render_xrefs_any_text(value: Any) -> str:
    """Render the multi-symbol sink-sweep (`xrefs --any`): one line per symbol,
    present (with counts) or absent (#218)."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    syms = list(value.get("items") or [])  # #275: was `symbols`
    lines = [f"xrefs --any: {value.get('present', 0)}/{value.get('count', len(syms))} symbol(s) present"]
    for s in syms:
        if not isinstance(s, dict):
            continue
        if s.get("present"):
            lines.append(
                f"  {s.get('symbol')}: {s.get('code_ref_count', 0)} code refs across "
                f"{s.get('caller_function_count', 0)} fn(s)  @ {s.get('address', '?')}")
        else:
            lines.append(f"  {s.get('symbol')}: absent")
    return "\n".join(lines)


def _render_xrefs_text(value: Any, limit: int | None = None) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    code_refs, data_refs, total_code, total_data = _xref_buckets(value)

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
    # Ambiguous same-name collision (thunk/real): surface the note so a zero-caller
    # member is never mistaken for dead code (#220).
    amb = value.get("ambiguous_symbol")
    if isinstance(amb, dict) and amb.get("note"):
        lines.insert(0, f"note: {amb['note']}")
        lines.insert(1, "")
    # Data-symbol resolution fallback (#224b).
    rsym = value.get("resolved_symbol")
    if isinstance(rsym, dict) and rsym.get("kind") == "data":
        lines.insert(0, f"note: resolved '{rsym.get('name')}' as a data symbol @ {rsym.get('address')}")
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

    code_refs, data_refs, total_code, total_data = _xref_buckets(value)
    for nice, kind, refs, total in (
        ("code refs", "code", code_refs, total_code),
        ("data refs", "data", data_refs, total_data),
    ):
        shown = refs[:limit] if limit else refs
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
            ref_kind = ref.get("kind") or kind
            # A stored function pointer the back-link scan discovered (#323): mark
            # it [function pointer] so an analyst sees it's a scan-found callback
            # table slot, distinct from a BN-modeled data ref (the section is in
            # the context suffix).
            fp = " [function pointer]" if ref.get("function_pointer") else ""
            lines.append(
                f"- {address}  {ref_kind}  {function}{_context_suffix(ref.get('context'))}{fp}"
            )
    if value.get("fn_pointer_scan_truncated"):
        lines.append(
            "\nnote: the function-pointer back-link scan was truncated (data "
            "sections exceeded the scan budget); some table references may be missing"
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
    # #471: show the slice window when the call set was paged/windowed.
    total_calls = value.get("total_calls")
    matched = value.get("matched_calls")
    call_hdr = f"calls: {len(calls)}"
    if isinstance(total_calls, int) and (
        value.get("offset") or value.get("limit") is not None or matched != total_calls
    ):
        call_hdr += f" of {matched if matched is not None else total_calls}"
        if matched is not None and matched != total_calls:
            call_hdr += f" in window (of {total_calls} total)"
        if value.get("has_more"):
            nxt = int(value.get("offset") or 0) + len(calls)
            call_hdr += f" -- more: rerun with --offset {nxt}"
    lines.append(call_hdr)
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


def _render_surface_text(value: Any) -> str:
    """#503: render the hidden code surface -- init/ctor pointers, candidate vtable/
    dispatch tables, and data-referenced code BN did not functionize."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    s = value.get("summary") or {}
    lines = [
        f"hidden surface: {s.get('init_sections', 0)} init section(s), "
        f"{s.get('candidate_tables', 0)} candidate table(s), "
        f"{s.get('missing_function_candidates', 0)} missing-function candidate(s)"
    ]
    for w in list(value.get("warnings") or []):
        lines.append(f"warning: {w}")

    init = list(value.get("init_sections") or [])
    if init:
        lines.append("")
        lines.append("init / ctor sections (pre-main code):")
        for sec in init:
            if not isinstance(sec, dict):
                continue
            lines.append(
                f"  {sec.get('name', '?')}  {sec.get('start', '?')}..{sec.get('end', '?')}  "
                f"entries={sec.get('total_entries', '?')}  "
                f"fn={sec.get('resolved_functions', 0)}  missing={sec.get('missing_functions', 0)}")

    tables = list(value.get("candidate_tables") or [])
    if tables:
        lines.append("")
        lines.append("candidate vtable / dispatch tables (runs of pointers-to-code):")
        for t in tables:
            if not isinstance(t, dict):
                continue
            lines.append(
                f"  {t.get('address', '?')} [{t.get('section', '?')}]  "
                f"entries={t.get('entries', '?')}  fn={t.get('resolved_functions', 0)}  "
                f"missing={t.get('missing_functions', 0)}")

    cands = list(value.get("missing_function_candidates") or [])
    if cands:
        code_likely = [c for c in cands if isinstance(c, dict) and c.get("code_likely")]
        lines.append("")
        lines.append(
            f"missing-function candidates (executable, data-referenced, no BN function): "
            f"{len(code_likely)} code-likely of {len(cands)}")
        # Show the high-confidence subset first; then the rest, marked with why.
        ordered = code_likely + [c for c in cands if isinstance(c, dict) and not c.get("code_likely")]
        for c in ordered:
            depth = c.get("decode_depth")
            why = []
            if not c.get("aligned"):
                why.append("unaligned")
            if isinstance(depth, int):
                why.append(f"decode={depth}")
            tag = "code-likely" if c.get("code_likely") else "weak"
            lines.append(
                f"  {c.get('address', '?')}  [{c.get('section') or '?'}]  "
                f"via {c.get('provenance', '?')}  [{tag}: {', '.join(why)}]")
        lines.append("")
        lines.append("(candidates -- NOT confirmed functions. `decode` = clean instructions "
                     "before an undefined one; a low decode reliably means data. Start with the "
                     "code-likely subset; verify with `disasm`, then `function create`.)")
    return "\n".join(lines)


def _render_call_descriptors_text(value: Any) -> str:
    """#469: one line per callsite of a registration API -- the declared descriptor
    field values (constants + resolved callback symbols), with unknown/computed
    fields marked explicitly rather than omitted."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    callee = value.get("callee", "<unknown>")
    lines = [f"descriptors passed to {callee} (arg {value.get('arg_index', '?')}): "
             f"{value.get('total', 0)} callsite(s)"]
    for warning in list(value.get("warnings") or []):
        lines.append(f"warning: {warning}")
    for row in list(value.get("items") or []):
        if not isinstance(row, dict):
            lines.append(_render_fallback_text(row))
            continue
        head = f"caller={row.get('caller', '?')} call={row.get('call_address', '?')}"
        status = row.get("status")
        # arg_out_of_range / not_a_local_descriptor: no fields to show. no_field_writes:
        # the descriptor was filled some other way (memcpy/template) -- still show the
        # (all-unknown) fields so the layout attempt is visible.
        if status in ("arg_out_of_range", "not_a_local_descriptor"):
            lines.append(f"{head} [{status}]")
            continue
        parts = []
        for f in list(row.get("fields") or []):
            if not isinstance(f, dict):
                continue
            name = f.get("name", "?")
            st = f.get("status")
            if st == "resolved":
                sym = f.get("symbol")
                mark = "~" if f.get("via") == "sibling_slot" else ""   # heuristic recovery
                val = f"{f.get('value')}" + (f" ({sym})" if sym else "")
                parts.append(f"{name}={mark}{val}")
            elif st == "computed":
                parts.append(f"{name}=<computed>")
            else:
                parts.append(f"{name}=<unknown>")
        suffix = "  [no field writes recovered -- memcpy/template init?]" if status == "no_field_writes" else ""
        lines.append(f"{head} " + " ".join(parts) + suffix)
    return "\n".join(lines)


def _render_virtual_call_text(value: Any) -> str:
    """#466: resolve an imported virtual call to provider vtable method(s)."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    factory = value.get("factory") or "<unresolved factory>"
    head = (f"virtual call @ {value.get('callsite', '?')} in {value.get('caller', '?')}: "
            f"vtable slot {value.get('slot_offset', '?')} (index {value.get('slot_index', '?')}), "
            f"object from {factory}")
    lines = [head]
    cands = list(value.get("candidates") or [])
    if not cands:
        # #531: an unresolved slot (e.g. an unaligned offset that can't map to a slot
        # index) carries a concrete reason -- surface it instead of the generic hint.
        reason = value.get("unresolved_reason")
        if reason:
            lines.append(f"  unresolved: {reason}")
        else:
            lines.append("  no provider class implements this slot "
                         "(check --providers, or the slot is beyond the recovered vtable)")
        return "\n".join(lines)
    if value.get("ambiguous"):
        lines.append(f"  AMBIGUOUS: {len(cands)} provider classes implement slot "
                     f"{value.get('slot_offset', '?')}")
    for c in cands:
        method = c.get("method") or "<unnamed>"
        entry = c.get("vtable_entry") or "?"
        # #533: include the concrete jump target (method_address) -- the pointer's
        # VALUE, distinct from vtable_entry (the slot's address). Render as hex when
        # present; tolerate None/absent (unrecovered target) without crashing.
        ma = c.get("method_address")
        if isinstance(ma, int):
            ma = hex(ma)
        target = str(ma) if ma else "?"
        lines.append(f"  {c.get('class', '?')}  ->  {method} @ {target}"
                     f"   [{c.get('provider', '?')} vtable {c.get('vtable', '?')} @ {entry}]")
    return "\n".join(lines)


def _render_record_table_text(value: Any) -> str:
    """#455: render a mixed-record dispatch table -- one block per record, each
    field labeled fn / data / scalar / null so a scalar isn't read as a bad slot."""
    lines = [
        f"record table @ {value.get('address', '<unknown>')}  "
        f"record-size: {value.get('record_size', '?')}  "
        f"ptr-fields: {', '.join(value.get('ptr_fields') or []) or '(none)'}"
    ]
    for warning in list(value.get("warnings") or []):
        lines.append(f"warning: {warning}")
    for row in list(value.get("items") or []):
        if not isinstance(row, dict):
            lines.append(_render_fallback_text(row))
            continue
        lines.append("")
        lines.append(f"[{row.get('row', '?')}] {row.get('base', '<unknown>')}")
        for f in list(row.get("fields") or []):
            if not isinstance(f, dict):
                continue
            off = f.get("offset", 0)
            off_s = f"+{off:#x}" if isinstance(off, int) else f"+{off}"
            kind = f.get("kind")
            # #467: a DECLARED typed field carries a name; show it so the record reads
            # as its struct fields (fn fields keep BN's resolved callee in `name`).
            fname = f.get("name") if kind in ("scalar", "char_array") else None
            nm = f" {fname}" if fname else ""
            if kind == "function_pointer":
                lines.append(f"  {off_s:<6} fn      {f.get('target', '?')}  {f.get('name') or ''}".rstrip())
            elif kind == "data_pointer":
                note = f'  "{f["preview"]}"' if f.get("preview") else (f"  {f['symbol']}" if f.get("symbol") else "")
                lines.append(f"  {off_s:<6} data    {f.get('target', '?')}{note}")
            elif kind == "char_array":  # #467 inline string field
                lines.append(f'  {off_s:<6} char[{f.get("size", "?")}]{nm}  "{f.get("value", "")}"'.rstrip())
            elif kind == "scalar":
                typ = f.get("type")
                if typ and str(typ).startswith("i") and isinstance(f.get("value"), int):
                    # #467: a SIGNED (i*) typed field shows its decimal value (+ hex),
                    # so -100 isn't rendered as 0xff9c and confused with 65436.
                    val_s = f"{f['value']} ({f.get('hex')})"
                elif f.get("hex"):
                    val_s = f["hex"]                       # unsigned typed field
                else:
                    val_s = f.get("value", "?")            # auto-scalar gap (hex string)
                lines.append(f"  {off_s:<6} scalar{nm}  {val_s}  ({f.get('size', '?')}B)")
            elif kind == "null":
                lines.append(f"  {off_s:<6} null")
            else:  # unmapped / unreadable
                lines.append(f"  {off_s:<6} {kind:<7} {f.get('value', '')}".rstrip())
    return "\n".join(lines)


def _render_pointer_table_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    if value.get("kind") == "record_table":  # #455 mixed-record mode
        return _render_record_table_text(value)
    lines = [
        f"pointer table @ {value.get('address', '<unknown>')}",
        f"pointer-size: {value.get('pointer_size', '<unknown>')}  stride: {value.get('stride', '<unknown>')}"
        f"  read-width: {value.get('read_width', value.get('pointer_size', '<unknown>'))}",
    ]
    suffix = _context_suffix(value.get("context"))
    if suffix:
        lines.append(f"context{suffix}")
    for warning in list(value.get("warnings") or []):
        lines.append(f"warning: {warning}")
    lines.append("")
    for item in list(value.get("items") or []):  # #275: was `entries`
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
    for match in list(value.get("items") or []):  # #275: was `matches`
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
    # Resolved RTTI data symbols (the real vtable/typeinfo the lens targets, #194)
    for sym in list(value.get("rtti_symbols") or []):
        if not isinstance(sym, dict):
            continue
        xr = sym.get("xrefs") if isinstance(sym.get("xrefs"), dict) else {}
        cc = len(list(xr.get("code_refs") or []))
        dc = len(list(xr.get("data_refs") or []))
        lines.append("")
        lines.append(f"rtti {sym.get('kind', '?')}: {sym.get('symbol', '')} @ {sym.get('address', '?')}"
                     f"  xrefs: {cc} code, {dc} data")
        tw = sym.get("table_window")
        if isinstance(tw, dict):
            # #303: the table window is the #275 envelope keyed on `items`; the
            # pre-#275 `entries` key always read 0, so a resolved RTTI vtable
            # window falsely rendered "(0 slots)" in text while the JSON carried
            # the real slots. (entries fallback for any legacy producer.)
            slot_count = len(list(tw.get("items") or tw.get("entries") or []))
            lines.append(f"    vtable window @ {tw.get('address', '?')} "
                         f"({slot_count} slots)")
    for hint in list(value.get("hints") or []):
        lines.append(f"hint: {hint}")
    return "\n".join(lines)


def _render_fanout_text(value: Any, inner_renderer: Callable[[Any], str] | None = None) -> str:
    """Render an --all-instances fan-out (#169 L1): a header per instance, then
    that instance's result rendered by the command's own text renderer (or a
    fallback), and an ``error:`` line for instances that couldn't be reached/
    resolved. Failures are per-instance rows, not a hard failure."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    rows = value.get("instances") or []
    ok = sum(1 for r in rows if isinstance(r, dict) and r.get("ok"))
    lines = [f"fan-out: {value.get('command', '?')} — {len(rows)} result(s) "
             f"({ok} ok, {len(rows) - ok} failed)"]
    expanded = value.get("auto_expanded_instances")
    if expanded:
        # #368: be explicit that a multi-target instance was surveyed in full, so
        # extra rows for one instance read as complete coverage, not a duplicate.
        lines.append(f"  (surveyed all targets of multi-target instance(s): {', '.join(map(str, expanded))})")
    slow = value.get("slow_rows")
    if slow:
        # #417: show where a broad survey spent its time so a long fan-out reads as
        # progress (which instance was slow), not a wedge.
        parts = []
        for s in slow:
            if not isinstance(s, dict):
                continue
            tgt = f"/{s['target']}" if s.get("target") else ""
            parts.append(f"{s.get('instance', '?')}{tgt} {s.get('duration_ms', '?')}ms")
        if parts:
            lines.append(f"  slowest: {', '.join(parts)}")
    for r in rows:
        if not isinstance(r, dict):
            continue
        header = f"\n== instance {r.get('instance', '?')}"
        if r.get("target"):
            header += f"  (target {r['target']})"
        header += " =="
        lines.append(header)
        if r.get("ok"):
            inner = r.get("result")
            if inner_renderer is not None:
                try:
                    lines.append(inner_renderer(inner))
                except Exception:
                    lines.append(_render_fallback_text(inner))
            else:
                lines.append(_render_fallback_text(inner))
        else:
            lines.append(f"  error: {r.get('error', '<unknown>')}")
    return "\n".join(lines)


def _render_orient_text(value: Any) -> str:
    """Render the orientation digest (#169 L2) as a compact triage card: analysis
    state up front (so an empty strings/function set from a --quick view isn't
    trusted), then function count, imports summary, sections, and the bounded
    strings sample."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    target = value.get("target") or {}
    name = target.get("basename") or target.get("filename") or target.get("name") or "<target>"
    state = value.get("analysis_state") or ("full" if value.get("analyzed") else "?")
    lines = [f"orientation: {name}  [analysis: {state}]"]
    if not value.get("analyzed", True):
        lines.append("  ! loaded with --quick — run `bn refresh` before trusting strings/functions")
    fc = value.get("function_count")
    if fc is not None:
        lines.append(f"  functions: {fc}")
    imp = value.get("imports_summary") or {}
    if isinstance(imp, dict):
        total = imp.get("total_symbols", imp.get("total"))
        by_kind = imp.get("by_kind") or {}
        kinds = ", ".join(f"{k}={v}" for k, v in list(by_kind.items())[:6])
        lines.append(f"  imports: {total if total is not None else '?'}" + (f" ({kinds})" if kinds else ""))
    secs = value.get("sections") or {}
    sec_items = secs.get("items") if isinstance(secs, dict) else None
    if isinstance(sec_items, list):
        names = " ".join(str(s.get("name", "?")) for s in sec_items[:12] if isinstance(s, dict))
        lines.append(f"  sections: {secs.get('total', len(sec_items))}  {names}")
    ss = value.get("strings_sample") or {}
    if isinstance(ss, dict) and ss.get("unavailable"):
        lines.append(f"  strings: unavailable — {ss['unavailable']}")
    elif isinstance(ss, dict):
        items = ss.get("items") or []
        # Disclose the min-length filter so orient's total reconciles with the
        # `bn strings` total (which uses a lower default) (#357).
        mn = value.get("strings_min_length")
        filt = f"min-length {mn}; " if mn is not None else ""
        lines.append(f"  strings ({filt}sample {len(items)} of {ss.get('total', len(items))}):")
        for s in items[:15]:
            if isinstance(s, dict):
                # `or ''` (not just the .get default) guards an explicit value:None.
                lines.append(f"    {s.get('address', '?')}  {(s.get('value') or '')[:80]!r}")
    return "\n".join(lines)


def _render_init_arrays_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    sections = list(value.get("items") or [])  # #275: was `sections`
    if not sections:
        # #448: on a `.so` you'd expect constructors, so a bare "none" reads like a
        # possible miss. State the authoritative reason so an empty result is
        # self-evidently correct, not a suspected gap.
        return "init arrays: none (no DT_INIT / DT_INIT_ARRAY present)"
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
        for item in list(table.get("items") or []):  # #275: embedded table is canonical too
            if not isinstance(item, dict):
                continue
            prefix = f"  [{item.get('index', '?'):>2}] {item.get('entry_address', '<unknown>')}"
            if not item.get("readable", True):
                lines.append(f"{prefix}  <unreadable>")
                continue
            lines.append(f"{prefix}  {item.get('value', '<unknown>')} -> {_render_target_line(item.get('target'))}")
    return "\n".join(lines)


def _render_callsites_text(value: Any, *, prefer_caller_static: bool = False) -> str:
    # callsites returns the {items,total,...} envelope (#131 / item 11). Keep the
    # paging metadata (#454: callsites now pages bridge-side like xrefs) so a
    # truncated high-fan-in survey states the true total + remainder in a footer.
    total = None
    has_more = False
    if isinstance(value, dict) and "items" in value:
        total = value.get("total")
        has_more = bool(value.get("has_more"))
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
    body = "\n\n".join(blocks)
    if has_more and isinstance(total, int):
        body += (
            f"\n\n... showing {len(value)} of {total} callsites; "
            "use --offset/--limit to page (or --format json for all)"
        )
    return body


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
    if kind == "pointer_escape":
        return (
            f"  pointer_escape @ {leaf.get('address')}"
            f"  buffer={leaf.get('buffer', '?')}"
            + (f"  {leaf.get('dest')}" if leaf.get("dest") else "")
            + (f"  -- {leaf.get('detail')}" if leaf.get("detail") else "")
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
    if kind == "arg_under_recovered":
        cal = leaf.get("callee") or {}
        return (
            f"  arg_under_recovered @ {leaf.get('address')}"
            f"  -> {cal.get('name', '?')} @ {cal.get('address', '?')}"
            f"  (recovered {leaf.get('recovered_params', '?')} param(s); "
            f"dropped arg(s) {leaf.get('dropped_args', [])})"
            + (f"  -- {leaf.get('note')}" if leaf.get("note") else "")
        )
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
    if kind == "arg_under_recovered":
        return (kind, (leaf.get("callee") or {}).get("name", "?"))
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
    hdr = f"frontiers ({total}"
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


def _taint_forward_verdict(value: dict[str, Any]) -> str:
    """One-line verdict for a forward-taint result, derived from existing fields."""
    findings = value.get("reached_sinks") or []
    leaves = value.get("leaves") or []
    stats = value.get("stats") or {}
    fns = stats.get("functions_visited")
    fns_part = f" · taint crossed {fns} fn(s)" if fns else ""
    trunc = f" · truncated @depth {stats.get('max_depth')}" if stats.get("truncated") else ""
    if findings:
        classes = ", ".join(sorted({(f.get("sink") or {}).get("class") or "?" for f in findings}))
        return f"verdict: {len(findings)} sink(s) reached ({classes}){fns_part}{trunc}"
    if leaves:
        return (f"verdict: NO modeled sink reached — {len(leaves)} tainted frontier(s) "
                f"(NOT an all-clear){fns_part}{trunc}")
    # Genuinely empty: no sink AND no frontier. This is the MOST caveated case, not
    # the least -- the engine reaching nothing does not mean the function is safe;
    # it is exactly the shape a structurally-invisible bug (use-after-free,
    # temporal, or an unmodeled source) produces. Carry the same "NOT an
    # all-clear" qualifier the partial-coverage paths do (#310).
    visited = f"visited {fns} fn(s)" if fns else "shallow coverage"
    return (f"verdict: no taint reached any sink or frontier — NOT an all-clear "
            f"({visited}; no modeled sink or tainted frontier found — also how a bug "
            f"the engine can't structurally see appears){trunc}")


def _taint_via_trail(value: dict[str, Any], finding: dict[str, Any]) -> str | None:
    """Compact callee trail for a sink, parsed from its path-step reasons:
    `<analyzed fn> → <callee> → … → <sink callee>`. None if no callees parse."""
    chain: list[str] = []
    fn = (value.get("function") or {}).get("name")
    if fn:
        chain.append(str(fn))
    for step in finding.get("path") or []:
        if not isinstance(step, dict):
            continue
        reason = str(step.get("reason") or "")
        m = re.search(r"calls (\S+) with tainted", reason)
        if not m:
            m = re.search(r"tainted arg\d+ reaches (\S+)", reason)
        if m:
            name = m.group(1)
            if not chain or chain[-1] != name:
                chain.append(name)
    return ("via: " + " → ".join(chain)) if len(chain) >= 2 else None


def _render_flow_line(f: dict[str, Any]) -> str:
    """One compact line for a forward finding: sink + address + arg + grouping
    signature + structural metrics. The full SSA path is shown only under --full."""
    sink = f.get("sink") or {}
    ai = sink.get("tainted_arg_index")
    arg = f" (arg {ai})" if ai is not None else ""
    m = f.get("metrics") or {}
    sig = (f.get("signature") or {}).get("rendered", "")
    unresolved = "y" if m.get("traverses_unresolved") else "n"
    facts = f"{{steps={m.get('steps', '?')} fns={m.get('fns_spanned', '?')} unresolved={unresolved}}}"
    head = f"[{sink.get('class') or '?'}] {sink.get('callee', '?')} @ {sink.get('address')}{arg}"
    return f"  {head}   {sig}   {facts}".rstrip()


def _render_taint_text(value: Any, full: bool = False) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    fn = value.get("function") or {}
    direction = value.get("direction", "forward")
    lines = [f"{direction} taint in {fn.get('name', '<unknown>')} @ {fn.get('address', '<unknown>')}"]

    if direction == "forward":
        srcs = value.get("sources") or []
        lines.append("sources: " + (", ".join(_describe_loc(s) for s in srcs) or "<none>"))
        findings = list(value.get("reached_sinks") or [])
        lines.append(_taint_forward_verdict(value))
        if findings:
            lines.append("")
            lines.append(f"flows ({len(findings)}):")
            for f in findings:
                # One compact line per flow by default (signature + metrics). Same-sink
                # findings are already unique per (callee,address,arg), so distinct sink
                # call-sites always render on their own line -- never folded behind a
                # count. --full appends the sink detail, via: trail, and full SSA path.
                lines.append(_render_flow_line(f))
                if full:
                    _detail = (f.get("sink") or {}).get('detail') or ''
                    if _detail:
                        lines.append(f"      -- {_detail}")
                    _via = _taint_via_trail(value, f)
                    if _via:
                        lines.append(f"    {_via}")
                    lines.extend(_render_taint_path(f.get("path") or []))
    else:
        sinks = value.get("sinks") or []
        lines.append("sinks: " + (", ".join(_describe_loc(s) for s in sinks) or "<none>"))
        slices = list(value.get("slices") or [])
        for sl in slices:
            sink = sl.get("sink") or {}
            origin = sl.get("origin") or {}
            m = sl.get("metrics") or {}
            sig = (sl.get("signature") or {}).get("rendered", "")
            n = sl.get("reached_via_call_sites", 1)
            xn = f"  (x{n} callsites)" if n and n > 1 else ""
            unresolved = "y" if m.get("traverses_unresolved") else "n"
            lines.append("")
            # Compact one line per slice by default; (xN) surfaces the engine's existing
            # per-(seed,sink,origin) call-site count. --full shows origin/crosses/steps.
            lines.append(
                f"  [{sink.get('class') or sink.get('kind') or '?'}] "
                f"{sink.get('callee') or sink.get('kind') or '?'} @ {sink.get('address')} "
                f"(seed {sink.get('seed', '?')})   {sig}   "
                f"{{steps={m.get('steps', '?')} fns={m.get('fns_spanned', '?')} "
                f"unresolved={unresolved}}}{xn}".rstrip())
            if full:
                lines.append(
                    f"  slice for {sink.get('callee') or sink.get('kind') or '?'} @ {sink.get('address')} (seed {sink.get('seed', '?')}):"
                )
                _ok = origin.get("kind")
                if _ok == "constant" and origin.get("value") is not None:
                    _val = origin["value"]
                    _vs = f"{_val:#x}" if isinstance(_val, int) else str(_val)
                    _extra = _vs + (f" ({origin['var']})" if origin.get("var") else "")
                else:
                    _extra = origin.get("callee") or origin.get("var") or ""
                _spill = " (via spill)" if origin.get("via_spill") else ""
                lines.append(f"  origin: {_ok} {_extra}{_spill}".rstrip())
                crossed = sl.get("crossed_functions") or []
                if crossed:
                    lines.append(f"  crosses: {' <- '.join(crossed)}")
                for step in sl.get("slice") or []:
                    if isinstance(step, dict):
                        lines.append(f"  {step.get('address')}  {step.get('op')}  {step.get('il_text', '')}".rstrip())
        status = value.get("sink_status") or []
        # A constant-length sink is "provably bounded" -- a SUCCESS, not a failed
        # seed -- so report it apart from genuinely-unseeded sinks (#310).
        bounded = [s for s in status if s.get("bounded")]
        unseeded = [s for s in status if not s.get("seeded", True) and not s.get("bounded")]
        if bounded:
            lines.append("")
            lines.append(f"provably bounded ({len(bounded)}):")
            for s in bounded:
                lines.append(f"  {_describe_loc(s)} -- {s.get('note', 'constant length, nothing to slice')}")
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
        lines.append(f"caveats ({len(assumptions)}):")
        for a in assumptions:
            lines.append(f"  - {a}")
    msrc = _render_model_sources(value.get("model_sources"))
    if msrc:
        lines.append("")
        lines.append(msrc)
    if value.get("soundness"):
        lines.append("")
        lines.append(f"soundness: {value['soundness']}")
    return "\n".join(lines)


def _render_taint_models_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    lines: list[str] = []
    srcs = value.get("sources") or []
    if srcs:
        lines.append(f"sources ({len(srcs)}):")
        for s in srcs:
            p = " [present]" if s.get("present") else (" [absent]" if "present" in s else "")
            lines.append(f"  {s['symbol']}  ->  {s.get('to', '')}{p}")
    sbc = value.get("sinks_by_class") or {}
    if sbc:
        lines.append("")
        total = sum(len(v) for v in sbc.values())
        lines.append(f"sinks ({total} in {len(sbc)} class(es)):")
        for cls, lst in sbc.items():
            lines.append(f"  [{cls}]")
            for e in lst:
                cs = f" ({e['callsites']} callsites)" if e.get("callsites") is not None else ""
                p = " [present]" if e.get("present") else (" [absent]" if "present" in e else "")
                addrs = ("  " + ", ".join(e["addresses"])) if e.get("addresses") else ""
                lines.append(f"    {e['symbol']} (arg {e.get('tainted_args')}){p}{cs}{addrs}")
    props = value.get("propagators") or []
    if props:
        lines.append("")
        lines.append(f"propagators ({len(props)}):")
        for p in props:
            lines.append(f"  {p['symbol']}  {p.get('from_to', '')}")
    ov = value.get("overlays") or []
    if ov:
        lines.append("")
        lines.append("overlays: " + ", ".join(o.get("path", o.get("kind", "?")) for o in ov))
    return "\n".join(lines) if lines else "no models match the filter"


def _render_model_sources(sources: Any) -> str:
    """#415: one-line disclosure of the active taint-model overlays in TEXT mode
    (the default), so an agent can confirm a ``--models`` / ``BN_TAINT_MODELS``
    overlay landed without parsing JSON or restarting the bridge."""
    if not isinstance(sources, list):
        return ""
    parts: list[str] = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        kind = s.get("kind")
        if kind == "builtin":
            parts.append("builtin")
        elif kind == "env_override":
            parts.append(f"env {s.get('env', 'BN_TAINT_MODELS')} ({s.get('path')})")
        elif kind == "override_default":
            parts.append(f"override ({s.get('path')})")
        elif kind == "user":
            loc = s.get("path") or s.get("via", "--models")
            parts.append(f"--models {loc} ({s.get('count', 0)} model(s))")
    return ("models: " + " + ".join(parts)) if parts else ""


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
    excluded = value.get("self_defined_excluded")
    if isinstance(excluded, int) and excluded > 0:
        lines.append(f"self-defined excluded: {excluded}")
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
        row = f"{address}  {size}  {string_type}  {rendered}".rstrip()
        # --probable-format-strings enrichment: surface the recovered printf
        # directives and code-xref count so the survey is scannable without
        # re-reading the JSON. Absent on a plain strings dump.
        directives = item.get("format_directives")
        if isinstance(directives, list) and directives:
            refs = item.get("code_refs")
            suffix = f"  [fmt: {' '.join(str(d) for d in directives)}"
            if isinstance(refs, int):
                suffix += f"; code_refs={refs}"
            suffix += "]"
            row += suffix
        lines.append(row)
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
    or a bare list for back-compat / internal callers (#122). Prefixes a W+X
    verdict (#453) so the security question ("any writable+executable region?")
    has a direct answer instead of being inferred from per-row perms."""
    body = _render_paged_list_text(value, "items", _render_sections_rows)
    if isinstance(value, dict) and value.get("wx_verdict"):
        verdict = value["wx_verdict"]
        if verdict == "wx_sections_present":
            names = value.get("writable_executable_items") or []
            line = f"w+x: {len(names)} section(s): {', '.join(names)}"
        elif verdict == "no_wx_sections_observed":
            line = "w+x: none observed"
        else:  # unknown_insufficient_metadata (#461)
            line = ("w+x: unknown -- section metadata is insufficient (mapped/raw view "
                    "with no segment permissions); NOT an all-clear")
        return line + "\n" + body
    return body


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
        if item.get("stale_engine"):
            lines.append(
                "  stale: loaded engine code (taint/IL modules) is out of date -- "
                "run `bn session restart " + str(item.get("instance_id") or "<id>") + "`")
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
        # Name the type(s) defined, not a bare count -- "which type?" is the first
        # thing an agent needs. Parser bookkeeping (parsed functions/variables) is
        # internal noise and moves out of the default line.
        names = list((item.get("defined_types") or {}).keys())
        if names:
            return f"types_declare {', '.join(names)}"
        return f"types_declare {item.get('count', 0)} types"
    return _render_fallback_text(item)


_BN_CONVENTION_RE = re.compile(r'__convention\("([^"]+)"\)')


def _clean_prototype(proto: Any) -> str | None:
    """Render BN's prototype readably: ``__convention("cdecl")`` -> ``__cdecl``."""
    if not isinstance(proto, str) or not proto:
        return None
    return _BN_CONVENTION_RE.sub(r"__\1", proto).strip()


def _set_prototype_detail(item: dict[str, Any]) -> list[str]:
    observed = item.get("observed")
    proto = observed.get("prototype") if isinstance(observed, dict) else None
    proto = _clean_prototype(proto)
    return ["  " + proto] if proto else []


def _layout_size(layout: Any) -> str | None:
    """Pull the ``size=0x..`` (or decimal) off a rendered type layout's header."""
    if not isinstance(layout, str) or not layout:
        return None
    match = re.search(r"size=(0x[0-9a-fA-F]+|\d+)", layout.splitlines()[0])
    return match.group(1) if match else None


def _layout_field_count(layout: Any) -> int:
    if not isinstance(layout, str):
        return 0
    return sum(1 for line in layout.splitlines()[1:] if line.strip().startswith("0x"))


def _size_delta(before_layout: Any, after_layout: Any) -> str | None:
    after = _layout_size(after_layout)
    if after is None:
        return None
    before = _layout_size(before_layout)
    if before is None or before == after:
        return f"size {after}"
    try:
        delta = int(after, 0) - int(before, 0)
    except (TypeError, ValueError):
        return f"size {before} -> {after}"
    return f"size {before} -> {after} ({'+' if delta >= 0 else ''}{delta})"


def _layout_field_deltas(layout_diff: Any) -> list[str]:
    """The +/- field lines from a unified layout diff (skips the struct-header and
    @@ hunk lines), so a type change shows just the fields that moved."""
    out: list[str] = []
    for line in (layout_diff or "").splitlines() if isinstance(layout_diff, str) else []:
        if len(line) >= 2 and line[0] in "+-" and line[1:].lstrip().startswith("0x"):
            out.append(f"  {line[0]} {line[1:].strip()}")
    return out


def _types_affected_lines(value: dict[str, Any]) -> list[str]:
    entries = [e for e in (value.get("affected_types") or []) if isinstance(e, dict)]
    multi = len(entries) > 1
    out: list[str] = []
    for entry in entries:
        name = entry.get("type_name") or entry.get("name") or "<type>"
        # Only prefix the type name when a batch touched more than one type --
        # for a single type the op-summary header already names it.
        prefix = f"{name}: " if multi else ""
        if entry.get("changed"):
            before_sz = _layout_size(entry.get("before_layout"))
            after_sz = _layout_size(entry.get("after_layout"))
            deltas = _layout_field_deltas(entry.get("layout_diff"))
            if before_sz is not None and after_sz is not None and before_sz != after_sz:
                out.append(f"  {prefix}{_size_delta(entry.get('before_layout'), entry.get('after_layout'))}")
            elif not deltas:
                # No field/size delta to show (e.g. a decl-only change) -- fall back
                # to the size so the line isn't empty.
                out.append(f"  {prefix}{_size_delta(entry.get('before_layout'), entry.get('after_layout')) or 'changed'}")
            # else: size unchanged but fields moved (e.g. a rename) -- the +/- field
            # lines below carry the change; a 'size 0xNN' line would just be noise.
            out.extend(deltas)
        else:
            after = entry.get("after_layout") or ""
            head = after.splitlines()[0].strip() if after.strip() else f"struct {name}"
            count = _layout_field_count(after)
            out.append(f"  {head}, {count} field{'s' if count != 1 else ''}")
    return out


def _blast_radius_line(value: dict[str, Any]) -> str | None:
    """One line of blast radius for a type op: how many functions reference the
    type and how many actually reflowed, with a few names (reflowed first)."""
    summary = value.get("affected_summary")
    if not isinstance(summary, dict):
        return None
    referenced = int(summary.get("referenced") or 0)
    reflowed = int(summary.get("reflowed") or 0)
    if referenced <= 0:
        return None
    # A directly-mutated function (set_prototype/rename target, tagged `direct`)
    # is not part of the type's reference set, so keep it out of these names --
    # in a mixed batch it belongs under the direct op's affected block instead.
    affected = [a for a in (value.get("affected_functions") or [])
                if isinstance(a, dict) and not a.get("direct")]
    names = [a.get("after_name") or a.get("before_name") for a in affected if a.get("changed")]
    names += [a.get("after_name") or a.get("before_name") for a in affected if not a.get("changed")]
    names = [n for n in names if n]
    line = f"  referenced by {referenced} fn{'s' if referenced != 1 else ''}, {reflowed} reflowed"
    if names[:5]:
        line += ": " + ", ".join(names[:5])
        if referenced > len(names[:5]):
            line += f" (+{referenced - len(names[:5])} more)"
    return line


def _is_type_result(result: Any) -> bool:
    """A type-shape result (type (re)declaration or struct field edit), whose
    blast radius is "functions referencing the type" -- as opposed to a direct op
    (rename/prototype/comment) that targets one specific function."""
    if not isinstance(result, dict):
        return False
    op = str(result.get("op") or "")
    return op == "types_declare" or op.startswith("struct_")


def _format_op_summary(item: dict[str, Any]) -> str:
    summary = _format_operation_result(item)
    if item.get("status"):
        summary += f" [{item['status']}]"
    if item.get("changed") is False and item.get("status") not in (None, "noop"):
        summary += " [no change]"
    if item.get("message"):
        summary += f" ({item['message']})"
    return summary


def _add_mutation_ok(value: Any) -> Any:
    """Add a top-level ``ok`` boolean to a full mutation/batch result so a uniform
    ``jq '.ok'`` check works across read and mutation commands (#447). ``ok`` is
    the verification-aware success: the bridge-reported ``success`` AND no failed
    op status. Additive -- ``success``/``committed`` are unchanged."""
    if not isinstance(value, dict) or "ok" in value:
        return value
    results = [r for r in (value.get("results") or []) if isinstance(r, dict)]
    failed = any(r.get("status") in FAILED_MUTATION_STATUSES for r in results)
    return {"ok": bool(value.get("success", True)) and not failed, **value}


def _mutation_summary(value: Any) -> Any:
    """#408: collapse a (single or batch) mutation result into a compact,
    schema-stable status object for an unattended agent control loop -- did
    anything change, did verification pass, was anything rolled back, what needs
    attention -- without parsing the full results/affected_functions/diff payload.
    The detailed result stays available without --summary."""
    if not isinstance(value, dict):
        return value
    results = [r for r in (value.get("results") or []) if isinstance(r, dict)]
    failed = [r for r in results if r.get("status") in FAILED_MUTATION_STATUSES]
    verified = sum(1 for r in results if r.get("status") == "verified")
    noop = sum(1 for r in results if r.get("status") == "noop")
    committed = bool(value.get("committed", False))
    rolled_back = value.get("rolled_back")
    success = bool(value.get("success", True)) and not failed
    first_error = None
    if failed:
        f0 = failed[0]
        first_error = f0.get("message") or f0.get("status") or "mutation failed"
    # A failure can carry its only explanation in the top-level `message` -- e.g. a
    # preview/revert cleanup that failed AFTER every op verified, so no result row
    # is in FAILED_MUTATION_STATUSES. Surface it so --summary never drops the error.
    if first_error is None and not success:
        first_error = value.get("message") or "mutation failed"
    return {
        "kind": "mutation_summary",
        # Top-level `ok` mirrors the read-command envelope so a uniform `jq '.ok'`
        # works across reads AND mutations (batch/mutation JSON used only
        # success/committed, so `.ok` read null -- #447).
        "ok": success,
        "success": success,
        "committed": committed,
        "preview": bool(value.get("preview", False)),
        "op_count": len(results),
        "changed_count": verified,        # ops that actually changed + verified
        "verified_count": verified,
        "noop_count": noop,
        "failed_count": len(failed),
        # True/False when a revert was attempted; None when none was needed.
        "rolled_back": (bool(rolled_back) if rolled_back is not None else None),
        "first_error": first_error,
        # The DB is left modified iff a live mutation actually CHANGED state
        # (committed AND something verified -- `committed` is True even for an
        # all-noop mutation, which leaves the DB clean), or a failure's revert
        # itself failed (rolled_back is explicitly False).
        "dirty_after": (committed and verified > 0) or rolled_back is False,
    }


def _render_mutation_summary_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    state = ("committed" if value.get("committed")
             else "preview" if value.get("preview")
             else "rolled back" if value.get("rolled_back")
             else "ok" if value.get("success") else "FAILED")
    parts = [
        f"mutation: {state}",
        f"changed={value.get('changed_count', 0)}",
        f"verified={value.get('verified_count', 0)}",
        f"noop={value.get('noop_count', 0)}",
        f"failed={value.get('failed_count', 0)}",
        f"dirty_after={value.get('dirty_after')}",
    ]
    line = "  ".join(parts)
    if value.get("first_error"):
        line += f"\nfirst_error: {value['first_error']}"
    return line


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
        # The banner already says applied+reverted; the bridge's matching
        # "Preview verified and reverted." message would only repeat it. (A
        # restore-failure preview is success=False and renders above instead.)
        lines.append("preview: change applied + reverted")
        lines.append("")

    has_type_op = any(_is_type_result(r) for r in results)
    has_direct_op = any(r.get("op") and not _is_type_result(r)
                        for r in results if isinstance(r, dict))

    if results:
        if len(results) == 1 and success and not failed and not preview:
            lines.append(_format_op_summary(results[0]))
        else:
            lines.append(f"results ({len(results)}):")
            for item in results:
                lines.append("- " + _format_op_summary(item))

    # "What landed" detail so a verified mutation confirms itself without a
    # follow-up read: the live prototype for each set_prototype. Emitted per
    # result (not only single-op batches) so a mixed batch still surfaces it.
    if not failed:
        for item in results:
            if item.get("op") == "set_prototype" and item.get("status") not in FAILED_MUTATION_STATUSES:
                lines.extend(_set_prototype_detail(item))

    # Type and direct detail are independent: a mixed batch shows BOTH the type's
    # size/field delta + blast radius AND the direct ops' affected-function diffs.
    # The blast radius replaces the per-function dump for type ops (which was
    # either empty or a wall of unrelated callers).
    if has_type_op:
        lines.extend(_types_affected_lines(value))
        blast = _blast_radius_line(value)
        if blast:
            lines.append(blast)
    if has_direct_op:
        affected_functions = [a for a in (value.get("affected_functions") or []) if isinstance(a, dict)]
        # In a mixed batch, only the direct-op targets belong here -- the type's
        # reflowed callers are summarised by the blast-radius line above. With no
        # type op, every changed function is a direct-op effect.
        changed_functions = [
            a for a in affected_functions
            if a.get("changed") and (a.get("direct") if has_type_op else True)
        ]
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
    "field_load": "field load",
    "call_or_jump_boundary": "call boundary",
    "definition": "definition",
    "phi_source": "phi source",
    "cross_function": "crosses into callee",
    # #416: the value was written by a callee through an out-pointer; backward
    # interprocedural tracing follows return values, not out-parameters.
    "interprocedural_out_param_not_followed": "out-param fill not followed",
}


def _render_trace_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    fn_name = value.get("function", "<unknown>")
    fn_addr = value.get("function_address", "<unknown>")
    target_addr = value.get("target_address", "<unknown>")
    arg_index = value.get("arg_index", 0)
    trace = list(value.get("trace") or [])
    hints = [h for h in (value.get("hints") or []) if h]

    # arg[N] (reg, "name") -- the calling-convention register + C-arg name (#166)
    arg_lbl = value.get("arg_label") or {}
    extra = ", ".join(
        x for x in (arg_lbl.get("register"),
                    (f'"{arg_lbl["name"]}"' if arg_lbl.get("name") else None)) if x)
    arg_desc = f"arg[{arg_index}]" + (f" ({extra})" if extra else "")
    header = f"backward trace of {arg_desc} in {fn_name} @ {target_addr}"
    step_word = "step" if len(trace) == 1 else "steps"
    info = f"  {fn_name} @ {fn_addr}  •  {len(trace)} {step_word}"
    if value.get("truncated"):
        info += "  •  truncated"

    if not trace:
        body = "\n".join(f"  hint: {h}" for h in hints) if hints \
            else "  constant or immediate — no SSA trace"
        return f"{header}\n{info}\n\n{body}"

    lines = [header, info, ""]
    current_fn: str | None = None
    for step in trace:
        if not isinstance(step, dict):
            lines.append(f"  {step}")
            continue

        fn_ctx = step.get("function_context")
        cross_fn = step.get("cross_function")
        ssa_var = step.get("ssa_label") or step.get("ssa_var", "")
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
            # Name the resolved callee at a call boundary so a library-call origin
            # reads as `call boundary (strlen)` not a bare PLT address (#193).
            if reason == "call_or_jump_boundary" and step.get("callee"):
                line += f" ({step['callee']})"
            if reason == "interprocedural_out_param_not_followed" and step.get("out_param_callee"):
                line += f" (via {step['out_param_callee']})"
            if reason == "field_load":
                meta = " ".join(
                    f"{k}={step[k]}" for k in ("base", "offset", "width") if step.get(k) is not None)
                if meta:
                    line += f"  [{meta}]"
            if il_text:
                line += f"  @ {addr}  {il_text}" if addr else f"  {il_text}"
            lines.append(line)
        else:
            if addr:
                lines.append(f"  {addr}  {il_text}")
            else:
                lines.append(f"  {il_text}")

    for h in hints:
        lines.append(f"  hint: {h}")
    return "\n".join(lines)


def _render_class_list_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    # #484 count-only: a bare count envelope (no items), with the non-class artifact
    # (#481) share broken out so the domain-class count is honest.
    if "count" in value and not value.get("items") and not value.get("classes"):
        n = value.get("count", 0)
        art = value.get("artifact_count") or 0
        tail = f" ({art} non-class RTTI/type artifact{'s' if art != 1 else ''})" if art else ""
        return f"classes: {n}{tail}"
    rows = list(value.get("items") or value.get("classes") or [])
    total = value.get("total", len(rows))
    header = f"classes: {len(rows)} shown of {total}"
    # Surface what was folded out so the count is self-documenting (#205/#309).
    hidden_parts = []
    cv = value.get("construction_vtables_suppressed") or 0
    if cv:
        hidden_parts.append(f"{cv} construction-vtable artifact{'s' if cv != 1 else ''} (--all to show)")
    th = value.get("thunks_suppressed") or 0
    if th:
        hidden_parts.append(f"{th} thunk{'s' if th != 1 else ''}")
    lib = value.get("library_suppressed") or 0
    if value.get("no_stl") and lib:
        hidden_parts.append(f"{lib} library/STL")
    ven = value.get("vendor_suppressed") or 0
    if value.get("no_vendor") and ven:
        hidden_parts.append(f"{ven} vendored")
    if hidden_parts:
        header += " (hidden: " + ", ".join(hidden_parts) + ")"
    lines = [header]
    for rec in rows:
        if not isinstance(rec, dict):
            lines.append(_render_fallback_text(rec))
            continue
        vt = "vtable" if rec.get("has_vtable") else "no-vtable"
        size = rec.get("size")
        size_s = size.get("value") if isinstance(size, dict) else size
        bases = ", ".join(b for b in (rec.get("bases") or []) if b)
        base_s = f"  : {bases}" if bases else ""
        # #481: mark a non-class RTTI/type-signature artifact (rtti confidence but no
        # methods and no vtable) so it doesn't read as a domain class.
        art_s = "  [artifact: non-class RTTI]" if rec.get("artifact") else ""
        lines.append(
            f"  {rec.get('name', '<unknown>')}  "
            f"methods={rec.get('method_count', 0)}  {vt}  "
            f"size={size_s if size_s is not None else '?'}  "
            f"[{rec.get('confidence', '?')}]{base_s}{art_s}"
        )
    return "\n".join(lines)


def _vtable_slot_label(s: dict[str, Any]) -> str:
    """Label for a vtable slot: the demangled method, `__cxa_pure_virtual`, a
    named external (cross-module) slot, `<null>`, or `<unnamed>` (#441)."""
    if s.get("pure_virtual"):
        return "__cxa_pure_virtual"
    method = s.get("method") if isinstance(s.get("method"), dict) else {}
    slot_name = method.get("display_name") or method.get("name")
    if slot_name:
        return slot_name
    if s.get("external"):
        ext = s.get("external_name")
        return f"{ext} [external]" if ext else "<external>"
    if s.get("null"):
        return "<null>"
    return "<unnamed>"


def _render_class_show_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    if value.get("ambiguous"):
        out = [f"ambiguous class {value.get('query', '')!r}: {len(value.get('matches') or [])} matches"]
        for rec in value.get("matches") or []:
            out.append("")
            out.append(_render_one_class(rec))
        return "\n".join(out)
    return _render_one_class(value)


def _render_one_class(rec: Any) -> str:
    if not isinstance(rec, dict):
        return _render_fallback_text(rec)
    size = rec.get("size")
    size_s = size.get("value") if isinstance(size, dict) else None
    vt = rec.get("vtable") if isinstance(rec.get("vtable"), dict) else None
    vt_addr = vt.get("address") if vt else None
    bases = ", ".join(b.get("name") or "?" for b in (rec.get("bases") or []))
    head = f"class {rec.get('name', '<unknown>')}"
    bits = []
    if size_s:
        bits.append(f"size {size_s}")
    if vt_addr:
        bits.append(f"vtable @ {vt_addr}")
    if bases:
        bits.append(f"base: {bases}")
    if bits:
        head += "  (" + ", ".join(bits) + ")"
    lines = [head, f"  [{rec.get('confidence', '?')}]"]
    methods = rec.get("methods") or []
    for m in methods:
        if m.get("kind") in ("ctor", "dtor"):
            lines.append(f"  {m['kind']:<6} {m.get('address', '?')}  {m.get('demangled', '')}")
    if vt and vt.get("slots"):
        for s in vt["slots"]:
            lines.append(f"  vtable [{s.get('index')}] {s.get('address', '?')}  {_vtable_slot_label(s)}")
    elif vt_addr:
        # A vtable symbol exists but no slots resolved. Either the vtable is
        # defined in another module (the local symbol is an import/GOT slot) or
        # it is a PIE/.data.rel.ro vtable whose pointers are applied at load time
        # via relocations (zero in the static image). In both cases there is no
        # decodable local body; say so rather than render fake or empty virtuals.
        lines.append("  vtable: symbol present but no slots resolved here "
                     "(defined in another module, or applied at load time via relocations)")
    # #412: secondary (multiple-inheritance) vtables -- shown compactly so a simple
    # single-inheritance class isn't cluttered (there are none to show there).
    for sec in rec.get("secondary_vtables") or []:
        if not isinstance(sec, dict):
            continue
        ott = sec.get("offset_to_top")
        ott_s = f" (offset-to-top {ott})" if ott is not None else ""
        lines.append(f"  secondary vtable @ {sec.get('address', '?')}{ott_s}:")
        for s in sec.get("slots") or []:
            lines.append(f"    [{s.get('index')}] {s.get('address', '?')}  {_vtable_slot_label(s)}")
    # Non-virtual member functions (kind=method). Virtual ones already appear as
    # vtable slots above; listing the symbol-side methods makes `class show`
    # useful for classes whose vtable is empty or absent (e.g. Controller).
    member_methods = [m for m in methods if m.get("kind") == "method"]
    if member_methods:
        lines.append(f"  methods ({len(member_methods)}):")
        for m in member_methods:
            lines.append(f"    {m.get('address', '?')}  {m.get('demangled', '')}")
    inst = rec.get("instances") if isinstance(rec.get("instances"), dict) else {}
    parts = []
    for site in inst.get("construction_sites") or []:
        sz = f" (size {site['size']})" if site.get("size") else ""
        fn = f" (in {site['function']})" if site.get("function") else ""
        parts.append(f"{site.get('kind', '?')} @ {site.get('address', '?')}{sz}{fn}")
    for g in inst.get("stored_globals") or []:
        parts.append(f"stored -> {g.get('symbol') or '?'} @ {g.get('address', '?')}")
    if parts:
        lines.append("  instances: " + " ; ".join(parts))
    return "\n".join(lines)
