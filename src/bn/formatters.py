from __future__ import annotations

import json
from typing import Any, Callable

FAILED_MUTATION_STATUSES = {"unsupported", "verification_failed"}


def _render_fallback_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True)


def _format_local_entry(item: dict[str, Any]) -> str:
    name = str(item.get("name", "<unknown>"))
    type_str = str(item.get("type", "<unknown>"))
    return f"  {name:<20} {type_str}"


def _text_field(field: str) -> Callable[[Any], str]:
    def render(value: Any) -> str:
        if isinstance(value, dict):
            text = value.get(field)
            if isinstance(text, str):
                return text
        return _render_fallback_text(value)

    return render


def _render_function_info_text(value: Any, verbose: bool = False) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    function = value.get("function") or {}
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
    lines = [f"loaded: {value.get('path', '<unknown>')}"]
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
                    lines.append(f"- {item.get('path', '<unknown>')}")
                for note in item.get("notes") or []:
                    lines.append(f"  note: {note}")
            else:
                lines.append(f"- {_render_fallback_text(item)}")
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


def _render_name_address_list_text(value: Any) -> str:
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
        lines.append(line)
    return "\n".join(lines)


def _group_refs_by_caller(refs: list[Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    order: list[tuple[str | None, str | None]] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        caller = ref.get("caller_function") if isinstance(ref.get("caller_function"), dict) else None
        key: tuple[str | None, str | None]
        if caller is not None:
            key = (caller.get("address"), caller.get("name"))
        else:
            key = (None, ref.get("function"))
        if key not in groups:
            groups[key] = {
                "caller_address": key[0],
                "caller_name": key[1],
                "sites": [],
            }
            order.append(key)
        groups[key]["sites"].append(str(ref.get("address", "<unknown>")))
    return [groups[k] for k in order]


def _render_xrefs_text(value: Any, limit: int | None = None) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    code_refs = list(value.get("code_refs") or [])
    data_refs = list(value.get("data_refs") or [])
    total_code = len(code_refs)
    total_data = len(data_refs)

    def _render_group(refs: list[Any], total: int, label: str) -> list[str]:
        groups = _group_refs_by_caller(refs)
        if not groups:
            return [f"{label}:", "- none"]
        site_word = "site" if total == 1 else "sites"
        fn_word = "function" if len(groups) == 1 else "functions"
        header = f"{label}: {total} {site_word} across {len(groups)} {fn_word}"
        shown = groups[:limit] if limit else groups
        rendered = [header]
        for group in shown:
            caller_addr = group["caller_address"] or "<unknown>"
            caller_name = group["caller_name"] or "<unknown>"
            sites = group["sites"]
            if len(sites) == 1:
                suffix = f"(1 site: {sites[0]})"
            else:
                suffix = f"({len(sites)} sites: {', '.join(sites)})"
            rendered.append(f"  {caller_addr}  {caller_name}  {suffix}")
        if limit and len(groups) > limit:
            rendered.append(f"  ... {len(groups) - limit} more functions (use --limit or --format json)")
        return rendered

    lines = [f"xrefs to {value.get('address', '<unknown>')} ({total_code} code, {total_data} data)", ""]
    lines.extend(_render_group(code_refs, total_code, "code refs"))
    lines.append("")
    lines.extend(_render_group(data_refs, total_data, "data refs"))
    return "\n".join(lines)


def _render_callsites_text(value: Any, *, prefer_caller_static: bool = False) -> str:
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
        primary = (
            f"caller_static {caller_static} | call {call_addr}"
            if prefer_caller_static
            else f"call {call_addr} | caller_static {caller_static}"
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


def _render_type_list_text(value: Any) -> str:
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


def _render_strings_text(value: Any) -> str:
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


def _render_sections_text(value: Any) -> str:
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
        status = "ok" if doctor and not doctor.get("error") else "error"
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
        prefix = "result:\n" if parts else "result:\n"
        parts.append(prefix + body)

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
