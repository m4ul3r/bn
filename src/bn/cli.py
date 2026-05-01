from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from . import session_state
from .output import render_artifact_envelope, write_output_result
from .paths import (
    claude_skills_dir,
    codex_home,
    codex_skills_dir,
    plugin_install_dir,
    plugin_source_dir,
    repo_root,
)
from .transport import (
    BridgeError,
    _send_request_to_instance,
    instance_selector,
    list_instances,
    send_request,
    spawn_instance,
)
from .version import VERSION, build_id_for_file

FAILED_MUTATION_STATUSES = {"unsupported", "verification_failed"}


class _HelpFullAction(argparse.Action):
    def __init__(
        self,
        option_strings: list[str],
        dest: str = argparse.SUPPRESS,
        default: str = argparse.SUPPRESS,
        help: str | None = None,
    ) -> None:
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help,
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | list[str] | None,
        option_string: str | None = None,
    ) -> None:
        if isinstance(parser, BnArgumentParser):
            parser.print_full_help()
        else:
            parser.print_help()
        parser.exit()


class BnArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.set_defaults(_parser=self)
        self.add_argument(
            "--help-full",
            action=_HelpFullAction,
            help="Show help for this command and all subcommands",
        )

    def _iter_full_help_parsers(self) -> list[argparse.ArgumentParser]:
        parsers: list[argparse.ArgumentParser] = [self]
        for action in self._actions:
            if isinstance(action, argparse._SubParsersAction):
                for parser in action.choices.values():
                    if isinstance(parser, BnArgumentParser):
                        parsers.extend(parser._iter_full_help_parsers())
                    else:
                        parsers.append(parser)
        return parsers

    def _full_help_actions(self) -> tuple[type[argparse.Action], ...]:
        return (argparse._HelpAction, _HelpFullAction)

    def format_help_for_full(self) -> str:
        formatter = self._get_formatter()
        help_action_types = self._full_help_actions()
        actions = [action for action in self._actions if not isinstance(action, help_action_types)]

        formatter.add_usage(self.usage, actions, self._mutually_exclusive_groups)
        formatter.add_text(self.description)

        for action_group in self._action_groups:
            group_actions = [
                action
                for action in action_group._group_actions
                if not isinstance(action, help_action_types)
            ]
            if not group_actions:
                continue
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(group_actions)
            formatter.end_section()

        formatter.add_text(self.epilog)
        return formatter.format_help()

    def format_full_help(self) -> str:
        sections: list[str] = []
        seen: set[int] = set()
        for parser in self._iter_full_help_parsers():
            parser_id = id(parser)
            if parser_id in seen:
                continue
            seen.add(parser_id)
            if isinstance(parser, BnArgumentParser):
                sections.append(parser.format_help_for_full().rstrip())
            else:
                sections.append(parser.format_help().rstrip())
        return "\n\n".join(sections) + "\n"

    def print_full_help(self, file: Any = None) -> None:
        if file is None:
            file = sys.stdout
        self._print_message(self.format_full_help(), file)


def _package_version() -> str:
    return VERSION


def _common_io_options(
    parser: argparse.ArgumentParser,
    *,
    default_format: str = "text",
) -> None:
    parser.add_argument(
        "--format",
        choices=("json", "text", "ndjson"),
        default=default_format,
        help="Output format",
    )
    parser.add_argument("--out", type=Path, help="Write output to a file instead of stdout")


def _instance_option(parser: argparse.ArgumentParser, *, is_root: bool = False) -> None:
    parser.add_argument(
        "--instance",
        default=os.environ.get("BN_INSTANCE") if is_root else argparse.SUPPRESS,
        help="Target a specific bridge instance by ID (env: BN_INSTANCE)",
    )


def _target_option(
    parser: argparse.ArgumentParser,
    *,
    required: bool,
    is_root: bool = False,
) -> None:
    kwargs: dict[str, Any] = {
        "help": (
            "Target selector from `bn target list` (`selector`, `target_id`, basename, filename, or view id); "
            "omit only when exactly one target is open, or use `active` to follow the GUI-selected target explicitly"
        ),
        "required": required,
    }
    if not is_root:
        kwargs["default"] = argparse.SUPPRESS
    parser.add_argument("-t", "--target", **kwargs)


# ---------------------------------------------------------------------------
# Declarative command registration
# ---------------------------------------------------------------------------

_COMMANDS: list[dict[str, Any]] = []

_GROUP_HELP: dict[tuple[str, ...], str] = {
    ("plugin",): "Install the Binary Ninja companion plugin",
    ("skill",): "Install the bundled agent skills",
    ("session",): "Manage bridge sessions",
    ("instance",): "Pin or clear the active bridge instance",
    ("target",): "Inspect Binary Ninja targets",
    ("function",): "Function discovery helpers",
    ("bundle",): "Export reusable bundles",
    ("py",): "Execute Python inside Binary Ninja",
    ("symbol",): "Rename functions or data",
    ("comment",): "Set or delete comments",
    ("proto",): "Inspect or set a user prototype",
    ("local",): "Inspect, rename, or retype locals",
    ("struct",): "Field-first structure editing",
    ("struct", "field"): "Operate on struct fields",
    ("batch",): "Apply a batch manifest",
}


def arg(*flags: str, **kwargs: Any) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Define an argument spec for :func:`command`."""
    return (flags, kwargs)


def mutex(required: bool, *args: tuple[tuple[str, ...], dict[str, Any]]) -> tuple[bool, list[tuple[tuple[str, ...], dict[str, Any]]]]:
    """Define a mutually exclusive argument group for :func:`command`."""
    return (required, list(args))


def command(
    *path: str,
    help: str = "",
    fmt: str = "text",
    target: bool = False,
    paged: bool = False,
    address_filter: bool = False,
    args: list[tuple[tuple[str, ...], dict[str, Any]]] | None = None,
    mutex_groups: list[tuple[bool, list[tuple[tuple[str, ...], dict[str, Any]]]]] | None = None,
) -> Callable:
    """Register a CLI command declaratively."""

    def decorator(fn: Callable[[argparse.Namespace], int]) -> Callable[[argparse.Namespace], int]:
        _COMMANDS.append({
            "path": path,
            "handler": fn,
            "help": help,
            "fmt": fmt,
            "target": target,
            "paged": paged,
            "address_filter": address_filter,
            "args": args or [],
            "mutex_groups": mutex_groups or [],
        })
        return fn

    return decorator


def _build_from_commands(root: BnArgumentParser) -> None:
    """Populate *root* with subcommands from the ``_COMMANDS`` registry."""
    subparser_actions: dict[tuple[str, ...], argparse._SubParsersAction] = {}
    node_parsers: dict[tuple[str, ...], argparse.ArgumentParser] = {(): root}

    def _get_subparsers(parent: tuple[str, ...]) -> argparse._SubParsersAction:
        if parent not in subparser_actions:
            dest = "_".join(parent) + "_command" if parent else "command"
            subparser_actions[parent] = node_parsers[parent].add_subparsers(dest=dest)
        return subparser_actions[parent]

    def _ensure_intermediate(path: tuple[str, ...]) -> argparse.ArgumentParser:
        if path in node_parsers:
            return node_parsers[path]
        if len(path) > 1:
            _ensure_intermediate(path[:-1])
        sub = _get_subparsers(path[:-1])
        parser = sub.add_parser(path[-1], help=_GROUP_HELP.get(path, ""))
        node_parsers[path] = parser
        return parser

    for spec in sorted(_COMMANDS, key=lambda s: len(s["path"])):
        path = spec["path"]
        parent = path[:-1]

        if parent:
            _ensure_intermediate(parent)

        if path in node_parsers:
            cmd = node_parsers[path]
        else:
            cmd = _get_subparsers(parent).add_parser(path[-1], help=spec["help"])
            node_parsers[path] = cmd

        _common_io_options(cmd, default_format=spec["fmt"])
        _instance_option(cmd)
        if spec["target"]:
            _target_option(cmd, required=False)
        if spec["address_filter"]:
            _add_function_address_args(cmd)
        if spec["paged"]:
            _add_paged_args(cmd)

        for flags, kwargs in spec["args"]:
            cmd.add_argument(*flags, **kwargs)

        for required, group_args in spec["mutex_groups"]:
            group = cmd.add_mutually_exclusive_group(required=required)
            for flags, kwargs in group_args:
                group.add_argument(*flags, **kwargs)

        cmd.set_defaults(handler=spec["handler"])


def _render_result(
    value: Any,
    *,
    fmt: str,
    out_path: Path | None,
    stem: str,
    spill_label: str | None = None,
    spill_context: Any = None,
) -> None:
    if out_path is None and isinstance(value, dict) and isinstance(value.get("artifact_path"), str):
        artifact = dict(value)
        artifact.setdefault("ok", True)
        artifact.setdefault("spilled", False)
        sys.stdout.write(render_artifact_envelope(artifact))
        return

    result = write_output_result(value, fmt=fmt, out_path=out_path, stem=stem)
    if result.spilled and result.artifact:
        label = spill_label or stem.replace("_", " ")
        artifact = result.artifact
        sys.stdout.write(result.rendered)
        print(f"warning: {label} output spilled to {artifact['artifact_path']}", file=sys.stderr)
        return
    sys.stdout.write(result.rendered)


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


def _implicit_target(args: argparse.Namespace) -> str:
    response = send_request(
        "list_targets",
        params={},
        target=None,
        instance_id=getattr(args, "instance", None),
    )
    targets = list(response["result"])
    if len(targets) == 1:
        return "active"
    if not targets:
        raise BridgeError("No BinaryView targets are open")
    raise BridgeError(
        "This command requires --target when multiple targets are open.\n"
        f"Open targets:\n{_render_target_choices(targets)}"
    )


def _resolve_target(
    args: argparse.Namespace,
    *,
    require_target: bool,
    allow_implicit_target: bool = False,
) -> str | None:
    target = getattr(args, "target", None)
    if require_target and not target:
        if allow_implicit_target:
            return _implicit_target(args)
        raise BridgeError("This command requires --target")
    return target


def _mutation_exit_code(result: Any) -> int:
    if not isinstance(result, dict):
        return 0
    results = list(result.get("results") or [])
    if any(isinstance(item, dict) and item.get("status") in FAILED_MUTATION_STATUSES for item in results):
        return 3
    if result.get("success") is False:
        return 3
    return 0


def _call(
    args: argparse.Namespace,
    op: str,
    params: dict[str, Any] | None = None,
    *,
    require_target: bool,
    allow_implicit_target: bool = False,
    text_renderer: Callable[[Any], str] | None = None,
    page_limit: int | None = None,
    page_offset: int = 0,
    page_label: str | None = None,
    stem: str,
    result_exit_code: Callable[[Any], int] | None = None,
    bridge_writes_output: bool = False,
) -> int:
    request_params = dict(params or {})
    effective_page_limit = None
    if page_limit is not None and page_limit >= 0:
        effective_page_limit = page_limit
        request_params["limit"] = page_limit + 1

    target = _resolve_target(
        args,
        require_target=require_target,
        allow_implicit_target=allow_implicit_target,
    )
    response = send_request(
        op,
        params=request_params,
        target=target,
        instance_id=getattr(args, "instance", None),
    )
    result = response["result"]
    exit_code = result_exit_code(result) if result_exit_code is not None else 0
    if effective_page_limit is not None and isinstance(result, list) and len(result) > effective_page_limit:
        result = result[:effective_page_limit]
        label = page_label or op
        next_offset = page_offset + effective_page_limit
        print(
            f"warning: {label} output truncated to {effective_page_limit} items; rerun with --offset {next_offset} or a larger --limit",
            file=sys.stderr,
        )
    spill_context = result
    if text_renderer is not None and args.format == "text":
        result = text_renderer(result)
    _render_result(
        result,
        fmt=args.format,
        out_path=None if bridge_writes_output else args.out,
        stem=stem,
        spill_label=page_label or op.replace("_", " "),
        spill_context=spill_context,
    )
    return exit_code


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


def _parse_line_range(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected START:END, got {value!r}")
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected START:END with integers, got {value!r}")
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError(f"invalid range: {start}:{end}")
    return (start, end)


@command("doctor", help="Validate bridge discovery and installation")
def _doctor(args: argparse.Namespace) -> int:
    install_dir = plugin_install_dir()
    source_dir = plugin_source_dir()
    install_bridge = install_dir / "bridge.py"
    source_bridge = source_dir / "bridge.py"
    install_build_id = build_id_for_file(install_bridge)
    source_build_id = build_id_for_file(source_bridge)
    instances = []
    for instance in list_instances():
        ping: dict[str, Any]
        try:
            response = _send_request_to_instance(
                instance,
                "doctor",
                params={},
                target=None,
            )
            ping = response["result"]
        except Exception as exc:
            ping = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        loaded_version = ping.get("plugin_version") if isinstance(ping, dict) else None
        loaded_build_id = ping.get("plugin_build_id") if isinstance(ping, dict) else None
        instances.append(
            {
                "pid": instance.pid,
                "socket_path": str(instance.socket_path),
                "plugin_version": instance.plugin_version,
                "plugin_build_id": loaded_build_id,
                "installed_plugin_build_id": install_build_id,
                "source_plugin_build_id": source_build_id,
                "stale_plugin_version": (
                    bool(loaded_version)
                    and str(loaded_version) != _package_version()
                ),
                "stale_plugin_code": (
                    bool(loaded_build_id)
                    and install_build_id is not None
                    and loaded_build_id != install_build_id
                ),
                "started_at": instance.started_at,
                "doctor": ping,
            }
        )

    result = {
        "cli_version": _package_version(),
        "plugin_source_dir": str(source_dir),
        "plugin_install_dir": str(install_dir),
        "plugin_source_build_id": source_build_id,
        "plugin_install_build_id": install_build_id,
        "instances": instances,
    }
    if args.format == "text":
        result = _render_doctor_text(result)
    _render_result(result, fmt=args.format, out_path=args.out, stem="doctor")
    return 0


@command("plugin", "install", help="Install the GUI plugin", fmt="json",
         args=[
             arg("--dest", type=Path, help="Custom install destination"),
             arg("--mode", choices=("symlink", "copy"), default="symlink"),
             arg("--force", action="store_true"),
         ])
def _plugin_install(args: argparse.Namespace) -> int:
    source = plugin_source_dir()
    dest = args.dest or plugin_install_dir()
    _install_tree(source, dest, mode=args.mode, force=args.force)

    _render_result(
        {
            "installed": True,
            "mode": args.mode,
            "source": str(source),
            "destination": str(dest),
        },
        fmt=args.format,
        out_path=args.out,
        stem="plugin-install",
    )
    return 0


def _install_tree(source: Path, dest: Path, *, mode: str, force: bool) -> None:
    if not source.exists():
        raise BridgeError(f"Source directory is missing: {source}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() or dest.is_symlink():
        if not force:
            raise BridgeError(f"Destination already exists: {dest}")
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)

    if mode == "copy":
        shutil.copytree(source, dest)
    else:
        os.symlink(source, dest, target_is_directory=True)


def _check_install_destination(dest: Path, *, force: bool) -> None:
    if force:
        return
    if dest.exists() or dest.is_symlink():
        raise BridgeError(f"Destination already exists: {dest}")


@command("skill", "install", help="Install the bundled agent skills", fmt="text",
         args=[
             arg("--dest", type=Path, help="Custom install destination"),
             arg("--mode", choices=("symlink", "copy"), default="symlink"),
             arg("--force", action="store_true"),
         ])
def _skill_install(args: argparse.Namespace) -> int:
    skills_root = repo_root() / "skills"
    explicit_dest = args.dest is not None
    target_roots = [args.dest] if explicit_dest else _default_skill_install_roots()
    install_plan = []
    results = []
    for source in sorted(skills_root.iterdir()):
        if not source.is_dir() or not (source / "SKILL.md").exists():
            continue
        destinations = []
        for target_root in target_roots:
            dest = target_root / source.name
            install_plan.append((source, dest))
            destinations.append(str(dest))
        results.append(
            {
                "skill": source.name,
                "source": str(source),
                "destination": destinations[0],
                "destinations": destinations,
            }
        )

    pending_installs = []
    skipped_destinations = []
    for source, dest in install_plan:
        if not explicit_dest and not args.force and (dest.exists() or dest.is_symlink()):
            skipped_destinations.append(str(dest))
            continue
        _check_install_destination(dest, force=args.force)
        pending_installs.append((source, dest))

    for source, dest in pending_installs:
        _install_tree(source, dest, mode=args.mode, force=args.force)

    result = {
        "installed": True,
        "mode": args.mode,
        "installed_destinations": [str(dest) for _, dest in pending_installs],
        "skipped_destinations": skipped_destinations,
        "skills": results,
    }
    if args.format == "text":
        result = _render_skill_install_text(result)
    _render_result(result, fmt=args.format, out_path=args.out, stem="skill-install")
    return 0


def _default_skill_install_roots() -> list[Path]:
    roots = [claude_skills_dir()]
    if codex_home().is_dir():
        roots.append(codex_skills_dir())
    return roots


@command("load", help="Load a binary into headless bridge",
         args=[
             arg("path", help="Path to binary or BNDB file"),
             arg("--no-bndb", action="store_true",
                 help="Don't auto-prefer a sibling .bndb file"),
         ])
def _load(args: argparse.Namespace) -> int:
    return _call(
        args,
        "load_binary",
        {
            "path": str(Path(args.path).expanduser().resolve()),
            "prefer_bndb": not args.no_bndb,
        },
        require_target=False,
        text_renderer=_render_load_text,
        stem="load",
    )


@command("close", help="Close a loaded binary",
         args=[arg("path", nargs="?", help="Path to close (omit to close all)")])
def _close(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    if args.path:
        params["path"] = str(Path(args.path).expanduser().resolve())
    return _call(
        args,
        "close_binary",
        params,
        require_target=False,
        text_renderer=_render_close_text,
        stem="close",
    )


@command("save", help="Save the current analysis database (.bndb)", target=True,
         args=[arg("path", nargs="?", help="Output path (defaults to <filename>.bndb)")])
def _save(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    if getattr(args, "path", None):
        params["path"] = str(Path(args.path).expanduser().resolve())
    return _call(
        args,
        "save_database",
        params,
        require_target=False,
        text_renderer=_render_save_text,
        stem="save",
    )


@command("session", "start", help="Start a new headless bridge session",
         args=[
             arg("binaries", nargs="*", help="Binary file paths to preload"),
             arg("--instance-id", help="Use a specific instance ID (default: random)"),
             arg("--no-bndb", action="store_true",
                 help="Don't auto-prefer a sibling .bndb file"),
         ])
def _session_start(args: argparse.Namespace) -> int:
    instance_id = getattr(args, "instance_id", None)
    instance = spawn_instance(instance_id)

    binaries = getattr(args, "binaries", None) or []
    prefer_bndb = not args.no_bndb
    loaded = []
    for binary in binaries:
        resolved = str(Path(binary).expanduser().resolve())
        try:
            resp = send_request(
                "load_binary",
                params={"path": resolved, "prefer_bndb": prefer_bndb},
                instance_id=instance.instance_id,
            )
            loaded.append(resp["result"])
        except BridgeError as exc:
            loaded.append({"path": resolved, "error": str(exc)})

    result: dict[str, Any] = {
        "instance_id": instance.instance_id,
        "pid": instance.pid,
        "socket_path": str(instance.socket_path),
    }
    if loaded:
        result["loaded"] = loaded

    if args.format == "text":
        result = _render_session_start_text(result)
    _render_result(result, fmt=args.format, out_path=args.out, stem="session-start")
    return 0


@command("session", "stop", help="Stop a running bridge session",
         args=[arg("instance", help="Instance ID to stop")])
def _session_stop(args: argparse.Namespace) -> int:
    import signal

    target_id = args.instance
    try:
        resp = send_request("shutdown", instance_id=target_id)
        result = {"instance_id": target_id, "stopped": True}
    except BridgeError:
        # Fallback: find instance and SIGTERM
        for inst in list_instances():
            if inst.instance_id == target_id or instance_selector(inst) == target_id:
                try:
                    os.kill(inst.pid, signal.SIGTERM)
                except OSError:
                    pass
                result = {"instance_id": target_id, "stopped": True, "method": "sigterm"}
                break
        else:
            raise BridgeError(f"No bridge instance found with id: {target_id}")

    if args.format == "text":
        result = _render_session_stop_text(result)
    _render_result(result, fmt=args.format, out_path=args.out, stem="session-stop")
    return 0


def _rss_mb(pid: int) -> float | None:
    """Read resident set size in MB from /proc/<pid>/status."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


@command("session", "list", help="List running bridge sessions")
def _session_list(args: argparse.Namespace) -> int:
    instances = list_instances()
    sticky_id = session_state.read().get("instance_id")
    entries = []
    total_rss = 0.0
    for inst in instances:
        rss = _rss_mb(inst.pid)
        selector = instance_selector(inst)
        entry: dict[str, Any] = {
            "selector": selector,
            "instance_id": inst.instance_id,
            "pid": inst.pid,
            "socket_path": str(inst.socket_path),
            "started_at": inst.started_at,
            "rss_mb": round(rss, 1) if rss is not None else None,
        }
        if sticky_id and (inst.instance_id == sticky_id or selector == sticky_id):
            entry["sticky"] = True
        entries.append(entry)
        if rss is not None:
            total_rss += rss
    result: dict[str, Any] = {
        "instances": entries,
        "total_rss_mb": round(total_rss, 1),
    }
    if args.format == "text":
        result = _render_session_list_text(result)
    _render_result(result, fmt=args.format, out_path=args.out, stem="session-list")
    return 0


@command("instance", "use", help="Pin a bridge instance for subsequent calls", fmt="text",
         args=[arg("instance_id", help="Instance ID to pin (see `bn session list`)")])
def _instance_use(args: argparse.Namespace) -> int:
    instance_id = args.instance_id
    instances = list_instances()
    matches = [
        inst for inst in instances
        if inst.instance_id == instance_id or instance_selector(inst) == instance_id
    ]
    if not matches:
        raise BridgeError(f"No running bridge instance with id: {instance_id}")
    resolved = matches[0].instance_id
    session_state.update(instance_id=resolved)
    result = {"instance_id": resolved, "set": True}
    if args.format == "text":
        result = f"instance: {resolved}"
    _render_result(result, fmt=args.format, out_path=args.out, stem="instance-use")
    return 0


@command("instance", "clear", help="Clear the pinned bridge instance", fmt="text")
def _instance_clear(args: argparse.Namespace) -> int:
    session_state.update(instance_id=None)
    result = {"instance_id": None, "set": False}
    if args.format == "text":
        result = "cleared"
    _render_result(result, fmt=args.format, out_path=args.out, stem="instance-clear")
    return 0


@command("target", "list", help="List open BinaryView targets")
def _target_list(args: argparse.Namespace) -> int:
    response = send_request(
        "list_targets",
        params={},
        instance_id=getattr(args, "instance", None),
    )
    result = response["result"]
    sticky = session_state.read().get("target")
    if sticky and isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and _target_matches(item, sticky):
                item["sticky"] = True
    if args.format == "text":
        result = _render_target_list_text(result)
    _render_result(result, fmt=args.format, out_path=args.out, stem="targets")
    return 0


def _target_matches(item: dict[str, Any], selector: str) -> bool:
    """True when *selector* names this target via any of its identifiers."""
    if selector == item.get("selector"):
        return True
    if selector == str(item.get("target_id", "")):
        return True
    if selector == str(item.get("view_id", "")):
        return True
    filename = item.get("filename")
    if isinstance(filename, str):
        if selector == filename:
            return True
        if selector == os.path.basename(filename):
            return True
    return False


@command("target", "use", help="Pin a target selector for subsequent calls", fmt="text",
         args=[arg("selector", help="Target selector to pin (see `bn target list`)")])
def _target_use(args: argparse.Namespace) -> int:
    session_state.update(target=args.selector)
    result = {"target": args.selector, "set": True}
    if args.format == "text":
        result = f"target: {args.selector}"
    _render_result(result, fmt=args.format, out_path=args.out, stem="target-use")
    return 0


@command("target", "clear", help="Clear the pinned target", fmt="text")
def _target_clear(args: argparse.Namespace) -> int:
    session_state.update(target=None)
    result = {"target": None, "set": False}
    if args.format == "text":
        result = "cleared"
    _render_result(result, fmt=args.format, out_path=args.out, stem="target-clear")
    return 0


@command("target", "info", help="Show one target", target=True)
def _target_info(args: argparse.Namespace) -> int:
    return _call(
        args,
        "target_info",
        {"selector": args.target},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_target_info_text,
        stem="target-info",
    )


@command("refresh", help="Refresh analysis for the selected target", target=True)
def _refresh(args: argparse.Namespace) -> int:
    return _call(
        args,
        "refresh",
        {},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_refresh_text,
        stem="refresh",
    )


@command("function", "list", help="List functions", target=True, paged=True, address_filter=True)
def _function_list(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    if args.min_address is not None:
        params["min_address"] = args.min_address
    if args.max_address is not None:
        params["max_address"] = args.max_address
    if args.offset:
        params["offset"] = args.offset
    return _call(
        args,
        "list_functions",
        params,
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_name_address_list_text,
        page_limit=args.limit,
        page_offset=args.offset,
        page_label="function list",
        stem="functions",
    )


@command("function", "search", help="Search functions by substring or regex",
         target=True, paged=True, address_filter=True,
         args=[
             arg("--regex", action="store_true",
                 help="Interpret query as a case-insensitive regular expression"),
             arg("query"),
         ])
def _function_search(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {
        "query": args.query,
        "regex": bool(args.regex),
    }
    if args.min_address is not None:
        params["min_address"] = args.min_address
    if args.max_address is not None:
        params["max_address"] = args.max_address
    if args.offset:
        params["offset"] = args.offset
    return _call(
        args,
        "search_functions",
        params,
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_name_address_list_text,
        page_limit=args.limit,
        page_offset=args.offset,
        page_label="function search",
        stem="function-search",
    )


@command("function", "info", help="Show function prototype and variables", target=True,
         args=[arg("identifier"),
               arg("--verbose", "-v", action="store_true", default=False,
                   help="Show full parameter and local variable details")])
def _function_info(args: argparse.Namespace) -> int:
    verbose = getattr(args, "verbose", False)
    return _call(
        args,
        "function_info",
        {"identifier": args.identifier},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=lambda v: _render_function_info_text(v, verbose=verbose),
        stem="function-info",
    )


@command("decompile", help="Render HLIL-style decompile text for a function", target=True,
         args=[
             arg("identifier"),
             arg("--addresses", action="store_true", default=False,
                 help="Show address prefixes on each line"),
             arg("--lines", type=_parse_line_range, default=None, metavar="START:END",
                 help="Show only lines START through END (1-indexed, inclusive)"),
         ])
def _decompile(args: argparse.Namespace) -> int:
    lines_range = getattr(args, "lines", None)

    def _render_decompile_text(value: Any) -> str:
        text = _text_field("text")(value)
        if lines_range is None:
            return text
        all_lines = text.splitlines()
        total = len(all_lines)
        start, end = lines_range
        sliced = all_lines[start - 1 : end]
        header = f"// lines {start}-{min(end, total)} of {total}"
        return header + "\n" + "\n".join(sliced)

    return _call(
        args,
        "decompile",
        {"identifier": args.identifier, "addresses": args.addresses},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_decompile_text,
        stem="decompile",
    )


@command("il", help="Dump IL for a function", target=True,
         args=[
             arg("identifier"),
             arg("--view", choices=("hlil", "mlil", "llil"), default="hlil"),
             arg("--ssa", action="store_true"),
         ])
def _il(args: argparse.Namespace) -> int:
    return _call(
        args,
        "il",
        {"identifier": args.identifier, "view": args.view, "ssa": bool(args.ssa)},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_text_field("text"),
        stem="il",
    )


@command("disasm", help="Disassemble a function", target=True,
         args=[arg("identifier")])
def _disasm(args: argparse.Namespace) -> int:
    return _call(
        args,
        "disasm",
        {"identifier": args.identifier},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_text_field("text"),
        stem="disasm",
    )


@command("xrefs", help="List xrefs to an address or function; use --field for struct field xrefs",
         target=True,
         args=[
             arg("identifier", nargs="?"),
             arg("--field", dest="field_spec",
                 help="Struct field xref spec (e.g., TrackRowCell.tile_type)"),
             arg("--limit", type=int, default=None,
                 help="Max number of code refs to show"),
         ])
def _xrefs(args: argparse.Namespace) -> int:
    field_spec = getattr(args, "field_spec", None)
    identifier = getattr(args, "identifier", None)
    limit = getattr(args, "limit", None)
    if field_spec:
        return _call(
            args,
            "field_xrefs",
            {"field": field_spec},
            require_target=True,
            allow_implicit_target=True,
            text_renderer=_render_field_xrefs_text,
            stem="field-xrefs",
        )
    if not identifier:
        raise BridgeError("xrefs requires an identifier or --field")
    return _call(
        args,
        "xrefs",
        {"identifier": identifier},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=lambda v: _render_xrefs_text(v, limit=limit),
        stem="xrefs",
    )


def _load_within_identifiers(path: Path) -> list[str]:
    identifiers = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        identifiers.append(line)
    return identifiers


@command("callsites", help="Find direct native callsites and exact caller_static addresses",
         target=True,
         args=[
             arg("callee"),
             arg("--context", type=int, default=3,
                 help="Number of previous and next instructions to include around each callsite"),
             arg("--caller-static", action="store_true",
                 help="Prefer caller_static-first text output for return-address mapping workflows"),
         ],
         mutex_groups=[
             mutex(False,
                   arg("--within", help="Containing function to search for callsites"),
                   arg("--within-file", type=Path,
                       help="Text file with one containing-function identifier per line (hex addresses accepted)")),
         ])
def _callsites(args: argparse.Namespace) -> int:
    if args.within is not None:
        within_identifiers = [args.within]
    elif args.within_file is not None:
        if not args.within_file.exists():
            raise BridgeError(f"Scope file not found: {args.within_file}")
        within_identifiers = _load_within_identifiers(args.within_file)
        if not within_identifiers:
            raise BridgeError(f"Scope file did not contain any function identifiers: {args.within_file}")
    else:
        raise BridgeError(
            "bn callsites needs a scope. Options:\n"
            f"  single caller:  bn callsites {args.callee} --within <function>\n"
            f"  many callers:   bn callsites {args.callee} --within-file <path>\n"
            f"  list callers:   bn xrefs {args.callee}"
        )

    return _call(
        args,
        "callsites",
        {
            "callee": args.callee,
            "within_identifiers": within_identifiers,
            "context": args.context,
            "caller_static": bool(args.caller_static),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=lambda value: _render_callsites_text(
            value,
            prefer_caller_static=bool(args.caller_static),
        ),
        stem="callsites",
    )


@command("types", help="List or search types", target=True, paged=True,
         args=[arg("--query")])
def _types(args: argparse.Namespace) -> int:
    return _call(
        args,
        "types",
        {"query": args.query, "offset": args.offset, "limit": args.limit},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_type_list_text,
        page_limit=args.limit,
        page_offset=args.offset,
        page_label="types",
        stem="types",
    )


@command("types", "show", help="Show one type", target=True,
         args=[arg("type_name")])
def _types_show(args: argparse.Namespace) -> int:
    return _call(
        args,
        "type_info",
        {
            "type_name": args.type_name,
            "require_struct": bool(getattr(args, "require_struct", False)),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_type_info_text,
        stem="type-show",
    )


@command("types", "declare", help="Import C declarations as user types", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("--file", type=Path, help="Read declarations from a file"),
             arg("--stdin", action="store_true", help="Read declarations from stdin"),
             arg("declaration", nargs="?"),
         ])
def _types_declare(args: argparse.Namespace) -> int:
    source_path = None
    if args.file is not None:
        if not args.file.exists():
            raise BridgeError(f"Declaration file not found: {args.file}")
        declaration = args.file.read_text(encoding="utf-8")
        source_path = str(args.file)
    elif args.stdin:
        declaration = sys.stdin.read()
    elif args.declaration:
        declaration = args.declaration
    else:
        raise BridgeError("Provide a declaration string, --file, or --stdin")

    return _call(
        args,
        "types_declare",
        {
            "declaration": declaration,
            "source_path": source_path,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="types-declare",
        result_exit_code=_mutation_exit_code,
    )


@command("strings", help="List or search strings", target=True, paged=True,
         args=[
             arg("--query"),
             arg("--min-length", type=int, default=None,
                 help="Exclude strings shorter than N characters"),
             arg("--section",
                 help="Only include strings in this section (e.g. .rodata, .rdata)"),
             arg("--no-crt", action="store_true", default=False,
                 help="Heuristic filter: exclude likely CRT/locale strings (platform-biased, best-effort)"),
         ])
def _strings(args: argparse.Namespace) -> int:
    return _call(
        args,
        "strings",
        {
            "query": args.query,
            "offset": args.offset,
            "limit": args.limit,
            "min_length": args.min_length,
            "section": args.section,
            "no_crt": args.no_crt,
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_strings_text,
        page_limit=args.limit,
        page_offset=args.offset,
        page_label="strings",
        stem="strings",
    )


@command("imports", help="List imports", target=True)
def _imports(args: argparse.Namespace) -> int:
    return _call(
        args,
        "imports",
        {},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_name_address_list_text,
        stem="imports",
    )


@command("sections", help="List binary sections with address ranges and permissions", target=True,
         args=[arg("--query", help="Filter sections by name substring")])
def _sections(args: argparse.Namespace) -> int:
    return _call(
        args,
        "sections",
        {"query": args.query},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_sections_text,
        stem="sections",
    )


@command("bundle", "function", help="Export a function bundle", fmt="json", target=True,
         args=[arg("identifier")])
def _bundle_function(args: argparse.Namespace) -> int:
    return _call(
        args,
        "bundle_function",
        {"identifier": args.identifier, "out_path": str(args.out) if args.out else None},
        require_target=True,
        allow_implicit_target=True,
        stem="function-bundle",
        bridge_writes_output=bool(args.out),
    )


@command("py", "exec", help="Execute a Python snippet", target=True,
         mutex_groups=[
             mutex(True,
                   arg("--script", type=Path, help="Read Python code from a file"),
                   arg("--code", help="Inline Python code"),
                   arg("--stdin", action="store_true")),
         ])
def _py_exec(args: argparse.Namespace) -> int:
    if getattr(args, "code", None) is not None:
        script = args.code
    elif args.script:
        if not args.script.exists():
            raise BridgeError(f"Script file not found: {args.script}. Use --code for inline Python.")
        script = args.script.read_text(encoding="utf-8")
    else:
        script = sys.stdin.read()

    return _call(
        args,
        "py_exec",
        {"script": script},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_py_exec_text,
        stem="py-exec",
    )


@command("symbol", "rename", help="Rename a symbol", target=True,
         args=[
             arg("--kind", choices=("auto", "function", "data"), default="auto"),
             arg("--preview", action="store_true"),
             arg("identifier"),
             arg("new_name"),
         ])
def _symbol_rename(args: argparse.Namespace) -> int:
    return _call(
        args,
        "rename_symbol",
        {
            "kind": args.kind,
            "identifier": args.identifier,
            "new_name": args.new_name,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="symbol-rename",
        result_exit_code=_mutation_exit_code,
    )


@command("comment", "list", help="List comments", target=True, paged=True,
         args=[arg("--query", help="Filter comments by substring")])
def _comment_list(args: argparse.Namespace) -> int:
    return _call(
        args,
        "list_comments",
        {"query": args.query, "offset": args.offset, "limit": args.limit},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_comment_list_text,
        page_limit=args.limit,
        page_offset=args.offset,
        page_label="comments",
        stem="comments",
    )


@command("comment", "set", help="Set a comment", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("--address"),
             arg("--function"),
             arg("comment"),
         ])
def _comment_set(args: argparse.Namespace) -> int:
    return _call(
        args,
        "set_comment",
        {
            "address": args.address,
            "function": args.function,
            "comment": args.comment,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="comment-set",
        result_exit_code=_mutation_exit_code,
    )


@command("comment", "get", help="Get a comment", target=True,
         args=[arg("--address"), arg("--function")])
def _comment_get(args: argparse.Namespace) -> int:
    return _call(
        args,
        "get_comment",
        {
            "address": args.address,
            "function": args.function,
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_comment_text,
        stem="comment-get",
    )


@command("comment", "delete", help="Delete a comment", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("--address"),
             arg("--function"),
         ])
def _comment_delete(args: argparse.Namespace) -> int:
    return _call(
        args,
        "delete_comment",
        {
            "address": args.address,
            "function": args.function,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="comment-delete",
        result_exit_code=_mutation_exit_code,
    )


@command("proto", "set", help="Set a prototype", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("identifier"),
             arg("prototype"),
         ])
def _proto_set(args: argparse.Namespace) -> int:
    return _call(
        args,
        "set_prototype",
        {
            "identifier": args.identifier,
            "prototype": args.prototype,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="prototype-set",
        result_exit_code=_mutation_exit_code,
    )


@command("proto", "get", help="Show the current prototype", target=True,
         args=[arg("identifier")])
def _proto_get(args: argparse.Namespace) -> int:
    return _call(
        args,
        "get_prototype",
        {"identifier": args.identifier},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_proto_text,
        stem="prototype-get",
    )


@command("local", "list", help="List locals with stable IDs", target=True,
         args=[arg("function")])
def _local_list(args: argparse.Namespace) -> int:
    return _call(
        args,
        "list_locals",
        {"identifier": args.function},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_local_list_text,
        stem="local-list",
    )


@command("local", "rename", help="Rename a local", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("function"),
             arg("variable", help="Stable local_id or legacy variable name"),
             arg("new_name"),
         ])
def _local_rename(args: argparse.Namespace) -> int:
    return _call(
        args,
        "local_rename",
        {
            "function": args.function,
            "variable": args.variable,
            "new_name": args.new_name,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="local-rename",
        result_exit_code=_mutation_exit_code,
    )


@command("local", "retype", help="Retype a local", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("function"),
             arg("variable", help="Stable local_id or legacy variable name"),
             arg("new_type"),
         ])
def _local_retype(args: argparse.Namespace) -> int:
    return _call(
        args,
        "local_retype",
        {
            "function": args.function,
            "variable": args.variable,
            "new_type": args.new_type,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="local-retype",
        result_exit_code=_mutation_exit_code,
    )


@command("struct", "field", "set", help="Set or replace a field", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("--no-overwrite", action="store_true"),
             arg("struct_name"),
             arg("offset"),
             arg("field_name"),
             arg("field_type"),
         ])
def _struct_field_set(args: argparse.Namespace) -> int:
    return _call(
        args,
        "struct_field_set",
        {
            "struct_name": args.struct_name,
            "offset": args.offset,
            "field_name": args.field_name,
            "field_type": args.field_type,
            "overwrite_existing": not args.no_overwrite,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="struct-field-set",
        result_exit_code=_mutation_exit_code,
    )


@command("struct", "show", help="Show one struct layout", target=True,
         args=[arg("struct_name")])
def _struct_show(args: argparse.Namespace) -> int:
    return _call(
        args,
        "type_info",
        {
            "type_name": args.struct_name,
            "require_struct": True,
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_type_info_text,
        stem="struct-show",
    )


@command("struct", "field", "rename", help="Rename a field", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("struct_name"),
             arg("old_name"),
             arg("new_name"),
         ])
def _struct_field_rename(args: argparse.Namespace) -> int:
    return _call(
        args,
        "struct_field_rename",
        {
            "struct_name": args.struct_name,
            "old_name": args.old_name,
            "new_name": args.new_name,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="struct-field-rename",
        result_exit_code=_mutation_exit_code,
    )


@command("struct", "field", "delete", help="Delete a field", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("struct_name"),
             arg("field_name"),
         ])
def _struct_field_delete(args: argparse.Namespace) -> int:
    return _call(
        args,
        "struct_field_delete",
        {
            "struct_name": args.struct_name,
            "field_name": args.field_name,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="struct-field-delete",
        result_exit_code=_mutation_exit_code,
    )


@command("batch", "apply", help="Apply a JSON manifest",
         args=[
             arg("--preview", action="store_true"),
             arg("manifest", type=Path),
         ])
def _batch_apply(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.preview:
        manifest["preview"] = True
    return _call(
        args,
        "batch_apply",
        manifest,
        require_target=False,
        text_renderer=_render_mutation_text,
        stem="batch-apply",
        result_exit_code=_mutation_exit_code,
    )


def _add_paged_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)


def _add_function_address_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--min-address",
        help="Only include functions whose start address is at or above this address",
    )
    parser.add_argument(
        "--max-address",
        help="Only include functions whose start address is at or below this address",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = BnArgumentParser(prog="bn", description="Agent-friendly Binary Ninja CLI")
    parser.set_defaults(handler=None)
    _instance_option(parser, is_root=True)
    _target_option(parser, required=False, is_root=True)
    _build_from_commands(parser)
    return parser


def _apply_sticky_defaults(args: argparse.Namespace) -> None:
    """Fill unset --instance / --target from per-project sticky state."""
    state = session_state.read()
    if not getattr(args, "instance", None):
        sticky_instance = state.get("instance_id")
        if sticky_instance:
            args.instance = sticky_instance
            args._sticky_instance = True
    if not getattr(args, "target", None):
        sticky_target = state.get("target")
        if sticky_target:
            args.target = sticky_target
            args._sticky_target = True


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] | None = getattr(args, "handler", None)
    if handler is None:
        selected_parser = getattr(args, "_parser", parser)
        selected_parser.print_help()
        return 1

    _apply_sticky_defaults(args)

    try:
        return handler(args)
    except BridgeError as exc:
        msg = str(exc)
        if getattr(args, "_sticky_instance", False) and _looks_like_dead_bridge(msg):
            msg += "\n\nThis came from sticky state. Clear it with `bn instance clear`."
        print(msg, file=sys.stderr)
        return 2


def _looks_like_dead_bridge(msg: str) -> bool:
    """True when *msg* points at a missing or unreachable bridge instance."""
    markers = (
        "No bridge instance found with id",
        "Failed to contact Binary Ninja bridge",
        "Timed out waiting for Binary Ninja bridge",
    )
    return any(marker in msg for marker in markers)
