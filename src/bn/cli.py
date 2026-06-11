from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable

from . import session_state
from .formatters import (
    FAILED_MUTATION_STATUSES,
    _format_operation_result,  # noqa: F401  -- re-exported for tests/scripts that monkeypatch bn.cli
    _render_target_choices,
)
from .output import render_envelope, render_error, write_output_result

# The names below are re-exported through this module on purpose: command
# handlers in bn.commands access them as `cli.<name>` so tests (and scripts)
# can monkeypatch a single well-known location, `bn.cli`.
from .paths import (  # noqa: F401
    claude_skills_dir,
    codex_home,
    codex_skills_dir,
    plugin_install_dir,
    plugin_source_dir,
    repo_root,
)
from .transport import (  # noqa: F401
    BridgeError,
    _send_request_to_instance,
    instance_selector,
    list_instances,
    send_request,
    spawn_instance,
)
from .version import VERSION, build_id_for_file  # noqa: F401


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


# The --format value as written on the command line, captured by main() before
# argparse runs so argparse's own usage/type errors can still honor
# --format json|ndjson (argparse fails before args.format is ever assigned).
# None when invoked outside main() (e.g. tests calling build_parser().parse_args
# directly), in which case error() behaves exactly like stock argparse.
_MACHINE_ERROR_FORMAT: str | None = None


def _requested_output_format(argv: list[str]) -> str | None:
    """The machine ``--format`` (json/ndjson) as written in *argv*, else None.

    Read before parsing so argparse errors can emit a JSON envelope. Last
    occurrence wins; stops at ``--`` (end of options).
    """
    fmt: str | None = None
    i = 0
    while i < len(argv):
        item = argv[i]
        if item == "--":
            break
        if item == "--format" and i + 1 < len(argv):
            fmt = argv[i + 1]
            i += 2
            continue
        if item.startswith("--format="):
            fmt = item.split("=", 1)[1]
        i += 1
    return fmt if fmt in ("json", "ndjson") else None


class BnArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.set_defaults(_parser=self)

        self.add_argument(
            "--help-full",
            action=_HelpFullAction,
            help="Show help for this command and all subcommands",
        )

    def error(self, message: str) -> None:  # type: ignore[override]
        # Argparse usage/type errors (bad --limit, unrecognized args, ...) print
        # usage to stderr and exit 2 with an EMPTY stdout, which breaks
        # `bn ... --format json | jq`. When a machine format was requested, emit
        # the same {"ok": false, "error": ...} envelope on stdout first, then
        # defer to argparse for the stderr usage text and the exit(2).
        if _MACHINE_ERROR_FORMAT in ("json", "ndjson"):
            self._print_message(
                render_error(f"{self.prog}: error: {message}", _MACHINE_ERROR_FORMAT),
                sys.stdout,
            )
        super().error(message)

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

    def parse_args(  # type: ignore[override]
        self,
        args: Any = None,
        namespace: Any = None,
    ) -> argparse.Namespace:
        parsed, extras = self.parse_known_args(args, namespace)
        if extras:
            # Route unrecognized arguments to the most-specific subcommand parser
            # so the usage/error text reflects the command the user actually ran,
            # not the bare root parser. Each subparser records itself as `_parser`.
            selected_parser = getattr(parsed, "_parser", self)
            selected_parser.error(
                f"unrecognized arguments: {' '.join(extras)}"
            )
        return parsed


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
    ("dataflow",): "Structured data-flow primitives (def-use, value-set, call graph)",
    ("taint",): "Taint analysis (forward source->sink, backward sink slicing)",
    ("evidence",): "Evidence-oriented reversing helpers",
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
        for spec in _COMMANDS:
            if spec["path"] == path:
                raise ValueError(
                    f"duplicate command path registration: {' '.join(path)!r} "
                    f"is already handled by {spec['handler'].__qualname__}"
                )
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
    paged: bool = False,
) -> None:
    if out_path is None and isinstance(value, dict) and isinstance(value.get("artifact_path"), str):
        artifact = dict(value)
        artifact.setdefault("ok", True)
        artifact.setdefault("spilled", False)
        sys.stdout.write(render_envelope(artifact, fmt))
        return

    result = write_output_result(value, fmt=fmt, out_path=out_path, stem=stem)
    if result.spilled and result.artifact:
        label = spill_label or stem.replace("_", " ")
        artifact = result.artifact
        artifact_path = artifact["artifact_path"]
        sys.stdout.write(result.rendered)
        hint = _spill_next_step_hint(stem, spill_context, artifact_path, paged=paged)
        print(
            f"warning: {label} output spilled to {artifact_path}; {hint}",
            file=sys.stderr,
        )
        return
    sys.stdout.write(result.rendered)


def _spill_next_step_hint(
    stem: str,
    spill_context: Any,
    artifact_path: str,
    *,
    paged: bool = False,
) -> str:
    """Build a command-keyed next-step slicing hint for spilled output.

    Mirrors the pagination truncation warning: line-oriented output (decompile,
    il, disasm) is line-sliced, list output from a paged command is paginated,
    and anything else points at --out or the artifact. ``paged`` is threaded
    from the command's @command declaration via ``_call``; only commands that
    actually expose --limit/--offset may suggest them.
    """

    if stem in ("decompile", "il", "disasm"):
        return "rerun with --lines START:END to fetch a slice instead"
    if paged and isinstance(spill_context, list):
        return "rerun with --limit/--offset to page through the results"
    return (
        f"rerun with --out <path> or read the artifact at {artifact_path} "
        "to inspect the full output"
    )


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
    spawn_missing_named: bool = False,
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
        spawn_missing_named=spawn_missing_named,
    )
    result = response["result"]
    exit_code = result_exit_code(result) if result_exit_code is not None else 0
    if effective_page_limit is not None and isinstance(result, list) and len(result) > effective_page_limit:
        result = result[:effective_page_limit]
        label = page_label or op
        next_offset = page_offset + effective_page_limit
        item_word = "item" if effective_page_limit == 1 else "items"
        print(
            f"warning: {label} output truncated to {effective_page_limit} {item_word}; rerun with --offset {next_offset} or a larger --limit",
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
        paged=page_limit is not None,
    )
    return exit_code


def _int_or_hex(value: str) -> int:
    """Parse a count/size accepting decimal or ``0x``/``0o``/``0b`` literals.

    RE work is hex-native (strides, ``memcpy`` sizes), so size args should take
    the same hex forms as address args rather than forcing decimal-only ``int``.
    """
    try:
        return int(value, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a decimal or hex (0x..) integer, got {value!r}"
        )


def _int_at_least(minimum: int, label: str) -> Callable[[str], int]:
    """Build an argparse type for a count/index flag: an integer >= *minimum*.

    Parses with ``int(value, 0)`` so count/size flags accept the same
    ``0x``/``0o``/``0b`` forms as address/size args -- a user who writes
    ``--limit 0x10`` should not be rejected when ``--length 0x40`` (wired to
    ``_int_or_hex``) is fine. Rejecting out-of-range values at the parse layer
    (argparse exit code 2) stops a negative value from leaking into Python's
    negative-slice semantics downstream, which silently drops trailing items
    and inverts the truncation math (a ``--limit -1`` would print "N more"
    counts that exceed the stated total). For count flags the minimum is 1: a
    count of nothing is always a mistake, never "unlimited" (that is expressed
    by omitting the flag).
    """
    def parse(value: str) -> int:
        try:
            parsed = int(value, 0)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(
                f"expected a decimal or hex (0x..) integer, got {value!r}")
        if parsed < minimum:
            raise argparse.ArgumentTypeError(
                f"{label} must be an integer >= {minimum}, got {parsed}")
        return parsed
    return parse


# Count flags (``--limit``) require >= 1; index flags (``--offset``) allow 0.
_positive_int = _int_at_least(1, "count")
_non_negative_int = _int_at_least(0, "index")


def _pick(positional: Any, flag: Any, label: str, *, required: bool = True) -> Any:
    """Resolve a value supplied either positionally or via a flag alias.

    Lets address/path args be passed uniformly (``bn read 0x.. `` or
    ``bn read --address 0x..``). Both-but-equal is fine; both-but-different
    raises, and absence raises only when ``required``.
    """
    if positional is not None and flag is not None and positional != flag:
        raise BridgeError(
            f"{label} given twice with different values ({positional!r} vs {flag!r})"
        )
    value = positional if positional is not None else flag
    if value is None and required:
        raise BridgeError(f"{label} is required")
    return value


def _parse_line_range(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected START:END, got {value!r}")
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected START:END with integers, got {value!r}")
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError(
            f"invalid range {start}:{end}; --lines is 1-indexed (START >= 1, END >= START)"
        )
    return (start, end)


def _add_paged_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--offset",
        type=_non_negative_int,
        default=0,
        help="Index of the first item to return (default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=100,
        help="Maximum number of items to return (default: 100)",
    )


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
    # Importing here populates _COMMANDS via @command decorators in submodules.
    # Deferred until call time to keep cli.py importable on its own.
    from . import commands  # noqa: F401

    parser = BnArgumentParser(
        prog="bn",
        description="Agent-friendly Binary Ninja CLI",
        epilog=(
            "Output over ~10k estimated tokens spills to disk; the command prints an "
            "envelope with the artifact path. Read that file directly -- do not pipe to grep."
        ),
    )
    parser.set_defaults(handler=None)
    _instance_option(parser, is_root=True)
    _target_option(parser, required=False, is_root=True)
    _build_from_commands(parser)
    return parser


def _selected_parser_for_argv(
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> argparse.ArgumentParser:
    selected = parser
    for item in argv:
        if item == "--":
            break
        if item.startswith("-"):
            continue
        subparser_action = next(
            (action for action in selected._actions if isinstance(action, argparse._SubParsersAction)),
            None,
        )
        if subparser_action is None or item not in subparser_action.choices:
            continue
        selected = subparser_action.choices[item]
    return selected


def _known_option_strings(parser: argparse.ArgumentParser) -> set[str]:
    options: set[str] = set()
    for action in parser._actions:
        options.update(action.option_strings)
    return options


# Options whose values are genuinely free-form data and may legitimately start
# with "-" (so the value must not be mistaken for a flag):
#   --query  free-form search text (strings/types/sections/comment list);
#            searching for the literal string "-h" or "--" must work
#   --code   inline Python source for `bn py exec`; snippets routinely start
#            with characters argparse would treat as options
_PROTECTED_DATA_OPTIONS = frozenset({"--query", "--code"})


def _protect_flag_like_option_values(
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> list[str]:
    """Let explicit data options accept values that look like flags.

    Argparse treats ``bn strings --query -h`` as a help flag instead of a query
    value. When the user has explicitly supplied an option that takes arbitrary
    data, preserve the next token as that option's value by rewriting to the
    ``--opt=value`` spelling before parsing.
    """
    protected_options = _PROTECTED_DATA_OPTIONS
    selected_parser = _selected_parser_for_argv(parser, argv)
    known_options = _known_option_strings(selected_parser)
    help_options = {"-h", "--help", "--help-full"}
    out: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in protected_options and index + 1 < len(argv):
            value = argv[index + 1]
            value_option = value.split("=", 1)[0]
            if value.startswith("-") and (value_option not in known_options or value_option in help_options):
                out.append(f"{item}={value}")
                index += 2
                continue
        out.append(item)
        index += 1
    return out


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
    global _MACHINE_ERROR_FORMAT
    parser = build_parser()
    parse_argv = sys.argv[1:] if argv is None else list(argv)
    # Capture --format before parsing so argparse usage/type errors (which fire
    # before args.format exists) can still emit a JSON error envelope.
    _MACHINE_ERROR_FORMAT = _requested_output_format(parse_argv)
    args = parser.parse_args(_protect_flag_like_option_values(parser, parse_argv))
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
        # Under a machine-readable format, also emit the error as JSON on stdout
        # so `bn ... --format json | jq` gets a parseable object instead of an
        # empty stream; the human-readable line still goes to stderr. Routed
        # through render_error so the envelope matches successful JSON output.
        if getattr(args, "format", None) in ("json", "ndjson"):
            sys.stdout.write(render_error(msg, args.format))
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
