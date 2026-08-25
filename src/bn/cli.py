from __future__ import annotations

import argparse
import os
import re
import stat
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from . import session_state
from .formatters import (
    FAILED_MUTATION_STATUSES,
    _add_mutation_ok,
    _format_operation_result,  # noqa: F401  -- re-exported for tests/scripts that monkeypatch bn.cli
    _mutation_summary,
    _render_fanout_text,
    _render_mutation_summary_text,
    _render_mutation_text,
    _render_target_choices,
)
from .output import render_envelope, render_error, render_value, write_output_result

# The names below are re-exported through this module on purpose: command
# handlers in bn.commands access them as `cli.<name>` so tests (and scripts)
# can monkeypatch a single well-known location, `bn.cli`.
from .paths import (  # noqa: F401
    claude_skills_dir,
    codex_home,
    codex_skills_dir,
    omp_agent_dir,
    omp_config_root,
    omp_skills_dir,
    plugin_install_dir,
    plugin_source_dir,
    remove_instance_markers,
    repo_root,
    skills_source_dir,
)
from .transport import (  # noqa: F401
    BridgeError,
    _send_request_to_instance,
    gc_instances,
    instance_selector,
    list_instances,
    send_request,
    spawn_instance,
    validate_instance_id,
    wait_for_teardown,
)
from .version import VERSION, build_id_for_file, build_id_for_package  # noqa: F401


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


_OUT_FORMAT_BY_SUFFIX = {".json": "json", ".ndjson": "ndjson"}


class _RecordExplicitFormat(argparse.Action):
    """Set ``--format`` AND record that the user gave it explicitly, so a later
    ``--out x.json`` only infers a format when none was chosen (#315)."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, "_format_explicit", True)


def _resolve_output_format(args: argparse.Namespace) -> str:
    """The effective output format, inferring json/ndjson from an ``--out`` path's
    extension when the user did not choose a format explicitly (#315).

    ``--out foo.json`` without ``--format json`` previously wrote the human TEXT
    renderer into a ``.json`` file (reads default to text), so a downstream
    ``jq``/``json.load`` silently broke. Infer the format from the extension when
    none was given; when an explicit ``--format`` disagrees with the extension,
    honor the explicit choice but warn so it is not a silent footgun.
    """
    fmt = getattr(args, "format", "text")
    out = getattr(args, "out", None)
    if out is None:
        return fmt
    inferred = _OUT_FORMAT_BY_SUFFIX.get(Path(str(out)).suffix.lower())
    if inferred is None or inferred == fmt:
        return fmt
    suffix = Path(str(out)).suffix
    if getattr(args, "_format_explicit", False):
        print(
            f"warning: --out {out} ends in {suffix} but --format {fmt} was given; "
            f"writing {fmt}. Pass --format {inferred} to write {inferred}.",
            file=sys.stderr,
        )
        return fmt
    print(
        f"note: inferring --format {inferred} from the {suffix} --out path; "
        f"pass an explicit --format to override.",
        file=sys.stderr,
    )
    return inferred


class BnArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.set_defaults(_parser=self)

        self.add_argument(
            "--help-full",
            action=_HelpFullAction,
            help="Show help for this command and all subcommands",
        )

    def _augment_subcommand_error(self, message: str) -> str:
        # `bn py '<code>'` fails with argparse's opaque "invalid choice: '<code>'
        # (choose from 'exec')" because a command GROUP takes a subcommand, not a
        # bare argument. For a single-subcommand group the intent is unambiguous,
        # so point at the real form (`bn py exec ...`) instead of leaving the
        # user staring at their own code echoed as a rejected choice.
        if "invalid choice:" not in message:
            return message
        for action in self._actions:
            if isinstance(action, argparse._SubParsersAction):
                choices = list(action.choices)
                if len(choices) == 1:
                    return (
                        f"{message}\n{self.prog} is a command group — pass your "
                        f"input to the '{choices[0]}' subcommand: "
                        f"`{self.prog} {choices[0]} ...`"
                    )
                break
        return message

    def error(self, message: str) -> None:  # type: ignore[override]
        message = self._augment_subcommand_error(message)
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
        action=_RecordExplicitFormat,
        help="Output format",
    )
    parser.add_argument(
        "--out", type=Path,
        help="Write output to a file instead of stdout (a .json/.ndjson path "
             "infers --format unless one is given)",
    )


def _instance_option(parser: argparse.ArgumentParser, *, is_root: bool = False) -> None:
    parser.add_argument(
        "-i",
        "--instance",
        default=os.environ.get("BN_INSTANCE") if is_root else argparse.SUPPRESS,
        help="Target a specific bridge instance by ID (env: BN_INSTANCE)",
    )


def _fanout_option(parser: argparse.ArgumentParser) -> None:
    """`--all-instances` / `--all-targets`: run a READ across every running bridge
    instance and/or every target in an instance, aggregating the per-(instance,
    target) results (#169 Layer 1). Attached only to commands the registry marks
    `fanout=True` (whole-target read surveys) -- never mutations -- so a write
    can't be fanned. SUPPRESS default so a flag is absent from the namespace
    unless passed."""
    parser.add_argument(
        "--all-instances",
        action="store_true",
        default=argparse.SUPPRESS,
        dest="all_instances",
        help="Run this read across every running bridge instance and aggregate the results",
    )
    parser.add_argument(
        "--all-targets",
        action="store_true",
        default=argparse.SUPPRESS,
        dest="all_targets",
        help="Run this read across every target open in the instance (combine with "
             "--all-instances for every instance x target)",
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
            "omit only when exactly one target is open, or use `active` to follow the GUI-selected target explicitly "
            "(destructive `close` does not honor `active` -- it needs a concrete selector, a path, or --all)"
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
    ("class",): "C++ object-model lens (classes, vtables, RTTI)",
    ("go",): "Go binary symbol recovery (.gopclntab)",
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


def preview_arg(
    help: str = "Apply, capture diffs, then revert without committing",
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """The shared ``--preview`` flag every mutation command exposes.

    A few commands phrase the help slightly differently (e.g. ``function create``
    says "Create, verify, then revert"); pass *help* to override.
    """
    return arg("--preview", action="store_true", help=help)


def summary_arg() -> tuple[tuple[str, ...], dict[str, Any]]:
    """The shared ``--summary`` / ``--quiet`` flag for mutation commands (#408).

    Since #645 the compact status summary is the DEFAULT, so this flag only still
    matters to force compactness under an explicit ``--format json`` / ``--out``.
    Kept accepted (never an error) for compatibility with existing agent scripts."""
    return arg("--summary", "--quiet", action="store_true", default=False,
               dest="summary",
               help="Force the compact status summary (counts, first_error, "
                    "dirty_after) even under an explicit --format json or --out. The "
                    "compact summary is already the default")


def verbose_arg() -> tuple[tuple[str, ...], dict[str, Any]]:
    """The shared ``--verbose`` / ``--diffs`` flag for mutation commands (#645).

    Mutations used to default to ``--format json`` and echo every per-op diff,
    ``requested``, ``observed``, and ``before_*`` field -- the single largest source
    of avoidable token burn in a write-heavy session (a `proto set` cost ~7 KB where
    the status line costs 225 bytes, a 115-op previewed batch cost 261 KB / 87k
    tokens). The compact status line is now the default and the full audit payload
    is opt-in through this flag or an explicit ``--format json`` / ``--out``."""
    return arg("--verbose", "--diffs", action="store_true", default=False,
               dest="verbose",
               help="Emit the full mutation audit payload (per-op requested/observed "
                    "state and diffs) instead of the compact status summary")


def mutation_output_args() -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    """Both mutation output-shaping flags, for splatting into a command's args."""
    return [summary_arg(), verbose_arg()]


def command(
    *path: str,
    help: str = "",
    fmt: str = "text",
    target: bool = False,
    paged: bool = False,
    address_filter: bool = False,
    args: list[tuple[tuple[str, ...], dict[str, Any]]] | None = None,
    mutex_groups: list[tuple[bool, list[tuple[tuple[str, ...], dict[str, Any]]]]] | None = None,
    prefer_when: str = "",
    see_also: tuple[str, ...] = (),
    fanout: bool = False,
) -> Callable:
    """Register a CLI command declaratively.

    ``prefer_when`` / ``see_also`` are the agent-routing hints surfaced by
    ``bn capabilities`` (#276): when a command overlaps a neighbor, ``prefer_when``
    says when to reach for this one and ``see_also`` names the neighbors (by their
    space-joined command path, e.g. ``"function search"``). They live on the
    command so the index stays registry-derived -- no hand-maintained second list.

    ``fanout`` opts a command into ``--all-instances`` (#169 L1). It is an explicit
    allow-list, NOT inferred from ``fmt``: only genuine whole-target READ surveys
    (no per-function/address identifier) set it, so a write or side-effecting
    command -- several of which default to ``fmt="text"`` (save/close/refresh/py
    exec/load) -- can never be fanned across every instance.
    """

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
            "prefer_when": prefer_when,
            "see_also": tuple(see_also),
            "fanout": fanout,
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
        # Carry -t/--instance on the intermediate group parser so they can appear
        # BEFORE the leaf (`bn bundle -t X function F`), at parity with
        # single-level commands and root-level -t. Both helpers use SUPPRESS
        # defaults on non-root parsers, so (a) an absent flag never lands in the
        # namespace and (b) the leaf subparser never clobbers a value set here.
        # --format/--out are deliberately left off: their real (non-SUPPRESS)
        # leaf defaults WOULD overwrite an intermediate-level value (#251).
        _instance_option(parser)
        _target_option(parser, required=False)
        node_parsers[path] = parser
        return parser

    # Shorter paths first: a node that is BOTH a leaf command and a parent group
    # (e.g. `('types',)` has handler `_types` and children `types show`/`types
    # declare`) is leaf-processed into node_parsers before _ensure_intermediate
    # is ever called for it as a parent, so the early `if path in node_parsers`
    # return prevents a second -t/--instance add (argparse "conflicting option
    # string"). Keep this sort if the registry/dual-role nodes are refactored.
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
        # Fan-out is an EXPLICIT allow-list (`fanout=True` on genuine whole-target
        # read surveys), not inferred from fmt -- several write/side-effecting
        # commands (save/close/refresh/py exec/load) also default to text, so a
        # format gate would fan a WRITE across every instance (#169 L1 review).
        if spec.get("fanout"):
            _fanout_option(cmd)
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
    spill_status: tuple[Any, Callable[[Any], str] | None] | None = None,
    provenance: dict[str, Any] | None = None,
) -> bool:
    """Render *value* to stdout; return True iff the output spilled to disk.

    The spilled flag lets the caller decide whether to add a further note (e.g. a
    display-truncation warning): a spill already prints its own pipe-trap note, so
    a caller-side note would be redundant when it fires."""
    if out_path is None and isinstance(value, dict) and isinstance(value.get("artifact_path"), str):
        artifact = dict(value)
        artifact.setdefault("ok", True)
        artifact.setdefault("spilled", False)
        for key, val in (provenance or {}).items():   # #653.8
            if val is not None:
                artifact.setdefault(key, val)
        sys.stdout.write(render_envelope(artifact, fmt))
        return bool(artifact.get("spilled"))

    result = write_output_result(value, fmt=fmt, out_path=out_path, stem=stem,
                                provenance=provenance)
    if result.spilled and result.artifact and spill_status is not None:
        # #645: NEVER put a spill envelope on stdout for a mutation. A read that
        # spills is recoverable (re-read the artifact); an atomic write whose result
        # is unparseable is not -- the agent's model of the BNDB silently desyncs
        # from the BNDB. The detail is already on disk; stdout keeps the small,
        # parseable status (with a pointer to the detail).
        artifact_path = result.artifact["artifact_path"]
        status_value, status_renderer = spill_status
        if isinstance(status_value, dict):
            status_value = {**status_value, "detail_artifact_path": str(artifact_path)}
        if fmt == "text" and status_renderer is not None:
            rendered = status_renderer(status_value)
            sys.stdout.write(rendered if rendered.endswith("\n") else rendered + "\n")
        else:
            sys.stdout.write(render_value(status_value, fmt))
        print(
            f"note: full mutation detail ({result.artifact.get('estimated_tokens')} est. "
            f"tokens) written to {artifact_path}; stdout carries the parseable status "
            f"summary. Re-run with --out FILE for the detail in a chosen location.",
            file=sys.stderr,
        )
        return False
    if result.spilled and result.artifact:
        label = spill_label or stem.replace("_", " ")
        artifact = result.artifact
        artifact_path = artifact["artifact_path"]
        # #216: a piped TEXT consumer (grep/awk/rg) reads only the small spill
        # envelope, so a no-match silently reads as "absent" -- a false negative
        # that has misled agents. Emit a loud, greppable marker as the FIRST
        # stdout line so the spill is impossible to miss in the stream itself (the
        # stderr note is lossy and invisible to the pipe consumer). Text only: a
        # marker line would corrupt the json/ndjson envelope, whose `spilled:true`
        # field is already machine-checkable.
        if fmt == "text" and _stdout_is_pipe():
            sys.stdout.write(f"__BN_SPILLED__ {artifact_path}\n")
        sys.stdout.write(result.rendered)
        hint = _spill_next_step_hint(
            stem, spill_context, artifact_path, paged=paged, text_format=(fmt == "text")
        )
        print(
            f"warning: {label} output spilled to {artifact_path}; {hint}",
            file=sys.stderr,
        )
        if _stdout_is_pipe():
            # Output spilled AND stdout is a pipe: a downstream grep/jq/awk reads
            # only this small envelope, never the spilled data, so a no-match
            # reads as "absent" -- a silent false negative. Make the trap loud
            # (#195). (A terminal / agent-capture isn't a pipe, so this is quiet
            # there.)
            print(
                f"note: stdout is a pipe -- grep/jq/awk receive only the spill "
                f"envelope, not the data, so a no-match is NOT a real absence. "
                f"Re-run with --out FILE and search the file (or grep {artifact_path}).",
                file=sys.stderr,
            )
        return True
    sys.stdout.write(result.rendered)
    if result.near_spill:
        # #409: fit this time, but close to the threshold -- warn so the agent slices
        # the next (larger) read pre-emptively instead of discovering the spill after
        # paying for it. Stderr only, so stdout/pipes stay clean.
        print(
            "note: output is within 20% of the spill threshold; a slightly larger next "
            "read (next page / bigger scope) will spill to disk -- bound it with "
            "--limit/--offset/--lines, or raise BN_SPILL_TOKENS.",
            file=sys.stderr,
        )
    return False


def _emit_result(
    args: argparse.Namespace,
    result: Any,
    *,
    text_renderer: Callable[[Any], str] | None = None,
    stem: str,
) -> None:
    """Render a locally-built result for handlers that can't go through :func:`_call`.

    The admin commands (doctor, session/instance/target management) orchestrate
    several bridge requests, so they assemble the result themselves instead of
    making one ``_call``. This is ``_call``'s rendering tail factored out so they
    share it: under ``--format text`` it applies *text_renderer* behind the same
    malformed-result guard ``_call`` uses (a renderer crash on a malformed/newer
    bridge response becomes a clean :class:`BridgeError` pointing at ``--format
    json``, never a raw traceback), then writes via :func:`_render_result`. With no
    *text_renderer* the value renders as-is in every format.
    """
    fmt = _resolve_output_format(args)
    if text_renderer is not None and fmt == "text":
        try:
            result = text_renderer(result)
        except (AttributeError, TypeError, KeyError, IndexError, ValueError) as exc:
            raise BridgeError(
                f"could not render the {stem} result as text -- the bridge response was "
                f"malformed or newer than this CLI. Rerun with --format json to see the "
                f"raw result. ({type(exc).__name__}: {exc})"
            ) from exc
    _render_result(result, fmt=fmt, out_path=args.out, stem=stem)


# Regex metacharacters that make a literal substring query silently match
# nothing without --regex. Deliberately EXCLUDES '.' -- it is too common in
# legitimate literal names/strings (std::x, a.b.c) to flag (#122).
_REGEX_METACHARS = "|()[]{}*+?^$\\"


def _maybe_regex_hint(args: argparse.Namespace, result: Any, query: str | None) -> None:
    """When a NON-regex search whose query contains regex metacharacters matched
    nothing, nudge toward --regex on stderr -- the query was taken literally, so
    a pattern like `init|fini` silently returns zero results with no clue (#122).
    Suppressed when --regex/--exact is set or the query is plain text."""
    if not query or getattr(args, "regex", False) or getattr(args, "exact", False):
        return
    # Every collection read is now a {kind, items, total, ...} envelope (#275),
    # so "empty" is just total == 0 -- no bare-list special-case.
    empty = isinstance(result, dict) and result.get("total") == 0
    if not empty or not any(ch in query for ch in _REGEX_METACHARS):
        return
    print(
        f'note: 0 matches; "{query}" was matched literally. It contains regex '
        "metacharacters -- add --regex to match it as a pattern.",
        file=sys.stderr,
    )


def _should_retry_as_regex(args: argparse.Namespace, result: Any, query: str) -> bool:
    """True when a literal search of *query* found nothing and the query looks like
    a regex that compiles -- the caller should re-run it as a pattern (#291.3).

    A first-pass alternation like `Parse|Process|Decode` is taken literally and
    returns a confident (misleading) `none`; auto-retrying it as a regex removes
    the trap. Guarded so we never change a search that DID match, and never turn a
    clean 0-result into a regex-compile error: only an explicit-literal query with
    unescaped metacharacters that both matched nothing and compiles is retried."""
    if getattr(args, "regex", False) or getattr(args, "exact", False):
        return False
    if not isinstance(result, dict) or result.get("total") != 0:
        return False
    if not any(ch in query for ch in _REGEX_METACHARS):
        return False
    try:
        re.compile(query)
    except re.error:
        return False
    return True


def _maybe_offset_hint(args: argparse.Namespace, result: Any, identifier: str | None) -> None:
    """When `xrefs <bare-number>` matched nothing and the value is small enough to
    look like a struct-field offset (< 0x10000), nudge toward --field: a bare value
    is taken as an absolute address, so a field offset like 0x308 silently returns
    zero xrefs with no clue. Suppressed for function-name identifiers and for
    plausible code/data addresses (>= 0x10000)."""
    if not identifier:
        return
    text = str(identifier).strip()
    try:
        value = int(text, 16) if text.lower().startswith("0x") else int(text, 10)
    except ValueError:
        return
    # Key off the canonical envelope `total`: the xrefs op ships items + total,
    # not the legacy code_refs/data_refs arrays (#184/#275).
    empty = isinstance(result, dict) and not result.get("total")
    if not empty or value >= 0x10000:
        return
    print(
        f"note: 0 xrefs to {text} -- a bare value is taken as an absolute address. "
        f"If {text} is a struct field offset, use `bn xrefs --field <Struct.field>` instead.",
        file=sys.stderr,
    )


def _stdout_is_pipe() -> bool:
    """True when stdout is a pipe (FIFO), e.g. ``bn ... | grep``.

    Distinguishes the pipe case from a terminal or a regular-file redirect so the
    spill pipe-trap caution (#195) fires only where a downstream filter would
    silently consume the envelope instead of the data. Defensive: any error
    inspecting the fd (e.g. a captured/replaced stdout with no real fileno)
    yields False, so the caution stays quiet rather than misfiring."""
    try:
        return stat.S_ISFIFO(os.fstat(sys.stdout.fileno()).st_mode)
    except Exception:
        return False


def _spill_next_step_hint(
    stem: str,
    spill_context: Any,
    artifact_path: str,
    *,
    paged: bool = False,
    text_format: bool = True,
) -> str:
    """Build a command-keyed next-step slicing hint for spilled output.

    Mirrors the pagination truncation warning: line-oriented output (decompile,
    il, disasm) is line-sliced, list output from a paged command is paginated,
    and anything else points at --out or the artifact. ``paged`` is threaded
    from the command's @command declaration via ``_call``; only commands that
    actually expose --limit/--offset may suggest them.

    ``--lines`` only slices the TEXT renderer, so it is only suggested for text
    output -- recommending it to a JSON consumer is a dead end (#120). In JSON
    mode the line-oriented commands fall through to the --out/artifact hint.
    """

    if text_format and stem in ("decompile", "il", "disasm"):
        return "rerun with --lines START:END to fetch a slice instead"
    # `paged` is only set for commands that actually expose --limit/--offset, so
    # it alone gates the paging hint (function list/search now page bridge-side
    # and return a dict envelope rather than a bare list, #59).
    if paged:
        return "rerun with --limit/--offset to page through the results"
    # The warning already names the artifact path ("spilled to <path>"); don't
    # repeat it a second time in the hint (#49).
    return "rerun with --out <path> to write it to a file, or read that artifact to inspect the full output"


class MultiTargetError(BridgeError):
    """Refusal because several targets are open and no selector was given.

    A distinct type so callers (close's `-t active` note) can match the
    condition without sniffing message prose."""


def _require_nonempty_instance(args: argparse.Namespace) -> None:
    """Reject an explicit-but-empty --instance before anything is sent.

    Fires for `-i ""` and for an empty exported BN_INSTANCE (the root flag
    defaults from the environment); the pin never substitutes for either
    (see _apply_sticky_defaults)."""
    instance = getattr(args, "instance", None)
    if instance is not None and not str(instance).strip():
        raise BridgeError(
            "--instance is empty: pass an instance id from `bn session list`, "
            "or omit -i (and unset an empty BN_INSTANCE) to use the default "
            "instance"
        )


def _implicit_target(args: argparse.Namespace) -> str:
    """Resolve the single open target to its pinned ``target_id``, else refuse.

    Returns the stable ``target_id`` from the peek row -- never the volatile
    ``active`` literal, which the bridge re-resolves at act time: for a write
    op a concurrent close/load between peek and act would land the operation
    on a DIFFERENT binary (#690 R3). View ids are never reused, so a stale
    pinned id fails as a safe unknown-selector error instead.
    """
    _require_nonempty_instance(args)
    response = send_request(
        "list_targets",
        params={},
        target=None,
        instance_id=getattr(args, "instance", None),
    )
    result = response.get("result") if isinstance(response, dict) else None
    if not isinstance(result, list):
        raise BridgeError(
            "malformed bridge reply to list_targets (no target list); the "
            "bridge may be stale -- restart it or pass an explicit --target"
        )
    targets = list(result)
    if not targets:
        raise BridgeError("No BinaryView targets are open")
    if len(targets) > 1:
        raise MultiTargetError(
            "This command requires --target when multiple targets are open.\n"
            f"Open targets:\n{_render_target_choices(targets)}"
        )
    row = targets[0]
    target_id = row.get("target_id") if isinstance(row, dict) else None
    if not target_id:
        raise BridgeError(
            "bridge listed the open target without a target_id; pass an "
            "explicit --target"
        )
    return str(target_id)


def _resolve_target(
    args: argparse.Namespace,
    *,
    require_target: bool,
    allow_implicit_target: bool = False,
) -> str | None:
    target = getattr(args, "target", None)
    # An explicit-but-empty selector (an unset shell variable, `-t ""`) is
    # never pin-filled (see _apply_sticky_defaults) and must never be
    # forwarded either: the bridge collapses "" to the focused GUI view with
    # no count check, so a write op would silently act on the wrong target.
    if target is not None and not str(target).strip():
        raise BridgeError(
            "--target is empty: pass a selector from `bn target list`, or "
            "omit --target to use the single open target"
        )
    if require_target and target is None:
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


def _mutate(
    args: argparse.Namespace,
    op: str,
    params: dict[str, Any],
    *,
    stem: str,
    require_target: bool = True,
    preview: bool | None = None,
    detail_renderer: Any = None,
    summary_transform: Any = None,
    **call_kwargs: Any,
) -> int:
    """:func:`_call` specialized for mutations.

    Every mutation command shares the same tail -- the mutation text renderer and
    the verification-aware exit code -- and differs only in its op, param dict,
    and stem. This bakes in that tail and, when *preview* is supplied, injects the
    ``preview`` param so handlers don't each repeat ``"preview": bool(args.preview)``.
    Extra keyword args (e.g. ``bridge_writes_output``) pass through to ``_call``.
    """
    if preview is not None:
        params = {**params, "preview": preview}
    # #408: --summary collapses the result to a compact, schema-stable status
    # object for an unattended control loop. The exit code is unchanged (computed
    # on the full result); only the rendered/returned payload is compacted.
    summary = bool(getattr(args, "summary", False))
    # #645: the compact status is now the DEFAULT. The full audit payload -- every
    # per-op diff, `requested`, `observed`, `before_*` -- was the single largest
    # source of avoidable token burn in a write-heavy session (a `proto set` cost
    # ~7 KB vs 225 bytes; a 115-op previewed batch cost 261 KB / 87k tokens). Detail
    # is opt-in: --verbose/--diffs, an explicit machine --format, or --out (where the
    # payload lands in a file rather than the context window).
    detail = (
        bool(getattr(args, "verbose", False))
        # --out puts the payload in a FILE, not the context window, so the caller
        # who asked for a file gets the full detail in it.
        or bool(getattr(args, "out", None))
        or (bool(getattr(args, "_format_explicit", False))
            and str(getattr(args, "format", "text")) in ("json", "ndjson"))
    )
    compact = summary or not detail
    if not getattr(args, "_format_explicit", False):
        # These commands declare fmt="json" for back-compat of the parser default;
        # mutations now render as TEXT by default, mirroring the read commands.
        # --format controls the MEDIUM, --verbose/--summary the DETAIL. An explicit
        # --format still wins, and an --out extension is still inferred downstream.
        setattr(args, "format", "text")
    return _call(
        args,
        op,
        params,
        require_target=require_target,
        # A mutation with a richer detail view of its own (e.g. `go rename`'s
        # per-candidate breakdown) supplies it via *detail_renderer*; the compact
        # status line stays shared so #645's default is identical everywhere.
        text_renderer=(_render_mutation_summary_text if compact
                       else (detail_renderer or _render_mutation_text)),
        result_exit_code=_mutation_exit_code,
        # A mutation whose result does not report through `results[]` supplies its
        # own compact transform; everything else shares `_mutation_summary`.
        result_transform=((summary_transform or _mutation_summary) if compact
                          else _add_mutation_ok),
        stem=stem,
        # #645: an atomic write whose result is unreadable is a correctness problem,
        # not a cost one -- a spilled 38-op batch put the spill envelope on stdout,
        # so `json.loads` raised and the agent could not confirm a batch that HAD
        # committed. Keep the parseable status on stdout no matter how big the
        # detail payload is; the detail goes to the artifact.
        # _call evaluates this against the ALREADY-transformed result, so both
        # transforms short-circuit on a `mutation_summary` envelope.
        spill_status=(summary_transform or _mutation_summary),
        spill_status_renderer=_render_mutation_summary_text,
        **call_kwargs,
    )


def _call(
    args: argparse.Namespace,
    op: str,
    params: dict[str, Any] | None = None,
    *,
    require_target: bool,
    # Defaults True: every target-required command wants the single-open-target
    # convenience. Note bare `bn close` ALSO rides the implicit-resolution
    # path (it calls _implicit_target itself, #664/#690) even though it passes
    # require_target=False here. Pass False only to force an explicit
    # -t/--target on a target-required command.
    allow_implicit_target: bool = True,
    text_renderer: Callable[[Any], str] | None = None,
    page_limit: int | None = None,
    page_offset: int = 0,
    page_label: str | None = None,
    paged_spill: bool = False,
    stem: str,
    result_exit_code: Callable[[Any], int] | None = None,
    result_transform: Callable[[Any], Any] | None = None,
    bridge_writes_output: bool = False,
    spawn_missing_named: bool = False,
    regex_hint_query: str | None = None,
    regex_fallback_query: str | None = None,
    offset_hint_identifier: str | None = None,
    truncation_note: Callable[[Any], str | None] | None = None,
    op_default_timeout: float | None = None,
    spill_status: Callable[[Any], Any] | None = None,
    spill_status_renderer: Callable[[Any], str] | None = None,
) -> int:
    _require_nonempty_instance(args)
    request_params = dict(params or {})
    # A long one-time op (load/refresh full analysis) raises its no-env default
    # client timeout so it isn't abandoned at the 600s read-op default on a very
    # large binary; BN_REQUEST_TIMEOUT still overrides it (#321).
    timeout_kwargs = (
        {"default_timeout": op_default_timeout} if op_default_timeout is not None else {}
    )
    effective_page_limit = None
    if page_limit is not None and page_limit >= 0:
        effective_page_limit = page_limit
        request_params["limit"] = page_limit + 1

    # #169 L1: --all-instances / --all-targets fan this read across instances
    # and/or targets and aggregate. Gated branch -- the normal single-target path
    # below is untouched when neither flag is set (every existing command/test
    # stays on it).
    if getattr(args, "all_instances", False) or getattr(args, "all_targets", False):
        return _fanout_call(
            args, op, request_params,
            require_target=require_target,
            allow_implicit_target=allow_implicit_target,
            text_renderer=text_renderer,
            timeout_kwargs=timeout_kwargs,
            stem=stem,
        )

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
        **timeout_kwargs,
    )
    if not isinstance(response, dict) or "result" not in response:
        raise BridgeError(
            "malformed bridge reply (no result); the bridge may be stale -- "
            "restart it and retry"
        )
    result = response["result"]
    # Auto-regex fallback (#291.3): a metacharacter query that matched nothing
    # literally is almost always meant as a pattern. Retry it once as a regex and
    # disclose the switch, instead of returning a confident literal `none`. Only
    # re-sends when the literal pass came back empty, so a query that DID match is
    # untouched.
    if regex_fallback_query is not None and _should_retry_as_regex(args, result, regex_fallback_query):
        retry_params = dict(request_params)
        retry_params["regex"] = True
        response = send_request(
            op,
            params=retry_params,
            target=target,
            instance_id=getattr(args, "instance", None),
            spawn_missing_named=spawn_missing_named,
            **timeout_kwargs,
        )
        result = response["result"]
        # An in-band marker so a --format json consumer (which reads stdout, not
        # the stderr note below) can tell the result set came from a regex
        # fallback rather than a literal match (#291.3 review).
        if isinstance(result, dict):
            result["regex_fallback"] = True
        print(
            f'note: "{regex_fallback_query}" matched no function literally; retried as '
            f"a regex (it contains metacharacters). Add --exact for a literal match.",
            file=sys.stderr,
        )
        regex_hint_query = None  # the retry supersedes the add-`--regex` nudge
    # Exit code is computed on the ORIGINAL result (it reads the full results
    # list); a result_transform (e.g. #408 --summary) only changes what is
    # rendered, never the verification-aware exit code.
    exit_code = result_exit_code(result) if result_exit_code is not None else 0
    # No bare-list CLI-side truncation: every paged read returns a {items, total,
    # offset, limit, returned, has_more} envelope and pages bridge-side (#275).
    _maybe_regex_hint(args, result, regex_hint_query)
    _maybe_offset_hint(args, result, offset_hint_identifier)
    if result_transform is not None:
        result = result_transform(result)
    spill_context = result
    fmt = _resolve_output_format(args)
    if text_renderer is not None and fmt == "text":
        try:
            result = text_renderer(result)
        except (AttributeError, TypeError, KeyError, IndexError, ValueError) as exc:
            # A malformed/unexpected bridge result (version skew, future protocol
            # change) must not crash a text renderer with a raw traceback -- that
            # breaks the exit-code contract (main() only catches BridgeError).
            # Surface a clean error pointing at --format json for the raw result (#101).
            raise BridgeError(
                f"could not render the {op} result as text -- the bridge response was "
                f"malformed or newer than this CLI. Rerun with --format json to see the "
                f"raw result. ({type(exc).__name__}: {exc})"
            ) from exc
    spilled = _render_result(
        result,
        fmt=fmt,
        out_path=None if bridge_writes_output else args.out,
        stem=stem,
        spill_label=page_label or op.replace("_", " "),
        spill_context=spill_context,
        # #645: a mutation supplies a compact status to print INSTEAD of a spill
        # envelope, so an atomic write's outcome is always parseable on stdout.
        spill_status=(
            (spill_status(spill_context), spill_status_renderer)
            if spill_status is not None else None
        ),
        # #653.8: stamp WHICH target/instance produced the artifact, so a stale or
        # foreign `--out` file is detectable by inspection rather than by
        # recognising unrelated symbol names.
        provenance={"target": target, "instance": getattr(args, "instance", None)},
        # paged_spill keeps the "--limit/--offset to page" spill hint for
        # commands (function list/search) that page bridge-side and so don't set
        # the client-side page_limit (#59).
        paged=(page_limit is not None) or paged_spill,
    )
    # A text renderer that display-truncates (e.g. xrefs capping caller groups)
    # produces output too small to spill, so the spill pipe-note never fires and a
    # piped grep/wc/jq silently undercounts (#439). When the body did NOT spill
    # (which already warns) and stdout is a pipe, let the command surface an
    # explicit truncation note from the RAW result (spill_context).
    if truncation_note is not None and not spilled and _stdout_is_pipe():
        note = truncation_note(spill_context)
        if note:
            print(note, file=sys.stderr)
    return exit_code


def _fanout_call(
    args: argparse.Namespace,
    op: str,
    request_params: dict[str, Any],
    *,
    require_target: bool,
    allow_implicit_target: bool,
    text_renderer: Callable[[Any], str] | None,
    timeout_kwargs: dict[str, Any],
    stem: str,
) -> int:
    """Run *op* across every running bridge instance (``--all-instances``) and/or
    every open target in each instance (``--all-targets``), and aggregate (#169 L1).

    Without ``--all-targets`` each instance resolves its single target by the
    normal rule (an explicit -t applies to all; else the implicit single target;
    ambiguous/none -> a per-instance ``ok:false`` row, not a hard failure). With
    ``--all-targets`` each instance's open targets are enumerated and fanned. The
    aggregate is one ``{kind: fanout, instances: [...]}`` value rendered
    per-(instance,target) (text via the command's own renderer) or JSON, through
    the normal spill path."""
    fan_instances = getattr(args, "all_instances", False)
    fan_targets = getattr(args, "all_targets", False)
    # Only a -t passed on the CLI counts as explicit (applies to every instance). A
    # STICKY target pin (filled by _apply_sticky_defaults, which sets _sticky_target)
    # is NOT explicit -- it must not suppress the multi-target auto-survey (#368).
    fan_target = getattr(args, "target", None)
    if fan_target is not None and not str(fan_target).strip():
        raise BridgeError(
            "--target is empty: pass a selector from `bn target list`, or "
            "omit --target to survey every target"
        )
    explicit_target = bool(fan_target) and not getattr(
        args, "_sticky_target", False
    )
    if fan_instances:
        instance_ids = [instance_selector(inst) for inst in list_instances()]
        if not instance_ids:
            raise BridgeError("--all-instances: no bridge instances are running")
    else:
        # --all-targets alone: just the resolved/explicit/default instance.
        instance_ids = [getattr(args, "instance", None)]

    def _instance_target_ids(iid: Any) -> list[Any]:
        tresp = send_request("list_targets", params={}, instance_id=iid, **timeout_kwargs)
        titems = tresp["result"]
        tlist = titems.get("items") if isinstance(titems, dict) else titems
        return [t.get("target_id") or t.get("selector")
                for t in (tlist or []) if isinstance(t, dict)]

    # Phase 1 -- PLAN each instance: resolve its target set (a list_targets peek
    # for --all-targets, or the #368 multi-target auto-survey). That peek is itself
    # a per-instance socket round-trip, so run the planning CONCURRENTLY too --
    # otherwise one wedged bridge's list_targets would block the whole survey
    # before any read even starts (#417). Each plan is independent and read-only.
    def _plan_instance(iid: Any) -> dict[str, Any]:
        plan: dict[str, Any] = {"iid": iid, "auto": False}
        if fan_targets:
            try:
                tsels = _instance_target_ids(iid)
            except BridgeError as exc:
                plan["error"] = f"list targets: {exc}"
                return plan
            if not tsels:
                plan["error"] = "no targets open"
                return plan
            plan["tsels"] = tsels
        elif fan_instances and not explicit_target:
            # #368 facet 1: don't drop a MULTI-target instance to an "ambiguous
            # target" error -- survey ALL its targets when there's more than one,
            # else fall back to the normal single resolve. An explicit -t still
            # applies that selector to every instance.
            try:
                ids = _instance_target_ids(iid)
            except BridgeError:
                ids = None
            if ids and all(i is not None for i in ids):
                plan["tsels"] = ids
                plan["auto"] = len(ids) > 1
            else:
                plan["tsels"] = [None]   # peek failed / 0 targets -> normal resolve
        else:
            plan["tsels"] = [None]  # resolve the single target normally below
        return plan

    if len(instance_ids) <= 1:
        plans = [_plan_instance(iid) for iid in instance_ids]
    else:
        with ThreadPoolExecutor(max_workers=min(8, len(instance_ids))) as pool:
            plans = list(pool.map(_plan_instance, instance_ids))

    # Flatten to ordered units; an enumeration error keeps its slot so the final
    # row order stays deterministic (instance order), regardless of which reads
    # finish first under the pool.
    units: list[tuple[str, Any]] = []   # ("error", row) | ("work", (iid, tsel))
    auto_expanded: list[Any] = []       # multi-target instances surveyed in full (#368)
    for plan in plans:
        iid = plan["iid"]
        if plan.get("auto"):
            auto_expanded.append(iid)
        if "error" in plan:
            units.append(("error", {"instance": iid, "ok": False, "error": plan["error"]}))
            continue
        for tsel in plan["tsels"]:
            units.append(("work", (iid, tsel)))

    def _run_one(item: tuple[Any, Any]) -> dict[str, Any]:
        iid, tsel = item
        row: dict[str, Any] = {"instance": iid}
        start = time.monotonic()
        try:
            if tsel is None:
                sub = argparse.Namespace(**vars(args))
                sub.instance = iid
                sub.all_instances = False  # per-(instance,target) call is a normal single call
                sub.all_targets = False
                target = _resolve_target(
                    sub, require_target=require_target,
                    allow_implicit_target=allow_implicit_target,
                )
            else:
                target = tsel
            response = send_request(
                op, params=request_params, target=target, instance_id=iid, **timeout_kwargs
            )
            row.update({"target": target, "ok": True, "result": response["result"]})
        except BridgeError as exc:
            row.update({"target": tsel, "ok": False, "error": str(exc)})
        # Per-row wall-clock so an agent can see WHERE a broad survey spent its
        # time, not just that it felt slow (#417).
        row["duration_ms"] = round((time.monotonic() - start) * 1000, 1)
        return row

    # Phase 2 -- EXECUTE the reads CONCURRENTLY with a bounded pool: each is an
    # independent socket round-trip to a distinct bridge, so a serial sweep over
    # many instances made broad surveys feel wedged (#417). A single item skips
    # the pool. Fan-out is read-only (the @command fanout flag is rejected on
    # mutating commands), so concurrent execution is safe.
    work_items = [payload for kind, payload in units if kind == "work"]
    if len(work_items) <= 1:
        executed = [_run_one(item) for item in work_items]
    else:
        with ThreadPoolExecutor(max_workers=min(8, len(work_items))) as pool:
            executed = list(pool.map(_run_one, work_items))

    # Assemble rows in unit order: enumeration-error rows keep their slot, the
    # executed reads fill the rest -- so output order is stable (instance order)
    # even though the reads completed out of order under the pool.
    rows: list[dict[str, Any]] = []
    executed_iter = iter(executed)
    for kind, payload in units:
        rows.append(payload if kind == "error" else next(executed_iter))

    ok_count = sum(1 for r in rows if r.get("ok"))
    result = {"kind": "fanout", "command": op, "count": len(rows),
              "ok_count": ok_count, "instances": rows}
    # Surface the slowest rows so an agent knows where a long survey went (#417).
    timed = [r for r in rows if isinstance(r.get("duration_ms"), int | float)]
    if len(timed) > 1:
        slowest = sorted(timed, key=lambda r: r["duration_ms"], reverse=True)
        result["slow_rows"] = [
            {"instance": r.get("instance"), "target": r.get("target"),
             "duration_ms": r["duration_ms"]}
            for r in slowest[:3]
        ]
    if auto_expanded:
        # #368 facet 1: be explicit that --all-instances surveyed every target of a
        # multi-target instance (so the row count > instance count is understood,
        # not read as a duplicate/bug).
        result["auto_expanded_instances"] = auto_expanded
    fmt = _resolve_output_format(args)
    rendered: Any = result
    if fmt == "text":
        rendered = _render_fanout_text(result, inner_renderer=text_renderer)
    _render_result(
        rendered, fmt=fmt, out_path=args.out, stem=stem or "fanout",
        spill_label="fanout", spill_context=result, paged=True,
    )
    # Exit non-zero when EVERY instance failed, so a scripted consumer keying on
    # the exit code doesn't read a total failure as success (#169 L1 review). A
    # partial success stays 0; callers inspect the per-instance `ok` rows.
    return 0 if ok_count else 2


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
# A depth-labeled non-negative validator so e.g. `--max-depth -1` reads
# "depth must be an integer >= 0", not the generic "index ..." (#49).
_depth_int = _int_at_least(0, "depth")
# `trace --max-depth` is a *step budget* -- a depth of 0 collects nothing, which
# the bridge rejects as `Invalid max_depth: 0`. Require >= 1 at parse time so the
# CLI contract matches the bridge instead of round-tripping to an error (#129).
_positive_depth_int = _int_at_least(1, "depth")


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
    # Accept both START:END and the natural START-END: the output header prints
    # the range with a hyphen (`// lines 1-6`), so an agent copying that back as
    # `--lines 1-6` must work, not just the colon form (#359). --lines is
    # 1-indexed, so a leading `-` can only be a (rejected) negative START.
    sep = ":" if ":" in value else "-"
    parts = value.split(sep)
    if len(parts) == 1:
        # A bare count `--lines N` means the first N lines (1..N), matching
        # `--count`/`--limit` N, so `disasm --lines 5` is no longer a dead end
        # that errors asking for START:END (#371.4). Use base-0 parsing so the
        # bare form accepts `0x..` exactly like --count/--limit (_positive_int).
        try:
            count = int(parts[0], 0)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(
                f"expected START:END, START-END, or a bare line count, got {value!r}"
            )
        if count < 1:
            raise argparse.ArgumentTypeError(
                f"invalid line count {count}; --lines is 1-indexed (N >= 1)"
            )
        return (1, count)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected START:END or START-END, got {value!r}")
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected START:END or START-END with integers, got {value!r}"
        )
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError(
            f"invalid range {start}-{end}; --lines is 1-indexed (START >= 1, END >= START)"
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
        default=None,
        help="Maximum number of items to return (default: 100; full body with --out)",
    )


def _effective_limit(args: argparse.Namespace) -> int | None:
    """Resolve a paged command's page limit.

    An explicit ``--limit`` always wins. Otherwise the default is 100 for an
    on-screen page, but an unlimited (full-body) export when ``--out`` is given
    -- so ``--out`` writes the complete result instead of silently capping at the
    default page (#165). The argparse default is therefore None (the sentinel for
    "not set"), not 100."""
    if getattr(args, "limit", None) is not None:
        return args.limit
    return None if getattr(args, "out", None) else 100


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
        # RawDescription so the capability map below keeps its line breaks --
        # the default formatter reflows the epilog into one wrapped blob, which
        # destroys a "which command when" table. It only affects description /
        # epilog text; argument and subcommand formatting is unchanged.
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Picking between overlapping commands:\n"
            "\n"
            "  Who calls this function?\n"
            "    callsites  exact caller -> callsite address mapping\n"
            "    xrefs      general cross-references (code and data, plus symbol presence)\n"
            "\n"
            "  Where does a value come from or go?\n"
            "    taint      follow data source -> sink across calls (forward / backward)\n"
            "    trace      backward-slice one call argument to its origin\n"
            "    dataflow   per-function def/use, possible values, and local call graph\n"
            "\n"
            "  C++ classes and raw structure?\n"
            "    class      C++ class hierarchy: vtables, bases, methods (from RTTI/symbols)\n"
            "    evidence   raw vtable/pointer tables, protobuf, .init_array\n"
            "\n"
            "  Finding a function?\n"
            "    function list    enumerate, filter, or count functions\n"
            "    function search  match by name or regex\n"
            "\n"
            "Output over ~10k estimated tokens spills to disk; the command prints an\n"
            "envelope with the artifact path. Read that file directly -- do not pipe to grep."
        ),
    )
    parser.add_argument("--version", action="version", version=f"bn {VERSION}")
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
    out: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in protected_options and index + 1 < len(argv):
            value = argv[index + 1]
            # Rewrite ANY following flag-like token to the --opt=value spelling,
            # including ones that collide with KNOWN options (--format, --target,
            # --limit). These data options are documented as free-form, so
            # `bn strings --query --format` must search the literal "--format",
            # not have argparse consume it as the format flag (#102). A user who
            # genuinely wants --query followed by a real flag uses = themselves.
            if value.startswith("-"):
                out.append(f"{item}={value}")
                index += 2
                continue
        out.append(item)
        index += 1
    return out


def _explicit_instance_options(argv: list[str]) -> tuple[bool, bool]:
    instance = False
    instance_id = False
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--":
            break
        if item in {"-i", "--instance"}:
            instance = True
            index += 2
            continue
        if item.startswith("--instance=") or (
            item.startswith("-i") and not item.startswith("--") and len(item) > 2
        ):
            instance = True
        if item == "--instance-id":
            instance_id = True
            index += 2
            continue
        if item.startswith("--instance-id="):
            instance_id = True
        index += 1
    return instance, instance_id


def _apply_sticky_defaults(args: argparse.Namespace) -> None:
    """Fill unset --instance / --target from per-project sticky state."""
    state = session_state.read()
    # Presence, not truthiness (#690 r3): `-i "$INST"` with $INST unset must
    # not be silently replaced by the pin -- the same doctrine as -t below.
    # The explicit-empty value is rejected in _call before anything is sent.
    if getattr(args, "instance", None) is None:
        sticky_instance = state.get("instance_id")
        if sticky_instance:
            args.instance = sticky_instance
            args._sticky_instance = True
    # Only an ABSENT -t is filled from the pin. An explicit `-t ""` stays as
    # given so the handler can tell "no selector" from "empty selector" (close
    # must reject the latter instead of letting a pin paper over it).
    if getattr(args, "target", None) is None:
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
    (
        args._explicit_instance,
        args._explicit_instance_id,
    ) = _explicit_instance_options(parse_argv)
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
