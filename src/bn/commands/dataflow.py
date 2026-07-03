from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..cli import _call, _depth_int, arg, command
from ..formatters import (
    _render_callgraph_text,
    _render_defuse_text,
    _render_taint_models_text,
    _render_taint_text,
    _render_values_text,
)
from ..transport import BridgeError


def _models_arg() -> tuple[tuple[str, ...], dict[str, Any]]:
    """The shared `--models FILE` option for taint commands (#317): load
    additional, project-specific taint models so taint follows flows through a
    target's own (non-libc-named) copy/format/exec wrappers."""
    return arg("--models", dest="models", default=None, metavar="FILE",
               help="JSON file of extra taint models (same schema as the builtin DB: "
                    'a {name: model} map) for project-internal copy/format/exec wrappers, '
                    "merged over the builtins so taint follows flows through your own "
                    "wrappers (#317). Persist them globally instead via BN_TAINT_MODELS.")


def _add_user_models(args: argparse.Namespace, params: dict[str, Any]) -> None:
    """Read `--models <file>` into ``params['user_models']`` (loud on a bad file,
    mirroring --resolve-map and the #97 no-silent-model-failure rule)."""
    path = getattr(args, "models", None)
    if not path:
        return
    try:
        params["user_models"] = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BridgeError(f"could not read --models {path}: {exc}")
    # #415: pass the file path through so the run's model_sources disclosure can
    # name WHICH --models file landed, not just a count.
    params["user_models_path"] = str(path)


@command("dataflow", "defuse", help="Show the SSA definition site and use sites of a variable",
         target=True,
         prefer_when="per-function SSA def/use of one variable; "
                     "use taint to follow a value across calls source->sink",
         see_also=("taint forward", "taint backward"),
         args=[
             arg("identifier", help="Function name or entry address (hex 0x.. or decimal)"),
             arg("--var", dest="var", required=True,
                 help="Variable selector: name, local_id, or name#version (SSA)"),
         ])
def _dataflow_defuse(args: argparse.Namespace) -> int:
    return _call(
        args,
        "defuse",
        {"identifier": args.identifier, "var": args.var},
        require_target=True,
        text_renderer=_render_defuse_text,
        stem="defuse",
    )


@command("dataflow", "callgraph", help="Resolved callees/callers (indirect targets via value-set)",
         target=True,
         args=[
             arg("identifier", help="Function name or entry address (hex 0x.. or decimal)"),
             arg("--direction", choices=("callees", "callers", "both"), default="both",
                 help="Which edges to resolve (default: both)"),
             arg("--no-resolve-indirect", dest="resolve_indirect", action="store_false", default=True,
                 help="Skip value-set resolution of indirect call targets"),
         ])
def _dataflow_callgraph(args: argparse.Namespace) -> int:
    return _call(
        args,
        "resolved_calls",
        {
            "identifier": args.identifier,
            "direction": args.direction,
            "resolve_indirect": bool(args.resolve_indirect),
        },
        require_target=True,
        text_renderer=_render_callgraph_text,
        stem="callgraph",
    )


@command("dataflow", "values", help="Value-set (possible values) at an address",
         target=True,
         args=[
             arg("identifier", help="Function name or entry address (hex 0x.. or decimal)"),
             arg("--at", dest="at", required=True,
                 help="Instruction address (hex 0x.. or decimal) within the function"),
         ])
def _dataflow_values(args: argparse.Namespace) -> int:
    return _call(
        args,
        "possible_values",
        {"identifier": args.identifier, "at": args.at},
        require_target=True,
        text_renderer=_render_values_text,
        stem="values",
    )


# param:/arg: indices are 0-based -- for memcpy(dst, src, len) the length is
# arg:memcpy:2, not :3 (#291.4).
_LOCATOR_HELP = (
    "Locator: param:<n> | var:<selector> | ret:<callee> | arg:<callee>:<n> | "
    "call:<callee> (seeds every output the callee's model declares -- return "
    "value AND output-pointer buffers). param/arg indices are 0-based."
)

_SINK_LOCATOR_HELP = (
    "Locator: param:<n> | var:<selector> | arg:<callee>:<n> (ret: is forward-only). "
    "param/arg indices are 0-based (e.g. memcpy length is arg:memcpy:2)."
)


@command("taint", "forward", help="Forward taint: trace untrusted data from sources to sinks",
         target=True,
         prefer_when="follow untrusted data forward source->sink across calls; "
                     "use dataflow for per-function def/use, evidence for raw structure",
         see_also=("taint backward", "trace", "dataflow defuse"),
         args=[
             arg("--function", "-f", dest="function", required=True,
                 help="Function to analyze (name or address)"),
             arg("--source", dest="sources", action="append", default=None, metavar="LOCATOR",
                 required=True,
                 help=f"Taint source (repeatable, at least one required). {_LOCATOR_HELP}"),
             arg("--max-depth", dest="max_depth", type=_depth_int, default=8,
                 help="Max interprocedural recursion depth into callees (default: 8; "
                      "0 = intraprocedural only)"),
             arg("--resolve-map", dest="resolve_map", default=None, metavar="FILE",
                 help="JSON file mapping indirect call addresses to target lists: "
                      '{"0x4011f0": ["0x401176", "0x401195"]}'),
             arg("--unknown-call", choices=("conservative", "stop"), default="conservative",
                 help="How to treat un-analyzed/external calls reached by taint (default: conservative)"),
             arg("--sink-class", dest="sink_classes", action="append", default=None,
                 choices=("file_write", "net_write"), metavar="CLASS",
                 help="Enable an opt-in sink class (repeatable). Off by default: "
                      "file_write (fwrite/write/fputs), net_write (send/sendto). "
                      "Use when auditing persistence / file-corruption / exfiltration paths."),
             _models_arg(),
             arg("--verbose", "-v", "--full", dest="full", action="store_true", default=False,
                 help="Show the full SSA path/slice for each flow (default: one compact line per flow)"),
         ])
def _taint_forward(args: argparse.Namespace) -> int:
    # --source is argparse-required, so an empty list cannot reach here.
    params: dict[str, Any] = {
        "direction": "forward",
        "function": args.function,
        "sources": list(args.sources),
        "max_depth": int(args.max_depth),
        "unknown_call": args.unknown_call,
        "enabled_sink_classes": list(args.sink_classes or []),
    }
    if args.resolve_map:
        try:
            params["resolve_map"] = json.loads(Path(args.resolve_map).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BridgeError(f"could not read --resolve-map {args.resolve_map}: {exc}")
    _add_user_models(args, params)
    return _call(
        args,
        "taint",
        params,
        require_target=True,
        text_renderer=(lambda v: _render_taint_text(v, full=bool(getattr(args, "full", False)))),
        stem="taint-forward",
    )


@command("taint", "backward", help="Backward taint: slice a sink's arguments back to their origin",
         target=True,
         prefer_when="slice a sink backward to its sources across calls; "
                     "use trace for a single argument's origin",
         see_also=("taint forward", "trace"),
         args=[
             arg("--function", "-f", dest="function", required=True,
                 help="Function to analyze (name or address)"),
             arg("--sink", dest="sinks", action="append", default=None, metavar="LOCATOR",
                 required=True,
                 help=f"Sink to slice from (repeatable, at least one required). {_SINK_LOCATOR_HELP}"),
             arg("--max-depth", dest="max_depth", type=_depth_int, default=8,
                 help="Max interprocedural depth to follow slices up into callers (default: 8; "
                      "0 = intraprocedural only). The in-function def-chain walk caps at 64 "
                      "steps; truncation is recorded under assumptions."),
             arg("--resolve-map", dest="resolve_map", default=None, metavar="FILE",
                 help="JSON file mapping indirect call addresses to target lists: "
                      '{"0x4011f0": ["0x401176", "0x401195"]}'),
             _models_arg(),
             arg("--verbose", "-v", "--full", dest="full", action="store_true", default=False,
                 help="Show the full SSA path/slice for each flow (default: one compact line per flow)"),
         ])
def _taint_backward(args: argparse.Namespace) -> int:
    # --sink is argparse-required, so an empty list cannot reach here.
    params: dict[str, Any] = {
        "direction": "backward",
        "function": args.function,
        "sinks": list(args.sinks),
        "max_depth": int(args.max_depth),
    }
    if args.resolve_map:
        try:
            params["resolve_map"] = json.loads(Path(args.resolve_map).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BridgeError(f"could not read --resolve-map {args.resolve_map}: {exc}")
    _add_user_models(args, params)
    return _call(
        args,
        "taint",
        params,
        require_target=True,
        text_renderer=(lambda v: _render_taint_text(v, full=bool(getattr(args, "full", False)))),
        stem="taint-backward",
    )


@command("taint", "models",
         help="List the taint model catalog (sources/sinks/propagators); with a target, which are present",
         target=True,
         prefer_when="enumerate known sinks/sources and (with a target) which appear in this binary",
         see_also=("taint forward", "taint backward"),
         args=[
             arg("--role", choices=("source", "sink", "propagator"), default=None,
                 help="Filter to one role"),
             arg("--class", dest="sink_class", default=None, metavar="CLASS",
                 help="Filter sinks to one bug class (e.g. overflow_len); implies --role sink"),
             arg("--present", action="store_true", default=False,
                 help="With a target: show only models present in the binary (errors without a target)"),
             arg("--callsites", action="store_true", default=False,
                 help="With --present: expand each present sink's callsite addresses"),
         ])
def _taint_models(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    if args.role:
        params["role"] = args.role
    if args.sink_class:
        params["class"] = args.sink_class
    if args.present:
        params["present"] = True
    if args.callsites:
        params["callsites"] = True
    return _call(
        args,
        "taint_models",
        params,
        require_target=False,               # catalog dump works with no target
        text_renderer=_render_taint_models_text,
        stem="taint-models",
    )
