from __future__ import annotations

import argparse
from typing import Any

from ..cli import _call, arg, command
from ..formatters import (
    _render_callgraph_text,
    _render_defuse_text,
    _render_taint_text,
    _render_values_text,
)
from ..transport import BridgeError


@command("dataflow", "defuse", help="Show the SSA definition site and use sites of a variable",
         target=True,
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
        allow_implicit_target=True,
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
        allow_implicit_target=True,
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
        allow_implicit_target=True,
        text_renderer=_render_values_text,
        stem="values",
    )


_LOCATOR_HELP = (
    "Locator: param:<n> | var:<selector> | ret:<callee> | arg:<callee>:<n>"
)


@command("taint", "forward", help="Forward taint: trace untrusted data from sources to sinks",
         target=True,
         args=[
             arg("--function", "-f", dest="function", required=True,
                 help="Function to analyze (name or address)"),
             arg("--source", dest="sources", action="append", default=None, metavar="LOCATOR",
                 help=f"Taint source (repeatable). {_LOCATOR_HELP}"),
             arg("--max-depth", dest="max_depth", type=int, default=8,
                 help="Max interprocedural recursion depth into callees (default: 8)"),
             arg("--unknown-call", choices=("conservative", "stop"), default="conservative",
                 help="How to treat un-analyzed/external calls reached by taint (default: conservative)"),
         ])
def _taint_forward(args: argparse.Namespace) -> int:
    if not args.sources:
        raise BridgeError("taint forward requires at least one --source")
    return _call(
        args,
        "taint",
        {
            "direction": "forward",
            "function": args.function,
            "sources": list(args.sources),
            "max_depth": int(args.max_depth),
            "unknown_call": args.unknown_call,
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_taint_text,
        stem="taint-forward",
    )


@command("taint", "backward", help="Backward taint: slice a sink's arguments back to their origin",
         target=True,
         args=[
             arg("--function", "-f", dest="function", required=True,
                 help="Function to analyze (name or address)"),
             arg("--sink", dest="sinks", action="append", default=None, metavar="LOCATOR",
                 help=f"Sink to slice from (repeatable). {_LOCATOR_HELP}"),
         ])
def _taint_backward(args: argparse.Namespace) -> int:
    if not args.sinks:
        raise BridgeError("taint backward requires at least one --sink")
    return _call(
        args,
        "taint",
        {
            "direction": "backward",
            "function": args.function,
            "sinks": list(args.sinks),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_taint_text,
        stem="taint-backward",
    )
