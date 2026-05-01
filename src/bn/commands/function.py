from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..cli import _call, _parse_line_range, arg, command, mutex
from ..formatters import (
    _render_callsites_text,
    _render_field_xrefs_text,
    _render_function_info_text,
    _render_name_address_list_text,
    _render_xrefs_text,
    _text_field,
)
from ..transport import BridgeError


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
