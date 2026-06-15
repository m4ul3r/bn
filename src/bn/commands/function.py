from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..cli import _call, _depth_int, _effective_limit, _mutation_exit_code, _non_negative_int, _parse_line_range, _positive_depth_int, _positive_int, arg, command, mutex
from ..formatters import (
    _render_callsites_text,
    _render_evidence_xrefs_text,
    _render_field_xrefs_text,
    _render_function_evidence_text,
    _render_init_arrays_text,
    _render_function_info_text,
    _render_mutation_text,
    _render_function_list_text,
    _render_name_address_list_text,
    _render_message_lens_text,
    _render_pointer_table_text,
    _render_structured_il_text,
    _render_trace_text,
    _render_xrefs_text,
    _slice_text_lines,
    _text_field,
)
from ..transport import BridgeError


@command("function", "list", help="List functions", target=True, paged=True, address_filter=True,
         args=[
             arg("--count", action="store_true", default=False,
                 help="Show total function count instead of listing"),
             arg("--sort", choices=["address", "size", "name"], default="address",
                 help="Order results: address (default), size (largest first), or name"),
         ])
def _function_list(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    if args.min_address is not None:
        params["min_address"] = args.min_address
    if args.max_address is not None:
        params["max_address"] = args.max_address
    if args.offset:
        params["offset"] = args.offset
    if args.count:
        params["count_only"] = True
        return _call(
            args,
            "list_functions",
            params,
            require_target=True,
            allow_implicit_target=True,
            text_renderer=lambda value: f"Total functions: {value.get('count', 0)}",
            stem="function-count",
        )
    # Bridge-authoritative paging: send the real limit/offset (not the generic
    # +1 page_limit) so the bridge returns the page WITH the true total, which
    # the renderer surfaces (#59). The bridge envelope is {functions, total, ...}.
    # _effective_limit defaults to 100 but uncaps for --out full-body export (#165).
    limit = _effective_limit(args)
    if limit is not None:
        params["limit"] = limit
    if args.sort != "address":
        params["sort"] = args.sort
    return _call(
        args,
        "list_functions",
        params,
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_function_list_text,
        page_label="function list",
        paged_spill=True,
        stem="functions",
    )


@command("function", "search", help="Search functions by substring or regex",
         target=True, paged=True, address_filter=True,
         args=[
             arg("query"),
             arg("--sort", choices=["address", "size", "name"], default="address",
                 help="Order results: address (default), size (largest first), or name"),
         ],
         mutex_groups=[
             mutex(False,
                   arg("--regex", action="store_true",
                       help="Interpret query as a case-insensitive regular expression"),
                   arg("--exact", action="store_true", default=False,
                       help="Match function names exactly (case-insensitive); avoids substring "
                            "false positives in C++ mangled names")),
         ])
def _function_search(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {
        "query": args.query,
        "regex": bool(args.regex),
        "exact": bool(args.exact),
    }
    if args.min_address is not None:
        params["min_address"] = args.min_address
    if args.max_address is not None:
        params["max_address"] = args.max_address
    if args.offset:
        params["offset"] = args.offset
    limit = _effective_limit(args)
    if limit is not None:
        params["limit"] = limit
    if args.sort != "address":
        params["sort"] = args.sort
    return _call(
        args,
        "search_functions",
        params,
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_function_list_text,
        page_label="function search",
        paged_spill=True,
        stem="function-search",
        regex_hint_query=args.query,
    )


@command("function", "info", help="Show function prototype and variables", target=True,
         args=[arg("identifier", help="Function name or entry address (hex 0x.. or decimal)"),
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


@command("function", "create",
         help="Create and analyze a function at an address auto-analysis missed",
         target=True, fmt="json",
         args=[
             arg("--preview", action="store_true",
                 help="Create, verify, then revert without committing"),
             arg("address", help="Address of the function entry point (hex or decimal)"),
         ])
def _function_create(args: argparse.Namespace) -> int:
    return _call(
        args,
        "function_create",
        {
            "address": args.address,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="function-create",
        result_exit_code=_mutation_exit_code,
    )


def _require_text_format(args: argparse.Namespace, flag: str) -> None:
    """Reject display-only flags outside text mode instead of ignoring them.

    Flags like --lines and xrefs --limit only affect the text renderer; JSON
    and ndjson output always carry the full payload, so silently accepting
    them would misrepresent what the caller asked for.
    """
    if args.format != "text":
        raise BridgeError(f"{flag} only applies to --format text")


@command("decompile", help="Render Binary Ninja Pseudo C for a function", target=True,
         args=[
             arg("identifier", help="Function name or entry address (hex 0x.. or decimal)"),
             arg("--addresses", action="store_true", default=False,
                 help="Show address prefixes on each line"),
             arg("--lines", type=_parse_line_range, default=None, metavar="START:END",
                 help="Show only lines START through END (1-indexed, inclusive)"),
             arg("--force-analysis", action="store_true", default=False,
                 help="If Binary Ninja skipped this function (e.g. too large), override the skip "
                      "and reanalyze it before decompiling (may be slow; takes the write lock)"),
         ])
def _decompile(args: argparse.Namespace) -> int:
    lines_range = getattr(args, "lines", None)
    if lines_range is not None:
        _require_text_format(args, "--lines")

    def _render_decompile_text(value: Any) -> str:
        text = _slice_text_lines(_text_field("text")(value), lines_range)
        warnings = value.get("warnings") if isinstance(value, dict) else None
        if warnings:
            text = text + "\n\n" + "\n".join(f"warning: {warning}" for warning in warnings)
        return text

    return _call(
        args,
        "decompile",
        {
            "identifier": args.identifier,
            "addresses": args.addresses,
            "force_analysis": args.force_analysis,
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_decompile_text,
        stem="decompile",
    )


@command("il", help="Dump IL for a function", target=True,
         args=[
             arg("identifier", help="Function name or entry address (hex 0x.. or decimal)"),
             arg("--view", choices=("hlil", "mlil", "llil"), default="hlil",
                 help="IL level to dump: hlil (default), mlil, or llil"),
             arg("--ssa", action="store_true",
                 help="Emit the SSA form of the selected IL view"),
             arg("--lines", type=_parse_line_range, default=None, metavar="START:END",
                 help="Show only lines START through END (1-indexed, inclusive)"),
         ])
def _il(args: argparse.Namespace) -> int:
    lines_range = getattr(args, "lines", None)
    if lines_range is not None:
        _require_text_format(args, "--lines")
    base = _text_field("text")
    return _call(
        args,
        "il",
        {"identifier": args.identifier, "view": args.view, "ssa": bool(args.ssa)},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=lambda value: _slice_text_lines(base(value), lines_range),
        stem="il",
    )


@command("function", "structured-il",
         help="Per-instruction structured IL (op, vars_read/written) for data-flow tooling",
         target=True,
         args=[
             arg("identifier", help="Function name or entry address (hex 0x.. or decimal)"),
             arg("--view", choices=("hlil", "mlil"), default="mlil",
                 help="IL level (default: mlil)"),
             arg("--no-ssa", dest="ssa", action="store_false", default=True,
                 help="Emit non-SSA form (default: SSA)"),
         ])
def _function_structured_il(args: argparse.Namespace) -> int:
    return _call(
        args,
        "structured_il",
        {"identifier": args.identifier, "view": args.view, "ssa": bool(args.ssa)},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_structured_il_text,
        stem="structured-il",
    )


@command("disasm", help="Disassemble a function", target=True,
         args=[
             arg("identifier", help="Function name or entry address (hex 0x.. or decimal)"),
             arg("--lines", type=_parse_line_range, default=None, metavar="START:END",
                 help="Show only lines START through END (1-indexed, inclusive)"),
         ])
def _disasm(args: argparse.Namespace) -> int:
    lines_range = getattr(args, "lines", None)
    if lines_range is not None:
        _require_text_format(args, "--lines")
    base = _text_field("text")
    return _call(
        args,
        "disasm",
        {"identifier": args.identifier},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=lambda value: _slice_text_lines(base(value), lines_range),
        stem="disasm",
    )


@command("xrefs", help="List xrefs to an address or function; use --field for struct field xrefs",
         target=True, paged=True,
         args=[
             arg("identifier", nargs="?",
                 help="Function name or address (hex 0x.. or decimal) to find inbound refs to"),
             arg("--field", dest="field_spec",
                 help="Struct field xref spec (e.g., TrackRowCell.tile_type)"),
         ])
def _xrefs(args: argparse.Namespace) -> int:
    field_spec = getattr(args, "field_spec", None)
    identifier = getattr(args, "identifier", None)
    if field_spec and identifier:
        raise BridgeError(
            "xrefs takes either an identifier or --field, not both "
            f"(got {identifier!r} and --field {field_spec!r})"
        )
    if field_spec:
        # --field is a distinct shape (field info + refs); paging args don't apply.
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
    # xrefs now adopts the canonical paging envelope (#164): JSON carries
    # {items, total, offset, limit, returned, has_more} (each item keeps its
    # kind) and --limit pages it instead of erroring. The text renderer reads the
    # back-compat code_refs/data_refs (full) and uses `limit` as a caller-group
    # display cap, so text behavior is unchanged.
    params: dict[str, Any] = {"identifier": identifier}
    if args.offset:
        params["offset"] = args.offset
    limit = _effective_limit(args)
    if limit is not None:
        params["limit"] = limit
    return _call(
        args,
        "xrefs",
        params,
        require_target=True,
        allow_implicit_target=True,
        text_renderer=lambda v: _render_xrefs_text(v, limit=limit),
        offset_hint_identifier=identifier,
        paged_spill=True,
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
             arg("callee", help="Callee function name or address whose callsites to locate"),
             arg("--context", type=_non_negative_int, default=3,
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


@command("evidence", "function",
         help="Summarize generic function evidence: thunk candidates, calls, IL, and argument hints",
         target=True,
         args=[
             arg("identifier", help="Function name or entry address (hex 0x.. or decimal)"),
             arg("--context", type=_non_negative_int, default=2,
                 help="Number of previous and next disassembly instructions to include around calls"),
         ])
def _evidence_function(args: argparse.Namespace) -> int:
    return _call(
        args,
        "function_evidence",
        {
            "identifier": args.identifier,
            "context": args.context,
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_function_evidence_text,
        stem="function-evidence",
    )


@command("evidence", "xrefs",
         help="List xrefs with section/segment/symbol/disassembly context",
         target=True, paged=True,
         args=[
             arg("identifier", help="Function name or address (hex 0x.. or decimal) to find inbound refs to"),
         ])
def _evidence_xrefs(args: argparse.Namespace) -> int:
    # Same canonical paging envelope as `xrefs` (#164): --limit pages the JSON
    # items; the text renderer reads the full code_refs/data_refs with `limit` as
    # a per-bucket display cap.
    params: dict[str, Any] = {"identifier": args.identifier}
    if args.offset:
        params["offset"] = args.offset
    limit = _effective_limit(args)
    if limit is not None:
        params["limit"] = limit
    return _call(
        args,
        "xrefs",
        params,
        require_target=True,
        allow_implicit_target=True,
        text_renderer=lambda value: _render_evidence_xrefs_text(value, limit=limit),
        paged_spill=True,
        stem="evidence-xrefs",
    )


@command("evidence", "table",
         help="Interpret memory at an address as a pointer table or vtable-like table",
         target=True,
         args=[
             arg("address", help="Table start address (hex 0x.. or decimal)"),
             arg("--entries", type=_positive_int, default=16,
                 help="Number of pointer entries to read"),
             arg("--stride", default=None,
                 help="Byte stride between entries (default: target pointer size)"),
         ])
def _evidence_table(args: argparse.Namespace) -> int:
    return _call(
        args,
        "pointer_table",
        {
            "address": args.address,
            "entries": args.entries,
            "stride": args.stride,
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_pointer_table_text,
        stem="pointer-table",
    )


@command("evidence", "message",
         help="Summarize message/type-name strings, xrefs, and nearby metadata tables",
         target=True,
         args=[
             arg("query", help="Message/type-name string substring to locate"),
             arg("--limit", type=_positive_int, default=20,
                 help="Max matching strings to summarize; result cap in all formats "
                      "(the reported total stays honest, with truncated=true when capped)"),
             arg("--table-entries", type=_non_negative_int, default=6,
                 help="Pointer entries to show around metadata data refs (0 = none)"),
         ])
def _evidence_message(args: argparse.Namespace) -> int:
    return _call(
        args,
        "message_lens",
        {
            "query": args.query,
            "limit": args.limit,
            "table_entries": args.table_entries,
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_message_lens_text,
        stem="message-lens",
    )


@command("evidence", "init",
         help="Summarize constructor/destructor pointer sections such as .init_array and .ctors",
         target=True,
         args=[
             arg("--limit", type=_positive_int, default=64,
                 help="Maximum entries to show per constructor/destructor section"),
         ])
def _evidence_init(args: argparse.Namespace) -> int:
    return _call(
        args,
        "init_arrays",
        {"limit": args.limit},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_init_arrays_text,
        stem="init-arrays",
    )


@command("trace",
         help="Backward slice: trace a call argument through SSA use-def chains to its origin",
         target=True,
         args=[
             arg("identifier", help="Function name or entry address containing the call"),
             arg("address", help="Address of the call instruction to trace from (hex 0x.. or decimal)"),
             arg("--arg", type=_non_negative_int, default=0,
                 help="Zero-based argument index to trace (default: 0)"),
             arg("--view", default="mlil", choices=("mlil", "hlil"),
                 help="IL view for SSA walking (default: mlil, broadest call coverage; "
                      "hlil misses calls whose return value is assigned)"),
             arg("--max-depth", type=_positive_depth_int, default=50,
                 help="Maximum trace steps before truncation (>= 1; default: 50)"),
             arg("--interprocedural", action="store_true", default=False,
                 help="Follow return values across call boundaries into callees"),
             arg("--ip-depth", type=_depth_int, default=2,
                 help="Max call depth for interprocedural tracing (default: 2; 0 disables crossing)"),
         ])
def _trace(args: argparse.Namespace) -> int:
    return _call(
        args,
        "backward_slice",
        {
            "identifier": args.identifier,
            "address": args.address,
            "arg_index": args.arg,
            "view": args.view,
            "max_depth": args.max_depth,
            "interprocedural": args.interprocedural,
            "ip_depth": args.ip_depth,
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_trace_text,
        stem="trace",
    )
