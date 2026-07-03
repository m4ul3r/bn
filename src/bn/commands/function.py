from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..cli import _call, _depth_int, _effective_limit, _mutate, _non_negative_int, _parse_line_range, _pick, _positive_depth_int, _positive_int, arg, command, mutex, preview_arg, summary_arg
from ..formatters import (
    _render_callsites_text,
    _disasm_linear_steer_note,
    _render_disasm_linear_text,
    _render_evidence_xrefs_text,
    _render_field_xrefs_text,
    _render_function_count_text,
    _render_function_evidence_text,
    _render_init_arrays_text,
    _render_function_info_text,
    _render_function_list_text,
    _render_name_address_list_text,
    _render_message_lens_text,
    _render_orient_text,
    _render_pointer_table_text,
    _render_structured_il_text,
    _render_trace_text,
    _render_xrefs_any_text,
    _render_xrefs_text,
    _resolution_note,
    _slice_text_lines,
    _text_field,
    _xref_buckets,
    _group_refs_by_caller,
)
from ..transport import BridgeError


@command("function", "list", help="List functions", target=True, paged=True, address_filter=True,
         fanout=True,
         prefer_when="enumerate, filter, or count functions; "
                     "use function search to match by name or regex",
         see_also=("function search",),
         args=[
             arg("--count", action="store_true", default=False,
                 help="Show total function count instead of listing"),
             arg("--sort", choices=["address", "size", "name"], default="address",
                 help="Order results: address (default), size (largest first), or name"),
             arg("--reverse", "--desc", action="store_true", default=False, dest="reverse",
                 help="Reverse the sort's natural order (e.g. --sort size --reverse = smallest first)"),
             arg("--demangle", action="store_true", default=False,
                 help="Show demangled C++ names in text (JSON always carries display_name)"),
             arg("--min-size", type=_positive_int, default=None, dest="min_size", metavar="N",
                 help="Only functions whose byte size is >= N (drop tiny PLT/GOT thunk "
                      "veneers, typically <= 16 bytes)"),
         ])
def _function_list(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    if args.min_address is not None:
        params["min_address"] = args.min_address
    if args.max_address is not None:
        params["max_address"] = args.max_address
    if getattr(args, "min_size", None) is not None:
        params["min_size"] = args.min_size
    if args.offset:
        params["offset"] = args.offset
    if args.count:
        params["count_only"] = True
        return _call(
            args,
            "list_functions",
            params,
            require_target=True,
            text_renderer=_render_function_count_text,
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
    if args.reverse:
        params["reverse"] = True
    return _call(
        args,
        "list_functions",
        params,
        require_target=True,
        text_renderer=lambda value: _render_function_list_text(value, demangle=args.demangle),
        page_label="function list",
        paged_spill=True,
        stem="functions",
    )


@command("function", "search", help="Search functions by substring or regex",
         fanout=True,
         target=True, paged=True, address_filter=True,
         prefer_when="match functions by name or regex; "
                     "use function list to enumerate, filter, or count",
         see_also=("function list",),
         args=[
             arg("query", nargs="?",
                 help="Substring or regex to match function names"),
             arg("--query", dest="query_flag", default=None,
                 help="Alias for the positional query (matches `strings --query` / `types --query`)"),
             arg("--count", action="store_true", default=False,
                 help="Show match count instead of listing"),
             arg("--sort", choices=["address", "size", "name"], default="address",
                 help="Order results: address (default), size (largest first), or name"),
             arg("--reverse", "--desc", action="store_true", default=False, dest="reverse",
                 help="Reverse the sort's natural order (e.g. --sort size --reverse = smallest first)"),
             arg("--demangle", action="store_true", default=False,
                 help="Show demangled C++ names in text (JSON always carries display_name)"),
             arg("--min-size", type=_positive_int, default=None, dest="min_size", metavar="N",
                 help="Only functions whose byte size is >= N (drop tiny PLT/GOT thunk "
                      "veneers, typically <= 16 bytes)"),
         ],
         mutex_groups=[
             mutex(False,
                   arg("--regex", action="store_true",
                       help="Interpret query as a case-insensitive regular expression"),
                   arg("--exact", action="store_true", default=False,
                       help="Match function names exactly (case-insensitive); avoids substring "
                            "false positives in C++ mangled names"),
                   arg("--word", action="store_true", default=False,
                       help="Match the query as a whole identifier token (word-boundary): for a "
                            "sink survey, `--word popen` hits popen/popen@plt but not the "
                            "substring FPs zipOpenArchive/my_popen_wrapper")),
         ])
def _function_search(args: argparse.Namespace) -> int:
    # #410: accept the query positionally OR via --query (matches strings/types
    # muscle memory). _pick errors on both-different / neither.
    query = _pick(args.query, getattr(args, "query_flag", None), "function search query")
    params: dict[str, Any] = {
        "query": query,
        "regex": bool(args.regex),
        "exact": bool(args.exact),
        "word": bool(getattr(args, "word", False)),
    }
    if args.min_address is not None:
        params["min_address"] = args.min_address
    if args.max_address is not None:
        params["max_address"] = args.max_address
    if getattr(args, "min_size", None) is not None:
        params["min_size"] = args.min_size
    if args.count:
        params["count_only"] = True
        return _call(
            args,
            "search_functions",
            params,
            require_target=True,
            text_renderer=_render_function_count_text,
            stem="function-search-count",
            # --count is the "is my query matching anything?" use case, so the
            # auto-regex retry (and its 0-result fallback hint) is most useful
            # here too (#252 review, #291.3).
            regex_hint_query=query,
            regex_fallback_query=query,
        )
    if args.offset:
        params["offset"] = args.offset
    limit = _effective_limit(args)
    if limit is not None:
        params["limit"] = limit
    if args.sort != "address":
        params["sort"] = args.sort
    if args.reverse:
        params["reverse"] = True
    return _call(
        args,
        "search_functions",
        params,
        require_target=True,
        text_renderer=lambda value: _render_function_list_text(value, demangle=args.demangle),
        page_label="function search",
        paged_spill=True,
        stem="function-search",
        regex_hint_query=query,
        regex_fallback_query=query,
    )


@command("function", "info", help="Show function prototype and variables", target=True,
         args=[arg("identifier", help="Function name or entry address (hex 0x.. or decimal)"),
               arg("--verbose", "-v", action="store_true", default=False,
                   help="Show full parameter and local variable details"),
               arg("--demangle", action="store_true", default=False,
                   help="Show the demangled C++ name in text (JSON always carries display_name)")])
def _function_info(args: argparse.Namespace) -> int:
    verbose = getattr(args, "verbose", False)
    demangle = getattr(args, "demangle", False)
    return _call(
        args,
        "function_info",
        {"identifier": args.identifier},
        require_target=True,
        text_renderer=lambda v: _render_function_info_text(v, verbose=verbose, demangle=demangle),
        stem="function-info",
    )


@command("function", "create",
         help="Create and analyze a function at an address auto-analysis missed",
         target=True, fmt="json",
         args=[
             preview_arg("Create, verify, then revert without committing"), summary_arg(),
             arg("address", help="Address of the function entry point (hex or decimal)"),
         ])
def _function_create(args: argparse.Namespace) -> int:
    return _mutate(
        args,
        "function_create",
        {"address": args.address},
        preview=bool(args.preview),
        stem="function-create",
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
        return _resolution_note(value) + text

    return _call(
        args,
        "decompile",
        {
            "identifier": args.identifier,
            "addresses": args.addresses,
            "force_analysis": args.force_analysis,
        },
        require_target=True,
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
        text_renderer=lambda value: _resolution_note(value) + _slice_text_lines(base(value), lines_range),
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
             arg("--lines", type=_parse_line_range, default=None, metavar="START:END",
                 help="Show only lines START through END (1-indexed, inclusive)"),
         ])
def _function_structured_il(args: argparse.Namespace) -> int:
    lines_range = getattr(args, "lines", None)
    if lines_range is not None:
        _require_text_format(args, "--lines")
    return _call(
        args,
        "structured_il",
        {"identifier": args.identifier, "view": args.view, "ssa": bool(args.ssa)},
        require_target=True,
        text_renderer=lambda value: _slice_text_lines(_render_structured_il_text(value), lines_range),
        stem="structured-il",
    )


@command("disasm", help="Disassemble a function (slice with --lines or --count)", target=True,
         args=[
             arg("identifier", help="Function name or entry address (hex 0x.. or decimal)"),
             # #382: BN defaults a whole ARM binary to one mode (often thumb2), so
             # --linear at an ARM-mode region decodes as Thumb (or vice versa).
             # Force the decode mode for --linear; the address's own function arch
             # is honored automatically when known.
             arg("--mode", choices=("arm", "thumb"), default=None,
                 help="Force ARM or Thumb decode for --linear on an ARM target"),
         ],
         # --lines and --count are two spellings of the same slice; one disasm
         # line is one instruction, so --count N is the first N instructions
         # (#291.2). Mutually exclusive -- combining a window and a count is
         # ambiguous.
         mutex_groups=[
             mutex(False,
                   arg("--lines", type=_parse_line_range, default=None, metavar="START:END",
                       help="Show only lines START through END (1-indexed, inclusive)"),
                   arg("--count", "--limit", type=_positive_int, default=None, metavar="N",
                       dest="count",
                       help="Show only the first N instructions (one instruction per line; "
                            "--limit is an accepted alias)"),
                   # --linear is a different MODE, not a slice of a function: it
                   # linearly disassembles N instructions from any mapped address,
                   # even one BN left as data (a missed handler / vtable slot), so
                   # you can inspect the bytes before `function create` (#314).
                   arg("--linear", nargs="?", const=32, type=_positive_int, default=None, metavar="N",
                       help="Linear-disassemble N instructions (default 32) from any mapped "
                            "address, independent of function membership")),
         ])
def _disasm(args: argparse.Namespace) -> int:
    linear = getattr(args, "linear", None)
    mode = getattr(args, "mode", None)
    if mode is not None and linear is None:
        raise BridgeError("--mode applies only to --linear disassembly")
    if linear is not None:
        # Linear mode: the bridge walks N instructions from the address and
        # returns exactly that window, so there is no client-side slice and no
        # text-only restriction -- it works in JSON too.
        params: dict[str, Any] = {"identifier": args.identifier, "linear": int(linear)}
        if mode is not None:
            params["mode"] = mode
        return _call(
            args,
            "disasm",
            params,
            require_target=True,
            text_renderer=_render_disasm_linear_text,
            stem="disasm",
        )
    count = getattr(args, "count", None)
    lines_range = getattr(args, "lines", None)
    if count is not None:
        # disasm is one instruction per line, so "first N instructions" is the
        # 1-indexed window 1..N -- reuse the shared slicer (header + spill).
        lines_range = (1, count)
    slice_flag = "--count" if count is not None else "--lines"
    if lines_range is not None:
        _require_text_format(args, slice_flag)
    base = _text_field("text")
    sliced = lines_range is not None
    return _call(
        args,
        "disasm",
        {"identifier": args.identifier},
        require_target=True,
        text_renderer=lambda value: _resolution_note(value)
        + _disasm_linear_steer_note(value, sliced=sliced)
        + _slice_text_lines(base(value), lines_range, flag=slice_flag),
        stem="disasm",
    )


@command("xrefs", help="List xrefs to an address or function; use --field for struct field xrefs",
         target=True, paged=True,
         prefer_when="general cross-references -- code and data refs, plus symbol presence; "
                     "use callsites for an exact caller->callsite address mapping",
         see_also=("callsites",),
         args=[
             arg("identifier", nargs="?",
                 help="Function name or address (hex 0x.. or decimal) to find inbound refs to"),
             arg("--field", dest="field_spec",
                 help="Struct field xref spec (e.g., TrackRowCell.tile_type)"),
             arg("--any", nargs="+", dest="any_symbols", metavar="SYMBOL",
                 help="Batch-probe several symbols (a sink sweep); absent symbols are "
                      "reported, not errors"),
         ])
def _xrefs(args: argparse.Namespace) -> int:
    field_spec = getattr(args, "field_spec", None)
    identifier = getattr(args, "identifier", None)
    any_symbols = getattr(args, "any_symbols", None)
    if any_symbols:
        # #410: accept comma-separated lists as well as space-separated, so
        # `--any read,recv,memcpy` (or `read, recv, memcpy`) probes three symbols
        # instead of one bogus "read,recv,memcpy" symbol (silent miss in a sink
        # sweep). Strip per-token whitespace so "read, recv" yields "recv", not
        # " recv" (which the exact-name bridge lookup would report absent).
        any_symbols = [t for chunk in any_symbols
                       for s in (chunk.split(",") if "," in chunk else [chunk])
                       if (t := s.strip())]
    if any_symbols:
        if identifier or field_spec:
            raise BridgeError("xrefs --any takes only the symbol list, not an identifier or --field")
        return _call(
            args,
            "xrefs_any",
            {"symbols": any_symbols},
            require_target=True,
            text_renderer=_render_xrefs_any_text,
            stem="xrefs-any",
        )
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
            text_renderer=_render_field_xrefs_text,
            stem="field-xrefs",
        )
    if not identifier:
        raise BridgeError("xrefs requires an identifier or --field")
    # xrefs adopts the canonical paging envelope (#164): JSON carries
    # {items, total, offset, limit, returned, has_more} (each item keeps its kind)
    # and --limit pages it. The op no longer ships the deprecated full
    # code_refs/data_refs arrays (#184), so paging now bounds the WHOLE payload.
    # Text mode groups the full set and uses `limit` only as a renderer-side
    # caller-group display cap, so it must fetch the full set -- don't forward
    # offset/limit to the op or the renderer would group only a slice.
    params: dict[str, Any] = {"identifier": identifier}
    limit = _effective_limit(args)
    if args.format != "text":
        if args.offset:
            params["offset"] = args.offset
        if limit is not None:
            params["limit"] = limit

    def _xrefs_pipe_truncation_note(result: Any) -> str | None:
        # Text mode fetches the full ref set and caps the *display* at `limit`
        # caller groups per section. That capped body is too small to spill, so
        # the spill pipe-note never fires -- a piped grep/wc/jq then undercounts by
        # orders of magnitude with no signal (#439). Surface an explicit note when
        # groups were actually hidden. `limit is None` (--out / explicit no-cap)
        # renders the full body, so nothing is hidden.
        if limit is None or not isinstance(result, dict):
            return None
        code_refs, data_refs, total_code, total_data = _xref_buckets(result)
        code_groups = len(_group_refs_by_caller(code_refs))
        data_groups = len(_group_refs_by_caller(data_refs))
        hidden = max(0, code_groups - limit) + max(0, data_groups - limit)
        if hidden <= 0:
            return None
        shown = min(code_groups, limit) + min(data_groups, limit)
        total_groups = code_groups + data_groups
        total_refs = total_code + total_data
        return (
            f"note: stdout is a pipe -- xrefs text body shows only {shown} of "
            f"{total_groups} caller groups ({total_refs} refs total); {hidden} "
            f"group(s) are truncated out of the stream, so a piped grep/wc/jq "
            f"undercounts. Re-run with --out FILE for the full body "
            f"(or --limit {total_groups} / --format json)."
        )

    return _call(
        args,
        "xrefs",
        params,
        require_target=True,
        text_renderer=lambda v: _render_xrefs_text(v, limit=limit),
        offset_hint_identifier=identifier,
        truncation_note=_xrefs_pipe_truncation_note,
        paged_spill=True,
        stem="xrefs",
    )


def _load_within_identifiers(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise BridgeError(
            "--within-file must be a UTF-8 text file with one function "
            f"identifier per line; got a binary file: {path}"
        ) from exc
    except OSError as exc:
        raise BridgeError(f"could not read --within-file {path}: {exc}") from exc
    identifiers = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        identifiers.append(line)
    return identifiers


@command("callsites", help="Find direct native callsites and exact caller_static addresses",
         target=True,
         prefer_when="exact caller->callsite address mapping; "
                     "use xrefs for general or data cross-references",
         see_also=("xrefs",),
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
    # Same canonical paging envelope and #184 payload-bounding as `xrefs`: JSON
    # pages the items (and the op drops the deprecated full arrays). Text fetches
    # the full set and uses `limit` as a per-bucket display cap, so don't forward
    # offset/limit to the op in text mode.
    # evidence xrefs is the deep/contextual reverse-link command: it also scans
    # data sections for stored function pointers so a callback-only function
    # (vtable/dispatch-table slot) isn't reported as dead (#323). Plain `xrefs`
    # stays fast and does not scan.
    params: dict[str, Any] = {"identifier": args.identifier, "fn_pointer_scan": True}
    limit = _effective_limit(args)
    if args.format != "text":
        if args.offset:
            params["offset"] = args.offset
        if limit is not None:
            params["limit"] = limit
    return _call(
        args,
        "xrefs",
        params,
        require_target=True,
        text_renderer=lambda value: _render_evidence_xrefs_text(value, limit=limit),
        paged_spill=True,
        stem="evidence-xrefs",
    )


@command("evidence", "table",
         help="Interpret memory at an address as a pointer table or vtable-like table",
         target=True,
         prefer_when="walk a raw vtable / pointer table as data; "
                     "use class show for a recovered C++ class's vtable and hierarchy",
         see_also=("class show",),
         args=[
             arg("address", help="Table start address (hex 0x.. or decimal)"),
             arg("--entries", type=_positive_int, default=16,
                 help="Number of pointer entries to read"),
             arg("--stride", default=None,
                 help="Byte stride between entries (default: target pointer size)"),
             arg("--width", default=None,
                 help="Bytes read per entry (default: min(stride, pointer size); "
                      "use 4 for a uint32[] table at --stride 4)"),
         ])
def _evidence_table(args: argparse.Namespace) -> int:
    return _call(
        args,
        "pointer_table",
        {
            "address": args.address,
            "entries": args.entries,
            "stride": args.stride,
            "width": args.width,
        },
        require_target=True,
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
        text_renderer=_render_message_lens_text,
        stem="message-lens",
    )


@command("evidence", "orient",
         fanout=True,
         help="One-shot orientation digest: target+analysis state, imports summary, a strings "
              "sample, function count, and sections — an internally-consistent triage card",
         target=True,
         prefer_when="first look at an unknown target — one consistent triage card instead of "
                     "running target info + imports + strings + function list + sections separately",
         see_also=("target info", "imports", "sections"),
         args=[
             arg("--strings-limit", type=_positive_int, default=20, dest="strings_limit",
                 help="Max strings in the bounded sample (default: 20)"),
         ])
def _evidence_orient(args: argparse.Namespace) -> int:
    return _call(
        args,
        "orient_digest",
        {"strings_limit": args.strings_limit},
        require_target=True,
        text_renderer=_render_orient_text,
        stem="orient",
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
        text_renderer=_render_init_arrays_text,
        stem="init-arrays",
    )


@command("trace",
         help="Backward slice: trace a call argument through SSA use-def chains to its origin",
         target=True,
         prefer_when="backward-slice a single call argument to its origin; "
                     "use taint backward for general sink-to-source slicing",
         see_also=("taint backward",),
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
                 help="Follow return values across call boundaries into callees. "
                      "Out-pointer/output-parameter writes are NOT followed; a value "
                      "loaded from a local an earlier call filled by-address is "
                      "reported as `interprocedural_out_param_not_followed` (naming "
                      "the callee), not traced into the callee (#416)"),
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
        text_renderer=_render_trace_text,
        stem="trace",
    )
