from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..cli import (_OUT_FORMAT_BY_SUFFIX, _call, _effective_limit, _int_or_hex, _mutate,
                   _mutation_exit_code, _mutation_preflight, _non_negative_int, _out_path_is_process_local, _pick,
                   _positive_int, _refuse_count_only_slices, arg, blank_selector, command,
                   decode_json_input, mutex, mutation_output_args, preview_arg,
                   read_text_input)
from ..formatters import (
    _discloses,
    _field_skewed,
    _nonnegative_count,
    _render_data_symbols_text,
    _render_data_vars_text,
    _render_function_bundle_text,
    _render_go_functions_text,
    _render_go_functions_summary_text,
    _render_go_rename_text,
    _go_rename_summary,
    _render_imports_summary_text,
    _render_name_address_list_text,
    _render_py_exec_text,
    _render_read_text,
    _render_sections_text,
    _render_strings_text,
    _stated_count,
)
from ..transport import BridgeError, unwrap_result
from ..wire_limits import MAX_OPS_ENV, batch_apply_max_ops


@_discloses
def _strings_count_text(value: Any) -> str:
    """The `strings --count` line, with the filter's denominator (#795).

    `Total strings: 30` from a `--probable-format-strings` run said nothing about
    the 1329 strings the filter dropped, so the denominator cost a SECOND
    unfiltered invocation. Mirrors `_imports_count_text`'s excluded-count tail:
    the bridge's own `filtered` count is disclosed parenthetically when it is
    non-zero, and the line is unchanged on an unfiltered dump.

    Both numbers are read through the COUNT CHOKE POINT, which is the same
    reading `_render_strings_text` gives them one surface over (#619 review).
    An `isinstance(int)` test here was a second decider over a question this
    codebase already answers, and it answered wrong in both directions: a
    producer that spells counts as text (`"1329"`) had its disclosure dropped
    entirely -- reinstating the extra invocation #795 removed -- while `bool` IS
    an `int`, so `filtered: true` rendered "(True filtered out by the active
    filters)". The headline goes through `_stated_count` for the same reason it
    does everywhere else: `Total strings: 0` fabricated from an unreadable
    counter reads byte-identically to an empty binary. The denominator goes
    through the CARDINALITY reader the listing surface uses, so a filter that
    claims to have dropped a negative number of strings is disclosed rather
    than restated as a quantity (#795 round-5 review)."""
    line = f"Total strings: {_stated_count(value, 'count')}"
    dropped = _nonnegative_count(value, "filtered")
    if dropped:
        line += f" ({dropped} filtered out by the active filters)"
    elif _field_skewed("filtered"):
        # The same wording the listing surface uses, so one payload cannot be
        # described two ways depending on which flag the caller passed.
        line += ("\n// the payload's filtered-string count is not a number that "
                 "can be read (use --format json)")
    return line


@command("strings", help="List or search strings", target=True, paged=True,
         fanout=True,
         args=[
             arg("--query"),
             arg("--regex", action="store_true", default=False,
                 help="Interpret --query as a case-insensitive regular expression"),
             arg("--min-length", type=_non_negative_int, default=None,
                 help="Exclude strings shorter than N characters"),
             arg("--max-length", type=_non_negative_int, default=None,
                 help="Exclude strings longer than N characters (drops long resource/blob data)"),
             arg("--section",
                 help="Only include strings in this section (e.g. .rodata, .rdata)"),
             arg("--no-crt", action="store_true", default=False,
                 help="Heuristic filter: exclude likely CRT/locale strings (platform-biased, best-effort)"),
             arg("--probable-format-strings", action="store_true", default=False,
                 help="Keep only strings that plausibly are C printf format strings "
                      "(validate the %% sequences against the printf directive grammar "
                      "instead of raw substring matching); annotate each with its "
                      "directives and code-xref count. Labels candidates, does NOT "
                      "assert a format-string vulnerability."),
             arg("--count", action="store_true", default=False,
                 help="Show the matching string count instead of listing"),
         ],
         estimable=True)
def _strings(args: argparse.Namespace) -> int:
    common = {
        "query": args.query,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "section": args.section,
        "no_crt": args.no_crt,
        "regex": bool(args.regex),
        "probable_format_strings": bool(args.probable_format_strings),
    }
    if args.count:
        _refuse_count_only_slices(args, command="strings")
        return _call(
            args,
            "strings",
            {**common, "count_only": True},
            require_target=True,
            text_renderer=_strings_count_text,
            stem="strings-count",
            regex_hint_query=args.query,
        )
    # Bridge-authoritative paging (#122): forward the real limit/offset so the
    # bridge returns the page WITH the true total in a {items, total, ...}
    # envelope, matching function list/search. paged_spill keeps the
    # "--limit/--offset to page" spill hint without the client-side limit+1 probe.
    rc = _call(
        args,
        "strings",
        {
            **common,
            "offset": args.offset,
            "limit": _effective_limit(args),
        },
        require_target=True,
        text_renderer=_render_strings_text,
        page_label="strings",
        paged_spill=True,
        stem="strings",
        regex_hint_query=args.query,
    )
    # The "narrow your noisy dump" tip only makes sense after a successful,
    # unfiltered dump. Emitting it BEFORE the request put it ahead of (and buried)
    # a --quick refusal / error; print it after a clean result instead.
    if (rc == 0 and args.section is None and args.query is None and args.min_length is None
            and args.max_length is None and not args.no_crt and not args.probable_format_strings):
        print(
            "tip: an unfiltered string dump includes .dynsym/.hash/.symtab noise; "
            "narrow with --section .rodata (or --query / --min-length) for signal.",
            file=sys.stderr,
        )
    return rc


@_discloses
def _imports_count_text(value: Any) -> str:
    """The `imports --count` line, with the filter's excluded-count tail.

    The sibling `_strings_count_text` is modelled on this line, and #795's
    round-2 review found the model was the defective one: `isinstance(int)` is a
    SECOND decider over a question the count choke point already answers, and it
    gets all three of the same shapes wrong. `bool` IS an `int`, so
    `self_defined_excluded: true` rendered "(True self-defined excluded)" -- a
    flag printed as a quantity; a producer that spells counts as text dropped
    the tail entirely; and the headline was interpolated raw, so a container
    landed in the line as a Python repr. Both numbers go through the choke
    point, under the boundary that discloses what it could not read (#619) --
    and the excluded count through the ONE reader the paged listing and the
    `--summary` card share, so the three surfaces cannot decide it three ways
    (#795 round-4 review, where this line alone stated a negative count).
    """
    line = f"Total imports: {_stated_count(value, 'count')}"
    excluded = _nonnegative_count(value, "self_defined_excluded")
    if excluded:
        line += f" ({excluded} self-defined excluded)"
    elif _field_skewed("self_defined_excluded"):
        line += ("\n// the payload's self-defined-excluded count is not a number "
                 "that can be read (use --format json)")
    return line


def _plain_count_text(label: str, value: Any) -> str:
    """A `--count` line that states ONE number and nothing else.

    The three surfaces below were inline `lambda value: f"{label}:
    {value.get('count', 0)}"` renderers -- the raw read this module's two other
    count lines were just taken off, and invisible to any guard because a
    lambda has no name to probe. One named renderer each, all reading through
    the choke point, so every `--count` line in the module answers the same way
    and a new one cannot be written as a lambda without tripping the guard in
    `tests/test_cli_misc.py` (#619/#795)."""
    return f"{label}: {_stated_count(value, 'count')}"


@_discloses
def _exports_count_text(value: Any) -> str:
    return _plain_count_text("Total exports", value)


@_discloses
def _sections_count_text(value: Any) -> str:
    return _plain_count_text("Total sections", value)


@_discloses
def _go_functions_count_text(value: Any) -> str:
    return _plain_count_text("Go functions", value)


@command("imports", help="List imports", target=True, paged=True,
         fanout=True,
         args=[arg("--summary", action="store_true", default=False,
                   help="Show aggregate counts by namespace and kind instead of the full list"),
               arg("--count", action="store_true", default=False,
                   help="Show the total import count instead of listing"),
               arg("--query", dest="query", default=None,
                   help="Filter to imports matching this substring (or regex with --regex) "
                        "against name/raw_name/library, e.g. a sink sweep"),
               arg("--regex", action="store_true", default=False,
                   help="Treat --query as a case-insensitive regex (alternation for sink families)"),
               arg("--include-got", action="store_true", default=False,
                   help="Include GOT-slot (address) entries that duplicate a PLT import "
                        "(collapsed by default)")],
         estimable=True)
def _imports(args: argparse.Namespace) -> int:
    query = getattr(args, "query", None)
    regex = bool(getattr(args, "regex", False))
    if args.count:
        # #767: --summary and the paging flags vanished silently here while
        # `go functions` refused the same combination at the parser; refuse by
        # name instead of returning a number the flags did not shape.
        _refuse_count_only_slices(args, command="imports")
        return _call(
            args,
            "imports",
            {"count_only": True, "include_got": bool(args.include_got),
             "query": query, "regex": regex},
            require_target=True,
            text_renderer=_imports_count_text,
            stem="imports-count",
        )
    summary_mode = bool(args.summary)
    if summary_mode:
        # #872: the summary is ONE aggregate object, so --limit/--offset were
        # forwarded to a bridge that cannot apply them and vanished there --
        # the same accepts-and-discards shape #767/#768 refused for --count.
        _refuse_count_only_slices(args, command="imports", mode="--summary")
    # Summary is a single aggregate object, so it ignores paging entirely. The
    # full list (often 500+ entries on firmware libs) pages bridge-side like
    # strings/function list, returning a {items, total, ...} envelope (#122).
    params = {"summary": summary_mode, "include_got": bool(args.include_got),
              "query": query, "regex": regex}
    if not summary_mode:
        params["offset"] = args.offset
        params["limit"] = _effective_limit(args)
    return _call(
        args,
        "imports",
        params,
        require_target=True,
        text_renderer=_render_imports_summary_text if summary_mode else _render_name_address_list_text,
        page_label="imports",
        # Only the list path pages; the summary aggregate has no remainder to
        # hint about, so it does not opt into the paging spill hint.
        paged_spill=not summary_mode,
        stem="imports-summary" if summary_mode else "imports",
    )


_EXPORT_ARGS = [
    arg(
        "--count",
        action="store_true",
        default=False,
        help="Show the total export count instead of listing",
    )
]


@command(
    "exports",
    help="List exported symbols (a binary's public API)",
    target=True,
    paged=True,
    fanout=True,
    args=_EXPORT_ARGS,
         estimable=True
)
@command(
    "exports",
    "list",
    help="List exported symbols (alias for `exports`)",
    target=True,
    paged=True,
    fanout=True,
    args=_EXPORT_ARGS,
         estimable=True
)
def _exports(args: argparse.Namespace) -> int:
    if args.count:
        _refuse_count_only_slices(args, command="exports")
        return _call(
            args,
            "list_exports",
            {"count_only": True},
            require_target=True,
            text_renderer=_exports_count_text,
            stem="exports-count",
        )
    params = {"offset": args.offset, "limit": _effective_limit(args)}
    return _call(
        args,
        "list_exports",
        params,
        require_target=True,
        text_renderer=_render_name_address_list_text,
        page_label="exports",
        paged_spill=True,
        stem="exports",
    )


@command("sections", help="List binary sections with address ranges and permissions", target=True,
         fanout=True,
         paged=True, args=[arg("--query",
                               help="Filter by a substring of the section name OR its semantics "
                                    "label (e.g. 'code' matches .text=ReadOnlyCode); broadens to "
                                    "all matching-semantics sections, not just name matches"),
                           arg("--count", action="store_true", default=False,
                               help="Show the section count instead of listing")],
         estimable=True)
def _sections(args: argparse.Namespace) -> int:
    if args.count:
        _refuse_count_only_slices(args, command="sections")
        return _call(
            args,
            "sections",
            {"query": args.query, "count_only": True},
            require_target=True,
            text_renderer=_sections_count_text,
            stem="sections-count",
        )
    # Bridge-authoritative paging (#122): forward the real limit/offset so the
    # bridge returns the {items, total, ...} envelope with the true total.
    return _call(
        args,
        "sections",
        {"query": args.query, "offset": args.offset, "limit": _effective_limit(args)},
        require_target=True,
        text_renderer=_render_sections_text,
        page_label="sections",
        paged_spill=True,
        stem="sections",
    )


@command("data", "vars",
         help="Typed data variables in the half-open address window [start, end)",
         target=True,
         prefer_when="you want BN's typed view of a data region (widths, decoded values, "
                     "pointer targets), not raw bytes; `read` gives the raw bytes",
         see_also=("read", "sections", "data symbols"),
         args=[
             arg("--start", required=True,
                 help="Window start address, inclusive (hex 0x.. or decimal)"),
             arg("--end", required=True,
                 help="Window end address, exclusive (hex 0x.. or decimal)"),
             arg("--limit", type=_positive_int, default=None, metavar="N",
                 help="Maximum rows to return (default 400); when truncated the result "
                      "sets has_more and text mode prints a --start resume hint"),
         ],
         estimable=True)
def _data_vars(args: argparse.Namespace) -> int:
    return _call(
        args,
        "data_vars",
        {"start": args.start, "end": args.end, "limit": args.limit},
        require_target=True,
        text_renderer=_render_data_vars_text,
        stem="data-vars",
    )


@command("data", "symbols",
         help="List named data symbols (address + name), including internal ones "
              "the exports list omits",
         target=True,
         paged=True,
         prefer_when="you need addressable data globals (including renamed/internal ones); "
                     "`exports` only shows the public surface",
         see_also=("exports", "data vars"),
         estimable=True)
def _data_symbols(args: argparse.Namespace) -> int:
    # #682 item 1: paged like every sibling list read. This command used to
    # hand-roll --limit/--offset with `default=None`, i.e. build and serialize
    # EVERY data symbol unless told otherwise; on a large view that was the
    # read lock held for the whole build, the inverse of what this read-locked
    # family exists for. The page is the default now, and a caller who
    # genuinely wants the whole set asks for it with a large --limit -- or with
    # --out, which `_effective_limit` deliberately uncaps (#165) so a
    # full-body export is not silently capped at the page.
    #
    # That uncap must be spelled OUT on the wire: the bridge op defaults an
    # omitted `limit` to the same 100-row page (so the direct programmatic
    # caller the issue names is bounded too), and `"all"` is its explicit
    # whole-set request. Omitting the key would now cap the `--out` export.
    # The token is deliberately not `0`: a caller-visible zero already means
    # "the schema, not the rows" everywhere else in this repo.
    limit = _effective_limit(args)
    return _call(
        args,
        "data_symbols",
        {"offset": args.offset, "limit": "all" if limit is None else limit},
        require_target=True,
        text_renderer=_render_data_symbols_text,
        page_label="data symbols",
        paged_spill=True,
        stem="data-symbols",
    )


@command("go", "functions",
         fanout=True,
         help="Recover Go function names from .gopclntab (Go 1.18/1.20+) — names the wall of sub_*",
         target=True, paged=True,
         prefer_when="a Go-compiled binary loads as a wall of sub_* (.gopclntab present); recover "
                     "the real pkg.Func names BN's default analysis doesn't consume",
         see_also=("function list", "sections"),
         mutex_groups=[
             mutex(False,
                   arg("--count", action="store_true", default=False,
                       help="Show the recovered Go function count instead of listing"),
                   arg("--summary", action="store_true", default=False,
                       help="Show recovered/defined/renamable counts + pclntab status (decide whether to `go rename`)")),
         ],
         estimable=True)
def _go_functions(args: argparse.Namespace) -> int:
    if args.count:
        _refuse_count_only_slices(args, command="go functions")
        return _call(
            args, "go_functions", {"count_only": True},
            require_target=True,
            text_renderer=_go_functions_count_text,
            stem="go-functions-count",
        )
    if args.summary:
        # #899 review: the sibling one MODE over, missed by a table keyed on
        # COMMANDS -- `go functions` was already listed for `--count`, so a
        # second aggregate on the same command was structurally invisible to
        # the coverage. Same shape as `imports --summary`: one object, so the
        # paging flags cannot apply.
        _refuse_count_only_slices(args, command="go functions", mode="--summary")
        return _call(
            args, "go_functions", {"summary": True},
            require_target=True,
            text_renderer=_render_go_functions_summary_text,
            stem="go-functions-summary",
        )
    return _call(
        args,
        "go_functions",
        {"offset": args.offset, "limit": _effective_limit(args)},
        require_target=True,
        text_renderer=_render_go_functions_text,
        page_label="go_functions",
        paged_spill=True,
        stem="go-functions",
    )


@command("go", "rename",
         help="Apply recovered Go names from .gopclntab to the database "
              "(renames auto-named sub_*/nullsub_* only — never your manual names)",
         target=True, fmt="json",
         prefer_when="after `go functions` shows the recovered names, apply them so the "
                     "Go binary is navigable; safe to re-run (idempotent, skips named functions)",
         see_also=("go functions",),
         args=[preview_arg("Apply the renames, capture diffs, then revert without committing"),
               *mutation_output_args()])
def _go_rename(args: argparse.Namespace) -> int:
    # #408 review: go rename is a bulk mutation (success/committed/results +
    # _mutation_exit_code), so it honors --summary like the other mutations.
    #
    # It routes through _mutate rather than hand-rolling the tail so #645's
    # compact-by-default applies here too. Hand-rolling meant --verbose/--diffs
    # parsed but did nothing, and go rename is the mutation MOST likely to emit a
    # huge payload -- it renames every candidate in the binary. Going through
    # _mutate also lands the #447 top-level `ok` on the full JSON (#604), which
    # the hand-rolled `result_transform=None` path omitted.
    return _mutate(
        args,
        "go_rename",
        {},
        preview=bool(args.preview),
        require_target=True,
        detail_renderer=_render_go_rename_text,
        summary_transform=_go_rename_summary,
        stem="go-rename",
    )


def _resolved_out_format(args: argparse.Namespace) -> str:
    """``cli._resolve_output_format``'s decision without its stderr side effects.

    The real resolver PRINTS the inference note (or the disagreement warning),
    and ``_call`` invokes it again a few lines below, so calling it here would
    emit the note twice. Its precedence -- an explicit ``--format`` wins,
    otherwise a recognised ``--out`` suffix is inferred -- is mirrored instead,
    and the two are pinned together over every ``--format`` x ``--out`` suffix
    combination by
    ``tests/test_cli_misc.py::test_bundle_delegation_format_tracks_the_cli_resolver``
    so the copies cannot drift (#670).
    """
    out = getattr(args, "out", None)
    fmt = getattr(args, "format", "text")
    if out is None:
        return fmt
    inferred = _OUT_FORMAT_BY_SUFFIX.get(Path(str(out)).suffix.lower())
    if inferred is None or getattr(args, "_format_explicit", False):
        return fmt
    return inferred


@command("bundle", "function", help="Export a function bundle", fmt="json", target=True,
         args=[arg("identifier"),
               arg("--include-annotations", action="store_true", default=False,
                   help="Include inherited comment bodies in the bundle's "
                        "decompilation (default: redact, matching bn decompile)")],
         estimable=True)
def _bundle_function(args: argparse.Namespace) -> int:
    # #665: `--out` is already absolute here (`_resolve_out_path`), so the
    # bridge writes it where the CALLER meant. The one destination the bridge
    # must NOT be handed is a process-local fd path -- `--out /proc/self/fd/<n>`
    # (the bn-kernel CLI backend's artifact contract) or `--out /dev/stdout`.
    # Those resolve in whichever process opens them, so the bridge would write
    # into its own fd <n> and the caller would read zero bytes behind an
    # ok/bytes/sha256 envelope. Write those in THIS process instead: with
    # `out_path=None` the bridge returns the bundle itself.
    #
    # #670: the bridge-side writer sees only the `--out` SUFFIX, so it cannot
    # honor an explicit --format that disagrees with it -- it would write NDJSON
    # for `--format json --out x.ndjson` while this CLI prints "writing json".
    # Delegate only when the format resolved here is the one the bridge would
    # actually emit for that path.
    #
    # `bridge_format` asks the question the BRIDGE asks (`_write_json_artifact`
    # keys off the literal `.ndjson` suffix), not the one the CLI's suffix MAP
    # answers. Reading it off the map instead would claim ndjson for any future
    # ndjson-mapped suffix the bridge still writes as JSON, which is exactly the
    # note/bytes contradiction #670 reported.
    out_suffix = Path(str(args.out)).suffix.lower() if args.out else ""
    bridge_format = "ndjson" if out_suffix == ".ndjson" else "json"
    bridge_writes = (
        bool(args.out)
        and not _out_path_is_process_local(args.out)
        and _resolved_out_format(args) == bridge_format
    )
    return _call(
        args,
        "bundle_function",
        {"identifier": args.identifier,
         "out_path": str(args.out) if bridge_writes else None,
         "include_annotations": bool(args.include_annotations)},
        require_target=True,
        text_renderer=_render_function_bundle_text,
        stem="function-bundle",
        bridge_writes_output=bridge_writes,
    )


@command("read", help="Read raw bytes at an address", target=True,
         args=[
             arg("address", nargs="?",
                 help="Address to read from (hex 0x.. or decimal)"),
             arg("--address", dest="address_flag", default=None,
                 help="Address to read from (alias for the positional)"),
             arg("--length", "--size", dest="length", default=16, type=_int_or_hex,
                 help="Number of bytes to read (decimal or hex 0x..; --size is an alias; default 16)"),
             arg("--encoding", choices=("hex", "bytes"), default="hex",
                 help="Byte payload encoding: hex hexdump (default) or raw bytes"),
         ],
         estimable=True)
def _read(args: argparse.Namespace) -> int:
    address = _pick(args.address, args.address_flag, "read address")
    if args.encoding == "bytes":
        return _read_raw_bytes(args, address)
    return _call(
        args,
        "read",
        {"address": address, "length": args.length},
        require_target=True,
        text_renderer=_render_read_text,
        stem="read",
    )


def _read_raw_bytes(args: argparse.Namespace, address: str) -> int:
    from .. import cli

    target = cli._resolve_target(args, require_target=True, allow_implicit_target=True)
    response = cli.send_request(
        "read",
        params={"address": address, "length": args.length},
        target=target,
        instance_id=getattr(args, "instance", None),
    )
    result = unwrap_result(response, "read")
    hex_payload = result.get("hex") if isinstance(result, dict) else None
    if not isinstance(hex_payload, str):
        raise BridgeError("bridge returned malformed read response (missing 'hex' payload)")
    try:
        data = bytes.fromhex(hex_payload)
    except ValueError:
        raise BridgeError("bridge returned malformed read response (invalid hex payload)") from None
    # #827 item 4 review: the bridge marks a PARTIAL read in the payload
    # (`capped` and/or `short_read`, plus `requested_length` and a `note`). The
    # hex renderer prints that note, but THIS path returns the bytes themselves
    # -- and a dump quietly shorter than the window the caller asked for is the
    # "bounded read that reads as the whole window" failure the note exists to
    # prevent, on the one path documented for piping a blob into another tool.
    # Both markers, not just the cap: a short read is the same dump with the
    # same consequence, and disclosing one of the two taught a reader that
    # silence here means a complete window. Disclose on stderr (stdout IS the
    # payload) and carry the markers into the --out summary.
    partial_fields: dict[str, Any] = {}
    # `result` is known to be a dict here: a non-dict one cannot carry the
    # string `hex` the refusal above requires.
    if result.get("capped") or result.get("short_read"):
        partial_fields = {key: True for key in ("capped", "short_read") if result.get(key)}
        if result.get("requested_length") is not None:
            # Absent rather than `null`: the summary states what the caller
            # asked for, and a bridge that did not send it has nothing to state.
            partial_fields["requested_length"] = result["requested_length"]
        note = str(result.get("note") or f"partial read: {len(data)} bytes returned")
        print(f"note: {note}", file=sys.stderr)
    summary = {"kind": "bytes", "address": address, "length": len(data), **partial_fields}
    if getattr(args, "estimate_output", False):
        # #796: this is `read`'s SECOND emit path, and the flag has to mean the
        # same thing on it. `_call` -> `_render_result` implements the preflight
        # for the `--encoding hex` half; this branch returns above that call, so
        # asking here is the only place the raw-byte payload can be measured
        # instead of written. `--out` is refused beside the flag by the parser's
        # own mutually exclusive group, so there is no destination to resolve.
        from ..output import estimate_bytes_result

        estimate = estimate_bytes_result(
            data,
            fmt=args.format,
            summary=summary,
            rerun_hint=cli._slice_hint_for_args(args, args.format),
        )
        sys.stdout.write(estimate.rendered)
        return 0
    if args.out:
        from ..output import write_bytes_result

        result = write_bytes_result(
            data,
            out_path=args.out,
            fmt=args.format,
            summary=summary,
        )
        sys.stdout.write(result.rendered)
    else:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    return 0


@command("py", "exec", help="Execute a Python snippet", target=True,
         args=[arg("code_pos", nargs="?", metavar="CODE",
                   help="Inline Python code (positional; same as --code)")],
         mutex_groups=[
             mutex(False,
                   arg("--script", type=Path, help="Read Python code from a file"),
                   arg("--code", help="Inline Python code"),
                   arg("--stdin", action="store_true")),
         ])
def _py_exec(args: argparse.Namespace) -> int:
    # Accept a bare positional as the code (the natural `bn py exec '<code>'`
    # form the skill examples imply), in addition to --code/--script/--stdin (#197).
    pos = getattr(args, "code_pos", None)
    flag = getattr(args, "code", None)
    if pos is not None and flag is not None:
        raise BridgeError("py exec: pass code positionally OR with --code, not both")
    inline = flag if flag is not None else pos
    if inline == "-":
        # Standardize the stdin idiom with `batch apply -`: a positional/`--code`
        # value of "-" reads the script from stdin, alongside the explicit
        # --stdin flag (#312).
        script = sys.stdin.read()
    elif inline is not None:
        script = inline
    elif args.script:
        # #864: one reader for every CLI text input -- it refuses a directory or
        # a FIFO/device by kind (a FIFO here blocked forever with no envelope)
        # and wraps the read, keeping #754's structured-refusal envelope.
        script = read_text_input(
            args.script, what="Script file", hint="Use --code for inline Python.")
    elif args.stdin:
        script = sys.stdin.read()
    else:
        raise BridgeError(
            "py exec needs code: pass it positionally (bn py exec '<code>'), "
            "or use --code / --script FILE / --stdin"
        )

    return _call(
        args,
        "py_exec",
        {"script": script},
        require_target=True,
        text_renderer=_render_py_exec_text,
        stem="py-exec",
    )


def _batch_target_from_cli(args: argparse.Namespace,
                           manifest: dict[str, Any]) -> str | None:
    """The CLI selector this invocation may apply to *manifest*, if any.

    An EXPLICIT ``-t`` is the per-invocation selector and WINS over a manifest
    ``"target"`` (#366). An AMBIENT one -- the sticky pin or an exported
    ``BN_TARGET``, both filled and marked by ``cli._apply_sticky_defaults`` --
    is not that: nobody named it on this command line, and ``batch_apply`` is
    a DESTRUCTIVE op. It may only FILL a manifest that named no target of its
    own; a manifest that DID name one keeps that choice, including when #227
    drops an instance-id placeholder for single-open resolution (#676 item 11).

    A BLANK value is not a selector at all and can fill nothing. This asked
    that with plain truthiness, which is true of ``"   "``: the resolver read
    a whitespace ambient value as the absence of a selector and kept it out
    of the request envelope, while this fill read the same value as a
    selector and wrote it into the manifest -- so it reached the bridge in
    the PAYLOAD of a destructive op, the one place the reference promises a
    blank value never goes. One shared predicate handles blankness; the
    caller must also decide once before it changes the manifest.
    """
    cli_target = getattr(args, "target", None)
    if cli_target is None or blank_selector(cli_target):
        return None
    if getattr(args, "_sticky_target", False) and manifest.get("target"):
        return None
    return cli_target


@command("batch", "apply", help="Apply a JSON manifest", fmt="json", target=True,
         args=[
             preview_arg("Apply the whole batch, capture diffs, then revert without committing"),
             *mutation_output_args(),
             arg("manifest", type=Path,
                 help=(
                     "JSON manifest source: a file path, or \"-\" to read from stdin. "
                     "A quoted heredoc on stdin is the recommended form -- the quoted "
                     "delimiter makes the whole payload literal, so comments with quotes, "
                     "apostrophes, $, or parens need no escaping:\n"
                     "  bn batch apply - <<'BN_EOF'\n"
                     "  {\"ops\": [{\"op\": \"set_comment\", \"address\": \"0x1000\", "
                     "\"comment\": \"len isn't checked\"}]}\n"
                     "  BN_EOF\n"
                     "Manifest shape: {\"target\": <selector>, \"ops\": [<op>, ...]}. "
                     "Each op is an object with an \"op\" kind plus its fields, e.g. "
                     "{\"op\": \"rename_symbol\", \"identifier\": \"sub_1000\", \"new_name\": \"parse\"} "
                     "or {\"op\": \"set_comment\", \"address\": \"0x1000\", \"comment\": \"...\"}. "
                     "Kinds: rename_symbol, set_comment, delete_comment, set_prototype, "
                     "local_rename, local_retype, struct_field_set, struct_field_rename, "
                     "struct_field_delete, types_declare. A missing required field is reported "
                     "as status 'invalid_request' naming the field.\n"
                     "Ceilings: a manifest over 5000 ops, or whose serialized request exceeds "
                     "the bridge's hard 32 MiB wire limit, is refused before anything is "
                     "sent. A large batch holds the write lock for the whole run and reverts "
                     "as ONE unit. BN_BATCH_APPLY_MAX_OPS=<n> raises the op limit (0 disables); "
                     "BN_BATCH_APPLY_MAX_BYTES=<n> may set a LOWER byte limit (0 restores the "
                     "hard limit). File and FIFO input also has a 64 MiB source-file cap."
                 )),
         ])
def _batch_apply(args: argparse.Namespace) -> int:
    # "-" reads the manifest from stdin (standard CLI convention), enabling the
    # quoted-heredoc form that needs no escaping for free-text comments (#104). A
    # literal file named "-" can still be passed as "./-".
    from_stdin = str(args.manifest) == "-"
    if from_stdin:
        source = "<stdin>"
        try:
            raw = sys.stdin.read()
        except OSError as exc:
            raise BridgeError(f"Could not read manifest from stdin: {exc}") from None
        if not raw.strip():
            raise BridgeError(
                'No manifest on stdin. Pipe a JSON object {"target": <selector>, '
                '"ops": [<op>, ...]}, e.g. via a quoted heredoc: '
                "bn batch apply - <<'BN_EOF' ... BN_EOF"
            )
    else:
        source = f"file {args.manifest}"
        # #864: same shared reader as the other --file shapes; a FIFO manifest
        # blocked here forever with no envelope.
        raw = read_text_input(args.manifest, what="Manifest file")
    # #864: and the same shared decoder, so a body the parser cannot take --
    # malformed, nested past its stack, or too large to build -- is refused
    # here rather than escaping as a traceback the way a `RecursionError` did.
    manifest = decode_json_input(raw, refusal=f"Invalid JSON in manifest ({source})")
    # The manifest must be a JSON object {"target": <sel>, "ops": [...]}. A bare
    # array (an easy mistake) would otherwise crash client-side in _call's
    # dict(params) -- and `manifest["preview"]` below assumes a dict. Validate
    # shape here and raise a clean BridgeError (#48).
    if not isinstance(manifest, dict):
        raise BridgeError(
            f"Manifest ({source}) must be a JSON object "
            f'{{"target": <selector>, "ops": [<op>, ...]}}, got a '
            f"{type(manifest).__name__}. (A bare list of ops should be wrapped as "
            f'{{"ops": [...]}}.)'
        )
    # #227: a fan-out agent can put its -i/--instance id in the manifest
    # "target". That id names the bridge, not a binary, so drop the placeholder
    # and let the bridge resolve its single open target. Decide which CLI
    # selector may apply BEFORE dropping it: reasking afterwards would let an
    # ambient BN_TARGET/pin fill the new vacancy and redirect the whole batch.
    manifest_named_target = bool(manifest.get("target"))
    cli_target = _batch_target_from_cli(args, manifest)
    inst = getattr(args, "instance", None)
    if inst and manifest.get("target") == inst:
        manifest.pop("target", None)
    with _mutation_preflight(args):
        if not isinstance(manifest.get("ops"), list):
            raise BridgeError(
                f'Manifest ({source}) must have an "ops" array (the list of '
                f"operations to apply)."
            )
        # #769: the op ceiling is checked here before the request. The byte
        # ceiling is checked on the actual serialized envelope in transport,
        # after instance/target resolution but before a socket send. A local
        # estimate cannot account exactly for the selected bridge identity.
        max_ops = batch_apply_max_ops()
        op_count = len(manifest["ops"])
        if max_ops is not None and op_count > max_ops:
            raise BridgeError(
                f"Manifest ({source}) has {op_count} operations, over the "
                f"{max_ops} limit. A batch this size holds the write lock for "
                f"the whole run and reverts as ONE unit, so a single failure "
                f"discards every sibling. Split it, or raise/disable the "
                f"ceiling with {MAX_OPS_ENV}=<n> (0 disables)."
            )
    # #690 r4: an explicit-but-empty manifest target (an unset shell variable
    # templated into the file) is an error -- it must not ride the focused-tab
    # convenience bridge-side, and a sticky pin must not silently paper over it.
    manifest_target = manifest.get("target")
    if blank_selector(manifest_target):
        raise BridgeError(
            f'Manifest ({source}) target is empty: set a selector from '
            '`bn target list`, or drop the "target" key to use the single '
            "open target"
        )
    # Accept -t/--target like every other target-required mutate command (#308).
    # An EXPLICIT CLI -t WINS over a manifest "target" (#366): it is the explicit
    # per-invocation selector, so a fan-out agent that copies the documented
    # {"target":"active"} example but passes a correct -t isn't sabotaged by the
    # in-payload value ("active" doesn't resolve under multi-target headless).
    # An AMBIENT one -- the sticky pin or an exported BN_TARGET, both filled and
    # marked by `_apply_sticky_defaults` -- is NOT that: nobody named it on this
    # command line, and `batch_apply` is a DESTRUCTIVE op. Letting it through
    # dispatched the whole manifest at the ambient selector and discarded the
    # target the file itself named, which is the same hazard a bare destructive
    # `close` refuses (#676 item 11). An ambient value may only FILL a manifest
    # that named none; without any CLI target the manifest "target" is honored
    # as before.
    if cli_target:
        manifest["target"] = cli_target
    elif getattr(args, "_sticky_target", False) and manifest_named_target:
        # The ambient value was demoted. Drop it from the ENVELOPE too, so the
        # request names ONE selector: the bridge resolves `batch_apply` from
        # the manifest's own target, or from its sole open target if #227
        # removed an instance-id placeholder. A second, different selector
        # riding beside it is a claim this invocation no longer makes.
        #
        # A BROKEN ambient default (an empty export or pin) lands here too,
        # and needs nothing extra: it is not a selector, so `_resolve_target`
        # treats it as the absence of one, and `batch_apply` requires no
        # target of its own -- the manifest named it. Clearing the marker
        # here as well used to be what kept the empty-selector refusal off
        # this command; it no longer is, because the refusal is now asked at
        # the resolution, and this invocation performs none (#676 item 11).
        args.target = None
    if args.preview:
        manifest["preview"] = True
    # preview is already set on the manifest above, so it is not passed through
    # _mutate's preview= injection here.
    return _mutate(
        args,
        "batch_apply",
        manifest,
        require_target=False,
        stem="batch-apply",
    )
