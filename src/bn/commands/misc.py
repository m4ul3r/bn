from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..cli import _call, _effective_limit, _int_or_hex, _mutation_exit_code, _non_negative_int, _pick, arg, command, mutex
from ..formatters import (
    _render_imports_summary_text,
    _render_mutation_text,
    _render_name_address_list_text,
    _render_py_exec_text,
    _render_read_text,
    _render_sections_text,
    _render_strings_text,
)
from ..transport import BridgeError


@command("strings", help="List or search strings", target=True, paged=True,
         args=[
             arg("--query"),
             arg("--regex", action="store_true", default=False,
                 help="Interpret --query as a case-insensitive regular expression"),
             arg("--min-length", type=_non_negative_int, default=None,
                 help="Exclude strings shorter than N characters"),
             arg("--section",
                 help="Only include strings in this section (e.g. .rodata, .rdata)"),
             arg("--no-crt", action="store_true", default=False,
                 help="Heuristic filter: exclude likely CRT/locale strings (platform-biased, best-effort)"),
             arg("--count", action="store_true", default=False,
                 help="Show the matching string count instead of listing"),
         ])
def _strings(args: argparse.Namespace) -> int:
    common = {
        "query": args.query,
        "min_length": args.min_length,
        "section": args.section,
        "no_crt": args.no_crt,
        "regex": bool(args.regex),
    }
    if args.count:
        return _call(
            args,
            "strings",
            {**common, "count_only": True},
            require_target=True,
            allow_implicit_target=True,
            text_renderer=lambda value: f"Total strings: {value.get('count', 0)}",
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
        allow_implicit_target=True,
        text_renderer=_render_strings_text,
        page_label="strings",
        paged_spill=True,
        stem="strings",
        regex_hint_query=args.query,
    )
    # The "narrow your noisy dump" tip only makes sense after a successful,
    # unfiltered dump. Emitting it BEFORE the request put it ahead of (and buried)
    # a --quick refusal / error; print it after a clean result instead.
    if rc == 0 and args.section is None and args.query is None and args.min_length is None and not args.no_crt:
        print(
            "tip: an unfiltered string dump includes .dynsym/.hash/.symtab noise; "
            "narrow with --section .rodata (or --query / --min-length) for signal.",
            file=sys.stderr,
        )
    return rc


def _imports_count_text(value: Any) -> str:
    line = f"Total imports: {value.get('count', 0)}"
    excluded = value.get("self_defined_excluded")
    if isinstance(excluded, int) and excluded > 0:
        line += f" ({excluded} self-defined excluded)"
    return line


@command("imports", help="List imports", target=True, paged=True,
         args=[arg("--summary", action="store_true", default=False,
                   help="Show aggregate counts by namespace and kind instead of the full list"),
               arg("--count", action="store_true", default=False,
                   help="Show the total import count instead of listing")])
def _imports(args: argparse.Namespace) -> int:
    if args.count:
        return _call(
            args,
            "imports",
            {"count_only": True},
            require_target=True,
            allow_implicit_target=True,
            text_renderer=_imports_count_text,
            stem="imports-count",
        )
    summary_mode = bool(args.summary)
    # Summary is a single aggregate object, so it ignores paging entirely. The
    # full list (often 500+ entries on firmware libs) pages bridge-side like
    # strings/function list, returning a {items, total, ...} envelope (#122).
    params = {"summary": summary_mode, "offset": args.offset}
    if not summary_mode:
        params["limit"] = _effective_limit(args)
    return _call(
        args,
        "imports",
        params,
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_imports_summary_text if summary_mode else _render_name_address_list_text,
        page_label="imports",
        # Only the list path pages; the summary aggregate has no remainder to
        # hint about, so it does not opt into the paging spill hint.
        paged_spill=not summary_mode,
        stem="imports-summary" if summary_mode else "imports",
    )


@command("sections", help="List binary sections with address ranges and permissions", target=True,
         paged=True, args=[arg("--query", help="Filter sections by name substring"),
                           arg("--count", action="store_true", default=False,
                               help="Show the section count instead of listing")])
def _sections(args: argparse.Namespace) -> int:
    if args.count:
        return _call(
            args,
            "sections",
            {"query": args.query, "count_only": True},
            require_target=True,
            allow_implicit_target=True,
            text_renderer=lambda value: f"Total sections: {value.get('count', 0)}",
            stem="sections-count",
        )
    # Bridge-authoritative paging (#122): forward the real limit/offset so the
    # bridge returns the {items, total, ...} envelope with the true total.
    return _call(
        args,
        "sections",
        {"query": args.query, "offset": args.offset, "limit": _effective_limit(args)},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_sections_text,
        page_label="sections",
        paged_spill=True,
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


@command("read", help="Read raw bytes at an address", target=True,
         args=[
             arg("address", nargs="?",
                 help="Address to read from (hex 0x.. or decimal)"),
             arg("--address", dest="address_flag", default=None,
                 help="Address to read from (alias for the positional)"),
             arg("--length", required=True, type=_int_or_hex,
                 help="Number of bytes to read (decimal or hex 0x..)"),
             arg("--encoding", choices=("hex", "bytes"), default="hex",
                 help="Byte payload encoding: hex hexdump (default) or raw bytes"),
         ])
def _read(args: argparse.Namespace) -> int:
    address = _pick(args.address, args.address_flag, "read address")
    if args.encoding == "bytes":
        return _read_raw_bytes(args, address)
    return _call(
        args,
        "read",
        {"address": address, "length": args.length},
        require_target=True,
        allow_implicit_target=True,
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
    result = response["result"]
    hex_payload = result.get("hex") if isinstance(result, dict) else None
    if not isinstance(hex_payload, str):
        raise BridgeError("bridge returned malformed read response (missing 'hex' payload)")
    try:
        data = bytes.fromhex(hex_payload)
    except ValueError:
        raise BridgeError("bridge returned malformed read response (invalid hex payload)") from None
    if args.out:
        from ..output import write_bytes_result

        result = write_bytes_result(
            data,
            out_path=args.out,
            fmt=args.format,
            summary={"kind": "bytes", "address": address, "length": len(data)},
        )
        sys.stdout.write(result.rendered)
    else:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    return 0


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


@command("batch", "apply", help="Apply a JSON manifest", fmt="json",
         args=[
             arg("--preview", action="store_true",
                 help="Apply the whole batch, capture diffs, then revert without committing"),
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
                     "as status 'invalid_request' naming the field."
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
        if not args.manifest.exists():
            raise BridgeError(f"Manifest file not found: {args.manifest}")
        try:
            raw = args.manifest.read_text(encoding="utf-8")
        except OSError as exc:
            raise BridgeError(f"Could not read manifest {args.manifest}: {exc}") from None
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BridgeError(f"Invalid JSON in manifest ({source}): {exc}") from None
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
    if not isinstance(manifest.get("ops"), list):
        raise BridgeError(
            f'Manifest ({source}) must have an "ops" array (the list of '
            f"operations to apply)."
        )
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
