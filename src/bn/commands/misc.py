from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..cli import _call, _mutation_exit_code, arg, command, mutex
from ..formatters import (
    _render_mutation_text,
    _render_name_address_list_text,
    _render_py_exec_text,
    _render_sections_text,
    _render_strings_text,
)
from ..transport import BridgeError


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
