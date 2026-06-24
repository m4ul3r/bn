from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..cli import _call, _effective_limit, _mutate, arg, command, preview_arg, summary_arg
from ..formatters import (
    _render_type_info_text,
    _render_type_list_text,
)
from ..transport import BridgeError


@command("types", help="List or search types", target=True, paged=True,
         fanout=True,
         args=[arg("--query"),
               arg("--count", action="store_true", default=False,
                   help="Show the total type count instead of listing")])
def _types(args: argparse.Namespace) -> int:
    if args.count:
        return _call(
            args,
            "types",
            {"query": args.query, "count_only": True},
            require_target=True,
            text_renderer=lambda value: f"Total types: {value.get('count', 0)}",
            stem="types-count",
        )
    params = {"query": args.query, "offset": args.offset}
    limit = _effective_limit(args)
    if limit is not None:
        params["limit"] = limit
    return _call(
        args,
        "types",
        params,
        require_target=True,
        text_renderer=_render_type_list_text,
        # Bridge returns the {items,total,...} envelope and applies the page, so
        # forward the real limit/offset (above) and keep the spill hint -- no
        # client-side limit+1 probe (#131, mirrors strings/imports/#130).
        paged_spill=True,
        page_label="types",
        stem="types",
    )


@command("types", "show", help="Show one type", target=True,
         args=[arg("type_name")])
def _types_show(args: argparse.Namespace) -> int:
    return _call(
        args,
        "type_info",
        {
            "type_name": args.type_name,
            # The bridge accepts require_struct; `types show` never restricts
            # to structs (that's `struct show`), so this is always False.
            "require_struct": False,
        },
        require_target=True,
        text_renderer=_render_type_info_text,
        stem="type-show",
    )


@command("types", "declare", help="Import C declarations as user types", target=True, fmt="json",
         args=[
             preview_arg(), summary_arg(),
             arg("--file", type=Path, help="Read declarations from a file"),
             arg("--stdin", action="store_true", help="Read declarations from stdin"),
             arg("declaration", nargs="?"),
         ])
def _types_declare(args: argparse.Namespace) -> int:
    # Exactly one declaration source. The handler used to pick file > stdin >
    # positional and silently ignore the rest, so a script could apply a
    # different declaration than the one it visibly passed (#94).
    provided = [
        label for label, present in (
            ("--file", args.file is not None),
            ("--stdin", bool(args.stdin)),
            ("a declaration string", args.declaration is not None),
        ) if present
    ]
    if not provided:
        raise BridgeError("Provide a declaration string, --file, or --stdin")
    if len(provided) > 1:
        raise BridgeError(
            f"Provide exactly one declaration source, but got {len(provided)}: "
            f"{', '.join(provided)}."
        )
    source_path = None
    if args.file is not None:
        if not args.file.exists():
            raise BridgeError(f"Declaration file not found: {args.file}")
        declaration = args.file.read_text(encoding="utf-8")
        source_path = str(args.file)
    elif args.stdin:
        declaration = sys.stdin.read()
    else:
        declaration = args.declaration

    return _mutate(
        args,
        "types_declare",
        {
            "declaration": declaration,
            "source_path": source_path,
        },
        preview=bool(args.preview),
        stem="types-declare",
    )


@command("struct", "show", help="Show one struct layout", target=True,
         args=[arg("struct_name")])
def _struct_show(args: argparse.Namespace) -> int:
    return _call(
        args,
        "type_info",
        {
            "type_name": args.struct_name,
            "require_struct": True,
        },
        require_target=True,
        text_renderer=_render_type_info_text,
        stem="struct-show",
    )


@command("struct", "field", "set", help="Set or replace a field", target=True, fmt="json",
         args=[
             preview_arg(), summary_arg(),
             arg("--no-overwrite", action="store_true"),
             arg("struct_name"),
             arg("offset"),
             arg("field_name"),
             arg("field_type"),
         ])
def _struct_field_set(args: argparse.Namespace) -> int:
    return _mutate(
        args,
        "struct_field_set",
        {
            "struct_name": args.struct_name,
            "offset": args.offset,
            "field_name": args.field_name,
            "field_type": args.field_type,
            "overwrite_existing": not args.no_overwrite,
        },
        preview=bool(args.preview),
        stem="struct-field-set",
    )


@command("struct", "field", "rename", help="Rename a field", target=True, fmt="json",
         args=[
             preview_arg(), summary_arg(),
             arg("struct_name"),
             arg("old_name", help="Field name or offset (e.g. count or 0x8)"),
             arg("new_name"),
         ])
def _struct_field_rename(args: argparse.Namespace) -> int:
    return _mutate(
        args,
        "struct_field_rename",
        {
            "struct_name": args.struct_name,
            "old_name": args.old_name,
            "new_name": args.new_name,
        },
        preview=bool(args.preview),
        stem="struct-field-rename",
    )


@command("struct", "field", "delete", help="Delete a field", target=True, fmt="json",
         args=[
             preview_arg(), summary_arg(),
             arg("struct_name"),
             arg("field_name", help="Field name or offset (e.g. count or 0x8)"),
         ])
def _struct_field_delete(args: argparse.Namespace) -> int:
    return _mutate(
        args,
        "struct_field_delete",
        {
            "struct_name": args.struct_name,
            "field_name": args.field_name,
        },
        preview=bool(args.preview),
        stem="struct-field-delete",
    )
