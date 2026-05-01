from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..cli import _call, _mutation_exit_code, arg, command
from ..formatters import (
    _render_mutation_text,
    _render_type_info_text,
    _render_type_list_text,
)
from ..transport import BridgeError


@command("types", help="List or search types", target=True, paged=True,
         args=[arg("--query")])
def _types(args: argparse.Namespace) -> int:
    return _call(
        args,
        "types",
        {"query": args.query, "offset": args.offset, "limit": args.limit},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_type_list_text,
        page_limit=args.limit,
        page_offset=args.offset,
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
            "require_struct": bool(getattr(args, "require_struct", False)),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_type_info_text,
        stem="type-show",
    )


@command("types", "declare", help="Import C declarations as user types", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("--file", type=Path, help="Read declarations from a file"),
             arg("--stdin", action="store_true", help="Read declarations from stdin"),
             arg("declaration", nargs="?"),
         ])
def _types_declare(args: argparse.Namespace) -> int:
    source_path = None
    if args.file is not None:
        if not args.file.exists():
            raise BridgeError(f"Declaration file not found: {args.file}")
        declaration = args.file.read_text(encoding="utf-8")
        source_path = str(args.file)
    elif args.stdin:
        declaration = sys.stdin.read()
    elif args.declaration:
        declaration = args.declaration
    else:
        raise BridgeError("Provide a declaration string, --file, or --stdin")

    return _call(
        args,
        "types_declare",
        {
            "declaration": declaration,
            "source_path": source_path,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="types-declare",
        result_exit_code=_mutation_exit_code,
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
        allow_implicit_target=True,
        text_renderer=_render_type_info_text,
        stem="struct-show",
    )


@command("struct", "field", "set", help="Set or replace a field", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("--no-overwrite", action="store_true"),
             arg("struct_name"),
             arg("offset"),
             arg("field_name"),
             arg("field_type"),
         ])
def _struct_field_set(args: argparse.Namespace) -> int:
    return _call(
        args,
        "struct_field_set",
        {
            "struct_name": args.struct_name,
            "offset": args.offset,
            "field_name": args.field_name,
            "field_type": args.field_type,
            "overwrite_existing": not args.no_overwrite,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="struct-field-set",
        result_exit_code=_mutation_exit_code,
    )


@command("struct", "field", "rename", help="Rename a field", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("struct_name"),
             arg("old_name"),
             arg("new_name"),
         ])
def _struct_field_rename(args: argparse.Namespace) -> int:
    return _call(
        args,
        "struct_field_rename",
        {
            "struct_name": args.struct_name,
            "old_name": args.old_name,
            "new_name": args.new_name,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="struct-field-rename",
        result_exit_code=_mutation_exit_code,
    )


@command("struct", "field", "delete", help="Delete a field", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("struct_name"),
             arg("field_name"),
         ])
def _struct_field_delete(args: argparse.Namespace) -> int:
    return _call(
        args,
        "struct_field_delete",
        {
            "struct_name": args.struct_name,
            "field_name": args.field_name,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="struct-field-delete",
        result_exit_code=_mutation_exit_code,
    )
