from __future__ import annotations

import argparse

from ..cli import _call, _mutation_exit_code, arg, command
from ..formatters import (
    _render_comment_list_text,
    _render_comment_text,
    _render_local_list_text,
    _render_mutation_text,
    _render_proto_text,
)


@command("symbol", "rename", help="Rename a symbol", target=True,
         args=[
             arg("--kind", choices=("auto", "function", "data"), default="auto"),
             arg("--preview", action="store_true"),
             arg("identifier"),
             arg("new_name"),
         ])
def _symbol_rename(args: argparse.Namespace) -> int:
    return _call(
        args,
        "rename_symbol",
        {
            "kind": args.kind,
            "identifier": args.identifier,
            "new_name": args.new_name,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="symbol-rename",
        result_exit_code=_mutation_exit_code,
    )


@command("comment", "list", help="List comments", target=True, paged=True,
         args=[arg("--query", help="Filter comments by substring")])
def _comment_list(args: argparse.Namespace) -> int:
    return _call(
        args,
        "list_comments",
        {"query": args.query, "offset": args.offset, "limit": args.limit},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_comment_list_text,
        page_limit=args.limit,
        page_offset=args.offset,
        page_label="comments",
        stem="comments",
    )


@command("comment", "set", help="Set a comment", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("--address"),
             arg("--function"),
             arg("comment"),
         ])
def _comment_set(args: argparse.Namespace) -> int:
    return _call(
        args,
        "set_comment",
        {
            "address": args.address,
            "function": args.function,
            "comment": args.comment,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="comment-set",
        result_exit_code=_mutation_exit_code,
    )


@command("comment", "get", help="Get a comment", target=True,
         args=[arg("--address"), arg("--function")])
def _comment_get(args: argparse.Namespace) -> int:
    return _call(
        args,
        "get_comment",
        {
            "address": args.address,
            "function": args.function,
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_comment_text,
        stem="comment-get",
    )


@command("comment", "delete", help="Delete a comment", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("--address"),
             arg("--function"),
         ])
def _comment_delete(args: argparse.Namespace) -> int:
    return _call(
        args,
        "delete_comment",
        {
            "address": args.address,
            "function": args.function,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="comment-delete",
        result_exit_code=_mutation_exit_code,
    )


@command("proto", "set", help="Set a prototype", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("identifier"),
             arg("prototype"),
         ])
def _proto_set(args: argparse.Namespace) -> int:
    return _call(
        args,
        "set_prototype",
        {
            "identifier": args.identifier,
            "prototype": args.prototype,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="prototype-set",
        result_exit_code=_mutation_exit_code,
    )


@command("proto", "get", help="Show the current prototype", target=True,
         args=[arg("identifier")])
def _proto_get(args: argparse.Namespace) -> int:
    return _call(
        args,
        "get_prototype",
        {"identifier": args.identifier},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_proto_text,
        stem="prototype-get",
    )


@command("local", "list", help="List locals with stable IDs", target=True,
         args=[arg("function")])
def _local_list(args: argparse.Namespace) -> int:
    return _call(
        args,
        "list_locals",
        {"identifier": args.function},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_local_list_text,
        stem="local-list",
    )


@command("local", "rename", help="Rename a local", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("function"),
             arg("variable", help="Stable local_id or legacy variable name"),
             arg("new_name"),
         ])
def _local_rename(args: argparse.Namespace) -> int:
    return _call(
        args,
        "local_rename",
        {
            "function": args.function,
            "variable": args.variable,
            "new_name": args.new_name,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="local-rename",
        result_exit_code=_mutation_exit_code,
    )


@command("local", "retype", help="Retype a local", target=True,
         args=[
             arg("--preview", action="store_true"),
             arg("function"),
             arg("variable", help="Stable local_id or legacy variable name"),
             arg("new_type"),
         ])
def _local_retype(args: argparse.Namespace) -> int:
    return _call(
        args,
        "local_retype",
        {
            "function": args.function,
            "variable": args.variable,
            "new_type": args.new_type,
            "preview": bool(args.preview),
        },
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_mutation_text,
        stem="local-retype",
        result_exit_code=_mutation_exit_code,
    )
