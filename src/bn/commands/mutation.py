from __future__ import annotations

import argparse

from ..cli import _call, _mutate, _pick, arg, command, preview_arg
from ..formatters import (
    _render_comment_list_text,
    _render_comment_text,
    _render_local_list_text,
    _render_proto_text,
)
from ..transport import BridgeError


_RENAME_ARGS = [
    arg("--kind", choices=("auto", "function", "data"), default="auto",
        help="Symbol kind to rename: auto-detect (default), function, or data"),
    preview_arg(),
    arg("identifier", help="Current symbol name or address (hex 0x.. or decimal)"),
    arg("new_name", help="New symbol name"),
]


@command("rename",
         help="Rename a function or data symbol (alias for `symbol rename`; "
              "use `local rename` / `struct field rename` for locals and fields)",
         target=True, fmt="json", args=_RENAME_ARGS)
@command("symbol", "rename", help="Rename a symbol", target=True, fmt="json", args=_RENAME_ARGS)
def _symbol_rename(args: argparse.Namespace) -> int:
    return _mutate(
        args,
        "rename_symbol",
        {
            "kind": args.kind,
            "identifier": args.identifier,
            "new_name": args.new_name,
        },
        preview=bool(args.preview),
        stem="symbol-rename",
    )


@command("comment", "list", help="List comments", target=True, paged=True,
         args=[arg("--query", help="Filter comments by substring")])
def _comment_list(args: argparse.Namespace) -> int:
    return _call(
        args,
        "list_comments",
        {"query": args.query, "offset": args.offset, "limit": args.limit},
        require_target=True,
        text_renderer=_render_comment_list_text,
        # Bridge returns the {items,total,...} envelope and applies the page, so
        # forward the real limit/offset (above) and keep the spill hint -- no
        # client-side limit+1 probe (#131, mirrors strings/imports/#130).
        paged_spill=True,
        page_label="comments",
        stem="comments",
    )


# A positional address mirrors `bn read 0x..` so the natural first guess
# `bn comment <verb> 0x1234 ...` works instead of erroring with the value
# silently dropped (#291.1). --address keeps working as the flag alias; both
# target a different location than --function.
def _comment_locator_args() -> list:
    return [
        arg("address", nargs="?",
            help="Address to comment (hex 0x.. or decimal); alias for --address"),
        arg("--address", dest="address_flag", default=None,
            help="Address to comment (alias for the positional)"),
        arg("--function", default=None,
            help="Attach the comment to a function (name or address) instead of an address"),
    ]


def _comment_locator(args: argparse.Namespace, verb: str) -> tuple[str | None, str | None]:
    """Reconcile the positional address with --address, then enforce exactly one of
    address / function. --address and --function target different locations and the
    bridge checks function first, so accepting both silently dropped the address
    (#94); a missing locator is a clear error rather than a dropped value (#291.1)."""
    address = _pick(args.address, args.address_flag, "comment address", required=False)
    function = args.function
    if address is not None and function is not None:
        raise BridgeError(f"comment {verb} takes an address or --function, not both")
    if address is None and function is None:
        raise BridgeError(
            f"comment {verb} needs a location: an address (positional or --address) or --function")
    return address, function


@command("comment", "set", help="Set a comment", target=True, fmt="json",
         args=[
             preview_arg(),
             arg("address", nargs="?",
                 help="Address to comment (hex 0x.. or decimal); alias for --address"),
             arg("comment", help="Comment text"),
             # Catch the over-specified `comment set <fn> <addr> "text"` form so it
             # gets a clear arity error instead of argparse blaming the comment
             # text as "unrecognized arguments" (#312).
             arg("extra", nargs="*", help=argparse.SUPPRESS),
             arg("--address", dest="address_flag", default=None,
                 help="Address to comment (alias for the positional)"),
             arg("--function", default=None,
                 help="Attach the comment to a function (name or address) instead of an address"),
         ])
def _comment_set(args: argparse.Namespace) -> int:
    if getattr(args, "extra", None):
        raise BridgeError(
            "comment set takes one address and one (quoted) comment: "
            "`comment set <addr> \"<text>\"`, or attach to a function with "
            "`comment set --function <name> \"<text>\"`. A comment is set at a "
            "single address, not at both a function and an address positionally; "
            f"got extra argument(s): {' '.join(args.extra)!r}. "
            "If the comment text has spaces, quote it as one argument."
        )
    address, function = _comment_locator(args, "set")
    return _mutate(
        args,
        "set_comment",
        {
            "address": address,
            "function": function,
            "comment": args.comment,
        },
        preview=bool(args.preview),
        stem="comment-set",
    )


@command("comment", "get", help="Get a comment", target=True,
         args=_comment_locator_args())
def _comment_get(args: argparse.Namespace) -> int:
    address, function = _comment_locator(args, "get")
    return _call(
        args,
        "get_comment",
        {
            "address": address,
            "function": function,
        },
        require_target=True,
        text_renderer=_render_comment_text,
        stem="comment-get",
    )


@command("comment", "delete", help="Delete a comment", target=True, fmt="json",
         args=[preview_arg(), *_comment_locator_args()])
def _comment_delete(args: argparse.Namespace) -> int:
    address, function = _comment_locator(args, "delete")
    return _mutate(
        args,
        "delete_comment",
        {
            "address": address,
            "function": function,
        },
        preview=bool(args.preview),
        stem="comment-delete",
    )


@command("proto", "set", help="Set a prototype", target=True, fmt="json",
         args=[
             preview_arg(),
             arg("identifier", help="Function name or address (hex 0x.. or decimal)"),
             arg("prototype", help="Full C prototype string, e.g. \"int __cdecl f(Player* self)\""),
         ])
def _proto_set(args: argparse.Namespace) -> int:
    return _mutate(
        args,
        "set_prototype",
        {
            "identifier": args.identifier,
            "prototype": args.prototype,
        },
        preview=bool(args.preview),
        stem="prototype-set",
    )


@command("proto", "get", help="Show the current prototype", target=True,
         args=[arg("identifier", help="Function name or address (hex 0x.. or decimal)")])
def _proto_get(args: argparse.Namespace) -> int:
    return _call(
        args,
        "get_prototype",
        {"identifier": args.identifier},
        require_target=True,
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
        text_renderer=_render_local_list_text,
        stem="local-list",
    )


@command("local", "rename", help="Rename a local", target=True, fmt="json",
         args=[
             preview_arg(),
             arg("function"),
             arg("variable", help="Stable local_id or legacy variable name"),
             arg("new_name"),
         ])
def _local_rename(args: argparse.Namespace) -> int:
    return _mutate(
        args,
        "local_rename",
        {
            "function": args.function,
            "variable": args.variable,
            "new_name": args.new_name,
        },
        preview=bool(args.preview),
        stem="local-rename",
    )


@command("local", "retype", help="Retype a local", target=True, fmt="json",
         args=[
             preview_arg(),
             arg("function"),
             arg("variable", help="Stable local_id or legacy variable name"),
             arg("new_type"),
         ])
def _local_retype(args: argparse.Namespace) -> int:
    return _mutate(
        args,
        "local_retype",
        {
            "function": args.function,
            "variable": args.variable,
            "new_type": args.new_type,
        },
        preview=bool(args.preview),
        stem="local-retype",
    )
