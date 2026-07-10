"""`bn tag` command group: tag types, reads, and mutations."""
from __future__ import annotations

import argparse

from ..cli import _call, _mutate, _pick, arg, command, preview_arg, summary_arg
from ..formatters import _render_tag_get_text, _render_tag_list_text, _render_tag_types_text
from ..transport import BridgeError


@command("tag", "types", help="List tag types (name, icon, built-in)", target=True)
def _tag_types(args: argparse.Namespace) -> int:
    return _call(
        args,
        "list_tag_types",
        {},
        require_target=True,
        text_renderer=_render_tag_types_text,
        stem="tag-types",
    )


def _tag_locator_args() -> list:
    return [
        arg("address", nargs="?",
            help="Address to read tags at (hex 0x.. or decimal); alias for --address"),
        arg("--address", dest="address_flag", default=None,
            help="Address to read tags at (alias for the positional)"),
        arg("--function", default=None,
            help="Read the whole-function tags of a function (name or address)"),
    ]


def _tag_locator(args: argparse.Namespace, verb: str) -> tuple[str | None, str | None]:
    address = _pick(args.address, args.address_flag, "tag address", required=False)
    function = args.function
    if address is not None and function is not None:
        raise BridgeError(f"tag {verb} takes an address or --function, not both")
    if address is None and function is None:
        raise BridgeError(
            f"tag {verb} needs a location: an address (positional or --address) or --function")
    return address, function


@command("tag", "get", help="Get tags at an address or on a function", target=True,
         args=_tag_locator_args())
def _tag_get(args: argparse.Namespace) -> int:
    address, function = _tag_locator(args, "get")
    return _call(
        args,
        "get_tags",
        {"address": address, "function": function},
        require_target=True,
        text_renderer=_render_tag_get_text,
        stem="tag-get",
    )


@command("tag", "list", help="List tags (all scopes, filterable)", target=True, paged=True,
         args=[
             arg("--function", default=None, help="Only tags belonging to this function"),
             arg("--address", default=None, help="Only tags at this address"),
             arg("--type", default=None, help="Filter by tag type name"),
             arg("--data", dest="data_only", action="store_true",
                 help="Only data-scope tags (not function/address tags)"),
             arg("--query", default=None, help="Filter by substring of the tag data"),
         ])
def _tag_list(args: argparse.Namespace) -> int:
    return _call(
        args,
        "list_tags",
        {
            "function": args.function,
            "address": args.address,
            "type": args.type,
            "data_only": bool(args.data_only),
            "query": args.query,
            "offset": args.offset,
            "limit": args.limit,
        },
        require_target=True,
        text_renderer=_render_tag_list_text,
        paged_spill=True,
        page_label="tags",
        stem="tags",
    )


@command("tag", "add", help="Add a tag at an address or on a function", target=True, fmt="json",
         args=[
             preview_arg(), summary_arg(),
             arg("address", nargs="?", help="Address to tag (hex 0x.. or decimal); alias for --address"),
             arg("--address", dest="address_flag", default=None, help="Address alias for the positional"),
             arg("--function", default=None, help="Tag the whole function (name or address)"),
             arg("--type", required=True, help="Tag type name (e.g. Important, Bookmarks)"),
             arg("--data", dest="data", default="", help="Tag text (data)"),
             arg("--data-scope", dest="force_data", action="store_true",
                 help="Force a data-scope tag at the address (not attached to a function)"),
         ])
def _tag_add(args: argparse.Namespace) -> int:
    """Add a tag at an address or on a function.

    Examples:
        bn tag add 0x1000 --type Important --data "note"
        bn tag add --function sub_1000 --type Important --data "doc"
        bn tag add 0x1000 --type Library --data "libc" --data-scope
    """
    address = _pick(args.address, args.address_flag, "tag address", required=False)
    if address is not None and args.function is not None:
        raise BridgeError("tag add takes an address or --function, not both")
    if address is None and args.function is None:
        raise BridgeError("tag add needs a location: an address or --function")
    if args.function is not None and args.force_data:
        raise BridgeError(
            "tag add: --data-scope can't be combined with --function "
            "(a function tag has no address); drop one")
    return _mutate(
        args,
        "tag_add",
        {"type": args.type, "data": args.data, "address": address,
         "function": args.function, "force_data": bool(args.force_data)},
        preview=bool(args.preview),
        stem="tag-add",
    )


@command("tag", "remove", help="Remove tag(s) by id, or by type at a location", target=True, fmt="json",
         args=[
             preview_arg(), summary_arg(),
             arg("address", nargs="?", help="Address to remove tags at (hex 0x.. or decimal)"),
             arg("--address", dest="address_flag", default=None, help="Address alias for the positional"),
             arg("--function", default=None, help="Remove whole-function tags of this function"),
             arg("--id", dest="tag_id", default=None, help="Remove exactly this tag id (from `tag list`)"),
             arg("--type", default=None, help="Tag type name to match"),
             arg("--data", dest="data_match", default=None, help="Only remove tags whose data equals this"),
             arg("--data-scope", dest="force_data", action="store_true",
                 help="Only remove data-scope tags at the address (skip function/address tags)"),
         ])
def _tag_remove(args: argparse.Namespace) -> int:
    """Remove tag(s) by id, or by type at a location.

    Examples:
        bn tag remove --id 0123abcd-...
        bn tag remove 0x1000 --type Important
        bn tag remove --function sub_1000 --type Important --data "doc"
    """
    address = _pick(args.address, args.address_flag, "tag address", required=False)
    if args.tag_id is None and args.type is None:
        raise BridgeError("tag remove needs --id, or --type with a location (address or --function)")
    if args.tag_id is None and address is None and args.function is None:
        raise BridgeError("tag remove by --type needs an address or --function")
    if address is not None and args.function is not None:
        raise BridgeError("tag remove takes an address or --function, not both")
    return _mutate(
        args,
        "tag_remove",
        {"tag_id": args.tag_id, "type": args.type, "data": args.data_match,
         "address": address, "function": args.function, "force_data": bool(args.force_data)},
        preview=bool(args.preview),
        stem="tag-remove",
    )


@command("tag", "type", "create", help="Create a custom tag type", target=True, fmt="json",
         args=[preview_arg(), summary_arg(),
               arg("name", help="Tag type name"),
               arg("--icon", required=True, help="Emoji icon for the tag type, e.g. 🎯")])
def _tag_type_create(args: argparse.Namespace) -> int:
    return _mutate(args, "tag_type_create",
                   {"name": args.name, "icon": args.icon},
                   preview=bool(args.preview), stem="tag-type-create")


@command("tag", "type", "remove", help="Remove a custom tag type", target=True, fmt="json",
         args=[preview_arg(), summary_arg(),
               arg("name", help="Custom tag type name (built-ins cannot be removed)")])
def _tag_type_remove(args: argparse.Namespace) -> int:
    return _mutate(args, "tag_type_remove",
                   {"name": args.name},
                   preview=bool(args.preview), stem="tag-type-remove")
