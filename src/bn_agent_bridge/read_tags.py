"""Tag read handlers (free functions over the ``ctx`` seam).

Import direction is one-way: this module imports ``read_misc`` / ``_shared``
(plus stdlib + binaryninja) and NEVER imports ``bridge`` / ``mutation_engine`` /
``seam`` -- the same seam rule as the other ``read_*`` modules.
"""
from __future__ import annotations

from typing import Any

from . import read_misc
from ._shared import (
    _BUILTIN_TAG_TYPE_NAMES, _parse_address, _require_mapped_address, _validate_count,
)


def _tag_type_entry(tt) -> dict[str, Any]:
    name = str(tt.name)
    return {
        "name": name,
        "icon": str(getattr(tt, "icon", "")),
        "visible": bool(getattr(tt, "visible", True)),
        "is_builtin": name in _BUILTIN_TAG_TYPE_NAMES,
    }


def _list_tag_types(ctx, selector: str | None) -> dict[str, Any]:
    bv = ctx._resolve_view(selector)
    types = [_tag_type_entry(tt) for tt in bv.tag_types.values()]
    types.sort(key=lambda t: t["name"])
    return {"tag_types": types, "count": len(types)}


def _tag_entry(tag, *, scope: str, address: int | None, function: str | None) -> dict[str, Any]:
    tt = tag.type
    return {
        "id": str(tag.id),
        "type": str(tt.name),
        "icon": str(getattr(tt, "icon", "")),
        "data": str(tag.data),
        "scope": scope,
        "address": hex(int(address)) if address is not None else None,
        "function": function,
    }


def _get_tags(ctx, selector: str | None, address, function) -> dict[str, Any]:
    bv = ctx._resolve_view(selector)
    if function and address is not None:
        raise RuntimeError(
            "Pass an address or --function, not both: they target different locations."
        )
    if function:
        fn = ctx._find_function(bv, function)
        tags = [
            _tag_entry(t, scope="function", address=None, function=fn.name)
            for t in fn.get_function_tags(auto=False)
        ]
        return {"function": fn.name, "address": hex(int(fn.start)),
                "tags": tags, "count": len(tags)}

    if address is None:
        raise RuntimeError("tag get requires an address or --function")

    addr = _parse_address(address)
    tags: list[dict[str, Any]] = []
    funcs = bv.get_functions_containing(addr)
    fname = funcs[0].name if funcs else None
    for fn in funcs:
        for t in fn.get_tags_at(addr, auto=False):
            tags.append(_tag_entry(t, scope="address", address=addr, function=fn.name))
    for t in bv.get_tags_at(addr, auto=False):
        tags.append(_tag_entry(t, scope="data", address=addr, function=fname))
    # Reject a typo'd/stale address only when it is BOTH unmapped AND tag-less
    # (parity with comment get / xrefs, #374).
    if not tags:
        _require_mapped_address(bv, addr)
    return {"address": hex(addr), "tags": tags, "count": len(tags)}


def _collect_tags(ctx, bv, *, function, address, data_only) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    def push(entry: dict[str, Any]) -> None:
        if entry["id"] in seen:
            return
        seen.add(entry["id"])
        out.append(entry)

    if function is not None:
        fn = ctx._find_function(bv, function)
        for t in fn.get_function_tags(auto=False):
            push(_tag_entry(t, scope="function", address=None, function=fn.name))
        # Function.tags is BN's own "all address tags on this function" sweep
        # (verified live: function.py TagList / get_tags_at) -- NOT
        # bv.get_tags(), which surfaces only view-level DATA tags (see below).
        for _arch, addr, t in fn.tags:
            push(_tag_entry(t, scope="address", address=addr, function=fn.name))
        return out

    if address is not None:
        addr = _parse_address(address)
        funcs = bv.get_functions_containing(addr)
        fname = funcs[0].name if funcs else None
        for fn in funcs:
            for t in fn.get_tags_at(addr, auto=False):
                push(_tag_entry(t, scope="address", address=addr, function=fn.name))
        for t in bv.get_tags_at(addr, auto=False):
            push(_tag_entry(t, scope="data", address=addr, function=fname))
        return out

    # whole view: bv.get_tags(auto=False) is BN's DATA tag sweep (never
    # function-scoped address tags, verified live), so every entry it yields is
    # scope="data" -- attributed to a containing function (if any) for context,
    # matching _get_tags's convention, but not reclassified as "address".
    for addr, t in bv.get_tags(auto=False):
        funcs = bv.get_functions_containing(addr)
        fname = funcs[0].name if funcs else None
        push(_tag_entry(t, scope="data", address=addr, function=fname))
    if not data_only:
        for fn in list(bv.functions):
            for t in fn.get_function_tags(auto=False):
                push(_tag_entry(t, scope="function", address=None, function=fn.name))
            for _arch, addr, t in fn.tags:
                push(_tag_entry(t, scope="address", address=addr, function=fn.name))
    return out


def _list_tags(ctx, selector: str | None, *, function=None, address=None,
               type=None, data_only: bool = False, query=None,
               offset: int = 0, limit: int | None = None) -> dict[str, Any]:
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    bv = ctx._resolve_view(selector)
    items = _collect_tags(ctx, bv, function=function, address=address, data_only=data_only)
    if data_only:
        items = [t for t in items if t["scope"] == "data"]
    if type is not None:
        items = [t for t in items if t["type"] == type]
    if query:
        needle = query.lower()
        items = [t for t in items if needle in t["data"].lower()]
    # Sort by NUMERIC address, not the hex string -- lexicographic order put
    # "0x1000" before "0x900" (final-review Fix 2). Function-scope tags carry
    # address=None; sink those first via -1 so they don't interleave with
    # address-ordered entries.
    items.sort(key=lambda t: (int(t["address"], 16) if t["address"] else -1, t["scope"], t["type"]))
    return read_misc._paged_list_result(items, offset=offset, limit=limit, kind="tags")
