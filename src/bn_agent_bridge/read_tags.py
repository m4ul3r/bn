"""Tag read handlers (free functions over the ``ctx`` seam).

Import direction is one-way: this module imports ``read_misc`` / ``_shared``
(plus stdlib + binaryninja) and NEVER imports ``bridge`` / ``mutation_engine`` /
``seam`` -- the same seam rule as the other ``read_*`` modules.
"""
from __future__ import annotations

from typing import Any

from . import read_misc
from ._shared import _BUILTIN_TAG_TYPE_NAMES, _parse_address, _require_mapped_address


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
