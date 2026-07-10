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
