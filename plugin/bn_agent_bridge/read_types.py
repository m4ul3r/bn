"""Type listing and type-info lookups.

The ``types`` (list/search) and ``type_info`` read ops that used to live on
``BinaryNinjaBridge`` move here as module-level free functions, each taking the
``BridgeContext`` seam (``ctx``) in place of ``self``. ``BinaryNinjaBridge``
keeps a thin delegating shim for every name the test suite / op binders
reference (``_types``, ``_type_info``).

Outbound calls resolve through:
  * ``ctx`` -- the type-entry builders and resolver relocated to the seam in
    Stage 2/4: ``_resolve_view``, ``_find_type``, ``_type_entry`` (and
    ``_current_type_entry``). Relocating the type-entry builders to the seam is
    what breaks the ``read_types <-> mutation_engine`` cycle, so this module
    never imports the mutation engine.
  * ``_shared`` -- module-free helpers (``_validate_count``).

Import direction is one-way: this module imports ``_shared`` (plus
``binaryninja``). It NEVER imports ``bridge``, ``seam``, or ``mutation_engine``
-- those depend on the seam, not on this module (design spec §3.2).
"""
from __future__ import annotations

import binaryninja as bn  # noqa: F401  (kept for parity with sibling read modules)

from . import read_misc
from ._shared import _validate_count


def _types(ctx, selector: str | None, *, query, offset: int, limit: int | None):
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    bv = ctx._resolve_view(selector)
    items = []
    needle = str(query).lower() if query else None
    for name, type_obj in list(bv.types.items()):
        entry = ctx._type_entry(name, type_obj)
        if needle and needle not in entry["name"].lower() and needle not in entry["decl"].lower():
            continue
        items.append(entry)
    items.sort(key=lambda item: item["name"].lower())
    # Honest paging envelope ({items,total,offset,limit,returned,has_more}),
    # matching strings/imports/sections/function-list (#122/#131).
    return read_misc._paged_list_result(items, offset=offset, limit=limit)


def _type_info(ctx, selector: str | None, type_name: str, *, require_struct: bool = False):
    bv = ctx._resolve_view(selector)
    resolved_name, type_obj = ctx._find_type(bv, type_name)
    members = getattr(type_obj, "members", None)
    if require_struct and members is None:
        raise RuntimeError(f"Type is not a struct-like type: {resolved_name}")
    return ctx._type_entry(resolved_name, type_obj)
