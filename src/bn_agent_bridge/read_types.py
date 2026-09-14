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

try:
    import binaryninja as bn  # noqa: F401  (kept for parity with sibling read modules)
except ModuleNotFoundError:  # importable without the Binary Ninja runtime (tests, tooling)
    bn = None  # type: ignore[assignment]

from . import read_misc
from ._shared import _validate_count


def _types(ctx, selector: str | None, *, query, offset: int, limit: int | None,
           count_only: bool = False):
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
    if count_only:
        return {"kind": "types", "count": len(items), "total": len(items)}
    items.sort(key=lambda item: item["name"].lower())
    # Honest paging envelope ({kind,items,total,offset,limit,returned,has_more}),
    # matching strings/imports/sections/function-list (#122/#131).
    return read_misc._paged_list_result(items, offset=offset, limit=limit, kind="types")


# ``typedef struct { ... } Alias;`` is registered by BN the idiomatic C way: the
# body becomes a named struct (auto-named ``_Alias`` when anonymous) and ``Alias``
# becomes a NamedTypeReference to it, so the alias object itself carries no
# ``members``. A chain of typedefs can be arbitrarily long (and a malformed one
# even self-referential), so the follow is bounded.
_MAX_TYPEDEF_FOLLOW = 16


def _is_named_type_ref(type_obj) -> bool:
    """True if *type_obj* is BN's ``TypeClass.NamedTypeReference`` (11), i.e. a
    typedef alias rather than a concrete type. Duck-typed: the unit fakes carry a
    plain string type_class."""
    tc = getattr(type_obj, "type_class", None)
    if tc is None:
        return False
    try:
        if int(tc) == 11:  # TypeClass.NamedTypeReferenceClass
            return True
    except (TypeError, ValueError):
        pass
    name = str(getattr(tc, "name", None) or tc)
    return "NamedTypeReference" in name or "typedef" in name


def _follow_typedef(bv, resolved_name: str, type_obj):
    """Follow a typedef chain to the underlying registered type so struct-shaped
    reads can see the body's members (#674). Returns ``(name, type_obj, reason)``;
    *reason* is None when the chain was followed to a terminal type, and otherwise
    says why it could not be: an unresolvable ``target()``, a cycle, or a chain
    past ``_MAX_TYPEDEF_FOLLOW``. A called-out reason is what lets the caller
    report an unresolvable alias distinctly from a type that really has no
    members."""
    seen: list = []
    while _is_named_type_ref(type_obj):
        if any(type_obj is prior for prior in seen):
            return resolved_name, type_obj, "the typedef chain is cyclic"
        if len(seen) >= _MAX_TYPEDEF_FOLLOW:
            return (resolved_name, type_obj,
                    f"the typedef chain is longer than {_MAX_TYPEDEF_FOLLOW} hops")
        seen.append(type_obj)
        try:
            target = type_obj.target(bv)
        except Exception:
            target = None
        if target is None:
            return (resolved_name, type_obj,
                    "its target type could not be resolved")
        registered = getattr(getattr(target, "registered_name", None), "name", None)
        if registered:
            resolved_name = str(registered)
        type_obj = target
    return resolved_name, type_obj, None


def _type_info(ctx, selector: str | None, type_name: str, *, require_struct: bool = False):
    bv = ctx._resolve_view(selector)
    resolved_name, type_obj = ctx._find_type(bv, type_name)
    if require_struct:
        resolved_name, type_obj, follow_error = _follow_typedef(bv, resolved_name, type_obj)
        if follow_error is not None:
            # The alias could not be followed at all: reporting "not a struct-like
            # type" here is actively wrong (it may well be struct-like) and hides
            # the real failure, so name the alias and the reason (#674 review).
            raise RuntimeError(
                f"Could not resolve typedef {resolved_name!r} to a struct: {follow_error}"
            )
        if getattr(type_obj, "members", None) is None:
            raise RuntimeError(f"Type is not a struct-like type: {resolved_name}")
    return ctx._type_entry(resolved_name, type_obj)
