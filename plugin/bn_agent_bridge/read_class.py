"""C++ object-model lens (#205): class registry, vtable layout, RTTI bases,
object size, and instance tracking, correlated from data Binary Ninja already
recovers (demangled symbols, RTTI data symbols, operator-new sizes).

All functions are read-only and take the BridgeContext seam (``ctx``); this
module never imports ``bridge`` or ``mutation_engine``."""
from __future__ import annotations

from typing import Any

from . import il_format
from ._shared import OperationFailure, _parse_address


def _strip_signature(name: str) -> str:
    """Return *name* with its trailing parameter-list ``(...)`` removed.

    Depth-aware over angle brackets so a '(' inside template args is ignored.
    The parameter list is the LAST top-level balanced paren group."""
    angle = 0
    paren = 0
    open_idx = None
    last_param_open = None
    for i, ch in enumerate(name):
        if ch == "<":
            angle += 1
        elif ch == ">" and angle:
            angle -= 1
        elif ch == "(" and angle == 0:
            if paren == 0:
                open_idx = i
            paren += 1
        elif ch == ")" and angle == 0 and paren:
            paren -= 1
            if paren == 0 and open_idx is not None:
                last_param_open = open_idx
    if last_param_open is not None:
        return name[:last_param_open]
    return name


def _toplevel_operator_index(head: str) -> int | None:
    """Index of a top-level ``operator`` keyword in *head*, else None. The
    method name begins here (it may itself contain '::', e.g. a conversion to a
    qualified type), so the class is whatever precedes the '::' before it."""
    angle = 0
    i = 0
    n = len(head)
    while i < n:
        ch = head[i]
        if ch == "<":
            angle += 1
        elif ch == ">" and angle:
            angle -= 1
        elif (
            angle == 0
            and head.startswith("operator", i)
            and (i == 0 or head[i - 1] in ":< ,(")
        ):
            return i
        i += 1
    return None


def _last_toplevel_scope(head: str) -> int | None:
    """Index of the last top-level ``::`` in *head* (angle-depth 0), else None."""
    angle = 0
    last = None
    i = 0
    n = len(head)
    while i < n:
        ch = head[i]
        if ch == "<":
            angle += 1
        elif ch == ">" and angle:
            angle -= 1
        elif ch == ":" and angle == 0 and i + 1 < n and head[i + 1] == ":":
            last = i
            i += 2
            continue
        i += 1
    return last


def _split_qualified_method(demangled: str) -> tuple[str | None, str]:
    """Split a demangled C++ name into ``(class_name, method)``.

    ``class_name`` is None for names with no scope qualifier. '::' inside
    ``<...>`` or ``(...)`` never splits. Handles ctor/dtor/operator forms."""
    name = (demangled or "").strip()
    if not name:
        return None, name
    head = _strip_signature(name)
    op_idx = _toplevel_operator_index(head)
    if op_idx is not None:
        cls = head[:op_idx].rstrip(": ")
        return (cls or None), name[op_idx:].strip()
    split = _last_toplevel_scope(head)
    if split is None:
        return None, name
    return head[:split], name[split + 2:].strip()


# Itanium RTTI data-symbol tags. BN renders the demangled short_name with a
# leading "vtable for " / "typeinfo for " / "typeinfo name for " marker (older
# BN uses an underscored "_vtable_for_" form); strip either to get the class.
_RTTI_TAGS = (
    ("_ZTV", "vtable", ("vtable for ", "_vtable_for_")),
    ("_ZTI", "typeinfo", ("typeinfo for ", "_typeinfo_for_")),
    ("_ZTS", "typeinfo_name", ("typeinfo name for ", "_typeinfo_name_for_")),
)


def _class_of_rtti_symbol(sym, prefixes) -> str | None:
    """Class name for an RTTI symbol: strip the demangled marker, else None."""
    short = str(getattr(sym, "short_name", "") or "")
    for marker in prefixes:
        if short.startswith(marker):
            return short[len(marker):].strip()
    return None


def _rtti_symbol_maps(bv) -> dict[str, dict[str, Any]]:
    """{class_name: {"vtable": sym, "typeinfo": sym, "typeinfo_name": sym}}."""
    maps: dict[str, dict[str, Any]] = {}
    for sym in bv.get_symbols():
        raw = str(getattr(sym, "raw_name", "") or getattr(sym, "name", "") or "")
        for prefix, key, markers in _RTTI_TAGS:
            if not raw.startswith(prefix):
                continue
            cls = _class_of_rtti_symbol(sym, markers)
            if cls:
                maps.setdefault(cls, {}).setdefault(key, sym)
            break
    return maps


def _last_component(class_name: str) -> str:
    """Final ``::`` component (ignoring template args) — the ctor/dtor name."""
    head = _strip_signature(class_name)
    idx = _last_toplevel_scope(head)
    comp = head[idx + 2:] if idx is not None else head
    # Drop any template suffix on the component itself.
    angle = comp.find("<")
    return comp[:angle] if angle != -1 else comp


def _method_kind(class_name: str, method: str) -> str:
    """ctor / dtor / method, from the demangled method spelling."""
    last = _last_component(class_name)
    name = method.split("(", 1)[0].strip()
    if name == f"~{last}":
        return "dtor"
    if name == last:
        return "ctor"
    return "method"


def _sym_entry(sym) -> dict[str, Any] | None:
    if sym is None:
        return None
    return {
        "address": hex(int(getattr(sym, "address", 0))),
        "symbol": str(getattr(sym, "raw_name", "") or getattr(sym, "name", "")),
    }


def _build_class_registry(ctx, bv, *, query: str | None = None) -> dict[str, dict[str, Any]]:
    """One scan -> {class_name: ClassRecord}. Methods, RTTI symbols, confidence.
    Per-class drill-downs (vtable layout, size, bases, instances) are added by
    ``_class_show`` only for the requested class (too costly for every class)."""
    rtti = _rtti_symbol_maps(bv)
    registry: dict[str, dict[str, Any]] = {}
    needle = query.lower() if query else None

    for fn in bv.functions:
        demangled = il_format._display_name(fn)
        cls, method = _split_qualified_method(demangled)
        if cls is None:
            continue
        rec = registry.get(cls)
        if rec is None:
            rec = registry[cls] = {
                "name": cls,
                "methods": [],
                "vtable": None,
                "typeinfo": None,
                "typeinfo_name": None,
                "size": None,
                "bases": [],
                "instances": [],
                "confidence": "name-only",
            }
        rec["methods"].append({
            "address": hex(int(getattr(fn, "start", 0))),
            "mangled": str(getattr(fn, "name", "")),
            "demangled": demangled,
            "kind": _method_kind(cls, method),
        })

    # Ensure RTTI-only classes (no demangled methods clustered) still appear.
    for cls in rtti:
        registry.setdefault(cls, {
            "name": cls, "methods": [], "vtable": None, "typeinfo": None,
            "typeinfo_name": None, "size": None, "bases": [], "instances": [],
            "confidence": "name-only",
        })

    for cls, rec in registry.items():
        syms = rtti.get(cls, {})
        rec["vtable"] = _sym_entry(syms.get("vtable"))
        rec["typeinfo"] = _sym_entry(syms.get("typeinfo"))
        rec["typeinfo_name"] = _sym_entry(syms.get("typeinfo_name"))
        if syms.get("vtable") or syms.get("typeinfo") or syms.get("typeinfo_name"):
            rec["confidence"] = "rtti"
        elif any(m["kind"] in ("ctor", "dtor") for m in rec["methods"]):
            rec["confidence"] = "ctor"
        else:
            rec["confidence"] = "name-only"

    if needle:
        registry = {k: v for k, v in registry.items() if needle in k.lower()}
    return registry


def _list_row(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": rec["name"],
        "method_count": len(rec["methods"]),
        "has_vtable": rec["vtable"] is not None,
        "size": rec["size"],
        "bases": [b.get("name") for b in rec.get("bases", [])],
        "confidence": rec["confidence"],
    }


def _class_list(
    ctx,
    selector: str | None,
    *,
    query: str | None = None,
    include_all: bool = False,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    bv = ctx._resolve_view(selector)
    registry = _build_class_registry(ctx, bv, query=query)
    rows = [
        _list_row(rec)
        for rec in registry.values()
        if include_all or rec["confidence"] in ("rtti", "ctor")
    ]
    rows.sort(key=lambda r: r["name"])
    total = len(rows)
    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return {"classes": rows, "total": total, "offset": offset, "include_all": include_all}


def _vtable_layout(ctx, bv, vtable_addr: int, *, max_slots: int = 64) -> dict[str, Any]:
    """Function slots of an Itanium vtable. Words [0] (offset-to-top) and [1]
    (typeinfo ptr) are header; slots start at +2*ptr_size. Reuses the
    Thumb-aware pointer-table reader."""
    ptr = ctx._pointer_size(bv)
    start = vtable_addr + 2 * ptr
    table = ctx._pointer_table_layout(bv, start, entries=max_slots, stride=ptr)
    slots: list[dict[str, Any]] = []
    for i, row in enumerate(table.get("entries") or []):
        target = row.get("target") or {}
        fn = target.get("function") if isinstance(target, dict) else None
        # A null/unmapped slot ends the vtable (next object / padding).
        if not row.get("readable") or target.get("status") not in ("function", "mapped"):
            break
        name = (fn or {}).get("name") if isinstance(fn, dict) else None
        slots.append({
            "index": i,
            "address": row.get("value"),
            "method": fn if isinstance(fn, dict) else None,
            "pure_virtual": name == "__cxa_pure_virtual",
            "unnamed": (isinstance(name, str) and name.startswith("sub_")) or (fn is None),
        })
    return {"address": hex(int(vtable_addr)), "slots": slots}


def _rtti_bases(ctx, bv, typeinfo_addr: int, *, kind_hint: str | None = None) -> list[dict[str, Any]]:
    """Base classes from an Itanium ``_ZTI`` object. ``kind_hint`` selects the
    layout: 'base' (no bases), 'si' (single), 'vmi' (multiple). When absent,
    infer structurally from the base-typeinfo pointers that resolve."""
    ptr = ctx._pointer_size(bv)
    name_field = typeinfo_addr + ptr          # word[1] = type-name ptr  # noqa: F841
    after_name = typeinfo_addr + 2 * ptr      # first layout-specific word

    def resolve(ti_addr: int, kind: str = "public") -> dict[str, Any]:
        return {
            "name": ctx._typeinfo_name_at(bv, ti_addr),
            "address": hex(int(ti_addr)),
            "kind": kind,
        }

    kind = kind_hint or _infer_rtti_kind(ctx, bv, typeinfo_addr, ptr)
    if kind == "base":
        return []
    if kind == "si":
        base = ctx._read_pointer_value(bv, after_name, size=ptr)
        return [resolve(base)] if base else []
    if kind == "vmi":
        # [flags:u32][base_count:u32] then base_count * (base-ti-ptr, off_flags)
        count = ctx._read_u32(bv, after_name + 4) or 0
        rec = after_name + 8
        out: list[dict[str, Any]] = []
        for _ in range(min(int(count), 64)):
            ti = ctx._read_pointer_value(bv, rec, size=ptr)
            off_flags = ctx._read_pointer_value(bv, rec + ptr, size=ptr) or 0
            rec += 2 * ptr
            if not ti:
                continue
            kind_s = "virtual" if (off_flags & 0x1) else "public"
            out.append(resolve(ti, kind_s))
        return out
    return []


def _infer_rtti_kind(ctx, bv, typeinfo_addr: int, ptr: int) -> str:
    """Best-effort layout inference when the __*_class_type_info selector
    symbol is unavailable: a resolvable single base ptr -> 'si'; a small
    plausible count followed by resolvable base ptrs -> 'vmi'; else 'base'."""
    after_name = typeinfo_addr + 2 * ptr
    candidate = ctx._read_pointer_value(bv, after_name, size=ptr)
    if candidate and ctx._typeinfo_name_at(bv, candidate):
        return "si"
    count = ctx._read_u32(bv, after_name + 4) or 0
    if 0 < count <= 16:
        first = ctx._read_pointer_value(bv, after_name + 8, size=ptr)
        if first and ctx._typeinfo_name_at(bv, first):
            return "vmi"
    return "base"


def _object_size(ctx, bv, record: dict[str, Any]) -> dict[str, Any] | None:
    """Object size with provenance. A defined BN type's width wins (authoritative
    when present); else the operator-new size at a construction site; else None
    (never fabricated). ``ctx._find_type`` raises on a miss, so the lookup is
    guarded."""
    try:
        found = ctx._find_type(bv, record["name"])
    except Exception:
        found = None
    if found is not None:
        _, type_obj = found
        width = int(getattr(type_obj, "width", 0) or 0)
        if width > 0:
            return {"value": hex(width), "source": "bn_type"}
    new = ctx._operator_new_size_at_ctor(bv, record)
    if new is not None:
        size, at = new
        return {"value": hex(int(size)), "source": "operator_new", "at": hex(int(at))}
    return None


def _instances(ctx, bv, record: dict[str, Any], *, cap: int = 128) -> dict[str, Any]:
    """Best-effort: where objects of this class are constructed and which
    globals hold one. Empty (not an error) when nothing is found."""
    sites = ctx._ctor_construction_sites(bv, record)[:cap]
    stored = ctx._global_vtable_stores(bv, record)[:cap] if record.get("vtable") else []
    return {"construction_sites": sites, "stored_globals": stored}


def _resolve_class_names(registry: dict[str, dict], name: str) -> list[str]:
    """Exact match, else all classes whose leaf component equals *name*."""
    if name in registry:
        return [name]
    leaf = _last_component(name) if "::" in name else name
    return sorted(k for k in registry if _last_component(k) == leaf)


def _enrich(ctx, bv, rec: dict[str, Any]) -> dict[str, Any]:
    if rec.get("vtable"):
        rec["vtable"] = ctx._vtable_layout_for(bv, int(rec["vtable"]["address"], 16)) or rec["vtable"]
    rec["size"] = ctx._object_size_for(bv, rec)
    rec["bases"] = ctx._bases_for(bv, rec)
    rec["instances"] = ctx._instances_for(bv, rec)
    return rec


def _class_show(ctx, selector: str | None, name: str) -> dict[str, Any]:
    bv = ctx._resolve_view(selector)
    registry = _build_class_registry(ctx, bv)
    matches = _resolve_class_names(registry, name)
    if not matches:
        raise OperationFailure(
            "unknown_class",
            f"No class named {name!r}. Run `bn class list` (add --all for "
            f"name-only clusters) to discover available classes.",
        )
    enriched = [_enrich(ctx, bv, registry[m]) for m in matches]
    if len(enriched) == 1:
        return enriched[0]
    return {"ambiguous": True, "query": name, "matches": enriched}
