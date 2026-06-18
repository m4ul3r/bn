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
