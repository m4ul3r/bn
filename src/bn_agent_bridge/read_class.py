"""C++ object-model lens (#205): class registry, vtable layout, RTTI bases,
object size, and instance tracking, correlated from data Binary Ninja already
recovers (demangled symbols, RTTI data symbols, operator-new sizes).

All functions are read-only and take the BridgeContext seam (``ctx``); this
module never imports ``bridge`` or ``mutation_engine``."""
from __future__ import annotations

import re
from typing import Any

from . import il_format
from ._shared import OperationFailure


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


# RTTI data symbols are identified by the DEMANGLED marker on the symbol's
# short_name -- group(1) = kind marker, group(2) = class name. The marker
# punctuation varies by BN version/platform: spaces ("vtable for X") or
# underscores ("_vtable_for_X", "typeinfo_for_X", "typeinfo_name_for_X"), with an
# optional leading underscore. Crucially, do NOT gate on the mangled raw-name
# prefix (`_ZTV`/`_ZTI`/`_ZTS`): on real targets BN sets a typeinfo symbol's
# raw_name to the demangled form (`_typeinfo_for_X`) and creates no `_ZTI...`
# symbol, so a raw-prefix gate silently drops typeinfo (and all RTTI bases).
# Order matters: "typeinfo name" must precede "typeinfo" in the alternation.
_RTTI_MARKER_RE = re.compile(r"^_?(vtable|typeinfo[ _]name|typeinfo)[ _]for[ _](.+)$")


def _rtti_kind_and_class(sym) -> tuple[str | None, str | None]:
    """(kind, class_name) for an RTTI data symbol -- kind is ``vtable`` /
    ``typeinfo`` / ``typeinfo_name`` -- else (None, None). Identified by the
    demangled marker on short_name, independent of the raw-name spelling."""
    short = str(getattr(sym, "short_name", "") or "")
    m = _RTTI_MARKER_RE.match(short)
    if not m:
        return None, None
    marker = m.group(1)
    kind = ("typeinfo_name" if "name" in marker
            else "typeinfo" if marker.startswith("typeinfo")
            else "vtable")
    return kind, m.group(2).strip()


def _class_of_rtti_symbol(sym) -> str | None:
    """Class name for an RTTI symbol (demangled marker stripped), else None."""
    return _rtti_kind_and_class(sym)[1]


def _rtti_symbol_maps(bv) -> dict[str, dict[str, Any]]:
    """{class_name: {"vtable": sym, "typeinfo": sym, "typeinfo_name": sym}}."""
    maps: dict[str, dict[str, Any]] = {}
    for sym in bv.get_symbols():
        kind, cls = _rtti_kind_and_class(sym)
        if kind and cls:
            maps.setdefault(cls, {}).setdefault(kind, sym)
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


# Standard-library / ABI-runtime top-level namespaces folded out by --no-stl.
_LIBRARY_NAMESPACES = frozenset({"std", "__gnu_cxx", "__cxxabiv1"})


def _first_toplevel_component(head: str) -> str:
    """The component before the first top-level ``::`` (angle-depth aware), or
    *head* itself when there is none -- i.e. the outermost namespace."""
    angle = 0
    i = 0
    n = len(head)
    while i < n:
        ch = head[i]
        if ch == "<":
            angle += 1
        elif ch == ">" and angle:
            angle -= 1
        elif ch == ":" and angle == 0 and i + 1 < n and head[i + 1] == ":":
            return head[:i]
        i += 1
    return head


def _is_library_class(name: str) -> bool:
    """True for a C++ standard-library / ABI-runtime class -- `std::`,
    `__gnu_cxx::`, `__cxxabiv1::` -- or a reserved-identifier implementation
    internal at top level (`__detail`, `_Hashtable`, `_Sp_counted_ptr_inplace`).
    Used by ``--no-stl`` to fold library noise out of the class listing so the
    domain classes surface."""
    first = _first_toplevel_component(_strip_signature(name))
    if first in _LIBRARY_NAMESPACES:
        return True
    base = first.split("<", 1)[0]  # drop any template args on the component
    return base.startswith("__") or (len(base) >= 2 and base[0] == "_" and base[1].isupper())


# Vendored libraries `--no-vendor` folds out -- a vendored/in-tree copy is library
# noise the same way std is, but isn't STL, so it gets its own opt-in flag (#309).
_VENDOR_NAMESPACES = frozenset({"boost"})


def _is_construction_vtable_artifact(name: str) -> bool:
    """A construction-vtable RTTI artifact, e.g. ``Derived{for `Base'}`` -- emitted
    for every class with a base, method_count 0. Not a real class; it inflates the
    class lens ~N-fold and buries the domain surface (#309)."""
    return "{for " in name


_THUNK_PREFIX_RE = re.compile(r"^_?(non-virtual|virtual)[ _]thunk[ _]to[ _]", re.IGNORECASE)


def _is_thunk_artifact(name: str) -> bool:
    """A thunk symbol (``non-virtual thunk to X`` / ``virtual thunk to X``, which
    BN may spell with underscores) mis-parsed into a fake class/namespace -- a
    compiler-generated forwarding stub, never a type (#309).

    Anchored to the leading thunk PREFIX -- a substring match would wrongly drop a
    real class/namespace that merely contains ``thunk_to`` (e.g. ``Thunk_to_handler``
    or ``thunk_to_ns::X``), and since thunks are suppressed even under ``--all``
    that class would be unrecoverable (#309 review)."""
    return bool(_THUNK_PREFIX_RE.match(name))


def _is_vendor_class(name: str) -> bool:
    """A class from a vendored/in-tree library `--no-vendor` folds out (boost),
    the same idea as `--no-stl` for std/ABI (#309)."""
    first = _first_toplevel_component(_strip_signature(name))
    base = first.split("<", 1)[0]
    return base in _VENDOR_NAMESPACES


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
    no_stl: bool = False,
    no_vendor: bool = False,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    bv = ctx._resolve_view(selector)
    registry = _build_class_registry(ctx, bv, query=query)
    candidates = []
    library_suppressed = 0
    vendor_suppressed = 0
    construction_vtables_suppressed = 0
    thunks_suppressed = 0
    for rec in registry.values():
        name = rec["name"]
        # A thunk is never a type -- drop it unconditionally (even under --all),
        # so it isn't surfaced as a class/namespace (#309).
        if _is_thunk_artifact(name):
            thunks_suppressed += 1
            continue
        # Construction-vtable artifacts (`X{for `Base'}`) ARE real RTTI objects,
        # just not classes -- hide by default, reveal under --all (#309).
        if not include_all and _is_construction_vtable_artifact(name):
            construction_vtables_suppressed += 1
            continue
        if not (include_all or rec["confidence"] in ("rtti", "ctor")):
            continue
        if no_stl and _is_library_class(name):
            library_suppressed += 1
            continue
        if no_vendor and _is_vendor_class(name):
            vendor_suppressed += 1
            continue
        candidates.append(rec)
    candidates.sort(key=lambda r: r["name"])
    # `total` counts candidates AFTER the confidence + --no-stl filters but before
    # paging, so it reflects what this query actually surfaced.
    total = len(candidates)
    page = candidates[offset:] if offset else candidates
    if limit is not None:
        page = page[:limit]
    # Populate RTTI bases for the RETURNED PAGE only: base decode is cheap (a few
    # typeinfo reads per class) and bounded to the page, so a single `class list`
    # recovers the inheritance graph without N `class show` calls. Vtable layout
    # and object size stay show-only -- they are far costlier per class. (#205 review)
    rows = []
    for rec in page:
        try:
            rec["bases"] = ctx._bases_for(bv, rec)
        except Exception:
            rec["bases"] = []
        rows.append(_list_row(rec))
    returned = len(rows)
    return {
        "kind": "classes",
        "items": rows,
        "total": total,
        "offset": offset,
        "limit": limit,
        "returned": returned,
        "has_more": offset + returned < total,
        "include_all": include_all,
        "no_stl": no_stl,
        "no_vendor": no_vendor,
        "library_suppressed": library_suppressed,
        "vendor_suppressed": vendor_suppressed,
        "construction_vtables_suppressed": construction_vtables_suppressed,
        "thunks_suppressed": thunks_suppressed,
    }


def _slot_is_code(target: dict[str, Any]) -> bool:
    """A real vtable slot is a CODE pointer: a resolved function, or an address
    the repo's classifier deems code (``context.kind == "code"`` -- function
    membership or a Code-semantics section).

    It deliberately does NOT accept "any mapped pointer in an executable
    segment": firmware ELFs routinely map ``.rodata`` into the same r-x load
    segment as ``.text``, so the executable bit is not evidence of code (see
    ``seam._address_is_code`` / the ``kind`` classification in
    ``_address_context``). Accepting it would render data/string pointers in an
    r-x mapping as fake unnamed virtual methods. A non-code pointer ends the
    scan rather than fabricating slots (#205 review)."""
    if target.get("status") == "function":
        return True
    return (target.get("context") or {}).get("kind") == "code"


def _vtable_layout(ctx, bv, vtable_addr: int, *, max_slots: int = 64) -> dict[str, Any]:
    """Function slots of an Itanium vtable. Words [0] (offset-to-top) and [1]
    (typeinfo ptr) are header; slots start at +2*ptr_size. Reuses the
    Thumb-aware pointer-table reader. Only CODE targets count as slots."""
    ptr = ctx._pointer_size(bv)
    # Itanium invariant: word[1] (vtable_addr + ptr) points to the class's
    # typeinfo. If it doesn't resolve to a typeinfo symbol, this address is NOT
    # the start of a real local vtable OBJECT -- it's an import/GOT pointer slot
    # (the vtable is defined in another module) or a PIE slot relocated to zero.
    # Decoding +2*ptr there would render adjacent GOT/data as fake slots
    # (#205 review), so report no slots and let the caller note it.
    ti_ptr = ctx._read_pointer_value(bv, vtable_addr + ptr, size=ptr)
    if not ti_ptr or ctx._typeinfo_name_at(bv, ti_ptr) is None:
        return {"address": hex(int(vtable_addr)), "slots": []}
    start = vtable_addr + 2 * ptr
    table = ctx._pointer_table_layout(bv, start, entries=max_slots, stride=ptr)
    slots: list[dict[str, Any]] = []
    # #303: the pointer-table reader (`_pointer_table_for_view`) returns the
    # canonical #275 envelope, whose rows live under `items`. This loop read the
    # pre-#275 `entries` key, so every vtable resolved to ZERO slots and
    # `class show` declared a recoverable dispatch table unrecoverable. Read
    # `items` (with an `entries` fallback for any legacy producer / test fake).
    for i, row in enumerate(table.get("items") or table.get("entries") or []):
        target = row.get("target") or {}
        fn = target.get("function") if isinstance(target, dict) else None
        # A null/unmapped/non-code slot ends the vtable (next object / padding /
        # a misidentified table over data).
        if not row.get("readable") or not _slot_is_code(target):
            break
        name = (fn or {}).get("name") if isinstance(fn, dict) else None
        method = None
        if isinstance(fn, dict):
            # The pointer-table function dict carries the MANGLED fn.name; add the
            # demangled display name (symbol short_name) so vtable slots read like
            # the methods list, not raw `_ZN...`. Mangled `name` is kept. (#205)
            method = {**fn, "display_name": _demangled_slot_name(bv, fn)}
        slots.append({
            "index": i,
            "address": row.get("value"),
            "method": method,
            "pure_virtual": name == "__cxa_pure_virtual",
            "unnamed": (isinstance(name, str) and name.startswith("sub_")) or (fn is None),
        })
    return {"address": hex(int(vtable_addr)), "slots": slots}


def _demangled_slot_name(bv, fn: dict[str, Any]) -> str | None:
    """Demangled display name for a vtable slot's function (the symbol's
    short_name, via il_format._display_name), falling back to the mangled
    ``fn.name`` when the function can't be resolved on *bv* (e.g. in tests)."""
    mangled = fn.get("name")
    addr = fn.get("address")
    getf = getattr(bv, "get_function_at", None)
    if callable(getf) and addr:
        try:
            func = getf(int(addr, 16))
        except Exception:
            func = None
        if func is not None:
            disp = il_format._display_name(func)
            if disp:
                return disp
    return mangled


def _rtti_bases(ctx, bv, typeinfo_addr: int, *, kind_hint: str | None = None) -> list[dict[str, Any]]:
    """Base classes from an Itanium ``_ZTI`` object. ``kind_hint`` selects the
    layout: 'base' (no bases), 'si' (single), 'vmi' (multiple). When absent,
    infer structurally from the base-typeinfo pointers that resolve."""
    ptr = ctx._pointer_size(bv)
    # Itanium layout: word[0] = vptr, word[1] = type-name ptr (skipped),
    # word[2+] (``after_name``) = the layout-specific fields.
    after_name = typeinfo_addr + 2 * ptr

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
            # __offset_flags carries two independent bits: 0x1 = virtual base,
            # 0x2 = public base. Both clear => private non-virtual. Report
            # access and virtual-ness separately so a private base is not
            # mislabeled "public".
            kind_parts = ["public" if (off_flags & 0x2) else "private"]
            if off_flags & 0x1:
                kind_parts.append("virtual")
            out.append(resolve(ti, " ".join(kind_parts)))
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


def _query_leaf(name: str) -> str:
    """Last TOP-LEVEL ``::`` component, template arguments PRESERVED. Unlike
    ``_last_component`` (which drops ``<...>`` for ctor/dtor-name comparison),
    this keeps the template args so a specific query like ``Vec<std::string>``
    is not collapsed to ``Vec`` and matched against unrelated specializations.
    The scope check is depth-aware, so ``::`` inside ``<...>`` is not a scope
    separator (#205 review)."""
    head = _strip_signature(name)
    idx = _last_toplevel_scope(head)
    return head[idx + 2:] if idx is not None else head


def _resolve_class_names(registry: dict[str, dict], name: str) -> list[str]:
    """Exact match, else all classes whose top-level leaf equals *name*'s leaf
    (template args preserved), so an unqualified query matches the same class
    across namespaces without conflating template specializations."""
    if name in registry:
        return [name]
    leaf = _query_leaf(name)
    return sorted(k for k in registry if _query_leaf(k) == leaf)


def _enrich(ctx, bv, rec: dict[str, Any]) -> dict[str, Any]:
    if rec.get("vtable"):
        rec["vtable"] = ctx._vtable_layout_for(bv, int(rec["vtable"]["address"], 16)) or rec["vtable"]
        # #412 (codex Finding 1): a multiple-inheritance class commonly keeps its
        # PRIMARY `_ZTV` symbol while the secondary base-subobject vtables are
        # unsymbolized. The symbolized primary above doesn't surface them, so back-
        # walk the typeinfo for the secondaries too. Keep the symbolized primary as
        # `rec["vtable"]`; attach only the recovered tables that are a DIFFERENT
        # address (the backwalk re-finds the primary via its own typeinfo ref).
        if rec.get("typeinfo"):
            recovered = _recover_vtables_from_typeinfo(
                ctx, bv, int(rec["typeinfo"]["address"], 16))
            if recovered:
                primary_addr = (rec["vtable"] or {}).get("address")
                secondaries = [s for s in recovered["secondary"]
                               if s.get("address") != primary_addr]
                if secondaries:
                    rec["secondary_vtables"] = secondaries
                    rec.setdefault("notes", []).append(
                        "secondary (multiple-inheritance) vtables recovered via "
                        "typeinfo backwalk (no _ZTV symbol for the secondary bases)")
    elif rec.get("typeinfo"):
        # #354: STRIPPED binary -- the `_ZTV` DataSymbol is gone so the registry has
        # no vtable address, but typeinfo survived. Backwalk typeinfo->vtable.
        recovered = _recover_vtables_from_typeinfo(ctx, bv, int(rec["typeinfo"]["address"], 16))
        if recovered:
            rec["vtable"] = recovered["primary"]
            if recovered["secondary"]:
                rec["secondary_vtables"] = recovered["secondary"]   # #412
            rec.setdefault("notes", []).append(
                "vtable recovered via typeinfo backwalk (no _ZTV symbol -- stripped binary)")
    rec["size"] = ctx._object_size_for(bv, rec)
    rec["bases"] = ctx._bases_for(bv, rec)
    rec["instances"] = ctx._instances_for(bv, rec)
    return rec


def _as_signed(value: int | None, ptr: int) -> int:
    """Interpret an unsigned ptr-sized word as a signed offset-to-top."""
    if value is None:
        return 0
    bits = ptr * 8
    return value - (1 << bits) if value >= (1 << (bits - 1)) else value


def _recover_vtables_from_typeinfo(ctx, bv, typeinfo_addr: int) -> dict[str, Any] | None:
    """#354/#412: in a STRIPPED binary the ``_ZTV`` DataSymbol is gone, so the class
    registry carries no vtable address -- but the typeinfo symbol usually survives.
    An Itanium vtable's word[1] holds the class typeinfo pointer, so a DATA xref to
    the typeinfo addr lands at ``vtable_addr + ptr_size``. Backwalk each such ref,
    validate the candidate as a real vtable via ``_vtable_layout_for`` (word[1] ==
    typeinfo, slots are CODE -- which filters typeinfo base-pointer refs from other
    classes' RTTI), and classify primary (offset-to-top 0) vs secondary (#412,
    non-zero word[0]). Returns ``{"primary": layout, "secondary": [layout, ...]}``
    or None when nothing resolves."""
    getter = getattr(bv, "get_data_refs", None)
    if not callable(getter):
        return None
    ptr = ctx._pointer_size(bv)
    groups: list[tuple[int, dict[str, Any]]] = []
    seen: set[int] = set()
    for ref in getter(int(typeinfo_addr)):
        vt_addr = int(ref) - ptr
        if vt_addr < 0 or vt_addr in seen:
            continue
        seen.add(vt_addr)
        layout = ctx._vtable_layout_for(bv, vt_addr)
        if not layout or not layout.get("slots"):
            continue
        ott = _as_signed(ctx._read_pointer_value(bv, vt_addr, size=ptr), ptr)
        layout["offset_to_top"] = ott
        layout["typeinfo_backwalk"] = True   # provenance: recovered without a _ZTV symbol
        groups.append((ott, layout))
    if not groups:
        return None
    # Classify by offset-to-top VALUE, not by sort rank. The primary subobject
    # sits at offset-to-top 0; a real secondary base subobject is at a NON-ZERO
    # (negative) offset-to-top. A second ref that resolves to a distinct vtable
    # whose offset-to-top is ALSO 0 is a construction-vtable / RTTI artifact, not
    # a secondary -- dropping it avoids emitting a bogus secondary (#412 review).
    zeros = [layout for ott, layout in groups if ott == 0]
    if zeros:
        primary = zeros[0]                            # several ==0 -> take the first
    else:
        primary = min(groups, key=lambda g: abs(g[0]))[1]   # none ==0 -> nearest 0
    secondary = [layout for ott, layout in groups if ott != 0]
    return {"primary": primary, "secondary": secondary}


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
