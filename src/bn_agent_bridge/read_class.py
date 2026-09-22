"""C++ object-model lens (#205): class registry, vtable layout, RTTI bases,
object size, and instance tracking, correlated from data Binary Ninja already
recovers (demangled symbols, RTTI data symbols, operator-new sizes).

All functions are read-only and take the BridgeContext seam (``ctx``); this
module never imports ``bridge`` or ``mutation_engine``."""
from __future__ import annotations

import difflib
import functools
import re
from typing import Any, NamedTuple

from . import il_format
from ._shared import OperationFailure, _validate_count
from .read_listing import _analysis_state_fields
from .read_types import _follow_typedef
from .seam import _view_memo


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


# BN classifies a GOT/import/external alias with one of these symbol types; a real
# local definition is a DataSymbol/FunctionSymbol/etc. (#529). An alias carries the
# same demangled RTTI name as the local definition, so if get_symbols() yields the
# alias first it would win the (class, kind) slot and class recovery would decode the
# GOT/extern stub as the vtable object -- reporting missing/empty slots despite a real
# local vtable. Prefer the local definition regardless of iteration order.
_ALIAS_SYMBOL_TYPES = (
    "ImportAddressSymbol",
    "ImportedFunctionSymbol",
    "ImportedDataSymbol",
    "ExternalSymbol",
)


def _is_alias_symbol(sym) -> bool:
    """True if *sym* is a GOT/import/external alias rather than a local definition.

    A real BN ``SymbolType`` is an ``IntEnum`` whose ``str()`` renders as the numeric
    value (``str(SymbolType.ExternalSymbol) == "5"``), so the member NAME must come
    from ``.name`` -- matching on ``str(sym.type)`` would never fire on a live BV.
    Falls back to ``str()`` for plain-string ``sym.type`` on hand-built test
    symbols in ``tests/test_read_class.py`` (e.g. ``"SymbolType.ExternalSymbol"``)."""
    st = getattr(sym, "type", None)
    if st is None:
        return False
    tname = getattr(st, "name", None) or str(st)
    tname = tname.rsplit(".", 1)[-1]   # hand-built test strings may have a "SymbolType." prefix
    return tname in _ALIAS_SYMBOL_TYPES


def _rtti_symbol_maps(bv) -> dict[str, dict[str, Any]]:
    """{class_name: {"vtable": sym, "typeinfo": sym, "typeinfo_name": sym}}.

    When several symbols share the same (class, kind) -- e.g. a GOT/import alias and
    the real local definition -- prefer the LOCAL definition regardless of the order
    ``get_symbols()`` yields them (#529). A local never gets overwritten by an alias,
    and an alias is replaced the moment a local for the same slot is seen."""
    maps: dict[str, dict[str, Any]] = {}
    for sym in bv.get_symbols():
        kind, cls = _rtti_kind_and_class(sym)
        if not (kind and cls):
            continue
        slot = maps.setdefault(cls, {})
        existing = slot.get(kind)
        if existing is None:
            slot[kind] = sym
        elif _is_alias_symbol(existing) and not _is_alias_symbol(sym):
            slot[kind] = sym
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


# #622 criterion (d) -- "class list / class show reuse a per-view registry (or
# cheaper show path) rather than demangling every function on every call" -- is
# DELIVERED by the per-view registry memo below, and the memo that follows it
# stays for the rebuilds that remain.
#
# The registry: `_scan_class_registry` is wrapped in `seam._view_memo(bv,
# "class_registry", ...)`, so a view's registry is built ONCE, keyed on that
# view's generation counter and dropped when BN reports any name-bearing change
# through its own notifications (seam's `_ViewChangeState`) -- a rename or a late
# analysis pass that adds named symbols therefore cannot be served stale, which
# the maintainer's comment on #622 calls worse than the cost being saved. Stated
# plainly: registry staleness is caught by BN's symbol/function notifications,
# and a view whose notification surface is unavailable is NEVER cached (every
# call rescans, exactly as before). `_build_class_registry` then hands out a COPY
# of the cached registry per call (filtered by `query=`), because `_class_list`
# writes `rec["bases"]` for its page rows and `_class_show` writes the drill-down
# keys -- a shared record would make one command's output depend on the other's
# call order.
#
# The classify memo below: it remains because it removes the per-name demangle +
# qualified-method split from a REBUILD, not just from a repeat call -- measured
# on a real C++ target: 5717 splits on the first registry build, 0 on the next.
# It is pinned by `test_class_name_classification_is_memoised_across_rebuilds`:
# its counter wraps `_split_qualified_method` and observes one split per function
# on the synthetic view's first build, then ZERO on the rebuild -- so dropping the
# memo fails a test instead of silently regressing.
#
# The partial `class show` build that materialised only the queried class stays
# REMOVED (a self-inflicted pessimisation): it measured SLOWER than the full
# build (median 0.190s filtered vs 0.131s full) on the ~5.8k-function C++ target
# of the round-1 review, and the registry memo now removes the rebuild that
# measurement was chasing. The round-1 review's uncached `class list` figures
# (~0.31-0.42s vs base ~0.33-0.42s on ~5.8k functions, ~1.56-1.60s vs base
# ~1.39-1.55s on ~16k) no longer describe a repeat call: the second call reuses
# the registry and enumerates nothing.
#
# The classify memo itself: every registry build classifies each function's name
# -- demangle, split the qualified method, and name the ctor/dtor/method kind.
# That classification is a PURE function of the two name spellings
# `il_format._display_name` consumes, so it is memoised here at the call site (the
# demangle itself lives in `il_format`, outside this module's scope). A rename
# changes those spellings, and therefore the key, so a stale entry for a renamed
# function is unreachable -- no invalidation hook is needed for the memo.
_CLASSIFY_CACHE_MAX = 65536


@functools.lru_cache(maxsize=_CLASSIFY_CACHE_MAX)
def _classify_names(short_name: str, name: str) -> tuple[str, str | None, str | None]:
    """``(demangled, class, kind)`` for a function's name spellings.

    ``short_name`` is the symbol's demangled short name when BN has one (its
    ``fn.name`` stays mangled for C++), else "". ``kind`` is
    ``ctor``/``dtor``/``method``, or None for a name with no scope qualifier --
    exactly ``_split_qualified_method`` + ``_method_kind`` over
    ``il_format._display_name``."""
    demangled = short_name or name
    cls, method = _split_qualified_method(demangled)
    return demangled, cls, (None if cls is None else _method_kind(cls, method))


def _classify_function(fn) -> tuple[str, str | None, str | None]:
    """Memoised :func:`_classify_names` for *fn*, read off the live object."""
    sym = getattr(fn, "symbol", None)
    short = getattr(sym, "short_name", None) if sym is not None else None
    return _classify_names(str(short) if short else "", str(getattr(fn, "name", "") or ""))


def _scan_class_registry(bv) -> dict[str, dict[str, Any]]:
    """ONE pass over the view -> {class_name: ClassRecord}. Methods, RTTI symbols,
    confidence. Per-class drill-downs (vtable layout, size, bases, instances) are
    added by ``_class_show`` only for the requested class (too costly for every
    class).

    This is the memoised half: it reads only ``bv``, so it is cached per view on
    the view's generation counter by :func:`_build_class_registry`. Callers must
    never mutate what it returns -- they get their own copy."""
    rtti = _rtti_symbol_maps(bv)
    registry: dict[str, dict[str, Any]] = {}

    for fn in bv.functions:
        demangled, cls, kind = _classify_function(fn)
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
            "kind": kind,
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

    return registry


def _build_class_registry(ctx, bv, *, query: str | None = None) -> dict[str, dict[str, Any]]:
    """The view's class registry, filtered by *query*, as a fresh COPY per call.

    The scan is memoised per view (:func:`_view_memo`), keyed on the view's
    generation counter and invalidated by BN's own symbol/function notifications
    -- a rename changes which classes exist, so the registry is rebuilt after any
    such change, and a view with no notification surface is never cached (every
    call rescans, exactly as before). ``ctx`` is unused today (the scan reads only
    ``bv``), kept for the callers' uniform ``(ctx, bv)`` seam shape.

    Records are copied on the way out because the callers WRITE to them:
    ``_class_list`` sets ``rec["bases"]`` for its page rows and ``_class_show``
    sets the drill-down keys (``vtable`` / ``size`` / ``bases`` / ``instances`` /
    ``notes``). Handing out the cached records would make one command's output
    depend on the other's call order.

    The copy reaches every container the record owns -- the lists AND the dicts
    inside them -- so nothing a caller can touch is still the memo's own state. A
    cache that hands out its interior is one in-place mutation away from serving a
    poisoned record to every later call on that view, which no in-repo caller does
    today and none should have to know not to do. Measured at ~1 ms for a
    ~6k-method registry, against the ~130 ms rebuild it protects."""
    registry = _view_memo(bv, "class_registry", lambda: _scan_class_registry(bv))
    needle = query.lower() if query else None
    return {
        name: {
            **rec,
            "methods": [dict(method) for method in rec["methods"]],
            "bases": list(rec["bases"]),
            "instances": list(rec["instances"]),
            "vtable": dict(rec["vtable"]) if isinstance(rec["vtable"], dict) else rec["vtable"],
            "typeinfo": dict(rec["typeinfo"]) if isinstance(rec["typeinfo"], dict) else rec["typeinfo"],
            "typeinfo_name": (dict(rec["typeinfo_name"])
                              if isinstance(rec["typeinfo_name"], dict) else rec["typeinfo_name"]),
        }
        for name, rec in registry.items()
        if needle is None or needle in name.lower()
    }


# #675.2: the DECLARED half of the lens. `class list` / `class show` cluster
# classes from symbols and RTTI, so a class the user DECLARED -- `bn types
# declare 'class Widget { virtual void draw(); };'` -- is invisible to that half:
# it has no `_ZTV`/`_ZTI` symbol and no demangled method symbol, so `class show
# Widget` answered "No class named 'Widget'" while `types --query Widget` found
# it. The records below are what the lens falls back to.
#
# They are read LIVE, never memoised, and deliberately NOT merged into the
# memoised registry. The measured reason is REDEFINITION, not first definition:
# a per-name cache of a declared type's facts answers the OLD width after the
# type is re-declared -- measured live on BN 6.1 through the real CLI, where
# `class show` reported `size 0x10` while the view's type was already `0x18`.
# Reading this half fresh per call removes that staleness whatever the
# registry's invalidation covers.
#
# An earlier version of this comment justified the live read by claiming a
# `types declare` (`define_user_type`) fires no symbol/function notification and
# therefore cannot invalidate the memo (#622's cache, #675's triage comment).
# That was MEASURED FALSE: `define_user_type` moves the per-view generation
# counter (observed 3 -> 4 in-process, and 0 -> 1 -> 2 across two CLI declares),
# so a declare DOES invalidate the registry memo on this build. Kept as a note
# rather than deleted: the live read is defensive with respect to that
# question, and correct for the redefinition case either way.
_DECLARED_CONFIDENCE = "declared-only"
_DECLARED_TYPE_NOTE = (
    "declared type -- no RTTI class, demangled methods or construction sites for "
    "this name in this view; the empty vtable/methods/instances are RTTI evidence "
    "that is absent, NOT evidence this class has none"
)


def _no_instances() -> dict[str, Any]:
    """The envelope :func:`_instances` returns when it found nothing.

    The class card reads ``instances`` as a CONTAINER, so the registry's
    pre-``_enrich`` ``[]`` would be disclosed as a malformed field on every
    declared card (#619). Built fresh per record: a shared dict would let one
    caller's append reach every other record."""
    return {
        "construction_sites": [],
        "stored_globals": [],
        "construction_sites_total": 0,
        "construction_sites_truncated": False,
        "stored_globals_total": 0,
        "stored_globals_truncated": False,
    }


# The value `declared_suppressed` carries when the view's type table could not be
# read: a STATED unknown, which `_stated_count` renders as `?`, rather than a `0`
# indistinguishable from a real "no declared classes here" (#619, #907 review).
_DECLARED_UNREADABLE = "unreadable"


# #675.2 / #907 review: WHICH types the declared half answers for. Two axes, and
# both were measured wrong before this block existed.
#
# KIND. `bv.types` holds enums, scalars, pointers, arrays, function prototypes
# and typedef aliases alongside structures, so an unfiltered enumeration answered
# `class show Color` (an enum) with `class Color (size 0x4) [declared-only]`, the
# note asserting its empty vtable is "RTTI evidence that is absent" about a type
# that can never carry one. The admitted kind is BN's `StructureTypeClass`,
# exactly C++'s three class-keys (`class`, `struct`, `union`).
#
# PROVENANCE. `bv.types` is not the set the user declared either -- it is every
# type BN has, including the ones IT imported. Measured on a stock system ELF
# with zero user declarations: 53 types, 25 of them structure-kind (the ELF
# format structs, `FILE`, the libc `_IO_*` family), so the lens rendered
# `classes: 0 shown of 0 (hidden: 25 declared class types (--all to show))`
# directly above its own note saying the target has no C++ type evidence at all,
# and `class show <an ELF format struct>` answered a class card. On a C++ target
# it additionally listed BN's own generated `<Class>::VTable` structs as peer
# "classes" of the class they belong to -- the artifact family #309/#481 exist to
# suppress. The population is therefore the USER type container: the set
# `types declare` (`define_user_type`) writes to, which is what #675 item 2 asks
# about and what this module's comments have always claimed.
_STRUCTURE_TYPE_CLASS = 4       # binaryninja.TypeClass.StructureTypeClass

# `binaryninja.NamedTypeReferenceClass` values that NAME a C++ class type:
# ClassNamedTypeClass, StructNamedTypeClass, UnionNamedTypeClass. An alias BN
# cannot resolve still states the kind it references, and that is not a rare
# shape: measured on BN 6.1, `typedef struct { ... } T;` DOES register its
# anonymous body (as `_T`) so the plain spelling resolves, but the NAMESPACED
# one -- `namespace n { typedef struct { ... } Q; }` -- reads `target(bv)` of
# None while the reference itself states StructNamedTypeClass, as does any
# alias whose body is defined without it. Refusing on an unresolvable target
# alone would therefore drop a common way a declared class reaches a view,
# while `typedef enum { ... } M;` (EnumNamedTypeClass, target also None) must
# still be refused. (An earlier version of this comment claimed the anonymous
# body is NEVER registered; that was refuted live -- #907 review rounds 3/4.)
_CLASS_NAMED_TYPE_CLASSES = frozenset({2, 3, 4})
_CLASS_NAMED_TYPE_PREFIXES = ("Class", "Struct", "Union")
# `EnumNamedTypeClass`. The remaining spellings -- `Unknown` (0) and `Typedef`
# (1) -- name neither a class nor a non-class: a typedef-kinded reference may
# well resolve to a structure, so reading it as "not a class" is a conclusion
# the handle does not support (#907 review round 3).
_NON_CLASS_NAMED_TYPE_CLASSES = frozenset({5})
_NON_CLASS_NAMED_TYPE_PREFIXES = ("Enum",)

# Returned for a declaration whose KIND could not be established at all -- the
# read raised, or the handle states nothing that classifies it. Distinct from
# `None` (read fine, positively not a class) because the two must be answered
# differently: an unknown kind may not be counted and may not be reported as an
# absent class (#907 review round 3).
_UNREADABLE_DECLARATION = object()


def _named_class_kind(type_obj) -> bool | None:
    """``True`` if *type_obj* is a NamedTypeReference naming a class type,
    ``False`` if it names something that is positively NOT one, and ``None``
    when the handle does not say.

    Prefix-matched, not substring-matched: every ``NamedTypeReferenceClass``
    spelling ends in ``NamedTypeClass``, so ``"Class" in name`` is true of
    ``EnumNamedTypeClass`` too."""
    ntc = getattr(type_obj, "named_type_class", None)
    if ntc is None:
        return None
    try:
        value = int(ntc)
    except (TypeError, ValueError):
        spelling = str(getattr(ntc, "name", None) or ntc)
        if spelling.startswith(_CLASS_NAMED_TYPE_PREFIXES):
            return True
        return False if spelling.startswith(_NON_CLASS_NAMED_TYPE_PREFIXES) else None
    if value in _CLASS_NAMED_TYPE_CLASSES:
        return True
    return False if value in _NON_CLASS_NAMED_TYPE_CLASSES else None


def _class_type_target(bv, name: str, type_obj):
    """The handle carrying a declaration's facts when the declaration is a C++
    class type -- the structure itself, the structure its alias chain reaches, or
    an unresolvable reference that still NAMES a class. ``None`` when the
    declaration was READ and is positively not a class, and
    :data:`_UNREADABLE_DECLARATION` when its kind could not be established at
    all.

    Kind test duck-typed over BN's ``TypeClass`` IntEnum and the plain string the
    unit fakes carry, the shape :func:`read_types._is_named_type_ref` uses.

    The resolved TARGET is preferred because that is where a declaration's facts
    live: a NamedTypeReference states no members and (for a resolvable alias) a
    width that is not its own, so a card built from the alias rendered
    ``kind: named_type_ref ... size=0x0`` for a 0x30-wide class and could not show
    the class the user asked about (#907 review). ``read_types``' struct reader
    resolves the same way, for the same reason (#674) -- one convention for which
    handle carries a declaration's facts. The record keeps the DECLARED name; the
    entry's own ``decl`` discloses the underlying body.

    Never admits a class on a read that failed -- but the THIRD answer is the
    point. Collapsing "I read this and it is an enum" into "I could not read
    this" let one raising handle vanish from the enumeration, so the listing
    stated a measured count that silently excluded it and `class show` of that
    very name answered a confident `No class named`: a fabricated count and a
    fabricated absence off a failed read, one level below the set guard this
    module already had (#907 review round 3). An unclassifiable declaration is
    reported as such by both surfaces instead.

    The read is guarded because this runs over every declaration on a read path:
    an exception from one type's ``type_class`` would otherwise take down the
    entire `class list` (:func:`_declared_size` guards its width read for the same
    reason)."""
    try:
        _, target, reason = _follow_typedef(bv, name, type_obj)
        if reason is not None:
            # The chain did not terminate (an unregistered anonymous body, a
            # cycle, too many hops). The reference it stopped on may still state
            # which kind it names, which is enough to admit the class -- carrying
            # only the facts it does have -- and to keep refusing an enum alias.
            # A reference that names neither (Unknown/Typedef spellings, or no
            # `named_type_class` at all) classifies nothing: unreadable, because
            # the chain it would have been decided on is exactly what broke.
            names_class = _named_class_kind(target)
            if names_class is None:
                return _UNREADABLE_DECLARATION
            return target if names_class else None
        tc = getattr(target, "type_class", None)
        if tc is None:
            return _UNREADABLE_DECLARATION
        try:
            is_structure = int(tc) == _STRUCTURE_TYPE_CLASS
        except (TypeError, ValueError):
            is_structure = "Structure" in str(getattr(tc, "name", None) or tc)
        return target if is_structure else None
    except Exception:
        return _UNREADABLE_DECLARATION


def _user_declared_entries(bv) -> list[tuple[str, Any]] | None:
    """``[(name, type_obj)]`` for the types the USER declared in this view, or
    ``None`` when that set could not be read.

    BN's user type container is the set `types declare` (``define_user_type``)
    writes to, and it is the only observable answer to "did the user declare
    this": on a freshly loaded view it is empty while ``bv.types`` already holds
    everything BN imported (see the PROVENANCE note above). A view that exposes
    no container at all has no user declarations to offer, which is an empty
    answer and not a failed read -- the lens simply has no declared half there,
    exactly as before #675.2."""
    container = getattr(bv, "user_type_container", None)
    if container is None:
        return []
    entries = container.types
    if entries is None:
        return None
    # `{type_id: (QualifiedName, Type)}` -- keyed by id, so the NAME comes out of
    # the value, and a declaration renamed in place keeps one entry.
    return [(str(entry[0]), entry[1]) for entry in entries.values()]


class _DeclaredSet(NamedTuple):
    """What the view's USER declarations came to.

    ``types`` is every declared class type, keyed by the name it is declared
    under -- the LISTING shows all of it and ``class show`` resolves against
    all of it, because those two must never disagree about which names exist.
    ``unreadable`` names the declarations whose kind could not be established.

    There is deliberately no "BN generated this one" subset. BN's C parser
    registers a body of its own for `typedef struct { ... } T;` (spelled `_T`,
    or `anonymous_<N>` for an array element), and three review rounds tried to
    fold it out of the listing on the shape of its NAME. Each round measured a
    new escape, and the decisive one is not an escape: the spelling cannot
    tell BN's `_T` from a `struct _X { ... };` the user hand-wrote beside
    `typedef struct _X X;` -- the GLib/GTK idiom -- because they are the same
    name, and no BN attribute distinguishes them (measured). A rule that
    suppresses a class the user DID declare makes `class show` answer a
    confident absence about a name this view has, which is the exact defect
    #675.2 exists to remove. The rule was withdrawn: an extra row for BN's own
    half is cosmetic, a fabricated absence is not (#907 review round 5)."""

    types: dict[str, Any]
    unreadable: tuple[str, ...]


def _declared_types(bv) -> _DeclaredSet | None:
    """The USER-declared class types in this view -- the LIVE read -- or ``None``
    when the user's declarations could not be READ AT ALL.

    One function owns the enumeration AND every filter, so every consumer of the
    declared half reads the same, current view (see the block comment above for
    why nothing here may be memoised) and describes the same POPULATION: the
    listing's rows, its hidden count and `class show`'s fallback all resolve
    here, which is what stopped the count from counting the type table while the
    rows were something else (#907 review). Filtering a caller instead would
    re-split them on the next change.

    ``None`` is distinct from an empty set on purpose. An unreadable set must not
    report as zero declared classes: `0` reads as "the lens looked and found
    none", which is the exact blindness #675.2 exists to remove, so the callers
    disclose it instead (`? user-declared class types` on the listing, and a miss
    that does not claim the name is absent). Guarded here rather than per entry
    because a raising container took `class list` down whole -- the RTTI half
    included, which has no stake in the declared types.

    ``unreadable`` carries the same distinction one level down, for a single
    declaration: its kind could not be established, so it is neither a class nor
    a proven absence and is named rather than swallowed (see
    :func:`_class_type_target`)."""
    try:
        entries = _user_declared_entries(bv)
        if entries is None:
            return None
    except Exception:
        return None
    declared: dict[str, Any] = {}
    unreadable: list[str] = []
    for name, type_obj in entries:
        target = _class_type_target(bv, name, type_obj)
        if target is _UNREADABLE_DECLARATION:
            unreadable.append(name)
        elif target is not None:
            declared[name] = target
    return _DeclaredSet(declared, tuple(unreadable))


def _declared_size(type_obj) -> dict[str, Any] | None:
    """The ``{value: N}`` size envelope for ONE defined type, or ``None`` when it
    states no width (never fabricated) -- the same shape :func:`_object_size`
    reports, so a declared class and an RTTI class state their size one way.

    Called only for a record that is actually HANDED OUT. Reading a type's width
    is a LAZY operation in BN: measured on a small C++ target, the first pass over
    its 234 defined types costs ~250 ms (~1 ms per type, cached by BN after), so
    paying it per record in the builder would put that on every `class list` --
    including the default listing, whose declared records are counted and never
    rendered -- and multiples of it on a DWARF-heavy view."""
    try:
        width = int(getattr(type_obj, "width", 0) or 0)
    except Exception:
        return None
    return {"value": hex(width), "source": "declared_type"} if width > 0 else None


def _declared_type_records(declared: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{name: ClassRecord}`` for *declared* -- one :func:`_declared_types`
    reading of the view, and nothing else (#675.2).

    Takes the mapping rather than the view because both callers need the type
    objects too (for the size a returned record states), and the kind filter now
    does real work per entry: re-enumerating here made every `class list` and
    every `class show` miss pay for the view's type table twice. One reading also
    means the records and the type objects a caller pairs them with cannot come
    from two different moments of a live view.

    Every record is in the class registry's shape, so one renderer and one JSON
    consumer read both halves of the lens, and every record carries
    ``confidence: "declared-only"`` plus the note that says which kind of absence
    its empty ``methods``/``vtable``/``instances`` is (see
    :data:`_DECLARED_TYPE_NOTE`).

    Built from the cheap facts only. ``size`` (see :func:`_declared_size`) and the
    canonical ``ctx._type_entry`` -- which walks and renders a type's members --
    are drill-downs the CALLERS attach to a record they return (a show's matches:
    both; a listing's page rows: the size), exactly as the vtable layout and
    object size are drill-downs on the RTTI side."""
    return {
        name: {
            "name": name,
            "methods": [],
            "vtable": None,
            "typeinfo": None,
            "typeinfo_name": None,
            "size": None,
            "bases": [],
            "instances": _no_instances(),
            "confidence": _DECLARED_CONFIDENCE,
            "notes": [_DECLARED_TYPE_NOTE],
        }
        for name in declared
    }


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


def _is_type_expression_name(name: str) -> bool:
    """True when *name* is a C++ TYPE EXPRESSION (a function type ``void (int)``, a
    pointer/reference ``char const*`` / ``Foo&``, an array ``int [4]``, or a bare
    fundamental type ``unsigned int``) rather than a class/namespace identifier.

    RTTI is emitted for such non-object types too, producing typeinfo-only pseudo-class
    rows that are NOT domain classes (#481). The discriminator must be the name shape,
    NOT "typeinfo-only": a real class (esp. anonymous-namespace or template
    instantiations) frequently has typeinfo recovered but no vtable symbol / clustered
    methods, so keying on missing-vtable alone mislabels real classes."""
    # Strip the anonymous-namespace marker and balanced template args, which
    # legitimately contain spaces/parens in a REAL class name.
    s = name.replace("(anonymous namespace)", "")
    out = []
    depth = 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">" and depth:
            depth -= 1
        elif depth == 0:
            out.append(ch)
    core = "".join(out).strip()
    if not core:
        return False
    # A function type / pointer / reference / array / multi-token fundamental type.
    if any(c in core for c in "(*[&") or " " in core:
        return True
    _FUNDAMENTAL = {
        "void", "bool", "char", "signed", "unsigned", "short", "int", "long",
        "float", "double", "wchar_t", "char8_t", "char16_t", "char32_t", "nullptr_t",
        "__int128", "decltype(nullptr)",
    }
    # A bare fundamental type has no `::` qualifier and is a known keyword.
    return "::" not in core and core.lstrip("_") in _FUNDAMENTAL


def _is_non_class_rtti_artifact(rec: dict[str, Any]) -> bool:
    """#481: an RTTI row that is typeinfo for a NON-object TYPE (function/pointer/
    fundamental), not a domain class. Requires rtti confidence, no vtable, no clustered
    methods AND a type-expression name (so a real typeinfo-only class isn't mislabeled)."""
    return bool(
        rec["confidence"] == "rtti"
        and rec["vtable"] is None
        and len(rec["methods"]) == 0
        and _is_type_expression_name(rec["name"])
    )


def _list_row(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": rec["name"],
        "method_count": len(rec["methods"]),
        "has_vtable": rec["vtable"] is not None,
        "size": rec["size"],
        "bases": [b.get("name") for b in rec.get("bases", [])],
        "confidence": rec["confidence"],
        # #481: typeinfo emitted for a NON-object TYPE (function/pointer/fundamental),
        # not a domain class -- tagged so it doesn't inflate the class inventory and
        # agents can filter it. Keyed on the name SHAPE (not just missing-vtable), so a
        # real typeinfo-only class (anonymous-namespace / template) isn't mislabeled.
        "artifact": _is_non_class_rtti_artifact(rec),
    }


def _class_lens_inputs(ctx, bv) -> dict[str, int]:
    """The raw inputs the class lens clusters from, so a ZERO-class result is
    attributable (#653.6).

    `classes: 0 shown of 0` is correct on a C target and identical to what a
    clustering failure would print, so an agent had to spend extra calls proving
    RTTI absence by hand (`strings --regex '_ZTV|_ZTI'`). Reporting the empty
    inputs makes the absence self-evident. Only computed for the zero case, where
    there is by definition no other work to do.
    """
    demangled = 0
    for fn in (getattr(bv, "functions", None) or []):
        try:
            _demangled, cls, _kind = _classify_function(fn)
        except Exception:
            continue
        if cls is not None:
            demangled += 1
    try:
        rtti = _rtti_symbol_maps(bv)
    except Exception:
        rtti = {}
    return {
        "demangled_cxx_methods": demangled,
        "rtti_typeinfo_symbols": sum(
            1 for syms in rtti.values() if syms.get("typeinfo") or syms.get("typeinfo_name")),
        "rtti_vtable_symbols": sum(1 for syms in rtti.values() if syms.get("vtable")),
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
    count_only: bool = False,
) -> dict[str, Any]:
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    bv = ctx._resolve_view(selector)
    registry = _build_class_registry(ctx, bv, query=query)
    # #675.2: the DECLARED records join the candidates -- read live, never folded
    # into the memoised registry (see `_declared_type_records`). A name the RTTI
    # half already clustered keeps its RTTI record, except when that "class" is
    # a thunk-shaped symbol artifact. A declared class under that same spelling
    # remains a real type and must not be lost with the artifact.
    needle = query.lower() if query else None
    # ONE live reading of the view's declared class types: the records below and
    # the type objects their returned rows take a size from (`_declared_size`)
    # both come from it. `None` means the declarations could not be read --
    # disclosed through the counter, never flattened to "no declared classes"
    # (#907 review), so the listing prints `? user-declared class types` and the
    # RTTI half still answers. A single declaration whose KIND could not be read
    # unmeasures the count the same way: it is neither in the set nor proven
    # outside it, so the number would exclude it silently (#907 review round 3).
    declared_set = _declared_types(bv)
    declared_types = declared_set.types if declared_set is not None else {}
    declared_unmeasurable = declared_set is None or bool(declared_set.unreadable)
    # No name-shape filter runs here: the rows ARE the user type container's
    # class types, which is the only population `class show` can resolve
    # against without one surface asserting an absence the other contradicts
    # (see :class:`_DeclaredSet`). BN's own parser half for an anonymous
    # typedef therefore lists as a peer row, and that is honest.
    declared_records = _declared_type_records(declared_types)
    declared_only = [
        rec for name, rec in declared_records.items()
        if (name not in registry or _is_thunk_artifact(name))
        and (needle is None or needle in name.lower())
    ]
    candidates = []
    library_suppressed = 0
    vendor_suppressed = 0
    construction_vtables_suppressed = 0
    thunks_suppressed = 0
    declared_suppressed = 0
    for rec in [*registry.values(), *declared_only]:
        name = rec["name"]
        # Both artifact filters read the shape of a DEMANGLED SYMBOL, so they
        # have no standing over a name the user typed into `types declare`.
        # Applied to the declared half they split the two surfaces: a declaration
        # spelled like a thunk vanished from `class list --all` and was
        # attributed to `N thunks` while `class show` of the same name returned a
        # full card, and the construction-vtable gate (which fires BEFORE the
        # declared counter) made the default header promise one fewer class than
        # `--all` then listed (#907 review round 3).
        symbol_artifact = rec["confidence"] != _DECLARED_CONFIDENCE
        # A thunk is never a type -- drop it unconditionally (even under --all),
        # so it isn't surfaced as a class/namespace (#309).
        if symbol_artifact and _is_thunk_artifact(name):
            thunks_suppressed += 1
            continue
        # Construction-vtable artifacts (`X{for `Base'}`) ARE real RTTI objects,
        # just not classes -- hide by default, reveal under --all (#309).
        if symbol_artifact and not include_all and _is_construction_vtable_artifact(name):
            construction_vtables_suppressed += 1
            continue
        if no_stl and _is_library_class(name):
            library_suppressed += 1
            continue
        if no_vendor and _is_vendor_class(name):
            vendor_suppressed += 1
            continue
        if not (include_all or rec["confidence"] in ("rtti", "ctor")):
            # #675.2: a declared class type is not RTTI/ctor-confirmed either, so
            # the same gate folds it out of the default listing -- counted, so the
            # fold-out is disclosed the way the library/vendor suppressions are
            # rather than leaving `class list` silently blind to it.
            #
            # The library/vendor filters already ran, so this count is exactly
            # the declared rows `--all` would add under the same flags. A
            # filtered declaration is counted under the filter that hid it,
            # even on the default listing.
            if rec["confidence"] == _DECLARED_CONFIDENCE:
                declared_suppressed += 1
            continue
        candidates.append(rec)
    candidates.sort(key=lambda r: r["name"])
    if declared_unmeasurable:
        # The disclosure choke point for a counter that could not be measured:
        # `_stated_count` renders a non-int as `?`, so the header says
        # `? user-declared class types (--all to show)` instead of a fabricated
        # `0` (#619's rule, #907's unreadable-table finding). One declaration
        # whose kind could not be read unmeasures it just as the whole set does:
        # the number would otherwise state a total that silently excludes it.
        declared_suppressed = _DECLARED_UNREADABLE
    # `total` counts candidates AFTER the confidence + --no-stl filters but before
    # paging, so it reflects what this query actually surfaced.
    total = len(candidates)
    # #484: count-only fast path for class-lens scale characterization -- respects
    # every filter (--query / --no-stl / --no-vendor / --all) and skips the per-page
    # base decode. `artifact_count` (non-class RTTI/type rows, #481) is broken out so
    # the domain-class count is honest.
    if count_only:
        artifact_count = sum(1 for r in candidates if _is_non_class_rtti_artifact(r))
        result = {
            "kind": "classes",
            "count": total,
            "total": total,
            "artifact_count": artifact_count,
            "include_all": include_all,
            "no_stl": no_stl,
            "no_vendor": no_vendor,
            "declared_suppressed": declared_suppressed,
            **_analysis_state_fields(bv),
        }
        if total == 0:
            result["inputs"] = _class_lens_inputs(ctx, bv)   # #653.6
        return result
    page = candidates[offset:] if offset else candidates
    if limit is not None:
        page = page[:limit]
    # Populate RTTI bases for the RETURNED PAGE only: base decode is cheap (a few
    # typeinfo reads per class) and bounded to the page, so a single `class list`
    # recovers the inheritance graph without N `class show` calls. Vtable layout
    # and object size stay show-only -- they are far costlier per class. (#205 review)
    rows = []
    for rec in page:
        if rec["confidence"] == _DECLARED_CONFIDENCE:
            # #675.2: the declared row's size, taken for this page only -- the
            # width is a lazy read in BN (see `_declared_size`).
            type_obj = declared_types.get(rec["name"])
            if type_obj is not None:
                rec["size"] = _declared_size(type_obj)
        try:
            rec["bases"] = ctx._bases_for(bv, rec)
        except Exception:
            rec["bases"] = []
        rows.append(_list_row(rec))
    returned = len(rows)
    result = {
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
        "declared_suppressed": declared_suppressed,
        **_analysis_state_fields(bv),
    }
    if total == 0:
        # #653.6: make "no classes" attributable -- C target vs failed clustering.
        result["inputs"] = _class_lens_inputs(ctx, bv)
    return result


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
    scan rather than fabricating slots (#205 review).

    #821: a `status == "function"` hit is NOT sufficient on its own. The
    normalizer reports `function` for an INTERIOR address too -- a data word
    that happens to equal some mid-function PC -- with `exact_start: False`
    on the entry. A vtable slot points at a function ENTRY, so an interior
    hit is not a virtual method; accepting it extended the slot list past
    the real end of the table until some later word happened to terminate
    it, and the warning was only post-hoc. Terminating is the same choice
    already made for a non-code pointer: end the scan rather than fabricate
    slots.

    Tri-state, like every other evidence gate here: only an AFFIRMATIVE
    `exact_start: False` terminates. An entry that does not carry the key --
    an older payload, or a producer that does not compute it -- says nothing
    about interiority and must not end a scan on absent evidence."""
    if target.get("status") == "function":
        entry = target.get("function")
        if isinstance(entry, dict) and entry.get("exact_start") is False:
            return False
        return True
    return (target.get("context") or {}).get("kind") == "code"


def _lookahead_row_confirms_continuation(row: dict[str, Any] | None) -> bool:
    """#706 follow-up: row `max_slots` -- read but never scanned as a slot --
    resolves the one ambiguity the raw cap can't: a trailing run of null rows
    inside the window reads identically whether the vtable ended there (the
    next object's zeroed padding) or is an unresolved-relocation run inside a
    table that keeps going past the cap (#441). A readable code/null/extern
    row one past the cap is the same kind of entry the window itself accepts
    as a slot, so the table demonstrably continues. Missing (the reader
    couldn't produce it), unreadable, or a mapped-data pointer is never
    treated as evidence of continuation -- absence of proof is not proof the
    table goes on."""
    if row is None or not row.get("readable"):
        return False
    target = row.get("target") if isinstance(row.get("target"), dict) else {}
    status = target.get("status")
    kind = (target.get("context") or {}).get("kind")
    return _slot_is_code(target) or status == "null" or kind == "extern"


def _vtable_row_slot(bv, index: int, row: dict[str, Any]) -> dict[str, Any] | None:
    """The slot a raw pointer-table ROW contributes to a vtable, or ``None``
    when the row is a genuine table BOUNDARY (unreadable, or a mapped
    data/unmapped pointer -- the next object). Shared by the capped window scan
    in ``_vtable_layout`` and the one-slot probe ``_vtable_slot_probe`` (#822)
    so both classify a row identically: a probe that accepted a row the window
    scan would have stopped at could name a method from the object AFTER the
    table."""
    if not row.get("readable"):
        return None
    target = row.get("target") if isinstance(row.get("target"), dict) else {}
    value = row.get("value")
    status = target.get("status")
    kind = (target.get("context") or {}).get("kind")
    if _slot_is_code(target):
        fn = target.get("function")
        name = (fn or {}).get("name") if isinstance(fn, dict) else None
        method = None
        if isinstance(fn, dict):
            # The pointer-table function dict carries the MANGLED fn.name; add
            # the demangled display name (symbol short_name) so slots read like
            # the methods list, not raw `_ZN...`. Mangled `name` is kept. (#205)
            method = {**fn, "display_name": _demangled_slot_name(bv, fn)}
        return {
            "index": index,
            "address": value,
            "method": method,
            "pure_virtual": name == "__cxa_pure_virtual",
            "unnamed": (isinstance(name, str) and name.startswith("sub_")) or (fn is None),
        }
    if status == "null":
        # Interior null slot (pure-virtual placeholder / unresolved reloc).
        # Kept so a leading/interior null doesn't truncate the vtable; a
        # trailing run of these is trimmed below.
        return {
            "index": index, "address": value, "method": None,
            "pure_virtual": False, "unnamed": True, "null": True,
        }
    if kind == "extern":
        # External slot: `__cxa_pure_virtual` (a pure-virtual marker) or a
        # cross-module virtual method. A valid vtable entry -- don't stop.
        ext_name = _slot_external_name(bv, value)
        return {
            "index": index, "address": value, "method": None,
            "pure_virtual": ext_name == "__cxa_pure_virtual",
            "external": True, "external_name": ext_name,
            "unnamed": ext_name is None,
        }
    return None


# #822: how many raw rows past the display window `_vtable_slot_probe` will walk
# to reach ONE requested slot. The slot index comes from an MLIL constant, so an
# obfuscated/garbage offset must not turn a single slot lookup into an unbounded
# table walk; hitting this bound is DISCLOSED (`vtable_slot_probe_limit`), never
# silently read as an absent slot.
_VTABLE_SLOT_PROBE_MAX_ROWS = 4096


def _vtable_row_stop_reason(row: dict[str, Any]) -> str:
    """WHY a row ended a vtable scan, for a row ``_vtable_row_slot`` answered
    ``None`` for. ``"boundary"``: the row read fine and is a mapped data /
    unmapped pointer -- the next object -- so the table ENDED there.
    ``"unreadable"``: the row's bytes could not be read at all, so the scan
    cannot tell whether the table continues past it. Shared by the capped window
    scan in ``_vtable_layout`` and the one-slot probe ``_vtable_slot_probe`` so
    neither reports a failed READ as an observed table end: only ``"boundary"``
    lets a caller claim an exact total, and only ``"boundary"`` proves the
    requested slot absent (#822)."""
    return "boundary" if row.get("readable") else "unreadable"


def _vtable_slot_probe(
    ctx, bv, vtable_addr: int, slot_index: int, *, scanned_through: int,
) -> dict[str, Any]:
    """Resolve ONE requested slot that lies past the display window (#822).

    ``_vtable_layout``'s cap bounds the LISTING, not the vtable: a caller asking
    for slot 70 of an 80-slot table must not be told the slot is unresolved just
    because `class show` stops displaying at 64. Reading row 70 in isolation
    would be unsound -- a boundary (the next object) between the window and row
    70 must end the table first -- so this walks the contiguous run of rows from
    where the window scan stopped up to the requested index.

    Returns ``{"status", "slot", "rows_scanned"}`` with ``status`` one of:
      * ``"resolved"`` -- the requested row is a valid vtable entry (``slot``);
      * ``"not_present"`` -- a genuine table boundary was reached first, so the
        requested index is past the end of this table;
      * ``"unknown"`` -- a row before the requested index could not be READ, so
        the table's end is unobserved and the slot is undecided;
      * ``"limit_reached"`` -- the walk would exceed
        ``_VTABLE_SLOT_PROBE_MAX_ROWS`` rows, so the slot is UNKNOWN.
    """
    if not vtable_addr or slot_index < 0:
        return {"status": "not_present", "slot": None, "rows_scanned": 0}
    ptr = ctx._pointer_size(bv)
    first_row = max(0, int(scanned_through))
    count = slot_index - first_row + 1
    if count <= 0:
        return {"status": "not_present", "slot": None, "rows_scanned": 0}
    if count > _VTABLE_SLOT_PROBE_MAX_ROWS:
        return {"status": "limit_reached", "slot": None, "rows_scanned": 0}
    start = int(vtable_addr) + 2 * ptr + first_row * ptr
    table = ctx._pointer_table_layout(bv, start, entries=count, stride=ptr)
    rows = table.get("items") or table.get("entries") or []
    slot: dict[str, Any] | None = None
    for offset, row in enumerate(rows):
        row_d = row if isinstance(row, dict) else {}
        slot = _vtable_row_slot(bv, first_row + offset, row_d)
        if slot is None:
            if _vtable_row_stop_reason(row_d) == "unreadable":
                # A read FAILURE, not an observed boundary: whether the table
                # continues past this row is unknown, so the requested slot is
                # undecided -- never "not present" (#822).
                return {"status": "unknown", "slot": None, "rows_scanned": offset + 1}
            # The table ended on a genuine boundary before the requested index --
            # the slot is absent, not merely unscanned.
            return {"status": "not_present", "slot": None, "rows_scanned": offset + 1}
    if slot is None or len(rows) < count:
        # The reader returned fewer rows than the requested index needs: this
        # table has no such row (absence of the row, not proof of a boundary).
        return {"status": "not_present", "slot": None, "rows_scanned": len(rows)}
    return {"status": "resolved", "slot": slot, "rows_scanned": count}


def _vtable_layout(ctx, bv, vtable_addr: int, *, max_slots: int = 64) -> dict[str, Any]:
    """Function slots of an Itanium vtable. Words [0] (offset-to-top) and [1]
    (typeinfo ptr) are header; slots start at +2*ptr_size. Reuses the
    Thumb-aware pointer-table reader. Only CODE targets count as slots.

    The returned ``scanned`` count is raw table entries examined, including
    any terminating row that stopped the scan -- it is neither ``len(slots)``
    nor a virtual-method count.

    ``max_slots`` caps the returned LISTING, and the result says so instead of
    letting a prefix read as the whole table (#584, #822):
      * ``total`` -- the EXACT number of vtable entries, or ``None`` when the
        table's end was never OBSERVED. Three shapes leave it unknown, named by
        ``truncated_reason``: the scan hit ``max_slots`` with the table still
        going (``"scan_capped"``), it stopped on a row it could not READ
        (``"unreadable_row"`` -- a failed read is not an end), or the address has
        no decodable local vtable body at all (``"no_local_vtable_body"``). A
        count the scan did not establish is never reported as exact. Entries, not
        virtual methods: a null/external placeholder row is an entry (the listing
        keeps it), a trailing padding run is not;
      * ``total_lower_bound`` -- an int only when ``total`` is ``None``: the
        validated minimum, i.e. the window's entries plus the lookahead row that
        proved the table continues (``None`` when nothing was proven -- the
        no-local-body case);
      * ``truncated``/``slots_truncated`` -- the listing is a PREFIX of a longer
        table (``slots_truncated`` is the same fact under the name #584 asked
        for; both are kept);
      * ``scan_truncated`` -- the total is not exact because a scan stopped
        before the table's end (the ``read_listing`` callsites convention:
        ``scan_truncated`` pairs with ``total: None``, never with a fabricated
        count);
      * ``truncated_reason`` -- the machine-readable why for a ``None`` ``total``
        (see above), ``None`` once the table's end was observed.
    ``_vtable_slot_probe`` resolves a requested slot past this window."""
    ptr = ctx._pointer_size(bv)
    # Itanium invariant: word[1] (vtable_addr + ptr) points to the class's
    # typeinfo. If it doesn't resolve to a typeinfo symbol, this address is NOT
    # the start of a real local vtable OBJECT -- it's an import/GOT pointer slot
    # (the vtable is defined in another module) or a PIE slot relocated to zero.
    # Decoding +2*ptr there would render adjacent GOT/data as fake slots
    # (#205 review), so report no slots and let the caller note it.
    ti_ptr = ctx._read_pointer_value(bv, vtable_addr + ptr, size=ptr)
    if not ti_ptr or ctx._typeinfo_name_at(bv, ti_ptr) is None:
        # #822: this is a DECIDED "no local vtable body", not a capped scan -- it
        # never read a slot, so an exact `total: 0` would count a table this gate
        # deliberately refused to decode (the GOT/relocated-to-zero shapes above,
        # plus RTTI-less builds) as an empty one. Report it unknown with the
        # reason; no bound (nothing was proven) and no scan stop, so
        # `evidence virtual-call` does not probe a table this gate rejected.
        return {
            "address": hex(int(vtable_addr)),
            "slots": [],
            "truncated": False,
            "max_slots": max_slots,
            "scanned": 0,
            "total": None,
            "total_lower_bound": None,
            "slots_truncated": False,
            "scan_truncated": False,
            "truncated_reason": "no_local_vtable_body",
        }
    start = vtable_addr + 2 * ptr
    # #706 follow-up: read one entry PAST the cap. Row `max_slots` is a
    # lookahead ONLY -- never a slot in its own right (the window below
    # excludes it from both scanning and `slots`) -- consulted after the loop
    # to resolve the trailing-null ambiguity (see the `else` branch).
    table = ctx._pointer_table_layout(bv, start, entries=max_slots + 1, stride=ptr)
    slots: list[dict[str, Any]] = []
    # #303: the pointer-table reader (`_pointer_table_for_view`) returns the
    # canonical #275 envelope, whose rows live under `items`. This loop read the
    # pre-#275 `entries` key, so every vtable resolved to ZERO slots and
    # `class show` declared a recoverable dispatch table unrecoverable. Read
    # `items` (with an `entries` fallback for any legacy producer / test fake).
    #
    # #441: a real vtable interleaves code slots with NULL slots (a pure-virtual
    # relocated to 0, or an unresolved relocation) and EXTERNAL slots
    # (`__cxa_pure_virtual`, or a cross-module virtual method). Terminating at the
    # first non-code slot (the old behavior) truncated the whole vtable at a
    # leading null/`__cxa_pure_virtual`, so PIE C++ classes came back with ZERO
    # slots even though `evidence table` resolved every one. Include null and
    # external slots as valid entries and terminate only at a genuine boundary --
    # an unmapped word (the next sub-vtable's offset-to-top, e.g. -8) or a mapped
    # DATA pointer (the next object's typeinfo). Trailing null slots are trimmed.
    raw_entries = table.get("items") or table.get("entries") or []
    window = raw_entries[:max_slots]
    scanned = 0
    truncated = False
    # #822 review (round 1): WHY the scan stopped, which is what decides whether
    # the table's end was OBSERVED and `total` may be exact. None == observed
    # (a genuine boundary row, or the reader had no more rows); "capped" == the
    # window filled with the table still going; "unreadable" == a row could not
    # be READ, which proves nothing about the end either way.
    stop_reason: str | None = None
    # #584: track whether the window scan hit `max_slots` without ever finding
    # a genuine terminator (unreadable slot / mapped data pointer). `for...else`
    # runs the `else` branch only when the loop completed without `break` --
    # i.e. every entry in the window was consumed as a valid slot, so the raw
    # scan alone can't tell whether the real vtable ends there or continues
    # past `max_slots` (the lookahead below resolves it).
    for i, row in enumerate(window):
        scanned = i + 1
        # #822: the row -> slot classification lives in `_vtable_row_slot` so the
        # one-slot probe past the window applies the SAME boundary rules; `None`
        # here is a terminator -- a failed read (`_vtable_row_stop_reason`
        # "unreadable", the end NOT observed) or a mapped data pointer / the next
        # object's typeinfo (a genuine boundary), which end the scan either way.
        row_d = row if isinstance(row, dict) else {}
        slot = _vtable_row_slot(bv, i, row_d)
        if slot is None:
            if _vtable_row_stop_reason(row_d) == "unreadable":
                stop_reason = "unreadable"
            break
        slots.append(slot)
    else:
        # #706 follow-up: the window's `for...else` completion is ambiguous by
        # itself -- a trailing null run consumed to reach `max_slots` reads
        # identically whether it's the next object's padding (table ended) or
        # an unresolved-reloc run inside a table that keeps going (#441). The
        # one-entry lookahead at row `max_slots` (never scanned as a slot)
        # resolves it directly instead of inferring from what the trim below
        # removes: it proves the table continues (or doesn't) independent of
        # `slots`'s final shape.
        if scanned >= max_slots > 0:
            lookahead = raw_entries[max_slots] if len(raw_entries) > max_slots else None
            if _lookahead_row_confirms_continuation(lookahead):
                truncated = True
                stop_reason = "capped"
            elif isinstance(lookahead, dict) and not lookahead.get("readable"):
                # The lookahead row itself could not be read: the window is full,
                # so 64 entries are proven, but nothing proves the table stops
                # there (#822 review round 1).
                stop_reason = "unreadable"
    # A vtable ends at a real or pure-virtual method, so a trailing run of null
    # slots is the next object's zeroed offset-to-top / padding, not a slot --
    # but ONLY once the table is known to have ended (`stop_reason` None): when
    # the table demonstrably continues, or a row that stopped the scan could not
    # be read, those same trailing nulls are interior placeholder slots (or
    # unknown), not padding, and trimming them would silently drop entries from a
    # table whose end was never observed.
    if stop_reason is None:
        while slots and slots[-1].get("null"):
            slots.pop()
    # #822 (the #584 residual): two different questions, two flags, one truth.
    # `truncated` answers "is my LISTING a prefix?"; `scan_truncated` answers
    # "is `total` exact?" -- and the answer to the second is what keeps a capped
    # or partially-read scan from reading as a complete table. An unobserved end
    # means the exact total is UNKNOWN (never a fabricated count); the entries
    # the scan did PROVE are a lower bound, plus the one lookahead row that
    # proved the table continues past a full window.
    return {
        "address": hex(int(vtable_addr)),
        "slots": slots,
        "truncated": truncated,
        "max_slots": max_slots,
        "scanned": scanned,
        "total": None if stop_reason is not None else len(slots),
        "total_lower_bound": (None if stop_reason is None
                              else len(slots) + (1 if truncated else 0)),
        "slots_truncated": truncated,
        "scan_truncated": stop_reason is not None,
        "truncated_reason": {"capped": "scan_capped",
                             "unreadable": "unreadable_row"}.get(stop_reason),
    }


def _slot_external_name(bv, value: Any) -> str | None:
    """Raw symbol name at an external vtable slot's target address (e.g.
    ``__cxa_pure_virtual``), or None if unresolved (#441)."""
    try:
        addr = int(value, 16) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        return None
    getter = getattr(bv, "get_symbol_at", None)
    sym = getter(addr) if callable(getter) else None
    if sym is None:
        return None
    return getattr(sym, "raw_name", None) or getattr(sym, "name", None)


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


#: #817: the ctor-new path cannot search at all -- the seam's
#: ``_operator_new_size_at_ctor`` is still an unconditional ``return None`` stub
#: (the #205 its TODO defers to is CLOSED). A size miss therefore means "one path
#: was exhausted and the other was never tried", which is what this reason code
#: says; a bare null said only "unknown".
_SIZE_UNAVAILABLE_REASON = "operator_new_recovery_unimplemented"


def _object_size(ctx, bv, record: dict[str, Any]) -> dict[str, Any] | None:
    """Object size with provenance. A defined BN type's width wins (authoritative
    when present); else the operator-new size at a construction site; else None
    (never fabricated). ``ctx._find_type`` raises on a miss, so the lookup is
    guarded.

    A miss stays ``None`` rather than becoming a value-less envelope: the class
    card renders any ``size`` DICT as ``size ?`` because #619 requires that "an
    envelope that carries no value still CLAIMED a size" be visible, so returning
    one for every miss would both change the default card of every class whose
    size is unrecoverable and make that distinction vacuous. The provenance of a
    miss travels beside the size instead -- see ``_enrich``'s ``size_source`` /
    ``size_reason`` (#817)."""
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
    globals hold one. Empty (not an error) when nothing is found.

    ``cap`` bounds each LISTING only (#822: the same shape as the vtable cap),
    so the result also reports what the cap hid: ``*_total`` is the exact number
    found and ``*_truncated`` says the listing is a prefix. A capped 128 must
    never read as "this class is constructed exactly 128 times"."""
    sites = ctx._ctor_construction_sites(bv, record)
    stored = ctx._global_vtable_stores(bv, record) if record.get("vtable") else []
    shown_sites = sites[:cap]
    shown_stored = stored[:cap]
    return {
        "construction_sites": shown_sites,
        "stored_globals": shown_stored,
        "construction_sites_total": len(sites),
        "construction_sites_truncated": len(sites) > len(shown_sites),
        "stored_globals_total": len(stored),
        "stored_globals_truncated": len(stored) > len(shown_stored),
    }


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
    from .read_evidence import _function_thunk_summary

    # Class-list's thunk suppression is name filtering, not method evidence.
    # Inspect only this class's exact entries; a same-name body is not a target.
    for method in rec["methods"]:
        func = ctx._find_function(bv, method["address"])
        method["thunk"] = _function_thunk_summary(ctx, bv, func)

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
    size = ctx._object_size_for(bv, rec)
    rec["size"] = size
    # #817: where the size came from is a JSON-surface fact of its own, so it
    # rides BESIDE the size instead of inside it -- one place to read the
    # provenance whether or not a size was recovered, and the default text card
    # (which renders only `size`) is unchanged for a class that has none. The
    # `size_reason` code then distinguishes "both paths searched, nothing found"
    # from what actually happens today: the ctor-new path is a stub and was
    # never tried. Null on a hit, mirroring `hlil_statement_reason`.
    rec["size_source"] = size["source"] if size else "unavailable"
    rec["size_reason"] = None if size else _SIZE_UNAVAILABLE_REASON
    rec["bases"] = ctx._bases_for(bv, rec)
    rec["instances"] = ctx._instances_for(bv, rec)
    return rec


def _class_name_suggestions(registry: dict[str, dict], name: str, *, limit: int = 5) -> str:
    """#413: a `" Did you mean: A, B?"` fragment for an unresolved class query, or
    "" when nothing is close. Combines difflib fuzzy matches over the full names
    with a leaf-prefix pass so both a typo (`Sesion`->`Session`) and a bare leaf
    (`Handler`->`ns::Handler`) get a useful pointer."""
    # Skip construction-vtable / thunk artifacts -- they aren't classes an analyst
    # would query, so they only add noise to a "did you mean" hint (#309).
    names = [k for k in registry
             if not _is_construction_vtable_artifact(k) and not _is_thunk_artifact(k)]
    if not names:
        return ""
    close = difflib.get_close_matches(name, names, n=limit, cutoff=0.6)
    leaf = _query_leaf(name).lower()
    leaf_hits = [k for k in names if leaf and _query_leaf(k).lower().startswith(leaf)]
    ordered = list(dict.fromkeys([*close, *leaf_hits]))[:limit]
    return f" Did you mean: {', '.join(ordered)}?" if ordered else ""


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
    # #622 (d): the per-view registry IS reused now (see `_build_class_registry`),
    # so a repeat `class show` enumerates nothing. What stays unchanged is the
    # removal of the partial "cheaper show path" -- a build that materialised only
    # the queried class measured SLOWER than the full one, so it was not
    # reimplemented. A miss needs the full registry anyway, because the suggestion
    # hint is drawn from every class name (#413).
    registry = _build_class_registry(ctx, bv)
    # #675.2: the RTTI/symbol half is blind to a class the user DECLARED -- it has
    # no `_ZTV`/`_ZTI` symbol and no demangled method to cluster -- so the declared
    # half is read here too. ONE live reading: the records and the type objects
    # their facts come from.
    #
    # Resolved over ONE namespace holding BOTH halves, never the registry first.
    # Resolving the registry alone and consulting the declared set only on a MISS
    # let an RTTI class in ANY namespace swallow every bare-leaf query: a user who
    # declared `Thing` in a view whose binary exposes `ns::Thing` got the RTTI card
    # back with `ambiguous` unset and nothing disclosed, while `class list --all`
    # listed and counted the declaration that card could no longer reach -- the two
    # surfaces disagreeing about which names exist, which is the one thing the
    # declared half exists to prevent. One namespace puts the resolver's documented
    # exact-match-first rule in charge instead: an exact declared name beats a
    # registry LEAF match, and a genuine leaf collision across the halves is
    # AMBIGUOUS and shows both (#907 review round 5).
    #
    # This costs the declared read on every show, including one the registry
    # answers. That is the price of the two surfaces describing one population:
    # `class list` already pays it per call, the container is the user's own
    # declarations rather than the view's type table, and the expensive per-record
    # facts (`_declared_size`, `ctx._type_entry`) stay drill-downs on a record
    # actually handed out.
    declared_set = _declared_types(bv)
    declared_types = declared_set.types if declared_set is not None else {}
    declared = _declared_type_records(declared_types)
    # A name BOTH halves carry keeps its RTTI record unless the registry entry
    # is a thunk-shaped artifact, which is never a class. The listing drops
    # that artifact and keeps the declaration; show must resolve the same name
    # to the declared card.
    registry_classes = {key: rec for key, rec in registry.items()
                        if key not in declared or not _is_thunk_artifact(key)}
    candidates = {**declared, **registry_classes}
    matches = _resolve_class_names(candidates, name)
    # Whether this query ALSO reaches a declaration whose kind could not be
    # established. Resolved over the same namespace PLUS the unreadable names, so
    # the exact-match-first rule decides who discloses: a fully qualified name
    # resolves to itself and says nothing, while a bare leaf that also reaches an
    # unreadable sibling must. The unreadable names cannot SELECT a card -- there
    # is no record behind them -- so they are resolved for disclosure only.
    unreadable_names = set(declared_set.unreadable) if declared_set else set()
    unreadable_here = [
        match for match in _resolve_class_names(
            dict.fromkeys([*candidates, *unreadable_names], {}), name)
        if match in unreadable_names
    ]
    # A match is not a licence to stop disclosing, and WHICH half answered must
    # not decide whether the reader is told: the listing prints
    # `? user-declared class types` about this very view at this very moment, so a
    # card that presents itself as the whole answer contradicts it. Both the
    # whole-set failure and the single unreadable declaration reach every returned
    # record, RTTI or declared (#907 review rounds 4 and 5).
    if declared_set is None:
        disclosure = ("the view's declared types could not be read, so this card"
                      " is not the whole answer")
    elif unreadable_here:
        disclosure = (f"{len(unreadable_here)} further declaration(s) of this name"
                      " could not be read, so this card is not the whole answer")
    else:
        disclosure = None
    if not matches:
        # A failed read is not an absent class: the miss says what it could not
        # read instead of asserting a "No class named" this call has no standing
        # to assert (#907 review). The message is byte-identical to the one it
        # has always emitted whenever the declarations read fine.
        #
        # Two failures reach here. The whole set may be unreadable, or THIS name
        # may be one of the declarations whose kind could not be established --
        # resolved by the same name resolution as the set itself, so a bare leaf
        # query still matches a qualified declaration. Answering the second with
        # a flat miss claimed a name was absent while holding the entry that
        # carries it (#907 review round 3).
        if declared_set is None:
            unreadable = (" The view's declared types could not be read, so a"
                          " declared class of this name is not ruled out.")
        elif _resolve_class_names(dict.fromkeys(declared_set.unreadable, {}), name):
            unreadable = (" A type this view declares under that name could not be"
                          " read, so a declared class of this name is not ruled out.")
        else:
            unreadable = ""
        raise OperationFailure(
            "unknown_class",
            f"No class named {name!r}.{_class_name_suggestions(registry, name)} "
            f"Run `bn class list` (add --all for name-only clusters) to discover "
            f"available classes.{unreadable}",
        )
    records = []
    for match in matches:
        if match in registry_classes:
            rec = _enrich(ctx, bv, registry_classes[match])
        else:
            # A declared record comes back as it was BUILT: `_enrich`'s RTTI
            # drill-downs (vtable layout, bases, instances) are precisely what
            # its note says this view has no evidence for, so running them would
            # only decorate the card.
            rec = declared[match]
            type_obj = declared_types.get(match)
            if type_obj is not None:
                rec["size"] = _declared_size(type_obj)
                # The canonical `types` entry for the declaration (decl, layout,
                # members) -- a SHOW-only drill-down, exactly as the vtable
                # layout and object size are, and for the same reason:
                # `_type_entry` walks and renders the members, which is
                # affordable for the one class being shown and not for a listing
                # (see `_declared_type_records`).
                rec["type"] = ctx._type_entry(match, type_obj)
        if disclosure:
            rec.setdefault("notes", []).append(disclosure)
        records.append(rec)
    if len(records) == 1:
        return records[0]
    return {"ambiguous": True, "query": name, "matches": records}
