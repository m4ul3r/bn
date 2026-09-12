"""The ``BridgeContext`` seam: resolution / ABI / address-context helpers.

``BridgeContext`` is the one shared dependency the read-op domains and the
mutation engine talk to instead of reaching back into ``BinaryNinjaBridge``.
Its only state is ``targets`` (the ``TargetManager`` passed in by the bridge);
every other helper takes a ``BinaryView`` (``bv``) explicitly and is otherwise
state-free.

``_find_type`` and ``_render_type_layout`` are relocated here from the read_types
and mutation clusters respectively (both are state-free), as are the shared
type-entry builders ``_type_entry`` and ``_current_type_entry``. This breaks the
one real import cycle (``read_types`` <-> ``mutation_engine``): each ends up
importing only this seam, never each other (design spec §3.2).

This module imports ONLY stdlib + binaryninja + ``._shared`` -- never ``bridge``.
"""
from __future__ import annotations

import difflib
import re
import threading
import weakref
from typing import Any

try:
    import binaryninja as bn
except ModuleNotFoundError:  # importable without the Binary Ninja runtime (tests, tooling)
    bn = None  # type: ignore[assignment]

from ._shared import (
    OperationFailure,
    _format_ambiguous_function_error,
    _format_ambiguous_symbol_error,
    _parse_address,
    _symbol_type_name,
)

_TYPE_CLASS_NAMES: dict[int, str] = {
    0: "void",
    1: "bool",
    2: "int",
    3: "float",
    4: "struct",
    5: "enum",
    6: "pointer",
    7: "array",
    8: "function",
    9: "varargs",
    10: "value",
    11: "named_type_ref",
    12: "wide_char",
}

# Symbol types whose function is an import/extern stub (PLT trampoline / extern
# ref), not the real body. When a name collision is only such stubs shadowing
# one real implementation, resolution auto-picks the implementation (#122).
_STUB_SYMBOL_TYPE_NAMES = frozenset({"ImportedFunctionSymbol", "ExternalSymbol"})


# Nested anonymous aggregates are expanded in the text layout and in the JSON
# members[] tree so their inner members aren't invisible (#370.2). The depth is
# capped only to bound recursion on a pathological (even self-referential)
# aggregate -- deep nesting is real, so the cap is generous -- and hitting it is
# DISCLOSED, never silent: the text layout renders an explicit
# "... truncated at depth N" line and the JSON entry whose children were cut
# carries "truncated": true, so a capped subtree is never read as an empty
# struct (#674).
_MAX_NESTED_LAYOUT_DEPTH = 32


# #622 (b): the per-view name index below is a cache, so it needs a cache
# INVALIDATION signal that cannot be missed. BN supplies one natively: a
# ``BinaryDataNotification`` subclass registered per view receives the view's own
# change callbacks. The base class is optional -- this module must import cleanly
# with no BN runtime and under the test double, and a view with no notification
# support is simply never cached (every lookup walks, i.e. base behaviour).
try:
    _NotificationBase = bn.BinaryDataNotification
except AttributeError:  # no BN runtime (tests / tooling) or an older core
    _NotificationBase = object


class _ViewChangeState(_NotificationBase):
    """Per-view change counter + memo slots, driven by BN's own notifications.

    Every name-bearing change BN reports bumps ``generation``, which invalidates
    every memo slot for that view at once. Over-invalidation is the SAFE
    direction: a callback fires SYNCHRONOUSLY inside the mutating call (verified
    against a live ``.bndb`` -- assigning ``fn.name`` delivers ``symbol_updated``
    before the assignment returns, ``define_user_symbol`` delivers
    ``symbol_removed`` + ``symbol_updated``, ``update_analysis_and_wait`` delivers
    ``function_updated`` per function), and the bridge serialises reads against
    writes, so a read that follows a mutation cannot observe the pre-mutation
    generation. No callback touches the view: the state holds no reference to it,
    only its own counter and memo slots.

    A callback is written out explicitly per event rather than funnelled through
    ``__getattr__``: BN's dispatch resolves the attribute on the CLASS, so an
    absent method would be an error rather than a catch-all.
    """

    def __init__(self):
        self.generation = 0
        self.values: dict[str, tuple[int, Any]] = {}
        self.lock = threading.Lock()

    def _changed(self):
        with self.lock:
            self.generation += 1

    def symbol_added(self, *_args):
        self._changed()

    def symbol_removed(self, *_args):
        self._changed()

    def symbol_updated(self, *_args):
        self._changed()

    def function_added(self, *_args):
        self._changed()

    def function_removed(self, *_args):
        self._changed()

    def function_updated(self, *_args):
        self._changed()

    def data_var_added(self, *_args):
        self._changed()

    def data_var_removed(self, *_args):
        self._changed()

    def data_var_updated(self, *_args):
        self._changed()

    def data_written(self, *_args):
        self._changed()

    def type_defined(self, *_args):
        self._changed()

    def type_undefined(self, *_args):
        self._changed()

    def rebased(self, *_args):
        self._changed()


# Keyed weakly by the view: the state (and every memo it holds) dies with the
# view's Python wrapper, so a closed target's index is never retained.
_VIEW_STATES: "weakref.WeakKeyDictionary[Any, _ViewChangeState]" = weakref.WeakKeyDictionary()
_VIEW_STATE_LOCK = threading.Lock()


def _view_state(bv) -> _ViewChangeState | None:
    """The per-view change state, registering this module's notifier on first use.

    Returns None when caching would be UNSOUND, and then every caller walks the
    view exactly as before: no BN runtime, no view, a view with no callable
    ``register_notification`` (the test doubles), an unhashable view, or a
    registration that raises. Registration is attempted inside the try and the
    state is stored ONLY on success, so a half-registered state can never be
    handed out.
    """
    if bn is None or bv is None:
        return None
    register = getattr(bv, "register_notification", None)
    if not callable(register):
        return None
    try:
        with _VIEW_STATE_LOCK:
            state = _VIEW_STATES.get(bv)
            if state is not None:
                return state
            state = _ViewChangeState()
            try:
                register(state)
            except Exception:
                return None
            _VIEW_STATES[bv] = state
            return state
    except TypeError:  # unhashable view -> not usable as a weak key
        return None


def _view_memo(bv, slot: str, build):
    """Memoise ``build()`` for *bv*'s CURRENT generation, or call it and cache
    nothing when *bv* has no sound invalidation signal (:func:`_view_state`).

    The value is tagged with the generation it was built at and is only stored if
    that generation still stands after the build -- a change landing mid-build can
    never be tagged fresh, so the next read rebuilds. ``build()`` itself runs
    OUTSIDE the lock: it enumerates the view, and holding the state lock across
    that would serialise unrelated readers for no benefit.
    """
    state = _view_state(bv)
    if state is None:
        return build()
    with state.lock:
        generation = state.generation
        cached = state.values.get(slot)
    if cached is not None and cached[0] == generation:
        return cached[1]
    value = build()
    with state.lock:
        if state.generation == generation:
            state.values[slot] = (generation, value)
    return value


def _view_memo_clear(bv, slot: str) -> None:
    """Drop *slot* from *bv*'s memo so the next lookup rebuilds it.

    The live staleness guards use this when a change BN did NOT notify makes a
    cached bucket disagree with the view: the generation counter is bypassed on
    purpose (it is unchanged by definition) while the memo discipline is kept.
    """
    try:
        state = _VIEW_STATES[bv]
    except (KeyError, TypeError):
        return
    with state.lock:
        state.values.pop(slot, None)


class _NameIndex:
    """Every function's spellings -> the functions carrying them, from ONE pass.

    ``exact``/``folded`` map a spelling (as-is / casefolded) to its bucket, in
    view order; ``corpus`` is every spelling of every function in view order, for
    the "did you mean" hint. Buckets are handed out as NEW lists (:meth:`_bucket`)
    so a caller can never alias -- and therefore never mutate -- the cached index.
    """

    __slots__ = ("exact", "folded", "corpus")

    def __init__(self):
        self.exact: dict[str, list[Any]] = {}
        self.folded: dict[str, list[Any]] = {}
        self.corpus: list[str] = []

    @staticmethod
    def _bucket(mapping: dict[str, list[Any]], key: str) -> list[Any]:
        """The bucket for *key*, deduped by start address in view order, as a NEW
        list (the walk's semantics)."""
        bucket = mapping.get(key)
        if not bucket:
            return []
        out: list[Any] = []
        seen: set[int] = set()
        for fn in bucket:
            marker = int(fn.start)
            if marker in seen:
                continue
            seen.add(marker)
            out.append(fn)
        return out

    def exact_matches(self, text: str) -> list[Any]:
        return self._bucket(self.exact, text)

    def folded_matches(self, text: str) -> list[Any]:
        return self._bucket(self.folded, text.lower())


class BridgeContext:
    """Resolution / ABI / address-context seam over a ``TargetManager``."""

    def __init__(self, targets):
        self.targets = targets

    def _resolve_view(self, selector: str | None):
        return self.targets.resolve(selector)

    def _find_function(self, bv, identifier, *, contained: bool = False):
        # A 0x-prefixed identifier is unambiguously an address attempt (function
        # names never start with 0x), so a parse failure or a miss should report
        # the address problem rather than silently degrading to a name search
        # that ends in a misleading "Function not found".
        #
        # `contained` opts into resolving a mid-function (interior) address to its
        # containing function (#193 Part 4): taint/trace report sinks at
        # instruction addresses, usually inside a callee, and the function-scoped
        # READ verbs should accept them so a sink address feeds straight into the
        # next command. It stays strict by default so a mutation can't rename or
        # retype the wrong (containing) function from a stray interior address.
        looks_like_hex = str(identifier).strip().lower().startswith("0x")
        addr = None
        try:
            addr = _parse_address(identifier)
        except ValueError:
            if looks_like_hex:
                raise RuntimeError(
                    f"Invalid address {identifier!r}: expected a 0x-prefixed hex or decimal value"
                ) from None
        # A value that parsed via `_parse_address` is an address attempt -- either
        # 0x-hex or a bare decimal, both documented address spellings. A decimal
        # interior address must therefore resolve via containment and report an
        # address miss like hex does, NOT silently degrade to a name search that
        # ends in a misleading "Function not found" (#626 review). A malformed
        # token (e.g. "foo") fails to parse -> addr is None -> name path below.
        looks_like_address = addr is not None
        if addr is not None:
            try:
                fn = bv.get_function_at(addr)
            except Exception:
                fn = None
            if fn is not None:
                return fn
            if contained and looks_like_address:
                try:
                    containers = self._functions_containing(bv, addr)
                except Exception:
                    containers = []
                if len(containers) == 1:
                    return containers[0]
                if len(containers) > 1:
                    raise RuntimeError(
                        f"Address {hex(addr)} lies inside multiple overlapping functions; "
                        "pass an exact function start or a name"
                    )
            if looks_like_address:
                raise RuntimeError(f"No function found at address {hex(addr)}")

        text = str(identifier)
        # The native index answers a unique non-stub hit without any walk; on a
        # miss, the per-view name index below serves the exact match, the
        # casefolded match and the "did you mean" corpus from ONE pass -- and a
        # WARM index serves all three without enumerating the view at all
        # (#622(b), see :meth:`_scan_functions_by_name`).
        native = self._native_exact_hit(bv, text)
        if native is not None:
            exact, folded, corpus = [native], [], []
        else:
            exact, folded, corpus = self._scan_functions_by_name(bv, text)
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            resolved = self._resolve_impl_over_stub(exact)
            if resolved is not None:
                return resolved
            raise RuntimeError(_format_ambiguous_function_error(identifier, exact))

        if len(folded) == 1:
            return folded[0]
        if len(folded) > 1:
            resolved = self._resolve_impl_over_stub(folded)
            if resolved is not None:
                return resolved
            raise RuntimeError(_format_ambiguous_function_error(identifier, folded))

        symbol = bv.get_symbol_by_raw_name(text)
        if symbol is not None:
            fn = bv.get_function_at(symbol.address)
            if fn is not None:
                return fn

        suggestions = difflib.get_close_matches(text, corpus, n=5, cutoff=0.5)
        if suggestions:
            raise RuntimeError(
                f"Function not found: {identifier}. Did you mean: {', '.join(suggestions)}"
            )
        raise RuntimeError(f"Function not found: {identifier}")

    def _resolve_impl_over_stub(self, matches: list[Any]):
        """When a name collision is only import/extern stubs shadowing exactly
        ONE real function body, return that body; otherwise None (caller raises
        the ambiguous error). The stub is reliably distinguishable from the
        implementation by ``symbol.type`` (ImportedFunctionSymbol/ExternalSymbol)
        on real BN; a genuine A/B duplicate (two real bodies) stays ambiguous --
        auto-pick must never guess between two real implementations (#122)."""
        impls = [
            fn for fn in matches
            if _symbol_type_name(fn) not in _STUB_SYMBOL_TYPE_NAMES
        ]
        return impls[0] if len(impls) == 1 else None

    def _same_name_stub_functions(self, bv, fn) -> list[Any]:
        """Same-name PLT/extern stub functions that shadow real body *fn*.

        For an exported function in a shared object, intra-library calls route
        through a same-name PLT stub (an ``ImportedFunctionSymbol``/``ExternalSymbol``)
        while the real body shows zero code callers; xrefs and callsites must
        union the stub's callers into the body's (#286). The stub is identified by
        SYMBOL TYPE -- the same stable signal :meth:`_resolve_impl_over_stub`
        trusts. (BN's ``is_thunk`` flag is analysis-timing dependent: it reads
        False before analysis settles, so it must NOT be the criterion here.)
        Returns the stub functions (empty when *fn* is itself the stub, has no
        name, or has no same-name stub sibling)."""
        name = getattr(fn, "name", None)
        if not name:
            return []
        try:
            group = self._find_functions_by_name(bv, str(name), case_sensitive=True)
        except Exception:
            return []
        fn_start = int(getattr(fn, "start", -1))
        # Only union when *fn* is the UNIQUE real definition for this name. With
        # two or more real bodies the stub's forwarding target is ambiguous, so
        # absorbing its callers into one of them would misattribute -- leave that
        # to the existing ambiguous-symbol surfacing (#122) instead.
        impls = [g for g in group if _symbol_type_name(g) not in _STUB_SYMBOL_TYPE_NAMES]
        if len(impls) != 1 or int(getattr(impls[0], "start", -2)) != fn_start:
            return []
        return [
            g for g in group
            if int(getattr(g, "start", -2)) != fn_start
            and _symbol_type_name(g) in _STUB_SYMBOL_TYPE_NAMES
        ]

    @staticmethod
    def _function_name_forms(fn) -> list[str]:
        """Every spelling a function can be addressed by: its display name, its
        raw (mangled) name, and -- crucially for stripped C++ where BN keeps the
        mangled name as ``fn.name`` -- the symbol's demangled ``short_name`` /
        ``full_name``. Lets ``foo::bar::recv`` resolve a function whose
        ``fn.name`` is the mangled ``_ZN3foo3bar4recvEi`` (#224a), uniformly
        across xrefs / callsites / decompile (all route through here).

        Deliberately NOT memoised: reading the four names off a live BinaryView
        costs far more than the de-duplication this does, so a cache keyed on the
        read values measured *slower* than re-reading them (#622)."""
        forms: list[str] = []
        for v in (getattr(fn, "name", None), getattr(fn, "raw_name", None)):
            if v:
                forms.append(str(v))
        sym = getattr(fn, "symbol", None)
        if sym is not None:
            for v in (getattr(sym, "short_name", None), getattr(sym, "full_name", None)):
                if v:
                    forms.append(str(v))
        out: list[str] = []
        for f in forms:
            if f and f not in out:
                out.append(f)
        return out

    def _build_name_index(self, bv) -> _NameIndex:
        """ONE pass over ``list(bv.functions)`` -> the per-view :class:`_NameIndex`.

        Exactly the pass :meth:`_scan_functions_by_name` used to run inline: every
        form of every function goes into the corpus verbatim and into the exact /
        folded buckets, in view order."""
        index = _NameIndex()
        for fn in list(bv.functions):
            forms = self._function_name_forms(fn)
            index.corpus.extend(forms)
            for form in forms:
                index.exact.setdefault(form, []).append(fn)
                index.folded.setdefault(form.lower(), []).append(fn)
        return index

    def _name_index(self, bv, *, refresh: bool = False) -> _NameIndex:
        """The per-view name index: built once per view generation, reused after.

        The index is PER VIEW and dies with the view's Python wrapper (a weak key),
        and it is rebuilt whenever BN reports ANY name-bearing change -- a rename
        makes a bucket stale, so over-invalidating is the safe direction. A view
        whose notification surface is unavailable is NEVER cached (every lookup
        walks, i.e. the behaviour before the index existed). ``refresh=True`` drops
        the memo slot first, for the live staleness guards."""
        if refresh:
            _view_memo_clear(bv, "name_index")
        return _view_memo(bv, "name_index", lambda: self._build_name_index(bv))

    def _bucket_is_live(self, bucket: list[Any], text: str, *, folded: bool) -> bool:
        """Does every member of a cached *bucket* still carry the queried spelling?

        The generation counter is the primary invalidation signal, but a change BN
        does not notify (an unaudited name write) would leave it untouched and let
        the cached bucket serve a spelling a function no longer has. Re-reading the
        members' spellings costs a handful of getattrs for the bucket (usually one
        member) and never a view walk."""
        needle = text.lower()
        for fn in bucket:
            forms = self._function_name_forms(fn)
            if folded:
                if not any(needle == form.lower() for form in forms):
                    return False
            elif text not in forms:
                return False
        return True

    def _scan_functions_by_name(
        self, bv, text: str
    ) -> tuple[list[Any], list[Any], list[str]]:
        """``(exact, folded, corpus)`` for *text* from the per-view name index.

        ``exact`` / ``folded`` are the case-sensitive / casefolded matches,
        deduped by start address in view order -- the authoritative multi-form
        match :meth:`_walk_functions_by_name` computes (the demangled short/full
        spellings live on the symbol, not in BN's name index, #224a). ``corpus``
        is EVERY spelling of EVERY function, in view order: the same list, in the
        same order, that the unbounded base revision handed difflib.

        The corpus is deliberately UNCAPPED -- no function/form cap, no
        first-character filter, no sampling, no length prefilter -- because any
        filter that can drop the intended spelling makes a FIRST-character typo
        lose the "Did you mean" hint, or replace it with an unrelated name that
        happens to share the typo (the round-1 regression). The removed length
        prefilter was also redundant: ``difflib.get_close_matches`` already gates
        every candidate on ``real_quick_ratio`` itself, so the prefilter only
        re-ran that gate -- measured on 20000 synthetic spellings, 0.2231s for the
        plain corpus vs 0.2986s for the prefiltered one (best of 5).

        What is bounded is the COST, not the corpus: the base revision walked
        ``bv.functions`` THREE times per miss (case-sensitive match, casefolded
        match, suggestion corpus) and the round-3 single-pass version still walked
        once per miss, so every lookup paid a full view enumeration. A WARM view
        now needs NO enumeration at all -- the index above is built once per view
        generation and reused, and a miss costs one dict lookup. Measured on a
        synthetic >=100-function view whose ``functions`` list counts
        ``__iter__``: base 3 enumerations per miss, 1 for round-3, 0 warm here
        (#622(b)). The pass COUNT is the bound; the corpus itself stays the full
        spelling set because no cap on it is sound.

        Two live guards keep even a change BN does not notify from serving a stale
        answer, each costing at most ONE extra rebuild per lookup (never a loop):
        a hit's buckets must still be live for the queried spelling, and a
        case-sensitive miss must not be contradicted by BN's own name index -- that
        index is a witness that the spelling exists, never the answer itself (the
        walk-backed rebuild supplies the COMPLETE group, round-3 blocker rule)."""
        index = self._name_index(bv)
        exact = index.exact_matches(text)
        folded = index.folded_matches(text)
        stale = not self._bucket_is_live(exact, text, folded=False)
        if not stale:
            stale = not self._bucket_is_live(folded, text, folded=True)
        if not stale and not exact:
            stale = bool(self._native_functions_by_name(bv, text))
        if stale:
            index = self._name_index(bv, refresh=True)
            exact = index.exact_matches(text)
            folded = index.folded_matches(text)
        return exact, folded, index.corpus

    def _find_functions_by_name(self, bv, text: str, *, case_sensitive: bool) -> list[Any]:
        """The COMPLETE same-name group: every function matching *text*.

        Always walk-backed -- it never consults :meth:`_native_exact_hit`. Its
        callers need the whole group, and BN's own name index can only ever return
        a strict SUBSET of the walk: it does not carry the demangled short/full
        spellings BN keeps only on the symbol (#224a). Answering from that subset
        silently drops a same-name member, which is how `xrefs <name>` once read a
        hot function as zero-caller -- the veneer carrying its call traffic was
        missing from the index, so the stub union had nothing to union
        (#286/#220/#201). The group callers: :meth:`_same_name_stub_functions`
        (the sole input to the xrefs/callsites PLT-veneer caller union), the xrefs
        name-collection path and ``read_xrefs._xrefs_import_symbol``'s body
        lookup, and :meth:`_resolve_rename_target` (which must raise on a genuine
        ambiguity rather than miss a member). Only :meth:`_find_function` --
        single-identifier resolution, where the unique-non-stub-hit trade-off is
        weighed -- may take the native shortcut."""
        return self._walk_functions_by_name(bv, text, case_sensitive=case_sensitive)

    def _native_exact_hit(self, bv, text: str):
        """The single function BN's own name index resolves for *text*, else None.

        Consulted ONLY by :meth:`_find_function`'s single-identifier resolution.
        The GROUP lookup :meth:`_find_functions_by_name` never uses it: its callers
        need every same-name member, for which this index is only ever a strict
        subset (#224a).

        The index is keyed on BN's OWN spellings, while the authoritative walk
        also matches the demangled short/full spellings BN keeps only on the
        symbol (#224a) -- so the index can only ever return a strict SUBSET of
        what the walk would, and a subset is never trusted when it could change
        the answer:

        * a STUB hit is not an answer -- it can be shadowing a same-name real
          body (#122/#286), and only the walk's full set lets
          :meth:`_resolve_impl_over_stub` pick the implementation;
        * several candidates need that same full set so the ambiguous-name error
          is raised instead of one candidate being silently picked.

        Both fall back to the walk. The one residual trade-off, disclosed: a
        unique NON-stub index hit is accepted without the walk, so if two REAL
        bodies shared one spelling through different raw names, this resolves the
        exact-spelling match where a walk-based lookup might have reported the
        ambiguity."""
        matches = self._native_functions_by_name(bv, text)
        if len(matches) == 1 and _symbol_type_name(matches[0]) not in _STUB_SYMBOL_TYPE_NAMES:
            return matches[0]
        return None

    def _native_functions_by_name(self, bv, text: str) -> list[Any]:
        """Functions BN's own name index resolves for *text*, or [] when the index
        cannot answer and the caller must walk.

        BN keeps a name index (``get_functions_by_name`` + the symbol name
        lookups), so an exact-name hit on a large view is answered without
        enumerating every function (#622). Two guards keep this honest:
        ``getattr`` because not every BinaryView-shaped test double implements the
        index, and a re-check of every candidate against the full spelling set
        ``_function_name_forms`` produces -- a native candidate only counts if the
        authoritative walk would have matched it too. An empty result is NOT
        authoritative (the index does not carry the demangled spellings BN keeps
        only on the symbol, #224a) and falls back to the walk."""
        candidates: list[Any] = []
        getter = getattr(bv, "get_functions_by_name", None)
        if callable(getter):
            candidates.extend(getter(text) or [])
        symbol_lookups = []
        for attr in ("get_symbols_by_name", "get_symbol_by_raw_name"):
            lookup = getattr(bv, attr, None)
            if callable(lookup):
                symbol_lookups.append(lookup(text))
        for found in symbol_lookups:
            for symbol in (found if isinstance(found, (list, tuple)) else [found]):
                if symbol is None:
                    continue
                get_at = getattr(bv, "get_function_at", None)
                fn = get_at(int(getattr(symbol, "address", 0))) if callable(get_at) else None
                if fn is not None:
                    candidates.append(fn)
        matches: list[Any] = []
        seen: set[int] = set()
        for fn in candidates:
            if text not in self._function_name_forms(fn):
                continue
            marker = int(getattr(fn, "start", 0))
            if marker in seen:
                continue
            seen.add(marker)
            matches.append(fn)
        return matches

    def _walk_functions_by_name(self, bv, text: str, *, case_sensitive: bool) -> list[Any]:
        """The authoritative multi-form match: exact, or casefolded, over every
        function's spellings. Needs the walk because the demangled short/full
        names live on the symbol, not in BN's name index (#224a).

        Thin wrapper over :meth:`_scan_functions_by_name`, so the index is built at
        most once whether the caller wants one of the two match sets or both; the
        result is unchanged from the per-mode walk this replaced."""
        exact, folded, _ = self._scan_functions_by_name(bv, text)
        return exact if case_sensitive else folded

    def _resolve_scope_functions(self, bv, identifiers: list[Any]) -> list[tuple[str, Any]]:
        if not identifiers:
            raise OperationFailure("invalid_scope", "callsites requires at least one scoped function")

        resolved = []
        seen: set[int] = set()
        for identifier in identifiers:
            fn = self._find_function(bv, identifier)
            marker = int(fn.start)
            if marker in seen:
                continue
            seen.add(marker)
            resolved.append((str(identifier), fn))
        return resolved

    def _find_symbols_by_name(self, bv, text: str, *, case_sensitive: bool) -> list[Any]:
        matches = []
        seen: set[tuple[int, str]] = set()

        if case_sensitive:
            candidates = list(bv.get_symbols_by_name(text))
            raw_match = bv.get_symbol_by_raw_name(text)
            if raw_match is not None:
                candidates.append(raw_match)
        else:
            folded = text.lower()
            candidates = []
            for symbol in list(bv.get_symbols()):
                names = [str(getattr(symbol, "name", "")), str(getattr(symbol, "raw_name", ""))]
                if folded in {name.lower() for name in names if name}:
                    candidates.append(symbol)

        for symbol in candidates:
            marker = (int(symbol.address), str(symbol.type))
            if marker in seen:
                continue
            seen.add(marker)
            matches.append(symbol)
        return matches

    def _resolve_rename_target(self, bv, identifier: Any, kind: str) -> dict[str, Any]:
        requested = {
            "kind": kind,
            "identifier": str(identifier),
        }

        try:
            address = _parse_address(identifier)
        except Exception:
            address = None

        if address is not None:
            fn = bv.get_function_at(address)
            symbol = bv.get_symbol_at(address)
            if kind == "function":
                if fn is None:
                    raise OperationFailure("unsupported", f"Function not found: {identifier}", requested=requested)
                return {
                    "kind": "function",
                    "address": int(fn.start),
                    "before_name": str(fn.name),
                }
            if kind == "data":
                return {
                    "kind": "data",
                    "address": int(address),
                    "before_name": str(symbol.name) if symbol is not None else None,
                }
            if fn is not None:
                return {
                    "kind": "function",
                    "address": int(fn.start),
                    "before_name": str(fn.name),
                }
            return {
                "kind": "data",
                "address": int(address),
                "before_name": str(symbol.name) if symbol is not None else None,
            }

        if kind in {"auto", "function"}:
            exact_functions = self._find_functions_by_name(bv, str(identifier), case_sensitive=True)
            resolved = exact_functions[0] if len(exact_functions) == 1 else (
                self._resolve_impl_over_stub(exact_functions) if len(exact_functions) > 1 else None
            )
            if resolved is not None:
                return {
                    "kind": "function",
                    "address": int(resolved.start),
                    "before_name": str(resolved.name),
                }
            if len(exact_functions) > 1:
                raise OperationFailure(
                    "unsupported",
                    _format_ambiguous_function_error(identifier, exact_functions),
                    requested=requested,
                )

            folded_functions = self._find_functions_by_name(bv, str(identifier), case_sensitive=False)
            resolved = folded_functions[0] if len(folded_functions) == 1 else (
                self._resolve_impl_over_stub(folded_functions) if len(folded_functions) > 1 else None
            )
            if resolved is not None:
                return {
                    "kind": "function",
                    "address": int(resolved.start),
                    "before_name": str(resolved.name),
                }
            if len(folded_functions) > 1:
                raise OperationFailure(
                    "unsupported",
                    _format_ambiguous_function_error(identifier, folded_functions),
                    requested=requested,
                )

        if kind == "function":
            raise OperationFailure("unsupported", f"Function not found: {identifier}", requested=requested)

        exact_symbols = [
            symbol
            for symbol in self._find_symbols_by_name(bv, str(identifier), case_sensitive=True)
            if symbol.type != bn.SymbolType.FunctionSymbol
        ]
        if len(exact_symbols) == 1:
            symbol = exact_symbols[0]
            return {
                "kind": "data",
                "address": int(symbol.address),
                "before_name": str(symbol.name),
            }
        if len(exact_symbols) > 1:
            raise OperationFailure(
                "unsupported",
                _format_ambiguous_symbol_error(identifier, exact_symbols),
                requested=requested,
            )

        folded_symbols = [
            symbol
            for symbol in self._find_symbols_by_name(bv, str(identifier), case_sensitive=False)
            if symbol.type != bn.SymbolType.FunctionSymbol
        ]
        if len(folded_symbols) == 1:
            symbol = folded_symbols[0]
            return {
                "kind": "data",
                "address": int(symbol.address),
                "before_name": str(symbol.name),
            }
        if len(folded_symbols) > 1:
            raise OperationFailure(
                "unsupported",
                _format_ambiguous_symbol_error(identifier, folded_symbols),
                requested=requested,
            )

        raise OperationFailure("unsupported", f"Symbol not found: {identifier}", requested=requested)

    def _functions_containing(self, bv, address: int):
        try:
            return list(bv.get_functions_containing(address))
        except Exception:
            fn = bv.get_function_at(address)
            return [fn] if fn is not None else []

    def _functions_referencing(self, bv, address: int, *, limit: int | None = None):
        """Distinct functions carrying a code xref to *address* -- the blast radius of
        a data-variable retype, since typing a global reflows every reader (#649).
        Best-effort: an empty list when the view exposes no code-ref API."""
        get_refs = getattr(bv, "get_code_refs", None)
        if not callable(get_refs):
            return []
        try:
            refs = list(get_refs(int(address)) or [])
        except Exception:
            return []
        out: list[Any] = []
        seen: set[int] = set()
        for ref in refs:
            fn = getattr(ref, "function", None)
            if fn is None:
                continue
            start = int(getattr(fn, "start", -1) or -1)
            if start in seen:
                continue
            seen.add(start)
            out.append(fn)
            if limit is not None and len(out) >= limit:
                break
        return out

    def _containment_meta(self, identifier, func):
        """Describe address resolution for function-scoped reads.

        Interior hex/decimal inputs carry the normalized address and function
        offset. Exact bare-decimal inputs are also disclosed with offset zero so
        a digit-only token can never be silently mistaken for a symbol name.
        Exact ``0x`` starts and ordinary names need no annotation.
        """
        try:
            addr = _parse_address(identifier)
        except ValueError:
            return None
        if addr is None:
            return None
        decimal_input = (
            isinstance(identifier, str)
            and identifier.strip().isdigit()
        )
        start = int(func.start)
        if int(addr) == start and not decimal_input:
            return None
        delta = int(addr) - start
        sign = "+" if delta >= 0 else "-"
        result = {
            "requested_address": hex(int(addr)),
            "offset": f"{sign}{hex(abs(delta))}",
        }
        if decimal_input:
            result["input_format"] = "decimal"
        return result

    def _sections_at(self, bv, address: int) -> list[dict[str, Any]]:
        try:
            sections = list(bv.get_sections_at(address))
        except Exception:
            sections = []
            for name, sec in getattr(bv, "sections", {}).items():
                try:
                    if int(sec.start) <= address < int(sec.end):
                        sections.append(sec)
                except Exception:
                    continue

        result = []
        for sec in sections:
            try:
                start = int(getattr(sec, "start", 0))
                end = int(getattr(sec, "end", 0))
            except Exception:
                start = end = 0
            result.append(
                {
                    "name": str(getattr(sec, "name", "")),
                    "start": hex(start),
                    "end": hex(end),
                }
            )
        return result

    def _segment_at(self, bv, address: int) -> dict[str, Any] | None:
        try:
            seg = bv.get_segment_at(address)
        except Exception:
            seg = None
        if seg is None:
            return None
        entry: dict[str, Any] = {
            "readable": bool(getattr(seg, "readable", False)),
            "writable": bool(getattr(seg, "writable", False)),
            "executable": bool(getattr(seg, "executable", False)),
        }
        for attr in ("start", "end"):
            value = getattr(seg, attr, None)
            if value is not None:
                try:
                    entry[attr] = hex(int(value))
                except Exception:
                    pass
        return entry

    def _symbol_at(self, bv, address: int) -> dict[str, Any] | None:
        try:
            symbol = bv.get_symbol_at(address)
        except Exception:
            symbol = None
        if symbol is None:
            return None
        raw_type = getattr(symbol, "type", None)
        kind = getattr(raw_type, "name", None) or str(raw_type)
        return {
            "name": str(getattr(symbol, "name", "")),
            "raw_name": str(getattr(symbol, "raw_name", getattr(symbol, "name", ""))),
            "type": kind,
        }

    def _function_entry_for_address(self, bv, address: int) -> dict[str, Any] | None:
        try:
            fn = bv.get_function_at(address)
        except Exception:
            fn = None
        if fn is None:
            functions = self._functions_containing(bv, address)
            fn = functions[0] if functions else None
        if fn is None:
            return None
        function_start = int(fn.start)
        entry = {
            "name": str(fn.name),
            "address": hex(function_start),
            "exact_start": function_start == int(address),
        }
        if function_start != int(address):
            delta = int(address) - function_start
            entry["offset"] = f"-{hex(abs(delta))}" if delta < 0 else hex(delta)
        return entry

    def _raw_sections_at(self, bv, address: int) -> list[Any]:
        try:
            return list(bv.get_sections_at(address))
        except Exception:
            result = []
            for _name, sec in getattr(bv, "sections", {}).items():
                try:
                    if int(sec.start) <= address < int(sec.end):
                        result.append(sec)
                except Exception:
                    continue
            return result

    def _section_semantics_name(self, sec) -> str:
        sem = getattr(sec, "semantics", None)
        return getattr(sem, "name", None) or str(sem)

    def _address_is_code(self, bv, address: int) -> bool:
        """True only when the address is real code.

        Keys on function membership and section *semantics* (ReadOnlyCode), not
        the segment's executable bit — firmware ELFs routinely map .rodata into
        the same r-x load segment as .text, so an executable segment is not
        evidence that an address is an instruction.
        """
        if self._functions_containing(bv, address):
            return True
        for sec in self._raw_sections_at(bv, address):
            if "Code" in self._section_semantics_name(sec):
                return True
        return False

    def _resolve_data_string(self, bv, address: int, *, max_chars: int = 96) -> dict[str, Any] | None:
        """Best-effort printable string at *address*, even when BN never
        atomized one there (e.g. single chars packed for std::string::append).

        Tries a NUL-terminated ASCII run first, then UTF-16LE. Common escaped
        text controls are allowed. Long strings are capped and marked
        truncated so evidence output stays compact. Returns None for non-string
        bytes so it can be used as a cheap "is this a string?" probe.
        """
        try:
            data = bytes(bv.read(int(address), max_chars * 2 + 2))
        except Exception:
            return None
        if not data:
            return None

        allowed_ascii = set(range(32, 127)) | {9, 10, 13}
        ascii_chars: list[str] = []
        for byte in data:
            if byte == 0:
                if ascii_chars:
                    return {
                        "value": "".join(ascii_chars),
                        "encoding": "ascii",
                        "truncated": False,
                    }
                break
            if byte not in allowed_ascii:
                ascii_chars = []
                break
            if len(ascii_chars) >= max_chars:
                return {
                    "value": "".join(ascii_chars),
                    "encoding": "ascii",
                    "truncated": True,
                }
            ascii_chars.append(chr(byte))
        else:
            if ascii_chars:
                return {
                    "value": "".join(ascii_chars),
                    "encoding": "ascii",
                    "truncated": True,
                }

        if ascii_chars:
            return {
                "value": "".join(ascii_chars),
                "encoding": "ascii",
                "truncated": True,
            }

        chars: list[str] = []
        index = 0
        terminated = False
        allowed_wide = allowed_ascii
        while index + 1 < len(data) and len(chars) < max_chars:
            lo, hi = data[index], data[index + 1]
            if lo == 0 and hi == 0:
                terminated = True
                break
            if hi != 0 or lo not in allowed_wide:
                chars = []
                break
            chars.append(chr(lo))
            index += 2
        if (
            len(chars) >= max_chars
            and index + 1 < len(data)
            and data[index] == 0
            and data[index + 1] == 0
        ):
            terminated = True
        if len(chars) >= 2:
            return {
                "value": "".join(chars),
                "encoding": "utf-16le",
                "truncated": not terminated and len(chars) >= max_chars,
            }
        return None

    def _address_context(self, bv, address: int, *, include_disasm: bool = False, arch=None,
                         assume_code: bool = False) -> dict[str, Any]:
        address = int(address)
        sections = self._sections_at(bv, address)
        segment = self._segment_at(bv, address)
        symbol = self._symbol_at(bv, address)
        function = self._function_entry_for_address(bv, address)
        context: dict[str, Any] = {
            "address": hex(address),
            "sections": sections,
            "segment": segment,
            "symbol": symbol,
            "function": function,
        }
        section_name = sections[0]["name"].lower() if sections else ""
        symbol_type = (symbol or {}).get("type") or ""
        if address == 0:
            kind = "null"
        elif segment is None and not sections:
            kind = "unmapped"
        elif assume_code or function is not None or self._address_is_code(bv, address):
            kind = "code"
        elif symbol_type == "ExternalSymbol" or "extern" in section_name:
            kind = "extern"
        else:
            resolved = self._resolve_data_string(bv, address)
            if resolved is not None:
                context["string"] = resolved
                kind = "string"
            else:
                kind = "data"
        context["kind"] = kind
        if include_disasm:
            if kind == "code":
                disasm_arch = arch
                if disasm_arch is None:
                    # Disassemble with the TARGET function's own arch, not the
                    # BinaryView default: in a mixed ARM/THUMB binary the default
                    # (ARM) misdecodes a THUMB2 target into a fabricated
                    # instruction. The function-start xref target is the common
                    # case here (#53).
                    disasm_arch = getattr(self._function_object_at(bv, address), "arch", None)
                context["disasm"] = self._safe_disassembly(bv, address, disasm_arch)
            else:
                context["disasm"] = None
                context["notes"] = [f"target is {kind}; disassembly suppressed"]
        return context

    def _function_object_at(self, bv, address: int):
        """The function whose body covers *address* (entry first), or None. Used
        to disassemble with the right per-function arch in a mixed-ISA binary."""
        getfn = getattr(bv, "get_function_at", None)
        fn = getfn(int(address)) if callable(getfn) else None
        if fn is not None:
            return fn
        containing = self._functions_containing(bv, int(address))
        return containing[0] if containing else None

    def _safe_disassembly(self, bv, address: int, arch=None) -> str:
        for args in ((address, arch) if arch is not None else (), (address,)):
            try:
                return bv.get_disassembly(*args) or ""
            except Exception:
                continue
        return ""

    def _pointer_size(self, bv) -> int:
        for obj in (bv, getattr(bv, "arch", None)):
            value = getattr(obj, "address_size", None)
            if value is None:
                continue
            try:
                size = int(value)
                if size > 0:
                    return size
            except Exception:
                pass
        return 4

    def _byteorder(self, bv) -> str:
        for obj in (bv, getattr(bv, "arch", None)):
            value = getattr(obj, "endianness", None)
            if value is None:
                continue
            # BN's `Endianness` is an IntEnum, and since Python 3.11 `str()` on an
            # IntEnum member yields the *number* ("1"), not "Endianness.BigEndian" --
            # so a substring test on `str(value)` can never see a big-endian view and
            # every BE target silently decodes little-endian. Classify by the enum's
            # name (BN's spelling), then by its integer value (BigEndian == 1), and
            # keep the text path last for plain-string callers/stubs.
            name = getattr(value, "name", None)
            if isinstance(name, str) and name:
                return "big" if "big" in name.lower() else "little"
            if isinstance(value, int) and not isinstance(value, bool):
                return "big" if int(value) == 1 else "little"
            if "big" in str(value).lower():
                return "big"
        return "little"

    def _supports_thumb_pointer_tags(self, bv) -> bool:
        if self._pointer_size(bv) != 4:
            return False
        names = []
        arch = getattr(bv, "arch", None)
        for obj in (arch, getattr(bv, "platform", None)):
            if obj is None:
                continue
            for attr in ("name", "raw_name"):
                value = getattr(obj, attr, None)
                if value:
                    names.append(str(value).lower())
            names.append(str(obj).lower())
        joined = " ".join(names)
        if "aarch64" in joined or "arm64" in joined:
            return False
        return "thumb" in joined or "arm" in joined

    def _read_pointer_value(self, bv, address: int, *, size: int | None = None) -> int | None:
        pointer_size = size or self._pointer_size(bv)
        try:
            data = bytes(bv.read(address, pointer_size))
        except Exception:
            return None
        if len(data) != pointer_size:
            return None
        return int.from_bytes(data, self._byteorder(bv), signed=False)

    def _normalize_code_pointer(self, bv, value: int) -> dict[str, Any]:
        raw = int(value)
        normalized = raw
        thumb_adjusted = False
        if raw & 1 and self._supports_thumb_pointer_tags(bv):
            candidate = raw & ~1
            candidate_function = self._function_entry_for_address(bv, candidate)
            if candidate_function is not None:
                normalized = candidate
                thumb_adjusted = True
                function = candidate_function
            else:
                function = self._function_entry_for_address(bv, normalized)
        else:
            function = self._function_entry_for_address(bv, normalized)
        context = self._address_context(bv, normalized, include_disasm=bool(function))
        segment = context.get("segment")
        status = "function" if function is not None else "mapped" if segment is not None else "null" if raw == 0 else "unmapped"
        plausible = status in {"function", "mapped", "null"}
        return {
            "raw": hex(raw),
            "normalized": hex(normalized),
            "thumb_adjusted": thumb_adjusted,
            "function": function,
            "status": status,
            "plausible": plausible,
            "context": context,
        }

    def _pointer_table_layout(self, bv, start, *, entries, stride):
        from . import read_evidence
        return read_evidence._pointer_table_for_view(
            self, bv, start, entries=entries, stride_size=stride
        )

    def _operator_new_size_at_ctor(self, bv, record):
        """(size, addr) of the operator-new allocation feeding a ctor's ``this``,
        else None. TODO(#205 Task 9): implement against live BN — backward-slice
        the ctor call's arg0 to its ``operator new(N)`` allocation and read N.
        Returns None for now (honest "unknown"); the BN-type-width path in
        _object_size still yields real sizes when a type is defined."""
        return None

    def _read_u32(self, bv, address):
        try:
            data = bytes(bv.read(address, 4))
        except Exception:
            return None
        return int.from_bytes(data, self._byteorder(bv), signed=False) if len(data) == 4 else None

    def _typeinfo_name_at(self, bv, address):
        """Class name for a _ZTI object address from its data symbol's demangled
        RTTI marker; None if unresolved. Reuses read_class's marker matcher so
        the space/underscore spelling variants are handled in one place."""
        sym = bv.get_symbol_at(address) if hasattr(bv, "get_symbol_at") else None
        if sym is None:
            return None
        from . import read_class
        return read_class._class_of_rtti_symbol(sym)

    def _global_vtable_stores(self, bv, record):
        """Global data symbols whose stored value is this class's vtable addr."""
        vt = record.get("vtable")
        if not vt:
            return []
        addr = int(vt["address"], 16)
        out = []
        getter = getattr(bv, "get_data_refs", None)
        for ref in (getter(addr) if callable(getter) else []):
            sym = bv.get_symbol_at(int(ref)) if hasattr(bv, "get_symbol_at") else None
            out.append({
                "symbol": str(getattr(sym, "short_name", "") or getattr(sym, "name", "")) if sym else None,
                "address": hex(int(ref)),
            })
        return out

    def _ctor_construction_sites(self, bv, record, *, cap=128):
        """Where this class is constructed: each ctor's inbound call sites, from
        code xrefs. This is sound (no arg recovery needed). Classifying the
        `this` storage (new/stack/global) and recovering the operator-new size
        needs MLIL arg recovery that BN often does not expose at these sites, so
        it is left as a best-effort gap: kind is "ctor-call" and size is None."""
        get_refs = getattr(bv, "get_code_refs", None)
        if not callable(get_refs):
            return []
        sites: list[dict[str, Any]] = []
        seen: set[int] = set()
        for m in record.get("methods", []):
            if m.get("kind") != "ctor":
                continue
            try:
                addr = int(m["address"], 16)
            except (KeyError, ValueError, TypeError):
                continue
            for ref in get_refs(addr):
                ra = int(getattr(ref, "address", 0) or 0)
                if ra == 0 or ra in seen:
                    continue
                seen.add(ra)
                caller = getattr(ref, "function", None)
                csym = getattr(caller, "symbol", None) if caller is not None else None
                cname = (getattr(csym, "short_name", None) or getattr(caller, "name", None)) \
                    if caller is not None else None
                sites.append({
                    "address": hex(ra),
                    "function": str(cname) if cname else None,
                    "kind": "ctor-call",
                    "size": None,
                })
                if len(sites) >= cap:
                    return sites
        return sites

    def _vtable_layout_for(self, bv, addr):
        from . import read_class
        return read_class._vtable_layout(self, bv, addr)

    def _object_size_for(self, bv, record):
        from . import read_class
        return read_class._object_size(self, bv, record)

    def _bases_for(self, bv, record):
        from . import read_class
        ti = record.get("typeinfo")
        if not ti:
            return []
        return read_class._rtti_bases(self, bv, int(ti["address"], 16))

    def _instances_for(self, bv, record):
        from . import read_class
        return read_class._instances(self, bv, record)

    # ---- relocated cycle-breakers (design spec §3.2): both state-free ----

    def _find_type(self, bv, type_name: str):
        type_obj = bv.get_type_by_name(type_name)
        if type_obj is not None:
            return type_name, type_obj

        needle = str(type_name).lower()
        available: list[str] = []
        for name, candidate in list(bv.types.items()):
            if str(name).lower() == needle:
                return str(name), candidate
            available.append(str(name))

        suggestions = difflib.get_close_matches(str(type_name), available, n=5, cutoff=0.5)
        # Always point at the substring search, whether or not difflib found a
        # near-miss. The common dead-end is a missing primitive typedef on a
        # target that defines the underlying type under another name (`uint32_t`
        # vs `unsigned int`), and on a real target difflib readily returns
        # UNRELATED `_t` typedefs as "close" (`uint32_t` -> wint_t, off64_t) --
        # so a hint gated on having NO close match never fires for exactly the
        # case it's meant to help. Drop a trailing `_t` so the query is the
        # typedef root (`uint32`, `size`) that matches the names a target
        # actually carries; keep the full name when there's no meaningful root.
        name = str(type_name)
        query = name[:-2] if name.endswith("_t") and len(name) > 2 else name
        search_hint = f"`bn types --query {query}` to search available types"
        if suggestions:
            raise RuntimeError(
                f"Type not found: {name}. Did you mean: {', '.join(suggestions)}. "
                f"Or try {search_hint}."
            )
        raise RuntimeError(
            f"Type not found: {name}. No similar type names; try {search_hint}."
        )

    @staticmethod
    def _is_anonymous_aggregate(member_type) -> bool:
        """True if *member_type* is a nested struct/union with NO tag name (so its
        inner members are unreachable via a separate `struct show <name>`). A bare
        `union`/`struct` (optionally `{...}`) is anonymous; `struct Inner` is not
        (#370.2)."""
        if not getattr(member_type, "members", None):
            return False
        m = re.match(r"^(struct|union|enum)\b(.*)$", str(member_type).strip())
        if not m:
            return False
        rest = m.group(2).strip()
        return rest == "" or rest.startswith("{")

    def _render_type_layout(self, type_obj) -> str:
        header = str(type_obj)
        try:
            width = int(getattr(type_obj, "width", 0))
            header = f"{header} // size=0x{width:x}"
        except Exception:
            pass

        members = getattr(type_obj, "members", None)
        if members is None:
            return header

        tc = str(getattr(getattr(type_obj, "type_class", None), "name", "") or "")
        lines = [header]
        self._append_member_lines(lines, members, "Enum" in tc, depth=0)
        return "\n".join(lines)

    def _append_member_lines(self, lines: list, members, is_enum: bool, *, depth: int) -> None:
        # Enum members carry a .value (the enumerator constant) but no .offset/
        # .type, so the struct-shaped line collapses every one to
        # "0x0000: <unknown> NAME", dropping the only meaningful datum. Render the
        # value instead for enums (#54). An anonymous nested aggregate is expanded
        # (indented) so its inner members are visible from the CLI (#370.2).
        pad = "  " * depth
        for member in list(members):
            name = str(getattr(member, "name", "<anonymous>"))
            value = getattr(member, "value", None)
            if is_enum or (getattr(member, "offset", None) is None and value is not None):
                try:
                    ival = int(value)
                    suffix = f" (0x{ival:x})" if ival >= 0 else ""
                    lines.append(f"{pad}{name} = {ival}{suffix}")
                except Exception:
                    lines.append(f"{pad}{name} = {value}")
            else:
                try:
                    offset = int(getattr(member, "offset", 0))
                except Exception:
                    offset = 0
                member_type = getattr(member, "type", None)
                lines.append(f"{pad}0x{offset:04x}: {member_type if member_type is not None else '<unknown>'} {name}")
                if self._is_anonymous_aggregate(member_type):
                    if depth < _MAX_NESTED_LAYOUT_DEPTH:
                        itc = str(getattr(getattr(member_type, "type_class", None), "name", "") or "")
                        self._append_member_lines(
                            lines, member_type.members, "Enum" in itc, depth=depth + 1)
                    else:
                        lines.append(
                            f"{pad}  ... truncated at depth {depth + 1}: further nested members omitted")

    def _member_entries(self, type_obj, depth: int = 0) -> list[dict] | None:
        """Structured members[] for the JSON readback, recursing into anonymous
        aggregates so their inner members aren't invisible (the JSON counterpart of
        the text expansion, #370.2). None for a non-aggregate type."""
        members = getattr(type_obj, "members", None)
        if members is None:
            return None
        out: list[dict] = []
        for member in list(members):
            entry: dict = {"name": str(getattr(member, "name", "<anonymous>"))}
            value = getattr(member, "value", None)
            offset = getattr(member, "offset", None)
            if offset is not None:
                try:
                    entry["offset"] = hex(int(offset))
                except Exception:
                    pass
            mtype = getattr(member, "type", None)
            if mtype is not None:
                entry["type"] = str(mtype)
            if value is not None and offset is None:
                entry["value"] = int(value) if isinstance(value, int) else str(value)
            if self._is_anonymous_aggregate(mtype):
                if depth < _MAX_NESTED_LAYOUT_DEPTH:
                    inner = self._member_entries(mtype, depth + 1)
                    if inner:
                        entry["members"] = inner
                else:
                    entry["truncated"] = True
            out.append(entry)
        return out

    def _type_entry(self, type_name, type_obj):
        type_class = getattr(type_obj, "type_class", None)
        kind = "unknown"
        if type_class is not None:
            try:
                kind = _TYPE_CLASS_NAMES.get(int(type_class), str(type_class))
            except (TypeError, ValueError):
                kind = str(type_class)
        entry = {
            "name": str(type_name),
            "kind": kind,
            "decl": str(type_obj),
            "layout": self._render_type_layout(type_obj),
        }
        members = self._member_entries(type_obj)
        if members is not None:
            entry["members"] = members
        return entry

    def _current_type_entry(self, bv, type_name: str):
        type_obj = bv.get_type_by_name(type_name)
        if type_obj is None:
            return None
        return self._type_entry(type_name, type_obj)
