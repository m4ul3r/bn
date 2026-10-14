"""Mutable view-tracking module globals shared across the bridge.

These globals track headless-loaded BinaryViews, the subset loaded with
``--quick`` (analysis not yet run), and per-view load-time verdicts. They were
hoisted out of ``bridge.py`` into this neutral module so that read-op domain
modules (e.g. ``read_misc``,
which reads ``_quick_loaded_views``) can consult them WITHOUT importing
``bridge.py`` (which would create an import cycle).

``bridge.py`` re-imports these same names, so ``bridge._headless_views`` and
``bridge._quick_loaded_views`` resolve to the SAME objects. Every importer must
share one object: the load/close/target_info/refresh handlers mutate these in
place (``.append`` / ``.pop`` / ``.clear`` / ``[:] =`` / ``.add`` / ``.discard``
/ ``in``), and tests mutate ``bridge._headless_views`` in place too. This module
imports ONLY stdlib -- never bridge.
"""
from __future__ import annotations

import threading
import time
import weakref
from typing import Any

_headless_views: list[Any] = []
_headless_views_lock = threading.Lock()
# Views loaded with --quick (analysis not run yet). Strings/full function set
# are unavailable until `bn refresh`, so commands consult this to stay honest
# instead of returning a misleading empty result. Weak so closed views drop out.
_quick_loaded_views: "weakref.WeakSet[Any]" = weakref.WeakSet()
# Views restored from a .bndb that had no saved analyzed product view -- a raw
# container with no functions/symbols (#458). Distinct from --quick: `bn refresh`
# on a raw-only image will not recover a format view, so these are reported as
# `analysis_state="unanalyzed"` rather than "quick". Weak so closed views drop out.
_unanalyzed_views: "weakref.WeakSet[Any]" = weakref.WeakSet()
# File session id (`bv.file.session_id`, see `view_session_key`) -> whether the
# database the view was restored from was VERIFIED against the raw file's
# content at load (True), or restored UNVERIFIED (False: no record and its
# stored bytes could not vouch for it; or an explicitly named .bndb whose record
# names other bytes than its sibling raw file now holds). Absent for a view no
# database was judged for (a raw load, a database with nothing to judge it
# against), so `target list`/`target info` publish null there. The load note is
# one-shot; this is what a later `target info` answers from.
#
# Keyed on the session id, NOT the wrapper: in GUI mode a tab reaches the
# already-open judge in `_load_binary` as a fresh, handle-equal wrapper that
# dies when the call returns, so a weak-keyed entry on it was gone by the next
# `target list`. A plain dict that `TargetManager.refresh()` prunes to the open
# set (on a complete snapshot) outlives any wrapper and drops closed views.
_source_verified: dict[str, bool] = {}
# File session id -> what the view carried when THIS bridge opened it (#973):
# None for a view opened from raw bytes, which cannot have inherited an
# annotation; for a view opened from a database, the annotation counts it held
# at open (`read_listing._annotation_summary`'s comments / function_comments /
# analyst_symbols / placeholder_symbols) plus the database's ``path``. Absent
# for a view the bridge did not open (a GUI tab) or whose counts could not be
# read at open, so the provenance hint keeps judging those by filename and
# current counts. Recorded once, before publish, so annotations written by
# this session -- and a save that re-homes the view -- never count as
# inherited. Pruned with `_source_verified`.
_opened_annotations: dict[str, dict[str, Any] | None] = {}
# Views a `bn refresh` is analysing right now -> monotonic start time (#996).
# The --quick marker stays set until that analysis returns, so without this a
# gated read told the agent to "Run `bn refresh`" while one was already running
# -- starting a second multi-minute analysis. Weak so a closed view drops out.
_refreshing_views: "weakref.WeakKeyDictionary[Any, float]" = weakref.WeakKeyDictionary()


def view_session_key(bv: Any) -> str:
    """The key `_source_verified` and the target rows share for *bv*: BN's
    file session id, which every wrapper of one open file reports alike; the
    wrapper's own identity when the view has none (a fake, a dead handle)."""
    try:
        return str(bv.file.session_id)
    except Exception:  # noqa: BLE001 - no file / no session id: fall back to identity
        return str(id(bv))


def _progress_fragment(bv: Any) -> str:
    """`` (N/M)`` from BN's live analysis progress, or "" when it says nothing."""
    try:
        ap = bv.analysis_progress
        count = int(getattr(ap, "count", 0) or 0)
        total = int(getattr(ap, "total", 0) or 0)
    except Exception:  # noqa: BLE001 - not all view types expose progress
        return ""
    return f" ({count}/{total})" if total else ""


def analysis_unavailable(
    bv: Any, what: str, directive: str, *, plural: bool = False
) -> str | None:
    """Why an analysis-dependent read cannot answer on *bv* yet, or None.

    *directive* is the "Run `bn refresh` ..." sentence for a view nobody is
    analysing. While a `bn refresh` is running on it the message says so and
    tells the caller to retry instead -- never to start another (#996). Both
    forms keep the literal "loaded with --quick" callers match on. *plural*
    picks "are" for a plural *what* ("Strings are not available").
    """
    if bv not in _quick_loaded_views:
        return None
    verb = "are" if plural else "is"
    started = _refreshing_views.get(bv)
    if started is not None:
        elapsed = max(0, int(time.monotonic() - started))
        return (
            f"{what} {verb} not available yet: this target was loaded with --quick and "
            f"full analysis is in progress{_progress_fragment(bv)}, started "
            f"{elapsed}s ago by `bn refresh`. Retry when it finishes -- do not "
            "start another refresh."
        )
    return (
        f"{what} {verb} not available: this target was loaded with --quick "
        f"(no analysis). {directive}"
    )


def require_analysis(bv: Any, what: str = "This operation") -> None:
    """Refuse an analysis-dependent read on a ``--quick``-loaded (unanalyzed) view.

    Analysis-dependent ops (xrefs, function info, callsites, taint, strings) would
    otherwise return a misleading empty/zero result -- or a wrong "not found" /
    "no call to <sink>" error that blames a typo -- on a view whose analysis has
    not run. Refuse with a directive pointing at ``bn refresh`` instead, so a cold
    agent isn't silently led to the wrong conclusion -- or, while a refresh is
    already running, with "retry when it finishes" (#996). Message contains the
    literal "loaded with --quick" so callers can match on it. No-op once analysis
    has run (the view is discarded from ``_quick_loaded_views`` by ``bn refresh``).
    """
    message = analysis_unavailable(
        bv, what, "Run `bn refresh` to run analysis first.")
    if message is not None:
        raise RuntimeError(message)
