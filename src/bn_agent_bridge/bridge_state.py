"""Mutable view-tracking module globals shared across the bridge.

These three globals track headless-loaded BinaryViews and the subset loaded
with ``--quick`` (analysis not yet run). They were hoisted out of ``bridge.py``
into this neutral module so that read-op domain modules (e.g. ``read_misc``,
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


def require_analysis(bv: Any, what: str = "This operation") -> None:
    """Refuse an analysis-dependent read on a ``--quick``-loaded (unanalyzed) view.

    Analysis-dependent ops (xrefs, function info, callsites, taint, strings) would
    otherwise return a misleading empty/zero result -- or a wrong "not found" /
    "no call to <sink>" error that blames a typo -- on a view whose analysis has
    not run. Refuse with a directive pointing at ``bn refresh`` instead, so a cold
    agent isn't silently led to the wrong conclusion. Message contains the literal
    "loaded with --quick" so callers can match on it. No-op once analysis has run
    (the view is discarded from ``_quick_loaded_views`` by ``bn refresh``).
    """
    if bv in _quick_loaded_views:
        raise RuntimeError(
            f"{what} is not available: this target was loaded with --quick "
            "(no analysis). Run `bn refresh` to run analysis first."
        )
