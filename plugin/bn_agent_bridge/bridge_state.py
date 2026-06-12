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
