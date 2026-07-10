from __future__ import annotations

import pytest

from _bridge_fakes import _load_bridge


def test_builtin_tag_type_names_cover_bn_defaults(monkeypatch):
    _load_bridge(monkeypatch)
    from bn_agent_bridge._shared import _BUILTIN_TAG_TYPE_NAMES

    # A representative sample of BN's built-in tag types (probed on BN 5.4).
    for name in ("Bookmarks", "Bugs", "Crashes", "Important", "Library", "Needs Analysis"):
        assert name in _BUILTIN_TAG_TYPE_NAMES
    assert "MyCustomType" not in _BUILTIN_TAG_TYPE_NAMES
    assert isinstance(_BUILTIN_TAG_TYPE_NAMES, frozenset)
