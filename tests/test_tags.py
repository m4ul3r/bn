from __future__ import annotations

import pytest

from bn_agent_bridge import read_tags
from _bridge_fakes import _FakeBV


class _Ctx:
    def __init__(self, bv):
        self._bv = bv

    def _resolve_view(self, selector):
        return self._bv


def test_list_tag_types_reports_builtin_flag_and_icon():
    bv = _FakeBV()
    bv.create_tag_type("Bugs", "B")           # a built-in name
    bv.create_tag_type("MyNotes", "N")        # a custom name
    result = read_tags._list_tag_types(_Ctx(bv), None)
    by_name = {t["name"]: t for t in result["tag_types"]}
    assert by_name["Bugs"]["is_builtin"] is True
    assert by_name["Bugs"]["icon"] == "B"
    assert by_name["MyNotes"]["is_builtin"] is False
    assert result["count"] == 2
