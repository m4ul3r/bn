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


from _bridge_fakes import _FakeFunction, _FakeBasicBlock


class _CtxFn(_Ctx):
    def _find_function(self, bv, ident):
        for fn in bv.functions:
            if fn.name == ident or hex(fn.start) == ident:
                return fn
        raise RuntimeError(f"function not found: {ident}")


def _bv_with_tagged_fn():
    fn = _FakeFunction(0x1000, "sub_1000")
    fn.basic_blocks = [_FakeBasicBlock(0x1000, 0x1040)]
    bv = _FakeBV(functions=[fn])
    fn.view = bv
    bv.create_tag_type("Important", "!")
    bv.create_tag_type("Bugs", "B")
    fn.add_tag("Important", "whole fn", None)
    fn.add_tag("Bugs", "at 0x1010", 0x1010)
    bv.add_tag(0x9000, "Bugs", "data tag")  # ensure add_tag needs the type
    return bv, fn


def test_get_tags_at_address_returns_address_and_function_tags():
    bv, fn = _bv_with_tagged_fn()
    result = read_tags._get_tags(_CtxFn(bv), None, "0x1010", None)
    datas = {(t["type"], t["data"], t["scope"]) for t in result["tags"]}
    assert ("Bugs", "at 0x1010", "address") in datas
    assert result["count"] == 1


def test_get_tags_by_function_returns_function_tags():
    bv, fn = _bv_with_tagged_fn()
    result = read_tags._get_tags(_CtxFn(bv), None, None, "sub_1000")
    datas = {(t["type"], t["data"], t["scope"]) for t in result["tags"]}
    assert ("Important", "whole fn", "function") in datas


def test_get_tags_rejects_both_locators():
    bv, _ = _bv_with_tagged_fn()
    with pytest.raises(RuntimeError, match="not both"):
        read_tags._get_tags(_CtxFn(bv), None, "0x1010", "sub_1000")


def test_get_tags_rejects_unmapped_address():
    """tag get on an unmapped, tag-less address must reject (parity with
    comment get / xrefs, #374) instead of returning a false empty result."""
    bv, _ = _bv_with_tagged_fn()
    bv.is_valid_offset = lambda addr: False
    with pytest.raises(RuntimeError, match="not mapped"):
        read_tags._get_tags(_CtxFn(bv), None, "0x2000", None)


def test_get_tags_mapped_address_without_tags_stays_clean():
    """A MAPPED address with no tags is a clean empty result -- only the
    unmapped case is rejected (#374)."""
    bv, _ = _bv_with_tagged_fn()
    bv.is_valid_offset = lambda addr: True
    result = read_tags._get_tags(_CtxFn(bv), None, "0x2000", None)
    assert result["tags"] == []
    assert result["count"] == 0


def test_get_tags_data_scope_outside_any_function():
    bv, _ = _bv_with_tagged_fn()
    result = read_tags._get_tags(_CtxFn(bv), None, "0x9000", None)
    datas = {(t["type"], t["data"], t["scope"], t["function"]) for t in result["tags"]}
    assert ("Bugs", "data tag", "data", None) in datas
