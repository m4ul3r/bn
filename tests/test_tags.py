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


def test_list_tags_all_scopes_deduped_and_paged():
    bv, fn = _bv_with_tagged_fn()  # 1 function tag, 1 address tag, 1 data tag
    result = read_tags._list_tags(_CtxFn(bv), None)
    scopes = sorted(t["scope"] for t in result["items"])
    assert scopes == ["address", "data", "function"]
    assert result["total"] == 3
    assert result["kind"] == "tags"


def test_list_tags_filter_by_type():
    bv, fn = _bv_with_tagged_fn()
    result = read_tags._list_tags(_CtxFn(bv), None, type="Important")
    assert [t["type"] for t in result["items"]] == ["Important"]


def test_list_tags_data_only():
    bv, fn = _bv_with_tagged_fn()
    result = read_tags._list_tags(_CtxFn(bv), None, data_only=True)
    assert all(t["scope"] == "data" for t in result["items"])
    assert result["total"] == 1


def test_list_tags_query_substring_on_data():
    bv, fn = _bv_with_tagged_fn()
    result = read_tags._list_tags(_CtxFn(bv), None, query="whole")
    assert [t["data"] for t in result["items"]] == ["whole fn"]


def test_list_tags_dedupes_same_tag_id_across_collection_sources():
    """The whole-view sweep in _collect_tags normally sees disjoint tags from
    bv.get_tags() (data) and fn.tags (address) -- so a `seen`-guard regression
    that reintroduced overlap between those sources wouldn't be caught by any
    existing test. Force the SAME _FakeTag object (same .id) to be reachable
    from BOTH bv.get_tags() and fn.tags, and assert it still collapses to a
    single entry -- proving the dedup guard is load-bearing, not vacuous."""
    fn = _FakeFunction(0x1000, "sub_1000")
    fn.basic_blocks = [_FakeBasicBlock(0x1000, 0x1040)]
    bv = _FakeBV(functions=[fn])
    fn.view = bv
    bv.create_tag_type("Bugs", "B")
    tag = bv.add_tag(0x1010, "Bugs", "shared tag")
    # Directly alias the same tag object into the function's address-tag sweep,
    # simulating a source-overlap regression (e.g. fn.tags also surfacing a
    # view-level data tag) without touching production code.
    fn._address_tags[0x1010] = [tag]

    result = read_tags._list_tags(_CtxFn(bv), None)
    matching = [t for t in result["items"] if t["id"] == str(tag.id)]
    assert len(matching) == 1
    assert result["total"] == 1


from bn_agent_bridge import mutation_engine
from bn_agent_bridge._shared import OperationFailure
from _bridge_fakes import _FakeTagMutationBV


def _mut_bv_with_fn():
    fn = _FakeFunction(0x1000, "sub_1000")
    fn.basic_blocks = [_FakeBasicBlock(0x1000, 0x1040)]
    bv = _FakeTagMutationBV()
    bv.functions = [fn]
    fn.view = bv
    bv.create_tag_type("Important", "!")
    return bv, fn


class _CtxMut(_CtxFn):
    pass


def test_op_tag_add_address_scope_attaches_to_function():
    bv, fn = _mut_bv_with_fn()
    op = {"op": "tag_add", "type": "Important", "data": "look here", "address": "0x1010"}
    result = mutation_engine._op_tag_add(_CtxMut(bv), bv, op)
    assert result["scope"] == "address"
    assert [(t.type.name, t.data) for t in fn.get_tags_at(0x1010)] == [("Important", "look here")]


def test_op_tag_add_function_scope():
    bv, fn = _mut_bv_with_fn()
    op = {"op": "tag_add", "type": "Important", "data": "doc", "function": "sub_1000"}
    result = mutation_engine._op_tag_add(_CtxMut(bv), bv, op)
    assert result["scope"] == "function"
    assert [(t.type.name, t.data) for t in fn.get_function_tags()] == [("Important", "doc")]


def test_op_tag_add_force_data_scope():
    bv, fn = _mut_bv_with_fn()
    op = {"op": "tag_add", "type": "Important", "data": "d", "address": "0x1010", "force_data": True}
    result = mutation_engine._op_tag_add(_CtxMut(bv), bv, op)
    assert result["scope"] == "data"
    assert [(t.type.name, t.data) for t in bv.get_tags_at(0x1010)] == [("Important", "d")]


def test_op_tag_add_no_function_falls_back_to_data():
    bv, fn = _mut_bv_with_fn()  # 0x9000 is outside sub_1000
    op = {"op": "tag_add", "type": "Important", "data": "d", "address": "0x9000"}
    result = mutation_engine._op_tag_add(_CtxMut(bv), bv, op)
    assert result["scope"] == "data"
    assert "no function" in result["message"].lower()


def test_op_tag_add_unknown_type_is_invalid_request():
    bv, fn = _mut_bv_with_fn()
    op = {"op": "tag_add", "type": "Nonexistent", "data": "d", "address": "0x1010"}
    with pytest.raises(OperationFailure) as exc:
        mutation_engine._op_tag_add(_CtxMut(bv), bv, op)
    assert exc.value.status == "invalid_request"


def test_verify_tag_add_address_scope_passes():
    bv, fn = _mut_bv_with_fn()
    op = {"op": "tag_add", "type": "Important", "data": "look here", "address": "0x1010"}
    result = mutation_engine._op_tag_add(_CtxMut(bv), bv, op)
    verified = mutation_engine._verify_tag_add(_CtxMut(bv), bv, result)
    assert verified["status"] == "verified"
    assert verified["observed"] == {"present": True}


def test_verify_tag_add_function_scope_passes():
    bv, fn = _mut_bv_with_fn()
    op = {"op": "tag_add", "type": "Important", "data": "doc", "function": "sub_1000"}
    result = mutation_engine._op_tag_add(_CtxMut(bv), bv, op)
    verified = mutation_engine._verify_tag_add(_CtxMut(bv), bv, result)
    assert verified["status"] == "verified"


def test_verify_tag_add_data_scope_passes():
    bv, fn = _mut_bv_with_fn()
    op = {"op": "tag_add", "type": "Important", "data": "d", "address": "0x1010", "force_data": True}
    result = mutation_engine._op_tag_add(_CtxMut(bv), bv, op)
    verified = mutation_engine._verify_tag_add(_CtxMut(bv), bv, result)
    assert verified["status"] == "verified"


def test_verify_tag_add_raises_verification_failed_when_tag_missing():
    bv, fn = _mut_bv_with_fn()
    # Describe a tag that was never added -- verification must reject it
    # instead of silently reporting success.
    result = {
        "op": "tag_add",
        "tag_type": "Important",
        "data": "never added",
        "scope": "address",
        "address": "0x1010",
        "function": "sub_1000",
        "requested": {},
    }
    with pytest.raises(OperationFailure) as exc:
        mutation_engine._verify_tag_add(_CtxMut(bv), bv, result)
    assert exc.value.status == "verification_failed"


def test_op_tag_remove_by_type_at_address():
    bv, fn = _mut_bv_with_fn()
    fn.add_tag("Important", "x", 0x1010)
    op = {"op": "tag_remove", "type": "Important", "address": "0x1010"}
    result = mutation_engine._op_tag_remove(_CtxMut(bv), bv, op)
    assert result["removed"] == 1
    assert fn.get_tags_at(0x1010) == []


def test_op_tag_remove_by_id():
    bv, fn = _mut_bv_with_fn()
    tag = fn.add_tag("Important", "x", None)
    op = {"op": "tag_remove", "tag_id": tag.id}
    result = mutation_engine._op_tag_remove(_CtxMut(bv), bv, op)
    assert result["removed"] == 1
    assert fn.get_function_tags() == []


def test_op_tag_remove_no_match_is_noop_shaped():
    bv, fn = _mut_bv_with_fn()
    op = {"op": "tag_remove", "type": "Important", "address": "0x1010"}
    result = mutation_engine._op_tag_remove(_CtxMut(bv), bv, op)
    assert result["removed"] == 0


def test_verify_tag_remove_noop_when_nothing_removed():
    bv, fn = _mut_bv_with_fn()
    result = {"op": "tag_remove", "removed": 0, "removed_tags": [],
              "requested": {}, "targets": []}
    verified = mutation_engine._verify_tag_remove(_CtxMut(bv), bv, result)
    assert verified["status"] == "noop"


def test_op_tag_remove_filters_by_data():
    bv, fn = _mut_bv_with_fn()
    fn.add_tag("Important", "keep", 0x1010)
    fn.add_tag("Important", "drop", 0x1010)
    op = {"op": "tag_remove", "type": "Important", "data": "drop", "address": "0x1010"}
    result = mutation_engine._op_tag_remove(_CtxMut(bv), bv, op)
    assert result["removed"] == 1
    assert [t.data for t in fn.get_tags_at(0x1010)] == ["keep"]


def test_verify_tag_remove_success_when_tag_gone():
    bv, fn = _mut_bv_with_fn()
    fn.add_tag("Important", "x", 0x1010)
    op = {"op": "tag_remove", "type": "Important", "address": "0x1010"}
    result = mutation_engine._op_tag_remove(_CtxMut(bv), bv, op)
    verified = mutation_engine._verify_tag_remove(_CtxMut(bv), bv, result)
    assert verified["status"] == "verified"
    assert verified["observed"] == {"still_present": []}


def test_op_tag_remove_by_id_address_scope():
    """CRITICAL: `bn tag remove --id <uuid>` for an ADDRESS-scope tag (the
    common case -- `tag add <addr> --type X`) used to silently no-op:
    the id-only branch swept bv.get_tags() (DATA scope) and each function's
    get_function_tags() (FUNCTION scope) but never a function's ADDRESS-scope
    tags (fn.tags). Same bug class Task 4 fixed for _collect_tags."""
    bv, fn = _mut_bv_with_fn()
    tag = fn.add_tag("Important", "x", 0x1010)
    op = {"op": "tag_remove", "tag_id": tag.id}
    result = mutation_engine._op_tag_remove(_CtxMut(bv), bv, op)
    assert result["removed"] == 1
    assert fn.get_tags_at(0x1010) == []


def test_op_tag_remove_by_type_leaves_other_types():
    bv, fn = _mut_bv_with_fn()
    bv.create_tag_type("Bookmarks", "*")
    fn.add_tag("Important", "important x", 0x1010)
    fn.add_tag("Bookmarks", "bookmark x", 0x1010)
    op = {"op": "tag_remove", "type": "Important", "address": "0x1010"}
    result = mutation_engine._op_tag_remove(_CtxMut(bv), bv, op)
    assert result["removed"] == 1
    remaining = [(t.type.name, t.data) for t in fn.get_tags_at(0x1010)]
    assert remaining == [("Bookmarks", "bookmark x")]


def test_op_tag_type_create_and_verify():
    bv, fn = _mut_bv_with_fn()
    op = {"op": "tag_type_create", "name": "MyNotes", "icon": "N"}
    result = mutation_engine._op_tag_type_create(_CtxMut(bv), bv, op)
    assert bv.get_tag_type("MyNotes") is not None
    verified = mutation_engine._verify_tag_type_create(_CtxMut(bv), bv, result)
    assert verified["status"] == "verified"


def test_op_tag_type_create_existing_is_noop():
    bv, fn = _mut_bv_with_fn()  # "Important" already exists
    op = {"op": "tag_type_create", "name": "Important", "icon": "!"}
    result = mutation_engine._op_tag_type_create(_CtxMut(bv), bv, op)
    verified = mutation_engine._verify_tag_type_create(_CtxMut(bv), bv, result)
    assert verified["status"] == "noop"


def test_op_tag_type_remove_custom():
    bv, fn = _mut_bv_with_fn()
    bv.create_tag_type("MyNotes", "N")
    op = {"op": "tag_type_remove", "name": "MyNotes"}
    result = mutation_engine._op_tag_type_remove(_CtxMut(bv), bv, op)
    assert bv.get_tag_type("MyNotes") is None
    verified = mutation_engine._verify_tag_type_remove(_CtxMut(bv), bv, result)
    assert verified["status"] == "verified"


def test_op_tag_type_remove_builtin_is_rejected():
    bv, fn = _mut_bv_with_fn()
    bv.create_tag_type("Bugs", "B")  # a built-in name
    op = {"op": "tag_type_remove", "name": "Bugs"}
    with pytest.raises(OperationFailure) as exc:
        mutation_engine._op_tag_type_remove(_CtxMut(bv), bv, op)
    assert exc.value.status == "invalid_request"
    assert bv.get_tag_type("Bugs") is not None  # not removed


def test_op_tag_type_remove_nonexistent_is_noop():
    bv, fn = _mut_bv_with_fn()
    op = {"op": "tag_type_remove", "name": "GhostType"}
    result = mutation_engine._op_tag_type_remove(_CtxMut(bv), bv, op)
    verified = mutation_engine._verify_tag_type_remove(_CtxMut(bv), bv, result)
    assert verified["status"] == "noop"
