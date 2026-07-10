from __future__ import annotations

from _bridge_fakes import _FakeBV, _FakeFunction, _FakeBasicBlock


def _bv_with_fn():
    fn = _FakeFunction(0x1000, "sub_1000")
    fn.basic_blocks = [_FakeBasicBlock(0x1000, 0x1040)]
    bv = _FakeBV(functions=[fn])
    fn.view = bv
    return bv, fn


def test_fake_bv_create_and_get_tag_type():
    bv, _ = _bv_with_fn()
    tt = bv.create_tag_type("Important", "!")
    assert tt.name == "Important"
    assert bv.get_tag_type("Important") is tt
    assert bv.get_tag_type("Nope") is None


def test_fake_bv_data_tag_roundtrip():
    bv, _ = _bv_with_fn()
    bv.create_tag_type("Library", "L")
    bv.add_tag(0x2000, "Library", "libc")
    tags = bv.get_tags_at(0x2000)
    assert [(t.type.name, t.data) for t in tags] == [("Library", "libc")]
    assert [(a, t.type.name) for a, t in bv.get_tags()] == [(0x2000, "Library")]
    bv.remove_user_data_tag(0x2000, tags[0])
    assert bv.get_tags_at(0x2000) == []


def test_fake_function_tag_and_address_tag_roundtrip():
    bv, fn = _bv_with_fn()
    bv.create_tag_type("Important", "!")
    bv.create_tag_type("Bugs", "B")
    fn.add_tag("Important", "whole fn", None)
    fn.add_tag("Bugs", "at 0x1010", 0x1010)
    assert [(t.type.name, t.data) for t in fn.get_function_tags()] == [("Important", "whole fn")]
    assert [(t.type.name, t.data) for t in fn.get_tags_at(0x1010)] == [("Bugs", "at 0x1010")]
    ft = fn.get_function_tags()[0]
    fn.remove_user_function_tag(ft)
    assert fn.get_function_tags() == []
    at = fn.get_tags_at(0x1010)[0]
    fn.remove_user_address_tag(0x1010, at)
    assert fn.get_tags_at(0x1010) == []
