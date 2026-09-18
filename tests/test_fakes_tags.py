from __future__ import annotations

from _bridge_fakes import _FakeBV, _FakeFunction, _FakeBasicBlock

#: The first id a fresh view's tag counter hands out ('fa5e' marks it as a fake).
_FIRST_TAG_ID = "0000fa5e-0000-0000-0000-000000000001"
_SECOND_TAG_ID = "0000fa5e-0000-0000-0000-000000000002"


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


def test_tag_ids_are_per_view_not_per_session():
    """#786: an absolute tag id must depend on the view the test built and nothing
    else. With one module-global counter, a view's first tag id was a function of
    how many tags every earlier test in the session (and, under -n, this worker's
    share of them) had created -- the id drifted with suite order and with worker
    assignment. A per-view counter makes the first tag on ANY fresh view the
    counter's first id, whatever ran before it."""
    first_bv, _ = _bv_with_fn()
    first_bv.create_tag_type("Library", "L")

    other_bv, _ = _bv_with_fn()
    other_bv.create_tag_type("Library", "L")
    other_bv.add_tag(0x2000, "Library", "burn ids on another view")

    tag = first_bv.add_tag(0x2000, "Library", "libc")
    assert tag.id == _FIRST_TAG_ID
    assert other_bv.get_tags_at(0x2000)[0].id == _FIRST_TAG_ID


def test_function_and_data_tags_share_the_view_id_space():
    """A function tag and a data tag on one view are a single id space (as they
    are in real BN), and two views never share one."""
    bv, fn = _bv_with_fn()
    bv.create_tag_type("Important", "!")
    fn.add_tag("Important", "whole fn", None)
    bv.add_tag(0x2000, "Important", "global")
    assert [fn.get_function_tags()[0].id, bv.get_tags_at(0x2000)[0].id] == [
        _FIRST_TAG_ID, _SECOND_TAG_ID,
    ]

    second_bv, second_fn = _bv_with_fn()
    second_bv.create_tag_type("Important", "!")
    second_fn.add_tag("Important", "whole fn", None)
    assert second_fn.get_function_tags()[0].id == _FIRST_TAG_ID
