from __future__ import annotations

from _bridge_fakes import (
    _FakeBasicBlock,
    _FakeBV,
    _FakeFunction,
    _FakeSection,
    _FakeSegment,
)


def test_fake_bv_read_respects_seeded_memory_map():
    """#616: real BN's bv.read returns b"" for an unmapped address; a fake with
    NO memory seeded at all must not invent b"\\x90" * length filler -- that
    used to hide every unmapped-path branch in production code (e.g. the
    function_create mappedness guard in test_function_create.py) behind a
    phantom NOP stream. Also pins the mapped-read and short-read halves of the
    contract: a mapped read returns exactly the seeded bytes, and a read past
    a blob's end is truncated at the boundary rather than raising or
    padding."""
    bv = _FakeBV()
    assert bv.read(0xdead, 4) == b""
    assert bv.read(0x0, 1) == b""
    seeded = _FakeBV(memory={0x1000: b"\x55\x48\x89\xe5"})
    assert seeded.read(0x1000, 4) == b"\x55\x48\x89\xe5"   # exact mapped read
    assert seeded.read(0x1002, 8) == b"\x89\xe5"           # short read: stops at blob end
    assert seeded.read(0xdead, 4) == b""                   # unmapped, map non-empty


def test_fake_bv_read_rejects_non_positive_length():
    """_FakeBV.read must not turn a negative length into a reversed-prefix
    slice (Python's b[0:-1] semantics) -- b"" is this double's defensive
    convention for a length no real caller can produce."""
    seeded = _FakeBV(memory={0x1000: b"\x55\x48\x89\xe5"})
    assert seeded.read(0x1000, -1) == b""
    assert seeded.read(0x1000, 0) == b""


def test_fake_bv_maps_only_what_was_seeded():
    """#783: `is_valid_offset` makes an unmapped address the DEFAULT, so the
    #374/`_require_mapped_address` rejection fires under the mocks without every
    unmapped-path test hand-patching the member (the #616 class). A bare view
    maps NOTHING; a layer maps exactly the addresses it was seeded with --
    never one past a blob, never an address the view never mentioned."""
    bare = _FakeBV()
    assert bare.is_valid_offset(0x0) is False
    assert bare.is_valid_offset(0x1000) is False
    assert bare.is_valid_offset(0xDEADBEEF) is False

    seeded = _FakeBV(memory={0x1000: b"\x55\x48\x89\xe5"})
    assert seeded.is_valid_offset(0x1000) is True    # first mapped byte
    assert seeded.is_valid_offset(0x1003) is True    # last mapped byte
    assert seeded.is_valid_offset(0x1004) is False   # one past the blob
    assert seeded.is_valid_offset(0xDEADBEEF) is False


def test_fake_bv_is_valid_offset_covers_segments_sections_and_function_bodies():
    """Every way a test tells the fake "this address is really there" counts as
    mapped: a segment extent, a section range, a function body. Anything else is
    unmapped -- a view whose only relation to an address is a cref/comment/tag at
    it stays unmapped, because that is the case #374's follow-up tests pin."""
    fn = _FakeFunction(0x401000, "sub_401000")
    fn.basic_blocks = [_FakeBasicBlock(0x401000, 0x401040)]
    bv = _FakeBV(
        functions=[fn],
        sections={".rodata": _FakeSection(".rodata", 0x5000, 0x5010)},
        segments={0x6000: _FakeSegment(readable=True, writable=True)},
        memory={0x6000: bytes(0x20)},
    )
    assert bv.is_valid_offset(0x401020) is True      # inside the function body
    assert bv.is_valid_offset(0x401040) is False     # one past the body
    assert bv.is_valid_offset(0x5008) is True        # inside the section
    assert bv.is_valid_offset(0x5010) is False       # one past the section
    assert bv.is_valid_offset(0x6010) is True        # inside the segment's blob
    assert bv.is_valid_offset(0x6020) is False       # one past it


def test_fake_bv_get_segment_at_is_range_aware():
    """#783: real BN's get_segment_at answers for the segment CONTAINING the
    address. Keying only the base address made `is_offset_executable` (and every
    other segment read) report None for an address inside the segment."""
    bv = _FakeBV(
        segments={0x400000: _FakeSegment(readable=True, executable=True)},
        memory={0x400000: b"\x90" * 0x200},
    )
    seg = bv.get_segment_at(0x400010)
    assert seg is not None
    assert (seg.readable, seg.executable) == (True, True)
    assert bv.get_segment_at(0x400000) is seg     # the base still resolves
    assert bv.get_segment_at(0x4001FF) is seg     # the last mapped byte
    assert bv.get_segment_at(0x400200) is None    # one past the extent


def test_fake_bv_get_segment_at_honours_an_explicit_extent():
    """A segment seeded with its own start/end uses those, so a test can model a
    segment wider or narrower than any blob it happened to seed."""
    bv = _FakeBV(segments={
        0x401000: _FakeSegment(start=0x401000, end=0x402000, readable=True, executable=True),
    })
    assert bv.get_segment_at(0x401FF0) is not None
    assert bv.get_segment_at(0x402000) is None
    assert bv.is_valid_offset(0x401FF0) is True


def test_fake_bv_segment_without_a_seeded_extent_covers_only_its_base_byte():
    """A `segments={addr: seg}` with no blob behind it has no extent the fake can
    honestly claim, so it covers the byte it was seeded at and nothing wider --
    a segment query must never invent coverage the view was not given."""
    bv = _FakeBV(segments={0x401000: _FakeSegment(readable=True)})
    assert bv.get_segment_at(0x401000) is not None
    assert bv.get_segment_at(0x401001) is None
    assert bv.get_segment_at(0x400FFF) is None


def test_fake_bv_function_body_and_mappedness_use_one_span_rule():
    """`get_functions_containing` and `is_valid_offset` must agree: an address
    inside a function body resolves to that function AND is mapped, or a read
    would resolve the body while its own address read as unmapped."""
    fn = _FakeFunction(0x1000, "sub_1000")
    fn.basic_blocks = [_FakeBasicBlock(0x1000, 0x1060)]
    bv = _FakeBV(functions=[fn])
    for addr in (0x1000, 0x1030, 0x105F):
        assert [f.name for f in bv.get_functions_containing(addr)] == ["sub_1000"]
        assert bv.is_valid_offset(addr) is True
    assert bv.get_functions_containing(0x1060) == []
    assert bv.is_valid_offset(0x1060) is False
