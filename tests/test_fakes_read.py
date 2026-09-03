from __future__ import annotations

from _bridge_fakes import _FakeBV


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
