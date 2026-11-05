"""Copy-source and model-role corrections from the forward-engine review:
an ELF image's declared object sizes in a rebased view (R1-Q5), a fill's
return that only bounds a copy once proven non-negative (R3-01), and the
register fallback for every argument position a model declares (R3-03)."""
from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest

from test_taint_engine import FBV, FExpr, FFunc, FInstr, FPVS, FSSA, FSSAFunc, FVar, te
from test_taint_forward_global_frontier import FGlobalBV, FRawView, _elf_with_object_sizes
from test_taint_width_and_relabel import _memcpy_sink, _recv_then_dup


@pytest.fixture
def models():
    return te.load_models()


# ---------------------------------------------------------------------------
# R1-Q5: declared object sizes in the view's address space
# ---------------------------------------------------------------------------

ET_EXEC, ET_DYN = 2, 3


def _image_engine(models, *, e_type, link_base, view_start, view_end, objects):
    """A 64-bit ELF whose only `PT_LOAD` is linked at *link_base*, loaded by
    a view spanning ``[view_start, view_end)``."""
    image = bytearray(_elf_with_object_sizes(objects))
    struct.pack_into("<H", image, 0x10, e_type)
    struct.pack_into("<Q", image, 0x20, 0x40)              # e_phoff
    struct.pack_into("<HH", image, 0x36, 56, 1)            # e_phentsize, e_phnum
    image[0x40:0x40 + 56] = struct.pack("<IIQQQQQQ", 1, 6, 0, link_base, link_base,
                                        0x6000, 0x6000, 0x1000)
    bv = FGlobalBV({})
    bv.parent_view = FRawView(bytes(image))
    bv.start, bv.end = view_start, view_end
    return te.TaintEngine(bv, models)


def test_a_rebased_fixed_address_image_declares_its_object_at_both_addresses(models):
    # Linked at 0x400000, the view rebased to 0x500000: BN moves the symbol to
    # 0x504060, while the image's absolute references (no relocations) still
    # name 0x404060. Both key the one 8-byte object; neither sizes past it.
    eng = _image_engine(models, e_type=ET_EXEC, link_base=0x400000, view_start=0x500000,
                        view_end=0x506000, objects={"g": (0x404060, 8)})
    assert eng._authoritative_width(0x504060) == 8
    assert eng._authoritative_width(0x404060) == 8
    assert eng._declared_object(0x404067) == (0x404060, 8)
    assert eng._declared_object(0x404068) is None
    assert eng._declared_object(0x504068) is None


def test_a_fixed_address_image_at_its_link_base_declares_one_address(models):
    eng = _image_engine(models, e_type=ET_EXEC, link_base=0x400000, view_start=0x400000,
                        view_end=0x406000, objects={"g": (0x404060, 8)})
    assert eng._declared_object_table() == [(0x404060, 8)]


def test_a_rebased_pie_declares_its_object_only_where_the_view_has_it(models):
    # A PIE reaches its objects PC-relatively or through relocations BN
    # applies, so its link-time address names nothing in the view.
    eng = _image_engine(models, e_type=ET_DYN, link_base=0, view_start=0x400000,
                        view_end=0x406000, objects={"g": (0x4060, 8)})
    assert eng._declared_object_table() == [(0x404060, 8)]
    assert eng._authoritative_width(0x4060) == 0


def test_a_link_address_the_view_maps_is_not_claimed_for_the_moved_object(models):
    # Rebased by one page: 0x404060 is now inside the view, where it is some
    # other storage's address, so only the moved object is declared.
    eng = _image_engine(models, e_type=ET_EXEC, link_base=0x400000, view_start=0x401000,
                        view_end=0x407000, objects={"g": (0x404060, 8)})
    assert eng._declared_object_table() == [(0x405060, 8)]


# ---------------------------------------------------------------------------
# R3-01: a fill's return bounds the copy only once it is a byte count
# ---------------------------------------------------------------------------

def _values(kind, start, end):
    pvs = FPVS(kind)
    pvs.ranges = [SimpleNamespace(start=start, end=end, step=1)]
    return pvs


@pytest.mark.parametrize("at_copy", [
    None,                                                  # BN states nothing
    _values("SignedRangeValue", -0x80000000, 0x7fffffff),  # no guard: -1 possible
    _values("UnsignedRangeValue", 0, 0xffffffff),          # zx of a 32-bit -1
], ids=["no-value-set", "unguarded", "zero-extended"])
def test_an_unchecked_fill_return_still_overreads_its_source(models, at_copy):
    # `n = recv(fd, buf, 0x40, 0); memcpy(malloc(n), buf, n)` with no
    # success check: recv's -1 is SIZE_MAX at the copy.
    sink = _memcpy_sink(models, *_recv_then_dup(count="const", length="sx", src="same",
                                                at_copy=at_copy))
    assert sink["class"] == "overread_len", sink
    assert "src_fill" not in sink
    assert sink["src_capacity"] == 0x40


def test_a_non_negative_guard_proves_the_fill_return(models):
    # `if (n < 0) return;` leaves 0..0x7fffffff: a zero-byte copy is a count too.
    sink = _memcpy_sink(models, *_recv_then_dup(
        count="const", length="sx", src="same",
        at_copy=_values("SignedRangeValue", 0, 0x7fffffff)))
    assert sink["class"] == "bounded_len"
    assert sink["src_fill"] == {"callee": "recv", "at": "0x1010"}
