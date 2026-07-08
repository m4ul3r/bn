"""#489: proactive call-model-truncation disclosure helpers in read_taint_slice."""
from __future__ import annotations

import types

from _bridge_fakes import _load_bridge


def _op(name):
    return types.SimpleNamespace(operation=types.SimpleNamespace(name=name))


def _reg(name):
    e = _op("LLIL_REG"); e.src = types.SimpleNamespace(name=name); return e


def _const(v):
    e = _op("LLIL_CONST"); e.constant = v; return e


def _add(left, right):
    e = _op("LLIL_ADD"); e.left = left; e.right = right; return e


def _store(dest, addr):
    e = _op("LLIL_STORE"); e.dest = dest; e.address = addr; return e


def _call(addr):
    e = _op("LLIL_CALL"); e.address = addr; e.dest = _const(0x1000); return e


def _func(block, *, sp="sp", arg_regs=("r0", "r1", "r2", "r3"), name="my_logger"):
    return types.SimpleNamespace(
        low_level_il=[block],
        arch=types.SimpleNamespace(stack_pointer=sp),
        calling_convention=types.SimpleNamespace(int_arg_regs=list(arg_regs)),
        name=name,
    )


def test_llil_sp_offset_forms(monkeypatch):
    rts = _load_bridge(monkeypatch).read_taint_slice
    assert rts._llil_sp_offset(_reg("sp"), "sp") == 0
    assert rts._llil_sp_offset(_add(_reg("sp"), _const(8)), "sp") == 8
    assert rts._llil_sp_offset(_reg("r0"), "sp") is None          # not sp
    assert rts._llil_sp_offset(_add(_reg("r0"), _const(8)), "sp") is None
    assert rts._llil_sp_offset(_add(_reg("sp"), _const(-4)), "sp") is None  # negative off


def test_stack_arg_store_offsets_collects_outgoing(monkeypatch):
    # sp+0, sp+4, sp+8 stores feed the call; a post-call store is ignored.
    rts = _load_bridge(monkeypatch).read_taint_slice
    call_addr = 0x1c
    block = [
        _store(_reg("sp"), 0x10),
        _store(_add(_reg("sp"), _const(4)), 0x12),
        _store(_add(_reg("sp"), _const(8)), 0x14),
        _call(call_addr),
        _store(_add(_reg("sp"), _const(0xc)), 0x20),   # AFTER the call -- excluded
    ]
    assert rts._stack_arg_store_offsets(_func(block), call_addr) == [0, 4, 8]


def _fmt_ptr(addr):
    e = _op("MLIL_CONST_PTR"); e.constant = addr; return e


def _plain():
    # a non-const arg expr -> resolves to no format string
    e = _op("MLIL_VAR"); e.src = types.SimpleNamespace(name="v"); return e


class _BV:
    """Fake BinaryView exposing just `.read` for format-string lookup."""
    def __init__(self, strings=None):
        self._s = dict(strings or {})

    def read(self, addr, cap):
        return self._s.get(int(addr), b"")


_FMT_STR = {0x5000: b"%d %d %d %d %d\x00"}   # a genuine format string in rodata


def test_truncation_note_fires_on_format_string_variadic(monkeypatch):
    # A recovered format-string arg (positive variadic signal) + contiguous stack
    # stores -> genuine variadic-auto-typed-fixed truncation; note fires (#489).
    rts = _load_bridge(monkeypatch).read_taint_slice
    call_addr = 0x1c
    block = [_store(_reg("sp"), 0x10),
             _store(_add(_reg("sp"), _const(4)), 0x12),
             _call(call_addr)]
    note = rts._call_model_truncation_note(
        _BV(_FMT_STR), _func(block), None, call_addr, [_fmt_ptr(0x5000)], "my_logger")
    assert note is not None
    assert "call-model truncation" in note and "my_logger" in note
    assert "sp+0x0" in note and "sp+0x4" in note and "proto set" in note


def test_truncation_note_silent_without_format_string_arg(monkeypatch):
    # THE key FP guard (review): the same contiguous-stack-store shape but NO
    # format-string arg -- a fixed-arity libc/BSD call (strchr/memset/bcopy) or a
    # callee-saved-spill collision. Must stay silent.
    rts = _load_bridge(monkeypatch).read_taint_slice
    call_addr = 0x1c
    block = [_store(_reg("sp"), 0x10),
             _store(_add(_reg("sp"), _const(4)), 0x12),
             _call(call_addr)]
    assert rts._call_model_truncation_note(
        _BV(), _func(block), None, call_addr, [_plain(), _plain()], "memset") is None
    # a const arg that is NOT a format string (no % specifier) also stays silent
    assert rts._call_model_truncation_note(
        _BV({0x6000: b"/tmp/x\x00"}), _func(block), None, call_addr, [_fmt_ptr(0x6000)], "open") is None


def test_truncation_note_silent_on_known_fixed_arity_libc(monkeypatch):
    # A fixed-arity libc function (strlen) taking a FORMAT-SHAPED string literal must
    # NOT fire even though the format gate matches -- the denylist catches it. This
    # is the one residual FP from review (strlen of a format string on MIPS) (#489).
    rts = _load_bridge(monkeypatch).read_taint_slice
    call_addr = 0x1c
    block = [_store(_reg("sp"), 0x10),
             _store(_add(_reg("sp"), _const(4)), 0x12),
             _call(call_addr)]
    bv = _BV({0x5000: b"0x%04x:%04X\x00"})
    for nm in ("strlen", "strlen@plt", "__strlen", "memcpy", "strcmp"):
        assert rts._call_model_truncation_note(
            bv, _func(block), None, call_addr, [_fmt_ptr(0x5000)], nm) is None, nm
    # a genuine variadic (not in the denylist) with the same shape still fires
    assert rts._call_model_truncation_note(
        bv, _func(block), None, call_addr, [_fmt_ptr(0x5000)], "my_log") is not None


def test_format_regex_ignores_natural_percent(monkeypatch):
    # "50% off" / "5% charge" must NOT be read as a format string (no space-flag).
    rts = _load_bridge(monkeypatch).read_taint_slice
    call_addr = 0x1c
    block = [_store(_reg("sp"), 0x10),
             _store(_add(_reg("sp"), _const(4)), 0x12),
             _call(call_addr)]
    bv = _BV({0x5000: b"50% off today, 5% charge\x00"})
    assert rts._call_model_truncation_note(
        bv, _func(block), None, call_addr, [_fmt_ptr(0x5000)], "some_func") is None


def test_truncation_note_silent_when_stack_params_recovered(monkeypatch):
    # len(params) > arg_regs -> BN recovered stack params -> not truncated -> silent.
    rts = _load_bridge(monkeypatch).read_taint_slice
    call_addr = 0x1c
    block = [_store(_reg("sp"), 0x10), _call(call_addr)]
    assert rts._call_model_truncation_note(
        _BV(_FMT_STR), _func(block), None, call_addr, [_fmt_ptr(0x5000)] * 5, "f") is None


def test_truncation_note_silent_without_stack_stores(monkeypatch):
    # Format-string arg but NO outgoing stack stores -> nothing dropped -> silent.
    rts = _load_bridge(monkeypatch).read_taint_slice
    call_addr = 0x1c
    block = [_call(call_addr)]
    assert rts._call_model_truncation_note(
        _BV(_FMT_STR), _func(block), None, call_addr, [_fmt_ptr(0x5000)], "f") is None


def test_truncation_note_silent_on_isolated_local_spill(monkeypatch):
    # Format-string arg but a single sp+0x8 store (a local spill, not an arg run) ->
    # silent. Exercises the contiguity guard independent of the format signal.
    rts = _load_bridge(monkeypatch).read_taint_slice
    call_addr = 0x1c
    block = [_store(_add(_reg("sp"), _const(8)), 0x14), _call(call_addr)]
    assert rts._call_model_truncation_note(
        _BV(_FMT_STR), _func(block), None, call_addr, [_fmt_ptr(0x5000)], "f") is None


def test_truncation_note_silent_on_noncontiguous_stores(monkeypatch):
    rts = _load_bridge(monkeypatch).read_taint_slice
    call_addr = 0x1c
    block = [_store(_reg("sp"), 0x10),
             _store(_add(_reg("sp"), _const(0x40)), 0x12),
             _call(call_addr)]
    assert rts._call_model_truncation_note(
        _BV(_FMT_STR), _func(block), None, call_addr, [_fmt_ptr(0x5000)], "f") is None


def test_stack_arg_offsets_reset_at_preceding_call(monkeypatch):
    # #489 review root-cause-A: an EARLIER call's stack stores must not be
    # attributed to a LATER call in the same block. The scan resets after the
    # preceding call, so only the (empty) window after it is considered here.
    rts = _load_bridge(monkeypatch).read_taint_slice
    early, late = 0x10, 0x28
    block = [
        _store(_reg("sp"), 0x0c),                       # early call's arg
        _store(_add(_reg("sp"), _const(4)), 0x0e),      # early call's arg
        _call(early),                                    # preceding call
        _call(late),                                     # THIS call -- no stores of its own
    ]
    assert rts._stack_arg_store_offsets(_func(block), late) == []
    assert rts._stack_arg_store_offsets(_func(block), early) == [0, 4]


def test_truncation_note_silent_when_arch_unknown(monkeypatch):
    # Unknown calling convention (no int_arg_regs) -> can't reason -> silent.
    rts = _load_bridge(monkeypatch).read_taint_slice
    call_addr = 0x1c
    block = [_store(_reg("sp"), 0x10), _call(call_addr)]
    func = types.SimpleNamespace(low_level_il=[block],
                                 arch=types.SimpleNamespace(stack_pointer="sp"),
                                 calling_convention=None, name="f")
    assert rts._call_model_truncation_note(_BV(), func, None, call_addr, [_fmt_ptr(0x5000)], "f") is None
