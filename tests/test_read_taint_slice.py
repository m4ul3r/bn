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


class _BVFn(_BV):
    """Fake BinaryView adding `.get_function_at` so `resolve_call_target` can
    resolve a direct call's callee entry address."""
    def __init__(self, strings=None, funcs=None):
        super().__init__(strings)
        self._funcs = dict(funcs or {})

    def get_function_at(self, addr):
        return self._funcs.get(int(addr))


def test_truncation_note_named_callee_renders_runnable_command(monkeypatch):
    # #534: a NAMED callee still emits the runnable `bn proto set <name> ...`
    # command verbatim -- unchanged behavior.
    rts = _load_bridge(monkeypatch).read_taint_slice
    call_addr = 0x1c
    block = [_store(_reg("sp"), 0x10),
             _store(_add(_reg("sp"), _const(4)), 0x12),
             _call(call_addr)]
    note = rts._call_model_truncation_note(
        _BV(_FMT_STR), _func(block), None, call_addr, [_fmt_ptr(0x5000)], "my_logger")
    assert 'bn proto set my_logger "<ret> my_logger(<fixed args>, ...)"' in note


def test_truncation_note_prose_when_no_identifier(monkeypatch):
    # #534: an unnamed/indirect callee with NO resolvable address (call_insn=None)
    # must NOT emit the uncopy-pasteable `bn proto set the callee ...` command --
    # it emits prose guidance instead.
    rts = _load_bridge(monkeypatch).read_taint_slice
    call_addr = 0x1c
    block = [_store(_reg("sp"), 0x10),
             _store(_add(_reg("sp"), _const(4)), 0x12),
             _call(call_addr)]
    note = rts._call_model_truncation_note(
        _BV(_FMT_STR), _func(block), None, call_addr, [_fmt_ptr(0x5000)], None)
    assert note is not None
    assert "call-model truncation" in note
    # The broken literal command must be gone.
    assert "proto set the callee" not in note
    # Still guidance pointing at proto set on the resolved callee.
    assert "proto set" in note


def test_truncation_note_uses_address_selector_when_name_unknown(monkeypatch):
    # #534: no name, but the direct call resolves to a callee entry address -->
    # use the ADDRESS as a runnable `bn proto set 0x... ...` selector.
    rts = _load_bridge(monkeypatch).read_taint_slice
    call_addr = 0x1c
    block = [_store(_reg("sp"), 0x10),
             _store(_add(_reg("sp"), _const(4)), 0x12),
             _call(call_addr)]
    call_insn = block[-1]                       # .dest = _const(0x1000)
    callee = types.SimpleNamespace(name="", start=0x1000)
    bv = _BVFn(_FMT_STR, {0x1000: callee})
    note = rts._call_model_truncation_note(
        bv, _func(block), call_insn, call_addr, [_fmt_ptr(0x5000)], None)
    assert note is not None
    assert "bn proto set 0x1000" in note
    assert "proto set the callee" not in note


def test_truncation_note_prose_when_indirect_call_unresolvable(monkeypatch):
    # #534: a real indirect call_insn (register-indirect dest, no constant and no
    # symbol name) resolves to `.address is None` -- exercise the prose branch via
    # the resolver returning no address, not just the call_insn=None shortcut.
    rts = _load_bridge(monkeypatch).read_taint_slice
    call_addr = 0x1c
    indirect = _op("LLIL_CALL")
    indirect.address = call_addr
    indirect.dest = _reg("r3")                  # register-indirect: no .constant/.name
    block = [_store(_reg("sp"), 0x10),
             _store(_add(_reg("sp"), _const(4)), 0x12),
             indirect]
    bv = _BVFn(_FMT_STR, {})                     # get_function_at -> None for any addr
    note = rts._call_model_truncation_note(
        bv, _func(block), indirect, call_addr, [_fmt_ptr(0x5000)], None)
    assert note is not None
    assert "call-model truncation" in note
    assert "proto set the callee" not in note
    assert "proto set" in note


def test_truncation_note_silent_when_arch_unknown(monkeypatch):
    # Unknown calling convention (no int_arg_regs) -> can't reason -> silent.
    rts = _load_bridge(monkeypatch).read_taint_slice
    call_addr = 0x1c
    block = [_store(_reg("sp"), 0x10), _call(call_addr)]
    func = types.SimpleNamespace(low_level_il=[block],
                                 arch=types.SimpleNamespace(stack_pointer="sp"),
                                 calling_convention=None, name="f")
    assert rts._call_model_truncation_note(_BV(), func, None, call_addr, [_fmt_ptr(0x5000)], "f") is None


# --- #552: top-level frontier/terminal roll-up ------------------------------


def _step(reason, *, terminates=True, **extra):
    return {"ssa_label": extra.pop("ssa_label", "v#1"), "reason": reason,
            "terminates": terminates, **extra}


def test_summarize_frontiers_groups_and_counts(monkeypatch):
    # Terminal steps roll up per reason; non-terminal steps are excluded.
    rts = _load_bridge(monkeypatch).read_taint_slice
    trace = [
        _step("definition", terminates=False),           # excluded (not terminal)
        _step("call_or_jump_boundary", callee="strlen", address="0x10"),
        _step("call_or_jump_boundary", callee="memcpy", address="0x14"),
        _step("call_or_jump_boundary", callee="strlen", address="0x18"),
        _step("undefined_or_global", address="0x20"),
    ]
    fr = rts._summarize_frontiers(trace)
    # Ordered by descending count: 3 call boundaries before 1 undefined.
    assert [g["reason"] for g in fr] == ["call_or_jump_boundary", "undefined_or_global"]
    assert [g["count"] for g in fr] == [3, 1]
    # Examples carry the compact machine-consumable subset (callee, address).
    boundary = fr[0]["examples"]
    assert boundary[0]["callee"] == "strlen" and boundary[0]["address"] == "0x10"
    assert boundary[1]["callee"] == "memcpy"


def test_summarize_frontiers_caps_examples_at_three(monkeypatch):
    rts = _load_bridge(monkeypatch).read_taint_slice
    trace = [_step("undefined_or_global", address=hex(i)) for i in range(10)]
    fr = rts._summarize_frontiers(trace)
    assert fr[0]["count"] == 10               # count reflects ALL terminals
    assert len(fr[0]["examples"]) == 3        # examples capped


def test_summarize_frontiers_empty_when_no_terminals(monkeypatch):
    rts = _load_bridge(monkeypatch).read_taint_slice
    assert rts._summarize_frontiers([]) == []
    assert rts._summarize_frontiers([_step("definition", terminates=False)]) == []


def test_summarize_frontiers_ties_broken_by_reason_name(monkeypatch):
    # Equal counts -> deterministic order by reason name.
    rts = _load_bridge(monkeypatch).read_taint_slice
    trace = [_step("memory_load"), _step("function_parameter")]
    fr = rts._summarize_frontiers(trace)
    assert [g["reason"] for g in fr] == ["function_parameter", "memory_load"]


def test_backward_slice_result_includes_frontiers(monkeypatch):
    # End-to-end: `_backward_slice` result carries a top-level `frontiers` roll-up
    # derived from the trace's terminal steps (#552).
    from _bridge_fakes import (
        _FakeBV, _FakeFunction, _FakeMLILFunction, _FakeMLILInsn, _FakeSSAVariable,
    )
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    var_r0 = _FakeSSAVariable("r0#1")
    var_r1 = _FakeSSAVariable("r1#2")
    call_insn = _FakeMLILInsn(
        0x10010, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA", vars_read=[var_r0])],
        vars_read=[var_r0])
    def_insn = _FakeMLILInsn(0x10008, operation="MLIL_SET_VAR_SSA", vars_read=[var_r1])
    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(
        instructions=[call_insn], definitions={var_r0: def_insn})
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._backward_slice("active", "test_func", "0x10010", arg_index=0)

    assert "frontiers" in result
    # The chain terminates once, at an undefined/no-reaching-def leaf.
    assert result["frontiers"] == [
        {"reason": "undefined_or_global", "count": 1,
         "examples": [{"ssa_label": "r1#2", "depth": 1}]},
    ]
    # #671: a complete slice claims completeness with an empty assumptions list.
    assert result["assumptions"] == []


def test_render_trace_frontiers_text(monkeypatch):
    # Text mode shows a concise frontier roll-up with counts + named callees.
    from bn.formatters import _render_trace_frontiers
    frontiers = [
        {"reason": "call_or_jump_boundary", "count": 2,
         "examples": [{"callee": "strlen"}, {"callee": "memcpy"}]},
        {"reason": "undefined_or_global", "count": 1, "examples": [{"address": "0x20"}]},
    ]
    out = "\n".join(_render_trace_frontiers(frontiers))
    assert "frontiers (3 terminal step(s) in 2 group(s)):" in out
    assert "call boundary  x2 (strlen, memcpy)" in out
    assert "undefined / no reaching definition  x1" in out
    # Empty / missing frontiers render nothing.
    assert _render_trace_frontiers([]) == []
    assert _render_trace_frontiers(None) == []


def test_backward_slice_assumptions_nonempty_when_truncated(monkeypatch):
    # #671: `assumptions` names the depth cap when the walk hits max_depth.
    from _bridge_fakes import (
        _FakeBV, _FakeFunction, _FakeMLILFunction, _FakeMLILInsn, _FakeSSAVariable,
    )
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    var_r0 = _FakeSSAVariable("r0#1")
    var_r1 = _FakeSSAVariable("r1#2")
    call_insn = _FakeMLILInsn(
        0x10010, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA", vars_read=[var_r0])],
        vars_read=[var_r0])
    def_insn = _FakeMLILInsn(0x10008, operation="MLIL_SET_VAR_SSA", vars_read=[var_r1])
    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(
        instructions=[call_insn], definitions={var_r0: def_insn})
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._backward_slice(
        "active", "test_func", "0x10010", arg_index=0, max_depth=1)

    assert result["truncated"] is True
    assert len(result["assumptions"]) == 1
    assert "depth cap reached (1)" in result["assumptions"][0]


def test_intra_mode_out_param_fill_not_mislabeled_undefined(monkeypatch):
    # #672: a value that bottoms out at a local whose address was passed into
    # a call must be labeled `out_param_not_followed` (naming the callee) in
    # INTRA mode too, not the misleading `undefined_or_global`.
    from _bridge_fakes import _FakeSSAVariable, _FakeSSAFunction
    rts = _load_bridge(monkeypatch).read_taint_slice

    local_var = object()  # sentinel local; no `identifier`, hashable by identity
    ssa_var = _FakeSSAVariable("local#1")
    ssa_var.var = local_var

    ssa_func = _FakeSSAFunction([], {})
    monkeypatch.setattr(
        rts, "_build_out_param_map",
        lambda ctx, bv, ssa_func: {local_var: [(0x10, "parse_input")]})

    trace = rts._build_backward_trace(
        ctx=None, bv=None, ssa_func=ssa_func, initial_vars=[ssa_var], max_depth=50,
        interprocedural=False, seed_addr=0x20)

    assert len(trace) == 1
    entry = trace[0]
    assert entry["reason"] == "out_param_not_followed"
    assert entry["out_param_callee"] == "parse_input"


def test_interprocedural_mode_out_param_fill_reason_unchanged(monkeypatch):
    # Interprocedural mode keeps its existing, distinctly-named reason.
    from _bridge_fakes import _FakeSSAVariable, _FakeSSAFunction
    rts = _load_bridge(monkeypatch).read_taint_slice

    local_var = object()  # sentinel local; hashable by identity
    ssa_var = _FakeSSAVariable("local#1")
    ssa_var.var = local_var
    ssa_func = _FakeSSAFunction([], {})
    monkeypatch.setattr(
        rts, "_build_out_param_map",
        lambda ctx, bv, ssa_func: {local_var: [(0x10, "parse_input")]})

    trace = rts._build_backward_trace(
        ctx=None, bv=None, ssa_func=ssa_func, initial_vars=[ssa_var], max_depth=50,
        interprocedural=True, seed_addr=0x20)

    assert trace[0]["reason"] == "interprocedural_out_param_not_followed"
    assert trace[0]["out_param_callee"] == "parse_input"


def test_backward_slice_assumptions_nonempty_when_ip_depth_exhausted(monkeypatch):
    # #671 (round-2): `assumptions` must name the `--ip-depth` cap, not just
    # `--max-depth`, when a crossable call boundary is stopped by the budget.
    from _bridge_fakes import (
        _FakeBV, _FakeConstPtr, _FakeFunction, _FakeMLILFunction, _FakeMLILInsn,
        _FakeSSAVariable,
    )
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    # Callee: returns callee_ret_var, which is a copy of a parameter -- the
    # crossing would otherwise fire (matches test_backward_slice_interprocedural_
    # follows_callee's shape).
    callee_ret_var = _FakeSSAVariable("result#1")
    callee_def_var = _FakeSSAVariable("tmp#2")
    callee_ret_insn = _FakeMLILInsn(0x20010, operation="MLIL_RET", vars_read=[callee_ret_var])
    callee_def_insn = _FakeMLILInsn(0x20008, operation="MLIL_SET_VAR_SSA", vars_read=[callee_def_var])
    callee = _FakeFunction(0x20000, "callee_fn")
    callee.medium_level_il = _FakeMLILFunction(
        instructions=[callee_ret_insn, callee_def_insn],
        definitions={callee_ret_var: callee_def_insn})

    ret_var = _FakeSSAVariable("r0#3")
    inner_call_insn = _FakeMLILInsn(0x1000c, operation="MLIL_CALL_SSA", dest=_FakeConstPtr(0x20000))
    target_call_insn = _FakeMLILInsn(
        0x10010, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA", vars_read=[ret_var])],
        vars_read=[ret_var])
    caller = _FakeFunction(0x10000, "caller_fn")
    caller.medium_level_il = _FakeMLILFunction(
        instructions=[inner_call_insn, target_call_insn],
        definitions={ret_var: inner_call_insn})
    bv = _FakeBV(functions=[caller, callee])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # --ip-depth 0: the budget is spent before the very first crossing attempt.
    result = instance._backward_slice(
        "active", "caller_fn", "0x10010", arg_index=0,
        interprocedural=True, ip_depth=0)

    assert result["truncated"] is False
    entry = result["trace"][0]
    assert entry["reason"] == "call_or_jump_boundary"
    assert entry["ip_depth_exhausted"] is True
    assert "call_depth_exhausted" not in entry
    assert len(result["assumptions"]) == 1
    assert "--ip-depth cap" in result["assumptions"][0]
    assert result["frontiers"] == [
        {"reason": "call_or_jump_boundary", "count": 1,
         "examples": [{"ssa_label": "r0#3", "address": "0x1000c", "depth": 0,
                       "callee": "callee_fn"}]},
    ]


def test_build_backward_trace_call_depth_guard_marks_boundary_not_dangling_cross_function(monkeypatch):
    # #671 (round-2): the hard `_call_depth > 10` recursion backstop must not
    # leave a dangling non-terminal `cross_function` step with empty
    # `frontiers`/`assumptions` (R2 finding A2) -- the crossing guard now
    # requires `_call_depth < 10` as a PRE-condition, so the boundary itself
    # is marked `call_depth_exhausted` instead.
    from _bridge_fakes import (
        _FakeBV, _FakeConstPtr, _FakeFunction, _FakeMLILFunction, _FakeMLILInsn,
        _FakeSSAVariable,
    )
    bridge = _load_bridge(monkeypatch)
    rts = bridge.read_taint_slice
    instance = bridge.BinaryNinjaBridge()

    callee_ret_var = _FakeSSAVariable("result#1")
    callee_def_var = _FakeSSAVariable("tmp#2")
    callee_ret_insn = _FakeMLILInsn(0x20010, operation="MLIL_RET", vars_read=[callee_ret_var])
    callee_def_insn = _FakeMLILInsn(0x20008, operation="MLIL_SET_VAR_SSA", vars_read=[callee_def_var])
    callee = _FakeFunction(0x20000, "callee_fn")
    callee.medium_level_il = _FakeMLILFunction(
        instructions=[callee_ret_insn, callee_def_insn],
        definitions={callee_ret_var: callee_def_insn})

    ret_var = _FakeSSAVariable("r0#3")
    inner_call_insn = _FakeMLILInsn(0x1000c, operation="MLIL_CALL_SSA", dest=_FakeConstPtr(0x20000))
    caller = _FakeFunction(0x10000, "caller_fn")
    caller.medium_level_il = _FakeMLILFunction(
        instructions=[inner_call_insn], definitions={ret_var: inner_call_insn})
    bv = _FakeBV(functions=[caller, callee])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # ip_depth=5 (budget not spent) but _call_depth=10 (hard backstop already
    # reached) -- reproduces R2's finding A2 repro exactly.
    trace = rts._build_backward_trace(
        instance.ctx, bv, caller.medium_level_il.ssa_form, [ret_var], max_depth=10,
        interprocedural=True, ip_depth=5, _call_depth=10, seed_addr=None)

    assert len(trace) == 1
    entry = trace[0]
    assert entry["reason"] == "call_or_jump_boundary"
    assert entry["terminates"] is True
    assert entry.get("call_depth_exhausted") is True
    assert "ip_depth_exhausted" not in entry
    assert "cross_function" not in entry
    assert rts._summarize_frontiers(trace) == [
        {"reason": "call_or_jump_boundary", "count": 1,
         "examples": [{"ssa_label": "r0#3", "address": "0x1000c", "depth": 0,
                       "callee": "callee_fn"}]},
    ]
    assert rts._trace_assumptions(trace, truncated=False, max_depth=10) == [
        "interprocedural crossing stopped at the internal recursion-depth "
        "safety limit (not adjustable via --ip-depth); the slice may be "
        "incomplete beyond that boundary"
    ]


def test_output_pointer_hint_names_callee_not_param_name(monkeypatch):
    # #662 (round-2, finding C): the output-pointer hint must name the resolved
    # callee, not the callee's PARAMETER name -- the same confusion the header
    # fix closes, in the one remaining internal consumer of `arg_label["name"]`.
    import types as _types
    from _bridge_fakes import (
        _FakeBV, _FakeConstPtr, _FakeFunction, _FakeMLILFunction, _FakeMLILInsn,
        _FakeSSAVariable,
    )
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    local = _FakeSSAVariable("rec#1")
    addr_of_local = _FakeMLILInsn(0x2010, operation="MLIL_ADDRESS_OF", src=local)
    call_insn = _FakeMLILInsn(
        0x2010, operation="MLIL_CALL_SSA",
        params=[addr_of_local], dest=_FakeConstPtr(0x3000))
    fill_record = _FakeFunction(0x3000, "fill_record")
    fill_record.parameter_vars = [_types.SimpleNamespace(name="out")]
    caller = _FakeFunction(0x2000, "caller")
    caller.medium_level_il = _FakeMLILFunction(instructions=[call_insn], definitions={})
    bv = _FakeBV(functions=[caller, fill_record])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._backward_slice("active", "caller", "0x2010", arg_index=0)

    assert result["arg_label"]["callee"] == "fill_record"
    assert result["arg_label"]["name"] == "out"
    assert len(result["hints"]) == 1
    assert "fill_record writes into the pointee" in result["hints"][0]
    assert "out writes into the pointee" not in result["hints"][0]


def test_backward_slice_assumptions_nonempty_for_unfollowed_out_param(monkeypatch):
    # #671/#672: `_backward_slice`'s `assumptions` must name an unfollowed
    # out-param fill, not just the depth cap -- end-to-end through
    # `_trace_assumptions`'s out-param branch (read_taint_slice.py:965-972),
    # which the direct `_build_backward_trace` tests never reach.
    from _bridge_fakes import (
        _FakeBV, _FakeFunction, _FakeMLILFunction, _FakeMLILInsn, _FakeSSAVariable,
    )
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    var_r0 = _FakeSSAVariable("r0#1")
    call_insn = _FakeMLILInsn(
        0x10010, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA", vars_read=[var_r0])],
        vars_read=[var_r0])
    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(instructions=[call_insn], definitions={})
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(
        bridge.read_taint_slice, "_build_out_param_map",
        lambda ctx, bv, ssa_func: {var_r0: [(0x5, "parse_input")]})

    result = instance._backward_slice("active", "test_func", "0x10010", arg_index=0)

    assert result["trace"][0]["reason"] == "out_param_not_followed"
    assert len(result["assumptions"]) == 1
    assert "parse_input" in result["assumptions"][0]
