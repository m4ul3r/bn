"""IC02: every selected sink site and origin has its own truthful accounting."""
from __future__ import annotations

import pytest
from bn.formatters import _render_taint_text
from bn_agent_bridge.taint_result import backward_diagnostics
from test_taint_backward_coverage import (
    _NAMES, _MEMCPY, _RECV, _backward, _call, _const, _length_sink, _var,
)
from test_taint_engine import FBV, FExpr, FFunc, FSite, FSSA, FSSAFunc, FVar, te


def _sites(operands):
    n = FVar("n", ident=101)
    n0 = FSSA(n, 0)
    calls = []
    for i, operand in enumerate(operands):
        arg = _var(n0) if operand == "variable" else _const(operand) if isinstance(operand, int) else operand
        params = [_const(0x5000), _const(0x6000)]
        if arg != "missing":
            params.append(arg)
        calls.append(_call(i, 0x810 + 0x10 * i, _MEMCPY, params))
    return FFunc("copy_sites", 0x800, FSSAFunc(calls), params=[n])


@pytest.mark.parametrize("operands, statuses, complete", [
    (["variable", "variable"], ["seeded", "seeded"], True),
    ([16, 4096], ["constant", "constant"], True),
    (["variable", 4096], ["seeded", "constant"], True),
    (["missing", "variable"], ["unseeded", "seeded"], False),
    (["missing", 4096], ["unseeded", "constant"], False),
    ([FExpr("MLIL_CONST_PTR", "fixed address", constant=0x7000), "variable"],
     ["unseeded", "seeded"], False),
    ([FExpr("MLIL_ADD", "fixed expression"), "variable"], ["unseeded", "seeded"], False),
])
def test_arg_locator_accounts_for_every_matching_site(operands, statuses, complete):
    result = _backward(_sites(operands), _NAMES)
    rows = result["sink_status"]
    assert [(r["address"], r["status"]) for r in rows] == [
        (hex(0x810 + 0x10 * i), status) for i, status in enumerate(statuses)]
    assert result["diagnostics"]["sinks_seeded"] == statuses.count("seeded")
    assert result["diagnostics"]["safe_to_report_complete_slice"] is complete
    constants = [r for r in rows if r["status"] == "constant"]
    assert [r["scalar"]["value"] for r in constants] == [v for v in operands if isinstance(v, int)]
    assert all(r["scalar"]["kind"] == "constant" and r["destination_fit"] == "unproved"
               and not r["seeded"] and "bounded" not in r for r in constants)
    text = _render_taint_text(result)
    for row in rows:
        assert row["address"] in text
        assert row["status"] in text
    for row in constants:
        assert hex(row["scalar"]["value"]) in text
    assert "provably bounded" not in text
    if constants:
        assert "capacity" in text
        assert "check that the --sink" not in result["diagnostics"]["next_action"]
    if not complete:
        assert [lf["address"] for lf in result["leaves"] if lf["kind"] == "sink_unseeded"] == ["0x810"]


def test_absent_locator_and_all_missing_still_refuse():
    engine = te.TaintEngine(FBV(_NAMES), te.load_models())
    with pytest.raises(te.TaintError, match="no call"):
        engine.backward(_sites(["variable"]), [te.parse_locator("arg:malloc:0")])
    with pytest.raises(te.TaintError, match="out of range"):
        engine.backward(_sites(["missing", "missing"]), [te.parse_locator("arg:memcpy:2")])


@pytest.mark.parametrize("opcode", ["MLIL_CONST", "MLIL_CONST_PTR"])
def test_constant_pointer_demand_is_not_a_scalar_failure(opcode):
    pointer = FExpr(opcode, "0x7000", constant=0x7000)
    fn = FFunc("run_hook", 0x800, FSSAFunc([_call(0, 0x810, 0x990, [pointer])]))
    result = te.TaintEngine(FBV({0x990: "popen"}), te.load_models()).backward(
        fn, [te.parse_locator("arg:popen:0")])
    [row] = result["sink_status"]
    assert (row["address"], row["status"], row["seeded"]) == ("0x810", "demand", False)
    assert "scalar" not in row and "bounded" not in row
    assert result["diagnostics"]["sinks_seeded"] == 0
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    assert result["leaves"][0]["kind"] == "pointer_demand_not_certified"
    assert result["leaves"][0]["objects"][0]["address"] == "0x7000"
    text = _render_taint_text(result)
    assert "demand" in text and "0x810" in text
    # These are contradictory scalar failure/capacity claims, not wording pins.
    for claim in ("reads no variable", "no def-chain", "def-chain was not followed", "UNSEEDED SINKS", "provably bounded"):
        assert claim not in text


@pytest.mark.parametrize("origin_kind", ["parameter", "source", "entry"])
@pytest.mark.parametrize("same_function", [False, True])
def test_origin_identity_survives_public_slice_dedup(origin_kind, same_function):
    sink = _length_sink("copy_len", 0x800, 10)
    callers = []
    for i in range(1 if same_function else 2):
        start = 0xa00 + i * 0x100
        # Deliberately identical names: addresses, not presentation, identify functions.
        v = FVar("input", ident=50)
        value = FSSA(v, 1 if origin_kind == "source" else 0)
        ins = []
        if origin_kind == "source":
            ins.append(_call(0, start + 4, _RECV, [], writes=[value]))
        ins.append(_call(len(ins), start + 0x10, sink.start, [_const(0), _const(0), _var(value)]))
        if same_function:
            ins.append(_call(len(ins), start + 0x20, sink.start, [_const(0), _const(0), _var(value)]))
        caller = FFunc("caller", start, FSSAFunc(ins), params=[v] if origin_kind == "parameter" else [])
        callers.append(caller)
    sink.caller_sites = [FSite(c, ins.address) for c in callers for ins in c.mlil.ssa_form.instructions
                         if ins.dest.constant == sink.start]
    result = _backward(sink, {**_NAMES, sink.start: sink.name})
    rows = result["slices"]
    assert len(rows) == len(callers)
    assert {r["origin"]["kind"] for r in rows} == {origin_kind}
    assert {r["origin"]["function"] for r in rows} == {hex(c.start) for c in callers}
    assert [r.get("reached_via_call_sites", 1) for r in rows] == ([2] if same_function else [1, 1])
    assert result["diagnostics"]["safe_to_report_complete_slice"] is True
    text = _render_taint_text(result)
    assert all(hex(c.start) in text for c in callers)


@pytest.mark.parametrize("counts", [(1,), (4,), (12,), (1, 4, 12)])
def test_dropped_callers_sums_sites_not_frontier_rows(counts):
    leaves = [{"kind": "caller_sites_truncated", "callers_dropped": count,
               "callers_total": count + 16, "callers_followed": 16,
               "function": {"name": f"handler_{i}", "address": hex(0x800 + i * 0x100)},
               "reasons": {"caller_cap": count}} for i, count in enumerate(counts)]
    diag = backward_diagnostics(leaves, [], sinks_seeded=1, slices=1)
    assert diag["frontier"]["dropped_callers"] == sum(counts)
    assert diag["frontier"]["by_kind"]["caller_sites_truncated"] == len(counts)
    assert diag["safe_to_report_complete_slice"] is False
    text = _render_taint_text({"direction": "backward", "leaves": leaves, "diagnostics": diag})
    assert f"{sum(counts)} dropped-caller-site(s)" in text


@pytest.mark.parametrize("resolution", ["value-set", "agent-map"])
def test_constant_site_retains_indirect_callee_attribution(resolution):
    from test_taint_engine import FPVS

    fn = _sites([64])
    call = fn.mlil.ssa_form.instructions[0]
    call.dest = FExpr("MLIL_VAR_SSA", "callback", possible_values=(
        FPVS("InSetOfValues", values=[_MEMCPY, 0x990, 0x991])
        if resolution == "value-set" else None))
    engine = te.TaintEngine(
        FBV(_NAMES), te.load_models(),
        resolve_map={"0x810": [hex(_MEMCPY)]} if resolution == "agent-map" else {})
    result = engine.backward(fn, [te.parse_locator("arg:memcpy:2")])
    # The qualification is public provenance, not a definite call to memcpy.
    assert any("0x810" in a and resolution in a for a in result["assumptions"])
    if resolution == "value-set":
        assert any("1 of 3 candidate targets" in a for a in result["assumptions"])


def test_all_sites_fail_refusal_identifies_each_unseeded_site():
    with pytest.raises(te.TaintError) as caught:
        _backward(_sites(["missing", "missing"]), _NAMES)
    lines = str(caught.value).splitlines()
    for address in ("0x810", "0x820"):
        matching = [line for line in lines if address in line]
        assert len(matching) == 1
        assert "arg:memcpy:2" in matching[0] and "[unseeded]" in matching[0]
    assert len(set(lines)) == len(lines)


def test_constant_capacity_is_rendered_once_from_structured_evidence():
    result = _backward(_sites([4096]), _NAMES)
    # Notes are not a second authority for a constant's structured fit verdict.
    result["sink_status"][0]["note"] = "STALE_CAPACITY_PROSE"
    text = _render_taint_text(result)
    [line] = [line for line in text.splitlines() if "[constant]" in line]
    assert "0x1000" in line and "unproved" in line
    assert line.count("capacity") == 1
    assert "STALE_CAPACITY_PROSE" not in line


@pytest.mark.parametrize("same_object", [False, True])
def test_same_function_field_origins_keep_distinct_global_slots(same_object):
    from test_taint_engine import FInstr

    sink = _length_sink("copy_len", 0x800, 10)
    instructions = []
    for i, base in enumerate([0x4000, 0x4000 if same_object else 0x5000]):
        value = FSSA(FVar(f"size_{i}", ident=80 + i), 1)
        load = FExpr("MLIL_LOAD_SSA", f"[{base:#x}]",
                     src=FExpr("MLIL_CONST_PTR", hex(base), constant=base), src_memory=0)
        load.size = 8
        instructions.append(FInstr(2 * i, 0xa00 + 0x10 * i, "MLIL_SET_VAR_SSA",
                                   f"size_{i} = [{base:#x}]", writes=[value], src=load))
        instructions.append(_call(2 * i + 1, 0xa04 + 0x10 * i, sink.start,
                                  [_const(0), _const(0), _var(value)]))
    caller = FFunc("load_sizes", 0xa00, FSSAFunc(instructions))
    sink.caller_sites = [FSite(caller, ins.address) for ins in instructions
                         if ins.operation.name == "MLIL_CALL_SSA"]
    result = _backward(sink, {**_NAMES, sink.start: sink.name})
    rows = result["slices"]
    assert len(rows) == (1 if same_object else 2)
    assert {r["origin"]["base"] for r in rows} == ({"0x4000"} if same_object else {"0x4000", "0x5000"})
    assert all(r["origin"]["function"] == "0xa00" for r in rows)
    assert [r.get("reached_via_call_sites", 1) for r in rows] == ([2] if same_object else [1, 1])
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


@pytest.mark.parametrize("same_function", [False, True])
def test_literal_origins_keep_function_identity_and_same_origin_collapse(same_function):
    sink = _length_sink("copy_len", 0x800, 10)
    callers = []
    for i in range(1 if same_function else 2):
        start = 0xa00 + 0x100 * i
        calls = [_call(0, start + 4, sink.start, [_const(0), _const(0), _const(64)])]
        if same_function:
            calls.append(_call(1, start + 8, sink.start, [_const(0), _const(0), _const(64)]))
        callers.append(FFunc("caller", start, FSSAFunc(calls)))
    sink.caller_sites = [FSite(c, ins.address) for c in callers for ins in c.mlil.ssa_form.instructions]
    result = _backward(sink, {**_NAMES, sink.start: sink.name})
    assert len(result["slices"]) == len(callers)
    assert {s["origin"]["function"] for s in result["slices"]} == {hex(c.start) for c in callers}
    assert all(s["origin"]["value"] == 64 for s in result["slices"])
    assert [s.get("reached_via_call_sites", 1) for s in result["slices"]] == ([2] if same_function else [1, 1])
    assert result["diagnostics"]["safe_to_report_complete_slice"] is True


def test_caller_frontier_text_preserves_each_functions_coverage():
    counts = [1, 4, 12]
    leaves = [{"kind": "caller_sites_truncated", "address": hex(0x800 + i * 0x100),
               "function": {"name": "same_name", "address": hex(0x800 + i * 0x100)},
               "callers_followed": 16, "callers_total": 16 + count, "callers_dropped": count}
              for i, count in enumerate(counts)]
    diag = backward_diagnostics(leaves, [], sinks_seeded=1, slices=1)
    text = _render_taint_text({"direction": "backward", "leaves": leaves, "diagnostics": diag})
    rows = [line for line in text.splitlines() if "caller_sites_truncated @" in line]
    assert len(rows) == 3
    for leaf in leaves:
        [row] = [line for line in rows if leaf["function"]["address"] in line]
        assert f"{leaf['callers_followed']} of {leaf['callers_total']}" in row
        assert f"{leaf['callers_dropped']} dropped" in row
    assert "17 dropped-caller-site(s)" in text


def test_caller_frontier_distinguishes_same_function_coverage_and_bounds_output():
    from bn.formatters import _render_grouped_leaves

    leaves = [{"kind": "caller_sites_truncated", "address": hex(0x800 + i * 0x100),
               "function": {"name": f"caller_{i}", "address": hex(0x800 + i * 0x100)},
               "callers_followed": 16, "callers_total": 17, "callers_dropped": 1}
              for i in range(20)]
    alternate = {**leaves[0], "callers_total": 20, "callers_dropped": 4}
    text = "\n".join(_render_grouped_leaves([leaves[0], alternate]))
    assert "1 dropped" in text and "4 dropped" in text
    bounded = "\n".join(_render_grouped_leaves(leaves))
    assert bounded.count("caller_sites_truncated @") == 12
    assert "8 more group(s)" in bounded and "--format json" in bounded


def test_same_named_local_buffers_in_two_functions_remain_distinct_origins():
    from test_taint_engine import _addr_of

    pointer = FVar("p", ident=60, typ="char*")
    use = _call(0, 0x804, 0x990, [_var(FSSA(pointer, 0)), _const(0)])
    use.src_memory = 0
    sink = FFunc("use_buffer", 0x800, FSSAFunc([use]), params=[pointer])
    callers = []
    for start in (0xa00, 0xb00):
        buf = FVar("buffer", ident=80, typ="char[16]")
        call = _call(0, start + 4, sink.start, [_addr_of(buf)])
        call.src_memory = 0
        callers.append(FFunc("caller", start, FSSAFunc([call])))
    sink.caller_sites = [FSite(c, c.start + 4) for c in callers]
    result = _backward(sink, {**_NAMES, 0x990: "popen", sink.start: sink.name}, "arg:popen:0")
    origins = [sl["origin"] for sl in result["slices"]]
    assert len(origins) == 2
    assert all(o["kind"] == "address_of_local_unresolved" and o["buffer"] == "buffer" for o in origins)
    assert {o["function"] for o in origins} == {"0xa00", "0xb00"}
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


@pytest.mark.parametrize("callee", [_RECV, 0x970], ids=["source", "unmodeled-call"])
def test_distinct_call_producers_in_one_function_are_distinct_origins(callee):
    sink = _length_sink("copy_len", 0x800, 10)
    instructions = []
    for i in range(2):
        value = FSSA(FVar(f"result_{i}", ident=90 + i), 1)
        instructions.append(_call(2 * i, 0xa00 + 0x10 * i, callee, [], writes=[value]))
        instructions.append(_call(2 * i + 1, 0xa04 + 0x10 * i, sink.start,
                                  [_const(0), _const(0), _var(value)]))
    caller = FFunc("caller", 0xa00, FSSAFunc(instructions))
    sink.caller_sites = [FSite(caller, 0xa04), FSite(caller, 0xa14)]
    result = _backward(sink, {**_NAMES, sink.start: sink.name})
    origins = [sl["origin"] for sl in result["slices"]]
    assert len(origins) == 2
    assert {o["address"] for o in origins} == {"0xa00", "0xa10"}
    assert all(o["function"] == "0xa00" for o in origins)
    assert result["diagnostics"]["safe_to_report_complete_slice"] is True


@pytest.mark.parametrize("difference", [None, "offset", "width"])
def test_field_origin_identity_includes_slot_offset_and_load_width(difference):
    from test_taint_backward_loads import _FieldProgram
    from test_taint_engine import FInstr

    sink = _length_sink("copy_len", 0x800, 10)
    fields = _FieldProgram()
    instructions = []
    for i in range(2):
        value = FSSA(FVar(f"size_{i}", ident=95 + i), 1)
        offset = 16 if i and difference == "offset" else 8
        width = 4 if i and difference == "width" else 8
        load = fields.load(fields.addr(fields.m0, offset), width)
        instructions.append(FInstr(2 * i, 0xa00 + 0x10 * i, "MLIL_SET_VAR_SSA",
                                   f"size_{i} = field", writes=[value], src=load))
        instructions.append(_call(2 * i + 1, 0xa04 + 0x10 * i, sink.start,
                                  [_const(0), _const(0), _var(value)]))
    caller = FFunc("load_fields", 0xa00, FSSAFunc(instructions), params=[fields.m])
    sink.caller_sites = [FSite(caller, 0xa04), FSite(caller, 0xa14)]
    result = _backward(sink, {**_NAMES, sink.start: sink.name})
    origins = [sl["origin"] for sl in result["slices"]]
    assert len(origins) == (2 if difference else 1)
    assert {o["function"] for o in origins} == {"0xa00"}
    assert {o["base"] for o in origins} == {"m#0"}
    assert {o["offset"] for o in origins} == ({"0x8", "0x10"} if difference == "offset" else {"0x8"})
    assert {o["width"] for o in origins} == ({4, 8} if difference == "width" else {8})


def test_different_opaque_writers_of_one_local_are_distinct_origins():
    from test_taint_engine import _addr_of

    pointer = FVar("p", ident=60, typ="char*")
    use = _call(0, 0x804, 0x990, [_var(FSSA(pointer, 0)), _const(0)])
    use.src_memory = 0
    sink = FFunc("use_buffer", 0x800, FSSAFunc([use]), params=[pointer])
    buf = FVar("buffer", ident=80, typ="char[16]")
    instructions, mem_defs = [], {}
    for i in range(2):
        # Alternative branches consume the same local after different calls to
        # the same opaque writer; neither consumer precedes the other.
        writer = _call(2 * i, 0xa00 + i * 0x10, 0x991, [_addr_of(buf)])
        writer.src_memory, writer.dest_memory = 0, 2 * i + 1
        consumer = _call(2 * i + 1, 0xa04 + i * 0x10, sink.start, [_addr_of(buf)])
        consumer.src_memory, consumer.dest_memory = writer.dest_memory, 2 * i + 2
        instructions.extend([writer, consumer])
        mem_defs.update({writer.dest_memory: writer, consumer.dest_memory: consumer})
    caller = FFunc("caller", 0xa00, FSSAFunc(instructions, mem_defs=mem_defs))
    sink.caller_sites = [FSite(caller, 0xa04), FSite(caller, 0xa14)]
    result = _backward(sink, {**_NAMES, 0x990: "popen", 0x991: "compose",
                              sink.start: sink.name}, "arg:popen:0")
    origins = [sl["origin"] for sl in result["slices"]]
    assert len(origins) == 2
    assert all(o["function"] == "0xa00" and o["buffer"] == "buffer" for o in origins)
    assert {o["writers"][0]["address"] for o in origins} == {"0xa00", "0xa10"}


def _two_formal_caller_query(caller_arguments, *, shared_sink=False, repeat_formal=False,
                             indirect=(), reverse=False):
    from test_taint_backward_coverage import _BV

    dst, n, m = (FVar(name, ident=110 + i) for i, name in enumerate(("dst", "n", "m")))
    n0, m0 = FSSA(n, 0), FSSA(m, 0)
    operands = ([_var(n0), _var(n0 if repeat_formal else m0)] if not shared_sink else
                [FExpr("MLIL_ADD", "n + m", reads=[n0, m0])])
    calls = [_call(i, 0x804 + i * 4, _MEMCPY, [_const(0), _const(0), arg])
             for i, arg in enumerate(operands)]
    if reverse:
        calls.reverse()
    target = FFunc("top3", 0x800, FSSAFunc(calls), params=[dst, n, m])
    for i, args in enumerate(caller_arguments):
        start = 0xa00 + i * 0x100
        call = _call(0, start + 4, target.start, [_const(0), *args])
        caller = FFunc(f"caller_{i}", start, FSSAFunc([call]))
        target.caller_sites.append(FSite(caller, call.address))
    bv = _BV({**_NAMES, target.start: target.name}, data_refs={target.start: list(indirect)})
    return target, bv, te.TaintEngine(bv, te.load_models())


@pytest.mark.parametrize("shared_sink", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_dropped_references_not_multiplied_by_demanded_formals(shared_sink, reverse):
    target, bv, engine = _two_formal_caller_query(
        [(_const(64), _const(32))], shared_sink=shared_sink,
        indirect=[0x403000], reverse=reverse)
    # The first formal also ascends through a caller with its own stored-address
    # boundary, reproducing two distinct dropped references across two functions.
    external = FVar("external", ident=120)
    caller = target.caller_sites[0].function
    caller.parameter_vars = [external]
    call = caller.mlil.ssa_form.instructions[0]
    call.params[1] = _var(FSSA(external, 0))
    call.vars_read = [FSSA(external, 0)]
    bv._data_refs[caller.start] = [0x403008]
    result = engine.backward(target, [te.parse_locator("arg:memcpy:2")])
    leaves = [lf for lf in result["leaves"] if lf["kind"] == "caller_sites_truncated"]
    assert len(leaves) == 2
    by_function = {lf["function"]["address"]: lf for lf in leaves}
    top = by_function["0x800"]
    assert (top["callers_total"], top["callers_followed"], top["callers_dropped"]) == (2, 1, 1)
    assert top["reasons"] == {"indirect_reference": 2}
    assert {(row["address"], row["param"]) for row in top["unfollowed"]} == {
        ("0x403000", 1), ("0x403000", 2)}
    assert by_function["0xa00"]["callers_dropped"] == 1
    assert result["diagnostics"]["frontier"]["dropped_callers"] == 2
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    text = _render_taint_text(result)
    assert "2 dropped-caller-site(s)" in text
    rows = [ln for ln in text.splitlines() if "caller_sites_truncated @" in ln]
    assert len(rows) == 2 and all("[2 leaves]" not in ln for ln in rows)


@pytest.mark.parametrize("overlap", [False, True])
def test_caller_coverage_unions_disjoint_and_overlapping_reference_losses(overlap):
    fixed = FExpr("MLIL_CONST_PTR", "fixed address", constant=0x4000)
    args = [(fixed, _const(32)), (_const(64), fixed)]
    if overlap:
        args.insert(0, (fixed, fixed))
    args.append((_const(64), _const(32)))
    target, _, engine = _two_formal_caller_query(args)
    result = engine.backward(target, [te.parse_locator("arg:memcpy:2")])
    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "caller_sites_truncated"]
    expected = 3 if overlap else 2
    assert (leaf["callers_total"], leaf["callers_followed"], leaf["callers_dropped"]) == (
        len(args), 1, expected)
    assert leaf["reasons"] == {"constant_address": 4 if overlap else 2}
    assert len({row["address"] for row in leaf["unfollowed"]}) == expected
    assert {row["param"] for row in leaf["unfollowed"]} == {1, 2}
    assert result["diagnostics"]["frontier"]["dropped_callers"] == expected
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


@pytest.mark.parametrize("indirect_count", [1, 20])
def test_capped_reference_union_uses_all_sites_not_bounded_examples(indirect_count):
    target, _, engine = _two_formal_caller_query(
        [(_const(i), _const(i + 1)) for i in range(20)],
        indirect=[0x404000 + 8 * i for i in range(indirect_count)])
    result = engine.backward(target, [te.parse_locator("arg:memcpy:2")])
    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "caller_sites_truncated"]
    assert (leaf["callers_total"], leaf["callers_followed"], leaf["callers_dropped"]) == (
        20 + indirect_count, 16, 4 + indirect_count)
    assert leaf["reasons"] == {"caller_cap": 8, "indirect_reference": indirect_count * 2}
    assert len(leaf["unfollowed"]) == min(16, indirect_count * 2)
    assert result["diagnostics"]["frontier"]["dropped_callers"] == 4 + indirect_count
    assert result["stats"]["truncation_cause"] == ["caller_cap"]
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


def test_repeated_formal_does_not_duplicate_demand_evidence_and_run_state_resets():
    target, bv, engine = _two_formal_caller_query(
        [(_const(64), _const(32))], repeat_formal=True, indirect=[0x403000])
    result = engine.backward(target, [te.parse_locator("arg:memcpy:2")])
    [leaf] = result["leaves"]
    assert leaf["callers_dropped"] == 1
    assert leaf["reasons"] == {"indirect_reference": 1}
    assert len(leaf["unfollowed"]) == 1
    # Reusing the public engine for a clean query must not retain prior frontiers.
    target.caller_sites.clear()
    bv._data_refs.clear()
    clean = engine.backward(target, [te.parse_locator("arg:memcpy:2")])
    assert clean["leaves"] == []
    assert clean["diagnostics"]["frontier"]["dropped_callers"] == 0
    assert clean["diagnostics"]["safe_to_report_complete_slice"] is True


def test_nested_formal_frontier_survives_outer_ascent_without_local_misses():
    dst, n, m = (FVar(name, ident=130 + i) for i, name in enumerate(("dst", "n", "m")))
    n0, m0 = FSSA(n, 0), FSSA(m, 0)
    target = FFunc("rotate", 0x800, FSSAFunc([
        _call(0, 0x804, _MEMCPY, [_const(0), _const(0), _var(n0)]),
        _call(1, 0x810, 0x800, [_const(0), _var(m0), _const(64)]),
    ]), params=[dst, n, m])
    target.caller_sites = [FSite(target, 0x810)]
    engine = te.TaintEngine(FBV({**_NAMES, target.start: target.name}), te.load_models())
    result = engine.backward(target, [te.parse_locator("arg:memcpy:2")], max_depth=1)
    [leaf] = result["leaves"]
    assert leaf["function"]["address"] == "0x800"
    assert (leaf["callers_total"], leaf["callers_followed"], leaf["callers_dropped"]) == (1, 0, 1)
    assert leaf["unfollowed"] == [{"address": "0x810", "caller": "rotate",
                                  "param": 2, "reason": "max_depth"}]
    assert result["diagnostics"]["frontier"]["dropped_callers"] == 1
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    assert result["stats"]["truncated"] is False
    # With room to follow the remapped formal, the literal is the root.
    complete = engine.backward(target, [te.parse_locator("arg:memcpy:2")], max_depth=2)
    assert complete["leaves"] == []
    assert complete["diagnostics"]["safe_to_report_complete_slice"] is True
