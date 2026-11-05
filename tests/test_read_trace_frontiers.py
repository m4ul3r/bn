"""`trace`: where a backward slice stops, and how that is reported.

Every terminal kind the use-def walk can stop at is a step with a reason and a
frontier roll-up category -- a typed field load with its real member offset, a
load from a global, a literal, an out-parameter fill with its call site -- and
the interprocedural text keeps the call each callee walk was entered from."""
from __future__ import annotations

import types

from _bridge_fakes import (
    _FakeBV,
    _FakeFunction,
    _FakeMLILFunction,
    _FakeMLILInsn,
    _FakeSSAVariable,
    _load_bridge,
)

from bn.formatters import _render_trace_text


def _var(v, addr):
    return _FakeMLILInsn(addr, operation="MLIL_VAR_SSA", vars_read=[v])


def _const(c, addr, op="MLIL_CONST"):
    return _FakeMLILInsn(addr, operation=op, constant=c)


def _sink(addr, *vars_):
    """`sink(v...)` at *addr*, every argument a plain SSA variable."""
    return _FakeMLILInsn(addr, operation="MLIL_CALL_SSA",
                         params=[_var(v, addr) for v in vars_], vars_read=list(vars_))


def _slice(monkeypatch, instructions, definitions, sink_addr, *, arg=0, symbols=(), **kw):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x2000, "parse_packet")
    fn.medium_level_il = _FakeMLILFunction(instructions, definitions=definitions)
    bv = _FakeBV(functions=[fn], symbols=list(symbols))
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    return instance._backward_slice("active", "parse_packet", hex(sink_addr), arg_index=arg, **kw)


def _struct_load(base, offset, addr, size=4):
    load = _FakeMLILInsn(addr, operation="MLIL_LOAD_STRUCT_SSA", src=_var(base, addr),
                         size=size, vars_read=[base])
    load.offset = offset
    return load


def test_typed_field_load_reports_the_member_offset(monkeypatch):
    # `data = hdr->data` (member at 0x38) and `len = zx.q(hdr->len)` (member at
    # 4): the offset lives on the LOAD_STRUCT instruction, not in an address
    # expression -- it used to be reported as 0x0, and the widened one not as
    # a load at all.
    hdr, data, wlen = (_FakeSSAVariable(n) for n in ("hdr#1", "data#1", "len#2"))
    data_def = _FakeMLILInsn(0x2008, operation="MLIL_SET_VAR_SSA",
                             src=_struct_load(hdr, 0x38, 0x2008, size=8), vars_read=[hdr])
    zx = _FakeMLILInsn(0x200c, operation="MLIL_ZX", src=_struct_load(hdr, 4, 0x200c))
    len_def = _FakeMLILInsn(0x200c, operation="MLIL_SET_VAR_SSA", src=zx, vars_read=[hdr])
    result = _slice(monkeypatch, [data_def, len_def, _sink(0x2010, data, wlen)],
                    {data: data_def, wlen: len_def}, 0x2010)
    data_step = next(s for s in result["trace"] if s["ssa_label"] == "data#1")
    assert (data_step["reason"], data_step["base"], data_step["offset"], data_step["width"]) == \
        ("field_load", "hdr#1", "0x38", 8)
    result = _slice(monkeypatch, [data_def, len_def, _sink(0x2010, data, wlen)],
                    {data: data_def, wlen: len_def}, 0x2010, arg=1)
    len_step = result["trace"][0]
    assert (len_step["reason"], len_step["base"], len_step["offset"], len_step["width"]) == \
        ("field_load", "hdr#1", "0x4", 4)
    # The struct pointer is still walked past the load.
    assert [s["ssa_label"] for s in result["trace"]] == ["len#2", "hdr#1"]


def test_global_load_is_a_named_frontier_with_an_xrefs_follow_up(monkeypatch):
    limit = _FakeSSAVariable("limit#2")
    load = _FakeMLILInsn(0x2008, operation="MLIL_LOAD_SSA", size=4,
                         src=_const(0x404020, 0x2008, op="MLIL_CONST_PTR"))
    limit_def = _FakeMLILInsn(0x2008, operation="MLIL_SET_VAR_SSA", src=load)
    sym = types.SimpleNamespace(address=0x404020, name="g_max_len", full_name="g_max_len")
    result = _slice(monkeypatch, [limit_def, _sink(0x2010, limit)], {limit: limit_def},
                    0x2010, symbols=[sym])
    [step] = result["trace"]
    assert step["terminates"] is True
    assert (step["reason"], step["global_address"], step["global_name"], step["address"]) == \
        ("global_load", "0x404020", "g_max_len", "0x2008")
    assert [(g["reason"], g["count"]) for g in result["frontiers"]] == [("global_load", 1)]
    assert result["frontiers"][0]["examples"][0]["global_address"] == "0x404020"
    assert any("bn xrefs 0x404020" in a for a in result["assumptions"]), result["assumptions"]
    text = _render_trace_text(result)
    assert "g_max_len @ 0x404020" in text


def test_every_terminal_kind_is_a_frontier_category(monkeypatch):
    # `n = phi(0, hdr->count, arg)`: a literal, a memory load and a parameter.
    # The roll-up used to count only the parameter: a SET_VAR load was not a
    # terminal and a constant was an ordinary `definition`.
    n, zero, cnt, arg = (_FakeSSAVariable(v) for v in ("n#3", "n#1", "n#2", "arg#0"))
    hdr = _FakeSSAVariable("hdr#1")
    phi = _FakeMLILInsn(0x2018, operation="MLIL_VAR_PHI", vars_read=[zero, cnt, arg])
    zero_def = _FakeMLILInsn(0x2008, operation="MLIL_SET_VAR_SSA", src=_const(0, 0x2008))
    cnt_def = _FakeMLILInsn(0x200c, operation="MLIL_SET_VAR_SSA",
                            src=_struct_load(hdr, 0x10, 0x200c), vars_read=[hdr])
    result = _slice(monkeypatch, [zero_def, cnt_def, phi, _sink(0x2020, n)],
                    {n: phi, zero: zero_def, cnt: cnt_def}, 0x2020)
    by_label = {s["ssa_label"]: s for s in result["trace"]}
    assert (by_label["n#1"]["reason"], by_label["n#1"]["terminates"], by_label["n#1"]["value"]) == \
        ("constant", True, "0x0")
    assert (by_label["n#2"]["reason"], by_label["n#2"]["terminates"]) == ("field_load", True)
    assert {g["reason"]: g["count"] for g in result["frontiers"]} == {
        "constant": 1, "field_load": 1, "undefined_or_global": 2}


def _out_param_program(fill_sites):
    """`fill(&rec)` at each of *fill_sites*, then `len = rec.len; sink(len)`."""
    rec = _FakeSSAVariable("rec")
    length = _FakeSSAVariable("len#3")
    calls = [_FakeMLILInsn(a, operation="MLIL_CALL_SSA",
                           params=[_FakeMLILInsn(a, operation="MLIL_ADDRESS_OF", src=rec)])
             for a in fill_sites]
    addr_expr = _FakeMLILInsn(0x2040, operation="MLIL_ADD",
                              left=_FakeMLILInsn(0x2040, operation="MLIL_ADDRESS_OF", src=rec),
                              right=_const(8, 0x2040))
    len_def = _FakeMLILInsn(0x2040, operation="MLIL_SET_VAR_SSA",
                            src=_FakeMLILInsn(0x2040, operation="MLIL_LOAD_SSA", src=addr_expr,
                                              size=4))
    return calls + [len_def, _sink(0x2050, length)], {length: len_def}


def test_out_param_step_names_the_fill_call_site(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    monkeypatch.setattr(bridge.read_taint_slice, "_callee_display_name",
                        lambda ctx, bv, ins: "read_record")
    instructions, definitions = _out_param_program([0x2010, 0x2030])
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x2000, "parse_packet")
    fn.medium_level_il = _FakeMLILFunction(instructions, definitions=definitions)
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._backward_slice("active", "parse_packet", "0x2050", arg_index=0)
    [step] = [s for s in result["trace"] if s["reason"] == "out_param_not_followed"]
    # The nearest fill preceding the read, not the first one in the function.
    assert (step["out_param_callee"], step["out_param_call_site"]) == ("read_record", "0x2030")
    [group] = [g for g in result["frontiers"] if g["reason"] == "out_param_not_followed"]
    assert group["examples"][0]["out_param_call_site"] == "0x2030"


def _phi_of_addresses(locals_):
    """`p#4 = phi(p#1, p#2, ...)`, each arm `p#k = &<local>[k]`."""
    p = _FakeSSAVariable("p#4")
    arms, defs, instrs = [], {}, []
    for k, local in enumerate(locals_, start=1):
        arm = _FakeSSAVariable(f"p#{k}")
        taken = _FakeMLILInsn(0x2000 + 4 * k, operation="MLIL_ADDRESS_OF_FIELD",
                              src=types.SimpleNamespace(name=local))
        defs[arm] = _FakeMLILInsn(0x2000 + 4 * k, operation="MLIL_SET_VAR_SSA", src=taken)
        instrs.append(defs[arm])
        arms.append(arm)
    defs[p] = _FakeMLILInsn(0x2020, operation="MLIL_VAR_PHI", vars_read=arms)
    return p, defs, instrs + [defs[p]]


def _address_hints(result):
    return [h for h in result["hints"] if "name_buf" in h or "path_buf" in h]


def test_phi_of_one_locals_addresses_keeps_the_address_of_local_hint(monkeypatch):
    p, defs, instrs = _phi_of_addresses(["name_buf", "name_buf", "name_buf"])
    result = _slice(monkeypatch, instrs + [_sink(0x2030, p)], defs, 0x2030)
    assert len(_address_hints(result)) == 1, result["hints"]
    assert {g["reason"]: g["count"] for g in result["frontiers"]} == {"address_of_local": 3}
    # One local, one caveat subject: the three steps are not three locals.
    [caveat] = [a for a in result["assumptions"] if "name_buf" in a]
    assert "more" not in caveat


def test_phi_of_two_locals_addresses_has_no_address_of_local_hint(monkeypatch):
    p, defs, instrs = _phi_of_addresses(["name_buf", "path_buf"])
    result = _slice(monkeypatch, instrs + [_sink(0x2030, p)], defs, 0x2030)
    assert _address_hints(result) == [], result["hints"]


def _cross(addr, var, callee):
    return {"ssa_label": var, "address": addr, "il_text": f"{var} = {callee}(x)",
            "reason": "cross_function", "terminates": False, "cross_function": True,
            "callee": callee}


def _callee_body(callee, first_addr):
    return [
        {"ssa_label": "v#1", "address": hex(first_addr), "il_text": "v#1 = value#0 << 8",
         "reason": "definition", "terminates": False, "function_context": callee},
        {"ssa_label": "value#0", "reason": "function_parameter", "terminates": True,
         "param_index": 1, "param_register": "rsi", "function_context": callee},
    ]


def test_interprocedural_text_keeps_call_sites_and_prints_a_repeated_walk_once():
    trace = (
        [{"ssa_label": "w#3", "address": "0x2010", "il_text": "w#3 = phi(w#1, w#2)",
          "reason": "phi_source", "terminates": False}]
        + [_cross("0x2020", "w#1", "read_u32")] + _callee_body("read_u32", 0x3000)
        + [_cross("0x2030", "w#2", "read_u32")] + _callee_body("read_u32", 0x3000)
        + [{"ssa_label": "base#0", "reason": "function_parameter", "terminates": True,
            "param_index": 0, "param_register": "rdi"}]
    )
    text = _render_trace_text({
        "function": "parse_packet", "function_address": "0x2000", "target_address": "0x2040",
        "arg_index": 0, "trace": trace, "frontiers": [], "assumptions": [], "hints": []})
    lines = text.splitlines()
    # Both call boundaries are printed, each where the walk crossed.
    assert any(line.strip().startswith("0x2020") for line in lines), text
    assert any(line.strip().startswith("0x2030") for line in lines), text
    # The callee's body is printed once; the second entry points back at it.
    assert sum(line.strip().startswith("0x3000") for line in lines) == 1, text
    second = next(i for i, line in enumerate(lines) if line.strip().startswith("0x2030"))
    assert "0x2020" in lines[second + 2], text
    # The callee's parameter is named as the callee's, and the caller's own
    # parameter after the walk is marked as back in the caller.
    assert any("param 1 of read_u32" in line for line in lines), text
    back = next(i for i, line in enumerate(lines) if "base#0" in line)
    assert "parse_packet" in lines[back - 1], text
    assert "of read_u32" not in lines[back], text
