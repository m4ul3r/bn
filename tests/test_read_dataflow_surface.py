"""`dataflow callgraph/values/defuse` and `function structured-il`: what each
reports, and how the text renders it."""
from __future__ import annotations

import types

import pytest
from _bridge_fakes import _load_bridge, _pvs

from bn.formatters import (
    _render_callgraph_text,
    _render_defuse_text,
    _render_structured_il_text,
    _render_values_text,
)


class _Expr:
    def __init__(self, text, *, op="MLIL_VAR_SSA", address=0, index=-1, **attrs):
        self.text, self.address, self.instr_index = text, address, index
        self.operation = types.SimpleNamespace(name=op)
        for k, v in attrs.items():
            setattr(self, k, v)

    def __str__(self):
        return self.text


def _fn(name, start, **kw):
    return types.SimpleNamespace(name=name, start=start, symbol=None, **kw)


def _instance(monkeypatch, func, il_by_start, bv=None):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda sel: bv or object())
    monkeypatch.setattr(instance.ctx, "_find_function", lambda bv_, ident, **kw: func)
    monkeypatch.setattr(bridge.il_format, "_il_function_for",
                        lambda fn, view, ssa: types.SimpleNamespace(
                            instructions=il_by_start[int(fn.start)]))
    return bridge, instance


class _CallgraphBV:
    def __init__(self, functions, code_refs=(), symbols=()):
        self._functions = {int(f.start): f for f in functions}
        self._code_refs = list(code_refs)
        self._symbols = {int(s.address): s for s in symbols}

    def get_function_at(self, addr):
        return self._functions.get(int(addr))

    def get_symbol_at(self, addr):
        return self._symbols.get(int(addr))

    def get_code_refs(self, addr):
        return [r for r in self._code_refs if r.target == int(addr)]


def test_callers_include_an_indirect_call_resolved_through_its_value_set(monkeypatch):
    # `dispatch` loads `handle_reserve` into a register and calls through it.
    # BN records no caller site for that, but the same value-set the callees
    # direction resolves names the target -- callers must say so too.
    target = _fn("handle_reserve", 0x5000, caller_sites=[], callers=[])
    dispatch = _fn("dispatch", 0x4000)
    call = _Expr("result#3(pkt#0)", op="MLIL_CALL_SSA", address=0x4040,
                 dest=_Expr("result#2", possible_values=_pvs("InSetOfValues",
                                                              values={0x5000, 0x5100})))
    other = _Expr("helper(pkt#0)", op="MLIL_CALL_SSA", address=0x4048,
                  dest=_Expr("0x5200", op="MLIL_CONST_PTR", constant=0x5200))
    bv = _CallgraphBV([target, dispatch, _fn("handle_query", 0x5100), _fn("helper", 0x5200)],
                      code_refs=[types.SimpleNamespace(target=0x5000, address=0x4030,
                                                       function=dispatch)])
    _, instance = _instance(monkeypatch, target, {0x4000: [call, other]}, bv)
    result = instance._resolved_calls("active", "handle_reserve", direction="callers")
    assert [(c["call_addr"], c["caller"]["name"], c["kind"]) for c in result["callers"]] == \
        [("0x4040", "dispatch", "indirect")]
    text = _render_callgraph_text(result)
    assert "dispatch @ 0x4000" in text and "0x4040" in text
    # Value-set resolution is what `--no-resolve-indirect` turns off.
    unresolved = instance._resolved_calls("active", "handle_reserve", direction="callers",
                                          resolve_indirect=False)
    assert unresolved["callers"] == []


def test_constant_call_to_a_bodiless_builtin_is_one_direct_row(monkeypatch):
    # `rep stosq` lifted as two MLIL calls to `__builtin_memset`, a symbol with
    # no function at its address: a direct call (the address is in the
    # instruction), reported once per site.
    func = _fn("fill_note", 0x4000, caller_sites=[], callers=[])
    calls = [_Expr("__builtin_memset(buf, 0, 0x40)", op="MLIL_CALL_SSA", address=0x4010,
                   index=i, dest=_Expr("0x6000", op="MLIL_CONST_PTR", constant=0x6000))
             for i in (7, 9)]
    sym = types.SimpleNamespace(address=0x6000, name="__builtin_memset",
                                short_name="__builtin_memset")
    bv = _CallgraphBV([func], symbols=[sym])
    _, instance = _instance(monkeypatch, func, {0x4000: calls}, bv)
    result = instance._resolved_calls("active", "fill_note", direction="callees")
    [row] = result["callees"]
    assert (row["kind"], row["call_addr"], row["target"]["address"], row["target"]["name"],
            row["target"]["has_body"]) == ("direct", "0x4010", "0x6000", "__builtin_memset", False)
    assert "indirect" not in _render_callgraph_text(result, per_site=True)


def _at_address(*specs):
    return [_Expr(text, op=op, address=0x4011e4, index=i,
                  possible_values=_pvs("UndeterminedValue"),
                  **({"src": _Expr(text.split(" = ", 1)[1], possible_values=pv)} if pv else {}))
            for i, op, text, pv in specs]


_SHARED = (
    (4, "MLIL_IF", "if (flag#1 == 0) then 5 else 7", None),
    (5, "MLIL_SET_VAR_SSA", "limit#2 = [0x404020].d",
     _pvs("UnsignedRangeValue", ranges=[types.SimpleNamespace(start=0, end=0xff, step=1)])),
    (6, "MLIL_GOTO", "goto 9", None),
    (7, "MLIL_SET_VAR_SSA", "limit#3 = limit#1", None),
)


def test_values_at_an_address_two_instructions_compute_lists_the_candidates(monkeypatch):
    func = _fn("handle", 0x4011ca)
    bridge, instance = _instance(monkeypatch, func, {0x4011ca: _at_address(*_SHARED)})
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._possible_values(None, "handle", "0x4011e4")
    assert exc.value.status == "ambiguous_instruction"
    assert [c["il_index"] for c in exc.value.observed["candidates"]] == [4, 5, 6, 7]


def test_values_at_address_and_il_index_answers_that_instruction(monkeypatch):
    func = _fn("handle", 0x4011ca)
    bridge, instance = _instance(monkeypatch, func, {0x4011ca: _at_address(*_SHARED)})
    result = instance._possible_values(None, "handle", "0x4011e4:5")
    assert (result["il_index"], result["expression"]) == (5, "limit#2 = [0x404020].d")
    assert result["possible_values"]["ranges"] == [{"start": 0, "end": 0xff, "step": 1}]
    assert [o["il_index"] for o in result["other_instructions"]] == [4, 6, 7]
    text = _render_values_text(result)
    assert "[0x0..0xff]" in text
    # Neither the serialized dict's repr nor BN's own repr of the set.
    assert "{'" not in text and result["possible_values"]["raw"] not in text, text
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._possible_values(None, "handle", "0x4011e4:9")
    assert exc.value.status == "no_instruction"
    with pytest.raises(ValueError):
        instance._possible_values(None, "handle", "0x4011e4:x")


def test_values_with_one_valued_instruction_answers_it_and_lists_the_rest(monkeypatch):
    func = _fn("handle", 0x4011ca)
    specs = ((10, "MLIL_VAR_PHI", "a#3 = ϕ(a#1, a#2)", None),
             (11, "MLIL_SET_VAR_SSA", "b#2 = zx.q(b#1)",
              _pvs("InSetOfValues", values={1, 2})))
    _, instance = _instance(monkeypatch, func, {0x4011ca: _at_address(*specs)})
    result = instance._possible_values(None, "handle", "0x4011e4")
    assert result["il_index"] == 11
    assert [o["il_index"] for o in result["other_instructions"]] == [10]
    assert "values={0x1, 0x2}" in _render_values_text(result)


_MANGLED = {"name": "_ZN4demo6Widget6handleEPKc", "display_name": "demo::Widget::handle",
            "address": "0x4011ca"}


@pytest.mark.parametrize("render, payload", [
    (_render_defuse_text, {"function": _MANGLED, "variable": {"ssa": "n#1"}, "uses": []}),
    (_render_values_text, {"function": _MANGLED, "at": "0x4011e4", "expression": "n#1 = 0",
                           "possible_values": {"type": "ConstantValue", "value": 0}}),
    (_render_structured_il_text, {"function": _MANGLED, "view": "mlil", "ssa": True,
                                  "instructions": []}),
])
def test_dataflow_headers_are_demangled_unless_mangled(render, payload):
    assert render(payload).splitlines()[0].startswith("demo::Widget::handle @ 0x4011ca")
    assert render(payload, mangled=True).splitlines()[0].startswith(
        "_ZN4demo6Widget6handleEPKc @ 0x4011ca")
