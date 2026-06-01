"""Tier-1 unit tests for the taint engine (no Binary Ninja required).

The engine is import-free of ``binaryninja`` by design, so we drive it with
small synthetic MLIL-SSA fakes that mirror the structures the bridge passes.
The fixtures reproduce the verified spike function ``process``:

    read(fd, &buf, 0x40)         ; buf becomes a source buffer
    len = (int)buf[0]            ; load from tainted buffer
    len = len + 4                ; arithmetic propagation
    memcpy(dst, buf, (size_t)len); SINK: tainted length -> overflow_len
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ENGINE_PATH = Path(__file__).resolve().parents[1] / "plugin" / "bn_agent_bridge" / "taint_engine.py"


def _load_engine():
    # Load as a top-level module so the wrapped `from .paths import ...` falls
    # back gracefully (no package context) — load_models still reads the
    # builtin JSON beside the file.
    spec = importlib.util.spec_from_file_location("bn_taint_engine_under_test", _ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("bn_taint_engine_under_test", module)
    spec.loader.exec_module(module)
    return module


te = _load_engine()


# --------------------------------------------------------------------------
# synthetic MLIL-SSA fakes
# --------------------------------------------------------------------------

class FVar:
    def __init__(self, name, ident=None, typ="int32_t"):
        self.name = name
        self.identifier = ident
        self.type = typ

    def __repr__(self):
        return f"<var {self.name}>"


class FSSA:
    def __init__(self, var: FVar, version: int):
        self.var = var
        self.version = version

    def __repr__(self):
        return f"<ssa {self.var.name}#{self.version}>"


class FOp:
    def __init__(self, name):
        self.name = name


class FExpr:
    def __init__(self, opname, text, reads=(), src=None, constant=None):
        self.operation = FOp(opname)
        self._text = text
        self.vars_read = list(reads)
        if src is not None:
            self.src = src
        if constant is not None:
            self.constant = constant

    def __str__(self):
        return self._text


class FInstr:
    def __init__(self, index, addr, opname, text, reads=(), writes=(), params=None, dest=None, src=None):
        self.instr_index = index
        self.address = addr
        self.operation = FOp(opname)
        self._text = text
        self.vars_read = list(reads)
        self.vars_written = list(writes)
        self.operands = []
        if params is not None:
            self.params = params
        if dest is not None:
            self.dest = dest
        if src is not None:
            self.src = src

    def __str__(self):
        return self._text


class FSSAFunc:
    def __init__(self, instrs):
        self.instructions = instrs

    def _match(self, ssa_var, collection_attr):
        for ins in self.instructions:
            for v in getattr(ins, collection_attr):
                if getattr(v, "var", None) is ssa_var.var and getattr(v, "version", None) == ssa_var.version:
                    yield ins

    def get_ssa_var_definition(self, ssa_var):
        for ins in self._match(ssa_var, "vars_written"):
            return ins
        return None

    def get_ssa_var_uses(self, ssa_var):
        return list(self._match(ssa_var, "vars_read"))


class FMLIL:
    def __init__(self, ssa):
        self.ssa_form = ssa


class FFunc:
    def __init__(self, name, start, ssa, params=()):
        self.name = name
        self.start = start
        self.mlil = FMLIL(ssa)
        self.parameter_vars = list(params)


class FBV:
    def __init__(self, addr_names):
        self._names = addr_names

    def get_function_at(self, addr):
        name = self._names.get(addr)
        if name is None:
            return None
        return type("F", (), {"name": name, "start": addr})()

    def get_symbol_at(self, addr):
        return None


@pytest.fixture
def process_func():
    buf = FVar("buf", typ="char[0x40]")
    rsi = FVar("rsi"); rax1 = FVar("rax_1"); rax2 = FVar("rax_2")
    rax3 = FVar("rax_3"); length = FVar("len"); rdx1 = FVar("rdx_1")
    rdi = FVar("rdi"); dst = FVar("dst")

    buf1 = FSSA(buf, 1)
    rsi1 = FSSA(rsi, 1)
    rdi1 = FSSA(rdi, 1)
    rax1_2 = FSSA(rax1, 2)
    rax2_3 = FSSA(rax2, 3)
    rax3_4 = FSSA(rax3, 4)
    len1 = FSSA(length, 1)
    len2 = FSSA(length, 2)
    rdx1_1 = FSSA(rdx1, 1)

    addr_of_buf = FExpr("MLIL_ADDRESS_OF", "&buf", src=buf)
    rsi_param = FExpr("MLIL_VAR_SSA", "rsi#1", reads=[rsi1])
    rdi_param = FExpr("MLIL_VAR_SSA", "rdi#1", reads=[rdi1])
    rdx_param = FExpr("MLIL_VAR_SSA", "rdx_1#1", reads=[rdx1_1])
    dst_param = FExpr("MLIL_VAR_SSA", "&dst", reads=[])
    const40 = FExpr("MLIL_CONST", "0x40", constant=0x40)

    instrs = [
        FInstr(0, 0x4011a4, "MLIL_SET_VAR_SSA", "rsi#1 = &buf", writes=[rsi1], src=addr_of_buf),
        FInstr(1, 0x4011a7, "MLIL_SET_VAR_SSA", "rdi#1 = rax#1", writes=[rdi1]),
        FInstr(2, 0x4011a9, "MLIL_CALL_SSA", "rax_1#2 = 0x401070(rdi#1, rsi#1, 0x40)",
               reads=[rdi1, rsi1], writes=[rax1_2],
               dest=FExpr("MLIL_CONST_PTR", "0x401070", constant=0x401070),
               params=[rdi_param, rsi_param, const40]),
        FInstr(3, 0x4011b2, "MLIL_SET_VAR_SSA", "rax_2#3 = buf[0]", reads=[buf1], writes=[rax2_3]),
        FInstr(4, 0x4011b6, "MLIL_SET_VAR_SSA", "rax_3#4 = sx.d(rax_2#3)", reads=[rax2_3], writes=[rax3_4]),
        FInstr(5, 0x4011b9, "MLIL_SET_VAR_SSA", "len#1 = rax_3#4", reads=[rax3_4], writes=[len1]),
        FInstr(6, 0x4011bc, "MLIL_SET_VAR_SSA", "len#2 = len#1 + 4", reads=[len1], writes=[len2]),
        FInstr(7, 0x4011ca, "MLIL_SET_VAR_SSA", "rdx_1#1 = sx.q(len#2)", reads=[len2], writes=[rdx1_1]),
        FInstr(8, 0x4011db, "MLIL_CALL_SSA", "0x401080(&dst, &buf, rdx_1#1)",
               reads=[rdx1_1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x401080", constant=0x401080),
               params=[dst_param, FExpr("MLIL_VAR_SSA", "&buf", reads=[]), rdx_param]),
    ]
    return FFunc("process", 0x401189, FSSAFunc(instrs), params=[FVar("fd")])


@pytest.fixture
def models():
    return te.load_models()


# --------------------------------------------------------------------------
# model DB + locator parsing
# --------------------------------------------------------------------------

def test_builtin_models_load():
    models = te.load_models()
    assert "memcpy" in models and models["memcpy"]["sink"]["class"] == "overflow_len"
    assert "system" in models and "recv" in models


def test_lookup_model_strips_decorations():
    models = te.load_models()
    name, model = te.lookup_model(models, "memcpy@plt")
    assert name == "memcpy"
    name, model = te.lookup_model(models, "_memcpy")
    assert name == "memcpy"
    name, model = te.lookup_model(models, "totally_unknown_fn")
    assert name is None and model is None


@pytest.mark.parametrize("spec,expected", [
    ("param:0", {"kind": "param", "index": 0}),
    ("var:buf", {"kind": "var", "selector": "buf"}),
    ("ret:recv", {"kind": "ret", "callee": "recv"}),
    ("arg:memcpy:2", {"kind": "arg", "callee": "memcpy", "index": 2}),
])
def test_parse_locator(spec, expected):
    assert te.parse_locator(spec) == expected


def test_parse_locator_rejects_garbage():
    with pytest.raises(te.TaintError):
        te.parse_locator("bogus:1")


# --------------------------------------------------------------------------
# forward taint
# --------------------------------------------------------------------------

def test_forward_reaches_memcpy(process_func, models):
    bv = FBV({0x401070: "read", 0x401080: "memcpy"})
    engine = te.TaintEngine(bv, models)
    result = engine.forward(process_func, [te.parse_locator("arg:read:1")])

    assert result["direction"] == "forward"
    assert len(result["reached_sinks"]) == 1
    sink = result["reached_sinks"][0]["sink"]
    assert sink["callee"] == "memcpy"
    assert sink["class"] == "overflow_len"
    assert sink["tainted_arg_index"] == 2
    # path threads the SSA chain from the read source to the memcpy sink
    path_ops = [s["op"] for s in result["reached_sinks"][0]["path"]]
    assert path_ops[0] == "MLIL_CALL_SSA"  # source: read
    assert path_ops[-1] == "MLIL_CALL_SSA"  # sink: memcpy
    assert "soundness" in result


def test_forward_no_flow_no_false_positive(models):
    # a function with no source->sink path must report zero sinks
    a = FVar("a"); r = FVar("r")
    a0 = FSSA(a, 0); r1 = FSSA(r, 1)
    ssa = FSSAFunc([
        FInstr(0, 0x10, "MLIL_SET_VAR_SSA", "r#1 = a#0 + 1", reads=[a0], writes=[r1]),
        FInstr(1, 0x14, "MLIL_RET", "return r#1", reads=[r1]),
    ])
    func = FFunc("add_one", 0x10, ssa, params=[a])
    engine = te.TaintEngine(FBV({}), models)
    result = engine.forward(func, [te.parse_locator("param:0")])
    assert result["reached_sinks"] == []


def test_forward_indirect_call_is_a_leaf_not_dropped(models):
    fp = FVar("fp"); arg = FVar("arg"); rr = FVar("rr")
    arg0 = FSSA(arg, 0); fp1 = FSSA(fp, 1); rr1 = FSSA(rr, 1)
    ssa = FSSAFunc([
        FInstr(0, 0x20, "MLIL_SET_VAR_SSA", "rr#1 = arg#0", reads=[arg0], writes=[rr1]),
        FInstr(1, 0x24, "MLIL_CALL_SSA", "fp#1(rr#1)",
               reads=[rr1, fp1], writes=[],
               dest=FExpr("MLIL_VAR_SSA", "fp#1", reads=[fp1]),
               params=[FExpr("MLIL_VAR_SSA", "rr#1", reads=[rr1])]),
    ])
    func = FFunc("dispatch", 0x20, ssa, params=[arg])
    engine = te.TaintEngine(FBV({}), models)
    result = engine.forward(func, [te.parse_locator("param:0")])
    assert any(l["kind"] == "indirect_call_unresolved" for l in result["leaves"])
    assert any("indirect call" in a for a in result["assumptions"])


def test_forward_unresolved_source_raises(models):
    func = FFunc("x", 0x0, FSSAFunc([]), params=[])
    engine = te.TaintEngine(FBV({}), models)
    with pytest.raises(te.TaintError):
        engine.forward(func, [te.parse_locator("arg:read:1")])


# --------------------------------------------------------------------------
# backward taint
# --------------------------------------------------------------------------

def test_backward_slices_from_memcpy_length(process_func, models):
    bv = FBV({0x401070: "read", 0x401080: "memcpy"})
    engine = te.TaintEngine(bv, models)
    result = engine.backward(process_func, [te.parse_locator("arg:memcpy:2")])

    assert result["direction"] == "backward"
    assert len(result["slices"]) == 1
    sl = result["slices"][0]
    assert sl["sink"]["callee"] == "memcpy"
    assert sl["sink"]["seed"] == "rdx_1#1"
    # def chain reaches the aliased read buffer with no further SSA definition
    assert sl["origin"]["kind"] == "parameter_or_entry"
    slice_addrs = [s["address"] for s in sl["slice"]]
    assert "0x4011bc" in slice_addrs  # len#2 = len#1 + 4 is on the slice
