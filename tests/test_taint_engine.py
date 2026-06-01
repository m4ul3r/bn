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


class FPVS:
    def __init__(self, type_name, mapping=None, values=None, value=None):
        self.type = FOp(type_name)
        if mapping is not None:
            self.mapping = mapping
        if values is not None:
            self.values = values
        if value is not None:
            self.value = value

    def __repr__(self):
        return f"<pvs {self.type.name}>"


class FExpr:
    def __init__(self, opname, text, reads=(), src=None, constant=None, possible_values=None, src_memory=None):
        self.operation = FOp(opname)
        self._text = text
        self.vars_read = list(reads)
        if src is not None:
            self.src = src
        if constant is not None:
            self.constant = constant
        if possible_values is not None:
            self.possible_values = possible_values
        if src_memory is not None:
            self.src_memory = src_memory

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
    def __init__(self, instrs, mem_defs=None):
        self.instructions = instrs
        self._mem_defs = mem_defs or {}

    def get_ssa_memory_definition(self, version):
        return self._mem_defs.get(int(version))

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
        self.instructions = ssa.instructions


class FSym:
    def __init__(self, type_name):
        self.type = type("T", (), {"name": type_name})()


class FSite:
    def __init__(self, function, address):
        self.function = function
        self.address = address


class FFunc:
    def __init__(self, name, start, ssa, params=(), symbol_type="FunctionSymbol",
                 is_thunk=False, caller_sites=()):
        self.name = name
        self.start = start
        self.mlil = FMLIL(ssa)
        self.parameter_vars = list(params)
        self.symbol = FSym(symbol_type)
        self.is_thunk = is_thunk
        self.caller_sites = list(caller_sites)


class FBV:
    def __init__(self, addr_names, funcs=None):
        self._names = addr_names
        self._funcs = funcs or {}

    def get_function_at(self, addr):
        if addr in self._funcs:
            return self._funcs[addr]
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


def _dispatch_table_program():
    """dispatch(buf) -> table[?](buf) indirect; targets run_sys(p)=system(p) and
    run_cpy(p)=strcpy(b,p). Returns (handler-equivalent dispatch func, bv)."""
    # run_sys @ 0x500: system(p)
    p_s = FVar("p"); ps0 = FSSA(p_s, 0)
    run_sys = FFunc("run_sys", 0x500, FSSAFunc([
        FInstr(0, 0x504, "MLIL_CALL_SSA", "0x900(p#0)", reads=[ps0], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x900", constant=0x900),
               params=[FExpr("MLIL_VAR_SSA", "p#0", reads=[ps0])]),
    ]), params=[p_s])
    # run_cpy @ 0x600: strcpy(b, p)
    p_c = FVar("p2"); pc0 = FSSA(p_c, 0)
    run_cpy = FFunc("run_cpy", 0x600, FSSAFunc([
        FInstr(0, 0x604, "MLIL_CALL_SSA", "0x910(b, p2#0)", reads=[pc0], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x910", constant=0x910),
               params=[FExpr("MLIL_VAR_SSA", "b", reads=[]),
                       FExpr("MLIL_VAR_SSA", "p2#0", reads=[pc0])]),
    ]), params=[p_c])
    # dispatch @ 0x700: indirect call resolved by VSA to {0x500, 0x600}
    buf = FVar("buf"); buf0 = FSSA(buf, 0)
    pvs = FPVS("LookupTableValue", mapping={0: 0x500, 1: 0x600})
    dispatch = FFunc("dispatch", 0x700, FSSAFunc([
        FInstr(0, 0x70c, "MLIL_CALL_SSA", "fp(buf#0)", reads=[buf0], writes=[],
               dest=FExpr("MLIL_VAR_SSA", "fp#1", reads=[], possible_values=pvs),
               params=[FExpr("MLIL_VAR_SSA", "buf#0", reads=[buf0])]),
    ]), params=[buf])
    bv = FBV({0x900: "system", 0x910: "strcpy"},
             funcs={0x500: run_sys, 0x600: run_cpy})
    return dispatch, bv


def test_forward_resolves_indirect_via_value_set(models):
    dispatch, bv = _dispatch_table_program()
    engine = te.TaintEngine(bv, models)
    result = engine.forward(dispatch, [te.parse_locator("param:0")])
    classes = {s["sink"]["class"] for s in result["reached_sinks"]}
    assert "command_injection" in classes  # via run_sys
    assert "overflow_unbounded" in classes  # via run_cpy
    assert any("resolved via value-set" in a for a in result["assumptions"])
    # no unresolved leaf since VSA pinned the targets
    assert not any(l["kind"] == "indirect_call_unresolved" for l in result["leaves"])


def test_forward_resolve_map_overrides_unresolved(models):
    # an indirect call with no VSA info, resolved by an agent-supplied map
    buf = FVar("buf"); buf0 = FSSA(buf, 0)
    p_s = FVar("p"); ps0 = FSSA(p_s, 0)
    run_sys = FFunc("run_sys", 0x500, FSSAFunc([
        FInstr(0, 0x504, "MLIL_CALL_SSA", "0x900(p#0)", reads=[ps0], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x900", constant=0x900),
               params=[FExpr("MLIL_VAR_SSA", "p#0", reads=[ps0])]),
    ]), params=[p_s])
    dispatch = FFunc("dispatch", 0x700, FSSAFunc([
        FInstr(0, 0x70c, "MLIL_CALL_SSA", "fp(buf#0)", reads=[buf0], writes=[],
               dest=FExpr("MLIL_VAR_SSA", "fp#1", reads=[]),  # no possible_values
               params=[FExpr("MLIL_VAR_SSA", "buf#0", reads=[buf0])]),
    ]), params=[buf])
    bv = FBV({0x900: "system"}, funcs={0x500: run_sys})

    # without a map -> unresolved leaf
    e1 = te.TaintEngine(bv, models)
    r1 = e1.forward(dispatch, [te.parse_locator("param:0")])
    assert any(l["kind"] == "indirect_call_unresolved" for l in r1["leaves"])

    # with an agent-supplied map -> resolved into run_sys -> system
    e2 = te.TaintEngine(bv, models, resolve_map={"0x70c": ["0x500"]})
    r2 = e2.forward(dispatch, [te.parse_locator("param:0")])
    assert any(s["sink"]["class"] == "command_injection" for s in r2["reached_sinks"])
    assert any("resolved via agent-map" in a for a in r2["assumptions"])


def test_forward_propagated_path_links_back_to_source(models):
    # handle(fd): read(&str); snprintf(cmd, sz, "echo %s", str); system(cmd)
    # the system finding's path must include the read source, not start at snprintf.
    s = FVar("str"); c = FVar("cmd"); rs = FVar("rs"); rc = FVar("rc"); fd = FVar("fd")
    rs1 = FSSA(rs, 1); rc1 = FSSA(rc, 1)
    ssa = FSSAFunc([
        FInstr(0, 0x10, "MLIL_SET_VAR_SSA", "rs#1 = &str", writes=[rs1],
               src=FExpr("MLIL_ADDRESS_OF", "&str", src=s)),
        FInstr(1, 0x14, "MLIL_CALL_SSA", "0x910(rdi, rs#1, 0xc8)", reads=[rs1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x910", constant=0x910),
               params=[FExpr("MLIL_VAR_SSA", "rdi", reads=[]),
                       FExpr("MLIL_VAR_SSA", "rs#1", reads=[rs1]),
                       FExpr("MLIL_CONST", "0xc8", constant=0xc8)]),
        FInstr(2, 0x18, "MLIL_SET_VAR_SSA", "rc#1 = &cmd", writes=[rc1],
               src=FExpr("MLIL_ADDRESS_OF", "&cmd", src=c)),
        FInstr(3, 0x1c, "MLIL_CALL_SSA", "0x920(rc#1, 0x100, \"echo %s\", rs#1)", reads=[rc1, rs1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x920", constant=0x920),
               params=[FExpr("MLIL_VAR_SSA", "rc#1", reads=[rc1]),
                       FExpr("MLIL_CONST", "0x100", constant=0x100),
                       FExpr("MLIL_CONST_PTR", "echo %s", constant=0x4050),
                       FExpr("MLIL_VAR_SSA", "rs#1", reads=[rs1])]),
        FInstr(4, 0x20, "MLIL_CALL_SSA", "0x930(rc#1)", reads=[rc1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x930", constant=0x930),
               params=[FExpr("MLIL_VAR_SSA", "rc#1", reads=[rc1])]),
    ])
    func = FFunc("handle", 0x10, ssa, params=[fd])
    bv = FBV({0x910: "read", 0x920: "snprintf", 0x930: "system"})
    engine = te.TaintEngine(bv, models)
    result = engine.forward(func, [te.parse_locator("arg:read:1")])
    assert len(result["reached_sinks"]) == 1
    path = result["reached_sinks"][0]["path"]
    reasons = " | ".join(s.get("reason", "") for s in path)
    assert "source: read" in reasons          # provenance reaches the source
    assert "snprintf" in reasons               # through the propagator
    # benign/socket noise must not appear for a clean modeled flow
    assert result["assumptions"] == []


def test_forward_outparam_propagates_to_caller(models):
    # fill(src, dst): strcpy(dst, src) -> dst is a tainted out-parameter
    src = FVar("src", ident=10); dst = FVar("dst", ident=11)
    src0 = FSSA(src, 0); dst0 = FSSA(dst, 0)
    fill = FFunc("fill", 0x800, FSSAFunc([
        FInstr(0, 0x804, "MLIL_CALL_SSA", "0x920(dst#0, src#0)", reads=[dst0, src0], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x920", constant=0x920),
               params=[FExpr("MLIL_VAR_SSA", "dst#0", reads=[dst0]),
                       FExpr("MLIL_VAR_SSA", "src#0", reads=[src0])]),
    ]), params=[src, dst])

    # handler(fd): read(&buf); fill(&buf, &out); system(out)
    buf = FVar("buf"); out = FVar("out"); fd = FVar("fd")
    rb = FVar("rb"); ro = FVar("ro")
    rb1 = FSSA(rb, 1); ro1 = FSSA(ro, 1); out0 = FSSA(out, 0)
    handler = FFunc("handler", 0x900, FSSAFunc([
        FInstr(0, 0x904, "MLIL_SET_VAR_SSA", "rb#1 = &buf", writes=[rb1],
               src=FExpr("MLIL_ADDRESS_OF", "&buf", src=buf)),
        FInstr(1, 0x908, "MLIL_CALL_SSA", "0x910(rdi#1, rb#1, 0x40)", reads=[rb1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x910", constant=0x910),
               params=[FExpr("MLIL_VAR_SSA", "rdi#1", reads=[]),
                       FExpr("MLIL_VAR_SSA", "rb#1", reads=[rb1]),
                       FExpr("MLIL_CONST", "0x40", constant=0x40)]),
        FInstr(2, 0x90c, "MLIL_SET_VAR_SSA", "ro#1 = &out", writes=[ro1],
               src=FExpr("MLIL_ADDRESS_OF", "&out", src=out)),
        FInstr(3, 0x910, "MLIL_CALL_SSA", "0x800(rb#1, ro#1)", reads=[rb1, ro1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x800", constant=0x800),
               params=[FExpr("MLIL_VAR_SSA", "rb#1", reads=[rb1]),
                       FExpr("MLIL_VAR_SSA", "ro#1", reads=[ro1])]),
        FInstr(4, 0x918, "MLIL_CALL_SSA", "0x930(out#0)", reads=[out0], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x930", constant=0x930),
               params=[FExpr("MLIL_VAR_SSA", "out#0", reads=[out0])]),
    ]), params=[fd])

    bv = FBV({0x910: "read", 0x920: "strcpy", 0x930: "system"}, funcs={0x800: fill})
    engine = te.TaintEngine(bv, models)
    result = engine.forward(handler, [te.parse_locator("arg:read:1")])
    classes = {s["sink"]["class"] for s in result["reached_sinks"]}
    # system is reachable ONLY if fill's out-parameter tainted `out`
    assert "command_injection" in classes


def test_forward_interprocedural_descends_into_callee(models):
    # copy_it(src, dst): rax#1 = src[0]; memcpy(dst, src, rax#1)  -> sink inside callee
    src = FVar("src"); dst = FVar("dst"); rax = FVar("rax")
    src1 = FSSA(src, 1); rax1 = FSSA(rax, 1)
    copy_ssa = FSSAFunc([
        FInstr(0, 0x2004, "MLIL_SET_VAR_SSA", "rax#1 = src#1[0]", reads=[src1], writes=[rax1]),
        FInstr(1, 0x2010, "MLIL_CALL_SSA", "0x1080(dst#1, src#1, rax#1)",
               reads=[rax1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x1080", constant=0x1080),
               params=[FExpr("MLIL_VAR_SSA", "dst#1", reads=[]),
                       FExpr("MLIL_VAR_SSA", "src#1", reads=[src1]),
                       FExpr("MLIL_VAR_SSA", "rax#1", reads=[rax1])]),
    ])
    copy_it = FFunc("copy_it", 0x2000, copy_ssa, params=[src, dst])

    # handler(fd): read(fd, &buf, 64); copy_it(&buf, &out)
    buf = FVar("buf", typ="char[0x40]"); out = FVar("out", typ="char[0x10]")
    fd = FVar("fd"); p0 = FVar("p0"); p1 = FVar("p1")
    rsi1 = FSSA(p0, 1); rsi2 = FSSA(p1, 1)
    handler_ssa = FSSAFunc([
        FInstr(0, 0x3000, "MLIL_SET_VAR_SSA", "rsi#1 = &buf", writes=[rsi1],
               src=FExpr("MLIL_ADDRESS_OF", "&buf", src=buf)),
        FInstr(1, 0x3008, "MLIL_CALL_SSA", "0x1050(rdi#1, rsi#1, 0x40)",
               reads=[rsi1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x1050", constant=0x1050),
               params=[FExpr("MLIL_VAR_SSA", "rdi#1", reads=[]),
                       FExpr("MLIL_VAR_SSA", "rsi#1", reads=[rsi1]),
                       FExpr("MLIL_CONST", "0x40", constant=0x40)]),
        FInstr(2, 0x3010, "MLIL_SET_VAR_SSA", "rsi_2#1 = &out", writes=[rsi2],
               src=FExpr("MLIL_ADDRESS_OF", "&out", src=out)),
        FInstr(3, 0x3018, "MLIL_CALL_SSA", "0x2000(&buf, &out)",
               reads=[rsi1, rsi2], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x2000", constant=0x2000),
               params=[FExpr("MLIL_VAR_SSA", "rsi#1", reads=[rsi1]),
                       FExpr("MLIL_VAR_SSA", "rsi_2#1", reads=[rsi2])]),
    ])
    handler = FFunc("handler", 0x3000, handler_ssa, params=[fd])

    bv = FBV({0x1050: "read", 0x1080: "memcpy"}, funcs={0x2000: copy_it})
    engine = te.TaintEngine(bv, models)
    result = engine.forward(handler, [te.parse_locator("arg:read:1")])

    # the sink lives in copy_it but must bubble up as a handler finding
    assert len(result["reached_sinks"]) == 1
    sink = result["reached_sinks"][0]["sink"]
    assert sink["callee"] == "memcpy" and sink["class"] == "overflow_len"
    # path must cross the call boundary into copy_it
    assert any("calls copy_it" in (s.get("reason") or "") for s in result["reached_sinks"][0]["path"])
    assert result["stats"]["functions_visited"] == 2


# --------------------------------------------------------------------------
# backward taint
# --------------------------------------------------------------------------

def test_forward_memory_ssa_store_to_load(models):
    # g(tval): [p] = tval; x = [p]; memcpy(d, s, x)
    # mem-SSA correlation must taint x (load reads the tainted store's bytes).
    tv = FVar("tval", ident=30); p = FVar("p", ident=31); x = FVar("x", ident=32)
    tv0 = FSSA(tv, 0); p1 = FSSA(p, 1); x1 = FSSA(x, 1)
    store = FInstr(0, 0x10, "MLIL_STORE_SSA", "[p#1] = tval#0", reads=[p1, tv0], writes=[],
                   dest=FExpr("MLIL_VAR_SSA", "p#1", reads=[p1]),
                   src=FExpr("MLIL_VAR_SSA", "tval#0", reads=[tv0]))
    store.src_memory = 0
    store.dest_memory = 1
    load = FInstr(1, 0x14, "MLIL_SET_VAR_SSA", "x#1 = [p#1]", reads=[p1], writes=[x1],
                  src=FExpr("MLIL_LOAD_SSA", "[p#1]", reads=[p1],
                            src=FExpr("MLIL_VAR_SSA", "p#1", reads=[p1]), src_memory=1))
    sink = FInstr(2, 0x18, "MLIL_CALL_SSA", "0x90(d, s, x#1)", reads=[x1], writes=[],
                  dest=FExpr("MLIL_CONST_PTR", "0x90", constant=0x90),
                  params=[FExpr("MLIL_VAR_SSA", "d", reads=[]),
                          FExpr("MLIL_VAR_SSA", "s", reads=[]),
                          FExpr("MLIL_VAR_SSA", "x#1", reads=[x1])])
    ssa = FSSAFunc([store, load, sink], mem_defs={1: store})
    func = FFunc("g", 0x10, ssa, params=[tv])   # param 0 = tval
    bv = FBV({0x90: "memcpy"})
    engine = te.TaintEngine(bv, models)
    result = engine.forward(func, [te.parse_locator("param:0")])
    assert any(s["sink"]["class"] == "overflow_len" for s in result["reached_sinks"])
    reasons = [st.get("reason", "") for s in result["reached_sinks"] for st in s["path"]]
    assert any("mem-SSA" in r for r in reasons)


def test_forward_memory_ssa_untainted_store_no_false_positive(models):
    # same shape but the store writes a constant -> load must NOT be tainted
    p = FVar("p", ident=31); x = FVar("x", ident=32)
    p1 = FSSA(p, 1); x1 = FSSA(x, 1)
    store = FInstr(0, 0x10, "MLIL_STORE_SSA", "[p#1] = 0x78", reads=[p1], writes=[],
                   dest=FExpr("MLIL_VAR_SSA", "p#1", reads=[p1]),
                   src=FExpr("MLIL_CONST", "0x78", constant=0x78))
    store.src_memory = 0
    store.dest_memory = 1
    load = FInstr(1, 0x14, "MLIL_SET_VAR_SSA", "x#1 = [p#1]", reads=[p1], writes=[x1],
                  src=FExpr("MLIL_LOAD_SSA", "[p#1]", reads=[p1],
                            src=FExpr("MLIL_VAR_SSA", "p#1", reads=[p1]), src_memory=1))
    sink = FInstr(2, 0x18, "MLIL_CALL_SSA", "0x90(d, s, x#1)", reads=[x1], writes=[],
                  dest=FExpr("MLIL_CONST_PTR", "0x90", constant=0x90),
                  params=[FExpr("MLIL_VAR_SSA", "d", reads=[]),
                          FExpr("MLIL_VAR_SSA", "s", reads=[]),
                          FExpr("MLIL_VAR_SSA", "x#1", reads=[x1])])
    ssa = FSSAFunc([store, load, sink], mem_defs={1: store})
    func = FFunc("g", 0x10, ssa, params=[FVar("unused", ident=99)])
    bv = FBV({0x90: "memcpy"})
    engine = te.TaintEngine(bv, models)
    # seed an unrelated param; the const store must not produce a tainted load
    result = engine.forward(func, [te.parse_locator("param:0")])
    assert result["reached_sinks"] == []


def test_backward_follows_into_caller(models):
    # use_len(dst, src, n): memcpy(dst, src, n) -- n is a parameter.
    # handler: n = recv(...); use_len(out, buf, n). Backward from memcpy length
    # must cross into handler and reach the recv source.
    dst = FVar("dst", ident=20); src = FVar("src", ident=21); n = FVar("n", ident=22)
    dst0 = FSSA(dst, 0); src0 = FSSA(src, 0); n0 = FSSA(n, 0)
    USE_LEN_CALL = 0x920
    use_len = FFunc("use_len", 0x800, FSSAFunc([
        FInstr(0, 0x804, "MLIL_CALL_SSA", "0x940(dst#0, src#0, n#0)", reads=[dst0, src0, n0], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x940", constant=0x940),
               params=[FExpr("MLIL_VAR_SSA", "dst#0", reads=[dst0]),
                       FExpr("MLIL_VAR_SSA", "src#0", reads=[src0]),
                       FExpr("MLIL_VAR_SSA", "n#0", reads=[n0])]),
    ]), params=[dst, src, n])

    fd = FVar("fd"); buf = FVar("buf"); out = FVar("out"); nh = FVar("nh")
    nh1 = FSSA(nh, 1); rb = FVar("rb"); rb1 = FSSA(rb, 1)
    handler = FFunc("handler", 0x900, FSSAFunc([
        FInstr(0, 0x904, "MLIL_SET_VAR_SSA", "rb#1 = &buf", writes=[rb1],
               src=FExpr("MLIL_ADDRESS_OF", "&buf", src=buf)),
        FInstr(1, 0x910, "MLIL_CALL_SSA", "nh#1 = 0x930(fd, rb#1, 0x40, 0)", reads=[rb1], writes=[nh1],
               dest=FExpr("MLIL_CONST_PTR", "0x930", constant=0x930),
               params=[FExpr("MLIL_VAR_SSA", "fd", reads=[]),
                       FExpr("MLIL_VAR_SSA", "rb#1", reads=[rb1]),
                       FExpr("MLIL_CONST", "0x40", constant=0x40),
                       FExpr("MLIL_CONST", "0", constant=0)]),
        FInstr(2, USE_LEN_CALL, "MLIL_CALL_SSA", "0x800(out, rb#1, nh#1)", reads=[rb1, nh1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x800", constant=0x800),
               params=[FExpr("MLIL_VAR_SSA", "out", reads=[]),
                       FExpr("MLIL_VAR_SSA", "rb#1", reads=[rb1]),
                       FExpr("MLIL_VAR_SSA", "nh#1", reads=[nh1])]),
    ]), params=[fd])
    use_len.caller_sites = [FSite(handler, USE_LEN_CALL)]

    bv = FBV({0x940: "memcpy", 0x930: "recv", 0x800: "use_len"})
    engine = te.TaintEngine(bv, models)
    result = engine.backward(use_len, [te.parse_locator("arg:memcpy:2")])

    assert result["slices"]
    # at least one slice crossed into the caller and reached the recv source
    crossed = [c for sl in result["slices"] for c in (sl.get("crossed_functions") or [])]
    assert "use_len" in crossed
    origins = [(sl["origin"]["kind"], sl["origin"].get("callee")) for sl in result["slices"]]
    assert ("source", "recv") in origins


def test_backward_slices_from_memcpy_length(process_func, models):
    bv = FBV({0x401070: "read", 0x401080: "memcpy"})
    engine = te.TaintEngine(bv, models)
    result = engine.backward(process_func, [te.parse_locator("arg:memcpy:2")])

    assert result["direction"] == "backward"
    assert len(result["slices"]) == 1
    sl = result["slices"][0]
    assert sl["sink"]["callee"] == "memcpy"
    assert sl["sink"]["seed"] == "rdx_1#1"
    # def chain reaches the aliased read buffer (a local, not a parameter)
    assert sl["origin"]["kind"] == "entry"
    slice_addrs = [s["address"] for s in sl["slice"]]
    assert "0x4011bc" in slice_addrs  # len#2 = len#1 + 4 is on the slice
