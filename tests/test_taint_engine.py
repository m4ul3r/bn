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
    # builtin JSON beside the file. Don't write bytecode into the plugin source
    # tree (the wheel build ships it as data files; see _load_bridge / #83).
    sys.dont_write_bytecode = True
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
    def __init__(self, index, addr, opname, text, reads=(), writes=(), params=None,
                 dest=None, src=None, prev=None, offset=None):
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
        if prev is not None:
            self.prev = prev
        if offset is not None:
            self.offset = offset

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


def test_secure_crt_annex_k_models_present_and_shaped():
    models = te.load_models()
    # Copy family: destsz after the dest shifts src->arg2, count->arg3.
    assert models["memcpy_s"]["propagates"] == [{"from": "*arg:2", "to": "*arg:0"}]
    assert models["memcpy_s"]["sink"]["tainted_args"] == [3]
    assert models["memcpy_s"]["sink"]["class"] == "fortified_overflow"
    assert models["memmove_s"]["sink"]["tainted_args"] == [3]
    # memset_s has no data propagation (arg2 is the fill byte), count at arg3.
    assert "propagates" not in models["memset_s"]
    assert models["memset_s"]["sink"]["tainted_args"] == [3]
    # strcpy_s/strcat_s: src at arg2 is the reportable arg (no count arg).
    assert models["strcpy_s"]["sink"]["tainted_args"] == [2]
    assert models["strcat_s"]["sink"]["tainted_args"] == [2]
    # strncpy_s/strncat_s: count at arg3.
    assert models["strncpy_s"]["sink"]["tainted_args"] == [3]
    assert models["strncat_s"]["sink"]["tainted_args"] == [3]
    # printf family: format at arg2, varargs first_index 3 (bufsz shifts right).
    assert models["sprintf_s"]["varargs"]["first_index"] == 3
    assert models["sprintf_s"]["sink"]["tainted_args"] == [2]
    assert models["sprintf_s"]["sink"]["class"] == "fortified_format"
    assert models["snprintf_s"]["varargs"]["first_index"] == 3
    # decorated / leading-underscore forms still resolve (e.g. MS _snprintf_s name)
    assert te.lookup_model(models, "_memcpy_s")[0] == "memcpy_s"
    assert te.lookup_model(models, "memcpy_s@plt")[0] == "memcpy_s"


def test_lookup_model_strips_decorations():
    models = te.load_models()
    name, model = te.lookup_model(models, "memcpy@plt")
    assert name == "memcpy"
    name, model = te.lookup_model(models, "_memcpy")
    assert name == "memcpy"
    name, model = te.lookup_model(models, "totally_unknown_fn")
    assert name is None and model is None


def test_chk_models_present_and_shaped():
    models = te.load_models()
    # FORTIFY variants resolve via the underscore-stripped key, NOT by collapsing
    # to the base function (which has different argument positions).
    name, model = te.lookup_model(models, "__memcpy_chk")
    assert name == "memcpy_chk"
    assert model["sink"]["tainted_args"] == [2]
    assert model["sink"]["class"] == "fortified_overflow"
    assert model["propagates"] == [{"from": "*arg:1", "to": "*arg:0"}]
    # decorated/PLT forms also resolve
    assert te.lookup_model(models, "__strcpy_chk@plt")[0] == "strcpy_chk"
    assert models["strcpy_chk"]["sink"]["tainted_args"] == [1]
    assert models["strcpy_chk"]["propagates"] == [{"from": "*arg:1", "to": "*arg:0"}]
    # memset_chk's arg1 is the fill byte (int c), so it must NOT propagate
    assert "propagates" not in models["memset_chk"]
    assert models["memset_chk"]["sink"]["tainted_args"] == [2]
    # fortified source variants seed the destination buffer + return value
    assert {"to": "*arg:1"} in models["read_chk"]["sources"]
    assert {"to": "*arg:0"} in models["fgets_chk"]["sources"]


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


def _fwrite_func():
    # dump(fd): read(fd, &buf, 0x40); fwrite(&buf, 1, 0x40, fp)
    buf = FVar("buf", typ="char[0x40]")
    rsi = FVar("rsi"); rdi = FVar("rdi"); rax = FVar("rax"); fp = FVar("fp")
    rsi1 = FSSA(rsi, 1); rdi1 = FSSA(rdi, 1); rax2 = FSSA(rax, 2); fp1 = FSSA(fp, 1)
    instrs = [
        FInstr(0, 0x10, "MLIL_SET_VAR_SSA", "rsi#1 = &buf", writes=[rsi1],
               src=FExpr("MLIL_ADDRESS_OF", "&buf", src=buf)),
        FInstr(1, 0x14, "MLIL_CALL_SSA", "rax#2 = read(rdi#1, rsi#1, 0x40)",
               reads=[rdi1, rsi1], writes=[rax2],
               dest=FExpr("MLIL_CONST_PTR", "0x401070", constant=0x401070),
               params=[FExpr("MLIL_VAR_SSA", "rdi#1", reads=[rdi1]),
                       FExpr("MLIL_VAR_SSA", "rsi#1", reads=[rsi1]),
                       FExpr("MLIL_CONST", "0x40", constant=0x40)]),
        FInstr(2, 0x18, "MLIL_CALL_SSA", "fwrite(&buf, 1, 0x40, fp#1)", reads=[fp1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x401090", constant=0x401090),
               params=[FExpr("MLIL_ADDRESS_OF", "&buf", src=buf),
                       FExpr("MLIL_CONST", "1", constant=1),
                       FExpr("MLIL_CONST", "0x40", constant=0x40),
                       FExpr("MLIL_VAR_SSA", "fp#1", reads=[fp1])]),
    ]
    bv = FBV({0x401070: "read", 0x401090: "fwrite"})
    return FFunc("dump", 0x10, FSSAFunc(instrs), params=[FVar("fd")]), bv


def test_optional_sink_off_by_default(models):
    func, bv = _fwrite_func()
    engine = te.TaintEngine(bv, models)
    result = engine.forward(func, [te.parse_locator("arg:read:1")])
    # fwrite is modeled but its sink is opt-in -> no finding, and (crucially) no
    # conservative-unmodeled-external assumption (the model suppresses that).
    assert result["reached_sinks"] == []
    assert not any("conservatively tainted" in a for a in result["assumptions"])


def test_optional_sink_on_with_class(models):
    func, bv = _fwrite_func()
    engine = te.TaintEngine(bv, models)
    result = engine.forward(func, [te.parse_locator("arg:read:1")],
                            enabled_sink_classes={"file_write"})
    sinks = result["reached_sinks"]
    assert len(sinks) == 1
    assert sinks[0]["sink"]["callee"] == "fwrite"
    assert sinks[0]["sink"]["class"] == "file_write"
    assert sinks[0]["sink"]["tainted_arg_index"] == 0
    # enabling an unrelated class leaves fwrite silent
    result2 = engine.forward(func, [te.parse_locator("arg:read:1")],
                             enabled_sink_classes={"net_write"})
    assert result2["reached_sinks"] == []


def _sprintf_vararg_func():
    # build(fd): read(fd,&buf,0x40); sprintf(&cmd,"echo %s",&buf); system(&cmd)
    # The format string is an UNTAINTED const ptr; the only taint into cmd is the
    # vararg &buf, so a command_injection at system proves vararg->dest propagation.
    buf = FVar("buf", typ="char[0x40]"); cmd = FVar("cmd", typ="char[0x80]")
    rsi = FVar("rsi"); rdi = FVar("rdi"); rax = FVar("rax"); rc = FVar("rc")
    rsi1 = FSSA(rsi, 1); rdi1 = FSSA(rdi, 1); rax2 = FSSA(rax, 2); rc1 = FSSA(rc, 1)
    instrs = [
        FInstr(0, 0x10, "MLIL_SET_VAR_SSA", "rsi#1 = &buf", writes=[rsi1],
               src=FExpr("MLIL_ADDRESS_OF", "&buf", src=buf)),
        FInstr(1, 0x14, "MLIL_CALL_SSA", "read(rdi#1, rsi#1, 0x40)", reads=[rdi1, rsi1], writes=[rax2],
               dest=FExpr("MLIL_CONST_PTR", "0x401070", constant=0x401070),
               params=[FExpr("MLIL_VAR_SSA", "rdi#1", reads=[rdi1]),
                       FExpr("MLIL_VAR_SSA", "rsi#1", reads=[rsi1]),
                       FExpr("MLIL_CONST", "0x40", constant=0x40)]),
        FInstr(2, 0x18, "MLIL_SET_VAR_SSA", "rc#1 = &cmd", writes=[rc1],
               src=FExpr("MLIL_ADDRESS_OF", "&cmd", src=cmd)),
        FInstr(3, 0x1c, "MLIL_CALL_SSA", "sprintf(rc#1, \"echo %s\", &buf)", reads=[rc1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x401080", constant=0x401080),
               params=[FExpr("MLIL_VAR_SSA", "rc#1", reads=[rc1]),
                       FExpr("MLIL_CONST_PTR", "echo %s", constant=0x4050),
                       FExpr("MLIL_ADDRESS_OF", "&buf", src=buf)]),
        FInstr(4, 0x20, "MLIL_CALL_SSA", "system(rc#1)", reads=[rc1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x401090", constant=0x401090),
               params=[FExpr("MLIL_VAR_SSA", "rc#1", reads=[rc1])]),
    ]
    bv = FBV({0x401070: "read", 0x401080: "sprintf", 0x401090: "system"})
    return FFunc("build", 0x10, FSSAFunc(instrs), params=[FVar("fd")]), bv


def test_forward_vararg_taints_dest_buffer(models):
    func, bv = _sprintf_vararg_func()
    engine = te.TaintEngine(bv, models)
    result = engine.forward(func, [te.parse_locator("arg:read:1")])
    classes = {s["sink"]["class"] for s in result["reached_sinks"]}
    # cmd only became tainted via the vararg -> *arg:0 propagation
    assert "command_injection" in classes
    # the tainted vararg is itself flagged at sprintf (unbounded family)
    assert any(s["sink"]["callee"] == "sprintf" and s["sink"]["tainted_arg_index"] == 2
               for s in result["reached_sinks"])


def test_forward_vararg_no_double_report(models):
    func, bv = _sprintf_vararg_func()
    engine = te.TaintEngine(bv, models)
    result = engine.forward(func, [te.parse_locator("arg:read:1")])
    # the sprintf vararg sink at arg2 must be recorded exactly once despite the
    # fixpoint revisiting the call across iterations.
    sprintf_arg2 = [s for s in result["reached_sinks"]
                    if s["sink"]["callee"] == "sprintf" and s["sink"]["tainted_arg_index"] == 2]
    assert len(sprintf_arg2) == 1


class FBVStr(FBV):
    """FBV that can also resolve a constant string by address (for format-string
    aware vararg gating)."""
    def __init__(self, addr_names, funcs=None, strings=None):
        super().__init__(addr_names, funcs)
        self._strings = strings or {}

    def read(self, addr, length):
        s = self._strings.get(int(addr))
        return (s.encode("latin-1") + b"\x00") if s is not None else b""


def _sprintf_unconsumed_vararg_func(fmt):
    # f(x): sprintf(&buf, fmt, 1, 7, 0xe, x)  -- x is the tainted param.
    # When fmt is a constant, only the varargs its conversions consume are live.
    buf = FVar("buf", typ="char[0x80]"); x = FVar("x"); rc = FVar("rc")
    x1 = FSSA(x, 1); rc1 = FSSA(rc, 1)
    instrs = [
        FInstr(0, 0x10, "MLIL_SET_VAR_SSA", "rc#1 = &buf", writes=[rc1],
               src=FExpr("MLIL_ADDRESS_OF", "&buf", src=buf)),
        FInstr(1, 0x14, "MLIL_CALL_SSA", f"sprintf(rc#1, {fmt!r}, 1, 7, 0xe, x#1)",
               reads=[rc1, x1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x401080", constant=0x401080),
               params=[FExpr("MLIL_VAR_SSA", "rc#1", reads=[rc1]),
                       FExpr("MLIL_CONST_PTR", fmt, constant=0x5000),
                       FExpr("MLIL_CONST", "1", constant=1),
                       FExpr("MLIL_CONST", "7", constant=7),
                       FExpr("MLIL_CONST", "0xe", constant=0xe),
                       FExpr("MLIL_VAR_SSA", "x#1", reads=[x1])]),
    ]
    bv = FBVStr({0x401080: "sprintf"}, strings={0x5000: fmt})
    return FFunc("fmt_fn", 0x10, FSSAFunc(instrs), params=[x]), bv


def test_forward_notes_tainted_memcpy_source(models):
    # memcpy(dst, src, n) with src = the tainted parameter. The copy isn't a sink
    # on its source operand, so reached_sinks stays empty -- but forward must NOTE
    # the src-side copy so it agrees with backward/trace instead of silently
    # reporting "no sinks reached" (#44).
    src = FVar("src"); dst = FVar("dst")
    src1 = FSSA(src, 1)
    instrs = [
        FInstr(0, 0x100, "MLIL_CALL_SSA", "memcpy(&dst, src#1, 0x20)", reads=[src1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x401080", constant=0x401080),
               params=[FExpr("MLIL_ADDRESS_OF", "&dst", src=dst),
                       FExpr("MLIL_VAR_SSA", "src#1", reads=[src1]),
                       FExpr("MLIL_CONST", "0x20", constant=0x20)]),
    ]
    func = FFunc("f", 0x100, FSSAFunc(instrs), params=[src])
    bv = FBV({0x401080: "memcpy"})
    engine = te.TaintEngine(bv, models)
    result = engine.forward(func, [te.parse_locator("param:0")])
    # the copy's source is NOT fabricated into a sink ...
    assert result["reached_sinks"] == [], result["reached_sinks"]
    # ... but the src-side copy is surfaced so forward no longer silently misses it
    assert any("copied into the destination" in a and "memcpy" in a
               for a in result["assumptions"]), result["assumptions"]


def test_forward_no_copy_note_when_source_is_already_a_sink(models):
    # strcpy flags its SOURCE (arg 1) as a sink, so forward already reports it --
    # the #44 copy note must NOT also fire there (it would be redundant and its
    # "not itself flagged as a sink" wording would be false). The note is for the
    # silent case (memcpy) only.
    src = FVar("src"); dst = FVar("dst")
    src1 = FSSA(src, 1)
    instrs = [
        FInstr(0, 0x100, "MLIL_CALL_SSA", "strcpy(&dst, src#1)", reads=[src1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x401090", constant=0x401090),
               params=[FExpr("MLIL_ADDRESS_OF", "&dst", src=dst),
                       FExpr("MLIL_VAR_SSA", "src#1", reads=[src1])]),
    ]
    func = FFunc("f", 0x100, FSSAFunc(instrs), params=[src])
    bv = FBV({0x401090: "strcpy"})
    engine = te.TaintEngine(bv, models)
    result = engine.forward(func, [te.parse_locator("param:0")])
    # strcpy's tainted source IS a real sink (arg 1) ...
    assert any(s["sink"]["callee"] == "strcpy" and s["sink"]["tainted_arg_index"] == 1
               for s in result["reached_sinks"]), result["reached_sinks"]
    # ... so no redundant copy note is emitted for it
    assert not any("copied into the destination" in a for a in result["assumptions"]), \
        result["assumptions"]


def test_forward_unconsumed_vararg_not_reported(models):
    # "%i.%i.%i" consumes 3 args (1,7,0xe); the tainted 4th vararg is never read
    # by the format -> provably dead -> must NOT be reported as a sprintf sink (#45).
    func, bv = _sprintf_unconsumed_vararg_func("%i.%i.%i")
    engine = te.TaintEngine(bv, te.load_models())
    result = engine.forward(func, [te.parse_locator("param:0")])
    sprintf_sinks = [s for s in result["reached_sinks"] if s["sink"]["callee"] == "sprintf"]
    assert sprintf_sinks == [], f"unconsumed tainted vararg wrongly reported: {sprintf_sinks}"


def test_forward_consumed_vararg_still_reported():
    # When the format DOES consume the tainted vararg, it stays a real sink: the
    # gating is precise, not a blanket suppression.
    func, bv = _sprintf_unconsumed_vararg_func("%i.%i.%i.%i")
    engine = te.TaintEngine(bv, te.load_models())
    result = engine.forward(func, [te.parse_locator("param:0")])
    assert any(s["sink"]["callee"] == "sprintf" and s["sink"]["tainted_arg_index"] == 5
               for s in result["reached_sinks"]), result["reached_sinks"]


def test_forward_nonconstant_format_keeps_all_varargs():
    # If the format string is not a discoverable constant, stay conservative and
    # still report the tainted vararg (no false negative).
    func, bv = _sprintf_unconsumed_vararg_func("%i.%i.%i")
    bv._strings = {}  # format no longer resolvable
    engine = te.TaintEngine(bv, te.load_models())
    result = engine.forward(func, [te.parse_locator("param:0")])
    assert any(s["sink"]["callee"] == "sprintf" and s["sink"]["tainted_arg_index"] == 5
               for s in result["reached_sinks"]), result["reached_sinks"]


def test_count_format_args_handles_literals_and_star():
    assert te._count_format_args("%i.%i.%i") == 3
    assert te._count_format_args("100%% done: %s") == 1   # %% is a literal
    assert te._count_format_args("%*d") == 2               # * width consumes an arg
    assert te._count_format_args("%.*f") == 2              # * precision consumes an arg
    assert te._count_format_args("no conversions here") == 0


def test_count_format_args_positional_returns_none():
    # POSIX positional %n$ specifiers reorder/reuse args; a linear count would
    # under-count them to 0 and wrongly suppress a real sink, so return None to
    # keep the caller conservative (#69).
    assert te._count_format_args("%2$s") is None
    assert te._count_format_args("%1$s %2$d") is None
    assert te._count_format_args("err %3$d: %1$s") is None
    # plain (non-positional) formats are still counted precisely
    assert te._count_format_args("%s %d") == 2


def test_forward_positional_format_does_not_suppress_sink(models):
    # A constant format with a positional specifier must NOT have its varargs
    # gated away -- the tainted vararg stays a reportable sink (#69, guarding the
    # #45 fix from over-reaching into a false negative).
    func, bv = _sprintf_unconsumed_vararg_func("%1$s %2$s %3$s")
    engine = te.TaintEngine(bv, te.load_models())
    result = engine.forward(func, [te.parse_locator("param:0")])
    assert any(s["sink"]["callee"] == "sprintf" and s["sink"]["tainted_arg_index"] == 5
               for s in result["reached_sinks"]), result["reached_sinks"]


def test_var_label_of_global():
    assert te.var_label_of((("global", 0x404060), None)) == "glob_0x404060"
    assert te.var_label_of((("global", 0x404060), 2)) == "glob_0x404060#2"


def test_node_label_prefers_recorded_human_label():
    # An id-keyed node carries no name in its key, so the key-only fallback can
    # only render "var#<identifier>". The label captured when the node was
    # tainted (the live register name) must win so JSON output reads like text.
    node = (("id", 1729382256911319086), 2)
    why = {node: {"label": "r1#2", "instr": None, "reason": "", "parents": []}}
    assert te.node_label(node, why) == "r1#2"


def test_node_label_falls_back_to_var_label_of_without_why():
    node = (("id", 99), 2)
    assert te.node_label(node, None) == "var#99#2"
    assert te.node_label(node, {}) == "var#99#2"
    # named/global keys are already readable via the fallback
    assert te.node_label((("name", "buf"), 3), {}) == "buf#3"
    assert te.node_label((("global", 0x404060), None), None) == "glob_0x404060"


def test_global_addr_rejects_readonly():
    class BVWritable:
        def is_offset_writable(self, a):
            return a >= 0x404000  # .bss/.data writable; .rodata below is not

    eng = te.TaintEngine(BVWritable(), te.load_models())
    ro = FExpr("MLIL_CONST_PTR", "0x402000", constant=0x402000)   # rodata
    rw = FExpr("MLIL_CONST_PTR", "0x404060", constant=0x404060)   # .bss
    assert eng._global_addr(None, ro) is None
    assert eng._global_addr(None, rw) == 0x404060


def test_forward_global_buffer_source_to_sink(models):
    # g(fd): read(fd, &glob, 0x40); len = glob[0]; memcpy(d, glob, len)
    # glob is referenced by an absolute address (MLIL_CONST_PTR) — the case
    # _pointee_var misses. Seeding the global + loading back out of it + the memcpy
    # length sink must all fire.
    GLOB = 0x404060
    rdi = FVar("rdi"); length = FVar("len")
    rdi1 = FSSA(rdi, 1); len1 = FSSA(length, 1)

    def glob_ptr():
        return FExpr("MLIL_CONST_PTR", hex(GLOB), constant=GLOB)

    instrs = [
        FInstr(0, 0x10, "MLIL_CALL_SSA", "read(rdi#1, 0x404060, 0x40)", reads=[rdi1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x401070", constant=0x401070),
               params=[FExpr("MLIL_VAR_SSA", "rdi#1", reads=[rdi1]), glob_ptr(),
                       FExpr("MLIL_CONST", "0x40", constant=0x40)]),
        FInstr(1, 0x14, "MLIL_SET_VAR_SSA", "len#1 = [0x404060]", writes=[len1],
               src=FExpr("MLIL_LOAD_SSA", "[0x404060]", src=glob_ptr())),
        FInstr(2, 0x18, "MLIL_CALL_SSA", "memcpy(d, 0x404060, len#1)", reads=[len1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x401080", constant=0x401080),
               params=[FExpr("MLIL_VAR_SSA", "&d", reads=[]), glob_ptr(),
                       FExpr("MLIL_VAR_SSA", "len#1", reads=[len1])]),
    ]
    bv = FBV({0x401070: "read", 0x401080: "memcpy"})
    func = FFunc("g", 0x10, FSSAFunc(instrs), params=[FVar("fd")])
    engine = te.TaintEngine(bv, models)
    result = engine.forward(func, [te.parse_locator("arg:read:1")])
    assert any(s["sink"]["callee"] == "memcpy" and s["sink"]["class"] == "overflow_len"
               and s["sink"]["tainted_arg_index"] == 2 for s in result["reached_sinks"])


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


def test_forward_descends_through_plt_thunk_into_local(models):
    # Same descend scenario, but the call is routed through a single-instruction
    # PLT/veneer thunk (is_thunk=True) to a locally-defined function. Forward
    # taint must follow the thunk and descend into the real implementation,
    # reaching the sink -- not dead-end treating the thunk as an opaque
    # external (#14).
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

    # j_copy_it: a single-tailcall thunk to copy_it. is_thunk=True so it is not
    # itself a descendable body -- the engine must follow it to copy_it.
    thunk = FFunc("j_copy_it", 0x2100, FSSAFunc([
        FInstr(0, 0x2100, "MLIL_TAILCALL_SSA", "tailcall(0x2000)",
               dest=FExpr("MLIL_CONST_PTR", "0x2000", constant=0x2000)),
    ]), is_thunk=True)

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
        FInstr(3, 0x3018, "MLIL_CALL_SSA", "0x2100(&buf, &out)",
               reads=[rsi1, rsi2], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x2100", constant=0x2100),
               params=[FExpr("MLIL_VAR_SSA", "rsi#1", reads=[rsi1]),
                       FExpr("MLIL_VAR_SSA", "rsi_2#1", reads=[rsi2])]),
    ])
    handler = FFunc("handler", 0x3000, handler_ssa, params=[fd])

    bv = FBV({0x1050: "read", 0x1080: "memcpy"}, funcs={0x2000: copy_it, 0x2100: thunk})
    engine = te.TaintEngine(bv, models)
    result = engine.forward(handler, [te.parse_locator("arg:read:1")])

    # the sink lives in copy_it (reached only by following the thunk) and must
    # bubble up as a handler finding, crossing the boundary into copy_it.
    assert len(result["reached_sinks"]) == 1
    sink = result["reached_sinks"][0]["sink"]
    assert sink["callee"] == "memcpy" and sink["class"] == "overflow_len"
    assert any("calls copy_it" in (s.get("reason") or "") for s in result["reached_sinks"][0]["path"])
    assert result["stats"]["functions_visited"] == 2  # handler + copy_it; thunk body skipped


def test_forward_remodels_thunk_to_modeled_external_sink(models):
    # A tainted length flows through j_memcpy -- a veneer whose own name is NOT
    # in the model DB -- which tailcalls the modeled memcpy. Forward taint must
    # re-run lookup_model on the post-thunk target and fire the memcpy sink, not
    # fall through to the conservative "external, no model" tail (#27).
    n = FVar("n", ident=40); n1 = FSSA(n, 1)
    thunk = FFunc("j_memcpy", 0x2100, FSSAFunc([
        FInstr(0, 0x2100, "MLIL_TAILCALL_SSA", "tailcall(0x1080)",
               dest=FExpr("MLIL_CONST_PTR", "0x1080", constant=0x1080)),
    ]), is_thunk=True)
    handler = FFunc("handler", 0x3000, FSSAFunc([
        FInstr(0, 0x3008, "MLIL_CALL_SSA", "0x2100(d, s, n#1)", reads=[n1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x2100", constant=0x2100),
               params=[FExpr("MLIL_VAR_SSA", "d", reads=[]),
                       FExpr("MLIL_VAR_SSA", "s", reads=[]),
                       FExpr("MLIL_VAR_SSA", "n#1", reads=[n1])]),
    ]), params=[n])
    bv = FBV({0x1080: "memcpy"}, funcs={0x2100: thunk})
    engine = te.TaintEngine(bv, models)
    result = engine.forward(handler, [te.parse_locator("param:0")])
    assert len(result["reached_sinks"]) == 1
    sink = result["reached_sinks"][0]["sink"]
    assert sink["callee"] == "memcpy" and sink["class"] == "overflow_len"
    assert sink["tainted_arg_index"] == 2


# --------------------------------------------------------------------------
# backward taint
# --------------------------------------------------------------------------

def test_forward_version_agnostic_pointee_via_aliased_store(models):
    # handler: d#5 = src; f(&d).  f(p): system(p).
    # The aliased store taints d at a specific version; &d references the whole
    # var (version None). The callsite must still see the arg as tainted
    # (version-agnostic pointee match) and descend, reaching the sink.
    src = FVar("src", ident=51); d = FVar("d", ident=50); p = FVar("p", ident=52)
    src0 = FSSA(src, 0); d5 = FSSA(d, 5); p1 = FSSA(p, 1)
    pp = FVar("pp", ident=60); pp0 = FSSA(pp, 0)
    sink_fn = FFunc("f", 0x800, FSSAFunc([
        FInstr(0, 0x804, "MLIL_CALL_SSA", "0x900(pp#0)", reads=[pp0], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x900", constant=0x900),
               params=[FExpr("MLIL_VAR_SSA", "pp#0", reads=[pp0])]),
    ]), params=[pp])
    handler = FFunc("handler", 0x900, FSSAFunc([
        FInstr(0, 0x904, "MLIL_SET_VAR_ALIASED", "d#5 = src#0", reads=[src0], writes=[d5]),
        FInstr(1, 0x908, "MLIL_SET_VAR_SSA", "p#1 = &d", writes=[p1],
               src=FExpr("MLIL_ADDRESS_OF", "&d", src=d)),
        FInstr(2, 0x90c, "MLIL_CALL_SSA", "0x800(p#1)", reads=[p1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x800", constant=0x800),
               params=[FExpr("MLIL_VAR_SSA", "p#1", reads=[p1])]),
    ]), params=[src])
    bv = FBV({0x800: "f", 0x900: "system"}, funcs={0x800: sink_fn})
    engine = te.TaintEngine(bv, models)
    result = engine.forward(handler, [te.parse_locator("param:0")])
    assert any(s["sink"]["class"] == "command_injection" for s in result["reached_sinks"])
    assert any("calls f" in (st.get("reason") or "")
               for s in result["reached_sinks"] for st in s["path"])


def test_forward_struct_field_store_taints_descriptor(models):
    # handle(fd): d.len = src (SET_VAR_ALIASED_FIELD, no vars_written); f(&d).
    # f(p): system(p). The field store must taint d.dest so &d descends.
    src = FVar("src", ident=70); d = FVar("d", ident=71); p = FVar("p", ident=72)
    src0 = FSSA(src, 0); d1 = FSSA(d, 1); d2 = FSSA(d, 2); p1 = FSSA(p, 1)
    pp = FVar("pp", ident=73); pp0 = FSSA(pp, 0)
    sink_fn = FFunc("f", 0x800, FSSAFunc([
        FInstr(0, 0x804, "MLIL_CALL_SSA", "0x900(pp#0)", reads=[pp0], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x900", constant=0x900),
               params=[FExpr("MLIL_VAR_SSA", "pp#0", reads=[pp0])]),
    ]), params=[pp])
    handler = FFunc("handle", 0x900, FSSAFunc([
        # field store exposes NO vars_written; only .dest/.prev/.src/.offset
        FInstr(0, 0x904, "MLIL_SET_VAR_ALIASED_FIELD", "d.len @ mem = src#0",
               reads=[d1, src0], writes=[], dest=d2, prev=d1, offset=8,
               src=FExpr("MLIL_VAR_SSA", "src#0", reads=[src0])),
        FInstr(1, 0x908, "MLIL_SET_VAR_SSA", "p#1 = &d", writes=[p1],
               src=FExpr("MLIL_ADDRESS_OF", "&d", src=d)),
        FInstr(2, 0x90c, "MLIL_CALL_SSA", "0x800(p#1)", reads=[p1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x800", constant=0x800),
               params=[FExpr("MLIL_VAR_SSA", "p#1", reads=[p1])]),
    ]), params=[src])
    bv = FBV({0x800: "f", 0x900: "system"}, funcs={0x800: sink_fn})
    engine = te.TaintEngine(bv, models)
    result = engine.forward(handler, [te.parse_locator("param:0")])
    assert any(s["sink"]["class"] == "command_injection" for s in result["reached_sinks"])
    reasons = [st.get("reason", "") for s in result["reached_sinks"] for st in s["path"]]
    assert any("struct field" in r for r in reasons)


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


def test_backward_param_seed_ascends_into_caller(models):
    # use_len(dst, src, n): memcpy(dst, src, n). Backward from param:2 (n)
    # seeds at n's earliest read and continues into the caller, reaching the
    # recv source — the same ascent as an arg: sink that bottoms out at a param.
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

    fd = FVar("fd"); buf = FVar("buf"); nh = FVar("nh")
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
    result = engine.backward(use_len, [te.parse_locator("param:2")])

    assert result["slices"]
    assert result["slices"][0]["sink"]["kind"] == "param"
    origins = [(sl["origin"]["kind"], sl["origin"].get("callee")) for sl in result["slices"]]
    assert ("source", "recv") in origins


def test_backward_dedupes_caller_fanout_by_origin(models):
    # use_len(dst, src, n): memcpy(dst, src, n). Two caller sites both pass the
    # recv length, so both ascents bottom out at the SAME origin (source recv).
    # Instead of two near-duplicate slices that repeat the in-function chain, the
    # result collapses to ONE slice noting it was reached via 2 call sites (#46).
    dst = FVar("dst", ident=20); src = FVar("src", ident=21); n = FVar("n", ident=22)
    dst0 = FSSA(dst, 0); src0 = FSSA(src, 0); n0 = FSSA(n, 0)
    use_len = FFunc("use_len", 0x800, FSSAFunc([
        FInstr(0, 0x804, "MLIL_CALL_SSA", "0x940(dst#0, src#0, n#0)", reads=[dst0, src0, n0], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x940", constant=0x940),
               params=[FExpr("MLIL_VAR_SSA", "dst#0", reads=[dst0]),
                       FExpr("MLIL_VAR_SSA", "src#0", reads=[src0]),
                       FExpr("MLIL_VAR_SSA", "n#0", reads=[n0])]),
    ]), params=[dst, src, n])

    buf = FVar("buf"); nh = FVar("nh"); nh1 = FSSA(nh, 1); rb = FVar("rb"); rb1 = FSSA(rb, 1)
    CALL_A = 0x920; CALL_B = 0x960
    recv = FInstr(1, 0x910, "MLIL_CALL_SSA", "nh#1 = 0x930(fd, rb#1, 0x40, 0)", reads=[rb1], writes=[nh1],
                  dest=FExpr("MLIL_CONST_PTR", "0x930", constant=0x930),
                  params=[FExpr("MLIL_VAR_SSA", "fd", reads=[]),
                          FExpr("MLIL_VAR_SSA", "rb#1", reads=[rb1]),
                          FExpr("MLIL_CONST", "0x40", constant=0x40),
                          FExpr("MLIL_CONST", "0", constant=0)])

    def _use_len_call(idx, addr):
        return FInstr(idx, addr, "MLIL_CALL_SSA", "0x800(out, rb#1, nh#1)", reads=[rb1, nh1], writes=[],
                      dest=FExpr("MLIL_CONST_PTR", "0x800", constant=0x800),
                      params=[FExpr("MLIL_VAR_SSA", "out", reads=[]),
                              FExpr("MLIL_VAR_SSA", "rb#1", reads=[rb1]),
                              FExpr("MLIL_VAR_SSA", "nh#1", reads=[nh1])])

    handler = FFunc("handler", 0x900, FSSAFunc([
        FInstr(0, 0x904, "MLIL_SET_VAR_SSA", "rb#1 = &buf", writes=[rb1],
               src=FExpr("MLIL_ADDRESS_OF", "&buf", src=buf)),
        recv,
        _use_len_call(2, CALL_A),
        _use_len_call(3, CALL_B),
    ]), params=[FVar("fd")])
    use_len.caller_sites = [FSite(handler, CALL_A), FSite(handler, CALL_B)]

    bv = FBV({0x940: "memcpy", 0x930: "recv", 0x800: "use_len"})
    engine = te.TaintEngine(bv, models)
    result = engine.backward(use_len, [te.parse_locator("param:2")])

    source_slices = [s for s in result["slices"] if s["origin"].get("callee") == "recv"]
    assert len(source_slices) == 1, [s["origin"] for s in result["slices"]]
    assert source_slices[0].get("reached_via_call_sites") == 2


def test_backward_ret_sink_rejected_with_guidance(process_func, models):
    bv = FBV({0x401070: "read", 0x401080: "memcpy"})
    engine = te.TaintEngine(bv, models)
    with pytest.raises(te.TaintError, match="forward-only"):
        engine.backward(process_func, [te.parse_locator("ret:read")])


def test_backward_arg_no_such_callee_names_the_callee(process_func, models):
    bv = FBV({0x401070: "read", 0x401080: "memcpy"})
    engine = te.TaintEngine(bv, models)
    # the locator is well-formed; there is simply no call to strcpy here.
    with pytest.raises(te.TaintError, match=r"no call to 'strcpy' found"):
        engine.backward(process_func, [te.parse_locator("arg:strcpy:0")])


def test_backward_arg_index_out_of_range_says_so(process_func, models):
    bv = FBV({0x401070: "read", 0x401080: "memcpy"})
    engine = te.TaintEngine(bv, models)
    # memcpy is called with 3 args; index 9 is out of range, not a bad locator.
    with pytest.raises(te.TaintError, match=r"out of range for memcpy"):
        engine.backward(process_func, [te.parse_locator("arg:memcpy:9")])


def test_backward_arg_with_no_variable_reads_says_so(process_func, models):
    bv = FBV({0x401070: "read", 0x401080: "memcpy"})
    engine = te.TaintEngine(bv, models)
    # process_func passes &buf (an address expr, no SSA reads) as memcpy arg 1:
    # there is nothing to slice, but the locator itself was fine.
    with pytest.raises(te.TaintError, match=r"reads no variable in the recovered IL"):
        engine.backward(process_func, [te.parse_locator("arg:memcpy:1")])


def test_parse_locator_rejects_negative_arg_index():
    # arg:<callee>:-1 must be rejected, not silently seed params[-1].
    with pytest.raises(te.TaintError, match=r"index must be >= 0"):
        te.parse_locator("arg:memcpy:-1")


def test_parse_locator_rejects_negative_param_index():
    with pytest.raises(te.TaintError, match=r"index must be >= 0"):
        te.parse_locator("param:-2")


def test_parse_locator_rejects_non_integer_index():
    with pytest.raises(te.TaintError, match=r"must be an integer"):
        te.parse_locator("arg:memcpy:two")


def test_backward_negative_index_guarded_for_direct_callers(process_func, models):
    # A programmatic caller that skips parse_locator and builds the dict directly
    # must still be rejected before the idx < len(params) check.
    bv = FBV({0x401070: "read", 0x401080: "memcpy"})
    engine = te.TaintEngine(bv, models)
    with pytest.raises(te.TaintError, match=r"must be >= 0"):
        engine.backward(process_func, [{"kind": "arg", "callee": "memcpy", "index": -1}])


def test_backward_per_sink_isolation_keeps_resolvable_slices(process_func, models):
    # memcpy is called here; strcpy is not. The un-seedable strcpy sink must not
    # discard the valid memcpy slice — exit with partial results plus a note.
    bv = FBV({0x401070: "read", 0x401080: "memcpy"})
    engine = te.TaintEngine(bv, models)
    result = engine.backward(
        process_func,
        [te.parse_locator("arg:memcpy:2"), te.parse_locator("arg:strcpy:0")],
    )
    assert result["slices"], "the resolvable memcpy slice must survive"
    assert all(sl["sink"]["callee"] == "memcpy" for sl in result["slices"])
    status = {s.get("callee"): s for s in result["sink_status"]}
    assert status["memcpy"]["seeded"] is True
    assert status["strcpy"]["seeded"] is False
    assert "no call to 'strcpy'" in status["strcpy"]["note"]


def test_backward_all_sinks_unseedable_is_hard_error(process_func, models):
    bv = FBV({0x401070: "read", 0x401080: "memcpy"})
    engine = te.TaintEngine(bv, models)
    with pytest.raises(te.TaintError, match=r"no backward seed resolved for any sink"):
        engine.backward(
            process_func,
            [te.parse_locator("arg:strcpy:0"), te.parse_locator("arg:fread:0")],
        )


def test_backward_single_unseedable_sink_preserves_original_error(process_func, models):
    # A single failing sink keeps the precise original message (not the
    # multi-sink aggregate), preserving the established single-sink UX.
    bv = FBV({0x401070: "read", 0x401080: "memcpy"})
    engine = te.TaintEngine(bv, models)
    with pytest.raises(te.TaintError, match=r"no call to 'strcpy' found"):
        engine.backward(process_func, [te.parse_locator("arg:strcpy:0")])


def test_backward_seeds_through_thunk_to_sink(models):
    # The function calls memcpy through a j_memcpy veneer. Backward seeding must
    # follow the thunk so `--sink arg:memcpy:0` finds the callsite -- the
    # backward dual of the forward thunk-follow (#35). Before the fix,
    # _find_callsites name-matched only the pre-thunk name and seeded nothing.
    a = FVar("a", ident=50); a1 = FSSA(a, 1)
    thunk = _tailcall_thunk("j_memcpy", 0x2100, 0x1080)
    caller = FFunc("caller", 0x3000, FSSAFunc([
        FInstr(0, 0x3008, "MLIL_CALL_SSA", "0x2100(a#1, s, n)", reads=[a1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x2100", constant=0x2100),
               params=[FExpr("MLIL_VAR_SSA", "a#1", reads=[a1]),
                       FExpr("MLIL_VAR_SSA", "s", reads=[]),
                       FExpr("MLIL_VAR_SSA", "n", reads=[])]),
    ]), params=[a])
    bv = FBV({0x1080: "memcpy"}, funcs={0x2100: thunk})
    engine = te.TaintEngine(bv, models)
    result = engine.backward(caller, [te.parse_locator("arg:memcpy:0")])
    status = {s.get("callee"): s for s in result["sink_status"]}
    assert status["memcpy"]["seeded"] is True, "thunked memcpy callsite must seed"
    assert result["slices"], "seeding through the thunk must produce a slice"


def test_backward_direct_callsite_still_seeds_without_thunk(models):
    # Guard: the non-thunk path is unchanged -- a direct memcpy call still seeds.
    a = FVar("a", ident=51); a1 = FSSA(a, 1)
    caller = FFunc("caller", 0x3000, FSSAFunc([
        FInstr(0, 0x3008, "MLIL_CALL_SSA", "0x1080(a#1, s, n)", reads=[a1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x1080", constant=0x1080),
               params=[FExpr("MLIL_VAR_SSA", "a#1", reads=[a1]),
                       FExpr("MLIL_VAR_SSA", "s", reads=[]),
                       FExpr("MLIL_VAR_SSA", "n", reads=[])]),
    ]), params=[a])
    bv = FBV({0x1080: "memcpy"})
    engine = te.TaintEngine(bv, models)
    result = engine.backward(caller, [te.parse_locator("arg:memcpy:0")])
    assert {s.get("callee"): s for s in result["sink_status"]}["memcpy"]["seeded"] is True


def test_backward_walk_truncation_recorded(process_func, models):
    # an engine-level def-chain cap of 1 cannot reach the end of the
    # rdx_1#1 <- len#2 <- len#1 chain; the cut must surface in assumptions
    bv = FBV({0x401070: "read", 0x401080: "memcpy"})
    engine = te.TaintEngine(bv, models, max_depth=1)
    result = engine.backward(process_func, [te.parse_locator("arg:memcpy:2")])
    assert any("truncated" in a for a in result["assumptions"])


def test_backward_constant_through_copy_labeled_constant(models):
    # size = 0 reaches memcpy's length arg through a variable copy:
    #   var_2c#1 = 0 ; r2#4 = var_2c#1 ; memcpy(dst, src, r2#4)
    # The slice must bottom out at `constant 0`, not the default `unresolved` --
    # for an auditor those are opposite risk conclusions (#43).
    v2c = FVar("var_2c"); r2 = FVar("r2")
    v2c1 = FSSA(v2c, 1); r2_4 = FSSA(r2, 4)
    instrs = [
        FInstr(0, 0x100, "MLIL_SET_VAR_SSA", "var_2c#1 = 0", writes=[v2c1],
               src=FExpr("MLIL_CONST", "0", constant=0)),
        FInstr(1, 0x104, "MLIL_SET_VAR_SSA", "r2#4 = var_2c#1", reads=[v2c1], writes=[r2_4],
               src=FExpr("MLIL_VAR_SSA", "var_2c#1", reads=[v2c1])),
        FInstr(2, 0x108, "MLIL_CALL_SSA", "memcpy(dst, src, r2#4)", reads=[r2_4], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x401080", constant=0x401080),
               params=[FExpr("MLIL_VAR_SSA", "dst", reads=[]),
                       FExpr("MLIL_VAR_SSA", "src", reads=[]),
                       FExpr("MLIL_VAR_SSA", "r2#4", reads=[r2_4])]),
    ]
    bv = FBV({0x401080: "memcpy"})
    func = FFunc("f", 0x100, FSSAFunc(instrs), params=[])
    engine = te.TaintEngine(bv, models)
    result = engine.backward(func, [te.parse_locator("arg:memcpy:2")])
    origins = [sl["origin"] for sl in result["slices"]]
    assert any(o.get("kind") == "constant" and o.get("value") == 0 for o in origins), origins


def test_backward_copy_of_nonconstant_stays_unresolved(models):
    # Guard the narrow #43 fix: a copy chain that does NOT bottom out at a literal
    # must keep its existing classification (not be mislabeled `constant`).
    a = FVar("a"); r2 = FVar("r2"); g = FVar("g")
    a1 = FSSA(a, 1); r2_4 = FSSA(r2, 4); g1 = FSSA(g, 1)
    instrs = [
        FInstr(0, 0x100, "MLIL_SET_VAR_SSA", "a#1 = g#1", reads=[g1], writes=[a1],
               src=FExpr("MLIL_VAR_SSA", "g#1", reads=[g1])),
        FInstr(1, 0x104, "MLIL_SET_VAR_SSA", "r2#4 = a#1", reads=[a1], writes=[r2_4],
               src=FExpr("MLIL_VAR_SSA", "a#1", reads=[a1])),
        FInstr(2, 0x108, "MLIL_CALL_SSA", "memcpy(dst, src, r2#4)", reads=[r2_4], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x401080", constant=0x401080),
               params=[FExpr("MLIL_VAR_SSA", "dst", reads=[]),
                       FExpr("MLIL_VAR_SSA", "src", reads=[]),
                       FExpr("MLIL_VAR_SSA", "r2#4", reads=[r2_4])]),
    ]
    bv = FBV({0x401080: "memcpy"})
    func = FFunc("f", 0x100, FSSAFunc(instrs), params=[])
    engine = te.TaintEngine(bv, models)
    result = engine.backward(func, [te.parse_locator("arg:memcpy:2")])
    origins = [sl["origin"] for sl in result["slices"]]
    assert all(o.get("kind") != "constant" for o in origins), origins


def test_find_callsites_matches_address_form_locator(models):
    # A call the IL renders only by address (PLT veneer / no recovered symbol):
    # arg:0x12fa4:1 must seed the same callsite the name form would (#58).
    a = FVar("a"); a1 = FSSA(a, 1)
    instrs = [
        FInstr(0, 0x100, "MLIL_SET_VAR_SSA", "a#1 = arg", writes=[a1]),
        FInstr(1, 0x108, "MLIL_CALL_SSA", "0x12fa4(dst, a#1, 0x10)", reads=[a1], writes=[],
               dest=FExpr("MLIL_CONST_PTR", "0x12fa4", constant=0x12fa4),
               params=[FExpr("MLIL_VAR_SSA", "dst", reads=[]),
                       FExpr("MLIL_VAR_SSA", "a#1", reads=[a1]),
                       FExpr("MLIL_CONST", "0x10", constant=0x10)]),
    ]
    func = FFunc("f", 0x100, FSSAFunc(instrs), params=[])
    bv = FBV({})  # 0x12fa4 has no symbol -> name match fails; address match must work
    engine = te.TaintEngine(bv, models)
    assert len(engine._find_callsites(instrs, "0x12fa4")) == 1
    assert len(engine._find_callsites(instrs, "0x12fa5")) == 1   # THUMB-bit tolerant
    assert engine._find_callsites(instrs, "0x9999") == []        # other address: no match
    # end-to-end: backward arg:0x12fa4:1 seeds (no "no call found" error)
    result = engine.backward(func, [te.parse_locator("arg:0x12fa4:1")])
    assert result["sink_status"][0]["seeded"] is True


def test_callee_as_addr_parsing():
    assert te.TaintEngine._callee_as_addr("0x12fa4") == 0x12fa4
    assert te.TaintEngine._callee_as_addr("4711") == 4711
    assert te.TaintEngine._callee_as_addr("memcpy") is None
    assert te.TaintEngine._callee_as_addr("Parameter::data") is None


# --------------------------------------------------------------------------
# unified call-target / thunk resolver (issue #7, PR1)
# --------------------------------------------------------------------------

class FBVSym(FBV):
    """FBV plus symbol-name resolution APIs (which the base FBV lacks)."""

    def __init__(self, addr_names, funcs=None, sym_names=None):
        super().__init__(addr_names, funcs)
        self._sym_names = sym_names or {}  # name -> address

    def get_symbols_by_name(self, name):
        addr = self._sym_names.get(name)
        return [type("S", (), {"address": addr})()] if addr is not None else []

    def get_symbol_by_raw_name(self, name):
        addr = self._sym_names.get(name)
        return type("S", (), {"address": addr})() if addr is not None else None


def _tailcall_thunk(name, start, target_addr):
    return FFunc(name, start, FSSAFunc([
        FInstr(0, start, "MLIL_TAILCALL_SSA", f"tailcall(0x{target_addr:x})",
               dest=FExpr("MLIL_CONST_PTR", f"0x{target_addr:x}", constant=target_addr)),
    ]))


def _leaf_func(name, start):
    return FFunc(name, start, FSSAFunc([FInstr(0, start, "MLIL_RET", "return", reads=[])]))


# -- extract_dest_address ---------------------------------------------------

def test_extract_dest_address_raw_int():
    assert te.extract_dest_address(FBV({}), 0x401000) == 0x401000


def test_extract_dest_address_const_ptr_without_symbol_apis():
    # base FBV has no get_symbols_by_name; must fall back to .constant, not crash
    dest = FExpr("MLIL_CONST_PTR", "0x401070", constant=0x401070)
    assert te.extract_dest_address(FBV({}), dest) == 0x401070


def test_extract_dest_address_import_name_before_constant():
    # .constant is the GOT slot; the import name resolves to the real entry
    dest = FExpr("MLIL_IMPORT", "memcpy", constant=0x600000)
    dest.name = "memcpy"
    bv = FBVSym({}, funcs={0x401050: _leaf_func("memcpy", 0x401050)},
                sym_names={"memcpy": 0x401050})
    assert te.extract_dest_address(bv, dest) == 0x401050


def test_extract_dest_address_unresolvable_returns_none():
    dest = FExpr("MLIL_VAR_SSA", "fp#1")  # no constant, no resolvable name
    assert te.extract_dest_address(FBV({}), dest) is None


# -- targets_from_pvs -------------------------------------------------------

def test_targets_from_pvs_constant():
    assert te.targets_from_pvs(FPVS("ConstantPointerValue", value=0x401000)) == [0x401000]


def test_targets_from_pvs_in_set_sorted_unique():
    assert te.targets_from_pvs(FPVS("InSetOfValues", values=[0x300, 0x100, 0x300, 0x200])) == [0x100, 0x200, 0x300]


def test_targets_from_pvs_lookup_table_mapping():
    assert te.targets_from_pvs(FPVS("LookupTableValue", mapping={0: 0x500, 1: 0x600})) == [0x500, 0x600]


def test_targets_from_pvs_none():
    assert te.targets_from_pvs(None) == []


# -- follow_thunk -----------------------------------------------------------

def test_follow_thunk_single_tailcall_resolves_target():
    real = _leaf_func("real_impl", 0x401200)
    thunk = _tailcall_thunk("j_real_impl", 0x401100, 0x401200)
    bv = FBV({}, funcs={0x401200: real, 0x401100: thunk})
    assert te.follow_thunk(bv, thunk) is real


def test_follow_thunk_non_thunk_returns_none():
    real = _leaf_func("real_impl", 0x401200)
    bv = FBV({}, funcs={0x401200: real})
    assert te.follow_thunk(bv, real) is None


def test_follow_thunk_self_loop_returns_none():
    # a PLT stub whose tailcall resolves back to itself must not recurse forever
    selfish = _tailcall_thunk("plt_stub", 0x401100, 0x401100)
    bv = FBV({}, funcs={0x401100: selfish})
    assert te.follow_thunk(bv, selfish) is None


def test_follow_thunk_two_step_cycle_terminates():
    # A->B->A: the old recursive form only rejected a direct self-loop, so this
    # multi-step cycle recursed without bound (issue #42). Must terminate.
    a = _tailcall_thunk("a", 0x401100, 0x401200)
    b = _tailcall_thunk("b", 0x401200, 0x401100)
    bv = FBV({}, funcs={0x401100: a, 0x401200: b})
    assert te.follow_thunk(bv, a) is b  # deepest target before the cycle closes


def test_follow_thunk_long_cycle_terminates():
    # A->B->C->A: a longer tailcall cycle must also terminate, not recurse.
    a = _tailcall_thunk("a", 0x401100, 0x401200)
    b = _tailcall_thunk("b", 0x401200, 0x401300)
    c = _tailcall_thunk("c", 0x401300, 0x401100)
    bv = FBV({}, funcs={0x401100: a, 0x401200: b, 0x401300: c})
    assert te.follow_thunk(bv, a) is c


def test_follow_thunk_multi_hop_chain_resolves_deepest():
    # A->B->real (acyclic, len>1): must follow all the way to the leaf, not stop
    # at the first hop. Guards against the cycle fix breaking real chains.
    real = _leaf_func("real_impl", 0x401300)
    a = _tailcall_thunk("a", 0x401100, 0x401200)
    b = _tailcall_thunk("b", 0x401200, 0x401300)
    bv = FBV({}, funcs={0x401100: a, 0x401200: b, 0x401300: real})
    assert te.follow_thunk(bv, a) is real


# -- resolve_call_target ----------------------------------------------------

def test_resolve_call_target_direct_const():
    target = _leaf_func("read", 0x401070)
    call = FInstr(0, 0x401000, "MLIL_CALL_SSA", "0x401070()",
                  dest=FExpr("MLIL_CONST_PTR", "0x401070", constant=0x401070))
    bv = FBV({}, funcs={0x401070: target})
    rt = te.resolve_call_target(bv, call)
    assert rt.function is target
    assert rt.address == 0x401070


def test_resolve_call_target_import_name():
    target = _leaf_func("memcpy", 0x401050)
    dest = FExpr("MLIL_IMPORT", "memcpy", constant=0x600000)
    dest.name = "memcpy"
    call = FInstr(0, 0x401000, "MLIL_CALL_SSA", "memcpy()", dest=dest)
    bv = FBVSym({}, funcs={0x401050: target}, sym_names={"memcpy": 0x401050})
    rt = te.resolve_call_target(bv, call)
    assert rt.function is target


def test_resolve_call_target_follows_thunk_when_requested():
    real = _leaf_func("real_impl", 0x401200)
    thunk = _tailcall_thunk("j_real_impl", 0x401100, 0x401200)
    call = FInstr(0, 0x401000, "MLIL_CALL_SSA", "0x401100()",
                  dest=FExpr("MLIL_CONST_PTR", "0x401100", constant=0x401100))
    bv = FBV({}, funcs={0x401100: thunk, 0x401200: real})
    assert te.resolve_call_target(bv, call, follow_thunks=False).function is thunk
    assert te.resolve_call_target(bv, call, follow_thunks=True).function is real


def test_resolve_call_target_unresolved_indirect():
    call = FInstr(0, 0x70c, "MLIL_CALL_SSA", "fp()", dest=FExpr("MLIL_VAR_SSA", "fp#1"))
    rt = te.resolve_call_target(FBV({}), call)
    assert rt.address is None
    assert rt.function is None


# ---------------------------------------------------------------------------
# #98 — arg: locator with C++ qualified callee names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec,expected", [
    ("arg:Namespace::method:1", {"kind": "arg", "callee": "Namespace::method", "index": 1}),
    ("arg:a::b::c:0", {"kind": "arg", "callee": "a::b::c", "index": 0}),
    ("arg:memcpy:1", {"kind": "arg", "callee": "memcpy", "index": 1}),  # plain still works
])
def test_parse_locator_arg_splits_at_last_colon(spec, expected):
    assert te.parse_locator(spec) == expected


def test_parse_locator_arg_requires_an_index():
    # No trailing :<n> -> error, not a silent mis-parse.
    with pytest.raises(te.TaintError, match=r"arg:<callee>:<n>"):
        te.parse_locator("arg:memcpy")
    with pytest.raises(te.TaintError, match=r"arg:<callee>:<n>"):
        te.parse_locator("arg::1")  # empty callee


# ---------------------------------------------------------------------------
# #97 — model-load failures are loud, not silent
# ---------------------------------------------------------------------------


def test_load_models_raises_on_corrupt_builtin(monkeypatch, tmp_path):
    bad = tmp_path / "builtin.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setattr(te, "_BUILTIN_MODELS", bad)
    with pytest.raises(te.TaintError, match="packaging bug"):
        te.load_models()


def test_load_models_raises_on_broken_override(monkeypatch, tmp_path):
    bad = tmp_path / "override.json"
    bad.write_text("{ broken", encoding="utf-8")
    monkeypatch.setattr(te, "taint_models_path", lambda: bad)
    with pytest.raises(te.TaintError, match="BN_TAINT_MODELS"):
        te.load_models()


def test_load_models_rejects_non_object_override(monkeypatch, tmp_path):
    bad = tmp_path / "override.json"
    bad.write_text('["a", "b"]', encoding="utf-8")
    monkeypatch.setattr(te, "taint_models_path", lambda: bad)
    with pytest.raises(te.TaintError, match="must be a JSON object"):
        te.load_models()


def test_load_models_rejects_non_dict_model_value(monkeypatch, tmp_path):
    bad = tmp_path / "override.json"
    bad.write_text('{"models": {"memcpy": "oops"}}', encoding="utf-8")
    monkeypatch.setattr(te, "taint_models_path", lambda: bad)
    with pytest.raises(te.TaintError, match="must be a JSON object"):
        te.load_models()


def test_load_models_accepts_valid_override_with_comment(monkeypatch, tmp_path):
    ok = tmp_path / "override.json"
    ok.write_text(
        '{"my_custom_sink": {"sink": {"class": "x", "tainted_args": [0]}}, "_comment_x": "doc text"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(te, "taint_models_path", lambda: ok)
    models = te.load_models()
    assert "my_custom_sink" in models           # override merged over builtin
    assert "memcpy" in models                   # builtin still present
    assert "_comment_x" not in [k for k in models if not k.startswith("_comment")]


# ---------------------------------------------------------------------------
# #89 — import + Thumb call-target resolution
# ---------------------------------------------------------------------------


def test_function_at_normalizes_thumb_low_bit():
    target = type("F", (), {"name": "handler", "start": 0x8000})()
    bv = FBV({}, funcs={0x8000: target})
    # Even address resolves directly; odd (Thumb-tagged) falls back to addr & ~1.
    assert te.function_at(bv, 0x8000) is target
    assert te.function_at(bv, 0x8001) is target
    assert te.function_at(bv, 0x9000) is None  # genuinely absent
    assert te.function_at(object(), 0x8000) is None  # no get_function_at


def test_resolve_direct_target_resolves_import_call():
    # An MLIL_IMPORT dest exposes a symbol .name and a GOT-slot .constant; the
    # resolver must resolve via the name to the function entry, not the GOT slot.
    memcpy_fn = type("F", (), {"name": "memcpy", "start": 0x1100})()

    class _ImportBV:
        def get_function_at(self, addr):
            return memcpy_fn if addr == 0x1100 else None

        def get_symbols_by_name(self, name):
            if name == "memcpy":
                return [type("S", (), {"address": 0x1100})()]
            return []

        def get_symbol_by_raw_name(self, name):
            return None

    dest = FExpr("MLIL_IMPORT", "memcpy", constant=0x2000)  # 0x2000 = GOT slot
    dest.name = "memcpy"
    call = FInstr(0, 0x500, "MLIL_CALL_SSA", "memcpy(...)", dest=dest, params=[])
    engine = te.TaintEngine(_ImportBV(), te.load_models())
    assert engine._resolve_direct_target(call) == 0x1100  # entry, not GOT slot


def test_resolve_direct_target_constant_direct_call_unchanged():
    fn = type("F", (), {"name": "helper", "start": 0x401070})()
    bv = FBV({}, funcs={0x401070: fn})
    dest = FExpr("MLIL_CONST_PTR", "0x401070", constant=0x401070)
    call = FInstr(0, 0x401000, "MLIL_CALL_SSA", "helper()", dest=dest, params=[])
    engine = te.TaintEngine(bv, te.load_models())
    assert engine._resolve_direct_target(call) == 0x401070
