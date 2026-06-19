from __future__ import annotations

import importlib
import importlib.util
import io
import json
import socket
import sys
import threading
import time
import types
import weakref
from pathlib import Path

import pytest


def _load_bridge(monkeypatch):
    # Loading the bridge from its file writes bytecode into
    # plugin/bn_agent_bridge/__pycache__, which the wheel build then ships as
    # data files (uv_build can't exclude inside data trees). Don't pollute the
    # source tree so dev builds stay as clean as a release build (#83).
    sys.dont_write_bytecode = True
    fake_bn = types.ModuleType("binaryninja")

    class SymbolType:
        FunctionSymbol = "SymbolType.FunctionSymbol"
        DataSymbol = "SymbolType.DataSymbol"
        ImportedFunctionSymbol = "SymbolType.ImportedFunctionSymbol"
        ImportedDataSymbol = "SymbolType.ImportedDataSymbol"
        ImportAddressSymbol = "SymbolType.ImportAddressSymbol"
        ExternalSymbol = "SymbolType.ExternalSymbol"

    class SymbolBinding:
        NoBinding = "SymbolBinding.NoBinding"
        LocalBinding = "SymbolBinding.LocalBinding"
        GlobalBinding = "SymbolBinding.GlobalBinding"
        WeakBinding = "SymbolBinding.WeakBinding"

    class Symbol:
        def __init__(self, symbol_type, address, name, binding=None):
            self.type = symbol_type
            self.address = address
            self.name = name
            self.raw_name = name
            self.binding = binding

    class Type:
        # Minimal stand-ins for the BN class methods the namespaced-type
        # fallback uses (#200). named_type_from_type yields a reference whose
        # str is the type name; pointer wraps with a trailing '*'.
        @staticmethod
        def named_type_from_type(name, type_obj):
            return _FakeType(str(name), type_class="NamedTypeReferenceClass")

        @staticmethod
        def pointer(arch, type_obj):
            return _FakeType(f"{type_obj}*", type_class="PointerTypeClass")

    class QualifiedName:
        # Models BN's QualifiedName: a multi-component name. Crucially, coercing a
        # raw string does NOT split on '::' -- it becomes a SINGLE component, just
        # like real BN -- so a raw "ns::Foo" lookup misses a type registered as
        # the components ['ns','Foo']. _FakeBV.get_type_by_name keys on the
        # component tuple, so the test reproduces the #200 lookup behavior.
        def __init__(self, components):
            if isinstance(components, str):
                self.name = [components]
            else:
                self.name = [str(c) for c in components]

        def __str__(self):
            return "::".join(self.name)

    fake_bn.SymbolType = SymbolType
    fake_bn.SymbolBinding = SymbolBinding
    fake_bn.Symbol = Symbol
    fake_bn.QualifiedName = QualifiedName
    fake_bn.Type = Type
    fake_bn.log_info = lambda *args, **kwargs: None
    fake_bn.log_warn = lambda *args, **kwargs: None
    fake_bn.log_error = lambda *args, **kwargs: None

    fake_bn.SSAVariable = SSAVariable

    fake_mainthread = types.ModuleType("binaryninja.mainthread")
    fake_mainthread.execute_on_main_thread_and_wait = lambda func: func()
    fake_mainthread.is_main_thread = lambda: True

    fake_plugin = types.ModuleType("binaryninja.plugin")

    class PluginCommand:
        @staticmethod
        def register(*args, **kwargs):
            return None

    fake_plugin.PluginCommand = PluginCommand

    monkeypatch.setitem(sys.modules, "binaryninja", fake_bn)
    monkeypatch.setitem(sys.modules, "binaryninja.mainthread", fake_mainthread)
    monkeypatch.setitem(sys.modules, "binaryninja.plugin", fake_plugin)
    monkeypatch.delitem(sys.modules, "binaryninjaui", raising=False)
    package_name = "bn_test_bridge"
    module_name = f"{package_name}.bridge"
    # Purge every cached bn_test_bridge.* submodule (bridge + its helper modules
    # il_format/vars/seam/_shared/op_registry/taint_engine/...) so each load
    # re-imports them against THIS call's fake `binaryninja`. Without this, a
    # helper module loaded by an earlier _load_bridge stays bound to a stale
    # fake_bn, and patches against the fresh `bridge.bn` never reach it.
    for cached in [name for name in sys.modules if name == package_name or name.startswith(f"{package_name}.")]:
        monkeypatch.delitem(sys.modules, cached, raising=False)

    bridge_path = Path(__file__).resolve().parents[1] / "plugin" / "bn_agent_bridge" / "bridge.py"
    package = types.ModuleType(package_name)
    package.__path__ = [str(bridge_path.parent)]
    monkeypatch.setitem(sys.modules, package_name, package)
    spec = importlib.util.spec_from_file_location(module_name, bridge_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _FakeFunction:
    def __init__(self, start: int, name: str, type_text: str = "int32_t()"):
        self.start = start
        self.name = name
        self.raw_name = name
        self.type = type_text
        self.parameter_vars = []
        self.stack_layout = []
        self.calling_convention = "__cdecl"
        self.return_type = "int32_t"
        self.basic_blocks = []
        self.low_level_il = []
        self.analysis_skipped = False
        self.analysis_skip_reason = "NoSkipReason"
        self.reanalyzed = False

    def reanalyze(self, *args, **kwargs):
        self.reanalyzed = True


class _FakeBasicBlock:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end


class _FakeInstructionInfo:
    def __init__(self, length: int):
        self.length = length


class _FakeArch:
    def __init__(self, lengths=None, *, name: str = "x86", address_size: int = 4):
        self.name = name
        self.address_size = address_size
        self.max_instr_length = 16
        self.lengths = dict(lengths or {})

    def __str__(self):
        return self.name

    def get_instruction_info(self, data, address):
        return _FakeInstructionInfo(self.lengths.get(int(address), 1))


class _FakeOperation:
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name


class _FakeConstPtr:
    def __init__(self, constant: int):
        self.operation = _FakeOperation("LLIL_CONST_PTR")
        self.constant = constant


class _FakeReg:
    def __init__(self, name: str):
        self.operation = _FakeOperation("LLIL_REG")
        self.name = name


class _FakeHLILInstructionNode:
    def __init__(self, text: str, *, condition=None, parent=None, expr_index: int = 0, instr_index: int = 0):
        self.text = text
        self.condition = condition
        self.parent = parent
        self.expr_index = expr_index
        self.instr_index = instr_index

    def __str__(self):
        return self.text


_FAKE_HLIL_TYPES: dict[str, type[_FakeHLILInstructionNode]] = {}


def _FakeHLILInstruction(
    text: str,
    *,
    class_name: str,
    condition=None,
    parent=None,
    expr_index: int = 0,
    instr_index: int = 0,
):
    cls = _FAKE_HLIL_TYPES.get(class_name)
    if cls is None:
        cls = type(class_name, (_FakeHLILInstructionNode,), {})
        _FAKE_HLIL_TYPES[class_name] = cls
    return cls(
        text,
        condition=condition,
        parent=parent,
        expr_index=expr_index,
        instr_index=instr_index,
    )


class _FakeLLILInstruction:
    def __init__(self, address: int, dest, *, operation: str = "LLIL_CALL", hlils=None):
        self.address = address
        self.dest = dest
        self.operation = _FakeOperation(operation)
        self.hlils = list(hlils or [])
        self.mlils = []
        self.mapped_medium_level_il = None


class _FakeVariable:
    def __init__(
        self,
        *,
        name: str,
        storage: int,
        var_type: str,
        identifier: int,
        index: int = 0,
        source_type: str = "StackVariableSourceType",
    ):
        self.name = name
        self.storage = storage
        self.type = var_type
        self.identifier = identifier
        self.index = index
        self.source_type = types.SimpleNamespace(name=source_type)


class _FakeStringRef:
    def __init__(self, start: int, length: int, value: str, string_type: int = 0):
        self.start = start
        self.length = length
        self.value = value
        self.type = string_type


class _FakeCodeRef:
    def __init__(self, address: int, function=None):
        self.address = address
        self.function = function


class _FakeSection:
    def __init__(self, name: str, start: int, end: int, semantics: int = 0):
        self.name = name
        self.start = start
        self.end = end
        self.semantics = semantics


class _FakeSegment:
    def __init__(self, *, readable: bool = True, writable: bool = False, executable: bool = False):
        self.readable = readable
        self.writable = writable
        self.executable = executable


class _FakeBV:
    def __init__(self, *, functions=None, symbols=None, types_=None, qualified_types_=None, arch=None, disassembly=None, instruction_lengths=None,
                 strings=None, sections=None, segments=None, memory=None, code_refs=None, data_refs=None):
        self.functions = list(functions or [])
        self._symbols = list(symbols or [])
        self.types = dict(types_ or {})
        # Types registered under a multi-component QualifiedName (keyed by the
        # component tuple), mirroring how BN registers namespaced C++ types -- a
        # raw "ns::Foo" string lookup misses these (#200).
        self._qualified_types = dict(qualified_types_ or {})
        self.arch = arch or _FakeArch(instruction_lengths)
        self._disassembly = dict(disassembly or {})
        self._instruction_lengths = dict(instruction_lengths or {})
        self.strings = list(strings or [])
        self.sections = dict(sections or {})
        self._segments = dict(segments or {})
        # Map of {base_address: bytes} describing contiguous mapped regions.
        self._memory = dict(memory or {})
        self._code_refs = dict(code_refs or {})
        self._data_refs = dict(data_refs or {})

    def get_function_at(self, address: int):
        for fn in self.functions:
            if int(fn.start) == int(address):
                return fn
        return None

    def update_analysis_and_wait(self):
        self.analysis_updated = True

    def get_symbols_by_name(self, name: str):
        return [symbol for symbol in self._symbols if getattr(symbol, "name", None) == name]

    def get_symbol_by_raw_name(self, name: str):
        for symbol in self._symbols:
            if getattr(symbol, "raw_name", None) == name:
                return symbol
        return None

    def get_symbols(self):
        return list(self._symbols)

    def get_symbol_at(self, address: int):
        for symbol in self._symbols:
            if int(symbol.address) == int(address):
                return symbol
        return None

    def get_type_by_name(self, name):
        fake_bn = sys.modules["binaryninja"]
        qn_cls = getattr(fake_bn, "QualifiedName", None)
        if qn_cls is not None and isinstance(name, qn_cls):
            # BN keys namespaced types by component tuple, NOT the joined string.
            hit = self._qualified_types.get(tuple(name.name))
            if hit is not None:
                return hit
        return self.types.get(str(name))

    def define_user_type(self, name, type_obj):
        fake_bn = sys.modules["binaryninja"]
        qn_cls = getattr(fake_bn, "QualifiedName", None)
        if qn_cls is not None and isinstance(name, qn_cls):
            self._qualified_types[tuple(name.name)] = type_obj
        else:
            self.types[str(name)] = type_obj

    def get_instruction_length(self, address: int):
        return self._instruction_lengths.get(int(address), 1)

    def get_disassembly(self, address: int, arch=None):
        return self._disassembly.get(int(address), "")

    def get_functions_containing(self, address: int):
        result = []
        for fn in self.functions:
            start = int(fn.start)
            end = start
            for block in getattr(fn, "basic_blocks", []) or []:
                end = max(end, int(block.end))
            if end == start:
                end = start + 1
            if start <= int(address) < end:
                result.append(fn)
        return result

    def get_code_refs(self, address: int):
        return list(self._code_refs.get(int(address), []))

    def get_data_refs(self, address: int):
        return list(self._data_refs.get(int(address), []))

    def get_symbols_of_type(self, sym_type):
        return [s for s in self._symbols if getattr(s, "type", None) == sym_type]

    def get_sections_at(self, address: int):
        result = []
        for sec in self.sections.values():
            if sec.start <= address < sec.end:
                result.append(sec)
        return result

    def get_segment_at(self, address: int):
        return self._segments.get(address)

    def read(self, address: int, length: int):
        if self._memory:
            # Binary Ninja's bv.read returns only the contiguous mapped bytes
            # starting at *address*, or b"" if the address is unmapped.
            for base, blob in self._memory.items():
                if base <= address < base + len(blob):
                    start = address - base
                    return blob[start:start + length]
            return b""
        return b"\x90" * length


class _FakeType:
    def __init__(self, decl: str, *, width: int = 0, members=None, type_class: str = "StructureTypeClass"):
        self._decl = decl
        self.width = width
        self.members = list(members) if members is not None else None
        self.type_class = type_class

    def __str__(self):
        return self._decl


class _FakeMember:
    def __init__(self, offset: int, name: str, type_text: str):
        self.offset = offset
        self.name = name
        self.type = type_text


class _FakeMutationBV(_FakeBV):
    def __init__(self):
        super().__init__()
        self.events: list[tuple[str, str] | str] = []

    def begin_undo_actions(self):
        self.events.append("begin")
        return "state"

    def update_analysis_and_wait(self):
        self.events.append("refresh")

    def revert_undo_actions(self, state):
        self.events.append(("revert", state))

    def commit_undo_actions(self, state):
        self.events.append(("commit", state))


class _ParseResult:
    def __init__(self, *, types=None, variables=None, functions=None):
        self.types = dict(types or {})
        self.variables = dict(variables or {})
        self.functions = dict(functions or {})


def test_resolve_rename_target_rejects_ambiguous_function_identifier(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "duplicate_name"),
            _FakeFunction(0x402000, "duplicate_name"),
        ]
    )

    with pytest.raises(bridge.OperationFailure, match="Ambiguous function identifier"):
        instance._resolve_rename_target(bv, "duplicate_name", "function")


class _FakeSymbol:
    def __init__(self, type_name: str):
        self.type = type("_FakeSymType", (), {"name": type_name})()


def test_find_function_auto_resolves_impl_over_import_stub(monkeypatch):
    """A name shared by a PLT/import stub and the real implementation resolves
    to the IMPLEMENTATION instead of erroring -- the stub is distinguishable by
    symbol.type, so the common collision Just Works with no new CLI surface
    (#122)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    stub = _FakeFunction(0x401030, "send")        # PLT/import trampoline
    stub.symbol = _FakeSymbol("ImportedFunctionSymbol")
    impl = _FakeFunction(0x401500, "send")        # real body
    impl.symbol = _FakeSymbol("FunctionSymbol")
    bv = _FakeBV(functions=[stub, impl])

    fn = instance._find_function(bv, "send")
    assert int(fn.start) == 0x401500

    # the rename resolver shares the same chokepoint -> same auto-resolution
    target = instance._resolve_rename_target(bv, "send", "function")
    assert target["address"] == 0x401500


def test_find_function_stays_ambiguous_for_two_real_bodies_with_kinds(monkeypatch):
    """Two genuine same-named bodies (the A/B-duplicate firmware case) stay
    ambiguous -- auto-pick must NOT guess -- and the error now names each
    candidate's symbol kind so the collision self-documents (#122)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    a = _FakeFunction(0x401000, "dup")
    a.symbol = _FakeSymbol("FunctionSymbol")
    b = _FakeFunction(0x402000, "dup")
    b.symbol = _FakeSymbol("FunctionSymbol")
    bv = _FakeBV(functions=[a, b])

    with pytest.raises(RuntimeError, match="Ambiguous function identifier") as excinfo:
        instance._find_function(bv, "dup")
    assert "[FunctionSymbol]" in str(excinfo.value)


def test_verify_rename_symbol_reports_noop(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x401000, "player_update")])

    result = instance._verify_operation(
        bv,
        {
            "op": "rename_symbol",
            "kind": "function",
            "address": "0x401000",
            "before_name": "player_update",
            "new_name": "player_update",
            "requested": {
                "op": "rename_symbol",
                "identifier": "player_update",
                "new_name": "player_update",
            },
        },
    )

    assert result["status"] == "noop"
    assert result["observed"]["name"] == "player_update"


def test_mutation_reverts_on_verification_failure(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    # _mutation moved to mutation_engine and calls these peers module-locally;
    # patch the seam helper on instance.ctx and the mutation peers on the module.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.mutation_engine, "_guess_affected_functions", lambda ctx, bv, operations: [])
    monkeypatch.setattr(bridge.mutation_engine, "_capture_function_snapshots", lambda ctx, bv, functions: {})
    monkeypatch.setattr(bridge.mutation_engine, "_capture_type_snapshots", lambda ctx, bv, operations: {})
    monkeypatch.setattr(bridge.mutation_engine, "_diff_snapshots", lambda ctx, before, after: [])
    monkeypatch.setattr(bridge.mutation_engine, "_diff_type_snapshots", lambda ctx, before, after: [])
    monkeypatch.setattr(
        bridge.mutation_engine,
        "_apply_operation",
        lambda ctx, bv, op, restores=None: {
            "op": "rename_symbol",
            "kind": "function",
            "address": "0x401000",
            "new_name": "player_update",
            "requested": {"identifier": "sub_401000", "new_name": "player_update"},
        },
    )
    monkeypatch.setattr(
        bridge.mutation_engine,
        "_verify_operation",
        lambda ctx, bv, result: {
            **result,
            "status": "verification_failed",
            "message": "Live rename verification failed at 0x401000",
        },
    )

    result = instance._mutation("active", False, [{"op": "rename_symbol"}])

    assert result["success"] is False
    assert result["committed"] is False
    assert ("revert", "state") in bv.events
    assert ("commit", "state") not in bv.events


def test_run_local_restores_runs_reverse_and_reports_failure(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    order: list[int] = []

    def mk(n, *, fail=False):
        def _restore():
            order.append(n)
            if fail:
                raise RuntimeError("boom")
        return _restore

    settled = []
    bv = types.SimpleNamespace(update_analysis_and_wait=lambda: settled.append(True))

    # A failing restore must not stop the others, and the result is False.
    ok = instance._run_local_restores(bv, [mk(1), mk(2, fail=True), mk(3)])
    assert order == [3, 2, 1]  # reverse of apply order
    assert ok is False
    assert settled == [True]  # view re-settled so the restore materializes

    order.clear()
    settled.clear()
    assert instance._run_local_restores(bv, [mk(1), mk(2)]) is True
    assert order == [2, 1]
    assert settled == [True]

    # Empty restore list is a no-op: no reanalysis triggered.
    settled.clear()
    assert instance._run_local_restores(bv, []) is True
    assert settled == []


def _mutation_with_stubs(monkeypatch, bridge, instance, bv, *, apply, verify=None):
    # _mutation moved to mutation_engine and calls its peers module-locally; patch
    # the seam helper on instance.ctx and the mutation peers on the module. The
    # passed apply/verify keep their (bv, op[, restores]) / (bv, result) signature
    # -- wrap them to drop the new leading ctx the module-level call now passes.
    me = bridge.mutation_engine
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(me, "_guess_affected_functions", lambda ctx, bv, operations: [])
    monkeypatch.setattr(me, "_capture_function_snapshots", lambda ctx, bv, functions: {})
    monkeypatch.setattr(me, "_capture_type_snapshots", lambda ctx, bv, operations: {})
    monkeypatch.setattr(me, "_diff_snapshots", lambda ctx, before, after: [])
    monkeypatch.setattr(me, "_diff_type_snapshots", lambda ctx, before, after: [])
    monkeypatch.setattr(me, "_apply_operation", lambda ctx, *a, **k: apply(*a, **k))
    if verify is not None:
        monkeypatch.setattr(me, "_verify_operation", lambda ctx, *a, **k: verify(*a, **k))


def test_apply_failure_runs_restores_even_when_undo_revert_fails(monkeypatch):
    """An apply failure must run the explicit non-journaled restores even when
    the undo revert fails — `and` short-circuit would skip them and leave
    local/prototype changes applied (#88)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    calls = {"restores": 0}

    def apply(bv_, op, restores=None):
        if op.get("op") == "boom":
            raise bridge.OperationFailure("unsupported", "nope", requested={})
        restores.append(lambda: None)
        return {"op": "local_rename", "requested": {}}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply)
    monkeypatch.setattr(bridge.mutation_engine, "_revert_undo_safely", lambda ctx, bv_, state: False)

    def run_restores(ctx, bv_, restores):
        calls["restores"] += 1
        assert len(restores) == 1
        return True

    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", run_restores)

    result = instance._mutation("active", False, [{"op": "local_rename"}, {"op": "boom"}])

    assert calls["restores"] == 1  # restores ran despite the failed undo revert
    assert result["success"] is False
    assert result["rolled_back"] is False  # undo revert failed, so not fully reverted
    assert "rollback itself failed" in result["message"]


def test_preview_restore_failure_is_not_success(monkeypatch):
    """A preview whose non-journaled restore failed left the view modified;
    it must not report success:true / exit 0 to automation (#88)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None):
        restores.append(lambda: None)
        return {"op": "local_rename", "requested": {}}

    _mutation_with_stubs(
        monkeypatch, bridge, instance, bv,
        apply=apply,
        verify=lambda bv_, result: {**result, "status": "verified"},
    )
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, restores: False)

    result = instance._mutation("active", True, [{"op": "local_rename"}])

    assert result["preview"] is True
    assert result["success"] is False
    assert result["committed"] is False
    assert result["rolled_back"] is False
    assert "failed" in result["message"]
    assert ("revert", "state") in bv.events
    assert ("commit", "state") not in bv.events


def test_preview_drift_restore_failure_is_not_success(monkeypatch):
    """If reverting BN's propagation onto aliased siblings (the var-drift
    restore) fails, the preview left the view modified and must report
    success:false / rolled_back:false (#88)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None):
        return {"op": "local_rename", "requested": {}}

    _mutation_with_stubs(
        monkeypatch, bridge, instance, bv,
        apply=apply,
        verify=lambda bv_, result: {**result, "status": "verified"},
    )
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, restores: True)
    # Force a non-empty var snapshot and a failing drift restore.
    monkeypatch.setattr(bridge.mutation_engine, "_capture_local_var_snapshots", lambda ctx, bv_, fns: {0x1: {1: ("a", "int")}})
    monkeypatch.setattr(bridge.mutation_engine, "_restore_local_var_drift", lambda ctx, bv_, snap: False)

    result = instance._mutation("active", True, [{"op": "local_rename"}])

    assert result["preview"] is True
    assert result["success"] is False
    assert result["rolled_back"] is False


def test_preview_with_successful_restore_still_succeeds(monkeypatch):
    """The restored-success coupling must not regress the normal preview path."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None):
        restores.append(lambda: None)
        return {"op": "local_rename", "requested": {}}

    _mutation_with_stubs(
        monkeypatch, bridge, instance, bv,
        apply=apply,
        verify=lambda bv_, result: {**result, "status": "verified"},
    )
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, restores: True)

    result = instance._mutation("active", True, [{"op": "local_rename"}])

    assert result["preview"] is True
    assert result["success"] is True
    assert result["committed"] is False
    assert result["rolled_back"] is True


def test_rolled_back_sibling_op_reports_reverted_not_unsupported(monkeypatch):
    """When a later op fails mid-batch and the batch is reverted, an op that
    ALREADY succeeded must be reported as 'reverted', not 'unsupported' -- it
    was supported and applied; a sibling failed. 'reverted' is not a failure
    status (#118)."""
    from bn.formatters import FAILED_MUTATION_STATUSES

    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None):
        if op.get("op") == "boom":
            raise bridge.OperationFailure("unsupported", "Function not found: x", requested={})
        return {"op": "rename_symbol", "status": "applied", "requested": {}}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply)
    monkeypatch.setattr(bridge.mutation_engine, "_revert_undo_safely", lambda ctx, bv_, state: True)
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, restores: True)

    result = instance._mutation("active", False, [{"op": "rename_symbol"}, {"op": "boom"}])

    assert result["success"] is False
    assert result["rolled_back"] is True
    statuses = [r["status"] for r in result["results"]]
    assert statuses[0] == "reverted"          # succeeded-then-reverted, honestly
    assert statuses[1] == "unsupported"        # the real failing op keeps its status
    assert "reverted" not in FAILED_MUTATION_STATUSES


def test_rolled_back_sibling_reports_rollback_failed_when_revert_fails(monkeypatch):
    """If the rollback itself fails, a preceding op may STILL be applied -- it
    must not be labeled 'reverted'. Use a distinct failed status so exit codes
    and rendering treat the left-modified view as the failure it is (#118)."""
    from bn.formatters import FAILED_MUTATION_STATUSES

    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def apply(bv_, op, restores=None):
        if op.get("op") == "boom":
            raise bridge.OperationFailure("unsupported", "nope", requested={})
        return {"op": "rename_symbol", "status": "applied", "requested": {}}

    _mutation_with_stubs(monkeypatch, bridge, instance, bv, apply=apply)
    monkeypatch.setattr(bridge.mutation_engine, "_revert_undo_safely", lambda ctx, bv_, state: False)
    monkeypatch.setattr(bridge.mutation_engine, "_run_local_restores", lambda ctx, bv_, restores: True)

    result = instance._mutation("active", False, [{"op": "rename_symbol"}, {"op": "boom"}])

    assert result["success"] is False
    assert result["rolled_back"] is False
    assert result["results"][0]["status"] == "rollback_failed"
    assert "rollback_failed" in FAILED_MUTATION_STATUSES


def test_capture_and_restore_local_var_drift_reverts_propagated_siblings(monkeypatch):
    """BN's create_user_var propagates a user name onto aliased siblings (naming
    a stack var also renames the aliased register). The drift snapshot/restore
    must put EVERY changed local back, not just the targeted one (#88)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    target = _FakeVariable(name="var_8", storage=-8, var_type="int32_t", identifier=10)
    sibling = _FakeVariable(name="r2", storage=2, var_type="int32_t", identifier=20,
                            source_type="RegisterVariableSourceType")
    fn = _FakeFunction(0x11744, "f")
    fn.stack_layout = [target]
    fn.hlil = types.SimpleNamespace(vars=[sibling])
    settled: list[bool] = []

    class _Func(_FakeFunction):
        def create_user_var(self, var, type_obj, name):
            var.name = name
            var.type = type_obj

    fn.__class__ = _Func
    bv = _FakeBV(functions=[fn])
    bv.update_analysis_and_wait = lambda: settled.append(True)

    before = instance._capture_local_var_snapshots(bv, [fn])
    assert before[0x11744][10] == ("var_8", "int32_t")
    assert before[0x11744][20] == ("r2", "int32_t")

    # Simulate apply + propagation: target renamed AND sibling auto-renamed.
    target.name = "Q8"
    sibling.name = "Q8_1"

    ok = instance._restore_local_var_drift(bv, before)
    assert ok is True
    assert target.name == "var_8"   # targeted var restored
    assert sibling.name == "r2"     # propagated sibling restored too
    assert settled == [True]        # reanalyzed exactly once (something drifted)


def test_restore_local_var_drift_noop_when_nothing_changed(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    var = _FakeVariable(name="keep", storage=-8, var_type="int32_t", identifier=10)
    fn = _FakeFunction(0x11744, "f")
    fn.stack_layout = [var]
    settled: list[bool] = []
    bv = _FakeBV(functions=[fn])
    bv.update_analysis_and_wait = lambda: settled.append(True)

    before = instance._capture_local_var_snapshots(bv, [fn])
    assert instance._restore_local_var_drift(bv, before) is True
    assert settled == []  # nothing drifted -> no reanalysis


def test_restore_local_var_drift_reports_failure_on_missing_function(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # Snapshot references a function the view can no longer resolve.
    snapshots = {0xdead: {10: ("v", "int32_t")}}
    bv = _FakeBV(functions=[])
    assert instance._restore_local_var_drift(bv, snapshots) is False


def test_op_local_rename_registers_restore_that_undoes_the_rename(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    var = _FakeVariable(
        name="r2_1", storage=2, var_type="uint32_t", identifier=123,
        source_type="RegisterVariableSourceType",
    )

    class _RecordingFunc:
        def __init__(self):
            self.start = 0x1000
            self.name = "f"
            self.calls: list[tuple] = []

        def create_user_var(self, v, type_obj, name):
            self.calls.append((name, str(type_obj)))
            v.name = name
            v.type = type_obj

    fn = _RecordingFunc()
    bv = types.SimpleNamespace(get_function_at=lambda addr: fn)
    # _op_local_rename moved to mutation_engine; it resolves _find_function via
    # the seam (instance.ctx), the variable helpers via vars_mod, and
    # _find_var_for_restore module-locally (now ctx-first).
    monkeypatch.setattr(instance.ctx, "_find_function", lambda _bv, ident: fn)
    monkeypatch.setattr(bridge.vars_mod, "_find_variable_selector", lambda _f, sel: (var, False))
    monkeypatch.setattr(bridge.mutation_engine, "_find_var_for_restore", lambda ctx, _f, identifier, storage, is_parameter: var)
    monkeypatch.setattr(bridge.vars_mod, "_local_id", lambda _f, _v, is_parameter: "lid")

    restores: list = []
    result = instance._op_local_rename(
        bv, {"op": "local_rename", "function": "f", "variable": "r2_1", "new_name": "tbl_count"}, restores
    )

    # before_name is the OLD name, the rename applied, and a restore was registered.
    assert result["before_name"] == "r2_1"
    assert result["new_name"] == "tbl_count"
    assert var.name == "tbl_count"
    assert len(restores) == 1

    # Replaying the restore puts the local back to its original name+type.
    restores[0]()
    assert var.name == "r2_1"
    assert str(var.type) == "uint32_t"
    assert fn.calls[-1] == ("r2_1", "uint32_t")


def test_op_local_rename_noop_registers_no_restore(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    var = _FakeVariable(name="keep", storage=2, var_type="int32_t", identifier=1)
    fn = types.SimpleNamespace(start=0x1000, name="f", create_user_var=lambda *a: None)
    monkeypatch.setattr(instance.ctx, "_find_function", lambda _bv, ident: fn)
    monkeypatch.setattr(bridge.vars_mod, "_find_variable_selector", lambda _f, sel: (var, False))
    monkeypatch.setattr(bridge.vars_mod, "_local_id", lambda _f, _v, is_parameter: "lid")

    restores: list = []
    instance._op_local_rename(bv := object(), {"op": "local_rename", "function": "f", "variable": "keep", "new_name": "keep"}, restores)
    assert restores == []  # renaming to the same name mutates nothing, so nothing to revert


def test_refresh_updates_analysis_and_returns_target_info(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance, "_target_info", lambda selector: {"selector": "SnailMail_unwrapped.exe.bndb"})

    result = instance._refresh("active")

    assert result["refreshed"] is True
    assert result["target"]["selector"] == "SnailMail_unwrapped.exe.bndb"
    assert "refresh" in bv.events


def test_parse_declaration_source_uses_platform_parser_with_source_path(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    recorded = {}

    class _Platform:
        def parse_types_from_source(self, source, **kwargs):
            recorded["source"] = source
            recorded["kwargs"] = kwargs
            return _ParseResult(types={"Player": "struct Player"})

    class _SourceBV(_FakeBV):
        def __init__(self):
            super().__init__()
            self.platform = _Platform()

        def parse_types_from_string(self, declaration):
            raise AssertionError("string parser should not be used when source parsing succeeds")

    header_path = tmp_path / "win32_min.h"
    header_path.write_text("typedef struct Player { int hp; } Player;", encoding="utf-8")
    bv = _SourceBV()

    parsed = instance._parse_declaration_source(bv, header_path.read_text(encoding="utf-8"), source_path=str(header_path))

    assert [name for name, _ in parsed["types"]] == ["Player"]
    assert recorded["kwargs"]["filename"] == str(header_path)
    assert recorded["kwargs"]["include_dirs"] == [str(header_path.parent.resolve())]


def test_op_types_declare_accepts_source_without_named_types(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _Platform:
        def parse_types_from_source(self, source, **kwargs):
            return _ParseResult(
                functions={"DirectInput8Create": "int32_t(void)"},
                variables={"GUID_SysKeyboard": "GUID"},
            )

    class _SourceOnlyBV(_FakeBV):
        def __init__(self):
            super().__init__()
            self.platform = _Platform()
            self.defined: list[tuple[str, str]] = []

        def parse_types_from_string(self, declaration):
            raise AssertionError("string parser should not be used when source parsing succeeds")

        def get_type_by_name(self, name):
            return None

        def define_user_type(self, name, type_obj):
            self.defined.append((name, type_obj))

    bv = _SourceOnlyBV()

    result = instance._op_types_declare(
        bv,
        {
            "op": "types_declare",
            "declaration": "extern const GUID GUID_SysKeyboard;",
            "source_path": "/tmp/win32_min.h",
        },
    )

    assert result["count"] == 0
    assert result["defined_types"] == {}
    assert result["parsed_functions"] == ["DirectInput8Create"]
    assert result["parsed_variables"] == ["GUID_SysKeyboard"]
    assert bv.defined == []


def test_op_types_declare_uses_canonical_defined_type_text(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    raw_type = _FakeType(
        "struct",
        width=0x2C,
        members=[
            _FakeMember(0x0, "state", "uint32_t"),
            _FakeMember(0x10, "transition_progress", "float"),
        ],
    )

    class _Platform:
        def parse_types_from_source(self, source, **kwargs):
            return _ParseResult(types={"DamageGaugeController": raw_type})

    class _CanonicalizingBV(_FakeBV):
        def __init__(self):
            super().__init__()
            self.platform = _Platform()

        def parse_types_from_string(self, declaration):
            raise AssertionError("string parser should not be used when source parsing succeeds")

        def define_user_type(self, name, type_obj):
            canonical = _FakeType(
                f"struct {name}",
                width=type_obj.width,
                members=getattr(type_obj, "members", None),
            )
            super().define_user_type(name, canonical)

    bv = _CanonicalizingBV()

    result = instance._op_types_declare(
        bv,
        {
            "op": "types_declare",
            "declaration": "struct DamageGaugeController { int state; };",
            "source_path": "/tmp/controller.h",
        },
    )

    assert result["defined_types"] == {"DamageGaugeController": "struct DamageGaugeController"}
    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert verified["observed"]["defined_types"]["DamageGaugeController"] == "struct DamageGaugeController"


def test_op_set_prototype_uses_string_user_type_for_bn_compat(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _SetterFunction(_FakeFunction):
        def __init__(self):
            super().__init__(0x43F200, "update_garbage_hazard", "void* __fastcall(void* arg1)")
            self.user_type_calls = []

        def set_user_type(self, value):
            self.user_type_calls.append(value)
            if isinstance(value, str):
                self.type = value

    class _PrototypeBV(_FakeBV):
        def parse_type_string(self, declaration):
            return _FakeType("void* __thiscall(struct GarbageHazardRuntime* self)", type_class="FunctionTypeClass"), None

    fn = _SetterFunction()
    bv = _PrototypeBV(functions=[fn])

    result = instance._op_set_prototype(
        bv,
        {
            "op": "set_prototype",
            "identifier": "update_garbage_hazard",
            "prototype": "void* __thiscall update_garbage_hazard(struct GarbageHazardRuntime* self)",
        },
    )

    assert fn.user_type_calls == ["void* __thiscall(struct GarbageHazardRuntime* self)"]
    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert verified["observed"]["prototype"] == "void* __thiscall(struct GarbageHazardRuntime* self)"


def test_apply_operation_user_error_message_has_no_class_name(monkeypatch):
    """A handler raising a user-facing RuntimeError (e.g. a mistyped function
    name -> 'Function not found') must surface a clean, actionable message --
    not 'unsupported: RuntimeError: ...' that reads like an internal crash
    (#122)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    me = bridge.mutation_engine

    def boom_user(ctx, bv, op):
        raise RuntimeError("Function not found: ghost")

    monkeypatch.setattr(me, "_op_set_comment", boom_user)
    bv = _FakeBV()

    with pytest.raises(bridge.OperationFailure) as excinfo:
        instance._apply_operation(bv, {"op": "set_comment", "comment": "x", "function": "ghost"})

    assert excinfo.value.status == "unsupported"
    assert excinfo.value.message == "Function not found: ghost"
    assert "RuntimeError" not in excinfo.value.message


def test_apply_operation_unexpected_error_gets_internal_error_status(monkeypatch):
    """A genuinely UNEXPECTED internal error gets the distinct 'internal_error'
    status (kept in FAILED_MUTATION_STATUSES so exit codes still flag it) and
    keeps the class name for debugging -- not the misleading 'unsupported'
    (#122)."""
    from bn.formatters import FAILED_MUTATION_STATUSES

    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    me = bridge.mutation_engine

    def boom_internal(ctx, bv, op):
        raise KeyError("unexpected")

    monkeypatch.setattr(me, "_op_set_comment", boom_internal)
    bv = _FakeBV()

    with pytest.raises(bridge.OperationFailure) as excinfo:
        instance._apply_operation(bv, {"op": "set_comment", "comment": "x", "function": "g"})

    assert excinfo.value.status == "internal_error"
    assert "KeyError" in excinfo.value.message
    assert "internal_error" in FAILED_MUTATION_STATUSES


def test_types_declare_malformed_declaration_is_clean_invalid_request(monkeypatch):
    """A malformed C declaration (the top `types declare` user mistake) raises a
    built-in SyntaxError from BN's parser -- which is NOT a RuntimeError/
    ValueError. It must surface as a clean invalid_request, not a leaked
    'SyntaxError:' class name or 'internal_error' (#122)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _BadDeclBV(_FakeBV):
        def parse_types_from_string(self, declaration):
            raise SyntaxError("error: input:1:1 expected unqualified-id")

    bv = _BadDeclBV()
    with pytest.raises(bridge.OperationFailure) as excinfo:
        instance._apply_operation(bv, {"op": "types_declare", "declaration": "this is not valid C"})

    assert excinfo.value.status == "invalid_request"
    assert "could not parse declaration" in excinfo.value.message
    assert "SyntaxError" not in excinfo.value.message


def test_mutation_malformed_types_declare_reports_clean_failure_not_escape(monkeypatch):
    """End-to-end: a malformed types_declare must flow through the mutation
    machinery as a clean, reverted invalid_request -- it must NOT escape the
    pre-apply snapshot pass as a raw SyntaxError out of _mutation (#122)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _BadDeclMutationBV(_FakeMutationBV):
        def parse_types_from_string(self, declaration):
            raise SyntaxError("error: input:1:1 expected unqualified-id")

    bv = _BadDeclMutationBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.mutation_engine, "_capture_function_snapshots", lambda ctx, bv_, fns: {})
    monkeypatch.setattr(bridge.mutation_engine, "_diff_snapshots", lambda ctx, b, a: [])

    result = instance._mutation("active", False, [{"op": "types_declare", "declaration": "garbage @#$"}])

    assert result["success"] is False
    statuses = [r.get("status") for r in result["results"]]
    assert "invalid_request" in statuses
    joined = " ".join(r.get("message", "") for r in result["results"])
    assert "SyntaxError" not in joined


def test_op_set_prototype_hints_to_declare_unknown_struct(monkeypatch):
    """A prototype that references a not-yet-defined struct makes
    parse_type_string fail. Surface a clear invalid_request that hints to
    declare the type first -- not a raw 'unsupported: SyntaxError: ...' that
    leaks the Python exception class and gives no next step (#122)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _UnknownTypeBV(_FakeBV):
        def parse_type_string(self, declaration):
            raise SyntaxError("unexpected token 'GarbageHazardRuntime'")

    fn = _FakeFunction(0x401000, "handler")
    bv = _UnknownTypeBV(functions=[fn])

    with pytest.raises(bridge.OperationFailure) as excinfo:
        instance._op_set_prototype(
            bv,
            {
                "op": "set_prototype",
                "identifier": "handler",
                "prototype": "int handler(struct GarbageHazardRuntime* self)",
            },
        )

    assert excinfo.value.status == "invalid_request"
    message = excinfo.value.message.lower()
    assert "declare" in message            # actionable next step
    assert "syntaxerror" not in message    # raw exception class must not leak


def test_parse_type_or_hint_shared_by_all_type_ops(monkeypatch):
    """set_prototype, local_retype, and struct_field_set all route their
    bv.parse_type_string through this helper, so an undefined-type reference
    yields a clean invalid_request + correct 'bn types declare' hint instead of
    a leaked exception class or BN's multi-line parser text (#122)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    me = bridge.mutation_engine

    class _BadParseBV(_FakeBV):
        def parse_type_string(self, decl):
            raise SyntaxError("error: <unknown>: Reference to unknown type Foo\n1 error generated.")

    with pytest.raises(bridge.OperationFailure) as excinfo:
        me._parse_type_or_hint(instance.ctx, _BadParseBV(), {"op": "local_retype"}, "struct Foo*", label="type")

    msg = excinfo.value.message
    assert excinfo.value.status == "invalid_request"
    assert "bn types declare" in msg        # correct command spelling (not `type`)
    assert "declare it first" in msg
    assert "syntaxerror" not in msg.lower()  # no raw Python exception class
    assert "\n" not in msg                   # BN's multi-line parser text collapsed


def test_split_qualified_name_is_bracket_depth_aware(monkeypatch):
    """The ::-split for namespaced lookups must split only at bracket depth 0, so
    template arguments are not torn apart (#200)."""
    me = _load_bridge(monkeypatch).mutation_engine
    assert me._split_qualified_name("ns::demo::Foo") == ["ns", "demo", "Foo"]
    assert me._split_qualified_name("Foo") == ["Foo"]
    # '::' inside template args must NOT split
    assert me._split_qualified_name("__alloc_traits<std::allocator<char> >::pointer") == [
        "__alloc_traits<std::allocator<char> >",
        "pointer",
    ]
    # the leading 'std::' IS a top-level separator; only the '::' INSIDE the
    # template args must be preserved.
    assert me._split_qualified_name("std::vector<std::pair<int, long> >::iterator") == [
        "std",
        "vector<std::pair<int, long> >",
        "iterator",
    ]


def test_parse_type_or_hint_resolves_namespaced_user_type(monkeypatch):
    """BN's C type-string parser rejects a ::-qualified user type even when it is
    defined. local retype / field type should fall back to resolving it via a
    multi-component QualifiedName lookup (BN does NOT match the raw "::"-string,
    so a naive get_type_by_name(string) misses it) and build a name-preserving
    pointer, so a C++ class type applies without a flat-name alias (#200)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    me = bridge.mutation_engine

    class _NsBV(_FakeBV):
        def parse_type_string(self, decl):
            # BN rejects the namespaced name outright.
            raise SyntaxError(
                "error: <unknown>:1:1 use of undeclared identifier 'ns'\n1 error generated."
            )

    # Registered the way BN registers a recovered namespaced type: under the
    # component tuple, NOT the raw "::"-joined string. A raw-string lookup misses
    # it (that is the bug the fix must survive); only the QualifiedName path hits.
    bv = _NsBV(
        functions=[],
        qualified_types_={("ns", "demo", "Foo"): _FakeType("struct ns::demo::Foo")},
    )
    # guard: a raw-string get_type_by_name MUST miss (mirrors real BN)
    assert bv.get_type_by_name("ns::demo::Foo") is None

    # pointer to a ::-qualified type resolves via the QualifiedName fallback. The
    # named type keeps its `struct` tag (matching BN's readback, so verify passes).
    t, name = me._parse_type_or_hint(
        instance.ctx, bv, {"op": "local_retype"}, "ns::demo::Foo*", label="type"
    )
    assert str(t) == "struct ns::demo::Foo*"
    assert name is None

    # the bare ::-qualified type (no pointer) resolves too
    t2, _ = me._parse_type_or_hint(
        instance.ctx, bv, {"op": "local_retype"}, "ns::demo::Foo", label="type"
    )
    assert str(t2) == "struct ns::demo::Foo"

    # double-pointer too
    t3, _ = me._parse_type_or_hint(
        instance.ctx, bv, {"op": "local_retype"}, "ns::demo::Foo **", label="type"
    )
    assert str(t3) == "struct ns::demo::Foo**"

    # a name that is NOT a known type still raises the actionable declare hint
    with pytest.raises(bridge.OperationFailure) as excinfo:
        me._parse_type_or_hint(
            instance.ctx, bv, {"op": "local_retype"}, "ns::demo::Unknown*", label="type"
        )
    assert excinfo.value.status == "invalid_request"
    assert "declare it first" in excinfo.value.message


def test_batch_struct_field_accepts_type_name_alias(monkeypatch):
    """A struct_field_* batch op may use `type_name` (the key the output /
    affected_types surface uses, and an analyst's natural reflex) as an alias
    for the canonical `struct_name`, instead of failing validation with
    'missing required field struct_name' (M12)."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine

    # the alias is normalized in place for every struct_field_* kind
    for kind in ("struct_field_set", "struct_field_rename", "struct_field_delete"):
        op = {"op": kind, "type_name": "Elf64_Sym"}
        me._normalize_struct_alias(op)
        assert op["struct_name"] == "Elf64_Sym", kind

    # an explicit struct_name always wins (alias never clobbers it)
    op = {"op": "struct_field_rename", "struct_name": "A", "type_name": "B"}
    me._normalize_struct_alias(op)
    assert op["struct_name"] == "A"

    # non-struct ops are left untouched
    op = {"op": "rename_symbol", "type_name": "X"}
    me._normalize_struct_alias(op)
    assert "struct_name" not in op

    # end-to-end through _apply_operation: validation no longer rejects a
    # type_name-only struct op, and the handler receives struct_name
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[])
    seen = {}

    def _stub_rename(ctx, bv_, op_):
        seen["op"] = op_
        return {"status": "verified"}

    monkeypatch.setattr(me, "_op_struct_field_rename", _stub_rename)
    result = me._apply_operation(
        instance.ctx, bv,
        {"op": "struct_field_rename", "type_name": "Elf64_Sym",
         "old_name": "st_info", "new_name": "sym_info"},
    )
    assert result == {"status": "verified"}
    assert seen["op"]["struct_name"] == "Elf64_Sym"


# --- batch field/locator parity audit (#173) -------------------------------
#
# Per-op parity table: for each of the 10 batch ops, the fields the batch
# validator constrains and the interactive command whose field set it must
# match. `required`/`one_of`/`enum` mirror REQUIRED_FIELDS / REQUIRED_ONE_OF /
# ENUM_FIELDS in mutation_engine; `test_batch_op_parity_table_locked_to_validators`
# asserts they stay in lock-step so the table can't silently drift from the code.
# `optional` lists handler-read op.get() fields (documented, not asserted).
#
#   op kind             | required                                  | one_of               | enum                          | optional            | interactive command
#   --------------------|-------------------------------------------|----------------------|-------------------------------|---------------------|--------------------
#   rename_symbol       | identifier, new_name                      | --                   | kind: auto/function/data      | kind                | bn rename / symbol rename
#   set_comment         | comment                                   | function|address     | --                            | --                  | bn comment set
#   delete_comment      | --                                        | function|address     | --                            | --                  | bn comment delete
#   set_prototype       | identifier, prototype                     | --                   | --                            | --                  | bn proto set
#   local_rename        | function, variable, new_name              | --                   | --                            | --                  | bn local rename
#   local_retype        | function, variable, new_type              | --                   | --                            | --                  | bn local retype
#   struct_field_set    | struct_name, field_type, offset, field_name | --                 | --                            | overwrite_existing, type_name | bn struct field set
#   struct_field_rename | struct_name, old_name, new_name           | --                   | --                            | type_name           | bn struct field rename
#   struct_field_delete | struct_name, field_name                   | --                   | --                            | type_name           | bn struct field delete
#   types_declare       | declaration                               | --                   | --                            | source_path         | bn types declare
_BATCH_OP_PARITY = {
    "rename_symbol":       {"required": ("identifier", "new_name"),                         "one_of": (),                           "enum": {"kind": ("auto", "function", "data")}, "cli": "bn rename"},
    "set_comment":         {"required": ("comment",),                                        "one_of": (("function", "address"),),   "enum": {},                                     "cli": "bn comment set"},
    "delete_comment":      {"required": (),                                                  "one_of": (("function", "address"),),   "enum": {},                                     "cli": "bn comment delete"},
    "set_prototype":       {"required": ("identifier", "prototype"),                         "one_of": (),                           "enum": {},                                     "cli": "bn proto set"},
    "local_rename":        {"required": ("function", "variable", "new_name"),                "one_of": (),                           "enum": {},                                     "cli": "bn local rename"},
    "local_retype":        {"required": ("function", "variable", "new_type"),                "one_of": (),                           "enum": {},                                     "cli": "bn local retype"},
    "struct_field_set":    {"required": ("struct_name", "field_type", "offset", "field_name"), "one_of": (),                         "enum": {},                                     "cli": "bn struct field set"},
    "struct_field_rename": {"required": ("struct_name", "old_name", "new_name"),             "one_of": (),                           "enum": {},                                     "cli": "bn struct field rename"},
    "struct_field_delete": {"required": ("struct_name", "field_name"),                       "one_of": (),                           "enum": {},                                     "cli": "bn struct field delete"},
    "types_declare":       {"required": ("declaration",),                                    "one_of": (),                           "enum": {},                                     "cli": "bn types declare"},
}


def _minimal_valid_op(kind):
    """Build the smallest batch op of *kind* that passes _apply_operation's field
    validation: every required field, one field from each one-of group, and a
    valid value for each enum field. Values only need to be present (validation
    checks presence/membership, not resolvability), so a downstream handler may
    still fail to resolve them -- the missing-field tests delete one field at a
    time and assert the rejection comes from validation, before dispatch."""
    op = {"op": kind}
    for field in _BATCH_OP_PARITY[kind]["required"]:
        op[field] = "0x0" if field == "offset" else "x"
    for group in _BATCH_OP_PARITY[kind]["one_of"]:
        op[group[0]] = "0x1000" if "address" in group else "x"
    for field, allowed in _BATCH_OP_PARITY[kind]["enum"].items():
        op[field] = allowed[0]
    return op


def test_batch_op_parity_table_locked_to_validators(monkeypatch):
    """The checked-in parity table must match the live validators exactly, and
    cover every dispatched op kind -- so the audit can't rot as ops are added or
    their field sets change (#173)."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine

    for kind, row in _BATCH_OP_PARITY.items():
        assert me.REQUIRED_FIELDS.get(kind, ()) == row["required"], kind
        assert me.REQUIRED_ONE_OF.get(kind, ()) == row["one_of"], kind
        assert me.ENUM_FIELDS.get(kind, {}) == row["enum"], kind

    # No validator names an op the table forgot, and vice versa.
    assert set(me.REQUIRED_FIELDS) == set(_BATCH_OP_PARITY)
    assert set(me.REQUIRED_ONE_OF) <= set(_BATCH_OP_PARITY)
    assert set(me.ENUM_FIELDS) <= set(_BATCH_OP_PARITY)


def test_batch_missing_required_field_rejected_per_op(monkeypatch):
    """Dropping any single required field from any of the 10 ops yields a clean
    invalid_request that NAMES the field -- never a raw KeyError mislabeled
    internal_error/unsupported from a handler hard-reading op[field] (#173)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()

    for kind, row in _BATCH_OP_PARITY.items():
        for field in row["required"]:
            op = _minimal_valid_op(kind)
            del op[field]
            with pytest.raises(bridge.OperationFailure) as exc:
                instance._apply_operation(bv, op)
            assert exc.value.status == "invalid_request", (kind, field)
            assert field in exc.value.message, (kind, field)


def test_batch_comment_ops_require_one_locator(monkeypatch):
    """set_comment / delete_comment target a function OR an address; a manifest
    op with neither is rejected with a clear invalid_request, not silently
    no-op'd against address 0 (#173, locator parity with #67/#94)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()

    for op in (
        {"op": "set_comment", "comment": "hi"},   # no function, no address
        {"op": "delete_comment"},                 # no function, no address
    ):
        with pytest.raises(bridge.OperationFailure) as exc:
            instance._apply_operation(bv, op)
        assert exc.value.status == "invalid_request"
        assert "one of" in exc.value.message
        assert "function" in exc.value.message and "address" in exc.value.message


def test_batch_comment_ops_reject_both_locators(monkeypatch):
    """function and address target DIFFERENT locations; passing both is
    ambiguous and rejected, rather than silently honoring one and dropping the
    other (#94 parity, now asserted for the batch path) (#173)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()

    for op in (
        {"op": "set_comment", "comment": "hi", "function": "f", "address": "0x1000"},
        {"op": "delete_comment", "function": "f", "address": "0x1000"},
    ):
        with pytest.raises(bridge.OperationFailure) as exc:
            instance._apply_operation(bv, op)
        assert exc.value.status == "invalid_request"
        assert "not both" in exc.value.message


def test_batch_rename_rejects_invalid_kind(monkeypatch):
    """An out-of-set rename `kind` must be rejected the way interactive
    `bn rename --kind` rejects it via argparse choices. The batch path has no
    argparse layer, and the unguarded handler SILENTLY treated an unknown kind
    as a (failing) data-symbol lookup -- so `kind: "garbage"` against a function
    that plainly exists produced a misleading "Symbol not found" instead of a
    clear "kind must be one of ..." (#173)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x1000, "foo")])

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._apply_operation(
            bv, {"op": "rename_symbol", "identifier": "foo", "new_name": "bar", "kind": "garbage"}
        )
    assert exc.value.status == "invalid_request"
    msg = exc.value.message
    assert "kind" in msg and "garbage" in msg
    assert "auto" in msg and "function" in msg and "data" in msg

    # The guard must NOT over-reject a valid kind: kind="function" still resolves
    # (here a noop, since new_name == current name) without raising.
    result = instance._apply_operation(
        bv, {"op": "rename_symbol", "identifier": "foo", "new_name": "foo", "kind": "function"}
    )
    assert result["op"] == "rename_symbol"


class _FakeCommentMutationBV(_FakeMutationBV):
    """Records begin/revert/commit (via _FakeMutationBV) and stores comments so a
    batch can apply one then have the whole batch reverted."""

    def __init__(self):
        super().__init__()
        self.comments: dict[int, str] = {}

    def get_comment_at(self, address):
        return self.comments.get(int(address), "")

    def set_comment_at(self, address, comment):
        if comment is None:
            self.comments.pop(int(address), None)
        else:
            self.comments[int(address)] = comment


def test_batch_invalid_op_rolls_back_prior_applied_op(monkeypatch):
    """A manifest whose 2nd op is malformed (missing a required field) fails the
    WHOLE batch: the 1st op's already-applied change is reverted (no partial
    apply -- the undo state is reverted, never committed) and the failing op is
    reported as a clean invalid_request (#173)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeCommentMutationBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._mutation(
        "active",
        False,
        [
            {"op": "set_comment", "address": "0x1000", "comment": "first op applied"},
            {"op": "set_comment", "address": "0x2000"},  # missing required 'comment'
        ],
    )

    assert result["success"] is False
    assert result["committed"] is False
    assert result["rolled_back"] is True
    assert ("revert", "state") in bv.events
    assert ("commit", "state") not in bv.events

    statuses = [r.get("status") for r in result["results"]]
    assert statuses[0] == "reverted"          # 1st op applied, then rolled back
    assert statuses[1] == "invalid_request"   # 2nd op rejected before apply
    assert "comment" in result["results"][1].get("message", "")


def test_preview_diff_truncated_to_stay_inline(monkeypatch):
    """A previewed mutation's per-function `diff` is capped so a single rename /
    proto preview on a large function stays inline instead of tripping the 10k
    spill threshold; the full diff stays available via --out. (M14)"""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine

    # a short diff passes through untouched
    short = "\n".join(f"line {i}" for i in range(10))
    assert me._truncate_preview_diff(short) == short

    # a long diff is capped to max_lines + a marker pointing at --out
    long = "\n".join(f"line {i}" for i in range(me.PREVIEW_DIFF_MAX_LINES + 500))
    out = me._truncate_preview_diff(long)
    body, _, marker = out.rpartition("\n")
    assert len(body.splitlines()) == me.PREVIEW_DIFF_MAX_LINES
    assert "diff truncated" in marker and "500 more" in marker and "--out" in marker

    # integration: a whole-body change yields a bounded diff, changed=True, and
    # the focused snippet excerpt is still present for the glance
    ctx = bridge.BinaryNinjaBridge().ctx
    big_before = "\n".join(f"old {i}" for i in range(2000))
    big_after = "\n".join(f"new {i}" for i in range(2000))
    diffs = me._diff_snapshots(
        ctx,
        {0x1000: {"text": big_before, "name": "f"}},
        {0x1000: {"text": big_after, "name": "f"}},
    )
    d = diffs[0]
    assert d["changed"] is True
    assert len(d["diff"].splitlines()) <= me.PREVIEW_DIFF_MAX_LINES + 1
    assert "before_excerpt" in d


def test_evidence_mlil_drops_clobber_lhs(monkeypatch):
    """evidence function's per-call `mlil` shows the call + its inputs, not the
    full caller-saved clobber set BN renders as the assignment LHS. (E17)"""
    bridge = _load_bridge(monkeypatch)
    render = bridge.read_evidence._mlil_call_text
    clobber = "arg1, arg2, x2, x3, lr, v0, v31 = call(0x471d60, arg1, arg2, x2, stack = &fp)"
    assert render(clobber) == "call(0x471d60, arg1, arg2, x2, stack = &fp)"
    assert render("call(0x401000, arg1)") == "call(0x401000, arg1)"  # no output: unchanged
    assert render(None) is None

    class _M:
        def __str__(self):
            return clobber

    assert render(_M()) == "call(0x471d60, arg1, arg2, x2, stack = &fp)"


def test_stack_var_span_annotation(monkeypatch):
    """Stack vars carry span_to_next (bytes to the next stack slot = the slot's
    capacity); register/flag locals (non-negative storage) do not. (F20)"""
    bridge = _load_bridge(monkeypatch)
    entries = [
        {"name": "buf", "storage": -1016},
        {"name": "x", "storage": -8},
        {"name": "reg", "storage": 53},     # register var -> no span
        {"name": "buf2", "storage": -1024},
    ]
    bridge.vars_mod._annotate_stack_spans(entries)
    by = {e["name"]: e for e in entries}
    # sorted stack: -1024 (buf2) -> -1016 (buf) -> -8 (x) -> 0 (frame base)
    assert by["buf2"]["span_to_next"] == 8       # -1016 - (-1024)
    assert by["buf"]["span_to_next"] == 1008     # -8 - (-1016)
    assert by["x"]["span_to_next"] == 8          # 0 - (-8)
    assert "span_to_next" not in by["reg"]       # register var untouched


def test_segment_entries_for_verbose_target_info(monkeypatch):
    """target info --verbose builds a segment map with r/w/x flags + length;
    a bv with no segments yields an empty list, not an error. (F21)"""
    bridge = _load_bridge(monkeypatch)
    seg = type("S", (), {"start": 0x1000, "end": 0x2000,
                         "readable": True, "writable": False, "executable": True})()
    bv = type("BV", (), {"segments": [seg]})()
    assert bridge._segment_entries(bv) == [{
        "start": "0x1000", "end": "0x2000", "length": 0x1000,
        "readable": True, "writable": False, "executable": True,
    }]
    assert bridge._segment_entries(type("BV", (), {})()) == []


def test_function_name_summary_counts_named_vs_auto(monkeypatch):
    """target info needs a function-count summary every agent reaches for.
    Auto-named functions are BN's sub_<addr> / j_sub_<addr> defaults; named are
    everything else EXCEPT import/extern stubs (whose names come from
    relocations), which get their own bucket so they don't inflate "named" on a
    stripped binary (#122)."""
    bridge = _load_bridge(monkeypatch)

    class _Sym:
        def __init__(self, type_name):
            self.type = type("_SymType", (), {"name": type_name})()

    imported = _FakeFunction(0x401400, "puts")
    imported.symbol = _Sym("ImportedFunctionSymbol")

    bv = _FakeBV(functions=[
        _FakeFunction(0x401000, "main"),
        _FakeFunction(0x401100, "parse_header"),
        _FakeFunction(0x401200, "sub_401200"),
        _FakeFunction(0x401300, "j_sub_401300"),
        imported,
    ])

    summary = bridge._function_name_summary(bv)

    assert summary["function_count"] == 5
    assert summary["named_function_count"] == 2       # main, parse_header
    assert summary["unnamed_function_count"] == 2      # sub_401200, j_sub_401300
    assert summary["imported_function_count"] == 1     # puts (PLT stub), not "named"


def test_op_set_prototype_registers_restore_for_preview(monkeypatch):
    # set_user_type is NOT journaled by BN's undo buffer, so --preview/rollback
    # must register an explicit restore that puts the prototype back, else the
    # previewed prototype silently persists in the view (#51). The restore uses
    # the .type property setter (a clean revert, no convention re-pinning).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _SetterFunction(_FakeFunction):
        def __init__(self):
            super().__init__(0x1000, "f", "int32_t(int32_t* arg1)")

        def set_user_type(self, value):
            self.type = value if isinstance(value, str) else str(value)

    class _PrototypeBV(_FakeBV):
        def parse_type_string(self, declaration):
            return _FakeType("void(uint32_t* p)", type_class="FunctionTypeClass"), None

    fn = _SetterFunction()
    bv = _PrototypeBV(functions=[fn])
    baseline = fn.type
    restores: list = []
    instance._op_set_prototype(
        bv, {"op": "set_prototype", "identifier": "f", "prototype": "void f(uint32_t* p)"}, restores
    )
    # the prototype was applied ...
    assert fn.type == "void(uint32_t* p)"
    # ... and exactly one restore was registered for the preview/rollback path
    assert len(restores) == 1
    # running it (as the preview path does) puts the original prototype back
    restores[0]()
    assert fn.type == baseline


def test_op_set_prototype_no_restore_when_unchanged(monkeypatch):
    # Setting the same prototype mutates nothing, so no restore is queued (the
    # revert path stays a true no-op).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _SetterFunction(_FakeFunction):
        def __init__(self):
            super().__init__(0x1000, "f", "int32_t(int32_t* arg1)")

        def set_user_type(self, value):
            self.type = value if isinstance(value, str) else str(value)

    class _PrototypeBV(_FakeBV):
        def parse_type_string(self, declaration):
            return _FakeType("int32_t(int32_t* arg1)", type_class="FunctionTypeClass"), None

    fn = _SetterFunction()
    bv = _PrototypeBV(functions=[fn])
    restores: list = []
    instance._op_set_prototype(
        bv, {"op": "set_prototype", "identifier": "f", "prototype": "int32_t f(int32_t* arg1)"}, restores
    )
    assert restores == []


def test_resolve_type_field_accepts_offset_and_suggests_near_match(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        types_={
            "Player": _FakeType(
                "struct Player",
                width=0x5000,
                members=[
                    _FakeMember(0x380, "player_slot", "uint32_t"),
                    _FakeMember(0x4340, "visible_life_stock", "uint32_t"),
                ],
            )
        }
    )

    by_offset = instance._resolve_type_field(bv, "Player.0x4340")
    assert by_offset["field_name"] == "visible_life_stock"
    assert by_offset["offset"] == 0x4340

    by_case = instance._resolve_type_field(bv, "Player.Visible_Life_Stock")
    assert by_case["field_name"] == "visible_life_stock"

    with pytest.raises(RuntimeError, match=r"Did you mean: visible_life_stock"):
        instance._resolve_type_field(bv, "Player.visible_life_stok")


def test_find_function_suggests_close_match_when_not_found(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "player_update"),
            _FakeFunction(0x402000, "player_render"),
        ]
    )

    with pytest.raises(RuntimeError) as exc_info:
        instance._find_function(bv, "player_updaet")

    message = str(exc_info.value)
    assert message.startswith("Function not found: player_updaet")
    assert "Did you mean: player_update" in message


def test_find_function_not_found_without_close_match(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x401000, "player_update")])

    with pytest.raises(RuntimeError) as exc_info:
        instance._find_function(bv, "zzzzzzzz")

    assert str(exc_info.value) == "Function not found: zzzzzzzz"


def test_parse_address_reports_friendly_error_for_garbage(monkeypatch):
    bridge = _load_bridge(monkeypatch)

    with pytest.raises(ValueError) as exc_info:
        bridge._parse_address("not_an_address")

    message = str(exc_info.value)
    assert "not a valid address" in message
    # The raw int() ValueError must not leak through.
    assert "invalid literal for int" not in message


def test_find_function_invalid_hex_reports_address_not_missing_function(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x401000, "player_update")])

    with pytest.raises(RuntimeError) as exc_info:
        instance._find_function(bv, "0xGGGG")

    message = str(exc_info.value)
    assert "Invalid address" in message
    assert "0xGGGG" in message
    assert "Function not found" not in message


def test_find_function_valid_address_with_no_function(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x401000, "player_update")])

    with pytest.raises(RuntimeError) as exc_info:
        instance._find_function(bv, "0x999999")

    assert str(exc_info.value) == "No function found at address 0x999999"


def test_find_type_suggests_close_match_when_not_found(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        types_={
            "Player": _FakeType("struct Player"),
            "Enemy": _FakeType("struct Enemy"),
        }
    )

    with pytest.raises(RuntimeError) as exc_info:
        instance._find_type(bv, "Playr")

    message = str(exc_info.value)
    assert message.startswith("Type not found: Playr")
    assert "Did you mean: Player" in message


def test_find_type_not_found_without_close_match(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(types_={"Player": _FakeType("struct Player")})

    with pytest.raises(RuntimeError) as exc_info:
        instance._find_type(bv, "zzzzzzzz")

    message = str(exc_info.value)
    assert message.startswith("Type not found: zzzzzzzz")
    # No close match -> point the user at the substring search command (#174).
    assert "Did you mean" not in message
    assert "bn types --query zzzzzzzz" in message


def test_find_type_missing_primitive_typedef_hints_query_root(monkeypatch):
    """The common dead-end is a missing primitive typedef (e.g. `uint32_t` on a
    target that defines `unsigned int`). With no close match, the hint suggests
    a substring search on the typedef root (`_t` dropped) so the user can find
    the underlying type they actually have (#174)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(types_={"unsigned int": _FakeType("unsigned int")})

    with pytest.raises(RuntimeError) as exc_info:
        instance._find_type(bv, "uint32_t")

    message = str(exc_info.value)
    assert message.startswith("Type not found: uint32_t")
    assert "bn types --query uint32" in message
    assert "bn types --query uint32_t" not in message


def test_find_type_primitive_typedef_hints_query_even_with_close_matches(monkeypatch):
    """On a real target, difflib returns UNRELATED `_t` typedefs as "close" to a
    missing primitive (`uint32_t` -> wint_t, off64_t, uint64_t), so a hint gated
    on get_close_matches() being empty never fires for exactly the case #174 is
    meant to help. The search hint must accompany the suggestions, not replace
    or hide behind them (PR #189 dogfood)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(types_={
        "wint_t": _FakeType("typedef int wint_t"),
        "off64_t": _FakeType("typedef long off64_t"),
        "uint64_t": _FakeType("typedef unsigned long uint64_t"),
    })

    with pytest.raises(RuntimeError) as exc_info:
        instance._find_type(bv, "uint32_t")

    message = str(exc_info.value)
    assert message.startswith("Type not found: uint32_t")
    assert "Did you mean:" in message               # difflib suggestions kept
    assert "bn types --query uint32" in message       # AND the search hint fires
    assert "bn types --query uint32_t" not in message  # `_t` root dropped


def test_list_locals_returns_stable_ids(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "player_update", "int32_t player_update(int32_t arg1)")
    fn.parameter_vars = [
        _FakeVariable(name="arg1", storage=4, var_type="int32_t", identifier=1001, index=0)
    ]
    fn.stack_layout = [
        _FakeVariable(name="var_4", storage=-4, var_type="float", identifier=2001, index=1)
    ]
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._list_locals_for_function("active", "player_update")

    assert result["function"]["name"] == "player_update"
    assert len(result["locals"]) == 2
    assert result["locals"][0]["local_id"].startswith("0x401000:param:")
    assert result["locals"][1]["local_id"].startswith("0x401000:local:")


def test_list_locals_skips_stack_aliases_for_parameters(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "player_update")
    parameter = _FakeVariable(name="arg1", storage=4, var_type="int32_t", identifier=1001)
    alias = _FakeVariable(name="arg1", storage=4, var_type="int32_t", identifier=1001)
    local = _FakeVariable(name="var_4", storage=-4, var_type="float", identifier=2001)
    fn.parameter_vars = [parameter]
    fn.stack_layout = [alias, local]

    locals_list = instance._list_locals(fn)

    assert len(locals_list) == 2
    assert [item["local_id"] for item in locals_list] == [
        "0x401000:param:stack:4:0:1001",
        "0x401000:local:stack:-4:0:2001",
    ]


def test_list_locals_surfaces_hlil_register_and_flag_vars(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401230, "keychecker_step", "int32_t keychecker_step(void* arg1, char arg2)")
    arg1 = _FakeVariable(name="arg1", storage=105, var_type="void*", identifier=5001,
                         source_type="RegisterVariableSourceType")
    arg2 = _FakeVariable(name="arg2", storage=104, var_type="char", identifier=5002,
                         source_type="RegisterVariableSourceType")
    ret = _FakeVariable(name="__return_addr", storage=0, var_type="void*", identifier=6001,
                        source_type="StackVariableSourceType")
    fn.parameter_vars = [arg1, arg2]
    fn.stack_layout = [ret]
    # Register/flag locals only visible through HLIL; arg1/arg2 reappear here
    # (same Variable identity) and must dedupe against the parameter entries.
    rsi_1 = _FakeVariable(name="rsi_1", storage=104, var_type="char", identifier=5011, index=11,
                          source_type="RegisterVariableSourceType")
    rdx_3 = _FakeVariable(name="rdx_3", storage=100, var_type="int32_t", identifier=5032, index=32,
                          source_type="RegisterVariableSourceType")
    cond = _FakeVariable(name="cond:0", storage=2147483648, var_type="bool", identifier=7000, index=15,
                         source_type="FlagVariableSourceType")
    fn.hlil = types.SimpleNamespace(vars=[arg1, arg2, rsi_1, rdx_3, cond])

    locals_list = instance._list_locals(fn)
    by_name = {item["name"]: item for item in locals_list}

    # params + stack + the 3 HLIL-only vars, with no duplicate arg1/arg2
    assert [item["name"] for item in locals_list].count("arg1") == 1
    assert [item["name"] for item in locals_list].count("arg2") == 1
    assert {"rsi_1", "rdx_3", "cond:0"} <= set(by_name)
    assert by_name["rsi_1"]["local_id"] == "0x401230:local:reg:104:11:5011"
    assert by_name["cond:0"]["local_id"] == "0x401230:local:flag:2147483648:15:7000"

    # The point of the fix: a register var is now resolvable for rename/retype,
    # by both its local_id and its name.
    found, is_param = instance._find_variable_selector(fn, by_name["rsi_1"]["local_id"])
    assert found is rsi_1 and is_param is False
    found_by_name, _ = instance._find_variable_selector(fn, "rdx_3")
    assert found_by_name is rdx_3


def test_list_locals_without_hlil_falls_back_to_param_and_stack(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # _FakeFunction has no `.hlil` attribute -> graceful fallback, no crash.
    fn = _FakeFunction(0x401000, "f", "int32_t f(int32_t arg1)")
    fn.parameter_vars = [_FakeVariable(name="arg1", storage=4, var_type="int32_t", identifier=1001)]
    fn.stack_layout = [_FakeVariable(name="var_4", storage=-4, var_type="int32_t", identifier=2001)]

    locals_list = instance._list_locals(fn)

    assert [item["name"] for item in locals_list] == ["arg1", "var_4"]


def test_find_variable_selector_prefers_local_id(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "player_update")
    shared = _FakeVariable(name="tmp", storage=-4, var_type="int32_t", identifier=2001)
    duplicate = _FakeVariable(name="tmp", storage=-8, var_type="int32_t", identifier=2002)
    fn.stack_layout = [shared, duplicate]

    local_id = instance._local_id(fn, duplicate, is_parameter=False)
    found, is_parameter = instance._find_variable_selector(fn, local_id)

    assert found is duplicate
    assert is_parameter is False


def test_function_info_includes_metadata(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "player_update", "int32_t player_update(int32_t arg1)")
    fn.parameter_vars = [
        _FakeVariable(name="arg1", storage=4, var_type="int32_t", identifier=1001, index=0)
    ]
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_info("active", "player_update")

    assert result["prototype"] == "int32_t player_update(int32_t arg1)"
    assert result["return_type"] == "int32_t"
    assert result["calling_convention"] == "__cdecl"
    assert result["size"] is None
    # A function with no unlifted instructions reports a 0 count, not absence.
    assert result["unimplemented_instructions"] == {"count": 0, "addresses": [], "truncated": False}


def test_function_info_reports_unimplemented_instructions(monkeypatch):
    """function info aggregates instructions BN's lifter could not model so an
    FP-heavy function isn't mistaken for fully analyzed (#206)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x405000, "transform")
    # Two unlifted FP instructions (e.g. AArch64 fnmsub) surface as LLIL_UNIMPL.
    fn.low_level_il = [_FakeBlock([
        _FakeMLILInsn(0x4056f8, operation="LLIL_UNIMPL"),
        _FakeMLILInsn(0x4056fc, operation="LLIL_UNIMPL"),
        _FakeMLILInsn(0x405700, operation="LLIL_SET_REG"),
    ])]
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_info("active", "transform")

    ui = result["unimplemented_instructions"]
    assert ui["count"] == 2
    assert ui["addresses"] == ["0x4056f8", "0x4056fc"]
    assert ui["truncated"] is False


def _install_fake_pseudo_c(monkeypatch, bridge, func, batches):
    """Install a fake `binaryninja.lineardisassembly` whose cursor yields `batches`.

    `batches` is a list of line-batches; each line is an (address, text) pair,
    mirroring how a LinearViewCursor returns LinearDisassemblyLine objects.
    """

    class _FakeContents:
        def __init__(self, address, text):
            self.address = address
            self._text = text

        def __str__(self):
            return self._text

    class _FakeLine:
        def __init__(self, address, text):
            self.contents = _FakeContents(address, text)

    class _FakeViewObject:
        def __init__(self, batches):
            self.batches = [[_FakeLine(a, t) for (a, t) in batch] for batch in batches]

    class _FakeCursor:
        def __init__(self, view_obj):
            self._batches = view_obj.batches
            self._i = 0

        def seek_to_begin(self):
            self._i = 0

        @property
        def lines(self):
            return self._batches[self._i] if self._i < len(self._batches) else []

        def next(self):
            self._i += 1
            return self._i < len(self._batches)

    class _FakeLinearViewObject:
        @staticmethod
        def single_function_language_representation(fn, settings=None, language="Pseudo C"):
            assert fn is func
            assert language == "Pseudo C"
            return _FakeViewObject(batches)

    fake_mod = types.ModuleType("binaryninja.lineardisassembly")
    fake_mod.LinearViewObject = _FakeLinearViewObject
    fake_mod.LinearViewCursor = _FakeCursor
    monkeypatch.setattr(bridge.bn, "lineardisassembly", fake_mod, raising=False)

    class _FakeDisassemblySettings:
        def set_option(self, option, state=True):
            return None

    class _FakeDisassemblyOption:
        ShowAddress = 0
        ShowTypeCasts = 10
        WaitForIL = 66
        DisableLineFormatting = 68

    monkeypatch.setattr(bridge.bn, "DisassemblySettings", _FakeDisassemblySettings, raising=False)
    monkeypatch.setattr(bridge.bn, "DisassemblyOption", _FakeDisassemblyOption, raising=False)


def test_decompile_renders_pseudo_c(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "player_update")
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    _install_fake_pseudo_c(
        monkeypatch,
        bridge,
        fn,
        [
            # BN indents the signature line by 2 spaces; we left-justify it.
            [(0x401000, "  int32_t player_update(int32_t arg1)")],
            [(0x401000, "{")],
            [(0x401004, "    return arg1 + 1;")],
            [(0x401008, "}")],
        ],
    )

    result = instance._decompile("active", "player_update")

    assert result["function"] == {"name": "player_update", "address": "0x401000"}
    assert result["text"] == (
        "int32_t player_update(int32_t arg1)\n{\n    return arg1 + 1;\n}"
    )


def test_decompile_pseudo_c_with_address_gutter(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "player_update")
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    _install_fake_pseudo_c(
        monkeypatch,
        bridge,
        fn,
        [
            [(0x401000, "")],  # leading blank separator -> trimmed, no orphan address
            [(0x401000, "  int32_t player_update(int32_t arg1)")],  # 2-space indent stripped
            [(0x401000, "")],  # internal blank -> empty line, not "00401000"
            [(0x401004, "    return arg1 + 1;")],
            [(0x401004, "")],  # trailing blank separator -> trimmed
        ],
    )

    result = instance._decompile("active", "player_update", addresses=True)

    assert result["text"] == (
        "00401000        int32_t player_update(int32_t arg1)\n"
        "\n"
        "00401004            return arg1 + 1;"
    )


def test_decompile_falls_back_to_hlil_when_pseudo_c_unavailable(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "player_update", "int32_t player_update()")
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    # No fake lineardisassembly module installed -> _pseudo_c_text raises and we
    # fall back to wrapped HLIL produced by _function_text. The renderer now
    # lives in il_format and _decompile_text calls it module-locally, so stub it
    # there (patching instance._function_text no longer intercepts that call).
    monkeypatch.setattr(bridge.il_format, "_function_text", lambda bv, func, **kw: "    return 1;")

    result = instance._decompile("active", "player_update")

    # The pseudo-C failure is surfaced via an explicit marker line instead of
    # silently presenting the HLIL fallback as a successful decompilation.
    lines = result["text"].splitlines()
    assert lines[0].startswith("// bn: decompilation failed (")
    assert "\n".join(lines[1:]) == "int32_t player_update()\n{\n    return 1;\n}"


def test_decompile_warns_on_skipped_analysis(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "big_fn")
    fn.analysis_skipped = True
    fn.analysis_skip_reason = "ExceedFunctionSize"
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    # Body has no telltale text -> warning must fire on the analysis_skipped flag alone.
    _install_fake_pseudo_c(
        monkeypatch, bridge, fn,
        [[(0x401000, "int32_t big_fn()")], [(0x401000, "{")], [(0x401000, "}")]],
    )

    result = instance._decompile("active", "big_fn")

    assert result["analysis_skipped"] is True
    assert result["analysis_forced"] is False
    assert result["analysis_force_requested"] is False
    assert fn.reanalyzed is False  # warn-only must NOT reanalyze
    assert any("big_fn" in w and "ExceedFunctionSize" in w and "--force-analysis" in w for w in result["warnings"])


def test_decompile_warns_on_placeholder_text_when_flag_clear(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "big_fn")  # analysis_skipped defaults False
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    _install_fake_pseudo_c(
        monkeypatch, bridge, fn,
        [
            [(0x401000, "int32_t big_fn()")],
            [(0x401000, "{")],
            [(0x401000, "    // This function is taking too long to analyze")],
            [(0x401000, "}")],
        ],
    )

    result = instance._decompile("active", "big_fn")

    assert result["analysis_skipped"] is False
    assert any("incomplete stub" in w for w in result["warnings"])


def test_decompile_force_analysis_reanalyzes_and_clears_warning(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "big_fn")
    fn.analysis_skipped = True
    fn.analysis_skip_reason = "ExceedFunctionSize"
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    _install_fake_pseudo_c(
        monkeypatch, bridge, fn,
        [
            [(0x401000, "int32_t big_fn()")],
            [(0x401000, "{")],
            [(0x401004, "    return 1;")],
            [(0x401008, "}")],
        ],
    )

    result = instance._decompile("active", "big_fn", force_analysis=True)

    assert fn.reanalyzed is True                       # reanalysis was triggered
    assert getattr(bv, "analysis_updated", False) is True
    assert fn.analysis_skipped is False               # skip override cleared
    assert result["analysis_forced"] is True
    assert result["analysis_force_requested"] is True
    assert result["analysis_skipped"] is False
    assert result["text"] == "int32_t big_fn()\n{\n    return 1;\n}"
    assert not any("stub" in w.lower() or "skipped analysis" in w for w in result["warnings"])


def test_list_functions_is_sorted_by_address(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x402000, "sub_402000"),
            _FakeFunction(0x401000, "sub_401000"),
        ]
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._list_functions("active")

    assert [item["address"] for item in result["functions"]] == ["0x401000", "0x402000"]
    assert result["total"] == 2 and result["has_more"] is False


def test_function_list_rows_carry_size_and_sort_by_size(monkeypatch):
    # The dogfood's most-repeated friction: no size field forces per-function
    # info loops / write-locked py exec to find large functions. Expose `size`
    # on every row and a `--sort size` that ranks largest-first.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    small = _FakeFunction(0x1000, "small_fn"); small.total_bytes = 16
    big = _FakeFunction(0x2000, "big_fn"); big.total_bytes = 4096
    mid = _FakeFunction(0x3000, "mid_fn"); mid.total_bytes = 256
    bv = _FakeBV(functions=[small, big, mid])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    res = instance._list_functions("active")
    by_name = {r["name"]: r for r in res["functions"]}
    assert by_name["small_fn"]["size"] == 16
    assert by_name["big_fn"]["size"] == 4096

    ranked = instance._list_functions("active", sort="size")
    assert [r["name"] for r in ranked["functions"]] == ["big_fn", "mid_fn", "small_fn"]


def test_function_search_rows_carry_size(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x2000, "parse_packet"); fn.total_bytes = 512
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    res = instance._search_functions("active", "parse")
    assert res["functions"][0]["size"] == 512


def test_list_functions_binder_forwards_sort(monkeypatch):
    # Regression: the op binder must FORWARD `sort` to the handler. A unit test
    # that calls the handler directly misses a binder that drops the param --
    # the live re-use gate caught exactly this, so guard it through the binder.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    small = _FakeFunction(0x1000, "small_fn"); small.total_bytes = 8
    big = _FakeFunction(0x2000, "big_fn"); big.total_bytes = 9000
    bv = _FakeBV(functions=[small, big])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    res = bridge._bind_list_functions(instance, {"sort": "size"}, "active")
    assert [r["name"] for r in res["functions"]] == ["big_fn", "small_fn"]
    res2 = bridge._bind_search_functions(instance, {"query": "_fn", "sort": "size"}, "active")
    assert [r["name"] for r in res2["functions"]] == ["big_fn", "small_fn"]


def test_function_binders_tolerate_none_limit(monkeypatch):
    """A raw-protocol / py-exec / batch caller can send `limit: None` (key
    present, null value) to list_functions / search_functions -- the CLI omits
    the key when None, but the bridge protocol accepts arbitrary params. The
    binder must read None as "no limit", not int(None) -- the same key-presence
    vs value-not-None guard bug that crashed `comment list`."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x1000, "alpha"), _FakeFunction(0x2000, "beta")])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    res = bridge._bind_list_functions(instance, {"limit": None, "offset": 0}, "active")
    assert len(res["functions"]) == 2
    res2 = bridge._bind_search_functions(instance, {"query": "alpha", "limit": None, "offset": 0}, "active")
    assert len(res2["functions"]) == 1


def test_function_list_envelope_exposes_items_alias_and_count_total(monkeypatch):
    # JSON-consistency: function list/search expose the universal `items` key
    # (alias of `functions`) so `data["items"]` works across every list command;
    # and --count carries `total` to match the list envelope's key.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x1000, "a"), _FakeFunction(0x2000, "b")])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    res = instance._list_functions("active")
    assert res["items"] == res["functions"]
    sres = instance._search_functions("active", "a")
    assert sres["items"] == sres["functions"]

    count = instance._list_functions("active", count_only=True)
    assert count["count"] == 2 and count["total"] == 2


def test_list_functions_can_filter_by_address_range(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "sub_401000"),
            _FakeFunction(0x402000, "sub_402000"),
            _FakeFunction(0x403000, "sub_403000"),
        ]
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._list_functions("active", min_address="0x401800", max_address="0x402fff")

    assert [item["address"] for item in result["functions"]] == ["0x402000"]
    assert result["total"] == 1


def test_search_functions_supports_regex(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "load_attachment"),
            _FakeFunction(0x402000, "detach_player"),
            _FakeFunction(0x403000, "update_camera"),
        ]
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._search_functions("active", "attach|detach", regex=True)

    assert [item["name"] for item in result["functions"]] == ["load_attachment", "detach_player"]


def test_search_functions_rejects_invalid_regex(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x401000, "load_attachment")])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure, match="Invalid function regex"):
        instance._search_functions("active", "(", regex=True)


def _callsites_items(instance, *args, **kwargs):
    """Unwrap the {items,total,...} callsites envelope to the row list (#131).

    callsites now returns the same paging envelope as the sibling list ops; these
    tests assert on the rows, so unwrap once here rather than in every assertion.
    Also asserts the envelope shape so the contract stays covered."""
    result = instance._callsites(*args, **kwargs)
    assert isinstance(result, dict) and "items" in result and "total" in result
    return result["items"]


def test_callsites_returns_local_hlil_assignment_and_pre_branch_condition(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    branch = _FakeHLILInstruction(
        "if (result == 2)",
        class_name="HighLevelILIf",
        condition="result == 2",
        expr_index=40,
        instr_index=40,
    )
    first_statement = _FakeHLILInstruction(
        "edx_1:eax_1 = sx.q(crt_rand())",
        class_name="HighLevelILVarInit",
        expr_index=32,
        instr_index=32,
    )
    first_sx = _FakeHLILInstruction(
        "sx.q(crt_rand())",
        class_name="HighLevelILSx",
        parent=first_statement,
        expr_index=31,
        instr_index=31,
    )
    first_call = _FakeHLILInstruction(
        "crt_rand()",
        class_name="HighLevelILCall",
        parent=first_sx,
        expr_index=30,
        instr_index=30,
    )
    second_statement = _FakeHLILInstruction(
        "eax_3, edx_2 = crt_rand()",
        class_name="HighLevelILVarInit",
        parent=branch,
        expr_index=42,
        instr_index=42,
    )
    second_call = _FakeHLILInstruction(
        "crt_rand()",
        class_name="HighLevelILCall",
        parent=second_statement,
        expr_index=41,
        instr_index=41,
    )
    callee = _FakeFunction(0x461746, "crt_rand")
    fn = _FakeFunction(0x412470, "bonus_pick_random_type")
    fn.basic_blocks = [_FakeBasicBlock(0x41249C, 0x4124D8)]
    fn.low_level_il = [
        [
            _FakeLLILInstruction(0x4124A0, _FakeConstPtr(0x461746), hlils=[first_call]),
            _FakeLLILInstruction(0x4124D1, _FakeConstPtr(0x461746), hlils=[second_call]),
        ]
    ]
    bv = _FakeBV(
        functions=[callee, fn],
        instruction_lengths={
            0x41249C: 2,
            0x41249E: 2,
            0x4124A0: 5,
            0x4124A5: 3,
            0x4124D1: 5,
            0x4124D6: 2,
        },
        disassembly={
            0x41249C: "mov eax, 0",
            0x41249E: "mov ebx, 0",
            0x4124A0: "call crt_rand",
            0x4124A5: "cmp eax, 0xd",
            0x4124D1: "call crt_rand",
            0x4124D6: "test al, 0x3f",
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance,
        "active",
        "crt_rand",
        within_identifiers=["bonus_pick_random_type"],
        context=2,
    )

    assert [row["caller_static"] for row in rows] == ["0x4124a5", "0x4124d6"]
    assert rows[0]["call_addr"] == "0x4124a0"
    assert rows[0]["instruction_length"] == 5
    assert rows[0]["call_index"] == 0
    assert rows[0]["within_query"] == "bonus_pick_random_type"
    assert rows[0]["hlil_statement"] == "edx_1:eax_1 = sx.q(crt_rand())"
    assert rows[0]["pre_branch_condition"] is None
    assert rows[1]["call_index"] == 1
    assert rows[1]["hlil_statement"] == "eax_3, edx_2 = crt_rand()"
    assert rows[1]["pre_branch_condition"] == "result == 2"
    assert [item["address"] for item in rows[0]["previous_instructions"]] == ["0x41249c", "0x41249e"]
    assert rows[0]["call_instruction"]["text"] == "call crt_rand"
    assert [item["address"] for item in rows[0]["next_instructions"][:1]] == ["0x4124a5"]


def test_callsites_finds_register_dest_call_via_code_ref_db(monkeypatch):
    # On stripped/kernel/MIPS targets a call's LLIL dest is a register/computed
    # value BN resolved via analysis and recorded in the code-ref DB (what xrefs
    # reads), NOT a literal const. callsites must agree with xrefs, not silently
    # drop the edge. Here the only call dest is a register, so the literal-const
    # match misses it and only the code-ref DB resolves it.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "target_fn")
    fn = _FakeFunction(0x412470, "caller_fn")
    fn.basic_blocks = [_FakeBasicBlock(0x4124A0, 0x4124A4)]
    fn.low_level_il = [[_FakeLLILInstruction(0x4124A0, _FakeReg("x8"))]]
    bv = _FakeBV(
        functions=[callee, fn],
        instruction_lengths={0x4124A0: 4},
        disassembly={0x4124A0: "blr x8"},
        code_refs={0x461746: [_FakeCodeRef(0x4124A0, fn)]},  # BN's DB knows the edge
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance, "active", "target_fn", within_identifiers=["caller_fn"], context=1)
    assert len(rows) == 1
    assert rows[0]["call_addr"] == "0x4124a0"
    assert rows[0]["caller_static"] == "0x4124a4"


def test_callsites_prefers_local_expression_over_broad_enclosing_hlil(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    branch = _FakeHLILInstruction(
        "if (config_fx_toggle != 0)",
        class_name="HighLevelILIf",
        condition="config_fx_toggle != 0",
        expr_index=100,
        instr_index=100,
    )
    broad_statement = _FakeHLILInstruction(
        "if (config_fx_toggle != 0)\nlong expression blob\nreturn",
        class_name="HighLevelILVarInit",
        parent=branch,
        expr_index=99,
        instr_index=99,
    )
    add_expr = _FakeHLILInstruction(
        "float.t(crt_rand() & 0xf) * 0.01 + 0.84",
        class_name="HighLevelILAdd",
        parent=broad_statement,
        expr_index=35,
        instr_index=9,
    )
    mul_expr = _FakeHLILInstruction(
        "float.t(crt_rand() & 0xf) * 0.01",
        class_name="HighLevelILMul",
        parent=add_expr,
        expr_index=34,
        instr_index=9,
    )
    cast_expr = _FakeHLILInstruction(
        "float.t(crt_rand() & 0xf)",
        class_name="HighLevelILIntToFloat",
        parent=mul_expr,
        expr_index=33,
        instr_index=9,
    )
    and_expr = _FakeHLILInstruction(
        "crt_rand() & 0xf",
        class_name="HighLevelILAnd",
        parent=cast_expr,
        expr_index=32,
        instr_index=9,
    )
    call_expr = _FakeHLILInstruction(
        "crt_rand()",
        class_name="HighLevelILCall",
        parent=and_expr,
        expr_index=31,
        instr_index=9,
    )
    callee = _FakeFunction(0x461746, "crt_rand")
    fn = _FakeFunction(0x427700, "fx_queue_add_random")
    fn.basic_blocks = [_FakeBasicBlock(0x427753, 0x427768)]
    fn.low_level_il = [[_FakeLLILInstruction(0x42775B, _FakeConstPtr(0x461746), hlils=[broad_statement, call_expr])]]
    bv = _FakeBV(
        functions=[callee, fn],
        instruction_lengths={
            0x427753: 5,
            0x427758: 3,
            0x42775B: 5,
            0x427760: 3,
        },
        disassembly={
            0x427753: "call helper",
            0x427758: "add esp, 0x4",
            0x42775B: "call crt_rand",
            0x427760: "and eax, 0xf",
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance,
        "active",
        "crt_rand",
        within_identifiers=["fx_queue_add_random"],
        context=2,
    )

    assert len(rows) == 1
    assert rows[0]["hlil_statement"] == "float.t(crt_rand() & 0xf) * 0.01 + 0.84"
    assert rows[0]["pre_branch_condition"] == "config_fx_toggle != 0"
    assert rows[0]["call_index"] == 0
    assert rows[0]["within_query"] == "fx_queue_add_random"


def test_callsites_within_file_scope_preserves_file_order_and_dedupes(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "crt_rand")
    alpha = _FakeFunction(0x401000, "alpha")
    alpha.basic_blocks = [_FakeBasicBlock(0x401010, 0x401016)]
    alpha.low_level_il = [[_FakeLLILInstruction(0x401010, _FakeConstPtr(0x461746))]]
    beta = _FakeFunction(0x402000, "beta")
    beta.basic_blocks = [_FakeBasicBlock(0x402020, 0x402026)]
    beta.low_level_il = [[_FakeLLILInstruction(0x402020, _FakeConstPtr(0x461746))]]
    bv = _FakeBV(
        functions=[callee, alpha, beta],
        instruction_lengths={0x401010: 5, 0x402020: 5},
        disassembly={0x401010: "call crt_rand", 0x402020: "call crt_rand"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance,
        "active",
        "crt_rand",
        within_identifiers=["beta", "alpha", "beta"],
        context=0,
    )

    assert [row["containing_function"]["name"] for row in rows] == ["beta", "alpha"]
    assert [row["caller_static"] for row in rows] == ["0x402025", "0x401015"]
    assert [row["within_query"] for row in rows] == ["beta", "alpha"]
    assert [row["call_index"] for row in rows] == [0, 0]


def test_callsites_ignores_indirect_calls_and_returns_null_context_when_unmapped(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "crt_rand")
    fn = _FakeFunction(0x500000, "fx_queue_add_random")
    fn.basic_blocks = [_FakeBasicBlock(0x500010, 0x50001A)]
    fn.low_level_il = [
        [
            _FakeLLILInstruction(0x500010, _FakeReg("eax")),
            _FakeLLILInstruction(0x500015, _FakeConstPtr(0x461746)),
        ]
    ]
    bv = _FakeBV(
        functions=[callee, fn],
        instruction_lengths={0x500010: 5, 0x500015: 5},
        disassembly={0x500010: "call eax", 0x500015: "call crt_rand"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance,
        "active",
        "crt_rand",
        within_identifiers=["fx_queue_add_random"],
        context=1,
    )

    assert len(rows) == 1
    assert rows[0]["call_addr"] == "0x500015"
    assert rows[0]["hlil_statement"] is None
    assert rows[0]["pre_branch_condition"] is None


def test_callsites_counts_tailcall_into_target(monkeypatch):
    # A tail-branch into the target (`return <addr>(...) __tailcall`, e.g. a
    # j_memcpy veneer) must be reported as a callsite -- xrefs and taint already
    # treat it as a call, so callsites must agree (#47).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "memcpy")
    fn = _FakeFunction(0x700000, "j_memcpy")
    fn.basic_blocks = [_FakeBasicBlock(0x700010, 0x700014)]
    fn.low_level_il = [[
        _FakeLLILInstruction(0x700010, _FakeConstPtr(0x461746), operation="LLIL_TAILCALL"),
    ]]
    bv = _FakeBV(
        functions=[callee, fn],
        instruction_lengths={0x700010: 4},
        disassembly={0x700010: "b #memcpy"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance,"active", "memcpy", within_identifiers=["j_memcpy"], context=1)

    assert len(rows) == 1
    assert rows[0]["call_addr"] == "0x700010"
    assert rows[0]["call_kind"] == "tailcall"


def test_callsites_marks_regular_call_kind(monkeypatch):
    # A normal bl/blx call is reported with call_kind 'call' (not tailcall).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "memcpy")
    fn = _FakeFunction(0x700000, "caller")
    fn.basic_blocks = [_FakeBasicBlock(0x700010, 0x700014)]
    fn.low_level_il = [[_FakeLLILInstruction(0x700010, _FakeConstPtr(0x461746))]]  # default LLIL_CALL
    bv = _FakeBV(
        functions=[callee, fn],
        instruction_lengths={0x700010: 4},
        disassembly={0x700010: "bl #memcpy"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance,"active", "memcpy", within_identifiers=["caller"], context=1)

    assert len(rows) == 1
    assert rows[0]["call_kind"] == "call"


def test_callsites_returns_null_for_coarse_only_hlil(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "crt_rand")
    broad_statement = _FakeHLILInstruction(
        "if (x)\nwhole function blob\nreturn",
        class_name="HighLevelILVarInit",
        expr_index=10,
        instr_index=10,
    )
    fn = _FakeFunction(0x600000, "coarse")
    fn.basic_blocks = [_FakeBasicBlock(0x600010, 0x600016)]
    fn.low_level_il = [[_FakeLLILInstruction(0x600010, _FakeConstPtr(0x461746), hlils=[broad_statement])]]
    bv = _FakeBV(
        functions=[callee, fn],
        instruction_lengths={0x600010: 5},
        disassembly={0x600010: "call crt_rand"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance,
        "active",
        "crt_rand",
        within_identifiers=["coarse"],
        context=1,
    )

    assert len(rows) == 1
    assert rows[0]["hlil_statement"] is None
    assert rows[0]["pre_branch_condition"] is None


def test_callsites_filters_placeholder_pre_branch_condition(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    branch = _FakeHLILInstruction(
        "do while (not(cond:0_1))",
        class_name="HighLevelILDoWhile",
        condition="not(cond:0_1)",
        expr_index=50,
        instr_index=50,
    )
    statement = _FakeHLILInstruction(
        "eax_1 = crt_rand()",
        class_name="HighLevelILVarInit",
        parent=branch,
        expr_index=51,
        instr_index=51,
    )
    call = _FakeHLILInstruction(
        "crt_rand()",
        class_name="HighLevelILCall",
        parent=statement,
        expr_index=52,
        instr_index=52,
    )
    callee = _FakeFunction(0x461746, "crt_rand")
    fn = _FakeFunction(0x700000, "placeholder_cond")
    fn.basic_blocks = [_FakeBasicBlock(0x700010, 0x700016)]
    fn.low_level_il = [[_FakeLLILInstruction(0x700010, _FakeConstPtr(0x461746), hlils=[call])]]
    bv = _FakeBV(
        functions=[callee, fn],
        instruction_lengths={0x700010: 5},
        disassembly={0x700010: "call crt_rand"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance,
        "active",
        "crt_rand",
        within_identifiers=["placeholder_cond"],
        context=1,
    )

    assert rows[0]["pre_branch_condition"] is None


def test_xrefs_include_address_context(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x401000, "caller")
    target = _FakeFunction(0x402000, "target")
    bv = _FakeBV(
        functions=[caller, target],
        symbols=[bridge.bn.Symbol(bridge.bn.SymbolType.DataSymbol, 0x5000, "type_name")],
        code_refs={0x5000: [_FakeCodeRef(0x401010, caller)]},
        data_refs={0x5000: [0x6000]},
        disassembly={0x401010: "ldr r0, =type_name"},
        sections={
            ".text": _FakeSection(".text", 0x400000, 0x410000),
            ".rodata": _FakeSection(".rodata", 0x5000, 0x7000),
        },
        segments={
            0x401010: _FakeSegment(readable=True, executable=True),
            0x5000: _FakeSegment(readable=True),
            0x6000: _FakeSegment(readable=True, writable=True),
        },
    )

    result = instance._xrefs_to_address(bv, 0x5000)

    assert result["target_context"]["symbol"]["name"] == "type_name"
    assert result["code_refs"][0]["context"]["disasm"] == "ldr r0, =type_name"
    assert result["code_refs"][0]["context"]["sections"][0]["name"] == ".text"
    assert result["data_refs"][0]["context"]["sections"][0]["name"] == ".rodata"
    # JSON carries the same summary counts the text header shows, so an agent
    # can size/triage without materializing the (spilling) code_refs[] array.
    assert result["code_ref_count"] == 1
    assert result["data_ref_count"] == 1
    assert result["caller_function_count"] == 1


def test_xrefs_to_address_emits_paging_envelope(monkeypatch):
    # #164: xrefs adopts the canonical {items,total,offset,limit,returned,has_more}
    # envelope (items = code refs then data refs, each keeping its kind), pages on
    # offset/limit, and keeps the #140 summary counts + the deprecated dual shape.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x401000, "caller")
    bv = _FakeBV(
        functions=[caller],
        code_refs={0x5000: [_FakeCodeRef(0x401010, caller), _FakeCodeRef(0x401020, caller)]},
        data_refs={0x5000: [0x6000]},
        sections={".text": _FakeSection(".text", 0x400000, 0x410000),
                  ".rodata": _FakeSection(".rodata", 0x5000, 0x7000)},
        segments={0x401010: _FakeSegment(readable=True, executable=True),
                  0x401020: _FakeSegment(readable=True, executable=True),
                  0x6000: _FakeSegment(readable=True, writable=True)},
    )
    full = instance._xrefs_to_address(bv, 0x5000)
    assert full["total"] == 3
    assert full["returned"] == 3
    assert full["has_more"] is False
    assert [it["kind"] for it in full["items"]] == ["code", "code", "data"]
    assert full["code_ref_count"] == 2 and full["data_ref_count"] == 1
    # deprecated dual shape stays full (function-info embeds it unpaged)
    assert len(full["code_refs"]) == 2 and len(full["data_refs"]) == 1

    page = instance._xrefs_to_address(bv, 0x5000, offset=0, limit=2)
    assert page["returned"] == 2 and page["has_more"] is True
    assert [it["kind"] for it in page["items"]] == ["code", "code"]
    assert page["total"] == 3
    # summary counts + dual shape reflect the FULL set regardless of paging
    assert page["code_ref_count"] == 2 and len(page["code_refs"]) == 2


def test_xrefs_op_drops_deprecated_arrays(monkeypatch):
    # #184: the `xrefs` OP response must NOT carry the full code_refs/data_refs
    # arrays -- they rode unbounded past --offset/--limit and spilled the JSON on
    # high-fanout symbols. Keep the full-set summary counts (#140) + the paged
    # `items`. The lower-level _xrefs_to_address still produces the dual shape,
    # which `function info` and evidence message-lensing embed directly (locked by
    # test_xrefs_to_address_emits_paging_envelope above).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x401000, "caller")
    bv = _FakeBV(
        functions=[caller],
        code_refs={0x5000: [_FakeCodeRef(0x401010, caller), _FakeCodeRef(0x401020, caller)]},
        data_refs={0x5000: [0x6000]},
        sections={".text": _FakeSection(".text", 0x400000, 0x410000),
                  ".rodata": _FakeSection(".rodata", 0x5000, 0x7000)},
        segments={0x401010: _FakeSegment(readable=True, executable=True),
                  0x401020: _FakeSegment(readable=True, executable=True),
                  0x6000: _FakeSegment(readable=True, writable=True)},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._xrefs(None, "0x5000", limit=1)

    # deprecated dual arrays are gone -> --limit truly bounds the payload
    assert "code_refs" not in result
    assert "data_refs" not in result
    # full-set summary counts survive paging (the triage signal)
    assert result["code_ref_count"] == 2
    assert result["data_ref_count"] == 1
    assert result["caller_function_count"] == 1
    assert result["total"] == 3
    # items is bounded by --limit
    assert result["returned"] == 1 and len(result["items"]) == 1
    assert result["has_more"] is True
    assert result["items"][0]["kind"] == "code"


def test_function_evidence_reports_calls_arguments_and_thunk_candidate(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x461746, "send_message")
    caller = _FakeFunction(0x412470, "build_response")
    call_expr = _FakeHLILInstruction(
        "send_message(6, &response)",
        class_name="HighLevelILCall",
        expr_index=30,
        instr_index=30,
    )
    call_expr.params = [6, "&response"]
    call_insn = _FakeLLILInstruction(0x4124A0, _FakeConstPtr(0x461746), hlils=[call_expr])
    call_insn.params = [_FakeReg("r0"), _FakeConstPtr(6), _FakeReg("r2")]
    caller.basic_blocks = [_FakeBasicBlock(0x41249C, 0x4124A8)]
    caller.low_level_il = [[call_insn]]
    thunk = _FakeFunction(0x500000, "j_send_message")
    thunk.basic_blocks = [_FakeBasicBlock(0x500000, 0x500004)]
    thunk.low_level_il = [[_FakeLLILInstruction(0x500000, _FakeConstPtr(0x461746), operation="LLIL_JUMP")]]
    bv = _FakeBV(
        functions=[callee, caller, thunk],
        instruction_lengths={0x41249C: 2, 0x41249E: 2, 0x4124A0: 4, 0x4124A4: 4},
        disassembly={
            0x41249C: "mov r1, #6",
            0x41249E: "mov r2, response",
            0x4124A0: "bl send_message",
            0x4124A4: "pop {pc}",
            0x500000: "b send_message",
        },
    )
    # _function_evidence now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_evidence("active", "build_response", context=1)
    call = result["calls"][0]

    assert call["direct"] is True
    assert call["target"]["function"]["name"] == "send_message"
    # primary args come from the single matched HLIL call; no merged/source-tagged noise
    assert call["argument_source"] == "hlil"
    assert [arg["text"] for arg in call["arguments"]] == ["6", "&response"]
    # other IL layers are quarantined as candidates, JSON-only
    assert any(c["source"] == "llil" for c in call["argument_candidates"])
    assert call["previous_instructions"][0]["text"] == "mov r2, response"

    thunk_result = instance._function_evidence("active", "j_send_message", context=0)
    assert thunk_result["thunk"]["is_candidate"] is True
    assert thunk_result["thunk"]["target"]["function"]["name"] == "send_message"


def test_xrefs_suppress_disasm_for_data_targets(monkeypatch):
    # ILX #1: a .rodata string target must not be disassembled into garbage,
    # even though firmware ELFs map .rodata into the r-x load segment.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x1A000, "caller")
    caller.basic_blocks = [_FakeBasicBlock(0x1A000, 0x1A100)]
    message = "basic_string::_M_construct null not valid"
    bv = _FakeBV(
        functions=[caller],
        code_refs={0x2A07C: [_FakeCodeRef(0x1A050, caller)]},
        disassembly={0x1A050: "ldr r0, =message"},
        sections={
            ".text": _FakeSection(".text", 0x10000, 0x20000),
            ".rodata": _FakeSection(".rodata", 0x2A000, 0x2B000),
        },
        segments={
            0x1A050: _FakeSegment(readable=True, executable=True),
            0x2A07C: _FakeSegment(readable=True, executable=True),  # rodata shares the r-x segment
        },
        memory={0x2A07C: message.encode() + b"\x00"},
    )

    result = instance._xrefs_to_address(bv, 0x2A07C)

    target = result["target_context"]
    assert target["kind"] == "string"
    assert target["string"]["value"] == message
    assert target["disasm"] is None
    assert target["notes"]
    # the referencing instruction is genuine code, so its disasm is kept
    assert result["code_refs"][0]["context"]["disasm"] == "ldr r0, =message"


def test_xrefs_resolve_multiline_strings_and_mark_truncation(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x401000, "usage")
    caller.basic_blocks = [_FakeBasicBlock(0x401000, 0x401100)]
    message = "Usage: %s [OPTION]... PATTERNS [FILE]...\n" + ("A" * 120)
    bv = _FakeBV(
        functions=[caller],
        code_refs={0x427840: [_FakeCodeRef(0x40EA7C, caller)]},
        disassembly={0x40EA7C: "lea rsi, [rel 0x427840]"},
        sections={
            ".text": _FakeSection(".text", 0x401000, 0x402000),
            ".rodata": _FakeSection(".rodata", 0x427000, 0x428000),
        },
        segments={
            0x40EA7C: _FakeSegment(readable=True, executable=True),
            0x427840: _FakeSegment(readable=True),
        },
        memory={0x427840: message.encode() + b"\x00"},
    )

    result = instance._xrefs_to_address(bv, 0x427840)

    target = result["target_context"]
    assert target["kind"] == "string"
    assert target["string"]["value"] == message[:96]
    assert "\n" in target["string"]["value"]
    assert target["string"]["truncated"] is True
    assert target["disasm"] is None
    assert result["code_refs"][0]["context"]["disasm"] == "lea rsi, [rel 0x427840]"


def test_function_evidence_resolves_pointer_constant_arguments(monkeypatch):
    # ILX #2: append(&var, 0x2a4f4) should annotate the constant with "4" [.rodata].
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x154F4, "append")
    caller = _FakeFunction(0x1A36C, "createBTService")
    call_expr = _FakeHLILInstruction(
        "append(&var_38, 0x2a4f4)", class_name="HighLevelILCall", expr_index=10, instr_index=10
    )
    call_expr.params = ["&var_38", "0x2a4f4"]
    call_insn = _FakeLLILInstruction(0x1A38E, _FakeConstPtr(0x154F4), hlils=[call_expr])
    caller.basic_blocks = [_FakeBasicBlock(0x1A38E, 0x1A392)]
    caller.low_level_il = [[call_insn]]
    bv = _FakeBV(
        functions=[callee, caller],
        instruction_lengths={0x1A38E: 4},
        disassembly={0x1A38E: "blx append"},
        sections={".rodata": _FakeSection(".rodata", 0x2A000, 0x2B000)},
        segments={0x2A4F4: _FakeSegment(readable=True)},
        memory={0x2A4F4: b"4\x00"},
    )
    # _function_evidence now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_evidence("active", "createBTService", context=0)
    call = result["calls"][0]
    constant = next(arg for arg in call["arguments"] if arg["text"] == "0x2a4f4")
    assert constant["resolved"]["string"] == "4"
    assert constant["resolved"]["section"] == ".rodata"


def test_function_evidence_does_not_merge_unrelated_hlil_call_args(monkeypatch):
    # ILX #3: one LLIL call mapping to two HLIL calls must not borrow the other's args.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    callee = _FakeFunction(0x16520, "getopt_long")
    caller = _FakeFunction(0x17140, "main")
    real = _FakeHLILInstruction(
        'getopt_long(argc, argv, "hb:d:")', class_name="HighLevelILCall", expr_index=5, instr_index=5
    )
    real.params = ["argc", "argv", '"hb:d:"']
    real.address = 0x17188
    unrelated = _FakeHLILInstruction(
        '__android_log_print(4, "aa accessory", x)',
        class_name="HighLevelILCall",
        expr_index=9,
        instr_index=9,
    )
    unrelated.params = ["4", '"aa accessory"', "x"]
    unrelated.address = 0x172A0
    call_insn = _FakeLLILInstruction(0x17188, _FakeConstPtr(0x16520), hlils=[real, unrelated])
    caller.basic_blocks = [_FakeBasicBlock(0x17140, 0x17200)]
    caller.low_level_il = [[call_insn]]
    bv = _FakeBV(
        functions=[callee, caller],
        instruction_lengths={0x17188: 4},
        disassembly={0x17188: "blx getopt_long"},
    )
    # _function_evidence now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_evidence("active", "main", context=0)
    call = result["calls"][0]
    assert call["argument_source"] == "hlil"
    assert [arg["text"] for arg in call["arguments"]] == ["argc", "argv", '"hb:d:"']
    assert all("aa accessory" not in arg["text"] for arg in call["arguments"])
    assert any(c["text"] == '"aa accessory"' for c in call["argument_candidates"])


def test_pointer_table_normalizes_thumb_function_pointers(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    target = _FakeFunction(0x401000, "handler")
    table = (0x401001).to_bytes(4, "little") + (0x402000).to_bytes(4, "little")

    class _ThumbTolerantBV(_FakeBV):
        def get_function_at(self, address: int):
            if int(address) == 0x401001:
                return target
            return super().get_function_at(address)

    bv = _ThumbTolerantBV(functions=[target], arch=_FakeArch(name="armv7"), memory={0x3000: table})
    # _pointer_table now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._pointer_table("active", "0x3000", entries=2)

    assert result["entries"][0]["value"] == "0x401001"
    assert result["entries"][0]["target"]["normalized"] == "0x401000"
    assert result["entries"][0]["target"]["thumb_adjusted"] is True
    assert result["entries"][0]["target"]["function"]["name"] == "handler"
    assert result["entries"][0]["target"]["function"]["exact_start"] is True
    assert result["entries"][0]["target"]["context"]["address"] == "0x401000"
    assert result["entries"][1]["target"]["function"] is None
    assert result["entries"][1]["target"]["plausible"] is False


def test_pointer_table_read_width_tracks_substride(monkeypatch):
    """`evidence table --stride 4` on 8-byte-aligned data must read 4 bytes wide,
    so odd slots read the (zero) high half instead of an 8-byte window overlapping
    the next pointer -> 0x40018000000000 garbage flagged [implausible] (#225)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    table = ((0x40c370).to_bytes(8, "little")
             + (0x400180).to_bytes(8, "little")
             + (0x40ca80).to_bytes(8, "little"))
    bv = _FakeBV(arch=_FakeArch(name="x86_64", address_size=8), memory={0x40b580: table})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._pointer_table("active", "0x40b580", entries=6, stride="4")
    assert result["read_width"] == 4
    vals = [e.get("value") for e in result["entries"]]
    assert vals[0] == "0x40c370" and vals[2] == "0x400180" and vals[4] == "0x40ca80"
    assert vals[1] == "0x0" and vals[3] == "0x0"            # zero high halves, not garbage
    assert "0x40018000000000" not in vals                   # no overlapping read

    # default (stride == pointer size) still reads 8-byte pointers
    r8 = instance._pointer_table("active", "0x40b580", entries=3, stride="8")
    assert r8["read_width"] == 8
    assert [e.get("value") for e in r8["entries"]] == ["0x40c370", "0x400180", "0x40ca80"]

    # explicit --width overrides the stride-derived width
    rw = instance._pointer_table("active", "0x40b580", entries=2, stride="8", width="4")
    assert rw["read_width"] == 4
    assert rw["entries"][0]["value"] == "0x40c370"


def test_message_lens_excludes_dynstr_and_resolves_rtti(monkeypatch):
    """On a symbol-retaining binary, evidence message must exclude the noisy
    .dynstr symbol-name matches and instead resolve the real RTTI data symbols
    (_ZTV/_ZTI/_ZTS<type>) with xrefs + a hint (#194)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    # the only string match is a .dynstr symbol-name string (noise)
    dynstr_str = _FakeStringRef(0x9000, 16, "_ZTVN5TCLAP3ArgE")
    # the real RTTI vtable data symbol the lens should surface
    vtable_sym = fake_bn.Symbol(fake_bn.SymbolType.DataSymbol, 0xA000, "_ZTVN5TCLAP3ArgE")
    vtable_sym.raw_name = "_ZTVN5TCLAP3ArgE"
    user = _FakeFunction(0x401000, "user")
    bv = _FakeBV(
        functions=[user],
        strings=[dynstr_str],
        symbols=[vtable_sym],
        sections={".dynstr": _FakeSection(".dynstr", 0x9000, 0x9100),
                  ".data.rel.ro": _FakeSection(".data.rel.ro", 0xA000, 0xB000)},
        code_refs={0xA000: [_FakeCodeRef(0x401010, user)]},
        segments={0x401010: _FakeSegment(readable=True, executable=True),
                  0xA000: _FakeSegment(readable=True, writable=True)},
        memory={0xA000: (0).to_bytes(8, "little") + (0xB100).to_bytes(8, "little")},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._message_lens("active", "N5TCLAP3ArgE", limit=5, table_entries=2)
    assert result["dynstr_excluded"] == 1
    assert result["count"] == 0                       # the lone .dynstr match was excluded
    rtti = result["rtti_symbols"]
    assert any(s["kind"] == "vtable" and s["symbol"] == "_ZTVN5TCLAP3ArgE" for s in rtti)
    vt = next(s for s in rtti if s["kind"] == "vtable")
    assert vt["address"] == "0xa000"
    assert len(vt["xrefs"]["code_refs"]) == 1
    assert result["hints"]                            # dynstr + rtti hints present


def test_pointer_table_does_not_thumb_normalize_non_arm_pointers(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    target = _FakeFunction(0x401000, "handler")
    table = (0x401001).to_bytes(4, "little")

    class _OddTolerantBV(_FakeBV):
        def get_function_at(self, address: int):
            if int(address) == 0x401001:
                return target
            return super().get_function_at(address)

    bv = _OddTolerantBV(functions=[target], arch=_FakeArch(name="x86"), memory={0x3000: table})
    # _pointer_table now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._pointer_table("active", "0x3000", entries=1)
    target_info = result["entries"][0]["target"]

    assert target_info["raw"] == "0x401001"
    assert target_info["normalized"] == "0x401001"
    assert target_info["thumb_adjusted"] is False
    assert target_info["function"]["name"] == "handler"
    assert target_info["function"]["exact_start"] is False
    assert target_info["function"]["offset"] == "0x1"
    assert target_info["context"]["address"] == "0x401001"
    assert any("inside functions" in warning for warning in result["warnings"])


def test_pointer_table_downgrades_inline_scalar_fields(monkeypatch):
    """A mixed record {function ptr, uint8 flag, ptr} read at a fixed stride must
    not count the inline scalar as a failed pointer resolution; only genuine
    pointer slots feed the 'do not resolve' warning (#170)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    handler = _FakeFunction(0x401000, "handler")
    # entry0: function pointer (plausible); entry1: a uint8 flag = 5 read as a
    # pointer-sized slot (inline scalar); entry2: a large unmapped value (a
    # genuine failed pointer slot that SHOULD still be counted).
    table = (
        (0x401000).to_bytes(4, "little")
        + (5).to_bytes(4, "little")
        + (0xDEADBEEF).to_bytes(4, "little")
    )
    bv = _FakeBV(functions=[handler], arch=_FakeArch(name="x86", address_size=4),
                 memory={0x3000: table})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._pointer_table("active", "0x3000", entries=3)
    rows = result["entries"]
    assert rows[0]["plausible"] is True
    assert rows[1]["likely_scalar"] is True and rows[1]["plausible"] is False
    assert rows[2]["likely_scalar"] is False and rows[2]["plausible"] is False
    warnings = " ".join(result["warnings"])
    assert "1 non-null entries do not resolve to mapped addresses" in warnings  # only entry2
    assert "inline scalar fields" in warnings                                   # entry1 noted



def test_function_evidence_marks_plt_stubs_as_thunk_candidates(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    plt = _FakeFunction(0x404020, "puts@plt")
    plt.basic_blocks = [_FakeBasicBlock(0x404020, 0x404026)]
    plt.low_level_il = [[_FakeLLILInstruction(0x404020, _FakeReg("rax"), operation="LLIL_JUMP")]]
    bv = _FakeBV(
        functions=[plt],
        sections={".plt.got": _FakeSection(".plt.got", 0x404020, 0x404100)},
        disassembly={0x404020: "jmp qword ptr [rip+0x2000]"},
    )
    # _function_evidence now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_evidence("active", "puts@plt", context=0)

    assert result["thunk"]["is_candidate"] is True
    assert "PLT" in result["thunk"]["reason"]


def test_pointer_table_warns_when_start_looks_like_code_not_table(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    memory = {0x64EA0: b"\xfb\x6b\xdb\xb2\x00\x2b\x00\xf0"}
    bv = _FakeBV(
        sections={".text": _FakeSection(".text", 0x64000, 0x65000)},
        segments={0x64EA0: _FakeSegment(readable=True, executable=True)},
        memory=memory,
    )
    # _pointer_table now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._pointer_table("active", "0x64ea0", entries=2)

    assert all(entry["plausible"] is False for entry in result["entries"])
    assert any("executable segment" in warning for warning in result["warnings"])
    assert any("low confidence" in warning for warning in result["warnings"])


def test_pointer_table_errors_on_unmapped_base(monkeypatch):
    """`evidence table` at an unmapped address must error like `bn read`, not
    return exit 0 with 16 fabricated readable:false slots and empty warnings
    (#119)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # Memory mapped elsewhere (so reads model real BN: b"" outside it); 0xdeadbeef
    # has no segment/section and is unreadable -> genuinely unmapped.
    bv = _FakeBV(memory={0x1000: b"\x00\x00\x00\x00"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(RuntimeError, match="0xdeadbeef.*not mapped"):
        instance._pointer_table("active", "0xdeadbeef", entries=16)


def test_pointer_table_for_view_warns_on_unmapped_base_without_erroring(monkeypatch):
    """The shared helper (used by message-lens / init-array windows) must FLAG
    an unmapped base instead of silently fabricating slots, but it must not
    abort the surrounding scan -- only the top-level command errors (#119)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(memory={0x1000: b"\x00\x00\x00\x00"})  # 0xdeadbeef unreadable -> unmapped
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    table = bridge.read_evidence._pointer_table_for_view(
        instance.ctx, bv, 0xDEADBEEF, entries=4, stride_size=4,
    )

    assert table["context"]["kind"] == "unmapped"
    assert any("unmapped" in warning for warning in table["warnings"])


def test_message_lens_summarizes_type_string_xrefs_and_metadata_window(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    builder = _FakeFunction(0x586A2, "build_type_name")
    memory = {0x6000: (0x586A3).to_bytes(4, "little") + (0x7000).to_bytes(4, "little")}
    bv = _FakeBV(
        functions=[builder],
        arch=_FakeArch(name="armv7"),
        strings=[_FakeStringRef(0x175B20, 19, "common.HeadUnitInfo")],
        code_refs={0x175B20: [_FakeCodeRef(0x586C0, builder)]},
        data_refs={0x175B20: [0x6008]},
        disassembly={0x586C0: "adr r1, common.HeadUnitInfo"},
        memory=memory,
    )
    # _message_lens now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._message_lens("active", "HeadUnitInfo", limit=5, table_entries=2)

    assert result["count"] == 1
    match = result["matches"][0]
    assert match["type_string"]["value"] == "common.HeadUnitInfo"
    assert match["xrefs"]["code_refs"][0]["function"] == "build_type_name"
    assert match["metadata_table_windows"][0]["address"] == "0x6000"
    assert match["metadata_table_windows"][0]["entries"][0]["target"]["thumb_adjusted"] is True
    # single match under the limit: honest total, not truncated
    assert result["total"] == 1
    assert result["truncated"] is False


def test_validate_count_enforces_minimum(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    # count flags require >= 1
    with pytest.raises(bridge.OperationFailure) as e:
        bridge._validate_count(0, label="limit", minimum=1)
    assert e.value.status == "invalid_request"
    with pytest.raises(bridge.OperationFailure):
        bridge._validate_count(-3, label="limit", minimum=1)
    # index flags allow 0 but reject negatives
    assert bridge._validate_count(0, label="offset", minimum=0) == 0
    with pytest.raises(bridge.OperationFailure):
        bridge._validate_count(-1, label="offset", minimum=0)
    # None handling and non-integer coercion
    assert bridge._validate_count(None, label="limit", minimum=1, allow_none=True) is None
    with pytest.raises(bridge.OperationFailure):
        bridge._validate_count(None, label="limit", minimum=1)  # allow_none=False
    with pytest.raises(bridge.OperationFailure):
        bridge._validate_count("abc", label="limit", minimum=1)


def test_bridge_ops_reject_out_of_range_count_params(monkeypatch):
    # Non-CLI callers (py exec / raw socket) reach the op handlers directly, so
    # the bridge must re-enforce the count/offset contract the CLI argparse
    # layer applies -- a negative/zero limit must not silently drop the tail or
    # return a degenerate empty-but-"truncated" result (#28).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # validation happens before _resolve_view, so no fake view is needed
    with pytest.raises(bridge.OperationFailure) as e:
        instance._message_lens("active", "x", limit=0)
    assert e.value.status == "invalid_request"
    with pytest.raises(bridge.OperationFailure):
        instance._list_functions("active", limit=-1)
    with pytest.raises(bridge.OperationFailure):
        instance._search_functions("active", "q", limit=-5)
    with pytest.raises(bridge.OperationFailure):
        instance._types("active", query=None, offset=0, limit=0)
    with pytest.raises(bridge.OperationFailure):
        instance._strings("active", query=None, offset=-1, limit=10)


def test_message_lens_reports_true_total_and_flags_truncation(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # 5 strings match the needle; with limit=2 only 2 rich matches come back,
    # but the reported total must be the honest 5 with truncated=True (issue #13).
    strings = [_FakeStringRef(0x1000 + i * 0x20, 9, f"Evt{i}_token") for i in range(5)]
    bv = _FakeBV(arch=_FakeArch(name="armv7"), strings=strings)
    # _message_lens now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._message_lens("active", "token", limit=2)

    assert result["count"] == 2          # only `limit` rich matches returned
    assert len(result["matches"]) == 2
    assert result["total"] == 5          # but the count reported is honest
    assert result["truncated"] is True


def test_message_lens_not_truncated_when_all_matches_fit(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    strings = [_FakeStringRef(0x1000 + i * 0x20, 9, f"Evt{i}_token") for i in range(3)]
    bv = _FakeBV(arch=_FakeArch(name="armv7"), strings=strings)
    # _message_lens now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._message_lens("active", "token", limit=20)

    assert result["count"] == result["total"] == 3
    assert result["truncated"] is False


def test_message_lens_metadata_window_stops_at_obvious_non_pointer(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    builder = _FakeFunction(0x586A2, "build_type_name")
    # First pointer resolves to code, then ASCII bytes "N6co" as little-endian
    # integer. The lens should include the bad entry as evidence and stop.
    memory = {
        0x6000: (
            (0x586A3).to_bytes(4, "little")
            + int.from_bytes(b"N6co", "little").to_bytes(4, "little")
            + int.from_bytes(b"mmon", "little").to_bytes(4, "little")
        )
    }
    bv = _FakeBV(
        functions=[builder],
        arch=_FakeArch(name="armv7"),
        strings=[_FakeStringRef(0x175BE4, 23, "N6common12HeadUnitInfoE")],
        data_refs={0x175BE4: [0x6008]},
        memory=memory,
    )
    # _message_lens now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._message_lens("active", "HeadUnitInfo", limit=5, table_entries=8)
    table = result["matches"][0]["metadata_table_windows"][0]

    assert len(table["entries"]) == 2
    assert table["entries"][1]["target"]["status"] == "unmapped"
    assert any("stopped after" in warning for warning in table["warnings"])


def test_bridge_handler_swallows_broken_pipe(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    warnings = []

    class _BrokenWriter:
        def write(self, data):
            raise BrokenPipeError(32, "Broken pipe")

    handler = bridge.BridgeHandler.__new__(bridge.BridgeHandler)
    handler.wfile = _BrokenWriter()
    monkeypatch.setattr(bridge.bn, "log_warn", lambda message: warnings.append(message))

    handler._write_response(b"{}", op="xrefs", request_id="req-123")

    assert warnings == [
        "BN Agent Bridge client disconnected before response could be delivered (op=xrefs, id=req-123)"
    ]


def test_bridge_handler_reraises_unrelated_write_errors(monkeypatch):
    bridge = _load_bridge(monkeypatch)

    class _FailingWriter:
        def write(self, data):
            raise OSError(5, "Input/output error")

    handler = bridge.BridgeHandler.__new__(bridge.BridgeHandler)
    handler.wfile = _FailingWriter()

    with pytest.raises(OSError, match="Input/output error"):
        handler._write_response(b"{}", op="xrefs")


def test_py_exec_non_serializable_result_falls_back_to_repr(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._py_exec("active", "result = object()")

    assert isinstance(result["result"], str)
    assert result["warnings"]


def test_diff_snapshots_marks_name_only_changes(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    diffs = instance._diff_snapshots(
        {
            0x401000: {
                "name": "sub_401000",
                "address": "0x401000",
                "text": "return 7;",
            }
        },
        {
            0x401000: {
                "name": "player_update",
                "address": "0x401000",
                "text": "return 7;",
            }
        },
    )

    assert len(diffs) == 1
    assert diffs[0]["changed"] is True
    assert diffs[0]["before_name"] == "sub_401000"
    assert diffs[0]["after_name"] == "player_update"
    assert diffs[0]["diff"] == "--- before:sub_401000\n+++ after:player_update"
    assert "before_excerpt" not in diffs[0]


def test_count_referenced_functions_is_uncapped_past_snapshot_cap(monkeypatch):
    """affected_functions is capped at 10 for snapshotting, but the reported
    blast radius (affected_summary.referenced) must be the true total -- a struct
    used by 200 functions previously surfaced as "10" with no hint of real scope."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx
    funcs = [_FakeFunction(0x1000 + i * 4, f"f{i}", "void(struct Widget* w)") for i in range(15)]
    bv = _FakeBV(functions=funcs)
    # Sidestep the C parser: the type resolution is exercised elsewhere.
    monkeypatch.setattr(me, "_operation_type_names", lambda c, b, op: ["Widget"])
    ops = [{"op": "types_declare", "declaration": "struct Widget { int x; };"}]

    assert len(me._guess_affected_functions(ctx, bv, ops)) == 10  # snapshot set, capped
    assert me._count_referenced_functions(ctx, bv, ops, fallback=10) == 15  # true total


def test_count_referenced_functions_falls_back_on_scan_error(monkeypatch):
    """A stubbed/odd view must never crash a mutation: the count degrades to the
    capped fallback rather than raising."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    ctx = bridge.BinaryNinjaBridge().ctx

    def _boom(*a, **k):
        raise RuntimeError("view scan blew up")

    monkeypatch.setattr(me, "_functions_for_op", _boom)
    assert me._count_referenced_functions(ctx, _FakeBV(), [{"op": "set_prototype"}], fallback=4) == 4


def test_mutation_mixed_batch_scopes_blast_radius_and_tags_direct(monkeypatch):
    """A mixed batch (types_declare + set_prototype) scopes the blast radius to
    the TYPE op and tags the direct op's affected function `direct`, so the
    set_prototype target is excluded from the type's referenced/reflowed counts
    and the formatter can keep the two apart (Codex review on #240)."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    uses_ep = _FakeFunction(0x10, "uses_ep", "void()")      # references the type
    handler = _FakeFunction(0x401000, "handler", "void()")  # set_prototype target

    def _fns_for_op(ctx, b, op, *, type_limit):
        return [uses_ep] if me._is_type_op(op) else [handler]

    diffs = [
        {"address": "0x10", "before_name": "uses_ep", "after_name": "uses_ep", "changed": True, "diff": ""},
        {"address": "0x401000", "before_name": "handler", "after_name": "handler", "changed": True, "diff": ""},
    ]

    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(me, "_functions_for_op", _fns_for_op)
    monkeypatch.setattr(me, "_guess_affected_functions", lambda ctx, b, ops: [])
    monkeypatch.setattr(me, "_capture_function_snapshots", lambda ctx, b, fns: {})
    monkeypatch.setattr(me, "_capture_type_snapshots", lambda ctx, b, ops: {})
    monkeypatch.setattr(me, "_diff_snapshots", lambda ctx, before, after: [dict(d) for d in diffs])
    monkeypatch.setattr(me, "_diff_type_snapshots", lambda ctx, before, after: [{"type_name": "Ep", "changed": True}])
    monkeypatch.setattr(me, "_apply_operation", lambda ctx, b, op, restores=None: {"op": op.get("op")})
    monkeypatch.setattr(me, "_verify_operation", lambda ctx, b, result: {**result, "status": "verified"})
    monkeypatch.setattr(me, "_annotate_operation_results", lambda ctx, results, type_diffs: results)

    result = instance._mutation("active", False,
                                [{"op": "types_declare"}, {"op": "set_prototype"}])

    assert result["success"] is True
    assert ("commit", "state") in bv.events
    # Blast radius counts the type's reach only -- handler (direct) is excluded.
    assert result["affected_summary"] == {"referenced": 1, "reflowed": 1}
    tags = {d["address"]: d["direct"] for d in result["affected_functions"]}
    assert tags == {"0x10": False, "0x401000": True}


def test_slim_type_result_drops_redundant_layouts(monkeypatch):
    """A verified types_declare result echoes the layout under defined_type_layouts
    AND observed.defined_type_layouts, duplicating affected_types[].after_layout.
    The output slim drops both heavy copies but keeps the short decl strings."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    layout = "struct Widget // size=0x4\n0x0000: int32_t x"
    result = {
        "op": "types_declare",
        "defined_types": {"Widget": "struct Widget"},
        "defined_type_layouts": {"Widget": layout},
        "observed": {"defined_types": {"Widget": "struct Widget"}, "defined_type_layouts": {"Widget": layout}},
    }
    slim = me._slim_type_result_for_output(result)
    assert "defined_type_layouts" not in slim
    assert "defined_type_layouts" not in slim["observed"]
    assert slim["defined_types"] == {"Widget": "struct Widget"}  # short decl kept
    assert slim["observed"]["defined_types"] == {"Widget": "struct Widget"}
    assert "defined_type_layouts" in result  # original untouched (copy, not mutate)

    other = {"op": "set_prototype", "observed": {"prototype": "void()"}}
    assert me._slim_type_result_for_output(other) is other  # non-type op passes through


def test_diff_snapshots_omits_excerpt_when_full_diff_fits(monkeypatch):
    """A small real body change: the full unified diff fits inline, so the focused
    before/after_excerpt would only duplicate it. The excerpt is reserved for the
    large-function case where the diff gets truncated (see the M14 test above)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    diffs = instance._diff_snapshots(
        {0x1000: {"text": "x = 1;\ny = prev;\nz = 3;", "name": "f"}},
        {0x1000: {"text": "x = 1;\ny = next;\nz = 3;", "name": "f"}},
    )
    d = diffs[0]
    assert d["changed"] is True
    assert "prev" in d["diff"] and "next" in d["diff"]  # change visible inline
    assert "before_excerpt" not in d and "after_excerpt" not in d


def test_diff_snapshots_marks_comment_only_change(monkeypatch):
    """A comment set/delete changes no HLIL body text, so a text-only snapshot
    reports changed:false with an empty diff. The diff/changed signal must also
    reflect comment state (#121)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    diffs = instance._diff_snapshots(
        {0x401000: {"name": "f", "address": "0x401000", "text": "return 7;", "comments": {}, "locals": {}}},
        {0x401000: {"name": "f", "address": "0x401000", "text": "return 7;",
                    "comments": {"0x401010": "decryption key"}, "locals": {}}},
    )

    assert len(diffs) == 1
    assert diffs[0]["changed"] is True
    assert diffs[0]["diff"]
    assert "decryption key" in diffs[0]["diff"]


def test_diff_snapshots_marks_local_only_change(monkeypatch):
    """A local rename/retype of a variable not rendered in the HLIL body leaves
    the body text identical; the diff/changed signal must reflect local
    name/type state too (#121)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    diffs = instance._diff_snapshots(
        {0x401000: {"name": "f", "address": "0x401000", "text": "return arg1;",
                    "comments": {}, "locals": {"1": "arg1:int32_t"}}},
        {0x401000: {"name": "f", "address": "0x401000", "text": "return arg1;",
                    "comments": {}, "locals": {"1": "session_id:int32_t"}}},
    )

    assert len(diffs) == 1
    assert diffs[0]["changed"] is True
    assert diffs[0]["diff"]
    assert "session_id" in diffs[0]["diff"]


def test_capture_function_snapshots_reads_global_comment_store(monkeypatch):
    """Comment ops write to BN's GLOBAL comment store (bv.set_comment_at /
    bv.address_comments), which is a DIFFERENT store from Function.comments.
    The snapshot must read the global store filtered to the function -- reading
    Function.comments (as a first cut did) sees nothing the op wrote and the
    comment --preview still shows changed:false (#121)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(bridge.il_format, "_function_text", lambda bv, fn, view="hlil": "body")
    fn = _FakeFunction(0x401000, "f")
    fn.basic_blocks = [_FakeBasicBlock(0x401000, 0x401020)]
    # Function.comments is the WRONG (function-local) store -- it must be ignored.
    fn.comments = {0x401000: "stale-local-store"}
    bv = _FakeMutationBV()
    bv.functions = [fn]
    # Where bv.set_comment_at actually lands: in-function + an out-of-function one.
    bv.address_comments = {0x401004: "decryption key", 0x500000: "outside"}

    snaps = instance._capture_function_snapshots(bv, [fn])

    assert snaps[0x401000]["comments"] == {"0x401004": "decryption key"}
    assert snaps[0x401000]["locals"] == {}


def test_read_write_lock_blocks_reader_until_writer_releases(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    lock = bridge._ReadWriteLock()
    writer_ready = threading.Event()
    writer_release = threading.Event()
    reader_entered = threading.Event()

    def writer():
        with lock.write():
            writer_ready.set()
            writer_release.wait(1)

    def reader():
        writer_ready.wait(1)
        with lock.read():
            reader_entered.set()

    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    writer_thread.start()
    reader_thread.start()

    assert writer_ready.wait(1)
    time.sleep(0.05)
    assert not reader_entered.is_set()

    writer_release.set()
    reader_thread.join(1)
    writer_thread.join(1)

    assert reader_entered.is_set()


def test_read_write_lock_allows_parallel_readers(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    lock = bridge._ReadWriteLock()
    entered: list[str] = []
    both_entered = threading.Event()
    release = threading.Event()

    def reader(name: str):
        with lock.read():
            entered.append(name)
            if len(entered) == 2:
                both_entered.set()
            release.wait(1)

    first = threading.Thread(target=reader, args=("first",))
    second = threading.Thread(target=reader, args=("second",))
    first.start()
    second.start()

    assert both_entered.wait(1)

    release.set()
    first.join(1)
    second.join(1)

    assert sorted(entered) == ["first", "second"]


def test_read_write_lock_waiting_writer_blocks_new_readers(monkeypatch):
    """A reader arriving while a writer is queued must not jump the queue,
    otherwise a steady reader stream starves the writer forever."""
    bridge = _load_bridge(monkeypatch)
    lock = bridge._ReadWriteLock()
    order: list[str] = []
    first_reader_in = threading.Event()
    first_reader_release = threading.Event()
    writer_done = threading.Event()
    second_reader_done = threading.Event()

    def first_reader():
        with lock.read():
            first_reader_in.set()
            first_reader_release.wait(2)

    def writer():
        with lock.write():
            order.append("writer")
        writer_done.set()

    def second_reader():
        with lock.read():
            order.append("reader")
        second_reader_done.set()

    t_first = threading.Thread(target=first_reader)
    t_first.start()
    assert first_reader_in.wait(1)

    t_writer = threading.Thread(target=writer)
    t_writer.start()
    # Wait until the writer is actually queued behind the active reader.
    deadline = time.monotonic() + 2
    while lock._writers_waiting == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert lock._writers_waiting == 1

    t_second = threading.Thread(target=second_reader)
    t_second.start()
    # The second reader must not enter while the writer is waiting.
    time.sleep(0.05)
    assert order == []

    first_reader_release.set()
    assert writer_done.wait(2)
    assert second_reader_done.wait(2)
    t_first.join(1)
    t_writer.join(1)
    t_second.join(1)

    assert order == ["writer", "reader"]


def test_collect_open_views_uses_tabs_api(monkeypatch):
    bridge = _load_bridge(monkeypatch)

    class _View:
        def __init__(self, data):
            self._data = data

        def getData(self):
            return self._data

    class _Frame:
        def __init__(self, data):
            self._data = data

        def getCurrentBinaryView(self):
            return self._data

        def getCurrentView(self):
            return _View(self._data)

    view_a = object()
    view_b = object()
    view_c = object()

    class _Context:
        def getCurrentViewFrame(self):
            return _Frame(view_c)

        def getTabs(self):
            return ["tab-a", "tab-b", "tab-c"]

        def getViewFrameForTab(self, tab):
            mapping = {
                "tab-a": _Frame(view_a),
                "tab-b": _Frame(view_b),
                "tab-c": _Frame(view_c),
            }
            return mapping[tab]

        def getViewForTab(self, tab):
            mapping = {
                "tab-a": _View(view_a),
                "tab-b": _View(view_b),
                "tab-c": _View(view_c),
            }
            return mapping[tab]

    fake_ui = types.SimpleNamespace(
        UIContext=types.SimpleNamespace(
            allContexts=lambda: [_Context()],
            activeContext=lambda: None,
        )
    )
    monkeypatch.setattr(bridge, "ui", fake_ui)

    views = bridge._collect_open_views()

    assert len(views) == 3
    assert set(id(view) for view in views) == {id(view_a), id(view_b), id(view_c)}


# --- I2: strings filtering ---


def test_strings_min_length_excludes_short_strings(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(strings=[
        _FakeStringRef(0x1000, 2, "ab"),
        _FakeStringRef(0x2000, 5, "hello"),
        _FakeStringRef(0x3000, 10, "helloworld"),
    ])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query=None, offset=0, limit=100, min_length=4)

    items = result["items"]
    assert len(items) == 2
    assert result["total"] == 2
    assert items[0]["value"] == "hello"
    assert items[1]["value"] == "helloworld"


def test_strings_section_filter_keeps_only_matching_section(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        strings=[
            _FakeStringRef(0x1000, 4, "code"),
            _FakeStringRef(0x5000, 6, "rodata"),
        ],
        sections={
            ".text": _FakeSection(".text", 0x1000, 0x2000),
            ".rodata": _FakeSection(".rodata", 0x5000, 0x6000),
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query=None, offset=0, limit=100, section=".rodata")

    items = result["items"]
    assert len(items) == 1
    assert items[0]["value"] == "rodata"


def test_strings_no_crt_excludes_locale_and_text_section(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        strings=[
            _FakeStringRef(0x1000, 2, "en"),           # locale code
            _FakeStringRef(0x2000, 5, "en-US"),         # locale code
            _FakeStringRef(0x3000, 3, "Mon"),           # day abbreviation
            _FakeStringRef(0x4000, 5, "UTF-8"),         # encoding name
            _FakeStringRef(0x5000, 6, "player"),        # real string
            _FakeStringRef(0x6000, 4, "data"),          # in .text section
        ],
        sections={
            ".text": _FakeSection(".text", 0x6000, 0x7000),
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query=None, offset=0, limit=100, no_crt=True)

    items = result["items"]
    assert len(items) == 1
    assert items[0]["value"] == "player"


def test_strings_filters_combine(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        strings=[
            _FakeStringRef(0x5000, 2, "ab"),            # too short
            _FakeStringRef(0x5001, 6, "player"),        # passes all
            _FakeStringRef(0x1000, 6, "system"),        # wrong section
            _FakeStringRef(0x5002, 5, "en-US"),         # CRT locale
        ],
        sections={
            ".text": _FakeSection(".text", 0x1000, 0x2000),
            ".rodata": _FakeSection(".rodata", 0x5000, 0x6000),
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query=None, offset=0, limit=100,
                               min_length=4, section=".rodata", no_crt=True)

    items = result["items"]
    assert len(items) == 1
    assert items[0]["value"] == "player"


def test_strings_regex_search_matches_or_patterns(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        strings=[
            _FakeStringRef(0x5000, 7, "Vehicle"),
            _FakeStringRef(0x5010, 12, "HeadUnitInfo"),
            _FakeStringRef(0x5020, 4, "skip"),
        ]
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query="vehicle|headunit", offset=0, limit=100, regex=True)

    assert [item["value"] for item in result["items"]] == ["Vehicle", "HeadUnitInfo"]


def test_strings_invalid_regex_is_actionable(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: _FakeBV())

    with pytest.raises(bridge.OperationFailure, match="Invalid string regex"):
        instance._strings(None, query="(", offset=0, limit=100, regex=True)


# --- I5: sections ---


def test_sections_returns_all_sections_with_permissions(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        sections={
            ".text": _FakeSection(".text", 0x1000, 0x5000, semantics=1),
            ".data": _FakeSection(".data", 0x5000, 0x6000, semantics=3),
            ".rodata": _FakeSection(".rodata", 0x6000, 0x7000, semantics=2),
        },
        segments={
            0x1000: _FakeSegment(readable=True, writable=False, executable=True),
            0x5000: _FakeSegment(readable=True, writable=True, executable=False),
            0x6000: _FakeSegment(readable=True, writable=False, executable=False),
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._sections(None)

    items = result["items"]
    assert len(items) == 3
    assert result["total"] == 3
    text_sec = items[0]
    assert text_sec["name"] == ".text"
    assert text_sec["start"] == "0x1000"
    assert text_sec["end"] == "0x5000"
    assert text_sec["length"] == 0x4000
    assert text_sec["semantics"] == "ReadOnlyCode"
    assert text_sec["readable"] is True
    assert text_sec["writable"] is False
    assert text_sec["executable"] is True

    data_sec = items[1]
    assert data_sec["name"] == ".data"
    assert data_sec["semantics"] == "ReadWriteData"
    assert data_sec["writable"] is True

    rodata_sec = items[2]
    assert rodata_sec["name"] == ".rodata"
    assert rodata_sec["semantics"] == "ReadOnlyData"
    assert rodata_sec["executable"] is False


def test_init_arrays_summarizes_constructor_pointer_sections(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    ctor = _FakeFunction(0x401000, "global_ctor")
    table = (0x401001).to_bytes(4, "little") + (0x402000).to_bytes(4, "little")
    bv = _FakeBV(
        functions=[ctor],
        arch=_FakeArch(name="armv7"),
        sections={
            ".init_array": _FakeSection(".init_array", 0x5000, 0x5008),
            ".data": _FakeSection(".data", 0x6000, 0x6010),
        },
        memory={0x5000: table},
    )
    # _init_arrays now resolves the view through the BridgeContext seam
    # (read_evidence), so patch the moved free function's resolution path.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._init_arrays("active", limit=4)

    assert result["pointer_size"] == 4
    assert len(result["sections"]) == 1
    section = result["sections"][0]
    assert section["name"] == ".init_array"
    assert section["total_entries"] == 2
    assert section["table"]["entries"][0]["target"]["function"]["name"] == "global_ctor"


def test_sections_query_filters_by_name(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        sections={
            ".text": _FakeSection(".text", 0x1000, 0x5000),
            ".rodata": _FakeSection(".rodata", 0x5000, 0x6000),
            ".data": _FakeSection(".data", 0x6000, 0x7000),
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._sections(None, query="data")

    items = result["items"]
    assert len(items) == 2
    names = [s["name"] for s in items]
    assert ".rodata" in names
    assert ".data" in names


def test_sections_query_matches_semantics_not_just_name(monkeypatch):
    # #257: --query should match the semantics label too, so `--query code`
    # finds executable sections (.text = ReadOnlyCode) even though "code" is
    # not in the section name. Match is case-insensitive.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        sections={
            ".text": _FakeSection(".text", 0x1000, 0x5000, semantics=1),     # ReadOnlyCode
            ".rodata": _FakeSection(".rodata", 0x5000, 0x6000, semantics=2),  # ReadOnlyData
            ".data": _FakeSection(".data", 0x6000, 0x7000, semantics=3),      # ReadWriteData
        },
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._sections(None, query="code")

    names = [s["name"] for s in result["items"]]
    assert names == [".text"]


def test_sections_query_semantics_match_is_case_insensitive(monkeypatch):
    # An uppercase query still matches the (CamelCase) semantics label.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(sections={".text": _FakeSection(".text", 0x1000, 0x5000, semantics=1)})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._sections(None, query="CODE")

    assert [s["name"] for s in result["items"]] == [".text"]


def test_sections_query_count_only_reflects_semantics_match(monkeypatch):
    # The count_only path must reflect the broader name-or-semantics match, not
    # the old name-only filter.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(sections={
        ".text": _FakeSection(".text", 0x1000, 0x5000, semantics=1),     # ReadOnlyCode
        ".rodata": _FakeSection(".rodata", 0x5000, 0x6000, semantics=2),  # ReadOnlyData
    })
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._sections(None, query="code", count_only=True)

    assert result["count"] == 1 and result["total"] == 1


def test_sections_query_semantics_broadens_beyond_name(monkeypatch):
    # Intentional #257 behavior change (pinned so it isn't read as a no-side-
    # effect bugfix): `--query data` now matches any section whose SEMANTICS is
    # ReadOnlyData/ReadWriteData even when "data" is absent from the name.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(sections={
        ".text": _FakeSection(".text", 0x1000, 0x5000, semantics=1),    # ReadOnlyCode
        ".roseg": _FakeSection(".roseg", 0x5000, 0x6000, semantics=2),  # ReadOnlyData, no "data" in name
        ".rwseg": _FakeSection(".rwseg", 0x6000, 0x7000, semantics=3),  # ReadWriteData, no "data" in name
    })
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._sections(None, query="data")

    assert sorted(s["name"] for s in result["items"]) == [".roseg", ".rwseg"]


def test_sections_null_segment_omits_rwx(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        sections={".bss": _FakeSection(".bss", 0x9000, 0xa000)},
        segments={0x1000: _FakeSegment(readable=True, writable=False, executable=True)},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._sections(None)

    items = result["items"]
    assert len(items) == 1
    assert "readable" not in items[0]


def test_sections_without_segments_omits_rwx(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _BareView:
        def __init__(self):
            self.sections = {".text": _FakeSection(".text", 0x1000, 0x2000)}

    bv = _BareView()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._sections(None)

    items = result["items"]
    assert len(items) == 1
    assert "readable" not in items[0]
    assert "writable" not in items[0]
    assert "executable" not in items[0]


# --- I8: enhanced imports ---


def test_imports_includes_function_data_and_address_symbols(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    func_sym = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x1000, "printf")
    func_sym.short_name = "printf"
    func_sym.namespace = "libc"

    data_sym = fake_bn.Symbol(fake_bn.SymbolType.ImportedDataSymbol, 0x2000, "__stdout")
    data_sym.short_name = "__stdout"
    data_sym.namespace = "libc"

    addr_sym = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x3000, "iat_entry")
    addr_sym.short_name = "iat_entry"
    addr_sym.namespace = ""

    bv = _FakeBV(symbols=[func_sym, data_sym, addr_sym])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)

    items = result["items"]
    assert len(items) == 3
    assert result["total"] == 3
    kinds = {item["name"]: item["kind"] for item in items}
    assert kinds["printf"] == "function"
    assert kinds["__stdout"] == "data"
    assert kinds["iat_entry"] == "address"


def test_imports_excludes_pic_self_defined_exports(monkeypatch):
    """On a PIC .so BN models the lib's own defined+exported functions as import
    veneers (ImportedFunctionSymbol) plus their GOT slots (ImportAddressSymbol),
    even though they're defined in the same module. These self-references must be
    excluded from the imports survey (and counted, not silently dropped), leaving
    only genuine external dependencies (#202)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    NAME = "_ZN5boost6system15system_categoryEv"
    # the real, DEFINED export (function body in this module)
    defined = fake_bn.Symbol(fake_bn.SymbolType.FunctionSymbol, 0x401c90, NAME)
    defined.short_name = NAME
    # BN's PIC self-references: an import veneer + a GOT slot, both library:null
    veneer = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x401980, NAME)
    veneer.short_name = NAME
    veneer.namespace = "BNINTERNALNAMESPACE"
    got = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x413f58, NAME)
    got.short_name = NAME
    got.namespace = "BNINTERNALNAMESPACE"
    # a genuine external import (no in-module definition)
    ext = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x401000, "memcpy")
    ext.short_name = "memcpy"
    ext.namespace = ""

    bv = _FakeBV(symbols=[defined, veneer, got, ext])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)
    names = [it["name"] for it in result["items"]]
    assert names == ["memcpy"]                    # only the genuine import remains
    assert result["total"] == 1
    assert result["self_defined_excluded"] == 2   # veneer + GOT slot, visibly dropped

    # the summary path excludes them too, and surfaces the count
    summary = instance._imports(None, summary=True)
    assert summary["total_symbols"] == 1
    assert summary["self_defined_excluded"] == 2


def test_imports_collapses_got_slot_duplicates(monkeypatch):
    """Each import appears as a function/data PLT entry AND an (address) GOT slot
    of the same name -- ~half the list is dups. Collapse the GOT slots by default;
    --include-got shows them (#212)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    fn = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x1000, "memcpy")
    fn.short_name = "memcpy"; fn.namespace = ""
    got = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x2000, "memcpy")  # GOT slot dup
    got.short_name = "memcpy"; got.namespace = ""
    uniq = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x3000, "__gmon_start__")
    uniq.short_name = "__gmon_start__"; uniq.namespace = ""
    bv = _FakeBV(symbols=[fn, got, uniq])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)
    names = [(it["name"], it["kind"]) for it in result["items"]]
    assert ("memcpy", "function") in names
    assert ("memcpy", "address") not in names         # GOT dup collapsed
    assert ("__gmon_start__", "address") in names     # genuinely-unique address kept
    assert result["got_collapsed"] == 1

    full = instance._imports(None, include_got=True)
    assert ("memcpy", "address") in [(it["name"], it["kind"]) for it in full["items"]]
    assert "got_collapsed" not in full


def test_imports_surfaces_et_rel_external_symbols(monkeypatch):
    """An ET_REL .ko has no Imported* symbols; its kernel-API refs are
    ExternalSymbols (.extern). `bn imports` must surface them (#213)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    printk = fake_bn.Symbol(fake_bn.SymbolType.ExternalSymbol, 0x0, "printk")
    printk.short_name = "printk"; printk.namespace = ""
    kmalloc = fake_bn.Symbol(fake_bn.SymbolType.ExternalSymbol, 0x8, "__kmalloc")
    kmalloc.short_name = "__kmalloc"; kmalloc.namespace = ""
    bv = _FakeBV(symbols=[printk, kmalloc])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)
    names = {it["name"]: it["kind"] for it in result["items"]}
    assert names == {"printk": "external", "__kmalloc": "external"}


def test_imports_external_dedups_against_got_only_imports(monkeypatch):
    """On a standard ELF most imports are GOT-only (ImportAddressSymbol, no PLT
    function entry) and the SAME names reappear as ExternalSymbol. The external
    dedup must skip them so they aren't double-listed (address + external) --
    reintroducing #212. The dedup is against ALL emitted names, not just
    function/data (#213 review)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    # memcpy: GOT-only import (address kind, no function twin) + an external dup
    got = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x1000, "memcpy")
    got.short_name = "memcpy"; got.namespace = ""
    ext = fake_bn.Symbol(fake_bn.SymbolType.ExternalSymbol, 0x0, "memcpy")
    ext.short_name = "memcpy"; ext.namespace = ""
    # a genuinely .ko-style external with no import twin -> must still appear
    only_ext = fake_bn.Symbol(fake_bn.SymbolType.ExternalSymbol, 0x8, "printk")
    only_ext.short_name = "printk"; only_ext.namespace = ""
    bv = _FakeBV(symbols=[got, ext, only_ext])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)
    rows = [(it["name"], it["kind"]) for it in result["items"]]
    assert ("memcpy", "address") in rows
    assert ("memcpy", "external") not in rows         # deduped against the address entry
    assert ("printk", "external") in rows             # unique external still listed
    assert result["total"] == 2                        # not 3 (no double-list)


def test_exports_lists_global_weak_definitions_only(monkeypatch):
    """`bn exports` lists the global/weak DEFINITIONS (the public API); local
    definitions are internal and excluded (#198)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    pub = fake_bn.Symbol(fake_bn.SymbolType.FunctionSymbol, 0x1000, "_ZN3foo3bar4recvEi",
                         binding=fake_bn.SymbolBinding.GlobalBinding)
    pub.short_name = "foo::bar::recv"
    weak = fake_bn.Symbol(fake_bn.SymbolType.DataSymbol, 0x2000, "g_table",
                          binding=fake_bn.SymbolBinding.WeakBinding)
    weak.short_name = "g_table"
    local = fake_bn.Symbol(fake_bn.SymbolType.FunctionSymbol, 0x3000, "helper",
                           binding=fake_bn.SymbolBinding.LocalBinding)
    local.short_name = "helper"
    bv = _FakeBV(symbols=[pub, weak, local])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._exports(None)
    names = {it["name"]: it for it in result["items"]}
    assert set(names) == {"_ZN3foo3bar4recvEi", "g_table"}       # local excluded
    assert names["_ZN3foo3bar4recvEi"]["display_name"] == "foo::bar::recv"  # demangled
    assert names["_ZN3foo3bar4recvEi"]["kind"] == "function"
    assert names["g_table"]["kind"] == "data"
    assert instance._exports(None, count_only=True)["count"] == 2


def test_xrefs_any_marks_ambiguous_symbol_present(monkeypatch):
    """In a sink sweep an AMBIGUOUS symbol (resolves to >=2 bodies) must be
    reported present (it exists), not absent -- otherwise a real sink reads as
    unlinked (#218 review)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x401000, "dup"), _FakeFunction(0x402000, "dup")])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    res = instance._xrefs_any(None, ["dup", "nope"])
    syms = {s["symbol"]: s for s in res["symbols"]}
    assert syms["dup"]["present"] is True and syms["dup"].get("ambiguous") is True
    assert syms["nope"]["present"] is False
    assert res["present"] == 1


def test_save_database_falls_back_to_writable_cache(monkeypatch, tmp_path):
    """A default-path save whose directory is unwritable (read-only firmware
    mount) falls back to a writable cache dir instead of losing annotations;
    an EXPLICIT --path failure stays a hard error (#214)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(bridge, "cache_home", lambda: tmp_path / "cache")

    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    ro_file = str(ro_dir / "firmware.bin")
    ro_bndb = ro_file + ".bndb"
    created: list[str] = []

    class _SaveBV:
        class file:
            filename = ro_file

        def create_database(self, dest):
            if str(dest) == ro_bndb:
                return False                       # simulate an unwritable default dir
            from pathlib import Path as _P
            _P(dest).parent.mkdir(parents=True, exist_ok=True)
            _P(dest).write_bytes(b"BNDB")
            created.append(str(dest))
            return True

    bv = _SaveBV()
    monkeypatch.setattr(instance.targets, "resolve", lambda sel: bv)
    monkeypatch.setattr(instance.targets, "clear_dirty", lambda b: None)

    result = instance._save_database(None)
    assert result["saved"] is True
    assert result["fallback"] is True
    assert result["requested_path"] == ro_bndb
    assert "cache" in result["path"] and result["path"].endswith(".bndb")
    assert created == [result["path"]]

    # explicit --path failure must NOT silently relocate
    with pytest.raises(RuntimeError, match="no file was written"):
        instance._save_database(None, path=ro_bndb)


def test_function_list_carries_demangled_display_name(monkeypatch):
    """function list entries carry a demangled display_name (#196)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "_ZN3foo3bar4recvEi")
    sym = _FakeSymbol("FunctionSymbol")
    sym.short_name = "foo::bar::recv"
    fn.symbol = sym
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._list_functions(None)
    item = result["items"][0]
    assert item["name"] == "_ZN3foo3bar4recvEi"
    assert item["display_name"] == "foo::bar::recv"


def test_imports_sorts_function_kind_first_then_library_name(monkeypatch):
    # Imports order by kind usefulness (function -> data -> address) first, then
    # library/name (#07). The data symbol here has the alphabetically-EARLIER
    # library, yet the function still sorts first -- kind dominates library.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    sym_b = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x2000, "zebra")
    sym_b.short_name = "zebra"
    sym_b.namespace = "libz"

    sym_a = fake_bn.Symbol(fake_bn.SymbolType.ImportedDataSymbol, 0x1000, "alpha")
    sym_a.short_name = "alpha"
    sym_a.namespace = "liba"

    bv = _FakeBV(symbols=[sym_b, sym_a])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)

    items = result["items"]
    assert items[0]["name"] == "zebra" and items[0]["kind"] == "function"
    assert items[0]["library"] == "libz"
    assert items[1]["name"] == "alpha" and items[1]["kind"] == "data"
    assert items[1]["library"] == "liba"


def test_imports_bn_sentinel_namespace_is_not_surfaced_as_library(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    sym = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x1000, "memcpy")
    sym.short_name = "memcpy"
    sym.namespace = "BNINTERNALNAMESPACE"

    bv = _FakeBV(symbols=[sym])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)

    item = result["items"][0]
    # The meaningless sentinel must not masquerade as a real library...
    assert item["library"] is None
    # ...but stays available under an honestly-named field.
    assert item["namespace"] == "BNINTERNALNAMESPACE"


def test_imports_summary_includes_needed_libraries(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    sym = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x1000, "memcpy")
    sym.short_name = "memcpy"
    sym.namespace = "BNEXTERNALNAMESPACE"

    bv = _FakeBV(symbols=[sym])
    bv.libraries = ["libssl.so.1.1", "libc.so.6"]
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None, summary=True)

    # DT_NEEDED is the real dependency signal, sorted and de-duped.
    assert result["needed_libraries"] == ["libc.so.6", "libssl.so.1.1"]
    # namespace grouping still reflects the raw BN namespace.
    assert result["namespaces"] == {"BNEXTERNALNAMESPACE": 1}


# --- read: raw bytes at an address ---


def test_read_returns_hex_and_ascii_for_mapped_address(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(memory={0x1000: b"\x48\x65\x6c\x6c\x6f\x00\x90\xff"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._read(None, "0x1000", 8)

    assert result["address"] == "0x1000"
    assert result["length"] == 8
    assert result["hex"] == "48656c6c6f0090ff"
    assert result["ascii"] == "Hello..."
    assert "short_read" not in result
    assert "note" not in result


def test_read_accepts_decimal_address(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(memory={0x1000: b"\x41\x42\x43\x44"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._read(None, "4096", 4)

    assert result["address"] == "0x1000"
    assert result["hex"] == "41424344"
    assert result["ascii"] == "ABCD"


def test_read_unmapped_address_raises_naming_the_address(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(memory={0x1000: b"\x41\x42\x43\x44"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(RuntimeError, match="0xdead.*not mapped"):
        instance._read(None, "0xdead", 16)


def test_read_short_read_returns_mapped_bytes_with_note(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(memory={0x1000: b"\x01\x02\x03\x04"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._read(None, "0x1000", 16)

    assert result["length"] == 4
    assert result["hex"] == "01020304"
    assert result["short_read"] is True
    assert result["requested_length"] == 16
    assert "short read" in result["note"]
    assert "0x1000" in result["note"]


# --- function create: create+analyze a missed function ---


class _FakeFunctionCreateBV(_FakeMutationBV):
    def __init__(self, *, functions=None, segments=None, memory=None):
        super().__init__()
        self.functions = list(functions or [])
        self._segments = dict(segments or {})
        self._memory = dict(memory or {})
        self.added: list[int] = []

    def add_function(self, addr: int):
        self.events.append(("add_function", addr))
        self.added.append(int(addr))
        fn = _FakeFunction(int(addr), f"sub_{addr:x}")
        self.functions.append(fn)
        return fn

    def remove_user_function(self, fn):
        # Model BN faithfully: add_function is NOT journaled by the undo buffer,
        # so revert_undo_actions never removes it -- only remove_user_function
        # does. The preview/rollback revert must call this explicitly (#117).
        self.events.append(("remove_user_function", int(fn.start)))
        self.functions = [f for f in self.functions if int(f.start) != int(fn.start)]

    def is_offset_executable(self, addr: int) -> bool:
        seg = self.get_segment_at(int(addr))
        return bool(seg is not None and seg.executable)


def test_function_create_at_executable_address_returns_verified(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_create(None, "0x1000", False)

    assert result["success"] is True
    assert result["committed"] is True
    assert bv.added == [0x1000]
    assert "refresh" in bv.events
    assert ("commit", "state") in bv.events
    res = result["results"][0]
    assert res["op"] == "function_create"
    assert res["status"] == "verified"
    assert res["address"] == "0x1000"
    assert res["function"] == "sub_1000"
    assert result["affected_functions"][0]["after_name"] == "sub_1000"


def test_function_create_preview_reverts_without_committing(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_create(None, "0x1000", True)

    assert result["success"] is True
    assert result["committed"] is False
    assert result["results"][0]["status"] == "verified"
    assert ("revert", "state") in bv.events
    assert ("commit", "state") not in bv.events


def test_function_create_existing_function_is_noop(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        functions=[_FakeFunction(0x1000, "player_update")],
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_create(None, "0x1000", False)

    assert result["success"] is True
    assert result["committed"] is False
    assert bv.added == []
    res = result["results"][0]
    assert res["status"] == "noop"
    assert res["function"] == "player_update"


def test_function_create_unmapped_address_is_rejected(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(RuntimeError, match="0xdead.*not mapped"):
        instance._function_create(None, "0xdead", False)

    assert bv.added == []


def test_function_create_non_executable_address_is_rejected(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x5000: _FakeSegment(readable=True, writable=True, executable=False)},
        memory={0x5000: b"\x01\x02\x03\x04"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(RuntimeError, match="0x5000.*not inside an executable segment"):
        instance._function_create(None, "0x5000", False)

    assert bv.added == []


def test_function_create_preview_actually_removes_function(monkeypatch):
    """--preview must leave NO trace. add_function is not journaled, so the
    preview revert has to explicitly remove the created function and read back
    that it is gone -- reporting 'reverted' while the function persists in the
    view is the bug (#117)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_create(None, "0x1000", True)

    assert result["preview"] is True
    assert result["committed"] is False
    assert result["success"] is True
    assert result["rolled_back"] is True
    assert result["results"][0]["status"] == "verified"
    assert ("remove_user_function", 0x1000) in bv.events
    # The crux: no function may persist at the address after a preview.
    assert bv.get_function_at(0x1000) is None


def test_function_create_preview_revert_failure_is_not_success(monkeypatch):
    """If removing the created function on preview-revert fails, the view is
    left modified -- report success:false / rolled_back:false, never a clean
    'reverted'. Honesty over an unverified revert claim (#117)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFunctionCreateBV(
        segments={0x1000: _FakeSegment(readable=True, executable=True)},
        memory={0x1000: b"\x55\x48\x89\xe5"},
    )
    # Removal silently does nothing, so the function persists past the revert.
    bv.remove_user_function = lambda fn: bv.events.append(("remove_attempt", int(fn.start)))
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._function_create(None, "0x1000", True)

    assert result["success"] is False
    assert result["rolled_back"] is False
    assert "may be left modified" in result["message"]
    assert bv.get_function_at(0x1000) is not None
    # The per-op status must not keep claiming 'verified' when the revert failed
    # and the function persists -- route it to 'failed:' like the batch engine.
    assert result["results"][0]["status"] == "rollback_failed"
    from bn.formatters import FAILED_MUTATION_STATUSES
    assert "rollback_failed" in FAILED_MUTATION_STATUSES


# ---------------------------------------------------------------------------
# Verification: local rename with SSA-style variable reconstruction
# ---------------------------------------------------------------------------


def test_verify_local_rename_passes_when_auto_name_persists_but_user_name_on_alt_var(monkeypatch):
    """After analysis BN may reconstruct variable objects at the same storage
    offset.  If the primary variable still reports its auto name but a second
    variable at the same offset carries the user-assigned name, verification
    should succeed.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    # Two variables at the same storage offset — simulates post-analysis state
    # where BN keeps both the auto-named and user-named entries.
    auto_var = _FakeVariable(name="var_48", storage=-72, var_type="int32_t", identifier=3001)
    user_var = _FakeVariable(name="wIndex", storage=-72, var_type="int32_t", identifier=3001)

    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [auto_var, user_var]

    bv = _FakeBV(functions=[fn])

    # Build a result dict as _op_local_rename would produce.
    result = {
        "op": "local_rename",
        "function": "process_usb",
        "address": "0x401000",
        "variable": "var_48",
        "local_id": "0x401000:local:stack:-72:0:3001",
        "storage": -72,
        "identifier": 3001,
        "source_type": "StackVariableSourceType",
        "is_parameter": False,
        "before_name": "var_48",
        "new_name": "wIndex",
        "requested": {"variable": "var_48", "new_name": "wIndex"},
    }

    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert verified["observed"]["variable"] == "wIndex"


def test_verify_local_rename_uses_identifier_lookup(monkeypatch):
    """Verification should prefer identifier-based lookup over raw storage
    matching so it finds the correct variable after analysis rebuilds the
    stack layout."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    # Variable at same storage but different identifier — should NOT be matched.
    other_var = _FakeVariable(name="var_48", storage=-72, var_type="int32_t", identifier=9999)
    renamed_var = _FakeVariable(name="wIndex", storage=-72, var_type="int32_t", identifier=3001)

    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [other_var, renamed_var]

    bv = _FakeBV(functions=[fn])

    result = {
        "op": "local_rename",
        "function": "process_usb",
        "address": "0x401000",
        "variable": "var_48",
        "local_id": "0x401000:local:stack:-72:0:3001",
        "storage": -72,
        "identifier": 3001,
        "source_type": "StackVariableSourceType",
        "is_parameter": False,
        "before_name": "var_48",
        "new_name": "wIndex",
        "requested": {"variable": "var_48", "new_name": "wIndex"},
    }

    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert verified["observed"]["variable"] == "wIndex"


def test_verify_local_rename_fails_when_name_truly_missing(monkeypatch):
    """If no variable at the storage offset has the expected name, verification
    should still fail."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    wrong_var = _FakeVariable(name="var_48", storage=-72, var_type="int32_t", identifier=3001)

    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [wrong_var]

    bv = _FakeBV(functions=[fn])

    result = {
        "op": "local_rename",
        "function": "process_usb",
        "address": "0x401000",
        "variable": "var_48",
        "local_id": "0x401000:local:stack:-72:0:3001",
        "storage": -72,
        "identifier": 3001,
        "source_type": "StackVariableSourceType",
        "is_parameter": False,
        "before_name": "var_48",
        "new_name": "wIndex",
        "requested": {"variable": "var_48", "new_name": "wIndex"},
    }

    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verification_failed"


def _local_retype_result(**overrides):
    result = {
        "op": "local_retype",
        "function": "process_usb",
        "address": "0x401000",
        "variable": "var_48",
        "local_id": "0x401000:local:stack:-72:0:3001",
        "storage": -72,
        "identifier": 3001,
        "source_type": "StackVariableSourceType",
        "is_parameter": False,
        "before_type": "int32_t",
        "expected_type": "char*",
        "requested": {"variable": "var_48", "new_type": "char*"},
    }
    result.update(overrides)
    return result


def test_verify_local_retype_uses_identifier_for_register_locals(monkeypatch):
    """A retyped register/HLIL-visible local lives in neither parameter_vars
    nor stack_layout, so storage-only resolution cannot see it and a change
    that actually landed would fail verification and roll back (#87).
    Identifier-based lookup over the canonical set must find it."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    reg_var = _FakeVariable(
        name="r2_1", storage=2, var_type="char*", identifier=3001,
        source_type="RegisterVariableSourceType",
    )
    fn = _FakeFunction(0x401000, "process_usb")
    fn.hlil = types.SimpleNamespace(vars=[reg_var])
    bv = _FakeBV(functions=[fn])

    result = _local_retype_result(variable="r2_1", storage=2)
    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert verified["observed"]["type"] == "char*"


def test_verify_local_retype_rejects_same_storage_different_identifier(monkeypatch):
    """A different variable at the same storage offset whose type happens to
    match the expected type must not count as success — verification would be
    reading the wrong variable (#87)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    # Neighbor listed FIRST so storage-only resolution would pick it.
    neighbor = _FakeVariable(name="other", storage=-72, var_type="char*", identifier=9999)
    actual = _FakeVariable(name="var_48", storage=-72, var_type="int32_t", identifier=3001)
    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [neighbor, actual]
    bv = _FakeBV(functions=[fn])

    verified = instance._verify_operation(bv, _local_retype_result())
    assert verified["status"] == "verification_failed"
    assert verified["observed"]["variable"] == "var_48"
    assert verified["observed"]["type"] == "int32_t"


def test_verify_local_retype_passes_when_new_type_on_alt_entry_same_identifier(monkeypatch):
    """After analysis BN may keep both an auto and a user entry at the same
    storage offset. If the primary entry still shows the old type but the
    alternate entry (same identifier) carries the expected type, verification
    should succeed — mirroring the rename path."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    stale = _FakeVariable(name="var_48", storage=-72, var_type="int32_t", identifier=3001)
    fresh = _FakeVariable(name="var_48", storage=-72, var_type="char*", identifier=3001)
    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [stale, fresh]
    bv = _FakeBV(functions=[fn])

    verified = instance._verify_operation(bv, _local_retype_result())
    assert verified["status"] == "verified"
    assert verified["observed"]["type"] == "char*"


def test_verify_local_retype_falls_back_to_storage_without_identifier(monkeypatch):
    """When no identifier was recorded, storage resolution remains the only
    handle and must still work."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    var = _FakeVariable(name="var_48", storage=-72, var_type="char*", identifier=3001)
    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [var]
    bv = _FakeBV(functions=[fn])

    verified = instance._verify_operation(bv, _local_retype_result(identifier=None))
    assert verified["status"] == "verified"


def test_verify_local_retype_fails_when_identifier_vanished(monkeypatch):
    """If the recorded identifier no longer resolves, verification must report
    failure rather than silently verifying a same-storage stranger."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    stranger = _FakeVariable(name="other", storage=-72, var_type="char*", identifier=9999)
    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [stranger]
    bv = _FakeBV(functions=[fn])

    verified = instance._verify_operation(bv, _local_retype_result())
    assert verified["status"] == "verification_failed"
    assert verified["observed"]["variable"] is None


def test_verify_local_retype_relocates_register_local_dropped_from_hlil(monkeypatch):
    """Narrowing a register-backed local (u32 -> u8) can drop it out of
    hlil.vars even though func.vars still carries it correctly narrowed, so the
    canonical (param/stack/hlil) scan misses it. Verification must relocate it
    by its stable identifier across the full func.vars set and report
    `verified`, not a cry-wolf `verification_failed` (#156)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    narrowed = _FakeVariable(
        name="x0_1", storage=34, var_type="uint8_t", identifier=3001,
        source_type="RegisterVariableSourceType",
    )
    fn = _FakeFunction(0x401000, "process_usb")
    fn.hlil = types.SimpleNamespace(vars=[])  # dropped out of HLIL after narrow
    fn.vars = [narrowed]                       # but still in the complete set
    bv = _FakeBV(functions=[fn])

    result = _local_retype_result(
        variable="x0_1", storage=34, identifier=3001,
        source_type="RegisterVariableSourceType",
        before_type="int32_t", expected_type="uint8_t",
    )
    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert verified["observed"]["type"] == "uint8_t"
    assert verified["observed"]["variable"] == "x0_1"


def test_verify_local_retype_relocates_register_local_narrowed_u16(monkeypatch):
    """Same relocation, u32 -> u16 (the other narrowing in #156's AC)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    narrowed = _FakeVariable(
        name="x0_1", storage=34, var_type="uint16_t", identifier=3001,
        source_type="RegisterVariableSourceType",
    )
    fn = _FakeFunction(0x401000, "process_usb")
    fn.hlil = types.SimpleNamespace(vars=[])
    fn.vars = [narrowed]
    bv = _FakeBV(functions=[fn])

    result = _local_retype_result(
        variable="x0_1", storage=34, identifier=3001,
        source_type="RegisterVariableSourceType",
        before_type="int32_t", expected_type="uint16_t",
    )
    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert verified["observed"]["type"] == "uint16_t"


def test_verify_local_retype_funcvars_match_is_identifier_exact(monkeypatch):
    """The func.vars fallback matches on the unique identifier only: a
    same-storage stranger with the expected type but a different identifier
    must not be accepted (mirrors the canonical-scan safety guarantee)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    stranger = _FakeVariable(
        name="other", storage=34, var_type="uint8_t", identifier=9999,
        source_type="RegisterVariableSourceType",
    )
    fn = _FakeFunction(0x401000, "process_usb")
    fn.hlil = types.SimpleNamespace(vars=[])
    fn.vars = [stranger]  # id 3001 truly gone
    bv = _FakeBV(functions=[fn])

    result = _local_retype_result(
        variable="x0_1", storage=34, identifier=3001,
        source_type="RegisterVariableSourceType",
        before_type="int32_t", expected_type="uint8_t",
    )
    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verification_failed"
    assert verified["observed"]["variable"] is None


def test_find_var_for_restore_relocates_register_local_via_func_vars(monkeypatch):
    """On revert, the non-journaled restore must also relocate a register local
    that dropped out of the canonical set; otherwise the closure raises and the
    clean preview falsely reports 'the view may be left modified' (#156)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    narrowed = _FakeVariable(
        name="x0_1", storage=34, var_type="uint8_t", identifier=3001,
        source_type="RegisterVariableSourceType",
    )
    fn = _FakeFunction(0x401000, "process_usb")
    fn.hlil = types.SimpleNamespace(vars=[])
    fn.vars = [narrowed]

    found = bridge.mutation_engine._find_var_for_restore(
        instance, fn, 3001, 34, False
    )
    assert found is narrowed


# ---------------------------------------------------------------------------
# Verification: prototype with implicit calling convention
# ---------------------------------------------------------------------------


def test_verify_prototype_passes_with_implicit_calling_convention(monkeypatch):
    """BN analysis may add __convention("cdecl") to the function type after
    set_user_type.  Verification should normalise calling conventions before
    comparing."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _ConventionFunction(_FakeFunction):
        def __init__(self):
            # After set_user_type + analysis, BN reports the type WITH
            # the implicit convention annotation.
            super().__init__(
                0x43F200,
                "parse_config",
                'int32_t __convention("cdecl")(char const* path)',
            )

        def set_user_type(self, value):
            # Store with convention added by analysis.
            self.type = 'int32_t __convention("cdecl")(char const* path)'

    class _ConventionBV(_FakeBV):
        def parse_type_string(self, declaration):
            # parse_type_string returns WITHOUT convention.
            return _FakeType("int32_t(char const* path)", type_class="FunctionTypeClass"), None

    fn = _ConventionFunction()
    bv = _ConventionBV(functions=[fn])

    result = instance._op_set_prototype(
        bv,
        {
            "op": "set_prototype",
            "identifier": "parse_config",
            "prototype": "int32_t parse_config(char const* path)",
        },
    )

    # expected_prototype comes from str(parse_type_string(...)): no convention
    assert result["expected_prototype"] == "int32_t(char const* path)"
    # observed will be the fn.type string WITH __convention("cdecl")
    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert '__convention("cdecl")' in verified["observed"]["prototype"]


def test_verify_prototype_still_fails_on_real_mismatch(monkeypatch):
    """When the actual return type or params differ, verification must still
    fail even after convention normalisation."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _MismatchFunction(_FakeFunction):
        def __init__(self):
            super().__init__(0x43F200, "parse_config", "void*(int32_t x)")

        def set_user_type(self, value):
            # Analysis "corrected" the type to something different.
            self.type = "void*(int32_t x)"

    class _MismatchBV(_FakeBV):
        def parse_type_string(self, declaration):
            return _FakeType("int32_t(char const* path)", type_class="FunctionTypeClass"), None

    fn = _MismatchFunction()
    bv = _MismatchBV(functions=[fn])

    result = instance._op_set_prototype(
        bv,
        {
            "op": "set_prototype",
            "identifier": "parse_config",
            "prototype": "int32_t parse_config(char const* path)",
        },
    )

    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verification_failed"


def test_verify_prototype_passes_when_bn_infers_pure_attribute(monkeypatch):
    """BN may re-infer a __pure / __noreturn attribute suffix after
    set_user_type (common on accessors). The requested type lacked it but is
    semantically identical, so the readback must normalise the attribute and
    report `verified` -- not verification_failed + revert the valid edit (#199)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _PureFunction(_FakeFunction):
        def __init__(self):
            # BN auto-typed this accessor int64_t() __pure before the edit.
            super().__init__(0x405250, "reset", "int64_t() __pure")

        def set_user_type(self, value):
            # After set_user_type + analysis BN re-adds the __pure suffix that
            # the requested prototype did not carry.
            self.type = "void(void* self) __pure"

    class _PureBV(_FakeBV):
        def parse_type_string(self, declaration):
            # parse_type_string returns the requested type WITHOUT __pure.
            return _FakeType("void(void* self)", type_class="FunctionTypeClass"), None

    fn = _PureFunction()
    bv = _PureBV(functions=[fn])

    result = instance._op_set_prototype(
        bv,
        {
            "op": "set_prototype",
            "identifier": "reset",
            "prototype": "void reset(void* self)",
        },
    )

    assert result["expected_prototype"] == "void(void* self)"
    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verified"
    assert "__pure" in verified["observed"]["prototype"]


class _LoadBV:
    def __init__(self):
        self.analysis_updated = False

    def update_analysis_and_wait(self):
        self.analysis_updated = True


def _setup_load_test(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(instance.targets, "refresh", lambda: [])
    bridge._headless_views.clear()

    binaryninja = sys.modules["binaryninja"]
    loaded_paths: list[str] = []

    def fake_load(path, update_analysis=True):
        loaded_paths.append(path)
        return _LoadBV()

    binaryninja.load = fake_load
    return bridge, instance, loaded_paths


def test_load_binary_prefers_sibling_bndb(monkeypatch, tmp_path):
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bndb = tmp_path / "foo.so.bndb"
    bndb.write_bytes(b"")

    result = instance._load_binary(str(raw))

    assert loaded_paths == [str(bndb)]
    assert result["path"] == str(bndb)
    assert result["requested_path"] == str(raw)
    assert result["notes"]
    assert "foo.so.bndb" in result["notes"][0]
    assert "--no-bndb" in result["notes"][0]
    bridge._headless_views.clear()


def test_load_binary_no_bndb_opt_out(monkeypatch, tmp_path):
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bndb = tmp_path / "foo.so.bndb"
    bndb.write_bytes(b"")

    result = instance._load_binary(str(raw), prefer_bndb=False)

    assert loaded_paths == [str(raw)]
    assert result["path"] == str(raw)
    assert result["notes"] == []
    bridge._headless_views.clear()


def test_load_binary_no_sibling(monkeypatch, tmp_path):
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")

    result = instance._load_binary(str(raw))

    assert loaded_paths == [str(raw)]
    assert result["path"] == str(raw)
    assert result["notes"] == []
    bridge._headless_views.clear()


def test_load_binary_quick_skips_analysis(monkeypatch, tmp_path):
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")

    result = instance._load_binary(str(raw), quick=True)

    assert result["analyzed"] is False
    assert any("--quick" in note for note in result["notes"])
    assert bridge._headless_views[-1].analysis_updated is False  # heavy phase skipped
    bridge._headless_views.clear()


def test_load_binary_full_runs_analysis(monkeypatch, tmp_path):
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")

    result = instance._load_binary(str(raw))

    assert result["analyzed"] is True
    assert bridge._headless_views[-1].analysis_updated is True
    bridge._headless_views.clear()


def test_load_binary_quick_is_noop_for_bndb(monkeypatch, tmp_path):
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    bndb = tmp_path / "foo.so.bndb"
    bndb.write_bytes(b"")

    result = instance._load_binary(str(bndb), quick=True)

    # A .bndb already carries its saved analysis: --quick is a no-op there.
    assert result["analyzed"] is True
    assert bridge._headless_views[-1].analysis_updated is True
    bridge._headless_views.clear()


def test_load_binary_runs_analysis_outside_write_lock(monkeypatch, tmp_path):
    # #99: load_binary must hold the exclusive lock around the BN open and the
    # publish, but NOT around the multi-minute update_analysis_and_wait -- else
    # doctor/target reads block for the whole load.
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    lock = instance._target_lock
    states: dict[str, bool] = {}
    binaryninja = sys.modules["binaryninja"]

    def fake_load(path, update_analysis=True):
        states["open_writer_held"] = lock._writer  # open is under the write lock
        bv = _LoadBV()
        original = bv.update_analysis_and_wait

        def analyze():
            states["analyze_writer_held"] = lock._writer  # analysis is unlocked
            original()

        bv.update_analysis_and_wait = analyze
        return bv

    binaryninja.load = fake_load
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")

    result = instance._load_binary(str(raw))

    assert states["open_writer_held"] is True       # BN open held the lock
    assert states["analyze_writer_held"] is False   # analysis ran unlocked
    assert result["analyzed"] is True
    assert lock._writer is False                     # lock released at the end
    bridge._headless_views.clear()


def test_load_binary_not_in_write_locked_ops(monkeypatch):
    # The dispatcher must NOT take the exclusive lock for the whole load (#99);
    # load_binary does its own fine-grained locking instead.
    bridge = _load_bridge(monkeypatch)
    assert "load_binary" not in bridge.WRITE_LOCKED_OPS
    assert "load_binary" not in bridge.READ_LOCKED_OPS


class _FakeFileBV:
    def __init__(self, filename: str, session_id: str = "0", view_name: str = "ELF"):
        self.file = types.SimpleNamespace(session_id=session_id, filename=filename)
        self.view_type = types.SimpleNamespace(name=view_name)


def _register_views(bridge, *bvs):
    bridge._headless_views.clear()
    bridge._headless_views.extend(bvs)


def test_committed_mutation_marks_view_dirty_until_saved(monkeypatch):
    """A committed (non-preview) mutation that actually changed state marks the
    view dirty so `close` can warn -- BN's bv.file.modified never flips True for
    our writes. A preview or a pure no-op must NOT dirty the view, and
    clear_dirty (called on save) resets it. (L15)"""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFileBV("/proj/svc", session_id="1")
    bridge._headless_views.clear()
    bridge._headless_views.extend([bv])
    instance.targets.refresh()  # assign the stable view_id
    tm = instance.targets

    assert tm._stable_view_id(bv) is not None
    assert tm.is_dirty(bv) is False

    # low-level mark/clear round-trips
    tm.mark_dirty(bv)
    assert tm.is_dirty(bv) is True
    tm.clear_dirty(bv)            # this is what _save_database calls
    assert tm.is_dirty(bv) is False

    me = bridge.mutation_engine
    verified = {"committed": True, "preview": False, "results": [{"status": "verified"}]}
    noop = {"committed": True, "preview": False, "results": [{"status": "noop"}]}
    previewed = {"committed": False, "preview": True, "results": [{"status": "verified"}]}

    # the facade marks dirty only on a committed write that actually changed state
    monkeypatch.setattr(me, "_mutation", lambda ctx, *a, **k: verified)
    instance._mutation("active", False, [{"op": "rename_symbol"}])
    assert tm.is_dirty(bv) is True

    tm.clear_dirty(bv)
    monkeypatch.setattr(me, "_mutation", lambda ctx, *a, **k: previewed)
    instance._mutation("active", True, [{"op": "rename_symbol"}])
    assert tm.is_dirty(bv) is False  # preview never dirties

    monkeypatch.setattr(me, "_mutation", lambda ctx, *a, **k: noop)
    instance._mutation("active", False, [{"op": "rename_symbol"}])
    assert tm.is_dirty(bv) is False  # a pure no-op changed nothing

    bridge._headless_views.clear()


def test_selector_uses_basename_when_unique(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    bv_a = _FakeFileBV("/proj/alpha.bndb", session_id="11")
    bv_b = _FakeFileBV("/proj/beta.bndb", session_id="22")
    _register_views(bridge, bv_a, bv_b)

    targets = bridge.TargetManager().refresh()
    selectors = {t["filename"]: t["selector"] for t in targets}

    assert selectors["/proj/alpha.bndb"] == "alpha.bndb"
    assert selectors["/proj/beta.bndb"] == "beta.bndb"
    bridge._headless_views.clear()


def test_selector_disambiguates_with_parent_dir(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    bv1 = _FakeFileBV("/work/01_arithmetic_lock/target.bndb", session_id="1")
    bv2 = _FakeFileBV("/work/02_bytecode_vm/target.bndb", session_id="2")
    bv3 = _FakeFileBV("/work/03_layered_seal/target.bndb", session_id="3")
    _register_views(bridge, bv1, bv2, bv3)

    targets = bridge.TargetManager().refresh()
    selectors = {t["filename"]: t["selector"] for t in targets}

    assert selectors["/work/01_arithmetic_lock/target.bndb"] == "01_arithmetic_lock/target.bndb"
    assert selectors["/work/02_bytecode_vm/target.bndb"] == "02_bytecode_vm/target.bndb"
    assert selectors["/work/03_layered_seal/target.bndb"] == "03_layered_seal/target.bndb"
    bridge._headless_views.clear()


def test_selector_falls_back_to_target_id_for_identical_paths(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    bv1 = _FakeFileBV("/work/dup/target.bndb", session_id="1")
    bv2 = _FakeFileBV("/work/dup/target.bndb", session_id="2")
    _register_views(bridge, bv1, bv2)

    targets = bridge.TargetManager().refresh()

    assert targets[0]["selector"] == targets[0]["target_id"]
    assert targets[1]["selector"] == targets[1]["target_id"]
    assert targets[0]["selector"] != targets[1]["selector"]
    bridge._headless_views.clear()


def test_refresh_rows_carry_analysis_state(monkeypatch):
    # target list rows must expose per-target analysis state so an agent can tell
    # a --quick view from a full one without a separate target info per target.
    bridge = _load_bridge(monkeypatch)
    bv_quick = _FakeFileBV("/proj/q.bndb", session_id="1")
    bv_full = _FakeFileBV("/proj/f.bndb", session_id="2")
    _register_views(bridge, bv_quick, bv_full)
    bridge._quick_loaded_views.add(bv_quick)

    rows = {t["filename"]: t for t in bridge.TargetManager().refresh()}
    assert rows["/proj/q.bndb"]["analysis_state"] == "quick"
    assert rows["/proj/q.bndb"]["analyzed"] is False
    assert rows["/proj/f.bndb"]["analysis_state"] == "full"
    assert rows["/proj/f.bndb"]["analyzed"] is True

    bridge._quick_loaded_views.discard(bv_quick)
    bridge._headless_views.clear()


def test_resolve_accepts_path_suffix_selector(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    bv1 = _FakeFileBV("/work/01_arithmetic_lock/target.bndb", session_id="1")
    bv2 = _FakeFileBV("/work/02_bytecode_vm/target.bndb", session_id="2")
    _register_views(bridge, bv1, bv2)

    manager = bridge.TargetManager()
    resolved = manager.resolve("02_bytecode_vm/target.bndb")

    assert resolved is bv2
    bridge._headless_views.clear()


class _ClosableBV:
    def __init__(self, filename: str, session_id: str = "0"):
        self.closed = False
        self.file = types.SimpleNamespace(
            session_id=session_id,
            filename=filename,
            modified=False,
            close=lambda: setattr(self, "closed", True),
        )
        self.view_type = types.SimpleNamespace(name="ELF")


def test_close_binary_by_target_selector_does_not_deadlock(monkeypatch):
    # Regression: _close_binary used to resolve() the selector *while holding*
    # the non-reentrant _headless_views_lock, and resolve() re-acquires it ->
    # permanent deadlock. Run it on a watchdog thread so a regression fails the
    # test instead of hanging the suite.
    import threading

    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv_a = _ClosableBV("/proj/alpha.so", session_id="11")
    bv_b = _ClosableBV("/proj/beta.so", session_id="22")
    _register_views(bridge, bv_a, bv_b)

    out: dict = {}

    def go():
        out["result"] = instance._close_binary(target="alpha.so")

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive(), "close by target deadlocked (resolve under views lock)"
    assert bv_a.closed and not bv_b.closed
    assert bv_a not in bridge._headless_views and bv_b in bridge._headless_views
    bridge._headless_views.clear()


def test_close_binary_all_flag_closes_everything(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv_a = _ClosableBV("/proj/alpha.so")
    bv_b = _ClosableBV("/proj/beta.so")
    _register_views(bridge, bv_a, bv_b)

    result = instance._close_binary(all_=True)

    assert len(result["closed"]) == 2
    assert bv_a.closed and bv_b.closed
    assert bridge._headless_views == []


def test_close_binary_by_path_still_matches(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv_a = _ClosableBV("/proj/alpha.so")
    bv_b = _ClosableBV("/proj/beta.so")
    _register_views(bridge, bv_a, bv_b)

    result = instance._close_binary(path="/proj/beta.so")

    assert [c["path"] for c in result["closed"]] == ["/proj/beta.so"]
    assert bv_b.closed and not bv_a.closed
    assert bv_b not in bridge._headless_views and bv_a in bridge._headless_views
    bridge._headless_views.clear()


def test_close_binary_rejects_path_and_all_together(monkeypatch):
    # A named path + all=true is contradictory; the all-branch used to silently
    # win and close everything. The bridge now rejects the combination so raw
    # socket clients are protected too (#85).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv_a = _ClosableBV("/proj/alpha.so")
    bv_b = _ClosableBV("/proj/beta.so")
    _register_views(bridge, bv_a, bv_b)

    with pytest.raises(RuntimeError, match="not both"):
        instance._close_binary(path="/proj/alpha.so", all_=True)
    assert not bv_a.closed and not bv_b.closed  # nothing destroyed
    bridge._headless_views.clear()


def test_close_binary_by_target_works_when_headless_views_empty(monkeypatch):
    # A GUI-opened view resolves fine but is NOT tracked in _headless_views. The
    # old "no binaries loaded" guard ran before the target branch, so every
    # target-based close failed on a GUI bridge. Target close must succeed even
    # with an empty _headless_views (#86 Problem B).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bridge._headless_views.clear()
    gui_view = _ClosableBV("/proj/gui-opened.bndb")
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: gui_view)

    result = instance._close_binary(target="gui-opened.bndb")

    assert gui_view.closed
    assert [c["path"] for c in result["closed"]] == ["/proj/gui-opened.bndb"]


def test_collect_open_views_merges_headless_views_in_gui_mode(monkeypatch):
    # `bn load` against a GUI bridge appends to _headless_views, but the UI walk
    # only enumerates tabs/contexts -- so a headless-loaded view would be
    # invisible to target list. _collect_open_views must merge them (#86 Problem A).
    bridge = _load_bridge(monkeypatch)

    ui_view = object()
    headless_view = object()

    class _Frame:
        def getCurrentBinaryView(self):
            return ui_view

    class _Context:
        def getCurrentViewFrame(self):
            return _Frame()

        def getTabs(self):
            return []

    fake_ui = types.SimpleNamespace(
        UIContext=types.SimpleNamespace(
            allContexts=lambda: [_Context()],
            activeContext=lambda: None,
        )
    )
    monkeypatch.setattr(bridge, "ui", fake_ui)
    bridge._headless_views.clear()
    bridge._headless_views.append(headless_view)

    views = bridge._collect_open_views()

    ids = {id(v) for v in views}
    assert id(ui_view) in ids and id(headless_view) in ids  # both visible
    # No duplicate if a view is in both the UI walk and _headless_views.
    bridge._headless_views.append(ui_view)
    views2 = bridge._collect_open_views()
    assert sum(1 for v in views2 if v is ui_view) == 1
    bridge._headless_views.clear()


def test_preload_binary_marks_quick_views_for_honesty(monkeypatch, tmp_path):
    # Headless `bn-agent --quick` preload must record the view in
    # _quick_loaded_views so target_info/strings stay honest (#90).
    bridge = _load_bridge(monkeypatch)
    bridge._headless_views.clear()
    bridge._quick_loaded_views.clear()
    binaryninja = sys.modules["binaryninja"]
    binaryninja.load = lambda path, update_analysis=True: _LoadBV()

    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bv = bridge._preload_binary(str(raw), quick=True)

    assert bv in bridge._quick_loaded_views          # marked quick
    assert bv.analysis_updated is False              # heavy phase skipped
    assert bv in bridge._headless_views

    # target_info and strings now tell the truth about the quick view.
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "refresh", lambda: [])
    info = instance._target_info("active")
    assert info["analyzed"] is False and info["analysis_state"] == "quick"
    with pytest.raises(RuntimeError, match="bn refresh"):
        instance._strings("active", query=None, offset=0, limit=10)
    bridge._headless_views.clear()
    bridge._quick_loaded_views.clear()


def test_preload_binary_full_analysis_not_marked_quick(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    bridge._headless_views.clear()
    bridge._quick_loaded_views.clear()
    binaryninja = sys.modules["binaryninja"]
    binaryninja.load = lambda path, update_analysis=True: _LoadBV()

    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bv = bridge._preload_binary(str(raw), quick=False)

    assert bv not in bridge._quick_loaded_views
    assert bv.analysis_updated is True
    bridge._headless_views.clear()


def test_preload_binary_prefers_sibling_bndb(monkeypatch, tmp_path):
    # Headless preload must mirror `bn load`: an adjacent <binary>.bndb carries
    # saved work (renames/comments/types) and must be loaded instead of
    # re-analyzing the raw binary from scratch (#178).
    bridge = _load_bridge(monkeypatch)
    bridge._headless_views.clear()
    binaryninja = sys.modules["binaryninja"]
    loaded_paths: list[str] = []
    binaryninja.load = lambda path, update_analysis=True: (
        loaded_paths.append(path) or _LoadBV()
    )

    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bndb = tmp_path / "foo.so.bndb"
    bndb.write_bytes(b"")

    bridge._preload_binary(str(raw), quick=False)

    assert loaded_paths == [str(bndb)]
    bridge._headless_views.clear()


def test_preload_binary_no_bndb_opt_out(monkeypatch, tmp_path):
    # `bn-agent foo --no-bndb` (prefer_bndb=False) must open the raw binary even
    # when a sidecar exists, matching `bn load --no-bndb` (#178).
    bridge = _load_bridge(monkeypatch)
    bridge._headless_views.clear()
    binaryninja = sys.modules["binaryninja"]
    loaded_paths: list[str] = []
    binaryninja.load = lambda path, update_analysis=True: (
        loaded_paths.append(path) or _LoadBV()
    )

    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bndb = tmp_path / "foo.so.bndb"
    bndb.write_bytes(b"")

    bridge._preload_binary(str(raw), quick=False, prefer_bndb=False)

    assert loaded_paths == [str(raw)]
    bridge._headless_views.clear()


def test_preload_binary_quick_is_noop_for_sibling_bndb(monkeypatch, tmp_path):
    # When preload resolves to the sidecar .bndb, --quick is a no-op there (the
    # .bndb already carries its analysis), so the view is fully analyzed and not
    # marked quick -- same contract as `bn load --quick` on a .bndb (#178/#90).
    bridge = _load_bridge(monkeypatch)
    bridge._headless_views.clear()
    bridge._quick_loaded_views.clear()
    binaryninja = sys.modules["binaryninja"]
    binaryninja.load = lambda path, update_analysis=True: _LoadBV()

    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bndb = tmp_path / "foo.so.bndb"
    bndb.write_bytes(b"")

    bv = bridge._preload_binary(str(raw), quick=True)

    assert bv not in bridge._quick_loaded_views
    assert bv.analysis_updated is True
    bridge._headless_views.clear()
    bridge._quick_loaded_views.clear()


def test_dispatch_rejects_non_boolean_all(monkeypatch):
    # Raw JSON params must be real booleans: "all": "false" is truthy under
    # bool() and used to close every target. Reject it as invalid_request (#91).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv_a = _ClosableBV("/proj/alpha.so")
    _register_views(bridge, bv_a)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._dispatch_on_main("close_binary", {"all": "false"}, None)
    assert exc.value.status == "invalid_request"
    assert not bv_a.closed  # nothing closed
    bridge._headless_views.clear()


def test_dispatch_rejects_non_boolean_quick(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._dispatch_on_main("load_binary", {"path": str(tmp_path / "x"), "quick": "false"}, None)
    assert exc.value.status == "invalid_request"


def test_validate_bool_accepts_real_booleans_and_default(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    assert bridge._validate_bool(None, label="quick", default=True) is True
    assert bridge._validate_bool(None, label="quick", default=False) is False
    assert bridge._validate_bool(True, label="quick", default=False) is True
    assert bridge._validate_bool(False, label="all", default=True) is False
    for bad in ("false", "true", 0, 1, "", "yes"):
        with pytest.raises(bridge.OperationFailure):
            bridge._validate_bool(bad, label="all", default=False)


def test_batch_apply_binder_rejects_nonboolean_preview(monkeypatch):
    """A raw/manifest client sending {"preview": "false"} must be rejected, not
    silently coerced to truthy preview mode (#128)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    with pytest.raises(bridge.OperationFailure) as exc:
        bridge._bind_batch_apply(instance, {"preview": "false", "ops": []}, None)
    assert exc.value.status == "invalid_request"


def test_mutation_binders_reject_nonboolean_preview(monkeypatch):
    """Every single-mutation binder validates its `preview` flag as a real JSON
    boolean before dispatching to _mutation (#128)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    binders = [
        bridge._bind_function_create,
        bridge._bind_rename_symbol,
        bridge._bind_set_comment,
        bridge._bind_delete_comment,
        bridge._bind_set_prototype,
        bridge._bind_local_rename,
        bridge._bind_local_retype,
        bridge._bind_struct_field_set,
        bridge._bind_struct_field_rename,
        bridge._bind_struct_field_delete,
        bridge._bind_types_declare,
    ]
    # `address` satisfies _bind_function_create's params["address"] lookup, which
    # is evaluated before the preview arg; harmless for the other binders.
    for binder in binders:
        with pytest.raises(bridge.OperationFailure) as exc:
            binder(instance, {"preview": "false", "address": "0x1000"}, None)
        assert exc.value.status == "invalid_request", binder.__name__


def test_struct_field_set_rejects_nonboolean_overwrite_existing(monkeypatch):
    """`overwrite_existing` is a documented boolean op field; a string must be
    rejected rather than coerced to True and silently overwriting (#128)."""
    bridge, instance, builder, bv = _struct_set_instance(monkeypatch, [(0, "x")])
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._op_struct_field_set(bv, {
            "struct_name": "S", "offset": "0x8", "field_name": "newf",
            "field_type": "int32_t", "overwrite_existing": "false"})
    assert exc.value.status == "invalid_request"
    assert builder.added == []  # never reached add_member_at_offset


def test_list_functions_count_only_returns_count(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[
        _FakeFunction(0x1000, "a"),
        _FakeFunction(0x2000, "b"),
        _FakeFunction(0x3000, "c"),
    ])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # count_only now carries `total` (matching the list envelope's key) alongside
    # the back-compat `count`.
    assert instance._list_functions(None, count_only=True) == {"count": 3, "total": 3}
    # count must match the full listing's reported total
    listing = instance._list_functions(None)
    assert listing["total"] == 3 and listing["returned"] == 3 and len(listing["functions"]) == 3
    assert listing["items"] == listing["functions"]  # universal items alias


def test_resolve_raises_on_ambiguous_basename(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    bv1 = _FakeFileBV("/work/01_arithmetic_lock/target.bndb", session_id="1")
    bv2 = _FakeFileBV("/work/02_bytecode_vm/target.bndb", session_id="2")
    _register_views(bridge, bv1, bv2)

    manager = bridge.TargetManager()
    with pytest.raises(RuntimeError, match="Ambiguous target selector"):
        manager.resolve("target.bndb")
    bridge._headless_views.clear()


def test_resolve_unknown_selector_lists_open_targets(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    bv1 = _FakeFileBV("/work/01_arithmetic_lock/target.bndb", session_id="1")
    bv2 = _FakeFileBV("/work/02_bytecode_vm/target.bndb", session_id="2")
    _register_views(bridge, bv1, bv2)

    manager = bridge.TargetManager()
    with pytest.raises(RuntimeError) as exc_info:
        manager.resolve("does_not_exist")

    message = str(exc_info.value)
    assert message.startswith("Unknown target selector: does_not_exist")
    assert "Open targets:" in message
    assert "01_arithmetic_lock/target.bndb" in message
    assert "02_bytecode_vm/target.bndb" in message
    assert "view_id=" in message
    assert "target_id=" in message
    assert "view_id / target_id are stable across `bn save`" in message
    bridge._headless_views.clear()


def test_serialize_error_keeps_user_facing_messages_clean(monkeypatch):
    bridge = _load_bridge(monkeypatch)

    runtime = RuntimeError("Function not found: foo")
    assert bridge._serialize_error(runtime) == "Function not found: foo"

    failure = bridge.OperationFailure("unsupported", "Symbol not found: bar")
    assert bridge._serialize_error(failure) == "Symbol not found: bar"

    value_error = ValueError("Unknown operation: bogus")
    assert bridge._serialize_error(value_error) == "Unknown operation: bogus"


def test_serialize_error_prefixes_unexpected_exceptions(monkeypatch):
    bridge = _load_bridge(monkeypatch)

    assert bridge._serialize_error(KeyError("offset")) == "internal error: KeyError: 'offset'"
    assert (
        bridge._serialize_error(AttributeError("'NoneType' has no attribute 'name'"))
        == "internal error: AttributeError: 'NoneType' has no attribute 'name'"
    )


def test_load_binary_already_bndb_skips_lookup(monkeypatch, tmp_path):
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    bndb = tmp_path / "foo.so.bndb"
    bndb.write_bytes(b"")

    result = instance._load_binary(str(bndb))

    assert loaded_paths == [str(bndb)]
    assert result["path"] == str(bndb)
    assert result["notes"] == []
    bridge._headless_views.clear()


# ---------------------------------------------------------------------------
# backward_slice tests
# ---------------------------------------------------------------------------

class SSAVariable:
    """Stand-in for binaryninja.SSAVariable for isinstance checks in tests."""
    pass


class _FakeSSAVariable(SSAVariable):
    """Minimal SSA variable stand-in — hashable, str-able, used as dict key."""
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, _FakeSSAVariable):
            return self.name == other.name
        return NotImplemented


class _FakeMLILInsn:
    """Fake MLIL/SSA instruction for backward-slice tests."""
    def __init__(
        self,
        address: int,
        *,
        operation: str = "MLIL_SET_VAR_SSA",
        params: list | None = None,
        vars_read: list | None = None,
        dest=None,
        src=None,
        size=None,
        left=None,
        right=None,
        constant=None,
    ):
        self._address = address
        self._operation_name = operation
        self._params = params or []
        self._vars_read = vars_read or []
        self.dest = dest
        # Operand attrs for load/address-expr fakes (#162); left unset -> getattr None.
        if src is not None:
            self.src = src
        if size is not None:
            self.size = size
        if left is not None:
            self.left = left
        if right is not None:
            self.right = right
        if constant is not None:
            self.constant = constant

    @property
    def address(self):
        return self._address

    @property
    def operation(self):
        return _FakeOperation(self._operation_name)

    @property
    def params(self):
        return self._params

    @property
    def vars_read(self):
        return list(self._vars_read)

    def __str__(self):
        return f"{self._operation_name} @ 0x{self._address:x}"


class _FakeSSAFunction:
    """Fake MLIL SSA function that tracks variable definitions."""
    def __init__(self, instructions: list[_FakeMLILInsn], definitions: dict | None = None):
        self._instructions = instructions
        self._definitions = dict(definitions or {})
        self.basic_blocks = [_FakeBlock(instructions)] if instructions else []

    def get_ssa_var_definition(self, ssa_var):
        return self._definitions.get(ssa_var)


class _FakeBlock:
    """Fake basic block wrapping a list of instructions."""
    def __init__(self, instructions: list):
        self._instructions = instructions

    def __iter__(self):
        return iter(self._instructions)

    def __len__(self):
        return len(self._instructions)


class _FakeMLILFunction:
    """Fake MLIL function wrapping instructions + SSA form."""
    def __init__(self, instructions: list[_FakeMLILInsn], definitions: dict | None = None):
        self._instructions = instructions
        self.basic_blocks = [_FakeBlock(instructions)] if instructions else []
        self.ssa_form = _FakeSSAFunction(instructions, definitions)

    def __iter__(self):
        return iter(self.basic_blocks)


def test_backward_slice_simple_chain(monkeypatch):
    """Trace a variable through one SET_VAR."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    var_r0 = _FakeSSAVariable("r0#1")
    var_r1 = _FakeSSAVariable("r1#2")

    call_insn = _FakeMLILInsn(
        0x10010,
        operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA", vars_read=[var_r0])],
        vars_read=[var_r0],
    )
    def_insn = _FakeMLILInsn(
        0x10008,
        operation="MLIL_SET_VAR_SSA",
        vars_read=[var_r1],
    )

    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(
        instructions=[call_insn],
        definitions={var_r0: def_insn},
    )
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._backward_slice("active", "test_func", "0x10010", arg_index=0)

    assert result["function"] == "test_func"
    assert result["function_address"] == "0x10000"
    assert result["target_address"] == "0x10010"
    assert result["arg_index"] == 0
    assert result["step_count"] == 2
    assert result["truncated"] is False
    assert result["trace"][0]["ssa_var"] == "r0#1"
    assert result["trace"][0]["terminates"] is False
    assert result["trace"][1]["ssa_var"] == "r1#2"
    assert result["trace"][1]["terminates"] is True
    # No reaching def and no parameter info available -> neutral terminal,
    # not a false "function parameter" claim.
    assert result["trace"][1]["reason"] == "undefined_or_global"


def test_backward_slice_undefined_var(monkeypatch):
    """Variable with no definition should terminate immediately."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    var_param = _FakeSSAVariable("arg1#0")

    call_insn = _FakeMLILInsn(
        0x10020,
        operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10020, operation="MLIL_VAR_SSA", vars_read=[var_param])],
        vars_read=[var_param],
    )

    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(instructions=[call_insn])
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._backward_slice("active", "test_func", "0x10020", arg_index=0)

    assert result["step_count"] == 1
    assert result["trace"][0]["ssa_var"] == "arg1#0"
    assert result["trace"][0]["terminates"] is True
    # The fake exposes no parameter_vars, so an undefined terminal is reported
    # neutrally rather than asserted to be a parameter.
    assert result["trace"][0]["reason"] == "undefined_or_global"


def test_backward_slice_labels_true_parameter(monkeypatch):
    """An undefined terminal that IS a formal parameter is labeled as such."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    var_param = _FakeSSAVariable("arg1#0")
    call_insn = _FakeMLILInsn(
        0x10020,
        operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10020, operation="MLIL_VAR_SSA", vars_read=[var_param])],
        vars_read=[var_param],
    )
    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(instructions=[call_insn])
    # Wire parameter info so the undefined terminal resolves to a real parameter
    # rather than the neutral "undefined" label.
    fn.parameter_vars = [_FakeSSAVariable("arg1")]
    fn.medium_level_il.ssa_form.source_function = fn
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._backward_slice("active", "test_func", "0x10020", arg_index=0)

    assert result["trace"][0]["ssa_var"] == "arg1#0"
    assert result["trace"][0]["terminates"] is True
    assert result["trace"][0]["reason"] == "function_parameter"


def test_backward_slice_depth_is_def_use_distance(monkeypatch):
    """`depth` is the real graph distance from the seed: operands of one
    definition are siblings sharing a depth, not a sequential append index."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    x = _FakeSSAVariable("x#3")
    y = _FakeSSAVariable("y#1")
    z = _FakeSSAVariable("z#2")
    # x = y <op> z : one definition reading two operands.
    def_x = _FakeMLILInsn(0x2000, operation="MLIL_SET_VAR_SSA", vars_read=[y, z], dest=x)
    call_insn = _FakeMLILInsn(
        0x2010,
        operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x2010, operation="MLIL_VAR_SSA", vars_read=[x])],
        vars_read=[x],
    )
    fn = _FakeFunction(0x2000, "f")
    fn.medium_level_il = _FakeMLILFunction(instructions=[def_x, call_insn], definitions={x: def_x})
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._backward_slice("active", "f", "0x2010", arg_index=0)
    by_var = {s["ssa_var"]: s for s in result["trace"]}

    assert by_var["x#3"]["depth"] == 0
    assert by_var["y#1"]["depth"] == 1
    assert by_var["z#2"]["depth"] == 1  # sibling of y#1, same depth (not 2)


def test_backward_slice_steps_carry_ssa_label_and_definition_reason(monkeypatch):
    """Every step gets a stable ssa_label, and an ordinary definition step now
    reports reason `definition` instead of null (#162)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    r0 = _FakeSSAVariable("r0#1")
    r1 = _FakeSSAVariable("r1#2")
    call_insn = _FakeMLILInsn(
        0x10010, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA", vars_read=[r0])], vars_read=[r0])
    def_insn = _FakeMLILInsn(0x10008, operation="MLIL_SET_VAR_SSA", vars_read=[r1])
    fn = _FakeFunction(0x10000, "f")
    fn.medium_level_il = _FakeMLILFunction(instructions=[call_insn], definitions={r0: def_insn})
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._backward_slice("active", "f", "0x10010", arg_index=0)
    assert result["trace"][0]["ssa_label"] == "r0#1"
    assert result["trace"][0]["reason"] == "definition"
    assert result["trace"][1]["ssa_label"] == "r1#2"


def test_backward_slice_field_load_carries_base_offset_width(monkeypatch):
    """A `len = [obj + 8]` field load reports reason `field_load` with structured
    base/offset/width metadata (#162)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    length = _FakeSSAVariable("len#3")
    obj = _FakeSSAVariable("obj#1")
    addr_expr = _FakeMLILInsn(
        0x2000, operation="MLIL_ADD",
        left=_FakeMLILInsn(0x2000, operation="MLIL_VAR_SSA", vars_read=[obj]),
        right=_FakeMLILInsn(0x2000, operation="MLIL_CONST", constant=8))
    load_expr = _FakeMLILInsn(0x2000, operation="MLIL_LOAD_SSA", src=addr_expr, size=4, vars_read=[obj])
    load_def = _FakeMLILInsn(0x2000, operation="MLIL_SET_VAR_SSA", src=load_expr, vars_read=[obj])
    call_insn = _FakeMLILInsn(
        0x2010, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x2010, operation="MLIL_VAR_SSA", vars_read=[length])], vars_read=[length])
    fn = _FakeFunction(0x2000, "f")
    fn.medium_level_il = _FakeMLILFunction([load_def, call_insn], definitions={length: load_def})
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._backward_slice("active", "f", "0x2010", arg_index=0)
    step = result["trace"][0]
    assert step["ssa_label"] == "len#3"
    assert step["reason"] == "field_load"
    assert step["base"] == "obj#1"
    assert step["offset"] == "0x8"
    assert step["width"] == 4


def test_backward_slice_phi_step_reason(monkeypatch):
    """A phi definition reports reason `phi_source` (#162)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    merged = _FakeSSAVariable("v#3")
    a = _FakeSSAVariable("v#1")
    b = _FakeSSAVariable("v#2")
    phi_def = _FakeMLILInsn(0x3000, operation="MLIL_VAR_PHI", vars_read=[a, b])
    call_insn = _FakeMLILInsn(
        0x3010, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x3010, operation="MLIL_VAR_SSA", vars_read=[merged])], vars_read=[merged])
    fn = _FakeFunction(0x3000, "f")
    fn.medium_level_il = _FakeMLILFunction([phi_def, call_insn], definitions={merged: phi_def})
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._backward_slice("active", "f", "0x3010", arg_index=0)
    assert result["trace"][0]["reason"] == "phi_source"


def test_backward_slice_arg_label_and_output_pointer_hint(monkeypatch):
    """An address-of arg with no value reads yields the calling-convention
    register label plus an output-pointer dead-end hint, not a bare empty trace
    (#166)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    addr_of = _FakeMLILInsn(0x4010, operation="MLIL_ADDRESS_OF", vars_read=[])
    other = _FakeMLILInsn(0x4010, operation="MLIL_VAR_SSA", vars_read=[])
    call_insn = _FakeMLILInsn(0x4010, operation="MLIL_CALL_SSA", params=[other, addr_of], vars_read=[])
    fn = _FakeFunction(0x4000, "f")
    fn.calling_convention = type("CC", (), {"int_arg_regs": ["x0", "x1", "x2"]})()
    fn.medium_level_il = _FakeMLILFunction([call_insn])
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._backward_slice("active", "f", "0x4010", arg_index=1)
    assert result["arg_label"]["index"] == 1
    assert result["arg_label"]["register"] == "x1"
    assert result["hints"]
    assert "pointer" in result["hints"][0]


def test_backward_slice_constant_arg_reports_value_hint(monkeypatch):
    """A constant/immediate arg (e.g. read(fd, buf, 0x1fff)'s count) has no SSA
    definition to trace. Instead of a renderer-only "constant or immediate" line
    with no value and an empty JSON `hints`, the bridge surfaces a structured
    hint naming the constant -- so text AND JSON consumers both see it."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    const_arg = _FakeMLILInsn(0x4010, operation="MLIL_CONST", vars_read=[], constant=0x1fff)
    other = _FakeMLILInsn(0x4010, operation="MLIL_VAR_SSA", vars_read=[])
    call_insn = _FakeMLILInsn(0x4010, operation="MLIL_CALL_SSA", params=[other, const_arg], vars_read=[])
    fn = _FakeFunction(0x4000, "f")
    fn.medium_level_il = _FakeMLILFunction([call_insn])
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._backward_slice("active", "f", "0x4010", arg_index=1)
    assert result["trace"] == []
    assert result["hints"]
    assert "0x1fff" in result["hints"][0]
    assert "constant" in result["hints"][0].lower()


def test_backward_slice_no_call_at_address(monkeypatch):
    """Address with no call instruction should raise."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(instructions=[])
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure, match="No call instruction"):
        instance._backward_slice("active", "test_func", "0x99999", arg_index=0)


def test_backward_slice_bad_arg_index(monkeypatch):
    """Out-of-range arg index should raise."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    call_insn = _FakeMLILInsn(
        0x10010,
        operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA")],
    )

    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(instructions=[call_insn])
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure, match="out of range"):
        instance._backward_slice("active", "test_func", "0x10010", arg_index=5)


def test_backward_slice_no_mlil(monkeypatch):
    """Function with no MLIL should raise."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = None
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure, match="has no mlil"):
        instance._backward_slice("active", "test_func", "0x10010", arg_index=0)


def test_backward_slice_interprocedural_follows_callee(monkeypatch):
    """Interprocedural trace crosses into a callee when the traced arg is a call return value."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    # Callee: returns callee_ret_var, which is a copy of a parameter (callee_def_var).
    callee_ret_var = _FakeSSAVariable("result#1")
    callee_def_var = _FakeSSAVariable("tmp#2")
    callee_ret_insn = _FakeMLILInsn(
        0x20010, operation="MLIL_RET", vars_read=[callee_ret_var],
    )
    callee_def_insn = _FakeMLILInsn(
        0x20008, operation="MLIL_SET_VAR_SSA", vars_read=[callee_def_var],
    )
    callee = _FakeFunction(0x20000, "callee_fn")
    callee.medium_level_il = _FakeMLILFunction(
        instructions=[callee_ret_insn, callee_def_insn],
        definitions={callee_ret_var: callee_def_insn},
    )

    # Caller: arg 0 of the traced call (0x10010) is `ret_var`, and `ret_var` is
    # defined by an *inner* call to callee_fn (0x1000c). The slice must therefore
    # cross the call boundary into callee_fn.
    ret_var = _FakeSSAVariable("r0#3")
    # dest is a const-ptr expression (like real MLIL), so _resolve_callee exercises
    # the int(dest)->TypeError->`.constant` fallback rather than the raw-int fast path.
    inner_call_insn = _FakeMLILInsn(
        0x1000c, operation="MLIL_CALL_SSA", dest=_FakeConstPtr(0x20000),
    )
    target_call_insn = _FakeMLILInsn(
        0x10010,
        operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA", vars_read=[ret_var])],
        vars_read=[ret_var],
    )
    caller = _FakeFunction(0x10000, "caller_fn")
    caller.medium_level_il = _FakeMLILFunction(
        instructions=[inner_call_insn, target_call_insn],
        definitions={ret_var: inner_call_insn},
    )
    # Register both functions so _resolve_callee can find callee by address.
    bv = _FakeBV(functions=[caller, callee])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._backward_slice(
        "active", "caller_fn", "0x10010", arg_index=0,
        interprocedural=True, ip_depth=2,
    )

    assert result["interprocedural"] is True
    trace = result["trace"]
    # Exactly one boundary crossing, into callee_fn.
    cross = [s for s in trace if s.get("cross_function")]
    assert len(cross) == 1, f"expected one cross-function step, got {trace}"
    assert cross[0]["callee"] == "callee_fn"
    assert cross[0]["reason"] == "cross_function"
    assert cross[0]["terminates"] is False
    # Recursion actually entered the callee body (steps tagged with its context)...
    assert any(s.get("function_context") == "callee_fn" for s in trace)
    # ...and bottomed out at an undefined terminal in the callee (its
    # parameter; the fake exposes no parameter_vars to confirm that, so it is
    # reported neutrally).
    assert trace[-1]["terminates"] is True
    assert trace[-1]["reason"] == "undefined_or_global"


def test_backward_slice_ip_rejects_llil(monkeypatch):
    """Interprocedural mode should still reject LLIL (no get_ssa_var_definition)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    call_insn = _FakeMLILInsn(
        0x10010, operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA")],
    )
    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(instructions=[call_insn])
    fn.low_level_il = _FakeMLILFunction(instructions=[call_insn])
    # Patch low_level_il.ssa_form to be a bare object without get_ssa_var_definition
    class _NoSsaDefs:
        basic_blocks = []
    fn.low_level_il.ssa_form = _NoSsaDefs()
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    with pytest.raises(bridge.OperationFailure, match="SSA form does not support"):
        instance._backward_slice("active", "test_func", "0x10010", arg_index=0,
                                 view="llil", interprocedural=True)


def test_backward_slice_arg_index_message_states_mlil_convention(monkeypatch):
    """An out-of-range --arg names the MLIL count and the 0-based/MLIL
    convention so a user reading pseudo-C doesn't reach for the wrong index (#226)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    # A call the decompiler may render with one visible argument: exactly one
    # MLIL param, so only --arg 0 is valid.
    call_insn = _FakeMLILInsn(
        0x10010,
        operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA")],
    )
    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(instructions=[call_insn])
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._backward_slice("active", "test_func", "0x10010", arg_index=1)
    msg = str(exc.value)
    assert "this call has 1 MLIL argument(s)" in msg
    assert "(index 0)" in msg
    assert "0-based" in msg and "MLIL" in msg


def test_backward_slice_call_boundary_names_callee(monkeypatch):
    """A value originating at a (non-interprocedural) call boundary names the
    resolved callee symbol instead of just terminating at the raw target (#193)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    # Callee body so the resolver returns a real function with a name.
    callee = _FakeFunction(0x20000, "strlen")
    callee.medium_level_il = _FakeMLILFunction(instructions=[])

    # Caller: arg 0 of the traced call is `ret_var`, defined by an inner call to
    # `strlen`. Default (non-interprocedural) mode terminates at the boundary.
    ret_var = _FakeSSAVariable("r0#3")
    inner_call_insn = _FakeMLILInsn(
        0x1000c, operation="MLIL_CALL_SSA", dest=_FakeConstPtr(0x20000),
    )
    target_call_insn = _FakeMLILInsn(
        0x10010,
        operation="MLIL_CALL_SSA",
        params=[_FakeMLILInsn(0x10010, operation="MLIL_VAR_SSA", vars_read=[ret_var])],
        vars_read=[ret_var],
    )
    caller = _FakeFunction(0x10000, "caller_fn")
    caller.medium_level_il = _FakeMLILFunction(
        instructions=[inner_call_insn, target_call_insn],
        definitions={ret_var: inner_call_insn},
    )
    bv = _FakeBV(functions=[caller, callee])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._backward_slice("active", "caller_fn", "0x10010", arg_index=0)

    boundary = [s for s in result["trace"] if s.get("reason") == "call_or_jump_boundary"]
    assert len(boundary) == 1, f"expected one call boundary, got {result['trace']}"
    assert boundary[0]["terminates"] is True
    assert boundary[0]["callee"] == "strlen"


# --- imports --summary ---


def test_imports_summary_aggregates_counts(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    sym_a = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x1000, "printf")
    sym_a.short_name = "printf"
    sym_a.namespace = "libc"

    sym_b = fake_bn.Symbol(fake_bn.SymbolType.ImportedDataSymbol, 0x2000, "__stdout")
    sym_b.short_name = "__stdout"
    sym_b.namespace = "libc"

    sym_c = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x3000, "read")
    sym_c.short_name = "read"
    sym_c.namespace = "libc"

    sym_d = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x4000, "foobar")
    sym_d.short_name = "foobar"
    sym_d.namespace = "libfoo"

    bv = _FakeBV(symbols=[sym_a, sym_b, sym_c, sym_d])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None, summary=True)

    assert result["total_symbols"] == 4
    assert result["namespaces"] == {"libc": 3, "libfoo": 1}
    assert result["by_kind"] == {"function": 3, "data": 1}


def test_imports_summary_empty(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._imports(None, summary=True)

    assert result["total_symbols"] == 0
    assert result["namespaces"] == {}
    assert result["by_kind"] == {}


def test_imports_default_sort_surfaces_function_imports_first(monkeypatch):
    # #07: address-kind internals must not bury function/libc imports in the
    # default listing -- a `head`/page read should show the function imports.
    # The address symbol is named to sort FIRST alphabetically, so only a
    # kind-priority sort (not a name sort) can surface the function import.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]
    addr_sym = fake_bn.Symbol(fake_bn.SymbolType.ImportAddressSymbol, 0x1000, "aaa_internal")
    addr_sym.short_name = "aaa_internal"
    fn_sym = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x2000, "memcpy")
    fn_sym.short_name = "memcpy"
    bv = _FakeBV(symbols=[addr_sym, fn_sym])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    items = instance._imports(None)["items"]
    assert items[0]["kind"] == "function" and items[0]["name"] == "memcpy"
    kinds = [it["kind"] for it in items]
    assert kinds.index("function") < kinds.index("address")


# --- xrefs import symbol resolution ---


def test_xrefs_falls_back_to_import_symbol_when_function_not_found(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    malloc_sym = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x20000, "malloc")
    malloc_sym.short_name = "malloc"
    malloc_sym.namespace = "libc"

    bv = _FakeBV(
        functions=[_FakeFunction(0x10000, "main")],
        symbols=[malloc_sym],
    )
    # _xrefs now resolves the view through the BridgeContext seam (read_xrefs).
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._xrefs(None, "malloc")

    assert result["import_resolved"] is True
    assert result["import_name"] == "malloc"
    assert result["address"] == "0x20000"


def test_xrefs_demangled_name_resolves_to_definition_not_veneer(monkeypatch):
    """A demangled C++ name matches an import veneer (PLT stub) via short_name,
    but the same symbol is also DEFINED in this module. xrefs must resolve to the
    real definition, not the stub, so the call-graph matches `xrefs <mangled>` /
    decompile rather than silently returning the veneer's refs (#201)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    MANGLED = "_ZN5proto3Msg6handleEv"
    DEMANGLED = "proto::Msg::handle"
    # the PLT import veneer (matched by the demangled short_name)
    veneer = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x403380, MANGLED)
    veneer.short_name = DEMANGLED
    veneer.raw_name = MANGLED
    veneer.namespace = "BNINTERNALNAMESPACE"
    # the real function body, defined in this module
    impl = _FakeFunction(0x405250, MANGLED)
    impl.symbol = _FakeSymbol("FunctionSymbol")

    bv = _FakeBV(functions=[impl], symbols=[veneer])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._xrefs(None, DEMANGLED)
    assert result["address"] == "0x405250"               # the definition, not 0x403380
    assert result["resolved_to_definition"] == "0x405250"
    assert result["import_resolved"] is True


def test_xrefs_thunk_real_collision_surfaces_ambiguity_and_picks_hot(monkeypatch):
    """A bare name that resolves to a 16-byte thunk AND the real body must not
    silently pick the zero-caller member: surface both under `ambiguous_symbol`
    and report xrefs for the member carrying the call traffic (#220)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    caller = _FakeFunction(0x500000, "caller")
    thunk = _FakeFunction(0x440030, "util_free")    # PLT-style thunk: hot
    thunk.is_thunk = True
    real = _FakeFunction(0x4d2e70, "util_free")     # real body: zero direct callers
    real.symbol = _FakeSymbol("FunctionSymbol")
    bv = _FakeBV(
        functions=[caller, thunk, real],
        code_refs={0x440030: [_FakeCodeRef(0x500010, caller), _FakeCodeRef(0x500020, caller)],
                   0x4d2e70: []},
        sections={".text": _FakeSection(".text", 0x400000, 0x500000)},
        segments={0x500010: _FakeSegment(readable=True, executable=True),
                  0x500020: _FakeSegment(readable=True, executable=True)},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._xrefs(None, "util_free")
    amb = result["ambiguous_symbol"]
    assert amb["resolved_to"] == "0x440030"                       # the hot member
    assert {m["address"] for m in amb["members"]} == {"0x440030", "0x4d2e70"}
    assert result["address"] == "0x440030"
    assert result["code_ref_count"] == 2


def test_xrefs_demangled_collision_prefers_definition_over_import_veneer(monkeypatch):
    """The #201 ⊕ #220 intersection: a demangled name matches BOTH the real body
    (FunctionSymbol) and a PIC import veneer (ImportedFunctionSymbol, is_thunk) --
    both present in bv.functions with the demangled short_name. xrefs must resolve
    to the DEFINITION, not the ref-carrying stub (the #220 ref-count tiebreak must
    not regress #201)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    DEMANGLED = "proto::Msg::handle"
    caller = _FakeFunction(0x500000, "caller")

    veneer = _FakeFunction(0x401050, "_ZN5proto3Msg6handleEv")   # PLT veneer: hot
    veneer.is_thunk = True
    vsym = _FakeSymbol("ImportedFunctionSymbol")
    vsym.short_name = DEMANGLED
    veneer.symbol = vsym

    impl = _FakeFunction(0x40114a, "_ZN5proto3Msg6handleEv")     # real body: 0 direct callers
    isym = _FakeSymbol("FunctionSymbol")
    isym.short_name = DEMANGLED
    impl.symbol = isym

    bv = _FakeBV(
        functions=[caller, veneer, impl],
        code_refs={0x401050: [_FakeCodeRef(0x500010, caller)], 0x40114a: []},
        sections={".text": _FakeSection(".text", 0x400000, 0x500000)},
        segments={0x500010: _FakeSegment(readable=True, executable=True)},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._xrefs(None, DEMANGLED)
    assert result["address"] == "0x40114a"               # the definition, NOT the stub
    assert result["resolved_to_definition"] == "0x40114a"
    assert "ambiguous_symbol" not in result              # stub-vs-impl, not a thunk/real collision


def test_find_function_resolves_demangled_via_symbol_short_name(monkeypatch):
    """A function whose `fn.name` BN kept mangled resolves by its demangled
    `symbol.short_name`/`full_name`, so callsites/decompile/xrefs all accept the
    same C++ name (#224a)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x405250, "_ZN3foo3bar4recvEi")   # BN kept fn.name mangled
    sym = _FakeSymbol("FunctionSymbol")
    sym.short_name = "foo::bar::recv"
    sym.full_name = "foo::bar::recv(int32_t)"
    fn.symbol = sym
    bv = _FakeBV(functions=[fn])

    assert int(instance._find_function(bv, "foo::bar::recv").start) == 0x405250
    assert int(instance._find_function(bv, "foo::bar::recv(int32_t)").start) == 0x405250
    assert int(instance._find_function(bv, "_ZN3foo3bar4recvEi").start) == 0x405250


def test_xrefs_resolves_data_symbol_by_name(monkeypatch):
    """`xrefs <data-symbol>` resolves a non-function symbol (a global table) to
    its address instead of failing with a misleading import-only error (#224b)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]
    data_sym = fake_bn.Symbol(fake_bn.SymbolType.DataSymbol, 0x56b688, "g_state_table")
    caller = _FakeFunction(0x401000, "user")
    bv = _FakeBV(
        functions=[caller],
        symbols=[data_sym],
        code_refs={0x56b688: [_FakeCodeRef(0x401010, caller)]},
        sections={".text": _FakeSection(".text", 0x400000, 0x410000)},
        segments={0x401010: _FakeSegment(readable=True, executable=True)},
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._xrefs(None, "g_state_table")
    assert result["address"] == "0x56b688"
    assert result["resolved_symbol"]["kind"] == "data"
    assert result["code_ref_count"] == 1


def test_xrefs_import_symbol_raises_for_unknown_symbol(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x10000, "main")])
    # _xrefs now resolves the view through the BridgeContext seam (read_xrefs).
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(RuntimeError, match="Function not found: nonexistent"):
        instance._xrefs(None, "nonexistent")


def test_scan_for_calls_to_finds_llil_calls(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    func_sym = sys.modules["binaryninja"].Symbol(
        sys.modules["binaryninja"].SymbolType.FunctionSymbol, 0x10000, "my_func"
    )
    func_sym.short_name = "my_func"

    insn_call = _FakeLLILInstruction(0x10010, _FakeConstPtr(0x20000))
    insn_tailcall = _FakeLLILInstruction(
        0x10020, _FakeConstPtr(0x20000), operation="LLIL_TAILCALL"
    )
    insn_other = _FakeLLILInstruction(0x10030, _FakeConstPtr(0x30000))

    fn = _FakeFunction(0x10000, "my_func")
    fn.low_level_il = [[insn_call, insn_tailcall, insn_other]]

    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._scan_for_calls_to(bv, 0x20000)

    assert len(result) == 2
    addresses = [int(r["address"], 16) for r in result]
    assert 0x10010 in addresses
    assert 0x10020 in addresses
    assert 0x10030 not in addresses


def test_scan_for_calls_to_deduplicates_same_address(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    insn = _FakeLLILInstruction(0x10010, _FakeConstPtr(0x20000))
    fn = _FakeFunction(0x10000, "my_func")
    fn.low_level_il = [[insn, insn]]

    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._scan_for_calls_to(bv, 0x20000)

    assert len(result) == 1


# --- function search --exact ---


def test_search_functions_exact_match(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "system"),
            _FakeFunction(0x402000, "QAudioSystemPlugin"),
            _FakeFunction(0x403000, "sprintf"),
        ]
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._search_functions("active", "system", exact=True)

    assert result["returned"] == 1 and result["total"] == 1
    assert result["functions"][0]["name"] == "system"
    assert result["functions"][0]["address"] == "0x401000"


def test_search_functions_exact_case_insensitive(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "System"),
            _FakeFunction(0x402000, "QSystemPlugin"),
        ]
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._search_functions("active", "system", exact=True)

    assert result["returned"] == 1
    assert result["functions"][0]["name"] == "System"


def test_search_functions_exact_no_match(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "system_ex"),
            _FakeFunction(0x402000, "_system"),
        ]
    )
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._search_functions("active", "system", exact=True)

    assert result["functions"] == [] and result["total"] == 0


# ---------------------------------------------------------------------------
# Protocol hardening: request size cap and non-dict payloads
# ---------------------------------------------------------------------------


class _RecordingWriter:
    def __init__(self):
        self.data = b""

    def write(self, data):
        self.data += data


def test_bridge_handler_rejects_oversized_request(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    monkeypatch.setattr(bridge, "MAX_REQUEST_BYTES", 16)

    handler = bridge.BridgeHandler.__new__(bridge.BridgeHandler)
    handler.rfile = io.BytesIO(b"x" * 64)  # no newline within the cap
    writer = _RecordingWriter()
    handler.wfile = writer

    handler.handle()

    response = json.loads(writer.data.decode("utf-8"))
    assert response["ok"] is False
    assert response["error"] == "request too large"


def test_bridge_handler_allows_request_exactly_at_cap_with_newline(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    line = b'{"op": "noop"      }\n'
    monkeypatch.setattr(bridge, "MAX_REQUEST_BYTES", len(line))

    dispatched = []
    handler = bridge.BridgeHandler.__new__(bridge.BridgeHandler)
    handler.rfile = io.BytesIO(line)
    handler.server = types.SimpleNamespace(
        bridge=types.SimpleNamespace(
            dispatch=lambda payload: dispatched.append(payload) or {"ok": True, "result": None, "error": None}
        )
    )
    writer = _RecordingWriter()
    handler.wfile = writer

    handler.handle()

    assert dispatched == [{"op": "noop"}]
    assert json.loads(writer.data.decode("utf-8"))["ok"] is True


def test_bridge_handler_rejects_non_dict_json(monkeypatch):
    bridge = _load_bridge(monkeypatch)

    dispatched = []
    handler = bridge.BridgeHandler.__new__(bridge.BridgeHandler)
    handler.rfile = io.BytesIO(b"[1, 2, 3]\n")
    handler.server = types.SimpleNamespace(
        bridge=types.SimpleNamespace(dispatch=lambda payload: dispatched.append(payload))
    )
    writer = _RecordingWriter()
    handler.wfile = writer

    handler.handle()

    response = json.loads(writer.data.decode("utf-8"))
    assert response["ok"] is False
    assert "JSON object" in response["error"]
    assert dispatched == []


# ---------------------------------------------------------------------------
# batch_apply: missing target must stay None, not become "None"
# ---------------------------------------------------------------------------


def test_batch_apply_passes_none_target_through(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    captured: dict = {}

    def fake_mutation(selector, preview, operations):
        captured["selector"] = selector
        captured["preview"] = preview
        captured["operations"] = operations
        return {"success": True}

    monkeypatch.setattr(instance, "_mutation", fake_mutation)

    instance._dispatch_on_main("batch_apply", {"ops": [{"op": "rename_symbol"}]}, None)
    # Both manifest target and request target are absent -> the single-open-
    # target default must still apply, so the selector stays None (not "None").
    assert captured["selector"] is None

    instance._dispatch_on_main("batch_apply", {"ops": []}, "alpha.bndb")
    assert captured["selector"] == "alpha.bndb"

    instance._dispatch_on_main(
        "batch_apply", {"ops": [], "target": "beta.bndb"}, "alpha.bndb"
    )
    assert captured["selector"] == "beta.bndb"


# ---------------------------------------------------------------------------
# TargetManager: _ids_by_object pruning and id() recycling
# ---------------------------------------------------------------------------


def test_target_manager_does_not_alias_recycled_object_ids(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    bv_a = _FakeFileBV("/proj/alpha.bndb", session_id="11")
    _register_views(bridge, bv_a)
    manager = bridge.TargetManager()
    manager.refresh()

    # Simulate CPython id() recycling: a stale map entry whose key collides
    # with a brand-new view but whose ref points at a different object.
    bv_b = _FakeFileBV("/proj/beta.bndb", session_id="22")
    manager._ids_by_object[id(bv_b)] = (weakref.ref(bv_a), "999")
    _register_views(bridge, bv_a, bv_b)

    targets = manager.refresh()
    by_file = {t["filename"]: t["view_id"] for t in targets}

    # The new view must get a fresh id, not inherit the stale "999".
    assert by_file["/proj/beta.bndb"] != "999"
    assert by_file["/proj/alpha.bndb"] != by_file["/proj/beta.bndb"]
    bridge._headless_views.clear()


def test_target_manager_prunes_dead_id_entries_on_refresh(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    bv_a = _FakeFileBV("/proj/alpha.bndb", session_id="11")
    _register_views(bridge, bv_a)
    manager = bridge.TargetManager()

    class _Doomed:
        pass

    doomed = _Doomed()
    manager._ids_by_object[id(doomed)] = (weakref.ref(doomed), "777")
    del doomed

    manager.refresh()

    assert all(vid != "777" for _, vid in manager._ids_by_object.values())
    assert len(manager._ids_by_object) == 1
    bridge._headless_views.clear()


# ---------------------------------------------------------------------------
# Verification: fallback must not accept an unrelated same-named variable
# ---------------------------------------------------------------------------


def test_verify_local_rename_rejects_unrelated_var_with_target_name(monkeypatch):
    """Two variables share a storage slot. The OTHER one (different identifier)
    already carries the requested name; the renamed variable still shows its
    auto name, i.e. the rename did not land. Verification must fail instead of
    crediting the neighbor's name."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    renamed = _FakeVariable(name="var_48", storage=-72, var_type="int32_t", identifier=3001)
    other = _FakeVariable(name="wIndex", storage=-72, var_type="int32_t", identifier=9999)

    fn = _FakeFunction(0x401000, "process_usb")
    fn.stack_layout = [renamed, other]

    bv = _FakeBV(functions=[fn])

    result = {
        "op": "local_rename",
        "function": "process_usb",
        "address": "0x401000",
        "variable": "var_48",
        "local_id": "0x401000:local:stack:-72:0:3001",
        "storage": -72,
        "identifier": 3001,
        "source_type": "StackVariableSourceType",
        "is_parameter": False,
        "before_name": "var_48",
        "new_name": "wIndex",
        "requested": {"variable": "var_48", "new_name": "wIndex"},
    }

    verified = instance._verify_operation(bv, result)
    assert verified["status"] == "verification_failed"


# ---------------------------------------------------------------------------
# xrefs: ambiguous function identifiers must not degrade to "not found"
# ---------------------------------------------------------------------------


def test_xrefs_reraises_ambiguous_function_identifier(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "duplicate_name"),
            _FakeFunction(0x402000, "duplicate_name"),
        ]
    )
    # _xrefs now resolves the view through the BridgeContext seam (read_xrefs).
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    with pytest.raises(RuntimeError, match="Ambiguous function identifier"):
        instance._xrefs(None, "duplicate_name")


# ---------------------------------------------------------------------------
# Visible degradation markers for IL / pseudo-C rendering failures
# ---------------------------------------------------------------------------


def test_function_text_marks_il_failure_instead_of_silent_prototype(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    warnings: list[str] = []
    monkeypatch.setattr(bridge.bn, "log_warn", lambda message: warnings.append(message))

    fn = _FakeFunction(0x401000, "player_update")  # has no .hlil attribute

    text = instance._function_text(None, fn, view="hlil")

    assert text.startswith("// bn: IL rendering failed (")
    assert "showing prototype only" in text.splitlines()[0]
    assert warnings  # failure was logged, not swallowed


# ---------------------------------------------------------------------------
# Registry write atomicity
# ---------------------------------------------------------------------------


def test_write_registry_is_atomic_and_leaves_no_temp_files(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    instance.registry_path = tmp_path / "registry.json"

    instance._write_registry()

    data = json.loads(instance.registry_path.read_text(encoding="utf-8"))
    assert data["socket_path"] == str(instance.socket_path)
    assert data["plugin_name"]
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]
    assert leftovers == []


# ---------------------------------------------------------------------------
# start(): never displace a live socket
# ---------------------------------------------------------------------------


def test_start_refuses_to_displace_live_socket(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    sock_path = tmp_path / "bridge.sock"

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(1)
    instance.socket_path = sock_path
    try:
        with pytest.raises(RuntimeError, match="refusing to displace"):
            instance.start()
        # The live socket file must still be there for its owner.
        assert sock_path.exists()
    finally:
        server.close()


def test_socket_is_live_false_for_stale_socket_file(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    sock_path = tmp_path / "stale.sock"

    # Bind then close: the filesystem entry remains but nothing is listening.
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.close()
    instance.socket_path = sock_path

    assert sock_path.exists()
    assert instance._socket_is_live() is False


# --- save_database: never report success when nothing was written -------------


class _SaveBV:
    """Minimal view for _save_database: records the path create_database got and
    optionally writes a file / returns a chosen bool, mimicking Binary Ninja."""

    def __init__(self, filename: str, *, result=True, write: bool = True):
        self.file = types.SimpleNamespace(filename=filename)
        self._result = result
        self._write = write
        self.created_with = None

    def create_database(self, out: str):
        self.created_with = out
        if self._write:
            Path(out).write_text("bndb")
        return self._result


def test_save_database_succeeds_when_file_is_written(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _SaveBV(str(tmp_path / "x.bin"), result=True, write=True)
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)

    out = tmp_path / "x.bndb"
    result = instance._save_database(None, str(out))

    assert result == {"saved": True, "path": str(out.resolve())}
    assert out.exists()
    assert bv.created_with == str(out.resolve())


def test_save_database_fails_when_create_database_returns_false(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # Unwritable-dir style failure: BN returns False and writes nothing.
    bv = _SaveBV(str(tmp_path / "x.bin"), result=False, write=False)
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)

    out = tmp_path / "x.bndb"
    with pytest.raises(RuntimeError, match="no file was written"):
        instance._save_database(None, str(out))
    assert not out.exists()


def test_save_database_fails_when_file_missing_despite_truthy_return(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # BN claims success but no file lands on disk -> still a hard failure.
    bv = _SaveBV(str(tmp_path / "x.bin"), result=True, write=False)
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)

    out = tmp_path / "x.bndb"
    with pytest.raises(RuntimeError, match="no file was written"):
        instance._save_database(None, str(out))


def test_save_database_errors_before_calling_bn_when_parent_dir_missing(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _SaveBV(str(tmp_path / "x.bin"))
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)

    missing = tmp_path / "nope" / "x.bndb"
    with pytest.raises(RuntimeError, match="directory does not exist"):
        instance._save_database(None, str(missing))
    # Fails fast: create_database is never attempted.
    assert bv.created_with is None


# --- field_xrefs: resolve data-ref types without the nonexistent get_type_at --


class _FieldRefBV:
    """View for _field_xrefs. Deliberately has NO get_type_at(): the fix must
    resolve data-ref types via get_data_var_at(), not the nonexistent method
    that previously crashed the whole --field query."""

    def __init__(self, *, code_refs, data_refs, symbols, data_vars, disassembly):
        self._code_refs = code_refs
        self._data_refs = data_refs
        self._symbols = symbols
        self._data_vars = data_vars
        self._disasm = disassembly

    def get_code_refs_for_type_field(self, type_name, offset):
        return list(self._code_refs.get((type_name, offset), []))

    def get_data_refs_for_type_field(self, type_name, offset):
        return list(self._data_refs.get((type_name, offset), []))

    def get_symbol_at(self, address):
        return self._symbols.get(int(address))

    def get_data_var_at(self, address):
        return self._data_vars.get(int(address))

    def get_disassembly(self, address, arch=None):
        return self._disasm.get(int(address), "")


def test_field_xrefs_resolves_data_var_type(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    caller = _FakeFunction(0x1000, "use_field")
    code_ref = types.SimpleNamespace(func=caller, address=0x1010, size=4, incomingType="int32_t")
    bv = _FieldRefBV(
        code_refs={("Foo", 4): [code_ref]},
        data_refs={("Foo", 4): [0x2000, 0x3000]},
        symbols={0x2000: types.SimpleNamespace(name="g_foo")},
        # 0x2000 has a data var (type resolves); 0x3000 has none (type -> None).
        data_vars={0x2000: types.SimpleNamespace(type="struct Foo")},
        disassembly={0x1010: "ldr r0, [r1, #4]"},
    )

    # _field_xrefs now resolves the view through the BridgeContext seam and calls
    # the module-level _resolve_type_field directly (read_xrefs), so patch both
    # where the moved free function reaches them.
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(
        bridge.read_xrefs,
        "_resolve_type_field",
        lambda ctx, view, spec: {"type_name": "Foo", "offset": 4, "field_name": "bar"},
    )

    # Must not raise (the old get_type_at call would AttributeError here).
    result = instance._field_xrefs("active", "Foo.bar")

    assert result["code_refs"][0]["function"] == "use_field"
    assert result["code_refs"][0]["disasm"] == "ldr r0, [r1, #4]"
    assert result["data_refs"] == [
        {"address": "0x2000", "symbol": "g_foo", "type": "struct Foo"},
        {"address": "0x3000", "symbol": None, "type": None},
    ]


# --- papercuts: pagination, clean errors, force-analysis echo -----------------


def test_imports_pagination_slices_offset_and_limit(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]
    syms = []
    for i in range(5):
        s = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x1000 + i, f"fn{i}")
        s.short_name = f"fn{i}"
        s.namespace = "lib"
        syms.append(s)
    bv = _FakeBV(symbols=syms)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    full = instance._imports(None)
    full_items = full["items"]
    assert [item["name"] for item in full_items] == ["fn0", "fn1", "fn2", "fn3", "fn4"]
    assert full["total"] == 5 and full["has_more"] is False
    page = instance._imports(None, offset=1, limit=2)
    # The page carries the slice AND the honest total/remainder metadata.
    assert page["items"] == full_items[1:3]
    assert page["total"] == 5
    assert page["offset"] == 1 and page["limit"] == 2 and page["returned"] == 2
    assert page["has_more"] is True   # offset 1 + 2 returned < 5 total
    # summary aggregates the whole set regardless of offset/limit.
    summary = instance._imports(None, summary=True, offset=1, limit=2)
    assert summary["total_symbols"] == 5


def test_imports_rejects_negative_offset_and_limit(monkeypatch):
    # Non-CLI callers (raw socket / py exec) must hit the same paging guard the
    # sibling list ops enforce, not a silent negative-index slice (#68).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(symbols=[])
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    with pytest.raises(bridge.OperationFailure) as e1:
        instance._imports(None, offset=-1)
    assert e1.value.status == "invalid_request"
    with pytest.raises(bridge.OperationFailure) as e2:
        instance._imports(None, limit=-5)
    assert e2.value.status == "invalid_request"
    with pytest.raises(bridge.OperationFailure) as e3:
        instance._imports(None, limit=0)   # zero limit is degenerate too
    assert e3.value.status == "invalid_request"


def test_list_ops_return_paged_envelope_with_true_total(monkeypatch):
    # #122: strings/imports/sections return the same {items, total, offset,
    # limit, returned, has_more} envelope as function list, so a truncating
    # limit still reports the honest total + remainder instead of a bare slice.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    strings = [_FakeStringRef(0x1000 + i * 0x10, 8, f"tok_{i:03d}") for i in range(5)]
    secs = {f".s{i}": _FakeSection(f".s{i}", 0x2000 + i * 0x100, 0x2080 + i * 0x100)
            for i in range(5)}
    syms = []
    for i in range(5):
        s = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x4000 + i, f"imp{i}")
        s.short_name = f"imp{i}"
        s.namespace = "lib"
        syms.append(s)
    bv = _FakeBV(strings=strings, sections=secs, symbols=syms)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    envelope_keys = {"items", "total", "offset", "limit", "returned", "has_more"}

    # A limit that truncates: 2 of 5 come back, but the total stays honest.
    strings_page = instance._strings(None, query=None, offset=0, limit=2)
    assert set(strings_page) == envelope_keys
    assert strings_page["total"] == 5
    assert strings_page["returned"] == 2
    assert len(strings_page["items"]) == 2
    assert strings_page["has_more"] is True

    imports_page = instance._imports(None, offset=0, limit=2)
    assert set(imports_page) == envelope_keys
    assert imports_page["total"] == 5 and imports_page["returned"] == 2
    assert imports_page["has_more"] is True

    sections_page = instance._sections(None, offset=0, limit=2)
    assert set(sections_page) == envelope_keys
    assert sections_page["total"] == 5 and sections_page["returned"] == 2
    assert sections_page["has_more"] is True

    # The last page (offset past the truncation point) reports no remainder.
    tail = instance._strings(None, query=None, offset=4, limit=2)
    assert tail["total"] == 5 and tail["returned"] == 1 and tail["has_more"] is False

    # limit=None means "no limit": every item, has_more False.
    everything = instance._strings(None, query=None, offset=0, limit=None)
    assert everything["returned"] == 5 and everything["has_more"] is False


def test_apply_operation_comment_function_only_form_accepted(monkeypatch):
    # The documented function-only comment form (no `address`) must pass
    # required-field validation, not be rejected as missing `address` (#67).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(bridge.mutation_engine, "_op_set_comment", lambda ctx, bv, op: {"ok": "set"})
    monkeypatch.setattr(bridge.mutation_engine, "_op_delete_comment", lambda ctx, bv, op: {"ok": "del"})
    assert instance._apply_operation(
        None, {"op": "set_comment", "function": "main", "comment": "hi"}) == {"ok": "set"}
    assert instance._apply_operation(
        None, {"op": "delete_comment", "function": "main"}) == {"ok": "del"}


def test_apply_operation_comment_address_form_still_accepted(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(bridge.mutation_engine, "_op_set_comment", lambda ctx, bv, op: {"ok": "set"})
    assert instance._apply_operation(
        None, {"op": "set_comment", "address": "0x1000", "comment": "hi"}) == {"ok": "set"}


def test_apply_operation_comment_requires_function_or_address(monkeypatch):
    # Neither locator field present -> precise invalid_request naming both options.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._apply_operation(None, {"op": "set_comment", "comment": "hi"})
    assert exc.value.status == "invalid_request"
    assert "function" in str(exc.value) and "address" in str(exc.value)


def test_apply_operation_set_comment_missing_comment_still_rejected(monkeypatch):
    # The genuinely-required field is still enforced precisely.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._apply_operation(None, {"op": "set_comment", "function": "main"})
    assert exc.value.status == "invalid_request"
    assert "comment" in str(exc.value)


def test_apply_operation_missing_op_key_is_invalid_request(monkeypatch):
    # A manifest op without an `op` key must be invalid_request naming `op`, NOT
    # silently dispatched as a rename_symbol (which risks a wrong mutation) (#48).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._apply_operation(None, {"identifier": "x", "new_name": "y"})
    assert exc.value.status == "invalid_request"
    assert "'op'" in str(exc.value)


def test_operation_failure_result_missing_op_is_honest(monkeypatch):
    # The per-op failure echo for an op missing `op` must not claim rename_symbol.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    exc = bridge.OperationFailure("invalid_request", "missing op")
    result = instance._operation_failure_result({"identifier": "x"}, exc)
    assert result["op"] == "<missing>"


def test_apply_operation_non_object_op_is_invalid_request(monkeypatch):
    # A non-object manifest op element (e.g. "ops": ["foo"]) must be a clean
    # invalid_request, not an AttributeError (#48).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._apply_operation(None, "not_an_object")
    assert exc.value.status == "invalid_request"
    # the failure-result/echo helpers must tolerate the non-dict op too
    assert instance._operation_requested("not_an_object") == {}
    assert instance._operation_failure_result("not_an_object", exc.value)["op"] == "<non-object>"


def test_sections_pagination_slices_offset_and_limit(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    secs = {
        ".a": _FakeSection(".a", 0x1000, 0x1100),
        ".b": _FakeSection(".b", 0x2000, 0x2100),
        ".c": _FakeSection(".c", 0x3000, 0x3100),
    }
    bv = _FakeBV(sections=secs)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    full = instance._sections(None)
    assert [s["name"] for s in full["items"]] == [".a", ".b", ".c"]
    assert full["total"] == 3 and full["has_more"] is False
    page = instance._sections(None, offset=1, limit=1)
    assert [s["name"] for s in page["items"]] == [".b"]
    # The truncated page still reports the true total + remainder.
    assert page["total"] == 3 and page["returned"] == 1 and page["has_more"] is True


def test_py_exec_reports_script_error_with_type_prefix(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # A NameError used to be tagged "internal error: NameError:" while a raised
    # ValueError surfaced as a bare message. Both now read "TypeName: message".
    with pytest.raises(RuntimeError, match=r"^NameError: name 'missing' is not defined$"):
        instance._py_exec("active", "missing")
    with pytest.raises(RuntimeError, match=r"^ValueError: boom$"):
        instance._py_exec("active", "raise ValueError('boom')")


def test_load_binary_corrupt_file_raises_clean_error(monkeypatch, tmp_path):
    bridge, instance, _ = _setup_load_test(monkeypatch)
    raw = tmp_path / "broken.bndb"
    raw.write_bytes(b"not a real database")

    def boom(path, update_analysis=True):
        raise Exception("Unable to create new BinaryView")

    sys.modules["binaryninja"].load = boom

    # A corrupt/truncated file used to escape as "internal error: Exception: ...".
    with pytest.raises(RuntimeError, match="may be corrupt"):
        instance._load_binary(str(raw))
    bridge._headless_views.clear()


def test_decompile_force_requested_but_not_skipped_echoes_flag(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "small_fn")  # analysis_skipped defaults False
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bridge.il_format, "_comment_map", lambda bv, func: {})
    _install_fake_pseudo_c(
        monkeypatch, bridge, fn,
        [[(0x401000, "int32_t small_fn()")], [(0x401000, "{")], [(0x401000, "}")]],
    )

    result = instance._decompile("active", "small_fn", force_analysis=True)

    # Nothing was skipped, so no reanalysis ran ...
    assert result["analysis_forced"] is False
    assert fn.reanalyzed is False
    # ... but the echo confirms --force-analysis was honored, not silently ignored.
    assert result["analysis_force_requested"] is True


# --- --quick honesty: don't return a misleading empty result on an unanalyzed view


def test_strings_requires_refresh_when_quick_loaded(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(strings=[])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # Quick-loaded: strings analysis hasn't run, so refuse rather than return [].
    bridge._quick_loaded_views.add(bv)
    with pytest.raises(RuntimeError, match="loaded with --quick"):
        instance._strings(None, query=None, offset=0, limit=100)

    # Once analysis lands, strings answers normally (here: genuinely empty).
    bridge._quick_loaded_views.discard(bv)
    result = instance._strings(None, query=None, offset=0, limit=100)
    assert result["items"] == [] and result["total"] == 0


def test_xrefs_requires_refresh_when_quick_loaded(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # Quick-loaded: code-ref analysis hasn't run, so a 0/0 result reads as
    # "no xrefs" rather than "not analyzed". Refuse with a directive instead.
    bridge._quick_loaded_views.add(bv)
    with pytest.raises(RuntimeError, match="loaded with --quick"):
        instance._xrefs(None, "main")
    bridge._quick_loaded_views.discard(bv)


def test_function_info_requires_refresh_when_quick_loaded(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # Quick-loaded: size/xref/signature fields are bogus until analysis runs.
    bridge._quick_loaded_views.add(bv)
    with pytest.raises(RuntimeError, match="loaded with --quick"):
        instance._function_info(None, "main")
    bridge._quick_loaded_views.discard(bv)


def test_callsites_requires_refresh_when_quick_loaded(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # Quick-loaded: "Function not found" would misattribute missing analysis
    # to a typo. Refuse with a directive instead.
    bridge._quick_loaded_views.add(bv)
    with pytest.raises(RuntimeError, match="loaded with --quick"):
        _callsites_items(instance,None, "strcpy", within_identifiers=["main"])
    bridge._quick_loaded_views.discard(bv)


def test_taint_requires_refresh_when_quick_loaded(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # Quick-loaded: "no call to <sink> found" would misdiagnose missing
    # analysis. Refuse with a directive instead.
    bridge._quick_loaded_views.add(bv)
    with pytest.raises(RuntimeError, match="loaded with --quick"):
        instance._taint(None, {"function": "main", "direction": "backward", "sinks": ["system"]})
    bridge._quick_loaded_views.discard(bv)


def test_target_info_reports_quick_analysis_state(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "refresh", lambda: [])

    bridge._quick_loaded_views.add(bv)
    info = instance._target_info("active")
    assert info["analyzed"] is False
    assert info["analysis_state"] == "quick"

    bridge._quick_loaded_views.discard(bv)
    info2 = instance._target_info("active")
    assert info2["analyzed"] is True
    assert info2["analysis_state"] == "full"


def test_refresh_clears_quick_state_and_enables_strings(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(strings=[])
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "refresh", lambda: [])

    bridge._quick_loaded_views.add(bv)
    instance._refresh(None)  # runs analysis and clears the quick flag

    assert bv not in bridge._quick_loaded_views
    assert getattr(bv, "analysis_updated", False) is True
    result = instance._strings(None, query=None, offset=0, limit=100)
    assert result["items"] == [] and result["total"] == 0


def test_load_quick_marks_view_full_load_does_not(monkeypatch, tmp_path):
    bridge, instance, _ = _setup_load_test(monkeypatch)

    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    result = instance._load_binary(str(raw), quick=True)
    assert result["analyzed"] is False
    quick_bv = bridge._headless_views[-1]
    assert quick_bv in bridge._quick_loaded_views

    raw2 = tmp_path / "bar.so"
    raw2.write_bytes(b"")
    full = instance._load_binary(str(raw2), quick=False)
    assert full["analyzed"] is True
    full_bv = bridge._headless_views[-1]
    assert full_bv not in bridge._quick_loaded_views

    bridge._headless_views.clear()


def test_apply_operation_missing_field_is_invalid_request(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    # rename_symbol requires 'identifier' (and 'new_name'); omitting it is a
    # malformed request, not an unsupported operation.
    with pytest.raises(bridge.OperationFailure) as excinfo:
        instance._apply_operation(bv, {"op": "rename_symbol"})
    assert excinfo.value.status == "invalid_request"
    assert "identifier" in str(excinfo.value)


def test_invalid_request_counts_as_a_failed_mutation_status():
    from bn.formatters import FAILED_MUTATION_STATUSES
    assert "invalid_request" in FAILED_MUTATION_STATUSES


class _FakeStructMember:
    def __init__(self, offset, name, type_text="int32_t"):
        self.offset = offset
        self.name = name
        self.type = type_text


class _FakeStructBuilder:
    def __init__(self, members):
        self.members = list(members)

    def index_by_name(self, name):
        for i, m in enumerate(self.members):
            if m.name == name:
                return i
        return None

    def __getitem__(self, name):
        for m in self.members:
            if m.name == name:
                return m
        return None

    def replace(self, index, type_, name, overwrite):
        self.members[index].name = name

    def remove(self, index):
        del self.members[index]


def _struct_instance(monkeypatch, members):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    builder = _FakeStructBuilder(members)
    # _op_struct_field_* moved to mutation_engine and call these module-locally,
    # so stub them on the module (patching instance._* no longer intercepts).
    monkeypatch.setattr(bridge.mutation_engine, "_struct_builder", lambda ctx, bv, name: ("S", builder))
    monkeypatch.setattr(bridge.mutation_engine, "_commit_struct_builder", lambda *a, **k: None)
    return bridge, instance, builder


def test_struct_field_rename_accepts_offset(monkeypatch):
    bridge, instance, builder = _struct_instance(
        monkeypatch, [_FakeStructMember(0, "a"), _FakeStructMember(8, "b")])
    res = instance._op_struct_field_rename(None, {"struct_name": "S", "old_name": "0x8", "new_name": "bb"})
    assert res["old_name"] == "b"  # offset 0x8 resolved to field 'b'
    assert builder.members[1].name == "bb"


def test_struct_field_rename_still_accepts_name(monkeypatch):
    bridge, instance, builder = _struct_instance(
        monkeypatch, [_FakeStructMember(0, "a"), _FakeStructMember(8, "b")])
    res = instance._op_struct_field_rename(None, {"struct_name": "S", "old_name": "a", "new_name": "aa"})
    assert res["old_name"] == "a"
    assert builder.members[0].name == "aa"


def test_struct_field_delete_accepts_offset(monkeypatch):
    bridge, instance, builder = _struct_instance(
        monkeypatch, [_FakeStructMember(0, "a"), _FakeStructMember(8, "b")])
    res = instance._op_struct_field_delete(None, {"struct_name": "S", "field_name": "0x8"})
    assert res["field_name"] == "b"
    assert [m.name for m in builder.members] == ["a"]


def test_struct_field_unknown_locator_is_invalid_request(monkeypatch):
    bridge, instance, builder = _struct_instance(monkeypatch, [_FakeStructMember(0, "a")])
    with pytest.raises(bridge.OperationFailure) as excinfo:
        instance._op_struct_field_rename(None, {"struct_name": "S", "old_name": "0x99", "new_name": "x"})
    assert excinfo.value.status == "invalid_request"


def test_struct_field_offset_delete_targets_right_field_on_duplicate_names(monkeypatch):
    # Two members share the name 'dup' at offsets 0x0 and 0x8. Deleting by
    # offset 0x8 must remove the SECOND one. A name round-trip would resolve via
    # index_by_name's first match (0x0) and silently delete the wrong field (#25).
    bridge, instance, builder = _struct_instance(
        monkeypatch, [_FakeStructMember(0, "dup"), _FakeStructMember(8, "dup")])
    res = instance._op_struct_field_delete(None, {"struct_name": "S", "field_name": "0x8"})
    assert res["field_name"] == "dup"
    assert [(m.offset, m.name) for m in builder.members] == [(0, "dup")]  # 0x8 gone, 0x0 kept


def test_struct_field_offset_rename_targets_right_field_on_duplicate_names(monkeypatch):
    bridge, instance, builder = _struct_instance(
        monkeypatch, [_FakeStructMember(0, "dup"), _FakeStructMember(8, "dup")])
    instance._op_struct_field_rename(None, {"struct_name": "S", "old_name": "0x8", "new_name": "renamed"})
    # only the member at offset 0x8 is renamed; offset 0x0 is untouched
    assert [(m.offset, m.name) for m in builder.members] == [(0, "dup"), (8, "renamed")]


class _AddableStructBuilder:
    def __init__(self):
        self.width = 4
        self.added = []

    def add_member_at_offset(self, name, type_, offset, overwrite):
        self.added.append((name, int(offset), overwrite))


def _struct_set_instance(monkeypatch, occupied_offsets):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    builder = _AddableStructBuilder()
    occupied_type = types.SimpleNamespace(
        members=[_FakeStructMember(off, name) for off, name in occupied_offsets])

    class _BV:
        def parse_type_string(self, s):
            return _FakeType("int32_t"), None

        def get_type_by_name(self, n):
            return occupied_type

    monkeypatch.setattr(bridge.mutation_engine, "_struct_builder", lambda ctx, bv, name: ("S", builder))
    monkeypatch.setattr(bridge.mutation_engine, "_commit_struct_builder", lambda *a, **k: None)
    return bridge, instance, builder, _BV()


def test_struct_field_set_no_overwrite_at_occupied_offset_refuses(monkeypatch):
    # --no-overwrite at an occupied offset must REFUSE, not append an overlapping
    # member (BN's add_member_at_offset(overwrite=False) silently overlaps) (#56).
    bridge, instance, builder, bv = _struct_set_instance(monkeypatch, [(0, "x")])
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._op_struct_field_set(bv, {
            "struct_name": "S", "offset": "0x0", "field_name": "dupfld",
            "field_type": "int32_t", "overwrite_existing": False})
    assert exc.value.status == "invalid_request"
    assert "x" in str(exc.value)              # names the existing member
    assert builder.added == []                # never reached add_member_at_offset


def test_struct_field_set_no_overwrite_at_free_offset_adds(monkeypatch):
    # The contrast case: --no-overwrite at a FREE offset still adds.
    bridge, instance, builder, bv = _struct_set_instance(monkeypatch, [(0, "x")])
    res = instance._op_struct_field_set(bv, {
        "struct_name": "S", "offset": "0x8", "field_name": "newf",
        "field_type": "int32_t", "overwrite_existing": False})
    assert res["before_member"] is None       # 0x8 is free
    assert builder.added and builder.added[0][0] == "newf"


def test_struct_field_set_no_overwrite_refuses_interior_overlap(monkeypatch):
    # An offset that lands INSIDE a wider member (0x4 within an 8-byte member at
    # 0x0) overlaps just as much as an exact-start collision -- must refuse (#56).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    builder = _AddableStructBuilder()
    big = types.SimpleNamespace(offset=0, name="big", type=types.SimpleNamespace(width=8))
    occupied_type = types.SimpleNamespace(members=[big])

    class _BV:
        def parse_type_string(self, s):
            return types.SimpleNamespace(width=4), None   # a 4-byte field

        def get_type_by_name(self, n):
            return occupied_type

    monkeypatch.setattr(bridge.mutation_engine, "_struct_builder", lambda ctx, bv, name: ("S", builder))
    monkeypatch.setattr(bridge.mutation_engine, "_commit_struct_builder", lambda *a, **k: None)
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._op_struct_field_set(_BV(), {
            "struct_name": "S", "offset": "0x4", "field_name": "mid",
            "field_type": "int32_t", "overwrite_existing": False})
    assert exc.value.status == "invalid_request"
    assert "big" in str(exc.value)            # names the spanned member
    assert builder.added == []


def test_struct_field_set_overwrite_at_occupied_offset_replaces(monkeypatch):
    # The other contrast case: default overwrite at an occupied offset still
    # applies (it replaces, not refuses).
    bridge, instance, builder, bv = _struct_set_instance(monkeypatch, [(0, "x")])
    res = instance._op_struct_field_set(bv, {
        "struct_name": "S", "offset": "0x0", "field_name": "replfld",
        "field_type": "int32_t", "overwrite_existing": True})
    assert res["before_member"]["field_name"] == "x"
    assert builder.added and builder.added[0] == ("replfld", 0, True)


def test_annotate_types_declare_verified_when_layout_changed(monkeypatch):
    # A redeclaration of an existing type NAME with a real layout change must be
    # 'verified', not 'noop' -- the authoritative signal is the layout diff, not
    # the decl-string compare that renders the same `struct QA` either way (#57).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    results = [{"op": "types_declare", "status": "noop", "defined_types": {"QA": "struct QA"}}]
    type_diffs = [{"type_name": "QA", "changed": True, "message": "layout changed"}]
    out = instance._annotate_operation_results(results, type_diffs)
    assert out[0]["status"] == "verified"
    assert out[0]["changed_types"] == {"QA": True}


def test_annotate_types_declare_noop_when_unchanged(monkeypatch):
    # A genuinely-identical redeclaration stays 'noop'.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    results = [{"op": "types_declare", "status": "verified", "defined_types": {"QA": "struct QA"}}]
    type_diffs = [{"type_name": "QA", "changed": False, "message": "no change"}]
    out = instance._annotate_operation_results(results, type_diffs)
    assert out[0]["status"] == "noop"


def _pvs(type_name, **kw):
    return types.SimpleNamespace(type=types.SimpleNamespace(name=type_name), **kw)


def _dataflow_values_instance(monkeypatch, ins):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    il = types.SimpleNamespace(instructions=[ins])
    func = types.SimpleNamespace(name="f", start=0x1000)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda sel: object())
    monkeypatch.setattr(instance.ctx, "_find_function", lambda bv, ident: func)
    monkeypatch.setattr(bridge.il_format, "_il_function_for", lambda fn, view, ssa: il)
    return bridge, instance


def test_possible_values_uses_source_when_instruction_undetermined(monkeypatch):
    # BN leaves a SET_VAR instruction's value-set undetermined while the SOURCE
    # expression carries the real const; report the source's value-set (#52).
    src = types.SimpleNamespace(possible_values=_pvs("ConstantValue", value=0xc48))
    ins = types.SimpleNamespace(address=0x1000, possible_values=_pvs("UndeterminedValue"), src=src)
    bridge, instance = _dataflow_values_instance(monkeypatch, ins)
    res = instance._possible_values(None, "f", "0x1000")
    assert res["value_basis"] == "source_expression"
    assert res["possible_values"]["type"] == "ConstantValue"
    assert res["possible_values"]["value"] == 0xc48


def test_possible_values_keeps_instruction_set_when_determined(monkeypatch):
    # When the instruction itself has a determined value-set, keep it (don't
    # blindly prefer the source).
    src = types.SimpleNamespace(possible_values=_pvs("UndeterminedValue"))
    ins = types.SimpleNamespace(address=0x1000, possible_values=_pvs("ConstantValue", value=7), src=src)
    bridge, instance = _dataflow_values_instance(monkeypatch, ins)
    res = instance._possible_values(None, "f", "0x1000")
    assert res["value_basis"] == "instruction"
    assert res["possible_values"]["value"] == 7


def test_possible_values_no_source_uses_instruction(monkeypatch):
    # An instruction with no .src (e.g. not an assignment) falls back to its own
    # value-set.
    ins = types.SimpleNamespace(address=0x1000, possible_values=_pvs("ConstantValue", value=3))
    bridge, instance = _dataflow_values_instance(monkeypatch, ins)
    res = instance._possible_values(None, "f", "0x1000")
    assert res["value_basis"] == "instruction"
    assert res["possible_values"]["value"] == 3


def test_pvs_determined_helper(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    assert instance._pvs_determined(_pvs("ConstantValue", value=1)) is True
    assert instance._pvs_determined(_pvs("UndeterminedValue")) is False
    assert instance._pvs_determined(None) is False


def test_render_type_layout_enum_shows_values(monkeypatch):
    # Enum members carry .value but no .offset/.type; the layout must show the
    # value, not collapse to "0x0000: <unknown> NAME" (#54).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    enum = types.SimpleNamespace(
        width=4,
        type_class=types.SimpleNamespace(name="EnumerationTypeClass"),
        members=[
            types.SimpleNamespace(name="ET_NONE", value=0),
            types.SimpleNamespace(name="ET_REL", value=1),
            types.SimpleNamespace(name="FLAG_HI", value=0x100),
        ],
    )
    out = instance._render_type_layout(enum)
    assert "ET_NONE = 0 (0x0)" in out
    assert "ET_REL = 1 (0x1)" in out
    assert "FLAG_HI = 256 (0x100)" in out
    assert "<unknown>" not in out


def test_render_type_layout_struct_unchanged(monkeypatch):
    # The struct rendering path is unaffected (offset: type name).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    struct = types.SimpleNamespace(
        width=8,
        type_class=types.SimpleNamespace(name="StructureTypeClass"),
        members=[
            types.SimpleNamespace(name="a", offset=0, type="int32_t"),
            types.SimpleNamespace(name="b", offset=4, type="char"),
        ],
    )
    out = instance._render_type_layout(struct)
    assert "0x0000: int32_t a" in out
    assert "0x0004: char b" in out


def _stub_code_context(monkeypatch, instance, function_entry):
    # _address_context and its resolution/address-context helpers now live on the
    # BridgeContext seam (instance.ctx); patch the helpers where the method under
    # test resolves them.
    monkeypatch.setattr(instance.ctx, "_sections_at", lambda bv, a: [{"name": ".text"}])
    monkeypatch.setattr(instance.ctx, "_segment_at", lambda bv, a: {"name": "seg"})
    monkeypatch.setattr(instance.ctx, "_symbol_at", lambda bv, a: None)
    monkeypatch.setattr(instance.ctx, "_function_entry_for_address", lambda bv, a: function_entry)
    monkeypatch.setattr(instance.ctx, "_address_is_code", lambda bv, a: True)


def test_address_context_disasm_uses_target_function_arch(monkeypatch):
    # target_context disasm must decode with the TARGET function's arch, not the
    # bv default -- a THUMB2 target must not be ARM-misdecoded into garbage (#53).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    thumb_arch = object()
    fn = types.SimpleNamespace(start=0x12e74, name="thumb_fn", arch=thumb_arch)

    class _BV:
        def get_function_at(self, a):
            return fn if a == 0x12e74 else None

    _stub_code_context(monkeypatch, instance, {"name": "thumb_fn"})
    recorded = {}

    def fake_safe(bv_, address, arch=None):
        recorded["arch"] = arch
        return "bx pc" if arch is thumb_arch else "udf #0xd478"

    monkeypatch.setattr(instance.ctx, "_safe_disassembly", fake_safe)
    ctx = instance._address_context(_BV(), 0x12e74, include_disasm=True)
    assert recorded["arch"] is thumb_arch           # used the target function's arch
    assert ctx["disasm"] == "bx pc"                 # not the ARM misdecode


def test_address_context_disasm_respects_explicit_arch(monkeypatch):
    # An explicitly-passed arch (the caller's arch for a code-ref site) is used
    # as-is, not overridden by the function-at-address derivation.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    explicit = object()
    _stub_code_context(monkeypatch, instance, None)
    recorded = {}

    def fake_safe(bv_, address, arch=None):
        recorded["arch"] = arch
        return "x"

    monkeypatch.setattr(instance.ctx, "_safe_disassembly", fake_safe)
    instance._address_context(object(), 0x1000, include_disasm=True, arch=explicit, assume_code=True)
    assert recorded["arch"] is explicit


def test_struct_field_offset_grammar_matches_set(monkeypatch):
    # A zero-padded offset that `struct field set` accepts (_parse_address) must
    # also resolve in rename/delete; int(text, 0) rejected leading zeros (#25).
    bridge, instance, builder = _struct_instance(
        monkeypatch, [_FakeStructMember(0, "a"), _FakeStructMember(8, "b")])
    res = instance._op_struct_field_rename(
        None, {"struct_name": "S", "old_name": "0008", "new_name": "bb"})
    assert res["old_name"] == "b"
    assert builder.members[1].name == "bb"


def test_single_mutation_missing_field_message_is_neutral(monkeypatch):
    # A single mutation missing a field must NOT be described as a "batch
    # operation" -- it names the op kind and field neutrally (#30).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    with pytest.raises(bridge.OperationFailure) as e:
        instance._apply_operation(bv, {"op": "local_rename", "function": "f", "variable": "v"})
    assert e.value.status == "invalid_request"
    assert "new_name" in str(e.value)
    assert "batch" not in str(e.value).lower()


def test_internal_keyerror_not_mislabeled_as_missing_field(monkeypatch):
    # A KeyError raised deeper than request-field reads (e.g. BN internals) must
    # NOT be reported as a missing request field now that fields are validated
    # up front (#30).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    def boom(ctx, b, o):
        raise KeyError("some_internal_key")

    monkeypatch.setattr(bridge.mutation_engine, "_op_rename_symbol", boom)
    with pytest.raises(bridge.OperationFailure) as e:
        instance._apply_operation(bv, {"op": "rename_symbol", "identifier": "x", "new_name": "y"})
    assert "missing required field" not in str(e.value)
    assert "KeyError" in str(e.value)


def test_unsupported_op_kind_uses_neutral_wording(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    with pytest.raises(bridge.OperationFailure) as e:
        instance._apply_operation(bv, {"op": "nonsuch_op"})
    assert e.value.status == "unsupported"
    assert "batch" not in str(e.value).lower()


def test_types_declare_missing_declaration_is_invalid_request(monkeypatch):
    # types_declare missing 'declaration' must report invalid_request naming the
    # field, not crash with a raw KeyError from the pre-apply snapshot pass (#30).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    with pytest.raises(bridge.OperationFailure) as e:
        instance._apply_operation(bv, {"op": "types_declare"})
    assert e.value.status == "invalid_request"
    assert "declaration" in str(e.value)


def test_affected_type_names_tolerates_malformed_types_declare(monkeypatch):
    # The pre-apply snapshot pass must not raise on a types_declare op missing
    # 'declaration'; it skips it so _apply_operation can reject it cleanly.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    assert instance._affected_type_names(None, [{"op": "types_declare"}]) == []


def test_struct_snapshot_uses_find_type_resolved_name(monkeypatch):
    """Struct ops resolve names case-insensitively via _find_type and commit
    under the resolved name; the snapshot pipeline must snapshot under that
    same name or affected_types silently loses the layout diff (#95)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    struct_type = _FakeType(
        "struct MyStruct", width=8,
        members=[_FakeMember(0, "field_0", "int64_t")],
    )
    bv = _FakeBV(types_={"MyStruct": struct_type})
    ops = [{"op": "struct_field_set", "struct_name": "mystruct"}]

    assert instance._affected_type_names(bv, ops) == ["MyStruct"]
    snapshots = instance._capture_type_snapshots(bv, ops)
    assert "MyStruct" in snapshots
    assert snapshots["MyStruct"]["layout"]


def test_struct_snapshot_tolerates_unresolvable_name(monkeypatch):
    # _find_type raises on unknown names; the pre-apply snapshot pass must fall
    # back to the raw name (and skip the snapshot) so _apply_operation can
    # surface the precise error instead.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(types_={})
    ops = [{"op": "struct_field_set", "struct_name": "NoSuchStruct"}]

    assert instance._affected_type_names(bv, ops) == ["NoSuchStruct"]
    assert instance._capture_type_snapshots(bv, ops) == {}


def test_diff_type_snapshots_populates_name(monkeypatch):
    """affected_types entries carry `name` (= the qualified type name), not just
    `type_name`, so an agent keying off .name doesn't read a real type change as
    anonymous/failed (#211)."""
    bridge = _load_bridge(monkeypatch)
    me = bridge.mutation_engine
    before: dict = {}
    after = {"Config": {"decl": "struct Config", "layout": "struct Config {\n  int a;\n}"}}
    diffs = me._diff_type_snapshots(None, before, after)
    assert len(diffs) == 1
    assert diffs[0]["name"] == "Config"
    assert diffs[0]["type_name"] == "Config"   # back-compat alias retained
    assert diffs[0]["changed"] is True


# ---------------------------------------------------------------------------
# Batch 5: bridge-side validation (#94 comment guard, #100 count validation)
# ---------------------------------------------------------------------------


def test_get_comment_rejects_both_locators(monkeypatch):
    # #94: a raw socket client sending both function and address must be rejected
    # (the CLI mutex group doesn't protect raw clients).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: _FakeBV())
    with pytest.raises(RuntimeError, match="not both"):
        instance._get_comment("active", "0x1000", "main")


def test_get_comment_function_aggregates_body_comments(monkeypatch):
    """`comment get --function` must aggregate ALL comments within the function's
    address range (matching `comment list`'s attribution), not just the
    entry-address comment -- a function with body comments but no entry comment
    previously reported (no comment), contradicting `comment list` (#203)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    fn = _FakeFunction(0x40a810, "ns::Cls::rwBuffer")
    fn.basic_blocks = [_FakeBasicBlock(0x40a810, 0x410900)]  # covers the body addrs

    class _CommentBV(_FakeBV):
        def get_comment_at(self, addr):
            return self.address_comments.get(int(addr), "")

    bv = _CommentBV(functions=[fn])
    # body comments only -- nothing at the entry address 0x40a810
    bv.address_comments = {0x4105f4: "validate length here", 0x410858: "off-by-one risk"}
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance.ctx, "_find_function", lambda _bv, ident: fn)

    result = instance._get_comment(None, None, "ns::Cls::rwBuffer")
    assert result["has_comment"] is True
    assert result["function"] == "ns::Cls::rwBuffer"
    assert [c["address"] for c in result["comments"]] == ["0x4105f4", "0x410858"]
    assert [c["comment"] for c in result["comments"]] == [
        "validate length here", "off-by-one risk"
    ]


def test_op_set_comment_rejects_both_locators(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._op_set_comment(_FakeBV(), {"op": "set_comment", "function": "main", "address": "0x1000", "comment": "x"})
    assert exc.value.status == "invalid_request"


def test_op_delete_comment_rejects_both_locators(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._op_delete_comment(_FakeBV(), {"op": "delete_comment", "function": "main", "address": "0x1000"})
    assert exc.value.status == "invalid_request"


def test_sections_rejects_negative_count(monkeypatch):
    # #100: _sections re-enforces the count contract for raw callers.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: _FakeBV(sections={}))
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._sections("active", offset=-1)
    assert exc.value.status == "invalid_request"


def test_list_comments_rejects_negative_count(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    bv.address_comments = {}
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    with pytest.raises(bridge.OperationFailure) as exc:
        instance._list_comments("active", limit=-3)
    assert exc.value.status == "invalid_request"


def test_list_comments_returns_paging_envelope(monkeypatch):
    # #131: comment list returns the {items,total,offset,limit,returned,has_more}
    # envelope (parity with strings/imports/sections), not a bare list.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    bv.address_comments = {0x1000: "first", 0x2000: "second", 0x3000: "third"}
    bv.get_functions_containing = lambda addr: []
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    res = instance._list_comments("active", offset=0, limit=2)
    assert res["total"] == 3
    assert res["returned"] == 2
    assert res["has_more"] is True
    assert [i["comment"] for i in res["items"]] == ["first", "second"]


def test_bind_list_comments_tolerates_none_limit(monkeypatch):
    """The CLI sends `limit: None` (the --limit default) for a bare `comment
    list`, so the key is present with value None. The binder must read that as
    "no limit" -- guarding on `params.get("limit") is not None`, like every
    sibling list binder -- not on key presence, which does int(None) and crashes
    the command with a raw `int() argument must be ... not 'NoneType'` TypeError
    (regression from the #131 paging-envelope adoption)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    bv.address_comments = {0x1000: "first", 0x2000: "second"}
    bv.get_functions_containing = lambda addr: []
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # Exactly what the CLI forwards for `bn comment list` with no --limit/--offset.
    res = bridge._bind_list_comments(instance, {"query": None, "offset": 0, "limit": None}, "active")
    assert res["total"] == 2
    assert res["returned"] == 2
    assert res["limit"] is None       # None means "no limit", parity with siblings
    assert res["has_more"] is False

    # An explicit --limit still pages.
    res2 = bridge._bind_list_comments(instance, {"query": None, "offset": 0, "limit": 1}, "active")
    assert res2["returned"] == 1
    assert res2["has_more"] is True
