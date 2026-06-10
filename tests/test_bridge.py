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
    fake_bn = types.ModuleType("binaryninja")

    class SymbolType:
        FunctionSymbol = "SymbolType.FunctionSymbol"
        DataSymbol = "SymbolType.DataSymbol"
        ImportedFunctionSymbol = "SymbolType.ImportedFunctionSymbol"
        ImportedDataSymbol = "SymbolType.ImportedDataSymbol"
        ImportAddressSymbol = "SymbolType.ImportAddressSymbol"

    class Symbol:
        def __init__(self, symbol_type, address, name):
            self.type = symbol_type
            self.address = address
            self.name = name
            self.raw_name = name

    fake_bn.SymbolType = SymbolType
    fake_bn.Symbol = Symbol
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
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.delitem(sys.modules, package_name, raising=False)

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
    def __init__(self, *, functions=None, symbols=None, types_=None, arch=None, disassembly=None, instruction_lengths=None,
                 strings=None, sections=None, segments=None, memory=None, code_refs=None, data_refs=None):
        self.functions = list(functions or [])
        self._symbols = list(symbols or [])
        self.types = dict(types_ or {})
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

    def get_type_by_name(self, name: str):
        return self.types.get(str(name))

    def define_user_type(self, name: str, type_obj):
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

    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance, "_guess_affected_functions", lambda bv, operations: [])
    monkeypatch.setattr(instance, "_capture_function_snapshots", lambda bv, functions: {})
    monkeypatch.setattr(instance, "_capture_type_snapshots", lambda bv, operations: {})
    monkeypatch.setattr(instance, "_diff_snapshots", lambda before, after: [])
    monkeypatch.setattr(instance, "_diff_type_snapshots", lambda before, after: [])
    monkeypatch.setattr(
        instance,
        "_apply_operation",
        lambda bv, op, restores=None: {
            "op": "rename_symbol",
            "kind": "function",
            "address": "0x401000",
            "new_name": "player_update",
            "requested": {"identifier": "sub_401000", "new_name": "player_update"},
        },
    )
    monkeypatch.setattr(
        instance,
        "_verify_operation",
        lambda bv, result: {
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
    monkeypatch.setattr(instance, "_find_function", lambda _bv, ident: fn)
    monkeypatch.setattr(instance, "_find_variable_selector", lambda _f, sel: (var, False))
    monkeypatch.setattr(instance, "_find_var_for_restore", lambda _f, identifier, storage, is_parameter: var)
    monkeypatch.setattr(instance, "_local_id", lambda _f, _v, is_parameter: "lid")

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
    monkeypatch.setattr(instance, "_find_function", lambda _bv, ident: fn)
    monkeypatch.setattr(instance, "_find_variable_selector", lambda _f, sel: (var, False))
    monkeypatch.setattr(instance, "_local_id", lambda _f, _v, is_parameter: "lid")

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

    assert str(exc_info.value) == "Type not found: zzzzzzzz"


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._function_info("active", "player_update")

    assert result["prototype"] == "int32_t player_update(int32_t arg1)"
    assert result["return_type"] == "int32_t"
    assert result["calling_convention"] == "__cdecl"
    assert result["size"] is None


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance, "_comment_map", lambda bv, func: {})
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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance, "_comment_map", lambda bv, func: {})
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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance, "_comment_map", lambda bv, func: {})
    # No fake lineardisassembly module installed -> _pseudo_c_text raises and we
    # fall back to wrapped HLIL produced by _function_text.
    monkeypatch.setattr(instance, "_function_text", lambda bv, func, **kw: "    return 1;")

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance, "_comment_map", lambda bv, func: {})
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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance, "_comment_map", lambda bv, func: {})
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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance, "_comment_map", lambda bv, func: {})
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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._list_functions("active")

    assert [item["address"] for item in result] == ["0x401000", "0x402000"]


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._list_functions("active", min_address="0x401800", max_address="0x402fff")

    assert [item["address"] for item in result] == ["0x402000"]


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._search_functions("active", "attach|detach", regex=True)

    assert [item["name"] for item in result] == ["load_attachment", "detach_player"]


def test_search_functions_rejects_invalid_regex(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x401000, "load_attachment")])
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure, match="Invalid function regex"):
        instance._search_functions("active", "(", regex=True)


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    rows = instance._callsites(
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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    rows = instance._callsites(
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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    rows = instance._callsites(
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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    rows = instance._callsites(
        "active",
        "crt_rand",
        within_identifiers=["fx_queue_add_random"],
        context=1,
    )

    assert len(rows) == 1
    assert rows[0]["call_addr"] == "0x500015"
    assert rows[0]["hlil_statement"] is None
    assert rows[0]["pre_branch_condition"] is None


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    rows = instance._callsites(
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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    rows = instance._callsites(
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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._pointer_table("active", "0x3000", entries=2)

    assert result["entries"][0]["value"] == "0x401001"
    assert result["entries"][0]["target"]["normalized"] == "0x401000"
    assert result["entries"][0]["target"]["thumb_adjusted"] is True
    assert result["entries"][0]["target"]["function"]["name"] == "handler"
    assert result["entries"][0]["target"]["function"]["exact_start"] is True
    assert result["entries"][0]["target"]["context"]["address"] == "0x401000"
    assert result["entries"][1]["target"]["function"] is None
    assert result["entries"][1]["target"]["plausible"] is False


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._pointer_table("active", "0x64ea0", entries=2)

    assert all(entry["plausible"] is False for entry in result["entries"])
    assert any("executable segment" in warning for warning in result["warnings"])
    assert any("low confidence" in warning for warning in result["warnings"])


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._message_lens("active", "HeadUnitInfo", limit=5, table_entries=2)

    assert result["count"] == 1
    match = result["matches"][0]
    assert match["type_string"]["value"] == "common.HeadUnitInfo"
    assert match["xrefs"]["code_refs"][0]["function"] == "build_type_name"
    assert match["metadata_table_windows"][0]["address"] == "0x6000"
    assert match["metadata_table_windows"][0]["entries"][0]["target"]["thumb_adjusted"] is True


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query=None, offset=0, limit=100, min_length=4)

    assert len(result) == 2
    assert result[0]["value"] == "hello"
    assert result[1]["value"] == "helloworld"


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query=None, offset=0, limit=100, section=".rodata")

    assert len(result) == 1
    assert result[0]["value"] == "rodata"


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query=None, offset=0, limit=100, no_crt=True)

    assert len(result) == 1
    assert result[0]["value"] == "player"


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query=None, offset=0, limit=100,
                               min_length=4, section=".rodata", no_crt=True)

    assert len(result) == 1
    assert result[0]["value"] == "player"


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._strings(None, query="vehicle|headunit", offset=0, limit=100, regex=True)

    assert [item["value"] for item in result] == ["Vehicle", "HeadUnitInfo"]


def test_strings_invalid_regex_is_actionable(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: _FakeBV())

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._sections(None)

    assert len(result) == 3
    text_sec = result[0]
    assert text_sec["name"] == ".text"
    assert text_sec["start"] == "0x1000"
    assert text_sec["end"] == "0x5000"
    assert text_sec["length"] == 0x4000
    assert text_sec["semantics"] == "ReadOnlyCode"
    assert text_sec["readable"] is True
    assert text_sec["writable"] is False
    assert text_sec["executable"] is True

    data_sec = result[1]
    assert data_sec["name"] == ".data"
    assert data_sec["semantics"] == "ReadWriteData"
    assert data_sec["writable"] is True

    rodata_sec = result[2]
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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._sections(None, query="data")

    assert len(result) == 2
    names = [s["name"] for s in result]
    assert ".rodata" in names
    assert ".data" in names


def test_sections_null_segment_omits_rwx(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        sections={".bss": _FakeSection(".bss", 0x9000, 0xa000)},
        segments={0x1000: _FakeSegment(readable=True, writable=False, executable=True)},
    )
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._sections(None)

    assert len(result) == 1
    assert "readable" not in result[0]


def test_sections_without_segments_omits_rwx(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _BareView:
        def __init__(self):
            self.sections = {".text": _FakeSection(".text", 0x1000, 0x2000)}

    bv = _BareView()
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._sections(None)

    assert len(result) == 1
    assert "readable" not in result[0]
    assert "writable" not in result[0]
    assert "executable" not in result[0]


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)

    assert len(result) == 3
    kinds = {item["name"]: item["kind"] for item in result}
    assert kinds["printf"] == "function"
    assert kinds["__stdout"] == "data"
    assert kinds["iat_entry"] == "address"


def test_imports_sorts_by_library_kind_name(monkeypatch):
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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)

    assert result[0]["name"] == "alpha"
    assert result[0]["library"] == "liba"
    assert result[1]["name"] == "zebra"
    assert result[1]["library"] == "libz"


def test_imports_bn_sentinel_namespace_is_not_surfaced_as_library(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    sym = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x1000, "memcpy")
    sym.short_name = "memcpy"
    sym.namespace = "BNINTERNALNAMESPACE"

    bv = _FakeBV(symbols=[sym])
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._imports(None)

    # The meaningless sentinel must not masquerade as a real library...
    assert result[0]["library"] is None
    # ...but stays available under an honestly-named field.
    assert result[0]["namespace"] == "BNINTERNALNAMESPACE"


def test_imports_summary_includes_needed_libraries(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    sym = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x1000, "memcpy")
    sym.short_name = "memcpy"
    sym.namespace = "BNEXTERNALNAMESPACE"

    bv = _FakeBV(symbols=[sym])
    bv.libraries = ["libssl.so.1.1", "libc.so.6"]
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._read(None, "4096", 4)

    assert result["address"] == "0x1000"
    assert result["hex"] == "41424344"
    assert result["ascii"] == "ABCD"


def test_read_unmapped_address_raises_naming_the_address(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(memory={0x1000: b"\x41\x42\x43\x44"})
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    with pytest.raises(RuntimeError, match="0xdead.*not mapped"):
        instance._read(None, "0xdead", 16)


def test_read_short_read_returns_mapped_bytes_with_note(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(memory={0x1000: b"\x01\x02\x03\x04"})
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    with pytest.raises(RuntimeError, match="0x5000.*not inside an executable segment"):
        instance._function_create(None, "0x5000", False)

    assert bv.added == []


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


class _FakeFileBV:
    def __init__(self, filename: str, session_id: str = "0", view_name: str = "ELF"):
        self.file = types.SimpleNamespace(session_id=session_id, filename=filename)
        self.view_type = types.SimpleNamespace(name=view_name)


def _register_views(bridge, *bvs):
    bridge._headless_views.clear()
    bridge._headless_views.extend(bvs)


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


def test_list_functions_count_only_returns_count(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[
        _FakeFunction(0x1000, "a"),
        _FakeFunction(0x2000, "b"),
        _FakeFunction(0x3000, "c"),
    ])
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    assert instance._list_functions(None, count_only=True) == {"count": 3}
    # count must match the full listing length
    assert len(instance._list_functions(None)) == 3


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
    ):
        self._address = address
        self._operation_name = operation
        self._params = params or []
        self._vars_read = vars_read or []
        self.dest = dest

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    assert result["trace"][1]["reason"] == "function_parameter_or_global"


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._backward_slice("active", "test_func", "0x10020", arg_index=0)

    assert result["step_count"] == 1
    assert result["trace"][0]["ssa_var"] == "arg1#0"
    assert result["trace"][0]["terminates"] is True
    assert result["trace"][0]["reason"] == "function_parameter_or_global"


def test_backward_slice_no_call_at_address(monkeypatch):
    """Address with no call instruction should raise."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = _FakeMLILFunction(instructions=[])
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    with pytest.raises(bridge.OperationFailure, match="out of range"):
        instance._backward_slice("active", "test_func", "0x10010", arg_index=5)


def test_backward_slice_no_mlil(monkeypatch):
    """Function with no MLIL should raise."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    fn = _FakeFunction(0x10000, "test_func")
    fn.medium_level_il = None
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    # ...and bottomed out at the callee's parameter.
    assert trace[-1]["terminates"] is True
    assert trace[-1]["reason"] == "function_parameter_or_global"


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    with pytest.raises(bridge.OperationFailure, match="SSA form does not support"):
        instance._backward_slice("active", "test_func", "0x10010", arg_index=0,
                                 view="llil", interprocedural=True)


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._imports(None, summary=True)

    assert result["total_symbols"] == 4
    assert result["namespaces"] == {"libc": 3, "libfoo": 1}
    assert result["by_kind"] == {"function": 3, "data": 1}


def test_imports_summary_empty(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._imports(None, summary=True)

    assert result["total_symbols"] == 0
    assert result["namespaces"] == {}
    assert result["by_kind"] == {}


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._xrefs(None, "malloc")

    assert result["import_resolved"] is True
    assert result["import_name"] == "malloc"
    assert result["address"] == "0x20000"


def test_xrefs_import_symbol_raises_for_unknown_symbol(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(functions=[_FakeFunction(0x10000, "main")])
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._search_functions("active", "system", exact=True)

    assert len(result) == 1
    assert result[0]["name"] == "system"
    assert result[0]["address"] == "0x401000"


def test_search_functions_exact_case_insensitive(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "System"),
            _FakeFunction(0x402000, "QSystemPlugin"),
        ]
    )
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._search_functions("active", "system", exact=True)

    assert len(result) == 1
    assert result[0]["name"] == "System"


def test_search_functions_exact_no_match(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "system_ex"),
            _FakeFunction(0x402000, "_system"),
        ]
    )
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    result = instance._search_functions("active", "system", exact=True)

    assert len(result) == 0


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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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

    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(
        instance,
        "_resolve_type_field",
        lambda view, spec: {"type_name": "Foo", "offset": 4, "field_name": "bar"},
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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    full = instance._imports(None)
    assert [item["name"] for item in full] == ["fn0", "fn1", "fn2", "fn3", "fn4"]
    page = instance._imports(None, offset=1, limit=2)
    assert page == full[1:3]
    # summary aggregates the whole set regardless of offset/limit.
    summary = instance._imports(None, summary=True, offset=1, limit=2)
    assert summary["total_symbols"] == 5


def test_sections_pagination_slices_offset_and_limit(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    secs = {
        ".a": _FakeSection(".a", 0x1000, 0x1100),
        ".b": _FakeSection(".b", 0x2000, 0x2100),
        ".c": _FakeSection(".c", 0x3000, 0x3100),
    }
    bv = _FakeBV(sections=secs)
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    full = instance._sections(None)
    assert [s["name"] for s in full] == [".a", ".b", ".c"]
    page = instance._sections(None, offset=1, limit=1)
    assert [s["name"] for s in page] == [".b"]


def test_py_exec_reports_script_error_with_type_prefix(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance, "_comment_map", lambda bv, func: {})
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
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)

    # Quick-loaded: strings analysis hasn't run, so refuse rather than return [].
    bridge._quick_loaded_views.add(bv)
    with pytest.raises(RuntimeError, match="loaded with --quick"):
        instance._strings(None, query=None, offset=0, limit=100)

    # Once analysis lands, strings answers normally (here: genuinely empty).
    bridge._quick_loaded_views.discard(bv)
    assert instance._strings(None, query=None, offset=0, limit=100) == []


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
    assert instance._strings(None, query=None, offset=0, limit=100) == []


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
