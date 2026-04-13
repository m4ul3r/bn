from __future__ import annotations

import importlib
import importlib.util
import sys
import threading
import time
import types
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


class _FakeBasicBlock:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end


class _FakeInstructionInfo:
    def __init__(self, length: int):
        self.length = length


class _FakeArch:
    def __init__(self, lengths=None):
        self.max_instr_length = 16
        self.lengths = dict(lengths or {})

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
                 strings=None, sections=None, segments=None):
        self.functions = list(functions or [])
        self._symbols = list(symbols or [])
        self.types = dict(types_ or {})
        self.arch = arch or _FakeArch(instruction_lengths)
        self._disassembly = dict(disassembly or {})
        self._instruction_lengths = dict(instruction_lengths or {})
        self.strings = list(strings or [])
        self.sections = dict(sections or {})
        self._segments = dict(segments or {})

    def get_function_at(self, address: int):
        for fn in self.functions:
            if int(fn.start) == int(address):
                return fn
        return None

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

    def get_disassembly(self, address: int):
        return self._disassembly.get(int(address), "")

    def get_code_refs(self, address: int):
        return []

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
        lambda bv, op: {
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
