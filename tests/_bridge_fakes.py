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
    def __init__(self, start: int, name: str, type_text: str = "int32_t()",
                 *, arch=None, total_bytes: int | None = None):
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
        # The #386 "looks like code" guard reads these off the created function.
        self.arch = arch
        self.total_bytes = total_bytes

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
    def __init__(self, lengths=None, *, name: str = "x86", address_size: int = 4,
                 instr_alignment: int = 1):
        self.name = name
        self.address_size = address_size
        self.max_instr_length = 16
        self.instr_alignment = instr_alignment
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
                 strings=None, sections=None, segments=None, memory=None, code_refs=None, data_refs=None, comments=None):
        self.functions = list(functions or [])
        self._comments = dict(comments or {})
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

    def get_instruction_length(self, address: int, arch=None):
        return self._instruction_lengths.get(int(address), 1)

    def get_disassembly(self, address: int, arch=None):
        return self._disassembly.get(int(address), "")

    def get_comment_at(self, address: int):
        return self._comments.get(int(address), "")

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


class _FakeSymbol:
    def __init__(self, type_name: str):
        self.type = type("_FakeSymType", (), {"name": type_name})()


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
    "function_create":     {"required": ("address",),                                        "one_of": (),                           "enum": {},                                     "cli": "bn function create"},
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


def _callsites_items(instance, *args, **kwargs):
    """Unwrap the {items,total,...} callsites envelope to the row list (#131).

    callsites now returns the same paging envelope as the sibling list ops; these
    tests assert on the rows, so unwrap once here rather than in every assertion.
    Also asserts the envelope shape so the contract stays covered."""
    result = instance._callsites(*args, **kwargs)
    assert isinstance(result, dict) and "items" in result and "total" in result
    return result["items"]


# --- function create: create+analyze a missed function ---


class _FakeFunctionCreateBV(_FakeMutationBV):
    def __init__(self, *, functions=None, segments=None, memory=None,
                 arch=None, instruction_lengths=None, created_total_bytes=4):
        super().__init__()
        self.functions = list(functions or [])
        self._segments = dict(segments or {})
        self._memory = dict(memory or {})
        # The #386 guard inspects the created function's arch / instruction
        # length / body size. Let a test inject a fixed-width arch, a
        # zero-length (undecodable) address, or an empty body.
        if arch is not None:
            self.arch = arch
        if instruction_lengths is not None:
            self._instruction_lengths = dict(instruction_lengths)
        self._created_total_bytes = created_total_bytes
        self.added: list[int] = []
        # Addresses BN has been told (via remove_user_function) the user does not
        # want a function at: a subsequent add_function there is dropped by
        # analysis. Modeling this is what gives the #304 regression test teeth.
        self._suppressed: set[int] = set()

    def add_function(self, addr: int):
        # The advisory auto-analysis hint: it DECLINES an address auto-analysis
        # already skipped (modeled here as a suppressed address) -- the #360
        # failure mode. The op no longer uses this for creation; kept so any
        # stray call is still modeled faithfully.
        self.events.append(("add_function", addr))
        if int(addr) in self._suppressed:
            return None
        fn = _FakeFunction(int(addr), f"sub_{addr:x}")
        self.functions.append(fn)
        return fn

    def create_user_function(self, addr: int):
        # The FORCED creation the op now uses (#360): it creates a user function
        # even at an address auto-analysis skipped/suppressed -- empirically the
        # forced path bypasses the remove_user_function suppression (#304).
        self.events.append(("create_user_function", addr))
        self.added.append(int(addr))
        self._suppressed.discard(int(addr))
        fn = _FakeFunction(int(addr), f"sub_{addr:x}", arch=self.arch,
                           total_bytes=self._created_total_bytes)
        self.functions.append(fn)
        return fn

    def remove_user_function(self, fn):
        # Model BN faithfully: function creation is not reliably undone by the
        # undo buffer, so revert_undo_actions never removes it -- only an explicit
        # removal does (#117). remove_user_function additionally records a
        # persistent user override that SUPPRESSES the address (#304 poison),
        # which only the auto add_function path honors.
        self.events.append(("remove_user_function", int(fn.start)))
        self.functions = [f for f in self.functions if int(f.start) != int(fn.start)]
        self._suppressed.add(int(fn.start))

    def remove_function(self, fn):
        # The non-poisoning removal: drops the function WITHOUT suppressing the
        # address, so a later create_user_function re-creates it (the #304 fix).
        self.events.append(("remove_function", int(fn.start)))
        self.functions = [f for f in self.functions if int(f.start) != int(fn.start)]

    def is_offset_executable(self, addr: int) -> bool:
        seg = self.get_segment_at(int(addr))
        return bool(seg is not None and seg.executable)


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


class _FakeFileBV:
    def __init__(self, filename: str, session_id: str = "0", view_name: str = "ELF"):
        self.file = types.SimpleNamespace(session_id=session_id, filename=filename)
        self.view_type = types.SimpleNamespace(name=view_name)


def _register_views(bridge, *bvs):
    bridge._headless_views.clear()
    bridge._headless_views.extend(bvs)


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


# ---------------------------------------------------------------------------
# Protocol hardening: request size cap and non-dict payloads
# ---------------------------------------------------------------------------


class _RecordingWriter:
    def __init__(self):
        self.data = b""

    def write(self, data):
        self.data += data


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


class _RehomingSaveBV:
    """Like _SaveBV, but mimics BN's create_database RE-HOMING the view's filename
    to the new .bndb -- the behavior _SaveBV omits and the cause of #256."""

    def __init__(self, filename: str):
        self.file = types.SimpleNamespace(filename=filename)
        self.created_with = None

    def create_database(self, out: str):
        self.created_with = out
        Path(out).write_text("bndb")
        self.file.filename = out  # real BN rebinds the live view to the new file
        return True


class _RehomingFailSaveBV:
    """create_database re-homes the live view's filename BEFORE failing (no file
    lands) -- BN can rebind bv.file.filename even on a save that doesn't
    complete, so the explicit-failure path must still restore identity (#256)."""

    def __init__(self, filename: str):
        self.file = types.SimpleNamespace(filename=filename)
        self.created_with = None

    def create_database(self, out: str):
        self.created_with = out
        self.file.filename = out  # re-home happens...
        return False              # ...but the save fails (nothing written)


class _RestoreFailFile:
    """A bv.file whose filename can be re-homed once (by create_database) but then
    refuses to be set back -- to exercise a restore that itself fails."""

    def __init__(self, name: str):
        self._name = name
        self.block = False

    @property
    def filename(self):
        return self._name

    @filename.setter
    def filename(self, value):
        if self.block:
            raise RuntimeError("cannot rebind filename")
        self._name = value


class _RestoreFailSaveBV:
    """create_database succeeds and re-homes the view, but restoring the original
    filename afterward raises -- the live view stays re-homed to the copy."""

    def __init__(self, filename: str):
        self.file = _RestoreFailFile(filename)
        self.created_with = None

    def create_database(self, out: str):
        self.created_with = out
        Path(out).write_text("bndb")
        self.file.filename = out  # re-home (allowed)
        self.file.block = True    # block the subsequent restore
        return True


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


def _stub_code_context(monkeypatch, instance, function_entry):
    # _address_context and its resolution/address-context helpers now live on the
    # BridgeContext seam (instance.ctx); patch the helpers where the method under
    # test resolves them.
    monkeypatch.setattr(instance.ctx, "_sections_at", lambda bv, a: [{"name": ".text"}])
    monkeypatch.setattr(instance.ctx, "_segment_at", lambda bv, a: {"name": "seg"})
    monkeypatch.setattr(instance.ctx, "_symbol_at", lambda bv, a: None)
    monkeypatch.setattr(instance.ctx, "_function_entry_for_address", lambda bv, a: function_entry)
    monkeypatch.setattr(instance.ctx, "_address_is_code", lambda bv, a: True)


__all__ = ['_load_bridge', '_FakeFunction', '_FakeBasicBlock', '_FakeInstructionInfo', '_FakeArch', '_FakeOperation', '_FakeConstPtr', '_FakeReg', '_FakeHLILInstructionNode', '_FAKE_HLIL_TYPES', '_FakeHLILInstruction', '_FakeLLILInstruction', '_FakeVariable', '_FakeStringRef', '_FakeCodeRef', '_FakeSection', '_FakeSegment', '_FakeBV', '_FakeType', '_FakeMember', '_FakeMutationBV', '_ParseResult', '_FakeSymbol', '_mutation_with_stubs', '_BATCH_OP_PARITY', '_minimal_valid_op', '_FakeCommentMutationBV', '_install_fake_pseudo_c', '_callsites_items', '_FakeFunctionCreateBV', '_local_retype_result', '_LoadBV', '_setup_load_test', '_FakeFileBV', '_register_views', '_ClosableBV', 'SSAVariable', '_FakeSSAVariable', '_FakeMLILInsn', '_FakeSSAFunction', '_FakeBlock', '_FakeMLILFunction', '_RecordingWriter', '_SaveBV', '_RehomingSaveBV', '_RehomingFailSaveBV', '_RestoreFailFile', '_RestoreFailSaveBV', '_FieldRefBV', '_FakeStructMember', '_FakeStructBuilder', '_struct_instance', '_AddableStructBuilder', '_struct_set_instance', '_pvs', '_dataflow_values_instance', '_stub_code_context']
