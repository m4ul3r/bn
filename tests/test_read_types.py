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

from _bridge_fakes import *  # noqa: F401,F403


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
