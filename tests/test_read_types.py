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


def test_types_declare_discloses_the_include_root_it_actually_searched_825(
        monkeypatch, tmp_path):
    # #825 item 3: the header's parent directory becomes an implicit include
    # root, and the result never said so. The assertion that matters is not
    # that a key exists but that the DISCLOSED root is the SEARCHED one --
    # a disclosure computed independently could drift from the parse and
    # confidently name a directory that was never on the path.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    recorded = {}

    class _Platform:
        def parse_types_from_source(self, source, **kwargs):
            recorded["kwargs"] = kwargs
            return _ParseResult(types={"Player": "struct Player"})

    class _SourceBV(_FakeBV):
        def __init__(self):
            super().__init__()
            self.platform = _Platform()

    nested = tmp_path / "inc"
    nested.mkdir()
    header_path = nested / "outer.h"
    header_path.write_text("typedef struct Player { int hp; } Player;", encoding="utf-8")
    bv = _SourceBV()

    result = instance._op_types_declare(bv, {
        "op": "types_declare",
        "declaration": header_path.read_text(encoding="utf-8"),
        "source_path": str(header_path),
    })

    assert result["include_root"] == recorded["kwargs"]["include_dirs"][0]
    assert result["include_root"] == str(nested.resolve())


def test_types_declare_inline_declaration_discloses_no_include_root_825(monkeypatch):
    # Must-not-fire twin: an inline declaration has no file, so there is no
    # implicit root and the result must not grow the key at all. A reader
    # checking `"include_root" in result` must not see it on every declare.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _SourceBV(_FakeBV):
        def parse_types_from_string(self, declaration):
            return _ParseResult(types={"Player": "struct Player"})

    result = instance._op_types_declare(_SourceBV(), {
        "op": "types_declare",
        "declaration": "typedef struct Player { int hp; } Player;",
    })

    assert "include_root" not in result


def test_types_declare_refuses_source_without_named_types(monkeypatch):
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

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._op_types_declare(
            bv,
            {
                "op": "types_declare",
                "declaration": "extern const GUID GUID_SysKeyboard;",
                "source_path": "/tmp/win32_min.h",
            },
        )
    assert exc.value.status == "invalid_request"
    assert "no named types" in exc.value.message
    assert exc.value.observed["defined_types"] == {}
    assert exc.value.observed["parsed_functions"] == ["DirectInput8Create"]
    assert exc.value.observed["parsed_variables"] == ["GUID_SysKeyboard"]
    assert bv.defined == []


def test_types_declare_refuses_a_partially_dropped_declaration_760(monkeypatch):
    """#760: the platform parser discards a declaration whose name collides with a
    built-in type WITHOUT raising, so a multi-declaration string defined one type,
    dropped another, and still reported `verified`. The drop is now a refusal --
    nothing is applied, so the caller cannot read a partial declaration as success.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    class _Platform:
        def parse_types_from_source(self, source, **kwargs):
            names = [name for name in ("widget_cfg_t",) if name in source]
            return _ParseResult(types={name: f"struct {name}" for name in names})

    class _PartialBV(_FakeBV):
        def __init__(self):
            super().__init__()
            self.platform = _Platform()
            self.defined: list[tuple[str, object]] = []

        def get_type_by_name(self, name):
            return None

        def define_user_type(self, name, type_obj):
            self.defined.append((name, type_obj))

    bv = _PartialBV()

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._op_types_declare(
            bv,
            {
                "op": "types_declare",
                "declaration": (
                    "struct uint32_t { int shadow_x; }; "
                    "struct widget_cfg_t { int y; };"
                ),
            },
        )

    assert exc.value.status == "invalid_request"
    assert "define no named type" in exc.value.message
    assert exc.value.observed["dropped_declarations"] == [
        "struct uint32_t { int shadow_x; };"
    ]
    assert exc.value.observed["defined_types"] == ["widget_cfg_t"]
    assert bv.defined == []          # refused before anything is applied


def _declare_probe_bv(monkeypatch, *, good=("widget_a_t", "widget_b_t"),
                      variables=("widget_inst",), raise_when=None):
    """A view whose platform parser behaves the way the real one does for #760.

    `good` are the names it materializes; anything else in an inspected fragment is
    dropped with no named type and no exception -- exactly how a declaration whose
    name collides with a built-in type disappears. `variables` come back through the
    `variables` slot, so a variable declaration with a brace initializer parses to
    zero types WITHOUT raising (the shape that broke an earlier cut of the guard).
    `raise_when` models a fragment that cannot parse on its own.
    """

    class _Platform:
        def parse_types_from_source(self, source, **kwargs):
            if raise_when is not None and raise_when(source):
                raise SyntaxError(f"cannot parse {source!r} alone")
            return _ParseResult(
                types={name: _FakeType(name, width=4, members=[])
                       for name in good if name in source},
                variables={name: "int" for name in variables if name in source},
            )

    class _DeclareBV(_FakeBV):
        def __init__(self):
            super().__init__()
            self.platform = _Platform()
            self.defined: list[str] = []
            self.types_defined: dict[str, object] = {}

        def get_type_by_name(self, name):
            return self.types_defined.get(str(name))

        def define_user_type(self, name, type_obj):
            self.types_defined[str(name)] = _FakeType(str(name), width=4, members=[])
            self.defined.append(str(name))

    return _DeclareBV()


def _declare(instance, bv, declaration):
    return instance._op_types_declare(
        bv, {"op": "types_declare", "declaration": declaration}
    )


def test_types_declare_refuses_a_drop_after_tricky_syntax_760(monkeypatch):
    """#760: the drop is caught even when earlier fragments carry the syntax that
    makes splitting hard -- a `;` inside a struct body and inside a comment."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _declare_probe_bv(monkeypatch)

    with pytest.raises(bridge.OperationFailure) as exc:
        _declare(instance, bv, (
            "struct widget_a_t { char *s; }; "        # a `;` inside a body
            "/* ; */ struct uint32_t { int x; };"     # a `;` inside a comment
        ))

    assert exc.value.status == "invalid_request"
    assert any("uint32_t" in item for item in exc.value.observed["dropped_declarations"])
    assert bv.defined == []


def test_types_declare_refuses_a_drop_after_a_digit_separator_760(monkeypatch):
    """#760 review item 4: an odd `'` (a C++14 digit separator, which the parser
    accepts) used to open a quote, merge every later fragment into one and switch the
    guard off for the rest of the string."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _declare_probe_bv(monkeypatch)

    with pytest.raises(bridge.OperationFailure) as exc:
        _declare(instance, bv, (
            "struct widget_a_t { int a; }; "
            "static const int widget_k = 1'000; "
            "struct uint32_t { int x; };"
        ))

    assert exc.value.status == "invalid_request"
    assert any("uint32_t" in item for item in exc.value.observed["dropped_declarations"])


def test_types_declare_refuses_a_drop_behind_a_nested_attribute_prefix_760(monkeypatch):
    """#760 review follow-up: `__attribute__((aligned(8)))` nests its parens, so a
    `[^)]*` prefix match stopped at the inner `)` and the drop stayed silent behind a
    prefix the docs claimed was covered."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _declare_probe_bv(monkeypatch)

    with pytest.raises(bridge.OperationFailure) as exc:
        _declare(instance, bv, (
            "__attribute__((aligned(8))) struct uint32_t { int x; }; "
            "struct widget_a_t { int a; };"
        ))

    assert exc.value.status == "invalid_request"
    assert any("uint32_t" in item for item in exc.value.observed["dropped_declarations"])
    assert bv.defined == []


def test_types_declare_refuses_a_drop_after_a_wide_char_literal_760(monkeypatch):
    """#760 review follow-up: the digit-separator rule must key on a DIGIT, not any
    alphanumeric -- `L';'` is a character literal, and treating its opening quote as a
    separator left the literal's closing quote to swallow the rest of the string."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _declare_probe_bv(monkeypatch)

    with pytest.raises(bridge.OperationFailure) as exc:
        _declare(instance, bv, (
            "struct widget_a_t { int a; }; "
            "wchar_t widget_w = L';'; "
            "struct uint32_t { int x; };"
        ))

    assert exc.value.status == "invalid_request"
    assert any("uint32_t" in item for item in exc.value.observed["dropped_declarations"])


def test_types_declare_allows_a_variable_declaration_with_a_brace_initializer_760(monkeypatch):
    """#760 review item 1: a variable declaration with a brace initializer is a USAGE,
    not a definition. An earlier cut of the classifier saw `{`, treated it as a body
    and refused the request -- breaking working input and diagnosing a variable as a
    dropped type declaration."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _declare_probe_bv(monkeypatch, variables=("widget_inst",))

    result = _declare(instance, bv, (
        "struct widget_a_t { int a; }; "
        "struct widget_known_t widget_inst = {};"
    ))

    assert set(result["defined_types"]) == {"widget_a_t"}
    assert result["count"] == 1
    assert bv.defined == ["widget_a_t"]          # nothing refused, nothing extra applied


def test_types_declare_allows_a_multi_declaration_that_all_land_760(monkeypatch):
    """#760 negative control: ordinary multi-type input is untouched."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _declare_probe_bv(monkeypatch)

    result = _declare(instance, bv, (
        "struct widget_a_t { int a; }; struct widget_b_t { int b; };"
    ))

    assert set(result["defined_types"]) == {"widget_a_t", "widget_b_t"}
    assert result["count"] == 2


def test_types_declare_allows_a_fragment_that_depends_on_an_earlier_one_760(monkeypatch):
    """#760: a fragment that only fails to parse alone because it uses a type an
    earlier fragment defines is skipped, not refused -- refusing it would turn a
    working declaration into an error."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _declare_probe_bv(
        monkeypatch,
        good=("widget_inner_t", "widget_outer_t"),
        raise_when=lambda source: (
            "widget_outer_t" in source and "widget_inner_t {" not in source
        ),
    )

    result = _declare(instance, bv, (
        "struct widget_inner_t { int a; }; "
        "struct widget_outer_t { struct widget_inner_t inner; };"
    ))

    assert set(result["defined_types"]) == {"widget_inner_t", "widget_outer_t"}


def test_declared_types_verifier_rejects_an_empty_apply_result(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    verified = instance._verify_operation(_FakeBV(), {
        "op": "types_declare", "defined_types": {},
        "requested": {"declaration": "struct Example { int value; };"},
    })
    assert verified["status"] == "verification_failed"
    assert verified["observed"]["defined_types"] == {}


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

    # A real idempotent declaration still resolves and is a successful noop.
    repeated = instance._op_types_declare(
        bv, {"op": "types_declare", "declaration": "struct DamageGaugeController { int state; };"}
    )
    assert instance._verify_operation(bv, repeated)["status"] == "noop"
    assert bv.get_type_by_name("DamageGaugeController") is not None

    # Matching before-state cannot turn a missing live type into a noop.
    monkeypatch.setattr(bv, "get_type_by_name", lambda name: None)
    missing = instance._verify_operation(bv, repeated)
    assert missing["status"] == "verification_failed"
    assert missing["observed"]["defined_types"]["DamageGaugeController"] is None


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("output", [[], ["--format", "json"], ["--format", "json", "--summary"]])
def test_empty_type_parse_rolls_back_and_reports_reason(
        monkeypatch, capsys, preview, output):
    import bn.cli

    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeCommentMutationBV(comments={0x1000: "original"})
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(bv, "parse_types_from_string", lambda declaration: _ParseResult(), raising=False)
    result = instance._mutation("active", preview, [
        {"op": "set_comment", "address": "0x1000", "comment": "temporary"},
        {"op": "types_declare", "declaration": "struct Example { int value; };"},
    ])
    assert result["success"] is False
    assert result["committed"] is False
    assert result["rolled_back"] is True
    assert bv.get_comment_at(0x1000) == "original"
    assert result["results"][-1]["status"] == "invalid_request"

    monkeypatch.setattr(bn.cli, "send_request", lambda *a, **k: {"ok": True, "result": result})
    argv = ["types", "declare", "--target", "active", "struct Example { int value; };"]
    assert bn.cli.main(argv + (["--preview"] if preview else []) + output) == 3
    stdout = capsys.readouterr().out
    if "--summary" in output:
        summary = json.loads(stdout)
        assert summary["ok"] is False
        assert summary["failed_count"] == 1
        assert summary["noop_count"] == 0
        assert "no named types" in summary["first_error"]
    elif output:
        payload = json.loads(stdout)
        assert payload["ok"] is False
        assert payload["results"][-1]["status"] == "invalid_request"
    else:
        assert "mutation: committed" not in stdout
        assert "first_error:" in stdout
        assert "no named types" in stdout


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


def test_parse_type_or_hint_resolves_const_qualified_namespaced_pointer(monkeypatch):
    """A const/volatile-qualified ::-qualified named pointer (common verbatim from
    the decompiler) must resolve the same as its unqualified form. BN's C parser
    rejects the ::-name, and the #200 fallback stripped only the trailing '*', so a
    leading/trailing const made the base-name lookup miss and the whole parse
    failed-closed with 'could not parse' (#389). A cv-qualifier is layout- and
    indirection-preserving, so resolving the unqualified type is safe."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    me = bridge.mutation_engine

    class _NsBV(_FakeBV):
        def parse_type_string(self, decl):
            raise SyntaxError(
                "error: <unknown>:1:1 use of undeclared identifier 'ns'\n1 error generated."
            )

    bv = _NsBV(
        functions=[],
        qualified_types_={("ns", "demo", "Foo"): _FakeType("struct ns::demo::Foo")},
    )

    # every cv-qualified pointer form resolves to the same pointer the
    # unqualified form would (the const-ness is dropped, not the type).
    for decl in (
        "ns::demo::Foo const*",
        "const ns::demo::Foo*",
        "ns::demo::Foo const *",
        "volatile ns::demo::Foo*",
        "const ns::demo::Foo const*",
    ):
        t, name = me._parse_type_or_hint(
            instance.ctx, bv, {"op": "local_retype"}, decl, label="type"
        )
        assert str(t) == "struct ns::demo::Foo*", decl
        assert name is None, decl

    # a bare cv-qualified value (no pointer) resolves too
    t2, _ = me._parse_type_or_hint(
        instance.ctx, bv, {"op": "local_retype"}, "ns::demo::Foo const", label="type"
    )
    assert str(t2) == "struct ns::demo::Foo"

    # an interior space in a template name must NOT be mangled by qualifier stripping
    # (only leading/trailing const/volatile and trailing '*' are stripped)
    bv2 = _NsBV(
        functions=[],
        qualified_types_={
            ("std", "vector<std::pair<int, long> >", "iterator"): _FakeType(
                "struct std::vector<std::pair<int, long> >::iterator"
            )
        },
    )
    t3, _ = me._parse_type_or_hint(
        instance.ctx, bv2, {"op": "local_retype"},
        "const std::vector<std::pair<int, long> >::iterator*", label="type",
    )
    assert str(t3) == "struct std::vector<std::pair<int, long> >::iterator*"


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




def test_render_type_layout_expands_anonymous_aggregate(monkeypatch):
    """An anonymous nested union/struct member rendered a bare `0x0004: union `
    line, hiding the inner members from the CLI (#370.2). They must be expanded
    (indented) in the text layout and present in the JSON `members[]`."""
    bridge = _load_bridge(monkeypatch)
    ctx = bridge.BinaryNinjaBridge().ctx

    inner = _FakeType("union ", width=4, members=[
        _FakeMember(0x0, "iv", "int32_t"),
        _FakeMember(0x0, "fv", "float"),
    ], type_class="StructureTypeClass")
    outer = _FakeType("struct Outer", width=0x8, members=[
        _FakeMember(0x0, "tag", "int32_t"),
        _FakeMember(0x4, "u", inner),       # anonymous-aggregate-typed member
    ])

    # text: inner members are visible (indented), not just "0x0004: union u"
    text = ctx._render_type_layout(outer)
    assert "iv" in text and "fv" in text, text

    # JSON: a structured members[] with the inner aggregate nested
    entry = ctx._type_entry("Outer", outer)
    assert isinstance(entry.get("members"), list)
    by_name = {m.get("name"): m for m in entry["members"]}
    assert "tag" in by_name and "u" in by_name
    inner_members = by_name["u"].get("members")
    assert isinstance(inner_members, list)
    assert {m.get("name") for m in inner_members} == {"iv", "fv"}


# ---------------------------------------------------------------------------
# typedef aliases and nesting depth (#674)
# ---------------------------------------------------------------------------


class _TypedefRef(_FakeType):
    """A BN NamedTypeReference: the alias carries no members of its own and
    resolves to the registered underlying type through `.target(bv)`."""

    def __init__(self, decl, target):
        super().__init__(decl, type_class="NamedTypeReferenceClass")
        self._target = target

    def target(self, bv):
        return self._target


def _registered(type_obj, name):
    type_obj.registered_name = types.SimpleNamespace(name=name)
    return type_obj


def test_struct_show_follows_typedef_to_underlying_struct(monkeypatch):
    """`struct show <typedef>` reported "Type is not a struct-like type: <alias>"
    because the alias itself has no members, while the underlying registered
    struct does. It must follow the typedef and render that body (layout and
    JSON members[]), while `types show` keeps reporting the alias (#674)."""
    bridge = _load_bridge(monkeypatch)
    ctx = bridge.BinaryNinjaBridge().ctx
    body = _registered(
        _FakeType("struct", width=0x10, members=[_FakeMember(0x0, "data", "uint8_t[16]")]),
        "_AnonymousBody",
    )
    alias = _TypedefRef("struct _AnonymousBody", body)
    bv = _FakeBV(types_={"AliasName": alias, "_AnonymousBody": body})
    monkeypatch.setattr(ctx, "_resolve_view", lambda selector: bv)

    entry = bridge.read_types._type_info(ctx, None, "AliasName", require_struct=True)
    assert entry["name"] == "_AnonymousBody"
    assert "0x0000: uint8_t[16] data" in entry["layout"]
    assert [m.get("name") for m in entry["members"]] == ["data"]

    # `types show` is unrestricted and still reports the alias entry unchanged.
    plain = bridge.read_types._type_info(ctx, None, "AliasName")
    assert plain["name"] == "AliasName"
    assert "members" not in plain


def test_struct_show_follows_a_typedef_chain(monkeypatch):
    """A typedef of a typedef resolves through every hop to the struct body
    (BN's `.target()` may itself hand back another NamedTypeReference)."""
    bridge = _load_bridge(monkeypatch)
    ctx = bridge.BinaryNinjaBridge().ctx
    body = _registered(
        _FakeType("struct", width=0x4, members=[_FakeMember(0x0, "hp", "int32_t")]),
        "_Body",
    )
    mid = _TypedefRef("struct _Body", body)
    outer = _TypedefRef("MidAlias", mid)
    bv = _FakeBV(types_={"OuterAlias": outer, "MidAlias": mid, "_Body": body})
    monkeypatch.setattr(ctx, "_resolve_view", lambda selector: bv)

    entry = bridge.read_types._type_info(ctx, None, "OuterAlias", require_struct=True)
    assert entry["name"] == "_Body"
    assert "0x0000: int32_t hp" in entry["layout"]


def test_struct_show_distinguishes_scalar_from_an_unfollowable_alias(monkeypatch):
    """Two different failures must not share one misleading message: an alias
    that WAS followed to a scalar is genuinely not struct-like, while an alias
    whose chain cannot be followed at all (self-referential here, and the same
    path covers a raised target()) must say so and name the reason (#674 review:
    the previous assertion pinned both to 'not a struct-like type')."""
    bridge = _load_bridge(monkeypatch)
    ctx = bridge.BinaryNinjaBridge().ctx
    scalar = _FakeType("int32_t", type_class="IntegerTypeClass")
    broken = _TypedefRef("RaisingAlias", None)

    def _raise(bv):
        raise RuntimeError("symbol resolution failed")

    broken.target = _raise
    loop = _TypedefRef("SelfAlias", None)
    loop._target = loop                      # pathological self-referential typedef
    bv = _FakeBV(types_={"TileCount": scalar, "SelfAlias": loop, "RaisingAlias": broken})
    monkeypatch.setattr(ctx, "_resolve_view", lambda selector: bv)

    # (i) a followed scalar keeps the original, accurate message.
    with pytest.raises(RuntimeError, match=r"not a struct-like type: TileCount"):
        bridge.read_types._type_info(ctx, None, "TileCount", require_struct=True)

    # (ii) an unresolvable chain reports the alias AND why, distinctly.
    with pytest.raises(RuntimeError) as cycle:
        bridge.read_types._type_info(ctx, None, "SelfAlias", require_struct=True)
    assert "SelfAlias" in str(cycle.value)
    assert "not a struct-like type" not in str(cycle.value)
    assert "cyclic" in str(cycle.value)

    with pytest.raises(RuntimeError) as raised:
        bridge.read_types._type_info(ctx, None, "RaisingAlias", require_struct=True)
    assert "RaisingAlias" in str(raised.value)
    assert "not a struct-like type" not in str(raised.value)
    assert "resolve" in str(raised.value).lower()


def _nested_anonymous_aggregate(levels: int):
    """An anonymous struct nested *levels* deep: the outermost type holds member
    `nested1`, whose own type holds `nested2`, ... down to a scalar `leaf` at the
    innermost depth."""
    current = _FakeType("struct", width=1, members=[_FakeMember(0x0, "leaf", "uint8_t")])
    for level in range(levels, 0, -1):
        current = _FakeType("struct", width=1, members=[_FakeMember(0x0, f"nested{level}", current)])
    return current


def _flagged_json_depth(entry):
    """Depth (0 = top-level member) of the first JSON member entry flagged
    `truncated: true` on the single-child path, or None when nothing is cut."""
    depth = 0
    nodes = entry.get("members")
    while nodes:
        node = nodes[0]
        if node.get("truncated"):
            return depth
        nodes = node.get("members")
        depth += 1
    return None


def test_deeply_nested_layout_renders_past_level_five(monkeypatch):
    """A 10-level anonymous aggregate used to stop expanding after level 5 with
    no marker, so the inner structs read as empty. Both renderers must reach the
    innermost members (#674)."""
    bridge = _load_bridge(monkeypatch)
    ctx = bridge.BinaryNinjaBridge().ctx
    deep = _nested_anonymous_aggregate(10)

    text = ctx._render_type_layout(deep)
    assert "nested1" in text and "nested10" in text
    assert "0x0000: uint8_t leaf" in text

    entry = ctx._type_entry("Deep", deep)
    assert _flagged_json_depth(entry) is None
    node = entry
    for level in range(1, 11):
        children = {m.get("name"): m for m in node["members"]}
        assert f"nested{level}" in children, (level, children)
        node = children[f"nested{level}"]
    assert [m.get("name") for m in node["members"]] == ["leaf"]


def test_nested_layout_truncation_is_disclosed(monkeypatch):
    """Past the nesting cap the cut must be disclosed, not silent: a text marker
    line and a `truncated: true` JSON flag on the entry whose children were cut,
    both agreeing on the depth (#674)."""
    bridge = _load_bridge(monkeypatch)
    ctx = bridge.BinaryNinjaBridge().ctx
    deep = _nested_anonymous_aggregate(40)

    text = ctx._render_type_layout(deep)
    marker = [line for line in text.splitlines() if "truncated at depth" in line]
    assert marker, text
    assert "leaf" not in text                      # the cut is real
    assert "nested1" in text                       # ...but the outer levels stay

    entry = ctx._type_entry("Deep", deep)
    flagged = _flagged_json_depth(entry)
    assert flagged is not None
    declared = int(marker[0].split("truncated at depth", 1)[1].split(":", 1)[0].strip())
    assert declared == flagged + 1


def test_self_referential_anonymous_aggregate_still_terminates(monkeypatch):
    """The cap must also bound a pathological aggregate that nests into itself:
    rendering terminates and discloses the cut instead of recursing forever."""
    bridge = _load_bridge(monkeypatch)
    ctx = bridge.BinaryNinjaBridge().ctx
    cycle = _FakeType("struct", width=8, members=[], type_class="StructureTypeClass")
    cycle.members.append(_FakeMember(0x0, "self", cycle))

    text = ctx._render_type_layout(cycle)
    assert "truncated at depth" in text
    assert _flagged_json_depth(ctx._type_entry("Cycle", cycle)) is not None


def test_types_listing_discloses_quick_analysis_state():
    """#820: `types` answers on a --quick view, so both envelopes carry the
    view-level analysis state -- what the loader parsed is not the whole type set
    analysis would produce."""
    bridge_state = importlib.import_module("bn_agent_bridge.bridge_state")
    read_types = importlib.import_module("bn_agent_bridge.read_types")
    bv = _FakeBV(types_={"Widget": _FakeType("struct Widget")})

    class _Ctx:
        def _resolve_view(self, sel):
            return bv

        def _type_entry(self, name, type_obj):
            return {"name": name, "kind": "struct", "decl": "struct Widget"}

    full = read_types._types(_Ctx(), None, query=None, offset=0, limit=None)
    assert full["kind"] == "types"
    assert full["analysis_state"] == "full"
    assert full["partial"] is False
    full_count = read_types._types(_Ctx(), None, query=None, offset=0, limit=None,
                                   count_only=True)
    assert full_count["analysis_state"] == "full"
    assert full_count["partial"] is False

    bridge_state._quick_loaded_views.add(bv)
    try:
        quick = read_types._types(_Ctx(), None, query=None, offset=0, limit=None)
        quick_count = read_types._types(_Ctx(), None, query=None, offset=0, limit=None,
                                        count_only=True)
    finally:
        bridge_state._quick_loaded_views.discard(bv)

    assert quick["analysis_state"] == "quick"
    assert quick["partial"] is True
    assert quick_count["analysis_state"] == "quick"
    assert quick_count["partial"] is True


def test_render_type_list_text_warns_when_quick_loaded():
    """#820: the type listing states its own partiality in text -- once, on the
    envelope (the per-row recursion is handed a bare list and must not repeat it)."""
    from bn.formatters import _render_type_list_text
    value = {
        "kind": "types",
        "items": [{"name": "Widget", "kind": "struct", "decl": "struct Widget"}],
        "total": 1, "offset": 0, "limit": None, "returned": 1, "has_more": False,
        "analysis_state": "quick", "partial": True,
    }
    out = _render_type_list_text(value)
    assert out.startswith("WARNING: target is quick-loaded; type list is partial.")
    assert "bn refresh" in out
    assert "Widget | struct" in out
    assert out.count("WARNING") == 1

    full = _render_type_list_text({**value, "analysis_state": "full", "partial": False})
    assert "WARNING" not in full


# --- #675 item 1: a reserved keyword tag is refused, not silently created ---


def test_a_reserved_keyword_tag_is_refused_675():
    """#675 item 1, restated. The issue says `struct _Bool {...}` SHADOWS the
    builtin; measured, there is no builtin to shadow -- on a clean view
    `_Bool` does not parse at all (`unknown type name '_Bool'`). The
    declaration CREATES the name, and a later bare `_Bool` field then means
    this struct: three of them lay out as 0xC where C says 3 bytes, at
    `verified`.

    This completes `_declarations_without_named_types` (#760) rather than
    adding a second instrument beside it. A tag colliding with a builtin the
    parser DOES implement is already refused; this is the same refusal for
    the reserved names it does not. The asymmetry was the defect.
    """
    from bn_agent_bridge import mutation_engine as me
    with pytest.raises(me.OperationFailure) as excinfo:
        me._refuse_reserved_keyword_tags(None, {"op": "types_declare"},
                                         [("_Bool", object())])
    assert excinfo.value.status == "invalid_request"
    msg = excinfo.value.message
    assert "_Bool" in msg
    # It must name the CONSEQUENCE, like the bitfield guard beside it, not
    # merely the syntax -- the harm is generated by a LATER command.
    assert "0xC" in msg or "verified" in msg


def test_a_struct_tag_spelling_is_normalised_before_the_check_675():
    """A `struct X` declaration may report its name as `struct X` rather than
    `X`, so a textual equality on the reported name would miss it."""
    from bn_agent_bridge import mutation_engine as me
    with pytest.raises(me.OperationFailure):
        me._refuse_reserved_keyword_tags(None, {"op": "types_declare"},
                                         [("struct _Complex", object())])


def test_ordinary_and_underscored_tags_still_declare_675():
    """Must-not-fire twin, and the one that bounds the set: the guard covers
    RESERVED keywords, not every underscore-prefixed identifier. Widening it
    to `_`-names would break legitimate modelling of `_mystate`-style tags,
    which is a bigger harm than the one being fixed."""
    from bn_agent_bridge import mutation_engine as me
    for tag in ("cfg", "struct cfg_t", "_mystate", "_Boolean", "bool_t"):
        me._refuse_reserved_keyword_tags(None, {"op": "types_declare"},
                                         [(tag, object())])   # must not raise


def test_the_guard_reads_parsed_names_not_the_source_text_675():
    """Checked over the names BN would DEFINE, so a `_Bool` in a comment or
    used as a FIELD name cannot trip it -- the ground truth is the parse
    result, not the string the user typed."""
    from bn_agent_bridge import mutation_engine as me
    # A declaration mentioning _Bool only in a field position parses to a
    # differently-named type, so the guard sees no offending tag.
    me._refuse_reserved_keyword_tags(None, {"op": "types_declare"},
                                     [("holder", object())])
