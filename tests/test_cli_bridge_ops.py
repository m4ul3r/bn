"""#623: every wire op string the CLI can send must exist on the bridge.

CLI command handlers and the bridge op registry are independent surfaces --
wire op names are bare string literals scattered across `src/bn/commands/*`,
`cli.py`, and `transport.py`, while handlers are registered with `@op("...")`
into `REGISTRY` on the bridge. `tests/test_op_registry.py` freezes the
bridge's own lock-set membership, but nothing previously cross-checked the
CLI side against it: a typo or a one-sided rename only surfaced as a runtime
`Unknown operation: ...` when that exact command was exercised.

This module statically collects every wire op string constant the CLI can
send (via `ast`, not regex -- a truncated/renamed literal must still parse)
and asserts it is a subset of `REGISTRY.names()`. No real BN is needed: the
bridge loads through the same `_bridge_fakes._load_bridge` seam
`test_op_registry.py` uses.

This is a ONE-WAY subset check by design (#623 acceptance criteria): a
registered op the CLI never sends is fine and expected -- nothing here
requires the reverse. (`doctor` and `cancel_request` are NOT examples of
that: both are in fact extracted by this module's own rules --
`_send_request_to_instance` and the dict-literal rule below, respectively
-- and both appear in the extraction; today the CLI-side extraction happens
to cover the bridge's full `REGISTRY.names()`.)

Extraction is bucketed per call shape, and each shape must still yield its
documented sentinel op -- that is what catches a stale entry in the shape
table (a renamed `_mutate` leaves 59 of 77 literals, clearing any plausible
count floor, yet silently unchecks every mutation op). The subset assertion
runs BEFORE those anti-vacuity guards so a plain typo -- which blanks a
sentinel when it lands on one -- always fails with the offending op name.

Batch-manifest *inner* op names carried inside a `bn batch` JSON payload
are out of scope: the dict-literal rule below only fires on a hand-built
transport *request envelope* (an `"op"` key with sibling `"id"`/`"params"`
keys), never on an arbitrary dict that merely happens to carry an `"op"`
key.
"""
from __future__ import annotations

import ast
from pathlib import Path

from _bridge_fakes import _load_bridge

CLI_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "bn"

# 0-based positional index of the wire-op argument for each call shape this
# extractor understands, also accepted as an `op=` keyword. Verified against
# the actual current signatures (re-check this table if a signature moves):
#   cli.py:    def _call(args, op, params=None, *, ...)
#   cli.py:    def _mutate(args, op, params, *, ...)
#   transport.py: def send_request(op, *, params=None, ...)
#   transport.py: def _send_request_to_instance(instance, op, *, ...)
_OP_ARG_INDEX = {
    "_call": 1,
    "_mutate": 1,
    "send_request": 0,
    "_send_request_to_instance": 1,
}

# The extractor's fifth shape: a hand-built request envelope dict literal.
# Named like the call shapes above so per-shape attribution below can talk
# about all five uniformly.
_DICT_LITERAL_SHAPE = 'dict literal {"id", "op", "params"}'

# Catch-all floor, checked last (the per-shape sentinels below diagnose a
# stale shape-table entry far more precisely). It only earns its keep for
# the case where every sentinel survives yet the tree stops holding "a
# whole CLI's worth" of op literals -- e.g. a mass deletion of call sites.
# If a refactor legitimately drops below this, lower it deliberately.
_MIN_PLAUSIBLE_OP_COUNT = 40

# One op name each shape produces in today's tree, checked against that
# shape's OWN extraction (not the union), so the guard does not depend on
# the sentinel being unique across shapes. The count floor above cannot
# detect a stale table entry on its own: dropping `_mutate` still leaves 59
# of 77 literals, well clear of the floor, so a renamed `_mutate` would
# leave every mutation op string unchecked and still green. Losing a
# sentinel names the exact broken shape.
_SHAPE_SENTINELS = {
    "_call": "decompile",  # commands/function.py
    "_mutate": "set_prototype",  # commands/mutation.py
    "send_request": "load_status",  # commands/admin.py
    "_send_request_to_instance": "doctor",  # commands/admin.py
    _DICT_LITERAL_SHAPE: "cancel_request",  # transport.py
}


def _callee_name(func: ast.expr) -> str | None:
    """Return the plain name of a call target: `f(...)` or `obj.f(...)`."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _string_constants(node: ast.expr) -> set[str]:
    """Collect every string-constant leaf reachable through a ternary.

    `"a" if cond else "b"` contributes both `"a"` and `"b"` (today's
    `"load_binary_async" if detached else "load_binary"` in admin.py). Any
    other non-constant leaf -- a variable, a call, an f-string -- contributes
    nothing: a dynamic op name is out of scope for this static check, never
    guessed at.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        return _string_constants(node.body) | _string_constants(node.orelse)
    return set()


def _op_arg_strings(call: ast.Call, index: int) -> set[str]:
    for kw in call.keywords:
        if kw.arg == "op":
            return _string_constants(kw.value)
    if index < len(call.args):
        return _string_constants(call.args[index])
    return set()


def extract_cli_wire_ops_by_shape(root: Path) -> dict[str, set[str]]:
    """Statically collect the CLI's wire op string constants, per shape.

    Walks every `*.py` file under *root* with `ast` and collects string
    constants from:

    - the 2nd positional (or `op=`) argument of `_call` / `_mutate`
    - the 1st positional (or `op=`) argument of `send_request` /
      `_send_request_to_instance`
    - `"op": "<literal>"` entries in a hand-built request *envelope* dict
      literal -- one with sibling `"id"` and `"params"` keys, e.g.
      `transport.py`'s `cancel_request` payload -- never an arbitrary dict
      that merely happens to carry an `"op"` key (a batch-manifest entry, a
      decoded response row)

    Returns one bucket per shape (every `_OP_ARG_INDEX` key plus
    `_DICT_LITERAL_SHAPE`), always present, empty when that shape has no
    literal call site left. Attribution is what makes the anti-vacuity
    guard self-sufficient: a shape whose bucket went empty is a stale
    extractor entry regardless of what the other shapes happen to emit.
    """
    by_shape: dict[str, set[str]] = {shape: set() for shape in _OP_ARG_INDEX}
    by_shape[_DICT_LITERAL_SHAPE] = set()
    for path in sorted(root.rglob("*.py")):
        # Bytes, not `Path.read_text()`: the latter decodes with the process
        # locale, so a non-ASCII source file (several exist under `src/bn`)
        # raised `UnicodeDecodeError` under `LC_ALL=C PYTHONUTF8=0`.
        # `ast.parse` honours the file's own PEP 263 coding declaration.
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                shape = _callee_name(node.func) or ""
                index = _OP_ARG_INDEX.get(shape)
                if index is not None:
                    by_shape[shape] |= _op_arg_strings(node, index)
            elif isinstance(node, ast.Dict):
                keys = {
                    key.value
                    for key in node.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
                if not {"id", "params"} <= keys:
                    continue
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value == "op":
                        by_shape[_DICT_LITERAL_SHAPE] |= _string_constants(value)
    return by_shape


def extract_cli_wire_ops(root: Path) -> set[str]:
    """Every wire op string constant the CLI can send, shapes unioned."""
    return set().union(*extract_cli_wire_ops_by_shape(root).values())


def test_cli_wire_ops_are_registered_on_bridge(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    by_shape = extract_cli_wire_ops_by_shape(CLI_SRC_ROOT)
    cli_ops = set().union(*by_shape.values())

    # #623's actual invariant, asserted FIRST so its message always names the
    # offending op. A typo'd op is also a *missing sentinel* when it lands on
    # a sentinel name (`set_prototype` -> `set_prototipe`, the issue's own
    # example); checking vacuity first would have reported "shape went stale"
    # and never printed `set_prototipe`.
    missing = cli_ops - bridge.REGISTRY.names()
    assert not missing, f"CLI op strings missing from REGISTRY: {missing}"

    # Every shape the extractor knows about must have a sentinel, or a newly
    # added `_OP_ARG_INDEX` entry could sit unguarded.
    assert set(_SHAPE_SENTINELS) == set(by_shape), (
        "_SHAPE_SENTINELS and the extractor's shapes have drifted: "
        f"unsentinelled {sorted(set(by_shape) - set(_SHAPE_SENTINELS))}, "
        f"unknown {sorted(set(_SHAPE_SENTINELS) - set(by_shape))}"
    )

    # Anti-vacuity, per shape: this is the guard that can actually name what
    # broke, so it runs before the whole-tree count floor below (a `_call`
    # rename trips both, and "shape `_call` extracts nothing" is the useful
    # message, not "found 27, expected 40").
    stale = {
        shape
        for shape, sentinel in _SHAPE_SENTINELS.items()
        if sentinel not in by_shape[shape]
    }
    assert not stale, (
        f"extractor found no literal for call shape(s) {sorted(stale)} -- "
        "their entry in _OP_ARG_INDEX (or the dict-literal rule) went "
        "stale, so every op string sent through them is now unchecked"
    )

    # Catch-all floor: every shape still yields its sentinel, yet the tree as
    # a whole no longer holds a plausible CLI's worth of op literals.
    assert len(cli_ops) >= _MIN_PLAUSIBLE_OP_COUNT, (
        f"extractor only found {len(cli_ops)} CLI wire op string(s) under "
        f"{CLI_SRC_ROOT} -- expected at least {_MIN_PLAUSIBLE_OP_COUNT}; this "
        "looks like a broken extractor (a call-shape table entry went stale), "
        "not a real shrink of the CLI"
    )


def test_extractor_finds_op_from_call_positional_arg(tmp_path):
    src_dir = tmp_path / "bn"
    src_dir.mkdir()
    (src_dir / "fake_command.py").write_text(
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "def _fake_command(args):\n"
        '    return _call(args, "not_a_real_op", {})\n'
    )

    ops = extract_cli_wire_ops(src_dir)

    assert "not_a_real_op" in ops
    # Prove the subset check this extractor feeds would actually fire: a
    # bogus op is not a subset of any real registry.
    fake_registry_names = {"real_op_alpha", "real_op_beta"}
    assert not (ops <= fake_registry_names)
    missing = ops - fake_registry_names
    assert missing == {"not_a_real_op"}


def test_extractor_finds_op_from_send_request_and_op_kwarg(tmp_path):
    src_dir = tmp_path / "bn"
    src_dir.mkdir()
    (src_dir / "fake_admin.py").write_text(
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "def _fake_admin(args, detached):\n"
        '    cli.send_request("also_not_a_real_op", params={})\n'
        "    return cli._send_request_to_instance(\n"
        "        instance,\n"
        '        op="kwarg_not_a_real_op",\n'
        "        params={},\n"
        "    )\n"
    )

    ops = extract_cli_wire_ops(src_dir)

    assert ops == {"also_not_a_real_op", "kwarg_not_a_real_op"}


def test_extractor_follows_ternary_op_selection(tmp_path):
    src_dir = tmp_path / "bn"
    src_dir.mkdir()
    (src_dir / "fake_load.py").write_text(
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "def _fake_load(args, detached):\n"
        "    return cli.send_request(\n"
        '        "fake_op_detached" if detached else "fake_op_sync",\n'
        "        params={},\n"
        "    )\n"
    )

    ops = extract_cli_wire_ops(src_dir)

    assert ops == {"fake_op_detached", "fake_op_sync"}


def test_extractor_finds_op_from_transport_style_dict_literal(tmp_path):
    src_dir = tmp_path / "bn"
    src_dir.mkdir()
    (src_dir / "fake_transport.py").write_text(
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "def _fake_cancel(request_id):\n"
        "    payload = {\n"
        '        "id": request_id,\n'
        '        "op": "fake_cancel_request",\n'
        '        "params": {},\n'
        "    }\n"
        "    return payload\n"
    )

    ops = extract_cli_wire_ops(src_dir)

    assert ops == {"fake_cancel_request"}


def test_extractor_ignores_dynamic_op_variables(tmp_path):
    src_dir = tmp_path / "bn"
    src_dir.mkdir()
    (src_dir / "fake_dynamic.py").write_text(
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "def _fake_passthrough(args, op, params):\n"
        "    # A pass-through wrapper forwarding a caller-supplied op is NOT a\n"
        "    # new wire-name mint site; only the literal at the true call site\n"
        "    # (elsewhere) is in scope.\n"
        "    return _call(args, op, params)\n"
    )

    ops = extract_cli_wire_ops(src_dir)

    assert ops == set()


def test_sentinel_catches_mutate_rename_the_count_floor_misses(tmp_path):
    """#709 review: dropping `_mutate` alone leaves 59/77 literals in the
    real tree -- well clear of `_MIN_PLAUSIBLE_OP_COUNT`. Reproduce that
    shape here: a fixture tree with plenty of `_call` literals (>= the
    floor) but zero `_mutate` call sites, standing in for a `_mutate`
    rename that the extractor's table was never updated for. The count
    floor alone would pass this; the per-shape sentinel must not.
    """
    src_dir = tmp_path / "bn"
    src_dir.mkdir()
    calls = "\n".join(f'    _call(args, "op_{i}", {{}})' for i in range(_MIN_PLAUSIBLE_OP_COUNT))
    (src_dir / "fake_commands.py").write_text(
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "def _fake_many_calls(args):\n"
        f"{calls}\n"
        "    # note: no _mutate(...) call site anywhere in this fixture tree,\n"
        "    # standing in for a renamed `_mutate` the table wasn't updated for\n"
    )

    by_shape = extract_cli_wire_ops_by_shape(src_dir)
    ops = set().union(*by_shape.values())

    # The count floor alone is satisfied...
    assert len(ops) >= _MIN_PLAUSIBLE_OP_COUNT
    # ...but the `_mutate` bucket is empty, so its sentinel is absent and the
    # guard names that exact shape -- which a bare count floor cannot do.
    assert by_shape["_mutate"] == set()
    stale = {
        shape
        for shape, sentinel in _SHAPE_SENTINELS.items()
        if sentinel not in by_shape[shape]
    }
    assert "_mutate" in stale


def test_sentinel_check_is_attributed_not_unioned(tmp_path):
    """A sentinel is checked against its OWN shape's bucket.

    If the guard tested the union, another shape emitting the same op name
    would mask a stale entry: here `send_request` emits `_call`'s sentinel
    while `_call` itself has no call site left, so a union test would pass.
    """
    src_dir = tmp_path / "bn"
    src_dir.mkdir()
    (src_dir / "fake_masking.py").write_text(
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "def _fake_masking(args):\n"
        f'    return send_request("{_SHAPE_SENTINELS["_call"]}", params={{}})\n'
    )

    by_shape = extract_cli_wire_ops_by_shape(src_dir)
    ops = set().union(*by_shape.values())

    # A union-based guard would see `_call`'s sentinel and call the shape healthy.
    assert _SHAPE_SENTINELS["_call"] in ops
    # Attributed, it is correctly reported stale.
    assert by_shape["_call"] == set()
    stale = {
        shape
        for shape, sentinel in _SHAPE_SENTINELS.items()
        if sentinel not in by_shape[shape]
    }
    assert "_call" in stale


def test_typoed_sentinel_op_is_reported_as_missing_not_as_a_stale_shape(tmp_path):
    """#709 review: the issue's own example typo (`set_prototype` ->
    `set_prototipe`) lands ON a sentinel name, so it both breaks the subset
    check and blanks that sentinel. The subset assertion runs first, so the
    failure message names the typo (#623 acceptance criterion) instead of
    blaming a stale `_OP_ARG_INDEX` entry -- and the `_mutate` bucket is
    still non-empty, so the shape is (correctly) not reported stale at all.
    """
    src_dir = tmp_path / "bn"
    src_dir.mkdir()
    (src_dir / "fake_mutation.py").write_text(
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "def _fake_proto_set(args):\n"
        '    return _mutate(args, "set_prototipe", {})\n'
    )

    by_shape = extract_cli_wire_ops_by_shape(src_dir)
    ops = set().union(*by_shape.values())
    fake_registry_names = {_SHAPE_SENTINELS["_mutate"], "rename_symbol"}

    # The subset check fires and names the typo.
    assert ops - fake_registry_names == {"set_prototipe"}
    # The `_mutate` shape still extracts fine -- a typo is not a stale table
    # entry, and must not be misreported as one.
    assert by_shape["_mutate"] == {"set_prototipe"}


def test_extractor_ignores_op_key_on_non_envelope_dict(tmp_path):
    """#709 review: the dict-literal rule must only fire on a request
    *envelope* (sibling `"id"`/`"params"` keys), never on an arbitrary dict
    that happens to carry an `"op"` key -- e.g. a decoded response row or a
    batch-manifest entry embedded as a literal.
    """
    src_dir = tmp_path / "bn"
    src_dir.mkdir()
    (src_dir / "fake_response.py").write_text(
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "def _fake_batch_entry():\n"
        "    return {\n"
        '        "op": "fake_batch_only_op",\n'
        '        "target": "alpha.so",\n'
        "    }\n"
    )

    ops = extract_cli_wire_ops(src_dir)

    assert ops == set()


def test_extractor_parses_non_ascii_source_regardless_of_locale(tmp_path):
    """#709 review: `ast.parse` must run on bytes, not `Path.read_text()`,
    so extraction cannot fail depending on the process locale/encoding
    (`Path.read_text()` used the locale encoding and raised
    `UnicodeDecodeError` for a non-ASCII file under `LC_ALL=C
    PYTHONUTF8=0`). Write a file with a non-ASCII string literal and
    confirm it still parses and yields the expected op.
    """
    src_dir = tmp_path / "bn"
    src_dir.mkdir()
    (src_dir / "fake_unicode.py").write_bytes(
        (
            "from __future__ import annotations\n"
            "\n"
            "\n"
            "def _fake_command(args):\n"
            '    label = "\u2192 not a real op label"  # non-ASCII arrow\n'
            '    return _call(args, "not_a_real_op", {})\n'
        ).encode("utf-8")
    )

    ops = extract_cli_wire_ops(src_dir)

    assert ops == {"not_a_real_op"}
