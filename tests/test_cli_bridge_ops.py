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

# A floor, not a target: the extractor should find "a whole CLI's worth" of
# op literals. If a refactor legitimately drops call sites below this, lower
# it deliberately -- but a silent drop to near-zero means the extractor
# broke (e.g. a rename of `_call`/`_mutate`), not that the CLI shrank.
_MIN_PLAUSIBLE_OP_COUNT = 40

# One op name each shape -- and ONLY that shape -- produces in today's
# tree. The count floor above cannot detect a stale table entry on its own:
# dropping `_mutate` still leaves 59 of 77 literals, well clear of the
# floor, so a renamed `_mutate` would leave every mutation op string
# unchecked and still green. Losing a sentinel names the exact broken shape.
_SHAPE_SENTINELS = {
    "_call": "decompile",  # commands/function.py
    "_mutate": "set_prototype",  # commands/mutation.py
    "send_request": "load_status",  # commands/admin.py
    "_send_request_to_instance": "doctor",  # commands/admin.py
    'dict literal {"id", "op", "params"}': "cancel_request",  # transport.py
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


def extract_cli_wire_ops(root: Path) -> set[str]:
    """Statically collect every wire op string constant the CLI can send.

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
    """
    ops: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                index = _OP_ARG_INDEX.get(_callee_name(node.func) or "")
                if index is not None:
                    ops |= _op_arg_strings(node, index)
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
                        ops |= _string_constants(value)
    return ops


def test_cli_wire_ops_are_registered_on_bridge(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    cli_ops = extract_cli_wire_ops(CLI_SRC_ROOT)

    # A broken extractor (e.g. a rename of `_call`/`send_request` this table
    # was not updated for) must not silently pass by finding nothing.
    assert len(cli_ops) >= _MIN_PLAUSIBLE_OP_COUNT, (
        f"extractor only found {len(cli_ops)} CLI wire op string(s) under "
        f"{CLI_SRC_ROOT} -- expected at least {_MIN_PLAUSIBLE_OP_COUNT}; this "
        "looks like a broken extractor (a call-shape table entry went stale), "
        "not a real shrink of the CLI"
    )

    stale = {
        shape for shape, sentinel in _SHAPE_SENTINELS.items() if sentinel not in cli_ops
    }
    assert not stale, (
        f"extractor found no literal for call shape(s) {sorted(stale)} -- "
        "their entry in _OP_ARG_INDEX (or the dict-literal rule) went "
        "stale, so every op string sent through them is now unchecked"
    )

    missing = cli_ops - bridge.REGISTRY.names()
    assert not missing, f"CLI op strings missing from REGISTRY: {missing}"


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
    shape here: a fixture tree with plenty of `_call`/`send_request`
    literals (>= the floor) but zero `_mutate` call sites, standing in for
    a `_mutate` rename that the extractor's table was never updated for.
    The count floor alone would pass this; the sentinel must not.
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

    ops = extract_cli_wire_ops(src_dir)

    # The count floor alone is satisfied...
    assert len(ops) >= _MIN_PLAUSIBLE_OP_COUNT
    # ...but the `_mutate` shape produced nothing, so its sentinel is absent,
    # and the sentinel check catches it even though the floor did not (this
    # fixture only exercises the `_call` shape, so the other four shapes'
    # sentinels are also naturally absent -- the point is that `_mutate`
    # specifically is now named, which a bare count floor cannot do).
    assert _SHAPE_SENTINELS["_mutate"] not in ops
    stale = {shape for shape, sentinel in _SHAPE_SENTINELS.items() if sentinel not in ops}
    assert "_mutate" in stale


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
