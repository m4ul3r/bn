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
bridge-only op with no CLI caller (e.g. `doctor`, which the CLI only reaches
via `_send_request_to_instance`, or `cancel_request`, which is a hand-built
transport payload) is fine and expected -- nothing here requires the reverse.
Batch-manifest *inner* op names (the `{"op": ...}` entries a `bn batch` JSON
payload carries) are also out of scope; those are validated by the bridge at
apply time (#361-style), not by this static CLI-source sweep.
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
    - `"op": "<literal>"` entries in a hand-built request dict literal
      (today: `transport.py`'s `cancel_request` payload)
    """
    ops: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                index = _OP_ARG_INDEX.get(_callee_name(node.func) or "")
                if index is not None:
                    ops |= _op_arg_strings(node, index)
            elif isinstance(node, ast.Dict):
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
