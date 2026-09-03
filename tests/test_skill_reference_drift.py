"""Docs-drift guards for `skills/bn/reference/` (#650, #651).

Agents read the skill reference as the API surface, not `--help`. Where the
reference omits a flag, agents conclude it does not exist: in one 5-agent dogfood
run **three** independently re-filed shipped features as missing capabilities, and
two wrote the same ad-hoc python to sort functions by size -- the workaround the
`--sort` flag was built to prevent.

So these are not style checks. Each assertion below encodes a documented claim that
the code must keep true (or a flag whose omission provably cost real work), and they
fail when either side drifts:

* every batch op kind and its required fields appear in `mutating.md` -- the batch
  path has no argparse layer, so an undocumented field name is discovered only by an
  atomic apply failing and reverting N good ops;
* the shipped `function list` / mutation flags agents re-invented stay documented;
* the compact mutation-status key table names exactly the keys the formatters emit
  -- #684 made that schema the signal that stops an agent discarding real work, so
  an omitted key is a consumer that never learns to check it;
* the JSON-envelope leaf keys the reference promises match what the handlers emit
  (`local list` is the exception that was documented in #248, lost in the SKILL.md ->
  reference/ split, and is now fixed at the source instead).
"""
from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path

import pytest

REFERENCE = Path(__file__).resolve().parent.parent / "skills" / "bn" / "reference"
READING = REFERENCE / "reading.md"
MUTATING = REFERENCE / "mutating.md"


@pytest.fixture(scope="module")
def mutation_engine():
    """The engine module, imported against a stub `binaryninja` (no BN needed)."""
    sys.modules.setdefault("binaryninja", types.ModuleType("binaryninja"))
    return importlib.import_module("bn_agent_bridge.mutation_engine")


def test_mutating_reference_documents_every_batch_op(mutation_engine):
    """#650: `mutation_engine.REQUIRED_FIELDS` defines 16 batch ops; `mutating.md`
    used to name TWO. Every op AND every required field name must be documented --
    a guessed field name fails at apply time and, the batch being atomic, takes
    every good op with it (one agent lost 12 that way)."""
    text = MUTATING.read_text(encoding="utf-8")
    missing_ops = [op for op in mutation_engine.REQUIRED_FIELDS if f"`{op}`" not in text]
    assert not missing_ops, f"batch ops undocumented in mutating.md: {missing_ops}"

    missing_fields = []
    for op, fields in mutation_engine.REQUIRED_FIELDS.items():
        for field in fields:
            if f"`{field}`" not in text:
                missing_fields.append(f"{op}.{field}")
    assert not missing_fields, (
        f"required batch fields undocumented in mutating.md: {missing_fields}")

    for op, groups in mutation_engine.REQUIRED_ONE_OF.items():
        for group in groups:
            for field in group:
                assert f"`{field}`" in text, f"{op} one-of field {field!r} undocumented"


def test_mutating_reference_documents_the_output_flags(mutation_engine):
    """#650/#645: three agents reported `batch apply` as having no summary mode and a
    `proto set` as unavoidably flooding context. Both flags shipped; only the docs
    were missing."""
    text = MUTATING.read_text(encoding="utf-8")
    for flag in ("--summary", "--quiet", "--verbose", "--format json", "--out"):
        assert flag in text, f"mutation output flag {flag!r} undocumented"
    # The compact default itself has to be stated, or an agent still expects the
    # old full-JSON default and parses the wrong thing.
    assert "status line" in text or "status summary" in text


def _documented_compact_status_keys() -> set[str]:
    """The keys named in the first column of `mutating.md`'s "Compact status
    keys" table. One row may name several keys (the counts share a row), so
    collect every backticked identifier in that cell."""
    section = MUTATING.read_text(encoding="utf-8").split("### Compact status keys", 1)
    assert len(section) == 2, "the compact-status key table section is gone"
    keys: set[str] = set()
    for line in section[1].splitlines():
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| key "):
            if keys:
                break                       # past the end of the table
            continue
        keys.update(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", line.split("|")[1]))
    return keys


def test_mutating_reference_documents_every_compact_status_key():
    """#684 made `measured` -- and the null-vs-zero count semantics around it --
    load-bearing for a JSON control loop, and the answer to "a value nobody is
    told to read is not loud enough" is that the compact schema is DOCUMENTED.
    A table that silently drifts from the emitted keys is the same
    remembered-exception failure mode as a forgotten `summary_transform`: it
    shipped omitting `kind` and `prototype_user_type_residue` (#630's residue
    flag, which a control loop must see). Pin the documented key set to the
    union of what `_mutation_summary` and `_go_rename_summary` actually emit
    across the measured, unmeasured and residue shapes."""
    from bn.formatters import _go_rename_summary, _mutation_summary

    measured = _mutation_summary({"success": True, "committed": True,
                                  "rolled_back": False, "results": [{"status": "noop"}]})
    unmeasured = _mutation_summary({"success": True, "committed": True, "results": []})
    residue = _mutation_summary({"success": False, "committed": False,
                                 "rolled_back": True, "message": "override stuck",
                                 "prototype_user_type_residue": True,
                                 "results": [{"status": "rollback_failed"}]})
    go = _go_rename_summary({"kind": "go_rename", "success": True, "committed": True,
                             "rolled_back": False, "go_renamed_candidates": 2,
                             "go_committed_count": 2, "go_verified_count": 2,
                             "go_failed_count": 0, "skipped_user_named": 1})
    emitted = set(measured) | set(unmeasured) | set(residue) | set(go)
    assert emitted == _documented_compact_status_keys()
    # ...and the table's "always present except prototype_user_type_residue"
    # caveat is itself true.
    assert "prototype_user_type_residue" in residue
    assert "prototype_user_type_residue" not in measured
    # #685: the two summary builders must keep emitting the SAME key set -- the
    # nulled unmeasured counts must not widen one side only.
    assert set(measured) == set(unmeasured) == set(go)


def test_reading_reference_documents_the_function_list_flags():
    """#650: `--sort` / `--reverse` / `--min-size` / `--demangle` all ship, and all
    were absent from `reading.md` -- so two agents wrote the same python to sort
    functions by size."""
    text = READING.read_text(encoding="utf-8")
    for flag in ("--sort", "--reverse", "--min-size", "--demangle", "--named", "--unnamed"):
        assert flag in text, f"`function list` flag {flag!r} undocumented in reading.md"


def test_reading_reference_leaf_keys_match_the_handlers(mutation_engine):
    """#651: `reading.md` claimed `items` is "always" the container while
    `local list` emitted only `locals` -- so `jq '.items[]'` on a function with 40
    recovered locals reported nothing, which reads like "no recovered variables".
    The exception was documented once (#248) and lost in a file split, so assert the
    CODE now satisfies the claim rather than re-documenting the exception."""
    read_decompile = importlib.import_module("bn_agent_bridge.read_decompile")

    class _Fn:
        start = 0x401120
        name = "handle_request"
        raw_name = "handle_request"

    class _Ctx:
        def _resolve_view(self, selector):
            return object()

        def _find_function(self, bv, identifier, contained=False):
            return _Fn()

        def _containment_meta(self, identifier, func):
            return None

    vars_mod = importlib.import_module("bn_agent_bridge.vars")
    original = vars_mod._list_locals
    try:
        vars_mod._list_locals = lambda fn: [{"name": "var_c", "is_parameter": False}]
        result = read_decompile._list_locals_for_function(_Ctx(), None, "handle_request")
    finally:
        vars_mod._list_locals = original

    assert result["items"] == result["locals"], "`items` must be the same list as `locals`"
    assert result["kind"] == "locals"
    text = READING.read_text(encoding="utf-8")
    assert "`local list` → `.items[]" in text or "local list` → `.items[]" in text
    # The old absolute claim must not have been restored verbatim.
    assert "`items` is **always** the data container" not in text


def test_reading_reference_documents_hex_string_addresses():
    """#653.7: `{"address": "0x401ed8"}` is a hex STRING everywhere; arithmetic needs
    `int(x, 16)`. Consistent, but undocumented -- one traceback to discover."""
    text = READING.read_text(encoding="utf-8")
    assert "hex STRING" in text or "hex strings" in text.lower()
