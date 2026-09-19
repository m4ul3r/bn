"""The wire contract, checked against a real consumer's decoder shape.

WHY THIS FILE EXISTS. Two panes spent a full day verifying every claim
against Binary Ninja and against each other, and never once against a
program that CONSUMES this tool's JSON. That blindness has one fixable
root cause: nothing in this repo decoded its own output with a strict
schema, so a field could change type or a container could be renamed and
every test here stayed green.

Two real defects came from exactly that gap:

  * The `syms`/`vars` -> `items` container rename (#275) silently blanked
    the consumer's data-symbol and data-var views. Its structs read the
    OLD key, and because both are `#[serde(default)]`, a missing container
    decodes as an EMPTY LIST rather than an error.
  * An unresolved CFG edge was briefly emitted as `{"to": null, ...}`.
    The consumer types that field `String` -- no `Option`, no default --
    so a null fails the decode; and because the failure is inside
    `Vec<CfgEdge>` the WHOLE cfg parse fails and the view renders empty.
    Caught before release by a cross-pane heads-up, not by any test.

The consumer is bn-tui (`/opt/bn-tui/src/bn.rs`), a separate repo. The
models below MIRROR its serde structs, with the source line noted so they
can be re-synced. They are deliberately a copy: this repo cannot import
that crate, and a test that invented its own idea of the contract would
prove nothing about the program that actually breaks.

WHAT "STRICT" MEANS HERE, and why each rule earns its place:

  * a required field (no serde default) MUST be present -- absent aborts
    the element, and for a `Vec` element that aborts the whole list;
  * a non-optional field MUST NOT be null -- this is the `to: null` case;
  * a field's JSON type MUST match the Rust type it decodes into;
  * the CONTAINER key must exist -- this is the rename case, and it is the
    one a `serde(default)` consumer cannot tell from "genuinely empty".

KNOWN_BREAKS records divergences that already shipped. They are listed
rather than asserted so this file fails on anything NEW while keeping the
existing debt visible and attributable. Removing an entry is how the fix
gets proven; adding one requires the scheduled-break note in the PR.

The register carries two kinds of entry. A `(op, "container")` entry is shipped
debt. A `(op, "field:<old>-><new>")` entry is a SCHEDULED rename: the bridge is
moving the key on purpose and the consumer update lands in the same window, so
the old name is expected to be GONE -- but the check moves to the new name
rather than skipping the field. Presence is demanded on the PRODUCER's fact
(`always_emitted`), not the consumer's (`required`): `name`/`type`/`width` are
always sent and always defaulted, so a half-done rename of one of those is a
permanent silent blank, not a tolerated absence.
"""
from __future__ import annotations

from typing import NamedTuple

import pytest

from _bridge_fakes import (_FakeBV, _FakeCFGBlock, _FakeCFGEdge, _FakeCFGLine,
                           _FakeFunction, _load_bridge)

# --- the consumer's decoder, mirrored -------------------------------------
# A field carries THREE independent facts, and conflating the first two is a
# hole this file already had: `required` is a fact about the CONSUMER (its
# struct has no serde default, so an absent field aborts the element), while
# `always_emitted` is a fact about the PRODUCER (this bridge emits the key on
# every row, so its absence means something changed here).
#
# They differ exactly for a field the bridge always sends and the consumer
# defaults -- name, type, width. Checking only `required` made a HALF-DONE
# rename invisible: drop the old key without adding the new one and the
# consumer renders a blank forever with no error, which is the silent-blank
# class this file exists to catch. The guard was catching it only for the
# fields a consumer would have noticed anyway.
class _Field(NamedTuple):
    types: tuple[type, ...]
    required: bool          # consumer has no serde default -> absence aborts
    nullable: bool          # consumer type is Option<...> -> null is legal
    always_emitted: bool    # this bridge emits it on every row


def _f(types, required, nullable, always_emitted):
    return _Field(types, required, nullable, always_emitted)


_CFG_INSN = {                      # bn.rs:501-506  struct CfgInsn
    "a": _f((str,), True, False, True),
    "t": _f((str,), True, False, True),
}
_CFG_EDGE = {                      # bn.rs:509-514  struct CfgEdge
    "to": _f((str,), True, False, True),   # `pub to: String` -- a null blanks the view
    "k": _f((str,), True, False, True),
}
_CFG_BLOCK = {                     # bn.rs:491-498  struct CfgBlock
    "start": _f((str,), True, False, True),
    "insns": _f((list,), False, False, True),
    "edges": _f((list,), False, False, True),
}
_DATA_VAR = {                      # bn.rs:535-558  struct DataVar
    # Always emitted (built in the row literal) even though the consumer
    # defaults all but the address: read_misc._data_var_row.
    "a": _f((str,), True, False, True),    # `pub addr: String`, no default
    "n": _f((str,), False, False, True),
    "t": _f((str,), False, False, True),
    "w": _f((int,), False, False, True),
    # Conditional: a decoded scalar, a pointer target and its decorations, and
    # the section only appear when the slot warrants them -- so an absence
    # here is ordinary and must stay tolerated even under a scheduled rename.
    "v": _f((int,), False, True, False),
    "p": _f((str,), False, True, False),
    "ps": _f((str,), False, True, False),
    "pstr": _f((str,), False, True, False),
    "sec": _f((str,), False, False, False),
}
_DATA_SYM = {                      # bn.rs:574-580  struct DataSym
    "a": _f((str,), False, False, True),
    "n": _f((str,), False, False, True),
}

# Container key the consumer reads, per op.
_CONTAINERS = {
    "cfg": "blocks",               # bn.rs:517-520  struct CfgJson
    "data_vars": "vars",           # bn.rs:561-565  struct DataMapJson
    "data_symbols": "syms",        # bn.rs:582-586  struct DataSymsJson
}

# Divergences that ALREADY shipped. Listed, not asserted, so new ones fail.
KNOWN_BREAKS = {
    ("data_vars", "container"): (
        "#275 renamed the container `vars` -> `items`; the consumer still "
        "reads `vars` with serde(default), so it decodes as an EMPTY list "
        "rather than an error. Tracked in bn-lens#45 / bn#892."),
    ("data_symbols", "container"): (
        "#275 renamed the container `syms` -> `items`; same silent-empty "
        "shape as data_vars. Tracked in bn-lens#45 / bn#892."),

    # #682 item 2, SCHEDULED break (not shipped debt): the bridge spells the
    # terse keys out because models read this JSON without the reference open.
    # The consumer's structs still spell the old names, so until it moves, `cfg`
    # loses a required field per row and `data vars` rows lose their address --
    # which for `a` aborts the element (and with it the whole Vec). The
    # consumer must be updated in the SAME window; the patch-ready mapping is
    # on the tracking issue. Tracked in bn#892 / bn-lens#45.
    ("cfg", "field:a->address"): (
        "#682 item 2 renames the insn address key `a` -> `address`. Tracked in "
        "bn#892 / bn-lens#45."),
    ("cfg", "field:t->text"): (
        "#682 item 2 renames the insn text key `t` -> `text`. Tracked in bn#892 "
        "/ bn-lens#45."),
    ("cfg", "field:k->branch_type"): (
        "#682 item 2 renames the edge kind key `k` -> `branch_type` (not `kind`, "
        "which is already the envelope discriminator on this payload). Tracked "
        "in bn#892 / bn-lens#45."),
    ("data_vars", "field:a->address"): (
        "#682 item 2 renames the row address key `a` -> `address`. Tracked in "
        "bn#892 / bn-lens#45."),
    ("data_vars", "field:n->name"): (
        "#682 item 2 renames the row symbol key `n` -> `name`; the consumer "
        "defaults it, so the rename fails SILENTLY there. Tracked in bn#892 / "
        "bn-lens#45."),
    ("data_vars", "field:t->type"): (
        "#682 item 2 renames the row type key `t` -> `type`; silent in the "
        "consumer. Tracked in bn#892 / bn-lens#45."),
    ("data_vars", "field:w->width"): (
        "#682 item 2 renames the row width key `w` -> `width`; silent in the "
        "consumer. Tracked in bn#892 / bn-lens#45."),
    ("data_vars", "field:v->value"): (
        "#682 item 2 renames the decoded scalar key `v` -> `value`; conditional "
        "on the slot, silent in the consumer. Tracked in bn#892 / bn-lens#45."),
    ("data_vars", "field:p->pointer"): (
        "#682 item 2 renames the pointer target key `p` -> `pointer`; "
        "conditional on the slot, silent in the consumer. Tracked in bn#892 / "
        "bn-lens#45."),
    ("data_vars", "field:ps->pointer_symbol"): (
        "#682 item 2 renames the pointer-symbol key `ps` -> `pointer_symbol`; "
        "`ps` was opaque without the reference open. Conditional on the slot, "
        "silent in the consumer. Tracked in bn#892 / bn-lens#45."),
    ("data_vars", "field:pstr->pointer_string"): (
        "#682 item 2 renames the pointer-string preview key `pstr` -> "
        "`pointer_string`; conditional on the slot. Tracked in bn#892 / "
        "bn-lens#45."),
    ("data_vars", "field:sec->section"): (
        "#682 item 2 renames the section key `sec` -> `section`; conditional on "
        "the slot, silent in the consumer. Tracked in bn#892 / bn-lens#45."),
}


def _scheduled_rename(op: str, field: str) -> str | None:
    """The name *field* is scheduled to move to under *op*, or None.

    Registered as `(op, "field:<old>-><new>")` in KNOWN_BREAKS, so the register
    stays the single source of truth and the two cannot drift."""
    prefix = f"field:{field}->"
    for key in KNOWN_BREAKS:
        if len(key) == 2 and key[0] == op and key[1].startswith(prefix):
            return key[1][len(prefix):]
    return None


def _check_row(row, model, where, problems, op):
    assert isinstance(row, dict), f"{where}: row is {type(row).__name__}, not an object"
    for field, spec in model.items():
        types, required, nullable = spec.types, spec.required, spec.nullable
        if field not in row:
            replacement = _scheduled_rename(op, field)
            if replacement is not None:
                # A SCHEDULED rename (#682 item 2): the old name is expected to
                # be gone, and the check MOVES to the new name rather than
                # skipping the field. Presence is demanded on the PRODUCER's
                # fact -- `always_emitted` -- not the consumer's `required`:
                # `name`/`type`/`width` are always emitted and consumer-
                # defaulted, so demanding only what the CONSUMER would crash on
                # would let a half-done rename of one of those through, which is
                # the same hole one layer in. `required` is still honoured when
                # set, because a rename is no excuse for an absent field the
                # consumer cannot default.
                names = "/".join(t.__name__ for t in types)
                if replacement not in row:
                    if spec.always_emitted or required:
                        consequence = (
                            "the consumer has no serde default for it, so the "
                            "element (and, inside a Vec, the whole list) fails "
                            "to decode" if required else
                            "the consumer defaults it, so nothing errors and "
                            "the value silently becomes blank in every view")
                        problems.append(
                            f"{where}: field {field!r} is absent AND its "
                            f"scheduled replacement {replacement!r} (#682 item "
                            f"2) is missing -- {consequence}")
                    continue
                if isinstance(row[replacement], bool) and int not in types:
                    problems.append(f"{where}: field {replacement!r} is a bool")
                elif not isinstance(row[replacement], types):
                    problems.append(
                        f"{where}: field {replacement!r} is "
                        f"{type(row[replacement]).__name__}, but the consumer "
                        f"decodes the renamed {field!r} as {names}")
                continue
            if required:
                problems.append(
                    f"{where}: required field {field!r} absent -- the consumer "
                    f"has no serde default for it, so the element (and, inside "
                    f"a Vec, the whole list) fails to decode")
            elif spec.always_emitted:
                # The PRODUCER side of the question. The consumer defaults
                # this field, so its absence is not a decode error -- it is a
                # permanent silent blank, which is worse: no error anywhere
                # and a view that quietly stops carrying the value. Flagging
                # it here is what makes a HALF-DONE rename (old key removed,
                # new key never added) visible instead of green.
                problems.append(
                    f"{where}: field {field!r} is absent, but this bridge "
                    f"emits it on every row -- the consumer defaults it, so "
                    f"nothing errors and the value silently becomes blank. "
                    f"If this is a rename, register it in KNOWN_BREAKS as "
                    f"`field:{field}-><new>` so the replacement is required")
            continue
        value = row[field]
        if value is None:
            if not nullable:
                problems.append(
                    f"{where}: field {field!r} is null, but the consumer "
                    f"decodes it into a non-Option type -- a null fails the "
                    f"decode and blanks the entire view")
            continue
        # bool is an int subclass in Python but not interchangeable on the wire.
        if isinstance(value, bool) and int not in types:
            problems.append(f"{where}: field {field!r} is a bool")
        elif not isinstance(value, types):
            names = "/".join(t.__name__ for t in types)
            problems.append(
                f"{where}: field {field!r} is {type(value).__name__}, "
                f"but the consumer decodes it as {names}")


def _check_container(result, op, problems):
    key = _CONTAINERS[op]
    if key not in result:
        if (op, "container") in KNOWN_BREAKS:
            return None
        problems.append(
            f"{op}: the consumer reads container {key!r}, which this payload "
            f"does not carry. Because that field is serde(default), the "
            f"consumer sees an EMPTY list rather than an error -- the failure "
            f"mode is a confidently blank view, not a diagnostic")
        return None
    return result[key]


def _cfg_result(monkeypatch, *, undetermined=False, null_edge=False):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fn = _FakeFunction(0x401000, "dispatch", "void dispatch(void)")
    tail = _FakeCFGBlock(0x401010, lines=[_FakeCFGLine(0x401010, "ret")])
    edges = [_FakeCFGEdge(tail, "TrueBranch")]
    if null_edge:
        edges.append(_FakeCFGEdge(None, "IndirectBranch"))
    head = _FakeCFGBlock(0x401000,
                         lines=[_FakeCFGLine(0x401000, "jmp rax")],
                         edges=edges, undetermined=undetermined)
    fn.basic_blocks = [head, tail]
    bv = _FakeBV(functions=[fn])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    return instance._cfg(None, "dispatch", view="asm")


def test_cfg_payload_decodes_under_the_consumers_strict_schema(monkeypatch):
    problems: list[str] = []
    result = _cfg_result(monkeypatch)
    blocks = _check_container(result, "cfg", problems)
    for i, block in enumerate(blocks or []):
        _check_row(block, _CFG_BLOCK, f"cfg.blocks[{i}]", problems, "cfg")
        for j, insn in enumerate(block.get("insns", [])):
            _check_row(insn, _CFG_INSN, f"cfg.blocks[{i}].insns[{j}]", problems, "cfg")
        for j, edge in enumerate(block.get("edges", [])):
            _check_row(edge, _CFG_EDGE, f"cfg.blocks[{i}].edges[{j}]", problems, "cfg")
    assert not problems, "\n".join(problems)


def test_cfg_unresolved_target_never_reaches_the_wire_as_a_null(monkeypatch):
    """THE regression this file was written for.

    An unresolved edge target must never be emitted as an edge ROW with
    `to: null`: the consumer's field is a bare `String`, and the failure is
    inside a `Vec`, so one null blanks the whole CFG. The condition is
    reported on the BLOCK instead, where an unknown key is ignored.
    """
    problems: list[str] = []
    result = _cfg_result(monkeypatch, null_edge=True)
    for i, block in enumerate(result["blocks"]):
        for j, edge in enumerate(block["edges"]):
            _check_row(edge, _CFG_EDGE, f"cfg.blocks[{i}].edges[{j}]", problems, "cfg")
    assert not problems, "\n".join(problems)
    # The information is not lost -- it moved somewhere additive.
    assert result["blocks"][0]["undetermined_edges"] is True


def test_cfg_block_extra_keys_are_additive_only(monkeypatch):
    """The consumer sets no `deny_unknown_fields`, so a NEW block key is
    ignored safely -- which is exactly why the unresolved disclosure belongs
    on the block. This pins the reasoning: if a required key were ever
    REPLACED rather than added, the strict check above would fire."""
    result = _cfg_result(monkeypatch, undetermined=True)
    block = result["blocks"][0]
    assert {"start", "insns", "edges"} <= set(block)
    assert block["undetermined_edges"] is True


def test_known_wire_breaks_are_listed_with_a_tracking_issue():
    """A divergence that already shipped stays VISIBLE rather than silently
    tolerated: every entry names what broke and where it is tracked, so the
    list reads as debt with an owner instead of an allowlist."""
    assert KNOWN_BREAKS, "if the breaks are fixed, delete the mechanism too"
    for key, reason in KNOWN_BREAKS.items():
        assert "#" in reason, f"{key} has no tracking issue"
        assert len(reason) > 40, f"{key} is not explained"


def test_data_vars_rows_decode_under_the_consumers_strict_schema(monkeypatch):
    # The row MODEL is only worth having if a real payload is driven through
    # it. `a` is the sharp one: the consumer declares `pub addr: String` with
    # no default, so a row missing it aborts the element -- and inside a Vec,
    # the whole list.
    from test_read_misc import _data_window_bv
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _data_window_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._data_vars(None, start="0x2000", end="0x3000")

    problems: list[str] = []
    rows = _check_container(result, "data_vars", problems)
    if rows is None:                       # known container rename
        rows = result["items"]
    for i, row in enumerate(rows):
        _check_row(row, _DATA_VAR, f"data_vars[{i}]", problems, "data_vars")
    assert not problems, "\n".join(problems)


def test_data_symbols_rows_decode_under_the_consumers_strict_schema(monkeypatch):
    from test_read_misc import _data_window_bv
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _data_window_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._data_symbols(None)

    problems: list[str] = []
    rows = _check_container(result, "data_symbols", problems)
    if rows is None:
        rows = result["items"]
    for i, row in enumerate(rows):
        _check_row(row, _DATA_SYM, f"data_symbols[{i}]", problems, "data_symbols")
    assert not problems, "\n".join(problems)


def test_the_container_check_fires_when_a_break_is_not_yet_known(monkeypatch):
    # Non-vacuity guard for KNOWN_BREAKS itself. The container rename is the
    # defect that shipped silently; with its entry removed the check MUST
    # fire, otherwise the allowlist is load-bearing in the wrong direction --
    # tolerating everything rather than exactly the listed debt.
    from test_read_misc import _data_window_bv
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _data_window_bv()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    result = instance._data_vars(None, start="0x2000", end="0x3000")

    problems: list[str] = []
    monkeypatch.delitem(KNOWN_BREAKS, ("data_vars", "container"))
    _check_container(result, "data_vars", problems)

    assert problems and "vars" in problems[0]
    assert "EMPTY list" in problems[0]


@pytest.mark.parametrize("op", sorted(_CONTAINERS))
def test_every_modelled_op_names_its_consumer_container(op):
    """The clerk rule for wire contracts: an op with a known external
    consumer must name the container that consumer reads. This is the axis
    neither pane checked all day -- we verified against BN and against each
    other, never against a program that decodes our JSON."""
    assert _CONTAINERS[op]


def test_a_half_done_rename_on_a_defaulted_field_is_not_tolerated():
    """The hole this file shipped with, found by sabotage rather than review.

    `required` answered "does the CONSUMER have a serde default"; for a
    rename the load-bearing question is "does the PRODUCER still emit this".
    Those differ for `n`/`t`/`w` -- always sent, always defaulted -- so
    dropping the old key WITHOUT adding the new one passed every check while
    the consumer rendered a blank forever with no error.

    Driven through `_check_row` directly rather than through an op, because
    the point is the RULE: a row that omits an always-emitted field must be
    a problem even when the consumer would not have crashed on it.
    """
    problems: list[str] = []
    row = {"a": "0x2000", "t": "int32_t", "w": 4}      # `n` dropped, nothing added
    _check_row(row, _DATA_VAR, "data_vars[0]", problems, "data_vars")

    assert problems, "a half-done rename on a defaulted field went unnoticed"
    assert "'n' is absent" in problems[0]
    assert "silently becomes blank" in problems[0]


def test_a_conditional_field_may_be_absent_without_complaint():
    """Must-not-fire twin, and the reason `always_emitted` is per-field rather
    than a blanket rule: a scalar's decoded `v`, a pointer's `p`/`ps`/`pstr`
    and the containing `sec` are emitted only when the slot warrants them.
    Flagging those would make every ordinary row a problem and the check
    would be discarded as noise -- which is how a guard dies."""
    problems: list[str] = []
    row = {"a": "0x2000", "n": "g_count", "t": "int32_t", "w": 4}
    _check_row(row, _DATA_VAR, "data_vars[0]", problems, "data_vars")

    assert problems == []


def test_a_scheduled_rename_never_tolerates_an_absence():
    """Non-vacuity guard for the FIELD half of KNOWN_BREAKS.

    A rename entry must tolerate the old name being GONE, not the field being
    gone: `_check_row` moves the requirement to the new name, on the
    `always_emitted` fact. Without this the field entries would degrade into an
    allowlist that accepts a payload carrying neither name -- the "confidently
    blank consumer" failure the container guard exists to prevent, one level
    down."""
    # The scheduled rename, satisfied: the new name is present and typed.
    problems: list[str] = []
    _check_row({"address": "0x1000", "text": "nop"}, _CFG_INSN,
               "cfg.blocks[0].insns[0]", problems, "cfg")
    assert not problems, problems

    # ...and with the new name MISSING, the old absence is not tolerated.
    problems = []
    _check_row({"address": "0x1000"}, _CFG_INSN,
               "cfg.blocks[0].insns[0]", problems, "cfg")
    assert problems, ("a row with neither the old nor the new `text` key "
                      "passed, so the tolerate-the-rename rule tolerates "
                      "everything")
    assert "'text'" in problems[0] and "missing" in problems[0]

    # A field no scheduled entry covers is still an ordinary break, not a rename.
    problems = []
    _check_row({}, _DATA_VAR, "data_vars[0]", problems, "data_symbols")
    assert problems and "'a'" in problems[0]

    # Must-not-fire twin for the CONDITIONAL fields: their rename entry must NOT
    # demand the replacement, or every ordinary row (no pointer, no decoded
    # scalar, no section) would be a problem and the check would be noise. This
    # is the same per-field reasoning `always_emitted` carries.
    problems = []
    _check_row({"address": "0x2000", "name": "g_count", "type": "int32_t",
                "width": 4}, _DATA_VAR, "data_vars[0]", problems, "data_vars")
    assert not problems, problems
