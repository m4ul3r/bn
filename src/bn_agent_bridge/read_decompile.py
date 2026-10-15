"""Decompile / IL / disasm / prototype / locals / dataflow read handlers.

The decompile/IL/disasm read-op cluster that used to live on
``BinaryNinjaBridge`` moves here as module-level free functions, each taking the
``BridgeContext`` seam (``ctx``) in place of ``self``. ``BinaryNinjaBridge``
keeps a thin delegating shim for every name the test suite / op binders
reference (``_decompile``, ``_function_info``, ``_get_prototype``,
``_list_locals_for_function``, ``_list_frame_for_function``, ``_il``, ``_disasm``, ``_structured_il``,
``_defuse``, ``_resolved_calls``, ``_possible_values``, ``_pvs_targets``,
``_force_function_analysis``).

Outbound calls resolve through:
  * ``ctx`` -- resolution helpers relocated to the seam (``_resolve_view``,
    ``_find_function``, ``_functions_containing``);
  * ``il_format`` -- the pure IL/HLIL/disasm renderers and SSA/value-set
    helpers (``_decompile_text``, ``_render_warnings``, ``_analysis_stub_warning``,
    ``_comment_map``, ``_function_metadata``, ``_function_text``, ``_disasm_text``,
    ``_il_function_for``, ``_il_op_name``, ``_ssa_var_entry``,
    ``_resolve_ssa_variable``, ``_serialize_pvs``, ``_pvs_determined``);
  * ``vars`` -- variable discovery (``_list_locals``, ``_find_variable_selector``);
  * ``taint_engine`` -- call-target / function resolution used by the dataflow
    ops (imported as the SAME ``from . import taint_engine as _taint`` alias the
    bridge uses);
  * ``_shared`` -- module-free helpers (``_parse_address``, ``OperationFailure``).

Import direction is one-way: this module imports ``il_format``, ``vars``,
``taint_engine``, ``read_types``, and ``_shared`` (plus stdlib + binaryninja).
It NEVER imports ``bridge`` or ``seam`` -- those import THIS module one-way.
"""
from __future__ import annotations

import re
import shlex

from typing import Any

try:
    import binaryninja as bn  # bn.Architecture[...] for the #382 --mode arch lookup
except ModuleNotFoundError:  # importable without the Binary Ninja runtime (tests, tooling)
    bn = None  # type: ignore[assignment]

from . import il_format
from . import read_types
from . import read_xrefs
from . import taint_engine as _taint
from . import vars as vars_mod
from ._shared import OperationFailure, _parse_address  # noqa: F401
from .bridge_state import require_analysis
from .read_listing import _analysis_state_fields


def _warning_row(code: str, summary: str, detail: str, **extra: Any) -> dict[str, Any]:
    """One decompile warning (#1000 item 1): a stable *code* to filter on, a
    one-line *summary* text mode prints, and the full *detail* (what it means
    and what to do) `--explain` prints and JSON always carries. *extra* holds
    the structured facts the summary was cut from (addresses, per-call rows).

    Text used to print *detail* alone, and agents filtered every `warning:` line
    out wholesale -- dropping the ones that mattered with the boilerplate."""
    if "address" not in extra:
        addresses = extra.get("addresses") or []
        calls = extra.get("calls") or []
        if addresses:
            extra["address"] = addresses[0]
        elif calls:
            extra["address"] = calls[0]["address"]
    return {"code": code, "summary": summary, "detail": detail, **extra}


def _stub_warning_row(func, detail: str, forced: bool) -> dict[str, Any]:
    """Wrap `il_format._analysis_stub_warning`'s sentence (shared with `bundle`)."""
    try:
        raw = func.analysis_skip_reason
        reason = getattr(raw, "name", None) or str(raw)
    except Exception:  # noqa: BLE001
        reason = None
    because = f" ({reason})" if reason else ""
    summary = (f"still an incomplete stub after --force-analysis{because}" if forced else
               f"incomplete stub: BN skipped analysis{because}; re-run with --force-analysis")
    return _warning_row("stub", summary, detail)


def _force_function_analysis(ctx, bv, func):
    """Override a skipped function's analysis and reanalyze it in place.

    Mutates the BinaryView (skip override + reanalysis), so callers must hold
    the exclusive write lock. Returns the (possibly rebuilt) function object.
    """
    start = int(func.start)
    try:
        func.analysis_skipped = False  # setter installs NeverSkipFunctionAnalysis
    except Exception:
        pass
    try:
        func.reanalyze()
    except Exception:
        pass
    bv.update_analysis_and_wait()
    return bv.get_function_at(start) or func


# Forcing analysis on an oversized, uncalled region is the data-as-code trap:
# BN tentatively made a function on a string/pointer table or packed data, and
# the forced decode runs away. Only nudge for genuinely large regions so a
# routine forced reanalysis of a normal function stays quiet.
_FORCED_DATA_MIN_BYTES = 0x4000


def _forced_data_region_warning(bv, func) -> dict[str, Any] | None:
    """A hedged verify-nudge when a force-analyzed function looks like it may be
    a DATA region BN tentatively typed as code (#371.1).

    The reliable signal common to the observed cases (a pointer/tag table AND
    high-entropy packed data) is structural, not content-based: a large region
    with zero inbound code refs AND zero inbound data refs. Byte/string
    heuristics miss the high-entropy case, so this stays a *verify* nudge, never
    a "this is data" verdict. Real oversized code almost always has inbound
    callers; requiring zero DATA refs too suppresses the false positive on a
    large `.init_array`/ctor-dispatched initializer (real code with no direct
    callers, but pointed at by the init table -- i.e. a data ref to its start).
    """
    try:
        size = int(getattr(func, "total_bytes", 0) or 0)
    except Exception:
        size = 0
    if size < _FORCED_DATA_MIN_BYTES:
        return None
    if read_xrefs._code_ref_count(bv, func.start) != 0:
        return None
    try:
        if len(list(bv.get_data_refs(func.start))) != 0:
            return None
    except Exception:
        return None
    return _warning_row(
        "forced_data",
        f"forced decode of {size} bytes with no inbound refs: may be a data region; verify",
        f"{func.name}: forced analysis decoded {size} bytes with 0 inbound code refs. "
        "BN sometimes makes a function on a DATA region (string/pointer table or packed "
        "data) that grows under forced decode; if the body reads as tables/strings/garbage "
        "rather than coherent code, treat this decode as suspect and verify before acting on it.",
    )


def _annotate_containment(ctx, result, identifier, func) -> None:
    """Tag a function-scoped READ result when the request resolved to a
    containing function rather than an exact start (#193 Part 4).

    No-op for an exact start or a name; otherwise adds ``resolved_from`` so an
    agent feeding a taint/trace sink address knows the hit was mid-function.
    """
    meta = ctx._containment_meta(identifier, func)
    if meta:
        result["resolved_from"] = meta


def _thunk_veneer_warning(ctx, bv, func) -> dict[str, Any] | None:
    """A `thunk/veneer -> <target>` warning when *func* is a PLT/GOT trampoline
    (#446), reusing `evidence function`'s detector so the signal is consistent.
    None when it is not a thunk."""
    from . import read_evidence  # lazy: read_evidence -> il_format, not this module
    summary = read_evidence._function_thunk_summary(ctx, bv, func)
    if not summary.get("is_candidate"):
        return None
    target = summary.get("target") if isinstance(summary.get("target"), dict) else None
    fn = target.get("function") if isinstance(target, dict) else None
    if isinstance(fn, dict) and (fn.get("name") or fn.get("address")):
        name, addr = fn.get("name"), fn.get("address")
        dest = f"{name} @ {addr}" if name and addr else (name or addr)
        if fn.get("exact_start") is False:
            return _warning_row(
                "thunk", f"thunk/veneer: branch lands inside {dest}, not at its entry; not confirmed",
                f"thunk/veneer: the branch target lands inside {dest}, not at its "
                f"entry -- a mid-function destination is not a trampoline entry "
                f"point, so this is not confirmed as a PLT/GOT thunk.")
        return _warning_row(
            "thunk", f"thunk/veneer -> {dest}",
            f"thunk/veneer -> {dest}: this is a PLT/GOT trampoline (a jump to "
            f"the real body), not a self-recursive function.")
    # #704 round 3: no target was resolved here -- the candidate was flagged
    # either because the function merely starts in a PLT/import section, or
    # because BN's pseudo-C rendered a bare `/* tailcall */` with no branch
    # destination this detector could follow. Asserting "PLT/GOT trampoline
    # ... the apparent self-call is the resolved target" claimed a resolution
    # the tool never made. State only the reason the candidate was flagged;
    # make no claim about what the unresolved target actually is.
    reason = summary.get("reason") or "no branch target could be resolved"
    return _warning_row(
        "thunk", "thunk/veneer candidate; branch target not resolved",
        f"thunk/veneer: {reason}; the branch target was not resolved, so "
        f"this may be a trampoline/veneer but that is not confirmed.")


_SESSION_ANNOTATION_MARK = "[this session] "


def _body_lines(bodies) -> set[str]:
    return {
        line.strip()
        for body in bodies
        for line in str(body).splitlines()
        if line.strip()
    }


def _redact_rendered_annotations(
    text: str, annotation_bodies: list[str], session_bodies: list[str] = (),
) -> tuple[str, int, int]:
    """Redact matching standalone or trailing ``//`` annotation bodies.

    Match complete stored body lines, not substrings in code or literals.
    Block comments are consumed unchanged; BN renders inherited notes as ``//``.

    A line of *session_bodies* -- a comment this bridge wrote -- stays visible,
    marked ``[this session]`` (#977). A line stored BOTH ways is redacted:
    the rendered text cannot say which store it came from, so the inherited
    reading wins. Returns ``(text, hidden, session_shown)``, counting the
    rendered comment spans each rule rewrote, so a caller can tell the reader
    something was hidden rather than leave ``<annotation redacted>`` looking
    like a write that did not land.
    """
    bodies = _body_lines(annotation_bodies)
    session = _body_lines(session_bodies) - bodies
    if not bodies and not session:
        return text, 0, 0
    counts = {"hidden": 0, "session": 0}

    def redact(match: re.Match[str]) -> str:
        body = match.group("body")
        if body is None:
            return match.group(0)
        stripped = body.strip()
        leading = body[:len(body) - len(body.lstrip())]
        trailing = body[len(body.rstrip()):]
        if stripped in bodies:
            counts["hidden"] += 1
            return f"//{leading}<annotation redacted>{trailing}"
        if stripped in session:
            counts["session"] += 1
            return f"//{leading}{_SESSION_ANNOTATION_MARK}{stripped}{trailing}"
        return match.group(0)

    redacted = il_format._COMMENT_SPAN_RE.sub(redact, text)
    return redacted, counts["hidden"], counts["session"]


def _annotation_entries(func, comments: dict) -> list[tuple[tuple[str, int], str]]:
    """Every stored annotation that can surface in this function's rendered
    text, keyed by where it is stored: ``("address", addr)`` for the global
    comment map (passed in, already computed by the caller), ``("local", addr)``
    for the function-local map, and ``("function_doc", start)`` for the
    function's own documentation comment. The keys are the ones
    ``TargetManager.note_session_annotations`` records, so a session-written
    comment is recognised by location AND body (#977).

    Shared with ``bridge._bundle_function``, which renders the same text through
    its own call path: #752 was precisely those two paths disagreeing about what
    counts as an annotation, so the rule lives in one place.
    """
    function_comment = str(getattr(func, "comment", "") or "")
    # #861: cheap insurance, NOT a reproduced hazard. Against BN 6.1
    # `Function.comments` builds and returns a FRESH local dict on every access
    # (so does `BinaryView.address_comments`), which means a mid-walk
    # `RuntimeError: dictionary changed size during iteration` cannot arise here,
    # and the #850 symptom this comment used to lean on was diagnosed on a
    # settling quick view rather than bisected to an accessor. What the snapshot
    # costs is one O(n) copy of an already-private dict; what it defends against
    # is a BN build that ever hands back the live store -- modelled by a
    # regression test whose fake `comments` adds an entry mid-walk. Snapshot the
    # WHOLE collection rather than its `.values()` view: if the store ever is
    # live, that view iterates it and a parallel text decompile dies of
    # "dictionary changed size during iteration".
    local_comments = dict(getattr(func, "comments", {}) or {})
    # Local and global bodies can differ at the same address; do not merge
    # their maps and silently discard one of the two annotations.
    entries = [(("address", int(str(addr), 16)), body) for addr, body in comments.items()]
    entries.extend((("local", int(addr)), body) for addr, body in local_comments.items())
    if function_comment:
        entries.append((("function_doc", int(func.start)), function_comment))
    return entries


def _redact_function_annotations(ctx, bv, func, comments: dict, text: str
                                 ) -> tuple[str, dict[str, Any]]:
    """The default (#740) decompile redaction for *func*, exempting the
    comments this bridge wrote (#977). Returns the text and the
    ``annotation_summary`` counts. Shared by ``decompile`` and ``bundle
    function`` so the two renderings cannot disagree (#752)."""
    entries = _annotation_entries(func, comments)
    try:
        ledger = ctx.targets.session_annotations(bv)
    except Exception:  # noqa: BLE001 - no ledger means nothing is exempt
        ledger = {}
    inherited = [body for key, body in entries if ledger.get(key) != body]
    session = [body for key, body in entries if ledger.get(key) == body]
    text, hidden, shown = _redact_rendered_annotations(text, inherited, session)
    return text, {
        "comment_count": len(entries),
        "redacted": True,
        "hidden": hidden,
        "session_shown": shown,
    }



# Mnemonics whose value is chosen WITHOUT a branch, which HLIL renders as an
# ordinary ternary or folds into a compound condition (#676 item 12). The skill
# documents these as known lies and tells the reader to confirm in `bn disasm`;
# a note in the output reaches the reader who did not read the skill, which is
# every reader at the moment they are wrong.
#
# Keyed on the MNEMONIC, not the architecture, and that is the whole reason
# this is cheap where the return-width note (#675 item 5) needed a measured
# arch set: the instruction IS the evidence. `csel` in a listing means a
# selected value in THIS function, whatever the target is, so there is no
# per-arch fact to inherit wrongly.
_FLATTENED_GUARD_MNEMONICS = {
    # AArch64: conditional compare (chains a second test into the flags, which
    # HLIL merges into one `&&`/`||`) and conditional select.
    "ccmp", "ccmn", "csel", "csinc", "csinv", "csneg", "cset", "csetm",
    # x86: the same lie with a different name.
    "cmova", "cmovae", "cmovb", "cmovbe", "cmovc", "cmove", "cmovg", "cmovge",
    "cmovl", "cmovle", "cmovna", "cmovnae", "cmovnb", "cmovnbe", "cmovnc",
    "cmovne", "cmovng", "cmovnge", "cmovnl", "cmovnle", "cmovno", "cmovnp",
    "cmovns", "cmovnz", "cmovo", "cmovp", "cmovpe", "cmovpo", "cmovs", "cmovz",
    # ARM32/Thumb: an IT block predicates the instructions that follow it, and
    # BN's per-instruction mnemonics may not repeat the IT-derived condition.
    "it", "ite", "itt", "itee", "itet", "itte", "ittt",
    "iteee", "iteet", "itete", "itett", "ittee", "ittet", "ittte", "itttt",
}


def _flattened_guard_warning(bv, func) -> dict[str, Any] | None:
    """Name the branch-free selects in *func*, or say nothing (#676 item 12).

    The harm is specific and has faked critical findings: HLIL flattens a
    `ccmp`/`csel` range guard into a ternary, so a bound that IS enforced can
    read as one that never fires (or the reverse). The reader cannot see that
    from the Pseudo-C -- by construction, since the flattening is what removed
    the evidence -- so the disclosure has to come from the listing underneath.

    This states ONLY what was observed: the mnemonics actually present, by
    name. The skill's other two documented lies -- a hoisted loop-invariant
    bound aliased to a moving pointer, and a dropped `<< 4` in a size
    accumulator -- are SEMANTIC properties with no mnemonic to key on, so
    nothing is claimed about them here. A note that implied this function had
    been checked for those would be worse than no note.
    """
    try:
        arch = getattr(func, "arch", None)
        seen: list[str] = []
        sites: list[int] = []
        for block in list(func.basic_blocks):
            addr = int(block.start)
            end = int(block.end)
            while addr < end:
                text, length = il_format._disasm_instruction(bv, addr, arch=arch)
                if not length or length <= 0:
                    break
                if text:
                    mnemonic = str(text).split(None, 1)[0].strip().lower()
                    if mnemonic in _FLATTENED_GUARD_MNEMONICS:
                        sites.append(addr)
                        if mnemonic not in seen:
                            seen.append(mnemonic)
                addr += length
        if not seen:
            return None
    except Exception:  # noqa: BLE001 - no listing is not evidence of no select
        return None
    names = ", ".join(sorted(seen))
    sites.sort()
    where = ", ".join(hex(a) for a in sites[:3]) + (
        f" +{len(sites) - 3} more" if len(sites) > 3 else "")
    return _warning_row(
        "select",
        f"branch-free {names} at {where}: a guard there may read as a ternary or "
        "flag math; confirm it in `bn disasm`",
        f"branch-free conditional(s) in this function ({names}): Pseudo-C "
        f"renders these as ternaries, folds them into one compound condition, "
        f"or decomposes them into raw flag arithmetic (`v`/`z`/`n` locals) -- "
        f"a two-sided range guard can come out as none of the three shapes a "
        f"reader is looking for, so a bound that IS enforced can read as one "
        f"that never fires. Confirm it in `bn disasm` before concluding on an "
        f"off-by-one, truncation or missing check.",
        addresses=[hex(a) for a in sites],
    )


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value))


def _untyped_pointer_accesses(func: Any) -> list[tuple[int, Any, set[tuple[int, int]]]]:
    """Direct incoming void-pointer accesses only, not alias/type recovery.

    SSA version zero ties the witness to the incoming parameter, not a later
    value stored in the same variable. Any unsupported memory address using
    that variable disqualifies it rather than cherry-picking its easy accesses,
    and so does any other use through which the pointer value can reach code
    the witness does not inspect: a copy, PHI, derived pointer, call argument
    or stored value. Only a comparison, whose result cannot carry the pointer,
    is exempt.
    """
    parameters = {
        _taint.var_key(var): (index, var)
        for index, var in enumerate(getattr(func, "parameter_vars", ()))
        if _enum_name(getattr(var.type, "type_class", None)) == "PointerTypeClass"
        and _enum_name(getattr(var.type.target, "type_class", None)) == "VoidTypeClass"
    }
    if not parameters:
        return []
    ssa = getattr(getattr(func, "mlil", None), "ssa_form", None)
    if ssa is None:
        return []
    accesses: dict[tuple, set[tuple[int, int]]] = {key: set() for key in parameters}
    unsupported: set[tuple] = set()
    # Expression indexes of incoming-parameter reads; each must be an accepted
    # access base or a comparison operand.
    uses: dict[int, tuple] = {}
    accounted: set[int] = set()

    def incoming(node: Any) -> tuple | None:
        if _taint.op_name(node) != "MLIL_VAR_SSA" or node.src.version != 0:
            return None
        key = _taint.var_key(node.src)
        return key if key in parameters else None

    def collect(ins: Any) -> None:
        op = _taint.op_name(ins)
        if op == "MLIL_VAR_SSA":
            key = incoming(ins)
            if key is not None:
                uses[int(ins.expr_index)] = key
            return
        if op == "MLIL_VAR_PHI":
            unsupported.update(
                _taint.var_key(src) for src in ins.src
                if src.version == 0 and _taint.var_key(src) in parameters)
            return
        if op in {
            "MLIL_VAR_SSA_FIELD", "MLIL_VAR_ALIASED", "MLIL_VAR_ALIASED_FIELD",
            "MLIL_ADDRESS_OF", "MLIL_ADDRESS_OF_FIELD",
        }:
            key = _taint.var_key(ins.src)
            if key in parameters:
                unsupported.add(key)
            return
        if op.startswith("MLIL_CMP_"):
            for operand in (ins.left, ins.right):
                if incoming(operand) is not None:
                    accounted.add(int(operand.expr_index))
            return
        if op not in {
            "MLIL_LOAD_SSA", "MLIL_LOAD_STRUCT_SSA",
            "MLIL_STORE_SSA", "MLIL_STORE_STRUCT_SSA",
        }:
            return
        address = ins.src if op.startswith("MLIL_LOAD") else ins.dest
        touched = {_taint.var_key(v) for v in address.vars_read} & parameters.keys()
        if not touched:
            return
        base = address
        offset = int(ins.offset) if "_STRUCT_" in op else 0
        if _taint.op_name(base) == "MLIL_ADD":
            left, right = base.left, base.right
            if _taint.op_name(left) == "MLIL_CONST":
                left, right = right, left
            if _taint.op_name(right) == "MLIL_CONST":
                offset += int(right.constant)
                base = left
        key = incoming(base)
        width = int(ins.size)
        if key is None or touched != {key} or offset < 0 or width <= 0:
            unsupported.update(touched)
            return
        accounted.add(int(base.expr_index))
        accesses[key].add((offset, width))

    # traverse visits nested loads as well as top-level stores. Its callback
    # returns None, so draining the generator need not retain any IL objects.
    for _ in ssa.traverse(collect):
        pass
    unsupported.update(key for index, key in uses.items() if index not in accounted)
    return [
        (index, var, accesses[key])
        for key, (index, var) in parameters.items()
        if key not in unsupported and len({off for off, _ in accesses[key]}) >= 2
    ]


_SCALAR_TYPE_CLASSES = frozenset({
    "BoolTypeClass", "IntegerTypeClass", "FloatTypeClass",
    "EnumerationTypeClass", "PointerTypeClass", "WideCharTypeClass",
})


def _struct_layout_warnings(bv: Any, func: Any) -> list[dict[str, Any]]:
    """Evidence-qualified known-layout assistance; never writes a type (#903).

    Match complete scalar members of concrete, base-less structs, resolving
    typedefs only for members at observed offsets/widths. Unclassifiable members
    -- unresolved typedefs, bitfield storage units, overlapping members and
    members of unknown (zero) width -- can block uniqueness but cannot supply a
    witness. Base-less classes, and layouts whose members overrun their recorded
    width (stale metadata after a member type was redeclared), compete but are
    never suggested. Nested, union, array and inherited layouts are outside the
    comparison, and the warning says so. Whole-struct aliases do not add
    candidates (their concrete declarations are already in bv.types). Matching
    two distinct offsets is a minimum witness, not proof of identity.
    """
    accesses = _untyped_pointer_accesses(func)
    if not accesses:
        return []
    observed_sets = [observed for _, _, observed in accesses]
    observed_keys = set().union(*observed_sets)
    layouts: list[tuple[str, dict[tuple[int, int], str | None]]] = []
    for name, type_obj in bv.types.items():
        if (_enum_name(type_obj.type_class) != "StructureTypeClass"
                or type_obj.base_structures):
            continue
        kind = _enum_name(type_obj.type)
        if kind not in {"StructStructureType", "ClassStructureType"}:
            continue
        witness = kind == "StructStructureType"
        scalars: dict[tuple[int, int], Any] = {}
        uncertain: list[tuple[int, int]] = []  # byte spans [start, end)
        end = 0
        # BN can size a struct ending in a bitfield to the bit, short of the
        # storage unit's last byte; the unit still bounds the layout.
        limit = int(type_obj.width)
        for member in sorted(type_obj.members, key=lambda member: member.offset):
            offset, width = int(member.offset), int(member.type.width)
            if width <= 0:
                # Unresolved type or flexible array: it may occupy anything
                # from here to the struct's end (nothing, if it is trailing).
                uncertain.append((offset, int(type_obj.width)))
                continue
            if offset < end:
                uncertain.append((offset, min(end, offset + width)))
            end = max(end, offset + width)
            if getattr(member, "bit_width", 0):
                # Compilers may access any part of a bitfield's storage unit.
                uncertain.append((offset, offset + width))
                limit = max(limit, offset + width)
            elif member.name:
                scalars[(offset, width)] = member
        if end > limit:
            # Members overrun the recorded width: the metadata disagrees with
            # itself, so this layout can block but never witness.
            witness = False
        # Named scalar = positive; None = uncertain; absent = incompatible.
        members: dict[tuple[int, int], str | None] = {}
        for offset, width in observed_keys:
            if any(offset < hi and lo < offset + width for lo, hi in uncertain):
                members[(offset, width)] = None
            elif (offset, width) in scalars:
                member = scalars[(offset, width)]
                _, member_type, follow_error = read_types._follow_typedef(
                    bv, str(member.name), member.type)
                if follow_error is not None:
                    members[(offset, width)] = None
                elif _enum_name(member_type.type_class) in _SCALAR_TYPE_CLASSES:
                    members[(offset, width)] = str(member.name) if witness else None
        if any(observed <= members.keys() for observed in observed_sets):
            layouts.append((str(name), members))

    warnings = []
    for index, var, observed in accesses:
        matches = [(name, members) for name, members in layouts if observed <= members.keys()]
        if len(matches) != 1:
            continue  # no witness or ambiguous: never choose a "best" struct
        name, members = matches[0]
        if any(members[access] is None for access in observed):
            continue  # the sole possible layout must still have positive witnesses
        evidence = ", ".join(
            f"+{offset:#x}/{width} bytes -> {members[(offset, width)]}"
            for offset, width in sorted(observed)
        )
        qualifiers = " ".join(
            q for q in ("const", "volatile") if bool(getattr(var.type.target, q, False))
        )
        pointer_type = f"{qualifiers + ' ' if qualifiers else ''}struct {name} *"
        address = hex(int(func.start))
        warnings.append(_warning_row(
            "layout",
            f"struct layout candidate: parameter {index + 1} ({var.name}) matches struct "
            f"{name} at {len(observed)} offset(s); no type applied",
            f"struct layout candidate for parameter {index + 1} ({var.name}): "
            f"observed direct MLIL accesses {evidence} match declared struct {name}. "
            "Only flat scalar members of base-less struct/class layouts were compared; "
            "a layout holding these offsets in a nested struct, union, array or base "
            "class was not, so another type may also fit. "
            "Offsets and widths alone do not prove the parameter's type. "
            f"Review `bn struct show {shlex.quote(name)}` and `bn disasm {address}`; "
            f"if confirmed, use `bn proto get {address}`, then `bn proto set {address}` "
            f"with the full prototype, changing only this parameter to "
            f"`{pointer_type}` (preserve other parameters, qualifiers and calling convention). "
            "No type was applied.",
        ))
    return warnings


# Symbol kinds whose body is not in this binary: an import's PLT stub, its GOT
# slot, or an extern. A call to one of these cannot be IPRA-optimized (#955), so
# an argument register written right before it is an argument.
_IMPORT_SYMBOL_KINDS = {
    "ImportedFunctionSymbol", "ImportAddressSymbol", "ExternalSymbol",
    "LibraryFunctionSymbol",
}
_ARITY_LISTED_MAX = 6


def _full_reg(arch, name: str) -> str:
    """The full-width register *name* is part of (`w2` -> `x2`, `esi` -> `rsi`)."""
    try:
        return str(arch.regs[name].full_width_reg)
    except Exception:  # noqa: BLE001 - an unknown register stands for itself
        return str(name)


def _llil_regs_read(arch, expr, acc: set[str]) -> None:
    if expr is None or not hasattr(expr, "operation"):
        return
    if il_format._il_op_name(expr) in ("LLIL_REG", "LLIL_REG_SSA", "LLIL_REG_SSA_PARTIAL"):
        src = getattr(expr, "src", None)
        name = getattr(src, "name", None)
        if name:
            acc.add(_full_reg(arch, str(name)))
    for operand in getattr(expr, "operands", None) or []:
        for item in operand if isinstance(operand, list) else [operand]:
            _llil_regs_read(arch, item, acc)


def _argument_setup_count(arch, window: list, arg_regs: list[str]) -> tuple[int, dict[str, Any]]:
    """How many leading argument registers *window* writes for the call after it.

    *window* is the LLIL between the previous call (or the block start) and the
    call, in order. LLIL blocks do NOT end at calls, so that bound is the
    caller's job, and it bounds everything here: nothing before an earlier call
    can set up this one, nor explain one of its writes (#985 review).

    A write to an argument register counts only when nothing reads that register
    between the write and the call: such a value has no consumer but the callee,
    since every argument register is caller-saved. Two writes are never counted:
    one that is read before the call (a temporary), and GCC's stack-protector
    scrub -- `ldr xN, [guard]; str xN, [sp, ..]; mov xN, #0` -- which zeroes a
    register it just stored, and is where the issue's invented `__stack_chk_guard`
    argument came from (#955). Returns (count, {written register: its LLIL write}),
    in argument order: the count is the highest written argument register's
    position + 1, because the ABI passes every lower register too.
    """
    read_after: set[str] = set()
    written: dict[str, Any] = {}
    stored_before: list[set[str]] = []
    stored: set[str] = set()
    for ins in window:
        stored_before.append(set(stored))
        if il_format._il_op_name(ins) == "LLIL_STORE":
            src = getattr(ins, "src", None)
            if il_format._il_op_name(src) == "LLIL_REG":
                stored.add(_full_reg(arch, str(getattr(src.src, "name", ""))))
    for pos in range(len(window) - 1, -1, -1):
        ins = window[pos]
        if il_format._il_op_name(ins) == "LLIL_SET_REG":
            name = _full_reg(arch, str(getattr(getattr(ins, "dest", None), "name", "")))
            if name in arg_regs and name not in written and name not in read_after:
                scrub = (il_format._llil_constant_value(getattr(ins, "src", None)) == 0
                         and name in stored_before[pos])
                if not scrub:
                    written[name] = ins
            _llil_regs_read(arch, getattr(ins, "src", None), read_after)
        else:
            _llil_regs_read(arch, ins, read_after)
    if not written:
        return 0, {}
    names = sorted(written, key=arg_regs.index)
    return arg_regs.index(names[-1]) + 1, {name: written[name] for name in names}


_CALL_OPS = ("LLIL_CALL", "LLIL_TAILCALL", "LLIL_CALL_STACK_ADJUST")
# A chain of selects (`ccmp`, then `csel`) lifts to one diamond per select.
_SELECT_DIAMOND_DEPTH = 4


def _block_key(block) -> int | None:
    try:
        return int(block.start)
    except Exception:  # noqa: BLE001 - a block without an index cannot be matched
        return None


def _lifted_select_prefix(block, depth: int = 0) -> tuple[list, set[int]]:
    """The set-up that precedes *block* when *block* is the join of a lifted
    branch-free select, and ``([], set())`` otherwise.

    BN lifts one `csel w2, w2, w3, ls` into a diamond: the block before it
    ends in `if (w2 u<= w3)`, and each arm is a one-instruction block that sets
    w2 and jumps to the join. A call right after the `csel` therefore starts
    its own LLIL block, and the window `_calls_with_setup_windows` hands
    `_argument_setup_count` held nothing -- so `ostream::write(this, buf, n)`,
    whose length is that `csel`, rendered two arguments with no warning
    (#1000 item 2). Returns (the head's tail after its last call + every arm,
    the ids of the arm instructions). Only that exact shape is followed: two
    or more arms, every arm lifted from ONE instruction at the head's `if`
    address and entered only from the head."""
    if depth >= _SELECT_DIAMOND_DEPTH:
        return [], set()
    try:
        arms = [edge.source for edge in list(block.incoming_edges)]
        if len(arms) < 2:
            return [], set()
        arm_bodies = [list(arm) for arm in arms]
        sites = {int(ins.address) for body in arm_bodies for ins in body}
        heads = [[edge.source for edge in list(arm.incoming_edges)] for arm in arms]
        if len(sites) != 1 or any(len(h) != 1 for h in heads):
            return [], set()
        head = heads[0][0]
        if any(_block_key(h[0]) != _block_key(head) for h in heads) or _block_key(head) is None:
            return [], set()
        head_body = list(head)
        if (not head_body or il_format._il_op_name(head_body[-1]) != "LLIL_IF"
                or int(head_body[-1].address) not in sites):
            return [], set()
    except Exception:  # noqa: BLE001 - an unreadable CFG is simply not a diamond
        return [], set()
    tail: list = []
    reset = False
    # The `if` itself stays in: a register only its condition reads (`cmp x6,
    # x7` before the `csel`) is consumed there, not handed to the call.
    for ins in head_body:
        op = il_format._il_op_name(ins)
        if "CALL" in op or op == "LLIL_SYSCALL":
            tail, reset = [], True
        else:
            tail.append(ins)
    prefix, arm_ids = ([], set()) if reset else _lifted_select_prefix(head, depth + 1)
    # Every arm, not one: an arm reads its other operand (`w2 = w3`), and a
    # register only one arm reads must not count as written-and-unread.
    arm = [ins for body in arm_bodies for ins in body
           if il_format._il_op_name(ins) != "LLIL_GOTO"]
    return prefix + tail + arm, arm_ids | {id(ins) for ins in arm}


def _calls_with_setup_windows(llil):
    """Yield ``(call, window, select_ids)`` for each LLIL call, *window* being
    the instructions since the previous call or the start of the call's block.
    A block that is the join of a lifted select continues the window back
    through the select (`_lifted_select_prefix`), whose arm instructions are
    *select_ids*. One linear pass, so the cost is the function's LLIL size,
    not its square."""
    for block in list(getattr(llil, "basic_blocks", None) or []):
        window, select_ids = _lifted_select_prefix(block)
        for ins in block:
            op = il_format._il_op_name(ins)
            if op in _CALL_OPS:
                yield ins, window, select_ids
                window, select_ids = [], set()
            elif "CALL" in op or op == "LLIL_SYSCALL":
                window, select_ids = [], set()
            else:
                window.append(ins)


# Sized deallocation (`operator delete(void*, size_t)` and the array / aligned
# forms). BN's type library declares only the pointer, so every sized delete in
# a C++ binary renders one argument short -- 370 of 383 findings on one stock
# binary (#985 review). The dropped value is the object size the allocator is
# told, not data any analysis follows, so reporting it buried the findings that
# matter.
_SIZED_DELETE_RE = re.compile(r"^_Zd[la]Pv[mj]")


def _demangled_arity(bv, raw_name: str) -> tuple[int, int, bool] | None:
    """(total, explicit, variadic) parameter counts from a mangled name, or None.

    BN's GNU3 demangler includes `this` for every nested name -- including a
    static member or a namespaced free function, which have none -- so *total*
    may overcount by one; *explicit* excludes a `this` parameter and never does.
    """
    demangle = getattr(bn, "demangle_gnu3", None)
    if demangle is None or not raw_name.startswith("_Z"):
        return None
    try:
        demangled_type, _ = demangle(bv.arch, raw_name)
        params = list(getattr(demangled_type, "parameters", None) or [])
    except Exception:  # noqa: BLE001 - an undemanglable name gives no signature
        return None
    if demangled_type is None:
        return None
    has_this = bool(params) and str(getattr(params[0], "name", "")) == "this"
    variadic = bool(getattr(demangled_type, "has_variable_arguments", False))
    return len(params), len(params) - int(has_this), variadic


def _call_arity_findings(bv, func) -> list[dict[str, Any]]:
    """One row per call whose rendered argument list is shorter than the call
    site's (#955), or -- for a mangled callee -- longer than its signature.

    HLIL renders exactly the callee prototype's parameter count. When that
    prototype is short -- BN's bundled C++ type libraries declare many library
    methods without `this` and drop trailing parameters -- the extra arguments
    vanish from the output with no trace: `std::filebuf::open(&f)` for a call
    whose disassembly sets up `this`, the path and the open mode.

    Two witnesses, so neither can fire alone where it is known to be wrong:
      * the demangled signature of a mangled callee. It counts `this` for a
        static member too, so a shortfall against it is only reported when the
        explicit parameters alone already exceed the rendered count, or when
        the call site itself writes a register past the rendered arguments;
      * the argument registers the call's block writes and nothing else reads
        (`_argument_setup_count`). Used alone only for a callee with no body
        here -- an import cannot be IPRA-compiled, so a register it is handed
        cannot be a value the caller keeps across the call.
    An excess (more rendered than the signature takes, the invented argument
    the issue saw) is reported against the mangled signature only, and never
    for a variadic one.

    Each row keeps what the inline note needs (#1000 item 2): the argument
    registers past the rendered ones, with the value the call site wrote into
    each when it wrote one. Empty when every call agrees; never raises.
    """
    try:
        llil = getattr(func, "llil", None) or getattr(func, "low_level_il", None)
        calls = list(_calls_with_setup_windows(llil)) if llil is not None else []
        cc = getattr(func, "calling_convention", None)
        arg_regs = [_full_reg(bv.arch, str(r)) for r in (getattr(cc, "int_arg_regs", None) or [])]
    except Exception:  # noqa: BLE001 - no IL means nothing to compare
        return []
    rows: list[dict[str, Any]] = []
    for call, window, select_ids in calls:
        try:
            target = il_format._llil_constant_value(getattr(call, "dest", None))
            address = int(getattr(call, "address", 0))
            if target is None:
                continue
            mlil = getattr(call, "mlil", None)
            params = getattr(mlil, "params", None)
            if params is None:
                continue
            rendered = len(list(params))
            symbol = bv.get_symbol_at(target)
            callee = bv.get_function_at(target)
            kind = str(getattr(getattr(symbol, "type", None), "name", ""))
            raw_name = str(getattr(symbol, "raw_name", "") or getattr(callee, "name", "") or "")
            if _SIZED_DELETE_RE.match(raw_name):
                continue
            no_body = kind in _IMPORT_SYMBOL_KINDS or bool(getattr(callee, "is_thunk", False))
            signature = _demangled_arity(bv, raw_name)
            setup, writes = _argument_setup_count(bv.arch, window, arg_regs) if arg_regs else (0, {})
        except Exception:  # noqa: BLE001 - one odd call must not hide the rest
            continue
        setup_text = f"the call site writes {', '.join(writes)} just before it"
        if signature is not None:
            # The demangled signature is the arity whenever there is one: the
            # register witness only corroborates it, never overrides it.
            total, explicit, variadic = signature
            if rendered > total and not variadic:
                expected = total
                evidence = [f"the demangled signature takes only {total}"]
            elif rendered >= total or not (explicit > rendered or setup > rendered):
                continue
            else:
                expected = total
                evidence = [f"the demangled signature takes {total}"
                            + (" with `this`" if total != explicit else "")]
                if setup > rendered:
                    evidence.append(setup_text)
        elif no_body and setup > rendered:
            expected, evidence = setup, [setup_text]
        else:
            continue
        values = _window_values(bv, func, window, select_ids)
        missing = [(reg, values.get(reg) if reg in writes else None)
                   for reg in arg_regs[rendered:expected]]
        rows.append({
            "address": address, "target": target,
            "label": _callee_label(symbol, callee, target),
            "rendered": rendered, "expected": expected,
            "evidence": tuple(evidence), "missing": missing,
        })
    return rows


_WRITE_TEXT_MAX = 40
_IL_VAR_ANNOTATION_RE = re.compile(r"\s*\{[^{}]*\}")


def _reg_bits(arch, name: str) -> int | None:
    try:
        size = int(arch.regs[name].size)
    except Exception:  # noqa: BLE001 - an unknown register has no known width
        return None
    return size * 8 if size > 0 else None


def _narrowed(arch, spelled: str, text: str, bits: int | None) -> tuple[str, int | None]:
    """*text*, a value *bits* wide (None: as wide as the full register), as
    read through the register spelling *spelled*, with the width it then
    has: `mov w0, w1` copies only the low 32 bits of a 64-bit `arg2`, which
    is `(uint32_t)arg2`, not `arg2`."""
    narrow = _reg_bits(arch, spelled)
    full = _reg_bits(arch, _full_reg(arch, spelled))
    if narrow is None or full is None or narrow >= full or (bits is not None and bits <= narrow):
        return text, bits
    if re.fullmatch(r"0x[0-9a-fA-F]+", text):
        return hex(int(text, 16) & ((1 << narrow) - 1)), narrow
    cast = f"(uint{narrow}_t){text}" if re.fullmatch(r"\w+", text) else f"(uint{narrow}_t)({text})"
    return cast, narrow


def _entry_name(bv, func, address: int | None, reg: str) -> tuple[str, int | None] | None:
    """What *reg* holds at *address* by BN's own dataflow, as (text, bits): a
    constant, or the function's parameter (with its width) when the register
    still holds its entry value. None otherwise."""
    if address is None:
        return None
    try:
        value = func.get_reg_value_at(address, reg)
        kind = str(getattr(getattr(value, "type", None), "name", ""))
        if kind in ("ConstantValue", "ConstantPointerValue"):
            return hex(int(value.value)), None
        if kind != "EntryValue":
            return None
        entry = _full_reg(bv.arch, str(getattr(value, "reg", "") or ""))
        for var in list(getattr(func, "parameter_vars", None) or []):
            if _register_name(bv.arch, var) == entry:
                width = getattr(getattr(var, "type", None), "width", None)
                return str(var.name), int(width) * 8 if isinstance(width, int) and width > 0 else None
    except Exception:  # noqa: BLE001 - no dataflow: say nothing rather than guess
        return None
    return None


def _llil_writes(arch, ins) -> list[str]:
    """Every register *ins* writes, full width: a plain set, a split pair
    (`mul`'s rdx:rax), any other `LLIL_SET_REG*` form, and an intrinsic's
    outputs (`mrs x5, tpidr_el0`). A write this misses would let a note name
    a value the register no longer holds."""
    op = il_format._il_op_name(ins)
    if op == "LLIL_INTRINSIC":
        targets = list(getattr(ins, "output", None) or [])
    elif op.startswith("LLIL_SET_REG"):
        targets = [getattr(ins, key, None) for key in ("dest", "hi", "lo")]
    else:
        return []
    names = [str(getattr(t, "name", "")) for t in targets if getattr(t, "name", None)]
    return [_full_reg(arch, name) for name in names]


def _window_values(bv, func, window: list, select_ids: set[int]) -> dict[str, str | None]:
    """The value each register holds after *window*, in words a note can show.

    Taking a write's LLIL source text as its value was wrong whenever a
    register that source reads changed before the call: `mov x2, x0` copies
    `strlen`'s result, then `x0 = &s`, and the note said `x2 = x0`. So the
    window is evaluated forward: a register copy carries the copied value
    along (`w4 = w2; w2 = w3; w3 = w4` leaves x3 holding the old x2), and a
    register read at the window's start is named by BN's dataflow there (a
    parameter or a constant) or left as itself only if nothing in the window
    writes it. Any other source is shown with the same naming applied to each
    register it reads (`w1 + 1` -> `arg2 + 1` when x1 is overwritten later),
    and not at all when one it reads was already rewritten in the window or
    cannot be named. A value read through a narrower register keeps that
    width (`(uint32_t)arg2`), a write narrower than 32 bits (or any partial
    write on an architecture that does not zero-extend) leaves the value
    unknown, and a write this does not model (a split pair, an intrinsic)
    clobbers the register. A lifted select arm is its mnemonic and address
    (`csel @ 0x4015cc`): one arm is not the value. None means unknown, never
    a guess."""
    arch = bv.arch
    written = {reg for ins in window for reg in _llil_writes(arch, ins)}
    start = None
    if window:
        try:
            start = int(window[0].address)
        except Exception:  # noqa: BLE001
            start = None
    # Each value is (text, bits): bits is how wide the value is when narrower
    # than its register, so a later copy through a narrower spelling knows
    # whether it truncates.
    values: dict[str, tuple[str, int | None] | None] = {}
    written_so_far: set[str] = set()
    regs = getattr(arch, "regs", None) or {}
    zero_extends = str(getattr(arch, "name", "")) in _ZERO_EXTENDING_RETURN_ARCHS

    def spelled_bits(spelled: str) -> int | None:
        narrow, full = _reg_bits(arch, spelled), _reg_bits(arch, _full_reg(arch, spelled))
        return narrow if narrow is not None and full is not None and narrow < full else None

    def at_start(reg: str, spelled: str) -> tuple[str, int | None] | None:
        named = _entry_name(bv, func, start, reg)
        if named is not None:
            return _narrowed(arch, spelled, *named)
        return (spelled, spelled_bits(spelled)) if reg not in written else None

    def named_source(src, bits: int | None) -> tuple[str, int | None] | None:
        read: set[str] = set()
        _llil_regs_read(arch, src, read)
        if read & written_so_far:
            return None
        text = _IL_VAR_ANNOTATION_RE.sub("", str(src or "")).strip()
        unknown = False

        def rename(match: re.Match[str]) -> str:
            nonlocal unknown
            token = match.group(0)
            if token not in regs:
                return token
            named = _entry_name(bv, func, start, _full_reg(arch, token))
            if named is not None:
                return _narrowed(arch, token, *named)[0]
            if _full_reg(arch, token) in written:
                unknown = True
            return token

        text = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", rename, text)
        if unknown or not text or len(text) > _WRITE_TEXT_MAX:
            return None
        return text, bits

    for ins in window:
        if il_format._il_op_name(ins) != "LLIL_SET_REG":
            for reg in _llil_writes(arch, ins):
                values[reg] = None
                written_so_far.add(reg)
            continue
        spelled_dest = str(getattr(getattr(ins, "dest", None), "name", ""))
        dest = _full_reg(arch, spelled_dest)
        src = getattr(ins, "src", None)
        try:
            dest_bits = spelled_bits(spelled_dest)
            if dest_bits is not None and not (dest_bits == 32 and zero_extends):
                values[dest] = None  # the rest of the register keeps an older value
            elif id(ins) in select_ids:
                words = str(bv.get_disassembly(int(ins.address)) or "").split()
                values[dest] = (f"{words[0]} @ {hex(int(ins.address))}", dest_bits) if words else None
            elif il_format._il_op_name(src) == "LLIL_REG":
                name = str(getattr(getattr(src, "src", None), "name", ""))
                full = _full_reg(arch, name)
                if full in values:
                    held = values[full]
                    values[dest] = None if held is None else _narrowed(arch, name, *held)
                else:
                    values[dest] = at_start(full, name)
            else:
                values[dest] = named_source(src, dest_bits)
        except Exception:  # noqa: BLE001 - a value is a nicety; the register stands alone
            values[dest] = None
        written_so_far.add(dest)
    return {reg: (value[0] if value is not None else None) for reg, value in values.items()}


def _arity_line_note(row: dict[str, Any]) -> str:
    """The `/* bn: ... */` note appended to the rendered call line."""
    if row["rendered"] > row["expected"]:
        return (f"/* bn: renders {row['rendered']} args, "
                f"signature takes {row['expected']} */")
    regs = ", ".join(f"{reg} = {value}" if value else reg for reg, value in row["missing"])
    count = row["expected"] - row["rendered"]
    return f"/* bn: +{count} arg{'s' if count != 1 else ''} not rendered: {regs} */"


def _short_callee(label: str) -> str:
    return label.rsplit(" @ ", 1)[0]


def _call_arity_warning(bv, func, findings: list[dict[str, Any]] | None = None,
                        inline: set[int] = frozenset()) -> dict[str, Any] | None:
    """The `arity` warning for *findings* (computed when not given): grouped
    per callee and shape in its detail, one short line in its summary, and one
    `calls` row per call. *inline* holds the call addresses whose rendered
    line carries the `/* bn: ... */` note (#1000 item 2)."""
    if findings is None:
        findings = _call_arity_findings(bv, func)
    if not findings:
        return None
    groups: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in findings:
        # One entry per callee and shape, so ten identical calls cannot push a
        # rarer finding past the listing cap (#985 review).
        group = groups.setdefault((row["target"], row["rendered"], row["expected"]), {
            "label": row["label"], "addresses": [], "evidence": {},
        })
        group["addresses"].append(row["address"])
        # Every call's evidence, counted: calls to one callee can differ (the
        # signature alone at one, signature plus register writes at another),
        # and the first call's text must not stand for all (#985 re-review).
        group["evidence"][row["evidence"]] = group["evidence"].get(row["evidence"], 0) + 1
    described = []
    short = []
    for (_target, rendered, expected), group in groups.items():
        addrs = group["addresses"]
        where = ", ".join(hex(a) for a in addrs[:3]) + (
            f" and {len(addrs) - 3} more" if len(addrs) > 3 else "")
        noun = "call" if len(addrs) == 1 else f"{len(addrs)} calls"
        variants = group["evidence"]
        if len(variants) == 1:
            evidence = "; ".join(next(iter(variants)))
        else:
            evidence = " / ".join(f"{n} of {len(addrs)}: {'; '.join(ev)}"
                                  for ev, n in variants.items())
        if rendered > expected:
            described.append(f"{group['label']} ({noun} at {where}) renders {rendered} "
                             f"arguments but {evidence}")
        else:
            described.append(f"{group['label']} ({noun} at {where}) renders {rendered} of "
                             f"{expected} arguments ({evidence})")
        short.append(f"{_short_callee(group['label'])} {rendered}/{expected} @ "
                     + ", ".join(hex(a) for a in addrs[:2])
                     + (f" +{len(addrs) - 2}" if len(addrs) > 2 else ""))
    calls = len(findings)
    listed = described[:_ARITY_LISTED_MAX]
    more = len(described) - len(listed)
    marked = sum(1 for row in findings if row["address"] in inline)
    if marked == calls:
        marked_text = ("Each call is marked inline with a `/* bn: ... */` note naming "
                       "the registers it does not render. ")
        summary_mark = ", marked inline"
    elif marked:
        marked_text = (f"{marked} of the {calls} calls are marked inline with a "
                       "`/* bn: ... */` note; the rest share a line BN rendered "
                       "differently. ")
        summary_mark = f", {marked} marked inline"
    else:
        marked_text = ""
        summary_mark = ""
    head = short[:3]
    summary = (f"{calls} call(s) render the wrong argument count{summary_mark}: "
               + "; ".join(head) + (f"; +{len(short) - 3} more callee(s)" if len(short) > 3 else ""))
    detail = (
        f"call arity: {calls} call(s) render a different argument count than "
        f"the call site passes: {'; '.join(listed)}"
        + (f"; and {more} more callee(s)" if more > 0 else "")
        + ". The callee's prototype (often from a type library) has the wrong arity, so "
        "the decompiled call silently drops or invents arguments. " + marked_text
        + "Read the real arguments with `bn callsites <callee address> --within "
        f"{hex(int(func.start))} --context 4`; fix the callee with `bn proto set`."
    )
    return _warning_row("arity", summary, detail, calls=[
        {
            "address": hex(row["address"]),
            "callee": _short_callee(row["label"]),
            "callee_address": hex(row["target"]),
            "rendered": row["rendered"],
            "expected": row["expected"],
            "missing": [{"register": reg, "value": value} for reg, value in row["missing"]],
            "inline": row["address"] in inline,
        }
        for row in findings
    ])


def _callee_label(symbol, callee, target: int) -> str:
    name = str(getattr(symbol, "short_name", "") or "") if symbol is not None else ""
    if not name and callee is not None:
        name = il_format._display_name(callee)
    name = name.split("(", 1)[0] if name else ""
    return f"{name} @ {hex(target)}" if name else hex(target)


def _register_name(arch, var) -> str | None:
    """The register *var* lives in, or None for a stack/flag variable."""
    source = getattr(getattr(var, "source_type", None), "name", "")
    if source != "RegisterVariableSourceType":
        return None
    try:
        return _full_reg(arch, str(arch.get_reg_name(var.storage)))
    except Exception:  # noqa: BLE001
        return None


def _callee_saved_parameters(bv, func) -> list[str]:
    """`argN @ reg` for each parameter BN placed in a register no calling
    convention passes arguments in: a callee-saved register or the link
    register (#966). A register the function's own convention lists as an
    argument register is never reported, so a custom convention that really
    passes values there stays silent."""
    cc = getattr(func, "calling_convention", None)
    arch = bv.arch
    try:
        saved = {_full_reg(arch, str(r)) for r in (getattr(cc, "callee_saved_regs", None) or [])}
        args = {_full_reg(arch, str(r))
                for r in [*(getattr(cc, "int_arg_regs", None) or []),
                          *(getattr(cc, "float_arg_regs", None) or [])]}
        link = getattr(arch, "link_reg", None)
        if link:
            saved.add(_full_reg(arch, str(link)))
        params = list(getattr(func, "parameter_vars", None) or [])
    except Exception:  # noqa: BLE001
        return []
    out = []
    for var in params:
        reg = _register_name(arch, var)
        if reg and reg in saved and reg not in args:
            out.append(f"{getattr(var, 'name', '?')} @ {reg}")
    return out


def _falls_through(bv, address: int) -> bool:
    """True when the instruction at *address* can continue into the next one:
    no branch at all, a conditional branch (its not-taken side), or a call to
    a callee that returns."""
    try:
        data = bv.read(address, bv.arch.max_instr_length)
        info = bv.arch.get_instruction_info(data, address)
        branches = list(getattr(info, "branches", None) or [])
    except Exception:  # noqa: BLE001 - undecodable: claim nothing
        return False
    kinds = {str(getattr(getattr(b, "type", None), "name", "")) for b in branches}
    if not kinds or "FalseBranch" in kinds:
        return True
    if kinds <= {"CallDestination"}:
        for branch in branches:
            callee = bv.get_function_at(int(getattr(branch, "target", 0) or 0))
            if callee is not None and not bool(getattr(callee, "can_return", True)):
                return False
        return True
    return False


def _block_instructions(bv, block) -> list[int]:
    start, end = int(block.start), int(block.end)
    addresses = []
    address = start
    while address < end:
        addresses.append(address)
        length = bv.get_instruction_length(address)
        if not length:
            return []
        address += length
    return addresses


def _llil_op_at(func, address: int) -> str:
    try:
        return il_format._il_op_name(func.get_llil_at(address))
    except Exception:  # noqa: BLE001
        return ""


# Mnemonics compilers emit as alignment padding. Matched on the decoded text:
# BN lifts the padding before a function it ended early as a synthetic
# `tailcall`, so the IL cannot tell padding from code.
_PADDING_MNEMONICS = {"nop", "int3", "hlt", "udf"}


def _is_padding(bv, address: int) -> bool:
    try:
        text = str(bv.get_disassembly(address) or "")
    except Exception:  # noqa: BLE001
        return False
    words = text.split()
    return bool(words) and words[0].lower() in _PADDING_MNEMONICS


def _runs_into_next(bv, func, block) -> int | None:
    """The address of *block*'s last real instruction when the block (of
    *func*) ends by executing into the next address, else None.

    Trailing alignment padding is not code: compilers pad after a call that
    never returns (`call [rax]; nop ...`, an abort through a pointer BN cannot
    see is noreturn), so a block whose real last instruction is a call followed
    only by padding is a function end, not a split."""
    addresses = _block_instructions(bv, block)
    padded = False
    while addresses and _is_padding(bv, addresses[-1]):
        addresses.pop()
        padded = True
    if not addresses:
        return None
    if padded and "CALL" in _llil_op_at(func, addresses[-1]):
        return None
    return addresses[-1] if _falls_through(bv, addresses[-1]) else None


def _split_pair(bv, head, tail) -> bool:
    """Whether *head* running into *tail* reads as a shared-tail split. Not
    when either is an EH/cold fragment (`_is_eh_fragment`), and not when the
    tail never returns: BN outlines a bare `call __stack_chk_fail` or abort
    path that way, and no return path of the head is hidden behind it."""
    return bool(getattr(tail, "can_return", True)) \
        and not _is_eh_fragment(bv, tail) and not _is_eh_fragment(bv, head)


def _fallthrough_successor(bv, func) -> tuple[Any, int] | None:
    """``(function, source)``: the function whose entry *func*'s body runs
    straight into, and the address of the instruction that runs into it.
    The source is usually NOT the end of the listing: any block can end at
    another function's entry (#985 review)."""
    for block in list(getattr(func, "basic_blocks", None) or []):
        other = bv.get_function_at(int(block.end))
        if other is None or int(other.start) == int(func.start):
            continue
        source = _runs_into_next(bv, func, block)
        if source is not None and _split_pair(bv, func, other):
            return other, source
    return None


def _fallthrough_predecessor(bv, func):
    """The function whose body runs straight into *func*'s entry, or None."""
    start = int(func.start)
    try:
        blocks = list(bv.get_basic_blocks_at(start - 1) or [])
    except Exception:  # noqa: BLE001
        return None
    for block in blocks:
        owner = getattr(block, "function", None)
        if owner is None or int(owner.start) == start or int(block.end) != start:
            continue
        if _runs_into_next(bv, owner, block) is not None and _split_pair(bv, owner, func):
            return owner
    return None


_DIRECT_CALL_OPS = ("LLIL_CALL", "LLIL_CALL_STACK_ADJUST")
# BN chains a long cleanup through synthetic calls, one fragment per object
# destroyed; a stock C++ binary had chains well past three links.
_FRAGMENT_DEPTH = 32
# BN's placeholder name for a function no symbol names. Fragments are never named.
_DEFAULT_NAME_RE = re.compile(r"^sub_[0-9a-fA-F]+$")


def _direct_call_refs(bv, start: int) -> tuple[list, list]:
    """(every code reference to *start*, the ones that are real calls)."""
    refs = list(bv.get_code_refs(start) or [])
    return refs, [r for r in refs
                  if _llil_op_at(r.function, int(r.address)) in _DIRECT_CALL_OPS]


def _is_eh_fragment(bv, func, memo: dict[int, bool] | None = None, _depth: int = 0) -> bool:
    """True when *func* is exception-cleanup or cold-path code BN carved out
    of a parent function: a landing pad, a `.cold` part, or a chain of them.

    Such a fragment runs inside its parent's frame, so BN reads the parent's
    live callee-saved registers (`rbp`, `rbx`, ...) as its "parameters". On a
    C++ binary with exceptions these were ~98% of the #966 notes, and their
    `proto set` advice was wrong: dropping `rbp` breaks the frame accesses
    (#985 review). Recognised by any of:
      * a `.cold` name, or a parameter typed `_Unwind_Exception` (BN's own
        landing-pad reading of the exception object);
      * nothing CALLS it and nothing takes its address -- it is reached only by
        jumps, falls, or the unwinder -- and it never returns, or nothing
        references it at all (an LSDA landing pad has no code or data
        reference; a vtable or callback-table entry makes it an ordinary
        function called indirectly);
      * it never returns and every function calling it is itself a fragment
        (BN chains a long cleanup through synthetic calls).
    A shared tail split off a normal function returns, so it is not matched.
    A function with a real name (`.cold` aside) is never a fragment: an
    exported method that nothing in the binary calls looks unreferenced too.
    *memo* caches verdicts by start address across one read's lookups.
    """
    memo = {} if memo is None else memo
    try:
        start = int(func.start)
    except Exception:  # noqa: BLE001
        return False
    if start in memo:
        return memo[start]
    memo[start] = False  # a cycle of callers proves nothing
    try:
        name = str(getattr(func, "name", "") or "")
        params = list(getattr(getattr(func, "type", None), "parameters", None) or [])
        if ".cold" in name:
            verdict = True
        elif not _DEFAULT_NAME_RE.match(name):
            # A named function -- exported, or from the symbol table -- is an
            # entry point in its own right: an exported method nothing in the
            # binary calls has no references either, and is no landing pad.
            verdict = False
        elif any("_Unwind_Exception" in str(getattr(p, "type", "")) for p in params):
            verdict = True
        else:
            refs, callers = _direct_call_refs(bv, start)
            returns = bool(getattr(func, "can_return", True))
            if not callers:
                # An address-taken function (vtable slot, callback table) is
                # called indirectly; a landing pad is never address-taken (the
                # LSDA locates it by offset), so any data reference vetoes it.
                verdict = not list(bv.get_data_refs(start) or []) and (not returns or not refs)
            elif returns or _depth >= _FRAGMENT_DEPTH:
                verdict = False
            else:
                verdict = all(
                    r.function is not None
                    and _is_eh_fragment(bv, r.function, memo, _depth + 1)
                    for r in callers)
    except Exception:  # noqa: BLE001 - unknown: treat as an ordinary function
        verdict = False
    memo[start] = verdict
    return verdict


def _has_callers(bv, start: int) -> bool:
    """Whether anything can call the function at *start*: a direct call site,
    or a data reference -- a vtable slot, a `.data.rel.ro` entry, a callback
    table. An address-taken function is an ordinary function reached
    indirectly, not a piece of another body (#985 re-review), the same test
    `_is_eh_fragment` applies before calling anything a landing pad."""
    if _direct_call_refs(bv, start)[1]:
        return True
    return bool(list(bv.get_data_refs(start) or []))


def _split_body_notes(bv, func) -> list[dict[str, Any]]:
    """Disclose a function body BN split at a shared tail (#966).

    When another function branches into the tail of this one, BN can make the
    tail a function of its own and end this one early. The damage is quiet:
    the head's prototype gains "parameters" in callee-saved registers and the
    frame pointer (values the tail only restores), and every caller passes
    stray arguments for them. Two read-only signals name it: parameters in
    registers no convention passes arguments in, and a body that runs
    straight into another function's entry (or is run into by one).

    An EH/cold fragment gets one note of its own instead: its register
    parameters are its parent's live registers, which nothing should drop.
    So do they on any piece entered only by jumps or fall-through: the
    `proto set` advice is only for a function something actually calls."""
    notes: list[dict[str, Any]] = []
    memo: dict[int, bool] = {}
    try:
        params = _callee_saved_parameters(bv, func)
        if params and _is_eh_fragment(bv, func, memo):
            return [_warning_row(
                "fragment",
                f"EH/cold fragment: its register parameters ({', '.join(params)}) are "
                "the parent's live registers, not arguments",
                "EH/cold fragment: BN made this exception-cleanup (landing pad) or "
                "cold-path code a function of its own. It runs inside its parent's frame, "
                f"so its register parameters ({', '.join(params)}) are the parent's live "
                "registers, not arguments and not a split to repair. Read it together with "
                f"the code that jumps here (`bn xrefs {hex(int(func.start))}`).",
            )]
        called = bool(params) and _has_callers(bv, int(func.start))
        successor = _fallthrough_successor(bv, func)
        predecessor = _fallthrough_predecessor(bv, func)
    except Exception:  # noqa: BLE001 - a read-only nudge never sinks the read
        return notes
    if successor is not None:
        other, source = successor
        notes.append(_warning_row(
            "split",
            f"block ending at {hex(source)} runs into {_function_label(other)}: "
            "part of this body is listed under that function",
            f"split body: the block ending at {hex(source)} runs straight into "
            f"{_function_label(other)} without a branch or return -- BN likely split "
            "this function at a shared tail, so part of its code is listed under that "
            "one. Read both (`bn disasm`) before trusting either decompile.",
        ))
    if predecessor is not None:
        notes.append(_warning_row(
            "split",
            f"{_function_label(predecessor)} runs into this entry: this may be its "
            "shared tail, not a function of its own",
            f"split body: {_function_label(predecessor)} runs straight into this "
            "function's entry -- this may be the shared tail of that function, not a "
            "function of its own; its parameters are likely that function's saved registers.",
        ))
    if params and called:
        notes.append(_warning_row(
            "split",
            f"prototype takes parameters in callee-saved/link registers ({', '.join(params)}); "
            "drop them with `bn proto set`",
            f"prototype takes parameters in callee-saved/link registers ({', '.join(params)}): "
            "no calling convention passes arguments there, so BN is reading a value the "
            "body only saves or restores -- typically because it split the body at a "
            "shared tail, or a path reads the register before setting it. Callers render "
            "stray arguments for them; drop them with `bn proto set`.",
        ))
    elif params:
        notes.append(_warning_row(
            "split",
            f"parameters in callee-saved/link registers ({', '.join(params)}) are live-ins "
            "of the code that jumps here, not arguments; nothing calls this function",
            f"prototype takes parameters in callee-saved/link registers ({', '.join(params)}), "
            "and nothing calls this function -- it is entered only by jumps or fall-through, "
            "so it is a piece of another function's body and these are registers live in "
            "the code that jumps here, not arguments. Do not drop them; read it with the "
            f"code that jumps here (`bn xrefs {hex(int(func.start))}`).",
        ))
    return notes


def _function_label(func) -> str:
    return f"{il_format._display_name(func)} @ {hex(int(func.start))}"


def _callee_split_notes(bv, func) -> list[dict[str, Any]]:
    """#966 from the CALLER's side, where the symptom is read: a direct callee
    whose prototype takes callee-saved/link-register parameters gets stray
    arguments rendered at every call."""
    try:
        llil = getattr(func, "llil", None) or getattr(func, "low_level_il", None)
        instructions = list(llil.instructions) if llil is not None else []
    except Exception:  # noqa: BLE001
        return []
    checked: set[int] = {int(func.start)}
    notes = []
    for call in instructions:
        if il_format._il_op_name(call) not in _CALL_OPS:
            continue
        target = il_format._llil_constant_value(getattr(call, "dest", None))
        if target is None or target in checked:
            continue
        checked.add(target)
        callee = bv.get_function_at(target)
        if callee is None:
            continue
        params = _callee_saved_parameters(bv, callee)
        if params and not _is_eh_fragment(bv, callee):
            notes.append(_warning_row(
                "callee_split",
                f"callee {_function_label(callee)} takes callee-saved/link-register "
                f"parameters ({', '.join(params)}): their arguments here are not real",
                f"callee {_function_label(callee)} takes parameters in callee-saved/link "
                f"registers ({', '.join(params)}), so the arguments rendered for them here "
                "are not real arguments -- its body is likely split at a shared tail "
                f"(`bn decompile {hex(target)}` names the split).",
            ))
    return notes


_SHORT_STRING_MAX = 64
_STRING_NOTE_LISTED_MAX = 8
# Below three bytes, untyped byte previews are too noisy (#985).
# This is only a display threshold, never evidence that bytes are a string.
_UNTYPED_STRING_MIN = 3
_CHAR_POINTER_RE = re.compile(r"^(?:const\s+)?(?:(?:un)?signed\s+)?char\b[^*]*\*(?:\s*const)?$")

# Narrow `char *` parameter positions of C library functions, keyed by the
# `il_format._normalize_libc_name` spelling (`__isoc99_sscanf`, `strcmp@plt`
# and `_strcmp` all land on one key). These known APIs provide string intent
# by name; inferred prototypes and arbitrary type-library signatures do not.
# The printf/scanf format positions are NOT repeated here: they are
# `il_format._VARIADIC_FORMAT_FAMILY`'s, read through `_variadic_format_family`,
# so the two tables cannot disagree.
_LIBC_CHAR_POINTER_PARAMS: dict[str, frozenset[int]] = {
    name: frozenset(positions) for name, positions in {
        # comparison and search
        "strcmp": (0, 1), "strncmp": (0, 1), "strcasecmp": (0, 1),
        "strncasecmp": (0, 1), "strcoll": (0, 1), "strstr": (0, 1),
        "strcasestr": (0, 1), "strspn": (0, 1), "strcspn": (0, 1),
        "strpbrk": (0, 1), "strchr": (0,), "strrchr": (0,), "strchrnul": (0,),
        "strlen": (0,), "strnlen": (0,), "strtok": (0, 1), "strtok_r": (0, 1),
        "strsep": (1,), "fnmatch": (0, 1),
        # copying and duplication
        "strcpy": (0, 1), "strncpy": (0, 1), "stpcpy": (0, 1), "stpncpy": (0, 1),
        "strcat": (0, 1), "strncat": (0, 1), "strlcpy": (0, 1), "strlcat": (0, 1),
        "strdup": (0,), "strndup": (0,), "strcpy_chk": (0, 1), "strcat_chk": (0, 1),
        "strncpy_chk": (0, 1), "stpcpy_chk": (0, 1),
        # conversion
        "atoi": (0,), "atol": (0,), "atoll": (0,), "atof": (0,), "strtol": (0,),
        "strtoul": (0,), "strtoll": (0,), "strtoull": (0,), "strtod": (0,),
        "strtof": (0,), "strtold": (0,), "strtoimax": (0,), "strtoumax": (0,),
        # formatted I/O beyond the format family's format argument
        "asprintf": (1,), "vprintf": (0,), "vfprintf": (1,), "vsprintf": (1,),
        "vsnprintf": (2,), "sscanf": (0,), "err": (1,), "errx": (1,), "warn": (0,),
        "warnx": (0,), "puts": (0,), "fputs": (0,), "perror": (0,), "strftime": (2,),
        # environment, locale, process
        "getenv": (0,), "secure_getenv": (0,), "setenv": (0, 1), "unsetenv": (0,),
        "putenv": (0,), "setlocale": (1,), "newlocale": (1,), "system": (0,),
        "popen": (0, 1), "execl": (0, 1), "execlp": (0, 1), "execle": (0, 1),
        "execv": (0,), "execvp": (0,), "execve": (0,), "dlopen": (0,), "dlsym": (1,),
        "bindtextdomain": (0, 1), "textdomain": (0,), "gettext": (0,),
        "dgettext": (0, 1), "iconv_open": (0, 1), "openlog": (0,),
        # files and paths
        "fopen": (0, 1), "fopen64": (0, 1), "freopen": (0, 1), "fdopen": (1,),
        "open": (0,), "open64": (0,), "openat": (1,), "creat": (0,), "access": (0,),
        "stat": (0,), "lstat": (0,), "stat64": (0,), "lstat64": (0,), "unlink": (0,),
        "remove": (0,), "rename": (0, 1), "mkdir": (0,), "rmdir": (0,), "chdir": (0,),
        "chmod": (0,), "chown": (0,), "opendir": (0,), "realpath": (0,),
        "readlink": (0,), "symlink": (0, 1), "link": (0, 1), "truncate": (0,),
        "mkstemp": (0,), "mkdtemp": (0,), "tmpnam": (0,),
        # networking
        "inet_addr": (0,), "inet_aton": (0,), "inet_pton": (1,),
        "gethostbyname": (0,), "getaddrinfo": (0, 1), "getservbyname": (0, 1),
    }.items()
}


def _read_only_data(bv, address: int) -> bool:
    """True when *address* is in read-only, non-code data."""
    try:
        sections = list(bv.get_sections_at(address) or [])
    except Exception:  # noqa: BLE001
        sections = []
    if sections:
        return any(
            str(getattr(getattr(s, "semantics", None), "name", "")) == "ReadOnlyDataSectionSemantics"
            for s in sections
        )
    segment = bv.get_segment_at(address)
    return bool(segment is not None and getattr(segment, "readable", False)
                and not getattr(segment, "writable", True)
                and not getattr(segment, "executable", True))


def _undefined_c_string(bv, address: int, *, min_length: int) -> str | None:
    """Decode printable UTF-8 bytes before a NUL, not evidence of their type."""
    import unicodedata

    if bv.get_string_at(address) is not None:
        return None
    if not _read_only_data(bv, address):
        return None
    data = bytes(bv.read(address, _SHORT_STRING_MAX) or b"")
    end = data.find(b"\0")
    if end < max(1, min_length):
        return None
    try:
        value = data[:end].decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not all(
            c.isprintable() or c in "\t\n\r" or unicodedata.category(c) == "Zs"
            for c in value):
        return None
    return value


def _char_pointer_parameter(bv, target: int | None, index: int) -> bool:
    """String evidence is a user-set parameter type or a known string API."""
    if target is None:
        return False
    callee = bv.get_function_at(target)
    if bool(getattr(callee, "has_user_type", False)):
        params = list(getattr(getattr(callee, "type", None), "parameters", None) or [])
        return index < len(params) and bool(
            _CHAR_POINTER_RE.match(str(getattr(params[index], "type", "")).strip()))
    parameters = list(getattr(callee, "parameter_vars", None) or [])
    if index < len(parameters):
        var = parameters[index]
        if callee.is_var_user_defined(var):
            return bool(_CHAR_POINTER_RE.match(str(var.type).strip()))
    symbol = bv.get_symbol_at(target)
    kind = str(getattr(getattr(symbol, "type", None), "name", ""))
    if kind not in {"ImportedFunctionSymbol", "ImportAddressSymbol", "ExternalSymbol"}:
        return False
    name = str(getattr(symbol, "raw_name", "") or "").split("@", 1)[0]
    key = il_format._normalize_libc_name(name)
    family = il_format._variadic_format_family(name)
    return (index in _LIBC_CHAR_POINTER_PARAMS.get(key, ())
            or (family is not None and family[0] == index))


def _character_array_capacity(bv: Any, address: int) -> int:
    """Existing character-array capacity, independent of string evidence."""
    var = bv.get_data_var_at(address)
    match = re.fullmatch(
        r"(?:const\s+)?(?:(?:un)?signed\s+)?char(?:\s+const)?\s*"
        r"\[\s*(0x[0-9a-fA-F]+|\d+)\s*\]", str(getattr(var, "type", "")).strip())
    return int(match[1], 16 if match[1].startswith("0x") else 10) if match else 0


def _may_advise_character_array(bv: Any, address: int, size: int) -> bool:
    var = bv.get_data_var_at(address)
    if var is not None and not bool(getattr(var, "auto_discovered", False)):
        return False
    return _character_array_capacity(bv, address) < size


def _raw_string_pointer_note(bv, func, text: str) -> dict[str, Any] | None:
    """Describe raw call arguments using type facts, never byte length alone.

    Only known string-consuming APIs or user-set character-pointer parameters
    support string naming. Inferred types and printable bytes do not.
    """
    try:
        mlil = getattr(func, "mlil", None) or getattr(func, "medium_level_il", None)
        instructions = list(mlil.instructions) if mlil is not None else []
    except Exception:  # noqa: BLE001
        return None
    found: dict[int, tuple[str, int, bool]] = {}
    # Every address the text renders raw, tokenised once rather than searched
    # per operand (#985 review: decompile cost on large functions).
    # BN's placeholder name for untyped data counts too: a library prototype
    # that types the argument as a pointer (aarch64 libc's `setlocale(int32_t,
    # int64_t*)`) renders it `&data_400b90`, not `0x400b90` (#1000 review).
    # Placeholder names and raw hex addresses carry no string-type evidence.
    raw = {int(token, 16) for token in re.findall(r"\b0x[0-9a-fA-F]+\b", text)}
    placeholders = {int(token, 16)
                    for token in re.findall(r"\bdata_([0-9a-fA-F]+)\b", text)} - raw
    for ins in instructions:
        if "CALL" not in il_format._il_op_name(ins):
            continue
        dest = getattr(ins, "dest", None)
        target = getattr(dest, "constant", None) \
            if il_format._il_op_name(dest) in ("MLIL_CONST_PTR", "MLIL_CONST", "MLIL_IMPORT") \
            else None
        for index, param in enumerate(list(getattr(ins, "params", None) or [])):
            if il_format._il_op_name(param) not in ("MLIL_CONST_PTR", "MLIL_CONST"):
                continue
            try:
                address = int(param.constant)
                if address in found or (address not in raw and address not in placeholders):
                    continue
                typed = _char_pointer_parameter(bv, target, index)
                value = _undefined_c_string(
                    bv, address, min_length=1 if typed else _UNTYPED_STRING_MIN)
            except Exception:  # noqa: BLE001 - one odd operand must not hide the rest
                continue
            if value is not None:
                call = int(getattr(ins, "address", 0))
                found[address] = (value, call, typed)
    if not found:
        return None
    rows = [f"{hex(a)} = {_c_quote(v)} (call at {hex(c)})" if typed
            else f"bytes at {hex(a)} read as {_c_quote(v)} (call at {hex(c)})"
            for a, (v, c, typed) in list(found.items())[:_STRING_NOTE_LISTED_MAX]]
    more = len(found) - len(rows)
    strings = [{"address": hex(a), "value": v, "call": hex(c)}
               for a, (v, c, typed) in found.items() if typed]
    previews = [{"address": hex(a), "value": v, "call": hex(c)}
                for a, (v, c, typed) in found.items() if not typed]
    advice = ""
    undeclared = [row for row in strings
                  if _may_advise_character_array(
                      bv, int(row["address"], 16), len(row["value"].encode("utf-8")) + 1)]
    if undeclared:
        first = undeclared[0]
        size = len(first["value"].encode("utf-8")) + 1
        retype = f"bn data retype {first['address']} 'char[{size}]'"
        advice = f" To declare a character array, use `{retype}`; BN may still render an address."
    counts = ", ".join(f"{n} {label}" for n, label in (
        (len(strings), "undefined string(s)"), (len(previews), "byte preview(s)")) if n)
    return _warning_row(
        "string" if strings else "bytes",
        f"{counts}: {'; '.join(rows[:2])}"
        + (f"; +{len(found) - 2} more" if len(found) > 2 else "") + advice,
        f"raw call argument(s): {counts}: {'; '.join(rows)}"
        + (f"; and {more} more" if more > 0 else "")
        + (". Byte previews do not establish a string type." if previews else ".") + advice,
        address=hex(next(iter(found.values()))[1]),
        strings=strings, bytes=previews,
    )


def _c_quote(value: str) -> str:
    escaped = (value.replace("\\", "\\\\").replace('"', '\\"')
               .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))
    return f'"{escaped}"'


_RESOLVED_JUMP_OPS = ("HLIL_CONST", "HLIL_CONST_PTR", "HLIL_IMPORT", "HLIL_EXTERN_PTR")


def _is_plt_stub(bv, func) -> bool:
    """An import stub or the PLT header (`push [..]; jmp [GOT+16]`): its jump
    is the dynamic linker's business, not code a reader needs explained."""
    try:
        start = int(func.start)
        symbol = bv.get_symbol_at(start)
        if str(getattr(getattr(symbol, "type", None), "name", "")) in _IMPORT_SYMBOL_KINDS:
            return True
        if bool(getattr(func, "is_thunk", False)):
            return True
        return any(".plt" in str(getattr(section, "name", ""))
                   for section in list(bv.get_sections_at(start) or []))
    except Exception:  # noqa: BLE001 - unknown: treat as ordinary code
        return False


def _unresolved_jump_findings(bv, func) -> list[dict[str, Any]]:
    """Indirect jumps that leave *func* with no target BN resolved (#1000 item 3).

    BN renders one as `❓ /* jump -> <expr> */` with no warning. It is EITHER
    a tail call through a pointer -- whose arguments sit unrendered in the
    argument registers at the jump -- OR a computed goto / jump table BN
    could not follow. Telling the two apart from the IL failed three review
    rounds (a dispatch BN split into its own function, a label stored into a
    global first, an object picked by an index), so this states both and
    claims neither; the classification is a follow-up. A jump with a
    successor block (one BN resolved), an import stub and the PLT header are
    skipped. One row per jump. Never raises."""
    if _is_plt_stub(bv, func):
        return []
    try:
        hlil = getattr(func, "hlil", None) or getattr(func, "high_level_il", None)
        statements = list(hlil.instructions) if hlil is not None else []
    except Exception:  # noqa: BLE001 - no HLIL means nothing rendered to explain
        return []
    # Keyed by address: BN repeats one machine jump in several switch cases,
    # and each HLIL copy is the same jump -- one row, not one per copy.
    rows: dict[int, dict[str, Any]] = {}
    for ins in statements:
        if il_format._il_op_name(ins) != "HLIL_JUMP":
            continue
        try:
            dest = getattr(ins, "dest", None)
            if il_format._il_op_name(dest) in _RESOLVED_JUMP_OPS:
                continue
            address = int(ins.address)
            if address in rows:
                continue
            block = getattr(func.get_llil_at(address), "il_basic_block", None)
            if block is None or list(getattr(block, "outgoing_edges", None) or []):
                continue
            table = _jump_data_address(bv, dest)
            target = str(dest)
        except Exception:  # noqa: BLE001 - one odd jump must not hide the rest
            continue
        rows[address] = {"address": address, "target": target, "table": table}
    return list(rows.values())


def _jump_data_address(bv, dest) -> int | None:
    """The first mapped, non-code address in the jump target's expression --
    the data the target is read from, `&data_435ea0` in
    `*(int32_t*)(&data_435ea0 + (i << 2)) + &data_435ea0` -- or None. An
    operand that cannot be read names no table; it never costs the row."""
    pending = [dest]
    try:
        while pending:
            expr = pending.pop(0)
            if il_format._il_op_name(expr) == "HLIL_CONST_PTR" and _is_data_address(bv, expr):
                return int(expr.constant)
            pending.extend(o for o in list(getattr(expr, "operands", None) or [])
                           if hasattr(o, "operation"))
    except Exception:  # noqa: BLE001 - an unreadable operand names no table
        return None
    return None


def _is_data_address(bv, expr) -> bool:
    try:
        value = int(expr.constant)
        return bool(bv.is_valid_offset(value)) and not list(bv.get_functions_containing(value) or [])
    except Exception:  # noqa: BLE001 - an unreadable constant names no table
        return False


_JUMP_LINE_NOTE = "/* bn: unresolved jump: a tail call (args not rendered) or a computed goto */"


def _unresolved_jump_warning(func, row: dict[str, Any], inline: bool) -> dict[str, Any]:
    """The neutral `jump` warning: what `❓` means, the target, and both
    readings with how to check each. It never says which one this is.

    The jump-table half is honest rather than complete (#1000 review rounds
    5-6): every step below was run as printed on a run-time-filled `.bss`
    table, a static absolute table, a 4-byte relative table and stack-built
    label arrays, and each says when it does not help. None is promised to
    find the targets."""
    where = hex(row["address"])
    try:
        cc = getattr(func, "calling_convention", None)
        arg_regs = [str(r) for r in (getattr(cc, "int_arg_regs", None) or [])]
    except Exception:  # noqa: BLE001
        arg_regs = []
    regs = f" ({', '.join(arg_regs)})" if arg_regs else ""
    table = row.get("table")
    steps = []
    if table is not None:
        steps.append(
            f"`bn xrefs {hex(table)}` lists the code that reads and writes the data the "
            "jump reads from -- references to its base address only, so a store to a later "
            "entry may be missing; when it is filled at run time (a `.bss` table) its "
            "writes are where the target addresses are stored")
        steps.append(
            f"`bn evidence table {hex(table)}` reads that data's current bytes, which name "
            "targets only for a static table (absolute pointers; with `--stride 4 --width "
            f"4`, 4-byte signed offsets that are added to the table base {hex(table)}); a "
            "table filled at run time reads as zeros")
    steps.append(
        "stores in this decompile that write code or table addresses into what the jump "
        "reads may name the targets or their table; "
        "a table a store names is read with `bn evidence table <address>`. "
        "For a `data_<hex>` placeholder, use its hexadecimal address; "
        "resolve a symbol with `bn xrefs <symbol>` (its header gives `<address>`)")
    detail = (
        f"unresolved indirect jump at {where} to `{row['target']}`: BN renders it as "
        "`❓ /* jump -> ... */` -- the ❓ marks control flow it could not resolve -- and "
        "finds no target for it. It is one of two things, and the IL alone does not say "
        "which. A tail call through a pointer, leaving the function: the callee's "
        f"arguments are then in the argument registers{regs} at the jump and are not "
        f"rendered; read them in `bn disasm {where}`, and type the callee (`bn proto "
        "set`) or the object the pointer is loaded from so BN can render the call. Or a "
        "computed goto or jump table BN did not resolve, staying in this function: the "
        "code it reaches may not be part of this function's decompile or disassembly. None "
        "of these steps is sure to find the targets: " + "; ".join(steps) + ". Read a "
        "target address's code with `bn disasm <target> --linear 16`."
        + (" Every line that shows this jump carries a `/* bn: ... */` note." if inline else "")
    )
    summary = (f"unresolved jump at {where} (❓): a tail call whose args are not rendered, "
               "or a computed goto/jump table; --explain says how to read each")
    return _warning_row("jump", summary, detail, address=where, target=row["target"],
                        table=hex(table) if table is not None else None)


def _resolved_branch_warnings(bv: Any, func: Any) -> list[dict[str, Any]]:
    """Report BN's indirect-branch records without reconstructing missing targets."""
    if _is_plt_stub(bv, func):
        return []
    groups: dict[int, set[int]] = {}
    for branch in getattr(func, "indirect_branches", None) or []:
        source = int(branch.source_addr)
        groups.setdefault(source, set()).add(int(branch.dest_addr))
    warnings = []
    for source, destinations in sorted(groups.items()):
        targets = []
        separate = []
        for address in sorted(destinations):
            target: dict[str, Any] = {"address": hex(address)}
            other = bv.get_function_at(address)
            if other is not None and int(other.start) == address and address != int(func.start):
                target["function"] = {"address": hex(address),
                                      "name": il_format._display_name(other)}
                separate.append(_function_label(other))
            targets.append(target)
        where = hex(source)
        listed = ", ".join(target["address"] for target in targets[:3])
        if len(targets) > 3:
            listed += f" +{len(targets) - 3} more"
        detail = (f"BN reports indirect branch targets at {where}: "
                  + ", ".join(target["address"] for target in targets)
                  + ". These are the targets BN resolved, not a claim that every "
                  "table entry or possible destination was recovered.")
        if separate:
            detail += (" BN models these target addresses as separate function starts: "
                       + "; ".join(separate)
                       + ". Their bodies are listed separately; read them with "
                       "`bn decompile <address>`.")
        warnings.append(_warning_row(
            "branch", f"BN resolved {len(targets)} indirect branch target(s) at {where}: "
            + listed + (f"; {len(separate)} separate function start(s)" if separate else ""),
            detail, address=where, targets=targets))
    return warnings


_OFFSET_RE = re.compile(r"__offset\((.+?), (-?(?:0x[0-9a-fA-F]+|\d+))\)")
_OFFSET_SUMMARY_MAX = 240


def _type_lookup(bv):
    """`name -> defined?`, resolved the way `bn types show` resolves it
    (`seam._find_type`): BN's own lookup first, then the view's type names
    compared as rendered, case-insensitively. `get_type_by_name("std::X")`
    takes the whole string as ONE name component and misses every namespaced
    type (#1000 review round 8), and splitting on `::` would cut through
    template arguments (`std::map<std::string, int>`); comparing against
    each type's rendered qualified name does neither. The name list is built
    once, on the first miss."""
    names: set[str] | None = None

    def by_bn(name: str) -> bool:
        try:
            return bv.get_type_by_name(name) is not None
        except Exception:  # noqa: BLE001 - an unparseable name falls back to the scan
            return False

    def defined(name: str) -> bool:
        nonlocal names
        if by_bn(name):
            return True
        if names is None:
            try:
                names = {str(key).lower() for key in list(bv.types.keys())}
            except Exception:  # noqa: BLE001 - no type list: nothing more is known
                names = set()
        return name.lower() in names

    return defined


def _this_pointee(func) -> str | None:
    """The named type `this` points at, when *func* has a `this` parameter."""
    for var in list(getattr(func, "parameter_vars", None) or [])[:1]:
        if str(getattr(var, "name", "")) != "this":
            return None
        target = getattr(getattr(var, "type", None), "target", None)
        if target is None:
            return None
        ref = getattr(target, "named_type_reference", None)
        name = getattr(ref, "name", None) if ref is not None and not callable(ref) else None
        if name is None:
            text = str(target)
            for prefix in ("const ", "struct ", "class ", "union "):
                text = text.removeprefix(prefix)
            name = text.removesuffix(" const").strip()
        return str(name) or None
    return None


def _name_list(names: list[str], budget: int) -> str:
    """Fit whole names and an omitted count within the caller's total budget."""
    shown: list[str] = []
    used = 0
    for index, name in enumerate(names):
        rest = len(names) - index - 1
        suffix = f" +{rest} more" if rest else ""
        added = len(name) + (2 if shown else 0)
        if used + added + len(suffix) > budget:
            break
        shown.append(name)
        used += added
    rest = len(names) - len(shown)
    return ", ".join(shown) + (f"{' ' if shown else ''}+{rest} more" if rest else "")


def _offset_warning(bv, func, text: str) -> dict[str, Any] | None:
    """Explain raw `__offset(T, N)` expressions and say which T exist (#1000
    item 4). The old note sent the reader to `bn types show T` / `bn struct
    show T` "as the authoritative layout" without checking T existed; T is
    routinely a name BN made from a symbol or a demangled signature with no
    definition behind it, so both commands answered `Type not found` and the
    reader started doubting the decompile. Each base type, and the type `this`
    points at, is looked up: the ones that exist are named, and the ones that
    do not are called what they are, with the command that would create one."""
    names: list[str] = []
    for match in _OFFSET_RE.finditer(text):
        if match.group(1) not in names:
            names.append(match.group(1))
    if not names and "__offset(" not in text:
        return None
    this_type = _this_pointee(func)
    checked = names + ([this_type] if this_type and this_type not in names else [])
    _type_defined = _type_lookup(bv)
    defined = [name for name in checked if _type_defined(name)]
    missing = [name for name in checked if name not in defined]
    parts = [("Decompile contains raw `__offset(T, N)` expressions: BN's spelling for a "
              "pointer N bytes into an object of type T where T has no member it can name there.")]
    if defined:
        parts.append(f"Defined here: {', '.join(defined)} (`bn types show <name>` has the layout).")
    vtables = [n for n in missing if n.startswith("vtable_for_")]
    plain = [n for n in missing if n not in vtables]
    if plain:
        with_vtable = [n for n in plain if _type_defined(f"{n}::VTable")]
        parts.append(
            f"No definition: {', '.join(plain)} -- "
            + ("each is a named reference" if len(plain) > 1 else "a named reference")
            + " BN created (from a symbol or a demangled signature) with no layout, so "
            "`bn types show` reports it as not found. Declare one with "
            "`bn types declare` (a `::` name goes in backticks: struct `ns::T` { ... };)."
            + (f" {', '.join(with_vtable)} already has a `::VTable` type, and `bn struct "
               f"show {shlex.quote(with_vtable[0])}` prints the declaration to start from."
               if with_vtable else ""))
    if vtables:
        parts.append(
            f"{', '.join(vtables)} name{'s' if len(vtables) == 1 else ''} a class's vtable "
            "OBJECT, not a struct to declare: `__offset(vtable_for_C, 2 * pointer size)` is "
            "the vtable's address point, the value a constructor stores into an object's vptr.")
    if this_type and this_type in missing and this_type not in names:
        parts.append(f"`this` points at {this_type}, which has no definition either.")
    groups = [(label, group) for label, group in (
        ("no type defined for ", plain),
        ("vtable object(s), not types: ", vtables),
        ("defined: ", defined),
    ) if group]
    prefix = "__offset(T, N) is a pointer N bytes into T; "
    budget = (_OFFSET_SUMMARY_MAX - len(prefix)
              - sum(len(label) for label, _ in groups) - 2 * max(0, len(groups) - 1))
    summary_parts = [label + _name_list(group, budget // len(groups))
                     for label, group in groups]
    summary = prefix + ("; ".join(summary_parts) or "its base types could not be read")
    return _warning_row("offset", summary, " ".join(parts), address=hex(int(func.start)),
                        types={"defined": defined, "missing": missing})


def _decompile_batch(
    ctx,
    selector: str | None,
    identifiers,
    *,
    addresses: bool = False,
    force_analysis: bool = False,
    include_annotations: bool = False,
):
    """Decompile several functions in ONE round trip (#676 item 5).

    The pain this removes is not typing: every agent that needed three
    functions wrote a bash `for` loop, which pays bridge round-trip latency
    per function and -- the part that actually costs -- turns one readable
    output into N shell invocations whose failures are N separate exit codes
    nobody aggregates.

    An identifier that does not resolve becomes a FAILED ROW, not an abort.
    That is the whole reason to batch: a typo in the third name must not
    discard the two functions that did resolve, because the caller already
    paid the analysis cost for them. The row carries the error text so the
    miss is visible per-identifier rather than as one collapsed failure.
    """
    rows: list[dict[str, Any]] = []
    for identifier in identifiers:
        try:
            rows.append({
                "identifier": identifier,
                "ok": True,
                "decompiled": _decompile(
                    ctx, selector, identifier,
                    addresses=addresses,
                    force_analysis=force_analysis,
                    include_annotations=include_annotations,
                ),
            })
        except Exception as exc:  # noqa: BLE001 - one miss must not sink the batch
            rows.append({
                "identifier": identifier,
                "ok": False,
                "error": str(exc) or exc.__class__.__name__,
            })
    return {
        "kind": "decompile_batch",
        "requested": len(rows),
        "resolved": sum(1 for row in rows if row["ok"]),
        "functions": rows,
    }


def _decompile(
    ctx,
    selector: str | None,
    identifier,
    *,
    addresses: bool = False,
    force_analysis: bool = False,
    include_annotations: bool = False,
):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    forced = False
    if force_analysis and bool(getattr(func, "analysis_skipped", False)):
        func = _force_function_analysis(ctx, bv, func)
        forced = True

    comments = il_format._comment_map(bv, func)
    # Computed before rendering so the rendered call line can carry its own
    # note (#1000 items 2-3): a trailing warning alone left the line reading
    # like the whole call.
    arity = _call_arity_findings(bv, func)
    jumps = _unresolved_jump_findings(bv, func)
    line_notes: dict[int, list[str]] = {}
    for row in arity:
        line_notes.setdefault(row["address"], []).append(_arity_line_note(row))
    jump_notes = {row["address"]: _JUMP_LINE_NOTE for row in jumps}
    placed: set[int] = set()
    text = il_format._decompile_text(bv, func, addresses=addresses,
                                     line_notes=line_notes, placed=placed,
                                     jump_notes=jump_notes)
    if include_annotations:
        annotation_summary = {
            "comment_count": len(_annotation_entries(func, comments)),
            "redacted": False,
            "hidden": 0,
            "session_shown": 0,
        }
    else:
        text, annotation_summary = _redact_function_annotations(
            ctx, bv, func, comments, text
        )

    warnings: list[dict[str, Any]] = []
    offset_warn = _offset_warning(bv, func, text)
    if offset_warn:
        warnings.append(offset_warn)
    stub = il_format._analysis_stub_warning(func, text, forced=forced)
    if stub:
        warnings.append(_stub_warning_row(func, stub, forced))
    if "/* tailcall */" in text:
        thunk_warn = _thunk_veneer_warning(ctx, bv, func)
        if thunk_warn:
            warnings.append(thunk_warn)
    if forced:
        data_warn = _forced_data_region_warning(bv, func)
        if data_warn:
            warnings.append(data_warn)
    guard_warn = _flattened_guard_warning(bv, func)
    if guard_warn:
        warnings.append(guard_warn)
    warnings.extend(_struct_layout_warnings(bv, func))
    arity_warn = _call_arity_warning(bv, func, arity, placed)
    if arity_warn:
        warnings.append(arity_warn)
    warnings.extend(_unresolved_jump_warning(func, row, row["address"] in placed)
                    for row in jumps)
    warnings.extend(_resolved_branch_warnings(bv, func))
    warnings.extend(_split_body_notes(bv, func))
    warnings.extend(_callee_split_notes(bv, func))
    string_note = _raw_string_pointer_note(bv, func, text)
    if string_note:
        warnings.append(string_note)
    for warning in warnings:
        warning.setdefault("address", hex(int(func.start)))
    result = {
        "function": {"name": func.name, "address": hex(func.start)},
        "text": text,
        "comments": comments if include_annotations else {},
        "annotation_summary": annotation_summary,
        # `warnings` keeps the one shape every read op uses -- a list of
        # sentences, each the full detail -- so a reader of `il`, `bundle`,
        # `evidence` or an older decompile is unaffected (#1000 review).
        # `warning_rows` is the same list, in order, as `{code, summary,
        # detail, ...}` with the structured facts; text mode prints from it.
        "warnings": [row["detail"] for row in warnings],
        "warning_rows": warnings,
        "analysis_skipped": bool(getattr(func, "analysis_skipped", False)),
        "analysis_force_requested": bool(force_analysis),
        "analysis_forced": forced,
        "include_annotations": bool(include_annotations),
        # #820: the quick-mode matrix lets decompile ANSWER (partial) rather than
        # refuse, so the envelope must carry the view-level state too. Without it
        # a --quick body renders plausible Pseudo-C with unresolved `sub_<addr>`
        # callees and `warnings: []`, and `analysis_skipped` (a PER-FUNCTION flag
        # on an analyzed view) reads as "nothing was degraded". Distinct from
        # analysis_skipped: this is the VIEW's state, forced=False notwithstanding.
        **_analysis_state_fields(bv),
    }
    _annotate_containment(ctx, result, identifier, func)
    return result


def _basic_block_ranges(func) -> list[dict[str, Any]]:
    """Address ranges of a function's basic blocks, with incoming/outgoing edges.

    #653.10: a 23 KB / 565-block dispatch function decompiles to 6324 lines
    (100k tokens), and `--lines` slices blindly because the line numbers are
    unknowable until AFTER the full decompile -- so every downstream question
    became grep/sed over a local file instead of a `bn` query. Block ranges make
    `disasm --linear <addr>` / `--lines` targetable without reading the whole body
    first, and on a jump-table dispatcher each case arm is its own block.
    """
    blocks = []
    for index, block in enumerate(getattr(func, "basic_blocks", None) or []):
        start = int(getattr(block, "start", 0) or 0)
        end = int(getattr(block, "end", 0) or 0)
        entry: dict[str, Any] = {
            "index": index,
            "start": hex(start),
            "end": hex(end),
            "length": max(0, end - start),
        }
        try:
            entry["outgoing"] = [hex(int(e.target.start)) for e in (block.outgoing_edges or [])]
        except Exception:
            entry["outgoing"] = []
        try:
            entry["incoming"] = [hex(int(e.source.start)) for e in (block.incoming_edges or [])]
        except Exception:
            entry["incoming"] = []
        blocks.append(entry)
    blocks.sort(key=lambda b: int(b["start"], 16))
    return blocks


def _function_info(ctx, selector: str | None, identifier, *, blocks: bool = False):
    bv = ctx._resolve_view(selector)
    require_analysis(bv, "Function info")
    func = ctx._find_function(bv, identifier, contained=True)
    metadata = il_format._function_metadata(func)
    variables = vars_mod._list_locals(func)
    parameters = [item for item in variables if item["is_parameter"]]
    locals_only = [item for item in variables if not item["is_parameter"]]
    # Use the genuine code-ref count so a page-aligned function isn't credited
    # with spurious adrp page-base materializations (#284).
    code_ref_count = read_xrefs._code_ref_count(bv, func.start)
    result = {
        "function": {
            "name": func.name,
            "address": hex(func.start),
            "raw_name": getattr(func, "raw_name", func.name),
            "display_name": il_format._display_name(func),
        },
        **metadata,
        "parameters": parameters,
        "locals": locals_only,
        "xref_count": code_ref_count,
        # Count (+ addresses) of instructions BN's lifter could not model, so a
        # function with unlifted computation isn't mistaken for fully analyzed
        # (#206). count:0 means no unlifted instruction was found.
        "unimplemented_instructions": il_format._unimplemented_instructions(func),
    }
    if blocks:
        # #653.10: opt-in -- a 565-block function's ranges are themselves bulky.
        result["blocks"] = _basic_block_ranges(func)
    _annotate_containment(ctx, result, identifier, func)
    return result


# Architectures whose ABI returns in a register WIDER than a 32-bit write,
# where that narrower write ZERO-EXTENDS the full register -- so BN infers a
# 64-bit return for a function that only ever writes 32 bits.
#
# Confirmed per arch rather than inherited from x86 (#675 item 5): on x86_64
# the `eax` write zero-extends `rax`, and AArch64 has the SAME relation
# (`w0` zero-extends `x0`) -- measured, a cross-built AArch64 binary shows
# the identical `uint64_t(int32_t, int32_t)` shape. ARM32/thumb2 does NOT
# belong: its registers are natively 32-bit, there is no widening, and the
# same measurement finds ZERO over-wide functions there.
#
# Emitting nothing outside this set is the point. A note that appeared on
# ARM firmware while explaining an x86/AArch64 register relation would be
# worse than no note: it would be a confident claim about a mechanism that
# target does not have.
_ZERO_EXTENDING_RETURN_ARCHS = frozenset({"x86_64", "aarch64"})


def _return_width_inference_note(bv, func) -> str | None:
    """Disclose a return width BN inferred from the full register (#675 item 5).

    `int add(int, int)` is recovered as `uint64_t(int32_t, int32_t)` because
    the callee writes `eax`/`w0` and that write zero-extends the 64-bit
    register, so BN cannot tell a 32-bit return from a 64-bit one. The
    prototype then states two different confidence levels in one line and
    marks neither, and the harm reaches a SECOND surface: a caller returning
    -1 renders `return 0xffffffff;`.

    A NOTE, not a different answer -- BN cannot do better here, and
    substituting a narrower type would be the fabrication. It misleads a
    reader rather than breaking a computation, which is why this discloses
    where the reserved-keyword guard refuses.

    The signal is narrow ON PURPOSE. `has_user_type is False` is true of
    nearly every function in a stripped binary, so a note keyed on it fires
    everywhere and is skipped within a week; return-type confidence reads
    its MAXIMUM on exactly the over-wide return. What selects the real
    population is the shape: a pointer-width integer return on an arch where
    a 32-bit write zero-extends, with the declared parameters all narrower.
    """
    try:
        arch = str(getattr(getattr(bv, "arch", None), "name", "") or "")
        if arch not in _ZERO_EXTENDING_RETURN_ARCHS:
            return None
        if bool(getattr(func, "has_user_type", False)):
            return None          # an analyst set this; it is not an inference
        rtype = getattr(func, "return_type", None)
        if rtype is None or int(getattr(rtype, "width", 0) or 0) != 8:
            return None
        params = list(getattr(func, "parameter_vars", []) or [])
        if not params:
            return None
        if any(int(getattr(getattr(p, "type", None), "width", 8) or 8) != 4
               for p in params):
            return None
    except Exception:  # noqa: BLE001 - no evidence is not a claim
        return None
    reg = "eax/rax" if arch == "x86_64" else "w0/x0"
    return (
        f"return width may be INFERRED, not observed: on {arch} a 32-bit "
        f"{reg.split('/')[0]} write zero-extends {reg.split('/')[1]}, so a "
        f"function returning a 32-bit value is indistinguishable from one "
        f"returning 64 bits and BN widens it. Every declared parameter here "
        f"is 32-bit, which is the shape that produces the ambiguity -- a "
        f"negative return will render as a large unsigned value (e.g. -1 as "
        f"0xffffffff) at call sites. Set the prototype if you know the width."
    )


def _get_prototype(ctx, selector: str | None, identifier):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    result = {
        "function": {
            "name": func.name,
            "address": hex(func.start),
            "raw_name": getattr(func, "raw_name", func.name),
            # #999 item 3: the demangled name `decompile` shows; `name` stays
            # the symbol name the text splices into a pasteable C declaration.
            "display_name": il_format._display_name(func),
        },
        **il_format._function_metadata(func),
    }
    note = _return_width_inference_note(bv, func)
    if note:
        result["return_width_note"] = note
    split_notes = [row["detail"] for row in _split_body_notes(bv, func)]
    if split_notes:
        result["split_body_notes"] = split_notes
    _annotate_containment(ctx, result, identifier, func)
    return result


def _list_locals_for_function(ctx, selector: str | None, identifier):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    variables = vars_mod._list_locals(func)
    result = {
        "function": {
            "name": func.name,
            "address": hex(func.start),
            "raw_name": getattr(func, "raw_name", func.name),
        },
        # #651: `items` is documented as ALWAYS the data container, and `local list`
        # was the one exception -- documented once in #248, then lost in the
        # SKILL.md -> reference/ split. `jq '.items[]'` on a function with 40
        # recovered locals silently reported nothing (a wrong key yields null, never
        # an error). Emit `items` so the documented rule is true, and keep `locals`
        # as a retained alias so existing scripts/skills keep working. Both are the
        # same list object -- there is no second source of truth to drift.
        "kind": "locals",
        "items": variables,
        "locals": variables,
    }
    _annotate_containment(ctx, result, identifier, func)
    return result


def _list_frame_for_function(ctx, selector: str | None, identifier):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    frame = vars_mod._list_frame(func)
    result = {
        "function": {
            "name": func.name,
            "address": hex(func.start),
            "raw_name": getattr(func, "raw_name", func.name),
        },
        "kind": "frame",
        "items": frame["slots"],
        "slot_count": frame["slot_count"],
        "variable_count": frame["variable_count"],
        "canary_storage": frame["canary_storage"],
    }
    _annotate_containment(ctx, result, identifier, func)
    return result


def _il(ctx, selector: str | None, identifier, view: str, ssa: bool):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    text = il_format._function_text(bv, func, view=view, ssa=ssa)
    result = {
        "function": {"name": func.name, "address": hex(func.start)},
        "view": view,
        "ssa": ssa,
        "text": text,
        "warnings": il_format._render_warnings(text),
    }
    _annotate_containment(ctx, result, identifier, func)
    return result


_CFG_VIEWS = ("asm", "mlil", "hlil")


def _cfg(ctx, selector: str | None, identifier, *, view: str = "asm"):
    """Basic-block CFG of one function: blocks with rendered lines and outgoing
    edges, at the asm, MLIL, or HLIL level.

    Block identity contract (what bn-lens keys on): at IL levels `start` and
    edge `to` are IL instruction INDEXES (hex), because first-line addresses
    collide when one assembly instruction expands into several IL blocks; at
    the asm level `start` is the block's start address. Each rendered line
    keeps a real address in `address` at every level.

    Key spelling (#682 item 2): lines are `{address, text}` and edges
    `{to, branch_type}` -- the pay-as-you-read contract is the neighbour
    `_structured_il`'s (`address`/`text`), not a one-letter wire form, because
    these payloads are read by models and `a`/`t`/`k` are opaque without the
    reference open. `branch_type` rather than `kind` because `kind` is already
    the ENVELOPE discriminator on this same payload ("kind": "cfg").
    """
    level = str(view or "asm").lower()
    if level not in _CFG_VIEWS:
        raise OperationFailure(
            "invalid_request",
            f"unknown cfg view {view!r}; expected one of: {', '.join(_CFG_VIEWS)}",
        )
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    warnings: list[str] = []
    fn = func
    if level != "asm":
        try:
            fn = func.mlil if level == "mlil" else func.hlil
        except Exception:
            fn = None
        if fn is None:
            warnings.append(
                f"{level} is not available for this function (analysis skipped or IL "
                "not materialized); returning an empty CFG"
            )
    blocks = []
    if fn is not None:
        # #682 item 4: BN hands back basic blocks in its own iteration order,
        # which at the asm level is not address order (observed interleaving
        # such as 0x401090, 0x4010b0, 0x40109d). IL levels happen to come out
        # sorted, so this is a no-op there and makes the asm render readable.
        # Sorting on `bb.start` matches the documented identity contract at
        # every level: an address at asm, an IL index at MLIL/HLIL.
        ordered = sorted(fn.basic_blocks, key=lambda bb: int(bb.start))
        for bb in ordered:
            insns = [
                {"address": hex(line.address), "text": "".join(str(t) for t in line.tokens)}
                for line in bb.disassembly_text
            ]
            # `edge.type` is a BranchType member, so `.name` is the readable
            # kind -- but fall back to str() rather than raising the whole op if
            # a core ever hands back a bare int (the same class of surprise as
            # a relocation/symbol enum arriving unwrapped).
            #
            # #682 item 3. A null-target edge is NOT emitted, and that is a
            # deliberate reversal of this PR's first cut. Two measurements
            # decided it. (a) The shape is unreachable on BN 6.1:
            # `BasicBlock._make_edges` asserts `BNNewBasicBlockReference` is
            # non-None and wraps it through `_create_instance`, which cannot
            # return None, and a sweep of 12,191 functions x 3 IL levels over
            # four targets found zero. (b) Emitting `to: null` would silently
            # BREAK a known consumer: bn-tui types the field `pub to: String`
            # (not Option, no serde default), so a null fails the decode --
            # and because the failure is inside `Vec<CfgEdge>`, the whole
            # `CfgJson` parse fails and the TUI shows an EMPTY CFG with no
            # error. Speculative code whose only effect, on the day it fires,
            # is to blank a consumer's view is worse than no code.
            #
            # An unresolved target is therefore reported the same way the
            # LIVE case is, on the block, where the disclosure is additive
            # and breaks nothing (bn-tui sets no `deny_unknown_fields`, so an
            # unknown block key is ignored).
            edges = []
            unresolved_target = False
            for edge in bb.outgoing_edges:
                kind = getattr(edge.type, "name", None) or str(edge.type)
                if edge.target is None:
                    unresolved_target = True
                    continue
                edges.append({"to": hex(edge.target.start), "branch_type": kind})
            block = {"start": hex(bb.start), "insns": insns, "edges": edges}
            # #682 item 3, LIVE half. A real unresolvable indirect jump
            # (`jmp rax`) produces NO EDGES AT ALL, so it rendered
            # byte-identically to a block that simply has no successor -- the
            # one distinction a control-flow view must not lose. BN answers
            # it directly: `has_undetermined_outgoing_edges` is the core's
            # own "I could not determine where this goes". Read defensively
            # because reduced views and IL block objects need not implement
            # it, and an absent or throwing probe is indeterminate -- not a
            # claim of either kind.
            #
            # What the marker claims, exactly: it reports BN's VERDICT, it
            # does not exhaustively classify lost successors. Measured live,
            # 383 of 48,528 asm blocks carry it and none of the 42,034 blocks
            # WITH resolved successors do -- but 368 jmp-terminated tail-call
            # stubs have zero edges with the flag FALSE, so an unmarked
            # edgeless block is not proof of a genuine dead end. It is also
            # asm-level on this build: at MLIL/HLIL the same block reports
            # False because BN models the jump as a `__tailcall` return. Both
            # are BN's answer being relayed faithfully, not a bridge drop.
            try:
                undetermined = bool(bb.has_undetermined_outgoing_edges)
            except Exception:  # noqa: BLE001 - absent/raising probe says nothing
                undetermined = False
            if undetermined or unresolved_target:
                block["undetermined_edges"] = True
            blocks.append(block)
    result = {
        "kind": "cfg",
        "function": {"name": func.name, "address": hex(func.start)},
        "view": level,
        "blocks": blocks,
        "warnings": warnings,
    }
    _annotate_containment(ctx, result, identifier, func)
    return result


def _disasm(ctx, selector: str | None, identifier, linear=None, mode=None,
            snap_to_instruction=False, line_start=None, line_end=None,
            strict_range=False):
    bv = ctx._resolve_view(selector)
    if linear is not None:
        if line_start is not None or line_end is not None:
            raise OperationFailure(
                "invalid_range", "disasm line ranges cannot be combined with linear mode"
            )
        return _disasm_linear(ctx, bv, identifier, int(linear), mode=mode,
                              snap_to_instruction=bool(snap_to_instruction))
    try:
        func = ctx._find_function(bv, identifier, contained=True)
    except Exception as exc:
        looks_like_address = True
        try:
            _parse_address(identifier)
        except ValueError:
            looks_like_address = False
        if looks_like_address and "No function found" in str(exc):
            raise RuntimeError(
                f"{exc}. To inspect the raw bytes there regardless of function "
                f"membership, use `disasm {identifier} --linear N`."
            ) from exc
        raise

    full_text = il_format._disasm_text(bv, func)
    all_lines = full_text.splitlines()
    total_lines = len(all_lines)
    returned_text = full_text
    selected_range = None
    if line_start is not None or line_end is not None:
        try:
            start = int(line_start)
            end = int(line_end)
        except (TypeError, ValueError) as exc:
            raise OperationFailure(
                "invalid_range", "disasm lines must be 1-indexed integer START:END"
            ) from exc
        if start < 1 or end < start:
            raise OperationFailure(
                "invalid_range",
                f"disasm lines must satisfy 1 <= START <= END, got {start}:{end}",
            )
        if start > total_lines or (strict_range and end > total_lines):
            raise OperationFailure(
                "invalid_range",
                f"disasm line range {start}:{end} is outside the "
                f"1:{total_lines} listing; these are 1-indexed line numbers, "
                "not addresses",
            )
        end = min(end, total_lines)
        returned_text = "\n".join(all_lines[start - 1:end])
        selected_range = {"start": start, "end": end}

    result = {
        "function": {"name": func.name, "address": hex(func.start)},
        "text": returned_text,
        "total_lines": total_lines,
        "returned_lines": len(returned_text.splitlines()) if returned_text else 0,
        "line_range": selected_range,
    }
    # #966: a listing that stops where BN split the body read as a function that
    # simply ended. Say where execution actually continues -- and from WHICH
    # instruction: the block that runs into the other entry is often mid-listing
    # (#985 review), so `from` is what a renderer anchors the note to.
    try:
        successor = _fallthrough_successor(bv, func)
    except Exception:  # noqa: BLE001 - a read-only nudge never sinks the read
        successor = None
    if successor is not None:
        other, source = successor
        result["continues_into"] = {
            "name": other.name,
            "address": hex(int(other.start)),
            "from": hex(source),
            "note": (
                f"falls through into {_function_label(other)}: BN made the code after "
                f"{hex(source)} a separate function (likely a shared tail split); "
                f"continue with `bn disasm {hex(int(other.start))}`"
            ),
        }
    _annotate_containment(ctx, result, identifier, func)
    return result


# Hard ceiling on a single linear-disassembly request so a pathological count
# can't walk the whole address space; the output spill already handles large
# (but bounded) dumps.
_LINEAR_DISASM_MAX = 100_000


def _resolve_linear_address(ctx, bv, identifier) -> int:
    """Resolve a linear-disasm target to a concrete address: a literal address as
    given, else a function/symbol NAME to its start. Raises a clear error when it
    resolves to nothing."""
    try:
        return _parse_address(identifier)
    except ValueError:
        pass
    try:
        func = ctx._find_function(bv, identifier)
        if func is not None:
            return int(func.start)
    except Exception:
        pass
    try:
        symbol = bv.get_symbol_by_raw_name(str(identifier))
    except Exception:
        symbol = None
    if symbol is not None:
        return int(symbol.address)
    raise ValueError(
        f"could not resolve {identifier!r} to an address; pass a 0x-prefixed "
        f"address or a known function/symbol name for --linear disassembly"
    )


_ARM_MODE_ARCHES = {
    ("arm", False): "armv7", ("arm", True): "armv7eb",
    ("thumb", False): "thumb2", ("thumb", True): "thumb2eb",
}


def _is_classic_arm_or_thumb_arch(name: str) -> bool:
    """True for classic 32-bit ARM/Thumb arch names (armv7/thumb2 and variants);
    False for AArch64 (#600 -- "arm64" starts with "arm" and would otherwise be
    misclassified as classic ARM/Thumb) and for any non-ARM arch.

    #827 item 3 pairs this with ``seam._supports_thumb_pointer_tags`` as the
    #600 drift pair. They share ONE genuinely identical rule -- the
    aarch64/arm64 exclusion, which IS what #600 fixed -- and differ in two ways
    that are not cosmetic:

    * this takes an arch NAME; the seam takes a ``bv``, gates on a 4-byte
      pointer size first, and collects candidate names from both ``arch`` and
      ``platform`` (``name``, ``raw_name`` and ``str()``);
    * this matches by PREFIX (``startswith("arm")``) while the seam matches by
      SUBSTRING over the joined names (``"arm" in joined``).

    The prefix/substring split is the load-bearing one, and it is MEASURED
    against the installed BN API rather than argued:

    * ARCH names containing arm/thumb -- ``armv7``, ``armv7eb``, ``thumb2``,
      ``thumb2eb``. All four start with "arm"/"thumb", so the two rules agree
      on every one: ZERO disagreements.
    * PLATFORM names containing arm/thumb -- eighteen of them, every one shaped
      ``<os>-<arch>`` (``linux-armv7``, ``ios-thumb2``, ``windows-armv7``, ...).
      NOT ONE starts with "arm"/"thumb", so prefix says False and substring
      says True on ALL EIGHTEEN.

    So the rules disagree on every ARM platform name BN can produce, and agree
    on every arch name. That also explains why nobody noticed: the seam joins
    arch AND platform into one string, and the arch name alone already matches
    under either rule -- the substring behaviour is load-bearing only when the
    arch name is absent, unhelpful, or the view is typed by platform. A merge
    onto the prefix rule would therefore look correct on every ordinary ARM
    binary and silently narrow the evidence set on exactly the awkward views
    this predicate exists for: right in testing, quietly weaker in the field.
    Reproduce with ``binaryninja.Platform`` / ``binaryninja.Architecture``
    enumeration; no target needed. Merging them blind is how the #600 drift was
    created in the first place.
    """
    n = (name or "").lower()
    if "aarch64" in n or "arm64" in n:
        return False
    return n.startswith("arm") or n.startswith("thumb")


def _linear_decode_arch(ctx, bv, address: int, mode):
    """The architecture to linearly decode at *address* (#382).

    BN defaults a whole ARM binary to ONE mode (often thumb2), so an ARM-mode
    region under a thumb2 default -- or vice versa -- otherwise decodes in the
    wrong mode. Resolution order:
      1. an explicit ``mode`` (arm/thumb) forces armv7/thumb2 (endianness taken
         from the bv arch), for the stripped/missed case with no function;
      2. else the containing function's own arch (BN's per-function ARM/Thumb
         mode) when known, so a known ARM function decodes as ARM automatically;
      3. else the bv default arch (prior behavior)."""
    bv_arch = getattr(bv, "arch", None)
    if mode is not None:
        # Harden the raw JSON/bridge path: the CLI restricts --mode to arm/thumb,
        # but a direct {"mode":"mips"} request would otherwise KeyError on
        # _ARM_MODE_ARCHES below and surface as an internal error (#382 review).
        if mode not in ("arm", "thumb"):
            raise ValueError(
                f"--mode must be 'arm' or 'thumb' (got {mode!r})"
            )
        cur = str(getattr(bv_arch, "name", "") or "")
        if not _is_classic_arm_or_thumb_arch(cur):
            raise ValueError(
                f"--mode {mode} is only for ARM/Thumb targets (this target is "
                f"{cur or 'unknown'})"
            )
        name = _ARM_MODE_ARCHES[(mode, cur.endswith("eb"))]
        try:
            return bn.Architecture[name]
        except Exception as exc:
            raise ValueError(f"could not load the {name} architecture for --mode {mode}: {exc}")
    try:
        containers = ctx._functions_containing(bv, int(address))
        if containers and getattr(containers[0], "arch", None) is not None:
            return containers[0].arch
    except Exception:
        pass
    return bv_arch


def _function_instruction_starts(bv, func) -> set[int]:
    """The set of recovered instruction START addresses in *func* (#550).

    Walks BN's basic blocks with physical text-decode widths, including the
    instructions covered by a Thumb IT analysis span. Used to detect a start that
    lands INSIDE a function but mid-instruction (which would decode junk that
    looks plausible)."""
    arch = getattr(func, "arch", None)
    starts: set[int] = set()
    try:
        blocks = list(func.basic_blocks)
    except Exception:
        return starts
    for block in blocks:
        try:
            addr = int(block.start)
            end = int(block.end)
        except Exception:
            continue
        while addr < end:
            starts.add(addr)
            _text, length = il_format._disasm_instruction(bv, addr, arch=arch)
            addr += length
    return starts


def _linear_boundary_check(bv, func, address: int) -> dict[str, Any] | None:
    """If *address* is inside *func* but not at a recovered instruction start,
    return a boundary-warning descriptor naming the nearest valid start(s) (#550);
    else None. A start exactly on a boundary is silent."""
    starts = _function_instruction_starts(bv, func)
    if not starts or address in starts:
        return None
    at_or_below = max((s for s in starts if s <= address), default=None)
    above = min((s for s in starts if s > address), default=None)
    return {
        "requested": hex(address),
        "in_function": {"name": str(func.name), "address": hex(int(func.start))},
        "nearest_start_at_or_below": hex(at_or_below) if at_or_below is not None else None,
        "nearest_start_above": hex(above) if above is not None else None,
    }


def _disasm_linear(ctx, bv, identifier, count: int, *, mode=None,
                   snap_to_instruction: bool = False) -> dict[str, Any]:
    """Linear disassembly of *count* units (one physical instruction or one
    undecodable byte) from any MAPPED address, independent of functions (#314).
    The stripped/static lane needs to read the bytes at a suspected missed handler
    -- a dispatch/vtable slot BN left as data -- before deciding whether to
    `function create` it; the
    function-scoped path refuses such addresses outright.

    The decode architecture honors the address's function arch (or an explicit
    ARM/Thumb ``mode``) so an ARM-mode region isn't mis-decoded as Thumb (#382)."""
    address = _resolve_linear_address(ctx, bv, identifier)
    if count <= 0:
        raise ValueError("--linear count must be a positive number of instructions")
    requested_count = int(count)
    count = min(requested_count, _LINEAR_DISASM_MAX)
    # #382 review Finding 2: an ARM code pointer commonly carries bit 0 as the
    # Thumb tag, so a value-set/symbol target can be odd while the instruction
    # lives at addr&~1. Decoding from the odd value starts one byte in. ARM has no
    # odd instruction addresses, so on an ARM/Thumb target an odd resolved address
    # is the Thumb tag -- mask it (and disclose) before decoding. Detect ARM-ness
    # from the bv arch name so this is independent of --mode / function arch, and
    # NEVER mask on non-ARM targets where odd code addresses are legitimate.
    thumb_tag_normalized = None
    bv_arch_name = str(getattr(getattr(bv, "arch", None), "name", "") or "")
    if (address & 1) and _is_classic_arm_or_thumb_arch(bv_arch_name):
        thumb_tag_normalized = address
        address &= ~1
    if not _address_is_mapped(bv, address):
        raise ValueError(
            f"address {hex(address)} is not mapped in this binary; nothing to "
            f"disassemble there"
        )
    # #550: detect a start that lands inside a known function but NOT at a recovered
    # instruction boundary -- linear decoding from mid-instruction produces junk that
    # reads like real control flow. Warn (naming the nearest valid starts) and, with
    # --snap-to-instruction, snap DOWN to the enclosing instruction's start.
    containing_func = None
    try:
        containers = ctx._functions_containing(bv, int(address))
        if containers:
            containing_func = containers[0]
    except Exception:
        containing_func = None
    boundary_warning = None
    snapped_from = None
    if containing_func is not None:
        boundary_warning = _linear_boundary_check(bv, containing_func, address)
        if boundary_warning is not None and snap_to_instruction:
            below = boundary_warning.get("nearest_start_at_or_below")
            if below is not None:
                snapped_from = address
                address = int(below, 16)
                boundary_warning = None  # snapped onto a real boundary; no longer junk
    arch = _linear_decode_arch(ctx, bv, address, mode)
    # Under an explicit --mode the decode is FORCED to that arch: a byte the forced
    # mode can't model must surface as `.byte` (the existing path below), NOT a
    # silent BV-default-arch decode that would contradict the forced-mode note
    # (#382 review). Without --mode, keep the lenient BV fallback (prior behavior).
    strict = mode is not None
    entries: list[dict[str, Any]] = []
    lines: list[str] = []
    addr = int(address)
    for _ in range(count):
        if not _address_is_mapped(bv, addr):
            break
        text, length = il_format._disasm_instruction(bv, addr, arch=arch, strict=strict)
        if not text:
            # Mapped, but no valid instruction decodes here (data / an invalid
            # opcode -- the very thing you point --linear at to confirm). Surface
            # the raw byte and advance one byte so the window keeps moving and
            # the caller sees these bytes aren't code, instead of silently
            # stopping at zero instructions.
            one = bv.read(addr, 1)
            if not one:
                break
            text = f".byte 0x{one.hex()}"
            length = 1
        raw = bv.read(addr, length)
        hex_bytes = raw.hex(" ") if raw else ""
        entries.append({
            "address": hex(addr),
            "bytes": hex_bytes,
            "length": int(length),
            "text": text,
        })
        lines.append(f"{hex(addr)}  {hex_bytes:<16} {text}")
        addr += length
    # Recompute containment at the FINAL (possibly snapped) start address.
    in_function = None
    if containing_func is not None:
        in_function = {
            "name": containing_func.name,
            "address": hex(int(containing_func.start)),
        }
    else:
        try:
            containers = ctx._functions_containing(bv, int(address))
            if containers:
                fn = containers[0]
                in_function = {"name": fn.name, "address": hex(int(fn.start))}
        except Exception:
            in_function = None
    note = (
        f"linear disassembly of {len(entries)} unit"
        f"{'' if len(entries) == 1 else 's'} from {hex(address)} "
        # #550: name the ordering explicitly -- this is address-linear (byte order),
        # NOT the basic-block/graph order a function-scoped `disasm <fn>` renders.
        f"(one physical instruction or one undecodable byte (.byte) per unit; "
        f"address-linear order, not function-bounded)"
    )
    if thumb_tag_normalized is not None:
        note += (
            f"; normalized a Thumb function-pointer tag (bit 0) "
            f"{hex(thumb_tag_normalized)} -> {hex(address)}"
        )
    if snapped_from is not None:
        note += (
            f"; snapped {hex(snapped_from)} back to the enclosing instruction "
            f"start {hex(address)} (--snap-to-instruction)"
        )
    if in_function is not None:
        note += f"; this address is inside {in_function['name']} @ {in_function['address']}"
    if boundary_warning is not None:
        # #550: the start is inside a function but mid-instruction -- the decode below
        # is very likely junk. Name the nearest valid starts so the caller can re-run.
        below = boundary_warning.get("nearest_start_at_or_below")
        above = boundary_warning.get("nearest_start_above")
        nearest = ", ".join(x for x in (below, above) if x) or "unknown"
        note += (
            f"; WARNING: {boundary_warning['requested']} is NOT a recovered instruction "
            f"boundary inside {boundary_warning['in_function']['name']} -- linear decode "
            f"from here may be junk. Nearest valid start(s): {nearest}. Re-run at a valid "
            f"start or add --snap-to-instruction"
        )
    arch_name = str(getattr(arch, "name", "") or "")
    if arch_name:
        # #382: disclose the decode mode so an ARM/Thumb decode isn't silently
        # trusted in the wrong mode (and so --mode's effect is visible).
        forced = " (forced via --mode)" if mode is not None else ""
        note += f"; decoded as {arch_name}{forced}"
    if requested_count > _LINEAR_DISASM_MAX:
        note += f"; capped at {_LINEAR_DISASM_MAX} (requested {requested_count})"
    return {
        "linear": True,
        "order": "address-linear",
        "function": in_function,
        "address": hex(address),
        "decode_arch": arch_name or None,
        "requested_count": requested_count,
        "capped": requested_count > _LINEAR_DISASM_MAX,
        "instruction_count": len(entries),
        "instructions": entries,
        "text": "\n".join(lines),
        "boundary_warning": boundary_warning,
        "snapped_from": hex(snapped_from) if snapped_from is not None else None,
        "note": note,
    }


def _address_is_mapped(bv, address: int) -> bool:
    """True if *address* looks mapped, with a read-probe fallback.

    #827 item 3 lists this beside ``_shared._require_mapped_address`` as a
    duplicated mapped check. They are NOT the same rule and are deliberately
    NOT merged: they disagree on the INDETERMINATE case, which is the only case
    that matters here.

    ``_require_mapped_address`` returns quietly when the view cannot answer --
    its contract is that a caller must not reject on an indeterminate result,
    so a mapped address with zero refs stays a clean exit 0 (#374). This one
    falls back to a 1-byte read and, if that also fails, answers **False** --
    it treats "cannot tell" as "not mapped". Collapsing the two would silently
    flip one of those policies: give this the permissive contract and a
    decompile at a genuinely bad address proceeds; give the other this one's
    contract and #374's clean-zero case starts erroring on any view without
    ``is_valid_offset``.

    Whether the read-probe fallback SHOULD answer False is a fair question, but
    changing it is a behaviour change neither #827 nor #600 asked for, so the
    divergence is named here rather than resolved silently.
    """
    try:
        if hasattr(bv, "is_valid_offset"):
            return bool(bv.is_valid_offset(int(address)))
    except Exception:
        pass
    try:
        return bool(bv.read(int(address), 1))
    except Exception:
        return False


def _structured_il(ctx, selector, identifier, *, view: str = "mlil", ssa: bool = True):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    il = il_format._il_function_for(func, view, ssa)
    instructions = []
    try:
        items = list(il.instructions)
    except Exception:
        items = []
    for ins in items:
        opn = il_format._il_op_name(ins)
        instructions.append({
            "il_index": int(getattr(ins, "instr_index", -1)),
            "address": hex(int(getattr(ins, "address", func.start))),
            "op": opn,
            "text": str(ins),
            "vars_read": [il_format._ssa_var_entry(v) for v in (getattr(ins, "vars_read", None) or [])],
            "vars_written": [il_format._ssa_var_entry(v) for v in (getattr(ins, "vars_written", None) or [])],
            "operands_summary": [str(o) for o in (getattr(ins, "operands", None) or [])],
            "is_call": "CALL" in opn,
        })
    result = {
        "function": {"name": func.name, "address": hex(func.start)},
        "view": view,
        "ssa": ssa,
        "instructions": instructions,
    }
    _annotate_containment(ctx, result, identifier, func)
    return result


def _defuse(ctx, selector, identifier, var_selector: str):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    il = il_format._il_function_for(func, "mlil", True)
    ssa_var, other_versions = il_format._resolve_ssa_variable(func, il, var_selector)

    try:
        definition = il.get_ssa_var_definition(ssa_var)
    except Exception:
        definition = None
    try:
        uses = list(il.get_ssa_var_uses(ssa_var) or [])
    except Exception:
        uses = []

    def _ref(ins):
        if ins is None:
            return None
        return {
            "il_index": int(getattr(ins, "instr_index", -1)),
            "address": hex(int(getattr(ins, "address", func.start))),
            "op": il_format._il_op_name(ins),
            "text": str(ins),
        }

    is_phi = definition is not None and "PHI" in il_format._il_op_name(definition)
    phi_sources = []
    if is_phi:
        for s in (getattr(definition, "src", None) or []):
            phi_sources.append(il_format._ssa_var_entry(s))

    result = {
        "function": {"name": func.name, "address": hex(func.start)},
        "variable": il_format._ssa_var_entry(ssa_var),
        "definition": _ref(definition),
        "uses": [_ref(u) for u in uses],
        "is_phi": is_phi,
        "phi_sources": phi_sources,
        "other_versions": other_versions or [],
        "hints": _call_model_truncation_hints(
            ctx, bv, func, il, [u for u in uses if u is not None]),
    }
    _annotate_containment(ctx, result, identifier, func)
    return result


def _call_model_truncation_hints(ctx, bv, func, il, uses) -> list[str]:
    """The #489 call-model-truncation disclosure for THIS function's calls (#797).

    A def-use read of a variable that feeds an under-recovered call used to show
    the argument set-up as an ordinary use and say nothing about it: BN clamps a
    direct call's MLIL parameters to the callee's recovered arity, so on a
    register ABI a variadic callee auto-typed fixed-arity leaves its STACK-passed
    arguments behind as `[sp+N].d = <var>` stores that nothing in the function
    reads back, and the store for the defused variable sits in `uses` looking
    like any other. `trace` already discloses exactly that state with the
    reviewed #489 note; this reads the SAME helper over the same calls AND
    resolves the callee name through the same two steps the sibling uses, so
    the two ops cannot disagree about whether a call's model was truncated (one
    gate, one wording, one remedy).

    Scoped to the calls this variable's USES feed, because that is what the
    disclosure CLAIMS to be: an explanation of a row in this listing. Built
    over every call in the function it was a different statement, and a wrong
    one: a variable with no uses at all printed a truncation note above
    `uses (0):`, and a use downstream of the call got a caveat about argument
    set-up it has no part in. A use belongs to a call's outgoing-argument run
    when it IS the call (the variable is a recovered parameter) or it sits
    between the previous call and this one -- which is exactly the region
    `_call_model_truncation_note` reads its dropped stack-arg stores from.

    The DEFINITION is deliberately not in that set (#797 round-4 review).
    Scoping over `[definition] + uses` re-opened the whole defect through the
    ordinary case: a value computed in this function has a definition ahead of
    the call, which marked the call fed and put the note back above
    `uses (0):`. A definition is where the value is PRODUCED -- it is no
    evidence that the value reaches any call -- so the uses are the scope, and
    a variable passed to the call shows up in them as the argument-set-up
    store (or as the call itself when it is a recovered parameter).

    Prints nothing for a call the helper is silent about -- unknown
    calling-convention arity, a callee that is known fixed-arity, no outgoing
    stack-arg run, or no caller-passed format string -- which is the
    no-false-positive direction that note was reviewed for. Returns [] (never
    raises) on any BN-API shortfall."""
    from . import read_taint_slice as _ts  # local: this module's import list is documented as il_format/vars/taint_engine/_shared

    hints: list[str] = []
    try:
        instructions = list(il.instructions)
    except Exception:
        return hints
    use_indexes = {int(getattr(use, "instr_index", -1)) for use in uses}
    feeding = False
    for ins in instructions:
        index = int(getattr(ins, "instr_index", -1))
        if "CALL" not in il_format._il_op_name(ins):
            # A use before the next call is part of that call's outgoing-argument
            # run; anything after the last call feeds nothing.
            feeding = feeding or index in use_indexes
            continue
        # Only a call THIS variable reaches: it is the call itself, or one of
        # the uses sits in the run of stores between the previous call and it.
        relevant, feeding = (feeding or index in use_indexes), False
        if not relevant:
            continue
        # The CALL SITE's own address: `_call_model_truncation_note` looks the
        # instruction up in the LLIL block by address to find the outgoing
        # stack-arg stores feeding it, so this is the address it wants -- not the
        # resolved callee's.
        call_addr = int(getattr(ins, "address", func.start))
        # The callee name, resolved EXACTLY as `trace` resolves it: the modeled
        # name first, then the resolved callee function -- both of which follow
        # thunks. Resolving it here without following thunks was the one place
        # the two ops could still disagree, and the disagreement was live in
        # both directions: the shared note's known-fixed-arity gate keys on this
        # name, so a PLT stub / veneer in front of `strlen` presented as
        # `j_strlen`, matched no denylist entry, and fired the exact residual
        # false positive that denylist exists to close -- on a call `trace` is
        # silent about. And when the note does fire, its `proto set <name>`
        # remedy would have named the veneer, whose prototype is not the one
        # that truncated the model.
        callee_name = _ts._modeled_callee_name(bv, ins)
        if not callee_name:
            try:
                callee_name = str(getattr(_ts._resolve_callee(ctx, bv, ins),
                                          "name", "") or "") or None
            except Exception:
                callee_name = None
        note = _ts._call_model_truncation_note(
            bv, func, ins, call_addr,
            list(getattr(ins, "params", None) or []),
            callee_name,
        )
        if note is not None:
            hints.append(f"call {hex(call_addr)}: {note}")
    return hints


def _pvs_targets(ctx, bv, pvs) -> list[dict[str, Any]]:
    if pvs is None:
        return []
    type_name = getattr(getattr(pvs, "type", None), "name", "") or ""
    addrs: list[int] = []
    if type_name in {"ConstantValue", "ConstantPointerValue", "ImportedAddressValue", "ExternalPointerValue"}:
        value = getattr(pvs, "value", None)
        if value is not None:
            try:
                addrs.append(int(value))
            except Exception:
                pass
    elif type_name == "InSetOfValues":
        for v in (getattr(pvs, "values", None) or []):
            try:
                addrs.append(int(v))
            except Exception:
                pass
    elif type_name == "LookupTableValue":
        mapping = getattr(pvs, "mapping", None)
        if isinstance(mapping, dict):
            for v in mapping.values():
                try:
                    addrs.append(int(v))
                except Exception:
                    pass
        else:
            for entry in (getattr(pvs, "table", None) or []):
                target = getattr(entry, "to", None)
                if target is not None:
                    try:
                        addrs.append(int(target))
                    except Exception:
                        pass
    targets = []
    for addr in addrs:
        # function_at normalizes the Thumb low bit so an odd value-set target
        # resolves to its function (#89 Problem B). The raw address is kept
        # in the reported "address".
        fn = _taint.function_at(bv, addr)
        name = None
        if fn is not None:
            name = str(getattr(fn, "name", None))
        else:
            sym = bv.get_symbol_at(addr) if hasattr(bv, "get_symbol_at") else None
            if sym is None and (addr & 1) and hasattr(bv, "get_symbol_at"):
                sym = bv.get_symbol_at(addr & ~1)
            name = str(getattr(sym, "name", None)) if sym is not None else None
        targets.append({"address": hex(addr), "name": name})
    return targets


def _resolved_calls(ctx, selector, identifier, *, direction: str = "both", resolve_indirect: bool = True):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    # `kind` lets a consumer of the {kind, ...} family identify a callgraph read
    # (it is a composite callees+callers structure, not a flat items list) (#371.2).
    result: dict[str, Any] = {
        "kind": "callgraph",
        "function": {"name": func.name, "address": hex(func.start)},
    }

    if direction in ("callees", "both"):
        il = il_format._il_function_for(func, "mlil", True)
        callees = []
        try:
            items = list(il.instructions)
        except Exception:
            items = []
        for ins in items:
            opn = il_format._il_op_name(ins)
            if "CALL" not in opn and "TAILCALL" not in opn:
                continue
            # resolve_call_target (no thunk following) resolves MLIL_IMPORT
            # calls and Thumb-tagged targets that const_target/exact-address
            # lookup miss (#89). Constant direct calls resolve identically.
            resolved = _taint.resolve_call_target(bv, ins, follow_thunks=False)
            target = resolved.address
            row = {
                "call_addr": hex(int(getattr(ins, "address", func.start))),
                "il_index": int(getattr(ins, "instr_index", -1)),
            }
            if target is not None:
                fn = resolved.function or (bv.get_function_at(target) if hasattr(bv, "get_function_at") else None)
                name = str(fn.name) if fn is not None else None
                if name is None:
                    sym = bv.get_symbol_at(target) if hasattr(bv, "get_symbol_at") else None
                    name = str(sym.name) if sym is not None else None
                row.update({"kind": "direct", "target": {"address": hex(target), "name": name}})
            else:
                row.update({"kind": "indirect", "dest_expr": str(getattr(ins, "dest", ""))})
                if resolve_indirect:
                    pvs = getattr(getattr(ins, "dest", None), "possible_values", None)
                    resolved = _pvs_targets(ctx, bv, pvs)
                    row["resolved"] = resolved
                    type_name = getattr(getattr(pvs, "type", None), "name", "") or "none"
                    row["resolution"] = "possible_values" if resolved else "unresolved"
                    row["resolution_detail"] = type_name
            callees.append(row)
        result["callees"] = callees

    if direction in ("callers", "both"):
        callers = []
        seen: set[int] = set()
        for site in (list(getattr(func, "caller_sites", None) or [])):
            addr = int(getattr(site, "address", 0))
            fn = getattr(site, "function", None)
            if fn is None:
                functions = ctx._functions_containing(bv, addr)
                fn = functions[0] if functions else None
            caller = (
                {"address": hex(int(fn.start)), "name": str(fn.name)} if fn is not None else None
            )
            callers.append({"call_addr": hex(addr), "caller": caller})
        if not callers:
            for fn in (list(getattr(func, "callers", None) or [])):
                marker = int(getattr(fn, "start", 0))
                if marker in seen:
                    continue
                seen.add(marker)
                callers.append({"caller": {"address": hex(marker), "name": str(fn.name)}})
        result["callers"] = callers

    _annotate_containment(ctx, result, identifier, func)
    return result


def _possible_values(ctx, selector, identifier, at):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    address = _parse_address(at)
    il = il_format._il_function_for(func, "mlil", True)
    target_ins = None
    try:
        for ins in list(il.instructions):
            if int(getattr(ins, "address", -1)) == address:
                target_ins = ins
                # SSA inserts PHIs at the following instruction's address.
                # Keep a PHI only if there is no actual operation at that site.
                if il_format._il_op_name(ins) not in ("MLIL_VAR_PHI", "MLIL_MEM_PHI"):
                    break
    except Exception:
        target_ins = None
    if target_ins is None:
        # No MLIL instruction begins at --at: returning a success dict with
        # expression:None makes a bogus address look like a real instruction
        # with an unknown value-set. Refuse instead of emitting a false read (#526).
        raise OperationFailure(
            "no_instruction",
            f"No MLIL instruction at {hex(address)} in {func.name}: --at must name the "
            f"address of an instruction within the function (use `bn function il "
            f"--view mlil` to list valid instruction addresses).",
            requested={"identifier": str(identifier), "at": hex(address)},
        )
    instr_pvs = getattr(target_ins, "possible_values", None) if target_ins is not None else None
    src_expr = getattr(target_ins, "src", None) if target_ins is not None else None
    src_pvs = getattr(src_expr, "possible_values", None) if src_expr is not None else None
    # BN leaves a SET_VAR/STORE *instruction* value-set undetermined while the
    # SOURCE expression (the value being assigned) carries the real value-set
    # -- const / range / lookup-table. Report the source's value-set for an
    # assignment so values surface as the help promises, instead of always
    # printing UndeterminedValue; fall back to the instruction-level set when
    # there is no source or only the instruction-level set is determined (#52).
    if il_format._pvs_determined(src_pvs) and not il_format._pvs_determined(instr_pvs):
        chosen, basis = src_pvs, "source_expression"
    elif il_format._pvs_determined(instr_pvs):
        chosen, basis = instr_pvs, "instruction"
    elif src_pvs is not None:
        chosen, basis = src_pvs, "source_expression"
    else:
        chosen, basis = instr_pvs, "instruction"
    result = {
        "function": {"name": func.name, "address": hex(func.start)},
        "at": hex(address),
        "expression": str(target_ins) if target_ins is not None else None,
        "value_basis": basis,
        "source_expression": str(src_expr) if src_expr is not None else None,
        "possible_values": il_format._serialize_pvs(chosen),
    }
    # The destination's addresses are not possible instruction/return values.
    # Expose the operand separately, even when its analysis is undetermined,
    # without changing the instruction/source selection above (#947).
    opn = il_format._il_op_name(target_ins)
    if opn.startswith(("MLIL_CALL", "MLIL_TAILCALL")):
        dest_expr = getattr(target_ins, "dest", None)
        result["call_destination"] = {
            "expression": str(dest_expr) if dest_expr is not None else None,
            "value_basis": "destination_expression",
            "possible_values": il_format._serialize_pvs(
                getattr(dest_expr, "possible_values", None) if dest_expr is not None else None
            ),
        }
    _annotate_containment(ctx, result, identifier, func)
    return result
