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


def _forced_data_region_warning(bv, func) -> str | None:
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
    return (
        f"{func.name}: forced analysis decoded {size} bytes with 0 inbound code refs. "
        "BN sometimes makes a function on a DATA region (string/pointer table or packed "
        "data) that grows under forced decode; if the body reads as tables/strings/garbage "
        "rather than coherent code, treat this decode as suspect and verify before acting on it."
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


def _thunk_veneer_warning(ctx, bv, func) -> str | None:
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
            return (f"thunk/veneer: the branch target lands inside {dest}, not at its "
                    f"entry -- a mid-function destination is not a trampoline entry "
                    f"point, so this is not confirmed as a PLT/GOT thunk.")
        return (f"thunk/veneer -> {dest}: this is a PLT/GOT trampoline (a jump to "
                f"the real body), not a self-recursive function.")
    # #704 round 3: no target was resolved here -- the candidate was flagged
    # either because the function merely starts in a PLT/import section, or
    # because BN's pseudo-C rendered a bare `/* tailcall */` with no branch
    # destination this detector could follow. Asserting "PLT/GOT trampoline
    # ... the apparent self-call is the resolved target" claimed a resolution
    # the tool never made. State only the reason the candidate was flagged;
    # make no claim about what the unresolved target actually is.
    reason = summary.get("reason") or "no branch target could be resolved"
    return (f"thunk/veneer: {reason}; the branch target was not resolved, so "
            f"this may be a trampoline/veneer but that is not confirmed.")


# Consume literals and block comments before looking for a line-comment span,
# so a quoted URL or a `//` inside `/* ... */` cannot start a redaction.
_COMMENT_SPAN_RE = re.compile(
    r""""(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|/\*.*?(?:\*/|\Z)|//(?P<body>[^\r\n]*)""",
    re.DOTALL,
)


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

    redacted = _COMMENT_SPAN_RE.sub(redact, text)
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


def _flattened_guard_warning(bv, func) -> str | None:
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
        for block in list(func.basic_blocks):
            addr = int(block.start)
            end = int(block.end)
            while addr < end:
                text, length = il_format._disasm_instruction(bv, addr, arch=arch)
                if not length or length <= 0:
                    break
                if text:
                    mnemonic = str(text).split(None, 1)[0].strip().lower()
                    if mnemonic in _FLATTENED_GUARD_MNEMONICS and mnemonic not in seen:
                        seen.append(mnemonic)
                addr += length
        if not seen:
            return None
    except Exception:  # noqa: BLE001 - no listing is not evidence of no select
        return None
    names = ", ".join(sorted(seen))
    return (
        f"branch-free conditional(s) in this function ({names}): Pseudo-C "
        f"renders these as ternaries, folds them into one compound condition, "
        f"or decomposes them into raw flag arithmetic (`v`/`z`/`n` locals) -- "
        f"a two-sided range guard can come out as none of the three shapes a "
        f"reader is looking for, so a bound that IS enforced can read as one "
        f"that never fires. Confirm it in `bn disasm` before concluding on an "
        f"off-by-one, truncation or missing check."
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


def _struct_layout_warnings(bv: Any, func: Any) -> list[str]:
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
        warnings.append(
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
            "No type was applied."
        )
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


def _argument_setup_count(arch, llil, call, arg_regs: list[str]) -> tuple[int, list[str]]:
    """How many leading argument registers the call's own block writes for it.

    Walks backwards from *call* to the previous call (or the block start) and
    keeps a write to an argument register only when nothing reads that register
    between the write and the call: such a value has no consumer but the callee,
    since every argument register is caller-saved. Two writes are never counted:
    one that is read before the call (a temporary), and GCC's stack-protector
    scrub -- `ldr xN, [guard]; str xN, [sp, ..]; mov xN, #0` -- which zeroes a
    register it just stored, and is where the issue's invented `__stack_chk_guard`
    argument came from (#955). Returns (count, written register names): the count
    is the highest written argument register's position + 1, because the ABI
    passes every lower register too.
    """
    block = getattr(call, "il_basic_block", None)
    index = int(getattr(call, "instr_index", -1))
    first = int(getattr(block, "start", index))
    read_after: set[str] = set()
    written: dict[str, Any] = {}
    preceding = [llil[i] for i in range(first, index)]
    for pos in range(len(preceding) - 1, -1, -1):
        ins = preceding[pos]
        op = il_format._il_op_name(ins)
        if "CALL" in op or op == "LLIL_SYSCALL":
            break
        if op == "LLIL_SET_REG":
            name = _full_reg(arch, str(getattr(getattr(ins, "dest", None), "name", "")))
            if name in arg_regs and name not in written and name not in read_after:
                src = getattr(ins, "src", None)
                scrub = (
                    il_format._llil_constant_value(src) == 0
                    and any(
                        il_format._il_op_name(prior) == "LLIL_STORE"
                        and il_format._il_op_name(getattr(prior, "src", None)) == "LLIL_REG"
                        and _full_reg(arch, str(getattr(prior.src.src, "name", ""))) == name
                        for prior in preceding[:pos]
                    )
                )
                if not scrub:
                    written[name] = ins
            _llil_regs_read(arch, getattr(ins, "src", None), read_after)
        else:
            _llil_regs_read(arch, ins, read_after)
    if not written:
        return 0, []
    names = sorted(written, key=arg_regs.index)
    return arg_regs.index(names[-1]) + 1, names


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


def _call_arity_warning(bv, func) -> str | None:
    """Name calls whose rendered argument list is shorter than the call site's
    (#955), or -- for a mangled callee -- longer than its signature.

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
    for a variadic one. Returns None when every call agrees, and never raises.
    """
    try:
        llil = getattr(func, "llil", None) or getattr(func, "low_level_il", None)
        instructions = list(llil.instructions) if llil is not None else []
        cc = getattr(func, "calling_convention", None)
        arg_regs = [_full_reg(bv.arch, str(r)) for r in (getattr(cc, "int_arg_regs", None) or [])]
    except Exception:  # noqa: BLE001 - no IL means nothing to compare
        return None
    findings: list[str] = []
    seen: set[int] = set()
    for call in instructions:
        try:
            if il_format._il_op_name(call) not in ("LLIL_CALL", "LLIL_TAILCALL"):
                continue
            target = il_format._llil_constant_value(getattr(call, "dest", None))
            address = int(getattr(call, "address", 0))
            if target is None or address in seen:
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
            no_body = kind in _IMPORT_SYMBOL_KINDS or bool(getattr(callee, "is_thunk", False))
            signature = _demangled_arity(bv, raw_name)
            setup, regs = _argument_setup_count(bv.arch, llil, call, arg_regs) if arg_regs else (0, [])
        except Exception:  # noqa: BLE001 - one odd call must not hide the rest
            continue
        label = _callee_label(symbol, callee, target)
        setup_text = f"the call site writes {', '.join(regs)} just before it"
        if signature is not None:
            # The demangled signature is the arity whenever there is one: the
            # register witness only corroborates it, never overrides it.
            total, explicit, variadic = signature
            if rendered > total and not variadic:
                seen.add(address)
                findings.append(f"{hex(address)} {label} renders {rendered} arguments "
                                f"but the demangled signature takes only {total}")
                continue
            if rendered >= total or not (explicit > rendered or setup > rendered):
                continue
            expected = total
            evidence = [f"the demangled signature takes {total}"
                        + (" with `this`" if total != explicit else "")]
            if setup > rendered:
                evidence.append(setup_text)
        elif no_body and setup > rendered:
            expected, evidence = setup, [setup_text]
        else:
            continue
        seen.add(address)
        findings.append(f"{hex(address)} {label} renders {rendered} of {expected} "
                        f"arguments ({'; '.join(evidence)})")
    if not findings:
        return None
    listed = findings[:_ARITY_LISTED_MAX]
    more = len(findings) - len(listed)
    return (
        f"call arity: {len(findings)} call(s) render a different argument count than "
        f"the call site passes: {'; '.join(listed)}"
        + (f"; and {more} more" if more > 0 else "")
        + ". The callee's prototype (often from a type library) has the wrong arity, so "
        "the decompiled call silently drops or invents arguments. Read the real "
        f"arguments with `bn callsites <callee address> --within {hex(int(func.start))} "
        "--context 4`; fix the callee with `bn proto set`."
    )


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


def _runs_into_next(bv, func, block) -> bool:
    """True when *block* (of *func*) ends by executing into the next address.

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
        return False
    if padded and "CALL" in _llil_op_at(func, addresses[-1]):
        return False
    return _falls_through(bv, addresses[-1])


def _fallthrough_successor(bv, func):
    """The function whose entry *func*'s body runs straight into, or None."""
    for block in list(getattr(func, "basic_blocks", None) or []):
        other = bv.get_function_at(int(block.end))
        if other is None or int(other.start) == int(func.start):
            continue
        if _runs_into_next(bv, func, block):
            return other
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
        if _runs_into_next(bv, owner, block):
            return owner
    return None


def _split_body_notes(bv, func) -> list[str]:
    """Disclose a function body BN split at a shared tail (#966).

    When another function branches into the tail of this one, BN can make the
    tail a function of its own and end this one early. The damage is quiet:
    the head's prototype gains "parameters" in callee-saved registers and the
    frame pointer (values the tail only restores), and every caller passes
    stray arguments for them. Two read-only signals name it: parameters in
    registers no convention passes arguments in, and a body that runs
    straight into another function's entry (or is run into by one)."""
    notes: list[str] = []
    try:
        params = _callee_saved_parameters(bv, func)
        successor = _fallthrough_successor(bv, func)
        predecessor = _fallthrough_predecessor(bv, func)
    except Exception:  # noqa: BLE001 - a read-only nudge never sinks the read
        return notes
    if successor is not None:
        notes.append(
            f"split body: this function runs straight into {_function_label(successor)} "
            "without a branch or return -- BN likely split it at a shared tail, so the "
            "rest of this function's code is listed under that one. Read both "
            "(`bn disasm`) before trusting either decompile."
        )
    if predecessor is not None:
        notes.append(
            f"split body: {_function_label(predecessor)} runs straight into this "
            "function's entry -- this may be the shared tail of that function, not a "
            "function of its own; its parameters are likely that function's saved registers."
        )
    if params:
        notes.append(
            f"prototype takes parameters in callee-saved/link registers ({', '.join(params)}): "
            "no calling convention passes arguments there, so BN is reading a value the "
            "body only saves or restores -- typically because it split the body at a "
            "shared tail, or a path reads the register before setting it. Callers render "
            "stray arguments for them; drop them with `bn proto set`."
        )
    return notes


def _function_label(func) -> str:
    return f"{il_format._display_name(func)} @ {hex(int(func.start))}"


def _callee_split_notes(bv, func) -> list[str]:
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
        if il_format._il_op_name(call) not in ("LLIL_CALL", "LLIL_TAILCALL"):
            continue
        target = il_format._llil_constant_value(getattr(call, "dest", None))
        if target is None or target in checked:
            continue
        checked.add(target)
        callee = bv.get_function_at(target)
        if callee is None:
            continue
        params = _callee_saved_parameters(bv, callee)
        if params:
            notes.append(
                f"callee {_function_label(callee)} takes parameters in callee-saved/link "
                f"registers ({', '.join(params)}), so the arguments rendered for them here "
                "are not real arguments -- its body is likely split at a shared tail "
                f"(`bn decompile {hex(target)}` names the split)."
            )
    return notes


_SHORT_STRING_MAX = 64
_STRING_NOTE_LISTED_MAX = 8


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


def _undefined_c_string(bv, address: int) -> str | None:
    """The printable NUL-terminated string at *address*, when BN defined none."""
    if bv.get_string_at(address) is not None:
        return None
    var = bv.get_data_var_at(address)
    if var is not None and "char" in str(getattr(var, "type", "")):
        return None
    if not _read_only_data(bv, address):
        return None
    data = bytes(bv.read(address, _SHORT_STRING_MAX) or b"")
    end = data.find(b"\0")
    if end <= 0:
        return None
    body = data[:end]
    if not all(0x20 <= c < 0x7f or c in (0x09, 0x0a, 0x0d) for c in body):
        return None
    return body.decode("ascii")


def _raw_string_pointer_note(bv, func, text: str) -> str | None:
    """Name call arguments rendered as a raw address that point at a short C
    string BN never defined (#970).

    BN's string scan has a minimum length, so `" -n"` in `s << "cat " << f <<
    " -n"` stays an untyped address: decompile prints `0x402ae0` and `strings`
    lists nothing. Only a constant argument whose hex spelling actually appears
    in the rendered text is reported, and only into read-only data, so a
    pointer BN already renders as a literal or a writable buffer never is.
    Read-only: defining the string is the caller's choice (`bn data retype`)."""
    try:
        mlil = getattr(func, "mlil", None) or getattr(func, "medium_level_il", None)
        instructions = list(mlil.instructions) if mlil is not None else []
    except Exception:  # noqa: BLE001
        return None
    found: dict[int, tuple[str, int]] = {}
    for ins in instructions:
        if "CALL" not in il_format._il_op_name(ins):
            continue
        for param in list(getattr(ins, "params", None) or []):
            if il_format._il_op_name(param) not in ("MLIL_CONST_PTR", "MLIL_CONST"):
                continue
            try:
                address = int(param.constant)
                if address in found or not re.search(rf"\b{hex(address)}\b", text):
                    continue
                value = _undefined_c_string(bv, address)
            except Exception:  # noqa: BLE001 - one odd operand must not hide the rest
                continue
            if value is not None:
                found[address] = (value, int(getattr(ins, "address", 0)))
    if not found:
        return None
    rows = [f"{hex(a)} = {_c_quote(v)} (call at {hex(c)})"
            for a, (v, c) in list(found.items())[:_STRING_NOTE_LISTED_MAX]]
    more = len(found) - len(rows)
    first, (value, _) = next(iter(found.items()))
    return (
        f"raw string pointer(s): {len(found)} call argument(s) render as an address "
        f"that holds a NUL-terminated string BN did not define (so `bn strings` does not "
        f"list it either): {'; '.join(rows)}" + (f"; and {more} more" if more > 0 else "")
        + f". To render it inline, define it: `bn data retype {hex(first)} "
        f"'char[{len(value) + 1}]'`."
    )


def _c_quote(value: str) -> str:
    escaped = (value.replace("\\", "\\\\").replace('"', '\\"')
               .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))
    return f'"{escaped}"'


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
    text = il_format._decompile_text(bv, func, addresses=addresses)
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

    warnings = il_format._render_warnings(text)
    stub = il_format._analysis_stub_warning(func, text, forced=forced)
    if stub:
        warnings.append(stub)
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
    arity_warn = _call_arity_warning(bv, func)
    if arity_warn:
        warnings.append(arity_warn)
    warnings.extend(_split_body_notes(bv, func))
    warnings.extend(_callee_split_notes(bv, func))
    string_note = _raw_string_pointer_note(bv, func, text)
    if string_note:
        warnings.append(string_note)
    result = {
        "function": {"name": func.name, "address": hex(func.start)},
        "text": text,
        "comments": comments if include_annotations else {},
        "annotation_summary": annotation_summary,
        "warnings": warnings,
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
        },
        **il_format._function_metadata(func),
    }
    note = _return_width_inference_note(bv, func)
    if note:
        result["return_width_note"] = note
    split_notes = _split_body_notes(bv, func)
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
    # simply ended. Say where execution actually continues.
    try:
        successor = _fallthrough_successor(bv, func)
    except Exception:  # noqa: BLE001 - a read-only nudge never sinks the read
        successor = None
    if successor is not None:
        result["continues_into"] = {
            "name": successor.name,
            "address": hex(int(successor.start)),
            "note": (
                f"execution falls through into {_function_label(successor)}: BN ended "
                "this function at another function's entry (likely a shared tail split); "
                f"continue with `bn disasm {hex(int(successor.start))}`"
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
    _annotate_containment(ctx, result, identifier, func)
    return result
