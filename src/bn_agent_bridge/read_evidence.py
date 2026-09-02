"""Raw-ABI call evidence, pointer tables, message lensing, .init_array walking.

The evidence/pointer-table cluster that used to live on ``BinaryNinjaBridge``
moves here as module-level free functions, each taking the ``BridgeContext``
seam (``ctx``) in place of ``self``. ``BinaryNinjaBridge`` keeps a thin
delegating shim for every name the test suite / op binders reference
(``_function_evidence``, ``_pointer_table``, ``_pointer_table_for_view``,
``_message_lens``, ``_init_arrays``, ...).

Outbound calls resolve through:
  * ``ctx`` -- resolution / address-context / ABI helpers relocated to the seam
    (``_resolve_view``, ``_find_function``, ``_address_context``,
    ``_pointer_size``, ``_read_pointer_value``, ``_normalize_code_pointer``,
    ``_sections_at``);
  * ``il_format`` -- the state-free IL/disasm helpers used by the call scan
    (``_iter_llil_instructions``, ``_il_op_name``, ``_structured_disasm_entries``,
    ``_disasm_entry``, ``_hlil_call_roots``, ``_hlil_statement_text``,
    ``_hlil_pre_branch_condition``, ``_decompile_text``, ``_function_metadata``,
    ``_render_warnings``, ``_llil_constant_value``);
  * ``read_xrefs`` -- ``_xrefs_to_address`` (used by message lensing);
  * ``_shared`` -- module-free helpers (``_parse_address``, ``_validate_count``,
    ``OperationFailure``).

Import direction is one-way: this module imports ``il_format``, ``read_xrefs``,
and ``_shared`` (plus stdlib + binaryninja). It NEVER imports ``bridge`` or
``seam`` -- those import THIS module one-way (design spec §3.2). It imports
``read_xrefs`` but ``read_xrefs`` NEVER imports this module.
"""
from __future__ import annotations

import re
from typing import Any

try:
    import binaryninja as bn  # noqa: F401  (kept for parity with sibling read_* modules)
except ModuleNotFoundError:  # importable without the Binary Ninja runtime (tests, tooling)
    bn = None  # type: ignore[assignment]

from . import il_format
from . import read_xrefs
from ._shared import OperationFailure, _parse_address, _validate_count, is_imported_function


def _call_destination_value(ctx, insn) -> int | None:
    return il_format._llil_constant_value(getattr(insn, "dest", None))


def _target_entry_for_call(ctx, bv, value: int | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return ctx._normalize_code_pointer(bv, value)


def _true_mlil(insn):
    """The LLIL call instruction's TRUE per-instruction MLIL (#661).

    ``insn.mapped_medium_level_il`` is a COALESCED form that renders the whole
    caller-register call site as ``call(dest, arg1, arg2, ...)`` -- naming the
    CALLER's ABI registers, not the callee's actual operands. ``insn.mlil`` is
    the direct per-instruction MLIL and renders the real call expression (e.g.
    ``0x401156(rdi, 3)``). Prefer the direct accessor; fall back to the mapped
    form only when it is unavailable (older BN builds / degenerate cases), and
    to None when neither exists -- callers must never fabricate an `mlil` line.

    Both accessors are wrapped rather than accessed via a bare ``getattr``
    default: BN's real ``.mlil``/``.mapped_medium_level_il`` properties can
    themselves raise (e.g. an internal ``assert result is not None, "MLIL not
    present"`` when the underlying analysis has no MLIL for this instruction)
    rather than raise ``AttributeError``, which ``getattr``'s default only
    suppresses. This is pre-existing fragility symmetric across BOTH
    accessors (not a regression introduced by preferring ``.mlil``); wrapping
    each independently matches the pattern already used elsewhere in this
    module for raising `.mlil`-family accesses (`:1101-1104`, `:1136-1139`,
    `:1188-1191`) and lets a failure on the primary accessor still try the
    fallback instead of propagating out of the whole `evidence function` op.
    """
    try:
        mlil = insn.mlil
    except Exception:
        mlil = None
    if mlil is not None:
        return mlil
    try:
        return insn.mapped_medium_level_il
    except Exception:
        return None


def _il_argument_texts(ctx, node) -> list[str]:
    for attr in ("params", "parameters"):
        params = getattr(node, attr, None)
        if params is None:
            continue
        try:
            return [str(item) for item in list(params)]
        except Exception:
            return [str(params)]
    return []


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_ARG_CONSTANT_RE = re.compile(r"0x[0-9a-fA-F]+")


def _resolve_argument_value(ctx, bv, text: str) -> dict[str, Any] | None:
    """Annotate a pointer-constant argument with what it points at.

    Generic: fixes std::string::append literals, log format strings, RTTI
    names, and service identifiers in one place. Returns None for arguments
    that are not a bare hex pointer or that resolve to nothing useful.
    """
    match = _ARG_CONSTANT_RE.fullmatch(text.strip())
    if match is None:
        return None
    address = _safe_int(int(match.group(0), 16))
    if not address:
        return None
    context = ctx._address_context(bv, address)
    resolved: dict[str, Any] = {"address": hex(address), "kind": context.get("kind")}
    string = context.get("string")
    if string:
        resolved["string"] = string.get("value")
        if string.get("encoding") and string.get("encoding") != "ascii":
            resolved["encoding"] = string["encoding"]
        if string.get("truncated"):
            resolved["truncated"] = True
    symbol = context.get("symbol")
    if symbol and symbol.get("name"):
        resolved["symbol"] = symbol["name"]
    function = context.get("function")
    if function and function.get("name"):
        resolved["function"] = function["name"]
    sections = context.get("sections")
    if sections:
        resolved["section"] = sections[0].get("name")
    if not any(key in resolved for key in ("string", "symbol", "function")):
        return None
    return resolved


def _call_arguments(ctx, bv, insn, call_addr: int) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Pick one primary argument source and quarantine uncertain extras.

    One LLIL call can map to several HLIL call expressions (BN folds adjacent
    statements); blindly merging their params attributes another call's
    arguments to this one. Prefer the single HLIL call whose address matches
    this call site; if that is ambiguous fall back to MLIL, then LLIL. Other
    candidates are returned separately (JSON-only, not shown in text).
    """
    roots = il_format._hlil_call_roots(insn)
    chosen = None
    matched = [r for r in roots if _safe_int(getattr(r, "address", None)) == int(call_addr)]
    if len(matched) == 1:
        chosen = matched[0]
    elif len(roots) == 1:
        chosen = roots[0]

    mlil = _true_mlil(insn)
    if chosen is not None:
        source, texts = "hlil", _il_argument_texts(ctx, chosen)
    elif mlil is not None:
        source, texts = "mlil", _il_argument_texts(ctx, mlil)
    else:
        source, texts = "llil", _il_argument_texts(ctx, insn)

    primary: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        entry: dict[str, Any] = {"index": index, "text": text}
        resolved = _resolve_argument_value(ctx, bv, text)
        if resolved is not None:
            entry["resolved"] = resolved
        primary.append(entry)

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = {(source, e["index"], e["text"]) for e in primary}

    def add_candidates(candidate_source: str, candidate_texts: list[str]) -> None:
        for index, text in enumerate(candidate_texts):
            marker = (candidate_source, index, text)
            if marker in seen:
                continue
            seen.add(marker)
            # #549: candidates are LOWER-confidence alternative renderings from a
            # different IL layer than the canonical `arguments` -- tag each with an
            # explicit low confidence + its provenance (`source`) so an agent never
            # mistakes a heuristic candidate for an authoritative argument and traces
            # the wrong value. `arguments` (source `argument_source`) is canonical.
            candidates.append({
                "source": candidate_source,
                "index": index,
                "text": text,
                "confidence": "low",
            })

    add_candidates("llil", _il_argument_texts(ctx, insn))
    if mlil is not None:
        add_candidates("mlil", _il_argument_texts(ctx, mlil))
    for root in roots:
        if root is chosen:
            continue
        # #476: a folded NEIGHBOR call (e.g. the outer `g` in `p = g(f(x))`) is also
        # in `roots`; adding its HLIL args leaks another call's candidates into this
        # record. Only same-address roots are alternative renderings of THIS call.
        if _safe_int(getattr(root, "address", None)) != int(call_addr):
            continue
        add_candidates("hlil", _il_argument_texts(ctx, root))
    return source, primary, candidates


def _callee_name_for_call(ctx, bv, dest_value, target) -> str | None:
    """Best-effort callee name for a call: the resolved target function's name,
    else the function/symbol at the (direct) destination address. None for an
    indirect/unresolved call."""
    if isinstance(target, dict):
        fn = target.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            return str(fn["name"])
    if dest_value is not None:
        getter = getattr(bv, "get_function_at", None)
        fn = getter(int(dest_value)) if callable(getter) else None
        if fn is not None and getattr(fn, "name", None):
            return str(fn.name)
        sym_getter = getattr(bv, "get_symbol_at", None)
        sym = sym_getter(int(dest_value)) if callable(sym_getter) else None
        if sym is not None and getattr(sym, "name", None):
            return str(sym.name)
    return None


def _callee_function_for_call(ctx, bv, dest_value, target):
    """The callee's BN function object for a DIRECT call, else None.

    Resolved two ways: an exact BN function start via ``bv.get_function_at``,
    or -- when that misses but the seam already resolved the destination to a
    function entry whose start EXACTLY matches it (``exact_start`` is True) --
    that function, re-looked-up from the entry's ``address`` field. A
    mid-function/containing-function match (``exact_start`` False, the seam's
    ``_functions_containing`` fallback) is deliberately NOT resolved here:
    treating a mid-function branch target as if it were a call to that
    function's ENTRY would check this call's arguments against the wrong
    (enclosing) function's declared arity -- a fresh silent-wrong-answer of
    exactly the #648 class this evidence exists to avoid. (#704: the previous
    secondary lookup tested ``fn_entry.get("start")``, a key
    ``_function_entry_for_address`` never emits -- dead code -- and it did not
    gate on ``exact_start`` at all.)
    """
    if dest_value is not None:
        getter = getattr(bv, "get_function_at", None)
        fn = getter(int(dest_value)) if callable(getter) else None
        if fn is not None:
            return fn
    if isinstance(target, dict):
        fn_entry = target.get("function")
        if isinstance(fn_entry, dict) and fn_entry.get("exact_start") is True:
            getter = getattr(bv, "get_function_at", None)
            try:
                addr = _parse_address(fn_entry.get("address"))
            except (TypeError, ValueError):
                addr = None
            if callable(getter) and addr is not None:
                return getter(addr)
    return None


def _abi_arg_register_count(bv, callee_fn) -> int | None:
    """How many integer arguments this platform passes in registers, or None."""
    cc = getattr(callee_fn, "calling_convention", None) if callee_fn is not None else None
    if cc is None:
        plat = getattr(bv, "platform", None)
        cc = getattr(plat, "default_calling_convention", None)
    regs = list(getattr(cc, "int_arg_regs", None) or [])
    return len(regs) or None


def _argument_arity_evidence(ctx, bv, dest_value, target,
                             arguments: list[dict[str, Any]]) -> dict[str, Any]:
    """Is the callee's ARITY known, or is HLIL enumerating ABI registers? (#648)

    When a callee has no recovered prototype BN assumes every argument register is
    live, and HLIL renders whatever happens to sit in them -- typically the NEXT
    call's argument staging, up to and including the stack canary. #549 separated
    canonical ``arguments`` from heuristic ``argument_candidates``; the residual was
    that the canonical field still claimed ``authoritative`` when nothing was known
    about the callee's arity. The distinguishing signal is available right here:
    ``memset`` reports 3 declared parameters (a bundled library type, so its
    ``authoritative`` stamp is EARNED), while an unprototyped vendor import reports
    zero -- verified against a live BN view.

    ``indirect_call`` (#704) means exactly what its name says -- the call's
    destination could not be resolved to a constant at all (``dest_value is
    None``), mirroring the sibling ``direct`` field on the call record
    (``direct: dest_value is not None``); the two are logical opposites and set
    ONLY when true, like ``abi_register_saturated``. It is a DIFFERENT condition
    from ``callee_unresolved`` -- no BN function object could be found for the
    destination, by whatever means (genuinely indirect, OR a direct call whose
    resolved constant is mid-function / an undefined address). A direct call to
    an unmatched address is ``indirect_call`` absent, ``callee_unresolved: True``:
    the call SHAPE is direct, but the callee's arity is still unknowable. HLIL
    can also render MORE OR FEWER arguments than a resolved callee's recovered
    prototype declares (an invented/dropped ABI-register arg -- ``arity_mismatch``,
    only checked against a non-empty rendered list; an empty list usually means no
    IL layer supplied one at all, not a real mismatch).

    Returns ``{"arity_unknown": bool, ...}``; ``arity_unknown`` is False whenever the
    callee cannot be resolved -- ``callee_unresolved`` carries that case instead.
    """
    evidence: dict[str, Any] = {"arity_unknown": False}
    if dest_value is None:
        evidence["indirect_call"] = True
        evidence["callee_unresolved"] = True
        return evidence
    callee_fn = _callee_function_for_call(ctx, bv, dest_value, target)
    if callee_fn is None:
        evidence["callee_unresolved"] = True
        return evidence
    if bool(getattr(callee_fn, "has_user_type", False)):
        return evidence          # a user prototype pins the arity
    func_type = getattr(callee_fn, "type", None)
    declared = getattr(func_type, "parameters", None)
    if declared is None:
        declared = getattr(callee_fn, "parameter_vars", None)
    try:
        declared_count = len(declared) if declared is not None else 0
    except TypeError:
        declared_count = 0
    is_variadic = il_format._function_is_variadic(callee_fn)
    if declared_count > 0:
        # A recovered/bundled prototype: the arity itself is known, but HLIL can
        # still have rendered MORE OR FEWER arguments than declared (an invented
        # ABI-register arg riding along, or an under-recovered call) -- flag that
        # without claiming the whole arity is unknown. Guarded on a non-empty
        # rendered list: an empty list usually means no IL layer supplied
        # arguments at all (e.g. a non-SSA LLIL call with neither `params` nor
        # `parameters`), not a genuine mismatch (#704 round-2 correction).
        if not is_variadic and arguments and len(arguments) != declared_count:
            evidence["arity_mismatch"] = True
            evidence["declared_arity"] = declared_count
        return evidence
    if is_variadic:
        return evidence          # a declared variadic with no fixed parameters
    # Zero declared parameters yet HLIL rendered arguments: the list is BN's
    # register guess, not the callee's signature. A genuinely void callee rendering
    # zero arguments agrees with its prototype and is left alone.
    if not arguments:
        return evidence
    evidence["arity_unknown"] = True
    abi_regs = _abi_arg_register_count(bv, callee_fn)
    if abi_regs is not None and len(arguments) >= abi_regs:
        # The strongest tell: the count saturates the ABI argument registers, i.e.
        # BN is enumerating registers rather than reporting parameters.
        evidence["abi_register_saturated"] = True
    return evidence


def _variadic_diagnostic(ctx, bv, dest_value, target, arg_source,
                         arguments: list[dict[str, Any]],
                         candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Provenance-labeled under-recovery diagnostic for an imported variadic
    (printf/scanf-family) call (#558).

    HLIL can render a scanf-family call showing only the fixed argument even
    though ABI setup supplied a format string and destination pointers, so a
    non-expert agent concludes the call has fewer arguments than it really does.
    This DESCRIBES the shortfall and points to a lower-IL follow-up; it never
    asserts a vulnerability. Returns None when the callee is not a recognized
    variadic format function.
    """
    callee_name = _callee_name_for_call(ctx, bv, dest_value, target)
    if callee_name is None:
        return None
    family = il_format._variadic_format_family(callee_name)
    # Fall back to BN's recovered prototype for a variadic callee we don't have a
    # format-arg-index table for -- still worth flagging, just without format parse.
    fmt_index: int | None
    is_scanf = False
    if family is not None:
        fmt_index, is_scanf = family
    else:
        callee_fn = None
        if dest_value is not None:
            getter = getattr(bv, "get_function_at", None)
            callee_fn = getter(int(dest_value)) if callable(getter) else None
        if callee_fn is None or not il_format._function_is_variadic(callee_fn):
            return None
        fmt_index = None

    recovered = len(arguments)
    diag: dict[str, Any] = {
        "callee": il_format._normalize_libc_name(callee_name),
        "is_variadic": True,
        "family": "scanf" if is_scanf else ("printf" if family is not None else None),
        "format_arg_index": fmt_index,
        "recovered_arg_count": recovered,
        "format_string": None,
        "format_conversions": None,
        "expected_min_arg_count": None,
        "confidence": "heuristic",
        "provenance": "abi-format-heuristic",
    }

    # Recover the format literal when HLIL retained it (the arg at fmt_index).
    conversions: int | None = None
    if fmt_index is not None and recovered > fmt_index:
        fmt_arg = arguments[fmt_index]
        literal = il_format._extract_format_literal(fmt_arg.get("text", ""))
        if literal is None:
            resolved = fmt_arg.get("resolved")
            if isinstance(resolved, dict) and isinstance(resolved.get("string"), str):
                literal = resolved["string"]
        if literal is not None:
            diag["format_string"] = literal
            conversions = il_format._count_format_conversions(literal, is_scanf=is_scanf)
            diag["format_conversions"] = conversions
            diag["expected_min_arg_count"] = fmt_index + 1 + conversions

    # Under-recovered when the recovered arg count falls short of what the format
    # (or, absent a parsed format, the mere presence of variadic setup) implies.
    if diag["expected_min_arg_count"] is not None:
        under = recovered < int(diag["expected_min_arg_count"])
    elif fmt_index is not None:
        # Format not parseable (often BN dropped it too): only the fixed args are
        # present, so no variadic argument was recovered -> likely under-recovered.
        under = recovered <= fmt_index + 1
    else:
        # Unknown-index variadic prototype: flag when nothing beyond a lone arg shows.
        under = recovered <= 1
    diag["under_recovered"] = bool(under)

    if under:
        role = "destination pointer(s)" if is_scanf else "variadic value(s)"
        if diag["expected_min_arg_count"] is not None:
            shortfall = (
                f"recovered {recovered} of an expected >= {diag['expected_min_arg_count']} "
                f"argument(s) (format + {conversions} {role})"
            )
        else:
            shortfall = f"recovered only {recovered} fixed argument(s); variadic {role} not surfaced in HLIL"
        diag["warning"] = (
            f"imported variadic call `{diag['callee']}` under-recovered in HLIL: {shortfall}. "
            f"Raw ABI candidates are in `argument_candidates` (low confidence); inspect "
            f"`bn disasm <caller> --linear` or `bn il <caller> --view llil` for the full "
            f"argument setup."
        )
        diag["follow_up"] = "disasm --linear / il --view llil"
    return diag


def _mlil_call_text(mlil) -> str | None:
    """Render an MLIL call, stripping a clobber-LHS assignment if present.

    Post-#661, `mlil` is usually the TRUE per-instruction form (`_true_mlil`
    prefers `insn.mlil`), which renders as `dest(args...)` with no LHS to
    strip. The stripping below still matters on the fallback path: when
    `insn.mlil` is unavailable and `_true_mlil` falls back to the COALESCED
    mapped form, BN renders it as "<written regs> = call(dest, args...)" --
    for a varargs/full-clobber callee the LHS is the entire caller-saved
    register set (~44 regs on aarch64), which buried the one thing the field
    is for. Drop the assignment LHS so the line mirrors the concise
    `arguments:` block; the full instruction (with outputs) is still
    available in the sibling `llil` field. (E17)
    """
    if mlil is None:
        return None
    text = str(mlil)
    marker = " = call("
    idx = text.find(marker)
    if idx != -1:
        return text[idx + len(" = "):]
    return text


def _function_call_evidence(ctx, bv, func, *, context: int) -> list[dict[str, Any]]:
    disasm_entries = il_format._structured_disasm_entries(bv, func)
    index_by_addr = {
        int(item["_address_int"]): index for index, item in enumerate(disasm_entries)
    }
    calls = []
    for insn in il_format._iter_llil_instructions(func):
        op_name = il_format._il_op_name(insn)
        if op_name not in {
            "LLIL_CALL",
            "LLIL_CALL_STACK_ADJUST",
            "LLIL_TAILCALL",
        }:
            continue
        call_addr = int(getattr(insn, "address", 0))
        disasm_index = index_by_addr.get(call_addr)
        previous: list[dict[str, Any]] = []
        next_instructions: list[dict[str, Any]] = []
        call_instruction = il_format._disasm_entry(bv, call_addr, arch=getattr(func, "arch", None))
        if disasm_index is not None:
            previous = [
                {"address": item["address"], "text": item["text"]}
                for item in disasm_entries[max(0, disasm_index - context) : disasm_index]
            ]
            next_instructions = [
                {"address": item["address"], "text": item["text"]}
                for item in disasm_entries[disasm_index + 1 : disasm_index + 1 + context]
            ]
            call_instruction = {
                "address": disasm_entries[disasm_index]["address"],
                "text": disasm_entries[disasm_index]["text"],
            }

        mlil = _true_mlil(insn)
        dest_value = _call_destination_value(ctx, insn)
        target = _target_entry_for_call(ctx, bv, dest_value)
        arg_source, arguments, argument_candidates = _call_arguments(ctx, bv, insn, call_addr)
        # #549: `arguments` (from `argument_source`) is canonical only when it came
        # from HLIL/ABI recovery; an mlil/llil fallback is itself heuristic. Surface
        # that trust level so downstream automation traces the right field.
        argument_confidence = "authoritative" if arg_source == "hlil" else "heuristic"
        # #648: `authoritative` meant "HLIL produced a list", not "the list is right".
        # On an unknown-arity callee HLIL invents ABI-register args (a neighbouring
        # call's staging, the stack canary), so demote and flag it -- confirmed wrong
        # against upstream source on a dogfood target. A call whose callee could
        # not be resolved at all (genuinely indirect, or a resolved destination
        # that matched no function) has no declared arity to check: it is NEVER
        # `authoritative`, regardless of source (#704: keyed on `callee_unresolved`,
        # not `indirect_call` -- the latter is purely a call-shape mirror of
        # `direct` and does not by itself mean the arity is unknown).
        arity = _argument_arity_evidence(ctx, bv, dest_value, target, arguments)
        if arity.get("callee_unresolved"):
            argument_confidence = "heuristic"
        elif (arity["arity_unknown"] or arity.get("arity_mismatch")) and argument_confidence == "authoritative":
            argument_confidence = "inferred"
        # #557: expose WHY the HLIL statement is null (reason code) rather than a bare null.
        hlil_statement, hlil_reason = il_format._hlil_statement_localization(insn)
        # #558: under-recovered imported variadic (scanf/printf-family) calls.
        variadic = _variadic_diagnostic(
            ctx, bv, dest_value, target, arg_source, arguments, argument_candidates)
        calls.append(
            {
                "address": hex(call_addr),
                "operation": op_name,
                "direct": dest_value is not None,
                "target": target,
                "llil": str(insn),
                "mlil": _mlil_call_text(mlil),
                "hlil_statement": hlil_statement,
                "hlil_statement_reason": hlil_reason,
                "pre_branch_condition": il_format._hlil_pre_branch_condition(insn),
                "argument_source": arg_source,
                "argument_confidence": argument_confidence,
                **arity,
                "arguments": arguments,
                "argument_candidates": argument_candidates,
                "variadic": variadic,
                "call_instruction": call_instruction,
                "previous_instructions": previous,
                "next_instructions": next_instructions,
            }
        )
    return calls


def _function_thunk_summary(ctx, bv, func) -> dict[str, Any]:
    sections = ctx._sections_at(bv, int(func.start))
    if any("plt" in str(section.get("name", "")).lower() for section in sections):
        return {
            "is_candidate": True,
            "reason": "function starts in a PLT/import trampoline section",
            "target": None,
            "sections": sections,
        }

    llil = [
        insn
        for insn in il_format._iter_llil_instructions(func)
        if il_format._il_op_name(insn) not in {"LLIL_NOP", "LLIL_UNDEF"}
    ]
    result: dict[str, Any] = {
        "is_candidate": False,
        "reason": None,
        "target": None,
        "sections": sections,
    }
    if not llil or len(llil) > 3:
        return result
    # #673/#704: track whether the loop's `continue` below fired because the
    # branch target resolved to a LOCAL, non-imported function -- as opposed to
    # an earlier disqualifier (op not in the branch set, unresolved dest_value,
    # unresolved target). Only the local-target case must suppress the pseudo-C
    # fallback below: a small function whose branch target IS a local defined
    # function is never a thunk FOR that function (see the comment inline), and
    # the fallback has no target/import check of its own, so leaving it
    # unguarded re-flagged exactly that case via a different code path --
    # unconditionally, since the only caller that reaches this fallback
    # (`read_decompile._thunk_veneer_warning`) already requires
    # `"/* tailcall */" in text` to be true. A genuinely unresolved/unlifted
    # tailcall (no local target was ever positively identified) must still
    # reach the fallback below, unaffected.
    saw_local_tailcall_target = False
    for insn in llil:
        op_name = il_format._il_op_name(insn)
        if op_name not in {"LLIL_JUMP", "LLIL_TAILCALL", "LLIL_CALL", "LLIL_CALL_STACK_ADJUST"}:
            continue
        dest_value = _call_destination_value(ctx, insn)
        target = _target_entry_for_call(ctx, bv, dest_value)
        if target is None:
            continue
        # #673: only an EXTERNAL branch target (import/PLT/GOT/external symbol) is a
        # thunk/veneer candidate. A tail call/jump to a LOCAL defined function --
        # e.g. an `.init_array` constructor tail-calling a local helper -- is not a
        # stub FOR that function; flagging it hid the real implementation behind a
        # "go to X" pointer instead of showing the constructor's own body.
        callee_fn = None
        if dest_value is not None:
            getter = getattr(bv, "get_function_at", None)
            callee_fn = getter(int(dest_value)) if callable(getter) else None
        if callee_fn is not None and not is_imported_function(callee_fn):
            saw_local_tailcall_target = True
            continue
        result.update(
            {
                "is_candidate": True,
                "reason": f"small function with {op_name.lower()} to another address",
                "target": target,
            }
        )
        return result

    if saw_local_tailcall_target:
        return result
    try:
        text = il_format._decompile_text(bv, func)
    except Exception:
        text = ""
    if "/* tailcall */" in text and len(llil) <= 3:
        result.update(
            {
                "is_candidate": True,
                "reason": "small function rendered as a pseudo-C tailcall",
            }
        )
    return result


def _cpp_method_this_caveat(func, decompiled_text: str = "") -> str | None:
    """#482: a C++ instance method whose implicit object pointer `this` BN recovered
    as a NON-pointer scalar (no DWARF) renders field accesses off a scalar formal and
    can show a real incoming register argument as uninitialized -- contradicting
    MLIL/disasm. We faithfully pass BN's uncertain prototype through, so emit a caveat
    rather than presenting it as fact (the ticket accepts a caveat). Returns the caveat
    or None.

    Requires all of: (1) a demangled ``Class::method`` name; (2) a recovered first
    parameter that is NOT a pointer; and (3) that first formal is actually used as a
    POINTER base (deref / member / offset-index) in the decompiled body. Gate (3) is
    what distinguishes a real mistyped-``this`` from the common false positives -- a
    STATIC method or a NAMESPACED FREE function whose non-pointer first arg is a plain
    scalar value -- since Itanium mangling can't tell namespace from class or static
    from instance by name alone (#482 FP audit). Fires only on symbol-bearing binaries
    (needs the demangled name); a fully-stripped image is a safe no-op."""
    # Use the DEMANGLED display name (symbol.short_name) -- func.name is the mangled
    # `_ZN...` on a symbol-bearing (but DWARF-less) C++ binary, which never has "::".
    name = il_format._display_name(func)
    if "::" not in name:
        return None
    try:
        pvars = list(getattr(func, "parameter_vars", []) or [])
    except Exception:
        return None
    if not pvars:
        return None
    first_type = getattr(pvars[0], "type", None)
    if first_type is None:
        return None
    # Pointer detection: default from the rendered type ("*"), and let BN's real
    # type_class confirm it when available (a typedef'd pointer / C++ reference may
    # not show a "*" but is modeled as a PointerTypeClass).
    is_pointer = "*" in str(first_type)
    try:
        if getattr(first_type, "type_class", None) == bn.TypeClass.PointerTypeClass:
            is_pointer = True
    except Exception:
        pass
    if is_pointer:
        return None
    # Gate (3): the scalar first formal must be used as a pointer base -- `p->f`,
    # `p[i]`, or a deref that contains it (`*(t*)(p + off)`). A static/free function
    # that merely uses the scalar as a value won't match, cutting the FP rate.
    first_name = str(getattr(pvars[0], "name", "") or "")
    if not first_name or not decompiled_text:
        return None
    escaped = re.escape(first_name)
    used_as_pointer = bool(re.search(
        r"\b" + escaped + r"\s*(?:->|\[)"          # p->f  or  p[i]
        r"|\*\s*\([^;\n]*\b" + escaped + r"\b",    # *(t*)(p + off) / *(t*)p
        decompiled_text,
    ))
    if not used_as_pointer:
        return None
    return (
        "possible under-recovered C++ prototype (no DWARF): the implicit object "
        "pointer `this` may be typed as a scalar (field accesses render off a scalar "
        "formal) and a real incoming register argument may render as an uninitialized "
        "variable -- cross-check `disasm --linear` / `il --view mlil` for the true ABI "
        "arguments, or recover the prototype with `proto set`."
    )


def _function_evidence(ctx, selector: str | None, identifier, *, context: int = 2,
                       offset: int = 0, limit: int | None = None,
                       address_window: tuple[int, int] | None = None):
    if context < 0:
        raise OperationFailure("invalid_context", f"Invalid evidence context size: {context}")
    if offset < 0:
        raise OperationFailure("invalid_request", f"Invalid offset: {offset}")
    if limit is not None and limit < 1:
        raise OperationFailure("invalid_request", f"Invalid limit: {limit}")
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    text = il_format._decompile_text(bv, func)
    warnings = list(il_format._render_warnings(text))
    this_caveat = _cpp_method_this_caveat(func, text)
    if this_caveat:
        warnings.append(this_caveat)

    calls = _function_call_evidence(ctx, bv, func, context=context)
    total_calls = len(calls)
    # #558: hoist per-call variadic under-recovery warnings to the function level so
    # they're visible regardless of which page is requested. Computed from the full
    # call set (before slicing) and address-tagged so an agent can find the callsite.
    for call in calls:
        variadic = call.get("variadic")
        if isinstance(variadic, dict) and variadic.get("under_recovered") and variadic.get("warning"):
            warnings.append(f"{call.get('address', '?')}: {variadic['warning']}")
    # #471: slicing/windowing controls so a large call-heavy dispatch function can be
    # inspected in bounded chunks instead of reading a full spill. Only sort by address
    # when a slice is actually requested -- the default (unsliced) output keeps its
    # original IL/discovery order so existing consumers see no change.
    slicing = bool(offset or limit is not None or address_window is not None)
    if slicing:
        calls.sort(key=lambda c: int(str(c.get("address", "0x0")), 16))
    if address_window is not None:
        lo, hi = address_window
        calls = [c for c in calls if lo <= int(str(c.get("address", "0x0")), 16) < hi]
    matched = len(calls)
    if offset:
        calls = calls[offset:]
    if limit is not None:
        calls = calls[:limit]
    returned = len(calls)

    result = {
        "function": {
            "name": func.name,
            "address": hex(func.start),
            "raw_name": getattr(func, "raw_name", func.name),
        },
        **il_format._function_metadata(func),
        "thunk": _function_thunk_summary(ctx, bv, func),
        "calls": calls,
        # #471 pagination metadata (present for both text and JSON consumers).
        "total_calls": total_calls,
        "matched_calls": matched,
        "offset": offset,
        "limit": limit,
        "returned": returned,
        "has_more": offset + returned < matched,
        "warnings": warnings,
    }
    # #626: annotate a mid-function (interior) request the same way the decompile
    # READs do (#193 Part 4). Inlined via the seam's `_containment_meta` rather
    # than importing read_decompile's `_annotate_containment`, to keep the
    # read_evidence -> read_decompile module import one-way (read_decompile
    # already lazy-imports read_evidence).
    meta = ctx._containment_meta(identifier, func)
    if meta:
        result["resolved_from"] = meta
    return result


def _pointer_table_for_view(
    ctx,
    bv,
    start: int,
    *,
    entries: int,
    stride_size: int,
    read_width: int | None = None,
    stop_after_invalid: int | None = None,
    error_on_unmapped: bool = False,
) -> dict[str, Any]:
    # Consult the address context up front: a table whose BASE is unmapped is
    # not backed by any segment, so every row would be a fabricated
    # readable:false slot. The top-level `evidence table` command errors like
    # `bn read`; internal reuse (message-lens / init-array windows) flags it as
    # a warning instead of aborting the surrounding scan (#119).
    table_context = ctx._address_context(bv, start)
    try:
        base_readable = len(bytes(bv.read(start, 1) or b"")) > 0
    except Exception:
        base_readable = False
    # "unmapped" only when the context says so AND we genuinely cannot read the
    # base. A readable base (even one without segment metadata) is not the
    # fabricated-readable:false-slots bug this guards against.
    base_unmapped = table_context.get("kind") == "unmapped" and not base_readable
    if base_unmapped and error_on_unmapped:
        raise RuntimeError(f"Address 0x{start:x} is not mapped (no bytes available)")
    pointer_size = ctx._pointer_size(bv)
    # Read width tracks the stride for sub-pointer strides: at `--stride 4` a
    # uint32[] table must be read 4 bytes wide, else an 8-byte read overlaps the
    # next slot and every entry decodes to garbage flagged [implausible] (#225).
    # Default = min(stride, pointer_size): stride 4 -> 4-byte reads; stride 8 ->
    # 8; stride 16 (a record with an 8-byte pointer per slot) -> 8. An explicit
    # read_width overrides.
    if read_width is None:
        read_width = min(stride_size, pointer_size) if stride_size > 0 else pointer_size
    read_width = max(1, int(read_width))
    # A non-null value below the lowest mapped address can't be a pointer into
    # the image -- at a fixed --stride it's almost always an inline scalar field
    # (a uint8/uint16 flag/enum in a mixed record), not a failed pointer slot.
    # Tag those so they don't inflate the "do not resolve" warning. (#170)
    mapped_floor = max(int(getattr(bv, "start", 0) or 0), 0x1000)
    rows = []
    warnings = []
    if base_unmapped:
        warnings.append(
            f"table base {hex(start)} is unmapped; entries are not backed by any segment"
        )
    invalid_run = 0
    for index in range(entries):
        entry_address = start + index * stride_size
        value = ctx._read_pointer_value(bv, entry_address, size=read_width)
        if value is None:
            rows.append(
                {
                    "index": index,
                    "entry_address": hex(entry_address),
                    "value": None,
                    "readable": False,
                    # Keep every row's `status` present so scripts can key on it
                    # uniformly (#480); the slot itself couldn't be read.
                    "status": "unreadable",
                }
            )
            invalid_run += 1
            if stop_after_invalid is not None and invalid_run >= stop_after_invalid:
                warnings.append(
                    f"stopped after {invalid_run} unreadable/implausible entries at {hex(entry_address)}"
                )
                break
            continue
        target = ctx._normalize_code_pointer(bv, value)
        likely_scalar = target["status"] == "unmapped" and 0 < value < mapped_floor
        # A legitimate inline scalar field is not an "invalid" run member either,
        # so it must not trip stop_after_invalid in interior windows.
        if target["plausible"] or target["status"] == "null" or likely_scalar:
            invalid_run = 0
        else:
            invalid_run += 1
        rows.append(
            {
                "index": index,
                "entry_address": hex(entry_address),
                "value": hex(value),
                "readable": True,
                # #480: the documented per-slot discriminator (function/mapped/null/
                # unmapped) lives at the row level, matching reading.md -- previously it
                # was only reachable at row["target"]["status"], so scripts keying on the
                # documented `status` field (in standalone AND nested init/message
                # pointer-table rows, which share this builder) silently got None.
                "status": target["status"],
                "plausible": bool(target["plausible"]),
                "likely_scalar": bool(likely_scalar),
                "target": target,
            }
        )
        if stop_after_invalid is not None and invalid_run >= stop_after_invalid:
            warnings.append(
                f"stopped after {invalid_run} unreadable/implausible entries at {hex(entry_address)}"
            )
            break

    segment = table_context.get("segment")
    section_names = [
        str(section.get("name", "")).lower()
        for section in list(table_context.get("sections") or [])
        if isinstance(section, dict)
    ]
    code_like_section = any(
        name in {".text", "__text"} or name.startswith(".plt")
        for name in section_names
    )
    data_like_section = any(
        marker in name
        for name in section_names
        for marker in ("data", "rodata", "got", "rdata", "bss", "init_array", "fini_array", "ctors", "dtors")
    )
    if isinstance(segment, dict) and segment.get("executable") and (code_like_section or not data_like_section):
        warnings.append("table start is in an executable segment; this may be code, not a pointer table")
    non_null_rows = [
        row for row in rows
        if row.get("readable") and row.get("value") not in {None, "0x0"}
    ]
    plausible_rows = [row for row in non_null_rows if row.get("plausible")]
    scalar_rows = [row for row in non_null_rows if row.get("likely_scalar")]
    # Genuine pointer slots that failed to resolve -- excludes inline scalars.
    unresolved_rows = [
        row for row in non_null_rows
        if not row.get("plausible") and not row.get("likely_scalar")
    ]
    if non_null_rows and not plausible_rows and unresolved_rows:
        warnings.append("no non-null entries resolve to mapped addresses; low confidence pointer table")
    elif unresolved_rows:
        warnings.append(
            f"{len(unresolved_rows)} non-null entries do not resolve to mapped addresses"
        )
    if scalar_rows:
        warnings.append(
            f"{len(scalar_rows)} non-null entries look like inline scalar fields, not pointers "
            "(small values below the lowest mapped address)"
        )
    interior_function_rows = [
        row for row in non_null_rows
        if isinstance(row.get("target"), dict)
        and isinstance(row["target"].get("function"), dict)
        and row["target"]["function"].get("exact_start") is False
    ]
    if interior_function_rows:
        warnings.append(
            f"{len(interior_function_rows)} entries resolve inside functions but not at function starts"
        )
    # #275: canonical envelope at the helper level, so an embedded table window
    # (evidence init/message) looks identical to the standalone `evidence table`
    # op -- one `.items[]` path everywhere, no `entries`/`items` divergence.
    return {
        "kind": "pointer_table",
        "address": hex(start),
        "pointer_size": pointer_size,
        "stride": stride_size,
        "read_width": read_width,
        "context": table_context,
        "items": rows,
        "total": len(rows),
        "warnings": warnings,
    }


def _scalar_field(ctx, bv, addr: int, offset: int, size: int) -> dict[str, Any]:
    """A scalar (non-pointer) record field: the little-endian value of up to 8
    bytes at *addr*, tagged with its record *offset* + byte *size* (#455). A field
    WIDER than 8 bytes cannot be rendered as one integer, so `value` is the low 8
    bytes and the field is flagged `truncated` -- reporting the full `size` with a
    silently-partial value would hide data (F5)."""
    full = int(size)
    n = min(full, 8)
    try:
        data = bytes(bv.read(addr, n) or b"")
    except Exception:
        data = b""
    field: dict[str, Any] = {"offset": int(offset), "kind": "scalar", "size": full}
    if n > 0 and len(data) == n:
        field["value"] = hex(int.from_bytes(data, ctx._byteorder(bv), signed=False))
        if full > 8:
            field["truncated"] = True
            field["read_bytes"] = n
            field["note"] = (
                f"scalar field is {full} bytes; `value` is the low {n} bytes only "
                f"(a scalar renders as at most a 64-bit integer)")
    else:
        field["unreadable"] = True
    return field


def _classify_ptr_field(offset: int, value: int, target: dict[str, Any]) -> dict[str, Any]:
    """Classify a declared record pointer field from its normalized target (#455):
    function_pointer / data_pointer (with a string preview + symbol when present) /
    null / unmapped."""
    status = target.get("status")
    if status == "function":
        fn = target.get("function") if isinstance(target.get("function"), dict) else {}
        return {"offset": int(offset), "kind": "function_pointer",
                "target": target.get("normalized"), "status": "function", "name": fn.get("name")}
    if status == "mapped":
        field: dict[str, Any] = {"offset": int(offset), "kind": "data_pointer",
                                 "target": target.get("normalized"), "status": "mapped"}
        context = target.get("context") if isinstance(target.get("context"), dict) else {}
        string = context.get("string")
        if isinstance(string, dict) and string.get("value"):
            field["preview"] = string["value"]
        symbol = context.get("symbol")
        if isinstance(symbol, dict) and symbol.get("name"):
            field["symbol"] = symbol["name"]
        return field
    if status == "null":
        return {"offset": int(offset), "kind": "null", "value": "0x0"}
    return {"offset": int(offset), "kind": "unmapped", "value": hex(value), "status": status}


_SCALAR_KIND_RE = re.compile(r"^([ui])(8|16|32|64)$")
_CHAR_ARRAY_RE = re.compile(r"^char\[(\d+)\]$")
_FIELD_SPEC_RE = re.compile(r"^([A-Za-z_]\w*):([^@]+)@(.+)$")


def _parse_field_spec(spec: str) -> dict[str, Any]:
    """#467: parse a ``name:type@offset`` typed record-field spec. ``type`` is a
    scalar ``u8/i8/u16/i16/u32/i32/u64/i64`` or an inline string ``char[N]``;
    ``offset`` is hex (``0x..``) or decimal. Raises OperationFailure on a bad spec."""
    m = _FIELD_SPEC_RE.match(str(spec).strip())
    if not m:
        raise OperationFailure("invalid_field",
                               f"--field must be name:type@offset, got {spec!r}")
    name, typ, off_s = m.group(1), m.group(2).strip(), m.group(3).strip()
    try:
        offset = int(off_s, 0)
    except ValueError:
        raise OperationFailure("invalid_field", f"bad offset {off_s!r} in --field {spec!r}")
    if offset < 0:
        raise OperationFailure("invalid_field", f"negative offset in --field {spec!r}")
    if typ == "ptr":
        # A pointer field; width is the target pointer size, resolved by the caller
        # (evidence calls --arg-struct #469, and the record-mode pointer path #467).
        return {"name": name, "kind": "ptr", "width": None, "offset": offset, "type": "ptr"}
    ca = _CHAR_ARRAY_RE.match(typ)
    if ca:
        width = int(ca.group(1))
        if width <= 0:
            raise OperationFailure("invalid_field",
                                   f"char[N] width must be positive in --field {spec!r}")
        return {"name": name, "kind": "char_array", "width": width,
                "offset": offset, "type": typ}
    sm = _SCALAR_KIND_RE.match(typ)
    if sm:
        return {"name": name, "kind": "scalar", "width": int(sm.group(2)) // 8,
                "signed": sm.group(1) == "i", "offset": offset, "type": typ}
    raise OperationFailure(
        "invalid_field",
        f"unknown --field type {typ!r} in {spec!r} (use u8/i8/u16/i16/u32/i32/u64/i64 or char[N])")


def _typed_field(ctx, bv, base: int, spec: dict[str, Any]) -> dict[str, Any]:
    """#467: decode one DECLARED typed field (scalar or inline char[N]) at
    ``base + spec['offset']``. A scalar renders both its integer value (honoring
    signedness) and raw hex; a char[N] renders the NUL-terminated string + raw."""
    off = spec["offset"]
    width = spec["width"]
    field: dict[str, Any] = {"name": spec["name"], "offset": off,
                             "kind": spec["kind"], "size": width, "type": spec.get("type")}
    try:
        data = bytes(bv.read(base + off, width) or b"")
    except Exception:
        data = b""
    if len(data) != width:
        field["unreadable"] = True
        return field
    if spec["kind"] == "char_array":
        field["value"] = data.split(b"\x00", 1)[0].decode("latin-1", "replace")
        field["raw"] = data.hex()
    else:
        order = ctx._byteorder(bv)
        field["value"] = int.from_bytes(data, order, signed=spec.get("signed", False))
        field["hex"] = hex(int.from_bytes(data, order, signed=False))
    return field


def _record_table_for_view(ctx, bv, start: int, *, entries: int, record_size: int,
                           ptr_fields: list[int], field_specs: list[dict] | None = None) -> dict[str, Any]:
    """#455: scan a MIXED-record dispatch table (scalar fields interleaved with
    function/data pointers) rather than a pure pointer table. Each record is
    ``record_size`` bytes; ``ptr_fields`` are the byte offsets within a record to
    read and classify as pointers. The bytes between/around the declared pointers
    are emitted as scalar fields, so an inline opcode/flags value is never misread
    as a failed pointer slot -- the exact noise a plain pointer-stride scan makes
    on these tables."""
    ptr = ctx._pointer_size(bv)
    fields_sorted = sorted(set(int(o) for o in ptr_fields))
    for off in fields_sorted:
        if off < 0 or off + ptr > record_size:
            raise OperationFailure(
                "invalid_ptr_field",
                f"pointer-field offset {hex(off)} + {ptr}-byte pointer exceeds "
                f"record-size {hex(record_size)}",
            )
    # #467: merge the declared TYPED fields (name:type@off -- scalar or char[N]) with
    # the pointer fields into one offset-sorted record layout. Zero pointer fields is
    # allowed (a scalar/string-only record); undeclared gaps stay auto-scalars.
    specs = list(field_specs or [])
    declared: list[dict[str, Any]] = [{"offset": o, "width": ptr, "_ptr": True} for o in fields_sorted]
    for s in specs:
        # A `ptr` typed field (#469 type, also usable here) is a named pointer field.
        width = ptr if s["kind"] == "ptr" else s["width"]
        if s["offset"] + width > record_size:
            raise OperationFailure(
                "invalid_field",
                f"field {s['name']!r} at {hex(s['offset'])} + {width} bytes exceeds "
                f"record-size {hex(record_size)}")
        declared.append({**s, "width": width, "_ptr": s["kind"] == "ptr"})
    declared.sort(key=lambda d: d["offset"])
    prev_end = 0
    for d in declared:
        if d["offset"] < prev_end:
            raise OperationFailure(
                "invalid_field",
                f"field at {hex(d['offset'])} overlaps the previous field (ends at {hex(prev_end)})")
        prev_end = d["offset"] + d["width"]

    rows: list[dict[str, Any]] = []
    unresolved = 0
    for i in range(entries):
        base = start + i * record_size
        record_fields: list[dict[str, Any]] = []
        cursor = 0
        for d in declared:
            off = d["offset"]
            if off > cursor:  # undeclared gap -> auto-scalar
                record_fields.append(_scalar_field(ctx, bv, base + cursor, cursor, off - cursor))
            if d["_ptr"]:
                value = ctx._read_pointer_value(bv, base + off, size=ptr)
                if value is None:
                    record_fields.append({"offset": off, "kind": "unreadable"})
                    unresolved += 1
                else:
                    field = _classify_ptr_field(off, value, ctx._normalize_code_pointer(bv, value))
                    if field["kind"] == "unmapped":
                        unresolved += 1
                    record_fields.append(field)
                cursor = off + ptr
            else:
                record_fields.append(_typed_field(ctx, bv, base, d))
                cursor = off + d["width"]
        if cursor < record_size:  # trailing undeclared gap
            record_fields.append(_scalar_field(ctx, bv, base + cursor, cursor, record_size - cursor))
        rows.append({"row": i, "base": hex(base), "fields": record_fields})
    warnings: list[str] = []
    if unresolved:
        warnings.append(
            f"{unresolved} declared pointer field(s) did not resolve to a mapped address -- "
            "check --record-size / --ptr-fields (a scalar field mis-declared as a pointer?)"
        )
    return {
        "kind": "record_table",
        "address": hex(start),
        "record_size": record_size,
        "ptr_fields": [hex(o) for o in fields_sorted],
        "fields": [{"name": s["name"], "type": s.get("type"), "offset": hex(s["offset"])}
                   for s in specs],
        "items": rows,
        "count": len(rows),
        "total": len(rows),
        "warnings": warnings,
    }


def _callee_ref_addrs(bv, callee) -> list[int]:
    """Callsite addresses that reference *callee* -- its entry plus any same-name
    PLT/import thunk (#286), so an imported registration primitive's real callers are
    found, not just direct calls to the entry."""
    targets = {int(getattr(callee, "start", -1))}
    name = getattr(callee, "name", None)
    if name:
        for sym in (bv.get_symbols_by_name(name) or []):
            targets.add(int(getattr(sym, "address", -1)))
    addrs: set[int] = set()
    get_refs = getattr(bv, "get_code_refs", None)
    if callable(get_refs):
        for t in targets:
            if t < 0:
                continue
            for ref in (get_refs(t) or []):
                a = int(getattr(ref, "address", -1))
                if a >= 0:
                    addrs.add(a)
    return sorted(addrs)


def _mlil_call_at(caller, call_addr: int):
    """The MLIL call instruction at *call_addr* in *caller*, or None."""
    try:
        instrs = list(caller.mlil.instructions)
    except Exception:
        return None
    for ins in instrs:
        if int(getattr(ins, "address", -1)) == call_addr and "CALL" in _op(ins):
            return ins
    return None


def _op(ins) -> str:
    try:
        return ins.operation.name
    except Exception:
        return ""


def _resolve_pointed_var(caller, param_expr, call_addr: int):
    """Resolve a call argument expression to the LOCAL variable it points at -- an
    ``&desc`` (MLIL_ADDRESS_OF) directly, or a register/temp copy of one (``rsi =
    &desc``). Follows up to a few SET_VAR copies back from the call. Returns the
    Variable or None (arg isn't a &local descriptor)."""
    expr = param_expr
    for _ in range(6):
        op = _op(expr)
        if "ADDRESS_OF" in op:
            return getattr(expr, "src", None)
        if op != "MLIL_VAR":
            return None
        var = getattr(expr, "src", None)
        if var is None:
            return None
        # last SET_VAR of this var before the call defines the copy.
        best = None
        best_addr = -1
        try:
            instrs = list(caller.mlil.instructions)
        except Exception:
            return None
        for ins in instrs:
            a = int(getattr(ins, "address", -1))
            if a >= call_addr or _op(ins) != "MLIL_SET_VAR":
                continue
            if getattr(ins, "dest", None) == var and a > best_addr:
                best, best_addr = ins, a
        if best is None:
            return None
        expr = getattr(best, "src", None)
    return None


def _stack_storage(v) -> int | None:
    """The stack slot offset of Variable *v* (negative, frame-relative), or None if
    *v* is not a stack variable (register/flag)."""
    try:
        if "Stack" in v.source_type.name:
            return int(v.storage)
    except Exception:
        pass
    return None


def _src_width(ins, src) -> int:
    """Byte width of a write -- the instruction store size, else the src expr size."""
    for obj in (ins, src):
        try:
            w = int(getattr(obj, "size", 0) or 0)
        except Exception:
            w = 0
        if w:
            return w
    return 1


def _descriptor_writes(caller, var, call_addr: int) -> list[tuple[int, int, int, Any, bool]]:
    """Raw writes into the local descriptor based at *var* strictly BEFORE *call_addr*,
    as ``(desc_offset, width, addr, src, is_sibling)``. Matches by DESCRIPTOR OFFSET so
    an optimizer that split the struct across sibling stack slots -- or write-combined
    several fields into one wide store to the base var -- is still recovered:
      - ``SET_VAR`` / ``SET_VAR_FIELD`` on *var*        -> offset 0 / ins.offset (same var)
      - ``SET_VAR`` / ``SET_VAR_FIELD`` on a sibling    -> (sib.storage - base.storage)(+ins.offset) (sibling)
      - ``STORE`` to ``&var + k``                       -> k (same var)
    A whole-var ``SET_VAR`` of the base itself is offset 0 (the P1 case the field-only
    scan missed). ``is_sibling`` flags a slot recovered from a DIFFERENT stack variable
    (lower confidence -- it could be an adjacent object, so the caller marks it)."""
    base_stor = _stack_storage(var)
    out: list[tuple[int, int, int, Any, bool]] = []
    try:
        instrs = list(caller.mlil.instructions)
    except Exception:
        return out
    for ins in instrs:
        a = int(getattr(ins, "address", -1))
        if a < 0 or a >= call_addr:
            continue
        op = _op(ins)
        dest = getattr(ins, "dest", None)
        src = getattr(ins, "src", None)
        off = None
        sibling = False
        if op == "MLIL_SET_VAR_FIELD":
            if dest == var:
                off = int(getattr(ins, "offset", 0))
            elif base_stor is not None and _stack_storage(dest) is not None:
                off = (_stack_storage(dest) - base_stor) + int(getattr(ins, "offset", 0))
                sibling = True
        elif op == "MLIL_SET_VAR":
            if dest == var:
                off = 0
            elif base_stor is not None and _stack_storage(dest) is not None:
                off = _stack_storage(dest) - base_stor
                sibling = True
        elif op == "MLIL_STORE":
            off = _store_offset_into(getattr(ins, "dest", None), var)
        if off is None:
            continue
        out.append((off, _src_width(ins, src), a, src, sibling))
    return out


def _covering_write(writes: list[tuple[int, int, int, Any, bool]], field_off: int, field_w: int):
    """The latest (by address) write whose byte span COVERS ``[field_off, field_off +
    field_w)`` -- an exact field store or a wider write-combined store containing it,
    else None. Returns ``(addr, write_off, src, is_sibling)``."""
    best = None
    for (woff, ww, a, src, sib) in writes:
        if woff <= field_off and field_off + field_w <= woff + ww:
            if best is None or a >= best[0]:
                best = (a, woff, src, sib)
    return best


def _store_offset_into(dest_expr, var) -> int | None:
    """If *dest_expr* is ``&var`` or ``&var + const``, the byte offset (0 for the
    bare address), else None."""
    if dest_expr is None:
        return None
    op = _op(dest_expr)
    if "ADDRESS_OF" in op and getattr(dest_expr, "src", None) == var:
        return 0
    if op in ("MLIL_ADD",):
        left = getattr(dest_expr, "left", None)
        right = getattr(dest_expr, "right", None)
        if left is not None and "ADDRESS_OF" in _op(left) and getattr(left, "src", None) == var \
                and "CONST" in _op(right):
            return int(getattr(right, "constant", getattr(right, "value", 0)) or 0)
    return None


def _decode_descriptor_field(bv, spec: dict[str, Any], write, ptr_size: int) -> dict[str, Any]:
    """Decode a covering *write* -- a ``(source_address, write_offset, src_expr,
    is_sibling)`` tuple or None -- per its declared *spec*. A CONST/CONST_PTR yields a
    value (a ptr also resolves a function/symbol name); a wider write-combined store is
    SLICED to the field's byte span; a non-constant src is reported as ``computed``
    rather than dropped; a missing write is ``unknown``. ``via: sibling_slot`` marks a
    value recovered from a different stack variable (lower confidence -- could be an
    adjacent object), and the source instruction address is recorded (#469)."""
    field: dict[str, Any] = {"name": spec["name"], "offset": spec["offset"],
                             "type": spec.get("type")}
    if write is None:
        field["status"] = "unknown"
        return field
    src_addr, write_off, src, is_sibling = write
    field["source_address"] = hex(int(src_addr))
    if is_sibling:
        field["via"] = "sibling_slot"
    if src is None:
        field["status"] = "unknown"
        return field
    if "CONST" not in _op(src):
        field["status"] = "computed"   # non-constant: merged/runtime value
        field["expr"] = str(src)
        return field
    raw = int(getattr(src, "constant", getattr(src, "value", 0)) or 0)
    field_w = ptr_size if spec["kind"] == "ptr" else int(spec.get("width") or 1)
    # Extract the field's bytes from the (possibly wider write-combined) store: shift
    # to the field's byte position and mask to its width. A no-op when the store is
    # exactly the field.
    shift = (int(spec["offset"]) - int(write_off)) * 8
    raw = (raw >> shift) & ((1 << (field_w * 8)) - 1)
    exact = int(write_off) == int(spec["offset"])
    field["status"] = "resolved"
    if spec["kind"] == "ptr":
        field["value"] = hex(raw)
        # Only resolve a symbol for a pointer store that STARTS at the field -- a
        # sub-word sliced out of a wider scalar block is not a real pointer.
        if exact:
            fn = bv.get_function_at(raw) if hasattr(bv, "get_function_at") else None
            if fn is not None:
                field["symbol"] = il_format._display_name(fn)
            else:
                sym = bv.get_symbol_at(raw) if hasattr(bv, "get_symbol_at") else None
                if sym is not None:
                    field["symbol"] = getattr(sym, "short_name", None) or getattr(sym, "name", None)
    elif spec["kind"] == "char_array":
        field["value"] = hex(raw)
    else:
        signed = bool(spec.get("signed"))
        if signed:
            bits = field_w * 8
            if raw >= (1 << (bits - 1)):
                raw -= (1 << bits)
        field["value"] = raw if signed else hex(raw)
    return field


def _call_descriptor_evidence(ctx, selector: str | None, identifier, *,
                              arg_index: int, field_specs: list[str] | None):
    """#469: treat call argument *arg_index* of every callsite of *identifier* as a
    pointer to a stack/local descriptor, and summarize the declared field values
    (constants + resolved callback symbols) written before each call."""
    if arg_index < 0:
        raise OperationFailure("invalid_request", f"--arg-struct must be >= 0, got {arg_index}")
    specs = [_parse_field_spec(f) for f in (field_specs or [])]
    if not specs:
        raise OperationFailure("invalid_request",
                               "evidence calls needs at least one --field NAME:TYPE@OFF")
    bv = ctx._resolve_view(selector)
    callee = ctx._find_function(bv, identifier)
    ptr_size = ctx._pointer_size(bv)
    for s in specs:
        if s["kind"] == "ptr":
            s["width"] = ptr_size

    # Refs that land inside the callee's own PLT/import thunk (the `jmp [GOT]` forwarder
    # named after the callee) are not real callsites -- skip them so a thunk self-ref
    # isn't reported as a degenerate descriptor row.
    name = getattr(callee, "name", None)
    thunk_addrs = {int(getattr(callee, "start", -1))}
    if name:
        for sym in (bv.get_symbols_by_name(name) or []):
            thunk_addrs.add(int(getattr(sym, "address", -1)))

    rows: list[dict[str, Any]] = []
    unresolved_calls = 0
    for call_addr in _callee_ref_addrs(bv, callee):
        callers = bv.get_functions_containing(call_addr) or []
        if not callers:
            continue
        caller = callers[0]
        if int(getattr(caller, "start", -2)) in thunk_addrs:
            continue  # ref sits inside the callee's own thunk, not a real caller
        row: dict[str, Any] = {
            "caller": il_format._display_name(caller),
            "caller_address": hex(int(getattr(caller, "start", 0))),
            "call_address": hex(call_addr),
        }
        call_ins = _mlil_call_at(caller, call_addr)
        params = list(getattr(call_ins, "params", []) or []) if call_ins is not None else []
        if arg_index >= len(params):
            row["status"] = "arg_out_of_range"
            row["fields"] = []
            unresolved_calls += 1
            rows.append(row)
            continue
        var = _resolve_pointed_var(caller, params[arg_index], call_addr)
        if var is None:
            row["status"] = "not_a_local_descriptor"
            row["fields"] = []
            unresolved_calls += 1
            rows.append(row)
            continue
        writes = _descriptor_writes(caller, var, call_addr)
        fields = []
        for s in specs:
            field_w = ptr_size if s["kind"] == "ptr" else int(s.get("width") or 1)
            cover = _covering_write(writes, int(s["offset"]), field_w)
            fields.append(_decode_descriptor_field(bv, s, cover, ptr_size))
        row["fields"] = fields
        # If NOTHING was recovered, the descriptor was likely filled some other way
        # (memcpy from a const template, a helper) -- flag it rather than present an
        # all-unknown row as a confident "ok" (#469 audit P3).
        row["status"] = "ok" if any(f.get("status") != "unknown" for f in fields) else "no_field_writes"
        rows.append(row)

    warnings: list[str] = []
    if unresolved_calls:
        warnings.append(
            f"{unresolved_calls} callsite(s) did not resolve arg {arg_index} to a local "
            "descriptor (indirect/aliased pointer, or the descriptor is not stack-local)")
    return {
        "kind": "call_descriptors",
        "callee": il_format._display_name(callee),
        "arg_index": arg_index,
        "fields": [{"name": s["name"], "type": s.get("type"), "offset": hex(s["offset"])}
                   for s in specs],
        "items": rows,
        "count": len(rows),
        "total": len(rows),
        "warnings": warnings,
    }


def _got_alias_target(ctx, bv, start: int, pointer_size: int):
    """If *start* is a ``.got``/``ImportAddressSymbol`` slot -- a single
    pointer-TO-a-table (e.g. a cross-module ``_ZTV`` vtable BN aliases into the
    local GOT) -- return ``(symbol_name, deref_target)``; else None.

    Walking such a slot as a pointer table fabricates the adjacent, UNRELATED GOT
    entries as bogus vtable slots: only slot[0] is the real pointer (#313). The
    caller refuses and names the real target so the analyst doesn't chase the
    fabricated slots."""
    sym = bv.get_symbol_at(start) if hasattr(bv, "get_symbol_at") else None
    is_alias = False
    if sym is not None:
        iat_type = getattr(bn.SymbolType, "ImportAddressSymbol", None)
        if iat_type is not None and getattr(sym, "type", None) == iat_type:
            is_alias = True
    if not is_alias:
        # Fallback: a .got/.got.plt slot even where BN didn't tag the symbol
        # type (some PIE layouts), via the address context's section names.
        names = _section_names_at(ctx._address_context(bv, start))
        if any(n.startswith(".got") for n in names):
            is_alias = True
    if not is_alias:
        return None
    try:
        deref = ctx._read_pointer_value(bv, start, size=pointer_size)
    except Exception:
        deref = None
    return (str(getattr(sym, "name", "")) if sym is not None else "", deref)


def _pointer_table(ctx, selector: str | None, address, *, entries: int = 16, stride=None,
                   width=None, record_size=None, ptr_fields=None, fields=None):
    if entries < 0:
        raise OperationFailure("invalid_entries", f"Invalid table entry count: {entries}")
    bv = ctx._resolve_view(selector)
    start = _parse_address(address)
    pointer_size = ctx._pointer_size(bv)
    # #455/#467: record-aware mode -- a mixed dispatch descriptor (scalar/string +
    # pointer fields), not a pure pointer table. Declared here (before the GOT-alias /
    # stride handling) since it scans records, not a strided pointer run.
    if (ptr_fields or fields) and record_size in (None, ""):
        raise OperationFailure(
            "invalid_request",
            "--field/--ptr-fields require --record-size to define the record stride",
        )
    if record_size not in (None, ""):
        rec_size = _parse_address(record_size)
        if rec_size <= 0:
            raise OperationFailure("invalid_record_size", f"Invalid record size: {rec_size}")
        # #467: typed scalar / char[N] fields (name:type@off). A record may be
        # scalar-only (zero pointer fields), so --ptr-fields is optional when --field
        # is given.
        specs = [_parse_field_spec(f) for f in (fields or [])]
        if not ptr_fields and not specs:
            raise OperationFailure(
                "invalid_ptr_fields",
                "--record-size needs --ptr-fields and/or --field: the pointer-field byte "
                "offsets and/or typed scalar/char[N] fields within each record, e.g. "
                "--record-size 0x18 --ptr-fields 0x8,0x10 or --field command:u32@0 "
                "--field name:char[16]@8",
            )
        offsets = [_parse_address(o) for o in (ptr_fields or [])]
        return _record_table_for_view(ctx, bv, start, entries=entries,
                                      record_size=rec_size, ptr_fields=offsets,
                                      field_specs=specs)
    # A GOT/import-address slot is a pointer TO a table, not a table; walking it
    # would fabricate adjacent unrelated GOT entries as bogus slots (#313).
    # Refuse and point at the real table (*slot[0]) instead.
    alias = _got_alias_target(ctx, bv, start, pointer_size)
    if alias is not None:
        sym_name, deref = alias
        name_part = f" ({sym_name})" if sym_name else ""
        if deref:
            target_part = (
                f". It is a single {pointer_size}-byte pointer whose value is "
                f"{hex(deref)} (its real target); run `evidence table {hex(deref)}` "
                f"to walk that."
            )
        else:
            target_part = " and its slot[0] pointer is unreadable."
        raise OperationFailure(
            "got_alias",
            f"{hex(start)} is a GOT/import-address slot{name_part}, not a pointer "
            f"table: walking it would present adjacent unrelated GOT entries as "
            f"bogus slots{target_part}",
        )
    stride_size = _parse_address(stride) if stride not in (None, "") else pointer_size
    if stride_size <= 0:
        raise OperationFailure("invalid_stride", f"Invalid table stride: {stride_size}")
    # Explicit --width overrides the stride-derived read width (#225).
    read_width = _parse_address(width) if width not in (None, "") else None
    if read_width is not None and read_width <= 0:
        raise OperationFailure("invalid_width", f"Invalid read width: {read_width}")
    # #275: _pointer_table_for_view already returns the canonical {kind, items,
    # total, ...} envelope -- identical standalone and embedded.
    return _pointer_table_for_view(
        ctx,
        bv,
        start,
        entries=entries,
        stride_size=stride_size,
        read_width=read_width,
        error_on_unmapped=True,
    )


def _section_names_at(context) -> set[str]:
    return {
        str(s.get("name", "")).lower()
        for s in (context.get("sections") or [])
        if isinstance(s, dict) and s.get("name")
    }


def _symbol_by_any_name(bv, name: str):
    """A symbol matching *name* by raw (mangled) name, then by display name."""
    graw = getattr(bv, "get_symbol_by_raw_name", None)
    if callable(graw):
        try:
            s = graw(name)
            if s is not None:
                return s
        except Exception:
            pass
    gbn = getattr(bv, "get_symbols_by_name", None)
    if callable(gbn):
        try:
            ss = list(gbn(name) or [])
            if ss:
                return ss[0]
        except Exception:
            pass
    return None


# RTTI data-symbol tags (Itanium ABI): vtable / typeinfo / typeinfo-name. For a
# mangled type fragment `N5TCLAP3ArgE`, the symbols are `_ZTVN5TCLAP3ArgE`, etc.
_RTTI_PREFIXES = (("_ZTV", "vtable"), ("_ZTI", "typeinfo"), ("_ZTS", "typeinfo-name"))


_CXX_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


def _itanium_typeinfo_fragment(name: str) -> str | None:
    """The Itanium RTTI type-string fragment for a DEMANGLED C++ name -- the form
    the class lens prints -- or None if *name* isn't a plain (possibly nested)
    identifier we can length-encode. ``media::codec::JsonCodec`` ->
    ``N5media5codec9JsonCodecE``; ``Codec`` -> ``5Codec`` (#305).

    Length-prefix mangling only: each ``::`` component is encoded ``<len><name>``,
    nested names wrap in ``N..E``. Templates / operators / anonymous namespaces
    are out of scope (return None) -- they need full Itanium mangling, and they
    already fail today; this strictly adds the common namespaced-class case so the
    name `class list`/`class show` print resolves in `evidence message`."""
    parts = [p for p in name.split("::") if p]
    if not parts or not all(_CXX_IDENT_RE.fullmatch(p) for p in parts):
        return None
    body = "".join(f"{len(p)}{p}" for p in parts)
    return f"N{body}E" if len(parts) > 1 else body


def _rtti_name_candidates(query: str) -> list[str]:
    """Type-name forms to try for RTTI resolution: the query as given, plus its
    Itanium typeinfo-string fragment when the query is a demangled name (so the
    fully-qualified name the lens prints resolves, not just the bare leaf) (#305)."""
    q = query.strip()
    candidates = [q] if q else []
    frag = _itanium_typeinfo_fragment(q) if q else None
    if frag and frag not in candidates:
        candidates.append(frag)
    return candidates


def _best_rtti_symbol(bv, name: str):
    """The best symbol for an RTTI name, preferring a real DATA definition over a
    `.got`/import-address ALIAS. An `_ZTV<T>` name commonly resolves to both a
    `.data.rel.ro` vtable OBJECT and a `.got` pointer-TO-it alias; walking the
    alias renders adjacent GOT entries as bogus slots, so pick the definition so
    the lens surfaces the real vtable (#305). Falls back to ``_symbol_by_any_name``
    when the view doesn't expose name->symbols enumeration (test fakes)."""
    gb = getattr(bv, "get_symbols_by_name", None)
    syms = list(gb(name)) if callable(gb) else []
    if not syms:
        return _symbol_by_any_name(bv, name)

    def _is_got_alias(sym) -> bool:
        addr = getattr(sym, "address", None)
        if addr is None:
            return True
        iat = getattr(bn.SymbolType, "ImportAddressSymbol", None)
        if iat is not None and getattr(sym, "type", None) == iat:
            return True
        secs = bv.get_sections_at(int(addr)) if hasattr(bv, "get_sections_at") else []
        return any(str(getattr(s, "name", "")).startswith(".got") for s in (secs or []))

    definitions = [s for s in syms if getattr(s, "address", None) is not None and not _is_got_alias(s)]
    if definitions:
        return definitions[0]
    return syms[0]


def _resolve_rtti_symbols(ctx, bv, query: str, table_entries: int) -> list[dict[str, Any]]:
    """Resolve a type-name to its RTTI DATA symbols -- the vtable / typeinfo /
    typeinfo-name objects that actually carry the metadata, in .rodata/.data.rel.ro
    with real xrefs -- which is what the lens is meant to find but a .dynstr
    name-string match never reaches (#194). Accepts the DEMANGLED fully-qualified
    name the class lens prints (mangled to the typeinfo fragment, #305) as well as
    the raw mangled fragment."""
    out: list[dict[str, Any]] = []
    seen_addrs: set[int] = set()
    ptr = ctx._pointer_size(bv)
    for q in _rtti_name_candidates(query):
        for prefix, kind in _RTTI_PREFIXES:
            sym = _best_rtti_symbol(bv, prefix + q)
            if sym is None or getattr(sym, "address", None) is None:
                continue
            addr = int(sym.address)
            if addr in seen_addrs:
                continue
            seen_addrs.add(addr)
            entry: dict[str, Any] = {
                "kind": kind,
                "symbol": prefix + q,
                "address": hex(addr),
                "xrefs": read_xrefs._xrefs_to_address(ctx, bv, addr),
            }
            # The vtable's slots (typeinfo pointer + virtual methods) are the
            # payload; show the table window so the lens directly surfaces them.
            if kind == "vtable" and table_entries:
                # #305: an `_ZTV<T>` name often resolves to BOTH a `.data.rel.ro`
                # vtable-object DEFINITION and a `.got` import ALIAS (a pointer TO
                # it). `_best_rtti_symbol` already prefers the definition; if only
                # the GOT alias resolved, walking it (or its unrelocated slot)
                # would fabricate adjacent GOT entries as bogus slots, so refuse
                # the window and say so honestly rather than lie (#305/#313).
                if _got_alias_target(ctx, bv, addr, ptr) is None:
                    entry["table_window"] = _pointer_table_for_view(
                        ctx, bv, addr, entries=table_entries,
                        stride_size=ptr, stop_after_invalid=2,
                    )
                else:
                    entry["vtable_is_got_alias"] = True
                    entry["note"] = (
                        "this _ZTV symbol is a GOT/import alias, not the vtable "
                        "object, and the .data.rel.ro definition was not found as a "
                        "symbol; run `evidence table` on the real vtable address"
                    )
            out.append(entry)
    return out


def _message_lens(ctx, selector: str | None, query: str, *, limit: int = 20, table_entries: int = 6):
    limit = _validate_count(limit, label="limit", minimum=1)
    table_entries = _validate_count(table_entries, label="table_entries", minimum=0)
    bv = ctx._resolve_view(selector)
    # Match the query as given AND its Itanium typeinfo-string fragment, so the
    # DEMANGLED fully-qualified name the class lens prints
    # (`media::codec::JsonCodec`) finds the RTTI string (`N5media5codec9JsonCodecE`),
    # not just the hand-stripped bare leaf (#305).
    needles = [query.lower()] if query else []
    name_fragment = _itanium_typeinfo_fragment(query) if query else None
    if name_fragment and name_fragment.lower() not in needles:
        needles.append(name_fragment.lower())
    matches = []
    total_matched = 0
    dynstr_excluded = 0
    for item in list(getattr(bv, "strings", [])):
        value = str(getattr(item, "value", ""))
        if needles and not any(n in value.lower() for n in needles):
            continue
        address = int(getattr(item, "start", 0))
        context = ctx._address_context(bv, address)
        # `.dynstr` matches are mangled SYMBOL-NAME strings, never the RTTI
        # metadata this lens targets; on a symbol-retaining binary they drown the
        # real result in 0-xref noise. Exclude them (a stripped binary has no
        # .dynstr, so this is safe there too) -- count for an honest total + hint
        # (#194).
        if ".dynstr" in _section_names_at(context):
            dynstr_excluded += 1
            continue
        # Count every (non-.dynstr) match so the reported total is honest, but
        # only build the expensive per-match evidence for the first `limit`.
        total_matched += 1
        if len(matches) >= limit:
            continue
        xrefs = read_xrefs._xrefs_to_address(ctx, bv, address)
        metadata_tables = []
        for ref in list(xrefs.get("data_refs") or [])[:3]:
            try:
                ref_addr = _parse_address(ref["address"])
            except Exception:
                continue
            start = max(0, ref_addr - ctx._pointer_size(bv) * 2)
            metadata_tables.append(
                _pointer_table_for_view(
                    ctx,
                    bv,
                    start,
                    entries=table_entries,
                    stride_size=ctx._pointer_size(bv),
                    stop_after_invalid=1,
                )
            )

        matches.append(
            {
                "type_string": {
                    "address": hex(address),
                    "value": value,
                    "length": int(getattr(item, "length", len(value))),
                    "context": context,
                },
                "xrefs": xrefs,
                "metadata_table_windows": metadata_tables,
            }
        )

    # Surface the real RTTI metadata directly (vtable/typeinfo/typeinfo-name data
    # symbols), the structures a .dynstr name match can never reach (#194).
    rtti_symbols = _resolve_rtti_symbols(ctx, bv, query, table_entries)

    hints: list[str] = []
    if name_fragment and "::" in (query or "") and (total_matched or rtti_symbols):
        # #305: be explicit that the demangled name was mangled to its Itanium
        # typeinfo-string form for the search (the form RTTI metadata carries).
        # Only when the query is actually ::-qualified (so the fragment, not the
        # plain needle, is the matcher) AND something matched -- else the hint
        # would over-claim on a bare-leaf or zero-match query (review #5).
        hints.append(
            f"matched the demangled name '{query}' via its Itanium typeinfo "
            f"fragment '{name_fragment}'."
        )
    if dynstr_excluded:
        hints.append(
            f"excluded {dynstr_excluded} match(es) in .dynstr (mangled symbol-name "
            f"strings, never RTTI metadata). This binary retains its symbol table; "
            f"resolve the _ZTV/_ZTI/_ZTS<type> data symbols directly, or run "
            f"`bn evidence table <vtable-addr>`."
        )
    if rtti_symbols:
        hints.append(
            f"resolved {len(rtti_symbols)} RTTI data symbol(s) for the type "
            f"(vtable/typeinfo/typeinfo-name) -- see rtti_symbols."
        )

    return {
        "kind": "messages",
        "query": query,
        "items": matches,
        "count": len(matches),
        "total": total_matched,
        "truncated": total_matched > len(matches),
        "dynstr_excluded": dynstr_excluded,
        "rtti_symbols": rtti_symbols,
        "hints": hints,
    }


_INIT_SECTION_HINTS = (
    "init_array",
    "preinit_array",
    "fini_array",
    ".ctors",
    ".dtors",
    "__mod_init_func",
    "__mod_term_func",
)


def _pe_tls_callbacks(bv) -> dict[str, int] | None:
    """#380: locate a PE's TLS callback array via IMAGE_TLS_DIRECTORY.
    AddressOfCallBacks (data directory[9]). Returns ``{"address", "count",
    "ptr_size"}`` for the null-terminated callback VA array, or None when the
    target isn't a PE / has no TLS callbacks. Parsed from the mapped headers (BN
    exposes no TLS-directory accessor); every read is bounds-checked."""
    if not callable(getattr(bv, "read", None)):
        return None
    base = int(getattr(bv, "start", 0))

    def u(addr: int, n: int) -> int | None:
        b = bv.read(addr, n)
        return int.from_bytes(b, "little") if b and len(b) == n else None

    if bv.read(base, 2) != b"MZ":
        return None
    e_lfanew = u(base + 0x3C, 4)
    if not e_lfanew:
        return None
    pe = base + e_lfanew
    if bv.read(pe, 4) != b"PE\x00\x00":
        return None
    opt = pe + 4 + 20  # PE signature (4) + COFF file header (20)
    magic = u(opt, 2)
    if magic == 0x20B:        # PE32+
        dd_off, ptr_size, aocb_off = 112, 8, 24
    elif magic == 0x10B:      # PE32
        dd_off, ptr_size, aocb_off = 96, 4, 12
    else:
        return None
    # NumberOfRvaAndSizes (the 4 bytes just before the DataDirectory array) must
    # cover index 9 (TLS); a non-conforming PE with fewer entries would otherwise
    # read a garbage RVA from the section table beyond the array (review nit).
    n_dirs = u(opt + dd_off - 4, 4)
    if n_dirs is None or n_dirs < 10:
        return None
    tls_rva = u(opt + dd_off + 9 * 8, 4)   # data directory[9] = TLS
    if not tls_rva:
        return None
    # AddressOfCallBacks is an absolute VA computed at link time against the
    # PREFERRED ImageBase (PE32+ opt+24 / PE32 opt+28). When BN mapped the image
    # at a different base (relocated / rebased), rebase the array VA into BN's
    # address space, or the read below hits an unmapped preferred-base address and
    # the callbacks are silently missed. A no-op when BN loaded at the preferred
    # base (image_base == bv.start), the common case.
    image_base = u(opt + (24 if magic == 0x20B else 28), ptr_size)
    aocb = u(base + tls_rva + aocb_off, ptr_size)   # AddressOfCallBacks (preferred-base VA)
    if not aocb:
        return None
    if image_base is not None:
        aocb = aocb - image_base + base
    count = 0
    addr = aocb
    while count < 4096:
        v = u(addr, ptr_size)
        if not v:
            break
        count += 1
        addr += ptr_size
    if count == 0:
        return None
    return {"address": aocb, "count": count, "ptr_size": ptr_size}


def _init_arrays(ctx, selector: str | None, *, limit: int = 64):
    if limit < 0:
        raise OperationFailure("invalid_limit", f"Invalid init-array limit: {limit}")
    bv = ctx._resolve_view(selector)
    pointer_size = ctx._pointer_size(bv)
    sections = []
    for name, sec in getattr(bv, "sections", {}).items():
        lowered = str(name).lower()
        if not any(hint in lowered for hint in _INIT_SECTION_HINTS):
            continue
        start = int(getattr(sec, "start", 0))
        end = int(getattr(sec, "end", 0))
        total_entries = max(0, (end - start) // pointer_size)
        shown_entries = min(total_entries, limit)
        table = _pointer_table_for_view(
            ctx,
            bv,
            start,
            entries=shown_entries,
            stride_size=pointer_size,
        )
        sections.append(
            {
                "name": str(name),
                "start": hex(start),
                "end": hex(end),
                "total_entries": total_entries,
                "shown_entries": shown_entries,
                "truncated": total_entries > shown_entries,
                "table": table,
            }
        )
    # #380: PE targets carry pre-entry code in the TLS callback array, which the
    # ELF section scan above misses. Surface it with the same pointer-table
    # evidence so `evidence init` isn't falsely empty on a PE with TLS callbacks.
    tls = _pe_tls_callbacks(bv)
    if tls is not None:
        shown = min(tls["count"], limit)
        # read_width pinned to the TLS pointer size: on a PE32+ the callbacks are
        # 8-byte VAs, but bv/arch may report a 4-byte address_size, which would
        # default read_width to 4 and truncate the high callback VAs (codex review).
        table = _pointer_table_for_view(
            ctx, bv, tls["address"], entries=shown, stride_size=tls["ptr_size"],
            read_width=tls["ptr_size"],
        )
        sections.append(
            {
                "name": "TLS callbacks (PE IMAGE_TLS_DIRECTORY.AddressOfCallBacks)",
                "start": hex(tls["address"]),
                "end": hex(tls["address"] + tls["count"] * tls["ptr_size"]),
                "total_entries": tls["count"],
                "shown_entries": shown,
                "truncated": tls["count"] > shown,
                "table": table,
            }
        )
    sections.sort(key=lambda item: int(item["start"], 16))
    # #275: `items` are the init/ctor sections (each retains its nested `entries`
    # table); `kind` discriminates the envelope.
    return {
        "kind": "init_arrays",
        "pointer_size": pointer_size,
        "items": sections,
        "total": len(sections),
    }


# Data sections that commonly hold code-pointer tables (vtables, dispatch/ops tables).
# Scanned by `evidence surface` for the hidden code surface (#503 / #169 L2). `.got`/
# `.got.plt` are deliberately EXCLUDED: their slots always point at code BN already
# functionized, so they can never yield a missing-function candidate -- they'd be pure
# noise that also starves the --max-tables budget (audit).
_SURFACE_SCAN_HINTS = (".data.rel.ro", ".data", ".rodata")

# `evidence surface` code-vs-data discriminator (#503). DECODE DEPTH -- the count of
# sequential valid, non-undefined instructions from a candidate -- is the real signal:
# on fixed-width RISC (the firmware target class) a real function decodes a long clean
# run while DATA (or a mid-instruction pointer) hits an undefined instruction almost
# immediately. Measured on aarch64 firmware: known functions median 24, .rodata data
# median 0; a single decode is useless (~90% of data decodes to one instruction). A
# candidate is `code_likely` when aligned AND its depth reaches the threshold; at 8 the
# false-negative rate on real functions is ~2%. (Weaker on dense x86, where data rarely
# hits an undefined -- there `aligned` + table context carry more weight.)
_MAX_DECODE_PROBE = 16
_STRONG_DECODE_DEPTH = 8


def _hidden_surface(ctx, selector: str | None, *, table_min_run: int = 3,
                    max_tables: int = 64, max_candidates: int = 128,
                    max_scan_bytes: int = 16_000_000) -> dict[str, Any]:
    """#503: enumerate the HIDDEN code surface a passive `function list` misses -- code
    reachable through data, not direct calls. READ-ONLY: it reports addresses BN did not
    turn into functions, it does NOT create them (that stays the agent's separate write).

    Three composed reports, under one read lock:
      - ``init_sections``: `.init_array`/`.ctors`/`.fini_array` constructor pointers (pre-
        main code), each flagged whether BN recovered a function for it.
      - ``candidate_tables``: runs of >= ``table_min_run`` consecutive pointers-to-executable
        -- vtable/dispatch-table candidates, with resolved-vs-missing slot counts. Scans the
        named data sections (`.data`/`.rodata`/`.data.rel.ro`) when present, else falls back
        to the readable SEGMENTS (a raw/monolithic firmware image with no named sections, its
        tables interleaved in a single r-x region -- the primary target class).
      - ``missing_function_candidates``: the deduped executable targets (from init + tables)
        with NO BN function, each carrying ``aligned`` and ``decode_depth`` (the clean-decode
        run length -- the real code-vs-data signal on RISC firmware) and a ``code_likely``
        classification (aligned + a long clean run). Still candidates to CONFIRM with
        `disasm`, NOT asserted functions -- but ``code_likely`` is the high-confidence subset
        to start with."""
    if table_min_run < 2:
        raise OperationFailure("invalid_request", "table-min-run must be >= 2")
    bv = ctx._resolve_view(selector)
    ps = ctx._pointer_size(bv)
    try:
        thumb_arch = bool(ctx._supports_thumb_pointer_tags(bv))
    except Exception:
        thumb_arch = False
    # Honor the view's endianness -- hardcoding little-endian would silently decode
    # every pointer to garbage on a big-endian target (MIPS/PPC/ARM-BE firmware, the
    # primary use case), reading as a false "no hidden surface".
    order = ctx._byteorder(bv)
    img_lo = int(getattr(bv, "start", 0) or 0)
    img_hi = int(getattr(bv, "end", 0) or 0) or (img_lo + (1 << 48))
    warnings: list[str] = []

    def has_fn(addr: int) -> bool:
        try:
            return bool(bv.get_functions_containing(addr))
        except Exception:
            return False

    def exec_target(addr: int) -> bool:
        try:
            seg = bv.get_segment_at(addr)
            if seg is None or not getattr(seg, "executable", False):
                return False
        except Exception:
            return False
        # #647: an executable SEGMENT is not enough. On a single-`LOAD` aarch64 PIE the
        # whole image -- `.rodata` included -- is mapped r-x, so every pointer into
        # read-only DATA passed this test and a 514-row `{char *desc; char *usage;}` help
        # table was reported as a 514-entry dispatch table with 514 missing functions.
        # Consult the section the way the `evidence table` path already does. `None`
        # (no section covers the address) keeps the old permissive answer, so the scan
        # is not blinded on the raw/monolithic firmware images with no named sections
        # that this command exists to serve.
        code_like = _section_is_code_like(bv, addr)
        return code_like is not False

    def norm_ptr(addr: int) -> int:
        # A Thumb function pointer is the even code address with the low bit set; the
        # code lives at addr & ~1. Normalize so a Thumb candidate is checked/reported
        # at its real entry, not the odd tagged value (audit: the scan path didn't).
        return (addr & ~1) if (thumb_arch and (addr & 1)) else addr

    arch = getattr(bv, "arch", None)
    # decode_depth discriminates code from data only on a FIXED-WIDTH ISA (RISC firmware);
    # on a dense variable-length ISA (x86) data decodes long clean runs too, so the
    # `code_likely` flag over-reports there. Detect it so the warning rides on the OUTPUT,
    # not just the skill doc (audit) -- a consumer who never read the skill isn't misled.
    _arch_name = str(getattr(arch, "name", "")).lower()
    _variable_length_isa = any(k in _arch_name for k in
                               ("x86", "x86_64", "x64", "i386", "i486", "i586", "i686", "amd64"))

    def _code_signals(addr: int) -> tuple[bool, int]:
        # `aligned` (target on the arch's code granularity) plus DECODE DEPTH -- the run
        # of sequential valid, non-undefined instructions from the target, capped at
        # _MAX_DECODE_PROBE. Depth is the strong code-vs-data signal (see the constants):
        # a real function decodes a long clean run, data hits an undefined instruction
        # almost at once. Neither is gated in the scan -- both are reported so the
        # classification is transparent and the agent can still confirm with `disasm`.
        thumb = thumb_arch and (addr & 1)
        norm = addr & ~1 if thumb else addr
        aligned = (norm % (2 if thumb else 4) == 0)
        depth = 0
        a = norm
        for _ in range(_MAX_DECODE_PROBE):
            try:
                raw = bytes(bv.read(a, 16) or b"")   # up to a max-length x86 instruction
                info = arch.get_instruction_info(raw, a) if (arch is not None and raw) else None
                length = int(getattr(info, "length", 0) or 0) if info is not None else 0
                text = bv.get_disassembly(a) or "" if length > 0 else ""
            except Exception:
                break
            if length <= 0 or (not text) or ("undefined" in text) or text.strip().startswith("udf"):
                break
            depth += 1
            a += length
        return aligned, depth

    # Deduped executable targets with no function -> the hidden-surface candidates.
    missing: dict[int, dict[str, Any]] = {}
    # #653.5: distinct candidates the cap suppressed. Tracked (cheaply -- no decode
    # probe) so the cap warning can disclose "128 of N" instead of a silent prefix
    # that leaves no basis for deciding whether raising --max-candidates is worth it.
    missing_over_cap: set[int] = set()

    def note_missing(addr: int, provenance: str) -> None:
        norm = norm_ptr(addr)
        if norm in missing or has_fn(norm):
            return
        if len(missing) >= max_candidates:
            missing_over_cap.add(norm)
            return
        aligned, depth = _code_signals(addr)
        missing[norm] = {
            "address": hex(norm),
            "provenance": provenance,
            "aligned": aligned,
            "decode_depth": depth,   # sequential valid instructions before undefined
            # the default classification: aligned AND a long clean decode run. Raw
            # `decode_depth` is reported too, so the agent can tighten/loosen.
            "code_likely": bool(aligned and depth >= _STRONG_DECODE_DEPTH),
            "section": _section_name_at(bv, norm),
        }
        # #647 defence in depth: a candidate that resolves to a printable string is
        # self-refuting evidence -- inline it the way `evidence table` does, so a
        # false lead reads as `-> "Set Channel Index\n"` instead of a bare address
        # an agent has to spend a second command disproving.
        preview = _string_preview_at(ctx, bv, norm)
        if preview:
            missing[norm]["string"] = preview

    # 1) init/ctor sections (reuse the existing evidence).
    init_sections: list[dict[str, Any]] = []
    # A high limit so init/ctor entries aren't silently under-counted vs total_entries
    # (the default `evidence init` limit of 64 would clip a large .init_array).
    for sec in (_init_arrays(ctx, selector, limit=4096).get("items") or []):
        rows = ((sec.get("table") or {}).get("items")) or []
        resolved = missing_here = 0
        for r in rows:
            tgt = r.get("target") or {}
            val = tgt.get("normalized") or r.get("value")
            if r.get("status") == "function":
                resolved += 1
            elif exec_target(_as_int(val)):
                missing_here += 1
                note_missing(_as_int(val), f"init:{sec.get('name')}")
        init_sections.append({
            "name": sec.get("name"), "start": sec.get("start"), "end": sec.get("end"),
            "total_entries": sec.get("total_entries"),
            "resolved_functions": resolved, "missing_functions": missing_here,
        })

    # 2) scan data regions for runs of consecutive pointers-to-executable. Prefer named
    # data sections; on a raw/monolithic firmware image with NO named data sections
    # (e.g. a VxWorks kernel BN loads as a single r-x segment, code + data interleaved)
    # fall back to the readable SEGMENTS so the feature works on its primary target class.
    scan_regions: list[tuple[str, int, int]] = []
    for name, section in (getattr(bv, "sections", {}) or {}).items():
        if any(h in str(name).lower() for h in _SURFACE_SCAN_HINTS):
            scan_regions.append((str(name), int(getattr(section, "start", 0)),
                                 int(getattr(section, "end", 0))))
    if not scan_regions:
        for seg in (getattr(bv, "segments", []) or []):
            if getattr(seg, "readable", False):
                scan_regions.append((f"segment@{hex(int(seg.start))}",
                                     int(seg.start), int(seg.end)))
        if scan_regions:
            warnings.append(
                "no named data sections matched -- scanned readable segment(s) instead "
                "(raw/monolithic firmware); code and data share the region so candidate "
                "tables carry more noise -- lean on the resolved/missing counts and confirm "
                "with disasm")

    candidate_tables: list[dict[str, Any]] = []
    tables_over_cap = 0
    scanned = 0
    scan_capped = False
    for name, start, end in scan_regions:
        if scanned >= max_scan_bytes:
            scan_capped = True
            break
        if end - start > max_scan_bytes - scanned:
            end = start + (max_scan_bytes - scanned)
            scan_capped = True
        scanned += max(0, end - start)
        try:
            data = bytes(bv.read(start, end - start) or b"")
        except Exception:
            continue
        run: list[int] = []

        def flush(run_addr: int, run_vals: list[int]) -> None:
            nonlocal tables_over_cap
            if len(run_vals) < table_min_run:
                return
            if len(candidate_tables) >= max_tables:
                tables_over_cap += 1   # #653.5: disclose the total, not a silent prefix
                return
            # Normalize each pointer before the miss count so a Thumb pointer
            # (stored as addr|1) is checked at its real even entry -- matching what
            # note_missing already does (#530). Without this, every Thumb slot counts
            # as missing (over-reports missing, under-reports resolved) on ARM/Thumb.
            miss = sum(1 for v in run_vals if not has_fn(norm_ptr(v)))
            for v in run_vals:
                note_missing(v, f"table:{hex(run_addr)}")
            candidate_tables.append({
                "address": hex(run_addr),
                "section": str(name),
                "entries": len(run_vals),        # the TRUE slot count
                "resolved_functions": len(run_vals) - miss,
                "missing_functions": miss,
                "slots": [hex(v) for v in run_vals[:32]],
                "slots_truncated": len(run_vals) > 32,   # `slots` shows the first 32
            })

        run_start = start
        off = 0
        while off + ps <= len(data):
            v = int.from_bytes(data[off:off + ps], order)
            addr = start + off
            if img_lo <= v < img_hi and exec_target(v):
                if not run:
                    run_start = addr
                run.append(v)
            else:
                flush(run_start, run)
                run = []
            off += ps
        flush(run_start, run)

    if scan_capped:
        warnings.append(
            f"data scan capped at {max_scan_bytes} bytes; regions (or their tails) beyond "
            "the budget were not scanned -- on a monolithic image the pointer tables often "
            "sit in the TAIL, so raise --max-scan-bytes to cover the whole image")
    tables_total = len(candidate_tables) + tables_over_cap
    candidates_total = len(missing) + len(missing_over_cap)
    if len(candidate_tables) >= max_tables:
        warnings.append(
            f"candidate tables capped at {max_tables} of {tables_total} (--max-tables)")
    if len(missing) >= max_candidates:
        warnings.append(
            f"missing-function candidates capped at {max_candidates} of {candidates_total} "
            "(--max-candidates)")
    if _variable_length_isa and missing:
        warnings.append(
            f"decode_depth is a WEAK code-vs-data discriminator on this variable-length ISA "
            f"({_arch_name or 'x86-family'}): data decodes long clean runs too, so `code_likely` "
            "over-reports here -- lean on `aligned` and each table's resolved/missing ratio, and "
            "confirm with disasm")

    return {
        "kind": "hidden_surface",
        "init_sections": init_sections,
        "candidate_tables": candidate_tables,
        "missing_function_candidates": sorted(missing.values(), key=lambda m: m["address"]),
        "summary": {
            "init_sections": len(init_sections),
            "candidate_tables": len(candidate_tables),
            "missing_function_candidates": len(missing),
            # #653.5: the pre-cap totals, so a capped run is attributable.
            "candidate_tables_total": tables_total,
            "missing_function_candidates_total": candidates_total,
            # the high-confidence subset (aligned + long clean decode run) -- start here.
            "code_likely_candidates": sum(1 for m in missing.values() if m["code_likely"]),
        },
        "warnings": warnings,
    }


def _as_int(v: Any) -> int:
    try:
        return int(str(v), 16) if isinstance(v, str) and v.lower().startswith("0x") else int(v)
    except (TypeError, ValueError):
        return -1


def _section_name_at(bv, addr: int) -> str | None:
    try:
        secs = bv.get_sections_at(addr)
        return str(secs[0].name) if secs else None
    except Exception:
        return None


# Section-name markers, kept in sync with the `evidence table` warning at
# `_table_for_view` -- that path already reasoned about code-like vs data-like
# sections while the scan that GENERATES candidates only tested segment perms (#647).
_CODE_SECTION_NAMES = {".text", "__text", ".init", ".fini"}
_DATA_SECTION_MARKERS = (
    "data", "rodata", "got", "rdata", "bss", "init_array", "fini_array",
    "ctors", "dtors", "dynstr", "dynsym", "interp", "eh_frame", "gnu.hash",
    "rela", "dynamic", "strtab", "symtab",
)


def _string_preview_at(ctx, bv, addr: int) -> str | None:
    """The printable string at *addr*, if BN found one there (#647)."""
    try:
        context = ctx._address_context(bv, int(addr)) or {}
    except Exception:
        return None
    string = context.get("string") or {}
    value = string.get("value") if isinstance(string, dict) else None
    return str(value) if value else None


def _semantics_name(section) -> str:
    # BN's SectionSemantics is an IntEnum, so str() yields the NUMBER ("1"), not
    # the member name -- read `.name` (verified against a live BN 5.x view).
    sem = getattr(section, "semantics", None)
    return str(getattr(sem, "name", "") or "")


def _section_is_code_like(bv, addr: int) -> bool | None:
    """Is *addr* inside a section BN considers code? None == no section knowledge.

    Semantics first (``ReadOnlyCode`` vs ``ReadOnly/ReadWriteData``), then the
    section name as a fallback for views whose semantics are all ``Default``.
    """
    try:
        secs = list(bv.get_sections_at(addr) or [])
    except Exception:
        return None
    if not secs:
        return None
    verdict: bool | None = None
    for sec in secs:
        sem = _semantics_name(sec)
        if "ReadOnlyCode" in sem:
            return True
        name = str(getattr(sec, "name", "") or "").lower()
        if name in _CODE_SECTION_NAMES or name.startswith(".plt"):
            return True
        if "Data" in sem or any(marker in name for marker in _DATA_SECTION_MARKERS):
            verdict = False
    return verdict


# --- #466 cross-target virtual-call resolution -------------------------------

def _vc_const(expr):
    """Integer constant of a MLIL CONST/CONST_PTR expr, else None."""
    if expr is None or "CONST" not in _op(expr):
        return None
    c = getattr(expr, "constant", None)
    try:
        return int(c)
    except (TypeError, ValueError):
        return None


def _vc_var(expr):
    """The Variable an MLIL_VAR expr reads (following a lone value), else None."""
    if expr is None:
        return None
    if _op(expr) == "MLIL_VAR":
        return getattr(expr, "src", None)
    return None


def _vc_vkey(var):
    return (getattr(var, "identifier", None), str(getattr(var, "name", var)))


def _vc_def_ins(instrs, var, before_addr):
    """The highest-address instruction below *before_addr* that defines *var* -- its
    reaching definition under physical order -- whether a SET_VAR (dest match) or a
    CALL (output-register match). None if none."""
    if var is None:
        return None
    vkey = _vc_vkey(var)
    best, best_addr = None, -1
    for ins in instrs:
        a = int(getattr(ins, "address", -1))
        if a >= before_addr or a <= best_addr:
            continue
        d = getattr(ins, "dest", None)
        is_def = False
        if _op(ins) in ("MLIL_SET_VAR", "MLIL_SET_VAR_FIELD") and d is not None:
            is_def = _vc_vkey(d) == vkey
        elif "CALL" in _op(ins):
            is_def = any(_vc_vkey(o) == vkey for o in (getattr(ins, "output", None) or []))
        if is_def:
            best, best_addr = ins, a
    return best


def _vc_slot_and_factory(caller, call_ins, ptr):
    """(slot_offset, factory_symbol|None) for a vtable-dispatch MLIL call
    ``[vtable + off](...)`` where ``vtable = [obj]`` and ``obj = factory()``.
    None when the call's target is not a vtable-slot load (a direct/register call
    with no vtable slot). The factory is best-effort provenance; slot_offset is the
    load-bearing result and is returned even when the factory can't be traced."""
    # Build the instruction list once -- reused by the aarch64 def-hop below and by
    # the factory trace.
    instrs = None
    call_addr = int(getattr(call_ins, "address", 0))
    dest = getattr(call_ins, "dest", None)
    # #544: on non-folded ISAs (aarch64, etc.) BN does not inline the vtable load
    # into the call target. The dispatch is two instructions -- `xN = [vtable + off]`
    # (a SET_VAR whose src is a LOAD) then `CALL xN` -- so `dest` is an MLIL_VAR, not
    # a LOAD. Follow the call-dest variable ONE reaching-def hop; if that def is a
    # SET_VAR/SET_VAR_FIELD whose src is a LOAD, use that LOAD as the dispatch load.
    # x86 (dest already a LOAD) takes the fast path unchanged.
    if dest is not None and "LOAD" not in _op(dest):
        v = _vc_var(dest)
        if v is not None:
            instrs = list(caller.mlil.instructions)
            d = _vc_def_ins(instrs, v, call_addr)
            if d is not None and _op(d) in ("MLIL_SET_VAR", "MLIL_SET_VAR_FIELD"):
                dsrc = getattr(d, "src", None)
                if dsrc is not None and "LOAD" in _op(dsrc):
                    dest = dsrc
    # Genuine register-indirect (non-vtable) calls still fall through this guard.
    if dest is None or "LOAD" not in _op(dest):
        return None
    addr_expr = getattr(dest, "src", None)
    if addr_expr is None:
        return None
    off, base = 0, addr_expr
    if _op(addr_expr) == "MLIL_ADD":
        rc = _vc_const(getattr(addr_expr, "right", None))
        if rc is not None:
            off, base = rc, getattr(addr_expr, "left", None)
        else:
            # MLIL_ADD is commutative; BN usually canonicalizes the constant to the
            # right, but not always -- handle `[const + base]` too so a vtable slot
            # with the offset on the LEFT is not mis-resolved to the whole ADD expr.
            lc = _vc_const(getattr(addr_expr, "left", None))
            if lc is not None:
                off, base = lc, getattr(addr_expr, "right", None)
    # Best-effort factory trace: base (the vtable) is `[obj]`; obj is `factory()`.
    factory = None
    try:
        if instrs is None:
            instrs = list(caller.mlil.instructions)
        vt_def = _vc_def_ins(instrs, _vc_var(base), call_addr)
        load_src = getattr(vt_def, "src", None) if vt_def is not None else None
        if load_src is not None and "LOAD" in _op(load_src):
            obj_var = _vc_var(getattr(load_src, "src", None))
            # `obj` is defined either directly by the factory CALL, or via a copy
            # (`obj = tmp; tmp = factory()`); follow one copy hop.
            obj_def = _vc_def_ins(instrs, obj_var, call_addr)
            if obj_def is not None and _op(obj_def) in ("MLIL_SET_VAR", "MLIL_SET_VAR_FIELD"):
                obj_def = _vc_def_ins(instrs, _vc_var(getattr(obj_def, "src", None)), call_addr)
            if obj_def is not None and "CALL" in _op(obj_def):
                faddr = _vc_const(getattr(obj_def, "dest", None))
                bv = getattr(caller, "view", None)
                sym = bv.get_symbol_at(int(faddr)) if (faddr is not None and bv is not None) else None
                if sym is not None:
                    factory = str(getattr(sym, "short_name", None) or getattr(sym, "name", None))
    except Exception:
        factory = None
    return off, factory


def _resolve_virtual_call(ctx, selector, at, providers=None):
    """#466: resolve an imported abstract/interface virtual call in the CONSUMER at
    address *at* to the concrete method(s) in a PROVIDER's vtable. Reports the
    consumer callsite, the factory/singleton the object came from, the vtable slot
    offset, and each candidate (provider class, vtable entry, target method). Marks
    ambiguity when multiple provider classes implement the slot. No BNDB mutation."""
    from . import read_class
    bv = ctx._resolve_view(selector)
    at_addr = _parse_address(at)
    caller = ctx._find_function(bv, hex(at_addr), contained=True)
    if caller is None:
        raise OperationFailure("no_function", f"no function contains {hex(at_addr)}")
    call_ins = _mlil_call_at(caller, at_addr)
    if call_ins is None:
        raise OperationFailure(
            "no_call",
            f"no call instruction at {hex(at_addr)} -- pass the address of the indirect "
            f"virtual call itself (the `[*obj + slot](...)` site)")
    ptr = ctx._pointer_size(bv)
    info = _vc_slot_and_factory(caller, call_ins, ptr)
    if info is None:
        raise OperationFailure(
            "not_virtual",
            f"the call at {hex(at_addr)} is not a recognized vtable dispatch "
            f"(`[*obj + slot](...)`); a direct or plain register-indirect call has no "
            f"vtable slot to resolve")
    slot_off, factory = info
    if ptr:
        # #531: validate the slot offset before flooring it to an index. slot_off is
        # an arbitrary _vc_const from an MLIL_ADD; if it is negative or not a whole
        # multiple of the pointer size, `slot_off // ptr` would silently floor to a
        # wrong slot index and name the wrong provider method. Report it unresolved
        # (with a reason) rather than emit a bogus concrete method.
        if slot_off < 0 or (slot_off % ptr) != 0:
            return {
                "kind": "virtual_call",
                "callsite": hex(at_addr),
                "caller": str(getattr(caller, "name", "") or ""),
                "factory": factory,
                "slot_offset": hex(slot_off),
                "slot_index": None,
                "candidates": [],
                "ambiguous": False,
                "resolved": False,
                "unresolved_reason": (
                    f"slot offset {hex(slot_off)} is not a non-negative multiple of the "
                    f"pointer size ({ptr}); it cannot be mapped to a vtable slot index "
                    f"without guessing the wrong method"),
            }
        slot_index = slot_off // ptr
    else:
        slot_index = slot_off
    # Provider views: the --providers selector, else resolve within the consumer.
    if providers:
        try:
            pv = ctx._resolve_view(providers)
        except Exception as exc:
            raise OperationFailure("bad_provider", f"could not resolve --providers {providers!r}: {exc}")
        provider_views = [(pv, str(providers))]
    else:
        provider_views = [(bv, str(selector) if selector else "self")]
    candidates: list[dict[str, Any]] = []
    for pv, pname in provider_views:
        try:
            maps = read_class._rtti_symbol_maps(pv)
        except Exception:
            maps = {}
        for cls_name, syms in maps.items():
            vt = syms.get("vtable")
            vt_addr = getattr(vt, "address", None) if vt is not None else None
            if vt_addr is None:
                continue
            try:
                layout = read_class._vtable_layout(ctx, pv, int(vt_addr))
            except Exception:
                continue
            slot = next((s for s in layout.get("slots", []) if s.get("index") == slot_index), None)
            if slot is None or not slot.get("method"):
                continue
            m = slot["method"]
            # vtable_entry is the ADDRESS of the slot (where the function pointer is
            # stored: vtable + 2-word Itanium header + index*ptr); method_address is
            # the pointer's VALUE (the target method). Keeping them distinct avoids
            # the "entry == target" ambiguity.
            pptr = ctx._pointer_size(pv)
            entry_addr = int(vt_addr) + 2 * pptr + slot_index * pptr
            candidates.append({
                "provider": pname,
                "class": cls_name,
                "vtable": hex(int(vt_addr)),
                "vtable_entry": hex(entry_addr),
                "method": (m.get("display_name") or m.get("name")) if isinstance(m, dict) else None,
                "method_address": (m.get("address") if isinstance(m, dict) else None),
            })
    # If the factory names a class whose candidate is present, surface it first so an
    # unambiguous provider is obvious even when several classes share the slot shape.
    candidates.sort(key=lambda c: (factory or "") not in (c.get("class") or ""))
    return {
        "kind": "virtual_call",
        "callsite": hex(at_addr),
        "caller": str(getattr(caller, "name", "") or ""),
        "factory": factory,
        "slot_offset": hex(slot_off),
        "slot_index": slot_index,
        "candidates": candidates,
        "ambiguous": len(candidates) > 1,
        "resolved": len(candidates) == 1,
    }
