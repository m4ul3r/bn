"""Raw-ABI call evidence: per-call argument recovery, arity confidence, thunks.

One bound op -- ``function_evidence`` (`bn evidence function`) -- over the
``ctx`` seam: walk a function's LLIL calls, recover each call's arguments from
HLIL/MLIL/LLIL, annotate what the arguments point at, and say how far the
recovered arity can be trusted.

This is one of the two components `read_evidence` held with ZERO shared
functions between them (#592): everything here reaches ``il_format``'s call and
disassembly helpers, and nothing here reaches the pointer-table/RTTI/init-array
cluster that stayed behind. ``read_evidence`` re-exports the names bridge.py's
delegating shims and the sibling readers still bind through it -- see the note
there; new code imports from this module.

Outbound calls resolve through:
  * ``ctx`` -- resolution / address-context helpers relocated to the seam
    (``_resolve_view``, ``_find_function``, ``_address_context``,
    ``_normalize_code_pointer``, ``_containment_meta``);
  * ``il_format`` -- the state-free IL/disasm helpers the call scan is built on
    (``_iter_llil_instructions``, ``_il_op_name``, ``_structured_disasm_entries``,
    ``_disasm_entry``, ``_hlil_call_roots``, ``_hlil_statement_localization``,
    ``_hlil_pre_branch_condition``, ``_decompile_text``, ``_function_metadata``,
    ``_render_warnings``, ``_llil_constant_value``, ``_function_is_variadic``,
    the variadic format helpers);
  * ``_shared`` -- module-free helpers (``_parse_address``, ``_validate_count``,
    ``is_imported_function``);
  * ``read_listing`` -- ``_analysis_state_fields`` (#820).

Import direction is one-way: this module imports ``il_format``, ``read_listing``
and ``_shared`` (plus stdlib + binaryninja). It NEVER imports ``bridge`` or
``seam`` -- those import it one-way (design spec 3.2).
"""
from __future__ import annotations

import re
from typing import Any

try:
    import binaryninja as bn  # noqa: F401  (kept for parity with sibling read_* modules)
except ModuleNotFoundError:  # importable without the Binary Ninja runtime (tests, tooling)
    bn = None  # type: ignore[assignment]

from . import il_format
from ._shared import _parse_address, _validate_count, is_imported_function
from .read_listing import _analysis_state_fields


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
    this call site; if that is ambiguous or has no argument list, fall back to
    MLIL, then LLIL. An explicit empty list is still a recovered zero-argument
    call. Other candidates are returned separately (JSON-only, not shown in text).
    """
    roots = il_format._hlil_call_roots(insn)
    chosen = None
    matched = [r for r in roots if _safe_int(getattr(r, "address", None)) == int(call_addr)]
    if len(matched) == 1:
        chosen = matched[0]
    elif len(roots) == 1:
        chosen = roots[0]

    mlil = _true_mlil(insn)
    source, texts = "llil", []
    for candidate_source, node in (("hlil", chosen), ("mlil", mlil), ("llil", insn)):
        if node is not None and any(
            getattr(node, attr, None) is not None for attr in ("params", "parameters")
        ):
            source, texts = candidate_source, _il_argument_texts(ctx, node)
            break

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


def _abi_arg_register_names(bv, callee_fn) -> list[str]:
    """The platform's integer argument registers, in ABI order (#882).

    Same two-step lookup `_abi_arg_register_count` has always used -- the callee's
    own calling convention, else the platform default -- but NAMES are what the
    callee-side witness needs: it asks which argument register a body READS, not
    how many exist. A `calling_convention` that is a bare string (the test fake's
    shape, and BN's pre-analysis state) has no `int_arg_regs`, so the platform
    default answers instead."""
    cc = getattr(callee_fn, "calling_convention", None) if callee_fn is not None else None
    if cc is None:
        plat = getattr(bv, "platform", None)
        cc = getattr(plat, "default_calling_convention", None)
    return [str(reg) for reg in (getattr(cc, "int_arg_regs", None) or []) if str(reg)]


def _abi_arg_register_count(bv, callee_fn) -> int | None:
    """How many integer arguments this platform passes in registers, or None."""
    return len(_abi_arg_register_names(bv, callee_fn)) or None


_C_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Mangling prefixes that are still valid C identifiers, so `_C_IDENTIFIER_RE`
# cannot exclude them: Itanium C++ and Rust's legacy scheme (`_Z`), Rust v0
# (`_R`), D (`_D`), older Swift (`_T`). A decorated name carries implicit
# parameters a count comparison cannot see (`this`, an sret return slot), so it
# is refused whatever its provenance -- unlike the schemes that use punctuation
# (MSVC `?name@@...`, current Swift `$s...`), which the identifier rule already
# rejects.
_MANGLED_PREFIXES = ("_Z", "_R", "_D", "_T")


def _library_signature_applies(callee_fn, name: str) -> bool:
    """Is an attached library's signature evidence about THIS callee, or merely
    about something that shares its name?

    The gate the first cut of #759 lacked: it keyed on the name alone, so a
    statically linked image defining its OWN function under a name a bundled
    library also carries got every call row to it demoted on a name collision
    that says nothing about the recovery (#862 review).

    Two ways a library signature does apply, and both are needed -- measured on
    a dynamically linked C++ target, where 174 of 175 library-name matches were
    imports and the ONE local match was `__popcountdi2`, statically linked from
    libgcc and genuinely under-recovered (the true positive this whole change
    rests on, which an import-only gate would have thrown away):

    * an **imported** callee resolves to the library's symbol by definition;
    * a **reserved identifier** -- C11 7.1.3 reserves a leading ``__`` to the
      implementation -- cannot be a conforming program's own function, so a
      statically linked copy is still the library's function. The ``_`` plus an
      uppercase letter half of that rule was REMOVED: it is where the mangling
      prefixes live (``_Z``, ``_R``, ``_D``, ``_T``), so it readmitted the
      collision class it was meant to exclude.

    An ordinary-named local definition is therefore refused, which is exactly
    the collision shape. Two costs, stated plainly rather than discovered later:
    a statically linked ordinary-named library function (a static ``memcpy``) is
    out of reach here, and so is any callee whose name carries a mangling prefix,
    since ``_library_param_count`` refuses those outright whatever their
    provenance. Both are part of the residual tracked in #865.
    """
    # `is_imported_function` is the module's existing predicate for this, already
    # imported here: it reads the symbol kind by NAME, so it works against BN's
    # enum and the test fake alike (#593's class of divergence) and there is one
    # answer to "is this callee an import" rather than two.
    if is_imported_function(callee_fn):
        return True
    # A LEADING DOUBLE UNDERSCORE only. The first cut also admitted `_` plus an
    # uppercase letter, which is where every mangling prefix lives -- `_Z`
    # (Itanium, and Rust's legacy scheme), `_R` (Rust v0), `_D` (D), `_T` (older
    # Swift) -- so a LOCAL Rust-mangled function colliding with a library entry
    # was still demoted: the round-1 collision class, narrowed to the one
    # decorated scheme the `_Z` refusal did not name (#862 review round 2).
    #
    # It also justified itself with C11 7.1.3 while being applied to every
    # language. Restricting it to `__` keeps the shape it was written for (a
    # statically linked `__popcountdi2`, `__memcpy_chk`, `__errno_location`) and
    # leaves every decorated name to prove real import provenance instead. The
    # cost is a reserved single-underscore C name such as `_Exit` defined
    # locally, which is covered whenever it is an import and is not worth
    # readmitting an entire mangling scheme for.
    return name.startswith("__")


def _undecorated_name(name: str) -> bool:
    """Is *name* a plain C identifier, with no language decoration?

    The shared refusal for every name that can carry IMPLICIT parameters a
    register/count comparison cannot see -- `this` on a method, an sret return
    slot on a by-value class return. Decorated schemes: Itanium/Rust (`_Z`), Rust
    v0 (`_R`), D (`_D`), older Swift (`_T`) -- all valid C identifiers, so the
    identifier rule alone cannot exclude them -- plus the schemes that use
    punctuation (MSVC `?name@@...`, current Swift `$s...`), clone suffixes
    (`.cold`, `.part`) and versioned symbols, which it does.

    Two callers, one rule: `_library_param_count` refuses a decorated name whose
    library signature might differ by an invisible parameter, and the #882
    callee-body witness refuses it for the same reason from the other direction.
    """
    return bool(name) and not name.startswith(_MANGLED_PREFIXES) \
        and _C_IDENTIFIER_RE.match(name) is not None


def _library_param_count(bv, callee_fn, name: str) -> tuple[int, str] | None:
    """The parameter count an attached type library declares for *name*, with the
    library that declared it -- or None when no library makes a usable claim.

    This is the INDEPENDENT witness #742 lacked. That guard compares the rendered
    argument list against the callee's own recovered prototype, so a callee whose
    type was itself mis-recovered agrees with itself and keeps `authoritative`
    (#759). A bundled library signature is not derived from this binary's
    analysis, so it can contradict the recovery.

    Two refusals, both measured rather than assumed:

    * **Mangled C++ names.** `this` on a method and an sret return-slot pointer on
      a by-value class return are implicit parameters a count comparison cannot
      see, so either side can differ by one with nothing wrong. On a C++-heavy
      target 3 of 4 raw firings were exactly that; excluding mangled names took
      the false-positive rate to 0 over 13,009 comparable call rows.
    * **A variadic library signature**, which states only its fixed count.
    """
    # `_Z` FIRST, and not merely as one decoration among many: an Itanium mangled
    # name IS a valid C identifier, so the identifier rule cannot exclude it --
    # and every such name begins `_Z`, which the reserved-identifier arm of
    # `_library_signature_applies` would otherwise ADMIT, re-opening exactly the
    # implicit-parameter false positives this refusal exists to stop. The
    # identifier rule then covers the schemes that use punctuation (MSVC
    # `?name@@...`, Swift `$s...`) plus clone suffixes and versioned symbols.
    if not _undecorated_name(name):
        return None
    if not _library_signature_applies(callee_fn, name):
        return None
    # EVERY attached library, not the first that happens to name the symbol
    # (#862 review round 2 minor a): with two libraries stating different counts,
    # first-wins made both the verdict and the reported `library_source` depend on
    # `bv.type_libraries` order, and a later library that AGREED with the recovery
    # was never consulted. Libraries that disagree with each other cannot settle
    # anything, so that is a refusal rather than a coin toss.
    claims: list[tuple[int, str]] = []
    for lib in (getattr(bv, "type_libraries", None) or []):
        try:
            obj = lib.get_named_object(name)
            if obj is None:
                continue
            if bool(getattr(obj, "has_variable_arguments", False)):
                return None
            params = getattr(obj, "parameters", None)
            if params is None:
                continue
            claims.append(
                (len(list(params)), str(getattr(lib, "name", "") or "type library")))
        except Exception:  # noqa: BLE001 - a malformed library must not fail the read
            continue
    if not claims or len({count for count, _ in claims}) != 1:
        return None
    return claims[0]


# BN resolves a variable by REGISTER-STORAGE ID, not by name: `Variable.storage`
# is the register's index in the architecture's register list, while the name the
# variable carries is whatever analysis assigned to it (`p`, `result`, `rcx_1`).
# That id -- not the name -- is what maps a callee variable back to an ABI
# argument position. `VariableSourceType.RegisterVariableSourceType` as an int: a
# stack or flag variable holds no argument register and is skipped outright.
_REGISTER_VARIABLE_SOURCE_TYPE = 1

# BN's MLIL SSA USE node: a READ of a variable at one version. A phi operand is a
# MERGE INPUT rather than a use at a point in the callee's code, so only this
# operation is counted -- which is the distinction both measured failure shapes
# need (see `_callee_used_arg_position`).
_VAR_SSA_READ_OP = "MLIL_VAR_SSA"

# One physical register per family, under every width an ABI can pass an argument
# in (#865), and still the ONE register -> position map the witness resolves
# through (#882): a name it folds to a position is looked up in the architecture's
# register list for the storage id a variable carries, and BN exposes a register's
# widths as separate entries there. Membership is by family, so a 32-bit ABI whose
# `int_arg_regs` says `eax` does NOT alias to a nonexistent `rax`. AArch64's x/w
# split is a rule, not a family.
_REGISTER_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("rax", "eax", "ax", "al"),
    ("rbx", "ebx", "bx", "bl"),
    ("rcx", "ecx", "cx", "cl"),
    ("rdx", "edx", "dx", "dl"),
    ("rsi", "esi", "si", "sil"),
    ("rdi", "edi", "di", "dil"),
    ("rbp", "ebp", "bp", "bpl"),
    ("rsp", "esp", "sp", "spl"),
    *tuple((f"r{n}", f"r{n}d", f"r{n}w", f"r{n}b") for n in range(8, 16)),
)


def _arg_register_index(arg_regs: list[str]) -> dict[str, int]:
    """Register-name -> argument-position for *arg_regs*, sub-registers included.

    Every name that can reach one of those registers is a key: the exact name
    `int_arg_regs` lists, its sub-register aliases from the family it belongs to,
    and (AArch64, ARM64e) the w-form of an x-register and vice versa. One map for
    every name the witness can meet, whichever width the register is stored
    under, so either form of an argument register resolves to the same position."""
    index: dict[str, int] = {}
    for position, name in enumerate(arg_regs):
        index.setdefault(name, position)
        for family in _REGISTER_FAMILIES:
            if name in family:
                for alias in family:
                    index.setdefault(alias, position)
        if name[:1] in ("x", "w") and name[1:].isdigit():
            other = ("w" if name[0] == "x" else "x") + name[1:]
            index.setdefault(other, position)
    return index


def _arg_register_storage_positions(bv, arg_regs: list[str]) -> dict[int, int]:
    """Register-storage id -> argument position, every width included.

    The register -> position map stays :func:`_arg_register_index`; this only
    resolves each name in it to the storage id BN gives that register in THIS
    view's architecture, which is what a variable actually carries. A name the
    architecture does not know (`w2` on a view whose register list stops at the
    x-form, an alias of a register that does not exist on this platform)
    contributes nothing, so the map can never invent a position the platform does
    not have; a variable whose storage is no register at all (BN stores some
    variables against register STACK indices the register table does not carry)
    resolves to nothing for the same reason."""
    positions: dict[int, int] = {}
    registers = getattr(getattr(bv, "arch", None), "regs", None) or {}
    for name, position in _arg_register_index(arg_regs).items():
        info = registers.get(name)
        index = getattr(info, "index", None)
        if index is not None:
            positions.setdefault(int(index), position)
    return positions


def _ssa_var_reads(expr, out: list) -> None:
    """Append every SSA variable *expr* READS, recursively.

    Only ``MLIL_VAR_SSA`` nodes are uses and only they are collected: a phi
    instruction's operands are its *inputs* (a merge, not a read at a point in the
    callee's code) and a ``MLIL_SET_VAR_SSA``'s destination is the raw
    ``Variable`` it defines, with no ``operation`` of its own -- while the
    assignment's SOURCE side is reached as an operand, so the registers a write
    computes from are still counted, correctly. A node with no ``operation`` is
    skipped, which is how the test fakes model a raw variable reference."""
    if expr is None:
        return
    operation = getattr(expr, "operation", None)
    if operation is None:
        return
    if getattr(operation, "name", None) == _VAR_SSA_READ_OP:
        src = getattr(expr, "src", None)
        if src is not None:
            out.append(src)
        return
    for operand in getattr(expr, "operands", None) or []:
        _ssa_var_reads(operand, out)


def _callee_used_arg_position(callee_fn, storage_positions: dict[int, int]) -> int | None:
    """The highest argument REGISTER POSITION whose INCOMING value the callee's own
    body reads, or None when the body witnesses nothing.

    This is #882's def-use question, asked where SSA makes dominance free. The
    value a caller passed in an argument register is version 0 of the variable BN
    materialized for that register, and every read after a definition on the path
    is of a later version. So the two shapes that falsified the layout-order scan
    (#865 review) stop being readable as uses:

    * a register the body WRITES before reading it -- a compiler reusing a
      caller-saved register as scratch (`rcx` on x86-64, `x3` on AArch64) -- is
      read at the assigned version, never at version 0;
    * a register written on one path and read on another (a loop body laid out
      before its initializer) is read at the PHI version at the merge, and a phi
      operand is an input, not a use.

    ``None`` means NO CLAIM, and it is the answer for a body BN never built an
    MLIL/SSA form for at all -- an import with no implementation in the image, a
    truncated view, a function BN never analyzed. That intersection (an import
    with no body) is exactly where this witness and the #862 library cross-check
    are BOTH blind, so it must stay silent rather than guess. A body that exists
    and simply reads no incoming argument register answers ``-1`` instead: a
    measurement, not a refusal."""
    try:
        mlil = getattr(callee_fn, "mlil", None)
    except Exception:  # noqa: BLE001 - a raising accessor is a body we cannot read
        return None
    ssa = getattr(mlil, "ssa_form", None) if mlil is not None else None
    if ssa is None:
        return None
    try:
        instructions = list(getattr(ssa, "instructions", None) or [])
    except Exception:  # noqa: BLE001 - a body we cannot iterate witnesses nothing
        return None
    if not instructions:
        return None
    highest = -1
    for insn in instructions:
        reads: list = []
        _ssa_var_reads(insn, reads)
        for src in reads:
            if getattr(src, "version", None) != 0:
                continue
            var = getattr(src, "var", None)
            if var is None or int(getattr(var, "source_type", -1)) != _REGISTER_VARIABLE_SOURCE_TYPE:
                continue
            position = storage_positions.get(int(getattr(var, "storage", -1)))
            if position is not None and position > highest:
                highest = position
    return highest


def _variadic_determination(callee_fn) -> bool | None:
    """Did BN DETERMINE whether this callee is variadic, and what did it decide?

    ``True``/``False`` is a determination; ``None`` means BN never made one, and
    the two must not be collapsed. BN states ``has_variable_arguments`` as a
    ``BoolWithConfidence`` -- a value AND whether analysis ever settled it -- and
    the object is truthy by its VALUE, so ``bool(flag)`` silently reads
    "never determined" as a firm "not variadic". Measured cost of reading it that
    way: 72 of 1242 call rows on an unmutated image demoted, every one of them a
    printf-style helper BN recovered as ``T(fixed..., char argN @ rax)`` without
    marking it variadic. Its prologue spills the whole register save area, and
    those spills are honest version-0 reads -- def-use soundness cannot separate
    them from consumed arguments, only the varargs flag can, and a flag nobody
    determined separates nothing.

    So the answer is only a determination when the flag is both PRESENT and
    DETERMINED. Absent type, absent flag, zero confidence, or a confidence that
    is not a number at all -> ``None``. A flag carrying no ``confidence``
    attribute is not a ``BoolWithConfidence`` but a stated bool (a declared
    signature, a test fake), and a stated value is a determination.

    ``is_variadic`` elsewhere in this module stays the two-state
    :func:`il_format._function_is_variadic`: for a diagnostic that only reports
    what the prototype SAYS (#558) and for `arity_mismatch` (#704), "not marked
    variadic" is the right reading. It is this witness, which contradicts the
    prototype rather than reporting it, that may not guess.
    """
    func_type = getattr(callee_fn, "type", None)
    if func_type is None:
        return None
    flag = getattr(func_type, "has_variable_arguments", None)
    if flag is None:
        return None
    confidence = getattr(flag, "confidence", None)
    if confidence is not None:
        try:
            determined = int(confidence) > 0
        except (TypeError, ValueError):
            return None
        if not determined:
            return None
    try:
        return bool(getattr(flag, "value", flag))
    except Exception:  # noqa: BLE001 - an unreadable flag determined nothing
        return None


def _callee_arg_use_witness(bv, callee_fn, *, variadic: bool | None) -> tuple[int, str] | None:
    """The arity the callee's own body DEMONSTRATES by USING an incoming argument,
    with the register that witnessed it -- or None for no claim.

    The callee-side witness of #865, sound this time (#882): it needs no name, no
    library and no import, because the question is asked of the callee's own
    variables. A parameter's incoming value is version 0 of the variable BN
    materialized for its argument register, so "a parameter beyond the declared
    arity is USED" is a def-use question with an exact answer -- a scratch write
    and a write-then-read on another path both leave a definition between entry
    and the read, and are therefore different versions, not uses. The comparison
    is like-for-like: an ABI register POSITION against the declared PARAMETER
    COUNT, which is only sound because parameter *i* of an integer-argument
    prototype is the ABI's argument register *i* (`_arg_register_index` is the one
    register -> position map, sub-register widths included).

    The variables come from the SSA reads themselves, NOT from
    ``callee_fn.parameter_vars``: BN makes a variable for every argument register
    the body touches, and only the DECLARED ones are parameter variables, so a
    ``parameter_vars``-only check would be vacuous exactly where this witness is
    needed -- measured on a corpus function whose under-recovered prototype declares
    one parameter while the extra register it consumes appears in ``func.vars``.

    A measurement that lands inside the declared registers answers None as well:
    nothing is under-recovered when the body uses only the parameters the
    prototype already declares. Refusals, each leaving the row untouched rather
    than guessing:

    * a **decorated name** -- the same refusal `_library_param_count` makes, from
      the other direction: a method's implicit `this` and a by-value class
      return's sret slot are ARGUMENT REGISTERS the body legitimately reads and
      the parameter count never mentions, so a body-vs-count comparison is
      meaningless there. Measured on the C++ probe: BN's own recovered prototype
      counts `this` consistently on both sides (declared 3 / body reads 3 for a
      2-parameter method), so nothing fires today -- but a prototype from a
      library or an analyst that omits `this` would otherwise be demoted for a
      parameter that is not missing at all;
    * a callee whose **variadic flag BN did not DETERMINE**, and a callee it
      determined IS variadic -- a variadic body reads the argument registers
      through the register save area / `va_list`, so a register read is not an
      arity there, and an undetermined flag cannot tell the two apart. See
      :func:`_variadic_determination`;
    * **no ABI register list** (a stack-arguments-only platform, or a view whose
      calling convention is unknown) -- no register is an argument register, so
      there is no position to compare against, and a view whose architecture
      exposes no register-storage ids maps nothing either;
    * a **body BN built no MLIL/SSA for** -- an import with no implementation in
      the image, a truncated view, or a malformed body that must not fail a read
      the way `_library_param_count` already refuses to let a malformed library
      fail one.
    """
    # Only a DETERMINED "not variadic" licenses the comparison at all: `None` is
    # BN never having settled the question, and a guess there is the 72-row
    # false-demotion class (#882 round 2).
    if variadic is not False:
        return None
    name = str(getattr(callee_fn, "name", "") or getattr(callee_fn, "raw_name", "") or "")
    if not _undecorated_name(name):
        return None
    arg_regs = _abi_arg_register_names(bv, callee_fn)
    if not arg_regs:
        return None
    try:
        positions = _arg_register_storage_positions(bv, arg_regs)
        if not positions:
            return None
        position = _callee_used_arg_position(callee_fn, positions)
    except Exception:  # noqa: BLE001 - a body we cannot walk witnesses nothing
        return None
    if position is None or position < 0:
        return None
    return position + 1, arg_regs[position]


def _argument_arity_evidence(ctx, bv, dest_value, target, arg_source: str,
                             arguments: list[dict[str, Any]],
                             *, read_cache: dict[int, tuple[int, str] | None] | None = None
                             ) -> dict[str, Any]:
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
    prototype declares (an invented/dropped ABI-register arg -- ``arity_mismatch``).
    An explicit empty HLIL list counts as zero; an unavailable list falls back to
    lower IL and cannot establish a mismatch.

    ``arity_mismatch`` is checked ONLY when ``arg_source == "hlil"`` (#704 round 3):
    ``arguments`` falls back to MLIL, then LLIL, whenever the HLIL roots are an
    ambiguous fold (``_call_arguments``); an mlil/llil-sourced list is already
    surfaced as heuristic (never ``authoritative``) and is not even attempting to
    describe the callee's operands the way HLIL does -- on the mapped-MLIL fallback
    it is literally the CALLER's ABI registers (#661). Comparing THAT count against
    the callee's declared arity and reporting the note as an "HLIL rendered N
    argument(s)" finding would attribute a provenance the tool never established,
    reintroducing #661's defect class in prose.

    Returns ``{"arity_unknown": bool, ...}``; ``arity_unknown`` is False whenever the
    callee cannot be resolved -- ``callee_unresolved`` carries that case instead.

    ``callee_under_recovered``/``callee_read_arity``/``declared_arity``/
    ``callee_arity_note`` (#882) is the callee-side witness, and it DOES move
    confidence: the callee's body USES an argument register the prototype does not
    declare -- as an incoming value, established by parameter def-use in SSA, so a
    scratch reuse of the register and a write-then-read on another path are not
    uses -- on an HLIL-sourced list whose length agrees with that prototype (where
    ``arity_mismatch`` is silent by construction, so nothing else here would report
    it). The row demotes to ``inferred`` and keeps the observation, because a
    demotion that hides its reason is the silent demotion this module exists to
    stop. See :func:`_callee_arg_use_witness`. ``read_cache`` memoizes the
    per-CALLEE witness across the call sites of one function.
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
    has_user_type = bool(getattr(callee_fn, "has_user_type", False))
    func_type = getattr(callee_fn, "type", None)
    declared = getattr(func_type, "parameters", None)
    if declared is None:
        declared = getattr(callee_fn, "parameter_vars", None)
    try:
        declared_count = len(declared) if declared is not None else None
    except TypeError:
        declared_count = None
    is_variadic = il_format._function_is_variadic(callee_fn)
    # #759: cross-check the RECOVERED prototype against a bundled library
    # signature before either branch below trusts it. Both of them compare the
    # rendered list against `declared_count`, so an under-recovered callee agrees
    # with itself -- measured on a real target as `__popcountdi2()` rendering zero
    # arguments with `authoritative` and no mismatch, against a library that
    # declares one parameter. Recorded whenever the two disagree, including the
    # zero-vs-N case the "genuinely void callee" branch below would wave through.
    # Deliberately NOT suppressed for a user prototype. Round 1 of this review
    # asked for that precedence and it was implemented as
    # `None if has_user_type else ...`, which the dogfood then measured as
    # turning the whole cross-check OFF wherever it matters: BN sets
    # `has_user_type` on almost every function and exposes no API to clear it
    # (which is why `proto set --preview` is refused), so after a `bn save` and
    # reopen it reads True for essentially everything -- 104/104 imports and
    # 853/853 locals on a reopened view, 957/959 on a corpus database. Against a
    # saved `.bndb`, the normal case, the gate became a no-op versus base.
    #
    # The round-1 concern was that a demotion must not SILENTLY overrule an
    # analyst's statement. That is met by DISCLOSURE, not by privilege: a pinned
    # prototype IS demoted like any other, and the row carries `declared_arity`,
    # `library_arity` and `library_source`, and the text line names the
    # disagreement as the reason it withheld `authoritative` and warns that the
    # row may be contradicting a prototype the analyst pinned -- so the claim is
    # visible, checkable with `bn proto get`, and can be judged wrong for this
    # binary. The renderer owns that wording; this comment does not quote it. Suppression bought the nuance
    # at the price of the feature everywhere it matters.
    library = _library_param_count(
        bv, callee_fn, str(getattr(callee_fn, "name", "") or ""))
    if (
        library is not None
        and declared_count is not None
        and not is_variadic
        and library[0] != declared_count
    ):
        evidence["prototype_unverified"] = True
        evidence["declared_arity"] = declared_count
        evidence["library_arity"] = library[0]
        evidence["library_source"] = library[1]
    # #865/#882: the witness for the shape #862 cannot reach -- nothing OUTSIDE
    # this binary settles the arity (no attached library names the callee, the
    # name is decorated, or it is an ordinary-named local definition), so the
    # recovered prototype is compared against something inside it instead: the
    # callee's own parameter variables. A body that USES an argument register the
    # prototype does not declare takes more arguments than the recovery admits,
    # which is a positive reason to distrust it -- not the absence of a reason to
    # trust it, which is what demoting on `has_user_type`/`is_import` alone
    # amounts to (#648's own precedent, where `memset`'s bundled 3-parameter
    # prototype EARNS `authoritative`).
    #
    # Gated to the vacuous AGREEMENT the issue reports: the rendered list matches
    # the declared arity, so `arity_mismatch` is silent and the row would
    # otherwise claim `authoritative` off a comparison of the recovery against
    # itself. Where the rendered list already disagrees, `arity_mismatch` demotes
    # and this adds nothing. Only an HLIL-sourced list is compared, for #704
    # round 3's reason: an MLIL/LLIL list is the CALLER's ABI registers (#661) and
    # its count is not a claim about the callee's operands.
    #
    # The witness takes the THREE-state varargs answer, not `is_variadic`: it
    # contradicts the recovered prototype, so it may only speak where BN actually
    # determined the callee is not variadic (#882 round 2).
    variadic = _variadic_determination(callee_fn)
    callee_use: tuple[int, str] | None
    if read_cache is None:
        callee_use = _callee_arg_use_witness(bv, callee_fn, variadic=variadic)
    else:
        # A dispatch function calls the same few callees hundreds of times, and the
        # witness is per CALLEE, not per call site -- pay it once per callee (#865).
        key = int(getattr(callee_fn, "start", 0) or 0)
        if key not in read_cache:
            read_cache[key] = _callee_arg_use_witness(bv, callee_fn, variadic=variadic)
        callee_use = read_cache[key]
    if (
        arg_source == "hlil"
        and callee_use is not None
        and declared_count is not None
        and callee_use[0] > declared_count
        and len(arguments) == declared_count
    ):
        used_arity, used_register = callee_use
        evidence["callee_under_recovered"] = True
        evidence["callee_read_arity"] = used_arity
        evidence["declared_arity"] = declared_count
        evidence["callee_arity_note"] = (
            f"the callee's body USES `{used_register}` (ABI argument "
            f"{used_arity - 1}, 0-based) as an incoming argument, a parameter "
            f"beyond the {declared_count} its recovered prototype declares. "
            f"Established by parameter def-use in SSA, so a register a path "
            f"writes before reading (a scratch reuse) and a value merged at a phi "
            f"are not uses. The rendered argument list may therefore be "
            f"under-recovered and this row's `arguments` confidence is withheld "
            f"from `authoritative`: check `bn proto get` on the callee and "
            f"`bn disasm --linear` at the call before trusting the list"
        )
    if declared_count is not None and (declared_count > 0 or has_user_type):
        # User prototypes also establish zero arity, but do not guarantee that
        # HLIL recovered that many arguments. Only compare an actual HLIL list;
        # lower-IL fallbacks retain their heuristic provenance (#704, #742).
        if arg_source == "hlil" and not is_variadic and len(arguments) != declared_count:
            evidence["arity_mismatch"] = True
            evidence["declared_arity"] = declared_count
        return evidence
    if is_variadic:
        return evidence          # a declared variadic with no fixed parameters
    # Zero declared parameters yet HLIL rendered arguments: the list is BN's
    # register guess, not the callee's signature. A genuinely void callee rendering
    # zero arguments agrees with its prototype and is left alone.
    if declared_count == 0 and not arguments:
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
    # #865: memo for the callee-side read witness, keyed by callee entry address.
    callee_read_cache: dict[int, tuple[int, str] | None] = {}
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
        arity = _argument_arity_evidence(ctx, bv, dest_value, target, arg_source, arguments,
                                         read_cache=callee_read_cache)
        if arity.get("callee_unresolved"):
            argument_confidence = "heuristic"
        elif (
            arity["arity_unknown"]
            or arity.get("arity_mismatch")
            # #882: the callee's own body USES an argument register its recovered
            # prototype does not declare -- a def-use fact in the callee's own SSA,
            # so a scratch reuse of the register and a write-then-read on another
            # path are not uses (the two artifacts that falsified the layout-order
            # scan in #865's review). The row keeps `callee_read_arity`,
            # `declared_arity` and `callee_arity_note` saying what was observed:
            # a demotion that hides its reason is the silent demotion this module
            # exists to stop.
            or arity.get("callee_under_recovered")
            # #759: a bundled library signature contradicting the recovered
            # prototype is a positive reason to distrust it, so the row stops
            # claiming authority and carries `library_arity` saying why.
            or arity.get("prototype_unverified")
        ) and argument_confidence == "authoritative":
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
    # reach the fallback below, unaffected. #704 round 3: suppressing
    # `is_candidate` here turned a false positive (flagging a local forwarder
    # as a PLT/GOT-style thunk) into a false negative (a genuine local
    # `j_`-style veneer became invisible). Recording `target` on this path --
    # while leaving `is_candidate` False -- keeps the fix (no unearned thunk
    # claim) and closes the visibility gap: `evidence function` still surfaces
    # the forwarding, without asserting a thunk/veneer verdict the tool has
    # not established for a LOCAL destination.
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
            if result["target"] is None:
                result["target"] = target
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
    # #827 item 8: the same shared validation (and wording) every other paged read
    # uses. A raw-socket / `py exec` client reaches this handler directly, so the
    # bridge re-enforces the CLI's argparse contract (`_positive_int` /
    # `_non_negative_int`) rather than raising ad-hoc messages of its own.
    context = _validate_count(context, label="context", minimum=0)
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    # #471 slicing/windowing controls so a large call-heavy dispatch function can be
    # inspected in bounded chunks instead of reading a full spill.
    slicing = bool(offset or limit is not None or address_window is not None)
    # #622: the Pseudo-C decompile is expensive and is needed ONLY for the
    # decompiler warnings and the C++ `this` caveat -- not for the call list. A
    # paged read therefore defers it and DISCLOSES the deferral, so the missing
    # warnings/caveat are never silently absent. The unsliced read keeps the
    # original order (decompile -> calls -> variadic hoist) and full fidelity.
    decompile_deferred = slicing
    warnings: list[str] = []
    if not decompile_deferred:
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
        # #882: TEXT-mode disclosure for the callee-side witness -- the reason the
        # row's `arguments` confidence was withheld from `authoritative`, in the
        # same words the row carries. Hoisted like the variadic warning above, and
        # for the same reason: computed from the full call set BEFORE slicing, so
        # the caveat is visible on whichever page is requested (an unsliced row
        # would show it, and a sliced page must not lose it) and a reader of the
        # card sees what JSON says.
        if call.get("callee_arity_note"):
            target = call.get("target")
            fn_entry = target.get("function") if isinstance(target, dict) else None
            callee = str((fn_entry or {}).get("name") or "the callee")
            warnings.append(
                f"{call.get('address', '?')}: NOTE -- {callee}: "
                f"{call['callee_arity_note']}"
            )
    if decompile_deferred:
        warnings.append(
            "Pseudo-C decompile deferred for this sliced read (offset/limit/address "
            "window): decompiler warnings and the C++ `this` caveat were not "
            "collected; an unsliced read gives full fidelity"
        )
    # #471: only sort by address when a slice is actually requested -- the default
    # (unsliced) output keeps its original IL/discovery order so existing consumers
    # see no change.
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
        # #819: the #275 discriminator. This card's rows live under `calls` (the
        # documented leaf, and what every renderer and consumer reads), not under
        # `items` -- the card is object-shaped (function + metadata + thunk + a
        # call list with its own #471 paging quad) and `calls` is the heaviest
        # array in any read here, so it is not duplicated under a second key.
        # `kind` is what was missing: without it a generic consumer cannot tell
        # this payload apart from any other object-shaped read.
        "kind": "function_evidence",
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
        # #820: a --quick view answers here (the matrix marks this op `partial`),
        # so disclose that the ABI/argument recovery was read off a function BN
        # has not analyzed -- the call list is real, its fidelity is not.
        **_analysis_state_fields(bv),
    }
    if decompile_deferred:
        # #622: additive honesty field -- present only when the decompile was
        # skipped, so an unsliced read's payload is byte-for-byte unchanged.
        # TEXT mode surfaces it too: `formatters._render_function_evidence_text`
        # prints the deferral sentence this function already appended to
        # `warnings`, and falls back to stating the flag itself when that list
        # arrived absent, empty or skewed.
        result["decompile_deferred"] = True
    # #626: annotate a mid-function (interior) request the same way the decompile
    # READs do (#193 Part 4). Inlined via the seam's `_containment_meta` rather
    # than importing read_decompile's `_annotate_containment`, to keep the
    # read_evidence -> read_decompile module import one-way (read_decompile
    # already lazy-imports read_evidence).
    meta = ctx._containment_meta(identifier, func)
    if meta:
        result["resolved_from"] = meta
    return result
