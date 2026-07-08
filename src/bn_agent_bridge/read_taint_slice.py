"""Taint entry point + interprocedural backward-slice machinery.

The ``taint`` op and the SSA backward-slice walker that used to live on
``BinaryNinjaBridge`` move here as module-level free functions, each taking the
``BridgeContext`` seam (``ctx``) in place of ``self``. ``BinaryNinjaBridge``
keeps a thin delegating shim for every name the test suite / op binders
reference (``_taint``, ``_backward_slice``, ``_build_backward_trace``,
``_is_parameter_ssa_var``, ``_resolve_callee``, ``_resolve_thunk``,
``_extract_dest_address``, ``_find_return_vars``, ``_ssa_vars_from``).

Outbound calls resolve through:
  * ``ctx`` -- resolution helpers relocated to the seam (``_resolve_view``,
    ``_find_function``);
  * ``il_format`` -- the state-free IL helpers (``_il_op_name``,
    ``_iter_il_instructions``);
  * ``vars`` -- variable discovery (``_find_variable_selector``);
  * ``taint_engine`` (imported as ``_taint``, matching ``bridge.py``) -- the
    taint model loader + engine + locator/call-target/thunk resolution
    (``load_models``, ``TaintEngine``, ``parse_locator``, ``resolve_call_target``,
    ``follow_thunk``, ``extract_dest_address``, ``TaintError``);
  * ``_shared`` -- module-free helpers (``_parse_address``, ``OperationFailure``).

Import direction is one-way: this module imports ``il_format``, ``vars``,
``taint_engine``, and ``_shared`` (plus stdlib + binaryninja). It NEVER imports
``bridge`` or ``seam`` -- those import THIS module one-way (design spec §3.2).
"""
from __future__ import annotations

import re
from typing import Any

import binaryninja as bn  # noqa: F401  (kept for parity with sibling read_* modules)
from binaryninja import SSAVariable

from . import il_format
from . import taint_engine as _taint
from . import vars as vars_mod
from ._shared import OperationFailure, _parse_address
from .bridge_state import require_analysis
from .read_taint_models import build_catalog


def _taint_op(ctx, selector, params: dict[str, Any]):
    bv = ctx._resolve_view(selector)
    require_analysis(bv, "Taint analysis")
    direction = str(params.get("direction", "forward"))
    func = ctx._find_function(bv, params["function"])
    try:
        # `user_models` carries project-internal wrapper models supplied via
        # `taint --models <file>` (#317); merged over the builtin/override DB.
        models = _taint.load_models(extra=params.get("user_models"))
    except _taint.TaintError as exc:
        # A broken builtin DB or BN_TAINT_MODELS override is now loud instead
        # of silently producing false negatives (#97).
        raise OperationFailure("unsupported", str(exc)) from exc
    # #415: capture the overlay disclosure from the SAME load that produced
    # `models` (before the long analysis), so a model file created/deleted
    # mid-run can't desync the reported sources from what was actually used.
    model_sources = _taint.model_overlay_sources(
        params.get("user_models"), user_models_path=params.get("user_models_path"))

    def _find_variable(fn, sel):
        var, _is_param = vars_mod._find_variable_selector(fn, sel)
        return var

    engine = _taint.TaintEngine(
        bv,
        models,
        find_variable=_find_variable,
        unknown_call_policy=str(params.get("unknown_call", "conservative")),
        resolve_map=params.get("resolve_map") or {},
    )
    try:
        if direction == "forward":
            locators = [_taint.parse_locator(s) for s in (params.get("sources") or [])]
            if not locators:
                raise _taint.TaintError("forward taint needs at least one --source")
            result = engine.forward(
                func, locators,
                max_depth=int(params.get("max_depth", 8)),
                enabled_sink_classes=set(params.get("enabled_sink_classes") or []),
            )
        elif direction == "backward":
            locators = [_taint.parse_locator(s) for s in (params.get("sinks") or [])]
            if not locators:
                raise _taint.TaintError("backward taint needs at least one --sink")
            result = engine.backward(func, locators, max_depth=int(params.get("max_depth", 8)))
        else:
            raise OperationFailure("unsupported", f"unknown taint direction: {direction}")
    except _taint.TaintError as exc:
        raise OperationFailure("unsupported", str(exc)) from exc
    # #415: disclose which taint-model overlays were in effect for this run, so an
    # agent can confirm a project-local --models / BN_TAINT_MODELS file landed
    # (load_models reads them per request -- no bridge restart needed). Uses the
    # value captured at load time above so it reflects what was actually merged.
    if isinstance(result, dict):
        result["model_sources"] = model_sources
    return result


def _taint_models_op(ctx, selector, params: dict[str, Any]):
    """`taint models`: dump the model catalog; with a target, annotate which
    modeled symbols are present in the binary (+ callsite counts)."""
    present_only = bool(params.get("present"))
    want_callsites = bool(params.get("callsites"))
    if (present_only or want_callsites) and selector is None:
        raise OperationFailure(
            "unsupported",
            "--present/--callsites need a target: presence is defined against a loaded binary")
    try:
        models = _taint.load_models(extra=params.get("user_models"))
    except _taint.TaintError as exc:
        raise OperationFailure("unsupported", str(exc)) from exc
    catalog = build_catalog(models, role=params.get("role"), sink_class=params.get("class"))
    catalog["overlays"] = _taint.model_overlay_sources(
        params.get("user_models"), user_models_path=params.get("user_models_path"))
    catalog["items"] = ([{"role": "source", **s} for s in catalog["sources"]]
                        + [{"role": "sink", **e}
                           for lst in catalog["sinks_by_class"].values() for e in lst]
                        + [{"role": "propagator", **p} for p in catalog["propagators"]])
    if selector is not None:
        bv = ctx._resolve_view(selector)
        present_keys, counts = _present_models(bv, models, want_callsites)
        _annotate_presence(catalog, present_keys, counts, present_only)
    return catalog


def _present_models(bv, models, want_callsites):
    """Map the binary's symbols to model keys via the engine's own normalization;
    return (present model-key set, {model-key: {callsites, addresses?}})."""
    present: set[str] = set()
    counts: dict[str, dict[str, Any]] = {}
    # #472: several distinct binary symbol spellings can normalize to the SAME
    # model key (memcpy + memcpy@plt -> memcpy; _Znwm + operator new -> Znwm), so
    # accumulate their callsites into one slot instead of overwriting -- an
    # assignment let whichever spelling was seen last (set iteration order) clobber
    # the rest, dropping a present sink to "(0 callsites)". Dedup addresses so an
    # alias resolving to the same site is not double-counted.
    seen_addrs: dict[str, set[str]] = {}
    names: set[str] = set()
    for fn in getattr(bv, "functions", []) or []:
        names.add(str(getattr(fn, "name", "")))
    for sym in (getattr(bv, "get_symbols", lambda: [])() or []):
        names.add(str(getattr(sym, "name", "")))
    for nm in names:
        key, _model = _taint.lookup_model(models, nm)
        if not key:
            continue
        present.add(key)
        if want_callsites:
            slot = counts.setdefault(key, {"callsites": 0, "addresses": []})
            seen = seen_addrs.setdefault(key, set())
            for sym in (getattr(bv, "get_symbols_by_name", lambda n: [])(nm) or []):
                for ref in (getattr(bv, "get_code_refs", lambda a: [])(
                        getattr(sym, "address", 0)) or []):
                    a = hex(int(getattr(ref, "address", 0)))
                    if a not in seen:
                        seen.add(a)
                        slot["addresses"].append(a)
            slot["callsites"] = len(slot["addresses"])
        else:
            counts.setdefault(key, {"callsites": None})
    return present, counts


def _annotate_presence(catalog, present_keys, counts, present_only):
    def _mark(entry):
        key = entry["symbol"]
        entry["present"] = key in present_keys
        info = counts.get(key) or {}
        if info.get("callsites") is not None:
            entry["callsites"] = info["callsites"]
            if "addresses" in info:
                entry["addresses"] = info["addresses"]
        return entry["present"]
    catalog["sources"] = [e for e in catalog["sources"] if (_mark(e) or not present_only)]
    for cls, lst in list(catalog["sinks_by_class"].items()):
        kept = [e for e in lst if (_mark(e) or not present_only)]
        if kept:
            catalog["sinks_by_class"][cls] = kept
        else:
            del catalog["sinks_by_class"][cls]
    catalog["propagators"] = [e for e in catalog["propagators"] if (_mark(e) or not present_only)]
    catalog["items"] = [e for e in catalog["items"] if (_mark(e) or not present_only)]


def _ssa_vars_from(vars_list: list) -> list[SSAVariable]:
    return [v for v in vars_list if isinstance(v, SSAVariable)]


def _ssa_label(ssa_var) -> str:
    """Stable analyst-friendly label for an SSA var, e.g. ``var_10#3`` -- avoids
    the verbose ``<SSAVariable ...>`` repr some BN builds emit for ``str()`` and
    keeps trace JSON machine-consumable (#162)."""
    base = getattr(ssa_var, "var", ssa_var)
    name = getattr(base, "name", None)
    version = getattr(ssa_var, "version", None)
    if name and version is not None:
        return f"{name}#{version}"
    return str(ssa_var)


def _const_int(expr) -> int | None:
    if expr is None or "CONST" not in il_format._il_op_name(expr):
        return None
    c = getattr(expr, "constant", None)
    try:
        return int(c) if c is not None else None
    except (TypeError, ValueError):
        return None


def _addr_base_offset_label(expr, depth: int = 0) -> tuple[str | None, int | None]:
    """Best-effort (base_label, offset) for a load address expression: a base
    pointer var plus a constant field offset. Either may be None when the address
    isn't a simple ``base [+ const]`` shape."""
    if expr is None or depth > 6:
        return None, None
    op = il_format._il_op_name(expr)
    if "ADD" in op or "SUB" in op:
        left = getattr(expr, "left", None)
        right = getattr(expr, "right", None)
        rc = _const_int(right)
        if rc is not None:
            bl, _ = _addr_base_offset_label(left, depth + 1)
            return bl, (rc if "ADD" in op else -rc)
        lc = _const_int(left)
        if lc is not None and "ADD" in op:
            br, _ = _addr_base_offset_label(right, depth + 1)
            return br, lc
        return None, None
    reads = _ssa_vars_from(getattr(expr, "vars_read", []) or [])
    if len(reads) == 1:
        return _ssa_label(reads[0]), 0
    return None, None


def _load_expr_of(def_insn):
    """The load expression a definition reads from, or None. Handles both MLIL
    forms: a top-level load def, and the common ``x = [addr]`` SET_VAR whose
    ``.src`` is the load."""
    if "LOAD" in il_format._il_op_name(def_insn):
        return def_insn
    src = getattr(def_insn, "src", None)
    if src is not None and "LOAD" in il_format._il_op_name(src):
        return src
    return None


def _field_load_meta(load_expr) -> tuple[str | None, int | None, int | None]:
    """(base_label, offset, width) for a load expression, best-effort. Lets a
    field load (`*(obj + off)`) carry structured metadata in the trace instead of
    just a `memory_load` reason (#162)."""
    width = getattr(load_expr, "size", None)
    base, offset = _addr_base_offset_label(getattr(load_expr, "src", None))
    return base, offset, (int(width) if isinstance(width, int) else None)


def _is_address_of(expr) -> bool:
    return "ADDRESS_OF" in il_format._il_op_name(expr)


def _arg_register(caller_func, arg_index: int) -> str | None:
    """The calling-convention integer-arg register for *arg_index* (e.g. `x1`,
    `rsi`, `r1`), or None when the convention isn't recoverable (#166)."""
    try:
        cc = getattr(caller_func, "calling_convention", None)
        regs = list(getattr(cc, "int_arg_regs", []) or [])
        if 0 <= arg_index < len(regs):
            return str(regs[arg_index])
    except Exception:
        return None
    return None


def _int_arg_reg_count(caller_func) -> int | None:
    """Number of integer-argument registers in the function's calling convention
    (6 on x86-64 SysV, 8 on AArch64, 4 on ARM/MIPS-o32), or None if the convention
    itself is unrecoverable. An arg at or beyond this index is passed on the STACK,
    where BN's MLIL/HLIL call model often omits it (#324). Returns **0** for a pure
    stack-argument ABI (i386 cdecl has no integer-arg registers) -- distinct from
    None -- so the stack-passed caveat still fires there (previously an empty
    int_arg_regs collapsed to None and the caveat was silently suppressed on i386)."""
    try:
        cc = getattr(caller_func, "calling_convention", None)
        if cc is None:
            return None
        return len(list(getattr(cc, "int_arg_regs", []) or []))
    except Exception:
        return None


def _llil_sp_offset(expr, sp_name: str):
    """If an LLIL address expr is the stack pointer `sp` or `sp + const` (const>=0),
    return the const offset; else None. Recognizes the outgoing-argument stack slot
    an `LLIL_STORE` writes (#489)."""
    if expr is None or not sp_name:
        return None
    op = il_format._il_op_name(expr)
    if op.endswith("_REG") or op == "LLIL_REG":
        r = getattr(expr, "src", None)
        return 0 if str(getattr(r, "name", r)) == sp_name else None
    if op.endswith("_ADD") or op == "LLIL_ADD":
        left = getattr(expr, "left", None)
        right = getattr(expr, "right", None)
        if _llil_sp_offset(left, sp_name) == 0:
            c = il_format._llil_constant_value(right)
            if c is not None and c >= 0:
                return int(c)
    return None


def _stack_arg_store_offsets(func, target_addr: int, *, max_off: int = 0x100) -> list[int]:
    """Distinct offsets of outgoing stack-arg stores (`LLIL_STORE` to `sp+const`,
    0<=const<max_off) in the LLIL basic block of the call at *target_addr*, at or
    before the call (the branch-delay-slot store at ==addr is included). These are
    the values pushed for the call. Empty on any BN-API shortfall so it never
    fabricates a signal (#489)."""
    try:
        llil = getattr(func, "low_level_il", None)
        sp = str(getattr(getattr(func, "arch", None), "stack_pointer", "") or "")
        if llil is None or not sp:
            return []
        offsets: set[int] = set()
        for bb in llil:
            insns = list(bb)
            addrs = [int(getattr(i, "address", -1)) for i in insns]
            if target_addr not in addrs:
                continue
            call_k = max(k for k, a in enumerate(addrs) if a == target_addr)
            # Start AFTER the nearest preceding call in the block: an earlier call's
            # outgoing stack-arg stores must not be misattributed to THIS call (the
            # dominant false-positive source found in review, #489).
            start = 0
            for k in range(call_k):
                if "CALL" in il_format._il_op_name(insns[k]):
                    start = k + 1
            for k in range(start, call_k + 1):
                ins = insns[k]
                if il_format._il_op_name(ins) != "LLIL_STORE":
                    continue
                off = _llil_sp_offset(getattr(ins, "dest", None), sp)
                if off is not None and 0 <= off < max_off:
                    offsets.add(off)
            break
        return sorted(offsets)
    except Exception:
        return []


# A printf conversion specifier: `%` + optional flags/width/precision + optional
# length modifier + a conversion char. Matches %d, %s, %08X, %-5.2f, %llu, %p, ...
# The space flag is deliberately excluded from the flag class so a natural string
# like "50% off" / "5% charge" (`% o`/`% c`) can't coincidentally match (#489).
_FMT_SPEC_RE = re.compile(r"%[-+#0-9.*]*(?:hh|h|ll|l|L|q|z|j|t)?[diouxXeEfFgGaAcsp]")

# Well-known FIXED-ARITY libc/BSD functions. A truncated-variadic note must never
# fire on these even when they take a format-SHAPED string literal (e.g. `strlen`
# measuring a format string before an sprintf/log elsewhere) -- that was the one
# residual false positive after the format-string gate (#489 review). Matched on
# the modeled callee name (decorations stripped).
_KNOWN_FIXED_ARITY = frozenset({
    "strlen", "strnlen", "strchr", "strrchr", "strchrnul", "strstr", "strcasestr",
    "strcmp", "strncmp", "strcasecmp", "strncasecmp", "strcoll", "strspn", "strcspn",
    "strpbrk", "strcpy", "strncpy", "strlcpy", "strcat", "strncat", "strlcat",
    "strdup", "strndup", "index", "rindex", "strtok", "strtok_r",
    "atoi", "atol", "atoll", "atof", "strtol", "strtoul", "strtoll", "strtoull",
    "strtod", "memcpy", "memmove", "memset", "memcmp", "memchr", "bcopy", "bcmp",
    "bzero", "puts", "fputs", "perror",
})


def _mlil_const_addr(expr):
    """The constant address an MLIL expr points at (a const / const-ptr / import),
    else None."""
    if expr is None:
        return None
    op = il_format._il_op_name(expr)
    if "CONST_PTR" in op or op == "MLIL_CONST" or "IMPORT" in op:
        c = getattr(expr, "constant", None)
        try:
            return int(c) if c is not None else None
        except (TypeError, ValueError):
            return None
    return None


def _read_cstring(bv, addr: int, *, cap: int = 256) -> str | None:
    try:
        data = bytes(bv.read(int(addr), cap) or b"")
    except Exception:
        return None
    if not data:
        return None
    nul = data.find(b"\x00")
    raw = data[:nul] if nul >= 0 else data
    try:
        return raw.decode("latin-1")
    except Exception:
        return None


def _params_have_format_string(bv, params) -> bool:
    """True when a recovered arg is a const pointer to a rodata string containing a
    printf conversion specifier -- a strong POSITIVE signal that the callee is a
    variadic (printf/log-family) function, so its dropped stack stores really are
    variadic args and not a fixed-arity libc/BSD call's frame noise. This is the
    discriminator that keeps the #489 frontier from firing on strchr/memset/bcopy/
    __gedf2 etc., whose recovered args have no format string (#489 review)."""
    if bv is None:
        return False
    for p in params or []:
        addr = _mlil_const_addr(p)
        if addr is None:
            continue
        s = _read_cstring(bv, addr)
        if s and _FMT_SPEC_RE.search(s):
            return True
    return False


def _call_model_truncation_note(bv, func, call_insn, target_addr: int,
                                params: list, callee_name: str | None) -> str | None:
    """#489: a PROACTIVE truncation disclosure. BN auto-types an internal variadic /
    many-arg callee as fixed-arity, so the call site's MLIL `.params` omit the
    stack-passed args -- and unlike the reactive out-of-range message (#324/#488),
    an analyst asking only for in-range args gets no hint the drop exists. Fire when
    BN recovered NO stack params (`len(params) <= int_arg_regs`) yet LLIL shows
    outgoing stack-arg stores feeding the call. Conservative: silent when the arch's
    arg-reg count is unknown, or when BN already recovered stack params. None = no
    disclosure."""
    reg_count = _int_arg_reg_count(func)
    if reg_count is None or len(params) > reg_count:
        return None
    # A known fixed-arity libc/BSD callee is never a truncated variadic, even if it
    # takes a format-shaped string arg (the residual FP the format-string gate alone
    # let through -- e.g. strlen of a format literal on MIPS, #489 review).
    _base = str(callee_name or "").split("@", 1)[0].lstrip("_")
    if _base in _KNOWN_FIXED_ARITY:
        return None
    # POSITIVE variadic signal (review): only fire when a recovered arg is a format
    # string. Offset shape alone can't tell a truncated variadic from a fixed-arity
    # libc/BSD call (strchr/memset/bcopy/__gedf2) whose frame happens to have low
    # contiguous sp stores -- those had a ~2-10% false-positive rate on real
    # firmware. Requiring a caller-passed format string makes "auto-typed
    # variadic-as-fixed" actually plausible and excludes those FPs (their args have
    # no format string). Scope: this catches the printf/log/err family (format
    # passed by the caller); it deliberately does NOT flag a variadic whose format
    # is INTERNAL (a `sprintf(buf,"...",...)` wrapper) or a non-format variadic --
    # a no-false-positive trade, since those can't be confirmed from the call site.
    if not _params_have_format_string(bv, params):
        return None
    raw = _stack_arg_store_offsets(func, target_addr)
    # Distinguish deliberate outgoing-argument pushes from isolated local spills:
    # require a CONTIGUOUS pointer-width run of >=2 stores STARTING near sp (the
    # low outgoing-arg region -- sp+0 on AAPCS, up to the 16-byte arg-save area on
    # MIPS-o32). An isolated store at sp+0x8/0x28 is a local, not an arg -- staying
    # silent there is the no-false-positive direction for a proactive frontier.
    try:
        word = int(getattr(getattr(func, "arch", None), "address_size", 0)) or 4
    except Exception:
        word = 4
    offsets: list[int] = []
    if len(raw) >= 2 and raw[0] <= 0x10:
        offsets = [raw[0]]
        for o in raw[1:]:
            if o == offsets[-1] + word:
                offsets.append(o)
            else:
                break
    if len(offsets) < 2:
        return None
    disp = callee_name or "the callee"
    off_list = ", ".join(f"sp+{hex(o)}" for o in offsets)
    # Pick a RUNNABLE `proto set` selector: the callee name if known, else its
    # resolved entry address (a direct call's constant dest still gives us an
    # address even when the target is unnamed). An unnamed/indirect callee with
    # neither has no runnable selector -- emit prose guidance, never the literal
    # `bn proto set the callee ...` command, which cannot be copy-pasted (#534).
    selector = callee_name or None
    if selector is None and call_insn is not None:
        try:
            addr = _taint.resolve_call_target(
                bv, call_insn, follow_thunks=True).address
        except Exception:
            addr = None
        if addr is not None:
            selector = hex(int(addr))
    if selector is not None:
        proto_name = callee_name or "<callee>"
        remedy = (
            f"declare it variadic with "
            f"`bn proto set {selector} \"<ret> {proto_name}(<fixed args>, ...)\"` "
            f"(or the full prototype) and re-run."
        )
    else:
        remedy = (
            "declare it variadic on the concrete callee once you resolve this "
            "indirect/unnamed call -- run "
            "`bn proto set <callee> \"<ret> <callee>(<fixed args>, ...)\"` against "
            "the resolved target's address or name and re-run."
        )
    return (
        f"call-model truncation (#489): {disp} was recovered with {len(params)} MLIL "
        f"argument(s) (all register-passed), but LLIL shows {len(offsets)} outgoing "
        f"stack-arg store(s) [{off_list}] feeding this call -- those stack-passed "
        f"arg(s) are DROPPED from the call model, most often a variadic callee "
        f"auto-typed as fixed-arity. `trace`/`taint`/`defuse` cannot see them until "
        f"the prototype is fixed: {remedy} Inspect the stores with "
        f"`bn il {getattr(func, 'name', '<fn>')} --view llil --ssa`."
    )


def _out_of_range_arg_msg(func, arg_index: int, n: int) -> str:
    """The precise out-of-range message for `trace --arg N`, with the #324 stack-
    passed / first-missing-register notes. Used when no #433 register fallback
    applies."""
    only = " (index 0)" if n == 1 else f" (indices 0..{n - 1})"
    # --arg is 0-based against the MLIL call's recovered parameters, which can
    # differ from the argument positions the decompiler renders (an implicit/
    # struct-return or coalesced arg shifts the count) (#226).
    msg = (
        f"Argument index {arg_index} out of range: this call has {n} "
        f"MLIL argument(s){only}. --arg is 0-based and indexes the MLIL call "
        f"parameters, which may differ from the decompiler's displayed args."
    )
    # #324: an index at/beyond the calling convention's integer-arg registers is
    # passed on the STACK, which BN's MLIL/HLIL call model frequently omits.
    reg_count = _int_arg_reg_count(func)
    if reg_count is not None and arg_index >= reg_count:
        where = (
            "this calling convention passes ALL arguments on the STACK (e.g. i386 "
            "cdecl)" if reg_count == 0 else
            f"arg {arg_index} is at/beyond the {reg_count} integer-arg register(s) "
            f"of this calling convention, so it is likely passed on the STACK"
        )
        msg += (
            f" Note: {where}; BN's MLIL/HLIL call model often omits stack-passed "
            f"(e.g. variadic) args -- inspect `bn il {func.name} --view llil "
            f"--ssa` for the stack stores feeding the call (#324)."
        )
    elif reg_count is not None and n < reg_count and arg_index == n:
        # The requested index is exactly the FIRST register slot BN did not
        # recover (still within the ABI's integer-arg registers) -- the tight
        # "BN stopped recovering args right here" signal.
        msg += (
            f" Note: arg {arg_index} is the first integer-arg register this "
            f"convention passes that BN did not recover as a param ({n} "
            f"recovered) -- if the callee's prototype is narrower than the "
            f"actual call (variadic mis-typed as fixed-arity, or an indirect "
            f"call through a too-narrow function-pointer type), it may be a "
            f"real register-passed arg BN dropped; inspect `bn il {func.name} "
            f"--view llil --ssa` for the args feeding the call (#324)."
        )
    return msg


def _modeled_callee_name(bv, call_insn) -> str | None:
    """Resolved callee name (through thunks) for the #433 model gate, or None."""
    try:
        rt = _taint.resolve_call_target(bv, call_insn, follow_thunks=True)
    except Exception:
        return None
    for cand in (getattr(rt, "name", None),
                 getattr(getattr(rt, "function", None), "name", None)):
        if cand:
            return str(cand)
    return None


def _recover_trace_arg_via_reg(ctx, bv, func, call_insn, target_addr, arg_index: int,
                               view: str) -> tuple[list[Any] | None, str | None, str | None]:
    """#433: recover an out-of-range trace arg from its calling-convention
    register when a MODELED copy sink's call under-recovered its params (a thunk/
    import with a too-narrow prototype dropped r1/r2). MLIL view only. Returns
    ``(initial_vars, reg, callee)`` or ``(None, None, None)`` -- the latter when
    not applicable, leaving the precise out-of-range error to stand. Gated on the
    sink model proving arg ``arg_index`` exists, so it never seeds a fabricated
    argument. Shares the reaching-def machinery with ``taint backward``."""
    if view != "mlil" or arg_index < 0:
        return (None, None, None)
    reg = _arg_register(func, arg_index)
    if reg is None:
        return (None, None, None)
    callee = _modeled_callee_name(bv, call_insn)
    if not callee or arg_index not in _taint.model_arg_indices(_taint.load_models(), callee):
        return (None, None, None)
    seeds = _taint.reaching_arg_seed_vars(func, target_addr, reg, bn)
    initial_vars = [v for v, _ in seeds]
    if not initial_vars:
        return (None, None, None)
    return (initial_vars, reg, callee)


def _arg_label(ctx, bv, call_insn, arg_index: int, caller_func) -> dict[str, Any]:
    """{index, [register], [name]} for the traced call argument: its
    calling-convention register plus the callee's C parameter name when the
    callee resolves (#166)."""
    label: dict[str, Any] = {"index": arg_index}
    reg = _arg_register(caller_func, arg_index)
    if reg:
        label["register"] = reg
    try:
        rt = _taint.resolve_call_target(bv, call_insn, follow_thunks=True)
        callee = getattr(rt, "function", None)
        pvars = list(getattr(callee, "parameter_vars", []) or []) if callee else []
        if 0 <= arg_index < len(pvars):
            nm = getattr(pvars[arg_index], "name", None)
            if nm:
                label["name"] = str(nm)
    except Exception:
        pass
    return label


def _build_backward_trace(
    ctx,
    bv,
    ssa_func,
    initial_vars: list,
    max_depth: int,
    *,
    interprocedural: bool = False,
    ip_depth: int = 2,
    view: str = "mlil",
    _call_depth: int = 0,
    base_depth: int = 0,
    seed_addr: int | None = None,
) -> list[dict[str, Any]]:
    """Recursively walk SSA use-def chains backward, optionally crossing call boundaries."""
    if _call_depth > 10:
        return []  # Safety: prevent runaway recursion
    trace: list[dict[str, Any]] = []
    # #416: locals whose address is passed into a call, computed once per
    # function (not per terminus). Only needed in interprocedural mode.
    out_param_map = _build_out_param_map(ctx, bv, ssa_func) if interprocedural else {}
    # Each worklist item carries its def-use distance from the seed so the
    # reported "depth" is the real graph depth (operands of one definition
    # share a depth) rather than a sequential append index. base_depth
    # offsets a callee sub-walk so its depths continue from the call site. The
    # third element is the address at which this var was READ (its consumer), so
    # the #416 out-param lookup can require the address-taking call to precede it.
    worklist: list[tuple[Any, int, int | None]] = [(v, 0, seed_addr) for v in initial_vars]
    visited: set[Any] = set()

    while worklist and len(trace) < max_depth:
        ssa_var, node_depth, ref_addr = worklist.pop(0)
        depth = base_depth + node_depth
        if not isinstance(ssa_var, SSAVariable):
            trace.append({
                "ssa_var": str(ssa_var),
                "ssa_label": str(ssa_var),
                "depth": depth,
                "terminates": True,
                "reason": "undefined_or_global",
            })
            continue
        if ssa_var in visited:
            continue
        visited.add(ssa_var)

        try:
            def_insn = ssa_func.get_ssa_var_definition(ssa_var)
        except AttributeError:
            if interprocedural:
                continue
            raise OperationFailure(
                "no_ssa_trace",
                f"{view} SSA form does not support get_ssa_var_definition (MLIL or HLIL required)",
            )
        entry: dict[str, Any] = {
            "ssa_var": str(ssa_var),
            "ssa_label": _ssa_label(ssa_var),
            "depth": depth,
        }

        if def_insn is None:
            # No reaching definition: a real parameter, or an undefined
            # local / global. Only claim "function parameter" when it
            # actually is one; otherwise stay neutral (don't mislead
            # provenance slices).
            entry["terminates"] = True
            if _is_parameter_ssa_var(ctx, ssa_func, ssa_var):
                entry["reason"] = "function_parameter"
            else:
                entry["reason"] = "undefined_or_global"
                # #416: a value that bottoms out at a LOCAL whose address was
                # passed into a call (the common stack-struct out-param form,
                # `parse(input, &rec); use(rec.len)`) was likely written by that
                # callee through an out-pointer. Interprocedural tracing follows
                # only return values, so name the callee instead of leaving an
                # "undefined" terminus that reads like proof of origin.
                if interprocedural:
                    out_callee = _lookup_out_param_callee(out_param_map, ssa_var, ref_addr)
                    if out_callee:
                        entry["reason"] = "interprocedural_out_param_not_followed"
                        entry["out_param_callee"] = out_callee
            trace.append(entry)
            continue

        entry["address"] = hex(int(getattr(def_insn, "address", 0)))
        entry["il_text"] = str(def_insn)
        entry["operation"] = il_format._il_op_name(def_insn)

        def_op = entry["operation"]
        if "CALL" in def_op or "JUMP" in def_op:
            if interprocedural and ip_depth > 0:
                callee = _resolve_callee(ctx, bv, def_insn)
                if callee is not None:
                    callee_mlil = getattr(callee, "medium_level_il", None)
                    if callee_mlil is not None and callee_mlil.ssa_form is not None:
                        if hasattr(callee_mlil.ssa_form, "get_ssa_var_definition"):
                            callee_ret_vars = _find_return_vars(ctx, callee_mlil.ssa_form, bv)
                            if callee_ret_vars:
                                entry["cross_function"] = True
                                entry["callee"] = callee.name
                                entry["terminates"] = False
                                entry["reason"] = "cross_function"
                                trace.append(entry)
                                callee_trace = _build_backward_trace(
                                    ctx, bv, callee_mlil.ssa_form, callee_ret_vars,
                                    max_depth - len(trace),
                                    interprocedural=True,
                                    ip_depth=ip_depth - 1,
                                    view=view,
                                    _call_depth=_call_depth + 1,
                                    base_depth=depth + 1,
                                    # The callee's seed = its RETURN vars; the
                                    # caller's call-site address belongs to a
                                    # different function's address space, so it must
                                    # NOT be used to order the callee's own calls.
                                    # Leave the seed ungated (None); intra-callee
                                    # enqueues still carry callee-local addresses.
                                    seed_addr=None,
                                )
                                for ct in callee_trace:
                                    ct.setdefault("function_context", callee.name)
                                trace.extend(callee_trace)
                                continue
            entry["terminates"] = True
            entry["reason"] = "call_or_jump_boundary"
            # Resolve the call target to its symbol so a library-call origin reads
            # as e.g. `call strlen` rather than a bare PLT address (#193).
            callee_nm = _callee_display_name(ctx, bv, def_insn)
            if callee_nm:
                entry["callee"] = callee_nm
            trace.append(entry)
            continue

        load_expr = _load_expr_of(def_insn)
        if load_expr is not None:
            base, offset, width = _field_load_meta(load_expr)
            # A resolvable base+offset means this is a struct/field load, not an
            # opaque pointer deref -- label it `field_load` and carry the
            # structured fields so the slice is machine-consumable (#162).
            entry["reason"] = "field_load" if base is not None else "memory_load"
            if base is not None:
                entry["base"] = base
            if offset is not None:
                entry["offset"] = hex(offset) if isinstance(offset, int) else offset
            if width is not None:
                entry["width"] = width
            # #416: in interprocedural mode, if this load reads from the address
            # of a local that was passed by-address into a call, the value was
            # likely written by that callee through an OUT-POINTER -- which
            # interprocedural tracing follows only for return values. Emit an
            # honest boundary reason naming the callee instead of letting the
            # slice stop silently at a `field_load`/`memory_load`.
            if interprocedural:
                load_local = _address_of_target(getattr(load_expr, "src", None))
                out_callee = _lookup_out_param_callee(
                    out_param_map, load_local, int(getattr(def_insn, "address", 0)))
                if out_callee:
                    entry["reason"] = "interprocedural_out_param_not_followed"
                    entry["out_param_callee"] = out_callee
                    # The value's true origin is the callee's write through the
                    # out-pointer; mark the load as a boundary. Index/base
                    # provenance (`*(&local + idx)`) is still walked below, so a
                    # tainted index is not dropped.
                    entry["terminates"] = True
                    trace.append(entry)
                    for rv in _ssa_vars_from(getattr(def_insn, "vars_read", []) or []):
                        if rv not in visited:
                            worklist.append((rv, node_depth + 1, int(getattr(def_insn, "address", 0))))
                    continue
            # Preserve prior per-form walk behavior: a top-level load def
            # terminated; a `x = [addr]` SET_VAR continued through its base
            # pointer (so provenance reaches where the struct came from).
            entry["terminates"] = "LOAD" in def_op
            trace.append(entry)
            for rv in _ssa_vars_from(getattr(def_insn, "vars_read", []) or []):
                if rv not in visited:
                    worklist.append((rv, node_depth + 1, int(getattr(def_insn, "address", 0))))
            continue

        entry["terminates"] = False
        # Populate `reason` from a controlled vocabulary instead of leaving it
        # null on ordinary steps: a phi merge is `phi_source`, everything else is
        # a plain `definition` (#162).
        entry["reason"] = "phi_source" if "PHI" in def_op else "definition"
        trace.append(entry)

        for rv in _ssa_vars_from(getattr(def_insn, "vars_read", []) or []):
            if rv not in visited:
                worklist.append((rv, node_depth + 1, int(getattr(def_insn, "address", 0))))

    return trace


def _is_parameter_ssa_var(ctx, ssa_func, ssa_var) -> bool:
    """True if *ssa_var* (an SSA variable with no reaching definition) is a
    formal parameter of the function, vs. an undefined local or a global.

    Matched against the source function's ``parameter_vars`` by identifier
    first, then by base name. Returns False whenever parameter information
    is unavailable, so the caller falls back to a neutral label rather than
    guessing 'parameter'."""
    source = getattr(ssa_func, "source_function", None)
    params = list(getattr(source, "parameter_vars", []) or [])
    if not params:
        return False
    base = getattr(ssa_var, "var", ssa_var)
    ident = getattr(base, "identifier", None)
    base_name = str(ssa_var).split("#")[0]
    for param in params:
        if ident is not None and getattr(param, "identifier", None) == ident:
            return True
        if base_name and str(getattr(param, "name", param)) == base_name:
            return True
    return False


def _resolve_callee(ctx, bv, call_insn):
    """Resolve a call instruction's callee to a BN function, or None.

    Thin wrapper over the canonical resolver in ``taint_engine``; follows
    thunks/veneers (single-instruction tailcalls) to their real target so
    interprocedural tracing works through PLT stubs and GCC thunks.
    """
    return _taint.resolve_call_target(bv, call_insn, follow_thunks=True).function


def _callee_display_name(ctx, bv, def_insn) -> str | None:
    """Best-effort symbol name for a call/jump target, so a value that
    originates at a call boundary names the callee instead of just showing the
    raw PLT target address -- the same resolution ``taint backward`` already
    performs (#193). Returns None for a genuinely indirect/unresolved target.
    """
    try:
        fn = _resolve_callee(ctx, bv, def_insn)
    except Exception:
        fn = None
    if fn is not None and getattr(fn, "name", None):
        return str(fn.name)
    # No resolvable function (a PLT stub with only a symbol, or an extern): fall
    # back to a symbol at the raw dest address, Thumb-bit normalized.
    dest = getattr(def_insn, "dest", None)
    if dest is None:
        return None
    try:
        addr = _extract_dest_address(bv, dest)
    except Exception:
        addr = None
    if addr is None:
        return None
    for cand in (addr, addr & ~1):
        try:
            sym = bv.get_symbol_at(cand)
        except Exception:
            sym = None
        if sym is not None and getattr(sym, "name", None):
            return str(sym.name)
    return None


def _resolve_thunk(ctx, bv, fn):
    """If fn is a single-instruction tailcall thunk, return its real target
    (delegates to ``taint_engine.follow_thunk``)."""
    return _taint.follow_thunk(bv, fn)


def _extract_dest_address(bv, dest):
    """Numeric address of a call/tailcall destination expression (delegates
    to ``taint_engine.extract_dest_address``)."""
    return _taint.extract_dest_address(bv, dest)


def _address_of_target(expr, depth: int = 0):
    """The local Variable an address expression takes the address of, handling
    both ``&local`` (MLIL_ADDRESS_OF) and ``&local + const`` (ADD/SUB of an
    address-of and a constant). Returns None when the expression is not the
    address of a stack/local variable. Best-effort, bounded recursion."""
    if expr is None or depth > 6:
        return None
    op = il_format._il_op_name(expr)
    if "ADDRESS_OF" in op:
        return getattr(expr, "src", None)
    if "ADD" in op or "SUB" in op:
        for side in (getattr(expr, "left", None), getattr(expr, "right", None)):
            target = _address_of_target(side, depth + 1)
            if target is not None:
                return target
    return None


def _var_address_key(v):
    """Match key for a local Variable: BN's stable ``identifier`` when present,
    else the variable object itself (its own equality). The same local taken by
    address in two places must compare equal."""
    ident = getattr(v, "identifier", None)
    return ident if ident is not None else v


def _build_out_param_map(ctx, bv, ssa_func) -> dict:
    """Map ``{local var key -> [(call_addr, callee_name), ...]}`` for locals whose
    address is passed into a call (#416).

    Built ONCE per function so the trace's per-terminus lookups don't rescan the
    whole function (perf), and so a lookup can require the call to PRECEDE the
    read (a call that receives ``&local`` after the value was read -- or the sink
    itself -- is not the source). Handles both the inlined ``f(&local)`` argument
    and the one-hop ``p = &local; f(p)`` form. Best-effort; ``{}`` on any
    BN-API shortfall, so it never fabricates a reason."""
    result: dict = {}
    try:
        holders: dict = {}      # SSA var holding &local -> local key
        calls: list = []
        for block in getattr(ssa_func, "basic_blocks", []) or []:
            for ins in block:
                if "CALL" in il_format._il_op_name(ins):
                    calls.append(ins)
                    continue
                taken = _address_of_target(getattr(ins, "src", None))
                if taken is not None:
                    lk = _var_address_key(getattr(taken, "var", None) or taken)
                    for w in _ssa_vars_from(getattr(ins, "vars_written", []) or []):
                        holders[w] = lk
                    dest = getattr(ins, "dest", None)
                    if isinstance(dest, SSAVariable):
                        holders[dest] = lk
        for ins in calls:
            try:
                addr = int(getattr(ins, "address", 0))
            except Exception:
                addr = 0
            local_keys: set = set()
            for param in getattr(ins, "params", []) or []:
                direct = _address_of_target(param)
                if direct is not None:
                    local_keys.add(_var_address_key(getattr(direct, "var", None) or direct))
                for rv in _ssa_vars_from(getattr(param, "vars_read", []) or []):
                    if rv in holders:
                        local_keys.add(holders[rv])
            if not local_keys:
                continue
            callee = _callee_display_name(ctx, bv, ins)
            if not callee:
                continue
            for lk in local_keys:
                result.setdefault(lk, []).append((addr, callee))
    except Exception:
        return {}
    return result


def _lookup_out_param_callee(out_param_map: dict, base_var, before_addr) -> str | None:
    """Callee that filled *base_var* via an out-pointer at a call PRECEDING
    *before_addr* (#416), or None. Ordering is load-bearing: a call that takes
    ``&local`` after the value was read is not its source."""
    if not out_param_map or base_var is None:
        return None
    key = _var_address_key(getattr(base_var, "var", None) or base_var)
    for addr, callee in out_param_map.get(key, []):
        if before_addr is None or addr < before_addr:
            return callee
    return None


def _find_return_vars(ctx, ssa_func, bv=None, _visited=None) -> list[SSAVariable]:
    """Find SSA variables that feed into RET instructions in a function.

    For functions that only contain a TAILCALL (PLT stubs, thunks), follows
    the tailcall to the real implementation and returns its return vars.
    """
    ret_vars: list[SSAVariable] = []
    has_ret = False
    for block in getattr(ssa_func, "basic_blocks", []) or []:
        for insn in block:
            op_name = il_format._il_op_name(insn)
            if op_name == "MLIL_RET":
                has_ret = True
                found = _ssa_vars_from(getattr(insn, "vars_read", []) or [])
                if not found:
                    src = getattr(insn, "src", []) or []
                    for s in src:
                        var = getattr(s, "var", None)
                        if var is not None and isinstance(var, SSAVariable):
                            found.append(var)
                if not found:
                    dest = getattr(insn, "dest", None)
                    if dest is not None:
                        found = _ssa_vars_from([dest] if not isinstance(dest, list) else dest)
                if not found:
                    non_ssa = getattr(insn, "non_ssa_form", None)
                    if non_ssa is not None:
                        found = _ssa_vars_from(getattr(non_ssa, "vars_read", []) or [])
                ret_vars.extend(found)
    # No MLIL_RET found — try to follow TAILCALL to real implementation
    if not has_ret and bv is not None:
        if _visited is None:
            _visited = set()
        fn_key = id(ssa_func)
        if fn_key in _visited:
            return ret_vars  # Cycle detected
        _visited.add(fn_key)
        for block in getattr(ssa_func, "basic_blocks", []) or []:
            for insn in block:
                if "TAILCALL" not in il_format._il_op_name(insn):
                    continue
                dest = getattr(insn, "dest", None)
                if dest is None:
                    break
                fn_source = getattr(ssa_func, "source_function", None)
                source_start = getattr(fn_source, "start", None)
                addr = _extract_dest_address(bv, dest)
                if addr is not None:
                    if source_start is not None and (addr == source_start or addr & ~1 == source_start):
                        break  # Self-loop (PLT stub → itself)
                    target = bv.get_function_at(addr)
                    if target is None:
                        target = bv.get_function_at(addr & ~1)
                    if target is not None and source_start is not None and target.start == source_start:
                        break  # Self-loop via different resolution path
                    if target is not None:
                        callee_mlil = getattr(target, "medium_level_il", None)
                        if callee_mlil and callee_mlil.ssa_form:
                            return _find_return_vars(ctx, callee_mlil.ssa_form, bv, _visited)
                break  # Only try the first instruction
    return ret_vars


def _backward_slice(
    ctx,
    selector: str | None,
    identifier: str,
    address: str,
    *,
    arg_index: int = 0,
    view: str = "mlil",
    max_depth: int = 50,
    interprocedural: bool = False,
    ip_depth: int = 2,
) -> dict[str, Any]:
    if max_depth < 1:
        raise OperationFailure("invalid_max_depth", f"Invalid max_depth: {max_depth}")
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier)
    target_addr = _parse_address(address)

    il_name = {"mlil": "medium_level_il", "llil": "low_level_il", "hlil": "high_level_il"}.get(view)
    if il_name is None:
        raise OperationFailure("invalid_view", f"Unsupported IL view: {view}")

    il_func = getattr(func, il_name, None)
    if il_func is None:
        raise OperationFailure("no_il", f"Function {func.name} has no {view}")
    ssa_func = il_func.ssa_form
    if ssa_func is None:
        raise OperationFailure("no_ssa", f"Function {func.name} has no {view} SSA form")

    if not hasattr(ssa_func, "get_ssa_var_definition"):
        raise OperationFailure(
            "no_ssa_trace",
            f"{view} SSA form does not support get_ssa_var_definition (MLIL or HLIL required)",
        )

    ssa_instructions = il_format._iter_il_instructions(ssa_func)
    call_insn = None
    for insn in ssa_instructions:
        if int(getattr(insn, "address", 0)) != target_addr:
            continue
        # "CALL" also matches TAILCALL/SYSCALL op names.
        if "CALL" in il_format._il_op_name(insn):
            call_insn = insn
            break

    if call_insn is None:
        hint = ""
        if view != "mlil":
            hint = " (try --view mlil, which has the broadest call coverage)"
        raise OperationFailure(
            "instruction_not_found",
            f"No call instruction at {address} in {func.name}{hint}",
        )

    params = list(getattr(call_insn, "params", []) or [])
    if not params:
        raise OperationFailure(
            "no_params",
            f"Call at {address} has no exposed parameters in {view}",
        )
    param_expr = None
    recovered_reg: str | None = None
    recovered_callee: str | None = None
    if 0 <= arg_index < len(params):
        param_expr = params[arg_index]
        initial_vars: list[Any] = _ssa_vars_from(getattr(param_expr, "vars_read", []) or [])
    else:
        # #433: the index is out of range -- but if a MODELED copy sink's call
        # under-recovered its args (a thunk/import with a too-narrow prototype
        # dropped r1/r2), recover the value from the calling-convention register's
        # reaching definition rather than dead-ending. Gated on the sink model so
        # we never fabricate an arg the call doesn't actually pass.
        initial_vars, recovered_reg, recovered_callee = _recover_trace_arg_via_reg(
            ctx, bv, func, call_insn, target_addr, arg_index, view)
        if not initial_vars:
            raise OperationFailure(
                "invalid_arg_index",
                _out_of_range_arg_msg(func, arg_index, len(params)))

    arg_label = _arg_label(ctx, bv, call_insn, arg_index, func)

    hints: list[str] = []
    # #489: proactive call-model-truncation disclosure -- fires for IN-RANGE args too,
    # so an analyst tracing a recovered arg is warned that stack-passed args were
    # dropped from the model (the reactive #324 message only fires for out-of-range).
    _callee_nm = _modeled_callee_name(bv, call_insn)
    if not _callee_nm:
        try:
            _cf = _resolve_callee(ctx, bv, call_insn)
            _callee_nm = str(getattr(_cf, "name", "") or "") or None
        except Exception:
            _callee_nm = None
    _trunc = _call_model_truncation_note(
        bv, func, call_insn, target_addr, params, _callee_nm or recovered_callee)
    if _trunc is not None:
        hints.append(_trunc)
    if recovered_reg is not None:
        hints.append(
            f"arg {arg_index} of {recovered_callee} was recovered from calling-"
            f"convention register {recovered_reg}: this call under-recovered its "
            f"arguments (a thunk/import with a too-narrow prototype dropped the "
            f"arg), so the trace seeds from the register's reaching definition "
            f"rather than the MLIL call's parameter list (#433)."
        )
    # An address-of arg with no SSA value reads is an output-pointer dead-end:
    # tracing it would follow where the *pointer* came from (a local buffer),
    # not the data the callee writes through it. Surface that instead of the
    # misleading "constant or immediate -- no SSA trace" (#166).
    if param_expr is not None and not initial_vars and _is_address_of(param_expr):
        callee_nm = arg_label.get("name") or "the callee"
        hints.append(
            f"arg {arg_index} is a pointer (address-of); this traces where the "
            f"pointer came from, not the data written through it. To follow data "
            f"{callee_nm} writes into the pointee, run a forward taint from the "
            f"call site (e.g. `taint forward --source call:<callee>`) or trace the "
            f"buffer's later consumers."
        )
    elif param_expr is not None and not initial_vars:
        # A constant/immediate arg (e.g. a literal length/flag) has no SSA
        # definition to trace. Surface the value as a structured hint so both
        # text and JSON consumers see *which* constant -- not just the renderer's
        # generic "constant or immediate" line (which carried no value and left
        # JSON `hints` empty).
        cval = _const_int(param_expr)
        if cval is not None:
            hints.append(
                f"arg {arg_index} is the constant {hex(cval)} -- a compile-time "
                f"immediate with no SSA definition to trace back."
            )

    trace = _build_backward_trace(
        ctx, bv, ssa_func, initial_vars, max_depth,
        interprocedural=interprocedural,
        ip_depth=ip_depth,
        view=view,
        # The seed args are consumed AT the traced call; an out-param fill must
        # precede it (#416 ordering).
        seed_addr=target_addr,
    )

    return {
        "function": func.name,
        "function_address": hex(func.start),
        "target_address": hex(target_addr),
        "arg_index": arg_index,
        "arg_label": arg_label,
        "view": view,
        "interprocedural": interprocedural,
        "ip_depth": ip_depth if interprocedural else 0,
        "truncated": len(trace) >= max_depth,
        "step_count": len(trace),
        "trace": trace,
        "hints": hints,
    }
