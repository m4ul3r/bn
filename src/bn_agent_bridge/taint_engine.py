"""Taint engine over Binary Ninja MLIL-SSA (forward propagation + backward slice).

This module is intentionally free of any ``binaryninja`` import: it operates on
whatever MLIL-SSA objects the bridge hands it (functions, instructions,
SSAVariables, PossibleValueSets). That keeps it unit-testable against the same
synthetic IL fakes the bridge tests use.

Scope (as shipped, not MVP): forward propagation and backward slicing, with
*bounded* interprocedural stepping (descent into modeled/in-binary callees and
ascent into callers, #146/#147), indirect-call resolution (value-set +
agent-supplied ``resolve_map``), external/import-stub resolution (MLIL_EXTERN_PTR
-> symbol model), per-callsite source attribution (additive ``by_source``), and
memory-SSA store/load correlation where addresses match (coarse otherwise). What
stays out of scope: precise path-sensitivity, full alias analysis, and any proof
of reachability. Every place the analysis is coarse or stops is surfaced in
``assumptions``/``leaves`` and the output always carries a ``soundness``
disclaimer — we never silently drop an edge.

API behaviour verified against /opt/binaryninja (see the design's spike):
  - ``func.mlil.ssa_form`` -> MediumLevelILFunction; ``.instructions`` iterable
  - instr: ``.instr_index`` ``.address`` ``.operation.name`` ``.vars_read``
    ``.vars_written`` ``.operands`` ``.params`` ``.dest``; ``str(instr)`` text
  - SSAVariable: ``.var`` (-> Variable) ``.var.name`` ``.version``
  - ``ssa.get_ssa_var_definition(v)`` / ``ssa.get_ssa_var_uses(v)``
  - expr ``.possible_values`` -> PossibleValueSet (``.type.name`` str)
"""
from __future__ import annotations

import re
import shlex
from typing import Any

# Structural split (#562): the engine re-exports the moved names so existing
# importers (`from bn_agent_bridge.taint_engine import X`, `_taint.X`) keep
# working unchanged. Behaviour is identical -- this only relocates code.
from .taint_models import (  # noqa: F401
    SOUNDNESS, _OVERFLOW_INDEX_CLASSES, _RECVMSG_FAMILY,
    _BROAD_SOURCE_BYTES, _BROAD_SOURCE_MEMBERS, _BUILTIN_MODELS,
    _model_buffer_source_args, TaintError, BoundedSink,
    _coerce_model_map, _validate_model_interior, load_models,
    model_overlay_sources, _canonical_cxx_alloc, lookup_model,
    _try_arg_index, model_arg_indices, model_is_benign_marker,
)
from .taint_il import (  # noqa: F401
    reaching_reg_def, reaching_arg_seed_vars, op_name, is_ssa_var,
    _FORMAT_SPEC_RE, _count_format_args, var_key, var_label,
    ssa_reads, ssa_writes, expr_reads, function_at, const_target,
    _mlil_ssa, _ssa_instructions, _symbols_by_name, _symbol_by_raw_name,
    extract_dest_address, targets_from_pvs, follow_thunk, ResolvedTarget,
    resolve_call_target,
)
from .taint_locators import (  # noqa: F401
    parse_locator, _locator_index, format_locator, var_label_of,
    node_label, _instr_dict, _render_source, _make_signature,
    derive_flow_facts, normalize_resolve_map,
)
from .taint_result import (  # noqa: F401
    BLOCKING_LEAF_KINDS,
    backward_diagnostics,
    command_template,
    forward_zero_diagnostics,
    indirect_pointer_slot_leaf,
    misanchored_recv_leaf,
)

# Module aliases so tests/callers can patch a moved symbol on its OWNER module
# (a re-exported binding on the engine is a separate name -- rebinding it would
# not affect the module that actually reads it).
from . import taint_il as _taint_il_mod  # noqa: F401
from . import taint_locators as _taint_locators_mod  # noqa: F401
from . import taint_models as _taint_models_mod  # noqa: F401
from . import taint_result as _taint_result_mod  # noqa: F401
from .il_format import (
    _display_name,
    _format_specs,
    _variadic_determination,
    unbounded_scanf_strings,
)
from .taint_models import _model_copy_roles


class _RegisterOperand:
    """A call argument BN dropped from the MLIL call's parameter list, read as
    the value its calling-convention register holds at the call (#1001 item 1).

    Shaped like an ``MLIL_VAR_SSA`` read of that value (``operation.name``,
    ``src``, ``vars_read``) so the forward walk's taint check can ask whether
    the dropped arg carries taint. It is evidence of REACH only: it is never
    handed to a sink classifier, so no class is lowered from a register read."""

    class _Op:
        name = "MLIL_VAR_SSA"

    operation = _Op

    def __init__(self, var: Any, register: str | None) -> None:
        self.src = var
        self.vars_read = [var]
        self.register = register

    def __str__(self) -> str:
        return f"{self.src} (via {self.register or '?'})"


# Caller sites the backward ascent follows per function. Module-level (not a
# local of ``_continue_into_callers``) because the cap is part of the result
# contract: crossing it sets ``stats.truncated``/``truncation_cause`` and emits a
# ``caller_sites_truncated`` leaf naming the counts (#810).
MAX_CALLERS = 16

# Memory-SSA versions the backward local-buffer writer walk follows from the
# sink back toward the function entry before it gives up. Exceeding it sets
# ``stats.truncated`` (cause ``memory_chain``) and emits a blocking leaf: a
# writer behind the cut is absent from the slice, and the PR-928 review showed
# a silent cut letting a one-byte store past it claim a complete slice.
MEM_CHAIN_BUDGET = 256

# BN's stores to a stack slot BY NAME (no address expression): an address-taken
# local written as a whole or by field offset. They are memory definitions
# without ``src_memory``/``dest_memory`` -- only ``ssa_memory_version``.
_NAMED_STORE_OPS = frozenset({
    "MLIL_SET_VAR_ALIASED", "MLIL_SET_VAR_ALIASED_FIELD", "MLIL_SET_VAR_FIELD",
})


class _BufferSeed:
    """A backward seed that is the CONTENTS of a buffer the sink was handed by
    address with the address folded into the call parameter itself
    (``popen(&cmd, "r")``, ``popen(&line + off, "r")``), rather than an SSA
    variable. The same shape one definition away (``p#1 = &cmd; popen(p#1)``)
    walks the buffer's writers; this makes the folded form take the same path
    instead of a hard "reads no variable" seed error -- or, with a variable
    offset, instead of following only the offset (round 4 P1). ``local`` is
    the local buffer, or ``addr`` a writable global the operand is built on
    (``g_hosts + (i << 6)``), which the slice reports as a leaf. ``name``
    renders as ``&cmd`` / the address through ``var_label`` so the slice's
    ``sink.seed`` says what was seeded."""
    __slots__ = ("local", "addr", "name")

    def __init__(self, local: Any = None, *, addr: int | None = None):
        self.local = local
        self.addr = addr
        self.name = f"&{var_label(local)}" if local is not None else hex(int(addr or 0))


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------

# The `resolve_map` default: no map was supplied. A distinct object, because a
# supplied JSON `null` document is invalid, not an omission.
_NO_RESOLVE_MAP: Any = object()


class TaintEngine:
    def __init__(
        self,
        bv: Any,
        models: dict[str, Any],
        *,
        find_variable: Any = None,
        unknown_call_policy: str = "conservative",
        resolve_map: Any = _NO_RESOLVE_MAP,
        resolve_map_origin: str = "--resolve-map",
        find_function: Any = None,
        max_iters: int = 256,
        max_depth: int = 64,
        mangled: bool = False,
    ):
        self.bv = bv
        self.models = models
        self._find_variable = find_variable
        self.unknown_call_policy = unknown_call_policy
        # agent-supplied indirect-call resolution, validated here -- once, naming
        # the file -- rather than when the walk first needs it (#1041 item 3):
        # {call_address: [target_address, ...]}. `find_function` resolves a
        # target spelled as a function name exactly as `-f` does. Only an
        # OMITTED map is empty: a supplied document of any other shape, JSON
        # `null` included, is refused by name (#1050 review rounds 1-2).
        self.resolve_map = normalize_resolve_map(
            {} if resolve_map is _NO_RESOLVE_MAP else resolve_map,
            origin=resolve_map_origin, find_function=find_function)
        self.max_iters = max_iters
        self.max_depth = max_depth
        self.mangled = mangled
        # follow_thunk is deterministic for a given (bv, function) and the
        # engine is created per request, so resolved thunks are cached for the
        # engine's lifetime -- avoids re-materializing a veneer's MLIL on every
        # candidate visit (forward) and every callsite scan (backward).
        self._thunk_cache: dict[int, Any] = {}
        # Per-function unlifted-instruction scan, cached by function start (#206).
        self._unimpl_cache: dict[int, list[int]] = {}
        # _buffer_target is purely STRUCTURAL (it resolves a pointer expr to its
        # buffer key via the SSA def graph; it never reads the taint set), but the
        # forward fixpoint reaches it -- via _pointee_tainted -- on every escape
        # check every iteration, so it dominates large-function runs (#420).
        # Memoize it per (function, SSA expr_index): the result is identical on
        # every pass. Keyed by the function token set on entry to a fixpoint run so
        # interprocedural descent into a callee can't alias a caller's expr_index.
        self._bt_cache: dict[Any, Any] = {}
        self._bt_func_token: Any = None

    # -- shared resolution ------------------------------------------------

    def _ssa_func(self, func: Any) -> Any:
        mlil = getattr(func, "mlil", None)
        if mlil is None:
            raise TaintError("function has no MLIL (analysis incomplete?)")
        ssaf = getattr(mlil, "ssa_form", None)
        if ssaf is None:
            raise TaintError("function has no MLIL SSA form")
        return ssaf

    def _instrs(self, ssaf: Any) -> list[Any]:
        try:
            return list(ssaf.instructions)
        except Exception as exc:  # pragma: no cover - defensive
            raise TaintError(f"cannot enumerate SSA instructions: {exc}")

    def _callee_name(self, addr: int | None) -> str | None:
        if addr is None or self.bv is None:
            return None
        # function_at normalizes the Thumb low bit so an odd code pointer still
        # resolves to its function name (#89).
        fn = function_at(self.bv, addr)
        if fn is not None and getattr(fn, "name", None):
            return str(fn.name)
        for cand in (addr, addr & ~1) if (addr & 1) else (addr,):
            try:
                sym = self.bv.get_symbol_at(cand)
            except Exception:
                sym = None
            if sym is not None and getattr(sym, "name", None):
                return str(sym.name)
        return None

    def _is_call(self, ins: Any) -> bool:
        return "CALL" in op_name(ins) or "TAILCALL" in op_name(ins)

    def _unimplemented_addrs(self, func: Any, instrs: list[Any]) -> list[int]:
        """Addresses of instructions BN's lifter could not model in *func*.

        Scans both the MLIL-SSA instructions already in hand (integer ops that
        surface as ``MLIL_UNIMPL``) and the function's LLIL (FP/SIMD ops like the
        AArch64 ``fnmsub``/``fmadd`` family are unlifted at decode and never reach
        MLIL, so a MLIL-only scan misses the motivating case). Cached per function
        start; defensive so the synthetic test fakes (no LLIL) just see the MLIL
        scan. Lets forward taint flag an otherwise-silent dataflow hole (#206)."""
        key = int(getattr(func, "start", 0))
        if key in self._unimpl_cache:
            return self._unimpl_cache[key]
        addrs: set[int] = set()
        for ins in instrs:
            if "UNIMPL" in op_name(ins):
                addrs.add(int(getattr(ins, "address", 0)))
        il = getattr(func, "low_level_il", None) or getattr(func, "llil", None)
        if il is not None:
            try:
                blocks = list(il)
            except Exception:
                blocks = list(getattr(il, "basic_blocks", None) or [])
            for block in blocks:
                try:
                    items = list(block)
                except Exception:
                    continue
                for ins in items:
                    if "UNIMPL" in op_name(ins):
                        addrs.add(int(getattr(ins, "address", 0)))
        result = sorted(addrs)
        self._unimpl_cache[key] = result
        return result

    @staticmethod
    def _is_stack_write(writes: list[Any]) -> bool:
        """True if any written SSA var is a STACK variable (vs a register/flag).

        Gates the pointer_escape leaf (#228) to genuine stack-descriptor captures
        (`stack_local = &buf`), so a register `x1 = &buf` that merely sets up the
        source call's own buffer argument is NOT mistaken for an escape. Returns
        False when the source type is unavailable (the unit fakes / unknown), so
        the conservative default is "not an escape" rather than a false leaf."""
        for w in writes:
            base = getattr(w, "var", w)
            st = getattr(base, "source_type", None)
            name = str(getattr(st, "name", None) or (str(st) if st is not None else ""))
            if "Stack" in name:
                return True
        return False

    @staticmethod
    def _type_class_name(t: Any) -> str:
        tc = getattr(t, "type_class", None)
        return str(getattr(tc, "name", None) or (str(tc) if tc is not None else ""))

    def _resolve_named_type(self, t: Any, depth: int = 0) -> Any:
        """Resolve a NamedTypeReference (a typedef'd struct pointer's pointee, the
        common firmware case) to its concrete type via the view, best-effort.
        Returns *t* unchanged when it isn't a named ref or can't be resolved."""
        if t is None or depth > 4 or "NamedTypeReference" not in self._type_class_name(t):
            return t
        for getter in (
            lambda: t.target(self.bv),
            lambda: self.bv.get_type_by_name(getattr(t, "name", None)),
        ):
            try:
                r = getter()
            except Exception:
                r = None
            if r is not None and r is not t:
                return self._resolve_named_type(r, depth + 1)
        return t

    def _broad_source_hint(self, pv: Any, idx: int) -> str | None:
        """A "broad source" nudge when a ``param:idx`` source is a pointer to a
        large aggregate: the whole struct is one coarse taint location, so it
        over-taints into unrelated code. Suggest a narrower locator (#219).

        Duck-typed over the BN ``Type`` API (``type_class.name`` / ``target`` /
        ``width`` / ``members``); a string/None type (the unit fakes) yields no
        hint, so this is silent unless a genuine aggregate pointer is seen."""
        t = getattr(pv, "type", None)
        if t is None or "Pointer" not in self._type_class_name(t):
            return None
        pointee = getattr(t, "target", None)
        if pointee is None:
            children = list(getattr(t, "children", None) or [])
            pointee = children[0] if children else None
        pointee = self._resolve_named_type(pointee)
        if pointee is None:
            return None
        pcn = self._type_class_name(pointee)
        if "Structure" not in pcn and "Array" not in pcn:
            return None
        try:
            size = int(getattr(pointee, "width", 0) or 0)
        except Exception:
            size = 0
        try:
            members = len(list(getattr(pointee, "members", None) or []))
        except Exception:
            members = 0
        if size < _BROAD_SOURCE_BYTES and members < _BROAD_SOURCE_MEMBERS:
            return None
        desc = ", ".join(
            x for x in (f"{members} fields" if members else None,
                        f"{hex(size)} bytes" if size else None) if x) or "large aggregate"
        return (
            f"broad source: param:{idx} is a pointer to a large aggregate ({desc}); "
            f"the whole struct is treated as one tainted location, which can "
            f"over-taint into unrelated code -- consider a narrower locator (e.g. the "
            f"actual input/recv buffer field) for a tighter result")

    def _resolve_direct_target(self, ins: Any) -> int | None:
        """Resolved direct/import call-target address, or None for a genuinely
        indirect call. Uses the shared resolver (follow_thunks=False, so thunk
        semantics are unchanged -- the call loop still follows thunks itself),
        which resolves MLIL_IMPORT and Thumb-tagged targets that const_target
        misses (#89 Problem A). Falls back to the raw constant for the synthetic
        test fakes / odd dests where no function can be confirmed."""
        if self.bv is not None:
            resolved = resolve_call_target(self.bv, ins, follow_thunks=False)
            if resolved.address is not None:
                return resolved.address
        return const_target(getattr(ins, "dest", None))

    def _call_params(self, ins: Any) -> list[Any]:
        params = getattr(ins, "params", None)
        if params is None:
            return []
        try:
            return list(params)
        except Exception:
            return []

    def _model_arg_indices(self, callee: str) -> set[int]:
        """Thin wrapper over the shared :func:`model_arg_indices` gate (#433)."""
        return model_arg_indices(self.models, callee)

    def _arg_register(self, func: Any, idx: int) -> str | None:
        """Calling-convention integer-argument register name for ``idx`` (e.g.
        ``r2`` on ARM, ``rdx`` on x86-64 SysV), or None if unrecoverable (#433)."""
        cc = getattr(func, "calling_convention", None)
        if cc is None:
            plat = getattr(self.bv, "platform", None)
            cc = getattr(plat, "default_calling_convention", None)
        regs = list(getattr(cc, "int_arg_regs", []) or [])
        if 0 <= idx < len(regs):
            return str(regs[idx])
        return None

    def _reaching_arg_seeds_via_reg(self, func: Any, sites: list[Any],
                                    idx: int) -> tuple[str | None, list[tuple[Any, Any]]]:
        """Recover a call argument BN dropped from the MLIL call's parameters by
        reading the calling-convention register for arg ``idx`` and bridging its
        reaching definition to the MLIL SSA var (#433), for each call site.
        Delegates to the shared :func:`reaching_arg_seed_vars` (a dominance walk
        of the LLIL-SSA reaching def). Returns ``(reg_name, [(mlil_ssa_var,
        call_site), ...])`` -- each recovered seed is paired with the CALL-site
        instruction (not the register's def site) so the slice records the sink at
        the call, matching the normal ``arg:`` seed path. The seed list is empty
        when nothing is recoverable (e.g. the unit fakes, which carry no LLIL)."""
        reg = self._arg_register(func, idx)
        if reg is None:
            return (None, [])
        try:
            import binaryninja as bn
        except Exception:
            return (reg, [])
        by_addr: dict[int, Any] = {}
        for c in sites:
            by_addr.setdefault(int(getattr(c, "address", -1)), c)
        seeds: list[tuple[Any, Any]] = []
        for addr, call_site in by_addr.items():
            for var, _defn in reaching_arg_seed_vars(func, addr, reg, bn):
                seeds.append((var, call_site))
        return (reg, seeds)

    def _pointee_var(self, ssaf: Any, expr: Any, depth: int = 0) -> Any:
        """Follow a pointer expression to the underlying stack Variable.

        Handles the common ``rsi#1 = &buf`` / aliased-buffer pattern: an
        ``MLIL_ADDRESS_OF`` yields its source Variable directly; an SSA var is
        chased through its definition. Returns a Variable or None.
        """
        if expr is None or depth > 6:
            return None
        name = op_name(expr)
        if "ADDRESS_OF" in name:
            return getattr(expr, "src", None) or getattr(expr, "var", None)
        if is_ssa_var(expr):
            try:
                d = ssaf.get_ssa_var_definition(expr)
            except Exception:
                d = None
            if d is not None:
                return self._pointee_var(ssaf, getattr(d, "src", None), depth + 1)
            return None
        # a var-ssa expression wrapping an SSAVariable
        reads = expr_reads(expr)
        if len(reads) == 1:
            try:
                d = ssaf.get_ssa_var_definition(reads[0])
            except Exception:
                d = None
            if d is not None:
                return self._pointee_var(ssaf, getattr(d, "src", None), depth + 1)
        return None

    def _pointee_tainted(self, ssaf: Any, expr: Any, tainted: set):
        """If *expr* is a pointer to a buffer (stack Variable or global) that has
        ANY tainted version, return a representative tainted node, else None.

        Version-agnostic on purpose: a buffer/struct slot is written under one
        SSA/memory version (e.g. an MLIL_SET_VAR_ALIASED store -> var#85) but a
        later ``&var`` references the whole variable (version None). Matching only
        the exact version would miss the pointer carrying that taint.
        """
        bt = self._buffer_target(ssaf, expr)
        if bt is None:
            return None
        k = bt[0]
        if (k, None) in tainted:
            return (k, None)
        for node in tainted:
            if node[0] == k:
                return node
        return None

    def _return_buffer_tainted(self, ssaf: Any, ins: Any, tainted: set) -> bool:
        """True if a RET returns a POINTER to a tainted buffer -- a callee that
        fills a heap/stack buffer with tainted bytes and returns it (the strdup /
        shell_quote idiom). ``read_taint`` only catches a tainted scalar return
        value; the returned *pointer's* value is the fresh allocation address, not
        attacker-derived, so the scalar check misses it and the caller sees a clean
        result -- the buffer's tainted CONTENT is dropped at the call boundary.
        Recognizing it lets the caller treat the result as tainted, so a later
        ``%s`` / ``strlen`` / copy of the returned buffer keeps propagating
        (#319a/#376 interprocedural return-of-tainted-buffer). Relies on the buffer
        being keyable (stack Variable, global, or heap alloc site)."""
        srcs = getattr(ins, "src", None)
        if srcs is None:
            return False
        if not isinstance(srcs, (list, tuple)):
            srcs = [srcs]
        seen: set = set()
        return any(self._expr_buffer_tainted(ssaf, e, tainted, seen, 0)
                   for e in srcs if e is not None)

    def _expr_buffer_tainted(self, ssaf: Any, expr: Any, tainted: set, seen: set, depth: int) -> bool:
        """Whether *expr* (a returned pointer) reaches a tainted buffer. First the
        convergent/stack/global buffer target; then -- crucially for a RETURN -- OR
        over a divergent PHI merge. A function may return ``ϕ(buf_A, buf_B)`` where
        each branch has its own alloc site (e.g. shell_quote's escape vs no-escape
        path, each its own ecalloc). The convergence-required heap key declines such
        a merge for STABLE store/read correlation, but for deciding whether the
        RESULT can carry taint, any reachable tainted buffer suffices -- the
        conservative, no-false-all-clear direction."""
        if expr is None or depth > 24:
            return False
        if self._pointee_tainted(ssaf, expr, tainted) is not None:
            return True
        var = self._as_single_ssa_var(expr)
        if var is None:
            reads = expr_reads(expr)
            var = reads[0] if len(reads) == 1 else None
        return self._var_buffer_tainted(ssaf, var, tainted, seen, depth + 1)

    def _var_buffer_tainted(self, ssaf: Any, var: Any, tainted: set, seen: set, depth: int) -> bool:
        if var is None or depth > 24:
            return False
        vk = self._ssa_var_key(var)
        if vk in seen:
            return False
        seen.add(vk)
        try:
            d = ssaf.get_ssa_var_definition(var)
        except Exception:
            d = None
        if d is None:
            return False
        dop = op_name(d)
        if "CALL" in dop:
            key = self._recognized_alloc_key(ssaf, var, d)
            return key is not None and self._key_tainted(key, tainted)
        if dop == "MLIL_SET_VAR_SSA":
            return self._expr_buffer_tainted(ssaf, getattr(d, "src", None), tainted, seen, depth)
        if "PHI" in dop:
            return any(self._var_buffer_tainted(ssaf, sv, tainted, seen, depth + 1)
                       for sv in (getattr(d, "src", None) or []))
        return False

    def _recognized_alloc_key(self, ssaf: Any, var: Any, d: Any):
        """``("heap", call_addr)`` if call-def *d* defines *var* via a recognized
        allocator -- a modeled / size-resolvable allocator (malloc, C++ new,
        single-arg wrapper) OR a callee whose name carries an allocator hint
        (malloc/calloc/realloc/ecalloc/xmalloc/strdup ...) OR a callee whose MODEL
        declares a buffer return ('*ret', #806). The name hint catches the 2-arg
        calloc family (e.g. less's `ecalloc`) the size-focused _dest_alloc misses;
        the model-declared return catches the buffer-returning helpers whose name
        hints at nothing (a concat helper) but whose model asserts the returned
        pointer is a buffer the callee wrote -- without it such a call site could
        never be re-resolved from a later pointer (its key would exist but be
        unreachable). Deliberately NOT the bare pointer-return heuristic -- too
        broad for a STABLE buffer key (it would conflate the returns of distinct
        non-allocator pointer functions and over-link). None for a non-allocator
        return pointer."""
        if d is None or "CALL" not in op_name(d):
            return None
        callee = self._callee_name(self._resolve_direct_target(d)) or ""
        callee_nm = callee.split("@", 1)[0].lstrip("_").lower()
        if (any(tok in callee_nm for tok in self._ALLOC_NAME_HINTS)
                or self._dest_alloc(ssaf, var)[0] is not None
                or self._model_declares_returned_buffer(callee)):
            return ("heap", int(getattr(d, "address", 0)))
        return None

    def _model_declares_returned_buffer(self, callee: str | None) -> bool:
        """True when *callee*'s model declares a buffer return -- a ``'*ret'``
        token (grammar: the pointee of the returned pointer) in a ``propagates``
        or ``sources`` rule. The model is asserting "this call hands back a
        pointer into a buffer I wrote", which makes the CALL SITE a buffer
        identity: the same ``("heap", call_addr)`` key space the allocator-name
        hints produce, so a later pointer derived from this call (a copy, an
        indexed cursor, a PHI entry) re-resolves to the key and correlates.
        Looked up through the SAME ``lookup_model``/name path the apply site uses,
        so recognition can never disagree with the model actually applied."""
        if not callee:
            return False
        _, model = lookup_model(self.models, callee)
        if not model:
            return False
        toks = [r.get("to") for r in (model.get("propagates") or [])]
        toks += [s.get("to") for s in (model.get("sources") or [])]
        return "*ret" in toks

    @staticmethod
    def _returned_buffer_key(ins: Any) -> tuple:
        """The buffer identity a model-declared buffer return ('*ret') is keyed
        by: the CALL SITE itself, in the ``("heap", addr)`` space
        :meth:`_heap_buffer_key` produces for allocator calls (and the store path
        keys through). One definition shared by every place a model's ``'*ret'``
        token is applied, so the key a model creates is always the key a later
        pointer re-resolves to through :meth:`_recognized_alloc_key`."""
        return ("heap", int(getattr(ins, "address", 0)))

    @staticmethod
    def _key_tainted(key: Any, tainted: set) -> bool:
        if (key, None) in tainted:
            return True
        return any(n[0] == key for n in tainted)

    # -- operand-role classification at a sink (#163) ----------------------
    # A tainted value can reach an overflow sink either as the buffer/length
    # OPERAND (a real overflow) or merely as an array INDEX/offset inside a
    # pointer computation (`base + i*stride`) -- an out-of-bounds access risk,
    # not an unbounded/length overflow. arg_taint() flattens the whole arg
    # expression, so without role analysis both look identical and the index
    # case over-states the overflow_* class.

    def _expr_has_taint(self, expr: Any, tainted: set) -> bool:
        """Any tainted SSA read anywhere in *expr* (BN's vars_read is recursive)."""
        for r in expr_reads(expr):
            if (var_key(r), getattr(r, "version", None)) in tainted or (var_key(r), None) in tainted:
                return True
        return False

    def _ptr_base_like(self, ssaf: Any, expr: Any, depth: int = 0) -> bool:
        """*expr* is (or resolves to) a base POINTER -- an address, not a scalar
        index/length. Conservative: recognizes &var / const-ptr / extern-ptr and
        SSA copies of those, plus stack-pointer vars via _pointee_var."""
        if expr is None or depth > 8:
            return False
        op = op_name(expr)
        if "ADDRESS_OF" in op or "CONST_PTR" in op or "EXTERN_PTR" in op:
            return True
        if is_ssa_var(expr):
            if self._pointee_var(ssaf, expr, depth + 1) is not None:
                return True
            try:
                d = ssaf.get_ssa_var_definition(expr)
            except Exception:
                d = None
            if d is not None and op_name(d) == "MLIL_SET_VAR_SSA":
                return self._ptr_base_like(ssaf, getattr(d, "src", None), depth + 1)
        return False

    def _is_scaled_taint(self, expr: Any, tainted: set, depth: int = 0) -> bool:
        """*expr* is a tainted value scaled by a constant stride -- ``i * const``
        or ``i << const`` -- the unambiguous array-index signal (a base pointer is
        never multiplied/shifted by a stride)."""
        if expr is None or depth > 6:
            return False
        op = op_name(expr)
        if op in ("MLIL_MUL", "MLIL_LSL"):
            left = getattr(expr, "left", None)
            right = getattr(expr, "right", None)
            if self._int_const(right) is not None and self._expr_has_taint(left, tainted):
                return True
            if op == "MLIL_MUL" and self._int_const(left) is not None and self._expr_has_taint(right, tainted):
                return True
        return False

    def _is_scaled_index(self, ssaf: Any, expr: Any, depth: int = 0) -> bool:
        """*expr* is a stride-scaled index ``i * const`` / ``i << const`` (taint-
        AGNOSTIC -- the index of a descriptor-array element is usually a plain loop
        counter, not tainted). A base pointer is never multiplied/shifted by a
        stride, so this is the unambiguous array-element-access signal. Follows one
        SSA copy hop so a `t = idx*0x20` temp is recognized."""
        if expr is None or depth > 6:
            return False
        op = op_name(expr)
        if op in ("MLIL_MUL", "MLIL_LSL"):
            if self._int_const(getattr(expr, "right", None)) is not None:
                return True
            if op == "MLIL_MUL" and self._int_const(getattr(expr, "left", None)) is not None:
                return True
        var = expr if is_ssa_var(expr) else None
        if var is None:
            # an MLIL_VAR_SSA expression wrapper (what an ADD operand actually is)
            reads = expr_reads(expr)
            var = reads[0] if len(reads) == 1 else None
        if var is not None:
            try:
                d = ssaf.get_ssa_var_definition(var)
            except Exception:
                d = None
            if d is not None and op_name(d) == "MLIL_SET_VAR_SSA":
                return self._is_scaled_index(ssaf, getattr(d, "src", None), depth + 1)
        return False

    def _elem_addr_parts(self, ssaf: Any, expr: Any, field: int = 0, seen: set | None = None,
                         depth: int = 0):
        """Strip a stride-scaled array index and constant field offset from a
        DESCRIPTOR-ARRAY element address, returning ``(base_expr, field_offset)``.
        Recognizes both element-address shapes:
          - explicit index: ``base + idx*stride (+ field)`` -- one ADD operand is a
            stride-scaled index (_is_scaled_index), the other is the base;
          - running pointer: ``p = ϕ(base, p + stride); [p + field]`` -- a loop-
            carried pointer whose PHI has a self+const stride increment; the base is
            the PHI's entry operand (back-edge cut by *seen*).
        Requires the scaled-index / stride signal, so a plain ``base + const`` or
        ``ptr + offset_var`` does NOT match (those are the existing pointee/buffer
        cases) -- this keeps element keying off arbitrary `[reg+off]` loads. Returns
        None when there is no array-element signal."""
        if expr is None or depth > 12:
            return None
        if seen is None:
            seen = set()
        op = op_name(expr)
        if op in ("MLIL_ADD", "MLIL_SUB"):
            left = getattr(expr, "left", None)
            right = getattr(expr, "right", None)
            rc = self._int_const(right)
            if rc is not None:
                return self._elem_addr_parts(ssaf, left, field + (rc if op == "MLIL_ADD" else -rc),
                                             seen, depth + 1)
            lc = self._int_const(left)
            if lc is not None and op == "MLIL_ADD":
                return self._elem_addr_parts(ssaf, right, field + lc, seen, depth + 1)
            # neither operand constant: one is the stride-scaled index, the other
            # the base pointer.
            for base_cand, idx_cand in ((left, right), (right, left)):
                if self._is_scaled_index(ssaf, idx_cand):
                    return (base_cand, field)
            return None
        if is_ssa_var(expr):
            vk = self._ssa_var_key(expr)
            if vk in seen:
                return None
            seen.add(vk)
            try:
                d = ssaf.get_ssa_var_definition(expr)
            except Exception:
                d = None
            if d is None:
                return None
            dop = op_name(d)
            if dop == "MLIL_SET_VAR_SSA":
                return self._elem_addr_parts(ssaf, getattr(d, "src", None), field, seen, depth + 1)
            if "PHI" in dop:
                # A loop-carried element pointer, in either compiler shape:
                #   (a) recomputed:  ϕ(base, base + idx*stride) -- one operand is
                #       itself an element address (base + scaled index);
                #   (b) running ptr: ϕ(base, self + stride) -- one operand is
                #       `self ± const`, the other is the entry base.
                operands = getattr(d, "src", None) or []
                # (a): an operand that resolves as an element address gives the base.
                for sv in operands:
                    if self._ssa_var_key(sv) in seen:
                        continue
                    r = self._elem_addr_parts(ssaf, sv, field, seen, depth + 1)
                    if r is not None:
                        return r
                # (b): the stride-bump operand confirms iteration; the OTHER is base.
                entry = None
                has_stride = False
                for sv in operands:
                    if self._is_self_plus_const(ssaf, sv, expr):
                        has_stride = True
                    elif entry is None:
                        entry = sv
                if has_stride and entry is not None:
                    return (entry, field)
            return None
        reads = expr_reads(expr)
        if len(reads) == 1:
            return self._elem_addr_parts(ssaf, reads[0], field, seen, depth + 1)
        return None

    def _is_self_plus_const(self, ssaf: Any, sv: Any, phi_var: Any, depth: int = 0) -> bool:
        """*sv* (a PHI operand) is ``phi_var (+/-) const`` -- the running-pointer
        stride increment -- so its def chases back through copies/`+const` to
        phi_var itself."""
        if sv is None or depth > 8:
            return False
        if is_ssa_var(sv) and self._ssa_var_key(sv) == self._ssa_var_key(phi_var):
            return depth > 0  # reached phi_var THROUGH a +const step, not the bare operand
        if is_ssa_var(sv):
            try:
                d = ssaf.get_ssa_var_definition(sv)
            except Exception:
                d = None
            if d is not None and op_name(d) == "MLIL_SET_VAR_SSA":
                return self._is_self_plus_const(ssaf, getattr(d, "src", None), phi_var, depth + 1)
            return False
        op = op_name(sv)
        if op in ("MLIL_ADD", "MLIL_SUB"):
            left = getattr(sv, "left", None)
            right = getattr(sv, "right", None)
            if self._int_const(right) is not None:
                return self._is_self_plus_const(ssaf, left, phi_var, depth + 1)
            if op == "MLIL_ADD" and self._int_const(left) is not None:
                return self._is_self_plus_const(ssaf, right, phi_var, depth + 1)
        return False

    def _elem_field_key(self, ssaf: Any, addr: Any):
        """Resolve a descriptor-array element-field address ``[base + idx*stride +
        field]`` to a stable coarse key ``("elem", base_key, base_off + field)``.
        Uses _addr_base_offset for the base, so an alloca'd / aligned VLA base
        (``(rsp - n + 7) & ~7`` -- the `struct hdr[hdr_num]` idiom) that
        _buffer_target cannot name still keys. None if *addr* is not an element
        address or its base is unresolvable. The key conflates all elements of the
        array's SAME field (may-analysis: a store to any element's field links a
        load of that field at any index -- the descriptor-array correlation #319b),
        but never two distinct arrays (distinct base keys) or two distinct fields."""
        parts = self._elem_addr_parts(ssaf, addr)
        if parts is None:
            return None
        base_expr, field = parts
        ba = self._addr_base_offset(ssaf, base_expr)
        if ba is None:
            return None
        return ("elem", ba[0], ba[1] + field)

    def _taint_operand_roles(self, ssaf: Any, expr: Any, tainted: set, *,
                             pointer_arg: bool = False, depth: int = 0) -> set[str]:
        """Roles in which tainted reads appear in *expr*: ``{'index'}`` and/or
        ``{'value'}``. ``index`` = a tainted (possibly stride-scaled) offset added
        to a base pointer; ``value`` = a tainted read anywhere else (the buffer
        base/pointee or a plain scalar/length). Conservative -- anything not
        provably an index counts as ``value`` so overflow_* is never under-stated.

        ``pointer_arg`` is set when the whole arg is known to be a source POINTER
        (an overflow_unbounded sink): then a stride-scaled tainted offset added to
        any UNtainted base is array indexing even when the base sits in a register
        we can't prove is a pointer. It stays False for scalar/length args so a
        computed length like ``header + count*elem`` is never taken for an index."""
        roles: set[str] = set()
        if expr is None or depth > 8 or not self._expr_has_taint(expr, tainted):
            return roles
        op = op_name(expr)
        # Resolve a bare var read -- a raw SSAVariable OR its MLIL_VAR_SSA
        # expression wrapper (what real BN emits as a call arg) -- to its
        # definition, so an address computed into a temp
        # (`t = base + i*stride; sink(t)`) is analyzed structurally instead of
        # being treated as an opaque tainted leaf. Without this the detector only
        # ever saw inline-arithmetic args and missed the real BN shape (#163).
        ssa_var = self._as_single_ssa_var(expr)
        if ssa_var is not None:
            try:
                d = ssaf.get_ssa_var_definition(ssa_var)
            except Exception:
                d = None
            if d is not None and op_name(d) == "MLIL_SET_VAR_SSA":
                return self._taint_operand_roles(ssaf, getattr(d, "src", None), tainted,
                                                 pointer_arg=pointer_arg, depth=depth + 1)
            roles.add("value")  # tainted leaf used directly
            return roles
        if op in ("MLIL_ADD", "MLIL_SUB"):
            left = getattr(expr, "left", None)
            right = getattr(expr, "right", None)
            l_taint = self._expr_has_taint(left, tainted)
            r_taint = self._expr_has_taint(right, tainted)
            # Pointer-arg only: a tainted offset added to an untainted base is
            # indexing for source-pointer args. Recognized pointer bases can use
            # unscaled offsets (`base + i`); unknown/register bases require a
            # stride-scaled offset. Scalar length args never use these pointer-base
            # shortcuts, so `CONST_PTR(k) + tainted_len` stays a value/length.
            if pointer_arg:
                if self._ptr_base_like(ssaf, left) and not l_taint and r_taint:
                    roles.add("index")
                    return roles
                if op == "MLIL_ADD" and self._ptr_base_like(ssaf, right) and not r_taint and l_taint:
                    roles.add("index")
                    return roles
                if self._is_scaled_taint(left, tainted) and not r_taint:
                    roles.add("index")
                    return roles
                if self._is_scaled_taint(right, tainted) and not l_taint:
                    roles.add("index")
                    return roles
            roles |= self._taint_operand_roles(ssaf, left, tainted, pointer_arg=pointer_arg, depth=depth + 1)
            roles |= self._taint_operand_roles(ssaf, right, tainted, pointer_arg=pointer_arg, depth=depth + 1)
            return roles
        subs = list(getattr(expr, "operands", None) or [])
        if not subs:
            subs = [s for s in (getattr(expr, "left", None), getattr(expr, "right", None)) if s is not None]
        if subs:
            for sub in subs:
                roles |= self._taint_operand_roles(ssaf, sub, tainted, pointer_arg=pointer_arg, depth=depth + 1)
            return roles
        roles.add("value")  # tainted leaf wrapper, used as a value
        return roles

    def _sink_taint_is_index_only(self, ssaf: Any, expr: Any, tainted: set, *, pointer_arg: bool = False) -> bool:
        """True iff every tainted read in the sink arg *expr* is an array
        index/offset and the pointee buffer itself is not tainted -- the case
        overflow_* over-states (#163). ``pointer_arg`` enables register-base index
        detection for source-pointer (overflow_unbounded) args."""
        if self._pointee_tainted(ssaf, expr, tainted) is not None:
            return False  # buffer contents tainted -> a real overflow operand
        return self._taint_operand_roles(ssaf, expr, tainted, pointer_arg=pointer_arg) == {"index"}

    # -- reused/aliased-slot length: propagation fact, not overflow verdict --
    # #307 FP-1. `bn taint` is a PROPAGATION tool, not an overflow *detector*:
    # it shows where taint flows and defers the overflow-vs-bounded judgement
    # (which needs control-dependence the engine lacks) to the model/agent. When
    # an `overflow_len` length reads an address-taken ("aliased") stack slot
    # whose taint arrived version-agnostically -- a `(var_key, None)` entry an
    # out-param call leaves when it writes through `&slot` -- and that slot has a
    # COMPETING in-function writer (the in-loop bounded store in the #307 shape),
    # the reaching definition is path-ambiguous. The taint genuinely reaches the
    # length via a path-insensitive reused-slot memory merge, but the engine
    # cannot stand behind an "attacker-controlled length" *verdict*. So the class
    # is neutralized to `tainted_len` (a tainted value reaches this length;
    # overflow determination deferred) while the SAME taint-reaches-arg flow
    # stays in reached_sinks -- nothing is hidden, so no false-negative.

    def _aliased_slot_var(self, ssaf: Any, expr: Any, depth: int = 0):
        """Follow pure SSA copies from *expr* to a read of an address-taken
        (aliased) local variable (``MLIL_VAR_ALIASED``), returning that base
        SSAVariable, else None. Recognizes ``len = slot@mem; memcpy(.., len)``
        through any chain of bare ``SET_VAR_SSA`` copies (the wpa_receive shape:
        ``x2 = x3; x3 = var_180 @ mem``)."""
        if expr is None or depth > 8:
            return None
        if op_name(expr) == "MLIL_VAR_ALIASED":
            reads = expr_reads(expr)
            if reads:
                return reads[0]
            return getattr(expr, "src", None) or getattr(expr, "var", None)
        v = self._as_single_ssa_var(expr)
        if v is None:
            reads = expr_reads(expr)
            v = reads[0] if len(reads) == 1 else None
        if v is None:
            return None
        try:
            d = ssaf.get_ssa_var_definition(v)
        except Exception:
            d = None
        if d is None or op_name(d) != "MLIL_SET_VAR_SSA":
            return None
        return self._aliased_slot_var(ssaf, getattr(d, "src", None), depth + 1)

    def _slot_has_direct_writer(self, ssaf: Any, instrs: Any, slot_key: Any) -> bool:
        """True if some instruction DIRECTLY writes the aliased slot keyed
        *slot_key* -- an ``MLIL_SET_VAR_ALIASED[_FIELD]`` or a ``STORE`` through
        ``&slot`` (unified by :meth:`_field_store_slot`). The out-param CALL that
        seeded the ``(k, None)`` taint is NOT one of these (it writes through a
        passed ``&slot`` arg, not a store the callee's frame owns), so a direct
        writer here is a COMPETING reaching def -- exactly the #307 in-loop
        bounded store that makes the version-agnostic taint path-ambiguous."""
        for ins in instrs:
            slot = self._field_store_slot(ssaf, ins)
            if slot is not None and slot[0] == slot_key:
                return True
        return False

    def _length_is_reused_aliased_slot(self, ssaf: Any, instrs: Any,
                                       length_expr: Any, tainted: set) -> bool:
        """The ambiguous #307 shape: the memcpy-family LENGTH reads an aliased
        stack slot whose taint arrived version-agnostically (``(k, None)``, from
        an out-param write through ``&slot``) AND the slot has a competing
        in-function writer, so the reaching def is path-ambiguous. Targeted: a
        clean single-def aliased slot (sole out-param writer, no competing store)
        stays ``overflow_len`` -- its length really is attacker-controlled with
        no ambiguity; a plain versioned tainted length carries no ``(k, None)``
        entry and is untouched."""
        aliased = self._aliased_slot_var(ssaf, length_expr)
        if aliased is None:
            return False
        slot_key = var_key(aliased)
        if (slot_key, None) not in tainted:
            return False
        return self._slot_has_direct_writer(ssaf, instrs, slot_key)

    # -- global/static buffers as taint locations -------------------------
    # A global buffer is referenced by an absolute address (MLIL_CONST_PTR), which
    # _pointee_var (stack-only) misses. We make it a single coarse taint location
    # keyed ("global", base_addr). Precise offset/aliasing is deliberately out of
    # scope (the domain of a heavyweight whole-program/CPG analyzer); we
    # over-approximate the whole buffer.

    def _heap_buffer_key(self, ssaf: Any, expr: Any):
        """Key a HEAP buffer by its allocation site: ``("heap", alloc_call_addr)``
        when *expr* is (or chases through pure copies + CONSTANT pointer arithmetic
        to) a var that a recognized allocator CALL defines. Lets a store through a
        heap pointer and a later read of a pointer from the SAME alloc site
        correlate -- the heap analogue of the stack-Variable / global keys, for the
        escape/encode-helper (#319a) and heap-output-buffer (#376) idioms. None if
        not a recognized heap allocation. (The alloc-in-a-loop boundary -- one call
        address producing distinct buffers per iteration -- is a known over-link
        risk handled separately if the corpus sweep shows it.)"""
        return self._heap_key_expr(ssaf, expr, 0, set())

    @staticmethod
    def _ssa_var_key(var: Any):
        """A hashable identity for an SSA var (Variable identity + version), so the
        chaser can break a loop back-edge (`cursor = ϕ(buf, cursor + 1)`) without a
        str-format dependency."""
        vv = getattr(var, "var", None)
        ident = getattr(vv, "identifier", None)
        base = ident if ident is not None else str(vv)
        return (base, getattr(var, "version", None))

    def _heap_key_expr(self, ssaf: Any, expr: Any, depth: int, seen: set):
        if expr is None or depth > 24:
            return None
        # `base + index` / `base - off`: the heap buffer is the POINTER operand; the
        # index (constant OR not) just selects a byte within it, so the whole buffer
        # is one coarse key (like a stack buffer at `&buf + i`). This is what makes
        # an indexed escape/encode-copy loop `out[i] = ...` resolve to `out`'s alloc.
        if op_name(expr) in ("MLIL_ADD", "MLIL_SUB"):
            for operand in (getattr(expr, "left", None), getattr(expr, "right", None)):
                k = self._heap_key_expr(ssaf, operand, depth + 1, seen)
                if k is not None:
                    return k
            return None
        var = self._as_single_ssa_var(expr)
        if var is None:
            reads = expr_reads(expr)
            var = reads[0] if len(reads) == 1 else None
        return self._heap_key_var(ssaf, var, depth, seen)

    def _heap_key_var(self, ssaf: Any, var: Any, depth: int, seen: set):
        """Chase an SSA var's definition to a recognized allocator's call site."""
        if var is None or depth > 24:
            return None
        vk = self._ssa_var_key(var)
        if vk in seen:  # loop back-edge / already on this chase -- stop
            return None
        seen.add(vk)
        try:
            d = ssaf.get_ssa_var_definition(var)
        except Exception:
            d = None
        if d is None:
            return None
        dop = op_name(d)
        if "CALL" in dop:
            return self._recognized_alloc_key(ssaf, var, d)
        # `ptr = base +/- index` / a pure SSA copy: chase the source expr so an
        # indexed escape/encode loop (`out = ecalloc(...); ...; cursor = out + i;
        # *cursor = ...`) resolves to `out`'s alloc site.
        if dop == "MLIL_SET_VAR_SSA":
            return self._heap_key_expr(ssaf, getattr(d, "src", None), depth + 1, seen)
        # PHI -- a loop-carried or branch-merged cursor (`cursor = ϕ(buf, cursor+1)`,
        # the indexed escape-loop idiom in less's shell_quote). Follow EVERY operand;
        # the back-edge operand re-enters `cursor` and is cut by `seen`, while the
        # entry operand reaches the buffer's alloc site. Key ONLY if every operand
        # that resolves agrees on ONE alloc site -- divergent allocs (`p = c ? a : b`
        # over two DISTINCT buffers) stay unkeyed, so a genuine merge is not
        # over-linked.
        if "PHI" in dop:
            keys = set()
            for src_var in getattr(d, "src", None) or []:
                k = self._heap_key_var(ssaf, src_var, depth + 1, seen)
                if k is not None:
                    keys.add(k)
            return next(iter(keys)) if len(keys) == 1 else None
        return None

    def _buffer_target(self, ssaf: Any, expr: Any):
        """Resolve a pointer expr to ``(key, label)`` for the buffer it points at:
        a stack Variable (preferred — keeps its name), a writable global, or a
        heap buffer keyed by allocation site. None if none. The key is what taint
        nodes are keyed on; label is for provenance.

        Memoized per ``(function, SSA expr_index)`` (#420): the resolution is purely
        structural -- it walks the SSA def graph and never consults the taint set --
        so it is identical on every fixpoint pass, yet the forward fixpoint reaches
        it (through ``_pointee_tainted``) on every escape check every iteration,
        where it dominates large-function runs. The function token is set on entry
        to each fixpoint run so a callee's expr_index can't alias a caller's."""
        tok = self._bt_func_token
        xi = getattr(expr, "expr_index", None)
        ck = (tok, xi) if (tok is not None and xi is not None) else None
        if ck is not None and ck in self._bt_cache:
            return self._bt_cache[ck]
        result = self._buffer_target_impl(ssaf, expr)
        if ck is not None:
            self._bt_cache[ck] = result
        return result

    def _buffer_target_impl(self, ssaf: Any, expr: Any):
        pv = self._pointee_var(ssaf, expr)
        if pv is not None:
            return (var_key(pv), var_label(pv))
        ga = self._global_addr(ssaf, expr)
        if ga is not None:
            return (("global", ga), f"glob_{hex(ga)}")
        hk = self._heap_buffer_key(ssaf, expr)
        if hk is not None:
            return (hk, f"heap_{hex(hk[1])}")
        return None

    def _global_addr(self, ssaf: Any, expr: Any, depth: int = 0) -> int | None:
        """The base address of a writable global/static buffer *expr* points at, or
        None. Follows an SSA copy chain to a constant pointer."""
        if expr is None or depth > 6:
            return None
        if "CONST_PTR" in op_name(expr):
            c = getattr(expr, "constant", None)
            if c is None:
                v = getattr(expr, "value", None)
                c = getattr(v, "value", v)
            try:
                addr = int(c)
            except Exception:
                return None
            return self._canon_global(addr)
        if is_ssa_var(expr):
            try:
                d = ssaf.get_ssa_var_definition(expr)
            except Exception:
                d = None
            if d is not None:
                return self._global_addr(ssaf, getattr(d, "src", None), depth + 1)
            return None
        reads = expr_reads(expr)
        if len(reads) == 1:
            try:
                d = ssaf.get_ssa_var_definition(reads[0])
            except Exception:
                d = None
            if d is not None:
                return self._global_addr(ssaf, getattr(d, "src", None), depth + 1)
        return None

    def _canon_global(self, addr: int) -> int | None:
        """Canonicalize a global address to its data-variable base (so different
        offsets into one buffer share a taint location) and require it be writable,
        so rodata strings / code pointers are not treated as buffers. Permissive
        when the BinaryView lacks these APIs (the unit-test fakes)."""
        getdv = getattr(self.bv, "get_data_var_at", None)
        if getdv is not None:
            try:
                dv = getdv(addr)
            except Exception:
                dv = None
            if dv is None:
                return None  # not a known data variable -> not a global buffer
            base = getattr(dv, "address", None)
            base = int(base) if base is not None else addr
            return base if self._is_writable(base) else None
        return addr if self._is_writable(addr) else None

    def _is_writable(self, addr: int) -> bool:
        fn = getattr(self.bv, "is_offset_writable", None)
        if fn is None:
            return True  # no introspection API (unit fakes) -> permissive
        try:
            return bool(fn(addr))
        except Exception:
            return False

    def _const_ptr_storage(self, addr: int) -> str:
        """Storage class of the constant address *addr* a writer operand points
        at: ``"readonly"`` (a string literal in ``.rodata``/code -- the bytes
        are established by the binary), ``"writable"`` (a ``.data``/``.bss``
        global any function may have filled -- NOT a literal, round 3 P1) or
        ``"unknown"`` (a view with neither section semantics nor segment
        permissions for the address: the unit fakes). Section semantics first
        (``get_sections_at``), then segment permissions."""
        secs = []
        getsecs = getattr(self.bv, "get_sections_at", None)
        if getsecs is not None:
            try:
                secs = list(getsecs(int(addr)) or [])
            except Exception:
                secs = []
        verdicts: set[str] = set()
        for sec in secs:
            sem = getattr(sec, "semantics", None)
            try:
                sem_i = int(sem)
            except (TypeError, ValueError):
                sem_i = {"ReadOnlyCodeSectionSemantics": 1, "ReadOnlyDataSectionSemantics": 2,
                         "ReadWriteDataSectionSemantics": 3}.get(
                    str(getattr(sem, "name", sem)), 0)
            if sem_i in (1, 2):
                verdicts.add("readonly")
            elif sem_i == 3:
                verdicts.add("writable")
        if "writable" in verdicts:
            return "writable"
        if "readonly" in verdicts:
            return "readonly"
        fn = getattr(self.bv, "is_offset_writable", None)
        if fn is None:
            return "unknown"
        try:
            return "writable" if fn(int(addr)) else "readonly"
        except Exception:
            return "unknown"

    # -- memory-SSA load/store correlation (Phase 3C) ---------------------
    # BN's memory SSA is a single global chain (a load's `src_memory` def is the
    # most-recent store, not necessarily an aliasing one), so we walk the version
    # chain ourselves and compare addresses. Used additively: it only ADDS taint
    # when a matching store wrote a tainted value, never removes coarse taint, so
    # it stays sound for a may-analysis while recovering heap/pointer flows the
    # AddressOf-only rule misses.

    def _int_const(self, expr: Any) -> int | None:
        if expr is None or "CONST" not in op_name(expr):
            return None
        c = getattr(expr, "constant", None)
        if c is None:
            return None
        try:
            return int(c)
        except Exception:
            return None

    def _const_ptr_addr(self, ssaf: Any, expr: Any, depth: int = 0) -> int | None:
        """Resolve a pointer expr to the constant address it points at, following
        an SSA copy chain. Unlike :meth:`_global_addr` this does NOT require the
        target be writable -- a rodata format string lives in read-only data."""
        if expr is None or depth > 6:
            return None
        if "CONST" in op_name(expr):
            return self._int_const(expr)
        if is_ssa_var(expr):
            try:
                d = ssaf.get_ssa_var_definition(expr)
            except Exception:
                d = None
            if d is not None:
                return self._const_ptr_addr(ssaf, getattr(d, "src", None), depth + 1)
            return None
        reads = expr_reads(expr)
        if len(reads) == 1:
            try:
                d = ssaf.get_ssa_var_definition(reads[0])
            except Exception:
                d = None
            if d is not None:
                return self._const_ptr_addr(ssaf, getattr(d, "src", None), depth + 1)
        return None

    def _const_format_string(self, ssaf: Any, expr: Any) -> str | None:
        """If *expr* points at a NUL-terminated constant string in the binary,
        return its text; else None (unknown/non-constant format -> stay
        conservative and treat every vararg as live).

        Only bytes in read-only storage are the format the call sees: a string
        in writable `.data` is merely its initial value, which any code may
        overwrite before the call (`strcpy(fmt, "%s")`), so it is no constant
        (#1050 review finding 1). The same storage test the backward walk uses
        for literal operands (`_const_ptr_storage`)."""
        addr = self._const_ptr_addr(ssaf, expr)
        if addr is None or self._const_ptr_storage(addr) == "writable":
            return None
        getstr = getattr(self.bv, "get_ascii_string_at", None)
        if getstr is not None:
            try:
                s = getstr(int(addr), 1)
                val = getattr(s, "value", None)
                if val is not None:
                    return str(val)
            except Exception:
                pass
        read = getattr(self.bv, "read", None)
        if read is not None:
            try:
                raw = read(int(addr), 512)
                if raw:
                    nul = raw.find(b"\x00")
                    return raw[: nul if nul >= 0 else len(raw)].decode("latin-1", "replace")
            except Exception:
                pass
        return None

    def _addr_base_offset(self, ssaf: Any, expr: Any, depth: int = 0):
        """Resolve an address expression to (base_var_key, offset), following SSA
        copies to a common root so a store and a load through different temp vars
        that ultimately hold the same pointer compare equal. None if unresolved."""
        if expr is None or depth > 8:
            return None
        op = op_name(expr)
        if "ADDRESS_OF" in op:
            v = getattr(expr, "src", None) or getattr(expr, "var", None)
            return (var_key(v), 0) if v is not None else None
        if "LOAD" in op:
            # A load of a constant address is a stable memory identity: two loads
            # of the same global pointer alias the same value, so `[*0xG + off]`
            # used at a recv site and a later re-load resolve to one slot key
            # (`("gload", 0xG)`, off) (#193 Part 1). Only a *constant* address
            # qualifies -- a load through a non-constant pointer is not a stable id.
            a = getattr(expr, "src", None)
            if a is not None and "CONST_PTR" in op_name(a):
                c = getattr(a, "constant", None)
                if c is None:
                    val = getattr(a, "value", None)
                    c = getattr(val, "value", val)
                try:
                    return (("gload", int(c)), 0)
                except (TypeError, ValueError):
                    return None
            return None
        if "CONST_PTR" in op:
            # A fixed-address global (struct) base: `*(G_struct + off)` where the
            # base is the literal address of a global struct, not a loaded pointer
            # (#193 redis `server.rdb_pipe_buff` shape). The address is constant, so
            # two loads of the same slot alias -- a stable identity like gload, and
            # SAFER: a literal base can't be re-pointed, so only the leaf-slot guard
            # is needed (no base-global re-point to check).
            c = getattr(expr, "constant", None)
            if c is None:
                val = getattr(expr, "value", None)
                c = getattr(val, "value", val)
            try:
                return (("gconst", int(c)), 0)
            except (TypeError, ValueError):
                return None
        if op in ("MLIL_ADD", "MLIL_SUB"):
            left = getattr(expr, "left", None)
            right = getattr(expr, "right", None)
            rc = self._int_const(right)
            base = self._addr_base_offset(ssaf, left, depth + 1)
            if base is not None and rc is not None:
                return (base[0], base[1] + (rc if op == "MLIL_ADD" else -rc))
            if op == "MLIL_ADD":
                lc = self._int_const(left)
                base2 = self._addr_base_offset(ssaf, right, depth + 1)
                if base2 is not None and lc is not None:
                    return (base2[0], base2[1] + lc)
            return None
        if is_ssa_var(expr):
            try:
                d = ssaf.get_ssa_var_definition(expr)
            except Exception:
                d = None
            if d is not None and op_name(d) == "MLIL_SET_VAR_SSA":
                sub = self._addr_base_offset(ssaf, getattr(d, "src", None), depth + 1)
                if sub is not None:
                    return sub
            return (var_key(expr), 0)
        reads = expr_reads(expr)
        if len(reads) == 1:
            return self._addr_base_offset(ssaf, reads[0], depth + 1)
        return None

    def _ptr_size(self) -> int:
        """Target pointer size in bytes (8 / 4), so the msghdr/iovec field offsets
        are laid out per the target ABI rather than a hard-coded 64-bit assumption."""
        sz = getattr(getattr(self.bv, "arch", None), "address_size", None)
        try:
            sz = int(sz)
        except (TypeError, ValueError):
            sz = 0
        return sz if sz in (4, 8) else 8

    def _field_store_slot(self, ssaf: Any, ins: Any):
        """The ``(base, offset)`` slot a store writes, unifying BN's store shapes:
        MLIL_STORE_SSA (dest is an address expr `&v + off`), MLIL_SET_VAR_(ALIASED_)FIELD
        (dest is the struct VAR -> `(v, 0)`, with the field byte offset in ``ins.offset``),
        and MLIL_SET_VAR_ALIASED (a whole address-taken scalar, e.g. a `size_t& capacity`
        out-param slot -> `(v, 0)`; #442). None when unresolvable."""
        op = op_name(ins)
        if "STORE" in op:
            return self._addr_base_offset(ssaf, getattr(ins, "dest", None))
        if op in ("MLIL_SET_VAR_FIELD", "MLIL_SET_VAR_ALIASED_FIELD"):
            ba = self._addr_base_offset(ssaf, getattr(ins, "dest", None))
            if ba is None:
                return None
            return (ba[0], ba[1] + int(getattr(ins, "offset", 0) or 0))
        if op == "MLIL_SET_VAR_ALIASED":
            dest = getattr(ins, "dest", None)
            return (var_key(dest), 0) if dest is not None else None
        return None

    def _value_stored_to_field(self, ssaf: Any, instrs: Any, target_slot: tuple, before_addr: int):
        """The value (src expr) of the nearest store to *target_slot* at an ADDRESS
        below *before_addr* (the recv callsite). Keyed on address, not MLIL
        instr_index: under basic-block reordering the MLIL linear index diverges
        from execution order, so an index-based "last store" can wrongly pick a
        not-reaching init (e.g. `iov_base = 0` emitted on a sibling branch) over the
        real buffer store that sits at a lower address. The msghdr/iovec setup is
        laid down physically just before the call, so the highest-address store
        below it is the reaching writer. CFG-insensitive (an unprovable pick is
        caught by the caller's honesty backstop -- it nudges rather than mis-seed).
        None if there is no such store."""
        best, best_addr = None, -1
        for ins in instrs:
            addr = getattr(ins, "address", None)
            if addr is None or addr >= before_addr or addr <= best_addr:
                continue
            slot = self._field_store_slot(ssaf, ins)
            if slot is not None and slot == target_slot:
                best, best_addr = getattr(ins, "src", None), addr
        return best

    def _recvmsg_iov_buffers(self, ssaf: Any, instrs: Any, call_ins: Any, hdr_idx: int) -> list:
        """The scatter-gather buffer pointer expr(s) a recvmsg/recvmmsg call fills,
        resolved statically: follow the msghdr at arg *hdr_idx* through its msg_iov
        field to the iovec, then each iovec's iov_base field to the buffer. Returns
        [] when the setup can't be resolved (dynamically-built iovec, etc.) -- the
        caller then falls back to the honest nudge, so a miss is never silent.

        Layout (glibc, ptr-size P): msghdr.msg_iov @ 2*P (after msg_name[P] +
        msg_namelen padded to P); iovec.iov_base @ 0, stride 2*P. recvmmsg's first
        mmsghdr has msg_hdr at offset 0, so the same offsets apply to its arg1."""
        params = self._call_params(call_ins)
        if hdr_idx < 0 or hdr_idx >= len(params):
            return []
        before = getattr(call_ins, "address", None)
        if before is None:
            return []
        hdr = self._addr_base_offset(ssaf, params[hdr_idx])
        if hdr is None:
            return []
        ptr = self._ptr_size()
        iov_ptr = self._value_stored_to_field(ssaf, instrs, (hdr[0], hdr[1] + 2 * ptr), before)
        if iov_ptr is None:
            return []
        iov = self._addr_base_offset(ssaf, iov_ptr)
        if iov is None:
            return []
        bufs = []
        # Walk consecutive iovec entries set up before the call (handles a single
        # iovec and a small static array); stop at the first entry with no iov_base
        # store, capped so a bogus base can't spin.
        for i in range(8):
            buf = self._value_stored_to_field(ssaf, instrs, (iov[0], iov[1] + i * 2 * ptr), before)
            if buf is None:
                break
            bufs.append(buf)
        return bufs

    def _mem_phi_sources(self, defn: Any) -> list[int]:
        src = getattr(defn, "src_memory", None)
        if isinstance(src, (list, tuple)):
            try:
                return [int(x) for x in src]
            except Exception:
                return []
        out = []
        for x in (getattr(defn, "src", None) or []):
            try:
                out.append(int(getattr(x, "version", x)))
            except Exception:
                pass
        return out

    @staticmethod
    def _store_covers_load(s_off, s_w, l_off, l_w):
        """Byte-range relation of a store (offset ``s_off``, width ``s_w``) to a
        load (``l_off`` / ``l_w``) on a shared base: ``"cover"`` when the store
        fully determines the loaded bytes (a strong update that may kill taint),
        ``"overlap"`` when it writes only some of them (a WEAK update -- an
        untainted narrow store must NOT kill a wider tainted store underneath),
        or ``"disjoint"``. Unknown widths fall back to exact-offset identity, so a
        store with no ``.size`` (or a fake that never set one) matches the
        pre-width ``(base, offset)`` slot comparison exactly (#562)."""
        if s_w is None or l_w is None:
            return "cover" if s_off == l_off else "disjoint"
        try:
            s_lo, s_hi = s_off, s_off + int(s_w)
            l_lo, l_hi = l_off, l_off + int(l_w)
        except (TypeError, ValueError):
            return "cover" if s_off == l_off else "disjoint"
        if s_lo <= l_lo and s_hi >= l_hi:
            return "cover"
        if s_hi <= l_lo or s_lo >= l_hi:
            return "disjoint"
        return "overlap"

    def _walk_mem(self, ssaf, mv, la, lw, tainted, seen, depth):
        if mv is None or depth > 64:
            return None
        try:
            mv = int(mv)
        except Exception:
            return None
        if mv in seen:
            return None
        seen.add(mv)
        try:
            defn = ssaf.get_ssa_memory_definition(mv)
        except Exception:
            defn = None
        if defn is None:
            return None
        op = op_name(defn)
        if "STORE" in op:
            sa = self._addr_base_offset(ssaf, getattr(defn, "dest", None))
            if sa is not None and sa[0] == la[0]:
                rel = self._store_covers_load(sa[1], getattr(defn, "size", None), la[1], lw)
                if rel != "disjoint":
                    for r in expr_reads(getattr(defn, "src", None)):
                        if (var_key(r), getattr(r, "version", None)) in tainted or (var_key(r), None) in tainted:
                            return (var_key(r), getattr(r, "version", None))
                    # Untainted store: only a FULL cover strong-kills the loaded
                    # bytes. A partial overlap leaves the uncovered bytes to an
                    # earlier writer, so keep walking rather than declaring the load
                    # untainted -- a narrow untainted store must not mask a wider
                    # tainted store underneath (#562 false all-clear).
                    if rel == "cover":
                        return None  # matching store wrote untainted data -> not tainted via memory
            return self._walk_mem(ssaf, getattr(defn, "src_memory", None), la, lw, tainted, seen, depth + 1)
        if "MEM_PHI" in op:
            for sv in self._mem_phi_sources(defn):
                res = self._walk_mem(ssaf, sv, la, lw, tainted, seen, depth + 1)
                if res is not None:
                    return res
            return None
        return None  # opaque writer (call/intrinsic): handled by models, not here

    def _load_tainted_via_memory(self, ssaf, load_expr, tainted):
        """If a load reads bytes that a tainted store wrote (matched by address
        along the memory-version chain), return the source node; else None."""
        la = self._addr_base_offset(ssaf, getattr(load_expr, "src", None))
        mv = getattr(load_expr, "src_memory", None)
        if la is None or mv is None:
            return None
        lw = getattr(load_expr, "size", None)
        return self._walk_mem(ssaf, mv, la, lw, tainted, set(), 0)

    def _addr_base_var(self, ssaf: Any, expr: Any, depth: int = 0):
        """The root SSA var / Variable an address expression is based on (the
        ``base`` of ``*(base + off)``), following SSA copies. Like
        :meth:`_addr_base_offset` but returns the var itself, so the backward
        slice can inspect *its* definition (allocator? parameter?). None if the
        base is not a single resolvable variable."""
        if expr is None or depth > 8:
            return None
        op = op_name(expr)
        if "ADDRESS_OF" in op:
            return getattr(expr, "src", None) or getattr(expr, "var", None)
        if op in ("MLIL_ADD", "MLIL_SUB"):
            left = getattr(expr, "left", None)
            right = getattr(expr, "right", None)
            if self._int_const(right) is not None:
                return self._addr_base_var(ssaf, left, depth + 1)
            if op == "MLIL_ADD" and self._int_const(left) is not None:
                return self._addr_base_var(ssaf, right, depth + 1)
            return None
        if is_ssa_var(expr):
            try:
                d = ssaf.get_ssa_var_definition(expr)
            except Exception:
                d = None
            if d is not None and op_name(d) == "MLIL_SET_VAR_SSA":
                sub = self._addr_base_var(ssaf, getattr(d, "src", None), depth + 1)
                if sub is not None:
                    return sub
            return expr
        reads = expr_reads(expr)
        if len(reads) == 1:
            return self._addr_base_var(ssaf, reads[0], depth + 1)
        return None

    def _source_call_fills(self, ssaf: Any, call_ins: Any, la: tuple):
        """If *call_ins* is a modeled source whose output-pointer buffer
        (``*arg:N``) resolves to ``la``'s base, return the callee name; else
        None. Lets a backward field load recover the receive/fill API that
        produced the bytes (e.g. ``read(fd, buf, n)``) instead of dead-ending."""
        name = self._callee_name(self._resolve_direct_target(call_ins))
        mkey, model = lookup_model(self.models, name)
        if not model:
            return None
        params = self._call_params(call_ins)
        for sd in model.get("sources") or []:
            to = str(sd.get("to") or "")
            if to.startswith("*arg:"):
                idx = _try_arg_index(to)
                if idx is not None and idx < len(params):
                    ba = self._addr_base_offset(ssaf, params[idx])
                    if ba is not None and ba[0] == la[0]:
                        return mkey or name
        return None

    def _reaching_writer(self, ssaf, mv, la, lw, seen, depth):
        """Walk the memory-SSA chain backward from version *mv* for the writer of
        address *la* = ``(base, offset)`` loaded at width *lw*. Returns
        ``("store", defn)`` for a ``MLIL_STORE`` whose bytes FULLY COVER the loaded
        range, ``("source", call_defn, callee)`` for a modeled source call that
        fills la's buffer, or None when the chain ends without a recoverable writer
        (a genuinely unresolved field load, #158). A partial overlap is NOT the
        sole writer of the loaded value, so the walk continues past it to the
        wider/earlier store -- mirroring the forward twin ``_walk_mem``; stopping
        on overlap would report a narrow recent store as THE origin and mask a
        wider tainted writer underneath (a backward false-clean)."""
        if mv is None or depth > 64:
            return None
        try:
            mv = int(mv)
        except Exception:
            return None
        if mv in seen:
            return None
        seen.add(mv)
        try:
            defn = ssaf.get_ssa_memory_definition(mv)
        except Exception:
            defn = None
        if defn is None:
            return None
        op = op_name(defn)
        if "STORE" in op:
            sa = self._addr_base_offset(ssaf, getattr(defn, "dest", None))
            if sa is not None and sa[0] == la[0] \
                    and self._store_covers_load(sa[1], getattr(defn, "size", None), la[1], lw) == "cover":
                return ("store", defn)
            return self._reaching_writer(ssaf, getattr(defn, "src_memory", None), la, lw, seen, depth + 1)
        if "MEM_PHI" in op:
            for sv in self._mem_phi_sources(defn):
                res = self._reaching_writer(ssaf, sv, la, lw, seen, depth + 1)
                if res is not None:
                    return res
            return None
        if self._is_call(defn):
            # Opaque memory writer: only recoverable if it is a modeled source
            # filling la's buffer; otherwise the field load is unresolved.
            hit = self._source_call_fills(ssaf, defn, la)
            if hit is not None:
                return ("source", defn, hit)
        return None

    def _model_out_arg_indices(self, model: dict[str, Any]) -> list[int]:
        """Argument indices a model WRITES through (``*arg:N`` in a ``propagates``
        ``to``, a ``sources`` ``to``, the ``varargs`` ``to`` or a ``writes``
        entry -- the literal fill of a ``memset``/``bzero``), in rule order."""
        out: list[int] = []
        toks = [r.get("to") for r in (model.get("propagates") or [])]
        toks += [s.get("to") for s in (model.get("sources") or [])]
        toks.append((model.get("varargs") or {}).get("to"))
        toks += list(model.get("writes") or [])
        for t in toks:
            if isinstance(t, str) and t.startswith("*arg:"):
                i = _try_arg_index(t)
                if i is not None and i not in out:
                    out.append(i)
        return out

    def _mem_in(self, ins: Any) -> int | None:
        """The memory-SSA version *ins* READS (the state before it executes).
        ``src_memory`` where BN carries one (``MLIL_STORE_SSA``/``CALL_SSA``/
        ``LOAD_SSA``); otherwise ``ssa_memory_version``, which every MLIL-SSA
        instruction has and which is the ONLY form on ``MLIL_SET_VAR_ALIASED``
        / ``SET_VAR_ALIASED_FIELD`` / ``SET_VAR_FIELD`` -- the shape a
        ``buf[i] = c`` terminator lifts to, where a ``src_memory``-only step
        dead-ended the chain and dropped every writer before it (PR-928 review
        P1). ``None`` when neither is available (the unit fakes / a non-SSA
        instruction)."""
        for attr in ("src_memory", "ssa_memory_version"):
            v = getattr(ins, attr, None)
            if v is None or isinstance(v, (list, tuple)):
                continue
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _reaches_avoiding(src: Any, dst: Any, avoid: list[Any]) -> bool | None:
        """Can control flow run from just after instruction *src* to *dst*
        without executing any instruction in *avoid*? An instruction-granular
        search over the IL basic blocks (``start``/``end`` instruction-index
        ranges, ``outgoing_edges``), so a loop back-edge that re-enters the
        source block before the reset is a path (round 4 P2: a priming
        reset before a loop DOMINATES the `strtok(NULL)` yet the tainted
        call at the bottom of the body reaches it around the back-edge).
        None when the view exposes no block / edge information, or when a
        block the search leaves has successors Binary Ninja could not
        determine (a computed jump): an answer the caller must treat as
        "reachable", never as a kill. An *avoid* instruction without block
        information is ignored (it proves no cut)."""
        def _idx(i: Any) -> int | None:
            try:
                return int(getattr(i, "instr_index"))
            except (AttributeError, TypeError, ValueError):
                return None

        sb = getattr(src, "il_basic_block", None)
        si, di = _idx(src), _idx(dst)
        if sb is None or getattr(dst, "il_basic_block", None) is None or si is None or di is None:
            return None
        stop: dict[int, bool] = {}
        for a in avoid:
            ai = _idx(a)
            if ai is not None and getattr(a, "il_basic_block", None) is not None:
                stop[ai] = False
        stop[di] = True

        def _scan(block: Any, lo: int) -> bool | None:
            """True: *dst* is the first stop at or after *lo* in *block*;
            False: a reset is; None: control falls through to the successors."""
            hits = [i for i in stop if lo <= i < int(block.end)]
            return stop[min(hits)] if hits else None

        try:
            first = _scan(sb, si + 1)
            if first is not None:
                return first
            seen: set[tuple[int, int]] = set()
            work = [sb]
            while work:
                block = work.pop()
                # A block BN marks with undetermined outgoing edges (a register-
                # indirect jump) may continue anywhere; its enumerated edges are
                # not all of them, so they cannot prove disconnection.
                if getattr(block, "has_undetermined_outgoing_edges", False) is True:
                    return None
                for edge in list(block.outgoing_edges):
                    t = getattr(edge, "target", None)
                    if t is None:
                        return None     # an undetermined successor: unproven
                    key = (int(t.start), int(t.end))
                    if key in seen:
                        continue
                    seen.add(key)
                    r = _scan(t, int(t.start))
                    if r is True:
                        return True
                    if r is None:
                        work.append(t)
        except Exception:   # noqa: BLE001 - a view without edges/ranges proves nothing
            return None
        return False

    def _address_terms(self, expr: Any) -> tuple[list[Any], list[int]]:
        """The ``(locals, const_addrs)`` an address expression is BUILT ON: every
        ``&v`` / ``&v:field`` and every ``CONST_PTR`` that is a term of its
        pointer-arithmetic tree, at ANY offset -- constant or not. ``&line +
        rbx`` reads ``rbx``, and a walk that followed only that read dropped the
        buffer the sink actually consumes (round 4 P1); ``0x4040a0 + (i << 6)``
        is an element of a ``.bss`` array, not an index. Only ``ADD`` (either
        side) and the minuend of a ``SUB`` are descended: anything else (a
        multiply, a mask) is a value, not an address."""
        locals_: list[Any] = []
        addrs: list[int] = []

        def visit(e: Any, depth: int) -> None:
            if e is None or depth > 12:
                return
            op = op_name(e)
            if "ADDRESS_OF" in op:
                v = getattr(e, "src", None) or getattr(e, "var", None)
                if v is not None and all(var_key(v) != var_key(x) for x in locals_):
                    locals_.append(v)
            elif "CONST_PTR" in op:
                c = self._int_const(e)
                if c is not None and c not in addrs:
                    addrs.append(c)
            elif op == "MLIL_ADD":
                visit(getattr(e, "left", None), depth + 1)
                visit(getattr(e, "right", None), depth + 1)
            elif op == "MLIL_SUB":
                visit(getattr(e, "left", None), depth + 1)

        visit(expr, 0)
        return locals_, addrs

    def _folded_buffer_seeds(self, expr: Any) -> list[_BufferSeed]:
        """Backward seeds for the buffers a call operand is an address INTO:
        every local it is built on, at any offset (``popen(&cmd)``,
        ``popen(&line + off)``), and -- beside a variable offset, the
        ``g_hosts[i]`` element shape -- every writable global. A bare
        constant address with no reads stays the "reads no variable" seed
        error it was (no def-chain at all)."""
        locals_, addrs = self._address_terms(expr)
        seeds = [_BufferSeed(loc) for loc in locals_]
        if expr_reads(expr):
            seeds += [_BufferSeed(addr=a) for a in addrs
                      if self._const_ptr_storage(a) == "writable"]
        return seeds

    def _must_offset(self, ssaf: Any, expr: Any, base_key: Any,
                     visited: set | None = None, depth: int = 0) -> int | None:
        """The constant byte offset into local *base_key* that the pointer
        *expr* holds on EVERY path, else None -- the must-alias question
        coverage asks. ``&base`` is 0, ``&base:field`` / ``&base + c`` is the
        constant, an SSA copy is its source, a ``VAR_PHI`` only when every arm
        agrees (a loop back-edge or an arm pointing at another local does not).
        ``_ptr_aliases_local`` answers the may-question (is this a writer at
        all); a may-writer records its step but covers no path (round 4 P2:
        ``p = flag ? a : b; read(fd, p, 63)`` covered ``a`` on the ``b`` path)."""
        if expr is None or depth > 12:
            return None
        op = op_name(expr)
        if "ADDRESS_OF" in op:
            v = getattr(expr, "src", None) or getattr(expr, "var", None)
            if v is None or var_key(v) != base_key:
                return None
            try:
                return int(getattr(expr, "offset", 0) or 0) if op == "MLIL_ADDRESS_OF_FIELD" else 0
            except (TypeError, ValueError):
                return None
        if op in ("MLIL_ADD", "MLIL_SUB"):
            left, right = getattr(expr, "left", None), getattr(expr, "right", None)
            rc = self._int_const(right)
            if rc is not None and "PTR" not in op_name(right):
                off = self._must_offset(ssaf, left, base_key, visited, depth + 1)
                return None if off is None else off + (rc if op == "MLIL_ADD" else -rc)
            lc = self._int_const(left)
            if op == "MLIL_ADD" and lc is not None and "PTR" not in op_name(left):
                off = self._must_offset(ssaf, right, base_key, visited, depth + 1)
                return None if off is None else off + lc
            return None
        if "LOAD" in op:
            # A reload of a slot: a must-alias only when the store reaching
            # this load on the memory chain wrote a must-alias of &base
            # (`g_stash = buf; strcpy(g_stash, host)`). A call or an
            # unresolved store in between leaves it unproven.
            la = self._addr_base_offset(ssaf, getattr(expr, "src", None))
            if la is None:
                return None
            rw = self._reaching_writer(ssaf, getattr(expr, "src_memory", None), la,
                                       getattr(expr, "size", None), set(), 0)
            if rw is None or rw[0] != "store":
                return None
            # _reaching_writer is a MAY query: a PHI can select a store
            # from just one arm. Until all-path slot identity is modeled,
            # require a straight memory chain to that producer (#951).
            version = getattr(expr, "src_memory", None)
            for _ in range(MEM_CHAIN_BUDGET):
                try:
                    producer = ssaf.get_ssa_memory_definition(int(version))
                except (TypeError, ValueError):
                    return None
                if producer is None or "MEM_PHI" in op_name(producer):
                    return None
                if producer == rw[1]:
                    break
                version = self._mem_in(producer)
            else:
                return None
            return self._must_offset(ssaf, getattr(rw[1], "src", None), base_key,
                                     visited, depth + 1)
        if is_ssa_var(expr):
            key = (var_key(expr), getattr(expr, "version", None))
            visited = visited if visited is not None else set()
            if key in visited:
                return None      # a loop back-edge: the cursor moves
            visited = visited | {key}
            try:
                d = ssaf.get_ssa_var_definition(expr)
            except Exception:
                d = None
            if d is None:
                return None
            dop = op_name(d)
            if dop == "MLIL_SET_VAR_SSA":
                return self._must_offset(ssaf, getattr(d, "src", None), base_key, visited, depth + 1)
            if dop == "MLIL_VAR_PHI":
                offs = {self._must_offset(ssaf, a, base_key, visited, depth + 1)
                        for a in (getattr(d, "src", None) or [])}
                return offs.pop() if len(offs) == 1 else None
            return None
        reads = expr_reads(expr)
        if len(reads) == 1 and "LOAD" not in op:
            return self._must_offset(ssaf, reads[0], base_key, visited, depth + 1)
        return None

    def _var_byte_size(self, v: Any) -> int | None:
        """Byte size of local *v* (a Variable / SSAVariable) from its type, or
        None when the type carries no width (then a constant-offset store can
        never be shown to write the WHOLE variable)."""
        t = getattr(getattr(v, "var", v), "type", None)
        w = getattr(t, "width", None)
        if w is not None:
            try:
                return int(w)
            except (TypeError, ValueError):
                return None
        m = re.fullmatch(r"(?:unsigned\s+)?char\s*\[(0x[0-9a-fA-F]+|\d+)\]", str(t or "").strip())
        return int(m.group(1), 0) if m else None

    def _ptr_aliases_local(self, ssaf: Any, expr: Any, base_key: Any, *,
                           escaped: bool, escape_addrs: dict[int, frozenset[int]],
                           visited: set | None = None,
                           depth: int = 0) -> str:
        """Classify possible aliases of local buffer *base_key*.

        ``"yes"`` includes address-derived pointers and reloads whose input
        memory version has a remembered stash slot. Those slot sets are
        monotone may-alias evidence, NOT reaching pointer values: definite
        overwrites do not kill them, and the later ``escaped`` fallback can
        promote a held old LOAD copied through a second slot. These cases
        can fabricate a writer/source even in a complete slice, as on base
        5aba57ee; epoch-keyed sets only repair the direct one-slot future-stash
        repro, not general flow-sensitive pointer provenance.

        ``"no"`` excludes known unrelated addresses and values fixed before
        this frame existed; ``"maybe"`` retains unresolved aliases after an
        escape or a depth cut. The walk must disclose those, never drop them."""
        if expr is None:
            return "no"
        if depth > 12:
            return "maybe"
        op = op_name(expr)
        if "ADDRESS_OF" in op:
            v = getattr(expr, "src", None) or getattr(expr, "var", None)
            return "yes" if v is not None and var_key(v) == base_key else "no"
        if "LOAD" in op:
            a = getattr(expr, "src", None)
            if a is not None and "CONST_PTR" in op_name(a):
                c = self._int_const(a)
                if c is not None and c in escape_addrs.get(
                        getattr(expr, "src_memory", None), frozenset()):
                    return "yes"
            return "maybe" if escaped else "no"
        if "CONST" in op or "EXTERN_PTR" in op or "IMPORT" in op:
            return "no"
        if is_ssa_var(expr):
            key = (var_key(expr), getattr(expr, "version", None))
            visited = visited if visited is not None else set()
            if key in visited:
                return "no"     # a loop back-edge arm adds no new base
            visited.add(key)
            try:
                d = ssaf.get_ssa_var_definition(expr)
            except Exception:
                d = None
            if d is None:
                # A parameter / undefined register holds a value fixed before
                # this frame existed: it cannot be this frame's stack address
                # (an escape stores &buf AFTER entry; a reload of it is a LOAD).
                return "no"
            dop = op_name(d)
            if dop == "MLIL_SET_VAR_SSA":
                return self._ptr_aliases_local(
                    ssaf, getattr(d, "src", None), base_key, escaped=escaped,
                    escape_addrs=escape_addrs, visited=visited, depth=depth + 1)
            if dop == "MLIL_VAR_PHI":
                arms = [self._ptr_aliases_local(
                            ssaf, a, base_key, escaped=escaped, escape_addrs=escape_addrs,
                            visited=visited, depth=depth + 1)
                        for a in (getattr(d, "src", None) or [])]
                if "yes" in arms:
                    return "yes"
                return "maybe" if "maybe" in arms or not arms else "no"
            if self._is_call(d):
                # A pointer returned by a call: a modeled callee that derives
                # its return from an argument (`strchr(buf, c)`, `p = memcpy(buf, ..)`)
                # points where that argument did; an unmodeled callee handed the
                # buffer may return a pointer into it.
                params = self._call_params(d)
                name = self._callee_name(const_target(getattr(d, "dest", None)))
                _, model = lookup_model(self.models, name)
                handed = [self._ptr_aliases_local(
                              ssaf, p, base_key, escaped=escaped, escape_addrs=escape_addrs,
                              visited=set(visited), depth=depth + 1)
                          for p in params]
                if model is None:
                    if "yes" in handed or "maybe" in handed or escaped:
                        return "maybe"
                    return "no"
                for rule in model.get("propagates") or []:
                    if str(rule.get("to") or "") != "ret":
                        continue
                    fi = _try_arg_index(str(rule.get("from") or ""))
                    if fi is not None and fi < len(handed) and handed[fi] != "no":
                        return handed[fi]
                return "no"
            # Any other definition (a partial register write, a field read):
            # not a pointer the walk can follow.
            return "maybe" if escaped else "no"
        if op in ("MLIL_ADD", "MLIL_SUB"):
            # Pointer arithmetic: the pointer is whichever side is built on
            # &base. Anything else (a xor, a multiply, a narrowing) is a value,
            # not an address.
            answers = [self._ptr_aliases_local(
                           ssaf, getattr(expr, a, None), base_key, escaped=escaped,
                           escape_addrs=escape_addrs, visited=visited, depth=depth + 1)
                       for a in ("left", "right")]
            if "yes" in answers:
                return "yes"
            return "maybe" if "maybe" in answers else "no"
        reads = expr_reads(expr)
        if len(reads) == 1:
            # A single-variable wrapper (`MLIL_VAR_SSA`, a cast of one var).
            return self._ptr_aliases_local(
                ssaf, reads[0], base_key, escaped=escaped,
                escape_addrs=escape_addrs, visited=visited, depth=depth + 1)
        return "no"

    def _local_buffer_writers(self, ssaf: Any, base_key: Any, mv: Any,
                              base_size: int | None = None) -> dict[str, Any]:
        """Every writer of the local buffer *base_key* on the memory-SSA chain
        ending at version *mv* (the version the sink READS), newest first,
        plus whether that chain accounts for the buffer on EVERY path.

        GOVERNING RULE (PR-928 review rounds 1-3): the walk never EXCLUDES a
        store, out-argument, pointer base or callee it cannot classify with
        certainty. Every exclusion in the earlier rounds became a false
        ``safe_to_report_complete_slice: true`` -- a phi'd loop-store pointer,
        a modeled call reloading the stashed address, a ``.bss`` operand read
        as a literal. Precision is applied where it is provable (a keyed slot,
        a dominating definition, a pointer that cannot hold this frame's stack
        address); anything else is a ``maybe`` writer the consumer turns into
        a blocking ``address_of_local_unresolved`` leaf.

        The backward twin of the forward out-param handling: a slice that
        reaches ``p = &buf`` must continue through the bytes' producers, not
        stop at the pointer. Returns ``{"writers": [...], "covered": bool,
        "truncated": bool}`` where each writer is a tuple headed by its kind:

        * ``("store", defn)`` -- a store whose address is built on the buffer,
          in any of BN's store shapes (``MLIL_STORE_SSA`` through an address
          expr, ``MLIL_SET_VAR_ALIASED[_FIELD]`` / ``SET_VAR_FIELD`` for a
          stack slot written by name -- the ``buf[n] = 0`` terminator); any
          offset, constant or not (a byte-wise loop fill is many stores
          through a phi'd cursor);
        * ``("maybe_store", defn, why)`` -- a store through a pointer the walk
          cannot resolve (a reloaded or parameter pointer after ``&buf``
          escaped, a pointer an unmodeled call returned after being handed
          the buffer): it MAY write the bytes, and the slice is not complete
          until it is accounted for;
        * ``("fill", defn, mkey, params, from_idxs, is_source, len_idxs)`` --
          a modeled call writing ``*arg:N`` where ``params[N]`` resolves to
          the buffer (directly, through a phi'd pointer, or by reloading the
          slot ``&buf`` escaped into); ``from_idxs`` are the argument indices
          the model derives those bytes from (``propagates`` ``from`` + the
          varargs run), ``len_idxs`` the length arguments that decide how
          many of them (a tainted length cuts the command), ``is_source``
          when a ``sources`` rule fills it (a receive API);
        * ``("benign", defn, callee)`` -- a ``{}``-modeled call handed the
          buffer: its model promises it writes nothing TAINTED, not that it
          writes nothing, so it is recorded (not dropped) but neither covers
          a path nor supplies an origin;
        * ``("escape", defn)`` -- a store whose VALUE is the buffer's address
          (``[global] = &buf``, a stack descriptor slot): from here on any
          callee may write the bytes through the stashed pointer;
        * ``("opaque", defn, callee, why)`` -- a call that may have written
          the bytes and the engine cannot tell: unmodeled and handed the
          buffer's address (or a pointer that may alias it), unmodeled and
          run after that address escaped to memory, or modeled with an
          out-argument pointer the walk cannot resolve while the address is
          escaped (the backward mirror of the forward ``pointer_escape`` leaf).

        ``covered`` is False when some path from the sink back to the function
        entry crosses NO writer that accounts for the buffer -- the
        ``if (flag) fill(buf); sink(buf)`` shape, whose memory phi has a bare
        entry-state arm: the sink then reads whatever the stack held, and a
        slice that names only the fill's origin is not complete. Coverage is
        decided by the BYTES a writer provably writes, never by its kind
        (round 4 P2): a writer covers a path only when its pointer is a
        MUST-alias of ``&base + 0`` (``_must_offset`` -- every phi arm
        resolves to this base at offset 0). The current classifier accepts
        a modeled fill / source / ``writes`` call unless a guarded constant
        length (selected from its sink metadata) is zero or smaller than the
        recovered object. Source-only models without that metadata are not
        guarded. This is not a consumed-region initialization proof (#931).
        It also accepts an unmodeled call handed
        ``&base`` itself (its opaque leaf already blocks), a whole-variable
        ``SET_VAR_ALIASED`` or a store at least the variable's width. An
        escape writes no byte; an indexed store (``buf[n] = 0``), a
        constant-offset store or fill (``buf[63] = 0``, ``strcpy(buf + 60,
        ..)``), a may-alias writer (a phi that may point at another local, a
        maybe-store) and a ``{}``-modeled call are writers that cover nothing.
        Coverage is the greatest fixed point of the all-arm equations;
        escape is the least fixed point of the any-arm equations, including
        loop-carried stores of the address. ``truncated`` is True when graph
        discovery or either fixed point exhausts the shared
        ``MEM_CHAIN_BUDGET``; no incomplete solve certifies coverage.
        Deliberately does NOT stop at a full-cover writer: ``strcat`` after
        ``strcpy`` both write offset 0 and both feed the sink, and a may-analysis
        that drops the earlier one loses an origin. A partial store never masks
        anything either (mirrors ``_reaching_writer``)."""
        writers: list[tuple] = []     # oldest first while walking; reversed on return
        # Discover the graph before solving it; no provisional cycle proof is
        # ever published. Coverage descends from top, escape ascends from bottom.
        memo: dict[int, tuple[bool, bool]] = {}
        truncated = False
        # Remembered may-alias stash slots keyed by memory version. Consulting
        # a LOAD's version fixes the direct one-slot future-stash repro only.
        # No overwrite kills or held-LOAD provenance through another slot:
        # both can still invent sources, including on base 5aba57ee.
        escape_addrs: dict[int, frozenset[int]] = {}

        def _alias(expr: Any, escaped: bool) -> str:
            return self._ptr_aliases_local(ssaf, expr, base_key, escaped=escaped,
                                           escape_addrs=escape_addrs)

        def _params_touching(params: list[Any], escaped: bool) -> tuple[list[int], list[int]]:
            """``(yes, maybe)`` argument indices pointing into the buffer."""
            yes: list[int] = []
            maybe: list[int] = []
            for i, p in enumerate(params):
                a = _alias(p, escaped)
                if a == "yes":
                    yes.append(i)
                elif a == "maybe":
                    maybe.append(i)
            return yes, maybe

        def _at_base(params: list[Any], idxs: list[int]) -> bool:
            """Is one of *idxs* a must-alias of ``&base + 0``?"""
            return any(i < len(params) and self._must_offset(ssaf, params[i], base_key) == 0
                       for i in idxs)

        def _call_writer(defn: Any, escaped: bool) -> bool:
            """Record *defn* (a call) as a writer when it can be one; True if
            it was recorded as one that COVERS the path -- only a call whose
            written pointer is a must-alias of ``&base + 0``."""
            params = self._call_params(defn)
            yes, maybe = _params_touching(params, escaped)
            name = self._callee_name(const_target(getattr(defn, "dest", None)))
            mkey, model = lookup_model(self.models, name)
            callee = name or str(getattr(defn, "dest", "indirect"))
            if model is None:
                if yes:
                    why = "has no taint model and received the address of"
                elif maybe:
                    why = "has no taint model and received a pointer that may alias"
                elif escaped:
                    why = "has no taint model and ran after the address of"
                else:
                    return False
                writers.append(("opaque", defn, callee, why))
                return _at_base(params, yes)
            if not yes and not maybe:
                # Trusted to its model: a libc model declares what the callee
                # writes, and none of its pointers reaches the buffer -- not
                # even by reloading the slot the address escaped into.
                return False
            outs = self._model_out_arg_indices(model)
            if model_is_benign_marker(model):
                # `{}` promises no taint flows through the call, nothing about
                # whether it writes its pointer arguments (round 3: memset's
                # kin were `{}` and read as "writes nothing"). Recorded, so the
                # call is in the slice; it neither covers nor originates.
                writers.append(("benign", defn, callee))
                return False
            if any(i in maybe for i in outs):
                writers.append(("opaque", defn, callee,
                                "writes through a pointer argument the walk cannot "
                                "resolve (it may alias the escaped address of)"))
                return False
            hit = [i for i in outs if i in yes]
            if not hit:
                return False
            from_idxs: list[int] = []
            for rule in model.get("propagates") or []:
                if _try_arg_index(str(rule.get("to") or "")) in hit:
                    fi = _try_arg_index(str(rule.get("from") or ""))
                    if fi is not None and fi < len(params) and fi not in from_idxs:
                        from_idxs.append(fi)
            va = model.get("varargs")
            if va is not None and _try_arg_index(str(va.get("to") or "")) in hit:
                for fi in range(max(int(va.get("first_index", 0)), 0), len(params)):
                    if fi not in from_idxs:
                        from_idxs.append(fi)
            is_source = any(
                _try_arg_index(str(s.get("to") or "")) in hit
                for s in (model.get("sources") or []))
            # The length operand(s) of a bounded fill decide how many of the
            # input bytes land in the buffer -- `memcpy(buf, "lit", n)` with a
            # tainted n cuts the command (round 3 P3).
            sink = model.get("sink") or {}
            len_idxs: list[int] = []
            if sink.get("len_arg") is not None:
                len_idxs = [int(sink["len_arg"])]
            elif str(sink.get("class") or "") in ("overflow_len", "fortified_overflow"):
                len_idxs = [int(i) for i in (sink.get("tainted_args") or []) if isinstance(i, int)]
            len_idxs = [i for i in len_idxs if i < len(params) and i not in from_idxs]
            # First-step extent guard (#931), not a proven-write-extent model:
            # a requested I/O length still need not be the actual write length.
            count_scale: int | None = 1
            if mkey in ("wmemcpy", "wmemmove", "wmemcpy_chk", "wmemmove_chk"):
                # These models count wchar_t elements, not bytes. Use the
                # callee's recovered destination pointee width, never a host
                # wchar_t assumption (it differs across target ABIs).
                callee_fn = function_at(self.bv, const_target(getattr(defn, "dest", None)))
                callee_params = list(getattr(callee_fn, "parameter_vars", []) or [])
                dest_type = getattr(callee_params[0], "type", None) if callee_params else None
                width = getattr(getattr(dest_type, "target", None), "width", None)
                count_scale = int(width) if width is not None and int(width) > 0 else None
            short_fill = any(
                op_name(params[i]) in ("MLIL_CONST", "MLIL_CONST_PTR")
                and (int(getattr(params[i], "constant", 0)) <= 0
                     or (count_scale is not None
                         and (base_size is None or
                              int(getattr(params[i], "constant", 0)) * count_scale < base_size)))
                for i in len_idxs)
            writers.append(("fill", defn, mkey or name, params, from_idxs, is_source, len_idxs))
            # A fill at a constant offset (`strcpy(buf + 60, "x")`) or through
            # a pointer that may point at another local writes bytes of this
            # buffer on some path, not the buffer (round 4 P2).
            return not short_fill and _at_base(params, hit)

        def _store_covers(defn: Any, slot: tuple) -> bool:
            """A keyed store covers the buffer only when it writes ALL of it."""
            op = op_name(defn)
            if op == "MLIL_SET_VAR_ALIASED":
                return True
            if base_size is None or slot[1] != 0:
                return False
            try:
                width = int(getattr(defn, "size", None))
            except (TypeError, ValueError):
                return False
            return width >= base_size

        graph: dict[int, tuple[Any, list[int]]] = {}
        pending = [mv]
        work = 0
        while pending:
            try:
                v = int(pending.pop())
            except (TypeError, ValueError):
                truncated = True
                continue
            if v in graph:
                continue
            work += 1
            if work > MEM_CHAIN_BUDGET:
                truncated = True
                break
            try:
                defn = ssaf.get_ssa_memory_definition(v)
                deps = ([] if defn is None else
                        self._mem_phi_sources(defn) if "MEM_PHI" in op_name(defn)
                        else [self._mem_in(defn)])
                deps = [int(s) for s in deps]
            except Exception:
                defn, deps = None, []
                truncated = True
            graph[v] = (defn, deps)
            memo[v] = (True, False)
            escape_addrs[v] = frozenset()
            pending.extend(deps)

        def transfer(v: int) -> tuple[bool, bool, frozenset[int]]:
            defn, deps = graph[v]
            if defn is None:
                res = (False, False, frozenset())
            elif "MEM_PHI" in op_name(defn):
                arms = [memo.get(s, (False, False)) for s in deps]
                res = (bool(arms) and all(c for c, _ in arms), any(e for _, e in arms),
                       frozenset().union(*(escape_addrs.get(s, frozenset()) for s in deps)))
            else:
                cov, esc = memo.get(deps[0], (False, False))
                addrs = escape_addrs.get(deps[0], frozenset())
                op = op_name(defn)
                if "STORE" in op or op in _NAMED_STORE_OPS:
                    slot = self._field_store_slot(ssaf, defn)
                    if slot is not None and slot[0] == base_key:
                        writers.append(("store", defn))
                        cov = _store_covers(defn, slot) or cov
                    elif "STORE" in op:
                        # Not keyed to the buffer: the address may still be
                        # built on it (a phi'd loop cursor, `[p + i]` with
                        # `p = &buf`), or be a pointer the walk cannot pin.
                        a = _alias(getattr(defn, "dest", None), esc)
                        if a == "yes":
                            writers.append(("store", defn))
                            # Through a pointer built on &buf: covers only
                            # at a must-alias offset 0 with the full width --
                            # an indexed `buf[n] = 0` writes one byte.
                            if base_size is not None \
                                    and self._must_offset(ssaf, getattr(defn, "dest", None),
                                                          base_key) == 0:
                                try:
                                    cov = int(getattr(defn, "size", 0)) >= base_size or cov
                                except (TypeError, ValueError):
                                    pass
                        elif a == "maybe":
                            # Disclosed by its own blocking leaf; it covers
                            # nothing -- it may write elsewhere.
                            writers.append(("maybe_store", defn,
                                            "stores through a pointer the walk cannot "
                                            "resolve, which may alias"))
                    val = getattr(defn, "src", None)
                    if _alias(val, esc) != "no":
                        # From here every writer through the stashed pointer
                        # is disclosed (a reload is a fill, an unmodeled call
                        # is opaque). The escape itself writes no byte of the
                        # buffer, so it covers no path (round 4 P2).
                        writers.append(("escape", defn))
                        esc = True
                        if "STORE" in op:
                            c = self._const_ptr_addr(ssaf, getattr(defn, "dest", None))
                            if c is not None:
                                addrs = addrs | {int(c)}
                elif self._is_call(defn):
                    cov = _call_writer(defn, esc) or cov
                res = (cov, esc, addrs)
            return res

        # Solve escape and remembered escape slots first. Writer classification
        # can change as these facts grow, so only collect final writer rows.
        for escape_phase in (True, False):
            while True:
                changed = False
                for v in reversed(graph):
                    work += 1
                    if work > MEM_CHAIN_BUDGET:
                        truncated = True
                        break
                    cov, esc, addrs = transfer(v)
                    old_cov, old_esc = memo[v]
                    new = ((old_cov, old_esc or esc) if escape_phase
                           else (old_cov and cov, old_esc))
                    new_addrs = escape_addrs[v] | addrs
                    changed |= new != memo[v] or new_addrs != escape_addrs[v]
                    memo[v] = new
                    escape_addrs[v] = new_addrs
                writers.clear()
                if truncated or not changed:
                    break
            if truncated:
                break
        # One bounded collection pass retains the nearby escaped-call frontier
        # in a discovered straight-line suffix after a proof cut. It cannot
        # certify coverage or solve loop backedges: discovery-cut loops can
        # still lose the named loop-carried call (also on base 5aba57ee).
        for v in reversed(graph):
            cov, esc, addrs = transfer(v)
            memo[v] = (memo[v][0] and cov, memo[v][1] or esc)
            escape_addrs[v] = escape_addrs[v] | addrs
        covered = not truncated and memo.get(int(mv), (False, False))[0]
        writers.reverse()
        return {"writers": writers, "covered": covered, "truncated": truncated}

    def _field_base_is_alloc_or_param(self, ssaf: Any, func: Any, base_var: Any) -> bool:
        """True when *base_var* (the pointer a field load is based on) is itself
        defined by an allocator call or is a function parameter -- the two cases
        where a silent dead-end reads as "locally allocated / clean" and so
        warrants an explicit ``field_load_unresolved`` leaf (#158). A plain stack
        buffer is neither, so its load keeps the existing slice behavior."""
        if base_var is None:
            return False
        try:
            defn = ssaf.get_ssa_var_definition(base_var)
        except Exception:
            defn = None
        if defn is None:
            return self._param_index_of(func, base_var) is not None
        if self._is_call(defn):
            name = self._callee_name(self._resolve_direct_target(defn))
            _, model = lookup_model(self.models, name)
            if model and (model.get("sink") or {}).get("class") == "alloc_size":
                return True
            if name and ("malloc" in name or "alloc" in name):
                return True
        return False

    def _param_index_of(self, func: Any, v: Any) -> int | None:
        """Index of the function parameter that *v* (an SSAVariable/Variable) is,
        or None. Matches by identifier first, then storage+name."""
        base = getattr(v, "var", v)
        bid = getattr(base, "identifier", None)
        bstore = getattr(base, "storage", None)
        bname = str(getattr(base, "name", base))
        for i, p in enumerate(list(getattr(func, "parameter_vars", []) or [])):
            if bid is not None and getattr(p, "identifier", None) == bid:
                return i
            if bstore is not None and getattr(p, "storage", None) == bstore and str(getattr(p, "name", p)) == bname:
                return i
        return None

    def _param_spill_index(self, func: Any, ssaf: Any, v: Any) -> int | None:
        """If terminal no-def var *v* is *provably* a spill of an incoming
        parameter, return that parameter index, else None (#434).

        Sound-leaning heuristic: fires only when the slot is written by EXACTLY
        ONE definition in *func* -- so the no-def reload is unambiguously that
        store -- and that definition is a DIRECT copy of a recovered parameter:
        a bare ``slot = <param>`` whose src is a single ``MLIL_VAR_SSA`` /
        ``VAR_ALIASED`` read of one variable that is a real parameter. It
        deliberately does NOT fire when the slot is written more than once (the
        slot may be reused for a later, unrelated value) or filled with a derived
        expression (``slot = param - 4``): the earlier version matched the first
        param-reading store regardless of SSA version or the src operation, and so
        misattributed a reused/derived slot to the parameter. Degrades to None on
        any BN-API shortfall, so it never invents a parameter."""
        try:
            vk = var_key(v)
            stores = [ins for ins in self._instrs(ssaf)
                      if any(var_key(w) == vk
                             for w in (getattr(ins, "vars_written", None) or []))]
            if len(stores) != 1:
                return None  # 0 = no spill store; >1 = ambiguous slot (reuse/multi-def)
            src = getattr(stores[0], "src", None)
            if op_name(src) not in ("MLIL_VAR_SSA", "MLIL_VAR_ALIASED"):
                return None  # not a bare var copy -> a derived value, not a clean spill
            reads = expr_reads(src)
            if len(reads) != 1:
                return None
            return self._param_index_of(func, reads[0])  # None unless it is a real parameter
        except Exception:
            return None

    def _resolve_to_param_index(self, func: Any, ssaf: Any, expr: Any, depth: int = 0) -> int | None:
        """Trace a pointer-arg expression back to one of *func*'s parameters
        (so we can tell that writing through it taints a caller out-parameter)."""
        if expr is None or depth > 6:
            return None
        cands = expr_reads(expr) or ([expr] if is_ssa_var(expr) else [])
        for r in cands:
            idx = self._param_index_of(func, r)
            if idx is not None:
                return idx
            try:
                d = ssaf.get_ssa_var_definition(r)
            except Exception:
                d = None
            if d is not None:
                res = self._resolve_to_param_index(func, ssaf, getattr(d, "src", None), depth + 1)
                if res is not None:
                    return res
        return None

    def _name_matches_callee(self, name: str | None, callee: str) -> bool:
        """Match a (possibly decorated) callsite name against a --sink callee."""
        if not name:
            return False
        if name == callee or name.split("@", 1)[0].lstrip("_") == callee.lstrip("_"):
            return True
        matched, _ = lookup_model({callee: True}, name)
        return bool(matched)

    def _callee_name_forms(self, addr: int | None) -> list[str]:
        """Every name spelling of the function/symbol at *addr*: display name, raw
        (mangled) name, and the symbol's demangled short_name/full_name. Lets a
        demangled --source callee (`arg:foo::bar::recv:1`) match a callsite whose
        fn.name BN kept mangled -- the same forms seam._function_name_forms uses
        for xrefs/callsites, kept import-free here (#224a)."""
        forms: list[str] = []
        if addr is None or self.bv is None:
            return forms
        fn = function_at(self.bv, addr)
        if fn is not None:
            for v in (getattr(fn, "name", None), getattr(fn, "raw_name", None)):
                if v:
                    forms.append(str(v))
            sym = getattr(fn, "symbol", None)
            if sym is not None:
                for v in (getattr(sym, "short_name", None), getattr(sym, "full_name", None)):
                    if v:
                        forms.append(str(v))
        for cand in ((addr, addr & ~1) if (addr & 1) else (addr,)):
            try:
                sym = self.bv.get_symbol_at(cand)
            except Exception:
                sym = None
            if sym is not None:
                for v in (getattr(sym, "name", None), getattr(sym, "short_name", None),
                          getattr(sym, "full_name", None)):
                    if v:
                        forms.append(str(v))
        out: list[str] = []
        for f in forms:
            if f and f not in out:
                out.append(f)
        return out

    def _callee_matches(self, addr: int | None, callee: str) -> bool:
        """True if the callee at *addr* matches the user's callee under ANY of its
        name forms (incl. the demangled short/full name) -- so callsite seeding
        agrees with xrefs/callsites resolution for C++ names (#224a)."""
        for form in self._callee_name_forms(addr):
            if self._name_matches_callee(form, callee):
                return True
        return False

    @staticmethod
    def _callee_as_addr(callee: str) -> int | None:
        """An address-form callee locator (``0x12fa4`` or a bare decimal) as an
        int, else None. Lets ``ret:<addr>`` / ``arg:<addr>:<n>`` name a callee by
        the address the IL renders (PLT veneers, indirect calls with no recovered
        symbol) instead of a symbol that may not exist (#58)."""
        if not isinstance(callee, str):
            return None
        s = callee.strip()
        try:
            if s.lower().startswith("0x"):
                return int(s, 16)
            if s.isdigit():
                return int(s, 10)
        except Exception:
            return None
        return None

    def _find_callsites(self, instrs: list[Any], callee: str, *,
                        resolve_indirect: bool = False,
                        wrapper_arg: int | None = None) -> list[Any]:
        callee_addr = self._callee_as_addr(callee)

        def addr_match(a: Any) -> bool:
            # Compare THUMB-bit-masked so 0x12de0 matches a 0x12de1 entry.
            return (callee_addr is not None and a is not None
                    and (int(a) & ~1) == (callee_addr & ~1))

        hits = []
        for ins in instrs:
            if not self._is_call(ins):
                continue
            target = const_target(getattr(ins, "dest", None))
            # Match by any name form of the callee (display / mangled / demangled
            # short+full -- #224a), or by the address the call renders -- the same
            # address xrefs/trace use -- so an address-form locator seeds the same
            # callsites the name form does.
            if self._callee_matches(target, callee) or addr_match(target):
                hits.append(ins)
                continue
            # Follow a thunk/veneer (j_memcpy -> memcpy) and match the resolved
            # name OR its resolved address, so backward/forward seeding reaches a
            # sink called through a stub -- the same resolution forward taint and
            # `bn trace` already perform.
            rt = resolve_call_target(self.bv, ins, follow_thunks=True)
            if rt.address is not None:
                if (rt.via == "thunk" and self._callee_matches(int(rt.address), callee)) \
                        or addr_match(rt.address):
                    hits.append(ins)
                    continue
            # #282: an INDIRECT (vtable/fn-ptr) call whose target resolves to the
            # callee via an agent --resolve-map pin or value-set analysis. The
            # dominant real-server I/O shape routes recv/read through a slot
            # (`conn->type->read`), so the recv source must be anchorable there --
            # the same resolution the forward descent already performs for the call.
            if resolve_indirect and target is None \
                    and self._indirect_target_matches(ins, callee, wrapper_arg=wrapper_arg):
                hits.append(ins)
        return hits

    def _indirect_call_resolution(self, ins: Any) -> tuple[list[int], str | None]:
        """Resolve an indirect call *ins* to candidate target addresses and the
        provenance (`agent-map` / `value-set`), mirroring the forward-descent
        resolution. Returns ``([], None)`` for a direct call or no resolution."""
        if const_target(getattr(ins, "dest", None)) is not None:
            return [], None
        mapped = self.resolve_map.get(int(getattr(ins, "address", 0)))
        if mapped:
            return list(mapped), "agent-map"
        cands = self._call_targets_from_pvs(
            getattr(getattr(ins, "dest", None), "possible_values", None))
        return (cands, "value-set") if cands else ([], None)

    def _indirect_target_matches(self, ins: Any, callee: str, *,
                                 wrapper_arg: int | None = None) -> bool:
        """True if indirect call *ins* resolves (map/value-set) to a target whose
        name matches *callee*, directly or through a thunk veneer (#282), or --
        when *wrapper_arg* is given -- through a THIN WRAPPER that forwards its
        ``wrapper_arg`` parameter to *callee* (#292)."""
        cands, _ = self._indirect_call_resolution(ins)
        for taddr in cands:
            if self._callee_matches(taddr, callee):
                return True
            cfn = function_at(self.bv, taddr)
            if cfn is not None:
                resolved = self._follow_thunk_cached(cfn)
                if resolved is not None and resolved is not cfn \
                        and self._callee_matches(int(getattr(resolved, "start", 0)), callee):
                    return True
                if wrapper_arg is not None and self._is_thin_wrapper_forwarding(cfn, callee, wrapper_arg):
                    return True
        return False

    def _thin_wrapper_for(self, ins: Any, callee: str, wrapper_arg: int) -> Any | None:
        """If indirect call *ins* resolves to *callee* SOLELY through a thin wrapper
        that forwards its ``wrapper_arg``, return that wrapper function; else None.
        Used to name the wrapper in the anchor honesty assumption (#292).

        A direct name/thunk match anywhere in the candidate set is the cleaner
        disclosure (the #282 plain note), so this returns None whenever any
        candidate matches *callee* directly -- independent of candidate ORDER, so
        the disclosure wording never flips with --resolve-map/value-set order."""
        if wrapper_arg is None:
            return None
        cands, _ = self._indirect_call_resolution(ins)
        wrapper = None
        for taddr in cands:
            if self._callee_matches(taddr, callee):
                return None  # a direct name match exists -> plain #282 disclosure
            cfn = function_at(self.bv, taddr)
            if cfn is None:
                continue
            resolved = self._follow_thunk_cached(cfn)
            if resolved is not None and resolved is not cfn \
                    and self._callee_matches(int(getattr(resolved, "start", 0)), callee):
                return None  # a thunk match exists -> plain disclosure
            if wrapper is None and self._is_thin_wrapper_forwarding(cfn, callee, wrapper_arg):
                wrapper = cfn
        return wrapper

    def _is_thin_wrapper_forwarding(self, fn: Any, callee: str, arg: int) -> bool:
        """True if in-binary *fn* is a THIN WRAPPER that forwards its parameter
        *arg* to *callee* (#292): it calls *callee* exactly once (directly or
        through a thunk) and the argument at *callee*'s position *arg* is derived
        from *fn*'s own parameter *arg* (a positional forward -- an identity read
        or a param-derived expression like ``param+off``). This keeps the
        ``arg:<callee>:<arg>`` model semantics intact at the wrapper-dispatched
        indirect call, while a function that merely calls *callee* with a local
        buffer (its *arg* is not the wrapper's param *arg*) or calls it more than
        once does NOT match -- no over-anchoring."""
        if not self._is_internal(fn):
            return False
        ssaf = _mlil_ssa(fn)
        if ssaf is None:
            return False
        # Direct/thunk callsites of `callee` inside the wrapper body only
        # (resolve_indirect=False avoids recursing back through this resolver).
        sites = self._find_callsites(_ssa_instructions(ssaf), callee, resolve_indirect=False)
        if len(sites) != 1:
            return False
        cparams = self._call_params(sites[0])
        if arg < 0 or arg >= len(cparams):
            return False
        return self._resolve_to_param_index(fn, ssaf, cparams[arg]) == arg

    def _note_indirect_anchors(self, calls: list[Any], callee: str, add_assumption,
                               *, wrapper_arg: int | None = None) -> None:
        """Record an honesty assumption for each seeded callsite that is an
        indirect call resolved to *callee* by map/value-set -- the anchor is
        best-effort (value-set) or agent-pinned (--resolve-map), not a direct
        symbol match (#282). A value-set match among several candidates discloses
        the multiplicity so it doesn't read like a precise pin. When the target is
        a thin wrapper that forwards to *callee* (#292), the assumption names the
        wrapper so the indirection is explicit."""
        for c in calls:
            if const_target(getattr(c, "dest", None)) is not None:
                continue
            cands, via = self._indirect_call_resolution(c)
            if via is None:
                # matched by an address-form locator / import resolution, not by
                # indirect map/value-set resolution -- nothing to disclose here.
                continue
            detail = via
            if via == "value-set" and len(cands) > 1:
                detail = f"value-set ({self._prose_name(callee)} is 1 of {len(cands)} candidate targets)"
            wrapper = self._thin_wrapper_for(c, callee, wrapper_arg) if wrapper_arg is not None else None
            if wrapper is not None:
                add_assumption(
                    f"{self._prose_name(callee)} anchored at indirect callsite "
                    f"{hex(int(getattr(c, 'address', 0)))} via thin wrapper "
                    f"{self._prose_function(wrapper)} (forwards arg{wrapper_arg} to {self._prose_name(callee)}; "
                    f"resolved via {detail})")
            else:
                add_assumption(
                    f"{self._prose_name(callee)} anchored at indirect callsite "
                    f"{hex(int(getattr(c, 'address', 0)))} (resolved via {detail})")

    def _no_callsite_error(self, instrs: list[Any], callee: str, func: Any,
                           *, seed_kind: str = "source") -> "TaintError":
        """A `no callsite of <callee>` TaintError that, when the function dispatches
        through unresolved indirect calls, names them and points at --resolve-map
        instead of a bare not-found -- so an indirectly-routed recv/read is not a
        silent dead end (#282). Shared by the forward source seed and the backward
        sink seed; *seed_kind* (``"source"``/``"sink"``) keeps the locator guidance
        role-correct (a sink seed must not be told to use ``--source``)."""
        flag = f"--{seed_kind}"
        indirect = [ins for ins in instrs if self._is_call(ins)
                    and const_target(getattr(ins, "dest", None)) is None]
        if indirect:
            addrs = ", ".join(hex(int(getattr(i, "address", 0))) for i in indirect[:4])
            more = "" if len(indirect) <= 4 else f", +{len(indirect) - 4} more"
            return TaintError(
                f"no callsite of {callee} found in {func.name}; the function has "
                f"{len(indirect)} indirect call(s) (e.g. {addrs}{more}) that neither "
                f"value-set nor a --resolve-map pin resolved to {callee} -- if {callee} "
                f"is dispatched indirectly (vtable/fn-ptr), pin the call to its target "
                f"with --resolve-map <call_addr>=<target_addr> and seed "
                f"{flag} arg:<target>:<n>. The {seed_kind} callee must name the pinned "
                f"target: pin to {callee} itself to keep its model, or to the in-binary "
                f"wrapper that calls {callee} and seed arg:<wrapper>:<n>")
        return TaintError(f"no callsite of {callee} found in {func.name}")

    # -- forward ----------------------------------------------------------

    def forward(self, func: Any, sources: list[dict[str, Any]], *,
                enabled_sink_classes: set[str] | None = None, max_depth: int = 8) -> dict[str, Any]:
        # Scope the structural _buffer_target memo (#420) to this top-level forward
        # run: clear it here so a reused engine (e.g. over a re-lifted SSA) can never
        # return a stale structural result, and so per-(function, expr_index) keys
        # from a prior run can't survive. The engine is per-request in production --
        # this is defensive, but it makes the cache lifetime provably one forward run.
        self._bt_cache = {}
        self._bt_func_token = None

        # optional-sink classes the caller opted into (e.g. file_write); a sink
        # marked "optional" in the model DB fires only if its class is in here.
        # Set once for the whole request and shared by every per-callsite re-run.
        self._enabled_sink_classes: set[str] = set(enabled_sink_classes or ())

        # #5 per-source attribution: a single ret/arg source whose callee is
        # called from N>1 sites would otherwise seed from ALL of them into one
        # merged verdict, conflating N distinct buffers. When that is the case,
        # re-run the propagation once per callsite (seeding from exactly that
        # site) so each flow is reported on its own, and expose the breakdown
        # additively under `by_source`. Single-callsite (the common case) and
        # any multi-source / param / var request are unchanged (no `by_source`).
        callsite_addrs = self._attributable_callsites(func, sources)
        if not callsite_addrs:
            result = self._forward_run(func, sources, max_depth=max_depth)
            if len(sources) > 1 and result["reached_sinks"]:
                self._attribute_sources(func, sources, result, max_depth=max_depth)
            return result
        return self._forward_attributed(func, sources, callsite_addrs, max_depth=max_depth)

    def _attribute_sources(self, func: Any, sources: list[dict[str, Any]],
                           result: dict[str, Any], *, max_depth: int) -> None:
        """#1041 item 6: name which --source reaches each flow of a run with
        several sources -- by outcome, not from the taint graph.

        The graph records only the first parent that tainted a node, so the
        source its rendered path starts from is not proof of the flow's
        provenance: an argument a callee stores aside could be named for a sink
        it never reached (#1050 review finding 3). Instead each source is run
        on its own, and a flow lists as `source_candidates` every source whose
        solo run reaches the same sink (callee, address, argument). The
        signature names a source only when exactly one does; otherwise it keeps
        `multiple`. A finding that is not attributed to --source at all (an
        unconditional sink, `?`) is left as it is."""
        reached: dict[tuple, list[str]] = {}
        for src in sources:
            try:
                solo = self._forward_run(func, [src], max_depth=max_depth)
            except TaintError:
                continue   # seeds nothing on its own: it reaches nothing alone
            loc = _render_source(self._describe_locator(src))
            for f in solo["reached_sinks"]:
                s = f.get("sink") or {}
                key = (s.get("callee"), s.get("address"), s.get("tainted_arg_index"))
                if loc not in reached.setdefault(key, []):
                    reached[key].append(loc)
        for f in result["reached_sinks"]:
            sig = f.get("signature") or {}
            if sig.get("source") != "multiple":
                continue
            s = f.get("sink") or {}
            candidates = reached.get(
                (s.get("callee"), s.get("address"), s.get("tainted_arg_index")), [])
            f["source_candidates"] = candidates
            if len(candidates) == 1:
                f["signature"] = _make_signature(
                    candidates[0], sig.get("chain") or [], sig.get("sink_class"),
                    sig.get("sink_callee"), display=self._display_for_name,
                    symbol=self._raw_symbol_for)

    def _forward_run(self, func: Any, sources: list[dict[str, Any]], *, max_depth: int,
                     only_callsite_addr: int | None = None) -> dict[str, Any]:
        """One forward propagation over *func* and its single-result contract.

        ``only_callsite_addr`` (internal, no CLI surface) restricts ret/arg
        source seeding to exactly that call address so a single callsite's flow
        can be isolated for per-source attribution (#5)."""
        # Per-run analysis state (each per-callsite re-run is independent):
        self._cache: dict[tuple, Any] = {}          # (func_start, frozenset(params)) -> summary
        self._funcs_visited: set[int] = set()
        self._max_depth_seen = 0
        self._truncated = False
        # #579/#576 review: taint truncation has DISTINCT causes that consumers
        # must tell apart -- an intra-function fixpoint that exhausts its
        # iteration budget ('fixpoint_exhausted') vs an interprocedural descent
        # that hits the depth bound ('max_depth') vs the Python recursion guard
        # ('recursion'). They share `self._truncated` for the gate, but each
        # renders a different remediation, so the cause is tracked separately and
        # carried through to stats/diagnostics/text.
        self._truncation_causes: set[str] = set()
        # #811: callees whose body this run could NOT analyse (MLIL unavailable
        # -- a partially analysed view, or a function BN had not finished
        # lifting). `_summarize` already degrades conservatively, but it did so
        # with a prose assumption only, so a result built over an unread callee
        # was indistinguishable from one built over a fully analysed program and
        # could still answer safe_to_report_all_clear. Tracked structurally, the
        # same way `_truncated`/`_truncation_causes` are, so a consumer can gate.
        self._analysis_incomplete: set[str] = set()
        self._only_callsite_addr = only_callsite_addr
        # #559: count the source callsites the seed actually matched, so a
        # zero-result query can report "matched N source callsites" instead of an
        # unexplained empty result. Only the top run's ret/arg/call sources touch
        # this -- descended callees are seeded with synthetic param: locators.
        self._seed_callsites = 0
        # #1041 item 1: a call:/arg: source seeded a buffer its model fills, so
        # an all-clear carries the receive-family caveat (#1050 review round 5).
        self._buffer_seeded = False

        try:
            sub = self._run_forward(func, sources, depth=0, max_depth=max_depth, top=True)
        except RecursionError:
            # Defense in depth: should not happen now that thunk-following and the
            # SSA/call walks are cycle-guarded, but a pathological binary must
            # degrade to a bounded, honest result rather than crash the whole op.
            self._truncated = True
            self._truncation_causes.add("recursion")
            sub = {"findings": [], "leaves": [], "assumptions": [
                f"analysis of {func.name} truncated: Python recursion limit reached "
                "while propagating taint (possible unresolved cycle); results incomplete"]}
        # collapse any duplicate sink reports (same callee/site/arg) that distinct
        # resolved targets or arg-set growth may have produced
        seen_sink: set[tuple] = set()
        unique_findings = []
        for f in sub["findings"]:
            s = f.get("sink", {})
            sig = (s.get("callee"), s.get("address"), s.get("tainted_arg_index"))
            if sig in seen_sink:
                continue
            seen_sink.add(sig)
            unique_findings.append(f)
        sources_echo = [self._describe_locator(s) for s in sources]
        for f in unique_findings:
            # #615 review F1 + "New": an unconditional finding (tainted_arg_index
            # is None) never traced from the run's declared --source -- and
            # neither does a DOWNSTREAM finding whose flow roots at a node the
            # unconditional block injected (e.g. a strcpy() fed only by a gets()
            # buffer). Both render the pre-existing "?" no-source sentinel
            # (taint_locators.py) instead of falsely attributing --source to a
            # flow that never traced there.
            _from_unconditional_flow = f.pop("_unconditional_flow", False)
            _unconditional = f.get("sink", {}).get("tainted_arg_index") is None or _from_unconditional_flow
            fm, fs = derive_flow_facts(
                direction="forward", path=f.get("path"), sink=f.get("sink"),
                sources=[] if _unconditional else sources_echo,
                leaves=sub["leaves"], fn_name=str(func.name), display=self._display_for_name,
                symbol=self._raw_symbol_for)
            f["metrics"] = fm
            f["signature"] = fs
        result = {
            "direction": "forward",
            "function": {"name": str(func.name), "display_name": _display_name(func),
                         "address": hex(int(func.start))},
            "sources": sources_echo,
            "reached_sinks": unique_findings,
            "leaves": sub["leaves"],
            "assumptions": sub["assumptions"],
            "stats": {
                "functions_visited": len(self._funcs_visited),
                "max_depth": self._max_depth_seen,
                "sinks": len(unique_findings),
                # Authoritative unresolved-leaf count so the TEXT header, the JSON
                # `leaves` array length, and stats all cite the same number (#181).
                "leaves": len(sub["leaves"]),
                "truncated": self._truncated,
                # Additive, distinguishes the truncation cause (fixpoint vs depth
                # vs recursion) so a consumer reports the right one (#579/#576).
                "truncation_cause": sorted(self._truncation_causes),
                # #811: a body the run could not read is a coverage hole distinct
                # from truncation -- nothing was cut short, a region was never
                # legible. Named so the two are not conflated.
                "analysis_incomplete": bool(self._analysis_incomplete),
                "analysis_incomplete_functions": sorted(self._analysis_incomplete),
            },
            # #812: the CONFIGURED knobs this run used. `stats.max_depth` is the
            # deepest depth actually REACHED, so a result carrying only that
            # could not answer "was I bounded by the cap, or did the flow simply
            # end?" -- the two look identical when the walk happens to stop at
            # the cap. Echoed alongside the measurement, never merged into it.
            "run_params": {
                "max_depth": int(max_depth),
                "max_iters": int(self.max_iters),
                "unknown_call": str(self.unknown_call_policy),
            },
            "soundness": SOUNDNESS,
        }
        # #559: a modeled source that reaches no sink returns an otherwise-empty
        # result an agent can misread as a clean breadth check. Attach a compact,
        # factual frontier diagnostic (NOT a verdict) explaining where the seed
        # went and why the frontier stopped.
        if not unique_findings:
            result["diagnostics"] = self._forward_zero_diagnostics(sub, func)
        return result

    def _forward_zero_diagnostics(self, sub: dict[str, Any], func: Any) -> dict[str, Any]:
        """Frontier diagnostics for a zero-sink forward run (#559).

        Purely descriptive: seed reach (matched source callsites, tainted SSA
        values produced, last propagated use) plus the frontier the propagation
        stopped at (unresolved / coarse-memory leaf counts, whether any unmodeled
        external/in-binary call was reached) and a suggested next action. Never
        asserts a vulnerability -- it explains "flow hit an unmodeled parser we
        couldn't follow" vs "nothing flows".

        Delegates to :func:`taint_result.forward_zero_diagnostics` (the moved,
        engine-state-free implementation); the only engine input is the matched
        seed-callsite count."""
        return forward_zero_diagnostics(
            sub, seed_callsites=int(getattr(self, "_seed_callsites", 0)),
            truncated=bool(getattr(self, "_truncated", False)),
            truncation_cause=sorted(getattr(self, "_truncation_causes", set())),
            analysis_incomplete=bool(getattr(self, "_analysis_incomplete", ())),
            buffer_seed=bool(getattr(self, "_buffer_seeded", False)),
            function=hex(int(func.start)))

    def _attributable_callsites(self, func: Any, sources: list[dict[str, Any]]) -> list[int]:
        """Distinct call addresses to attribute a single ret/arg source across.

        Returns ``[]`` (no attribution) unless there is exactly ONE source, it is
        a ret/arg locator, and its callee is called from >1 distinct sites -- the
        only case where per-callsite attribution adds anything (#5)."""
        if len(sources) != 1:
            return []
        src = sources[0]
        if src.get("kind") not in ("ret", "arg"):
            return []
        callee = src.get("callee")
        if not callee:
            return []
        try:
            ssaf = self._ssa_func(func)
            instrs = self._instrs(ssaf)
        except TaintError:
            return []
        addrs: list[int] = []
        for c in self._find_callsites(instrs, callee, resolve_indirect=True):
            a = int(getattr(c, "address", 0))
            if a not in addrs:
                addrs.append(a)
        return addrs if len(addrs) > 1 else []

    def _forward_attributed(self, func: Any, sources: list[dict[str, Any]],
                            callsite_addrs: list[int], *, max_depth: int) -> dict[str, Any]:
        """Run the propagation once per source callsite and merge: top-level
        ``reached_sinks``/``leaves`` are the UNION across runs (back-compat),
        plus an additive ``by_source`` map keyed by call address (#5)."""
        callee = sources[0].get("callee")
        by_source: dict[str, Any] = {}
        base: dict[str, Any] | None = None
        union_findings: list[dict[str, Any]] = []
        union_leaves: list[dict[str, Any]] = []
        union_assumptions: list[str] = []
        funcs_visited: set[int] = set()
        max_depth_seen = 0
        truncated = False
        truncation_causes: set[str] = set()
        per_source_diag: dict[str, dict[str, Any]] = {}
        incomplete_funcs: set[str] = set()

        for addr in callsite_addrs:
            res = self._forward_run(func, sources, max_depth=max_depth, only_callsite_addr=addr)
            funcs_visited |= self._funcs_visited        # union of per-run visited sets
            if base is None:
                base = res
            by_source[hex(addr)] = {
                "reached_sinks": res["reached_sinks"],
                "leaves": res["leaves"],
                # #812: this callsite's BLOCKING frontier-leaf count, derived
                # from the canonical BLOCKING_LEAF_KINDS. The text renderer used
                # to compute its own "(N frontier)" marker by counting a single
                # hard-coded kind (`unmodeled_callee`) out of the ten that
                # block a claim, so every other frontier -- an under-recovered
                # arg, a coarse store, a pointer escape, an unresolved indirect
                # call, an unlifted instruction -- displayed as zero and the
                # marker vanished entirely. Counting it HERE keeps one owner for
                # the vocabulary: the CLI has no access to this set (no bridge
                # import) and must not grow a second copy of it (#827 item 3 is
                # the same defect elsewhere).
                "frontier": sum(1 for lf in res["leaves"]
                                if isinstance(lf, dict)
                                and lf.get("kind") in BLOCKING_LEAF_KINDS),
            }
            per_source_diag[hex(addr)] = res.get("diagnostics") or {}
            union_findings.extend(res["reached_sinks"])
            union_leaves.extend(res["leaves"])
            union_assumptions.extend(res["assumptions"])
            max_depth_seen = max(max_depth_seen, res["stats"]["max_depth"])
            truncated = truncated or res["stats"]["truncated"]
            truncation_causes |= set(res["stats"].get("truncation_cause") or [])
            incomplete_funcs |= set(
                res["stats"].get("analysis_incomplete_functions") or [])

        # Union the per-callsite results back into the historical top-level shape.
        # Stats follow the pinned rule: max_depth = max across runs; functions_visited
        # = size of the union (a function reached from two callsites counts once);
        # sinks = count of the deduped union; truncated = any run truncated.
        seen_sink: set[tuple] = set()
        findings: list[dict[str, Any]] = []
        for f in union_findings:
            s = f.get("sink", {})
            sig = (s.get("callee"), s.get("address"), s.get("tainted_arg_index"))
            if sig in seen_sink:
                continue
            seen_sink.add(sig)
            findings.append(f)
        leaves: list[dict[str, Any]] = []
        for lf in union_leaves:
            if lf not in leaves:
                leaves.append(lf)
        assumptions: list[str] = []
        for a in union_assumptions:
            if a not in assumptions:
                assumptions.append(a)
        assumptions.append(
            f"per-source attribution: {len(callsite_addrs)} callsites of {callee} analyzed "
            f"independently ({len(callsite_addrs)} propagations); top-level reached_sinks/leaves "
            f"are the union, by_source has the per-callsite split")

        out = {
            "direction": "forward",
            "function": base["function"],
            "sources": base["sources"],
            "reached_sinks": findings,
            "leaves": leaves,
            "assumptions": assumptions,
            "by_source": by_source,
            "stats": {
                "functions_visited": len(funcs_visited),
                "max_depth": max_depth_seen,
                "sinks": len(findings),
                # Authoritative deduped union leaf count (#181). frontier_total is
                # the pre-dedup sum across per-callsite runs, so an agent can
                # reconcile sum(by_source leaves) against the collapsed union.
                "leaves": len(leaves),
                "frontier_total": len(union_leaves),
                "truncated": truncated,
                "truncation_cause": sorted(truncation_causes),
                # #811: union of the callee bodies no per-callsite run could read.
                "analysis_incomplete": bool(incomplete_funcs),
                "analysis_incomplete_functions": sorted(incomplete_funcs),
            },
            # #812: same configured-knob echo as the single-run envelope.
            "run_params": {
                "max_depth": int(max_depth),
                "max_iters": int(self.max_iters),
                "unknown_call": str(self.unknown_call_policy),
            },
            "soundness": SOUNDNESS,
        }
        # #559/F2: attach a zero-sink frontier diagnostic when the whole
        # attributed union reached no sink -- but RECOMPUTE it over the UNIONED
        # leaves/assumptions/truncated, not the first callsite's own gate. Copying
        # base["diagnostics"] let a blocking frontier leaf emitted by callsite #2+
        # coexist with safe_to_report_all_clear=True from callsite #1 -- a false
        # all-clear on the default multi-callsite path that contradicts the same
        # result's own leaves array.
        #
        # #805: the DESCRIPTIVE half was still copied from the representative
        # (first) run, which made it an artifact of callsite ITERATION ORDER --
        # reversing `callsite_addrs` moved `last_use` while the gate stayed put,
        # so two runs over one binary disagreed about where taint was last seen.
        # Both fields are now defined over the whole union:
        #   tainted_values -- the SUM across per-callsite runs. They are
        #     independent propagations, so the sum is the total work the union
        #     represents; a value reached from two callsites counts once per run
        #     and the split stays in by_source.
        #   last_use -- kept ONLY when every run that has one agrees. N
        #     independent runs have no single last use, and electing a
        #     representative is exactly what made this order-dependent, so the
        #     scalar goes null and `last_use_by_source` carries the real answer
        #     per callsite. The ambiguity is disclosed, not resolved by accident.
        if not findings:
            _diags = list(per_source_diag.values())
            _tainted_values = sum(int(d.get("tainted_values") or 0) for d in _diags)
            _uses = [d.get("last_use") for d in _diags if d.get("last_use")]
            _agreed = _uses[0] if _uses and all(u == _uses[0] for u in _uses) else None
            union_sub = {
                "diag": {
                    "tainted_values": _tainted_values,
                    "last_use": _agreed,
                },
                "leaves": leaves,
                "assumptions": assumptions,
                "stats": {"truncated": truncated},
            }
            out["diagnostics"] = forward_zero_diagnostics(
                union_sub,
                seed_callsites=len(callsite_addrs),
                truncated=truncated, truncation_cause=sorted(truncation_causes),
                analysis_incomplete=bool(incomplete_funcs),
                buffer_seed=bool(getattr(self, "_buffer_seeded", False)),
                function=hex(int(func.start)))
            out["diagnostics"]["last_use_by_source"] = {
                a: d.get("last_use") for a, d in per_source_diag.items()
            }
        return out

    def _follow_thunk_cached(self, fn: Any) -> Any | None:
        """``follow_thunk`` memoized per resolved function start address."""
        if fn is None:
            return None
        key = int(getattr(fn, "start", 0))
        cache = self._thunk_cache
        if key in cache:
            return cache[key]
        resolved = follow_thunk(self.bv, fn)
        cache[key] = resolved
        return resolved

    # Allocator name -> index of its SIZE argument. calloc is omitted on purpose:
    # its size is n*elsize (a product), which the same-SSA-value proof below
    # can't establish, so calloc'd buffers simply never downgrade (safe).
    _ALLOC_SIZE_ARG = {
        "malloc": 0, "xmalloc": 0, "g_malloc": 0, "g_malloc0": 0,
        "valloc": 0, "pvalloc": 0, "alloca": 0, "__builtin_alloca": 0,
        "realloc": 1, "reallocf": 1, "g_realloc": 1,
        # C++ operator new / new[] -- canonical keys (see _canonical_cxx_alloc).
        "Znwm": 0, "Znam": 0,
    }

    # 2-arg (nmemb, size) allocators: the total allocation is nmemb*size, which the
    # single-expression size machinery can't represent. The dominant bounded idiom
    # pins one operand to the constant 1 (`calloc(1, n)` / `calloc(n, 1)`), where the
    # total equals the OTHER operand -- so we can hand that operand to the downgrade
    # comparison. Any non-trivial nmemb*size stays a safe over-report (#500). Only
    # names whose signature is exactly (nmemb, size) belong here, else the wrong
    # operand could be matched and a real overflow hidden.
    _CALLOC_SIZE_ARGS = {
        "calloc": (0, 1), "xcalloc": (0, 1), "ecalloc": (0, 1),
    }

    def _pointer_alloc_root(self, ssaf: Any, var: Any, depth: int = 0) -> Any:
        """Follow a destination pointer through pure SSA copies (``t = s``) and
        CONSTANT pointer arithmetic (``t = base +/- const``) to the root pointer
        var -- the one an allocator call defines. Lets ``d2 = dst + 0x13;
        memcpy(d2, ...)`` still recognize the allocation behind ``dst`` when the
        compiler materializes the offset pointer into its own SSA var (#307 FP-2);
        the constant offset itself is independently folded into the bound by
        ``_addr_base_offset``. Stops at a PHI, a non-constant index, or a multi-read
        source, so a tainted/unknown-offset dest still flags as a possible
        overflow (never over-downgrades)."""
        cur = var
        seen: set[tuple[Any, Any]] = set()
        for _ in range(8):
            marker = (var_key(cur), getattr(cur, "version", None))
            if marker in seen:
                break
            seen.add(marker)
            try:
                d = ssaf.get_ssa_var_definition(cur)
            except Exception:
                d = None
            if d is None or op_name(d) != "MLIL_SET_VAR_SSA":
                break
            src = getattr(d, "src", None)
            nxt = self._as_single_ssa_var(src)
            if nxt is None and op_name(src) in ("MLIL_ADD", "MLIL_SUB"):
                left = getattr(src, "left", None)
                right = getattr(src, "right", None)
                if self._int_const(right) is not None:
                    nxt = self._as_single_ssa_var(left)
                elif op_name(src) == "MLIL_ADD" and self._int_const(left) is not None:
                    nxt = self._as_single_ssa_var(right)
            if nxt is None:
                break
            cur = nxt
        return cur

    def _m_create_call_via(self, ssaf: Any, d: Any, depth: int = 0) -> Any | None:
        """The ``basic_string::_M_create`` CALL that produces this destination --
        directly, or through the range-ctor SSO ``PHI(heap_buf, local_buf)`` and
        copies -- else None. Only libstdc++'s ``_M_create`` (matched by name across
        every symbol spelling) qualifies, so a real overflow whose dest is an
        unrelated PHI is never mistaken for the STL pattern (#442)."""
        if d is None or depth > 6:
            return None
        op = op_name(d)
        if "CALL" in op:
            addr = self._resolve_direct_target(d)
            forms = self._callee_name_forms(addr)
            nm = self._callee_name(addr)
            if nm:
                forms = forms + [nm]
            # Require the libstdc++ basic_string::_M_create specifically -- the Itanium
            # mangled component `9_M_createE`, or a demangled form naming basic_string --
            # so an unrelated user symbol containing the "_M_create" substring can't be
            # mistaken for the STL allocator (review hardening).
            if any(f and "_M_create" in f
                   and ("9_M_createE" in f or "basic_string" in f)
                   for f in forms):
                return d
            return None
        if "VAR_PHI" in op:
            for sv in (getattr(d, "src", None) or []):
                if not is_ssa_var(sv):
                    continue
                root = self._pointer_alloc_root(ssaf, sv)
                try:
                    sd = ssaf.get_ssa_var_definition(root)
                except Exception:
                    sd = None
                hit = self._m_create_call_via(ssaf, sd, depth + 1)
                if hit is not None:
                    return hit
        return None

    def _m_create_capacity_size(self, ssaf: Any, d: Any) -> Any | None:
        """The requested-size expr for a destination produced by
        ``basic_string::_M_create(this, &capacity)`` -- the value stored into the
        by-reference ``&capacity`` slot just before the call. ``_M_create``
        allocates exactly that many bytes, so a ``memcpy`` of the same length is
        provably bounded and downgrades to ``bounded_len`` (#442). None when the
        dest is not an ``_M_create`` buffer or the capacity store can't be resolved
        (stays a safe over-report -- never a false downgrade). The post-call reload
        of the slot (the ROUNDED capacity) is deliberately not used."""
        call = self._m_create_call_via(ssaf, d)
        if call is None:
            return None
        params = self._call_params(call)
        if len(params) < 2:
            return None
        # arg0 = this, arg1 = &capacity (the in/out size reference).
        slot = self._addr_base_offset(ssaf, params[1])
        if slot is None:
            return None
        call_addr = getattr(call, "address", None)
        if call_addr is None:
            return None
        return self._value_stored_to_field(ssaf, self._instrs(ssaf), slot, call_addr)

    def _alloc_size_expr(self, ssaf: Any, ptr_expr: Any) -> Any | None:
        """If *ptr_expr*'s buffer was produced by an allocator call in this
        function, return the allocator's SIZE argument expression, else None."""
        var = ptr_expr if is_ssa_var(ptr_expr) else None
        if var is None:
            reads = expr_reads(ptr_expr)
            if len(reads) == 1:
                var = reads[0]
        if var is None:
            return None
        # Follow copies + constant pointer arithmetic to the allocator-defined
        # root so an offset pointer held in its own SSA var is still recognized
        # (#307 FP-2).
        var = self._pointer_alloc_root(ssaf, var)
        try:
            d = ssaf.get_ssa_var_definition(var)
        except Exception:
            d = None
        if d is None:
            return None
        # #442: std::string/std::vector range ctor -- basic_string::_M_create sizes
        # the destination via its by-REFERENCE capacity operand (`_M_create(this,
        # &cap)`), and the copy dest is a PHI of that heap buffer and the SSO local
        # buffer. Neither the by-ref size nor the PHI is handled by the by-value
        # allocator path below, so recognize it explicitly.
        m_create_size = self._m_create_capacity_size(ssaf, d)
        if m_create_size is not None:
            return m_create_size
        if "CALL" not in op_name(d):
            return None
        callee = self._callee_name(self._resolve_direct_target(d))
        base = (callee or "").split("@", 1)[0].lstrip("_")
        cargs = self._CALLOC_SIZE_ARGS.get(base)
        if cargs is not None:
            # calloc(nmemb, size): total = nmemb*size. Return the non-unit operand
            # when the other is the constant 1, so `calloc(1, n)` / `calloc(n, 1)`
            # downgrade like `malloc(n)`; otherwise the product isn't representable
            # here -> None (safe over-report, never a false downgrade) (#500).
            aparams = self._call_params(d)
            nmemb_i, size_i = cargs
            if nmemb_i < len(aparams) and size_i < len(aparams):
                nmemb_e, size_e = aparams[nmemb_i], aparams[size_i]
                if self._int_const(nmemb_e) == 1:
                    return size_e
                if self._int_const(size_e) == 1:
                    return nmemb_e
            return None
        idx = self._ALLOC_SIZE_ARG.get(base)
        if idx is None:
            # C++ operator new/new[] (mangled variants / demangled) -> canonical key.
            canon = _canonical_cxx_alloc(base)
            idx = self._ALLOC_SIZE_ARG.get(canon) if canon else None
        if idx is None:
            return None
        aparams = self._call_params(d)
        return aparams[idx] if idx < len(aparams) else None

    def _arg_ptr_is_indirect_load(self, ssaf: Any, expr: Any, depth: int = 0) -> bool:
        """True when a buffer-pointer arg is itself loaded from memory -- a global
        or struct-field pointer slot (`recvfrom(fd, G.pkt, n)` where ``G.pkt`` is
        ``*(G+off)``) rather than a direct ``&stackbuf``. In that case the engine
        anchors the seed to the loaded pointer value but does not correlate it with
        a later re-load of the same slot, so the recv->parse flow can be missed --
        worth an honest note when the buffer couldn't be anchored (#193)."""
        if expr is None or depth > 6:
            return False
        if "LOAD" in op_name(expr):
            return True
        v = self._as_single_ssa_var(expr)
        if v is None:
            reads = expr_reads(expr)
            v = reads[0] if len(reads) == 1 else None
        if v is None:
            return False
        try:
            d = ssaf.get_ssa_var_definition(v)
        except Exception:
            d = None
        if d is None:
            return False
        src = getattr(d, "src", None)
        if src is None:
            return False
        if "LOAD" in op_name(src):
            return True
        return self._arg_ptr_is_indirect_load(ssaf, src, depth + 1)

    def _buffer_slot_key(self, ssaf: Any, expr: Any, depth: int = 0):
        """The memory slot a buffer pointer was *loaded from*: for ``bufp = [base
        + off]`` return ``slotkey(base + off)`` = ``(base_key, off)``. None when the
        pointer is not an indirect load or the slot address can't be resolved to a
        stable key. Pairs with `_arg_ptr_is_indirect_load` -- that says "indirect",
        this names the slot so a recv store and a later re-load compare equal
        (#193 Part 1)."""
        if expr is None or depth > 6:
            return None
        if "LOAD" in op_name(expr):
            return self._addr_base_offset(ssaf, getattr(expr, "src", None))
        v = self._as_single_ssa_var(expr)
        if v is None:
            reads = expr_reads(expr)
            v = reads[0] if len(reads) == 1 else None
        if v is None:
            return None
        try:
            d = ssaf.get_ssa_var_definition(v)
        except Exception:
            d = None
        if d is None:
            return None
        src = getattr(d, "src", None)
        if src is None:
            return None
        if "LOAD" in op_name(src):
            return self._addr_base_offset(ssaf, getattr(src, "src", None))
        return self._buffer_slot_key(ssaf, src, depth + 1)

    def _store_to_slot_between(self, ssaf: Any, instrs: Any, key: tuple, lo_idx: int, hi_idx: int) -> bool:
        """True if a store between *lo_idx* (the recv) and *hi_idx* (the re-load)
        invalidates slot *key*'s identity -- either a store to the slot itself
        (re-pointing the buffer), OR, for a ``("gload", G)`` base, a store to the
        base global G (re-pointing the context pointer, so the re-load reads a
        DIFFERENT object's slot). Missing the latter taints the wrong buffer with
        the caveat suppressed -- the worst VR failure mode (adversarial-review #2).
        Conservative and CFG-insensitive: considers every store on any path, so it
        never *under*-blocks; an unprovable case stays uncorrelated (honest)."""
        base = key[0] if key else None
        base_global = base[1] if isinstance(base, tuple) and base and base[0] == "gload" else None
        for ins in instrs:
            idx = getattr(ins, "instr_index", None)
            if idx is None or not (lo_idx < idx < hi_idx):
                continue
            if "STORE" not in op_name(ins):
                continue
            dest = getattr(ins, "dest", None)
            sk = self._addr_base_offset(ssaf, dest)
            if sk is not None and sk == key:
                return True
            if base_global is not None and self._store_addr_const(ssaf, dest) == base_global:
                return True
        return False

    def _store_addr_const(self, ssaf: Any, expr: Any, depth: int = 0) -> int | None:
        """The constant address a store writes *to* (``[0xG] = v`` -> ``0xG``),
        following SSA copies to a CONST_PTR. Used to detect a re-point of a
        ``("gload", G)`` slot's base global. None if the destination isn't a
        constant address."""
        if expr is None or depth > 6:
            return None
        if "CONST_PTR" in op_name(expr):
            c = getattr(expr, "constant", None)
            if c is None:
                v = getattr(expr, "value", None)
                c = getattr(v, "value", v)
            try:
                return int(c)
            except (TypeError, ValueError):
                return None
        if is_ssa_var(expr):
            try:
                d = ssaf.get_ssa_var_definition(expr)
            except Exception:
                d = None
            return self._store_addr_const(ssaf, getattr(d, "src", None), depth + 1) if d is not None else None
        reads = expr_reads(expr)
        if len(reads) == 1:
            try:
                d = ssaf.get_ssa_var_definition(reads[0])
            except Exception:
                d = None
            if d is not None:
                return self._store_addr_const(ssaf, getattr(d, "src", None), depth + 1)
        return None

    def _register_indirect_buffer_slot(self, ssaf, ptr_expr, callsite, callee, idx,
                                       buffer_slots, add_assumption,
                                       add_leaf=None, recv_buffer_covered=False) -> bool:
        """When a buffer source's pointer is an indirect load `[slot]`, register the
        slot so a later re-load of it can be correlated forward (#193 Part 1). If the
        slot can't be named, fall back to today's honest "may be missed" caveat. When
        it can, defer that caveat -- `_forward_run` decides post-fixpoint whether the
        slot actually correlated (positive note) or not (the caveat).

        Honesty gate (#562): the "may be missed" caveat is assumption-only, so on a
        zero-sink run it did NOT block the claim gate even though analysis provably
        did not follow the pointed-to payload. Emit a real blocking leaf
        (``source_seed_misanchored`` via :func:`indirect_pointer_slot_leaf`) on the
        SAME (un)correlated sub-path so the gate withholds via the existing leaf
        mechanism -- exactly like the recv-buffer case. ``recv_buffer_covered`` is
        set by the ``arg:`` caller when it already emits a recv-buffer misanchored
        leaf for this (callee, idx), so the leaf is not double-counted there; the
        ``*arg:`` (call: preset) caller has no such leaf and passes False.

        Returns True when the pointer IS an indirect load, i.e. this helper owns
        the disclosure for it (an immediate or a deferred leaf); False when it did
        nothing, so the caller must disclose the unkeyed buffer itself."""
        if not self._arg_ptr_is_indirect_load(ssaf, ptr_expr):
            return False
        pending = (
            f"source {callee} arg{idx} buffer pointer is loaded indirectly (from a "
            f"global/struct slot); the seed anchors to the pointer value, not the "
            f"pointee, and is not correlated with later re-loads of the same slot -- a "
            f"flow that re-loads the pointer and parses it may be missed. Consider "
            f"seeding the parser entry directly with param:N")
        key = self._buffer_slot_key(ssaf, ptr_expr)
        # The blocking leaf that mirrors the caveat (None when the recv-buffer
        # caller already covers this (callee, idx), or when no leaf sink was
        # provided -- e.g. a legacy caller). It fires on exactly the sub-paths
        # where the assumption fires, never when the slot correlates forward.
        pending_leaf = None
        if add_leaf is not None and not recv_buffer_covered:
            pending_leaf = indirect_pointer_slot_leaf(
                callee=str(callee), arg_index=idx,
                address=hex(int(getattr(callsite, "address", 0))),
                slot=key)
        if key is None:
            add_assumption(pending)
            if pending_leaf is not None:
                add_leaf(pending_leaf)
            return True
        recv_node = None
        for r in expr_reads(ptr_expr):
            recv_node = (var_key(r), getattr(r, "version", None))
            break
        buffer_slots[key] = {
            "recv_idx": getattr(callsite, "instr_index", None),
            "callee": callee, "idx": idx, "recv_node": recv_node, "pending": pending,
            "pending_leaf": pending_leaf,
        }
        return True

    def _format_destination_end(self, ssaf: Any, model: Any, params: list[Any]) -> int | None:
        """For a scanf-family `call:` preset: one past the last argument the
        call's CONSTANT format actually writes (``first_dest`` plus its
        assigning conversions), else None (no format sink, an unreadable
        format, or `%n$` positions -- seed the modeled run as before).

        BN recovers a variadic call's arity from register writes, so a
        `scanf("%s", buf)` can show a stale third argument (the stack-guard
        address); seeding it named a "buffer" scanf never writes and, under
        #1041 item 1, reported it unkeyed (#1050 re-dogfood). The format says
        which arguments are destinations; nothing is guessed."""
        sink = (model or {}).get("sink") or {}
        fa, first = sink.get("format_arg"), sink.get("first_dest")
        if fa is None or first is None or int(fa) >= len(params):
            return None
        fmt = self._const_format_string(ssaf, params[int(fa)])
        if fmt is None:
            return None
        specs = _format_specs(fmt, is_scanf=True)
        if any(s.positional for s in specs):
            return None
        return int(first) + sum(1 for s in specs if not s.suppressed)

    def _unchecked_indirect_calls(self, instrs: list[Any], fills: list[Any], callee: str,
                                  add_leaf: Any) -> None:
        """#1041 item 1 gate (#1050 review rounds 4-5): a blocking leaf at every
        unresolved indirect call in the SOURCE function that can run after a
        call that fills a buffer.

        Which pointer such a callback receives is NOT worked out -- aliasing is
        left out. The rule is an outcome instead: a filled buffer exists, an
        unresolved indirect call in this function can run after it, so whatever
        that call does with the bytes is unchecked, and the run is not an
        all-clear -- whether or not the buffer was keyed and whether or not a
        slot reload correlated. Unknown reachability counts as reachable. A call
        is resolved when it is direct, mapped by `--resolve-map`, or pinned by
        its value set -- the same resolution the walk descends with.

        Its scope is this function only: indirect calls inside callees the walk
        enters, mapped or value-set targets, and transfers BN lifts as an MLIL
        `jump` are not gated (the receive-family all-clear caveat says so). A
        system call is not a function-pointer call and is never gated (#1050
        review round 5). The cost: an unrelated unresolved indirect call after
        a receive also withholds the all-clear."""
        for ins in instrs:
            if not self._is_call(ins) or "SYSCALL" in op_name(ins) \
                    or any(ins is f for f in fills):
                continue
            if self._resolve_direct_target(ins) is not None:
                continue
            cands, _via = self._indirect_call_resolution(ins)
            if cands:
                continue
            if not any(self._reaches_avoiding(f, ins, []) is not False for f in fills):
                continue
            at = hex(int(getattr(ins, "address", 0)))
            add_leaf({
                "kind": "indirect_call_unchecked",
                "address": at,
                "callee": str(callee),
                "fills": sorted(hex(int(getattr(f, "address", 0))) for f in fills),
                "il_text": str(ins),
                "detail": (
                    f"the indirect call at {at} can run after {self._prose_name(callee)} fills a "
                    f"buffer, and bn did not resolve its callee: indirect calls that may "
                    f"receive this buffer are not followed -- their callees are unchecked. "
                    f"Pinning the target with --resolve-map makes the walk descend into it "
                    f"with the arguments taint reaches; it does not show that the target "
                    f"receives this buffer"),
            })

    def _as_single_ssa_var(self, expr: Any) -> Any | None:
        """The lone SSA variable an expression reads, but ONLY when the
        expression is a bare var read (MLIL_VAR_SSA / an SSAVariable) -- not
        inline arithmetic. Returns None otherwise."""
        if is_ssa_var(expr):
            return expr
        if op_name(expr) == "MLIL_VAR_SSA":
            reads = expr_reads(expr)
            if len(reads) == 1:
                return reads[0]
        return None

    def _canonical_ssa_var(self, ssaf: Any, var: Any, depth: int = 0) -> Any:
        """Follow PURE copy definitions (t = s) to the root SSA variable.

        Compiled ARM routinely splits one length value across register copies
        (r0_5 = r5_2 for malloc, r2_1 = r5_2 for memcpy), so two copies of the
        same value have different var+version. Following only SET_VAR_SSA
        definitions whose source is a single bare var read preserves the value
        exactly, so comparing canonical roots is still a PROOF of equality --
        arithmetic, PHI, and call defs all stop the walk (#46 item 1)."""
        cur = var
        seen: set[tuple[Any, Any]] = set()
        for _ in range(8):
            marker = (var_key(cur), getattr(cur, "version", None))
            if marker in seen:
                break
            seen.add(marker)
            try:
                d = ssaf.get_ssa_var_definition(cur)
            except Exception:
                d = None
            if d is None or op_name(d) != "MLIL_SET_VAR_SSA":
                break
            src = self._as_single_ssa_var(getattr(d, "src", None))
            if src is None:
                break
            cur = src
        return cur

    def _same_ssa_value(self, ssaf: Any, a: Any, b: Any) -> bool:
        """True only when *a* and *b* provably hold the same runtime value:
        each must be a bare SSA var read, and their canonical (copy-chain) roots
        must be the same var + version. Any other shape (inline arithmetic,
        multiple reads, constants, differing roots) -> False, so a length is
        only ever declared bounded when provably equal to the allocation size,
        never by a heuristic (#46 item 1)."""
        va = self._as_single_ssa_var(a)
        vb = self._as_single_ssa_var(b)
        if va is None or vb is None:
            return False
        ra = self._canonical_ssa_var(ssaf, va)
        rb = self._canonical_ssa_var(ssaf, vb)
        return (
            var_key(ra) == var_key(rb)
            and getattr(ra, "version", None) == getattr(rb, "version", None)
        )

    def _dest_alloc(self, ssaf: Any, ptr_expr: Any) -> tuple[Any | None, bool]:
        """``(size_expr, assumed_wrapper)`` for the allocator that produced the
        buffer *ptr_expr* points into. Recognizes the modeled allocators (and
        C++ ``new``) via :meth:`_alloc_size_expr`; failing that, conservatively
        treats an UNMODELED single-argument call feeding the buffer as an
        allocator wrapper whose sole arg is the size (``assumed_wrapper=True``) so
        ``dst = wrap(len); memcpy(dst, src, len)`` can downgrade too (#229)."""
        size = self._alloc_size_expr(ssaf, ptr_expr)
        if size is not None:
            return size, False
        var = ptr_expr if is_ssa_var(ptr_expr) else None
        if var is None:
            reads = expr_reads(ptr_expr)
            if len(reads) == 1:
                var = reads[0]
        if var is None:
            return None, False
        # Same copy + constant-pointer-arithmetic walk as the modeled path, so a
        # wrapper-allocated buffer reached through an offset pointer var is still
        # recognized (#307 FP-2).
        var = self._pointer_alloc_root(ssaf, var)
        try:
            d = ssaf.get_ssa_var_definition(var)
        except Exception:
            d = None
        if d is None or "CALL" not in op_name(d):
            return None, False
        aparams = self._call_params(d)
        # Only a single-arg call qualifies as an assumed size-by-len wrapper, AND
        # it must be CORROBORATED as an allocator -- by an allocator-ish name or a
        # pointer return type. Without corroboration a non-allocator one-arg call
        # (e.g. `get_scratch(len)` returning a fixed buffer) would be mistaken for
        # an allocator and hide a real overflow; staying overflow_len (no
        # downgrade) is the safe direction (review Finding 2). The bound check
        # still additionally requires the sole arg to be linear in the copy length.
        if len(aparams) == 1 and self._looks_like_allocator(d):
            return aparams[0], True
        return None, False

    _ALLOC_NAME_HINTS = ("alloc", "dup", "salloc")

    def _looks_like_allocator(self, call_ins: Any) -> bool:
        """Corroborate that a call is plausibly an allocator wrapper: its name
        contains an allocator hint (malloc/calloc/realloc/xalloc/.../strdup), the
        resolved callee returns a pointer, or its BODY provably forwards the size
        arg to a known allocator (#307). Conservative -- an opaque-named,
        unknown-return wrapper with no allocator call is NOT downgraded (#229
        review Finding 2)."""
        addr = self._resolve_direct_target(call_ins)
        name = (self._callee_name(addr) or "").split("@", 1)[0].lstrip("_").lower()
        if any(tok in name for tok in self._ALLOC_NAME_HINTS):
            return True
        fn = function_at(self.bv, addr) if addr is not None else None
        rt = getattr(fn, "return_type", None)
        if rt is not None:
            tcn = self._type_class_name(rt)
            if "Pointer" in tcn or str(rt).rstrip().endswith("*"):
                return True
        # Body-level corroboration: a STRIPPED allocator wrapper (sub_XXXX, no
        # recovered pointer return type) that provably forwards its sole arg to a
        # known allocator's size position -- common in firmware, the case the
        # name/return-type heuristics miss (#307). A non-allocator one-arg helper
        # (e.g. one that returns a fixed scratch buffer, with no allocator call)
        # does NOT match, so a real overflow into a fixed buffer still flags.
        return self._is_allocator_wrapper_body(fn)

    def _is_allocator_wrapper_body(self, fn: Any) -> bool:
        """True if in-binary single-call thin wrapper *fn* forwards its first
        parameter to a KNOWN allocator's SIZE argument (#307). Strict: exactly one
        call in the body, to a known allocator, with the wrapper's param 0 at the
        allocator's size position -- so `acquire(n){ return malloc(n); }` is
        recognized even when stripped of name and return type, while a helper with
        no allocator call (or that passes a local, not its param) is not."""
        if fn is None or not self._is_internal(fn):
            return False
        ssaf = _mlil_ssa(fn)
        if ssaf is None:
            return False
        calls = [i for i in _ssa_instructions(ssaf) if "CALL" in op_name(i)]
        if len(calls) != 1:
            return False
        call = calls[0]
        base = (self._callee_name(self._resolve_direct_target(call)) or "").split("@", 1)[0].lstrip("_")
        size_idx = self._ALLOC_SIZE_ARG.get(base)
        if size_idx is None:
            canon = _canonical_cxx_alloc(base)
            size_idx = self._ALLOC_SIZE_ARG.get(canon) if canon else None
        if size_idx is None:
            return False
        cparams = self._call_params(call)
        if size_idx >= len(cparams):
            return False
        return self._resolve_to_param_index(fn, ssaf, cparams[size_idx]) == 0

    def _linear_in_var(self, ssaf: Any, expr: Any, depth: int = 0):
        """Decompose *expr* into ``(canonical_ssa_var, const)`` when it is ``v``,
        ``v + const``, ``const + v`` or ``v - const`` (following pure SSA copies
        to v's canonical root); ``None`` otherwise. The const is 0 for a bare var.
        Lets the bound check compare an allocation ``len + C`` against a copy
        ``len + D`` by their shared length var and constant offsets (#229)."""
        if expr is None or depth > 6:
            return None
        v = self._as_single_ssa_var(expr)
        if v is not None:
            try:
                d = ssaf.get_ssa_var_definition(v)
            except Exception:
                d = None
            if d is not None and op_name(d) == "MLIL_SET_VAR_SSA":
                sub = self._linear_in_var(ssaf, getattr(d, "src", None), depth + 1)
                if sub is not None:
                    return sub
            return (self._canonical_ssa_var(ssaf, v), 0)
        op = op_name(expr)
        if op in ("MLIL_ADD", "MLIL_SUB"):
            left = getattr(expr, "left", None)
            right = getattr(expr, "right", None)
            rc = self._int_const(right)
            if rc is not None:
                sub = self._linear_in_var(ssaf, left, depth + 1)
                if sub is not None:
                    return (sub[0], sub[1] + (rc if op == "MLIL_ADD" else -rc))
            if op == "MLIL_ADD":
                lc = self._int_const(left)
                if lc is not None:
                    sub = self._linear_in_var(ssaf, right, depth + 1)
                    if sub is not None:
                        return (sub[0], sub[1] + lc)
        return None

    def _bounded_copy_reason(self, ssaf: Any, params: list[Any], length_idx: int,
                             dest_idx: int = 0) -> str | None:
        """A human reason when the copy (dest=params[dest_idx],
        length=params[length_idx]) provably fits the buffer the destination was
        allocated with in this same function, else None.

        Generalizes the exact ``dst = malloc(n); memcpy(dst, src, n)`` case
        (#46 item 1) to ``dst = alloc(len + C); memcpy(dst + c, src, len + D)``,
        which is bounded iff ``c + D <= C`` for the same length var, and to
        unmodeled single-arg allocator wrappers (#229)."""
        if dest_idx >= len(params) or length_idx >= len(params):
            return None
        dest_expr = params[dest_idx]
        len_expr = params[length_idx]
        size_expr, assumed_wrapper = self._dest_alloc(ssaf, dest_expr)
        if size_expr is None:
            return None
        off_info = self._addr_base_offset(ssaf, dest_expr)
        c = off_info[1] if (off_info is not None and isinstance(off_info[1], int)) else 0
        if c < 0:
            return None
        wrap = " (assumed allocator wrapper sized by the copy length)" if assumed_wrapper else ""
        # Exact same-value, no dest offset: the original #46 case.
        if c == 0 and self._same_ssa_value(ssaf, size_expr, len_expr):
            return ("the destination is allocated with the same size in this "
                    "function -- provably bounded, not an overflow" + wrap)
        alloc = self._linear_in_var(ssaf, size_expr)
        copy = self._linear_in_var(ssaf, len_expr)
        if alloc is None or copy is None:
            return None
        (av, ac), (cv, cc) = alloc, copy
        if not (var_key(av) == var_key(cv)
                and getattr(av, "version", None) == getattr(cv, "version", None)):
            return None
        # SOUNDNESS: a NEGATIVE copy-length addend means `len - C`, which in
        # unsigned C underflows to a huge value when len < C (a real overflow);
        # it is also how BN surfaces a >= 2^63 addend (constants are sign-extended
        # at bit 63). Likewise a negative alloc addend (`alloc(len - C)`) can
        # under-allocate. Either way the copy is NOT provably bounded, so never
        # downgrade -- staying overflow_len is the safe (over-report) direction.
        if cc < 0 or ac < 0:
            return None
        if c + cc <= ac:
            return (f"the destination is allocated with len+{hex(ac)} and the copy "
                    f"writes len+{hex(cc)} bytes at offset {hex(c)} "
                    f"({hex(c)}+{hex(cc)} <= {hex(ac)}) -- provably bounded" + wrap)
        return None

    def _provably_bounded_length(self, ssaf: Any, params: list[Any], length_idx: int,
                                 dest_idx: int = 0) -> bool:
        """Backward-compatible bool wrapper over :meth:`_bounded_copy_reason`."""
        return self._bounded_copy_reason(ssaf, params, length_idx, dest_idx) is not None

    # #980 review: the copies that read EXACTLY the length argument's byte count
    # from their source. A NUL-stopping reader (strncpy, strncat, strlcpy,
    # memccpy) reads at most `n` and usually far less, and snprintf's "source"
    # is its format string -- sizing those against `n` turned the canonical safe
    # `snprintf(malloc(n), n, "id=%d", id)` into a false over-read. The wide
    # copies count ELEMENTS, not bytes, so a byte capacity cannot judge them.
    _EXACT_LENGTH_COPIES = frozenset({"memcpy", "memmove", "mempcpy", "bcopy",
                                      "__builtin_memcpy", "memcpy_chk", "memmove_chk",
                                      "mempcpy_chk"})

    def _source_side_verdict(self, ssaf: Any, params: list[Any], length_idx: int,
                             mkey: str | None, model: dict[str, Any],
                             eff_sink: dict[str, Any]) -> dict[str, Any]:
        """#980: the copy-SOURCE half of a destination-bounded copy.

        ``dst = malloc(n); memcpy(dst, src, n)`` cannot overflow ``dst``, but with
        a fixed-size ``src`` and an attacker-controlled ``n`` it over-reads
        ``src``. Returns *eff_sink* re-headlined to ``overread_len`` when the
        source is a fixed-size object (a typed global, a stack local, a
        constant-size allocation) and the length is not provably within it;
        otherwise still ``bounded_len``, stating whether the source fit or could
        not be sized at all -- a bounded write is never left reading as "safe"
        while the read side was never looked at. Only for the exact-length
        copies in ``_EXACT_LENGTH_COPIES``; any other model is returned as is."""
        if mkey not in self._EXACT_LENGTH_COPIES:
            return eff_sink
        roles = _model_copy_roles(model)
        if roles is None or roles[0] >= len(params):
            return eff_sink
        srcs = [roles[0]]
        src_expr, len_expr = params[srcs[0]], params[length_idx]
        cap = self._buffer_capacity(ssaf, src_expr)
        unchecked = {**eff_sink, "src_capacity": None,
                     "relabel": eff_sink["relabel"] + "; source capacity not checked",
                     "detail": eff_sink["detail"] + f"; the copy source (arg{srcs[0]}) could not "
                     "be sized against the length, so an over-read of it is NOT ruled out -- "
                     "check that it holds at least the copied length"}
        if cap is None:
            return unchecked
        size, where, alloc_size = cap
        bound = self._length_upper_bound(ssaf, len_expr)
        fits = ((alloc_size is not None and self._same_ssa_value(ssaf, alloc_size, len_expr))
                or (size is not None and bound is not None and bound <= size))
        if fits:
            return {**eff_sink, "src_capacity": size, "src_capacity_source": where,
                    "relabel": eff_sink["relabel"] + "; source holds the copied length",
                    "detail": eff_sink["detail"] + f"; the copy source ({where}) provably "
                    "holds the copied length"}
        # An over-read needs a FINITE source the length can exceed: a known size,
        # or a stack/global object of unrecovered size (still fixed) against a
        # length with no upper bound at all. An allocation of some other unknown
        # size could be large enough, so it stays unchecked.
        if size is None and (alloc_size is not None or bound is not None):
            return {**unchecked, "src_capacity_source": where}
        return {**eff_sink, "class": "overread_len", "src_capacity": size,
                "src_capacity_source": where,
                "relabel": eff_sink["relabel"] + f"; source is {where}, not provably >= length",
                "detail": eff_sink["detail"] + f"; BUT the copy source (arg{srcs[0]}) is {where} "
                "and the attacker-derived length is not provably within it -- an over-read of "
                "the source"}

    def _buffer_capacity(self, ssaf: Any, expr: Any,
                         depth: int = 0) -> tuple[int | None, str, Any] | None:
        """``(bytes available from expr, description, alloc_size_expr)`` for a
        pointer into a fixed-size object, else None. Follows SSA copies and
        constant offsets to a typed global (``&table``), a stack local
        (``&buf``), or an allocation in this function. The size is BN's type
        width for an object (a recovered type, so an estimate with its
        provenance in the description); for an allocation it is the constant
        size, or None with the size expression when that is not a constant."""
        if expr is None or depth > 6:
            return None
        op = op_name(expr)
        if op in ("MLIL_ADD",):
            off = self._int_const(getattr(expr, "right", None))
            base = getattr(expr, "left", None)
            if off is None:
                off = self._int_const(base)
                base = getattr(expr, "right", None)
            if off is None or off < 0:
                return None
            inner = self._buffer_capacity(ssaf, base, depth + 1)
            if inner is None or inner[0] is None:
                return None
            if off >= inner[0]:
                return None
            return inner[0] - off, f"{inner[1]} at +{hex(off)}", None
        if "CONST" in op:
            addr = self._int_const(expr)
            dv = None
            try:
                dv = self.bv.get_data_var_at(int(addr)) if addr is not None else None
            except Exception:
                dv = None
            width = getattr(getattr(dv, "type", None), "width", None)
            if not isinstance(width, int) or width <= 0:
                return None
            return width, f"a {width}-byte global typed `{dv.type}` @ {hex(addr)}", None
        if op == "MLIL_ADDRESS_OF":
            var = getattr(expr, "src", None)
            width = self._var_byte_size(var)
            name = getattr(var, "name", var)
            if not width:
                # BN often types a stripped binary's stack buffer `void`: its
                # size is unrecovered, but a stack local is still a fixed object.
                return None, f"the stack local `{name}` (size not recovered)", None
            return width, f"a {width}-byte local `{name}` typed `{getattr(var, 'type', '?')}`", None
        v = self._as_single_ssa_var(expr)
        if v is None:
            return None
        size_expr, _wrapper = self._dest_alloc(ssaf, expr)
        if size_expr is not None:
            const = self._int_const(size_expr)
            return const, (f"a {const}-byte allocation" if const is not None
                           else f"an allocation of `{size_expr}` bytes"), size_expr
        try:
            d = ssaf.get_ssa_var_definition(v)
        except Exception:
            d = None
        if d is not None and op_name(d) == "MLIL_SET_VAR_SSA":
            return self._buffer_capacity(ssaf, getattr(d, "src", None), depth + 1)
        return None

    def _length_upper_bound(self, ssaf: Any, expr: Any) -> int | None:
        """The largest value *expr* can take, when BN's value set (or a modeled
        receive bound, #159) states one; None when it is unbounded/unknown."""
        const = self._int_const(expr)
        if const is not None:
            return const
        bound = self._syscall_bound_for_length(ssaf, expr)
        if bound is not None:
            return bound
        pvs = getattr(expr, "possible_values", None)
        kind = str(getattr(getattr(pvs, "type", None), "name", ""))
        try:
            if kind in ("ConstantValue", "ConstantPointerValue"):
                return int(pvs.value)
            if kind in ("UnsignedRangeValue", "SignedRangeValue"):
                ends = [int(r.end) for r in pvs.ranges]
                return max(ends) if ends and min(int(r.start) for r in pvs.ranges) >= 0 else None
            if kind == "InSetOfValues":
                vals = [int(x) for x in pvs.values]
                return max(vals) if vals and min(vals) >= 0 else None
        except Exception:
            return None
        return None

    def _residual_chunk_length(self, ssaf: Any, params: list[Any], length_idx: int,
                               dest_idx: int) -> tuple[str, str] | None:
        """``(total_expr, cursor_expr)`` when the copy is the chunked-read
        residual idiom -- ``n = total - cursor; op(buf + cursor, n)`` -- else None.

        This is the shape #499's closing note wanted suppressed to make
        ``--sink-class recv_overflow`` usable by default::

            while (progress < cap) {
                n = cap - progress;
                got = read(fd, staging + progress, n);
                progress += got;
            }

        It recognises the shape; it deliberately does NOT downgrade the class.
        The loop invariant bounds the length by ``total`` and the cumulative
        write extent by ``cursor + (total - cursor) == total`` -- but "bounded by
        total" is only safety when ``total <= sizeof(dest)``, and this engine
        cannot decide that here:

        * ``_dest_alloc`` sizes a buffer only from an allocating CALL, so a fixed
          stack array (the exact destination in #791's own repro) resolves to
          nothing -- there is no declared-size lookup anywhere in this module.
        * The extent identity needs ``cursor <= total``. That is the loop guard,
          and nothing here does control-dependence or dominance, so it cannot be
          established. Unsigned wrap makes the failure mode concrete rather than
          theoretical: with ``cursor > total`` the subtraction underflows to a
          huge length.

        So downgrading on the SHAPE alone would silently clear a genuine
        attacker-controlled write into an undersized buffer -- a false negative,
        which this engine treats as its worst outcome. What the shape can
        honestly do is tell the reader exactly which question is open, which is
        what the caller attaches to the finding.
        """
        if length_idx >= len(params) or dest_idx >= len(params):
            return None
        sub = next((e for e in self._def_chain(ssaf, params[length_idx])
                    if op_name(e) == "MLIL_SUB"), None)
        if sub is None:
            return None
        left, right = getattr(sub, "left", None), getattr(sub, "right", None)
        # Both operands must be variables: a constant subtrahend is `v - C`,
        # which `_linear_in_var` already models and `_bounded_copy_reason`
        # already reasons about. This idiom is var-minus-var.
        if self._int_const(left) is not None or self._int_const(right) is not None:
            return None
        cursor_ids = self._chain_identities(ssaf, right)
        if not cursor_ids or not self._chain_identities(ssaf, left):
            return None
        # The SAME cursor must OFFSET the destination at UNIT STRIDE -- that is
        # what makes this a residual chunk rather than an unrelated subtraction,
        # and it is the entire basis of the sentence the caller attaches
        # ("the cumulative extent is bounded by `total`"). That sentence is the
        # identity `cursor + (total - cursor) == total`, so it needs an ADD whose
        # one operand IS the cursor:
        #
        # * a SCALED index (`&staging + (cursor << 2)`, an array of >1-byte
        #   elements) writes `cursor*stride + (total - cursor)` bytes, which
        #   exceeds `total` for any nonzero cursor;
        # * a pointer LOADED from a table the cursor indexes
        #   (`dst = [&table + cursor]`) is not offset by the cursor at all -- the
        #   cursor picks WHICH buffer, so `total` says nothing about the extent.
        #
        # Testing "the cursor appears among the destination chain's reads" admits
        # both, and then the disclosure asserts a bound that does not hold and
        # sends the triager to the wrong question. Requiring the ADD rejects both
        # structurally: a scale sits in a SHL/MUL whose operand chain does not
        # resolve to the cursor, and a load is not an ADD.
        #
        # The chain walk stays, for the reason it was added: on real MLIL the
        # address arithmetic lives behind its own variable, and at -O0 behind
        # SEVERAL copies. Cross-dogfood measured `rsi#2 -> rcx_1#2 -> ADD` on the
        # destination and `rdx_1#2 -> n#2 -> rax_3#6(SUB)` on the length at -O0,
        # plus an `MLIL_SX` between the operand and the SUB whenever the length
        # is written `int n` (read takes size_t, so the widening is always
        # there). A single hop found 2 of 6 real variants.
        # The ADD must be `base + cursor` with the cursor applied EXACTLY ONCE to
        # an otherwise opaque base. `any ADD with the cursor as an operand` is
        # not enough, because the other operand can itself carry arithmetic:
        #   (&buf + hdrlen) + cursor -> extent is `hdrlen + total`
        #   (&buf + cursor)  + cursor -> extent is `total + cursor` (stride 2)
        # In both, a destination that genuinely holds `total` bytes still
        # overflows, so the sentence would send the reader to the wrong question.
        # "Opaque" is deliberately broad -- an address-of, a const pointer, a
        # malloc return, a bare pointer parameter all qualify -- because the
        # extent identity holds for ANY base the cursor offsets once. Only
        # further arithmetic in the base disqualifies it.
        for expr in self._def_chain(ssaf, params[dest_idx]):
            if op_name(expr) != "MLIL_ADD":
                continue
            sides = (getattr(expr, "left", None), getattr(expr, "right", None))
            for i, side in enumerate(sides):
                if not (self._chain_identities(ssaf, side) & cursor_ids):
                    continue
                if self._is_opaque_base(ssaf, sides[1 - i], cursor_ids):
                    return str(left), str(right)
        return None

    # An ALLOW-list, deliberately: a denylist of displacing opcodes is the wrong
    # shape for a soundness gate, because every opcode nobody thought of defaults
    # to "safe to claim". Measured on the denylist version: a base displaced with
    # `MLIL_ADDRESS_OF_FIELD` (`&buf[K]`, the canonical lift of a constant offset
    # into a fixed array -- the destination kind #791's own repro uses), or with
    # `MLIL_OR`/`MLIL_ADC` (the spellings an optimiser picks for the same
    # address arithmetic), all still qualified, so two spellings of one source
    # line disagreed. These are the forms a pointer BASE can legitimately take;
    # anything else is either a displacement or something this engine has not
    # reasoned about, and in both cases the honest answer is to withhold the
    # claim rather than make it.
    _OPAQUE_BASE_OPS = frozenset({
        # the value itself, and copies/widenings of it
        "MLIL_VAR", "MLIL_VAR_SSA", "MLIL_VAR_ALIASED", "MLIL_VAR_SSA_FIELD",
        "MLIL_SX", "MLIL_ZX",
        # a whole object's address, a link-time address, an immediate
        "MLIL_ADDRESS_OF", "MLIL_CONST_PTR", "MLIL_CONST", "MLIL_IMPORT",
        # a pointer read out of memory: opaque, and the cursor offsets it once
        "MLIL_LOAD", "MLIL_LOAD_SSA",
    })

    def _is_opaque_base(self, ssaf: Any, expr: Any, cursor_ids: set) -> bool:
        """Whether *expr* is a pointer base carrying no displacement of its own.

        The non-cursor half of a residual-chunk destination. A base reached
        through copies and width extensions counts (that is what ``_def_chain``
        sees through); one computed by further address arithmetic does not,
        because then the write extent is that displacement PLUS the chunk rather
        than the chunk alone.

        Three things make it False, and each was a measured false claim:

        * an op outside the allow-list above -- a displacement, or an operation
          this engine has not reasoned about;
        * a base that IS the cursor (`cursor + cursor`, stride 2 with the base
          folded away): the extent is `total + cursor`;
        * a chain that ran out of budget without terminating. ``_def_chain`` is
          bounded, so "no arithmetic in the chain" really means "none within the
          budget"; a displaced base behind more copies than that was claimed.
          Real chains are two or three hops, so refusing at the bound costs
          nothing and is the only answer that cannot be wrong.
        """
        if expr is None:
            return False
        limit = 8
        chain = self._def_chain(ssaf, expr, limit=limit)
        if len(chain) >= limit:
            return False
        if not chain or any(op_name(e) not in self._OPAQUE_BASE_OPS for e in chain):
            return False
        return not (self._chain_identities(ssaf, expr) & cursor_ids)

    def _def_chain(self, ssaf: Any, expr: Any, limit: int = 8) -> list[Any]:
        """*expr* and the expressions it resolves to through pure SSA copies and
        width extensions, nearest first.

        Bounded and cycle-safe. Sign/zero extensions are transparent because a
        widened value denotes the same quantity -- `sx.q(n)` IS `n` for the
        purpose of asking "what computed this".
        """
        out: list[Any] = []
        seen: set = set()
        cur = expr
        for _ in range(limit):
            if cur is None:
                break
            out.append(cur)
            if op_name(cur) in ("MLIL_SX", "MLIL_ZX"):
                cur = getattr(cur, "src", None)
                continue
            var = self._as_single_ssa_var(cur)
            if var is None:
                break
            ident = (var_key(var), getattr(var, "version", None))
            if ident in seen:
                break
            seen.add(ident)
            try:
                defn = ssaf.get_ssa_var_definition(var)
            except Exception:
                defn = None
            cur = getattr(defn, "src", None) if defn is not None else None
        return out

    def _chain_identities(self, ssaf: Any, expr: Any) -> set:
        """Canonical ``(key, version)`` identities *expr* can denote along its
        copy/extension chain -- the comparison key for "is this the same value",
        tolerant of the widening and copying a real lifter inserts."""
        ids: set = set()
        for e in self._def_chain(ssaf, expr):
            var = self._as_single_ssa_var(e)
            if var is None:
                continue
            canon = self._canonical_ssa_var(ssaf, var)
            ids.add((var_key(canon), getattr(canon, "version", None)))
        return ids

    def _const_value(self, expr: Any) -> int | None:
        """The integer constant of a CONST/CONST_PTR expression, else None."""
        if op_name(expr) in ("MLIL_CONST", "MLIL_CONST_PTR"):
            c = getattr(expr, "constant", None)
            try:
                return int(c) if c is not None else None
            except (TypeError, ValueError):
                return None
        return None

    def _syscall_bound_for_length(self, ssaf: Any, length_expr: Any) -> int | None:
        """If the copy length is the return value of a modeled receive call whose
        bounding count argument is a constant, return that constant max, else
        None. Recognizes e.g. ``n = read(fd, buf, 0x1000); memcpy(dst, buf, n)``
        -- the length is attacker-derived but provably ``<= 0x1000``, so it is a
        bounded copy, not an unbounded overflow (#159). Mirrors
        ``_alloc_size_expr``'s def-chain walk."""
        var = length_expr if is_ssa_var(length_expr) else None
        if var is None:
            reads = expr_reads(length_expr)
            if len(reads) == 1:
                var = reads[0]
        if var is None:
            return None
        var = self._canonical_ssa_var(ssaf, var)
        try:
            d = ssaf.get_ssa_var_definition(var)
        except Exception:
            d = None
        if d is None or "CALL" not in op_name(d):
            return None
        callee = self._callee_name(self._resolve_direct_target(d))
        _, model = lookup_model(self.models, callee)
        rb = (model or {}).get("return_bound") or {}
        bidx = rb.get("max_from_arg")
        if bidx is None:
            return None
        aparams = self._call_params(d)
        if bidx >= len(aparams):
            return None
        return self._const_value(aparams[bidx])

    def _local_definition_for(self, name: str | None) -> Any | None:
        """An in-binary DEFINED function with this name, or None.

        On the receive/deserialization surface a locally-defined exported
        function is often invoked through a PLT/GOT stub to its re-imported
        export, so the call resolves to the import side and forward taint can't
        descend into the real body without a manual --resolve-map. When the same
        name also has a DEFINED (non-imported) function symbol with a body in
        this binary, bridge to it (#46 item 3). Conservative: returns a function
        only when it's a real internal definition, never a thunk/import, so it
        can't change thunk-following semantics."""
        if not name or self.bv is None:
            return None
        for sym in _symbols_by_name(self.bv, name):
            stype = str(getattr(getattr(sym, "type", None), "name", "") or "")
            if "Imported" in stype or "External" in stype:
                continue  # the import side, not the definition
            try:
                addr = int(getattr(sym, "address"))
            except Exception:
                continue
            fn = function_at(self.bv, addr)
            if self._is_internal(fn):
                return fn
        return None

    def _is_internal(self, fn: Any) -> bool:
        """True if a call target is an in-binary function worth descending into
        (not a PLT/import thunk, which we model instead)."""
        return self._not_internal_reason(fn) is None

    @staticmethod
    def _not_internal_reason(fn: Any) -> str | None:
        """Why a call target is NOT descended into, or None when it is.

        ``no_function`` / ``import`` are the genuinely external shapes;
        ``thunk`` is an in-binary stub the caller may still resolve to its
        target. ``analysis_skipped:<BN reason>`` and ``no_mlil`` are IN-BINARY
        bodies Binary Ninja did not analyse (an oversized function past
        ``analysis.limits.maxFunctionSize``, a failed lift) -- the caveat for
        those must not call the function "external" (#1001 item 3). The import
        check runs first: a PLT stub is a thunk AND an import symbol."""
        if fn is None:
            return "no_function"
        sym = getattr(fn, "symbol", None)
        stype = str(getattr(getattr(sym, "type", None), "name", "") or "")
        if stype in {"ImportedFunctionSymbol", "LibraryFunctionSymbol",
                     "ImportAddressSymbol", "ExternalSymbol"}:
            return "import"
        if getattr(fn, "is_thunk", False):
            return "thunk"
        try:
            has_body = len(list(fn.mlil.instructions)) > 0
        except Exception:
            has_body = False
        if has_body:
            return None
        if getattr(fn, "analysis_skipped", False):
            try:
                raw = fn.analysis_skip_reason
                # An IntEnum: str() is the bare number, so prefer the member name.
                why = getattr(raw, "name", None) or str(raw)
            except Exception:
                why = "unknown"
            return f"analysis_skipped:{why}"
        return "no_mlil"

    @staticmethod
    def _unanalysed_body_phrase(reason: str | None) -> str | None:
        """The caveat phrase for an IN-BINARY callee whose body BN never
        analysed, or None for a genuinely external one (see
        :meth:`_not_internal_reason`)."""
        if not reason:
            return None
        if reason.startswith("analysis_skipped:"):
            why = reason.split(":", 1)[1]
            if why == "ExceedFunctionSizeSkipReason":
                why = "it exceeds analysis.limits.maxFunctionSize"
            return f"Binary Ninja skipped its analysis: {why}"
        if reason == "no_mlil":
            return "Binary Ninja produced no MLIL for it"
        return None

    def _thunk_tail_targets(self, fn: Any) -> list[Any]:
        """Every function a thunk tail-calls, found on ANY of its instructions
        and in instruction order -- `follow_thunk` only takes a one-instruction
        body, but BN also marks a longer body that ends in a tail jump (an
        oversized function after `--force-analysis`) a thunk (#1001 review).
        Empty when no tail call has a constant target that is a function."""
        ssa = _mlil_ssa(fn)
        if ssa is None:
            return []
        targets: list[Any] = []
        for insn in _ssa_instructions(ssa):
            if "TAILCALL" not in op_name(insn) or getattr(insn, "dest", None) is None:
                continue
            try:
                addr = extract_dest_address(self.bv, insn.dest)
            except Exception:
                addr = None
            target = function_at(self.bv, addr) if addr is not None else None
            if target is not None and target is not fn and target not in targets:
                targets.append(target)
        return targets

    def _summarize(self, callee: Any, param_set: frozenset, depth: int, max_depth: int) -> dict[str, Any]:
        """Analyze *callee* with the given tainted parameter indices, caching the
        result per (callee, tainted-param-set) so it is computed once."""
        key = (int(callee.start), param_set)
        if key in self._cache:
            cached = self._cache[key]
            if cached is None:  # in-progress -> recursion cycle
                return {"reached_return": True, "out_params": frozenset(), "findings": [], "leaves": [],
                        "assumptions": [f"recursion cycle at {self._prose_function(callee)}; return conservatively tainted"],
                        "frontier": "recursion cycle stopped descent"}
            return cached
        self._cache[key] = None  # mark in-progress (cycle guard)
        locators = [{"kind": "param", "index": i} for i in sorted(param_set)]
        # _run_forward(callee) repoints self._bt_func_token at the callee; restore
        # the caller's token afterward so its fixpoint keeps hitting its own cache
        # entries (the #420 _buffer_target memo is keyed by this token).
        saved_tok = self._bt_func_token
        try:
            sub = self._run_forward(callee, locators, depth, max_depth, top=False)
        except TaintError as exc:
            # #811: record the unread body structurally as well as in prose. The
            # conservative degrade below (reached_return=True) is unchanged --
            # only the disclosure is added, never the propagation polarity.
            self._analysis_incomplete.add(str(getattr(callee, "name", "?")))
            sub = {"reached_return": True, "out_params": frozenset(), "findings": [], "leaves": [],
                   "assumptions": [f"could not analyze {self._prose_function(callee)}: {exc}; return conservatively tainted"],
                   "frontier": f"body could not be analyzed ({exc})"}
        finally:
            self._bt_func_token = saved_tok
        self._cache[key] = sub
        return sub

    @staticmethod
    def _frontier_leaf(ins: Any, callee_fn: Any, tainted_arg_indices, note: str) -> dict[str, Any]:
        """An honest "taint stopped here" leaf for an unmodeled in-binary callee
        that was NOT descended into (no mappable params, depth bound, recursion
        cycle, or an unanalyzable body). Extends the existing leaves[] honesty
        channel (#8) -- the call->callee hand-off that would otherwise vanish."""
        return {
            "kind": "unmodeled_callee",
            "address": hex(int(getattr(ins, "address", 0))),
            "callee": {"name": str(getattr(callee_fn, "name", "?")),
                       "display_name": _display_name(callee_fn) or "?",
                       "address": hex(int(getattr(callee_fn, "start", 0)))},
            "tainted_args": sorted(tainted_arg_indices),
            "note": note,
        }

    def _arg_under_recovered_leaf(self, ins: Any, callee_fn: Any, dropped: list,
                                  n_params: int) -> dict[str, Any] | None:
        """Disclose a PARTIAL arg drop: taint DID descend into *callee_fn*, but some
        tainted arg indices sit beyond its recovered arity (BN under-recovered the
        callee's signature), so those flows were dropped. Returns a leaf or None;
        degrades to None on any BN-API shortfall so it never fabricates a frontier.

        Two confirmation paths:
        - Register-passed indices are gated by ``_reg_reads_as_input`` (only args the
          callee actually consumes), so a leftover register value never cries wolf.
        - Stack-passed indices (i386 cdecl / MIPS-o32 / x86-64 varargs, #324) are
          confirmed by the caller-side recovery that placed them in ``dropped``: a
          store to the outgoing-argument stack slot is a deliberate argument, so the
          taint into it is disclosed rather than silently dropped (previously these
          were skipped, the "emit a truncation frontier" half of #324)."""
        variadic = _variadic_determination(callee_fn)
        if variadic is True:
            cc = getattr(callee_fn, "calling_convention", None)
            arg_regs = list(getattr(cc, "int_arg_regs", []) or [])
            # Named formals never enumerate the variadic tail. Every recovered
            # tainted tail argument is an untracked flow, regardless of ABI.
            name = str(getattr(callee_fn, "name", "?"))
            return {
                "kind": "arg_under_recovered",
                "address": hex(int(getattr(ins, "address", 0))),
                "callee": {"name": name, "display_name": _display_name(callee_fn) or name,
                           "address": hex(int(getattr(callee_fn, "start", 0)))},
                "recovered_params": n_params, "dropped_args": sorted(dropped),
                "stack_dropped_args": [i for i in sorted(dropped) if cc is not None and i >= len(arg_regs)],
                "cause": "varargs_not_followed", "callee_variadic": True,
                "note": self._varargs_tail_note(self._prose_function(callee_fn), n_params, dropped,
                                                hex(int(getattr(callee_fn, "start", 0)))),
            }
        try:
            cc = (getattr(callee_fn, "calling_convention", None)
                  or self.bv.platform.default_calling_convention)
            arg_regs = list(getattr(cc, "int_arg_regs", []) or [])
        except Exception:
            return None
        reg_confirmed: list[int] = []
        stack_confirmed: list[int] = []
        for i in sorted(dropped):
            if i >= len(arg_regs):
                # Stack-passed. Disclose only on a convention that HAS integer-arg
                # registers, where an index beyond that cutoff is unambiguously in the
                # outgoing-argument region and BN's direct-call arity-clamping makes a
                # spurious over-recovered stack param rare. On a pure-stack ABI (i386
                # cdecl, no arg regs) the register-style "does the callee read it" gate
                # cannot apply at all and BN's stack-arg modeling is weakest, so stay
                # silent rather than emit an unverifiable frontier -- BN also clamps
                # i386 direct calls to callee arity, so a real drop rarely reaches here
                # anyway (#324 FP audit).
                if arg_regs:
                    stack_confirmed.append(i)
                continue
            try:
                if self._reg_reads_as_input(callee_fn, arg_regs[i]):
                    reg_confirmed.append(i)
            except Exception:
                continue
        confirmed = sorted(reg_confirmed + stack_confirmed)
        if not confirmed:
            return None
        name = str(getattr(callee_fn, "name", "?"))
        common = {
            "kind": "arg_under_recovered",
            "address": hex(int(getattr(ins, "address", 0))),
            "callee": {"name": name,
                       "display_name": _display_name(callee_fn) or name,
                       "address": hex(int(getattr(callee_fn, "start", 0)))},
            "recovered_params": n_params,
            "dropped_args": confirmed,
            "stack_dropped_args": stack_confirmed,
        }
        if variadic is None:
            return {
                **common, "cause": "prototype_status_unknown", "callee_variadic": None,
                "note": (f"tainted arg(s) {confirmed} passed to {self._prose_function(callee_fn)} "
                         f"exceed its {n_params} recovered named parameter(s); these flows "
                         f"were NOT followed. Variadic status is undetermined: inspect the "
                         f"callee prototype and argument consumption before diagnosing a "
                         f"signature discrepancy or a tracking limitation."),
            }
        return {
            **common, "cause": "signature_under_recovered", "callee_variadic": False,
            "note": (f"tainted arg(s) {confirmed} passed to {self._prose_function(callee_fn)} "
                     f"exceed its {n_params} recovered parameter(s), and the callee is "
                     f"determined nonvariadic. The confirmed extra arguments disagree with "
                     f"that declaration; their flow was NOT followed. Recover the real "
                     f"prototype and apply `bn proto set {shlex.quote(name)} \"<prototype>\"`, then re-run."),
        }

    @staticmethod
    def _varargs_tail_note(name: str, n_params: int, idxs: list, at: str) -> str:
        """The #948 disclosure for taint landing in a determined-variadic
        callee's `...` tail: a tracking limit, stated with the observed counts,
        and never a prototype repair. No `param:N` names a vararg, so the seed
        inside the callee is a template (#1033 item 6)."""
        seed = command_template(f"bn taint forward -f {at} --source var:VAR")
        return (f"tainted arg(s) {sorted(idxs)} passed to {name} land in its variadic "
                f"tail: {name} is declared variadic (`...`) with {n_params} named "
                f"parameter(s), and taint is mapped only onto named parameters, so the "
                f"flow into its varargs was NOT followed. This is a tracking limit, not "
                f"a signature defect -- the prototype is already variadic and "
                f"redeclaring it changes nothing. Inspect how {name} consumes its "
                f"va_list (e.g. a v*printf call), or seed taint inside it from the "
                f"value it reads ({seed}).")

    def _descend(self, ins: Any, callee_fn: Any, tainted_args: dict, why: dict,
                 depth: int, max_depth: int, *, via: str | None = None,
                 caller_instrs: list | None = None,
                 caller_fn: Any = None) -> dict[str, Any]:
        """Recurse into a (direct or resolved-indirect) internal callee and return
        its findings with a caller-side path prefix prepended, plus whether it
        propagates taint to its return.

        When the callee is NOT actually descended into (no mappable parameters,
        depth bound reached, recursion cycle, or an unanalyzable body), the
        tainted call->callee hand-off is recorded as an ``unmodeled_callee``
        frontier leaf instead of silently dropping the flow (#8)."""
        n_params = len(list(getattr(callee_fn, "parameter_vars", []) or []))
        valid = frozenset(i for i in tainted_args if i < n_params)
        out: dict[str, Any] = {"findings": [], "reached_return": False, "leaves": [],
                               "assumptions": [], "out_params": frozenset(), "handoff_node": []}
        if not valid:
            out["reached_return"] = True
            leaf = self._frontier_leaf(
                ins, callee_fn, tainted_args,
                f"tainted arg(s) {sorted(tainted_args)} exceed the {n_params} recovered "
                f"named parameter(s) of {self._prose_function(callee_fn)}; flow was NOT "
                "followed. Inspect the callee prototype and argument consumption.")
            variadic = _variadic_determination(callee_fn)
            leaf.update(callee_variadic=variadic)
            if variadic is True:
                # #948: every tainted arg is in a correctly-variadic tail.
                leaf.update(cause="varargs_not_followed", callee_variadic=True,
                            note=self._varargs_tail_note(
                                self._prose_function(callee_fn), n_params, list(tainted_args),
                                hex(int(getattr(callee_fn, "start", 0)))))
                out["assumptions"].append(
                    f"tainted args to {self._prose_function(callee_fn)} land in its variadic tail, which "
                    f"is not followed; conservative")
            else:
                if variadic is False:
                    leaf.update(cause="signature_under_recovered",
                                note=leaf["note"] + " Recovered call arguments exceed a determined "
                                "nonvariadic declaration: recover the actual signature with "
                                f"`bn proto set {shlex.quote(callee_fn.name)} \"<prototype>\"` and re-run.")
                else:
                    leaf["cause"] = "prototype_status_unknown"
                out["assumptions"].append(
                    f"tainted args to {self._prose_function(callee_fn)} fall beyond its parameters; conservative")
            out["leaves"].append(leaf)
            return out
        if depth + 1 > max_depth:
            self._truncated = True
            self._truncation_causes.add("max_depth")
            out["reached_return"] = True
            out["assumptions"].append(
                f"max interprocedural depth {max_depth} reached at {self._prose_function(callee_fn)}; not descended")
            out["leaves"].append(self._frontier_leaf(
                ins, callee_fn, tainted_args,
                f"tainted data passed to in-binary callee with no model; depth bound "
                f"({max_depth}) stopped descent -- investigate or raise --depth"))
            return out
        sub = self._summarize(callee_fn, valid, depth + 1, max_depth)
        first_hit = tainted_args[sorted(valid)[0]][0]
        prefix = self._reconstruct_path(first_hit, why)
        note = f"calls {self._prose_function(callee_fn)} with tainted arg(s) {sorted(valid)}"
        if via:
            note = f"[{via}-resolved] " + note
        prefix.append(_instr_dict(ins, reason=note, tainted=[node_label(first_hit, why)],
                                  callee=getattr(callee_fn, "name", None)))
        out["handoff_node"] = [n for i in sorted(valid) for n in tainted_args[i]]
        for f in sub["findings"]:
            rebuilt = {"sink": f["sink"], "path": prefix + f["path"]}
            # Round-3 blocker: a callee's OWN unconditional-flow tag (set by its
            # own direct-sink detection, e.g. an internal gets()) must survive
            # this rebuild -- previously it was silently dropped, so a finding
            # correctly tagged inside the callee re-rendered as causally reached
            # by the caller's --source once it crossed this boundary.
            if f.get("_unconditional_flow"):
                rebuilt["_unconditional_flow"] = True
            out["findings"].append(rebuilt)
        out["leaves"] = list(sub["leaves"])
        # Disclose a PARTIAL arg drop: taint descended with the in-range args, but
        # any tainted arg beyond the callee's recovered arity was filtered out of
        # `valid` above and would otherwise vanish silently (#442-class). Gated so
        # it only fires on args the callee actually reads.
        dropped = [i for i in tainted_args if i >= n_params]
        if dropped:
            under = self._arg_under_recovered_leaf(ins, callee_fn, dropped, n_params)
            if under is not None:
                out["leaves"].append(under)
        frontier = sub.get("frontier")
        if frontier:
            out["leaves"].append(self._frontier_leaf(
                ins, callee_fn, tainted_args,
                f"tainted data passed to in-binary callee with no model; {frontier} "
                f"-- investigate"))
        out["assumptions"] = list(sub["assumptions"])
        out["reached_return"] = sub["reached_return"]
        out["out_params"] = sub.get("out_params", frozenset())
        out["out_param_elems"] = sub.get("out_param_elems", frozenset())
        if not sub["reached_return"]:
            lost = self._valueless_return_leaf(ins, callee_fn, tainted_args, caller_instrs)
            if lost is not None:
                out["leaves"].append(lost)
        if not ssa_writes(ins) and "TAILCALL" not in op_name(ins):
            dropped = self._dropped_result_leaf(ins, callee_fn, tainted_args, caller_fn,
                                                caller_instrs)
            if dropped is not None:
                out["leaves"].append(dropped)
        return out

    def _dropped_result_leaf(self, ins: Any, callee_fn: Any, tainted_args: dict,
                             caller_fn: Any, caller_instrs: list | None) -> dict[str, Any] | None:
        """#1034 item 5, at the call site (re-dogfood of #1050): BN's MLIL gives
        a call with tainted arguments NO result, yet the caller's machine code
        reads the calling convention's return register after it before writing
        it -- so the caller uses a value BN dropped, and whatever the callee
        returned is not followed. The callee's shape is not predicted; the fact
        is read off the caller's LLIL (:meth:`_return_register_read_after`)."""
        reg = self._return_register_read_after(caller_fn, ins, caller_instrs)
        if reg is None:
            return None
        call_at = hex(int(getattr(ins, "address", 0)))
        fn_at = hex(int(getattr(caller_fn, "start", 0)))
        leaf = self._frontier_leaf(
            ins, callee_fn, tainted_args,
            f"tainted arg(s) {sorted(tainted_args)} passed to "
            f"{self._prose_function(callee_fn)} at {call_at}: Binary Ninja's IL gives the "
            f"call no result, but the caller's machine code reads the return register "
            f"{reg} after it, so the value the callee returns is not followed -- read the "
            f"call site with `bn disasm {fn_at}`")
        leaf["cause"] = "return_value_unrecovered"
        leaf["return_register"] = reg
        return leaf

    def _return_register_read_after(self, caller_fn: Any, ins: Any,
                                    caller_instrs: list | None) -> str | None:
        """The calling convention's integer return register when the caller's
        LLIL, on some path after the call at *ins*, reads it before anything
        writes all of it; else None (also whenever the view cannot answer).

        Another call clobbers it (end of that path). A partial write without
        an implicit extension (`al`) does not end the path; `w0`/`eax` writes
        zero-extend and do. Falling off a return counts as a read only when the
        caller's MLIL returns carry a value -- a void caller's `ret` uses
        nothing."""
        try:
            cc = getattr(caller_fn, "calling_convention", None) \
                or self.bv.platform.default_calling_convention
            arch = getattr(caller_fn, "arch", None) or self.bv.arch
            # The scalar return register only: the high half (`x1`/`rdx`) of a
            # two-register return is an ordinary argument register after a call.
            ret_regs = {str(r) for r in (getattr(cc, "int_return_reg", None),) if r}
            call_llil = caller_fn.get_low_level_il_at(int(getattr(ins, "address", 0)))
            llil = caller_fn.llil
        except Exception:  # noqa: BLE001 - no LLIL, no answer
            return None
        if not ret_regs or call_llil is None:
            return None

        def reg_info(name: str) -> tuple[str, bool]:
            """(full-width register, whether a write of *name* replaces all of it)."""
            try:
                info = arch.regs[name]
            except Exception:  # noqa: BLE001
                return name, True
            full = str(info.full_width_reg)
            extend = int(getattr(info, "extend", 0) or 0)
            return full, full == name or extend != 0

        full_ret = {reg_info(r)[0] for r in ret_regs}
        value_returning = any(op_name(i) == "MLIL_RET" and getattr(i, "src", None)
                              for i in caller_instrs or [])

        def reads(expr: Any, acc: set[str]) -> None:
            if expr is None or not hasattr(expr, "operation"):
                return
            if op_name(expr) in ("LLIL_REG", "LLIL_REG_SSA", "LLIL_REG_SSA_PARTIAL"):
                name = getattr(getattr(expr, "src", None), "name", None)
                if name:
                    acc.add(reg_info(str(name))[0])
            for operand in getattr(expr, "operands", None) or []:
                for item in operand if isinstance(operand, list) else [operand]:
                    reads(item, acc)

        def scan(index: int, end: int) -> str | bool | None:
            """A register name: read first. True: the path ends here (written,
            clobbered, or returned). None: falls through to the successors."""
            for i in range(index, end):
                cur = llil[i]
                opn = op_name(cur)
                got: set[str] = set()
                reads(cur, got)
                hit = got & full_ret
                if hit:
                    return sorted(hit)[0]
                if "CALL" in opn or opn in ("LLIL_TAILCALL", "LLIL_JUMP", "LLIL_JUMP_TO"):
                    return True
                if opn == "LLIL_RET":
                    return sorted(full_ret)[0] if value_returning else True
                if opn in ("LLIL_SET_REG", "LLIL_SET_REG_SPLIT"):
                    for dest in (getattr(cur, "dest", None), getattr(cur, "hi", None),
                                 getattr(cur, "lo", None)):
                        name = getattr(dest, "name", None)
                        if name:
                            full, whole = reg_info(str(name))
                            if full in full_ret and whole:
                                return True
            return None

        try:
            block = call_llil.il_basic_block
            first = scan(int(call_llil.instr_index) + 1, int(block.end))
            if isinstance(first, str):
                return first
            if first is True:
                return None
            seen = {int(block.start)}
            work = [block]
            while work:
                for edge in list(work.pop().outgoing_edges):
                    t = edge.target
                    if int(t.start) in seen:
                        continue
                    seen.add(int(t.start))
                    r = scan(int(t.start), int(t.end))
                    if isinstance(r, str):
                        return r
                    if r is None:
                        work.append(t)
        except Exception:  # noqa: BLE001 - a view without blocks proves nothing
            return None
        return None

    def _valueless_return_leaf(self, ins: Any, callee_fn: Any, tainted_args: dict,
                               caller_instrs: list | None) -> dict[str, Any] | None:
        """#1034 item 5: a frontier where BN's recovered IL cannot say whether a
        tainted argument reaches the callee's return.

        On AArch64 `long ident(long v) { return v; }` is a bare `ret` that leaves
        x0 unchanged, and BN recovers it as `return` with no operand. Taint then
        found no tainted return and the caller's flow vanished into a confident
        all-clear, though the x86-64 build of the same source reports the sinks.
        No guess about the register is made here; two facts BN itself reports
        are compared after the walk: every return of the callee carries no value,
        while the call site consumes one -- BN gave the call an output, or it is
        a tail call from a function whose own returns carry a value. Then the
        flow is unknown, and a frontier says so instead of a clean result."""
        try:
            callee_instrs = self._instrs(self._ssa_func(callee_fn))
        except TaintError:
            return None
        rets = [i for i in callee_instrs if op_name(i) == "MLIL_RET"]

        def operands(ret: Any) -> list:
            srcs = getattr(ret, "src", None)
            if not isinstance(srcs, (list, tuple)):
                srcs = [] if srcs is None else [srcs]
            return list(srcs)

        if not rets or any(operands(r) for r in rets):
            return None
        if ssa_writes(ins):
            consumer = "the call's result is used"
        elif "TAILCALL" in op_name(ins) and any(
                op_name(i) == "MLIL_RET" and operands(i) for i in caller_instrs or []):
            consumer = "it is tail-called by a function that returns a value"
        else:
            return None
        at = hex(int(getattr(callee_fn, "start", 0)))
        leaf = self._frontier_leaf(
            ins, callee_fn, tainted_args,
            f"tainted arg(s) {sorted(tainted_args)} passed to "
            f"{self._prose_function(callee_fn)}, whose returns Binary Ninja recovered "
            f"with no value, while {consumer}: whether the arguments reach that value "
            f"is unknown (a register the body leaves unchanged is not modeled) -- read "
            f"the body with `bn disasm {at}`")
        leaf["cause"] = "return_value_unrecovered"
        return leaf

    def _unrecovered_arg_frontier(self, ins: Any, tainted: set) -> dict[str, Any] | None:
        """Honest frontier for the #381 silent drop. When a tainted value sits in
        an ABI argument register that BN did NOT include in this call's recovered
        params -- the callee's arity was under-recovered (Thumb 0-arity miss /
        variadic mis-prototype) -- the flow into the callee would otherwise vanish
        with no breadcrumb (the bare ``continue`` below). Return an
        ``unmodeled_callee`` frontier leaf when a *missing* arg register (one past
        the recovered arity) holds a tainted SSA value, else None.

        Block-local + register-passed only: it inspects the last write to each
        missing arg register within the call's own basic block (the dominant
        codegen -- args set immediately before the call). Stack-passed varargs
        (i386 cdecl) are NOT covered here -- that overlaps #324. Degrades to None
        on any BN-API shortfall, so it never fabricates a frontier."""
        try:
            target = self._resolve_direct_target(ins)
            if target is None:
                return None
            callee = function_at(self.bv, target)
            if not self._is_internal(callee):
                return None
            cc = (getattr(callee, "calling_convention", None)
                  or self.bv.platform.default_calling_convention)
            arg_regs = list(getattr(cc, "int_arg_regs", []) or [])
            n = len(list(getattr(callee, "parameter_vars", []) or []))
            if n >= len(arg_regs):
                return None  # every ABI arg register is accounted for by a param
            arch = self.bv.arch
            # Only the FIRST unrecovered arg slot (register position == recovered
            # arity n). Checking higher slots too would false-positive on a legit
            # low-arity callee whose r1..r3 merely hold tainted leftovers; the
            # first-missing register being SET UP (written) tainted right before
            # the call in the call's own block is the tight "BN under-recovered an
            # actual argument" signal (e.g. Thumb 0-arity: r0 = argv[1]; call f()).
            try:
                first_missing = arch.get_reg_index(arg_regs[n])
            except Exception:
                return None
            missing = {first_missing: n}
            blk = getattr(ins, "il_basic_block", None)
            if blk is None:
                return None
            # BN's VariableSourceType.RegisterVariableSourceType == 1 (this module
            # imports no binaryninja symbols, so compare the stable enum int).
            REGISTER_SOURCE = 1
            def _reads_tainted(cur: Any) -> bool:
                for r in ssa_reads(cur):
                    if (var_key(r), getattr(r, "version", None)) in tainted \
                            or (var_key(r), None) in tainted:
                        return True
                return False

            # Last write to each missing arg register before the call (a later
            # write overwrites an earlier dead one, so keep only the reaching def).
            # The register carries taint if its written SSA var is itself a tainted
            # node OR its defining instruction reads a tainted value (e.g.
            # ``r0 = [argv+4]`` -- the load result isn't auto-tainted, but the
            # argv read is), mirroring the engine's read_taint/arg_taint check.
            last: dict[int, Any] = {}
            for cur in blk:
                if getattr(cur, "instr_index", -1) >= getattr(ins, "instr_index", -1):
                    break
                for w in getattr(cur, "vars_written", []) or []:
                    var = getattr(w, "var", None)
                    if var is None:
                        continue
                    try:
                        is_reg = int(var.source_type) == REGISTER_SOURCE
                    except Exception:
                        is_reg = False
                    storage = getattr(var, "storage", None)
                    if is_reg and storage in missing:
                        last[storage] = (cur, var_key(w), getattr(w, "version", None))
            for storage, (cur, key, ver) in last.items():
                if not ((key, ver) in tainted or (key, None) in tainted or _reads_tainted(cur)):
                    continue
                pos = missing[storage]
                # Decisive gate against false frontiers: only flag when the callee
                # actually READS this register as an input (read-before-write in its
                # body). A tainted value merely *sitting* in a non-argument register
                # (a loop-carried phi, a leftover return value, or arg-setup for a
                # PRIOR call) is not a dropped argument -- the callee never consumes
                # it. Requiring the callee to use the register as an incoming value
                # is what separates a genuine under-recovered arg from a leftover.
                if not self._reg_reads_as_input(callee, arg_regs[pos]):
                    continue
                return self._frontier_leaf(
                    ins, callee, [pos],
                    f"in-binary callee {getattr(callee, 'name', '?')} reads arg "
                    f"register {arg_regs[pos]} as an input but BN recovered only "
                    f"{n} parameter(s), so a tainted value passed there is not in "
                    f"the call's args and the flow into the callee was not followed "
                    f"-- the callee's args were under-recovered (taint #381)")
        except Exception:
            return None
        return None

    def _reg_reads_as_input(self, callee: Any, reg_name: str) -> bool:
        """True if *callee* consumes register *reg_name* as an incoming value --
        i.e. it reads the register's ENTRY SSA version (``reg#0``), the value the
        caller passed. This is the discriminator that stops a stale/leftover value
        in a NON-argument register (a loop-carried phi, a tainted return value, or
        arg-setup for a PRIOR call) from fabricating a #381 frontier: only a
        register the callee actually reads as input can be a dropped argument.

        ``reg#0`` is the entry value -- it is only ever READ (SSA writes produce
        versions >= 1), so finding it anywhere in the SSA operands means the callee
        uses the register's incoming value. Scans the LLIL-SSA operands directly so
        this module keeps importing no binaryninja symbols. Degrades to False on
        any shortfall, so it never *adds* a frontier on uncertainty."""
        try:
            lssa = getattr(getattr(callee, "llil", None), "ssa_form", None)
            if lssa is None:
                return False

            def _has_entry_read(node: Any, depth: int = 0) -> bool:
                if depth > 12:
                    return False
                for op in getattr(node, "operands", []) or []:
                    if type(op).__name__ == "SSARegister":
                        if getattr(op, "version", None) == 0 \
                                and str(getattr(op, "reg", "")) == reg_name:
                            return True
                    elif hasattr(op, "operands"):
                        if _has_entry_read(op, depth + 1):
                            return True
                return False

            for ins in lssa.instructions:
                if _has_entry_read(ins):
                    return True
            return False
        except Exception:
            return False

    def _call_targets_from_pvs(self, pvs: Any) -> list[int]:
        """Concrete call-target addresses from a PossibleValueSet (delegates to
        the module-level :func:`targets_from_pvs`)."""
        return targets_from_pvs(pvs)

    def _run_forward(self, func: Any, locators: list[dict[str, Any]], depth: int,
                     max_depth: int, *, top: bool) -> dict[str, Any]:
        ssaf = self._ssa_func(func)
        instrs = self._instrs(ssaf)
        self._funcs_visited.add(int(getattr(func, "start", 0)))
        self._max_depth_seen = max(self._max_depth_seen, depth)
        # Scope the structural _buffer_target cache to THIS function so a callee's
        # expr_index can't collide with a caller's (#420). _summarize saves/restores
        # this token around the one recursive descent call. Address 0 is a valid
        # function start (a firmware/VxWorks reset-vector entry), so gate on
        # `is not None` -- only a genuinely absent start (a degenerate fake) leaves
        # the token None and makes _buffer_target bypass the cache.
        _start = getattr(func, "start", None)
        self._bt_func_token = int(_start) if _start is not None else None

        tainted: set[tuple] = set()
        why: dict[tuple, dict[str, Any]] = {}
        assumptions: list[str] = []
        leaves: list[dict[str, Any]] = []
        findings: list[dict[str, Any]] = []
        recorded_sinks: set[tuple] = set()
        processed_calls: set[tuple] = set()  # (call_addr, tainted-arg-set) already descended
        out_params: set[int] = set()         # this func's params whose pointee got tainted
        out_param_elems: set = set()         # (param_idx, field) descriptor-array elems filled (#319b)
        # Static-state tokenizers (`continuation` models, e.g. strtok): every
        # call SITE in THIS function whose `null_arg` was tainted, keyed by
        # (model key, IL index) -> (tainted node, call address, call
        # instruction). A later `strtok(NULL, ..)` reads that state, not an
        # argument, so without it every token after the first silently dropped
        # taint -- no leaf, no assumption. The set only grows, and a new site
        # forces another fixpoint pass (#932): an optimised lift can list the
        # continuation BEFORE the initializer that arms it, and a pass that
        # discovered state without revisiting left that continuation unarmed.
        continuation_state: dict[tuple[str, int], tuple] = {}
        # model key -> every non-NULL call of a continuation model in this
        # function, collected up front so the reset test does not depend on IL
        # order. A candidate is a clean reset while its string is untainted
        # (taint only grows, so it can stop being one, never become one again);
        # clean resets kill a continuation only when they lie on EVERY path from
        # the tainted call to the `strtok(NULL)` (`_reaches_avoiding`).
        reset_candidates: dict[str, list[tuple]] = {}
        for _cand in instrs:
            if not self._is_call(_cand):
                continue
            _cname = self._callee_name(self._resolve_direct_target(_cand))
            _ckey, _cmodel = lookup_model(self.models, _cname)
            _co = (_cmodel or {}).get("continuation")
            if _co is None:
                continue
            _ci = int(_co.get("null_arg", 0))
            _cp = self._call_params(_cand)
            if _ci < len(_cp) and self._int_const(_cp[_ci]) != 0:
                reset_candidates.setdefault(_ckey or _cname or "?", []).append(
                    (int(getattr(_cand, "address", 0)), _cand, _cp, _ci))

        def clean_resets(ckey: str) -> list[tuple]:
            return [(a, rins) for a, rins, rparams, ci in reset_candidates.get(ckey, [])
                    if self._token_hit_node(ssaf, rparams, f"*arg:{ci}", tainted) is None]
        # (continuation call, state site) -> the facts its caveat is written from
        # once the fixpoint has converged.
        continuations: dict[tuple, tuple] = {}

        def ins_key(i: Any) -> int:
            return int(getattr(i, "instr_index", getattr(i, "address", 0)))
        reached_return = False

        def add_assumption(msg: str) -> None:
            if msg not in assumptions:
                assumptions.append(msg)

        def taint_node(node: tuple, label: str, ins: Any, reason: str, parents: list[tuple]) -> bool:
            if node in tainted:
                return False
            tainted.add(node)
            why[node] = {"label": label, "instr": ins, "reason": reason, "parents": list(parents)}
            return True

        # #615 review "New": nodes the unconditional-sink block below injects
        # taint into, INDEPENDENTLY of --source -- tracked so a DOWNSTREAM sink
        # reached only via that injected taint (e.g. a strcpy() fed by a gets()
        # buffer) is not rendered as though the run's declared --source causally
        # reached it (it didn't; the buffer's own always-unsafe read created the
        # taint). `_rooted_at_unconditional` walks a finding's causal chain (the
        # same `why`-parent walk `_reconstruct_path` uses) to check this.
        unconditional_roots: set[tuple] = set()

        def _rooted_at_unconditional(node: tuple) -> bool:
            if not unconditional_roots:
                return False
            # #827: this is a pure REACHABILITY question -- "is any ancestor of
            # `node` an unconditionally-injected root?" -- so unlike
            # `_reconstruct_path` (which must render one chain) it has no reason
            # to prefer parents[0]. Following only the first parent answered
            # False whenever the injected root sat behind a PHI's second parent,
            # UNDER-tagging the finding and letting a sink reached through
            # always-unsafe injected taint be rendered as though the run's
            # declared --source causally reached it. Walk every parent; `seen`
            # bounds the traversal on cyclic `why` chains exactly as before.
            stack = [node]
            seen: set = set()
            while stack:
                cur = stack.pop()
                if cur is None or cur in seen:
                    continue
                if cur in unconditional_roots:
                    return True
                seen.add(cur)
                entry = why.get(cur)
                if entry is None:
                    continue
                stack.extend(entry.get("parents") or [])
            return False

        def _tag_unconditional_flow(finding: dict[str, Any], hit_nodes: list[tuple]) -> dict[str, Any]:
            if hit_nodes and any(_rooted_at_unconditional(n) for n in hit_nodes):
                finding["_unconditional_flow"] = True
            return finding

        # #193 Part 1: recv-buffer slots registered by the seed (slot key -> info)
        # and the subset that actually correlated to a forward re-load. Both are
        # local to this run, so a descended callee's run keeps its own.
        buffer_slots: dict = {}
        correlated_slots: set = set()
        seeded = self._seed_forward(func, ssaf, instrs, locators, taint_node, add_assumption,
                                    buffer_slots, leaves)
        if not seeded:
            if top:
                raise TaintError("no taint sources resolved; check --source locator")
            return {"reached_return": False, "out_params": set(), "findings": [],
                    "leaves": [], "assumptions": []}

        # Honesty signal: a visited function with unlifted instructions is a
        # potential silent dataflow hole -- BN couldn't model those ops, so taint
        # through them isn't tracked. Surface it the way unmodeled calls/coarse
        # stores already are, instead of flowing through silently (#206). The
        # function-scoped ASSUMPTION below is flow-INSENSITIVE (fires whenever the
        # function contains any unlifted op); it is kept as an informational note.
        # The claim gate is driven instead by the flow-SENSITIVE leaf emitted in
        # the fixpoint loop (only when a tainted value actually reaches such an
        # instruction) so the gate does not over-withhold on every SIMD/FP function.
        unimpl = self._unimplemented_addrs(func, instrs)
        unimpl_addr_set = set(unimpl)
        if unimpl:
            sample = ", ".join(hex(a) for a in unimpl[:5])
            more = "" if len(unimpl) <= 5 else f", +{len(unimpl) - 5} more"
            add_assumption(
                f"{self._prose_function(func)} contains {len(unimpl)} unlifted/unimplemented "
                f"instruction(s) (e.g. {sample}{more}); BN's lifter could not model "
                f"them, so a tainted value passing through is not tracked (possible "
                f"silent hole) -- see `bn function info` for the full list")

        # Broad-source nudge: only for the user's own param: source (top run), not
        # the synthetic param locators of descended callees (#219).
        if top:
            for loc in locators:
                if loc.get("kind") != "param":
                    continue
                try:
                    pv = self._param_var(func, int(loc["index"]))
                except (KeyError, ValueError, TypeError):
                    pv = None
                if pv is not None:
                    hint = self._broad_source_hint(pv, int(loc["index"]))
                    if hint:
                        add_assumption(hint)

        def read_taint(ins: Any) -> list[tuple]:
            hit = []
            for r in ssa_reads(ins):
                k = var_key(r); ver = getattr(r, "version", None)
                if (k, ver) in tainted:
                    hit.append((k, ver))
                elif (k, None) in tainted:
                    hit.append((k, None))
            return hit

        def arg_taint(expr: Any) -> list[tuple]:
            # An argument carries taint either as a tainted scalar value (a length
            # register) or as a pointer to a tainted buffer (system(char*),
            # f(buf), helper(buf)). Check both so pointer args fire too.
            hit = []
            for r in expr_reads(expr):
                k = var_key(r); ver = getattr(r, "version", None)
                if (k, ver) in tainted:
                    hit.append((k, ver))
                elif (k, None) in tainted:
                    hit.append((k, None))
            if not hit:
                node = self._pointee_tainted(ssaf, expr, tainted)
                if node is not None:
                    hit.append(node)
            return hit

        def cons_return(ins: Any, reason: str) -> bool:
            nonlocal reached_return
            if "TAILCALL" in op_name(ins):
                if reached_return:
                    return False
                reached_return = True
                return True
            done = False
            for w in ssa_writes(ins):
                node = (var_key(w), getattr(w, "version", None))
                if taint_node(node, var_label(w), ins, reason, []):
                    done = True
            return done

        # The element-field key for a store dest / load src is purely STRUCTURAL
        # (depends only on the SSA def graph, not the taint set), but the fixpoint
        # revisits each instruction every iteration -- so memoize it per instruction
        # to avoid re-running the recursive address resolution N times (#420).
        elem_key_cache: dict = {}

        def elem_key(ins: Any, addr: Any):
            idx = getattr(ins, "instr_index", None)
            if idx is None:
                return self._elem_field_key(ssaf, addr)
            if idx not in elem_key_cache:
                elem_key_cache[idx] = self._elem_field_key(ssaf, addr)
            return elem_key_cache[idx]

        def apply_model(ins, params, model, mkey, name, *, site_taddr=None):
            """Apply a function model's sink-detection + taint propagation at a call
            site. Shared by the direct-call and resolved-indirect-external branches.

            Returns ``(changed, propagated_argidx)`` where ``propagated_argidx`` is
            the set of ``*arg:N`` indices the model propagated *into* — the caller
            uses it to bubble those up as out-params. This helper does NOT bubble
            out-params itself (so the resolved-external branch keeps its historical
            behavior of not creating out-params). ``site_taddr`` selects the dedup
            signature: ``(addr, argidx)`` for direct calls, ``(addr, argidx, taddr)``
            per resolved target so distinct indirect targets each record once."""
            changed = False
            propagated: set[int] = set()

            def apply_token(tok, parents):
                # Tail calls have no SSA result: a modeled return instead exits
                # this function. Only invoke this after the model's source hit.
                if tok in ("ret", "*ret") and "TAILCALL" in op_name(ins):
                    return cons_return(ins, f"return of {name or '?'} (model propagate)")
                return self._apply_to_token(
                    ssaf, ins, params, tok, taint_node, name or "?", parents=parents)
            addr = int(getattr(ins, "address", 0))

            def _note_unkeyed_store(dest_tok):
                # A modeled propagate targeted a memory destination (*arg:N) but
                # _apply_to_token could not key it -- _buffer_target failed to
                # correlate the pointer to a tracked buffer (a field-derived and/or
                # non-constant index). Record a coarse_memory_store frontier so the
                # tainted write is not dropped into a silent hole that would let the
                # claim gate emit a false all-clear (#562). Distinguished from the
                # benign "buffer already tainted" case by re-checking _buffer_target.
                if not (isinstance(dest_tok, str) and dest_tok.startswith("*arg:")):
                    return
                try:
                    di = int(dest_tok.split("arg:", 1)[1])
                except (ValueError, IndexError):
                    return
                if di >= len(params) or self._buffer_target(ssaf, params[di]) is not None:
                    return
                leaf = {
                    "kind": "coarse_memory_store",
                    "address": hex(int(getattr(ins, "address", 0))),
                    "dest_expr": str(params[di]),
                    "il_text": str(ins),
                    "detail": (
                        f"{name or '?'} propagates a tainted value into a buffer "
                        f"(arg{di}) whose pointer the engine could not correlate to a "
                        "tracked buffer (field-derived and/or non-constant index); "
                        "downstream reads of these bytes are not followed -- a "
                        "'no sinks reached' result past this point is not proof of safety"
                    ),
                }
                if leaf not in leaves:
                    leaves.append(leaf)

            def _sink_arg_reach(argidx: int, sink: dict[str, Any]) -> tuple[list, dict | None]:
                """``(hit_nodes, arity)`` for a modeled sink's arg.

                In range: the taint the call's own param carries. Out of range
                (#577: an ARM-Thumb `j_memcpy` typed `memcpy(int32_t)` drops the
                length register) the arg is recovered through its calling-
                convention register (the #433 reg bridge) and reported only when
                that value really carries taint, so a merely under-recovered but
                quiet sink does not cry wolf (#1001 item 1: it used to be an
                `arg_under_recovered` frontier LEAF, which kept a sink the engine
                SAW reached out of the verdict and `flows`).

                ``arity`` is None when BN recovered every arg the model declares
                (the checked arg, `tainted_args`, `len_arg`, `buf_arg`, the copy
                roles); otherwise it names them in ``dropped``. A register value
                is evidence of REACH only: no class is ever lowered from it (a
                bounded copy, a receive bound, ...), because the register bridge
                cannot stand behind what a dropped arg holds -- the class a
                dropped arg would decide stays the model's own until argument
                recovery is complete. Precise classification through dropped
                registers is a follow-up (#1001 review round 3)."""
                declared = [argidx, *(sink.get("tainted_args") or [])]
                declared += [sink[k] for k in ("len_arg", "buf_arg") if sink.get(k) is not None]
                declared += list(_model_copy_roles(model) or ())
                dropped = sorted({int(i) for i in declared
                                  if isinstance(i, int) and not isinstance(i, bool) and i >= len(params)})
                reg = None
                if argidx < len(params):
                    hit = arg_taint(params[argidx])
                else:
                    try:
                        reg, seeds = self._reaching_arg_seeds_via_reg(func, [ins], argidx)
                    except Exception:
                        return [], None
                    hit = []
                    for v, _site in seeds:
                        nodes = arg_taint(_RegisterOperand(v, reg))
                        if not nodes and self._var_buffer_tainted(ssaf, v, tainted, set(), 0):
                            # A pointer reaching a tainted buffer the store/load
                            # keys do not name (a PHI of allocations): cite it.
                            nodes = [(var_key(v), getattr(v, "version", None))]
                        if nodes:
                            hit = nodes
                            break
                if not hit or not dropped:
                    return hit, None
                taddr = site_taddr if site_taddr is not None else self._resolve_direct_target(ins)
                variadic = _variadic_determination(function_at(self.bv, taddr)) if taddr is not None else None
                inspect = f"inspect `bn proto get {hex(int(taddr))}` and the call's argument provenance" \
                    if taddr is not None else "resolve the call target and inspect its argument provenance"
                if variadic is True:
                    advice = ("the callee is already declared variadic; this is incomplete "
                              "call recovery, not evidence of a prototype defect. Redeclaring "
                              "the same prototype does not recover its arguments; " + inspect)
                elif variadic is False:
                    advice = ("the callee is determined nonvariadic; " + inspect
                              + ", and apply `bn proto set` only if independent evidence "
                              "demonstrates a declaration discrepancy, then re-run")
                else:
                    advice = ("the callee's variadic status is undetermined; " + inspect
                              + " before diagnosing a signature discrepancy or recovery limit")
                # `class_from` says what the finding's class rests on:
                # `recovered_args` -- the checked arg was recovered, so the
                # in-range classifiers ran on the recovered args and the dropped
                # ones were simply not used; `model` -- the checked arg itself
                # was dropped, so no classifier ran and the model's class stands.
                return hit, {"recovered": len(params), "dropped": dropped, "register": reg,
                             "callee_ref": hex(int(taddr)) if taddr is not None else None,
                             "callee_variadic": variadic, "recovery_advice": advice,
                             "class_from": "recovered_args" if argidx < len(params) else "model"}

            def format_gated_sink(sink: dict[str, Any]) -> bool:
                """#1041 item 4: a scanf-family `%s`/`%[` conversion with no
                width writes an input token of any length into its destination.
                Armed only by what the CONSTANT format says -- never guessed
                from a format the engine cannot read, which is disclosed instead
                -- and, when the model names `tainted_args` (sscanf's input
                string), only while that input is tainted. Without them the
                input is a stream or stdin and the sink is unconditional, like
                gets(): each named destination is injected as an unconditional
                root, so flows out of it are not attributed to --source."""
                fa, first = int(sink["format_arg"]), int(sink["first_dest"])
                prose = self._prose_name(mkey or name or "?")
                gate_hit: list = []
                gate_idx = None
                for gi in sink.get("tainted_args") or []:
                    if gi < len(params):
                        gate_hit = arg_taint(params[gi])
                        if gate_hit:
                            gate_idx = gi
                            break
                if sink.get("tainted_args") and not gate_hit:
                    return False
                if fa >= len(params):
                    return False
                usig = (addr, "format") if site_taddr is None else (addr, "format", site_taddr)
                if usig in recorded_sinks:
                    return False
                fmt = self._const_format_string(ssaf, params[fa])
                convs = unbounded_scanf_strings(fmt) if fmt is not None else None
                if convs is None:
                    # The conversions are unknown, so whether this call writes
                    # an unbounded token is unknown: a frontier the claim gate
                    # withholds on, never a silent "no sink" (#1050 review
                    # finding 4). Its conversions are not guessed.
                    recorded_sinks.add(usig)
                    why_not = ("is not a constant string in read-only data" if fmt is None
                               else "uses POSIX %n$ positions")
                    leaf = {
                        "kind": "format_unchecked",
                        "address": hex(addr),
                        "callee": str(mkey or name or "?"),
                        "format_arg": fa,
                        "reason": "positional" if fmt is not None else "not_constant",
                        "detail": (f"{prose}() at {hex(addr)}: its format (arg{fa}) {why_not}, "
                                   f"so its conversions could not be checked for a width-less "
                                   f"%s/%[ -- whether it writes an unbounded token is unknown"),
                    }
                    if leaf not in leaves:
                        leaves.append(leaf)
                    return False
                if not convs:
                    recorded_sinks.add(usig)
                    return False
                recorded_sinks.add(usig)
                dests = [{"conversion": spec, "dest_arg": first + k} for k, spec in convs]
                listed = ", ".join(f"{d['conversion']} into arg{d['dest_arg']}" for d in dests)
                eff = {**sink, "format_conversions": dests,
                       "detail": (f"width-less conversion(s) in the constant format write an "
                                  f"input token of unbounded length: {listed}")}
                if gate_idx is not None:
                    findings.append(_tag_unconditional_flow(
                        self._make_finding(ins, mkey or name, gate_idx, eff, gate_hit, why),
                        gate_hit))
                    return False
                hit_nodes = []
                grew = False
                for d in dests:
                    di = d["dest_arg"]
                    dest = (self._buffer_target(ssaf, params[di]) if di < len(params) else None)
                    if dest is not None:
                        node, lbl = (dest[0], None), dest[1]
                    else:
                        node = (("name", f"{mkey or name}_arg{di}@{hex(addr)}"), None)
                        lbl = var_label_of(node)
                    if taint_node(node, lbl, ins,
                                  f"{mkey or name}(): width-less {d['conversion']} reads an "
                                  f"unbounded token", []):
                        grew = True
                    unconditional_roots.add(node)
                    hit_nodes.append(node)
                add_assumption(
                    f"{prose}(): unconditional {sink.get('class') or '?'} sink at {hex(addr)} "
                    f"independently of --source ({listed}; the finding does not require any "
                    f"tracked taint to reach this call)")
                findings.append(_tag_unconditional_flow(
                    self._make_finding(ins, mkey or name, None, eff, hit_nodes, why), hit_nodes))
                return grew

            sink = model.get("sink")
            # opt-in sinks (e.g. file_write) stay silent unless their gate was
            # enabled for this run; still a "modeled" call, so no fallback noise.
            # The `gate` field (falling back to `class`) lets a sink keep an
            # accurate bug class -- recv/read report `overflow_len` -- while gating
            # under a distinct opt-in name `recv_overflow`, since always-on recv/read
            # length sinks are ~100% FP on the fill-loop idiom (#499).
            if sink is not None and sink.get("optional"):
                _gate = sink.get("gate") or sink.get("class")
                if _gate not in self._enabled_sink_classes:
                    sink = None
            if sink is not None and sink.get("format_arg") is not None:
                if format_gated_sink(sink):
                    changed = True
                sink = None
            if sink is not None:
                # #615/review F3: an unconditional sink is one whose arming fields
                # are BOTH empty -- no `tainted_args` and no `len_arg` -- independent
                # of `class`. This mirrors the catalog (`read_taint_models.py:
                # _arg_phrase`) and validator (`taint_models.py:142-143`), which
                # already treat that shape as always-unsafe regardless of the
                # sink's class name; gating on `class == "unbounded_input"` left
                # user models under other class names permanently unfireable, and
                # let a `class: "unbounded_input"` model WITH real tainted_args
                # double-arm (once here, once via the conditional loop below with
                # a different dedup key). It must be reported even when no tracked
                # taint reaches any argument, since the danger is the call itself.
                # Fires once per resolved target (review F4: `site_taddr`
                # distinguishes resolved-indirect candidates sharing one callsite
                # address), deduped like every other sink below.
                if not sink.get("tainted_args") and sink.get("len_arg") is None:
                    usig = ((addr, "unconditional") if site_taddr is None
                            else (addr, "unconditional", site_taddr))
                    if usig not in recorded_sinks:
                        recorded_sinks.add(usig)
                        # review F5: do NOT guess arg0 as the destination -- the
                        # same anti-pattern the len_arg/no-buf_arg branch below
                        # already forbids (audit D2, below). Prefer the model's
                        # declared `buf_arg`, then its `sources[].to == "*arg:N"`
                        # (gets declares `*arg:0`), and only fall back to an
                        # undeclared synthetic node (no real buffer taint
                        # injected) when neither is present.
                        _buf_idx = sink.get("buf_arg")
                        if _buf_idx is None:
                            _src_tos = [str(s.get("to")) for s in (model.get("sources") or [])
                                        if str(s.get("to", "")).startswith("*arg:")]
                            if _src_tos:
                                _buf_idx = _try_arg_index(_src_tos[0])
                        dest = (self._buffer_target(ssaf, params[_buf_idx])
                                if _buf_idx is not None and _buf_idx < len(params) else None)
                        if dest is not None:
                            node = (dest[0], None)
                            node_lbl = dest[1]
                        else:
                            _tag = _buf_idx if _buf_idx is not None else "undeclared"
                            node = (("name", f"{mkey or name or 'unbounded_input'}_arg{_tag}@{hex(addr)}"), None)
                            node_lbl = var_label_of(node)
                        if taint_node(node, node_lbl, ins,
                                      f"{mkey or name}(): unbounded read, always unsafe", []):
                            changed = True
                        unconditional_roots.add(node)
                        # P3: three distinct provenance shapes -- do not claim a
                        # buffer taint that did not happen. `dest is not None` is
                        # the only case where a real buffer node was tainted;
                        # a declared-but-unresolvable buf_arg (index known, target
                        # not resolved) falls back to a synthetic
                        # `<name>_arg{N}@<addr>` node no downstream read can match,
                        # same as the fully-undeclared case, just worded to keep
                        # the declared index visible for diagnosis.
                        if dest is not None:
                            _taint_phrase = f"taints arg{_buf_idx} "
                        elif _buf_idx is not None:
                            _taint_phrase = (
                                f"declares arg{_buf_idx} as its buffer but the target "
                                "could not be resolved, so no buffer taint was injected "
                                "(synthetic node only) ")
                        else:
                            _taint_phrase = "injects no buffer taint (no declared destination) "
                        add_assumption(
                            f"{self._prose_name(mkey or name)}(): unconditional {sink.get('class') or '?'} sink "
                            + _taint_phrase
                            + f"at {hex(addr)} independently of --source (always-unsafe API; "
                            "the finding does not require any tracked taint to reach this call)")
                        findings.append(_tag_unconditional_flow(
                            self._make_finding(ins, mkey or name, None, sink, [node], why), [node]))
                # #443: a bounded-write sink (wrapped recv/read) is armed by its
                # `len_arg` -- an attacker-controlled write length -- in addition to
                # any explicit `tainted_args`.
                _sink_args = list(sink.get("tainted_args") or [])
                if sink.get("len_arg") is not None and sink["len_arg"] not in _sink_args:
                    _sink_args.append(int(sink["len_arg"]))
                for argidx in _sink_args:
                    ht, arity = _sink_arg_reach(argidx, sink)
                    # Only the call's own params classify a sink. A checked arg
                    # BN dropped was read from its ABI register, which is evidence
                    # of reach, not of value: no class is lowered from it, so the
                    # model's class stands (#1001 review round 3).
                    classify = argidx < len(params)
                    if ht:
                        sig = (addr, argidx) if site_taddr is None else (addr, argidx, site_taddr)
                        if sig not in recorded_sinks:
                            recorded_sinks.add(sig)
                            eff_sink = sink
                            # Downgrade a tainted memcpy-family LENGTH from
                            # overflow to bounded when the copy provably fits
                            # the buffer the destination was allocated with in
                            # this function -- the exact dst=malloc(n);
                            # memcpy(dst,src,n) case AND the generalized
                            # dst=alloc(len+C); memcpy(dst+c,src,len+D) with
                            # c+D<=C, plus unmodeled single-arg allocator
                            # wrappers (#46 item 1, #229). The length is
                            # attacker-derived but the copy cannot overflow.
                            # Still reported, just relabeled, so it's not noise
                            # in the overflow set.
                            # #443: for a recv-style bounded-write sink the
                            # destination is `buf_arg` (not memcpy's arg0), so the
                            # provably-bounded downgrade checks the right buffer.
                            if argidx == sink.get("len_arg") and sink.get("buf_arg") is None:
                                # Armed by len_arg but NO buf_arg declared: do NOT
                                # guess arg0 as the destination -- arg0 may be a
                                # sibling buffer sized to the length while the real
                                # write dest is a different, smaller arg, which would
                                # wrongly downgrade a real overflow to bounded_len
                                # (audit D2). No declared dest => no bounded downgrade.
                                _bnd_reason = None
                            else:
                                # #980 review: the copy's destination is the arg
                                # its model copies INTO, not a fixed arg0 --
                                # bcopy(src, dst, n) writes arg1, and sizing its
                                # SOURCE as the destination relabelled a real
                                # overflow of dst bounded_len.
                                _roles = _model_copy_roles(model)
                                _dest_idx = (int(sink["buf_arg"])
                                             if (sink.get("buf_arg") is not None
                                                 and argidx == sink.get("len_arg"))
                                             else (_roles[1] if _roles else 0))
                                # #808: a declared len_arg+buf_arg pair IS a
                                # bounded-write declaration, so the provably-bounded
                                # relabel must not hinge on the model's class hint --
                                # one entry arms several args under ONE class, and a
                                # class describing a sibling arg (snprintf: the write
                                # length arg1 shares format_or_overflow with the
                                # format arg2) says nothing about this length. Still
                                # gated on argidx == len_arg, so a non-length arg of
                                # such an entry is never relabeled by another arg's
                                # length. Inert for every pre-existing model: the
                                # read/recv family reaches the same `_bnd_reason`
                                # through its `overflow_len` class already, and a
                                # `len_arg` without `buf_arg` took the branch above.
                                _len_pair = (sink.get("buf_arg") is not None
                                             and argidx == sink.get("len_arg"))
                                _bnd_reason = (self._bounded_copy_reason(ssaf, params, argidx, dest_idx=_dest_idx)
                                               if classify and (sink.get("class") == "overflow_len" or _len_pair)
                                               else None)
                            if _bnd_reason is not None:
                                eff_sink = {
                                    **sink,
                                    "class": "bounded_len",
                                    "detail": (sink.get("detail") or "")
                                    + f" (attacker-derived length, but {_bnd_reason})",
                                    "relabel": "destination sized by the copy length",
                                }
                                # #980: bounding the WRITE says nothing about the
                                # READ. Size the copy source and re-headline to
                                # `overread_len` when it is a fixed-size object
                                # the length is not provably within.
                                eff_sink = self._source_side_verdict(
                                    ssaf, params, argidx, mkey, model, eff_sink)
                            elif classify and sink.get("class") == "overflow_len":
                                # Length is the return of a modeled receive
                                # call bounded by a constant count arg
                                # (n = read(fd, buf, MAX); memcpy(dst, buf, n)):
                                # attacker-derived but provably <= MAX (#159).
                                _bnd = self._syscall_bound_for_length(ssaf, params[argidx])
                                if _bnd is not None:
                                    eff_sink = {
                                        **sink,
                                        "class": "bounded_len",
                                        "source_bound": hex(_bnd),
                                        "detail": (sink.get("detail") or "")
                                        + f" (length is a modeled receive return provably "
                                        f"bounded by {hex(_bnd)} -- bounded copy, not an "
                                        "unbounded overflow)",
                                    }
                            # Reserve overflow_* for buffer/length-operand
                            # taint: if the taint reaches this sink only
                            # through an array index/offset (`base + i*stride`)
                            # and the buffer itself is untainted, reclassify to
                            # the distinct, lower-confidence `tainted_index`
                            # (an OOB-access risk, not an unbounded/length
                            # overflow) rather than over-stating overflow_*
                            # (#163). Conservative: ambiguous cases stay
                            # overflow_* so a real overflow is never hidden.
                            # The arg is a copy-SOURCE pointer (vs a length
                            # scalar) when the model propagates a buffer FROM
                            # it (`*arg:<argidx>` in propagates.from) -- true
                            # for strcpy/strcpy_chk arg1, false for the memcpy/
                            # memcpy_chk length arg2. This drives the register-
                            # base index broadening per-arg, so a fortified
                            # length stays an overflow while a fortified source
                            # index downgrades.
                            _ptr_arg = eff_sink.get("class") == "overflow_unbounded" \
                                or argidx in _model_buffer_source_args(model)
                            if classify and eff_sink.get("class") in _OVERFLOW_INDEX_CLASSES \
                                    and self._sink_taint_is_index_only(
                                        ssaf, params[argidx], tainted, pointer_arg=_ptr_arg):
                                _prior = eff_sink.get("class")
                                eff_sink = {
                                    **eff_sink,
                                    "class": "tainted_index",
                                    "via": "index",
                                    "detail": (eff_sink.get("detail") or "")
                                    + " (taint reaches this sink through an array "
                                    "index/offset, not the copied buffer or the length "
                                    f"operand -- reclassified from {_prior}; an "
                                    "out-of-bounds access risk, not a plain "
                                    "unbounded/length overflow)",
                                }
                            # #307 FP-1: a still-`overflow_len` length that
                            # reads a reused/address-taken stack slot whose
                            # taint arrived version-agnostically (an out-param
                            # write through &slot) with a competing in-function
                            # writer is a path-ambiguous reaching def -- the
                            # engine can't stand behind the "attacker-controlled
                            # length" VERDICT (that needs control-dependence it
                            # lacks). Re-headline to the NEUTRAL `tainted_len`
                            # (propagation fact: a tainted value reaches this
                            # length; overflow-vs-bounded deferred). Nothing is
                            # hidden -- the same taint-reaches-arg flow stays in
                            # reached_sinks, so there is NO false-negative, only
                            # the unsound overflow label is dropped. Runs after
                            # the bounded_len / tainted_index reclassifications
                            # so those more-specific verdicts take precedence.
                            if classify and eff_sink.get("class") == "overflow_len" \
                                    and self._length_is_reused_aliased_slot(
                                        ssaf, instrs, params[argidx], tainted):
                                eff_sink = {
                                    **eff_sink,
                                    "class": "tainted_len",
                                    "via": "reused_aliased_slot",
                                    "detail": "a tainted value reaches this length "
                                    "argument via a reused/address-taken stack slot "
                                    "(written through &slot by an out-param call, with a "
                                    "competing in-function writer); the reaching definition "
                                    "is path-ambiguous, so overflow-vs-bounded is deferred -- "
                                    "corroborate with `taint backward`",
                                }
                            # #791 (residual of #499): the chunked-read
                            # residual idiom `n = total - cursor;
                            # read(buf + cursor, n)`. #499's closing note
                            # asked for this to be SUPPRESSED so
                            # recv_overflow became usable by default. It is
                            # deliberately NOT suppressed and NOT downgraded
                            # -- see `_residual_chunk_length` for why the
                            # shape cannot carry that conclusion: the
                            # invariant bounds the write by `total`, while
                            # safety needs `total <= sizeof(dest)`, and this
                            # engine can size neither a stack array nor the
                            # loop guard. #791's own repro is precisely the
                            # unsafe case (an attacker-controlled `cap` into
                            # a fixed stack buffer), so suppressing on shape
                            # would have cleared a real overflow.
                            #
                            # What IS honest is naming the open question, so
                            # a reader triaging an opt-in recv_overflow queue
                            # can see in the row itself that the length is a
                            # residual chunk and that the single thing to
                            # check is the destination's capacity against
                            # `total`. The class is untouched.
                            if classify and eff_sink.get("class") == "overflow_len":
                                _resid = self._residual_chunk_length(
                                    ssaf, params, argidx,
                                    int(eff_sink.get("buf_arg") or 0))
                                if _resid is not None:
                                    _total, _cursor = _resid
                                    eff_sink = {
                                        **eff_sink,
                                        "length_shape": "residual_chunk",
                                        "residual_total": _total,
                                        "residual_cursor": _cursor,
                                        "detail": (eff_sink.get("detail") or "")
                                        + f" (length is the residual chunk "
                                        f"`{_total} - {_cursor}` written at offset "
                                        f"`{_cursor}`, so the loop's cumulative extent "
                                        f"is bounded by `{_total}` -- NOT downgraded: "
                                        "bounded-by-total is only safe when the "
                                        "destination holds `total` bytes, which this "
                                        "engine cannot establish here (it sizes a buffer "
                                        "only from an allocating call, and proving "
                                        "cursor <= total needs the loop guard). Check "
                                        "the destination's capacity against the total. "
                                        "residual_total/residual_cursor are IL variable "
                                        "names, not source identifiers -- an optimised "
                                        "build reuses registers, so read them against "
                                        "the IL rather than expecting `cap`/`progress`)",
                                    }
                            # #980: a finding re-headlined away from its catalog
                            # class names that class, so `taint models` and the
                            # finding are visibly the same sink.
                            if eff_sink.get("class") != sink.get("class"):
                                eff_sink = {**eff_sink, "catalog_class": sink.get("class")}
                            if arity is not None:
                                # #1001 item 1: the model declares args BN's
                                # prototype dropped. Say so on the finding, and
                                # what its class rests on: the recovered args
                                # only (dropped ones not used), or -- when the
                                # checked arg itself was dropped -- the model's
                                # own class, unproven until argument recovery is complete.
                                _reg = arity["register"]
                                _fix = arity["recovery_advice"]
                                if arity["class_from"] == "model":
                                    _basis = (f"so the class is the model's own and unproven -- {_fix}")
                                else:
                                    _basis = ("so the class was decided from the recovered args only, "
                                              f"without the dropped ones -- {_fix}")
                                eff_sink = {
                                    **eff_sink,
                                    **({"via": "abi_register"} if _reg else {}),
                                    "arity_mismatch": arity,
                                    "detail": (eff_sink.get("detail") or "")
                                    + (f" (reached through ABI register {_reg}: " if _reg else " (")
                                    + f"Binary Ninja recovered only {arity['recovered']} parameter(s) "
                                    f"for this call and the model's arg(s) {arity['dropped']} were "
                                    f"dropped, {_basis})",
                                }
                            findings.append(_tag_unconditional_flow(
                                self._make_finding(ins, mkey or name, argidx, eff_sink, ht, why), ht))
            co = model.get("continuation")
            if co is not None:
                cidx = int(co.get("null_arg", 0))
                ckey = mkey or name or "?"
                if cidx < len(params):
                    if self._int_const(params[cidx]) == 0:
                        resets = clean_resets(ckey)
                        for (state_key, _site), (st_node, st_addr, st_ins) in list(
                                continuation_state.items()):
                            if state_key != ckey:
                                continue
                            # A clean non-NULL call resets libc's state -- but
                            # it kills the continuation only when the clean
                            # calls lie on EVERY path from the tainted call to
                            # this one. A clean call under `if (mode)` leaves the
                            # tainted line on the other path (round 3 P1), and a
                            # priming reset before a loop DOMINATES the NULL
                            # call yet the tainted call at the bottom of the
                            # body reaches it around the back-edge (round 4
                            # P2) -- so the test is reachability with the
                            # resets removed, not dominance. It runs with NO
                            # resets too (#932): every tainted site is state now,
                            # not just the latest in IL order, so a site that
                            # cannot reach this call at all must not arm it. A
                            # view without block information answers None and
                            # stays armed.
                            killed = self._reaches_avoiding(
                                st_ins, ins, [rins for _, rins in resets]) is False
                            if killed:
                                continue
                            for rule in model.get("propagates") or []:
                                if _try_arg_index(str(rule.get("from") or "")) != cidx:
                                    continue
                                if apply_token(rule.get("to"), [st_node]):
                                    changed = True
                            # Disclosed after convergence, from the FINAL taint:
                            # which calls are clean resets changes across passes,
                            # and a caveat written mid-fixpoint kept naming a
                            # site "clean" that a later pass tainted (round 1
                            # finding 7).
                            continuations[(ins_key(ins), state_key, _site)] = (
                                name or mkey, ckey, addr, st_addr, st_ins, ins)
                    else:
                        hit = self._token_hit_node(ssaf, params, f"*arg:{cidx}", tainted)
                        state_key = (ckey, int(getattr(ins, "instr_index", addr)))
                        if hit is not None and state_key not in continuation_state:
                            continuation_state[state_key] = (hit, addr, ins)
                            changed = True
            for rule in model.get("propagates") or []:
                to = rule.get("to")
                frm = rule.get("from")
                hit = self._token_hit_node(ssaf, params, frm, tainted)
                applied = False
                if hit is not None:
                    applied = apply_token(to, [hit])
                    if applied:
                        changed = True
                    else:
                        _note_unkeyed_store(to)
                    # #804 asked to gate this bookkeeping on the destination being
                    # KEYABLE (_buffer_target non-None), reading an unkeyable
                    # *arg:N as "no write happened". Deliberately NOT done: the
                    # premise does not hold, and the change is a false negative.
                    # `propagated` does not become an out-param on its own -- the
                    # two consumers below both pass it through
                    # `_resolve_to_param_index`, so the ONLY unkeyable stores that
                    # survive are the ones whose destination is this function's own
                    # pointer PARAMETER. A bare pointer parameter has no
                    # intra-function buffer identity by construction (that is
                    # exactly why _buffer_target returns None for it), yet the
                    # model states the callee writes through it, and "a write
                    # through my pointer parameter" is the definition of an
                    # out-param -- the interprocedural keying is
                    # `_resolve_to_param_index`, not `_buffer_target`. Gating here
                    # silently deleted the `fill_len(&slot, buf); memcpy(_,_, slot)`
                    # chain (test_forward_reused_aliased_length_neutralized_to_
                    # tainted_len, a reduced real receive-handler shape): the sink
                    # vanished from reached_sinks entirely. Over-tainting a
                    # caller's buffer costs a reviewable finding; under-tainting
                    # loses the bug.
                    if to and to.startswith("*arg:"):
                        k = int(to.split("arg:", 1)[1])
                        if k < len(params):
                            propagated.add(k)
                    # Visibility for a tainted copy *source* (#44): the buffer a
                    # source operand (arg N>=1) points at is copied into the
                    # destination buffer (arg 0). The flow is propagated -- so a
                    # downstream sink on the destination IS reachable forward --
                    # but the copy itself is not a sink, so without this note a
                    # source-seed that lands here would report a bare "no sinks
                    # reached" even though backward/trace both reach it. Record the
                    # src-side copy so the three tools agree. Deduped per site.
                    # Gate on whether this rule propagates from tainted input into a
                    # KEYABLE destination -- NOT on `applied` (taint_node() returning
                    # a NEW lattice entry). A keyable dest already tainted by an
                    # earlier predecessor makes taint_node()/`applied` return False
                    # even though this copy from tainted input truly occurred, so
                    # gating on `applied` suppresses a real copy note -- a false
                    # negative, which is the worst outcome this engine can produce.
                    # The genuine "propagation did not occur" case is an UNKEYABLE
                    # dest (_buffer_target None), which must still stay silent --
                    # exactly what `dest_keyable` distinguishes.
                    dest_keyable = (to == "*arg:0" and len(params) > 0
                                    and self._buffer_target(ssaf, params[0]) is not None)
                    if dest_keyable and isinstance(frm, str) and frm.startswith("*arg:"):
                        try:
                            src_i = int(frm.split("arg:", 1)[1])
                        except Exception:
                            src_i = None
                        # Only when the source operand is NOT already a reported
                        # sink of this model: that is exactly the memcpy/memmove
                        # case #44 flags (source silently propagated, never
                        # reported). strcpy/strcpy_s already flag their source arg
                        # as a sink, so it shows up in reached_sinks and needs no
                        # note -- and the "not flagged" wording would be false.
                        sink_args = (model.get("sink") or {}).get("tainted_args") or []
                        if src_i is not None and src_i >= 1 and src_i not in sink_args:
                            add_assumption(
                                f"tainted buffer copied into the destination of "
                                f"{self._prose_name(name or '?')} at {hex(int(getattr(ins, 'address', 0)))} "
                                f"(arg{src_i} -> arg0); propagated to the destination, "
                                f"not itself flagged as a sink")
            # #863: a model may decline to claim its callee's real destinations
            # for a STRUCTURAL reason rather than an arity one -- vsscanf writes
            # through the va_list in arg2, which this engine has no resolver for,
            # so its entry deliberately models only the `*arg:0 -> ret` edge. The
            # #851 residual above cannot see that: it compares the call's arity
            # against a modeled `*arg:N` run, and such a model declares no
            # destination run at all. The result was an all-clear that read as "no
            # flow" when the truth was "the model never claimed those
            # destinations". Disclose it with a weak-seed marker so the gate
            # withholds the all-clear, exactly as the arity residual does.
            #
            # MUST-NOT-FIRE: gated on taint actually reaching this call's
            # arguments. `apply_model` runs for EVERY call to a modeled callee in
            # a walked function, so an ordinary vsscanf sitting in an otherwise
            # untainted body has no tainted param here and stays silent -- the
            # disclosure marks runs whose data could have landed in the unclaimed
            # destinations, not the mere presence of the callee.
            if (model or {}).get("destinations_unmodeled") and any(
                    arg_taint(p) for p in params):
                add_assumption(
                    f"destinations_unmodeled @ {hex(int(getattr(ins, 'address', 0)))}: "
                    f"tainted input reaches {self._prose_name(name or '?')}, whose destination "
                    "operands this model deliberately does not claim (they are "
                    "reached through a va_list the engine cannot resolve). Writes "
                    "into those destinations are not followed, so a 'no sinks "
                    "reached' result past this call is not proof of safety"
                )
            # #851, the sscanf half: the propagator run is unrolled exactly like the
            # source run, so an arity_capped model whose call carries more params
            # than its longest `*arg:N` destination leaves the residual destinations
            # unmodeled -- and an unmodeled destination the format writes is where a
            # tainted source would have landed. Disclose it and withhold the
            # all-clear, as the source path does. Deliberately not gated on the
            # per-rule hit or on `applied`: "no propagation happened" and "the run is
            # shorter than the call" are different facts, and only the second is
            # disclosed here.
            if (model or {}).get("arity_capped"):
                _prop_dests = [
                    _try_arg_index(str(rule.get("to") or ""))
                    for rule in (model.get("propagates") or [])
                    if str(rule.get("to", "")).startswith("*arg:")
                ]
                _prop_dests = [i for i in _prop_dests if i is not None]
                if _prop_dests and len(params) > max(_prop_dests) + 1:
                    _ca = hex(int(getattr(ins, "address", 0)))
                    _mx, _mn = max(_prop_dests), min(_prop_dests)
                    add_assumption(
                        f"scanf_arity_residual @ {_ca}: {self._prose_name(name or '?')} model "
                        f"covers *arg:{_mn}..{_mx} ({len(_prop_dests)} "
                        f"destination(s)); call has {len(params)} param(s) -- "
                        f"{len(params) - (_mx + 1)} potential destination(s) "
                        f"beyond the modeled run (arg:{_mx + 1}+) are not "
                        f"modeled. scanf_arity_residual"
                    )
            # variadic propagation: every tainted vararg (from first_index on) flows
            # into the dest buffer and is itself reportable. Uses the actual call
            # params, so no format-string parsing is needed; arg_taint already covers
            # both a tainted scalar (%d) and a pointer to a tainted buffer (%s).
            va = model.get("varargs")
            if va is not None:
                base = int(va.get("first_index", 0))
                vto = va.get("to")
                vsink = model.get("sink") if va.get("sink") else None
                eff_vsink = vsink
                upper = len(params)
                # When the format string (the arg right before the varargs, for the
                # printf family) is a compile-time constant, only the varargs its
                # conversion specifiers actually consume can affect the output or
                # the dest buffer. A tainted vararg past the last conversion is a
                # provably-dead flow and must not be reported or propagated (#45).
                # A non-constant / tainted format keeps the conservative all-args
                # behavior (still a real sink).
                if base >= 1 and (base - 1) < len(params):
                    fmt = self._const_format_string(ssaf, params[base - 1])
                    if fmt is not None:
                        consumed = _count_format_args(fmt)
                        if consumed is not None:  # None -> positional %n$; stay conservative (#69)
                            upper = min(upper, base + consumed)
                        # #477: a resolvable constant format means the format operand is
                        # NOT attacker-controlled, so a tainted vararg is formatting
                        # DATA, not format-string control. Re-headline off the
                        # format-string class -- keep the finding (tainted data into an
                        # unbounded printf-family write is a real overflow concern) but
                        # don't cry format-string-injection. A tainted format doesn't
                        # resolve to a constant (fmt is None) so it stays the stronger
                        # class; the genuine tainted-format case is the separate
                        # sink.tainted_args path.
                        eff_vsink = self._reclassify_constant_format_sink(eff_vsink, fmt)
                for i in range(max(base, 0), upper):
                    ht = arg_taint(params[i])
                    if not ht:
                        continue
                    if vto:
                        if apply_token(vto, [ht[0]]):
                            changed = True
                        else:
                            _note_unkeyed_store(vto)
                        # #804: left ungated for the same reason as the static
                        # propagates loop above -- see the rationale there.
                        if vto.startswith("*arg:"):
                            k = int(vto.split("arg:", 1)[1])
                            if k < len(params):
                                propagated.add(k)
                    if eff_vsink is not None:
                        # record under the real param index i so this shares the
                        # recorded_sinks / top-level dedup with any static sink.
                        sig = (addr, i) if site_taddr is None else (addr, i, site_taddr)
                        if sig not in recorded_sinks:
                            recorded_sinks.add(sig)
                            findings.append(_tag_unconditional_flow(
                                self._make_finding(ins, mkey or name, i, eff_vsink, ht, why), ht))
            return changed, propagated

        for _ in range(self.max_iters):
            changed = False
            for ins in instrs:
                opn = op_name(ins)

                # #206 flow-sensitive honesty: a tainted value that reaches an
                # unlifted/unimplemented instruction passes through an op BN's
                # lifter could not model, so propagation past it is a silent hole.
                # Reuse the SAME unlifted detection that produced the assumption
                # count (MLIL_UNIMPL* op OR an address in the LLIL/MLIL unlifted
                # set), and gate on taint actually reaching it -- a tainted SSA
                # read operand, OR a defined var already in the tainted flow. Emit
                # a blocking leaf (deduped by address) so the claim gate withholds
                # ONLY when taint truly reaches such an instruction; a function
                # that merely contains unlifted ops does not block the gate.
                if unimpl_addr_set or "UNIMPL" in opn:
                    _uaddr = int(getattr(ins, "address", 0))
                    if "UNIMPL" in opn or _uaddr in unimpl_addr_set:
                        _reaches = bool(read_taint(ins)) or any(
                            (var_key(w), getattr(w, "version", None)) in tainted
                            for w in ssa_writes(ins))
                        if _reaches:
                            _ul = {
                                "kind": "unlifted_instruction_reached",
                                "address": hex(_uaddr),
                                "il_text": str(ins),
                                "detail": (
                                    "a tainted value reaches an unlifted/unimplemented "
                                    "instruction here; BN's lifter could not model this "
                                    "op, so propagation through it is not tracked (a "
                                    "silent dataflow hole) -- a 'no sinks reached' result "
                                    "past this point is NOT proof of safety; inspect the "
                                    "instruction or seed downstream of it"),
                            }
                            if _ul not in leaves:
                                leaves.append(_ul)

                if opn == "MLIL_RET":
                    # Native RET operands live in src even when vars_read is empty.
                    srcs = getattr(ins, "src", None)
                    if not isinstance(srcs, (list, tuple)):
                        srcs = [] if srcs is None else [srcs]
                    if (read_taint(ins) or any(arg_taint(src) for src in srcs)
                            or self._return_buffer_tainted(ssaf, ins, tainted)):
                        reached_return = True
                    continue

                if self._is_call(ins):
                    target = self._resolve_direct_target(ins)
                    name = self._callee_name(target)
                    mkey, model = lookup_model(self.models, name)
                    params = self._call_params(ins)

                    # 1+2) model-driven sink detection + propagation (shared helper).
                    # `modeled_key` records WHICH model was applied HERE, so the
                    # descent below does not apply that model a second time. It is the
                    # DB key `lookup_model` matched -- NOT the raw callee spelling:
                    # one model key is reachable through several spellings
                    # (`memcpy@plt`, `_memcpy`, a mangled/demangled pair), and the
                    # descent re-looks-up the resolved body's own spelling, so a
                    # raw-name comparison would read one model as two and apply it
                    # twice at this one call site.
                    modeled_key = mkey if model is not None else None
                    if model is not None:
                        mchanged, propagated = apply_model(ins, params, model, mkey, name)
                        if mchanged:
                            changed = True
                        # out-param: a propagate that writes through one of THIS
                        # function's parameters taints a caller out-param.
                        for k in propagated:
                            pidx = self._resolve_to_param_index(func, ssaf, params[k])
                            if pidx is not None and pidx not in out_params:
                                out_params.add(pidx)
                                changed = True
                        # Deliberately NO `continue` here (#807). A model is an
                        # overLAY on a callee this binary may also have a body for,
                        # not a summary that replaces it: the walk used to stop at
                        # the call, so a model that described SOME of a wrapper's
                        # behavior hid the rest of its body -- and a partial overlay
                        # then answered `reached_sinks: []` with
                        # `safe_to_report_all_clear: true` while the UNmodeled run of
                        # the same query found the sink. Models only ever ADD taint
                        # (there is no suppression rule), so descending can only add
                        # coverage: the modeled run's SINK coverage stays a superset of
                        # the unmodeled one. (The leaves channel is not a superset --
                        # a modeled call skips the #381 unrecovered-arg frontier
                        # below.) The common model shape -- a libc/kernel name whose
                        # callee is an import -- has no body, so `descend_internal` is
                        # False for it and following its thunk is a cache hit.

                    # 3) resolve the target(s) and descend.
                    tainted_args = {i: arg_taint(p) for i, p in enumerate(params) if arg_taint(p)}
                    if not tainted_args:
                        # The MLIL call recovered no tainted arg -- but BN may have
                        # under-recovered the callee's arity (Thumb 0-arity miss /
                        # variadic), dropping a tainted arg-register value that never
                        # appears in `params`. Emit an honest frontier instead of a
                        # silent false-negative (#381). Unmodeled only: a model that
                        # armed this call owns that story itself (#577).
                        if model is None:
                            fr = self._unrecovered_arg_frontier(ins, tainted)
                            if fr is not None and fr not in leaves:
                                leaves.append(fr)
                        continue
                    # descend each callsite once per tainted-arg set (the fixpoint
                    # revisits instructions; without this, findings would duplicate)
                    call_key = (int(getattr(ins, "address", 0)), frozenset(tainted_args.keys()))
                    if call_key in processed_calls:
                        continue
                    processed_calls.add(call_key)

                    if target is not None:
                        candidates, via = [target], None
                    else:
                        mapped = self.resolve_map.get(int(getattr(ins, "address", 0)))
                        if mapped:
                            candidates = list(mapped)
                            via = "agent-map"
                        else:
                            candidates = self._call_targets_from_pvs(
                                getattr(getattr(ins, "dest", None), "possible_values", None))
                            via = "value-set" if candidates else None

                    if not candidates:
                        leaf = {
                            "kind": "indirect_call_unresolved",
                            "address": hex(int(getattr(ins, "address", 0))),
                            "dest_expr": str(getattr(ins, "dest", "")),
                            "il_text": str(ins),
                            "detail": "tainted value flows into an indirect call whose target VSA could not pin",
                        }
                        if leaf not in leaves:
                            leaves.append(leaf)
                        continue

                    ret_tainted = False
                    descend_outparams: set[int] = set()
                    descend_outparam_elems: set = set()  # (param_idx, field) #319b
                    resolved_names: list[str] = []
                    for taddr in candidates:
                        # function_at normalizes the Thumb low bit: a value-set
                        # target can be odd while the function lives at taddr&~1
                        # (#89 Problem B).
                        cfn = function_at(self.bv, taddr)
                        nm = self._callee_name(taddr)
                        # Canonical name for the "resolved via ... to:" assumption:
                        # the function/symbol AT the resolved address, captured
                        # BEFORE any thunk re-model reassigns `nm`/`descend_fn`. The
                        # descent branches below reported the followed target in one
                        # path and the veneer's own symbol in another, which flipped
                        # non-deterministically for a tail-call thunk (#290). Pinning
                        # the reported name to taddr makes it deterministic and means
                        # exactly what it says: the indirect call resolved to this
                        # address (whose symbol is `report_name`).
                        report_name = nm or hex(taddr)
                        mk, md = lookup_model(self.models, nm)
                        cfn_internal = self._is_internal(cfn)
                        # A .plt/veneer thunk must be resolved to its real target,
                        # then BOTH re-modeled and re-classified: the candidate is
                        # not a body, so follow a single-instruction thunk and re-run
                        # the model lookup + internal check on THAT target. Without
                        # re-modeling, a decorated veneer whose own name misses the
                        # model DB but that tail-calls a modeled sink (j_memcpy ->
                        # memcpy) would fall through to the conservative external tail
                        # and the sink be silently missed.
                        #
                        # Followed for a MODELED candidate too, on the direct AND the
                        # indirect path (#807). A model is an overlay on the callee,
                        # not a summary that replaces it, so a modeled veneer still
                        # hides whatever body it fronts. Gating this on the candidate's
                        # model left the indirect path -- an unmodeled call site whose
                        # resolved candidate carries a model -- stopping at the veneer
                        # and minting the same false all-clear #807 is about. Descent is
                        # independent of the model: following never costs a model.
                        rnm = rmk = rmd = None
                        descend_fn = cfn
                        descend_internal = cfn_internal
                        if cfn is not None and not cfn_internal:
                            resolved = self._follow_thunk_cached(cfn)
                            if resolved is not None and resolved is not cfn:
                                rnm = self._callee_name(int(getattr(resolved, "start", 0))) \
                                    or (str(resolved.name) if getattr(resolved, "name", None) else None)
                                rmk, rmd = lookup_model(self.models, rnm)
                                if self._is_internal(resolved):
                                    # An in-binary target is a BODY, and a body is
                                    # descended whether or not a model covers its name
                                    # (#807).
                                    descend_fn = resolved
                                    descend_internal = True
                        # Re-imported export: the call routed through a PLT/GOT
                        # stub to a symbol that is ALSO defined in this binary.
                        # Bridge to the local definition so taint descends into
                        # its body instead of stopping at a conservative external
                        # leaf (#46 item 3). Only when UNMODELED at both levels and
                        # not already internal, and only to a real in-binary
                        # definition -- a model anywhere in the chain already owns
                        # this call's story.
                        if md is None and rmd is None and not descend_internal:
                            local = self._local_definition_for(nm)
                            if local is not None:
                                descend_fn = local
                                descend_internal = True
                                add_assumption(
                                    f"bridged re-imported export {self._prose_name(nm)} to its in-binary "
                                    f"definition at {hex(int(getattr(local, 'start', 0)))}")
                        # EVERY distinct model in the chain applies at this call: the
                        # call site's own already did above, and here the resolved
                        # candidate's and the reached target's. A model is an overlay on
                        # the callee that runs, so letting either stand in for the
                        # other -- in either direction -- makes the modeled run report
                        # LESS than the unmodeled one, which is #807 at one more level:
                        # an armed sink declared on the callee the user actually called
                        # was discarded by the reached target's model, and an armed sink
                        # on the reached target was discarded by the candidate's.
                        # Same DB key is applied at most once (that is what makes this
                        # idempotent), and both share ONE reporting signature per call
                        # site, so two keys describing one callee still report one
                        # finding at this instruction instead of the same fact twice.
                        sig_taddr = taddr if target is None else None
                        applied_keys: set = set()
                        for _key, _mdl, _mname in ((mk, md, nm), (rmk, rmd, rnm)):
                            if _mdl is None or _key is None or _key == modeled_key \
                                    or _key in applied_keys:
                                continue
                            applied_keys.add(_key)
                            mchanged, _ = apply_model(ins, params, _mdl, _key,
                                                      _mname or nm, site_taddr=sig_taddr)
                            if mchanged:
                                changed = True
                        if descend_internal:
                            d = self._descend(ins, descend_fn, tainted_args, why, depth, max_depth,
                                              via=via, caller_instrs=instrs, caller_fn=func)
                            # Round-3 blocker: `_descend`'s rebuild strips per-finding
                            # taint-graph context, so a finding's provenance can only be
                            # re-checked HERE, against the CALLER's own why-chain, using
                            # the nodes that actually crossed into the callee -- one per
                            # tainted argument at the call site, not just the lowest-
                            # indexed one (a lower-indexed --source-derived arg must not
                            # mask a higher-indexed unconditionally-injected one). If ANY
                            # hand-off node is itself rooted at an unconditional injection
                            # (e.g. a caller-side gets()), every finding descended through
                            # it is equally unattributable to --source, no matter how deep
                            # the descent went (the merge at each intervening level
                            # re-applies this same check with its own hand-off nodes, so
                            # multi-level descent composes).
                            for f in d["findings"]:
                                findings.append(_tag_unconditional_flow(f, d.get("handoff_node") or []))
                            for lf in d["leaves"]:
                                if lf not in leaves:
                                    leaves.append(lf)
                            for a in d["assumptions"]:
                                add_assumption(a)
                            ret_tainted = ret_tainted or d["reached_return"]
                            descend_outparams |= set(d.get("out_params") or ())
                            descend_outparam_elems |= set(d.get("out_param_elems") or ())
                        elif md is None and rmd is None:
                            # Genuinely unmodeled: no model anywhere in the chain (the
                            # call site's own included), so the conservative external
                            # rule is the only information available. (A callee whose
                            # model WAS applied -- at any level -- needs nothing here:
                            # the model already describes its return and propagation.)
                            # #1001 item 3: an IN-BINARY body BN did not analyse
                            # (oversized, failed lift) lands here too, and is not
                            # "external". Name it as what it is, with the remedy,
                            # and record it as an unread body (#811). The phrase
                            # keeps "has no model": taint_result's gate keys on it.
                            _reason = self._not_internal_reason(descend_fn)
                            _unread = self._unanalysed_body_phrase(_reason)
                            _who = nm or hex(taddr)
                            _shown = self._prose_name(nm) if nm else hex(taddr)
                            _at = hex(int(getattr(descend_fn, "start", taddr)))
                            _thunk = None
                            if _reason == "thunk" and self._follow_thunk_cached(descend_fn) is None:
                                # In-binary, and not a stub onto an import either
                                # (that one resolves). `follow_thunk` only takes a
                                # one-instruction tail call, so a longer body BN
                                # still calls a thunk (an oversized function after
                                # `--force-analysis`) stops here even when its
                                # tail-call target resolves -- name every target.
                                _tails = self._thunk_tail_targets(descend_fn)
                                if _tails:
                                    _named = [(self._prose_function(t),
                                               hex(int(getattr(t, "start", 0)))) for t in _tails]
                                    _thunk = (f"in-binary {_shown} is a thunk Binary Ninja made of "
                                              + ("a tail call" if len(_named) == 1 else "tail calls")
                                              + " to " + ", ".join(f"{n} @ {a}" for n, a in _named)
                                              + ", which taint does not follow through a multi-"
                                              "instruction thunk",
                                              ("read the target with " if len(_named) == 1
                                               else "read the targets with ")
                                              + ", ".join(f"`bn decompile {a}`" for _n, a in _named))
                                else:
                                    _thunk = (f"in-binary {_shown} is a thunk whose tail-call "
                                              "target did not resolve",
                                              f"read it with `bn disasm {_at}`")
                            elif _unread is not None:
                                self._analysis_incomplete.add(str(_who))
                                _fix = (f"run `bn decompile {_at} --force-analysis` (may be slow) "
                                        "and re-run this taint query")
                            if _thunk is not None:
                                _caveat = (f"{_thunk[0]}; it has no model; return conservatively "
                                           f"tainted -- {_thunk[1]}")
                                _stop_desc = f"thunk ({_thunk[0]}; {_thunk[1]})"
                            elif _unread is not None:
                                _caveat = (f"in-binary {_shown} was not analysed ({_unread}) and has no "
                                           f"model; return conservatively tainted -- {_fix}")
                                _stop_desc = f"in-binary function ({_unread}; {_fix})"
                            else:
                                _caveat = f"external {_shown} has no model; return conservatively tainted"
                                _stop_desc = "external"
                            if self.unknown_call_policy != "stop":
                                ret_tainted = True
                                add_assumption(_caveat)
                            else:
                                # F4/#562: `stop` deliberately does NOT propagate taint
                                # through an unmodeled external -- but taint DID reach it,
                                # so record an honest frontier leaf. Without this the gate
                                # sees no `ret_tainted` and no "has no model" assumption and
                                # reports a false all-clear at exactly the callsite the mode
                                # was chosen to treat MORE conservatively (the escape signal
                                # must not be silenced along with the propagation).
                                _stop_leaf = {
                                    "kind": "unmodeled_callee",
                                    "address": hex(int(getattr(ins, "address", 0))),
                                    "callee": {"name": str(_who),
                                               "display_name": (_display_name(descend_fn)
                                                                if descend_fn is not None
                                                                else str(_who)),
                                               "address": hex(int(taddr))},
                                    "tainted_args": sorted(tainted_args.keys()),
                                    "note": ("--unknown-call stop: taint reached this unmodeled "
                                             + _stop_desc
                                             + " and was NOT propagated through it; flow past "
                                             "this call is not analyzed -- NOT an all-clear"),
                                }
                                if _stop_leaf not in leaves:
                                    leaves.append(_stop_leaf)
                        # EXACTLY ONE entry per resolved CANDIDATE, so the append
                        # sits after the branch chain rather than in each arm. Base
                        # appended through three mutually-exclusive branches; the two
                        # decisions are independent now (#807), so a target that is
                        # both modeled and descended would otherwise be named twice in
                        # the "resolved via ... to:" assumption (#290 pins its form).
                        resolved_names.append(report_name)

                    if ret_tainted and cons_return(ins, "return of resolved call propagates taint"):
                        changed = True
                    # callee wrote tainted data through pointer arg(s) -> taint the
                    # caller's buffer, and bubble up if that buffer is our own param
                    for j in descend_outparams:
                        if j < len(params):
                            pv = self._pointee_var(ssaf, params[j])
                            if pv is not None and taint_node((var_key(pv), None), var_label(pv), ins,
                                                             f"out-param {j} written by callee", []):
                                changed = True
                            pidx = self._resolve_to_param_index(func, ssaf, params[j])
                            if pidx is not None and pidx not in out_params:
                                out_params.add(pidx)
                                changed = True
                    # #319b: a callee that filled a descriptor-ARRAY element field of
                    # an array we passed in -> taint the caller-side ("elem", base,
                    # field) key (base resolved on OUR side, so an alloca'd VLA's
                    # alignment offset is folded in correctly), and bubble up if the
                    # array is itself our own param.
                    for (j, field) in descend_outparam_elems:
                        if j < len(params):
                            ba = self._addr_base_offset(ssaf, params[j])
                            if ba is not None:
                                ek = ("elem", ba[0], ba[1] + field)
                                if taint_node((ek, None), f"elem+{hex(ek[2])}", ins,
                                              f"descriptor-array element +{hex(field)} written "
                                              f"by callee (out-param {j})", []):
                                    changed = True
                                pidx = self._resolve_to_param_index(func, ssaf, params[j])
                                if pidx is not None and (pidx, ba[1] + field) not in out_param_elems:
                                    out_param_elems.add((pidx, ba[1] + field))
                                    changed = True
                    if via:
                        add_assumption(
                            f"indirect call at {hex(int(getattr(ins, 'address', 0)))} resolved via {via} to: "
                            f"{', '.join(self._prose_name(n) for n in resolved_names)}")
                    continue

                # typed-struct field store: MLIL_SET_VAR_FIELD / _ALIASED_FIELD write
                # a struct variable but expose NO vars_written (only .dest/.prev/.src),
                # so the generic rule misses them. Taint .dest (the new struct version)
                # when the stored value -- or any prior version of the struct -- is
                # tainted, keeping the whole descriptor tainted across field writes.
                if opn in ("MLIL_SET_VAR_ALIASED_FIELD", "MLIL_SET_VAR_FIELD"):
                    dest = getattr(ins, "dest", None)
                    if dest is not None and is_ssa_var(dest):
                        hits = arg_taint(getattr(ins, "src", None))
                        if not hits:
                            prev = getattr(ins, "prev", None)
                            if prev is not None:
                                pk = var_key(prev)
                                for n in tainted:
                                    if n[0] == pk:
                                        hits = [n]
                                        break
                        if hits:
                            node = (var_key(dest), getattr(dest, "version", None))
                            off = getattr(ins, "offset", 0)
                            if taint_node(node, var_label(dest), ins,
                                          f"tainted store into struct field +{off}", hits):
                                changed = True
                    continue

                # memory-SSA: a load that reads bytes a tainted store wrote (heap/
                # pointer aliasing the AddressOf rule misses). Additive + sound.
                if opn == "MLIL_SET_VAR_SSA":
                    src_expr = getattr(ins, "src", None)
                    if src_expr is not None and "LOAD" in op_name(src_expr):
                        msrc = self._load_tainted_via_memory(ssaf, src_expr, tainted)
                        reason = "loads value stored through a tainted pointer (mem-SSA)"
                        assume = "heap/pointer load resolved via memory-SSA store correlation"
                        if msrc is None:
                            # read()-filled globals are written by an opaque call, so
                            # mem-SSA can't correlate; match the load's absolute address
                            # against the coarse global taint location instead.
                            ga = self._global_addr(ssaf, getattr(src_expr, "src", None))
                            if ga is not None and (("global", ga), None) in tainted:
                                msrc = (("global", ga), None)
                                reason = "loads from a tainted global buffer (global_approx)"
                                assume = "global buffer modeled coarsely as one taint location (global_approx)"
                        if msrc is None and buffer_slots:
                            # #193 Part 1: this load reads a global/struct slot that a
                            # recv buffer pointer was loaded from -- so it re-loads the
                            # SAME (filled) buffer pointer. Correlate it, unless the slot
                            # was provably re-pointed between the recv and here.
                            sk = self._addr_base_offset(ssaf, getattr(src_expr, "src", None))
                            slot = buffer_slots.get(sk) if sk is not None else None
                            if slot is not None:
                                lo = slot.get("recv_idx")
                                hi = getattr(ins, "instr_index", None)
                                # Only a load AFTER the recv is a re-load; the recv's own
                                # buffer-pointer load (at/before recv_idx) is the seed, not
                                # a re-load, and must not self-correlate. Then require no
                                # intervening store re-pointed the slot.
                                is_reload = lo is not None and hi is not None and hi > lo
                                if is_reload and not self._store_to_slot_between(ssaf, instrs, sk, lo, hi):
                                    msrc = slot.get("recv_node") or (("bufslot", sk), None)
                                    reason = ("recv buffer pointer re-loaded from the same "
                                              "global/struct slot (slot-correlated, #193)")
                                    assume = (f"recv buffer pointer re-loaded via slot {sk}; "
                                              f"correlated forward into the re-load (#193)")
                                    correlated_slots.add(sk)
                        if msrc is None:
                            # #319b: a load of a descriptor-array element field
                            # `[base + idx*stride + field]` -- correlate it against a
                            # tainted ("elem", base, field) that the array's fill (in
                            # this function or carried from a callee) produced, so the
                            # parser-filled hdr[] reconnects to the per-element copy.
                            ek = elem_key(ins, getattr(src_expr, "src", None))
                            if ek is not None and (ek, None) in tainted:
                                msrc = (ek, None)
                                reason = ("loads a tainted descriptor-array element field "
                                          "(elem_approx)")
                                assume = ("descriptor-array element read correlated by "
                                          "(array base, field offset) (elem_approx)")
                        if msrc is not None:
                            for w in ssa_writes(ins):
                                node = (var_key(w), getattr(w, "version", None))
                                if taint_node(node, var_label(w), ins, reason, [msrc]):
                                    changed = True
                            add_assumption(assume)

                # generic value flow: any tainted value-read taints all writes
                reads = read_taint(ins)
                if reads:
                    label_reason = self._reason_for(opn)
                    for w in ssa_writes(ins):
                        node = (var_key(w), getattr(w, "version", None))
                        if taint_node(node, var_label(w), ins, label_reason, reads):
                            changed = True
                    # store-through-pointer: taint the pointee buffer (coarse)
                    if "STORE" in opn and not ssa_writes(ins):
                        dest = getattr(ins, "dest", None)
                        pv = self._pointee_var(ssaf, dest)
                        if pv is not None:
                            node = (var_key(pv), None)
                            if taint_node(node, var_label(pv), ins, "store into tainted buffer (memory_approx)", reads):
                                changed = True
                                add_assumption("memory aliasing modeled coarsely via pointer-base/AddressOf (memory_approx)")
                        elif self._heap_buffer_key(ssaf, dest) is not None:
                            # store through a HEAP pointer (an escape/encode helper
                            # writing a derived buffer): key the buffer by its alloc
                            # site so a later read of a pointer from the same alloc
                            # correlates (#319a). Coarse, version-agnostic.
                            hk = self._heap_buffer_key(ssaf, dest)
                            if taint_node((hk, None), f"heap_{hex(hk[1])}", ins,
                                          "store into tainted heap buffer (alloc-site, memory_approx)", reads):
                                changed = True
                                add_assumption("heap buffer aliasing modeled coarsely by allocation site (memory_approx)")
                        elif (ek := elem_key(ins, dest)) is not None:
                            # #319b: a tainted store to a descriptor-ARRAY element
                            # field `[base + idx*stride + field]` -- key it by
                            # (array-base, field) so a later read of THAT field at
                            # any index correlates (the parser-fills-output-array
                            # idiom that otherwise drops to a coarse frontier).
                            if taint_node((ek, None), f"elem+{hex(ek[2])}", ins,
                                          "store into tainted descriptor-array element "
                                          "field (elem_approx)", reads):
                                changed = True
                                add_assumption("descriptor-array element modeled coarsely "
                                               "by (array base, field offset) (elem_approx)")
                            # If the array base is one of THIS function's params, the
                            # caller owns the array -- carry the element-field taint
                            # across the return (parser fills caller's hdr[]).
                            parts = self._elem_addr_parts(ssaf, dest)
                            if parts is not None:
                                pidx = self._resolve_to_param_index(func, ssaf, parts[0])
                                if pidx is not None and (pidx, parts[1]) not in out_param_elems:
                                    out_param_elems.add((pidx, parts[1]))
                                    changed = True
                        else:
                            # tainted store through a pointer parameter -> out-param
                            pidx = self._resolve_to_param_index(func, ssaf, dest)
                            if pidx is not None and pidx not in out_params:
                                out_params.add(pidx)
                                changed = True
                            # T1: _pointee_var could not correlate the store
                            # destination to a tracked buffer (a struct-field
                            # pointer and/or a non-constant index), so the engine
                            # cannot follow downstream reads of these bytes. An
                            # out-param write is at best a coarse caller-side
                            # approximation (and at the analysis root it goes
                            # nowhere). Record a frontier leaf + assumption rather
                            # than dropping the flow silently -- an empty
                            # leaves+assumptions on a function that DID propagate
                            # taint into a store is exactly the honesty gap the
                            # soundness disclaimer promises not to produce.
                            saddr = hex(int(getattr(ins, "address", 0)))
                            leaf = {
                                "kind": "coarse_memory_store",
                                "address": saddr,
                                "dest_expr": str(dest),
                                "il_text": str(ins),
                                "detail": (
                                    "tainted value stored through a pointer the engine "
                                    "could not correlate to a tracked buffer (field-derived "
                                    "and/or non-constant index); downstream reads of these "
                                    "bytes are not followed"
                                ),
                            }
                            if leaf not in leaves:
                                leaves.append(leaf)

                # Address-of-tainted escape (#228): an assignment or store whose
                # VALUE is a POINTER to a tainted buffer (`stack_local = &buf`,
                # `*p = &buf`) is invisible to read_taint -- AddressOf targets are
                # not value-reads -- so a captured pointer to tainted data is
                # silently dropped, yielding a confident 0-leaf "no sinks reached"
                # even though the buffer escaped (the worst VR failure mode: an
                # agent reads it as proof of safety). When the generic flow above
                # found no tainted value-read, check whether the source is a
                # pointer to a tainted buffer and record a pointer_escape leaf so
                # the all-clear is never silent, best-effort propagating the taint.
                if not reads:
                    src_val = getattr(ins, "src", None)
                    esc = (self._pointee_tainted(ssaf, src_val, tainted)
                           if src_val is not None else None)
                    escaped = False
                    dest_desc = None
                    if esc is not None:
                        writes = ssa_writes(ins)
                        if writes and self._is_stack_write(writes):
                            # `stack_local = &buf`: the pointer is stashed into a
                            # stack descriptor/local (not a call-arg register), so a
                            # later `&descriptor` handed to a handler re-loads it out
                            # of the engine's sight. Tainting the dest also lets a
                            # single-var descriptor propagate when passed by address.
                            for w in writes:
                                node = (var_key(w), getattr(w, "version", None))
                                if taint_node(node, var_label(w), ins,
                                              "captured pointer to tainted buffer (pointer_escape)", [esc]):
                                    changed = True
                            escaped = True
                            dest_desc = f"stashed into stack local {var_label(writes[0])}"
                        elif "STORE" in opn:
                            # `*p = &buf`: pointer to a tainted buffer written into
                            # memory. Taint the pointee (the descriptor) coarsely so
                            # a later `&descriptor` propagates, and record the escape.
                            pv = self._pointee_var(ssaf, getattr(ins, "dest", None))
                            if pv is not None and taint_node((var_key(pv), None), var_label(pv), ins,
                                                             "descriptor holds pointer to tainted buffer (pointer_escape)", [esc]):
                                changed = True
                            escaped = True
                            dest_desc = "stored into memory through a pointer"
                    if escaped:
                        saddr = hex(int(getattr(ins, "address", 0)))
                        buf_lbl = node_label(esc, why)
                        leaf = {
                            "kind": "pointer_escape",
                            "address": saddr,
                            "buffer": buf_lbl,
                            "dest": dest_desc,
                            "il_text": str(ins),
                            "detail": (
                                "the address of a tainted buffer escapes here -- a "
                                "pointer to tainted data is captured into a local/"
                                "descriptor/memory the engine cannot correlate "
                                "downstream; flows that re-load this pointer (e.g. a "
                                "descriptor field passed by address to a handler) are "
                                "not followed, so a 'no sinks reached' result is NOT "
                                "proof of safety -- investigate the consumer"
                            ),
                        }
                        if leaf not in leaves:
                            leaves.append(leaf)
            if not changed:
                break
        else:
            # #579 truncation honesty: the fixpoint exhausted `max_iters` WITHOUT
            # a convergent `break` -- taint was still propagating on the final
            # pass, so coverage is incomplete. Flag it so the zero-sink gate
            # withholds the all-clear instead of reporting a silent (unsound)
            # "no sinks reached" indistinguishable from a converged run.
            self._truncated = True
            self._truncation_causes.add("fixpoint_exhausted")

        # The in-function continuations the fixpoint followed, written from the
        # converged taint, one caveat per (continuation, arming site). A clean
        # call is named as a partial reset only when it provably lies on a path
        # from the tainted call to this one -- a clean `strtok` that runs BEFORE
        # the tainted call is on no such path (#1050 review round 3). One whose
        # path relation is unknown (no block information, or successors BN
        # could not determine) is named neutrally.
        def on_path(src: Any, dsts: list[Any],
                    resets: list[tuple]) -> tuple[list[int], list[int]]:
            proven: list[int] = []
            unknown: list[int] = []
            for ra, rins in resets:
                into = self._reaches_avoiding(src, rins, [])
                out = [self._reaches_avoiding(rins, d, []) for d in dsts]
                if into is True and any(o is True for o in out):
                    proven.append(ra)
                elif into is not False and any(o is not False for o in out):
                    unknown.append(ra)
            return proven, unknown

        def reset_clause(name: str, proven: list[int], unknown: list[int], what: str) -> str:
            parts = []
            if proven:
                parts.append(f"a clean {name} @ {', '.join(hex(a) for a in proven)} resets "
                             f"the state on SOME path but not on every path {what}")
            if unknown:
                parts.append(f"a clean {name} @ {', '.join(hex(a) for a in unknown)} was not "
                             f"shown to lie on every path {what} (reachability unknown: no "
                             f"block information, or successors Binary Ninja could not "
                             f"determine)")
            return "; ".join(parts)

        for (_ck, _sk, _site), (cname, ckey, addr, st_addr, st_ins, cins) in sorted(
                continuations.items(), key=lambda kv: kv[0]):
            proven, unknown = on_path(st_ins, [cins], clean_resets(ckey))
            clause = reset_clause(self._prose_name(cname), proven, unknown,
                                  "from the tainted call to this one")
            partial = f"; {clause}, so the tainted line is kept" if clause else ""
            add_assumption(
                f"{self._prose_name(cname)}(NULL, ...) @ {hex(addr)} continues the "
                f"tokenizer state of the tainted {self._prose_name(cname)} call @ "
                f"{hex(st_addr)}, which reaches it on a path with no clean "
                f"reset; a continuation of a buffer tokenized in "
                f"ANOTHER function is not followed{partial}")

        # A tokenizer's static state still tainted when this function returns:
        # a `strtok(NULL, ..)` in a CALLER or a CALLEE continues it, and the
        # continuation model is per function, so that flow is not followed. The
        # walk KNOWS it armed the state and returned with it set -- silence here
        # was the M-15 hole one call deeper (PR-928 review P2). Disclosed under
        # a marker the zero-sink gate withholds the all-clear on.
        rets = [i for i in instrs if op_name(i) == "MLIL_RET"]
        for (ckey, _site), (_, st_addr, st_ins) in continuation_state.items():
            resets = clean_resets(ckey)
            # The state is clean at return only when the clean resets lie on
            # EVERY path from the tainted call to every return (the same
            # reachability test as the in-function continuation, round 4 P2);
            # a return reachable around them -- or a view without block
            # information -- leaves it tainted on a path.
            if rets and resets and all(
                    self._reaches_avoiding(st_ins, r, [rins for _, rins in resets]) is False
                    for r in rets):
                continue
            proven, unknown = on_path(st_ins, rets, resets)
            clause = reset_clause(self._prose_name(ckey), proven, unknown, "to a return")
            partial = f" ({clause})" if clause else ""
            # #1033 item 6: which function continues the tokenizer is not known
            # here, so the seed inside it is a template, not a command.
            reseed = command_template(
                f"bn taint forward -f FUNCTION --source {shlex.quote('ret:' + ckey)}")
            add_assumption(
                f"tokenizer_state_escapes @ {hex(st_addr)}: the static tokenizer "
                f"state of {self._prose_name(ckey)} is tainted by the {self._prose_name(ckey)} call @ {hex(st_addr)} when "
                f"{self._prose_function(func)} returns{partial}; a {self._prose_name(ckey)}(NULL, ...) continuation in "
                f"ANOTHER function (a caller or a callee) is not followed, so a 'no "
                f"sinks reached' result is NOT an all-clear -- seed the tokens inside "
                f"the function that continues the tokenizer ({reseed})")

        # #193 Part 1 honesty: for each registered recv-buffer slot that the fixpoint
        # did NOT correlate to a re-load, emit the deferred "may be missed" caveat --
        # the flow really wasn't followed. Slots that DID correlate already carry their
        # positive note (added at the re-load), so the misleading caveat is suppressed.
        for sk, slot in buffer_slots.items():
            if sk not in correlated_slots:
                add_assumption(slot["pending"])
                # #562: the caveat is assumption-only; also emit the deferred
                # blocking leaf so a zero-sink run whose only escape is this
                # uncorrelated indirect-pointer slot withholds the all-clear.
                pl = slot.get("pending_leaf")
                if pl is not None and pl not in leaves:
                    leaves.append(pl)

        # #559: for the top run, capture the seed-frontier facts a zero-result
        # query needs to explain WHY the frontier stopped -- how many SSA values
        # the seed produced and the last one taint reached. `why` preserves
        # insertion order, so the last record with a backing instruction is the
        # last propagated use (the param/var seed itself has instr=None).
        diag = None
        if top:
            last_use = None
            for node in reversed(list(why)):
                rec = why[node]
                ins = rec.get("instr")
                if ins is not None:
                    last_use = {"label": rec.get("label"),
                                "address": hex(int(getattr(ins, "address", 0))),
                                "reason": rec.get("reason")}
                    break
            diag = {"tainted_values": len(tainted), "last_use": last_use}

        return {"reached_return": reached_return, "out_params": frozenset(out_params),
                "out_param_elems": frozenset(out_param_elems),
                "findings": findings, "leaves": leaves, "assumptions": assumptions,
                "diag": diag}

    def _reason_for(self, opn: str) -> str:
        if "PHI" in opn:
            return "phi join of tainted versions"
        if "LOAD" in opn:
            return "load derived from tainted value (memory_approx)"
        for tok, txt in (("ADD", "arithmetic"), ("SUB", "arithmetic"), ("MUL", "arithmetic"),
                         ("AND", "arithmetic"), ("OR", "arithmetic"), ("XOR", "arithmetic"),
                         ("SX", "sign/zero extension"), ("ZX", "sign/zero extension"),
                         ("LOW_PART", "truncation")):
            if tok in opn:
                return f"{txt} of tainted operand"
        return "assignment/copy of tainted value"

    def _token_hit_node(self, ssaf: Any, params: list[Any], tok: str | None, tainted: set):
        """The tainted node satisfying an arg/pointee token, or None. Returning
        the node (not just a bool) lets propagation record provenance so a
        propagated buffer's path links back to the original source."""
        if not tok or not (tok.startswith("*arg:") or tok.startswith("arg:")):
            return None
        pointee = tok.startswith("*arg:")
        idx = int(tok.split("arg:", 1)[1])
        if idx >= len(params):
            return None
        expr = params[idx]
        if pointee:
            node = self._pointee_tainted(ssaf, expr, tainted)
            if node is not None:
                return node
        # a tainted scalar/pointer arg (incl. a tainted pointer parameter whose
        # pointee is tainted in our coarse model) satisfies the token
        for r in expr_reads(expr):
            ver = getattr(r, "version", None)
            if (var_key(r), ver) in tainted:
                return (var_key(r), ver)
            if (var_key(r), None) in tainted:
                return (var_key(r), None)
        return None

    def _apply_to_token(self, ssaf: Any, ins: Any, params: list[Any], tok: str | None,
                        taint_node, callee: str, parents: list | None = None) -> bool:
        parents = parents or []
        if not tok:
            return False
        if tok == "ret":
            done = False
            for w in ssa_writes(ins):
                node = (var_key(w), getattr(w, "version", None))
                if taint_node(node, var_label(w), ins, f"return of {callee} (model propagate)", parents):
                    done = True
            return done
        if tok == "*ret":
            # Grammar: '*ret' is the buffer the RETURNED POINTER points at -- not
            # the pointer's own value. Both channels are tainted: the returned
            # value (loads through the pointer, `arg_taint`'s value-read check --
            # the coarse memory approximation every pointer-return path relies
            # on), AND the buffer itself, keyed by THIS call site's alloc-site key
            # so a pointer re-derived from the call (a copy, an indexed cursor, a
            # PHI entry, or the same pointer handed to another model's '*arg:N')
            # re-resolves to it through `_heap_buffer_key`. Without the key the
            # promise was only ever met as "the pointer is tainted" (the old
            # `tok == "ret" or tok == "*ret"` collapse), so a store into that
            # buffer never correlated downstream unless the callee's NAME happened
            # to carry an allocator hint (#806). Keyed off the call address
            # unconditionally: the model is the authority that this site hands
            # back a buffer it wrote, unlike the name-hint path which has to guess.
            done = False
            hk = self._returned_buffer_key(ins)
            if taint_node((hk, None), f"heap_{hex(hk[1])}", ins,
                          f"buffer returned by {callee} (model '*ret', alloc-site memory_approx)",
                          parents):
                done = True
            for w in ssa_writes(ins):
                node = (var_key(w), getattr(w, "version", None))
                if taint_node(node, var_label(w), ins, f"return of {callee} (model propagate)", parents):
                    done = True
            return done
        if tok.startswith("*arg:") or tok.startswith("arg:"):
            pointee = tok.startswith("*arg:")
            idx = int(tok.split("arg:", 1)[1])
            if idx >= len(params):
                return False
            if pointee:
                bt = self._buffer_target(ssaf, params[idx])
                if bt is not None:
                    key, label = bt
                    return taint_node((key, None), label, ins,
                                      f"buffer written by {callee} (model propagate)", parents)
                return False
            done = False
            for r in expr_reads(params[idx]):
                node = (var_key(r), getattr(r, "version", None))
                if taint_node(node, var_label(r), ins, f"output of {callee} (model propagate)", parents):
                    done = True
            return done
        return False

    def _param_not_found_error(self, func: Any, idx: int) -> "TaintError":
        """Explain an unavailable named parameter without assuming a bad prototype."""
        try:
            n = len(list(getattr(func, "parameter_vars", []) or []))
            have = f" (BN recovered {n} parameter(s))"
        except Exception:
            have = ""
        variadic = _variadic_determination(func)
        if variadic is True:
            advice = (
                "the function is already declared variadic; param:N selects recovered "
                "named parameters, not the variadic tail. Tracking that tail is a "
                "varargs limitation; inspect how the function consumes its va_list "
                "and seed a recovered variable inside it")
        elif variadic is False:
            advice = (
                "the function is determined nonvariadic; inspect its prototype and "
                "argument provenance. Only if independent argument evidence demonstrates "
                "a declaration discrepancy, apply "
                f"`bn proto set {hex(int(func.start))} \"<prototype>\"` and re-run")
        else:
            advice = (
                "variadic status is undetermined; inspect the function's prototype "
                "and argument consumption before diagnosing a declaration discrepancy "
                "or a tracking limitation")
        return TaintError(
            f"parameter {idx} not found on {self._prose_function(func)}{have}: {advice}")

    def _seed_forward(self, func, ssaf, instrs, sources, taint_node, add_assumption,
                      buffer_slots=None, leaves=None) -> bool:
        if buffer_slots is None:
            buffer_slots = {}
        if leaves is None:
            leaves = []

        def add_leaf(leaf: dict[str, Any]) -> None:
            # A structured seed-honesty leaf (#562): dedupe so a re-seeded run
            # doesn't stack duplicates. Distinct from assumptions -- this lands
            # in result["leaves"] and feeds the frontier accounting/claim gate.
            if leaf and leaf not in leaves:
                leaves.append(leaf)

        seeded = False
        for src in sources:
            kind = src.get("kind")
            if kind == "param":
                idx = int(src["index"])
                pv = self._param_var(func, idx)
                if pv is None:
                    raise self._param_not_found_error(func, idx)
                if taint_node((var_key(pv), None), str(getattr(pv, "name", pv)), None,
                              f"source: parameter {idx}", []):
                    seeded = True
            elif kind == "var":
                v = self._resolve_var(func, src["selector"])
                if taint_node((var_key(v), None), str(getattr(v, "name", v)), None,
                              f"source: variable {src['selector']}", []):
                    seeded = True
            elif kind in ("ret", "arg"):
                callee = src["callee"]
                # For an arg:<callee>:<n> source, allow anchoring at an indirect
                # call resolved to a thin wrapper that forwards its arg n to callee
                # (#292); ret: has no such arg to forward.
                wrapper_arg = int(src["index"]) if kind == "arg" else None
                calls = self._find_callsites(instrs, callee, resolve_indirect=True,
                                             wrapper_arg=wrapper_arg)
                if not calls:
                    raise self._no_callsite_error(instrs, callee, func)
                self._note_indirect_anchors(calls, callee, add_assumption, wrapper_arg=wrapper_arg)
                # #306: recvmsg/recvmmsg write the received bytes into the
                # scatter-gather buffers at msghdr->msg_iov[i].iov_base, NOT the
                # msghdr pointer (arg 1) or fd (arg 0). Seeding the msghdr arg
                # therefore taints the header, not the payload, and reads as a
                # false all-clear. Nudge the user to seed the filled buffer var so
                # the silent miss becomes actionable.
                if kind == "arg" and (callee or "").split("@", 1)[0].lstrip("_") in _RECVMSG_FAMILY:
                    _fn = hex(int(func.start))
                    _var_seed = command_template(f"bn taint forward -f {_fn} --source var:VAR")
                    # #1050 review finding 5: offer `call:` only when the engine's
                    # own iovec resolver -- the one a call: run uses -- finds a
                    # buffer to seed here. A msghdr built elsewhere (a parameter)
                    # leaves call: with nothing to seed, and that command failed.
                    _, _rm = lookup_model(self.models, callee)
                    _hdr = next((int(t.rsplit(":", 1)[1])
                                 for t in (str(s.get("to") or "")
                                           for s in (_rm or {}).get("sources") or [])
                                 if t.startswith("*iovec:") and t.rsplit(":", 1)[1].isdigit()),
                                None)
                    # Exactly the call: path's own criterion for seeding an
                    # iovec buffer: a keyed buffer, or a pointer value to taint.
                    _resolves = _hdr is not None and any(
                        self._buffer_target(ssaf, b) is not None or expr_reads(b)
                        for c in calls for b in self._recvmsg_iov_buffers(ssaf, instrs, c, _hdr))
                    if _resolves:
                        _how = (f"Use `bn taint forward -f {_fn} --source "
                                f"{shlex.quote('call:' + str(callee))}`, which resolves the iovec "
                                f"buffer from the msg_iov/iov_base setup here, or seed the filled "
                                f"buffer directly ({_var_seed}).")
                    else:
                        _how = (f"The iovec is not built in this function, so call:{callee} "
                                f"cannot resolve its buffer either; seed the filled buffer "
                                f"directly ({_var_seed}), or run taint from the caller that "
                                f"builds the iovec.")
                    add_assumption(
                        f"{self._prose_name(callee)} writes the received bytes to the scatter-gather iov "
                        f"buffer(s) (msghdr->msg_iov[i].iov_base; for recvmmsg, per msgvec "
                        f"entry), not the header pointer this --source seeds; the payload "
                        f"taint is NOT followed from here. {_how}"
                    )
                    # #562: structured leaf so JSON `leaves` is non-empty and the
                    # claim gate withholds an all-clear. An arg:recvmsg:N seed
                    # ALWAYS mis-anchors (it seeds the msghdr*, never the payload),
                    # so emit unconditionally here -- dogfood: agents misread a
                    # bare assumptions-only + 0-sinks result as clean.
                    add_leaf(misanchored_recv_leaf(
                        callee=str(callee),
                        arg_index=int(src.get("index", 1)),
                        reason="msghdr_not_payload",
                        call_resolves_iovec=_resolves,
                    ))
                if kind == "ret":
                    # A ret: source on a function whose model also fills an
                    # output-pointer buffer would silently miss those bytes;
                    # point the user at call: instead of a false all-clear (#157).
                    # But if the user ALSO passed a sibling arg:<callee>:<n> or
                    # call:<callee> source for the SAME callee that actually
                    # seeds that buffer, the nudge is redundant and misleading,
                    # so suppress it.
                    _, _hm = lookup_model(self.models, callee)
                    _outs = [str(s.get("to")) for s in (_hm or {}).get("sources") or []
                             if str(s.get("to", "")).startswith("*arg:")]
                    # Indices of the callee's modeled output buffers (*arg:N).
                    _out_idxs = {i for i in (_try_arg_index(o) for o in _outs)
                                 if i is not None}
                    # A sibling covers the output ONLY if it seeds one: call:<callee>
                    # presets every modeled output, but arg:<callee>:N covers it just
                    # when N is a modeled *arg:N index. A non-output arg sibling
                    # (e.g. arg:read:0 while read fills *arg:1) leaves the buffer
                    # unseeded, so the nudge must still fire.
                    sibling_covers_output = any(
                        s is not src
                        and s.get("callee") == callee
                        and (s.get("kind") == "call"
                             or (s.get("kind") == "arg"
                                 and s.get("index") in _out_idxs))
                        for s in sources
                    )
                    if _outs and not sibling_covers_output:
                        add_assumption(
                            f"{self._prose_name(callee)} also writes tainted data to {', '.join(_outs)} per its "
                            f"model; --source ret:{callee} seeds only the return -- try "
                            f"--source call:{callee} to also taint the output buffer(s)"
                        )
                # #5: a per-callsite re-run seeds from exactly one call address;
                # the "seeded from all" conflation note becomes a per-callsite note.
                only = getattr(self, "_only_callsite_addr", None)
                if only is not None:
                    calls = [c for c in calls
                             if (int(getattr(c, "address", 0)) & ~1) == (only & ~1)]
                    if not calls:
                        continue  # this source has no callsite at the attributed address
                    add_assumption(f"seeded from {self._prose_name(callee)} callsite at {hex(only)} (per-source attribution)")
                elif len(calls) > 1:
                    add_assumption(f"{len(calls)} callsites of {self._prose_name(callee)}; seeded from all")
                self._seed_callsites += len(calls)   # #559 frontier diagnostics
                ret_seeded = False
                for c in calls:
                    if kind == "ret":
                        for w in ssa_writes(c):
                            if taint_node((var_key(w), getattr(w, "version", None)), var_label(w), c,
                                          f"source: return of {callee}", []):
                                seeded = True
                                ret_seeded = True
                    else:  # arg:<callee>:<n> -> the buffer that arg n points at
                        idx = int(src["index"])
                        params = self._call_params(c)
                        if idx < len(params):
                            bt = self._buffer_target(ssaf, params[idx])
                            if bt is not None:
                                key, label = bt
                                if taint_node((key, None), label, c,
                                              f"source: {callee} fills arg{idx} buffer", []):
                                    seeded = True
                            else:
                                ptr_seeded = False
                                for r in expr_reads(params[idx]):
                                    if taint_node((var_key(r), getattr(r, "version", None)), var_label(r), c,
                                                  f"source: {callee} arg{idx}", []):
                                        seeded = True
                                        ptr_seeded = True
                                # #562: a recv-family arg seed that tainted only the
                                # pointer value (buffer not keyed) is the arg:recv:1
                                # false-all-clear shape -- emit a structured leaf so
                                # the claim gate stops reading the empty result as
                                # clean. Distinct from recvmsg (handled above): here
                                # the buffer simply couldn't be anchored.
                                _rbase = (callee or "").split("@", 1)[0].lstrip("_")
                                # The "buffer_not_keyed" leaf only applies when the
                                # SEEDED arg is the output-buffer pointer (read/recv/
                                # recvfrom/pread buf=arg1, fread ptr=arg0). A non-buffer
                                # arg seed (e.g. arg:read:0 = fd) is a different mistake
                                # and must NOT be mislabeled as a buffer that couldn't be
                                # keyed (#562 LOW) -- gate on the buffer arg index.
                                _buf_arg = {
                                    "recv": 1, "recvfrom": 1, "read": 1,
                                    "pread": 1, "fread": 0,
                                }.get(_rbase)
                                _recv_covered = bool(ptr_seeded and idx == _buf_arg)
                                # The buffer couldn't be anchored to a stack var or
                                # writable global. If the pointer is itself loaded
                                # from a global/struct slot, register the slot so a
                                # later re-load correlates forward (#193 Part 1); the
                                # helper falls back to today's honest caveat when the
                                # slot can't be named, and (#562) emits a blocking leaf
                                # on the uncorrelated sub-path unless the recv-buffer
                                # leaf below already covers this (callee, idx).
                                self._register_indirect_buffer_slot(
                                    ssaf, params[idx], c, callee, idx, buffer_slots,
                                    add_assumption, add_leaf=add_leaf,
                                    recv_buffer_covered=_recv_covered)
                                if _recv_covered:
                                    add_leaf(misanchored_recv_leaf(
                                        callee=str(callee),
                                        arg_index=idx,
                                        address=hex(int(getattr(c, "address", 0))),
                                        reason="buffer_not_keyed",
                                    ))
                if kind == "arg":
                    # #1041 item 1 gate (#1050 review rounds 4-5): this arg is a
                    # buffer the model says the call fills; the gate checks only
                    # unresolved indirect calls in THIS function, not syscalls.
                    _, _fm = lookup_model(self.models, callee)
                    _idx = int(src["index"])
                    if any(str(s.get("to") or "") in (f"*arg:{_idx}", f"*iovec:{_idx}")
                           for s in (_fm or {}).get("sources") or []):
                        self._buffer_seeded = True
                        self._unchecked_indirect_calls(instrs, calls, callee, add_leaf)
                if kind == "ret" and not ret_seeded:
                    # T3: callsites of `callee` exist but NONE consume its return
                    # value (a void or discarded return), so a ret: source has
                    # nothing to seed. This is a well-formed locator, NOT the
                    # "check --source locator" misdiagnosis the generic
                    # not-seeded failure would produce -- name the real cause.
                    raise TaintError(
                        f"{callee} return value is not consumed at any of its "
                        f"{len(calls)} callsite(s) in {func.name} (void or discarded "
                        f"return); a ret: source has nothing to seed -- use arg:<n> "
                        f"for an output-pointer argument, or seed at a callsite that "
                        f"uses the return value"
                    )
            elif kind == "call":
                # Preset: seed every output the callee's taint model declares --
                # the return value AND each output-pointer buffer (*arg:N) -- so a
                # receive/fill API like read/recv/recvfrom taints the buffer it
                # writes, not just (or instead of) its return value (#157).
                callee = src["callee"]
                calls = self._find_callsites(instrs, callee, resolve_indirect=True)
                if not calls:
                    raise self._no_callsite_error(instrs, callee, func)
                self._note_indirect_anchors(calls, callee, add_assumption)
                only = getattr(self, "_only_callsite_addr", None)
                if only is not None:
                    calls = [c for c in calls
                             if (int(getattr(c, "address", 0)) & ~1) == (only & ~1)]
                    if not calls:
                        continue
                    add_assumption(f"seeded from {self._prose_name(callee)} callsite at {hex(only)} (per-source attribution)")
                elif len(calls) > 1:
                    add_assumption(f"{len(calls)} callsites of {self._prose_name(callee)}; seeded from all")
                self._seed_callsites += len(calls)   # #559 frontier diagnostics
                _, model = lookup_model(self.models, callee)
                src_defs = (model or {}).get("sources") or []
                if not src_defs:
                    raise TaintError(
                        f"call:{callee} has no taint-model sources to seed (the model "
                        f"declares no ret/*arg:N output); use ret:{callee} or "
                        f"arg:{callee}:<n> explicitly"
                    )
                for c in calls:
                    params = self._call_params(c)
                    dest_end = self._format_destination_end(ssaf, model, params)
                    for sd in src_defs:
                        to = str(sd.get("to") or "")
                        if to == "ret" or to == "*ret":
                            # 'ret' seeds the returned VALUE. '*ret' is the buffer
                            # the returned pointer points at: key it at THIS call
                            # site (the model is the authority that the call hands
                            # back a buffer it wrote) in addition to the value, so
                            # a pointer re-derived from the call correlates
                            # through _heap_buffer_key (#806).
                            if to == "*ret":
                                hk = self._returned_buffer_key(c)
                                if taint_node((hk, None), f"heap_{hex(hk[1])}", c,
                                              f"source: {callee} returns a buffer "
                                              f"(call: preset, model '*ret')", []):
                                    seeded = True
                            for w in ssa_writes(c):
                                if taint_node((var_key(w), getattr(w, "version", None)), var_label(w), c,
                                              f"source: return of {callee} (call: preset)", []):
                                    seeded = True
                        elif to.startswith("*iovec:"):
                            # recvmsg/recvmmsg (#306): the payload lands in the
                            # iovec buffer(s), two pointer hops from the msghdr arg.
                            try:
                                hdr_idx = int(to.split(":")[-1])
                            except (TypeError, ValueError):
                                hdr_idx = -1
                            bufs = self._recvmsg_iov_buffers(ssaf, instrs, c, hdr_idx)
                            iov_seeded = False
                            _out_param_seen: set[int] = set()
                            for buf in bufs:
                                bt = self._buffer_target(ssaf, buf)
                                if bt is not None:
                                    key, label = bt
                                    if taint_node((key, None), label, c,
                                                  f"source: {callee} fills msg_iov[i].iov_base buffer "
                                                  f"(call: preset)", []):
                                        seeded = True
                                        iov_seeded = True
                                else:
                                    for r in expr_reads(buf):
                                        if taint_node((var_key(r), getattr(r, "version", None)),
                                                      var_label(r), c,
                                                      f"source: {callee} iovec buffer (call: preset)", []):
                                            seeded = True
                                            iov_seeded = True
                                # #452: the iovec buffer is this function's PARAMETER --
                                # a receive helper whose CALLER owns the destination
                                # (`recv_body(fd, dst, len)` builds a stack iovec around
                                # `dst`). The payload lands in the caller's buffer, which
                                # nothing here consumes, so without disclosure the helper
                                # reads as a bare "no taint reached". Name the out-param so
                                # an agent re-runs taint from the caller.
                                pidx = self._resolve_to_param_index(func, ssaf, buf)
                                if pidx is not None and pidx not in _out_param_seen:
                                    _out_param_seen.add(pidx)
                                    _ca = hex(int(getattr(c, "address", 0)))
                                    add_assumption(
                                        f"recvmsg_out_param @ {_ca}: {self._prose_name(callee)} fills the "
                                        f"caller-provided buffer passed as param:{pidx} "
                                        f"(msg_iov[i].iov_base is this function's parameter). "
                                        f"The received payload lands in the CALLER's buffer, "
                                        f"not a local here -- re-run taint from the caller to "
                                        f"follow it into the parser, or seed the caller's "
                                        f"buffer directly (#452).")
                            # Honesty backstop: a resolved iovec that seeds NOTHING (an
                            # entry whose iov_base is a constant / unrecovered expr, e.g.
                            # the reaching store was mis-picked or zero-initialized) must
                            # still nudge -- otherwise the recv path reads as a clean
                            # all-clear, the worst failure mode. Covers both the
                            # no-buffers and the buffers-but-nothing-seeded cases.
                            if not iov_seeded:
                                _ca = hex(int(getattr(c, "address", 0)))
                                add_assumption(
                                    f"recvmsg_iovec_unresolved @ {_ca}: {self._prose_name(callee)} fills the "
                                    f"scatter-gather buffer(s) at msghdr->msg_iov[i].iov_base, but "
                                    f"the iovec setup (msg_iov -> iov_base) at this callsite could "
                                    f"not be statically resolved -- likely a dynamically-built "
                                    f"iovec or a receive-helper whose caller passes the out-buffer "
                                    f"(#452). The payload taint is NOT followed from here; seed the "
                                    f"filled buffer directly "
                                    f"({command_template(f'bn taint forward -f {hex(int(func.start))} --source var:VAR')}) "
                                    f"or, for a helper, seed the buffer the caller passes in.")
                        elif to.startswith("*arg:"):
                            idx = _try_arg_index(to)
                            if idx is not None and idx < len(params) \
                                    and (dest_end is None or idx < dest_end):
                                bt = self._buffer_target(ssaf, params[idx])
                                if bt is not None:
                                    key, label = bt
                                    if taint_node((key, None), label, c,
                                                  f"source: {callee} fills arg{idx} buffer (call: preset)", []):
                                        seeded = True
                                else:
                                    for r in expr_reads(params[idx]):
                                        if taint_node((var_key(r), getattr(r, "version", None)), var_label(r), c,
                                                      f"source: {callee} arg{idx} (call: preset)", []):
                                            seeded = True
                                    # #562: no recv-buffer leaf fires on the call:
                                    # preset path. The indirect-slot helper owns an
                                    # indirect-load pointer (its slot may correlate
                                    # forward, #193, so it decides post-fixpoint);
                                    # every other unkeyable destination -- e.g.
                                    # `c->buf + have`, a field load plus an offset --
                                    # left only the pointer seeded, so disclose it as
                                    # the arg: path does (#1041 item 1). A constant
                                    # NULL destination receives nothing.
                                    owned = self._register_indirect_buffer_slot(
                                        ssaf, params[idx], c, callee, idx, buffer_slots,
                                        add_assumption, add_leaf=add_leaf,
                                        recv_buffer_covered=False)
                                    if not owned and self._int_const(params[idx]) != 0:
                                        add_leaf(misanchored_recv_leaf(
                                            callee=str(callee), arg_index=idx,
                                            address=hex(int(getattr(c, "address", 0))),
                                            reason="buffer_not_keyed", seed="call"))
                        elif to.startswith("arg:"):
                            idx = _try_arg_index(to)
                            if idx is not None and idx < len(params):
                                for r in expr_reads(params[idx]):
                                    if taint_node((var_key(r), getattr(r, "version", None)), var_label(r), c,
                                                  f"source: {callee} arg{idx} value (call: preset)", []):
                                        seeded = True
                    # #851: when the model is `arity_capped`, the unrolled
                    # *arg:N destination run may be shorter than the actual
                    # call's destination count. If the call has more params than
                    # max(modeled *arg:N) + 1, the residual destinations are
                    # not seeded and the all-clear is not safe to report.
                    if (model or {}).get("arity_capped"):
                        _modeled_idxs = [
                            _try_arg_index(str(sd.get("to") or ""))
                            for sd in src_defs
                            if str(sd.get("to", "")).startswith("*arg:")
                        ]
                        _modeled_idxs = [i for i in _modeled_idxs if i is not None]
                        if _modeled_idxs and len(params) > max(_modeled_idxs) + 1:
                            _ca = hex(int(getattr(c, "address", 0)))
                            _mx = max(_modeled_idxs)
                            _mn = min(_modeled_idxs)
                            _n_extra = len(params) - (_mx + 1)
                            add_assumption(
                                f"scanf_arity_residual @ {_ca}: {self._prose_name(callee)} model "
                                f"covers *arg:{_mn}..{_mx} "
                                f"({len(_modeled_idxs)} destination(s)); "
                                f"call has {len(params)} param(s) -- "
                                f"{_n_extra} potential destination(s) beyond "
                                f"the modeled run (arg:{_mx + 1}+) are not "
                                f"seeded. scanf_arity_residual"
                            )
                # #1041 item 1 gate (#1050 review rounds 4-5): the model fills a
                # buffer, so an unresolved indirect call in THIS function that
                # can run after the receive is an unchecked consumer of it.
                if any(str(sd.get("to") or "").startswith(("*arg:", "*iovec:"))
                       for sd in src_defs):
                    self._buffer_seeded = True
                    self._unchecked_indirect_calls(instrs, calls, callee, add_leaf)
            else:
                raise TaintError(f"unknown source kind: {kind}")
        return seeded

    # printf-family format-implying sink class -> the overflow class that remains
    # once the format operand is proven a compile-time constant (#477). The dest
    # buffer write is still unbounded; it is simply not a format-string bug.
    _CONSTANT_FORMAT_RECLASS = {
        "format_or_overflow": "overflow_unbounded",
        "fortified_format": "fortified_overflow",
    }

    def _reclassify_constant_format_sink(self, sink, fmt):
        """#477: when a printf-family sink's format operand is a resolved constant,
        a tainted vararg is formatting DATA, not format-string control. Downgrade the
        finding's class off the format-string class to its overflow counterpart and
        record the concrete constant format, so the headline is honest. Leaves
        non-format-implying sinks (and any class we don't map) untouched."""
        if sink is None:
            return sink
        new_cls = self._CONSTANT_FORMAT_RECLASS.get(sink.get("class"))
        if new_cls is None:
            return sink
        return {
            **sink,
            "class": new_cls,
            "format_constant": fmt,
            "detail": (
                f"tainted formatting data with a constant format {fmt!r} -- NOT a "
                "format-string bug; the concern is the unbounded write into the "
                "destination buffer"
            ),
        }

    def _make_finding(self, ins, callee, argidx, sink, hit_nodes, why) -> dict[str, Any]:
        path = self._reconstruct_path(hit_nodes[0], why)
        # #615 review F1: argidx is None for an unconditional sink (no argument
        # taint required) -- the reason must not claim an arg-taint flow that
        # never happened.
        reason = (f"{callee}(): unconditional sink reached (no argument taint required)"
                  if argidx is None else f"tainted arg{argidx} reaches {callee}")
        _arity = sink.get("arity_mismatch")
        if sink.get("via") == "abi_register" and isinstance(_arity, dict):
            reason += (f" through ABI register {_arity.get('register')} "
                       f"(arity mismatch: BN recovered {_arity.get('recovered')} param(s))")
        tainted_labels = [node_label(n, why) for n in hit_nodes]
        # Round-3 F1 residual: an unconditional sink's hit node is taint_node()'d
        # AT this exact call with empty parents (the call itself IS the
        # injection site), so `_reconstruct_path` above already renders one path
        # entry for `ins`. Appending a second entry for the same instruction
        # below double-counts a zero-hop flow (metrics.steps == 2 for a finding
        # with no propagation hops at all). Collapse into the existing entry
        # instead of duplicating it; a real (non-zero-hop) finding's last
        # reconstructed step is an earlier instruction, so this never fires for
        # those and multi-hop paths render unchanged.
        last = path[-1] if path else None
        if (last is not None and last.get("il_index") == int(getattr(ins, "instr_index", -1))
                and last.get("address") == hex(int(getattr(ins, "address", 0)))):
            last["reason"] = reason
            last["tainted"] = tainted_labels
            last["callee"] = callee
        else:
            path.append(_instr_dict(ins, reason=reason, tainted=tainted_labels, callee=callee))
        return {
            "sink": {
                "callee": callee,
                # #1001 item 4: `callee` is the MODEL key, which drops a C++
                # symbol's leading underscore; the real symbol and its demangled
                # display name ride beside it (always present, like #978's).
                "callee_symbol": self._symbol_names(callee)[0],
                "callee_display_name": self._display_for_name(callee),
                "address": hex(int(getattr(ins, "address", 0))),
                "tainted_arg_index": argidx,
                "class": sink.get("class"),
                "detail": sink.get("detail"),
                **({"format_constant": sink["format_constant"]} if sink.get("format_constant") is not None else {}),
                **({"format_conversions": sink["format_conversions"]} if sink.get("format_conversions") else {}),
                **({"source_bound": sink["source_bound"]} if sink.get("source_bound") else {}),
                **({"via": sink["via"]} if sink.get("via") else {}),
                # A model-declared argument was dropped, whether the tainted
                # argument itself came through a register or was recovered.
                **({"arity_mismatch": sink["arity_mismatch"]} if sink.get("arity_mismatch") else {}),
                # #791: the residual-chunk disclosure. This dict is an explicit
                # allow-list, not a spread of `sink`, so a classifier field that
                # is not named here never reaches the caller -- the reason the
                # shape had to be listed rather than just set upstream.
                **({"length_shape": sink["length_shape"]} if sink.get("length_shape") else {}),
                **({"residual_total": sink["residual_total"]} if sink.get("residual_total") else {}),
                **({"residual_cursor": sink["residual_cursor"]} if sink.get("residual_cursor") else {}),
                # #980: why the finding's class differs from the catalog's, and
                # what the copy SOURCE was sized as (None = could not be sized).
                **({"catalog_class": sink["catalog_class"]} if sink.get("catalog_class") else {}),
                **({"relabel": sink["relabel"]} if sink.get("relabel") else {}),
                **({"src_capacity": sink["src_capacity"]} if "src_capacity" in sink else {}),
                **({"src_capacity_source": sink["src_capacity_source"]}
                   if sink.get("src_capacity_source") else {}),
            },
            "path": path,
        }

    def _symbol_names(self, name: Any) -> tuple[str, str]:
        """``(raw symbol name, display name)`` for a name a taint result prints
        (#1001 item 4); both fall back to *name* itself when no symbol has it.

        Models key a callee with its leading underscores stripped
        (``ZNSt13basic_filebuf...``, ``strcpy_chk``), which is not a real
        symbol, neither reads nor demangles, and does not resolve in `bn proto
        get`; the raw spelling is tried with them restored. The display name is
        the symbol's ``short_name`` -- the spelling ``decompile`` prints (#978)
        and an identifier that resolves -- so a C symbol shows as its real name
        (``__strcpy_chk``), not the model key (#1001 review)."""
        name = str(name) if name else ""
        cache = self.__dict__.setdefault("_symbol_name_cache", {})
        if name in cache:
            return cache[name]
        found = (name, name)
        for cand in ((name, "_" + name, "__" + name) if name else ()):
            sym = _symbol_by_raw_name(self.bv, cand)
            raw = getattr(sym, "name", None) if sym is not None else None
            if isinstance(raw, str) and raw:
                short = getattr(sym, "short_name", None)
                found = (raw, short if isinstance(short, str) and short else raw)
                break
        cache[name] = found
        return found

    def _display_for_name(self, name: Any) -> str:
        """The demangled display name for *name* (see :meth:`_symbol_names`)."""
        return self._symbol_names(name)[1]

    def _raw_symbol_for(self, name: Any) -> str:
        """The real raw symbol for *name* (see :meth:`_symbol_names`)."""
        return self._symbol_names(name)[0]

    def _prose_name(self, name: Any) -> str:
        """Name a callee in diagnostic prose, without changing machine keys."""
        return self._symbol_names(name)[0 if self.mangled else 1]

    def _prose_function(self, func: Any) -> str:
        return (str(getattr(func, "name", "?")) if self.mangled
                else _display_name(func) or str(getattr(func, "name", "?")))

    def _reconstruct_path(self, node, why) -> list[dict[str, Any]]:
        chain = []
        seen = set()
        cur = node
        while cur is not None and cur in why and cur not in seen:
            seen.add(cur)
            entry = why[cur]
            ins = entry.get("instr")
            parents = entry.get("parents") or []
            if ins is not None:
                step = _instr_dict(ins, reason=entry.get("reason"),
                                   tainted=[entry.get("label", "?")])
                # #827: this walk follows ONE predecessor per step, but a value
                # defined at a branch join (a PHI of two tainted parents, e.g.
                # `x = cond ? a : b`) has several. Rendering parents[0] alone
                # dropped the alternates SILENTLY, so a reader took the single
                # rendered chain for the value's whole provenance -- and the
                # signature derived from this path inherited the same blind spot.
                # Disclose the join by naming how many parents were not followed;
                # the rendered chain stays one real path, but it can no longer be
                # mistaken for the only one.
                if len(parents) > 1:
                    step["alternate_parents"] = len(parents) - 1
                chain.append(step)
            cur = parents[0] if parents else None
        chain.reverse()
        return chain

    # -- backward ---------------------------------------------------------

    def backward(self, func: Any, sinks: list[dict[str, Any]], *, max_depth: int = 8) -> dict[str, Any]:
        self._bw_leaves: list[dict[str, Any]] = []
        self._bw_assumptions: list[str] = []
        # #810: backward-run truncation state, mirroring the forward contract
        # (``_truncated`` + ``_truncation_causes``, #579/#576). The caller-site cap
        # used to be disclosed as an assumption string ONLY -- prose no consumer can
        # gate on -- so a capped ascent came back with the same clean envelope as a
        # full one.
        self._bw_truncated = False
        self._bw_truncation_causes: set[str] = set()
        slices: list[dict[str, Any]] = []

        # Function-level resolution is sink-independent: a missing MLIL/SSA form
        # means the whole function can't be analyzed, so it stays a hard error.
        ssaf = self._ssa_func(func)
        instrs = self._instrs(ssaf)

        # Per-sink isolation: a sink that can't be seeded (callee not called
        # here, arg is a constant, ...) records a status note and the analysis
        # continues with the remaining sinks. Only an all-sinks-fail run is a
        # hard error, preserving the original single-sink behavior.
        sink_status: list[dict[str, Any]] = []
        errors: list[tuple[dict[str, Any], str]] = []
        for sink in sinks:
            desc = self._describe_locator(sink)
            try:
                seeds = self._seed_backward(func, ssaf, instrs, sink)
            except BoundedSink as exc:
                # A constant-length sink is a SUCCESSFUL "provably bounded"
                # conclusion, not a seed failure: record it as such and do NOT
                # add it to `errors`, so an all-bounded slice returns a clean
                # result (exit 0, --out written) instead of the all-failed hard
                # error (#310).
                sink_status.append({**desc, "seeded": False, "bounded": True, "note": str(exc)})
                continue
            except TaintError as exc:
                errors.append((sink, str(exc)))
                sink_status.append({**desc, "seeded": False, "note": str(exc)})
                continue
            n_before = len(slices)
            sink_slices: list[dict[str, Any]] = []
            # Collapse near-duplicate slices: the same origin reached through many
            # caller call-sites otherwise emits one full copy of the ~20-step
            # in-function chain per caller, which buries the single real origin and
            # spills to disk. Keep one representative per (seed, sink-site, origin)
            # and count how many call sites reached it (#46).
            seen_origin: dict[tuple, dict[str, Any]] = {}
            try:
                for seed_var, sink_ins in seeds:
                    sink_addr = hex(int(getattr(sink_ins, "address", 0)))
                    seed_label = var_label(seed_var)
                    for sl in self._backward_slice(func, seed_var, 0, max_depth, set(),
                                                   seed_ins=sink_ins):
                        o = sl["origin"]
                        key = (seed_label, sink_addr, o.get("kind"), o.get("var"),
                               o.get("value"), o.get("callee"), o.get("index"))
                        rep = seen_origin.get(key)
                        if rep is not None:
                            rep["reached_via_call_sites"] = rep.get("reached_via_call_sites", 1) + 1
                            continue
                        entry = {
                            "sink": {
                                "kind": sink.get("kind"),
                                "callee": sink.get("callee"),
                                "address": sink_addr,
                                "seed": seed_label,
                            },
                            "origin": sl["origin"],
                            "crossed_functions": sl["crossed"],
                            "slice": sl["steps"],
                        }
                        seen_origin[key] = entry
                        sink_slices.append(entry)
                slices.extend(sink_slices)
            except RecursionError:
                slices.extend(sink_slices)  # keep the partial, already-deduped slices
                # Per-sink isolation: a pathological cycle truncates this sink's
                # slice (keeping any partial steps already collected) without
                # taking down the other sinks or the whole op.
                self._bw_assume(
                    f"backward slice for {format_locator(sink)} truncated: "
                    "Python recursion limit reached (possible unresolved cycle)")
                # #810: the envelope carries the same truncation the sink_status row
                # and the assumption do, so `stats.truncated` is never a false
                # "complete" claim for a run that lost slices.
                self._bw_truncated = True
                self._bw_truncation_causes.add("recursion")
                sink_status.append({**desc, "seeded": True, "truncated": True,
                                    "slices": len(slices) - n_before,
                                    "note": "recursion limit reached while slicing; results incomplete"})
                continue
            sink_status.append({**desc, "seeded": True, "slices": len(slices) - n_before})

        if sinks and len(errors) == len(sinks):
            # Every sink failed to seed -> hard error (no partial results to keep).
            if len(errors) == 1:
                raise TaintError(errors[0][1])
            raise TaintError(
                "no backward seed resolved for any sink:\n  "
                + "\n  ".join(f"{format_locator(s)}: {m}" for s, m in errors))

        for sl in slices:
            sm, ss = derive_flow_facts(
                direction="backward", path=sl.get("slice"), sink=sl.get("sink"),
                origin=sl.get("origin"), crossed_functions=sl.get("crossed_functions"),
                display=self._display_for_name, symbol=self._raw_symbol_for)
            sl["metrics"] = sm
            sl["signature"] = ss
            # #1001 item 4 review: the slice row names its sink callee as the
            # forward flow row does -- demangled in text, raw beside it.
            _sk = sl.get("sink")
            if isinstance(_sk, dict) and _sk.get("callee"):
                _sym, _shown = self._symbol_names(_sk["callee"])
                sl["sink"] = {**_sk, "callee_symbol": _sym, "callee_display_name": _shown}
            # The origin's display name rides on the slice, beside `origin`, so
            # the origin object -- the slice's provenance claim -- is unchanged.
            _og = sl.get("origin")
            if isinstance(_og, dict) and _og.get("callee"):
                sl["origin_callee_display_name"] = self._display_for_name(_og["callee"])
        return {
            "direction": "backward",
            "function": {"name": str(func.name), "display_name": _display_name(func),
                         "address": hex(int(func.start))},
            "sinks": [self._describe_locator(s) for s in sinks],
            "sink_status": sink_status,
            "slices": slices,
            "leaves": self._bw_leaves,
            "assumptions": self._bw_assumptions,
            # Authoritative leaf count, matching the forward contract so the TEXT
            # header / JSON array / stats all reconcile for backward too (#181).
            # #810: `truncated`/`truncation_cause` are the forward stats pair
            # (#579/#576) -- a backward run whose caller ascent was capped is
            # INCOMPLETE and must not report from stats as a complete one.
            "stats": {"leaves": len(self._bw_leaves), "slices": len(slices),
                      "truncated": self._bw_truncated,
                      "truncation_cause": sorted(self._bw_truncation_causes)},
            # #812: backward had no diagnostics block at all while forward did,
            # so a curtailed slice (callers dropped at the ascent cap, a def
            # chain bottomed out at an unresolved field load) was shaped exactly
            # like an exhaustive one. Attached unconditionally -- backward's
            # slices ARE its result, so their completeness is a live question at
            # every count, unlike forward's zero-sink-only gate.
            "diagnostics": backward_diagnostics(
                self._bw_leaves, self._bw_assumptions,
                sinks_seeded=sum(1 for s in sink_status if s.get("seeded")),
                slices=len(slices),
                # A slice whose own origin is `unresolved` is the walk admitting
                # it stopped short; the gate must see it even with no leaf.
                unresolved_origins=sum(
                    1 for sl in slices if (sl.get("origin") or {}).get("kind") == "unresolved"),
                truncated=self._bw_truncated,
                truncation_cause=sorted(self._bw_truncation_causes)),
            # #812: configured knobs, echoed as forward does. `max_iters` is
            # deliberately absent: it bounds the forward intra-function fixpoint,
            # which backward has no equivalent of, and echoing an irrelevant knob
            # is how the "raise --max-iters" advice ended up on runs that could
            # never be fixed by it.
            "run_params": {
                "max_depth": int(max_depth),
                "unknown_call": str(self.unknown_call_policy),
            },
            "soundness": SOUNDNESS,
        }

    def _bw_assume(self, msg: str) -> None:
        if msg not in self._bw_assumptions:
            self._bw_assumptions.append(msg)

    def _backward_slice(self, func: Any, seed_var: Any, depth: int, max_depth: int,
                        visited_funcs: set, seed_ins: Any = None) -> list[dict[str, Any]]:
        """Backward def-chain slice within *func*, continuing into callers when it
        bottoms out at a parameter. Returns one slice per origin path."""
        ssaf = self._ssa_func(func)
        steps: list[dict[str, Any]] = []
        visited_vars: set = set()
        origin = {"kind": "unresolved"}
        # A LITERAL origin reached through a local buffer's writers (a `strcat`
        # appending a constant, a `buf[n] = 0` terminator) is "weak": the buffer's
        # writers are walked newest-first, and the sink's bytes also come from
        # every OLDER writer, so a parameter/source/frontier origin found later
        # replaces it instead of being dropped (PR-928 review P1: `strcpy(cmd,
        # host); cmd[n] = 0; strcat(cmd, " -c 1")` reported origin `call:strcat`,
        # and the caller ascent, keyed on a parameter origin, never ran). Among
        # weak origins a literal FILL (`memcpy(buf, "lit", n)`) outranks a
        # constant store: the bytes the sink reads come from the fill, the
        # 1-byte terminator only ends them (round 3 P3). A slice names ONE
        # origin, and only a parameter origin continues into the callers, so
        # when a buffer has both a source and a parameter input (`read(fd,
        # buf); buf[i] ^= key[i]`) the parameter wins: the source stays in the
        # steps, the caller ascent is not lost. Ranks: 0 parameter, 1 any
        # other strong origin (source, frontier), 2 literal fill, 3 constant.
        origin_rank = 0
        terminal_params: dict[int, Any] = {}

        def _set_origin(o, *, weak: int | bool = 0):
            nonlocal origin, origin_rank
            if weak is True:
                rank = 3
            elif weak:
                rank = int(weak) + 1
            else:
                rank = 0 if o.get("kind") == "parameter" else 1
            if origin["kind"] == "unresolved" or origin_rank > rank:
                origin = o
                origin_rank = rank

        # (buffer key, memory version) pairs already walked, so two buffers that
        # copy into each other inside a loop cannot recurse without end.
        visited_bufs: set = set()

        def _buffer_leaf(at, label, detail, **extra):
            leaf = {
                "kind": "address_of_local_unresolved",
                "address": hex(int(getattr(at, "address", 0))),
                "buffer": label,
                **extra,
                "il_text": str(at),
                "detail": detail,
            }
            if leaf not in self._bw_leaves:
                self._bw_leaves.append(leaf)

        def _global_leaf(at, label, addr, subject):
            """A writable-storage global the bytes are read from: any function
            may have set them, so they are unaccounted for (round 3 P1; round 4
            P1 for an element `g_hosts[i]` at a non-constant offset)."""
            _buffer_leaf(
                at, label,
                f"{subject} is built on the address {hex(addr)} of a global in "
                f"writable storage, whose bytes any function may have set; they are "
                f"unaccounted for and the slice is NOT complete -- re-run `bn taint "
                f"forward --source var:<global>` or inspect the writes to it",
                **{"global": hex(addr)})
            _set_origin({"kind": "global", "address": hex(addr), "buffer": label})

        def _follow_addresses(at, expr, d, mv, label, subject) -> tuple[bool, bool]:
            """Walk every buffer *expr* is an address INTO, at any offset:
            each local it is built on (``&line + rbx``) has its writers walked
            at *mv*, each writable global (``0x4040a0 + (i << 6)``) is a leaf.
            Returns ``(walked_local, global_leaf)``. The offset's own reads
            are the caller's to walk -- both feed the bytes (round 4 P1: the
            walk followed only the index and claimed a complete slice)."""
            locals_, addrs = self._address_terms(expr)
            for loc in locals_:
                _walk_local_buffer(at, loc, d, mv)
            glob = False
            for addr in addrs:
                if self._const_ptr_storage(addr) == "writable":
                    _global_leaf(at, label, addr, subject)
                    glob = True
            return bool(locals_), glob

        def _follow_input(at, expr, d, mv, label, what) -> str:
            """Continue the slice through one INPUT operand of a writer of
            *label* -- the value a store wrote, an argument a modeled fill
            derived the bytes from. *mv* is the memory version the writer
            READ, which is where a pointee operand's contents are read.
            Returns ``"followed"``, ``"literal"`` (an established constant:
            the bytes come from the writer itself) or ``"unresolved"`` (an
            operand with no SSA reads that is neither a literal nor a local
            the walk can follow -- a global load, an indirect address).

            An address operand is a BUFFER, not a literal and not just its
            offset (`strcpy(&cmd, &other)`, PR-928 review P1; `strcpy(buf,
            &other[i])`, round 4 P1): its bytes are read at THIS writer's
            input memory. Both `"ping %s"` and `g_host` reach a fill as
            CONST_PTR; only an address into READ-ONLY storage is a literal."""
            reads = expr_reads(expr)
            walked, glob = _follow_addresses(at, expr, d + 1, mv, label,
                                             f"{what} local buffer {label}")
            for r in reads:
                walk(r, d + 1, mv)
            if reads or walked:
                return "followed"
            if glob:
                return "unresolved"
            if "CONST" in op_name(expr) or self._address_terms(expr)[1]:
                # A scalar constant, or a constant address into read-only
                # storage (at a constant offset).
                return "literal"
            _buffer_leaf(
                at, label,
                f"{what} local buffer {label} is `{expr}`, which reads no variable "
                f"and is neither a literal nor a local the walk can follow; the "
                f"bytes it supplies are unaccounted for and the slice is NOT "
                f"complete")
            _set_origin({"kind": "address_of_local_unresolved", "buffer": label})
            return "unresolved"

        def _walk_local_buffer(defn, local, d, mv):
            """Continue the slice through the CONTENTS of *local*, read at
            memory version *mv* by the instruction that consumed its address
            (*defn* is the address-of definition, or the consuming call for a
            folded `popen(&cmd)` seed / a nested `strcpy(&cmd, &other)`)."""
            label = var_label(local)
            if mv is None:
                _buffer_leaf(
                    defn, label,
                    f"the sink reads the contents of local buffer {label} but the "
                    f"walk could not account for them (no memory version at the "
                    f"sink to walk); the slice stops at the address-of and is NOT "
                    f"complete -- re-run `bn taint forward --source {shlex.quote(f'var:{label}')}` "
                    f"or inspect the writes to the buffer")
                _set_origin({"kind": "address_of_local_unresolved", "buffer": label})
                return
            bkey = (var_key(local), int(mv))
            if bkey in visited_bufs:
                return
            visited_bufs.add(bkey)
            res = self._local_buffer_writers(ssaf, var_key(local), mv,
                                             base_size=self._var_byte_size(local))
            opaque: list[str] = []
            for w in res["writers"]:
                kind, wdefn = w[0], w[1]
                # The writer's own input memory: where a store's value or a
                # fill's source buffer was read. Using the SINK's version here
                # let a later overwrite of a source buffer (after it had been
                # copied) pose as the copy's origin (PR-928 review P2).
                wmv = self._mem_in(wdefn)
                if kind == "store":
                    steps.append(_instr_dict(wdefn, reason=f"store into local buffer {label}"))
                    src = getattr(wdefn, "src", None)
                    if _follow_input(wdefn, src, d, wmv, label, "the value stored into") == "literal":
                        cval = self._int_const(src)
                        _set_origin({"kind": "constant", "var": label,
                                     **({"value": cval} if cval is not None else {})},
                                    weak=2)
                elif kind == "maybe_store":
                    # A store through a pointer the walk cannot pin: it MAY
                    # write the bytes. Walked (its value is in the slice) AND
                    # disclosed -- never dropped.
                    _, _, why = w
                    steps.append(_instr_dict(
                        wdefn, reason=f"store that may write local buffer {label}"))
                    _follow_input(wdefn, getattr(wdefn, "src", None), d, wmv, label,
                                  "the value stored through a pointer that may alias")
                    _buffer_leaf(
                        wdefn, label,
                        f"`{wdefn}` {why} {label}; whether it wrote the bytes the sink "
                        "reads is unknown, so the slice is NOT complete -- inspect "
                        "the pointer's definition or seed inside the producing loop")
                    opaque.append(str(wdefn))
                elif kind == "escape":
                    steps.append(_instr_dict(
                        wdefn, reason=f"address of local buffer {label} stored to memory"))
                elif kind == "benign":
                    _, _, callee = w
                    steps.append(_instr_dict(
                        wdefn, reason=f"local buffer {label} passed to {self._prose_name(callee)} "
                                      f"(benign model: writes nothing tainted)"))
                    # The completeness claim now RESTS on a `{}` declaration
                    # for a callee that was handed the buffer -- say so where
                    # a reader of the result looks (round 4 P3).
                    self._bw_assume(
                        f"local buffer {label} is passed to {self._prose_name(callee)} @ "
                        f"{hex(int(getattr(wdefn, 'address', 0)))}, whose model is {{}} "
                        f"(benign: writes nothing tainted); the slice trusts that "
                        f"declaration -- if {self._prose_name(callee)} copies an input into the buffer, "
                        f"model it with `propagates` or the slice is incomplete")
                elif kind == "fill":
                    _, _, mkey, params, from_idxs, is_source, len_idxs = w
                    steps.append(_instr_dict(
                        wdefn, reason=f"local buffer {label} written by {self._prose_name(mkey)}"))
                    if is_source:
                        _set_origin({"kind": "source", "callee": mkey})
                    states = [_follow_input(wdefn, params[i], d, wmv, label,
                                            f"input {i} of {self._prose_name(mkey)} writing")
                              for i in from_idxs]
                    for i in len_idxs:
                        if expr_reads(params[i]):
                            steps.append(_instr_dict(
                                wdefn, reason=f"length argument {i} of {self._prose_name(mkey)} bounds "
                                              f"how much of the input lands in {label}"))
                            for r in expr_reads(params[i]):
                                walk(r, d + 1, wmv)
                    if not is_source and all(s == "literal" for s in states):
                        # Every input the model names is an established
                        # constant (a literal format/string): the bytes come
                        # from the call itself.
                        _set_origin({"kind": "call", "callee": mkey}, weak=1)
                else:  # opaque: a call may have written the bytes
                    _, _, callee, why = w
                    steps.append(_instr_dict(
                        wdefn, reason=f"local buffer {label} may be written by {self._prose_name(callee)}"))
                    _buffer_leaf(
                        wdefn, label,
                        f"{self._prose_name(callee)} {why} {label}; whether it wrote the bytes the sink "
                        "reads is unknown, so the slice is NOT complete -- model the "
                        "call or seed inside it",
                        callee=callee)
                    opaque.append(callee)
            if opaque:
                # Deferred past the loop so a real origin reached through another
                # writer (a modeled fill from a parameter) wins the slice's
                # origin and the opaque call stays what it is: a frontier leaf.
                _set_origin({"kind": "address_of_local_unresolved",
                             "buffer": label, "callee": opaque[0]})
            if res["truncated"]:
                # Discovery or fixed-point proof work was cut. Known writer
                # rows remain visible, but missing writers or an unfinished
                # solve prevent a complete-slice claim.
                self._bw_truncated = True
                self._bw_truncation_causes.add("memory_chain")
                _buffer_leaf(
                    defn, label,
                    f"the memory-SSA proof for local buffer {label} exhausted "
                    f"the shared allowance ({MEM_CHAIN_BUDGET} work units across "
                    "graph discovery and fixed-point sweeps); earlier writers may "
                    "be missing or the coverage/escape solve may be unfinished, "
                    "so the slice is NOT complete -- seed closer to the sink or "
                    "inspect the writes and unresolved calls")
                _set_origin({"kind": "address_of_local_unresolved", "buffer": label})
            elif not res["covered"]:
                benign = [w[2] for w in res["writers"] if w[0] == "benign"]
                if benign and len(benign) == len(res["writers"]):
                    why = (f"the only call that touches it is {', '.join(dict.fromkeys(benign))}, "
                           "modeled {} -- which promises no TAINTED write, not that "
                           "the call fills the buffer -- so no path is proven to write it")
                elif res["writers"]:
                    why = ("on at least one path from the function entry to the "
                           "sink nothing is proven to write the whole buffer -- a "
                           "conditional or loop-only fill, or only partial, indexed, "
                           "offset or may-alias writes -- so there the sink may read "
                           "whatever the stack held")
                else:
                    why = "no writer of the buffer was found on the sink's memory chain"
                _buffer_leaf(
                    defn, label,
                    f"the sink reads the contents of local buffer {label} but the "
                    f"walk could not account for them ({why}); the slice is NOT "
                    f"complete -- re-run `bn taint forward --source {shlex.quote(f'var:{label}')}` "
                    "or inspect the writes to the buffer")
                # Classified like `field_load_unresolved`: the origin names the
                # stop and the leaf is the one frontier entry, so the frontier
                # count is not inflated by counting one stop twice.
                _set_origin({"kind": "address_of_local_unresolved", "buffer": label})

        def walk(v, d, mv):
            """Backward over the def-chain of SSA value *v*. *mv* is the memory
            version at which the value is CONSUMED as a buffer (the sink call,
            or the modeled fill / store that read it) -- it rides unchanged
            through copies and arithmetic, because a pointer's pointee is read
            where the pointer is dereferenced, not where it was defined."""
            nonlocal origin
            if d > self.max_depth:
                self._bw_assume(
                    f"def-chain walk truncated at {self.max_depth} steps (engine max_depth)")
                return
            key = (var_key(v), getattr(v, "version", None))
            if key in visited_vars:
                return
            visited_vars.add(key)
            try:
                defn = ssaf.get_ssa_var_definition(v)
            except Exception:
                defn = None
            if defn is None:
                pidx = self._param_index_of(func, v)
                via_spill = False
                if pidx is None:
                    pidx = self._param_spill_index(func, ssaf, v)
                    via_spill = pidx is not None
                if pidx is not None:
                    terminal_params[pidx] = v
                    _set_origin({"kind": "parameter", "index": pidx, "var": var_label(v),
                                 **({"via_spill": True} if via_spill else {})})
                    if via_spill:
                        self._bw_assume(
                            f"stack local {var_label(v)} appears to be a spill of param "
                            f"{pidx} (its slot has a single store of the incoming parameter); "
                            f"canonicalized to param:{pidx} so caller-ascent continues (#434)")
                else:
                    _set_origin({"kind": "entry", "var": var_label(v)})
                return
            if self._is_call(defn):
                target = const_target(getattr(defn, "dest", None))
                name = self._callee_name(target)
                mkey, model = lookup_model(self.models, name)
                steps.append(_instr_dict(defn, reason=f"defined by call to {name or 'indirect'}"))
                if model and model.get("sources"):
                    origin = {"kind": "source", "callee": mkey or name}
                elif target is None:
                    origin = {"kind": "indirect_call", "var": var_label(v)}
                    leaf = {"kind": "indirect_call_unresolved",
                            "address": hex(int(getattr(defn, "address", 0))), "il_text": str(defn)}
                    if leaf not in self._bw_leaves:
                        self._bw_leaves.append(leaf)
                else:
                    # A modeled return that DERIVES from an argument (`memcpy`
                    # returns its destination, `strchr` a pointer into its
                    # haystack): the value is that argument's, so the slice
                    # continues through it -- `popen(memcpy(buf, ..))` at -O2
                    # stopped here as a complete `call: memcpy` (round 3).
                    ret_from = [_try_arg_index(str(r.get("from") or ""))
                                for r in (model or {}).get("propagates") or []
                                if str(r.get("to") or "") == "ret"]
                    cparams = self._call_params(defn) if ret_from else []
                    states: list[str] = []
                    for fi in ret_from:
                        if fi is None or fi >= len(cparams):
                            continue
                        steps.append(_instr_dict(
                            defn, reason=f"return of {mkey or name} derives from its argument {fi}"))
                        states.append(_follow_input(
                            defn, cparams[fi], d, mv, var_label(v),
                            f"argument {fi} of {mkey or name} returned by"))
                    if not states:
                        _set_origin({"kind": "call", "callee": name})
                    elif all(s == "literal" for s in states):
                        _set_origin({"kind": "call", "callee": mkey or name}, weak=1)
                return
            steps.append(_instr_dict(defn, reason="definition"))
            src_expr = getattr(defn, "src", None)
            # `p = &local` / `p = &line + off` / `p = g_hosts + (i << 6)`: the
            # sink consumes the CONTENTS of whatever buffer the pointer is
            # built on, not the pointer, so the slice must continue through
            # whatever wrote them -- the `snprintf(buf, n, fmt, obj->field);
            # popen(buf)` shape used to stop here with origin `unresolved` and
            # no leaf, and a variable offset followed only the index (round 4
            # P1). The bytes are read at the consumer's memory version *mv*,
            # so the writers are found on its memory-SSA chain; the offset's
            # own reads are walked too -- they pick which bytes.
            walked, glob = _follow_addresses(defn, src_expr, d, mv, var_label(v),
                                             f"the pointer {var_label(v)}")
            if walked or glob:
                for r in expr_reads(src_expr):
                    walk(r, d + 1, mv)
                return
            if src_expr is not None and op_name(src_expr) in ("MLIL_VAR_ALIASED",
                                                               "MLIL_VAR_ALIASED_FIELD"):
                # `i = buf[0] @ mem#k`: a read of an address-taken local BY
                # NAME -- the bytes are the buffer's contents at this
                # instruction's memory version, so the writers on that chain
                # are the value's producers (the xor-loop `buf[i] ^= key[i]`
                # reads the `read` fill through exactly this shape; round 3).
                aliased = getattr(src_expr, "src", None)
                aliased = getattr(aliased, "var", aliased)
                if aliased is not None:
                    _walk_local_buffer(defn, aliased, d, self._mem_in(defn))
                    return
            if src_expr is not None and "LOAD" in op_name(src_expr):
                la = self._addr_base_offset(ssaf, getattr(src_expr, "src", None))
                if la is not None:
                    rec = self._reaching_writer(
                        ssaf, getattr(src_expr, "src_memory", None), la,
                        getattr(src_expr, "size", None), set(), 0)
                    if rec is not None and rec[0] == "store":
                        # Recover the reaching store and continue the slice through
                        # the value it wrote -- not the base pointer's allocation,
                        # which dead-ends at malloc and reads as "clean" (#158).
                        store_defn = rec[1]
                        steps.append(_instr_dict(store_defn, reason="reaching store to field"))
                        st_reads = expr_reads(getattr(store_defn, "src", None))
                        if st_reads:
                            for r in st_reads:
                                walk(r, d + 1, mv)
                        else:
                            cval = self._int_const(getattr(store_defn, "src", None))
                            if cval is not None:
                                _set_origin({"kind": "constant", "value": cval})
                        return
                    if rec is not None and rec[0] == "source":
                        # The buffer was filled by a modeled receive/fill API; the
                        # loaded bytes originate from that source.
                        call_defn, callee = rec[1], rec[2]
                        steps.append(_instr_dict(
                            call_defn, reason=f"field filled by source {callee}"))
                        _set_origin({"kind": "source", "callee": callee})
                        return
                    base_var = self._addr_base_var(ssaf, getattr(src_expr, "src", None))
                    if self._field_base_is_alloc_or_param(ssaf, func, base_var):
                        # No reaching store/source in scope and the base is an
                        # allocation/parameter: a silent stop here reads as
                        # "locally allocated / clean". Surface it honestly (#158).
                        width = getattr(src_expr, "size", None)
                        base_label = var_label(base_var) if base_var is not None else None
                        leaf = {
                            "kind": "field_load_unresolved",
                            "address": hex(int(getattr(defn, "address", 0))),
                            "base": base_label,
                            "offset": (hex(la[1]) if isinstance(la[1], int) else None),
                            "width": (int(width) if isinstance(width, int) else None),
                            "il_text": str(defn),
                        }
                        if leaf not in self._bw_leaves:
                            self._bw_leaves.append(leaf)
                        _set_origin({"kind": "field_load_unresolved",
                                     "base": base_label,
                                     "offset": leaf["offset"], "width": leaf["width"]})
                        return
            reads = ssa_reads(defn)
            # A definition that reads no further SSA vars is a leaf: if its source
            # is a compile-time constant, the slice bottoms out at that literal.
            # Label it `constant` (as `trace` does) instead of leaving the default
            # `unresolved` -- for an auditor those are opposite risk conclusions
            # (#43). The narrow case the issue targets: a constant reached through
            # one or more variable copies (var = 0; r2 = var), where the direct
            # immediate is already handled at seeding time. Weak: reached as one
            # input of a local buffer's writer (`rsi = " -c 1"; strcat(cmd, rsi)`)
            # it must not outrank the parameter an older writer copied in.
            if not reads:
                cval = self._int_const(getattr(defn, "src", None))
                if cval is not None:
                    _set_origin({"kind": "constant", "value": cval, "var": var_label(v)},
                                weak=True)
            for r in reads:
                walk(r, d + 1, mv)

        seed_mv = self._mem_in(seed_ins) if seed_ins is not None else None
        if isinstance(seed_var, _BufferSeed):
            # `popen(&cmd, "r")` with the address folded into the parameter:
            # the seed IS the buffer's contents, read by the sink call itself.
            if seed_var.local is not None:
                _walk_local_buffer(seed_ins, seed_var.local, 0, seed_mv)
            else:
                _global_leaf(seed_ins, seed_var.name, int(seed_var.addr), "the sink argument")
        else:
            walk(seed_var, 0, seed_mv)
        base_steps = list(reversed(steps))

        # continue into callers only when the value's origin here is a parameter
        if (origin["kind"] == "parameter" and depth < max_depth
                and int(getattr(func, "start", 0)) not in visited_funcs):
            cont = self._continue_into_callers(
                func, terminal_params, depth, max_depth,
                visited_funcs | {int(getattr(func, "start", 0))}, base_steps)
            if cont:
                return cont
        return [{"steps": base_steps, "origin": origin, "crossed": []}]

    def _continue_into_callers(self, func, terminal_params, depth, max_depth,
                               visited_funcs, base_steps) -> list[dict[str, Any]]:
        caller_sites = list(getattr(func, "caller_sites", None) or [])
        if not caller_sites:
            return []
        results: list[dict[str, Any]] = []
        # The cap offers only the first MAX_CALLERS sites; ``ascended`` records how
        # many of those the ascent actually followed, because an offered site still
        # yields nothing when it has no call instruction at its recorded address,
        # its caller is unlifted, or the sink's arg index is past the recovered
        # arity (#810). The disclosure reports the measurement, never the constant.
        considered = caller_sites[:MAX_CALLERS]
        ascended: set[int] = set()
        for pidx, pvar in terminal_params.items():
            followed = False
            for idx, site in enumerate(considered):
                caller = getattr(site, "function", None)
                if caller is None:
                    continue
                addr = int(getattr(site, "address", 0))
                try:
                    c_instrs = self._instrs(self._ssa_func(caller))
                except TaintError:
                    continue
                call_ins = next((i for i in c_instrs
                                 if self._is_call(i) and int(getattr(i, "address", 0)) == addr), None)
                if call_ins is None:
                    continue
                cparams = self._call_params(call_ins)
                if pidx >= len(cparams):
                    continue
                # `run(&buf)` / `run(&buf + i)` with the address folded into
                # the call parameter: the callee's parameter is the buffer's
                # contents, so the ascent walks the caller's writers of it --
                # the same path `p = &buf; run(p)` takes -- beside the offset.
                seeds: list[Any] = (expr_reads(cparams[pidx])
                                    + self._folded_buffer_seeds(cparams[pidx]))
                if not seeds:
                    self._bw_assume(
                        f"arg {pidx} at {hex(addr)} in {self._prose_function(caller)} is not a trackable "
                        "scalar; caller slice not followed")
                    continue
                boundary = _instr_dict(call_ins, reason=f"passed as arg {pidx} to {func.name}")
                for sv in seeds:
                    for sub in self._backward_slice(caller, sv, depth + 1, max_depth, visited_funcs,
                                                    seed_ins=call_ins):
                        results.append({
                            "steps": sub["steps"] + [boundary] + base_steps,
                            "origin": sub["origin"],
                            "crossed": [str(func.name)] + sub["crossed"],
                        })
                        followed = True
                        ascended.add(idx)
            if not followed:
                results.append({"steps": base_steps,
                                "origin": {"kind": "parameter", "index": pidx, "var": var_label(pvar)},
                                "crossed": []})
        if len(caller_sites) > MAX_CALLERS:
            self._bw_note_caller_cap(
                func, len(caller_sites), len(considered), len(ascended))
        return results

    def _bw_note_caller_cap(self, func: Any, total: int, considered: int,
                            followed: int) -> None:
        """Disclose that the caller ascent dropped caller sites (#810).

        ``_continue_into_callers`` offers only the first ``MAX_CALLERS`` caller
        sites of a function whose slice bottoms out at a parameter, and an offered
        site still yields nothing when its recorded address carries no call
        instruction, its caller is unlifted, the sink's arg index is past the
        recovered arity, or the arg has no trackable scalar read. ``followed`` is
        how many sites actually ascended -- reported as measured, never as the cap
        constant, so a consumer gating on the count is not told the cap's coverage
        when resolution lost most of it. The drop was previously an assumption
        string only -- prose no consumer can gate on, so a capped ascent returned
        exactly the envelope of a complete one. Record the truncation on the run
        (``stats.truncated``/``truncation_cause``, the pair the forward side
        already carries, #579/#576), emit a blocking frontier leaf into the
        existing ``leaves`` channel so the loss is machine-readable in ``--format
        json`` and named in the text frontiers + verdict lines, and keep the
        assumption as the human-readable half carrying the same measurement."""
        dropped = int(total) - int(followed)
        unresolved = int(considered) - int(followed)
        self._bw_assume(f"{self._prose_function(func)} has {total} callers; "
                        f"caller ascent followed {followed}, capped at {MAX_CALLERS}")
        self._bw_truncated = True
        self._bw_truncation_causes.add("caller_cap")
        leaf = {
            "kind": "caller_sites_truncated",
            "address": hex(int(getattr(func, "start", 0))),
            "function": {"name": str(getattr(func, "name", "?")),
                         "display_name": _display_name(func) or "?",
                         "address": hex(int(getattr(func, "start", 0)))},
            "callers_total": int(total),
            "callers_followed": int(followed),
            "callers_dropped": dropped,
            "note": (f"caller-site cap: the ascent followed {followed} of {total} "
                     f"callers ({unresolved} of the {considered} sites the cap let "
                     f"through could not be followed, {dropped - unresolved} left "
                     "past the cap) -- an origin reachable only from an unfollowed "
                     "caller is missing from this slice"),
        }
        if leaf not in self._bw_leaves:
            self._bw_leaves.append(leaf)

    def _seed_backward(self, func, ssaf, instrs, sink) -> list[tuple]:
        kind = sink.get("kind")
        out = []
        if kind == "arg":
            callee = sink["callee"]
            idx = int(sink["index"])
            if idx < 0:
                # Guards programmatic callers that build the sink dict directly;
                # the CLI path is already rejected in parse_locator. A negative
                # idx would pass ``idx < len(params)`` and seed params[-1].
                raise TaintError(
                    f"--sink arg index {idx} is invalid: argument indices are "
                    f"0-based and must be >= 0")
            sites = self._find_callsites(instrs, callee, resolve_indirect=True)
            if not sites:
                # If the function dispatches through unresolved indirect calls,
                # name them and point at --resolve-map instead of a bare not-found
                # -- an indirectly-dispatched sink is not a silent dead end (#282).
                if any(self._is_call(ins)
                       and const_target(getattr(ins, "dest", None)) is None
                       for ins in instrs):
                    raise self._no_callsite_error(instrs, callee, func, seed_kind="sink")
                raise TaintError(
                    f"no call to {callee!r} found in {func.name}; check the --sink callee name")
            # Disclose an indirect (vtable/fn-ptr) anchor + value-set multiplicity,
            # mirroring the forward seed path (#282).
            self._note_indirect_anchors(sites, callee, self._bw_assume)
            saw_in_range = False
            max_params = 0
            for c in sites:
                params = self._call_params(c)
                max_params = max(max_params, len(params))
                if idx < len(params):
                    saw_in_range = True
                    reads = expr_reads(params[idx])
                    for r in reads:
                        out.append((r, c))
                    # `popen(&cmd, "r")` / `popen(&line + off, "r")`: BN folded
                    # the address into the parameter. The sink reads the
                    # buffer's contents, which the slice walks exactly as it
                    # does for `p = &cmd; popen(p)` -- a seed error (or, with
                    # a variable offset, a slice of only the offset) was the
                    # same bug with a second answer.
                    for bs in self._folded_buffer_seeds(params[idx]):
                        out.append((bs, c))
            if not out:
                # The locator was fine; the arg itself can't be sliced. Say so
                # precisely instead of blaming the --sink locator.
                if not saw_in_range:
                    # #433: BN under-recovers a copy-sink call's arguments when the
                    # callee is reached through a thunk/import/IFUNC carrying a
                    # too-narrow prototype -- e.g. an ARM-Thumb `j_memcpy` typed as
                    # `memcpy(int32_t)` drops r1/r2 from the MLIL (and LLIL) call, so
                    # the length (arg 2) is unseedable by params. Fall back to the
                    # calling-convention REGISTER for arg `idx`, but only for a
                    # MODELED sink and only for an index the model proves exists, so
                    # we never fabricate an argument that isn't actually passed.
                    if idx in self._model_arg_indices(callee):
                        reg, reg_seeds = self._reaching_arg_seeds_via_reg(func, sites, idx)
                        if reg_seeds:
                            out.extend(reg_seeds)
                            self._bw_assume(
                                f"--sink arg {idx} of {self._prose_name(callee)} was recovered from "
                                f"calling-convention register {reg}: this call site "
                                f"did not recover the modeled argument in MLIL; that alone "
                                f"does not diagnose a prototype defect. The value "
                                f"is read from the register's reaching definition rather "
                                f"than the MLIL call's parameter list (#433)")
                if not out and not saw_in_range:
                    # State the recovered arg count and the valid 0-based range so
                    # the off-by-one is obvious -- e.g. memcpy(dst, src, len) has 3
                    # args, so the length is index 2, not 3 (#291.4).
                    if max_params == 0:
                        detail = "its call site(s) expose no arguments in the recovered IL"
                    else:
                        detail = (
                            f"its call site(s) expose {max_params} argument(s), so valid "
                            f"indices are 0..{max_params - 1} (arg indices are 0-based)")
                    determinations = []
                    for site in sites:
                        try:
                            target = resolve_call_target(self.bv, site, follow_thunks=True)
                            determinations.append(_variadic_determination(target.function))
                        except Exception:
                            determinations.append(None)
                    if determinations and all(v is True for v in determinations):
                        detail += (". The callee is already declared variadic; omitted "
                                   "arguments are a call-recovery limitation, not evidence "
                                   "of a prototype defect. Inspect the call's argument provenance.")
                    elif determinations and all(v is False for v in determinations):
                        detail += (". The callee is determined nonvariadic. Only if independent "
                                   "argument evidence demonstrates a declaration discrepancy, "
                                   f"apply `bn proto set {shlex.quote(callee)} \"<prototype>\"` and re-run.")
                    else:
                        detail += (". Variadic status is undetermined; inspect the callee "
                                   "prototype and argument provenance before diagnosing a discrepancy.")
                    raise TaintError(
                        f"--sink arg index {idx} is out of range for {self._prose_name(callee)}: {detail}")
                if not out:
                    # Nothing to slice. A scalar CONSTANT literal (e.g. a fixed copy
                    # length) is a provably-bounded SUCCESS (#310). Gate on the
                    # operation being MLIL_CONST specifically -- NOT MLIL_CONST_PTR,
                    # which is a constant ADDRESS (a global dest/src pointer): that's
                    # an address expression with no def-chain, which must stay a seed
                    # error, not a misleading "bounded length" verdict.
                    arg_is_const = any(
                        op_name(params[idx]) == "MLIL_CONST"
                        for c in sites
                        for params in (self._call_params(c),)
                        if idx < len(params)
                    )
                    if arg_is_const:
                        raise BoundedSink(
                            f"--sink arg {idx} of {callee} is a compile-time constant in the "
                            f"recovered IL -- provably bounded, with no def-chain to slice backward")
                    raise TaintError(
                        f"--sink arg {idx} of {callee} reads no variable in the recovered IL "
                        f"(it is an address or fixed expression) -- there is no def-chain "
                        f"to slice backward")
        elif kind == "var":
            v = self._resolve_var(func, sink["selector"])
            # seed from the latest SSA use of the variable in the function
            for ins in reversed(instrs):
                for r in ssa_reads(ins):
                    if var_key(r) == var_key(v):
                        out.append((r, ins))
                        break
                if out:
                    break
        elif kind == "param":
            idx = int(sink["index"])
            pv = self._param_var(func, idx)
            if pv is None:
                raise self._param_not_found_error(func, idx)
            # seed from the earliest SSA read of the parameter (its entry value);
            # the walk bottoms out at the parameter and continues into callers
            for ins in instrs:
                for r in ssa_reads(ins):
                    if var_key(r) == var_key(pv):
                        out.append((r, ins))
                        break
                if out:
                    break
        elif kind == "ret":
            raise TaintError(
                "ret:<callee> is a forward-only locator: in the caller a return "
                "value has no def-chain to slice. Slice the variable that receives "
                "it (var:<name>), or use 'bn trace <fn> <call_addr> --arg N' for "
                "callee return-value provenance.")
        else:
            raise TaintError(
                f"unsupported backward sink kind: {kind!r}. backward --sink accepts "
                "param:<n> | var:<selector> | arg:<callee>:<n>; "
                "call:/model: are forward-only source seeds (a call's outputs have no "
                "def-chain to slice in the caller -- slice the variable that receives "
                "them, var:<name>, or arg:<callee>:<n> for an argument).")
        if not out:
            raise TaintError("no backward seed resolved; check --sink locator")
        return out

    # -- locator helpers --------------------------------------------------

    def _param_var(self, func, idx: int):
        try:
            params = list(func.parameter_vars)
        except Exception:
            params = []
        if 0 <= idx < len(params):
            return params[idx]
        return None

    def _resolve_var(self, func, selector: str):
        if self._find_variable is None:
            raise TaintError("variable selectors require a resolver (bridge-only)")
        try:
            return self._find_variable(func, selector)
        except Exception:
            # Accept the SSA-versioned form `name#version` that `dataflow defuse
            # --var` displays and accepts: taint seeds the base variable (it
            # tracks SSA versions internally), so strip the #version and retry the
            # base name. Removes the copy-paste trap where a versioned name from
            # defuse dead-ended in the taint locator (#356).
            if "#" in selector:
                base = selector.rsplit("#", 1)[0]
                if base and base != selector:
                    try:
                        return self._find_variable(func, base)
                    except Exception:
                        # Neither the versioned form nor its base resolved; report
                        # what the user actually typed (not the stripped base) so
                        # the error is actionable.
                        raise TaintError(
                            f"Variable not found: {selector} (also tried the base "
                            f"name {base!r} after dropping the SSA #version)")
            raise

    def _describe_locator(self, loc: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in loc.items() if k != "_resolved"}

