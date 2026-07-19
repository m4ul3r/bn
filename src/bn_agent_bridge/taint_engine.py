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
    _try_arg_index, model_arg_indices,
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
    derive_flow_facts,
)
from .taint_result import (  # noqa: F401
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


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------

class TaintEngine:
    def __init__(
        self,
        bv: Any,
        models: dict[str, Any],
        *,
        find_variable: Any = None,
        unknown_call_policy: str = "conservative",
        resolve_map: dict[str, Any] | None = None,
        max_iters: int = 256,
        max_depth: int = 64,
    ):
        self.bv = bv
        self.models = models
        self._find_variable = find_variable
        self.unknown_call_policy = unknown_call_policy
        # agent-supplied indirect-call resolution: {call_addr_hex: [target_addr, ...]}
        self.resolve_map = resolve_map or {}
        self.max_iters = max_iters
        self.max_depth = max_depth
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
        (malloc/calloc/realloc/ecalloc/xmalloc/strdup ...). The name hint catches
        the 2-arg calloc family (e.g. less's `ecalloc`) the size-focused _dest_alloc
        misses. Deliberately NOT the bare pointer-return heuristic -- too broad for
        a STABLE buffer key (it would conflate the returns of distinct non-allocator
        pointer functions and over-link). None for a non-allocator return pointer."""
        if d is None or "CALL" not in op_name(d):
            return None
        callee_nm = (self._callee_name(self._resolve_direct_target(d))
                     or "").split("@", 1)[0].lstrip("_").lower()
        if (any(tok in callee_nm for tok in self._ALLOC_NAME_HINTS)
                or self._dest_alloc(ssaf, var)[0] is not None):
            return ("heap", int(getattr(d, "address", 0)))
        return None

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
        conservative and treat every vararg as live)."""
        addr = self._const_ptr_addr(ssaf, expr)
        if addr is None:
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
        mapped = self.resolve_map.get(hex(int(getattr(ins, "address", 0))))
        if mapped:
            return [int(x, 16) if isinstance(x, str) else int(x) for x in mapped], "agent-map"
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
                detail = f"value-set ({callee} is 1 of {len(cands)} candidate targets)"
            wrapper = self._thin_wrapper_for(c, callee, wrapper_arg) if wrapper_arg is not None else None
            if wrapper is not None:
                add_assumption(
                    f"{callee} anchored at indirect callsite "
                    f"{hex(int(getattr(c, 'address', 0)))} via thin wrapper "
                    f"{getattr(wrapper, 'name', '?')} (forwards arg{wrapper_arg} to {callee}; "
                    f"resolved via {detail})")
            else:
                add_assumption(
                    f"{callee} anchored at indirect callsite "
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
            return self._forward_run(func, sources, max_depth=max_depth)
        return self._forward_attributed(func, sources, callsite_addrs, max_depth=max_depth)

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
        self._only_callsite_addr = only_callsite_addr
        # #559: count the source callsites the seed actually matched, so a
        # zero-result query can report "matched N source callsites" instead of an
        # unexplained empty result. Only the top run's ret/arg/call sources touch
        # this -- descended callees are seeded with synthetic param: locators.
        self._seed_callsites = 0

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
            fm, fs = derive_flow_facts(
                direction="forward", path=f.get("path"), sink=f.get("sink"),
                sources=sources_echo, leaves=sub["leaves"], fn_name=str(func.name))
            f["metrics"] = fm
            f["signature"] = fs
        result = {
            "direction": "forward",
            "function": {"name": str(func.name), "address": hex(int(func.start))},
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
            },
            "soundness": SOUNDNESS,
        }
        # #559: a modeled source that reaches no sink returns an otherwise-empty
        # result an agent can misread as a clean breadth check. Attach a compact,
        # factual frontier diagnostic (NOT a verdict) explaining where the seed
        # went and why the frontier stopped.
        if not unique_findings:
            result["diagnostics"] = self._forward_zero_diagnostics(sub)
        return result

    def _forward_zero_diagnostics(self, sub: dict[str, Any]) -> dict[str, Any]:
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
            truncation_cause=sorted(getattr(self, "_truncation_causes", set())))

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

        for addr in callsite_addrs:
            res = self._forward_run(func, sources, max_depth=max_depth, only_callsite_addr=addr)
            funcs_visited |= self._funcs_visited        # union of per-run visited sets
            if base is None:
                base = res
            by_source[hex(addr)] = {
                "reached_sinks": res["reached_sinks"],
                "leaves": res["leaves"],
            }
            union_findings.extend(res["reached_sinks"])
            union_leaves.extend(res["leaves"])
            union_assumptions.extend(res["assumptions"])
            max_depth_seen = max(max_depth_seen, res["stats"]["max_depth"])
            truncated = truncated or res["stats"]["truncated"]
            truncation_causes |= set(res["stats"].get("truncation_cause") or [])

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
            },
            "soundness": SOUNDNESS,
        }
        # #559/F2: attach a zero-sink frontier diagnostic when the whole
        # attributed union reached no sink -- but RECOMPUTE it over the UNIONED
        # leaves/assumptions/truncated, not the first callsite's own gate. Copying
        # base["diagnostics"] let a blocking frontier leaf emitted by callsite #2+
        # coexist with safe_to_report_all_clear=True from callsite #1 -- a false
        # all-clear on the default multi-callsite path that contradicts the same
        # result's own leaves array. The descriptive tainted_values/last_use stay
        # from the representative (first) run; the gate/frontier reflect the union.
        if not findings:
            base_diag = base.get("diagnostics") or {}
            union_sub = {
                "diag": {
                    "tainted_values": base_diag.get("tainted_values", 0),
                    "last_use": base_diag.get("last_use"),
                },
                "leaves": leaves,
                "assumptions": assumptions,
                "stats": {"truncated": truncated},
            }
            out["diagnostics"] = forward_zero_diagnostics(
                union_sub,
                seed_callsites=len(callsite_addrs),
                truncated=truncated, truncation_cause=sorted(truncation_causes))
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
                                       add_leaf=None, recv_buffer_covered=False) -> None:
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
        ``*arg:`` (call: preset) caller has no such leaf and passes False."""
        if not self._arg_ptr_is_indirect_load(ssaf, ptr_expr):
            return
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
            return
        recv_node = None
        for r in expr_reads(ptr_expr):
            recv_node = (var_key(r), getattr(r, "version", None))
            break
        buffer_slots[key] = {
            "recv_idx": getattr(callsite, "instr_index", None),
            "callee": callee, "idx": idx, "recv_node": recv_node, "pending": pending,
            "pending_leaf": pending_leaf,
        }

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
        if fn is None or getattr(fn, "is_thunk", False):
            return False
        sym = getattr(fn, "symbol", None)
        stype = str(getattr(getattr(sym, "type", None), "name", "") or "")
        if stype in {"ImportedFunctionSymbol", "LibraryFunctionSymbol",
                     "ImportAddressSymbol", "ExternalSymbol"}:
            return False
        try:
            return len(list(fn.mlil.instructions)) > 0
        except Exception:
            return False

    def _summarize(self, callee: Any, param_set: frozenset, depth: int, max_depth: int) -> dict[str, Any]:
        """Analyze *callee* with the given tainted parameter indices, caching the
        result per (callee, tainted-param-set) so it is computed once."""
        key = (int(callee.start), param_set)
        if key in self._cache:
            cached = self._cache[key]
            if cached is None:  # in-progress -> recursion cycle
                return {"reached_return": True, "out_params": frozenset(), "findings": [], "leaves": [],
                        "assumptions": [f"recursion cycle at {callee.name}; return conservatively tainted"],
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
            sub = {"reached_return": True, "out_params": frozenset(), "findings": [], "leaves": [],
                   "assumptions": [f"could not analyze {callee.name}: {exc}; return conservatively tainted"],
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
        stack_note = ""
        if stack_confirmed:
            stack_note = (
                f" Arg(s) {stack_confirmed} are STACK-passed (beyond the "
                f"{len(arg_regs)} integer-arg register(s)) -- most often a variadic "
                f"callee auto-typed as fixed-arity, so declare it variadic: "
                f"`bn proto set {name} \"<ret> {name}(<fixed args>, ...)\"`.")
        return {
            "kind": "arg_under_recovered",
            "address": hex(int(getattr(ins, "address", 0))),
            "callee": {"name": name,
                       "address": hex(int(getattr(callee_fn, "start", 0)))},
            "recovered_params": n_params,
            "dropped_args": confirmed,
            "stack_dropped_args": stack_confirmed,
            "note": (f"tainted arg(s) {confirmed} passed to {name} but Binary Ninja "
                     f"recovered only {n_params} parameter(s) -- the callee's "
                     f"signature is likely under-recovered, so taint into it was "
                     f"dropped. Recover the real prototype and apply "
                     f"`bn proto set {name} \"<prototype>\"`, then re-run this taint "
                     f"query.{stack_note}"),
        }

    def _descend(self, ins: Any, callee_fn: Any, tainted_args: dict, why: dict,
                 depth: int, max_depth: int, *, via: str | None = None) -> dict[str, Any]:
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
                               "assumptions": [], "out_params": frozenset()}
        if not valid:
            out["reached_return"] = True
            out["assumptions"].append(f"tainted args to {callee_fn.name} fall beyond its parameters; conservative")
            out["leaves"].append(self._frontier_leaf(
                ins, callee_fn, tainted_args,
                "tainted data passed to in-binary callee whose parameters BN did not "
                "recover, so taint into it was dropped. If it is a real callee with a "
                "wrong signature, apply `bn proto set <callee> \"<prototype>\"` and "
                "re-run; otherwise investigate"))
            return out
        if depth + 1 > max_depth:
            self._truncated = True
            self._truncation_causes.add("max_depth")
            out["reached_return"] = True
            out["assumptions"].append(
                f"max interprocedural depth {max_depth} reached at {callee_fn.name}; not descended")
            out["leaves"].append(self._frontier_leaf(
                ins, callee_fn, tainted_args,
                f"tainted data passed to in-binary callee with no model; depth bound "
                f"({max_depth}) stopped descent -- investigate or raise --depth"))
            return out
        sub = self._summarize(callee_fn, valid, depth + 1, max_depth)
        first_hit = tainted_args[sorted(valid)[0]][0]
        prefix = self._reconstruct_path(first_hit, why)
        note = f"calls {callee_fn.name} with tainted arg(s) {sorted(valid)}"
        if via:
            note = f"[{via}-resolved] " + note
        prefix.append(_instr_dict(ins, reason=note, tainted=[node_label(first_hit, why)],
                                  callee=getattr(callee_fn, "name", None)))
        for f in sub["findings"]:
            out["findings"].append({"sink": f["sink"], "path": prefix + f["path"]})
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
        return out

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
                f"{func.name} contains {len(unimpl)} unlifted/unimplemented "
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

            def _note_under_recovered_sink_arg(argidx: int) -> None:
                # #577: a modeled sink's OWN declared arg index (e.g. memcpy's
                # length at arg 2) sits beyond the params BN recovered for this
                # call -- an under-recovered arity, the shape an ARM-Thumb
                # `j_memcpy` typed `memcpy(int32_t)` produces by dropping the
                # length register. The in-range sink loop below never inspects
                # it, and the modeled branch's trailing `continue` bypasses every
                # other honesty path, so a tainted value passed there vanishes and
                # a zero-sink run falsely reports all-clear. `backward` RAISES on
                # this exact condition; the forward walk must at least disclose it.
                # Recover the dropped arg through its calling-convention register
                # (the #433 reg bridge) and, ONLY when it actually carries taint,
                # emit a blocking `arg_under_recovered` leaf so the existing leaves
                # gate withholds. Gating on real taint keeps a merely-under-
                # recovered but quiet sink from crying wolf on every function that
                # calls it. Deduped per call site by leaf identity.
                try:
                    reg, seeds = self._reaching_arg_seeds_via_reg(func, [ins], argidx)
                except Exception:
                    return
                if not seeds:
                    return

                def _carries_taint(v: Any) -> bool:
                    k = var_key(v); ver = getattr(v, "version", None)
                    if (k, ver) in tainted or (k, None) in tainted:
                        return True
                    return self._var_buffer_tainted(ssaf, v, tainted, set(), 0)

                if not any(_carries_taint(v) for v, _site in seeds):
                    return
                # Conform to the established `arg_under_recovered` leaf contract
                # (see `_arg_under_recovered_leaf`): `callee` is ALWAYS a
                # {name,address} dict and the leaf ALWAYS carries BOTH `dropped_args`
                # and `stack_dropped_args` (empty when none), so the text renderer
                # (`_leaf_group_key`/`_render_leaf_line`) and the JSON consumers treat
                # this leaf identically to the descend-side one. A string callee or a
                # missing address/stack field diverges from that schema (#576/#577).
                # The address is the resolved call target: `site_taddr` for the
                # resolved-indirect branch, else the direct-call target; "0x0" is the
                # consistent sentinel the established path emits when unresolvable
                # (`hex(int(getattr(callee_fn, "start", 0)))`). This modeled-sink arg
                # was recovered through its ABI register (the #433 reg bridge), so it
                # is register-passed -- `stack_dropped_args` is empty here.
                callee_name = mkey or name or "?"
                taddr = site_taddr if site_taddr is not None else self._resolve_direct_target(ins)
                callee: dict[str, Any] = {
                    "name": callee_name,
                    "address": hex(int(taddr)) if taddr is not None else "0x0",
                }
                leaf = {
                    "kind": "arg_under_recovered",
                    "address": hex(int(getattr(ins, "address", 0))),
                    "callee": callee,
                    "recovered_params": len(params),
                    "dropped_args": [argidx],
                    "stack_dropped_args": [],
                    "register": reg,
                    "note": (
                        f"a tainted value reaches arg {argidx} of the modeled sink "
                        f"{callee_name} through its ABI register {reg}, but Binary "
                        f"Ninja recovered only {len(params)} parameter(s) for this "
                        f"call, so the sink's declared arg index is out of range and "
                        f"the tainted flow into it was NOT reported -- a 'no sinks "
                        f"reached' result here is not proof of safety. Recover the "
                        f"real prototype and apply `bn proto set {callee_name} "
                        f"\"<prototype>\"`, then re-run this taint query."),
                }
                if leaf not in leaves:
                    leaves.append(leaf)

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
            if sink is not None:
                # #443: a bounded-write sink (wrapped recv/read) is armed by its
                # `len_arg` -- an attacker-controlled write length -- in addition to
                # any explicit `tainted_args`.
                _sink_args = list(sink.get("tainted_args") or [])
                if sink.get("len_arg") is not None and sink["len_arg"] not in _sink_args:
                    _sink_args.append(int(sink["len_arg"]))
                # #615: an empty tainted_args with no len_arg is the existing
                # catalog convention for an ALWAYS-unsafe API (e.g. gets()) --
                # the loop below has nothing to iterate, so without this arm
                # the sink never fires regardless of any taint reaching the
                # call, a pure false negative. Report it unconditionally, once
                # per call site, with no fabricated arg taint.
                if not _sink_args and sink.get("len_arg") is None:
                    sig = (addr, None) if site_taddr is None else (addr, None, site_taddr)
                    if sig not in recorded_sinks:
                        recorded_sinks.add(sig)
                        findings.append(self._make_unconditional_finding(ins, mkey or name, sink))
                for argidx in _sink_args:
                    if argidx < len(params):
                        ht = arg_taint(params[argidx])
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
                                    _dest_idx = (int(sink["buf_arg"])
                                                 if (sink.get("buf_arg") is not None
                                                     and argidx == sink.get("len_arg")) else 0)
                                    _bnd_reason = (self._bounded_copy_reason(ssaf, params, argidx, dest_idx=_dest_idx)
                                                   if sink.get("class") == "overflow_len" else None)
                                if _bnd_reason is not None:
                                    eff_sink = {
                                        **sink,
                                        "class": "bounded_len",
                                        "detail": (sink.get("detail") or "")
                                        + f" (attacker-derived length, but {_bnd_reason})",
                                    }
                                elif sink.get("class") == "overflow_len":
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
                                if eff_sink.get("class") in _OVERFLOW_INDEX_CLASSES \
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
                                if eff_sink.get("class") == "overflow_len" \
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
                                findings.append(self._make_finding(ins, mkey or name, argidx, eff_sink, ht, why))
                    else:
                        # #577: the sink's OWN declared arg index is out of range
                        # of the recovered params -- disclose the under-recovered
                        # arity instead of silently skipping it (which would let a
                        # false all-clear survive when a tainted value is passed
                        # there through the dropped ABI register).
                        _note_under_recovered_sink_arg(argidx)
            for rule in model.get("propagates") or []:
                to = rule.get("to")
                frm = rule.get("from")
                hit = self._token_hit_node(ssaf, params, frm, tainted)
                applied = False
                if hit is not None:
                    applied = self._apply_to_token(ssaf, ins, params, to, taint_node, name or "?", parents=[hit])
                    if applied:
                        changed = True
                    else:
                        _note_unkeyed_store(to)
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
                    if applied and isinstance(frm, str) and frm.startswith("*arg:") and to == "*arg:0":
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
                                f"{name or '?'} at {hex(int(getattr(ins, 'address', 0)))} "
                                f"(arg{src_i} -> arg0); propagated to the destination, "
                                f"not itself flagged as a sink")
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
                        if self._apply_to_token(ssaf, ins, params, vto, taint_node, name or "?", parents=[ht[0]]):
                            changed = True
                        else:
                            _note_unkeyed_store(vto)
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
                            findings.append(self._make_finding(ins, mkey or name, i, eff_vsink, ht, why))
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
                    if read_taint(ins) or self._return_buffer_tainted(ssaf, ins, tainted):
                        reached_return = True
                    continue

                if self._is_call(ins):
                    target = self._resolve_direct_target(ins)
                    name = self._callee_name(target)
                    mkey, model = lookup_model(self.models, name)
                    params = self._call_params(ins)

                    # 1+2) model-driven sink detection + propagation (shared helper)
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
                        continue  # modeled (sink/propagate/source-only); body not descended

                    # 3) no model: resolve the target(s) and descend.
                    tainted_args = {i: arg_taint(p) for i, p in enumerate(params) if arg_taint(p)}
                    if not tainted_args:
                        # The MLIL call recovered no tainted arg -- but BN may have
                        # under-recovered the callee's arity (Thumb 0-arity miss /
                        # variadic), dropping a tainted arg-register value that never
                        # appears in `params`. Emit an honest frontier instead of a
                        # silent false-negative (#381).
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
                        mapped = self.resolve_map.get(hex(int(getattr(ins, "address", 0))))
                        if mapped:
                            candidates = [int(x, 16) if isinstance(x, str) else int(x) for x in mapped]
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
                        # then BOTH re-modeled and re-classified. When the candidate
                        # is neither a descendable in-binary function nor a modeled
                        # import, follow a single-instruction thunk to its real
                        # target and re-run the model lookup + internal check on
                        # THAT target. Without re-modeling, a decorated veneer whose
                        # own name misses the model DB but that tail-calls a modeled
                        # sink (j_memcpy -> memcpy) would fall through to the
                        # conservative external tail and the sink be silently missed.
                        descend_fn = cfn
                        descend_internal = cfn_internal
                        if md is None and cfn is not None and not cfn_internal:
                            resolved = self._follow_thunk_cached(cfn)
                            if resolved is not None and resolved is not cfn:
                                rnm = self._callee_name(int(getattr(resolved, "start", 0))) \
                                    or (str(resolved.name) if getattr(resolved, "name", None) else None)
                                rmk, rmd = lookup_model(self.models, rnm)
                                if rmd is not None:
                                    nm, mk, md = rnm, rmk, rmd
                                elif self._is_internal(resolved):
                                    descend_fn = resolved
                                    descend_internal = True
                        # Re-imported export: the call routed through a PLT/GOT
                        # stub to a symbol that is ALSO defined in this binary.
                        # Bridge to the local definition so taint descends into
                        # its body instead of stopping at a conservative external
                        # leaf (#46 item 3). Only when unmodeled and not already
                        # internal, and only to a real in-binary definition.
                        if md is None and not descend_internal:
                            local = self._local_definition_for(nm)
                            if local is not None:
                                descend_fn = local
                                descend_internal = True
                                add_assumption(
                                    f"bridged re-imported export {nm} to its in-binary "
                                    f"definition at {hex(int(getattr(local, 'start', 0)))}")
                        if md is not None:
                            # resolved target is a modeled external
                            mchanged, _ = apply_model(ins, params, md, mk, nm, site_taddr=taddr)
                            if mchanged:
                                changed = True
                            resolved_names.append(report_name)
                        elif descend_internal:
                            d = self._descend(ins, descend_fn, tainted_args, why, depth, max_depth, via=via)
                            findings.extend(d["findings"])
                            for lf in d["leaves"]:
                                if lf not in leaves:
                                    leaves.append(lf)
                            for a in d["assumptions"]:
                                add_assumption(a)
                            ret_tainted = ret_tainted or d["reached_return"]
                            descend_outparams |= set(d.get("out_params") or ())
                            descend_outparam_elems |= set(d.get("out_param_elems") or ())
                            resolved_names.append(report_name)
                        else:
                            if self.unknown_call_policy != "stop":
                                ret_tainted = True
                                add_assumption(f"external {nm or hex(taddr)} has no model; return conservatively tainted")
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
                                    "callee": {"name": str(nm or hex(taddr)),
                                               "address": hex(int(taddr))},
                                    "tainted_args": sorted(tainted_args.keys()),
                                    "note": ("--unknown-call stop: taint reached this unmodeled "
                                             "external and was NOT propagated through it; flow past "
                                             "this call is not analyzed -- NOT an all-clear"),
                                }
                                if _stop_leaf not in leaves:
                                    leaves.append(_stop_leaf)
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
                            f"{', '.join(resolved_names)}")
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

    def _token_tainted(self, ssaf: Any, ins: Any, params: list[Any], tok: str | None, tainted: set) -> bool:
        return self._token_hit_node(ssaf, params, tok, tainted) is not None

    def _apply_to_token(self, ssaf: Any, ins: Any, params: list[Any], tok: str | None,
                        taint_node, callee: str, parents: list | None = None) -> bool:
        parents = parents or []
        if not tok:
            return False
        if tok == "ret" or tok == "*ret":
            done = False
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
        """#464: seeding/slicing a param index past a function's recovered arity is
        often BN under-recovering its signature -- e.g. a vtable method whose
        data/args parameter was DROPPED uniformly across the call chain, so the
        per-callsite ``arg_under_recovered`` frontier (which needs an arity
        *mismatch*) can't catch it. Disclose the proto-set remedy rather than a
        bare not-found, so the disclose -> proto set -> re-run loop still applies."""
        try:
            n = len(list(getattr(func, "parameter_vars", []) or []))
            have = f" (BN recovered {n} parameter(s))"
        except Exception:
            have = ""
        return TaintError(
            f"parameter {idx} not found on {func.name}{have}. If BN under-recovered "
            f"this function's signature (a dropped parameter is a common vtable / "
            f"stripped-proto shape), apply `bn proto set {func.name} \"<prototype>\"` "
            f"and re-run this taint query")

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
                    add_assumption(
                        f"{callee} writes the received bytes to the scatter-gather iov "
                        f"buffer(s) (msghdr->msg_iov[i].iov_base; for recvmmsg, per msgvec "
                        f"entry), not the header pointer this --source seeds; the payload "
                        f"taint is NOT followed from here. Use --source call:{callee}, which "
                        f"resolves the iovec buffer from the msg_iov/iov_base setup "
                        f"automatically, or seed the filled buffer directly (--source "
                        f"var:<buf>) when the iovec is built dynamically."
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
                            f"{callee} also writes tainted data to {', '.join(_outs)} per its "
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
                    add_assumption(f"seeded from {callee} callsite at {hex(only)} (per-source attribution)")
                elif len(calls) > 1:
                    add_assumption(f"{len(calls)} callsites of {callee}; seeded from all")
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
                    add_assumption(f"seeded from {callee} callsite at {hex(only)} (per-source attribution)")
                elif len(calls) > 1:
                    add_assumption(f"{len(calls)} callsites of {callee}; seeded from all")
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
                    for sd in src_defs:
                        to = str(sd.get("to") or "")
                        if to == "ret":
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
                                        f"recvmsg_out_param @ {_ca}: {callee} fills the "
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
                                    f"recvmsg_iovec_unresolved @ {_ca}: {callee} fills the "
                                    f"scatter-gather buffer(s) at msghdr->msg_iov[i].iov_base, but "
                                    f"the iovec setup (msg_iov -> iov_base) at this callsite could "
                                    f"not be statically resolved -- likely a dynamically-built "
                                    f"iovec or a receive-helper whose caller passes the out-buffer "
                                    f"(#452). The payload taint is NOT followed from here; seed the "
                                    f"filled buffer directly (--source var:<buf>) or, for a helper, "
                                    f"seed the buffer the caller passes in.")
                        elif to.startswith("*arg:"):
                            idx = _try_arg_index(to)
                            if idx is not None and idx < len(params):
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
                                    # preset path, so the indirect-slot leaf is the
                                    # only thing that can withhold the all-clear here.
                                    self._register_indirect_buffer_slot(
                                        ssaf, params[idx], c, callee, idx, buffer_slots,
                                        add_assumption, add_leaf=add_leaf,
                                        recv_buffer_covered=False)
                        elif to.startswith("arg:"):
                            idx = _try_arg_index(to)
                            if idx is not None and idx < len(params):
                                for r in expr_reads(params[idx]):
                                    if taint_node((var_key(r), getattr(r, "version", None)), var_label(r), c,
                                                  f"source: {callee} arg{idx} value (call: preset)", []):
                                        seeded = True
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
        path.append(_instr_dict(ins, reason=f"tainted arg{argidx} reaches {callee}",
                                tainted=[node_label(n, why) for n in hit_nodes],
                                callee=callee))
        return {
            "sink": {
                "callee": callee,
                "address": hex(int(getattr(ins, "address", 0))),
                "tainted_arg_index": argidx,
                "class": sink.get("class"),
                "detail": sink.get("detail"),
                **({"format_constant": sink["format_constant"]} if sink.get("format_constant") is not None else {}),
                **({"source_bound": sink["source_bound"]} if sink.get("source_bound") else {}),
                **({"via": sink["via"]} if sink.get("via") else {}),
            },
            "path": path,
        }

    def _make_unconditional_finding(self, ins, callee, sink) -> dict[str, Any]:
        """Build a finding for an always-unsafe sink (empty ``tainted_args``,
        no ``len_arg``) that has no arg taint to chain -- no real taint node
        exists to hand to :meth:`_reconstruct_path`, so this builds a single-step
        path describing the always-unsafe API reached at this call site instead
        (#615)."""
        path = [_instr_dict(ins, reason=f"always-unsafe API {callee} reached",
                            callee=callee)]
        return {
            "sink": {
                "callee": callee,
                "address": hex(int(getattr(ins, "address", 0))),
                "tainted_arg_index": None,
                "class": sink.get("class"),
                "detail": sink.get("detail"),
            },
            "path": path,
        }

    def _reconstruct_path(self, node, why) -> list[dict[str, Any]]:
        chain = []
        seen = set()
        cur = node
        while cur is not None and cur in why and cur not in seen:
            seen.add(cur)
            entry = why[cur]
            ins = entry.get("instr")
            if ins is not None:
                chain.append(_instr_dict(ins, reason=entry.get("reason"),
                                        tainted=[entry.get("label", "?")]))
            parents = entry.get("parents") or []
            cur = parents[0] if parents else None
        chain.reverse()
        return chain

    # -- backward ---------------------------------------------------------

    def backward(self, func: Any, sinks: list[dict[str, Any]], *, max_depth: int = 8) -> dict[str, Any]:
        self._bw_leaves: list[dict[str, Any]] = []
        self._bw_assumptions: list[str] = []
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
                    for sl in self._backward_slice(func, seed_var, 0, max_depth, set()):
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
                origin=sl.get("origin"), crossed_functions=sl.get("crossed_functions"))
            sl["metrics"] = sm
            sl["signature"] = ss
        return {
            "direction": "backward",
            "function": {"name": str(func.name), "address": hex(int(func.start))},
            "sinks": [self._describe_locator(s) for s in sinks],
            "sink_status": sink_status,
            "slices": slices,
            "leaves": self._bw_leaves,
            "assumptions": self._bw_assumptions,
            # Authoritative leaf count, matching the forward contract so the TEXT
            # header / JSON array / stats all reconcile for backward too (#181).
            "stats": {"leaves": len(self._bw_leaves), "slices": len(slices)},
            "soundness": SOUNDNESS,
        }

    def _bw_assume(self, msg: str) -> None:
        if msg not in self._bw_assumptions:
            self._bw_assumptions.append(msg)

    def _backward_slice(self, func: Any, seed_var: Any, depth: int, max_depth: int,
                        visited_funcs: set) -> list[dict[str, Any]]:
        """Backward def-chain slice within *func*, continuing into callers when it
        bottoms out at a parameter. Returns one slice per origin path."""
        ssaf = self._ssa_func(func)
        steps: list[dict[str, Any]] = []
        visited_vars: set = set()
        origin = {"kind": "unresolved"}
        terminal_params: dict[int, Any] = {}

        def _set_origin(o):
            nonlocal origin
            if origin["kind"] == "unresolved":
                origin = o

        def walk(v, d):
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
                    if origin["kind"] == "unresolved":
                        origin = {"kind": "parameter", "index": pidx, "var": var_label(v),
                                  **({"via_spill": True} if via_spill else {})}
                    if via_spill:
                        self._bw_assume(
                            f"stack local {var_label(v)} appears to be a spill of param "
                            f"{pidx} (its slot has a single store of the incoming parameter); "
                            f"canonicalized to param:{pidx} so caller-ascent continues (#434)")
                elif origin["kind"] == "unresolved":
                    origin = {"kind": "entry", "var": var_label(v)}
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
                    origin = {"kind": "call", "callee": name}
                return
            steps.append(_instr_dict(defn, reason="definition"))
            src_expr = getattr(defn, "src", None)
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
                                walk(r, d + 1)
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
            # immediate is already handled at seeding time.
            if not reads and origin["kind"] == "unresolved":
                cval = self._int_const(getattr(defn, "src", None))
                if cval is not None:
                    origin = {"kind": "constant", "value": cval, "var": var_label(v)}
            for r in reads:
                walk(r, d + 1)

        walk(seed_var, 0)
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
        MAX_CALLERS = 16
        results: list[dict[str, Any]] = []
        for pidx, pvar in terminal_params.items():
            followed = False
            for site in caller_sites[:MAX_CALLERS]:
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
                seeds = expr_reads(cparams[pidx])
                if not seeds:
                    # arg has no SSA value reads (e.g. &localbuf); can't follow scalar slice
                    self._bw_assume(
                        f"arg {pidx} at {hex(addr)} in {caller.name} is not a trackable scalar; "
                        "caller slice not followed")
                    continue
                boundary = _instr_dict(call_ins, reason=f"passed as arg {pidx} to {func.name}")
                for sv in seeds:
                    for sub in self._backward_slice(caller, sv, depth + 1, max_depth, visited_funcs):
                        results.append({
                            "steps": sub["steps"] + [boundary] + base_steps,
                            "origin": sub["origin"],
                            "crossed": [str(func.name)] + sub["crossed"],
                        })
                        followed = True
            if len(caller_sites) > MAX_CALLERS:
                self._bw_assume(f"{func.name} has {len(caller_sites)} callers; followed first {MAX_CALLERS}")
            if not followed:
                results.append({"steps": base_steps,
                                "origin": {"kind": "parameter", "index": pidx, "var": var_label(pvar)},
                                "crossed": []})
        return results

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
                    for r in expr_reads(params[idx]):
                        out.append((r, c))
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
                                f"--sink arg {idx} of {callee} was recovered from "
                                f"calling-convention register {reg}: this call site "
                                f"under-recovered its arguments (a thunk/import with a "
                                f"too-narrow prototype dropped the arg), so the value "
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
                    # #464: an index past the recovered arity is frequently BN
                    # under-recovering the callee's signature (e.g. an ARM IFUNC libc
                    # sink typed by its resolver as 0/1 args), which silently drops the
                    # length/src slice. Name the proto-set remedy, not just "out of range".
                    detail += (
                        f". If {callee} is a real sink whose signature BN under-recovered "
                        f"(a common IFUNC / stripped-proto shape), apply "
                        f"`bn proto set {callee} \"<prototype>\"` and re-run this slice")
                    raise TaintError(
                        f"--sink arg index {idx} is out of range for {callee}: {detail}")
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
                raise TaintError(f"parameter {idx} not found on {func.name}")
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

