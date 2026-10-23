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

import json
import re
import shlex
from typing import Any, NamedTuple

# Structural split (#562): the engine re-exports the moved names so existing
# importers (`from bn_agent_bridge.taint_engine import X`, `_taint.X`) keep
# working unchanged. Behaviour is identical -- this only relocates code.
from .taint_models import (  # noqa: F401
    SOUNDNESS, _OVERFLOW_INDEX_CLASSES, _RECVMSG_FAMILY,
    _BROAD_SOURCE_BYTES, _BROAD_SOURCE_MEMBERS, _BUILTIN_MODELS,
    _model_buffer_source_args, TaintError,
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

# A value-preserving read of one variable. Conversions (SX/ZX/LOW_PART, a
# partial-register `rax.eax` field read) are not copies of a pointer.
_COPY_OPS = frozenset({"MLIL_VAR_SSA", "MLIL_VAR"})

# BN's outlined store runs and the argument that states their length
# (`__builtin_strcpy` states it through its literal source instead).
_BUILTIN_LEN_ARG = {"__builtin_memset": 2, "__builtin_memcpy": 2,
                    "__builtin_memmove": 2, "__builtin_strncpy": 2}

# The backward leaf kind for the object a pointer with no local behind it
# points to (a call's return, a parameter): its extent is unknown to the walk.
_POINTER_LEAF = "pointer_object_unresolved"
_POINTER_NOUN = "the object"

# The backward leaf kind for stores that cover a consumed region the walk
# cannot bound: the demand reaches outside the object, or its offset is not
# one constant (round 2). The stores are reported as facts.
_EXTENT_LEAF = "object_extent_unknown"

# The backward leaf kind for a consumed region every byte of which stores the
# walk can see write on every path: a fact, never a certified proof -- it rests
# on BN's recovery of the object's extent (round 3).
_PROOF_LEAF = "region_store_proof_not_certified"

# `_follow_input`'s default demand: the writer consumes the operand itself as
# a pointer, at offset 0 of what it points to.
_AS_CONSUMER = object()


def _ssa_root_key(v: Any) -> tuple:
    """The base key of the object an SSA pointer value *v* points to, for the
    writer walk of a pointer with no local behind it (a call's return, a
    parameter): ``("ssa", var_key, version)``."""
    return ("ssa", var_key(v), getattr(v, "version", None))


def _is_ssa_root(base_key: Any, key: tuple) -> bool:
    """Is ``(var_key, version)`` *key* the root of SSA base key *base_key*?"""
    return (isinstance(base_key, tuple) and len(base_key) == 3 and base_key[0] == "ssa"
            and (base_key[1], base_key[2]) == key)


def _is_pointer_typed(v: Any) -> bool:
    """Does BN type the variable *v* (SSA or not) as a pointer?"""
    t = getattr(getattr(v, "var", v), "type", None)
    if t is None:
        return False
    tc = getattr(t, "type_class", None)
    if tc is not None:
        return "POINTER" in str(getattr(tc, "name", tc)).upper()
    return str(t).strip().endswith("*")


# The open end of a consumed region whose object has no recorded width: only a
# write of the whole variable can cover it.
_REGION_OPEN = 1 << 62


def _iv_add(ivs: tuple | None, lo: int, hi: int) -> tuple | None:
    """Byte intervals *ivs* (sorted, disjoint ``(lo, hi)`` pairs; None is the
    top element, every byte) with ``[lo, hi)`` added."""
    if ivs is None or hi <= lo:
        return ivs
    out: list[tuple[int, int]] = []
    for a, b in ivs:
        if b < lo or hi < a:
            out.append((a, b))
        else:
            lo, hi = min(lo, a), max(hi, b)
    out.append((lo, hi))
    return tuple(sorted(out))


def _iv_meet(x: tuple | None, y: tuple | None) -> tuple | None:
    """The bytes in both *x* and *y* (None is every byte)."""
    if x is None:
        return y
    if y is None:
        return x
    out: list[tuple[int, int]] = []
    for a, b in x:
        for c, d in y:
            lo, hi = max(a, c), min(b, d)
            if lo < hi:
                out.append((lo, hi))
    return tuple(sorted(out))


def _iv_gaps(ivs: tuple | None, lo: int, hi: int) -> list[tuple[int, int]]:
    """The parts of ``[lo, hi)`` that *ivs* does not contain."""
    if ivs is None:
        return []
    gaps: list[tuple[int, int]] = []
    cur = lo
    for a, b in ivs:
        if b <= cur:
            continue
        if a >= hi:
            break
        if a > cur:
            gaps.append((cur, a))
        cur = max(cur, b)
        if cur >= hi:
            break
    if cur < hi:
        gaps.append((cur, hi))
    return gaps



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
    ``sink.seed`` says what was seeded. ``expr`` is the call operand, whose
    must-offset into ``local`` is where the consumed region starts (#933)."""
    __slots__ = ("local", "addr", "name", "expr")

    def __init__(self, local: Any = None, *, addr: int | None = None, expr: Any = None):
        self.local = local
        self.addr = addr
        self.expr = expr
        self.name = f"&{var_label(local)}" if local is not None else hex(int(addr or 0))


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------

# The `resolve_map` default: no map was supplied. A distinct object, because a
# supplied JSON `null` document is invalid, not an omission.
_NO_RESOLVE_MAP: Any = object()

# Global-access kinds (`_global_access`) the walk SAW read the bytes, as
# opposed to the ones that only mean "it could" -- an address materialised
# with no classifiable use, a call whose handling of the bytes is unknown, a
# reference the scan could not find, a body whose IL could not be read, and
# the two ESCAPES (`return`, `address_escape`), which hand the address
# somewhere else rather than read through it. The frontier's prose says
# "reads" only for the first set (round-3 review F5/F17).
_PROVEN_GLOBAL_READS = frozenset({"load", "call_arg"})

# Sentinel for a memo whose value may legitimately be None.
_MISSING: Any = object()


def _code(text: Any) -> str:
    """*text* as inline code in leaf prose -- unless it carries a backtick
    (a function named with one), which would unbalance the framing and let
    the command router read a name as the start of a `bn ...` command; then
    it is JSON-quoted plain text (#931 round 8)."""
    t = str(text)
    return json.dumps(t) if "`" in t else f"`{t}`"


class _Use(NamedTuple):
    """One classified use of a tracked pointer inside one function.

    `offset_known` is the honest half of `offset`: a displacement the scan
    ESTABLISHED, versus one it could not, which must not render as the base
    (round-10 review M1). `span` is how many bytes the use touches, when the
    model and the call both say.
    """

    at: int | None
    kind: str
    offset: int
    seed: str | None
    span: int | None
    offset_known: bool


# What a reached pointer-scan bound is REPORTED as. Both are blocking: the
# scan did not finish, so neither "it only writes" nor "nothing is here" was
# established (BN21-BOUND-02/03).
_SCAN_CAUSE_KINDS = {
    "pointer_scan_budget": "scan_budget_exhausted",
    "pointer_return_depth": "pointer_depth_exhausted",
}
_SCAN_KIND_CAUSES = {v: k for k, v in _SCAN_CAUSE_KINDS.items()}


class _ScanBudget:
    """The node allowance ONE pointer scan may spend, and why it stopped.

    A bound that is reached is a fact about the answer, not a detail of the
    walk: node exhaustion and the internal return-pointer depth both used to
    read out as "nothing here" (BN21-BOUND-02/03), so the budget carries the
    cause back and the classifier turns it into a frontier."""

    __slots__ = ("nodes", "cause")

    # A pointer-return chain costs `_PTR_RETURN_STEP` levels per hop, so the
    # depth limit is a WRAPPER-CHAIN limit, not a CLI `--max-depth`. Reaching
    # it is reported under its own cause; raising `--max-depth` cannot move it.
    def __init__(self, nodes: int) -> None:
        self.nodes = int(nodes)
        self.cause: str | None = None

    def spend(self) -> bool:
        """Charge one node; False once the allowance is gone."""
        if self.nodes <= 0:
            self.cause = self.cause or "pointer_scan_budget"
            return False
        self.nodes -= 1
        return True

    def depth_exhausted(self) -> None:
        self.cause = self.cause or "pointer_return_depth"


class _SalvageList(list):
    """A findings/leaves list that mirrors every append into a run-level
    buffer, so a `RecursionError` unwinding the walk cannot DISCARD results
    the walk had already observed (round-3 review R3)."""

    __slots__ = ("_keep",)

    def __init__(self, keep: list) -> None:
        super().__init__()
        self._keep = keep

    def append(self, item: Any) -> None:
        super().append(item)
        self._keep.append(item)

    def extend(self, items: Any) -> None:
        items = list(items)
        super().extend(items)
        self._keep.extend(items)


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
        # Definite stash-slot proofs (`_definite_slot_store`), per function
        # and load: structural, so one answer serves every fixed-point sweep.
        self._slot_proofs: dict[Any, tuple[Any, dict[Any, Any]]] = {}
        # Global-memory continuation (IC03): BN's own code references to a
        # writable global, keyed by address, and the bounded call-graph
        # reachability answers derived from them. Both are facts about the
        # VIEW, so they outlive the per-source re-runs of `_forward_run`, and
        # both are only ever consulted for a global the walk already tainted --
        # a targeted xref query, not a whole-program solve.
        self._canon_global_cache: dict[int, int | None] = {}
        self._global_keyed_offsets: dict[int, set[int]] = {}
        self._global_write_span: dict[int, int] = {}
        self._global_tainted_ranges: dict[int, list[tuple]] = {}
        self._global_write_unbounded: set[int] = set()
        self._global_search_truncated: set[int] = set()
        self._touches_global_cache: dict[int, bool] = {}
        self._alias_chain_cache: dict[int, tuple] = {}
        self._global_extent_unproven: set[int] = set()
        self._ssa_def_cache: dict[tuple, Any] = {}
        self._returned_params_cache: dict[int, tuple] = {}
        self._returned_params_active: set[int] = set()
        self._return_offset_active: set[tuple] = set()
        self._global_refs: dict[int, dict[int, int]] = {}
        # globals whose reference sweep hit its chunk budget, so "no further
        # references" was never established for them.
        self._global_refs_truncated: set[int] = set()
        self._global_reach: dict[tuple[int, int, int], tuple] = {}
        # (function start, global) -> the first use that can READ the global,
        # or (None, "write_only"). Per function body, so a write-only helper
        # is classified once however many calls reach it.
        self._global_access_cache: dict[tuple[int, int], tuple] = {}
        # (function start, global) -> addresses a bounded native decode of an
        # unread body found the global at. Only ever filled for a callee whose
        # MLIL does not exist.
        self._native_refs_cache: dict[tuple[int, int], tuple] = {}
        # (callee start, parameter index) -> that parameter's own access
        # verdict inside the callee's body, so a helper called from many
        # sites is classified once.
        self._param_access_cache: dict[tuple[int, int], tuple] = {}
        # global address -> the object's byte extent, typed or inferred.
        self._global_extent_cache: dict[int, int] = {}
        # function start -> (indirect targets this run can pin, unresolved
        # indirect call addresses) inside that body.
        self._indirect_sites_cache: dict[int, tuple] = {}

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
            raw = self._global_addr(ssaf, expr, canon=False)
            if raw is not None and int(raw) != ga:
                # The key is the OBJECT; the offset it was reached at is kept
                # so the leaf can say which bytes were written, and so a
                # partial write is never read as covering the whole object.
                self._global_keyed_offsets.setdefault(ga, set()).add(int(raw) - ga)
            return (("global", ga), f"glob_{hex(ga)}")
        hk = self._heap_buffer_key(ssaf, expr)
        if hk is not None:
            return (hk, f"heap_{hex(hk[1])}")
        return None

    def _global_addr(self, ssaf: Any, expr: Any, depth: int = 0, *,
                     canon: bool = True) -> int | None:
        """The base address of a writable global/static buffer *expr* points at, or
        None. Follows an SSA copy chain to a constant pointer.

        ``canon=False`` returns the constant the expression actually names,
        without folding it to its data variable's base -- what an INTERIOR
        reference needs when the view has no type for the object and BN
        invented a second data variable at the referenced offset."""
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
            if not canon:
                return addr if self._is_writable(addr) else None
            return self._canon_global(addr)
        if is_ssa_var(expr):
            try:
                d = ssaf.get_ssa_var_definition(expr)
            except Exception:
                d = None
            if d is not None:
                return self._global_addr(ssaf, getattr(d, "src", None), depth + 1,
                                         canon=canon)
            return None
        reads = expr_reads(expr)
        if len(reads) == 1:
            try:
                d = ssaf.get_ssa_var_definition(reads[0])
            except Exception:
                d = None
            if d is not None:
                return self._global_addr(ssaf, getattr(d, "src", None), depth + 1,
                                         canon=canon)
        return None

    def _canon_global(self, addr: int) -> int | None:
        """Canonicalize a global address to its data-variable base (so different
        offsets into one buffer share a taint location) and require it be writable,
        so rodata strings / code pointers are not treated as buffers. Permissive
        when the BinaryView lacks these APIs (the unit-test fakes)."""
        cached = self._canon_global_cache.get(addr, _MISSING)
        if cached is not _MISSING:
            return cached
        out = self._canon_global_uncached(addr)
        self._canon_global_cache[addr] = out
        return out

    def _canon_global_uncached(self, addr: int) -> int | None:
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
            base = self._containing_object(base)
            return base if self._is_writable(base) else None
        return addr if self._is_writable(addr) else None

    def _containing_object(self, base: int) -> int:
        """Fold a nameless interior data variable onto the object it lies in.

        `memcpy(g + 8, src, 64)` makes BN invent a bare data variable at
        `g + 8`, so the write keyed a DIFFERENT location from the one
        `system(g)` reads and a real flow answered all-clear (round-3 review
        R1). An OBJECT is what carries a symbol; a nameless variable inside a
        symbolled object's extent belongs to it, and one that lies outside
        every extent stays itself, so disjoint neighbours never merge."""
        getsym = getattr(self.bv, "get_symbol_at", None)
        prev = getattr(self.bv, "get_previous_data_var_before", None)
        if not callable(getsym) or not callable(prev):
            return base            # a view that cannot say: claim nothing
        try:
            if getsym(base) is not None:
                return base        # already a named object
        except Exception:
            return base
        at = base
        for _ in range(64):
            try:
                got = prev(at)
            except Exception:
                return base
            addr = got if isinstance(got, int) else getattr(got, "address", None)
            if addr is None or int(addr) >= at:
                return base
            at = int(addr)
            try:
                named = getsym(at)
            except Exception:
                return base
            if named is None:
                continue           # another invented interior variable
            # The nearest NAMED predecessor: it owns `base` only if its own
            # extent reaches that far. `_global_extent` stops where the next
            # symbolled neighbour begins, so a separate object never swallows
            # this one.
            return at if at + self._global_extent(at) > base else base
        return base

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
        for literal operands (`_const_ptr_storage`).

        A format is known only when its terminating NUL was seen (#1050
        review round 8): BN's ASCII string stops at the first non-ASCII byte,
        and a bounded read can end mid-format, and either prefix silently drops
        later conversions. A string BN returns is accepted only when the byte
        right after it is NUL; the raw read must find the NUL within its window.
        Anything else is unreadable (None)."""
        addr = self._const_ptr_addr(ssaf, expr)
        if addr is None or self._const_ptr_storage(addr) == "writable":
            return None
        read = getattr(self.bv, "read", None)
        getstr = getattr(self.bv, "get_ascii_string_at", None)
        if getstr is not None:
            try:
                s = getstr(int(addr), 1)
                val = getattr(s, "value", None)
                if val is not None:
                    text = str(val)
                    width = len(text.encode("latin-1", "replace"))
                    if read is None or read(int(addr) + width, 1) == b"\x00":
                        return text
            except Exception:
                pass
        if read is not None:
            try:
                raw = read(int(addr), 4096)
                nul = raw.find(b"\x00") if raw else -1
                if nul >= 0:
                    return raw[:nul].decode("latin-1", "replace")
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

    def _pointer_width(self, size: Any) -> bool:
        """Is an access or value *size* the target's pointer width? An absent
        size (exact-copy IL need not carry one) is not a narrowing; a known
        size that differs is (#1034 item 2: an 8-byte stash reloaded as 4)."""
        if size is None:
            return True
        try:
            return int(size) == self._ptr_size()
        except (TypeError, ValueError):
            return False

    def _field_store_slot(self, ssaf: Any, ins: Any):
        """The ``(base, offset)`` slot a store writes, unifying BN's store shapes:
        MLIL_STORE_SSA (dest is an address expr `&v + off`), MLIL_STORE_STRUCT_SSA
        (`p->field = x`: the field offset is ``ins.offset``), MLIL_SET_VAR_(ALIASED_)FIELD
        (dest is the struct VAR -> `(v, 0)`, with the field byte offset in ``ins.offset``),
        and MLIL_SET_VAR_ALIASED (a whole address-taken scalar, e.g. a `size_t& capacity`
        out-param slot -> `(v, 0)`; #442). None when unresolvable."""
        op = op_name(ins)
        if "STORE" in op:
            return self._mem_access_slot(ssaf, ins)
        if op in ("MLIL_SET_VAR_FIELD", "MLIL_SET_VAR_ALIASED_FIELD"):
            ba = self._addr_base_offset(ssaf, getattr(ins, "dest", None))
            if ba is None:
                return None
            return (ba[0], ba[1] + int(getattr(ins, "offset", 0) or 0))
        if op == "MLIL_SET_VAR_ALIASED":
            dest = getattr(ins, "dest", None)
            return (var_key(dest), 0) if dest is not None else None
        return None

    def _mem_access_slot(self, ssaf: Any, access: Any, *, exact: bool = False):
        """The ``(base, offset)`` of the bytes a LOAD expression reads or a
        STORE instruction writes. BN's ``*_STRUCT`` forms (``p->lens+4.d``,
        ``p->lens = x``) keep the field offset in ``.offset`` beside a bare
        base, so the address alone names offset 0 of every field. *exact*
        resolves the address with :meth:`_exact_addr` -- the identity every
        certainty decision must use; the default is the may-oriented
        :meth:`_addr_base_offset` the forward correlation keys on."""
        op = op_name(access)
        addr = getattr(access, "src" if "LOAD" in op else "dest", None)
        ba = self._exact_addr(ssaf, addr) if exact else self._addr_base_offset(ssaf, addr)
        if ba is None or "_STRUCT" not in op:
            return ba
        try:
            return (ba[0], ba[1] + int(getattr(access, "offset", 0) or 0))
        except (TypeError, ValueError):
            return None

    def _exact_addr(self, ssaf: Any, expr: Any):
        """``(key, offset)`` naming the address *expr* computes EXACTLY, else
        None. Unlike :meth:`_addr_base_offset` it follows only value-preserving
        steps -- an SSA copy, a pointer-width ``VAR`` read, a constant offset --
        and never a conversion, a partial-register field (``slot#0.edi``) or
        a lone read inside an arbitrary expression: a narrowed or extended
        address is not the full one (#1053 review 1). An SSA root is keyed
        with its version (one value); a pointer loaded from a constant
        address is keyed with the memory version it was loaded at, because
        the slot may be rewritten between two loads of it. An address built
        on a ``VAR_PHI`` (a loop cursor, `key_1 = phi(key, key + 1)`) names a
        different address on every trip, so no store can be matched to it."""
        res = self._exact_addr_inner(ssaf, expr, 0)
        return None if res == "phi" else res

    def _exact_addr_inner(self, ssaf: Any, expr: Any, depth: int):
        """:meth:`_exact_addr`, with ``"phi"`` for a phi-rooted address."""
        if expr is None or depth > 12:
            return None
        op = op_name(expr)
        if "ADDRESS_OF" in op:
            v = getattr(expr, "src", None) or getattr(expr, "var", None)
            if v is None:
                return None
            try:
                off = int(getattr(expr, "offset", 0) or 0) if op == "MLIL_ADDRESS_OF_FIELD" else 0
            except (TypeError, ValueError):
                return None
            return (var_key(v), off)
        if "CONST_PTR" in op:
            c = self._int_const(expr)
            return (("gconst", c), 0) if c is not None else None
        if "LOAD" in op:
            a = getattr(expr, "src", None)
            c = self._int_const(a) if a is not None and "CONST_PTR" in op_name(a) else None
            if c is None or not self._pointer_width(getattr(expr, "size", None)) \
                    or "_STRUCT" in op:
                return None
            return (("gload", c, getattr(expr, "src_memory", None)), 0)
        if op in ("MLIL_ADD", "MLIL_SUB"):
            left, right = getattr(expr, "left", None), getattr(expr, "right", None)
            rc, lc = self._int_const(right), self._int_const(left)
            if rc is not None:
                base, delta = self._exact_addr_inner(ssaf, left, depth + 1), (
                    rc if op == "MLIL_ADD" else -rc)
            elif op == "MLIL_ADD" and lc is not None:
                base, delta = self._exact_addr_inner(ssaf, right, depth + 1), lc
            else:
                return None
            return base if base is None or base == "phi" else (base[0], base[1] + delta)
        if is_ssa_var(expr):
            try:
                d = ssaf.get_ssa_var_definition(expr)
            except Exception:
                d = None
            dop = op_name(d) if d is not None else None
            if dop == "MLIL_VAR_PHI":
                return "phi"
            if dop == "MLIL_SET_VAR_SSA":
                sub = self._exact_addr_inner(ssaf, getattr(d, "src", None), depth + 1)
                if sub is not None:
                    return sub
            # A fresh value (a parameter, a call result, a conversion): the
            # SSA name is its identity.
            return (("ssa", var_key(expr), getattr(expr, "version", None)), 0)
        reads = expr_reads(expr)
        if op in _COPY_OPS and len(reads) == 1 and self._pointer_width(getattr(expr, "size", None)):
            return self._exact_addr_inner(ssaf, reads[0], depth + 1)
        return None

    def _gconst_addr(self, slot: Any) -> int | None:
        """The absolute address an exact ``gconst`` slot names, else None."""
        if slot is None:
            return None
        try:
            return int(slot[0][1]) + int(slot[1]) if slot[0][0] == "gconst" else None
        except (TypeError, ValueError, IndexError):
            return None

    def _addr_root_kind(self, ssaf: Any, expr: Any, depth: int = 0) -> str | None:
        """What the base ``_addr_base_offset`` resolves *expr* to IS: ``"local"``
        (the address of a stack variable), ``"abs"`` (a constant address) or
        ``"ptr"`` (a pointer value -- a parameter, a call result, a loaded
        pointer -- that may point anywhere). None when unresolved. Two distinct
        locals, or a local and a constant address, are distinct objects; a
        pointer value can alias either."""
        if expr is None or depth > 8:
            return None
        op = op_name(expr)
        if "ADDRESS_OF" in op:
            return "local"
        if "LOAD" in op:
            return "ptr"
        if "CONST_PTR" in op:
            return "abs"
        if op in ("MLIL_ADD", "MLIL_SUB"):
            left, right = getattr(expr, "left", None), getattr(expr, "right", None)
            if self._int_const(right) is not None:
                return self._addr_root_kind(ssaf, left, depth + 1)
            if op == "MLIL_ADD" and self._int_const(left) is not None:
                return self._addr_root_kind(ssaf, right, depth + 1)
            return None
        if is_ssa_var(expr):
            try:
                d = ssaf.get_ssa_var_definition(expr)
            except Exception:
                d = None
            if d is not None and op_name(d) == "MLIL_SET_VAR_SSA":
                sub = self._addr_root_kind(ssaf, getattr(d, "src", None), depth + 1)
                if sub is not None:
                    return sub
            return "ptr"
        if self._outer_loads(expr):
            return "ptr"        # a converted loaded pointer may point anywhere
        reads = expr_reads(expr)
        if len(reads) == 1:
            return self._addr_root_kind(ssaf, reads[0], depth + 1)
        return None

    def _store_load_relation(self, ssaf: Any, store: Any, load: Any, la: tuple) -> str:
        """How a store's bytes relate to the bytes *load* reads at the EXACT
        slot *la*: ``"cover"`` / ``"overlap"`` / ``"disjoint"`` on a shared
        base, else ``"disjoint"`` only for provably distinct objects (two
        different locals, a local and a constant address, two non-overlapping
        constant ranges) and ``"may"`` otherwise -- a store through another
        pointer or through an address that is not exactly resolved
        (:meth:`_exact_addr`) can write the loaded bytes."""
        dest = getattr(store, "dest", None)
        sa = self._mem_access_slot(ssaf, store, exact=True)
        try:
            lw = int(getattr(load, "size", None)) if getattr(load, "size", None) is not None else None
            sw = int(getattr(store, "size", None)) if getattr(store, "size", None) is not None else None
        except (TypeError, ValueError):
            return "may"
        if sa is not None and sa[0] == la[0]:
            return self._store_covers_load(sa[1], sw, la[1], lw)
        if sa is None:
            return "may"
        sk = self._addr_root_kind(ssaf, dest)
        lk = self._addr_root_kind(ssaf, getattr(load, "src", None))
        if {sk, lk} == {"local", "abs"} or (sk == lk == "local"):
            return "disjoint"
        s_lo, l_lo = self._gconst_addr(sa), self._gconst_addr(la)
        if s_lo is not None and l_lo is not None and lw is not None \
                and (l_lo + lw <= s_lo or (sw is not None and s_lo + sw <= l_lo)):
            return "disjoint"
        return "may"

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

    def _call_load_effect(self, call_ins: Any):
        """What the call *call_ins* may have done to bytes a later load reads:
        every call may have written them. No model enumerates all of its
        callee's memory writes -- a failing read sets errno, a getenv may copy
        into an internal buffer, strtok rewrites a delimiter, stat fills its
        struct (#1053 review rounds 1-3) -- so no call is proof of "no write"
        and no call's write extent is trusted. A modeled source writing
        through an ``*arg:N`` is named as a possible producer:
        ``("may_source", callee)``; anything else is ``("opaque",)``. Either
        way the loaded bytes are not accounted for."""
        name = self._callee_name(self._resolve_direct_target(call_ins))
        mkey, model = lookup_model(self.models, name)
        if model and not model_is_benign_marker(model) and any(
                str(s.get("to") or "").startswith("*arg:") for s in (model.get("sources") or [])):
            return ("may_source", mkey or name)
        return ("opaque",)

    def _load_writers(self, ssaf: Any, load: Any, la: tuple) -> list[tuple]:
        """Every memory definition that may supply bytes the LOAD expression
        *load* reads at EXACT slot *la* (:meth:`_exact_addr`), over EVERY
        memory-SSA path back from it -- never just the first phi arm that
        resolves (a ``if (c) p->n = x; use(p->n)`` load is ``x`` on one path
        and whatever the caller left on the other). Each entry is headed by
        its kind:

        * ``("store", defn)`` -- a store that covers every loaded byte; the
          path stops there;
        * ``("partial", defn)`` -- a store overlapping only some loaded bytes
          (another width or offset on the same base, #1041 item 2). It
          supplies those bytes, so it is a writer, and the path continues for
          the rest -- a narrow untainted store never masks a wider writer;
        * ``("may", defn)`` -- a store the walk cannot prove disjoint (through
          another pointer, or an address it cannot resolve exactly); the path
          continues past it;
        * ``("may_source", call, callee)`` -- a modeled source that may have
          filled them: named as a possible producer, and the path continues
          past it -- but no write extent is trusted, so it is a frontier;
        * ``("opaque", defn)`` -- any other memory definition that may write
          them: every other call (no call is proof of "no write",
          :meth:`_call_load_effect`) or an intrinsic; the path stops;
        * ``("entry", None)`` -- the path reaches the function's entry memory:
          the bytes are whatever the caller left;
        * ``("cut", None)`` -- ``MEM_CHAIN_BUDGET`` ran out.

        A loop back-edge to a version already walked adds nothing new."""
        out: list[tuple] = []
        seen: set[int] = set()
        pending = [getattr(load, "src_memory", None)]
        lw = getattr(load, "size", None)
        lk = self._addr_root_kind(ssaf, getattr(load, "src", None))
        while pending:
            mv = pending.pop()
            try:
                mv = int(mv)
            except (TypeError, ValueError):
                out.append(("entry", None))
                continue
            if mv in seen:
                continue
            seen.add(mv)
            if len(seen) > MEM_CHAIN_BUDGET:
                out.append(("cut", None))
                break
            try:
                defn = ssaf.get_ssa_memory_definition(mv)
            except Exception:
                defn = None
            if defn is None:
                out.append(("entry", None))
                continue
            op = op_name(defn)
            if "MEM_PHI" in op:
                arms = self._mem_phi_sources(defn)
                if not arms:
                    out.append(("entry", None))
                pending.extend(arms)
                continue
            if "STORE" in op or op in _NAMED_STORE_OPS:
                if "STORE" in op:
                    rel = self._store_load_relation(ssaf, defn, load, la)
                else:
                    slot = self._field_store_slot(ssaf, defn)
                    if slot is not None and slot[0] == la[0]:
                        rel = self._store_covers_load(
                            slot[1], getattr(defn, "size", None), la[1], lw)
                    else:
                        # A stack variable written by name: another local or a
                        # constant address cannot be it; a pointer may.
                        rel = "disjoint" if lk in ("local", "abs") else "may"
                if rel == "cover":
                    out.append(("store", defn))
                    continue
                if rel == "overlap":
                    out.append(("partial", defn))
                elif rel == "may":
                    out.append(("may", defn))
                pending.append(self._mem_in(defn))
                continue
            if self._is_call(defn):
                eff = self._call_load_effect(defn)
                if eff[0] == "may_source":
                    out.append(("may_source", defn, eff[1]))
                    pending.append(self._mem_in(defn))
                    continue
            out.append(("opaque", defn))
        return out

    def _definite_slot_store(self, ssaf: Any, load: Any) -> Any | None:
        """The one store whose value a pointer-sized LOAD reloads on every
        path, else None -- definite stash identity, not a may-alias fact.

        The memory chain from the load must be straight back to a store of
        exactly the reloaded slot at pointer width, through nothing that may
        write it: every store in between is provably disjoint (another
        local, another constant range, the same base at another offset). A
        phi, a call, a store that overlaps or may alias the slot, a narrowed
        reload or a narrow/wide store leave the reloaded value unproven
        (#1034 items 1 and 2). Slot identity is EXACT (:meth:`_exact_addr`):
        a store through ``slot#0.edi`` is not a store to ``[slot#0]``
        (#1053 review 1).

        The answer depends only on the IL, so it is cached per function and
        load -- misses included: the escape fixed point asks again for every
        reload on every sweep (#1053 review 5)."""
        start = getattr(getattr(ssaf, "source_function", None), "start", None)
        fkey = ("fn", start) if isinstance(start, int) else ("ssa", id(ssaf))
        holder = self._slot_proofs.get(fkey)
        if holder is None or (fkey[0] == "ssa" and holder[0] is not ssaf):
            holder = (ssaf, {})
            self._slot_proofs[fkey] = holder
        idx = getattr(load, "expr_index", None)
        mem = getattr(load, "src_memory", None)
        key = ("expr", idx, mem) if idx is not None else ("obj", id(load), mem)
        hit = holder[1].get(key)
        if hit is None or (key[0] == "obj" and hit[0] is not load):
            hit = (load, self._definite_slot_store_uncached(ssaf, load, holder[1]))
            holder[1][key] = hit
        return hit[1]

    def _definite_slot_store_uncached(self, ssaf: Any, load: Any, memo: dict) -> Any | None:
        """The chain walk behind :meth:`_definite_slot_store`. Its answer
        from a memory version depends only on the exact slot and width, so
        each version is classified once per slot (*memo*), however many
        reloads walk through it."""
        if not self._pointer_width(getattr(load, "size", None)):
            return None
        la = self._mem_access_slot(ssaf, load, exact=True)
        if la is None:
            return None
        lw = getattr(load, "size", None)
        lk = self._addr_root_kind(ssaf, getattr(load, "src", None))
        version = getattr(load, "src_memory", None)
        walked: list[Any] = []
        found: Any = None
        for _ in range(MEM_CHAIN_BUDGET):
            mkey = ("chain", la, lw, version)
            if mkey in memo:
                found = memo[mkey]
                break
            walked.append(mkey)
            try:
                defn = ssaf.get_ssa_memory_definition(int(version))
            except Exception:   # noqa: BLE001 - an unreadable chain proves nothing
                break
            if defn is None:
                break
            op = op_name(defn)
            if "STORE" in op:
                rel = self._store_load_relation(ssaf, defn, load, la)
                slot = self._mem_access_slot(ssaf, defn, exact=True)
            elif op in _NAMED_STORE_OPS:
                slot = self._field_store_slot(ssaf, defn)
                if slot is not None and slot[0] == la[0]:
                    rel = self._store_covers_load(slot[1], getattr(defn, "size", None), la[1], lw)
                else:
                    rel = "disjoint" if lk in ("local", "abs") else "may"
            else:
                break
            if rel == "disjoint":
                version = self._mem_in(defn)
                continue
            if rel == "cover" and slot is not None and slot[1] == la[1] \
                    and self._pointer_width(getattr(defn, "size", None)):
                found = defn
            break
        for mkey in walked:
            memo[mkey] = found
        return found

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

    @staticmethod
    def _outer_loads(expr: Any, depth: int = 0) -> list[Any]:
        """The memory reads a value expression is built from: every LOAD
        sub-expression not nested inside another load's address -- the bare
        ``[p].d``, the ``[p + 0xc].d`` under ``sx.q(...)`` (#1041 item 2),
        both operands of ``[a].d + [b].d``."""
        if expr is None or depth > 8:
            return []
        op = op_name(expr)
        if "LOAD" in op:
            return [expr]
        if "ADDRESS_OF" in op or "CONST" in op:
            return []
        kids = [c for c in (getattr(expr, a, None) for a in ("src", "left", "right"))
                if hasattr(c, "operation")]
        if not kids:
            kids = [c for c in (getattr(expr, "operands", None) or [])
                    if hasattr(c, "operation")]
        out: list[Any] = []
        for c in kids:
            out.extend(TaintEngine._outer_loads(c, depth + 1))
        return out

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
        seeds = [_BufferSeed(loc, expr=expr) for loc in locals_]
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
            # A reload of a slot: a must-alias only when ONE store reaches
            # this load on every path, wrote exactly the reloaded pointer-
            # sized slot, and wrote a must-alias of &base (`g_stash = buf;
            # strcpy(g_stash, host)`). A narrowed reload, a call, a phi or a
            # store that may alias the slot in between leaves it unproven.
            st = self._definite_slot_store(ssaf, expr)
            if st is None:
                return None
            return self._must_offset(ssaf, getattr(st, "src", None), base_key,
                                     visited, depth + 1)
        if is_ssa_var(expr):
            key = (var_key(expr), getattr(expr, "version", None))
            if _is_ssa_root(base_key, key):
                return 0
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
        if len(reads) == 1 and op in _COPY_OPS and self._pointer_width(getattr(expr, "size", None)):
            # A value-preserving copy of one pointer variable. A conversion
            # (`zx.q(rax.d)`, a LOW_PART or partial-register read) is not the
            # pointer: its value can name any address (#1034 item 2).
            return self._must_offset(ssaf, reads[0], base_key, visited, depth + 1)
        return None

    def _builtin_store_extent(self, defn: Any, name: str | None, params: list[Any]) -> int | None:
        """Bytes a ``__builtin_*`` call writes from its destination, when BN
        outlined it from inline stores and those stores say so; else None.

        BN renders an inline store run (``mov qword [rsp], rax`` ...) as a
        call such as ``__builtin_memset(&cmd, 0, 0x40)``. Its length is
        trusted only after the fact (#931): the call's own LLIL must hold no
        call, jump or intrinsic, and its stores, all relative to one SSA
        base register, must write one contiguous run exactly as long as the
        length the arguments state (a constant count, or a literal source's
        bytes up to and including its NUL). A real call to the libc function
        (an ``LLIL_CALL``) never qualifies, nor does a run with a gap."""
        if not name or not name.startswith("__builtin_"):
            return None
        stated: int | None = None
        if name in _BUILTIN_LEN_ARG:
            i = _BUILTIN_LEN_ARG[name]
            if i < len(params) and op_name(params[i]) == "MLIL_CONST":
                stated = self._int_const(params[i])
        elif name == "__builtin_strcpy" and len(params) > 1 \
                and op_name(params[1]) == "MLIL_CONST_DATA":
            try:
                data = bytes(params[1].constant_data.data)
            except Exception:
                data = b""
            if b"\x00" in data:
                stated = data.index(b"\x00") + 1
        if stated is None or stated <= 0:
            return None
        try:
            llils = list(getattr(defn, "llils", None) or [])
        except Exception:
            return None
        base = None
        spans: tuple | None = ()
        for ll in llils:
            op = op_name(ll)
            if any(t in op for t in ("CALL", "SYSCALL", "INTRINSIC", "JUMP", "GOTO",
                                     "RET", "IF", "TRAP", "UNDEF", "UNIMPL")):
                return None
            if not op.startswith("LLIL_STORE"):
                continue
            dest = getattr(ll, "dest", None)
            dop = op_name(dest)
            off = 0
            if dop in ("LLIL_ADD", "LLIL_SUB"):
                c = self._int_const(getattr(dest, "right", None))
                if c is None:
                    return None
                off = c if dop == "LLIL_ADD" else -c
                dest = getattr(dest, "left", None)
                dop = op_name(dest)
            if dop != "LLIL_REG_SSA":
                return None
            key = str(dest)
            if base is None:
                base = key
            elif key != base:
                return None
            try:
                width = int(getattr(ll, "size", 0))
            except (TypeError, ValueError):
                return None
            if width <= 0:
                return None
            spans = _iv_add(spans, off, off + width)
        if base is None or not spans or len(spans) != 1:
            return None
        lo, hi = spans[0]
        return stated if hi - lo == stated else None

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
                           escaped_at: Any = None,
                           visited: set | None = None,
                           depth: int = 0) -> str:
        """Classify possible aliases of local buffer *base_key*.

        ``"yes"`` is an address-derived pointer: ``&base`` through copies,
        arithmetic and phis, the value a definite stash store wrote
        (``_definite_slot_store``), or a pointer-sized reload of a constant
        slot remembered at the LOAD's own memory version as holding a
        ``"yes"`` value. A remembered slot is may-alias evidence: it is killed
        by a store that overwrites it, and a value that only MAY alias the
        buffer never makes a slot "yes" (#1034 item 1).

        A LOAD is judged at its OWN memory epoch: *escaped_at(version,
        default)* says whether the address had escaped by the version the
        load read, so a pointer loaded before ``&buf`` was stored anywhere is
        "no" even when it is copied through another slot later. *escaped* is
        the epoch of the consuming writer and governs everything else.

        A conversion of a pointer (``zx.q(rax.d)``, a partial-register read)
        and a narrowed reload are never "yes": the converted value may name
        any address, so they are "maybe" (#1034 item 2).

        ``"no"`` excludes known unrelated addresses and values fixed before
        this frame existed; ``"maybe"`` retains unresolved aliases after an
        escape or a depth cut. The walk must disclose those, never drop them."""
        def esc_at(version: Any) -> bool:
            if escaped_at is None:
                return escaped
            return bool(escaped_at(version, escaped))

        def rec(e: Any, *, seen: set | None = None) -> str:
            return self._ptr_aliases_local(
                ssaf, e, base_key, escaped=escaped, escape_addrs=escape_addrs,
                escaped_at=escaped_at, visited=visited if seen is None else seen,
                depth=depth + 1)

        if expr is None:
            return "no"
        if depth > 12:
            return "maybe"
        op = op_name(expr)
        if "ADDRESS_OF" in op:
            v = getattr(expr, "src", None) or getattr(expr, "var", None)
            return "yes" if v is not None and var_key(v) == base_key else "no"
        if "LOAD" in op:
            st = self._definite_slot_store(ssaf, expr)
            if st is not None:
                return rec(getattr(st, "src", None))
            mv = getattr(expr, "src_memory", None)
            c = self._gconst_addr(self._mem_access_slot(ssaf, expr, exact=True))
            if c is not None:
                try:
                    held = escape_addrs.get(int(mv), frozenset())
                except (TypeError, ValueError):
                    held = frozenset()
                if c in held:
                    return "yes" if self._pointer_width(getattr(expr, "size", None)) else "maybe"
            return "maybe" if esc_at(mv) else "no"
        if op in ("MLIL_VAR_ALIASED", "MLIL_VAR_ALIASED_FIELD"):
            # A read of an address-taken local BY NAME (`p = end` after
            # `strtol(buf, &end, 10)`) is a memory read: a call handed both
            # may have planted a pointer into the buffer there (#1053 review 4).
            return "maybe" if escaped else "no"
        if "CONST" in op or "EXTERN_PTR" in op or "IMPORT" in op:
            return "no"
        if is_ssa_var(expr):
            key = (var_key(expr), getattr(expr, "version", None))
            if _is_ssa_root(base_key, key):
                return "yes"    # the pointer object's own root value
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
                return rec(getattr(d, "src", None))
            if dop == "MLIL_VAR_PHI":
                arms = [rec(a) for a in (getattr(d, "src", None) or [])]
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
                handed = [rec(p, seen=set(visited)) for p in params]
                if model is None:
                    if "yes" in handed or "maybe" in handed or escaped:
                        return "maybe"
                    return "no"
                pointee = False
                for rule in model.get("propagates") or []:
                    if str(rule.get("to") or "") != "ret":
                        continue
                    src = str(rule.get("from") or "")
                    fi = _try_arg_index(src)
                    if fi is None or fi >= len(handed):
                        continue
                    if handed[fi] != "no":
                        return handed[fi]
                    pointee = pointee or src.startswith("*")
                if pointee and esc_at(self._mem_in(d)):
                    # `*arg:N -> ret` derives the return from what the
                    # argument POINTS AT, not the argument: strsep(&p)
                    # returns the pointer stored in p, strtok_r continues
                    # from its save slot. Once &buf has escaped, that memory
                    # may hold a pointer into the buffer (S2).
                    return "maybe"
                return "no"
            if dop == "MLIL_SET_VAR_SSA_FIELD":
                # A field of a non-address-taken struct (or a partial
                # register) set by a plain store: the variable is built on
                # the stored value AND its previous version (S1). A pointer
                # read back out of it may be the stored one; it is never the
                # whole value, so the answer is at most "maybe".
                arms = [rec(getattr(d, "src", None)), rec(getattr(d, "prev", None))]
                return "maybe" if "yes" in arms or "maybe" in arms else "no"
            # Any other definition (a partial register write, a field read):
            # not a pointer the walk can follow.
            return "maybe" if escaped else "no"
        if op in ("MLIL_ADD", "MLIL_SUB"):
            # Pointer arithmetic: the pointer is whichever side is built on
            # &base. Anything else (a xor, a multiply, a narrowing) is a value,
            # not an address.
            answers = [rec(getattr(expr, a, None)) for a in ("left", "right")]
            if "yes" in answers:
                return "yes"
            return "maybe" if "maybe" in answers else "no"
        reads = expr_reads(expr)
        loads = self._outer_loads(expr)
        if loads:
            # A conversion of a loaded value (`zx.q([stash].d)`): the reads
            # inside the loads are addresses, not the value, so classify the
            # loads themselves; a converted pointer may point anywhere.
            rest = list(reads)
            for ld in loads:
                for r in expr_reads(ld):
                    if r in rest:
                        rest.remove(r)
            answers = [rec(ld) for ld in loads] + [rec(r) for r in rest]
            return "maybe" if "yes" in answers or "maybe" in answers else "no"
        if len(reads) == 1:
            # A single-variable wrapper: a copy keeps the answer; a conversion
            # of a pointer into the buffer may point anywhere.
            inner = rec(reads[0])
            if inner == "yes" and not (op in _COPY_OPS
                                       and self._pointer_width(getattr(expr, "size", None))):
                return "maybe"
            return inner
        return "no"

    def _local_buffer_writers(self, ssaf: Any, base_key: Any, mv: Any,
                              base_size: int | None = None,
                              region: tuple[int, int] | None = None) -> dict[str, Any]:
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
        * ``("fill", defn, mkey, params, from_idxs, is_source, len_idxs, callee)`` --
          a modeled call writing ``*arg:N`` where ``params[N]`` resolves to
          the buffer (directly, through a phi'd pointer, or by reloading the
          slot ``&buf`` escaped into); ``from_idxs`` are the argument indices
          the model derives those bytes from (``propagates`` ``from`` + the
          varargs run), ``len_idxs`` the length arguments that decide how
          many of them (a tainted length cuts the command), ``is_source``
          when a ``sources`` rule fills it (a receive API), ``callee`` the
          raw symbol (for a runnable ``bn callsites`` hint);
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

        ``covered`` is False when, on some path from the sink back to the
        function entry, some byte of the CONSUMED REGION (*region*, ``(lo,
        hi)`` offsets into the local; the whole object by default) is not
        proven written -- the ``if (flag) fill(buf); sink(buf)`` shape, whose
        memory phi has a bare entry-state arm: the sink then reads whatever
        the stack held, and a slice that names only the fill's origin is not
        complete. Coverage is decided by the BYTES a writer provably writes,
        never by its kind (round 4 P2), and only a plain store the walk can
        see proves them (#931): a store whose address is a must-alias
        (``_must_offset`` / an exact slot) of ``&base + k`` writes ``[k, k +
        width)``, a whole-variable ``SET_VAR_ALIASED`` writes all of it, and
        so does a ``__builtin_*`` call BN outlined from inline stores
        (``_builtin_store_extent`` checks its LLIL). No other call proves a
        byte: a requested length is not the number of bytes a read
        delivered, an append does not initialise what precedes it, and a
        formatter can fail -- such calls are writers that cover nothing,
        and their inputs are still walked. An escape writes no byte; an
        indexed store (``buf[n] = 0``), a may-alias writer (a phi that may
        point at another local, a maybe-store) and a ``{}``-modeled call
        cover nothing either. Coverage is the greatest fixed point of the
        all-arm equations over byte intervals (an arm meets by
        intersection, a store adds its interval); escape is the least fixed
        point of the any-arm equations, including loop-carried stores of
        the address. ``truncated`` is True when graph discovery or either
        fixed point exhausts the shared ``MEM_CHAIN_BUDGET``; no incomplete
        solve certifies coverage. ``region`` is the consumed region the
        proof was asked about and ``unproven`` its sub-ranges not proven
        written on every path (None after a cut). Deliberately does NOT stop
        at a full-cover writer: ``strcat`` after ``strcpy`` both feed the
        sink, and a may-analysis that drops the earlier one loses an origin.
        A partial store never masks anything either (mirrors
        ``_load_writers``)."""
        if base_size is not None and base_size <= 0:
            base_size = None
        end = base_size if base_size is not None else _REGION_OPEN
        r_lo, r_hi = region if region is not None else (0, end)
        if r_lo < 0 or r_lo >= end or r_hi <= r_lo:
            r_lo, r_hi = 0, end
        writers: list[tuple] = []     # oldest first while walking; reversed on return
        # Discover the graph before solving it; no provisional cycle proof is
        # ever published. Coverage descends from top (None: every byte),
        # escape ascends from bottom.
        memo: dict[int, tuple[tuple | None, bool]] = {}
        truncated = False
        # Constant stash slots that may hold a "yes" pointer into the buffer,
        # keyed by memory version: added by a pointer-width store of a "yes"
        # value, killed by any other store that writes the slot (#1034 item 1).
        escape_addrs: dict[int, frozenset[int]] = {}
        # Addresses of `__builtin_*` fills whose stores were verified: they
        # are stores, not calls, for the disclosure.
        proven_fills: set[int] = set()

        def _escaped_at(version: Any, default: bool) -> bool:
            """Had the address escaped by memory *version*? A LOAD's own
            epoch, so a pointer read before the escape stays unrelated."""
            try:
                return memo[int(version)][1]
            except (KeyError, TypeError, ValueError):
                return default

        def _alias(expr: Any, escaped: bool) -> str:
            return self._ptr_aliases_local(ssaf, expr, base_key, escaped=escaped,
                                           escape_addrs=escape_addrs,
                                           escaped_at=_escaped_at)

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

        def _cover(ivs: tuple | None, k: int | None, width: Any) -> tuple | None:
            """*ivs* plus the bytes ``[k, k + width)``, clipped to the region."""
            try:
                w = int(width)
            except (TypeError, ValueError):
                return ivs
            if k is None or w <= 0:
                return ivs
            return _iv_add(ivs, max(k, r_lo), min(k + w, r_hi))

        def _store_cover(ivs: tuple | None, defn: Any, op: str, slot: tuple) -> tuple | None:
            """*ivs* plus the bytes a store keyed to the buffer writes: all
            of it for a whole-variable ``SET_VAR_ALIASED``; ``[k, k + width)``
            for a named field store, or for an address store whose EXACT slot
            is ``&base + k`` (a may-oriented slot names no byte)."""
            if op == "MLIL_SET_VAR_ALIASED":
                return _iv_add(ivs, r_lo, r_hi)
            if "STORE" in op:
                ex = self._mem_access_slot(ssaf, defn, exact=True)
                k = ex[1] if ex is not None and ex[0] == base_key else None
            else:
                k = slot[1]
            return _cover(ivs, k if isinstance(k, int) else None, getattr(defn, "size", None))

        def _call_writer(defn: Any, escaped: bool) -> tuple[int, int] | None:
            """Record *defn* (a call) as a writer when it can be one. Returns
            the bytes it proves written -- only for a ``__builtin_*`` call
            BN outlined from stores it verifies (``_builtin_store_extent``);
            any other call covers nothing (#931)."""
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
                    return None
                writers.append(("opaque", defn, callee, why))
                return None
            if not yes and not maybe:
                # Trusted to its model: a libc model declares what the callee
                # writes, and none of its pointers reaches the buffer -- not
                # even by reloading the slot the address escaped into.
                return None
            outs = self._model_out_arg_indices(model)
            if model_is_benign_marker(model):
                # `{}` promises no taint flows through the call, nothing about
                # whether it writes its pointer arguments (round 3: memset's
                # kin were `{}` and read as "writes nothing"). Recorded, so the
                # call is in the slice; it neither covers nor originates.
                writers.append(("benign", defn, callee))
                return None
            if any(i in maybe for i in outs):
                writers.append(("opaque", defn, callee,
                                "writes through a pointer argument the walk cannot "
                                "resolve (it may alias the escaped address of)"))
                return None
            hit = [i for i in outs if i in yes]
            if not hit:
                return None
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
            writers.append(("fill", defn, mkey or name, params, from_idxs, is_source,
                            len_idxs, callee))
            if 0 not in hit:
                return None
            k = self._must_offset(ssaf, params[0], base_key)
            ext = self._builtin_store_extent(defn, name, params)
            if k is None or ext is None:
                return None
            proven_fills.add(int(getattr(defn, "address", 0) or 0))
            return (k, k + ext)

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
            memo[v] = (None, False)
            escape_addrs[v] = frozenset()
            pending.extend(deps)

        ptr_size = self._ptr_size()

        def _kill(addrs: frozenset[int], dest: Any, exact: Any,
                  length: Any) -> frozenset[int]:
            """Remembered slots left after a STORE of *length* bytes through
            *dest* (exact slot *exact*): those it may overwrite are forgotten.
            A store at an exact constant address kills the slots it overlaps;
            a store through the address of a stack local -- including a
            reload PROVEN to hold `&buf` -- cannot reach a global slot; any
            other store, through a pointer the walk cannot pin, may overwrite
            every one. (Calls forget every slot; see ``transfer``.)"""
            if not addrs:
                return addrs
            lo = self._gconst_addr(exact)
            if lo is None:
                if self._addr_root_kind(ssaf, dest) == "local" \
                        or self._must_offset(ssaf, dest, base_key) is not None:
                    return addrs
                return frozenset()
            try:
                hi = lo + int(length) if length is not None else None
            except (TypeError, ValueError):
                hi = None
            return frozenset(s for s in addrs if s + ptr_size <= lo or (hi is not None and hi <= s))

        def transfer(v: int) -> tuple[tuple | None, bool, frozenset[int]]:
            defn, deps = graph[v]
            if defn is None:
                res = ((), False, frozenset())
            elif "MEM_PHI" in op_name(defn):
                arms = [memo.get(s, ((), False)) for s in deps]
                cov: tuple | None = () if not arms else None
                for c, _ in arms:
                    cov = _iv_meet(cov, c)
                res = (cov, any(e for _, e in arms),
                       frozenset().union(*(escape_addrs.get(s, frozenset()) for s in deps)))
            else:
                cov, esc = memo.get(deps[0], ((), False))
                addrs = escape_addrs.get(deps[0], frozenset())
                op = op_name(defn)
                if "STORE" in op or op in _NAMED_STORE_OPS:
                    slot = self._field_store_slot(ssaf, defn)
                    if slot is not None and slot[0] == base_key:
                        writers.append(("store", defn))
                        cov = _store_cover(cov, defn, op, slot)
                    elif "STORE" in op:
                        # Not keyed to the buffer: the address may still be
                        # built on it (a phi'd loop cursor, `[p + i]` with
                        # `p = &buf`), or be a pointer the walk cannot pin.
                        a = _alias(getattr(defn, "dest", None), esc)
                        if a == "yes":
                            writers.append(("store", defn))
                            # Through a pointer built on &buf: covers its
                            # bytes only at a must-alias offset -- an indexed
                            # `buf[n] = 0` writes one byte nobody can name.
                            cov = _cover(cov, self._must_offset(
                                ssaf, getattr(defn, "dest", None), base_key),
                                getattr(defn, "size", None))
                        elif a == "maybe":
                            # Disclosed by its own blocking leaf; it covers
                            # nothing -- it may write elsewhere.
                            writers.append(("maybe_store", defn,
                                            "stores through a pointer the walk cannot "
                                            "resolve, which may alias"))
                    val = getattr(defn, "src", None)
                    held = _alias(val, esc)
                    if "STORE" in op:
                        # The store overwrites every remembered slot it may
                        # touch; it remembers its own slot only for a
                        # pointer-width "yes" value at an EXACT constant
                        # address. A value that merely MAY alias the buffer
                        # is an escape, never a definite stash (#1034 item 1).
                        exact = self._mem_access_slot(ssaf, defn, exact=True)
                        lo = self._gconst_addr(exact)
                        addrs = _kill(addrs, getattr(defn, "dest", None), exact,
                                      getattr(defn, "size", None) or self._ptr_size())
                        if lo is not None and held == "yes" \
                                and self._pointer_width(getattr(defn, "size", None)):
                            addrs = addrs | {lo}
                    if held != "no":
                        # From here every writer through the stashed pointer
                        # is disclosed (a reload is a fill, an unmodeled call
                        # is opaque). The escape itself writes no byte of the
                        # buffer, so it covers no path (round 4 P2).
                        writers.append(("escape", defn))
                        esc = True
                elif self._is_call(defn):
                    ext = _call_writer(defn, esc)
                    if ext is not None:
                        cov = _cover(cov, ext[0], ext[1] - ext[0])
                    # A call handed a pointer that may alias the buffer may
                    # RETAIN or PLANT it (strtok_r's save pointer, strtol's
                    # end pointer): from here on any reload is a may-alias,
                    # exactly as after a store of `&buf` (#1053 review 4).
                    yes, maybe = _params_touching(self._call_params(defn), esc)
                    if yes or maybe:
                        esc = True
                    # Any call may overwrite a remembered slot: no model
                    # enumerates every memory write of its callee (#1053
                    # review rounds 2-3), so a remembered stash never
                    # survives a call.
                    addrs = frozenset()
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
                           else (_iv_meet(old_cov, cov), old_esc))
                    new_addrs = escape_addrs[v] | addrs
                    changed |= new != memo[v] or new_addrs != escape_addrs[v]
                    memo[v] = new
                    escape_addrs[v] = new_addrs
                writers.clear()
                if truncated or not changed:
                    break
            if truncated:
                break
        if truncated:
            # A proof cut leaves coverage unproven, but the escape facts of
            # the subgraph already discovered are still a least fixed point
            # worth reaching: one pass in discovery order cannot carry an
            # escape around a loop back-edge, and the loop-carried call it
            # would expose (`while (n--) { compose(); stash = buf; }` after
            # a long prefix) was lost to a generic proof-cut leaf (#1034
            # item 3). Escape-only sweeps are monotone, so they stop on their
            # own; the separate allowance only bounds a pathological graph.
            extra = 0
            while extra <= 8 * MEM_CHAIN_BUDGET:
                changed = False
                for v in reversed(graph):
                    extra += 1
                    _, esc, addrs = transfer(v)
                    new_esc = memo[v][1] or esc
                    new_addrs = escape_addrs[v] | addrs
                    changed |= new_esc != memo[v][1] or new_addrs != escape_addrs[v]
                    memo[v] = (memo[v][0], new_esc)
                    escape_addrs[v] = new_addrs
                writers.clear()
                if not changed:
                    break
        # The collection pass: classify every writer against the solved facts.
        for v in reversed(graph):
            cov, esc, addrs = transfer(v)
            memo[v] = (_iv_meet(memo[v][0], cov), memo[v][1] or esc)
            escape_addrs[v] = escape_addrs[v] | addrs
        final = memo.get(int(mv), ((), False))[0]
        unproven = None if truncated else _iv_gaps(final, r_lo, r_hi)
        writers.reverse()
        return {"writers": writers, "covered": unproven == [], "truncated": truncated,
                "region": (r_lo, r_hi), "unproven": unproven,
                "proven_fills": proven_fills}

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
        # Everything this run learns about an object comes from the writes
        # THIS run made, so it cannot survive into the next one: a
        # by-source walk ran the long source first and the short source
        # inherited its extent, which made attribution depend on the order
        # the sources were enumerated in (round-17 review F40).
        self._reset_object_evidence()
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
        self._salvage = {"findings": [], "leaves": []}

        try:
            sub = self._run_forward(func, sources, depth=0, max_depth=max_depth, top=True)
        except RecursionError:
            # Defense in depth: should not happen now that thunk-following and the
            # SSA/call walks are cycle-guarded, but a pathological binary must
            # degrade to a bounded, honest result rather than crash the whole op.
            # What was ALREADY OBSERVED survives: discarding it reported
            # `reached_sinks: []` for a `system(s)` the walk had seen several
            # instructions before the cycle, which is strictly worse than the
            # crash (round-3 review R3). The unwind loses the walk, not the
            # findings -- those are mirrored into `_salvage` as they are made.
            self._truncated = True
            self._truncation_causes.add("recursion")
            sub = {"findings": list(self._salvage["findings"]),
                   "leaves": list(self._salvage["leaves"]) + [{
                       "kind": "analysis_truncated",
                       "address": hex(int(getattr(func, "start", 0))),
                       "cause": "recursion",
                       "detail": (
                           "the walk hit Python's recursion limit (an unresolved "
                           "cycle) and unwound: what it had already found is "
                           "reported, what it had not reached was never walked. "
                           "An empty or short result here is NOT an all-clear -- "
                           "re-run with a lower --max-depth to get a bounded, "
                           "complete answer over the region it can finish."),
                   }],
                   "assumptions": [
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

    def _format_destination_shortfall(self, ins: Any, ssaf: Any, callee: str, model: Any,
                                      params: list[Any]) -> dict | None:
        """A blocking `format_destination_unrecovered` leaf when the call's fully
        read constant format writes more destinations than BN recovered
        arguments for (#1050 review round 8): a forwarded incoming pointer left
        in its argument register is passed, yet BN's MLIL call does not list it,
        so it was never seeded. The missing arguments are not guessed and the
        model range is not widened; the gap is disclosed by count. It only
        adds a leaf, never removes a seed (round 9 removed destination
        trimming). The constant-format test checks only the format's START
        address: a format whose suffix lies in writable storage can be
        under-counted, which then yields no leaf, as on base."""
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
        first = int(first)
        dest_end = first + sum(1 for s in specs if not s.suppressed)
        if dest_end <= len(params):
            return None
        missing = dest_end - len(params)
        at = hex(int(getattr(ins, "address", 0)))
        return {
            "kind": "format_destination_unrecovered",
            "address": at,
            "callee": str(callee),
            "il_text": str(ins),
            "destinations": dest_end - first,
            "recovered": max(0, len(params) - first),
            "missing": missing,
            "detail": (
                f"{self._prose_name(callee)}() at {at}: its constant format writes "
                f"{dest_end - first} destination(s) but Binary Ninja recovered only "
                f"{max(0, len(params) - first)} of those arguments, so {missing} "
                f"destination(s) were not seeded -- what they receive is unchecked"),
        }

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
            out["leaves"].append(self._dropped_result_leaf(ins, callee_fn, tainted_args,
                                                           caller_fn))
        return out

    def _dropped_result_leaf(self, ins: Any, callee_fn: Any, tainted_args: dict,
                             caller_fn: Any) -> dict[str, Any]:
        """#1034 item 5, at the call site (#1050 review round 9): BN's MLIL gives
        a descended call with tainted arguments NO result, so whatever the
        callee returns is not followed. Whether the caller uses the return
        register is not checked -- two rounds of LLIL scanning each missed a
        read (a jump table, a loop back edge) -- so every such call is a
        blocking frontier. The cost: a void callee that receives tainted
        arguments withholds the all-clear too."""
        call_at = hex(int(getattr(ins, "address", 0)))
        fn_at = hex(int(getattr(caller_fn, "start", 0) if caller_fn is not None
                        else getattr(ins, "address", 0)))
        leaf = self._frontier_leaf(
            ins, callee_fn, tainted_args,
            f"tainted arg(s) {sorted(tainted_args)} passed to "
            f"{self._prose_function(callee_fn)} at {call_at}: Binary Ninja recovered no "
            f"result for this call; bn does not check whether the caller uses the return "
            f"register -- check the disassembly after the call with `bn disasm {fn_at}`")
        leaf["cause"] = "return_value_unrecovered"
        return leaf

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

    # Bound on the call graph walked to answer ONE global-continuation
    # question: at most this many functions, the call's own target included.
    # The depth budget beside it is the descent the walk itself would have
    # made. Exhausting EITHER bound while a function that references the
    # global is still unexamined is DISCLOSED as a truncated reachability
    # answer -- never silently reported as "nothing reads it" (round-3
    # review F2: the cap used to turn an unfinished search into an all-clear).
    _GLOBAL_REACH_FUNCS = 512

    # Span of ONE object a reference sweep will ask BN for in a single
    # range query. A larger object is swept in consecutive chunks; the
    # object's own extent is never clipped to this (round-3 review F15: a
    # 128 KiB buffer whose reader sits at +100000 was silently excluded).
    _GLOBAL_REF_CHUNK = 1 << 16
    # Chunks one object may be swept in before the sweep gives up. Reaching
    # it is DISCLOSED as a truncated reference search, never answered as
    # "nothing references it".
    _GLOBAL_REF_CHUNKS = 1024

    def _global_extent(self, ga: int) -> int:
        """Byte width of the global object based at *ga*.

        The data variable's own type first. A stripped binary usually has no
        type for it -- BN reports `void`, width 0 -- and answering 1 there
        excluded every consumer whose code reference sits at an interior
        address, which is the whole point of asking (round-3 review I1). So
        when the type cannot say, the LAYOUT does: the object runs to the
        next data variable, or to the end of its section, whichever comes
        first, bounded. That is exactly where a disjoint neighbour begins, so
        the neighbour's references are still outside."""
        cached = self._global_extent_cache.get(ga)
        if cached is not None:
            return cached
        typed = 0
        getdv = getattr(self.bv, "get_data_var_at", None)
        if callable(getdv):
            try:
                width = getattr(getattr(getdv(ga), "type", None), "width", None)
                typed = int(width) if width else 0
            except Exception:
                typed = 0
        # The recovered type is a LOWER bound, never the last word. Without
        # debug info BN sizes an initialised buffer from the bytes it can
        # see -- `char global_bytes[64] = "seed"` comes back `char[0x5]` --
        # so trusting it stopped the object five bytes in and a consumer
        # reading `g + 8` was never discovered (round-3 review F11). The
        # layout bound below is computed either way and the wider of the two
        # wins; it still stops where a symbolled neighbour begins.
        bounds: list[int] = []
        nxt = getattr(self.bv, "get_next_data_var_after", None)
        getsym = getattr(self.bv, "get_symbol_at", None)
        if callable(nxt):
            # The next data variable that carries a SYMBOL. An untyped
            # global makes BN invent a bare data variable at each offset
            # something references, so stopping at the first data
            # variable would cut the object at its own first interior
            # reference -- the very reference being looked for.
            addr = ga
            for _ in range(64):
                try:
                    nxt_addr = getattr(nxt(addr), "address", None)
                except Exception:
                    break
                if nxt_addr is None or int(nxt_addr) <= addr:
                    break
                addr = int(nxt_addr)
                named = None
                if callable(getsym):
                    try:
                        named = getsym(addr)
                    except Exception:
                        named = None
                if named is not None or not callable(getsym):
                    bounds.append(addr)
                    break
        getsecs = getattr(self.bv, "get_sections_at", None)
        if callable(getsecs):
            try:
                for sec in (getsecs(ga) or []):
                    end = getattr(sec, "end", None)
                    if end is not None and int(end) > ga:
                        bounds.append(int(end))
            except Exception:
                pass
        # No type AND no layout to infer from: claim nothing. Widening
        # on a view that cannot answer would sweep a neighbouring
        # object's references into this one.
        laid_out = (min(bounds) - ga) if bounds else 0
        # The program's own copy is better evidence than the layout: a symbol
        # INSIDE an untyped object (`.set mid, g + 64`) is a data variable
        # like any other, so the layout bound cut the object in half and a
        # reader at +80 fell outside it (BN21-OBJ-01). A copy of 96 bytes at
        # `g` proves `g` is at least 96 wide -- and only a copy that really
        # spans the boundary can merge two neighbours.
        authoritative = self._authoritative_width(ga)
        if authoritative:
            # A type a HUMAN applied answers where this object ends; nothing
            # inferred widens past it (round-13 review F31).
            extent = max(int(authoritative), 1)
            self._global_extent_cache[ga] = extent
            return extent
        written = int(self._global_write_span.get(ga) or 0)
        if ga in self._global_write_unbounded:
            # A copy whose LENGTH nothing states may have filled everything
            # after its destination; the object runs at least through the
            # storage the next symbols name (round-13 review F28).
            reach, alias_cut = self._forward_window(ga)
            written = max(written, reach)
            self._global_extent_unproven.add(ga)
            if alias_cut:
                # The alias enumeration itself ran out -- a different bound
                # from the range sweep's, and one the leaf must name (F39).
                self._global_search_truncated.add(ga)
        if 0 < laid_out < written:
            # BN21-OBJ-01: the object has no recovered type, so its end was
            # guessed from LAYOUT -- and an ALIAS symbol inside it
            # (`.set mid, g + 64`) is a data variable like any other, which
            # halved the object and hid a reader at +80. The program copies
            # PAST that bound, so either the object is wider than the layout
            # says or this copy overflows into the neighbour. Neither is a
            # clear: the identity/extent is recorded as unproven, and the
            # sweep covers what was actually written so a reader inside it is
            # still named.
            self._global_extent_unproven.add(ga)
        # An INFERRED type is a lower bound, never an upper one: BN sizes
        # `char initialized[128] = "seed"` as `char[0x5]`, so a copy of 112
        # bytes into it is evidence about the object, not an overflow of it
        # (round-13 review F28). Only an authoritative type, handled above,
        # says where the object ends.
        extent = max(typed, laid_out, written, 1)
        extent = max(1, int(extent))
        self._global_extent_cache[ga] = extent
        return extent

    @staticmethod
    def _model_len_index(model: dict[str, Any] | None) -> int | None:
        """The argument index a model's copy length comes from, if it states
        one -- `memcpy(dst, src, n)` is 2."""
        sink = (model or {}).get("sink") or {}
        if sink.get("len_arg") is not None:
            try:
                return int(sink["len_arg"])
            except (TypeError, ValueError):
                return None
        if str(sink.get("class")) in ("overflow_len", "fortified_overflow"):
            args = list(sink.get("tainted_args") or ())
            if len(args) == 1:
                try:
                    return int(args[0])
                except (TypeError, ValueError):
                    return None
        return None

    def _note_store_span(self, ssaf: Any, dest: Any, ga: int, ins: Any) -> None:
        """Record the bytes a DIRECT store wrote into the object at *ga*.

        The same evidence a modeled copy's length gives: an inlined
        `memcpy` is a run of stores, and their spans together say how far
        the write reached."""
        raw = self._global_addr(ssaf, dest, canon=False)
        offset = (int(raw) - ga) if raw is not None and int(raw) >= ga else 0
        try:
            size = int(getattr(ins, "size", 0) or 0)
        except (TypeError, ValueError):
            size = 0
        if offset:
            # WHICH bytes this store reached, reported beside the object --
            # the same field a modeled copy into an interior address fills.
            self._global_keyed_offsets.setdefault(ga, set()).add(offset)
        ranges = self._global_tainted_ranges.setdefault(ga, [])
        interval = (offset, offset + size) if size > 0 else (offset, None)
        if interval not in ranges:
            ranges.append(interval)
        if size <= 0:
            if ga not in self._global_write_unbounded:
                self._global_write_unbounded.add(ga)
                self._invalidate_object(ga)
            return
        if offset + size > int(self._global_write_span.get(ga) or 0):
            self._global_write_span[ga] = offset + size
            self._invalidate_object(ga)

    def _note_write_span(self, ssaf: Any, params: list, idx: int,
                         model: dict[str, Any] | None) -> None:
        """Record how far a modeled copy writes into a global object.

        An untyped global's extent is otherwise inferred from LAYOUT, which
        an alias/subobject symbol inside the object silently halves
        (BN21-OBJ-01). What the program copies is direct evidence of how wide
        the object is, so it is kept as a lower bound on the extent."""
        if idx >= len(params):
            return
        ga = self._global_addr(ssaf, params[idx])
        if ga is None:
            return
        raw = self._global_addr(ssaf, params[idx], canon=False)
        offset = (int(raw) - ga) if raw is not None and int(raw) >= ga else 0
        length = self._model_len_index(model)
        n = None if length is None else self._int_const(params[length]) \
            if length < len(params) else None
        # The bytes this copy made tainted, as an interval of the object: an
        # overlap test on intervals is what tells a read that reaches them
        # from one that stops short (round-11 review F21). A length nothing
        # states makes the interval UNBOUNDED -- `memcpy(g, s, n)` may have
        # filled everything after its destination, and claiming it stopped
        # at the one length that WAS constant is the clearing direction.
        ranges = self._global_tainted_ranges.setdefault(ga, [])
        interval = (offset, None if n is None or int(n) <= 0 else offset + int(n))
        if interval not in ranges:
            ranges.append(interval)
        if n is None or int(n) <= 0:
            # An unbounded write is evidence about the object too: it is
            # recorded so the extent stops being clipped by an inferred
            # initialiser width (F28).
            if ga not in self._global_write_unbounded:
                self._global_write_unbounded.add(ga)
                self._invalidate_object(ga)
            return
        span = offset + int(n)
        if span > int(self._global_write_span.get(ga) or 0):
            self._global_write_span[ga] = span
            self._invalidate_object(ga)

    def _global_use_offset(self, expr: Any, ga: int) -> int:
        """Byte offset INSIDE the global at *ga* that *expr* names, else 0.

        `system(g + 4)` lifts to a constant pointer at the interior address;
        the taint key is the object's base, so the leaf reports the base and
        this offset rather than losing which bytes the consumer reached
        (round-3 review F8)."""
        best: int | None = None
        stack: list[tuple[Any, int]] = [(expr, 0)]
        while stack:
            node, depth = stack.pop()
            if node is None or depth > 10:
                continue
            raw = getattr(node, "constant", None)
            if raw is not None:
                try:
                    value = int(raw)
                except (TypeError, ValueError):
                    value = None
                if value is not None and ga <= value and (
                        value < ga + self._global_extent(ga)
                        or self._canon_global(value) == ga):
                    off = value - ga
                    best = off if best is None else min(best, off)
            for attr in ("src", "dest", "left", "right", "params"):
                sub = getattr(node, attr, None)
                if isinstance(sub, (list, tuple)):
                    stack.extend((s, depth + 1) for s in sub)
                elif sub is not None:
                    stack.append((sub, depth + 1))
            stack.extend((s, depth + 1) for s in (getattr(node, "operands", None) or ()))
        return best or 0

    def _names_global(self, expr: Any, ga: int, depth: int = 0) -> bool:
        """Whether *expr* carries a CONSTANT inside the global at *ga*.

        A pure constant scan: no SSA definitions are followed, so asking it
        of every instruction in a body stays linear. It answers the two
        questions a chain walk cannot be afforded for -- "is the object
        named in this call outside any recovered argument" and "is it named
        by a use this scan could not classify"."""
        if expr is None or depth > 10:
            return False
        raw = getattr(expr, "constant", None)
        if raw is not None:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                value = None
            if value is not None and (
                    ga <= value < ga + self._global_extent(ga)
                    or self._canon_global(value) == ga):
                return True
        for attr in ("src", "dest", "left", "right", "params"):
            sub = getattr(expr, attr, None)
            if isinstance(sub, (list, tuple)):
                if any(self._names_global(x, ga, depth + 1) for x in sub):
                    return True
            elif sub is not None and self._names_global(sub, ga, depth + 1):
                return True
        return any(self._names_global(x, ga, depth + 1)
                   for x in (getattr(expr, "operands", None) or ()))

    def _global_ref_functions(self, ga: int) -> dict[int, int]:
        """``{function start: lowest referencing address}`` for every code
        reference to the global OBJECT based at *ga*, read from BN's own
        code-reference DB (the source `bn xrefs` reads).

        The whole object is asked for, not just its first byte: a consumer
        that names `g + 4` has its code reference recorded at the interior
        address, and `_canon_global` folds that back to this base, so missing
        it left an ordinary indexed reader undiscovered (round-3 review F8).
        The width is the data variable's own, so a DISJOINT neighbouring
        object's references are not swept in.

        ``{}`` whenever that DB cannot answer -- a view with no
        ``get_code_refs`` (the unit fakes, a partially loaded view), a raising
        call, refs with no owning function. An absent answer is NOT evidence:
        the caller stays silent rather than inventing a boundary, because the
        alternative is a frontier on every call after any global write.

        A reference here is any mention of the object; whether the function
        can READ the bytes is decided by :meth:`_global_access`."""
        cached = self._global_refs.get(ga)
        if cached is not None:
            return cached
        out: dict[int, int] = {}
        self._global_refs_truncated.discard(ga)
        get_refs = getattr(self.bv, "get_code_refs", None)
        if callable(get_refs):
            extent = self._global_extent(ga)
            refs: list[Any] = []
            # One range query per chunk -- BN's own range-aware API, never a
            # byte at a time -- until the object is covered. An object too
            # large for the chunk budget is RECORDED as a truncated sweep so
            # the caller discloses it instead of answering "no references".
            covered, chunks = 0, 0
            while covered < extent:
                if chunks >= self._GLOBAL_REF_CHUNKS:
                    self._global_refs_truncated.add(ga)
                    break
                span = min(self._GLOBAL_REF_CHUNK, extent - covered)
                start_at = int(ga) + covered
                try:
                    refs.extend(get_refs(start_at, span) or [])
                except Exception:
                    try:
                        refs.extend(get_refs(start_at) or [])
                    except Exception:
                        pass
                covered += span
                chunks += 1
            for ref in refs:
                start = getattr(getattr(ref, "function", None), "start", None)
                if start is None:
                    continue
                start = int(start)
                try:
                    at = int(getattr(ref, "address", start))
                except (TypeError, ValueError):
                    at = start
                if start not in out or at < out[start]:
                    out[start] = at
        self._global_refs[ga] = out
        return out

    # Bytes of ONE unread body a native search will read before giving up.
    # It only ever runs for a callee whose MLIL does not exist, and only for
    # a global the walk already tainted, and the answer is cached.
    _NATIVE_SCAN_BYTES = 1 << 20

    def _native_global_refs(self, fn: Any, ga: int) -> tuple[int, ...]:
        """Addresses inside *fn* where a bounded NATIVE read finds the global
        at *ga* materialised (round-3 review F6).

        BN records no code references out of a body it never analysed, so the
        code-reference DB cannot speak for an unread consumer. The bytes can:
        search the function's own range for the address's little-endian
        encodings -- an architecture-level read that does not depend on
        analysis -- and walk each hit back to the instruction that contains
        it, so the evidence is an instruction address and not a raw offset.
        A linear decode is deliberately not used: the body is unread BECAUSE
        it is enormous, and the materialisation is usually at its end.

        Returns ``()`` when the view cannot be read -- the frontier is
        emitted either way; only the evidence attached to it is lost."""
        key = (int(getattr(fn, "start", 0)), int(ga))
        cached = self._native_refs_cache.get(key)
        if cached is not None:
            return cached
        read = getattr(self.bv, "read", None)
        out: list[int] = []
        if callable(read):
            start = int(getattr(fn, "start", 0))
            end = getattr(fn, "highest_address", None)
            span = (int(end) - start + 16) if end is not None and int(end) > start \
                else self._NATIVE_SCAN_BYTES
            span = max(1, min(span, self._NATIVE_SCAN_BYTES))
            try:
                data = bytes(read(start, span) or b"")
            except Exception:
                data = b""
            needles = set()
            if 0 <= ga < 1 << 32:
                needles.add(ga.to_bytes(4, "little", signed=False))
            if 0 <= ga < 1 << 64:
                needles.add(ga.to_bytes(8, "little", signed=False))
            length_at = getattr(self.bv, "get_instruction_length", None)
            seen: set[int] = set()
            for needle in sorted(needles, key=len, reverse=True):
                pos = data.find(needle)
                while pos >= 0 and len(out) < 8:
                    hit = start + pos
                    site = hit
                    if callable(length_at):
                        for back in range(1, 16):
                            try:
                                size = int(length_at(hit - back) or 0)
                            except Exception:
                                break
                            if size > back:
                                site = hit - back
                                break
                    if site not in seen:
                        seen.add(site)
                        out.append(site)
                    pos = data.find(needle, pos + 1)
        result = tuple(sorted(out))
        self._native_refs_cache[key] = result
        return result

    # Expression shapes a POINTER can be carried through without ceasing to
    # point at the same object: SSA copies, PHI merges, widening/narrowing
    # casts and constant or register displacement.
    _PTR_CARRY_OPS = ("ADD", "SUB", "ZX", "SX", "VAR_SSA", "VAR_ALIASED", "PHI")
    # One SSA hop costs TWO levels here (the reading expression, then the
    # variable), and an unoptimised body stacks several before a call -- so
    # the budget is sized for a copy chain, not for a handful of nodes.
    # Stepping into a callee's returns costs a whole block of it, which is
    # what bounds the nesting.
    _PTR_RESOLVE_DEPTH = 96
    _PTR_RETURN_STEP = 16
    # Expression nodes ONE function scan may visit, shared across its
    # instructions: the walk fans out across every operand, so a depth
    # budget alone is exponential on a wide tree. Exhausting it is
    # DISCLOSED, never answered as "nothing here".
    _PTR_SCAN_NODES = 50000
    # Expression nodes ONE resolution may visit. The walk fans out across
    # every operand, so a deep budget without a node budget is exponential
    # on a wide tree; this keeps a single question linear in practice.
    _PTR_RESOLVE_NODES = 50000

    def _ssa_def(self, ssaf: Any, var: Any) -> Any:
        """``get_ssa_var_definition`` memoised for one walk.

        The pointer walk asks for the same definitions over and over as it
        follows a long copy chain from every instruction in a body; each ask
        is a Binary Ninja API call."""
        key = (id(ssaf), var_key(var), getattr(var, "version", None))
        hit = self._ssa_def_cache.get(key, _MISSING)
        if hit is not _MISSING:
            return hit
        try:
            defn = ssaf.get_ssa_var_definition(var)
        except Exception:
            defn = None
        self._ssa_def_cache[key] = defn
        return defn

    def _ptr_resolves(self, ssaf: Any, expr: Any, base: Any, depth: int = 0,
                      seen: set | None = None, *, alias: bool = True,
                      budget: _ScanBudget | None = None) -> bool:
        """Whether *expr* may hold a pointer *base* accepts.

        Follows SSA definitions generally -- so EVERY arm of a PHI merge is
        tried, not just a lone reader -- plus pointer arithmetic in either
        operand order and widening/narrowing casts, because `p + off`,
        `off + p` and `(char *)p - 4` all still point into the same object
        (round-3 review F4). It never answers False from a shape it did not
        understand: an unrecognised expression simply contributes nothing,
        and every caller's default is "consumer"."""
        if budget is None:
            budget = _ScanBudget(self._PTR_RESOLVE_NODES)
        if expr is None:
            return False
        if depth > self._PTR_RESOLVE_DEPTH:
            # BN21-BOUND-03: this is the INTERNAL wrapper-chain bound, not the
            # CLI's call depth. Saying "no alias" here lost a real one.
            budget.depth_exhausted()
            return False
        if not budget.spend():
            return False
        seen = set() if seen is None else seen
        idx = getattr(expr, "expr_index", None)
        if idx is not None:
            node = ("expr", int(idx))
            if node in seen:
                return False
            seen.add(node)
        if base(expr):
            return True
        if is_ssa_var(expr):
            key = (var_key(expr), getattr(expr, "version", None))
            if key in seen:
                return False
            seen.add(key)
            defn = self._ssa_def(ssaf, expr)
            if defn is None:
                return False
            if self._is_call(defn):
                if not alias:
                    # The RETURN of a pointer is not a read of its bytes:
                    # when the question is "does this RET hand the object
                    # back", a call's returned alias is exactly what must not
                    # count (round-3 review F17).
                    return False
                # `p = memcpy(dst, ..)` hands BACK its destination: the model
                # says `arg:0 -> ret`, so the result is that same object and
                # `p + 1` still points into it (round-3 review I3). Without
                # this the alias looked like a fresh pointer and a consumer
                # reading through it was cleared as write-only.
                params = self._call_params(defn)
                _k, model = lookup_model(
                    self.models, self._callee_name(self._resolve_direct_target(defn)))
                for rule in ((model or {}).get("propagates") or ()):
                    if str(rule.get("to") or "") not in ("ret", "*ret"):
                        continue
                    idx = _try_arg_index(str(rule.get("from") or ""))
                    if idx is not None and idx < len(params) and self._ptr_resolves(
                            ssaf, params[idx], base, depth + 1, seen, alias=alias,
                            budget=budget):
                        return True
                # No model, or none that names the return: a body may still
                # hand the pointer back (`char *reset_ret(void) { return
                # memset(g, 0, n); }`). Ask it, so the CALLER's use of the
                # result is the read that blocks (round-3 review F17).
                body, _unread = self._callee_body(defn)
                bodies = [body] if body is not None else [
                    # An indirect site the agent PINNED with --resolve-map is
                    # resolved for the reachability walk already; the alias
                    # walk has to honour the same pin, or supplying the map
                    # turns a disclosed frontier into a false all-clear
                    # (round-3 review R2).
                    cfn for cfn in (
                        function_at(self.bv, int(t))
                        for t in (self.resolve_map.get(
                            int(getattr(defn, "address", 0))) or ()))
                    if cfn is not None]
                for cand in bodies:
                    if self._fn_returns_pointer(
                            cand, base, depth + self._PTR_RETURN_STEP, budget=budget):
                        return True
                    # ...or it hands back one of its own parameters, in which
                    # case the result is whatever THIS call passed there (F19).
                    for pidx in self._fn_returned_params(cand):
                        if pidx < len(params) and self._ptr_resolves(
                                ssaf, params[pidx], base,
                                depth + self._PTR_RETURN_STEP, seen,
                                alias=alias, budget=budget):
                            return True
                return False
            srcs = getattr(defn, "src", None)
            if not isinstance(srcs, (list, tuple)):
                srcs = [] if srcs is None else [srcs]
            return any(self._ptr_resolves(ssaf, s, base, depth + 1, seen,
                                          alias=alias, budget=budget)
                       for s in srcs)
        if not hasattr(expr, "operation"):
            return False
        if not any(tok in op_name(expr) for tok in self._PTR_CARRY_OPS):
            return False
        subs: list[Any] = []
        for attr in ("left", "right", "src"):
            sub = getattr(expr, attr, None)
            if isinstance(sub, (list, tuple)):
                subs.extend(sub)
            elif sub is not None:
                subs.append(sub)
        subs.extend(expr_reads(expr))
        subs.extend(getattr(expr, "operands", None) or ())
        return any(self._ptr_resolves(ssaf, s, base, depth + 1, seen,
                                      alias=alias, budget=budget)
                   for s in subs)

    @staticmethod
    def _ret_sources(ins: Any) -> list[Any]:
        """A RET's returned expressions as a list.

        BN hands `MLIL_RET.src` back as a list on some builds and as a
        GENERATOR on others; treating a generator as one opaque operand made
        every `return <pointer>` resolve to nothing."""
        srcs = getattr(ins, "src", None)
        if srcs is None:
            return []
        if isinstance(srcs, (list, tuple)):
            return list(srcs)
        if hasattr(srcs, "operation"):
            return [srcs]
        try:
            return list(srcs)
        except TypeError:
            return [srcs]

    def _fn_returned_params(self, fn: Any) -> tuple[int, ...]:
        """Parameter indices *fn* may hand back as its return value.

        `char *reset_arg(char *p) { return memset(p, 0, 1); }` returns its
        own destination, so at the call site the result is whatever the
        CALLER passed -- the remapping the alias needs (round-3 review F19).
        Searching the callee's body for the global's address instead finds
        nothing: inside the callee it is a formal parameter."""
        key = int(getattr(fn, "start", 0))
        cached = self._returned_params_cache.get(key)
        if cached is not None:
            return cached
        formal_count = len(list(getattr(fn, "parameter_vars", []) or []))
        if key in self._returned_params_active:
            # A cycle: `char *f(char *p, int n) { return n ? f(p, n-1) : p; }`.
            # Unrolling it is what reached the Python recursion limit
            # (round-3 review R3). Answering "returns nothing" would be a
            # guess in the CLEARING direction, so a cycle answers with every
            # formal -- the conservative side, and never cached as the
            # function's own answer.
            return tuple(range(formal_count))
        self._returned_params_active.add(key)
        out: list[int] = []
        ssaf = None
        instrs: list[Any] = []
        formals: list[Any] = []
        try:
            ssaf = self._ssa_func(fn)
            instrs = self._instrs(ssaf)
            formals = list(getattr(fn, "parameter_vars", []) or [])
        except TaintError:
            pass
        rets = [ins for ins in instrs if "RET" in op_name(ins)]
        for idx, formal in enumerate(formals):
            root = (lambda k: lambda e: is_ssa_var(e) and var_key(e) == k)(var_key(formal))
            if any(self._ptr_resolves(ssaf, src, root)
                   for ins in rets for src in self._ret_sources(ins)):
                out.append(idx)
        self._returned_params_active.discard(key)
        result = tuple(out)
        self._returned_params_cache[key] = result
        return result

    def _fn_returns_pointer(self, fn: Any, base: Any, depth: int = 0,
                            budget: _ScanBudget | None = None) -> bool:
        """Whether *fn* returns a pointer *base* accepts.

        A callee that hands the object's address back does not read its
        bytes -- the caller that consumes the result does, and this is what
        lets the caller's own scan see it (round-3 review F17)."""
        if depth > self._PTR_RESOLVE_DEPTH:
            if budget is not None:
                budget.depth_exhausted()
            return False
        try:
            ssaf = self._ssa_func(fn)
            instrs = self._instrs(ssaf)
        except TaintError:
            return False
        for ins in instrs:
            if "RET" not in op_name(ins):
                continue
            if any(self._ptr_resolves(ssaf, s, base, depth + 1, budget=budget)
                   for s in self._ret_sources(ins)):
                return True
        return False

    def _expr_mentions(self, ssaf: Any, expr: Any, base: Any, depth: int = 0,
                       budget: _ScanBudget | None = None) -> bool:
        """Whether any sub-expression of *expr* may be the tracked pointer.

        One node budget is shared across the whole walk: giving each
        sub-expression its own turned a function with a long accumulator
        chain into a quadratic sweep over BN's API."""
        if budget is None:
            budget = _ScanBudget(self._PTR_RESOLVE_NODES)
        if expr is None or depth > 10 or not budget.spend():
            return False
        if is_ssa_var(expr):
            return self._ptr_resolves(ssaf, expr, base, budget=budget)
        if not hasattr(expr, "operation"):
            return False
        if self._ptr_resolves(ssaf, expr, base, budget=budget):
            return True
        subs: list[Any] = []
        for attr in ("src", "dest", "left", "right", "params"):
            sub = getattr(expr, attr, None)
            if isinstance(sub, (list, tuple)):
                subs.extend(sub)
            elif sub is not None:
                subs.append(sub)
        subs.extend(getattr(expr, "operands", None) or ())
        return any(self._expr_mentions(ssaf, s, base, depth + 1, budget) for s in subs)

    def _expr_loads(self, ssaf: Any, expr: Any, base: Any, depth: int = 0,
                    budget: _ScanBudget | None = None) -> bool:
        """Whether any sub-expression of *expr* LOADS through the pointer.

        A store's destination is reached only through ``operands``, where it
        is an address expression and not a load, so ``[p] = 0`` answers False
        while ``[p] = [p] + 1`` and ``x = [q]`` (``q = p``) answer True."""
        if budget is None:
            budget = _ScanBudget(self._PTR_RESOLVE_NODES)
        if expr is None or depth > 10 or not budget.spend() \
                or not hasattr(expr, "operation"):
            return False
        if "LOAD" in op_name(expr) and self._ptr_resolves(
                ssaf, getattr(expr, "src", None), base, budget=budget):
            return True
        subs: list[Any] = []
        for attr in ("src", "left", "right", "params"):
            sub = getattr(expr, attr, None)
            if isinstance(sub, (list, tuple)):
                subs.extend(sub)
            elif sub is not None:
                subs.append(sub)
        subs.extend(getattr(expr, "operands", None) or ())
        return any(self._expr_loads(ssaf, s, base, depth + 1, budget) for s in subs)

    def _load_ptr(self, ssaf: Any, expr: Any, base: Any, depth: int = 0,
                  budget: _ScanBudget | None = None) -> Any:
        """The ADDRESS expression of the first LOAD through *base*, or None.

        `observed = p[8]` reads at +8, and the displacement lives in that
        address expression -- a load verdict that reported no expression at
        all fell back to the literal scan and claimed the base (round-11
        review F22)."""
        if budget is None:
            budget = _ScanBudget(self._PTR_RESOLVE_NODES)
        if expr is None or depth > 10 or not budget.spend() \
                or not hasattr(expr, "operation"):
            return None
        if "LOAD" in op_name(expr):
            src = getattr(expr, "src", None)
            if self._ptr_resolves(ssaf, src, base, budget=budget):
                return src
        subs: list[Any] = []
        for attr in ("src", "left", "right", "params"):
            sub = getattr(expr, attr, None)
            if isinstance(sub, (list, tuple)):
                subs.extend(sub)
            elif sub is not None:
                subs.append(sub)
        subs.extend(getattr(expr, "operands", None) or ())
        for one in subs:
            got = self._load_ptr(ssaf, one, base, depth + 1, budget)
            if got is not None:
                return got
        return None

    def _global_base(self, ssaf: Any, ga: int):
        """The root test for the global OBJECT based at *ga*.

        Deliberately independent of *ssaf*: it tests ONE expression node (a
        constant pointer), while the def-chain walking is `_ptr_resolves`'s
        job. That lets the same predicate be applied inside another
        function's IL when a call's returned pointer has to be followed.

        Its own base, or any address inside its extent: `g + 4` is still `g`,
        and on a view with no type for the object BN folds that interior
        address to a data variable of its own, so the canonical test alone
        misses it (round-3 review I1)."""
        extent = self._global_extent(ga)

        def base(expr: Any) -> bool:
            if self._global_addr(None, expr) == ga:
                return True
            raw = self._global_addr(None, expr, canon=False)
            return raw is not None and ga <= raw < ga + extent
        return base

    def _may_be_global(self, ssaf: Any, expr: Any, ga: int) -> bool:
        return self._ptr_resolves(ssaf, expr, self._global_base(ssaf, ga))

    def _expr_mentions_global(self, ssaf: Any, expr: Any, ga: int) -> bool:
        return self._expr_mentions(ssaf, expr, self._global_base(ssaf, ga))

    def _expr_loads_global(self, ssaf: Any, expr: Any, ga: int) -> bool:
        return self._expr_loads(ssaf, expr, self._global_base(ssaf, ga))

    @staticmethod
    def _model_arg_role(model: dict[str, Any] | None, idx: int) -> str:
        """What *model* says the callee does with the BYTES argument *idx*
        points at: ``"write"`` (a destination it fills), ``"read"`` (a source
        it consumes, a declared sink operand, or a vararg folded into the
        destination) or ``"unknown"`` (round-3 review F5).

        Destination and source roles are decided SEPARATELY, and a read-write
        argument is a read: `memcpy(g, g, 8)` consumes the bytes even though
        arg0 is also the destination.

        A ``"write"`` here is the model's own, ADDITIVE claim -- never a
        promise that the callee reads nothing else (round-3 review F7). Only
        :meth:`_param_destination_only` can turn it into a write-only verdict
        for a callee this binary has a body for."""
        if not model:
            return "unknown"
        tok = f"*arg:{idx}"
        written = read = False
        for rule in (model.get("sources") or ()):
            if str(rule.get("to") or "") == tok:
                # A source model's output buffer is what the callee FILLS --
                # `read(fd, g, 64)` writes g, it does not consume the bytes
                # already there (round-3 review F20). Declared in `sources`
                # rather than `propagates`, so asking only the propagate
                # rules left the role unstated and the call "unknown".
                written = True
        if tok in (model.get("reads_dest") or ()):
            # An APPEND reads its destination to find where to append
            # (round-3 review I4): `strcat(g, "suffix")` consumes g's bytes
            # before it writes any, so its destination is never write-only.
            read = True
        if tok in (model.get("writes") or ()):
            written = True
        for rule in (model.get("propagates") or ()):
            if str(rule.get("to") or "") == tok:
                written = True
            if str(rule.get("from") or "") == tok:
                read = True
        varargs = model.get("varargs") or {}
        if str(varargs.get("to") or "") == tok:
            written = True
        first = varargs.get("first_index")
        if first is not None and idx >= int(first):
            read = True          # a vararg the model folds INTO the destination
        if idx in ((model.get("sink") or {}).get("tainted_args") or ()):
            read = True
        if read:
            return "read"
        return "write" if written else "unknown"

    def _callee_body(self, ins: Any) -> tuple[Any | None, bool]:
        """``(body, unread)`` for the call *ins*: the in-binary body it runs
        (following a thunk), and whether BN measured an in-binary body here
        and skipped analysing it.

        ``unread`` matters because a model on such a callee is still only an
        overlay: there IS a body, and nothing read it (round-3 review F7)."""
        taddr = self._resolve_direct_target(ins)
        if taddr is None:
            return None, False
        cfn = function_at(self.bv, taddr)
        if cfn is None:
            return None, False
        if self._is_internal(cfn):
            return cfn, False
        resolved = self._follow_thunk_cached(cfn)
        if resolved is not None and self._is_internal(resolved):
            return resolved, False
        reason = str(self._not_internal_reason(cfn) or "")
        return None, reason.startswith("analysis_skipped:")

    def _param_destination_only(self, callee: Any, idx: int, depth: int = 0) -> bool:
        """Whether *callee*'s own body only WRITES through parameter *idx*.

        A model is additive flow evidence; a body is the authority (#807, and
        round-3 review F7: a partial ``{"propagates": [...]}`` overlay read as
        "and it reads nothing else" suppressed a real consumer --
        ``copy_and_use(dst, src)`` whose body calls ``system(dst)`` BEFORE the
        copy the overlay describes). So when the callee has an analysed body,
        the write-only question is asked of that body, with the same
        classifier the global uses, rooted at the parameter."""
        if depth > 2:
            return False
        try:
            pv = self._param_var(callee, idx)
        except Exception:
            pv = None
        if pv is None:
            return False
        key = (int(getattr(callee, "start", 0)), int(idx))
        cached = self._param_access_cache.get(key)
        if cached is None:
            pkey = var_key(pv)
            root = (lambda k: lambda e: is_ssa_var(e) and var_key(e) == k)(pkey)
            cached = self._pointer_access(
                callee, root, fallback=None, depth=depth + 1)
            if cached[1] == "return":
                # EVERY feasible use decides this, never the first one. BN
                # lists `if (n) return p;` before the `system(p)` on the
                # other arm, so answering from the first use called a body
                # that calls a sink "writes only" (round-16 review F38).
                uses: list = []
                self._pointer_access(callee, root, fallback=None,
                                     depth=depth + 1, collect=uses)
                if any(use.kind != "return" for use in uses):
                    cached = next(use for use in uses if use.kind != "return")
            self._param_access_cache[key] = cached
        # Round-14 review F37: `char *identity(char *p) { return p; }` reads
        # nothing. Returning the pointer is not a use of the bytes (round-3
        # review F17 established that for the global's own scan), and what
        # the CALLER does with the result is classified by the caller's own
        # scan -- which follows the returned alias.
        return cached[1] in ("write_only", "return")

    def _is_pointer_carry(self, ssaf: Any, ins: Any, base: Any,
                          budget: _ScanBudget | None = None) -> bool:
        """`p = g`, `p = q + 4`, `p = phi(..)`: the pointer is MOVED, not used.

        Materialising an address is not an access -- the access happens at
        the terminal uses of the resulting value, which this same scan
        classifies (round-3 review F5: a write-only `memset(p, 0, 64)` helper
        was called a reader because the `lea` that built `p` came first)."""
        opn = op_name(ins)
        if "SET_VAR" not in opn and "VAR_PHI" not in opn:
            return False
        src = getattr(ins, "src", None)
        if isinstance(src, (list, tuple)):
            return any(self._ptr_resolves(ssaf, s, base, budget=budget) for s in src)
        return self._ptr_resolves(ssaf, src, base, budget=budget)

    def _pointer_access(self, fn: Any, base: Any, *, fallback: int | None,
                        depth: int = 0, offset_of: Any = None,
                        names_object: Any = None, offset_expr: Any = None,
                        span_of: Any = None, collect: list | None = None) -> Any:
        """The first use inside *fn* that can READ the bytes the tracked
        pointer addresses, as a :class:`_Use`, or ``kind="write_only"`` ONLY
        when every terminal use is a proven write into them.

        The default is "consumer": a use this classifier cannot account for
        as a write is read-capable, because the cost of a wrong "it only
        writes" is the false all-clear this frontier exists to withhold.
        Kinds: ``load`` (the bytes are loaded), ``call_arg`` (the pointer
        reaches an argument whose callee READS it -- by a model, where there
        is no body to contradict it), ``unknown_call_arg`` (it reaches a call
        whose handling of those bytes is not known: an unmodeled callee, a
        role nothing states, a body that could not be read, or an argument
        Binary Ninja did not recover -- an honest unknown, never a claimed
        read and never a clear), ``address_escape`` (the address is stored
        into memory), ``return``, ``unclassified`` (a reference this scan
        could not account for) and ``unreadable`` (the body's MLIL could not
        be read at all).

        With ``collect`` every read-capable use is appended to that list and
        the scan runs to the end -- what the overlap test needs, since the
        FIRST use of a neighbouring object says nothing about whether a LATER
        one reaches into this one."""
        try:
            ssaf = self._ssa_func(fn)
            instrs = self._instrs(ssaf)
        except TaintError:
            return _Use(fallback, "unreadable", 0, None, None, True)

        def verdict(ins: Any, kind: str, seed: str | None = None,
                    expr: Any = None, span: int | None = None) -> Any:
            """The offset names WHICH bytes of the object the use reached
            (F8), and ``offset_known`` says whether it was ESTABLISHED.

            A literal `system(g + 8)` carries its displacement in the
            instruction, but a pointer handed back through a wrapper carries
            it in the chain -- the literal scan found nothing there and
            reported 0, which reads as "the base" rather than "not known"
            (round-10 review M1). The chain is followed first; a displacement
            neither route establishes is marked unknown instead of 0."""
            offset, known = 0, True
            if expr is not None and offset_expr is not None:
                resolved = offset_expr(ssaf, expr)
                if resolved is None:
                    literal = int(offset_of(ins)) if offset_of is not None else 0
                    offset, known = literal, bool(literal)
                else:
                    offset = int(resolved)
            elif offset_of is not None:
                offset = int(offset_of(ins))
            return _Use(int(getattr(ins, "address", 0)), kind, offset, seed,
                        span, known)

        def arg_seed(ins: Any, hit: list[int]) -> str | None:
            """`arg:<callee>:<index>` for a pointer that reaches a named
            call's argument -- the locator that seeds those bytes inside the
            consumer."""
            name = self._callee_name(self._resolve_direct_target(ins))
            return f"arg:{name}:{hit[0]}" if name and hit else None

        wrote = False
        first_at = int(getattr(instrs[0], "address", 0)) if instrs else None
        scan = _ScanBudget(self._PTR_SCAN_NODES)

        def resolves(expr: Any, *, alias: bool = True) -> bool:
            return self._ptr_resolves(ssaf, expr, base, alias=alias, budget=scan)

        def emit(use: Any) -> Any:
            """Return the first use, or gather every one of them."""
            if collect is None:
                return use
            collect.append(use)
            return None

        for ins in instrs:
            opn = op_name(ins)
            if scan.nodes <= 0:
                # The scan ran out before it could account for the rest of
                # this body. BN21-BOUND-02: say WHICH bound stopped it, so
                # the leaf reads as an unfinished classification and not as a
                # reader this walk identified.
                scan.cause = scan.cause or "pointer_scan_budget"
                hit_use = emit(verdict(ins, "scan_budget_exhausted"))
                if hit_use is not None:
                    return hit_use
                break
            if self._is_call(ins):
                params = self._call_params(ins)
                hit = [i for i, p in enumerate(params) if resolves(p)]
                if not hit:
                    # In the instruction but in no RECOVERED argument: BN
                    # under-recovered this call's arity, so what the callee
                    # does with the bytes is unknown, not a read.
                    if names_object is not None and names_object(ins):
                        hit_use = emit(verdict(ins, "unknown_call_arg"))
                        if hit_use is not None:
                            return hit_use
                    continue
                span = None if span_of is None else span_of(ins, hit)
                body, unread_body = self._callee_body(ins)
                kind = None
                if body is not None:
                    # The body is the authority; a model is only an overlay on
                    # it (#807 / F7). Ask the body itself.
                    if all(self._param_destination_only(body, i, depth) for i in hit):
                        wrote = True
                        continue
                    kind = "unknown_call_arg"
                elif unread_body:
                    # There IS an in-binary body and nothing read it, so no
                    # model on this name can promise it only writes (F7).
                    kind = "unknown_call_arg"
                else:
                    _key, model = lookup_model(
                        self.models,
                        self._callee_name(self._resolve_direct_target(ins)))
                    roles = [self._model_arg_role(model, i) for i in hit]
                    if "read" in roles:
                        kind = "call_arg"         # a modeled source operand
                    elif all(r == "write" for r in roles):
                        wrote = True              # a modeled destination, and
                        continue                  # no body to contradict it
                    else:
                        kind = "unknown_call_arg"
                # The argument the callee READS is the one whose
                # displacement describes this access: `memcpy(g, g + 48, 32)`
                # reads at +48 and writes at 0 (round-11 review F21/F22).
                read_hit = next(
                    (i for i in hit
                     if self._model_arg_role(
                         lookup_model(self.models, self._callee_name(
                             self._resolve_direct_target(ins)))[1], i) == "read"),
                    hit[0])
                hit_use = emit(verdict(ins, kind, arg_seed(ins, hit),
                                       params[read_hit], span))
                if hit_use is not None:
                    return hit_use
                continue
            if "RET" in opn:
                srcs = self._ret_sources(ins)
                # Returning the POINTER is not reading the bytes, and the
                # caller -- which this same scan classifies -- is where any
                # read of them happens. `alias=False` keeps the genuine
                # escape (`return g;`) while dropping the residual alias BN
                # recovers for a tail `return memset(g, 0, n);`, whose value
                # the caller usually discards (round-3 review F17).
                if any(resolves(s, alias=False) for s in srcs):
                    hit_use = emit(verdict(ins, "return"))
                    if hit_use is not None:
                        return hit_use
                continue
            if "STORE" in opn:
                dest_hit = resolves(getattr(ins, "dest", None))
                src_hit = self._expr_mentions(ssaf, getattr(ins, "src", None), base,
                                              budget=scan)
                if dest_hit and not src_hit:
                    wrote = True       # `p[i] = <value that is not p>`
                    continue
                if dest_hit or src_hit:
                    # `p[i] = p[j] + 1` reads, and `*q = p` escapes the
                    # address to memory this walk cannot follow.
                    loaded = self._load_ptr(ssaf, ins, base, budget=scan)
                    hit_use = emit(verdict(
                        ins, "load" if self._expr_loads(ssaf, ins, base, budget=scan)
                        else "address_escape",
                        expr=loaded if loaded is not None
                        else (getattr(ins, "dest", None) if dest_hit else None),
                        span=int(getattr(ins, "size", 0) or 0) or None))
                    if hit_use is not None:
                        return hit_use
                continue
            if self._expr_loads(ssaf, ins, base, budget=scan):
                hit_use = emit(verdict(
                    ins, "load", expr=self._load_ptr(ssaf, ins, base, budget=scan),
                    span=int(getattr(ins, "size", 0) or 0) or None))
                if hit_use is not None:
                    return hit_use
                continue
            if self._is_pointer_carry(ssaf, ins, base, budget=scan):
                continue              # moved, not used: see the terminal uses
            if names_object is not None and names_object(ins):
                hit_use = emit(verdict(ins, "unclassified"))
                if hit_use is not None:
                    return hit_use
        if collect is not None:
            return _Use(None, "collected", 0, None, None, True)
        # Nothing read-capable. A body with a proven write and nothing else
        # is a write-only use; a body where this scan accounted for no use at
        # all is disclosed rather than cleared.
        if scan.cause is not None:
            # A bound was reached somewhere in this body, so "every use is a
            # proven write" and "no use at all" are both claims this scan did
            # not establish (BN21-BOUND-02/03).
            return _Use(fallback if fallback is not None else first_at,
                        _SCAN_CAUSE_KINDS[scan.cause], 0, None, None, True)
        if wrote:
            return _Use(None, "write_only", 0, None, None, True)
        return _Use(fallback, "unclassified", 0, None, None, True)

    def _ptr_offset(self, ssaf: Any, expr: Any, root: Any, depth: int = 0,
                    seen: set | None = None) -> int | None:
        """The byte displacement from *root* that *expr* points at, or None
        when this scan cannot establish ONE value.

        Affine provenance, not a yes/no alias test: a displacement composes
        along the chain (`advance(p)` returning `p + 16`, consumed as
        `advance(..) + 8`, is +24), and a PHI whose arms disagree, or a hop
        this walk cannot follow, is UNKNOWN -- never 0, which would read as
        the object's base (round-11 review F22). *root* answers the leaf
        question: the displacement of an expression that IS the tracked
        object, or None."""
        if expr is None or depth > self._PTR_RESOLVE_DEPTH:
            return None
        seen = set() if seen is None else seen
        idx = getattr(expr, "expr_index", None)
        if idx is not None:
            if ("off", int(idx)) in seen:
                return None
            seen.add(("off", int(idx)))
        rooted = root(expr)
        if rooted is not None:
            return int(rooted)
        opn = op_name(expr)
        if is_ssa_var(expr):
            key = (var_key(expr), getattr(expr, "version", None))
            if key in seen:
                return None
            seen.add(key)
            defn = self._ssa_def(ssaf, expr)
            if defn is None:
                return None
            if self._is_call(defn):
                return self._call_offset(ssaf, defn, root, depth, seen)
            srcs = getattr(defn, "src", None)
            if not isinstance(srcs, (list, tuple)):
                srcs = [] if srcs is None else [srcs]
            return self._merge_offsets(ssaf, srcs, root, depth, seen)
        if any(tok in opn for tok in ("ADD", "SUB")):
            left, right = getattr(expr, "left", None), getattr(expr, "right", None)
            sign = -1 if "SUB" in opn else 1
            for ptr, other in ((left, right), (right, left)):
                base_off = self._ptr_offset(ssaf, ptr, root, depth + 1, seen)
                if base_off is None:
                    continue
                delta = self._int_const(other)
                return None if delta is None else base_off + sign * int(delta)
            return None
        subs: list[Any] = []
        for attr in ("src", "operands"):
            sub = getattr(expr, attr, None)
            subs.extend(list(sub) if isinstance(sub, (list, tuple))
                        else ([] if sub is None else [sub]))
        # A `MLIL_VAR_SSA` wrapper holds its variable in `reads`, not in an
        # operand attribute -- the same place `_ptr_resolves` finds it.
        subs.extend(expr_reads(expr))
        for one in subs:
            got = self._ptr_offset(ssaf, one, root, depth + 1, seen)
            if got is not None:
                return got
        return None

    def _merge_offsets(self, ssaf: Any, arms: list, root: Any, depth: int,
                       seen: set) -> int | None:
        """The displacement a MERGE of *arms* carries, as a lattice join.

        Exact only when every FEASIBLE arm -- one that can hold this pointer
        at all -- resolves, and they all resolve to the SAME displacement.
        An arm this walk cannot resolve, or one that disagrees, makes the
        answer unknown (round-12 review F27). Each arm gets its own cycle
        state: sharing one made a revisited arm answer "nothing here", which
        was then dropped and read as agreement."""
        def feasible(arm: Any) -> bool:
            return self._ptr_resolves(ssaf, arm, lambda e: root(e) is not None)

        answers: set[int] = set()
        for arm in arms:
            if len(arms) > 1 and not feasible(arm):
                continue        # this arm cannot be the pointer at all
            got = self._ptr_offset(ssaf, arm, root, depth + 1, set(seen))
            if got is None:
                if len(arms) == 1 or feasible(arm):
                    return None
                continue
            answers.add(got)
        if len(answers) == 1:
            return answers.pop()
        return None

    def _call_offset(self, ssaf: Any, defn: Any, root: Any, depth: int,
                     seen: set) -> int | None:
        """The displacement a CALL's returned pointer carries.

        Three routes, each composing the callee's own adjustment with the
        actual argument's: a modeled `arg:N -> ret`, a body that hands back
        one of its FORMALS (possibly displaced -- `return p + 16`), and a
        body whose returns name the object itself."""
        params = self._call_params(defn)
        _k, model = lookup_model(
            self.models, self._callee_name(self._resolve_direct_target(defn)))
        for rule in ((model or {}).get("propagates") or ()):
            if str(rule.get("to") or "") not in ("ret", "*ret"):
                continue
            arg = _try_arg_index(str(rule.get("from") or ""))
            if arg is not None and arg < len(params):
                got = self._ptr_offset(ssaf, params[arg], root, depth + 1, seen)
                if got is not None:
                    return got
        body, _unread = self._callee_body(defn)
        if body is None:
            return None
        formals = [pidx for pidx in self._fn_returned_params(body)
                   if pidx < len(params)]
        answers: set[int] = set()
        for pidx in formals:
            actual = self._ptr_offset(ssaf, params[pidx], root, depth + 1,
                                      set(seen))
            if actual is None:
                # This argument cannot be the object at all, so it is not a
                # path the pointer takes; one that COULD be and does not
                # resolve is handled by `_ptr_offset` returning None below.
                if self._ptr_resolves(ssaf, params[pidx],
                                      lambda e: root(e) is not None):
                    return None
                continue
            inner = self._fn_returned_param_offset(body, pidx, depth + 1)
            # The callee's own adjustment is part of the answer: treating a
            # returned formal as the argument itself dropped `+ 16` (F22).
            if inner is None:
                return None
            answers.add(actual + inner)
        # A body may ALSO return the object by name on another path --
        # `return n ? p : tracked + 24` -- which is a second feasible
        # displacement, not a fallback for when the formals say nothing
        # (round-14 review F36). "No such path" and "a path whose
        # displacement is not one value" are DIFFERENT answers: the second
        # makes the whole result unknown however exact the formals are
        # (round-16 review F36).
        rooted, direct = self._fn_return_routes(body, root, depth + 1)
        if rooted and direct is None:
            return None
        if rooted:
            answers.add(direct)
        if answers:
            # `choose(p, q, n)` may hand back EITHER formal: two feasible
            # actuals are two displacements, so there is no single one
            # (round-13 review F30).
            return answers.pop() if len(answers) == 1 else None
        return None

    def _global_root(self, ga: int):
        """Leaf test for the object at *ga*: the displacement a constant
        pointer into it names."""
        extent = self._global_extent(ga)

        def root(expr: Any) -> int | None:
            value = self._int_const(expr)
            if value is None:
                return None
            value = int(value)
            return value - ga if ga <= value < ga + extent else None
        return root

    def _global_ptr_offset(self, ssaf: Any, expr: Any, ga: int) -> int | None:
        """The displacement into the object at *ga* that *expr* points at."""
        return self._ptr_offset(ssaf, expr, self._global_root(ga))

    def _fn_returned_param_offset(self, fn: Any, pidx: int,
                                  depth: int = 0) -> int | None:
        """The displacement *fn* adds to formal *pidx* before returning it."""
        key = (int(getattr(fn, "start", 0)), int(pidx), "formal")
        if key in self._return_offset_active or depth > self._PTR_RESOLVE_DEPTH:
            return None
        self._return_offset_active.add(key)
        try:
            formals = list(getattr(fn, "parameter_vars", []) or [])
            if pidx >= len(formals):
                return None
            pkey = var_key(formals[pidx])

            def root(expr: Any) -> int | None:
                return 0 if is_ssa_var(expr) and var_key(expr) == pkey else None
            return self._fn_return_offset_rooted(fn, root, depth + 1)
        finally:
            self._return_offset_active.discard(key)

    def _fn_return_offset_rooted(self, fn: Any, root: Any,
                                 depth: int = 0) -> int | None:
        """The displacement from *root* that *fn* RETURNS, if it returns one."""
        return self._fn_return_routes(fn, root, depth)[1]

    def _fn_return_routes(self, fn: Any, root: Any,
                          depth: int = 0) -> tuple[bool, int | None]:
        """``(a return path reaches *root*, its displacement or None)``.

        Tri-state on purpose: a body with no such path says nothing about
        the answer, while one whose path has no single displacement makes
        the answer unknown (round-16 review F36)."""
        if depth > self._PTR_RESOLVE_DEPTH:
            return (False, None)
        key = (int(getattr(fn, "start", 0)), id(root))
        if key in self._return_offset_active:
            return (False, None)
        self._return_offset_active.add(key)
        try:
            ssaf = self._ssa_func(fn)
            instrs = self._instrs(ssaf)
        except TaintError:
            self._return_offset_active.discard(key)
            return (False, None)
        try:
            # EVERY return that can carry this pointer, joined: a wrapper
            # returning `p + 8` on one path and `p + 24` on another has no
            # single displacement, and taking the first one reported a
            # certainty it did not have (round-12 review F27).
            arms = [src for ins in instrs if "RET" in op_name(ins)
                    for src in self._ret_sources(ins)
                    if self._ptr_resolves(ssaf, src, lambda e: root(e) is not None)]
            if not arms:
                return (False, None)
            return (True, self._merge_offsets(ssaf, arms, root, depth, set()))
        finally:
            self._return_offset_active.discard(key)

    def _fn_return_offset(self, fn: Any, ga: int, depth: int = 0) -> int | None:
        return self._fn_return_offset_rooted(fn, self._global_root(ga), depth)

    def _model_span(self, ins: Any, hit: list[int]) -> int | None:
        """How many bytes a modeled call touches through the argument that
        holds the pointer, when the model and the call both say so."""
        _k, model = lookup_model(
            self.models, self._callee_name(self._resolve_direct_target(ins)))
        length = self._model_len_index(model)
        if length is None:
            return None
        params = self._call_params(ins)
        if length >= len(params):
            return None
        n = self._int_const(params[length])
        return int(n) if n is not None and int(n) > 0 else None

    def _call_sees_global(self, ssaf: Any, call: Any, sites: Any) -> bool:
        """Whether tainted bytes written at any index in *sites* can be live
        at *call*.

        The question is CONTROL FLOW, not IL numbering: `if (flag) reader();
        else memcpy(g, s, 64);` has the write at a lower index than the
        reader and no path between them, so numeric order answered yes for a
        branch that can never carry those bytes (round-11 review F23). Index
        order decides only WITHIN one basic block, where it is the same
        thing. A function with no readable CFG fails closed -- temporal
        uncertainty is not proof of flow, but it is not proof against it
        either."""
        if not sites:
            return True
        try:
            at = int(getattr(call, "instr_index", getattr(call, "address", 0)))
        except Exception:
            return True
        return any(self._il_reaches(ssaf, int(site), at) for site in sites)

    def _il_reaches(self, ssaf: Any, src: int, dst: int) -> bool:
        """Whether IL index *dst* is reachable from *src* through the CFG."""
        blocks = list(getattr(ssaf, "basic_blocks", None) or ())
        if not blocks:
            return True            # no CFG to read: claim nothing

        def holding(idx: int) -> Any:
            for block in blocks:
                try:
                    if int(block.start) <= idx < int(block.end):
                        return block
                except Exception:
                    return None
            return None
        start, goal = holding(src), holding(dst)
        if start is None or goal is None:
            return True
        if dst > src and int(getattr(start, "start", -1)) == int(
                getattr(goal, "start", -2)):
            return True            # same block, later instruction
        # Otherwise a PATH is needed -- including the one a loop's back edge
        # makes from the write round to an earlier call, which is a real way
        # for those bytes to reach it.
        seen: set[int] = set()
        wave = [start]
        while wave:
            nxt = []
            for block in wave:
                try:
                    edges = list(getattr(block, "outgoing_edges", None) or ())
                except Exception:
                    return True
                for edge in edges:
                    target = getattr(edge, "target", None)
                    if target is None:
                        return True
                    key = int(getattr(target, "start", -1))
                    if key in seen:
                        continue
                    if key == int(getattr(goal, "start", -2)):
                        return True
                    seen.add(key)
                    nxt.append(target)
            wave = nxt
        return False

    # How far back an enclosing object may begin, and how many symbolled
    # predecessors the search will walk to find it. An alias/subobject names
    # the inside of ONE object, so the window is the object's own plausible
    # size; exhausting the hop budget is DISCLOSED, never answered as "no
    # enclosing object" (round-12 review F25).
    _ALIAS_WINDOW = 1 << 20
    _ALIAS_HOPS = 64

    def _preceding_objects(self, ga: int) -> tuple[list[int], bool]:
        """``(candidate enclosing objects, budget exhausted)`` for *ga*.

        A symbol is not proof of a separate object: `.set tail, packet + 64`
        names the INTERIOR of one. Several aliases may name the same object
        -- `mid1`, `mid2`, `mid3`, `tail` all inside `packet` -- so the
        search walks the whole CONTIGUOUS run of symbolled predecessors in
        the same section rather than a fixed couple of hops, which hid the
        base reader behind the fourth alias (round-12 review F25)."""
        cached = self._alias_chain_cache.get(ga)
        if cached is not None:
            return cached
        prev = getattr(self.bv, "get_previous_data_var_before", None)
        getsym = getattr(self.bv, "get_symbol_at", None)
        if not callable(prev) or not callable(getsym):
            out: tuple = ([], False)
            self._alias_chain_cache[ga] = out
            return out
        section = self._section_names(ga)
        found: list[int] = []
        exhausted = True          # cleared only by a reason to STOP looking
        at = int(ga)
        for _ in range(self._ALIAS_HOPS * 64):
            if len(found) >= self._ALIAS_HOPS:
                break
            try:
                got = prev(at)
            except Exception:
                break
            addr = got if isinstance(got, int) else getattr(got, "address", None)
            if addr is None or int(addr) >= at:
                exhausted = False   # the section ran out: nothing left to see
                break
            at = int(addr)
            if ga - at > self._ALIAS_WINDOW:
                # A bound reached is not evidence: an object whose base lies
                # further back than the window is still an object this write
                # may be inside (round-13 review F29).
                break
            try:
                named = getsym(at)
            except Exception:
                break
            if named is None:
                continue        # a bare interior variable BN invented
            if self._section_names(at) != section:
                exhausted = False   # another section is another object
                break
            if self._authoritative_width(at) and at + self._authoritative_width(at) <= ga:
                exhausted = False   # a human said where that object ends
                break
            found.append(at)
        out = (found, exhausted)
        self._alias_chain_cache[ga] = out
        return out

    def _section_names(self, addr: int) -> tuple:
        """The sections an address lies in, as a comparable key."""
        getsecs = getattr(self.bv, "get_sections_at", None)
        if not callable(getsecs):
            return ()
        try:
            return tuple(sorted(str(getattr(sec, "name", getattr(sec, "start", "")))
                                for sec in (getsecs(int(addr)) or [])))
        except Exception:
            return ()

    def _reset_object_evidence(self) -> None:
        """Drop everything derived from a PREVIOUS run's global writes.

        The immutable analysis caches (thunk resolution, SSA definitions,
        which bodies name a global at all) stay: they describe the binary,
        not what a source wrote into it."""
        self._global_write_span.clear()
        self._global_tainted_ranges.clear()
        self._global_write_unbounded.clear()
        self._global_keyed_offsets.clear()
        self._global_extent_unproven.clear()
        self._global_search_truncated.clear()
        self._global_refs_truncated.clear()
        # ...and the answers computed FROM them.
        self._global_extent_cache.clear()
        self._canon_global_cache.clear()
        self._global_refs.clear()
        self._global_reach.clear()
        self._global_access_cache.clear()
        self._param_access_cache.clear()
        self._alias_chain_cache.clear()

    def _invalidate_object(self, ga: int) -> None:
        """New evidence about ONE object's extent: drop what was derived
        from the old one, for that object only. Clearing every cached sweep
        instead made each new global write re-derive the whole program's
        reachability, which is quadratic on a binary that fills several
        globals in a row."""
        self._global_extent_cache.pop(ga, None)
        self._global_refs.pop(ga, None)
        self._global_extent_unproven.discard(ga)
        self._alias_chain_cache.pop(ga, None)
        for key in [k for k in self._global_reach if k[1] == ga]:
            del self._global_reach[key]
        for key in [k for k in self._global_access_cache if k[1] == ga]:
            del self._global_access_cache[key]

    def _forward_window(self, ga: int) -> tuple[int, bool]:
        """``(how far the NEXT symbols' storage reaches past *ga*, the alias
        enumeration ran out)``.

        What an unbounded write into this object may have filled: the object
        runs at least through the symbols that follow it contiguously, which
        is where an interior alias of it lives. An ALIAS COUNT never decides
        where it ends -- running out of them is reported, not answered
        (round-17 review F39)."""
        nxt = getattr(self.bv, "get_next_data_var_after", None)
        getsym = getattr(self.bv, "get_symbol_at", None)
        if not callable(nxt) or not callable(getsym):
            return (0, False)
        at, seen_named, end = int(ga), 0, 0
        exhausted = False
        section = self._section_names(ga)
        for _ in range(self._ALIAS_HOPS * 8):
            try:
                got = nxt(at)
            except Exception:
                break
            addr = got if isinstance(got, int) else getattr(got, "address", None)
            if addr is None or int(addr) <= at:
                break
            at = int(addr)
            if at - ga > self._ALIAS_WINDOW or self._section_names(at) != section:
                break
            try:
                if getsym(at) is None:
                    continue
            except Exception:
                break
            if self._authoritative_width(at):
                # Only a type a HUMAN applied ends this object's storage.
                # A type BN inferred says nothing about boundaries: one
                # byte access at `chain + 32` makes that alias an
                # auto-discovered `uint8_t`, and stopping there hid the
                # reader past it (round-16 review F33).
                break
            seen_named += 1
            end = at - ga
            if seen_named >= self._ALIAS_HOPS:
                # An ALIAS COUNT must not decide where an object ends: the
                # 65th interior symbol excluded a reader past it and the run
                # cleared (round-17 review F39). Cover the rest of the
                # credible storage -- to the section's end, bounded by the
                # window -- and record that the enumeration itself did not
                # finish.
                exhausted = True
                end = max(end, self._section_remainder(ga))
                break
        # The LAST named symbol's own storage is part of the run: an alias
        # at `+64` of a 256-byte object is followed by a reader at `+80`,
        # so stopping at the symbol itself excluded the reader (round-14
        # review F33). Carry on to where the run ends.
        if not end:
            return (0, exhausted)
        return (end + self._trailing_extent(ga + end), exhausted)

    def _section_remainder(self, ga: int) -> int:
        """How much of *ga*'s own section lies past it, bounded by the
        window -- the credible storage an unfinished alias enumeration
        leaves unaccounted for."""
        getsecs = getattr(self.bv, "get_sections_at", None)
        if not callable(getsecs):
            return 0
        try:
            ends = [int(getattr(sec, "end", 0)) for sec in (getsecs(int(ga)) or [])]
        except Exception:
            return 0
        ends = [end - ga for end in ends if end > ga]
        return min(max(ends, default=0), self._ALIAS_WINDOW)

    def _trailing_extent(self, at: int) -> int:
        """How far the LAST symbol in a contiguous run reaches: to the next
        SYMBOLLED data variable, else to the end of its section.

        Not to the next data variable of any kind: an untyped object makes
        BN invent a bare variable at each offset something references, so
        stopping at one cuts the run at the very reference being looked for
        (the same rule `_global_extent`'s layout bound applies)."""
        nxt = getattr(self.bv, "get_next_data_var_after", None)
        getsym = getattr(self.bv, "get_symbol_at", None)
        if callable(nxt) and callable(getsym):
            cur = at
            for _ in range(self._ALIAS_HOPS * 8):
                try:
                    got = nxt(cur)
                except Exception:
                    break
                addr = got if isinstance(got, int) else getattr(got, "address", None)
                if addr is None or int(addr) <= cur:
                    break
                cur = int(addr)
                try:
                    if getsym(cur) is not None:
                        return cur - at
                except Exception:
                    break
        getsecs = getattr(self.bv, "get_sections_at", None)
        if callable(getsecs):
            try:
                for sec in (getsecs(at) or []):
                    end = getattr(sec, "end", None)
                    if end is not None and int(end) > at:
                        return int(end) - at
            except Exception:
                pass
        return 1

    def _touches_any_global(self, fn: Any) -> bool:
        """Whether *fn* names ANY writable global address at all.

        The cheap question behind "could an unexamined object matter here":
        a body with no global operand reads no object's bytes, whatever the
        alias search did not get to (round-14 review F35)."""
        key = int(getattr(fn, "start", 0))
        cached = self._touches_global_cache.get(key)
        if cached is not None:
            return cached
        found = False
        try:
            for ins in self._instrs(self._ssa_func(fn)):
                for const in self._instruction_constants(ins):
                    if self._canon_global(const) is not None:
                        found = True
                        break
                if found:
                    break
        except TaintError:
            found = True        # a body that cannot be read claims nothing
        self._touches_global_cache[key] = found
        return found

    def _instruction_constants(self, ins: Any, depth: int = 0) -> list:
        """Every integer constant an instruction's operands name."""
        out: list[int] = []
        stack = [(ins, 0)]
        while stack:
            node, d = stack.pop()
            if node is None or d > 8:
                continue
            value = self._int_const(node)
            if value is not None:
                out.append(int(value))
            for attr in ("src", "dest", "left", "right", "params", "operands"):
                sub = getattr(node, attr, None)
                if isinstance(sub, (list, tuple)):
                    stack.extend((one, d + 1) for one in sub)
                elif sub is not None and hasattr(sub, "operation"):
                    stack.append((sub, d + 1))
        return out

    def _authoritative_width(self, ga: int) -> int:
        """The width of a type a HUMAN applied to this object, else 0.

        BN marks the data variables it discovered itself (`auto_discovered`);
        a type the user applied is evidence about the object, while one
        inferred from an initialiser is only a lower bound (round-13 review
        F31 against F26)."""
        getdv = getattr(self.bv, "get_data_var_at", None)
        if not callable(getdv):
            return 0
        try:
            dv = getdv(ga)
            if dv is None or getattr(dv, "auto_discovered", True):
                return 0
            width = getattr(getattr(dv, "type", None), "width", None)
            return int(width) if width else 0
        except Exception:
            return 0

    def _typed_width(self, ga: int) -> int:
        """The width of the object's RECOVERED type, or 0 when it has none."""
        getdv = getattr(self.bv, "get_data_var_at", None)
        if not callable(getdv):
            return 0
        try:
            width = getattr(getattr(getdv(ga), "type", None), "width", None)
            return int(width) if width else 0
        except Exception:
            return 0

    def _tainted_ranges(self, ga: int) -> list[tuple[int, int]]:
        """The byte intervals of the object at *ga* this run has tainted,
        absolute. Falls back to the whole object when no copy named a
        length -- claiming less than was written is the wrong direction."""
        ranges = [(ga + lo, None if hi is None else ga + hi)
                  for lo, hi in (self._global_tainted_ranges.get(ga) or ())
                  if hi is None or hi > lo]
        return ranges or [(ga, ga + self._global_extent(ga))]

    def _overlap_access(self, fn: Any, ssaf: Any, ga: int) -> Any:
        """A use of a NEIGHBOURING object whose bytes reach into *ga*.

        `.set packet_tail, packet + 64` gives the same storage two symbols.
        A copy of `packet[0..112)` reads the bytes a write to `packet_tail`
        put there, but it names the other symbol, so neither the reference
        sweep nor the classifier saw it and the run cleared (round-10 review
        C1). The test is on INTERVALS: the read's addressed bytes against the
        bytes this run tainted. A read whose length is not a constant -- a
        NUL-terminated `system(packet)`, a runtime `memcpy(.., packet, n)` --
        cannot be shown to stop short, so it overlaps and the leaf says the
        object's extent is what is unproven (round-11 review F21).

        Only an ABUTTING neighbour is considered: a separate object with a
        gap after it is not storage this write can have reached."""
        tainted = self._tainted_ranges(ga)
        start = int(getattr(fn, "start", 0))
        candidates, exhausted = self._preceding_objects(ga)
        if exhausted and self._touches_any_global(fn):
            # The enumeration stopped with plausible enclosing objects
            # unexamined, so "nothing encloses it" was never established --
            # but only where it could MATTER. A fully analysed body that
            # names no global at all reads no object, enclosing or not, so
            # the unexamined storage cannot change its answer (round-14
            # review F35).
            self._global_search_truncated.add(ga)
        for pb in candidates:
            if start not in self._global_ref_functions(pb):
                continue        # this body does not touch that object at all
            authoritative = self._authoritative_width(pb)
            if authoritative and pb + authoritative <= ga:
                # A type a HUMAN applied is an answer about where that
                # object ends -- unlike one BN inferred from an initialiser
                # (round-13 review F31). This object begins past it, so they
                # are separate storage.
                continue
            # NOTE: an INFERRED type is not consulted here. BN sizes an
            # initialised buffer from the bytes it can see -- `char
            # init_packet[128] = "seed"` comes back `char[0x5]` -- so a type
            # is a LOWER bound on the object and can never prove where it
            # ends (round-12 review F26, the same rule `_global_extent`
            # applies). Separation is established by SECTION and by the
            # interval test below, never by a width BN inferred.
            uses: list = []
            self._pointer_access(
                fn, self._global_base(ssaf, pb), fallback=None,
                offset_of=lambda ins: self._global_use_offset(ins, pb),
                offset_expr=lambda f, e: self._global_ptr_offset(f, e, pb),
                span_of=self._model_span,
                names_object=lambda ins: self._names_global(ins, pb),
                collect=uses)
            for use in uses:
                if use.offset_known and use.span:
                    lo, hi, bounded = pb + use.offset, pb + use.offset + int(use.span), True
                elif use.offset_known:
                    # A length nothing states reads until it stops -- which
                    # this walk cannot bound.
                    lo, hi, bounded = pb + use.offset, None, False
                else:
                    lo, hi, bounded = pb, None, False
                for tlo, thi in tainted:
                    if thi is not None and lo >= thi:
                        continue
                    if hi is not None and hi <= tlo:
                        continue
                    if not bounded:
                        # Disclose WHY it could not be ruled out, rather than
                        # presenting an unbounded read as a measured one.
                        self._global_extent_unproven.add(ga)
                    return _Use(use.at, use.kind, max(lo, tlo) - ga, use.seed,
                                use.span, use.offset_known and bounded)
        return None

    def _global_access(self, fn: Any, ga: int) -> Any:
        """:meth:`_pointer_access` rooted at the global at *ga*, cached."""
        key = (int(getattr(fn, "start", 0)), int(ga))
        cached = self._global_access_cache.get(key)
        if cached is not None:
            return cached
        try:
            ssaf = self._ssa_func(fn)
        except TaintError:
            out = _Use(self._global_ref_functions(ga).get(key[0]),
                       "unreadable", 0, None, None, True)
            self._global_access_cache[key] = out
            return out
        out = self._pointer_access(
            fn, self._global_base(ssaf, ga),
            fallback=self._global_ref_functions(ga).get(key[0]),
            offset_of=lambda ins: self._global_use_offset(ins, ga),
            offset_expr=lambda f, e: self._global_ptr_offset(f, e, ga),
            span_of=self._model_span,
            names_object=lambda ins: self._names_global(ins, ga))
        if out.at is None or out.kind == "write_only":
            # Nothing of THIS object was read here -- but an access to a
            # neighbouring symbol may still span into it (C1).
            overlap = self._overlap_access(fn, ssaf, ga)
            if overlap is not None:
                out = overlap
        self._global_access_cache[key] = out
        return out

    def _indirect_sites(self, fn: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """``(pinned targets, unresolved call addresses)`` for the indirect
        calls in *fn*.

        BN's ``Function.callees`` lists resolved edges only, so a body whose
        whole job is `cb()` looks like a leaf to a call-graph walk (round-3
        review I2). Each indirect site is re-resolved the way the forward
        walk resolves one -- an agent ``--resolve-map`` pin first, then the
        value set -- and whatever still does not resolve is named, so the
        walk reports "I could not follow this" instead of "there is nothing
        here"."""
        key = int(getattr(fn, "start", 0))
        cached = self._indirect_sites_cache.get(key)
        if cached is not None:
            return cached
        pinned: list[int] = []
        unresolved: list[int] = []
        try:
            instrs = self._instrs(self._ssa_func(fn))
        except TaintError:
            instrs = []
        for ins in instrs:
            if not self._is_call(ins) or self._resolve_direct_target(ins) is not None:
                continue
            addr = int(getattr(ins, "address", 0))
            targets = tuple(self.resolve_map.get(addr) or ()) or tuple(
                self._call_targets_from_pvs(
                    getattr(getattr(ins, "dest", None), "possible_values", None)))
            if targets:
                pinned.extend(int(t) for t in targets)
            else:
                unresolved.append(addr)
        out = (tuple(pinned), tuple(unresolved))
        self._indirect_sites_cache[key] = out
        return out

    def _global_refs_reachable(self, fn: Any, ga: int, budget: int) -> tuple[tuple, dict]:
        """``(reads, reach)`` for the global at *ga* as seen from *fn*.

        ``reads`` is ``((function start, address, kind, offset), ...)`` for
        the functions that can READ the global and sit within *budget*
        further call levels of *fn* (level 0 is *fn* itself), ordered by
        function address so one leaf is byte-identical across the fixpoint's
        repeated visits. A function whose every reference is a store into the
        global is not a read and is left out (F3), and ``offset`` names which
        bytes of the object the use reached (F8).

        ``reach`` is the structured status of the search:
        ``complete`` -- the walk ran out of call-graph, or every function that
        references the global was examined, so "no reader" is an answer; or
        ``truncated`` with ``cause`` ``depth_budget``/``function_budget``,
        ``functions_examined``/``function_budget`` and the referencing
        functions left ``unexamined`` -- the walk stopped early with
        somewhere the bytes could still be read unvisited, so "no reader" is
        NOT an answer and the caller must disclose it (F2).

        A reachable in-binary DESCENDANT whose body BN never analysed is
        itself a truncation: its own references cannot be classified, so a
        wrapper around an unread consumer answers ``unread_callee_body``
        exactly as the immediate callee does (round-3 review F9), and the
        functions are returned in ``reach["unread_bodies"]`` so the caller
        can record them as analysis-incomplete.

        The walk follows BN's ``Function.callees`` edges only -- a resolved
        call graph, the same evidence `bn dataflow callgraph` renders. An
        unresolved indirect edge simply contributes no successor."""
        known = self._global_ref_functions(ga)
        if not known:
            return (), self._reach_status(None, 0, [], {})
        start = int(getattr(fn, "start", 0))
        key = (start, int(ga), int(budget))
        cached = self._global_reach.get(key)
        if cached is not None:
            return cached
        found: dict[int, tuple] = {}
        unread: dict[int, str] = {}
        opaque: list[int] = []
        # `seen` is what the walk has QUEUED (the bound that keeps a dense
        # graph finite); `done` is what it actually examined. Only the latter
        # can rule a reference out -- a function discovered on the last wave
        # and then dropped by the depth bound was never looked at.
        seen = {start}
        done: set[int] = set()
        wave, level, cut = [fn], 0, None
        while wave:
            if level > budget:
                cut = "depth_budget"
                break
            nxt = []
            for cur in wave:
                cs = int(getattr(cur, "start", 0))
                done.add(cs)
                reason = str(self._not_internal_reason(cur) or "")
                if reason.startswith("analysis_skipped:"):
                    # BN21-REACH-01: a model is ADDITIVE evidence over a body,
                    # never a substitute for one. Exempting a modeled name
                    # here let a truthful partial overlay (`arg:0 -> ret`)
                    # DELETE the disclosure that this in-binary body was never
                    # read -- accurate positive evidence made the answer less
                    # conservative, which is exactly backwards. An import has
                    # no body and never reaches this branch.
                    # F9: an in-binary descendant BN measured and skipped.
                    # Nothing read its references, so the walk past it proves
                    # nothing -- whatever depth it sits at.
                    unread[cs] = str(getattr(cur, "name", None) or hex(cs))
                    for site in self._native_global_refs(cur, ga)[:1]:
                        found.setdefault(
                            cs, _Use(site, "unreadable", 0, None, None, True))
                    continue
                if cs not in found:
                    # EVERY examined function is classified, not just the
                    # ones BN records a reference from: a pointer handed back
                    # through a CHAIN of returning wrappers reaches a reader
                    # that never names the address itself (round-3 review
                    # F18). A function with no use of the object answers with
                    # no address and costs one bounded scan.
                    use = self._global_access(cur, ga)
                    if use.at is not None:
                        found[cs] = use
                try:
                    callees = list(getattr(cur, "callees", None) or ())
                except Exception:
                    callees = []
                # I2: `callees` lists resolved edges only. Follow the indirect
                # sites this run CAN pin, and name the ones it cannot --
                # inside real BODIES only. A PLT stub is a register-indirect
                # jump through the GOT by construction; counting it as a call
                # bn failed to resolve would raise this boundary on every
                # library call in the walk.
                pinned, unpinned = (
                    self._indirect_sites(cur) if self._is_internal(cur) else ((), ()))
                opaque.extend(unpinned)
                for taddr in pinned:
                    pinned_fn = function_at(self.bv, taddr)
                    if pinned_fn is not None:
                        callees.append(pinned_fn)
                for callee in callees:
                    nstart = getattr(callee, "start", None)
                    if nstart is None:
                        continue
                    nstart = int(nstart)
                    if nstart in seen:
                        continue
                    if len(seen) >= self._GLOBAL_REACH_FUNCS:
                        cut = "function_budget"
                        continue
                    seen.add(nstart)
                    nxt.append(callee)
            # The budget stops DISCOVERY, never examination: everything
            # already queued is still looked at, so the bound only hides what
            # it genuinely refused to enqueue.
            wave, level = nxt, level + 1
        unexamined = sorted(set(known) - done)
        if not unexamined:
            # Every function BN records a reference from was examined, so the
            # bound that stopped the walk cannot have hidden a reader.
            cut = None
        if opaque and cut is None:
            # I2: control left the examined region through a call bn could
            # not name, so "no reader is reachable" was never established.
            cut = "unresolved_call_target"
        scan_cut = next((_SCAN_KIND_CAUSES[u.kind] for _a, u in
                         sorted(found.items()) if u.kind in _SCAN_KIND_CAUSES), None)
        if scan_cut and cut is None:
            # BN21-BOUND-02/03: a classification that ran out of nodes, or out
            # of internal return-pointer depth, did not finish -- so "these
            # are the readers" is not an answer either.
            cut = scan_cut
        if unread:
            # An unread descendant outranks a budget: the walk did not stop
            # early, it walked INTO something it could not read (F9).
            cut = "unread_callee_body"
        status = self._reach_status(cut, len(done), unexamined, known)
        status["unread_bodies"] = [
            {"name": unread[k], "address": hex(k)} for k in sorted(unread)]
        status["unresolved_calls"] = [hex(a) for a in sorted(set(opaque))[:8]]
        out = (tuple(sorted((st, u.at, u.kind, u.offset, u.seed or "",
                             u.offset_known) for st, u in found.items())),
               status)
        self._global_reach[key] = out
        return out

    def _reach_status(self, cause: str | None, examined: int,
                      unexamined: list[int],
                      sites: dict[int, int] | None = None) -> dict[str, Any]:
        """The structured status of one reachability search.

        ``unexamined_reference_sites`` is the NATIVE evidence that the answer
        is unfinished: for every function BN records a reference to this
        global from that the walk never examined, the function and the
        instruction address of that reference. An unread callee body records
        no code reference of its own, so these sites are what stands behind
        its boundary (round-3 review F6)."""
        sites = sites or {}
        return {
            "status": "complete" if cause is None else "truncated",
            "cause": cause,
            "functions_examined": int(examined),
            "function_budget": self._GLOBAL_REACH_FUNCS,
            "unexamined_references_total": len(unexamined),
            "unexamined_references": [hex(a) for a in unexamined[:8]],
            "unread_bodies": [],
            "unresolved_calls": [],
            "unexamined_reference_sites": [
                {"name": str(getattr(function_at(self.bv, s), "name", None)
                             or self._callee_name(s) or hex(s)),
                 "address": hex(s), "at": hex(sites.get(s, s))}
                for s in unexamined[:8]],
        }

    def _global_continuation_leaf(self, ins: Any, ga: int, callee_fn: Any,
                                  references: tuple, descended: bool,
                                  reach: dict[str, Any]) -> dict[str, Any]:
        """The blocking frontier for tainted global bytes crossing a call the
        walk does not carry them across (IC03 / TT-PILOT-FW03-01).

        Named references and a truncated search are not alternatives: an
        unread callee body carries BOTH -- the native code reference proving
        it touches this global, and the fact that its use of the bytes could
        not be read. ``access`` says which, and the prose never claims a read
        it did not see."""
        rows = []
        for start, at, kind, offset, seed, offset_known in references[:8]:
            rfn = function_at(self.bv, start)
            nm = str(getattr(rfn, "name", None) or self._callee_name(start) or hex(start))
            rows.append({"name": nm,
                         "display_name": (_display_name(rfn) or nm) if rfn is not None else nm,
                         "address": hex(start), "at": hex(at), "access": kind,
                         "offset": int(offset),
                         # M1: whether that displacement was ESTABLISHED. A
                         # pointer whose chain this scan could not follow has
                         # an UNKNOWN displacement, which is not the base.
                         "offset_known": bool(offset_known),
                         "seed": str(seed or "") or None})
        crossed = ("this call is descended with its tainted argument(s) only"
                   if descended else
                   "no argument of this call carries taint, so it is not descended")
        if rows:
            seed_at = rows[0]["address"]
        elif callee_fn is not None:
            seed_at = hex(int(getattr(callee_fn, "start", 0)))
        else:
            sites = reach.get("unexamined_reference_sites") or []
            seed_at = str(sites[0].get("address")) if sites else hex(ga)
        # R4: a locator the agent can RUN, or none at all. A reader that
        # names the object by constant address has no variable to seed, and
        # `--source var:VAR` there was a template with nothing to fill in --
        # every spelling of it is rejected by the var: grammar.
        # The function and the locator must come from the SAME reference: a
        # locator read off one row and an address off another produced
        # `-f <producer> --source arg:system:0` for a producer with no
        # `system` call in it, which exits 2 (round-13 review F32).
        seeded = next((r for r in rows if r.get("seed")), None)
        # A concrete locator is a COMMAND, not a template: nothing in it is
        # left to fill in, so it must not carry the fill-in-the-blanks marker
        # that tells a reader (and the paste test) it cannot be run.
        reseed = (f"`bn taint forward -f {seeded['address']} "
                  f"--source {seeded['seed']}`" if seeded else None)
        parts = []
        if rows:
            where = ", ".join(
                f"{r['display_name']} @ {r['at']}"
                + (f" (+{hex(r['offset'])})" if r["offset"] and r["offset_known"]
                   else (" (+?)" if not r["offset_known"] else ""))
                + f" ({r['access']})" for r in rows)
            proven = any(r["access"] in _PROVEN_GLOBAL_READS for r in rows)
            verb = "reads" if proven else "may read"
            parts.append(
                f"tainted bytes were copied into the writable global {hex(ga)}, and code "
                f"reachable from this call {verb} it ({where}).")
        else:
            crossing = ("this call's target did not resolve"
                        if reach.get("cause") == "unresolved_call_target"
                        else "this call crosses into code whose use of it could not be read")
            parts.append(
                f"tainted bytes were copied into the writable global {hex(ga)}, and "
                f"{crossing}.")
        if reach.get("status") == "truncated":
            bound = {
                "depth_budget": "the configured interprocedural depth",
                "unresolved_thunk": "an in-binary thunk whose tail call did not resolve",
                "unresolved_call_target": (
                    "an indirect call whose target bn could not resolve "
                    "(pin it with --resolve-map and re-run)"),
                "reference_sweep_budget": (
                    "the bound on how much of one object's address range the "
                    "reference sweep will ask for, so references past it were "
                    "never enumerated"),
                "unread_callee_body": (
                    "an in-binary callee body Binary Ninja did not analyse "
                    "(re-run after `bn decompile <callee> --force-analysis`)"),
                "object_search_budget": (
                    "the bound on how far back the search for an ENCLOSING "
                    "object may look, so storage this write may be inside "
                    "was never examined -- this bound is internal, "
                    "`--max-depth` does not move it"),
                "object_extent_unproven": (
                    "an object with no recovered type that is written PAST "
                    "the next symbol in the layout -- so where it ends, and "
                    "which references belong to it, are not established "
                    "(give the global a type with `bn retype` and re-run)"),
                "pointer_scan_budget": (
                    "the bound on how many expression nodes ONE function's "
                    "pointer scan may visit, so what the rest of that body "
                    "does with the object was never classified -- this bound "
                    "is internal, `--max-depth` does not move it"),
                "pointer_return_depth": (
                    "the bound on how far a pointer may be followed back "
                    "through RETURNING wrappers, so whether the object comes "
                    "back out of that chain is unknown -- this bound is "
                    "internal, `--max-depth` does not move it"),
            }.get(str(reach.get("cause")),
                  f"the {reach.get('function_budget')}-function reachability bound")
            unread_named = ", ".join(
                f"{b.get('name')} @ {b.get('address')}"
                for b in (reach.get("unread_bodies") or []))
            if unread_named:
                bound += f" -- {unread_named}"
            opaque_named = ", ".join(reach.get("unresolved_calls") or [])
            if opaque_named:
                bound += f" (unresolved call(s) at {opaque_named})"
            unreached = ", ".join(
                f"{r.get('name')} @ {r.get('at')}"
                for r in (reach.get("unexamined_reference_sites") or [])) or "n/a"
            parts.append(
                f"The search for code reading it past this call stopped at {bound} after "
                f"{reach.get('functions_examined')} function(s), with "
                f"{reach.get('unexamined_references_total')} referencing function(s) never "
                f"examined ({unreached}), so whether those bytes reach a sink past this "
                f"call is UNKNOWN, not clear.")
        advice = (f" then re-seed inside the consumer on the value it loads ({reseed})."
                  if reseed else
                  " the consumer reaches those bytes by constant address, so there is no"
                  " variable inside it to seed: audit the references by hand.")
        parts.append(
            f"Taint is tracked per function, and {crossed}: what those bytes reach past "
            f"this point is NOT followed -- a 'no sinks reached' result here is not proof "
            f"of safety. List every reference with `bn xrefs {hex(ga)}` and audit them,"
            f"{advice}")
        detail = " ".join(parts)
        return {
            "kind": "global_memory_continuation",
            "address": hex(int(getattr(ins, "address", 0))),
            "global": {"address": hex(ga), "label": f"glob_{hex(ga)}",
                       # WHICH bytes of the object the walk keyed here: an
                       # interior write (`memcpy(g + 8, ..)`) is the same
                       # object, and naming the offset keeps a partial write
                       # from reading as the whole object (R1).
                       "written_offsets": [
                           hex(o) for o in
                           sorted(self._global_keyed_offsets.get(ga) or ())[:8]]},
            "callee": None if callee_fn is None else {
                "name": str(getattr(callee_fn, "name", "?")),
                "display_name": (_display_name(callee_fn)
                                 or str(getattr(callee_fn, "name", "?"))),
                "address": hex(int(getattr(callee_fn, "start", 0)))},
            "references": rows,
            # BN21-BOUND-02: whether a reference was PROVEN to read the bytes,
            # or merely could not be ruled out. Every surface renders the one
            # the evidence supports instead of claiming a read it did not see.
            "reads_proven": any(r["access"] in _PROVEN_GLOBAL_READS for r in rows),
            "reachability": reach,
            "call_descended": bool(descended),
            "il_text": str(ins),
            "reseed": reseed,
            "detail": detail,
        }

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
        leaves: list[dict[str, Any]] = _SalvageList(self._salvage["leaves"])
        findings: list[dict[str, Any]] = _SalvageList(self._salvage["findings"])
        recorded_sinks: set[tuple] = set()
        processed_calls: set[tuple] = set()  # (call_addr, tainted-arg-set) already descended
        out_params: set[int] = set()         # this func's params whose pointee got tainted
        out_param_elems: set = set()         # (param_idx, field) descriptor-array elems filled (#319b)
        # IC03: the writable globals THIS run has tainted, {address: node}, kept
        # by `taint_node` as the taint set grows. A global is program state, not
        # frame state: the call branch below asks, per global, whether code
        # reachable from a call references it, and discloses the boundary when
        # it does. Tracking it here (rather than rescanning `tainted` at every
        # call on every fixpoint pass) keeps that question O(1) to ask.
        live_globals: dict[int, tuple] = {}
        # ...and WHEN each became tainted, as the IL index of the first
        # instruction that tainted it. The fixpoint keeps `live_globals`
        # across passes, so a later pass saw a global live at every call --
        # including calls that run BEFORE the only write, which were then
        # reported as carrying tainted bytes across (round-10 review M2).
        live_global_sites: dict[int, set] = {}
        # (call address, global address) already disclosed, so the fixpoint's
        # repeated visits report one boundary, not one per pass.
        global_frontiers_seen: set[tuple[int, int]] = set()
        # Call targets resolved for the global question, per call address, so
        # the fixpoint's repeated visits pay BN's value-set query once.
        global_call_targets: dict[int, tuple[int, ...]] = {}
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
            key = node[0]
            if type(key) is tuple and len(key) == 2 and key[0] == "global" \
                    and ins is not None:
                # EVERY site, deduped -- INCLUDING a write to a global this
                # run has already tainted. Recording only the first one let a
                # mutually exclusive earlier write hide the later write that
                # actually reaches the reader (round-12 review F24), so this
                # runs before the already-tainted early return.
                live_global_sites.setdefault(int(key[1]), set()).add(ins_key(ins))
            if node in tainted:
                return False
            tainted.add(node)
            if type(key) is tuple and len(key) == 2 and key[0] == "global":
                live_globals.setdefault(int(key[1]), node)
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

        def note_global_continuation(ins: Any, target: int | None, descended: bool) -> None:
            """Disclose tainted global bytes crossing a call boundary (IC03).

            One leaf per (call address, global address). It is emitted when a
            resolved in-binary target -- following a PLT/veneer thunk with the
            same shared resolver the descent uses (round-3 review F1) -- can
            reach a function that READS that global, and also when the
            reachability search was truncated before it could rule a reader
            out (F2). An unresolved indirect call has no body whose references
            could be read, so none is invented for it -- the existing
            ``indirect_call_unresolved`` frontier owns that call."""
            addr = int(getattr(ins, "address", 0))
            # Cheapest filter first: a global already disclosed at this call,
            # or one BN records no reference to anywhere, asks nothing of the
            # call graph. The fixpoint revisits this instruction on every pass,
            # so an unreachable-but-referenced global must not re-resolve the
            # call's targets each time -- memoize them per call site.
            pending = [ga for ga in sorted(live_globals)
                       if (addr, ga) not in global_frontiers_seen
                       and self._call_sees_global(ssaf, ins,
                                                  live_global_sites.get(ga))
                       and (self._global_ref_functions(ga)
                            or ga in self._global_refs_truncated)]
            if not pending:
                return
            cands = global_call_targets.get(addr)
            if cands is None:
                if target is not None:
                    cands = (target,)
                else:
                    cands = tuple(self.resolve_map.get(addr) or ()) or tuple(
                        self._call_targets_from_pvs(
                            getattr(getattr(ins, "dest", None), "possible_values", None)))
                global_call_targets[addr] = cands
            if not cands:
                # F10: control leaves this function for somewhere bn could not
                # name, with tainted global bytes live and code in this binary
                # that references them. Nothing proves the target is not one
                # of those readers, so the boundary is disclosed rather than
                # skipped. Only when no ARGUMENT carries taint: an indirect
                # call that does carry one already raises
                # `indirect_call_unresolved`, and two leaves for one call
                # would say the same thing twice. Pinning the site with
                # --resolve-map (or a value set bn can pin) replaces this with
                # the real target's answer.
                if descended:
                    return
                for ga in list(pending):
                    known = self._global_ref_functions(ga)
                    reach = self._reach_status(
                        "unresolved_call_target", 0, sorted(known), known)
                    global_frontiers_seen.add((addr, ga))
                    pending.remove(ga)
                    leaf = self._global_continuation_leaf(
                        ins, ga, None, (), descended, reach)
                    leaf["dest_expr"] = str(getattr(ins, "dest", ""))
                    if leaf not in leaves:
                        leaves.append(leaf)
                return
            budget = max(0, max_depth - depth)
            for taddr in cands:
                cfn = function_at(self.bv, taddr)
                if cfn is None:
                    continue
                unread: str | None = None
                reason = self._not_internal_reason(cfn)
                if reason is not None:
                    # A .plt/veneer thunk is not a body, but it is not a dead
                    # end either: resolve it with the same helper the descent
                    # uses (#807), so a global consumed behind a thunk is not
                    # dropped (F1). Two in-binary shapes then fail CLOSED,
                    # because their references cannot be read and "nothing
                    # reads the global" is not an answer: a thunk whose tail
                    # call does not resolve, and a body BN never analysed (an
                    # oversized function past analysis.limits.maxFunctionSize,
                    # a failed lift) -- round-3 review R3. A resolved import is
                    # external; the model/unmodeled-callee machinery owns that
                    # call, as before.
                    resolved = self._follow_thunk_cached(cfn)
                    if resolved is not None:
                        rstart = int(getattr(resolved, "start", 0))
                        if self._is_internal(resolved):
                            cfn = resolved
                        elif str(self._not_internal_reason(resolved) or "").startswith(
                                "analysis_skipped:"):
                            # A veneer onto an IN-BINARY body BN measured and
                            # skipped. Following the thunk and then calling
                            # the result "external" lost both the boundary
                            # and the analysis-incomplete signal that the
                            # SAME body gets when it is called directly
                            # (round-3 review F16). Keep the resolved target
                            # and fail closed on it.
                            cfn = resolved
                            unread = "unread_callee_body"
                            self._analysis_incomplete.add(
                                str(getattr(resolved, "name", None) or hex(rstart)))
                        elif any(self._global_access(cfn, ga).at is not None
                                 for ga in pending):
                            # BN calls a one-line `void reader(void) {
                            # system(g + 80); }` a THUNK once the compiler
                            # turns it into a tail jump, and following it to
                            # the import made the whole body external -- so
                            # the bytes it reads were never disclosed. Its
                            # own body IS readable and does reference the
                            # tainted object, so it is kept as the callee.
                            # A genuine PLT stub references nothing and
                            # still falls through to external.
                            pass
                        else:
                            continue      # a PLT stub onto an import: external
                    elif reason == "thunk":
                        unread = "unresolved_thunk"
                    elif self._unanalysed_body_phrase(reason) is not None and (
                            # A body BN MEASURED and skipped is a hole whatever
                            # its name is modeled as (BN21-REACH-01); a name
                            # with no IL at all may still be an import stub,
                            # whose model owns the story as before.
                            reason.startswith("analysis_skipped:")
                            or lookup_model(self.models,
                                            self._callee_name(int(taddr)))[1] is None):
                        # An IN-BINARY body whose MLIL this run cannot read --
                        # BN skipped it (oversized past
                        # analysis.limits.maxFunctionSize, a failed lift) or
                        # produced none at all. Its references cannot be
                        # classified, so "nothing reads the global" is not an
                        # answer and the walk must not continue to an
                        # all-clear (round-3 review F6). A model on the name
                        # does NOT excuse it (BN21-REACH-01): an import has no
                        # body and never gets here, while a measured, skipped
                        # in-binary body is a hole whatever its name is
                        # modeled as.
                        unread = "unread_callee_body"
                        self._analysis_incomplete.add(
                            str(getattr(cfn, "name", None) or hex(int(taddr))))
                    else:
                        continue
                for ga in list(pending):
                    if unread is not None:
                        # Evidence for a body whose IL does not exist: BN
                        # records no code reference FROM a function it never
                        # analysed, so the code-reference DB cannot speak for
                        # it. A bounded native decode can -- and does not
                        # depend on analysis. Whatever it finds is cited
                        # `unreadable`, so the prose says "may read", never
                        # "reads"; the references the walk never reached are
                        # listed beside it.
                        known = self._global_ref_functions(ga)
                        own = int(getattr(cfn, "start", 0))
                        if own in known:
                            sites = (known[own],)
                        elif unread == "unread_callee_body":
                            sites = self._native_global_refs(cfn, ga)
                        else:
                            sites = ()
                        refs = tuple((own, a, "unreadable", 0, "", True)
                                     for a in sites[:8])
                        reach = self._reach_status(
                            unread, 0,
                            sorted(k for k in known if not (refs and k == own)), known)
                    else:
                        refs, reach = self._global_refs_reachable(cfn, ga, budget)
                        if ga in self._global_search_truncated \
                                and reach["cause"] is None \
                                and (refs or self._touches_any_global(cfn)):
                            # F29: the search for an enclosing object ran out
                            # of budget with storage unexamined.
                            reach = dict(reach, status="truncated",
                                         cause="object_search_budget")
                        if ga in self._global_extent_unproven and refs \
                                and reach["cause"] is None:
                            # Doubt about where the object ENDS qualifies the
                            # readers it found; on its own it is not a
                            # boundary, or every unbounded copy into a global
                            # would raise one with nothing behind it. A
                            # search that ran out of budget (above) is
                            # different: there the doubt IS the finding.
                            # BN21-OBJ-01: the object's own extent is not
                            # established, so "these are its references" is
                            # not established either.
                            reach = dict(reach, status="truncated",
                                         cause="object_extent_unproven")
                        if ga in self._global_refs_truncated \
                                and reach["cause"] is None:
                            # F15: the object is larger than the reference
                            # sweep's chunk budget, so "these are all the
                            # references" was never established.
                            reach = dict(reach, status="truncated",
                                         cause="reference_sweep_budget")
                        for body in reach.get("unread_bodies") or ():
                            # F9: a wrapper around an unread consumer is the
                            # same hole as calling it directly.
                            self._analysis_incomplete.add(str(body.get("name")))
                    if not refs and reach["status"] != "truncated":
                        continue
                    global_frontiers_seen.add((addr, ga))
                    pending.remove(ga)
                    leaf = self._global_continuation_leaf(
                        ins, ga, cfn, refs, descended, reach)
                    if leaf not in leaves:
                        leaves.append(leaf)
                    # A BUDGET really did cut coverage short: say so with the
                    # cause, so `truncated`/`truncation_cause` and the
                    # remediation agree with the leaf (F2). An unreadable
                    # callee is not a budget -- it is disclosed by the leaf
                    # and, for an unread body, by analysis_incomplete.
                    # BN21-BOUND-01/02/03: the sweep, the pointer-scan node
                    # pool and the internal return-pointer depth are bounds
                    # too. Each was disclosed by the leaf and absent from the
                    # aggregate, so a consumer reading `stats` alone saw a
                    # COMPLETE run.
                    aggregate = {
                        "depth_budget": "max_depth",
                        "function_budget": "global_reference_budget",
                        "reference_sweep_budget": "global_reference_budget",
                        "object_extent_unproven": "global_object_extent",
                        "object_search_budget": "global_object_search",
                        "pointer_scan_budget": "pointer_scan_budget",
                        "pointer_return_depth": "pointer_return_depth",
                    }.get(str(reach["cause"]))
                    if aggregate:
                        self._truncated = True
                        self._truncation_causes.add(aggregate)

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
                            self._note_write_span(ssaf, params, k, model)
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
                        # A source operand that ALSO fires this model's sink is
                        # two facts, not one: the copy still happened, and the
                        # destination still holds those bytes. Suppressing the
                        # copy note there (because `strcpy` flags its source)
                        # hid the propagation behind the sink -- the half of
                        # #44 that stayed open. Only the WORDING changes with
                        # the sink: the note never claims "not flagged" about
                        # an operand that was.
                        sink_args = (model.get("sink") or {}).get("tainted_args") or []
                        if src_i is not None and src_i >= 1:
                            tail = ("also reported as a sink of this call"
                                    if src_i in sink_args
                                    else "not itself flagged as a sink")
                            add_assumption(
                                f"tainted buffer copied into the destination of "
                                f"{self._prose_name(name or '?')} at {hex(int(getattr(ins, 'address', 0)))} "
                                f"(arg{src_i} -> arg0); propagated to the destination, "
                                f"{tail}")
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
                                # A printf-family destination is written too,
                                # by a length no model states: the object's
                                # extent must stop being clipped by what BN
                                # inferred, exactly as for a copy (round-14
                                # review F33).
                                self._note_write_span(ssaf, params, k, model)
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
                    # IC03 / TT-PILOT-FW03-01: tainted bytes that landed in a
                    # keyable writable global are PROGRAM state. This walk keeps
                    # taint per function and hands nothing but arguments across a
                    # call, so a callee reading that global consumes bytes the
                    # walk never follows -- and when no argument is tainted the
                    # call below is not even descended. Disclose that boundary
                    # wherever BN's own code references show a reader reachable
                    # from the call, for a descended call too (descent carries
                    # the ARGUMENT's taint, never the global's). Evidence-gated
                    # on both sides: no tainted global, no reachable reference,
                    # or no code-ref DB means nothing is emitted, so this is
                    # never a blanket marker on every call after a global write.
                    if live_globals:
                        note_global_continuation(ins, target, bool(tainted_args))
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
                        elif (gs := self._global_addr(ssaf, dest)) is not None:
                            # A direct store into a known writable global is the
                            # SAME event a modeled copy records -- an inlined
                            # `memcpy` at -O1 is a run of these -- so it is keyed
                            # by the object and records the bytes it wrote.
                            # Falling to `coarse_memory_store` instead left the
                            # continuation machinery with nothing to work from
                            # (campaign item, closed here).
                            self._note_store_span(ssaf, dest, gs, ins)
                            if taint_node(((("global", gs)), None), f"glob_{hex(gs)}",
                                          ins, "store into tainted global "
                                          "(object base, global_approx)", reads):
                                changed = True
                                add_assumption(
                                    "global buffer aliasing modeled coarsely by "
                                    "object base (global_approx)")
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
                    short = self._format_destination_shortfall(c, ssaf, callee, model, params)
                    if short is not None:
                        add_leaf(short)
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
        # Shared by every seed and recursive ascent in this run. Reference
        # coverage is per function/site, not per demanded formal.
        self._bw_caller_frontiers: dict[int, dict[str, Any]] = {}
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

        # An identical locator given twice is one query: expanding it twice
        # repeated every site row, seed and slice. Distinct locators -- another
        # callee at the same mapped address included -- stay distinct.
        unique: dict[str, dict[str, Any]] = {}
        for sink in sinks:
            unique.setdefault(json.dumps(sink, sort_keys=True, default=repr), sink)
        sinks = list(unique.values())
        # Resolve argument locators into individual call sites BEFORE seeding:
        # one site's SSA reads must never hide another site's constant or missing
        # operand. Non-argument locators and absent calls retain their refusal path.
        selections: list[tuple[dict[str, Any], Any]] = []
        for sink in sinks:
            sites = (self._find_callsites(instrs, sink["callee"], resolve_indirect=True)
                     if sink.get("kind") == "arg" else [])
            selections.extend((sink, site) for site in sites or [None])
        sink_status: list[dict[str, Any]] = []
        errors: list[tuple[dict[str, Any], str]] = []
        for sink, site in selections:
            desc = self._describe_locator(sink)
            site_instrs = instrs if site is None else [site]
            if site is not None:
                desc["address"] = hex(int(site.address))
                params = self._call_params(site)
                idx = int(sink["index"])
                if 0 <= idx < len(params) and self._sink_role(sink) == "value":
                    operand = params[idx]
                    if op_name(operand) == "MLIL_CONST":
                        self._note_indirect_anchors(site_instrs, sink["callee"], self._bw_assume)
                        sink_status.append({
                            **desc, "status": "constant", "seeded": False,
                            "scalar": {"kind": "constant", "value": int(operand.constant)},
                            "destination_fit": "unproved"})
                        continue
            try:
                seeds = self._seed_backward(func, ssaf, site_instrs, sink)
            except TaintError as exc:
                if self._bw_unseeded_demand(func, ssaf, site_instrs, sink):
                    sink_status.append({
                        **desc, "status": "demand", "seeded": False,
                        "note": "the selected operand has a semantic demand; "
                                "its pointer bytes or argument role remain unresolved "
                                "(see frontier leaves)"})
                    continue
                errors.append((desc, str(exc)))
                sink_status.append({**desc, "status": "unseeded", "seeded": False,
                                    "note": str(exc)})
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
                    # An `arg:` sink that reads through the argument its
                    # locator SELECTS (popen's command, write's buffer) has a
                    # demand: it starts there, at offset 0 of what that
                    # operand points to (round 1), never at another argument
                    # that reads the same SSA value (round 2). The sink
                    # model's role for the argument decides (round 6); an
                    # argument with no explicit value role is walked as a
                    # pointer demand too (round 7). A malloc size or a
                    # memcpy length is a value, with no demand, as on base. A
                    # seed recovered from a register has no operand; its
                    # demand starts at the seed itself.
                    anchor = None
                    role = "value"
                    operand = None
                    if sink.get("kind") == "arg":
                        sparams = self._call_params(sink_ins)
                        sidx = int(sink.get("index", -1))
                        operand = sparams[sidx] if 0 <= sidx < len(sparams) else None
                        role = self._sink_role(sink)
                        if role != "value":
                            aexpr = (self._seed_anchor(seed_var, operand)
                                     if operand is not None
                                     else seed_var.expr if isinstance(seed_var, _BufferSeed)
                                     else seed_var)
                            anchor = (aexpr, 0) if aexpr is not None else None
                    for sl in self._backward_slice(func, seed_var, 0, max_depth, set(),
                                                   seed_ins=sink_ins, anchor=anchor):
                        o = sl["origin"]
                        # Every kind's complete provenance participates, including
                        # nested region/cut facts; display names live outside origin.
                        key = (seed_label, sink_addr, json.dumps(o, sort_keys=True))
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
                    if role != "value":
                        self._bw_pointer_demand(
                            func, ssaf, sink_ins, sidx,
                            operand if operand is not None else
                            (seed_var.expr if isinstance(seed_var, _BufferSeed) else seed_var),
                            recovered=operand is None, sink=sink, role=role)
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
                sink_status.append({**desc, "status": "truncated", "seeded": True, "truncated": True,
                                    "slices": len(slices) - n_before,
                                    "note": "recursion limit reached while slicing; results incomplete"})
                continue
            sink_status.append({**desc, "status": "seeded", "seeded": True,
                                "slices": len(slices) - n_before})

        if selections and len(errors) == len(selections):
            # Every sink failed to seed -> hard error (no partial results to keep).
            if len(errors) == 1 and "address" not in errors[0][0]:
                raise TaintError(errors[0][1])
            # One line per (locator, explanation), naming its sites: a hundred
            # sites refused for one reason are one explanation, not a hundred.
            # The text names at most MAX_CALLERS sites per line and counts the
            # rest; every site row rides on the refusal's `observed`.
            groups: dict[tuple[str, str], list[str | None]] = {}
            for s, m in errors:
                groups.setdefault((format_locator(s), m), []).append(s.get("address"))
            lines = []
            for (loc, m), addresses in list(groups.items())[:MAX_CALLERS]:
                named = [a for a in addresses if a is not None]
                where = ""
                if named:
                    where = " @ " + ", ".join(named[:MAX_CALLERS])
                    if len(named) > MAX_CALLERS:
                        where += f" (+{len(named) - MAX_CALLERS} more site(s))"
                lines.append(f"{loc}{where} [unseeded]: {m}")
            if len(groups) > MAX_CALLERS:
                rest = list(groups.values())[MAX_CALLERS:]
                lines.append(f"(+{len(rest)} more refusal(s) covering "
                             f"{sum(len(a) for a in rest)} site(s))")
            refusal = TaintError(
                f"no backward seed resolved for any sink ({len(errors)} site(s)):\n  "
                + "\n  ".join(lines))
            refusal.observed = {"sink_status": sink_status}
            raise refusal
        # #931 round 7: a multi-sink query is complete only when every selected
        # operand is. A sink that could not be seeded at all (no call to it
        # here, an index out of range) is not part of the answer, so the
        # answer is not complete: a blocking leaf per refused sink.
        for s, msg in errors:
            leaf = {"kind": "sink_unseeded", "locator": format_locator(s),
                    "function": hex(int(func.start)),
                    **({"address": s["address"]} if "address" in s else {}),
                    "detail": (f"the sink {format_locator(s)} was not sliced ({msg}), so "
                               f"this multi-sink result does not account for it and is "
                               f"NOT complete")}
            if leaf not in self._bw_leaves:
                self._bw_leaves.append(leaf)

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
                sinks_constant=sum(1 for s in sink_status if s["status"] == "constant"),
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
                        visited_funcs: set, seed_ins: Any = None,
                        anchor: Any = None) -> list[dict[str, Any]]:
        """Backward def-chain slice within *func*, continuing into callers when it
        bottoms out at a parameter. Returns one slice per origin path. *anchor*
        is the consumer's demand on what the seed points to (``walk``): set for
        an ``arg:`` sink, and for the caller ascent of a parameter the callee
        read through; None when no consumer reads through the seed."""
        ssaf = self._ssa_func(func)
        steps: list[dict[str, Any]] = []
        visited_vars: set = set()
        fn_ref = hex(int(getattr(func, "start", 0) or 0))
        origin = {"kind": "unresolved", "function": fn_ref, "var": var_label(seed_var)}
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
        # origin per path, and among the buffer writers the parameter wins
        # (`read(fd, buf); buf[i] ^= key[i]`: the source stays in the steps).
        # Whether a formal's callers are followed never depends on that
        # representative (IC01): every formal the walk bottoms out at is
        # collected in `terminal_params` and ascends on its own, and a
        # non-formal representative (a modeled source's return beside a
        # formal, `malloc(n + recv(..))`) keeps its own slice beside the
        # ascent. Ranks: 0 parameter, 1 any
        # other strong origin (source, frontier), 2 literal fill, 3 constant.
        origin_rank = 0
        terminal_params: dict[int, Any] = {}
        # Each formal's own origin row (a spill-canonicalised one keeps its
        # `via_spill`), so the slice a formal ends is the formal's, not the
        # representative's.
        param_origins: dict[int, dict[str, Any]] = {}
        # The demand delta a consumer placed on each parameter it read through
        # (None: at an unknown offset), so the caller ascent reads the
        # caller's object where the callee did (round 1).
        param_demands: dict[int, int | None] = {}
        # Def-chain values / loads the engine's step budget (`self.max_depth`)
        # cut on some path, keyed as `visited_vars` / `visited_loads` key them.
        # One some other path walked is no cut; each that stays one is an
        # origin this slice did not reach -- a structured `unresolved` slice,
        # not just the assumption prose (IC01).
        budget_cuts: dict[tuple, str] = {}
        # The shallowest depth each value / load was walked from, and the
        # values walked again from a shallower path after a cut (see `walk`).
        walked_at: dict[tuple, int] = {}
        load_at: dict[tuple, int] = {}
        rewalked: set = set()
        # Every non-formal origin the walk reached, in order. A recursive back
        # edge contributes no formal origin (the enclosing frame answers it),
        # so these producers are all it can contribute -- kept, not dropped.
        producers: list[dict[str, Any]] = []
        # The calls whose RETURN the def chain combines into the value (an
        # unmodeled `getpid()`, a modeled source's return): each is a root of
        # its own whatever else represents the slice, so a formal beside it
        # -- or how many callers answer that formal -- never trades it away.
        returned: list[dict[str, Any]] = []
        # The (function, formal) demands each back edge closed on, for the
        # whole ascent from one sink seed: `_continue_into_callers` reads
        # which demand a cycle answered (its own, or one further up). And
        # every caller site outside the functions being solved that supplied
        # a value -- an entry into whatever cycle the ascent is in.
        if depth == 0:
            self._bw_cycle_hits: list[tuple] = []
            self._bw_entries: list[tuple] = []

        def _set_origin(o, *, weak: int | bool = 0):
            nonlocal origin, origin_rank
            o["function"] = fn_ref
            if weak is True:
                rank = 3
            elif weak:
                rank = int(weak) + 1
            else:
                rank = 0 if o.get("kind") == "parameter" else 1
            if rank == 0:
                param_origins.setdefault(int(o["index"]), o)
            elif o not in producers:
                producers.append(o)
            if rank == 1 and o.get("kind") == "call" and o not in returned:
                returned.append(o)
            if origin["kind"] == "unresolved" or origin_rank > rank:
                origin = o
                origin_rank = rank

        def _set_source_origin(o):
            """A modeled source's return IS the value: it represents the slice
            over anything else, a formal found before or after it included.
            Pinned at the top rank, so a later weak origin cannot displace it
            (the outright assignment left a stale rank that could). A formal
            beside it still ascends on its own (IC01)."""
            nonlocal origin, origin_rank
            o["function"] = fn_ref
            if o not in producers:
                producers.append(o)
            if o not in returned:
                returned.append(o)
            origin, origin_rank = o, 0

        # (buffer key, memory version) pairs already walked, so two buffers that
        # copy into each other inside a loop cannot recurse without end.
        visited_bufs: set = set()
        # (instruction, load) pairs already followed: a store's value may
        # reload the slot it writes (`[p] = [p] + 1` around a loop).
        visited_loads: set = set()

        def _forward_cmd(label: str) -> str:
            """The forward run on a local buffer of THIS function, complete
            as printed (#1033 item 6): `-f` and the quoted `var:` source."""
            return f"`bn taint forward -f {fn_ref} --source {shlex.quote(f'var:{label}')}`"

        def _buffer_leaf(at, label, detail, *, kind="address_of_local_unresolved", **extra):
            leaf = {
                "kind": kind,
                "address": hex(int(getattr(at, "address", 0))),
                ("pointer" if kind == _POINTER_LEAF else "buffer"): label,
                "function": fn_ref,
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
                f"unaccounted for and the slice is NOT complete -- list the code "
                f"that references it with `bn xrefs {hex(addr)}`",
                **{"global": hex(addr)})
            _set_origin({"kind": "global", "address": hex(addr), "buffer": label})

        def _follow_addresses(at, expr, d, mv, label, subject, anchor=None) -> tuple[bool, bool]:
            """Walk every buffer *expr* is an address INTO, at any offset:
            each local it is built on (``&line + rbx``) has its writers walked
            at *mv*, each writable global (``0x4040a0 + (i << 6)``) is a leaf.
            Returns ``(walked_local, global_leaf)``. The offset's own reads
            are the caller's to walk -- both feed the bytes (round 4 P1: the
            walk followed only the index and claimed a complete slice).
            *anchor* is the consumer's demand (``_pointer_region``); without
            one the whole local is consumed."""
            locals_, addrs = self._address_terms(expr)
            for loc in locals_:
                _walk_local_buffer(at, loc, d, mv,
                                   _pointer_region(anchor, loc) if anchor is not None else None)
            glob = False
            for addr in addrs:
                if self._const_ptr_storage(addr) == "writable":
                    _global_leaf(at, label, addr, subject)
                    glob = True
            return bool(locals_), glob

        def _follow_input(at, expr, d, mv, label, what, anchor=_AS_CONSUMER,
                          noun="local buffer") -> str:
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
            CONST_PTR; only an address into READ-ONLY storage is a literal.

            *anchor* is the demand on what the operand points at: by default
            the writer consumes the operand itself as a pointer (``(expr,
            0)``, a fill's source); a stored value passes its own demand."""
            if anchor is _AS_CONSUMER:
                anchor = (expr, 0)
            reads = expr_reads(expr)
            walked, glob = _follow_addresses(at, expr, d + 1, mv, label,
                                             f"{what} {noun} {label}", anchor)
            if reads:
                # A loaded operand (`buf[i] = sx.d([p].b)`) is the loaded
                # bytes, not the pointer it was loaded through.
                _walk_expr(at, expr, d, mv, reads, anchor)
            if reads or walked:
                return "followed"
            if glob:
                return "unresolved"
            if "CONST" in op_name(expr) or self._address_terms(expr)[1]:
                # A scalar constant, or a constant address into read-only
                # storage (at a constant offset).
                return "literal"
            leaf_kind = _POINTER_LEAF if noun == _POINTER_NOUN else "address_of_local_unresolved"
            _buffer_leaf(
                at, label,
                f"{what} {noun} {label} is `{expr}`, which reads no variable "
                f"and is neither a literal nor a local the walk can follow; the "
                f"bytes it supplies are unaccounted for and the slice is NOT "
                f"complete", kind=leaf_kind)
            _set_origin({"kind": leaf_kind,
                         ("pointer" if leaf_kind == _POINTER_LEAF else "buffer"): label})
            return "unresolved"

        def _pointer_region(anchor, local):
            """The region of *local* a consumer reads through the pointer its
            demand *anchor* names (#933): ``(expr, delta)`` reads from the
            must-offset of ``expr`` into *local* plus ``delta`` to the end of
            the object. The demand is the CONSUMER's -- the sink's operand, a
            fill's input -- carried back through copies, arithmetic and phis
            by ``_must_offset``; it is never rebuilt from the definition the
            walk happens to reach (`p = b + 8; popen(p - 8)` reads from 0, round
            1). An offset that is not one constant on every path (a moving
            cursor, disagreeing phi arms, a delta lost through memory) is
            unknown (``lo`` None), and a known one is kept as it is -- even
            before the object or past its end, never clamped (round 2)."""
            expr, delta = anchor
            off = None
            if expr is not None and delta is not None:
                o = self._must_offset(ssaf, expr, var_key(local))
                off = None if o is None else o + delta
            return (off, None, True)

        def _read_region(expr):
            """The bytes a by-name read of an address-taken local reads:
            ``buf:4.d`` is ``[4, 8)``; a whole-variable read is the whole
            object, exactly."""
            if op_name(expr) == "MLIL_VAR_ALIASED":
                return (0, None, False)
            if op_name(expr) != "MLIL_VAR_ALIASED_FIELD":
                return None
            try:
                off, w = int(getattr(expr, "offset", 0) or 0), int(getattr(expr, "size", 0) or 0)
            except (TypeError, ValueError):
                return None
            return (off, off + w, False) if w > 0 else None

        def _walk_writers(res, label, d, *, noun="local buffer",
                          kind="address_of_local_unresolved", value_demand=None):
            """Put every writer row of a ``_local_buffer_writers`` result in the
            slice and walk its inputs: a store's value (under *value_demand*:
            the stored bytes are data, unless the consumer dereferences what
            it read back), a fill's inputs and length operands. Returns the
            opaque (may-write) callees and stores, already disclosed as
            *kind* leaves."""
            opaque: list[dict[str, str]] = []
            key = "pointer" if kind == _POINTER_LEAF else "buffer"
            for w in res["writers"]:
                wkind, wdefn = w[0], w[1]
                # The writer's own input memory: where a store's value or a
                # fill's source buffer was read. Using the SINK's version here
                # let a later overwrite of a source buffer (after it had been
                # copied) pose as the copy's origin (PR-928 review P2).
                wmv = self._mem_in(wdefn)
                if wkind == "store":
                    steps.append(_instr_dict(wdefn, reason=f"store into {noun} {label}"))
                    src = getattr(wdefn, "src", None)
                    if _follow_input(wdefn, src, d, wmv, label, "the value stored into",
                                     value_demand, noun) == "literal":
                        cval = self._int_const(src)
                        _set_origin({"kind": "constant", "var": label,
                                     **({"value": cval} if cval is not None else {})},
                                    weak=2)
                elif wkind == "maybe_store":
                    # A store through a pointer the walk cannot pin: it MAY
                    # write the bytes. Walked (its value is in the slice) AND
                    # disclosed -- never dropped.
                    _, _, why = w
                    steps.append(_instr_dict(
                        wdefn, reason=f"store that may write {noun} {label}"))
                    _follow_input(wdefn, getattr(wdefn, "src", None), d, wmv, label,
                                  "the value stored through a pointer that may alias",
                                  value_demand, noun)
                    _buffer_leaf(
                        wdefn, label,
                        f"`{wdefn}` {why} {label}; whether it wrote the bytes the sink "
                        "reads is unknown, so the slice is NOT complete -- inspect "
                        "the pointer's definition or seed inside the producing loop",
                        kind=kind)
                    opaque.append({"address": hex(int(wdefn.address)),
                                   "kind": wkind, "callee": str(wdefn)})
                elif wkind == "escape":
                    steps.append(_instr_dict(
                        wdefn, reason=f"address of {noun} {label} stored to memory"))
                elif wkind == "benign":
                    _, _, callee = w
                    steps.append(_instr_dict(
                        wdefn, reason=f"{noun} {label} passed to {self._prose_name(callee)} "
                                      f"(benign model: writes nothing tainted)"))
                    # The completeness claim now RESTS on a `{}` declaration
                    # for a callee that was handed the buffer -- say so where
                    # a reader of the result looks (round 4 P3).
                    self._bw_assume(
                        f"{noun} {label} is passed to {self._prose_name(callee)} @ "
                        f"{hex(int(getattr(wdefn, 'address', 0)))}, whose model is {{}} "
                        f"(benign: writes nothing tainted); the slice trusts that "
                        f"declaration -- if {self._prose_name(callee)} copies an input into the buffer, "
                        f"model it with `propagates` or the slice is incomplete")
                elif wkind == "fill":
                    _, _, mkey, params, from_idxs, is_source, len_idxs, _raw = w
                    steps.append(_instr_dict(
                        wdefn, reason=f"{noun} {label} written by {self._prose_name(mkey)}"))
                    if is_source:
                        _set_origin({"kind": "source", "callee": mkey,
                                     "address": hex(int(wdefn.address))})
                    states = [_follow_input(wdefn, params[i], d, wmv, label,
                                            f"input {i} of {self._prose_name(mkey)} writing",
                                            noun=noun)
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
                        _set_origin({"kind": "call", "callee": mkey,
                                     "address": hex(int(wdefn.address))}, weak=1)
                else:  # opaque: a call may have written the bytes
                    _, _, callee, why = w
                    steps.append(_instr_dict(
                        wdefn, reason=f"{noun} {label} may be written by {self._prose_name(callee)}"))
                    _buffer_leaf(
                        wdefn, label,
                        f"{self._prose_name(callee)} {why} {label}; whether it wrote the bytes the sink "
                        "reads is unknown, so the slice is NOT complete -- model the "
                        "call or seed inside it",
                        kind=kind, callee=callee)
                    opaque.append({"address": hex(int(wdefn.address)),
                                   "kind": wkind, "callee": callee})
            if opaque:
                # Deferred past the loop so a real origin reached through another
                # writer (a modeled fill from a parameter) wins the slice's
                # origin and the opaque call stays what it is: a frontier leaf.
                opaque.sort(key=lambda w: (w["address"], w["kind"], w["callee"]))
                _set_origin({"kind": kind, key: label, "callee": opaque[0]["callee"],
                             "writers": opaque})
            if res["truncated"]:
                # Discovery or fixed-point proof work was cut. Known writer
                # rows remain visible, but missing writers or an unfinished
                # solve prevent a complete-slice claim.
                self._bw_truncated = True
                self._bw_truncation_causes.add("memory_chain")
            return opaque

        def _unproven_calls(res):
            """The fill rows that are real calls (not a verified store run)."""
            proven = res.get("proven_fills") or set()
            return [w for w in res["writers"] if w[0] == "fill"
                    and int(getattr(w[1], "address", 0) or 0) not in proven]

        def _callsites_check(calls, fallback):
            """The by-hand check for call-filled bytes: runnable `bn callsites`
            commands (a `__builtin_*` has no symbol to list), else *fallback*."""
            cmds = [f"`bn callsites {shlex.quote(raw)} --within {fn_ref}`"
                    for raw in dict.fromkeys(w[7] for w in calls)
                    if isinstance(raw, str) and raw and not raw.startswith("__builtin_")]
            if not cmds:
                return fallback
            return (f"check by hand that each call's destination and length cover "
                    f"them on every path ({', '.join(cmds)} prints the call with its "
                    f"arguments and instructions)")

        call_why = ("but no call's write extent is taken as proof that it "
                    "initialised those bytes (#931: a requested length is not "
                    "what a read delivered, an append does not initialise what "
                    "precedes it, a formatter can fail)")

        def _walk_local_buffer(defn, local, d, mv, region=None, value_demand=None):
            """Continue the slice through the CONTENTS of *local*, read at
            memory version *mv* by the instruction that consumed its address
            (*defn* is the address-of definition, or the consuming call for a
            folded `popen(&cmd)` seed / a nested `strcpy(&cmd, &other)`).
            *region* is ``(lo, hi, via_pointer)``: the bytes consumed, ``hi``
            None for "to the end of the object"; None is the whole object.
            *value_demand* is passed to the stored values (``_walk_writers``)."""
            label = var_label(local)
            if mv is None:
                _buffer_leaf(
                    defn, label,
                    f"the sink reads the contents of local buffer {label} but the "
                    f"walk could not account for them (no memory version at the "
                    f"sink to walk); the slice stops at the address-of and is NOT "
                    f"complete -- re-run {_forward_cmd(label)} "
                    f"or inspect the writes to the buffer")
                _set_origin({"kind": "address_of_local_unresolved", "buffer": label})
                return
            size = self._var_byte_size(local)
            end = size if size is not None and size > 0 else _REGION_OPEN
            # The demand: where the consumer reads, as the consumer computed it.
            # A start or end outside the object is kept, never clamped (round 2).
            d_lo = region[0] if region is not None else None
            d_hi = (region[1] if region is not None and region[1] is not None
                    else (end if d_lo is not None else None))
            outside = d_lo is not None and (d_lo < 0 or d_lo >= end
                                            or (d_hi is not None and d_hi > end))
            # The store facts are solved over the demand when it lies in the
            # object, else over the whole object (they are reported, but they
            # prove nothing about bytes outside it or at an unknown offset).
            lo, hi = (d_lo, d_hi) if d_lo is not None and not outside else (0, end)
            bkey = (var_key(local), int(mv), d_lo, d_hi, value_demand is not None)
            if bkey in visited_bufs:
                return
            visited_bufs.add(bkey)
            res = self._local_buffer_writers(ssaf, var_key(local), mv,
                                             base_size=size, region=(lo, hi))
            _walk_writers(res, label, d, value_demand=value_demand)
            facts = {
                "region": {"offset": d_lo,
                           "size": (d_hi - d_lo) if d_lo is not None and d_hi is not None
                           and d_hi < _REGION_OPEN else None},
                "unproven": None if res["unproven"] is None else
                [[a, b if b < _REGION_OPEN else None] for a, b in res["unproven"]],
            }
            if res["truncated"]:
                _buffer_leaf(
                    defn, label,
                    f"the memory-SSA proof for local buffer {label} exhausted "
                    f"the shared allowance ({MEM_CHAIN_BUDGET} work units across "
                    "graph discovery and fixed-point sweeps); earlier writers may "
                    "be missing or the coverage/escape solve may be unfinished, "
                    "so the slice is NOT complete -- seed closer to the sink or "
                    "inspect the writes and unresolved calls")
                _set_origin({"kind": "address_of_local_unresolved", "buffer": label})
            elif outside:
                where = (f"{-d_lo:#x} bytes before it" if d_lo < 0
                         else f"at offset {d_lo:#x}, past its {end:#x}-byte end"
                         if d_lo >= end else
                         f"at [{d_lo:#x}, {d_hi:#x}), past its {end:#x}-byte end")
                _buffer_leaf(
                    defn, label,
                    f"the consumer reads through a pointer into local buffer {label} "
                    f"{where}: those bytes belong to no object the walk knows, and no "
                    f"store proves them, so the slice is NOT complete -- inspect the "
                    f"pointer arithmetic with `bn il {fn_ref} --view mlil`",
                    kind=_EXTENT_LEAF, cause="outside_object", **facts)
                _set_origin({"kind": _EXTENT_LEAF, "buffer": label,
                             "cause": "outside_object", **facts})
            elif res["covered"] and d_lo is None:
                _buffer_leaf(
                    defn, label,
                    f"stores the walk can see write every byte of local buffer {label}, "
                    f"but the consumer's offset into it is not one constant on every "
                    f"path (a moving cursor, phi arms that disagree, or a pointer "
                    f"reloaded from memory), so it may read outside {label}; those "
                    f"stores are facts, not a proof, so the slice is NOT complete -- "
                    f"inspect the pointer arithmetic with `bn il {fn_ref} --view mlil`",
                    kind=_EXTENT_LEAF, cause="offset_unknown", **facts)
                _set_origin({"kind": _EXTENT_LEAF, "buffer": label,
                             "cause": "offset_unknown", **facts})
            elif res["covered"]:
                # Stores the walk can see write every consumed byte on every
                # path. That is reported as a fact and never certifies the
                # slice (#931/#933 round 3): the object's extent rests on BN's
                # variable recovery, which a name-only rename marks
                # user-defined, and three review rounds forged a store proof.
                span = f"[{lo:#x}, {hi:#x})" if hi < _REGION_OPEN else f"[{lo:#x}, end)"
                _buffer_leaf(
                    defn, label,
                    f"stores the walk can see write every byte of {span} of local "
                    f"buffer {label} on every path, but a region proof from stores is "
                    f"not certified: it rests on BN's recovery of the object's extent "
                    f"and of the consumer's offset, and covered bytes do not establish "
                    f"a NUL terminator, so the slice is NOT complete -- "
                    f"check by hand that the consumer reads only those bytes and that "
                    f"{label} is the whole object (`bn il {fn_ref} --view mlil` shows "
                    f"the stores and the pointer arithmetic)",
                    kind=_PROOF_LEAF,
                    stores_cover=[lo, hi if hi < _REGION_OPEN else None], **facts)
                # The origin stays what the writers established (a stored
                # constant, a parameter): the stores are facts, the leaf blocks.
            else:
                r_lo, r_hi = res["region"]
                span = (f"bytes [{r_lo:#x}, {r_hi:#x})" if r_hi < _REGION_OPEN
                        else f"bytes from {r_lo:#x} to its end (its width is unknown)")
                gaps = res["unproven"] or []
                gap_txt = ", ".join(
                    f"[{a:#x}, {b:#x})" if b < _REGION_OPEN else f"[{a:#x}, end)"
                    for a, b in gaps[:4]) + (", ..." if len(gaps) > 4 else "")
                proven = res.get("proven_fills") or set()
                calls = _unproven_calls(res)
                benign = [w[2] for w in res["writers"] if w[0] == "benign"]
                stores = [w for w in res["writers"] if w[0] in ("store", "maybe_store")
                          or (w[0] == "fill" and int(getattr(w[1], "address", 0) or 0) in proven)]
                if benign and len(benign) == len(res["writers"]):
                    why = (f"the only call that touches it is {', '.join(dict.fromkeys(benign))}, "
                           "modeled {} -- which promises no TAINTED write, not that "
                           "the call fills the buffer")
                elif calls or stores:
                    parts = []
                    if calls:
                        names = list(dict.fromkeys(self._prose_name(w[2]) for w in calls))
                        parts.append(
                            f"{', '.join(names)} write{'s' if len(names) == 1 else ''} "
                            f"into it, {call_why}")
                    if stores:
                        parts.append(
                            "the stores that reach them are a conditional or loop-only "
                            "fill, or only partial, indexed, offset or may-alias writes")
                    why = "; ".join(parts)
                elif res["writers"]:
                    why = ("its other writers (an escape, a may-alias call) write no "
                           "byte the walk can name")
                else:
                    why = "no writer of the buffer was found on the sink's memory chain"
                check = _callsites_check(
                    calls, f"re-run {_forward_cmd(label)} or inspect the writes to the buffer")
                if calls and check.startswith("check by hand"):
                    check += f", or re-run {_forward_cmd(label)}"
                _buffer_leaf(
                    defn, label,
                    f"the sink reads {span} of local buffer {label}, but on at least "
                    f"one path from the function entry no plain store the walk can see "
                    f"writes {gap_txt}; {why}, so there the sink may read whatever the "
                    f"stack held and the slice is NOT complete -- {check}",
                    region={"offset": r_lo,
                            "size": (r_hi - r_lo) if r_hi < _REGION_OPEN else None},
                    unproven=[[a, b if b < _REGION_OPEN else None] for a, b in gaps],
                    **({"calls": list(dict.fromkeys(w[7] for w in calls))} if calls else {}))
                # Classified like `field_load_unresolved`: the origin names the
                # stop and the leaf is the one frontier entry, so the frontier
                # count is not inflated by counting one stop twice.
                _set_origin({"kind": "address_of_local_unresolved", "buffer": label})

        def _walk_pointer_object(at, root, d, mv, how, *, before):
            """The consumer reads what pointer *root* points to, and no local is
            behind it -- *root* is a call's return (an allocator, an unmodeled
            helper) or a parameter (round 1). Its writers in this function are
            walked like a local's, so a `strcpy(b, host)` and its `host` stay in
            the slice. Nothing proves the bytes: the walk knows neither the
            object's extent nor what it held *before* this function wrote it.
            Always a ``pointer_object_unresolved`` leaf -- never complete."""
            label = var_label(root)
            bkey = (_ssa_root_key(root), mv)
            if bkey in visited_bufs:
                return
            visited_bufs.add(bkey)
            calls: list = []
            if mv is not None:
                res = self._local_buffer_writers(ssaf, _ssa_root_key(root), mv)
                _walk_writers(res, label, d, noun=_POINTER_NOUN, kind=_POINTER_LEAF)
                calls = _unproven_calls(res)
            call_txt = ""
            if calls:
                names = list(dict.fromkeys(self._prose_name(w[2]) for w in calls))
                call_txt = (f"; {', '.join(names)} write{'s' if len(names) == 1 else ''} "
                            f"into it, {call_why}")
            check = _callsites_check(
                calls, "inspect the code that allocates and fills the object, or seed "
                       "where it is filled")
            _buffer_leaf(
                at, label,
                f"the slice reads through {label} into the object it points to ({how}), "
                f"which is no local of this function: the walk knows neither its extent "
                f"nor what it held before {before}, and no store it can see proves the "
                f"bytes read{call_txt}, so the slice is NOT complete -- {check}",
                kind=_POINTER_LEAF,
                **({"calls": list(dict.fromkeys(w[7] for w in calls))} if calls else {}))

        def _arg_locals(params):
            """Every local a call argument is an address into, at any offset."""
            out: list = []
            for p in params:
                locs = list(self._address_terms(p)[0])
                if not locs and self._addr_root_kind(ssaf, p) == "local":
                    root = self._addr_base_var(ssaf, p)
                    locs = [root] if root is not None else []
                for loc in locs:
                    if all(var_key(loc) != var_key(o) for o in out):
                        out.append(loc)
            return out

        def _demand_through(anchor, v):
            """The demand on what SSA value *v* points to, given the demand
            *anchor* on a value built from it: the anchor's must-offset from
            *v* plus its delta, or an unknown offset (``(None, None)``: the
            whole object) when that is not one constant. None stays None."""
            if anchor is None:
                return None
            expr, delta = anchor
            if expr is None or delta is None:
                return (None, None)
            off = self._must_offset(ssaf, expr, _ssa_root_key(v))
            return (None, None) if off is None else (v, off + delta)

        def _returned_object(defn, v, d, mv, anchor, name, model, *, fresh):
            """The consumer reads what call *defn* returned (``v``). The return
            may be any object the call was handed to write or derive it from
            (`p = fgets(c, ..)`): the model's written ``*arg:N`` and returned
            arguments, or every argument of an unmodeled callee. A local one
            is checked as a whole -- where in it the return points is unknown;
            any other is walked under the same demand, so a heap allocation or
            a pointer parameter becomes a ``pointer_object_unresolved``
            frontier exactly as when the consumer reads it directly (round 2).
            A return not proven to be an argument (*fresh*: an allocator, an
            unmodeled helper) is a pointer object of its own. Only for a
            consumer that reads through the value (*anchor*) and a value BN
            types as a pointer (round 1)."""
            if anchor is None or not _is_pointer_typed(v):
                return
            params = self._call_params(defn)
            if model is None:
                idxs = list(range(len(params)))
            else:
                idxs = []
                for rule in (model.get("sources") or []) + (model.get("propagates") or []):
                    to, frm = str(rule.get("to") or ""), str(rule.get("from") or "")
                    cand = (_try_arg_index(to) if to.startswith("*arg:")
                            else _try_arg_index(frm) if to == "ret" else None)
                    if cand is not None and cand < len(params) and cand not in idxs:
                        idxs.append(cand)
            for i in idxs:
                p = params[i]
                blocking_before = sum(1 for lf in self._bw_leaves
                                      if lf.get("kind") in BLOCKING_LEAF_KINDS)
                locs = _arg_locals([p])
                for loc in locs:
                    _walk_local_buffer(defn, loc, d, mv, (None, None, True))
                # A constant address of writable storage (`fgets(g_buf, ..)`,
                # `fgets(g_req.cmd, ..)`) has no SSA reads to follow: it is the
                # same `global` frontier as consuming that global directly
                # (round 3).
                for addr in self._address_terms(p)[1]:
                    if self._const_ptr_storage(addr) == "writable":
                        _global_leaf(defn, var_label(v), addr,
                                     f"the pointer {var_label(v)} returned by "
                                     f"{self._prose_name(name)}")
                if not locs:
                    for r in expr_reads(p):
                        walk(r, d + 1, mv, (p, None))
                # Default deny (re-dogfood of round 3): the object the call was
                # handed must end in a disclosed frontier -- a local's leaf, a
                # `global` leaf, a pointer object's leaf. An object the walk
                # resolved to none of those (a global reached through the GOT,
                # `rdi = global_buf` as MLIL_IMPORT; a constant in read-only
                # storage; any other expression) is a frontier of its own.
                if sum(1 for lf in self._bw_leaves
                       if lf.get("kind") in BLOCKING_LEAF_KINDS) == blocking_before:
                    what, slot = str(p), None
                    e = p
                    for _ in range(8):
                        rd = expr_reads(e)
                        if op_name(e) == "MLIL_IMPORT" or len(rd) != 1 or op_name(e) not in _COPY_OPS:
                            break
                        try:
                            dd = ssaf.get_ssa_var_definition(rd[0])
                        except Exception:
                            dd = None
                        if dd is None or op_name(dd) != "MLIL_SET_VAR_SSA":
                            break
                        e = getattr(dd, "src", None)
                    if op_name(e) == "MLIL_IMPORT":
                        c = getattr(e, "constant", None)
                        what, slot = str(e), (int(c) if isinstance(c, int) else None)
                    via = (f" (its address is loaded from the GOT slot {slot:#x})"
                           if slot is not None else "")
                    _buffer_leaf(
                        defn, what,
                        f"the pointer {var_label(v)} returned by {self._prose_name(name)} "
                        f"may point into `{what}`{via}, an object the walk cannot see as "
                        f"a local buffer; its bytes are not accounted for, so the slice "
                        f"is NOT complete -- inspect it with `bn il {fn_ref} --view mlil`",
                        kind=_POINTER_LEAF)
            if fresh:
                _walk_pointer_object(defn, v, d, mv, f"returned by {self._prose_name(name)}",
                                     before="it was written here")

        def walk(v, d, mv, anchor=None):
            """Backward over the def-chain of SSA value *v*. *mv* is the memory
            version at which the value is CONSUMED as a buffer (the sink call,
            or the modeled fill / store that read it) -- it rides unchanged
            through copies and arithmetic, because a pointer's pointee is read
            where the pointer is dereferenced, not where it was defined.
            *anchor* is the consumer's demand on what the value points to:
            ``(expr, delta)`` reads from ``expr``'s target plus ``delta``
            (``expr`` None or ``delta`` None: at an unknown offset), or None
            when no consumer reads through the value (a ``var:`` sink, an
            index). It rides along unchanged, like *mv*, so the region is
            always the consumer's, never the definition's (round 1)."""
            nonlocal origin
            ckey = (var_key(v), getattr(v, "version", None),
                    None if anchor is None else (str(anchor[0]), anchor[1]))
            if d > self.max_depth:
                # Over budget on THIS path only: a value another path walks
                # is no cut -- its chain is followed there. Settled when the
                # slice is built.
                if ckey not in visited_vars:
                    budget_cuts[("var", ckey)] = var_label(v)
                return
            # A value first walked deeper, with the budget cutting something,
            # is walked again from this shallower path: what was cut beneath
            # it may be within reach now -- and so may what was cut beneath a
            # load first entered deeper, so those loads are re-enterable too
            # (`_follow_load` re-walks one only from a shallower depth). Only
            # then, and only while shallower, so each value and load is walked
            # at most `max_depth` times.
            seen = walked_at.get(ckey)
            if seen is None or (d < seen and budget_cuts):
                if seen is not None:
                    visited_vars.discard(ckey)
                    rewalked.add(ckey)
                    for lk in [lk for lk, ld in load_at.items() if ld > d]:
                        visited_loads.discard(lk)
                walked_at[ckey] = d
            key = (var_key(v), getattr(v, "version", None),
                   None if anchor is None else (str(anchor[0]), anchor[1]))
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
                    if anchor is not None:
                        # The ascent reads the caller's object at the same
                        # demand, relative to the argument it passed.
                        dem = _demand_through(anchor, v)
                        param_demands[pidx] = None if dem is None else dem[1]
                        if _is_pointer_typed(v):
                            # A pointer parameter's object is no local here:
                            # its writers in this function are walked, and
                            # nothing proves its bytes (round 1).
                            _walk_pointer_object(seed_ins if seed_ins is not None else v, v,
                                                 d, mv, f"parameter {pidx}",
                                                 before="this function ran")
                else:
                    _set_origin({"kind": "entry", "var": var_label(v)})
                return
            if self._is_call(defn):
                target = const_target(getattr(defn, "dest", None))
                name = self._callee_name(target)
                mkey, model = lookup_model(self.models, name)
                steps.append(_instr_dict(defn, reason=f"defined by call to {name or 'indirect'}"))
                if model and model.get("sources"):
                    # The representative, ranked (IC01): the outright
                    # assignment left a stale rank a later weak origin could
                    # undercut; a formal beside it still ascends on its own.
                    _set_source_origin({"kind": "source", "callee": mkey or name,
                                        "address": hex(int(defn.address))})
                    # A modeled source's returned bytes are the source -- but
                    # a return into an object it was handed -- a local
                    # (`fgets(c, ..)`), a heap allocation, a pointer parameter
                    # -- reads that object, whose bytes no call proves.
                    _returned_object(defn, v, d, mv, anchor, name, model, fresh=False)
                elif target is None:
                    origin = {"kind": "indirect_call", "var": var_label(v),
                              "function": fn_ref, "address": hex(int(defn.address))}
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
                    rules = [(_try_arg_index(str(r.get("from") or "")),
                              str(r.get("from") or "").startswith("*"))
                             for r in (model or {}).get("propagates") or []
                             if str(r.get("to") or "") == "ret"]
                    cparams = self._call_params(defn) if rules else []
                    states: list[str] = []
                    dem = _demand_through(anchor, v)
                    for fi, pointee in rules:
                        if fi is None or fi >= len(cparams):
                            continue
                        steps.append(_instr_dict(
                            defn, reason=f"return of {mkey or name} derives from its argument {fi}"))
                        # `arg:N -> ret` returns the argument itself, so the
                        # demand carries over; `*arg:N -> ret` returns a
                        # pointer somewhere into it (strchr), at an unknown
                        # offset. Without a consumer, the whole object.
                        arg_dem = (None if dem is None
                                   else (cparams[fi], None if pointee else dem[1]))
                        states.append(_follow_input(
                            defn, cparams[fi], d, mv, var_label(v),
                            f"argument {fi} of {mkey or name} returned by", arg_dem))
                    if not states:
                        _set_origin({"kind": "call", "callee": name,
                                     "address": hex(int(defn.address))})
                        _returned_object(defn, v, d, mv, anchor, name, model, fresh=True)
                    elif all(s == "literal" for s in states):
                        _set_origin({"kind": "call", "callee": mkey or name,
                                     "address": hex(int(defn.address))}, weak=1)
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
            # own reads are walked too -- they pick which bytes. The region
            # is the consumer's demand (*anchor*), not this definition's.
            walked, glob = _follow_addresses(defn, src_expr, d, mv, var_label(v),
                                             f"the pointer {var_label(v)}", anchor)
            if walked or glob:
                _walk_expr(defn, src_expr, d, mv, expr_reads(src_expr), anchor)
                return
            if src_expr is not None and op_name(src_expr) in ("MLIL_VAR_ALIASED",
                                                               "MLIL_VAR_ALIASED_FIELD"):
                # `i = buf[0] @ mem#k`: a read of an address-taken local BY
                # NAME -- the bytes are the buffer's contents at this
                # instruction's memory version, so the writers on that chain
                # are the value's producers (the xor-loop `buf[i] ^= key[i]`
                # reads the `read` fill through exactly this shape; round 3).
                # A consumer that reads through the value read back reads
                # through the stored values, at an unknown offset.
                aliased = getattr(src_expr, "src", None)
                aliased = getattr(aliased, "var", aliased)
                if aliased is not None:
                    _walk_local_buffer(defn, aliased, d, self._mem_in(defn),
                                       _read_region(src_expr),
                                       None if anchor is None else (None, None))
                    return
            if src_expr is not None and self._outer_loads(src_expr):
                # A definition that READS MEMORY -- a bare LOAD, or one under
                # a conversion or arithmetic (`sx.q([p+0xc].d)`, #1041 item 2):
                # the value is the loaded bytes, never the pointer they were
                # loaded through. Walking the address instead named the
                # pointer's allocator as the origin of a complete slice.
                _walk_expr(defn, src_expr, d, mv, ssa_reads(defn), anchor)
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
                src0 = getattr(defn, "src", None)
                cval = self._int_const(src0)
                if cval is not None:
                    _set_origin({"kind": "constant", "value": cval, "var": var_label(v)},
                                weak=True)
                elif anchor is not None and op_name(src0) in ("MLIL_IMPORT", "MLIL_EXTERN_PTR"):
                    # A consumer reads through an address BN resolved through
                    # the GOT or an extern (a preemptible global in a shared
                    # object): an object the walk cannot see as a local buffer,
                    # so a frontier -- whatever origin another path supplies
                    # (re-dogfood of round 3, default deny).
                    c = getattr(src0, "constant", None)
                    via = (f" (its address is loaded from the GOT slot {int(c):#x})"
                           if isinstance(c, int) else "")
                    _buffer_leaf(
                        defn, str(src0),
                        f"the slice reads through {var_label(v)} into `{src0}`{via}, an "
                        f"object the walk cannot see as a local buffer; its bytes are not "
                        f"accounted for, so the slice is NOT complete -- inspect it with "
                        f"`bn il {fn_ref} --view mlil`",
                        kind=_POINTER_LEAF)
            for r in reads:
                walk(r, d + 1, mv, anchor)

        def _walk_expr(at, expr, d, mv, reads, anchor=None):
            """Walk the value expression *expr* of instruction *at*: each
            memory read in it through the writers of the loaded bytes
            (``_follow_load``), and as SSA values only the *reads* OUTSIDE
            those loads' addresses. *anchor* is the consumer's demand on what
            the value points to; through a load it survives only as a demand
            at an unknown offset (the loaded pointer's target)."""
            loads = self._outer_loads(expr)
            rest = list(reads)
            for ld in loads:
                for r in expr_reads(ld):
                    if r in rest:
                        rest.remove(r)
            for ld in loads:
                _follow_load(at, ld, d, None if anchor is None else (None, None))
            for r in rest:
                walk(r, d + 1, mv, anchor)

        def _follow_load(at, load, d, demand=None):
            """Continue the slice through the writers of the bytes *load*
            reads, on every memory path back from it (``_load_writers``). A
            load from a local buffer walks the buffer's writers with their
            coverage proof; a covering store or a modeled source fill is an
            origin; a partial or may-alias store is walked AND, like any path
            that reaches an unmodeled writer or the function entry, leaves a
            ``field_load_unresolved`` frontier -- the slice is not complete.
            *demand* is the consumer's demand on what the LOADED value points
            to (a reloaded pointer the sink reads through); the stored values
            the walk reaches carry it."""
            key = (int(getattr(at, "address", 0) or 0), str(load),
                   getattr(load, "src_memory", None), demand is not None)
            if key in visited_loads:
                return
            visited_loads.add(key)
            if d > self.max_depth:
                # Not walked: unmark it so a shallower path still can, and
                # settle the cut when the slice is built.
                visited_loads.discard(key)
                budget_cuts[("load", key)] = str(load)
                return
            # Re-entered after a shallower re-walk unmarked it (see `walk`):
            # walk it again only from a shallower depth than before.
            if load_at.get(key, d + 1) <= d:
                return
            load_at[key] = d
            addr = getattr(load, "src", None)
            lmv = getattr(load, "src_memory", None)
            # The address's reads (pointer and index); a load whose address
            # operand is unavailable is judged by its own reads.
            addr_reads = expr_reads(addr) if addr is not None else expr_reads(load)
            local_terms, _ = self._address_terms(addr)
            if not local_terms and self._addr_root_kind(ssaf, addr) == "local":
                root = self._addr_base_var(ssaf, addr)
                local_terms = [root] if root is not None else []
            if local_terms:
                # `x = t[i]` / `[&buf + 8]`: the bytes are the buffer's
                # contents at the load; the index's reads pick which bytes.
                ex = self._mem_access_slot(ssaf, load, exact=True)
                for loc in local_terms:
                    reg = None
                    if ex is not None and ex[0] == var_key(loc) and isinstance(ex[1], int):
                        w = getattr(load, "size", None)
                        reg = (ex[1], ex[1] + w, False) if isinstance(w, int) and w > 0 else None
                    _walk_local_buffer(at, loc, d, lmv, reg, demand)
                # A read that is a must-alias of `&local + c` (`p` in `p->b`
                # after `p = &r`) is the address, not data: walking it would
                # walk the whole local again as if the load consumed it all.
                addr_reads = [r for r in addr_reads
                              if not any(self._must_offset(ssaf, r, var_key(loc)) is not None
                                         for loc in local_terms)]
                for r in addr_reads:
                    walk(r, d + 1, lmv)
                return
            # Exact identity only (#1053 review 1): a narrowed, extended or
            # phi-rooted address has no slot any store can be matched to.
            la = self._mem_access_slot(ssaf, load, exact=True)
            fixed = self._gconst_addr(la)
            gaps: list[str] = []
            _, consts = self._address_terms(addr)
            kinds = [self._const_ptr_storage(c) for c in consts]
            if fixed is None and addr_reads and consts:
                # `tbl[i]`: a constant table at a variable index or offset --
                # even when `[(i << 3) + tbl]` keys as an exact slot on the
                # scaled index (#1053 review 4). The index picks the bytes and
                # is walked; a writable table is a `global` frontier. The
                # bytes are never taken as the binary's own, even in a
                # non-writable segment: an unbounded index, a read-only
                # section in a writable segment and loader-written entries
                # (IFUNC/TEXTREL, IRELATIVE) all defeated that claim (#1053
                # reviews 5 and 6).
                for r in addr_reads:
                    walk(r, d + 1, lmv)
                for c, k in zip(consts, kinds):
                    if k == "writable":
                        _global_leaf(at, hex(c), c, "the loaded address")
                if all(k == "writable" for k in kinds):
                    return      # each writable table is already a `global` leaf
                gaps.append("it reads a constant table at a variable index or offset, "
                            "and the walk does not prove which bytes the table held")
            elif la is None:
                # `[host + i]`: as for a pointer operand of a fill, the pointer
                # and the index are walked (they lead to the buffer and pick
                # the bytes) -- but no store can be matched to the bytes. A
                # constant address in read-only storage is no exception: the
                # storage class does not prove the file supplied the bytes
                # (#1053 reviews 5-6).
                for r in addr_reads:
                    walk(r, d + 1, lmv)
                gaps.append("its address is not exactly a base plus a constant offset, "
                            "so no store can be matched to it")
            else:
                for w in self._load_writers(ssaf, load, la):
                    kind = w[0]
                    if kind in ("store", "partial", "may"):
                        store = w[1]
                        reason = {"store": "reaching store to field",
                                  "partial": "store writing part of the loaded bytes",
                                  "may": "store that may write the loaded bytes"}[kind]
                        steps.append(_instr_dict(store, reason=reason))
                        value = getattr(store, "src", None)
                        if expr_reads(value) or self._outer_loads(value):
                            _walk_expr(store, value, d + 1, self._mem_in(store), expr_reads(value),
                                       demand)
                        elif kind == "store":
                            cval = self._int_const(value)
                            if cval is not None:
                                _set_origin({"kind": "constant", "value": cval})
                        if kind == "may":
                            gaps.append(f"`{store}` stores through a pointer the walk "
                                        "cannot prove disjoint from them")
                    elif kind == "may_source":
                        steps.append(_instr_dict(w[1], reason=f"field may be filled by source {w[2]}"))
                        _set_origin({"kind": "source", "callee": w[2],
                                     "address": hex(int(w[1].address))})
                        gaps.append(f"`{w[1]}` may have written them (no call's write "
                                    "extent is trusted)")
                    elif kind == "opaque":
                        gaps.append(f"`{w[1]}` may have written them")
                    elif kind == "entry":
                        # Read-only storage is no exception: `.data.rel.ro`
                        # without RELRO is writable, and the loader may write a
                        # non-writable segment (#1053 reviews 5-6).
                        gaps.append(
                            f"on some path nothing in this function wrote them (the "
                            f"{'global' if fixed is not None else 'object'} "
                            f"holds what was stored before this function ran)")
                    else:
                        gaps.append(f"the memory chain exceeded {MEM_CHAIN_BUDGET} versions")
                if not gaps:
                    return
            base_var = self._addr_base_var(ssaf, addr)
            width = getattr(load, "size", None)
            base_label = (hex(fixed) if fixed is not None
                          else var_label(base_var) if base_var is not None else None)
            offset = (hex(la[1]) if la is not None and fixed is None and isinstance(la[1], int)
                      else "0x0" if fixed is not None else None)
            leaf = {
                "kind": "field_load_unresolved",
                "address": hex(int(getattr(at, "address", 0))),
                "base": base_label,
                "offset": offset,
                "width": (int(width) if isinstance(width, int) else None),
                "il_text": str(at),
                "detail": ("the loaded bytes are not accounted for: "
                           + "; ".join(dict.fromkeys(gaps))
                           + " -- the slice is NOT complete"),
            }
            if leaf not in self._bw_leaves:
                self._bw_leaves.append(leaf)
            _set_origin({"kind": "field_load_unresolved", "base": base_label,
                         "offset": offset, "width": leaf["width"]})

        seed_mv = self._mem_in(seed_ins) if seed_ins is not None else None
        if isinstance(seed_var, _BufferSeed):
            # `popen(&cmd, "r")` with the address folded into the parameter:
            # the seed IS the buffer's contents, read by the sink call itself.
            if seed_var.local is not None:
                _walk_local_buffer(seed_ins, seed_var.local, 0, seed_mv,
                                   _pointer_region(anchor, seed_var.local)
                                   if anchor is not None else None)
            else:
                _global_leaf(seed_ins, seed_var.name, int(seed_var.addr), "the sink argument")
        else:
            walk(seed_var, 0, seed_mv, anchor)
        base_steps = list(reversed(steps))

        if rewalked:
            # A value walked again records its definitions again; keep the
            # first record of each step.
            seen_steps: list[dict[str, Any]] = []
            for st in steps:
                if st not in seen_steps:
                    seen_steps.append(st)
            base_steps = list(reversed(seen_steps))
        # Every formal the slice bottomed out at is a demand on the callers,
        # whatever origin represents the slice (IC01). A demand already being
        # solved further up this ascent -- the same (function, formal) -- is
        # answered there, by the frame enumerating that formal's callers: the
        # formal is a cycle's back edge, not an origin, and stops the ascent.
        # So is a formal whose every resolved caller turned out to be such a
        # back edge further up (mutual recursion: `a(x) -> b(x) -> a(y + ..)`).
        # What the back edge ADDS on its way round -- `f(n + getpid())` -- is
        # not answered upstream, so every producer it reached keeps a slice
        # (below). A recursive call that remaps the demand to ANOTHER formal
        # is a new demand and ascends.
        fkey = self._caller_key(func)
        demands = {p: param_origins.get(p) or {"kind": "parameter", "index": p,
                                                "var": var_label(v), "function": fn_ref}
                   for p, v in terminal_params.items() if (fkey, p) not in visited_funcs}
        out: list[dict[str, Any]] = []
        cycled: set[int] = set()
        if demands:
            out, cycled = self._continue_into_callers(
                func, demands, depth, max_depth,
                visited_funcs | {(fkey, p) for p in demands}, base_steps, param_demands)
        back_edge = bool(cycled) or len(demands) < len(terminal_params)
        self._bw_cycle_hits.extend((fkey, p) for p in terminal_params if p not in demands)
        # A cut stands only if no path walked that value or load.
        cuts = sorted({
            label for (kind, k), label in budget_cuts.items()
            if k not in (walked_at if kind == "var" else load_at)})
        if cuts:
            self._bw_assume(
                f"def-chain walk truncated at {self.max_depth} steps (engine max_depth)")
        # A non-formal representative keeps its own slice beside the formals'
        # ascent and leads, as it did before the formals ascended -- a source
        # next to a formal is never traded for the formal's callers, in either
        # operand order.
        if origin["kind"] != "parameter" and not (cuts and origin["kind"] == "unresolved"):
            out.insert(0, {"steps": base_steps, "origin": origin, "crossed": []})
        # Each independent producer keeps its own slice, the representative
        # aside (already one): on a back edge every producer the walk reached
        # (the cycle answers nothing else), elsewhere every call RETURN the
        # value combines -- so neither a formal beside it nor the callers
        # that answer that formal (one entry or several) trade it away.
        extra = producers if back_edge else returned
        out += [{"steps": base_steps, "origin": o, "crossed": []}
                for o in extra if o is not origin and o["kind"] != "unresolved"]
        if cuts:
            # Replaces a bare `unresolved` origin: the cut is why it is one.
            # `var` names the cut values IN this function: the public slice
            # dedup keys on it, so two callers' distinct cuts stay two rows.
            out.append({"steps": base_steps, "crossed": [], "origin": {
                "kind": "unresolved", "cause": "def_chain_budget",
                "budget": int(self.max_depth), "function": fn_ref, "unwalked": cuts,
                "var": f"{', '.join(cuts)} in {fn_ref}"}})
        return out

    @staticmethod
    def _seed_anchor(seed_var: Any, operand: Any) -> Any:
        """The selected *operand* the seed came from, as a demand's expression:
        a folded buffer seed's own operand, else *operand* when it reads
        *seed_var* (the seed itself when the operand is a plain copy of it).
        None when the operand does not read the seed."""
        if isinstance(seed_var, _BufferSeed):
            return seed_var.expr
        want = (var_key(seed_var), getattr(seed_var, "version", None))
        reads = expr_reads(operand)
        if any((var_key(r), getattr(r, "version", None)) == want for r in reads):
            return seed_var if op_name(operand) in _COPY_OPS and len(reads) == 1 else operand
        return None

    def _bw_pointer_demand(self, func: Any, ssaf: Any, sink_ins: Any, idx: int,
                           pointer: Any, *, recovered: bool, sink: dict[str, Any],
                           role: str) -> None:
        """THE POLICY (#931 rounds 6-7): neither a call nor a store certifies
        a buffer's bytes, so a sink that reads through a pointer is NEVER
        complete, and an argument is a value only when its model says so.
        A modeled pointer argument gets a blocking
        ``pointer_demand_not_certified`` leaf; an argument with no model role
        at all gets a blocking ``sink_role_unknown`` leaf whose detail prints
        the overlay that declares a value role. Both list every object the
        pointer may read, each described on its own. The description
        (``_demand_objects``) is descriptive only: nothing here decides
        completeness but the role (``_sink_role``)."""
        objects = self._demand_objects(func, ssaf, pointer) if pointer is not None else []
        uniq: list[dict[str, Any]] = []
        for o in objects:
            if o not in uniq:
                uniq.append(o)

        def say(o: dict[str, Any]) -> str:
            if o["kind"] == "returned":
                inner = ", ".join(say(i) for i in o["may_point_into"])
                return (f"the pointer {_code(o['object'])} returned"
                        + (f", into {inner}" if inner else "")
                        + (f" ({o['note']})" if o.get("note") else ""))
            extra = []
            if "offset" in o:
                extra.append(f"offset {o['offset']:#x}" if isinstance(o["offset"], int)
                             and o["offset"] >= 0 else
                             f"offset -{-o['offset']:#x}" if isinstance(o["offset"], int)
                             else "offset unknown")
            if o.get("slot"):
                extra.append(f"GOT slot {o['slot']}")
            if o.get("segment_writable") is not None:
                extra.append("in a writable segment" if o["segment_writable"]
                             else "in a read-only segment")
            return (f"{o['kind'].replace('_', ' ')} {_code(o['object'])}"
                    + (f" ({', '.join(extra)})" if extra else ""))

        fn_ref = hex(int(getattr(func, "start", 0) or 0))
        listed = "; ".join(say(o) for o in uniq) or "no object the walk could name"
        shown = var_label(pointer) if is_ssa_var(pointer) else str(pointer)
        how = (f"its argument {idx}, recovered from a register ({_code(shown)})" if recovered
               else f"its argument {idx} ({_code(shown)})")
        leaf = {
            "kind": "pointer_demand_not_certified" if role == "pointer" else "sink_role_unknown",
            "address": hex(int(getattr(sink_ins, "address", 0) or 0)),
            "function": fn_ref,
            "operand": shown,
            "objects": uniq,
            "il_text": str(sink_ins),
            "detail": (f"the sink reads bytes through {how}, and no call or store "
                       f"certifies a buffer's bytes (#931): a sink that reads through a "
                       f"pointer is never complete. It may read: {listed}. Check each "
                       f"object by hand (`bn il {fn_ref} --view mlil`)"),
        }
        if role != "pointer":
            # #931 round 8: guidance, not a generated command. A printed
            # overlay replaced the callee's whole model, a printed re-run
            # dropped the query's other sinks and options, and a redirect
            # overwrote a file: the leaf names the callee and argument, and
            # `declare_value_role` is a TEMPLATE entry (UPPER-CASE
            # placeholders, data only) the user merges into the callee's
            # current model and passes with `--models FILE` to the same
            # command they ran.
            callee = str(sink.get("callee") or "")
            addr = self._callee_as_addr(callee)
            name = (self._callee_name(addr) if addr is not None else None) or callee
            leaf["callee"] = name
            leaf["arg_index"] = idx
            leaf["declare_value_role"] = {
                "template": True,
                "entry": {name: {"sink": {"class": "VALUE_CLASS", "tainted_args": [idx]},
                                 "OTHER_ROLES": "COPY_FROM_THE_CURRENT_MODEL"}},
                "placeholders": {
                    "VALUE_CLASS": sorted(self._VALUE_SINK_CLASSES),
                    "OTHER_ROLES": "every other key of the callee's current model "
                                   "(sources, propagates, ...), copied unchanged",
                },
            }
            leaf["detail"] = (
                f"no sink model gives argument {idx} of {_code(self._prose_name(callee))} "
                f"a role, so the slice treats it as a pointer the sink reads through "
                f"(#931): it is NOT complete. It may read: {listed}. If the argument is "
                f"a value (a size, a count, a flag), declare it in a model overlay: the "
                f"leaf's `declare_value_role.entry` is a TEMPLATE")
        if recovered:
            leaf["recovered_from_register"] = True
        if leaf not in self._bw_leaves:
            self._bw_leaves.append(leaf)

    # The sink classes whose `tainted_args` are VALUES (a size, a length, a
    # count) and those whose `tainted_args` are POINTERS the sink reads
    # through (a command, a path, a format, a copied string). A mixed class
    # (`file_write`, `net_write`: a buffer AND a length) names its pointer
    # arguments in `sink.pointer_args`; its other tainted args are values.
    _VALUE_SINK_CLASSES = frozenset({"alloc_size", "overflow_len", "fortified_overflow"})
    _POINTER_SINK_CLASSES = frozenset({
        "command_injection", "path_injection", "format_string", "format_or_overflow",
        "fortified_format", "overflow_unbounded", "unbounded_input"})

    def _sink_arg_role(self, callee: str, idx: int) -> str | None:
        """The role of argument *idx* of the modeled sink *callee* --
        ``"pointer"`` (the sink reads bytes through it), ``"value"`` (a
        size, a length) or None when the model does not say. The locator's
        callee is the target `_find_callsites` matched, whether directly,
        through a thunk, or pinned by `--resolve-map` / a value set, so the
        role is the pinned target's. In order: `sink.pointer_args`,
        `buf_arg` / `format_arg` (the format string a format-gated sink
        reads, #1041 item 4) / `len_arg`, a `*arg:N` pointee in the model's
        sources, propagations or writes, then the sink class of a tainted
        arg."""
        _, model = lookup_model(self.models, callee)
        if model is None:
            addr = self._callee_as_addr(callee)
            name = self._callee_name(addr) if addr is not None else None
            if name:
                _, model = lookup_model(self.models, name)
        if not isinstance(model, dict):
            return None
        sink = model.get("sink") if isinstance(model.get("sink"), dict) else {}
        explicit = sink.get("pointer_args")
        if isinstance(explicit, list) and idx in explicit:
            return "pointer"
        if sink.get("buf_arg") == idx or sink.get("format_arg") == idx:
            return "pointer"
        if sink.get("len_arg") == idx:
            return "value"
        toks = [str(r.get(k) or "") for key in ("sources", "propagates")
                for r in (model.get(key) or []) if isinstance(r, dict) for k in ("from", "to")]
        toks += [str(t) for t in (model.get("writes") or [])]
        if any(t.startswith("*arg:") and _try_arg_index(t) == idx for t in toks):
            return "pointer"
        if idx in (sink.get("tainted_args") or []):
            if isinstance(explicit, list):
                return "value"
            cls = sink.get("class")
            if cls in self._VALUE_SINK_CLASSES:
                return "value"
            if cls in self._POINTER_SINK_CLASSES:
                return "pointer"
        return None

    def _sink_role(self, sink: dict[str, Any]) -> str:
        """The role of the selected `arg:` sink argument: ``"pointer"`` or
        ``"value"`` as its model says (``_sink_arg_role``), else
        ``"unknown"`` (#931 round 7). An argument is a value ONLY when a
        model -- builtin or a user overlay, an explicit claim either way --
        says so; BN's typing never decides, so an unknown role is walked and
        reported like a pointer demand."""
        role = self._sink_arg_role(str(sink.get("callee") or ""), int(sink.get("index", -1)))
        return role if role is not None else "unknown"

    def _bw_unseeded_demand(self, func: Any, ssaf: Any, instrs: list[Any],
                            sink: dict[str, Any]) -> bool:
        """An `arg:` sink that could not be seeded (`popen("ls")`, a fixed
        address, a constant): when its argument is not an explicit value,
        the sink still reads through what each call hands it, so each call
        site whose recovered parameters include the argument gets its
        blocking leaf. True when any did -- the sink is then an incomplete
        answer, not a refusal (#931 round 7)."""
        if sink.get("kind") != "arg":
            return False
        role = self._sink_role(sink)
        if role == "value":
            return False
        idx = int(sink.get("index", -1))
        done = False
        for site in self._find_callsites(instrs, str(sink.get("callee") or ""),
                                         resolve_indirect=True):
            params = self._call_params(site)
            if 0 <= idx < len(params):
                self._bw_pointer_demand(func, ssaf, site, idx, params[idx], recovered=False,
                                        sink=sink, role=role)
                done = True
        return done

    def _demand_objects(self, func: Any, ssaf: Any, expr: Any, *, disp: int | None = 0,
                        seen: set | None = None, depth: int = 0) -> list[dict[str, Any]]:
        """Every object a pointer *expr* may point into, each described --
        DESCRIPTIVE ONLY (#931 rounds 5-6): the description fills a leaf's
        detail and never decides completeness. Follows copies, phis and
        pointer arithmetic to the roots: a local's address, a constant
        address (``literal`` in read-only section storage, ``global`` in
        writable storage, with the segment's writability as a separate
        fact), a GOT/extern import, a loaded pointer, a parameter, a call's
        return (with the objects it may point into), or ``unknown``.

        *disp* is the displacement the consumer's pointer has from the
        expression being resolved, accumulated from the operand down through
        constant arithmetic (None once any non-constant term joins), so an
        object's ``offset`` is where the CONSUMER points -- `p = b + 8;
        popen(p - 8)` reads ``b`` at 0 -- never the definition's own offset."""
        seen = seen if seen is not None else set()
        if expr is None or depth > 16:
            return [{"kind": "unknown", "object": str(expr)}]
        op = op_name(expr)

        def rec(e, dd=disp):
            return self._demand_objects(func, ssaf, e, disp=dd, seen=seen, depth=depth + 1)

        def at(base: int | None) -> int | None:
            return None if disp is None or base is None else base + disp

        if "ADDRESS_OF" in op:
            v = getattr(expr, "src", None) or getattr(expr, "var", None)
            off = getattr(expr, "offset", None) if op == "MLIL_ADDRESS_OF_FIELD" else 0
            return [{"kind": "local", "object": var_label(v),
                     "offset": at(int(off) if isinstance(off, int) else None)}]
        if op in ("MLIL_IMPORT", "MLIL_EXTERN_PTR"):
            c = getattr(expr, "constant", None)
            return [{"kind": "got_or_extern", "object": str(expr), "offset": at(0),
                     **({"slot": hex(int(c))} if isinstance(c, int) else {})}]
        if "CONST" in op:
            c = self._int_const(expr)
            if c is None:
                return [{"kind": "unknown", "object": str(expr)}]
            storage = self._const_ptr_storage(c)
            seg = None
            wr = getattr(self.bv, "is_offset_writable", None)
            if wr is not None:
                try:
                    seg = bool(wr(int(c)))
                except Exception:
                    seg = None
            kind = ("literal" if storage == "readonly" and "PTR" in op
                    else "global" if storage == "writable" else "constant_address")
            return [{"kind": kind, "object": str(expr), "address": hex(c), "offset": at(0),
                     "segment_writable": seg}]
        if "LOAD" in op or op in ("MLIL_VAR_ALIASED", "MLIL_VAR_ALIASED_FIELD"):
            return [{"kind": "loaded_pointer", "object": str(expr), "offset": None}]
        if op in ("MLIL_ADD", "MLIL_SUB"):
            left, right = getattr(expr, "left", None), getattr(expr, "right", None)
            sides = [s for s in (left, right) if s is not None]
            consts = {id(s): self._int_const(s) for s in sides
                      if self._int_const(s) is not None and "PTR" not in op_name(s)}
            terms = [s for s in sides if id(s) not in consts]
            ptr = [s for s in terms
                   if any(t in op_name(s) for t in ("ADDRESS_OF", "CONST_PTR", "IMPORT", "EXTERN"))
                   or any(_is_pointer_typed(r) for r in expr_reads(s))]
            base = ptr or terms
            out: list[dict[str, Any]] = []
            for s in base:
                if disp is None or len(terms) != 1 or (op == "MLIL_SUB" and s is not left):
                    dd = None
                else:
                    k = sum(consts.values())
                    dd = disp + (-k if op == "MLIL_SUB" else k)
                out += rec(s, dd)
            return out or [{"kind": "unknown", "object": str(expr)}]
        if is_ssa_var(expr):
            key = (var_key(expr), getattr(expr, "version", None), disp)
            if key in seen:
                return []
            seen.add(key)
            try:
                d = ssaf.get_ssa_var_definition(expr)
            except Exception:
                d = None
            if d is None:
                pidx = self._param_index_of(func, expr)
                return [{"kind": "parameter", "object": var_label(expr), "offset": disp,
                         "pointer": _is_pointer_typed(expr),
                         **({"index": pidx} if pidx is not None else {})}]
            dop = op_name(d)
            if dop == "MLIL_SET_VAR_SSA":
                return rec(getattr(d, "src", None))
            if dop == "MLIL_VAR_PHI":
                out = []
                for a in getattr(d, "src", None) or []:
                    out += rec(a)
                return out
            if self._is_call(d):
                name = self._callee_name(const_target(getattr(d, "dest", None)))
                params = self._call_params(d)
                # The arguments the model says the return points into (or
                # writes through), else every pointer-looking argument. Where
                # in them the return points is unknown (strchr).
                model = lookup_model(self.models, name)[1] if name else None
                want: list[int] = []
                for rule in ((model or {}).get("sources") or []) + (
                        (model or {}).get("propagates") or []):
                    to, frm = str(rule.get("to") or ""), str(rule.get("from") or "")
                    cand = (_try_arg_index(to) if to.startswith("*arg:")
                            else _try_arg_index(frm) if to == "ret" else None)
                    if cand is not None and cand not in want:
                        want.append(cand)
                picked = ([params[i] for i in want if i < len(params)] if want else params)
                inner: list[dict[str, Any]] = []
                for p in picked:
                    if expr_reads(p) or any(t in op_name(p) for t in (
                            "ADDRESS_OF", "CONST_PTR", "IMPORT", "EXTERN")):
                        for o in rec(p, None):
                            if (o["kind"] != "parameter" or o.get("pointer")) and o not in inner:
                                inner.append(o)
                note = ({"note": "BN recovered no argument the return may point into"}
                        if not picked or any(i >= len(params) for i in want) else {})
                return [{"kind": "returned", "object": f"{name or 'an indirect call'}()",
                         "callee": name, "may_point_into": inner, **note}]
            return [{"kind": "unknown", "object": var_label(expr)}]
        reads = expr_reads(expr)
        if len(reads) == 1 and op in _COPY_OPS:
            return rec(reads[0])
        return [{"kind": "unknown", "object": str(expr)}]

    def _continue_into_callers(self, func, demands, depth, max_depth,
                               visited_funcs, base_steps,
                               param_demands: dict[int, int | None] | None = None
                               ) -> tuple[list[dict[str, Any]], set[int]]:
        """Ascend each formal demand of *func* (``{index: origin row}``) into
        every known caller, and account for each caller it did not follow.

        A formal is only an origin when nothing in the binary supplies it. A
        known caller the ascent did not follow -- its site past the
        ``MAX_CALLERS`` cap, beyond the configured ``max_depth``, with no call
        instruction at the recorded address, an unlifted caller, an arity BN
        recovered short, an argument with no def chain to walk -- and a
        reference that takes the function's address without calling it (a
        dispatch table entry; `dispatch(copy_handler, n)` handing it on as a
        callback: its callers are indirect, and none is a direct call site)
        each leave the slice short of the origin that caller supplies. They
        were skipped with at most a caveat, so a slice that dropped them
        certified itself complete (IC01). Every such (site, formal) is now a
        row of ONE ``caller_sites_truncated`` frontier, with its reason. A
        caller's literal is not a skip: it is the origin, reached. Process-entry
        and startup references are ordinary ``indirect_reference`` boundaries
        too: names, imports, operand roles and optional entry metadata prove
        neither caller-supplied origins nor completeness. No entry/runtime
        classifier is consulted, and every known unresolved reference is kept.
        Returns ``(slices, cycled)``: at least one slice per formal -- a
        formal no caller resolved keeps its own parameter slice -- except the
        ``cycled`` formals, every resolved caller of which was a back edge to
        a demand already being solved upstream: they have no origin of their
        own, and the caller keeps the producers its walk reached instead."""
        caller_sites = list(getattr(func, "caller_sites", None) or [])
        start = int(getattr(func, "start", 0))
        # address -> referencing function (None: a data reference)
        indirect: dict[int, Any] = {}
        get_data_refs = getattr(self.bv, "get_data_refs", None)
        try:
            indirect.update((int(a), None) for a in (
                get_data_refs(start) if callable(get_data_refs) else []))
        except Exception:
            pass
        get_code_refs = getattr(self.bv, "get_code_refs", None)
        direct = {int(getattr(s, "address", 0)) for s in caller_sites}
        try:
            for ref in (get_code_refs(start) if callable(get_code_refs) else []):
                a, rf = int(getattr(ref, "address", 0)), getattr(ref, "function", None)
                if a in direct:
                    continue
                indirect.setdefault(a, rf)
        except Exception:
            pass
        ascend = depth < max_depth
        # The cap offers only the first MAX_CALLERS sites, and an offered site
        # may still be unfollowable; the disclosure reports the measurement,
        # never the constant (#810).
        considered = caller_sites[:MAX_CALLERS] if ascend else []
        results: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        misses: set[tuple[int, int, str]] = set()

        def _drop(address: int, caller: Any, pidx: int, reason: str, *, listed: bool = True) -> None:
            key = (int(address), pidx, reason)
            if key in misses:
                return
            misses.add(key)
            if listed and len(rows) < MAX_CALLERS:
                rows.append({"address": hex(int(address)),
                             "caller": None if caller is None else self._prose_function(caller),
                             "param": pidx, "reason": reason})

        cycled: set[int] = set()
        fkey = self._caller_key(func)
        on_stack = {fk for fk, _ in visited_funcs}
        for pidx, porigin in demands.items():
            followed = False
            # Each site either supplies the value from outside any cycle, or
            # loops back to a demand this ascent is already solving: this
            # demand itself (`f(n - 1)`, or `a -> b -> a` remapping back to
            # it) or one further up (mutual recursion that re-enters an
            # ancestor demand, whose own callers answer it). A site that loops
            # back to this demand still supplies it when the loop is entered
            # from outside somewhere along the way (`start_b -> b -> a -> b`):
            # a caller outside the functions being solved supplied a value.
            # A demand whose cycle closes on itself with no such entry is the
            # cycle's root: its formal stays an origin beside whatever the
            # cycle adds. One whose every site loops to an ancestor has no
            # origin of its own -- never the internal formal.
            via_cycle = self_closed = supplied = False
            for site in considered:
                caller = getattr(site, "function", None)
                addr = int(getattr(site, "address", 0))
                reason = None
                call_ins = None
                if caller is None:
                    reason = "caller_unknown"
                else:
                    try:
                        c_instrs = self._instrs(self._ssa_func(caller))
                    except TaintError:
                        reason = "caller_unlifted"
                    else:
                        call_ins = next((i for i in c_instrs if self._is_call(i)
                                         and int(getattr(i, "address", 0)) == addr), None)
                        if call_ins is None:
                            reason = "no_call_instruction"
                cparams = self._call_params(call_ins) if call_ins is not None else []
                if reason is None and pidx >= len(cparams):
                    reason = "arity_short"
                if reason is not None:
                    _drop(addr, caller, pidx, reason)
                    continue
                arg = cparams[pidx]
                # `run(&buf)` / `run(&buf + i)` with the address folded into
                # the call parameter: the callee's parameter is the buffer's
                # contents, so the ascent walks the caller's writers of it --
                # the same path `p = &buf; run(p)` takes -- beside the offset.
                seeds: list[Any] = expr_reads(arg) + self._folded_buffer_seeds(arg)
                boundary = _instr_dict(call_ins, reason=f"passed as arg {pidx} to {func.name}")
                if not seeds:
                    value = self._int_const(arg) if op_name(arg) == "MLIL_CONST" else None
                    if value is not None:
                        # A literal is the origin this caller supplies. One a
                        # consumer reads THROUGH (`run(0)`) names no object,
                        # and that sink's pointer demand already withholds
                        # completeness; a constant ADDRESS stays unfollowed.
                        results.append({"steps": [boundary] + base_steps,
                                        "origin": {"kind": "constant", "value": value,
                                                   "function": hex(int(caller.start))},
                                        "crossed": [str(func.name)]})
                        followed = supplied = True
                        if self._caller_key(caller) not in on_stack:
                            self._bw_entries.append((self._caller_key(caller), addr))
                        continue
                    self._bw_assume(
                        f"arg {pidx} at {hex(addr)} in {self._prose_function(caller)} is not a trackable "
                        "scalar; caller slice not followed")
                    _drop(addr, caller, pidx, "constant_address"
                          if "CONST" in op_name(arg) or op_name(arg) in ("MLIL_IMPORT", "MLIL_EXTERN_PTR")
                          else "argument_not_trackable")
                    continue
                site_slices = 0
                mark, emark = len(self._bw_cycle_hits), len(self._bw_entries)
                for sv in seeds:
                    # A parameter the callee read through is read in the
                    # caller's object at the same delta from the argument.
                    sub_anchor = None
                    if param_demands and pidx in param_demands:
                        sub_anchor = (self._seed_anchor(sv, arg), param_demands[pidx])
                    for sub in self._backward_slice(caller, sv, depth + 1, max_depth, visited_funcs,
                                                    seed_ins=call_ins, anchor=sub_anchor):
                        results.append({
                            "steps": sub["steps"] + [boundary] + base_steps,
                            "origin": sub["origin"],
                            "crossed": [str(func.name)] + sub["crossed"],
                        })
                        followed = True
                        site_slices += 1
                # An entry into the cycle: a caller outside the functions
                # being solved that supplied the value without looping back.
                if (site_slices and self._caller_key(caller) not in on_stack
                        and not self._bw_cycle_hits[mark:]):
                    self._bw_entries.append((self._caller_key(caller), addr))
                closes_here = (fkey, pidx) in self._bw_cycle_hits[mark:]
                entered = len(self._bw_entries) > emark
                self_closed |= closes_here
                if site_slices and (entered or not closes_here):
                    supplied = True
                elif not site_slices:
                    via_cycle = True
            if self_closed and not supplied:
                results.append({"steps": base_steps, "origin": porigin, "crossed": []})
            elif not followed:
                if via_cycle:
                    cycled.add(pidx)
                else:
                    results.append({"steps": base_steps, "origin": porigin, "crossed": []})

        for site in caller_sites[len(considered):]:
            for pidx in demands:
                # Past the cap the sites are counted, not listed (#810).
                _drop(int(getattr(site, "address", 0)), getattr(site, "function", None), pidx,
                      "max_depth" if not ascend else "caller_cap", listed=not ascend)
        for a, rf in sorted(indirect.items()):
            for pidx in demands:
                _drop(a, rf, pidx, "indirect_reference")
        self._bw_note_unfollowed_callers(
            func, direct=direct, indirect=set(indirect), formals=set(demands),
            rows=rows, misses=misses)
        return results, cycled

    @staticmethod
    def _caller_key(func: Any) -> tuple[int, str]:
        """A function's identity in the ascent's demand keys."""
        return (int(getattr(func, "start", 0)), str(getattr(func, "name", "")))

    def _bw_note_unfollowed_callers(
        self, func: Any, *, direct: set[int], indirect: set[int], formals: set[int],
        rows: list[dict[str, Any]], misses: set[tuple[int, int, str]],
    ) -> None:
        """One caller frontier per function, across every demanded formal.

        Coverage counts distinct reference addresses, each classed by the
        formals demanded of it: FULLY followed (every demanded formal
        followed), PARTIALLY followed (some), or UNFOLLOWED (none).
        ``callers_dropped`` is the partial and unfollowed references together
        -- every reference that left at least one demand unfollowed -- and the
        (reference, formal) demand counts say how many demands that was.
        Reasons and examples are separate evidence about (reference, formal,
        reason) boundaries, so their tallies may exceed the number of dropped
        references. Keep full identity internally even when the public
        examples are capped at MAX_CALLERS.
        """
        start = int(func.start)
        state = self._bw_caller_frontiers.get(start)
        if state is None:
            state = {
                "direct": set(), "known": set(), "dropped": set(), "misses": set(),
                "demanded": {}, "missed": {},
                "reasons": {}, "examples": {}, "leaf": None, "cap_assumption": None,
            }
            self._bw_caller_frontiers[start] = state
        # Merge after descent into the current accumulator: a recursive ascent
        # may already have recorded this function's other formal demands.
        state["direct"].update(direct)
        state["known"].update(direct)
        state["known"].update(indirect)
        for address in direct | indirect:
            state["demanded"].setdefault(address, set()).update(formals)
        for address, param, reason in misses:
            state["dropped"].add(address)
            state["missed"].setdefault(address, set()).add(param)
            key = (address, param, reason)
            if key not in state["misses"]:
                state["misses"].add(key)
                state["reasons"][reason] = state["reasons"].get(reason, 0) + 1
        for row in rows:
            key = (row["address"], row["param"], row["reason"])
            if key not in state["examples"] and len(state["examples"]) < MAX_CALLERS:
                state["examples"][key] = row
        if not state["dropped"]:
            return

        total = len(state["known"])
        dropped = len(state["dropped"])
        fully = total - dropped
        partial: list[tuple[int, set[int], set[int]]] = []
        for address in sorted(state["dropped"]):
            demanded = state["demanded"].get(address, set())
            missed = state["missed"][address]
            if demanded - missed:
                partial.append((address, demanded - missed, missed))
        unfollowed = dropped - len(partial)
        demands_total = sum(len(state["demanded"].get(a, set())) for a in state["known"])
        demands_unfollowed = sum(len(m) for m in state["missed"].values())
        reasons = dict(sorted(state["reasons"].items()))
        name = self._prose_function(func)
        capped = "caller_cap" in reasons
        if capped:
            self._bw_truncated = True
            self._bw_truncation_causes.add("caller_cap")
            assumption = (
                f"{name} has {len(state['direct'])} callers; caller ascent fully followed "
                f"{len(state['direct'] - state['dropped'])}, capped at {MAX_CALLERS} "
                f"(function {start:#x})")
            old = state["cap_assumption"]
            if old != assumption:
                if old is not None:
                    self._bw_assumptions.remove(old)
                self._bw_assume(assumption)
                state["cap_assumption"] = assumption
        note = (
            ("caller-site cap: " if capped else "")
            + f"of {total} distinct caller reference(s), the ascent fully followed {fully} "
            f"(every demanded formal), partially followed {len(partial)} and followed "
            f"none of {unfollowed}; {demands_unfollowed} of {demands_total} "
            "(reference, formal) demand(s) unfollowed: "
            + ", ".join(f"{n} {r}" for r, n in reasons.items()))
        leaf = state["leaf"]
        if leaf is None:
            leaf = {
                "kind": "caller_sites_truncated", "address": hex(start),
                "function": {"name": str(getattr(func, "name", "?")),
                             "display_name": _display_name(func) or "?",
                             "address": hex(start)},
            }
            state["leaf"] = leaf
            self._bw_leaves.append(leaf)
        leaf.update({
            "callers_total": total,
            "callers_fully_followed": fully,
            "callers_partially_followed": len(partial),
            "callers_unfollowed": unfollowed,
            "callers_dropped": dropped,
            "demands_total": demands_total,
            "demands_unfollowed": demands_unfollowed,
            "reasons": reasons,
            "unfollowed": list(state["examples"].values()),
            # Which formals each partially followed reference did and did not
            # supply; bounded like `unfollowed`, counted exactly above.
            "partially_followed": [
                {"address": hex(a), "followed_params": sorted(f), "unfollowed_params": sorted(m)}
                for a, f, m in partial[:MAX_CALLERS]],
            "note": (note + " -- an origin reachable only from an unfollowed caller "
                     "is missing from this slice"),
        })

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

