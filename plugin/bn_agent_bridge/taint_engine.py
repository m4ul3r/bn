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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # paths is a symlink into the bridge dir; tolerate import-time absence
    from .paths import taint_models_path
except Exception:  # pragma: no cover - defensive
    taint_models_path = None  # type: ignore[assignment]

_BUILTIN_MODELS = Path(__file__).resolve().parent / "taint_models.json"

SOUNDNESS = (
    "may-analysis (interprocedural, summary-based, depth-bounded); memory is "
    "tracked via SSA store/load correlation where addresses match and coarsely "
    "otherwise; unresolved indirect/external calls are surfaced as assumptions/"
    "leaves; NOT a proof of reachability"
)

# Overflow sink classes whose finding is reclassified to `tainted_index` when the
# taint reaches the sink only through an array index/offset (#163). Covers the
# plain copy/length classes and the fortified (__*_chk / *_s) family, which spans
# both copy-source pointer args and length args -- the per-arg pointer role is
# resolved separately from the model's buffer-propagation rules, not the class.
_OVERFLOW_INDEX_CLASSES = frozenset({"overflow_unbounded", "overflow_len", "fortified_overflow"})

# Scatter-gather receive calls: the received bytes land in msghdr->msg_iov[i].
# iov_base, not the msghdr pointer arg, so seeding the arg taints the header, not
# the payload (#306). Names compared after stripping leading underscores / @plt.
_RECVMSG_FAMILY = frozenset({"recvmsg", "recvmmsg"})

# A param: source that is a pointer to an aggregate at least this large (by byte
# size OR member count) is flagged as a "broad source" -- the whole struct is
# treated as one tainted location, which over-taints into unrelated code (#219).
_BROAD_SOURCE_BYTES = 0x40
_BROAD_SOURCE_MEMBERS = 8


def _model_buffer_source_args(model: dict[str, Any]) -> frozenset[int]:
    """Arg indices the model propagates a buffer FROM (``*arg:N`` in a
    ``propagates`` rule's ``from``). These are the copy-SOURCE pointer args (e.g.
    strcpy / __strcpy_chk arg1), as opposed to length scalars -- used to decide
    whether the index-role broadening applies to a given sink arg (#163)."""
    out: set[int] = set()
    for rule in model.get("propagates") or []:
        frm = rule.get("from")
        # Require the POINTEE form ``*arg:N`` -- the buffer arg N points at. A
        # scalar ``arg:N`` (arg N's value, e.g. GLib g_slist_append's
        # ``from: "arg:1"``) is NOT a buffer source, and treating it as one would
        # enable the pointer index broadening on a scalar/length arg (#163 review).
        if isinstance(frm, str) and frm.startswith("*arg:"):
            try:
                out.add(int(frm[len("*arg:"):]))
            except ValueError:
                pass
    return frozenset(out)


class TaintError(RuntimeError):
    """User-facing taint configuration/resolution error."""


class BoundedSink(Exception):
    """A backward sink whose argument is a compile-time constant (e.g. a fixed
    copy length): provably bounded, with no def-chain to slice. This is a
    SUCCESSFUL conclusion, not a seed failure -- it must NOT count toward the
    all-sinks-failed hard error, so a bounded sink returns a clean result
    (exit 0, --out written) instead of looking like a crash (#310)."""


# --------------------------------------------------------------------------
# model database
# --------------------------------------------------------------------------

def _coerce_model_map(raw: Any, *, source: str) -> dict[str, Any]:
    """Validate a parsed model DB and return its name->model map.

    Accepts either ``{"models": {...}}`` or a bare ``{name: model}`` map; rejects
    any other top-level shape so a malformed file can't silently merge to nothing
    (a model whose value isn't a dict couldn't carry source/sink/propagate data).
    """
    if isinstance(raw, dict) and "models" in raw:
        raw = raw.get("models")
    if not isinstance(raw, dict):
        raise TaintError(
            f"{source} must be a JSON object of name->model (or {{\"models\": {{...}}}}); "
            f"got {type(raw).__name__}"
        )
    for name, model in raw.items():
        # `_comment*` keys are free-text documentation in the DB (their string
        # values never match a symbol, so lookup_model ignores them); every real
        # model name must map to an object that can carry source/sink/propagate.
        if str(name).startswith("_comment"):
            continue
        if not isinstance(model, dict):
            raise TaintError(
                f"{source}: model {name!r} must be a JSON object, got {type(model).__name__}"
            )
    return raw


def load_models(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the merged function-model DB: builtin <- user override <- extra.

    Model-load failures used to be swallowed, silently degrading into missing
    source/sink/propagation models -- false negatives indistinguishable from
    analysis limits (#97). Now a broken builtin DB (a packaging bug) and a broken
    BN_TAINT_MODELS override (a user typo that should be loud, not silent) both
    raise TaintError, which the taint command surfaces as a clean error.
    """
    models: dict[str, Any] = {}
    try:
        raw = json.loads(_BUILTIN_MODELS.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TaintError(
            f"builtin taint model DB at {_BUILTIN_MODELS} could not be loaded: {exc}. "
            "This is a packaging bug -- reinstall the bridge."
        ) from exc
    models.update(_coerce_model_map(raw, source=f"builtin taint model DB ({_BUILTIN_MODELS})"))
    if taint_models_path is not None:
        override_path = taint_models_path()
        if override_path.exists():
            try:
                raw = json.loads(override_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise TaintError(
                    f"BN_TAINT_MODELS override at {override_path} could not be loaded: {exc}. "
                    "Fix or remove the file (it overrides the builtin models)."
                ) from exc
            models.update(_coerce_model_map(raw, source=f"BN_TAINT_MODELS override ({override_path})"))
    if extra:
        models.update(extra)
    return models


def _canonical_cxx_alloc(name: str) -> str | None:
    """Canonical model key for a C++ ``operator new`` / ``operator new[]``
    spelling -- mangled or demangled -- or None when *name* is neither (#204).

    BN renders these allocators either Itanium-mangled (``Znwm``/``Znwj`` for
    ``operator new``, ``Znam``/``Znaj`` for ``operator new[]``; the ``m``/``j``
    is the 64-/32-bit ``size_t`` overload, with optional ``St11align_val_t`` /
    ``RKSt9nothrow_t`` suffixes) or demangled (``operator new(unsigned long)``).
    All of those allocate with the size at arg 0, so they collapse to the two
    keys ``Znwm`` / ``Znam``. Placement new (``_ZnwmPv`` / ``operator
    new(unsigned long, void*)``) constructs in caller-supplied storage -- it
    does NOT allocate -- so it is excluded to avoid a false ``alloc_size`` sink.

    *name* is expected already stripped of a leading ``_`` (as ``lookup_model``
    passes it).
    """
    if not name:
        return None
    if name.startswith("operator new"):
        if "void*" in name or "void *" in name:        # placement new
            return None
        return "Znam" if name.startswith("operator new[]") else "Znwm"
    if name.startswith("Znw") or name.startswith("Zna"):
        if name[3:4] not in ("m", "j"):                # not a size_t overload
            return None
        if name[4:].startswith("Pv"):                  # placement new (..., void*)
            return None
        return "Znam" if name.startswith("Zna") else "Znwm"
    return None


def lookup_model(models: dict[str, Any], name: str | None) -> tuple[str | None, dict[str, Any] | None]:
    """Match a (possibly decorated) symbol name against the model DB.

    Tries the raw name, then the part before ``@`` (``memcpy@plt`` ->
    ``memcpy``), then with leading underscores stripped, then the canonical
    C++ allocator key (so mangled ``_Znam`` and demangled ``operator new[]``
    both resolve to the ``Znam`` model -- #204).
    """
    if not name:
        return None, None
    candidates = [name]
    base = name.split("@", 1)[0]
    if base != name:
        candidates.append(base)
    stripped = base.lstrip("_")
    if stripped and stripped != base:
        candidates.append(stripped)
    alias = _canonical_cxx_alloc(stripped or base)
    if alias and alias not in candidates:
        candidates.append(alias)
    for cand in candidates:
        if cand in models:
            return cand, models[cand]
    return None, None


def _try_arg_index(token: str) -> int | None:
    """Parse N from a model source token ``arg:N`` / ``*arg:N``; None if malformed."""
    try:
        return int(str(token).split("arg:", 1)[1])
    except (IndexError, ValueError):
        return None


# --------------------------------------------------------------------------
# small IL helpers (defensive getattr style, matching bridge.py)
# --------------------------------------------------------------------------

def op_name(item: Any) -> str:
    operation = getattr(item, "operation", None)
    name = getattr(operation, "name", None)
    return str(name) if name else str(operation)


def is_ssa_var(v: Any) -> bool:
    return hasattr(v, "version") and hasattr(v, "var")


# printf-family conversion specifier: optional flags, width, precision, length
# modifier, then a conversion char. `%%` is a literal (consumes nothing); a `*`
# width or precision each consume one extra int arg.
_FORMAT_SPEC_RE = re.compile(
    r"%([-+ 0#]*)(\*|[0-9]+)?(?:\.(\*|[0-9]+))?(?:hh|h|ll|l|L|q|j|z|t)?([diouxXeEfFgGaAcspn%])")


def _count_format_args(fmt: str) -> int | None:
    """Number of varargs a printf-style *constant* format string consumes, so a
    tainted vararg beyond it can be recognized as a provably-dead flow (#45).

    Returns None when the consumed count can't be determined reliably -- a POSIX
    positional ``%n$`` specifier reorders/reuses args, so a positional format is
    not the linear consumption this regex pass assumes (#69). The caller stays
    conservative (every vararg live) on None, avoiding a false-negative sink."""
    if re.search(r"%\d+\$", fmt):
        return None
    n = 0
    for m in _FORMAT_SPEC_RE.finditer(fmt):
        if m.group(4) == "%":
            continue
        if m.group(2) == "*":
            n += 1
        if m.group(3) == "*":
            n += 1
        n += 1
    return n


def var_key(v: Any) -> tuple[str, Any]:
    """Stable identity for a Variable or the base of an SSAVariable."""
    base = getattr(v, "var", v)
    ident = getattr(base, "identifier", None)
    if ident is not None:
        try:
            return ("id", int(ident))
        except Exception:
            pass
    return ("name", str(getattr(base, "name", base)))


def var_label(v: Any) -> str:
    base = getattr(v, "var", v)
    name = str(getattr(base, "name", base))
    version = getattr(v, "version", None)
    return f"{name}#{version}" if version is not None else name


def ssa_reads(ins: Any) -> list[Any]:
    """SSAVariables an instruction reads by value (excludes AddressOf targets,
    which appear in vars_read as plain Variables)."""
    return [v for v in (getattr(ins, "vars_read", None) or []) if is_ssa_var(v)]


def ssa_writes(ins: Any) -> list[Any]:
    return [v for v in (getattr(ins, "vars_written", None) or []) if is_ssa_var(v)]


def expr_reads(expr: Any) -> list[Any]:
    return [v for v in (getattr(expr, "vars_read", None) or []) if is_ssa_var(v)]


def function_at(bv: Any, addr: int | None) -> Any | None:
    """``get_function_at`` with ARM/Thumb low-bit normalization.

    A code pointer on ARM carries the Thumb tag in its LSB, so an indirect/
    value-set target address can be odd while the function lives at ``addr & ~1``.
    The exact lookup is tried first (raw address preserved for diagnostics), then
    the normalized one (#89). Returns None when neither resolves or *bv* lacks
    ``get_function_at`` (the synthetic test fakes).
    """
    if addr is None or not hasattr(bv, "get_function_at"):
        return None
    try:
        fn = bv.get_function_at(addr)
    except Exception:
        fn = None
    if fn is None and (addr & 1):
        try:
            fn = bv.get_function_at(addr & ~1)
        except Exception:
            fn = None
    return fn


def const_target(expr: Any) -> int | None:
    """Constant call destination (direct call) or None (indirect).

    Accepts MLIL_CONST_PTR *and* MLIL_EXTERN_PTR. On a statically-linked object
    (e.g. a kernel .ko) a direct `bl` to an external helper (strlen / sscanf /
    memcpy / copy_from_user) renders as an EXTERN_PTR whose `.constant` IS the
    stub address that `get_symbol_at` resolves to the real name -- so without
    this the callee name (and its sink/source model) is never recovered and the
    call is misreported as an unresolved indirect call, all-clearing every .ko
    sink reached through a stub. (T2; MLIL_IMPORT stays out: its `.constant` is
    a GOT slot, resolved by name via resolve_call_target/extract_dest_address.)
    """
    if expr is None:
        return None
    name = op_name(expr)
    if "CONST" not in name and "EXTERN_PTR" not in name:
        return None
    c = getattr(expr, "constant", None)
    if c is None:
        c = getattr(expr, "value", None)
        c = getattr(c, "value", c)
    try:
        return int(c)
    except Exception:
        return None


def _instr_dict(ins: Any, reason: str | None = None, tainted: list[str] | None = None) -> dict[str, Any]:
    out = {
        "il_index": int(getattr(ins, "instr_index", -1)),
        "address": hex(int(getattr(ins, "address", 0))),
        "op": op_name(ins),
        "il_text": str(ins),
    }
    if reason is not None:
        out["reason"] = reason
    if tainted is not None:
        out["tainted"] = tainted
    return out


# --------------------------------------------------------------------------
# unified call-target / thunk resolver
# --------------------------------------------------------------------------
# Canonical home (issue #7) for "what does this call target, through thunks?".
# Faithful ports of the trace resolver that lived in bridge.py; the bridge now
# delegates here. Kept import-free of binaryninja: all BN access is via
# duck-typed methods on the passed objects, and symbol-table lookups are guarded
# so the resolver degrades gracefully against the synthetic fakes the tests use.


def _mlil_ssa(fn: Any) -> Any:
    """MLIL SSA form of a function, tolerating the ``mlil`` vs ``medium_level_il``
    attribute alias (real BN exposes both; the test fakes expose only ``mlil``)."""
    mlil = getattr(fn, "mlil", None) or getattr(fn, "medium_level_il", None)
    if mlil is None:
        return None
    return getattr(mlil, "ssa_form", None)


def _ssa_instructions(ssaf: Any) -> list[Any]:
    """All SSA instructions, preferring ``.instructions`` and falling back to
    flattening ``.basic_blocks`` (both are valid on real BN)."""
    try:
        return list(ssaf.instructions)
    except Exception:
        out: list[Any] = []
        for block in (getattr(ssaf, "basic_blocks", None) or []):
            try:
                out.extend(list(block))
            except Exception:
                continue
        return out


def _symbols_by_name(bv: Any, name: str) -> list[Any]:
    fn = getattr(bv, "get_symbols_by_name", None)
    if fn is None:
        return []
    try:
        return list(fn(name) or [])
    except Exception:
        return []


def _symbol_by_raw_name(bv: Any, name: str) -> Any:
    fn = getattr(bv, "get_symbol_by_raw_name", None)
    if fn is None:
        return None
    try:
        return fn(name)
    except Exception:
        return None


def extract_dest_address(bv: Any, dest: Any) -> int | None:
    """Numeric address of a call/tailcall destination expression, or None.

    Handles a raw int, MLIL_CONST_PTR (``.constant``), and MLIL_IMPORT. For
    imports the symbol name is resolved *before* ``.constant`` because
    ``.constant`` is the GOT slot, not the function entry point.
    """
    try:
        return int(dest)
    except (ValueError, TypeError):
        pass
    name = getattr(dest, "name", None) or str(dest)
    if name:
        for sym in _symbols_by_name(bv, name):
            fn = bv.get_function_at(sym.address)
            if fn is not None:
                return int(fn.start)
        sym = _symbol_by_raw_name(bv, name)
        if sym is not None:
            fn = bv.get_function_at(sym.address)
            if fn is not None:
                return int(fn.start)
    addr = getattr(dest, "constant", None)
    if addr is not None:
        return int(addr)
    return None


def targets_from_pvs(pvs: Any) -> list[int]:
    """Extract concrete call-target addresses from a PossibleValueSet.

    Handles constants, in-set values, and lookup tables (function-pointer
    tables expose ``.mapping`` {idx: addr} / ``.table``, not ``.values``).
    """
    if pvs is None:
        return []
    tname = str(getattr(getattr(pvs, "type", None), "name", "") or "")
    out: list[int] = []

    def _add(v):
        try:
            out.append(int(v))
        except Exception:
            pass

    if tname in {"ConstantValue", "ConstantPointerValue", "ImportedAddressValue", "ExternalPointerValue"}:
        _add(getattr(pvs, "value", None))
    elif tname == "InSetOfValues":
        for v in (getattr(pvs, "values", None) or []):
            _add(v)
    elif tname == "LookupTableValue":
        mapping = getattr(pvs, "mapping", None)
        if isinstance(mapping, dict):
            for v in mapping.values():
                _add(v)
        else:
            for entry in (getattr(pvs, "table", None) or []):
                _add(getattr(entry, "to", None))
    return sorted({a for a in out if a})


def follow_thunk(bv: Any, fn: Any) -> Any | None:
    """If *fn* is a single-instruction tailcall thunk, return its real target.

    Follows thunk chains (PLT stubs, GCC veneers) iteratively, guarding against
    cycles of *any* length via a visited-set of function starts. The old
    recursive form only rejected a direct A->A self-loop, so a multi-step
    tailcall cycle in the binary (A->B->A, or longer) recursed without bound and
    raised ``RecursionError``. Returns the deepest resolved target, or None when
    *fn* is not a thunk.
    """
    seen: set[int] = {int(getattr(fn, "start", -1))}
    cur = fn
    result: Any | None = None
    while True:
        ssa = _mlil_ssa(cur)
        if ssa is None:
            break
        instructions = _ssa_instructions(ssa)
        if len(instructions) != 1:
            break
        insn = instructions[0]
        if "TAILCALL" not in op_name(insn):
            break
        dest = getattr(insn, "dest", None)
        if dest is None:
            break
        addr = extract_dest_address(bv, dest)
        if addr is None:
            break
        target = bv.get_function_at(addr)
        if target is None:
            target = bv.get_function_at(addr & ~1)
        if target is None or int(getattr(target, "start", -2)) in seen:
            break
        seen.add(int(getattr(target, "start", -1)))
        result = target
        cur = target
    return result


@dataclass
class ResolvedTarget:
    """Result of resolving a call instruction's callee."""
    address: int | None         # entry of the resolved function (post-thunk if followed)
    function: Any | None        # bv function object, or None
    via: str | None = None      # "direct" | "import" | "thunk" | None
    thunk_chain: list[int] = field(default_factory=list)  # addresses traversed while following thunks


def resolve_call_target(bv: Any, call_insn: Any, *, follow_thunks: bool = False) -> ResolvedTarget:
    """Resolve a call instruction's callee to a function.

    Mirrors the trace resolver: direct numeric/``.constant`` dest first, then
    import-name resolution (the primary path for PLT stubs, where ``.constant``
    is a GOT slot), then optional thunk following. Value-sets and agent-supplied
    resolve-maps are deliberately *not* consulted here — those are forward-taint
    concerns (see ``targets_from_pvs`` and the engine's resolve_map handling).
    """
    dest = getattr(call_insn, "dest", None)
    if dest is None:
        return ResolvedTarget(None, None)

    fn = None
    via: str | None = None
    try:
        addr = int(dest)
    except (ValueError, TypeError):
        addr = getattr(dest, "constant", None)
    if addr is not None and addr != 0:
        fn = bv.get_function_at(int(addr))
        if fn is None:
            fn = bv.get_function_at(int(addr) & ~1)
        if fn is not None:
            via = "direct"

    if fn is None:
        name = getattr(dest, "name", None)
        if name:
            for sym in _symbols_by_name(bv, name):
                fn = bv.get_function_at(sym.address)
                if fn is not None:
                    via = "import"
                    break
            if fn is None:
                sym = _symbol_by_raw_name(bv, name)
                if sym is not None:
                    fn = bv.get_function_at(sym.address)
                    if fn is not None:
                        via = "import"

    if fn is None:
        return ResolvedTarget(None, None)

    thunk_chain: list[int] = []
    if follow_thunks:
        resolved = follow_thunk(bv, fn)
        if resolved is not None and resolved is not fn:
            thunk_chain = [int(getattr(fn, "start", 0))]
            fn = resolved
            via = "thunk"

    return ResolvedTarget(int(getattr(fn, "start", 0)), fn, via, thunk_chain)


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

    # -- global/static buffers as taint locations -------------------------
    # A global buffer is referenced by an absolute address (MLIL_CONST_PTR), which
    # _pointee_var (stack-only) misses. We make it a single coarse taint location
    # keyed ("global", base_addr). Precise offset/aliasing is deliberately out of
    # scope (the domain of a heavyweight whole-program/CPG analyzer); we
    # over-approximate the whole buffer.

    def _buffer_target(self, ssaf: Any, expr: Any):
        """Resolve a pointer expr to ``(key, label)`` for the buffer it points at:
        a stack Variable (preferred — keeps its name) or a writable global. None if
        neither. The key is what taint nodes are keyed on; label is for provenance."""
        pv = self._pointee_var(ssaf, expr)
        if pv is not None:
            return (var_key(pv), var_label(pv))
        ga = self._global_addr(ssaf, expr)
        if ga is not None:
            return (("global", ga), f"glob_{hex(ga)}")
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

    def _walk_mem(self, ssaf, mv, la, tainted, seen, depth):
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
            if sa is not None and sa == la:
                for r in expr_reads(getattr(defn, "src", None)):
                    if (var_key(r), getattr(r, "version", None)) in tainted or (var_key(r), None) in tainted:
                        return (var_key(r), getattr(r, "version", None))
                return None  # matching store wrote untainted data -> not tainted via memory
            return self._walk_mem(ssaf, getattr(defn, "src_memory", None), la, tainted, seen, depth + 1)
        if "MEM_PHI" in op:
            for sv in self._mem_phi_sources(defn):
                res = self._walk_mem(ssaf, sv, la, tainted, seen, depth + 1)
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
        return self._walk_mem(ssaf, mv, la, tainted, set(), 0)

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

    def _reaching_writer(self, ssaf, mv, la, seen, depth):
        """Walk the memory-SSA chain backward from version *mv* for the writer of
        address *la* = ``(base, offset)``. Returns ``("store", defn)`` for a
        matching ``MLIL_STORE``, ``("source", call_defn, callee)`` for a modeled
        source call that fills la's buffer, or None when the chain ends without a
        recoverable writer (a genuinely unresolved field load, #158)."""
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
            if sa is not None and sa == la:
                return ("store", defn)
            return self._reaching_writer(ssaf, getattr(defn, "src_memory", None), la, seen, depth + 1)
        if "MEM_PHI" in op:
            for sv in self._mem_phi_sources(defn):
                res = self._reaching_writer(ssaf, sv, la, seen, depth + 1)
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
        self._only_callsite_addr = only_callsite_addr

        try:
            sub = self._run_forward(func, sources, depth=0, max_depth=max_depth, top=True)
        except RecursionError:
            # Defense in depth: should not happen now that thunk-following and the
            # SSA/call walks are cycle-guarded, but a pathological binary must
            # degrade to a bounded, honest result rather than crash the whole op.
            self._truncated = True
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
        return {
            "direction": "forward",
            "function": {"name": str(func.name), "address": hex(int(func.start))},
            "sources": [self._describe_locator(s) for s in sources],
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
            },
            "soundness": SOUNDNESS,
        }

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

        return {
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
            },
            "soundness": SOUNDNESS,
        }

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
        try:
            d = ssaf.get_ssa_var_definition(var)
        except Exception:
            d = None
        if d is None or "CALL" not in op_name(d):
            return None
        callee = self._callee_name(self._resolve_direct_target(d))
        base = (callee or "").split("@", 1)[0].lstrip("_")
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
                                       buffer_slots, add_assumption) -> None:
        """When a buffer source's pointer is an indirect load `[slot]`, register the
        slot so a later re-load of it can be correlated forward (#193 Part 1). If the
        slot can't be named, fall back to today's honest "may be missed" caveat. When
        it can, defer that caveat -- `_forward_run` decides post-fixpoint whether the
        slot actually correlated (positive note) or not (the caveat)."""
        if not self._arg_ptr_is_indirect_load(ssaf, ptr_expr):
            return
        pending = (
            f"source {callee} arg{idx} buffer pointer is loaded indirectly (from a "
            f"global/struct slot); the seed anchors to the pointer value, not the "
            f"pointee, and is not correlated with later re-loads of the same slot -- a "
            f"flow that re-loads the pointer and parses it may be missed. Consider "
            f"seeding the parser entry directly with param:N")
        key = self._buffer_slot_key(ssaf, ptr_expr)
        if key is None:
            add_assumption(pending)
            return
        recv_node = None
        for r in expr_reads(ptr_expr):
            recv_node = (var_key(r), getattr(r, "version", None))
            break
        buffer_slots[key] = {
            "recv_idx": getattr(callsite, "instr_index", None),
            "callee": callee, "idx": idx, "recv_node": recv_node, "pending": pending,
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
        try:
            sub = self._run_forward(callee, locators, depth, max_depth, top=False)
        except TaintError as exc:
            sub = {"reached_return": True, "out_params": frozenset(), "findings": [], "leaves": [],
                   "assumptions": [f"could not analyze {callee.name}: {exc}; return conservatively tainted"],
                   "frontier": f"body could not be analyzed ({exc})"}
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
                "tainted data passed to in-binary callee with no model and no "
                "mappable parameters; investigate"))
            return out
        if depth + 1 > max_depth:
            self._truncated = True
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
        prefix.append(_instr_dict(ins, reason=note, tainted=[node_label(first_hit, why)]))
        for f in sub["findings"]:
            out["findings"].append({"sink": f["sink"], "path": prefix + f["path"]})
        out["leaves"] = list(sub["leaves"])
        frontier = sub.get("frontier")
        if frontier:
            out["leaves"].append(self._frontier_leaf(
                ins, callee_fn, tainted_args,
                f"tainted data passed to in-binary callee with no model; {frontier} "
                f"-- investigate"))
        out["assumptions"] = list(sub["assumptions"])
        out["reached_return"] = sub["reached_return"]
        out["out_params"] = sub.get("out_params", frozenset())
        return out

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

        tainted: set[tuple] = set()
        why: dict[tuple, dict[str, Any]] = {}
        assumptions: list[str] = []
        leaves: list[dict[str, Any]] = []
        findings: list[dict[str, Any]] = []
        recorded_sinks: set[tuple] = set()
        processed_calls: set[tuple] = set()  # (call_addr, tainted-arg-set) already descended
        out_params: set[int] = set()         # this func's params whose pointee got tainted
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
        seeded = self._seed_forward(func, ssaf, instrs, locators, taint_node, add_assumption, buffer_slots)
        if not seeded:
            if top:
                raise TaintError("no taint sources resolved; check --source locator")
            return {"reached_return": False, "out_params": set(), "findings": [],
                    "leaves": [], "assumptions": []}

        # Honesty signal: a visited function with unlifted instructions is a
        # potential silent dataflow hole -- BN couldn't model those ops, so taint
        # through them isn't tracked. Surface it the way unmodeled calls/coarse
        # stores already are, instead of flowing through silently (#206).
        unimpl = self._unimplemented_addrs(func, instrs)
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
            sink = model.get("sink")
            # opt-in sinks (e.g. file_write) stay silent unless their class was
            # enabled for this run; still a "modeled" call, so no fallback noise.
            if sink is not None and sink.get("optional") and sink.get("class") not in self._enabled_sink_classes:
                sink = None
            if sink is not None:
                for argidx in sink.get("tainted_args", []) or []:
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
                                _bnd_reason = (self._bounded_copy_reason(ssaf, params, argidx)
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
                                findings.append(self._make_finding(ins, mkey or name, argidx, eff_sink, ht, why))
            for rule in model.get("propagates") or []:
                to = rule.get("to")
                frm = rule.get("from")
                hit = self._token_hit_node(ssaf, params, frm, tainted)
                if hit is not None:
                    if self._apply_to_token(ssaf, ins, params, to, taint_node, name or "?", parents=[hit]):
                        changed = True
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
                    if isinstance(frm, str) and frm.startswith("*arg:") and to == "*arg:0":
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
                for i in range(max(base, 0), upper):
                    ht = arg_taint(params[i])
                    if not ht:
                        continue
                    if vto:
                        if self._apply_to_token(ssaf, ins, params, vto, taint_node, name or "?", parents=[ht[0]]):
                            changed = True
                        if vto.startswith("*arg:"):
                            k = int(vto.split("arg:", 1)[1])
                            if k < len(params):
                                propagated.add(k)
                    if vsink is not None:
                        # record under the real param index i so this shares the
                        # recorded_sinks / top-level dedup with any static sink.
                        sig = (addr, i) if site_taddr is None else (addr, i, site_taddr)
                        if sig not in recorded_sinks:
                            recorded_sinks.add(sig)
                            findings.append(self._make_finding(ins, mkey or name, i, vsink, ht, why))
            return changed, propagated

        for _ in range(self.max_iters):
            changed = False
            for ins in instrs:
                opn = op_name(ins)

                if opn == "MLIL_RET":
                    if read_taint(ins):
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
                            resolved_names.append(report_name)
                        else:
                            if self.unknown_call_policy != "stop":
                                ret_tainted = True
                                add_assumption(f"external {nm or hex(taddr)} has no model; return conservatively tainted")
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

        # #193 Part 1 honesty: for each registered recv-buffer slot that the fixpoint
        # did NOT correlate to a re-load, emit the deferred "may be missed" caveat --
        # the flow really wasn't followed. Slots that DID correlate already carry their
        # positive note (added at the re-load), so the misleading caveat is suppressed.
        for sk, slot in buffer_slots.items():
            if sk not in correlated_slots:
                add_assumption(slot["pending"])

        return {"reached_return": reached_return, "out_params": frozenset(out_params),
                "findings": findings, "leaves": leaves, "assumptions": assumptions}

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

    def _seed_forward(self, func, ssaf, instrs, sources, taint_node, add_assumption, buffer_slots=None) -> bool:
        if buffer_slots is None:
            buffer_slots = {}
        seeded = False
        for src in sources:
            kind = src.get("kind")
            if kind == "param":
                idx = int(src["index"])
                pv = self._param_var(func, idx)
                if pv is None:
                    raise TaintError(f"parameter {idx} not found on {func.name}")
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
                        f"taint is NOT followed from here. Seed the filled buffer directly "
                        f"(--source var:<buf>) to trace the received data."
                    )
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
                                for r in expr_reads(params[idx]):
                                    if taint_node((var_key(r), getattr(r, "version", None)), var_label(r), c,
                                                  f"source: {callee} arg{idx}", []):
                                        seeded = True
                                # The buffer couldn't be anchored to a stack var or
                                # writable global. If the pointer is itself loaded
                                # from a global/struct slot, register the slot so a
                                # later re-load correlates forward (#193 Part 1); the
                                # helper falls back to today's honest caveat when the
                                # slot can't be named.
                                self._register_indirect_buffer_slot(
                                    ssaf, params[idx], c, callee, idx, buffer_slots, add_assumption)
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
                                    self._register_indirect_buffer_slot(
                                        ssaf, params[idx], c, callee, idx, buffer_slots, add_assumption)
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

    def _make_finding(self, ins, callee, argidx, sink, hit_nodes, why) -> dict[str, Any]:
        path = self._reconstruct_path(hit_nodes[0], why)
        path.append(_instr_dict(ins, reason=f"tainted arg{argidx} reaches {callee}",
                                tainted=[node_label(n, why) for n in hit_nodes]))
        return {
            "sink": {
                "callee": callee,
                "address": hex(int(getattr(ins, "address", 0))),
                "tainted_arg_index": argidx,
                "class": sink.get("class"),
                "detail": sink.get("detail"),
                **({"source_bound": sink["source_bound"]} if sink.get("source_bound") else {}),
                **({"via": sink["via"]} if sink.get("via") else {}),
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
                if pidx is not None:
                    terminal_params[pidx] = v
                    if origin["kind"] == "unresolved":
                        origin = {"kind": "parameter", "index": pidx, "var": var_label(v)}
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
                        ssaf, getattr(src_expr, "src_memory", None), la, set(), 0)
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
                    # State the recovered arg count and the valid 0-based range so
                    # the off-by-one is obvious -- e.g. memcpy(dst, src, len) has 3
                    # args, so the length is index 2, not 3 (#291.4).
                    if max_params == 0:
                        detail = "its call site(s) expose no arguments in the recovered IL"
                    else:
                        detail = (
                            f"its call site(s) expose {max_params} argument(s), so valid "
                            f"indices are 0..{max_params - 1} (arg indices are 0-based)")
                    raise TaintError(
                        f"--sink arg index {idx} is out of range for {callee}: {detail}")
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
            raise TaintError(f"unsupported backward sink kind: {kind}")
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
        return self._find_variable(func, selector)

    def _describe_locator(self, loc: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in loc.items() if k != "_resolved"}


def var_label_of(node: tuple) -> str:
    key, version = node
    if key[0] == "global":
        name = f"glob_{hex(key[1])}"
    elif key[0] == "name":
        name = key[1]
    else:
        name = f"var#{key[1]}"
    return f"{name}#{version}" if version is not None else str(name)


def node_label(node: tuple, why: dict | None = None) -> str:
    """Human-readable label for a taint node.

    Prefers the label captured when the node was first tainted (the live
    variable's ``name#version``, e.g. ``r1#2``). Falls back to deriving one from
    the node key, which can only yield ``var#<identifier>`` for id-keyed
    variables because :func:`var_key` intentionally drops the (unstable) name in
    favor of the stable identifier for set membership -- so the recorded label
    is what keeps JSON/ndjson output as readable as the text renderer.
    """
    if why is not None:
        record = why.get(node)
        if record:
            label = record.get("label")
            if label:
                return str(label)
    return var_label_of(node)


# --------------------------------------------------------------------------
# locator grammar parsing (string -> dict) — shared by CLI/bridge
# --------------------------------------------------------------------------

def parse_locator(spec: str) -> dict[str, Any]:
    """Parse a source/sink locator string into a dict.

    Grammar (MVP):
      param:<n>            -> {"kind":"param","index":n}
      var:<selector>       -> {"kind":"var","selector":...}
      ret:<callee>         -> {"kind":"ret","callee":...}
      arg:<callee>:<n>     -> {"kind":"arg","callee":...,"index":n}
    """
    if not spec:
        raise TaintError("empty locator")
    head, _, rest = spec.partition(":")
    if head == "param":
        return {"kind": "param", "index": _locator_index(rest, "param")}
    if head == "var":
        if not rest:
            raise TaintError("var: locator needs a selector")
        return {"kind": "var", "selector": rest}
    if head == "ret":
        if not rest:
            raise TaintError("ret: locator needs a callee")
        return {"kind": "ret", "callee": rest}
    if head == "arg":
        # Split at the LAST colon so a C++ qualified callee keeps its own colons:
        # "arg:Ns::method:1" -> callee="Ns::method", n="1". partition(":") split at
        # the FIRST colon and mis-parsed callee="Ns" (#98).
        callee, sep, n = rest.rpartition(":")
        if not sep or not callee or not n:
            raise TaintError("arg: locator must be arg:<callee>:<n>")
        return {"kind": "arg", "callee": callee, "index": _locator_index(n, f"arg:{callee}")}
    if head in ("call", "model"):
        # call:<callee> / model:<callee> -- seed ALL outputs the callee's taint
        # model declares (return value AND output-pointer buffers), so a
        # receive-style API like read/recv that writes its tainted bytes through
        # an output-pointer arg is no longer a silent all-clear (#157).
        if not rest:
            raise TaintError(f"{head}: locator needs a callee")
        return {"kind": "call", "callee": rest}
    raise TaintError(f"unknown locator kind: {head!r} (use param:/var:/ret:/arg:/call:/model:)")


def _locator_index(text: str, what: str) -> int:
    """Parse a 0-based argument/parameter index, rejecting negatives.

    A negative index would otherwise pass ``idx < len(params)`` and silently
    seed ``params[-1]`` (the *last* argument) -- a confidently-wrong slice.
    """
    try:
        n = int(text)
    except (TypeError, ValueError):
        raise TaintError(f"{what} index must be an integer, got {text!r}")
    if n < 0:
        raise TaintError(
            f"{what} index must be >= 0 (argument indices are 0-based), got {n}")
    return n


def format_locator(loc: dict[str, Any]) -> str:
    """Render a locator dict back to its ``kind:...`` string for diagnostics."""
    kind = loc.get("kind")
    if kind == "arg":
        return f"arg:{loc.get('callee')}:{loc.get('index')}"
    if kind == "param":
        return f"param:{loc.get('index')}"
    if kind == "var":
        return f"var:{loc.get('selector')}"
    if kind == "ret":
        return f"ret:{loc.get('callee')}"
    return str(kind)
