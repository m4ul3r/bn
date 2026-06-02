"""Intraprocedural taint engine over Binary Ninja MLIL-SSA.

This module is intentionally free of any ``binaryninja`` import: it operates on
whatever MLIL-SSA objects the bridge hands it (functions, instructions,
SSAVariables, PossibleValueSets). That keeps it unit-testable against the same
synthetic IL fakes the bridge tests use.

Scope (MVP): single-function forward propagation and single-function backward
slicing. Interprocedural stepping, indirect-call resolution and precise
memory-SSA aliasing are explicitly deferred — every place the analysis is
coarse or stops is surfaced in ``assumptions``/``leaves`` and the output always
carries a ``soundness`` disclaimer. We never silently drop an edge.

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


class TaintError(RuntimeError):
    """User-facing taint configuration/resolution error."""


# --------------------------------------------------------------------------
# model database
# --------------------------------------------------------------------------

def load_models(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the merged function-model DB: builtin <- user override <- extra."""
    models: dict[str, Any] = {}
    try:
        raw = json.loads(_BUILTIN_MODELS.read_text(encoding="utf-8"))
        models.update(raw.get("models") or {})
    except Exception:  # pragma: no cover - builtin should always parse
        pass
    if taint_models_path is not None:
        try:
            override_path = taint_models_path()
            if override_path.exists():
                raw = json.loads(override_path.read_text(encoding="utf-8"))
                # accept either {"models": {...}} or a bare {name: model} map
                models.update(raw.get("models") if isinstance(raw, dict) and "models" in raw else raw)
        except Exception:
            pass
    if extra:
        models.update(extra)
    return models


def lookup_model(models: dict[str, Any], name: str | None) -> tuple[str | None, dict[str, Any] | None]:
    """Match a (possibly decorated) symbol name against the model DB.

    Tries the raw name, then the part before ``@`` (``memcpy@plt`` ->
    ``memcpy``), then with leading underscores stripped.
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
    for cand in candidates:
        if cand in models:
            return cand, models[cand]
    return None, None


# --------------------------------------------------------------------------
# small IL helpers (defensive getattr style, matching bridge.py)
# --------------------------------------------------------------------------

def op_name(item: Any) -> str:
    operation = getattr(item, "operation", None)
    name = getattr(operation, "name", None)
    return str(name) if name else str(operation)


def is_ssa_var(v: Any) -> bool:
    return hasattr(v, "version") and hasattr(v, "var")


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


def const_target(expr: Any) -> int | None:
    """Constant call destination (direct call) or None (indirect)."""
    if expr is None:
        return None
    if "CONST" not in op_name(expr):
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
        fn = None
        try:
            fn = self.bv.get_function_at(addr)
        except Exception:
            fn = None
        if fn is not None and getattr(fn, "name", None):
            return str(fn.name)
        try:
            sym = self.bv.get_symbol_at(addr)
        except Exception:
            sym = None
        if sym is not None and getattr(sym, "name", None):
            return str(sym.name)
        return None

    def _is_call(self, ins: Any) -> bool:
        return "CALL" in op_name(ins) or "TAILCALL" in op_name(ins)

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
        """If *expr* is a pointer to a variable that has ANY tainted SSA version,
        return a representative tainted node, else None.

        Version-agnostic on purpose: a buffer/struct slot is written under one
        SSA/memory version (e.g. an MLIL_SET_VAR_ALIASED store -> var#85) but a
        later ``&var`` references the whole variable (version None). Matching only
        the exact version would miss the pointer carrying that taint.
        """
        pv = self._pointee_var(ssaf, expr)
        if pv is None:
            return None
        k = var_key(pv)
        if (k, None) in tainted:
            return (k, None)
        for node in tainted:
            if node[0] == k:
                return node
        return None

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

    def _find_callsites(self, instrs: list[Any], callee: str) -> list[Any]:
        hits = []
        for ins in instrs:
            if not self._is_call(ins):
                continue
            target = const_target(getattr(ins, "dest", None))
            name = self._callee_name(target)
            matched, _ = lookup_model({callee: True}, name) if name else (None, None)
            if (name and (name == callee or name.split("@", 1)[0].lstrip("_") == callee.lstrip("_"))) or matched:
                hits.append(ins)
        return hits

    # -- forward ----------------------------------------------------------

    def forward(self, func: Any, sources: list[dict[str, Any]], *,
                enabled_sink_classes: set[str] | None = None, max_depth: int = 8) -> dict[str, Any]:
        # Per-call analysis state (reset each public call):
        # optional-sink classes the caller opted into (e.g. file_write); a sink
        # marked "optional" in the model DB fires only if its class is in here.
        self._enabled_sink_classes: set[str] = set(enabled_sink_classes or ())
        self._cache: dict[tuple, Any] = {}          # (func_start, frozenset(params)) -> summary
        self._funcs_visited: set[int] = set()
        self._max_depth_seen = 0
        self._truncated = False

        sub = self._run_forward(func, sources, depth=0, max_depth=max_depth, top=True)
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
                "truncated": self._truncated,
            },
            "soundness": SOUNDNESS,
        }

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
                        "assumptions": [f"recursion cycle at {callee.name}; return conservatively tainted"]}
            return cached
        self._cache[key] = None  # mark in-progress (cycle guard)
        locators = [{"kind": "param", "index": i} for i in sorted(param_set)]
        try:
            sub = self._run_forward(callee, locators, depth, max_depth, top=False)
        except TaintError as exc:
            sub = {"reached_return": True, "out_params": frozenset(), "findings": [], "leaves": [],
                   "assumptions": [f"could not analyze {callee.name}: {exc}; return conservatively tainted"]}
        self._cache[key] = sub
        return sub

    def _descend(self, ins: Any, callee_fn: Any, tainted_args: dict, why: dict,
                 depth: int, max_depth: int, *, via: str | None = None) -> dict[str, Any]:
        """Recurse into a (direct or resolved-indirect) internal callee and return
        its findings with a caller-side path prefix prepended, plus whether it
        propagates taint to its return."""
        n_params = len(list(getattr(callee_fn, "parameter_vars", []) or []))
        valid = frozenset(i for i in tainted_args if i < n_params)
        out: dict[str, Any] = {"findings": [], "reached_return": False, "leaves": [],
                               "assumptions": [], "out_params": frozenset()}
        if not valid:
            out["reached_return"] = True
            out["assumptions"].append(f"tainted args to {callee_fn.name} fall beyond its parameters; conservative")
            return out
        if depth + 1 > max_depth:
            self._truncated = True
            out["reached_return"] = True
            out["assumptions"].append(
                f"max interprocedural depth {max_depth} reached at {callee_fn.name}; not descended")
            return out
        sub = self._summarize(callee_fn, valid, depth + 1, max_depth)
        first_hit = tainted_args[sorted(valid)[0]][0]
        prefix = self._reconstruct_path(first_hit, why)
        note = f"calls {callee_fn.name} with tainted arg(s) {sorted(valid)}"
        if via:
            note = f"[{via}-resolved] " + note
        prefix.append(_instr_dict(ins, reason=note, tainted=[var_label_of(first_hit)]))
        for f in sub["findings"]:
            out["findings"].append({"sink": f["sink"], "path": prefix + f["path"]})
        out["leaves"] = list(sub["leaves"])
        out["assumptions"] = list(sub["assumptions"])
        out["reached_return"] = sub["reached_return"]
        out["out_params"] = sub.get("out_params", frozenset())
        return out

    def _call_targets_from_pvs(self, pvs: Any) -> list[int]:
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

        seeded = self._seed_forward(func, ssaf, instrs, locators, taint_node, add_assumption)
        if not seeded:
            if top:
                raise TaintError("no taint sources resolved; check --source locator")
            return {"reached_return": False, "out_params": set(), "findings": [],
                    "leaves": [], "assumptions": []}

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
                                findings.append(self._make_finding(ins, mkey or name, argidx, sink, ht, why))
            for rule in model.get("propagates") or []:
                to = rule.get("to")
                hit = self._token_hit_node(ssaf, params, rule.get("from"), tainted)
                if hit is not None:
                    if self._apply_to_token(ssaf, ins, params, to, taint_node, name or "?", parents=[hit]):
                        changed = True
                    if to and to.startswith("*arg:"):
                        k = int(to.split("arg:", 1)[1])
                        if k < len(params):
                            propagated.add(k)
            # variadic propagation: every tainted vararg (from first_index on) flows
            # into the dest buffer and is itself reportable. Uses the actual call
            # params, so no format-string parsing is needed; arg_taint already covers
            # both a tainted scalar (%d) and a pointer to a tainted buffer (%s).
            va = model.get("varargs")
            if va is not None:
                base = int(va.get("first_index", 0))
                vto = va.get("to")
                vsink = model.get("sink") if va.get("sink") else None
                for i in range(max(base, 0), len(params)):
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
                    target = const_target(getattr(ins, "dest", None))
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
                        add_assumption(f"indirect call at {leaf['address']} reached by taint; target unresolved")
                        continue

                    ret_tainted = False
                    descend_outparams: set[int] = set()
                    resolved_names: list[str] = []
                    for taddr in candidates:
                        cfn = self.bv.get_function_at(taddr) if hasattr(self.bv, "get_function_at") else None
                        nm = self._callee_name(taddr)
                        mk, md = lookup_model(self.models, nm)
                        if md is not None:
                            # resolved target is a modeled external
                            mchanged, _ = apply_model(ins, params, md, mk, nm, site_taddr=taddr)
                            if mchanged:
                                changed = True
                            resolved_names.append(nm or hex(taddr))
                        elif self._is_internal(cfn):
                            d = self._descend(ins, cfn, tainted_args, why, depth, max_depth, via=via)
                            findings.extend(d["findings"])
                            for lf in d["leaves"]:
                                if lf not in leaves:
                                    leaves.append(lf)
                            for a in d["assumptions"]:
                                add_assumption(a)
                            ret_tainted = ret_tainted or d["reached_return"]
                            descend_outparams |= set(d.get("out_params") or ())
                            resolved_names.append(str(cfn.name))
                        else:
                            if self.unknown_call_policy != "stop":
                                ret_tainted = True
                                add_assumption(f"external {nm or hex(taddr)} has no model; return conservatively tainted")
                            resolved_names.append(nm or hex(taddr))

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
                        if msrc is not None:
                            for w in ssa_writes(ins):
                                node = (var_key(w), getattr(w, "version", None))
                                if taint_node(node, var_label(w), ins,
                                              "loads value stored through a tainted pointer (mem-SSA)", [msrc]):
                                    changed = True
                            add_assumption("heap/pointer load resolved via memory-SSA store correlation")

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
            if not changed:
                break

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
                pv = self._pointee_var(ssaf, params[idx])
                if pv is not None:
                    return taint_node((var_key(pv), None), var_label(pv), ins,
                                      f"buffer written by {callee} (model propagate)", parents)
                return False
            done = False
            for r in expr_reads(params[idx]):
                node = (var_key(r), getattr(r, "version", None))
                if taint_node(node, var_label(r), ins, f"output of {callee} (model propagate)", parents):
                    done = True
            return done
        return False

    def _seed_forward(self, func, ssaf, instrs, sources, taint_node, add_assumption) -> bool:
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
                calls = self._find_callsites(instrs, callee)
                if not calls:
                    raise TaintError(f"no callsite of {callee} found in {func.name}")
                if len(calls) > 1:
                    add_assumption(f"{len(calls)} callsites of {callee}; seeded from all")
                for c in calls:
                    if kind == "ret":
                        for w in ssa_writes(c):
                            if taint_node((var_key(w), getattr(w, "version", None)), var_label(w), c,
                                          f"source: return of {callee}", []):
                                seeded = True
                    else:  # arg:<callee>:<n> -> the buffer that arg n points at
                        idx = int(src["index"])
                        params = self._call_params(c)
                        if idx < len(params):
                            pv = self._pointee_var(ssaf, params[idx])
                            if pv is not None:
                                if taint_node((var_key(pv), None), str(getattr(pv, "name", pv)), c,
                                              f"source: {callee} fills arg{idx} buffer", []):
                                    seeded = True
                            else:
                                for r in expr_reads(params[idx]):
                                    if taint_node((var_key(r), getattr(r, "version", None)), var_label(r), c,
                                                  f"source: {callee} arg{idx}", []):
                                        seeded = True
            else:
                raise TaintError(f"unknown source kind: {kind}")
        return seeded

    def _make_finding(self, ins, callee, argidx, sink, hit_nodes, why) -> dict[str, Any]:
        path = self._reconstruct_path(hit_nodes[0], why)
        path.append(_instr_dict(ins, reason=f"tainted arg{argidx} reaches {callee}",
                                tainted=[var_label_of(n) for n in hit_nodes]))
        return {
            "sink": {
                "callee": callee,
                "address": hex(int(getattr(ins, "address", 0))),
                "tainted_arg_index": argidx,
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

        for sink in sinks:
            ssaf = self._ssa_func(func)
            instrs = self._instrs(ssaf)
            for seed_var, sink_ins in self._seed_backward(func, ssaf, instrs, sink):
                for sl in self._backward_slice(func, seed_var, 0, max_depth, set()):
                    slices.append({
                        "sink": {
                            "callee": sink.get("callee"),
                            "address": hex(int(getattr(sink_ins, "address", 0))),
                            "seed": var_label(seed_var),
                        },
                        "origin": sl["origin"],
                        "crossed_functions": sl["crossed"],
                        "slice": sl["steps"],
                    })
        return {
            "direction": "backward",
            "function": {"name": str(func.name), "address": hex(int(func.start))},
            "sinks": [self._describe_locator(s) for s in sinks],
            "slices": slices,
            "leaves": self._bw_leaves,
            "assumptions": self._bw_assumptions,
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

        def walk(v, d):
            nonlocal origin
            if d > self.max_depth:
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
            for r in ssa_reads(defn):
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
            for c in self._find_callsites(instrs, callee):
                params = self._call_params(c)
                if idx < len(params):
                    for r in expr_reads(params[idx]):
                        out.append((r, c))
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
    name = key[1] if key[0] == "name" else f"var#{key[1]}"
    return f"{name}#{version}" if version is not None else str(name)


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
        return {"kind": "param", "index": int(rest)}
    if head == "var":
        if not rest:
            raise TaintError("var: locator needs a selector")
        return {"kind": "var", "selector": rest}
    if head == "ret":
        if not rest:
            raise TaintError("ret: locator needs a callee")
        return {"kind": "ret", "callee": rest}
    if head == "arg":
        callee, _, n = rest.partition(":")
        if not callee or not n:
            raise TaintError("arg: locator must be arg:<callee>:<n>")
        return {"kind": "arg", "callee": callee, "index": int(n)}
    raise TaintError(f"unknown locator kind: {head!r} (use param:/var:/ret:/arg:)")
