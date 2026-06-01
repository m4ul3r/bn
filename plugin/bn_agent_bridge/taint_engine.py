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
    "may-analysis (interprocedural, summary-based, depth-bounded); coarse memory "
    "and unresolved indirect/external calls are surfaced as assumptions/leaves; "
    "NOT a proof of reachability"
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
        max_iters: int = 256,
        max_depth: int = 64,
    ):
        self.bv = bv
        self.models = models
        self._find_variable = find_variable
        self.unknown_call_policy = unknown_call_policy
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
                sink_filter: set[str] | None = None, max_depth: int = 8) -> dict[str, Any]:
        # Per-call analysis state (reset each public call):
        self._cache: dict[tuple, Any] = {}          # (func_start, frozenset(params)) -> summary
        self._funcs_visited: set[int] = set()
        self._max_depth_seen = 0
        self._truncated = False

        sub = self._run_forward(func, sources, depth=0, max_depth=max_depth, top=True)
        return {
            "direction": "forward",
            "function": {"name": str(func.name), "address": hex(int(func.start))},
            "sources": [self._describe_locator(s) for s in sources],
            "reached_sinks": sub["findings"],
            "leaves": sub["leaves"],
            "assumptions": sub["assumptions"],
            "stats": {
                "functions_visited": len(self._funcs_visited),
                "max_depth": self._max_depth_seen,
                "sinks": len(sub["findings"]),
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
                return {"reached_return": True, "out_params": set(), "findings": [], "leaves": [],
                        "assumptions": [f"recursion cycle at {callee.name}; return conservatively tainted"]}
            return cached
        self._cache[key] = None  # mark in-progress (cycle guard)
        locators = [{"kind": "param", "index": i} for i in sorted(param_set)]
        try:
            sub = self._run_forward(callee, locators, depth, max_depth, top=False)
        except TaintError as exc:
            sub = {"reached_return": True, "out_params": set(), "findings": [], "leaves": [],
                   "assumptions": [f"could not analyze {callee.name}: {exc}; return conservatively tainted"]}
        self._cache[key] = sub
        return sub

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
                pv = self._pointee_var(ssaf, expr)
                if pv is not None and (var_key(pv), None) in tainted:
                    hit.append((var_key(pv), None))
            return hit

        def cons_return(ins: Any, reason: str) -> bool:
            done = False
            for w in ssa_writes(ins):
                node = (var_key(w), getattr(w, "version", None))
                if taint_node(node, var_label(w), ins, reason, []):
                    done = True
            return done

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

                    # 1) model-driven sink detection
                    if model and model.get("sink") is not None:
                        sink = model["sink"]
                        for argidx in sink.get("tainted_args", []) or []:
                            if argidx < len(params):
                                ht = arg_taint(params[argidx])
                                if ht:
                                    sig = (int(getattr(ins, "address", 0)), argidx)
                                    if sig not in recorded_sinks:
                                        recorded_sinks.add(sig)
                                        findings.append(self._make_finding(ins, mkey or name, argidx, sink, ht, why))

                    # 2) model-driven propagation
                    if model and model.get("propagates"):
                        for rule in model["propagates"]:
                            if self._token_tainted(ssaf, ins, params, rule.get("from"), tainted):
                                if self._apply_to_token(ssaf, ins, params, rule.get("to"), taint_node, name or "?"):
                                    changed = True
                        continue

                    if model is not None:
                        continue  # modeled (e.g. source-only); body intentionally not descended

                    # 3) no model: tainted argument positions at this callsite
                    tainted_args = {i: arg_taint(p) for i, p in enumerate(params) if arg_taint(p)}
                    if not tainted_args:
                        continue

                    if target is None:
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

                    callee_fn = self.bv.get_function_at(target) if hasattr(self.bv, "get_function_at") else None
                    if self._is_internal(callee_fn):
                        n_params = len(list(getattr(callee_fn, "parameter_vars", []) or []))
                        valid = frozenset(i for i in tainted_args if i < n_params)
                        if not valid:
                            if cons_return(ins, f"return of variadic/over-arg call {callee_fn.name} (conservative)"):
                                changed = True
                            add_assumption(f"tainted args to {callee_fn.name} fall beyond its parameters; conservative")
                            continue
                        if depth + 1 > max_depth:
                            self._truncated = True
                            if cons_return(ins, f"max-depth: return of {callee_fn.name} (conservative)"):
                                changed = True
                            add_assumption(
                                f"max interprocedural depth {max_depth} reached at {callee_fn.name}; not descended")
                            continue
                        # 4) interprocedural: descend with a cached summary, bubble findings up
                        sub = self._summarize(callee_fn, valid, depth + 1, max_depth)
                        first_hit = tainted_args[sorted(valid)[0]][0]
                        prefix = self._reconstruct_path(first_hit, why)
                        prefix.append(_instr_dict(
                            ins, reason=f"calls {callee_fn.name} with tainted arg(s) {sorted(valid)}",
                            tainted=[var_label_of(first_hit)]))
                        for f in sub["findings"]:
                            findings.append({"sink": f["sink"], "path": prefix + f["path"]})
                        for lf in sub["leaves"]:
                            if lf not in leaves:
                                leaves.append(lf)
                        for a in sub["assumptions"]:
                            add_assumption(a)
                        if sub["reached_return"]:
                            for w in ssa_writes(ins):
                                node = (var_key(w), getattr(w, "version", None))
                                if taint_node(node, var_label(w), ins,
                                              f"return of {callee_fn.name} propagates taint", [first_hit]):
                                    changed = True
                    else:
                        # external with no model
                        if self.unknown_call_policy != "stop":
                            if cons_return(ins, f"return of unmodeled external {name or hex(target)} (conservative)"):
                                changed = True
                            add_assumption(f"external {name or hex(target)} has no model; return conservatively tainted")
                    continue

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
                        pv = self._pointee_var(ssaf, getattr(ins, "dest", None))
                        if pv is not None:
                            node = (var_key(pv), None)
                            if taint_node(node, var_label(pv), ins, "store into tainted buffer (memory_approx)", reads):
                                changed = True
                                add_assumption("memory aliasing modeled coarsely via pointer-base/AddressOf (memory_approx)")
            if not changed:
                break

        return {"reached_return": reached_return, "out_params": set(),
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

    def _token_tainted(self, ssaf: Any, ins: Any, params: list[Any], tok: str | None, tainted: set) -> bool:
        if not tok:
            return False
        if tok.startswith("*arg:") or tok.startswith("arg:"):
            pointee = tok.startswith("*arg:")
            idx = int(tok.split("arg:", 1)[1])
            if idx >= len(params):
                return False
            if pointee:
                pv = self._pointee_var(ssaf, params[idx])
                return pv is not None and (var_key(pv), None) in tainted
            for r in expr_reads(params[idx]):
                if (var_key(r), getattr(r, "version", None)) in tainted or (var_key(r), None) in tainted:
                    return True
            return False
        return False

    def _apply_to_token(self, ssaf: Any, ins: Any, params: list[Any], tok: str | None, taint_node, callee: str) -> bool:
        if not tok:
            return False
        if tok == "ret" or tok == "*ret":
            done = False
            for w in ssa_writes(ins):
                node = (var_key(w), getattr(w, "version", None))
                if taint_node(node, var_label(w), ins, f"return of {callee} (model propagate)", []):
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
                                      f"buffer written by {callee} (model propagate)", [])
                return False
            done = False
            for r in expr_reads(params[idx]):
                node = (var_key(r), getattr(r, "version", None))
                if taint_node(node, var_label(r), ins, f"output of {callee} (model propagate)", []):
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

    def backward(self, func: Any, sinks: list[dict[str, Any]]) -> dict[str, Any]:
        ssaf = self._ssa_func(func)
        instrs = self._instrs(ssaf)
        slices: list[dict[str, Any]] = []
        assumptions: list[str] = []
        leaves: list[dict[str, Any]] = []

        for sink in sinks:
            seeds = self._seed_backward(func, ssaf, instrs, sink)
            for seed_var, sink_ins in seeds:
                steps: list[dict[str, Any]] = []
                origin = {"kind": "unresolved"}
                visited: set = set()

                def walk(v, depth):
                    nonlocal origin
                    if depth > self.max_depth:
                        return
                    key = (var_key(v), getattr(v, "version", None))
                    if key in visited:
                        return
                    visited.add(key)
                    try:
                        d = ssaf.get_ssa_var_definition(v)
                    except Exception:
                        d = None
                    if d is None:
                        origin = {"kind": "parameter_or_entry", "var": var_label(v)}
                        return
                    if self._is_call(d):
                        target = const_target(getattr(d, "dest", None))
                        name = self._callee_name(target)
                        mkey, model = lookup_model(self.models, name)
                        steps.append(_instr_dict(d, reason=f"defined by call to {name or 'indirect'}"))
                        if model and model.get("sources"):
                            origin = {"kind": "source", "callee": mkey or name}
                        elif target is None:
                            origin = {"kind": "indirect_call", "var": var_label(v)}
                            leaf = {"kind": "indirect_call_unresolved",
                                    "address": hex(int(getattr(d, "address", 0))),
                                    "il_text": str(d)}
                            if leaf not in leaves:
                                leaves.append(leaf)
                        else:
                            origin = {"kind": "call", "callee": name}
                        return
                    steps.append(_instr_dict(d, reason="definition"))
                    for r in ssa_reads(d):
                        walk(r, depth + 1)

                walk(seed_var, 0)
                steps.reverse()
                slices.append({
                    "sink": {
                        "callee": sink.get("callee"),
                        "address": hex(int(getattr(sink_ins, "address", 0))),
                        "seed": var_label(seed_var),
                    },
                    "origin": origin,
                    "slice": steps,
                })
        return {
            "direction": "backward",
            "function": {"name": str(func.name), "address": hex(int(func.start))},
            "sinks": [self._describe_locator(s) for s in sinks],
            "slices": slices,
            "leaves": leaves,
            "assumptions": assumptions,
            "soundness": SOUNDNESS,
        }

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
