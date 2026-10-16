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
  * ``_shared`` -- module-free helpers (``_parse_address``, ``OperationFailure``);
  * ``read_listing`` -- ``_analysis_state_fields``, the ONE definition of the
    ``{analysis_state, partial}`` disclosure every read op attaches (#820/#811).
    Imported, never re-derived, so the taint surface cannot drift into a second
    spelling of "this answer may be incomplete".

Import direction is one-way: this module imports ``il_format``, ``vars``,
``taint_engine``, ``_shared`` and ``read_listing`` (plus stdlib + binaryninja).
It NEVER imports ``bridge`` or ``seam`` -- those import THIS module one-way
(design spec §3.2); ``read_listing`` does not import this module either, so the
added edge stays acyclic.
"""
from __future__ import annotations

import re
import shlex
from typing import Any

try:
    import binaryninja as bn  # noqa: F401  (kept for parity with sibling read_* modules)
except ModuleNotFoundError:  # importable without the Binary Ninja runtime (tests, tooling)
    bn = None  # type: ignore[assignment]
try:
    from binaryninja import SSAVariable
except ModuleNotFoundError:  # importable without the Binary Ninja runtime (tests, tooling)
    class SSAVariable:  # sentinel: keeps isinstance(...) valid (and False) without BN
        pass

from . import il_format
from . import taint_engine as _taint
from . import vars as vars_mod
from ._shared import OperationFailure, _parse_address, _validate_bool, is_imported_function
from .bridge_state import require_analysis
from .read_taint_models import build_catalog
from .read_listing import _analysis_state_fields
from .target_hint import positional_selector


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
        params.get("user_models"), user_models_path=params.get("user_models_path"),
        user_models_via=params.get("user_models_via"))

    def _find_variable(fn, sel):
        var, _is_param = vars_mod._find_variable_selector(fn, sel)
        return var

    # #812: `max_iters` is an engine ctor knob, now reachable from the CLI
    # (`taint forward --max-iters`) so the "raise --max-iters" remediation a
    # fixpoint-truncated run prints is something the user can actually do.
    # Absent from the request -> the engine's own default, so every existing
    # caller and the backward path are unaffected.
    _engine_kwargs: dict[str, Any] = {}
    if params.get("max_iters") is not None:
        _engine_kwargs["max_iters"] = int(params["max_iters"])
    # #1041 item 3: the map is validated when the engine is built, so a bad
    # entry is refused before the walk, naming the file the CLI read it from;
    # a target spelled as a name resolves exactly as `-f` does.
    _map_path = params.get("resolve_map_path")
    # Present-but-null is a supplied document (refused by name), not an
    # omission (#1050 review round 2): pass the map only when the request has one.
    if "resolve_map" in params:
        _engine_kwargs["resolve_map"] = params["resolve_map"]
    try:
        engine = _taint.TaintEngine(
            bv,
            models,
            find_variable=_find_variable,
            unknown_call_policy=str(params.get("unknown_call", "conservative")),
            resolve_map_origin=(f"--resolve-map {_map_path}" if _map_path
                                else "--resolve-map"),
            find_function=lambda name: int(ctx._find_function(bv, name).start),
            mangled=_validate_bool(params.get("mangled"), label="mangled", default=False),
            **_engine_kwargs,
        )
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
        # #811: disclose whether this answer was computed over a quick-loaded
        # (partial) view, using the SAME {analysis_state, partial} shape every
        # other read op attaches. `require_analysis` above refuses a quick view
        # outright, so today this always reports "full" from the taint path --
        # it is here so the contract is uniform and a future quick-tolerant
        # taint mode cannot ship a silent partial answer. The per-callee
        # "body could not be read" hole is a DIFFERENT and narrower condition
        # and is reported separately in stats.analysis_incomplete.
        result.update(_analysis_state_fields(bv))
    return result


def _taint_models_op(ctx, selector, params: dict[str, Any]):
    """`taint models`: dump the model catalog; with a target, annotate which
    modeled symbols are present in the binary (+ enriched callsite rows). The
    result is always a CATALOG, not a findings list -- see ``build_catalog``'s
    top-level ``presence_catalog``/``is_finding`` markers (#555)."""
    present_only = bool(params.get("present"))
    # #958: `--all-sites` shows the non-audit rows (import stubs, self stubs,
    # constant-argument inline builtins) that are otherwise only counted.
    all_sites = bool(params.get("all_sites"))
    want_callsites = bool(params.get("callsites")) or all_sites
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
        params.get("user_models"), user_models_path=params.get("user_models_path"),
        user_models_via=params.get("user_models_via"))
    catalog["items"] = ([{"role": "source", **s} for s in catalog["sources"]]
                        + [{"role": "sink", **e}
                           for lst in catalog["sinks_by_class"].values() for e in lst]
                        + [{"role": "propagator", **p} for p in catalog["propagators"]])
    if selector is not None:
        bv = ctx._resolve_view(selector)
        present_keys, info = _present_models(bv, models, want_callsites, all_sites=all_sites)
        _annotate_presence(catalog, present_keys, info, present_only)
        # #811: presence is computed by walking this view's functions/symbols, so
        # on a quick-loaded view a modeled sink can be reported ABSENT merely
        # because its caller was never analysed -- a false all-clear in catalog
        # form. Unlike `_taint_op` this handler has no `require_analysis` gate
        # (adding one would break `taint models --present` on every quick view),
        # so disclose the state instead and let the caller judge.
        catalog.update(_analysis_state_fields(bv))
    return catalog


def _containing_fn(bv, addr: int):
    """Function whose body contains ``addr`` (best-effort, BN-shape-tolerant so the
    unit fakes without ``get_functions_containing`` degrade to None, not a crash)."""
    try:
        fns = list(getattr(bv, "get_functions_containing")(int(addr)))
        if fns:
            return fns[0]
    except Exception:
        pass
    try:
        return getattr(bv, "get_function_at")(int(addr))
    except Exception:
        return None


# Linker-generated import stubs live in these sections; a function there is a
# stub whatever BN named or typed it (#958).
_STUB_SECTIONS = frozenset({".plt", ".plt.sec", ".plt.got", ".iplt", "__stubs",
                            "__auth_stubs", ".MIPS.stubs"})


def _is_import_stub_fn(bv, fn) -> bool:
    """True if ``fn`` is an import stub (a PLT entry / linker veneer), NOT merely
    a function whose body is one tail call.

    #958: BN's ``is_thunk`` flag and a single-tailcall body are also true of an
    ordinary application wrapper (`void log_copy(d, s, n) { memcpy(d, s, n); }`
    compiled to `b memcpy`), whose tail branch IS an audit site; labelling it
    `import_thunk` hid a real caller. A stub is recognised by what the linker
    makes of it instead: an import symbol, a stub section, or a veneer name."""
    if is_imported_function(fn):
        return True
    name = str(getattr(fn, "name", "") or "")
    if name.endswith("_veneer"):
        return True
    try:
        sections = bv.get_sections_at(int(getattr(fn, "start", 0)))
    except Exception:
        return False
    return any(str(getattr(s, "name", "")) in _STUB_SECTIONS for s in sections or [])


def _classify_callsite(bv, addr: int, sink_key: str, models) -> tuple[str | None, str]:
    """(containing-function-name, kind) for a present-model callsite.

    ``kind`` is one of ``app_caller`` (a real application caller body),
    ``import_thunk`` (an import stub -- #560; its forwarding jump is not a
    caller, so `_present_models` drops the row, #958), or ``self_stub`` (the
    modeled symbol's own body referencing itself, e.g. a self-tailcall stub).
    Self-stub sites are NON-audit: they inflate a sink's callsite count
    without being places an analyst would triage."""
    fn = _containing_fn(bv, addr)
    if fn is None:
        return None, "app_caller"
    fname = str(getattr(fn, "name", "")) or None
    if _is_import_stub_fn(bv, fn):
        return fname, "import_thunk"
    ck, _ = _taint.lookup_model(models, fname or "")
    if ck is not None and ck == sink_key:
        return fname, "self_stub"
    return fname, "app_caller"


_CALL_OPS = frozenset({"LLIL_CALL", "LLIL_CALL_STACK_ADJUST", "LLIL_TAILCALL", "LLIL_JUMP",
                       "MLIL_CALL", "MLIL_TAILCALL", "MLIL_CALL_UNTYPED",
                       "MLIL_TAILCALL_UNTYPED"})
# RegisterValueType names whose `.value` is the branch target itself.
_TARGET_VALUE_TYPES = frozenset({"ConstantValue", "ConstantPointerValue",
                                 "ImportedAddressValue", "ExternalPointerValue"})


def _branch_target(dest) -> int | None:
    """The address a call/jump destination expression resolves to: a constant,
    a load from a constant slot (`call [rip+GOT]`, a PLT `jmp [GOT]`), or the
    value BN's dataflow pins the operand to. None for anything else."""
    if dest is None:
        return None
    op = il_format._il_op_name(dest)
    if "CONST" in op or "IMPORT" in op:
        c = getattr(dest, "constant", None)
        return int(c) if isinstance(c, int) else None
    if "LOAD" in op:
        src = getattr(dest, "src", None)
        if "CONST" in il_format._il_op_name(src):
            c = getattr(src, "constant", None)
            return int(c) if isinstance(c, int) else None
    try:
        value = getattr(dest, "value", None)
        if getattr(getattr(value, "type", None), "name", None) in _TARGET_VALUE_TYPES:
            return int(value.value)
    except Exception:
        return None
    return None


def _callsite_shape(fn, addr: int, targets: set[int], flagged_args: list[int]) -> str | None:
    """How the instruction at *addr* (a code reference to the modeled symbol)
    uses that symbol (#958):

    * ``"call"``   -- a call or tail branch whose target is one of *targets*;
    * ``"inline"`` -- not a call in LLIL, but BN's MLIL synthesised an inline
      builtin call there (`str xzr, ...` runs lifted to `__builtin_memset`) and
      every argument the model flags is a constant, so no taint can arm it;
    * ``"data"``   -- anything else: the `adrp` that loads the GOT page the
      import's slot sits on, the stub's own `ldr`/`add` of its slot;
    * ``None``     -- the view cannot answer (no function, no IL API), so the
      caller keeps the row rather than dropping a site it could not judge."""
    get_llil = getattr(fn, "get_llil_at", None) if fn is not None else None
    if not callable(get_llil):
        return None
    try:
        llil = get_llil(int(addr))
    except Exception:
        return None
    if llil is None:
        return None
    if il_format._il_op_name(llil) in _CALL_OPS:
        return "call" if _branch_target(getattr(llil, "dest", None)) in targets else "data"
    try:
        mlil = getattr(llil, "mlil", None)
    except Exception:
        mlil = None
    if mlil is not None and il_format._il_op_name(mlil) in _CALL_OPS \
            and _branch_target(getattr(mlil, "dest", None)) in targets:
        params = list(getattr(mlil, "params", []) or [])
        flagged = [params[i] for i in flagged_args if 0 <= i < len(params)]
        if flagged and all("CONST" in il_format._il_op_name(p) for p in flagged):
            return "inline"
        return "call"
    return "data"


def _flagged_args(model: dict[str, Any] | None) -> list[int]:
    """The argument indices a sink model arms (tainted_args + len_arg)."""
    sink = (model or {}).get("sink") or {}
    idxs = [int(a) for a in (sink.get("tainted_args") or []) if isinstance(a, int)]
    if isinstance(sink.get("len_arg"), int) and sink["len_arg"] not in idxs:
        idxs.append(sink["len_arg"])
    return idxs


def _pick_resolved(key: str, raws: list[str]) -> str | None:
    """The canonical binary spelling for xrefs/callsites (#556): prefer the exact
    model key, then an undecorated spelling (no ``@plt``), then the shortest."""
    if not raws:
        return None
    if key in raws:
        return key
    plain = [r for r in raws if "@" not in r]
    return min(plain or raws, key=lambda s: (len(s), s))


def _disasm_at(bv, addr: int) -> str:
    """One line of disassembly at *addr*, or ``""`` when the view cannot answer.

    Guarded exactly like the sibling row emitters in ``read_xrefs`` so a BN shape
    without ``get_disassembly`` (or a failing read at an unmapped address)
    degrades to an empty string instead of failing the whole listing."""
    fn = getattr(bv, "get_disassembly", None)
    if not callable(fn):
        return ""
    try:
        return fn(int(addr)) or ""
    except Exception:
        return ""


def _present_models(bv, models, want_callsites, *, all_sites: bool = False):
    """Map the binary's symbols to model keys via the engine's own normalization;
    return ``(present model-key set, {model-key: info})`` where ``info`` carries
    the raw binary spellings (#556) and, under ``want_callsites``, enriched
    callsite rows with containing function + kind (#553/#560).

    #958: a row is a callsite only when the instruction CALLS the symbol (or
    tail-branches to it). A code reference is not a call: every `adrp` of the
    page an import's GOT slot sits on references that slot, which listed 47
    `memcpy` "callsites" for a binary with three. An import stub's own forwarding
    jump (the x86 `.plt.sec` `jmp [GOT]`) is not a caller either, and AArch64's
    `br x17` never was one, so neither is counted and the count is the same on
    both. The remaining non-audit rows (self stubs and constant-argument inline
    builtins) are counted but, unless ``all_sites``, left out of ``callsites``
    so the list is the work queue it is used as."""
    present: set[str] = set()
    info: dict[str, dict[str, Any]] = {}
    # #472: several distinct binary symbol spellings can normalize to the SAME
    # model key (memcpy + memcpy@plt -> memcpy; _Znwm + operator new -> Znwm), so
    # accumulate their callsites into one slot instead of overwriting, and track
    # every raw spelling so the model output can name what xrefs/callsites expect.
    raw_names: dict[str, set[str]] = {}
    names: set[str] = set()
    for fn in getattr(bv, "functions", []) or []:
        names.add(str(getattr(fn, "name", "")))
    for sym in (getattr(bv, "get_symbols", lambda: [])() or []):
        names.add(str(getattr(sym, "name", "")))
    keyed: dict[str, list[Any]] = {}
    for nm in names:
        key, _model = _taint.lookup_model(models, nm)
        if not key:
            continue
        present.add(key)
        raw_names.setdefault(key, set()).add(nm)
        info.setdefault(key, {})
        if want_callsites:
            keyed.setdefault(key, []).extend(
                getattr(bv, "get_symbols_by_name", lambda n: [])(nm) or [])
    for key, syms in keyed.items():
        slot = info[key]
        # Every address the symbol answers to (stub, GOT slot, extern) is a call
        # target; a call to any of them is a call to the import.
        targets = {int(getattr(s, "address", 0)) for s in syms}
        flagged = _flagged_args(models.get(key))
        rows: list[dict[str, Any]] = []
        seen: set[int] = set()
        for target in sorted(targets):
            for ref in (getattr(bv, "get_code_refs", lambda a: [])(target) or []):
                ai = int(getattr(ref, "address", 0))
                # Dedup addresses so an alias resolving to the same site is not
                # double-counted (#472).
                if ai in seen:
                    continue
                seen.add(ai)
                container = _containing_fn(bv, ai)
                shape = _callsite_shape(container, ai, targets, flagged)
                if shape == "data":
                    continue
                fname, kind = _classify_callsite(bv, ai, key, models)
                if kind == "import_thunk":
                    continue
                if shape == "inline" and kind == "app_caller":
                    kind = "inline_builtin"
                # #794: carry one line of context per row. `disasm` is the key
                # the sibling address-row emitters (`read_xrefs` ref rows,
                # `seam`'s call-context row) already use.
                # #978's rule: JSON keeps the raw `function`; text prints the
                # demangled `function_display_name` unless --mangled.
                rows.append({"address": hex(ai), "function": fname, "kind": kind,
                             "function_display_name": (il_format._display_name(container)
                                                       if container is not None else None),
                             "disasm": _disasm_at(bv, ai)})
        rows.sort(key=lambda r: int(r["address"], 16))
        slot["callsite_count"] = len(rows)
        # #560: application-caller subset -- the real VR work queue.
        slot["audit_callsite_count"] = sum(1 for r in rows if r["kind"] == "app_caller")
        hidden: dict[str, int] = {}
        if not all_sites:
            for r in rows:
                if r["kind"] != "app_caller":
                    hidden[r["kind"]] = hidden.get(r["kind"], 0) + 1
            rows = [r for r in rows if r["kind"] == "app_caller"]
        slot["callsites"] = rows
        slot["hidden_callsites"] = hidden
    for key, rn in raw_names.items():
        slot = info[key]
        slot["raw_symbols"] = sorted(rn)
        slot["resolved_symbol"] = _pick_resolved(key, slot["raw_symbols"])
    return present, info


def _annotate_presence(catalog, present_keys, info, present_only):
    def _mark(entry):
        key = entry["symbol"]
        entry["present"] = key in present_keys
        data = info.get(key) or {}
        # #556: the raw imported/exported spelling(s) so a consumer can pick the
        # right identifier for xrefs/callsites itself, rather than guessing which
        # command wants the normalized alias vs the raw symbol.
        if data.get("raw_symbols"):
            entry["raw_symbol"] = data.get("resolved_symbol")
            entry["resolved_symbol"] = data.get("resolved_symbol")
            entry["accepted_aliases"] = data["raw_symbols"]
        # #553/#560: callsite rows (address + containing function + kind) plus the
        # raw and audit-only counts. Present-only mode (no --callsites) has no
        # count, matching the prior behavior.
        if "callsite_count" in data:
            entry["callsite_count"] = data["callsite_count"]
            entry["callsites"] = data.get("callsites", [])
            entry["audit_callsite_count"] = data.get("audit_callsite_count", 0)
            entry["hidden_callsites"] = data.get("hidden_callsites", {})
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
    """Whether a recovered constant string contains a printf conversion.

    This narrows the candidate diagnostic to format-shaped calls. It proves
    neither that the callee is variadic nor that nearby stores feed the call.
    """
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
    """Disclose possible call-model truncation from nearby stack candidates.

    A format-shaped argument and a contiguous stack-store run warrant inspection,
    not a claim of argument provenance. Stay silent when the convention is unknown
    or the call already has recovered stack arguments.
    """
    reg_count = _int_arg_reg_count(func)
    if reg_count is None or len(params) > reg_count:
        return None
    # A known fixed-arity libc/BSD callee is never a truncated variadic, even if it
    # takes a format-shaped string arg (the residual FP the format-string gate alone
    # let through -- e.g. strlen of a format literal on MIPS, #489 review).
    _base = str(callee_name or "").split("@", 1)[0].lstrip("_")
    if _base in _KNOWN_FIXED_ARITY:
        return None
    # Keep the diagnostic focused on format-shaped calls. This filter does not
    # establish a prototype defect or distinguish arguments from local spills.
    if not _params_have_format_string(bv, params):
        return None
    raw = _stack_arg_store_offsets(func, target_addr)
    # A contiguous pointer-width run near sp is a candidate region only; local
    # and spill stores can have the same shape.
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
    fn_name = shlex.quote(hex(int(func.start)) if getattr(func, "start", None) is not None
                          else str(getattr(func, "name", "<fn>")))
    target = None
    if call_insn is not None:
        try:
            target = _taint.resolve_call_target(bv, call_insn, follow_thunks=True)
        except Exception:
            target = None
    # #948: the stores are found by OFFSET SHAPE (a contiguous run near sp in
    # the call's block), which does not prove they are this call's arguments --
    # so they are stated as candidates, never as proven-dropped arguments.
    observed = (
        f"{disp} was recovered with {len(params)} MLIL argument(s) (all "
        f"register-passed), and the call's LLIL block has {len(offsets)} candidate "
        f"outgoing stack store(s) [{off_list}] before it (identified by offset shape "
        f"only, not proven to feed the call)")
    variadic = il_format._variadic_determination(getattr(target, "function", None))
    if variadic is True:
        # #948: the callee is ALREADY declared variadic, so "declare it variadic"
        # promised a fix that changes nothing. If those stores are arguments,
        # this is a call-recovery limit at this site, not a prototype defect.
        return (
            f"possible call-model truncation (#489): {observed}. {disp} is already "
            f"declared variadic, so this is not a prototype defect and redeclaring it "
            f"will not restore them: if the stores are stack-passed varargs, BN's "
            f"call model at this site omits them and `trace`/`taint`/`defuse` cannot "
            f"see them. Inspect the stores with `bn il {fn_name} --view llil --ssa`."
        )
    status = ("the callee is determined nonvariadic" if variadic is False else
              "the callee's variadic status is undetermined")
    return (
        f"possible call-model truncation (#489): {observed}; {status}. "
        f"Inspect the stores and callee prototype with "
        f"`bn il {fn_name} --view llil --ssa`. Nearby local/spill stores and a "
        f"format literal do not establish argument provenance. If independent "
        f"ABI/value evidence confirms omitted arguments, distinguish a recovery "
        f"limitation from a declaration discrepancy before changing the prototype."
    )


def _out_of_range_arg_msg(func, arg_index: int, n: int, *,
                          variadic_callee: str | None = None,
                          callee_variadic: bool | None = None) -> str:
    """The precise out-of-range message for `trace --arg N`, with the #324 stack-
    passed / first-missing-register notes. Used when no #433 register fallback
    applies. *variadic_callee* names a callee BN DETERMINED variadic (#948): its
    missing args are a call-recovery limit, never a mis-typed prototype."""
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
            f"(e.g. variadic) args -- inspect `bn il --view llil --ssa "
            f"{positional_selector(func.name)}` for the stack stores feeding the call (#324)."
        )
    elif reg_count is not None and n < reg_count and arg_index == n:
        # The requested index is exactly the FIRST register slot BN did not
        # recover (still within the ABI's integer-arg registers) -- the tight
        # "BN stopped recovering args right here" signal.
        msg += (
            f" Note: arg {arg_index} is the first integer-arg register this "
            f"convention passes that BN did not recover as a param ({n} "
            f"recovered); the requested index alone does not prove that an "
            f"argument exists or that the prototype is wrong. Inspect "
            f"`bn il --view llil --ssa {positional_selector(func.name)}` for independent evidence "
            f"that a value feeds the call (#324)."
        )
    if variadic_callee:
        msg += (
            f" {variadic_callee} is already declared variadic: if an argument is "
            f"missing here, that is a limit of BN's call recovery at this site, not a "
            f"prototype defect -- redeclaring the prototype will not add it."
        )
    elif callee_variadic is False:
        msg += (" The callee is determined nonvariadic. If independent argument "
                "provenance demonstrates extra arguments, recover and apply the "
                "actual prototype; an out-of-range request alone is not a discrepancy.")
    else:
        msg += (" The callee's variadic status is undetermined; inspect its prototype "
                "and argument provenance before diagnosing missing arguments.")
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
    """{index, [register], [callee], [name]} for the traced call argument:
    its calling-convention register, the resolved callee's name (#662), and
    the callee's C parameter name when it resolves (#166). ``name`` is kept
    for JSON back-compat only -- the traced value is the caller's operand at
    the callsite, not necessarily anything the callee's parameter is named,
    so renderers must key headers off ``callee``, not ``name``."""
    label: dict[str, Any] = {"index": arg_index}
    reg = _arg_register(caller_func, arg_index)
    if reg:
        label["register"] = reg
    try:
        rt = _taint.resolve_call_target(bv, call_insn, follow_thunks=True)
        callee = getattr(rt, "function", None)
        callee_nm = str(getattr(callee, "name", "") or "")
        if callee_nm:
            label["callee"] = callee_nm
            label["callee_display_name"] = il_format._display_name(callee) or callee_nm
        pvars = list(getattr(callee, "parameter_vars", []) or []) if callee else []
        if 0 <= arg_index < len(pvars):
            nm = getattr(pvars[arg_index], "name", None)
            if nm:
                label["name"] = str(nm)
    except Exception:
        pass
    return label

_MAX_CALL_DEPTH = 10  # hard interprocedural recursion ceiling, independent of --ip-depth


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
    trace: list[dict[str, Any]] = []
    # #416/#672: locals whose address is passed into a call, computed once
    # per function (not per terminus). Used in both intra and interprocedural
    # modes so an out-param fill is never mistaken for "undefined".
    out_param_map = _build_out_param_map(ctx, bv, ssa_func)
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
                # This callee's SSA form doesn't support the query (rare: the
                # crossing guard already requires `hasattr(...,
                # "get_ssa_var_definition")` before recursing here). Record a
                # terminal step rather than vanishing, so an empty
                # `assumptions` list still means "complete within the
                # requested mode and budgets" (see `_trace_assumptions`).
                trace.append({
                    "ssa_var": str(ssa_var),
                    "ssa_label": _ssa_label(ssa_var),
                    "depth": depth,
                    "terminates": True,
                    "reason": "ssa_definition_unavailable",
                })
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
            param_index = _parameter_index(ssa_func, ssa_var)
            if param_index is not None:
                entry["reason"] = "function_parameter"
                # #959: WHICH parameter, and the register it arrives in. The
                # roll-up used to say only "function parameter", so the reader
                # had to map `arg4` to an ABI slot by hand -- and on AArch64 the
                # count suffix `x1` read as the register itself.
                entry["param_index"] = param_index
                param_reg = _parameter_register(ssa_func, param_index)
                if param_reg:
                    entry["param_register"] = param_reg
            else:
                entry["reason"] = "undefined_or_global"
                # #416/#672: a value that bottoms out at a LOCAL whose address
                # was passed into a call (the common stack-struct out-param
                # form, `parse(input, &rec); use(rec.len)`) was likely written
                # by that callee through an out-pointer. Interprocedural
                # tracing follows only return values, and INTRA tracing never
                # crosses the call at all, so in both modes name the callee
                # instead of leaving an "undefined" terminus that reads like
                # proof of origin.
                out_callee = _lookup_out_param_callee(out_param_map, ssa_var, ref_addr)
                if out_callee:
                    entry["reason"] = ("interprocedural_out_param_not_followed" if interprocedural
                                        else "out_param_not_followed")
                    entry["out_param_callee"] = out_callee
                    entry["out_param_callee_display_name"] = _shown_symbol_name(bv, out_callee)
            trace.append(entry)
            continue

        entry["address"] = hex(int(getattr(def_insn, "address", 0)))
        entry["il_text"] = str(def_insn)
        entry["operation"] = il_format._il_op_name(def_insn)

        def_op = entry["operation"]
        if "CALL" in def_op or "JUMP" in def_op:
            if interprocedural and ip_depth > 0 and _call_depth < _MAX_CALL_DEPTH:
                callee = _resolve_callee(ctx, bv, def_insn)
                if callee is not None:
                    callee_mlil = getattr(callee, "medium_level_il", None)
                    if callee_mlil is not None and callee_mlil.ssa_form is not None:
                        if hasattr(callee_mlil.ssa_form, "get_ssa_var_definition"):
                            callee_ret_vars = _find_return_vars(ctx, callee_mlil.ssa_form, bv)
                            if callee_ret_vars:
                                entry["cross_function"] = True
                                entry["callee"] = callee.name
                                entry["callee_display_name"] = il_format._display_name(callee)
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
                                    ct.setdefault("function_context_display_name",
                                                  il_format._display_name(callee))
                                trace.extend(callee_trace)
                                continue
            entry["terminates"] = True
            entry["reason"] = "call_or_jump_boundary"
            if interprocedural and ip_depth <= 0:
                entry["ip_depth_exhausted"] = True
            elif interprocedural and _call_depth >= _MAX_CALL_DEPTH:
                entry["call_depth_exhausted"] = True
            # Resolve the call target to its symbol so a library-call origin reads
            # as e.g. `call strlen` rather than a bare PLT address (#193).
            callee_nm = _callee_display_name(ctx, bv, def_insn)
            if callee_nm:
                entry["callee"] = callee_nm
                entry["callee_display_name"] = _shown_symbol_name(bv, callee_nm)
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
            # #416/#672: if this load reads from the address of a local that
            # was passed by-address into a call, the value was likely written
            # by that callee through an OUT-POINTER -- which interprocedural
            # tracing follows only for return values, and intra tracing never
            # crosses the call at all. Emit an honest boundary reason naming
            # the callee instead of letting the slice stop silently at a
            # `field_load`/`memory_load`.
            load_local = _address_of_target(getattr(load_expr, "src", None))
            out_callee = _lookup_out_param_callee(
                out_param_map, load_local, int(getattr(def_insn, "address", 0)))
            if out_callee:
                entry["reason"] = ("interprocedural_out_param_not_followed" if interprocedural
                                    else "out_param_not_followed")
                entry["out_param_callee"] = out_callee
                entry["out_param_callee_display_name"] = _shown_symbol_name(bv, out_callee)
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

        # #1001 item 2: `p = &local` (or `&local + k`) has no SSA reads, so a
        # plain definition here ended the worklist with no frontier and no
        # hint -- the slice just stopped one step after the call. The value is
        # a POINTER to a stack object; what the sink consumes is the bytes stored
        # into it, which this use-def walk does not follow. Stop here honestly,
        # naming the local, so the roll-up, caveat and hint can point at the
        # command that does walk those stores (`taint backward`).
        taken = (_address_of_target(getattr(def_insn, "src", None))
                 if "SET_VAR" in def_op else None)
        if taken is not None:
            entry["terminates"] = True
            entry["reason"] = "address_of_local"
            entry["local"] = str(getattr(taken, "name", None) or taken)
        else:
            entry["terminates"] = False
            # Populate `reason` from a controlled vocabulary instead of leaving it
            # null on ordinary steps: a phi merge is `phi_source`, everything else is
            # a plain `definition` (#162).
            entry["reason"] = "phi_source" if "PHI" in def_op else "definition"
        trace.append(entry)

        # Still walk any index/offset operand (`&local + i`): a tainted index
        # is provenance too.
        for rv in _ssa_vars_from(getattr(def_insn, "vars_read", []) or []):
            if rv not in visited:
                worklist.append((rv, node_depth + 1, int(getattr(def_insn, "address", 0))))

    return trace


# Compact per-example fields carried into a frontier roll-up (#552). Kept to the
# machine-consumable subset already present on a terminal step -- never recomputed.
# Deliberately omits ip_depth_exhausted/call_depth_exhausted: those are
# top-level cap markers surfaced only via `_trace_assumptions`, not per-step
# fields, so a capped boundary and a genuinely terminal one look identical here.
_FRONTIER_EXAMPLE_KEYS = (
    "ssa_label", "address", "depth", "callee", "callee_display_name", "out_param_callee",
    "base", "offset", "width", "function_context", "function_context_display_name",
    "out_param_callee_display_name", "param_index", "param_register", "local",
)
_FRONTIER_EXAMPLE_CAP = 3


def _summarize_frontiers(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Roll up the trace's TERMINAL steps into per-reason frontier groups (#552).

    A factual summary of where the backward slice STOPPED -- unresolved loads,
    caller/call boundaries, constants, parameters -- derived only from the
    terminal steps already present in ``trace`` (never recomputed, never a fresh
    walk). This classifies STOPPING REASONS, not vulnerabilities: it is a
    roll-up, not a verdict. Groups are ordered by descending count (ties broken
    by reason name) so the dominant frontier reads first; each carries up to
    ``_FRONTIER_EXAMPLE_CAP`` compact examples in trace order.
    """
    groups: dict[str, dict[str, Any]] = {}
    for step in trace:
        if not isinstance(step, dict) or not step.get("terminates"):
            continue
        reason = step.get("reason") or "unspecified"
        grp = groups.setdefault(reason, {"reason": reason, "count": 0, "examples": []})
        grp["count"] += 1
        if len(grp["examples"]) < _FRONTIER_EXAMPLE_CAP:
            grp["examples"].append(
                {k: step[k] for k in _FRONTIER_EXAMPLE_KEYS if step.get(k) is not None})
    return sorted(groups.values(), key=lambda g: (-g["count"], g["reason"]))


def _trace_assumptions(trace: list[dict[str, Any]], *, truncated: bool, max_depth: int,
                       mangled: bool = False) -> list[str]:
    """#671: honest caveats for a backward ``trace`` result, mirroring the
    ``assumptions`` key already emitted by ``taint forward``/``taint
    backward`` (see ``_render_taint_text``'s ``caveats`` block). Non-empty
    ONLY when the slice is provably incomplete -- an empty list is a
    positive claim of completeness WITHIN THE REQUESTED MODE AND BUDGETS,
    not an unconditional one (intra-mode ``call_or_jump_boundary`` steps are
    deliberately left unmarked: crossing was never attempted, so there is
    nothing to flag as capped). Every entry here must come from a stopping
    condition already recorded on the trace/return dict, never a guess."""
    assumptions: list[str] = []
    if truncated:
        assumptions.append(
            f"depth cap reached ({max_depth}); slice is incomplete -- "
            f"raise --max-depth to continue"
        )
    out_param_reasons = {"interprocedural_out_param_not_followed", "out_param_not_followed"}
    for step in trace:
        if isinstance(step, dict) and step.get("reason") in out_param_reasons:
            callee = ((step.get("out_param_callee") if mangled else
                       step.get("out_param_callee_display_name") or step.get("out_param_callee"))
                      or "a callee")
            assumptions.append(
                f"value crosses an out-parameter fill by {callee} that this "
                f"trace did not follow; the true origin may be inside it"
            )
            break
    if any(isinstance(step, dict) and step.get("ip_depth_exhausted") for step in trace):
        assumptions.append(
            "crossing was not attempted at one or more call boundaries because "
            "the --ip-depth budget was spent before reaching them; the slice "
            "may be incomplete beyond those boundaries"
        )
    if any(isinstance(step, dict) and step.get("call_depth_exhausted") for step in trace):
        assumptions.append(
            "interprocedural crossing stopped at the internal recursion-depth "
            "safety limit (not adjustable via --ip-depth); the slice may be "
            "incomplete beyond that boundary"
        )
    if any(isinstance(step, dict) and step.get("reason") == "ssa_definition_unavailable"
           for step in trace):
        assumptions.append(
            "a value's SSA definition could not be queried on one or more "
            "steps (the SSA form does not support the lookup there); the "
            "slice stops at that value without full provenance"
        )
    locals_taken = [str(step.get("local")) for step in trace
                    if isinstance(step, dict) and step.get("reason") == "address_of_local"]
    if locals_taken:
        assumptions.append(
            f"the slice reaches the address of local {locals_taken[0]}"
            + (f" (and {len(locals_taken) - 1} more)" if len(locals_taken) > 1 else "")
            + "; stores into it (array elements, struct fields) are not followed, so "
            "the bytes the call consumes have no provenance here"
        )
    return assumptions


def _parameter_index(ssa_func, ssa_var) -> int | None:
    """The 0-based position of *ssa_var* (an SSA variable with no reaching
    definition) among the function's formal parameters, or None when it is an
    undefined local or a global.

    Matched against the source function's ``parameter_vars`` by identifier
    first, then by base name. Returns None whenever parameter information
    is unavailable, so the caller falls back to a neutral label rather than
    guessing 'parameter'."""
    source = getattr(ssa_func, "source_function", None)
    params = list(getattr(source, "parameter_vars", []) or [])
    if not params:
        return None
    base = getattr(ssa_var, "var", ssa_var)
    ident = getattr(base, "identifier", None)
    base_name = str(ssa_var).split("#")[0]
    for pos, param in enumerate(params):
        if ident is not None and getattr(param, "identifier", None) == ident:
            return pos
        if base_name and str(getattr(param, "name", param)) == base_name:
            return pos
    return None


def _is_parameter_ssa_var(ctx, ssa_func, ssa_var) -> bool:
    """True if *ssa_var* is a formal parameter of the function (see
    :func:`_parameter_index`)."""
    return _parameter_index(ssa_func, ssa_var) is not None


def _parameter_register(ssa_func, param_index: int) -> str | None:
    """The register parameter *param_index* arrives in, read from the parameter
    variable's own storage -- not from its position in the calling convention's
    integer-register list, which a float or stack-passed parameter earlier in
    the list would shift (#959). None for a stack-passed parameter or when the
    storage is unreadable."""
    source = getattr(ssa_func, "source_function", None)
    try:
        param = list(getattr(source, "parameter_vars", []) or [])[param_index]
        # VariableSourceType.RegisterVariableSourceType == 1 (no BN import here).
        if int(getattr(param, "source_type", -1)) != 1:
            return None
        name = source.arch.get_reg_name(int(param.storage))
    except Exception:
        return None
    return str(name) if name else None


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


def _shown_symbol_name(bv, raw_name: str | None) -> str | None:
    """The demangled display name (the symbol's ``short_name``, #978) for a raw
    symbol name a trace step carries, else the raw name itself (#1001 item 4)."""
    if not raw_name:
        return raw_name
    try:
        sym = bv.get_symbol_by_raw_name(raw_name)
    except Exception:
        sym = None
    short = getattr(sym, "short_name", None) if sym is not None else None
    return short if isinstance(short, str) and short else raw_name


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
    mangled: bool = False,
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
            try:
                _oor_callee = _resolve_callee(ctx, bv, call_insn)
            except Exception:
                _oor_callee = None
            raise OperationFailure(
                "invalid_arg_index",
                _out_of_range_arg_msg(
                    func, arg_index, len(params),
                    variadic_callee=(str(getattr(_oor_callee, "name", "") or "the callee")
                                     if il_format._variadic_determination(_oor_callee) is True
                                     else None),
                    callee_variadic=il_format._variadic_determination(_oor_callee)))

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
    # An address-of arg is a pointer to a local: tracing it follows where the
    # *pointer* came from, not the bytes stored into the pointee. The hint used
    # to fire only when the MLIL call parameter itself was `&local` (#166); on
    # AArch64 the address arrives through a register copy (`x1 = &var_68`), so
    # the same dead end is now also recognised on the parameter's resolved
    # definition (#1001 item 2). Constant args keep their own hint.
    direct_addr = (param_expr is not None and not initial_vars
                   and (_is_address_of(param_expr) or _address_of_target(param_expr) is not None))
    if param_expr is not None and not initial_vars and not direct_addr:
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

    if direct_addr:
        _taken = _address_of_target(param_expr)
        local_name = str(getattr(_taken, "name", None) or _taken) if _taken is not None else ""
    else:
        # Only when the argument's OWN value is that address: its definition
        # chain is pure copies down to `&local`. A value LOADED from a local, or
        # a phi of two locals, also reaches an `address_of_local` step, but
        # "arg N is the address of local X" would be false there (#1001 review).
        local_name = _copied_address_of_local(ssa_func, initial_vars)
    if local_name is not None:
        hints.append(_address_of_local_hint(arg_index, local_name, arg_label, func,
                                           mangled=mangled))

    truncated = len(trace) >= max_depth
    return {
        "function": func.name,
        # #1001 item 4: text names the function and callee as `decompile` does;
        # JSON keeps the raw names beside them.
        "function_display_name": il_format._display_name(func),
        "function_address": hex(func.start),
        "target_address": hex(target_addr),
        "arg_index": arg_index,
        "arg_label": arg_label,
        # #755: the traced callee, stated STRUCTURALLY with an explicit null when
        # it does not resolve (an indirect call through a register or vtable
        # slot). `arg_label` merely OMITS its `callee` key in that case, so a JSON
        # consumer could not tell "we looked, and this call has no resolvable
        # callee" from "this bridge never computed one" -- the same ambiguity the
        # text header had. Derived from `arg_label` in this one expression so the
        # header and this field cannot disagree about which call was answered.
        "callee": arg_label.get("callee") or None,
        "view": view,
        "interprocedural": interprocedural,
        "ip_depth": ip_depth if interprocedural else 0,
        "truncated": truncated,
        "step_count": len(trace),
        "trace": trace,
        # Top-level roll-up of where the slice stopped, so an agent needn't scan
        # the whole `trace` array to compare stopping conditions across callsites
        # (#552). Factual summary, not a verdict.
        "frontiers": _summarize_frontiers(trace),
        # #671: non-empty ONLY when the slice is provably incomplete (depth cap
        # or an unfollowed out-param fill) -- mirrors `taint forward`/`backward`'s
        # `assumptions` key so `caveats (N)` renders in text mode too.
        "assumptions": _trace_assumptions(trace, truncated=truncated, max_depth=max_depth,
                                         mangled=mangled),
        "hints": hints,
    }


def _copied_address_of_local(ssa_func, initial_vars: list) -> str | None:
    """The local whose address the traced value IS, through pure SSA copies
    (``x1 = x8; x8 = &var_68``), or None -- a load, phi, arithmetic on a
    non-address or a call anywhere on the chain means the value is something
    else that merely involves the local."""
    if len(initial_vars) != 1:
        return None
    cur = initial_vars[0]
    for _ in range(8):
        try:
            d = ssa_func.get_ssa_var_definition(cur)
        except Exception:
            return None
        if d is None or "SET_VAR" not in il_format._il_op_name(d):
            return None
        src = getattr(d, "src", None)
        taken = _address_of_target(src)
        if taken is not None:
            return str(getattr(taken, "name", None) or taken)
        if il_format._il_op_name(src) != "MLIL_VAR_SSA":
            return None
        reads = _ssa_vars_from(getattr(src, "vars_read", []) or [])
        if len(reads) != 1:
            return None
        cur = reads[0]
    return None


def _address_of_local_hint(arg_index: int, local_name: str, arg_label: dict, func,
                           *, mangled: bool = False) -> str:
    """The #166 / #1001 item 2 hint for an argument that is the address of a
    local: trace follows the pointer, not the bytes in the pointee. Every
    command it prints runs as printed (#1001 review): the `arg:` sink locator
    and the forward source need a resolved callee, and `call:` seeds only a
    callee whose model declares outputs -- otherwise the source is the arg
    itself. The callee is named by its demangled display name in prose and by
    its raw symbol in the commands."""
    what = f"local {local_name}" if local_name else "a local"
    head = (f"arg {arg_index} is the address of {what}: trace follows where the "
            f"pointer came from, not the bytes stored into it.")
    callee = arg_label.get("callee")
    fn_ref = hex(int(func.start))
    if not callee:
        return (head + " The call's target did not resolve, so no `arg:` locator can "
                f"name it; read the stores into the local with `bn decompile {fn_ref}`, "
                "or resolve the target before slicing.")
    shown = callee if mangled else arg_label.get("callee_display_name") or callee
    _, model = _taint.lookup_model(_taint.load_models(), callee)
    # `call:` seeds what the model declares the callee writes; use it only when
    # that includes THIS argument's buffer (`*arg:N`) -- for recvfrom's
    # `addrlen`, `call:recvfrom` seeds the payload and return, not addrlen.
    writes_this = any(str((s or {}).get("to", "")) == f"*arg:{arg_index}"
                      for s in (model or {}).get("sources") or [])
    source = shlex.quote(f"call:{callee}" if writes_this else f"arg:{callee}:{arg_index}")
    sink = shlex.quote(f"arg:{callee}:{arg_index}")
    return (head + f" To slice the stores into it before the call (array elements, "
            f"struct fields), run `bn taint backward -f {fn_ref} --sink "
            f"{sink}`; if {shown} fills it instead (an output "
            f"pointer), run `bn taint forward -f {fn_ref} --source {source}`.")
