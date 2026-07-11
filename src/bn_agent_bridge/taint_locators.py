"""Locator grammar + signature/label helpers.

Split out of ``taint_engine`` (pure structural move, #562). Parses/renders
source/sink locators and builds address-free flow signatures/labels. The
engine re-exports these names for back-compat.
"""
from __future__ import annotations

from typing import Any

from .taint_models import TaintError
from .taint_il import op_name


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
    if kind in ("call", "model"):
        # call:/model: seed every output the callee's taint model declares; both
        # parse to kind "call" today, but round-trip either spelling for safety.
        return f"{kind}:{loc.get('callee')}"
    return str(kind)


def _instr_dict(ins: Any, reason: str | None = None, tainted: list[str] | None = None,
                callee: str | None = None) -> dict[str, Any]:
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
    if callee is not None:
        out["callee"] = str(callee)
    return out


def _render_source(source: Any) -> str:
    """Canonical locator string for a forward-flow source echo.

    The forward run echoes its sources as locator *dicts* (``_describe_locator``),
    so rendering them with ``str()`` leaks Python's ``dict.__repr__`` into
    ``signature.rendered`` (``"{'kind': 'param', 'index': 0} -> ..."``). Route
    locator dicts through the shared canonical grammar (:func:`format_locator`)
    instead, so the rendered signature reads ``param:0 -> ...`` and round-trips
    with :func:`parse_locator` (#551). Non-dict sources -- the backward origin's
    callee/kind string and the ``"multiple"``/``"?"`` sentinels -- pass through
    unchanged. The structured ``source`` object stays available separately in the
    result's top-level ``sources`` echo."""
    if isinstance(source, dict) and source.get("kind"):
        return format_locator(source)
    return str(source)


def _make_signature(source: Any, chain: list[str], sink_class: str | None,
                    sink_callee: str | None) -> dict[str, Any]:
    cls = f"[{sink_class}] " if sink_class else ""
    src = _render_source(source)
    parts = [src] + [str(c) for c in chain] + [f"{cls}{sink_callee}"]
    return {
        "source": src,
        "chain": [str(c) for c in chain],
        "sink_class": sink_class,
        "sink_callee": sink_callee,
        "rendered": " → ".join(parts),
    }


def derive_flow_facts(*, direction: str,
                      path: list[dict[str, Any]] | None = None,
                      sink: dict[str, Any] | None = None,
                      sources: list[str] | None = None,
                      leaves: list[dict[str, Any]] | None = None,
                      fn_name: str | None = None,
                      origin: dict[str, Any] | None = None,
                      crossed_functions: list[str] | None = None,
                      ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Structural triage facts + an address-free grouping signature for one flow.

    Pure: derived from the already-assembled result (the flow's reconstructed
    path/slice, the run's source echo, the run-global leaves). No BN access, no
    engine state -- unit-testable against fabricated dicts. Returns
    (metrics, signature). ``traverses_unresolved`` is an honest structural
    correlation (a leaf address coincides with a path address), NOT causal proof.
    See design spec 2026-07-03-taint-triage-output.
    """
    steps = path or []
    if direction == "forward":
        callees = [s.get("callee") for s in steps if s.get("callee")]
        sink_callee = (sink or {}).get("callee")
        chain: list[str] = []
        for c in callees[:-1]:                       # drop trailing sink callee
            if c and c != sink_callee and c not in chain:
                chain.append(str(c))
        srcs = sources or []
        source = srcs[0] if len(srcs) == 1 else ("multiple" if len(srcs) > 1 else "?")
        sink_class = (sink or {}).get("class")
        step_addrs = {s.get("address") for s in steps}
        traverses = any(lf.get("address") in step_addrs
                        for lf in (leaves or []) if lf.get("address"))
        metrics = {"steps": len(steps), "fns_spanned": 1 + len(chain),
                   "traverses_unresolved": bool(traverses)}
        return metrics, _make_signature(source, chain, sink_class, sink_callee)

    # backward
    crossed = [str(c) for c in (crossed_functions or [])]
    chain = []
    for c in crossed[1:]:                            # crossed[0] is the sink's fn
        if c and c not in chain:
            chain.append(c)
    o = origin or {}
    source = o.get("callee") or o.get("kind") or "?"
    sink_callee = (sink or {}).get("callee") or (sink or {}).get("kind")
    traverses = o.get("kind") in {"unresolved", "indirect_call", "field_load_unresolved"}
    metrics = {"steps": len(steps), "fns_spanned": max(1, len(set(crossed)) + 1),
               "traverses_unresolved": bool(traverses)}
    return metrics, _make_signature(source, chain, None, sink_callee)
