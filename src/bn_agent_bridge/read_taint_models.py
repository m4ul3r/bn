"""Catalog view over the taint-model DB (``bn taint models``).

Pure model->catalog shaping lives here so it is unit-testable without BN; the
target-aware presence annotation is added by the op handler in
``read_taint_slice.py`` using the BinaryView. Import-free of ``bridge``/``seam``.
"""
from __future__ import annotations

from typing import Any

# #555: `bn taint models [--present]` is a CATALOG of modeled sinks/sources --
# NOT a list of taint findings. This note rides at the top of every catalog so an
# agent (or a human skimming JSON) cannot misread the callsite inventory as
# confirmed vulnerabilities. A listed callsite is a finding ONLY if the flagged
# argument is actually tainted there, which only `bn taint` can decide.
CATALOG_NOTE = (
    "present model/callsite catalog; NOT taint findings. Each entry marks a "
    "MODELED sink/source that EXISTS in the binary; a listed callsite becomes a "
    "finding only if the flagged argument is actually tainted there -- verify "
    "with `bn taint backward`/`forward`. A constant/non-tainted argument is not a bug."
)


def _arg_phrase(indices: list[Any]) -> str | None:
    """"argument 2" / "arguments 0 or 1" for a sink's tainted-arg index list, or
    None when the list is empty (an unconditional sink like ``gets``). Entries may
    be pre-formatted strings carrying a marker (``"1 (length)"``, #808)."""
    idxs = [str(i) for i in indices]
    if not idxs:
        return None
    if len(idxs) == 1:
        return f"argument {idxs[0]}"
    return "arguments " + " or ".join(idxs)


def _sink_model_description(cls: str, sink: dict[str, Any]) -> str:
    """Conditional, non-verdict wording for a modeled sink (#555): says WHAT the
    model flags and UNDER WHAT CONDITION, keeping the "... if argument N is
    tainted" framing so a constant-argument callsite is never implied to be a bug.
    """
    _args: list[Any] = list(sink.get("tainted_args", []) or [])
    # #808: one entry can arm BOTH explicit tainted args and a write LENGTH
    # (`snprintf(dst, n, fmt, ...)`: the format is arg2, the write length into arg0
    # is arg1, declared with the #443 `len_arg`/`buf_arg` pair). The condition must
    # name every index the engine arms, or the catalog reads as though the length
    # arg were unmodeled. The length goes first so the marker reads as a property
    # of that one index.
    _la = sink.get("len_arg")
    if _la is not None and _la not in _args:
        _args.insert(0, f"{_la} (length)")
    phrase = _arg_phrase(_args)
    if phrase is None:
        # No tainted-arg condition (e.g. gets()): still a catalog entry, not a finding.
        return f"{cls} sink -- catalog entry (always-unsafe API); not a finding by itself"
    verb = "is" if " or " not in phrase else "are"
    return f"{cls} sink -- flagged as a finding ONLY IF {phrase} {verb} tainted at a callsite"


def build_catalog(models: dict[str, Any], *, role: str | None = None,
                  sink_class: str | None = None) -> dict[str, Any]:
    """Group the model DB into sources / sinks-by-class / propagators.

    ``role`` filters to one role; ``sink_class`` filters sinks to one bug class
    (and implies ``role='sink'``). Doc keys (``_comment``-prefixed) are skipped
    exactly as the engine's coercion skips them; a non-dict entry is skipped here
    rather than raised on, because this catalog is a report over a DB the engine
    has already accepted -- ``_coerce_model_map`` is the gate that refuses a
    non-dict model with ``TaintError`` before any of this runs.

    Every entry carries ``model_name`` (the normalized alias that taint commands
    accept -- #556) and ``is_finding: false`` (#555); sinks additionally carry a
    conditional ``model_description``. Presence/callsite/raw-symbol fields are
    layered on later by the target-aware annotation in ``read_taint_slice``.
    """
    want = role or ("sink" if sink_class else None)
    sources: list[dict[str, Any]] = []
    sinks_by_class: dict[str, list[dict[str, Any]]] = {}
    propagators: list[dict[str, Any]] = []
    for name, model in models.items():
        # #849: skip the DOC-key prefix only, the way the engine's own coercion
        # does (``taint_models._coerce_model_map``). Skipping every ``_``-leading
        # key also dropped real models -- ``__isoc99_scanf``/``__isoc99_fscanf``,
        # ``__isoc99_vsscanf``, the ``_IO_*`` family -- which the engine resolves
        # on a real binary, so the catalog (and the ``--present`` audit built on
        # it) under-reported the models actually applied.
        if str(name).startswith("_comment") or not isinstance(model, dict):
            continue
        if model.get("sources") and want in (None, "source"):
            tos = ", ".join(str(s.get("to")) for s in model["sources"])
            sources.append({"symbol": name, "model_name": name, "is_finding": False,
                            "to": tos})
        sink = model.get("sink")
        if sink and want in (None, "sink"):
            cls = sink.get("class") or "?"
            if sink_class is None or cls == sink_class:
                entry = {
                    "symbol": name, "model_name": name, "is_finding": False,
                    "tainted_args": sink.get("tainted_args", []),
                    "class": cls, "detail": sink.get("detail"),
                    "model_description": _sink_model_description(cls, sink)}
                # #443: surface a bounded-write sink's length/buffer argument indices.
                if sink.get("len_arg") is not None:
                    entry["len_arg"] = sink.get("len_arg")
                if sink.get("buf_arg") is not None:
                    entry["buf_arg"] = sink.get("buf_arg")
                sinks_by_class.setdefault(cls, []).append(entry)
        if model.get("propagates") and want in (None, "propagator"):
            fts = ", ".join(f"{p.get('from')}->{p.get('to')}" for p in model["propagates"])
            propagators.append({"symbol": name, "model_name": name, "is_finding": False,
                                "from_to": fts})
    return {
        # #555: loud, machine- and human-visible "this is a catalog, not findings".
        "presence_catalog": True,
        "is_finding": False,
        "catalog_note": CATALOG_NOTE,
        "sources": sources,
        "sinks_by_class": sinks_by_class,
        "propagators": propagators,
    }
