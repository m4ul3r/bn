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
    None when the list is empty (an unconditional sink like ``gets``)."""
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
    phrase = _arg_phrase(sink.get("tainted_args", []) or [])
    if phrase is None and sink.get("len_arg") is not None:
        phrase = f"argument {sink.get('len_arg')} (length)"
    if phrase is None:
        # No tainted-arg condition (e.g. gets()): still a catalog entry, not a finding.
        return f"{cls} sink -- catalog entry (always-unsafe API); not a finding by itself"
    verb = "is" if " or " not in phrase else "are"
    return f"{cls} sink -- flagged as a finding ONLY IF {phrase} {verb} tainted at a callsite"


def build_catalog(models: dict[str, Any], *, role: str | None = None,
                  sink_class: str | None = None) -> dict[str, Any]:
    """Group the model DB into sources / sinks-by-class / propagators.

    ``role`` filters to one role; ``sink_class`` filters sinks to one bug class
    (and implies ``role='sink'``). Doc keys (``_``-prefixed) and non-dict entries
    are skipped, matching the engine's model coercion.

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
        if str(name).startswith("_") or not isinstance(model, dict):
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
