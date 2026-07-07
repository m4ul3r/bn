"""Catalog view over the taint-model DB (``bn taint models``).

Pure model->catalog shaping lives here so it is unit-testable without BN; the
target-aware presence annotation is added by the op handler in
``read_taint_slice.py`` using the BinaryView. Import-free of ``bridge``/``seam``.
"""
from __future__ import annotations

from typing import Any


def build_catalog(models: dict[str, Any], *, role: str | None = None,
                  sink_class: str | None = None) -> dict[str, Any]:
    """Group the model DB into sources / sinks-by-class / propagators.

    ``role`` filters to one role; ``sink_class`` filters sinks to one bug class
    (and implies ``role='sink'``). Doc keys (``_``-prefixed) and non-dict entries
    are skipped, matching the engine's model coercion.
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
            sources.append({"symbol": name, "to": tos})
        sink = model.get("sink")
        if sink and want in (None, "sink"):
            cls = sink.get("class") or "?"
            if sink_class is None or cls == sink_class:
                entry = {
                    "symbol": name, "tainted_args": sink.get("tainted_args", []),
                    "class": cls, "detail": sink.get("detail")}
                # #443: surface a bounded-write sink's length/buffer argument indices.
                if sink.get("len_arg") is not None:
                    entry["len_arg"] = sink.get("len_arg")
                if sink.get("buf_arg") is not None:
                    entry["buf_arg"] = sink.get("buf_arg")
                sinks_by_class.setdefault(cls, []).append(entry)
        if model.get("propagates") and want in (None, "propagator"):
            fts = ", ".join(f"{p.get('from')}->{p.get('to')}" for p in model["propagates"])
            propagators.append({"symbol": name, "from_to": fts})
    return {"sources": sources, "sinks_by_class": sinks_by_class, "propagators": propagators}
