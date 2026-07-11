"""Forward-run result-shaping / zero-sink diagnostics.

Split out of ``taint_engine`` (pure structural move, #562). The natural home
for later result-honesty helpers; today it hosts the zero-sink frontier
diagnostics. Pure: derived from an already-assembled sub-result dict plus the
run's seed-callsite count -- no BN access, no engine state.
"""
from __future__ import annotations

from typing import Any


def forward_zero_diagnostics(sub: dict[str, Any], *, seed_callsites: int) -> dict[str, Any]:
    """Frontier diagnostics for a zero-sink forward run (#559).

    Purely descriptive: seed reach (matched source callsites, tainted SSA
    values produced, last propagated use) plus the frontier the propagation
    stopped at (unresolved / coarse-memory leaf counts, whether any unmodeled
    external/in-binary call was reached) and a suggested next action. Never
    asserts a vulnerability -- it explains "flow hit an unmodeled parser we
    couldn't follow" vs "nothing flows"."""
    diag = sub.get("diag") or {}
    leaves = sub.get("leaves") or []
    assumptions = sub.get("assumptions") or []
    by_kind: dict[str, int] = {}
    for lf in leaves:
        k = str(lf.get("kind", "?"))
        by_kind[k] = by_kind.get(k, 0) + 1
    UNRESOLVED = ("unmodeled_callee", "arg_under_recovered", "indirect_call_unresolved")
    COARSE = ("coarse_memory_store", "pointer_escape")
    unresolved_n = sum(by_kind.get(k, 0) for k in UNRESOLVED)
    coarse_n = sum(by_kind.get(k, 0) for k in COARSE)
    # An external callee with no taint model returns conservatively tainted
    # and is disclosed as an assumption, not a leaf -- fold it into the
    # "unmodeled call reached" signal so a parser behind an import stub counts.
    ext_no_model = any("has no model" in a for a in assumptions)
    unmodeled_reached = bool(unresolved_n or ext_no_model)
    tainted_values = int(diag.get("tainted_values", 0))

    if unmodeled_reached:
        next_action = (
            "taint reached an unmodeled call frontier (an unresolved/in-binary "
            "callee or an external callee with no taint model); recover the "
            "callee prototype with `bn proto set` or re-run `bn taint forward` "
            "seeded inside it to follow the flow further")
    elif coarse_n:
        next_action = (
            "taint escaped through a coarse-memory frontier (a pointer/store not "
            "precisely tracked); inspect the frontier leaves or seed the "
            "destination buffer directly with `--source var:<buf>`")
    elif tainted_values <= 1:
        next_action = (
            "the source seeded but produced no further tainted uses; verify the "
            "`--source` locator matches the intended value")
    else:
        next_action = (
            "taint propagated but reached no modeled sink and hit no unresolved "
            "frontier; the flow appears to dead-end locally -- inspect the last "
            "propagated use or widen `--max-depth`")

    return {
        "source_callsites": int(seed_callsites),
        "tainted_values": tainted_values,
        "last_use": diag.get("last_use"),
        "unmodeled_calls_reached": unmodeled_reached,
        "frontier": {
            "unresolved": unresolved_n,
            "coarse_memory": coarse_n,
            "by_kind": by_kind,
        },
        "next_action": next_action,
    }
