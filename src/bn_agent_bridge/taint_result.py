"""Forward-run result-shaping / zero-sink diagnostics + claim gates.

Split out of ``taint_engine`` (pure structural move, #562). The natural home
for result-honesty helpers; today it hosts the zero-sink frontier diagnostics
(#559/#571) and the honesty claim gate (#562). Pure: derived from an
already-assembled sub-result dict plus the run's seed-callsite count -- no BN
access, no engine state.

Philosophy (repo-wide): ``bn taint`` shows PROPAGATION, it never asserts a
vulnerability. The claim gate here is the honest ANTI-verdict:
``safe_to_report_all_clear=false`` WITHHOLDS an all-clear (there is a reason the
empty result may not be clean); ``=true`` is framed as may-analysis, never a
proof of safety. Neither value ever asserts a bug -- findings are surfaced by
``reached_sinks``, not by this gate.

Unified diagnostics schema (attached to a zero-sink forward result under
``result["diagnostics"]``). #571's descriptive frontier fields are KEPT intact;
#562's honesty signals are FOLDED IN alongside them (one block, one renderer):

    {
      # --- #571 descriptive frontier (kept, unchanged) ---
      "source_callsites": int,       # matched source callsites the seed hit
      "tainted_values": int,         # tainted SSA values produced
      "last_use": {...} | None,      # last propagated use (label/address/reason)
      "unmodeled_calls_reached": bool,
      "frontier": {
        "unresolved": int,           # unmodeled_callee / arg_under_recovered / indirect
        "coarse_memory": int,        # coarse_memory_store / pointer_escape
        "seed_misanchored": int,     # #562: source_seed_misanchored / weak_buffer_seed
        "by_kind": {kind: count},
      },
      "next_action": str,            # single unified suggestion (see below)

      # --- #562 honesty gate (folded in) ---
      "safe_to_report_all_clear": bool,
      "all_clear_reason": str,
    }

Reconciliation notes:
  - #571's ``next_action`` (single string) is the ONE suggestion field. #562's
    ``suggested_next`` list is NOT re-introduced (no competing field); its
    seed-reseed guidance is folded into ``next_action`` (top priority) and the
    ``source_seed_misanchored`` leaf's own ``suggested_source``/``note``.
  - ``safe_to_report_all_clear`` lives INSIDE the zero-sink diagnostics block --
    the only context where "is this an all-clear?" is a live question. A run
    WITH findings has something to triage (``reached_sinks`` non-empty), so no
    diagnostics block and no gate is attached; the findings ARE the signal.
"""
from __future__ import annotations

from typing import Any

# Leaf kinds that mean "taint stopped somewhere attacker-relevant" -- an empty
# reached_sinks list carrying any of these is NEVER an all-clear. The seed-side
# honesty leaves (#562) are included so a weak/mis-anchored recv seed blocks the
# gate the same way a real frontier leaf does.
BLOCKING_LEAF_KINDS = frozenset({
    "coarse_memory_store",
    "pointer_escape",
    "indirect_call_unresolved",
    "unmodeled_callee",
    "arg_under_recovered",
    "field_load_unresolved",
    "source_seed_misanchored",
    "weak_buffer_seed",
})

# The seed-honesty leaf kinds specifically (subset of BLOCKING_LEAF_KINDS): a
# weak ``arg:<recv*>:N`` seed that did not key the filled buffer (#562/#306).
_SEED_MISANCHORED_KINDS = frozenset({"source_seed_misanchored", "weak_buffer_seed"})

# Substrings in ASSUMPTIONS that mean the seed was incomplete / wrong-shape even
# when no structured leaf was produced (e.g. the recvmsg scatter-gather nudge or
# a recvmsg_iovec_unresolved on the call: path). Kept specific so ordinary
# assumptions ("N callsites of recv; seeded from all") never trip the gate.
_WEAK_SEED_ASSUMPTION_MARKERS = (
    "scatter-gather",
    "payload taint is NOT followed",
    "recvmsg_iovec_unresolved",
    "recvmsg_out_param",
    "source_seed_misanchored",
    "weak_buffer_seed",
)

# Receive APIs whose ``arg:N`` seed is easy to mis-anchor (header/pointer vs the
# filled payload buffer). Used by :func:`misanchored_recv_leaf` callers.
_RECV_FAMILY = frozenset({
    "recv", "recvfrom", "recvmsg", "recvmmsg",
    "read", "pread", "fread",
})


def _assumption_has_weak_seed(assumptions: list[str]) -> bool:
    for a in assumptions:
        s = str(a)
        if any(m in s for m in _WEAK_SEED_ASSUMPTION_MARKERS):
            return True
    return False


def _derive_all_clear(
    leaves: list[dict[str, Any]],
    assumptions: list[str],
    *,
    truncated: bool,
    unmodeled_reached: bool = False,
) -> tuple[bool, str]:
    """The honesty claim gate for a ZERO-SINK forward result.

    Returns ``(safe_to_report_all_clear, reason)``. True ONLY when no tainted
    frontier leaf remains, no weak/mis-anchored-seed condition holds, taint did
    not reach an unmodeled call frontier, and the run was not truncated. Even
    True is a may-analysis (the reason says so) -- never a proof of safety. This
    function never asserts a vulnerability; a False value WITHHOLDS an
    all-clear, it does not claim a bug.

    ``unmodeled_reached`` is the caller-computed "taint reached an unmodeled
    call" signal (an unresolved in-binary callee counted in the frontier, OR an
    external callee disclosed only as a ``"...has no model; return
    conservatively tainted"`` assumption with NO leaf -- the most common
    frontier). The engine returns such a callee conservatively tainted, so
    analysis genuinely ESCAPED and an empty ``reached_sinks`` is not an
    all-clear even though no blocking leaf was emitted (#562).
    """
    blocking = [str(lf.get("kind")) for lf in leaves
                if lf.get("kind") in BLOCKING_LEAF_KINDS]
    weak_seed = _assumption_has_weak_seed(assumptions)

    if blocking:
        # De-dup preserving order for a readable reason.
        seen: list[str] = []
        for k in blocking:
            if k not in seen:
                seen.append(k)
        return False, (
            f"no modeled sink reached, but {len(blocking)} blocking frontier "
            f"leaf(s) ({', '.join(seen)}) remain -- NOT an all-clear")
    if leaves:
        return False, (
            f"no modeled sink reached, but {len(leaves)} frontier leaf(s) "
            f"remain -- NOT an all-clear")
    if weak_seed:
        return False, (
            "no sinks/leaves, but the source seed was incomplete or mis-anchored "
            "(see caveats) -- NOT an all-clear; reseed with --source call:<recv> "
            "or var:<buf>")
    if unmodeled_reached:
        return False, (
            "no modeled sink reached, but taint reached an unmodeled call "
            "frontier (an unresolved in-binary callee or an external callee with "
            "no taint model, returned conservatively tainted) -- analysis "
            "escaped there, NOT an all-clear")
    if truncated:
        return False, (
            "analysis truncated (depth/recursion) -- incomplete coverage, NOT an "
            "all-clear")
    return True, (
        "no modeled sink and no tainted frontier in the analyzed region; still a "
        "may-analysis -- not a proof of safety")


def forward_zero_diagnostics(sub: dict[str, Any], *, seed_callsites: int,
                             truncated: bool = False) -> dict[str, Any]:
    """Frontier diagnostics + honesty gate for a zero-sink forward run (#559/#562).

    Purely descriptive: seed reach (matched source callsites, tainted SSA
    values produced, last propagated use) plus the frontier the propagation
    stopped at (unresolved / coarse-memory / seed-misanchored leaf counts,
    whether any unmodeled external/in-binary call was reached) and a single
    suggested next action. Also folds in the honesty claim gate
    (``safe_to_report_all_clear`` + ``all_clear_reason``). Never asserts a
    vulnerability -- it explains "flow hit an unmodeled parser we couldn't
    follow" / "the seed did not key the buffer" vs "nothing flows"."""
    diag = sub.get("diag") or {}
    leaves = sub.get("leaves") or []
    assumptions = sub.get("assumptions") or []
    # ``truncated`` is a run-level flag (depth/recursion cutoff); the engine
    # threads it explicitly since the sub-result carries no stats block.
    truncated = bool(truncated or (sub.get("stats") or {}).get("truncated"))
    by_kind: dict[str, int] = {}
    for lf in leaves:
        k = str(lf.get("kind", "?"))
        by_kind[k] = by_kind.get(k, 0) + 1
    UNRESOLVED = ("unmodeled_callee", "arg_under_recovered", "indirect_call_unresolved")
    COARSE = ("coarse_memory_store", "pointer_escape")
    unresolved_n = sum(by_kind.get(k, 0) for k in UNRESOLVED)
    coarse_n = sum(by_kind.get(k, 0) for k in COARSE)
    seed_misanchored_n = sum(by_kind.get(k, 0) for k in _SEED_MISANCHORED_KINDS)
    # An external callee with no taint model returns conservatively tainted
    # and is disclosed as an assumption, not a leaf -- fold it into the
    # "unmodeled call reached" signal so a parser behind an import stub counts.
    ext_no_model = any("has no model" in a for a in assumptions)
    unmodeled_reached = bool(unresolved_n or ext_no_model)
    tainted_values = int(diag.get("tainted_values", 0))

    if seed_misanchored_n:
        # A weak/mis-anchored recv-family seed is the highest-priority footgun:
        # the pointer was tainted but the received buffer was not keyed, so an
        # empty result is a SEED failure, not a clean path.
        next_action = (
            "the --source arg seed did not key the received buffer (only the "
            "pointer value was tainted); reseed with --source call:<recv> (seeds "
            "the filled buffer/iovec payload) or --source var:<buf> so the "
            "payload is followed -- an empty result here means the seed was "
            "wrong, not that the path is clean")
    elif unmodeled_reached:
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

    safe, reason = _derive_all_clear(
        leaves, assumptions, truncated=truncated,
        unmodeled_reached=unmodeled_reached)

    return {
        "source_callsites": int(seed_callsites),
        "tainted_values": tainted_values,
        "last_use": diag.get("last_use"),
        "unmodeled_calls_reached": unmodeled_reached,
        "frontier": {
            "unresolved": unresolved_n,
            "coarse_memory": coarse_n,
            "seed_misanchored": seed_misanchored_n,
            "by_kind": by_kind,
        },
        "next_action": next_action,
        # #562 honesty gate, folded into the single diagnostics block.
        "safe_to_report_all_clear": safe,
        "all_clear_reason": reason,
    }


def misanchored_recv_leaf(
    *,
    callee: str,
    arg_index: int,
    address: str | None = None,
    reason: str = "buffer_not_keyed",
) -> dict[str, Any]:
    """Structured leaf for a weak ``arg:<recv*>:<n>`` seed (#306/#562 dogfood
    footgun): the pointer arg was tainted but the FILLED buffer was not keyed as
    a stable taint location, so payload stores/loads may be silently missed. A
    real taint-graph leaf -- it lands in ``result["leaves"]`` and feeds the
    frontier accounting -- so an empty ``reached_sinks`` with this leaf stops
    reading as a clean all-clear. This is a propagation/coverage fact, not a
    vulnerability claim."""
    base = (callee or "").split("@", 1)[0].lstrip("_")
    if base in ("recvmsg", "recvmmsg"):
        detail = (
            f"arg:{base}:{arg_index} seeds the msghdr*/msgvec pointer, not the "
            f"scatter-gather payload at msg_iov[i].iov_base -- buffer content is "
            f"NOT followed from this seed ({reason})")
        suggest = f"call:{base}"
    else:
        detail = (
            f"arg:{base}:{arg_index} could not key the filled buffer as a stable "
            f"taint location ({reason}); only the pointer value was seeded, so "
            f"payload stores/loads may be missed")
        suggest = f"call:{base}"
    leaf: dict[str, Any] = {
        "kind": "source_seed_misanchored",
        "callee": base,
        "arg_index": arg_index,
        "detail": detail,
        "suggested_source": suggest,
        "note": (
            f"NOT an all-clear -- reseed with --source {suggest} or var:<buf>; "
            f"empty reached_sinks with this leaf means the seed was wrong, not "
            f"that the path is clean"),
    }
    if address is not None:
        leaf["address"] = address
    return leaf
