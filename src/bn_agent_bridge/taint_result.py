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
    # #206: a tainted value reached an unlifted/unimplemented instruction, so
    # BN's lifter could not model the op and propagation through it is a silent
    # hole -- an empty reached_sinks past it is NOT an all-clear. Emitted
    # flow-sensitively (only when taint actually reaches such an instruction),
    # so a function that merely CONTAINS unlifted ops does not block the gate.
    "unlifted_instruction_reached",
    # #810: a backward ascent followed only the first N caller sites of a
    # parameter-origin slice; origins reachable only from the dropped callers are
    # absent from the result. Same class as the other frontier leaves: the walk
    # stopped with data unexamined, so the slice is NOT a complete answer.
    "caller_sites_truncated",
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
    # #851: a scanf-family call with more actual destinations than the model's
    # fixed unrolled run; the residual args are unseeded and the all-clear is
    # not safe to report.
    "scanf_arity_residual",
    # #863: a model that declines to claim its callee's destinations for a
    # STRUCTURAL reason (vsscanf writes through a va_list the engine cannot
    # resolve) rather than an arity one. The arity residual above cannot see it
    # -- there is no modeled `*arg:N` run to out-length -- so without this marker
    # a run whose taint reached such a call reported an all-clear that meant "no
    # flow" when the truth was "the model never claimed those destinations".
    "destinations_unmodeled",
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


def _truncation_hint(truncation_cause: list[str] | None) -> str:
    """Human remediation clause for a truncated run, keyed by cause (#579/#576).

    The two causes need DIFFERENT next actions -- a fixpoint that exhausted its
    per-function iteration budget is fixed with ``--max-iters``, while an
    interprocedural depth cutoff is fixed with ``--max-depth`` -- so a consumer
    that saw only "depth/recursion" was told the wrong remediation."""
    causes = list(truncation_cause or [])
    if "fixpoint_exhausted" in causes:
        return ("intra-function fixpoint exhausted its iteration budget before "
                "converging -- raise --max-iters or narrow the source")
    if "max_depth" in causes:
        return "interprocedural descent hit the depth bound -- raise --max-depth"
    if "recursion" in causes:
        return "Python recursion limit reached (possible unresolved cycle)"
    if "caller_cap" in causes:
        # #812: backward-only cause. Without this branch it fell through to the
        # generic "depth/recursion cutoff" string, which names the wrong knob --
        # no depth or recursion bound was hit, the caller ascent stopped at its
        # per-site cap with callers unexamined.
        return ("the backward caller ascent hit its per-site cap -- some calling "
                "sites were never followed; narrow the sink or inspect the "
                "capped sites named in the caller_sites_truncated leaf")
    return "depth/recursion cutoff"


def _derive_all_clear(
    leaves: list[dict[str, Any]],
    assumptions: list[str],
    *,
    truncated: bool,
    unmodeled_reached: bool = False,
    truncation_cause: list[str] | None = None,
    analysis_incomplete: bool = False,
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
    if analysis_incomplete:
        # #811: a callee whose MLIL was missing mid-run (a partially analysed
        # view, or a function BN had not finished lifting) is caught by
        # _summarize, which conservatively taints its return and records a prose
        # assumption. Prose no consumer can gate on is exactly how the caller-cap
        # under-disclosure started, so the condition is threaded structurally and
        # withholds the all-clear here: the body was never read, so "no sink in
        # it" is an absence of evidence, not evidence of absence.
        return False, (
            "no modeled sink reached, but one or more callee bodies could not be "
            "analysed (MLIL unavailable -- the view may be partially analysed); "
            "their contents were never examined, NOT an all-clear")
    if truncated:
        return False, (
            f"analysis truncated ({_truncation_hint(truncation_cause)}) -- "
            "incomplete coverage, NOT an all-clear")
    return True, (
        "no modeled sink and no tainted frontier in the analyzed region; still a "
        "may-analysis -- not a proof of safety")


def forward_zero_diagnostics(sub: dict[str, Any], *, seed_callsites: int,
                             truncated: bool = False,
                             truncation_cause: list[str] | None = None,
                             analysis_incomplete: bool = False) -> dict[str, Any]:
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
    stats = sub.get("stats") or {}
    truncated = bool(truncated or stats.get("truncated"))
    truncation_cause = list(truncation_cause or stats.get("truncation_cause") or [])
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
        unmodeled_reached=unmodeled_reached, truncation_cause=truncation_cause,
        analysis_incomplete=analysis_incomplete)

    return {
        "source_callsites": int(seed_callsites),
        "tainted_values": tainted_values,
        "last_use": diag.get("last_use"),
        "unmodeled_calls_reached": unmodeled_reached,
        # Additive: the distinct truncation cause(s), so a zero-sink consumer
        # sees WHY coverage was incomplete and the matching remediation (#579/#576).
        "truncated": truncated,
        "truncation_cause": truncation_cause,
        # #811: at least one callee body could not be analysed during this run,
        # so the region the result speaks for is smaller than it looks.
        "analysis_incomplete": bool(analysis_incomplete),
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


def backward_diagnostics(
    leaves: list[dict[str, Any]],
    assumptions: list[str],
    *,
    sinks_seeded: int,
    slices: int,
    truncated: bool = False,
    truncation_cause: list[str] | None = None,
) -> dict[str, Any]:
    """Frontier diagnostics + completeness gate for a BACKWARD run (#812).

    Forward attached a diagnostics block and backward did not, so the two
    directions disagreed about whether a caller could tell an exhaustive answer
    from a curtailed one: a backward slice that dropped callers at the ascent
    cap or bottomed out at an unresolved field load came back in the same shape
    as a slice that reached every origin.

    The gate here is deliberately NOT ``safe_to_report_all_clear``. That key
    answers forward's question -- "no modeled sink was reached" -- and a
    backward run never asks it: it starts AT a sink and walks toward origins, so
    reusing the name would attach forward's meaning to a different claim. What
    backward's own state can actually support is whether the def-chain was
    followed to its origins with nothing dropped, which is what
    ``safe_to_report_complete_slice`` reports. A False value withholds a
    completeness claim; like every gate in this module it never asserts a bug.

    Emitted unconditionally (forward's block is zero-sink only) because a
    backward run has no "found something, so the findings are the signal" case:
    its slices ARE the result, and their completeness is a live question whether
    there are none, one, or many.
    """
    by_kind: dict[str, int] = {}
    for lf in leaves or []:
        k = str(lf.get("kind", "?"))
        by_kind[k] = by_kind.get(k, 0) + 1
    # Same vocabulary the forward frontier groups by, so one leaf kind never
    # means two things across directions. `caller_sites_truncated` is counted
    # under its own heading: it is not an unresolved callee or a coarse store
    # but a deliberately abandoned ascent, and conflating it would hide the one
    # frontier a user can act on by re-running with a narrower sink.
    UNRESOLVED = ("unmodeled_callee", "arg_under_recovered",
                  "indirect_call_unresolved", "field_load_unresolved")
    COARSE = ("coarse_memory_store", "pointer_escape")
    unresolved_n = sum(by_kind.get(k, 0) for k in UNRESOLVED)
    coarse_n = sum(by_kind.get(k, 0) for k in COARSE)
    dropped_callers_n = by_kind.get("caller_sites_truncated", 0)

    blocking = [str(lf.get("kind")) for lf in (leaves or [])
                if lf.get("kind") in BLOCKING_LEAF_KINDS]
    weak_seed = _assumption_has_weak_seed(assumptions or [])
    truncation_cause = list(truncation_cause or [])

    if blocking:
        seen: list[str] = []
        for k in blocking:
            if k not in seen:
                seen.append(k)
        complete, reason = False, (
            f"{len(blocking)} frontier leaf(s) ({', '.join(seen)}) remain -- the "
            "def-chain was not followed to every origin, so this slice is NOT a "
            "complete account of what reaches the sink")
    elif truncated:
        complete, reason = False, (
            f"analysis truncated ({_truncation_hint(truncation_cause)}) -- "
            "origins behind the cut are absent, NOT a complete slice")
    elif weak_seed:
        complete, reason = False, (
            "the sink seed was incomplete or mis-anchored (see caveats), so the "
            "walk may have started from the wrong value -- NOT a complete slice")
    elif not sinks_seeded:
        complete, reason = False, (
            "no sink seeded, so nothing was walked -- an empty slice list here "
            "is a seeding outcome, not a complete answer")
    else:
        complete, reason = True, (
            "every seeded sink was walked to its origins with no frontier leaf "
            "and no truncation; still a may-analysis over the recovered IL -- "
            "not a proof that no other value reaches the sink")

    if dropped_callers_n:
        next_action = (
            "the caller ascent was capped, so some calling sites were never "
            "followed; re-run against a specific caller or narrow the sink to "
            "see the origins behind the dropped sites")
    elif unresolved_n:
        next_action = (
            "the walk bottomed out at an unresolved def (an indirect call or a "
            "field load the engine could not key); recover the callee prototype "
            "with `bn proto set` or seed inside the producing function")
    elif coarse_n:
        next_action = (
            "the walk crossed a coarse-memory frontier (a pointer/store not "
            "precisely tracked); inspect the frontier leaves or re-seed on the "
            "destination buffer directly")
    elif not sinks_seeded:
        # Must precede the no-slices branch: with nothing seeded there are also
        # no slices, so the generic "the sink seeded but no slice was produced"
        # fired and contradicted this same block's own
        # `complete_slice_reason` ("no sink seeded, so nothing was walked").
        # Two lines of one diagnostic disagreeing about whether a sink seeded is
        # worse than either line alone (found in cross-dogfood).
        next_action = (
            "no sink seeded, so nothing was walked; check that the --sink "
            "locator names a call this function actually makes and an operand "
            "that reads a variable")
    elif not slices:
        next_action = (
            "the sink seeded but no slice was produced; confirm the --sink "
            "locator names the operand you meant")
    else:
        next_action = (
            "the slice reached its origins; classify each origin (parameter / "
            "modeled source / constant) to decide whether the sink is "
            "attacker-reachable")

    return {
        "sinks_seeded": int(sinks_seeded),
        "slices": int(slices),
        "truncated": bool(truncated),
        "truncation_cause": truncation_cause,
        "frontier": {
            "unresolved": unresolved_n,
            "coarse_memory": coarse_n,
            "dropped_callers": dropped_callers_n,
            "by_kind": by_kind,
        },
        "next_action": next_action,
        "safe_to_report_complete_slice": complete,
        "complete_slice_reason": reason,
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


def indirect_pointer_slot_leaf(
    *,
    callee: str,
    arg_index: int,
    address: str | None = None,
    slot: Any = None,
) -> dict[str, Any]:
    """Structured leaf for an indirect-load buffer-pointer arg seed (#193/#562):
    the source's buffer pointer is itself loaded from a global/struct slot
    (``recvfrom(fd, *(ctx+off), n)``), so the seed anchors to the loaded POINTER
    value, not the pointee -- and the engine did not correlate it with a later
    re-load of the same slot. A flow that re-loads the pointer and parses the
    payload may therefore be silently missed. Kind ``source_seed_misanchored``
    (the seed anchored the pointer, not the payload -- exactly a mis-anchored
    seed), so it lands in ``result["leaves"]`` / ``frontier.seed_misanchored``
    and the claim gate WITHHOLDS an all-clear the same way the recv-buffer case
    does. A propagation/coverage fact, not a vulnerability claim."""
    base = (callee or "").split("@", 1)[0].lstrip("_")
    detail = (
        f"arg:{base}:{arg_index} buffer pointer is loaded indirectly (from a "
        f"global/struct slot); the seed anchors to the pointer value, not the "
        f"pointee, and was not correlated with a later re-load of the same slot "
        f"-- a flow that re-loads the pointer and parses it may be missed")
    leaf: dict[str, Any] = {
        "kind": "source_seed_misanchored",
        "callee": base,
        "arg_index": arg_index,
        "detail": detail,
        "suggested_source": f"param:N (seed the parser entry directly)",
        "note": (
            "NOT an all-clear -- the indirect-load pointer slot was not "
            "correlated forward; seed the parser entry directly with param:N "
            "or the filled buffer with --source var:<buf>"),
    }
    if slot is not None:
        leaf["slot"] = str(slot)
    if address is not None:
        leaf["address"] = address
    return leaf
