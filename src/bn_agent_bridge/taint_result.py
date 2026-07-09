"""Post-process taint-engine results into agent-safe claim gates.

Kept free of Binary Ninja and of the main fixpoint engine so it stays
unit-testable. Sibling modules hold models / IL helpers / locators; this
file owns claim gates only. The engine calls :func:`enrich_forward_result`
on every forward result; text/JSON consumers read ``confidence`` +
``diagnostics`` without re-deriving the policy.
"""
from __future__ import annotations

from typing import Any

# Frontier kinds that mean "taint stopped somewhere attacker-relevant" — an
# empty reached_sinks list with any of these is NEVER an all-clear.
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

# Substrings in assumptions that mean the seed was incomplete / wrong shape.
_WEAK_SEED_ASSUMPTION_MARKERS = (
    "scatter-gather",
    "payload taint is NOT followed",
    "loaded indirectly",
    "not correlated with later re-loads",
    "recvmsg_iovec_unresolved",
    "recvmsg_out_param",
    "try --source call:",
    "Consider seeding the parser",
    "source_seed_misanchored",
    "weak_buffer_seed",
)

# Receive APIs whose arg:N seed is easy to mis-anchor (header vs payload).
_RECV_FAMILY = frozenset({
    "recv", "recvfrom", "recvmsg", "recvmmsg",
    "read", "pread", "fread",
})


def _leaf_kinds(leaves: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for lf in leaves:
        k = lf.get("kind")
        if k and k not in out:
            out.append(str(k))
    return out


def _assumption_has_weak_seed(assumptions: list[str]) -> bool:
    for a in assumptions:
        s = str(a)
        if any(m in s for m in _WEAK_SEED_ASSUMPTION_MARKERS):
            return True
    return False


def suggest_next_actions(
    *,
    sources: list[dict[str, Any]] | None,
    leaves: list[dict[str, Any]] | None,
    assumptions: list[str] | None,
    reached_sinks: list[dict[str, Any]] | None,
) -> list[str]:
    """Actionable next steps for an empty or frontier-only forward result."""
    actions: list[str] = []
    sinks = reached_sinks or []
    leaves = leaves or []
    sources = sources or []
    assumptions = assumptions or []
    kinds = set(_leaf_kinds(leaves))

    for src in sources:
        kind = src.get("kind")
        callee = str(src.get("callee") or "").split("@", 1)[0].lstrip("_")
        if kind == "arg" and callee in _RECV_FAMILY:
            base = callee
            actions.append(
                f"reseed with --source call:{base} (seeds every modeled output: "
                f"return + filled buffer / iovec payload), not arg:{base}:N alone")
            if base in ("recvmsg", "recvmmsg"):
                actions.append(
                    "arg:recvmsg:1 / arg:recvmmsg:1 seeds the msghdr*, not the "
                    "iovec payload — use call:recvmsg or var:<filled_buf>")
        if kind == "ret" and callee in _RECV_FAMILY:
            actions.append(
                f"also seed --source call:{callee} so output-pointer buffers are "
                f"tainted, not only the return length")

    if "coarse_memory_store" in kinds:
        actions.append(
            "audit coarse_memory_store leaves with decompile/disasm — hand-rolled "
            "copy/store loops never become modeled overflow sinks")
    if "pointer_escape" in kinds:
        actions.append(
            "chase pointer_escape destinations (descriptor/local that captured "
            "&tainted_buf) — re-loads are not followed automatically")
    if "indirect_call_unresolved" in kinds:
        actions.append(
            "pin indirect targets with --resolve-map "
            "'{\"<call_addr>\": [\"<target_addr>\"]}' and re-run, or seed "
            "param:0 on the resolved handler")
    if "source_seed_misanchored" in kinds or "weak_buffer_seed" in kinds:
        actions.append(
            "buffer content was not keyed; seed the parser entry with param:N "
            "or the filled buffer with var:<name>")

    if not sinks and not leaves and not actions:
        actions.append(
            "empty sinks+leaves is NOT proof of safety (may-analysis); try a "
            "different --source (call:<recv>), raise --max-depth, or walk sinks "
            "manually with bn xrefs / decompile")

    # Dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def derive_forward_confidence(
    *,
    reached_sinks: list[dict[str, Any]] | None = None,
    leaves: list[dict[str, Any]] | None = None,
    assumptions: list[str] | None = None,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Machine-readable claim gate for a forward-taint result.

    ``safe_to_report_all_clear`` is True only when no modeled sink was reached,
    no blocking frontier leaf exists, no weak-seed assumption is present, and
    the run was not truncated. Even then the engine remains a may-analysis —
    callers must still print the soundness disclaimer. Findings (non-empty
    sinks) always set the flag False (there is something to triage).
    """
    sinks = list(reached_sinks or [])
    leaves = list(leaves or [])
    assumptions = [str(a) for a in (assumptions or [])]
    stats = stats or {}
    blocking = [lf for lf in leaves if lf.get("kind") in BLOCKING_LEAF_KINDS]
    weak_seed = _assumption_has_weak_seed(assumptions)
    truncated = bool(stats.get("truncated"))

    if sinks:
        return {
            "safe_to_report_all_clear": False,
            "reason": (
                f"{len(sinks)} modeled sink(s) reached — triage findings; "
                f"not an all-clear"),
            "blocking_leaf_kinds": _leaf_kinds(blocking),
            "has_modeled_sinks": True,
            "has_blocking_frontier": bool(blocking),
            "weak_seed": weak_seed,
            "truncated": truncated,
        }

    if blocking:
        kinds = _leaf_kinds(blocking)
        return {
            "safe_to_report_all_clear": False,
            "reason": (
                f"no modeled sink reached, but {len(blocking)} blocking "
                f"frontier leaf(s) ({', '.join(kinds)}) — NOT an all-clear"),
            "blocking_leaf_kinds": kinds,
            "has_modeled_sinks": False,
            "has_blocking_frontier": True,
            "weak_seed": weak_seed,
            "truncated": truncated,
        }

    if leaves:
        return {
            "safe_to_report_all_clear": False,
            "reason": (
                f"no modeled sink reached, but {len(leaves)} frontier leaf(s) "
                f"remain — NOT an all-clear"),
            "blocking_leaf_kinds": [],
            "has_modeled_sinks": False,
            "has_blocking_frontier": False,
            "weak_seed": weak_seed,
            "truncated": truncated,
        }

    if weak_seed:
        return {
            "safe_to_report_all_clear": False,
            "reason": (
                "no sinks/leaves, but source seed was incomplete or mis-anchored "
                "(see assumptions) — NOT an all-clear; reseed with call:<recv>"),
            "blocking_leaf_kinds": [],
            "has_modeled_sinks": False,
            "has_blocking_frontier": False,
            "weak_seed": True,
            "truncated": truncated,
        }

    if truncated:
        return {
            "safe_to_report_all_clear": False,
            "reason": "analysis truncated (depth/recursion) — incomplete coverage",
            "blocking_leaf_kinds": [],
            "has_modeled_sinks": False,
            "has_blocking_frontier": False,
            "weak_seed": False,
            "truncated": True,
        }

    # Empty sinks + empty leaves + no weak seed: still not a *proof* of safety,
    # but there is no engine-visible frontier either. Flag true only as
    # "no modeled/frontier signal" — agents must still respect soundness.
    return {
        "safe_to_report_all_clear": True,
        "reason": (
            "no modeled sink and no tainted frontier in the analyzed region; "
            "still a may-analysis — not a proof of safety"),
        "blocking_leaf_kinds": [],
        "has_modeled_sinks": False,
        "has_blocking_frontier": False,
        "weak_seed": False,
        "truncated": False,
    }


def enrich_forward_result(result: dict[str, Any]) -> dict[str, Any]:
    """Attach ``confidence`` + ``diagnostics`` to a forward-taint result dict.

    Idempotent: re-running overwrites the same keys from current sinks/leaves.
    Mutates and returns *result*.
    """
    if not isinstance(result, dict) or result.get("direction") != "forward":
        return result

    sinks = list(result.get("reached_sinks") or [])
    leaves = list(result.get("leaves") or [])
    assumptions = list(result.get("assumptions") or [])
    stats = dict(result.get("stats") or {})
    sources = list(result.get("sources") or [])

    conf = derive_forward_confidence(
        reached_sinks=sinks, leaves=leaves, assumptions=assumptions, stats=stats)
    suggestions = suggest_next_actions(
        sources=sources, leaves=leaves, assumptions=assumptions, reached_sinks=sinks)

    diagnostics: dict[str, Any] = {
        "suggested_next": suggestions,
        "leaf_kinds": _leaf_kinds(leaves),
        "empty_result": (not sinks and not leaves),
    }
    if not sinks and not leaves:
        diagnostics["note"] = (
            "zero sinks and zero leaves — agents MUST NOT treat this as proof of "
            "safety; check confidence.safe_to_report_all_clear and suggested_next")
    elif not sinks and leaves:
        diagnostics["note"] = (
            "zero modeled sinks with non-empty frontiers — investigate leaves "
            "before claiming clean")

    result["confidence"] = conf
    result["diagnostics"] = diagnostics
    # Surface the gate at the top level for agents that only read one field.
    result["safe_to_report_all_clear"] = bool(conf.get("safe_to_report_all_clear"))
    return result


def misanchored_recv_leaf(
    *,
    callee: str,
    arg_index: int,
    address: str | None = None,
    reason: str = "buffer_not_keyed",
) -> dict[str, Any]:
    """Structured leaf for a weak ``arg:<recv*>:<n>`` seed (dogfood footgun)."""
    base = (callee or "").split("@", 1)[0].lstrip("_")
    if base in ("recvmsg", "recvmmsg"):
        detail = (
            f"arg:{base}:{arg_index} seeds the msghdr*/msgvec pointer, not the "
            f"scatter-gather payload at msg_iov[i].iov_base — buffer content is "
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
            f"NOT an all-clear — reseed with --source {suggest} or var:<buf>; "
            f"empty reached_sinks with this leaf means the seed was wrong, not that "
            f"the path is clean"),
    }
    if address is not None:
        leaf["address"] = address
    return leaf
