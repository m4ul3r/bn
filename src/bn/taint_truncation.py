"""The one vocabulary of taint truncation causes (``stats.truncation_cause``).

A truncated taint run names WHY it stopped short, and each cause has its own
next step: an interprocedural depth cutoff is fixed with ``--max-depth``, an
unconverged fixpoint with ``--max-iters``, and every internal search budget
with nothing but a narrower seed. Two renderers used to keep their own partial
copies -- the CLI verdict line (``formatters._taint_truncation_note``) and the
bridge's all-clear reason (``taint_result._truncation_hint``) -- and both fell
back to a depth wording for any cause they did not know, so every budget the
global-memory continuation added was reported as ``truncated @depth N`` and
sent the reader to ``--max-depth``, the one knob that cannot move it.

This module is the single table both read. ``tests/test_taint_truncation_vocabulary.py``
enumerates every cause literal the engine can record and fails when one is
missing here. Symlinked into ``src/bn_agent_bridge/`` like ``paths.py``, so the
bridge and the CLI cannot drift; stdlib-only for the same reason.
"""
from __future__ import annotations

from typing import NamedTuple


class TruncationCause(NamedTuple):
    """``what`` stopped the run, and the one ``remedy`` that addresses it."""

    what: str
    remedy: str


# The remedy for a bound no flag reaches. Saying so is the point: the reader
# otherwise reaches for the only knob in sight, `--max-depth`.
INTERNAL_BUDGET = "internal budget, no flag raises it; narrow the seed"

TRUNCATION_CAUSES: dict[str, TruncationCause] = {
    "max_depth": TruncationCause(
        "interprocedural depth bound",
        "raise --max-depth"),
    "fixpoint_exhausted": TruncationCause(
        "fixpoint unconverged (intra-function iteration budget exhausted)",
        "raise --max-iters or narrow the source"),
    "recursion": TruncationCause(
        "recursion limit (possible unresolved cycle)",
        "internal limit, no flag raises it; what was found before the unwind is "
        "reported -- narrow the seed (see the analysis_truncated frontier)"),
    "caller_cap": TruncationCause(
        "caller-site cap reached (not every caller was followed)",
        "internal cap, no flag raises it; narrow the sink or follow the sites "
        "named in the caller_sites_truncated frontier"),
    "memory_chain": TruncationCause(
        "memory-chain budget exhausted while proving a local buffer's writers "
        "(discovery or fixed-point solve unfinished)",
        "internal budget, no flag raises it; seed closer to the sink -- see the "
        "address_of_local_unresolved frontier"),
    "abi_provenance_budget": TruncationCause(
        "argument-register provenance scan ran out of its block budget before "
        "a call",
        "internal budget, no flag raises it; narrow the seed, or read the "
        "call's disassembly for the arguments it passes"),
    "global_reference_budget": TruncationCause(
        "search for code reading a tainted global exhausted its function or "
        "reference-sweep budget",
        "internal budget, no flag raises it; audit the references the "
        "global_memory_continuation frontier names, or seed inside a consumer"),
    "global_object_extent": TruncationCause(
        "a tainted global's object extent is unproven",
        "retype the global as the global_memory_continuation frontier advises "
        "(`data retype`), or narrow the seed"),
    "global_object_search": TruncationCause(
        "search for the object enclosing a tainted global write ran out",
        INTERNAL_BUDGET),
    "global_object_layout": TruncationCause(
        "walk to a tainted global's next named neighbour ran out",
        INTERNAL_BUDGET),
    "global_veneer_chain": TruncationCause(
        "thunk/veneer chain toward a tainted global's reader exceeded its bound",
        INTERNAL_BUDGET),
    "pointer_scan_budget": TruncationCause(
        "one function's pointer-use scan ran out of nodes",
        INTERNAL_BUDGET),
    "pointer_return_depth": TruncationCause(
        "pointer followed back through returning wrappers past the internal "
        "chain depth (not --max-depth)",
        INTERNAL_BUDGET),
    "parameter_scan_budget": TruncationCause(
        "write-only-parameter scan ran out of hops",
        INTERNAL_BUDGET),
    "recurrence_search_budget": TruncationCause(
        "search for whether a loop carries a literal back to its phi ran out; "
        "the literal is reported as a recurrence_initializer it could not rule out",
        INTERNAL_BUDGET),
}


def describe_truncation_cause(cause: str) -> str:
    """``what -- remedy`` for a known cause; an unknown one VERBATIM.

    A cause this table does not know comes from a bridge newer than this
    vocabulary. It is named as itself and pointed at the JSON -- never
    rendered as a depth cutoff, which is the misreport this table exists to
    end."""
    entry = TRUNCATION_CAUSES.get(cause)
    if entry is None:
        return (f"{cause} (unrecognised truncation cause; see "
                "stats.truncation_cause in --format json)")
    return f"{entry.what} -- {entry.remedy}"
