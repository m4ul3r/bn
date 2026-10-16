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

import shlex
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
    # #1041 item 4 / #1050 review finding 4: a scanf-family call whose format
    # is not a read-only constant (or uses %n$ positions) may write an
    # unbounded token; its conversions could not be checked, so an empty
    # reached_sinks past it is not an all-clear.
    "format_unchecked",
    # #1041 item 1 gate (#1050 review round 4): an indirect call whose callee
    # bn did not resolve can run after a call that fills a buffer. Which
    # pointer it receives is not worked out; what it does with the bytes is
    # unchecked, so an empty reached_sinks is not an all-clear.
    "indirect_call_unchecked",
    # #810: a backward ascent followed only the first N caller sites of a
    # parameter-origin slice; origins reachable only from the dropped callers are
    # absent from the result. Same class as the other frontier leaves: the walk
    # stopped with data unexamined, so the slice is NOT a complete answer.
    "caller_sites_truncated",
    # A backward slice reached `p = &local_buf` (or a folded `sink(&buf)`) and
    # the walk could not account for the buffer's CONTENTS: no memory version
    # to walk, an unmodeled call took the buffer's address or ran after it
    # escaped to memory, a path to the sink with no writer at all (a
    # conditional fill), a writer's input the walk could not interpret, or a
    # memory chain longer than the walk budget. The sink consumes the bytes,
    # not the pointer, so a slice that stops at the address-of is NOT complete.
    "address_of_local_unresolved",
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
    if "memory_chain" in causes:
        # A cut may interrupt discovery or a fixed-point sweep. Existing
        # writer rows need not be missing for the proof to be incomplete.
        return ("the backward proof through a local buffer's writers exhausted "
                "its shared memory-chain budget -- graph discovery or the "
                "coverage/escape solve is unfinished; seed closer to the sink "
                "or inspect the writes and unresolved calls named in the "
                "address_of_local_unresolved leaf")
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
    if any("tokenizer_state_escapes" in str(a) for a in assumptions):
        # A `continuation` model (strtok) left its static state tainted when
        # the function returned; a `strtok(NULL, ..)` in another function
        # continues it and is not followed, so the empty result speaks only
        # for this function's own tokens (PR-928 review P2).
        return False, (
            "no modeled sink reached, but a tokenizer's static state is left "
            "tainted at return (see the tokenizer_state_escapes caveat) -- a "
            "continuation in another function is not followed, NOT an all-clear")
    if weak_seed:
        return False, (
            "no sinks/leaves, but the source seed was incomplete or mis-anchored "
            "(see caveats) -- NOT an all-clear")
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


def command_template(command: str) -> str:
    """Mark printed advice that bn cannot complete (#1033 item 6).

    Advice in backticks is pasted as-is, so a command missing an argument the
    result does not know must say so rather than look runnable. The unknown
    parts are UPPER-CASE words (``VAR``, ``N``) -- never ``<name>``, which a
    shell reads as a redirection before bn ever parses the line -- and the
    marker below precedes the backticks so a reader (and the test that pastes
    every printed command) can tell a template from a command."""
    return f"template, fill in the UPPER-CASE words: `{command}`"


def _forward_next_action(leaves: list[dict[str, Any]], *, by_kind: dict[str, int],
                         unmodeled_reached: bool, ext_no_model: bool,
                         tainted_values: int, function: str | None) -> str:
    """The single suggested next step for a zero-sink forward run.

    Every command it prints in backticks either runs as printed (the function
    and source are filled in from the result) or is a :func:`command_template`
    (#1033 item 6). ``function`` is the analysed function's address; without it
    nothing can be filled in and the advice stays prose."""
    fn = function

    def forward(fn_ref: str, source: str) -> str:
        return f"bn taint forward -f {shlex.quote(fn_ref)} --source {shlex.quote(source)}"

    UNRESOLVED = ("unmodeled_callee", "arg_under_recovered", "indirect_call_unresolved")
    unresolved = [lf for lf in leaves if str(lf.get("kind", "?")) in UNRESOLVED]
    misanchored = [lf for lf in leaves if lf.get("kind") in _SEED_MISANCHORED_KINDS]
    if misanchored:
        # A weak/mis-anchored seed is the highest-priority footgun: the pointer
        # was tainted but the received buffer was not keyed, so an empty result
        # is a SEED failure, not a clean path. Only a leaf that names a source
        # that keys MORE (arg:recvmsg -> call:recvmsg) earns a reseed command;
        # an unkeyable buffer fails the same way under arg: and call: (#1041
        # item 1), so recommending the other locator would loop the agent.
        reseed = next((str(lf["suggested_source"]) for lf in misanchored
                       if str(lf.get("suggested_source") or "").startswith("call:")), None)
        head = ("the seed tainted a buffer pointer but did not key the buffer it "
                "fills, so payload stores/loads were not followed, and indirect "
                "calls that may receive this buffer are not followed -- their "
                "callees are unchecked; an empty result here means the seed was "
                "wrong, not that the path is clean")
        if reseed:
            run = (f"`{forward(fn, reseed)}`" if fn else f"--source {reseed}")
            return (f"{head}; reseed with {run}, which seeds the iovec buffer this "
                    f"function builds")
        if any(lf.get("reason") == "msghdr_not_payload" for lf in misanchored):
            # #1050 review finding 5: call:recvmsg resolves the iovec only when
            # this function builds it; here it does not, so that reseed would
            # seed nothing and is not offered.
            seed = (f" ({command_template(forward(fn, 'var:VAR'))})" if fn else "")
            return (f"{head}. The iovec is not built in this function, so "
                    f"call:recvmsg cannot resolve its buffer either; seed the filled "
                    f"buffer directly{seed}, or run taint from the caller that builds "
                    f"the iovec")
        if fn:
            return (f"{head}. Reseeding with arg: or call: keys the same pointer and "
                    f"fails the same way; seed a value read from the buffer instead "
                    f"({command_template(forward(fn, 'var:VAR'))}) or the parser that "
                    f"consumes it ({command_template(forward('PARSER', 'param:N'))})")
        return (f"{head}. Reseeding with arg: or call: keys the same pointer and "
                f"fails the same way; seed a value read from the buffer (var:) or "
                f"the consuming parser's parameter (param:) instead")
    callbacks = [lf for lf in leaves if lf.get("kind") == "indirect_call_unchecked"]
    if callbacks and all(lf.get("kind") in ("indirect_call_unchecked", "format_unchecked")
                         for lf in leaves):
        at = ", ".join(sorted({str(lf.get("address")) for lf in callbacks}))
        return ("no sink reached, but indirect calls that may receive this buffer are "
                f"not followed -- their callees are unchecked ({at}); resolve those "
                "targets (a value set, or --resolve-map) and re-run, or audit the "
                "callees by hand")
    unchecked = [lf for lf in leaves if lf.get("kind") == "format_unchecked"]
    if unchecked and len(unchecked) == len(leaves):
        at = ", ".join(sorted({str(lf.get("address")) for lf in unchecked}))
        return ("no sink reached, but a scanf-family call's format could not be read as "
                f"a read-only constant ({at}), so whether it writes an unbounded token "
                "is unknown -- inspect that call's format operand")
    if unmodeled_reached and not ext_no_model and all(
            lf.get("cause") == "varargs_not_followed" for lf in unresolved):
        # #948: every unresolved frontier is taint entering the `...` tail of a
        # callee BN already types variadic. Prototype repair is the wrong
        # advice there: the declaration is correct, the engine just does not
        # map varargs onto anything -- and no param:N names the tail, so the
        # seed inside it is a template.
        callee = ((unresolved[0].get("callee") or {}) if unresolved else {})
        inside = callee.get("address") if isinstance(callee, dict) else None
        seed = (f" ({command_template(forward(str(inside), 'var:VAR'))})"
                if inside else "")
        return ("taint reached the variadic tail (`...`) of a callee already declared "
                "variadic, which the engine does not follow -- its prototype is correct, "
                "so do not redeclare it; inspect how the callee consumes its va_list or "
                f"seed taint inside it from the value it reads{seed}")
    if unmodeled_reached and not ext_no_model and all(
            lf.get("cause") == "signature_under_recovered" for lf in unresolved):
        callee = ((unresolved[0].get("callee") or {}) if unresolved else {})
        at = callee.get("address") if isinstance(callee, dict) else None
        fix = (f" ({command_template(f'bn proto set {at} PROTOTYPE')})" if at else "")
        return ("confirmed extra arguments disagree with a determined nonvariadic "
                f"declaration; recover the actual callee prototype with proto set{fix} "
                "and re-run, or seed inside the callee to inspect the incomplete flow")
    if unmodeled_reached:
        # An in-binary callee the walk did not descend (depth bound, recursion,
        # unanalysable body) and whose tainted argument is a named parameter
        # can be seeded exactly: param:<that index> inside it. Leaves with a
        # `cause` are the past-the-arity cases, where no param:N exists.
        inside = None
        for lf in unresolved:
            callee = lf.get("callee")
            args = lf.get("tainted_args") or []
            if (lf.get("kind") == "unmodeled_callee" and not lf.get("cause")
                    and isinstance(callee, dict) and callee.get("address") and args):
                inside = forward(str(callee["address"]), f"param:{min(args)}")
                break
        seed = (f"re-run seeded inside the first one with `{inside}`" if inside
                else "seed taint inside the callee")
        return ("taint reached an unmodeled call frontier; inspect the frontier leaves "
                f"and callee argument consumption, or {seed}. Repair a prototype only "
                "where confirmed extra arguments disagree with a determined "
                "nonvariadic declaration")
    if by_kind.get("coarse_memory_store", 0) or by_kind.get("pointer_escape", 0):
        # The coarse leaf exists precisely because the destination could not be
        # keyed, so no var: names it; the template seeds a value read back.
        seed = (f" ({command_template(forward(fn, 'var:VAR'))})" if fn else "")
        return ("taint escaped through a coarse-memory frontier (a pointer/store not "
                "precisely tracked); inspect the frontier leaves or seed a value read "
                f"back from those bytes{seed}")
    if tainted_values <= 1:
        return ("the source seeded but produced no further tainted uses; verify the "
                "`--source` locator matches the intended value")
    return ("taint propagated but reached no modeled sink and hit no unresolved "
            "frontier; the flow appears to dead-end locally -- inspect the last "
            "propagated use or widen `--max-depth`")


def forward_zero_diagnostics(sub: dict[str, Any], *, seed_callsites: int,
                             truncated: bool = False,
                             truncation_cause: list[str] | None = None,
                             analysis_incomplete: bool = False,
                             function: str | None = None) -> dict[str, Any]:
    """Frontier diagnostics + honesty gate for a zero-sink forward run (#559/#562).

    Purely descriptive: seed reach (matched source callsites, tainted SSA
    values produced, last propagated use) plus the frontier the propagation
    stopped at (unresolved / coarse-memory / seed-misanchored leaf counts,
    whether any unmodeled external/in-binary call was reached) and a single
    suggested next action. Also folds in the honesty claim gate
    (``safe_to_report_all_clear`` + ``all_clear_reason``). Never asserts a
    vulnerability -- it explains "flow hit an unmodeled parser we couldn't
    follow" / "the seed did not key the buffer" vs "nothing flows".

    ``function`` is the analysed function's address, used to fill in the
    commands the next action prints (#1033 item 6)."""
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

    next_action = _forward_next_action(
        leaves, by_kind=by_kind, unmodeled_reached=unmodeled_reached,
        ext_no_model=ext_no_model, tainted_values=tainted_values, function=function)

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
    unresolved_origins: int = 0,
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

    ``unresolved_origins`` is the number of slices whose ``origin.kind`` is
    ``unresolved`` -- the walk's own admission that the def-chain bottomed out
    at a definition it could not classify (an ``&local`` it did not follow, a
    load it could not key). That admission lives on the SLICE, not in
    ``leaves``, so a leaf-only gate answered ``safe_to_report_complete_slice:
    true`` beside ``traverses_unresolved: true`` and ``frontier.unresolved: 0``
    on the same result. It is counted in ``frontier.unresolved`` and withholds
    the claim like any blocking leaf.
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
                  "indirect_call_unresolved", "field_load_unresolved",
                  "address_of_local_unresolved")
    COARSE = ("coarse_memory_store", "pointer_escape")
    unresolved_n = sum(by_kind.get(k, 0) for k in UNRESOLVED) + int(unresolved_origins)
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
    elif unresolved_origins:
        complete, reason = False, (
            f"{int(unresolved_origins)} slice(s) bottomed out at an unresolved "
            "definition (origin.kind == 'unresolved') -- the def-chain was not "
            "followed to its origin, so this slice is NOT a complete account of "
            "what reaches the sink")
    elif truncated:
        complete, reason = False, (
            f"analysis truncated ({_truncation_hint(truncation_cause)}) -- "
            "origins behind the cut are absent, NOT a complete slice")
    # DEFENCE-IN-DEPTH, and deliberately kept as such: no assumption today's
    # backward walk records contains any `_WEAK_SEED_ASSUMPTION_MARKERS`
    # substring (measured over the six it can emit), so this cannot fire from
    # the only caller -- the weak backward seed shape that does exist arrives as
    # a BLOCKING `arg_under_recovered` leaf and is withheld above. It stays
    # because the marker set is shared with forward and keeps growing (#851 and
    # #863 each added one), and a gate that silently stops covering a shape is
    # the failure this whole block exists to prevent. Pinned directly by
    # `test_backward_weak_sink_seed_withholds_completeness_812` rather than
    # through a run that would not prove it exists.
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
    elif truncated and truncation_cause:
        # A named cut (the memory-chain budget, a depth bound) has its own
        # remediation; the generic unresolved text below would send the reader
        # to `bn proto set` for a walk that simply ran out of budget (round 3).
        next_action = _truncation_hint(truncation_cause)
    elif unresolved_n:
        unresolved_leaves = [lf for lf in leaves or [] if lf.get("kind") in UNRESOLVED]
        next_action = (
            "the walk bottomed out at an unresolved definition; inspect the frontier "
            "leaves and argument consumption, seed inside the producing function, "
            "or re-run `bn taint forward --source var:<buf>` on the buffer")
        if any(lf.get("kind") == "address_of_local_unresolved" and lf.get("callee")
               for lf in unresolved_leaves):
            next_action += (
                ". A known call may have written the local buffer; inspect and model "
                "that call's buffer writes and input propagation, or seed inside it")
        elif unresolved_leaves and not unresolved_origins and all(
                lf.get("cause") == "varargs_not_followed" for lf in unresolved_leaves):
            next_action += (
                ". The callee is already declared variadic; the engine does not follow "
                "its variadic tail. Inspect its va_list consumption rather than "
                "redeclaring the prototype")
        elif unresolved_leaves and not unresolved_origins and all(
                lf.get("cause") == "signature_under_recovered" for lf in unresolved_leaves):
            next_action += (
                ". The callee is determined nonvariadic; apply `bn proto set` only "
                "where independent argument evidence demonstrates a declaration "
                "discrepancy, then re-run")
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
    seed: str = "arg",
    call_resolves_iovec: bool = True,
) -> dict[str, Any]:
    """Structured leaf for a seed that tainted a buffer POINTER but did not key
    the FILLED buffer as a stable taint location (#306/#562 dogfood footgun), so
    payload stores/loads may be silently missed. A real taint-graph leaf -- it
    lands in ``result["leaves"]`` and feeds the frontier accounting -- so an
    empty ``reached_sinks`` with this leaf stops reading as a clean all-clear.
    This is a propagation/coverage fact, not a vulnerability claim.

    ``seed`` names the locator kind that produced it: ``arg`` (an
    ``arg:<callee>:N`` source) or ``call`` (the ``call:<callee>`` preset, whose
    modeled ``*arg:N`` output could not be keyed, #1041 item 1). Only the
    recvmsg header case has a reseed that keys more -- ``call:`` resolves the
    iovec payload -- and only when the engine's own iovec resolver found and
    keyed a buffer at this call (``call_resolves_iovec``, #1050 review finding
    5); otherwise that reseed seeds nothing and is not suggested. For
    ``buffer_not_keyed`` both locators resolve the buffer through the same
    pointer, so ``call:`` would fail the same way and is never suggested
    (``suggested_source`` is None)."""
    base = (callee or "").split("@", 1)[0].lstrip("_")
    locator = f"call:{base}" if seed == "call" else f"arg:{base}:{arg_index}"
    suggest: str | None
    if base in ("recvmsg", "recvmmsg") and seed != "call":
        detail = (
            f"arg:{base}:{arg_index} seeds the msghdr*/msgvec pointer, not the "
            f"scatter-gather payload at msg_iov[i].iov_base -- buffer content is "
            f"NOT followed from this seed ({reason})")
        if call_resolves_iovec:
            suggest = f"call:{base}"
            note = (f"NOT an all-clear -- reseed with --source {suggest}; empty "
                    f"reached_sinks with this leaf means the seed was wrong, not that "
                    f"the path is clean")
        else:
            suggest = None
            note = (f"NOT an all-clear -- the iovec is not built in this function, so "
                    f"call:{base} cannot resolve its buffer either; seed the filled "
                    f"buffer with var:, or run taint from the caller that builds it")
    else:
        # #1041 item 1, narrowed after #1050 review round 3: whether a callback
        # later receives this same pointer cannot be decided soundly from the
        # SSA copies (aliases made before the call, reloads), so it is disclosed
        # unconditionally rather than claimed for some shapes.
        detail = (
            f"{locator} could not key the buffer arg {arg_index} points at as a "
            f"stable taint location ({reason}); only the pointer value was "
            f"seeded, so payload stores/loads may be missed, and indirect calls "
            f"that may receive this buffer are not followed -- their callees are "
            f"unchecked")
        suggest = None
        note = (
            "NOT an all-clear -- the filled buffer is reached through a pointer "
            "the engine cannot key (e.g. a struct field plus an offset), and "
            "arg: and call: seeds key it the same way; seed a value read from "
            "the buffer with var:, or the consuming parser's parameter with "
            "param:, instead")
    leaf: dict[str, Any] = {
        "kind": "source_seed_misanchored",
        "callee": base,
        "arg_index": arg_index,
        "reason": reason,
        "seed": seed,
        "detail": detail,
        "suggested_source": suggest,
        "note": note,
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
            "or a value read from the filled buffer with var:"),
    }
    if slot is not None:
        leaf["slot"] = str(slot)
    if address is not None:
        leaf["address"] = address
    return leaf
