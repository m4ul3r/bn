"""Function creation, comment reads, and inline Python execution.

The ``function_create`` op (plus its own undo/preview handling), the comment
read ops (``get_comment`` / ``list_comments``), and the ``py_exec`` op move here
as module-level free functions, each taking the ``BridgeContext`` seam (``ctx``)
in place of ``self``. ``BinaryNinjaBridge`` keeps a thin delegating shim for
every name the test suite / op binders reference (``_function_create``,
``_get_comment``, ``_list_comments``, ``_normalize_py_result``, ``_py_exec``) --
the existing ``_taint`` -> ``taint_engine`` precedent.

``_bundle_function`` deliberately STAYS on ``BinaryNinjaBridge``: it calls the
bridge-core ``_target_info`` (which reads bridge view-state), so it cannot be a
free function over ``ctx`` without re-coupling.

Outbound calls resolve through:
  * ``ctx`` -- resolution helpers relocated to the seam (``_resolve_view``,
    ``_find_function``);
  * ``read_misc`` -- ``_is_executable_address`` (function_create refuses to
    create a function outside an executable region);
  * ``mutation_engine`` -- ``_revert_undo_safely`` (the shared
    revert-or-report-failure helper used by function_create's preview/rollback);
  * ``_shared`` -- module-free helpers (``_parse_address``, ``_validate_count``).

Import direction is one-way: this module imports ``read_misc``,
``mutation_engine``, and ``_shared`` (plus stdlib + binaryninja). It NEVER
imports ``bridge`` or ``seam``; ``read_misc`` / ``mutation_engine`` never import
THIS module (design spec §3.2).
"""
from __future__ import annotations

import contextlib
import io
from typing import Any

try:
    import binaryninja as bn
except ModuleNotFoundError:  # importable without the Binary Ninja runtime (tests, tooling)
    bn = None  # type: ignore[assignment]

from . import il_format
from . import mutation_engine
from . import read_misc
from .bridge_state import _quick_loaded_views
from ._shared import (
    OperationFailure,
    _parse_address,
    _require_mapped_address,
    _validate_count,
)


def _function_delete(ctx, bv, address, *, exclusive=None):
    """Journal an irreversible deletion and report only post-analysis facts.

    The caller holds the write gate. Analysis runs outside the exclusive read
    lock; a suppression override cannot be rolled back, even if analysis fails.
    """
    scope = exclusive or contextlib.nullcontext
    addr = _parse_address(address)
    requested = {"op": "function_delete", "address": hex(addr)}
    warning = "No clean undo: BN retains a user override suppressing function creation here."
    result = None
    phase = "preflight"
    try:
        with scope():
            fn = bv.get_function_at(addr)
            if fn is None:
                raise OperationFailure("invalid_request", "No function starts at this address.",
                                       requested=requested, observed={"request_applied": False})
            name = str(fn.name)
            predecessors = {ref.function.start for ref in bv.get_code_refs(addr)
                            if ref.function is not None and ref.function.start != addr}
            # Cross-function fall-through need not have a code reference.
            for candidate in bv.functions:
                if candidate.start != addr and any(block.end == addr for block in candidate.basic_blocks):
                    predecessors.add(candidate.start)
            state = bv.begin_undo_actions()
            # Establish the response BEFORE entering the irreversible API. All
            # observations start unknown; failed readback must not invent state.
            observed = {
                "function_removed": None,
                "function_at_entry": None,
                "bodies_containing_deleted_entry": None,
                "reanalysis_completed": False,
                "clean_undo_available": False,
            }
            row = {
                "op": "function_delete", "address": hex(addr), "function": name,
                "status": "verification_failed", "irreversible_change": True,
                "requested": requested, "observed": observed,
            }
            result = {
                "preview": False, "success": False, "committed": True,
                "rolled_back": False, "results": [row],
                "affected_functions": [], "affected_types": [],
            }
            phase = "remove_user_function"
            try:
                bv.remove_user_function(fn)
            except Exception as exc:
                # The API itself can raise after writing. Retain that error
                # while still closing its journal and marking the view dirty.
                observed["apply_error"] = str(exc)
                observed["error"] = str(exc)
                observed["error_stage"] = phase
            phase = "mark_dirty"
            ctx.targets.mark_dirty(bv)
            phase = "commit_undo_actions"
            bv.commit_undo_actions(state)
            for start in predecessors:
                phase = "predecessor_lookup"
                parent = bv.get_function_at(start)
                if parent is not None:
                    phase = "reanalyze"
                    parent.reanalyze()
        phase = "update_analysis_and_wait"
        bv.update_analysis_and_wait()
        observed["reanalysis_completed"] = True
        with scope():
            phase = "get_function_at"
            remaining = bv.get_function_at(addr)
            # Publish each observation only after its entire read succeeds.
            entry_name = str(remaining.name) if remaining is not None else None
            observed["function_removed"] = remaining is None
            observed["function_at_entry"] = entry_name
            phase = "get_functions_containing"
            observed["bodies_containing_deleted_entry"] = [
                {"address": hex(parent.start), "name": str(parent.name)}
                for parent in bv.get_functions_containing(addr) if parent.start != addr
            ]
    except Exception as exc:
        if result is None:
            raise  # Preflight never entered the irreversible mutation.
        # One boundary covers EVERY later phase, including context-manager
        # exit, journal closing, reanalysis, and lazy readback iteration.
        observed.setdefault("error", str(exc))
        observed.setdefault("error_stage", phase)
        if phase == "update_analysis_and_wait":
            observed["analysis_error"] = str(exc)
        elif phase in ("predecessor_lookup", "reanalyze"):
            observed["apply_error"] = str(exc)
        if observed["error"] != str(exc):
            observed["later_error"] = str(exc)
            observed["later_error_stage"] = phase

    removed = observed["function_removed"]
    containing = observed["bodies_containing_deleted_entry"]
    outcome = ("Function is absent at the entry." if removed is True else
               "A function still starts at the deleted entry." if removed is False else
               "Function entry readback is unavailable.")
    if containing is None:
        outcome += " Containing-body readback is unavailable."
    elif containing:
        outcome += " Entry is now inside: " + ", ".join(parent["name"] for parent in containing) + "."
    else:
        outcome += " No other analyzed body contains the deleted entry."
    if "error" in observed:
        outcome += f" Failed at {observed['error_stage']}: {observed['error']}."
        if "later_error" in observed:
            outcome += f" Failed at {observed['later_error_stage']}: {observed['later_error']}."
    success = removed is True and "error" not in observed
    result["success"] = success
    row["status"] = "verified" if success else "verification_failed"
    result["message"] = row["message"] = outcome + " " + warning
    return result


def _remove_created_function(ctx, bv, addr: int) -> bool:
    """Remove a just-created function and confirm it is gone.

    Function creation (``create_user_function``) is not reliably undone by
    ``revert_undo_actions`` -- the function can persist in the view -- so the
    preview / rollback revert must explicitly remove it, reanalyze, and read back
    that no function starts at the address. Returns True only when the function is
    actually gone, so callers never report a revert they did not verify (#117).

    Use ``remove_function``, NOT ``remove_user_function``: the latter records a
    persistent user "do not create a function here" override. Empirically that
    override does NOT block the forced ``create_user_function`` apply path, but
    ``remove_function`` is still the cleaner revert (no lingering suppression
    flag) and is what kept a ``--preview`` (which reverts) from sabotaging a
    later live commit back when the apply used the advisory ``add_function``
    (#304). ``remove_function`` removes a ``create_user_function``-created
    function cleanly, and it can be re-created afterward (verified on real BN).
    """
    fn = bv.get_function_at(addr)
    if fn is None:
        return True
    remove_clean = getattr(bv, "remove_function", None)
    try:
        if remove_clean is not None:
            remove_clean(fn)
        else:  # older BN without remove_function: accept the poison over a no-op
            bv.remove_user_function(fn)
        bv.update_analysis_and_wait()
    except Exception as exc:  # noqa: BLE001 - report the revert as failed, don't raise
        bn.log_error(
            f"BN Agent Bridge: failed to remove created function at 0x{addr:x} on revert: {exc!r}"
        )
        return False
    if bv.get_function_at(addr) is None:
        return True
    # remove_function did not take (rare). Fall back to remove_user_function so
    # the preview is still reverted -- a left-behind function is worse than the
    # address-suppression side-effect.
    try:
        bv.remove_user_function(bv.get_function_at(addr))
        bv.update_analysis_and_wait()
    except Exception as exc:  # noqa: BLE001
        bn.log_error(
            f"BN Agent Bridge: fallback removal of created function at 0x{addr:x} failed: {exc!r}"
        )
        return False
    return bv.get_function_at(addr) is None


def _instruction_boundary(bv, fn, addr: int) -> bool | None:
    """Whether *addr* starts an instruction of *fn*'s block, or None if unknowable.

    Which of BN's two outcomes a create INSIDE a function gets (verified on
    BN 6.1, #927 review): at an instruction boundary BN truncates the host at
    the new start; at a mid-instruction address it leaves the host intact and
    adds an overlapping function that decodes a different instruction stream
    from the same bytes -- the shape of a mistyped address. Walked from the
    containing block's start with BN's own instruction lengths, so a
    fixed-width ISA and a variable-length one are judged the same way.
    """
    try:
        block = fn.get_basic_block_at(addr)
        if block is None:
            return None
        arch = getattr(block, "arch", None)
        cursor = int(block.start)
        while cursor < addr:
            length = int(bv.get_instruction_length(cursor, arch) if arch is not None
                         else bv.get_instruction_length(cursor))
            if length <= 0:
                return None
            cursor += length
        return cursor == addr
    except Exception:  # noqa: BLE001 - an unanswerable probe is "unknown", not a verdict
        return None


def _containing_function_rows(bv, addr: int, *, requested: dict) -> list[dict[str, Any]]:
    """Functions that already CONTAIN *addr*, excluding one starting there.

    Read before a create, because afterwards the address is itself a
    function start and `get_functions_containing` reports the new one. Each
    row names the host, its start, how far into it the address lies and
    whether the address is an instruction boundary there -- the facts a
    caller needs to tell a mistyped address from a deliberate split.

    This is the evidence a GUARD acts on, so a view that cannot answer is
    refused (`internal_error`), not waved through: the #675 disclosure could
    afford to say nothing, but a guard that fails open lets the exact harm it
    exists to stop -- a silent truncation -- through on the one view where
    nobody can see it coming.
    """
    try:
        rows = []
        for fn in (bv.get_functions_containing(addr) or []):
            start = int(getattr(fn, "start", addr))
            if start == addr:
                continue
            row: dict[str, Any] = {
                "name": str(fn.name),
                "address": hex(start),
                "offset": hex(addr - start),
            }
            boundary = _instruction_boundary(bv, fn, addr)
            if boundary is not None:
                row["instruction_boundary"] = boundary
            rows.append(row)
        return rows
    except Exception as exc:  # noqa: BLE001 - fail CLOSED: no evidence is not a licence
        raise OperationFailure(
            "internal_error",
            f"Could not determine whether {hex(addr)} lies inside an existing "
            f"function ({exc!r}); refusing to create one there rather than risk "
            f"truncating a function this probe cannot see.",
            requested=requested,
            observed={"address": hex(addr)},
        ) from exc


def _split_outcome(rows: list[dict[str, Any]]) -> str:
    """What a create inside these hosts does to them, in BN's two outcomes.

    ONE sentence for the refusal (standalone and batch) and the `--help` /
    reference wording follows it: truncation at an instruction boundary, an
    overlapping function decoding a different stream otherwise. Says which
    when every host answers the same way, and covers both when the probe
    could not tell or the hosts disagree.
    """
    verdicts = {row.get("instruction_boundary") for row in rows}
    if verdicts == {True}:
        return (
            "Binary Ninja truncates the host at the new start: it keeps only "
            "the bytes before it and the new function owns the rest."
        )
    if verdicts == {False}:
        return (
            "The address is not an instruction boundary there, so Binary Ninja "
            "leaves the host intact and adds a second, overlapping function "
            "that decodes a different instruction stream from the same bytes "
            "-- the shape of a mistyped address."
        )
    return (
        "Binary Ninja either truncates the host at the new start (when the "
        "address is an instruction boundary in it) or leaves the host intact "
        "beside a second, overlapping function decoding a different "
        "instruction stream (when it is not)."
    )


def _boundary_word(row: dict[str, Any]) -> str:
    boundary = row.get("instruction_boundary")
    if boundary is True:
        return "instruction boundary"
    if boundary is False:
        return "mid-instruction"
    return "boundary unknown"


def _overlap_refusal(addr: int, overlapped: list[dict[str, Any]], requested: dict,
                     *, flag: str) -> OperationFailure:
    """The refusal for a create INSIDE an existing function (M-03).

    A forced ``create_user_function`` at an interior address does not add a
    second function the way the #675 disclosure assumed -- not reliably: at
    an instruction boundary BN truncates the host at the new start and the
    new ``sub_*`` owns the rest, so a mistyped or over-trusted address
    silently cut a user-named, user-prototyped function down to a few bytes
    behind a clean ``committed changed=1 verified=1``; at a mid-instruction
    address it leaves the host intact and lands an overlapping function on
    a bogus instruction stream. The split is legitimate when BN
    over-extended the host, so it stays reachable -- but only on request,
    and the refusal names the host, its start, the offset and which outcome
    this address gets, so the caller can tell which case this is. ONE
    builder for both create paths, so the standalone command and a batch op
    refuse in the same words; *flag* is the spelling the caller's surface
    accepts.
    """
    listed = ", ".join(
        f"{o['name']} (starts at {o['address']}, offset +{int(o['offset'], 16):#x}, "
        f"{_boundary_word(o)})"
        for o in overlapped
    )
    return OperationFailure(
        "invalid_request",
        f"Refused to create a function at {hex(addr)}: the address lies inside "
        f"{listed}. {_split_outcome(overlapped)} If the host was over-extended "
        f"by auto-analysis and that is intended, pass {flag}; the result then "
        f"reports the overlap and what became of the host.",
        requested=requested,
        observed={"address": hex(addr), "overlaps": overlapped},
    )


def _observe_split(bv, addr: int, overlapped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stamp each overlap row with what the create DID to its host.

    Read after analysis: a host that no longer covers the new start was
    truncated there; one that still covers it is intact and now overlaps the
    new function. Observed, not predicted from the boundary probe, so the
    row reports what happened rather than what was expected to.
    """
    observed = []
    for row in overlapped:
        row = dict(row)
        try:
            host = bv.get_function_at(int(row["address"], 16))
            if host is None:
                row["truncated"] = True
            else:
                covers = any(int(r.start) <= addr < int(r.end)
                             for r in _function_ranges(host))
                row["truncated"] = not covers
        except Exception:  # noqa: BLE001 - leave the outcome unstamped rather than guessed
            pass
        observed.append(row)
    return observed


def _with_overlap_note(row: dict, overlapped: list[dict[str, Any]]) -> dict:
    """Add the overlap disclosure to a create row, if there is one.

    ONE builder for both create paths, so the two cannot state the same
    fact differently -- the same reason the duplicate-start note is shared
    across its three surfaces. Only an ``allow_overlap`` create reaches
    this with a non-empty list; the default refuses first. A row already
    stamped by ``_observe_split`` says what happened to the host; an
    unstamped one (a deferred batch create, before its analysis) says what
    can happen.
    """
    if not overlapped:
        return row
    fates = []
    for o in overlapped:
        where = f"{o['name']} @ {o['address']}"
        truncated = o.get("truncated")
        if truncated is True:
            fates.append(f"{where} was truncated at {row.get('address', '<unknown>')} "
                         f"(it keeps only the bytes before the new start)")
        elif truncated is False:
            fates.append(f"{where} is intact and now overlaps the new function, "
                         f"which decodes a different instruction stream from the "
                         f"same bytes")
        else:
            fates.append(f"{where} is either truncated at the new start or left "
                         f"intact beside an overlapping function")
    row = dict(row)
    row["overlaps"] = overlapped
    row["note"] = (
        f"this address is inside an existing function: {'; '.join(fates)}; "
        f"that is legitimate when BN over-extended the neighbour, and a "
        f"mistake if the address was wrong"
    )
    return row


# Branch kinds after which execution still reaches the NEXT instruction: a
# conditional branch's not-taken edge. A call is deliberately absent -- the
# instruction after a call to a noreturn routine is a routine start often.
_FALLTHROUGH_BRANCH_KINDS = frozenset({"FalseBranch"})


def _create_fallthrough_source(bv, addr: int) -> int | None:
    """The instruction that runs straight into *addr*, if one provably does.

    #997 item 2: an address the previous instruction falls through into is
    the middle of a body, not an entry -- the shape of an aarch64 `adrp` page
    base nominated as a missing function. (Named apart from
    `read_decompile._fallthrough_predecessor`, which asks about a FUNCTION's
    neighbour by different rules.) Only a FIXED-WIDTH ISA can name the
    previous instruction (on x86 any byte before *addr* might start one), so
    every other ISA answers None, as does an ARM view: ARM and Thumb code
    share it, and the instruction set of the bytes before *addr* is not known
    there. Anything not decoded as a real instruction, or padding (`nop`),
    answers None too. A DATA variable over the word(s) before *addr* is a
    literal pool between two routines -- unless the instruction before the
    variable's START itself runs into it, which a pool never has (it sits
    after a `ret`/`b`): then the variable is code some other code merely loads
    (its every word must decode and run on), and the fall-through stands. An
    advisory probe: an unanswerable view yields None, never a refusal.
    """
    try:
        arch = getattr(bv, "arch", None)
        arch_name = str(getattr(arch, "name", "") or "").lower()
        if ("arm" in arch_name or "thumb" in arch_name) and "aarch64" not in arch_name \
                and "arm64" not in arch_name:
            return None
        width = int(getattr(arch, "instr_alignment", 0) or 0)
        if width <= 0 or width != int(getattr(arch, "max_instr_length", 0) or 0):
            return None
        prev = addr - width
        if not _falls_through(bv, arch, width, prev):
            return None
        var_start = _data_var_start(bv, prev)
        if var_start is not None:
            if var_start > prev or (prev - var_start) % width:
                return None
            if any(not _falls_through(bv, arch, width, word)
                   for word in range(var_start, prev, width)):
                return None
            before = var_start - width
            if _data_var_start(bv, before) is not None or not _falls_through(bv, arch, width, before):
                return None
        return prev
    except Exception:  # noqa: BLE001 - advisory: no evidence means no warning
        return None


def _data_var_start(bv, address: int) -> int | None:
    """Start of the data variable BN holds over *address*, or None."""
    get_data_var_at = getattr(bv, "get_data_var_at", None)
    var = get_data_var_at(address) if callable(get_data_var_at) else None
    if var is None:
        return None
    start = getattr(var, "address", None)
    return int(start) if start is not None else address


def _falls_through(bv, arch, width: int, address: int) -> bool:
    """Does the fixed-width instruction at *address* run into the next one?
    No branch, or a conditional branch's not-taken edge; never padding or a
    word that does not decode."""
    data = bytes(bv.read(address, width) or b"")
    if len(data) != width:
        return False
    info = arch.get_instruction_info(data, address)
    if info is None or int(getattr(info, "length", 0) or 0) != width:
        return False
    tokens, _length = arch.get_instruction_text(data, address) or ([], 0)
    mnemonic = str(tokens[0]).strip().lower() if tokens else ""
    if not mnemonic or mnemonic in ("nop", "udf", "hlt", "brk"):
        return False
    kinds = {str(getattr(getattr(b, "type", None), "name", ""))
             for b in (getattr(info, "branches", None) or ())}
    return not kinds or bool(kinds & _FALLTHROUGH_BRANCH_KINDS)


def _with_fallthrough_note(row: dict, predecessor: int | None) -> dict:
    """Disclose a create at an address reached by fall-through (#997 item 2).

    ONE builder for both create paths. The create still goes ahead -- BN's
    own boundaries are not authoritative there, and the address may be an
    entry of a routine laid out after a non-returning tail -- but the row
    and the compact summary say what the bytes before it show.
    """
    if predecessor is None:
        return row
    row = dict(row)
    row["fallthrough_from"] = hex(predecessor)
    warning = (f"{row.get('address', '<unknown>')} is reached by fall-through from "
               f"{hex(predecessor)}; likely mid-function, not an entry")
    row["note"] = f"{row['note']}; {warning}" if row.get("note") else warning
    return row


def _function_ranges(fn) -> list:
    ranges = getattr(fn, "address_ranges", None)
    return list(ranges) if ranges is not None else []


def _range_pairs(fn) -> tuple[tuple[int, int], ...]:
    """The half-open ranges *fn* covers, comparable across reads."""
    return tuple(sorted((int(r.start), int(r.end)) for r in _function_ranges(fn)))


def _function_starts_within(bv, ranges) -> set[int]:
    """Function starts inside *ranges*, walked with BN's own next-start index
    so the cost is the functions there, not every function in the view."""
    starts: set[int] = set()
    walk = getattr(bv, "get_next_function_start_after", None)
    if walk is None:
        return starts
    for lo, hi in ranges:
        if bv.get_function_at(lo) is not None:
            starts.add(lo)
        cursor = lo
        while True:
            nxt = walk(cursor)
            if nxt is None or int(nxt) <= cursor or int(nxt) >= hi:
                break
            cursor = int(nxt)
            starts.add(cursor)
    return starts


def _snapshot_split_hosts(bv, overlapped: list[dict[str, Any]]) -> dict[str, Any]:
    """What a revert of a deliberate-split create must put back (#927 P1).

    Removing the created function is NOT restoring the hosts: on real BN a
    create inside two functions that share a tail left both shortened and
    an analysis-created function owning the tail -- and the preview reported
    `rolled_back: true`. So the hosts' ranges and symbols are recorded
    before the create, plus every function start their ranges already held,
    so the repair can tell a function analysis added from one that was
    there.

    Taken immediately before the create, so inside a batch it is the host's
    INTERMEDIATE state (an earlier sibling may have renamed it). That is the
    right reference for what the repair owns -- ranges, existence and the
    AUTO symbols, none of which the undo journal restores, none of which a
    sibling op writes, and each of which the restores undo in reverse apply
    order -- and the wrong one for anything the journal owns (a user name,
    prototype, comments, tags, locals): the batch undoes those
    transaction-wide BEFORE the restores run, so re-applying them from here
    resurrects a sibling's reverted change (#927 round 3). The repair never
    writes them; the transaction baseline (`_split_host_baseline`) verifies
    them. `name` is kept for the audit rows and the log lines only.
    """
    hosts: dict[int, dict[str, Any]] = {}
    for row in overlapped:
        start = int(row["address"], 16)
        fn = bv.get_function_at(start)
        if fn is None:
            continue
        hosts[start] = {
            "name": str(fn.name),
            "auto_symbols": _auto_symbols_at(bv, start),
            "ranges": _range_pairs(fn),
        }
    span = sorted({r for host in hosts.values() for r in host["ranges"]})
    return {"hosts": hosts, "span": span, "starts": _function_starts_within(bv, span)}


def _is_auto_symbol(symbol) -> bool:
    return symbol is not None and bool(getattr(symbol, "auto", False))


def _auto_symbols_at(bv, start: int) -> dict[str, Any]:
    """The REGISTERED auto symbols at *start*, by name -- ALL of them, not
    the one `fn.symbol` prefers. A user rename sits on top of the auto
    symbols without removing them, so the preferred symbol hides what
    analysis did underneath: a host cut down to a jump thunk gains an auto
    `j_*` symbol that outranks its loader name, and after the batch's undo
    drops the user name the thunk name is what shows. The analysis-owned
    set is the thing to put back; the view's own listing is the only place
    it is visible.

    Only the listing, never `fn.symbol` as a stand-in for an empty one: an
    implicit `sub_<addr>` name is synthesized, not registered, so it is in
    `fn.symbol` and absent from the listing -- and a user rename on top
    makes the listing non-empty without it. Falling back mixed the two:
    recorded under the rename as no auto symbols, the host read after the
    undo as having "added" `sub_<addr>`, which `undefine_auto_symbol`
    cannot remove, and a perfectly restored preview reported a failed
    rollback (#927 round 4). The implicit name is verified by the
    transaction baseline's name check instead."""
    return {str(s.name): s for s in bv.get_symbols(start, 1) if _is_auto_symbol(s)}


def _split_host_baseline(ctx, bv, operations: list) -> dict[int, dict[str, Any]]:
    """The TRANSACTION baseline of every function a batch's creates could
    cut: name and ranges of each function containing a `function_create`
    address, read before op 0 applies. `_split_hosts_at_baseline` verifies
    the rollback against this -- the state the whole transaction must
    return to, which no per-op snapshot taken mid-batch can know (#927
    round 3). A create at a gap address records nothing; one inside an
    EARLIER sibling's function finds no function here yet, and that
    sibling's own restore removes it. A malformed address records nothing
    either: this runs before the transaction opens, outside the apply
    handler that turns an op's failure into a `results` row, so raising
    here demoted the batch's structured, rolled-back exit-3 result to a
    bare error (#927 round 4). `_op_function_create` rejects the operand
    itself, exactly as the affected-function pass defers to it."""
    baseline: dict[int, dict[str, Any]] = {}
    for op in operations:
        if not (isinstance(op, dict) and op.get("op") == "function_create" and op.get("address")):
            continue
        try:
            addr = _parse_address(op["address"])
        except ValueError:
            continue
        for fn in ctx._functions_containing(bv, addr):
            start = int(fn.start)
            if start not in baseline:
                baseline[start] = {"name": str(fn.name), "ranges": _range_pairs(fn)}
    return baseline


def _host_ranges_off(bv, hosts: dict[int, dict[str, Any]]) -> list[str]:
    """Which of *hosts* are gone or cover other bytes than recorded."""
    off: list[str] = []
    for start, before in hosts.items():
        fn = bv.get_function_at(start)
        if fn is None:
            off.append(f"{before['name']} @ {hex(start)} is gone")
            continue
        now = _range_pairs(fn)
        if now != before["ranges"]:
            off.append(
                f"{before['name']} @ {hex(start)} covers "
                f"{[(hex(a), hex(b)) for a, b in now]} instead of "
                f"{[(hex(a), hex(b)) for a, b in before['ranges']]}")
    return off


def _host_auto_symbols_off(bv, hosts: dict[int, dict[str, Any]]) -> list[tuple[int, list, list]]:
    """Per host whose auto symbols differ from the snapshot: (start, the
    auto symbols analysis ADDED, the recorded ones now MISSING)."""
    off: list[tuple[int, list, list]] = []
    for start, before in hosts.items():
        if bv.get_function_at(start) is None:
            continue
        now = _auto_symbols_at(bv, start)
        recorded = before["auto_symbols"]
        added = [now[name] for name in now if name not in recorded]
        missing = [recorded[name] for name in recorded if name not in now]
        if added or missing:
            off.append((start, added, missing))
    return off


def _reanalyze_hosts(bv, snapshot: dict[str, Any]) -> None:
    for start in snapshot["hosts"]:
        fn = bv.get_function_at(start)
        if fn is not None:
            fn.reanalyze()
    bv.update_analysis_and_wait()


def _restore_split_hosts(ctx, bv, addr: int, snapshot: dict[str, Any]) -> bool:
    """Put the hosts of a reverted deliberate-split create back, and SAY whether they are.

    Called once the created function is gone. On real BN the hosts usually
    re-extend on their own; when they do not (the shared-tail case, #927 P1),
    the bytes they lost belong to a function analysis created, which is
    dropped -- first with the non-poisoning ``remove_function``, then, if
    analysis re-creates it, with ``remove_user_function`` (the same escalation
    ``_remove_created_function`` makes: a shortened host is worse than a "no
    function here" mark at an address that had no function before either).
    Only starts the snapshot did not record are ever removed.

    This repairs ONLY what analysis destroyed and the undo journal cannot
    restore: ranges, existence, and the AUTO symbol set (a host cut down to
    a jump thunk gains an auto `j_*` symbol that outranks its old name, and
    reanalysis does not take it back): auto symbols analysis added are
    undefined, recorded ones it dropped are redefined. A user name is the
    journal's -- inside a batch the snapshot may hold a sibling's
    since-reverted rename, so writing it here would undo the undo (#927
    round 3); the caller verifies it against the transaction baseline
    instead. Verified after every step; False means the view is NOT as it
    was, and the caller must report `rollback_failed` rather than a clean
    revert.
    """
    if not snapshot.get("hosts"):
        return True
    try:
        ranges_off = _host_ranges_off(bv, snapshot["hosts"])
        if ranges_off:
            keep = set(snapshot["starts"]) | set(snapshot["hosts"]) | {addr}
            for remover_name in ("remove_function", "remove_user_function"):
                remover = getattr(bv, remover_name, None)
                if remover is None:
                    continue
                for start in sorted(_function_starts_within(bv, snapshot["span"]) - keep):
                    extra = bv.get_function_at(start)
                    if extra is not None:
                        remover(extra)
                _reanalyze_hosts(bv, snapshot)
                ranges_off = _host_ranges_off(bv, snapshot["hosts"])
                if not ranges_off:
                    break
        symbols_off = _host_auto_symbols_off(bv, snapshot["hosts"])
        if symbols_off:
            for _start, added, missing in symbols_off:
                for symbol in added:
                    bv.undefine_auto_symbol(symbol)
                for symbol in missing:
                    bv.define_auto_symbol(symbol)
            bv.update_analysis_and_wait()
            symbols_off = _host_auto_symbols_off(bv, snapshot["hosts"])
    except Exception as exc:  # noqa: BLE001 - report the restore as failed, don't raise
        bn.log_error(
            f"BN Agent Bridge: restoring the functions a create at 0x{addr:x} "
            f"split failed on revert: {exc!r}"
        )
        return False
    if ranges_off or symbols_off:
        names_off = [
            f"{hex(start)} has auto symbols analysis added {[str(s.name) for s in added]} "
            f"and lost {[str(s.name) for s in missing]}"
            for start, added, missing in symbols_off
        ]
        bn.log_error(
            f"BN Agent Bridge: the functions a create at 0x{addr:x} split were not "
            f"restored on revert: {'; '.join(ranges_off + names_off)}"
        )
        return False
    return True


def _split_hosts_at_baseline(bv, baseline: dict[int, dict[str, Any]], *, what: str) -> bool:
    """Whether every host in *baseline* exists with its baseline ranges AND
    name -- the final check before a rollback is reported clean. Run after
    the undo and every restore, so a name the journal should have put back
    and did not, or one a repair wrote over it, is a failed rollback and not
    a `rolled_back: true` over a renamed function (#927 round 3). Any name
    difference counts here: nothing but a failed revert can legitimately
    leave the view there. Logs what is off; *what* names the caller."""
    if not baseline:
        return True
    try:
        off = _host_ranges_off(bv, baseline)
        for start, before in baseline.items():
            fn = bv.get_function_at(start)
            if fn is not None and str(fn.name) != before["name"]:
                off.append(f"{hex(start)} is named {fn.name!r} instead of {before['name']!r}")
    except Exception as exc:  # noqa: BLE001 - an unreadable host is not a verified one
        bn.log_error(f"BN Agent Bridge: could not verify the hosts {what} split: {exc!r}")
        return False
    if off:
        bn.log_error(
            f"BN Agent Bridge: the hosts {what} split are not as they were before "
            f"the transaction: {'; '.join(off)}"
        )
        return False
    return True


def _revert_split_hosts(ctx, bv, addr: int, snapshot: dict[str, Any]) -> bool:
    """The standalone command's whole host revert: repair, then verify against
    the baseline -- which, with no sibling op in the transaction, is the
    snapshot itself (every name included)."""
    return (_restore_split_hosts(ctx, bv, addr, snapshot)
            and _split_hosts_at_baseline(bv, snapshot["hosts"], what=f"a create at 0x{addr:x}"))


def _split_host_rows(bv, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """One `affected_functions` row per host of a committed split, so the
    audit lists the function that was cut, not only the one that was made."""
    rows = []
    for start, before in snapshot["hosts"].items():
        fn = bv.get_function_at(start)
        after_name = str(fn.name) if fn is not None else None
        changed = fn is None or _range_pairs(fn) != before["ranges"] or after_name != before["name"]
        rows.append({
            "address": hex(start),
            "before_name": before["name"],
            "after_name": after_name,
            "changed": bool(changed),
        })
    return rows


def _function_create(ctx, selector: str | None, address, preview: bool, *,
                     allow_overlap: bool = False, exclusive=None):
    """Create a function at *address*, verify it, and revert if asked or failed.

    #628 lock split: ``exclusive`` is the caller's EXCLUSIVE target-lock scope -- a
    context-manager FACTORY (``bridge._function_create`` passes
    ``self._target_lock.write``). The caller holds the bridge write GATE for the
    whole call, which serializes writers; the exclusive lock is taken here only
    around the phases that mutate or snapshot BN state. The post-create
    ``update_analysis_and_wait()`` deliberately runs OUTSIDE it, under the gate
    alone, so concurrent reads stay live instead of starving for a multi-minute
    reanalysis -- parity with load_binary (#99), refresh (#321), go_rename (#365)
    and the #628 mutation engine. The revert's own reanalysis
    (``_remove_created_function``) stays exclusive on purpose: a reader must never
    observe a half-reverted view. ``exclusive=None`` means "no exclusive scope" (a
    no-op), so direct engine calls and tests keep working.

    ``allow_overlap`` is the deliberate-split switch (M-03): a create at an
    address INSIDE an existing function is refused by default -- see
    ``_overlap_refusal`` -- and goes ahead, disclosing the overlap in its
    row, only when it is True. The preview obeys the same rule: a preview
    that previews a refused create would report `verified` for a change
    the live command then refuses.
    """
    exclusive_scope = exclusive if exclusive is not None else contextlib.nullcontext
    with exclusive_scope():
        bv = ctx._resolve_view(selector)
        addr = _parse_address(address)
        requested = {"op": "function_create", "address": hex(addr)}
        if allow_overlap:
            requested["allow_overlap"] = True
        # #479: on a --quick-loaded view the initial analysis never ran, so the
        # update_analysis_and_wait() below would run the full (multi-minute) analysis
        # as a side effect of a mutation -- a stall the caller never asked for, and
        # before #628 one that also held the exclusive write lock so even a later
        # `target info` blocked. Refuse fast and point at `bn refresh` (which is built
        # to run that analysis once) instead. Also protects --preview, whose whole
        # promise is a bounded, revertible probe. #628 only shortens the exclusive
        # window the reanalysis runs in; this refusal is unchanged -- except that
        # it is the batch op's `OperationFailure("invalid_request")` now (M-24's
        # twin, #927 review): a bare RuntimeError put the standalone command at
        # exit 2 for the refusal `batch apply` reports at exit 3.
        if bv in _quick_loaded_views:
            raise OperationFailure(
                "invalid_request",
                "Cannot create a function: this target was loaded with --quick (no "
                "analysis). Creating a function would run the full analysis under the "
                "write lock and wedge the instance. Run `bn refresh` first.",
                requested=requested,
            )

        # #675 item 14: capture the CONTAINING functions before the create --
        # afterwards this address IS a function start and the answer changes.
        # A create at a mid-function address reported `verified` and said
        # nothing; M-03 then showed the "overlap" is a truncation of the host,
        # so it is refused unless `allow_overlap` asks for the split, and
        # disclosed in the row when it does.
        #
        # BOTH create paths need this. `bn function create` runs here;
        # `batch apply` runs `mutation_engine._op_function_create`. Patching
        # only the one I found first left the command under test unchanged,
        # which the live run caught.
        overlapped = _containing_function_rows(bv, addr, requested=requested)
        existing = bv.get_function_at(addr)
        if existing is not None:
            return {
                "preview": preview,
                "success": True,
                "committed": False,
                "message": "A function already starts at this address.",
                "results": [
                    {
                        "op": "function_create",
                        "status": "noop",
                        "address": hex(addr),
                        "function": str(existing.name),
                        "message": "A function already starts at this address.",
                        "requested": requested,
                    }
                ],
                "affected_functions": [],
                "affected_types": [],
            }

        # Refuse to create junk functions: the address must be mapped and live
        # inside an executable region. Auto-analysis skips exactly these handler
        # entry points (reachable only via data/function-pointer tables), so we
        # still create them on request -- but only where code can actually run.
        # M-24: OperationFailure, not a bare RuntimeError -- the batch path
        # already refuses the same address as `invalid_request`, and a bridge
        # mutation refusal is exit 3 on both surfaces, not 2 on one of them.
        if len(bytes(bv.read(addr, 1))) == 0:
            raise OperationFailure(
                "invalid_request",
                f"Cannot create function: address {hex(addr)} is not mapped",
                requested=requested,
            )
        if not read_misc._is_executable_address(ctx, bv, addr):
            raise OperationFailure(
                "invalid_request",
                f"Cannot create function: address {hex(addr)} is not inside an executable segment",
                requested=requested,
            )
        if overlapped and not allow_overlap:
            raise _overlap_refusal(addr, overlapped, requested, flag="--allow-overlap")

        # #927 P1: what a revert must put back. Removing the created function
        # does not restore the hosts on every geometry (two hosts sharing a
        # tail were left shortened behind `rolled_back: true`), so record
        # them now and verify them after every revert below.
        hosts = _snapshot_split_hosts(bv, overlapped) if overlapped else {"hosts": {}}
        fallthrough_from = _create_fallthrough_source(bv, addr)
        state = bv.begin_undo_actions()
    try:
        with exclusive_scope():
            # create_user_function (FORCED), not add_function (an advisory auto hint):
            # auto-analysis declines exactly the addresses it already skipped -- the
            # data-table / missed-handler entries this op exists to recover -- so
            # add_function returned verification_failed on its own documented use-case
            # (#360). The forced path also bypasses any prior remove_user_function
            # "no function here" suppression, so a preview's cleanup can't sabotage a
            # later live create.
            bv.create_user_function(addr)

        # #628: the reanalysis after the create is the long pole on a large
        # target -- run it under the write gate only (still held by the caller,
        # so no second writer can start) so concurrent reads stay live.
        bv.update_analysis_and_wait()
        with exclusive_scope():
            created = bv.get_function_at(addr)
            if created is None:
                reverted = mutation_engine._revert_undo_safely(ctx, bv, state)
                reverted = _revert_split_hosts(ctx, bv, addr, hosts) and reverted
                return {
                    "preview": preview,
                    "success": False,
                    "committed": False,
                    "rolled_back": reverted,
                    "message": (
                        "Rolled back because no function was created at the address."
                        if reverted
                        else "No function was created at the address AND the rollback failed; "
                        "the view may be left partially modified."
                    ),
                    "results": [
                        {
                            "op": "function_create",
                            "status": "verification_failed",
                            "address": hex(addr),
                            "message": f"No function starts at 0x{addr:x} after analysis.",
                            "requested": requested,
                            "observed": {"address": hex(addr), "function": None},
                        }
                    ],
                    "affected_functions": [],
                    "affected_types": [],
                }

            guard_reason = mutation_engine._function_looks_like_code(bv, created, addr)
            if guard_reason is not None:
                # The forced create landed a junk function on non-code; revert the
                # journal and explicitly remove the created function (creation is not
                # reliably undone), then fail honestly instead of reporting the
                # fabricated function verified (#386).
                mutation_engine._revert_undo_safely(ctx, bv, state)
                removed = _remove_created_function(ctx, bv, addr)
                removed = _revert_split_hosts(ctx, bv, addr, hosts) and removed
                return {
                    "preview": preview,
                    "success": False,
                    "committed": False,
                    "rolled_back": removed,
                    "message": mutation_engine._function_create_guard_message(addr, guard_reason),
                    "results": [
                        {
                            "op": "function_create",
                            "status": "verification_failed",
                            "address": hex(addr),
                            "message": mutation_engine._function_create_guard_message(addr, guard_reason),
                            "requested": requested,
                            "observed": {"address": hex(addr), "function": str(created.name)},
                        }
                    ],
                    "affected_functions": [],
                    "affected_types": [],
                }

            function_name = str(created.name)
            op_status = "verified"
            # Observed after analysis: which hosts the create truncated and
            # which it left intact (the mid-instruction outcome), and the
            # audit row per host -- read BEFORE a preview's revert so the
            # preview reports what the live run would do to them.
            overlapped = _observe_split(bv, addr, overlapped)
            host_rows = _split_host_rows(bv, hosts)
            if preview:
                # Function creation isn't reliably undone by revert_undo_actions
                # (the same non-journaled class as create_user_var / set_user_type),
                # so after reverting the journal explicitly remove the created
                # function and read back that it is gone -- never claim a revert we
                # did not verify (#117).
                bv.revert_undo_actions(state)
                reverted = _remove_created_function(ctx, bv, addr)
                # #927 P1: the created function being gone is not the hosts
                # being back. Verified, repaired where BN allows, and reported
                # as a failed rollback otherwise -- never a clean revert on
                # trust.
                reverted = _revert_split_hosts(ctx, bv, addr, hosts) and reverted
                committed = False
                success = reverted
                if reverted:
                    message = "Preview verified and reverted."
                else:
                    # The function was created+verified, but removing it on revert
                    # failed -- it may still be in the view. Mark the op the way the
                    # batch engine marks a failed rollback so the text renderer
                    # routes it to 'failed:' instead of '[verified]' and the per-op
                    # status stops contradicting success:false (#117).
                    message = (
                        "Preview verified, but reverting it failed: the created "
                        "function, or a function it split, may be left modified "
                        "in the view."
                    )
                    op_status = "rollback_failed"
                    # The fabricated function is still LIVE in the view, but BN's
                    # bv.file.modified never flips True for our create -- so without
                    # this mark `bn close` computes unsaved=false and never warns
                    # about the leftover. Mark dirty so close still warns, even though
                    # the op itself reports failure. A clean preview revert (the
                    # branch above) intentionally does NOT dirty the view (#545). Same
                    # guarded helper the committed #519 path uses.
                    try:
                        ctx.targets.mark_dirty(bv)
                    except Exception:
                        pass
            else:
                bv.commit_undo_actions(state)
                reverted = None
                committed = True
                success = True
                message = "Function created and verified in the live Binary Ninja session."
                # This standalone path bypasses the generic _mutation() shim that
                # marks the view dirty, and BN's bv.file.modified does NOT flip True
                # after our verified mutation -- so without this, `bn close` computes
                # unsaved=false and silently drops the created function (#519). Mark
                # dirty only on a committed, verified (non-preview) create; preview
                # reverts and the noop / guard-rejected paths return earlier.
                try:
                    ctx.targets.mark_dirty(bv)
                except Exception:
                    pass
            result = {
                "preview": preview,
                "success": success,
                "committed": committed,
                "message": message,
                "results": [
                    _with_fallthrough_note(_with_overlap_note({
                        "op": "function_create",
                        "status": op_status,
                        "address": hex(addr),
                        "function": function_name,
                        "requested": requested,
                    }, overlapped), fallthrough_from)
                ],
                # The host(s) a deliberate split cut are affected functions
                # too (#927 review): the compact face carries them under
                # `overlapping_functions`; the full audit must not list only
                # the function that was made.
                "affected_functions": [
                    {
                        "address": hex(addr),
                        "before_name": None,
                        "after_name": function_name,
                        "changed": True,
                    },
                    *host_rows,
                ],
                "affected_types": [],
            }
            if preview:
                result["rolled_back"] = reverted
            return result
    except Exception as exc:
        with exclusive_scope():
            # revert_undo_actions does not reliably remove a just-created function, so
            # also explicitly drop any function left at the address (#117).
            undo_ok = mutation_engine._revert_undo_safely(ctx, bv, state)
            removed = _remove_created_function(ctx, bv, addr)
            removed = _revert_split_hosts(ctx, bv, addr, hosts) and removed
        if not (undo_ok and removed):
            raise RuntimeError(
                f"{exc} (additionally, rollback failed; the view may be left partially modified)"
            ) from exc
        raise


def _get_comment(ctx, selector: str | None, address, function):
    bv = ctx._resolve_view(selector)
    if function and address is not None:
        raise RuntimeError(
            "Pass --address or --function, not both: they target different locations."
        )
    if function:
        fn = ctx._find_function(bv, function)
        # Aggregate ALL comments within the function's address range, not just the
        # entry-address comment -- a comment lands on the interesting call/branch,
        # not the prologue, so the entry-only read reported (no comment) for a
        # function that has several, contradicting `comment list` (#203). Reads the
        # same global address_comments and uses containing-function membership, so
        # the two agree for the common case. (One divergence: an address shared by
        # OVERLAPPING functions is reported here for EACH containing function --
        # inclusive -- where `comment list` attributes it to just the first; both
        # are defensible and neither loses data.)
        address_comments = getattr(bv, "address_comments", {}) or {}
        comments = []
        for addr in sorted(address_comments):
            text = address_comments[addr]
            if not text:
                continue
            funcs = bv.get_functions_containing(addr)
            if not any(int(f.start) == int(fn.start) for f in funcs):
                continue
            comments.append({"address": hex(int(addr)), "comment": text})
        function_doc = str(getattr(fn, "comment", "") or "")
        return {
            "function": fn.name,
            "address": hex(fn.start),
            "function_doc": function_doc,
            "has_function_doc": bool(function_doc),
            "comments": comments,
            "comment_count": len(comments),
            "has_comment": bool(comments),
        }

    if address is None:
        raise RuntimeError("comment get requires --address or --function")

    comment_address = _parse_address(address)
    comment = bv.get_comment_at(comment_address)
    # Reject an unmapped address rather than reporting a false 'no comment' for a
    # typo'd/stale address -- parity with read/decompile (#374). Never suppress a
    # real comment: only an address that is BOTH unmapped AND comment-less is the
    # typo case (mirrors the xrefs refs gate).
    if not comment:
        _require_mapped_address(bv, comment_address)
    return {
        "address": hex(comment_address),
        "comment": comment or "",
        "has_comment": bool(comment),
    }


_COMMENT_SCOPES = ("all", "address", "function")


def _list_comments(
    ctx,
    selector: str | None,
    *,
    query: str | None = None,
    offset: int = 0,
    limit: int | None = None,
    scope: str = "all",
):
    # Re-enforce the count contract (see _sections) so a negative offset/
    # limit is a clean invalid_request, not a silent negative-slice (#100).
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    scope = (scope or "all").lower()
    if scope not in _COMMENT_SCOPES:
        raise OperationFailure(
            "invalid_request",
            f"unknown comment scope {scope!r}; expected one of {', '.join(_COMMENT_SCOPES)}",
        )
    bv = ctx._resolve_view(selector)
    needle = query.lower() if query else None
    items = []
    if scope in ("all", "address"):
        for addr in sorted(bv.address_comments):
            text = bv.address_comments[addr]
            if not text:
                continue
            if needle and needle not in text.lower():
                continue
            funcs = bv.get_functions_containing(addr)
            owner = funcs[0] if funcs else None
            items.append({
                "address": hex(addr),
                "function": owner.name if owner is not None else None,
                # #978: the demangled spelling the text face shows.
                "function_display_name": (
                    il_format._display_name(owner) if owner is not None else None),
                "comment": text,
                "scope": "address",
            })
    if scope in ("all", "function"):
        # #643: function documentation comments (`fn.comment`, written by
        # `comment set --function`) live on the function object, NOT in
        # bv.address_comments -- so the ONLY discovery command could never see
        # them and `comment list --query TODO` reported `none` for a TODO the
        # write had just confirmed as `verified`. #203 made `comment get
        # --function` a superset of both stores for exactly this reason; this is
        # the mirror direction, so list is complete by default too.
        for fn in getattr(bv, "functions", None) or []:
            doc = str(getattr(fn, "comment", "") or "")
            if not doc:
                continue
            if needle and needle not in doc.lower():
                continue
            items.append({
                "address": hex(int(fn.start)),
                "function": fn.name,
                "function_display_name": il_format._display_name(fn),
                "comment": doc,
                "scope": "function_doc",
            })
    # Address-ordered so paging is stable across the two stores; at a shared
    # address the function doc sorts first (it describes the whole function).
    items.sort(key=lambda item: (int(item["address"], 16), 0 if item["scope"] == "function_doc" else 1))
    # Honest paging envelope ({items,total,offset,limit,returned,has_more}),
    # matching strings/imports/sections/function-list (#122/#131). The helper
    # applies the offset/limit slice itself off the full filtered set.
    return read_misc._paged_list_result(items, offset=offset, limit=limit, kind="comments")


def _normalize_py_result(ctx, value: Any) -> tuple[Any, list[str]]:
    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, (bool, int, float, str)):
            return item
        if isinstance(item, (list, tuple)):
            return [normalize(part) for part in item]
        if isinstance(item, dict):
            return {str(key): normalize(val) for key, val in item.items()}
        raise TypeError(type(item).__name__)

    try:
        return normalize(value), []
    except TypeError:
        return repr(value), ["`result` was not JSON-serializable; returned repr(result) instead."]


# py_exec runs arbitrary Python inside the BN process with a live `bv` -- full
# in-process code execution, i.e. complete compromise of the bridge user. It runs
# unconditionally: the bridge trusts whoever can reach its owner-only socket
# (peercred + 0o600 socket / 0o700 dir) to run arbitrary code as themselves.
def _py_exec(ctx, selector: str | None, script: str):
    bv = ctx._resolve_view(selector)
    stdout = io.StringIO()
    explicitly_dirty = False

    def modified() -> bool:
        try:
            return bool(getattr(bv.file, "modified", False))
        except Exception:
            return False

    def mark_dirty() -> None:
        nonlocal explicitly_dirty
        explicitly_dirty = True

    before_modified = modified()
    scope = {
        "bn": bn,
        "binaryninja": bn,
        "bv": bv,
        "mark_dirty": mark_dirty,
        "result": None,
    }
    dirty = False
    try:
        with contextlib.redirect_stdout(stdout):
            try:
                exec(script, scope, scope)
            except (SystemExit, KeyboardInterrupt) as exc:
                # These are BaseException, not Exception, so the broad handler below
                # would let them through and unwind the worker thread -- the client
                # then sees a misleading "empty response / worker faulted" instead of
                # the real cause. A snippet calling sys.exit()/raising SystemExit (or
                # a stray KeyboardInterrupt) is the user's own bug, not a bridge
                # fault, so report it cleanly and keep the worker alive (#387).
                code = getattr(exc, "code", None)
                detail = f" (code {code!r})" if isinstance(exc, SystemExit) else ""
                raise RuntimeError(
                    f"py exec snippet raised {type(exc).__name__}{detail}; "
                    "the worker was protected"
                ) from exc
            except Exception as exc:  # noqa: BLE001 - user script errors are user-facing
                # Report every script failure the same way -- "TypeName: message".
                # Previously a ValueError surfaced as a bare message while a
                # NameError was tagged "internal error: NameError:", because only
                # some builtins are whitelisted as user-facing. The user's own
                # script raised this, so it is always a user-facing error.
                raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        dirty = explicitly_dirty or (not before_modified and modified())
        if dirty:
            ctx.targets.mark_dirty(bv)

    result_value, warnings = _normalize_py_result(ctx, scope.get("result"))
    result = {
        "stdout": stdout.getvalue(),
        "result": result_value,
        "warnings": warnings,
        "dirty": dirty,
    }
    return result
