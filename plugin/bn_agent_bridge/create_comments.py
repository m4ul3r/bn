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

import binaryninja as bn

from . import mutation_engine
from . import read_misc
from ._shared import _parse_address, _require_mapped_address, _validate_count


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


def _function_create(ctx, selector: str | None, address, preview: bool):
    bv = ctx._resolve_view(selector)
    addr = _parse_address(address)
    requested = {"op": "function_create", "address": hex(addr)}

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
    if len(bytes(bv.read(addr, 1))) == 0:
        raise RuntimeError(
            f"Cannot create function: address 0x{addr:x} is not mapped"
        )
    if not read_misc._is_executable_address(ctx, bv, addr):
        raise RuntimeError(
            f"Cannot create function: address 0x{addr:x} is not inside an executable segment"
        )

    state = bv.begin_undo_actions()
    try:
        # create_user_function (FORCED), not add_function (an advisory auto hint):
        # auto-analysis declines exactly the addresses it already skipped -- the
        # data-table / missed-handler entries this op exists to recover -- so
        # add_function returned verification_failed on its own documented use-case
        # (#360). The forced path also bypasses any prior remove_user_function
        # "no function here" suppression, so a preview's cleanup can't sabotage a
        # later live create.
        bv.create_user_function(addr)
        bv.update_analysis_and_wait()
        created = bv.get_function_at(addr)
        if created is None:
            reverted = mutation_engine._revert_undo_safely(ctx, bv, state)
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
        if preview:
            # Function creation isn't reliably undone by revert_undo_actions
            # (the same non-journaled class as create_user_var / set_user_type),
            # so after reverting the journal explicitly remove the created
            # function and read back that it is gone -- never claim a revert we
            # did not verify (#117).
            bv.revert_undo_actions(state)
            reverted = _remove_created_function(ctx, bv, addr)
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
                    "Preview verified, but removing the created function on "
                    "revert failed; the view may be left modified."
                )
                op_status = "rollback_failed"
        else:
            bv.commit_undo_actions(state)
            reverted = None
            committed = True
            success = True
            message = "Function created and verified in the live Binary Ninja session."
        result = {
            "preview": preview,
            "success": success,
            "committed": committed,
            "message": message,
            "results": [
                {
                    "op": "function_create",
                    "status": op_status,
                    "address": hex(addr),
                    "function": function_name,
                    "requested": requested,
                }
            ],
            "affected_functions": [
                {
                    "address": hex(addr),
                    "before_name": None,
                    "after_name": function_name,
                    "changed": True,
                }
            ],
            "affected_types": [],
        }
        if preview:
            result["rolled_back"] = reverted
        return result
    except Exception as exc:
        # revert_undo_actions does not reliably remove a just-created function, so
        # also explicitly drop any function left at the address (#117).
        undo_ok = mutation_engine._revert_undo_safely(ctx, bv, state)
        removed = _remove_created_function(ctx, bv, addr)
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
        return {
            "function": fn.name,
            "address": hex(fn.start),
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


def _list_comments(
    ctx,
    selector: str | None,
    *,
    query: str | None = None,
    offset: int = 0,
    limit: int | None = None,
):
    # Re-enforce the count contract (see _sections) so a negative offset/
    # limit is a clean invalid_request, not a silent negative-slice (#100).
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    bv = ctx._resolve_view(selector)
    needle = query.lower() if query else None
    items = []
    for addr in sorted(bv.address_comments):
        text = bv.address_comments[addr]
        if not text:
            continue
        if needle and needle not in text.lower():
            continue
        funcs = bv.get_functions_containing(addr)
        func_name = funcs[0].name if funcs else None
        items.append({
            "address": hex(addr),
            "function": func_name,
            "comment": text,
        })
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


def _py_exec(ctx, selector: str | None, script: str):
    bv = ctx._resolve_view(selector)
    stdout = io.StringIO()
    scope = {
        "bn": bn,
        "binaryninja": bn,
        "bv": bv,
        "result": None,
    }
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
    result_value, warnings = _normalize_py_result(ctx, scope.get("result"))
    result = {
        "stdout": stdout.getvalue(),
        "result": result_value,
        "warnings": warnings,
    }
    return result
