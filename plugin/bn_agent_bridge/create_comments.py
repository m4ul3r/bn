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
from ._shared import _parse_address, _validate_count


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
        bv.add_function(addr)
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

        function_name = str(created.name)
        if preview:
            bv.revert_undo_actions(state)
            message = "Preview verified and reverted."
        else:
            bv.commit_undo_actions(state)
            message = "Function created and verified in the live Binary Ninja session."
        return {
            "preview": preview,
            "success": True,
            "committed": bool(not preview),
            "message": message,
            "results": [
                {
                    "op": "function_create",
                    "status": "verified",
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
    except Exception as exc:
        if not mutation_engine._revert_undo_safely(ctx, bv, state):
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
        comment = bv.get_comment_at(fn.start)
        return {
            "function": fn.name,
            "address": hex(fn.start),
            "comment": comment or "",
            "has_comment": bool(comment),
        }

    if address is None:
        raise RuntimeError("comment get requires --address or --function")

    comment_address = _parse_address(address)
    comment = bv.get_comment_at(comment_address)
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
    if offset:
        items = items[offset:]
    if limit is not None:
        items = items[:limit]
    return items


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
