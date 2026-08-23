from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .. import cli
from ..cli import _call, _pick, arg, command
from ..transport import BridgeError, REFRESH_REQUEST_TIMEOUT
from ..formatters import (
    _render_close_text,
    _render_target_choices,
    _render_load_text,
    _render_refresh_text,
    _render_save_text,
    _render_target_info_text,
)


@command("load", help="Load a binary into headless bridge",
         args=[
             arg("path", help="Path to binary or BNDB file"),
             # `--instance-id` is an alias for the global `--instance` spawn name,
             # matching `bn session start --instance-id` so the auto-spawn flag is
             # spelled the same across both entry points (#258). SUPPRESS default
             # so an unset alias never clobbers a root --instance / BN_INSTANCE.
             arg("--instance-id", dest="instance", default=argparse.SUPPRESS,
                 metavar="INSTANCE",
                 help="ID for the bridge instance to auto-spawn (alias of --instance)"),
             arg("--no-bndb", action="store_true",
                 help="Don't auto-prefer a sibling .bndb file"),
             arg("--quick", "--no-analysis", action="store_true",
                 help="Load without full analysis (fast, ~1s): sections/imports/symbols are ready "
                      "immediately; strings and the full function set need `bn refresh`"),
             arg("--no-marker", action="store_true", default=False, dest="no_marker",
                 help="Don't drop a project-local `.bn-<id>` marker (#80); markers let a bare "
                      "`bn` in this project resolve this instance among many (env: BN_NO_MARKERS)"),
         ])
def _load(args: argparse.Namespace) -> int:
    import os
    _env = (os.environ.get("BN_NO_MARKERS") or "").strip().lower()
    no_marker = bool(args.no_marker) or (_env not in ("", "0", "false", "no", "off"))
    return _call(
        args,
        "load_binary",
        {
            "path": str(Path(args.path).expanduser().resolve()),
            "prefer_bndb": not args.no_bndb,
            "quick": bool(args.quick),
            # The bridge drops a `.bn-<id>` marker in this project root (#80) so a
            # later bare `bn` here resolves this instance; pass our cwd as the
            # project anchor (the bridge process's cwd is unrelated).
            "workdir": os.getcwd(),
            "no_marker": no_marker,
        },
        require_target=False,
        text_renderer=_render_load_text,
        stem="load",
        # `bn load --instance <new-id>` auto-spawns that named bridge instead of
        # erroring, so a fresh isolated instance is one command, not two.
        spawn_missing_named=True,
        # A full (non-quick) initial analysis of a very large binary can exceed
        # the 600s read-op default; give the one-time load an hour by default
        # (BN_REQUEST_TIMEOUT still overrides), so it isn't abandoned mid-analysis
        # (#321). `--quick` returns fast and isn't affected.
        op_default_timeout=REFRESH_REQUEST_TIMEOUT,
    )


@command("close", help="Close a loaded binary", target=True,
         args=[
             arg("path", nargs="?", help="Path to close (omit to close the selected target)"),
             arg("--all", action="store_true", help="Close all loaded binaries"),
         ])
def _close(args: argparse.Namespace) -> int:
    # Only honor an explicit -t/--target. A sticky pin must not silently pick
    # which target a bare `close` tears down, and a stale pin must not make
    # cleanup fail -- close needs to stay robust. `_call` resolves `args.target`
    # into the request target, so drop any sticky-injected value and let the
    # resolution below decide (single open target -> that one; multiple ->
    # refuse with the open-target list).
    if getattr(args, "_sticky_target", False):
        args.target = None
    explicit_target = getattr(args, "target", None)
    # Presence, not truthiness: `bn close ""` (an unset shell variable expanding
    # to an empty positional) is an operand that was GIVEN, so it must take
    # part in the conflict checks below and then be rejected as empty -- not
    # collapse into a bare close (which would tear down the sole open target),
    # and `bn close "" --all` must not slip past the path+--all guard.
    explicit_path = getattr(args, "path", None)
    # The three ways to say what to close -- `-t`, a path, `--all` -- are
    # mutually exclusive. The bridge gives a target priority over a path and
    # --all, and --all priority over a path, so any pair would silently drop one
    # operand (`bn close <path> --all` closed EVERYTHING despite naming a file,
    # #85). A destructive op must not guess which one was meant: reject every
    # combination before sending anything.
    if explicit_path is not None and args.all:
        raise BridgeError(
            "Pass a path or --all, not both: a named path closes only that "
            "target; --all closes every loaded target."
        )
    if explicit_target is not None and explicit_path is not None:
        raise BridgeError(
            "Pass --target or a path, not both: --target closes that target; "
            "a path closes the loaded binary at that path."
        )
    if explicit_target is not None and args.all:
        raise BridgeError(
            "Pass --target or --all, not both: --target closes only that "
            "target; --all closes every loaded target."
        )
    # `-t ""` is neither an explicit selector nor a bare close. It used to fall
    # into the implicit-resolution branch (`not target`) and tear down the
    # single open target; an empty selector on a destructive op is an error.
    if explicit_target is not None and not str(explicit_target).strip():
        raise BridgeError(
            "--target is empty: pass a selector from `bn target list`, a path, "
            "or --all (omit --target entirely to close the single open target)."
        )
    # Same for an empty positional: `bn close ""` must not become a bare close.
    if explicit_path is not None and not str(explicit_path).strip():
        raise BridgeError(
            "path is empty: pass a real path, a --target selector from "
            "`bn target list`, or --all (omit the path entirely to close the "
            "single open target)."
        )
    params: dict[str, Any] = {}
    if explicit_path is not None:
        params["path"] = str(Path(explicit_path).expanduser().resolve())
    if args.all:
        params["all"] = True
    # #664: a bare `bn close` closes the single open target and refuses under
    # several (same hint + open-target list as every other target-required
    # command). Previously close skipped resolution and the bridge treated "no
    # target" as close-ALL, the inverse of `bn save`'s refusal.
    #
    # Round 2 (TOCTOU): the implicit resolution PINS the target_id observed in
    # the list_targets peek instead of sending the volatile literal `active`,
    # which the bridge would re-resolve at close time -- a concurrent close/load
    # between peek and close would land the destructive close on a DIFFERENT
    # binary. View ids are never reused, so any interleaving turns the pinned
    # id into a safe unknown-selector error. `-t active` is the same volatile
    # literal spelled explicitly and is pinned the same way.
    if not (args.all or explicit_path is not None) and (
        explicit_target is None or str(explicit_target).strip() == "active"
    ):
        args.target = _pinned_sole_target(args)
    return _call(
        args,
        "close_binary",
        params,
        require_target=False,
        text_renderer=_render_close_text,
        stem="close",
    )


def _pinned_sole_target(args: argparse.Namespace) -> str:
    """Return the ``target_id`` of the single open target, else refuse.

    Same refusal/hint as the generic implicit resolution, but the selector sent
    to the bridge is the stable ``target_id`` from the peek row, never the
    re-resolved ``active`` literal (see :func:`_close`)."""
    # Looked up on ``bn.cli`` at call time (not imported) so tests patch one seam.
    response = cli.send_request(
        "list_targets",
        params={},
        target=None,
        instance_id=getattr(args, "instance", None),
    )
    targets = list(response["result"])
    if not targets:
        raise BridgeError("No BinaryView targets are open")
    if len(targets) > 1:
        raise BridgeError(
            "This command requires --target when multiple targets are open.\n"
            f"Open targets:\n{_render_target_choices(targets)}"
        )
    target_id = targets[0].get("target_id")
    if not target_id:
        raise BridgeError("Bridge listed the open target without a target_id; rerun with an explicit --target")
    return str(target_id)


@command("save", help="Save the current analysis database (.bndb)", target=True,
         args=[
             arg("path", nargs="?", help="Output path (defaults to <filename>.bndb)"),
             arg("--path", dest="path_flag", default=None,
                 help="Output path (alias for the positional)"),
         ])
def _save(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    out_path = _pick(args.path, getattr(args, "path_flag", None), "save path", required=False)
    if out_path:
        params["path"] = str(Path(out_path).expanduser().resolve())
    return _call(
        args,
        "save_database",
        params,
        require_target=False,
        text_renderer=_render_save_text,
        stem="save",
    )


@command("refresh",
         help="Run full analysis on the selected target (e.g. after a --quick load). "
              "On a large target this can take minutes; reads stay responsive meanwhile, "
              "so poll `bn target info` on another connection to watch analysis_progress.",
         target=True)
def _refresh(args: argparse.Namespace) -> int:
    return _call(
        args,
        "refresh",
        {},
        require_target=True,
        text_renderer=_render_refresh_text,
        stem="refresh",
        # The one-time full re-analysis can run far past the 600s read-op default
        # on a very large binary; give it an hour by default (#321). Users still
        # override (or disable) via BN_REQUEST_TIMEOUT for the truly huge case.
        op_default_timeout=REFRESH_REQUEST_TIMEOUT,
    )


@command("target", "info", help="Show one target", target=True,
         fanout=True,
         args=[
             arg("--verbose", "-v", action="store_true",
                 help="Include the segment map (r/w/x address ranges)"),
         ])
def _target_info(args: argparse.Namespace) -> int:
    return _call(
        args,
        "target_info",
        {"selector": args.target, "verbose": bool(args.verbose)},
        require_target=True,
        text_renderer=_render_target_info_text,
        stem="target-info",
    )
