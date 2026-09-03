from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .. import cli
from ..cli import _call, _pick, arg, command
from ..transport import BridgeError, REFRESH_REQUEST_TIMEOUT
from ..formatters import (
    _render_close_text,
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
         ])
def _load(args: argparse.Namespace) -> int:
    import os
    # `bn load "$BIN"` with $BIN unset resolves "" to the cwd DIRECTORY, which
    # passes the bridge's exists() check and fails confusingly bridge-side --
    # same unset-shell-var doctrine as close/save (#690 r4).
    if not str(args.path).strip():
        raise BridgeError("path is empty: pass the binary or .bndb file to load")
    explicitly_routed = bool(
        getattr(args, "_explicit_instance", False)
        or getattr(args, "_explicit_instance_id", False)
    )
    if not explicitly_routed and len(cli.list_instances()) > 1:
        raise BridgeError(
            "bn load refuses ambient instance routing while multiple bridges are "
            "running. Pass an explicit -i/--instance for an existing bridge, or "
            "--instance-id to create a named bridge."
        )
    return _call(
        args,
        "load_binary",
        {
            "path": str(Path(args.path).expanduser().resolve()),
            "prefer_bndb": not args.no_bndb,
            "quick": bool(args.quick),
            # Associate successful loads with the caller's canonical project root
            # in the private instance registry. No state is written to the checkout.
            "workdir": os.getcwd(),
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
             arg("path", nargs="?", help="Path to close (omit to close the single open target)"),
             arg("--all", action="store_true", help="Close all open targets, including GUI tabs bn did not load"),
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
    # id into a safe unknown-selector error. Round 3: the pinning lives in the
    # shared cli._implicit_target (every implicit resolution pins now, #690 R3)
    # and only the EXACT "active" literal is intercepted -- a padded ' active '
    # was a safe unknown-selector error before round 2 and must stay one, not
    # become a destructive sole-target close.
    if not (args.all or explicit_path is not None) and (
        explicit_target is None or explicit_target == "active"
    ):
        try:
            # Looked up on ``bn.cli`` at call time so tests patch one seam.
            args.target = cli._implicit_target(args)
        except cli.MultiTargetError as exc:
            if explicit_target == "active":
                raise BridgeError(
                    f"{exc}\nnote: `-t active` follows the GUI selection, "
                    "which is not honored for close: pass a concrete "
                    "selector, a path, or --all (closes every open target, "
                    "including GUI tabs bn did not load)."
                ) from None
            raise
    return _call(
        args,
        "close_binary",
        params,
        require_target=False,
        text_renderer=_render_close_text,
        stem="close",
    )


@command("target", "close", help="Close one open target by selector", fmt="text",
         prefer_when="you have a selector from `bn target list` and want that one "
                     "target gone; `close` is the same operation with --target/a "
                     "path/--all",
         see_also=("close", "target list", "session stop"),
         args=[
             arg("selector",
                 help="Target selector to close (see `bn target list`)"),
         ])
def _target_close(args: argparse.Namespace) -> int:
    """Close exactly the named target.

    Discoverability, not new behavior: `target list`/`use`/`info` live under
    `target`, so agents look for the close verb there first and previously got an
    argparse error before guessing `bn close -t`. This delegates to the canonical
    `close` handler -- same unsaved warnings, same selector resolution, same
    renderer -- so the two spellings cannot drift.
    """
    selector = getattr(args, "selector", None)
    # Presence, not truthiness: `bn target close "$SEL"` with SEL unset expands
    # to an empty operand that must ERROR, never degrade into "close whatever is
    # open". The canonical handler rejects an empty selector too, but phrases the
    # recovery in terms of `--target`/a path/`--all` -- none of which this parser
    # accepts -- so the check is stated here in this command's own vocabulary.
    if selector is None or not str(selector).strip():
        raise BridgeError(
            "selector is empty: pass a target selector from `bn target list` "
            "(use `bn close --all` to close every loaded target)."
        )
    args.target = selector
    # `active` follows the GUI selection and is re-resolved bridge-side, so it is
    # never a safe name for a destructive close; the canonical handler already
    # refuses to forward it and resolves a concrete target instead.
    # This spelling closes ONE named target and nothing else: neither operand
    # that could widen or redirect the close is reachable from this parser, so
    # pin them off before delegating.
    args.path = None
    args.all = False
    args._sticky_target = False
    return _close(args)


@command("save", help="Save the current analysis database (.bndb)", target=True,
         args=[
             arg("path", nargs="?", help="Output path (defaults to <filename>.bndb)"),
             arg("--path", dest="path_flag", default=None,
                 help="Output path (alias for the positional)"),
         ])
def _save(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    out_path = _pick(args.path, getattr(args, "path_flag", None), "save path", required=False)
    # Presence, not truthiness: `bn save "$OUT"` with $OUT unset must error,
    # not silently save to the default path (#690 r3, same doctrine as close).
    if out_path is not None and not str(out_path).strip():
        raise BridgeError(
            "path is empty: pass a real output path, or omit it to save to "
            "the default <filename>.bndb"
        )
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
