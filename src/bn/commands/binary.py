from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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
             arg("path", nargs="?", help="Path to close (omit to close all)"),
             arg("--all", action="store_true", help="Close all loaded binaries"),
         ])
def _close(args: argparse.Namespace) -> int:
    # Only honor an explicit -t/--target. A sticky pin must not turn a bare
    # `close` (documented as close-all) into close-one, and a stale pin must
    # not make cleanup fail -- close needs to stay robust. `_call` resolves
    # `args.target` into the request target, so drop any sticky-injected value.
    if getattr(args, "_sticky_target", False):
        args.target = None
    # A named path and --all are mutually exclusive: the bridge gives --all
    # priority, so `bn close <path> --all` would silently close EVERY target
    # despite naming one file. Reject the combination instead of surprising the
    # user in a multi-target session (#85).
    if args.path and args.all:
        raise BridgeError(
            "Pass a path or --all, not both: a named path closes only that "
            "target; --all (or a bare `bn close`) closes every loaded target."
        )
    params: dict[str, Any] = {}
    if args.path:
        params["path"] = str(Path(args.path).expanduser().resolve())
    if args.all:
        params["all"] = True
    return _call(
        args,
        "close_binary",
        params,
        require_target=False,
        text_renderer=_render_close_text,
        stem="close",
    )


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


@command("refresh", help="Refresh analysis for the selected target", target=True)
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
