from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..cli import _call, _pick, arg, command
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
             arg("--no-bndb", action="store_true",
                 help="Don't auto-prefer a sibling .bndb file"),
             arg("--quick", "--no-analysis", action="store_true",
                 help="Load without full analysis (fast, ~1s): sections/imports/strings/symbols "
                      "are ready immediately; run `bn refresh` for full function analysis"),
         ])
def _load(args: argparse.Namespace) -> int:
    return _call(
        args,
        "load_binary",
        {
            "path": str(Path(args.path).expanduser().resolve()),
            "prefer_bndb": not args.no_bndb,
            "quick": bool(args.quick),
        },
        require_target=False,
        text_renderer=_render_load_text,
        stem="load",
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
        allow_implicit_target=True,
        text_renderer=_render_refresh_text,
        stem="refresh",
    )


@command("target", "info", help="Show one target", target=True)
def _target_info(args: argparse.Namespace) -> int:
    return _call(
        args,
        "target_info",
        {"selector": args.target},
        require_target=True,
        allow_implicit_target=True,
        text_renderer=_render_target_info_text,
        stem="target-info",
    )
