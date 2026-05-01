from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..cli import _call, arg, command
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
         ])
def _load(args: argparse.Namespace) -> int:
    return _call(
        args,
        "load_binary",
        {
            "path": str(Path(args.path).expanduser().resolve()),
            "prefer_bndb": not args.no_bndb,
        },
        require_target=False,
        text_renderer=_render_load_text,
        stem="load",
    )


@command("close", help="Close a loaded binary",
         args=[arg("path", nargs="?", help="Path to close (omit to close all)")])
def _close(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    if args.path:
        params["path"] = str(Path(args.path).expanduser().resolve())
    return _call(
        args,
        "close_binary",
        params,
        require_target=False,
        text_renderer=_render_close_text,
        stem="close",
    )


@command("save", help="Save the current analysis database (.bndb)", target=True,
         args=[arg("path", nargs="?", help="Output path (defaults to <filename>.bndb)")])
def _save(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    if getattr(args, "path", None):
        params["path"] = str(Path(args.path).expanduser().resolve())
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
