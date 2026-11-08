"""`bn project` command group: Binary Ninja project (`.bnpr`) support.

Contracts live in BNPR_PLAN.md §5.6. One bridge instance owns a project; every
agent routes to it. `project open` / `project attach` pin that instance; the
other subcommands send to whatever instance the normal resolution picks.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .. import cli
from ..cli import _call, arg, command
from ..formatters import (
    _render_project_attach_text,
    _render_project_close_text,
    _render_project_delete_text,
    _render_project_export_text,
    _render_project_file_text,
    _render_project_files_text,
    _render_project_info_text,
    _render_project_load_text,
)
from ..transport import BridgeError, REFRESH_REQUEST_TIMEOUT, _format_instance_choices
from .admin import _associate_project_roots


def _canonical_project_path(raw: str) -> str:
    """BNPR_PLAN.md §5.1: the resolved `.bnpr` directory; a `project.bnpm`
    argument names its parent."""
    if not str(raw).strip():
        raise BridgeError("path is empty: pass the .bnpr project directory to open")
    path = Path(raw).expanduser().resolve()
    if path.name == "project.bnpm":
        path = path.parent
    return str(path)


def _published_project_path(instance: Any) -> str | None:
    """The project path an instance's registry publishes (§5.4), or None."""
    meta = getattr(instance, "meta", None)
    project = meta.get("project") if isinstance(meta, dict) else None
    path = project.get("path") if isinstance(project, dict) else None
    return path if isinstance(path, str) and path else None


def _owning_instance(instances: list[Any], path: str) -> Any | None:
    for inst in instances:
        if _published_project_path(inst) == path:
            return inst
    return None


def _pin_instance(instance: Any) -> str:
    """Pin the instance by SELECTOR, as `bn instance use` does: the fixed GUI
    bridge has instance_id=None and a None pin would delete the key."""
    selector = cli.instance_selector(instance)
    cli.session_state.update(instance_id=selector)
    return selector


@command("project", "open", help="Open a .bnpr project in a bridge (or attach to the one that owns it)",
         args=[
             arg("path", help="Path to the .bnpr project directory (or its project.bnpm)"),
             arg("--instance-id", dest="instance", default=argparse.SUPPRESS,
                 metavar="INSTANCE",
                 help="ID for the bridge instance to auto-spawn (alias of --instance)"),
         ])
def _project_open(args: argparse.Namespace) -> int:
    cli._require_nonempty_instance(args)
    path = _canonical_project_path(args.path)
    # Step 1 (§5.6): a running bridge already owns this project -> pin it and
    # send nothing. Two processes on one project are unsafe (F6/F7), so the
    # owner is always preferred over spawning or routing elsewhere.
    instances = cli.list_instances()
    owner = _owning_instance(instances, path)
    if owner is not None:
        selector = _pin_instance(owner)
        result: Any = {"attached": True, "instance_id": selector, "path": path}
        cli._emit_result(args, result, text_renderer=_render_project_attach_text,
                         stem="project-open")
        return 0
    # Step 2: `bn load`'s routing rule. BN_INSTANCE and -i/--instance-id name an
    # instance explicitly; only the sticky pin is ambient and refused.
    explicitly_routed = bool(
        getattr(args, "_explicit_instance", False)
        or getattr(args, "_explicit_instance_id", False)
        or (getattr(args, "instance", None)
            and not getattr(args, "_sticky_instance", False))
    )
    if not explicitly_routed and len(instances) > 1:
        raise BridgeError(
            "bn project open refuses ambient instance routing while multiple bridges "
            "are running. Pass an explicit -i/--instance (or export BN_INSTANCE) for "
            "an existing bridge, or --instance-id to create a named bridge; the "
            "`bn instance use` pin is not used to pick where a project lands.\n"
            "Running instances:\n" + _format_instance_choices(instances)
        )
    response = cli.send_request(
        "project_open",
        params={"path": path},
        instance_id=getattr(args, "instance", None),
        spawn_missing_named=True,
    )
    result = cli.unwrap_result(response, "project_open")
    # The answering instance (#996 stamp) becomes the sticky pin, and the
    # caller's cwd is associated with it exactly as `bn session start` does.
    answered = response.get("instance") if isinstance(response, dict) else None
    pinned = answered if isinstance(answered, str) and answered else getattr(args, "instance", None)
    if pinned:
        _associate_project_roots(pinned, [os.getcwd()])
        cli.session_state.update(instance_id=pinned)
        if isinstance(result, dict):
            result.setdefault("instance", pinned)
    cli._emit_result(args, result, text_renderer=_render_project_info_text, stem="project-open")
    return 0


@command("project", "attach", help="Pin the running bridge that owns a .bnpr project",
         args=[arg("path", help="Path to the .bnpr project directory (or its project.bnpm)")])
def _project_attach(args: argparse.Namespace) -> int:
    cli._require_nonempty_instance(args)
    path = _canonical_project_path(args.path)
    owner = _owning_instance(cli.list_instances(), path)
    if owner is None:
        raise BridgeError(f"no running bridge owns {path}; run bn project open {path}")
    selector = _pin_instance(owner)
    result: Any = {"attached": True, "instance_id": selector, "path": path}
    cli._emit_result(args, result, text_renderer=_render_project_attach_text,
                     stem="project-attach")
    return 0


@command("project", "info", help="Show the open project: name, path, counts, open targets")
def _project_info(args: argparse.Namespace) -> int:
    return _call(
        args,
        "project_info",
        {},
        require_target=False,
        text_renderer=_render_project_info_text,
        stem="project-info",
    )


@command("project", "files", help="List the files in the open project",
         args=[
             arg("--folder", default=None, metavar="PATH",
                 help="Path-in-project of a folder to list (default: the root)"),
             arg("--flat", action="store_true",
                 help="Only the folder's direct children (default: recurse)"),
         ])
def _project_files(args: argparse.Namespace) -> int:
    return _call(
        args,
        "project_files",
        {"folder": args.folder, "recursive": not args.flat},
        require_target=False,
        text_renderer=_render_project_files_text,
        stem="project-files",
    )


@command("project", "load", help="Open a project file as a target",
         args=[
             arg("selector", help="Project file: id, >=8-char id prefix, path-in-project, or "
                                  "unique file name"),
             arg("--no-bndb", action="store_true",
                 help="Open the raw file even when a sibling <name>.bndb exists in its folder"),
             arg("--quick", "--no-analysis", action="store_true",
                 help="Load without full analysis (fast); `bn refresh` completes it"),
         ])
def _project_load(args: argparse.Namespace) -> int:
    if not str(args.selector).strip():
        raise BridgeError("selector is empty: pass the project file to load")
    return _call(
        args,
        "project_load",
        {
            "selector": args.selector,
            "prefer_bndb": not args.no_bndb,
            "quick": bool(args.quick),
            "workdir": os.getcwd(),
        },
        require_target=False,
        text_renderer=_render_project_load_text,
        stem="project-load",
        # Same one-time full-analysis budget as `bn load` (#321).
        op_default_timeout=REFRESH_REQUEST_TIMEOUT,
    )


@command("project", "import", help="Copy a file from disk into the open project",
         args=[
             arg("path", help="File on disk to import"),
             arg("--folder", default=None, metavar="PATH",
                 help="Path-in-project of the destination folder (created if missing)"),
             arg("--name", default=None, help="Name inside the project (default: basename)"),
             arg("--description", default=None, help="Description for the project file"),
         ])
def _project_import(args: argparse.Namespace) -> int:
    if not str(args.path).strip():
        raise BridgeError("path is empty: pass the file to import")
    params: dict[str, Any] = {"path": str(Path(args.path).expanduser().resolve())}
    if args.folder is not None:
        params["folder"] = args.folder
    if args.name is not None:
        params["name"] = args.name
    if args.description is not None:
        params["description"] = args.description
    return _call(
        args,
        "project_import",
        params,
        require_target=False,
        text_renderer=_render_project_file_text,
        stem="project-import",
    )


@command("project", "export", help="Copy a project file's bytes out to a path on disk",
         args=[
             arg("selector", help="Project file: id, >=8-char id prefix, path-in-project, or "
                                  "unique file name"),
             arg("dest", help="Destination path on disk"),
         ])
def _project_export(args: argparse.Namespace) -> int:
    if not str(args.selector).strip():
        raise BridgeError("selector is empty: pass the project file to export")
    if not str(args.dest).strip():
        raise BridgeError("dest is empty: pass the destination path")
    return _call(
        args,
        "project_export",
        {"selector": args.selector, "dest": str(Path(args.dest).expanduser().resolve())},
        require_target=False,
        text_renderer=_render_project_export_text,
        stem="project-export",
    )


@command("project", "delete", help="Delete a file from the open project (refused while it is an open target)",
         args=[arg("selector", help="Project file: id, >=8-char id prefix, path-in-project, or "
                                    "unique file name")])
def _project_delete(args: argparse.Namespace) -> int:
    if not str(args.selector).strip():
        raise BridgeError("selector is empty: pass the project file to delete")
    return _call(
        args,
        "project_delete",
        {"selector": args.selector},
        require_target=False,
        text_renderer=_render_project_delete_text,
        stem="project-delete",
    )


@command("project", "close", help="Close every project target and release the project",
         args=[arg("--force", action="store_true",
                   help="Discard unsaved changes in project targets instead of refusing")])
def _project_close(args: argparse.Namespace) -> int:
    return _call(
        args,
        "project_close",
        {"force": bool(args.force)},
        require_target=False,
        text_renderer=_render_project_close_text,
        stem="project-close",
    )
