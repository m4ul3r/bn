"""`bn project` command group: Binary Ninja project (`.bnpr`) support.

One bridge instance owns a project; every agent routes to it. `project open` /
`project attach` / `project init` pin that instance; the other subcommands send
to whatever instance the normal resolution picks.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from .. import cli, paths, rootfs_layout
from ..cli import _call, _mutate, arg, command, mutation_output_args, preview_arg
from ..formatters import (
    _project_link_summary,
    _project_finding_summary,
    _render_mutation_summary_text,
    _render_project_attach_text,
    _render_project_close_text,
    _render_project_delete_text,
    _render_project_export_text,
    _render_project_file_text,
    _render_project_files_text,
    _render_project_info_text,
    _render_project_init_text,
    _render_project_libs_text,
    _render_project_link_text,
    _render_project_load_text,
    _render_project_sync_text,
    _render_project_finding_text,
    _render_project_findings_text,
    _render_project_finding_context_text,
    _render_project_finding_check_text,
    _render_project_finding_types_text,
)
from ..mutation_statuses import FAILED_MUTATION_STATUSES
from ..transport import BridgeError, REFRESH_REQUEST_TIMEOUT, _format_instance_choices
from .admin import _associate_project_roots


def _canonical_project_path(raw: str) -> str:
    """The resolved `.bnpr` directory; a `project.bnpm` argument names its
    parent."""
    if not str(raw).strip():
        raise BridgeError("path is empty: pass the .bnpr project directory to open")
    path = Path(raw).expanduser().resolve()
    if path.name == "project.bnpm":
        path = path.parent
    return str(path)


def _published_project_path(instance: Any) -> str | None:
    """The project path an instance's registry publishes, or None."""
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


def _attach_or_open(args: argparse.Namespace, path: str, op: str,
                    params: dict[str, Any], stem: str) -> tuple[Any, str | None, bool]:
    """Route a project to ONE bridge: pin the running owner when there is one
    (sending nothing), else apply `bn load`'s routing rule and send *op*.

    Returns `(result, pinned_selector, attached)`; an attached result is the
    `project attach` row, otherwise the bridge's own result with `instance` set.
    """
    # A running bridge already owns this project -> pin it and send nothing.
    # Two processes on one project are unsafe (BN refuses `create_file` and
    # corrupts a shared database), so the owner always beats spawning or
    # routing elsewhere.
    instances = cli.list_instances()
    owner = _owning_instance(instances, path)
    if owner is not None:
        selector = _pin_instance(owner)
        return {"attached": True, "instance_id": selector, "path": path}, selector, True
    # `bn load`'s routing rule. BN_INSTANCE and -i/--instance-id name an
    # instance explicitly; only the sticky pin is ambient and refused.
    explicitly_routed = bool(
        getattr(args, "_explicit_instance", False)
        or getattr(args, "_explicit_instance_id", False)
        or (getattr(args, "instance", None)
            and not getattr(args, "_sticky_instance", False))
    )
    if not explicitly_routed and len(instances) > 1:
        raise BridgeError(
            f"bn {stem.replace('-', ' ')} refuses ambient instance routing while multiple "
            "bridges are running. Pass an explicit -i/--instance (or export BN_INSTANCE) "
            "for an existing bridge, or --instance-id to create a named bridge; the "
            "`bn instance use` pin is not used to pick where a project lands.\n"
            "Running instances:\n" + _format_instance_choices(instances)
        )
    response = cli.send_request(
        op,
        params=params,
        instance_id=getattr(args, "instance", None),
        spawn_missing_named=True,
    )
    result = cli.unwrap_result(response, op)
    # The answering instance (#996 stamp) becomes the sticky pin, and the
    # caller's cwd is associated with it exactly as `bn session start` does.
    answered = response.get("instance") if isinstance(response, dict) else None
    pinned = answered if isinstance(answered, str) and answered else getattr(args, "instance", None)
    if pinned:
        _associate_project_roots(pinned, [os.getcwd()])
        cli.session_state.update(instance_id=pinned)
        if isinstance(result, dict):
            result.setdefault("instance", pinned)
    return result, pinned, False


_INSTANCE_ID_ARG = arg("--instance-id", dest="instance", default=argparse.SUPPRESS,
                       metavar="INSTANCE",
                       help="ID for the bridge instance to auto-spawn (alias of --instance)")


@command("project", "open", help="Open a .bnpr project in a bridge (or attach to the one that owns it)",
         args=[
             arg("path", help="Path to the .bnpr project directory (or its project.bnpm)"),
             _INSTANCE_ID_ARG,
         ])
def _project_open(args: argparse.Namespace) -> int:
    cli._require_nonempty_instance(args)
    path = _canonical_project_path(args.path)
    result, _pinned, attached = _attach_or_open(args, path, "project_open", {"path": path},
                                                "project-open")
    cli._emit_result(
        args, result,
        text_renderer=_render_project_attach_text if attached else _render_project_info_text,
        stem="project-open")
    return 0


def _lookup_rows(lookup: list[Any]) -> list[dict[str, Any]]:
    return [{"path": lp.path_in_root, "source": lp.source} for lp in lookup]


def _cache_fallback_problem(directory: str) -> str | None:
    """Why the cache `projects` directory cannot be created or written, judged
    without creating anything: the nearest existing ancestor must be a
    writable directory. None when it can."""
    probe = directory
    while not os.path.lexists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    if not os.path.isdir(probe):
        return f"{probe} exists and is not a directory"
    if not os.access(probe, os.W_OK | os.X_OK):
        return f"{probe} is not writable"
    return None


def _unusable_cache_fallback(directory: str, problem: str, rootfs_parent: str) -> BridgeError:
    return BridgeError(
        f"cannot create the cache projects directory {directory}: {problem}. The rootfs's "
        f"parent {rootfs_parent} is not writable either, so there is nowhere to create the "
        "project; set BN_CACHE_DIR to a writable directory.")


def _init_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Everything `project init` decides without a bridge: the rootfs, the
    project path (existing preferred, existing fallback, else create at the
    preferred location when its parent is writable, else the fallback) and the
    loader's lookup paths with their library candidates."""
    explicit = args.root is not None
    start = os.path.realpath(os.path.expanduser(args.root)) if explicit else os.getcwd()
    if explicit and not os.path.isdir(start):
        raise BridgeError(f"not a directory: {start}")
    info = rootfs_layout.detect_root(start, explicit=explicit)
    if info is None:
        raise BridgeError(
            f"not inside a firmware root filesystem (looked from {start} up to /); "
            "pass an explicit firmware root: bn project init ROOT")
    preferred, fallback = rootfs_layout.project_location(info.root, str(paths.projects_dir()))
    # `fallback_from` names the unwritable parent when the cache fallback is
    # chosen over the preferred location, so the reason is visible (#1075).
    fallback_from = None
    if os.path.isdir(preferred):
        project_path, exists = preferred, True
    elif os.path.isdir(fallback):
        project_path, exists = fallback, True
    elif os.access(os.path.dirname(preferred) or os.sep, os.W_OK):
        project_path, exists = preferred, False
    else:
        project_path, exists = fallback, False
        fallback_from = os.path.dirname(preferred) or os.sep
        # Planned here, so `--dry-run` cannot promise `[new]` for a location
        # the real run would fail to create (#1075 review).
        problem = _cache_fallback_problem(os.path.dirname(fallback))
        if problem is not None:
            raise _unusable_cache_fallback(os.path.dirname(fallback), problem, fallback_from)
    # An explicit root that matches no detector is still usable as a project
    # root; its loader is unknown, so library sync has nothing to scan.
    lookup = rootfs_layout.lookup_paths(info.root, info.platform)
    candidates, notes = rootfs_layout.scan_libraries(info.root, lookup)
    return {
        "root": info.root,
        "platform": info.platform,
        "markers": list(info.markers),
        "project_path": project_path,
        "project_exists": exists,
        "fallback_from": fallback_from,
        "name": args.name or os.path.basename(info.root.rstrip(os.sep)) or "rootfs",
        "lookup_paths": _lookup_rows(lookup),
        "candidate_libraries": len(candidates),
        "notes": list(notes),
    }


@command("project", "init",
         help="Create or open the project for an extracted firmware root filesystem and "
              "import its loader-visible shared libraries",
         args=[
             arg("root", nargs="?", default=None,
                 help="Rootfs directory (default: detected by walking up from the cwd)"),
             arg("--name", default=None, help="Project name (default: the root's basename)"),
             arg("--no-libs", action="store_true",
                 help="Do not import the shared libraries from the loader's lookup paths"),
             arg("--dry-run", action="store_true",
                 help="Print what would be detected, created and imported; touch nothing"),
             _INSTANCE_ID_ARG,
         ])
def _project_init(args: argparse.Namespace) -> int:
    cli._require_nonempty_instance(args)
    plan = _init_plan(args)
    if args.dry_run:
        result: dict[str, Any] = {"dry_run": True, **plan}
        cli._emit_result(args, result, text_renderer=_render_project_init_text,
                         stem="project-init")
        return 0
    path = plan["project_path"]
    if plan["project_exists"]:
        op, params = "project_open", {"path": path}
    else:
        # The cache fallback parent does not exist on a fresh machine, and the
        # bridge opens it (its creation gate) before creating the project, so
        # it died with FileNotFoundError (#1075). Create it here, owner-private
        # like every other cache directory.
        if plan["fallback_from"] is not None:
            directory = os.path.dirname(path)
            try:
                paths.ensure_private_dir(Path(directory))
            except OSError as exc:  # raced with the plan's own check
                raise _unusable_cache_fallback(
                    directory, exc.strerror or str(exc), plan["fallback_from"]) from exc
        op, params = "project_create", {
            "path": path, "name": plan["name"],
            "rootfs": {"root": plan["root"], "platform": plan["platform"]},
        }
    opened, pinned, attached = _attach_or_open(args, path, op, params, "project-init")
    result = {
        "dry_run": False, **plan,
        "created": bool(isinstance(opened, dict) and opened.get("created")),
        "attached": attached,
        "instance": pinned,
        "project": opened,
        "sync": None,
        "sync_error": None,
    }
    rc = 0
    if not args.no_libs and plan["platform"] != "unknown":
        try:
            response = cli.send_request(
                "project_sync_libs",
                params={"root": plan["root"], "platform": plan["platform"], "dry_run": False},
                instance_id=pinned,
            )
            result["sync"] = cli.unwrap_result(response, "project_sync_libs")
        except BridgeError as exc:
            # The project IS open (and pinned); only the import step failed.
            # Partial, like `session start`: the summary so far, then exit 1.
            result["sync_error"] = str(exc)
            print(f"error: project_sync_libs: {exc}", file=sys.stderr)
            rc = 1
    cli._emit_result(args, result, text_renderer=_render_project_init_text, stem="project-init")
    return rc


@command("project", "libs", help="List the rootfs libraries in the project (or sync them)",
         args=[
             arg("--sync", action="store_true",
                 help="Import the loader-visible shared libraries that are not in the project yet"),
             arg("--root", default=None, metavar="DIR",
                 help="Rootfs to scan (default: the one stored in the project)"),
             arg("--dry-run", action="store_true",
                 help="With --sync: report what would be imported without importing"),
         ])
def _project_libs(args: argparse.Namespace) -> int:
    if not args.sync:
        if args.root is not None or args.dry_run:
            raise BridgeError("--root and --dry-run only apply with --sync")
        return _call(
            args,
            "project_libs",
            {},
            require_target=False,
            text_renderer=_render_project_libs_text,
            stem="project-libs",
        )
    params: dict[str, Any] = {"dry_run": bool(args.dry_run)}
    if args.root is not None:
        if not str(args.root).strip():
            raise BridgeError("--root is empty: pass the rootfs directory")
        params["root"] = os.path.realpath(os.path.expanduser(args.root))
    return _call(
        args,
        "project_sync_libs",
        params,
        require_target=False,
        text_renderer=_render_project_sync_text,
        stem="project-libs-sync",
    )


@command("project", "link",
         help="Bind the target's imported symbols to the project libraries that export them",
         target=True, fmt="json",
         args=[
             preview_arg("Plan the links and report them without writing"),
             arg("--relink", action="store_true",
                 help="Also reassign imports that already have a library"),
             *mutation_output_args(),
         ])
def _project_link(args: argparse.Namespace) -> int:
    # The bridge op takes `dry_run`, not `preview`, so the flag is mapped here
    # instead of through `_mutate(preview=...)`.
    return _mutate(
        args,
        "project_link",
        {"dry_run": bool(args.preview), "relink": bool(args.relink)},
        require_target=True,
        detail_renderer=_render_project_link_text,
        summary_transform=_project_link_summary,
        stem="project-link",
    )


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


def _project_load_exit_code(result: Any) -> int:
    link = result.get("link")
    if link is None:
        return 0
    return 3 if link.get("status") in FAILED_MUTATION_STATUSES or link.get("success") is False else 0


@command("project", "load", help="Open a project file as a target",
         args=[
             arg("selector", help="Project file: id, >=8-char id prefix, path-in-project, or "
                                  "unique file name"),
             arg("--no-bndb", action="store_true",
                 help="Open the raw file even when a sibling <name>.bndb exists in its folder"),
             arg("--quick", "--no-analysis", action="store_true",
                 help="Load without full analysis (fast); `bn refresh` completes it"),
             arg("--no-link", action="store_true",
                 help="Skip binding the file's imports to the project's rootfs libraries"),
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
            "link": not args.no_link,
            "workdir": os.getcwd(),
        },
        require_target=False,
        text_renderer=_render_project_load_text,
        stem="project-load",
        result_exit_code=_project_load_exit_code,
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


def _finding_document(source: str) -> Any:
    """Read JSON only; evidence paths and prose are never executed."""
    if source == "-":
        try:
            raw = sys.stdin.read()
        except (OSError, UnicodeError) as exc:
            raise BridgeError(f"Could not read finding document from stdin: {exc}") from None
        label = "stdin"
    else:
        raw = cli.read_text_input(Path(source).expanduser(), what="Finding document")
        label = source
    return cli.decode_json_input(raw, refusal=f"Invalid JSON in finding document ({label})")


def _finding_call(args: argparse.Namespace, action: str, params: dict[str, Any],
                  renderer: Any) -> int:
    writing = action in {"add", "update", "review"}
    if writing:
        args._mutation_call = True
    return _call(
        args, f"project_finding_{action}", params, require_target=False,
        text_renderer=renderer, stem=f"project-finding-{action}",
        result_exit_code=(lambda result: cli._mutation_exit_code(result, _project_finding_summary))
        if writing else None,
        spill_status=_project_finding_summary if writing else None,
        spill_status_renderer=_render_mutation_summary_text if writing else None,
    )


@command("project", "finding", "add", help="Record an evidence-backed claim (not automatic acceptance)",
         args=[arg("document", metavar="DOCUMENT", help="JSON document path, or - for stdin"),
               arg("--author", required=True, help="Declared author identity")])
def _project_finding_add(args: argparse.Namespace) -> int:
    return _finding_call(args, "add",
                         {"document": _finding_document(args.document), "author": args.author},
                         _render_project_finding_text)


@command("project", "finding", "update", help="Replace a claim, preserving history and resetting review",
         args=[arg("id"), arg("document", metavar="DOCUMENT", help="JSON document path, or - for stdin"),
               arg("--expected-revision", required=True, type=cli._positive_int),
               arg("--author", required=True)])
def _project_finding_update(args: argparse.Namespace) -> int:
    return _finding_call(args, "update",
                         {"id": args.id, "document": _finding_document(args.document),
                          "expected_revision": args.expected_revision, "author": args.author},
                         _render_project_finding_text)


@command("project", "finding", "review", help="Append an explicit independent review",
         args=[arg("id"), arg("document", metavar="DOCUMENT", help="Review JSON path, or - for stdin"),
               arg("--expected-revision", required=True, type=cli._positive_int),
               arg("--reviewer", required=True)])
def _project_finding_review(args: argparse.Namespace) -> int:
    return _finding_call(args, "review",
                         {"id": args.id, "document": _finding_document(args.document),
                          "expected_revision": args.expected_revision, "reviewer": args.reviewer},
                         _render_project_finding_text)


@command("project", "finding", "show", help="Show a finding and its evidence",
         args=[arg("id"), arg("--revision", type=cli._positive_int, default=None)])
def _project_finding_show(args: argparse.Namespace) -> int:
    return _finding_call(args, "show", {"id": args.id, "revision": args.revision},
                         _render_project_finding_text)


@command("project", "finding", "check", help="Check freshness without accepting or proving the claim",
         args=[arg("id"), arg("--revision", type=cli._positive_int, default=None)])
def _project_finding_check(args: argparse.Namespace) -> int:
    return _finding_call(args, "check", {"id": args.id, "revision": args.revision},
                         _render_project_finding_check_text)


def _finding_page_args() -> list[Any]:
    return [arg("--offset", type=cli._non_negative_int, default=0),
            arg("--limit", type=cli._positive_int, default=50, help="Page size (maximum 200)")]


@command("project", "finding", "history", help="Show immutable finding revisions",
         args=[arg("id"), *_finding_page_args()])
def _project_finding_history(args: argparse.Namespace) -> int:
    return _finding_call(args, "history",
                         {"id": args.id, "offset": args.offset, "limit": args.limit},
                         _render_project_findings_text)


def _finding_filter_args() -> list[Any]:
    return [
        arg("--file", default=None, help="Project file selector"),
        arg("--function", default=None, help="Function address or label"),
        arg("--query", default=None, help="Case-insensitive text search"),
        arg("--type", action="append", default=None, help="Finding type (repeat for OR filtering)"),
        arg("--status", choices=("proposed", "accepted", "rejected", "disputed", "superseded"),
            default=None),
        arg("--freshness", choices=("current", "stale", "unknown"), default=None),
        *_finding_page_args(),
    ]


def _finding_filters(args: argparse.Namespace) -> dict[str, Any]:
    return {name: getattr(args, name)
            for name in ("file", "function", "query", "type", "status", "freshness", "offset", "limit")}


@command("project", "finding", "list", help="List findings with separate review and freshness states",
         args=_finding_filter_args())
def _project_finding_list(args: argparse.Namespace) -> int:
    return _finding_call(args, "list", _finding_filters(args), _render_project_findings_text)


@command("project", "finding", "context", help="Group reusable findings and findings needing review",
         args=_finding_filter_args())
def _project_finding_context(args: argparse.Namespace) -> int:
    return _finding_call(args, "context", _finding_filters(args), _render_project_finding_context_text)


@command("project", "finding", "types", help="Show standard and recorded finding types and actions")
def _project_finding_types(args: argparse.Namespace) -> int:
    return _finding_call(args, "types", {}, _render_project_finding_types_text)
