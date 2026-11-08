"""Binary Ninja project (`.bnpr`) support for the bridge.

Contracts, measured BN facts and the ownership model live in `BNPR_PLAN.md` at the
repo root; the section numbers in comments below refer to it. In one line: a
project is owned by exactly one bridge (F6/F7), each file loaded from it is an
ordinary target, and a project view is saved through the project API (F3), never
by filename suffix.

This module never imports `bridge` (import cycle) and imports `binaryninja` only
inside functions (tests substitute a fake module). The handlers that need the
bridge's module-level view bookkeeping (`_project_load`, `_project_close`) are
methods on `BinaryNinjaBridge`; everything else lives here as free functions that
take the bridge instance.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import bridge_registry_path, instances_dir
from .proc_identity import identity_verdict

# A project database blob has no extension (F1); its kind is read off the bytes.
_SQLITE_MAGIC = b"SQLite format 3\x00"
# §5.1: an id prefix shorter than this is not a selector (UUID hex is dense
# enough that 8 characters are unique in any realistic project, and a shorter
# prefix invites `-t 7` style accidents).
MIN_ID_PREFIX = 8
# Cap on the "did you mean" listing appended to selector errors.
_LISTING_CAP = 50


def canonical_project_path(path: str | os.PathLike[str]) -> Path:
    """The `.bnpr` DIRECTORY for *path*, which may name the directory or its
    `project.bnpm` metadata file (BN accepts both; the registry publishes one)."""
    resolved = Path(path).expanduser().resolve()
    if resolved.name == "project.bnpm":
        resolved = resolved.parent
    return resolved


@dataclass(slots=True)
class ProjectState:
    project: Any  # the live binaryninja.Project
    path: str  # canonical .bnpr directory
    project_id: str
    name: str


class ProjectManager:
    """At most ONE open project per bridge (§3).

    Holds the strong reference to the `Project` handle for the bridge's lifetime,
    resolves §5.1 file selectors, builds §5.2 file rows, and performs the F3 save
    recipe. Thread-safe for the handful of state transitions it has; the heavy
    BN calls run under whichever dispatcher lock the op declared.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: ProjectState | None = None

    # -- state -------------------------------------------------------------

    @property
    def current(self) -> ProjectState | None:
        with self._lock:
            return self._state

    def require(self) -> ProjectState:
        state = self.current
        if state is None:
            raise RuntimeError(
                "No project is open in this bridge; run `bn project open <path.bnpr>` first"
            )
        return state

    def open(self, path: str, *, self_pid: int) -> tuple[ProjectState, bool]:
        """Open *path*; ``(state, opened)`` where ``opened`` is False when the same
        project was already open here (F8 makes a second BN open a no-op anyway,
        but the registry and the result should say which happened)."""
        canonical = canonical_project_path(path)
        with self._lock:
            state = self._state
            if state is not None:
                if state.path == str(canonical):
                    return state, False
                raise RuntimeError(
                    f"A different project is already open in this bridge: {state.path}. "
                    "Run `bn project close` first (one project per bridge instance)."
                )
            if not (canonical / "project.bnpm").is_file():
                raise RuntimeError(
                    f"Not a Binary Ninja project: {canonical} (no project.bnpm inside). "
                    "Pass the .bnpr directory or its project.bnpm file."
                )
            # §5.5: refuse a project another LIVE bridge publishes, before BN
            # opens it (BN itself would still let a second process open it, and
            # only fail later on the first create_file -- F6).
            owner = find_other_owner(str(canonical), self_pid=self_pid)
            if owner is not None:
                raise RuntimeError(
                    f"project already open in bridge instance {owner['instance_id']} "
                    f"(pid {owner['pid']}): run `bn project attach {canonical}` "
                    f"(or pass -i {owner['instance_id']})"
                )
            import binaryninja

            try:
                project = binaryninja.Project.open_project(str(canonical))
            except Exception as exc:  # noqa: BLE001 - ProjectException and friends
                raise RuntimeError(
                    f"Binary Ninja could not open project {canonical}: {exc}. "
                    "Projects require a Commercial or Ultimate licence."
                ) from exc
            if project is None:
                raise RuntimeError(f"Binary Ninja could not open project {canonical}")
            state = ProjectState(
                project=project,
                path=str(canonical),
                project_id=str(getattr(project, "id", "") or ""),
                name=str(getattr(project, "name", "") or canonical.stem),
            )
            self._state = state
            return state, True

    def close(self) -> ProjectState | None:
        """Close the project handle. Callers close the project's TARGETS first;
        this only releases the project itself."""
        with self._lock:
            state = self._state
            self._state = None
        if state is None:
            return None
        try:
            state.project.close()
        except Exception:  # noqa: BLE001 - already dropped from our state; report nothing
            pass
        return state

    # -- file lookups ------------------------------------------------------

    def files(self) -> list[Any]:
        return list(self.require().project.files)

    @staticmethod
    def path_in_project(pf: Any) -> str:
        """`/folder/name` (F1 shape). Computed from the folder chain when BN's
        own answer is unavailable, so a row never lacks a path."""
        try:
            direct = pf.get_path_in_project()
        except Exception:  # noqa: BLE001
            direct = None
        if direct:
            return str(direct) if str(direct).startswith("/") else "/" + str(direct)
        parts = [str(pf.name)]
        folder = getattr(pf, "folder", None)
        while folder is not None:
            parts.append(str(folder.name))
            folder = getattr(folder, "parent", None)
        return "/" + "/".join(reversed(parts))

    @staticmethod
    def folder_path(folder: Any) -> str:
        """`/a/b` for a folder, `""` for the root."""
        parts: list[str] = []
        while folder is not None:
            parts.append(str(folder.name))
            folder = getattr(folder, "parent", None)
        return "/" + "/".join(reversed(parts)) if parts else ""

    def resolve_file(self, selector: Any) -> Any:
        """§5.1 file selector -> ProjectFile, or a RuntimeError that lists the
        candidates (ambiguous) or every file (unknown)."""
        if not isinstance(selector, str) or not selector.strip():
            raise RuntimeError(
                "project file selector must be a non-empty string: a file id, an id "
                f"prefix (>= {MIN_ID_PREFIX} chars), a path in the project, or a unique file name"
            )
        wanted = selector.strip()
        files = self.files()
        by_id = [pf for pf in files if str(pf.id) == wanted]
        if len(by_id) == 1:
            return by_id[0]
        path_wanted = wanted if wanted.startswith("/") else "/" + wanted
        by_path = [pf for pf in files if self.path_in_project(pf) == path_wanted]
        if len(by_path) == 1:
            return by_path[0]
        if len(by_path) > 1:
            raise RuntimeError(self._ambiguous(wanted, by_path))
        by_name = [pf for pf in files if str(pf.name) == wanted]
        if len(by_name) == 1:
            return by_name[0]
        if len(by_name) > 1:
            raise RuntimeError(self._ambiguous(wanted, by_name))
        if len(wanted) >= MIN_ID_PREFIX:
            by_prefix = [pf for pf in files if str(pf.id).startswith(wanted)]
            if len(by_prefix) == 1:
                return by_prefix[0]
            if len(by_prefix) > 1:
                raise RuntimeError(self._ambiguous(wanted, by_prefix))
        raise RuntimeError(
            "\n".join(
                [
                    f"No project file matches {wanted!r}. Project files:",
                    *self._listing(files),
                ]
            )
        )

    def _ambiguous(self, selector: str, matches: list[Any]) -> str:
        return "\n".join(
            [
                f"Ambiguous project file selector: {selector!r} matches {len(matches)} files",
                *self._listing(matches),
            ]
        )

    def _listing(self, files: list[Any]) -> list[str]:
        rows = [f"  {pf.id}  {self.path_in_project(pf)}" for pf in files[:_LISTING_CAP]]
        if len(files) > _LISTING_CAP:
            rows.append(f"  ... {len(files) - _LISTING_CAP} more")
        return rows

    def sibling_database(self, pf: Any) -> Any | None:
        """The `<name>.bndb` project file in *pf*'s folder, if any: the project
        analogue of the adjacent-`.bndb` preference (#717). None for a file that
        is itself a `.bndb`."""
        name = str(pf.name)
        if name.endswith(".bndb"):
            return None
        wanted = name + ".bndb"
        project = self.require().project
        try:
            siblings = project.get_files_in_folder(getattr(pf, "folder", None))
        except Exception:  # noqa: BLE001 - fall back to the full list
            siblings = [
                f
                for f in project.files
                if self.folder_path(getattr(f, "folder", None))
                == self.folder_path(getattr(pf, "folder", None))
            ]
        matches = [f for f in siblings if str(f.name) == wanted]
        return matches[0] if len(matches) == 1 else None

    def ensure_folder(self, path: str | None) -> Any | None:
        """The folder at path-in-project *path*, created level by level when
        missing. None (the root) for empty/`/`."""
        if not path or path.strip("/") == "":
            return None
        project = self.require().project
        parent = None
        for component in [c for c in path.split("/") if c]:
            existing = [
                f
                for f in project.folders
                if str(f.name) == component
                and self.folder_path(getattr(f, "parent", None))
                == self.folder_path(parent)
            ]
            if existing:
                parent = existing[0]
                continue
            parent = project.create_folder(parent, component)
        return parent

    # -- rows --------------------------------------------------------------

    def file_row(
        self, pf: Any, open_targets: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """§5.2 file row. *open_targets* maps project_file_id -> target_id."""
        blob = str(getattr(pf, "path_on_disk", "") or "")
        exists = bool(getattr(pf, "exists_on_disk", False))
        size: int | None = None
        is_database = False
        if exists and blob:
            try:
                size = os.path.getsize(blob)
                with open(blob, "rb") as fh:
                    is_database = fh.read(len(_SQLITE_MAGIC)) == _SQLITE_MAGIC
            except OSError:
                size = None
        sibling = self.sibling_database(pf)
        return {
            "id": str(pf.id),
            "name": str(pf.name),
            "path": self.path_in_project(pf),
            "folder": self.folder_path(getattr(pf, "folder", None)),
            "description": str(getattr(pf, "description", "") or ""),
            "size": size,
            "exists_on_disk": exists,
            "is_database": is_database,
            "open_target_id": (open_targets or {}).get(str(pf.id)),
            "sibling_database_id": str(sibling.id) if sibling is not None else None,
        }

    # -- save (F3) ---------------------------------------------------------

    def create_database_file(self, bv: Any, pf: Any) -> Any:
        """Create the project-tracked `<name>.bndb` for the raw project file *pf*
        and write *bv*'s database into it. Returns the new ProjectFile.

        F3: `create_file(b"")` makes a row but neither a blob nor its parent
        directory; `create_database` needs the directory. A write that fails
        leaves no half-registered file behind.
        """
        project = self.require().project
        name = f"{pf.name}.bndb"
        try:
            new = project.create_file(
                b"",
                getattr(pf, "folder", None),
                name,
                f"analysis database for {pf.name}",
            )
        except Exception as exc:  # noqa: BLE001 - ProjectException
            raise RuntimeError(
                f"Binary Ninja could not add {name} to the project: {exc}"
            ) from exc
        dest = str(new.path_on_disk)
        try:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            created = bv.create_database(dest)
        except Exception as exc:  # noqa: BLE001 - surface BN I/O errors, after cleanup
            self._discard_file(project, new)
            raise RuntimeError(
                f"Failed to save project database to {dest}: {exc}"
            ) from exc
        if created is False or not Path(dest).exists():
            self._discard_file(project, new)
            raise RuntimeError(
                f"Failed to save project database to {dest}: Binary Ninja reported no "
                "file was written"
            )
        return new

    @staticmethod
    def _discard_file(project: Any, pf: Any) -> None:
        try:
            project.delete_file(pf)
        except Exception:  # noqa: BLE001 - best effort; the primary error is reported
            pass


# -- ownership (§5.5) ------------------------------------------------------


def _record_is_live(payload: dict[str, Any], pid: int) -> bool:
    verdict = identity_verdict(payload, pid)
    if verdict == "proven":
        return True
    if verdict == "mismatch":
        return False
    # "unrecorded": no identity to compare; a pid that still answers is treated
    # as live (the conservative direction -- refusing to open costs one command,
    # a wrong open costs F6 failures later).
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def find_other_owner(project_path: str, *, self_pid: int) -> dict[str, Any] | None:
    """The live bridge (other than *self_pid*) whose registry publishes
    *project_path*, as ``{"instance_id", "pid", "registry"}``, or None."""
    candidates: list[Path] = [bridge_registry_path()]
    try:
        candidates.extend(sorted(instances_dir().glob("*.json")))
    except OSError:
        pass
    for registry in candidates:
        try:
            payload = json.loads(registry.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        project = payload.get("project")
        if not isinstance(project, dict) or project.get("path") != project_path:
            continue
        pid = payload.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid == self_pid:
            continue
        if _record_is_live(payload, pid):
            return {
                "instance_id": str(payload.get("instance_id") or registry.stem),
                "pid": pid,
                "registry": str(registry),
            }
    return None


# -- op handlers (bound in bridge.py) --------------------------------------


def _open_project_targets(bridge: Any) -> list[dict[str, Any]]:
    """Target rows that belong to the open project (one refresh)."""
    return [row for row in bridge.targets.refresh() if row.get("project_file_id")]


def project_info(bridge: Any) -> dict[str, Any]:
    state = bridge.projects.require()
    project = state.project
    try:
        file_count = len(project.files)
    except Exception:  # noqa: BLE001
        file_count = None
    try:
        folder_count = len(project.folders)
    except Exception:  # noqa: BLE001
        folder_count = None
    return {
        "path": state.path,
        "id": state.project_id,
        "name": state.name,
        "description": str(getattr(project, "description", "") or ""),
        "file_count": file_count,
        "folder_count": folder_count,
        "open_targets": [
            {
                "target_id": row.get("target_id"),
                "view_id": row.get("view_id"),
                "project_file_id": row.get("project_file_id"),
                "project_path": row.get("project_path"),
                "selector": row.get("selector"),
            }
            for row in _open_project_targets(bridge)
        ],
    }


def project_open(bridge: Any, path: Any) -> dict[str, Any]:
    if not isinstance(path, str) or not path.strip():
        raise RuntimeError(
            "project_open: path must be the .bnpr directory (or its project.bnpm)"
        )
    _, opened = bridge.projects.open(path, self_pid=os.getpid())
    bridge._write_registry()
    info = project_info(bridge)
    info["opened"] = opened
    return info


def project_files(
    bridge: Any, folder: Any = None, recursive: Any = True
) -> dict[str, Any]:
    manager = bridge.projects
    state = manager.require()
    if folder is not None and not isinstance(folder, str):
        raise RuntimeError(
            "project_files: folder must be a path in the project or null"
        )
    wanted = ("/" + folder.strip("/")) if folder and folder.strip("/") else ""
    open_targets = {
        str(row["project_file_id"]): str(row.get("target_id") or "")
        for row in _open_project_targets(bridge)
    }
    rows: list[dict[str, Any]] = []
    for pf in state.project.files:
        here = manager.folder_path(getattr(pf, "folder", None))
        if wanted and not (
            here == wanted or (recursive and here.startswith(wanted + "/"))
        ):
            continue
        if not wanted and not recursive and here:
            continue
        rows.append(manager.file_row(pf, open_targets))
    rows.sort(key=lambda r: r["path"])
    folders = sorted(
        (
            {"id": str(f.id), "path": manager.folder_path(f)}
            for f in state.project.folders
        ),
        key=lambda r: r["path"],
    )
    return {"files": rows, "folders": folders}


def project_import(
    bridge: Any,
    path: Any,
    folder: Any = None,
    name: Any = None,
    description: Any = None,
) -> dict[str, Any]:
    manager = bridge.projects
    state = manager.require()
    if not isinstance(path, str) or not path.strip():
        raise RuntimeError("project_import: path must be the file to import")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"project_import: not a file: {source}")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise RuntimeError("project_import: name must be a non-empty string when given")
    target_folder = manager.ensure_folder(folder if isinstance(folder, str) else None)
    try:
        pf = state.project.create_file_from_path(
            str(source),
            target_folder,
            (name or source.name).strip(),
            str(description or ""),
        )
    except Exception as exc:  # noqa: BLE001 - ProjectException
        raise RuntimeError(f"Binary Ninja could not import {source}: {exc}") from exc
    bridge._write_registry()
    return manager.file_row(pf)


def project_export(bridge: Any, selector: Any, dest: Any) -> dict[str, Any]:
    manager = bridge.projects
    pf = manager.resolve_file(selector)
    if not isinstance(dest, str) or not dest.strip():
        raise RuntimeError("project_export: dest must be the output path")
    out = Path(dest).expanduser().resolve()
    if out.exists():
        raise RuntimeError(
            f"project_export: {out} already exists; remove it or choose another path"
        )
    if not out.parent.is_dir():
        raise RuntimeError(f"project_export: directory does not exist: {out.parent}")
    try:
        ok = pf.export(str(out))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Binary Ninja could not export {manager.path_in_project(pf)}: {exc}"
        ) from exc
    if ok is False or not out.exists():
        raise RuntimeError(
            f"Binary Ninja reported no file written when exporting {manager.path_in_project(pf)} to {out}"
        )
    return {
        "exported": True,
        "dest": str(out),
        "project_file_id": str(pf.id),
        "project_path": manager.path_in_project(pf),
    }


def project_delete(bridge: Any, selector: Any) -> dict[str, Any]:
    manager = bridge.projects
    state = manager.require()
    pf = manager.resolve_file(selector)
    open_rows = [
        row
        for row in _open_project_targets(bridge)
        if row.get("project_file_id") == str(pf.id)
    ]
    if open_rows:
        sel = open_rows[0].get("selector") or open_rows[0].get("target_id")
        raise RuntimeError(
            f"project file {manager.path_in_project(pf)} is open as target {sel}; "
            "close it first (`bn close -t <selector>`)"
        )
    path = manager.path_in_project(pf)
    file_id = str(pf.id)
    try:
        ok = state.project.delete_file(pf)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Binary Ninja could not delete {path}: {exc}") from exc
    if ok is False:
        raise RuntimeError(f"Binary Ninja refused to delete {path}")
    bridge._write_registry()
    return {"deleted": True, "project_file_id": file_id, "project_path": path}


def save_project_database(bridge: Any, bv: Any, path: Any) -> dict[str, Any]:
    """§5.2 project branch of `save_database`. The caller established
    ``bv.project_file is not None`` and holds the write lock."""
    manager = bridge.projects
    if path is not None:
        raise RuntimeError(
            "project-backed target: bn save takes no output path; use "
            "`bn project export <selector> <dest>`"
        )
    pf = bv.project_file
    # F4: after a first save BN keeps reporting the RAW file here, but the
    # record we published knows the database file; prefer that.
    record = bridge.targets.project_binding(bv)
    file_id = record[0] if record else str(pf.id)
    project_path = record[1] if record else manager.path_in_project(pf)
    has_database = bool(getattr(bv.file, "has_database", False))
    if has_database:
        try:
            ok = bv.file.save_auto_snapshot()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to save project database snapshot: {exc}"
            ) from exc
        if ok is False:
            raise RuntimeError(
                "Binary Ninja could not write the project database snapshot"
            )
        bridge.targets.clear_dirty(bv)
        blob = str(getattr(bv.file, "filename", "") or "")
        return {
            "ok": True,
            "saved": True,
            "path": blob,
            "project_file_id": file_id,
            "project_path": project_path,
            "mode": "snapshot",
        }

    # Raw project file: the F3 recipe. Refuse to create a SECOND database for a
    # binary that already has its `<name>.bndb` sibling (F9: BN would allow the
    # duplicate, and two databases for one binary is never what a save means).
    sibling = manager.sibling_database(pf)
    if sibling is not None:
        raise RuntimeError(
            f"{manager.path_in_project(pf)} already has a database file "
            f"{manager.path_in_project(sibling)} ({sibling.id}); load that file instead. "
            "A second database for one binary is not created implicitly."
        )
    new = manager.create_database_file(bv, pf)
    new_id = str(new.id)
    new_path = manager.path_in_project(new)
    blob = str(new.path_on_disk)
    bridge.targets.clear_dirty(bv)
    bridge.targets.note_database(bv, blob)
    bridge.targets.note_project_database(bv, new_id, new_path)
    bridge._write_registry()
    return {
        "ok": True,
        "saved": True,
        "path": blob,
        "project_file_id": new_id,
        "project_path": new_path,
        "mode": "created",
    }
