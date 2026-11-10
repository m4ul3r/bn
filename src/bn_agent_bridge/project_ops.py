"""Binary Ninja project (`.bnpr`) support for the bridge.

A project is owned by exactly one bridge: a second process cannot create files
while another owns it, and concurrent database writers can corrupt a database.
Each file loaded from it is an ordinary target, saved through the project API
rather than by filename suffix.

This module never imports `bridge` (import cycle) and imports `binaryninja` only
inside functions (tests substitute a fake module). The handlers that need the
bridge's module-level view bookkeeping (`_project_load`, `_project_close`) are
methods on `BinaryNinjaBridge`; everything else lives here as free functions that
take the bridge instance.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import bridge_registry_path, instances_dir
from .proc_identity import identity_verdict

# A project database blob has no extension; its kind is read off the bytes.
_SQLITE_MAGIC = b"SQLite format 3\x00"
# An id prefix shorter than this is not a selector (UUID hex is dense
# enough that 8 characters are unique in any realistic project, and a shorter
# prefix invites `-t 7` style accidents).
MIN_ID_PREFIX = 8
# Cap on the listing appended to an AMBIGUOUS selector error (the matches
# themselves) ...
_LISTING_CAP = 50
# ... and on the "nearest by file name" suggestions a MISS gets (#1077): a
# miss used to dump up to _LISTING_CAP rows in storage order, findings ledger
# records included, which told the reader nothing about what they mistyped.
_SUGGESTION_CAP = 5
# Project metadata key holding the firmware rootfs inventory (root, platform,
# lookup paths, imported libraries) that `project_sync_libs` writes and
# `project_link` reads.
ROOTFS_METADATA_KEY = "bn.rootfs"
# The format of that blob. It is project-level data any tool or any other bn
# version can write, so it is untrusted input: `validate_rootfs_metadata` is
# the one gate every reader goes through (#1071).
ROOTFS_METADATA_VERSION = 1
# A document this bn cannot read cannot be rebuilt in place either: the `libs`
# map is what says which project files are the inventory, so a sync over a
# fresh document would import every library again beside the old copies.
_ROOTFS_REINIT = ("this bn cannot rebuild an inventory it cannot read: upgrade bn, or move "
                  "the project aside and re-run `bn project init <root>`")
# What `_rootfs_raw` returns for an ABSENT key, so no stored value -- not even
# a stored null -- can be mistaken for "this project records no rootfs".
_NO_METADATA: Any = object()
# Project metadata key recording, per linked target, which project file each
# of its NEEDED libraries is bound to (#1078). Written by `project_link` and
# the load-time auto-link; read by `project delete` to name the targets a
# provider serves, saved databases included.
LINKS_METADATA_KEY = "bn.links"
LINKS_METADATA_VERSION = 1
# Append-only finding revisions are managed by project_findings, never through
# ordinary file import/delete or BinaryView save operations.
FINDINGS_FOLDER = ".bn-findings"

# This inode must persist across closes: unlinking it permits two independent
# locks on the same project. Keep it outside BN's metadata and data directories.
_LEASE_FILENAME = ".bn-bridge.lock"
_LEASE_LOCK = threading.RLock()
# A live native handle must not lose its lease when a GUI bridge is replaced or
# its last Python owner disappears. Only a successful explicit close removes it.
_LEASE_OWNERS: dict[int, ProjectManager] = {}
# Do not invoke native handle destructors from a post-fork child callback.
_FORKED_OWNERS: list[ProjectManager] = []


def _shape_name(value: Any) -> str:
    return type(value).__name__


def _clip(text: str, limit: int = 80) -> str:
    """*text* for a refusal message: what was refused, not a megabyte of it."""
    return text if len(text) <= limit else text[:limit] + "..."


def _refuse_shape(path: str, expected: str, value: Any) -> None:
    raise RuntimeError(
        f"rootfs metadata key '{path}' must be {expected}, got {_shape_name(value)}; "
        f"{_ROOTFS_REINIT}"
    )


def _require_str(document: dict[str, Any], key: str, path: str, *, nullable: bool = False,
                 required: bool = True) -> None:
    if key not in document:
        if required:
            raise RuntimeError(f"rootfs metadata key '{path}' is missing; {_ROOTFS_REINIT}")
        return
    value = document[key]
    if isinstance(value, str) or (nullable and value is None):
        return
    _refuse_shape(path, "a string or null" if nullable else "a string", value)


def _require_str_list(document: dict[str, Any], key: str, path: str) -> None:
    if key not in document:
        return
    value = document[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        _refuse_shape(path, "an array of strings", value)


def _require_int_or_null(document: dict[str, Any], key: str, path: str) -> None:
    if key not in document:
        return
    value = document[key]
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return
    _refuse_shape(path, "an integer or null", value)


def validate_rootfs_metadata(raw: Any) -> dict[str, Any]:
    """*raw* (the decoded `bn.rootfs` blob) as the version-1 document the
    consumers in `project_rootfs` index into, or a user-facing refusal naming
    the offending key path and the way back.

    Every shape checked here is one a consumer relies on: `libs` is iterated
    with `.items()` and each entry read for `rel_realpath`/`names`/`paths`/
    `soname`/`sha256`/`e_machine`/`elf_class`; `lookup_paths` rows are read
    for `path`/`source`; `root`/`platform`/`synced_at` are reported. Before
    this gate `{"libs": ["bad"]}` reached `.items()` as an `internal error:
    AttributeError`, and `version: 999` was read as version 1 (#1071).
    Unknown extra keys are accepted and ignored (a later bn may add some),
    but a sync rewrites the document from what this bn understands and does
    not carry them over. Only `libs` must be present; the rest are checked
    when they are."""
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"rootfs metadata must be a JSON object, got {_shape_name(raw)}; {_ROOTFS_REINIT}")
    version = raw.get("version")
    if type(version) is not int or version != ROOTFS_METADATA_VERSION:
        # The integer current version only: `"1"`, `1.0` and `true` are not it.
        if version is None:
            stated = "has no version"
        elif type(version) is int:
            stated = f"is version {version}"
        else:
            stated = (f"is version {_clip(repr(version))} "
                      f"(a {_shape_name(version)}, not an integer)")
        raise RuntimeError(
            f"this project's rootfs metadata {stated}; this bn understands version "
            f"{ROOTFS_METADATA_VERSION}; {_ROOTFS_REINIT}")
    # Only `libs` is required: every other key is read with a tolerant
    # `.get(...)` by its consumers, and the issue's own repro document carries
    # just `version`/`platform`/`libs` (#1071). Present keys must have the shape
    # the consumer indexes.
    _require_str(raw, "root", "root", nullable=True, required=False)
    _require_str(raw, "platform", "platform", nullable=True, required=False)
    _require_str(raw, "synced_at", "synced_at", nullable=True, required=False)
    lookup_paths = raw.get("lookup_paths", [])
    if not isinstance(lookup_paths, list):
        _refuse_shape("lookup_paths", "an array", lookup_paths)
    for index, row in enumerate(lookup_paths):
        path = f"lookup_paths[{index}]"
        if not isinstance(row, dict):
            _refuse_shape(path, "an object", row)
        _require_str({"path": row.get("path")}, "path", f"{path}.path")
        _require_str({"source": row.get("source")}, "source", f"{path}.source")
    if "libs" not in raw:
        raise RuntimeError(f"rootfs metadata key 'libs' is missing; {_ROOTFS_REINIT}")
    libs = raw["libs"]
    if not isinstance(libs, dict):
        _refuse_shape("libs", "an object", libs)
    for file_id, entry in libs.items():
        path = f"libs[{_clip(repr(file_id))}]"
        if not isinstance(entry, dict):
            _refuse_shape(path, "an object", entry)
        _require_str(entry, "rel_realpath", f"{path}.rel_realpath")
        _require_str(entry, "soname", f"{path}.soname", nullable=True, required=False)
        _require_str(entry, "sha256", f"{path}.sha256", required=False)
        _require_str_list(entry, "names", f"{path}.names")
        _require_str_list(entry, "paths", f"{path}.paths")
        _require_int_or_null(entry, "e_machine", f"{path}.e_machine")
        _require_int_or_null(entry, "elf_class", f"{path}.elf_class")
    return raw


def _after_fork_child() -> None:
    # flock locks belong to open file descriptions shared across fork. Closing
    # the child's duplicate is safe; LOCK_UN would unlock the parent's lease.
    # Inherited managers reject use before touching inherited Python/BN locks.
    try:
        _FORKED_OWNERS.extend(_LEASE_OWNERS.values())
        for fd in _LEASE_OWNERS:
            os.close(fd)
        _LEASE_OWNERS.clear()
    finally:
        _LEASE_LOCK.release()


os.register_at_fork(
    before=_LEASE_LOCK.acquire,
    after_in_parent=_LEASE_LOCK.release,
    after_in_child=_after_fork_child,
)


def refuse_raw_project_path(path: str | os.PathLike[str]) -> None:
    """Reject standalone access to project storage, including symlink aliases."""
    requested = Path(path).expanduser().absolute()
    for candidate in (requested, requested.resolve()):
        for ancestor in candidate.parents:
            if (ancestor / "project.bnpm").is_file():
                raise RuntimeError(
                    f"Cannot load project storage as a standalone binary: {candidate}. "
                    f"Use `bn project open {ancestor}` and `bn project load <selector>`."
                )


class LinkRecordUnreadable(RuntimeError):
    """The project's `bn.links` metadata is not a record this bn can read."""

    def __init__(self, detail: str):
        super().__init__(
            f"the project's link record (`{LINKS_METADATA_KEY}` metadata) cannot be read: "
            f"{detail}; it is left untouched")


def validate_link_record(raw: Any) -> dict[str, Any]:
    """*raw* (the decoded `bn.links` blob) as the version-1 record its readers
    index into -- `{"version": 1, "targets": {<file id>: {"project_path": str,
    "providers": {<NEEDED name>: <file id>}}}}` -- or `LinkRecordUnreadable`
    naming the offending key.

    It is project-level data any tool or any other bn version can write, so
    it is untrusted input, like `bn.rootfs`. Before this gate a `providers`
    list reached `.items()` as an `internal error:`, the delete guard read a
    record it could not parse as "no dependents", and a `version: 999` record
    was rewritten as version 1 on the next link (#1078 review)."""
    if not isinstance(raw, dict):
        raise LinkRecordUnreadable(f"it must be a JSON object, got {type(raw).__name__}")
    version = raw.get("version")
    if isinstance(version, bool) or version != LINKS_METADATA_VERSION:
        raise LinkRecordUnreadable(
            f"it is version {version!r}; this bn understands version {LINKS_METADATA_VERSION}")
    targets = raw.get("targets")
    if not isinstance(targets, dict):
        raise LinkRecordUnreadable(
            f"key 'targets' must be an object, got {type(targets).__name__}")
    for target_id, entry in targets.items():
        where = f"targets[{target_id!r}]"
        if not isinstance(entry, dict):
            raise LinkRecordUnreadable(
                f"key '{where}' must be an object, got {type(entry).__name__}")
        if not isinstance(entry.get("project_path"), str):
            raise LinkRecordUnreadable(f"key '{where}.project_path' must be a string")
        providers = entry.get("providers")
        if not isinstance(providers, dict) or not all(
            isinstance(name, str) and isinstance(provider, str)
            for name, provider in providers.items()
        ):
            raise LinkRecordUnreadable(
                f"key '{where}.providers' must map library names to file ids")
    return raw


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
    """At most ONE open project per bridge.

    Holds the strong reference to the `Project` handle for the bridge's lifetime,
    resolves file selectors, builds file rows, and performs the first-save
    recipe. Thread-safe for the handful of state transitions it has; the heavy
    BN calls run under whichever dispatcher lock the op declared.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: ProjectState | None = None
        self._pid = os.getpid()
        self._lease_fd: int | None = None
        self._gate_fd: int | None = None
        # Parsed dynamic-section facts (exports, NEEDED) of project library
        # blobs, keyed by (file id, size, mtime) so a replaced blob re-parses.
        # Export tables are never stored in project metadata; `project_link`
        # rebuilds them from the blobs on demand.
        self.export_cache: dict[tuple[str, int, int], Any] = {}

    # -- state -------------------------------------------------------------

    def _check_process(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError(
                "An inherited project manager cannot be used after fork; "
                "create a new bridge in the child process"
            )

    def _acquire_gate(self, canonical: Path) -> None:
        # BN refuses creation into a directory containing the lease file, so
        # it cannot exist until create_project returns. All opens/creates briefly
        # lock the parent directory to cover that handoff without another file.
        # This needs no parent write access or shared lock-file ownership policy.
        with _LEASE_LOCK:
            fd = os.open(canonical.parent, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException as exc:
                os.close(fd)
                if isinstance(exc, BlockingIOError):
                    raise RuntimeError(
                        f"Project open/create is in progress under {canonical.parent}; "
                        "retry after the other bridge finishes."
                    ) from exc
                raise
            self._gate_fd = fd
            _LEASE_OWNERS[fd] = self

    def _release_gate(self) -> None:
        with _LEASE_LOCK:
            fd = self._gate_fd
            if fd is not None:
                os.close(fd)
                self._gate_fd = None
                del _LEASE_OWNERS[fd]

    def _acquire_lease(self, canonical: Path) -> None:
        with _LEASE_LOCK:
            lease = canonical / _LEASE_FILENAME
            try:
                # Honour the creator's umask and group-sharing policy. Existing
                # lease permissions are never widened implicitly.
                fd = os.open(lease, os.O_CREAT | os.O_RDWR, 0o660)
            except PermissionError as exc:
                raise RuntimeError(
                    f"Cannot access project lease {lease}: check its ownership and "
                    "read/write permissions (and directory access) for this user/group. "
                    "Ask the project owner to adjust permissions; do not delete the "
                    "lease file, because another bridge may still hold its lock."
                ) from exc
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException as exc:
                os.close(fd)
                if isinstance(exc, BlockingIOError):
                    raise RuntimeError(
                        f"Project already open in another bridge: {canonical}. "
                        "Its exclusive project lease is held (possibly under a "
                        "different BN_CACHE_DIR); close it in the owning bridge first."
                    ) from exc
                raise
            self._lease_fd = fd
            _LEASE_OWNERS[fd] = self

    def _release_lease(self) -> None:
        with _LEASE_LOCK:
            fd = self._lease_fd
            if fd is not None:
                # Never unlink the file or explicitly unlock a shared fork fd.
                os.close(fd)
                self._lease_fd = None
                del _LEASE_OWNERS[fd]

    @property
    def current(self) -> ProjectState | None:
        self._check_process()
        with self._lock:
            return self._state

    def require(self) -> ProjectState:
        state = self.current
        if state is None:
            raise RuntimeError(
                "No project is open in this bridge; run `bn project open <path.bnpr>` first"
            )
        return state

    def _adopt(self, project: Any, canonical: Path) -> ProjectState:
        # Publish before reading fallible native properties: cleanup failure
        # must retain the strong handle reference and whichever lease protects it.
        state = ProjectState(project, str(canonical), "", canonical.stem)
        self._state = state
        state.project_id = str(getattr(project, "id", "") or "")
        state.name = str(getattr(project, "name", "") or canonical.stem)
        return state

    def create(self, path: str, name: str) -> ProjectState:
        """Create under the handoff gate and retain the opened native handle."""
        self._check_process()
        canonical = canonical_project_path(path)
        with self._lock:
            if self._state is not None:
                raise RuntimeError(
                    f"A different project is already open in this bridge: {self._state.path}. "
                    "Run `bn project close` first (one project per bridge instance)."
                )
            self._acquire_gate(canonical)
            try:
                if canonical.exists():
                    raise RuntimeError(
                        f"project_create: {canonical} already exists; "
                        f"run `bn project open {canonical}`"
                    )
                import binaryninja

                try:
                    project = binaryninja.Project.create_project(str(canonical), name)
                except Exception as exc:  # noqa: BLE001 - ProjectException
                    raise RuntimeError(
                        f"Binary Ninja could not create project {canonical}: {exc}. "
                        "Projects require a Commercial or Ultimate licence."
                    ) from exc
                if project is None:
                    raise RuntimeError(f"Binary Ninja could not create project {canonical}")
                state = self._adopt(project, canonical)
                self._acquire_lease(canonical)
            except BaseException:
                if self._state is not None:
                    # If close fails, keep the parent gate as a conservative
                    # lifetime lease until a later explicit close succeeds.
                    self.close()
                else:
                    self._release_gate()
                raise
            self._release_gate()
            return state

    def open(self, path: str, *, self_pid: int) -> tuple[ProjectState, bool]:
        """Open *path*; ``(state, opened)`` where ``opened`` is False when the same
        project was already open here (BN dedupes a second open in-process,
        but the registry and the result should say which happened)."""
        self._check_process()
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
            # Registry ownership gives useful routing guidance, but only the
            # project-local lifetime lease arbitrates concurrent bridge opens.
            owner = find_other_owner(str(canonical), self_pid=self_pid)
            if owner is not None:
                raise RuntimeError(
                    f"project already open in bridge instance {owner['instance_id']} "
                    f"(pid {owner['pid']}): run `bn project attach {canonical}` "
                    f"(or pass -i {owner['instance_id']})"
                )
            self._acquire_gate(canonical)
            try:
                self._acquire_lease(canonical)
            finally:
                self._release_gate()
            try:
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
            except BaseException:
                self._release_lease()
                raise
            try:
                state = self._adopt(project, canonical)
            except BaseException:
                self.close()
                raise
            return state, True

    def close(self) -> ProjectState | None:
        """Close after callers close all project TARGETS.

        Failure retains the native handle, state, cache and exclusive lease so
        another bridge cannot open a database that may still be in use.
        """
        self._check_process()
        with self._lock:
            state = self._state
            if state is None:
                return None
            try:
                if state.project.close() is False:
                    raise RuntimeError("Binary Ninja returned False")
            except Exception as exc:  # noqa: BLE001 - retain ownership on failure
                raise RuntimeError(
                    f"Binary Ninja could not close project {state.path}: {exc}"
                ) from exc
            self._state = None
            self.export_cache.clear()
            self._release_lease()
            self._release_gate()
            return state

    # -- rootfs metadata ----------------------------------------------------

    def _rootfs_raw(self) -> Any:
        """The decoded `bn.rootfs` blob as stored, or `_NO_METADATA` when the
        key is absent. ONLY an absent key is "none" (#1071 review): a stored
        `null` or empty string is metadata somebody wrote, and reading it as
        "none" let the next sync start from an empty inventory and import
        every library a second time. Text that is not JSON -- or nests too
        deeply to decode -- is a refusal too."""
        project = self.require().project
        try:
            raw = project.query_metadata(ROOTFS_METADATA_KEY)
        except KeyError:  # Binary Ninja raises KeyError for a missing key
            return _NO_METADATA
        except Exception as exc:  # noqa: BLE001 - unreadable is not "absent"
            raise RuntimeError(
                f"rootfs metadata could not be read ({type(exc).__name__}: {_clip(str(exc))}); "
                f"{_ROOTFS_REINIT}") from exc
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "replace")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (ValueError, RecursionError) as exc:
                raise RuntimeError(
                    f"rootfs metadata is not valid JSON ({type(exc).__name__}: "
                    f"{_clip(str(exc))}); {_ROOTFS_REINIT}") from exc
        return raw

    def get_rootfs(self) -> dict[str, Any] | None:
        """The `bn.rootfs` project metadata, validated (`validate_rootfs_metadata`),
        or None when the project records no rootfs. The ONE gate: every reader
        -- `project_libs`, `project_sync_libs`, `link_view`, the load-time
        auto-link -- reads through here, so a future version or a malformed
        shape is a user-facing refusal everywhere and an `internal error:`
        nowhere (#1071)."""
        raw = self._rootfs_raw()
        return None if raw is _NO_METADATA else validate_rootfs_metadata(raw)

    def set_rootfs(self, value: dict[str, Any]) -> None:
        """Store *value* as the `bn.rootfs` project metadata (JSON text)."""
        self._set_json_metadata(ROOTFS_METADATA_KEY, value, "rootfs")

    def get_links(self) -> dict[str, Any]:
        """The `bn.links` record, validated (`validate_link_record`): which
        project file each linked target's NEEDED libraries are bound to
        (#1078). A fresh empty record when the project stores none. The ONE
        gate: a record this bn cannot read raises `LinkRecordUnreadable`, so
        no reader indexes into a shape it did not check and no writer replaces
        a record it could not read."""
        project = self.require().project
        try:
            raw = project.query_metadata(LINKS_METADATA_KEY)
        except Exception:  # noqa: BLE001 - BN raises for a missing key
            return {"version": LINKS_METADATA_VERSION, "targets": {}}
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "replace")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError as exc:
                raise LinkRecordUnreadable(f"it is not valid JSON ({exc})") from exc
        return validate_link_record(raw)

    def set_links(self, value: dict[str, Any]) -> None:
        self._set_json_metadata(LINKS_METADATA_KEY, value, "link")

    def _get_json_metadata(self, key: str) -> dict[str, Any] | None:
        """Raw decoded value of *key*, unvalidated: None for a missing key, bad
        JSON, or any read failure. Used only as the post-write readback in
        `_set_json_metadata` (for both `bn.rootfs` and `bn.links`) -- never as
        a reader's own gate, which `_rootfs_raw`/`get_links` are instead."""
        project = self.require().project
        try:
            raw = project.query_metadata(key)
        except Exception:  # noqa: BLE001 - BN raises for a missing key
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "replace")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return None
        return raw

    def _set_json_metadata(self, key: str, value: dict[str, Any], what: str) -> None:
        project = self.require().project
        try:
            if project.store_metadata(key, json.dumps(value, sort_keys=True)) is False:
                raise RuntimeError("Binary Ninja returned False")
            if self._get_json_metadata(key) != value:
                raise RuntimeError("stored metadata did not match the requested value")
        except Exception as exc:  # noqa: BLE001 - ProjectException
            raise RuntimeError(
                f"Binary Ninja could not store the {what} metadata in the project: {exc}"
            ) from exc

    def clear_rootfs(self) -> None:
        """Remove the `bn.rootfs` metadata: the rollback for a sync that found
        none to restore (storing `{}` instead would leave a blob the gate above
        refuses as versionless). Checked like `set_rootfs`: the native result,
        then a readback."""
        project = self.require().project
        try:
            if project.remove_metadata(ROOTFS_METADATA_KEY) is False:
                raise RuntimeError("Binary Ninja returned False")
            if self._rootfs_raw() is not _NO_METADATA:
                raise RuntimeError("the metadata is still present after the removal")
        except Exception as exc:  # noqa: BLE001 - ProjectException
            raise RuntimeError(
                f"Binary Ninja could not remove the rootfs metadata from the project: {exc}"
            ) from exc

    # -- file lookups ------------------------------------------------------

    def files(self) -> list[Any]:
        return list(self.require().project.files)

    @staticmethod
    def path_in_project(pf: Any) -> str:
        """`/folder/name`. Computed from the folder chain when BN's
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
        """File selector -> ProjectFile, or a RuntimeError that lists the
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
        raise RuntimeError(self._miss(wanted, files))

    def _miss(self, wanted: str, files: list[Any]) -> str:
        """The error for a selector naming no file: the files whose NAMES are
        nearest the selector's last component, at most `_SUGGESTION_CAP`, in a
        fixed order, and never a findings-ledger record (#1077)."""
        suggestions = self._nearest_by_name(wanted, files)
        if not suggestions:
            return (
                f"No project file matches {wanted!r}; no project file has a similar "
                "name. Full listing: `bn project files`."
            )
        return "\n".join([
            f"No project file matches {wanted!r}. Nearest by file name:",
            *self._listing(suggestions),
            "Full listing: `bn project files`.",
        ])

    def _nearest_by_name(self, wanted: str, files: list[Any]) -> list[Any]:
        """*files* ranked by how closely their name matches the last component
        of *wanted*, ties by path: the same name (or its `.bndb`-stripped
        stem), then one a prefix of the other, then a substring, then a
        `difflib` ratio of 0.6 or better. Ledger records are never offered."""
        import difflib

        base = wanted.strip().rstrip("/").rsplit("/", 1)[-1].lower()
        if base.endswith(".bndb"):
            base = base[: -len(".bndb")]
        if not base:
            return []
        ranked: list[tuple[int, str, Any]] = []
        for pf in files:
            path = self.path_in_project(pf)
            if is_finding_path(path):
                continue
            name = str(pf.name).lower()
            stem = name[: -len(".bndb")] if name.endswith(".bndb") else name
            if base in (name, stem):
                rank = 0
            elif stem.startswith(base) or base.startswith(stem):
                rank = 1
            elif base in stem or stem in base:
                rank = 2
            elif difflib.SequenceMatcher(None, base, stem).ratio() >= 0.6:
                rank = 3
            else:
                continue
            ranked.append((rank, path, pf))
        ranked.sort(key=lambda item: (item[0], item[1], str(item[2].id)))
        return [pf for _, _, pf in ranked[:_SUGGESTION_CAP]]

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

    def sibling_index(self, files: list[Any]) -> dict[str, list[Any]]:
        """path-in-project -> the files AT that path, over *files* (one pass).
        `sibling_database` answers for one file; a listing of N files must not
        ask BN for N folder listings (O(N^2) FFI under the read lock)."""
        index: dict[str, list[Any]] = {}
        for pf in files:
            index.setdefault(self.path_in_project(pf), []).append(pf)
        return index

    def sibling_database(
        self, pf: Any, index: dict[str, list[Any]] | None = None, *,
        allow_ambiguous: bool = False, operation: str | None = None,
    ) -> Any | None:
        """The unique `<name>.bndb` beside *pf*, or None when absent.

        Loads and saves refuse ambiguous siblings instead of treating them as
        absent. Listings may opt out so the conflicting files remain visible.
        *operation* names the caller in that refusal for a command that took no
        selector (`bn save`), whose user cannot be told their selector was
        ambiguous (#1073 item 2); without it the selector wording stands.
        """
        name = str(pf.name)
        if name.endswith(".bndb"):
            return None
        wanted_path = self.path_in_project(pf) + ".bndb"
        if index is not None:
            matches = index.get(wanted_path, [])
        else:
            project = self.require().project
            try:
                siblings = project.get_files_in_folder(getattr(pf, "folder", None))
            except Exception:  # noqa: BLE001 - fall back to the full list
                siblings = list(project.files)
            matches = [f for f in siblings if self.path_in_project(f) == wanted_path]
        if len(matches) > 1 and not allow_ambiguous:
            if operation is None:
                raise RuntimeError(self._ambiguous(wanted_path, matches))
            raise RuntimeError("\n".join([
                f"{operation} refused: {len(matches)} database files are named {wanted_path!r} "
                "in this project; load one of them explicitly or delete the duplicate:",
                *self._listing(matches),
            ]))
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
        self, pf: Any, open_targets: dict[str, str] | None = None,
        index: dict[str, list[Any]] | None = None,
    ) -> dict[str, Any]:
        """File row. *open_targets* maps project_file_id -> target_id;
        *index* is `sibling_index()` over the listing being built, so one row
        costs one path lookup rather than a folder listing."""
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
        sibling = self.sibling_database(pf, index, allow_ambiguous=True)
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

    # -- save --------------------------------------------------------------

    def create_database_file(self, bv: Any, pf: Any) -> Any:
        """Create the project-tracked `<name>.bndb` for the raw project file *pf*
        and write *bv*'s database into it. Returns the new ProjectFile.

        `create_file(b"")` makes a row but neither a blob nor its parent
        directory; `create_database` needs the directory. Failed writes verify
        cleanup and disclose any row/blob that could not be removed.
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
        parent = Path(dest).parent
        # Remember whether THIS save made the blob's directory: a rollback may
        # remove only that, and only while it is still empty (#1073 item 1).
        parent_created = not parent.exists()
        try:
            parent.mkdir(parents=True, exist_ok=True)
            created = bv.create_database(dest)
            if created is False or not Path(dest).is_file():
                raise RuntimeError("Binary Ninja reported no file was written")
        except BaseException as exc:
            primary_error = f"Failed to save project database to {dest}: {exc}"
            self._discard_file(project, new, dest, primary_error)
            if parent_created:
                try:
                    os.rmdir(parent)  # refuses a non-empty directory by itself
                except OSError:
                    pass
            if not isinstance(exc, Exception):
                raise
            raise RuntimeError(primary_error) from exc
        return new

    @staticmethod
    def _discard_file(project: Any, pf: Any, dest: str, primary_error: str) -> None:
        file_id = str(pf.id)
        cleanup_errors: list[str] = []
        try:
            if project.delete_file(pf) is False:
                cleanup_errors.append("Binary Ninja refused to delete the project file")
        except Exception as exc:  # noqa: BLE001 - verify both kinds of residue below
            cleanup_errors.append(str(exc))
        row_remains: bool | None = None
        blob_remains: bool | None = None
        try:
            row_remains = any(str(candidate.id) == file_id for candidate in project.files)
        except Exception as exc:  # noqa: BLE001 - an unverified rollback is not success
            cleanup_errors.append(f"could not verify project row removal: {exc}")
        try:
            os.lstat(dest)
            blob_remains = True
        except FileNotFoundError:
            blob_remains = False
        except OSError as exc:
            cleanup_errors.append(f"could not verify blob removal: {exc}")
        if row_remains is False and blob_remains is False:
            return
        from ._shared import OperationFailure

        raise OperationFailure(
            "rollback_failed",
            f"{primary_error}; cleanup failed for project file {file_id} at {dest}",
            requested={"project_file_id": file_id, "path": dest},
            observed={
                "rolled_back": False,
                "primary_error": primary_error,
                "project_file_id": file_id,
                "path": dest,
                "row_remains": row_remains,
                "blob_remains": blob_remains,
                "cleanup_errors": cleanup_errors,
            },
        )


# -- ownership -------------------------------------------------------------


def _record_is_live(payload: dict[str, Any], pid: int) -> bool:
    verdict = identity_verdict(payload, pid)
    if verdict == "proven":
        return True
    if verdict == "mismatch":
        return False
    # "unrecorded": no identity to compare; a pid that still answers is treated
    # as live (the conservative direction -- refusing to open costs one command,
    # a wrong open causes project file creation to fail later).
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
    files = list(state.project.files)
    index = manager.sibling_index(files)
    rows: list[dict[str, Any]] = []
    for pf in files:
        here = manager.folder_path(getattr(pf, "folder", None))
        if wanted and not (
            here == wanted or (recursive and here.startswith(wanted + "/"))
        ):
            continue
        if not wanted and not recursive and here:
            continue
        rows.append(manager.file_row(pf, open_targets, index))
    rows.sort(key=lambda r: r["path"])
    folders = sorted(
        (
            {"id": str(f.id), "path": manager.folder_path(f)}
            for f in state.project.folders
        ),
        key=lambda r: r["path"],
    )
    return {"files": rows, "folders": folders}


def is_finding_path(path: str) -> bool:
    """Whether *path* (a path in the project) lies in the findings ledger."""
    components = [part for part in path.split("/") if part]
    return bool(components) and components[0] == FINDINGS_FOLDER


def _refuse_finding_path(path: str) -> None:
    if is_finding_path(path):
        raise RuntimeError(
            "reserved findings history: use `bn project finding` to append a "
            "revision or review; existing finding records cannot be overwritten or deleted"
        )


def _refuse_nul(label: str, value: str) -> None:
    """Binary Ninja cuts a project name at the first NUL, so a check on the
    requested string would look past the path that is actually created."""
    if "\0" in value:
        raise RuntimeError(f"project_import: {label} must not contain a NUL byte")


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
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise RuntimeError("project_import: name must be a non-empty string when given")
    if isinstance(name, str):
        _refuse_nul("name", name)
    if isinstance(folder, str):
        _refuse_nul("folder", folder)
    # The name defaults to the REQUESTED basename: importing the applet link
    # `usr/bin/record -> play` stored a second `play` at `/usr/bin/play` when
    # it came from the resolved path (#1079). The bytes still come through
    # the resolved path.
    requested = Path(path).expanduser()
    source = requested.resolve()
    if not source.is_file():
        raise RuntimeError(f"project_import: not a file: {source}")
    final_name = (name or requested.name or source.name).strip()
    folder_path = folder if isinstance(folder, str) else ""
    _refuse_finding_path(folder_path + "/" + final_name)
    path_in_project = "/" + "/".join(
        [*[part for part in folder_path.split("/") if part], final_name])
    existing = [
        pf for pf in state.project.files if manager.path_in_project(pf) == path_in_project
    ]
    if existing:
        # Two files at one path-in-project make every later selector for it
        # ambiguous; the duplicate is refused rather than created.
        raise RuntimeError(
            f"project_import: {path_in_project} already exists in the project "
            f"(file {existing[0].id}); import it under another name with --name <other>, "
            f"or remove the existing file first with `bn project delete {existing[0].id}`"
        )
    target_folder = manager.ensure_folder(folder_path or None)
    try:
        pf = state.project.create_file_from_path(
            str(source),
            target_folder,
            final_name,
            str(description or ""),
        )
    except Exception as exc:  # noqa: BLE001 - ProjectException
        raise RuntimeError(f"Binary Ninja could not import {source}: {exc}") from exc
    bridge._write_registry()
    row = manager.file_row(pf)
    if requested.is_symlink():
        row["source_symlink"] = f"{requested} -> {source}"
    return row


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


def project_delete(bridge: Any, selector: Any, force: Any = False) -> dict[str, Any]:
    manager = bridge.projects
    state = manager.require()
    pf = manager.resolve_file(selector)
    _refuse_finding_path(manager.path_in_project(pf))
    open_targets = _open_project_targets(bridge)
    open_rows = [
        row for row in open_targets if row.get("project_file_id") == str(pf.id)
    ]
    if open_rows:
        sel = open_rows[0].get("selector") or open_rows[0].get("target_id")
        raise RuntimeError(
            f"project file {manager.path_in_project(pf)} is open as target {sel}; "
            "close it first (`bn close -t <selector>`)"
        )
    path = manager.path_in_project(pf)
    file_id = str(pf.id)
    # Only a real `True` forces: `bool("false")` is True, and this flag deletes
    # a file other targets depend on. The binder refuses a non-boolean on the
    # wire; a direct caller gets the same strictness.
    forced = force is True
    # #1078: a file some target's imports are bound to. Deleting it silently
    # left those links dangling and the next link (or restart replay) free to
    # rebind the name onto whatever alias matched. Refuse unless forced; the
    # targets are named either way. A record that cannot be READ cannot answer
    # the question, so it refuses too: unknown is not "nothing depends on it".
    record_error: str | None = None
    try:
        dependents = linked_dependents(manager, pf, open_targets)
    except LinkRecordUnreadable as exc:
        dependents = []
        record_error = str(exc)
    if record_error is not None and not forced:
        raise RuntimeError(
            f"cannot tell whether any target is linked to project file {path} ({file_id}): "
            f"{record_error}. Pass --force to delete anyway."
        )
    if dependents and not forced:
        raise RuntimeError(
            f"project file {path} ({file_id}) is the linked provider of "
            f"{len(dependents)} target(s): {_dependents_text(dependents)}. "
            "Deleting it leaves their imports without a provider; pass --force to "
            "delete anyway, then import the right file and `bn project link -t <selector>`."
        )
    try:
        ok = state.project.delete_file(pf)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Binary Ninja could not delete {path}: {exc}") from exc
    if ok is False:
        raise RuntimeError(f"Binary Ninja refused to delete {path}")
    # The deleted file's own link record (when it was a linked target) is
    # stale; records that POINT at it stay, since the dangling binding they
    # describe is still what those databases hold. An unreadable record is
    # never rewritten.
    if record_error is None:
        try:
            record = manager.get_links()
            if file_id in record["targets"]:
                del record["targets"][file_id]
                manager.set_links(record)
        except Exception:  # noqa: BLE001 - a stale record never fails a delete
            pass
    bridge._write_registry()
    result: dict[str, Any] = {"deleted": True, "project_file_id": file_id, "project_path": path}
    if dependents:
        result["dependents"] = dependents
    if record_error is not None:
        result["link_record_error"] = record_error
    return result


def linked_dependents(
    manager: Any, pf: Any, open_targets: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The targets whose recorded links depend on project file *pf*: one row
    per target with its project path, the NEEDED names concerned, and whether
    that target is open now or a saved database (#1078). Ordered by path.

    A library is ONE provider in two files: the raw file the inventory is
    keyed by, and the `<name>.bndb` a binding is backed by once it exists. A
    record naming either depends on both -- deleting the raw file behind a
    database-backed binding orphans it just the same -- so the sibling pair is
    matched, and such a row carries `provider_path`, the file the record
    names. A recorded target that is no longer a project file depends on
    nothing. Raises `LinkRecordUnreadable` for a record that cannot be read.
    """
    record = manager.get_links()
    files = manager.files()
    live = {str(owned.id) for owned in files}
    index = manager.sibling_index(files)
    path = manager.path_in_project(pf)
    file_id = str(pf.id)
    related = {file_id: path}
    for sibling in index.get(path + ".bndb", []):
        related[str(sibling.id)] = path + ".bndb"
    if path.endswith(".bndb"):
        for raw in index.get(path[: -len(".bndb")], []):
            related[str(raw.id)] = path[: -len(".bndb")]
    open_ids = {
        str(row.get("project_file_id")): row for row in (open_targets or [])
        if row.get("project_file_id")
    }
    out: list[dict[str, Any]] = []
    for target_id, entry in record["targets"].items():
        if str(target_id) not in live:
            continue
        matched = {
            name: provider for name, provider in entry["providers"].items()
            if provider in related
        }
        if not matched:
            continue
        open_row = open_ids.get(str(target_id))
        row = {
            "project_file_id": str(target_id),
            "project_path": entry["project_path"],
            "libraries": sorted(matched),
            "open": open_row is not None,
            "selector": (open_row.get("selector") or open_row.get("target_id")) if open_row else None,
        }
        through = sorted({related[provider] for provider in matched.values()} - {path})
        if through:
            row["provider_path"] = through[0]
        out.append(row)
    out.sort(key=lambda row: (row["project_path"], row["project_file_id"]))
    return out


def _dependents_text(dependents: list[dict[str, Any]]) -> str:
    parts = []
    for row in dependents:
        where = f"open as {row['selector']}" if row["open"] else "saved database"
        through = f" through {row['provider_path']}" if row.get("provider_path") else ""
        parts.append(f"{row['project_path'] or row['project_file_id']} ({where}; "
                     f"{', '.join(row['libraries'])}{through})")
    return "; ".join(parts)


def _other_database_writers(bridge: Any, bv: Any, blob: str) -> list[str]:
    rows = bridge.targets.refresh()
    current = bridge._loaded_target_row(rows, bv)
    current_id = current.get("target_id") if current is not None else None
    canonical = os.path.realpath(blob)
    return [
        str(row.get("selector") or row["target_id"]) + f" ({row['target_id']})"
        for row in rows
        if row.get("target_id") != current_id
        and any(
            value and os.path.realpath(value) == canonical
            for value in (row.get("database_path"), row.get("filename"))
        )
    ]


def save_project_database(bridge: Any, bv: Any, path: Any) -> dict[str, Any]:
    """Project branch of `save_database`. The caller established
    ``bv.project_file is not None`` and holds the write lock."""
    manager = bridge.projects
    if path is not None:
        raise RuntimeError(
            "project-backed target: bn save takes no output path; use "
            "`bn project export <selector> <dest>`"
        )
    pf = bv.project_file
    # After a first save BN keeps reporting the RAW file here, but the
    # record we published knows the database file; prefer that.
    record = bridge.targets.project_binding(bv)
    file_id = record[0] if record else str(pf.id)
    project_path = record[1] if record else manager.path_in_project(pf)
    _refuse_finding_path(project_path)
    has_database = bool(getattr(bv.file, "has_database", False))
    if has_database:
        blob = str(manager.resolve_file(file_id).path_on_disk)
        writers = _other_database_writers(bridge, bv, blob)
        if writers:
            raise RuntimeError(
                f"Refusing to save {project_path}: another open target writes the same "
                f"project database: {', '.join(writers)}. Close the duplicate targets first."
            )
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
    sibling = manager.sibling_database(pf, operation="save")
    if sibling is not None:
        writers = _other_database_writers(bridge, bv, str(sibling.path_on_disk))
        open_note = f" Open targets: {', '.join(writers)}." if writers else ""
        raise RuntimeError(
            f"{manager.path_in_project(pf)} already has a database file "
            f"{manager.path_in_project(sibling)} ({sibling.id}); load that file instead. "
            f"A second database for one binary is not created implicitly.{open_note}"
        )
    new = manager.create_database_file(bv, pf)
    new_id = str(new.id)
    new_path = manager.path_in_project(new)
    blob = str(new.path_on_disk)
    bridge.targets.clear_dirty(bv)
    bridge.targets.note_database(bv, blob)
    bridge.targets.note_project_database(bv, new_id, new_path)
    # #1078: the target's link record moves with its identity -- the links
    # now live in the database file, and `project delete` reports dependents
    # by that file. Best-effort: a metadata hiccup never fails a save.
    try:
        record = bridge.projects.get_links()
        entry = record.get("targets", {}).pop(str(pf.id), None)
        if entry is not None:
            entry["project_path"] = new_path
            record["targets"][new_id] = entry
            bridge.projects.set_links(record)
    except Exception:  # noqa: BLE001
        pass
    bridge._write_registry()
    return {
        "ok": True,
        "saved": True,
        "path": blob,
        "project_file_id": new_id,
        "project_path": new_path,
        "mode": "created",
    }
