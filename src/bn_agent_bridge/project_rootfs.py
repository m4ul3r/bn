"""Firmware-rootfs projects: create the project at a rootfs root, import the
shared libraries the target's dynamic loader could find, and link a loaded
binary's imported symbols to the project library that exports them.

What Binary Ninja gives us, measured on 6.0: the ELF loader already creates
one `ExternalLibrary` per `DT_NEEDED` (`bv.libraries`, `bv.get_external_library`)
with `backing_file=None`, and one `ExternalLocation` per external import with
`library=None` (ELF does not say which NEEDED provides a symbol).
`add_external_library` on a name that exists returns None, so libraries are
always looked up first. Setting `library.backing_file` and `location.library`
persists through a database save and reload, but `bv.file.modified` does not
flip, so the bridge marks the target dirty itself.

Pure filesystem reasoning (rootfs detection, loader lookup paths, candidate
scanning) lives in `rootfs_layout`; ELF facts (NEEDED, SONAME, RPATH,
verneed-attributed imports, exports) in `elf_dynamic`. Neither imports BN; this
module imports `binaryninja` only inside `project_create` (tests substitute a
fake module) and never imports `bridge`.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import project_ops
from .elf_dynamic import ElfDynamicInfo, parse_elf_dynamic, parse_elf_file
from .rootfs_layout import (
    DiscoveryBudget, detect_root, discovery_budget, lookup_paths, scan_libraries,
)

METADATA_VERSION = 1
# ld.so builds the global scope breadth-first over NEEDED; a firmware tree
# with a deeper chain than this is pathological, so the walk is bounded.
_MAX_SCOPE_DEPTH = 8


# -- project_create ----------------------------------------------------------


def project_create(bridge: Any, path: Any, name: Any = None, rootfs: Any = None) -> dict[str, Any]:
    """Create a new `.bnpr` at *path*, then own it exactly like `project_open`
    (ownership guard, registry). *rootfs* = `{"root", "platform"}` seeds the
    `bn.rootfs` metadata with an empty library inventory."""
    manager = bridge.projects
    if not isinstance(path, str) or not path.strip():
        raise RuntimeError("project_create: path must be the .bnpr directory to create")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise RuntimeError("project_create: name must be a non-empty string when given")
    canonical = project_ops.canonical_project_path(path)
    if canonical.exists():
        raise RuntimeError(
            f"project_create: {canonical} already exists; run `bn project open {canonical}`"
        )
    current = manager.current
    if current is not None:
        raise RuntimeError(
            f"A different project is already open in this bridge: {current.path}. "
            "Run `bn project close` first (one project per bridge instance)."
        )
    initial: dict[str, Any] | None = None
    if rootfs is not None:
        if not isinstance(rootfs, dict) or not isinstance(rootfs.get("root"), str):
            raise RuntimeError('project_create: rootfs must be {"root": ..., "platform": ...}')
        initial = _empty_metadata(
            os.path.realpath(rootfs["root"]), str(rootfs.get("platform") or "unknown"))
    project_name = (name or canonical.stem).strip()

    manager.create(str(canonical), project_name)
    if initial is not None:
        manager.set_rootfs(initial)
    bridge._write_registry()
    info = project_ops.project_info(bridge)
    info["opened"] = True
    info["created"] = True
    return info


def _empty_metadata(root: str, platform: str) -> dict[str, Any]:
    return {
        "version": METADATA_VERSION,
        "root": root,
        "platform": platform,
        "lookup_paths": [],
        "libs": {},
        "synced_at": None,
    }


# -- project_sync_libs -------------------------------------------------------


def project_sync_libs(
    bridge: Any, root: Any = None, platform: Any = None, dry_run: bool = False, *,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Discover unlocked, then revalidate and commit one atomic inventory update."""
    manager = bridge.projects
    with bridge._target_lock.read():
        state = manager.require()
        meta = manager.get_rootfs() or {}
        live_ids = {str(pf.id) for pf in state.project.files}
    budget = DiscoveryBudget(cancelled=cancelled)
    with discovery_budget(budget):
        if root is not None and (not isinstance(root, str) or not root.strip() or "\0" in root):
            raise RuntimeError("project_sync_libs: root must be a directory path when given")
        if platform is not None and (not isinstance(platform, str) or not platform.strip()):
            raise RuntimeError("project_sync_libs: platform must be a string when given")
        chosen_root = root or meta.get("root")
        if not chosen_root:
            raise RuntimeError(
                "project_sync_libs: this project records no rootfs; pass root (or run "
                "`bn project init <root>`)"
            )
        if not isinstance(chosen_root, str) or "\0" in chosen_root:
            raise RuntimeError("project_sync_libs: recorded root must be a directory path")
        chosen_root = os.path.realpath(os.path.expanduser(chosen_root))
        if not os.path.isdir(chosen_root):
            raise RuntimeError(f"project_sync_libs: rootfs is not a directory: {chosen_root}")
        chosen_platform = platform or meta.get("platform")
        if not chosen_platform or chosen_platform == "unknown":
            detected = detect_root(chosen_root, explicit=True)
            chosen_platform = detected.platform if detected is not None else "unknown"
        notes: list[str] = []
        paths = lookup_paths(chosen_root, str(chosen_platform))
        if not paths:
            notes.append(f"no library lookup paths for platform {chosen_platform!r} under {chosen_root}")
        candidates, scan_notes = scan_libraries(chosen_root, paths)
        notes.extend(scan_notes)
        existing = {
            str(entry.get("rel_realpath"))
            for file_id, entry in (meta.get("libs") or {}).items() if file_id in live_ids
        }
        parsed: dict[str, ElfDynamicInfo | None] = {}
        # Validate every candidate and parse new blobs before entering bulk mode.
        for cand in candidates:
            budget.check()
            project_ops._refuse_finding_path("/".join(
                project_ops.folder_components(cand.dir_in_root) + [cand.names[0]]))
            if cand.rel_realpath not in existing:
                parsed[cand.rel_realpath] = parse_elf_file(cand.realpath, check=budget.check)
        notes.extend(budget.notes)
        with bridge._write_operation_lock():
            budget.check()
            if manager.current is not state or (manager.get_rootfs() or {}) != meta or (
                {str(pf.id) for pf in state.project.files} != live_ids
            ):
                raise RuntimeError("project_sync_libs: project changed during discovery; retry the sync")
            return _apply_sync(
                manager, state, meta, chosen_root, str(chosen_platform),
                paths, candidates, parsed, notes, bool(dry_run), budget,
            )


def _apply_sync(
    manager: Any, state: Any, meta: dict[str, Any], chosen_root: str, chosen_platform: str,
    paths: list[Any], candidates: list[Any], parsed: dict[str, ElfDynamicInfo | None],
    notes: list[str], dry_run: bool, budget: DiscoveryBudget,
) -> dict[str, Any]:
    live_ids = {str(pf.id) for pf in state.project.files}
    libs = {file_id: dict(entry) for file_id, entry in (meta.get("libs") or {}).items()
            if file_id in live_ids}
    by_rel = {str(entry.get("rel_realpath")): (file_id, entry) for file_id, entry in libs.items()}
    imported_files: list[tuple[Any, dict[str, Any]]] = []
    planned: list[dict[str, Any]] = []
    changed: list[dict[str, str]] = []
    skipped_existing = 0
    skipped_not_elf = 0

    to_import: list[tuple[Any, ElfDynamicInfo]] = []
    for cand in candidates:
        if cand.rel_realpath in by_rel:
            skipped_existing += 1
            file_id, entry = by_rel[cand.rel_realpath]
            if entry.get("sha256") != cand.sha256:
                changed.append({"rel_realpath": cand.rel_realpath, "project_file_id": file_id})
            else:
                # Lookup aliases can change without changing the library bytes.
                entry["names"] = _library_names(cand.names, entry.get("soname"))
                entry["paths"] = list(cand.paths)
            continue
        info = parsed[cand.rel_realpath]
        if info is None:
            skipped_not_elf += 1
            notes.append(f"{cand.rel_realpath}: not a dynamic ELF object; skipped")
            continue
        notes.extend(f"{cand.rel_realpath}: {note}" for note in info.notes)
        to_import.append((cand, info))
    lookup_rows = [{"path": lp.path_in_root, "source": lp.source} for lp in paths]

    if dry_run:
        for cand, info in to_import:
            planned.append({
                "id": None,
                "name": cand.names[0],
                "path": cand.dir_in_root.rstrip("/") + "/" + cand.names[0],
                "rel_realpath": cand.rel_realpath,
                "names": _library_names(cand.names, info.soname),
                "paths": list(cand.paths),
                "sha256": cand.sha256,
                "size": cand.size,
            })
        if to_import:
            notes.append(f"dry run: {len(to_import)} libraries would be imported; nothing written")
    else:
        project = state.project
        metadata_attempted = False
        try:
            if to_import:
                import_error: BaseException | None = None
                # BN's bulk context ends bulk mode only after a normal yield.
                # Capture failures inside it, then propagate after it exits.
                with project.bulk_operation():
                    try:
                        for cand, info in to_import:
                            budget.check()
                            names = _library_names(cand.names, info.soname)
                            folder = manager.ensure_folder(cand.dir_in_root)
                            pf = project.create_file_from_path(
                                cand.realpath, folder, cand.names[0],
                                "rootfs library; names: " + ", ".join(names),
                            )
                            entry = {
                                "rel_realpath": cand.rel_realpath,
                                "soname": info.soname,
                                "names": names,
                                "paths": list(cand.paths),
                                "sha256": cand.sha256,
                                "e_machine": info.e_machine,
                                "elf_class": info.elf_class,
                            }
                            imported_files.append((pf, entry))
                            libs[str(pf.id)] = entry
                    except BaseException as exc:
                        import_error = exc
                if import_error is not None:
                    raise import_error
            budget.check()
            metadata_attempted = True
            manager.set_rootfs({
                "version": METADATA_VERSION,
                "root": chosen_root,
                "platform": str(chosen_platform),
                "lookup_paths": lookup_rows,
                "libs": libs,
                "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
        except BaseException as exc:
            rollback_errors: list[str] = []
            for pf, _ in reversed(imported_files):
                try:
                    if project.delete_file(pf) is False:
                        raise RuntimeError("delete_file returned false")
                except Exception as rollback_exc:  # noqa: BLE001
                    rollback_errors.append(f"file {pf.id}: {rollback_exc}")
            if metadata_attempted:
                try:
                    manager.set_rootfs(meta)
                except Exception as rollback_exc:  # noqa: BLE001
                    rollback_errors.append(f"metadata: {rollback_exc}")
            if rollback_errors:
                raise RuntimeError(
                    f"Library sync failed: {exc}; rollback incomplete: "
                    + "; ".join(rollback_errors)
                ) from exc
            raise
        if imported_files:
            index = manager.sibling_index(list(state.project.files))
            planned = [manager.file_row(pf, None, index) for pf, _ in imported_files]
    return {
        "root": chosen_root,
        "platform": str(chosen_platform),
        "lookup_paths": lookup_rows,
        "imported": planned,
        "skipped_existing": skipped_existing,
        "changed": changed,
        "skipped_not_elf": skipped_not_elf,
        "dry_run": bool(dry_run),
        "notes": notes,
    }


def _library_names(names: list[str], soname: str | None) -> list[str]:
    """Every spelling a `DT_NEEDED` may use for this file: the real basename
    first, then the symlink spellings, then the SONAME when it is new."""
    out = list(names)
    if soname and soname not in out:
        out.append(soname)
    return out


# -- project_libs ------------------------------------------------------------


def project_libs(bridge: Any) -> dict[str, Any]:
    manager = bridge.projects
    state = manager.require()
    meta = manager.get_rootfs()
    if meta is None:
        raise RuntimeError(
            "this project records no rootfs libraries; run `bn project init <root>` or "
            "`bn project libs --sync --root <root>`"
        )
    files = list(state.project.files)
    by_id = {str(pf.id): pf for pf in files}
    index = manager.sibling_index(files)
    rows: list[dict[str, Any]] = []
    for file_id, entry in (meta.get("libs") or {}).items():
        pf = by_id.get(str(file_id))
        if pf is None:
            continue  # A deleted provider is not part of the live library inventory.
        sibling = manager.sibling_database(pf, index, allow_ambiguous=True)
        rows.append({
            "project_file_id": str(file_id),
            "project_path": manager.path_in_project(pf) if pf is not None else None,
            "rel_realpath": entry.get("rel_realpath"),
            "soname": entry.get("soname"),
            "names": list(entry.get("names") or []),
            "e_machine": entry.get("e_machine"),
            "elf_class": entry.get("elf_class"),
            "sibling_database_id": str(sibling.id) if sibling is not None else None,
        })
    rows.sort(key=lambda r: (r["project_path"] is None, r["project_path"] or "", r["project_file_id"]))
    return {
        "root": meta.get("root"),
        "platform": meta.get("platform"),
        "synced_at": meta.get("synced_at"),
        "libs": rows,
    }


# -- project_link ------------------------------------------------------------


@dataclass(slots=True)
class _Library:
    """One project library from the `bn.rootfs` inventory."""
    file_id: str
    pf: Any
    rel_realpath: str
    names: list[str]
    paths: list[str]
    e_machine: int | None
    elf_class: int | None
    project_path: str
    sibling: Any  # the `<name>.bndb` project file, or None
    # Name by which the global-scope walk first reached this library.
    needed_as: str = ""



@dataclass(slots=True)
class _Plan:
    libraries: list[dict[str, Any]] = field(default_factory=list)
    scope: list[str] = field(default_factory=list)
    # (location, library name, bare symbol, set target_symbol?) per assignment.
    assignments: list[tuple[Any, str, str, bool]] = field(default_factory=list)
    # (library name, backing project file) per rebinding; planning never writes.
    rebinds: list[tuple[str, Any]] = field(default_factory=list)
    linked: int = 0
    already_linked: int = 0
    relinked: int = 0
    # Per-symbol outcome rows: {"symbol", "address", "library", "action"}.
    locations: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[dict[str, str]] = field(default_factory=list)
    ambiguous: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def project_link(
    bridge: Any, target: Any, dry_run: bool = False, relink: bool = False,
) -> dict[str, Any]:
    bv = bridge.targets.resolve(target)
    return link_view(bridge, bv, dry_run=bool(dry_run), relink=bool(relink))


def link_view(
    bridge: Any, bv: Any, *, dry_run: bool = False, relink: bool = False,
    from_load: bool = False,
) -> dict[str, Any]:
    """Bind *bv*'s NEEDED libraries to the project files that provide them and
    assign each unassigned external location the library exporting its symbol.

    Resolution prefers matching loader filenames of the target's ELF class and
    machine, ordered by RPATH then loader lookup paths (RUNPATH after the
    environment). SONAME-only inventory aliases are a disclosed fallback, not
    allowed to outrank an exact filename. The global scope is the breadth-first
    closure over NEEDED. A symbol binds to its known verneed provider, or to the
    first scope export unless parsing explicitly reports unread requirements.

    *from_load* keeps verified deterministic links clean. A failed automatic
    apply is dirty, just like an explicit link, because writes may have landed.
    """
    manager = bridge.projects
    state = manager.require()
    binding = bridge.targets.project_binding(bv)
    pf = getattr(bv, "project_file", None)
    if binding is not None:
        # A first save records the database identity while BN can keep exposing
        # the raw ProjectFile, even after that raw row/blob has been deleted.
        file_id, project_path = binding
        pf = next((
            owned for owned in state.project.files
            if str(owned.id) == file_id and manager.path_in_project(owned) == project_path
        ), None)
        if pf is None:
            raise RuntimeError("project_link: target belongs to a different project")
    elif pf is not None and not any(
        _same_file(pf, owned)
        and os.path.realpath(str(pf.path_on_disk)) == os.path.realpath(str(owned.path_on_disk))
        for owned in state.project.files
    ):
        raise RuntimeError("project_link: target belongs to a different project")
    meta = manager.get_rootfs() or {}
    inventory = _inventory(manager, state, meta)
    plan = _Plan()

    target_elf = _target_elf_info(bv)
    is_elf = _view_type_name(bv) == "ELF"
    needed = _view_libraries(bv)
    if is_elf and target_elf is not None:
        needed = list(target_elf.needed)

    resolver = _Resolver(manager, inventory, meta, target_elf, plan)
    direct: dict[str, _Library | None] = {}
    scope: list[_Library] = []
    if is_elf:
        direct, scope = resolver.global_scope(needed)
    else:
        for name in needed:
            direct[name] = resolver.resolve(name, by_basename=True)
            if direct[name] is not None:
                scope.append(direct[name])
    plan.scope = [lib.needed_as for lib in scope]

    # Step 3: bind each NEEDED library's backing project file.
    lib_objects: dict[str, Any] = {}
    for name in needed:
        entry = direct.get(name)
        lib = _external_library(bv, name, create=False)
        lib_objects[name] = lib
        plan.libraries.append(_library_row(name, entry))
        if entry is None:
            continue
        backing = entry.sibling if entry.sibling is not None else entry.pf
        if not _same_file(getattr(lib, "backing_file", None), backing):
            plan.rebinds.append((name, backing))

    # Step 4: assign a library to every external location lacking one.
    if is_elf:
        _plan_locations(bv, scope, direct, lib_objects, target_elf, resolver, plan,
                        relink=relink)

    writes = len(plan.assignments) + len(plan.rebinds)
    status = "noop" if writes == 0 else "verified"
    attempted = False
    verified = False
    if not dry_run and writes:
        try:
            for name, backing in plan.rebinds:
                lib = lib_objects.get(name)
                if lib is None:
                    attempted = True
                    if not from_load:
                        bridge.targets.mark_dirty(bv)
                    lib = _external_library(bv, name, create=True)
                    if lib is None:
                        raise RuntimeError(f"Binary Ninja could not create external library {name}")
                    lib_objects[name] = lib
                attempted = True
                if not from_load:
                    bridge.targets.mark_dirty(bv)
                lib.backing_file = backing
            for loc, name, bare, set_target in plan.assignments:
                attempted = True
                if not from_load:
                    bridge.targets.mark_dirty(bv)
                lib = lib_objects.get(name) or _external_library(bv, name, create=False)
                loc.library = lib
                if set_target:
                    loc.target_symbol = bare
            failures = _readback(bv, plan)
            if failures:
                status = "verification_failed"
                plan.notes.extend(failures)
            else:
                verified = True
        except Exception as exc:  # noqa: BLE001 - preserve partial-write evidence
            status = "internal_error"
            plan.notes.append(f"import linking apply/readback failed: {exc}")
        finally:
            if from_load and attempted and not verified:
                bridge.targets.mark_dirty(bv)
    success = status not in {"verification_failed", "internal_error"}
    if not success:
        plan.notes.append(
            "link counters describe the planned changes, not verified completed writes")
    return {
        "status": status,
        "success": success,
        "preview": bool(dry_run),
        "committed": (not dry_run) and success,
        "dry_run": bool(dry_run),
        "libraries": plan.libraries,
        "scope": plan.scope,
        "linked": plan.linked,
        "already_linked": plan.already_linked,
        "relinked": plan.relinked,
        "rebound": len(plan.rebinds),
        "locations": plan.locations,
        "unresolved": plan.unresolved,
        "ambiguous_libraries": plan.ambiguous,
        "notes": plan.notes,
        "measured": success,
    }


def _inventory(manager: Any, state: Any, meta: dict[str, Any]) -> list[_Library]:
    libs = meta.get("libs") or {}
    if not libs:
        return []
    files = list(state.project.files)
    by_id = {str(pf.id): pf for pf in files}
    index = manager.sibling_index(files)
    out: list[_Library] = []
    for file_id, entry in libs.items():
        pf = by_id.get(str(file_id))
        if pf is None:
            continue  # deleted from the project since the sync; not linkable
        out.append(_Library(
            file_id=str(file_id),
            pf=pf,
            rel_realpath=str(entry.get("rel_realpath") or ""),
            names=[str(n) for n in (entry.get("names") or [])],
            paths=[str(p) for p in (entry.get("paths") or [entry.get("rel_realpath") or ""])],
            e_machine=entry.get("e_machine"),
            elf_class=entry.get("elf_class"),
            project_path=manager.path_in_project(pf),
            sibling=manager.sibling_database(pf, index),
        ))
    return out


class _Resolver:
    """NEEDED name -> project library, with the loader's preference order."""

    def __init__(self, manager: Any, inventory: list[_Library], meta: dict[str, Any],
                 target_elf: ElfDynamicInfo | None, plan: _Plan) -> None:
        self._manager = manager
        self._inventory = inventory
        self._plan = plan
        self._class = target_elf.elf_class if target_elf is not None else None
        self._machine = target_elf.e_machine if target_elf is not None else None
        self._meta = meta
        self._target_elf = target_elf
        self._cache: dict[tuple[Any, ...], _Library | None] = {}

    def resolve(
        self, name: str, *, by_basename: bool = False,
        requester: ElfDynamicInfo | None = None,
        inherited_rpath: tuple[str, ...] = (),
    ) -> _Library | None:
        rank = _directory_rank(self._meta, requester, inherited_rpath)
        key = (name, by_basename, tuple(rank.items()))
        if key in self._cache:
            return self._cache[key]
        wanted = os.path.basename(name) if by_basename else name
        exact_path = not by_basename and "/" in name
        if exact_path and not name.startswith("/"):
            self._plan.notes.append(
                f"{name}: relative DT_NEEDED path requires the firmware process working directory")
            self._cache[key] = None
            return None
        normalized = os.path.normpath(name) if exact_path else ""
        matches = [
            lib for lib in self._inventory
            if self._compatible(lib) and (
                normalized in lib.paths or normalized == lib.rel_realpath
                if exact_path else wanted in lib.names
            )
        ]
        filename_matches: list[_Library] = []
        if not exact_path:
            # SONAME is an identity alias, not a filename ld.so searches for.
            # Never let an unrelated spelling outrank an available exact file.
            filename_matches = [
                lib for lib in matches
                if any(os.path.basename(path) == wanted for path in lib.paths)
            ]
            if filename_matches:
                matches = filename_matches
            elif matches:
                self._plan.notes.append(
                    f"{name}: only SONAME/identity aliases match; no loader filename match")
        chosen: _Library | None = None
        # list.sort temporarily empties its list while evaluating keys. Capture
        # this before sorting: matches may be the filename_matches list itself.
        rank_filename_paths = not exact_path and bool(filename_matches)
        if matches:
            matches.sort(key=lambda lib: (
                min((rank.get(os.path.dirname(path) or "/", len(rank))
                     for path in lib.paths
                     if not rank_filename_paths
                     or os.path.basename(path) == wanted), default=len(rank)),
                lib.project_path))
            chosen = matches[0]
            if len(matches) > 1:
                self._plan.ambiguous.append({
                    "name": name,
                    "chosen": chosen.project_path,
                    "candidates": [lib.project_path for lib in matches],
                })
        self._cache[key] = chosen
        return chosen

    def _compatible(self, lib: _Library) -> bool:
        # A target whose header could not be read matches any library; so does
        # an inventory row without class/machine.
        if self._class and lib.elf_class and lib.elf_class != self._class:
            return False
        if self._machine and lib.e_machine and lib.e_machine != self._machine:
            return False
        return True

    def info(self, lib: _Library) -> ElfDynamicInfo | None:
        """Parsed dynamic facts of the library blob, from the manager's cache."""
        blob = str(getattr(lib.pf, "path_on_disk", "") or "")
        try:
            st = os.stat(blob)
        except OSError:
            return None
        key = (lib.file_id, int(st.st_size), int(st.st_mtime_ns))
        cache = self._manager.export_cache
        if key in cache:
            return cache[key]
        info = parse_elf_file(blob)
        for stale in [k for k in cache if k[0] == lib.file_id]:
            del cache[stale]
        cache[key] = info
        return info

    def global_scope(self, needed: list[str]) -> tuple[dict[str, _Library | None], list[_Library]]:
        """Breadth-first over NEEDED, then each resolved library's own NEEDED
        (depth-bounded, each name and each library once), in ld.so order."""
        direct: dict[str, _Library | None] = {}
        scope: list[_Library] = []
        seen_names: set[str] = set()
        seen_ids: set[str] = set()
        queue: deque[tuple[str, int, ElfDynamicInfo | None, tuple[str, ...]]] = deque(
            (name, 0, self._target_elf, ()) for name in needed)
        while queue:
            name, depth, requester, inherited_rpath = queue.popleft()
            if name in seen_names:
                if depth == 0 and name not in direct:
                    direct[name] = self.resolve(
                        name, requester=requester, inherited_rpath=inherited_rpath)
                continue
            seen_names.add(name)
            lib = self.resolve(name, requester=requester, inherited_rpath=inherited_rpath)
            if depth == 0:
                direct[name] = lib
            if lib is None or lib.file_id in seen_ids:
                continue
            seen_ids.add(lib.file_id)
            lib.needed_as = name
            scope.append(lib)
            if depth >= _MAX_SCOPE_DEPTH:
                self._plan.notes.append(
                    f"global scope cut at depth {_MAX_SCOPE_DEPTH} below {name}")
                continue
            info = self.info(lib)
            if info is None:
                self._plan.notes.append(
                    f"{lib.project_path}: library blob could not be parsed; its NEEDED "
                    "and exports are unknown")
                continue
            ancestor_rpath = tuple(_split_search_path(
                requester.rpath if requester is not None and requester.runpath is None else None
            )) + inherited_rpath
            for dep in info.needed:
                # RPATH travels down the loader ancestry; RUNPATH does not.
                queue.append((dep, depth + 1, info, ancestor_rpath))
        return direct, scope


def _directory_rank(
    meta: dict[str, Any], target_elf: ElfDynamicInfo | None,
    inherited_rpath: tuple[str, ...] = (),
) -> dict[str, int]:
    """Requester and ancestor RPATH, environment, requester RUNPATH, defaults.

    A requester carrying RUNPATH ignores the RPATH chain for its own lookup.
    """
    ordered: list[str] = []
    rpath = _split_search_path(
        target_elf.rpath if target_elf is not None and target_elf.runpath is None else None)
    runpath = _split_search_path(target_elf.runpath if target_elf is not None else None)
    env_entries: list[str] = []
    rest: list[str] = []
    for row in meta.get("lookup_paths") or []:
        path = str(row.get("path") or "")
        source = str(row.get("source") or "")
        (env_entries if source.startswith("ld_library_path:") else rest).append(path)
    ordered.extend(rpath)
    if target_elf is None or target_elf.runpath is None:
        ordered.extend(inherited_rpath)
    ordered.extend(env_entries)
    ordered.extend(runpath)
    ordered.extend(rest)
    rank: dict[str, int] = {}
    for path in ordered:
        norm = os.path.normpath("/" + path.strip("/")) if path else ""
        if norm and norm not in rank:
            rank[norm] = len(rank)
    return rank


def _split_search_path(value: str | None) -> list[str]:
    """DT_RPATH / DT_RUNPATH entries: absolute only; `$ORIGIN`-style tokens are
    not expanded, so such an entry is treated as absent."""
    if not value:
        return []
    return [
        entry for entry in str(value).split(":")
        if entry.startswith("/") and "$" not in entry
    ]


def _plan_locations(
    bv: Any, scope: list[_Library], direct: dict[str, _Library | None],
    lib_objects: dict[str, Any], target_elf: ElfDynamicInfo | None,
    resolver: _Resolver, plan: _Plan, *, relink: bool,
) -> None:
    import_libraries = target_elf.import_libraries if target_elf is not None else {}
    exports: list[tuple[_Library, dict[str, Any]]] = []
    for lib in scope:
        info = resolver.info(lib)
        exports.append((lib, info.exports if info is not None else {}))
    # Transitive libraries that are not a direct NEEDED get an ExternalLibrary
    # of their own when a symbol binds to them.
    transitive: dict[str, Any] = {}

    for loc in _external_locations(bv):
        current = getattr(loc, "library", None)
        if current is not None and not relink:
            plan.already_linked += 1
            plan.locations.append(_location_row(loc, _library_name(current), "already_linked"))
            continue
        target_symbol = getattr(loc, "target_symbol", None)
        source = getattr(loc, "source_symbol", None)
        raw = str(target_symbol or getattr(source, "name", "") or "")
        bare = raw.split("@", 1)[0]
        if not bare:
            plan.unresolved.append({"symbol": raw, "reason": "location has no symbol name"})
            continue
        chosen: _Library | None = None
        verneed = import_libraries.get(bare)
        if verneed is not None and direct.get(verneed) is not None:
            chosen = direct[verneed]
        elif verneed is None and (target_elf is None or target_elf.versions_unread is None):
            for lib, table in exports:
                if bare in table:
                    chosen = lib
                    break
        if chosen is None:
            reason = (
                f"library {verneed} not in project"
                if verneed is not None and direct.get(verneed) is None
                else (
                    "symbol version requirements were not read: "
                    + target_elf.versions_unread
                    if target_elf is not None and target_elf.versions_unread is not None
                    else "not exported by any linked library"
                )
            )
            plan.unresolved.append({"symbol": bare, "reason": reason})
            continue
        if chosen.needed_as not in lib_objects and chosen.needed_as not in transitive:
            lib_obj = _external_library(bv, chosen.needed_as, create=False)
            transitive[chosen.needed_as] = lib_obj
            plan.libraries.append(_library_row(chosen.needed_as, chosen))
            backing = chosen.sibling if chosen.sibling is not None else chosen.pf
            if not _same_file(getattr(lib_obj, "backing_file", None), backing):
                plan.rebinds.append((chosen.needed_as, backing))
        if current is not None and _library_name(current) == chosen.needed_as:
            plan.already_linked += 1
            plan.locations.append(_location_row(loc, chosen.needed_as, "already_linked"))
            continue
        plan.assignments.append((loc, chosen.needed_as, bare, not target_symbol))
        action = "linked" if current is None else "relinked"
        if current is None:
            plan.linked += 1
        else:
            plan.relinked += 1
        plan.locations.append(_location_row(loc, chosen.needed_as, action))


def _location_row(loc: Any, library: str | None, action: str) -> dict[str, Any]:
    source = getattr(loc, "source_symbol", None)
    raw = str(getattr(loc, "target_symbol", None) or getattr(source, "name", "") or "")
    address = getattr(source, "address", None)
    return {
        "symbol": raw.split("@", 1)[0],
        "address": int(address) if isinstance(address, int) else None,
        "library": library,
        "action": action,
    }


def _readback(bv: Any, plan: _Plan) -> list[str]:
    """Verify backing IDs, location libraries, and newly filled symbol names."""
    failures: list[str] = []
    for name, backing in plan.rebinds:
        lib = _external_library(bv, name, create=False)
        got = getattr(lib, "backing_file", None)
        if lib is None or not _same_file(got, backing):
            failures.append(
                f"{name}: expected backing file {backing.id}, read back {getattr(got, 'id', None)}")
    observed = {_location_key(loc): loc for loc in _external_locations(bv)} if plan.assignments else {}
    for loc, wanted, bare, set_target in plan.assignments:
        actual = observed.get(_location_key(loc))
        got = _library_name(getattr(actual, "library", None))
        if actual is None or got != wanted:
            failures.append(f"{bare}: expected library {wanted}, read back {got}")
        if set_target and (actual is None or getattr(actual, "target_symbol", None) != bare):
            failures.append(f"{bare}: target symbol did not read back as {bare}")
    return failures


def _location_key(loc: Any) -> Any:
    source = getattr(loc, "source_symbol", None)
    address = getattr(source, "address", None)
    name = getattr(source, "name", None)
    if address is None and name is None:
        return id(loc)
    return (address, str(name) if name is not None else None)


def _external_locations(bv: Any) -> list[Any]:
    getter = getattr(bv, "get_external_locations", None)
    if not callable(getter):
        return []
    return list(getter() or [])


def _external_library(bv: Any, name: str, *, create: bool) -> Any:
    """BN's library object for *name*: the one the loader created, else a new
    one when *create* (never `add_external_library` for an existing name)."""
    getter = getattr(bv, "get_external_library", None)
    lib = getter(name) if callable(getter) else None
    if lib is None and create:
        adder = getattr(bv, "add_external_library", None)
        if callable(adder):
            lib = adder(name)
    return lib


def _library_row(name: str, entry: _Library | None) -> dict[str, Any]:
    if entry is None:
        return {"name": name, "resolved": False, "project_file_id": None,
                "project_path": None, "backing": None}
    return {
        "name": name,
        "resolved": True,
        "project_file_id": entry.file_id,
        "project_path": entry.project_path,
        "backing": "database" if entry.sibling is not None else "raw",
    }


def _library_name(lib: Any) -> str | None:
    if lib is None:
        return None
    return str(getattr(lib, "name", lib))


def _same_file(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    return str(getattr(a, "id", a)) == str(getattr(b, "id", b))


def _view_type_name(bv: Any) -> str:
    try:
        value = getattr(bv, "view_type", None)
        return str(getattr(value, "name", value) or "")
    except Exception:  # noqa: BLE001
        return ""


def _view_libraries(bv: Any) -> list[str]:
    """`bv.libraries` in BN's order (one per DT_NEEDED), deduplicated."""
    try:
        raw = list(getattr(bv, "libraries", None) or [])
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    for item in raw:
        name = str(getattr(item, "name", item))
        if name and name not in out:
            out.append(name)
    return out


def _target_elf_info(bv: Any) -> ElfDynamicInfo | None:
    """The target's own dynamic-section facts: from the original file when it
    is still on disk (the project blob, or the file a database was built from),
    else from the raw parent view's bytes."""
    file = getattr(bv, "file", None)
    for attr in ("original_filename", "filename"):
        path = str(getattr(file, attr, "") or "")
        if path and Path(path).is_file():
            info = parse_elf_file(path)
            if info is not None:
                return info
    raw = getattr(bv, "parent_view", None)
    read = getattr(raw, "read", None)
    if not callable(read):
        return None
    try:
        base = int(getattr(raw, "start", 0) or 0)
        length = getattr(raw, "length", None)
        if length is None:
            return None
        return parse_elf_dynamic(lambda offset, size: read(base + offset, size), int(length))
    except Exception:  # noqa: BLE001 - an unreadable header matches any library
        return None
