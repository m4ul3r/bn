"""Administrative commands: doctor, installs, sessions, and sticky pins.

Shared infrastructure (transport, paths, session state) is accessed through
the ``bn.cli`` module at call time -- ``cli.send_request(...)`` rather than a
``from ..cli import send_request`` -- so tests and scripts can monkeypatch
``bn.cli`` as the single well-known location.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NoReturn

from .. import cli
from ..cli import arg, command
from ..formatters import (
    _render_capabilities_text,
    _render_doctor_text,
    _render_instance_find_text,
    _render_instance_gc_text,
    _render_instance_use_text,
    _render_pin_clear_text,
    _render_session_list_text,
    _render_session_start_text,
    _render_session_status_text,
    _render_session_stop_text,
    _render_skill_install_text,
    _render_target_list_text,
    _render_target_use_text,
)
from ..transport import BridgeError, _resolve_timeout, unwrap_result


@command("capabilities",
         help="Index of every command, what it's for, and when to pick it over a neighbor "
              "(JSON for agents; no target needed)")
def _capabilities(args: argparse.Namespace) -> int:
    # Derived entirely from the @command registry -- no hand-maintained second
    # list -- so the index can't drift from the actual command surface (#276).
    items = [
        {
            "command": " ".join(spec["path"]),
            "group": spec["path"][0],
            "help": spec.get("help", ""),
            "requires_target": bool(spec.get("target", False)),
            "default_format": spec.get("fmt", "text"),
            "prefer_when": spec.get("prefer_when", ""),
            "see_also": list(spec.get("see_also", ())),
        }
        for spec in sorted(cli._COMMANDS, key=lambda spec: spec["path"])
    ]
    result = {"kind": "capabilities", "items": items, "count": len(items)}
    cli._emit_result(args, result, text_renderer=_render_capabilities_text, stem="capabilities")
    return 0


def _render_help_index_text(value: Any) -> str:
    items = value.get("items", []) if isinstance(value, dict) else []
    if isinstance(value, dict) and value.get("family"):
        lines = [
            f"{item.get('command')}: {item.get('help', '')}"
            for item in items
            if isinstance(item, dict)
        ]
        return "\n".join(lines) or "no commands in that family"
    groups = ", ".join(value.get("groups", [])) if isinstance(value, dict) else ""
    return (
        f"command families: {groups}\n"
        "use `bn <family> --help` for concise grammar\n"
        "machine-readable catalog: `bn capabilities --format json`"
    )


@command(
    "help",
    help="Show command families or one family's commands",
    args=[arg("family", nargs="?", help="Optional command family")],
)
def _help_index(args: argparse.Namespace) -> int:
    family = getattr(args, "family", None)
    specs = sorted(cli._COMMANDS, key=lambda spec: spec["path"])
    if family:
        specs = [spec for spec in specs if spec["path"][0] == family]
        if not specs:
            raise BridgeError(
                f"unknown command family {family!r}; run `bn help` to list families"
            )
    items = [
        {
            "command": " ".join(spec["path"]),
            "help": spec.get("help", ""),
        }
        for spec in specs
    ]
    result = {
        "kind": "help",
        "family": family,
        "groups": sorted({spec["path"][0] for spec in cli._COMMANDS}),
        "items": items,
        "count": len(items),
        "capabilities_command": "bn capabilities --format json",
    }
    cli._emit_result(
        args,
        result,
        text_renderer=_render_help_index_text,
        stem="help",
    )
    return 0


@command("doctor", help="Validate bridge discovery and installation")
def _doctor(args: argparse.Namespace) -> int:
    install_dir = cli.plugin_install_dir()
    source_dir = cli.plugin_source_dir()
    install_bridge = install_dir / "bridge.py"
    source_bridge = source_dir / "bridge.py"
    install_build_id = cli.build_id_for_file(install_bridge)
    source_build_id = cli.build_id_for_file(source_bridge)
    # Whole-package fingerprint (all *.py + model DB), so an edited engine module
    # (e.g. taint_engine.py) is flagged even though bridge.py is unchanged (#161).
    install_engine_id = cli.build_id_for_package(install_dir)
    source_engine_id = cli.build_id_for_package(source_dir)
    requested = getattr(args, "instance", None)
    candidates = cli.list_instances()
    if requested:
        # Honor --instance: scope the report to the one bridge instead of dumping
        # every running instance (a wall of text on a busy host).
        candidates = [
            inst for inst in candidates
            if inst.instance_id == requested or cli.instance_selector(inst) == requested
        ]
        if not candidates:
            raise BridgeError(
                f"No bridge instance found with id: {requested}. "
                "See `bn session list` for running instances."
            )
    instances = []
    for instance in candidates:
        ping: dict[str, Any]
        try:
            response = cli._send_request_to_instance(
                instance,
                "doctor",
                params={},
                target=None,
            )
            ping = unwrap_result(response, "doctor")
        except Exception as exc:
            ping = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        loaded_version = ping.get("plugin_version") if isinstance(ping, dict) else None
        loaded_build_id = ping.get("plugin_build_id") if isinstance(ping, dict) else None
        loaded_engine_id = ping.get("engine_build_id") if isinstance(ping, dict) else None
        # Engine (Binary Ninja) identity of the process behind this instance --
        # the only place the tool ever names the BN version it is driving.
        bn_version = ping.get("bn_version") if isinstance(ping, dict) else None
        bn_build_id = ping.get("bn_build_id") if isinstance(ping, dict) else None
        # Health signal in the JSON itself, matching what the text renderer
        # prints (status=ok/error). Without these a scripted JSON health check
        # could not tell a reachable instance from an unreachable one -- it had
        # to re-derive reachability from the absence of `doctor.error`. (L16)
        reachable = isinstance(ping, dict) and not ping.get("error")
        instances.append(
            {
                "instance_id": getattr(instance, "instance_id", None),
                "pid": instance.pid,
                "reachable": reachable,
                "status": "ok" if reachable else "error",
                "socket_path": str(instance.socket_path),
                "plugin_version": instance.plugin_version,
                "plugin_build_id": loaded_build_id,
                "bn_version": bn_version,
                "bn_build_id": bn_build_id,
                "installed_plugin_build_id": install_build_id,
                "source_plugin_build_id": source_build_id,
                "stale_plugin_version": (
                    bool(loaded_version)
                    and str(loaded_version) != cli.VERSION
                ),
                "stale_plugin_code": (
                    bool(loaded_build_id)
                    and install_build_id is not None
                    and loaded_build_id != install_build_id
                ),
                "engine_build_id": loaded_engine_id,
                "installed_engine_build_id": install_engine_id,
                # The loaded engine package diverges from on-disk: this live
                # session is running stale logic (e.g. an edited taint_engine.py
                # after a git pull). `bn session restart <id>` reloads it (#161).
                "stale_engine": (
                    bool(loaded_engine_id)
                    and install_engine_id is not None
                    and loaded_engine_id != install_engine_id
                ),
                "started_at": instance.started_at,
                "doctor": ping,
            }
        )

    result = {
        "cli_version": cli.VERSION,
        "plugin_source_dir": str(source_dir),
        "plugin_install_dir": str(install_dir),
        "plugin_source_build_id": source_build_id,
        "plugin_install_build_id": install_build_id,
        "engine_source_build_id": source_engine_id,
        "engine_install_build_id": install_engine_id,
        "instances": instances,
    }
    cli._emit_result(args, result, text_renderer=_render_doctor_text, stem="doctor")
    # #620(c): a doctor report that never fails exit code hides unreachable
    # instances from scripted/CI callers who only check the exit status.
    # Deliberately reachable-only (not staleness): stale_plugin_version /
    # stale_plugin_code / stale_engine stay informational (see
    # _render_doctor_text) and never gate the exit code -- a reachable bridge
    # that happens to be running slightly stale code is not a CI failure, and
    # gating on it would false-positive the common case of a live session that
    # simply hasn't been restarted since the last `git pull` (#161). Zero
    # instances registered is not a probe failure either: any([]) is False, so
    # doctor still exits 0 when nothing is running (#620 C.3).
    unhealthy = any(not inst["reachable"] for inst in instances)
    return 1 if unhealthy else 0


@command("plugin", "install", help="Install the GUI plugin", fmt="json",
         args=[
             arg("--dest", type=Path, help="Custom install destination"),
             arg("--mode", choices=("symlink", "copy"), default="symlink"),
             arg("--force", action="store_true"),
         ])
def _plugin_install(args: argparse.Namespace) -> int:
    source = cli.plugin_source_dir()
    dest = args.dest or cli.plugin_install_dir()
    _install_tree(source, dest, mode=args.mode, force=args.force)

    cli._emit_result(
        args,
        {
            "installed": True,
            "mode": args.mode,
            "source": str(source),
            "destination": str(dest),
        },
        stem="plugin-install",
    )
    return 0


# Bytecode an install never writes, but that importing the installed plugin or
# skill in place leaves behind. `copytree` refuses to copy it, and `--force`
# tolerates it in a destination it is replacing -- but it is never evidence of
# ownership: a directory holding only bytecode is refused, because nothing in
# it proves an install of this artifact wrote it.
_INSTALL_BYTECODE_PATTERNS = ("__pycache__", "*.pyc", "*.pyo")
_IGNORED_INSTALL_ENTRIES = shutil.ignore_patterns(*_INSTALL_BYTECODE_PATTERNS)


def _is_install_bytecode(name: str) -> bool:
    """Whether `name` is a bytecode cache an install never copies or writes."""
    return any(fnmatch.fnmatch(name, pattern) for pattern in _INSTALL_BYTECODE_PATTERNS)


def _foreign_install_entry(dest: Path, source: Path) -> str | None:
    """The first entry under `dest` a copy-mode install of `source` would not
    have produced, as a slash-joined path relative to `dest`, or None when
    every entry mirrors the source tree -- directories included, so a user file
    added inside a directory the source also has is not mistaken for ours.
    """
    for entry in sorted(dest.iterdir()):
        if _is_install_bytecode(entry.name):
            continue
        counterpart = source / entry.name
        if entry.is_symlink() or not counterpart.exists():
            return entry.name
        if entry.is_dir() != counterpart.is_dir():
            return entry.name
        if entry.is_dir():
            nested = _foreign_install_entry(entry, counterpart)
            if nested is not None:
                return f"{entry.name}/{nested}"
    return None


def _any_file_matches_source(dest: Path, source: Path) -> bool:
    """Whether one file under `dest` is still byte-identical to its source
    counterpart. That is the content proof this tree is an install of `source`
    rather than a directory that merely reuses its entry names -- a user's own
    `plugin.json` sitting where ours sits is not ours (#766).
    """
    for entry in sorted(dest.iterdir()):
        if _is_install_bytecode(entry.name):
            continue
        counterpart = source / entry.name
        if entry.is_symlink():
            continue
        if entry.is_dir():
            if counterpart.is_dir() and _any_file_matches_source(entry, counterpart):
                return True
        elif counterpart.is_file() and entry.read_bytes() == counterpart.read_bytes():
            return True
    return False


def _unsafe_replace_reason(dest: Path, source: Path) -> str | None:
    """Why `--force` must not remove directory `dest`, or None when it may.

    `--dest` is caller-supplied, so `--force` used to `shutil.rmtree` whatever
    was there -- unrelated user data included (#766). A directory is only
    replaceable when it is provably this artifact's own: empty, or a previous
    install of `source` -- the source tree's own layout, entry for entry, with
    at least one file still byte-identical to the source's, and the bytecode
    caches importing it in place leaves behind tolerated but never counted as
    proof. Entry names alone decide nothing: a directory that merely reuses
    them is somebody else's. Everything else is refused with the deliberate
    alternative named, the same "refuse and name the escape hatch" shape as
    the bridge's destructive-op gate, rather than learned from a half-populated
    install. Paths are compared as `_install_path` resolved them, so a symlinked
    ancestor cannot smuggle the rmtree past the check, and the pure path rules
    (`_never_an_install_destination`) were already applied to this same path.
    """
    if not any(dest.iterdir()):
        return None
    foreign = _foreign_install_entry(dest, source)
    if foreign is not None:
        return (
            "it is not a previous install of this artifact "
            f"({foreign} is not what this install writes)"
        )
    if not _any_file_matches_source(dest, source):
        return (
            "it is not a previous install of this artifact (no file under it "
            "still matches the source tree)"
        )
    return None


def _install_path(dest: Path) -> Path:
    """The path an install at `dest` works on, decided without touching disk.

    Ancestors are resolved -- so `store/missing/..` is `store`, not a directory
    that only exists once something has created `missing` -- while a symlink AT
    `dest` stays the link it is: it is unlinked, never followed, so its own
    location is what gets replaced and what the path rules must judge. Planning
    and acting on this one path is what keeps a spelling from naming two
    different directories either side of a `mkdir`.
    """
    return dest.parent.resolve() / dest.name if dest.is_symlink() else dest.resolve()


def _never_an_install_destination(dest: Path, source: Path) -> str | None:
    """Why `dest` can never be an install destination, or None when it can.

    Pure path rules, judged on the path the install would work on (`dest` as
    `_install_path` returned it, so a symlink is its own location and not its
    target): they hold whether or not `dest` exists yet, because creating an
    install there is as wrong as replacing one. A filesystem root, the home
    directory, or a destination containing the bn cache/skill roots -- the
    instance registry, sticky pins and every installed skill live under them --
    is not a destination, and neither is the source tree being copied, in
    either direction: inside it, a copy-mode install walks the copy it is
    writing, and a destination containing it would bury the artifact.
    """
    if dest == dest.parent:
        return "it is a filesystem root"
    if dest == Path.home().resolve():
        return "it is your home directory"
    for root in (
        cli.cache_home(),
        cli.claude_skills_dir(),
        cli.codex_skills_dir(),
        cli.omp_skills_dir(),
    ):
        if root.resolve().is_relative_to(dest):
            return f"it contains the bn installation root {root}"
    source_resolved = source.resolve()
    if dest == source_resolved or dest.is_relative_to(source_resolved):
        return f"it is the install source {source} itself or inside it"
    if source_resolved.is_relative_to(dest):
        return f"it contains the install source {source}"
    return None


def _unusable_parent_reason(dest: Path) -> str | None:
    """Why `dest`'s parent chain cannot be written to, or None when it can.

    `mkdir(parents=True)` fails on a component that is not a directory, or when
    the deepest component that already exists is not writable. Deciding that in
    the plan keeps the failure out of the install loop, where it would arrive
    after the destinations reached first had already been written.
    `os.access` is a hint, not a guarantee -- the install still maps any
    remaining OS error (a race, a filesystem that fills) to a BridgeError, with
    the destinations already written left in place.
    """
    for parent in reversed(dest.parents):
        if (parent.exists() or parent.is_symlink()) and not parent.is_dir():
            return f"{parent} is not a directory"
    existing = dest.parent
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if not os.access(existing, os.W_OK | os.X_OK):
        return f"{existing} is not writable"
    return None


def _install_tree(source: Path, dest: Path, *, mode: str, force: bool) -> None:
    if not source.exists():
        raise BridgeError(f"Source directory is missing: {source}")

    dest = _plan_install(source, dest, force=force)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BridgeError(
            f"Cannot create the parent directory of {dest}: {exc}"
        ) from exc

    if dest.is_symlink():
        dest.unlink()
    elif dest.is_dir():
        shutil.rmtree(dest)

    if mode == "copy":
        shutil.copytree(source, dest, ignore=_IGNORED_INSTALL_ENTRIES)
    else:
        os.symlink(source, dest, target_is_directory=True)


def _refuse_replacement(dest: Path, reason: str) -> NoReturn:
    raise BridgeError(
        f"Refusing to replace {dest} with --force: {reason}. "
        "--force replaces this install's own destination (an empty directory, "
        "or a previous install of the same artifact), never an unrelated path. "
        "Move it aside by hand and re-run if that is what you want."
    )


def _plan_install(source: Path, dest: Path, *, force: bool) -> Path:
    """The path this install will write to, or a refusal raised.

    Both `plugin install` and the `skill install` pre-pass call this, on the
    same untouched disk, so the pre-pass cannot approve a destination the
    install then refuses: that disagreement used to leave a refused
    multi-destination `skill install` with the destinations it reached first
    already replaced (#766).

    The pure path rules come first and hold for a destination that does not
    exist yet (`_never_an_install_destination`). Without `--force` any existing
    destination is an error. With it, only a symlink (removed as a link, never
    recursed into), an empty directory, or a previous install of `source` may
    be replaced; a regular file under `--dest` is somebody else's and is refused
    instead of unlinked.
    """
    dest = _install_path(dest)
    forbidden = _never_an_install_destination(dest, source)
    if forbidden is not None:
        raise BridgeError(
            f"Refusing to install into {dest}: {forbidden}. Choose a "
            "destination outside the bn installation and outside the tree "
            "being copied."
        )
    unusable = _unusable_parent_reason(dest)
    if unusable is not None:
        raise BridgeError(f"Cannot install into {dest}: {unusable}.")
    if not force:
        if dest.exists() or dest.is_symlink():
            raise BridgeError(f"Destination already exists: {dest}")
        return dest
    if dest.is_symlink() or not dest.exists():
        return dest
    if not dest.is_dir():
        _refuse_replacement(dest, "it is not a directory")
    refusal = _unsafe_replace_reason(dest, source)
    if refusal is not None:
        _refuse_replacement(dest, refusal)
    return dest


@command("skill", "install", help="Install the bundled agent skills", fmt="text",
         args=[
             arg("--dest", type=Path, help="Custom install destination"),
             arg("--mode", choices=("symlink", "copy"), default="symlink"),
             arg("--force", action="store_true"),
         ])
def _skill_install(args: argparse.Namespace) -> int:
    skills_root = cli.skills_source_dir()
    explicit_dest = args.dest is not None
    target_roots = [args.dest] if explicit_dest else _default_skill_install_roots()
    install_plan = []
    results = []
    for source in sorted(skills_root.iterdir()):
        if not source.is_dir() or not (source / "SKILL.md").exists():
            continue
        destinations = []
        for target_root in target_roots:
            dest = target_root / source.name
            install_plan.append((source, dest))
            destinations.append(str(dest))
        results.append(
            {
                "skill": source.name,
                "source": str(source),
                "destination": destinations[0],
                "destinations": destinations,
            }
        )

    pending_installs = []
    skipped_destinations = []
    for source, dest in install_plan:
        if not explicit_dest and not args.force and (dest.exists() or dest.is_symlink()):
            skipped_destinations.append(str(dest))
            continue
        # Planned, not just checked: the path validated here is the path the
        # install writes to, and every destination is refused before the first
        # one is written -- so no refusal can leave half a store installed. An
        # OS error while writing (a filesystem that fills, a race) still can,
        # with the destinations reached first left in place.
        # Two target roots can name one directory (a symlinked skills root, or
        # an equal CLAUDE_HOME and CODEX_HOME), and installing there twice would
        # write the same skills over the copy the first entry just made.
        planned = _plan_install(source, dest, force=args.force)
        if any(queued == planned for _, queued in pending_installs):
            continue
        pending_installs.append((source, planned))

    for source, dest in pending_installs:
        _install_tree(source, dest, mode=args.mode, force=args.force)

    result = {
        # #620(b): report installed=false when every destination was skipped,
        # rather than always claiming success. skipped_destinations already
        # explains why.
        "installed": bool(pending_installs),
        "mode": args.mode,
        "installed_destinations": [str(dest) for _, dest in pending_installs],
        "skipped_destinations": skipped_destinations,
        "skills": results,
    }
    cli._emit_result(args, result, text_renderer=_render_skill_install_text, stem="skill-install")
    return 0


def _default_skill_install_roots() -> list[Path]:
    roots = [cli.claude_skills_dir()]
    if cli.codex_home().is_dir():
        roots.append(cli.codex_skills_dir())
    if cli.omp_config_root().exists() or cli.omp_agent_dir().exists():
        roots.append(cli.omp_skills_dir())
    return roots


@command("session", "start", help="Start a new headless bridge session",
         args=[
             arg("binaries", nargs="*", help="Binary file paths to preload"),
             arg("--instance-id", help="Use a specific instance ID (default: random)"),
             arg("--no-bndb", action="store_true",
                 help="Don't auto-prefer a sibling .bndb file"),
             arg("--quick", "--no-analysis", action="store_true",
                 help="Preload without full analysis (fast); run `bn refresh` for full analysis"),
             arg("--detach", action="store_true", default=False,
                 help="Queue BNDB loading in the bridge and return immediately; "
                      "poll with `bn -i ID session status`"),
         ])
def _session_start(args: argparse.Namespace) -> int:
    if getattr(args, "_explicit_instance", False):
        selected = getattr(args, "instance", None)
        suffix = f" {selected}" if selected else " NAME"
        raise cli.BridgeError(
            "global -i/--instance selects an existing bridge and does not name "
            "a new `session start` instance. Use "
            f"`bn session start <binary> --instance-id{suffix}`."
        )
    binaries = getattr(args, "binaries", None) or []
    detached = bool(getattr(args, "detach", False))
    if detached and not binaries:
        raise BridgeError("session start --detach requires at least one binary path")
    instance_id = getattr(args, "instance_id", None)
    _resolve_timeout(None)
    instance = cli.spawn_instance(instance_id)
    project_roots, association_error = _associate_project_roots(
        instance.instance_id, [os.getcwd()]
    )

    prefer_bndb = not args.no_bndb
    quick = bool(getattr(args, "quick", False))
    # Each successful load also records this root, including detached completion.
    workdir = os.getcwd()
    loaded = []
    for binary in binaries:
        resolved = str(Path(binary).expanduser().resolve())
        try:
            resp = cli.send_request(
                "load_binary_async" if detached else "load_binary",
                params={
                    "path": resolved,
                    "prefer_bndb": prefer_bndb,
                    "quick": quick,
                    "workdir": workdir,
                },
                instance_id=instance.instance_id,
            )
            item = unwrap_result(
                resp, "load_binary_async" if detached else "load_binary")
            valid = (
                isinstance(item, dict)
                and (
                    (
                        detached
                        and isinstance(item.get("job_id"), str)
                        and item.get("state") in {"queued", "running"}
                    )
                    or (
                        not detached
                        and item.get("loaded") is True
                        and isinstance(item.get("targets"), list)
                        and bool(item["targets"])
                    )
                )
            )
            if not valid:
                loaded.append(
                    {
                        "path": resolved,
                        "error": (
                            "detached load was not queued"
                            if detached
                            else "load returned success without an open target"
                        ),
                    }
                )
            else:
                loaded.append(item)
        except BridgeError as exc:
            loaded.append({"path": resolved, "error": str(exc)})

    failures = [item for item in loaded if isinstance(item, dict) and item.get("error")]
    successes = [item for item in loaded if isinstance(item, dict) and not item.get("error")]

    result: dict[str, Any] = {
        "instance_id": instance.instance_id,
        "pid": instance.pid,
        "socket_path": str(instance.socket_path),
        "detached": detached,
    }
    if loaded:
        result["loaded"] = loaded
    if project_roots:
        result["project_roots"] = project_roots
    if association_error is not None:
        result["project_association_error"] = association_error

    # If the caller asked to preload binaries but none loaded, the freshly spawned
    # bridge is an empty zombie the caller would have to hunt down and stop. Shut
    # it down here and report the outcome under `stopped` / `cleanup_errors`.
    if binaries and not successes:
        cleanup_errors: list[str] = []
        # One verified process pin for the whole TERM -> wait -> KILL escalation
        # (#694): the pid is pinned and its identity proven once, and both signals
        # go through that same pin, so cleanup can never terminate an unrelated
        # process that recycled this pid.
        with cli.BridgeProcessSignal(instance) as signaller:
            try:
                cli.send_request("shutdown", instance_id=instance.instance_id)
            except BridgeError as exc:
                cleanup_errors.append(f"shutdown request failed: {exc}")
                reason = signaller.send(signal.SIGTERM)
                if reason is not None:
                    cleanup_errors.append(f"SIGTERM not sent: {reason}")
            stopped = cli.wait_for_teardown(instance, timeout=5.0)
            if not stopped:
                reason = signaller.send(signal.SIGKILL)
                if reason is not None:
                    cleanup_errors.append(f"SIGKILL not sent: {reason}")
                stopped = cli.wait_for_teardown(instance, timeout=2.0)
        result["stopped"] = stopped
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
        if not stopped:
            result["cleanup_error"] = (
                f"bridge instance {instance.instance_id} (pid {instance.pid}) "
                "did not fully tear down"
            )

    cli._emit_result(args, result, text_renderer=_render_session_start_text, stem="session-start")
    return 1 if failures or association_error else 0


@command(
    "session",
    "status",
    help="Poll detached binary load jobs",
    args=[arg("job_id", nargs="?", help="Optional detached load job ID")],
)
def _session_status(args: argparse.Namespace) -> int:
    instance_id = getattr(args, "instance", None)
    if not instance_id:
        raise BridgeError(
            "session status requires -i/--instance <id> from session start"
        )
    response = cli.send_request(
        "load_status",
        params={"job_id": getattr(args, "job_id", None)},
        instance_id=instance_id,
    )
    result = unwrap_result(response, "load_status")
    if not isinstance(result, dict):
        raise BridgeError("malformed load_status response")
    cli._emit_result(
        args,
        result,
        text_renderer=_render_session_status_text,
        stem="session-status",
    )
    return 0


@command(
    "session",
    "stop",
    help="Stop a running bridge session",
    args=[
        arg(
            "instance_id",
            nargs="?",
            metavar="instance",
            help="Instance ID to stop (positional, or pass -i/--instance <id>)",
        ),
        arg(
            "--instance-id",
            dest="instance_id_flag",
            default=None,
            help="Instance ID to stop (alias for the positional or -i/--instance)",
        ),
    ],
)
def _session_stop(args: argparse.Namespace) -> int:
    # #456: accept the id positionally OR via -i/--instance (every other command is
    # driven with -i/--instance, so `session stop -i <id>` is the natural
    # cleanup shape). The positional wins when both are given.
    # Presence, not truthiness (#690 r4): `bn session stop "$ID"` with $ID
    # unset must error, not fall through to the sticky-pin-filled -i and shut
    # down the PINNED bridge.
    positional = getattr(args, "instance_id", None)
    alias = getattr(args, "instance_id_flag", None)
    # #588: the final fallback must NOT pick up a sticky-pinned -i -- only a
    # non-sticky args.instance (an explicit -i/--instance OR the BN_INSTANCE
    # env default; see _instance_option/_apply_sticky_defaults in cli.py) may
    # supply the target for a bare `session stop`. Otherwise target_id stays
    # unset and the "requires an instance id" branch below fires instead of
    # silently stopping whatever instance the project happens to be pinned to.
    target_id = (
        positional
        if positional is not None
        else alias
        if alias is not None
        else getattr(args, "instance", None)
        if not getattr(args, "_sticky_instance", False)
        else None
    )
    if target_id is not None and not str(target_id).strip():
        raise BridgeError(
            "session stop: instance id is empty; pass an id from `bn session list`"
        )
    if not target_id:
        raise BridgeError(
            "session stop requires an instance id: `bn session stop <id>` "
            "(or `bn session stop -i/--instance <id>`). See `bn session list`."
        )
    # Resolve the instance up front so we can confirm teardown by pid + files
    # after the shutdown, and SIGTERM it if the socket request fails.
    # Lifecycle lookup, not normal discovery: a bridge whose socket file is gone
    # is hidden from `session list` because nothing can be dispatched to it, but
    # that unreachable process is exactly the one this command must be able to
    # stop (#694).
    inst = cli.find_lifecycle_instance(target_id)
    # One verified pin for the whole escalation, opened BEFORE the graceful
    # shutdown so the process this command may later signal is pinned throughout
    # (#694): the pid cannot be recycled while pinned, and both SIGTERM and
    # SIGKILL address that same proven process.
    signaller = cli.BridgeProcessSignal(inst) if inst is not None else None
    try:
        try:
            cli.send_request("shutdown", instance_id=target_id)
            method = "shutdown"
        except BridgeError:
            # Fallback: signal the pinned process directly.
            if inst is None or signaller is None:
                raise BridgeError(f"No bridge instance found with id: {target_id}")
            reason = signaller.send(signal.SIGTERM)
            if reason is not None:
                print(f"error: {reason}", file=sys.stderr)
                return 1
            method = "sigterm"

        result: dict[str, Any] = {"instance_id": target_id, "stopped": True}
        if method == "sigterm":
            result["method"] = method

        # Block until the socket/registry are gone and the process has exited, so a
        # follow-on `bn session start --instance-id <same>` can't race the dying
        # instance and fail as a duplicate (#92 Problem B). Escalate to SIGKILL if
        # graceful teardown stalls, and report failure if it never converges.
        if inst is not None and signaller is not None:
            if not cli.wait_for_teardown(inst, timeout=5.0):
                # Escalate through the SAME verified pin: SIGKILL against an
                # unproven or recycled pid is the most destructive form of the bug
                # (#694), so an unproven pin sends nothing and reports instead.
                reason = signaller.send(signal.SIGKILL)
                converged = (
                    cli.wait_for_teardown(inst, timeout=2.0)
                    if reason is None
                    else False
                )
                if not converged:
                    result["stopped"] = False
                    detail = f" {reason}." if reason is not None else ""
                    result["error"] = (
                        f"bridge instance {target_id} (pid {inst.pid}) did not fully "
                        f"tear down; registry/socket may be stale.{detail}"
                    )
                    cli._emit_result(args, result, text_renderer=_render_session_stop_text, stem="session-stop")
                    return 1
                result["method"] = "sigkill"

        cli._emit_result(args, result, text_renderer=_render_session_stop_text, stem="session-stop")
        return 0
    finally:
        if signaller is not None:
            signaller.close()


@command("session", "restart", help="Stop a bridge session and respawn it (same id), reloading its targets",
         args=[arg("instance", help="Instance ID to restart")])
def _session_restart(args: argparse.Namespace) -> int:
    """Cleanly reload a live session running stale code (#161): capture its open
    targets, stop the process, respawn under the same instance id, and reload the
    same binaries -- so the new bridge serves current code without the caller
    hunting for the right paths."""
    target_id = args.instance
    # Lifecycle lookup (#694): a socket-less bridge is hidden from normal
    # discovery, but restarting it is a legitimate way to recover it.
    inst = cli.find_lifecycle_instance(target_id)
    if inst is None:
        raise BridgeError(
            f"No bridge instance found with id: {target_id}. See `bn session list`."
        )
    resolved_id = inst.instance_id

    # Capture loaded targets (+ their analysis state) BEFORE tearing down.
    reload_targets: list[dict[str, Any]] = []
    reload_capture_failed = False
    reload_capture_error: str | None = None
    try:
        resp = cli._send_request_to_instance(inst, "list_targets", params={}, target=None)
        captured: list[dict[str, Any]] = []
        for t in (unwrap_result(resp, "list_targets") or []):
            path = t.get("filename")
            if path:
                captured.append({"path": path, "quick": t.get("analysis_state") == "quick"})
        # Commit only after the whole capture succeeds -- a mid-iteration
        # exception (a malformed row) must never leave a partially populated
        # reload_targets that gets reloaded and reported as a complete `loaded`.
        reload_targets = captured
    except Exception as exc:
        # #620(a): don't silently drop the target-enumeration failure -- the
        # restart still proceeds (with no targets to reload), but the caller
        # needs to know why nothing came back. Recorded in the structured
        # result (not just stderr) so `loaded == []` at rc 0 means only "list
        # succeeded with zero rows", matching _associate_project_roots's
        # project_association_error pattern below.
        reload_capture_failed = True
        reload_capture_error = f"{type(exc).__name__}: {exc}"
        print(
            f"warning: could not list open targets before restarting {target_id}: "
            f"{reload_capture_error}",
            file=sys.stderr,
        )

    # Preserve private project associations across the same-id restart. Reloads
    # below deliberately carry no caller cwd, so restarting elsewhere cannot
    # associate an unrelated project.
    inherited_roots: list[str] = []
    if isinstance(inst.meta, dict):
        roots = inst.meta.get("project_roots")
        if isinstance(roots, list):
            inherited_roots = [root for root in roots if isinstance(root, str)]

    # Stop the old process: graceful shutdown, then SIGTERM/SIGKILL escalation,
    # blocking until the socket/registry are gone so the respawn can reuse the id.
    # Both signals are gated on durable process identity: a restart must never
    # terminate an unrelated process that recycled the recorded pid (#694).
    with cli.BridgeProcessSignal(inst) as signaller:
        try:
            cli.send_request("shutdown", instance_id=resolved_id)
        except BridgeError:
            _require_signal_delivered(signaller.send(signal.SIGTERM), target_id)
        if not cli.wait_for_teardown(inst, timeout=5.0):
            _require_signal_delivered(signaller.send(signal.SIGKILL), target_id)
            if not cli.wait_for_teardown(inst, timeout=2.0):
                raise BridgeError(
                    f"bridge instance {target_id} (pid {inst.pid}) did not tear down; "
                    "cannot restart cleanly. Stop it manually and re-spawn."
                )

    instance = cli.spawn_instance(resolved_id)
    # Reload targets without associating the restart command's cwd.
    #
    # #753: `prefer_bndb` is False, not True. The captured `path` is the row's
    # `filename`, which is BN's own `bv.file.filename` -- the file each view
    # already IS, sidecar or raw. Asking for the sidecar preference again
    # therefore cannot help (a target already backed by a `.bndb` names that
    # `.bndb`) and can only substitute a DIFFERENT file for a target that was
    # deliberately opened raw, e.g. via `--no-bndb`. Worse, when that sidecar
    # was itself open as a second target, both reloads resolved to one file and
    # the instance came back with fewer targets than it had. `session restart`
    # is the remedy `bn doctor` prints for a stale bridge; it must return the
    # same targets, backed by the same files.
    reloaded: list[Any] = []
    for t in reload_targets:
        try:
            r = cli.send_request(
                "load_binary",
                params={"path": t["path"], "prefer_bndb": False, "quick": t["quick"]},
                instance_id=instance.instance_id,
            )
            reloaded.append(unwrap_result(r, "load_binary"))
        except BridgeError as exc:
            reloaded.append({"path": t["path"], "error": str(exc)})

    project_roots, association_error = _associate_project_roots(
        instance.instance_id, inherited_roots
    )

    failures = [x for x in reloaded if isinstance(x, dict) and x.get("error")]
    result: dict[str, Any] = {
        "instance_id": instance.instance_id,
        "pid": instance.pid,
        "socket_path": str(instance.socket_path),
        "restarted": True,
        # Rendered by the session-start text renderer under "loaded".
        "loaded": reloaded,
    }
    if project_roots:
        result["project_roots"] = project_roots
    if association_error is not None:
        result["project_association_error"] = association_error
    if reload_capture_failed:
        result["reload_capture_failed"] = True
        result["reload_capture_error"] = reload_capture_error
    cli._emit_result(args, result, text_renderer=_render_session_start_text, stem="session-restart")
    return 1 if failures or association_error or reload_capture_failed else 0


def _require_signal_delivered(reason: str | None, target_id: str) -> None:
    """Raise when a teardown signal was not delivered to the verified process.

    The registry pid is not an identity on its own: a crashed bridge's pid can be
    recycled, and SIGTERM/SIGKILL would then kill an unrelated process. The
    signaller pins and verifies the process, so *reason* being set means nothing
    was sent -- abort the restart loudly rather than respawn over a live bridge
    (#694).
    """
    if reason is not None:
        raise BridgeError(
            f"{reason}. Bridge instance {target_id} was left as it is; stop it "
            "manually, then run `bn session start`."
        )


def _associate_project_roots(
    instance_id: str | None,
    roots: list[str],
) -> tuple[list[str], str | None]:
    """Associate project roots on a bridge instance.

    Shared by `session start` (associating the caller's cwd on a fresh
    bridge) and `session restart` (restoring roots inherited from the old
    process). Returns ``(associated_roots, error)``: `associated_roots` is
    always a list (possibly empty), and `error` is set only when the
    association request itself failed or the bridge replied with something
    malformed. Skipped roots and request errors are warned to stderr so the
    reason stays visible even when stdout is redirected or the caller only
    checks the exit code.
    """
    if not roots:
        return [], None
    try:
        response = cli.send_request(
            "associate_project_roots",
            params={"roots": roots},
            instance_id=instance_id,
        )
        result = unwrap_result(response, "associate_project_roots")
    except BridgeError as exc:
        error = str(exc)
        print(f"warning: project association failed: {error}", file=sys.stderr)
        return [], error
    if not isinstance(result, dict):
        error = "bridge returned a malformed project association reply"
        print(f"warning: {error}", file=sys.stderr)
        return [], error
    for entry in result.get("skipped") or []:
        if isinstance(entry, dict):
            print(
                f"warning: project association {entry.get('path')} was skipped "
                f"({entry.get('reason')})",
                file=sys.stderr,
            )
    associated = result.get("associated")
    return (associated if isinstance(associated, list) else []), None


def _rss_mb(pid: int) -> float | None:
    """Read resident set size in MB from /proc/<pid>/status."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def _running_instances_result(selector_filter: str | None = None) -> dict[str, Any]:
    """Snapshot running bridge instances, optionally selecting one."""
    instances = cli.list_instances()
    sticky_id = cli.session_state.read().get("instance_id")
    entries = []
    total_rss = 0.0
    for inst in instances:
        rss = _rss_mb(inst.pid)
        selector = cli.instance_selector(inst)
        if selector_filter and selector_filter not in {
            inst.instance_id,
            selector,
        }:
            continue
        entry: dict[str, Any] = {
            "selector": selector,
            "instance_id": inst.instance_id,
            "pid": inst.pid,
            "socket_path": str(inst.socket_path),
            "started_at": inst.started_at,
            "rss_mb": round(rss, 1) if rss is not None else None,
        }
        # #80: the registry now records each instance's open binaries, so list them
        # here without an N-instance `target list` round-trip. Absent (older bridge
        # or nothing open) -> omit, so the entry stays clean.
        binaries = inst.meta.get("binaries") if isinstance(inst.meta, dict) else None
        if binaries:
            entry["binaries"] = list(binaries)
        project_roots = (
            inst.meta.get("project_roots") if isinstance(inst.meta, dict) else None
        )
        if (
            isinstance(project_roots, list)
            and project_roots
            and all(isinstance(root, str) for root in project_roots)
        ):
            entry["project_roots"] = list(project_roots)
        if sticky_id and (inst.instance_id == sticky_id or selector == sticky_id):
            entry["sticky"] = True
        entries.append(entry)
        if rss is not None:
            total_rss += rss
    # {kind, items} envelope for cross-command consistency with the rest of the
    # CLI's collection reads (#358); total_rss_mb stays as an extra field.
    return {"kind": "instances", "items": entries, "total_rss_mb": round(total_rss, 1)}


#: Budget for the `session list` unsaved-target probe, ONE deadline for the
#: whole command rather than one per instance. Short on purpose: a fleet triage
#: must not block on a wedged bridge, and an unanswered instance is reported as
#: unknown rather than as zero unsaved work.
#:
#: Applied with `resolved=True` so `BN_REQUEST_TIMEOUT` cannot widen it -- that
#: variable exists to give a long LOAD more room (and `none`/`off`/`0` disables
#: the deadline entirely), and a triage read that inherits it is exactly the
#: command a hung bridge takes hostage.
#:
#: Shared, not per instance and not per WAVE: the probes fan out `_PROBE_FAN`
#: at a time, so a per-probe budget made the listing cost
#: `ceil(n / _PROBE_FAN)` budgets -- 10s for the fourteen-bridge fleet in #733
#: and 125s for 200 -- and a Ctrl-C waited out the whole of it, because the
#: pool joins on exit. Each worker gets the REMAINING slice; a bridge the
#: deadline never reached is disclosed as unknown by name. Healthy bridges
#: answer in milliseconds, so this costs a real fleet nothing.
_UNSAVED_PROBE_TIMEOUT = 5.0

#: Concurrency cap for the probe, matching the survey planner's `min(8, n)`.
_PROBE_FAN = 8


def _probe_one_instance(entry: dict[str, Any], deadline: float) -> None:
    """Annotate ONE session row with its unsaved-target count, or with why not.

    `strict=True`, because this is a SAFETY count. `list_targets` runs the
    bridge's `TargetManager.refresh()`, whose non-strict view walk silently
    omits a tab when a UI query raises -- and a count read off that snapshot is
    a definitive "nothing would be discarded" about work the reader cannot see.
    Strict makes the bridge refuse the lossy snapshot instead, which arrives
    here as a transport error and is disclosed as unknown (#733 F1 review). The
    bridge no longer prunes its unsaved ledger off an incomplete walk either,
    so the state survives the failure rather than being erased by this read.

    The refresh itself is id minting plus the #713 prune of markers for views
    that are no longer open -- idempotent bookkeeping over the current open
    set, but a WRITE to it, newly reached by a triage command, so it is stated
    here.
    """
    # The legacy GUI pair registers no instance id, and its selector
    # (`default`) is what `choose_instance` matches on
    # (`instance_selector(inst) == instance_id`), so keying on `instance_id`
    # alone reported every GUI bridge as unknown without even asking it.
    selector = entry.get("instance_id") or entry.get("selector")
    if not isinstance(selector, str) or not selector:
        entry["unsaved_targets_unavailable"] = "the registry names no instance id"
        return
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        # Disclosed by name, never counted as zero: the point of this field is
        # that "0 unsaved" means "nothing would be discarded".
        entry["unsaved_targets_unavailable"] = (
            "the probe budget expired before this bridge was reached"
        )
        return
    try:
        rows = unwrap_result(
            cli.send_request(
                "list_targets",
                params={"strict": True},
                instance_id=selector,
                timeout=remaining,
                resolved=True,
            ),
            "list_targets",
        )
    except (BridgeError, OSError) as exc:
        # `BridgeError` is the declared surface -- `validate_instance_id`
        # translates the id-grammar `ValueError` into one and the socket
        # layer raises it for every transport failure -- and `OSError` is
        # belt: a listing must degrade to "unknown" rather than turn into a
        # traceback. First line, capped: a transport error can be a
        # multi-line "Multiple instances..." listing or a long "Start one
        # with: ..." hint, and this goes into a one-line-per-instance view.
        #
        # `or` twice, because an exception CAN have an empty message: a bare
        # `TimeoutError()` made `splitlines()[0]` an IndexError that escaped
        # the caught set and turned the whole listing into a traceback, and an
        # empty reason string renders as neither a count nor an `unknown`.
        reason = (str(exc).splitlines() or [""])[0][:120]
        entry["unsaved_targets_unavailable"] = reason or type(exc).__name__
        return
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        entry["unsaved_targets_unavailable"] = (
            "the bridge returned a malformed target listing"
        )
        return
    if any("unsaved" not in row for row in rows):
        entry["unsaved_targets_unavailable"] = (
            "this bridge does not report per-target unsaved state"
        )
        return
    entry["unsaved_targets"] = sum(1 for row in rows if row.get("unsaved") is True)


def _probe_unsaved_targets(snapshot: dict[str, Any]) -> None:
    """Annotate each session row with a LIVE count of unsaved targets (#733 F1).

    Registry meta is written on load/close only, so it cannot answer this; the
    ledger lives in the bridge. Each row gets `unsaved_targets` (an int) or
    `unsaved_targets_unavailable` (why not) -- never a fabricated 0, which reads
    as "nothing would be discarded" and is the decision this probe exists for.

    Concurrent, under ONE deadline for the whole sweep (`_UNSAVED_PROBE_TIMEOUT`),
    so the listing's worst case is that budget rather than one per bridge or one
    per wave of `_PROBE_FAN`. Each worker writes only into its own row dict, so
    the snapshot's order is the registry's regardless of completion order.
    """
    entries = [entry for entry in (snapshot.get("items") or [])
               if isinstance(entry, dict)]
    if not entries:
        return
    deadline = time.monotonic() + _UNSAVED_PROBE_TIMEOUT
    if len(entries) == 1:
        _probe_one_instance(entries[0], deadline)
        return
    with ThreadPoolExecutor(max_workers=min(_PROBE_FAN, len(entries))) as pool:
        list(pool.map(lambda entry: _probe_one_instance(entry, deadline), entries))


@command("session", "list", help="List running bridge sessions")
def _session_list(args: argparse.Namespace) -> int:
    # An explicit `-i ''` is an empty SELECTOR, not "no selector": the filter in
    # _running_instances_result is truthiness-based, so an empty string silently
    # disabled it and listed every session -- the opposite of what an explicit
    # selector asks for, and the same trap `bn session stop ""` already refuses
    # (#694). Rejected here through the shared guard so the message matches every
    # other command's.
    cli._require_nonempty_instance(args)
    selector_filter = (
        getattr(args, "instance", None)
        if getattr(args, "_explicit_instance", False)
        else None
    )
    result: Any = _running_instances_result(selector_filter)
    # #733 F1: "is it safe to stop these bridges?" answered on the read path.
    # A live probe, not registry meta: `_write_registry` runs on load/close
    # only, so a registry-sourced count would report "0 unsaved" for a bridge
    # holding unsaved renames -- the one wrong answer this field prevents.
    _probe_unsaved_targets(result)
    cli._emit_result(args, result, text_renderer=_render_session_list_text, stem="session-list")
    return 0


@command("instance", "list", help="List running bridge instances (alias for `session list`)")
def _instance_list(args: argparse.Namespace) -> int:
    # `instance list` ignores the selector entirely, but an explicit empty one is
    # still a caller mistake and is refused everywhere else in the CLI (#694).
    cli._require_nonempty_instance(args)
    # No unsaved probe here, unlike its `session list` alias: `instance list`
    # answers "which instance has libfoo.so?" from the registry alone, and #80
    # made round-trip-freedom that command's design contract.
    result: Any = _running_instances_result()
    cli._emit_result(args, result, text_renderer=_render_session_list_text, stem="instance-list")
    return 0


def _binary_query_matches(binary: str, query: str) -> bool:
    """Whether *binary* (an open binary's path) satisfies a `find` *query*: the
    exact path, the exact basename, or a substring of the basename (so `libfoo`
    finds `libfoo.so.1.2.11`). A path-form query (with a separator) also matches as
    a path-component-aligned suffix, and -- only for an ABSOLUTE query -- by
    resolved-path equality (a relative query would resolve against the CLI's cwd,
    not the bridge's, so resolving it would be misleading) (#80)."""
    if not binary or not query:
        return False
    if binary == query:
        return True
    if "/" in query:
        if Path(query).is_absolute():
            try:
                if Path(binary).resolve() == Path(query).resolve():
                    return True
            except Exception:
                pass
        # Anchor the suffix at a separator so `bar/libfoo.so` does NOT match
        # `/foobar/libfoo.so` (a mid-component byte suffix is a wrong answer).
        return binary.endswith("/" + query)
    base = Path(binary).name
    return base == query or query in base


@command("instance", "find", help="Find which running instance has a binary open (by path or name)",
         args=[arg("query", help="Binary path or (sub)name to locate among open binaries")])
def _instance_find(args: argparse.Namespace) -> int:
    snapshot = _running_instances_result()
    query = args.query
    items = []
    for entry in snapshot.get("items", []):
        for binary in entry.get("binaries") or []:
            if _binary_query_matches(str(binary), query):
                items.append({
                    "instance_id": entry.get("instance_id"),
                    "selector": entry.get("selector"),
                    "socket_path": entry.get("socket_path"),
                    "binary": binary,
                })
    result: Any = {"kind": "instance_matches", "query": query, "items": items, "count": len(items)}
    cli._emit_result(args, result, text_renderer=_render_instance_find_text, stem="instance-find")
    return 0


@command("instance", "use", help="Pin a bridge instance for subsequent calls", fmt="text",
         args=[arg("instance_id", help="Instance ID to pin (see `bn session list`)")])
def _instance_use(args: argparse.Namespace) -> int:
    instance_id = args.instance_id
    instances = cli.list_instances()
    matches = [
        inst for inst in instances
        if inst.instance_id == instance_id or cli.instance_selector(inst) == instance_id
    ]
    if not matches:
        raise BridgeError(f"No running bridge instance with id: {instance_id}")
    # Store the SELECTOR, not the raw instance_id. The fixed GUI bridge has
    # instance_id=None; persisting None makes session_state.update() DELETE the
    # pin (None means "remove the key"), so `bn instance use default` silently
    # left no pin and bare commands kept failing with "Multiple instances".
    # instance_selector() maps None -> "default", which resolution honors (#93).
    resolved = cli.instance_selector(matches[0])
    state = cli.session_state.read()
    prev_instance = state.get("instance_id")
    prev_target = state.get("target")
    cli.session_state.update(instance_id=resolved)
    result: Any = {"instance_id": resolved, "set": True}
    # #368 facet 3: a sticky target pin belongs to the instance it was set under.
    # When switching FROM a DIFFERENT pinned instance, clear it -- otherwise a
    # coincidentally matching selector in the new instance silently resolves a bare
    # command to a semantically different target. (Same-instance re-pin keeps it;
    # a first instance pin with no prior pinned instance is not a switch, so keep it
    # too -- prev_instance is None there, and None != resolved must NOT clear.)
    if prev_target and prev_instance is not None and prev_instance != resolved:
        cli.session_state.update(target=None)
        result["cleared_target_pin"] = prev_target
    cli._emit_result(args, result, text_renderer=_render_instance_use_text, stem="instance-use")
    return 0


@command("instance", "clear", help="Clear the pinned bridge instance", fmt="text")
def _instance_clear(args: argparse.Namespace) -> int:
    cli.session_state.update(instance_id=None)
    result: Any = {"instance_id": None, "set": False}
    cli._emit_result(args, result, text_renderer=_render_pin_clear_text, stem="instance-clear")
    return 0


@command("instance", "gc",
         help="Reap dead instances' leftover logs/sockets from ~/.cache/bn", fmt="text")
def _instance_gc(args: argparse.Namespace) -> int:
    # CLI-side maintenance: purge cache litter left by crashed/long-gone bridges
    # (the lazy liveness sweep keeps .log breadcrumbs forever) without touching
    # any live instance or the shared spawn lock (#80).
    result: Any = cli.gc_instances()
    cli._emit_result(args, result, text_renderer=_render_instance_gc_text, stem="instance-gc")
    return 0


@command("target", "list", help="List open BinaryView targets")
def _target_list(args: argparse.Namespace) -> int:
    response = cli.send_request(
        "list_targets",
        params={},
        instance_id=getattr(args, "instance", None),
    )
    result = unwrap_result(response, "list_targets")
    sticky = cli.session_state.read().get("target")
    if sticky and isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and _target_matches(item, sticky):
                item["sticky"] = True
    # Wrap the bare list in the {kind, items} envelope the rest of the CLI's
    # collection reads use, so an agent doesn't special-case target list (#358).
    envelope: Any = {"kind": "targets", "items": result if isinstance(result, list) else []}
    cli._emit_result(args, envelope, text_renderer=_render_target_list_text, stem="targets")
    return 0


def _target_matches(item: dict[str, Any], selector: str) -> bool:
    """True when *selector* names this target via any of its identifiers."""
    if selector == item.get("selector"):
        return True
    if selector == str(item.get("target_id", "")):
        return True
    if selector == str(item.get("view_id", "")):
        return True
    filename = item.get("filename")
    if isinstance(filename, str):
        if selector == filename:
            return True
        if selector == os.path.basename(filename):
            return True
    return False


@command("target", "use", help="Pin a target selector for subsequent calls", fmt="text",
         args=[arg("selector", help="Target selector to pin (see `bn target list`)")])
def _target_use(args: argparse.Namespace) -> int:
    # Validate the selector against the open targets BEFORE persisting, so a typo
    # doesn't silently poison the sticky project pin and brick every subsequent
    # selectorless target-required command (the consuming paths already reject an
    # unknown selector with exit 2) (#55).
    response = cli.send_request(
        "list_targets", params={}, instance_id=getattr(args, "instance", None)
    )
    targets = unwrap_result(response, "list_targets") or []
    matched = isinstance(targets, list) and any(
        isinstance(item, dict) and _target_matches(item, args.selector) for item in targets
    )
    if not matched:
        open_sel = ", ".join(
            str(item.get("selector")) for item in targets if isinstance(item, dict)
        ) or "<none open>"
        raise BridgeError(
            f"Unknown target selector: {args.selector}. Open targets: {open_sel}. "
            f"Run `bn target list` to see them. The pin was left unchanged."
        )
    cli.session_state.update(target=args.selector)
    result: Any = {"target": args.selector, "set": True}
    cli._emit_result(args, result, text_renderer=_render_target_use_text, stem="target-use")
    return 0


@command("target", "clear", help="Clear the pinned target", fmt="text")
def _target_clear(args: argparse.Namespace) -> int:
    cli.session_state.update(target=None)
    result: Any = {"target": None, "set": False}
    cli._emit_result(args, result, text_renderer=_render_pin_clear_text, stem="target-clear")
    return 0
