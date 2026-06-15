"""Administrative commands: doctor, installs, sessions, and sticky pins.

Shared infrastructure (transport, paths, session state) is accessed through
the ``bn.cli`` module at call time -- ``cli.send_request(...)`` rather than a
``from ..cli import send_request`` -- so tests and scripts can monkeypatch
``bn.cli`` as the single well-known location.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Any

from .. import cli
from ..cli import arg, command
from ..formatters import (
    _render_doctor_text,
    _render_instance_use_text,
    _render_pin_clear_text,
    _render_session_list_text,
    _render_session_start_text,
    _render_session_stop_text,
    _render_skill_install_text,
    _render_target_list_text,
    _render_target_use_text,
)
from ..transport import BridgeError


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
            ping = response["result"]
        except Exception as exc:
            ping = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        loaded_version = ping.get("plugin_version") if isinstance(ping, dict) else None
        loaded_build_id = ping.get("plugin_build_id") if isinstance(ping, dict) else None
        loaded_engine_id = ping.get("engine_build_id") if isinstance(ping, dict) else None
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
    if args.format == "text":
        result = _render_doctor_text(result)
    cli._render_result(result, fmt=args.format, out_path=args.out, stem="doctor")
    return 0


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

    cli._render_result(
        {
            "installed": True,
            "mode": args.mode,
            "source": str(source),
            "destination": str(dest),
        },
        fmt=args.format,
        out_path=args.out,
        stem="plugin-install",
    )
    return 0


def _install_tree(source: Path, dest: Path, *, mode: str, force: bool) -> None:
    if not source.exists():
        raise BridgeError(f"Source directory is missing: {source}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() or dest.is_symlink():
        if not force:
            raise BridgeError(f"Destination already exists: {dest}")
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)

    if mode == "copy":
        shutil.copytree(source, dest)
    else:
        os.symlink(source, dest, target_is_directory=True)


def _check_install_destination(dest: Path, *, force: bool) -> None:
    if force:
        return
    if dest.exists() or dest.is_symlink():
        raise BridgeError(f"Destination already exists: {dest}")


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
        _check_install_destination(dest, force=args.force)
        pending_installs.append((source, dest))

    for source, dest in pending_installs:
        _install_tree(source, dest, mode=args.mode, force=args.force)

    result = {
        "installed": True,
        "mode": args.mode,
        "installed_destinations": [str(dest) for _, dest in pending_installs],
        "skipped_destinations": skipped_destinations,
        "skills": results,
    }
    if args.format == "text":
        result = _render_skill_install_text(result)
    cli._render_result(result, fmt=args.format, out_path=args.out, stem="skill-install")
    return 0


def _default_skill_install_roots() -> list[Path]:
    roots = [cli.claude_skills_dir()]
    if cli.codex_home().is_dir():
        roots.append(cli.codex_skills_dir())
    return roots


@command("session", "start", help="Start a new headless bridge session",
         args=[
             arg("binaries", nargs="*", help="Binary file paths to preload"),
             arg("--instance-id", help="Use a specific instance ID (default: random)"),
             arg("--no-bndb", action="store_true",
                 help="Don't auto-prefer a sibling .bndb file"),
             arg("--quick", "--no-analysis", action="store_true",
                 help="Preload without full analysis (fast); run `bn refresh` for full analysis"),
         ])
def _session_start(args: argparse.Namespace) -> int:
    instance_id = getattr(args, "instance_id", None)
    instance = cli.spawn_instance(instance_id)

    binaries = getattr(args, "binaries", None) or []
    prefer_bndb = not args.no_bndb
    quick = bool(getattr(args, "quick", False))
    loaded = []
    for binary in binaries:
        resolved = str(Path(binary).expanduser().resolve())
        try:
            resp = cli.send_request(
                "load_binary",
                params={"path": resolved, "prefer_bndb": prefer_bndb, "quick": quick},
                instance_id=instance.instance_id,
            )
            loaded.append(resp["result"])
        except BridgeError as exc:
            loaded.append({"path": resolved, "error": str(exc)})

    failures = [item for item in loaded if isinstance(item, dict) and item.get("error")]
    successes = [item for item in loaded if isinstance(item, dict) and not item.get("error")]

    result: dict[str, Any] = {
        "instance_id": instance.instance_id,
        "pid": instance.pid,
        "socket_path": str(instance.socket_path),
    }
    if loaded:
        result["loaded"] = loaded

    # If the caller asked to preload binaries but none loaded, the freshly
    # spawned bridge is an empty zombie they'd have to hunt down and stop. Shut
    # it down and exit non-zero so the failure is visible to scripts.
    if binaries and not successes:
        try:
            cli.send_request("shutdown", instance_id=instance.instance_id)
        except BridgeError:
            pass
        result["stopped"] = True

    if args.format == "text":
        result = _render_session_start_text(result)
    cli._render_result(result, fmt=args.format, out_path=args.out, stem="session-start")
    return 1 if failures else 0


@command("session", "stop", help="Stop a running bridge session",
         args=[arg("instance", help="Instance ID to stop")])
def _session_stop(args: argparse.Namespace) -> int:
    target_id = args.instance
    # Resolve the instance up front so we can confirm teardown by pid + files
    # after the shutdown, and SIGTERM it if the socket request fails.
    inst = next(
        (
            i for i in cli.list_instances()
            if i.instance_id == target_id or cli.instance_selector(i) == target_id
        ),
        None,
    )
    try:
        cli.send_request("shutdown", instance_id=target_id)
        method = "shutdown"
    except BridgeError:
        # Fallback: SIGTERM the process directly.
        if inst is None:
            raise BridgeError(f"No bridge instance found with id: {target_id}")
        try:
            os.kill(inst.pid, signal.SIGTERM)
        except OSError as exc:
            print(
                f"error: failed to stop bridge instance {target_id} "
                f"(pid {inst.pid}): {exc}",
                file=sys.stderr,
            )
            return 1
        method = "sigterm"

    result: dict[str, Any] = {"instance_id": target_id, "stopped": True}
    if method == "sigterm":
        result["method"] = method

    # Block until the socket/registry are gone and the process has exited, so a
    # follow-on `bn session start --instance-id <same>` can't race the dying
    # instance and fail as a duplicate (#92 Problem B). Escalate to SIGKILL if
    # graceful teardown stalls, and report failure if it never converges.
    if inst is not None:
        if not cli.wait_for_teardown(inst, timeout=5.0):
            with contextlib.suppress(OSError):
                os.kill(inst.pid, signal.SIGKILL)
            if not cli.wait_for_teardown(inst, timeout=2.0):
                result["stopped"] = False
                result["error"] = (
                    f"bridge instance {target_id} (pid {inst.pid}) did not fully "
                    "tear down; registry/socket may be stale."
                )
                if args.format == "text":
                    result = _render_session_stop_text(result)
                cli._render_result(result, fmt=args.format, out_path=args.out, stem="session-stop")
                return 1
            result["method"] = "sigkill"

    if args.format == "text":
        result = _render_session_stop_text(result)
    cli._render_result(result, fmt=args.format, out_path=args.out, stem="session-stop")
    return 0


@command("session", "restart", help="Stop a bridge session and respawn it (same id), reloading its targets",
         args=[arg("instance", help="Instance ID to restart")])
def _session_restart(args: argparse.Namespace) -> int:
    """Cleanly reload a live session running stale code (#161): capture its open
    targets, stop the process, respawn under the same instance id, and reload the
    same binaries -- so the new bridge serves current code without the caller
    hunting for the right paths."""
    target_id = args.instance
    inst = next(
        (
            i for i in cli.list_instances()
            if i.instance_id == target_id or cli.instance_selector(i) == target_id
        ),
        None,
    )
    if inst is None:
        raise BridgeError(
            f"No bridge instance found with id: {target_id}. See `bn session list`."
        )
    resolved_id = inst.instance_id

    # Capture loaded targets (+ their analysis state) BEFORE tearing down.
    reload_targets: list[dict[str, Any]] = []
    try:
        resp = cli._send_request_to_instance(inst, "list_targets", params={}, target=None)
        for t in (resp.get("result") or []):
            path = t.get("filename")
            if path:
                reload_targets.append({"path": path, "quick": t.get("analysis_state") == "quick"})
    except Exception:
        pass

    # Stop the old process: graceful shutdown, then SIGTERM/SIGKILL escalation,
    # blocking until the socket/registry are gone so the respawn can reuse the id.
    try:
        cli.send_request("shutdown", instance_id=resolved_id)
    except BridgeError:
        with contextlib.suppress(OSError):
            os.kill(inst.pid, signal.SIGTERM)
    if not cli.wait_for_teardown(inst, timeout=5.0):
        with contextlib.suppress(OSError):
            os.kill(inst.pid, signal.SIGKILL)
        if not cli.wait_for_teardown(inst, timeout=2.0):
            raise BridgeError(
                f"bridge instance {target_id} (pid {inst.pid}) did not tear down; "
                "cannot restart cleanly. Stop it manually and re-spawn."
            )

    instance = cli.spawn_instance(resolved_id)
    reloaded: list[Any] = []
    for t in reload_targets:
        try:
            r = cli.send_request(
                "load_binary",
                params={"path": t["path"], "prefer_bndb": True, "quick": t["quick"]},
                instance_id=instance.instance_id,
            )
            reloaded.append(r["result"])
        except BridgeError as exc:
            reloaded.append({"path": t["path"], "error": str(exc)})

    failures = [x for x in reloaded if isinstance(x, dict) and x.get("error")]
    result: dict[str, Any] = {
        "instance_id": instance.instance_id,
        "pid": instance.pid,
        "socket_path": str(instance.socket_path),
        "restarted": True,
        # Rendered by the session-start text renderer under "loaded".
        "loaded": reloaded,
    }
    if args.format == "text":
        result = _render_session_start_text(result)
    cli._render_result(result, fmt=args.format, out_path=args.out, stem="session-restart")
    return 1 if failures else 0


def _rss_mb(pid: int) -> float | None:
    """Read resident set size in MB from /proc/<pid>/status."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def _running_instances_result() -> dict[str, Any]:
    """Snapshot every running bridge instance (shared by session/instance list)."""
    instances = cli.list_instances()
    sticky_id = cli.session_state.read().get("instance_id")
    entries = []
    total_rss = 0.0
    for inst in instances:
        rss = _rss_mb(inst.pid)
        selector = cli.instance_selector(inst)
        entry: dict[str, Any] = {
            "selector": selector,
            "instance_id": inst.instance_id,
            "pid": inst.pid,
            "socket_path": str(inst.socket_path),
            "started_at": inst.started_at,
            "rss_mb": round(rss, 1) if rss is not None else None,
        }
        if sticky_id and (inst.instance_id == sticky_id or selector == sticky_id):
            entry["sticky"] = True
        entries.append(entry)
        if rss is not None:
            total_rss += rss
    return {"instances": entries, "total_rss_mb": round(total_rss, 1)}


@command("session", "list", help="List running bridge sessions")
def _session_list(args: argparse.Namespace) -> int:
    result: Any = _running_instances_result()
    if args.format == "text":
        result = _render_session_list_text(result)
    cli._render_result(result, fmt=args.format, out_path=args.out, stem="session-list")
    return 0


@command("instance", "list", help="List running bridge instances (alias for `session list`)")
def _instance_list(args: argparse.Namespace) -> int:
    result: Any = _running_instances_result()
    if args.format == "text":
        result = _render_session_list_text(result)
    cli._render_result(result, fmt=args.format, out_path=args.out, stem="instance-list")
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
    cli.session_state.update(instance_id=resolved)
    result: Any = {"instance_id": resolved, "set": True}
    if args.format == "text":
        result = _render_instance_use_text(result)
    cli._render_result(result, fmt=args.format, out_path=args.out, stem="instance-use")
    return 0


@command("instance", "clear", help="Clear the pinned bridge instance", fmt="text")
def _instance_clear(args: argparse.Namespace) -> int:
    cli.session_state.update(instance_id=None)
    result: Any = {"instance_id": None, "set": False}
    if args.format == "text":
        result = _render_pin_clear_text(result)
    cli._render_result(result, fmt=args.format, out_path=args.out, stem="instance-clear")
    return 0


@command("target", "list", help="List open BinaryView targets")
def _target_list(args: argparse.Namespace) -> int:
    response = cli.send_request(
        "list_targets",
        params={},
        instance_id=getattr(args, "instance", None),
    )
    result = response["result"]
    sticky = cli.session_state.read().get("target")
    if sticky and isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and _target_matches(item, sticky):
                item["sticky"] = True
    if args.format == "text":
        result = _render_target_list_text(result)
    cli._render_result(result, fmt=args.format, out_path=args.out, stem="targets")
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
    targets = response.get("result") or []
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
    if args.format == "text":
        result = _render_target_use_text(result)
    cli._render_result(result, fmt=args.format, out_path=args.out, stem="target-use")
    return 0


@command("target", "clear", help="Clear the pinned target", fmt="text")
def _target_clear(args: argparse.Namespace) -> int:
    cli.session_state.update(target=None)
    result: Any = {"target": None, "set": False}
    if args.format == "text":
        result = _render_pin_clear_text(result)
    cli._render_result(result, fmt=args.format, out_path=args.out, stem="target-clear")
    return 0
