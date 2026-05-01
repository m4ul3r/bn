from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from . import session_state
from .formatters import (
    FAILED_MUTATION_STATUSES,
    _format_operation_result,  # noqa: F401  -- re-exported for tests/scripts that monkeypatch bn.cli
    _render_doctor_text,
    _render_session_list_text,
    _render_session_start_text,
    _render_session_stop_text,
    _render_skill_install_text,
    _render_target_choices,
    _render_target_list_text,
)
from .output import render_artifact_envelope, write_output_result
from .paths import (
    claude_skills_dir,
    codex_home,
    codex_skills_dir,
    plugin_install_dir,
    plugin_source_dir,
    repo_root,
)
from .transport import (
    BridgeError,
    _send_request_to_instance,
    instance_selector,
    list_instances,
    send_request,
    spawn_instance,
)
from .version import VERSION, build_id_for_file


class _HelpFullAction(argparse.Action):
    def __init__(
        self,
        option_strings: list[str],
        dest: str = argparse.SUPPRESS,
        default: str = argparse.SUPPRESS,
        help: str | None = None,
    ) -> None:
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help,
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | list[str] | None,
        option_string: str | None = None,
    ) -> None:
        if isinstance(parser, BnArgumentParser):
            parser.print_full_help()
        else:
            parser.print_help()
        parser.exit()


class BnArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.set_defaults(_parser=self)
        self.add_argument(
            "--help-full",
            action=_HelpFullAction,
            help="Show help for this command and all subcommands",
        )

    def _iter_full_help_parsers(self) -> list[argparse.ArgumentParser]:
        parsers: list[argparse.ArgumentParser] = [self]
        for action in self._actions:
            if isinstance(action, argparse._SubParsersAction):
                for parser in action.choices.values():
                    if isinstance(parser, BnArgumentParser):
                        parsers.extend(parser._iter_full_help_parsers())
                    else:
                        parsers.append(parser)
        return parsers

    def _full_help_actions(self) -> tuple[type[argparse.Action], ...]:
        return (argparse._HelpAction, _HelpFullAction)

    def format_help_for_full(self) -> str:
        formatter = self._get_formatter()
        help_action_types = self._full_help_actions()
        actions = [action for action in self._actions if not isinstance(action, help_action_types)]

        formatter.add_usage(self.usage, actions, self._mutually_exclusive_groups)
        formatter.add_text(self.description)

        for action_group in self._action_groups:
            group_actions = [
                action
                for action in action_group._group_actions
                if not isinstance(action, help_action_types)
            ]
            if not group_actions:
                continue
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(group_actions)
            formatter.end_section()

        formatter.add_text(self.epilog)
        return formatter.format_help()

    def format_full_help(self) -> str:
        sections: list[str] = []
        seen: set[int] = set()
        for parser in self._iter_full_help_parsers():
            parser_id = id(parser)
            if parser_id in seen:
                continue
            seen.add(parser_id)
            if isinstance(parser, BnArgumentParser):
                sections.append(parser.format_help_for_full().rstrip())
            else:
                sections.append(parser.format_help().rstrip())
        return "\n\n".join(sections) + "\n"

    def print_full_help(self, file: Any = None) -> None:
        if file is None:
            file = sys.stdout
        self._print_message(self.format_full_help(), file)


def _package_version() -> str:
    return VERSION


def _common_io_options(
    parser: argparse.ArgumentParser,
    *,
    default_format: str = "text",
) -> None:
    parser.add_argument(
        "--format",
        choices=("json", "text", "ndjson"),
        default=default_format,
        help="Output format",
    )
    parser.add_argument("--out", type=Path, help="Write output to a file instead of stdout")


def _instance_option(parser: argparse.ArgumentParser, *, is_root: bool = False) -> None:
    parser.add_argument(
        "--instance",
        default=os.environ.get("BN_INSTANCE") if is_root else argparse.SUPPRESS,
        help="Target a specific bridge instance by ID (env: BN_INSTANCE)",
    )


def _target_option(
    parser: argparse.ArgumentParser,
    *,
    required: bool,
    is_root: bool = False,
) -> None:
    kwargs: dict[str, Any] = {
        "help": (
            "Target selector from `bn target list` (`selector`, `target_id`, basename, filename, or view id); "
            "omit only when exactly one target is open, or use `active` to follow the GUI-selected target explicitly"
        ),
        "required": required,
    }
    if not is_root:
        kwargs["default"] = argparse.SUPPRESS
    parser.add_argument("-t", "--target", **kwargs)


# ---------------------------------------------------------------------------
# Declarative command registration
# ---------------------------------------------------------------------------

_COMMANDS: list[dict[str, Any]] = []

_GROUP_HELP: dict[tuple[str, ...], str] = {
    ("plugin",): "Install the Binary Ninja companion plugin",
    ("skill",): "Install the bundled agent skills",
    ("session",): "Manage bridge sessions",
    ("instance",): "Pin or clear the active bridge instance",
    ("target",): "Inspect Binary Ninja targets",
    ("function",): "Function discovery helpers",
    ("bundle",): "Export reusable bundles",
    ("py",): "Execute Python inside Binary Ninja",
    ("symbol",): "Rename functions or data",
    ("comment",): "Set or delete comments",
    ("proto",): "Inspect or set a user prototype",
    ("local",): "Inspect, rename, or retype locals",
    ("struct",): "Field-first structure editing",
    ("struct", "field"): "Operate on struct fields",
    ("batch",): "Apply a batch manifest",
}


def arg(*flags: str, **kwargs: Any) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Define an argument spec for :func:`command`."""
    return (flags, kwargs)


def mutex(required: bool, *args: tuple[tuple[str, ...], dict[str, Any]]) -> tuple[bool, list[tuple[tuple[str, ...], dict[str, Any]]]]:
    """Define a mutually exclusive argument group for :func:`command`."""
    return (required, list(args))


def command(
    *path: str,
    help: str = "",
    fmt: str = "text",
    target: bool = False,
    paged: bool = False,
    address_filter: bool = False,
    args: list[tuple[tuple[str, ...], dict[str, Any]]] | None = None,
    mutex_groups: list[tuple[bool, list[tuple[tuple[str, ...], dict[str, Any]]]]] | None = None,
) -> Callable:
    """Register a CLI command declaratively."""

    def decorator(fn: Callable[[argparse.Namespace], int]) -> Callable[[argparse.Namespace], int]:
        _COMMANDS.append({
            "path": path,
            "handler": fn,
            "help": help,
            "fmt": fmt,
            "target": target,
            "paged": paged,
            "address_filter": address_filter,
            "args": args or [],
            "mutex_groups": mutex_groups or [],
        })
        return fn

    return decorator


def _build_from_commands(root: BnArgumentParser) -> None:
    """Populate *root* with subcommands from the ``_COMMANDS`` registry."""
    subparser_actions: dict[tuple[str, ...], argparse._SubParsersAction] = {}
    node_parsers: dict[tuple[str, ...], argparse.ArgumentParser] = {(): root}

    def _get_subparsers(parent: tuple[str, ...]) -> argparse._SubParsersAction:
        if parent not in subparser_actions:
            dest = "_".join(parent) + "_command" if parent else "command"
            subparser_actions[parent] = node_parsers[parent].add_subparsers(dest=dest)
        return subparser_actions[parent]

    def _ensure_intermediate(path: tuple[str, ...]) -> argparse.ArgumentParser:
        if path in node_parsers:
            return node_parsers[path]
        if len(path) > 1:
            _ensure_intermediate(path[:-1])
        sub = _get_subparsers(path[:-1])
        parser = sub.add_parser(path[-1], help=_GROUP_HELP.get(path, ""))
        node_parsers[path] = parser
        return parser

    for spec in sorted(_COMMANDS, key=lambda s: len(s["path"])):
        path = spec["path"]
        parent = path[:-1]

        if parent:
            _ensure_intermediate(parent)

        if path in node_parsers:
            cmd = node_parsers[path]
        else:
            cmd = _get_subparsers(parent).add_parser(path[-1], help=spec["help"])
            node_parsers[path] = cmd

        _common_io_options(cmd, default_format=spec["fmt"])
        _instance_option(cmd)
        if spec["target"]:
            _target_option(cmd, required=False)
        if spec["address_filter"]:
            _add_function_address_args(cmd)
        if spec["paged"]:
            _add_paged_args(cmd)

        for flags, kwargs in spec["args"]:
            cmd.add_argument(*flags, **kwargs)

        for required, group_args in spec["mutex_groups"]:
            group = cmd.add_mutually_exclusive_group(required=required)
            for flags, kwargs in group_args:
                group.add_argument(*flags, **kwargs)

        cmd.set_defaults(handler=spec["handler"])


def _render_result(
    value: Any,
    *,
    fmt: str,
    out_path: Path | None,
    stem: str,
    spill_label: str | None = None,
    spill_context: Any = None,
) -> None:
    if out_path is None and isinstance(value, dict) and isinstance(value.get("artifact_path"), str):
        artifact = dict(value)
        artifact.setdefault("ok", True)
        artifact.setdefault("spilled", False)
        sys.stdout.write(render_artifact_envelope(artifact))
        return

    result = write_output_result(value, fmt=fmt, out_path=out_path, stem=stem)
    if result.spilled and result.artifact:
        label = spill_label or stem.replace("_", " ")
        artifact = result.artifact
        sys.stdout.write(result.rendered)
        print(f"warning: {label} output spilled to {artifact['artifact_path']}", file=sys.stderr)
        return
    sys.stdout.write(result.rendered)


def _implicit_target(args: argparse.Namespace) -> str:
    response = send_request(
        "list_targets",
        params={},
        target=None,
        instance_id=getattr(args, "instance", None),
    )
    targets = list(response["result"])
    if len(targets) == 1:
        return "active"
    if not targets:
        raise BridgeError("No BinaryView targets are open")
    raise BridgeError(
        "This command requires --target when multiple targets are open.\n"
        f"Open targets:\n{_render_target_choices(targets)}"
    )


def _resolve_target(
    args: argparse.Namespace,
    *,
    require_target: bool,
    allow_implicit_target: bool = False,
) -> str | None:
    target = getattr(args, "target", None)
    if require_target and not target:
        if allow_implicit_target:
            return _implicit_target(args)
        raise BridgeError("This command requires --target")
    return target


def _mutation_exit_code(result: Any) -> int:
    if not isinstance(result, dict):
        return 0
    results = list(result.get("results") or [])
    if any(isinstance(item, dict) and item.get("status") in FAILED_MUTATION_STATUSES for item in results):
        return 3
    if result.get("success") is False:
        return 3
    return 0


def _call(
    args: argparse.Namespace,
    op: str,
    params: dict[str, Any] | None = None,
    *,
    require_target: bool,
    allow_implicit_target: bool = False,
    text_renderer: Callable[[Any], str] | None = None,
    page_limit: int | None = None,
    page_offset: int = 0,
    page_label: str | None = None,
    stem: str,
    result_exit_code: Callable[[Any], int] | None = None,
    bridge_writes_output: bool = False,
) -> int:
    request_params = dict(params or {})
    effective_page_limit = None
    if page_limit is not None and page_limit >= 0:
        effective_page_limit = page_limit
        request_params["limit"] = page_limit + 1

    target = _resolve_target(
        args,
        require_target=require_target,
        allow_implicit_target=allow_implicit_target,
    )
    response = send_request(
        op,
        params=request_params,
        target=target,
        instance_id=getattr(args, "instance", None),
    )
    result = response["result"]
    exit_code = result_exit_code(result) if result_exit_code is not None else 0
    if effective_page_limit is not None and isinstance(result, list) and len(result) > effective_page_limit:
        result = result[:effective_page_limit]
        label = page_label or op
        next_offset = page_offset + effective_page_limit
        print(
            f"warning: {label} output truncated to {effective_page_limit} items; rerun with --offset {next_offset} or a larger --limit",
            file=sys.stderr,
        )
    spill_context = result
    if text_renderer is not None and args.format == "text":
        result = text_renderer(result)
    _render_result(
        result,
        fmt=args.format,
        out_path=None if bridge_writes_output else args.out,
        stem=stem,
        spill_label=page_label or op.replace("_", " "),
        spill_context=spill_context,
    )
    return exit_code


def _parse_line_range(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected START:END, got {value!r}")
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected START:END with integers, got {value!r}")
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError(f"invalid range: {start}:{end}")
    return (start, end)


@command("doctor", help="Validate bridge discovery and installation")
def _doctor(args: argparse.Namespace) -> int:
    install_dir = plugin_install_dir()
    source_dir = plugin_source_dir()
    install_bridge = install_dir / "bridge.py"
    source_bridge = source_dir / "bridge.py"
    install_build_id = build_id_for_file(install_bridge)
    source_build_id = build_id_for_file(source_bridge)
    instances = []
    for instance in list_instances():
        ping: dict[str, Any]
        try:
            response = _send_request_to_instance(
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
        instances.append(
            {
                "pid": instance.pid,
                "socket_path": str(instance.socket_path),
                "plugin_version": instance.plugin_version,
                "plugin_build_id": loaded_build_id,
                "installed_plugin_build_id": install_build_id,
                "source_plugin_build_id": source_build_id,
                "stale_plugin_version": (
                    bool(loaded_version)
                    and str(loaded_version) != _package_version()
                ),
                "stale_plugin_code": (
                    bool(loaded_build_id)
                    and install_build_id is not None
                    and loaded_build_id != install_build_id
                ),
                "started_at": instance.started_at,
                "doctor": ping,
            }
        )

    result = {
        "cli_version": _package_version(),
        "plugin_source_dir": str(source_dir),
        "plugin_install_dir": str(install_dir),
        "plugin_source_build_id": source_build_id,
        "plugin_install_build_id": install_build_id,
        "instances": instances,
    }
    if args.format == "text":
        result = _render_doctor_text(result)
    _render_result(result, fmt=args.format, out_path=args.out, stem="doctor")
    return 0


@command("plugin", "install", help="Install the GUI plugin", fmt="json",
         args=[
             arg("--dest", type=Path, help="Custom install destination"),
             arg("--mode", choices=("symlink", "copy"), default="symlink"),
             arg("--force", action="store_true"),
         ])
def _plugin_install(args: argparse.Namespace) -> int:
    source = plugin_source_dir()
    dest = args.dest or plugin_install_dir()
    _install_tree(source, dest, mode=args.mode, force=args.force)

    _render_result(
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
    skills_root = repo_root() / "skills"
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
    _render_result(result, fmt=args.format, out_path=args.out, stem="skill-install")
    return 0


def _default_skill_install_roots() -> list[Path]:
    roots = [claude_skills_dir()]
    if codex_home().is_dir():
        roots.append(codex_skills_dir())
    return roots


@command("session", "start", help="Start a new headless bridge session",
         args=[
             arg("binaries", nargs="*", help="Binary file paths to preload"),
             arg("--instance-id", help="Use a specific instance ID (default: random)"),
             arg("--no-bndb", action="store_true",
                 help="Don't auto-prefer a sibling .bndb file"),
         ])
def _session_start(args: argparse.Namespace) -> int:
    instance_id = getattr(args, "instance_id", None)
    instance = spawn_instance(instance_id)

    binaries = getattr(args, "binaries", None) or []
    prefer_bndb = not args.no_bndb
    loaded = []
    for binary in binaries:
        resolved = str(Path(binary).expanduser().resolve())
        try:
            resp = send_request(
                "load_binary",
                params={"path": resolved, "prefer_bndb": prefer_bndb},
                instance_id=instance.instance_id,
            )
            loaded.append(resp["result"])
        except BridgeError as exc:
            loaded.append({"path": resolved, "error": str(exc)})

    result: dict[str, Any] = {
        "instance_id": instance.instance_id,
        "pid": instance.pid,
        "socket_path": str(instance.socket_path),
    }
    if loaded:
        result["loaded"] = loaded

    if args.format == "text":
        result = _render_session_start_text(result)
    _render_result(result, fmt=args.format, out_path=args.out, stem="session-start")
    return 0


@command("session", "stop", help="Stop a running bridge session",
         args=[arg("instance", help="Instance ID to stop")])
def _session_stop(args: argparse.Namespace) -> int:
    import signal

    target_id = args.instance
    try:
        resp = send_request("shutdown", instance_id=target_id)
        result = {"instance_id": target_id, "stopped": True}
    except BridgeError:
        # Fallback: find instance and SIGTERM
        for inst in list_instances():
            if inst.instance_id == target_id or instance_selector(inst) == target_id:
                try:
                    os.kill(inst.pid, signal.SIGTERM)
                except OSError:
                    pass
                result = {"instance_id": target_id, "stopped": True, "method": "sigterm"}
                break
        else:
            raise BridgeError(f"No bridge instance found with id: {target_id}")

    if args.format == "text":
        result = _render_session_stop_text(result)
    _render_result(result, fmt=args.format, out_path=args.out, stem="session-stop")
    return 0


def _rss_mb(pid: int) -> float | None:
    """Read resident set size in MB from /proc/<pid>/status."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


@command("session", "list", help="List running bridge sessions")
def _session_list(args: argparse.Namespace) -> int:
    instances = list_instances()
    sticky_id = session_state.read().get("instance_id")
    entries = []
    total_rss = 0.0
    for inst in instances:
        rss = _rss_mb(inst.pid)
        selector = instance_selector(inst)
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
    result: dict[str, Any] = {
        "instances": entries,
        "total_rss_mb": round(total_rss, 1),
    }
    if args.format == "text":
        result = _render_session_list_text(result)
    _render_result(result, fmt=args.format, out_path=args.out, stem="session-list")
    return 0


@command("instance", "use", help="Pin a bridge instance for subsequent calls", fmt="text",
         args=[arg("instance_id", help="Instance ID to pin (see `bn session list`)")])
def _instance_use(args: argparse.Namespace) -> int:
    instance_id = args.instance_id
    instances = list_instances()
    matches = [
        inst for inst in instances
        if inst.instance_id == instance_id or instance_selector(inst) == instance_id
    ]
    if not matches:
        raise BridgeError(f"No running bridge instance with id: {instance_id}")
    resolved = matches[0].instance_id
    session_state.update(instance_id=resolved)
    result = {"instance_id": resolved, "set": True}
    if args.format == "text":
        result = f"instance: {resolved}"
    _render_result(result, fmt=args.format, out_path=args.out, stem="instance-use")
    return 0


@command("instance", "clear", help="Clear the pinned bridge instance", fmt="text")
def _instance_clear(args: argparse.Namespace) -> int:
    session_state.update(instance_id=None)
    result = {"instance_id": None, "set": False}
    if args.format == "text":
        result = "cleared"
    _render_result(result, fmt=args.format, out_path=args.out, stem="instance-clear")
    return 0


@command("target", "list", help="List open BinaryView targets")
def _target_list(args: argparse.Namespace) -> int:
    response = send_request(
        "list_targets",
        params={},
        instance_id=getattr(args, "instance", None),
    )
    result = response["result"]
    sticky = session_state.read().get("target")
    if sticky and isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and _target_matches(item, sticky):
                item["sticky"] = True
    if args.format == "text":
        result = _render_target_list_text(result)
    _render_result(result, fmt=args.format, out_path=args.out, stem="targets")
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
    session_state.update(target=args.selector)
    result = {"target": args.selector, "set": True}
    if args.format == "text":
        result = f"target: {args.selector}"
    _render_result(result, fmt=args.format, out_path=args.out, stem="target-use")
    return 0


@command("target", "clear", help="Clear the pinned target", fmt="text")
def _target_clear(args: argparse.Namespace) -> int:
    session_state.update(target=None)
    result = {"target": None, "set": False}
    if args.format == "text":
        result = "cleared"
    _render_result(result, fmt=args.format, out_path=args.out, stem="target-clear")
    return 0


def _add_paged_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)


def _add_function_address_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--min-address",
        help="Only include functions whose start address is at or above this address",
    )
    parser.add_argument(
        "--max-address",
        help="Only include functions whose start address is at or below this address",
    )


def build_parser() -> argparse.ArgumentParser:
    # Importing here populates _COMMANDS via @command decorators in submodules.
    # Deferred until call time to keep cli.py importable on its own.
    from . import commands  # noqa: F401

    parser = BnArgumentParser(prog="bn", description="Agent-friendly Binary Ninja CLI")
    parser.set_defaults(handler=None)
    _instance_option(parser, is_root=True)
    _target_option(parser, required=False, is_root=True)
    _build_from_commands(parser)
    return parser


def _apply_sticky_defaults(args: argparse.Namespace) -> None:
    """Fill unset --instance / --target from per-project sticky state."""
    state = session_state.read()
    if not getattr(args, "instance", None):
        sticky_instance = state.get("instance_id")
        if sticky_instance:
            args.instance = sticky_instance
            args._sticky_instance = True
    if not getattr(args, "target", None):
        sticky_target = state.get("target")
        if sticky_target:
            args.target = sticky_target
            args._sticky_target = True


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] | None = getattr(args, "handler", None)
    if handler is None:
        selected_parser = getattr(args, "_parser", parser)
        selected_parser.print_help()
        return 1

    _apply_sticky_defaults(args)

    try:
        return handler(args)
    except BridgeError as exc:
        msg = str(exc)
        if getattr(args, "_sticky_instance", False) and _looks_like_dead_bridge(msg):
            msg += "\n\nThis came from sticky state. Clear it with `bn instance clear`."
        print(msg, file=sys.stderr)
        return 2


def _looks_like_dead_bridge(msg: str) -> bool:
    """True when *msg* points at a missing or unreachable bridge instance."""
    markers = (
        "No bridge instance found with id",
        "Failed to contact Binary Ninja bridge",
        "Timed out waiting for Binary Ninja bridge",
    )
    return any(marker in msg for marker in markers)
