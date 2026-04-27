from __future__ import annotations

import hashlib
import os
import platform
import tempfile
from pathlib import Path


PLUGIN_NAME = "bn_agent_bridge"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def claude_home() -> Path:
    env = os.environ.get("CLAUDE_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".claude"


def cache_home() -> Path:
    env = os.environ.get("BN_CACHE_DIR")
    if env:
        return Path(env).expanduser()

    system = platform.system()
    home = Path.home()
    if system == "Darwin":
        return home / "Library" / "Caches" / "bn"
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "bn"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "bn"
    return home / ".cache" / "bn"


def instances_dir() -> Path:
    return cache_home() / "instances"


def sessions_dir() -> Path:
    return cache_home() / "sessions"


def project_root(start: Path | None = None) -> Path:
    """Walk up from *start* (default: cwd) looking for a `.git` ancestor.

    Falls back to the resolved start directory when no marker is found, so
    sticky state still has a stable key in non-git checkouts.
    """
    cwd = (start or Path.cwd()).resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / ".git").exists():
            return candidate
    return cwd


def session_state_path(start: Path | None = None) -> Path:
    root = project_root(start)
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
    return sessions_dir() / f"{digest}.json"


def bridge_registry_path(instance_id: str | None = None) -> Path:
    if instance_id is None:
        return cache_home() / f"{PLUGIN_NAME}.json"
    return instances_dir() / f"{instance_id}.json"


def bridge_socket_path(instance_id: str | None = None) -> Path:
    if instance_id is None:
        return cache_home() / f"{PLUGIN_NAME}.sock"
    return instances_dir() / f"{instance_id}.sock"


def spill_root() -> Path:
    root = Path(tempfile.gettempdir()) / "bn-spills"
    root.mkdir(parents=True, exist_ok=True)
    return root


def plugin_source_dir() -> Path:
    return repo_root() / "plugin" / PLUGIN_NAME


def binary_ninja_plugin_dir() -> Path:
    env = os.environ.get("BN_PLUGIN_DIR")
    if env:
        return Path(env).expanduser()

    system = platform.system()
    home = Path.home()
    if system == "Darwin":
        return home / "Library" / "Application Support" / "Binary Ninja" / "plugins"
    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Binary Ninja" / "plugins"
    return home / ".binaryninja" / "plugins"


def plugin_install_dir() -> Path:
    return binary_ninja_plugin_dir() / PLUGIN_NAME


def claude_skills_dir() -> Path:
    return claude_home() / "skills"


