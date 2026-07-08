from __future__ import annotations

import hashlib
import importlib.util
import os
import platform
import re
import sys
from pathlib import Path


PLUGIN_NAME = "bn_agent_bridge"

# A bridge instance id is joined directly into filesystem paths
# (instances_dir()/<id>.{json,sock}). Path semantics make an unvalidated id a
# traversal primitive: "../evil" escapes instances_dir() and "/abs" replaces it
# entirely, placing a bridge's files outside the cache tree -- never listed by
# instance enumeration (which only globs instances_dir()/*.json) and impossible
# to stop normally (#84, #523). The CLI validates ids before they reach a
# bridge, but the bridge boundary (bn-agent --instance-id -> start_headless ->
# the path helpers below) had NO such check, so a raw/adversarial id reached the
# filesystem unsanitized. Enforce a strict basename grammar here, the shared
# chokepoint both processes route through. Kept dependency-light (stdlib only,
# ValueError) because paths.py is symlinked into the bridge package.
_INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_instance_id(instance_id: str) -> str:
    """Reject any instance id that isn't a safe path basename.

    Accepts letters, digits, '_', '-', '.'; rejects empty strings, '.'/'..',
    and anything containing a path separator or other character (which also
    rules out absolute paths and traversal). Raises ValueError before the id is
    joined into a filesystem path. Returns the id unchanged when valid. Mirrors
    the CLI's transport.validate_instance_id rules so both sides agree.
    """
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("Instance id must be a non-empty string")
    if instance_id in (".", "..") or not _INSTANCE_ID_RE.fullmatch(instance_id):
        raise ValueError(
            f"Invalid instance id: {instance_id!r}. Use only letters, digits, "
            "'_', '-', and '.' (no path separators, no '.'/'..', no absolute paths)."
        )
    return instance_id


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def claude_home() -> Path:
    env = os.environ.get("CLAUDE_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".claude"


def codex_home() -> Path:
    env = os.environ.get("CODEX_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".codex"


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


# -- project-local instance markers (#80) ---------------------------------
# A `.bn-<instance_id>` file dropped in a project root lets a bare `bn` command
# resolve the right bridge among many by walking up from cwd -- a visible,
# project-local pointer (vs the hashed sticky-pin). The registry stays the source
# of truth; the marker is just a pointer, validated against live instances on read.
MARKER_PREFIX = ".bn-"


def marker_name(instance_id: str) -> str:
    return f"{MARKER_PREFIX}{instance_id}"


def find_instance_markers(start: Path | None = None, *, max_depth: int = 40):
    """Yield ``(instance_id, path)`` for each ``.bn-<id>`` marker found walking up
    from *start* (default cwd), nearest first (#80)."""
    try:
        cwd = (start or Path.cwd()).resolve()
    except OSError:
        return
    for depth, d in enumerate((cwd, *cwd.parents)):
        if depth >= max_depth:
            break
        try:
            markers = sorted(d.glob(f"{MARKER_PREFIX}*"))
        except OSError:
            continue
        for m in markers:
            try:
                if m.is_file():
                    yield m.name[len(MARKER_PREFIX):], m
            except OSError:
                continue


def remove_instance_markers(instance_id: str, start: Path | None = None) -> list[Path]:
    """Delete every ``.bn-<instance_id>`` marker found walking up from *start*
    (default cwd) and return the paths removed (#436). Markers are not cleaned on
    ``session stop`` otherwise, so they accumulate in the repo root as stray VCS
    noise. Only markers whose id matches *instance_id* exactly are removed; an
    unreadable/undeletable marker is skipped rather than raised."""
    removed: list[Path] = []
    for found_id, path in find_instance_markers(start):
        if found_id != instance_id:
            continue
        try:
            path.unlink()
            removed.append(path)
        except OSError:
            continue
    return removed


def bridge_registry_path(instance_id: str | None = None) -> Path:
    if instance_id is None:
        return cache_home() / f"{PLUGIN_NAME}.json"
    # #523: validate at the path boundary so a traversal/absolute id can never be
    # joined into instances_dir(). None (legacy GUI fixed pair) is exempt above.
    return instances_dir() / f"{validate_instance_id(instance_id)}.json"


def bridge_socket_path(instance_id: str | None = None) -> Path:
    if instance_id is None:
        return cache_home() / f"{PLUGIN_NAME}.sock"
    # #523: see bridge_registry_path -- same basename enforcement.
    return instances_dir() / f"{validate_instance_id(instance_id)}.sock"


def spill_root() -> Path:
    # Lives under the per-user cache (not a shared /tmp dir) so multi-user
    # hosts don't hit permission collisions or leak decompiled artifacts
    # world-readably. Honors BN_CACHE_DIR via cache_home().
    root = cache_home() / "spills"
    root.mkdir(parents=True, exist_ok=True)
    return root


def taint_models_path() -> Path:
    """User-override file for taint function-models, merged over the builtin DB.

    The builtin models ship beside the bridge (``taint_models.json``); this
    optional file lets a user or agent add/override source/propagate/sink
    semantics without editing the plugin. Honoured by the taint engine when it
    exists; missing is fine.
    """
    env = os.environ.get("BN_TAINT_MODELS")
    if env:
        return Path(env).expanduser()
    return cache_home() / "taint_models.json"


def plugin_source_dir() -> Path:
    """Where ``bn plugin install`` copies/symlinks the bridge from.

    Editable checkout: the plugin package lives under src/. Wheel install:
    repo_root() points into site-packages and that path doesn't exist, so fall
    back to the bridge packaged into site-packages (importable as the
    PLUGIN_NAME module). repo_root() can't be trusted in a wheel (#83)."""
    repo_plugin = repo_root() / "src" / PLUGIN_NAME
    if repo_plugin.exists():
        return repo_plugin
    try:
        spec = importlib.util.find_spec(PLUGIN_NAME)
    except Exception:
        spec = None
    if spec is not None and spec.origin:
        return Path(spec.origin).resolve().parent
    return repo_plugin  # nonexistent -> _install_tree raises a clean error


def skills_source_dir() -> Path:
    """Where ``bn skill install`` reads the bundled skills from.

    Editable checkout: the repo's ``skills/`` dir. Wheel install: the skills are
    shipped as install-prefix data files, so they land directly under
    ``sys.prefix`` (the installer copies each skill dir there); the install
    handler filters to dirs containing ``SKILL.md`` (#83)."""
    repo_skills = repo_root() / "skills"
    if repo_skills.exists():
        return repo_skills
    return Path(sys.prefix)


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


def codex_skills_dir() -> Path:
    return codex_home() / "skills"
