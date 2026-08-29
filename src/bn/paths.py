from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import platform
import re
import stat
import sys
from pathlib import Path


_log = logging.getLogger("bn.paths")


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


def ensure_private_dir(path: Path) -> Path:
    """Create *path* (and any missing parents) owned-private to this user.

    The leaf directory is created with ``mode=0o700`` and an already-existing
    directory is tightened to ``0o700`` (compared via ``stat.S_IMODE``). This is
    the single chokepoint for cache / instances / sessions / spills directory
    creation (#612): agent artifacts -- decompiled output spills, bridge
    registries, sticky session pins -- must not land group/world-readable under
    a permissive ``umask``.

    Mode enforcement is best-effort-with-warning: platforms that ignore Unix
    permission bits (e.g. Windows) simply keep whatever the OS grants, and an
    ``OSError`` from the ``chmod`` (unsupported filesystem, foreign-owned dir)
    does NOT fail directory setup -- but it is no longer swallowed silently. On
    such a filesystem the directory can stay group/world-accessible, so the
    failure is surfaced via a clear warning naming the path (#612 follow-up):
    silently failing open would contradict the documented mode enforcement and
    is invisible where these modes are the primary cross-user boundary (non-Linux,
    where the socket peer-credential check is skipped). The ``mkdir`` itself is
    NOT swallowed -- a genuine failure (permission denied, race) still raises, as
    the previous bare ``mkdir(parents=True, exist_ok=True)`` calls did. Returns
    *path* so call sites can chain (``ensure_private_dir(d) / "file"``).
    """
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        if stat.S_IMODE(path.stat().st_mode) != 0o700:
            os.chmod(path, 0o700)
    except OSError as exc:
        _log.warning(
            "could not tighten permissions on %s to 0o700 (%s); it may remain "
            "group/world-accessible", path, exc,
        )
    return path


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


def omp_config_root() -> Path:
    config_dir = os.environ.get("PI_CONFIG_DIR") or ".omp"
    return Path.home() / config_dir


def omp_agent_dir() -> Path:
    if "OMP_PROFILE" in os.environ:
        profile = os.environ["OMP_PROFILE"].strip()
    else:
        profile = os.environ.get("PI_PROFILE", "").strip()

    config_root = omp_config_root()
    if profile and profile != "default":
        return config_root / "profiles" / profile / "agent"

    configured_agent = os.environ.get("PI_CODING_AGENT_DIR")
    if configured_agent:
        return Path(configured_agent).expanduser()
    return config_root / "agent"


def omp_skills_dir() -> Path:
    return omp_agent_dir() / "skills"


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


# A Unix-domain socket address is a fixed-size `sockaddr_un.sun_path` char array
# -- 108 bytes on Linux, 104 on the BSDs/macOS -- and bind() needs room for the
# trailing NUL, so the usable path is one byte shorter. Nothing bounded this, so
# a deep BN_CACHE_DIR plus a descriptive instance id produced a path the kernel
# rejects, surfacing as a bare `OSError: AF_UNIX path too long` from bind() deep
# inside bridge startup -- after the CLI had already committed to the id.
SOCKET_PATH_MAX_BYTES = 104 if platform.system() in ("Darwin", "FreeBSD", "OpenBSD", "NetBSD") else 108


def socket_path_budget() -> int:
    """Longest socket path, in bytes, that ``bind()`` will accept here."""
    return SOCKET_PATH_MAX_BYTES - 1


def _socket_path_too_long(path: Path) -> bool:
    return len(str(path).encode("utf-8", "surrogateescape")) > socket_path_budget()


def socket_too_long_message(path: Path, *, instance_id: str | None) -> str:
    actual = len(str(path).encode("utf-8", "surrogateescape"))
    hint = (
        "Point BN_CACHE_DIR at a shorter directory"
        if instance_id is None
        else "Use a shorter --instance-id, or point BN_CACHE_DIR at a shorter directory"
    )
    return (
        f"Unix socket path is too long: {actual} bytes, limit "
        f"{socket_path_budget()} ({path}). {hint}."
    )


def bridge_socket_path(instance_id: str | None = None) -> Path:
    """The bridge's socket path, or ValueError if it exceeds the kernel limit.

    The socket basename stays byte-identical to the instance id. An earlier
    revision compacted an over-budget name to ``<prefix>.<sha256-8>.sock``, but
    that silently breaks `bn instance gc`: it reverse-maps sockets to instances
    by comparing a socket's stem against the live registry stems (which keep the
    full id), so a compacted socket reads as a registry-less orphan and gets
    unlinked out from under a RUNNING bridge -- after which the missing socket
    makes the liveness sweep purge the registry too, orphaning the process and
    its BNDB. Refusing an impossible path is strictly better than inventing a
    reachable one that other tooling cannot recognize.
    """
    if instance_id is None:
        path = cache_home() / f"{PLUGIN_NAME}.sock"
    else:
        # #523: see bridge_registry_path -- same basename enforcement.
        path = instances_dir() / f"{validate_instance_id(instance_id)}.sock"
    if _socket_path_too_long(path):
        raise ValueError(socket_too_long_message(path, instance_id=instance_id))
    return path


def spill_root() -> Path:
    # Lives under the per-user cache (not a shared /tmp dir) so multi-user
    # hosts don't hit permission collisions or leak decompiled artifacts
    # world-readably. Honors BN_CACHE_DIR via cache_home().
    return ensure_private_dir(cache_home() / "spills")


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
