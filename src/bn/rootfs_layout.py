"""Pure filesystem reasoning about an extracted firmware root filesystem.

Answers three questions without Binary Ninja: is this directory (or one of its
parents) a rootfs, where would its dynamic loader look for shared libraries, and
which real shared objects live on those paths. The CLI uses it to detect and
report; the bridge imports it (through the ``src/bn_agent_bridge`` symlink) to
sync libraries into a project. Stdlib only, no ``bn``/``bn_agent_bridge`` imports:
the module must import cleanly from both packages.

Every path is a ``str``. ``root`` is always canonical (``os.path.realpath``);
``path_in_root`` / ``rel_realpath`` are root-relative with a leading ``/``.
Symlinks below that root use firmware semantics: absolute targets restart at
the extraction root, and relative targets may not traverse above it. Trees
must remain unchanged during discovery; this is not a concurrent-mutation sandbox.
"""

from __future__ import annotations

import errno
import fnmatch
import hashlib
import math
import os
import re
import stat
import time
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps


def _positive_limit(name: str, default: int | float) -> int | float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw) if isinstance(default, int) else float(raw)
        if value <= 0 or not math.isfinite(value):
            raise ValueError
        return value
    except (ValueError, OverflowError) as exc:
        kind = "integer" if isinstance(default, int) else "number"
        raise RuntimeError(f"{name} must be a positive finite {kind}") from exc


_MIB = 1 << 20


@dataclass
class DiscoveryBudget:
    """One aggregate limit for lookup, hashing and ELF discovery.

    ``max_work`` (``BN_ROOTFS_MAX_WORK``) is spent in units of one per file
    opened -- a loader config read, a directory listed, a library hashed or
    parsed -- plus one per MiB read from the root. What is *inside* a file
    (directory entries, config lines, dynamic symbols) costs nothing: the old
    per-symbol charge spent ~460k units on a ~450-library firmware for three
    seconds of work (#1075), while the same tree is ~1k units here and the
    deadline bounds pathological trees. ``check()`` only probes -- cancellation,
    the deadline, and the limits as charged so far -- so it is cheap to call
    per entry or per symbol; ``charge()`` is what spends.
    """

    max_work: int = field(default_factory=lambda: _positive_limit("BN_ROOTFS_MAX_WORK", 100_000))
    max_bytes: int = field(default_factory=lambda: _positive_limit("BN_ROOTFS_MAX_BYTES", 1 << 30))
    timeout: float = field(default_factory=lambda: _positive_limit("BN_ROOTFS_TIMEOUT", 30.0))
    max_text_bytes: int = field(
        default_factory=lambda: _positive_limit("BN_ROOTFS_MAX_CONFIG_BYTES", 1 << 20))
    cancelled: Callable[[], bool] | None = None
    # The command that restarts the process which read these limits from its
    # environment (`bn session restart <id>`); None when there is none to name
    # -- the GUI plugin bridge, with no instance id, or a CLI-side caller with
    # no bridge to restart at all (#1096 A4).
    restart: str | None = None
    # A GUI-only fallback instruction, used only when `restart` is None and
    # there IS a process to restart (the GUI plugin bridge): a template with
    # `{variable}`/`{doubled}` placeholders, filled in at refusal time (#1096
    # A6a). `restart`'s own `{self.restart}` stays a literal command precisely
    # because every existing caller constructs it that way; this is a
    # separate field rather than a second meaning for the same one.
    gui_restart_hint: str | None = None
    notes: list[str] = field(default_factory=list)
    _files: int = 0
    _bytes: int = 0
    _started: float = field(default_factory=lambda: time.monotonic())

    def __post_init__(self) -> None:
        if type(self.max_work) is not int or self.max_work <= 0:
            raise ValueError("rootfs max_work must be a positive integer")
        if type(self.max_bytes) is not int or self.max_bytes <= 0:
            raise ValueError("rootfs max_bytes must be a positive integer")
        if type(self.max_text_bytes) is not int or self.max_text_bytes <= 0:
            raise ValueError("rootfs max_text_bytes must be a positive integer")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("rootfs timeout must be positive and finite")

    @property
    def work_used(self) -> int:
        return self._files + self._bytes // _MIB

    def charge(self, *, files: int = 0, bytes_read: int = 0) -> None:
        """Spend *files* file units and *bytes_read* bytes, then refuse if a
        limit is now exceeded."""
        self._files += files
        self._bytes += bytes_read
        self.check()

    def check(self) -> None:
        """Refuse on cancellation, the deadline, or a limit the work charged so
        far exceeds. Charges nothing."""
        reason = None
        # (variable, a doubled value) for a limit a restart can raise.
        knob: tuple[str, str] | None = None
        if self.cancelled is not None and self.cancelled():
            reason = "request cancelled"
        elif time.monotonic() - self._started >= self.timeout:
            reason = f"discovery deadline exceeded (BN_ROOTFS_TIMEOUT={self.timeout:g} seconds)"
            knob = ("BN_ROOTFS_TIMEOUT", f"{2 * self.timeout:g}")
        elif self.work_used > self.max_work:
            # Discovery stops at the first unit over the limit, so the limit is
            # the only number known: "more than N", never the tree's need.
            reason = (f"work limit exceeded (used more than {self.max_work} units, "
                      f"BN_ROOTFS_MAX_WORK={self.max_work}, charged per file opened "
                      "and per MiB read)")
            knob = ("BN_ROOTFS_MAX_WORK", str(2 * self.max_work))
        elif self._bytes > self.max_bytes:
            reason = (f"discovery byte limit exceeded (read more than {self.max_bytes} bytes, "
                      f"BN_ROOTFS_MAX_BYTES={self.max_bytes})")
            knob = ("BN_ROOTFS_MAX_BYTES", str(2 * self.max_bytes))
        if reason is None:
            return
        message = f"rootfs discovery refused: {reason}; no inventory committed"
        if knob is not None:
            variable, doubled = knob
            tail = None
            if self.restart:
                # The CLI's environment never reaches the bridge that does the
                # work; the knob only applies to a process started with it.
                tail = f"{variable}={doubled} {self.restart}"
            elif self.gui_restart_hint:
                tail = self.gui_restart_hint.format(variable=variable, doubled=doubled)
            if tail is not None:
                message += (f". The owning bridge reads {variable} at start, so set it and "
                            "restart; the total needed is not known, so doubling is only a "
                            f"starting point: {tail}")
        raise RuntimeError(message)


_discovery: ContextVar[DiscoveryBudget | None] = ContextVar("rootfs_discovery", default=None)


@contextmanager
def discovery_budget(budget: DiscoveryBudget | None = None):
    """Share limits across a complete sync; standalone calls are bounded too."""
    existing = _discovery.get()
    if existing is not None and budget is None:
        yield existing
        return
    budget = budget if budget is not None else DiscoveryBudget()
    token = _discovery.set(budget)
    try:
        budget.check()
        yield budget
    finally:
        _discovery.reset(token)


def _bounded(function):
    @wraps(function)
    def bounded(*args, **kwargs):
        with discovery_budget():
            return function(*args, **kwargs)
    return bounded


def _check_work() -> None:
    """Probe the shared budget (cancellation, deadline, limits); free."""
    budget = _discovery.get()
    if budget is not None:
        budget.check()


def _charge_work(*, files: int = 0, bytes_read: int = 0) -> None:
    """Spend file / byte units against the shared budget, then probe it."""
    budget = _discovery.get()
    if budget is not None:
        budget.charge(files=files, bytes_read=bytes_read)


def _invalid_path(rel: str) -> bool:
    if "\0" not in rel:
        return False
    budget = _discovery.get()
    if budget is not None and "skipped loader path containing NUL" not in budget.notes:
        budget.notes.append("skipped loader path containing NUL")
    return True


@dataclass(slots=True)
class RootfsInfo:
    root: str
    platform: str  # "linux" | "android" | "unknown"
    markers: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LookupPath:
    path_in_root: str
    source: str  # ld_library_path:<file> | ld_so_conf:<file> | default | android_default


@dataclass(slots=True)
class LibraryCandidate:
    realpath: str
    rel_realpath: str
    dir_in_root: str
    names: list[str]
    sha256: str
    size: int
    mtime: float
    paths: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------

_LINUX_LAYOUT = ("bin", "sbin", "lib", "lib64", "usr", "etc")
_LINUX_LOADER = (
    "etc/ld.so.conf",
    "etc/ld.so.cache",
    "lib/ld-*.so*",
    "lib*/ld-linux*.so*",
    "lib*/libc.so*",
    "usr/lib*/libc.so*",
    "lib/ld-musl-*.so*",
    "init",
    "linuxrc",
    "etc/inittab",
)

_LD_LIBRARY_PATH_RE = re.compile(r"LD_LIBRARY_PATH=[\"']?([^\"'\s;]+)")
_LD_LIBRARY_PATH_FILES = (
    "etc/profile",
    "etc/profile.d/*.sh",
    "etc/environment",
    "etc/init.d/*",
    "etc/rc*",
    "etc/rc.d/**",
)
_LINUX_DEFAULTS = (
    "/lib", "/usr/lib", "/lib64", "/usr/lib64", "/lib32", "/usr/lib32", "/usr/local/lib",
)
_MULTIARCH_GLOBS = ("lib/*-linux-*", "usr/lib/*-linux-*")
_ANDROID_DEFAULTS = (
    "/system/lib64", "/system/lib", "/vendor/lib64", "/vendor/lib",
    "/system/lib64/vndk*", "/system/lib/vndk*", "/apex/*/lib64", "/apex/*/lib",
    "/odm/lib64", "/odm/lib",
)
_ELF_MAGIC = b"\x7fELF"


def _resolve_in_root(root: str, rel: str) -> str:
    """Resolve an existing firmware path without letting the host follow links."""
    if _invalid_path(rel):
        raise OSError(errno.EINVAL, "embedded NUL in firmware path")
    pending = deque(rel.split("/"))
    parts: list[str] = []
    links = 0
    while pending:
        _check_work()
        part = pending.popleft()
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise OSError(errno.EXDEV, "resolves outside the root", rel)
            parts.pop()
            continue
        path = os.path.join(root, *parts, part)
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode):
            links += 1
            if links > 40:
                raise OSError(errno.ELOOP, "too many firmware symlinks", rel)
            target = os.readlink(path)
            if target.startswith("/"):
                parts.clear()
            pending.extendleft(reversed(target.split("/")))
            continue
        if pending and not stat.S_ISDIR(mode):
            raise OSError(errno.ENOTDIR, "not a directory", path)
        parts.append(part)
    return os.path.join(root, *parts)


def _resolved_path(root: str, rel: str) -> str | None:
    try:
        return _resolve_in_root(root, rel)
    except OSError:
        return None


def _is_dir_in_root(root: str, rel: str) -> bool:
    path = _resolved_path(root, rel)
    return path is not None and os.path.isdir(path)


@_bounded
def _glob_in_root(root: str, pattern: str, *, recursive: bool = False) -> list[str]:
    """Preserve loader aliases; memoize recursive scraping by resolved state."""
    if _invalid_path(pattern):
        return []
    components = [part for part in pattern.split("/") if part]
    hits: set[str] = set()
    stack: list[tuple[str, int]] = [("", 0)]
    visited: set[tuple[str, int]] = set()
    while stack:
        _check_work()
        rel, index = stack.pop()
        full = _resolved_path(root, rel)
        if full is None:
            continue
        if index == len(components):
            hits.add(rel or "/")
            continue
        if not os.path.isdir(full) or (recursive and (full, index) in visited):
            continue
        if recursive:
            visited.add((full, index))
        part = components[index]
        descend = recursive and part == "**"
        if descend:
            stack.append((rel, index + 1))
        elif not any(ch in part for ch in "*?["):
            stack.append((rel + "/" + part, index + 1))
            continue
        names: list[str] = []
        try:
            with os.scandir(full) as entries:
                _charge_work(files=1)
                for entry in entries:
                    _check_work()
                    if (not entry.name.startswith(".") or part.startswith(".")) and (
                        descend or fnmatch.fnmatchcase(entry.name, part)
                    ):
                        names.append(entry.name)
        except OSError:
            continue
        for name in reversed(sorted(names)):
            child = rel + "/" + name
            if descend:
                if index + 1 == len(components) and _resolved_path(root, child) is not None:
                    hits.add(child)
                stack.append((child, index))
            else:
                stack.append((child, index + 1))
    return sorted(hits)


def _markers_present(root: str, patterns: tuple[str, ...]) -> list[str]:
    return [pat for pat in patterns if _glob_in_root(root, pat)]


def _score(root: str) -> RootfsInfo | None:
    if _resolved_path(root, "system/build.prop") is not None:
        return RootfsInfo(root, "android", ["system/build.prop"])
    if _is_dir_in_root(root, "system/lib") and _is_dir_in_root(root, "system/bin"):
        return RootfsInfo(root, "android", ["system/lib", "system/bin"])
    layout = _markers_present(root, _LINUX_LAYOUT)
    if len(layout) < 2:
        return None
    loader = _markers_present(root, _LINUX_LOADER)
    if not loader:
        return None
    return RootfsInfo(root, "linux", layout + loader)


@_bounded
def detect_root(start: str, *, explicit: bool = False) -> RootfsInfo | None:
    """Find the rootfs containing ``start``.

    Walks up from ``start`` without adopting host ``/``, its direct children,
    or another mount of host ``/usr``. With ``explicit=True`` only ``start`` is
    scored (the caller named the root), and a directory matching no detector
    yields ``platform="unknown"``. A non-directory is ``None`` either way.
    """
    cur = os.path.realpath(os.path.expanduser(start))
    if not os.path.isdir(cur):
        return None
    if explicit:
        return _score(cur) or RootfsInfo(cur, "unknown", [])
    try:
        host_usr = os.stat("/usr")
        host_usr_id = (host_usr.st_dev, host_usr.st_ino)
    except OSError:
        host_usr_id = None
    parent = os.path.dirname(cur)
    while os.path.dirname(parent) != parent:
        _check_work()
        try:
            candidate = os.stat(cur)
            candidate_id = (candidate.st_dev, candidate.st_ino)
        except OSError:
            candidate_id = None
        if candidate_id is not None and candidate_id != host_usr_id:
            info = _score(cur)
            if info is not None:
                return info
        cur, parent = parent, os.path.dirname(parent)
    return None


def project_location(root: str, cache_projects_dir: str) -> tuple[str, str]:
    """``(preferred, fallback)`` project paths for ``root``.

    Preferred sits beside the root (``<parent>/<basename>.bnpr``); the fallback
    lives under ``cache_projects_dir`` keyed by a hash of the canonical root so
    read-only extraction trees still get a stable project.
    """
    root = os.path.realpath(root)
    base = os.path.basename(root) or "rootfs"
    preferred = os.path.join(os.path.dirname(root), base + ".bnpr")
    digest = hashlib.sha256(root.encode("utf-8", "surrogateescape")).hexdigest()[:16]
    return preferred, os.path.join(cache_projects_dir, digest + ".bnpr")


# ---------------------------------------------------------------------------
# lookup paths
# ---------------------------------------------------------------------------


@contextmanager
def _open_regular(path: str):
    # NONBLOCK closes the stat/open FIFO race; NOFOLLOW refuses replaced links.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "not a regular file", path)
        handle = os.fdopen(fd, "rb")
        fd = -1
        with handle:
            yield handle
    finally:
        if fd >= 0:
            os.close(fd)


def _read_text(root: str, rel: str) -> str:
    try:
        path = _resolve_in_root(root, rel)
        with _open_regular(path) as fh:
            budget = _discovery.get()
            limit = budget.max_text_bytes if budget is not None else 1 << 20
            if budget is not None:
                # Bound even the allocation by the remaining aggregate allowance.
                limit = min(limit, max(0, budget.max_bytes - budget._bytes))
            data = fh.read(limit + 1)
            _charge_work(files=1, bytes_read=len(data))
            if len(data) > limit:
                # Naming the variable (#1098 review): when the AGGREGATE
                # allowance is what bound `limit` above, `_charge_work` just
                # charged `limit + 1` bytes, pushing `_bytes` to
                # `max_bytes + 1` -- `check()` inside it always raises first,
                # already correctly naming `BN_ROOTFS_MAX_BYTES`. So the only
                # way execution reaches this line at all is the OTHER case:
                # the per-config-file limit bound, and it is what to name.
                detail = (f" (BN_ROOTFS_MAX_CONFIG_BYTES={budget.max_text_bytes})"
                         if budget is not None else "")
                raise RuntimeError(
                    f"rootfs discovery refused: loader config {rel} exceeds the text "
                    f"limit{detail}; no inventory committed")
            return data.decode("utf-8", "replace")
    except OSError:
        return ""


def _scrape_files(root: str) -> list[str]:
    """Root-relative regular files that may export ``LD_LIBRARY_PATH``, loader-check order."""
    out: list[str] = []
    seen: set[str] = set()
    for pat in _LD_LIBRARY_PATH_FILES:
        recursive = "**" in pat
        for rel in _glob_in_root(root, pat, recursive=recursive):
            full = _resolved_path(root, rel)
            if full is None or full in seen or not os.path.isfile(full):
                continue
            seen.add(full)
            out.append(rel)
    return out


@_bounded
def env_hint(root: str) -> list[tuple[str, str]]:
    """Every ``LD_LIBRARY_PATH=`` assignment found under ``etc/`` as ``(file, value)``.

    ``file`` is root-relative with a leading ``/``; ``value`` is the raw matched
    text (unsplit, ``$LD_LIBRARY_PATH`` references intact) for the dry-run report.
    """
    root = os.path.realpath(root)
    hits: list[tuple[str, str]] = []
    for rel in _scrape_files(root):
        for m in _LD_LIBRARY_PATH_RE.finditer(_read_text(root, rel)):
            hits.append((rel, m.group(1)))
    return hits


def _ld_library_path_entries(root: str) -> list[tuple[str, str]]:
    """``(path, file)`` pairs: absolute entries only, ``$LD_LIBRARY_PATH`` dropped."""
    out: list[tuple[str, str]] = []
    for rel, value in env_hint(root):
        for entry in value.split(":"):
            if not entry or entry.startswith("$") or not entry.startswith("/"):
                continue
            out.append((entry, rel))
    return out


def _ld_so_conf_entries(root: str, rel_conf: str, visited: set[str],
                        out: list[tuple[str, str]]) -> None:
    pending: list[tuple[str, str | None]] = [(rel_conf, None)]
    while pending:
        _check_work()
        conf, raw = pending.pop()
        if raw is None:
            full = _resolved_path(root, conf)
            if full is None or full in visited or not os.path.isfile(full):
                continue
            visited.add(full)
            pending.extend((conf, line) for line in reversed(_read_text(root, conf).splitlines()))
            continue
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("include") and (len(line) == 7 or line[7].isspace()):
            includes: list[str] = []
            for pattern in line[7:].split():
                if not pattern.startswith("/"):
                    pattern = os.path.join(os.path.dirname(conf), pattern)
                includes.extend(_glob_in_root(root, pattern))
            pending.extend((rel, None) for rel in reversed(includes))
            continue
        if line.startswith("hwcap"):
            continue
        for entry in re.split(r"[\s:,]+", line):
            if entry.startswith("/") and not _invalid_path(entry):
                out.append((entry, conf))


def _linux_defaults(root: str) -> list[str]:
    paths = list(_LINUX_DEFAULTS)
    for pat in _MULTIARCH_GLOBS:
        paths.extend(_glob_in_root(root, pat))
    return paths


def _android_defaults(root: str) -> list[str]:
    paths: list[str] = []
    for pat in _ANDROID_DEFAULTS:
        if "*" in pat:
            paths.extend(_glob_in_root(root, pat))
        else:
            paths.append(pat)
    return paths


@_bounded
def lookup_paths(root: str, platform: str) -> list[LookupPath]:
    """Directories the loader of ``platform`` searches, in loader order.

    Linux: ``LD_LIBRARY_PATH`` scraped from startup files, then ``etc/ld.so.conf``
    (with ``include`` recursion), then the compiled-in defaults plus existing
    multiarch dirs. Android: the Android default set, then the Linux defaults.
    ``"unknown"`` yields ``[]``. Only existing directories are returned, each once
    (first occurrence wins), spelled as the loader would see them; symlinked
    directories are followed for the existence check only.
    """
    root = os.path.realpath(root)
    ordered: list[tuple[str, str]] = []
    if platform == "linux":
        ordered.extend(_ld_library_path_entries(root))
        confs: list[tuple[str, str]] = []
        _ld_so_conf_entries(root, "/etc/ld.so.conf", set(), confs)
        ordered.extend(confs)
        ordered.extend((p, "") for p in _linux_defaults(root))
    elif platform == "android":
        ordered.extend((p, "android_default") for p in _android_defaults(root))
        ordered.extend((p, "") for p in _linux_defaults(root))
    else:
        return []

    out: list[LookupPath] = []
    seen: set[str] = set()
    for raw, origin in ordered:
        rel = "/" + raw.strip("/")
        if rel == "/" or rel in seen:
            continue
        seen.add(rel)
        if not _is_dir_in_root(root, rel):
            continue
        if origin == "":
            source = "default"
        elif origin == "android_default":
            source = origin
        elif origin.startswith("/etc/ld.so.conf"):
            source = "ld_so_conf:" + origin
        else:
            source = "ld_library_path:" + origin
        out.append(LookupPath(rel, source))
    return out


# ---------------------------------------------------------------------------
# library scan
# ---------------------------------------------------------------------------


def _is_so_name(name: str) -> bool:
    return name.endswith(".so") or ".so." in name


def _library_digest(path: str) -> tuple[os.stat_result, str] | None:
    with _open_regular(path) as fh:
        st = os.fstat(fh.fileno())
        magic = fh.read(4)
        _charge_work(files=1, bytes_read=len(magic))
        if magic != _ELF_MAGIC:
            return None
        h = hashlib.sha256(magic)
        while chunk := fh.read(_MIB):
            _charge_work(bytes_read=len(chunk))
            h.update(chunk)
        return st, h.hexdigest()


@_bounded
def scan_libraries(root: str, paths: list[LookupPath]) -> tuple[list[LibraryCandidate], list[str]]:
    """Shared objects on ``paths`` (non-recursive), one candidate per real file.

    A candidate is a regular file (or symlink to one) named ``*.so`` / ``*.so.*``
    starting with ``\\x7fELF``. Entries resolving to the same firmware file
    merge: ``names`` holds the real basename first, then every other spelling
    sorted. ``paths`` holds discovered full root-relative loader spellings,
    deduplicated in lookup order (sorted names within each directory), without
    inventing a spelling for a target outside the searched directories.
    Candidates keep first-seen order. Returns ``(candidates, notes)``;
    notes explain skipped entries (relative root escapes, unreadable files).
    """
    root = os.path.realpath(root)
    prefix = root.rstrip(os.sep) + os.sep
    by_real: dict[str, tuple[LibraryCandidate, set[str]]] = {}
    notes: list[str] = []
    for lp in paths:
        _check_work()
        try:
            dir_full = _resolve_in_root(root, lp.path_in_root)
            with os.scandir(dir_full) as directory:
                _charge_work(files=1)
                entries = []
                for entry in directory:
                    _check_work()
                    entries.append(entry)
                entries.sort(key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
            _check_work()
            if not _is_so_name(entry.name):
                continue
            spelling = lp.path_in_root.rstrip("/") + "/" + entry.name
            try:
                real = _resolve_in_root(root, spelling)
                if not stat.S_ISREG(os.lstat(real).st_mode):
                    continue
            except OSError as exc:
                if exc.errno == errno.EXDEV:
                    notes.append(f"skipped {spelling}: resolves outside the root")
                continue
            if real in by_real:
                cand, spellings = by_real[real]
                if spelling not in spellings:
                    spellings.add(spelling)
                    cand.paths.append(spelling)
                continue
            try:
                result = _library_digest(real)
                if result is None:
                    continue
                st, digest = result
            except OSError as exc:
                notes.append(f"skipped {spelling}: {exc.strerror or exc}")
                continue
            rel_real = "/" + real[len(prefix):].replace(os.sep, "/")
            cand = LibraryCandidate(
                realpath=real,
                rel_realpath=rel_real,
                dir_in_root=os.path.dirname(rel_real),
                names=[],
                sha256=digest,
                size=st.st_size,
                mtime=st.st_mtime,
                paths=[spelling],
            )
            by_real[real] = (cand, {spelling})
    out: list[LibraryCandidate] = []
    for cand, spellings in by_real.values():
        real_base = os.path.basename(cand.realpath)
        names = {os.path.basename(path) for path in spellings}
        cand.names = [real_base] + sorted(names - {real_base})
        out.append(cand)
    return out, notes
