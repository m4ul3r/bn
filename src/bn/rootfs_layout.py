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
import os
import re
import stat
from collections import deque
from dataclasses import dataclass, field


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

# Text files are read whole for scraping; cap so a stray blob on a rootfs cannot
# pull gigabytes into memory.
_MAX_TEXT_READ = 1 << 20
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
    pending = deque(rel.split("/"))
    parts: list[str] = []
    links = 0
    while pending:
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


def _glob_in_root(root: str, pattern: str, *, recursive: bool = False) -> list[str]:
    """Expand firmware paths, retaining spelling and bounding recursive link loops."""
    components = [part for part in pattern.split("/") if part]
    hits: set[str] = set()
    stack: list[tuple[str, int, frozenset[str]]] = [("", 0, frozenset())]
    while stack:
        rel, index, ancestors = stack.pop()
        full = _resolved_path(root, rel)
        if full is None:
            continue
        if index == len(components):
            hits.add(rel or "/")
            continue
        if not os.path.isdir(full):
            continue
        part = components[index]
        descend = recursive and part == "**"
        if descend:
            if full in ancestors:
                continue
            stack.append((rel, index + 1, frozenset()))
            ancestors = ancestors | {full}
        elif not any(ch in part for ch in "*?["):
            stack.append((rel + "/" + part, index + 1, ancestors))
            continue
        try:
            with os.scandir(full) as entries:
                names = [entry.name for entry in entries
                         if (not entry.name.startswith(".") or part.startswith("."))
                         and (descend or fnmatch.fnmatchcase(entry.name, part))]
        except OSError:
            continue
        for name in names:
            child = rel + "/" + name
            if descend:
                # Files are also terminal matches for a trailing **.
                if index + 1 == len(components):
                    if _resolved_path(root, child) is not None:
                        hits.add(child)
                stack.append((child, index, ancestors))
            else:
                stack.append((child, index + 1, ancestors))
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


def detect_root(start: str, *, explicit: bool = False) -> RootfsInfo | None:
    """Find the rootfs containing ``start``.

    Walks up from ``start`` but never implicitly scores host ``/``. With
    ``explicit=True`` only ``start`` itself is scored (the caller named the root),
    and a directory matching no detector yields ``platform="unknown"`` instead of
    ``None``. A non-directory ``start`` is ``None`` either way.
    """
    cur = os.path.realpath(os.path.expanduser(start))
    if not os.path.isdir(cur):
        return None
    if explicit:
        return _score(cur) or RootfsInfo(cur, "unknown", [])
    while os.path.dirname(cur) != cur:
        info = _score(cur)
        if info is not None:
            return info
        parent = os.path.dirname(cur)
        cur = parent
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


def _read_text(root: str, rel: str) -> str:
    try:
        path = _resolve_in_root(root, rel)
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return ""
        with open(path, "rb") as fh:
            return fh.read(_MAX_TEXT_READ).decode("utf-8", "replace")
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
            if rel in seen or full is None or not os.path.isfile(full):
                continue
            seen.add(rel)
            out.append(rel)
    return out


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
    full = _resolved_path(root, rel_conf)
    if full is None or full in visited or not os.path.isfile(full):
        return
    visited.add(full)
    conf_dir = os.path.dirname(rel_conf)
    for raw in _read_text(root, rel_conf).splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("include") and (len(line) == 7 or line[7].isspace()):
            for pattern in line[7:].split():
                if not pattern.startswith("/"):
                    pattern = os.path.join(conf_dir, pattern)
                for rel in _glob_in_root(root, pattern):
                    _ld_so_conf_entries(root, rel, visited, out)
            continue
        if line.startswith("hwcap"):
            continue
        for entry in re.split(r"[\s:,]+", line):
            if entry.startswith("/"):
                out.append((entry, rel_conf))


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


def _is_elf(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == _ELF_MAGIC
    except OSError:
        return False


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


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
        try:
            dir_full = _resolve_in_root(root, lp.path_in_root)
            with os.scandir(dir_full) as directory:
                entries = sorted(directory, key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
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
            if not _is_elf(real):
                continue
            try:
                st = os.stat(real)
                digest = _sha256(real)
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
