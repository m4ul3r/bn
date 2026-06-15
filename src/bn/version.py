from __future__ import annotations

import hashlib
from pathlib import Path


def _resolve_version() -> str:
    """Resolve the bn version from a single canonical source.

    The literal lives in ``pyproject.toml`` only -- the ``uv_build`` backend
    requires a static ``project.version`` there and cannot read it from a
    Python attribute, so pyproject is necessarily the build's source of truth.
    This module therefore *derives* the value instead of duplicating it:

    1. From the repo's ``pyproject.toml`` when running from a checkout. This is
       reached relative to this file (``parents[2]/pyproject.toml``); ``resolve()``
       follows the bridge's symlinked ``version.py``, so it works for ``uv run``,
       the headless agent, and the GUI plugin symlinked into Binary Ninja's
       plugins dir -- the contexts where the ``bn`` distribution may not be
       importable via ``importlib.metadata``.
    2. From installed distribution metadata (``importlib.metadata``) for a wheel
       install, where ``pyproject.toml`` is not shipped but the dist is present.

    Both paths reflect the same canonical literal, so CLI and bridge always
    agree (keeping ``bn doctor``'s ``stale_plugin_version`` check honest).
    """
    try:
        import tomllib

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        version = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
        if isinstance(version, str) and version:
            return version
    except Exception:
        pass

    try:
        from importlib.metadata import PackageNotFoundError, version as _dist_version

        try:
            return _dist_version("bn")
        except PackageNotFoundError:
            pass
    except Exception:
        pass

    return "0+unknown"


VERSION = _resolve_version()


def build_id_for_file(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()[:12]


def build_id_for_package(directory: Path) -> str | None:
    """Combined fingerprint of a bridge package: the sha256 of every ``*.py``
    source plus the model DB (``*.json``) in *directory*, hashed in sorted order
    (name + bytes). The single-file :func:`build_id_for_file` on ``bridge.py``
    misses edits to sibling modules, so a live session running an edited
    ``taint_engine.py`` looked fresh; this whole-package hash lets ``doctor`` flag
    a stale *engine* (#161). Returns None if the directory can't be read."""
    try:
        files = sorted(
            list(directory.glob("*.py")) + list(directory.glob("*.json")),
            key=lambda p: p.name,
        )
    except OSError:
        return None
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        try:
            data = path.read_bytes()
        except OSError:
            return None
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()[:12]
