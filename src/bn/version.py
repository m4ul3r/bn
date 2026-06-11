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
