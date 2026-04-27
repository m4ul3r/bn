from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .paths import project_root, session_state_path, sessions_dir

KEYS = ("instance_id", "target")


def read() -> dict[str, Any]:
    """Return the current session state, or empty dict on missing/malformed."""
    path = session_state_path()
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def update(**fields: Any) -> dict[str, Any]:
    """Merge *fields* into on-disk state. ``None`` removes a key."""
    state = read()
    for key, value in fields.items():
        if value is None:
            state.pop(key, None)
        else:
            state[key] = value
    state["project_root"] = str(project_root())
    _atomic_write(state)
    return state


def _atomic_write(state: dict[str, Any]) -> None:
    path = session_state_path()
    sessions_dir().mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(state, fh)
        Path(tmp).replace(path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
