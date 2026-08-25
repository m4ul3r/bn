"""Idempotently restore the bn_kernel import after an OMP eval-kernel reset."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import shutil
import sys
from pathlib import Path

_configured_skill_dir = globals().get("skill_dir")
if _configured_skill_dir is None:
    _bootstrap_file = globals().get("__file__")
    if not _bootstrap_file:
        raise RuntimeError(
            "bn_kernel bootstrap cannot locate its skill directory; set skill_dir "
            "to the absolute installed bn-kernel skill path before exec"
        )
    _skill_dir = Path(_bootstrap_file).resolve().parent
else:
    _skill_dir = Path(_configured_skill_dir).expanduser().resolve()

_source_path = _skill_dir / "src" / "bn_kernel" / "__init__.py"
_source_hash = hashlib.sha256(_source_path.read_bytes()).hexdigest()
_source_dir = str(_skill_dir / "src")
_source_resolved = _source_path.resolve()

_clean_path = []
for _entry in sys.path:
    try:
        _candidate = (Path(_entry) / "bn_kernel" / "__init__.py").resolve()
    except (OSError, RuntimeError):
        _clean_path.append(_entry)
        continue
    if _candidate.exists() and _candidate != _source_resolved:
        continue
    if str(Path(_entry).resolve()) != str(Path(_source_dir).resolve()):
        _clean_path.append(_entry)
sys.path[:] = [_source_dir, *_clean_path]


def _module_matches_source(module) -> bool:
    if module is None:
        return False
    module_path = Path(str(getattr(module, "__file__", ""))).resolve()
    if module_path != _source_path.resolve():
        return False
    if getattr(module, "__bn_kernel_source_hash__", None) != _source_hash:
        return False
    try:
        disasm_parameters = inspect.signature(module.Session.disasm).parameters
        scoped_parameters = inspect.signature(module.scoped).parameters
    except (AttributeError, TypeError, ValueError):
        return False
    return (
        {"count", "lines"}.issubset(disasm_parameters)
        and "callback" in scoped_parameters
    )


_bootstrap_action = "reused"


bn_kernel = sys.modules.get("bn_kernel")
if not _module_matches_source(bn_kernel):
    for _module_name in tuple(sys.modules):
        if _module_name == "bn_kernel" or _module_name.startswith("bn_kernel."):
            sys.modules.pop(_module_name, None)
    _cache_dir = _source_path.parent / "__pycache__"
    if _cache_dir.exists():
        shutil.rmtree(_cache_dir)
    importlib.invalidate_caches()

    _previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        bn_kernel = importlib.import_module("bn_kernel")
    finally:
        sys.dont_write_bytecode = _previous_dont_write_bytecode
    bn_kernel.__bn_kernel_source_hash__ = _source_hash
    _bootstrap_action = "reloaded"

if not _module_matches_source(bn_kernel):
    raise RuntimeError(
        "bn_kernel bootstrap loaded a stale or incompatible module; restart the "
        "eval kernel before continuing"
    )

print(
    "bn_kernel bootstrap: "
    f"{_bootstrap_action}; source={_source_path}; sha256={_source_hash[:12]}"
)
