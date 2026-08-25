"""Idempotently restore the bn_kernel import after an OMP eval-kernel reset."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import shutil
import sys
from pathlib import Path

_configured_skill_dir = globals().get("skill_dir")
_skill_dir = (
    Path(_configured_skill_dir)
    if _configured_skill_dir is not None
    else Path(__file__).resolve().parent
)
_source_path = _skill_dir / "src" / "bn_kernel" / "__init__.py"
_source_hash = hashlib.sha256(_source_path.read_bytes()).hexdigest()
_source_dir = str(_skill_dir / "src")
if _source_dir not in sys.path:
    sys.path.insert(0, _source_dir)


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
