from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest


def test_wheel_excludes_python_bytecode_from_bridge_package(tmp_path):
    """No bytecode in the wheel -- from the module trees OR the skills data tree.

    `skills/` ships as install-prefix data (`tool.uv.build-backend.data`), which is
    a different inclusion path from the `src/` modules, so it needs its own
    sentinels: a config that filters the module trees can still ship
    `<name>-<version>.data/data/bn-kernel/__pycache__/...`.

    The copy deliberately ignores `__pycache__` so the tree starts bytecode-free
    and every leak is attributable to a sentinel planted below, not to whatever a
    local test run happened to leave in the worktree.
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build the wheel")

    repo = Path(__file__).resolve().parents[1]
    tree = tmp_path / "tree"
    tree.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(repo / name, tree / name)
    for name in ("src", "skills"):
        shutil.copytree(
            repo / name,
            tree / name,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

    sentinel = b"not real bytecode"
    bytecode_dirs = (
        # module tree (src-layout package)
        tree / "src" / "bn_agent_bridge" / "__pycache__",
        # data tree: the skill root, where `bootstrap.py` sits next to SKILL.md
        tree / "skills" / "bn-kernel" / "__pycache__",
        # data tree, nested: the importable kernel source inside the skill
        tree / "skills" / "bn-kernel" / "src" / "bn_kernel" / "__pycache__",
    )
    for directory in bytecode_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "bootstrap.cpython-314.pyc").write_bytes(sentinel)
    loose_bytecode = (
        tree / "src" / "bn_agent_bridge" / "sentinel.pyo",
        tree / "skills" / "bn-kernel" / "sentinel.pyo",
        tree / "skills" / "bn-kernel" / "sentinel.pyc",
    )
    for path in loose_bytecode:
        path.write_bytes(sentinel)

    out_dir = tmp_path / "dist"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(out_dir), str(tree)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    wheels = sorted(out_dir.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()

    leaked = [
        name for name in names
        if "/__pycache__/" in name or name.endswith((".pyc", ".pyo"))
    ]
    assert leaked == [], leaked

    # The exclude must not pass by nuking the trees it is filtering.
    assert any(name == "bn_agent_bridge/plugin.json" for name in names)
    expected_data = {
        "bn-kernel/SKILL.md",
        "bn-kernel/bootstrap.py",
        "bn-kernel/scripts/smoke.py",
        "bn-kernel/src/bn_kernel/__init__.py",
    }
    shipped_data = {
        name.split(".data/data/", 1)[1]
        for name in names
        if ".data/data/" in name
    }
    assert expected_data <= shipped_data, expected_data - shipped_data
