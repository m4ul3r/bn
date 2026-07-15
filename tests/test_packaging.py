from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest


def test_wheel_excludes_python_bytecode_from_bridge_package(tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build the wheel")

    repo = Path(__file__).resolve().parents[1]
    tree = tmp_path / "tree"
    tree.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(repo / name, tree / name)
    for name in ("src", "skills"):
        shutil.copytree(repo / name, tree / name, symlinks=True)

    pycache = tree / "src" / "bn_agent_bridge" / "__pycache__"
    # exist_ok: copytree above may have already carried a __pycache__ from the
    # repo's working tree (present after any local test run), so don't assume the
    # copied tree is bytecode-free -- just ensure our sentinel is in it.
    pycache.mkdir(exist_ok=True)
    (pycache / "sentinel.cpython-314.pyc").write_bytes(b"not real bytecode")
    (tree / "src" / "bn_agent_bridge" / "sentinel.pyo").write_bytes(b"not real bytecode")

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
    assert leaked == []
    assert any(name == "bn_agent_bridge/plugin.json" for name in names)
