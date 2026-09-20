from __future__ import annotations

import os
import shutil
import subprocess
import sys
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


def test_pyproject_declares_the_posix_only_platform():
    """#824: the package cannot run on Windows (fcntl locks, AF_UNIX transport)
    and its metadata said nothing about it -- the first symptom was a bare
    ModuleNotFoundError for fcntl out of `import bn.cli`."""
    import tomllib

    repo = Path(__file__).resolve().parents[1]
    data = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))

    classifiers = data["project"]["classifiers"]
    assert "Operating System :: POSIX" in classifiers
    assert "Operating System :: POSIX :: Linux" in classifiers


def test_a_missing_fcntl_names_the_posix_requirement_instead_of_failing_to_import():
    """The other half of #824 item 1, and the half that is actually a behaviour:
    the ENTRY GATE. On a non-POSIX interpreter `import bn.cli` died with a bare
    `ModuleNotFoundError: No module named 'fcntl'` raised out of a transitive
    import, which tells the caller nothing about why. The classifiers pinned
    above are metadata; this pins what README promises a user will see.

    Stripping both `try/except ImportError` gates left the whole suite green
    before this cell existed, so the gate could have rotted or been deleted
    silently. Setting a module to ``None`` in ``sys.modules`` is the documented
    way to make its import fail, and a child interpreter is the only way to
    reach the gate from a POSIX host, where the real import always succeeds.
    """
    probe = (
        "import sys\n"
        "sys.modules['fcntl'] = None\n"
        "try:\n"
        "    import bn.cli\n"
        "except RuntimeError as exc:\n"
        "    print('RuntimeError:', exc)\n"
        "except BaseException as exc:\n"
        "    print(type(exc).__name__ + ':', exc)\n"
        "else:\n"
        "    print('imported with no gate')\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe],
                          capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[1])

    assert proc.returncode == 0, proc.stderr
    # A named RuntimeError, not the bare ModuleNotFoundError that was the
    # reported first symptom, and not a silent success.
    assert proc.stdout.startswith("RuntimeError:"), proc.stdout
    assert "POSIX-only" in proc.stdout
    assert "ModuleNotFoundError" not in proc.stdout
