from __future__ import annotations

import configparser
import importlib
import os
import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _stage_build_tree(tree: Path) -> Path:
    """The files a wheel build reads, copied out of the working tree.

    The copy deliberately ignores `__pycache__` so the tree starts bytecode-free
    and every leak is attributable to a sentinel planted by a test, not to
    whatever a local test run happened to leave behind. `symlinks=True` keeps the
    bridge's shared modules as the symlinks they are in the repo, which is what
    the wheel-fidelity pin below checks the build against.
    """
    tree.mkdir(parents=True, exist_ok=True)
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(REPO / name, tree / name)
    for name in ("src", "skills"):
        shutil.copytree(
            REPO / name,
            tree / name,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
    return tree


def _build_wheel(tree: Path, out_dir: Path) -> Path:
    """`uv build --wheel` over *tree*, returning the single wheel it produced."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build the wheel")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(out_dir), str(tree)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    wheels = sorted(out_dir.glob("*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0]


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    """One plain build of this tree, shared by the wheel-content pins."""
    root = tmp_path_factory.mktemp("wheels")
    return _build_wheel(_stage_build_tree(root / "tree"), root / "dist")


def test_wheel_excludes_python_bytecode_from_bridge_package(tmp_path):
    """No bytecode in the wheel -- from the module trees OR the skills data tree.

    `skills/` ships as install-prefix data (`tool.uv.build-backend.data`), which is
    a different inclusion path from the `src/` modules, so it needs its own
    sentinels: a config that filters the module trees can still ship
    `<name>-<version>.data/data/bn-kernel/__pycache__/...`.
    """
    tree = _stage_build_tree(tmp_path / "tree")
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

    built = _build_wheel(tree, tmp_path / "dist")
    with zipfile.ZipFile(built) as archive:
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


def _declared_scripts() -> dict[str, str]:
    """`[project.scripts]` -- the console scripts the project declares."""
    with open(REPO / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


def _console_scripts(wheel: Path) -> dict[str, str]:
    """`[console_scripts]` as the wheel records it; `{}` if it ships none."""
    with zipfile.ZipFile(wheel) as archive:
        members = [
            name for name in archive.namelist()
            if name.endswith(".dist-info/entry_points.txt")
        ]
        assert len(members) <= 1, members
        if not members:
            return {}
        parser = configparser.ConfigParser()
        parser.read_string(archive.read(members[0]).decode("utf-8"))
    return dict(parser["console_scripts"])


def _entry_point_drift(wheel: Path) -> list[str]:
    """Ways the wheel's console scripts differ from the declared ones."""
    shipped = _console_scripts(wheel)
    return [
        f"{name}: the wheel has {shipped.get(name)!r}, pyproject declares {target!r}"
        for name, target in sorted(_declared_scripts().items())
        if shipped.get(name) != target
    ]


# The commands a user gets on PATH from the wheel. Pinned by NAME, because that
# is the contract every agent-facing doc is written against: `bn` and `bn-agent`
# are what README, CLAUDE.md and the skills tell an agent to run. The comparison
# below cannot see a rename (pyproject and the wheel move together), so a rename
# that would strand every documented invocation reds here instead.
INSTALLED_COMMANDS = frozenset({"bn", "bn-agent"})


def _shared_bridge_modules() -> list[Path]:
    """The bridge modules that are SYMLINKS to their CLI source (#607)."""
    return sorted(
        path for path in (REPO / "src" / "bn_agent_bridge").glob("*.py")
        if path.is_symlink()
    )


def _shared_module_drift(wheel: Path) -> list[str]:
    """Shared modules the wheel ships as something other than the CLI source.

    A missing member counts as drift: a backend that drops the symlink without
    materializing it forks the two programs exactly as quietly as a stale copy
    does.
    """
    problems: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        shipped = set(archive.namelist())
        for source in _shared_bridge_modules():
            member = f"bn_agent_bridge/{source.name}"
            if member not in shipped:
                problems.append(f"{source.name}: missing from the wheel")
            elif archive.read(member) != source.read_bytes():
                problems.append(
                    f"{source.name}: the wheel copy differs from "
                    f"{source.resolve().relative_to(REPO)}"
                )
    return problems


def test_wheel_pins_the_console_entry_points_the_project_declares(wheel):
    """`pyproject.toml`'s `[project.scripts]` is where `bn` and `bn-agent` are
    declared, and the wheel's `entry_points.txt` is what becomes two commands on
    PATH -- nothing opened it at all (#788), so neither the declared surface nor
    the shipped one was pinned.

    Three claims: the commands a user ends up with are the ones the docs name,
    each is the target pyproject declares, and each target resolves -- a console
    script pointing at a renamed module installs a command that dies on use.
    """
    shipped = _console_scripts(wheel)
    assert set(shipped) == INSTALLED_COMMANDS, (
        f"the wheel installs {sorted(shipped)}; every agent-facing doc tells the "
        f"reader to run {sorted(INSTALLED_COMMANDS)}"
    )
    assert _entry_point_drift(wheel) == []
    for target in _declared_scripts().values():
        module, _, attribute = target.partition(":")
        assert callable(getattr(importlib.import_module(module), attribute)), target


def test_wheel_materializes_the_shared_bridge_modules_from_their_cli_source(wheel):
    """`src/bn_agent_bridge/{paths,version,proc_identity,socket_evidence,
    target_hint}.py` are symlinks into `src/bn/`, so the CLI and the bridge agree
    on layout, version, process identity, socket evidence and the multi-target
    hint grammar without duplication (#607).

    A wheel cannot carry a symlink: the backend materializes a copy, and a build
    that materialized a *stale* one would fork the two programs while every test
    in the tree stayed green, because they read the symlink (#788).
    """
    shared = _shared_bridge_modules()
    assert {"paths.py", "version.py"} <= {path.name for path in shared}, shared
    assert _shared_module_drift(wheel) == []


def test_the_wheel_content_pins_are_not_vacuous(tmp_path):
    """Negative control for the two pins above.

    Both are comparisons against the built wheel, and an empty comparison passes
    everything: a zip that was never read, or a member name that no longer
    matched, reports no drift at all. So each helper is run against a synthetic
    wheel carrying the defects the pins exist to catch -- a shared module
    materialized as an unrefreshed copy, a shared module dropped entirely, and a
    console script missing from `entry_points.txt`.
    """
    drifted = tmp_path / "drifted-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(drifted, "w") as archive:
        archive.writestr(
            "bn_agent_bridge/version.py", b"# a copy nobody refreshed\n"
        )
        archive.writestr(
            "drifted-0.0.0.dist-info/entry_points.txt",
            "[console_scripts]\nbn = bn.cli:main\n",
        )

    drift = _shared_module_drift(drifted)
    assert [entry.split(":", 1)[0] for entry in drift] == [
        path.name for path in _shared_bridge_modules()
    ]
    assert "differs" in next(e for e in drift if e.startswith("version.py"))
    assert "missing from the wheel" in next(e for e in drift if e.startswith("paths.py"))
    assert [entry.split(":", 1)[0] for entry in _entry_point_drift(drifted)] == ["bn-agent"]
