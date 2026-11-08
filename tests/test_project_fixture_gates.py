"""Behavioral coverage for native-project and rootfs fixture prerequisites."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import conftest


def _fake_project_probe(tmp_path, monkeypatch, *, product, failure=None):
    """Exercise the real child interpreter without importing native BN."""
    python_dir = tmp_path / "fake-bn-python"
    python_dir.mkdir()
    marker = tmp_path / "project-events.jsonl"
    (python_dir / "binaryninja.py").write_text(
        f"""import json
import os
from pathlib import Path

MARKER = Path({str(marker)!r})
FAILURE = {failure!r}


def record(event, **values):
    with MARKER.open('a') as stream:
        stream.write(json.dumps(dict(event=event, **values)) + '\\n')


record('import', pid=os.getpid(), plugins=os.environ.get('BN_DISABLE_USER_PLUGINS'))


def core_product():
    return {product!r}


def disable_default_log():
    pass


class Project:
    def __init__(self, path):
        self.path = path

    @classmethod
    def create_project(cls, path, name):
        record('create', path=path, product=core_product())
        Path(path).mkdir()
        (Path(path) / 'partial-project').write_text('created')
        if FAILURE == 'create':
            raise RuntimeError('native project creation denied')
        if FAILURE == 'none':
            return None
        return cls(path)

    def close(self):
        record('close', path=self.path)
        if FAILURE == 'close':
            raise RuntimeError('native project close failed')
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(conftest, "bn_python_dir", lambda: python_dir)
    monkeypatch.setattr(conftest, "require_real_bn", lambda: None)
    # The probe must override an inherited value, not merely rely on the host.
    monkeypatch.setenv("BN_DISABLE_USER_PLUGINS", "0")
    real_run = subprocess.run
    reports = []

    def run_probe(command, **kwargs):
        assert command[:2] == [sys.executable, "-c"]
        result = real_run(command, **kwargs)
        assert result.returncode == 0, result.stderr
        reports.extend(
            json.loads(line.removeprefix("BN_PROJECT_CAPABILITY="))
            for line in result.stdout.splitlines()
            if line.startswith("BN_PROJECT_CAPABILITY=")
        )
        return result

    monkeypatch.setattr(conftest.subprocess, "run", run_probe)
    return marker, reports


def _assert_probe_cleanup(marker, *, product, closed):
    events = [json.loads(line) for line in marker.read_text().splitlines()]
    imported, created, *remaining = events
    assert imported["event"] == "import"
    assert imported["pid"] != os.getpid()
    assert imported["plugins"] == "1"
    assert created["event"] == "create"
    assert created["product"] == product
    project_path = Path(created["path"])
    assert remaining == ([{"event": "close", "path": str(project_path)}] if closed else [])
    assert not project_path.exists()
    assert not project_path.parent.exists()


@pytest.mark.parametrize("strict", [False, True])
def test_personal_product_with_working_project_api_is_available(tmp_path, monkeypatch, strict):
    monkeypatch.setenv("BN_REQUIRE_REAL_TESTS", "1" if strict else "0")
    marker, reports = _fake_project_probe(tmp_path, monkeypatch, product="Personal")
    parent_module = sys.modules.get("binaryninja")

    try:
        assert conftest.require_project_capability() is None
    except pytest.skip.Exception as exc:
        pytest.fail(f"Working project API was incorrectly skipped: {exc}")

    assert reports == [{"available": True}]
    assert sys.modules.get("binaryninja") is parent_module
    _assert_probe_cleanup(marker, product="Personal", closed=True)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    ("failure", "reason", "closed"),
    [
        ("create", "native project creation denied", False),
        ("none", "create_project returned None", False),
        ("close", "native project close failed", True),
    ],
)
def test_commercial_product_without_working_project_api_refuses_lane(
    tmp_path, monkeypatch, strict, failure, reason, closed
):
    monkeypatch.setenv("BN_REQUIRE_REAL_TESTS", "1" if strict else "0")
    marker, reports = _fake_project_probe(
        tmp_path, monkeypatch, product="Commercial", failure=failure
    )

    # Catch both so a regression from strict failure to skip cannot turn green.
    with pytest.raises((pytest.skip.Exception, pytest.fail.Exception)) as caught:
        conftest.require_project_capability()

    expected = pytest.fail.Exception if strict else pytest.skip.Exception
    assert isinstance(caught.value, expected)
    assert reason in str(caught.value)
    if strict:
        assert "BN_REQUIRE_REAL_TESTS" in str(caught.value)
    assert reports == [{"available": False, "reason": reason}]
    _assert_probe_cleanup(marker, product="Commercial", closed=closed)


@pytest.mark.parametrize("strict", [False, True])
def test_rootfs_capability_gate_precedes_compiler_discovery_and_build(
    tmp_path, monkeypatch, strict
):
    monkeypatch.setenv("BN_REQUIRE_REAL_TESTS", "1" if strict else "0")
    marker, reports = _fake_project_probe(
        tmp_path, monkeypatch, product="Commercial", failure="create"
    )

    def unexpected_toolchain_work():
        raise AssertionError("Unsupported project API must stop before toolchain work")

    monkeypatch.setattr(conftest, "find_rootfs_cross_compiler", unexpected_toolchain_work)
    monkeypatch.setattr(conftest, "build_rootfs_fixture", unexpected_toolchain_work)

    with pytest.raises((pytest.skip.Exception, pytest.fail.Exception)) as caught:
        conftest.rootfs_fixture.__wrapped__()

    expected = pytest.fail.Exception if strict else pytest.skip.Exception
    assert isinstance(caught.value, expected)
    assert "native project creation denied" in str(caught.value)
    assert reports == [{"available": False, "reason": "native project creation denied"}]
    _assert_probe_cleanup(marker, product="Commercial", closed=False)


@pytest.mark.parametrize("strict", [False, True])
def test_rootfs_missing_compiler_preserves_strict_refusal(monkeypatch, strict):
    monkeypatch.setenv("BN_REQUIRE_REAL_TESTS", "1" if strict else "0")
    calls = []

    def capability():
        calls.append("capability")

    def compiler():
        calls.append("compiler")
        return None

    def unexpected_build():
        raise AssertionError("Missing compiler must stop before building")

    monkeypatch.setattr(conftest, "require_project_capability", capability)
    monkeypatch.setattr(conftest, "find_rootfs_cross_compiler", compiler)
    monkeypatch.setattr(conftest, "build_rootfs_fixture", unexpected_build)

    with pytest.raises((pytest.skip.Exception, pytest.fail.Exception)) as caught:
        conftest.rootfs_fixture.__wrapped__()

    expected = pytest.fail.Exception if strict else pytest.skip.Exception
    assert isinstance(caught.value, expected)
    assert "no Linux musl cross compiler" in str(caught.value)
    assert "CROSS_CC" in str(caught.value)
    if strict:
        assert "BN_REQUIRE_REAL_TESTS" in str(caught.value)
    assert calls == ["capability", "compiler"]


@pytest.fixture
def rootfs_build_directory(tmp_path, monkeypatch):
    fixtures = tmp_path / "fixtures"
    monkeypatch.setattr(conftest, "FIXTURES_DIR", fixtures)
    monkeypatch.setattr(conftest, "ROOTFS_FIXTURE_DIR", fixtures / "rootfs")
    for name in ("CROSS_CC", "LIBC_SRC", "LDMUSL"):
        monkeypatch.delenv(name, raising=False)

    def which(command):
        assert command == "make"
        return "/fake/bin/make"

    monkeypatch.setattr(conftest.shutil, "which", which)
    return tmp_path / "rootfs-output"


def _write_rootfs_outputs(out_dir):
    for relative in (
        "etc/ld.so.conf",
        "etc/ld.so.conf.d/vendor.conf",
        "etc/profile",
        "lib/libc.so",
        "opt/vendor/lib/libgreet.so.1",
        "usr/bin/app",
        "usr/bin/libstray.so",
    ):
        path = out_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture output")
    (out_dir / "opt/vendor/lib/libgreet.so").symlink_to("libgreet.so.1")
    (out_dir / "lib/ld-musl-aarch64.so.1").symlink_to("libc.so")


def test_rootfs_make_nonzero_is_build_error_even_with_complete_outputs(
    rootfs_build_directory, monkeypatch
):
    out_dir = rootfs_build_directory

    def make(command, **kwargs):
        _write_rootfs_outputs(out_dir)
        return subprocess.CompletedProcess(command, 2, "compiler output", "linker failed")

    monkeypatch.setattr(conftest.subprocess, "run", make)

    with pytest.raises(conftest.FixtureBuildError) as caught:
        conftest.build_rootfs_fixture(out_dir=out_dir)

    message = str(caught.value)
    assert "exit status: 2" in message
    assert "compiler output" in message
    assert "linker failed" in message


def test_rootfs_make_timeout_is_build_error_with_diagnostics(
    rootfs_build_directory, monkeypatch
):
    def make(command, **kwargs):
        raise subprocess.TimeoutExpired(
            command, kwargs["timeout"], output=b"compiler stalled", stderr=b"linker waiting"
        )

    monkeypatch.setattr(conftest.subprocess, "run", make)

    with pytest.raises(conftest.FixtureBuildError) as caught:
        conftest.build_rootfs_fixture(out_dir=rootfs_build_directory)

    assert "timed out" in str(caught.value)
    assert "compiler stalled" in str(caught.value)
    assert "linker waiting" in str(caught.value)
    assert isinstance(caught.value.__cause__, subprocess.TimeoutExpired)
