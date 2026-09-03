"""Shared pytest fixtures for the bn test suite.

`fake_transport` removes the per-test boilerplate of redeclaring a
`fake_send_request` closure: it installs a fake `bn.cli.send_request` that
records every call and returns canned results keyed by op, and hands back the
recorded-calls list so a test can assert on the request the CLI built (the
CLI's contract is argv -> bridge request, so this is a real assertion, not a
tautology). Bridge-side tests keep using the `_bridge_fakes._load_bridge` seam.

`_hermetic_env` is autouse: it pins the process environment every test (and
every subprocess a test spawns) runs under, so `uv run pytest` green means the
same thing on every machine. See `tests/test_suite_isolation.py`.

`integration_fixtures` + `require_real_bn` own the real-BN lane's gate (#590):
the generated `tests/fixtures/*_x86_64` binaries stay untracked, so the suite
builds them itself rather than skipping the only real-BN net silently.
"""
from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import bn.cli
import pytest
from bn.headless import _find_bn_python

sys.dont_write_bytecode = True

# --- real-BN integration lane (#590) --------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

#: Every binary `tests/fixtures/Makefile` builds. Untracked by design (they are
#: compiler output); `build_integration_fixtures()` is what puts them on disk.
REQUIRED_INTEGRATION_FIXTURES = (
    "hello_x86_64",
    "add_x86_64",
    "crypto_x86_64",
    "statemachine_x86_64",
    "parser_x86_64",
    "dispatch_table_x86_64",
)

_BUILD_LOCK_NAME = ".build.lock"
_BUILD_THREAD_LOCK = threading.Lock()

#: Set to fail instead of skip when the real-BN tier does not run -- for a
#: licensed lane, "skipped" must not be able to masquerade as green.
STRICT_ENV_VAR = "BN_REQUIRE_REAL_TESTS"

class FixtureBuildError(RuntimeError):
    """Raised when BN is present but the fixture toolchain/build is not."""


def bn_python_dir() -> Path | None:
    """The `python/` dir of a real BN install, or None.

    Platform-default discovery is the CLI's own (`bn.headless._find_bn_python`)
    rather than a second, narrower copy: one that only knew `/opt/binaryninja`
    would call BN "absent" on a host where `bn` itself finds it (e.g. the Darwin
    default), skipping the real-BN lane silently -- the exact #590 shape.

    `BN_INSTALL_DIR` is authoritative *here* though, unlike in `_find_bn_python`,
    which treats it as a first guess and falls back to the platform defaults.
    For a test gate that fallback is wrong in both directions: it makes
    `BN_INSTALL_DIR=/nonexistent` (how a lane pins which install to test, and how
    the gate itself is tested) quietly run against whatever BN happens to be in
    /opt. Pointing at an install that isn't there is an absent install, not an
    invitation to find another one.

    Resolved at call time, never at import: tests repoint the env var.
    """
    override = os.environ.get("BN_INSTALL_DIR")
    if override:
        candidate = Path(override).expanduser() / "python"
        return candidate if candidate.is_dir() else None
    return _find_bn_python()


def real_bn_available() -> bool:
    """Whether a real Binary Ninja install is importable.

    Deliberately says nothing about whether the fixtures are *built* (#590):
    conflating the two is what let a fresh checkout report "27 skipped, exit 0"
    with BN installed. Absence of BN is a skip; unbuilt fixtures are our job.
    """
    return bn_python_dir() is not None


def _strict_mode() -> bool:
    return os.environ.get(STRICT_ENV_VAR, "").strip().lower() not in ("", "0", "false", "no")


def require_real_bn() -> None:
    """Gate a real-BN test: skip visibly, or fail under strict mode."""
    if real_bn_available():
        return
    searched = os.environ.get("BN_INSTALL_DIR") or "the platform default install dir"
    message = (
        f"Binary Ninja not found ({searched}) -- real-BN tests did not run."
    )
    if _strict_mode():
        pytest.fail(f"{message} {STRICT_ENV_VAR} is set, so this is a failure.",
                    pytrace=False)
    pytest.skip(message)


#: Wall-clock ceiling for the six-binary build. Generous: six -O0 compiles are
#: sub-second, so blowing this means something is wedged, not merely slow.
_BUILD_TIMEOUT_SECONDS = 300


def _run_fixture_make(env: dict[str, str], out_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "-C", str(FIXTURES_DIR), f"OUTDIR={out_dir}", "all"],
        capture_output=True, text=True, timeout=_BUILD_TIMEOUT_SECONDS, env=env,
    )


def build_integration_fixtures(
    *,
    make_env: dict[str, str] | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    """Build every `*_x86_64` fixture binary into *out_dir*, returning paths.

    *out_dir* defaults to `tests/fixtures/` -- where the real-BN lane reads
    them. Tests that only exercise the build itself pass a tmp dir, so a
    unit-only run stays side-effect-free.

    Race-safe across pytest-xdist workers (an flock on `.build.lock`) and
    across threads in one process (`fcntl` locks are per-process, so the
    threading lock is not redundant). Every failure mode -- missing toolchain,
    make error, timeout -- raises `FixtureBuildError` with the diagnostics
    attached rather than degrading to a skip or leaking a raw subprocess error
    past the callers that catch the documented type.
    """
    out_dir = Path(out_dir) if out_dir is not None else FIXTURES_DIR
    env = {**os.environ, **(make_env or {})}
    for tool in ("make", env.get("CC", "cc")):
        if shutil.which(tool) is None and not Path(tool).is_file():
            raise FixtureBuildError(
                f"{tool!r} not found; the real-BN integration fixtures need a C "
                f"toolchain. Run: make -C tests/fixtures"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    with _BUILD_THREAD_LOCK:
        with open(out_dir / _BUILD_LOCK_NAME, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                proc = _run_fixture_make(env, out_dir)
            except subprocess.TimeoutExpired as exc:
                raise FixtureBuildError(
                    "Building the integration fixtures timed out after "
                    f"{_BUILD_TIMEOUT_SECONDS}s, so the real-BN tests cannot run. "
                    "Check for a wedged compiler and run: make -C tests/fixtures\n"
                    f"  stdout: {_decode(exc.stdout)}\n"
                    f"  stderr: {_decode(exc.stderr)}"
                ) from exc
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    missing = [n for n in REQUIRED_INTEGRATION_FIXTURES
               if not (out_dir / n).is_file()]
    if proc.returncode != 0 or missing:
        raise FixtureBuildError(
            "Binary Ninja is installed but the integration fixtures could not be "
            "built, so the real-BN tests cannot run. Fix the C toolchain and run: "
            f"make -C tests/fixtures\n"
            f"  exit status: {proc.returncode}\n"
            f"  missing: {', '.join(missing) or 'none'}\n"
            f"  stdout: {proc.stdout.strip()}\n"
            f"  stderr: {proc.stderr.strip()}"
        )
    return [out_dir / n for n in REQUIRED_INTEGRATION_FIXTURES]


def _decode(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        stream = stream.decode("utf-8", "replace")
    return stream.strip()


@pytest.fixture(scope="session")
def integration_fixtures() -> list[Path]:
    """Session-scoped owner of fixture generation for the real-BN lane."""
    require_real_bn()
    return build_integration_fixtures()


def pytest_runtest_setup(item):
    """Apply the real-BN gate to every `@pytest.mark.real_bn` test."""
    if item.get_closest_marker("real_bn") is not None:
        require_real_bn()

# Variables that make Python 3.14's stdlib argparse colorize usage/help text.
# `bn` has no color code of its own, but ~8 assertions on usage strings break
# when the developer's shell exports FORCE_COLOR (#589). requires-python is
# >=3.14, so this is permanent -- pin it rather than rewrite the assertions,
# which are testing the right thing.
_COLOR_FORCING_VARS = ("FORCE_COLOR", "CLICOLOR_FORCE", "PYTHON_COLORS")


@pytest.fixture(autouse=True)
def _hermetic_env(request, monkeypatch, tmp_path_factory):
    """Make every test's environment deterministic and free of real user state.

    - plain argparse output regardless of the developer's shell;
    - `BN_CACHE_DIR` pointed at a fresh per-test directory, so nothing can read
      or write the developer's real `~/.cache/bn` (instance registries, sticky
      `bn target use` / `bn instance use` pins) and no two tests can observe
      each other's cache state.

    A test may still override any of these with `monkeypatch` (many do); the
    override is restored to this isolated state at teardown. Tests that
    genuinely exercise platform-default path selection can opt out of the cache
    pin with `@pytest.mark.no_cache_isolation` -- narrow, and never for tests
    that merely touch the cache.
    """
    for var in _COLOR_FORCING_VARS:
        monkeypatch.delenv(var, raising=False)
    # #615 review F6: the CLI now reads BN_TAINT_MODELS directly (dataflow.py),
    # so an ambient value in the developer/CI shell must not leak into tests --
    # same precedent as the color vars above. A test that wants it set uses
    # monkeypatch.setenv itself (runs after this fixture, so it still overrides).
    monkeypatch.delenv("BN_TAINT_MODELS", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")

    # An ambient BN_INSTANCE in the developer's/CI's shell is "same effect as
    # always passing -i" (runtime.md) and silently changes instance
    # resolution for any test that doesn't itself pin one -- most visibly
    # `session stop`'s no-sticky-fallback guard (#588), which an ambient
    # BN_INSTANCE bypasses exactly like an explicit -i would. Scrub it so
    # suite results don't depend on the caller's environment; a test that
    # wants BN_INSTANCE sets it itself via monkeypatch.setenv.
    monkeypatch.delenv("BN_INSTANCE", raising=False)

    if request.node.get_closest_marker("no_cache_isolation") is None:
        cache_root = tmp_path_factory.mktemp("bn-cache")
        monkeypatch.setenv("BN_CACHE_DIR", str(cache_root))
    yield


@pytest.fixture
def fake_transport(monkeypatch):
    def install(results=None, *, default=None):
        results = results or {}
        calls = []

        def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                              instance_id=None, spawn_missing_named=False, **kwargs):
            calls.append({"op": op, "params": params, "target": target})
            if op in results:
                return results[op]
            if default is not None:
                return default
            raise AssertionError(f"unexpected op: {op}")

        monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
        return calls

    return install
