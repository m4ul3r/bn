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

`shared_bn` is one headless bridge for the whole pytest session, because that
lane's cost was process lifecycle rather than the work under test. Each test
still loads its own private copy of its binary and the fixture closes it
again, so the per-test isolation survives the process no longer being
per-test. Tests whose subject IS the process keep starting their own.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
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


def _invalidate_stale_bndb_sidecars(binaries: list[Path]) -> None:
    """Drop any `<binary>.bndb` whose mtime predates the binary it caches.

    The bridge resolves an adjacent `<binary>.bndb` in preference to the binary
    itself (`bn_agent_bridge.bridge._resolve_bndb_sidecar`), so a sidecar left
    behind by an earlier build of a since-changed fixture is analysed instead of
    the freshly compiled program -- surfacing as an assertion about the *old*
    program's call graph, indistinguishable from a read-path regression (#717).

    Mtime is the whole test, and only strictly-older sidecars go: a `.bndb`
    newer than its binary is the legitimate saved analysis. `make -C
    tests/fixtures clean` removes them too; this catches the long-lived checkout
    that never ran it.

    Callers hold the build lock (see `build_integration_fixtures`), so no other
    worker is rebuilding or removing these files -- but a sidecar can still be
    gone between the `stat()` and the `unlink()`, so the disappearances that
    race is allowed to produce (`FileNotFoundError`) are treated as "already
    invalidated" rather than escaping as a fixture-build flake.

    Any OTHER `OSError` is the opposite case and must not be swallowed: a stale
    database the build cannot remove is exactly the one the lane would go on to
    load, so it becomes a `FixtureBuildError` -- the type every caller of
    `build_integration_fixtures` documents and catches -- rather than a raw
    errno escaping past them.
    """
    for binary in binaries:
        sidecar = Path(str(binary) + ".bndb")
        try:
            sidecar_mtime = sidecar.stat().st_mtime
        except FileNotFoundError:
            continue
        if sidecar_mtime < binary.stat().st_mtime:
            try:
                sidecar.unlink(missing_ok=True)
            except OSError as exc:
                raise FixtureBuildError(
                    f"a saved analysis database older than the fixture it caches "
                    f"could not be removed, so the real-BN tests would analyse the "
                    f"stale database instead of the freshly built program. Remove "
                    f"it by hand and run: make -C tests/fixtures clean\n"
                    f"  database: {sidecar}\n"
                    f"  error: {exc}"
                ) from exc


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
                built = [out_dir / n for n in REQUIRED_INTEGRATION_FIXTURES]
                missing = [n for n in REQUIRED_INTEGRATION_FIXTURES
                           if not (out_dir / n).is_file()]
                if proc.returncode == 0 and not missing:
                    # Under the SAME lock as the build (#717): the
                    # stat/unlink pair below is only free of cross-process
                    # races while nobody else is rebuilding or clearing these
                    # sidecars, and `_BUILD_THREAD_LOCK` alone does not cover
                    # the other pytest-xdist workers sharing `out_dir`.
                    _invalidate_stale_bndb_sidecars(built)
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
    return built


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


# --- shared real-BN bridge -------------------------------------------------

#: The `bn` console script. The real-BN lane drives the installed entry point
#: as a subprocess rather than `-m bn.cli`, which the `bn` package name
#: shadows.
_BN_CLI = [str(Path(sys.executable).parent / "bn")]

#: Starting the shared bridge loads no binary, so this covers process spawn
#: plus BN import only.
_SHARED_BRIDGE_START_TIMEOUT = 120.0
_SHARED_BRIDGE_STOP_TIMEOUT = 30.0

#: One `bn load` into the LIVE bridge: ~0.2s for a fixture binary. A lane that
#: analyses something big (a cross-built `-static` probe) passes its own.
SHARED_LOAD_TIMEOUT = 120.0


class SharedBridge:
    """One headless bridge, reused by every test in the real-BN lane.

    The lane's cost was process lifecycle, not the work under test. Measured
    on a 6-core laptop, warm: `bn session start` 2.9s (fork + BN import +
    analysis) and `session stop` 0.6s, against 0.2s for `bn load` into a live
    bridge and ~0.4s for a read command. A bridge per test therefore spent
    ~3.5s of startup to run a handful of sub-second commands, 44 times in
    tests/test_integration.py alone.

    The isolation that mattered is kept: every test gets a private
    `BinaryView` over a private COPY of its binary, loaded at test start and
    closed at test end, so a retype/rename/tag/comment/create in one test
    cannot be observed by another -- and `bn save` writes its `.bndb` beside
    the copy, never beside the shared fixture binary.

    Exactly one target is open while a test runs, which is what the lane's
    commands assume when they omit `--target`. `begin()` refuses a bridge that
    is not clean and `end()` closes everything again, so that invariant is
    enforced rather than hoped for: a leak is reported against the test that
    leaked instead of failing the next test.

    NOT for lifecycle tests. `session start/stop/restart`, multi-instance
    selection and bndb-restore-on-reload are about the process this fixture
    amortises away; they keep spawning their own bridges.
    """

    def __init__(self, instance_id: str, cache_dir: Path) -> None:
        self.instance_id = instance_id
        self.cache_dir = cache_dir
        self._scratch: Path | None = None
        self._loaded: list[str] = []

    def env(self) -> dict[str, str]:
        """The environment a call against this bridge must carry.

        Built at call time, never snapshotted: the autouse `_hermetic_env`
        rewrites the environment per test (#589). `BN_CACHE_DIR` is forced
        because the shared bridge's registry lives in the session-scoped cache,
        not in the caller's per-test one.
        """
        env = dict(os.environ)
        env["BN_CACHE_DIR"] = str(self.cache_dir)
        return env

    def run(self, *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
        """Run `bn --instance <shared> ...`; the returncode is the caller's to assert."""
        return subprocess.run(
            [*_BN_CLI, "--instance", self.instance_id, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=self.env(),
        )

    def json(self, *args: str, timeout: float = 60.0):
        """Run a command that must succeed and parse its JSON."""
        res = self.run(*args, "--format", "json", timeout=timeout)
        assert res.returncode == 0, f"bn {' '.join(args)} failed: {res.stderr}\n{res.stdout}"
        return json.loads(res.stdout)

    def load(self, binary: Path | str, *, copy: bool = True,
             timeout: float = SHARED_LOAD_TIMEOUT) -> str:
        """Load *binary* into the shared bridge; returns its path selector.

        Copied into the test's own scratch directory by default, so a test that
        mutates or saves cannot reach the shared fixture binary or another
        test's view. `copy=False` is for a binary the test already built into
        its own `tmp_path`.

        An adjacent `<binary>.bndb` travels with the copy: the bridge loads a
        saved database in preference to the binary (#717), so leaving it behind
        would silently turn a primed sub-second load back into a full analysis.
        """
        source = Path(binary)
        if copy:
            assert self._scratch is not None, "SharedBridge.load() outside a shared_bn test"
            path = self._scratch / source.name
            shutil.copy2(source, path)
            sidecar = source.with_name(source.name + ".bndb")
            if sidecar.exists():
                shutil.copy2(sidecar, path.with_name(path.name + ".bndb"))
        else:
            path = source.resolve()
        res = self.run("load", str(path), "--format", "json", timeout=timeout)
        assert res.returncode == 0, f"load {path} failed: {res.stderr}\n{res.stdout}"
        # `path` is what the bridge actually opened -- the `.bndb` when the
        # sidecar won -- and it is the ONLY spelling `target list` and `close`
        # agree with, so it is what gets recorded and handed back.
        opened = json.loads(res.stdout)["path"]
        self._loaded.append(opened)
        return opened

    def open_targets(self) -> list[str]:
        """Absolute paths of every target currently open on the bridge."""
        res = self.run("target", "list", "--format", "json", timeout=30.0)
        assert res.returncode == 0, f"target list failed: {res.stderr}\n{res.stdout}"
        return [item["filename"] for item in json.loads(res.stdout)["items"]]

    def begin(self, scratch: Path) -> None:
        """Hand the bridge to one test, refusing to hand over a dirty one."""
        self._scratch = scratch
        self._loaded = []
        leaked = self.open_targets()
        assert not leaked, (
            f"the shared bridge still has {leaked} open before this test -- a "
            "previous test left a target behind, which would change target "
            "resolution here")

    def end(self) -> list[str]:
        """Close everything and clear sticky state; returns targets nobody declared.

        Closes what `target list` reports rather than only what `load()`
        recorded: a test that loaded by hand must not be able to poison the
        rest of the session. The undeclared ones are returned so the test that
        leaked them is the one that fails.
        """
        undeclared = [path for path in self.open_targets() if path not in self._loaded]
        failures = []
        for path in [*reversed(self._loaded), *undeclared]:
            res = self.run("close", path, timeout=30.0)
            if res.returncode != 0 and path in self.open_targets():
                failures.append(f"{path}: {res.stderr.strip() or res.stdout.strip()}")
        self._loaded = []
        self._scratch = None
        # Sticky `bn target use` / `bn instance use` pins live under the cache
        # root, which is shared here; a pin set by one test would silently
        # redirect the next one's resolution.
        shutil.rmtree(sessions_dir_for(self.cache_dir), ignore_errors=True)
        assert not failures, f"the shared bridge could not be cleaned up: {failures}"
        return undeclared


def sessions_dir_for(cache_dir: Path) -> Path:
    """`bn.paths.sessions_dir()` for a cache root that is not the ambient one."""
    return cache_dir / "sessions"


@pytest.fixture(scope="session")
def _shared_bridge(tmp_path_factory) -> Iterator[SharedBridge]:
    require_real_bn()
    cache_dir = tmp_path_factory.mktemp("bn-shared-bridge")
    env = dict(os.environ)
    env["BN_CACHE_DIR"] = str(cache_dir)
    env["NO_COLOR"] = "1"
    started = subprocess.run(
        [*_BN_CLI, "session", "start", "--format", "json"],
        capture_output=True, text=True, timeout=_SHARED_BRIDGE_START_TIMEOUT, env=env,
    )
    assert started.returncode == 0, (
        f"shared bridge failed to start: {started.stderr}\n{started.stdout}")
    bridge = SharedBridge(json.loads(started.stdout)["instance_id"], cache_dir)
    try:
        yield bridge
    finally:
        subprocess.run(
            [*_BN_CLI, "session", "stop", bridge.instance_id],
            capture_output=True, text=True, timeout=_SHARED_BRIDGE_STOP_TIMEOUT, env=env,
        )


@pytest.fixture
def shared_bn(_shared_bridge, tmp_path, monkeypatch) -> Iterator[SharedBridge]:
    """The shared bridge, scrubbed before and after one test.

    `BN_CACHE_DIR` is re-pointed at the shared cache for the duration: the
    autouse per-test pin would otherwise hide the bridge from any subprocess
    the test spawns itself, and `SharedBridge.env()` and the test would then
    disagree about which cache they are talking to.
    """
    monkeypatch.setenv("BN_CACHE_DIR", str(_shared_bridge.cache_dir))
    scratch = tmp_path / "shared-bn"
    scratch.mkdir()
    _shared_bridge.begin(scratch)
    try:
        yield _shared_bridge
    finally:
        undeclared = _shared_bridge.end()
    assert not undeclared, (
        f"this test left {undeclared} open on the shared bridge; load through "
        "`shared_bn.load()` so the fixture closes it")


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
