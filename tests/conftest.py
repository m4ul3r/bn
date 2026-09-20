"""Shared pytest fixtures for the bn test suite.

`fake_transport` removes the per-test boilerplate of redeclaring a
`fake_send_request` closure: it installs a fake `bn.cli.send_request` that
records every call and returns canned results keyed by op, and hands back the
recorded-calls list so a test can assert on the request the CLI built (the
CLI's contract is argv -> bridge request, so this is a real assertion, not a
tautology). Bridge-side tests keep using the `_bridge_fakes._load_bridge` seam.

`_hermetic_env` and `_hermetic_session_env` are both autouse: they apply ONE
rule (`SCRUBBED_ENV_VARS`) at two scopes, so the environment every test -- and
every subprocess a test or a SESSION-scoped fixture spawns -- runs under is
the same on every machine. The session half exists because pytest builds
session-scoped fixtures first, and a process started there (the shared bridge)
outlives the test that started it. See `tests/test_suite_isolation.py`.

`integration_fixtures` + `require_real_bn` own the real-BN lane's gate (#590):
the generated `tests/fixtures/*_x86_64` binaries stay untracked, so the suite
builds them itself rather than skipping the only real-BN net silently.
`BN_REQUIRE_REAL_TESTS` is the strict flag over both halves of that gate: it
refuses ANY skip the machine cannot un-skip, an absent BN via `require_real_bn`
and an environment-bound skip via `refuse_silent_skip`.

`shared_bn` is one headless bridge for the whole pytest session, because that
lane's cost was process lifecycle rather than the work under test. Each test
still loads its own private copy of its binary and the fixture closes it
again, so the per-test isolation survives the process no longer being
per-test. Tests whose subject IS the process keep starting their own.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn

import bn.cli
import pytest
from bn.headless import _find_bn_python
from bn.proc_identity import PIDFD_AVAILABLE, PinUnavailable, pin_process
from bn.transport import DEFAULT_REQUEST_TIMEOUT

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


def refuse_silent_skip(reason: str) -> NoReturn:
    """Skip on a condition the machine cannot change -- loudly under strict mode.

    `require_real_bn` covers an absent BN, which a machine CAN fix by
    installing one. A skip nothing can un-skip (running as root, so the DAC
    checks a permission test asserts are bypassed) is worse: it is a test
    nobody ever runs, and on a root CI container it disappears permanently
    with no signal. Strict mode is the flag that says "this lane is supposed
    to be complete", so under it such a skip is a failure, not a pass.

    *reason* carries its own remedy, because only the caller knows one: this
    helper is not root-specific and must not advise as though it were.
    """
    if _strict_mode():
        pytest.fail(
            f"{reason} -- {STRICT_ENV_VAR} is set, so this skip is a failure "
            "rather than a silent pass.",
            pytrace=False,
        )
    pytest.skip(reason)


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

#: `BN_IDLE_TIMEOUT` for every bridge a TEST spawns. A test that spawns a
#: bridge and does not stop it leaves a ~450 MB BN process nothing can reach --
#: pytest rotates the per-test cache away, so its registry (and with it
#: `session list` / `gc`) goes with it. The reaper only has to outlive a test.
_TEST_BRIDGE_IDLE_TIMEOUT = 120.0

#: `BN_IDLE_TIMEOUT` for the SESSION-scoped shared bridge, which is
#: legitimately idle between real-BN tests: long enough that no test gap can
#: reach it, short enough that it still dies on its own if its `session stop`
#: below ever fails.
_SHARED_BRIDGE_IDLE_TIMEOUT = 3600.0

#: Grace period for a SIGTERM'd leaked bridge to exit before the sweep reports it.
_LEAK_SWEEP_GRACE = 3.0

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
def _shared_bridge(_hermetic_session_env, tmp_path_factory) -> Iterator[SharedBridge]:
    # The scrub is requested BY NAME, not relied on for being autouse: this
    # process outlives the test that starts it, so an ambient variable that
    # reaches it here can never be taken back (#730 review).
    require_real_bn()
    cache_dir = tmp_path_factory.mktemp("bn-shared-bridge")
    env = dict(os.environ)
    env["BN_CACHE_DIR"] = str(cache_dir)
    env["BN_IDLE_TIMEOUT"] = str(_SHARED_BRIDGE_IDLE_TIMEOUT)
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

#: Every variable the suite refuses to inherit from the caller, in ONE place so
#: the per-test and per-session gates cannot scrub different sets:
#:
#: - the color-forcing trio above;
#: - `BN_TAINT_MODELS`, which the CLI reads directly (dataflow.py, #615 review
#:   F6): an ambient overlay silently changes taint semantics, and a malformed
#:   one fails the run outright;
#: - `BN_INSTANCE`, which is "the same effect as always passing -i"
#:   (runtime.md) and silently redirects instance resolution for any test that
#:   does not pin one -- most visibly `session stop`'s no-sticky-fallback
#:   guard (#588), which an ambient value bypasses exactly like an explicit -i.
#: - `BN_SPILL_TOKENS`, the opt-in spill threshold (#409): an ambient value
#:   re-arms disk output for every test that assumes the default, which is now
#:   "print the payload, write nothing".
#:
#: A test that WANTS one of these sets it itself with `monkeypatch.setenv`,
#: which runs after both gates.
SCRUBBED_ENV_VARS = (*_COLOR_FORCING_VARS, "BN_TAINT_MODELS", "BN_INSTANCE", "BN_SPILL_TOKENS")


def _apply_hermetic_env(patch: pytest.MonkeyPatch) -> None:
    """The scrub-and-pin rules, applied through *patch* at whatever scope owns it."""
    for var in SCRUBBED_ENV_VARS:
        patch.delenv(var, raising=False)
    patch.setenv("NO_COLOR", "1")
    # #733 F5: a test that spawns a bridge and does not stop it leaves a
    # ~450 MB BN process that nothing can reach -- pytest rotates the per-test
    # cache away, so its registry (and with it `session list` / `gc`) goes too.
    # `transport._spawn_instance_unlocked` forwards the whole `os.environ` to
    # the child, so a pin here reaches every auto-spawned and `session
    # start`-spawned bridge. The reaper is headless-only, arms after preload,
    # and never fires while a request or load job is in flight, so it cannot
    # destabilise a test; it just means a leak dies on its own.
    #
    # HERE rather than in `_hermetic_env`, for the reason this function exists:
    # a rule that lives only at function scope misses every session-scoped
    # spawner, which is exactly how an ambient `BN_TAINT_MODELS` reached the
    # shared bridge (#730). Pinned, not scrubbed: a name that is both would
    # make `tests/test_suite_isolation.py`'s polluted child contradict the
    # scrub it asserts.
    patch.setenv("BN_IDLE_TIMEOUT", str(_TEST_BRIDGE_IDLE_TIMEOUT))


@pytest.fixture(scope="session", autouse=True)
def _hermetic_session_env() -> Iterator[None]:
    """Scrub the ambient environment for the WHOLE session, not only per test.

    `_hermetic_env` below is function-scoped, and pytest builds session-scoped
    fixtures FIRST -- so anything that spawns a process at session scope
    captured `os.environ` before a single variable had been scrubbed. The
    shared bridge is exactly that: it inherited an ambient `BN_TAINT_MODELS`
    into a process that OUTLIVES the test which started it, where no later
    `monkeypatch.delenv` can reach it, and the taint lane then ran against the
    developer's model overlay (a malformed one failing the run outright).
    Before the shared bridge existed every bridge was spawned inside a test,
    after the per-test scrub, which is why the hole only opened now.

    Scrubbing at session scope closes it for every session-scoped spawner
    rather than for one of them; the per-test fixture then re-applies the same
    rules over whatever a test did in between.
    """
    with pytest.MonkeyPatch.context() as patch:
        _apply_hermetic_env(patch)
        yield


@pytest.fixture(scope="session")
def session_scope_environment(_hermetic_session_env) -> dict[str, str]:
    """`os.environ` as a SESSION-scoped fixture sees it.

    A peer of the session-scoped spawners, so it observes what they observe.
    Its assertion lives in `tests/test_suite_isolation.py`, which also owns the
    polluted child run that keeps that assertion from passing vacuously on a
    developer machine with a clean shell.
    """
    return dict(os.environ)


@pytest.fixture(autouse=True)
def _hermetic_env(request, monkeypatch, tmp_path_factory):
    """Make every test's environment deterministic and free of real user state.

    - plain argparse output regardless of the developer's shell;
    - none of `SCRUBBED_ENV_VARS` inherited from the caller;
    - `BN_CACHE_DIR` pointed at a fresh per-test directory, so nothing can read
      or write the developer's real `~/.cache/bn` (instance registries, sticky
      `bn target use` / `bn instance use` pins) and no two tests can observe
      each other's cache state.
    - `BN_IDLE_TIMEOUT` pinned, so a bridge a test leaks self-terminates
      instead of surviving the whole run unreachable (#733 F5).

    A test may still override any of these with `monkeypatch` (many do); the
    override is restored to this isolated state at teardown. Tests that
    genuinely exercise platform-default path selection can opt out of the cache
    pin with `@pytest.mark.no_cache_isolation` -- narrow, and never for tests
    that merely touch the cache.
    """
    _apply_hermetic_env(monkeypatch)
    if request.node.get_closest_marker("no_cache_isolation") is None:
        cache_root = tmp_path_factory.mktemp("bn-cache")
        monkeypatch.setenv("BN_CACHE_DIR", str(cache_root))
    yield


@pytest.fixture
def fake_transport(monkeypatch):
    """Install a recording `bn.cli.send_request`; returns the installer.

    Each recorded call is ONE dict describing the request the CLI built: `op`,
    `params` and `target`, plus every keyword `bn.cli.send_request` accepts --
    `timeout`, `default_timeout`, `connect_retries`, `instance_id`,
    `spawn_missing_named`, `resolved`, `idle_probe`. Recorded in full rather
    than only op/params/target (#787): a fake that drops the routing kwargs
    cannot catch a routing regression, so a test could assert the op while the
    CLI sent the call to the wrong INSTANCE, with the wrong spawn policy, or
    without the `resolved` flag a shrinking end-to-end budget depends on -- and
    the suite stayed green.

    The signature is the real one's parameter for parameter, with NO `**kwargs`
    catch-all: a recorder more forgiving than the function it replaces accepts a
    call the live `send_request` raises TypeError on, so a handler shipping a
    misspelled or removed routing kwarg would pass the mocked suite and break
    against a real bridge. Because it binds identically, the recorded values are
    the ones the real call would bind: `timeout=None` means the CLI left timeout
    resolution to `send_request` (it does on every primary call), not that the
    fake defaulted.
    """
    def install(results=None, *, default=None):
        results = results or {}
        calls = []

        def fake_send_request(op, *, params=None, target=None, timeout=None,
                              default_timeout=DEFAULT_REQUEST_TIMEOUT,
                              connect_retries=4, instance_id=None,
                              spawn_missing_named=False, resolved=False,
                              idle_probe=False):
            calls.append({
                "op": op,
                "params": params,
                "target": target,
                "timeout": timeout,
                "default_timeout": default_timeout,
                "connect_retries": connect_retries,
                "instance_id": instance_id,
                "spawn_missing_named": spawn_missing_named,
                "resolved": resolved,
                "idle_probe": idle_probe,
            })
            if op in results:
                return results[op]
            if default is not None:
                return default
            raise AssertionError(f"unexpected op: {op}")

        monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
        return calls

    return install


def process_discovery_available(proc: Path = Path("/proc")) -> bool:
    """Whether this host can be ASKED which processes are running.

    The sweep answers an unreadable `/proc` with an empty list -- unknowable,
    never guessed -- so a test that requires discovery must gate on this rather
    than assert an empty answer is a pass (#733 F5 review).
    """
    try:
        return proc.joinpath(str(os.getpid()), "cmdline").exists()
    except OSError:
        return False


def _bn_agent_row(entry: Path) -> dict[str, str] | None:
    """The leak row for one `/proc/<pid>` entry, or None if it is not a bridge.

    Shared by the scan and the reap, so the reap can re-verify that the pid it
    is about to signal still describes the SAME bridge it scanned.
    """
    try:
        argv = entry.joinpath("cmdline").read_bytes().split(b"\0")
        environ = entry.joinpath("environ").read_bytes().split(b"\0")
    except OSError:
        return None                       # a pid may vanish mid-walk
    # `os.fsdecode`, not `decode("utf-8", "replace")`: a filesystem path is
    # bytes, and replacement turned a non-UTF-8 byte in a cache path into
    # U+FFFD while `Path`/`str(under)` keeps it through surrogateescape, so the
    # ownership comparison below excluded a bridge that WAS in this worker's
    # root (#733 F5 review).
    decoded = [os.fsdecode(part) for part in argv if part]
    # All three spellings a bridge can be launched under: the console script,
    # `transport._find_bn_agent`'s fallback when no `bn-agent` sits beside
    # `sys.executable` (`python -m bn.headless`, what a `pip install --user`
    # layout gets), and the package's own module form.
    if not any(
        os.path.basename(part) == "bn-agent"
        or part in ("bn.headless", "bn_agent_bridge")
        for part in decoded
    ):
        return None
    cache_dir = ""
    for var in environ:
        if var.startswith(b"BN_CACHE_DIR="):
            cache_dir = os.fsdecode(var[len(b"BN_CACHE_DIR="):])
            break
    instance_id = "<unknown>"
    for index, part in enumerate(decoded):
        if part == "--instance-id" and index + 1 < len(decoded):
            instance_id = decoded[index + 1]
            break
    return {"pid": entry.name, "instance_id": instance_id, "cache_dir": cache_dir}


def bn_agent_leaks(under: Path, *, proc: Path = Path("/proc")) -> list[dict[str, str]]:
    """Every live `bn-agent` whose `BN_CACHE_DIR` lies under *under*.

    Walks /proc directly, like `bn.proc_identity` and `bn.socket_evidence` do
    (there is no psutil dependency). A host without /proc answers an empty
    list: unknowable, never guessed -- so a caller that needs a real answer
    asks `process_discovery_available()` first.

    Scoped by cache root rather than by process tree, because a bridge is
    spawned detached from the test that asked for it -- and scoped to THIS
    worker's basetemp by the caller, so one xdist worker's sweep cannot see
    another's bridge.

    The scoping has one disclosed blind spot: a bridge whose `BN_CACHE_DIR` is
    NOT under *under* is invisible here -- an `@pytest.mark.no_cache_isolation`
    test's bridge, or one spawned with the variable unset. Deliberate: the
    alternative is failing this run for another repo's (or another developer's)
    live bridge on the same host. Belt 1's `BN_IDLE_TIMEOUT` pin still reaps
    such a leak, on a timer rather than loudly.
    """
    leaks: list[dict[str, str]] = []
    try:
        entries = list(proc.iterdir())
    except OSError:
        return leaks
    root = str(under)
    for entry in entries:
        if not entry.name.isdigit():
            continue
        row = _bn_agent_row(entry)
        if row is None:
            continue
        cache_dir = row["cache_dir"]
        if cache_dir != root and not cache_dir.startswith(f"{root}{os.sep}"):
            continue
        leaks.append(row)
    # Pid order, so a multi-leak failure message reads the same twice and the
    # sweep's own tests can assert the rows rather than a set.
    return sorted(leaks, key=lambda row: int(row["pid"]))


def _pid_is_gone(pid: int, *, proc: Path = Path("/proc")) -> bool:
    """Whether *pid* has stopped running -- a zombie counts.

    `os.kill(pid, 0)` succeeds for a process that has EXITED but not been
    reaped, and a leak the suite itself spawned is a child nobody waits on, so
    signal 0 alone reports a terminated bridge as still alive. The kernel's own
    state letter is read instead, exactly like `bn.proc_identity` does.
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    try:
        stat = proc.joinpath(str(pid), "stat").read_bytes()
    except OSError:
        return True
    # `pid (comm) state ...`, and comm may itself contain spaces/parens, so
    # the state letter is the first field after the LAST `)`.
    tail = stat.rpartition(b")")[2].split()
    return bool(tail) and tail[0] == b"Z"


def _terminate_scanned_leak(
    row: dict[str, str], *, proc: Path = Path("/proc")
) -> str:
    """SIGTERM the process *row* describes, or refuse; returns the detail line.

    The scan records a pid and the signal comes later -- after the rest of the
    walk and up to `_LEAK_SWEEP_GRACE` per preceding leak -- so a bare
    `os.kill` on that stale number can terminate an unrelated process if the
    bridge exited and the kernel recycled the pid in between (#733 F5 review).

    Two defences, in order. `bn.proc_identity.pin_process` holds the pid to ONE
    process for as long as the pin is open, which closes the race outright.
    Only where the platform provides no pidfd primitives at all -- including
    the interpreter that already skips this repo's pidfd tests -- is the signal
    sent unpinned, and then the row is RE-DERIVED from `/proc` immediately
    before it, which narrows the window from the whole sweep to that
    instruction pair; that is as close as a pidfd-less platform gets, and it
    beats never signalling, which leaves a ~450 MB process running on exactly
    the host where the reap is most useful. Which path was taken is stated in
    the line, so the residual is visible.

    Every OTHER pin failure REFUSES. `pin_process` raises `PinUnavailable` for
    operational reasons too -- EMFILE, EPERM, a process that has already gone
    -- and treating those as the no-pidfd tradeoff dropped a pidfd-CAPABLE host
    to an unpinned kill while reporting "no pidfd on this platform", i.e. it
    lost the protection and misstated why (#733 F5 review round 2). The real
    reason is carried into the line instead.
    """
    pid = int(row["pid"])
    try:
        pin = pin_process(pid)
    except PinUnavailable as exc:
        if PIDFD_AVAILABLE:
            return f"NOT signalled: the pid could not be pinned ({exc})"
        pin = None
    with contextlib.ExitStack() as closing:
        if pin is not None:
            closing.enter_context(pin)
        if _bn_agent_row(proc.joinpath(str(pid))) != row:
            # The pid no longer describes what was scanned, so whatever it
            # names now is not this suite's to kill.
            return ("gone before it could be signalled (the pid no longer "
                    "names this bridge)")
        how = "pinned" if pin is not None else "unpinned (no pidfd on this platform)"
        try:
            if pin is not None:
                pin.send(signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return "already exited, nothing to signal"
        except OSError as exc:
            return f"SIGTERM refused ({exc})"
        exited = False
        deadline = time.monotonic() + _LEAK_SWEEP_GRACE
        while time.monotonic() < deadline:
            if _pid_is_gone(pid, proc=proc):
                exited = True
                break
            time.sleep(0.1)
    return f"SIGTERM sent {how}, exited={exited}"


def reap_bn_agent_leaks(
    leaks: list[dict[str, str]], *, proc: Path = Path("/proc")
) -> list[str]:
    """SIGTERM each leak and return one human line per leak, with its outcome."""
    return [
        f"instance {row['instance_id']} (pid {row['pid']}) "
        f"BN_CACHE_DIR={row['cache_dir']} — "
        f"{_terminate_scanned_leak(row, proc=proc)}"
        for row in leaks
    ]


@pytest.fixture(scope="session", autouse=True)
def _refuse_leaked_bridges(tmp_path_factory) -> Iterator[None]:
    """A leaked bridge is a failure, not something to clean up quietly (#733 F5).

    Ordering, stated because the whole fixture depends on it: this is autouse
    at session scope, so it is SET UP before the non-autouse `_shared_bridge`
    (autouse names lead the fixture closure) and therefore finalized AFTER it.
    The shared bridge is untouched while the session runs, and one that
    survived its own `session stop` is correctly reported as a leak. If the
    ordering ever inverts, the symptom is a false leak report naming the shared
    bridge's instance id -- fix that by passing its id into the sweep as an
    exclusion, not by dropping this fixture.

    The basetemp is THIS worker's (`getbasetemp()` returns the per-xdist-worker
    directory), so one worker cannot fail another's run.
    """
    yield
    leaks = bn_agent_leaks(tmp_path_factory.getbasetemp())
    if leaks:
        pytest.fail(
            "the suite leaked a headless bridge; it was reaped, but a leak is a "
            "defect:\n" + "\n".join(reap_bn_agent_leaks(leaks)),
            pytrace=False,
        )
