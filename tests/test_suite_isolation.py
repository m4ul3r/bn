"""Regression tests for the suite-wide hermeticity fixture (#589).

The only quality gate in this repo is a green `uv run pytest`, so the gate
itself has to be deterministic: it must not depend on the developer's shell
(``FORCE_COLOR`` makes stdlib argparse colorize usage text and breaks eight
assertions), and it must never read or write the developer's real
``~/.cache/bn`` state (sticky session pins, instance registries).

`tests/conftest.py` installs an autouse `_hermetic_env` fixture; these tests
assert that it is actually in force, that it can still be deliberately
overridden by a test that needs to (no over-correction), and that two tests
cannot observe each other's cache state.
"""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import threading
from pathlib import Path

import bn.cli
import pytest
from _pytest.outcomes import Failed, Skipped
from bn import headless, paths, session_state


# --- 1. color determinism -------------------------------------------------

def test_color_env_is_pinned_regardless_of_developer_shell():
    assert os.environ.get("NO_COLOR") == "1"
    for var in ("FORCE_COLOR", "CLICOLOR_FORCE", "PYTHON_COLORS"):
        assert var not in os.environ, f"{var} leaked into the test environment"


def test_bn_taint_models_env_does_not_leak_into_test_environment():
    # #615 review F6: the CLI now reads BN_TAINT_MODELS directly per invocation
    # (dataflow.py), so a developer/CI shell that exports it must not leak into
    # the suite -- same hermeticity class as the color vars above. Without the
    # _hermetic_env delenv, `BN_TAINT_MODELS=/tmp/definitely-missing-models.json
    # uv run pytest tests/test_dataflow.py -q` fails 4 unrelated tests.
    assert "BN_TAINT_MODELS" not in os.environ


#: Set by the polluted child run below to the path of a receipt the child must
#: write. Deliberately NOT a scrubbed name: it is how the parent proves the
#: child inherited an environment at all.
_POLLUTION_RECEIPT = "BN_HERMETICITY_RECEIPT"


def test_a_session_scoped_fixture_sees_a_scrubbed_environment(session_scope_environment):
    """The scrub must be in force at SESSION scope, not only per test.

    Pytest builds session-scoped fixtures first, so anything that spawns a
    process there -- the shared bridge, the fixture build -- used to capture
    `os.environ` before a single variable had been scrubbed. A bridge outlives
    the test that started it, so an overlay inherited there can never be taken
    back by a later `delenv`, and the taint lane then ran against whatever the
    developer's shell exported (#730 review).

    On a clean shell this passes for free, so
    `test_the_session_scope_scrub_survives_a_polluted_shell` below runs it
    again in a child whose shell is deliberately dirty. In that child the
    receipt branch records what actually arrived, which is what keeps the
    parent's green from being a green about nothing.
    """
    from conftest import SCRUBBED_ENV_VARS

    leaked = [var for var in SCRUBBED_ENV_VARS if var in session_scope_environment]
    assert not leaked, f"{leaked} reached a session-scoped fixture"
    assert session_scope_environment.get("NO_COLOR") == "1"

    receipt = os.environ.get(_POLLUTION_RECEIPT)
    if receipt:
        # Inside the polluted child: the unscrubbed probe name survived, so an
        # environment WAS inherited, and the scrubbed names did not survive it.
        still_set = [var for var in SCRUBBED_ENV_VARS if var in os.environ]
        Path(receipt).write_text("probe arrived", encoding="utf-8")
        assert not still_set, f"{still_set} survived into the child's test environment"


def test_the_session_scope_scrub_survives_a_polluted_shell(tmp_path):
    """...and the assertion above is not vacuous: run it in a child pytest whose
    environment carries every variable the suite refuses to inherit.

    `BN_TAINT_MODELS` is the one with teeth -- an unparseable overlay is what
    failed the taint lane against the shared bridge -- so it points at one
    here. The receipt is the differential: a child that silently inherited
    nothing would report the same green without writing it.
    """
    from conftest import SCRUBBED_ENV_VARS

    bogus_models = tmp_path / "models.json"
    bogus_models.write_text("{ not json", encoding="utf-8")
    receipt = tmp_path / "receipt.txt"
    env = dict(os.environ)
    env.update({var: "1" for var in SCRUBBED_ENV_VARS})
    env["BN_TAINT_MODELS"] = str(bogus_models)
    env[_POLLUTION_RECEIPT] = str(receipt)

    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        ["uv", "run", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/test_suite_isolation.py::test_a_session_scoped_fixture_sees_a_scrubbed_environment"],
        cwd=repo, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert receipt.is_file(), (
        "the child never took the receipt branch, so it did not inherit the "
        f"pollution this test exists to defeat: {proc.stdout}")


def test_argparse_usage_text_is_never_colorized():
    """Positive control: the bug (#589) was ANSI codes in usage/help text."""
    help_text = bn.cli.build_parser().format_help()
    assert "\x1b[" not in help_text
    assert "usage: bn" in help_text


def test_a_test_may_still_opt_into_color(monkeypatch):
    """Negative control: the fixture pins the default, it does not hard-disable
    argparse colorization for a test that deliberately asks for it."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "3")
    assert "\x1b[" in bn.cli.build_parser().format_help()


def test_color_override_is_restored_to_the_isolated_state():
    """Runs after the override test above: monkeypatch put the pin back."""
    assert os.environ.get("NO_COLOR") == "1"
    assert "FORCE_COLOR" not in os.environ


# --- 2. cache / session isolation ----------------------------------------

def test_cache_root_is_isolated_from_real_user_state():
    root = paths.cache_home()
    assert os.environ.get("BN_CACHE_DIR") == str(root)
    assert root.is_dir()
    real = Path.home() / ".cache" / "bn"
    assert real not in (root, *root.parents)


def _real_cache_home(monkeypatch) -> Path:
    """The cache root the fixture is shielding us from, computed the same way
    `paths.cache_home()` would on this platform with the pin absent."""
    with monkeypatch.context() as m:
        m.delenv("BN_CACHE_DIR", raising=False)
        return paths.cache_home()


def test_session_state_cannot_read_real_sticky_pins(monkeypatch):
    """The real repo may carry a sticky `bn target use` pin; a test must not see
    it, and a test's own pin must not land in the developer's real cache."""
    isolated = paths.session_state_path()
    real = _real_cache_home(monkeypatch) / "sessions" / isolated.name

    # Discriminating: with the fixture reverted these two are the same file.
    assert isolated != real
    assert Path.home() not in isolated.parents
    assert not isolated.exists(), "a sticky pin leaked into the isolated cache"

    # A developer who has ever run `bn target use` here legitimately HAS a real
    # session file for this project, so assert it is UNTOUCHED rather than
    # absent. Asserting absence fails on any such working copy and passes only
    # while unrelated test pollution perturbs the project-root hash that names
    # this file -- i.e. for the wrong reason, and only in whole-suite order.
    before = real.read_bytes() if real.exists() else None

    session_state.update(target="isolation-probe")
    assert isolated.exists()
    assert session_state.read()["target"] == "isolation-probe"

    after = real.read_bytes() if real.exists() else None
    assert after == before, "a test's sticky pin reached the developer's real cache"
    assert after is None or b"isolation-probe" not in after


_SEEN_ROOTS: list[Path] = []


@pytest.mark.parametrize("name", ["first", "second"])
def test_two_tests_cannot_observe_each_others_cache_state(monkeypatch, name):
    root = paths.cache_home()
    # Assert isolation BEFORE mutating: if the fixture regresses, `root` is the
    # developer's real ~/.cache/bn and this test would otherwise write into it.
    assert root != _real_cache_home(monkeypatch)
    assert Path.home() not in root.parents
    assert root not in _SEEN_ROOTS, "cache root shared between tests"
    _SEEN_ROOTS.append(root)
    leaked = list(root.rglob("leaked-*"))
    assert leaked == [], f"state leaked from a previous test: {leaked}"
    (root / f"leaked-{name}").write_text(name)


@pytest.mark.no_cache_isolation
def test_a_test_may_opt_out_of_the_cache_pin():
    """Negative control for the opt-out branch in conftest's `_hermetic_env`:
    a marked test sees no `BN_CACHE_DIR` at all, so it can exercise
    `paths.cache_home()`'s platform-default selection (which the pin
    short-circuits). Unmarked tests get the pin -- see the tests above."""
    assert "BN_CACHE_DIR" not in os.environ
    assert paths.cache_home() == _platform_default_cache_home()


def _platform_default_cache_home() -> Path:
    system = platform.system()
    home = Path.home()
    if system == "Darwin":
        return home / "Library" / "Caches" / "bn"
    if system == "Windows" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "bn"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "bn"
    return home / ".cache" / "bn"


def test_a_test_may_still_point_bn_cache_dir_elsewhere(monkeypatch, tmp_path):
    """Negative control: the fixture must not defeat an explicit per-test pin
    (tests/test_transport.py does this dozens of times)."""
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv("BN_CACHE_DIR", str(other))
    assert paths.cache_home() == other


def test_cache_override_is_restored_to_the_isolated_state():
    assert os.environ.get("BN_CACHE_DIR") == str(paths.cache_home())
    assert paths.cache_home().is_dir()


# --- #590: the real-BN integration lane must deploy itself ----------------
#
# The 27 `test_integration.py` tests gated on `HELLO_BINARY.exists()`, and
# nothing in a pytest run built the fixtures -- so a fresh checkout with BN
# installed reported "27 skipped, exit 0" and the skip was indistinguishable
# from a pass. The generated binaries stay untracked; conftest builds them.

import conftest


def _cc_available() -> bool:
    return shutil.which("cc") is not None and shutil.which("make") is not None


def test_generated_fixtures_stay_untracked():
    """Negative control for the fix's shape: the answer to #590 is to build
    the binaries, never to commit them."""
    tracked = subprocess.run(
        ["git", "ls-files", "tests/fixtures"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True, check=True,
    ).stdout.split()
    for name in conftest.REQUIRED_INTEGRATION_FIXTURES:
        assert f"tests/fixtures/{name}" not in tracked


def test_real_bn_gate_ignores_generated_fixture_existence():
    """The bug: the module gate conflated "no BN" with "fixtures not built".
    Availability is a property of the BN install alone."""
    src = (Path(__file__).parent / "test_integration.py").read_text()
    gate = src.split("pytestmark")[1].split("\n\n")[0]
    for name in conftest.REQUIRED_INTEGRATION_FIXTURES:
        assert name not in gate, f"module gate still skips on {name} existence"
    assert "real_bn" in gate


def _marks_of(module_name: str) -> list:
    module = importlib.import_module(module_name)
    marks = getattr(module, "pytestmark", [])
    return list(marks) if isinstance(marks, (list, tuple)) else [marks]


def test_test_integration_requires_the_build_fixture():
    """The real-BN lane must actually *apply* the build fixture. Asserting on
    the applied mark, not on the source text: the module mentions
    "integration_fixtures" in prose too, so a grep passes even if the
    `usefixtures` mark is deleted and the binaries are never built."""
    usefixtures = [m for m in _marks_of("test_integration") if m.name == "usefixtures"]
    assert usefixtures, "test_integration.py no longer applies any usefixtures mark"
    assert any("integration_fixtures" in m.args for m in usefixtures)


def test_real_bn_lanes_go_through_the_shared_gate():
    """Both real-BN lanes carry the `real_bn` marker, so BN_REQUIRE_REAL_TESTS
    reaches them and neither can skip silently (#590). test_taint_integration
    used a bare module-level skipif with a hardcoded BN path."""
    for module_name in ("test_integration", "test_taint_integration"):
        names = [m.name for m in _marks_of(module_name)]
        assert "real_bn" in names, f"{module_name} is not on the shared real-BN gate"
        assert "skipif" not in names, (
            f"{module_name} still has its own skipif, which bypasses strict mode"
        )


def test_bn_discovery_follows_the_cli_platform_defaults(monkeypatch, tmp_path):
    """The gate must not re-derive BN discovery more narrowly than `bn` itself.
    A copy that only knew `/opt/binaryninja` would report BN absent on a Darwin
    host where the CLI finds it -- 27 tests silently skipped, exit 0 (#590).
    Patching `bn.headless`'s own table is what proves the reuse."""
    install = tmp_path / "Binary Ninja.app" / "Contents" / "Resources"
    (install / "python").mkdir(parents=True)

    monkeypatch.delenv("BN_INSTALL_DIR", raising=False)
    monkeypatch.setattr(headless.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(headless, "_DEFAULT_BN_DIRS", {"Darwin": [str(install)]})

    assert conftest.real_bn_available()
    assert conftest.bn_python_dir() == install / "python"


def test_bn_install_dir_override_is_authoritative(monkeypatch, tmp_path):
    """Negative control for the above: reusing the CLI's discovery must not
    inherit its env-var-is-a-hint fallback. `BN_INSTALL_DIR` pointing nowhere
    means "absent" -- it must never silently fall through to a platform-default
    install, or a lane pinned at one BN would test whichever other one exists
    (and the gate's own repro would run the real lane instead of failing)."""
    real = tmp_path / "real"
    (real / "python").mkdir(parents=True)
    monkeypatch.setattr(headless.platform, "system", lambda: "Linux")
    monkeypatch.setattr(headless, "_DEFAULT_BN_DIRS", {"Linux": [str(real)]})

    monkeypatch.setenv("BN_INSTALL_DIR", str(tmp_path / "nonexistent"))
    assert not conftest.real_bn_available()

    monkeypatch.setenv("BN_INSTALL_DIR", str(real))
    assert conftest.bn_python_dir() == real / "python"


@pytest.mark.skipif(not _cc_available(), reason="cc/make not available")
def test_building_fixtures_produces_every_required_binary(tmp_path):
    """Positive: the autobuild really produces the whole set, so a fresh
    checkout's integration lane has something to run against.

    Built into `tmp_path`, not the working tree: the no-BN unit lane has no use
    for the binaries, and #589's hermeticity means it must not create them."""
    built = conftest.build_integration_fixtures(out_dir=tmp_path)
    assert {p.name for p in built} == set(conftest.REQUIRED_INTEGRATION_FIXTURES)
    for path in built:
        assert path.parent == tmp_path
        assert path.is_file() and path.stat().st_size > 0


def _plant_sidecar(binary: Path, *, offset: float) -> Path:
    """A stand-in `<binary>.bndb` whose mtime sits *offset* seconds from
    *binary*'s, which is the only thing the invalidation reads."""
    sidecar = Path(str(binary) + ".bndb")
    sidecar.write_bytes(b"stand-in for a saved analysis database")
    stamp = binary.stat().st_mtime + offset
    os.utime(sidecar, (stamp, stamp))
    return sidecar


@pytest.mark.skipif(not _cc_available(), reason="cc/make not available")
def test_a_bndb_older_than_its_binary_is_dropped_by_the_build(tmp_path):
    """#717: the bridge loads an adjacent `<binary>.bndb` in preference to the
    binary itself, so a database left behind by an earlier build of a
    since-changed fixture makes the real-BN lane assert about the OLD program --
    a failure indistinguishable from a read-path regression, and one that
    survives `make clean && make` on a long-lived checkout.

    The observable outcome is that the stale database is GONE once the build
    has run, so the next load falls through to the freshly compiled binary."""
    built = conftest.build_integration_fixtures(out_dir=tmp_path)
    stale = _plant_sidecar(built[0], offset=-60)
    assert stale.is_file()

    conftest.build_integration_fixtures(out_dir=tmp_path)

    assert not stale.exists(), (
        "a .bndb older than the binary it caches survived the build, so the "
        "real-BN lane would analyse the stale database instead of the fixture"
    )


@pytest.mark.skipif(not _cc_available(), reason="cc/make not available")
def test_a_bndb_newer_than_its_binary_survives_the_build(tmp_path):
    """The other half of #717's conditional, and the reason it has to BE a
    conditional: a database that post-dates its binary is the legitimate saved
    analysis OF that binary. Invalidating unconditionally would discard real
    work on every run and turn a clean re-run into a rebuild instead of a
    noop -- a regression wearing the fix's clothes."""
    built = conftest.build_integration_fixtures(out_dir=tmp_path)
    fresh = [_plant_sidecar(binary, offset=60) for binary in built]

    conftest.build_integration_fixtures(out_dir=tmp_path)

    assert [s.name for s in fresh if not s.is_file()] == [], (
        "the build invalidated a .bndb that post-dates its binary, so a clean "
        "re-run is not a noop and saved analysis is discarded every time"
    )


@pytest.mark.skipif(not _cc_available(), reason="cc/make not available")
def test_make_clean_removes_the_saved_analysis_databases(tmp_path):
    """#717's other half, in the Makefile: `clean` used to remove only the
    binaries, so the documented "run make clean" recovery left every `.bndb`
    in place and the stale analysis survived the one command a developer
    reaches for. Without `$(BNDBS)` in the `clean` target this is RED."""
    built = conftest.build_integration_fixtures(out_dir=tmp_path)
    planted = [_plant_sidecar(binary, offset=60) for binary in built]

    subprocess.run(
        ["make", "-C", str(conftest.FIXTURES_DIR), f"OUTDIR={tmp_path}", "clean"],
        capture_output=True, text=True, check=True,
    )

    assert [s.name for s in planted if s.exists()] == [], (
        "make clean left saved analysis databases behind, so the documented "
        "recovery from a stale fixture database does not actually clear it"
    )


@pytest.mark.skipif(not _cc_available(), reason="cc/make not available")
def test_missing_compiler_raises_before_invoking_make(tmp_path):
    """Positive: BN present + no toolchain must be a loud error, not a skip.
    This is the pre-check branch (the named CC is not on PATH at all)."""
    with pytest.raises(conftest.FixtureBuildError) as excinfo:
        conftest.build_integration_fixtures(
            make_env={"CC": "definitely-not-a-compiler"}, out_dir=tmp_path)
    message = str(excinfo.value)
    assert "make -C tests/fixtures" in message
    assert "definitely-not-a-compiler" in message


@pytest.mark.skipif(not _cc_available(), reason="cc/make not available")
def test_make_failure_raises_with_actionable_diagnostics(tmp_path):
    """Positive: the *other* failure branch -- a CC that exists but cannot
    compile -- must surface make's diagnostics. `/bin/false` reaches make (the
    which() pre-check passes), so this covers the rich branch the pre-check
    test above never gets to."""
    with pytest.raises(conftest.FixtureBuildError) as excinfo:
        conftest.build_integration_fixtures(
            make_env={"CC": "/bin/false"}, out_dir=tmp_path)
    message = str(excinfo.value)
    assert "make -C tests/fixtures" in message
    assert "exit status:" in message
    # The whole set is missing and every name is named -- that list is the
    # actionable part, and it is what a bare `returncode != 0` check would drop.
    for name in conftest.REQUIRED_INTEGRATION_FIXTURES:
        assert name in message
    assert not list(tmp_path.glob("*_x86_64"))


@pytest.mark.skipif(not _cc_available(), reason="cc/make not available")
def test_build_timeout_raises_fixture_build_error(monkeypatch, tmp_path):
    """Negative control on the error contract: a slow/wedged make must still
    come out as FixtureBuildError. A raw TimeoutExpired escapes the session
    fixture with no guidance and past every caller that catches the documented
    type."""
    def timing_out(env, out_dir):
        raise subprocess.TimeoutExpired(cmd=["make"], timeout=300,
                                        output=b"partial", stderr=b"boom")

    monkeypatch.setattr(conftest, "_run_fixture_make", timing_out)
    with pytest.raises(conftest.FixtureBuildError) as excinfo:
        conftest.build_integration_fixtures(out_dir=tmp_path)
    assert "timed out" in str(excinfo.value)
    assert "make -C tests/fixtures" in str(excinfo.value)


@pytest.mark.skipif(not _cc_available(), reason="cc/make not available")
def test_unremovable_stale_bndb_raises_fixture_build_error(tmp_path):
    """The same error contract over #717's invalidation: a stale database the
    build cannot remove is precisely the case the lane must NOT proceed from,
    because proceeding loads it. So it has to surface as FixtureBuildError --
    the documented type every caller catches -- not as a raw OSError.

    A directory sitting at the `<binary>.bndb` path is a real filesystem state
    that `stat()`s like a stale sidecar and makes `unlink()` raise, so this
    needs no monkeypatching of the code under test."""
    built = conftest.build_integration_fixtures(out_dir=tmp_path)
    blocked = Path(str(built[0]) + ".bndb")
    blocked.mkdir()
    stamp = built[0].stat().st_mtime - 60
    os.utime(blocked, (stamp, stamp))

    with pytest.raises(conftest.FixtureBuildError) as excinfo:
        conftest.build_integration_fixtures(out_dir=tmp_path)
    message = str(excinfo.value)
    assert blocked.name in message
    assert "make -C tests/fixtures" in message


@pytest.mark.skipif(not _cc_available(), reason="cc/make not available")
def test_concurrent_builds_are_serialized(tmp_path):
    """Fixture generation has one owner: parallel pytest workers must not race
    the same output files.

    The instrumented runner does not return until either all four threads are
    inside it or a dwell elapses, so overlap is *forced*, not hoped for. The
    original version returned as fast as `make` (a sub-100ms no-op once the
    binaries exist), so four staggered threads could serialize by luck and the
    test passed with the locking deleted.

    Without the lock all four sit in the runner together -> peak == 4 -> fail.
    With it, each waits out the dwell alone -> peak == 1."""
    workers = 4
    dwell = 0.5
    all_entered = threading.Event()
    lock = threading.Lock()
    peak = 0
    in_flight = 0

    # Build once for real so the post-build existence check passes for everyone.
    conftest.build_integration_fixtures(out_dir=tmp_path)

    def instrumented(env, out_dir):
        nonlocal peak, in_flight
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
            if in_flight == workers:
                all_entered.set()
        try:
            all_entered.wait(timeout=dwell)
            return subprocess.CompletedProcess([], 0, "", "")
        finally:
            with lock:
                in_flight -= 1

    real_run = conftest._run_fixture_make
    conftest._run_fixture_make = instrumented
    try:
        threads = [threading.Thread(
            target=conftest.build_integration_fixtures, kwargs={"out_dir": tmp_path})
            for _ in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)
            assert not t.is_alive(), "a build thread deadlocked on the build lock"
    finally:
        conftest._run_fixture_make = real_run

    assert peak == 1, f"{peak} builds ran concurrently -- the build lock is not held"


def test_bn_absence_skips_by_default_but_fails_in_strict_mode(monkeypatch):
    """Positive + negative control for the strict gate: absence is a visible
    skip by default, and `BN_REQUIRE_REAL_TESTS=1` turns it into a failure so
    a licensed lane cannot report green without running."""
    monkeypatch.setattr(conftest, "real_bn_available", lambda: False)

    monkeypatch.delenv("BN_REQUIRE_REAL_TESTS", raising=False)
    with pytest.raises(Skipped):
        conftest.require_real_bn()

    monkeypatch.setenv("BN_REQUIRE_REAL_TESTS", "1")
    with pytest.raises(Failed) as excinfo:
        conftest.require_real_bn()
    assert "BN_REQUIRE_REAL_TESTS" in str(excinfo.value)


def test_strict_mode_does_not_fail_when_bn_is_present(monkeypatch):
    """Negative control: strict mode must not turn a working lane red."""
    monkeypatch.setattr(conftest, "real_bn_available", lambda: True)
    monkeypatch.setenv("BN_REQUIRE_REAL_TESTS", "1")
    conftest.require_real_bn()


def test_an_unfixable_skip_is_a_failure_in_strict_mode(monkeypatch):
    """The strict gate covers skips no machine can un-skip, not just an absent
    BN: `test_transport.py`'s two directory-mode permission tests cannot run as
    root, so on a root CI container they vanish for good unless strict mode
    turns them red."""
    reason = "root ignores the directory mode this test relies on"

    monkeypatch.delenv("BN_REQUIRE_REAL_TESTS", raising=False)
    with pytest.raises(Skipped):
        conftest.refuse_silent_skip(reason)

    monkeypatch.setenv("BN_REQUIRE_REAL_TESTS", "1")
    with pytest.raises(Failed) as excinfo:
        conftest.refuse_silent_skip(reason)
    assert reason in str(excinfo.value)
    # The message names the knob that produced the failure, so a reader who did
    # not set it knows where it came from; the remedy rides in the reason.
    assert conftest.STRICT_ENV_VAR in str(excinfo.value)
