"""Real-BN lane for Binary Ninja project (`.bnpr`) support (BNPR_PLAN.md §6 row 8).

Projects are a Commercial/Ultimate feature (F10), so beyond the usual `real_bn`
gate this module checks `binaryninja.core_product()` and refuses to skip
silently under `BN_REQUIRE_REAL_TESTS=1` when the edition lacks them.

Project ownership is bridge-global (§3: one project per bridge instance), so the
lane does NOT use `shared_bn`: it spawns one private headless instance per
module, with its own `BN_CACHE_DIR`, and creates the project with the BN API in
a tmp dir. Everything after that goes through the CLI (`bn project ...`) when
the command group is registered, else over the wire with the same ops -- the
`Lane` helper hides which, so every assertion below is about the §5 contracts.

Each test imports the fixture binary into its OWN project folder and
(re)opens the project idempotently, so the module stays correct when xdist
hands its tests to different workers (each worker gets its own bridge).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterator

import pytest
from conftest import bn_python_dir, refuse_silent_skip, require_real_bn

from bn.transport import BridgeError

FIXTURES_DIR = Path(__file__).parent / "fixtures"
HELLO_BINARY = FIXTURES_DIR / "hello_x86_64"

pytestmark = [
    pytest.mark.real_bn,
    pytest.mark.usefixtures("integration_fixtures"),
]

_BN_CLI = [str(Path(sys.executable).parent / "bn")]
#: Spawn + BN import + full analysis of the small fixture.
_SESSION_START_TIMEOUT = 120.0
_LOAD_TIMEOUT = 120.0
_SESSION_STOP_TIMEOUT = 60.0
#: restart = stop + spawn + project_open + full analysis of each replayed target.
_RESTART_TIMEOUT = 240.0

_PROJECT_EDITIONS = ("Commercial", "Ultimate")


def _bn_api_python(code: str, *, timeout: float = 120.0) -> str:
    """Run *code* against the real `binaryninja` module in a subprocess.

    Never imported into the pytest process: the mocked suite installs a fake
    `binaryninja` module, and the real one pulls the BN core into the test
    runner for the rest of the run.
    """
    python_dir = bn_python_dir()
    assert python_dir is not None, "require_real_bn() must have passed before this"
    env = dict(os.environ)
    env["BN_DISABLE_USER_PLUGINS"] = "1"
    res = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(python_dir)!r})\n"
         "import binaryninja\n"
         "binaryninja.disable_default_log()\n" + code],
        capture_output=True, text=True, timeout=timeout, env=env,
    )
    assert res.returncode == 0, f"BN API helper failed: {res.stderr}\n{res.stdout}"
    return res.stdout.strip()


def _require_project_edition() -> None:
    product = _bn_api_python("print(binaryninja.core_product())")
    if not any(edition in product for edition in _PROJECT_EDITIONS):
        refuse_silent_skip(
            f"Binary Ninja projects need a Commercial or Ultimate licence; this install "
            f"reports {product!r} -- the project lane did not run"
        )


class Lane:
    """One private bridge + one `.bnpr`, driven through the CLI or the wire."""

    def __init__(self, instance_id: str, cache_dir: Path, project_path: Path,
                 cli_available: bool) -> None:
        self.instance_id = instance_id
        self.cache_dir = cache_dir
        self.project_path = project_path
        self.cli_available = cli_available

    # -- process plumbing ------------------------------------------------------

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["BN_CACHE_DIR"] = str(self.cache_dir)
        env["BN_DISABLE_USER_PLUGINS"] = "1"
        return env

    def bn(self, *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
        """`bn ...` with no instance routing (lifecycle commands)."""
        return subprocess.run(
            [*_BN_CLI, *args], capture_output=True, text=True, timeout=timeout, env=self.env(),
        )

    def run(self, *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
        """`bn -i <lane instance> ...`; the returncode is the caller's to assert."""
        return self.bn("-i", self.instance_id, *args, timeout=timeout)

    def json(self, *args: str, timeout: float = 60.0) -> Any:
        res = self.run(*args, "--format", "json", timeout=timeout)
        assert res.returncode == 0, f"bn {' '.join(args)} failed: {res.stderr}\n{res.stdout}"
        return json.loads(res.stdout)

    def wire(self, op: str, params: dict[str, Any] | None = None, *,
             target: str | None = None, instance_id: str | None = None,
             timeout: float = _LOAD_TIMEOUT) -> Any:
        """Send *op* straight to the bridge; refusals surface as `BridgeError`."""
        from bn.transport import send_request, unwrap_result
        response = send_request(
            op, params=params or {}, target=target, timeout=timeout,
            instance_id=instance_id or self.instance_id,
        )
        return unwrap_result(response, op)

    def _cli_or_wire(self, cli_args: list[str], op: str, params: dict[str, Any], *,
                     timeout: float = 60.0) -> Any:
        """Prefer `bn project <sub>`; fall back to the op it wraps (§5.6).

        In CLI mode a refusal is exit 2 with the bridge's text on stderr; it is
        re-raised as `BridgeError` so callers assert one shape in both modes.
        """
        if not self.cli_available:
            return self.wire(op, params, timeout=timeout)
        res = self.run(*cli_args, "--format", "json", timeout=timeout)
        if res.returncode == 2:
            raise BridgeError(res.stderr.strip() or res.stdout.strip())
        assert res.returncode == 0, f"bn {' '.join(cli_args)} failed: {res.stderr}\n{res.stdout}"
        return json.loads(res.stdout)

    # -- project ops -----------------------------------------------------------

    def project_info(self) -> dict[str, Any]:
        return self._cli_or_wire(["project", "info"], "project_info", {})

    def project_files(self, folder: str | None = None) -> list[dict[str, Any]]:
        args = ["project", "files"] + (["--folder", folder] if folder else [])
        params: dict[str, Any] = {"folder": folder} if folder else {}
        result = self._cli_or_wire(args, "project_files", params)
        return result["files"] if isinstance(result, dict) else result

    def project_import(self, path: Path, folder: str) -> dict[str, Any]:
        return self._cli_or_wire(
            ["project", "import", str(path), "--folder", folder],
            "project_import", {"path": str(path), "folder": folder},
        )

    def project_load(self, selector: str, *, prefer_bndb: bool = True) -> dict[str, Any]:
        args = ["project", "load", selector] + ([] if prefer_bndb else ["--no-bndb"])
        return self._cli_or_wire(
            args, "project_load", {"selector": selector, "prefer_bndb": prefer_bndb},
            timeout=_LOAD_TIMEOUT,
        )

    def project_close(self, *, force: bool = False) -> dict[str, Any]:
        args = ["project", "close"] + (["--force"] if force else [])
        return self._cli_or_wire(args, "project_close", {"force": force})

    def ensure_open(self) -> dict[str, Any]:
        """Open the lane's project if this bridge does not hold it already.

        Always the wire op: `bn project open` attaches to a published owner
        instead of sending `project_open`, and the lane wants the bridge-side
        `opened` flag (§5.2) regardless of pins.
        """
        return self.wire("project_open", {"path": str(self.project_path)})

    def ensure_imported(self, folder: str) -> dict[str, Any]:
        """Import the fixture into *folder* once; returns the raw file row."""
        path = f"/{folder}/{HELLO_BINARY.name}"
        for row in self.project_files():
            if row["path"] == path:
                return row
        return self.project_import(HELLO_BINARY, folder)

    # -- target helpers --------------------------------------------------------

    def target_info(self, selector: str) -> dict[str, Any]:
        return self.json("target", "info", "-t", selector)

    def target_rows(self) -> list[dict[str, Any]]:
        return self.json("target", "list")["items"]

    def function_names(self, selector: str) -> set[str]:
        listing = self.json("function", "list", "--limit", "500", "-t", selector)
        items = listing.get("items") if isinstance(listing, dict) else listing
        return {f["name"] for f in items}

    def rename(self, selector: str, old: str, new: str) -> None:
        res = self.run("rename", "-t", selector, old, new)
        assert res.returncode == 0, f"rename {old} -> {new} failed: {res.stderr}\n{res.stdout}"

    def save(self, selector: str, *extra: str) -> subprocess.CompletedProcess[str]:
        return self.run("save", "-t", selector, *extra, "--format", "json", timeout=_LOAD_TIMEOUT)


def _start_instance(instance_id: str, env: dict[str, str]) -> None:
    started = subprocess.run(
        [*_BN_CLI, "session", "start", "--instance-id", instance_id, "--format", "json"],
        capture_output=True, text=True, timeout=_SESSION_START_TIMEOUT, env=env,
    )
    assert started.returncode == 0, f"bridge failed to start: {started.stderr}\n{started.stdout}"
    assert json.loads(started.stdout)["instance_id"] == instance_id


def _stop_instance(instance_id: str, env: dict[str, str]) -> None:
    subprocess.run(
        [*_BN_CLI, "session", "stop", instance_id, "--discard-unsaved"],
        capture_output=True, text=True, timeout=_SESSION_STOP_TIMEOUT, env=env,
    )


def _cache_dir_within_socket_budget(tmp_path_factory, instance_id: str) -> tuple[Path, bool]:
    """A `BN_CACHE_DIR` whose longest socket path `bind()` accepts here.

    pytest's basetemp on macOS (`/private/var/folders/<..>/T/pytest-of-<user>/
    pytest-N/`) alone is near the 103-byte `sun_path` budget, so the per-test
    cache the rest of the suite uses cannot hold a bridge socket there. Fall
    back to a short private dir under the system tmp root, which the caller
    removes; returns `(dir, is_fallback)`.
    """
    from bn.paths import socket_path_budget
    longest = f"instances/{instance_id}-b.sock"
    candidate = tmp_path_factory.mktemp("bnpr-lane-cache")
    if len(str(candidate / longest).encode("utf-8")) <= socket_path_budget():
        return candidate, False
    fallback = Path(tempfile.mkdtemp(prefix="bnpr-", dir="/tmp"))
    assert len(str(fallback / longest).encode("utf-8")) <= socket_path_budget(), fallback
    return fallback, True


@pytest.fixture(scope="module")
def _project_bridge(_hermetic_session_env, tmp_path_factory) -> Iterator[Lane]:
    require_real_bn()
    _require_project_edition()
    instance_id = f"bnpr-{uuid.uuid4().hex[:8]}"
    cache_dir, cache_is_fallback = _cache_dir_within_socket_budget(tmp_path_factory, instance_id)
    project_path = tmp_path_factory.mktemp("bnpr-lane-project") / "lane.bnpr"
    _bn_api_python(
        f"p = binaryninja.Project.create_project({str(project_path)!r}, 'lane')\n"
        "assert p is not None, 'create_project returned None'\n"
        "p.close()"
    )
    assert (project_path / "project.bnpm").is_file(), "project.bnpm was not created"

    env = dict(os.environ)
    env["BN_CACHE_DIR"] = str(cache_dir)
    env["BN_DISABLE_USER_PLUGINS"] = "1"
    cli_available = subprocess.run(
        [*_BN_CLI, "project", "--help"], capture_output=True, text=True, timeout=60.0, env=env,
    ).returncode == 0
    _start_instance(instance_id, env)
    lane = Lane(instance_id, cache_dir, project_path, cli_available)
    try:
        yield lane
    finally:
        _stop_instance(instance_id, env)
        if cache_is_fallback:
            shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture
def lane(_project_bridge, monkeypatch) -> Lane:
    """The module's bridge, with the in-process transport pointed at its cache."""
    monkeypatch.setenv("BN_CACHE_DIR", str(_project_bridge.cache_dir))
    monkeypatch.setenv("BN_DISABLE_USER_PLUGINS", "1")
    return _project_bridge


# --- tests --------------------------------------------------------------------


def test_open_info_and_reopen(lane: Lane):
    first = lane.ensure_open()
    assert first["name"] == "lane"
    assert first["path"] == str(lane.project_path)
    assert first["opened"] in (True, False)   # False when an earlier test opened it

    info = lane.project_info()
    assert info["name"] == "lane"
    assert info["path"] == str(lane.project_path)
    assert isinstance(info["file_count"], int) and isinstance(info["open_targets"], list)

    # Same canonical path again (the project.bnpm spelling, §5.1): a no-op success.
    again = lane.wire("project_open", {"path": str(lane.project_path / "project.bnpm")})
    assert again["opened"] is False
    assert again["path"] == str(lane.project_path)

    if lane.cli_available:
        # §5.6 step 1: the CLI sees the published owner and attaches to it.
        res = lane.run("project", "open", str(lane.project_path), "--format", "json")
        assert res.returncode == 0, res.stderr
        attached = json.loads(res.stdout)
        assert attached["attached"] is True
        assert attached["instance_id"] == lane.instance_id

    # Published for other bridges and `session restart` (§5.4).
    registry = json.loads((lane.cache_dir / "instances" / f"{lane.instance_id}.json").read_text())
    assert registry["project"] == {"path": str(lane.project_path), "id": info["id"], "name": "lane"}


def test_import_load_save_and_database_round_trip(lane: Lane, tmp_path):
    lane.ensure_open()
    raw = lane.ensure_imported("fw")
    assert raw["path"] == "/fw/hello_x86_64"
    assert raw["is_database"] is False
    assert raw["folder"] == "/fw"   # folder paths are path-in-project form (§5.1)
    rows = {row["path"]: row for row in lane.project_files()}
    assert rows["/fw/hello_x86_64"]["is_database"] is False
    if lane.cli_available:
        text = lane.run("project", "files")
        assert text.returncode == 0, text.stderr
        assert "raw" in text.stdout and "/fw/hello_x86_64" in text.stdout

    # Load the raw file: the target is addressable by path-in-project, bare name
    # and the leading-slash form (§5.3).
    loaded = lane.project_load("fw/hello_x86_64", prefer_bndb=False)
    assert loaded["project_file_id"] == raw["id"]
    assert loaded["project_path"] == "/fw/hello_x86_64"
    assert loaded["substituted_from"] is None
    row = next(r for r in lane.target_rows() if r.get("project_path") == "/fw/hello_x86_64")
    assert row["project_file_id"] == raw["id"]
    # The advertised selector is the shortest unique suffix of the path in the
    # project (bridge `_compute_selectors`); the longer forms resolve too.
    assert row["selector"] == "hello_x86_64"
    for selector in ("hello_x86_64", "fw/hello_x86_64", "/fw/hello_x86_64"):
        assert lane.target_info(selector)["target_id"] == row["target_id"], selector

    # First save of a raw project file creates a sibling <name>.bndb project
    # file (F3) and re-homes the target onto it (§5.2 save_database).
    # `greet` is `_greet` on Mach-O; pick it by suffix rather than by platform.
    greet = next(n for n in lane.function_names("fw/hello_x86_64") if n.lstrip("_") == "greet")
    lane.rename("fw/hello_x86_64", greet, "lane_renamed_once")
    assert lane.target_info("fw/hello_x86_64")["unsaved"] is True
    saved = lane.save("fw/hello_x86_64")
    assert saved.returncode == 0, f"{saved.stderr}\n{saved.stdout}"
    saved_result = json.loads(saved.stdout)
    assert saved_result["mode"] == "created"
    assert saved_result["project_path"] == "/fw/hello_x86_64.bndb"
    assert saved_result["project_file_id"] != raw["id"]
    rows = {r["path"]: r for r in lane.project_files("fw")}
    assert rows["/fw/hello_x86_64.bndb"]["is_database"] is True
    assert rows["/fw/hello_x86_64"]["sibling_database_id"] == saved_result["project_file_id"]
    if lane.cli_available:
        text = lane.run("project", "files", "--folder", "fw")
        assert "bndb" in text.stdout and "/fw/hello_x86_64.bndb" in text.stdout
    after = lane.target_info("hello_x86_64.bndb")
    assert after["target_id"] == row["target_id"]
    assert after["selector"] == "hello_x86_64.bndb"
    assert after["project_path"] == "/fw/hello_x86_64.bndb"
    assert after["unsaved"] is False

    # Second save: a snapshot into the same database (F5).
    lane.rename("hello_x86_64.bndb", "lane_renamed_once", "lane_renamed_twice")
    saved2 = lane.save("hello_x86_64.bndb")
    assert saved2.returncode == 0, f"{saved2.stderr}\n{saved2.stdout}"
    assert json.loads(saved2.stdout)["mode"] == "snapshot"

    # An explicit output path is refused for a project-backed target.
    refused = lane.save("hello_x86_64.bndb", str(tmp_path / "x.bndb"))
    assert refused.returncode == 2, f"{refused.returncode}: {refused.stdout}\n{refused.stderr}"
    assert "project-backed target" in refused.stderr
    assert "bn project export" in refused.stderr
    assert not (tmp_path / "x.bndb").exists()

    # Close refuses while a project target is dirty, naming it; --force discards.
    lane.rename("hello_x86_64.bndb", "lane_renamed_twice", "lane_renamed_discarded")
    with pytest.raises(BridgeError) as refusal:
        lane.project_close()
    assert "unsaved targets: hello_x86_64.bndb" in str(refusal.value)
    assert lane.target_info("hello_x86_64.bndb")["unsaved"] is True   # nothing was closed
    closed = lane.project_close(force=True)
    assert closed["closed"] is True
    assert closed["path"] == str(lane.project_path)
    # `targets_closed` carries `_close_binary` snapshots: the blob path inside
    # the project plus the discarded-work flag.
    assert len(closed["targets_closed"]) == 1
    snapshot = closed["targets_closed"][0]
    assert snapshot["unsaved"] is True
    assert Path(snapshot["path"]).is_relative_to(lane.project_path)
    assert not any(r.get("project_file_id") for r in lane.target_rows())
    with pytest.raises(BridgeError, match="No project is open"):
        lane.project_info()

    # Reopen; loading the RAW selector now substitutes the sibling database,
    # which holds the saved rename and not the discarded one.
    reopened = lane.ensure_open()
    assert reopened["opened"] is True
    reloaded = lane.project_load("fw/hello_x86_64")
    assert reloaded["substituted_from"] == raw["id"]
    assert reloaded["project_file_id"] == saved_result["project_file_id"]
    assert reloaded["project_path"] == "/fw/hello_x86_64.bndb"
    names = lane.function_names("hello_x86_64.bndb")
    assert "lane_renamed_twice" in names
    assert "lane_renamed_discarded" not in names
    assert "lane_renamed_once" not in names
    assert lane.target_info("hello_x86_64.bndb")["unsaved"] is False


def test_second_instance_cannot_open_an_owned_project(lane: Lane):
    lane.ensure_open()
    other = f"{lane.instance_id}-b"
    _start_instance(other, lane.env())
    try:
        with pytest.raises(BridgeError) as refusal:
            lane.wire("project_open", {"path": str(lane.project_path)}, instance_id=other)
        message = str(refusal.value)
        assert "already open in bridge instance" in message
        assert lane.instance_id in message
        assert "bn project attach" in message
        if lane.cli_available:
            # The CLI never reaches the guard: an explicit `-i other` still
            # attaches to the published owner (§5.6 step 1).
            res = lane.bn("-i", other, "project", "open", str(lane.project_path),
                          "--format", "json")
            assert res.returncode == 0, res.stderr
            assert json.loads(res.stdout) == {
                "attached": True, "instance_id": lane.instance_id,
                "path": str(lane.project_path),
            }
            attach = lane.bn("project", "attach", str(lane.project_path), "--format", "json")
            assert attach.returncode == 0, attach.stderr
            assert json.loads(attach.stdout)["instance_id"] == lane.instance_id
        # The other bridge is untouched: no project, no project targets.
        with pytest.raises(BridgeError, match="No project is open"):
            lane.wire("project_info", instance_id=other)
    finally:
        _stop_instance(other, lane.env())


def test_session_restart_replays_project_targets(lane: Lane):
    lane.ensure_open()
    raw = lane.ensure_imported("rs")
    loaded = lane.project_load("rs/hello_x86_64", prefer_bndb=False)
    assert loaded["project_path"] == "/rs/hello_x86_64"
    before = {r["project_path"]: r for r in lane.target_rows() if r.get("project_file_id")}
    assert "/rs/hello_x86_64" in before
    # Every project target of this bridge must be clean for a plain restart; the
    # lane's other tests leave theirs saved.
    assert not any(r.get("unsaved") for r in lane.target_rows()), lane.target_rows()
    registry = json.loads((lane.cache_dir / "instances" / f"{lane.instance_id}.json").read_text())
    assert {t["project_file_id"] for t in registry["project_targets"]} == {
        r["project_file_id"] for r in before.values()}

    res = lane.bn("session", "restart", lane.instance_id, "--format", "json",
                  timeout=_RESTART_TIMEOUT)
    assert res.returncode == 0, f"restart rc {res.returncode}: {res.stderr}\n{res.stdout}"
    payload = json.loads(res.stdout)
    assert payload["restarted"] is True and payload["instance_id"] == lane.instance_id
    assert payload["project"] == {"path": str(lane.project_path), "opened": True}
    replayed = {x.get("project_file_id") for x in payload["loaded"] if isinstance(x, dict)}
    assert raw["id"] in replayed
    assert not any(x.get("error") for x in payload["loaded"] if isinstance(x, dict)), payload

    # The new process owns the project and serves the same targets by project
    # identity (F2): project_path intact, the blob never loaded as a plain file.
    info = lane.project_info()
    assert info["path"] == str(lane.project_path)
    after = {r["project_path"]: r for r in lane.target_rows() if r.get("project_file_id")}
    assert set(after) == set(before)
    assert after["/rs/hello_x86_64"]["project_file_id"] == raw["id"]
    assert after["/rs/hello_x86_64"]["selector"] in ("hello_x86_64", "rs/hello_x86_64")
    assert all(r.get("project_file_id") for r in lane.target_rows()), lane.target_rows()
    assert lane.target_info("rs/hello_x86_64")["project_path"] == "/rs/hello_x86_64"
    registry = json.loads((lane.cache_dir / "instances" / f"{lane.instance_id}.json").read_text())
    assert registry["project"]["path"] == str(lane.project_path)
