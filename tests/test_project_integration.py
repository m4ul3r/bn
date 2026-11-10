"""Real-BN lane for Binary Ninja project (`.bnpr`) support.

Projects are gated by an isolated native create/close capability probe, not
the product label. Unsupported installs refuse to skip silently under
`BN_REQUIRE_REAL_TESTS=1`.

Project ownership is bridge-global (one project per bridge instance), so the
lane does NOT use `shared_bn`: it spawns one private headless instance per
module, with its own `BN_CACHE_DIR`, and creates the project with the BN API in
a tmp dir. Everything after that goes through the CLI (`bn project ...`) when
the command group is registered, else over the wire with the same ops -- the
`Lane` helper hides which, so each assertion covers the same public behavior.

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import pytest
from conftest import bn_python_dir, require_project_capability, require_real_bn

from bn.transport import BridgeError

FIXTURES_DIR = Path(__file__).parent / "fixtures"
HELLO_BINARY = FIXTURES_DIR / "hello_x86_64"

pytestmark = [
    pytest.mark.real_bn,
    pytest.mark.usefixtures("project_integration_fixtures"),
]

_BN_CLI = [str(Path(sys.executable).parent / "bn")]
#: Spawn + BN import + full analysis of the small fixture.
_SESSION_START_TIMEOUT = 120.0
_LOAD_TIMEOUT = 120.0
_SESSION_STOP_TIMEOUT = 60.0
#: restart = stop + spawn + project_open + full analysis of each replayed target.
_RESTART_TIMEOUT = 240.0


@pytest.fixture(scope="session")
def project_integration_fixtures(request: pytest.FixtureRequest) -> list[Path]:
    """Refuse unsupported native projects before invoking fixture toolchains."""
    require_project_capability()
    return request.getfixturevalue("integration_fixtures")


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
    """Keep the native-lane helper callable while probing actual capability."""
    require_project_capability()


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
        """Prefer `bn project <sub>`; fall back to the op it wraps.

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
        """The file rows in *folder*: the first page (#1080), which holds every
        row of a lane project -- they stay far below the default limit."""
        args = ["project", "files"] + (["--folder", folder] if folder else [])
        params: dict[str, Any] = {"folder": folder} if folder else {}
        result = self._cli_or_wire(args, "project_files", params)
        return result["items"] if isinstance(result, dict) else result

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
        `opened` flag regardless of pins.
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

    # Same canonical path again (the project.bnpm spelling): a no-op success.
    again = lane.wire("project_open", {"path": str(lane.project_path / "project.bnpm")})
    assert again["opened"] is False
    assert again["path"] == str(lane.project_path)

    if lane.cli_available:
        # The CLI sees the published owner and attaches to it.
        res = lane.run("project", "open", str(lane.project_path), "--format", "json")
        assert res.returncode == 0, res.stderr
        attached = json.loads(res.stdout)
        assert attached["attached"] is True
        assert attached["instance_id"] == lane.instance_id

    # Published for other bridges and `session restart`.
    registry = json.loads((lane.cache_dir / "instances" / f"{lane.instance_id}.json").read_text())
    assert registry["project"] == {"path": str(lane.project_path), "id": info["id"], "name": "lane"}


def test_import_load_save_and_database_round_trip(lane: Lane, tmp_path):
    lane.ensure_open()
    raw = lane.ensure_imported("fw")
    assert raw["path"] == "/fw/hello_x86_64"
    assert raw["is_database"] is False
    assert raw["folder"] == "/fw"   # folder paths use path-in-project form
    rows = {row["path"]: row for row in lane.project_files()}
    assert rows["/fw/hello_x86_64"]["is_database"] is False
    if lane.cli_available:
        text = lane.run("project", "files")
        assert text.returncode == 0, text.stderr
        assert "raw" in text.stdout and "/fw/hello_x86_64" in text.stdout

    # Load the raw file: the target is addressable by path-in-project, bare name
    # and the leading-slash form.
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
    # file and re-homes the target onto it.
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

    # Second save: a snapshot into the same database.
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
            # attaches to the published owner.
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
    # identity: project_path intact, the blob never loaded as a plain file.
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


# --- firmware rootfs projects -------------------------------------------------


@pytest.fixture
def rootfs_lane(rootfs_fixture, _hermetic_session_env, tmp_path, tmp_path_factory,
                monkeypatch) -> Iterator[tuple[Lane, Path]]:
    """A private instance and a copied tree: no project or BNDB enters the repo."""
    _require_project_edition()
    root = shutil.copytree(rootfs_fixture, tmp_path / "firmware", symlinks=True)
    root = root.resolve()
    instance_id = f"rootfs-{uuid.uuid4().hex[:8]}"
    cache_dir, cache_is_fallback = _cache_dir_within_socket_budget(tmp_path_factory, instance_id)
    monkeypatch.setenv("BN_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("BN_DISABLE_USER_PLUGINS", "1")
    lane = Lane(instance_id, cache_dir, root.with_suffix(".bnpr"), cli_available=True)
    lane.cli_available = lane.bn("project", "init", "--help").returncode == 0
    _start_instance(instance_id, lane.env())
    try:
        yield lane, root
    finally:
        _stop_instance(instance_id, lane.env())
        if cache_is_fallback:
            shutil.rmtree(cache_dir, ignore_errors=True)


def _rootfs_init(lane: Lane, root: Path, *, cwd: Path | None = None) -> dict[str, Any]:
    if lane.cli_available:
        args = ["project", "init", *([] if cwd else [str(root)]), "--format", "json"]
        result = subprocess.run(
            [*_BN_CLI, "-i", lane.instance_id, *args], cwd=cwd,
            capture_output=True, text=True, timeout=_LOAD_TIMEOUT, env=lane.env(),
        )
        assert result.returncode == 0, f"{result.stderr}\n{result.stdout}"
        return json.loads(result.stdout)
    # The bridge-only lane exercises the same create/open + sync operations.
    exists = lane.project_path.exists()
    if exists:
        opened = lane.ensure_open()
    else:
        opened = lane.wire("project_create", {
            "path": str(lane.project_path), "name": root.name,
            "rootfs": {"root": str(root), "platform": "linux"},
        })
    sync = lane.wire("project_sync_libs", {"root": str(root), "platform": "linux"})
    return {
        "created": not exists, "project_path": opened["path"],
        "platform": sync["platform"], "lookup_paths": sync["lookup_paths"], "sync": sync,
    }


def _rootfs_link(lane: Lane, selector: str, *, relink: bool = False) -> dict[str, Any]:
    args = ["project", "link", "-t", selector, "--relink" if relink else "--preview"]
    if lane.cli_available:
        return lane.json(*args)
    return lane.wire("project_link", {"dry_run": not relink, "relink": relink}, target=selector)


_ROOTFS_IMPORT_LIBRARIES = {
    "greet": "libgreet.so.1",
    "vendor_sum": "libgreet.so.1",
    "printf": "libc.so",
    "strdup": "libc.so",
    "strlen": "libc.so",
    "free": "libc.so",
}


def _assert_rootfs_links(lane: Lane, selector: str) -> dict[str, Any]:
    imports = lane.json("imports", "-t", selector, "--limit", "100")
    rows = imports["items"]
    by_name = {row["name"]: row for row in rows}
    assert _ROOTFS_IMPORT_LIBRARIES.keys() <= by_name.keys(), rows
    weak_hook = "firmware_startup_hook"
    assert weak_hook in by_name, rows
    assert by_name[weak_hook]["library"] is None, by_name[weak_hook]

    preview = _rootfs_link(lane, selector)
    # The imports reader currently attributes namespace/verneed, not project
    # ExternalLocations. The link preview exposes the actual location readback
    # even for unversioned musl symbols whose imports row still says null.
    locations = {row["symbol"]: row["library"] for row in preview["locations"]}
    for symbol, library in _ROOTFS_IMPORT_LIBRARIES.items():
        assert locations[symbol] == library, preview
        if by_name[symbol]["library"] is not None:
            assert by_name[symbol]["library"] == library, by_name[symbol]
    unresolved = {row["symbol"] for row in preview["unresolved"]}
    assert weak_hook in unresolved, preview
    assert all(symbol not in locations for symbol in unresolved), preview
    return preview


class TestRootfsProjects:
    def test_init_link_persistence_and_attach(self, rootfs_lane):
        lane, root = rootfs_lane
        initialized = _rootfs_init(lane, root)
        assert initialized["created"] is True
        assert initialized["project_path"] == str(root.parent / f"{root.name}.bnpr")
        assert (lane.project_path / "project.bnpm").is_file()
        assert initialized["platform"] == "linux"
        lookup = {row["path"]: row["source"] for row in initialized["lookup_paths"]}
        assert lookup["/opt/vendor/lib"].startswith("ld_so_conf:")
        assert "/lib" in lookup
        assert "/opt/extra/lib" not in lookup
        imported = initialized["sync"]["imported"]
        assert len(imported) == 2, imported
        assert {row["name"] for row in imported} == {"libc.so", "libgreet.so.1"}
        inventory = lane._cli_or_wire(["project", "libs"], "project_libs", {})["libs"]
        assert len(inventory) == 2, inventory
        libraries = {Path(row["rel_realpath"]).name: row for row in inventory}
        assert set(libraries) == {"libc.so", "libgreet.so.1"}
        assert any(name.startswith("ld-musl-") and name.endswith(".so.1")
                   for name in libraries["libc.so"]["names"])
        assert "libgreet.so" in libraries["libgreet.so.1"]["names"]
        assert not any("libstray.so" in row["names"] for row in inventory)

        # Init imports only loader-visible libraries. Executables are explicitly
        # imported into the project before project_load selects them.
        raw = lane.project_import(root / "usr/bin/app", "usr/bin")
        loaded = lane.project_load("usr/bin/app")
        assert loaded["project_file_id"] == raw["id"]
        assert loaded["link"]["linked"] >= 6, loaded
        assert lane.target_info("app")["unsaved"] is False
        _assert_rootfs_links(lane, "app")

        saved = lane.save("app")
        assert saved.returncode == 0, f"{saved.stderr}\n{saved.stdout}"
        saved_result = json.loads(saved.stdout)
        assert saved_result["project_path"] == "/usr/bin/app.bndb"
        closed = lane.run("close", "-t", "app.bndb")
        assert closed.returncode == 0, f"{closed.stderr}\n{closed.stdout}"
        reloaded = lane.project_load("usr/bin/app")
        assert reloaded["substituted_from"] == raw["id"]
        assert reloaded["project_file_id"] == saved_result["project_file_id"]
        assert reloaded["link"]["already_linked"] >= 6, reloaded
        assert reloaded["link"]["linked"] == 0, reloaded
        assert lane.target_info("app.bndb")["unsaved"] is False
        preview = _assert_rootfs_links(lane, "app.bndb")
        assert preview["already_linked"] >= 6
        assert lane.target_info("app.bndb")["unsaved"] is False
        relinked = _rootfs_link(lane, "app.bndb", relink=True)
        assert relinked["status"] in ("verified", "noop"), relinked

        reopened = _rootfs_init(lane, root, cwd=root / "usr/bin")
        assert reopened["created"] is False
        assert reopened["project_path"] == str(lane.project_path)
        assert reopened["sync"]["skipped_existing"] == 2
        assert reopened["sync"]["imported"] == []

        other = f"{lane.instance_id}-b"
        _start_instance(other, lane.env())
        try:
            if lane.cli_available:
                attached = lane.bn(
                    "-i", other, "project", "open", str(lane.project_path), "--format", "json",
                )
                assert attached.returncode == 0, f"{attached.stderr}\n{attached.stdout}"
                assert json.loads(attached.stdout) == {
                    "attached": True, "instance_id": lane.instance_id,
                    "path": str(lane.project_path),
                }
            else:
                with pytest.raises(BridgeError, match="already open in bridge instance"):
                    lane.wire("project_open", {"path": str(lane.project_path)}, instance_id=other)
            with pytest.raises(BridgeError, match="No project is open"):
                lane.wire("project_info", instance_id=other)
        finally:
            _stop_instance(other, lane.env())


def test_findings_review_conflicts_persist_and_stale_after_source_deletion(lane: Lane, tmp_path):
    """Actual project records survive reopen; concurrent reviews cannot lose a revision."""
    lane.ensure_open()
    source = tmp_path / "packet.bin"
    source.write_bytes(b"PKTA\x00payload")
    file = lane.project_import(source, "finding-review")
    document = {
        "type": "file-format", "kind": "observation",
        "claim": "The sample starts with the ASCII PKTA marker.",
        "scope": [{"file": file["id"]}],
        "evidence": [{"check": "bytes", "file": file["id"], "offset": 0, "length": 4,
                      "expected": {"hex": "504b5441"}}],
        "limitations": ["The marker alone does not establish parser behavior."],
        "next_steps": ["Inspect the parser before making stronger format claims."],
    }
    claim_path = tmp_path / "claim.json"
    claim_path.write_text(json.dumps(document))
    added = lane.json("project", "finding", "add", str(claim_path), "--author", "analyst-a")
    finding_id = added["id"]
    assert (added["type"], added["review_state"], added["freshness"]) == (
        "file-format", "proposed", "current")
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps({
        "verdict": "accept", "rationale": "Independently checked the four persisted bytes.",
        "evidence": document["evidence"],
    }))
    self_review = lane.run(
        "project", "finding", "review", finding_id, str(review_path),
        "--expected-revision", "1", "--reviewer", "analyst-a")
    assert self_review.returncode == 2, self_review.stdout + self_review.stderr
    assert lane.json("project", "finding", "show", finding_id)["revision"] == 1
    reviewed = lane.json(
        "project", "finding", "review", finding_id, str(review_path),
        "--expected-revision", "1", "--reviewer", "reviewer-b")
    assert (reviewed["revision"], reviewed["review_state"]) == (2, "accepted")

    # Both callers have read revision 2. Only one may append revision 3.
    def concurrent_review(reviewer: str) -> subprocess.CompletedProcess[str]:
        return lane.run(
            "project", "finding", "review", finding_id, str(review_path),
            "--expected-revision", "2", "--reviewer", reviewer, "--format", "json")

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(concurrent_review, ["reviewer-c", "reviewer-d"]))
    assert sorted(result.returncode for result in outcomes) == [0, 2], [
        (result.returncode, result.stdout, result.stderr) for result in outcomes]
    refused = next(result for result in outcomes if result.returncode == 2)
    assert "revision" in refused.stderr.lower()
    context = lane.json(
        "project", "finding", "context", "--file", file["id"], "--type", "file-format")
    assert [row["id"] for row in context["groups"]["accepted"]] == [finding_id]
    assert lane.json(
        "project", "finding", "list", "--file", file["id"], "--type", "vulnerability"
    )["items"] == []

    lane.project_close(force=True)
    lane.ensure_open()
    reopened = lane.json("project", "finding", "show", finding_id)
    assert (reopened["revision"], reopened["review_state"], reopened["freshness"]) == (
        3, "accepted", "current")
    history = lane.json("project", "finding", "history", finding_id)
    assert [record["revision"] for record in history["items"]] == [1, 2, 3]
    assert history["items"][0]["review_state"] == "proposed"

    ledger_files = lane.project_files(f".bn-findings/{finding_id}")
    assert len(ledger_files) == 3
    protected = lane.run("project", "delete", ledger_files[0]["id"])
    assert protected.returncode == 2
    assert "reserved findings history" in protected.stderr
    load_refused = lane.run("project", "load", ledger_files[0]["id"])
    assert load_refused.returncode == 2
    assert "reserved findings history" in load_refused.stderr

    lane.json("project", "delete", file["id"])
    stale = lane.json("project", "finding", "check", finding_id)
    assert stale["freshness"] == "stale" and stale["review_state"] == "accepted"
    assert stale["reasons"]
    stale_context = lane.json("project", "finding", "context", "--query", "PKTA")
    assert finding_id not in {row["id"] for row in stale_context["groups"]["accepted"]}
    assert finding_id in {row["id"] for row in stale_context["groups"]["needs_review"]}


def test_findings_function_evidence_tracks_saved_analysis_not_dirty_views(lane: Lane, tmp_path):
    """Exercise real HLIL APIs and invalidate accepted evidence after a saved edit."""
    lane.ensure_open()
    raw = lane.ensure_imported("finding-function")
    lane.project_load(raw["id"])
    saved = lane.save(raw["id"])
    assert saved.returncode == 0, saved.stderr
    database_id = json.loads(saved.stdout)["project_file_id"]
    functions = lane.json("function", "list", "-t", database_id, "--limit", "100")["items"]
    main = next(row for row in functions if row["name"] in ("main", "_main"))
    # A reopened database can defer HLIL. Explicitly materialize it with an
    # ordinary read; the findings checker itself never generates missing IL.
    lane.json("decompile", main["address"], "-t", database_id)
    evidence = [{"check": "function", "file": database_id, "function": main["address"],
                 "expected": {"name": main["name"]}}]
    claim_path = tmp_path / "function-claim.json"
    claim_path.write_text(json.dumps({
        "type": "function-role", "kind": "observation",
        "claim": "The entry function has the recorded analysis name and prototype.",
        "scope": [{"file": database_id, "function": main["address"]}], "evidence": evidence,
    }))
    added = lane.json("project", "finding", "add", str(claim_path), "--author", "analyst-a")
    assert added["freshness"] == "current", added["freshness_reasons"]
    review_path = tmp_path / "function-review.json"
    review_path.write_text(json.dumps({
        "verdict": "accept", "rationale": "Independently inspected the saved function.",
        "evidence": evidence,
    }))
    accepted = lane.json(
        "project", "finding", "review", added["id"], str(review_path),
        "--expected-revision", "1", "--reviewer", "reviewer-b")
    assert accepted["review_state"] == "accepted" and accepted["freshness"] == "current"
    lane.rename(database_id, main["address"], "reviewed_entry")
    dirty = lane.json("project", "finding", "check", added["id"])
    assert dirty["freshness"] != "current"
    assert lane.save(database_id).returncode == 0
    lane.json("close", "-t", database_id)
    stale = lane.json("project", "finding", "check", added["id"])
    assert stale["freshness"] == "stale" and stale["review_state"] == "accepted"


def test_findings_links_invalidate_when_a_project_provider_disappears(rootfs_lane, tmp_path):
    """Bindings must depend on provider files, not just the unchanged caller BNDB."""
    lane, root = rootfs_lane
    _rootfs_init(lane, root)
    lane.project_import(root / "usr/bin/app", "usr/bin")
    lane.project_load("usr/bin/app")
    saved = lane.save("app")
    assert saved.returncode == 0, saved.stderr
    database_id = json.loads(saved.stdout)["project_file_id"]
    claim_path = tmp_path / "bindings.json"
    claim_path.write_text(json.dumps({
        "type": "dependency", "kind": "observation",
        "claim": "The caller has the captured project-library assignments.",
        "scope": [{"file": database_id}],
        "evidence": [{"check": "links", "file": database_id}],
        "limitations": ["Bindings alone do not prove runtime-loader resolution or cross-library taint."],
    }))
    finding = lane.json("project", "finding", "add", str(claim_path), "--author", "link-analyst")
    assert finding["freshness"] == "current", finding["freshness_reasons"]
    observed = finding["evidence"][0]["observed"]
    # The fixture's two native ProjectFile providers must be identified.
    inventory = lane.json("project", "libs")["libs"]
    provider_ids = {row["project_file_id"] for row in inventory}
    assert set(observed["providers"]) == provider_ids
    provider = next(row for row in inventory if "libgreet.so.1" in row["names"])
    # #1078: the provider is bound by the saved database; a bare delete refuses
    # and names that target, `--force` deletes and reports it.
    refused = lane.run("project", "delete", provider["project_file_id"])
    assert refused.returncode == 2, refused.stdout
    assert "is the linked provider of" in refused.stderr
    assert "/usr/bin/app.bndb" in refused.stderr and "--force" in refused.stderr
    assert provider["project_file_id"] in {
        row["project_file_id"] for row in lane.json("project", "libs")["libs"]}
    forced = lane.json("project", "delete", provider["project_file_id"], "--force")
    assert [row["project_path"] for row in forced["dependents"]] == ["/usr/bin/app.bndb"]
    lane.json("close", "-t", database_id)
    checked = lane.json("project", "finding", "check", finding["id"])
    assert checked["freshness"] == "stale", checked
    assert any("provider" in reason.lower() for reason in checked["reasons"])
