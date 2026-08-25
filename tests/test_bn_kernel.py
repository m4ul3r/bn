from __future__ import annotations

import asyncio
import json
import os
import sys
import subprocess
import threading
import time
import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "skills" / "bn-kernel" / "src"),
)

_previous_dont_write_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    import bn_kernel  # noqa: E402
finally:
    sys.dont_write_bytecode = _previous_dont_write_bytecode
from bn.transport import BridgeError as NativeBridgeError  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _fake_bn(tmp_path: Path, script: str) -> Path:
    executable = tmp_path / "fake-bn"
    executable.write_text("#!/usr/bin/env python3\n" + script)
    executable.chmod(0o755)
    return executable


def _payload_script(payload: object, *, rc: int = 0, stderr: str = "") -> str:
    return f"""
import json, sys
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
open(out, "w").write(json.dumps({payload!r}))
sys.stderr.write({stderr!r})
sys.exit({rc})
"""


RECORDER = """
import json, sys
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
json.dump({"argv": argv}, open(out, "w"))
"""


@pytest.fixture(autouse=True)
def _clear_bn_bin(monkeypatch):
    monkeypatch.delenv("BN_BIN", raising=False)
    bn_kernel._ACTIVE_SESSIONS.clear()
    bn_kernel._ACTIVE_SCOPED_CALLBACKS.clear()
    bn_kernel._ACTIVE_SCOPED_BINDINGS.clear()
    bn_kernel._WARNED_BINDING_PAIRS.clear()
    yield
    bn_kernel._ACTIVE_SESSIONS.clear()
    bn_kernel._ACTIVE_SCOPED_CALLBACKS.clear()
    bn_kernel._ACTIVE_SCOPED_BINDINGS.clear()
    bn_kernel._WARNED_BINDING_PAIRS.clear()


def test_bootstrap_restores_module_after_sys_modules_loss():
    bootstrap = (
        Path(__file__).resolve().parents[1] / "skills" / "bn-kernel" / "bootstrap.py"
    )
    code = (
        "import pathlib, sys\n"
        f"source = pathlib.Path({str(bootstrap)!r}).read_text()\n"
        f"first_globals = {{'skill_dir': pathlib.Path({str(bootstrap.parent)!r})}}\n"
        "exec(compile(source, 'bootstrap.py', 'exec'), first_globals, first_globals)\n"
        "assert first_globals['bn_kernel'].__name__ == 'bn_kernel'\n"
        "sys.modules.pop('bn_kernel', None)\n"
        f"second_globals = {{'skill_dir': pathlib.Path({str(bootstrap.parent)!r})}}\n"
        "exec(compile(source, 'bootstrap.py', 'exec'), second_globals, second_globals)\n"
        "assert second_globals['bn_kernel'].__name__ == 'bn_kernel'\n"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("bn_kernel bootstrap: reloaded") == 2
    assert f"source={bootstrap.parent / 'src' / 'bn_kernel' / '__init__.py'}" in result.stdout
    assert "sha256=" in result.stdout


def test_bootstrap_evicts_stale_module_and_bytecode():
    bootstrap = (
        Path(__file__).resolve().parents[1] / "skills" / "bn-kernel" / "bootstrap.py"
    )
    code = (
        "import pathlib, sys\n"
        f"skill_dir = pathlib.Path({str(bootstrap.parent)!r})\n"
        f"source = pathlib.Path({str(bootstrap)!r}).read_text()\n"
        "ns = {'skill_dir': skill_dir}\n"
        "exec(compile(source, 'bootstrap.py', 'exec'), ns, ns)\n"
        "stale = ns['bn_kernel']\n"
        "stale.__bn_kernel_source_hash__ = 'stale'\n"
        "stale.Session = object()\n"
        "cache = skill_dir / 'src' / 'bn_kernel' / '__pycache__'\n"
        "cache.mkdir(exist_ok=True)\n"
        "(cache / 'stale.pyc').write_bytes(b'stale')\n"
        "ns2 = {'skill_dir': skill_dir}\n"
        "exec(compile(source, 'bootstrap.py', 'exec'), ns2, ns2)\n"
        "assert ns2['bn_kernel'] is not stale\n"
        "assert hasattr(ns2['bn_kernel'].Session, 'disasm')\n"
        "assert not (cache / 'stale.pyc').exists()\n"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr

def test_run_reads_complete_json_artifact_and_unwraps_items(monkeypatch, tmp_path):
    payload = {
        "kind": "functions",
        "items": [{"name": f"function_{index}"} for index in range(2000)],
        "total": 2000,
    }
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, _payload_script(payload))))
    session = bn_kernel.Session(backend="cli")

    value = _run(session.run("function", "list"))

    assert value == payload["items"]
    assert session.last is not None
    assert session.last.payload == payload
    assert session.last.total == 2000
    assert session.last.backend == "cli"


def test_run_orders_root_flags_and_converts_keyword_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, RECORDER)))
    session = bn_kernel.Session(
        instance="worker", target="sample.bin", backend="cli"
    )

    payload = _run(
        session.run(
            "function",
            "search",
            "parse",
            unwrap=False,
            word=True,
            exact=False,
            min_size=20,
            tag=["one", "two"],
            ignored=None,
        )
    )

    argv = payload["argv"]
    assert argv[:7] == [
        "-i",
        "worker",
        "-t",
        "sample.bin",
        "function",
        "search",
        "parse",
    ]
    assert argv.count("--tag") == 2
    assert argv[argv.index("--tag") : argv.index("--tag") + 4] == [
        "--tag",
        "one",
        "--tag",
        "two",
    ]
    assert "--word" in argv
    assert "--exact" not in argv
    assert "--ignored" not in argv
    assert argv[argv.index("--min-size") : argv.index("--min-size") + 2] == [
        "--min-size",
        "20",
    ]
    assert argv[argv.index("--format") + 1] == "json"
    assert argv[argv.index("--out") + 1].startswith(("/proc/self/fd/", "/dev/fd/"))
    assert Path(argv[argv.index("--out") + 1]).exists() is False


def test_run_raw_reads_complete_text(monkeypatch, tmp_path):
    text = "line one\nline two\n"
    script = f"""
import sys
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
open(out, "w").write({text!r})
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    assert _run(session.run("disasm", "main", raw=True)) == text
    assert session.last is not None
    assert session.last.payload == text
    assert session.last.value == text


def test_run_unwraps_first_text_key_and_can_return_payload(monkeypatch, tmp_path):
    payload = {"listing": "mov eax, eax", "body": "ignored"}
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, _payload_script(payload))))
    session = bn_kernel.Session(backend="cli")

    assert _run(session.run("disasm", "main")) == "mov eax, eax"
    assert _run(session.run("disasm", "main", unwrap=False)) == payload


def test_result_is_immutable_and_uses_tuples():
    result = bn_kernel.Result(
        value=[], payload={"total": 4}, notes=("note",), argv=("strings",), backend="cli"
    )

    assert result.total == 4
    assert result.notes == ("note",)
    assert result.argv == ("strings",)
    with pytest.raises(FrozenInstanceError):
        result.backend = "native"  # type: ignore[misc]




def test_help_returns_in_band_cli_grammar_without_artifact(monkeypatch, tmp_path):
    script = """
import sys
argv = sys.argv[1:]
assert "--out" not in argv
assert "--format" not in argv
sys.stdout.write("usage: bn evidence {orient,surface,calls}\\n")
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(
        instance="worker", target="sample", backend="native"
    )

    text = _run(session.help("evidence"))

    assert text == "usage: bn evidence {orient,surface,calls}\n"
    assert session.last is not None
    assert session.last.argv == (
        "-i",
        "worker",
        "-t",
        "sample",
        "evidence",
        "--help-full",
    )
    assert session.last.backend == "cli"


def test_help_maps_cli_error(monkeypatch, tmp_path):
    script = """
import sys
sys.stderr.write("unknown command group\\n")
sys.exit(1)
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))

    with pytest.raises(bn_kernel.CliError, match="unknown command group"):
        _run(bn_kernel.Session(backend="cli").help("unknown"))


def test_help_error_omits_argparse_usage_blob(monkeypatch, tmp_path):
    script = """
import sys
sys.stderr.write(
    "usage: bn [-h] {function,evidence}\\n"
    "bn: error: argument command: invalid choice: 'search'\\n"
)
sys.exit(1)
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))

    with pytest.raises(bn_kernel.CliError) as caught:
        _run(bn_kernel.Session(backend="cli").help("search"))

    assert str(caught.value) == (
        "bn: error: argument command: invalid choice: 'search'"
    )
@pytest.mark.parametrize(
    ("rc", "exception"),
    [
        (1, bn_kernel.CliError),
        (2, bn_kernel.BridgeError),
        (3, bn_kernel.VerificationFailed),
        (9, bn_kernel.BnError),
    ],
)
def test_run_maps_exit_codes(monkeypatch, tmp_path, rc, exception):
    monkeypatch.setenv(
        "BN_BIN", str(_fake_bn(tmp_path, _payload_script({}, rc=rc, stderr="failure\n")))
    )

    with pytest.raises(exception, match="failure") as caught:
        _run(bn_kernel.Session(backend="cli").run("target", "info"))

    assert caught.value.returncode == rc
    assert caught.value.argv[:2] == ("target", "info")


def test_run_uses_stdout_then_default_for_failure_message(monkeypatch, tmp_path):
    stdout_script = """
import sys
sys.stdout.write("stdout failure\\n")
sys.exit(1)
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, stdout_script)))
    with pytest.raises(bn_kernel.CliError, match="stdout failure"):
        _run(bn_kernel.Session(backend="cli").run("bad"))

    silent_script = "import sys; sys.exit(1)\n"
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, silent_script)))
    with pytest.raises(bn_kernel.CliError, match="bn failed"):
        _run(bn_kernel.Session(backend="cli").run("bad"))


def test_missing_executable_is_returncode_127(monkeypatch):
    monkeypatch.setenv("BN_BIN", "")
    monkeypatch.setattr(bn_kernel.shutil, "which", lambda name: None)

    with pytest.raises(bn_kernel.BnError, match="bn executable not found") as caught:
        _run(bn_kernel.Session(backend="cli").run("target", "info"))

    assert caught.value.returncode == 127
    assert caught.value.argv == ("bn",)


def test_timeout_kills_reaps_and_removes_artifact(monkeypatch, tmp_path):
    marker = tmp_path / "artifact-path"
    script = f"""
import sys, time
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
open({str(marker)!r}, "w").write(out)
time.sleep(30)
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))

    with pytest.raises(bn_kernel.BnError, match="timed out") as caught:
        _run(bn_kernel.Session(timeout=0.1, backend="cli").run("target", "info"))

    assert caught.value.returncode == 124
    assert marker.exists()
    assert not Path(marker.read_text()).exists()


@pytest.mark.parametrize("body", ["", "not json"])
def test_zero_exit_with_invalid_json_is_an_error(monkeypatch, tmp_path, body):
    script = f"""
import sys
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
open(out, "w").write({body!r})
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))

    with pytest.raises(bn_kernel.BnError, match="invalid JSON"):
        _run(bn_kernel.Session(backend="cli").run("target", "info"))


def test_all_pages_and_replaces_last_with_aggregate(monkeypatch, tmp_path):
    script = """
import json, sys
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
offset = int(argv[argv.index("--offset") + 1])
limit = int(argv[argv.index("--limit") + 1])
rows = [{"i": i} for i in range(offset, min(offset + limit, 7))]
json.dump({"items": rows, "offset": offset, "returned": len(rows), "total": 7,
           "has_more": offset + len(rows) < 7, "kind": "rows"}, open(out, "w"))
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(instance="worker", backend="cli")

    rows = _run(session.all("function", "list", page=3))

    assert [row["i"] for row in rows] == list(range(7))
    assert session.last is not None
    assert session.last.payload == {
        "items": rows,
        "offset": 0,
        "returned": 7,
        "total": 7,
        "has_more": False,
        "kind": "rows",
    }
    assert session.last.argv[:3] == ("-i", "worker", "function")


def test_all_honors_initial_offset_and_total_cap(monkeypatch, tmp_path):
    script = """
import json, sys
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
offset = int(argv[argv.index("--offset") + 1])
limit = int(argv[argv.index("--limit") + 1])
rows = [{"i": i} for i in range(offset, offset + limit)]
json.dump({"items": rows, "total": 100, "has_more": True}, open(out, "w"))
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    rows = _run(session.all("strings", page=3, limit=5, offset="10"))

    assert [row["i"] for row in rows] == [10, 11, 12, 13, 14]
    assert session.last is not None
    assert session.last.payload["offset"] == 10
    assert session.last.payload["returned"] == 5
    assert session.last.payload["has_more"] is True


def test_all_zero_limit_makes_no_cli_call(monkeypatch):
    monkeypatch.setenv("BN_BIN", "/does/not/exist")
    session = bn_kernel.Session(backend="cli")

    assert _run(session.all("strings", limit=0, offset=4)) == []
    assert session.last is not None
    assert session.last.payload == {
        "items": [],
        "offset": 4,
        "returned": 0,
        "has_more": False,
        "total": None,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"page": 0}, "page must be at least 1"),
        ({"limit": -1}, "limit must be non-negative"),
    ],
)
def test_all_rejects_invalid_bounds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _run(bn_kernel.Session(backend="cli").all("strings", **kwargs))


def test_all_rejects_zero_progress(monkeypatch, tmp_path):
    payload = {"items": [], "has_more": True, "total": 1}
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, _payload_script(payload))))

    with pytest.raises(bn_kernel.BnError, match="strings.*empty page"):
        _run(bn_kernel.Session(backend="cli").all("strings"))




def test_all_rejects_per_page_returned_mismatch(monkeypatch, tmp_path):
    payload = {
        "items": [{"value": "x"}],
        "returned": 0,
        "total": 1,
        "has_more": False,
    }
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, _payload_script(payload))))

    with pytest.raises(bn_kernel.BnError, match="returned.*items"):
        _run(bn_kernel.Session(backend="cli").all("strings"))


def test_all_rejects_total_change_on_later_page(monkeypatch, tmp_path):
    script = """
import json, sys
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
offset = int(argv[argv.index("--offset") + 1])
payload = (
    {"items": [{"value": "a"}], "returned": 1, "total": 2, "has_more": True}
    if offset == 0
    else {"items": [{"value": "b"}], "returned": 1, "total": 3, "has_more": False}
)
json.dump(payload, open(out, "w"))
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))

    with pytest.raises(bn_kernel.BnError, match="total changed"):
        _run(bn_kernel.Session(backend="cli").all("strings", page=1))


def test_all_timeout_is_one_end_to_end_budget(monkeypatch, tmp_path):
    script = """
import json, sys, time
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
offset = int(argv[argv.index("--offset") + 1])
time.sleep(0.03)
json.dump(
    {
        "items": [{"value": str(offset)}],
        "returned": 1,
        "offset": offset,
        "total": 2,
        "has_more": offset == 0,
    },
    open(out, "w"),
)
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    started = time.monotonic()

    with pytest.raises(bn_kernel.BnError, match="timed out"):
        _run(
            bn_kernel.Session(timeout=1, backend="cli").all(
                "strings", page=1, timeout=0.05
            )
        )

    assert time.monotonic() - started < 0.1


def test_all_invalidates_last_after_mid_pagination_failure(monkeypatch, tmp_path):
    script = """
import json, sys
argv = sys.argv[1:]
offset = int(argv[argv.index("--offset") + 1])
out = argv[argv.index("--out") + 1]
if offset == 0:
    json.dump(
        {
            "items": [{"value": "first"}],
            "offset": 0,
            "returned": 1,
            "total": 2,
            "has_more": True,
        },
        open(out, "w"),
    )
else:
    sys.stderr.write("second page failed")
    sys.exit(2)
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    with pytest.raises(bn_kernel.BridgeError, match="second page failed"):
        _run(session.all("strings", page=1))

    assert session.last is None
class FakeClient:
    def __init__(self):
        self.calls: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
        self.thread_ids: list[int] = []

    def request(self, op: str, params: dict[str, Any] | None = None):
        self.thread_ids.append(threading.get_ident())
        self.calls.append(("request", op, dict(params or {}), {}))
        if op == "target_info":
            return {"arch": "x86"}
        if op == "function_info":
            return {
                "function": {"name": "main"},
                "size": 12,
                "size_known": True,
                "imported": False,
            }
        if op in {"decompile", "disasm", "il"}:
            return {"text": f"{op} text"}
        raise AssertionError(op)

    def collect(
        self,
        op: str,
        params: dict[str, Any] | None = None,
        *,
        limit: int | None = None,
    ):
        self.thread_ids.append(threading.get_ident())
        self.calls.append(("collect", op, dict(params or {}), {"limit": limit}))
        item = {"op": op}
        if op in {"list_functions", "search_functions"}:
            item.update({"size": 12, "size_known": True})
        if op == "strings":
            item["value"] = "sample"
        if op == "callsites":
            item.update(
                {
                    "callee": {"name": "sink", "address": "0x2000"},
                    "containing_function": {
                        "name": "caller",
                        "address": "0x1000",
                    },
                    "call_addr": "0x1010",
                    "caller_static": "0x1015",
                }
            )
        return {
            "items": [item],
            "offset": 0,
            "returned": 1,
            "has_more": True,
            "total": 8,
        }


def _native_session() -> tuple[bn_kernel.Session, FakeClient]:
    session = bn_kernel.Session(instance="worker", target="sample", backend="native")
    client = FakeClient()
    session._client = client
    return session, client


def test_backend_selection_auto_cli_and_native(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(bn_kernel, "_load_native_client", lambda *args: sentinel)
    assert bn_kernel.Session(backend="auto").backend == "native"

    def unavailable(*args):
        raise ImportError("missing")

    monkeypatch.setattr(bn_kernel, "_load_native_client", unavailable)
    assert bn_kernel.Session(backend="auto").backend == "cli"
    with pytest.raises(bn_kernel.BnError, match="backend='cli'"):
        bn_kernel.Session(backend="native")


def test_backend_selection_rejects_invalid_value():
    with pytest.raises(ValueError, match="backend"):
        bn_kernel.Session(backend="other")  # type: ignore[arg-type]


def test_auto_does_not_hide_broken_native_import(monkeypatch):
    def broken(*args):
        raise RuntimeError("broken package")

    monkeypatch.setattr(bn_kernel, "_load_native_client", broken)
    with pytest.raises(RuntimeError, match="broken package"):
        bn_kernel.Session(backend="auto")


def test_distinct_live_bindings_warn_once(monkeypatch):
    bn_kernel._ACTIVE_SESSIONS.clear()
    bn_kernel._WARNED_BINDING_PAIRS.clear()
    first = bn_kernel.Session(instance="agent-a", target="one", backend="cli")

    with pytest.warns(RuntimeWarning, match="shared retained kernel namespace"):
        second = bn_kernel.Session(instance="agent-b", target="two", backend="cli")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        third = bn_kernel.Session(instance="agent-b", target="two", backend="cli")

    assert not caught
    assert first.instance == "agent-a"
    assert second.instance == third.instance == "agent-b"


def test_scoped_passes_function_local_session_to_sync_and_async_callbacks():
    async def async_cell(session):
        await asyncio.sleep(0)
        return session.instance, session.target, session.backend

    def sync_cell(session):
        return session.instance, session.target, session.backend

    assert _run(
        bn_kernel.scoped(
            async_cell,
            instance="worker",
            target="sample",
            backend="cli",
        )
    ) == ("worker", "sample", "cli")
    assert _run(
        bn_kernel.scoped(
            sync_cell,
            instance="worker",
            target="sample",
            backend="cli",
        )
    ) == ("worker", "sample", "cli")


def test_scoped_rejects_callback_rebound_to_different_target():
    async def work(session):
        return session.instance

    assert _run(
        bn_kernel.scoped(work, instance="agent-a", target="one", backend="cli")
    ) == "agent-a"

    with pytest.raises(bn_kernel.BnError, match="callback binding mismatch"):
        _run(
            bn_kernel.scoped(
                work,
                instance="agent-b",
                target="two",
                backend="cli",
            )
        )


def test_scoped_allows_inactive_retained_session_from_other_target():
    retained = bn_kernel.Session(
        instance="agent-a", target="one", backend="cli"
    )

    async def local(session):
        return session.instance

    with pytest.warns(RuntimeWarning, match="shared retained kernel namespace"):
        result = _run(
            bn_kernel.scoped(
                local,
                instance="agent-b",
                target="two",
                backend="cli",
            )
        )

    assert result == "agent-b"
    assert retained.instance == "agent-a"


def test_scoped_rejects_concurrent_foreign_binding():
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def first(session):
            entered.set()
            await release.wait()
            return session.instance

        async def second(session):
            return session.instance

        task = asyncio.create_task(
            bn_kernel.scoped(
                first, instance="agent-a", target="one", backend="cli"
            )
        )
        await entered.wait()
        try:
            with pytest.raises(bn_kernel.BnError, match="foreign active scoped"):
                await bn_kernel.scoped(
                    second,
                    instance="agent-b",
                    target="two",
                    backend="cli",
                )
        finally:
            release.set()
            await task

    _run(scenario())


def test_native_request_runs_off_event_loop_thread_and_updates_last():
    session, client = _native_session()
    loop_thread = threading.get_ident()

    value = _run(session.info(verbose=True))

    assert value == {"arch": "x86"}
    assert client.thread_ids and client.thread_ids[0] != loop_thread
    assert client.calls == [("request", "target_info", {"verbose": True}, {})]
    assert session.last == bn_kernel.Result(
        value={"arch": "x86"},
        payload={"arch": "x86"},
        notes=(),
        argv=("target_info",),
        backend="native",
    )


@pytest.mark.parametrize("method", ["request", "collect"])
def test_native_operations_have_caller_visible_timeout(method):
    session = bn_kernel.Session(instance="worker", timeout=0.01, backend="native")

    class TimeoutClient:
        def request(self, op, params=None):
            raise NativeBridgeError("timed out waiting for bridge response")

        def collect(self, op, params=None, *, limit=None):
            raise NativeBridgeError("timed out waiting for bridge response")

    session._client = TimeoutClient()
    operation = session.info() if method == "request" else session.strings()

    with pytest.raises(bn_kernel.BnError, match="timed out") as caught:
        _run(operation)

    assert caught.value.returncode == 124



@pytest.mark.parametrize(
    "payload",
    [
        {"items": [], "returned": 0, "total": 3, "has_more": False},
        {"items": None, "returned": 0, "total": 0, "has_more": False},
        [],
        None,
    ],
)
def test_strings_rejects_silent_empty_or_shape_drift(payload):
    session = bn_kernel.Session(instance="worker", backend="native")

    class StringsClient:
        def collect(self, op, params=None, *, limit=None):
            assert op == "strings"
            return payload

    session._client = StringsClient()

    with pytest.raises(bn_kernel.BnError, match="strings.*contract"):
        _run(session.strings())



@pytest.mark.parametrize(
    "payload",
    [
        [{"name": "main"}],
        {"items": "bad"},
        {
            "items": [{"name": "main", "address": "0x1000"}],
            "offset": 0,
            "returned": 1,
            "total": 1,
            "has_more": False,
        },
    ],
)
def test_functions_rejects_collection_shape_drift(payload):
    session = bn_kernel.Session(instance="worker", backend="native")

    class FunctionsClient:
        def collect(self, op, params=None, *, limit=None):
            assert op == "list_functions"
            return payload

    session._client = FunctionsClient()
    with pytest.raises(bn_kernel.BnError, match="list_functions.*contract"):
        _run(session.functions())


def test_strings_timeout_is_control_not_bridge_filter(monkeypatch):
    session = bn_kernel.Session(instance="worker", backend="native")
    loaded = []

    class RecordingClient:
        def collect(self, op, params=None, *, limit=None):
            assert op == "strings"
            assert params == {"query": "fmt"}
            return {
                "items": [{"value": "sample"}],
                "offset": 0,
                "returned": 1,
                "total": 1,
                "has_more": False,
            }

    def load_client(instance, target, timeout):
        loaded.append((instance, target, timeout))
        return RecordingClient()

    monkeypatch.setattr(bn_kernel, "_load_native_client", load_client)

    assert _run(session.strings(query="fmt", timeout=2)) == [
        {"value": "sample"}
    ]
    assert loaded == [("worker", None, 2)]


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("functions", ()),
        ("search", ("query",)),
        ("strings", ()),
        ("imports", ()),
        ("sections", ()),
    ],
)
def test_curated_helpers_reject_unknown_kwargs(method, args):
    session = bn_kernel.Session(instance="worker", backend="native")

    with pytest.raises(TypeError, match="unexpected keyword.*typo"):
        _run(getattr(session, method)(*args, typo=True))

@pytest.mark.parametrize(
    ("method", "args", "kwargs", "kind", "operation", "params", "call_kwargs"),
    [
        ("functions", (), {"limit": 4, "min_size": 20}, "collect", "list_functions", {"min_size": 20}, {"limit": 4}),
        ("search", ("parse",), {"limit": 3, "exact": True}, "collect", "search_functions", {"query": "parse", "exact": True}, {"limit": 3}),
        ("function_info", ("main",), {"blocks": True}, "request", "function_info", {"identifier": "main", "blocks": True}, {}),
        ("decompile", ("main",), {"addresses": True, "force_analysis": True}, "request", "decompile", {"identifier": "main", "addresses": True, "force_analysis": True}, {}),
        ("disasm", ("main",), {"linear": 4, "mode": "bytes", "snap_to_instruction": True}, "request", "disasm", {"identifier": "main", "linear": 4, "mode": "bytes", "snap_to_instruction": True}, {}),
        ("il", ("main",), {"view": "mlil", "ssa": True}, "request", "il", {"identifier": "main", "view": "mlil", "ssa": True}, {}),
        ("xrefs", ("main",), {"limit": 2, "fn_pointer_scan": True}, "collect", "xrefs", {"identifier": "main", "fn_pointer_scan": True}, {"limit": 2}),
        ("callsites", ("sink",), {"within": ["a", "b"], "context": 5, "limit": 6}, "collect", "callsites", {"callee": "sink", "within_identifiers": ["a", "b"], "context": 5}, {"limit": 6}),
        ("strings", (), {"limit": 7, "query": "fmt"}, "collect", "strings", {"query": "fmt"}, {"limit": 7}),
        ("imports", (), {"limit": 8, "include_got": True}, "collect", "imports", {"include_got": True}, {"limit": 8}),
        ("sections", (), {"limit": 9, "query": "text"}, "collect", "sections", {"query": "text"}, {"limit": 9}),
    ],
)
def test_native_helper_operation_mappings(
    method, args, kwargs, kind, operation, params, call_kwargs
):
    session, client = _native_session()

    _run(getattr(session, method)(*args, **kwargs))

    assert client.calls == [(kind, operation, params, call_kwargs)]
    assert session.last is not None
    assert session.last.argv == (operation,)
    assert session.last.backend == "native"


def test_native_helper_text_and_items_have_same_return_shapes():
    session, _ = _native_session()

    assert _run(session.decompile("main")) == "decompile text"
    assert _run(session.functions(limit=1)) == [
        {"op": "list_functions", "size": 12, "size_known": True}
    ]


def test_native_search_retries_zero_hit_regex_like_query_and_discloses():
    session = bn_kernel.Session(instance="worker", backend="native")

    class SearchClient:
        def __init__(self):
            self.calls = []

        def collect(self, op, params=None, *, limit=None):
            self.calls.append((op, dict(params or {}), limit))
            if params.get("regex"):
                return {
                    "items": [{"name": "ParseRecord", "size": 12, "size_known": True}],
                    "total": 1,
                    "has_more": False,
                }
            return {"items": [], "total": 0, "has_more": False}

    client = SearchClient()
    session._client = client

    assert _run(session.search("Parse|Decode", limit=5)) == [
        {"name": "ParseRecord", "size": 12, "size_known": True}
    ]
    assert client.calls == [
        ("search_functions", {"query": "Parse|Decode"}, 5),
        ("search_functions", {"query": "Parse|Decode", "regex": True}, 5),
    ]
    assert session.last is not None
    assert session.last.payload["regex_fallback"] is True
    assert "regex" in session.last.notes[0]


def test_native_search_retries_dotted_regex_query_and_marks_plain_zero():
    session = bn_kernel.Session(instance="worker", backend="native")

    class SearchClient:
        def __init__(self):
            self.calls = []

        def collect(self, op, params=None, *, limit=None):
            self.calls.append(dict(params or {}))
            return {
                "items": [],
                "offset": 0,
                "returned": 0,
                "total": 0,
                "has_more": False,
            }

    client = SearchClient()
    session._client = client

    assert _run(session.search("Parse.Record")) == []
    assert client.calls == [
        {"query": "Parse.Record"},
        {"query": "Parse.Record", "regex": True},
    ]
    assert session.last is not None
    assert session.last.payload["regex_fallback"] is True

    client.calls.clear()
    assert _run(session.search("NoSuchLiteral")) == []
    assert client.calls == [{"query": "NoSuchLiteral"}]
    assert session.last is not None
    assert session.last.payload["regex_fallback"] is False
    assert "no regex fallback" in session.last.notes[-1]


def test_search_regex_fallback_shares_one_timeout_budget(monkeypatch):
    session = bn_kernel.Session(instance="worker", backend="native")
    loaded_timeouts = []
    calls = []

    class SearchClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def collect(self, op, params=None, *, limit=None):
            calls.append(dict(params or {}))
            time.sleep(0.02)
            if params.get("regex"):
                return {
                    "items": [
                        {"name": "ParseRecord", "size": 12, "size_known": True}
                    ],
                    "offset": 0,
                    "returned": 1,
                    "total": 1,
                    "has_more": False,
                }
            return {
                "items": [],
                "offset": 0,
                "returned": 0,
                "total": 0,
                "has_more": False,
            }

    def load_client(instance, target, timeout):
        loaded_timeouts.append(timeout)
        return SearchClient(timeout)

    monkeypatch.setattr(bn_kernel, "_load_native_client", load_client)

    assert _run(session.search("Parse|Decode", timeout=0.1))
    assert len(calls) == 2
    assert 0 < loaded_timeouts[1] < loaded_timeouts[0] <= 0.1


@pytest.mark.parametrize(
    "query,filters,payload",
    [
        ("plain", {}, {"items": [], "total": 0, "has_more": False}),
        ("Parse|Decode", {"exact": True}, {"items": [], "total": 0, "has_more": False}),
        ("Parse|Decode", {"regex": True}, {"items": [], "total": 0, "has_more": False}),
        ("[invalid", {}, {"items": [], "total": 0, "has_more": False}),
        (
            "Parse|Decode",
            {},
            {
                "items": [
                    {"name": "literal", "size": 12, "size_known": True}
                ],
                "total": 1,
                "has_more": False,
            },
        ),
    ],
)


def test_native_search_does_not_retry_when_cli_would_not(query, filters, payload):
    session = bn_kernel.Session(instance="worker", backend="native")

    class SearchClient:
        def __init__(self):
            self.calls = []

        def collect(self, op, params=None, *, limit=None):
            self.calls.append((op, dict(params or {}), limit))
            return payload


    client = SearchClient()
    session._client = client

    _run(session.search(query, **filters))

    assert len(client.calls) == 1
@pytest.mark.parametrize(
    "payload",
    [
        {"text": None},
        {"decompiled": "body under undocumented key"},
        {},
        None,
    ],
)
def test_decompile_rejects_missing_text_contract(payload):
    session = bn_kernel.Session(instance="worker", backend="native")

    class DecompileClient:
        def request(self, op, params=None):
            assert op == "decompile"
            return payload

    session._client = DecompileClient()

    with pytest.raises(bn_kernel.BnError, match="decompile.*text contract"):
        _run(session.decompile("main"))


@pytest.mark.parametrize(
    "payload",
    [
        {
            "text": "int big_fn() {\n// This function is taking too long to analyze\n}",
            "analysis_skipped": False,
            "warnings": ["decompile is an incomplete stub"],
        },
        {
            "text": "int big_fn() {\n}",
            "analysis_skipped": True,
            "warnings": [],
        },
    ],
)
def test_decompile_rejects_analysis_placeholder(payload):
    session = bn_kernel.Session(instance="worker", backend="native")

    class PlaceholderClient:
        def request(self, op, params=None):
            return payload

    session._client = PlaceholderClient()

    with pytest.raises(bn_kernel.BnError, match="incomplete.*force_analysis"):
        _run(session.decompile("big_fn"))


def test_disasm_supports_bridge_ranges_and_preserves_metadata():
    session = bn_kernel.Session(instance="worker", backend="native")
    full_text = "0x1 one\n0x2 two\n0x3 three\n0x4 four"

    class DisasmClient:
        def request(self, op, params=None):
            assert op == "disasm"
            lines = full_text.splitlines()
            start = params.get("line_start")
            end = params.get("line_end")
            selected = lines if start is None else lines[start - 1 : min(end, len(lines))]
            return {
                "text": "\n".join(selected),
                "total_lines": len(lines),
                "returned_lines": len(selected),
                "line_range": (
                    {"start": start, "end": min(end, len(lines))}
                    if start is not None
                    else None
                ),
            }

    session._client = DisasmClient()

    assert _run(session.disasm("main", count=2)) == "0x1 one\n0x2 two"
    assert session.last is not None
    assert session.last.payload["line_range"] == {"start": 1, "end": 2}
    assert _run(session.disasm("main", lines=(2, 3))) == "0x2 two\n0x3 three"
    with pytest.raises(ValueError, match="count.*lines"):
        _run(session.disasm("main", count=1, lines=(1, 1)))
def test_function_info_flattens_helper_value_but_preserves_raw_payload(
    monkeypatch, tmp_path
):
    payload = {
        "function": {
            "name": "main",
            "address": "0x401000",
            "raw_name": "main",
            "display_name": "main",
        },
        "size": 24,
        "size_known": True,
        "imported": False,
        "parameters": [],
    }
    native = bn_kernel.Session(instance="worker", backend="native")

    class InfoClient:
        def request(self, op, params=None):
            assert op == "function_info"
            return payload

    native._client = InfoClient()
    value = _run(native.function_info("main"))
    assert value == {
        "name": "main",
        "address": "0x401000",
        "raw_name": "main",


        "display_name": "main",
        "size": 24,
        "size_known": True,
        "imported": False,
        "parameters": [],
    }
    assert native.last is not None
    assert native.last.payload == payload

    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, _payload_script(payload))))
    cli = bn_kernel.Session(instance="worker", backend="cli")
    assert _run(cli.function_info("main")) == value
    assert cli.last is not None
    assert cli.last.payload == payload
def test_decompile_allows_legitimate_placeholder_like_program_text():
    session = bn_kernel.Session(instance="worker", backend="native")
    payload = {
        "text": 'int main() { puts("Loading..."); return 0; }',
        "analysis_skipped": False,
        "warnings": [],
    }

    class DecompileClient:
        def request(self, op, params=None):
            return payload

    session._client = DecompileClient()

    assert _run(session.decompile("main")) == payload["text"]


def test_native_bridge_errors_use_adapter_error_family():
    session, _ = _native_session()

    class FailingClient:
        def request(self, op, params=None):
            raise NativeBridgeError("bridge unavailable")

    session._client = FailingClient()

    with pytest.raises(bn_kernel.BridgeError, match="bridge unavailable") as caught:
        _run(session.info())

    assert caught.value.returncode == 2
    assert caught.value.argv == ("target_info",)
    assert isinstance(caught.value.__cause__, NativeBridgeError)


def test_native_assert_target_accepts_basename_and_full_path(tmp_path):
    session = bn_kernel.Session(instance="worker", backend="native")
    filename = tmp_path / "sample.bndb"

    class IdentityClient:
        def request(self, op, params=None):
            assert op == "target_info"
            return {"basename": filename.name, "filename": str(filename)}

    session._client = IdentityClient()

    assert _run(session.assert_target(filename.name))["filename"] == str(filename)
    assert _run(session.assert_target("sample"))["basename"] == filename.name
    assert _run(session.assert_target(filename))["basename"] == filename.name


def test_assert_target_accepts_shorter_caller_timeout(monkeypatch):
    session = bn_kernel.Session(instance="worker", timeout=1, backend="native")
    requested_timeouts = []

    class TimeoutIdentityClient:
        def request(self, op, params=None):
            raise NativeBridgeError("Timed out waiting for bridge response")

    def load_client(instance, target, timeout):
        requested_timeouts.append(timeout)
        return TimeoutIdentityClient()

    monkeypatch.setattr(bn_kernel, "_load_native_client", load_client)

    with pytest.raises(bn_kernel.BnError, match="timed out") as caught:
        _run(session.assert_target("sample.bndb", timeout=0.01))

    assert caught.value.returncode == 124
    assert requested_timeouts == [0.01]


def test_assert_target_cli_timeout_kills_subprocess(monkeypatch, tmp_path):
    marker = tmp_path / "pid"
    script = f"""
import os, time
open({str(marker)!r}, "w").write(str(os.getpid()))
time.sleep(30)
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(instance="worker", timeout=5, backend="cli")

    with pytest.raises(bn_kernel.BnError, match="timed out"):
        _run(session.assert_target("sample.bndb", timeout=0.05))

    assert marker.exists()
    pid = int(marker.read_text())
    try:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


def test_assert_target_rejects_foreign_target_on_native_and_cli(
    monkeypatch, tmp_path
):
    native = bn_kernel.Session(instance="worker", backend="native")

    class IdentityClient:
        def request(self, op, params=None):
            return {
                "basename": "foreign.bndb",
                "filename": str(tmp_path / "foreign.bndb"),
            }

    native._client = IdentityClient()
    with pytest.raises(bn_kernel.BridgeError, match="target identity mismatch"):
        _run(native.assert_target("expected.bndb"))

    payload = {
        "basename": "foreign.bndb",
        "filename": str(tmp_path / "foreign.bndb"),
    }
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, _payload_script(payload))))
    cli = bn_kernel.Session(instance="worker", backend="cli")
    with pytest.raises(bn_kernel.BridgeError, match="target identity mismatch"):
        _run(cli.assert_target("expected.bndb"))


@pytest.mark.parametrize(
    "annotations",
    [
        {"comments": 1, "function_comments": 0, "user_symbols": 0},
        {"comments": 0, "function_comments": 1, "user_symbols": 0},
    ],
)
def test_assert_unannotated_rejects_inherited_state(annotations):
    session = bn_kernel.Session(instance="worker", backend="native")

    class OrientClient:
        def request(self, op, params=None):
            assert op == "orient_digest"
            return {"existing_annotations": annotations}

    session._client = OrientClient()

    with pytest.raises(bn_kernel.BridgeError, match="inherited comments"):
        _run(session.assert_unannotated())


def test_assert_unannotated_locates_and_can_explicitly_allow_contamination():
    session = bn_kernel.Session(instance="worker", backend="native")
    digest = {
        "existing_annotations": {
            "comments": 1,
            "comment_locations": [
                {"address": "0x1234", "comment": "inherited note"}
            ],
            "function_comments": 0,
            "function_comment_locations": [],
            "user_symbols": 0,
        }
    }

    class OrientClient:
        def request(self, op, params=None):
            return digest

    session._client = OrientClient()

    with pytest.raises(bn_kernel.BridgeError, match="0x1234"):
        _run(session.assert_unannotated())
    assert (
        _run(session.assert_unannotated(allow_contaminated=True))
        == digest
    )


def test_assert_unannotated_returns_digest_and_preserves_last():
    session = bn_kernel.Session(instance="worker", backend="native")
    digest = {
        "kind": "orient_digest",
        "existing_annotations": {
            "comments": 0,
            "function_comments": 0,
            "user_symbols": 540,
        },
    }

    class OrientClient:
        def request(self, op, params=None):
            assert op == "orient_digest"
            assert params == {"strings_limit": 1}
            return digest

    session._client = OrientClient()

    assert _run(session.assert_unannotated()) == digest
    assert session.last is not None
    assert session.last.payload == digest


@pytest.mark.parametrize("within", [[], (), ""])
def test_callsites_rejects_empty_scope(within):
    session, _ = _native_session()
    with pytest.raises(ValueError, match="within"):
        _run(session.callsites("sink", within=within))


@pytest.mark.parametrize(
    "row",
    [
        {"call_addr": "0x10", "caller_static": "0x15"},
        {
            "call_addr": "0x10",
            "caller_static": "0x15",
            "containing_function": None,
            "callee": {"name": "sink", "address": "0x20"},
        },
        {
            "call_addr": "10",
            "caller_static": "15",
            "containing_function": {"name": "caller", "address": "0x1"},
            "callee": {"name": "sink", "address": "0x20"},
        },
    ],
)
def test_callsites_rejects_unattributed_or_noncanonical_rows(row):
    session = bn_kernel.Session(instance="worker", backend="native")

    class CallsitesClient:
        def collect(self, op, params=None, *, limit=None):
            return {
                "items": [row],
                "offset": 0,
                "returned": 1,
                "total": 1,
                "has_more": False,
            }

    session._client = CallsitesClient()

    with pytest.raises(bn_kernel.BnError, match="callsites.*row contract"):
        _run(session.callsites("sink"))


def test_callsites_without_scope_uses_all_callers_on_native_and_cli(
    monkeypatch, tmp_path
):
    native, client = _native_session()

    native_rows = _run(native.callsites("sink", limit=2))
    assert len(native_rows) == 1
    assert native_rows[0]["op"] == "callsites"
    assert client.calls == [
        (
            "collect",
            "callsites",
            {
                "callee": "sink",
                "within_identifiers": [],
                "context": 3,
            },
            {"limit": 2},
        )
    ]

    script = """
import json, sys
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
row = {
    "argv": argv,
    "callee": {"name": "sink", "address": "0x2000"},
    "containing_function": {"name": "caller", "address": "0x1000"},
    "call_addr": "0x1010",
    "caller_static": "0x1015",
}
json.dump({"items": [row], "has_more": False, "total": 1}, open(out, "w"))
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    cli = bn_kernel.Session(instance="worker", target="sample", backend="cli")
    rows = _run(cli.callsites("sink", limit=2))
    assert "--within" not in rows[0]["argv"]
    assert "--within-file" not in rows[0]["argv"]


def test_cli_callsites_uses_within_for_single_scope(monkeypatch, tmp_path):
    script = """
import json, sys
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
row = {
    "argv": argv,
    "callee": {"name": "sink", "address": "0x2000"},
    "containing_function": {"name": "caller", "address": "0x1000"},
    "call_addr": "0x1010",
    "caller_static": "0x1015",
}
json.dump({"items": [row], "has_more": False, "total": 1}, open(out, "w"))
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    rows = _run(session.callsites("sink", within="caller", limit=1))

    argv = rows[0]["argv"]
    assert argv[argv.index("--within") + 1] == "caller"
    assert "--within-file" not in argv


def test_cli_callsites_uses_secure_temporary_scope_file(monkeypatch, tmp_path):
    script = """
import json, os, stat, sys
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
scope = argv[argv.index("--within-file") + 1]
row = {
    "scope": scope,
    "content": open(scope).read(),
    "mode": stat.S_IMODE(os.stat(scope).st_mode),
    "callee": {"name": "sink", "address": "0x2000"},
    "containing_function": {"name": "caller", "address": "0x1000"},
    "call_addr": "0x1010",
    "caller_static": "0x1015",
}
json.dump({"items": [row], "has_more": False, "total": 1}, open(out, "w"))
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    rows = _run(session.callsites("sink", within=["one", "two"]))

    assert rows[0]["content"] == "one\ntwo\n"
    assert rows[0]["mode"] == 0o600
    assert not Path(rows[0]["scope"]).exists()


def test_cli_curated_helpers_match_native_return_shapes(monkeypatch, tmp_path):
    script = """
import json, sys
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
if "decompile" in argv:
    payload = {"text": "decompile text"}
else:
    payload = {"items": [{"op": "list_functions", "size": 12, "size_known": True}], "has_more": False, "total": 1}
json.dump(payload, open(out, "w"))
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    assert _run(session.decompile("main")) == "decompile text"
    assert _run(session.functions(limit=1)) == [
        {"op": "list_functions", "size": 12, "size_known": True}
    ]


def test_cli_xrefs_forwards_function_pointer_scan(monkeypatch, tmp_path):
    script = """
import json, sys
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
json.dump({"items": [{"argv": argv}], "has_more": False, "total": 1}, open(out, "w"))
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    rows = _run(
        session.xrefs("main", limit=1, fn_pointer_scan=True)
    )

    assert "--fn-pointer-scan" in rows[0]["argv"]


def test_one_shot_run_uses_explicit_binding(monkeypatch, tmp_path):
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, RECORDER)))

    payload = _run(
        bn_kernel.run(
            "target", "info", instance="worker", target="sample", unwrap=False
        )
    )

    assert payload["argv"][:6] == [
        "-i",
        "worker",
        "-t",
        "sample",
        "target",
        "info",
    ]


def test_brief_is_bounded_and_reports_exact_remainder():
    rows = [
        {"name": "one", "size": 1, "extra": "x"},
        {"name": "two", "size": 2, "extra": "y"},
        {"name": "three", "size": 3, "extra": "z"},
    ]

    assert bn_kernel.brief(rows, "name", "size", n=2) == (
        "one  1\ntwo  2\n... 1 more of 3"
    )
    assert bn_kernel.brief([], "name") == "(0 rows)"
    assert bn_kernel.brief(rows, "name", n=0) == "... 3 more of 3"
    with pytest.raises(ValueError, match="n must be non-negative"):
        bn_kernel.brief(rows, n=-1)
    assert bn_kernel.brief(
        [{"callee": {"name": "sink"}, "call_addr": "0x1000"}],
        "callee.name",
        "call_addr",
    ) == "sink  0x1000"
    with pytest.raises(KeyError, match="brief key 'callee.address'.*row 0"):
        bn_kernel.brief(
            [{"callee": {"name": "sink"}}],
            "callee.address",
        )


@pytest.mark.parametrize(
    "rows",
    [
        {"items": []},
        "plain text",
        ["row text"],
    ],
)
def test_brief_rejects_non_row_payloads_with_actionable_error(rows):
    with pytest.raises(TypeError, match="brief.*row mappings"):
        bn_kernel.brief(rows)
