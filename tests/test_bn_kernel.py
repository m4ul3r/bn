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

def _load_bn_kernel_smoke():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "bn-kernel"
        / "scripts"
        / "smoke.py"
    )
    spec = importlib.util.spec_from_file_location("bn_kernel_smoke_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _canonical_target_info(**overrides):
    """The shape `target_info` actually returns: bridge.py `_function_name_summary`
    plus filename/basename/import_symbol_count."""
    payload = {
        "kind": "target",
        "filename": "/tmp/sample.bndb",
        "basename": "sample.bndb",
        "function_count": 3,
        "named_function_count": 2,
        "unnamed_function_count": 0,
        "imported_function_count": 1,
        "import_symbol_count": 4,
    }
    payload.update(overrides)
    return payload


@pytest.fixture(autouse=True)
def _clear_bn_bin(monkeypatch):
    monkeypatch.delenv("BN_BIN", raising=False)
    monkeypatch.delenv("BN_BACKEND", raising=False)
    bn_kernel._ACTIVE_SESSIONS.clear()
    bn_kernel._ACTIVE_SCOPED_CALLBACKS.clear()
    bn_kernel._ACTIVE_SCOPED_BINDINGS.clear()
    bn_kernel._WARNED_BINDING_PAIRS.clear()
    yield
    bn_kernel._ACTIVE_SESSIONS.clear()
    bn_kernel._ACTIVE_SCOPED_CALLBACKS.clear()
    bn_kernel._ACTIVE_SCOPED_BINDINGS.clear()
    bn_kernel._WARNED_BINDING_PAIRS.clear()

def test_smoke_default_instance_is_unique_to_the_process(monkeypatch):
    smoke = _load_bn_kernel_smoke()
    monkeypatch.setattr(sys, "argv", ["smoke.py"])

    args = smoke._parser().parse_args()

    assert args.instance == f"bn-kernel-smoke-{os.getpid()}"


def test_smoke_arms_idle_fallback_and_closes_target_before_stop(monkeypatch):
    from types import SimpleNamespace

    smoke = _load_bn_kernel_smoke()
    calls = []
    responses = iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "instance_id": "owned-worker",
                        "loaded": [{"targets": [{"selector": "sample.bndb"}]}],
                    }
                ),
                stderr="",
            ),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
    )

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return next(responses)

    async def fake_exercise(instance, backend):
        assert instance == "owned-worker"

    monkeypatch.setattr(smoke.shutil, "which", lambda name: "/usr/bin/bn")
    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(smoke, "_exercise", fake_exercise)
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke.py", "/tmp/sample.bin", "--instance", "owned-worker"],
    )
    monkeypatch.setenv("BN_IDLE_TIMEOUT", "off")

    assert smoke.main() == 0
    assert calls[0][1]["env"]["BN_IDLE_TIMEOUT"] == "3600"
    assert calls[1][0] == [
        "/usr/bin/bn",
        "-i",
        "owned-worker",
        "target",
        "close",
        "sample.bndb",
    ]
    assert calls[2][0] == ["/usr/bin/bn", "session", "stop", "owned-worker"]


def test_smoke_start_failure_still_attempts_exact_instance_stop(monkeypatch):
    from types import SimpleNamespace

    smoke = _load_bn_kernel_smoke()
    calls = []
    responses = iter(
        [
            SimpleNamespace(returncode=2, stdout="", stderr="start failed"),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
    )

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return next(responses)

    monkeypatch.setattr(smoke.shutil, "which", lambda name: "/usr/bin/bn")
    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke.py", "/tmp/sample.bin", "--instance", "failed-worker"],
    )

    assert smoke.main() == 1
    assert calls[-1] == ["/usr/bin/bn", "session", "stop", "failed-worker"]


def test_smoke_start_collision_preserves_pre_existing_instance(monkeypatch):
    from types import SimpleNamespace

    smoke = _load_bn_kernel_smoke()
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return SimpleNamespace(
            returncode=2,
            stdout="",
            stderr=(
                "error: Bridge instance already exists with id: "
                "occupied-worker\n"
            ),
        )

    monkeypatch.setattr(smoke.shutil, "which", lambda name: "/usr/bin/bn")
    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke.py", "/tmp/sample.bin", "--instance", "occupied-worker"],
    )

    assert smoke.main() == 1
    assert calls == [
        [
            "/usr/bin/bn",
            "session",
            "start",
            "/tmp/sample.bin",
            "--instance-id",
            "occupied-worker",
            "--format",
            "json",
        ]
    ]


def test_smoke_target_close_failure_does_not_suppress_instance_stop(monkeypatch):
    from types import SimpleNamespace

    smoke = _load_bn_kernel_smoke()
    calls = []
    responses = iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"loaded": [{"targets": [{"selector": "sample.bndb"}]}]}
                ),
                stderr="",
            ),
            SimpleNamespace(returncode=2, stdout="", stderr="close failed"),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
    )

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return next(responses)

    async def fake_exercise(instance, backend):
        return None

    monkeypatch.setattr(smoke.shutil, "which", lambda name: "/usr/bin/bn")
    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(smoke, "_exercise", fake_exercise)
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke.py", "/tmp/sample.bin", "--instance", "close-worker"],
    )

    assert smoke.main() == 1
    assert calls[-2] == [
        "/usr/bin/bn",
        "-i",
        "close-worker",
        "target",
        "close",
        "sample.bndb",
    ]
    assert calls[-1] == ["/usr/bin/bn", "session", "stop", "close-worker"]

def test_smoke_target_close_failure_exception_still_attempts_instance_stop(
    monkeypatch,
):
    from types import SimpleNamespace

    smoke = _load_bn_kernel_smoke()
    calls = []
    responses = iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"loaded": [{"targets": [{"selector": "sample.bndb"}]}]}
                ),
                stderr="",
            ),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
    )

    def fake_run(argv, **kwargs):
        command = list(argv)
        calls.append(command)
        if command[1:4] == ["-i", "close-exception-worker", "target"]:
            raise OSError("target close failed")
        return next(responses)

    async def fake_exercise(instance, backend):
        return None

    monkeypatch.setattr(smoke.shutil, "which", lambda name: "/usr/bin/bn")
    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(smoke, "_exercise", fake_exercise)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke.py",
            "/tmp/sample.bin",
            "--instance",
            "close-exception-worker",
        ],
    )

    assert smoke.main() == 1
    assert calls[-2] == [
        "/usr/bin/bn",
        "-i",
        "close-exception-worker",
        "target",
        "close",
        "sample.bndb",
    ]
    assert calls[-1] == [
        "/usr/bin/bn",
        "session",
        "stop",
        "close-exception-worker",
    ]


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


def test_bootstrap_without_path_context_fails_actionably():
    bootstrap = (
        Path(__file__).resolve().parents[1] / "skills" / "bn-kernel" / "bootstrap.py"
    )
    namespace = {"__builtins__": __builtins__}

    with pytest.raises(RuntimeError, match="set skill_dir.*absolute"):
        exec(
            compile(bootstrap.read_text(encoding="utf-8"), "bootstrap.py", "exec"),
            namespace,
            namespace,
        )


def test_bootstrap_keeps_foreign_bn_kernel_source_path_but_precedes_it(tmp_path):
    """A foreign/stale `bn_kernel` package elsewhere on `sys.path` (typically
    shared site-packages, which also carries unrelated dependencies) must be
    neutralized by import precedence and module eviction, never by deleting the
    `sys.path` entry that contains it -- that used to make every unrelated
    import from that root fail (#694 item 4)."""
    bootstrap = (
        Path(__file__).resolve().parents[1] / "skills" / "bn-kernel" / "bootstrap.py"
    )
    foreign_src = tmp_path / "foreign" / "src"
    foreign_package = foreign_src / "bn_kernel"
    foreign_package.mkdir(parents=True)
    (foreign_package / "__init__.py").write_text("FOREIGN = True\n")
    original_path = list(sys.path)
    sys.path.insert(0, str(foreign_src))
    namespace = {"skill_dir": bootstrap.parent}
    try:
        exec(
            compile(bootstrap.read_text(encoding="utf-8"), "bootstrap.py", "exec"),
            namespace,
            namespace,
        )
        assert str(foreign_src) in sys.path
        assert sys.path[0] == str(bootstrap.parent / "src")
        assert namespace["bn_kernel"].__name__ == "bn_kernel"
        assert not hasattr(namespace["bn_kernel"], "FOREIGN")
        assert hasattr(namespace["bn_kernel"].Session, "disasm")
    finally:
        sys.path[:] = original_path


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
        value=[],
        payload={"total": 4, "returned": 0, "has_more": True},
        notes=("note",),
        argv=("strings",),
        backend="cli",
    )

    assert result.total == 4
    assert result.returned == 0
    assert result.has_more is True
    assert result.notes == ("note",)
    assert result.argv == ("strings",)
    with pytest.raises(FrozenInstanceError):
        result.backend = "native"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("payload", "returned", "has_more"),
    [
        ("text", None, None),
        ({}, None, None),
        ({"returned": -1, "has_more": 1}, None, None),
        ({"returned": True, "has_more": "yes"}, None, None),
        ({"returned": 3, "has_more": False}, 3, False),
    ],
)
def test_result_collection_metadata_properties_are_typed(
    payload, returned, has_more
):
    result = bn_kernel.Result(
        value=[],
        payload=payload,
        notes=(),
        argv=("strings",),
        backend="cli",
    )

    assert result.returned == returned
    assert result.has_more is has_more




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
        "--help",
    )
    assert session.last.backend == "cli"


def test_help_maps_kernel_search_helper_to_cli_family(monkeypatch, tmp_path):
    script = """
import json, sys
sys.stdout.write(json.dumps(sys.argv[1:]))
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    argv = json.loads(_run(session.help("search")))

    assert argv[-3:] == ["function", "search", "--help"]


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
json.dump({"items": rows, "offset": offset, "limit": limit,
           "returned": len(rows), "total": 7,
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
json.dump({"items": rows, "offset": offset, "total": 100, "has_more": True},
          open(out, "w"))
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    rows = _run(session.all("strings", page=3, limit=5, offset="10"))

    assert [row["i"] for row in rows] == [10, 11, 12, 13, 14]
    assert session.last is not None
    assert session.last.payload["offset"] == 10
    assert session.last.payload["returned"] == 5
    assert session.last.payload["has_more"] is True
    assert session.last.payload["limit"] == 5


def test_all_zero_limit_probes_one_row_and_discards_it(monkeypatch, tmp_path):
    """`limit=0` must issue exactly one real CLI request with WIRE `--limit 1`
    (the bridge enforces `minimum=1`), validate that probed row through the
    normal page contract, then discard it -- the caller sees zero rows but the
    schema metadata (kind/total/row_fields) survives, and a probed row flips
    `has_more` true even when the bridge under-reports it."""
    counter = tmp_path / "pages.count"
    script = _page_script(
        'page = {"kind": "sections", "items": [{"name": "sec0"}], "offset": offset,\n'
        '        "limit": limit, "returned": 1, "total": 5, "has_more": False,\n'
        '        "row_fields": ["name", "start", "end", "length", "semantics"]}',
        counter=counter,
        cap=1,
    )
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    rows = _run(session.all("sections", limit=0, offset=4))

    assert rows == []
    assert counter.read_text() == "1"
    assert session.last is not None
    assert session.last.payload["offset"] == 4
    assert session.last.payload["returned"] == 0
    assert session.last.payload["limit"] == 0
    assert session.last.payload["kind"] == "sections"
    assert session.last.payload["total"] == 5
    # The bridge said has_more=False, but the probe found a row: from the
    # caller's zero-row position there IS more to fetch.
    assert session.last.payload["has_more"] is True
    assert session.last.returned == 0
    assert session.last.has_more is True
    assert session.last.payload["row_fields"] == [
        "name", "start", "end", "length", "semantics",
    ]
    assert session.last.value == []


def test_all_zero_limit_probe_finds_no_rows(monkeypatch, tmp_path):
    """When the probe itself comes back empty, `has_more` falls through to the
    bridge's own (also-false) value -- there is nothing at this offset at all.
    Also pins the wire-level limit to 1, not 0."""
    seen_limit = tmp_path / "seen_limit"
    script = _page_script(
        f'open({str(seen_limit)!r}, "w").write(str(limit))\n'
        'page = {"kind": "sections", "items": [], "offset": offset, "limit": limit,\n'
        '        "returned": 0, "total": 5, "has_more": False,\n'
        '        "row_fields": ["name", "start", "end"]}'
    )
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    rows = _run(session.all("sections", limit=0, offset=5))

    assert rows == []
    assert seen_limit.read_text() == "1"
    assert session.last is not None
    assert session.last.payload["returned"] == 0
    assert session.last.payload["limit"] == 0
    assert session.last.payload["has_more"] is False


def test_all_zero_limit_requires_row_fields(monkeypatch, tmp_path):
    script = _page_script(
        'page = {"kind": "sections", "items": [{"name": "sec0"}], "offset": offset,\n'
        '        "limit": limit, "returned": 1, "total": 3, "has_more": True}'
    )
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    with pytest.raises(bn_kernel.BnError, match="row_fields"):
        _run(session.all("sections", limit=0))

    assert session.last is None


def test_all_zero_limit_empty_page_undeclared_kind_omits_row_fields(monkeypatch, tmp_path):
    """Same real-bridge-behaviour pin as the client test: an undeclared `kind`
    (outside `_DECLARED_ROW_FIELDS`) with an empty probe page has no row to
    derive a schema from, so the bridge's real `_annotate_row_fields` leaves
    `row_fields` off entirely. That must not be treated as malformed."""
    from bn_agent_bridge._shared import _annotate_row_fields

    raw = {
        "kind": "types",
        "items": [],
        "offset": 6,
        "limit": 1,
        "returned": 0,
        "total": 0,
        "has_more": False,
    }
    page = _annotate_row_fields(dict(raw))
    assert "row_fields" not in page  # pins the real bridge behaviour this defends

    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, _payload_script(page))))
    session = bn_kernel.Session(backend="cli")

    rows = _run(session.all("types", limit=0, offset=6))

    assert rows == []
    assert session.last is not None
    assert session.last.payload["offset"] == 6
    assert session.last.payload["returned"] == 0
    assert session.last.payload["limit"] == 0
    assert session.last.payload["kind"] == "types"
    assert session.last.payload["has_more"] is False
    assert "row_fields" not in session.last.payload


def test_all_zero_limit_empty_page_declared_kind_keeps_row_fields(monkeypatch, tmp_path):
    """A declared kind's row_fields tuple is populated even with zero rows, so
    the empty-probe path must still surface it."""
    from bn_agent_bridge._shared import _annotate_row_fields, _DECLARED_ROW_FIELDS

    raw = {
        "kind": "sections",
        "items": [],
        "offset": 0,
        "limit": 1,
        "returned": 0,
        "total": 0,
        "has_more": False,
    }
    page = _annotate_row_fields(dict(raw))
    assert page["row_fields"] == list(_DECLARED_ROW_FIELDS["sections"])

    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, _payload_script(page))))
    session = bn_kernel.Session(backend="cli")

    rows = _run(session.all("sections", limit=0))

    assert rows == []
    assert session.last.payload["row_fields"] == list(_DECLARED_ROW_FIELDS["sections"])


@pytest.mark.parametrize(
    "row_fields",
    ["name", [1, 2], ["name", 2]],
    ids=["not-a-list", "non-string-items", "mixed-types"],
)
def test_all_zero_limit_empty_page_malformed_row_fields_raises(monkeypatch, tmp_path, row_fields):
    """A present-but-malformed `row_fields` on an EMPTY probe page must still
    raise -- only a genuinely ABSENT `row_fields` is legitimate for an empty
    page (undeclared kind). The malformed shapes must not slip through the
    `if probed_items:` gate just because the probe found no row, and the raise
    must still clear `Session.last`."""
    page = {
        "kind": "sections",
        "items": [],
        "offset": 0,
        "limit": 1,
        "returned": 0,
        "total": 0,
        "has_more": False,
        "row_fields": row_fields,
    }
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, _payload_script(page))))
    session = bn_kernel.Session(backend="cli")

    with pytest.raises(bn_kernel.BnError, match="row_fields"):
        _run(session.all("sections", limit=0))

    assert session.last is None


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
        "offset": 0,
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
    {"items": [{"value": "a"}], "offset": 0, "returned": 1, "total": 2,
     "has_more": True}
    if offset == 0
    else {"items": [{"value": "b"}], "offset": offset, "returned": 1, "total": 3,
          "has_more": False}
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


def test_all_between_page_timeout_carries_shared_guidance(monkeypatch, tmp_path):
    """The between-page deadline check (fired BEFORE launching the next page's
    subprocess, not from a slow subprocess's own asyncio.wait_for) used to raise
    a short hand-written message with no analysis-progress guidance. It must
    carry the same `_timeout_message` text every other bn-kernel timeout does."""
    counter = tmp_path / "pages.count"
    script = _page_script(
        'page = {"items": [{"value": str(offset)}], "offset": offset,\n'
        '        "returned": 1, "has_more": True, "total": None}',
        counter=counter,
        cap=1,
    )
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    # `time.monotonic()` is called twice directly inside `Session.all()`: once to
    # anchor the deadline, once per loop iteration to compute the remaining
    # budget. Faking it globally would also feed asyncio's own event-loop clock
    # (same `time` module) and crash unrelated internals, so intercept only the
    # calls whose caller frame is `Session.all` itself and let everything else
    # (subprocess spawn/wait) run on real time. Schedule: first call resolves the
    # deadline anchor; the second (before page 1) leaves the full budget; the
    # third (before page 2, i.e. between pages) reads as far past the deadline --
    # without any subprocess ever actually stalling.
    import inspect

    real_monotonic = bn_kernel.time.monotonic
    schedule = iter([0.0, 0.0, 100.0])

    def fake_monotonic():
        caller = inspect.currentframe().f_back
        if caller is not None and caller.f_code.co_name == "all" and caller.f_code.co_filename == bn_kernel.__file__:
            try:
                return next(schedule)
            except StopIteration:
                pass
        return real_monotonic()

    monkeypatch.setattr(bn_kernel.time, "monotonic", fake_monotonic)
    session = bn_kernel.Session(timeout=1, backend="cli")

    with pytest.raises(bn_kernel.BnError) as excinfo:
        _run(session.all("strings", page=1, timeout=5))

    message = str(excinfo.value)
    assert "requested end-to-end budget" in message
    assert "bn -i NAME target info" in message
    assert counter.read_text() == "1"


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
            return _canonical_target_info(arch="x86")
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


def test_backend_environment_is_validated_and_applies_to_auto(monkeypatch):
    monkeypatch.setenv("BN_BACKEND", "nonsense")
    with pytest.raises(ValueError, match="BN_BACKEND.*auto.*cli.*native"):
        bn_kernel.Session()
    with pytest.raises(ValueError, match="BN_BACKEND"):
        bn_kernel.Session(backend="cli")

    monkeypatch.setenv("BN_BACKEND", "cli")
    assert bn_kernel.Session().backend == "cli"


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


def test_scoped_cancellation_during_registration_cleans_binding(monkeypatch):
    class CancelAfterSet(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            raise asyncio.CancelledError

    bindings = CancelAfterSet()
    monkeypatch.setattr(bn_kernel, "_ACTIVE_SCOPED_BINDINGS", bindings)

    async def callback(session):
        return session.instance

    with pytest.raises(asyncio.CancelledError):
        _run(
            bn_kernel.scoped(
                callback,
                instance="worker",
                target="sample",
                backend="cli",
            )
        )

    assert bindings == {}
    assert bn_kernel._ACTIVE_SCOPED_CALLBACKS == set()


def test_native_request_runs_off_event_loop_thread_and_updates_last():
    session, client = _native_session()
    loop_thread = threading.get_ident()

    value = _run(session.info(verbose=True))

    expected = _canonical_target_info(arch="x86")
    assert value == expected
    assert client.thread_ids and client.thread_ids[0] != loop_thread
    assert client.calls == [("request", "target_info", {"verbose": True}, {})]
    assert session.last == bn_kernel.Result(
        value=expected,
        payload=expected,
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
    message = str(caught.value)
    assert "requested end-to-end budget" in message
    assert "bn -i NAME target info" in message


@pytest.mark.parametrize("method", ["request", "collect"])
def test_native_timeout_enforces_wall_clock_budget(method):
    session = bn_kernel.Session(instance="worker", timeout=0.03, backend="native")

    class SlowClient:
        def request(self, op, params=None):
            time.sleep(0.2)
            return {"arch": "x86"}

        def collect(self, op, params=None, *, limit=None):
            time.sleep(0.2)
            return {"items": [], "total": 0, "has_more": False}

    session._client = SlowClient()
    async def scenario():
        operation = session.info() if method == "request" else session.strings()
        started = time.monotonic()
        with pytest.raises(bn_kernel.BnError, match="timed out after 0.03s"):
            await operation
        return time.monotonic() - started

    assert _run(scenario()) < 0.1
    assert session.last is None



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
    assert session.last is None



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
    assert session.last is None


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


def test_strings_defaults_to_bounded_page_but_allows_explicit_full_collection():
    session, client = _native_session()

    _run(session.strings())
    _run(session.strings(limit=None))

    assert client.calls[-2:] == [
        ("collect", "strings", {}, {"limit": 100}),
        ("collect", "strings", {}, {"limit": None}),
    ]


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
        ("decompile", ("main",), {"addresses": True, "force_analysis": True}, "request", "decompile", {"identifier": "main", "addresses": True, "force_analysis": True, "include_annotations": False}, {}),
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


def test_native_search_dot_retries_even_when_literal_dot_matches_subset():
    session = bn_kernel.Session(instance="worker", backend="native")
    calls = []

    class SearchClient:
        def collect(self, op, params=None, *, limit=None):
            calls.append(dict(params or {}))
            if params.get("regex"):
                rows = [
                    {"name": "first", "size": 12, "size_known": True},
                    {"name": "second", "size": 12, "size_known": True},
                ]
            else:
                rows = [
                    {"name": "literal.dot", "size": 12, "size_known": True}
                ]
            return {
                "items": rows,
                "offset": 0,
                "returned": len(rows),
                "total": len(rows),
                "has_more": False,
            }

    session._client = SearchClient()

    assert len(_run(session.search("."))) == 2
    assert calls == [{"query": "."}, {"query": ".", "regex": True}]
    assert session.last is not None
    assert session.last.payload["regex_fallback"] is True


def test_native_search_rejects_invalid_regex_like_literal():
    session = bn_kernel.Session(instance="worker", backend="native")

    class SearchClient:
        def collect(self, op, params=None, *, limit=None):
            return {"items": [], "total": 0, "has_more": False}

    session._client = SearchClient()

    with pytest.raises(bn_kernel.BnError, match="invalid regex-like.*exact=True"):
        _run(session.search("[invalid"))
    assert session.last is None


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
            return _canonical_target_info(
                basename=filename.name, filename=str(filename)
            )

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
            return _canonical_target_info(
                basename="foreign.bndb",
                filename=str(tmp_path / "foreign.bndb"),
            )

    native._client = IdentityClient()
    with pytest.raises(bn_kernel.BridgeError, match="target identity mismatch"):
        _run(native.assert_target("expected.bndb"))

    payload = _canonical_target_info(
        basename="foreign.bndb",
        filename=str(tmp_path / "foreign.bndb"),
    )
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


@pytest.mark.parametrize(
    "malformed_locations",
    [42, {"address": "0x1234"}],
    ids=["int", "mapping"],
)
def test_assert_unannotated_survives_malformed_comment_locations(
    malformed_locations,
):
    """`comment_locations`/`function_comment_locations` are OPTIONAL rendering
    detail, not part of the contamination verdict. A shape this gate cannot
    read as a list (an int, a mapping) used to spread into a list literal and
    crash with a bare TypeError, converting a real contamination finding into
    an unhandled exception instead of the refusal it must still raise (#694
    item 12)."""
    session = bn_kernel.Session(instance="worker", backend="native")
    digest = {
        "existing_annotations": {
            "comments": 1,
            "comment_locations": malformed_locations,
            "function_comments": 0,
            "function_comment_locations": [],
            "user_symbols": 0,
        }
    }

    class OrientClient:
        def request(self, op, params=None):
            return digest

    session._client = OrientClient()

    with pytest.raises(bn_kernel.BridgeError, match="inherited comments") as caught:
        _run(session.assert_unannotated())

    assert "comments=1" in str(caught.value)
    assert "malformed" in str(caught.value)



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
    assert session.last is None


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
json.dump({"items": [row], "offset": 0, "has_more": False, "total": 1},
          open(out, "w"))
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
json.dump({"items": [row], "offset": 0, "has_more": False, "total": 1},
          open(out, "w"))
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
json.dump({"items": [row], "offset": 0, "has_more": False, "total": 1},
          open(out, "w"))
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
    payload = {"items": [{"op": "list_functions", "size": 12, "size_known": True}], "offset": 0, "has_more": False, "total": 1}
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
json.dump({"items": [{"argv": argv}], "offset": 0, "has_more": False, "total": 1},
          open(out, "w"))
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


def test_brief_missing_key_error_lists_the_actual_row_keys():
    # The exact VIBE24 failure: `brief(sections_rows, "address", "size")` raised
    # KeyError('address'), suggested dotted paths, and never mentioned that
    # section rows use start/end/length/name -- so the agent had to re-run a read
    # just to discover the schema. Row schemas differ ON PURPOSE across
    # collections; the error is the only place that can say which one you hit.
    rows = [{"start": "0x1000", "end": "0x2000", "length": 4096, "name": ".text"}]

    with pytest.raises(KeyError) as exc:
        bn_kernel.brief(rows, "address", "size")

    message = str(exc.value)
    assert "brief key 'address' is missing at row 0" in message
    assert "available keys" in message
    for key in ("start", "end", "length", "name"):
        assert f"'{key}'" in message
    # Flat rows have no nested mappings, so dotted-path advice would be noise
    # that sends the reader looking for a nesting level that does not exist.
    assert "dotted" not in message


def test_brief_missing_key_error_keeps_dotted_guidance_when_rows_nest():
    # callsites rows DO nest (callee/containing_function), so here the dotted
    # path is the actual fix and must still be offered -- naming the specific
    # nested keys rather than a generic example.
    rows = [{
        "callee": {"name": "memcpy", "address": "0x1000"},
        "containing_function": {"name": "parse", "address": "0x2000"},
        "call_addr": "0x1010",
    }]

    with pytest.raises(KeyError) as exc:
        bn_kernel.brief(rows, "name")

    message = str(exc.value)
    assert "available keys" in message
    assert "'callee'" in message and "'call_addr'" in message
    assert "dotted" in message
    assert "callee.*" in message and "containing_function.*" in message


def test_brief_nested_miss_reports_the_row_keys_too():
    rows = [{"callee": {"name": "memcpy"}, "call_addr": "0x1010"}]

    with pytest.raises(KeyError) as exc:
        bn_kernel.brief(rows, "callee.address")

    message = str(exc.value)
    assert "brief key 'callee.address' is missing at row 0" in message
    assert "available keys" in message
    assert "'callee'" in message


def test_result_exposes_the_row_field_hint_from_the_payload():
    # The hint rides the payload the bridge already returns, so there is no
    # second catalog to drift: whatever the read declared is what `.row_fields`
    # reports, and a payload without the hint reports None rather than guessing.
    with_hint = bn_kernel.Result(
        value=[{"start": "0x1000"}],
        payload={"kind": "sections", "items": [{"start": "0x1000"}],
                 "row_fields": ["name", "start", "end", "length", "semantics"]},
        notes=(),
        argv=("sections",),
        backend="cli",
    )
    assert with_hint.row_fields == [
        "name", "start", "end", "length", "semantics",
    ]

    without_hint = bn_kernel.Result(
        value=[], payload={"kind": "sections", "items": []}, notes=(),
        argv=("sections",), backend="cli",
    )
    assert without_hint.row_fields is None

    text_payload = bn_kernel.Result(
        value="listing", payload="listing", notes=(), argv=("disasm",), backend="cli",
    )
    assert text_payload.row_fields is None


# --- `Session.last is None` after a failed operation ----------------------
#
# The VIBE24 report called stale success state a footgun: after a read failed,
# `s.last` could still hold the PREVIOUS read's rows, and an agent inspecting
# `s.last.payload` for diagnostics would silently read someone else's data as if
# it belonged to the failed call. These reproduce the five representative failure
# shapes -- request, mid-pagination, function row contract, callsite
# attribution, invalid regex-like query -- on both backends and pin the contract:
# a failed operation leaves NO `last`, and it is not enough that the call raised.
#
# The precondition matters: each session first performs a SUCCESSFUL read, so a
# passing assertion proves the failure actively cleared state rather than there
# never having been any.

def _seeded_cli_session(monkeypatch, tmp_path, script: str) -> bn_kernel.Session:
    """A CLI-backed session whose `last` already holds a successful result."""
    seed = _fake_bn(
        tmp_path,
        _payload_script({
            "kind": "strings", "items": [{"address": "0x1", "value": "seed"}],
            "offset": 0, "limit": None, "returned": 1, "total": 1,
            "has_more": False,
        }),
    )
    monkeypatch.setenv("BN_BIN", str(seed))
    session = bn_kernel.Session(backend="cli")
    _run(session.strings())
    assert session.last is not None and session.last.value
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path / "second", script)))
    return session


@pytest.fixture
def second_dir(tmp_path):
    (tmp_path / "second").mkdir()
    return tmp_path


def test_last_cleared_after_request_failure(monkeypatch, second_dir):
    session = _seeded_cli_session(monkeypatch, second_dir, """
import sys
sys.stderr.write("bridge refused the request")
sys.exit(2)
""")

    with pytest.raises(bn_kernel.BridgeError, match="bridge refused"):
        _run(session.functions())

    assert session.last is None


def test_last_cleared_after_mid_pagination_failure(monkeypatch, second_dir):
    session = _seeded_cli_session(monkeypatch, second_dir, """
import json, sys
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
offset = int(argv[argv.index("--offset") + 1]) if "--offset" in argv else 0
if offset == 0:
    json.dump(
        {"kind": "strings", "items": [{"address": "0x1", "value": "page-one"}],
         "offset": 0, "limit": 1, "returned": 1, "total": 4, "has_more": True},
        open(out, "w"),
    )
else:
    sys.stderr.write("page two exploded")
    sys.exit(2)
""")

    with pytest.raises(bn_kernel.BridgeError, match="page two exploded"):
        _run(session.all("strings", page=1))

    # The first page SUCCEEDED and its rows are gone: a partial collection that
    # survived as `last` would look like a complete answer.
    assert session.last is None


def test_last_cleared_after_function_row_contract_failure(monkeypatch, second_dir):
    session = _seeded_cli_session(monkeypatch, second_dir, _payload_script({
        "kind": "functions",
        # Rows without the size/size_known contract: a successful bn exit with a
        # payload the kernel refuses to vouch for.
        "items": [{"address": "0x401000", "name": "parse"}],
        "offset": 0, "limit": None, "returned": 1, "total": 1, "has_more": False,
    }))

    with pytest.raises(bn_kernel.BnError, match="row contract violation"):
        _run(session.functions())

    # bn exited 0, so `last` was legitimately populated before validation ran.
    # It must not survive the contract rejection.
    assert session.last is None


def test_last_cleared_after_callsite_attribution_failure(monkeypatch, second_dir):
    session = _seeded_cli_session(monkeypatch, second_dir, _payload_script({
        "kind": "callsites",
        # call_addr present but unattributed: no containing_function, so the row
        # cannot support a source-to-sink claim.
        "items": [{"callee": {"name": "sink", "address": "0x2000"},
                   "call_addr": "0x1010"}],
        "offset": 0, "limit": None, "returned": 1, "total": 1, "has_more": False,
    }))

    with pytest.raises(bn_kernel.BnError, match="callsites row contract violation"):
        _run(session.callsites("sink"))

    assert session.last is None


def test_last_cleared_after_invalid_regex_like_search_on_both_backends(monkeypatch, second_dir):
    # Native: the kernel itself owns the gate, and it fires AFTER a successful
    # zero-hit literal pass has already populated `last`.
    # Same (instance, target) binding as the CLI session below: the point under
    # test is `last` invalidation, not the cross-binding warning.
    session = bn_kernel.Session(backend="native")

    class ZeroHitClient:
        def collect(self, op, params=None, *, limit=None):
            return {"items": [], "total": 0, "has_more": False}

    session._client = ZeroHitClient()
    assert _run(session.search("NoSuchLiteral")) == []
    assert session.last is not None

    with pytest.raises(bn_kernel.BnError, match="invalid regex-like search query"):
        _run(session.search("[invalid"))
    assert session.last is None

    # CLI: the gate lives in `bn` itself (bn.cli._should_retry_as_regex), which
    # exits 2 rather than reporting a confident empty result. The kernel must
    # propagate that and drop the previous read's rows -- the fake mirrors the
    # real CLI's refusal for this query shape.
    cli_session = _seeded_cli_session(monkeypatch, second_dir, """
import sys
sys.stderr.write("invalid regex-like search query '[invalid': unterminated "
                 "character set at position 0; pass --exact to search for it literally")
sys.exit(2)
""")

    with pytest.raises(bn_kernel.BnError, match="invalid regex-like search query"):
        _run(cli_session.search("[invalid"))
    assert cli_session.last is None


def test_last_cleared_after_native_request_and_contract_failures():
    # Same contract on the native backend, where `last` is set from an in-process
    # client rather than a CLI artifact.
    session, client = _native_session()
    _run(session.decompile("main"))
    assert session.last is not None

    def failing_request(op, params=None):
        raise NativeBridgeError("native bridge refused")

    client.request = failing_request
    with pytest.raises(bn_kernel.BnError, match="native bridge refused"):
        _run(session.decompile("main"))
    assert session.last is None

    session, client = _native_session()
    _run(session.functions())
    assert session.last is not None

    def bad_rows(op, params=None, *, limit=None):
        return {"items": [{"address": "0x401000"}], "offset": 0, "returned": 1,
                "total": 1, "has_more": False}

    client.collect = bad_rows
    with pytest.raises(bn_kernel.BnError, match="row contract violation"):
        _run(session.functions())
    assert session.last is None


def test_assert_unannotated_keeps_its_successful_payload():
    # The other half of the contract: clearing `last` on failure must not throw
    # away a payload that DID succeed. A contamination refusal is a policy
    # verdict on a good orientation read, and the digest behind it is exactly
    # what an agent needs to decide whether to proceed.
    session = bn_kernel.Session(instance="worker", target="sample", backend="native")

    class OrientClient:
        def request(self, op, params=None):
            assert op == "orient_digest"
            return {
                "kind": "orient_digest",
                "target": {"basename": "sample.bndb"},
                "function_count": 3,
                # The bridge always publishes all three counters
                # (read_listing.py `_existing_annotations`), and the gate now
                # requires them so an unreadable digest cannot pass as clean.
                "existing_annotations": {
                    "comments": 2,
                    "function_comments": 0,
                    "user_symbols": 0,
                    "comment_locations": [
                        {"address": "0x401000", "name": "parse_packet"},
                    ],
                },
            }

    session._client = OrientClient()
    with pytest.raises(bn_kernel.BnError, match="0x401000"):
        _run(session.assert_unannotated())

    assert session.last is not None
    assert session.last.payload["existing_annotations"]["comments"] == 2


# -- #694 pre-review blockers -------------------------------------------------


def _shape_session(payload):
    """A native session whose single fake op always returns *payload*."""
    session = bn_kernel.Session(instance="worker", target="sample", backend="native")

    class OnePayload:
        def request(self, op, params=None):
            return payload

        def collect(self, op, params=None, *, limit=None):
            return payload

    session._client = OnePayload()
    return session


def _seeded_native_session():
    """A native session whose `last` already holds a real successful page."""
    session, client = _native_session()
    rows = _run(session.functions(limit=1))
    assert rows and session.last is not None
    return session


MALFORMED_DIGESTS = [
    ("digest is a list", ["orient"]),
    ("digest is a string", "orientation unavailable"),
    ("digest is null", None),
    ("no existing_annotations", {"kind": "orient", "entry": "0x401000"}),
    ("existing_annotations is a list", {"existing_annotations": ["comments"]}),
    ("existing_annotations is null", {"existing_annotations": None}),
    (
        "counter is not an integer",
        {"existing_annotations": {"comments": "2", "function_comments": 0,
                                  "user_symbols": 0}},
    ),
    (
        "counter is a bool",
        {"existing_annotations": {"comments": True, "function_comments": 0,
                                  "user_symbols": 0}},
    ),
    (
        "counter is negative",
        {"existing_annotations": {"comments": -1, "function_comments": 0,
                                  "user_symbols": 0}},
    ),
    (
        "counter is missing",
        {"existing_annotations": {"comments": 0, "function_comments": 0}},
    ),
]


@pytest.mark.parametrize(
    "label,payload", MALFORMED_DIGESTS, ids=[case[0] for case in MALFORMED_DIGESTS]
)
def test_assert_unannotated_fails_closed_on_a_malformed_digest(label, payload):
    """The contamination gate must never certify a shape it cannot read. Silently
    treating an unreadable digest as `comments=0` reports contaminated data clean."""
    session = _shape_session(payload)

    with pytest.raises(bn_kernel.BnError, match="orient digest contract violation"):
        _run(session.assert_unannotated())

    assert session.last is None


@pytest.mark.parametrize(
    "label,payload", MALFORMED_DIGESTS, ids=[case[0] for case in MALFORMED_DIGESTS]
)
def test_assert_unannotated_rejects_malformed_digest_even_when_bypassed(
    label, payload
):
    """`allow_contaminated=True` waives the contamination *policy*, not the payload
    contract; an unreadable digest is still an unreadable digest."""
    session = _shape_session(payload)

    with pytest.raises(bn_kernel.BnError, match="orient digest contract violation"):
        _run(session.assert_unannotated(allow_contaminated=True))

    assert session.last is None


def test_assert_unannotated_accepts_a_well_formed_clean_digest():
    session = _shape_session(
        {
            "kind": "orient",
            "existing_annotations": {
                "comments": 0,
                "function_comments": 0,
                "user_symbols": 7,
            },
        }
    )

    digest = _run(session.assert_unannotated())

    assert digest["existing_annotations"]["user_symbols"] == 7
    assert session.last is not None


PREFLIGHT_CASES = [
    ("functions unknown kwarg", lambda s: s.functions(bogus=1), TypeError),
    ("search unknown kwarg", lambda s: s.search("parse", bogus=1), TypeError),
    ("strings unknown kwarg", lambda s: s.strings(bogus=1), TypeError),
    ("imports unknown kwarg", lambda s: s.imports(bogus=1), TypeError),
    ("sections unknown kwarg", lambda s: s.sections(bogus=1), TypeError),
    (
        "disasm count and lines",
        lambda s: s.disasm("main", count=2, lines=(1, 2)),
        ValueError,
    ),
    ("disasm zero count", lambda s: s.disasm("main", count=0), ValueError),
    ("disasm line zero", lambda s: s.disasm("main", lines=(0, 5)), ValueError),
    ("xrefs empty identifier", lambda s: s.xrefs(""), ValueError),
    ("callsites empty within", lambda s: s.callsites("sink", within=[]), ValueError),
]


@pytest.mark.parametrize(
    "label,call,error", PREFLIGHT_CASES, ids=[case[0] for case in PREFLIGHT_CASES]
)
def test_preflight_validation_error_clears_last(label, call, error):
    """A raise must never leave the previous read's rows behind as `last`. Preflight
    checks fire before the backend method that clears it, so they need their own."""
    session = _seeded_native_session()

    with pytest.raises(error):
        _run(call(session))

    assert session.last is None


def test_assert_target_mismatch_clears_last(tmp_path):
    """A foreign target is a failure, not the documented `assert_unannotated`
    contamination exception; leaving the foreign digest in `last` invites a read
    of another binary's rows."""
    session = _shape_session(
        _canonical_target_info(
            filename="/tmp/actual.bndb", basename="actual.bndb"
        )
    )

    with pytest.raises(bn_kernel.BridgeError, match="target identity mismatch"):
        _run(session.assert_target("expected.bndb"))

    assert session.last is None


SHAPE_CASES = [
    ("info non-mapping", "info", lambda s: s.info(), ["not", "a", "mapping"]),
    ("info empty mapping", "info", lambda s: s.info(), {}),
    (
        "info bad counter",
        "info",
        lambda s: s.info(),
        {"filename": "/tmp/sample.bndb", "function_count": -3},
    ),
    ("il non-mapping", "il", lambda s: s.il("main"), ["hlil"]),
    ("il no text key", "il", lambda s: s.il("main"), {"kind": "il", "view": "hlil"}),
    ("il empty text", "il", lambda s: s.il("main"), {"text": "   "}),
    (
        "xrefs rows are not mappings",
        "xrefs",
        lambda s: s.xrefs("main"),
        {"items": ["0x401000"], "offset": 0, "returned": 1, "has_more": False,
         "total": 1},
    ),
    (
        "imports rows are not mappings",
        "imports",
        lambda s: s.imports(),
        {"items": ["printf"], "offset": 0, "returned": 1, "has_more": False,
         "total": 1},
    ),
    (
        "sections rows nest a page envelope",
        "sections",
        lambda s: s.sections(),
        {"items": [{"items": [], "has_more": False}], "offset": 0, "returned": 1,
         "has_more": False, "total": 1},
    ),
]


@pytest.mark.parametrize(
    "label,helper,call,payload", SHAPE_CASES, ids=[case[0] for case in SHAPE_CASES]
)
def test_native_helper_shape_parity_rejects_malformed_payloads(
    label, helper, call, payload
):
    session = _shape_session(payload)

    with pytest.raises(bn_kernel.BnError):
        _run(call(session))

    assert session.last is None


@pytest.mark.parametrize(
    "label,helper,call,payload", SHAPE_CASES, ids=[case[0] for case in SHAPE_CASES]
)
def test_cli_helper_shape_parity_rejects_malformed_payloads(
    monkeypatch, tmp_path, label, helper, call, payload
):
    """The CLI backend reached its rows through a different code path, so the
    validators have to sit after the backend branch, not inside one of them."""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, _payload_script(payload))))
    session = bn_kernel.Session(instance="worker", target="sample", backend="cli")

    with pytest.raises(bn_kernel.BnError):
        _run(call(session))

    assert session.last is None


def test_cross_binding_warning_fires_once_per_distinct_pair():
    """One warning per novel pair, and per registration.

    Steps 1-3 build up bindings A, B, C, recording {A,B} then {A,C}. Step 4
    re-registers C while B is live: {A,C} is already known, so a check that only
    consulted the lowest-sorting binding short-circuited there and never disclosed
    the novel {B,C} overlap. Step 5 proves the disclosure does not then repeat once
    every relevant pair is recorded.
    """
    alpha = ("bnk-a", "alpha.bndb")
    bravo = ("bnk-b", "bravo.bndb")
    charlie = ("bnk-c", "charlie.bndb")
    held = []
    counts = []

    for instance, target in [alpha, bravo, charlie, charlie, charlie]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            held.append(
                bn_kernel.Session(instance=instance, target=target, backend="cli")
            )
        counts.append(
            len([item for item in caught if item.category is RuntimeWarning])
        )

    #        A   B    C    C(novel {B,C})   C(all pairs known)
    assert counts == [0, 1, 1, 1, 0]
    assert bn_kernel._WARNED_BINDING_PAIRS == {
        frozenset((alpha, bravo)),
        frozenset((alpha, charlie)),
        frozenset((bravo, charlie)),
    }


def test_cross_binding_warning_is_not_repeated_for_a_known_pair():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        held = [
            bn_kernel.Session(instance="bnk-a", target="alpha.bndb", backend="cli"),
            bn_kernel.Session(instance="bnk-b", target="bravo.bndb", backend="cli"),
        ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        held.append(
            bn_kernel.Session(instance="bnk-b", target="bravo.bndb", backend="cli")
        )

    assert [item for item in caught if item.category is RuntimeWarning] == []


def _page_script(body: str, *, counter: Path | None = None, cap: int = 0) -> str:
    """A fake `bn` that emits one page per call.

    `counter`/`cap` add a persistent page guard so a regression test for an
    *unbounded* loop fails fast on the old implementation instead of paging
    forever until the harness times out.
    """
    guard = ""
    if counter is not None:
        guard = f"""
try:
    seen = int(open({str(counter)!r}).read() or "0")
except OSError:
    seen = 0
open({str(counter)!r}, "w").write(str(seen + 1))
if seen + 1 > {cap}:
    sys.stderr.write("page guard tripped after {cap} pages")
    sys.exit(1)
"""
    return f"""
import json, os, sys
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
limit = int(argv[argv.index("--limit") + 1])
offset = int(argv[argv.index("--offset") + 1])
{guard}
{body}
json.dump(page, open(out, "w"))
"""


def test_all_rejects_a_page_whose_offset_does_not_echo_the_request(
    monkeypatch, tmp_path
):
    script = _page_script(
        'page = {"items": [{"value": "a"}, {"value": "b"}], "offset": 0,\n'
        '        "returned": 2, "has_more": True, "total": None}'
    )
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    with pytest.raises(bn_kernel.BnError, match=r"offset 0 for requested offset 2"):
        _run(session.all("strings", page=2, limit=6))

    assert session.last is None


def test_all_rejects_a_page_larger_than_the_requested_page_size(monkeypatch, tmp_path):
    script = _page_script(
        'page = {"items": [{"value": str(i)} for i in range(limit * 3)],\n'
        '        "offset": offset, "returned": limit * 3, "has_more": False,\n'
        '        "total": limit * 3}'
    )
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    with pytest.raises(
        bn_kernel.BnError, match=r"returned 6 items for requested limit 2"
    ):
        _run(session.all("strings", page=2, limit=2))

    assert session.last is None


def test_all_refuses_an_intrinsically_unbounded_collection(monkeypatch, tmp_path):
    counter = tmp_path / "pages.count"
    script = _page_script(
        'page = {"items": [{"value": str(offset)}], "offset": offset,\n'
        '        "returned": 1, "has_more": True, "total": None}',
        counter=counter,
        cap=6,
    )
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "off")
    session = bn_kernel.Session(backend="cli")

    with pytest.raises(bn_kernel.BnError, match="intrinsically unbounded"):
        _run(session.all("strings", page=1))

    assert int(counter.read_text()) == 1

    assert session.last is None


def test_all_keeps_a_bounded_collection_working_without_a_deadline(
    monkeypatch, tmp_path
):
    """`callsites` defaults to a finite limit; disabling the request timeout must
    not turn that supported shape into a refusal."""
    script = _page_script(
        'page = {"items": [{"value": str(offset)}], "offset": offset,\n'
        '        "returned": 1, "has_more": True, "total": None}'
    )
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "0")
    session = bn_kernel.Session(backend="cli")

    rows = _run(session.all("callsites", "sink", page=1, limit=3))

    assert [row["value"] for row in rows] == ["0", "1", "2"]
    assert session.last is not None
    assert session.last.payload["has_more"] is True


def test_all_applies_the_env_override_to_its_outer_multi_page_deadline(
    monkeypatch, tmp_path
):
    """BN_REQUEST_TIMEOUT is the end-to-end budget. The CLI fallback computed its
    outer deadline from `self.timeout` alone, so N slow pages could each stay under
    the env limit while the collection blew far past it."""
    script = _page_script(
        "import time\n"
        "time.sleep(0.25)\n"
        'page = {"items": [{"value": str(offset)}], "offset": offset,\n'
        '        "returned": 1, "has_more": True, "total": 100}'
    )
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "0.6")
    session = bn_kernel.Session(backend="cli")
    session.timeout = 60.0

    started = time.monotonic()
    with pytest.raises(bn_kernel.BnError) as caught:
        _run(session.all("strings", page=1, limit=8))
    elapsed = time.monotonic() - started

    assert caught.value.returncode == 124
    assert elapsed < 2.5, elapsed
    assert session.last is None


def test_run_hands_the_remaining_budget_to_the_child_process(monkeypatch, tmp_path):
    """The child re-reads BN_REQUEST_TIMEOUT from its own environment, so a page
    must be told the remaining budget or bridge cancellation waits for the
    original full value."""
    script = _page_script(
        'page = {"items": [{"value": os.environ.get("BN_REQUEST_TIMEOUT", "unset")}],\n'
        '        "offset": offset, "returned": 1, "has_more": offset < 2,\n'
        '        "total": 3}'
    )
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "30")
    session = bn_kernel.Session(backend="cli")

    rows = _run(session.all("strings", page=1))

    budgets = [float(row["value"]) for row in rows]
    assert len(budgets) == 3
    assert all(budget <= 30.0 for budget in budgets)
    assert budgets[0] > budgets[1] > budgets[2], budgets


def test_run_disables_the_child_budget_when_the_env_override_is_off(
    monkeypatch, tmp_path
):
    script = _payload_script({"text": "ok"})
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "none")
    session = bn_kernel.Session(backend="cli")

    assert _run(session.run("target", "info")) == "ok"


def test_session_rejects_an_invalid_env_request_timeout(monkeypatch, tmp_path):
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, _payload_script({"a": 1}))))
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "banana")
    session = bn_kernel.Session(backend="cli")

    with pytest.raises(bn_kernel.BnError, match="BN_REQUEST_TIMEOUT"):
        _run(session.run("target", "info"))


def _bootstrap_child_code(cache_stub: str = "") -> str:
    bootstrap = (
        Path(__file__).resolve().parents[1] / "skills" / "bn-kernel" / "bootstrap.py"
    )
    return (
        "import pathlib, shutil, sys\n"
        f"{cache_stub}"
        f"source = pathlib.Path({str(bootstrap)!r}).read_text()\n"
        f"scope = {{'skill_dir': pathlib.Path({str(bootstrap.parent)!r})}}\n"
        "exec(compile(source, 'bootstrap.py', 'exec'), scope, scope)\n"
        "assert scope['bn_kernel'].__name__ == 'bn_kernel'\n"
    )


def test_bootstrap_tolerates_losing_the_cache_claim_race():
    """Eviction is an atomic claim-by-rename. Losing the top-level rename means a
    sibling already claimed the directory, so no stale bytecode remains here."""
    stub = (
        "import pathlib as _pl\n"
        "def _lost(self, target):\n"
        "    raise FileNotFoundError(2, 'No such file or directory', str(self))\n"
        "_pl.Path.rename = _lost\n"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", _bootstrap_child_code(stub)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "bn_kernel bootstrap:" in result.stdout


def test_bootstrap_propagates_a_permission_failure_claiming_the_cache():
    """A permission failure means stale bytecode may still be on disk and win the
    import; that must stay loud rather than be swallowed with the race."""
    stub = (
        "import pathlib as _pl\n"
        "def _denied(self, target):\n"
        "    raise PermissionError(13, 'Permission denied', str(self))\n"
        "_pl.Path.rename = _denied\n"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", _bootstrap_child_code(stub)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "PermissionError" in result.stderr


def test_bootstrap_does_not_swallow_a_failure_removing_the_claimed_cache():
    """After a successful claim the directory is private to this process, so a
    FileNotFoundError from inside the tree is a real bug, not the sibling race --
    swallowing it here is what would let stale bytecode survive unnoticed."""
    stub = (
        "import pathlib as _pl\n"
        "_pl.Path.rename = lambda self, target: None\n"
        "def _vanished(path, *args, **kwargs):\n"
        "    raise FileNotFoundError(2, 'No such file or directory', str(path))\n"
        "shutil.rmtree = _vanished\n"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", _bootstrap_child_code(stub)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr


def test_bootstrap_survives_concurrent_cache_eviction(tmp_path):
    """Every fresh eval-agent process takes the eviction branch. With a shared
    installed skill they race on one __pycache__ directory."""
    import shutil as _shutil

    source_skill = (
        Path(__file__).resolve().parents[1] / "skills" / "bn-kernel"
    )
    skill = tmp_path / "bn-kernel"
    _shutil.copytree(
        source_skill, skill, ignore=_shutil.ignore_patterns("__pycache__")
    )
    cache = skill / "src" / "bn_kernel" / "__pycache__"
    cache.mkdir(parents=True)
    for index in range(1500):
        (cache / f"stale_{index}.pyc").write_bytes(b"\x00" * 64)

    barrier = tmp_path / "GO"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import pathlib, time\n"
        f"barrier = pathlib.Path({str(barrier)!r})\n"
        "while not barrier.exists():\n"
        "    time.sleep(0.002)\n"
        f"skill_dir = {str(skill)!r}\n"
        f"exec(open({str(skill / 'bootstrap.py')!r}).read())\n"
    )

    processes = [
        subprocess.Popen(
            [sys.executable, "-I", str(worker)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(12)
    ]
    time.sleep(0.4)
    barrier.write_text("go")

    failures = []
    for process in processes:
        _out, err = process.communicate(timeout=180)
        if process.returncode != 0:
            failures.append(err.strip().splitlines()[-1:] or ["(no stderr)"])

    assert not failures, failures
    assert not cache.exists()


# -- #694 remediation round 2 -------------------------------------------------


def test_native_search_shares_one_budget_across_literal_and_regex_phases(monkeypatch):
    """`search` resolves BN_REQUEST_TIMEOUT once, but `_native_collect` used to
    re-resolve each remaining slice back to the full env value, so a two-phase
    literal->regex search could spend the budget twice.

    Margins are deliberately loose: a 0.5s budget against 0.3s phases leaves phase
    one 0.2s of slack under CI load, while the old behaviour still needed ~0.6s
    (0.3s per phase, each with a freshly re-expanded budget) and so still fails.
    """
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "0.5")
    session = bn_kernel.Session(instance="worker", backend="native")

    class SlowSearchClient:
        def __init__(self):
            self.phases: list[bool] = []

        def collect(self, op, params=None, *, limit=None):
            self.phases.append(bool((params or {}).get("regex")))
            time.sleep(0.3)
            return {
                "items": [],
                "offset": 0,
                "returned": 0,
                "total": 0,
                "has_more": False,
            }

    client = SlowSearchClient()
    session._client = client

    started = time.monotonic()
    with pytest.raises(bn_kernel.BnError) as caught:
        _run(session.search("."))
    elapsed = time.monotonic() - started

    assert caught.value.returncode == 124
    assert client.phases == [False, True]
    assert elapsed < 1.2, elapsed
    assert session.last is None


def test_all_requires_the_page_to_echo_an_integer_offset(monkeypatch, tmp_path):
    """A bridge repeating page 1 while omitting `offset` slipped past every other
    check when `total` was known, so a bounded collection returned duplicates."""
    script = _page_script(
        'page = {"items": [{"value": "a"}, {"value": "b"}], "returned": 2,\n'
        '        "has_more": True, "total": 4}'
    )
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    with pytest.raises(bn_kernel.BnError, match="did not publish an integer offset"):
        _run(session.all("strings", page=2, limit=4))

    assert session.last is None


@pytest.mark.parametrize(
    "echoed", ['"0"', "True", "None"], ids=["str", "bool", "null"]
)
def test_all_rejects_a_non_integer_echoed_offset(monkeypatch, tmp_path, echoed):
    script = _page_script(
        'page = {"items": [{"value": "a"}], "returned": 1, "has_more": False,\n'
        f'        "total": 1, "offset": {echoed}}}'
    )
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    with pytest.raises(bn_kernel.BnError, match="integer offset"):
        _run(session.all("strings", page=2, limit=4))

    assert session.last is None


TOTAL_TRANSITIONS = [
    (
        "int to none",
        '{"items": [{"value": "a"}], "offset": 0, "returned": 1, "has_more": True,'
        ' "total": 5}',
        '{"items": [{"value": "b"}], "offset": 1, "returned": 1, "has_more": True}',
    ),
    (
        "int to different int",
        '{"items": [{"value": "a"}], "offset": 0, "returned": 1, "has_more": True,'
        ' "total": 5}',
        '{"items": [{"value": "b"}], "offset": 1, "returned": 1, "has_more": True,'
        ' "total": 9}',
    ),
]


@pytest.mark.parametrize(
    "label,first,second",
    TOTAL_TRANSITIONS,
    ids=[case[0] for case in TOTAL_TRANSITIONS],
)
def test_all_rejects_any_cross_page_total_transition(
    monkeypatch, tmp_path, label, first, second
):
    """`total` is monotone (#694 item 3): once a page has DETERMINED a total, a
    later page dropping it back to null or reporting a different int is drift,
    not a legitimate refinement, and must still be rejected."""
    script = _page_script(
        f"page = {first} if offset == 0 else {second}"
    )
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    with pytest.raises(bn_kernel.BnError, match="total changed across pages"):
        _run(session.all("strings", page=1, limit=2))

    assert session.last is None


def test_all_accepts_a_capped_callsites_scan_completing_on_the_final_page(
    monkeypatch, tmp_path
):
    """A high-fan-in `callsites` collection routinely reports `total: null` on a
    capped page and the exact count once a later page's caller scan completes.
    That `None -> int` refinement is the NORMAL end of a large collection and
    must not be rejected as a transition (#694 item 3); the published aggregate
    total must be the determined int, not the earlier null."""
    script = _page_script(
        'page = ({"items": [{"value": "a"}], "offset": 0, "returned": 1,\n'
        '         "has_more": True, "total": None, "total_lower_bound": 2,\n'
        '         "scan_truncated": True}\n'
        '        if offset == 0 else\n'
        '        {"items": [{"value": "b"}], "offset": 1, "returned": 1,\n'
        '         "has_more": False, "total": 2, "scan_truncated": False})'
    )
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))
    session = bn_kernel.Session(backend="cli")

    rows = _run(session.all("evidence", "calls", page=1))

    assert [row["value"] for row in rows] == ["a", "b"]
    assert session.last is not None
    assert session.last.payload["total"] == 2


def test_help_terminates_and_reaps_child_on_cancellation(monkeypatch, tmp_path):
    """`help()` used to handle `asyncio.TimeoutError` but not `CancelledError`,
    unlike `_run_resolved`, so cancelling an awaited `help()` leaked the running
    `bn --help` child instead of terminating and reaping it (#694 item 11)."""
    marker = tmp_path / "started"
    script = f"""
import sys, time
open({str(marker)!r}, "w").write("up")
time.sleep(30)
"""
    monkeypatch.setenv("BN_BIN", str(_fake_bn(tmp_path, script)))

    terminate_calls: list[Any] = []
    real_terminate = bn_kernel._terminate

    def spy_terminate(process: Any) -> None:
        terminate_calls.append(process)
        real_terminate(process)

    monkeypatch.setattr(bn_kernel, "_terminate", spy_terminate)

    async def scenario() -> bn_kernel.Session:
        session = bn_kernel.Session(backend="cli")
        task = asyncio.ensure_future(session.help("target"))
        while not marker.exists():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return session

    session = _run(scenario())

    assert len(terminate_calls) == 1
    assert session.last is None


@pytest.mark.parametrize(
    "dropped",
    [
        "filename",
        "basename",
        "function_count",
        "named_function_count",
        "unnamed_function_count",
        "imported_function_count",
        "import_symbol_count",
    ],
)
def test_info_requires_every_canonical_target_key(dropped):
    payload = _canonical_target_info()
    payload.pop(dropped)
    session = _shape_session(payload)

    with pytest.raises(bn_kernel.BnError, match="target info contract violation"):
        _run(session.info())

    assert session.last is None


INVALID_TARGET_INFO = [
    ("foreign payload", None),
    ("filename wrong type", {"filename": 17}),
    ("basename wrong type", {"basename": ["sample.bndb"]}),
    ("count is a bool", {"function_count": True}),
    ("count is a string", {"named_function_count": "2"}),
    ("count is negative", {"imported_function_count": -1}),
    ("import_symbol_count is a string", {"import_symbol_count": "4"}),
]


@pytest.mark.parametrize(
    "label,overrides",
    INVALID_TARGET_INFO,
    ids=[case[0] for case in INVALID_TARGET_INFO],
)
def test_info_rejects_invalid_canonical_target_values(label, overrides):
    payload = (
        {"kind": "il", "view": "hlil", "function": {"name": "main"}}
        if overrides is None
        else _canonical_target_info(**overrides)
    )
    session = _shape_session(payload)

    with pytest.raises(bn_kernel.BnError, match="target info contract violation"):
        _run(session.info())

    assert session.last is None


def test_info_accepts_the_canonical_payload_and_a_null_import_count():
    session = _shape_session(_canonical_target_info())
    assert _run(session.info())["function_count"] == 3

    # The bridge deliberately publishes None when the imports count raises
    # (bridge.py `_target_info`), so None must stay valid for that key alone.
    null_count = _shape_session(_canonical_target_info(import_symbol_count=None))
    assert _run(null_count.info())["import_symbol_count"] is None

    null_names = _shape_session(
        _canonical_target_info(filename=None, basename=None)
    )
    assert _run(null_names.info())["basename"] is None
