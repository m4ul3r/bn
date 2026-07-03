"""Tests for the bn skill helper scripts (skills/bn-*/scripts/).

These guard the sink-enumeration regex in sink-sweep.sh -- the helper that runs
"the whole reason the skill exists" -- against the false-all-clear gaps in #372:
the FORTIFY (`*_chk`) sink family and bare `execv` must not be silently dropped.
The regex is exercised exactly as the script applies it (`grep -E "$SINK_RE"`),
not re-implemented in Python, so the test fails if the shipped regex regresses.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "skills" / "bn-vr" / "scripts" / "sink-sweep.sh"

# A fake `bn` that faithfully models argparse "last --out wins": when the caller
# forwards a --out, the script's internal --out is overridden and its temp file
# is left empty -- the exact mechanism of the #438 false all-clear.
_FAKE_BN = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import sys, json
    args = sys.argv[1:]
    sub = args[0] if args else ""
    rest = args[1:]

    def last_out(a):
        val, j = None, 0
        while j < len(a):
            t = a[j]
            if t in ("--out", "-o") and j + 1 < len(a):
                val = a[j + 1]; j += 2; continue
            if t.startswith("--out=") or t.startswith("-o="):
                val = t.split("=", 1)[1]
            j += 1
        return val

    def emit(body, out):
        if out:
            open(out, "w").write(body)
        else:
            sys.stdout.write(body)

    if sub == "imports":
        payload = json.dumps({"kind": "imports", "items": [
            {"name": "memcpy"}, {"name": "__snprintf_chk"},
            {"name": "strcpy"}, {"name": "malloc"},
        ]}) + "\\n"
        emit(payload, last_out(rest))
    elif sub == "xrefs":
        sink = rest[0] if rest else "?"
        emit(f"xrefs to {sink} (2 code, 0 data)\\n  0x1000 caller_a\\n", last_out(rest))
    else:
        sys.exit(0)
    """
)


def _run_sweep(tmp_path, extra_args):
    """Run sink-sweep.sh with a fake `bn` (+ real jq) on PATH."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "bn"
    fake.write_text(_FAKE_BN, encoding="utf-8")
    fake.chmod(0o755)
    # Only `bn` is faked; real coreutils/jq come from the inherited PATH.
    env = {**os.environ, "PATH": str(bindir) + os.pathsep + os.environ.get("PATH", "")}
    return subprocess.run(
        ["bash", str(_SCRIPT), "-i", "x", "-t", "1", *extra_args],
        capture_output=True, text=True, env=env,
    )


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq required for the e2e sweep test")
def test_sweep_reports_sinks_without_out(tmp_path):
    """Baseline: with no caller --out, the sweep finds the sinks."""
    proc = _run_sweep(tmp_path, [])
    assert proc.returncode == 0, proc.stderr
    assert "3 dangerous-sink import(s)" in proc.stdout, proc.stdout
    assert "no dangerous-sink imports" not in proc.stdout


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq required for the e2e sweep test")
def test_sweep_with_caller_out_still_finds_sinks(tmp_path):
    """#438: a caller-supplied --out must NOT produce a false all-clear."""
    caller_out = tmp_path / "caller.txt"
    proc = _run_sweep(tmp_path, ["--out", str(caller_out)])
    assert "no dangerous-sink imports" not in proc.stdout, (
        "caller --out produced a FALSE all-clear (#438)\n" + proc.stdout + proc.stderr
    )
    assert "3 dangerous-sink import(s)" in proc.stdout, proc.stdout + proc.stderr
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq required for the e2e sweep test")
def test_sweep_with_caller_out_equals_form(tmp_path):
    """The --out=FILE form must be stripped too."""
    caller_out = tmp_path / "caller2.txt"
    proc = _run_sweep(tmp_path, [f"--out={caller_out}"])
    assert "no dangerous-sink imports" not in proc.stdout, proc.stdout + proc.stderr
    assert "3 dangerous-sink import(s)" in proc.stdout, proc.stdout + proc.stderr


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq required for the e2e sweep test")
def test_sweep_fails_loud_when_bn_produces_no_json(tmp_path):
    """#438 acceptance: an empty/failed `bn imports` errors, not a false all-clear."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "bn"
    # A bn that emits nothing (simulates a genuine failure / empty capture).
    fake.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    env = {**os.environ, "PATH": str(bindir) + os.pathsep + os.environ.get("PATH", "")}
    proc = subprocess.run(
        ["bash", str(_SCRIPT), "-i", "x", "-t", "1"],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "no dangerous-sink imports" not in proc.stdout
    assert "no valid JSON" in proc.stderr


def _sink_re() -> str:
    text = _SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"^SINK_RE='([^']*)'", text, re.MULTILINE)
    assert match, "SINK_RE assignment not found in sink-sweep.sh"
    return match.group(1)


def _matches(name: str) -> bool:
    """True iff `name` survives the script's `grep -E "$SINK_RE"` filter."""
    proc = subprocess.run(
        ["grep", "-E", _sink_re()],
        input=name + "\n",
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and proc.stdout.strip() == name


@pytest.mark.parametrize(
    "name",
    [
        # FORTIFY _chk family -- the whole point of #372.
        "__memcpy_chk",
        "__memmove_chk",
        "__strcpy_chk",
        "__strcat_chk",
        "__sprintf_chk",
        "__snprintf_chk",
        "__vsnprintf_chk",
        "__vfprintf_chk",
        "__printf_chk",
        "__asprintf_chk",
        "__vasprintf_chk",
        # bare execv was missing from the exec alternation.
        "execv",
        # the plain sinks must still match (no regression).
        "memcpy",
        "strcpy",
        "sprintf",
        "snprintf",
        "system",
        "popen",
        "execve",
        "execlp",
        # decorated forms (PLT) still match via the trailing (@.*)? group.
        "memcpy@plt",
        "__snprintf_chk@plt",
    ],
)
def test_sink_re_matches_dangerous_sinks(name):
    assert _matches(name), f"sink-sweep SINK_RE should flag {name!r} as a dangerous sink"


@pytest.mark.parametrize(
    "name",
    [
        # The malloc family is deliberately excluded (every binary calls it).
        "malloc",
        "calloc",
        "free",
        # Unrelated libc.
        "strlen",
        "getpid",
        "__libc_start_main",
        # A name that merely contains a sink substring must not match (anchored).
        "my_memcpy_wrapper",
        "system_init",
    ],
)
def test_sink_re_excludes_benign_names(name):
    assert not _matches(name), f"sink-sweep SINK_RE should NOT flag {name!r}"
