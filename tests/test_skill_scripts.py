"""Tests for the bn skill helper scripts (skills/bn-*/scripts/).

These guard the sink-enumeration regex in sink-sweep.sh -- the helper that runs
"the whole reason the skill exists" -- against the false-all-clear gaps in #372:
the FORTIFY (`*_chk`) sink family and bare `execv` must not be silently dropped.
The regex is exercised exactly as the script applies it (`grep -E "$SINK_RE"`),
not re-implemented in Python, so the test fails if the shipped regex regresses.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "skills" / "bn-vr" / "scripts" / "sink-sweep.sh"


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
