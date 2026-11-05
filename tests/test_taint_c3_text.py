"""Taint text residuals (C3): the frontier elision, the resolve-map note and
the --estimate-output help.

* A busy forward run folded its one-leaf `global_memory_continuation` blocks
  into "... and N more group(s)", so text showed none of them, even under
  --verbose, while JSON had every one (V4-01).
* An unmatched --resolve-map pin was echoed in JSON `run_params` but text said
  nothing (W07-14).
* The taint `--estimate-output` help promised "the flag that slices it" while
  the runtime advice says no flag slices taint output (V4-05).
"""
from __future__ import annotations

import re

import bn.cli
import pytest

from bn.formatters import _render_grouped_leaves, _render_taint_text

KIND = "global_memory_continuation"
GLOBALS = ("0x60a100", "0x60a200", "0x60a300")
ELISION = re.compile(r"\.\.\. and (\d+) more group\(s\) \((\d+) leaf\(s\)\)")


def _unmodeled(callee: str, at: str) -> dict:
    return {"kind": "unmodeled_callee", "address": at,
            "callee": {"name": callee, "display_name": callee, "address": "0x402000"}}


def _continuation(glob: str) -> dict:
    return {"kind": KIND, "address": "0x401100",
            "callee": {"name": "emit", "display_name": "emit", "address": "0x402100"},
            "global": {"address": glob, "label": f"glob_{glob}", "written_offsets": []},
            "references": [], "reachability": {"status": "complete"},
            "reads_proven": False, "reseed": None, "detail": "d"}


def _busy_leaves() -> list[dict]:
    """Fourteen two-leaf unmodeled groups outrank three one-leaf continuation
    groups, so the continuations sort past the default top-12 cap."""
    leaves = [_unmodeled(f"helper_{i}", f"0x4010{i:02x}")
              for i in range(14) for _ in range(2)]
    return leaves + [_continuation(g) for g in GLOBALS]


def _continuation_headers(lines: list[str]) -> list[str]:
    return [row for row in lines if KIND in row and not row.startswith("   ")]


def test_continuation_groups_are_never_folded_into_the_elision():
    leaves = _busy_leaves()
    lines = _render_grouped_leaves(leaves)
    headers = _continuation_headers(lines)
    assert len(headers) == len(GLOBALS)
    for glob in GLOBALS:
        assert sum(glob in row for row in headers) == 1
    # Twelve unmodeled groups render; the two left are all the elision counts.
    unmodeled_rows = [row for row in lines if "helper_" in row]
    assert len(unmodeled_rows) == 12
    (summary,) = [m for m in map(ELISION.search, lines) if m]
    assert (int(summary.group(1)), int(summary.group(2))) == (2, 4)


def test_verbose_renders_every_group_with_no_elision():
    leaves = _busy_leaves()
    lines = _render_grouped_leaves(leaves, full=True)
    assert not any(ELISION.search(row) for row in lines)
    assert len(_continuation_headers(lines)) == len(GLOBALS)
    for i in range(14):
        assert sum(bool(re.search(rf"\bhelper_{i}\b", row)) for row in lines) == 1


def _taint_result(resolve_map) -> dict:
    return {"direction": "forward", "function": {"name": "f", "address": "0x401000"},
            "sources": [], "flows": [], "sinks": [], "leaves": [],
            "run_params": {"resolve_map": resolve_map}}


def _resolve_rows(text: str) -> list[str]:
    return [row for row in text.splitlines() if row.startswith("resolve-map")]


def test_unmatched_resolve_map_pins_get_one_text_line():
    text = _render_taint_text(_taint_result(
        {"path": "/tmp/pins.json", "entries": 2, "matched": ["0x401200"],
         "unmatched": ["0x4011f0", "0x401300"]}))
    (row,) = _resolve_rows(text)
    assert "0x4011f0" in row and "0x401300" in row and "/tmp/pins.json" in row
    assert "0x401200" not in row


@pytest.mark.parametrize("resolve_map", [
    None,
    {"path": "/tmp/pins.json", "entries": 1, "matched": ["0x401200"], "unmatched": []},
])
def test_no_resolve_map_line_when_all_matched_or_no_map(resolve_map):
    assert _resolve_rows(_render_taint_text(_taint_result(resolve_map))) == []


def _estimate_help(argv: list[str], capsys) -> str:
    with pytest.raises(SystemExit):
        bn.cli.main([*argv, "--help"])
    text = " ".join(capsys.readouterr().out.split())
    start = text.index("--estimate-output")
    return text[start:text.index("Mutually exclusive with --out", start)]


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_taint_estimate_help_matches_the_runtime_advice(direction, capsys):
    help_text = _estimate_help(["taint", direction], capsys)
    # The runtime advice for taint: --max-depth bounds the analysis, --out
    # takes the output; no slicing flag is promised.
    assert "--max-depth" in help_text and "--out" in help_text
    assert "flag that slices it" not in help_text


def test_slicing_commands_keep_their_estimate_help(capsys):
    assert "flag that slices it" in _estimate_help(["disasm"], capsys)
