"""Forward frontier text: one block per fact, wrapped, with one routed command
(P2-04).

A global write that stays live across many later calls produced one
`global_memory_continuation` leaf per (call site x global); text printed each
as a single 1-2 KB line repeating the reader list and four commands. The JSON
keeps every leaf; text groups them by the global, lists the call sites and
readers with counts, and gives one follow-up command.
"""
from __future__ import annotations

import re
import shlex

import bn.cli
import pytest

from bn.formatters import _render_grouped_leaves, _render_taint_text, command_routing

KIND = "global_memory_continuation"
LONG_DETAIL = ("tainted bytes were copied into the writable global and code reachable "
               "from this call may read it; " * 30)


def _continuation(call: str, callee: str, glob: str, *, readers=(), truncated=True,
                  reseed=None) -> dict:
    return {
        "kind": KIND, "address": call,
        "callee": {"name": callee, "display_name": callee, "address": "0x402000"},
        "global": {"address": glob, "label": f"glob_{glob}", "written_offsets": []},
        "references": [{"name": name, "display_name": name, "address": "0x403000",
                        "at": at, "access": "load", "offset": 0, "offset_known": True}
                       for name, at in readers],
        "reachability": ({"status": "truncated", "cause": "object_extent_unproven",
                          "causes": ["object_extent_unproven"], "functions_examined": 9,
                          "function_budget": 512, "unexamined_references_total": 3}
                         if truncated else {"status": "complete"}),
        "reads_proven": bool(readers), "reseed": reseed, "detail": LONG_DETAIL,
    }


def _session_log_leaves() -> list[dict]:
    """One write into `session_log` stays live across five later calls, and a
    second write into `peer_name` across one."""
    calls = [("0x401100", "flush_log"), ("0x401130", "emit_record"),
             ("0x401160", "emit_record"), ("0x401190", "close_peer"),
             ("0x4011c0", "rotate_log")]
    readers = [("log_reader", "0x404010"), ("log_dump", "0x404210")]
    return ([_continuation(at, fn, "0x60a100", readers=readers) for at, fn in calls]
            + [_continuation("0x4011f0", "greet_peer", "0x60a400", truncated=False)])


def _blocks(lines: list[str]) -> list[list[str]]:
    """Split rendered frontier rows into blocks, one per header row."""
    out: list[list[str]] = []
    for line in lines[1:]:
        if line.startswith("  ") and not line.startswith("   "):
            out.append([line])
        else:
            out[-1].append(line)
    return out


def test_continuation_leaves_group_by_global_with_truthful_counts():
    leaves = _session_log_leaves()
    lines = _render_grouped_leaves(leaves, width=100)
    blocks = [b for b in _blocks(lines) if KIND in b[0]]
    # One block per global, never one row per call site.
    assert len(blocks) == 2
    first = " ".join(blocks[0])
    assert "0x60a100" in blocks[0][0] and "[5 leaves]" in blocks[0][0]
    for call in ("0x401100", "0x401130", "0x401160", "0x401190"):
        assert call in first
    # Four examples, then the exact count of the rest.
    assert "+1 more" in first
    assert "0x60a400" in blocks[1][0] and "leaves]" not in blocks[1][0]
    # The counts the block states add up to the leaves it stands for.
    stated = sum(int(m) for b in blocks for m in re.findall(r"\[(\d+) leaves\]", b[0])) \
        + sum(1 for b in blocks if "leaves]" not in b[0])
    assert stated == len(leaves)
    assert lines[0].startswith(f"frontiers ({len(leaves)}")


def test_continuation_block_is_structured_and_names_one_command():
    lines = _render_grouped_leaves(_session_log_leaves(), width=100)
    block = next(b for b in _blocks(lines) if "0x60a100" in b[0])
    # Header, then indented body rows; the readers each appear once even though
    # five leaves carry them.
    assert all(row.startswith("      ") for row in block[1:])
    body = "\n".join(block)
    assert body.count("log_reader @ 0x404010") == 1
    assert "object_extent_unproven" in body
    assert len(re.findall(r"`bn [^`]+`", body)) == 1
    # The prose detail stays in JSON unless --verbose asks for it.
    assert "copied into the writable global and code reachable" not in body
    verbose = "\n".join(_render_grouped_leaves(_session_log_leaves(), width=100, full=True))
    assert "copied into the writable global" in verbose


def test_a_leaf_reseed_is_the_one_command_when_present():
    reseed = "`bn taint forward -f 0x404000 --source var:entry`"
    leaves = [_continuation("0x401100", "flush_log", "0x60a100", reseed=reseed),
              _continuation("0x401130", "emit_record", "0x60a100")]
    body = "\n".join(_render_grouped_leaves(leaves, width=120))
    assert re.findall(r"`bn [^`]+`", body) == [reseed]


@pytest.mark.parametrize("width", [60, 80, 120])
def test_every_frontier_row_wraps_at_the_width(width):
    other = {"kind": "pointer_escape", "address": "0x401300", "buffer": "heap_0x401290",
             "detail": LONG_DETAIL}
    lines = _render_grouped_leaves(_session_log_leaves() + [other], width=width, full=True)
    # Only a row's fact head (before its `  -- ` prose) may run past the
    # width, and only when it alone is wider; prose always wraps.
    long = [line for line in lines
            if len(line) > width and "`" not in line
            and len(line.split("  -- ")[0]) <= width]
    assert not long, long
    # Wrapping never splits a command: each one is whole on its own row.
    for line in lines:
        assert line.count("`") % 2 == 0, line


def test_a_wrapped_row_keeps_every_fact_on_its_first_line():
    """The parenthetical facts of a frontier row -- tainted args, skip reason,
    ABI-recovered args, modeled -- are what a reader (or a grep over first
    lines) acts on; only the trailing prose wraps."""
    leaf = {"kind": "unread_callee_body", "address": "0x401230",
            "callee": {"name": "decode_frame", "display_name": "decode_frame",
                       "address": "0x402400"},
            "tainted_args": [0], "skip_reason": "body too large", "modeled": True,
            "abi_recovered_args": [2], "detail": LONG_DETAIL}
    lines = _render_grouped_leaves([leaf], width=80)
    first = next(line for line in lines if "unread_callee_body @" in line)
    for fact in ("tainted arg(s) [0]", "body too large", "ABI-recovered arg(s) [2]",
                 "modeled"):
        assert fact in first, first
    assert len(lines) > 2 and all(len(line) <= 80 for line in lines[2:])


def test_rendered_follow_up_carries_the_invocation_routing():
    with command_routing(target="probe.bndb", instance="audit-7"):
        body = "\n".join(_render_grouped_leaves(_session_log_leaves(), width=200))
    commands = re.findall(r"`(bn [^`]+)`", body)
    assert commands == ["bn -i audit-7 -t probe.bndb xrefs 0x60a100",
                        "bn -i audit-7 -t probe.bndb xrefs 0x60a400"]


def test_cli_text_frontier_command_keeps_instance_and_target(monkeypatch, capsys):
    result = {"direction": "forward", "function": {"name": "serve", "address": "0x401000"},
              "sources": ["param:0"], "reached_sinks": [], "leaves": _session_log_leaves(),
              "assumptions": [], "stats": {"functions_visited": 3}, "soundness": "x"}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None,
                          spawn_missing_named=False):
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "probe.bndb"}]}
        assert op == "taint"
        return {"ok": True, "result": result}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setenv("COLUMNS", "100")
    rc = bn.cli.main(["-i", "audit-7", "taint", "forward", "-t", "probe.bndb",
                      "-f", "serve", "--source", "param:0"])
    assert rc == 0
    out = capsys.readouterr().out
    commands = re.findall(r"`(bn [^`]+)`", out)
    assert commands
    for command in commands:
        assert "-i audit-7" in command and "-t probe.bndb" in command, command


@pytest.mark.parametrize("target", ["audit`copy name.bndb", "it's ``two`` ticks.bndb",
                                    "plain name.bndb", "`edge`"])
@pytest.mark.parametrize("width", [40, 60, 100])
def test_a_routed_command_survives_wrapping_whatever_its_selector_holds(target, width):
    """R4-02: the wrap protection took the first backtick after an opening one
    as the end of a command, so a selector carrying a backtick split the
    command across two rows. Both the CLI-routed reseed and the renderer's own
    command are printed as one fenced span; each sits whole on one row and
    shell-parses back to the routing it was given."""
    instance = "audit 7"
    leaves = [_continuation("0x401100", "flush_log", "0x60a100",
                            reseed="`bn taint forward -f log_reader --source param:0`"),
              _continuation("0x4011f0", "greet_peer", "0x60a400")]
    bn.cli._contextualize_diagnostic_hints({"leaves": leaves}, target=target,
                                           instance=instance)
    with command_routing(target=target, instance=instance):
        lines = _render_grouped_leaves(leaves, width=width)
    commands = []
    for line in lines:
        if "next: " not in line:
            continue
        span = line.split("next: ", 1)[1]
        fence = len(span) - len(span.lstrip("`"))
        # The whole command on this row: the closing fence is the row's end.
        assert fence and span.endswith("`" * fence), line
        body = span[fence:-fence]
        # A code span a reader can paste: no run inside it is as long as the
        # fence, which would end the span there.
        assert all(len(run) < fence for run in re.findall(r"`+", body)), line
        commands.append(shlex.split(body))
    assert [words[5:] for words in commands] == [
        ["taint", "forward", "-f", "log_reader", "--source", "param:0"],
        ["xrefs", "0x60a400"]]
    for words in commands:
        assert words[:5] == ["bn", "-i", instance, "-t", target]
