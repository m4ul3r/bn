from __future__ import annotations

import json
import types

import bn.cli
import pytest

from _cli_helpers import *  # noqa: F401,F403



def test_function_list_uses_implicit_target_when_single_target_is_open(fake_transport, capsys):
    calls = fake_transport({
        "list_targets": {
            "ok": True,
            "result": [{"target_id": "123:1:7", "selector": "SnailMail_unwrapped.exe.bndb"}],
        },
        "list_functions": {"ok": True, "result": {"functions": [{"name": "sub_401000", "address": "0x401000"}],
                                                  "total": 1, "offset": 0, "limit": 100, "returned": 1, "has_more": False}},
    })

    rc = bn.cli.main(["function", "list"])
    assert rc == 0
    assert [call["op"] for call in calls] == ["list_targets", "list_functions"]
    assert calls[1]["params"] == {"limit": 100}
    assert calls[1]["target"] == "active"
    output = capsys.readouterr().out
    assert output == "0x401000  sub_401000\n"
    assert '"name"' not in output


def test_function_list_requires_target_when_multiple_targets_are_open(fake_transport, capsys):
    fake_transport({
        "list_targets": {
            "ok": True,
            "result": [
                {
                    "target_id": "123:1:7",
                    "selector": "SnailMail_unwrapped.exe.bndb",
                    "active": True,
                },
                {"target_id": "123:2:8", "selector": "other.exe.bndb", "active": False},
            ],
        },
    })

    rc = bn.cli.main(["function", "list"])

    assert rc == 2
    assert capsys.readouterr().err == (
        "This command requires --target when multiple targets are open.\n"
        "Open targets:\n"
        "- SnailMail_unwrapped.exe.bndb [active] (target_id: 123:1:7)\n"
        "- other.exe.bndb (target_id: 123:2:8)\n"
    )


def test_function_list_returns_full_result_set(fake_transport, capsys):
    calls = fake_transport({"list_functions": {
        "ok": True,
        "result": {"functions": [{"name": f"sub_{index:06x}", "address": hex(index)} for index in range(150)],
                   "total": 150, "offset": 0, "limit": 200, "returned": 150, "has_more": False},
    }})

    rc = bn.cli.main(["function", "list", "--target", "active", "--format", "json", "--limit", "200"])

    assert rc == 0
    assert calls[-1]["op"] == "list_functions"
    assert calls[-1]["params"] == {"limit": 200}
    stdout, stderr = capsys.readouterr()
    payload = json.loads(stdout)
    assert len(payload["functions"]) == 150
    assert payload["total"] == 150 and payload["has_more"] is False
    assert stderr == ""


def test_function_list_warns_when_output_auto_spills(fake_transport, monkeypatch, capsys):
    captured = {}

    fake_transport({"list_functions": {
        "ok": True,
        "result": {"functions": [
            {"name": "sub_401000", "address": "0x401000"},
            {"name": "sub_402000", "address": "0x402000"},
        ], "total": 2, "offset": 0, "limit": 100, "returned": 2, "has_more": False},
    }})

    def fake_write_output_result(value, *, fmt, out_path, stem):
        captured["value"] = value
        captured["fmt"] = fmt
        captured["out_path"] = out_path
        captured["stem"] = stem
        return types.SimpleNamespace(
            rendered=(
                "ok: true\n"
                "spilled: true\n"
                "path: /tmp/functions.txt\n"
                "format: text\n"
                "bytes: 1234\n"
                "tokens: 23456\n"
                "tokenizer: estimate\n"
                "sha256: deadbeef\n"
                "summary: kind=string chars=42\n"
            ),
            spilled=True,
            artifact={
                "artifact_path": "/tmp/functions.txt",
                "bytes": 1234,
                "format": "text",
                "sha256": "deadbeef",
                "spilled": True,
                "summary": {"kind": "string", "chars": 42},
                "tokenizer": "estimate",
                "tokens": 23456,
            },
        )

    monkeypatch.setattr(bn.cli, "write_output_result", fake_write_output_result)

    rc = bn.cli.main(["function", "list", "--target", "active"])

    assert rc == 0
    stdout, stderr = capsys.readouterr()
    assert stdout.startswith("ok: true\nspilled: true\npath: /tmp/functions.txt\n")
    assert captured["value"] == "0x401000  sub_401000\n0x402000  sub_402000"
    assert stderr == (
        "warning: function list output spilled to /tmp/functions.txt; "
        "rerun with --limit/--offset to page through the results\n"
    )


def test_decompile_spill_warning_suggests_line_slicing(fake_transport, monkeypatch, capsys):
    fake_transport({"decompile": {"ok": True, "result": {"text": "long decompiled text"}}})

    def fake_write_output_result(value, *, fmt, out_path, stem):
        assert stem == "decompile"
        return _spill_artifact_namespace("/tmp/decompile.txt")

    monkeypatch.setattr(bn.cli, "write_output_result", fake_write_output_result)

    rc = bn.cli.main(["decompile", "sub_401000", "--target", "active"])

    assert rc == 0
    _, stderr = capsys.readouterr()
    assert stderr == (
        "warning: decompile output spilled to /tmp/decompile.txt; "
        "rerun with --lines START:END to fetch a slice instead\n"
    )


def test_decompile_json_spill_warning_does_not_suggest_lines(fake_transport, monkeypatch, capsys):
    # --lines is a text-only flag; when decompile --format json spills, the hint
    # must NOT tell JSON consumers to rerun with --lines (a dead end) -- it
    # should point at --out / the artifact instead (#120).
    fake_transport({"decompile": {"ok": True, "result": {"text": "long decompiled text"}}})

    def fake_write_output_result(value, *, fmt, out_path, stem):
        assert stem == "decompile"
        return _spill_artifact_namespace("/tmp/decompile.json")

    monkeypatch.setattr(bn.cli, "write_output_result", fake_write_output_result)

    rc = bn.cli.main(["decompile", "sub_401000", "--target", "active", "--format", "json"])

    assert rc == 0
    _, stderr = capsys.readouterr()
    assert "--lines" not in stderr
    assert "--out" in stderr


def test_function_list_forwards_address_filters(fake_transport, capsys):
    calls = fake_transport({"list_functions": {"ok": True, "result": []}})

    rc = bn.cli.main(
        [
            "function",
            "list",
            "--target",
            "active",
            "--min-address",
            "0x401000",
            "--max-address",
            "0x402000",
        ]
    )

    assert rc == 0
    assert calls[-1]["op"] == "list_functions"
    assert calls[-1]["params"]["min_address"] == "0x401000"
    assert calls[-1]["params"]["max_address"] == "0x402000"
    assert capsys.readouterr().out == "none\n"


def test_function_search_can_request_regex_matching(fake_transport, capsys):
    calls = fake_transport({"search_functions": {"ok": True, "result": {"functions": [{"name": "load_attachment", "address": "0x401000"}],
                                                                        "total": 1, "offset": 0, "limit": 100, "returned": 1, "has_more": False}}})

    rc = bn.cli.main(["function", "search", "--target", "active", "--regex", "attach|detach"])

    assert rc == 0
    assert calls[-1]["op"] == "search_functions"
    assert calls[-1]["params"]["query"] == "attach|detach"
    assert calls[-1]["params"]["regex"] is True
    assert "offset" not in calls[-1]["params"]
    assert calls[-1]["params"]["limit"] == 100
    assert capsys.readouterr().out == "0x401000  load_attachment\n"


def test_function_commands_accept_paging_flags():
    parser = bn.cli.build_parser()

    args = parser.parse_args(["function", "list", "--limit", "10"])
    assert args.limit == 10
    assert args.offset == 0

    args = parser.parse_args(["function", "search", "--offset", "10", "--limit", "50", "attach"])
    assert args.offset == 10
    assert args.limit == 50
    assert args.query == "attach"


def test_callsites_both_scope_flags_still_rejected():
    parser = bn.cli.build_parser()

    # Passing both scope flags is still a mutex violation handled by argparse.
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "callsites",
                "crt_rand",
                "--within",
                "bonus_pick_random_type",
                "--within-file",
                "functions.txt",
            ]
        )


def test_callsites_missing_scope_raises_actionable_error(fake_transport, capsys):
    fake_transport()

    rc = bn.cli.main(["callsites", "crt_rand", "--target", "active"])

    # BridgeError surfaces as a nonzero exit with a human-facing message.
    assert rc != 0
    combined = capsys.readouterr()
    text = combined.err + combined.out
    assert "--within" in text
    assert "--within-file" in text
    assert "bn xrefs crt_rand" in text


def test_function_info_uses_active_target_and_text_renderer(fake_transport, capsys):
    calls = fake_transport({"function_info": {
        "ok": True,
        "result": {
            "function": {"name": "sub_401000", "address": "0x401000"},
            "prototype": "int32_t sub_401000(int32_t arg1)",
            "return_type": "int32_t",
            "calling_convention": "__cdecl",
            "size": 24,
            "parameters": [{"name": "arg1", "type": "int32_t", "storage": 0, "is_parameter": True, "local_id": "0x401000:param:stack:0:0:1"}],
            "locals": [{"name": "var_4", "type": "int32_t", "storage": -4, "is_parameter": False, "local_id": "0x401000:local:stack:-4:1:2"}],
        },
    }})

    rc = bn.cli.main(["function", "info", "--format", "text", "--target", "active", "sub_401000"])

    assert rc == 0
    assert calls[-1]["op"] == "function_info"
    assert calls[-1]["target"] == "active"
    output = capsys.readouterr().out
    assert "sub_401000 @ 0x401000" in output
    assert "calling convention: __cdecl" in output
    assert "size: 24" in output
    assert "xrefs: 0" in output
    assert "locals: 1 variables" in output
    # compact mode should NOT show full parameter/local details
    assert "id=0x401000:param:stack:0:0:1" not in output


def test_xrefs_field_routes_to_field_xrefs(fake_transport, capsys):
    calls = fake_transport({"field_xrefs": {
        "ok": True,
        "result": {
            "field": {
                "type_name": "TrackRowCell",
                "field_name": "tile_type",
                "offset": 8,
                "field_type": "uint32_t",
            },
            "code_refs": [{"address": "0x401000", "function": "sub_401000", "incoming_type": "TrackRowCell*", "disasm": "mov eax, [ecx+8]"}],
            "data_refs": [],
        },
    }})

    rc = bn.cli.main(["xrefs", "--field", "TrackRowCell.tile_type", "--format", "text", "--target", "active"])

    assert rc == 0
    assert calls[-1]["op"] == "field_xrefs"
    assert calls[-1]["params"]["field"] == "TrackRowCell.tile_type"
    assert calls[-1]["target"] == "active"
    output = capsys.readouterr().out
    assert "TrackRowCell.tile_type" in output
    assert "code refs:" in output


def test_xrefs_text_format_renders_summary(fake_transport, capsys):
    fake_transport({"xrefs": {
        "ok": True,
        "result": {
            "address": "0x401000",
            "code_refs": [
                {
                    "address": "0x402000",
                    "function": "sub_402000",
                    "caller_function": {"address": "0x401f00", "name": "sub_402000"},
                }
            ],
            "data_refs": [
                {
                    "address": "0x403000",
                    "function": "sub_403000",
                    "caller_function": {"address": "0x402f00", "name": "sub_403000"},
                }
            ],
        },
    }})

    rc = bn.cli.main(["xrefs", "--format", "text", "--target", "active", "sub_401000"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "xrefs to 0x401000" in output
    assert "code refs: 1 site across 1 function" in output
    assert "0x401f00  sub_402000  (1 site: 0x402000)" in output
    assert "data refs: 1 site across 1 function" in output
    assert "0x402f00  sub_403000  (1 site: 0x403000)" in output


def test_evidence_xrefs_routes_and_renders_context(fake_transport, capsys):
    calls = fake_transport({"xrefs": {
        "ok": True,
        "result": {
            "address": "0x175b20",
            "target_context": {
                "sections": [{"name": ".rodata"}],
                "symbol": {"name": "common.HeadUnitInfo", "type": "DataSymbol"},
                "string": {
                    "value": "Usage: %s [OPTION]...\n",
                    "encoding": "ascii",
                    "truncated": True,
                },
            },
            "code_refs": [
                {
                    "address": "0x586c0",
                    "function": "sub_586a2",
                    "kind": "code",
                    "context": {
                        "sections": [{"name": ".text"}],
                        "segment": {"readable": True, "writable": False, "executable": True},
                        "disasm": "adr r1, common.HeadUnitInfo",
                    },
                }
            ],
            "data_refs": [],
        },
    }})

    rc = bn.cli.main(["evidence", "xrefs", "--format", "text", "--target", "active", "0x175b20"])

    assert rc == 0
    assert calls[-1]["op"] == "xrefs"
    assert calls[-1]["params"]["identifier"] == "0x175b20"
    output = capsys.readouterr().out
    assert "target | section=.rodata | symbol=common.HeadUnitInfo[DataSymbol]" in output
    assert 'string="Usage: %s [OPTION]...\\n" [truncated]' in output
    assert "0x586c0  code  sub_586a2" in output
    assert "seg=r-x" in output


def test_render_xrefs_text_from_items_when_arrays_dropped():
    # #184: the xrefs op no longer ships the full code_refs/data_refs arrays, so
    # the text renderer reconstructs the grouped view from `items` (split by kind)
    # and the full-set summary counts. Output must match the dual-array rendering.
    from bn import formatters
    value = {
        "address": "0x401000",
        "code_ref_count": 1,
        "data_ref_count": 1,
        "caller_function_count": 1,
        "items": [
            {"address": "0x402000", "function": "sub_402000", "kind": "code",
             "caller_function": {"address": "0x401f00", "name": "sub_402000"}},
            {"address": "0x403000", "function": "sub_403000", "kind": "data",
             "caller_function": {"address": "0x402f00", "name": "sub_403000"}},
        ],
        "total": 2, "offset": 0, "limit": None, "returned": 2, "has_more": False,
    }
    out = formatters._render_xrefs_text(value)
    assert "xrefs to 0x401000 (1 code, 1 data)" in out
    assert "code refs: 1 site across 1 function" in out
    assert "0x401f00  sub_402000  (1 site: 0x402000)" in out
    assert "data refs: 1 site across 1 function" in out
    assert "0x402f00  sub_403000  (1 site: 0x403000)" in out


def test_render_evidence_xrefs_text_from_items_when_arrays_dropped():
    # #184: same for the evidence-xrefs renderer -- render from `items` + summary
    # counts when the deprecated arrays are absent.
    from bn import formatters
    value = {
        "address": "0x175b20",
        "target_context": {},
        "code_ref_count": 1,
        "data_ref_count": 0,
        "items": [
            {"address": "0x586c0", "function": "sub_586a2", "kind": "code"},
        ],
        "total": 1, "offset": 0, "limit": None, "returned": 1, "has_more": False,
    }
    out = formatters._render_evidence_xrefs_text(value)
    assert "xrefs to 0x175b20" in out
    assert "code refs:" in out
    assert "0x586c0  code  sub_586a2" in out
    assert "data refs:" in out
    assert "- none" in out


def test_evidence_function_routes_and_renders_calls(fake_transport, capsys):
    calls = fake_transport({"function_evidence": {
        "ok": True,
        "result": {
            "function": {"name": "build_response", "address": "0x412470"},
                "prototype": "int32_t build_response(void*)",
                "calling_convention": "__cdecl",
                "thunk": {"is_candidate": False},
                "calls": [
                    {
                        "address": "0x4124a0",
                        "operation": "LLIL_CALL",
                        "direct": True,
                        "target": {
                            "raw": "0x461746",
                            "normalized": "0x461746",
                            "function": {"name": "send_message", "address": "0x461746"},
                        },
                        "llil": "call(0x461746)",
                        "hlil_statement": "send_message(6, &response)",
                        "argument_source": "hlil",
                        "arguments": [
                            {"index": 0, "text": "6"},
                            {
                                "index": 1,
                                "text": "0x2a4f4",
                                "resolved": {"string": "4", "section": ".rodata"},
                            },
                        ],
                        "argument_candidates": [
                            {"source": "mlil", "index": 0, "text": "r0"},
                        ],
                    }
                ],
            },
        }})

    rc = bn.cli.main(["evidence", "function", "--target", "active", "--context", "1", "build_response"])

    assert rc == 0
    assert calls[-1]["op"] == "function_evidence"
    assert calls[-1]["params"] == {"identifier": "build_response", "context": 1}
    output = capsys.readouterr().out
    assert "build_response @ 0x412470" in output
    assert "target: send_message @ 0x461746" in output
    assert "arguments: (hlil)" in output
    assert '0x2a4f4 -> "4" [.rodata]' in output
    # uncertain extras are JSON-only; not shown in text
    assert "r0" not in output


def test_evidence_table_routes_and_renders_targets(fake_transport, capsys):
    calls = fake_transport({"pointer_table": {
        "ok": True,
        "result": {
            "address": "0x64ea0",
                "pointer_size": 4,
                "stride": 4,
                "warnings": [
                    "table start is in an executable segment; this may be code, not a pointer table"
                ],
                "entries": [
                    {
                        "index": 0,
                        "entry_address": "0x64ea0",
                        "value": "0x46971",
                        "readable": True,
                        "plausible": True,
                        "target": {
                            "raw": "0x46971",
                            "normalized": "0x46970",
                            "thumb_adjusted": True,
                            "function": {"name": "sub_46970", "address": "0x46970"},
                        },
                    }
                ],
            },
        }})

    rc = bn.cli.main(["evidence", "table", "--target", "active", "--entries", "1", "0x64ea0"])

    assert rc == 0
    assert calls[-1]["op"] == "pointer_table"
    assert calls[-1]["params"]["address"] == "0x64ea0"
    assert calls[-1]["params"]["entries"] == 1
    output = capsys.readouterr().out
    assert "pointer table @ 0x64ea0" in output
    assert "warning: table start is in an executable segment" in output
    assert "sub_46970 @ 0x46970 (raw 0x46971) [thumb-adjusted]" in output


def test_evidence_table_renders_interior_function_targets(fake_transport, capsys):
    fake_transport({"pointer_table": {
        "ok": True,
        "result": {
            "address": "0x402000",
                "pointer_size": 8,
                "stride": 8,
                "warnings": ["1 entries resolve inside functions but not at function starts"],
                "entries": [
                    {
                        "index": 0,
                        "entry_address": "0x402000",
                        "value": "0x401001",
                        "readable": True,
                        "plausible": True,
                        "target": {
                            "raw": "0x401001",
                            "normalized": "0x401001",
                            "thumb_adjusted": False,
                            "function": {
                                "name": "target",
                                "address": "0x401000",
                                "exact_start": False,
                                "offset": "0x1",
                            },
                        },
                    }
                ],
            },
        }})

    rc = bn.cli.main(["evidence", "table", "--target", "active", "--entries", "1", "0x402000"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "warning: 1 entries resolve inside functions but not at function starts" in output
    assert "target @ 0x401000+0x1 (target 0x401001, not start)" in output
    assert "[thumb-adjusted]" not in output


def test_evidence_message_routes_and_renders_lens(fake_transport, capsys):
    calls = fake_transport({"message_lens": {
        "ok": True,
        "result": {
            "query": "HeadUnitInfo",
                "count": 1,
                "matches": [
                    {
                        "type_string": {
                            "address": "0x175b20",
                            "value": "common.HeadUnitInfo",
                            "context": {"sections": [{"name": ".rodata"}]},
                        },
                        "xrefs": {
                            "code_refs": [{"address": "0x586c0", "function": "sub_586a2"}],
                            "data_refs": [{"address": "0x175bd8"}],
                        },
                        "metadata_table_windows": [{"address": "0x175bd0"}],
                    }
                ],
            },
        }})

    rc = bn.cli.main(
        [
            "evidence",
            "message",
            "--target",
            "active",
            "--limit",
            "5",
            "--table-entries",
            "4",
            "HeadUnitInfo",
        ]
    )

    assert rc == 0
    assert calls[-1]["op"] == "message_lens"
    assert calls[-1]["params"] == {"query": "HeadUnitInfo", "limit": 5, "table_entries": 4}
    output = capsys.readouterr().out
    assert "message lens: HeadUnitInfo (1 matches)" in output
    assert "0x175b20  \"common.HeadUnitInfo\"" in output
    assert "metadata table windows: 1" in output


def test_callsites_routes_within_scope_and_renders_text(fake_transport, capsys):
    calls = fake_transport({"callsites": {
        "ok": True,
        "result": [
            {
                "callee": {"name": "crt_rand", "address": "0x461746"},
                    "containing_function": {
                        "name": "bonus_pick_random_type",
                        "address": "0x412470",
                    },
                    "within_query": "bonus_pick_random_type",
                    "call_index": 0,
                    "call_addr": "0x4124a0",
                    "instruction_length": 5,
                    "caller_static": "0x4124a5",
                    "call_instruction": {"address": "0x4124a0", "text": "call crt_rand"},
                    "previous_instructions": [
                        {"address": "0x41249c", "text": "mov eax, 0"},
                    ],
                    "next_instructions": [
                        {"address": "0x4124a5", "text": "cmp eax, 0xd"},
                    ],
                    "hlil_statement": "edx_1:eax_1 = sx.q(crt_rand())",
                    "pre_branch_condition": "result == 2",
                }
            ],
        }})

    rc = bn.cli.main(
        [
            "callsites",
            "--format",
            "text",
            "--target",
            "active",
            "--within",
            "bonus_pick_random_type",
            "--caller-static",
            "crt_rand",
        ]
    )

    assert rc == 0
    assert calls[-1]["op"] == "callsites"
    assert calls[-1]["target"] == "active"
    assert calls[-1]["params"]["callee"] == "crt_rand"
    assert calls[-1]["params"]["within_identifiers"] == ["bonus_pick_random_type"]
    assert calls[-1]["params"]["context"] == 3
    assert calls[-1]["params"]["caller_static"] is True
    output = capsys.readouterr().out
    assert output.startswith("caller_static 0x4124a5 | call 0x4124a0")
    assert "within: bonus_pick_random_type @ 0x412470" in output
    assert "call-index: 0" in output
    assert "within-query: bonus_pick_random_type" in output
    assert "hlil: edx_1:eax_1 = sx.q(crt_rand())" in output
    assert "pre-branch: result == 2" in output
    assert "> 0x4124a0  call crt_rand" in output


def test_callsites_text_omits_null_hlil_and_pre_branch(fake_transport, capsys):
    fake_transport({"callsites": {
        "ok": True,
        "result": [
            {
                "callee": {"name": "crt_rand", "address": "0x461746"},
                "containing_function": {"name": "fx_queue_add_random", "address": "0x427700"},
                "within_query": "fx_queue_add_random",
                "call_index": 3,
                "call_addr": "0x427806",
                "instruction_length": 5,
                "caller_static": "0x42780b",
                "call_instruction": {"address": "0x427806", "text": "call crt_rand"},
                "previous_instructions": [],
                "next_instructions": [],
                "hlil_statement": None,
                "pre_branch_condition": None,
            }
        ],
    }})

    rc = bn.cli.main(["callsites", "--format", "text", "--target", "active", "--within", "fx_queue_add_random", "crt_rand"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "call-index: 3" in output
    assert "hlil:" not in output
    assert "pre-branch:" not in output


def test_target_summary_text_shows_function_counts():
    # target info should surface the function-count + named-vs-auto summary that
    # every agent immediately reaches for (#122).
    from bn import formatters
    out = formatters._render_target_summary({
        "selector": "active",
        "arch": "x86_64",
        "function_count": 412,
        "named_function_count": 87,
        "unnamed_function_count": 313,
        "imported_function_count": 12,
    })
    assert "412 functions" in out
    assert "87 named" in out
    assert "313 auto-named" in out
    assert "12 imported" in out


def test_function_search_hints_regex_on_zero_matches_with_metachars(monkeypatch, capsys):
    """A non-regex search whose query has regex metacharacters and matches
    nothing should suggest --regex instead of a bare 'none' (#122)."""
    monkeypatch.setattr(bn.cli, "send_request", _zero_function_search)
    rc = bn.cli.main(["function", "search", "init|fini", "--target", "active"])
    assert rc == 0
    _, err = capsys.readouterr()
    assert "--regex" in err
    assert "init|fini" in err


def test_xrefs_no_offset_hint_for_real_address(monkeypatch, capsys):
    """A plausible code/data address (>= 0x10000) with 0 xrefs is a normal
    'nothing references this' result -- no struct-field hint."""
    monkeypatch.setattr(bn.cli, "send_request", _empty_xrefs)
    rc = bn.cli.main(["xrefs", "0x401000", "--target", "active"])
    assert rc == 0
    _, err = capsys.readouterr()
    assert "--field" not in err


def test_function_search_no_regex_hint_when_regex_flag_set(monkeypatch, capsys):
    """--regex already set: the query IS a pattern, so no hint even at 0 matches."""
    monkeypatch.setattr(bn.cli, "send_request", _zero_function_search)
    rc = bn.cli.main(["function", "search", "init|fini", "--regex", "--target", "active"])
    assert rc == 0
    _, err = capsys.readouterr()
    assert "add --regex" not in err


def test_function_search_no_regex_hint_for_plain_query(monkeypatch, capsys):
    """A plain query with no metacharacters and 0 matches gets no regex hint."""
    monkeypatch.setattr(bn.cli, "send_request", _zero_function_search)
    rc = bn.cli.main(["function", "search", "plainname", "--target", "active"])
    assert rc == 0
    _, err = capsys.readouterr()
    assert "add --regex" not in err


def test_taint_max_depth_zero_allowed():
    # 0 is a meaningful "intraprocedural only" choice and must be accepted.
    parser = bn.cli.build_parser()
    ns = parser.parse_args(
        ["taint", "forward", "-f", "h", "--source", "param:0", "--max-depth", "0", "--target", "active"])
    assert ns.max_depth == 0


def test_trace_max_depth_zero_rejected_at_parse_time(capsys):
    # `trace --max-depth 0` is a 0-step budget the bridge rejects; the CLI must
    # reject it at parse time so the contract matches (#129). --ip-depth 0 stays
    # valid (it means "do not cross call boundaries").
    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["trace", "f", "0x10", "--target", "active", "--max-depth", "0"])
    assert exc.value.code == 2
    _, err = capsys.readouterr()
    assert "depth must be an integer >= 1" in err


def test_decompile_text_format_unwraps_text_field(fake_transport, capsys):
    fake_transport({"decompile": {
        "ok": True,
        "result": {
            "function": {"name": "sub_401000", "address": "0x401000"},
            "text": "return 7;",
        },
    }})

    rc = bn.cli.main(["decompile", "--format", "text", "--target", "active", "sub_401000"])

    assert rc == 0
    assert capsys.readouterr().out == "return 7;\n"


def test_callsites_empty_result_shows_descriptive_message(fake_transport, capsys):
    fake_transport({"callsites": {"ok": True, "result": []}})

    rc = bn.cli.main(["callsites", "--format", "text", "--target", "active", "--within", "main", "sub_401000"])

    assert rc == 0
    assert capsys.readouterr().out == "no callsites found\n"


def test_function_list_pagination_states_true_total(fake_transport, capsys):
    # #59: the bridge returns the page WITH the true total; the footer states it
    # (showing N of TOTAL (REMAINING more)) on stdout. The CLI sends the real
    # limit (not limit+1).
    calls = fake_transport({"list_functions": {
        "ok": True,
        "result": {
            "functions": [{"name": f"sub_{i:06x}", "address": hex(i)} for i in range(20)],
            "total": 6350, "offset": 0, "limit": 20, "returned": 20, "has_more": True,
        },
    }})

    rc = bn.cli.main(["function", "list", "--target", "active", "--limit", "20"])

    assert rc == 0
    assert calls[-1]["params"]["limit"] == 20            # real limit, not +1
    stdout, _ = capsys.readouterr()
    assert "// showing 20 of 6350 (6330 more)" in stdout   # true total + remainder
    assert "--offset 20" in stdout


def test_function_list_json_carries_paging_metadata(fake_transport, capsys):
    # #59: machine consumers get total/has_more in JSON, not only a stderr note.
    fake_transport({"list_functions": {"ok": True, "result": {
        "functions": [{"name": "sub_0", "address": "0x0"}],
        "total": 6350, "offset": 0, "limit": 1, "returned": 1, "has_more": True,
    }}})
    rc = bn.cli.main(["function", "list", "--target", "active", "--limit", "1", "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 6350 and payload["has_more"] is True and payload["returned"] == 1


def test_function_search_pagination_forwards_offset(fake_transport, capsys):
    calls = fake_transport({"search_functions": {"ok": True, "result": [{"name": "sub_401000", "address": "0x401000"}]}})

    rc = bn.cli.main(["function", "search", "--target", "active", "--offset", "50", "--limit", "25", "sub"])

    assert rc == 0
    assert calls[-1]["params"]["offset"] == 50
    assert calls[-1]["params"]["limit"] == 25


def test_session_start_all_loads_fail_stops_bridge_and_exits_nonzero(monkeypatch, capsys):
    from bn.transport import BridgeError, BridgeInstance

    fake_inst = BridgeInstance(
        pid=999,
        socket_path=__import__("pathlib").Path("/tmp/test.sock"),
        registry_path=__import__("pathlib").Path("/tmp/test.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={},
        instance_id="ghost9",
    )
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: fake_inst)

    ops = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        ops.append(op)
        if op == "load_binary":
            raise BridgeError(f"File not found: {params['path']}")
        if op == "shutdown":
            return {"ok": True, "result": {"shutting_down": True}}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "start", "/tmp/nonexistent.so", "--format", "json"])

    # Non-zero exit, and the empty zombie bridge is shut down rather than leaked.
    assert rc == 1
    assert "shutdown" in ops
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["stopped"] is True


def test_il_lines_slices_output_with_header(fake_transport, capsys):
    fake_transport({"il": {"ok": True, "result": {"text": "line1\nline2\nline3\nline4\nline5"}}})

    rc = bn.cli.main(["il", "main", "--target", "active", "--lines", "2:4"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "// lines 2-4 of 5" in out
    assert "line2" in out and "line3" in out and "line4" in out
    assert "line1" not in out
    assert "line5" not in out


def test_disasm_lines_slices_output_with_header(fake_transport, capsys):
    fake_transport({"disasm": {"ok": True, "result": {"text": "aaa\nbbb\nccc\nddd"}}})

    rc = bn.cli.main(["disasm", "0x1000", "--target", "active", "--lines", "1:2"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "// lines 1-2 of 4" in out
    assert "aaa" in out and "bbb" in out
    assert "ccc" not in out and "ddd" not in out


def test_function_list_count_prints_total(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "x"}]}
        if op == "list_functions":
            assert params.get("count_only") is True
            return {"ok": True, "result": {"count": 4242}}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    rc = bn.cli.main(["function", "list", "--count"])

    assert rc == 0
    assert "Total functions: 4242" in capsys.readouterr().out


def test_taint_forward_passes_enabled_sink_classes(fake_transport):
    calls = fake_transport({
        "list_targets": {"ok": True, "result": [{"target_id": "1:1:1", "selector": "a.bndb"}]},
        "taint": {"ok": True, "result": {"direction": "forward",
                                         "function": {"name": "handler", "address": "0x10"},
                                         "sources": [], "reached_sinks": [], "leaves": [],
                                         "assumptions": [], "soundness": "may"}},
    })

    rc = bn.cli.main(["taint", "forward", "-f", "handler", "--source", "arg:read:1",
                      "--sink-class", "file_write", "--sink-class", "net_write",
                      "--format", "json"])
    assert rc == 0
    assert calls[-1]["op"] == "taint"
    assert calls[-1]["params"]["enabled_sink_classes"] == ["file_write", "net_write"]


def test_taint_forward_sink_classes_default_empty(fake_transport):
    calls = fake_transport({
        "list_targets": {"ok": True, "result": [{"target_id": "1:1:1", "selector": "a.bndb"}]},
        "taint": {"ok": True, "result": {"direction": "forward",
                                         "function": {"name": "handler", "address": "0x10"},
                                         "sources": [], "reached_sinks": [], "leaves": [],
                                         "assumptions": [], "soundness": "may"}},
    })

    rc = bn.cli.main(["taint", "forward", "-f", "handler", "--source", "arg:read:1",
                      "--format", "json"])
    assert rc == 0
    assert calls[-1]["params"]["enabled_sink_classes"] == []


def test_trace_routes_and_renders_text(fake_transport, capsys):
    calls = fake_transport({"backward_slice": {
        "ok": True,
        "result": {
            "function": "test_func",
                "function_address": "0x10000",
                "target_address": "0x10010",
                "arg_index": 0,
                "view": "mlil",
                "truncated": False,
                "step_count": 2,
                "trace": [
                    {
                        "ssa_var": "r0#1",
                        "depth": 0,
                        "address": "0x10010",
                        "il_text": "MLIL_CALL_SSA @ 0x10010",
                        "operation": "MLIL_SET_VAR_SSA",
                        "terminates": False,
                        "reason": None,
                    },
                    {
                        "ssa_var": "r1#2",
                        "depth": 1,
                        "address": None,
                        "il_text": None,
                        "operation": "undefined",
                        "terminates": True,
                        "reason": "function_parameter_or_global",
                    },
                ],
            },
        }})

    rc = bn.cli.main([
        "trace",
        "--format", "text",
        "--target", "active",
        "--arg", "0",
        "test_func",
        "0x10010",
    ])

    assert rc == 0
    assert calls[-1]["op"] == "backward_slice"
    assert calls[-1]["target"] == "active"
    assert calls[-1]["params"]["identifier"] == "test_func"
    assert calls[-1]["params"]["address"] == "0x10010"
    assert calls[-1]["params"]["arg_index"] == 0
    output = capsys.readouterr().out
    assert "backward trace of arg[0]" in output
    assert "test_func" in output
    assert "r0#1" in output or "0x10010" in output
    assert "r1#2" in output
    assert "function parameter" in output


def test_trace_defaults_to_arg_0(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = params or {}
        return {"ok": True, "result": {"function": "f", "trace": [], "step_count": 0, "truncated": False}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["trace", "f", "0x10010", "--target", "active"])
    assert rc == 0
    assert captured["params"]["arg_index"] == 0


def test_trace_respects_view_and_max_depth(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = dict(params or {})
        return {"ok": True, "result": {"function": "f", "trace": [], "step_count": 0, "truncated": False}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main([
        "trace", "f", "0x10010",
        "--target", "active",
        "--view", "hlil",
        "--max-depth", "10",
    ])
    assert rc == 0
    assert captured["params"]["view"] == "hlil"
    assert captured["params"]["max_depth"] == 10


def test_trace_text_renderer_empty_trace(fake_transport, capsys):
    fake_transport({"backward_slice": {"ok": True, "result": {"function": "f", "trace": [], "step_count": 0, "truncated": False}}})

    rc = bn.cli.main(["trace", "f", "0x10010", "--target", "active"])
    assert rc == 0
    output = capsys.readouterr().out
    assert "constant or immediate" in output or "no SSA trace" in output


def test_trace_json_renders_structure(fake_transport, capsys):
    fake_transport({"backward_slice": {
        "ok": True,
        "result": {
            "function": "f",
            "function_address": "0x10000",
            "target_address": "0x10010",
            "arg_index": 0,
            "view": "mlil",
            "truncated": False,
            "step_count": 2,
            "trace": [
                {"ssa_var": "x#1", "depth": 0, "terminates": False},
                {"ssa_var": "y#2", "depth": 1, "terminates": True, "reason": "function_parameter_or_global"},
            ],
        },
    }})

    rc = bn.cli.main(["trace", "f", "0x10010", "--target", "active", "--format", "json"])
    assert rc == 0
    output = capsys.readouterr().out
    import json as _json
    parsed = _json.loads(output)
    assert parsed["function"] == "f"
    assert len(parsed["trace"]) == 2


# --- function search --exact CLI routing + mutual exclusion ---


def test_function_search_exact_routes(fake_transport, capsys):
    calls = fake_transport({"search_functions": {"ok": True, "result": [{"name": "system", "address": "0x401000"}]}})

    rc = bn.cli.main(["function", "search", "--target", "active", "--exact", "system"])

    assert rc == 0
    assert calls[-1]["op"] == "search_functions"
    assert calls[-1]["params"]["query"] == "system"
    assert calls[-1]["params"]["exact"] is True
    assert calls[-1]["params"]["regex"] is False


def test_function_search_regex_and_exact_are_mutually_exclusive(monkeypatch):
    # argparse must reject both flags together (exit code 2), before any request.
    def fail_send_request(*a, **k):
        raise AssertionError("send_request should not be called when args are invalid")

    monkeypatch.setattr(bn.cli, "send_request", fail_send_request)

    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["function", "search", "--target", "active", "--regex", "--exact", "system"])
    assert exc.value.code == 2


def test_xrefs_json_limit_pages_instead_of_erroring(monkeypatch, capsys):
    # #164: xrefs --format json --limit N now pages (forwards offset/limit to the
    # bridge) instead of rejecting the flag.
    captured = _capture_xrefs_call(monkeypatch)
    rc = bn.cli.main(["xrefs", "sub_401000", "--target", "active", "--format", "json", "--limit", "3"])
    assert rc == 0
    assert captured["op"] == "xrefs"
    assert captured["params"].get("limit") == 3


def test_evidence_xrefs_json_limit_pages_instead_of_erroring(monkeypatch, capsys):
    captured = _capture_xrefs_call(monkeypatch)
    rc = bn.cli.main(
        ["evidence", "xrefs", "sub_401000", "--target", "active", "--format", "json", "--limit", "3", "--offset", "6"]
    )
    assert rc == 0
    assert captured["op"] == "xrefs"
    assert captured["params"].get("limit") == 3
    assert captured["params"].get("offset") == 6


def test_xrefs_text_does_not_page_the_op(monkeypatch, capsys):
    # #184: text mode groups the FULL set, so the CLI must not forward --limit/
    # --offset to the op (that would page `items` and group only a slice). --limit
    # stays a renderer-side caller-group display cap.
    captured = _capture_xrefs_call(monkeypatch)
    rc = bn.cli.main(
        ["xrefs", "sub_401000", "--target", "active", "--format", "text", "--limit", "5"]
    )
    assert rc == 0
    assert captured["op"] == "xrefs"
    assert "limit" not in captured["params"]
    assert "offset" not in captured["params"]


def test_evidence_xrefs_text_does_not_page_the_op(monkeypatch, capsys):
    captured = _capture_xrefs_call(monkeypatch)
    rc = bn.cli.main(
        ["evidence", "xrefs", "sub_401000", "--target", "active", "--format", "text", "--limit", "5"]
    )
    assert rc == 0
    assert captured["op"] == "xrefs"
    assert "limit" not in captured["params"]
    assert "offset" not in captured["params"]


def test_xrefs_offset_hint_uses_total_not_dropped_arrays(fake_transport, capsys):
    # #184: with code_refs/data_refs dropped from the op response, the struct-field
    # offset hint must key off the full-set `total`, not the (now-absent) arrays --
    # otherwise a small bare offset WITH refs would wrongly trigger the "0 xrefs"
    # nudge.
    fake_transport({"xrefs": {"ok": True, "result": {
        "address": "0x308", "target_context": {},
        "code_ref_count": 2, "data_ref_count": 0, "caller_function_count": 1,
        "items": [{"address": "0x401000", "function": "f", "kind": "code"}],
        "total": 2, "offset": 0, "limit": None, "returned": 1, "has_more": True,
    }}})
    rc = bn.cli.main(["xrefs", "0x308", "--target", "active"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "struct field offset" not in err  # refs exist -> no nudge


def test_function_list_out_exports_full_body(monkeypatch, capsys, tmp_path):
    # #165: `function list --out` must not silently cap at the default 100-item
    # page -- it sends no limit so the bridge returns the full body.
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = params or {}
        return {"ok": True, "result": {"functions": [], "items": [], "total": 0,
                                       "offset": 0, "limit": None, "returned": 0, "has_more": False}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    out = tmp_path / "fns.json"
    rc = bn.cli.main(["function", "list", "--out", str(out), "--format", "json", "--target", "active"])
    assert rc == 0
    assert "limit" not in captured["params"]  # uncapped full-body export


def test_xrefs_identifier_and_field_are_mutually_exclusive(monkeypatch, capsys):
    _assert_no_bridge_call(monkeypatch)

    rc = bn.cli.main(["xrefs", "some_func", "--field", "T.x", "--target", "active"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "not both" in err
    assert "some_func" in err
    assert "T.x" in err


def test_slice_text_lines_start_beyond_end_keeps_header_sane():
    from bn import formatters

    out = formatters._slice_text_lines("a\nb\nc", (10, 12))

    assert out == "// lines 0 of 3 (start 10 is beyond the last line)"


def test_trace_render_step_grammar_singular_and_plural():
    from bn import formatters
    base = {
        "function": "f", "function_address": "0x1000", "target_address": "0x1010",
        "arg_index": 0, "truncated": False,
        "trace": [{"ssa_var": "x#1", "terminates": True, "reason": "undefined_or_global", "depth": 0}],
    }
    out1 = formatters._render_trace_text(base)
    assert "1 step" in out1 and "1 steps" not in out1
    two = dict(base, trace=base["trace"] + [
        {"ssa_var": "y#1", "terminates": True, "reason": "undefined_or_global", "depth": 1}])
    assert "2 steps" in formatters._render_trace_text(two)


def test_trace_render_shows_arg_register_and_field_load_meta():
    # header labels the traced arg with its register + C name; a field_load step
    # renders base/offset/width (#162, #166).
    from bn import formatters
    value = {
        "function": "parse", "function_address": "0x1000", "target_address": "0x1040",
        "arg_index": 1, "arg_label": {"index": 1, "register": "x1", "name": "buf"},
        "truncated": False, "step_count": 1,
        "trace": [{"ssa_var": "len#3", "ssa_label": "len#3", "depth": 0,
                   "reason": "field_load", "terminates": True,
                   "base": "obj#1", "offset": "0x8", "width": 4,
                   "il_text": "len#3 = [obj#1 + 8]", "address": "0x1030"}],
        "hints": [],
    }
    text = formatters._render_trace_text(value)
    assert 'arg[1] (x1, "buf")' in text
    assert "field load" in text
    assert "base=obj#1" in text and "offset=0x8" in text and "width=4" in text


def test_trace_render_output_pointer_hint_replaces_empty_trace_message():
    # an empty trace WITH an output-pointer hint shows the hint, not the bare
    # "constant or immediate" dead-end (#166).
    from bn import formatters
    value = {
        "function": "h", "function_address": "0x1000", "target_address": "0x1040",
        "arg_index": 1, "arg_label": {"index": 1, "register": "x1"},
        "truncated": False, "step_count": 0, "trace": [],
        "hints": ["arg 1 is a pointer (address-of); trace the pointee instead."],
    }
    text = formatters._render_trace_text(value)
    assert "constant or immediate" not in text
    assert "hint:" in text
    assert "pointer" in text


def test_xrefs_distinct_functionless_refs_are_not_merged_under_one_label():
    # Two function-less data refs with DIFFERENT symbols must render as two
    # distinct, correctly-labeled lines -- not collapse into one group stamped
    # with the first ref's symbol (which mislabeled the second site) (#24).
    from bn import formatters
    value = {
        "address": "0x18d58", "code_refs": [],
        "data_refs": [
            {"address": "0xaaaa", "caller_function": None, "function": None,
             "context": {"sections": [{"name": ".got"}], "symbol": {"name": "sym_a"}}},
            {"address": "0xbbbb", "caller_function": None, "function": None,
             "context": {"sections": [{"name": ".data"}], "symbol": {"name": "sym_b"}}},
        ],
    }
    out = formatters._render_xrefs_text(value)
    # function-less refs are "locations", not "functions" (#49)
    assert "data refs: 2 sites across 2 locations" in out
    # each site is labeled with its OWN symbol, on a line with its OWN address
    assert any("0xaaaa" in ln and "sym_a" in ln and "sym_b" not in ln for ln in out.splitlines())
    assert any("0xbbbb" in ln and "sym_b" in ln and "sym_a" not in ln for ln in out.splitlines())


def test_xrefs_same_label_functionless_refs_still_coalesce():
    # Refs sharing a resolved label are still grouped (two sites, one line).
    from bn import formatters
    value = {
        "address": "0x18d58", "code_refs": [],
        "data_refs": [
            {"address": "0xaaaa", "caller_function": None, "function": None,
             "context": {"symbol": {"name": "sym_a"}}},
            {"address": "0xbbbb", "caller_function": None, "function": None,
             "context": {"symbol": {"name": "sym_a"}}},
        ],
    }
    out = formatters._render_xrefs_text(value)
    assert "data refs: 2 sites across 1 location" in out   # function-less -> location (#49)
    assert any("0xaaaa" in ln and "0xbbbb" in ln for ln in out.splitlines())


def test_evidence_xrefs_text_reports_truncation_total():
    # Capping evidence xrefs must surface the true total + a "showing first N"
    # marker, not silently drop refs (#31, the #13 honesty convention).
    from bn import formatters
    value = {
        "address": "0x1000",
        "code_refs": [{"address": hex(0x2000 + i), "kind": "code"} for i in range(14)],
        "data_refs": [],
    }
    out = formatters._render_evidence_xrefs_text(value, limit=3)
    assert "code refs: 14 total, showing first 3" in out
    # only 3 ref lines rendered
    assert sum(1 for ln in out.splitlines() if ln.startswith("- 0x")) == 3
    # uncapped render shows the plain header (no truncation marker)
    out_full = formatters._render_evidence_xrefs_text(value)
    assert "code refs:" in out_full and "total, showing" not in out_full


def test_function_list_reverse_threads_to_op(monkeypatch):
    """--reverse threads a reverse param so the bridge flips the sort order (#221)."""
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = params
        return {"ok": True, "result": {"items": [], "total": 0, "offset": 0,
                                       "limit": None, "returned": 0, "has_more": False}}
    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["function", "list", "--target", "active", "--sort", "size", "--reverse"])
    assert rc == 0
    assert captured["params"].get("reverse") is True
    assert captured["params"].get("sort") == "size"


def test_xrefs_any_batch_probes_symbols(monkeypatch, capsys):
    """`xrefs --any` probes several symbols in one call; absent ones are reported,
    not errors, and the command exits 0 (#218)."""
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "xrefs_any"
        assert params["symbols"] == ["memcpy", "strcpy", "system"]
        return {"ok": True, "result": {
            "count": 3, "present": 2,
            "symbols": [
                {"symbol": "memcpy", "present": True, "code_ref_count": 12,
                 "caller_function_count": 4, "address": "0x1000"},
                {"symbol": "strcpy", "present": False, "note": "Function not found: strcpy"},
                {"symbol": "system", "present": True, "code_ref_count": 1,
                 "caller_function_count": 1, "address": "0x2000"},
            ]}}
    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["xrefs", "--target", "active", "--any", "memcpy", "strcpy", "system"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "memcpy: 12 code refs across 4 fn(s)" in out
    assert "strcpy: absent" in out


