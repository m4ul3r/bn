from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from bn.output import DEFAULT_SPILL_TOKEN_LIMIT
from bn.output import OutputWriteError
from bn.output import estimate_tokens
from bn.output import write_output


def _token_count(text: str) -> int:
    return estimate_tokens(text.encode("utf-8"))


def _parse_envelope(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        key, value = line.split(":", 1)
        result[key] = value.strip()
    return result


def test_default_spill_token_limit_is_10k():
    assert DEFAULT_SPILL_TOKEN_LIMIT == 10_000


def test_summary_reports_array_count_for_paged_envelope():
    # A paged-list envelope must summarize the array's element count + logical
    # total, not the count of envelope KEYS (which read as count=6 on any spill).
    from bn.output import _summary
    s = _summary({"items": [1, 2, 3], "total": 42, "offset": 0,
                  "limit": 3, "returned": 3, "has_more": True})
    assert s["count"] == 3 and s["total"] == 42 and s["page_key"] == "items"
    # function-listing envelope (key 'functions') is handled too
    s2 = _summary({"functions": [1, 2], "total": 2, "offset": 0,
                   "limit": None, "returned": 2, "has_more": False})
    assert s2["count"] == 2 and s2["total"] == 2
    # a plain (non-envelope) object still reports its key count
    assert _summary({"a": 1, "b": 2})["count"] == 2


def test_write_output_renders_small_payload_without_spill(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))

    rendered = write_output({"ok": True}, fmt="json", out_path=None, stem="small")

    payload = json.loads(rendered)
    assert payload["ok"] is True


def test_write_output_spills_large_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    payload = {"data": [f"item-{index:04d}" for index in range(1000)]}

    rendered = write_output(
        payload,
        fmt="json",
        out_path=None,
        stem="large",
        spill_token_limit=256,
    )

    # Under --format json the stdout envelope must itself be valid JSON (issue #10)
    # so that `bn <cmd> --format json | jq` keeps working at spill scale.
    envelope = json.loads(rendered)
    # Spills live under the (BN_CACHE_DIR-overridable) cache root, not /tmp.
    assert envelope["artifact_path"].startswith(str(tmp_path / "spills"))
    assert envelope["spilled"] is True
    artifact_text = Path(envelope["artifact_path"]).read_text()
    assert envelope["tokenizer"] == "estimate"
    assert int(envelope["tokens"]) == _token_count(artifact_text)
    # Filename carries pid + random component so parallel agents spilling in
    # the same second can't clobber each other.
    name = Path(envelope["artifact_path"]).name
    assert re.fullmatch(rf"large-\d{{6}}-{os.getpid()}-[0-9a-f]{{4}}\.json", name)


def test_spilled_paged_envelope_hoists_canonical_total(tmp_path, monkeypatch):
    # #311: a spilled JSON envelope must expose the logical `total` at the TOP
    # LEVEL so `jq '.total'` returns the real count whether or not the read
    # spilled -- `jq '.items'` reads null on a spill (data is on disk), which
    # otherwise misreads a 209-result read as "0".
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    payload = {
        "kind": "xrefs",
        "items": [{"address": f"0x{i:x}"} for i in range(120)],
        "total": 209, "offset": 0, "limit": 120, "returned": 120, "has_more": True,
    }
    envelope = json.loads(write_output(payload, fmt="json", out_path=None,
                                       stem="xrefs", spill_token_limit=64))
    assert envelope["spilled"] is True
    assert "items" not in envelope          # the trap: items are on disk
    assert envelope["total"] == 209         # canonical count, spill-stable
    assert envelope["summary"]["total"] == 209
    assert envelope["summary"]["count"] == 120  # the on-disk page size, not the total


def test_spilled_non_paged_value_has_no_spurious_total(tmp_path, monkeypatch):
    # The negative: a non-paged spill (no items/functions page -- e.g. a big
    # decompile string or a plain dict) must NOT get a spurious top-level total;
    # only paged collections carry a logical total (#311).
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    envelope = json.loads(write_output({"text": "x" * 5000, "warnings": ["w"]},
                                       fmt="json", out_path=None, stem="decompile",
                                       spill_token_limit=64))
    assert envelope["spilled"] is True
    assert "total" not in envelope


def test_write_output_spills_text_payload_with_txt_suffix(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    payload = "\n".join(f"line {index} with distinctive content" for index in range(1000))

    rendered = write_output(
        payload,
        fmt="text",
        out_path=None,
        stem="large-text",
        spill_token_limit=256,
    )

    envelope = _parse_envelope(rendered)
    assert envelope["path"].endswith(".txt")
    assert envelope["spilled"] == "true"


def test_text_spill_envelope_stays_plaintext(tmp_path, monkeypatch):
    # Only json/ndjson change to a machine-readable envelope; text keeps the
    # human-readable key:value form (issue #10 must not regress text output).
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    payload = "\n".join(f"line {index} distinctive" for index in range(1000))

    rendered = write_output(
        payload, fmt="text", out_path=None, stem="t", spill_token_limit=256
    )

    with pytest.raises(json.JSONDecodeError):
        json.loads(rendered)
    envelope = _parse_envelope(rendered)
    assert envelope["spilled"] == "true"
    assert envelope["path"].endswith(".txt")


def test_ndjson_streams_paged_envelope_records():
    """ndjson on a paged envelope emits ONE record per item per line plus a
    trailing {"_meta": true, ...} line -- real newline-delimited streaming, not
    the whole envelope collapsed onto a single line. (J5)"""
    from bn.output import render_value

    env = {"items": [{"i": 0}, {"i": 1}, {"i": 2}], "total": 3, "offset": 0,
           "limit": 3, "returned": 3, "has_more": False}
    lines = render_value(env, "ndjson").strip().split("\n")
    assert len(lines) == 4  # 3 records + 1 meta
    recs = [json.loads(line) for line in lines]
    assert recs[:3] == [{"i": 0}, {"i": 1}, {"i": 2}]
    assert recs[3]["_meta"] is True
    assert recs[3]["total"] == 3 and recs[3]["has_more"] is False
    assert "items" not in recs[3]

    # function-list dual key: stream by items, meta excludes BOTH page arrays
    env2 = {"functions": [{"a": 1}], "items": [{"a": 1}], "total": 1, "has_more": False}
    lines2 = render_value(env2, "ndjson").strip().split("\n")
    assert len(lines2) == 2
    meta2 = json.loads(lines2[1])
    assert "functions" not in meta2 and "items" not in meta2

    # a non-paged dict still renders as a single line (decompile, target info, ...)
    assert len(render_value({"text": "x", "name": "f"}, "ndjson").strip().split("\n")) == 1


def test_ndjson_spill_envelope_is_one_json_line(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    payload = [{"i": index} for index in range(1000)]

    rendered = write_output(
        payload, fmt="ndjson", out_path=None, stem="nd", spill_token_limit=256
    )

    assert len(rendered.splitlines()) == 1
    envelope = json.loads(rendered)
    assert envelope["spilled"] is True
    assert envelope["artifact_path"].endswith(".ndjson")


def test_write_output_uses_token_limit_not_byte_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    payload = "x" * 1000
    token_limit = _token_count(payload + "\n") + 1

    rendered = write_output(
        payload,
        fmt="text",
        out_path=None,
        stem="byte-heavy",
        spill_token_limit=token_limit,
    )

    assert rendered == payload + "\n"


def test_write_output_spill_filenames_do_not_collide(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    payload = {"data": [f"item-{index:04d}" for index in range(1000)]}

    paths = set()
    for _ in range(3):
        rendered = write_output(
            payload,
            fmt="json",
            out_path=None,
            stem="same-stem",
            spill_token_limit=256,
        )
        paths.add(json.loads(rendered)["artifact_path"])

    assert len(paths) == 3


def test_write_output_falls_back_to_full_output_when_spill_write_fails(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    payload = {"data": [f"item-{index:04d}" for index in range(1000)]}
    # json is emitted compact now (#215), so the fallback full output is compact too.
    expected = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"

    def _boom(self, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_bytes", _boom)

    rendered = write_output(
        payload,
        fmt="json",
        out_path=None,
        stem="spill-fail",
        spill_token_limit=256,
    )

    assert rendered == expected
    err = capsys.readouterr().err
    assert "warning: failed to write spill artifact" in err
    assert "printing full output" in err


def test_write_output_raises_clean_error_when_explicit_out_write_fails(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    out_path = tmp_path / "artifacts" / "payload.json"

    def _boom(self, data):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(Path, "write_bytes", _boom)

    with pytest.raises(OutputWriteError, match=r"Failed to write --out file .*payload\.json"):
        write_output({"ok": True}, fmt="json", out_path=out_path, stem="out-fail")


def test_output_write_error_is_a_bridge_error():
    # cli.main() only renders BridgeError cleanly; OutputWriteError must stay
    # in that hierarchy or --out failures regress into tracebacks.
    from bn.transport import BridgeError

    assert issubclass(OutputWriteError, BridgeError)


def test_write_output_reports_exact_tokens_for_explicit_out_path(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))

    out_path = tmp_path / "artifacts" / "payload.json"
    rendered = write_output(
        {"message": "token-aware output"},
        fmt="json",
        out_path=out_path,
        stem="explicit-out",
    )

    envelope = json.loads(rendered)
    artifact_text = out_path.read_text()
    assert envelope["artifact_path"] == str(out_path)
    assert envelope["spilled"] is False
    assert envelope["tokenizer"] == "estimate"
    assert int(envelope["tokens"]) == _token_count(artifact_text)
