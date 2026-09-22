from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from bn.output import DEFAULT_SLICE_NOTE_TOKENS
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


def test_default_slice_note_threshold_is_10k():
    assert DEFAULT_SLICE_NOTE_TOKENS == 10_000


def test_default_spill_retention_is_14_days_591():
    """runtime.md states this window to agents; the two must not drift."""
    from bn.output import DEFAULT_SPILL_RETENTION_DAYS

    assert DEFAULT_SPILL_RETENTION_DAYS == 14


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


def test_ndjson_does_not_clobber_a_payload_that_carries_its_own_meta():
    """`_meta` is a sentinel the paging fan-out INVENTS, so a payload already
    carrying that key cannot be represented as a stream: the trailing record
    would overwrite the caller's own value. Such a payload falls through to the
    single-record form, matching the bridge-side `--out` writer
    (`_shared.py::_write_json_artifact`) so the two stay interchangeable."""
    from bn.output import render_value

    rendered = render_value(
        {"items": [{"a": 1}], "total": 1, "_meta": "caller's own"}, "ndjson"
    )
    lines = rendered.strip().split("\n")
    assert len(lines) == 1, lines
    record = json.loads(lines[0])
    assert record["_meta"] == "caller's own"
    assert record["items"] == [{"a": 1}]


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

    # #612: spill writes go through _write_private_bytes (os.open at 0o600), not
    # Path.write_bytes, so inject the disk-full failure there.
    def _boom(path, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("bn.output._write_private_bytes", _boom)

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


def test_a_failed_spill_write_still_draws_the_slicing_note(tmp_path, monkeypatch, capsys):
    """The OSError fallback puts the FULL payload on stdout, which is exactly the
    case the slicing note exists for -- but that return left `token_count` at its
    0 default, so a read 80x its armed bound arrived with no guidance at all
    (dogfood pass 3, reproduced on two targets)."""
    from bn.output import write_output_result
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))

    def _boom(path, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("bn.output._write_private_bytes", _boom)
    payload = {"kind": "functions",
               "items": [f"0x401000 sub_{i:06d}" for i in range(4000)]}

    res = write_output_result(payload, fmt="json", out_path=None, stem="functions",
                              spill_token_limit=1000)

    assert res.spilled is False and res.artifact is None
    assert res.token_count >= 10_000
    assert res.truncation_risk is True
    assert "printing full output" in capsys.readouterr().err


def test_a_failed_spill_below_the_default_still_draws_the_note(tmp_path, monkeypatch, capsys):
    """An ARMED threshold below 10 000 is a request for a file at that size, so
    when the write fails the note must fire from the ARMED bound rather than the
    default: armed 50 with a 2 823-token payload (56x the bound) printed the
    failure warning and no guidance at all (dogfood pass 4)."""
    from bn.output import write_output_result
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("bn.output._write_private_bytes",
                        lambda *a, **k: (_ for _ in ()).throw(OSError(28, "full")))

    for limit, tokens in ((50, 2823), (1000, 5608)):
        res = write_output_result("A" * (tokens * 3 - 1), fmt="text", out_path=None,
                                  stem="functions", spill_token_limit=limit)
        assert res.spilled is False and res.token_count == tokens
        assert res.truncation_risk is True, (limit, tokens)
    assert "failed to write spill artifact" in capsys.readouterr().err


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


def test_resolve_spill_limit_is_opt_in_409(monkeypatch):
    from bn.output import resolve_spill_limit
    monkeypatch.delenv("BN_SPILL_TOKENS", raising=False)
    assert resolve_spill_limit() is None          # unset -> never spill
    monkeypatch.setenv("BN_SPILL_TOKENS", "40000")
    assert resolve_spill_limit() == 40000
    monkeypatch.setenv("BN_SPILL_TOKENS", "0x1000")
    assert resolve_spill_limit() == 0x1000
    # non-positive / junk -> no threshold: a typo must never re-arm disk output
    for bad in ("0", "-5", "notanumber", ""):
        monkeypatch.setenv("BN_SPILL_TOKENS", bad)
        assert resolve_spill_limit() is None


def test_the_envelope_carries_a_derived_rerun_hint_and_none_without_one(tmp_path, monkeypatch):
    """The `rerun` key is the CLI's DERIVED hint, handed in by `_render_result`.

    The stem-keyed builder that used to live in this module is gone (it named
    flags its command rejects), so a direct library call with no hint gets no
    `rerun` key at all -- there is no command to name a flag for -- and a hint
    passed in is what the envelope (and its text rendering) carries.
    """
    from bn.output import render_artifact_envelope, write_output_result
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    value = {"kind": "functions", "items": [{"name": f"f{i}"} for i in range(500)], "total": 500}

    derived = "rerun with --limit/--offset to page through the results"
    res = write_output_result(value, fmt="json", out_path=None, stem="functions",
                              spill_token_limit=64, rerun_hint=derived)
    assert res.spilled is True
    assert res.artifact["rerun"] == derived
    assert res.artifact["spill_token_limit"] == 64
    assert "rerun" in res.rendered  # rendered JSON envelope carries the knob key
    assert "rerun:" in render_artifact_envelope(res.artifact)

    bare = write_output_result(value, fmt="json", out_path=None, stem="functions",
                               spill_token_limit=64)
    assert bare.spilled is True
    assert "rerun" not in bare.artifact
    assert "rerun" not in bare.rendered


def test_near_spill_flag_409(tmp_path, monkeypatch):
    from bn.output import write_output_result
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    # comfortably small -> not near
    small = write_output_result({"kind": "x", "items": [1]}, fmt="json", out_path=None,
                                stem="functions", spill_token_limit=100000)
    assert small.near_spill is False and small.spilled is False
    # sized to land in [80%,100%] of a small limit -> near_spill, not spilled
    payload = {"kind": "x", "items": ["y" * 10 for _ in range(30)]}
    import json as _json
    tok = -(-len(_json.dumps(payload)) // 3)
    res = write_output_result(payload, fmt="json", out_path=None, stem="functions",
                              spill_token_limit=int(tok / 0.9))
    assert res.spilled is False and res.near_spill is True


def test_artifact_carries_target_and_instance_provenance_653(tmp_path):
    """#653.8: two agents sharing a scratchpad both wrote `fns.json`; one silently
    read the other's list -- a different target, a different binary -- and concluded
    its own name recovery covered 6 of 1006 functions. Nothing in the artifact made
    that detectable, so stamp WHICH target/instance produced it."""
    from bn.output import write_output_result

    out = tmp_path / "fns.json"
    res = write_output_result({"kind": "functions", "items": [1, 2]}, fmt="json",
                              out_path=out, stem="functions",
                              provenance={"target": "firmware.bndb", "instance": "a1b2c3"})
    assert res.artifact["target"] == "firmware.bndb"
    assert res.artifact["instance"] == "a1b2c3"
    # `sha256` remains the digest of THIS artifact's bytes (not the binary's).
    assert len(res.artifact["sha256"]) == 64
    assert "target: firmware.bndb" in res.rendered or '"target":"firmware.bndb"' in res.rendered


def test_spill_envelope_carries_provenance_653(tmp_path, monkeypatch):
    from bn.output import write_output_result

    monkeypatch.setenv("BN_SPILL_TOKENS", "10")
    res = write_output_result({"kind": "functions", "items": ["x" * 200]}, fmt="json",
                              out_path=None, stem="functions",
                              provenance={"target": "firmware.bndb", "instance": "a1b2c3"})
    assert res.spilled is True
    assert res.artifact["target"] == "firmware.bndb"
    assert res.artifact["instance"] == "a1b2c3"


def test_provenance_omits_unknown_values_653():
    """A None target/instance must not stamp a misleading null."""
    from bn.output import write_output_result

    res = write_output_result({"kind": "x"}, fmt="json", out_path=None, stem="x",
                              provenance={"target": None, "instance": None})
    assert res.artifact is None      # small output: no artifact envelope at all


# --- #591: spill retention -------------------------------------------------

def _make_day(root, day):
    d = root / day.strftime("%Y%m%d")
    d.mkdir(parents=True)
    (d / "decompile-120000-1-ab.txt").write_text("artifact")
    return d


def test_prune_removes_only_days_past_the_retention_window_591(tmp_path, monkeypatch):
    """The measured failure was 1.0 GB / 4187 files with no prune path at all.
    Retention is by whole day-directory, and the window boundary is inclusive:
    a day exactly `retention` days old is still inside the window."""
    from datetime import date, timedelta

    from bn.output import _prune_old_spill_days

    monkeypatch.delenv("BN_SPILL_RETENTION_DAYS", raising=False)
    today = date(2026, 9, 15)
    stale = _make_day(tmp_path, today - timedelta(days=15))
    boundary = _make_day(tmp_path, today - timedelta(days=14))
    fresh = _make_day(tmp_path, today - timedelta(days=1))
    current = _make_day(tmp_path, today)

    removed = _prune_old_spill_days(tmp_path, today)

    assert removed == [stale]
    assert not stale.exists()
    assert boundary.exists() and fresh.exists() and current.exists()


def test_prune_leaves_anything_it_did_not_create_591(tmp_path, monkeypatch):
    """Only a name that parses exactly as %Y%m%d is eligible. Everything else
    in the spill root belongs to someone else and survives regardless of age --
    a prune that guessed from name shape would delete a user's notes."""
    from datetime import date, timedelta

    from bn.output import _prune_old_spill_days

    monkeypatch.delenv("BN_SPILL_RETENTION_DAYS", raising=False)
    today = date(2026, 9, 15)
    stale = _make_day(tmp_path, today - timedelta(days=90))
    keep_dirs = [tmp_path / "notes", tmp_path / "2026-06-09", tmp_path / "20260609-old"]
    for d in keep_dirs:
        d.mkdir()
    loose_file = tmp_path / "20260609"          # right name, but NOT a directory
    loose_file.write_text("someone else's file")

    removed = _prune_old_spill_days(tmp_path, today)

    assert removed == [stale]
    assert all(d.exists() for d in keep_dirs)
    assert loose_file.read_text() == "someone else's file"


def test_retention_zero_disables_pruning_591(tmp_path, monkeypatch):
    from datetime import date, timedelta

    from bn.output import _prune_old_spill_days

    monkeypatch.setenv("BN_SPILL_RETENTION_DAYS", "0")
    today = date(2026, 9, 15)
    ancient = _make_day(tmp_path, today - timedelta(days=365))

    assert _prune_old_spill_days(tmp_path, today) == []
    assert ancient.exists()


def test_a_typo_in_the_retention_env_falls_back_to_the_default_591(monkeypatch):
    """A bad value must not mean 'keep forever' -- that is the bug, and a typo
    silently restoring it is how the 1 GB directory happened."""
    from bn.output import DEFAULT_SPILL_RETENTION_DAYS, resolve_spill_retention_days

    for bad in ("", "  ", "forever", "-5"):
        monkeypatch.setenv("BN_SPILL_RETENTION_DAYS", bad)
        assert resolve_spill_retention_days() == DEFAULT_SPILL_RETENTION_DAYS
    monkeypatch.setenv("BN_SPILL_RETENTION_DAYS", "3")
    assert resolve_spill_retention_days() == 3


def test_spilling_prunes_the_stale_days_it_finds_591(tmp_path, monkeypatch):
    """End-to-end: the prune is wired into the write path, so an agent that
    never runs a cleanup command still gets a bounded spill root."""
    from datetime import datetime, timedelta, timezone

    import bn.output as output

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("BN_SPILL_TOKENS", "10")
    monkeypatch.delenv("BN_SPILL_RETENTION_DAYS", raising=False)
    monkeypatch.setattr(output, "_spill_pruned", False)

    stale = _make_day(output.spill_root(), datetime.now(timezone.utc).date() - timedelta(days=30))

    res = output.write_output_result({"items": ["x" * 400]}, fmt="json",
                                     out_path=None, stem="functions")

    assert res.spilled is True
    assert not stale.exists()


def test_the_spill_sweep_runs_once_per_process_591(tmp_path, monkeypatch):
    """A command that spills fifty pages must not re-scan the spill root fifty
    times; the sweep is a process-lifetime hygiene pass, not per-artifact."""
    import bn.output as output

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("BN_SPILL_TOKENS", "10")
    monkeypatch.setattr(output, "_spill_pruned", False)
    calls = []
    monkeypatch.setattr(output, "_prune_old_spill_days",
                        lambda root, today: calls.append(root) or [])

    for _ in range(3):
        output.write_output_result({"items": ["x" * 400]}, fmt="json",
                                   out_path=None, stem="functions")

    assert len(calls) == 1


def test_prune_requires_a_canonical_date_name_591(tmp_path, monkeypatch):
    """`strptime`'s numeric fields accept UNPADDED input, so `202611` parses as
    2026-01-01 and a directory by that name -- which the writer, which always
    renders via `strftime`, could never have produced -- was recursively
    deleted along with its contents. The round-trip is the check."""
    from datetime import date

    from bn.output import _prune_old_spill_days

    monkeypatch.delenv("BN_SPILL_RETENTION_DAYS", raising=False)
    # Each parses under %Y%m%d, is NOT what strftime emits, and resolves to a
    # date old enough to be inside the prune window -- so only the round-trip
    # guard saves them. (A future-dated name would survive on age alone and
    # prove nothing.)
    for name in ("202611", "2026011", "202612", "2026061"):
        d = tmp_path / name
        d.mkdir(exist_ok=True)
        (d / "keep.txt").write_text("someone else's data")

    removed = _prune_old_spill_days(tmp_path, date(2026, 9, 15))

    assert removed == []
    for entry in tmp_path.iterdir():
        assert (entry / "keep.txt").read_text() == "someone else's data"


def test_prune_still_removes_the_canonical_stale_day_591(tmp_path, monkeypatch):
    """Negative control for the round-trip guard: tightening the name check
    must not stop the prune doing its job on a name the writer did emit."""
    from datetime import date

    from bn.output import _prune_old_spill_days

    monkeypatch.delenv("BN_SPILL_RETENTION_DAYS", raising=False)
    stale = _make_day(tmp_path, date(2026, 1, 1))
    assert stale.name == "20260101"

    assert _prune_old_spill_days(tmp_path, date(2026, 9, 15)) == [stale]
    assert not stale.exists()


def test_an_enormous_retention_window_keeps_everything_591(tmp_path, monkeypatch):
    """A window wider than the calendar overflowed `date` arithmetic. The spill
    fallback catches only OSError, so the OverflowError denied the caller both
    its result AND its artifact -- a config value silently discarding output."""
    from datetime import date, timedelta

    import bn.output as output

    monkeypatch.setenv("BN_SPILL_RETENTION_DAYS", "9999999")
    ancient = _make_day(tmp_path, date(2026, 9, 15) - timedelta(days=365))

    assert output._prune_old_spill_days(tmp_path, date(2026, 9, 15)) == []
    assert ancient.exists()


def test_an_enormous_retention_window_still_returns_the_output_591(tmp_path, monkeypatch):
    """End-to-end: the overflow was raised from inside the spill write path."""
    import bn.output as output

    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("BN_SPILL_TOKENS", "10")
    monkeypatch.setenv("BN_SPILL_RETENTION_DAYS", "9999999")
    monkeypatch.setattr(output, "_spill_pruned", False)

    res = output.write_output_result({"items": ["x" * 400]}, fmt="json",
                                     out_path=None, stem="functions")

    assert res.spilled is True
    assert res.artifact["bytes"] > 0


def test_no_spill_by_default_and_truncation_risk_flagged(tmp_path, monkeypatch):
    """The default writes nothing to disk and keeps the whole payload on stdout;
    a payload this large is the consumer's problem, and the caller is told so."""
    from bn.output import write_output_result
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("BN_SPILL_TOKENS", raising=False)
    payload = {"kind": "functions",
               "items": [f"0x401000 sub_{i:06d}" for i in range(4000)]}
    res = write_output_result(payload, fmt="json", out_path=None, stem="functions")
    assert res.spilled is False and res.artifact is None
    assert res.truncation_risk is True and res.near_spill is False
    assert res.token_count >= 10_000
    assert not (tmp_path / "spills").exists()      # nothing touched the disk
    assert "sub_003999" in res.rendered            # the payload, not an envelope
    assert "artifact_path" not in res.rendered


def test_opting_in_restores_spill_and_suppresses_the_note(tmp_path, monkeypatch):
    from bn.output import write_output_result
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("BN_SPILL_TOKENS", "10")
    res = write_output_result({"kind": "functions", "items": ["x" * 200]}, fmt="json",
                              out_path=None, stem="functions")
    assert res.spilled is True
    assert res.truncation_risk is False            # the configured limit governs
    assert res.token_count > 0                     # documented as always populated


def test_an_armed_threshold_above_the_payload_still_draws_the_note(tmp_path, monkeypatch):
    """Arming a threshold ABOVE the payload used to silence the note entirely, so
    a read between 10 000 tokens and 80 % of the threshold got no slicing guidance
    at all -- while the same read with nothing armed printed it (dogfood C3)."""
    from bn.output import write_output_result
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    payload = {"kind": "functions",
               "items": [f"0x401000 sub_{i:06d}" for i in range(4000)]}

    monkeypatch.setenv("BN_SPILL_TOKENS", "1000000")     # far above the payload
    res = write_output_result(payload, fmt="json", out_path=None, stem="functions")
    assert res.spilled is False and res.near_spill is False
    assert res.truncation_risk is True
    assert not (tmp_path / "spills").exists()

    # Inside the 20 % band the sharper signal replaces it; two notes for one read
    # is noise.
    monkeypatch.setenv("BN_SPILL_TOKENS", "30000")        # the payload is ~29 344
    res = write_output_result(payload, fmt="json", out_path=None, stem="functions")
    assert res.spilled is False
    assert res.near_spill is True and res.truncation_risk is False


# --- #823: `bn spill gc` (and the hardening it shares with the write path) ---


def _tree_bytes(path: Path) -> int:
    """Independent byte measure of a directory tree, so the report's own number
    is never the thing that checks itself."""
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _no_rmtree(monkeypatch, output, *, error: OSError | None = None) -> list[str]:
    """Replace the removal primitive output.py reaches, recording every path.

    A local stand-in rather than a patch of the shared ``shutil`` module: the
    assertion under test is which paths are HANDED to the remover, and with a
    stand-in nothing else in the session can be deleted by a bug in the code
    being tested. With *error*, every call raises it -- the shape a removal that
    lost a race (or hit a busy mount) presents.
    """
    calls: list[str] = []

    class _Rmtree:
        @staticmethod
        def rmtree(path, *args, **kwargs):
            calls.append(str(path))
            if error is not None:
                raise error

    monkeypatch.setattr(output, "shutil", _Rmtree)
    return calls


def test_gc_never_hands_a_symlinked_day_to_rmtree_823(tmp_path, monkeypatch):
    """The pre-#823 loop tested ``entry.is_dir()``, which FOLLOWS a symlink, so a
    symlink named as an old day was passed to ``shutil.rmtree`` and survived
    only because rmtree refuses symlinks and the ``OSError`` was swallowed -- a
    refusal nobody asked for and nobody could see.

    The spy is the assertion: "the target survived" passes on the old code too,
    because rmtree is what refused it. What must not happen is the call.
    """
    from datetime import date

    import bn.output as output

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("someone else's data")
    link = tmp_path / "20260101"                      # ancient, and a symlink
    link.symlink_to(outside, target_is_directory=True)
    calls = _no_rmtree(monkeypatch, output)

    # The write path's sweep first: it is the one that used to hand the symlink
    # over, and the stand-in does not raise, so a call is visible as a removal.
    assert output._prune_old_spill_days(tmp_path, date(2026, 9, 15)) == []
    assert calls == []

    report = output.gc_spills(root=tmp_path, today=date(2026, 9, 15), dry_run=False)

    assert calls == []
    assert link.is_symlink() and (outside / "keep.txt").read_text() == "someone else's data"
    assert report["removed_count"] == 0 and report["candidate_count"] == 0
    assert report["skipped"] == [{"path": str(link), "reason": "symlink"}]


def test_gc_leaves_a_plain_file_named_as_a_day_823(tmp_path):
    """A plain FILE called ``20260101`` used to disappear through
    ``entry.is_dir()`` without a trace, so the gc could not tell "nothing to do"
    from "something with a day's name is here and is not a day directory". It is
    now a disclosed refusal, and the file is untouched."""
    from datetime import date

    import bn.output as output

    loose = tmp_path / "20260101"
    loose.write_text("someone else's file")

    report = output.gc_spills(root=tmp_path, today=date(2026, 9, 15))

    assert report["skipped"] == [{"path": str(loose), "reason": "not a directory"}]
    assert report["candidate_count"] == 0 and report["removed_count"] == 0
    assert report["kept_count"] == 0 and report["total_bytes"] == 0
    assert loose.read_text() == "someone else's file"


def test_gc_removes_the_stale_day_and_reports_reclaimed_bytes_823(tmp_path, monkeypatch):
    """The other half of the hardening: the real stale day directory still goes,
    and the summary states what it freed instead of a bare count -- candidates
    vs removed, bytes reclaimed, and what was kept."""
    from datetime import date, timedelta

    import bn.output as output

    monkeypatch.delenv("BN_SPILL_RETENTION_DAYS", raising=False)
    today = date(2026, 9, 15)
    stale = _make_day(tmp_path, today - timedelta(days=30))
    fresh = _make_day(tmp_path, today - timedelta(days=1))

    report = output.gc_spills(root=tmp_path, today=today)

    assert not stale.exists() and fresh.exists()
    assert report["kind"] == "spill_gc" and report["dry_run"] is False
    assert report["root"] == str(tmp_path) and report["older_than_days"] == 14
    assert report["removed_count"] == 1 and report["candidate_count"] == 1
    assert report["reclaimed_bytes"] == _tree_bytes(fresh)
    assert report["candidate_bytes"] == report["reclaimed_bytes"]
    assert [row["day"] for row in report["removed"]] == [stale.name]
    assert report["removed"][0]["files"] == 1
    assert report["kept_count"] == 1 and report["kept_bytes"] == _tree_bytes(fresh)
    # `total_bytes` is the eligible days as INSPECTED, so it is the two numbers
    # that partition them: what was reclaimed plus what stayed.
    assert report["total_bytes"] == report["candidate_bytes"] + report["kept_bytes"]


def test_gc_dry_run_reports_candidates_without_removing_823(tmp_path):
    """``--dry-run`` is the inspect half of the request: the whole report --
    which days, how many bytes, what stays -- with nothing removed and
    ``reclaimed_bytes`` left at 0, because nothing was reclaimed."""
    from datetime import date, timedelta

    import bn.output as output

    today = date(2026, 9, 15)
    stale = _make_day(tmp_path, today - timedelta(days=30))
    fresh = _make_day(tmp_path, today - timedelta(days=1))

    report = output.gc_spills(root=tmp_path, today=today, dry_run=True)

    assert report["dry_run"] is True
    assert stale.exists() and fresh.exists()
    assert report["candidate_count"] == 1 and report["removed_count"] == 0
    assert report["removed"] == [] and report["reclaimed_bytes"] == 0
    assert report["candidate_bytes"] == _tree_bytes(stale)
    assert report["candidates"][0]["path"] == str(stale)
    assert report["kept_count"] == 1 and report["kept_bytes"] == _tree_bytes(fresh)


def test_gc_max_bytes_evicts_the_oldest_days_beyond_the_cap_823(tmp_path, monkeypatch):
    """``--max-bytes`` is a second bound, not a replacement for the window: on
    its own the age pass decides, and with a cap the OLDEST surviving days join
    the candidates until the day directories fit -- which is what lets an
    engagement with a wide retention window still hold the root to a budget."""
    from datetime import date, timedelta

    import bn.output as output

    today = date(2026, 9, 15)
    stale = _make_day(tmp_path, today - timedelta(days=30))
    (stale / "big.bin").write_bytes(b"x" * 400)
    older_fresh = _make_day(tmp_path, today - timedelta(days=2))
    (older_fresh / "big.bin").write_bytes(b"x" * 400)
    fresh = _make_day(tmp_path, today - timedelta(days=1))
    (fresh / "big.bin").write_bytes(b"x" * 400)
    cap = _tree_bytes(fresh) + 1

    without_cap = output.gc_spills(root=tmp_path, today=today, dry_run=True)
    assert [row["day"] for row in without_cap["candidates"]] == [stale.name]

    report = output.gc_spills(root=tmp_path, today=today, max_bytes=cap, dry_run=True)

    # Oldest first, and the cap is what pulled the still-inside-the-window day in.
    assert [row["day"] for row in report["candidates"]] == [stale.name, older_fresh.name]
    assert report["kept_count"] == 1 and report["kept_bytes"] == _tree_bytes(fresh)
    assert report["candidate_bytes"] == _tree_bytes(stale) + _tree_bytes(older_fresh)
    assert report["max_bytes"] == cap and stale.exists() and older_fresh.exists()


def test_gc_reports_a_failed_removal_instead_of_counting_it_823(tmp_path, monkeypatch):
    """Candidates and removed are separate numbers because they differ exactly
    when a removal FAILED: a day that survived a failed removal must not be
    reported as reclaimed space, nor silently folded into "kept"."""
    from datetime import date, timedelta

    import bn.output as output

    today = date(2026, 9, 15)
    stale = _make_day(tmp_path, today - timedelta(days=30))
    _no_rmtree(monkeypatch, output, error=OSError("resource busy"))

    report = output.gc_spills(root=tmp_path, today=today)

    assert stale.exists()
    assert report["candidate_count"] == 1 and report["removed_count"] == 0
    assert report["reclaimed_bytes"] == 0 and report["removed"] == []
    assert report["errors"] == [{"path": str(stale), "error": "resource busy"}]
    assert report["kept_count"] == 0


def test_gc_window_defaults_to_the_env_retention_and_the_flag_overrides_it_823(
    tmp_path, monkeypatch
):
    """The default window has to be the one the write path prunes with, or a bare
    ``bn spill gc`` and the next spill would disagree about what is stale. The
    flag overrides it in both directions, including over
    ``BN_SPILL_RETENTION_DAYS=0`` (keep-everything), which is the documented way
    to pin a cache -- an explicit request to reclaim must still be able to."""
    from datetime import date, timedelta

    import bn.output as output

    today = date(2026, 9, 15)
    five = _make_day(tmp_path, today - timedelta(days=5))
    two = _make_day(tmp_path, today - timedelta(days=2))

    monkeypatch.setenv("BN_SPILL_RETENTION_DAYS", "3")
    windowed = output.gc_spills(root=tmp_path, today=today, dry_run=True)
    assert windowed["older_than_days"] == 3
    assert [row["day"] for row in windowed["candidates"]] == [five.name]

    monkeypatch.setenv("BN_SPILL_RETENTION_DAYS", "0")
    pinned = output.gc_spills(root=tmp_path, today=today, dry_run=True)
    assert pinned["older_than_days"] == 0 and pinned["candidate_count"] == 0
    widened = output.gc_spills(root=tmp_path, today=today, older_than_days=1, dry_run=True)
    assert widened["older_than_days"] == 1
    assert [row["day"] for row in widened["candidates"]] == [five.name, two.name]


def test_render_spill_gc_text_states_an_unreadable_counter_as_unknown_823():
    """An unreadable counter must print `?` in the BODY, not a real number: a
    `0` there is the #683 fabricated-zero harm wearing a footnote that arrives
    after the line a caller acts on.

    Named coverage for the four `_render_spill_gc_text` pairs the count
    differential in `tests/test_cli_formatters.py` lists as not-stated: its
    probe payload carries no `dry_run` flag (so the branch that states
    `removed_count`/`reclaimed_bytes` never opens) and no candidate ROW (so
    `bytes`/`files`, read one level down, are never reached). The real shapes
    are driven here instead.
    """
    from bn.formatters import _render_spill_gc_text

    row = {"day": "20260101", "path": "/cache/spills/20260101",
           "bytes": "many", "files": "many"}

    sweep = _render_spill_gc_text({
        "kind": "spill_gc", "dry_run": False, "candidates": [row],
        "candidate_count": 1, "candidate_bytes": 10,
        "removed_count": "many", "reclaimed_bytes": "many", "kept_count": 0,
    })
    assert "spill gc: reclaimed ? of 1 candidate day(s) (? bytes), 0 kept" in sweep
    assert "  20260101  ? bytes  ? files" in sweep
    assert " 0 bytes" not in sweep and "reclaimed 0 of" not in sweep

    dry = _render_spill_gc_text({
        "kind": "spill_gc", "dry_run": True, "candidates": [row],
        "candidate_count": "many", "candidate_bytes": "many", "kept_count": 0,
    })
    assert "spill gc: dry run, ? day(s) would be reclaimed (? bytes), 0 kept" in dry

    # ...and the flag that separates "would be" from "was": `"maybe"` is the
    # shape a raw truthiness test reads as a real yes.
    unknown = _render_spill_gc_text({
        "kind": "spill_gc", "dry_run": "maybe",
        "candidate_count": 2, "candidate_bytes": 10, "kept_count": 1,
    })
    assert unknown.startswith("spill gc: ? dry run unknown -- ")


def test_gc_max_bytes_spares_the_current_day_823(tmp_path):
    """A cap is a request to shrink the cache, never to delete the directory
    the spill writer is still appending to: the size pass must stop at TODAY,
    or `--max-bytes 0` reaps the live day -- the same footgun `--older-than 0`
    is refused at parse time to prevent.

    The consequence of the refusal is stated in the report rather than implied:
    a cap below the live day's own bytes leaves `kept_bytes > max_bytes`.
    """
    from datetime import date, timedelta

    import bn.output as output

    today = date(2026, 9, 15)
    live = _make_day(tmp_path, today)
    (live / "big.bin").write_bytes(b"x" * 400)
    old = _make_day(tmp_path, today - timedelta(days=30))

    report = output.gc_spills(root=tmp_path, today=today, max_bytes=0)

    assert [row["day"] for row in report["candidates"]] == [old.name]
    assert live.exists() and not old.exists()
    assert report["kept_count"] == 1 and report["kept_bytes"] == _tree_bytes(live)
    assert report["kept_bytes"] > report["max_bytes"] == 0

    # Same root, dry run: the live day is not a candidate even with nothing left
    # but a cap of zero to satisfy.
    dry = output.gc_spills(root=tmp_path, today=today, max_bytes=0, dry_run=True)
    assert [row["day"] for row in dry["candidates"]] == []
    assert dry["kept_bytes"] == _tree_bytes(live)


def test_gc_discloses_a_non_canonical_day_name_823(tmp_path):
    """`202611` parses as 2026-01-01 under `%Y%m%d` and is not a name the writer
    (which always calls `strftime`) can produce. It was already refused, but
    SILENTLY -- so the report could not distinguish "nothing here" from "here is
    a day-shaped name this command will not touch", while a symlink and a plain
    file each got a reason.

    A name that is not day-shaped at all stays silent on purpose: `notes` makes
    no claim about a spill day, which is the line #618 draws.
    """
    from datetime import date

    import bn.output as output

    odd = tmp_path / "202611"
    odd.mkdir()
    (odd / "keep.txt").write_text("someone else's data")
    unrelated = tmp_path / "notes"
    unrelated.mkdir()

    report = output.gc_spills(root=tmp_path, today=date(2026, 9, 15))

    assert report["skipped"] == [{"path": str(odd), "reason": "non-canonical day name"}]
    assert report["candidate_count"] == 0 and report["removed_count"] == 0
    assert (odd / "keep.txt").read_text() == "someone else's data"
    assert unrelated.exists()
