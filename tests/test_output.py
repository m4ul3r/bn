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

    envelope = _parse_envelope(rendered)
    # Spills live under the (BN_CACHE_DIR-overridable) cache root, not /tmp.
    assert envelope["path"].startswith(str(tmp_path / "spills"))
    assert envelope["spilled"] == "true"
    artifact_text = Path(envelope["path"]).read_text()
    assert envelope["tokenizer"] == "estimate"
    assert int(envelope["tokens"]) == _token_count(artifact_text)
    # Filename carries pid + random component so parallel agents spilling in
    # the same second can't clobber each other.
    name = Path(envelope["path"]).name
    assert re.fullmatch(rf"large-\d{{6}}-{os.getpid()}-[0-9a-f]{{4}}\.json", name)


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
        paths.add(_parse_envelope(rendered)["path"])

    assert len(paths) == 3


def test_write_output_falls_back_to_full_output_when_spill_write_fails(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    payload = {"data": [f"item-{index:04d}" for index in range(1000)]}
    expected = json.dumps(payload, indent=2, sort_keys=True) + "\n"

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

    envelope = _parse_envelope(rendered)
    artifact_text = out_path.read_text()
    assert envelope["path"] == str(out_path)
    assert envelope["spilled"] == "false"
    assert envelope["tokenizer"] == "estimate"
    assert int(envelope["tokens"]) == _token_count(artifact_text)
