from __future__ import annotations

import json
import tempfile

from bn.output import write_output


def test_write_output_renders_small_payload_without_spill(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))

    rendered = write_output({"ok": True}, fmt="json", out_path=None, stem="small")

    payload = json.loads(rendered)
    assert payload["ok"] is True


def test_write_output_spills_large_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))

    rendered = write_output(
        {"data": "x" * 100_000},
        fmt="json",
        out_path=None,
        stem="large",
        spill_threshold=256,
    )

    envelope = json.loads(rendered)
    artifact_root = tempfile.gettempdir()
    assert envelope["artifact_path"].startswith(artifact_root)
