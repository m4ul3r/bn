from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tiktoken

from .paths import spill_root


DEFAULT_SPILL_TOKEN_LIMIT = 10_000
# `tiktoken` does not currently resolve the dotted `gpt-5.4` alias directly.
GPT_5_4_TOKENIZER = "o200k_base"


@dataclass(frozen=True)
class OutputWriteResult:
    rendered: str
    artifact: dict[str, Any] | None = None
    spilled: bool = False


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def render_value(value: Any, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n"

    if fmt == "ndjson":
        if isinstance(value, list):
            lines = [
                json.dumps(item, sort_keys=True, default=_json_default) for item in value
            ]
            return "\n".join(lines) + ("\n" if lines else "")
        return json.dumps(value, sort_keys=True, default=_json_default) + "\n"

    if isinstance(value, str):
        return value if value.endswith("\n") else value + "\n"
    return json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n"


def _summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"kind": "object", "keys": sorted(value.keys())[:10], "count": len(value)}
    if isinstance(value, list):
        return {"kind": "array", "count": len(value)}
    if isinstance(value, str):
        return {"kind": "string", "chars": len(value)}
    return {"kind": type(value).__name__}


def _spill_path(stem: str, suffix: str) -> Path:
    now = datetime.now(timezone.utc)
    directory = spill_root() / now.strftime("%Y%m%d")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{stem}-{now.strftime('%H%M%S')}{suffix}"


@functools.cache
def _token_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(GPT_5_4_TOKENIZER)


def _artifact_payload(
    *,
    artifact_path: Path,
    fmt: str,
    encoded: bytes,
    token_count: int,
    value: Any,
    spilled: bool,
) -> dict[str, Any]:
    return {
        "ok": True,
        "spilled": spilled,
        "artifact_path": str(artifact_path),
        "format": fmt,
        "bytes": len(encoded),
        "tokens": token_count,
        "tokenizer": GPT_5_4_TOKENIZER,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "summary": _summary(value),
    }


def _format_envelope_value(value: Any) -> str:
    if isinstance(value, list | tuple):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)
    return str(value)


def render_artifact_envelope(payload: dict[str, Any]) -> str:
    lines = []
    if "ok" in payload:
        lines.append(f"ok: {str(bool(payload.get('ok'))).lower()}")
    if "spilled" in payload:
        lines.append(f"spilled: {str(bool(payload.get('spilled'))).lower()}")
    if "artifact_path" in payload:
        lines.append(f"path: {payload['artifact_path']}")
    for key in ("format", "bytes", "tokens", "tokenizer", "sha256"):
        if key in payload:
            lines.append(f"{key}: {payload[key]}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        summary_parts = []
        kind = summary.get("kind")
        if kind is not None:
            summary_parts.append(f"kind={_format_envelope_value(kind)}")
        for key in sorted(summary):
            if key == "kind":
                continue
            summary_parts.append(f"{key}={_format_envelope_value(summary[key])}")
        if summary_parts:
            lines.append(f"summary: {' '.join(summary_parts)}")
    return "\n".join(lines) + "\n"


def write_output_result(
    value: Any,
    *,
    fmt: str,
    out_path: Path | None,
    stem: str,
    spill_token_limit: int = DEFAULT_SPILL_TOKEN_LIMIT,
) -> OutputWriteResult:
    rendered = render_value(value, fmt)
    encoded = rendered.encode("utf-8")
    token_count = len(_token_encoding().encode(rendered))

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(encoded)
        artifact = _artifact_payload(
            artifact_path=out_path,
            fmt=fmt,
            encoded=encoded,
            token_count=token_count,
            value=value,
            spilled=False,
        )
        return OutputWriteResult(
            rendered=render_artifact_envelope(artifact),
            artifact=artifact,
            spilled=False,
        )

    if token_count <= spill_token_limit:
        return OutputWriteResult(rendered=rendered)

    suffix = ".ndjson" if fmt == "ndjson" else ".txt" if fmt == "text" else ".json"
    spill_path = _spill_path(stem, suffix)
    spill_path.write_bytes(encoded)
    artifact = _artifact_payload(
        artifact_path=spill_path,
        fmt=fmt,
        encoded=encoded,
        token_count=token_count,
        value=value,
        spilled=True,
    )
    return OutputWriteResult(
        rendered=render_artifact_envelope(artifact),
        artifact=artifact,
        spilled=True,
    )


def write_output(
    value: Any,
    *,
    fmt: str,
    out_path: Path | None,
    stem: str,
    spill_token_limit: int = DEFAULT_SPILL_TOKEN_LIMIT,
) -> str:
    return write_output_result(
        value,
        fmt=fmt,
        out_path=out_path,
        stem=stem,
        spill_token_limit=spill_token_limit,
    ).rendered
