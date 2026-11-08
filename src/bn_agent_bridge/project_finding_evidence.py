"""Read-only, declarative project evidence; observations are not claim proofs.

Only persisted ProjectFile blobs are read. Semantic checks additionally require
an already-loaded, clean project database; no view loading, analysis requests,
commands, or user-supplied Python are performed here. A database digest covers
saved settings, models and types conservatively, not just rendered IL.

Project ownership excludes concurrent database writers. If a database is
replaced behind an open native view, reload that view before capturing new
semantic evidence: the SDK cannot prove the old view matches that replacement.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from typing import Any
import sys

from .version import VERSION

_MAX_CHECKS = 128
_MAX_BYTES = 65536
_MAX_FUNCTION_BYTES = 1024 * 1024
_MAX_SEMANTIC_BYTES = 1024 * 1024
_MAX_INSTRUCTIONS = 16384
_MAX_BINDINGS = 16384
_SQLITE_MAGIC = b"SQLite format 3\x00"


class _Unavailable(RuntimeError):
    """An observation cannot establish currency, rather than a bad request."""


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _text(value: Any, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()) or len(value) > 16384:
        raise RuntimeError(f"{label} must be a {'possibly empty' if empty else 'nonempty'} string of at most 16384 characters")
    return value


def _integer(value: Any, label: str, minimum: int, maximum: int = (1 << 64) - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise RuntimeError(f"{label} must be an integer between {minimum} and {maximum}")
    return value


def _request(item: Any) -> dict:
    if not isinstance(item, dict):
        raise RuntimeError("Each evidence check must be an object")
    kind = item.get("check")
    fields = {
        "file": {"file", "expected", "description"},
        "bytes": {"file", "offset", "length", "expected", "description"},
        "function": {"file", "function", "expected", "description"},
        "links": {"file", "description"},
        "note": {"text"},
    }
    if not isinstance(kind, str) or kind not in fields:
        raise RuntimeError("Evidence check must be file, bytes, function, links, or note")
    if set(item) - fields[kind] - {"check"}:
        raise RuntimeError(f"Unknown {kind} evidence fields: {sorted(set(item) - fields[kind] - {'check'})}")
    result = {"check": kind}
    if kind == "note":
        result["text"] = _text(item.get("text"), "note text")
        return result
    result["file"] = _text(item.get("file"), "project file selector")
    if "description" in item:
        result["description"] = _text(item["description"], "description")
    if kind == "bytes":
        result["offset"] = _integer(item.get("offset"), "offset", 0)
        result["length"] = _integer(item.get("length"), "length", 1, _MAX_BYTES)
        if result["offset"] + result["length"] > (1 << 64) - 1:
            raise RuntimeError("Evidence byte range overflows a 64-bit file offset")
    if kind == "function":
        address = item.get("function")
        if isinstance(address, str) and re.fullmatch(r"0[xX][0-9a-fA-F]{1,16}", address):
            address = int(address, 16)
        result["function"] = _integer(address, "function address", 0)
    if "expected" in item:
        expected = item["expected"]
        allowed = {"file": {"sha256"}, "bytes": {"hex"}, "function": {"name", "prototype"}}[kind]
        if not isinstance(expected, dict) or not expected or set(expected) - allowed:
            raise RuntimeError(f"expected must be a nonempty object containing only {sorted(allowed)}")
        normalized = {}
        for key, value in expected.items():
            if key in {"sha256", "hex"}:
                if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]+", value):
                    raise RuntimeError(f"expected {key} must be hexadecimal")
                required = 64 if key == "sha256" else result["length"] * 2
                if len(value) != required:
                    raise RuntimeError(f"expected {key} must contain exactly {required} hexadecimal characters")
                normalized[key] = value.lower()
            else:
                normalized[key] = _text(value, f"expected {key}")
        result["expected"] = normalized
    return result


def _stamp(value: os.stat_result) -> tuple:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _blob(pf: Any, *, offset: int | None = None, length: int | None = None) -> dict:
    """Hash with bounded memory and reject replacement/truncation while reading."""
    path = pf.path_on_disk
    with open(path, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise _Unavailable("Project blob is not a regular persisted file")
        digest = hashlib.sha256()
        if offset is not None:
            stream.seek(offset)
            data = stream.read(length)
            if len(data) != length:
                raise _Unavailable("Persisted file byte range is a short read")
            digest.update(data)
            result = {"hex": data.hex(), "sha256": digest.hexdigest()}
        else:
            count = 0
            while chunk := stream.read(1024 * 1024):
                count += len(chunk)
                digest.update(chunk)
            if count != before.st_size:
                raise _Unavailable("Project blob changed size during observation")
            result = {"sha256": digest.hexdigest(), "size": count}
        if _stamp(before) != _stamp(os.fstat(stream.fileno())) or _stamp(before) != _stamp(os.stat(path)):
            raise _Unavailable("Project blob changed during observation")
    return result


def _exact_file(bridge: Any, file_id: str) -> Any | None:
    matches = [pf for pf in bridge.projects.files() if str(pf.id) == file_id]
    if len(matches) > 1:
        raise _Unavailable("Project contains duplicate exact file IDs")
    return matches[0] if matches else None


def _expected_reason(record: dict, observed: dict | None) -> str | None:
    for key, wanted in record.get("expected", {}).items():
        if observed is not None and key in observed and observed[key] != wanted:
            return f"Expected {key} does not match the observed value"
    return None


def _shape(value: Any, fields: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise RuntimeError(f"{label} must contain exactly {sorted(fields)}")


def _sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"{label} must be a canonical SHA-256 digest")


def _blob_schema(value: Any) -> None:
    _shape(value, {"sha256", "size"}, "Captured blob observation")
    _sha256(value["sha256"], "Captured blob sha256")
    _integer(value["size"], "Captured blob size", 0)


def _library_schema(value: Any) -> None:
    _shape(value, {"name", "provider_file_id", "resolution"}, "Captured library")
    _text(value["name"], "Captured library name")
    if value["resolution"] == "bound":
        _text(value["provider_file_id"], "Captured provider file ID")
    elif value["resolution"] != "unresolved" or value["provider_file_id"] is not None:
        raise RuntimeError("Captured library resolution and provider ID disagree")


def _observed_schema(record: dict) -> None:
    """Require every fact used by a complete machine-checkable baseline."""
    kind, observed = record["check"], record["observed"]
    if kind == "file":
        _blob_schema(observed)
        return
    if kind == "bytes":
        _shape(observed, {"hex", "sha256"}, "Captured byte observation")
        value = observed["hex"]
        if (not isinstance(value, str) or len(value) != 2 * record["length"]
                or re.fullmatch(r"[0-9a-f]+", value) is None):
            raise RuntimeError("Captured byte hex must match the requested length")
        _sha256(observed["sha256"], "Captured byte sha256")
        if hashlib.sha256(bytes.fromhex(value)).hexdigest() != observed["sha256"]:
            raise RuntimeError("Captured byte hex and sha256 disagree")
        return
    fields = {"source", "tool_versions"}
    if kind == "function":
        fields |= {"function", "name", "prototype", "ranges", "hlil_sha256",
                   "hlil_expressions", "hlil_instructions"}
    else:
        fields |= {"libraries", "locations", "providers", "limitation"}
    _shape(observed, fields, f"Captured {kind} observation")
    _blob_schema(observed["source"])
    versions = observed["tool_versions"]
    _shape(versions, {"core", "bridge", "engine_build_id"}, "Captured tool versions")
    for key, value in versions.items():
        _text(value, f"Captured {key} version")
    if kind == "function":
        _integer(observed["function"], "Captured function address", 0)
        if observed["function"] != record["function"]:
            raise RuntimeError("Captured function address does not match its request")
        for key in ("name", "prototype"):
            _text(observed[key], f"Captured function {key}")
        _sha256(observed["hlil_sha256"], "Captured HLIL sha256")
        for key in ("hlil_expressions", "hlil_instructions"):
            _integer(observed[key], f"Captured {key}", 1, _MAX_INSTRUCTIONS)
        ranges = observed["ranges"]
        if not isinstance(ranges, list) or not 1 <= len(ranges) <= 4096:
            raise RuntimeError("Captured function ranges must be a nonempty bounded list")
        total = 0
        for region in ranges:
            _shape(region, {"start", "length", "sha256"}, "Captured function range")
            start = _integer(region["start"], "Captured range start", 0)
            length = _integer(region["length"], "Captured range length", 1, _MAX_FUNCTION_BYTES)
            if start + length > 1 << 64:
                raise RuntimeError("Captured function range overflows its address space")
            total += length
            _sha256(region["sha256"], "Captured range sha256")
        if total > _MAX_FUNCTION_BYTES or ranges != sorted(ranges, key=lambda row: row["start"]):
            raise RuntimeError("Captured function ranges are oversized or not canonical")
        return
    libraries, locations, providers = (observed[key] for key in ("libraries", "locations", "providers"))
    if (not isinstance(libraries, list) or len(libraries) > _MAX_BINDINGS
            or not isinstance(locations, list) or len(locations) > _MAX_BINDINGS
            or not isinstance(providers, dict) or len(providers) > _MAX_BINDINGS):
        raise RuntimeError("Captured external bindings must be bounded collections")
    by_name = {}
    provider_ids = set()
    for library in libraries:
        _library_schema(library)
        if library["name"] in by_name:
            raise RuntimeError("Captured external library names must be unique")
        by_name[library["name"]] = library
        if library["provider_file_id"] is not None:
            provider_ids.add(library["provider_file_id"])
    for location in locations:
        _shape(location, {"source_name", "source_address", "source_type",
                          "target_address", "target_symbol", "library"}, "Captured external location")
        for key in ("source_name", "source_type"):
            _text(location[key], f"Captured location {key}")
        _integer(location["source_address"], "Captured source address", 0)
        if location["target_address"] is not None:
            _integer(location["target_address"], "Captured target address", 0)
        if location["target_symbol"] is not None:
            _text(location["target_symbol"], "Captured target symbol")
        library = location["library"]
        if library is not None:
            _library_schema(library)
            if by_name.get(library["name"]) != library:
                raise RuntimeError("Captured location library has no matching library fact")
    if set(providers) != provider_ids:
        raise RuntimeError("Captured provider fingerprints do not match bound libraries")
    for file_id, value in providers.items():
        _text(file_id, "Captured provider file ID")
        _blob_schema(value)
    _text(observed["limitation"], "Captured binding limitation")
    if libraries != sorted(libraries, key=_json) or locations != sorted(locations, key=_json):
        raise RuntimeError("Captured external binding facts are not canonical")


def validate_captured_evidence(captured: list[dict]) -> None:
    """Refuse malformed persisted baselines before storage or live rechecks."""
    if not isinstance(captured, list) or not 1 <= len(captured) <= _MAX_CHECKS:
        raise RuntimeError(f"Captured evidence must contain between 1 and {_MAX_CHECKS} checks")
    metadata = {"file_id", "file_path", "observed", "fingerprint", "capture_state", "reason"}
    for record in captured:
        if not isinstance(record, dict):
            raise RuntimeError("Each captured evidence check must be an object")
        if not {"observed", "fingerprint", "capture_state"} <= set(record):
            raise RuntimeError("Captured evidence is missing baseline fields")
        request = {key: value for key, value in record.items() if key not in metadata}
        kind = record.get("check")
        if kind != "note":
            if "file" in request:
                raise RuntimeError("Captured evidence must use an exact file ID, not a selector")
            request["file"] = _text(record.get("file_id"), "Captured file ID")
            path = _text(record.get("file_path"), "Captured file path")
            if not path.startswith("/"):
                raise RuntimeError("Captured evidence must use a canonical project path")
        elif "file_id" in record or "file_path" in record:
            raise RuntimeError("Captured notes cannot contain file identities")
        if _request(request) != request:
            raise RuntimeError("Captured evidence request is not canonical")
        if "reason" in record:
            _text(record["reason"], "Captured evidence reason")
        state, observed, fingerprint = (record[key] for key in ("capture_state", "observed", "fingerprint"))
        if not isinstance(state, str) or state not in {"current", "stale", "unknown"}:
            raise RuntimeError("Captured evidence has an invalid capture_state")
        if state == "unknown":
            if fingerprint is not None:
                raise RuntimeError("Unknown captured evidence cannot have a complete fingerprint")
            if observed is not None:
                if kind not in {"function", "links"}:
                    raise RuntimeError("Unknown evidence cannot have a complete observation")
                _shape(observed, {"source"}, "Partial semantic observation")
                _blob_schema(observed["source"])
            continue
        if kind == "note":
            raise RuntimeError("Manual notes must remain unknown without a baseline")
        _observed_schema(record)
        _sha256(fingerprint, "Captured evidence fingerprint")
        if fingerprint != _digest(observed):
            raise RuntimeError("Captured evidence fingerprint does not match its observation")
        mismatch = _expected_reason(record, observed)
        if (state == "stale") != (mismatch is not None):
            raise RuntimeError("Captured evidence capture_state disagrees with its expected observation")


def capture_evidence(bridge: Any, evidence: list[dict]) -> list[dict]:
    """Validate declarative input and capture immutable source/analysis baselines."""
    bridge.projects.require()
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= _MAX_CHECKS:
        raise RuntimeError(f"evidence must contain between 1 and {_MAX_CHECKS} checks")
    # Validate every request and selector before observing any native state.
    pending = []
    for item in evidence:
        record = _request(item)
        pf = None
        if record["check"] != "note":
            pf = bridge.projects.resolve_file(record.pop("file"))
            record.update(file_id=str(pf.id), file_path=bridge.projects.path_in_project(pf))
        pending.append((record, pf))
    captured = []
    for record, pf in pending:
        observed, state, reason = _observe(bridge, record, pf)
        mismatch = _expected_reason(record, observed)
        if mismatch:
            state, reason = "stale", mismatch
        record.update(observed=observed, fingerprint=_digest(observed) if state != "unknown" else None,
                      capture_state=state)
        if reason:
            record["reason"] = reason
        captured.append(record)
    return captured


def _versions() -> dict:
    import binaryninja as bn

    module = sys.modules.get(f"{__package__}.bridge")
    engine = getattr(module, "ENGINE_BUILD_ID", None)
    core = bn.core_version()
    if not engine or not isinstance(core, str) or not core or VERSION == "0+unknown":
        raise _Unavailable("Core/bridge tool version identity is unavailable")
    return {"core": core, "bridge": VERSION, "engine_build_id": engine}


def _clean_view(bridge: Any, pf: Any) -> Any:
    bv = bridge.targets.view_for_project_file(str(pf.id))
    if bv is None:
        raise _Unavailable("Exact project file has no already-loaded analysis view")
    binding = bridge.targets.project_binding(bv)
    if binding is not None:
        identity = binding[0]
    else:
        native_file = bv.project_file
        identity = str(native_file.id) if native_file is not None else None
        if native_file is not None and not os.path.samefile(native_file.path_on_disk, pf.path_on_disk):
            raise _Unavailable("Loaded view is backed by a different project blob")
    if identity != str(pf.id):
        raise _Unavailable("Loaded view does not match the exact captured ProjectFile ID")
    if bv.file.has_database is not True:
        raise _Unavailable("Semantic evidence requires a durable project BNDB")
    if (bv.file.modified is not False or bv.file.analysis_changed is not False
            or bridge.targets.is_dirty(bv) is not False):
        raise _Unavailable("Analysis has unsaved changes; save its project BNDB before capture")
    database_path = bridge.targets.recorded_database(bv) or bv.file.filename
    if not database_path or not os.path.samefile(database_path, pf.path_on_disk):
        raise _Unavailable("Loaded analysis is not backed by this exact persisted BNDB")
    with open(pf.path_on_disk, "rb") as stream:
        if stream.read(len(_SQLITE_MAGIC)) != _SQLITE_MAGIC:
            raise _Unavailable("Semantic evidence requires a persisted BNDB, not a raw artifact")
    from .bridge_state import _quick_loaded_views, _unanalyzed_views

    if bv in _quick_loaded_views or bv in _unanalyzed_views or bv.analysis_state.name != "IdleState":
        raise _Unavailable("Analysis is incomplete or still running")
    return bv


def _function_observation(bv: Any, address: int) -> dict:
    function = bv.get_function_at(address)
    if function is None:
        raise _Unavailable("No function exists at the requested start address")
    name = _text(function.name, "observed function name")
    if function.type is None:
        raise _Unavailable("Function prototype is unavailable")
    prototype = _text(str(function.type), "observed function prototype")
    ranges = []
    total = 0
    for region in function.address_ranges:
        start, end = int(region.start), int(region.end)
        if start < 0 or end <= start or len(ranges) >= 4096:
            raise _Unavailable("Function byte ranges are invalid or exceed the observation bound")
        total += end - start
        if total > _MAX_FUNCTION_BYTES:
            raise _Unavailable("Function bytes exceed the observation bound")
        data = bv.read(start, end - start)
        if data is None or len(data) != end - start:
            raise _Unavailable("Function byte range is a short read")
        ranges.append({"start": start, "length": end - start,
                       "sha256": hashlib.sha256(data).hexdigest()})
    if not ranges:
        raise _Unavailable("Function has no complete byte ranges")
    hlil = function.hlil_if_available
    if hlil is None:
        raise _Unavailable("HLIL is unavailable; evidence never generates analysis")
    semantic = _hlil_digest(hlil)
    return {"function": int(function.start), "name": name, "prototype": prototype,
            "ranges": sorted(ranges, key=lambda row: row["start"]), **semantic}


def _hlil_digest(hlil: Any) -> dict:
    count = hlil.get_expr_count()
    instructions = len(hlil)
    if (type(count) is not int or not 0 < count <= _MAX_INSTRUCTIONS
            or type(instructions) is not int or not 0 < instructions <= _MAX_INSTRUCTIONS):
        raise _Unavailable("HLIL is empty, incomplete, or exceeds the expression bound")
    digest = hashlib.sha256()
    size = 0
    for index in range(count):
        expression = hlil.get_expr(index, as_ast=False)
        if expression is None:
            raise _Unavailable("HLIL contains an unavailable expression")
        # Non-AST rendering avoids recursively rendering an entire function for
        # each block. The raw encoding retains expression references; rendered
        # text additionally covers decoded operand-list values and variables.
        row = {"index": index, "operation": expression.operation.name,
               "address": int(expression.address), "size": int(expression.size),
               "operands": [int(value) for value in expression.core_operands],
               "text": str(expression)}
        encoded = _json(row)
        size += len(encoded)
        if size > _MAX_SEMANTIC_BYTES:
            raise _Unavailable("HLIL semantic observation exceeds the byte bound")
        digest.update(encoded)
        digest.update(b"\n")
    if hlil.get_expr_count() != count or len(hlil) != instructions:
        raise _Unavailable("HLIL changed during observation")
    return {"hlil_sha256": digest.hexdigest(), "hlil_expressions": count,
            "hlil_instructions": instructions}


def _links_observation(bridge: Any, bv: Any) -> dict:
    libraries = bv.get_external_libraries()
    locations = bv.get_external_locations()
    if len(libraries) > _MAX_BINDINGS or len(locations) > _MAX_BINDINGS:
        raise _Unavailable("External binding facts exceed the observation bound")
    providers = {}
    library_rows = {}

    def library_fact(library: Any) -> dict:
        name = _text(library.name, "external library name")
        backing = library.backing_file
        row = {"name": name, "provider_file_id": None, "resolution": "unresolved"}
        if backing is not None:
            file_id = str(backing.id)
            pf = _exact_file(bridge, file_id)
            if pf is None or not os.path.samefile(backing.path_on_disk, pf.path_on_disk):
                raise _Unavailable(f"External library {name} has an unresolved backing provider ID {file_id}")
            row.update(provider_file_id=file_id, resolution="bound")
            if file_id not in providers:
                providers[file_id] = _blob(pf)
        if name in library_rows and library_rows[name] != row:
            raise _Unavailable("External library name has inconsistent provider bindings")
        library_rows[name] = row
        return row

    for library in libraries:
        library_fact(library)
    location_rows = []
    for location in locations:
        source = location.source_symbol
        if source is None:
            raise _Unavailable("External location has no source symbol")
        library = location.library
        binding = library_fact(library) if library is not None else None
        target_address = location.target_address
        target_symbol = location.target_symbol
        if target_address is not None:
            _integer(target_address, "external target address", 0)
        if target_symbol is not None:
            _text(target_symbol, "external target symbol")
        location_rows.append({
            "source_name": _text(source.raw_name, "external source name"),
            "source_address": _integer(source.address, "external source address", 0),
            "source_type": source.type.name,
            "target_address": target_address, "target_symbol": target_symbol,
            "library": binding,
        })
    # Provider files may change independently of the consumer database.
    for file_id, baseline in providers.items():
        pf = _exact_file(bridge, file_id)
        if pf is None or _blob(pf) != baseline:
            raise _Unavailable("External provider changed during observation")
    facts = {"libraries": sorted(library_rows.values(), key=_json),
             "locations": sorted(location_rows, key=_json),
             "providers": providers,
             "limitation": "Binding facts only; not dynamic-loader correctness or transitive dataflow"}
    if len(_json(facts)) > _MAX_SEMANTIC_BYTES:
        raise _Unavailable("External binding facts exceed the byte bound")
    return facts


def _observe(bridge: Any, record: dict, pf: Any) -> tuple[dict | None, str, str | None]:
    kind = record["check"]
    if kind == "note":
        return None, "unknown", "Manual note is uncheckable and does not prove an interpretation"
    observed = None
    try:
        if kind == "file":
            return _blob(pf), "current", None
        if kind == "bytes":
            offset = _integer(record.get("offset"), "offset", 0)
            length = _integer(record.get("length"), "length", 1, _MAX_BYTES)
            return _blob(pf, offset=offset, length=length), "current", None
        observed = {"source": _blob(pf)}
        bv = _clean_view(bridge, pf)
        versions = _versions()
        if kind == "function":
            facts = _function_observation(bv, _integer(record.get("function"), "function", 0))
        elif kind == "links":
            facts = _links_observation(bridge, bv)
        else:
            raise _Unavailable("Unsupported evidence check")
        # Native GUI edits and background analysis do not share the bridge lock.
        # Check native and bridge dirty state plus the durable source again.
        if _clean_view(bridge, pf) != bv or _blob(pf) != observed["source"]:
            raise _Unavailable("Analysis backing changed during observation")
        observed.update(facts, tool_versions=versions)
        return observed, "current", None
    except Exception as exc:  # Missing facts and native exceptions fail closed.
        return observed, "unknown", f"Observation unavailable: {type(exc).__name__}: {exc}"


def _provider_recheck(bridge: Any, baseline: dict) -> tuple[str, str | None]:
    """Known provider changes remain stale even without a live consumer view."""
    providers = baseline.get("providers", {})
    if not isinstance(providers, dict) or len(providers) > _MAX_BINDINGS:
        return "unknown", "Captured provider evidence is invalid"
    errors = []
    for file_id, fingerprint in providers.items():
        try:
            pf = _exact_file(bridge, file_id)
            if pf is None:
                return "stale", f"Captured provider ProjectFile ID {file_id} no longer exists"
            if _blob(pf) != fingerprint:
                return "stale", f"Captured provider content changed: {file_id}"
        except Exception as exc:
            errors.append(f"Provider observation unavailable: {type(exc).__name__}: {exc}")
    return ("unknown", "; ".join(errors)) if errors else ("current", None)


def check_evidence(bridge: Any, captured: list[dict]) -> dict:
    """Reobserve without refreshing baselines; stale outranks unknown."""
    checks = []
    validate_captured_evidence(captured)
    for index, record in enumerate(captured):
        kind = record.get("check") if isinstance(record, dict) else None
        observed = None
        try:
            if kind not in {"file", "bytes", "function", "links", "note"}:
                raise _Unavailable("Invalid captured evidence check")
            pf = None if kind == "note" else _exact_file(bridge, record["file_id"])
            if kind != "note" and pf is None:
                state, reason = "stale", "Exact captured project file ID no longer exists"
            else:
                observed, state, reason = _observe(bridge, record, pf)
                mismatch = _expected_reason(record, observed)
                baseline = record.get("observed")
                source_changed = (
                    isinstance(baseline, dict) and isinstance(observed, dict)
                    and "source" in baseline and "source" in observed
                    and baseline["source"] != observed["source"]
                )
                if mismatch or source_changed:
                    state, reason = "stale", mismatch or "Persisted source blob changed"
                elif record.get("capture_state") == "stale":
                    state, reason = "stale", "Original capture did not satisfy its expected observation"
                elif record.get("capture_state") == "unknown" or not record.get("fingerprint"):
                    state, reason = "unknown", "Original capture has no complete durable baseline; revise to recapture"
                elif state == "current" and _digest(observed) != record["fingerprint"]:
                    state, reason = "stale", "Observed evidence changed from its captured baseline"
        except Exception as exc:  # Native failures never imply currency.
            state, reason = "unknown", f"Evidence recheck unavailable: {type(exc).__name__}: {exc}"
            if isinstance(record, dict) and record.get("capture_state") == "stale":
                state, reason = "stale", f"Original capture did not satisfy its expectation; {reason}"
        if (state != "stale" and kind == "links" and isinstance(record.get("observed"), dict)
                and state == "unknown"):
            provider_state, provider_reason = _provider_recheck(bridge, record["observed"])
            if provider_state == "stale":
                state, reason = provider_state, provider_reason
            elif provider_reason:
                reason = f"{reason}; {provider_reason}"
        row = {"index": index, "check": kind, "freshness": state}
        if reason:
            row["reason"] = reason
        if observed is not None:
            row["observed"] = observed
        checks.append(row)
    machine = [row for row in checks if row["check"] != "note"]
    states = {row["freshness"] for row in machine}
    freshness = "stale" if "stale" in states else "unknown" if not machine or "unknown" in states else "current"
    reasons = [f"Evidence {row['index']}: {row['reason']}" for row in checks
               if row.get("reason") and (row["check"] != "note" or not machine)]
    return {"freshness": freshness, "reasons": reasons, "checks": checks}
