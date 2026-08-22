"""Decompile / IL / disasm / prototype / locals / dataflow read handlers.

The decompile/IL/disasm read-op cluster that used to live on
``BinaryNinjaBridge`` moves here as module-level free functions, each taking the
``BridgeContext`` seam (``ctx``) in place of ``self``. ``BinaryNinjaBridge``
keeps a thin delegating shim for every name the test suite / op binders
reference (``_decompile``, ``_function_info``, ``_get_prototype``,
``_list_locals_for_function``, ``_il``, ``_disasm``, ``_structured_il``,
``_defuse``, ``_resolved_calls``, ``_possible_values``, ``_pvs_targets``,
``_force_function_analysis``).

Outbound calls resolve through:
  * ``ctx`` -- resolution helpers relocated to the seam (``_resolve_view``,
    ``_find_function``, ``_functions_containing``);
  * ``il_format`` -- the pure IL/HLIL/disasm renderers and SSA/value-set
    helpers (``_decompile_text``, ``_render_warnings``, ``_analysis_stub_warning``,
    ``_comment_map``, ``_function_metadata``, ``_function_text``, ``_disasm_text``,
    ``_il_function_for``, ``_il_op_name``, ``_ssa_var_entry``,
    ``_resolve_ssa_variable``, ``_serialize_pvs``, ``_pvs_determined``);
  * ``vars`` -- variable discovery (``_list_locals``, ``_find_variable_selector``);
  * ``taint_engine`` -- call-target / function resolution used by the dataflow
    ops (imported as the SAME ``from . import taint_engine as _taint`` alias the
    bridge uses);
  * ``_shared`` -- module-free helpers (``_parse_address``, ``OperationFailure``).

Import direction is one-way: this module imports ``il_format``, ``vars``,
``taint_engine``, and ``_shared`` (plus stdlib + binaryninja). It NEVER imports
``bridge`` or ``seam`` -- those import THIS module one-way (design spec §3.2).
"""
from __future__ import annotations

from typing import Any

try:
    import binaryninja as bn  # bn.Architecture[...] for the #382 --mode arch lookup
except ModuleNotFoundError:  # importable without the Binary Ninja runtime (tests, tooling)
    bn = None  # type: ignore[assignment]

from . import il_format
from . import read_xrefs
from . import taint_engine as _taint
from . import vars as vars_mod
from ._shared import OperationFailure, _parse_address  # noqa: F401
from .bridge_state import require_analysis


def _force_function_analysis(ctx, bv, func):
    """Override a skipped function's analysis and reanalyze it in place.

    Mutates the BinaryView (skip override + reanalysis), so callers must hold
    the exclusive write lock. Returns the (possibly rebuilt) function object.
    """
    start = int(func.start)
    try:
        func.analysis_skipped = False  # setter installs NeverSkipFunctionAnalysis
    except Exception:
        pass
    try:
        func.reanalyze()
    except Exception:
        pass
    bv.update_analysis_and_wait()
    return bv.get_function_at(start) or func


# Forcing analysis on an oversized, uncalled region is the data-as-code trap:
# BN tentatively made a function on a string/pointer table or packed data, and
# the forced decode runs away. Only nudge for genuinely large regions so a
# routine forced reanalysis of a normal function stays quiet.
_FORCED_DATA_MIN_BYTES = 0x4000


def _forced_data_region_warning(bv, func) -> str | None:
    """A hedged verify-nudge when a force-analyzed function looks like it may be
    a DATA region BN tentatively typed as code (#371.1).

    The reliable signal common to the observed cases (a pointer/tag table AND
    high-entropy packed data) is structural, not content-based: a large region
    with zero inbound code refs AND zero inbound data refs. Byte/string
    heuristics miss the high-entropy case, so this stays a *verify* nudge, never
    a "this is data" verdict. Real oversized code almost always has inbound
    callers; requiring zero DATA refs too suppresses the false positive on a
    large `.init_array`/ctor-dispatched initializer (real code with no direct
    callers, but pointed at by the init table -- i.e. a data ref to its start).
    """
    try:
        size = int(getattr(func, "total_bytes", 0) or 0)
    except Exception:
        size = 0
    if size < _FORCED_DATA_MIN_BYTES:
        return None
    if read_xrefs._code_ref_count(bv, func.start) != 0:
        return None
    try:
        if len(list(bv.get_data_refs(func.start))) != 0:
            return None
    except Exception:
        return None
    return (
        f"{func.name}: forced analysis decoded {size} bytes with 0 inbound code refs. "
        "BN sometimes makes a function on a DATA region (string/pointer table or packed "
        "data) that grows under forced decode; if the body reads as tables/strings/garbage "
        "rather than coherent code, treat this decode as suspect and verify before acting on it."
    )


def _annotate_containment(ctx, result, identifier, func) -> None:
    """Tag a function-scoped READ result when the request resolved to a
    containing function rather than an exact start (#193 Part 4).

    No-op for an exact start or a name; otherwise adds ``resolved_from`` so an
    agent feeding a taint/trace sink address knows the hit was mid-function.
    """
    meta = ctx._containment_meta(identifier, func)
    if meta:
        result["resolved_from"] = meta


def _thunk_veneer_warning(ctx, bv, func) -> str | None:
    """A `thunk/veneer -> <target>` warning when *func* is a PLT/GOT trampoline
    (#446), reusing `evidence function`'s detector so the signal is consistent.
    None when it is not a thunk."""
    from . import read_evidence  # lazy: read_evidence -> il_format, not this module
    summary = read_evidence._function_thunk_summary(ctx, bv, func)
    if not summary.get("is_candidate"):
        return None
    target = summary.get("target") if isinstance(summary.get("target"), dict) else None
    if target and (target.get("name") or target.get("address")):
        name, addr = target.get("name"), target.get("address")
        dest = f"{name} @ {addr}" if name and addr else (name or addr)
        return (f"thunk/veneer -> {dest}: this is a PLT/GOT trampoline (a jump to "
                f"the real body), not a self-recursive function.")
    return ("thunk/veneer: this is a PLT/GOT trampoline (a small jump stub), not a "
            "real function body; the apparent self-call is the resolved target.")


def _decompile(ctx, selector: str | None, identifier, *, addresses: bool = False, force_analysis: bool = False):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    forced = False
    if force_analysis and bool(getattr(func, "analysis_skipped", False)):
        func = _force_function_analysis(ctx, bv, func)
        forced = True
    text = il_format._decompile_text(bv, func, addresses=addresses)
    warnings = il_format._render_warnings(text)
    stub = il_format._analysis_stub_warning(func, text, forced=forced)
    if stub:
        warnings.append(stub)
    # #446: a PLT/GOT veneer decompiles as an apparent infinite self-recursion
    # (`return foo();`) because the trampoline resolves back to the same symbol
    # name. Emit a thunk warning naming the real target instead of leaving the
    # reader to misread a 4-instruction jump as recursion. Gated on the tailcall
    # render (the exact confusing case) so a normal decompile pays nothing; the
    # detector then confirms the small jump-to-target shape.
    if "/* tailcall */" in text:
        thunk_warn = _thunk_veneer_warning(ctx, bv, func)
        if thunk_warn:
            warnings.append(thunk_warn)
    if forced:
        data_warn = _forced_data_region_warning(bv, func)
        if data_warn:
            warnings.append(data_warn)
    comments = il_format._comment_map(bv, func)
    result = {
        "function": {"name": func.name, "address": hex(func.start)},
        "text": text,
        "comments": comments,
        "warnings": warnings,
        "analysis_skipped": bool(getattr(func, "analysis_skipped", False)),
        # `analysis_force_requested` echoes the --force-analysis flag; `analysis_forced`
        # is True only when a reanalysis actually ran this call. On a second forced
        # decompile the function is no longer skipped, so nothing reruns and
        # analysis_forced is False -- the echo tells callers the flag wasn't ignored.
        "analysis_force_requested": bool(force_analysis),
        "analysis_forced": forced,
    }
    _annotate_containment(ctx, result, identifier, func)
    return result


def _basic_block_ranges(func) -> list[dict[str, Any]]:
    """Address ranges of a function's basic blocks, with incoming/outgoing edges.

    #653.10: a 23 KB / 565-block dispatch function decompiles to 6324 lines
    (100k tokens), and `--lines` slices blindly because the line numbers are
    unknowable until AFTER the full decompile -- so every downstream question
    became grep/sed over a local file instead of a `bn` query. Block ranges make
    `disasm --linear <addr>` / `--lines` targetable without reading the whole body
    first, and on a jump-table dispatcher each case arm is its own block.
    """
    blocks = []
    for index, block in enumerate(getattr(func, "basic_blocks", None) or []):
        start = int(getattr(block, "start", 0) or 0)
        end = int(getattr(block, "end", 0) or 0)
        entry: dict[str, Any] = {
            "index": index,
            "start": hex(start),
            "end": hex(end),
            "length": max(0, end - start),
        }
        try:
            entry["outgoing"] = [hex(int(e.target.start)) for e in (block.outgoing_edges or [])]
        except Exception:
            entry["outgoing"] = []
        try:
            entry["incoming"] = [hex(int(e.source.start)) for e in (block.incoming_edges or [])]
        except Exception:
            entry["incoming"] = []
        blocks.append(entry)
    blocks.sort(key=lambda b: int(b["start"], 16))
    return blocks


def _function_info(ctx, selector: str | None, identifier, *, blocks: bool = False):
    bv = ctx._resolve_view(selector)
    require_analysis(bv, "Function info")
    func = ctx._find_function(bv, identifier, contained=True)
    metadata = il_format._function_metadata(func)
    variables = vars_mod._list_locals(func)
    parameters = [item for item in variables if item["is_parameter"]]
    locals_only = [item for item in variables if not item["is_parameter"]]
    # Use the genuine code-ref count so a page-aligned function isn't credited
    # with spurious adrp page-base materializations (#284).
    code_ref_count = read_xrefs._code_ref_count(bv, func.start)
    result = {
        "function": {
            "name": func.name,
            "address": hex(func.start),
            "raw_name": getattr(func, "raw_name", func.name),
            "display_name": il_format._display_name(func),
        },
        **metadata,
        "parameters": parameters,
        "locals": locals_only,
        "xref_count": code_ref_count,
        # Count (+ addresses) of instructions BN's lifter could not model, so a
        # function with unlifted computation isn't mistaken for fully analyzed
        # (#206). count:0 means no unlifted instruction was found.
        "unimplemented_instructions": il_format._unimplemented_instructions(func),
    }
    if blocks:
        # #653.10: opt-in -- a 565-block function's ranges are themselves bulky.
        result["basic_blocks"] = _basic_block_ranges(func)
    _annotate_containment(ctx, result, identifier, func)
    return result


def _get_prototype(ctx, selector: str | None, identifier):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    result = {
        "function": {
            "name": func.name,
            "address": hex(func.start),
            "raw_name": getattr(func, "raw_name", func.name),
        },
        **il_format._function_metadata(func),
    }
    _annotate_containment(ctx, result, identifier, func)
    return result


def _list_locals_for_function(ctx, selector: str | None, identifier):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    variables = vars_mod._list_locals(func)
    result = {
        "function": {
            "name": func.name,
            "address": hex(func.start),
            "raw_name": getattr(func, "raw_name", func.name),
        },
        # #651: `items` is documented as ALWAYS the data container, and `local list`
        # was the one exception -- documented once in #248, then lost in the
        # SKILL.md -> reference/ split. `jq '.items[]'` on a function with 40
        # recovered locals silently reported nothing (a wrong key yields null, never
        # an error). Emit `items` so the documented rule is true, and keep `locals`
        # as a retained alias so existing scripts/skills keep working. Both are the
        # same list object -- there is no second source of truth to drift.
        "kind": "locals",
        "items": variables,
        "locals": variables,
    }
    _annotate_containment(ctx, result, identifier, func)
    return result


def _il(ctx, selector: str | None, identifier, view: str, ssa: bool):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    text = il_format._function_text(bv, func, view=view, ssa=ssa)
    result = {
        "function": {"name": func.name, "address": hex(func.start)},
        "view": view,
        "ssa": ssa,
        "text": text,
        "warnings": il_format._render_warnings(text),
    }
    _annotate_containment(ctx, result, identifier, func)
    return result


def _disasm(ctx, selector: str | None, identifier, linear=None, mode=None,
            snap_to_instruction: bool = False):
    bv = ctx._resolve_view(selector)
    if linear is not None:
        return _disasm_linear(ctx, bv, identifier, int(linear), mode=mode,
                              snap_to_instruction=bool(snap_to_instruction))
    try:
        func = ctx._find_function(bv, identifier, contained=True)
    except Exception as exc:
        # An address BN never made part of a function is exactly the stripped-lane
        # case --linear exists for: point there instead of at a dead end (#314).
        # Fire ONLY for that specific "no function here" dead end at an address
        # (hex or decimal) -- not for an ambiguous overlap, a bad selector, or a
        # name-not-found, where the --linear hint would mislead. A fresh
        # RuntimeError (not type(exc)(...)) avoids assuming the original
        # exception's constructor takes a single string.
        looks_like_address = True
        try:
            _parse_address(identifier)
        except ValueError:
            looks_like_address = False
        if looks_like_address and "No function found" in str(exc):
            raise RuntimeError(
                f"{exc}. To inspect the raw bytes there regardless of function "
                f"membership, use `disasm {identifier} --linear N`."
            ) from exc
        raise
    result = {
        "function": {"name": func.name, "address": hex(func.start)},
        "text": il_format._disasm_text(bv, func),
    }
    _annotate_containment(ctx, result, identifier, func)
    return result


# Hard ceiling on a single linear-disassembly request so a pathological count
# can't walk the whole address space; the output spill already handles large
# (but bounded) dumps.
_LINEAR_DISASM_MAX = 100_000


def _resolve_linear_address(ctx, bv, identifier) -> int:
    """Resolve a linear-disasm target to a concrete address: a literal address as
    given, else a function/symbol NAME to its start. Raises a clear error when it
    resolves to nothing."""
    try:
        return _parse_address(identifier)
    except ValueError:
        pass
    try:
        func = ctx._find_function(bv, identifier)
        if func is not None:
            return int(func.start)
    except Exception:
        pass
    try:
        symbol = bv.get_symbol_by_raw_name(str(identifier))
    except Exception:
        symbol = None
    if symbol is not None:
        return int(symbol.address)
    raise ValueError(
        f"could not resolve {identifier!r} to an address; pass a 0x-prefixed "
        f"address or a known function/symbol name for --linear disassembly"
    )


_ARM_MODE_ARCHES = {
    ("arm", False): "armv7", ("arm", True): "armv7eb",
    ("thumb", False): "thumb2", ("thumb", True): "thumb2eb",
}


def _is_classic_arm_or_thumb_arch(name: str) -> bool:
    """True for classic 32-bit ARM/Thumb arch names (armv7/thumb2 and variants);
    False for AArch64 (#600 -- "arm64" starts with "arm" and would otherwise be
    misclassified as classic ARM/Thumb) and for any non-ARM arch."""
    n = (name or "").lower()
    if "aarch64" in n or "arm64" in n:
        return False
    return n.startswith("arm") or n.startswith("thumb")


def _linear_decode_arch(ctx, bv, address: int, mode):
    """The architecture to linearly decode at *address* (#382).

    BN defaults a whole ARM binary to ONE mode (often thumb2), so an ARM-mode
    region under a thumb2 default -- or vice versa -- otherwise decodes in the
    wrong mode. Resolution order:
      1. an explicit ``mode`` (arm/thumb) forces armv7/thumb2 (endianness taken
         from the bv arch), for the stripped/missed case with no function;
      2. else the containing function's own arch (BN's per-function ARM/Thumb
         mode) when known, so a known ARM function decodes as ARM automatically;
      3. else the bv default arch (prior behavior)."""
    bv_arch = getattr(bv, "arch", None)
    if mode is not None:
        # Harden the raw JSON/bridge path: the CLI restricts --mode to arm/thumb,
        # but a direct {"mode":"mips"} request would otherwise KeyError on
        # _ARM_MODE_ARCHES below and surface as an internal error (#382 review).
        if mode not in ("arm", "thumb"):
            raise ValueError(
                f"--mode must be 'arm' or 'thumb' (got {mode!r})"
            )
        cur = str(getattr(bv_arch, "name", "") or "")
        if not _is_classic_arm_or_thumb_arch(cur):
            raise ValueError(
                f"--mode {mode} is only for ARM/Thumb targets (this target is "
                f"{cur or 'unknown'})"
            )
        name = _ARM_MODE_ARCHES[(mode, cur.endswith("eb"))]
        try:
            return bn.Architecture[name]
        except Exception as exc:
            raise ValueError(f"could not load the {name} architecture for --mode {mode}: {exc}")
    try:
        containers = ctx._functions_containing(bv, int(address))
        if containers and getattr(containers[0], "arch", None) is not None:
            return containers[0].arch
    except Exception:
        pass
    return bv_arch


def _function_instruction_starts(bv, func) -> set[int]:
    """The set of recovered instruction START addresses in *func* (#550).

    Walks BN's basic blocks with the function's own arch instruction lengths --
    the authoritative decode boundaries. Used to detect a ``--linear`` start that
    lands INSIDE a function but mid-instruction (which would decode junk that
    looks plausible)."""
    arch = getattr(func, "arch", None)
    starts: set[int] = set()
    try:
        blocks = list(func.basic_blocks)
    except Exception:
        return starts
    for block in blocks:
        try:
            addr = int(block.start)
            end = int(block.end)
        except Exception:
            continue
        while addr < end:
            starts.add(addr)
            addr += max(1, il_format._instruction_length(bv, addr, arch=arch))
    return starts


def _linear_boundary_check(bv, func, address: int) -> dict[str, Any] | None:
    """If *address* is inside *func* but not at a recovered instruction start,
    return a boundary-warning descriptor naming the nearest valid start(s) (#550);
    else None. A start exactly on a boundary is silent."""
    starts = _function_instruction_starts(bv, func)
    if not starts or address in starts:
        return None
    at_or_below = max((s for s in starts if s <= address), default=None)
    above = min((s for s in starts if s > address), default=None)
    return {
        "requested": hex(address),
        "in_function": {"name": str(func.name), "address": hex(int(func.start))},
        "nearest_start_at_or_below": hex(at_or_below) if at_or_below is not None else None,
        "nearest_start_above": hex(above) if above is not None else None,
    }


def _disasm_linear(ctx, bv, identifier, count: int, *, mode=None,
                   snap_to_instruction: bool = False) -> dict[str, Any]:
    """Linear disassembly of *count* instructions from an arbitrary MAPPED address,
    independent of function membership (#314). The stripped/static lane needs to
    read the bytes at a suspected missed handler -- a dispatch/vtable slot BN left
    as data -- before deciding whether to `function create` it; the
    function-scoped path refuses such addresses outright.

    The decode architecture honors the address's function arch (or an explicit
    ARM/Thumb ``mode``) so an ARM-mode region isn't mis-decoded as Thumb (#382)."""
    address = _resolve_linear_address(ctx, bv, identifier)
    if count <= 0:
        raise ValueError("--linear count must be a positive number of instructions")
    requested_count = int(count)
    count = min(requested_count, _LINEAR_DISASM_MAX)
    # #382 review Finding 2: an ARM code pointer commonly carries bit 0 as the
    # Thumb tag, so a value-set/symbol target can be odd while the instruction
    # lives at addr&~1. Decoding from the odd value starts one byte in. ARM has no
    # odd instruction addresses, so on an ARM/Thumb target an odd resolved address
    # is the Thumb tag -- mask it (and disclose) before decoding. Detect ARM-ness
    # from the bv arch name so this is independent of --mode / function arch, and
    # NEVER mask on non-ARM targets where odd code addresses are legitimate.
    thumb_tag_normalized = None
    bv_arch_name = str(getattr(getattr(bv, "arch", None), "name", "") or "")
    if (address & 1) and _is_classic_arm_or_thumb_arch(bv_arch_name):
        thumb_tag_normalized = address
        address &= ~1
    if not _address_is_mapped(bv, address):
        raise ValueError(
            f"address {hex(address)} is not mapped in this binary; nothing to "
            f"disassemble there"
        )
    # #550: detect a start that lands inside a known function but NOT at a recovered
    # instruction boundary -- linear decoding from mid-instruction produces junk that
    # reads like real control flow. Warn (naming the nearest valid starts) and, with
    # --snap-to-instruction, snap DOWN to the enclosing instruction's start.
    containing_func = None
    try:
        containers = ctx._functions_containing(bv, int(address))
        if containers:
            containing_func = containers[0]
    except Exception:
        containing_func = None
    boundary_warning = None
    snapped_from = None
    if containing_func is not None:
        boundary_warning = _linear_boundary_check(bv, containing_func, address)
        if boundary_warning is not None and snap_to_instruction:
            below = boundary_warning.get("nearest_start_at_or_below")
            if below is not None:
                snapped_from = address
                address = int(below, 16)
                boundary_warning = None  # snapped onto a real boundary; no longer junk
    arch = _linear_decode_arch(ctx, bv, address, mode)
    # Under an explicit --mode the decode is FORCED to that arch: a byte the forced
    # mode can't model must surface as `.byte` (the existing path below), NOT a
    # silent BV-default-arch decode that would contradict the forced-mode note
    # (#382 review). Without --mode, keep the lenient BV fallback (prior behavior).
    strict = mode is not None
    entries: list[dict[str, Any]] = []
    lines: list[str] = []
    addr = int(address)
    for _ in range(count):
        if not _address_is_mapped(bv, addr):
            break
        length = max(1, il_format._instruction_length(bv, addr, arch=arch, strict=strict))
        entry = il_format._disasm_entry(bv, addr, arch=arch, strict=strict)
        text = entry.get("text") or ""
        if not text:
            # Mapped, but no valid instruction decodes here (data / an invalid
            # opcode -- the very thing you point --linear at to confirm). Surface
            # the raw byte and advance one byte so the window keeps moving and
            # the caller sees these bytes aren't code, instead of silently
            # stopping at zero instructions.
            one = bv.read(addr, 1)
            if not one:
                break
            text = f".byte 0x{one.hex()}"
            length = 1
        raw = bv.read(addr, length)
        hex_bytes = raw.hex(" ") if raw else ""
        entries.append({
            "address": hex(addr),
            "bytes": hex_bytes,
            "length": int(length),
            "text": text,
        })
        lines.append(f"{addr:08x}  {hex_bytes:<16} {text}")
        addr += length
    # Recompute containment at the FINAL (possibly snapped) start address.
    in_function = None
    if containing_func is not None:
        in_function = {
            "name": containing_func.name,
            "address": hex(int(containing_func.start)),
        }
    else:
        try:
            containers = ctx._functions_containing(bv, int(address))
            if containers:
                fn = containers[0]
                in_function = {"name": fn.name, "address": hex(int(fn.start))}
        except Exception:
            in_function = None
    note = (
        f"linear disassembly of {len(entries)} instruction"
        f"{'' if len(entries) == 1 else 's'} from {hex(address)} "
        # #550: name the ordering explicitly -- this is address-linear (byte order),
        # NOT the basic-block/graph order a function-scoped `disasm <fn>` renders.
        f"(address-linear order, not function-bounded)"
    )
    if thumb_tag_normalized is not None:
        note += (
            f"; normalized a Thumb function-pointer tag (bit 0) "
            f"{hex(thumb_tag_normalized)} -> {hex(address)}"
        )
    if snapped_from is not None:
        note += (
            f"; snapped {hex(snapped_from)} back to the enclosing instruction "
            f"start {hex(address)} (--snap-to-instruction)"
        )
    if in_function is not None:
        note += f"; this address is inside {in_function['name']} @ {in_function['address']}"
    if boundary_warning is not None:
        # #550: the start is inside a function but mid-instruction -- the decode below
        # is very likely junk. Name the nearest valid starts so the caller can re-run.
        below = boundary_warning.get("nearest_start_at_or_below")
        above = boundary_warning.get("nearest_start_above")
        nearest = ", ".join(x for x in (below, above) if x) or "unknown"
        note += (
            f"; WARNING: {boundary_warning['requested']} is NOT a recovered instruction "
            f"boundary inside {boundary_warning['in_function']['name']} -- linear decode "
            f"from here may be junk. Nearest valid start(s): {nearest}. Re-run at a valid "
            f"start or add --snap-to-instruction"
        )
    arch_name = str(getattr(arch, "name", "") or "")
    if arch_name:
        # #382: disclose the decode mode so an ARM/Thumb decode isn't silently
        # trusted in the wrong mode (and so --mode's effect is visible).
        forced = " (forced via --mode)" if mode is not None else ""
        note += f"; decoded as {arch_name}{forced}"
    if requested_count > _LINEAR_DISASM_MAX:
        note += f"; capped at {_LINEAR_DISASM_MAX} (requested {requested_count})"
    return {
        "linear": True,
        "order": "address-linear",
        "function": in_function,
        "address": hex(address),
        "decode_arch": arch_name or None,
        "requested_count": requested_count,
        "capped": requested_count > _LINEAR_DISASM_MAX,
        "instruction_count": len(entries),
        "instructions": entries,
        "text": "\n".join(lines),
        "boundary_warning": boundary_warning,
        "snapped_from": hex(snapped_from) if snapped_from is not None else None,
        "note": note,
    }


def _address_is_mapped(bv, address: int) -> bool:
    try:
        if hasattr(bv, "is_valid_offset"):
            return bool(bv.is_valid_offset(int(address)))
    except Exception:
        pass
    try:
        return bool(bv.read(int(address), 1))
    except Exception:
        return False


def _structured_il(ctx, selector, identifier, *, view: str = "mlil", ssa: bool = True):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    il = il_format._il_function_for(func, view, ssa)
    instructions = []
    try:
        items = list(il.instructions)
    except Exception:
        items = []
    for ins in items:
        opn = il_format._il_op_name(ins)
        instructions.append({
            "il_index": int(getattr(ins, "instr_index", -1)),
            "address": hex(int(getattr(ins, "address", func.start))),
            "op": opn,
            "text": str(ins),
            "vars_read": [il_format._ssa_var_entry(v) for v in (getattr(ins, "vars_read", None) or [])],
            "vars_written": [il_format._ssa_var_entry(v) for v in (getattr(ins, "vars_written", None) or [])],
            "operands_summary": [str(o) for o in (getattr(ins, "operands", None) or [])],
            "is_call": "CALL" in opn,
        })
    result = {
        "function": {"name": func.name, "address": hex(func.start)},
        "view": view,
        "ssa": ssa,
        "instructions": instructions,
    }
    _annotate_containment(ctx, result, identifier, func)
    return result


def _defuse(ctx, selector, identifier, var_selector: str):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    il = il_format._il_function_for(func, "mlil", True)
    ssa_var, other_versions = il_format._resolve_ssa_variable(func, il, var_selector)

    try:
        definition = il.get_ssa_var_definition(ssa_var)
    except Exception:
        definition = None
    try:
        uses = list(il.get_ssa_var_uses(ssa_var) or [])
    except Exception:
        uses = []

    def _ref(ins):
        if ins is None:
            return None
        return {
            "il_index": int(getattr(ins, "instr_index", -1)),
            "address": hex(int(getattr(ins, "address", func.start))),
            "op": il_format._il_op_name(ins),
            "text": str(ins),
        }

    is_phi = definition is not None and "PHI" in il_format._il_op_name(definition)
    phi_sources = []
    if is_phi:
        for s in (getattr(definition, "src", None) or []):
            phi_sources.append(il_format._ssa_var_entry(s))

    result = {
        "function": {"name": func.name, "address": hex(func.start)},
        "variable": il_format._ssa_var_entry(ssa_var),
        "definition": _ref(definition),
        "uses": [_ref(u) for u in uses],
        "is_phi": is_phi,
        "phi_sources": phi_sources,
        "other_versions": other_versions or [],
    }
    _annotate_containment(ctx, result, identifier, func)
    return result


def _pvs_targets(ctx, bv, pvs) -> list[dict[str, Any]]:
    if pvs is None:
        return []
    type_name = getattr(getattr(pvs, "type", None), "name", "") or ""
    addrs: list[int] = []
    if type_name in {"ConstantValue", "ConstantPointerValue", "ImportedAddressValue", "ExternalPointerValue"}:
        value = getattr(pvs, "value", None)
        if value is not None:
            try:
                addrs.append(int(value))
            except Exception:
                pass
    elif type_name == "InSetOfValues":
        for v in (getattr(pvs, "values", None) or []):
            try:
                addrs.append(int(v))
            except Exception:
                pass
    elif type_name == "LookupTableValue":
        mapping = getattr(pvs, "mapping", None)
        if isinstance(mapping, dict):
            for v in mapping.values():
                try:
                    addrs.append(int(v))
                except Exception:
                    pass
        else:
            for entry in (getattr(pvs, "table", None) or []):
                target = getattr(entry, "to", None)
                if target is not None:
                    try:
                        addrs.append(int(target))
                    except Exception:
                        pass
    targets = []
    for addr in addrs:
        # function_at normalizes the Thumb low bit so an odd value-set target
        # resolves to its function (#89 Problem B). The raw address is kept
        # in the reported "address".
        fn = _taint.function_at(bv, addr)
        name = None
        if fn is not None:
            name = str(getattr(fn, "name", None))
        else:
            sym = bv.get_symbol_at(addr) if hasattr(bv, "get_symbol_at") else None
            if sym is None and (addr & 1) and hasattr(bv, "get_symbol_at"):
                sym = bv.get_symbol_at(addr & ~1)
            name = str(getattr(sym, "name", None)) if sym is not None else None
        targets.append({"address": hex(addr), "name": name})
    return targets


def _resolved_calls(ctx, selector, identifier, *, direction: str = "both", resolve_indirect: bool = True):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    # `kind` lets a consumer of the {kind, ...} family identify a callgraph read
    # (it is a composite callees+callers structure, not a flat items list) (#371.2).
    result: dict[str, Any] = {
        "kind": "callgraph",
        "function": {"name": func.name, "address": hex(func.start)},
    }

    if direction in ("callees", "both"):
        il = il_format._il_function_for(func, "mlil", True)
        callees = []
        try:
            items = list(il.instructions)
        except Exception:
            items = []
        for ins in items:
            opn = il_format._il_op_name(ins)
            if "CALL" not in opn and "TAILCALL" not in opn:
                continue
            # resolve_call_target (no thunk following) resolves MLIL_IMPORT
            # calls and Thumb-tagged targets that const_target/exact-address
            # lookup miss (#89). Constant direct calls resolve identically.
            resolved = _taint.resolve_call_target(bv, ins, follow_thunks=False)
            target = resolved.address
            row = {
                "call_addr": hex(int(getattr(ins, "address", func.start))),
                "il_index": int(getattr(ins, "instr_index", -1)),
            }
            if target is not None:
                fn = resolved.function or (bv.get_function_at(target) if hasattr(bv, "get_function_at") else None)
                name = str(fn.name) if fn is not None else None
                if name is None:
                    sym = bv.get_symbol_at(target) if hasattr(bv, "get_symbol_at") else None
                    name = str(sym.name) if sym is not None else None
                row.update({"kind": "direct", "target": {"address": hex(target), "name": name}})
            else:
                row.update({"kind": "indirect", "dest_expr": str(getattr(ins, "dest", ""))})
                if resolve_indirect:
                    pvs = getattr(getattr(ins, "dest", None), "possible_values", None)
                    resolved = _pvs_targets(ctx, bv, pvs)
                    row["resolved"] = resolved
                    type_name = getattr(getattr(pvs, "type", None), "name", "") or "none"
                    row["resolution"] = "possible_values" if resolved else "unresolved"
                    row["resolution_detail"] = type_name
            callees.append(row)
        result["callees"] = callees

    if direction in ("callers", "both"):
        callers = []
        seen: set[int] = set()
        for site in (list(getattr(func, "caller_sites", None) or [])):
            addr = int(getattr(site, "address", 0))
            fn = getattr(site, "function", None)
            if fn is None:
                functions = ctx._functions_containing(bv, addr)
                fn = functions[0] if functions else None
            caller = (
                {"address": hex(int(fn.start)), "name": str(fn.name)} if fn is not None else None
            )
            callers.append({"call_addr": hex(addr), "caller": caller})
        if not callers:
            for fn in (list(getattr(func, "callers", None) or [])):
                marker = int(getattr(fn, "start", 0))
                if marker in seen:
                    continue
                seen.add(marker)
                callers.append({"caller": {"address": hex(marker), "name": str(fn.name)}})
        result["callers"] = callers

    _annotate_containment(ctx, result, identifier, func)
    return result


def _possible_values(ctx, selector, identifier, at):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier, contained=True)
    address = _parse_address(at)
    il = il_format._il_function_for(func, "mlil", True)
    target_ins = None
    try:
        for ins in list(il.instructions):
            if int(getattr(ins, "address", -1)) == address:
                target_ins = ins
                break
    except Exception:
        target_ins = None
    if target_ins is None:
        # No MLIL instruction begins at --at: returning a success dict with
        # expression:None makes a bogus address look like a real instruction
        # with an unknown value-set. Refuse instead of emitting a false read (#526).
        raise OperationFailure(
            "no_instruction",
            f"No MLIL instruction at {hex(address)} in {func.name}: --at must name the "
            f"address of an instruction within the function (use `bn function il "
            f"--view mlil` to list valid instruction addresses).",
            requested={"identifier": str(identifier), "at": hex(address)},
        )
    instr_pvs = getattr(target_ins, "possible_values", None) if target_ins is not None else None
    src_expr = getattr(target_ins, "src", None) if target_ins is not None else None
    src_pvs = getattr(src_expr, "possible_values", None) if src_expr is not None else None
    # BN leaves a SET_VAR/STORE *instruction* value-set undetermined while the
    # SOURCE expression (the value being assigned) carries the real value-set
    # -- const / range / lookup-table. Report the source's value-set for an
    # assignment so values surface as the help promises, instead of always
    # printing UndeterminedValue; fall back to the instruction-level set when
    # there is no source or only the instruction-level set is determined (#52).
    if il_format._pvs_determined(src_pvs) and not il_format._pvs_determined(instr_pvs):
        chosen, basis = src_pvs, "source_expression"
    elif il_format._pvs_determined(instr_pvs):
        chosen, basis = instr_pvs, "instruction"
    elif src_pvs is not None:
        chosen, basis = src_pvs, "source_expression"
    else:
        chosen, basis = instr_pvs, "instruction"
    result = {
        "function": {"name": func.name, "address": hex(func.start)},
        "at": hex(address),
        "expression": str(target_ins) if target_ins is not None else None,
        "value_basis": basis,
        "source_expression": str(src_expr) if src_expr is not None else None,
        "possible_values": il_format._serialize_pvs(chosen),
    }
    _annotate_containment(ctx, result, identifier, func)
    return result
