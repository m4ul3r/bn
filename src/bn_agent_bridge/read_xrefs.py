"""Cross-reference resolution: code/data xrefs, import-symbol xrefs, field xrefs.

The xref-resolution methods that used to live on ``BinaryNinjaBridge`` move here
as module-level free functions, each taking the ``BridgeContext`` seam (``ctx``)
in place of ``self``. ``BinaryNinjaBridge`` keeps a thin delegating shim for
every name the test suite / op binders reference (``_xrefs``,
``_xrefs_to_address``, ``_scan_for_calls_to``, ``_resolve_type_field``, ...).

Outbound calls resolve through:
  * ``ctx`` -- resolution / address-context / type helpers relocated to the seam
    (``_resolve_view``, ``_find_function``, ``_functions_containing``,
    ``_address_context``, ``_find_type``);
  * ``il_format`` -- the state-free IL helpers used by the call scan
    (``_iter_llil_instructions``, ``_il_op_name``, ``_llil_constant_value``);
  * ``_shared`` -- module-free helpers (``_parse_address``).

Import direction is one-way: this module imports ``il_format`` and ``_shared``
(plus stdlib + binaryninja). It NEVER imports ``bridge``, ``seam``,
``read_evidence``, ``read_misc``, or ``create_comments`` -- those import THIS
module one-way (design spec §3.2).
"""
from __future__ import annotations

import difflib
from typing import Any

try:
    import binaryninja as bn
except ModuleNotFoundError:  # importable without the Binary Ninja runtime (tests, tooling)
    bn = None  # type: ignore[assignment]

from . import il_format
from . import taint_engine as _taint
from ._shared import _parse_address, _require_mapped_address, _validate_count
from .bridge_state import require_analysis

# Import symbol kinds, in resolution-preference order. Mirrors the literal that
# also lives as ``BinaryNinjaBridge._IMPORT_SYMBOL_TYPES`` (used by the imports
# op, which stays on the bridge); kept here verbatim so the xref free functions
# need no callback into the class.
_IMPORT_SYMBOL_TYPES: list[tuple[str, str]] = [
    ("ImportedFunctionSymbol", "function"),
    ("ImportedDataSymbol", "data"),
    ("ImportAddressSymbol", "address"),
]


def _xrefs(ctx, selector: str | None, identifier, *, offset: int = 0, limit: int | None = None,
           fn_pointer_scan: bool = False):
    bv = ctx._resolve_view(selector)
    require_analysis(bv, "Cross-references")
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    try:
        address = _parse_address(identifier)
    except Exception:
        # NAME path. A bare name can resolve to >=2 functions -- GCC emits a
        # 16-byte PLT-style thunk AND the real body under the same name in shared
        # libs. The thunk usually carries all the call traffic while the real body
        # shows zero callers, so silently resolving to one member reads a hot
        # utility as dead code (#220). Detect the collision and surface it -- but
        # only when it's a thunk/real group (resolvable); two genuine real bodies
        # stay an ambiguous error (#122).
        bodies = ctx._find_functions_by_name(bv, str(identifier), case_sensitive=True)
        if not bodies:
            bodies = ctx._find_functions_by_name(bv, str(identifier), case_sensitive=False)
        if len(bodies) >= 2:
            # An import/extern STUB shadowing exactly one real body is the #201
            # PIC self-reference case (a demangled name matching both a PLT veneer
            # and the local definition): resolve to the DEFINITION and never report
            # the stub over it -- the stub usually carries the call traffic, so a
            # pure ref-count tiebreak would pick it (regressing #201). Reuse the
            # same impl-over-stub chokepoint _find_function uses. Only a genuine
            # same-name group with NO stub-typed member (a GCC thunk/real pair,
            # both FunctionSymbol) reaches the collision surfacing (#220).
            impl = ctx._resolve_impl_over_stub(bodies)
            if impl is not None:
                result = _xrefs_to_address(ctx, bv, int(impl.start), offset=offset, limit=limit,
                                           fn_pointer_scan=fn_pointer_scan)
                if any(int(getattr(f, "start", -1)) != int(impl.start) for f in bodies):
                    result["resolved_to_definition"] = hex(int(impl.start))
                return _drop_legacy_ref_arrays(result)
            reals = [f for f in bodies if not _is_thunk_like(bv, f)]
            if len(reals) <= 1:
                return _drop_legacy_ref_arrays(
                    _xrefs_ambiguous(ctx, bv, str(identifier), bodies, offset=offset, limit=limit)
                )
            # else: >=2 genuine non-thunk bodies -> let _find_function raise ambiguous.
        try:
            address = ctx._find_function(bv, identifier).start
        except RuntimeError as exc:
            # An ambiguous identifier is actionable as-is; replacing it
            # with "not found / not an import symbol" would be misleading.
            # Only fall back to import-symbol lookup for genuine misses.
            if "Ambiguous" in str(exc):
                raise
            return _drop_legacy_ref_arrays(
                _xrefs_import_symbol(ctx, bv, identifier, offset=offset, limit=limit)
            )
    else:
        # Raw-address path (parse succeeded): reject an unmapped address rather
        # than returning a false-negative empty xref set (#374). A function start
        # (the name path above) is always mapped, so only the literal-address
        # case needs the guard. But NEVER reject an address BN actually holds refs
        # FOR -- 0x0 is the placeholder for unresolved indirect-call sites (many
        # real code refs, is_valid_offset False), and a tail-call can target an
        # out-of-image address; rejecting those would discard a real answer. Only
        # an address that is BOTH unmapped AND ref-less is the typo case (#374
        # follow-up).
        has_refs = bool(
            list(bv.get_code_refs(int(address))) or list(bv.get_data_refs(int(address)))
        )
        if not has_refs:
            _require_mapped_address(bv, int(address))
    return _drop_legacy_ref_arrays(
        _xrefs_to_address(ctx, bv, address, offset=offset, limit=limit,
                          fn_pointer_scan=fn_pointer_scan)
    )


def _is_thunk_like(bv, fn) -> bool:
    """A function that forwards to another (a PLT-style veneer / GCC same-name
    thunk): BN's ``is_thunk`` flag, or a single-tailcall body that ``follow_thunk``
    resolves to a different target. Used to tell a thunk/real same-name pair
    (resolvable, #220) from two genuine implementations (ambiguous, #122)."""
    if bool(getattr(fn, "is_thunk", False)):
        return True
    try:
        resolved = _taint.follow_thunk(bv, fn)
    except Exception:
        resolved = None
    return resolved is not None and int(getattr(resolved, "start", -1)) != int(getattr(fn, "start", -2))


def _reg_name(obj) -> str:
    """Register name from an ILRegister (``SET_REG.dest``), an ``LLIL_REG`` expr,
    or a test fake -- robust across real BN and the unit fakes."""
    if obj is None:
        return ""
    name = getattr(obj, "name", None)
    if isinstance(name, str):
        return name
    src = getattr(obj, "src", None)  # LLIL_REG expr -> ILRegister
    if src is not None:
        return _reg_name(src) or str(src)
    return str(obj)


def _expr_contains_reg(expr, reg: str) -> bool:
    """True if *expr* (recursively) reads register *reg* via an ``LLIL_REG``."""
    if expr is None or getattr(expr, "operation", None) is None:
        return False
    if il_format._il_op_name(expr) == "LLIL_REG" and _reg_name(expr) == reg:
        return True
    for operand in getattr(expr, "operands", None) or []:
        if _expr_contains_reg(operand, reg):
            return True
    return False


def _expr_reg_offset(expr, reg: str):
    """The constant in-page offset *k* if *expr* contains ``ADD(REG(reg), CONST(k))``
    (either operand order, at any depth), else ``None``. Captures both an explicit
    ``add xN, xN, #k`` and a ``[xN, #k]`` load/store address expression."""
    if expr is None or getattr(expr, "operation", None) is None:
        return None
    if il_format._il_op_name(expr) == "LLIL_ADD":
        operands = list(getattr(expr, "operands", None) or [])
        if len(operands) == 2:
            for maybe_reg, maybe_const in (operands, operands[::-1]):
                if (getattr(maybe_reg, "operation", None) is not None
                        and il_format._il_op_name(maybe_reg) == "LLIL_REG"
                        and _reg_name(maybe_reg) == reg):
                    k = il_format._llil_constant_value(maybe_const)
                    if k is not None:
                        return k
    for operand in getattr(expr, "operands", None) or []:
        found = _expr_reg_offset(operand, reg)
        if found is not None:
            return found
    return None


def _adrp_pagebase_is_spurious(adrp_il, following_ils, page_base: int) -> bool:
    """Decide whether an adrp that materializes *page_base* is a spurious xref to
    a function starting at that page base (#284).

    An ``adrp xN, <page>`` produces the 4 KB page base; the real referent is
    ``page + offset`` from the paired ``add``/``ldr``/``str``. Given the adrp's
    LLIL instruction and the LLIL instructions that follow it in the same basic
    block, the page base is *spurious* iff the first consumer of the destination
    register offsets it by a NONZERO constant (so the true target is elsewhere in
    the page). A zero offset, a direct use (function-pointer take), a redefinition
    before use, or a non-``SET_REG`` ref (a call/branch) are all genuine -- the
    rule only ever drops on positive nonzero-offset evidence, so it can never hide
    a real caller."""
    if il_format._il_op_name(adrp_il) != "LLIL_SET_REG":
        return False
    if il_format._llil_constant_value(getattr(adrp_il, "src", None)) != int(page_base):
        return False
    dest = _reg_name(getattr(adrp_il, "dest", None))
    if not dest:
        return False
    for nxt in following_ils:
        body = getattr(nxt, "src", None) if il_format._il_op_name(nxt) == "LLIL_SET_REG" else nxt
        offset = _expr_reg_offset(body, dest)
        if offset is not None:
            return offset != 0
        if _expr_contains_reg(body, dest):
            return False  # pointer used as-is -> genuine &fn
        if il_format._il_op_name(nxt) == "LLIL_SET_REG" and _reg_name(getattr(nxt, "dest", None)) == dest:
            return False  # page base redefined before use -> not a ref
    return False


def _is_spurious_adrp_pagebase(bv, ref, address: int) -> bool:
    """BV glue around :func:`_adrp_pagebase_is_spurious` for one code ref."""
    a = int(getattr(ref, "address", 0))
    disasm = ""
    get_disassembly = getattr(bv, "get_disassembly", None)
    if callable(get_disassembly):
        try:
            disasm = get_disassembly(a) or ""
        except Exception:
            disasm = ""
    # adrp is the only AArch64 op that materializes a page base; gating on the
    # mnemonic keeps the filter from ever touching x86/other-arch refs.
    if disasm.split()[:1] != ["adrp"]:
        return False
    fn = getattr(ref, "function", None)
    get_llil_at = getattr(fn, "get_low_level_il_at", None)
    if not callable(get_llil_at):
        return False
    try:
        il = get_llil_at(a)
    except Exception:
        il = None
    if il is None:
        return False
    bb = getattr(il, "il_basic_block", None)
    following = []
    if bb is not None:
        idx = int(getattr(il, "instr_index", -1))
        try:
            following = [j for j in bb if int(getattr(j, "instr_index", -1)) > idx]
            following.sort(key=lambda j: int(getattr(j, "instr_index", 0)))
        except Exception:
            following = []
    return _adrp_pagebase_is_spurious(il, following, int(address))


def _genuine_code_refs(bv, address: int) -> list:
    """Code refs to *address*, with spurious adrp page-base materializations
    dropped when *address* is page-aligned (#284). Non-page-aligned targets
    cannot be an adrp page base, so their refs pass through untouched."""
    get_code_refs = getattr(bv, "get_code_refs", None)
    if not callable(get_code_refs):
        return []
    try:
        raw = list(get_code_refs(int(address)))
    except Exception:
        return []
    if int(address) & 0xFFF:
        return raw
    return [ref for ref in raw if not _is_spurious_adrp_pagebase(bv, ref, int(address))]


def _code_ref_count(bv, address: int) -> int:
    try:
        return len(_genuine_code_refs(bv, int(address)))
    except Exception:
        return 0


def _xrefs_ambiguous(ctx, bv, identifier: str, bodies, *, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
    """Report xrefs for a same-name collision (#220): pick the member carrying the
    most code refs (the thunk usually holds the traffic; the real body shows
    zero), and surface every member under ``ambiguous_symbol`` so an analyst never
    mistakes a hot utility for dead code."""
    members = []
    for fn in bodies:
        addr = int(fn.start)
        members.append({
            "address": hex(addr),
            "name": str(fn.name),
            "size": il_format._function_size(fn),
            "code_ref_count": _code_ref_count(bv, addr),
            "is_thunk": _is_thunk_like(bv, fn),
        })
    members.sort(key=lambda m: (-m["code_ref_count"], int(m["address"], 16)))
    chosen = members[0]
    chosen_addr = int(chosen["address"], 16)
    result = _xrefs_to_address(ctx, bv, chosen_addr, offset=offset, limit=limit)
    others = ", ".join(
        f"{m['address']} ({m['code_ref_count']} refs)" for m in members if m["address"] != chosen["address"]
    )
    result["ambiguous_symbol"] = {
        "identifier": identifier,
        "members": members,
        "resolved_to": chosen["address"],
        "note": (
            f"'{identifier}' resolves to {len(members)} functions under the same name "
            f"(thunk + real body); reporting xrefs for {chosen['address']} "
            f"(most code refs: {chosen['code_ref_count']}). Other members: {others}"
        ),
    }
    return result


def _xrefs_any(ctx, selector: str | None, symbols: list[Any]) -> dict[str, Any]:
    """Batch-probe several symbols in one call (a VR sink sweep): resolve+count
    refs for each, recording present+counts or absent. "Not linked" is a valid
    answer, not an error, so the call succeeds (exit 0) regardless of which
    symbols exist -- no `set -e` shell loop aborting on the first absent one
    (#218)."""
    bv = ctx._resolve_view(selector)
    require_analysis(bv, "Cross-references")
    results: list[dict[str, Any]] = []
    present = 0
    for raw in symbols:
        sym = str(raw)
        try:
            res = _xrefs(ctx, selector, sym)
        except RuntimeError as exc:
            msg = str(exc)
            if "Ambiguous" in msg:
                # The symbol DOES exist (it resolves to >=2 bodies); "absent"
                # would be a false negative for a sink sweep. Record it present
                # but ambiguous so the analyst probes it directly (#218 review).
                present += 1
                results.append({"symbol": sym, "present": True, "ambiguous": True,
                                "note": msg.splitlines()[0]})
            else:
                results.append({"symbol": sym, "present": False,
                                "note": msg.splitlines()[0]})
            continue
        present += 1
        results.append({
            "symbol": sym,
            "present": True,
            "address": res.get("address"),
            "code_ref_count": res.get("code_ref_count", 0),
            "data_ref_count": res.get("data_ref_count", 0),
            "caller_function_count": res.get("caller_function_count", 0),
        })
    return {"kind": "symbol_presence", "items": results, "count": len(results), "present": present}


def _drop_legacy_ref_arrays(envelope: dict[str, Any]) -> dict[str, Any]:
    """Strip the deprecated full ``code_refs``/``data_refs`` arrays from the
    ``xrefs`` OP response so ``--offset``/``--limit`` bound the entire serialized
    payload, not just ``items`` (#184). On a high-fanout symbol the arrays rode
    full regardless of paging and spilled the JSON even at ``--limit 1``. The
    full-set summary counts (``code_ref_count``/``data_ref_count``/
    ``caller_function_count``) and the paged ``items`` (each carrying its
    ``kind``) are everything a "who references X" triage needs. The lower-level
    builders (``_xrefs_to_address``/``_xrefs_import_symbol`` via
    ``_xref_envelope``) still produce the dual shape, which ``function info`` and
    evidence message-lensing embed by calling them directly."""
    envelope.pop("code_refs", None)
    envelope.pop("data_refs", None)
    return envelope


def _xref_envelope(address, target_context, code_refs, data_refs, *,
                   offset: int = 0, limit: int | None = None,
                   extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Wrap xref results in the canonical paging envelope (#164).

    ``items`` is the unified list (code refs first, then data refs), each row
    carrying its ``kind`` (code|data). ``code_refs``/``data_refs`` are kept as a
    deprecated dual shape for back-compat, the text renderer, and ``function
    info`` (which embeds the full set, unpaged). Summary counts (#140) reflect
    the FULL set regardless of paging."""
    caller_addrs = {
        ref["caller_function"]["address"]
        for ref in code_refs
        if isinstance(ref.get("caller_function"), dict) and ref["caller_function"].get("address")
    }
    items = code_refs + data_refs
    total = len(items)
    page = items[offset:]
    if limit is not None:
        page = page[:limit]
    out: dict[str, Any] = {
        "kind": "xrefs",
        "address": hex(address) if isinstance(address, int) else address,
        "target_context": target_context,
        "code_ref_count": len(code_refs),
        "data_ref_count": len(data_refs),
        "caller_function_count": len(caller_addrs),
        # Internal dual shape: the top-level `xrefs` op strips these via
        # _drop_legacy_ref_arrays (#184); they survive only where function info /
        # evidence message-lensing embed this envelope and read the full set.
        "code_refs": code_refs,
        "data_refs": data_refs,
        # Canonical paging envelope.
        "items": page,
        "total": total,
        "offset": offset,
        "limit": limit,
        "returned": len(page),
        "has_more": (offset + len(page)) < total,
    }
    if extra:
        out.update(extra)
    return out


# Cap the bytes scanned for stored function pointers so the back-link scan can't
# blow up on a huge data segment; if hit, the result flags it as truncated.
_FN_PTR_SCAN_BYTE_BUDGET = 32 * 1024 * 1024

# Section-name PREFIXES to SKIP: relocation/symbol/string/unwind/note metadata
# are not pointer-table homes -- scanning them only widens the false-positive
# surface (a reloc addend equals the target address) and wastes budget (#323
# review). Matched by prefix so `.data.rel.ro` (a genuine pointer table) is NOT
# caught by the relocation-table `.rela`/`.rel.` prefixes. Kept as a deny-list
# (not an allow-list) so a firmware blob's custom-named data sections are still
# scanned -- the firmware case is the whole point.
_FN_PTR_SCAN_DENY_PREFIXES = (
    ".rela", ".rel.", ".dynamic", ".dynsym", ".dynstr", ".symtab", ".strtab",
    ".hash", ".gnu.hash", ".gnu.version", ".eh_frame", ".gcc_except", ".note",
    ".comment", ".interp", ".plt", ".got.plt", ".debug",
)


def _data_section_ranges(bv) -> list[tuple[str, int, int]]:
    """Readable, initialized, non-code data sections to scan for stored function
    pointers (``.data``, ``.data.rel.ro``, ``.rodata``, ``.init_array``, ...).
    Code, relocation/symbol/unwind metadata, and uninitialized (NOBITS/.bss)
    sections are skipped -- a pointer-sized match there is noise, not a
    function-pointer table slot (#323)."""
    out: list[tuple[str, int, int]] = []
    sections = getattr(bv, "sections", None) or {}
    try:
        values = list(sections.values())
    except Exception:
        return out
    for sec in values:
        name = str(getattr(sec, "name", ""))
        lname = name.lower()
        sem = str(getattr(getattr(sec, "semantics", None), "name", getattr(sec, "semantics", "")))
        if "Code" in sem:
            continue
        # .bss / NOBITS hold no file data (zero-fill); scanning wastes budget and
        # trips the short-read truncation check.
        if "bss" in lname or "NoBackedSection" in sem or "External" in sem:
            continue
        if any(lname.startswith(p) for p in _FN_PTR_SCAN_DENY_PREFIXES):
            continue
        if "extern" in lname or "synthetic" in lname:
            continue
        start = int(getattr(sec, "start", 0) or 0)
        length = int(getattr(sec, "length", 0) or 0)
        if length <= 0:  # some Section objects expose end, not length
            length = max(0, int(getattr(sec, "end", 0) or 0) - start)
        if length <= 0:
            continue
        out.append((name, start, length))
    return out


def _function_pointer_data_refs(ctx, bv, address: int, existing_addrs: set[int]):
    """Scan data sections for pointer-sized words equal to *address* -- a function
    stored in a dispatch/vtable/codec table (#323). BN often does not model these
    as data refs (especially reloc-applied PIE pointers, which read all-zero
    until relocated), so a callback-only function otherwise reads as dead. Returns
    ``([(slot, section_name, thumb), ...], truncated)`` for slots BN missed,
    pointer-aligned only (a fn-ptr table is aligned; an unaligned hit is almost
    always a coincidental byte run)."""
    # Never scan for the image-base / a body-less pseudo-function (e.g. an ELF
    # header BN models as a function at bv.start): its address is a common
    # constant (the load base) that recurs throughout .rodata/headers, producing
    # only false positives, never a real callback table slot (#323 review).
    if int(address) == int(getattr(bv, "start", -1)):
        return [], False
    fn = bv.get_function_at(int(address)) if hasattr(bv, "get_function_at") else None
    if fn is not None:
        blocks = getattr(fn, "basic_blocks", None)
        if blocks is not None and len(list(blocks)) == 0:
            return [], False  # body-less synthetic/data pseudo-function
    pointer_size = ctx._pointer_size(bv)
    order = ctx._byteorder(bv)
    needles: dict[bytes, bool] = {int(address).to_bytes(pointer_size, order): False}
    arch_name = str(getattr(getattr(bv, "arch", None), "name", "")).lower()
    if "arm" in arch_name or "thumb" in arch_name:  # a fn ptr may be stored as addr|1
        needles[(int(address) | 1).to_bytes(pointer_size, order)] = True
    refs: list[tuple[int, str, bool]] = []
    seen = set(existing_addrs)
    budget = _FN_PTR_SCAN_BYTE_BUDGET
    truncated = False
    for name, start, length in _data_section_ranges(bv):
        if budget <= 0:
            truncated = True
            break
        read_len = min(length, budget)
        if read_len < length:  # the byte budget (not a short read) clipped this section
            truncated = True
        try:
            data = bytes(bv.read(start, read_len) or b"")
        except Exception:
            continue
        budget -= len(data)
        for needle, thumb in needles.items():
            pos = 0
            while True:
                i = data.find(needle, pos)
                if i < 0:
                    break
                pos = i + 1
                slot = start + i
                if slot % pointer_size != 0 or slot in seen:
                    continue
                seen.add(slot)
                refs.append((slot, name, thumb))
    refs.sort(key=lambda t: t[0])
    return refs, truncated


def _xrefs_to_address(ctx, bv, address: int, *, offset: int = 0, limit: int | None = None,
                      fn_pointer_scan: bool = False) -> dict[str, Any]:
    code_refs = []
    data_refs = []
    # Drop spurious adrp page-base materializations for a page-aligned target
    # (#284); non-page-aligned targets pass through unchanged.
    raw_code_refs = _genuine_code_refs(bv, address)
    # #286: an exported function's intra-lib callers reference its same-name PLT
    # stub, not the body. Union the stub(s)' callers into the body's xrefs so a
    # hot exported function isn't reported as having zero code callers.
    raw_code_refs, stub_starts = _union_stub_code_refs(ctx, bv, address, raw_code_refs)
    for ref in sorted(raw_code_refs, key=lambda item: int(item.address)):
        fn = getattr(ref, "function", None)
        caller = (
            {"address": hex(int(fn.start)), "name": str(fn.name)}
            if fn is not None
            else None
        )
        ref_addr = int(ref.address)
        ref_arch = getattr(ref, "arch", None) or getattr(fn, "arch", None)
        code_refs.append(
            {
                "function": fn.name if fn is not None else None,
                "address": hex(ref_addr),
                "caller_function": caller,
                "kind": "code",
                "context": ctx._address_context(
                    bv, ref_addr, include_disasm=True, arch=ref_arch, assume_code=True
                ),
            }
        )
    get_data_refs = getattr(bv, "get_data_refs", None)
    raw_data_refs = list(get_data_refs(address)) if callable(get_data_refs) else []
    for ref_addr in sorted(raw_data_refs):
        ref_addr = int(ref_addr)
        functions = ctx._functions_containing(bv, ref_addr)
        fn = functions[0] if functions else None
        caller = (
            {"address": hex(int(fn.start)), "name": str(fn.name)}
            if fn is not None
            else None
        )
        data_refs.append(
            {
                "function": fn.name if fn is not None else None,
                "address": hex(ref_addr),
                "caller_function": caller,
                "kind": "data",
                "context": ctx._address_context(bv, ref_addr),
            }
        )
    fn_pointer_truncated = False
    if fn_pointer_scan:
        # Back-link a function reached only through a stored function pointer
        # (vtable / codec table / fuse_operations / plugin registry) that BN did
        # not model as a data ref, so it isn't falsely reported as dead (#323).
        existing = {int(r["address"], 16) for r in data_refs}
        fp_refs, fn_pointer_truncated = _function_pointer_data_refs(ctx, bv, address, existing)
        for slot, _section_name, thumb in fp_refs:
            functions = ctx._functions_containing(bv, slot)
            fn = functions[0] if functions else None
            caller = (
                {"address": hex(int(fn.start)), "name": str(fn.name)}
                if fn is not None
                else None
            )
            entry = {
                "function": fn.name if fn is not None else None,
                "address": hex(slot),
                "caller_function": caller,
                "kind": "data",
                "function_pointer": True,
                "context": ctx._address_context(bv, slot),
            }
            if thumb:
                entry["thumb_pointer"] = True
            data_refs.append(entry)
        data_refs.sort(key=lambda r: int(r["address"], 16))
    envelope = _xref_envelope(
        address,
        ctx._address_context(bv, address, include_disasm=True),
        code_refs,
        data_refs,
        offset=offset,
        limit=limit,
    )
    if fn_pointer_truncated:
        envelope["fn_pointer_scan_truncated"] = True
    if stub_starts:
        # Disclose that some callers were reached through a same-name PLT stub,
        # so the unioned count is self-documenting rather than surprising (#286).
        envelope["stub_callers_via"] = [hex(s) for s in stub_starts]
    return envelope


def _union_stub_code_refs(ctx, bv, address: int, code_refs: list):
    """Append code refs to same-name PLT/extern stubs that forward to the
    function at *address* (#286), de-duped by ref address. Returns
    ``(merged_refs, [stub_start, ...])``; the stub list is empty when there is no
    function at *address* or it has no same-name stub sibling."""
    get_fn_at = getattr(bv, "get_function_at", None)
    impl = get_fn_at(int(address)) if callable(get_fn_at) else None
    if impl is None:
        return code_refs, []
    try:
        stubs = ctx._same_name_stub_functions(bv, impl)
    except Exception:
        stubs = []
    if not stubs:
        return code_refs, []
    merged = list(code_refs)
    seen = {int(getattr(r, "address", -1)) for r in merged}
    stub_starts = []
    for stub in stubs:
        stub_start = int(getattr(stub, "start", -1))
        if stub_start < 0:
            continue  # robustness: never query refs to a sentinel address
        stub_starts.append(stub_start)
        for ref in _genuine_code_refs(bv, stub_start):
            ra = int(getattr(ref, "address", -1))
            if ra not in seen:
                merged.append(ref)
                seen.add(ra)
    return merged, stub_starts


def _import_symbol_name(sym) -> str:
    """Preferred display name for an import symbol."""
    return str(
        getattr(sym, "short_name", None)
        or getattr(sym, "full_name", None)
        or sym.name
    )


def _find_import_symbol(ctx, bv, name: str):
    needle = name.lower()
    for attr_name, kind in _IMPORT_SYMBOL_TYPES:
        sym_type = getattr(bn.SymbolType, attr_name, None)
        if sym_type is None:
            continue
        for sym in list(bv.get_symbols_of_type(sym_type)):
            if _import_symbol_name(sym).lower() == needle:
                return sym
    return None


def _find_data_symbol(ctx, bv, name: str):
    """A non-function symbol (DataSymbol, etc.) by name -- demangled or raw -- or
    None. Lets `xrefs g_state_table` resolve a global/data symbol instead of
    failing with a misleading 'use bn imports' (#224b)."""
    cands: list[Any] = []
    getbn = getattr(bv, "get_symbols_by_name", None)
    if callable(getbn):
        try:
            cands = list(getbn(name) or [])
        except Exception:
            cands = []
    if not cands:
        graw = getattr(bv, "get_symbol_by_raw_name", None)
        if callable(graw):
            try:
                s = graw(name)
                if s is not None:
                    cands = [s]
            except Exception:
                cands = []
    for s in cands:
        st = str(getattr(getattr(s, "type", None), "name", "") or "")
        if "Function" in st:
            continue  # functions are handled by _find_function; we want data here
        if getattr(s, "address", None) is not None:
            return s
    return None


def _xrefs_import_symbol(ctx, bv, identifier: str, *, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
    sym = _find_import_symbol(ctx, bv, identifier)
    if sym is None:
        # #224b: not a function or import -- try a data symbol (a global table,
        # state struct, ...) and xref its address before erroring, so the result
        # matches `xrefs <address>` instead of a misleading import-only error.
        data_sym = _find_data_symbol(ctx, bv, identifier)
        if data_sym is not None:
            result = _xrefs_to_address(ctx, bv, int(data_sym.address), offset=offset, limit=limit)
            result["resolved_symbol"] = {
                "name": str(identifier),
                "kind": "data",
                "address": hex(int(data_sym.address)),
            }
            return result
        available: list[str] = []
        for attr_name, kind in _IMPORT_SYMBOL_TYPES:
            sym_type = getattr(bn.SymbolType, attr_name, None)
            if sym_type is None:
                continue
            for s in list(bv.get_symbols_of_type(sym_type)):
                available.append(_import_symbol_name(s))
        suggestions = difflib.get_close_matches(identifier, sorted(set(available)), n=5, cutoff=0.5)
        msg = f"Function not found: {identifier}."
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}"
        msg += (" Not found as an import or data symbol either. "
                "Use 'bn imports' for imports, or pass the address directly.")
        raise RuntimeError(msg)

    # #201: a demangled C++ name matches an import veneer (PLT stub) via its
    # short_name, but the same symbol may also be DEFINED in this module (a PIC
    # self-reference). Resolving xrefs to the stub gives the wrong call-graph, so
    # redirect to the real definition -- reusing the same impl-over-stub resolver
    # `_find_function` uses for a name collision. `xrefs <mangled>` and decompile
    # already reach the definition; this makes `xrefs <demangled>` consistent.
    raw_name = str(getattr(sym, "raw_name", sym.name))
    bodies = ctx._find_functions_by_name(bv, raw_name, case_sensitive=True)
    impl = ctx._resolve_impl_over_stub(bodies) if bodies else None
    if impl is not None and int(impl.start) != int(sym.address):
        result = _xrefs_to_address(ctx, bv, int(impl.start), offset=offset, limit=limit)
        result["import_resolved"] = True
        result["import_name"] = str(identifier)
        result["resolved_to_definition"] = hex(int(impl.start))
        return result

    sym_address = int(sym.address)
    result = _xrefs_to_address(ctx, bv, sym_address, offset=offset, limit=limit)
    result["import_resolved"] = True
    result["import_name"] = str(identifier)

    if not result.get("code_refs"):
        manual = _scan_for_calls_to(ctx, bv, sym_address)
        if manual:
            # Rebuild the envelope so the manually-discovered code refs land in
            # both the deprecated `code_refs` and the canonical `items` page.
            result = _xref_envelope(
                sym_address, result["target_context"], manual, result["data_refs"],
                offset=offset, limit=limit,
                extra={"import_resolved": True, "import_name": str(identifier),
                       "code_refs_scanned": True},
            )

    return result


def _scan_for_calls_to(ctx, bv, target_address: int) -> list[dict[str, Any]]:
    code_refs = []
    seen: set[int] = set()
    for fn in list(bv.functions):
        for insn in il_format._iter_llil_instructions(fn):
            op_name = il_format._il_op_name(insn)
            if op_name not in {"LLIL_CALL", "LLIL_CALL_STACK_ADJUST", "LLIL_TAILCALL"}:
                continue
            dest_value = il_format._llil_constant_value(getattr(insn, "dest", None))
            if dest_value != target_address:
                continue
            ref_addr = int(getattr(insn, "address", 0))
            if ref_addr in seen:
                continue
            seen.add(ref_addr)
            fn_arch = getattr(fn, "arch", None)
            code_refs.append({
                "function": str(fn.name),
                "address": hex(ref_addr),
                "caller_function": {
                    "address": hex(int(fn.start)),
                    "name": str(fn.name),
                },
                "kind": "code",
                "context": ctx._address_context(
                    bv, ref_addr, include_disasm=True, arch=fn_arch, assume_code=True
                ),
            })
    code_refs.sort(key=lambda item: int(item["address"], 16))
    return code_refs


def _resolve_type_field(ctx, bv, field_spec: str):
    type_name, sep, field_name = str(field_spec).rpartition(".")
    if not sep or not type_name or not field_name:
        raise RuntimeError("Field selector must be in the form Struct.field")

    resolved_name, type_obj = ctx._find_type(bv, type_name)
    members = getattr(type_obj, "members", None)
    if members is None:
        raise RuntimeError(f"Type is not a struct-like type: {resolved_name}")

    member_list = list(members)

    def field_info(member, index: int):
        return {
            "type_name": resolved_name,
            "field_name": str(getattr(member, "name", "")) or field_name,
            "offset": int(getattr(member, "offset", 0)),
            "member_index": index,
            "field_type": str(getattr(member, "type", "")),
        }

    for index, member in enumerate(member_list):
        if str(getattr(member, "name", "")) != field_name:
            continue
        return field_info(member, index)

    folded_matches = [
        (index, member)
        for index, member in enumerate(member_list)
        if str(getattr(member, "name", "")).lower() == field_name.lower()
    ]
    if len(folded_matches) == 1:
        index, member = folded_matches[0]
        return field_info(member, index)

    try:
        requested_offset = _parse_address(field_name)
    except Exception:
        requested_offset = None
    if requested_offset is not None:
        for index, member in enumerate(member_list):
            if int(getattr(member, "offset", 0)) != requested_offset:
                continue
            return field_info(member, index)
        raise RuntimeError(f"Field not found: {resolved_name}.0x{requested_offset:x}")

    available = [str(getattr(member, "name", "")) for member in member_list if str(getattr(member, "name", ""))]
    suggestions = difflib.get_close_matches(field_name, available, n=5, cutoff=0.5)
    if suggestions:
        raise RuntimeError(
            f"Field not found: {resolved_name}.{field_name}. Did you mean: {', '.join(suggestions)}"
        )
    raise RuntimeError(f"Field not found: {resolved_name}.{field_name}")


def _field_xrefs(ctx, selector: str | None, field_spec: str,
                 *, offset: int = 0, limit: int | None = None):
    # #532: validate paging inputs on the same contract every other paged op uses,
    # so a raw-socket / py-exec caller can't pass a negative offset or limit<=0 and
    # get Python slice semantics instead of a clean error.
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    bv = ctx._resolve_view(selector)
    field = _resolve_type_field(ctx, bv, field_spec)

    code_refs = []
    for ref in sorted(
        list(bv.get_code_refs_for_type_field(field["type_name"], field["offset"])),
        key=lambda item: int(getattr(item, "address", 0)),
    ):
        func = getattr(ref, "func", None)
        address = int(getattr(ref, "address", 0))
        code_refs.append(
            {
                "function": func.name if func is not None else None,
                "address": hex(address),
                "size": int(getattr(ref, "size", 0)),
                "incoming_type": str(getattr(ref, "incomingType", "")) or None,
                "disasm": bv.get_disassembly(address) or "",
            }
        )

    data_refs = []
    for address in sorted(list(bv.get_data_refs_for_type_field(field["type_name"], field["offset"]))):
        symbol = bv.get_symbol_at(address)
        # BinaryView has no get_type_at(); the data variable defined at the
        # address carries the type. The old call raised AttributeError and
        # took the whole --field query down whenever a field had data refs.
        data_var = bv.get_data_var_at(address)
        type_obj = getattr(data_var, "type", None) if data_var is not None else None
        data_refs.append(
            {
                "address": hex(address),
                "symbol": symbol.name if symbol is not None else None,
                "type": str(type_obj) if type_obj is not None else None,
            }
        )

    # #275: unified items envelope -- code refs first then data refs, each row
    # tagged with its `kind` (code|data) -- with the `field` descriptor as
    # top-level metadata. The legacy code_refs/data_refs arrays are dropped.
    items = ([{**ref, "kind": "code"} for ref in code_refs]
             + [{**ref, "kind": "data"} for ref in data_refs])
    # #532: page the unified list like every other xref path (canonical paging
    # envelope keys: offset/limit/returned/has_more/total). Hot fields used to
    # return the whole list ignoring --limit/--offset and spill to disk.
    total = len(items)
    page = items[offset:]
    if limit is not None:
        page = page[:limit]
    return {
        "kind": "field_xrefs",
        "field": field,
        "items": page,
        "total": total,
        "offset": offset,
        "limit": limit,
        "returned": len(page),
        "has_more": (offset + len(page)) < total,
    }
