from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import re
from typing import Any, Callable, Iterator, Sequence

from .target_hint import open_target_lines, target_row
from .transport import BridgeError

# "rollback_failed" = an op succeeded but the batch revert that should have
# undone it failed, so the view may be left modified -- a real failure. A
# cleanly rolled-back sibling ("reverted") is NOT a failure and is omitted (#118).
# "internal_error" = an unexpected engine bug (distinct from an unsupported
# request); still a failure, so exit codes/rendering flag it (#122).
FAILED_MUTATION_STATUSES = {"unsupported", "verification_failed", "invalid_request", "rollback_failed", "internal_error"}

# Control chars (C0 minus the ones we name, plus DEL) in a symbol name would
# break a --format text row across lines or corrupt the terminal. Escape them so
# the row stays on one line and the name is still readable (#370.1). JSON output
# is untouched -- it round-trips the raw name faithfully. #771 routes the other
# operator-settable free-text cells -- comment text, tag data, local names, and
# the import library / raw symbol name columns -- through the same helper, for
# the same reason: each is one cell of a row.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _escape_control_chars(text: Any) -> str:
    s = str(text)
    s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return _CONTROL_CHAR_RE.sub(lambda m: f"\\x{ord(m.group()):02x}", s)


def _render_fallback_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True)


def _render_function_bundle_text(value: Any) -> str:
    """`bundle function` is a composite JSON artifact (function info + decompile +
    IL + xrefs + types) with no compact text form. Without a renderer the explicit
    --format text path emitted a single line of escaped JSON; pretty-print it with
    a note instead so a text-defaulting agent gets something readable (#362)."""
    if not isinstance(value, (dict, list)):
        return _render_fallback_text(value)
    try:
        body = json.dumps(value, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        return _render_fallback_text(value)
    return ("# function bundle: composite JSON (use --format json for machine "
            "consumption)\n" + body)


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce a nested field to a dict for safe ``.get()`` chains.

    A renderer that does ``_field_dict(value, "function")`` still crashes when the
    field is present but a NON-dict (a string/list from a malformed or future
    bridge result), because the non-dict is truthy and reaches ``.get()``. This
    returns ``{}`` for anything that isn't a dict so the renderer degrades to
    placeholder text instead of an AttributeError (#101)."""
    return value if isinstance(value, dict) else {}


def _skew_note(*fields: str) -> str:
    """Disclose a container field that was PRESENT but held a value of the WRONG
    shape, instead of rendering it as if it were empty.

    ``_as_dict`` stops the AttributeError, but coercing a malformed
    field to ``{}``/``[]`` makes the row render byte-identically to a genuinely
    empty result: the caller reads a confident "nothing here" and cannot tell
    the payload was unusable. Base raised loudly, so degrading must stay loud --
    an undetectable wrong answer is worse than the crash it replaced (#619)."""
    plural = "s" if len(fields) > 1 else ""
    return (f"! malformed {', '.join(fields)} field{plural}: not the expected "
            "shape -- the rows or counts it carries may be missing or partial "
            "(use --format json)")


# The field names a renderer coerced away during the CURRENT render. Recorded at
# the coercion itself rather than declared per renderer: a declaration only ever
# sees the renderer's own top-level payload, so a container coerced on a nested
# dict, on a list ELEMENT, or inside a helper the renderer hands its whole
# payload to stayed silent -- which is how the first two attempts at this fix
# each closed the instances they enumerated and left the class open (#619).
_SKEWED_FIELDS: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "_skewed_fields", default=None)


def _record_skew(field: str) -> None:
    """Note one skewed field for the enclosing ``@_discloses`` render, if any.

    A no-op outside a render (the JSON path, a summary transform, a unit call to
    a helper), so coercion helpers stay usable everywhere."""
    seen = _SKEWED_FIELDS.get()
    if seen is not None and field not in seen:
        seen.append(field)


def _field_list(source: Any, *keys: str) -> list[Any]:
    """``source[key]`` as a list, recording the skew when the key is PRESENT but
    holds something that is neither ``None`` nor a list, at ANY depth.

    Three states, and the return value alone cannot carry all three -- it is a
    list, so ABSENT and PRESENT-AND-EMPTY both arrive as ``[]``. A renderer that
    must tell them apart asks ``_field_present``; it must never re-derive the
    answer from the raw payload, because a second definition of PRESENT drifts
    from this one (a round-6 repair did exactly that, and fabricated a count row
    for an explicit null). ABSENT is a missing key OR an explicit null -- nothing
    was claimed. PRESENT-AND-EMPTY is a real result: we looked and found none.
    PRESENT-BUT-WRONG-SHAPE is a skew to disclose. Testing the raw value for
    TRUTH instead of presence collapsed the third into the first for every FALSY
    wrong shape -- ``0``, ``""``, ``False``, a ``{}`` where a list belongs --
    which rendered a payload the renderer could not use as a confident empty
    result (#619).

    Extra ``keys`` are retained aliases (#651): the first that actually holds a
    non-empty list wins. ``source.get("items") or source.get("locals")`` instead
    short-circuits on a truthy MALFORMED canonical key, so the alias holding the
    real rows was never consulted and the renderer reported a confident empty
    listing (#619)."""
    src = _as_dict(source)
    rows: list[Any] = []
    for key in keys:
        if key not in src:
            continue
        raw = src[key]
        if isinstance(raw, list):
            if raw and not rows:
                rows = raw
        elif raw is not None:
            _record_skew(key)
    return rows


def _field_dict(source: Any, key: str) -> dict[str, Any]:
    """``source[key]`` as a dict, recording the skew when the key is PRESENT but
    holds something that is neither ``None`` nor a dict, at ANY depth. The dict
    mirror of ``_field_list``, including its three-state distinction (#619)."""
    src = _as_dict(source)
    if key not in src:
        return {}
    raw = src[key]
    if isinstance(raw, dict):
        return raw
    if raw is not None:
        _record_skew(key)
    return {}


# Two questions a renderer can ask about a container field, and they are NOT the
# same one. Both live here so a call site names which it means instead of
# spelling its own test: the whole defect class this module keeps re-growing is
# a SECOND place deciding a question the choke point already decides, which then
# drifts. A renderer that spelled PRESENT as ``key in source`` disagreed with the
# helpers about an explicit null and printed a confident zero for a payload that
# had claimed nothing (#619).
def _field_present(source: Any, key: str) -> bool:
    """Did ``source[key]`` CLAIM anything? The key is there AND is not an
    explicit null -- the same reading ``_field_list``/``_field_dict`` use, so
    "we looked and found none" is distinguishable from "nothing was said"."""
    src = _as_dict(source)
    return key in src and src[key] is not None


def _field_declared(source: Any, key: str) -> bool:
    """Does the ENVELOPE carry ``key`` at all, null included? A different
    question: it decides which SHAPE of payload arrived (a paged envelope versus
    a bare list, a callgraph with a callees section versus one without), not
    whether that field has contents. Null counts here and does not count for
    ``_field_present`` -- that is the distinction, stated once (#619)."""
    return key in _as_dict(source)


def _field_skewed(key: str) -> bool:
    """Was ``source[key]`` PRESENT but the WRONG SHAPE? The THIRD question, and
    the choke point is the only thing that can answer it.

    ``_field_list``/``_field_dict`` return the same empty container for a field
    that was genuinely empty and for one they could not use, and
    ``_field_present`` is True for both -- so a renderer that wanted "unusable"
    and spelled it ``_field_present`` was a SECOND decider over a question this
    module had already decided, and it answered wrong on the WELL-FORMED input:
    an empty ``definition`` object rendered as a raw Python repr where the
    diagnostic belonged, with nothing disclosed because nothing was wrong. This
    reads the record the choke point just wrote, so there is one answer (#619).

    Only meaningful inside a ``@_discloses`` boundary and after the read: a
    renderer asks it about a key it has already taken through the choke point.
    """
    return key in (_SKEWED_FIELDS.get() or ())


def _count_field(source: Any, key: str) -> int:
    """``source[key]`` as a count, recording the skew when the key is PRESENT but
    holds something no count can be read out of.

    The count sibling of ``_field_list``/``_field_dict``, and it exists for both
    of that pair's failure modes at once. ``int(source.get(key) or 0)`` RAISED on
    a string or a container -- each of `go rename`'s six counters cost the WHOLE
    summary, where the same payload with the counter ABSENT rendered cleanly --
    and quietly answering ``0`` instead is the other half of the same bug: a
    fabricated zero is indistinguishable from a real one, and a zero on this op
    is exactly the "nothing changed, don't save" reading that #683 discarded a
    rename batch to. A numeric string or float still reads, as it always did;
    anything else is a skew for the enclosing boundary to disclose (#619) --
    including a NON-FINITE number, which JSON can spell (``1e999`` decodes to
    ``inf``, and ``json.dumps`` round-trips it as ``Infinity``) and ``int()``
    answers with an ``ArithmeticError``. That is refused like any other
    unreadable shape rather than costing the whole render."""
    src = _as_dict(source)
    if key not in src:
        return 0
    raw = src[key]
    if raw is None:
        return 0                       # an explicit null claimed nothing
    if isinstance(raw, bool):
        _record_skew(key)              # a flag where a count belongs
        return 0
    if isinstance(raw, int):
        return raw
    try:
        return int(raw)
    except (ArithmeticError, TypeError, ValueError):
        _record_skew(key)
        return 0


def _stated_count(source: Any, key: str) -> str:
    """``_count_field`` for a line that STATES the number: the count when it
    read, and ``?`` when the key was present in a shape no count reads out of.

    The trailing ``@_discloses`` note is not enough on its own for a HEADLINE a
    caller acts on: "go rename: 0 renamed, 0 failed, 0 skipped" from an
    unreadable counter reads byte-identically to a genuine all-noop run, which
    is the #683 fabricated-zero harm with a footnote attached. The compact
    status already states an unreadable count as unknown (``changed=None``);
    this is that rule for the text views (#619/#685).
    """
    count = _count_field(source, key)
    return "?" if _field_skewed(key) else str(count)


def _text_value(source: Any, key: str) -> str | None:
    """``source[key]`` as TEXT, recording the skew when the key is PRESENT but
    holds something no text can be read out of.

    The string sibling of ``_field_list``/``_field_dict``/``_count_field``, and
    it exists for the same reason they do: a renderer that tests the value's
    shape inline drops the whole line it was going to render, and the row then
    reads byte-identically to a row where the payload said NOTHING. The live
    case: ``set_prototype``'s verification line is the renderer's whole subject
    -- did the prototype I set actually land -- and an ``observed`` object
    carrying a non-string ``prototype`` rendered exactly like an op that
    reported no observation at all, undisclosed (#619).

    ``None`` means "no text here", which covers ABSENT, an explicit null and
    PRESENT-AND-EMPTY: an empty string is a real answer, so it is not a skew and
    a caller may legitimately render nothing for it. Anything that is present
    and is not a string is the third state, and the enclosing boundary says so.
    """
    src = _as_dict(source)
    if key not in src:
        return None
    raw = src[key]
    if raw is None:
        return None                    # an explicit null claimed nothing
    if isinstance(raw, str):
        return raw or None
    _record_skew(key)
    return None


def _flag_field(source: Any, key: str) -> bool | None:
    """``source[key]`` as a FLAG, recording the skew when the key is PRESENT but
    holds something no boolean reads out of.

    The flag sibling of ``_count_field``/``_text_value``, and it exists for the
    one shape a raw truthiness test gets exactly BACKWARDS: the string
    ``"false"`` is True to Python. ``has_more`` decides whether a paged view
    states a resume instruction at all -- the one actionable claim in the
    footer -- so a ``"false"`` there printed "rerun with --offset 51" on the
    LAST page, sending a pager after a window that does not exist, while the
    three counts beside it had just been given a shape contract (#619).

    A real bool reads as itself and so does the 0/1 a bridge may send for a
    flag -- ``bool`` IS an ``int`` and a wire format that numbers its booleans
    is stating one, exactly as ``_count_field`` accepts a numeric string.
    ``None`` means "no flag here" (ABSENT or an explicit null, neither of which
    claimed anything); anything else present is the third state and the
    enclosing boundary says so.
    """
    src = _as_dict(source)
    if key not in src:
        return None
    raw = src[key]
    if raw is None:
        return None                    # an explicit null claimed nothing
    if isinstance(raw, (bool, int)):
        return bool(raw)
    _record_skew(key)
    return None


def _row_list(source: Any, key: str) -> list[dict[str, Any]]:
    """``source[key]``'s ROWS, recording the skew when an ELEMENT of it is
    present but is not a row anything here can read.

    The ELEMENT-granularity sibling of ``_field_list``, and it exists because
    the field-level answer was only half of one. Four sites spelled it
    ``[r for r in _field_list(v, "results") if isinstance(r, dict)]``; the
    fifth (`go rename`'s failure list) took the list raw and let the shared
    builder's own ``isinstance`` filter drop the element instead. Either way it
    is DISCARDED silently -- so a batch carrying one row
    nobody could classify beside one that verified reported ``ok: true``,
    ``failed_count: 0`` and ``first_error: null``, and the text card showed
    only the readable row with nothing anywhere saying a row had been dropped.
    That is #683's fabrication at the row set: the same payload with the whole
    ``results`` field unreadable is already refused, and a list of the WRONG
    ELEMENTS is the shape this module's own element sweep calls the one "an
    older bridge version actually sends".

    Skipping the element is still the right RENDER -- there is nothing in it to
    show -- so what this adds is the third state: the enclosing boundary names
    the field, ``_add_mutation_ok`` and the compact summary withhold ``ok``
    rather than claiming it, and a count derived from the survivors is no
    longer a measurement of the whole batch. Exactly the answer
    ``_is_failed_status`` gives for an unreadable row STATUS, asked one level
    out about the row itself (#619/#685).

    A ``None`` ELEMENT is a skew too, and that is not the field-level rule
    turned around: an explicit null FIELD claimed nothing, while a null INSIDE
    the list is a row position the sender filled with no row -- one op of the
    batch that this summary cannot account for, exactly like any other element
    it cannot read."""
    rows: list[dict[str, Any]] = []
    dropped = False
    for item in _field_list(source, key):
        if isinstance(item, dict):
            rows.append(item)
        else:
            dropped = True
    if dropped:
        _record_skew(key)
    return rows


def _is_failed_status(row: Any) -> bool:
    """Does this op result's status NAME a failure? The one place the question is
    answered -- and it answers it in THREE states, not two.

    Three sites spelled it ``str(row.get("status")) in FAILED_MUTATION_STATUSES``
    and a fourth tested the RAW value against the set, which is the same
    second-decider defect this module keeps re-growing -- and this time the two
    answers did not merely differ, the raw one RAISED: an unhashable status (a
    dict or list from a malformed or future bridge result) is ``TypeError:
    cannot use 'dict' as a set element`` and it cost the WHOLE mutation card,
    where the same row with ``status`` absent rendered cleanly (#619).

    But coercing with ``str()`` is not the fix either, and that is the half the
    crash hid. ``str({...})`` is not in the set, so an UNREADABLE status would
    answer "not a failure" -- indistinguishable from a row that genuinely
    passed, on the value ``ok`` and the summary's ``failed`` count are derived
    from. That is the fabricated-zero shape (#683) wearing a different coat.
    The status goes through the text choke point instead, so the third state
    survives: FAILED (True), NOT-FAILED (False, silent), UNREADABLE (False AND
    a recorded skew, which the enclosing boundary discloses and which
    ``_add_mutation_ok`` turns into a withheld ``ok`` rather than a pass).

    ABSENT and an explicit null claim nothing and stay silent, exactly as they
    did before -- base read them as "not a failure" too."""
    return _text_value(row, "status") in FAILED_MUTATION_STATUSES


def _discloses(fn: Callable[..., str] | None = None, *,
               prefix: bool = False) -> Callable[..., str]:
    """Append the skew disclosure for every container this text renderer coerced
    away, however deep it was read.

    Coercing a malformed field to ``{}``/``[]`` stops the AttributeError, but
    the result renders byte-identically to a genuinely EMPTY one -- the caller
    reads a confident "nothing here" from data the renderer could not use, which
    is worse than the crash it replaced: a truncated taint run loses its
    "truncated @depth N" clause and reads as complete, and a forward-taint view
    prints "NO modeled sink reached" -- a security all-clear -- from an unusable
    payload. Wrapping the whole render covers EVERY return path, including the
    early ones ("none", "no sessions", the no-possible-values return) that a
    per-branch line misses, and draining a recorded set rather than re-reading
    declared keys means there is no key list to drift from the code (#619).

    ``prefix=True`` marks a renderer whose output is concatenated AHEAD of
    another render -- the CLI builds `disasm` as note + steer + body -- so its
    note must END with a newline or it glues onto the next renderer's first
    line. Such a renderer being no boundary at all is the worse failure, and it
    was live: ``_resolution_note`` and ``_disasm_linear_steer_note`` read
    through the choke point, but every CLI caller invokes them BARE, so the skew
    they recorded landed in the default (absent) recorder and was dropped -- the
    containing-function note this module calls load-bearing vanished in silence
    on six text subcommands (#619)."""
    def decorate(inner: Callable[..., str]) -> Callable[..., str]:
        @functools.wraps(inner)
        def rendered(*args: Any, **kwargs: Any) -> str:
            token = _SKEWED_FIELDS.set([])
            try:
                out = inner(*args, **kwargs)
                skewed = sorted(_SKEWED_FIELDS.get() or ())
            finally:
                _SKEWED_FIELDS.reset(token)
            if not skewed or not isinstance(out, str):
                return out
            note = _skew_note(*skewed)
            if prefix:
                return out + note + "\n"
            return out + ("\n" if out else "") + note
        return rendered
    return decorate if fn is None else decorate(fn)


@contextlib.contextmanager
def disclosure_boundary() -> Iterator[list[str]]:
    """``@_discloses``, opened explicitly for a consumer that is NOT a renderer.

    The decorator is the boundary for everything that RETURNS text. The xrefs
    pipe note is the other shape: ``cli._call`` hands the SAME raw payload to a
    second consumer that answers a note-or-``None`` on stderr, and
    ``_record_skew`` is a no-op with no boundary on the stack -- so a skewed ref
    bucket coerced to ``[]``, counted zero caller groups, and the note reported
    "nothing was truncated" while the body renderer was disclosing that same
    payload as malformed. Yields the list the skew is recorded into, so the
    caller can state the third answer (unreadable) instead of the confident one.
    """
    skewed: list[str] = []
    token = _SKEWED_FIELDS.set(skewed)
    try:
        yield skewed
    finally:
        _SKEWED_FIELDS.reset(token)


def _discloses_in_summary(fn: Callable[..., Any]) -> Callable[..., Any]:
    """``@_discloses`` for a transform that returns the #685 summary DICT rather
    than text.

    The CLI runs a mutation's compact status as a `result_transform`: the
    transform consumes the RAW payload and the text renderer is handed its
    OUTPUT, so by the time ``_discloses`` installs a recorder the payload -- and
    with it every skew the choke point recorded off it -- is already gone. A
    present-but-malformed ``results[]`` therefore produced a status
    byte-identical to one built from no ``results[]`` at all: the confident
    answer from an unusable payload that ``_discloses`` exists to stop, one hop
    UPSTREAM of every renderer, on the DEFAULT mutation text path. ``_record_skew``
    called a summary transform a no-op context; it is the opposite -- the only
    context where the payload does not survive to a renderer (#619).

    Same drain, different carrier: the note travels in ``first_error``, the one
    summary key the agent contract already tells a control loop to read, so the
    text render and the JSON path both carry it and no new key joins the
    documented schema."""
    @functools.wraps(fn)
    def transformed(value: Any, *args: Any, **kwargs: Any) -> Any:
        token = _SKEWED_FIELDS.set([])
        try:
            out = fn(value, *args, **kwargs)
            skewed = sorted(_SKEWED_FIELDS.get() or ())
        finally:
            _SKEWED_FIELDS.reset(token)
        # `out is value` is the idempotent short-circuit returning the caller's
        # own dict: never mutate that, and it read nothing to disclose anyway.
        if not skewed or not isinstance(out, dict) or out is value:
            return out
        note = _skew_note(*skewed)
        existing = out.get("first_error")
        return {**out, "first_error": f"{existing} ({note})" if existing else note}
    return transformed


def _fmt_count(value: Any) -> str:
    """Render a keyed-aggregate count for a right-aligned column, disclosing an
    uncoercible one instead of raising inside the format spec: ``f"{None:>5}"``
    is a TypeError, so one malformed count in a breakdown costs the whole text
    view. The placeholder stays one character wide so a degraded row cannot skew
    the column it sits in, and no real count renders as ``?`` (#619)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return "?"
    return str(value)


def _fmt_field_offset(value: Any) -> str:
    """Render a struct-field offset, disclosing an uncoercible one instead of
    fabricating ``+0x0`` -- a zero is a real, common offset, so a degraded row
    must not be able to impersonate one (#619)."""
    if isinstance(value, int) and not isinstance(value, bool):
        return f"+0x{value:x}"
    return f"+<unknown: {value!r}>"


def _int_or_default(value: Any, default: int = 0) -> int:
    """Coerce a field expected to be an int, degrading a ``None``/non-numeric
    value to ``default`` instead of raising inside ``int(...)`` or an ordering
    comparison. Distinct from ``read_evidence._as_int``, which parses hex
    strings and returns ``-1`` on failure -- same-named helpers with
    different contracts are a trap, so this one names its defaulting
    behavior explicitly (#619)."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fmt_offset(value: Any) -> str:
    """Render a paging offset for a footer, disclosing an uncoercible one
    instead of fabricating a specific ``0`` the payload never stated. ``None``
    (an omitted offset) legitimately means "first page", so it alone renders
    as ``0`` (#619)."""
    if value is None:
        return "0"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return "<unknown>"


def _render_string_literal(value: Any, *, truncated: bool = False) -> str:
    text = json.dumps(value, ensure_ascii=True)
    if truncated:
        text += " [truncated]"
    return text


def _format_local_entry(item: dict[str, Any]) -> str:
    # Both cells are operator-set (`local rename` / `local retype`), so a control
    # char in either would split the `params:` / `locals:` row across two lines;
    # escape before the width padding so the column measures what actually prints
    # (#771). The type cell carries the same row, so it takes the same escaper --
    # one raw cell splits the row just as well as the name.
    name = _escape_control_chars(item.get("name", "<unknown>"))
    type_str = _escape_control_chars(item.get("type", "<unknown>"))
    line = f"  {name:<20} {type_str}"
    # local_id is the stable handle `local rename` / `local retype` take; show it
    # so the text view is self-sufficient and doesn't force a --format json
    # round-trip just to drive those commands (#122). Other internal fields
    # (storage / source / identifier) stay out of the slim view.
    local_id = item.get("local_id")
    if local_id:
        line += f"  [id: {local_id}]"
    return line


def _text_field(field: str) -> Callable[[Any], str]:
    def render(value: Any) -> str:
        if isinstance(value, dict):
            text = value.get(field)
            if isinstance(text, str):
                return text
        return _render_fallback_text(value)

    return render


def _slice_text_lines(
    text: str, lines_range: tuple[int, int] | None, *, marker: str = "//", flag: str = "--lines"
) -> str:
    """Return only lines START..END (1-indexed, inclusive) with a count header.

    Shared by `decompile`, `il`, `disasm`, and `structured-il` so every
    line-oriented view slices the same way. Slicing happens before the spill
    check, so `--lines` also keeps large functions inline. *flag* names the
    caller's flag in the out-of-range error so a `disasm --count N` failure
    reads `--count`, not `--lines` (#291.2 review).
    """
    if lines_range is None:
        return text
    all_lines = text.splitlines()
    total = len(all_lines)
    start, end = lines_range
    if start > total:
        # A start past the last line is a user error, not a result. Raising keeps
        # it from rendering as a `//` line (mistakable for code) and exiting 0:
        # BridgeError propagates to main() -> stderr diagnostic, non-zero exit, no
        # stdout a scripted consumer could read as a real slice (#253).
        raise BridgeError(
            f"{flag} start {start} is beyond the last line "
            f"(output has {total} line{'s' if total != 1 else ''}); "
            f"omit {flag} or choose a start within range"
        )
    sliced = all_lines[start - 1 : end]
    header = f"{marker} lines {start}-{min(end, total)} of {total}"
    return header + "\n" + "\n".join(sliced)


# A BOUNDARY, not just a reader: every CLI caller concatenates this note ahead of
# another render and none of them wraps it, so without one of its own the skew it
# records below reaches no recorder and the note disappears in silence (#619).
@_discloses(prefix=True)
def _resolution_note(value: Any) -> str:
    """A leading note when a function-scoped read annotated how it resolved the
    requested address (#193 Part 4).

    Two distinct disclosures share one `resolved_from` envelope, and text mode
    must say exactly what the JSON says:

    * a non-zero offset means an INTERIOR address resolved to its containing
      function -- without the note, text output silently shows a function whose
      start differs from what was asked, which reads like the wrong answer;
    * ``input_format: decimal`` means a digit-only token was read as an address
      rather than a symbol name. At offset ``+0x0`` that is the ONLY news: the
      read answered for exactly the function the caller named, so claiming
      containment there contradicts the payload.

    Returns '' when not applicable.
    """
    if not isinstance(value, dict):
        return ""
    # Through the choke point: a malformed resolution envelope used to drop this
    # whole note in silence, so a function-scoped read answered for a DIFFERENT
    # address than the caller asked for with the disclosure this docstring calls
    # load-bearing simply gone. An unusable envelope still yields no note -- it
    # states nothing it cannot support -- but the skew now reaches the render's
    # disclosure instead of disappearing (#619).
    resolved_from = _field_dict(value, "resolved_from")
    if not resolved_from:
        return ""
    function = _field_dict(value, "function")
    name = function.get("name", "?")
    address = function.get("address", "?")
    requested = resolved_from.get("requested_address")
    decimal = resolved_from.get("input_format") == "decimal"
    spelling = " (decimal input)" if decimal else ""
    if _is_exact_start(resolved_from):
        if not decimal:
            return ""
        return (
            f"// bn: decimal input resolved to {requested}, the exact start of "
            f"{name}\n"
        )
    return (
        f"// bn: {requested}{spelling} is inside {name} "
        f"@ {address} ({resolved_from.get('offset')}); showing the containing function\n"
    )


def _is_exact_start(resolved_from: dict[str, Any]) -> bool:
    """True when a `resolved_from` disclosure names the function's own start.

    Only a bare-decimal exact request is disclosed with a zero offset (so a
    digit-only token can't be mistaken for a symbol name); every containment
    resolution carries a non-zero offset.
    """
    offset = str(resolved_from.get("offset") or "")
    try:
        return int(offset.replace("+", "").replace("-", "") or "1", 16) == 0
    except ValueError:
        return False


# A BOUNDARY for the same reason as `_resolution_note`: `disasm` builds its text
# as note + steer + body, and nothing wraps the steer (#619).
@_discloses(prefix=True)
def _disasm_linear_steer_note(value: Any, *, sliced: bool) -> str:
    """A disasm-only note when a mid-function address was sliced (#371.3).

    `disasm <mid-addr> --count N` (or `--lines`) slices from the function
    PROLOGUE, not the requested address, so an agent inspecting a call site via
    an xref address silently gets the prologue. Point at `--linear`, which
    decodes N instructions from the exact address regardless of function
    membership. Fires only when a slice is active AND the address resolved
    mid-function -- an exact start or a whole-function dump has no trap. A
    bare-decimal EXACT start also carries `resolved_from` (offset +0x0) purely to
    disclose the spelling, and there the slice already begins at the requested
    address, so the steer would be false advice.
    """
    if not sliced or not isinstance(value, dict):
        return ""
    resolved_from = _field_dict(value, "resolved_from")
    if not resolved_from or _is_exact_start(resolved_from):
        return ""
    addr = resolved_from.get("requested_address", "?")
    return (
        f"// bn: --count/--lines slices from the function start, not {addr}; "
        f"to disassemble from {addr} itself use `disasm {addr} --linear N`\n"
    )


@_discloses
def _render_disasm_linear_text(value: Any) -> str:
    """Render a linear (non-function-bounded) disassembly: a leading `// bn:` note
    so it's clearly NOT a function listing, then the address/bytes/mnemonic lines
    (#314)."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    note = value.get("note")
    body = _text_field("text")(value)
    if note:
        return f"// bn: {note}\n{body}" if body else f"// bn: {note}"
    return body


@_discloses
def _render_capabilities_text(value: Any) -> str:
    """Render the #276 capability index as a grouped, scannable catalog: each
    top-level group, its commands with one-line help, and the prefer-when /
    see-also routing hints where a command overlaps a neighbor."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    items = _field_list(value, "items")
    lines: list[str] = []
    current_group: str | None = None
    for item in items:
        # A malformed list element (non-dict) must render as a placeholder line,
        # not crash the whole catalog with an AttributeError (#619).
        if not isinstance(item, dict):
            lines.append(f"  {item!r}")
            continue
        group = item.get("group", "")
        if group != current_group:
            if lines:
                lines.append("")
            lines.append(f"{group}:")
            current_group = group
        command = item.get("command", "")
        help_text = item.get("help", "")
        lines.append(f"  {command}  --  {help_text}" if help_text else f"  {command}")
        if item.get("prefer_when"):
            lines.append(f"      prefer when: {item['prefer_when']}")
        see_also = _field_list(item, "see_also")
        if see_also:
            lines.append(f"      see also: {', '.join(str(x) for x in see_also)}")
    return "\n".join(lines)


@_discloses
def _render_function_info_text(value: Any, verbose: bool = False, demangle: bool = False) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    function = _field_dict(value, "function")
    header_name = function.get("name", "<unknown>")
    if demangle and function.get("display_name"):
        header_name = function["display_name"]
    lines = [
        f"{_escape_control_chars(header_name)} @ {function.get('address', '<unknown>')}",
        str(value.get("prototype", "")),
        f"calling convention: {value.get('calling_convention', '<unknown>')}",
        f"size: {value.get('size', '<unknown>')}",
        f"xrefs: {value.get('xref_count', 0)}",
    ]

    locals_only = _field_list(value, "locals")
    if locals_only:
        lines.append(f"locals: {len(locals_only)} variables")

    # Surface unlifted instructions (#206) -- a function whose computation BN
    # couldn't model otherwise reads as fully analyzed.
    unimpl = _field_dict(value, "unimplemented_instructions")
    if unimpl.get("count"):
        addrs = _field_list(unimpl, "addresses")
        shown = ", ".join(str(addr) for addr in addrs)
        if unimpl.get("truncated"):
            shown += ", …"
        suffix = f" (e.g. {shown})" if shown else ""
        lines.append(
            f"unlifted instructions: {unimpl['count']} — BN could not model these; "
            f"dataflow through them is not tracked{suffix}")

    if verbose:
        parameters = _field_list(value, "parameters")
        if parameters:
            lines.append("")
            lines.append("parameters:")
            for item in parameters:
                if not isinstance(item, dict):
                    lines.append(f"  {item!r}")
                    continue
                lines.append(_format_local_entry(item))
        lines.append("")
        if locals_only:
            lines.append("locals:")
            for item in locals_only:
                if not isinstance(item, dict):
                    lines.append(f"  {item!r}")
                    continue
                lines.append(_format_local_entry(item))
        else:
            lines.append("locals: none")

    # PRESENT through the choke point: a present-but-wrong-shaped `blocks`
    # dropped the whole block listing, so a huge dispatcher rendered exactly
    # like a function whose blocks were never reported (#619). A present-and-
    # EMPTY list still prints the section, the way base printed it.
    blocks = _field_list(value, "blocks")
    if _field_present(value, "blocks"):
        # #653.10: block ranges make a huge dispatcher readable a region at a time
        # (`disasm --linear <start>` / `--lines` once you know where you are)
        # instead of forcing a 6000-line decompile first.
        lines.append("")
        lines.append(f"basic blocks: {len(blocks)}")
        for b in blocks:
            if not isinstance(b, dict):
                continue
            out = ", ".join(str(edge) for edge in _field_list(b, "outgoing")) or "-"
            # Rows are ADDRESS-ordered (so they are targetable), while `index` is
            # BN's own block index -- label it, or the out-of-order numbers read
            # as a sort bug rather than as CFG order.
            lines.append(
                f"  blk{_fmt_count(b.get('index', '?')):<4} {b.get('start', '?')}..{b.get('end', '?')}  "
                f"{b.get('length', '?')} bytes  -> {out}")

    return _resolution_note(value) + "\n".join(lines)


@_discloses
def _render_proto_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    prototype = value.get("prototype")
    if not isinstance(prototype, str):
        return _render_fallback_text(value)
    # BN renders the prototype anonymously (`uint64_t (int32_t arg1)`); splice in
    # the function name so the output is a copy-pasteable C declaration (#222).
    note = _resolution_note(value)
    fn = value.get("function")
    name = fn.get("name") if isinstance(fn, dict) else None
    head, sep, rest = prototype.partition("(")
    if name and sep:
        head = head.rstrip()
        # Skip only when the return-type already ENDS with the name as its own
        # declarator token (an already-named prototype) -- a naive `name in head`
        # substring test wrongly skipped when the name was a substring of the
        # return type (e.g. name "t" in "uint64_t") (#222 review).
        already_named = head.split()[-1:] == [name] if head else False
        if not already_named:
            return note + f"{head} {name}({rest}"
    return note + prototype


@_discloses
def _render_local_list_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    function = _field_dict(value, "function")
    # #651: `items` is the canonical container; `locals` is the retained alias.
    all_items = _field_list(value, "items", "locals")
    # A malformed (non-dict) element cannot say whether it is a parameter, so it
    # is counted as neither -- disclosing it separately keeps both counts honest
    # instead of raising or folding it into a group it may not belong to (#619).
    params = [item for item in all_items if isinstance(item, dict) and item.get("is_parameter")]
    locals_only = [item for item in all_items
                   if isinstance(item, dict) and not item.get("is_parameter")]
    malformed = [item for item in all_items if not isinstance(item, dict)]

    # The header name is escaped exactly like `function info`'s header (#370.1):
    # the same cell must not split this row just because a different renderer
    # prints it (#771).
    header = (f"{_escape_control_chars(function.get('name', '<unknown>'))} @ "
              f"{function.get('address', '<unknown>')}")
    header += f" ({len(params)} params, {len(locals_only)} locals)"
    lines = [header]

    if params:
        lines.extend(["", "params:"])
        for item in params:
            lines.append(_format_local_entry(item))
    if locals_only:
        lines.extend(["", "locals:"])
        for item in locals_only:
            lines.append(_format_local_entry(item))
    if malformed:
        lines.extend(["", f"malformed entries ({len(malformed)}):"])
        for item in malformed:
            lines.append(f"  {item!r}")
    if not params and not locals_only and not malformed:
        lines.extend(["", "no locals"])
    return _resolution_note(value) + "\n".join(lines)


@_discloses
def _render_type_info_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    layout = value.get("layout")
    if isinstance(layout, str) and layout:
        return layout
    decl = value.get("decl")
    if isinstance(decl, str) and decl:
        return decl
    return _render_fallback_text(value)


@_discloses
def _render_field_xrefs_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    field = _field_dict(value, "field")
    lines = [
        f"{field.get('type_name', '<unknown>')}.{field.get('field_name', '<unknown>')} @ {_fmt_field_offset(field.get('offset'))}",
        f"type: {field.get('field_type', '<unknown>')}",
        "",
        "code refs:",
    ]
    # #275: refs come as a unified `items` list, each tagged with its `kind`.
    items = _field_list(value, "items")
    code_refs = [it for it in items if isinstance(it, dict) and it.get("kind") == "code"]
    data_refs = [it for it in items if isinstance(it, dict) and it.get("kind") == "data"]
    # A non-dict ref carries no `kind`, so both filters above drop it. Disclose
    # it as its own group rather than letting it leave the inventory silently.
    malformed_refs = [it for it in items if not isinstance(it, dict)]
    if code_refs:
        for ref in code_refs:
            details = [str(ref.get("address", "<unknown>"))]
            if ref.get("function"):
                details.append(str(ref["function"]))
            if ref.get("incoming_type"):
                details.append(f"type={ref['incoming_type']}")
            if ref.get("disasm"):
                details.append(str(ref["disasm"]))
            lines.append("- " + " | ".join(details))
    else:
        lines.append("- none")

    lines.extend(["", "data refs:"])
    if data_refs:
        for ref in data_refs:
            # Every part is str()'d before the join: the code-ref block above
            # already learned that one wrong-typed part costs the WHOLE listing,
            # and the data-ref block was the same bug with the lesson missing.
            details = [str(ref.get("address", "<unknown>"))]
            if ref.get("symbol"):
                details.append(str(ref["symbol"]))
            if ref.get("type"):
                details.append(f"type={ref['type']}")
            lines.append("- " + " | ".join(details))
    else:
        lines.append("- none")

    if malformed_refs:
        lines.extend(["", f"malformed refs ({len(malformed_refs)}):"])
        for ref in malformed_refs:
            lines.append(f"- {ref!r}")

    # #532: field xrefs now page like every other xref path. Surface the paging
    # metadata whenever the page isn't the whole ref set -- either more pages remain
    # (has_more) or an --offset skipped earlier refs -- so a partial view (including
    # the last, has_more=False page of an --offset run) isn't read as the full set.
    total = value.get("total")
    returned = value.get("returned", len(items))
    offset = value.get("offset", 0) or 0
    has_more = bool(value.get("has_more"))
    if isinstance(total, int) and (has_more or offset or returned != total):
        note = f"showing {returned} of {total} refs (offset {offset})"
        if has_more:
            note += "; more available -- raise --limit or use --offset"
        lines.extend(["", note])

    return "\n".join(lines)


@_discloses
def _render_comment_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    # `comment get --function` aggregates all in-function comments as a list (#203),
    # and, alongside them, the whole-function documentation (`fn.comment`) as
    # `function_doc` -- surfaced above the address comments so the two stores read
    # as one coherent view instead of the doc silently going missing from `get`.
    # The aggregate view is what a CLAIMED `comments` selects, asked as PRESENT
    # through the choke point: a present-but-wrong-shaped one used to fall
    # through to the single-comment form and render "(no comment)" -- byte
    # identical to a payload that never mentioned comments at all (#619).
    comments = _field_list(value, "comments")
    if _field_present(value, "comments"):
        lines = []
        doc = value.get("function_doc")
        if doc:
            # The doc rides at the top of this listing as a row (the same store
            # `comment list` marks `[doc] `), so a multi-line doc gets the same
            # escaping as the address rows below (#771).
            lines.append(f"[doc] {_escape_control_chars(doc)}")
        if not comments and not doc:
            return "(no comment)"
        lines.extend(
            f"{c.get('address', '?')}  {_escape_control_chars(c.get('comment', ''))}"
            for c in comments if isinstance(c, dict)
        )
        return "\n".join(lines)
    comment = value.get("comment")
    if isinstance(comment, str):
        # Single-comment form (`comment get <addr>`): the payload IS the comment,
        # a document rather than a row, so its own newlines are the content and
        # stay raw -- same as the decompile/IL/type-layout text renderers (#771).
        return comment if comment else "(no comment)"
    return _render_fallback_text(value)


@_discloses
def _render_comment_list_text(value: Any) -> str:
    # Paged envelope ({items,total,...}) -> render the page + the shared footer;
    # a bare list falls through to the per-item body below (back-compat) (#131).
    if _field_declared(value, "items"):
        return _render_paged_list_text(value, "items", _render_comment_list_text)
    if not isinstance(value, list):
        return _render_fallback_text(value)
    if not value:
        return "none"
    lines = []
    for item in value:
        if not isinstance(item, dict):
            lines.append(_render_fallback_text(item))
            continue
        address = item.get("address", "<unknown>")
        func = item.get("function") or "<global>"
        comment = item.get("comment", "")
        # #643: mark function documentation comments so the two stores are
        # distinguishable in one listing -- same `[doc]` marker `comment get
        # --function` already uses.
        prefix = "[doc] " if item.get("scope") == "function_doc" else ""
        # Both cells carry settable text -- `comment` via `comment set`, and the
        # containing function's symbol name via `rename`, which accepts a control
        # char (`_require_nonempty_name` rejects only empty/whitespace). Either
        # one raw splits the row (#771).
        lines.append(f"{address}  {_escape_control_chars(func)}  {prefix}"
                     f"{_escape_control_chars(comment)}")
    return "\n".join(lines)


@_discloses
def _render_tag_types_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    types = _field_list(value, "tag_types")
    if not types:
        return "none"
    lines = []
    for t in types:
        if not isinstance(t, dict):
            lines.append(_render_fallback_text(t))
            continue
        builtin = "  [builtin]" if t.get("is_builtin") else ""
        # Both cells are operator-set and pass to the view with no charset check
        # (`tag type create <name> [--icon]`), so they take the same escaper as
        # the tag row -- one raw cell splits the row (#771).
        lines.append(f"{_escape_control_chars(t.get('icon', ''))}  "
                     f"{_escape_control_chars(t.get('name', '<unknown>'))}{builtin}")
    return "\n".join(lines)


@_discloses
def _render_tag_get_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    tags = _field_list(value, "tags")
    if not tags:
        return "(no tags)"
    return "\n".join(_render_tag_row(t) for t in tags if isinstance(t, dict))


def _render_tag_row(t: dict) -> str:
    # A function-scope tag has no address; show the function name it belongs to
    # instead of a bare placeholder. The JSON already carries `function`, so the
    # text renderer just surfaces it (address scope keeps the address, which is
    # the more precise locator when both are present).
    loc = t.get("address") or t.get("function") or "<function>"
    # Every settable cell carries text that can split the row: `data` via
    # `tag add --data`, `type`/`icon` via `tag type create` (both reach the view
    # with no charset check), and the function-scope `loc` is a symbol name
    # (`rename` accepts a control char). Each takes the escaper -- one raw cell
    # splits the row just as well as `data` did (#771).
    return (f"{_escape_control_chars(loc)}  [{t.get('scope', '?')}]  "
            f"{_escape_control_chars(t.get('icon', ''))} "
            f"{_escape_control_chars(t.get('type', ''))}  "
            f"{_escape_control_chars(t.get('data', ''))}")


@_discloses
def _render_tag_list_text(value: Any) -> str:
    if _field_declared(value, "items"):
        return _render_paged_list_text(value, "items", _render_tag_list_text)
    if not isinstance(value, list):
        return _render_fallback_text(value)
    if not value:
        return "none"
    return "\n".join(_render_tag_row(t) for t in value if isinstance(t, dict))


@_discloses
def _render_refresh_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    target = value.get("target")
    if isinstance(target, dict):
        return f"refreshed: true\n\n{_render_target_summary(target)}"
    return _render_fallback_text(value)


@_discloses
def _render_load_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    suffix = "  [not analyzed]" if value.get("analyzed") is False else ""
    lines = [f"loaded: {value.get('path', '<unknown>')}{suffix}"]
    for note in _field_list(value, "notes"):
        lines.append(f"note: {note}")
    targets = _field_list(value, "targets")
    if targets:
        lines.append("")
        lines.append("targets:")
        for t in targets:
            if isinstance(t, dict):
                lines.append("- " + str(t.get("selector") or t.get("basename") or "<unknown>"))
            else:
                lines.append("- " + _render_fallback_text(t))
    return "\n".join(lines)


@_discloses
def _render_close_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    closed = _field_list(value, "closed")
    if not closed:
        return "no binaries closed"

    def _row(entry: Any) -> tuple[str, bool]:
        if isinstance(entry, dict):
            return str(entry.get("path", "")), bool(entry.get("unsaved"))
        return str(entry), False

    rows = [_row(e) for e in closed]
    unsaved_any = any(unsaved for _, unsaved in rows)

    if len(rows) == 1:
        path, unsaved = rows[0]
        lines = [f"closed: {path}"]
    else:
        lines = ["closed:"]
        for path, unsaved in rows:
            marker = "  [unsaved changes discarded]" if unsaved else ""
            lines.append(f"- {path}{marker}")

    if unsaved_any:
        lines.append("")
        lines.append(
            "warning: unsaved mutations were discarded. "
            "use `bn save` before `bn close` to persist them."
        )
    return "\n".join(lines)


@_discloses
def _render_save_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    line = f"saved: {value.get('path', '<unknown>')}"
    collision = _field_dict(value, "collides_with_open_target")
    if collision:
        # #857 r4: a save landing on a file another target already has open makes
        # the two targets one database, so a later restart returns one target for
        # both. Not silent.
        line += (
            "\nnote: also open as target "
            f"{collision.get('selector', '<unknown>')}; the two targets are now "
            "one database and a session restart will return one for both"
        )
    if value.get("fallback"):
        # The default path was unwritable (e.g. a read-only firmware mount); the
        # database landed in the writable cache instead (#214).
        line += (f"\nnote: {value.get('requested_path')} was not writable; "
                 f"saved to the cache instead")
    return line


@_discloses
def _render_session_start_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    lines = [
        f"instance: {value.get('instance_id', '<unknown>')}",
        f"pid: {value.get('pid', '<unknown>')}",
        f"socket: {value.get('socket_path', '<unknown>')}",
    ]
    loaded = _field_list(value, "loaded")
    if loaded:
        lines.append("")
        lines.append("loaded:")
        for item in loaded:
            if isinstance(item, dict):
                error = item.get("error")
                if error:
                    # #857 r4 minor: name the path actually TRIED. A restart
                    # reloads a saved target from its backing database, so
                    # printing only the target's filename pointed the reader at a
                    # file that was never opened and hid the missing database.
                    attempted = item.get("attempted_path")
                    where = f"{item.get('path', '<unknown>')}"
                    if attempted and attempted != item.get("path"):
                        where += f" (tried {attempted})"
                    lines.append(f"- {where} [error: {error}]")
                elif value.get("detached"):
                    lines.append(
                        f"- {item.get('path', '<unknown>')} "
                        f"[job {item.get('job_id', '?')} "
                        f"{item.get('state', 'queued')}]"
                    )
                    if item.get("status_command"):
                        lines.append(f"  poll: {item['status_command']}")
                else:
                    mark = "  [not analyzed]" if item.get("analyzed") is False else ""
                    lines.append(f"- {item.get('path', '<unknown>')}{mark}")
                    for tgt in _field_list(item, "targets"):
                        if isinstance(tgt, dict) and tgt.get("selector"):
                            lines.append(
                                f"  target: {tgt['selector']}"
                                f"   (pass -t {tgt['selector']}; id {tgt.get('target_id', '?')})")
                for note in _field_list(item, "notes"):
                    lines.append(f"  note: {note}")
            else:
                lines.append(f"- {_render_fallback_text(item)}")
    project_roots = _field_list(value, "project_roots")
    if project_roots:
        lines.append("")
        lines.append(f"projects: {', '.join(str(root) for root in project_roots)}")
    association_error = value.get("project_association_error")
    if association_error:
        lines.append("")
        lines.append(f"project association error: {association_error}")
        lines.append(
            f"hint: bare `bn` commands from this checkout won't route to this "
            f"session; pass -i {value.get('instance_id', '<id>')}"
        )
    if value.get("reload_capture_failed"):
        lines.append("")
        lines.append(
            f"target capture error: {value.get('reload_capture_error', '<unknown>')}"
        )
        lines.append(
            "hint: open targets could not be listed before the restart, so none were reloaded"
        )
    if value.get("stopped"):
        lines.append("")
        lines.append("session stopped: no binaries loaded successfully")
    return "\n".join(lines)


@_discloses
def _render_session_status_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    items = _field_list(value, "items")
    if not items:
        return "no load jobs"
    lines = []
    for item in items:
        if not isinstance(item, dict):
            lines.append(_render_fallback_text(item))
            continue
        lines.append(
            f"{item.get('job_id', '<unknown>')}  "
            f"{item.get('state', '<unknown>')}  "
            f"{item.get('path', '<unknown>')}"
        )
        if item.get("error"):
            lines.append(f"  error: {item['error']}")
        result = _field_dict(item, "result")
        if result:
            for target in _field_list(result, "targets"):
                if isinstance(target, dict) and target.get("selector"):
                    lines.append(f"  target: {target['selector']}")
    # A job-specific poll that has NOT finished names the exact command to
    # re-run, so text mode never leaves the caller reconstructing it. Terminal
    # jobs drop the hint: re-polling a finished job is pure waste and would
    # contradict `terminal: true`.
    status_command = value.get("status_command")
    if status_command and value.get("terminal") is False:
        lines.append(f"  poll: {status_command}")
    return "\n".join(lines)


@_discloses
def _render_session_stop_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    line = f"stopped: {value.get('instance_id', '<unknown>')}"
    method = value.get("method")
    if method:
        line += f" ({method})"
    return line


@_discloses
def _render_session_list_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    instances = _field_list(value, "items", "instances")
    if not instances:
        return "no sessions"
    lines = []
    for item in instances:
        if not isinstance(item, dict):
            lines.append(_render_fallback_text(item))
            continue
        head = str(item.get("selector") or item.get("instance_id") or "<unknown>")
        if item.get("sticky"):
            head += " [sticky]"
        parts = [head, f"pid={item.get('pid', '<unknown>')}"]
        rss = item.get("rss_mb")
        if rss is not None:
            parts.append(f"rss={rss}MB")
        if item.get("started_at"):
            parts.append(f"started={item['started_at']}")
        lines.append("  ".join(parts))
        if item.get("socket_path"):
            lines.append(f"  socket: {item['socket_path']}")
        binaries = _field_list(item, "binaries")
        if binaries:
            lines.append(f"  open: {', '.join(str(b) for b in binaries)}")
        # #733 F1: read directly, NOT through `_stated_count` -- that helper
        # reads an explicit null as a real 0, which is exactly the fabricated
        # "nothing would be discarded" this field exists to avoid.
        unsaved = item.get("unsaved_targets")
        if isinstance(unsaved, int) and not isinstance(unsaved, bool):
            lines.append(f"  unsaved targets: {unsaved}")
        elif item.get("unsaved_targets_unavailable"):
            lines.append(
                f"  unsaved targets: unknown — {item['unsaved_targets_unavailable']}"
            )
        project_roots = _field_list(item, "project_roots")
        if project_roots:
            lines.append(
                f"  projects: {', '.join(str(root) for root in project_roots)}"
            )
    total_rss = value.get("total_rss_mb")
    if total_rss is not None and instances:
        lines.append("")
        lines.append(f"total rss: {total_rss}MB")
    return "\n".join(lines)


@_discloses
def _render_instance_find_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    items = _field_list(value, "items")
    if not items:
        return f"no instance has a binary matching {value.get('query')!r}"
    lines = []
    for item in items:
        if not isinstance(item, dict):
            lines.append(f"{item!r}")
            continue
        selector = str(item.get("selector") or item.get("instance_id") or "<unknown>")
        lines.append(f"{selector}  (instance {item.get('instance_id')})")
        lines.append(f"  {item.get('binary')}")
    return "\n".join(lines)


def _pointer_size_label(size: Any) -> str:
    """``4`` -> ``4 bytes``; anything not a usable byte count renders as nothing.

    A bogus or absent width must not print as a confident ``0 bytes`` -- the
    detail rows drop empty values, so an older bridge that doesn't report the
    field simply omits the line.
    """
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 8:
        return ""
    return f"{size} byte{'s' if size != 1 else ''}"


def _yes_no(flag: Any) -> str:
    """``True``/``False`` -> ``yes``/``no``; anything else renders as nothing, so
    an older bridge that does not report the field simply omits the row."""
    return "yes" if flag is True else "no" if flag is False else ""


def _render_target_summary(value: dict[str, Any]) -> str:
    view_id = value.get("view_id")
    label = value.get("selector") or value.get("target_id") or "<unknown>"
    prefix = f"[{view_id}] " if view_id is not None else ""
    lines = [f"{prefix}{label}"]
    if value.get("active"):
        lines[0] += " [active]"
    if value.get("sticky"):
        lines[0] += " [sticky]"
    # Flag a --quick (unanalyzed) view in text, not just JSON, so a cold agent
    # doesn't trust an empty/partial result from a view whose analysis is pending.
    if value.get("analyzed") is False:
        lines[0] += " [not analyzed]"
    # #733 F1: the destructive question ("will closing this discard work?")
    # answered where a reader is already looking.
    if value.get("unsaved") is True:
        lines[0] += " [unsaved]"

    details = [
        ("target", value.get("target_id")),
        ("view", value.get("view_id")),
        ("kind", value.get("view_name")),
        # Surface analysis_state (full/quick) in text too -- the bn-re methodology
        # tells agents to gate their survey on it, and on a quick-loaded view it
        # explains an apparently-empty result rather than "empty binary" (#378).
        ("analysis", value.get("analysis_state")),
        # The committed-but-unsaved ledger and BN's generic modified bit, both
        # rendered as text so an absent key drops out of the detail rows while a
        # real `False` still prints (#733 F1).
        ("unsaved", _yes_no(value.get("unsaved"))),
        ("engine modified", _yes_no(value.get("engine_modified"))),
        ("file", value.get("filename")),
        ("arch", value.get("arch")),
        ("platform", value.get("platform")),
        # Pointer width + byte order, so a text-mode reader decoding a raw dump
        # doesn't have to infer them from the arch *name* (which says nothing for
        # an architecture it hasn't memorized). Rendered as bytes, matching the
        # `address_size` field rather than restating it in bits.
        ("pointer size", _pointer_size_label(value.get("address_size"))),
        ("endianness", value.get("endianness")),
        # Preferred/image base BN loaded at (#564) -- for a PIE binary this is the
        # rebase anchor a debugger handoff needs, so surface it beside entry.
        ("image base", value.get("image_base")),
        ("entry", value.get("entry_point")),
    ]
    for key, item in details:
        if item not in (None, ""):
            lines.append(f"\t{key}: {item}")
    # Live analysis phase/counts while a long `bn refresh` runs (#321), so a text
    # user can watch a large-target analysis on another connection instead of
    # guessing it's wedged. Shown for any *active* phase -- some phases (Discovery,
    # ExtendedAnalyze) legitimately report 0/0 -- and hidden only when idle/not
    # started, so the line doesn't flicker away mid-analysis. Counts appended when
    # a meaningful total is known.
    prog = _field_dict(value, "analysis_progress")
    state = prog.get("state")
    if state and state not in ("IdleState", "InitialState"):
        total = prog.get("total") or 0
        counts = f" {prog.get('count')}/{total}" if total else ""
        lines.append(f"\tanalysis progress: {state}{counts}")
    # Function-count + named-vs-auto-named summary that every agent reaches for
    # first (#122). Counts reflect the current analysis state (a --quick view
    # reports what it has so far; analysis_state already flags that).
    fn_count = value.get("function_count")
    if fn_count is not None:
        summary = f"{fn_count} functions"
        named = value.get("named_function_count")
        unnamed = value.get("unnamed_function_count")
        if named is not None and unnamed is not None:
            parts = [f"{named} named", f"{unnamed} auto-named"]
            imported = value.get("imported_function_count")
            if imported:
                parts.append(f"{imported} imported")
            summary += f" ({', '.join(parts)})"
        lines.append(f"\tfunctions: {summary}")
    import_symbols = value.get("import_symbol_count")
    if import_symbols is not None:
        imported_functions = value.get("imported_function_count")
        callable_suffix = (
            f", {imported_functions} callable function targets"
            if imported_functions is not None
            else ""
        )
        lines.append(
            f"\timports: {import_symbols} symbols{callable_suffix}"
        )
    # Segment-level detail only when present -- target info --verbose adds it;
    # target list rows do not, so this stays out of the list view. (F21)
    segments = _field_list(value, "segments")
    if segments:
        lines.append("\tsegments:")
        for seg in (s for s in segments if isinstance(s, dict)):
            perms = "".join(
                flag if seg.get(name) else "-"
                for name, flag in (("readable", "r"), ("writable", "w"), ("executable", "x"))
            )
            lines.append(
                f"\t\t{seg.get('start')}-{seg.get('end')} {perms} ({seg.get('length')} bytes)"
            )
    return "\n".join(lines)


@_discloses
def _render_target_list_text(value: Any) -> str:
    # Accept both the {kind, items} envelope (#358) and a bare list (older
    # callers / raw socket clients).
    if _field_declared(value, "items"):
        # Called for its RECORDING side effect, not its value: a malformed
        # listing must reach the disclosure, but this renderer then echoes the
        # raw payload below, which is strictly MORE information than a note --
        # so the echo keeps the raw value and the skew is recorded anyway (#619).
        _field_list(value, "items")
        items = _as_dict(value)["items"]
    else:
        items = value
    if not isinstance(items, list):
        return _render_fallback_text(value)
    if not items:
        return "no targets"
    return "\n\n".join(
        _render_target_summary(item) if isinstance(item, dict) else _render_fallback_text(item)
        for item in items
    )


@_discloses
def _render_target_info_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    return _render_target_summary(value)


def _render_target_choice(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    return target_row(value)


def _render_target_choices(value: Any) -> str:
    """The multi-target hint block, rendered from a `list_targets` reply.

    Byte-identical to what the bridge resolver prints for the same condition
    (#688): one grammar, `src/bn/target_hint.py`, so an agent that learned the
    pre-flight refusal parses the resolver's too. Row rendering stays per-item
    here because these rows come off the wire and a non-dict one must degrade
    rather than raise.
    """
    if not isinstance(value, list):
        return _render_fallback_text(value)
    if not value:
        return "none"
    return "\n".join(open_target_lines(value, row=_render_target_choice))


@_discloses
def _render_instance_use_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    line = f"instance: {value.get('instance_id', '<unknown>')}"
    cleared = value.get("cleared_target_pin")
    if cleared:
        line += f"\ncleared stale target pin {cleared!r} (belonged to the previous instance)"
    return line


@_discloses
def _render_target_use_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    return f"target: {value.get('target', '<unknown>')}"


@_discloses
def _render_pin_clear_text(value: Any) -> str:
    """Render `instance clear` / `target clear` confirmations."""
    return "cleared"


@_discloses
def _render_instance_gc_text(value: Any) -> str:
    """Render the `instance gc` cache-cleanup summary."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    # Counts are summed, so a non-numeric one raised a TypeError and cost the
    # whole text view -- the same crash #619 replaced everywhere else, still live
    # here because these fields are scalars rather than containers.
    # Through the count choke point: silently defaulting these to 0 stopped the
    # TypeError but made an unreadable count render as a confident "0 reaped",
    # which is the other half of the same bug (#619).
    # `_stated_count`, because every one of these numbers is STATED: "reaped 0
    # logs" from an unreadable counter reads byte-identically to a real zero,
    # and the caller's reading of it ("nothing was reaped") is a decision the
    # trailing disclosure note arrives too late for.
    logs = _stated_count(value, "logs_removed")
    socks = _stated_count(value, "sockets_removed")
    last = _stated_count(value, "last_used_removed")
    regs = _stated_count(value, "registries_purged")
    live = value.get("live_instances", 0)
    # The same rule twice on purpose: `_stated_count` for what is PRINTED, the
    # int for the "anything at all?" test. A refused counter must not be summed
    # into a confident "nothing to reap" either -- that is the same claim as a
    # fabricated zero, one branch up. (`_count_field` re-reads are idempotent;
    # `_record_skew` dedupes.)
    reaped = (_count_field(value, "logs_removed")
              + _count_field(value, "sockets_removed")
              + _count_field(value, "last_used_removed")
              + _count_field(value, "registries_purged"))
    if reaped == 0 and not any(_field_skewed(key) for key in
                               ("logs_removed", "sockets_removed",
                                "last_used_removed", "registries_purged")):
        return f"gc: nothing to reap ({live} live instance{'' if live == 1 else 's'})"
    return (
        f"gc: reaped {logs} log{'' if logs == '1' else 's'}, "
        f"{socks} orphan socket{'' if socks == '1' else 's'}, "
        f"{last} last-used sidecar{'' if last == '1' else 's'}, "
        f"{regs} dead registr{'y' if regs == '1' else 'ies'} "
        f"({live} live instance{'' if live == 1 else 's'} kept)"
    )


def _render_name_address_rows(value: Any, *, demangle: bool = False) -> str:
    """Render a BARE list of name/address rows (imports, function pages). With
    ``demangle``, show the demangled ``display_name`` instead of the raw name so
    a C++ listing is greppable/clusterable without c++filt (#196)."""
    if not isinstance(value, list):
        return _render_fallback_text(value)
    if not value:
        return "none"

    lines = []
    for item in value:
        if not isinstance(item, dict):
            lines.append(_render_fallback_text(item))
            continue
        address = item.get("address", "<unknown>")
        name = item.get("name") or item.get("function") or "<unknown>"
        if demangle and item.get("display_name"):
            name = item["display_name"]
        line = f"{address}  {_escape_control_chars(name)}"
        kind = item.get("kind")
        if kind and kind != "function":
            line += f" ({kind})"
        library = item.get("library")
        if library:
            line += f" [{_escape_control_chars(library)}]"
        raw_name = item.get("raw_name")
        if raw_name and raw_name != name:
            # Both extra columns carry target-supplied text (an import's library
            # and the pre-demangle symbol), so escape them like the name cell
            # above -- one raw cell splits the row just as well (#771). The
            # `raw_name != name` test still compares the RAW values.
            line += f" (raw: {_escape_control_chars(raw_name)})"
        size = item.get("size")
        if size is not None:
            # #411: surface basic_block_count (a real complexity metric) here too,
            # since text is the DEFAULT read output -- otherwise agents only ever
            # see the misleading byte span. Omit the blocks clause when the count
            # is absent/None (e.g. an older bridge, or a guarded bad function).
            blocks = item.get("basic_block_count")
            if blocks is not None:
                line += f"  ({size} bytes, {blocks} blocks)"
            else:
                line += f"  ({size} bytes)"
        lines.append(line)
    return "\n".join(lines)


@_discloses
def _render_name_address_list_text(value: Any) -> str:
    """Render imports: the paged {items, total, ...} envelope (with a footer),
    or a bare list for back-compat / internal callers (#122)."""
    body = _render_paged_list_text(value, "items", _render_name_address_rows)
    # Surface PIC self-references dropped from the survey so the exclusion isn't
    # silent (#202).
    excluded = value.get("self_defined_excluded") if isinstance(value, dict) else None
    if isinstance(excluded, int) and excluded > 0:
        note = (
            f"// {excluded} self-defined export(s) excluded "
            "(this module's own symbols modeled as import veneers / GOT slots)"
        )
        body = note if body == "none" else f"{body}\n{note}"
    return body


@_discloses
def _render_go_rename_text(value: Any) -> str:
    """Render `go rename` (#217): a compact summary (verified / failed / skipped
    counts) with the preview/rollback banner and a capped FAILED list -- never a
    per-success wall, since a bulk auto-name renames hundreds/thousands at once.
    `results` carries only failures; the applied names are in the database."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    # Through the COUNT choke point, for the three counters this renderer
    # actually reads. The CLI installs it as the `--verbose` DETAIL view for
    # `go rename` (the compact status is the default since #645), and it read
    # `skipped_user_named`, `go_renamed_candidates` and `go_verified_count`
    # through a helper that silently defaults an unreadable count to 0 and
    # records nothing. A candidate counter arriving in the wrong shape
    # therefore rendered the confident "nothing to do -- no auto-named Go
    # functions to rename" for a batch that had just committed 1783 renames,
    # undisclosed: the #683 harm again, in the sibling of the transform that
    # was fixed for it. A second count contract beside the choke point is how
    # this keeps coming back, so there is now one (#619). The other three
    # counters -- `go_committed_count`, `go_failed_count` and
    # `skipped_changed_during_apply` -- this view never reads at all, which is
    # its own disclosure gap and is routed by name (see the ROUTED block in
    # `tests/test_cli_formatters.py`, item 1).
    # `_stated_count` for the three counters this view INTERPOLATES: an
    # unreadable one renders `?`, because "go rename: 0 renamed, 0 failed, 0
    # skipped" is byte-identical to a genuine all-noop run and the headline is
    # the line a caller acts on -- the trailing disclosure note arrives after
    # the decision. `targeted` stays an int: it GATES control flow below and
    # already has the stronger whole-line refusal.
    skipped = _stated_count(value, "skipped_user_named")
    targeted = _count_field(value, "go_renamed_candidates")
    if not targeted:
        if _field_skewed("go_renamed_candidates"):
            # "nothing to do" is an ACTIONABLE claim, not a missing detail: a
            # caller reads it and stops. A fabricated 0 must never be allowed
            # to make it, so say what is actually known instead -- the note the
            # boundary appends says which field could not be read.
            return ("go rename: cannot say what this run did -- the candidate "
                    "count was unreadable, so neither the renames nor the "
                    "skips can be reported; re-read with --format json")
        return ("go rename: nothing to do — no auto-named (sub_*) Go functions to rename "
                f"({_stated_count(value, 'defined_count')} defined at pcln addresses, "
                f"{skipped} already user-named)")
    failed = _row_list(value, "results")
    # The default is a MEASUREMENT (targeted minus the failures), so it is used
    # only when the envelope claimed no verified count at all -- asking
    # `_count_field` for an absent key would fabricate a 0 over it.
    verified = (_stated_count(value, "go_verified_count")
                if _field_present(value, "go_verified_count")
                else str(targeted - len(failed)))
    preview = bool(value.get("preview"))
    committed = bool(value.get("committed", True))
    lines: list[str] = []
    if preview:
        lines.append("preview: renames applied + reverted (nothing committed)")
        lines.append(f"go rename (preview): {verified} would rename, {len(failed)} failed, "
                     f"{skipped} skipped (already user-named)")
    elif not committed and (value.get("success") is False or failed):
        # All-or-nothing: a failure reverted the WHOLE batch, so NOTHING landed --
        # don't claim "N renamed" for rows that passed readback before the revert.
        if value.get("rolled_back") is False:
            lines.append("rollback failed: the view may be left modified")
        else:
            lines.append("rolled back: the batch was reverted because a rename failed — "
                         "NOTHING was committed")
        lines.append(f"go rename: 0 renamed ({verified} would have, {len(failed)} failed, "
                     f"{skipped} skipped); fix the failure(s) below and re-run")
    else:
        lines.append(f"go rename: {verified} renamed, {len(failed)} failed, "
                     f"{skipped} skipped (already user-named)")
    for r in failed[:50]:
        lines.append(f"  failed: {r.get('new_name', '?')} @ {r.get('address', '?')} "
                     f"({r.get('status', '?')})")
    if len(failed) > 50:
        lines.append(f"  ... and {len(failed) - 50} more failed")
    return "\n".join(lines)


@_discloses
def _render_go_functions_text(value: Any) -> str:
    """Render the Go pcln function lens (#217): a header with the detected Go
    version + how many recovered addresses already map to a BN function, the
    optional PIE-rebase note, then the name/address rows + paging footer."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    head = "go functions"
    if value.get("go_version"):
        head += f" ({value['go_version']})"
    parts = []
    if isinstance(value.get("total"), int):
        parts.append(f"{value['total']} recovered")
    if isinstance(value.get("defined_count"), int):
        parts.append(f"{value['defined_count']} mapped to a BN function")
    if parts:
        head += ": " + ", ".join(parts)
    lines = [head]
    if value.get("truncated"):
        # #528: a partial walk must not read as a complete count.
        lines.append(
            f"warning: partial pcln walk -- {value.get('recovered')} of "
            f"{value.get('expected')} declared functions recovered "
            f"({value.get('skipped')} skipped/truncated); the table is malformed or truncated."
        )
    if value.get("note"):
        lines.append(f"note: {value['note']}")
    lines.append(_render_paged_list_text(value, "items", _render_name_address_rows))
    return "\n".join(lines)


@_discloses
def _render_go_functions_summary_text(value: Any) -> str:
    """#414: compact go-metadata summary -- enough to decide whether to run
    `go rename` without listing every function."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    head = "go functions summary"
    if value.get("go_version"):
        head += f" ({value['go_version']})"
    lines = [head]
    for label, key in (("recovered", "recovered"), ("defined", "defined"),
                       ("undefined", "undefined"), ("renamable", "renamable")):
        if isinstance(value.get(key), int):
            lines.append(f"  {label}: {value[key]}")
    if value.get("truncated"):
        # #528: disclose that the declared table was only partially recovered.
        lines.append(
            f"  expected: {value.get('expected')} (partial walk -- "
            f"{value.get('skipped')} skipped/truncated)"
        )
    ts, tsb = value.get("text_start"), value.get("text_start_bv")
    if ts is not None:
        rebase = "" if (tsb is None or tsb == ts) else f"  (BN text {tsb} -- rebase needed)"
        lines.append(f"  text_start: {ts}{rebase}")
    return "\n".join(lines)


# The conditions under which a page's three counts cannot describe any window,
# stated as a SET rather than as a chain of comparisons inside the footer.
#
# The first cut of this refusal was one comparison, `offset + returned > total`,
# and it was incomplete on its own stated terms: a readable NEGATIVE count
# walked past it and printed "showing -5 of 100 (105 more); rerun with --offset
# -5" -- a remaining count larger than the whole set and a resume offset no
# pager can use, with every count readable so no shape refusal fired either. A
# chain also cannot be enumerated: a test can only re-list the comparisons it
# believes are there, which is the second-declaration drift this module keeps
# growing. A named set can be walked, so
# `test_the_paging_footer_refuses_every_impossible_page_it_names` asserts each
# member actually refuses and is NAMED in the refusal -- a member that is
# deleted because a sibling happens to catch the same payload still fails (#619).
_IMPOSSIBLE_PAGE: tuple[tuple[str, Callable[[int, int, int], bool]], ...] = (
    ("total is negative", lambda total, returned, offset: total < 0),
    ("returned is negative", lambda total, returned, offset: returned < 0),
    ("offset is negative", lambda total, returned, offset: offset < 0),
    ("offset + returned exceeds total",
     lambda total, returned, offset: offset + returned > total),
)


def _paging_footer(value: dict[str, Any], items: list[Any],
                   page_unreadable: bool) -> str | None:
    """Build the "// showing N of TOTAL" footer for a paged-list envelope.

    Shared by every paged list renderer (function list/search, strings, imports,
    sections) so the honest-total convention reads identically across them (#59,
    #122). Returns None when the page IS the whole set (no paging happened), the
    envelope lacks a total to report against, or any field this footer reads
    arrived in a shape it could not be read out of -- the three counts, the
    `has_more` flag that decides whether a resume instruction exists at all, or
    the PAGE ITSELF. A footer states no count and no resume offset it could not
    derive (#619).

    Counts that are all readable but cannot describe any window (`_IMPOSSIBLE_PAGE`)
    get a stated refusal naming each condition instead of a footer, so a
    self-contradicting envelope does not render like an unpaged one.

    *page_unreadable* is the third state of the page, which only the caller can
    answer because the page key is a runtime argument."""
    # A page the caller could not read makes the third count a fabrication:
    # `returned` defaults to the length of *items*, and that is a MEASUREMENT
    # only while the page was readable -- the choke point hands back an empty
    # list for a page it could not use, so an unreadable page rendered
    # "showing 0 of 100 (50 more); rerun with --offset 50" from an offset of
    # 50, the non-advancing resume loop again, reached without any of the three
    # named counts being wrong. The skew is already recorded by the caller's
    # read, so the boundary still names the field (#619). Refused BELOW rather
    # than here, so that ONCE A TOTAL IS PRESENT the same keys are consulted
    # whichever refusal fires -- the no-total return above reads only `total`,
    # which is the one path that has nothing to footer against at all.

    # ONE count contract for all three of these. Spelling the total's test as
    # `isinstance(total, int)` made it a THIRD one: it rejected the numeric
    # string `_count_field` accepts two lines below, and which the go-rename
    # counter test asserts must NOT disclose, so a bridge reporting counts as
    # strings got "! malformed total" and no footer at all. Ask the choke point
    # instead -- and ask it the three-state question, because "absent" is not
    # "unusable": an envelope with no total has nothing to footer against and
    # nothing to disclose, while a present-but-unreadable one drops the footer
    # AND says so. Dropping it silently made a skewed envelope render
    # byte-identically to an unpaged one (#619).
    if not _field_present(value, "total"):
        return None
    # All three inside ONE capture, and every one of them refusing. The
    # comment above claimed one count contract for the three counts this footer
    # names, and only `total` actually had it: `returned` and `offset` reach
    # ARITHMETIC, so a skew there was absorbed into the 0 the choke point
    # returns and the footer went on to state a resume instruction DERIVED from
    # that fabrication -- `--offset 1` for a page that began at 50 (a pager
    # re-reads the window it already holds), or `--offset 0` with `has_more`
    # true, which does not ADVANCE at all and loops an unattended pager forever
    # on page one. A count nobody could derive is not stated, exactly as the
    # op row, the type row and the blast-radius line refuse theirs; the
    # enclosing boundary names the field. `returned` keeps its item-count
    # default, which is a measurement of a page the caller established was
    # readable, so it is asked for only when the envelope actually claimed one.
    #
    # Nested rather than asking `_field_skewed` three times: the ambient set
    # holds every skew this render recorded, including a `total`/`offset` key
    # on some unrelated ROW, and a footer suppressed by another field's skew
    # would be the mirror fabrication (#619).
    token = _SKEWED_FIELDS.set([])
    try:
        total = _count_field(value, "total")
        returned = (_count_field(value, "returned") if _field_present(value, "returned")
                    else len(items))
        offset = _count_field(value, "offset")
        # The flag goes through a choke point too, and inside the same capture:
        # it decides whether the footer states a resume instruction at all, so
        # a raw truthiness test on it was the one shape that reads BACKWARDS --
        # `has_more: "false"` is True to Python and printed "rerun with
        # --offset 51" on the last page.
        more = _flag_field(value, "has_more")
        unreadable = sorted(_SKEWED_FIELDS.get() or ())
    finally:
        _SKEWED_FIELDS.reset(token)
    for key in unreadable:
        _record_skew(key)
    if unreadable or page_unreadable:
        return None
    # The three counts must also agree with EACH OTHER, and with themselves.
    # Readable and impossible put a negative remaining count and a negative
    # `--offset -4` into an actionable instruction, and that arithmetic is not
    # a measurement of anything. Every refused condition is NAMED in the line,
    # and the payload's own numbers are stated instead of a derivation from
    # them, for the same reason `go rename` refuses a counter its own rows
    # contradict -- and stated rather than dropped, because a silent drop
    # renders a self-contradicting envelope byte-identically to an unpaged one.
    # The conditions live in `_IMPOSSIBLE_PAGE` so they can be enumerated by a
    # test rather than re-listed by one (#619).
    impossible = [name for name, holds in _IMPOSSIBLE_PAGE
                  if holds(total, returned, offset)]
    if impossible:
        return (f"// page position not stated: {'; '.join(impossible)} "
                f"(total {total}, returned {returned}, offset {offset})")
    if more:
        remaining = total - (offset + returned)
        next_offset = offset + returned
        return (
            f"// showing {returned} of {total} ({remaining} more); "
            f"rerun with --offset {next_offset} or a larger --limit"
        )
    if offset or returned != total:
        return f"// showing {returned} of {total}"
    return None


@_discloses
def _render_paged_list_text(
    value: Any, page_key: str, item_renderer: Callable[[Any], str]
) -> str:
    """Render a paged-list envelope ({<page_key>, total, ...}) with a footer.

    Delegates the body to *item_renderer* (which renders a bare list) and
    appends the shared paging footer. Falls back to rendering *value* as a bare
    list when it isn't an envelope, so internal callers or older bridges that
    still hand over a plain list keep working (#122)."""
    if not _field_declared(value, page_key):
        return item_renderer(value)  # back-compat / fallback for a bare list
    # The page key is a runtime argument, so this one site stands in for six
    # listing renderers -- and a raw `or []` here was invisible to the coercion
    # guard precisely because the key is not a literal (#619).
    items = _field_list(value, page_key)
    # The third state of the PAGE, asked HERE because the page key is a runtime
    # argument and this is the only site that knows it -- and asked BEFORE the
    # item renderer runs, so a row's own same-named field cannot answer for the
    # page. A footer measured off a page nobody could read is the fabrication
    # `_paging_footer` refuses for its three named counts (#619).
    page_unreadable = _field_skewed(page_key)
    body = item_renderer(items)
    footer = _paging_footer(value, items, page_unreadable)
    if footer is None:
        return body
    return footer if body == "none" else f"{body}\n\n{footer}"


_QUICK_PARTIAL_WARNING = (
    "WARNING: target is quick-loaded; {what} is partial. "
    "Run `bn refresh` for full analysis."
)


def _quick_partial_prefix(value: Any, what: str = "function list/count") -> str:
    """A leading warning line when a read envelope is quick-loaded/partial
    (#437), so a text reader doesn't trust a partial answer as a complete one.
    Empty for a fully-analyzed view.

    ``what`` names the partial artifact in the reader's own terms (#820): every
    op that answers on a quick view now carries the same `partial` flag, so the
    one message would otherwise tell a `decompile`/`class list`/`types`/
    `evidence function` reader that a "function list/count" was partial."""
    if isinstance(value, dict) and value.get("partial"):
        return _QUICK_PARTIAL_WARNING.format(what=what) + "\n"
    return ""


@_discloses
def _render_function_count_text(value: Any, *, label: str = "Total functions",
                                what: str = "function list/count") -> str:
    """Render a `function list/search --count` result, prefixing the quick-load
    partiality warning when the count is partial (#437).

    ``what`` names the artifact in the warning text (#820); callers that reuse
    this renderer for non-function counts (e.g. ``types --count``) should pass
    a matching ``what`` so the warning is self-describing.

    #653.1: `function search <q> --count` used the SAME "Total functions:" label as
    the whole-binary `function list --count`, so "Total functions: 17" beside
    "Total functions: 175" read as a contradiction rather than as matches vs total.
    """
    count = value.get("count", 0) if isinstance(value, dict) else 0
    return f"{_quick_partial_prefix(value, what)}{label}: {count}"


@_discloses
def _render_function_list_text(value: Any, *, demangle: bool = False) -> str:
    """Render a paged function listing, with a footer stating the true total and
    remainder (#59). Prefers the canonical `items` key (every other list command
    uses it), falling back to the deprecated byte-identical `functions` alias for
    an older bridge that emits only the latter (#223). ``demangle`` shows the
    demangled display_name (#196). A quick-loaded (partial) listing is prefixed
    with a warning so the page isn't mistaken for the whole binary (#437)."""
    page_key = "items" if _field_declared(value, "items") else "functions"
    return _quick_partial_prefix(value) + _render_paged_list_text(
        value, page_key, lambda items: _render_name_address_rows(items, demangle=demangle))


def _group_refs_by_caller(refs: list[Any]) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[str, Any]] = {}
    order: list[tuple] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        # Through the choke point, so a malformed `caller_function` discloses
        # itself rather than silently grouping the ref as if it had no
        # containing function. The second test asks a different question from
        # `_field_present` -- "did it hold a usable dict?", not "did it claim
        # anything?" -- so a PRESENT but empty dict still takes the function
        # branch as it did before, while a malformed one keeps grouping by its
        # own label. Using presence here instead would merge refs with DIFFERENT
        # malformed callers into one group and lose their separate contexts,
        # which is the collapsing this grouping exists to prevent (#619).
        caller = _field_dict(ref, "caller_function")
        key: tuple
        if caller or isinstance(ref.get("caller_function"), dict):
            key = ("fn", str(caller.get("address")), str(caller.get("name")))
            caller_address = caller.get("address")
            caller_name = caller.get("name")
        else:
            # No containing function: group by the ref's own resolved label
            # (symbol, else section) so refs under DIFFERENT symbols/sections do
            # not collapse into one group that gets stamped with the first
            # ref's label (which mislabeled the others). A label-less ref is
            # never coalesced -- it keys on its own address so each renders as a
            # distinct "<unknown>" line with its own context.
            fn_label = ref.get("function")
            label = (_unknown_ref_label(_field_dict(ref, "context"))
                     or ("" if fn_label is None else str(fn_label)))
            if label:
                key = ("label", label)
            else:
                key = ("site", str(ref.get("address", "<unknown>")))
            caller_address = None
            caller_name = label or None
        if key not in groups:
            groups[key] = {
                "caller_address": caller_address,
                "caller_name": caller_name,
                "sites": [],
                "context": _field_dict(ref, "context") or None,
            }
            order.append(key)
        groups[key]["sites"].append(str(ref.get("address", "<unknown>")))
    return [groups[k] for k in order]


def _unknown_ref_label(context: Any) -> str:
    """A concise fallback label (symbol name, else section name) for a ref that
    has no containing function, so it isn't rendered as a bare "<unknown>".

    Reads through the choke point: a skewed `symbol` here silently downgraded the
    ref to "<unknown>", which reads as "this ref belongs to nothing" rather than
    "the label could not be read" (#619)."""
    if not isinstance(context, dict):
        return ""
    symbol = _field_dict(context, "symbol")
    if symbol.get("name"):
        return str(symbol["name"])
    for item in _field_list(context, "sections"):
        if isinstance(item, dict) and item.get("name"):
            return str(item["name"])
    return ""


def _xref_buckets(value: dict[str, Any]) -> tuple[list[Any], list[Any], int, int]:
    """Split an xrefs response into ``(code_refs, data_refs, total_code, total_data)``.

    Tolerates both shapes: the deprecated dual arrays (still embedded by
    ``function info`` and emitted by field xrefs) and the #184 items-only ``xrefs``
    op response (reconstruct the buckets by splitting ``items`` on ``kind``).
    Totals come from the full-set summary counts (``code_ref_count`` /
    ``data_ref_count``) when present, so the header stays honest even though the op
    no longer ships the full arrays."""
    code_refs = value.get("code_refs")
    data_refs = value.get("data_refs")
    if code_refs is None and data_refs is None:
        items = _field_list(value, "items")
        code_refs = [r for r in items if isinstance(r, dict) and r.get("kind") == "code"]
        data_refs = [r for r in items if isinstance(r, dict) and r.get("kind") == "data"]
    else:
        # A skewed bucket used to be counted element by element -- one "ref" per
        # CHARACTER in the header -- and then rendered as "- none" below it (#619).
        code_refs = _field_list(value, "code_refs")
        data_refs = _field_list(value, "data_refs")
    total_code = value.get("code_ref_count")
    total_code = len(code_refs) if total_code is None else total_code
    total_data = value.get("data_ref_count")
    total_data = len(data_refs) if total_data is None else total_data
    return code_refs, data_refs, total_code, total_data


@_discloses
def _render_xrefs_any_text(value: Any) -> str:
    """Render the multi-symbol sink-sweep (`xrefs --any`): one line per symbol,
    present (with counts) or absent (#218)."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    syms = _field_list(value, "items")  # #275: was `symbols`
    lines = [f"xrefs --any: {value.get('present', 0)}/{value.get('count', len(syms))} symbol(s) present"]
    for s in syms:
        if not isinstance(s, dict):
            continue
        if s.get("present"):
            lines.append(
                f"  {s.get('symbol')}: {s.get('code_ref_count', 0)} code refs across "
                f"{s.get('caller_function_count', 0)} fn(s)  @ {s.get('address', '?')}")
        else:
            lines.append(f"  {s.get('symbol')}: absent")
    return "\n".join(lines)


@_discloses
def _render_xrefs_text(value: Any, limit: int | None = None) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    code_refs, data_refs, total_code, total_data = _xref_buckets(value)

    import_label = ""
    if value.get("import_resolved"):
        scanned = " (scanned)" if value.get("code_refs_scanned") else ""
        import_label = f"import: {value.get('import_name', '<unknown>')}{scanned}"

    def _render_group(refs: list[Any], total: int, label: str) -> list[str]:
        groups = _group_refs_by_caller(refs)
        if not groups:
            return [f"{label}:", "- none"]
        site_word = "site" if total == 1 else "sites"
        # The groups are distinct CALLERS, which may be functions OR function-less
        # locations (a data ref whose caller_function is null). Only call them
        # "functions" when every group really is one; otherwise use the neutral
        # "locations" so a function-less bucket isn't miscounted as a function (#49).
        # A function group carries a caller_address; a function-less one has None
        # (its caller_name holds a fallback symbol/section label, so that field
        # can't be the discriminator).
        all_functions = all(group.get("caller_address") for group in groups)
        grp_singular = "function" if all_functions else "location"
        grp_plural = "functions" if all_functions else "locations"
        grp_word = grp_singular if len(groups) == 1 else grp_plural
        header = f"{label}: {total} {site_word} across {len(groups)} {grp_word}"
        shown = groups[:limit] if limit else groups
        rendered = [header]
        for group in shown:
            caller_addr = group["caller_address"] or "<unknown>"
            caller_name = (
                group["caller_name"]
                or _unknown_ref_label(_field_dict(group, "context"))
                or "<unknown>"
            )
            sites = group["sites"]
            if len(sites) == 1:
                suffix = f"(1 site: {sites[0]})"
            else:
                suffix = f"({len(sites)} sites: {', '.join(sites)})"
            rendered.append(f"  {caller_addr}  {caller_name}  {suffix}")
        if limit and len(groups) > limit:
            rendered.append(
                f"  ... {len(groups) - limit} more {grp_plural} "
                "(increase --limit or use --format json for all)"
            )
        return rendered

    lines = [f"xrefs to {value.get('address', '<unknown>')} ({total_code} code, {total_data} data)", ""]
    if import_label:
        lines.insert(0, import_label)
        lines.insert(1, "")
    # Ambiguous same-name collision (thunk/real): surface the note so a zero-caller
    # member is never mistaken for dead code (#220).
    amb = _field_dict(value, "ambiguous_symbol")
    if amb.get("note"):
        lines.insert(0, f"note: {amb['note']}")
        lines.insert(1, "")
    # Data-symbol resolution fallback (#224b).
    rsym = _field_dict(value, "resolved_symbol")
    if rsym.get("kind") == "data":
        lines.insert(0, f"note: resolved '{rsym.get('name')}' as a data symbol @ {rsym.get('address')}")
        lines.insert(1, "")
    lines.extend(_render_group(code_refs, total_code, "code refs"))
    lines.append("")
    lines.extend(_render_group(data_refs, total_data, "data refs"))
    # #622's honesty fields reached the JSON envelope only: a caller scan that
    # stopped at its budget rendered byte-identically to a complete one, so an
    # empty list read as "no callers" (see the fn_pointer_scan_truncated note
    # below for the same rule on the evidence card). Both reads go through the
    # choke point, so a SKEWED flag or note is disclosed rather than dropped.
    if _flag_field(value, "truncated"):
        note = _text_value(value, "scan_note")
        lines.append("")
        lines.append(
            "note: the caller scan was TRUNCATED -- this list is partial, so an "
            "empty or short result is NOT proof there are no callers"
            + (f" ({note})" if note else "")
        )
    return "\n".join(lines)


def _context_suffix(context: Any) -> str:
    """The `| section=… | seg=rwx | symbol=…` tail on a ref row.

    Every sub-field goes through the choke point. They used to be `.get()` plus
    an `isinstance` filter, which is the #619 defect one container down: a ref
    whose `symbol` or `sections` arrived skewed rendered a row with NO context at
    all -- byte-identical to a ref that genuinely has none -- and the caller
    reads the absence as fact. The filters stay (a wrong shape still renders
    nothing rather than garbage); what changes is that the skew is now RECORDED,
    so the render says so."""
    if not isinstance(context, dict):
        return ""
    parts = []
    sections = _field_list(context, "sections")
    if sections:
        names = [str(item.get("name", "")) for item in sections if isinstance(item, dict) and item.get("name")]
        if names:
            parts.append("section=" + ",".join(names))
    segment = _field_dict(context, "segment")
    if segment:
        perms = (
            ("r" if segment.get("readable") else "-")
            + ("w" if segment.get("writable") else "-")
            + ("x" if segment.get("executable") else "-")
        )
        parts.append(f"seg={perms}")
    symbol = _field_dict(context, "symbol")
    if symbol.get("name"):
        sym_type = symbol.get("type")
        if sym_type:
            parts.append(f"symbol={symbol['name']}[{sym_type}]")
        else:
            parts.append(f"symbol={symbol['name']}")
    string = _field_dict(context, "string")
    if string.get("value"):
        enc = string.get("encoding")
        label = "string" if enc in (None, "ascii") else f"string({enc})"
        parts.append(
            f"{label}={_render_string_literal(string['value'], truncated=bool(string.get('truncated')))}"
        )
    # Through the text choke point, not an inline isinstance: a
    # container-shaped `disasm` dropped the clause and left the ref row
    # byte-identical to a ref carrying NO context at all, undisclosed, on six
    # renderers -- the same defect as the prototype leaf, at the sub-field the
    # docstring above claimed already went through it (#619).
    disasm = _text_value(context, "disasm")
    if disasm:
        parts.append(f"disasm={disasm}")
    return " | " + " | ".join(parts) if parts else ""


@_discloses
def _render_evidence_xrefs_text(value: Any, limit: int | None = None) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    lines = [f"xrefs to {value.get('address', '<unknown>')}"]
    suffix = _context_suffix(_field_dict(value, "target_context"))
    if suffix:
        lines.append(f"target{suffix}")

    code_refs, data_refs, total_code, total_data = _xref_buckets(value)
    for nice, kind, refs, total in (
        ("code refs", "code", code_refs, total_code),
        ("data refs", "data", data_refs, total_data),
    ):
        shown = refs[:limit] if limit else refs
        lines.append("")
        # Report the true total and a truncation marker when capped, matching
        # the honesty convention used by strings/function list/evidence message
        # -- a bare cap with no "N more" would hide that refs exist (#31).
        if limit and total > len(shown):
            lines.append(f"{nice}: {total} total, showing first {len(shown)}")
        else:
            lines.append(f"{nice}:")
        if not shown:
            lines.append("- none")
            continue
        for ref in shown:
            if not isinstance(ref, dict):
                lines.append("- " + _render_fallback_text(ref))
                continue
            address = ref.get("address", "<unknown>")
            function = ref.get("function") or "<unknown>"
            ref_kind = ref.get("kind") or kind
            # A stored function pointer the back-link scan discovered (#323): mark
            # it [function pointer] so an analyst sees it's a scan-found callback
            # table slot, distinct from a BN-modeled data ref (the section is in
            # the context suffix).
            fp = " [function pointer]" if ref.get("function_pointer") else ""
            lines.append(
                f"- {address}  {ref_kind}  {function}{_context_suffix(_field_dict(ref, 'context'))}{fp}"
            )
    if value.get("fn_pointer_scan_truncated"):
        lines.append(
            "\nnote: the function-pointer back-link scan was truncated (data "
            "sections exceeded the scan budget); some table references may be missing"
        )
    return "\n".join(lines)


def _render_target_line(target: Any) -> str:
    if not isinstance(target, dict):
        return "<unknown>"
    if target.get("status") == "unmapped":
        raw = target.get("raw") or "<unknown>"
        return f"{raw} [unmapped/non-pointer]"
    if target.get("status") == "null":
        return "0x0 [null]"
    raw = target.get("raw")
    normalized = target.get("normalized")
    fn = _field_dict(target, "function")
    if fn.get("name"):
        fn_address = fn.get("address", normalized or raw or "<unknown>")
        base = f"{fn.get('name')} @ {fn_address}"
        if fn.get("exact_start") is False:
            offset = fn.get("offset")
            if offset:
                base += str(offset) if str(offset).startswith("-") else f"+{offset}"
            actual = normalized or raw
            if actual and actual != fn_address:
                base += f" (target {actual}, not start)"
            else:
                base += " (not start)"
    else:
        addr = normalized or raw or "<unknown>"
        context = _field_dict(target, "context")
        # Through the choke point: a skewed `symbol`/`string`/`sections` here
        # silently downgraded a resolved target to a bare address, which reads
        # as "this pointer names nothing" (#619).
        symbol = _field_dict(context, "symbol")
        string = _field_dict(context, "string")
        sections = _field_list(context, "sections")
        section_name = None
        if sections and isinstance(sections[0], dict):
            section_name = sections[0].get("name")
        if symbol.get("name"):
            base = f"{symbol['name']} @ {addr}"
            annot = [str(a) for a in (section_name, symbol.get("type")) if a]
            if annot:
                base += f" [{', '.join(annot)}]"
        elif string.get("value"):
            enc = string.get("encoding")
            base = json.dumps(string["value"], ensure_ascii=True)
            annot = [
                str(a)
                for a in (
                    section_name,
                    (enc if enc and enc != "ascii" else None),
                    ("truncated" if string.get("truncated") else None),
                )
                if a
            ]
            if annot:
                base += f" [{', '.join(annot)}]"
        else:
            base = str(addr)
    if raw and normalized and raw != normalized:
        base += f" (raw {raw})"
    if target.get("thumb_adjusted"):
        base += " [thumb-adjusted]"
    if target.get("plausible") is False:
        base += " [low-confidence]"
    return base


@_discloses
def _render_function_evidence_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    function = _field_dict(value, "function")
    lines = [
        f"{function.get('name', '<unknown>')} @ {function.get('address', '<unknown>')}",
        f"prototype: {value.get('prototype', '<unknown>')}",
        f"calling convention: {value.get('calling_convention', '<unknown>')}",
    ]
    # #820: the view-level analysis state leads the card. Prepended HERE rather
    # than at the final join because an empty call set returns early -- and a
    # quick-loaded read with no calls is exactly the card a reader would take for
    # a complete answer.
    quick_prefix = _quick_partial_prefix(value, "function evidence")
    if quick_prefix:
        lines.insert(0, quick_prefix.rstrip("\n"))
    thunk = _field_dict(value, "thunk")
    # The target envelope goes through the choke point too: a malformed one used
    # to vanish from the card because the raw truth test never told anyone the
    # payload was unusable (#619).
    thunk_target = _field_dict(thunk, "target")
    if thunk.get("is_candidate"):
        lines.append(f"thunk: candidate ({thunk.get('reason', 'no reason recorded')})")
        if thunk_target:
            lines.append(f"  target: {_render_target_line(thunk_target)}")
    elif thunk_target:
        lines.append("thunk: no (tail branch to a local function, not a trampoline)")
        lines.append(f"  tail branch -> {_render_target_line(thunk_target)}")
    else:
        lines.append("thunk: no")

    # The deferral the bridge records when a sliced read skipped the Pseudo-C
    # decompile: it writes the sentence into `warnings` AND sets the flag, and
    # TEXT mode printed neither -- it dropped the whole `warnings` list, so a
    # sliced card read like a full-fidelity one. Placed before the early
    # `return` on an empty call set, so it is reached either way.
    warnings = _field_list(value, "warnings")
    for warning in warnings:
        lines.append(f"warning: {warning}")
    if _flag_field(value, "decompile_deferred") and not warnings:
        # The flag without its sentence: the payload said the decompile was
        # skipped and carried no text saying so, so state it here rather than
        # render a sliced card that reads like a full-fidelity one. Guarded on
        # `not warnings` by design -- the bridge already writes the deferral
        # sentence there, and printing both states one claim twice.
        lines.append("warning: Pseudo-C decompile deferred for this sliced read; "
                     "decompiler warnings were not collected -- re-read unsliced "
                     "for full fidelity")

    calls = _field_list(value, "calls")
    lines.append("")
    # #471: show the slice window when the call set was paged/windowed.
    total_calls = value.get("total_calls")
    matched = value.get("matched_calls")
    call_hdr = f"calls: {len(calls)}"
    if isinstance(total_calls, int) and (
        value.get("offset") or value.get("limit") is not None or matched != total_calls
    ):
        call_hdr += f" of {matched if matched is not None else total_calls}"
        if matched is not None and matched != total_calls:
            call_hdr += f" in window (of {total_calls} total)"
        if value.get("has_more"):
            # Through the count choke point, like every other arithmetic read of
            # a paging counter (see the paged-listing footer): `int(x or 0)`
            # RAISED on a string and on a container, and it cost the WHOLE
            # evidence card where the same payload with `offset` absent rendered
            # cleanly (#619). The choke point answers 0 and discloses instead.
            nxt = _count_field(value, "offset") + len(calls)
            if _field_skewed("offset"):
                # An ACTIONABLE number, not a descriptive one: a fabricated
                # resume offset sends an agent paging from a window the payload
                # never stated, which is the invented-offset harm the paging
                # footer was fixed for (#722). Refuse the hint instead.
                call_hdr += " -- more: page position unreadable, re-read with --format json"
            else:
                call_hdr += f" -- more: rerun with --offset {nxt}"
    lines.append(call_hdr)
    if not calls:
        return "\n".join(lines)

    for call in calls:
        if not isinstance(call, dict):
            lines.append("- " + _render_fallback_text(call))
            continue
        call_addr = call.get("address", "<unknown>")
        operation = call.get("operation", "<unknown>")
        direct = "direct" if call.get("direct") else "indirect"
        lines.append(f"- {call_addr}  {operation}  {direct}")
        target = _field_dict(call, "target")
        if target:
            lines.append(f"  target: {_render_target_line(target)}")
        instr = _field_dict(call, "call_instruction")
        if instr:
            lines.append(f"  instruction: {instr.get('address', call_addr)}  {instr.get('text', '')}".rstrip())
        if call.get("hlil_statement"):
            lines.append(f"  hlil: {call['hlil_statement']}")
        elif call.get("hlil_statement_reason"):
            # #557: expose WHY the HLIL statement is null.
            lines.append(f"  hlil: null ({call['hlil_statement_reason']})")
        if call.get("mlil"):
            lines.append(f"  mlil: {call['mlil']}")
        if call.get("llil"):
            lines.append(f"  llil: {call['llil']}")
        argument_rows = _field_list(call, "arguments")
        args = [arg for arg in argument_rows if isinstance(arg, dict)]
        # Confidence and arity diagnostics also describe empty argument lists.
        if args or call.get("argument_confidence") or call.get("arity_mismatch"):
            source = call.get("argument_source")
            # #549: mark whether `arguments` is canonical (authoritative HLIL/ABI) or a
            # heuristic lower-IL fallback, so an agent traces the right field.
            confidence = call.get("argument_confidence")
            # The tag is joined, so a non-string source or confidence cost the
            # WHOLE evidence card -- and it renders cleanly with the field
            # absent, which is the asymmetry #619 is about.
            tag = " ".join(str(x) for x in (source, confidence) if x)
            lines.append("  arguments:" + (f" ({tag})" if tag else ""))
            for arg in args:
                lines.append(f"    {arg.get('text', '')}"
                             f"{_render_resolved_arg(_field_dict(arg, 'resolved'))}")
            if call.get("arity_unknown"):
                # #648: the callee has no recovered prototype, so BN assumed every
                # argument register was live and HLIL rendered whatever sat in them.
                note = ("  arity: UNKNOWN — callee has no recovered prototype, so these "
                        "arguments are BN's register guess, not its signature")
                if call.get("abi_register_saturated"):
                    note += "; the count saturates the ABI argument registers"
                lines.append(note + ". Confirm with `bn proto get <callee>` / `proto set`.")
            if call.get("arity_mismatch"):
                # #648/#704: a resolved, non-variadic callee's recovered prototype
                # declares a DIFFERENT argument count than the canonical list rendered
                # (extra OR missing). `_argument_arity_evidence` only sets this flag
                # when `argument_source == "hlil"` (#704 round-3), but name the layer
                # from `source` rather than hard-coding "HLIL" so this note can never
                # attribute the list to a layer that did not actually produce it.
                declared = call.get("declared_arity")
                # `str()` first: the layer name is only a LABEL, and a non-string
                # `argument_source` made `.upper()` an AttributeError that cost
                # the whole evidence card (#619).
                layer = str(source).upper() if source else "the rendered list"
                # A missing/malformed list or discarded member supplies no exact
                # count. Validate this row, not render-wide skew from other calls.
                if (
                    isinstance(declared, int)
                    and isinstance(call.get("arguments"), list)
                    and len(args) == len(argument_rows)
                ):
                    note = (f"  arity: MISMATCH — {layer} rendered {len(args)} argument(s) but "
                            f"the callee's recovered prototype declares {declared}")
                else:
                    note = (f"  arity: MISMATCH — {layer} rendered a different argument count "
                            "than the callee's recovered prototype declares")
                lines.append(note + "; the canonical list is not the declared signature. "
                             "Confirm with `bn proto get <callee>` / `proto set`.")
        if call.get("callee_unresolved") and not call.get("arity_unknown"):
            # #648/#704: the call's destination could not be matched to any callee
            # function at all (genuinely indirect, or a resolved-but-unmatched
            # direct destination) -- no signature exists to check arguments
            # against, so `argument_confidence` was demoted regardless of source.
            lines.append(
                "  arity: UNKNOWN — call target could not be resolved to a callee "
                "function, so no signature exists to check arguments against."
            )
        if call.get("prototype_unverified"):
            # #759/#862: the third demotion cause. `arity_unknown`,
            # `arity_mismatch` and `callee_unresolved` each explain themselves on
            # this surface; a row demoted because a bundled library CONTRADICTS
            # the callee's recovered prototype rendered a bare
            # `arguments: (hlil inferred)` with no reason, which is the silent
            # demotion this issue family exists to stop.
            declared_n = call.get("declared_arity")
            library_n = call.get("library_arity")
            source_lib = call.get("library_source")
            counts = (
                f"declares {declared_n} but {source_lib or 'an attached type library'} "
                f"declares {library_n}"
                if isinstance(declared_n, int) and isinstance(library_n, int)
                else "disagrees with an attached type library"
            )
            lines.append(
                f"  arity: UNVERIFIED — the callee's recovered prototype {counts}, "
                "so the recovered signature is not corroborated and these arguments "
                "may be under-recovered. That disagreement is one reason this row is "
                "not fully corroborated; any other `arity:` line above names another, "
                "and a list recovered from MLIL/LLIL is heuristic whatever the library "
                "says. Check `bn proto get <callee>`: if you "
                "pinned that prototype deliberately, this row disagrees with your "
                "statement -- BN cannot tell a pinned prototype from a recovered one "
                "here, so decide which is right for this binary."
            )
        variadic = _field_dict(call, "variadic")
        if variadic.get("is_variadic"):
            # #558: surface variadic under-recovery / recovered format string.
            if variadic.get("under_recovered") and variadic.get("warning"):
                lines.append(f"  variadic: UNDER-RECOVERED — {variadic['warning']}")
            elif variadic.get("format_string") is not None:
                lines.append(
                    f"  variadic: {variadic.get('callee', '?')} "
                    f"format={variadic['format_string']!r} "
                    f"conversions={variadic.get('format_conversions')}"
                )
    return "\n".join(lines)


def _render_resolved_arg(resolved: Any) -> str:
    if not isinstance(resolved, dict):
        return ""
    section = resolved.get("section")
    suffix = f" [{section}]" if section else ""
    if resolved.get("string") is not None:
        value = json.dumps(resolved["string"], ensure_ascii=True)
        encoding = resolved.get("encoding")
        if encoding:
            value += f"({encoding})"
        if resolved.get("truncated"):
            value += " [truncated]"
        return f" -> {value}{suffix}"
    if resolved.get("symbol"):
        return f" -> {resolved['symbol']}{suffix}"
    if resolved.get("function"):
        return f" -> {resolved['function']}{suffix}"
    return ""


@_discloses
def _render_surface_text(value: Any) -> str:
    """#503: render the hidden code surface -- init/ctor pointers, candidate vtable/
    dispatch tables, and data-referenced code BN did not functionize."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    s = _field_dict(value, "summary")
    # An all-zero card is what a clean scan with nothing to report prints, so a
    # malformed summary must not be able to impersonate one (#619). `?`, not 0,
    # for the counts it should have carried; the disclosure names the field.
    miss: Any = "?" if value.get("summary") and not s else 0
    lines = [
        f"hidden surface: {s.get('init_sections', miss)} init section(s), "
        f"{s.get('candidate_tables', miss)} candidate table(s), "
        f"{s.get('missing_function_candidates', miss)} missing-function candidate(s)"
    ]
    for w in _field_list(value, "warnings"):
        lines.append(f"warning: {w}")

    init = _field_list(value, "init_sections")
    if init:
        lines.append("")
        lines.append("init / ctor sections (pre-main code):")
        for sec in init:
            if not isinstance(sec, dict):
                continue
            lines.append(
                f"  {sec.get('name', '?')}  {sec.get('start', '?')}..{sec.get('end', '?')}  "
                f"entries={sec.get('total_entries', '?')}  "
                f"fn={sec.get('resolved_functions', 0)}  missing={sec.get('missing_functions', 0)}")

    tables = _field_list(value, "candidate_tables")
    if tables:
        lines.append("")
        lines.append("candidate vtable / dispatch tables (runs of pointers-to-code):")
        for t in tables:
            if not isinstance(t, dict):
                continue
            lines.append(
                f"  {t.get('address', '?')} [{t.get('section', '?')}]  "
                f"entries={t.get('entries', '?')}  fn={t.get('resolved_functions', 0)}  "
                f"missing={t.get('missing_functions', 0)}")

    cands = _field_list(value, "missing_function_candidates")
    if cands:
        code_likely = [c for c in cands if isinstance(c, dict) and c.get("code_likely")]
        lines.append("")
        lines.append(
            f"missing-function candidates (executable, data-referenced, no BN function): "
            f"{len(code_likely)} code-likely of {len(cands)}")
        # Show the high-confidence subset first; then the rest, marked with why.
        ordered = code_likely + [c for c in cands if isinstance(c, dict) and not c.get("code_likely")]
        for c in ordered:
            depth = c.get("decode_depth")
            why = []
            if not c.get("aligned"):
                why.append("unaligned")
            if isinstance(depth, int):
                why.append(f"decode={depth}")
            tag = "code-likely" if c.get("code_likely") else "weak"
            # #647: a candidate that resolves to a printable string is self-refuting --
            # show it, so a false lead reads as `-> "Set Channel Index"` rather than a
            # bare address the agent must spend a command disproving.
            preview = c.get("string")
            if preview:
                text = str(preview)
                clipped = text[:48]
                suffix = "  -> " + _render_string_literal(clipped, truncated=len(text) > 48)
            else:
                suffix = ""
            lines.append(
                f"  {c.get('address', '?')}  [{c.get('section') or '?'}]  "
                f"via {c.get('provenance', '?')}  [{tag}: {', '.join(why)}]{suffix}")
        lines.append("")
        lines.append("(candidates -- NOT confirmed functions. `decode` = clean instructions "
                     "before an undefined one; a low decode reliably means data. Start with the "
                     "code-likely subset; verify with `disasm`, then `function create`.)")
    return "\n".join(lines)


@_discloses
def _render_call_descriptors_text(value: Any) -> str:
    """#469: one line per callsite of a registration API -- the declared descriptor
    field values (constants + resolved callback symbols), with unknown/computed
    fields marked explicitly rather than omitted."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    callee = value.get("callee", "<unknown>")
    lines = [f"descriptors passed to {callee} (arg {value.get('arg_index', '?')}): "
             f"{value.get('total', 0)} callsite(s)"]
    for warning in _field_list(value, "warnings"):
        lines.append(f"warning: {warning}")
    for row in _field_list(value, "items"):
        if not isinstance(row, dict):
            lines.append(_render_fallback_text(row))
            continue
        head = f"caller={row.get('caller', '?')} call={row.get('call_address', '?')}"
        status = row.get("status")
        # arg_out_of_range / not_a_local_descriptor: no fields to show. no_field_writes:
        # the descriptor was filled some other way (memcpy/template) -- still show the
        # (all-unknown) fields so the layout attempt is visible.
        if status in ("arg_out_of_range", "not_a_local_descriptor"):
            lines.append(f"{head} [{status}]")
            continue
        parts = []
        for f in _field_list(row, "fields"):
            if not isinstance(f, dict):
                continue
            name = f.get("name", "?")
            st = f.get("status")
            if st == "resolved":
                sym = f.get("symbol")
                mark = "~" if f.get("via") == "sibling_slot" else ""   # heuristic recovery
                val = f"{f.get('value')}" + (f" ({sym})" if sym else "")
                parts.append(f"{name}={mark}{val}")
            elif st == "computed":
                parts.append(f"{name}=<computed>")
            else:
                parts.append(f"{name}=<unknown>")
        suffix = "  [no field writes recovered -- memcpy/template init?]" if status == "no_field_writes" else ""
        lines.append(f"{head} " + " ".join(parts) + suffix)
    return "\n".join(lines)


@_discloses
def _render_virtual_call_text(value: Any) -> str:
    """#466: resolve an imported virtual call to provider vtable method(s)."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    factory = value.get("factory") or "<unresolved factory>"
    head = (f"virtual call @ {value.get('callsite', '?')} in {value.get('caller', '?')}: "
            f"vtable slot {value.get('slot_offset', '?')} (index {value.get('slot_index', '?')}), "
            f"object from {factory}")
    lines = [head]
    cands = _field_list(value, "candidates")
    if not cands:
        # #531: an unresolved slot (e.g. an unaligned offset that can't map to a slot
        # index) carries a concrete reason -- surface it instead of the generic hint.
        # #822: the typed discriminator rides along in parens (the
        # `hlil: null (reason_code)` shape) so a reader sees WHICH kind of
        # non-resolution this is: a capped scan is not an absent slot.
        reason = value.get("unresolved_reason")
        code = value.get("unresolved_reason_code")
        code_s = f" ({code})" if code else ""
        if reason:
            lines.append(f"  unresolved: {reason}{code_s}")
        else:
            lines.append("  no provider class implements this slot "
                         "(check --providers; a slot the provider's table ends "
                         f"before is absent, not truncated){code_s}")
        return "\n".join(lines)
    if value.get("ambiguous"):
        lines.append(f"  AMBIGUOUS: {len(cands)} provider classes implement slot "
                     f"{value.get('slot_offset', '?')}")
    # #706 follow-up (round-2 finding 9): a reason can be attached ALONGSIDE a
    # resolved/ambiguous candidate set (an unscanned provider's capped vtable
    # scan might supply another candidate) -- surface it here instead of only
    # inside the `not cands` branch above, which would silently drop it.
    for warning in _field_list(value, "warnings"):
        lines.append(f"  warning: {warning}")
    for c in cands:
        if not isinstance(c, dict):
            lines.append(f"  {c!r}")
            continue
        method = c.get("method") or "<unnamed>"
        entry = c.get("vtable_entry") or "?"
        # #533: include the concrete jump target (method_address) -- the pointer's
        # VALUE, distinct from vtable_entry (the slot's address). Render as hex when
        # present; tolerate None/absent (unrecovered target) without crashing.
        ma = c.get("method_address")
        if isinstance(ma, int):
            ma = hex(ma)
        target = str(ma) if ma else "?"
        lines.append(f"  {c.get('class', '?')}  ->  {method} @ {target}"
                     f"   [{c.get('provider', '?')} vtable {c.get('vtable', '?')} @ {entry}]")
    return "\n".join(lines)


@_discloses
def _render_record_table_text(value: Any) -> str:
    """#455: render a mixed-record dispatch table -- one block per record, each
    field labeled fn / data / scalar / null so a scalar isn't read as a bad slot."""
    lines = [
        f"record table @ {value.get('address', '<unknown>')}  "
        f"record-size: {value.get('record_size', '?')}  "
        f"ptr-fields: {', '.join(str(p) for p in _field_list(value, 'ptr_fields')) or '(none)'}"
    ]
    for warning in _field_list(value, "warnings"):
        lines.append(f"warning: {warning}")
    for row in _field_list(value, "items"):
        if not isinstance(row, dict):
            lines.append(_render_fallback_text(row))
            continue
        lines.append("")
        lines.append(f"[{row.get('row', '?')}] {row.get('base', '<unknown>')}")
        for f in _field_list(row, "fields"):
            if not isinstance(f, dict):
                continue
            off = f.get("offset", 0)
            off_s = f"+{off:#x}" if isinstance(off, int) else f"+{off}"
            kind = f.get("kind")
            # #467: a DECLARED typed field carries a name; show it so the record reads
            # as its struct fields (fn fields keep BN's resolved callee in `name`).
            fname = f.get("name") if kind in ("scalar", "char_array") else None
            nm = f" {fname}" if fname else ""
            if kind == "function_pointer":
                lines.append(f"  {off_s:<6} fn      {f.get('target', '?')}  {f.get('name') or ''}".rstrip())
            elif kind == "data_pointer":
                note = f'  "{f["preview"]}"' if f.get("preview") else (f"  {f['symbol']}" if f.get("symbol") else "")
                lines.append(f"  {off_s:<6} data    {f.get('target', '?')}{note}")
            elif kind == "char_array":  # #467 inline string field
                lines.append(f'  {off_s:<6} char[{f.get("size", "?")}]{nm}  "{f.get("value", "")}"'.rstrip())
            elif kind == "scalar":
                typ = f.get("type")
                if typ and str(typ).startswith("i") and isinstance(f.get("value"), int):
                    # #467: a SIGNED (i*) typed field shows its decimal value (+ hex),
                    # so -100 isn't rendered as 0xff9c and confused with 65436.
                    val_s = f"{f['value']} ({f.get('hex')})"
                elif f.get("hex"):
                    val_s = f["hex"]                       # unsigned typed field
                else:
                    val_s = f.get("value", "?")            # auto-scalar gap (hex string)
                lines.append(f"  {off_s:<6} scalar{nm}  {val_s}  ({f.get('size', '?')}B)")
            elif kind == "null":
                lines.append(f"  {off_s:<6} null")
            # Unmapped / unreadable: `kind` is whatever arrived, and a field row
            # with no kind at all is the shape an older bridge sends --
            # `{None:<7}` is a TypeError, so one such ELEMENT cost the whole
            # table (#619).
            else:
                lines.append(
                    f"  {off_s:<6} {'?' if kind is None else str(kind):<7} "
                    f"{f.get('value', '')}".rstrip())
    return "\n".join(lines)


@_discloses
def _render_pointer_table_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    if value.get("kind") == "record_table":  # #455 mixed-record mode
        return _render_record_table_text(value)
    lines = [
        f"pointer table @ {value.get('address', '<unknown>')}",
        f"pointer-size: {value.get('pointer_size', '<unknown>')}  stride: {value.get('stride', '<unknown>')}"
        f"  read-width: {value.get('read_width', value.get('pointer_size', '<unknown>'))}",
    ]
    # Through the choke point, so a malformed `context` envelope discloses
    # instead of rendering the table byte-identically to one with no context.
    suffix = _context_suffix(_field_dict(value, "context"))
    if suffix:
        lines.append(f"context{suffix}")
    for warning in _field_list(value, "warnings"):
        lines.append(f"warning: {warning}")
    lines.append("")
    for item in _field_list(value, "items"):  # #275: was `entries`
        if not isinstance(item, dict):
            lines.append(_render_fallback_text(item))
            continue
        prefix = f"[{_fmt_count(item.get('index', '?')):>2}] {item.get('entry_address', '<unknown>')}"
        if not item.get("readable", True):
            lines.append(f"{prefix}  <unreadable>")
            continue
        plausibility = "" if item.get("plausible", True) else "  [implausible]"
        lines.append(f"{prefix}  {item.get('value', '<unknown>')} -> "
                     f"{_render_target_line(_field_dict(item, 'target'))}{plausibility}")
    return "\n".join(lines)


@_discloses
def _render_message_lens_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    shown = value.get("count", 0)
    total = value.get("total", shown)
    header = f"message lens: {value.get('query', '<unknown>')} ({total} matches"
    if value.get("truncated"):
        header += f"; showing first {shown}, increase --limit for the rest"
    header += ")"
    lines = [header]
    for match in _field_list(value, "items"):  # #275: was `matches`
        if not isinstance(match, dict):
            lines.append(_render_fallback_text(match))
            continue
        type_string = _field_dict(match, "type_string")
        lines.append("")
        lines.append(f"{type_string.get('address', '<unknown>')}  {json.dumps(type_string.get('value', ''), ensure_ascii=True)}")
        suffix = _context_suffix(_field_dict(type_string, "context"))
        if suffix:
            lines.append(f"  context{suffix}")
        xrefs = _field_dict(match, "xrefs")
        code_count = len(_field_list(xrefs, "code_refs"))
        data_count = len(_field_list(xrefs, "data_refs"))
        lines.append(f"  xrefs: {code_count} code, {data_count} data")
        for ref in _field_list(xrefs, "code_refs")[:3]:
            if isinstance(ref, dict):
                lines.append(f"    code {ref.get('address', '<unknown>')}  {ref.get('function') or '<unknown>'}{_context_suffix(_field_dict(ref, 'context'))}")
        for ref in _field_list(xrefs, "data_refs")[:3]:
            if isinstance(ref, dict):
                lines.append(f"    data {ref.get('address', '<unknown>')}{_context_suffix(_field_dict(ref, 'context'))}")
        table_windows = _field_list(match, "metadata_table_windows")
        if table_windows:
            lines.append(f"  metadata table windows: {len(table_windows)}")
            for table in table_windows[:2]:
                if isinstance(table, dict):
                    lines.append(f"    table @ {table.get('address', '<unknown>')}")
                    for warning in _field_list(table, "warnings")[:2]:
                        lines.append(f"      warning: {warning}")
    # Resolved RTTI data symbols (the real vtable/typeinfo the lens targets, #194)
    for sym in _field_list(value, "rtti_symbols"):
        if not isinstance(sym, dict):
            continue
        xr = _field_dict(sym, "xrefs")
        cc = len(_field_list(xr, "code_refs"))
        dc = len(_field_list(xr, "data_refs"))
        lines.append("")
        lines.append(f"rtti {sym.get('kind', '?')}: {sym.get('symbol', '')} @ {sym.get('address', '?')}"
                     f"  xrefs: {cc} code, {dc} data")
        tw = _field_dict(sym, "table_window")
        if tw:
            # #303: the table window is the #275 envelope keyed on `items`; the
            # pre-#275 `entries` key always read 0, so a resolved RTTI vtable
            # window falsely rendered "(0 slots)" in text while the JSON carried
            # the real slots. (entries fallback for any legacy producer.)
            slot_count = len(_field_list(tw, "items", "entries"))
            lines.append(f"    vtable window @ {tw.get('address', '?')} "
                         f"({slot_count} slots)")
    for hint in _field_list(value, "hints"):
        lines.append(f"hint: {hint}")
    return "\n".join(lines)


@_discloses
def _render_fanout_text(value: Any, inner_renderer: Callable[[Any], str] | None = None) -> str:
    """Render an --all-instances fan-out (#169 L1): a header per instance, then
    that instance's result rendered by the command's own text renderer (or a
    fallback), and an ``error:`` line for instances that couldn't be reached/
    resolved. Failures are per-instance rows, not a hard failure."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    rows = _field_list(value, "instances")
    ok = sum(1 for r in rows if isinstance(r, dict) and r.get("ok"))
    lines = [f"fan-out: {value.get('command', '?')} — {len(rows)} result(s) "
             f"({ok} ok, {len(rows) - ok} failed)"]
    # Through the choke point: a non-iterable here used to raise (an int) or
    # render one bogus entry per CHARACTER (a string), and both cost the whole
    # fan-out view instead of one line (#619).
    expanded = _field_list(value, "auto_expanded_instances")
    if expanded:
        # #368: be explicit that a multi-target instance was surveyed in full, so
        # extra rows for one instance read as complete coverage, not a duplicate.
        lines.append(f"  (surveyed all targets of multi-target instance(s): {', '.join(map(str, expanded))})")
    slow = _field_list(value, "slow_rows")
    if slow:
        # #417: show where a broad survey spent its time so a long fan-out reads as
        # progress (which instance was slow), not a wedge.
        parts = []
        for s in slow:
            if not isinstance(s, dict):
                continue
            tgt = f"/{s['target']}" if s.get("target") else ""
            parts.append(f"{s.get('instance', '?')}{tgt} {s.get('duration_ms', '?')}ms")
        if parts:
            lines.append(f"  slowest: {', '.join(parts)}")
    for r in rows:
        if not isinstance(r, dict):
            continue
        header = f"\n== instance {r.get('instance', '?')}"
        if r.get("target"):
            header += f"  (target {r['target']})"
        header += " =="
        lines.append(header)
        if r.get("ok"):
            inner = r.get("result")
            if inner_renderer is not None:
                try:
                    lines.append(inner_renderer(inner))
                except BridgeError:
                    # A `BridgeError` is not a render failure: it is this CLI
                    # declaring it cannot TRUST the reply, and the documented
                    # contract for that is exit 2 from `main()`. Swallowing it
                    # here rendered a silent fallback at exit 0 for a payload
                    # the tool had just refused -- unparseable data reading as
                    # a clean result, and worse here than anywhere, because
                    # `--all-instances` is a SURVEY: a fallback row is
                    # indistinguishable from a row that genuinely had little to
                    # say (#619).
                    raise
                except Exception:
                    # Everything else still falls back, and the survey goes on.
                    # That intent is right: one instance's odd-but-parseable
                    # payload tripping a renderer must not cost the other nine
                    # rows. Different cause, different handling.
                    lines.append(_render_fallback_text(inner))
            else:
                lines.append(_render_fallback_text(inner))
        else:
            lines.append(f"  error: {r.get('error', '<unknown>')}")
    return "\n".join(lines)


@_discloses
def _render_orient_text(value: Any) -> str:
    """Render the orientation digest (#169 L2) as a compact triage card: analysis
    state up front (so an empty strings/function set from a --quick view isn't
    trusted), then function count, imports summary, sections, and the bounded
    strings sample."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    target = _field_dict(value, "target")
    name = target.get("basename") or target.get("filename") or target.get("name") or "<target>"
    state = value.get("analysis_state") or ("full" if value.get("analyzed") else "?")
    lines = [f"orientation: {name}  [analysis: {state}]"]
    if not value.get("analyzed", True):
        lines.append("  ! loaded with --quick — run `bn refresh` before trusting strings/functions")
    fc = value.get("function_count")
    if fc is not None:
        lines.append(f"  functions: {fc}")
    imp = _field_dict(value, "imports_summary")
    total = imp.get("total_symbols", imp.get("total"))
    by_kind = _field_dict(imp, "by_kind")
    kinds = ", ".join(f"{k}={v}" for k, v in list(by_kind.items())[:6])
    lines.append(f"  imports: {total if total is not None else '?'}" + (f" ({kinds})" if kinds else ""))
    secs = _field_dict(value, "sections")
    sec_items = _field_list(secs, "items")
    # Present-and-empty and present-but-malformed both render the count row, so
    # the malformed rendering is a strict SUPERSET of the empty one (the note is
    # appended by the decorator); a genuinely ABSENT listing omits it. PRESENT is
    # asked of the choke point, never re-derived: spelling it `"items" in secs`
    # disagreed with the helper about an explicit null and printed a confident
    # `sections: 0` for a payload that had claimed nothing (#619).
    if sec_items or _field_present(secs, "items"):
        names = " ".join(str(s.get("name", "?")) for s in sec_items[:12] if isinstance(s, dict))
        lines.append(f"  sections: {secs.get('total', len(sec_items))}  {names}")
    # PRESENT, not DECLARED and not raw truth: an `existing_annotations` the
    # payload CLAIMED renders its counts even when every count is zero (base
    # rendered a `{}` that way), and a malformed one discloses instead of
    # dropping the whole block as if the target had never been annotated (#619).
    ea = _field_dict(value, "existing_annotations")
    if _field_present(value, "existing_annotations"):
        # #561: disclose annotations already present so an agent doesn't over-credit
        # itself or trust inherited names/comments as current-run analysis.
        if ea.get("unavailable"):
            lines.append(f"  existing annotations: unavailable — {ea['unavailable']}")
        else:
            # #733 F2: the analyst/placeholder split is rendered only when the
            # bridge actually reports it, so an older bridge's digest prints
            # exactly the line it printed before rather than two fabricated
            # zeroes.
            row = (
                f"  existing annotations: comments={ea.get('comments', 0)}, "
                f"function-docs={ea.get('function_comments', 0)}, "
                f"user-symbols={ea.get('user_symbols', 0)}"
            )
            if _field_present(ea, "analyst_symbols"):
                # Each fragment gated on its OWN key and stated through
                # `_stated_count`: a bridge that reports `analyst_symbols`
                # without `placeholder_symbols` printed a bare
                # `placeholders=None`, and `_stated_count` alone would have
                # printed `0` for an absent key -- the fabricated zero this
                # module refuses everywhere else. Absent -> the fragment is
                # omitted; present but unreadable -> `?` (#733 F2 review).
                row += f", analyst-symbols={_stated_count(ea, 'analyst_symbols')}"
                if _field_present(ea, "placeholder_symbols"):
                    row += (
                        f", placeholders={_stated_count(ea, 'placeholder_symbols')}"
                    )
            row += f", cache-restored={ea.get('analysis_cache_restored', False)}"
            lines.append(row)
            if ea.get("provenance_hint"):
                lines.append(f"  ! {ea['provenance_hint']}")
    ss = _field_dict(value, "strings_sample")
    if ss.get("unavailable"):
        lines.append(f"  strings: unavailable — {ss['unavailable']}")
    else:
        items = _field_list(ss, "items")
        # Disclose the min-length filter so orient's total reconciles with the
        # `bn strings` total (which uses a lower default) (#357).
        mn = value.get("strings_min_length")
        filt = f"min-length {mn}; " if mn is not None else ""
        # #646: name the sections the sample came from, so a low-signal sample is
        # attributable instead of looking like the whole binary's flavour.
        drawn = _field_list(ss, "sample_sections")
        from_where = f"; from {', '.join(str(d) for d in drawn)}" if drawn else ""
        lines.append(
            f"  strings ({filt}sample {len(items)} of {ss.get('total', len(items))}{from_where}):")
        for s in items[:15]:
            if isinstance(s, dict):
                # `or ''` (not just the .get default) guards an explicit value:None.
                raw = s.get("value")
                shown = raw if isinstance(raw, str) else ("" if raw is None else repr(raw))
                lines.append(f"    {s.get('address', '?')}  {shown[:80]!r}")
    return "\n".join(lines)


@_discloses
def _render_init_arrays_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    sections = _field_list(value, "items")  # #275: was `sections`
    if not sections:
        # #448: on a `.so` you'd expect constructors, so a bare "none" reads like a
        # possible miss. State the authoritative reason so an empty result is
        # self-evidently correct, not a suspected gap.
        return "init arrays: none (no DT_INIT / DT_INIT_ARRAY present)"
    lines = [f"init arrays: {len(sections)} section(s), pointer-size={value.get('pointer_size', '<unknown>')}"]
    for section in sections:
        if not isinstance(section, dict):
            lines.append(_render_fallback_text(section))
            continue
        lines.append("")
        lines.append(
            f"{section.get('name', '<unknown>')} "
            f"{section.get('start', '<unknown>')}-{section.get('end', '<unknown>')} "
            f"entries={section.get('total_entries', '?')}"
        )
        if section.get("truncated"):
            lines.append(f"  showing first {section.get('shown_entries', '?')} entries")
        table = _field_dict(section, "table")
        for warning in _field_list(table, "warnings"):
            lines.append(f"  warning: {warning}")
        for item in _field_list(table, "items"):  # #275: embedded table is canonical too
            if not isinstance(item, dict):
                continue
            prefix = f"  [{_fmt_count(item.get('index', '?')):>2}] {item.get('entry_address', '<unknown>')}"
            if not item.get("readable", True):
                lines.append(f"{prefix}  <unreadable>")
                continue
            lines.append(f"{prefix}  {item.get('value', '<unknown>')} -> "
                         f"{_render_target_line(_field_dict(item, 'target'))}")
    return "\n".join(lines)


@_discloses
def _render_callsites_text(value: Any, *, prefer_caller_static: bool = False) -> str:
    # callsites returns the {items,total,...} envelope (#131 / item 11). Keep the
    # paging metadata (#454: callsites now pages bridge-side like xrefs) so a
    # truncated high-fan-in survey states the true total + remainder in a footer.
    total = None
    offset = None
    lower_bound = None
    callers_scanned = None
    caller_total = None
    scan_truncated = False
    has_more = False
    caller_scan_note = None
    if _field_declared(value, "items"):
        total = value.get("total")
        offset = value.get("offset")
        lower_bound = value.get("total_lower_bound")
        callers_scanned = value.get("callers_scanned")
        caller_total = value.get("caller_total")
        scan_truncated = bool(value.get("scan_truncated"))
        has_more = bool(value.get("has_more"))
        caller_scan_note = value.get("caller_scan_note")
        value = _field_list(value, "items")
    if not isinstance(value, list):
        return _render_fallback_text(value)
    if not value and not isinstance(total, int):
        if total is not None:
            # #619 follow-up: a non-int `total` on an empty page is not a
            # confirmed zero -- it is a malformed/unusable count. Saying "no
            # callsites found" here would assert the same confident-zero the
            # over-shot-page fix above was written to stop fabricating.
            return (
                f"no callsites on this page (offset {_fmt_offset(offset)}); "
                "total count is not a number, so a zero result cannot be confirmed"
            )
        if caller_scan_note:
            # #816: an empty page under a partial caller scan is the one place
            # "no callsites found" is exactly the false certainty the scan note
            # exists to prevent -- say what was actually established.
            return (
                "no callsites found among the callers examined; the caller scan was "
                f"incomplete ({caller_scan_note}), so absence is not established"
            )
        return "no callsites found"

    blocks = []
    for row in value:
        if not isinstance(row, dict):
            blocks.append(_render_fallback_text(row))
            continue

        callee = _field_dict(row, "callee")
        containing = _field_dict(row, "containing_function")
        call_addr = row.get("call_addr", "<unknown>")
        caller_static = row.get("caller_static", "<unknown>")
        call_index = row.get("call_index")
        # A tailcall (tail-branch into the target) has no real return site, so flag
        # it -- its caller_static is the byte after the branch, not a return addr (#47).
        kind_tag = "  [tailcall]" if row.get("call_kind") == "tailcall" else ""
        primary = (
            f"caller_static {caller_static} | call {call_addr}{kind_tag}"
            if prefer_caller_static
            else f"call {call_addr} | caller_static {caller_static}{kind_tag}"
        )
        lines = [
            primary,
            (
                f"within: {containing.get('name', '<unknown>')} @ "
                f"{containing.get('address', '<unknown>')}"
            ),
            f"callee: {callee.get('name', '<unknown>')} @ {callee.get('address', '<unknown>')}",
        ]
        if call_index is not None:
            lines.append(f"call-index: {call_index}")
        if row.get("within_query"):
            lines.append(f"within-query: {row['within_query']}")
        if row.get("hlil_statement"):
            lines.append(f"hlil: {row['hlil_statement']}")
        elif row.get("hlil_statement_reason"):
            # #557: say WHY the HLIL statement is null instead of silently omitting it.
            lines.append(f"hlil: null ({row['hlil_statement_reason']})")
        if row.get("pre_branch_condition"):
            lines.append(f"pre-branch: {row['pre_branch_condition']}")
        variadic = _field_dict(row, "callee_variadic")
        if variadic.get("is_variadic"):
            # #558: steer to the argument-recovery views for an imported variadic callee.
            lines.append(
                f"variadic-callee: {variadic.get('name', '?')} — HLIL may show only fixed "
                f"args; run `bn evidence function {containing.get('name', '<caller>')}` "
                f"or `bn disasm {containing.get('name', '<caller>')} --linear`"
            )

        call_instruction = _field_dict(row, "call_instruction")
        previous = _field_list(row, "previous_instructions")
        next_instructions = _field_list(row, "next_instructions")
        lines.append("context:")
        for item in previous:
            if isinstance(item, dict):
                lines.append(f"  {item.get('address', '<unknown>')}  {item.get('text', '')}".rstrip())
        if row.get("disasm_context_reason"):
            # #816: the call site is real (its identity fields are usable) but the
            # disassembly sweep produced no entry at its address, so the context is
            # null by evidence, not by omission -- say WHY, mirroring `hlil: null (...)`.
            lines.append(f"> unavailable ({row['disasm_context_reason']})")
        else:
            lines.append(
                f"> {call_instruction.get('address', '<unknown>')}  {call_instruction.get('text', '')}".rstrip()
            )
        for item in next_instructions:
            if isinstance(item, dict):
                lines.append(f"  {item.get('address', '<unknown>')}  {item.get('text', '')}".rstrip())
        blocks.append("\n".join(lines))
    body = "\n\n".join(block for block in blocks if block)
    # #611: a partial LAST page (offset > 0, has_more false, fewer rows than the
    # total) must still say so -- otherwise it silently reads as the complete
    # result instead of one page of a paged survey. An over-shot page (offset
    # past the end) must footer too instead of hitting the empty-list early
    # return and reading as "never called" while JSON reports the real total.
    if isinstance(total, int) and (has_more or _int_or_default(offset) > 0 or len(value) != total):
        footer = f"... showing {len(value)} of {total} callsites (offset {_fmt_offset(offset)})"
        if has_more:
            # Only a mid-survey page has somewhere to page forward to -- a true
            # last page (has_more False) must not repeat the hint.
            footer += "; use --offset/--limit to page (or --format json for all)"
        body = f"{body}\n\n{footer}" if body else footer
    elif has_more and scan_truncated and isinstance(lower_bound, int):
        scan = (
            f"; scanned {callers_scanned} of {caller_total} callers"
            if isinstance(callers_scanned, int) and isinstance(caller_total, int)
            else ""
        )
        footer = (
            f"... showing {len(value)} rows; at least {lower_bound} "
            f"callsites{scan}; exact total not computed to keep the scan bounded. "
            "Use --offset/--limit to page."
        )
        body = f"{body}\n\n{footer}" if body else footer
    if caller_scan_note:
        # #816: the caller enumeration itself was partial, so everything printed
        # above is a LOWER BOUND. Name the reason -- a short caller list must never
        # read as "not called". Same disclosure shape as the truncated
        # function-pointer scan in `_render_evidence_xrefs_text`.
        note = (
            "note: the caller scan was incomplete "
            f"({caller_scan_note}); the callsites above are a lower bound, not the "
            "whole set"
        )
        body = f"{body}\n\n{note}" if body else note
    if not blocks and not body:
        body = "no callsites found"
    return body


@_discloses
def _render_structured_il_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    fn = _field_dict(value, "function")
    form = "ssa" if value.get("ssa") else "non-ssa"
    lines = [f"{fn.get('name', '<unknown>')} @ {fn.get('address', '<unknown>')}  ({value.get('view', 'mlil')} {form})"]
    for ins in _field_list(value, "instructions"):
        # A malformed list element (non-dict) must render as fallback text, not
        # crash the whole listing with an AttributeError (#101).
        if not isinstance(ins, dict):
            lines.append(f"  {ins}")
            continue
        reads = ",".join(str(_as_dict(v).get("ssa", _as_dict(v).get("name", "?")))
                         for v in _field_list(ins, "vars_read"))
        writes = ",".join(str(_as_dict(v).get("ssa", _as_dict(v).get("name", "?")))
                          for v in _field_list(ins, "vars_written"))
        head = f"  [{ins.get('il_index')}] {ins.get('address')}  {ins.get('op')}  {ins.get('text', '')}".rstrip()
        lines.append(head)
        if reads or writes:
            lines.append(f"        r:[{reads}]  w:[{writes}]")
    return "\n".join(lines)


@_discloses
def _render_defuse_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    fn = _field_dict(value, "function")
    var = _field_dict(value, "variable")
    lines = [
        f"{fn.get('name', '<unknown>')} @ {fn.get('address', '<unknown>')}",
        f"variable: {var.get('ssa', var.get('name', '?'))}  ({var.get('type', '?')})",
    ]
    definition = _field_dict(value, "definition")
    if definition:
        lines.append(f"def: {definition.get('address')}  {definition.get('op')}  {definition.get('text', '')}".rstrip())
    elif _field_skewed("definition"):
        # PRESENT but not a usable definition object. `<none (parameter/entry/
        # aliased)>` is a confident claim about WHY there is no definition, so a
        # falsy wrong shape (`0`, `""`, `False`, `[]`) rendering it read as that
        # diagnosis instead of as an unusable field -- render what arrived and
        # let the choke point's note disclose the skew (#619).
        #
        # Asked of the choke point, not re-derived: `_field_present` is True for
        # a WELL-FORMED empty dict too, so spelling the test that way printed
        # `def: {}` -- a raw Python literal, undisclosed -- for a payload that
        # had simply found no definition. One question, one answer.
        lines.append(f"def: {_as_dict(value).get('definition')!r}")
    else:
        lines.append("def: <none (parameter/entry/aliased)>")
    if value.get("is_phi"):
        srcs = ", ".join(
            (str(s.get("ssa", s.get("name", "?"))) if isinstance(s, dict) else repr(s))
            for s in _field_list(value, "phi_sources"))
        lines.append(f"phi sources: {srcs}")
    uses = _field_list(value, "uses")
    lines.append(f"uses ({len(uses)}):")
    for u in uses:
        if not isinstance(u, dict):
            lines.append(f"  {u!r}")
            continue
        lines.append(f"  {u.get('address')}  {u.get('op')}  {u.get('text', '')}".rstrip())
    others = _field_list(value, "other_versions")
    if others:
        lines.append(f"other versions of {var.get('name', '?')}: {others}")
    return "\n".join(lines)


@_discloses
def _render_callgraph_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    fn = _field_dict(value, "function")
    lines = [f"{fn.get('name', '<unknown>')} @ {fn.get('address', '<unknown>')}"]
    # PRESENT, not merely declared: an explicit null CLAIMED nothing about the
    # callees, so asserting `callees (0):` from it is the same confident zero
    # this change exists to refuse. A present-and-empty list still prints the
    # row -- that is a real "we looked and found none" (#619).
    if _field_present(value, "callees"):
        callees = _field_list(value, "callees")
        lines.append(f"callees ({len(callees)}):")
        for c in callees:
            if not isinstance(c, dict):
                lines.append(f"  {c!r}")
                continue
            if c.get("kind") == "direct":
                tgt = _field_dict(c, "target")
                lines.append(f"  {c.get('call_addr')}  direct -> {tgt.get('name', '<unknown>')} @ {tgt.get('address')}")
            else:
                resolved = _field_list(c, "resolved")
                if resolved:
                    tgts = ", ".join(f"{_as_dict(r).get('name', '?')}@{_as_dict(r).get('address')}" for r in resolved)
                    suffix = f"resolved: {tgts}"
                else:
                    suffix = f"UNRESOLVED ({c.get('resolution_detail', 'indirect')})"
                lines.append(f"  {c.get('call_addr')}  indirect [{c.get('dest_expr', '')}]  {suffix}")
    if _field_present(value, "callers"):
        callers = _field_list(value, "callers")
        lines.append(f"callers ({len(callers)}):")
        for c in callers:
            if not isinstance(c, dict):
                lines.append(f"  {c!r}")
                continue
            caller = _field_dict(c, "caller")
            site = f"{c.get('call_addr')}  " if c.get("call_addr") else ""
            lines.append(f"  {site}{caller.get('name', '<unknown>')} @ {caller.get('address', '?')}")
    return "\n".join(lines)


@_discloses
def _render_values_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    fn = _field_dict(value, "function")
    lines = [f"{fn.get('name', '<unknown>')} @ {fn.get('address', '<unknown>')}"]
    lines.append(f"at {value.get('at')}: {value.get('expression', '<no instruction at address>')}")
    pvs = _field_dict(value, "possible_values")
    if not pvs:
        # `<unavailable>` is what an ABSENT field prints: a present but malformed
        # one must not impersonate "the analysis had no answer" (#619).
        lines.append(f"possible values: <malformed: {value['possible_values']!r}>"
                     if value.get("possible_values")
                     else "possible values: <unavailable>")
        return "\n".join(lines)
    # `str()` because the type is INTERPOLATED and then appended to: a non-string
    # `type` beside a readable `value` made `summary +=` a TypeError and cost the
    # whole values card, where the same payload without the type renders (#619).
    summary = str(pvs.get("type", "?"))
    if "value" in pvs:
        summary += f"  value={pvs['value']:#x}" if isinstance(pvs["value"], int) else f"  value={pvs['value']}"
    if pvs.get("values"):
        summary += f"  values={pvs['values']}"
    if pvs.get("ranges"):
        summary += f"  ranges={pvs['ranges']}"
    lines.append(f"possible values: {summary}")
    lines.append(f"  raw: {pvs.get('raw', '')}")
    if value.get("value_basis") == "source_expression" and value.get("source_expression"):
        lines.append(f"  of source expression: {value['source_expression']}")
    return "\n".join(lines)


def _render_leaf_line(leaf: Any) -> str:
    """One text line for a single unresolved-frontier leaf."""
    if not isinstance(leaf, dict):
        return f"  {leaf!r}"
    kind = leaf.get("kind")
    if kind == "unmodeled_callee":
        cal = _field_dict(leaf, "callee")
        args = _field_list(leaf, "tainted_args")
        return (
            f"  unmodeled_callee @ {leaf.get('address')}"
            f"  -> {cal.get('name', '?')} @ {cal.get('address', '?')}"
            f"  (tainted arg(s) {args})"
            + (f"  -- {leaf.get('note')}" if leaf.get("note") else "")
        )
    if kind == "pointer_escape":
        return (
            f"  pointer_escape @ {leaf.get('address')}"
            f"  buffer={leaf.get('buffer', '?')}"
            + (f"  {leaf.get('dest')}" if leaf.get("dest") else "")
            + (f"  -- {leaf.get('detail')}" if leaf.get("detail") else "")
        )
    if kind == "field_load_unresolved":
        bits = []
        if leaf.get("base") is not None:
            bits.append(f"base={leaf['base']}")
        if leaf.get("offset") is not None:
            bits.append(f"offset={leaf['offset']}")
        if leaf.get("width") is not None:
            bits.append(f"width={leaf['width']}")
        meta = ("  " + " ".join(bits)) if bits else ""
        return f"  field_load_unresolved @ {leaf.get('address')}{meta}"
    if kind == "arg_under_recovered":
        cal = _field_dict(leaf, "callee")
        return (
            f"  arg_under_recovered @ {leaf.get('address')}"
            f"  -> {cal.get('name', '?')} @ {cal.get('address', '?')}"
            f"  (recovered {leaf.get('recovered_params', '?')} param(s); "
            f"dropped arg(s) {_field_list(leaf, 'dropped_args')})"
            + (f"  -- {leaf.get('note')}" if leaf.get("note") else "")
        )
    if kind == "caller_sites_truncated":
        fn = _field_dict(leaf, "function")
        return (
            f"  caller_sites_truncated @ {leaf.get('address')}"
            f"  -> {fn.get('name', '?')} @ {fn.get('address', '?')}"
            f"  ({leaf.get('callers_followed', '?')} of {leaf.get('callers_total', '?')} "
            f"caller(s) followed; {leaf.get('callers_dropped', '?')} dropped)"
            + (f"  -- {leaf.get('note')}" if leaf.get("note") else "")
        )
    return (
        f"  {kind} @ {leaf.get('address')}  [{leaf.get('dest_expr', leaf.get('il_text', ''))}]"
        + (f"  -- {leaf.get('detail')}" if leaf.get("detail") else "")
    )


def _leaf_group_key(leaf: Any) -> tuple:
    """Collapse near-identical frontier leaves: one group per callee for
    unmodeled calls, per (base, offset) for field loads, per kind otherwise."""
    if not isinstance(leaf, dict):
        # Group malformed leaves by (type, repr): a type key can't collide with a
        # real `kind` string, and keying on the repr too means two DISTINCT
        # broken leaves each render instead of one hiding behind an `(xN)` count
        # of rows that only look alike because both are broken.
        return (type(leaf), repr(leaf))
    kind = leaf.get("kind")
    kind = kind if isinstance(kind, str) else str(kind)
    if kind == "unmodeled_callee":
        return (kind, str(_field_dict(leaf, "callee").get("name", "?")))
    if kind == "field_load_unresolved":
        return (kind, str(leaf.get("base")), str(leaf.get("offset")))
    if kind == "arg_under_recovered":
        return (kind, str(_field_dict(leaf, "callee").get("name", "?")))
    return (kind,)


def _render_grouped_leaves(leaves: list[Any], *, top_n: int = 12) -> list[str]:
    """Render unresolved leaves grouped by kind/callee with counts and a top-N
    cap (full detail stays in --format json) so real binaries don't flood the
    text output with a wall of near-identical leaves (#160)."""
    groups: dict[tuple, dict[str, Any]] = {}
    order: list[tuple] = []
    for leaf in leaves:
        gk = _leaf_group_key(leaf)
        g = groups.get(gk)
        if g is None:
            groups[gk] = {"rep": leaf, "count": 1}
            order.append(gk)
        else:
            g["count"] += 1
    order.sort(key=lambda gk: groups[gk]["count"], reverse=True)
    total = len(leaves)
    ngroups = len(order)
    hdr = f"frontiers ({total}"
    if ngroups != total:
        hdr += f" in {ngroups} group(s)"
    hdr += "):"
    out = [hdr]
    for gk in order[:top_n]:
        g = groups[gk]
        line = _render_leaf_line(g["rep"])
        if g["count"] > 1:
            line += f"  (x{g['count']})"
        out.append(line)
    if ngroups > top_n:
        hidden = order[top_n:]
        hidden_leaves = sum(groups[gk]["count"] for gk in hidden)
        out.append(
            f"  ... and {len(hidden)} more group(s) ({hidden_leaves} leaf(s)); "
            "see --format json for the full list")
    return out


def _render_taint_path(steps: list[Any]) -> list[str]:
    out = []
    last = len(steps) - 1
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        marker = ">" if i == last else " "
        reason = step.get("reason")
        line = f"  {marker} {step.get('address')}  {step.get('op')}  {step.get('il_text', '')}".rstrip()
        out.append(line)
        if reason:
            out.append(f"        <- {reason}")
    return out


def _taint_truncation_note(stats: dict[str, Any]) -> str:
    """Verdict-line truncation clause that names the CAUSE (#579/#576).

    A fixpoint-exhaustion truncation and an interprocedural depth cutoff share
    the run-level ``truncated`` flag but mean different things: the former's
    ``max_depth`` stat is 0 (a same-function iteration limit), so the historical
    ``truncated @depth 0`` misreported it as a depth-recursion cutoff. Render each
    cause distinctly with its own remediation. When ``truncation_cause`` is absent
    (an older bridge), preserve the historical ``@depth N`` wording verbatim."""
    if not stats.get("truncated"):
        return ""
    causes = _field_list(stats, "truncation_cause")
    if not causes:
        return f" · truncated @depth {stats.get('max_depth')}"
    parts: list[str] = []
    if "max_depth" in causes:
        parts.append(f"@depth {stats.get('max_depth')} "
                     "(interprocedural depth bound; raise --max-depth)")
    if "fixpoint_exhausted" in causes:
        parts.append("fixpoint unconverged (intra-function iteration budget "
                     "exhausted; raise --max-iters or narrow the source)")
    if "recursion" in causes:
        parts.append("recursion limit (possible unresolved cycle)")
    if "caller_cap" in causes:
        # #810: the backward caller ascent follows only the first N caller sites of
        # a parameter-origin slice; the rest are dropped (no knob raises the cap).
        # Name the cause and where the dropped count lives -- the frontier leaf.
        parts.append("caller-site cap reached (not every caller was followed; "
                     "see the caller_sites_truncated frontier)")
    if not parts:
        return f" · truncated @depth {stats.get('max_depth')}"
    return " · truncated " + "; ".join(parts)


def _taint_forward_verdict(value: dict[str, Any]) -> str:
    """One-line verdict for a forward-taint result, derived from existing fields."""
    findings = _field_list(value, "reached_sinks")
    leaves = _field_list(value, "leaves")
    stats = _field_dict(value, "stats")
    fns = stats.get("functions_visited")
    fns_part = f" · taint crossed {fns} fn(s)" if fns else ""
    trunc = _taint_truncation_note(stats)
    if findings:
        sinks = [_field_dict(f, "sink") for f in findings]
        classes = ", ".join(sorted({str(s.get("class") or "?") for s in sinks}))
        return f"verdict: {len(findings)} sink(s) reached ({classes}){fns_part}{trunc}"
    if leaves:
        return (f"verdict: NO modeled sink reached — {len(leaves)} tainted frontier(s) "
                f"(NOT an all-clear){fns_part}{trunc}")
    # Genuinely empty: no sink AND no frontier. This is the MOST caveated case, not
    # the least -- the engine reaching nothing does not mean the function is safe;
    # it is exactly the shape a structurally-invisible bug (use-after-free,
    # temporal, or an unmodeled source) produces. Carry the same "NOT an
    # all-clear" qualifier the partial-coverage paths do (#310).
    visited = f"visited {fns} fn(s)" if fns else "shallow coverage"
    return (f"verdict: no taint reached any sink or frontier — NOT an all-clear "
            f"({visited}; no modeled sink or tainted frontier found — also how a bug "
            f"the engine can't structurally see appears){trunc}")


def _taint_via_trail(value: dict[str, Any], finding: dict[str, Any]) -> str | None:
    """Compact callee trail for a sink, parsed from its path-step reasons:
    `<analyzed fn> → <callee> → … → <sink callee>`. None if no callees parse."""
    chain: list[str] = []
    fn = _field_dict(value, "function").get("name")
    if fn:
        chain.append(str(fn))
    for step in _field_list(finding, "path"):
        if not isinstance(step, dict):
            continue
        reason = str(step.get("reason") or "")
        m = re.search(r"calls (\S+) with tainted", reason)
        if not m:
            m = re.search(r"tainted arg\d+ reaches (\S+)", reason)
        if m:
            name = m.group(1)
            if not chain or chain[-1] != name:
                chain.append(name)
    return ("via: " + " → ".join(chain)) if len(chain) >= 2 else None


def _render_flow_line(f: dict[str, Any]) -> str:
    """One compact line for a forward finding: sink + address + arg + grouping
    signature + structural metrics. The full SSA path is shown only under --full."""
    sink = _field_dict(f, "sink")
    ai = sink.get("tainted_arg_index")
    arg = f" (arg {ai})" if ai is not None else ""
    m = _field_dict(f, "metrics")
    sig = _field_dict(f, "signature").get("rendered", "")
    unresolved = "y" if m.get("traverses_unresolved") else "n"
    facts = f"{{steps={m.get('steps', '?')} fns={m.get('fns_spanned', '?')} unresolved={unresolved}}}"
    head = f"[{sink.get('class') or '?'}] {sink.get('callee', '?')} @ {sink.get('address')}{arg}"
    return f"  {head}   {sig}   {facts}".rstrip()


def _render_forward_diagnostics(diag: dict[str, Any]) -> list[str]:
    """Compact frontier diagnostic for a zero-result forward taint query (#559).

    Factual only: where the seed reached and why the frontier stopped -- never a
    vulnerability verdict."""
    if not isinstance(diag, dict):
        return []
    fr = _field_dict(diag, "frontier")
    lu = _field_dict(diag, "last_use")
    out = ["diagnostics:"]
    out.append(
        f"  seed: matched {diag.get('source_callsites', 0)} source callsite(s), "
        f"produced {diag.get('tainted_values', 0)} tainted value(s)")
    if lu:
        _reason = f" ({lu['reason']})" if lu.get("reason") else ""
        out.append(f"  last propagated use: {lu.get('label', '?')} @ {lu.get('address', '?')}{_reason}")
    else:
        out.append("  last propagated use: <none — seed did not propagate>")
    out.append(
        f"  unmodeled call(s) reached: {'yes' if diag.get('unmodeled_calls_reached') else 'no'}")
    _fr = (f"  frontier: {fr.get('unresolved', 0)} unresolved, "
           f"{fr.get('coarse_memory', 0)} coarse-memory")
    _sm = fr.get("seed_misanchored", 0)
    if _sm:
        _fr += f", {_sm} seed-misanchored"
    out.append(_fr)
    # #562 honesty gate, folded into the same block: the anti-verdict, never a
    # vulnerability claim. Only present when the diagnostics carry it.
    if "safe_to_report_all_clear" in diag:
        gate = diag.get("safe_to_report_all_clear")
        out.append(f"  safe_to_report_all_clear: {'true' if gate else 'false'}"
                   + (" (may-analysis, not a proof)" if gate else ""))
        if diag.get("all_clear_reason"):
            out.append(f"    reason: {diag['all_clear_reason']}")
    if diag.get("next_action"):
        out.append(f"  next: {diag['next_action']}")
    return out


@_discloses
def _render_taint_text(value: Any, full: bool = False) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    fn = _field_dict(value, "function")
    direction = value.get("direction", "forward")
    lines = [f"{direction} taint in {fn.get('name', '<unknown>')} @ {fn.get('address', '<unknown>')}"]

    if direction == "forward":
        srcs = _field_list(value, "sources")
        lines.append("sources: " + (", ".join(_describe_loc(s) for s in srcs) or "<none>"))
        findings = _field_list(value, "reached_sinks")
        lines.append(_taint_forward_verdict(value))
        diagnostics = _field_dict(value, "diagnostics")
        if not findings and diagnostics:
            lines.extend(_render_forward_diagnostics(diagnostics))
        if findings:
            lines.append("")
            lines.append(f"flows ({len(findings)}):")
            for f in findings:
                # One compact line per flow by default (signature + metrics). Same-sink
                # findings are already unique per (callee,address,arg), so distinct sink
                # call-sites always render on their own line -- never folded behind a
                # count. --full appends the sink detail, via: trail, and full SSA path.
                if not isinstance(f, dict):
                    lines.append(f"  {f!r}")
                    continue
                lines.append(_render_flow_line(f))
                if full:
                    _detail = _field_dict(f, "sink").get('detail') or ''
                    if _detail:
                        lines.append(f"      -- {_detail}")
                    _via = _taint_via_trail(value, f)
                    if _via:
                        lines.append(f"    {_via}")
                    lines.extend(_render_taint_path(_field_list(f, "path")))
    else:
        sinks = _field_list(value, "sinks")
        lines.append("sinks: " + (", ".join(_describe_loc(s) for s in sinks) or "<none>"))
        # #810: a truncated backward run is INCOMPLETE (capped caller ascent, or a
        # recursion-limited slice). Without this the text showed the same slice
        # list a complete run shows; a complete run adds no line, so its output is
        # byte-identical to before.
        trunc = _taint_truncation_note(_field_dict(value, "stats"))
        if trunc:
            lines.append(f"verdict: INCOMPLETE{trunc}")
        slices = _field_list(value, "slices")
        for sl in slices:
            if not isinstance(sl, dict):
                lines.append(f"  {sl!r}")
                continue
            sink = _field_dict(sl, "sink")
            origin = _field_dict(sl, "origin")
            m = _field_dict(sl, "metrics")
            sig = _field_dict(sl, "signature").get("rendered", "")
            n = sl.get("reached_via_call_sites", 1)
            xn = (f"  (x{n} callsites)"
                  if isinstance(n, int) and not isinstance(n, bool) and n > 1 else "")
            unresolved = "y" if m.get("traverses_unresolved") else "n"
            lines.append("")
            # Compact one line per slice by default; (xN) surfaces the engine's existing
            # per-(seed,sink,origin) call-site count. --full shows origin/crosses/steps.
            lines.append(
                f"  [{sink.get('class') or sink.get('kind') or '?'}] "
                f"{sink.get('callee') or sink.get('kind') or '?'} @ {sink.get('address')} "
                f"(seed {sink.get('seed', '?')})   {sig}   "
                f"{{steps={m.get('steps', '?')} fns={m.get('fns_spanned', '?')} "
                f"unresolved={unresolved}}}{xn}".rstrip())
            if full:
                lines.append(
                    f"  slice for {sink.get('callee') or sink.get('kind') or '?'} @ {sink.get('address')} (seed {sink.get('seed', '?')}):"
                )
                _ok = origin.get("kind")
                if _ok == "constant" and origin.get("value") is not None:
                    _val = origin["value"]
                    _vs = f"{_val:#x}" if isinstance(_val, int) else str(_val)
                    _extra = _vs + (f" ({origin['var']})" if origin.get("var") else "")
                else:
                    _extra = origin.get("callee") or origin.get("var") or ""
                _spill = " (via spill)" if origin.get("via_spill") else ""
                lines.append(f"  origin: {_ok} {_extra}{_spill}".rstrip())
                crossed = _field_list(sl, "crossed_functions")
                if crossed:
                    lines.append(f"  crosses: {' <- '.join(str(c) for c in crossed)}")
                for step in _field_list(sl, "slice"):
                    if isinstance(step, dict):
                        lines.append(f"  {step.get('address')}  {step.get('op')}  {step.get('il_text', '')}".rstrip())
        status = _field_list(value, "sink_status")
        # A constant-length sink is "provably bounded" -- a SUCCESS, not a failed
        # seed -- so report it apart from genuinely-unseeded sinks (#310). A
        # malformed (non-dict) status row is neither.
        bounded = [s for s in status if isinstance(s, dict) and s.get("bounded")]
        unseeded = [s for s in status
                    if isinstance(s, dict) and not s.get("seeded", True) and not s.get("bounded")]
        if bounded:
            lines.append("")
            lines.append(f"provably bounded ({len(bounded)}):")
            for s in bounded:
                lines.append(f"  {_describe_loc(s)} -- {s.get('note', 'constant length, nothing to slice')}")
        if unseeded:
            lines.append("")
            lines.append(f"UNSEEDED SINKS ({len(unseeded)}):")
            for s in unseeded:
                lines.append(f"  {_describe_loc(s)} -- {s.get('note', 'could not seed')}")

    by_source = _field_dict(value, "by_source")
    if direction == "forward" and by_source:
        lines.append("")
        lines.append(f"PER-SOURCE ({len(by_source)} call site(s)):")
        for addr, br in by_source.items():
            if not isinstance(br, dict):
                lines.append(f"  {addr}: {br!r}")
                continue
            bsinks = [s for s in _field_list(br, "reached_sinks") if isinstance(s, dict)]
            bleaves = _field_list(br, "leaves")
            if bsinks:
                desc = ", ".join(
                    f"{_field_dict(s, 'sink').get('class', '?')} {_field_dict(s, 'sink').get('callee', '?')}"
                    for s in bsinks)
            else:
                desc = "no sinks"
            nfront = sum(1 for l in bleaves if isinstance(l, dict) and l.get("kind") == "unmodeled_callee")
            if bleaves:
                desc += f"; {len(bleaves)} leaf(s)" + (f" ({nfront} frontier)" if nfront else "")
            lines.append(f"  {addr}: {desc}")

    leaves = _field_list(value, "leaves")
    if leaves:
        lines.append("")
        lines.extend(_render_grouped_leaves(leaves))
    assumptions = _field_list(value, "assumptions")
    if assumptions:
        lines.append("")
        lines.append(f"caveats ({len(assumptions)}):")
        for a in assumptions:
            lines.append(f"  - {a}")
    msrc = _render_model_sources(_field_list(value, "model_sources"))
    if msrc:
        lines.append("")
        lines.append(msrc)
    if value.get("soundness"):
        lines.append("")
        lines.append(f"soundness: {value['soundness']}")
    return "\n".join(lines)


@_discloses
def _render_taint_models_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    lines: list[str] = []
    # #555: lead with the non-findings disclaimer so the callsite inventory below
    # is never skimmed as a taint/vuln findings list.
    note = value.get("catalog_note")
    if note:
        lines.append("NOTE: " + str(note))
        lines.append("")
    srcs = _field_list(value, "sources")
    if srcs:
        lines.append(f"sources ({len(srcs)}):")
        for s in srcs:
            if not isinstance(s, dict):
                lines.append(f"  {s!r}")
                continue
            p = " [present]" if s.get("present") else (" [absent]" if "present" in s else "")
            lines.append(f"  {s.get('symbol', '<unknown>')}  ->  {s.get('to', '')}{p}")
    sbc = _field_dict(value, "sinks_by_class")
    if sbc:
        lines.append("")
        # A class whose entry list is malformed contributes NO sink rows: this
        # inventory is read as ground truth, so counting a row that may not
        # exist is worse than disclosing the class as unusable. The per-class
        # skew is recorded by key, so the render's own note names it.
        #
        # The class itself is still LISTED and still counted, whether its entry
        # list is empty, absent or unusable. "We looked at this class and found
        # no modeled sink" is a real result, and dropping it from "in N class(es)"
        # understates the inventory in exactly the direction -- fewer classes
        # examined than were -- that this whole change exists to refuse (#619).
        classes = {cls: _field_list(sbc, cls) for cls in sbc}
        total = sum(len(entries) for entries in classes.values())
        lines.append(f"sinks ({total} in {len(classes)} class(es)); NOT findings:")
        for cls, entries in classes.items():
            lines.append(f"  [{cls}]")
            for e in entries:
                if not isinstance(e, dict):
                    lines.append(f"    {e!r}")
                    continue
                lines.extend(_render_taint_sink_entry(e))
    props = _field_list(value, "propagators")
    if props:
        lines.append("")
        lines.append(f"propagators ({len(props)}):")
        for p in props:
            if not isinstance(p, dict):
                lines.append(f"  {p!r}")
                continue
            lines.append(f"  {p.get('symbol', '<unknown>')}  {p.get('from_to', '')}")
    ov = _field_list(value, "overlays")
    if ov:
        lines.append("")
        lines.append("overlays: " + ", ".join(
            str(_as_dict(o).get("path", _as_dict(o).get("kind", "?"))) for o in ov))
    return "\n".join(lines) if lines else "no models match the filter"


def _render_taint_sink_entry(e: dict[str, Any]) -> list[str]:
    """One present/catalog sink entry: the model line plus, under --callsites, its
    enriched callsite rows (address + containing function + non-audit kind)."""
    p = " [present]" if e.get("present") else (" [absent]" if "present" in e else "")
    count = e.get("callsite_count")
    if count is not None:
        audit = e.get("audit_callsite_count")
        if isinstance(audit, int) and audit != count:
            cs = f" ({count} callsites, {audit} application)"
        else:
            cs = f" ({count} callsites)"
    else:
        cs = ""
    desc = f"  -- {e['model_description']}" if e.get("model_description") else ""
    out = [f"    {e.get('symbol', '<unknown>')} (arg {e.get('tainted_args')}){p}{cs}{desc}"]
    for c in _field_list(e, "callsites"):
        if not isinstance(c, dict):
            out.append(f"      {c!r}")
            continue
        fn = c.get("function") or "?"
        kind = c.get("kind")
        tag = f" [{kind}]" if kind and kind != "app_caller" else ""
        out.append(f"      {c.get('address')}  {fn}{tag}")
    return out


def _render_model_sources(sources: Any) -> str:
    """#415: one-line disclosure of the active taint-model overlays in TEXT mode
    (the default), so an agent can confirm a ``--models`` / ``BN_TAINT_MODELS``
    overlay landed without parsing JSON or restarting the bridge."""
    if not isinstance(sources, list):
        return ""
    parts: list[str] = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        kind = s.get("kind")
        if kind == "builtin":
            parts.append("builtin")
        elif kind == "env_override":
            parts.append(f"env {s.get('env', 'BN_TAINT_MODELS')} ({s.get('path')})")
        elif kind == "override_default":
            parts.append(f"override ({s.get('path')})")
        elif kind == "user":
            loc = s.get("path") or s.get("via", "--models")
            parts.append(f"--models {loc} ({s.get('count', 0)} model(s))")
    return ("models: " + " + ".join(parts)) if parts else ""


def _describe_loc(loc: Any) -> str:
    if not isinstance(loc, dict):
        return str(loc)
    kind = loc.get("kind")
    if kind == "param":
        return f"param:{loc.get('index')}"
    if kind == "var":
        return f"var:{loc.get('selector')}"
    if kind == "ret":
        return f"ret:{loc.get('callee')}"
    if kind == "arg":
        return f"arg:{loc.get('callee')}:{loc.get('index')}"
    return str(kind)


@_discloses
def _render_type_list_text(value: Any) -> str:
    # Paged envelope ({items,total,...}) -> render the page + the shared footer;
    # a bare list falls through to the per-item body below (back-compat) (#131).
    # #820: the quick-load warning goes on the ENVELOPE branch only -- the
    # recursive call is handed the bare `items` list, which carries no state (and
    # could not: a list has no place to record it), so it cannot double-prefix.
    if _field_declared(value, "items"):
        return _quick_partial_prefix(value, "type list") + _render_paged_list_text(
            value, "items", _render_type_list_text)
    if not isinstance(value, list):
        return _render_fallback_text(value)
    if not value:
        return "none"

    lines = []
    for item in value:
        if not isinstance(item, dict):
            lines.append(_render_fallback_text(item))
            continue
        name = item.get("name", "<unknown>")
        kind = item.get("kind", "<unknown>")
        decl = item.get("decl")
        line = f"{name} | {kind}"
        if decl:
            line += f" | {decl}"
        lines.append(line)
    return "\n".join(lines)


@_discloses
def _render_imports_summary_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    total = value.get("total_symbols", 0)
    # Label matches the JSON key (`total_symbols`) instead of drifting to
    # "total imports".
    lines = [f"total symbols: {total}"]
    excluded = value.get("self_defined_excluded")
    if isinstance(excluded, int) and excluded > 0:
        lines.append(f"self-defined excluded: {excluded}")
    needed = _field_list(value, "needed_libraries")
    if needed:
        lines.append("")
        lines.append("needed libraries (DT_NEEDED):")
        for lib in needed:
            lines.append(f"  {lib}")
    # Skip the breakdown sections entirely when empty (e.g. a 0-import target),
    # rather than printing dangling "by namespace:"/"by kind:" headers.
    namespaces = _field_dict(value, "namespaces")
    if namespaces:
        lines.append("")
        lines.append("by namespace:")
        for ns, count in sorted(namespaces.items(), key=lambda x: -_int_or_default(x[1])):
            lines.append(f"  {_fmt_count(count):>5}  {ns if ns else '(unnamed)'}")
    by_kind = _field_dict(value, "by_kind")
    if by_kind:
        lines.append("")
        lines.append("by kind:")
        for kind, count in sorted(by_kind.items(), key=lambda x: -_int_or_default(x[1])):
            lines.append(f"  {_fmt_count(count):>5}  {kind}")
    return "\n".join(lines)


def _render_strings_rows(value: Any) -> str:
    """Render a BARE list of string rows."""
    if not isinstance(value, list):
        return _render_fallback_text(value)
    if not value:
        return "none"

    lines = []
    for item in value:
        if not isinstance(item, dict):
            lines.append(_render_fallback_text(item))
            continue
        address = item.get("address", "<unknown>")
        length = item.get("length", "?")
        chars = item.get("chars")
        string_type = item.get("type", "")
        rendered = json.dumps(item.get("value", ""), ensure_ascii=True)
        if chars is not None and isinstance(length, int) and chars != length:
            size = f"chars={chars} bytes={length}"
        elif chars is not None:
            size = f"chars={chars}"
        else:
            size = f"len={length}"
        row = f"{address}  {size}  {string_type}  {rendered}".rstrip()
        # --probable-format-strings enrichment: surface the recovered printf
        # directives and code-xref count so the survey is scannable without
        # re-reading the JSON. Absent on a plain strings dump.
        directives = _field_list(item, "format_directives")
        if directives:
            refs = item.get("code_refs")
            suffix = f"  [fmt: {' '.join(str(d) for d in directives)}"
            if isinstance(refs, int):
                suffix += f"; code_refs={refs}"
            suffix += "]"
            row += suffix
        lines.append(row)
    return "\n".join(lines)


@_discloses
def _render_strings_text(value: Any) -> str:
    """Render strings: the paged {items, total, ...} envelope (with a footer),
    or a bare list for back-compat / internal callers (#122)."""
    return _render_paged_list_text(value, "items", _render_strings_rows)


def _render_sections_rows(value: Any) -> str:
    """Render a BARE list of section rows."""
    if not isinstance(value, list):
        return _render_fallback_text(value)
    if not value:
        return "none"

    lines = []
    for item in value:
        if not isinstance(item, dict):
            lines.append(_render_fallback_text(item))
            continue
        name = item.get("name", "<unknown>")
        start = item.get("start", "?")
        end = item.get("end", "?")
        length = item.get("length", "?")
        semantics = item.get("semantics", "")
        perms = ""
        if "readable" in item:
            perms = ("r" if item["readable"] else "-") + ("w" if item.get("writable") else "-") + ("x" if item.get("executable") else "-")
        line = f"{start}-{end}  {_fmt_count(length):>8}  {perms:>3}  {str(semantics):<20}  {name}"
        lines.append(line.rstrip())
    return "\n".join(lines)


@_discloses
def _render_cfg_text(value: Any) -> str:
    """Render the cfg result: a function header, then each block's rendered
    lines and outgoing edges. Block `start` / edge `to` are IL instruction
    indexes at IL levels (the identity contract), so they are echoed verbatim."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    func = _field_dict(value, "function")
    parts = [f"{func.get('name', '?')} @ {func.get('address', '?')} ({value.get('view', '?')})"]
    for warning in _field_list(value, "warnings"):
        parts.append(f"// {warning}")
    for block in _field_list(value, "blocks"):
        parts.append("")
        if not isinstance(block, dict):
            parts.append(f"block {block!r}")
            continue
        parts.append(f"block {block.get('start', '?')}")
        for insn in _field_list(block, "insns"):
            if not isinstance(insn, dict):
                parts.append(f"  {insn!r}")
                continue
            parts.append(f"  {insn.get('a', '?')}  {insn.get('t', '')}")
        for edge in _field_list(block, "edges"):
            if not isinstance(edge, dict):
                parts.append(f"  -> {edge!r}")
                continue
            parts.append(f"  -> {edge.get('to', '?')} [{edge.get('k', '?')}]")
    return "\n".join(parts)


@_discloses
def _render_data_vars_text(value: Any) -> str:
    """Render the data_vars window: one row per typed data variable, with the
    decoded scalar (`= v`), pointer target (`-> p sym` / `-> p "str"`), and
    section, plus a resume hint when the row cap truncated the window."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    rows = _field_list(value, "items")  # #275: was `vars`
    lines = []
    for row in rows:
        if not isinstance(row, dict):
            lines.append(f"  {row!r}")
            continue
        cells = [str(row.get("a", "?")), str(row.get("t", "?")), f"w={row.get('w', '?')}"]
        if row.get("n"):
            cells.append(str(row["n"]))
        if "v" in row:
            cells.append(f"= {row['v']}")
        if "p" in row:
            target = f"-> {row['p']}"
            if row.get("ps"):
                target += f" {row['ps']}"
            elif row.get("pstr") is not None:
                target += f' "{row["pstr"]}"'
            cells.append(target)
        if row.get("sec"):
            cells.append(f"[{row['sec']}]")
        lines.append("  ".join(cells))
    body = "\n".join(lines) if lines else "none"
    if value.get("has_more"):
        hint = ""
        # Through the choke point: a container-shaped address on the LAST row
        # dropped the resume hint, and a paged window with no way to resume is
        # exactly the confident-looking partial answer this module exists to
        # stop. The row cell above renders the container's repr, so the value is
        # not invisible -- the missing HINT was, and now it is named.
        last = _text_value(rows[-1], "a") if rows else None
        if last is not None:
            try:
                hint = f"; resume with --start {hex(int(last, 16) + 1)}"
            except ValueError:
                pass
        body += f"\n// more data vars remain in the window{hint}"
    return body


@_discloses
def _render_data_symbols_text(value: Any) -> str:
    """Render the data_symbols listing, with a paging footer when the caller
    asked for a bounded page (`--limit`/`--offset`) and more remain."""
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    syms = _field_list(value, "items")  # #275: was `syms`
    if not syms:
        return "none"
    body = "\n".join(
        f"{sym.get('a', '?')}  {sym.get('n', '')}" if isinstance(sym, dict) else f"  {sym!r}"
        for sym in syms)
    if value.get("has_more"):
        # Reaches arithmetic, so it goes through the count choke point rather
        # than a silent default: one count contract, not two (#619).
        shown = _count_field(value, "offset") + len(syms)
        if _field_skewed("offset"):
            # An ACTIONABLE number: a resume offset fabricated from an
            # unreadable page position sends a pager back over a window it
            # already read, or loops it on page one (#722's harm class).
            body += (f"\n// showing {len(syms)} of {value.get('total', '?')}"
                     f"; page position unreadable -- re-read with --format json")
        else:
            body += (f"\n// showing {shown} of {value.get('total', '?')}"
                     f"; resume with --offset {shown}")
    return body


@_discloses
def _render_sections_text(value: Any) -> str:
    """Render sections: the paged {items, total, ...} envelope (with a footer),
    or a bare list for back-compat / internal callers (#122). Prefixes a W+X
    verdict (#453) so the security question ("any writable+executable region?")
    has a direct answer instead of being inferred from per-row perms."""
    body = _render_paged_list_text(value, "items", _render_sections_rows)
    if isinstance(value, dict) and value.get("wx_verdict"):
        verdict = value["wx_verdict"]
        if verdict == "wx_sections_present":
            # Per ELEMENT, not just per field: an older bridge sent this as a
            # list of section ROWS rather than a list of names, and `", ".join`
            # raised TypeError -- which cost the whole sections listing AND
            # this W+X security verdict, the one line the view exists to
            # answer. A name that is not a name renders as itself instead
            # (#619).
            names = _field_list(value, "writable_executable_items")
            shown = ", ".join(n if isinstance(n, str) else f"<unknown: {n!r}>"
                              for n in names)
            line = f"w+x: {len(names)} section(s): {shown}"
        elif verdict == "no_wx_sections_observed":
            line = "w+x: none observed"
        else:  # unknown_insufficient_metadata (#461)
            line = ("w+x: unknown -- section metadata is insufficient (mapped/raw view "
                    "with no segment permissions); NOT an all-clear")
        return line + "\n" + body
    return body


@_discloses
def _render_read_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    hex_str = value.get("hex")
    if not isinstance(hex_str, str):
        return _render_fallback_text(value)

    try:
        data = bytes.fromhex(hex_str)
    except ValueError:
        return _render_fallback_text(value)

    address = value.get("address", "0x0")
    try:
        base = int(str(address), 16) if str(address).lower().startswith("0x") else int(address)
    except (TypeError, ValueError):
        base = 0

    lines: list[str] = []
    width = 16
    for offset in range(0, len(data), width):
        chunk = data[offset:offset + width]
        hex_bytes = " ".join(f"{b:02x}" for b in chunk)
        hex_bytes = f"{hex_bytes:<{width * 3 - 1}}"
        ascii_chunk = "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in chunk)
        lines.append(f"{base + offset:08x}: {hex_bytes}  {ascii_chunk}")

    if not lines:
        lines.append(f"{base:08x}: (no bytes)")

    # Choke point: a container-shaped `note` dropped the note ENTIRELY, so a
    # truncated or partial read rendered byte-identically to a complete one
    # (#619).
    note = _text_value(value, "note")
    if note:
        lines.append("")
        lines.append(f"note: {note}")

    return "\n".join(lines)


@_discloses
def _render_doctor_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    lines = [
        f"cli version: {value.get('cli_version', '<unknown>')}",
        f"plugin source: {value.get('plugin_source_dir', '<unknown>')}",
        f"plugin install: {value.get('plugin_install_dir', '<unknown>')}",
        f"plugin source build: {value.get('plugin_source_build_id', '<unknown>')}",
        f"plugin install build: {value.get('plugin_install_build_id', '<unknown>')}",
        "",
        "instances:",
    ]
    instances = _field_list(value, "instances")
    if not instances:
        lines.append("- none")
        return "\n".join(lines)

    for item in instances:
        if not isinstance(item, dict):
            lines.append("- " + _render_fallback_text(item))
            continue
        doctor = _field_dict(item, "doctor")
        # Prefer the status the JSON carries (L16) so text and JSON can't drift;
        # fall back to deriving it for any caller that built the dict the old way.
        status = item.get("status") or ("ok" if doctor and not doctor.get("error") else "error")
        lines.append(
            "- "
            + f"pid={item.get('pid', '<unknown>')} plugin={item.get('plugin_version', '<unknown>')} status={status}"
        )
        build_id = item.get("plugin_build_id")
        if build_id:
            lines.append(f"  build: {build_id}")
        # Name the engine the bridge is actually driving: with two BN majors in
        # play (5.x vs 6.x) the same command can behave differently, and nothing
        # else in doctor output distinguishes them.
        bn_version = item.get("bn_version")
        if bn_version:
            bn_build = item.get("bn_build_id")
            suffix = f" (build {bn_build})" if bn_build else ""
            lines.append(f"  binary ninja: {bn_version}{suffix}")
        if item.get("stale_plugin_version"):
            lines.append("  stale: loaded plugin version differs from CLI version")
        if item.get("stale_plugin_code"):
            lines.append("  stale: loaded plugin code does not match installed plugin file")
        if item.get("stale_engine"):
            lines.append(
                "  stale: loaded engine code (taint/IL modules) is out of date -- "
                "run `bn session restart " + str(item.get("instance_id") or "<id>") + "`")
        if item.get("started_at"):
            lines.append(f"  started: {item['started_at']}")
        if item.get("socket_path"):
            lines.append(f"  socket: {item['socket_path']}")
        error = doctor.get("error")
        if error:
            lines.append(f"  error: {error}")
    return "\n".join(lines)


def _operation_row(item: dict[str, Any]) -> tuple[str, list[str]]:
    """One mutation op result as a row, plus the fields it could not read.

    The keys are read under a LOCAL recorder and handed back, because both
    callers need the answer and neither can rely on a boundary: `bn.cli`
    re-exports `_format_operation_result` so tests and scripts can call it
    DIRECTLY, and a direct caller installs no ``@_discloses`` boundary at all --
    so a skew recorded here drained nowhere and a malformed ``requested``
    rendered byte-identically to the field being absent (#619). Same idiom as
    ``_add_mutation_ok`` and the go-rename summary: decide locally, disclose in
    the row, and let the row say WHICH op could not be read, which an aggregate
    note at the end of the card cannot."""
    token = _SKEWED_FIELDS.set([])
    try:
        row = _operation_row_text(item)
        unreadable = sorted(_SKEWED_FIELDS.get() or ())
    finally:
        _SKEWED_FIELDS.reset(token)
    return row, unreadable


def _format_operation_result(item: dict[str, Any]) -> str:
    """The op row a DIRECT caller gets, disclosure included."""
    row, unreadable = _operation_row(item)
    return f"{row}  {_skew_note(*unreadable)}" if unreadable else row


def _operation_row_text(item: dict[str, Any]) -> str:
    op = item.get("op", "<unknown>")
    if not isinstance(op, str):                    # an unhashable op crashed the `in` test
        op = str(op)
    requested = _field_dict(item, "requested")

    def _get(key: str, default: str = "<unknown>") -> str:
        return item.get(key) or requested.get(key, default)

    if op == "function_create":
        name = item.get("function")
        suffix = f" ({name})" if name else ""
        return f"function_create {_get('address')}{suffix}"
    if op == "rename_symbol":
        return f"rename_symbol {_get('kind', 'auto')} {_get('address')} -> {_get('new_name')}"
    if op == "set_comment":
        target = item.get("function") or requested.get("function") or _get("address")
        return f"set_comment {target}"
    if op == "delete_comment":
        target = item.get("function") or requested.get("function") or _get("address")
        return f"delete_comment {target}"
    if op == "set_prototype":
        return f"set_prototype {_get('function')} @ {_get('address')}"
    if op in {"local_rename", "local_retype"}:
        target = item.get("local_id") or item.get("variable") or requested.get("variable", "<unknown>")
        return f"{op} {_get('function')}::{target}"
    if op == "struct_field_set":
        return (
            f"struct_field_set {_get('struct_name')} "
            f"{_get('offset')} {_get('field_name')} {_get('field_type')}"
        )
    if op == "struct_field_rename":
        return (
            f"struct_field_rename {_get('struct_name')} "
            f"{_get('old_name')} -> {_get('new_name')}"
        )
    if op == "struct_field_delete":
        return f"struct_field_delete {_get('struct_name')}::{_get('field_name')}"
    if op == "types_declare":
        # Name the type(s) defined, not a bare count -- "which type?" is the first
        # thing an agent needs. Parser bookkeeping (parsed functions/variables) is
        # internal noise and moves out of the default line.
        declared = _field_dict(item, "defined_types")
        names = [str(name) for name in declared]
        if names:
            return f"types_declare {', '.join(names)}"
        # No names, so the COUNT is the whole claim -- and it may only be stated
        # when the payload stated it. `item.get("count", 0)` over an UNREADABLE
        # listing printed "types_declare 0 types", which reads as a declare that
        # defined nothing: the fabricated zero #683 discarded a committed rename
        # batch to, one op over. Disclosing it is not enough either, because the
        # count is what a control loop reads and the note is not. So the row
        # REFUSES the claim, exactly as the go-rename view does.
        #
        # Three ways the payload can fail to state it, and the third was still
        # printing the fabricated zero: an unreadable listing, an unreadable
        # count, and NEITHER KEY AT ALL. A readable listing is a measurement
        # even when it is empty -- we looked and defined none -- so that zero
        # stays; an envelope carrying no listing and no count measured nothing,
        # and "0 types" is the exact reading #683 acted on.
        stated = _field_present(item, "count")
        if _field_skewed("defined_types") and not stated:
            return "types_declare <unreadable defined_types>"
        count = _count_field(item, "count")
        if _field_skewed("count"):
            return "types_declare <unreadable count>"
        if not stated and not _field_present(item, "defined_types"):
            return "types_declare <count not stated>"
        return f"types_declare {count} types"
    return _render_fallback_text(item)


_BN_CONVENTION_RE = re.compile(r'__convention\("([^"]+)"\)')


def _clean_prototype(proto: Any) -> str | None:
    """Render BN's prototype readably: ``__convention("cdecl")`` -> ``__cdecl``."""
    if not isinstance(proto, str) or not proto:
        return None
    return _BN_CONVENTION_RE.sub(r"__\1", proto).strip()


def _set_prototype_detail(item: dict[str, Any]) -> list[str]:
    # BOTH reads go through a choke point, not an inline isinstance. An unusable
    # `observed` dropped the prototype line and left the row byte-identical to a
    # row that carried no observation at all -- the renderer's whole subject is
    # that the prototype it SET was verified (#619). The same is true one level
    # in: a well-formed `observed` whose `prototype` is not a string dropped the
    # same line just as silently, which is the leaf under the container this
    # comment already claimed to have closed.
    proto = _clean_prototype(_text_value(_field_dict(item, "observed"), "prototype"))
    return ["  " + proto] if proto else []


def _layout_size(layout: Any) -> str | None:
    """Pull the ``size=0x..`` (or decimal) off a rendered type layout's header."""
    if not isinstance(layout, str) or not layout:
        return None
    match = re.search(r"size=(0x[0-9a-fA-F]+|\d+)", layout.splitlines()[0])
    return match.group(1) if match else None


def _layout_field_count(layout: Any) -> int:
    if not isinstance(layout, str):
        return 0
    return sum(1 for line in layout.splitlines()[1:] if line.strip().startswith("0x"))


def _size_delta(before_layout: Any, after_layout: Any) -> str | None:
    after = _layout_size(after_layout)
    if after is None:
        return None
    before = _layout_size(before_layout)
    if before is None or before == after:
        return f"size {after}"
    try:
        delta = int(after, 0) - int(before, 0)
    except (TypeError, ValueError):
        return f"size {before} -> {after}"
    return f"size {before} -> {after} ({'+' if delta >= 0 else ''}{delta})"


def _layout_field_deltas(layout_diff: Any) -> list[str]:
    """The +/- field lines from a unified layout diff (skips the struct-header and
    @@ hunk lines), so a type change shows just the fields that moved."""
    out: list[str] = []
    for line in (layout_diff or "").splitlines() if isinstance(layout_diff, str) else []:
        if len(line) >= 2 and line[0] in "+-" and line[1:].lstrip().startswith("0x"):
            out.append(f"  {line[0]} {line[1:].strip()}")
    return out


def _types_affected_lines(value: dict[str, Any]) -> list[str]:
    entries = [e for e in (_field_list(value, "affected_types")) if isinstance(e, dict)]
    multi = len(entries) > 1
    out: list[str] = []
    for entry in entries:
        name = entry.get("type_name") or entry.get("name") or "<type>"
        # Only prefix the type name when a batch touched more than one type --
        # for a single type the op-summary header already names it.
        prefix = f"{name}: " if multi else ""
        # All three layout reads go through the text choke point, ONCE each, and
        # the locals are passed on from there. Read raw, every one of them was an
        # inline `isinstance(..., str)` filter inside the layout helpers, and a
        # container-shaped value therefore dropped its clause byte-identically
        # to the key being ABSENT with nothing disclosed: no size line, no
        # field deltas. The unchanged branch was worse than silent -- `after`
        # kept the container (a dict is truthy, so `or ""` never fired) and
        # `after.strip()` was `AttributeError: 'dict' object has no attribute
        # 'strip'`, which cost the WHOLE mutation card where the same entry
        # without `after_layout` rendered cleanly (#619).
        # Read inside a per-ENTRY capture so the third state stays attributable
        # to THIS entry: `_field_skewed` answers for the whole render, and a
        # batch's earlier entry would otherwise decide this one's count. Then
        # re-record, so the card still discloses by name.
        token = _SKEWED_FIELDS.set([])
        try:
            before_layout = _text_value(entry, "before_layout")
            after_layout = _text_value(entry, "after_layout")
            layout_diff = _text_value(entry, "layout_diff")
            unreadable = list(_SKEWED_FIELDS.get() or ())
        finally:
            _SKEWED_FIELDS.reset(token)
        for key in unreadable:
            _record_skew(key)
        if entry.get("changed"):
            before_sz = _layout_size(before_layout)
            after_sz = _layout_size(after_layout)
            deltas = _layout_field_deltas(layout_diff)
            if before_sz is not None and after_sz is not None and before_sz != after_sz:
                out.append(f"  {prefix}{_size_delta(before_layout, after_layout)}")
            elif not deltas:
                # No field/size delta to show (e.g. a decl-only change) -- fall back
                # to the size so the line isn't empty.
                out.append(f"  {prefix}{_size_delta(before_layout, after_layout) or 'changed'}")
            # else: size unchanged but fields moved (e.g. a rename) -- the +/- field
            # lines below carry the change; a 'size 0xNN' line would just be noise.
            out.extend(deltas)
        else:
            after = after_layout or ""
            head = after.splitlines()[0].strip() if after.strip() else f"struct {name}"
            if "after_layout" in unreadable:
                # ONE count contract in this module, not two. A field count
                # derived from a layout nobody could read is #683's fabrication
                # -- "0 fields" is "this type is empty" to a reader -- and the
                # note beside it is not what a control loop reads. The op row
                # prints `<count not stated>` for exactly this class; so does
                # this row. An ABSENT or EMPTY layout is still a measured zero
                # and still says "0 fields".
                out.append(f"  {head}, field count not stated (unreadable layout)")
            else:
                count = _layout_field_count(after)
                out.append(f"  {head}, {count} field{'s' if count != 1 else ''}")
    return out


def _blast_radius_line(value: dict[str, Any]) -> str | None:
    """One line of blast radius for a type op: how many functions reference the
    type and how many actually reflowed, with a few names (reflowed first)."""
    # Through the choke point: a malformed summary dropped this entire line with
    # no note. An empty one yields no line either way (`referenced` is 0), so
    # only the silence changes (#619).
    summary = _field_dict(value, "affected_summary")
    if not summary:
        return None
    # Through the COUNT choke point, and refusing rather than fabricating. Both
    # of `int(source.get(key) or 0)`'s failure modes were live here -- the exact
    # expression `_count_field`'s docstring quotes as the bug it replaces. A
    # string, dict or list `referenced` RAISED, and the CLI degrades that to
    # exit 2 with empty stdout, so a mutation that COMMITTED reported no card at
    # all; a bool silently fabricated "referenced by 1 fn, 2 reflowed" with no
    # note, in the same render where the op row prints `<count not stated>`.
    # One count contract in this module, on every surface that states one.
    token = _SKEWED_FIELDS.set([])
    try:
        referenced = _count_field(summary, "referenced")
        reflowed = _count_field(summary, "reflowed")
        unreadable = sorted(_SKEWED_FIELDS.get() or ())
    finally:
        _SKEWED_FIELDS.reset(token)
    for key in unreadable:
        _record_skew(key)
    if unreadable:
        return ("  blast radius not stated: " + ", ".join(unreadable)
                + " could not be read")
    if referenced <= 0:
        return None
    # A directly-mutated function (set_prototype/rename target, tagged `direct`)
    # is not part of the type's reference set, so keep it out of these names --
    # in a mixed batch it belongs under the direct op's affected block instead.
    affected = [a for a in (_field_list(value, "affected_functions"))
                if isinstance(a, dict) and not a.get("direct")]
    names = [a.get("after_name") or a.get("before_name") for a in affected if a.get("changed")]
    names += [a.get("after_name") or a.get("before_name") for a in affected if not a.get("changed")]
    names = [n for n in names if n]
    line = f"  referenced by {referenced} fn{'s' if referenced != 1 else ''}, {reflowed} reflowed"
    if names[:5]:
        line += ": " + ", ".join(names[:5])
        if referenced > len(names[:5]):
            line += f" (+{referenced - len(names[:5])} more)"
    return line


def _is_type_result(result: Any) -> bool:
    """A type-shape result (type (re)declaration or struct field edit), whose
    blast radius is "functions referencing the type" -- as opposed to a direct op
    (rename/prototype/comment) that targets one specific function."""
    if not isinstance(result, dict):
        return False
    op = str(result.get("op") or "")
    return op == "types_declare" or op.startswith("struct_")


def _format_op_summary(item: dict[str, Any]) -> str:
    # The row's own status/message suffixes go on the ROW, and the disclosure
    # note goes last -- calling `_format_operation_result` here would append the
    # note first and leave ` [verified]` dangling after it.
    summary, unreadable = _operation_row(item)
    if item.get("status"):
        summary += f" [{item['status']}]"
    if item.get("changed") is False and item.get("status") not in (None, "noop"):
        summary += " [no change]"
    if item.get("message"):
        summary += f" ({item['message']})"
    return f"{summary}  {_skew_note(*unreadable)}" if unreadable else summary


def _add_mutation_ok(value: Any) -> Any:
    """Add a top-level ``ok`` boolean to a full mutation/batch result so a uniform
    ``jq '.ok'`` check works across read and mutation commands (#447). ``ok`` is
    the verification-aware success: the bridge-reported ``success`` AND no failed
    op status. Additive -- ``success``/``committed`` are unchanged, and an ``ok``
    the payload already carries is left alone.

    One exception, and it is the whole point of the exception: on an op that
    reports through its OWN counters (``go rename``) the op's compact status
    decides ``ok``, this transform does not re-derive it, and it OVERRIDES an
    ``ok`` already on the payload -- because the compact path overrides it too,
    and the two must not answer differently (#447)."""
    if not isinstance(value, dict):
        return value
    # `go rename` reports through its own counters, and this transform reads
    # only `success` and `results[]` -- which for that op holds the failure
    # rows ALONE. A counter arriving in a shape no count reads out of therefore
    # left `ok: true` HERE while the op's compact status -- the CLI's default
    # since #645 -- refused the identical payload, so a uniform `jq '.ok'`
    # flipped on whether the caller asked for detail: #447's half-parity again,
    # on the one channel `_go_rename_summary` exists to read. The CLI selects
    # between these two transforms on `--verbose`/`--out`/an explicit machine
    # `--format`, so the flip is one flag away on the op whose whole reason for
    # existing is #683.
    #
    # DELEGATED rather than re-derived. A second derivation of this op's `ok`
    # is what drifted in the first place, and it would drift again the next
    # time either side learns a new refusal -- the rows/counter contradiction
    # already exists on one side only.
    #
    # ABOVE the already-has-`ok` short-circuit, and OVERRIDING any `ok` the
    # payload arrived with, because the compact path overrides it too: the
    # summary builder states `ok` unconditionally. Left below the
    # short-circuit, an envelope that carried a stale `ok: true` kept it here
    # and was refused there -- the same flip the delegation exists to close,
    # surviving behind the guard that was supposed to make this transform
    # idempotent. One decider means one on EVERY input (#447/#619/#685).
    if value.get("kind") == "go_rename":
        rest = {key: val for key, val in value.items() if key != "ok"}
        return {"ok": bool(_go_rename_summary(value)["ok"]), **rest}
    if "ok" in value:
        return value
    # Read inside a capture: this transform runs BEFORE any renderer, so the
    # choke point has no boundary to record into. `ok` is "the bridge reported
    # success AND no op row failed"; with the rows unreadable the second half is
    # not established, so it must not be claimed. Fail safe for the same reason
    # `dirty_after` does: a spurious `ok: false` costs a look, a fabricated
    # `ok: true` closes an agent's control loop on a batch nobody checked (#619).
    token = _SKEWED_FIELDS.set([])
    try:
        results = _row_list(value, "results")
        # INSIDE the capture, not after it. A row whose `status` is unreadable
        # is not a row that passed, and reading it outside the capture put its
        # skew in the default (absent) recorder: `unusable` stayed False and the
        # batch claimed `ok: true` over a row nobody could classify.
        failed = any(_is_failed_status(r) for r in results)
        unusable = bool(_SKEWED_FIELDS.get())
    finally:
        _SKEWED_FIELDS.reset(token)
    return {"ok": bool(value.get("success", True)) and not failed and not unusable,
            **value}


# The unmeasured explanation, in ONE place, with the cause as a hole.
#
# The text renderer has to name the CAUSE too -- `skills/bn/reference/mutating.md`
# quotes its warning line verbatim as documented sample output -- and the
# summary carries no separate key for it, because the key set IS the documented
# #685 contract. So the cause travels in `first_error` and the renderer reads it
# back out by splitting on this same template: the format and the parse come
# from one string, and moving the template moves both.
# `test_the_unmeasured_cause_round_trips_for_every_cause` asserts that.
_UNMEASURED_NOTE = (
    "unmeasured: {cause}, so changed/verified/noop/failed counts could not be "
    "derived (None, not a confirmed 0) and dirty_after defaults to True as a "
    "fail-safe -- do not assume nothing changed"
)
_UNMEASURED_HEAD, _UNMEASURED_TAIL = _UNMEASURED_NOTE.split("{cause}")


def _unmeasured_cause(first_error: Any) -> str:
    """The cause phrase back out of a summary's ``first_error``, or ``""``.

    Empty when the note is absent or malformed rather than guessing: a renderer
    that fabricated a cause would be asserting the one thing it does not know,
    which is exactly what the hardcoded "this op reported no results[] rows"
    warning did once a second cause existed (#619)."""
    text = first_error if isinstance(first_error, str) else ""
    start = text.find(_UNMEASURED_HEAD)
    if start < 0:
        return ""
    rest = text[start + len(_UNMEASURED_HEAD):]
    end = rest.find(_UNMEASURED_TAIL)
    return rest[:end] if end >= 0 else ""


def _build_mutation_summary(
    *,
    measured: bool,
    op_count: int,
    reported_success: bool,
    failure_rows: Sequence[dict[str, Any]],
    failed: int | None,
    changed: int | None,
    verified: int | None,
    noop: int | None,
    committed: bool,
    preview: bool,
    rolled_back: Any,
    message: Any,
    proto_residue: bool = False,
    default_error: str = "mutation failed",
    unmeasured_cause: str = "this op reported no results[] rows",
    unusable: bool = False,
) -> dict[str, Any]:
    """The ONE compact-status schema every mutation summary emits (#685).

    The callers differ only in their INPUTS: `_mutation_summary` derives its
    counts from `results[]`, while `go rename` reports through its own `go_*`
    counters. Everything that must not drift between them lives here -- the key
    set, the `ok`/`success` mirroring (#447), the `first_error` fallbacks, the
    `dirty_after` rule and the conditional residue key (#630). This table is
    documented in `skills/bn/reference/mutating.md`: a key change is a contract
    change.
    """
    # `unusable` is the one answer this module gives to "can success be claimed
    # off a payload we could not read". `_add_mutation_ok` already withholds
    # `ok` when `results[]` is unreadable; the compact summary claimed
    # `ok: true` on the same payload, so a uniform `jq '.ok'` -- the entire
    # point of #447 -- FLIPPED depending on whether `--summary` was passed.
    # Merely EMPTY rows are not unusable and both paths still say ok there,
    # which is what keeps this parity rather than a behaviour change (#619).
    success = reported_success and not failed and not unusable
    # The failure explanation, in the one order both ops share: the first failure
    # ROW's own message/status (an `unsupported` early return puts its only
    # explanation in `results[0]["message"]`), then the top-level `message` (a
    # revert that failed AFTER every op verified has no failure row at all), then
    # a per-op default. Gating on `not success` rather than on `failed` is what
    # keeps the last two reachable while `failed` is 0.
    # The explanation goes through the TEXT choke point, not a raw read: a
    # container `message` on a failure row landed in the documented schema's
    # `first_error` as a dict, and the compact renderer printed its Python
    # repr where an agent reads the one key its contract tells it to check.
    # Unreadable therefore means "this row explained nothing" -- the next
    # fallback answers instead -- and the boundary names the field (#619/#685).
    first_error: Any = None
    if not success:
        for row in failure_rows:
            first_error = _text_value(row, "message") or _text_value(row, "status")
            if first_error:
                break
        if first_error is None:
            first_error = message or default_error
    if proto_residue:
        # An unclearable has_user_type override left behind by a reverted
        # proto-set on an AUTO function is behaviorally meaningful residue that a
        # control loop must see even in the compact summary (#630): it means the
        # view is left modified. Prefer the residue-explaining top-level message
        # -- a bare failed-row status ("rollback_failed") does not tell the loop
        # what actually went wrong.
        first_error = message or first_error or (
            "prototype has_user_type override could not be cleared"
        )
    unmeasured = not measured
    if unmeasured:
        # Review of the first cut of this fix (#684): `dirty_after: None` is
        # FALSY under every truthiness check a control loop actually writes --
        # `jq 'if .dirty_after then'`, `if summary["dirty_after"]:`,
        # `if (!s.dirty_after) close()`. An unmeasured envelope and a confirmed
        # all-noop therefore produced the IDENTICAL "skip save" decision, which
        # made the original fix a no-op on the JSON path. Fail safe instead:
        # the derived counts are unknown (None, not a confident 0) and
        # `dirty_after` defaults to True, so a naive falsy check SAVES. A
        # spurious `bn save` is cheap; a discarded rename batch is the #683 bug
        # this guard exists to catch. Also load `first_error` -- the one
        # summary key an agent contract already tells callers to check -- with
        # the same explanation, layered on top of whatever failure message was
        # already found above.
        unmeasured_explanation = _UNMEASURED_NOTE.format(cause=unmeasured_cause)
        first_error = (f"{first_error} ({unmeasured_explanation})" if first_error
                       else unmeasured_explanation)
    summary = {
        "kind": "mutation_summary",
        # Top-level `ok` mirrors the read-command envelope so a uniform `jq '.ok'`
        # works across reads AND mutations (batch/mutation JSON used only
        # success/committed, so `.ok` read null -- #447).
        "ok": success,
        "success": success,
        "committed": committed,
        "preview": preview,
        # False when `results[]` came back empty on an op that was expected to
        # populate it. The four derived counts below are then UNKNOWN (None, not
        # a confident zero); `op_count` stays 0 (literally true -- zero rows) and
        # `dirty_after` is a deliberate fail-safe True, NOT unknown (#684).
        "measured": not unmeasured,
        "op_count": op_count,
        "changed_count": (None if unmeasured else changed),
        "verified_count": (None if unmeasured else verified),
        "noop_count": (None if unmeasured else noop),
        "failed_count": (None if unmeasured else failed),
        # True/False when a revert was attempted; None when none was needed.
        "rolled_back": (bool(rolled_back) if rolled_back is not None else None),
        "first_error": first_error,
        # The DB is left modified iff a live mutation actually CHANGED state
        # (committed AND something changed -- `committed` is True even for an
        # all-noop mutation, which leaves the DB clean), a failure's revert
        # itself failed, or an unclearable has_user_type override was left
        # behind (#630).
        #
        # The revert test is `rolled_back is False AND not committed`, not
        # `rolled_back is False` alone: #652 made the bridge emit `rolled_back`
        # unconditionally (`restored if (preview or failed) else False`), so the
        # SUCCESS path now carries an explicit False where the key used to be
        # absent. Since `committed` is `(not preview) and (not failed)`, the key
        # is meaningful exactly when `committed` is False -- gating on it is what
        # keeps an all-noop commit (which reverts nothing) from reading dirty.
        #
        # UNMEASURED overrides all of the above to True (not None): see the
        # fail-safe rationale above (#684 review). An agent that never reads
        # `measured` still gets the safe answer from `dirty_after` alone.
        "dirty_after": (True if unmeasured else (
            (committed and bool(changed))
            or (rolled_back is False and not committed)
            or proto_residue
        )),
    }
    if proto_residue:
        summary["prototype_user_type_residue"] = True
    return summary


@_discloses_in_summary
def _mutation_summary(value: Any) -> Any:
    """#408: collapse a (single or batch) mutation result into a compact,
    schema-stable status object for an unattended agent control loop -- did
    anything change, did verification pass, was anything rolled back, what needs
    attention -- without parsing the full results/affected_functions/diff payload.
    The detailed result stays available without --summary.

    Derives the shared builder's inputs from `results[]` (#685)."""
    if not isinstance(value, dict):
        return value
    # Idempotent: `_call` evaluates `spill_status` against the ALREADY-transformed
    # result, so on the compact path this runs on its own output. Without this
    # guard the second pass sees no `results` and re-zeroes every count -- today
    # only wasted work (a ~200-byte summary never crosses the spill threshold),
    # but a spilled mutation would print an all-zero status.
    if value.get("kind") == "mutation_summary":
        return value
    # Read inside a NESTED capture so this transform can tell an UNREADABLE
    # listing from an empty one, then re-record so the enclosing drain still
    # discloses it by name. `_add_mutation_ok` withholds `ok` on exactly this
    # payload; the compact summary must give the same answer (#447/#619).
    token = _SKEWED_FIELDS.set([])
    try:
        results = _row_list(value, "results")
        # The THIRD state of the row set, asked of the choke point rather than
        # spelled here: a second decider over "was this readable" is the defect
        # this module keeps re-growing, and `_field_skewed` is where that
        # question already lives. Asked INSIDE the capture, so it answers about
        # THIS read and not about some same-named key an outer render recorded.
        rows_unreadable = _field_skewed("results")
        # The ROW STATUSES are classified inside the capture too, for exactly the
        # reason the listing is. Read outside it, an unreadable `status` still
        # DISCLOSED -- the enclosing summary drain caught it -- but it never
        # reached `unusable`, so this transform answered `ok: true` on the same
        # payload `_add_mutation_ok` refuses, and a uniform `jq '.ok'` FLIPPED
        # depending on whether `--summary` was passed. Half a parity is not a
        # parity: a row nobody could classify is not a row that passed, on both
        # derivations or on neither (#447/#619).
        failed = [r for r in results if _is_failed_status(r)]
        unreadable = list(_SKEWED_FIELDS.get() or ())
    finally:
        _SKEWED_FIELDS.reset(token)
    for key in unreadable:
        _record_skew(key)
    verified = sum(1 for r in results if r.get("status") == "verified")
    noop = sum(1 for r in results if r.get("status") == "noop")
    # #684: every genuine `mutation_engine` op populates at least one `results[]`
    # row per requested operation -- `_mutation()` refuses an empty operation list
    # outright -- so an op that reaches this GENERIC summary (no registered
    # `summary_transform`) with an EMPTY `results[]` is never reporting a real
    # zero-change measurement. It means the op reports through its OWN counters
    # instead (the shape `_go_rename_summary` exists to handle) and forgot to
    # register that escape hatch. A genuine zero-change result (e.g. a rename that
    # matched the current name already) still comes through as a `noop` STATUS ROW
    # inside a non-empty `results[]` -- it stays measured and distinct from the
    # unmeasured case the builder fails safe on.
    #
    # An unreadable ROW SET is unmeasured for the same reason an empty one is,
    # and at EVERY granularity it can be unreadable at: `results` arriving as
    # the wrong FIELD already landed here (the choke point hands back an empty
    # list), while a list one ELEMENT of which is not a row, or a row whose
    # STATUS nobody could read, used to leave the four derived counts stated
    # off what survived -- `failed_count: 0` over a batch carrying a row that
    # could not be classified, which is the fabricated zero the builder's
    # fail-safe exists to stop, with `dirty_after` falling to False beside it.
    # `ok` alone is not enough: a loop that branches on `dirty_after` or reads
    # `failed_count` never looks at it.
    #
    # Derived from the whole capture rather than from the row-set read alone,
    # because the sibling caller of the one builder derives it that way
    # (`_go_rename_summary`: `measured = not unreadable and ...`) and the two
    # answering `measured` differently for the same defect is the same drift
    # #685 exists to close, one key over from `ok` (#619/#685).
    unmeasured = not results or bool(unreadable)
    return _build_mutation_summary(
        measured=not unmeasured,
        # The rows this summary could READ, which is why it stays an int while
        # the four derived counts go unknown: it is a literal count of what
        # reached the classifier, not a claim about how many ops the batch
        # requested. When the summary is unmeasured that distinction matters
        # and the warning beside it names the CAUSE -- no rows, an unreadable
        # row set, an unclassifiable status -- without asserting any size for
        # the batch, because nothing here measured one.
        op_count=len(results),
        reported_success=bool(value.get("success", True)),
        failure_rows=failed,
        failed=len(failed),
        changed=verified,
        verified=verified,
        noop=noop,
        committed=bool(value.get("committed", False)),
        preview=bool(value.get("preview", False)),
        rolled_back=value.get("rolled_back"),
        # Through the text choke point for the same reason the failure row's
        # message is: this value IS the documented `first_error` on the
        # no-failure-row path, and a container there printed as a repr.
        message=_text_value(value, "message"),
        proto_residue=bool(value.get("prototype_user_type_residue")),
        unusable=bool(unreadable),
        # The public reference quotes the no-rows phrase verbatim as the
        # meaning "this op reports through its own counters instead", so an
        # unreadable ROW must not borrow it: a reader sent to look for a
        # missing `results[]` would find one, populated, and stop.
        unmeasured_cause=("this op's results[] could not be read"
                          if rows_unreadable
                          else "an op row's status could not be read" if unreadable
                          else "this op reported no results[] rows"),
    )


# `go rename`'s own counters -- the ONE list, and the read loop below iterates
# it, so a seventh counter cannot be read outside the capture that decides
# whether this summary was measured at all. A tuple maintained BESIDE the reads
# would be a second declaration to drift; this one IS the reads.
_GO_RENAME_COUNTERS = (
    "go_renamed_candidates",
    "go_committed_count",
    "go_verified_count",
    "go_failed_count",
    "skipped_user_named",
    "skipped_changed_during_apply",
)


@_discloses_in_summary
def _go_rename_summary(value: Any) -> Any:
    """Compact status for `go rename`, which reports through its OWN counters.

    `_mutation_summary` derives every count from `results[]`. For this op that
    array holds only the FAILURE rows (bridge `"results": failed_rows`), while
    the work done is reported via `go_renamed_candidates` / `go_verified_count`
    / `go_committed_count` / `go_failed_count` / `skipped_user_named`. Running it
    through the generic summary therefore rendered a run that renamed 1783
    functions as `changed=0 ... dirty_after=False`, and a caller reading that
    closes without saving and silently discards every recovered name.

    Derives the shared builder's inputs from those counters (#685), so the
    compact SCHEMA -- and with it the `first_error` and `dirty_after` rules --
    has exactly one definition.
    """
    if not isinstance(value, dict) or value.get("kind") != "go_rename":
        return _mutation_summary(value)
    committed = bool(value.get("committed", False))
    preview = bool(value.get("preview", False))
    # EVERY counter this summary derives a decision key from, read in ONE place
    # so "which counters feed the decision" and "which counters were readable"
    # cannot become two answers that drift apart. `measured` is DERIVED from
    # those reads; asserting it True was a live defect, because `_count_field`
    # answers 0 for a counter it could not read and 0 is this op's
    # "nothing happened, do not save" verdict (#619/#683).
    #
    # Read inside a NESTED capture and re-recorded afterwards: the enclosing
    # `@_discloses_in_summary` still discloses each unreadable counter by name,
    # and the answer here does not depend on a boundary being installed.
    token = _SKEWED_FIELDS.set([])
    try:
        counters = {key: _count_field(value, key) for key in _GO_RENAME_COUNTERS}
        # The failure ROWS are read inside the capture too. Read in the
        # builder's argument list -- after the reset -- an unreadable
        # `results[]` still DISCLOSED through the enclosing summary drain but
        # never reached `unusable`, so this op's compact status claimed
        # `ok: true` on the payload `_add_mutation_ok` refuses. The compact path
        # is the DEFAULT for `go rename`, so `jq '.ok'` flipped on whether the
        # caller asked for detail -- the same half-parity #447 forbids, on the
        # second caller of the one builder (#619/#685).
        failure_rows = _row_list(value, "results")
        # And the row STATUSES are classified here, which closing the list
        # granularity alone still left out. `ok` and `failed_count` came off
        # `go_failed_count` ALONE, so a row that NAMES a failure -- or one whose
        # status nobody could read -- beside a counter saying zero produced
        # `ok: true, failed_count: 0, first_error: null` with no disclosure,
        # while `_add_mutation_ok` classifies those same rows and refuses. The
        # compact path is this op's DEFAULT, so `jq '.ok'` flipped on the detail
        # flag again, one granularity in. Both of `_add_mutation_ok`'s
        # derivations now happen here: the unreadable status records its own
        # skew inside this capture, and a readable failure row is counted below.
        row_failures = [r for r in failure_rows if _is_failed_status(r)]
        unreadable = list(_SKEWED_FIELDS.get() or ())
    finally:
        _SKEWED_FIELDS.reset(token)
    for key in unreadable:
        _record_skew(key)
    candidates = counters["go_renamed_candidates"]
    committed_count = counters["go_committed_count"]
    verified = counters["go_verified_count"]
    failed = counters["go_failed_count"]
    skipped = counters["skipped_user_named"]
    rolled_back = value.get("rolled_back")

    # `changed` is what is LIVE in the view when the call returns, never the plan:
    #   committed -> what actually landed;
    #   preview   -> what WOULD land, i.e. the rows that verified (NOT the
    #                candidate count, which over-reports every candidate the
    #                apply skipped because it changed underneath us);
    #   otherwise -> a live run that failed and was reverted: nothing landed.
    # Reporting `candidates` in that last case is the mirror image of the bug
    # this function exists to fix, and `_render_go_rename_text` already refuses
    # to claim "N renamed" for rows that passed readback before a revert.
    # The counter `changed` is READ FROM is this summary's measurement source,
    # and the third state matters here exactly as it does for a container:
    # ABSENT means the op claimed nothing, so there is nothing to report
    # against. `_count_field` answers 0 for an absent key, which on this op is
    # the "nothing changed, do not save" verdict -- so a partial or
    # version-skewed envelope with the counters MISSING produced the decision
    # keys of a genuine all-noop commit, reaching #683's harm by absence
    # instead of by wrong shape. `_mutation_summary` already calls its own
    # missing measurement source unmeasured; both callers of the one builder
    # must answer that the same way (#619/#685).
    if committed:
        changed = committed_count
        source = "go_committed_count"
    elif preview:
        changed = verified
        source = "go_verified_count"
    else:
        # Nothing landed, and that is established by the revert rather than by
        # a counter -- so there is no measurement source to require.
        changed = 0
        source = None
    # The rows and the counter answer the SAME question on this op, and the
    # bridge builds them from ONE list: `"results": failed_rows` beside
    # `"go_failed_count": len(failed_rows)`. So they may not disagree in
    # EITHER direction, and refusing is the only answer available here --
    # inventing a count from whichever side looks more trustworthy would be
    # the mirror fabrication.
    #
    # The earlier cut refused only `rows > counter` and excused the other
    # direction as a CAPPED failure listing. The payload is not capped: only
    # the `--verbose` text view's display is (`_render_go_rename_text` prints
    # 50 rows and then "... and N more"), and the failure count it states
    # beside them is the full row count. Believing the larger counter
    # therefore let that view and this compact summary state DIFFERENT failure
    # counts for one payload while both read `measured` -- the drift between
    # an op's two views that #685 exists to close, one key over from `ok`.
    rows_contradict = len(row_failures) != failed
    measured = (not unreadable and not rows_contradict
                and (source is None or _field_present(value, source)))

    # The failure explanation (a failure row's message/status, then the top-level
    # `message`) and `dirty_after` come from the shared builder: gating on
    # `not success`, not on `failed`, is what keeps a revert that failed AFTER
    # every rename verified -- zero failure rows, message only -- from reporting
    # failed=0 with no error while the view sits partially renamed.
    return _build_mutation_summary(
        # This op measures through its own counters rather than through
        # `results[]` -- but "measures through" is not "measured": a counter
        # that arrived in a shape no count reads out of leaves the derived
        # counts exactly as unknown as an empty `results[]` does, and the
        # builder's fail-safe (counts None, dirty_after True) is what stops a
        # fabricated 0 from rendering the decision keys of a genuine all-noop
        # commit. Hardcoding True here made that fail-safe unreachable from the
        # one op whose whole reason for existing is #683.
        measured=measured,
        unusable=bool(unreadable) or rows_contradict,
        # Named precisely, because this phrase is what the compact TEXT prints
        # and what `_unmeasured_cause` reads back out: saying "counters" for an
        # unreadable ROW sends a reader to the wrong field.
        unmeasured_cause=(
            "this op's own counters could not be read"
            if any(key in _GO_RENAME_COUNTERS for key in unreadable)
            else "this op's failure rows could not be read" if unreadable
            else "this op's failure rows contradict its own counters"
            if rows_contradict
            else "this op reported none of its own counters"),
        # NOT disjoint sets: the wire `skipped_user_named` FOLDS apply-time
        # "changed underneath us" skips in (bridge: skipped_total =
        # skipped_user_named + skipped_during_apply) while those same rows stay
        # inside go_renamed_candidates. Distinct functions considered =
        # candidates + scan-time-only skips -- keeping verified+noop+failed <=
        # op_count, the invariant every other mutation summary holds.
        op_count=candidates + skipped
                 - counters["skipped_changed_during_apply"],
        reported_success=bool(value.get("success", True)),
        # `results[]` holds only the FAILURE rows for this op.
        failure_rows=failure_rows,
        failed=failed,
        changed=changed,
        verified=verified,
        # Already-user-named functions are deliberately left alone: no change,
        # which is what `noop` means elsewhere.
        noop=skipped,
        committed=committed,
        preview=preview,
        rolled_back=rolled_back,
        message=_text_value(value, "message"),
        default_error="go rename failed",
    )


@_discloses
def _render_mutation_summary_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    state = ("committed" if value.get("committed")
             else "preview" if value.get("preview")
             else "rolled back" if value.get("rolled_back")
             else "ok" if value.get("success") else "FAILED")
    parts = [
        f"mutation: {state}",
        f"changed={value.get('changed_count', 0)}",
        f"verified={value.get('verified_count', 0)}",
        f"noop={value.get('noop_count', 0)}",
        f"failed={value.get('failed_count', 0)}",
        f"dirty_after={value.get('dirty_after')}",
    ]
    line = "  ".join(parts)
    if value.get("measured") is False:
        # #684 review: `dirty_after` is no longer "UNKNOWN" here -- it is a
        # fail-safe True (see _mutation_summary) precisely so a naive
        # `if not dirty_after` consumer still saves. Only the derived counts
        # are genuinely unknown (None above, not a confirmed zero-change).
        # Advise INSPECTION, not a re-run: this mutation already committed (or
        # attempted to), so running it again is a different execution that
        # cannot recover what the first one did, and duplicates state for a
        # non-idempotent op.
        # It NAMES the cause, and does not assert it: there is more than one
        # way to be unmeasured (no `results[]` rows; an op's own counters
        # arriving in a shape no count reads out of; that op reporting none of
        # them at all), so the one place that knows puts the cause in
        # `first_error` and this reads it back through the shared template.
        # Hardcoding "this op reported no results[] rows" printed the WRONG
        # cause the moment a second one existed, and dropping the clause
        # entirely broke the sample output the public reference quotes
        # verbatim -- for the documented cause this line is byte-identical to
        # it again.
        # Through the choke point. Read raw, an unreadable `first_error` dropped
        # the CAUSE clause silently: the warning came out byte-identical to one
        # carrying no cause at all, on the one line whose whole job is to say
        # why the counts are unknown (#619).
        cause = _unmeasured_cause(_text_value(value, "first_error"))
        line += ("\nwarning: unmeasured -- "
                 + (f"{cause}; " if cause else "")
                 + "the changed/verified/noop/failed counts above are UNKNOWN. "
                 "dirty_after is reported True as a fail-safe, not confirmed. "
                 "Do not assume nothing changed: read the view back (e.g. "
                 "`bn target info` or a targeted readback) and `bn save` "
                 "before closing.")
    if value.get("first_error"):
        line += f"\nfirst_error: {value['first_error']}"
    return line


@_discloses
def _render_mutation_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    preview = bool(value.get("preview"))
    success = bool(value.get("success", True))
    committed = bool(value.get("committed", False))
    results = _row_list(value, "results")
    failed = [r for r in results if _is_failed_status(r)]

    lines: list[str] = []

    if not success or failed:
        if not committed:
            # Only claim a rollback we actually performed. When rolled_back is
            # explicitly False the revert failed and the view may be modified --
            # saying "rolled back" there contradicts the honest message and
            # re-states the very symptom #117 set out to fix.
            if value.get("rolled_back") is False:
                lines.append("rollback failed: the view may be left modified")
            else:
                lines.append("rolled back: live verification failed")
        # A non-string `message` (a dict/list from a malformed or future bridge
        # result) reached `"\n".join` and raised, so one wrong-typed field cost
        # every line of the failure report above it. Dump it instead (#619).
        msg = value.get("message")
        if msg:
            lines.append(msg if isinstance(msg, str) else _render_fallback_text(msg))
        for item in failed:
            lines.append("failed: " + _format_op_summary(item))
            if item.get("requested"):
                lines.append("  requested: " + json.dumps(item["requested"], sort_keys=True))
            if item.get("observed"):
                lines.append("  observed: " + json.dumps(item["observed"], sort_keys=True))
        lines.append("")
    elif preview:
        # The banner already says applied+reverted; the bridge's matching
        # "Preview verified and reverted." message would only repeat it. (A
        # restore-failure preview is success=False and renders above instead.)
        lines.append("preview: change applied + reverted")
        # A non-default message carries real information the banner does not --
        # e.g. the #582 has_user_type residue disclosure on an otherwise clean,
        # exit-0 preview. Surface it so text mode never drops the caveat.
        msg = value.get("message")
        if msg and msg != "Preview verified and reverted.":
            lines.append(msg if isinstance(msg, str) else _render_fallback_text(msg))
        lines.append("")

    has_type_op = any(_is_type_result(r) for r in results)
    has_direct_op = any(r.get("op") and not _is_type_result(r) for r in results)

    if results:
        if len(results) == 1 and success and not failed and not preview:
            lines.append(_format_op_summary(results[0]))
        else:
            lines.append(f"results ({len(results)}):")
            for item in results:
                lines.append("- " + _format_op_summary(item))

    # "What landed" detail so a verified mutation confirms itself without a
    # follow-up read: the live prototype for each set_prototype. Emitted per
    # result (not only single-op batches) so a mixed batch still surfaces it.
    if not failed:
        for item in results:
            if item.get("op") == "set_prototype" and not _is_failed_status(item):
                lines.extend(_set_prototype_detail(item))

    # Type and direct detail are independent: a mixed batch shows BOTH the type's
    # size/field delta + blast radius AND the direct ops' affected-function diffs.
    # The blast radius replaces the per-function dump for type ops (which was
    # either empty or a wall of unrelated callers).
    if has_type_op:
        lines.extend(_types_affected_lines(value))
        blast = _blast_radius_line(value)
        if blast:
            lines.append(blast)
    if has_direct_op:
        affected_functions = [a for a in (_field_list(value, "affected_functions")) if isinstance(a, dict)]
        # In a mixed batch, only the direct-op targets belong here -- the type's
        # reflowed callers are summarised by the blast-radius line above. With no
        # type op, every changed function is a direct-op effect.
        changed_functions = [
            a for a in affected_functions
            if a.get("changed") and (a.get("direct") if has_type_op else True)
        ]
        if changed_functions:
            lines.extend(["", f"affected functions ({len(changed_functions)}):"])
            for item in changed_functions:
                before_name = item.get("before_name") or item.get("after_name") or "<unknown>"
                after_name = item.get("after_name") or before_name
                summary = f"{item.get('address', '<unknown>')} {before_name}"
                if after_name != before_name:
                    summary += f" -> {after_name}"
                lines.append("- " + summary)
                if preview and item.get("diff"):
                    lines.append(str(item["diff"]))

    return "\n".join(lines).rstrip() + "\n"


@_discloses
def _render_py_exec_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    parts: list[str] = []
    # Choke point: a container-shaped `stdout` rendered the EMPTY STRING, which
    # is exactly what a run that printed nothing renders, undisclosed (#619).
    stdout = _text_value(value, "stdout")
    if stdout:
        parts.append(stdout.rstrip("\n"))

    result = value.get("result")
    if result is not None:
        body = result if isinstance(result, str) else json.dumps(result, indent=2, sort_keys=True)
        parts.append("result:\n" + body)

    warnings = _field_list(value, "warnings")
    if warnings:
        parts.append("warnings:\n" + "\n".join(f"- {warning}" for warning in warnings))

    artifact = _field_dict(value, "artifact")
    if artifact.get("artifact_path"):
        parts.append(f"artifact: {artifact['artifact_path']}")

    if not parts:
        return ""
    return "\n\n".join(parts)


@_discloses
def _render_skill_install_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)

    installed = _field_list(value, "installed_destinations")
    skipped = _field_list(value, "skipped_destinations")
    lines = []

    if installed:
        lines.append(f"Installed skills ({value.get('mode', 'unknown')}):")
        lines.extend(f"- {dest}" for dest in installed)
    else:
        lines.append("Skills already installed.")

    if skipped:
        lines.append("Skipped existing destinations:")
        lines.extend(f"- {dest}" for dest in skipped)

    return "\n".join(lines) + "\n"


_TRACE_REASON_LABELS: dict[str, str] = {
    "function_parameter": "function parameter",
    # A terminal with no reaching SSA definition that is NOT a confirmed
    # parameter: an undefined local or a global. Don't claim "function
    # parameter" — that misleads provenance/attacker-source slices.
    "undefined_or_global": "undefined / no reaching definition",
    "function_parameter_or_global": "function parameter, global, or undefined",
    "memory_load": "memory load",
    "field_load": "field load",
    "call_or_jump_boundary": "call boundary",
    "definition": "definition",
    "phi_source": "phi source",
    "cross_function": "crosses into callee",
    # #416/#672: the value was written by a callee through an out-pointer;
    # backward tracing follows return values, not out-parameters, in either
    # mode.
    "interprocedural_out_param_not_followed": "out-param fill not followed",
    "out_param_not_followed": "out-param fill not followed",
}


@_discloses
def _render_trace_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    fn_name = value.get("function", "<unknown>")
    fn_addr = value.get("function_address", "<unknown>")
    target_addr = value.get("target_address", "<unknown>")
    # `_stated_count`, not a bare `.get`: the int sibling of the `_text_value`
    # reads below, and the third component of this one descriptor to need it.
    # A wrong-shaped index was interpolated raw -- `arg[{'a': 1}]`, `arg[[x]]`,
    # `arg[nan]` -- with no disclosure, which is the same defect rated major for
    # `register` one round earlier, two lines away (#858 review round 5). The
    # reader records the skew for the enclosing boundary and `_stated_count`
    # renders `?` for it, so the slot states "unreadable" rather than a value
    # nobody can act on.
    arg_index = _stated_count(value, "arg_index")
    trace = _field_list(value, "trace")
    hints = [h for h in _field_list(value, "hints") if h]

    # arg[N] of <callee> (reg) -- the callee resolved for this callsite plus
    # the calling-convention register (#662): the value being sliced is the
    # CALLER's operand at the callsite, not a named callee parameter, so the
    # header no longer prints a parameter name (which suggested a seed that
    # was never what was traced).
    arg_lbl = _field_dict(value, "arg_label")
    arg_desc = f"arg[{arg_index}]"
    # #755: never leave the callee slot silent when the producer COMPUTED one. An
    # indirect call (a vtable slot, a register target) resolves to no name, and
    # the header then said nothing about which call it answered -- two such calls
    # in one function rendered headers differing only by address, so an analyst
    # who copied a nearby address got an equally confident slice about a
    # different call with no signal to catch it with.
    #
    # What counts as "computed" is an `arg_label` that arrived as an OBJECT.
    # Neither a missing key nor an explicit null does, because in this module a
    # nulled field claims nothing: that is `_field_present`'s stated rule ("the
    # key is there AND is not an explicit null"), it is how `_field_list` and
    # `_field_dict` already read one, and it is the reading the mirror test
    # relies on when it feeds `{key: None}` as a BENIGN payload for every
    # discovered read. `_field_declared` answers a deliberately DIFFERENT
    # question -- which shape of envelope arrived -- so using it here made an
    # explicit `"arg_label": null` render the affirmative "we looked and found no
    # resolvable callee", which is the absent-vs-null conflation one level up
    # (#858 review round 2).
    #
    # A name may still arrive on either key; the top-level `callee` is the name
    # when known and, being nullable, claims nothing by itself.
    computed = isinstance(value.get("arg_label"), dict)
    # `_text_value`, not a bare `.get`: it is this module's string sibling of
    # `_field_list`/`_field_dict`/`_count_field`, so a key PRESENT in a shape no
    # name reads out of records the skew for the enclosing boundary to disclose
    # instead of interpolating a raw Python repr into the header (a dict rendered
    # as `arg[0] of {'name': 'x'}`) or reading a number as "unresolved" in
    # silence (#858 review round 2 minor).
    # Both keys go through the reader, and NOT with `or`: short-circuiting on a
    # truthy first key would leave a skew on the second one unrecorded, which is
    # the alias trap `_field_list`'s docstring names. Read both, then choose
    # (#858 review round 3 minor).
    label_callee = _text_value(arg_lbl, "callee")
    top_callee = _text_value(value, "callee")
    callee_name = label_callee or top_callee
    if callee_name:
        arg_desc += f" of {callee_name}"
    elif computed:
        arg_desc += " of <unresolved callee>"
    # The register is the SIBLING read one line down, and it had the same bare
    # `.get`: a wrong-shaped value was interpolated raw into the header with no
    # disclosure -- `arg[0] ({'reg': 'rdi'})`. Same reader, same rule (#858
    # review round 3 major).
    register = _text_value(arg_lbl, "register")
    if register:
        arg_desc += f" ({register})"
    header = f"backward trace of {arg_desc} in {fn_name} @ {target_addr}"
    step_word = "step" if len(trace) == 1 else "steps"
    info = f"  {fn_name} @ {fn_addr}  •  {len(trace)} {step_word}"
    if value.get("truncated"):
        info += "  •  truncated"

    if not trace:
        body = "\n".join(f"  hint: {h}" for h in hints) if hints \
            else "  constant or immediate — no SSA trace"
        # Defensive: every `assumptions` entry is carried on a step already
        # appended to `trace` (the depth cap, the out-param reasons, and the
        # --ip-depth/recursion-depth markers all require a non-empty trace),
        # so `assumptions` cannot actually be non-empty here today. Kept as
        # pure renderer robustness against a future producer relaxing that
        # invariant, not because this path is currently reachable.
        assumptions = [a for a in _field_list(value, "assumptions") if a]
        if assumptions:
            body += f"\n\ncaveats ({len(assumptions)}):\n" + \
                "\n".join(f"  - {a}" for a in assumptions)
        return f"{header}\n{info}\n\n{body}"

    lines = [header, info, ""]
    current_fn: str | None = None
    for step in trace:
        if not isinstance(step, dict):
            lines.append(f"  {step}")
            continue

        fn_ctx = step.get("function_context")
        cross_fn = step.get("cross_function")
        ssa_var = step.get("ssa_label") or step.get("ssa_var", "")
        addr = step.get("address")
        il_text = step.get("il_text") or ""
        reason = step.get("reason") or ""
        terminates = bool(step.get("terminates"))

        if cross_fn:
            callee = step.get("callee", "")
            lines.append(f"  ── enters {callee} ──")
            current_fn = callee
            continue

        if fn_ctx and fn_ctx != current_fn:
            lines.append(f"  ── in {fn_ctx} ──")
            current_fn = fn_ctx

        if terminates:
            label = _TRACE_REASON_LABELS.get(reason, reason.replace("_", " "))
            line = f"  {ssa_var}  —  {label}"
            # Name the resolved callee at a call boundary so a library-call origin
            # reads as `call boundary (strlen)` not a bare PLT address (#193).
            if reason == "call_or_jump_boundary" and step.get("callee"):
                line += f" ({step['callee']})"
            if (reason in ("interprocedural_out_param_not_followed", "out_param_not_followed")
                    and step.get("out_param_callee")):
                line += f" (via {step['out_param_callee']})"
            if reason == "field_load":
                meta = " ".join(
                    f"{k}={step[k]}" for k in ("base", "offset", "width") if step.get(k) is not None)
                if meta:
                    line += f"  [{meta}]"
            if il_text:
                line += f"  @ {addr}  {il_text}" if addr else f"  {il_text}"
            lines.append(line)
        else:
            if addr:
                lines.append(f"  {addr}  {il_text}")
            else:
                lines.append(f"  {il_text}")

    # Through the choke point: a malformed `frontiers` container rendered a
    # backward trace byte-identically to one whose frontier was genuinely empty,
    # so the terminal steps a reader uses to judge the slice went missing with
    # no note (#619).
    lines.extend(_render_trace_frontiers(_field_list(value, "frontiers")))
    assumptions = [a for a in _field_list(value, "assumptions") if a]
    if assumptions:
        lines.append("")
        lines.append(f"caveats ({len(assumptions)}):")
        for a in assumptions:
            lines.append(f"  - {a}")

    for h in hints:
        lines.append(f"  hint: {h}")
    return "\n".join(lines)


def _render_trace_frontiers(frontiers: Any) -> list[str]:
    """Concise top-level roll-up of where the slice stopped (#552): one line per
    reason group with its count and a couple of examples. A factual summary of
    stopping conditions, not a verdict."""
    if not isinstance(frontiers, list) or not frontiers:
        return []
    total = sum(c for c in (g.get("count", 0) for g in frontiers if isinstance(g, dict))
                if isinstance(c, int) and not isinstance(c, bool))
    out = ["", f"  frontiers ({total} terminal step(s) in {len(frontiers)} group(s)):"]
    for g in frontiers:
        if not isinstance(g, dict):
            continue
        reason = g.get("reason") or "unspecified"
        if not isinstance(reason, str):            # an unhashable reason crashed the lookup
            reason = str(reason)
        label = _TRACE_REASON_LABELS.get(reason, reason.replace("_", " "))
        # Name the resolved callee(s) at a boundary so the roll-up reads
        # `call boundary x3 (strlen, memcpy)` rather than a bare count.
        callees = []
        for ex in (_field_list(g, "examples")):
            if not isinstance(ex, dict):
                continue
            nm = ex.get("callee") or ex.get("out_param_callee")
            if nm and nm not in callees:
                callees.append(str(nm))
        line = f"    {label}  x{g.get('count', 0)}"
        if callees:
            line += f" ({', '.join(callees)})"
        out.append(line)
    return out


def _class_inputs_note(value: Any) -> str:
    """#653.6: on a ZERO-class result, print the empty inputs so the absence is
    ATTRIBUTABLE -- "this target is C" reads identically to "the lens failed to
    cluster" otherwise, and an agent had to prove RTTI absence by hand."""
    # PRESENT through the choke point, not a raw shape test: a malformed
    # `inputs` dropped the whole attribution note, so a zero-class result read
    # byte-identically to one the lens had never been given evidence for --
    # the exact confusion #653.6 added this note to end (#619).
    if not _field_present(value, "inputs"):
        return ""
    inputs = _field_dict(value, "inputs")
    return (
        f"\n  inputs: demangled C++ symbols: {inputs.get('demangled_cxx_methods', 0)}, "
        f"RTTI typeinfo: {inputs.get('rtti_typeinfo_symbols', 0)}, "
        f"vtable symbols: {inputs.get('rtti_vtable_symbols', 0)}"
        " — zero across all three means the target has no C++ type evidence, "
        "not that the lens failed"
    )


@_discloses
def _render_class_list_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    # #820: the class lens answers on a --quick view, so its inventory leads with
    # the analysis-state warning -- both the count-only line and the listing,
    # since either is read as the target's complete class set.
    quick_prefix = _quick_partial_prefix(value, "class list")
    # The recorded read happens BEFORE the count-only branch, never after it.
    # #484 count-only is a bare count envelope (no items), with the non-class
    # artifact (#481) share broken out so the domain-class count is honest --
    # but deciding that on the raw containers' truth returned before the choke
    # point ran, so a FALSY wrong-shaped listing (`{}`, `0`, `""`, `False`)
    # rendered the count line as if the listing had simply been absent, with no
    # disclosure. Reading first leaves the branch condition alone and still
    # makes the skew reach the note (#619).
    rows = _field_list(value, "items", "classes")
    if "count" in value and not value.get("items") and not value.get("classes"):
        n = value.get("count", 0)
        art = value.get("artifact_count") or 0
        tail = f" ({art} non-class RTTI/type artifact{'s' if art != 1 else ''})" if art else ""
        return f"{quick_prefix}classes: {n}{tail}{_class_inputs_note(value)}"
    total = value.get("total", len(rows))
    header = f"classes: {len(rows)} shown of {total}"
    # Surface what was folded out so the count is self-documenting (#205/#309).
    hidden_parts = []
    cv = value.get("construction_vtables_suppressed") or 0
    if cv:
        hidden_parts.append(f"{cv} construction-vtable artifact{'s' if cv != 1 else ''} (--all to show)")
    th = value.get("thunks_suppressed") or 0
    if th:
        hidden_parts.append(f"{th} thunk{'s' if th != 1 else ''}")
    lib = value.get("library_suppressed") or 0
    if value.get("no_stl") and lib:
        hidden_parts.append(f"{lib} library/STL")
    ven = value.get("vendor_suppressed") or 0
    if value.get("no_vendor") and ven:
        hidden_parts.append(f"{ven} vendored")
    # #675.2: declared class types are folded out by the confidence gate the way
    # name-only clusters are, so the default listing says they exist instead of
    # reading as a lens that never saw the class the user declared.
    ds = _count_field(value, "declared_suppressed")
    if ds or _field_skewed("declared_suppressed"):
        # `_stated_count`, not the raw count: an unreadable counter must not print
        # as `0 declared class types`, which reads as "the lens looked and found
        # none". The noun is "declared CLASS type" because that is the population
        # the counter counts and `--all` then lists -- it counted the view's whole
        # type table when it said "declared type" (#907 review).
        stated = _stated_count(value, "declared_suppressed")
        hidden_parts.append(
            f"{stated} declared class type{'s' if ds != 1 else ''} (--all to show)")
    if hidden_parts:
        header += " (hidden: " + ", ".join(hidden_parts) + ")"
    header += _class_inputs_note(value)
    lines = [header]
    for rec in rows:
        if not isinstance(rec, dict):
            lines.append(_render_fallback_text(rec))
            continue
        vt = "vtable" if rec.get("has_vtable") else "no-vtable"
        size = rec.get("size")
        size_s = size.get("value") if isinstance(size, dict) else size
        # `bases` arrives as name-dicts (what class show renders) or bare strings;
        # joining a dict crashed the whole listing (#619).
        bases = ", ".join(str((b.get("name") or "?") if isinstance(b, dict) else b)
                          for b in _field_list(rec, "bases") if b)
        base_s = f"  : {bases}" if bases else ""
        # #481: mark a non-class RTTI/type-signature artifact (rtti confidence but no
        # methods and no vtable) so it doesn't read as a domain class.
        art_s = "  [artifact: non-class RTTI]" if rec.get("artifact") else ""
        lines.append(
            f"  {rec.get('name', '<unknown>')}  "
            f"methods={rec.get('method_count', 0)}  {vt}  "
            f"size={size_s if size_s is not None else '?'}  "
            f"[{rec.get('confidence', '?')}]{base_s}{art_s}"
        )
    return quick_prefix + "\n".join(lines)


def _vtable_slot_label(s: dict[str, Any]) -> str:
    """Label for a vtable slot: the demangled method, `__cxa_pure_virtual`, a
    named external (cross-module) slot, `<null>`, or `<unnamed>` (#441)."""
    if s.get("pure_virtual"):
        return "__cxa_pure_virtual"
    method = _field_dict(s, "method")
    slot_name = method.get("display_name") or method.get("name")
    if slot_name:
        return slot_name
    if s.get("external"):
        ext = s.get("external_name")
        return f"{ext} [external]" if ext else "<external>"
    if s.get("null"):
        return "<null>"
    return "<unnamed>"


@_discloses
def _render_class_show_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    if value.get("ambiguous"):
        # Count the rows actually rendered, not the raw field: a skewed `matches`
        # counted one match per CHARACTER of a string, which is invented data in
        # the one line a reader uses to decide how ambiguous the query was.
        matches = _field_list(value, "matches")
        out = [f"ambiguous class {value.get('query', '')!r}: {len(matches)} matches"]
        for rec in matches:
            out.append("")
            out.append(_render_one_class(rec))
        return "\n".join(out)
    return _render_one_class(value)


def _class_method_label(method: dict[str, Any]) -> str:
    label = f"{method.get('address', '?')}  {method.get('demangled', '')}"
    thunk = _field_dict(method, "thunk")
    target = _field_dict(thunk, "target")
    if thunk.get("is_candidate"):
        destination = f"target -> {_render_target_line(target)}" if target else "target unresolved"
        reason = thunk.get("reason") or "no reason recorded"
        label += f"  [thunk/veneer candidate: {reason}; {destination}]"
    elif target:
        label += f"  [local branch -> {_render_target_line(target)}; not a confirmed thunk/veneer]"
    return label


def _render_one_class(rec: Any) -> str:
    if not isinstance(rec, dict):
        return _render_fallback_text(rec)
    # `size` arrives as a `{value: N}` envelope OR as a bare scalar -- the list
    # view already renders both, and this card dropped the bare form, so a class
    # that DID state its size rendered byte-identically to one that never said
    # (#619). Not a choke-point read: a scalar is a legitimate shape here, not a
    # skew, so every PRESENT value renders rather than disclosing.
    size = rec.get("size")
    if isinstance(size, dict):
        # An envelope that carries no `value` still CLAIMED a size, so it
        # renders `?` rather than vanishing: dropping it made the card
        # byte-identical to a class whose size was never stated (#619).
        size_s = size.get("value")
        if size_s is None:
            size_s = "?"
    else:
        size_s = size
    # Through the choke point, so a malformed `vtable` container discloses
    # itself instead of rendering byte-identically to a class that simply has
    # none -- the cluster #619 names. `{}` is what absent and malformed both
    # degrade to, and every read below already treats it as "no vtable".
    vt = _field_dict(rec, "vtable")
    vt_addr = vt.get("address")
    # No element filter: a declared but EMPTY base entry is a base the payload
    # claimed, so it renders as `?` the way base rendered it. Dropping it turned
    # "has an unnamed base" into "has no such base" in a hierarchy view (#619).
    bases = ", ".join(str((b.get("name") or "?") if isinstance(b, dict) else (b if b else "?"))
                      for b in _field_list(rec, "bases"))
    head = f"class {rec.get('name', '<unknown>')}"
    bits = []
    if size_s is not None:
        bits.append(f"size {size_s}")
    if vt_addr:
        bits.append(f"vtable @ {vt_addr}")
    if bases:
        bits.append(f"base: {bases}")
    if bits:
        head += "  (" + ", ".join(bits) + ")"
    lines = [head, f"  [{rec.get('confidence', '?')}]"]
    methods = _field_list(rec, "methods")
    for m in methods:
        if not isinstance(m, dict):
            lines.append(f"  method {m!r}")
            continue
        if m.get("kind") in ("ctor", "dtor"):
            lines.append(f"  {m.get('kind'):<6} {_class_method_label(m)}")
    # A malformed slot container must fall through to the explanation below, not
    # to a class card that shows a vtable address and then nothing at all -- that
    # rendered strictly LESS than the genuinely-empty case (#619).
    vt_slots = _field_list(vt, "slots")
    if vt_slots:
        for s in vt_slots:
            if not isinstance(s, dict):
                lines.append(f"  vtable {s!r}")
                continue
            lines.append(f"  vtable [{s.get('index')}] {s.get('address', '?')}  {_vtable_slot_label(s)}")
    elif vt_addr:
        # A vtable symbol exists but no slots resolved. Either the vtable is
        # defined in another module (the local symbol is an import/GOT slot) or
        # it is a PIE/.data.rel.ro vtable whose pointers are applied at load time
        # via relocations (zero in the static image). In both cases there is no
        # decodable local body; say so rather than render fake or empty virtuals.
        lines.append("  vtable: symbol present but no slots resolved here "
                     "(defined in another module, or applied at load time via relocations)")
    if vt.get("truncated"):
        # #822: the cap note now carries the bound instead of only the cap, so a
        # reader can see how much the cap hid -- the validated minimum, never a
        # count. (At a fixed cap the bound does NOT recover the real total: an
        # 80-entry and a 4000-entry table both report `max_slots + 1`. An exact
        # total is what `total` is for, and it is present exactly when the
        # table's end was observed.)
        bound = vt.get("total_lower_bound")
        bound_s = f"of at least {bound} entries" if isinstance(bound, int) else "of an unknown total"
        lines.append(
            f"  vtable: showing {len(vt_slots)} slots {bound_s}; scan capped at "
            f"{vt.get('max_slots')} -- more may exist (raise the cap or inspect the "
            "table directly)"
        )
    elif vt.get("scan_truncated"):
        # #822 review (round 1): the total is not exact for a reason the display
        # cap did not cause -- the scan stopped at an entry it could not READ, so
        # the table's end was never observed. Say which, instead of claiming a
        # complete table or blaming a cap nothing hit.
        bound = vt.get("total_lower_bound")
        bound_s = f"of at least {bound} entries" if isinstance(bound, int) else "of an unknown total"
        lines.append(
            f"  vtable: showing {len(vt_slots)} slots {bound_s}; the scan stopped at an "
            "unreadable entry -- more may exist (inspect the table directly)"
        )
    # #412: secondary (multiple-inheritance) vtables -- shown compactly so a simple
    # single-inheritance class isn't cluttered (there are none to show there).
    for sec in _field_list(rec, "secondary_vtables"):
        if not isinstance(sec, dict):
            continue
        ott = sec.get("offset_to_top")
        ott_s = f" (offset-to-top {ott})" if ott is not None else ""
        lines.append(f"  secondary vtable @ {sec.get('address', '?')}{ott_s}:")
        for s in _field_list(sec, "slots"):
            if not isinstance(s, dict):
                lines.append(f"    {s!r}")
                continue
            lines.append(f"    [{s.get('index')}] {s.get('address', '?')}  {_vtable_slot_label(s)}")
        if sec.get("truncated"):
            sec_bound = sec.get("total_lower_bound")
            sec_bound_s = (f"of at least {sec_bound} entries" if isinstance(sec_bound, int)
                           else "of an unknown total")
            lines.append(
                f"    vtable: showing {len(_field_list(sec, 'slots'))} slots {sec_bound_s}; "
                f"scan capped at {sec.get('max_slots')} -- "
                "more may exist (raise the cap or inspect the table directly)"
            )
        elif sec.get("scan_truncated"):
            sec_bound = sec.get("total_lower_bound")
            sec_bound_s = (f"of at least {sec_bound} entries" if isinstance(sec_bound, int)
                           else "of an unknown total")
            lines.append(
                f"    vtable: showing {len(_field_list(sec, 'slots'))} slots {sec_bound_s}; "
                "the scan stopped at an unreadable entry -- "
                "more may exist (inspect the table directly)"
            )
    # Non-virtual member functions (kind=method). Virtual ones already appear as
    # vtable slots above; listing the symbol-side methods makes `class show`
    # useful for classes whose vtable is empty or absent (e.g. Controller).
    member_methods = [m for m in methods if isinstance(m, dict) and m.get("kind") == "method"]
    if member_methods:
        lines.append(f"  methods ({len(member_methods)}):")
        for m in member_methods:
            lines.append(f"    {_class_method_label(m)}")
    inst = _field_dict(rec, "instances")
    parts = []
    for site in _field_list(inst, "construction_sites"):
        if not isinstance(site, dict):
            parts.append(repr(site))
            continue
        sz = f" (size {site['size']})" if site.get("size") else ""
        fn = f" (in {site['function']})" if site.get("function") else ""
        parts.append(f"{site.get('kind', '?')} @ {site.get('address', '?')}{sz}{fn}")
    for g in _field_list(inst, "stored_globals"):
        if not isinstance(g, dict):
            parts.append(f"stored -> {g!r}")
            continue
        parts.append(f"stored -> {g.get('symbol') or '?'} @ {g.get('address', '?')}")
    if parts:
        lines.append("  instances: " + " ; ".join(parts))
    hidden = []
    for field, label in (("construction_sites", "construction sites"),
                         ("stored_globals", "stored globals")):
        if inst.get(f"{field}_truncated"):
            hidden.append(f"{label}: showing {len(_field_list(inst, field))} "
                          f"of {inst.get(f'{field}_total')}")
    if hidden:
        # #822: the 128-per-list cap is disclosed with exact totals, so a capped
        # result is never read as a complete one (the vtable cap's shape).
        lines.append("  instances (capped): " + "; ".join(hidden))
    # #675.2: a declared-type record states WHY its vtable/methods/instances are
    # empty (no RTTI class, no demangled methods, no construction sites for this
    # name in this view), which is the one line that tells absence from silence on
    # a card whose every other evidence block is legitimately empty. Through the
    # choke point, so a malformed `notes` container discloses instead of dropping
    # the line that carries the distinction.
    for note in _field_list(rec, "notes"):
        lines.append(f"  note: {note}")
    return "\n".join(lines)
