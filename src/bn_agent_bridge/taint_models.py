"""Taint model database: load/validate/lookup/overlay + TaintError.

Split out of ``taint_engine`` (pure structural move, #562). Import-free of
``binaryninja``; the engine re-exports these names for back-compat.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

try:  # paths is a symlink into the bridge dir; tolerate import-time absence
    from .paths import taint_models_path
except Exception:  # pragma: no cover - defensive
    taint_models_path = None  # type: ignore[assignment]


_BUILTIN_MODELS = Path(__file__).resolve().parent / "taint_models.json"

SOUNDNESS = (
    "may-analysis (interprocedural, summary-based, depth-bounded); memory is "
    "tracked via SSA store/load correlation where addresses match and coarsely "
    "otherwise; unresolved indirect/external calls are surfaced as assumptions/"
    "leaves; NOT a proof of reachability"
)

# Overflow sink classes whose finding is reclassified to `tainted_index` when the
# taint reaches the sink only through an array index/offset (#163). Covers the
# plain copy/length classes and the fortified (__*_chk / *_s) family, which spans
# both copy-source pointer args and length args -- the per-arg pointer role is
# resolved separately from the model's buffer-propagation rules, not the class.
_OVERFLOW_INDEX_CLASSES = frozenset({"overflow_unbounded", "overflow_len", "fortified_overflow"})

# Scatter-gather receive calls: the received bytes land in msghdr->msg_iov[i].
# iov_base, not the msghdr pointer arg, so seeding the arg taints the header, not
# the payload (#306). Names compared after stripping leading underscores / @plt.
_RECVMSG_FAMILY = frozenset({"recvmsg", "recvmmsg"})

# A param: source that is a pointer to an aggregate at least this large (by byte
# size OR member count) is flagged as a "broad source" -- the whole struct is
# treated as one tainted location, which over-taints into unrelated code (#219).
_BROAD_SOURCE_BYTES = 0x40
_BROAD_SOURCE_MEMBERS = 8


def _model_buffer_source_args(model: dict[str, Any]) -> frozenset[int]:
    """Arg indices the model propagates a buffer FROM (``*arg:N`` in a
    ``propagates`` rule's ``from``). These are the copy-SOURCE pointer args (e.g.
    strcpy / __strcpy_chk arg1), as opposed to length scalars -- used to decide
    whether the index-role broadening applies to a given sink arg (#163)."""
    out: set[int] = set()
    for rule in model.get("propagates") or []:
        frm = rule.get("from")
        # Require the POINTEE form ``*arg:N`` -- the buffer arg N points at. A
        # scalar ``arg:N`` (arg N's value, e.g. GLib g_slist_append's
        # ``from: "arg:1"``) is NOT a buffer source, and treating it as one would
        # enable the pointer index broadening on a scalar/length arg (#163 review).
        if isinstance(frm, str) and frm.startswith("*arg:"):
            idx = _try_arg_index(frm)
            if idx is not None:
                out.add(idx)
    return frozenset(out)


def _model_copy_roles(model: dict[str, Any]) -> tuple[int, int] | None:
    """``(source_arg, destination_arg)`` of a buffer-copy model: its single
    ``*arg:S -> *arg:D`` propagation with ``S != D``, or None when it has no
    such rule or more than one. memcpy is ``(1, 0)``; bcopy, whose arguments
    are reversed (``bcopy(src, dst, n)``), is ``(0, 1)`` (#980 review)."""
    roles = set()
    for rule in model.get("propagates") or []:
        frm, to = rule.get("from"), rule.get("to")
        if not (isinstance(frm, str) and isinstance(to, str)
                and frm.startswith("*arg:") and to.startswith("*arg:")):
            continue
        src, dst = _try_arg_index(frm), _try_arg_index(to)
        if src is None or dst is None:
            continue
        if src != dst:
            roles.add((src, dst))
    return next(iter(roles)) if len(roles) == 1 else None


class TaintError(RuntimeError):
    """User-facing taint configuration/resolution error."""


# --------------------------------------------------------------------------
# model database
# --------------------------------------------------------------------------

def _coerce_model_map(raw: Any, *, source: str) -> dict[str, Any]:
    """Validate a parsed model DB and return its name->model map.

    Accepts either ``{"models": {...}}`` or a bare ``{name: model}`` map; rejects
    any other top-level shape so a malformed file can't silently merge to nothing
    (a model whose value isn't a dict couldn't carry source/sink/propagate data).
    """
    if isinstance(raw, dict) and "models" in raw:
        raw = raw.get("models")
    if not isinstance(raw, dict):
        raise TaintError(
            f"{source} must be a JSON object of name->model (or {{\"models\": {{...}}}}); "
            f"got {type(raw).__name__}"
        )
    validate_model_map(raw, source=source)
    return raw


def validate_model_map(models: dict[str, Any], *, source: str) -> None:
    """Validate every model of a bare ``{name: model}`` map, or raise
    :class:`TaintError` naming the model and field. The one entry point for
    model validation: the DB loader runs it on each layer, and the engine
    runs it on the map it is handed, so no malformed token or index ever
    reaches argument selection."""
    for name, model in models.items():
        # `_comment*` keys are free-text documentation in the DB (their string
        # values never match a symbol, so lookup_model ignores them); every real
        # model name must map to an object that can carry source/sink/propagate.
        if str(name).startswith("_comment"):
            continue
        if not isinstance(model, dict):
            raise TaintError(
                f"{source}: model {name!r} must be a JSON object, got {type(model).__name__}"
            )
        _validate_model_interior(name, model, source)


# The keys the engine indexes on a model -- a model's ROLE vocabulary. An entry
# that declares none of them is not a model: a sink's fields pasted at the root
# (`{"app_recv": {"class": "overflow_len", "len_arg": 1}}`) counted as an
# active overlay model, rendered in no catalog, and never armed the sink the
# author meant (#915). The one roleless shape that IS meaningful is the EMPTY
# object: the builtin DB marks functions that neither source, propagate nor
# sink taint with `{}` so a tainted argument does not trigger the conservative
# "unmodeled external" fallback (`_comment_benign` in taint_models.json). An
# entry whose only keys are `_comment*` annotations is that same marker.
_MODEL_ROLE_KEYS = ("sink", "sources", "propagates", "varargs", "continuation",
                    "writes", "return_bound", "arity_capped", "destinations_unmodeled")
# The roles that move or fill bytes. The other three are ANNOTATIONS of a flow
# model (a return bound, an arity cap, a destination disclosure): an overlay
# entry made of annotations alone is merged INTO the existing model of the same
# name, never a replacement for it -- `{"read": {"return_bound": ..}}` used to
# replace the builtin `read` and drop it from every slice (round 4 P3).
_FLOW_ROLE_KEYS = ("sink", "sources", "propagates", "varargs", "continuation", "writes")
# The fields that ARM a sink. At the root with no `sink` beside them they are
# the #915 flat shape -- the author wrote a sink that can never fire. Prose
# fields (`detail`, `optional`, `gate`) are not flagged: they arm nothing, and
# the docstring's promise that unknown keys beside a role are left alone
# covers an annotation such as `{"propagates": [...], "detail": "wrapper"}`.
_SINK_ARMING_KEYS = ("class", "tainted_args", "len_arg", "buf_arg", "format_arg")


def _effective_keys(model: Any) -> list[str]:
    """The keys of *model* that are not ``_comment*`` annotations."""
    if not isinstance(model, dict):
        return []
    return [k for k in model if not str(k).startswith("_comment")]


def model_is_benign_marker(model: Any) -> bool:
    """True for the documented no-op entry: ``{}`` or ``_comment*`` keys only."""
    return isinstance(model, dict) and not _effective_keys(model)


def model_has_role(model: Any) -> bool:
    """True when *model* carries at least one recognised role key -- the entries
    the overlay disclosure counts as models (a benign marker is disclosed
    separately, #915)."""
    return isinstance(model, dict) and any(k in model for k in _MODEL_ROLE_KEYS)


def _validate_model_interior(name: str, model: dict[str, Any], source: str) -> None:
    """Validate the interior shape of a single model so a structurally-malformed
    USER model (`taint --models`, #317) is a clean, attributable error naming the
    model+field -- not an unhandled AttributeError/TypeError deep in apply_model
    that surfaces as a misleading `internal error:` (#317 review). Only the fields
    the engine indexes are checked; unknown keys beside a recognised role are
    left alone for forward-compat. A NON-EMPTY entry with no recognised role at
    all is rejected (#915): it merges to nothing, yet it was counted as an active
    model and, being a truthy dict, took every `model is not None` branch of the
    walk (skipping the unrecovered-arg frontier a genuinely unmodeled call would
    have raised) -- a no-op that still changed the result. The same holds for a
    role declared EMPTY (`"sources": []`, `"propagates": []`, `"varargs": {}`):
    it counted as one model, rendered no catalog row, and silently replaced the
    builtin entry of the same name with nothing (PR-928 review P2)."""
    def _fail(msg: str) -> None:
        raise TaintError(f"{source}: model {name!r} {msg}")

    if model_is_benign_marker(model):
        return
    misplaced = [k for k in _SINK_ARMING_KEYS if k in model]
    if misplaced and "sink" not in model:
        # The #915 flat shape: the fields that would arm a sink sit at the root,
        # where they do nothing. Name the key and the nesting that would arm it.
        shown = ", ".join(f"`{k}`" for k in misplaced)
        _fail(f"has sink field(s) {shown} at the top level; they must be nested "
              f"under `sink` -- {{\"{name}\": {{\"sink\": {{\"class\": ..., "
              f"\"tainted_args\": [...]}}}}}}")
    if not model_has_role(model):
        _fail(f"declares no recognised model role (keys: "
              f"{', '.join(repr(k) for k in _effective_keys(model))}); a model needs "
              f"at least one of {'/'.join(_MODEL_ROLE_KEYS)}, each nested under the "
              f"symbol name ({{\"{name}\": {{\"sink\": {{...}}}}}}); an empty {{}} marks "
              f"a benign function that neither sources, propagates nor sinks taint")

    if "sink" in model:
        sink = model["sink"]
        if not isinstance(sink, dict):
            _fail(f"`sink` must be an object, got {type(sink).__name__}")
        if not isinstance(sink.get("class"), str):
            _fail("`sink` requires a string `class`")
        # #443: a bounded-WRITE sink (a wrapped recv/read that writes len bytes into
        # buf) is declared with `len_arg` (the attacker-controlled write length that
        # arms the sink) and optionally `buf_arg` (the destination, for the
        # provably-bounded downgrade). #931 round 6: `pointer_args` are the
        # arguments of a mixed buffer+length sink class the sink reads bytes
        # through. #1041 item 4: a FORMAT-GATED sink (scanf's width-less %s)
        # arms on the constant format at `format_arg`; `first_dest` is the
        # argument the first assigning conversion writes. All are argument
        # indices, checked by `_validate_model_arguments`.
        if sink.get("tainted_args") is None and sink.get("len_arg") is None \
                and sink.get("format_arg") is None:
            _fail("`sink` requires `tainted_args`, `len_arg` or `format_arg` "
                  "(nothing arms the sink)")
    for key in ("sources", "propagates"):
        if key in model:
            seq = model[key]
            if not isinstance(seq, list) or not all(isinstance(e, dict) for e in seq):
                _fail(f"`{key}` must be a list of objects")
            if not seq:
                _fail(f"declares `{key}` with no rules -- an empty role arms nothing "
                      f"yet counted as an active model and replaced any builtin "
                      f"`{name}` model with it; give it rules or use {{}} to mark "
                      f"{name} benign")
            for e in seq:
                if not isinstance(e.get("to"), str):
                    _fail(f"`{key}` rule {e!r} needs a string `to` (`ret` / `*arg:N`)")
                if key == "propagates" and not isinstance(e.get("from"), str):
                    _fail(f"`propagates` rule {e!r} needs a string `from` (`arg:N` / `*arg:N`)")
    if "varargs" in model:
        va = model["varargs"]
        if not isinstance(va, dict):
            _fail(f"`varargs` must be an object, got {type(va).__name__}")
        if not isinstance(va.get("to"), str):
            _fail("`varargs` needs a string `to` (the `*arg:N` the varargs run is "
                  "formatted into) -- an empty `varargs` declaration arms nothing")
    if "continuation" in model:
        # A static-state tokenizer (strtok): a call whose `null_arg` is the
        # NULL constant continues the state a previous non-NULL call set.
        co = model["continuation"]
        if not isinstance(co, dict):
            _fail(f"`continuation` must be an object, got {type(co).__name__}")
        if "null_arg" not in co:
            _fail("`continuation.null_arg` must be a non-negative integer argument index")
    if "writes" in model:
        # A literal fill (`bzero`): the callee writes the pointee of `*arg:N`
        # from no argument. The backward local-buffer walk counts it as a
        # covering writer whose bytes come from the call itself.
        ws = model["writes"]
        if not isinstance(ws, list) or not ws:
            _fail("`writes` must be a non-empty list of `*arg:N` tokens (the pointees "
                  "the callee fills from no argument); an empty `writes` arms nothing "
                  f"-- give it tokens or use {{}} to mark {name} benign")
        for t in ws:
            parsed = parse_model_token(t)
            if parsed is None or parsed[0] != "*arg":
                _fail(f"`writes` entry {t!r} must be a `*arg:N` token (a pointee the callee "
                      f"fills, N a non-negative decimal argument index); a scalar `arg:N` "
                      f"is not written through -- fix the token or use {{}} to mark "
                      f"{name} benign")
    # The three annotation roles: a falsy value (`{}`, `0`, `false`) declares
    # nothing, yet passed as a role, counted as a model and replaced the
    # builtin of the same name with nothing (round 3 P3) -- the residue of
    # the empty-role rule above.
    if "return_bound" in model:
        rb = model["return_bound"]
        if not isinstance(rb, dict) or model_arg_index(rb.get("max_from_arg")) is None:
            _fail("`return_bound` must be an object with a non-negative integer "
                  "`max_from_arg` (the argument that bounds the return); an empty "
                  "`return_bound` declares nothing -- give it `max_from_arg` or use {} "
                  f"to mark {name} benign")
    for flag in ("arity_capped", "destinations_unmodeled"):
        if flag in model and model[flag] is not True:
            _fail(f"`{flag}` must be `true` -- `{model[flag]!r}` declares nothing yet "
                  f"would count as an active model; drop the key or use {{}} to mark "
                  f"{name} benign")
    _validate_model_arguments(name, model, source)


def _validate_model_arguments(name: str, model: dict[str, Any], source: str) -> None:
    """Every argument token and index a model states, against ONE grammar.

    The engine selects call arguments by these: a malformed token raised a
    ValueError mid-walk, and a negative index picked an argument from the END
    of the call's recovered list through Python's negative indexing (IC04
    review). Tokens are ``ret``/``*ret`` or an exact ``arg:N``/``*arg:N``/
    ``*iovec:N``, each role accepting only the forms it gives meaning to;
    every index is a non-negative decimal up to ``_MAX_ARG_INDEX``. Only the
    fields present are checked -- whether a role arms anything is the
    loader's question (:func:`_validate_model_interior`)."""
    index = f"a non-negative decimal argument index (0-{_MAX_ARG_INDEX})"

    def _fail(msg: str) -> None:
        raise TaintError(f"{source}: model {name!r} {msg}")

    def _need_index(field: str, value: Any) -> None:
        if model_arg_index(value) is None:
            _fail(f"`{field}` must be {index}, got {value!r}")

    def _need_indices(field: str, value: Any) -> None:
        if not (isinstance(value, list) and all(model_arg_index(i) is not None for i in value)):
            _fail(f"`{field}` must be a list of argument indices, each {index}; got {value!r}")

    def _need_token(field: str, token: Any, kinds: tuple[str, ...]) -> None:
        if token in ("ret", "*ret"):
            ok = token in kinds
        else:
            parsed = parse_model_token(token)
            ok = parsed is not None and parsed[0] in kinds
        if not ok:
            spelled = " / ".join(f"`{k}`" if k in ("ret", "*ret") else f"`{k}:N`" for k in kinds)
            _fail(f"`{field}` is {token!r}; expected {spelled}, with N {index} written "
                  f"without a sign, spaces or leading zeros")

    def _need_obj(field: str, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            _fail(f"`{field}` must be an object, got {type(value).__name__}")
        return value

    def _rules(field: str) -> list[dict[str, Any]]:
        seq = model.get(field)
        if not isinstance(seq, list) or not all(isinstance(e, dict) for e in seq):
            _fail(f"`{field}` must be a list of objects")
        return seq

    if "sink" in model:
        sink = _need_obj("sink", model["sink"])
        for k in ("tainted_args", "pointer_args"):
            if sink.get(k) is not None:
                _need_indices(f"sink.{k}", sink[k])
        for k in ("len_arg", "buf_arg", "format_arg"):
            if sink.get(k) is not None:
                _need_index(f"sink.{k}", sink[k])
        fa = sink.get("format_arg")
        if fa is not None:
            fd = sink.get("first_dest")
            if model_arg_index(fd) is None or fd <= fa:
                _fail("`sink.format_arg` needs an integer `first_dest` after it (the "
                      f"argument the first assigning conversion writes), {index}")
    if "propagates" in model:
        for e in _rules("propagates"):
            _need_token("propagates.from", e.get("from"), ("arg", "*arg"))
            _need_token("propagates.to", e.get("to"), ("arg", "*arg", "ret", "*ret"))
    if "sources" in model:
        for e in _rules("sources"):
            _need_token("sources.to", e.get("to"), ("arg", "*arg", "ret", "*ret", "*iovec"))
    if "varargs" in model:
        va = _need_obj("varargs", model["varargs"])
        _need_token("varargs.to", va.get("to"), ("arg", "*arg", "ret", "*ret"))
        # Required: the engine starts the varargs run here, and a missing or
        # null index crashed the walk (round-2 review).
        if model_arg_index(va.get("first_index")) is None:
            _fail("`varargs.first_index` is required: the first argument the varargs run "
                  f"reads, {index}; got {va.get('first_index', '<missing>')!r}")
    if "continuation" in model:
        _need_index("continuation.null_arg", _need_obj("continuation", model["continuation"]).get("null_arg"))
    for field in ("writes", "reads_dest"):
        if field in model:
            toks = model[field]
            if not isinstance(toks, list):
                _fail(f"`{field}` must be a list of `*arg:N` tokens, got {toks!r}")
            for t in toks:
                _need_token(field, t, ("*arg",))
    if "return_bound" in model:
        _need_index("return_bound.max_from_arg",
                    _need_obj("return_bound", model["return_bound"]).get("max_from_arg"))
    if "string_args" in model:
        _need_indices("string_args", model["string_args"])
    if "append" in model:
        ap = _need_obj("append", model["append"])
        for k in ("dest", "src"):
            if k in ap:
                _need_index(f"append.{k}", ap[k])
    if "ret_offset" in model:
        ro = _need_obj("ret_offset", model["ret_offset"])
        for k in ("count_arg", "string_length_arg", "scan_arg", "stop_arg", "count_arg_limit"):
            if k in ro:
                _need_index(f"ret_offset.{k}", ro[k])


def validate_model_arguments(models: dict[str, Any], *, source: str) -> None:
    """:func:`_validate_model_arguments` over a whole ``{name: model}`` map --
    the guard the engine runs on the map it is handed, so trusted data that
    skipped the loader still never reaches argument selection malformed."""
    for name, model in models.items():
        if str(name).startswith("_comment"):
            continue
        if not isinstance(model, dict):
            raise TaintError(
                f"{source}: model {name!r} must be a JSON object, got {type(model).__name__}")
        _validate_model_arguments(name, model, source)


def _is_annotation_only(model: Any) -> bool:
    """True for an entry carrying annotation roles but no flow role."""
    return model_has_role(model) and not any(k in model for k in _FLOW_ROLE_KEYS)


def _overlay(models: dict[str, Any], layer: dict[str, Any], *, source: str) -> None:
    """Apply one validated overlay *layer* onto *models*: an entry with a flow
    role (or a ``{}`` marker) replaces the model of its name; an
    annotation-only entry is merged into the existing flow model, keeping its
    roles. An annotation with no flow model of that name to annotate is
    refused -- it would arm nothing, yet a non-None model skips the
    unmodeled-call frontier for the name."""
    for name, model in layer.items():
        if _is_annotation_only(model):
            base = models.get(name)
            if not (isinstance(base, dict) and any(k in base for k in _FLOW_ROLE_KEYS)):
                raise TaintError(
                    f"{source}: model {name!r} declares only annotation key(s) "
                    f"({', '.join(k for k in _effective_keys(model))}) and there is no "
                    f"model of {name} with a flow role "
                    f"({'/'.join(_FLOW_ROLE_KEYS)}) for them to annotate; declare its "
                    f"flow role beside them, or use {{}} to mark {name} benign")
            models[name] = {**base, **model}
        else:
            models[name] = model


def load_models(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the merged function-model DB: builtin <- user override <- extra.

    Model-load failures used to be swallowed, silently degrading into missing
    source/sink/propagation models -- false negatives indistinguishable from
    analysis limits (#97). Now a broken builtin DB (a packaging bug) and a broken
    BN_TAINT_MODELS override (a user typo that should be loud, not silent) both
    raise TaintError, which the taint command surfaces as a clean error.
    """
    models: dict[str, Any] = {}
    try:
        raw = json.loads(_BUILTIN_MODELS.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TaintError(
            f"builtin taint model DB at {_BUILTIN_MODELS} could not be loaded: {exc}. "
            "This is a packaging bug -- reinstall the bridge."
        ) from exc
    src = f"builtin taint model DB ({_BUILTIN_MODELS})"
    _overlay(models, _coerce_model_map(raw, source=src), source=src)
    if taint_models_path is not None:
        override_path = taint_models_path()
        if override_path.exists():
            try:
                raw = json.loads(override_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise TaintError(
                    f"BN_TAINT_MODELS override at {override_path} could not be loaded: {exc}. "
                    "Fix or remove the file (it overrides the builtin models)."
                ) from exc
            src = f"BN_TAINT_MODELS override ({override_path})"
            _overlay(models, _coerce_model_map(raw, source=src), source=src)
    if extra:
        # User-supplied models (e.g. `taint --models project.json` for
        # project-internal copy/format wrappers, #317) are validated through the
        # same coercion as the builtin/override DBs, so a malformed user file is
        # a loud error, not a silent merge-to-nothing (#97). User entries win on
        # a name clash -- they're the most specific to the target -- except an
        # annotation-only entry, which annotates the model it names.
        src = "user-provided models (--models / extra)"
        _overlay(models, _coerce_model_map(extra, source=src), source=src)
    return models


def model_overlay_sources(extra: dict[str, Any] | None = None, *,
                          user_models_path: str | None = None,
                          user_models_via: str | None = None) -> list[dict[str, Any]]:
    """#415: the active taint-model overlay sources (most-specific last), so a
    `taint` run discloses WHICH models are in effect.

    load_models() re-reads the builtin DB, the BN_TAINT_MODELS-pointed file, and
    any ``--models`` file on EVERY request, so editing a project-local model file
    (or passing ``--models``) takes effect on the next taint command with no
    bridge restart -- this disclosure makes that visible (and lets a status check
    confirm an overlay landed). ``user_models_path`` (when known) names the
    file so the disclosure points at WHICH file landed, not just a count, and
    ``user_models_via`` (when known) names WHICH knob supplied it -- ``--models``
    or ``$BN_TAINT_MODELS`` -- since #669 lets the client forward either."""
    sources: list[dict[str, Any]] = [{"kind": "builtin", "path": str(_BUILTIN_MODELS)}]
    override_path_str: str | None = None
    if taint_models_path is not None:
        try:
            override = taint_models_path()
        except Exception:
            override = None
        if override is not None and override.exists():
            # taint_models_path() returns the BN_TAINT_MODELS path when that env
            # var is set, else the default ~/.cache/bn/taint_models.json -- both of
            # which load_models honors. Only claim the env var when it's actually
            # set; otherwise label the default-file override honestly (review).
            override_path_str = str(override)
            if os.environ.get("BN_TAINT_MODELS"):
                sources.append({"kind": "env_override", "env": "BN_TAINT_MODELS",
                                "path": override_path_str})
            else:
                sources.append({"kind": "override_default", "path": override_path_str})
    if extra:
        # Count the actual models the way load_models()/_coerce_model_map does:
        # unwrap a ``{"models": {...}}`` envelope and skip ``_comment*`` doc keys,
        # so the disclosed count matches what was really merged (review: a wrapped
        # file with two inner models otherwise reported count 1). #915: `count`
        # is the entries that contributed a ROLE (sink/sources/propagates/...);
        # a benign `{}` marker merges a real no-op and is disclosed under its own
        # `benign` key, so "1 model(s)" can never describe an entry that armed
        # nothing. (A non-empty roleless entry never gets here -- load_models
        # rejects it before the walk.)
        inner = extra.get("models") if isinstance(extra, dict) and "models" in extra else extra
        entries = ([m for k, m in inner.items() if not str(k).startswith("_comment")]
                   if isinstance(inner, dict) else [])
        count = sum(1 for m in entries if model_has_role(m))
        benign = sum(1 for m in entries if model_is_benign_marker(m))
        # #615 review F8: when the bridge was ALSO spawned with BN_TAINT_MODELS
        # pointed at this SAME file, the override above already discloses it --
        # skip the duplicate `user` entry so the same file isn't listed twice
        # under different `kind`s (overstating the active overlay stack). A
        # client-forwarded file that differs from the spawn-time override still
        # discloses both layers (the common, expected case).
        same_file_as_override = (
            user_models_path is not None and override_path_str is not None
            and str(Path(user_models_path).expanduser()) == str(Path(override_path_str).expanduser()))
        if not same_file_as_override:
            user: dict[str, Any] = {"kind": "user", "via": user_models_via or "--models",
                                    "count": count}
            if benign:
                user["benign"] = benign
            if user_models_path:
                user["path"] = str(user_models_path)
            sources.append(user)
        else:
            # Round-3 follow-up (#707): folding into the override entry must
            # not lose the model count the unfolded `user` entry would have
            # carried -- attach it to the just-appended override entry
            # (`env_override` or `override_default`, whichever matched)
            # instead of silently dropping it.
            for entry in reversed(sources):
                if entry.get("kind") in ("env_override", "override_default"):
                    entry["count"] = count
                    if benign:
                        entry["benign"] = benign
                    break
    return sources


def _canonical_cxx_alloc(name: str) -> str | None:
    """Canonical model key for a C++ ``operator new`` / ``operator new[]``
    spelling -- mangled or demangled -- or None when *name* is neither (#204).

    BN renders these allocators either Itanium-mangled (``Znwm``/``Znwj`` for
    ``operator new``, ``Znam``/``Znaj`` for ``operator new[]``; the ``m``/``j``
    is the 64-/32-bit ``size_t`` overload, with optional ``St11align_val_t`` /
    ``RKSt9nothrow_t`` suffixes) or demangled (``operator new(unsigned long)``).
    All of those allocate with the size at arg 0, so they collapse to the two
    keys ``Znwm`` / ``Znam``. Placement new (``_ZnwmPv`` / ``operator
    new(unsigned long, void*)``) constructs in caller-supplied storage -- it
    does NOT allocate -- so it is excluded to avoid a false ``alloc_size`` sink.

    *name* is expected already stripped of a leading ``_`` (as ``lookup_model``
    passes it).
    """
    if not name:
        return None
    if name.startswith("operator new"):
        if "void*" in name or "void *" in name:        # placement new
            return None
        return "Znam" if name.startswith("operator new[]") else "Znwm"
    if name.startswith("Znw") or name.startswith("Zna"):
        if name[3:4] not in ("m", "j"):                # not a size_t overload
            return None
        if name[4:].startswith("Pv"):                  # placement new (..., void*)
            return None
        return "Znam" if name.startswith("Zna") else "Znwm"
    return None


# Genuine glibc LFS64 transitional symbols -> their base name (#603). Built with
# ``-D_FILE_OFFSET_BITS=64`` or ``-D_LARGEFILE64_SOURCE`` (the common default on
# 64-bit Linux), glibc renames the ``off_t``-taking I/O calls to a ``64``-
# suffixed alias that is semantically identical to its base -- only the offset
# width differs -- so it shares the base's taint model. This is an EXPLICIT
# ALLOWLIST of the real transitional pairs, NOT a generic strip-``64`` rule: a
# blanket "drop the 64 and reuse whatever base matches" is unsound -- it
# mis-models user symbols that merely end in "64". `read`/`write`/`close` are
# NOT off_t-taking LFS APIs (there is no glibc `read64`/`write64`/`close64`), so
# an internal `write64` serializer would falsely borrow the `write` file_write
# sink and an internal `close64` running `system()` would falsely borrow the
# benign empty `close` model (suppressing descent -> a confident false
# all-clear). Only names on this list alias, and only when the base is itself a
# real model key (enforced in ``lookup_model``). Restricted to the LFS APIs
# this catalog actually models; add a pair here only when its base model is
# present. Other genuine LFS64 symbols exist (lseek64, mmap64, ftruncate64,
# preadv64/pwritev64) but are omitted until their base is modeled.
_LFS64_ALIASES: dict[str, str] = {
    "pread64": "pread",
    "pwrite64": "pwrite",
    # #957: the path_injection sinks' large-file spellings.
    "open64": "open",
    "openat64": "openat",
    "creat64": "creat",
    "fopen64": "fopen",
    "freopen64": "freopen",
    "stat64": "stat",
    "lstat64": "lstat",
    "fstatat64": "fstatat",
    "xstat64": "xstat",
    "lxstat64": "lxstat",
    "truncate64": "truncate",
}


def _canonical_lfs64(name: str) -> str | None:
    """Base-model key for a genuine glibc LFS64 transitional symbol on the
    explicit allowlist, or None otherwise (#603).

    Tried LAST in ``lookup_model``, so a name that carries its own model keeps
    it; ``lookup_model`` additionally only ADOPTS the result when the base is a
    real model key. *name* is expected already stripped of a leading ``_`` (as
    ``lookup_model`` passes it), so the internal ``__pread64`` spelling resolves
    too.
    """
    return _LFS64_ALIASES.get(name)


def lookup_model(models: dict[str, Any], name: str | None) -> tuple[str | None, dict[str, Any] | None]:
    """Match a (possibly decorated) symbol name against the model DB.

    Tries the raw name, then the part before ``@`` (``memcpy@plt`` ->
    ``memcpy``), then with leading underscores stripped, then the canonical
    C++ allocator key (so mangled ``_Znam`` and demangled ``operator new[]``
    both resolve to the ``Znam`` model -- #204), then the transitional LFS64
    base key (so ``pwrite64`` / ``pread64`` resolve to ``pwrite`` / ``pread`` --
    #603).
    """
    if not name:
        return None, None
    candidates = [name]
    base = name.split("@", 1)[0]
    if base != name:
        candidates.append(base)
    stripped = base.lstrip("_")
    if stripped and stripped != base:
        candidates.append(stripped)
    alias = _canonical_cxx_alloc(stripped or base)
    if alias and alias not in candidates:
        candidates.append(alias)
    lfs64 = _canonical_lfs64(stripped or base)
    if lfs64 and lfs64 not in candidates:
        candidates.append(lfs64)
    for cand in candidates:
        if cand in models:
            return cand, models[cand]
    return None, None


# The one argument-index grammar every model token and index field shares:
# a non-negative decimal with no sign and no leading zero, at most
# `_MAX_ARG_INDEX`. A negative index would select an argument from the END of
# the call's recovered list through Python's negative indexing, and a huge one
# is an overflow form no calling convention passes -- both are refused rather
# than parsed (IC04 review).
_MAX_ARG_INDEX = 255
_ARG_TOKEN_RE = re.compile(r"(\*?)(arg|iovec):(0|[1-9][0-9]{0,2})")


def parse_model_token(token: Any) -> tuple[str, int] | None:
    """``(kind, N)`` for an exact ``arg:N`` / ``*arg:N`` / ``*iovec:N`` token
    (``kind`` is ``"arg"``, ``"*arg"`` or ``"*iovec"``); None for anything
    else, including ``ret``/``*ret`` and every malformed spelling."""
    if not isinstance(token, str):
        return None
    m = _ARG_TOKEN_RE.fullmatch(token)
    if m is None:
        return None
    kind, n = m.group(1) + m.group(2), int(m.group(3))
    if kind == "iovec" or n > _MAX_ARG_INDEX:
        return None
    return kind, n


def _try_arg_index(token: Any) -> int | None:
    """N from an exact ``arg:N`` / ``*arg:N`` token; None if malformed."""
    parsed = parse_model_token(token)
    return parsed[1] if parsed is not None and parsed[0] in ("arg", "*arg") else None


def model_arg_index(value: Any) -> int | None:
    """*value* when it is a valid argument index field, else None."""
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _MAX_ARG_INDEX:
        return value
    return None


# --------------------------------------------------------------------------
# #433: register fallback for under-recovered copy-sink call args. Shared by
# `taint backward` (TaintEngine) and `trace` (read_taint_slice) -- both seed on
# an MLIL SSA var recovered from a calling-convention register when a thunk/
# import with a too-narrow prototype dropped the arg from the MLIL call.
# --------------------------------------------------------------------------

def model_arg_indices(models: dict[str, Any], callee: str) -> set[int]:
    """Argument indices a MODELED sink provably references -- the union of its
    ``propagates`` ``*arg:N`` operands, its ``sink.tainted_args``, and its
    ``varargs.first_index``. Empty for an unmodeled callee. Gates the register
    fallback so it only ever recovers an argument the model proves is actually
    passed (never fabricates one beyond the sink's real arity)."""
    _, model = lookup_model(models, callee)
    if not model:
        return set()
    idxs: set[int] = set()
    for spec in (model.get("propagates") or []):
        if not isinstance(spec, dict):
            continue
        for key in ("from", "to"):
            tok = _try_arg_index(spec.get(key))
            if tok is not None:
                idxs.add(tok)
    sink = model.get("sink") or {}
    for a in (sink.get("tainted_args") or []):
        if model_arg_index(a) is not None:
            idxs.add(a)
    first = model_arg_index((model.get("varargs") or {}).get("first_index"))
    if first is not None:
        idxs.add(first)
    return idxs

