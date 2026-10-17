"""Taint model database: load/validate/lookup/overlay + TaintError.

Split out of ``taint_engine`` (pure structural move, #562). Import-free of
``binaryninja``; the engine re-exports these names for back-compat.
"""
from __future__ import annotations

import json
import os
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
            try:
                out.add(int(frm[len("*arg:"):]))
            except ValueError:
                pass
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
        try:
            src, dst = int(frm[len("*arg:"):]), int(to[len("*arg:"):])
        except ValueError:
            continue
        if src != dst:
            roles.add((src, dst))
    return next(iter(roles)) if len(roles) == 1 else None


class TaintError(RuntimeError):
    """User-facing taint configuration/resolution error."""


class BoundedSink(Exception):
    """A backward sink whose argument is a compile-time constant (e.g. a fixed
    copy length): provably bounded, with no def-chain to slice. This is a
    SUCCESSFUL conclusion, not a seed failure -- it must NOT count toward the
    all-sinks-failed hard error, so a bounded sink returns a clean result
    (exit 0, --out written) instead of looking like a crash (#310)."""


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
    for name, model in raw.items():
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
    return raw


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

    def _is_int(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

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
        ta = sink.get("tainted_args")
        if ta is not None and not (isinstance(ta, list) and all(_is_int(i) for i in ta)):
            _fail("`sink.tainted_args` must be a list of integers")
        # #443: a bounded-WRITE sink (a wrapped recv/read that writes len bytes into
        # buf) is declared with `len_arg` (the attacker-controlled write length that
        # arms the sink) and optionally `buf_arg` (the destination, for the
        # provably-bounded downgrade). Both are argument indices.
        la = sink.get("len_arg")
        if la is not None and (not _is_int(la) or la < 0):
            _fail("`sink.len_arg` must be a non-negative integer argument index")
        ba = sink.get("buf_arg")
        if ba is not None and (not _is_int(ba) or ba < 0):
            _fail("`sink.buf_arg` must be a non-negative integer argument index")
        # #931 round 6: the arguments of a mixed buffer+length sink class the
        # sink reads bytes through (`write`'s buffer, not its length).
        pa = sink.get("pointer_args")
        if pa is not None and not (isinstance(pa, list) and all(_is_int(i) and i >= 0
                                                                for i in pa)):
            _fail("`sink.pointer_args` must be a list of non-negative integer argument indices")
        # #1041 item 4: a FORMAT-GATED sink (scanf's width-less %s) arms on the
        # constant format at `format_arg`; `first_dest` is the argument the first
        # assigning conversion writes.
        fa = sink.get("format_arg")
        if fa is not None:
            if not _is_int(fa) or fa < 0:
                _fail("`sink.format_arg` must be a non-negative integer argument index")
            fd = sink.get("first_dest")
            if not _is_int(fd) or fd <= fa:
                _fail("`sink.format_arg` needs an integer `first_dest` after it (the "
                      "argument the first assigning conversion writes)")
        if ta is None and la is None and fa is None:
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
        fi = va.get("first_index")
        if fi is not None and not _is_int(fi):
            _fail("`varargs.first_index` must be an integer")
    if "continuation" in model:
        # A static-state tokenizer (strtok): a call whose `null_arg` is the
        # NULL constant continues the state a previous non-NULL call set.
        co = model["continuation"]
        if not isinstance(co, dict):
            _fail(f"`continuation` must be an object, got {type(co).__name__}")
        na = co.get("null_arg")
        if not _is_int(na) or na < 0:
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
            if not (isinstance(t, str) and t.startswith("*arg:") and _try_arg_index(t) is not None):
                _fail(f"`writes` entry {t!r} must be a `*arg:N` token (a pointee the callee "
                      f"fills); a scalar `arg:N` is not written through -- fix the token or "
                      f"use {{}} to mark {name} benign")
    # The three annotation roles: a falsy value (`{}`, `0`, `false`) declares
    # nothing, yet passed as a role, counted as a model and replaced the
    # builtin of the same name with nothing (round 3 P3) -- the residue of
    # the empty-role rule above.
    if "return_bound" in model:
        rb = model["return_bound"]
        if not isinstance(rb, dict) or not _is_int(rb.get("max_from_arg")) or rb["max_from_arg"] < 0:
            _fail("`return_bound` must be an object with a non-negative integer "
                  "`max_from_arg` (the argument that bounds the return); an empty "
                  "`return_bound` declares nothing -- give it `max_from_arg` or use {} "
                  f"to mark {name} benign")
    for flag in ("arity_capped", "destinations_unmodeled"):
        if flag in model and model[flag] is not True:
            _fail(f"`{flag}` must be `true` -- `{model[flag]!r}` declares nothing yet "
                  f"would count as an active model; drop the key or use {{}} to mark "
                  f"{name} benign")


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


def _try_arg_index(token: str) -> int | None:
    """Parse N from a model source token ``arg:N`` / ``*arg:N``; None if malformed."""
    try:
        return int(str(token).split("arg:", 1)[1])
    except (IndexError, ValueError):
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
        try:
            idxs.add(int(a))
        except (TypeError, ValueError):
            pass
    first = (model.get("varargs") or {}).get("first_index")
    if isinstance(first, int):
        idxs.add(first)
    return idxs

