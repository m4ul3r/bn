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


def _validate_model_interior(name: str, model: dict[str, Any], source: str) -> None:
    """Validate the interior shape of a single model so a structurally-malformed
    USER model (`taint --models`, #317) is a clean, attributable error naming the
    model+field -- not an unhandled AttributeError/TypeError deep in apply_model
    that surfaces as a misleading `internal error:` (#317 review). Only the fields
    the engine indexes are checked; unknown keys are left alone for forward-compat."""
    def _fail(msg: str) -> None:
        raise TaintError(f"{source}: model {name!r} {msg}")

    def _is_int(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

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
        if ta is None and la is None:
            _fail("`sink` requires `tainted_args` or `len_arg` (nothing arms the sink)")
    for key in ("sources", "propagates"):
        if key in model:
            seq = model[key]
            if not isinstance(seq, list) or not all(isinstance(e, dict) for e in seq):
                _fail(f"`{key}` must be a list of objects")
    if "varargs" in model:
        va = model["varargs"]
        if not isinstance(va, dict):
            _fail(f"`varargs` must be an object, got {type(va).__name__}")
        fi = va.get("first_index")
        if fi is not None and not _is_int(fi):
            _fail("`varargs.first_index` must be an integer")


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
    models.update(_coerce_model_map(raw, source=f"builtin taint model DB ({_BUILTIN_MODELS})"))
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
            models.update(_coerce_model_map(raw, source=f"BN_TAINT_MODELS override ({override_path})"))
    if extra:
        # User-supplied models (e.g. `taint --models project.json` for
        # project-internal copy/format wrappers, #317) are validated through the
        # same coercion as the builtin/override DBs, so a malformed user file is
        # a loud error, not a silent merge-to-nothing (#97). User entries win on
        # a name clash -- they're the most specific to the target.
        models.update(_coerce_model_map(extra, source="user-provided models (--models / extra)"))
    return models


def model_overlay_sources(extra: dict[str, Any] | None = None, *,
                          user_models_path: str | None = None) -> list[dict[str, Any]]:
    """#415: the active taint-model overlay sources (most-specific last), so a
    `taint` run discloses WHICH models are in effect.

    load_models() re-reads the builtin DB, the BN_TAINT_MODELS-pointed file, and
    any ``--models`` file on EVERY request, so editing a project-local model file
    (or passing ``--models``) takes effect on the next taint command with no
    bridge restart -- this disclosure makes that visible (and lets a status check
    confirm an overlay landed). ``user_models_path`` (when known) names the
    ``--models`` file so the disclosure points at WHICH file landed, not just a
    count."""
    sources: list[dict[str, Any]] = [{"kind": "builtin", "path": str(_BUILTIN_MODELS)}]
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
            if os.environ.get("BN_TAINT_MODELS"):
                sources.append({"kind": "env_override", "env": "BN_TAINT_MODELS",
                                "path": str(override)})
            else:
                sources.append({"kind": "override_default", "path": str(override)})
    if extra:
        # Count the actual models the way load_models()/_coerce_model_map does:
        # unwrap a ``{"models": {...}}`` envelope and skip ``_comment*`` doc keys,
        # so the disclosed count matches what was really merged (review: a wrapped
        # file with two inner models otherwise reported count 1).
        inner = extra.get("models") if isinstance(extra, dict) and "models" in extra else extra
        count = sum(1 for k in inner if not str(k).startswith("_comment")) if isinstance(inner, dict) else 0
        user: dict[str, Any] = {"kind": "user", "via": "--models", "count": count}
        if user_models_path:
            user["path"] = str(user_models_path)
        sources.append(user)
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


def _canonical_lfs64(name: str) -> str | None:
    """Canonical base-model key for a transitional LFS64 symbol -- the base name
    plus a ``64`` suffix -- or None when *name* is not such a variant (#603).

    Built with ``-D_FILE_OFFSET_BITS=64`` or ``-D_LARGEFILE64_SOURCE`` (the
    common default on 64-bit Linux), glibc renames every ``off_t``-taking I/O
    call to a ``64``-suffixed alias: ``pread`` -> ``pread64``, ``pwrite`` ->
    ``pwrite64``, ``lseek`` -> ``lseek64``, ``open`` -> ``open64``, and so on.
    The 64-bit-offset variant is semantically identical to its base -- only the
    offset width differs -- so it shares the base's taint model. Modeled as a
    lookup canonicalization (like ``_canonical_cxx_alloc``) rather than
    duplicating every base model under a ``<name>64`` key. lookup_model only
    ADOPTS the result when the de-suffixed base is itself a real model key, so a
    coincidental ``…64`` symbol whose base is unmodeled never spuriously aliases;
    and it is tried LAST, so a name that carries its own ``…64`` model (e.g. the
    endian intrinsic ``bswap_64``) keeps that model.

    *name* is expected already stripped of a leading ``_`` (as ``lookup_model``
    passes it), so the internal ``__pread64`` spelling resolves too.
    """
    if len(name) > 2 and name.endswith("64") and not name[-3].isdigit():
        return name[:-2]
    return None


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

