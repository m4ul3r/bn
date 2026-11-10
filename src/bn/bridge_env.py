"""Which ``BN_*`` settings travel with a bridge across `session restart` (#1076).

The bridge records the ``BN_*`` environment it started under in its registry
entry and `session restart` replays it, so a restart from another shell or
agent does not silently drop ``BN_ROOTFS_MAX_WORK`` and friends. Two sides
read one rule here -- the bridge when it records, the CLI when it replays --
so this module is symlinked into ``src/bn_agent_bridge/`` like ``paths.py``.

Not every ``BN_*`` key is a bridge setting. The CLI has its own: routing
(``BN_INSTANCE``, ``BN_TARGET``), request budgets, output spilling, the local
batch-apply limits. The bridge never reads them, so recording them would
only put the shell that happened to start the bridge into every later
restart's ``env:`` line -- and replaying ``BN_INSTANCE`` into a bridge is a
statement about a CLI that does not exist. They are neither recorded nor
replayed. Binary Ninja's own variables (``BN_DISABLE_USER_PLUGINS``,
``BN_USER_DIRECTORY``, ``BN_LICENSE_FILE``, ...) are read by the bridge
PROCESS even though this code never names them, so the rule is a denylist of
what the CLI owns, never an allowlist of what this code knows.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Read only by the CLI process (`src/bn/cli.py`, `transport.py`, `output.py`,
# the command modules); the bridge never consults them.
CLI_ONLY_KEYS = frozenset({
    "BN_INSTANCE",                 # -i default: which bridge a CLI call routes to
    "BN_TARGET",                   # -t default
    "BN_REQUEST_TIMEOUT",          # CLI-side request budget
    "BN_NO_WAIT",                  # refuse-instead-of-queue request flag
    "BN_SPAWN_TIMEOUT",            # how long the CLI waits for a spawn to register
    "BN_SPILL_TOKENS",             # output spillover threshold
    "BN_SPILL_RETENTION_DAYS",     # spill directory pruning
    "BN_BATCH_APPLY_MAX_OPS",      # local batch-apply limits; the bridge's wire
    "BN_BATCH_APPLY_MAX_BYTES",    #   ceiling is a constant it does not read from env
    "BN_PLUGIN_DIR",               # `bn plugin install` / doctor destination
})

# Recorded but never replayed: BN_CACHE_DIR names the registry the SPAWNING
# cli waits on for the child's registration, so a value from the shell that
# started the old bridge would send the new one's record elsewhere and time
# the spawn out. The restart found the instance under the caller's registry,
# so the caller's value is the one that is right. It stays in the record
# because it says where that bridge registered, which `doctor` and a human
# reading the file want to know.
_NEVER_REPLAYED = CLI_ONLY_KEYS | {"BN_CACHE_DIR"}


# A credential is passed through the environment to keep it OFF the filesystem,
# so its value is never written into the registry (review of #1076): the
# licence text itself, and any key whose name says it carries a secret.
# `BN_LICENSE_FILE` is a path, not a credential, and is recorded like any other
# setting. The registry publishes the NAMES of the keys withheld, so a restart
# can say what it could not carry instead of silently dropping it.
_CREDENTIAL_KEYS = frozenset({"BN_LICENSE"})
_CREDENTIAL_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _bridge_setting(key: str) -> bool:
    return key.startswith("BN_") and key not in CLI_ONLY_KEYS


def is_credential_key(key: str) -> bool:
    upper = key.upper()
    return upper in _CREDENTIAL_KEYS or any(marker in upper for marker in _CREDENTIAL_MARKERS)


def recorded_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """The ``BN_*`` settings of *environ* a bridge publishes in its registry:
    every bridge setting except a credential, whose value is never stored."""
    return {
        key: value for key, value in environ.items()
        if _bridge_setting(key) and not is_credential_key(key)
    }


def withheld_keys(environ: Mapping[str, str]) -> list[str]:
    """Names of the credential keys set in *environ* whose values
    `recorded_environment` withheld. Names only."""
    return sorted(
        key for key in environ if _bridge_setting(key) and is_credential_key(key)
    )


def _well_formed_key(key: Any) -> bool:
    return isinstance(key, str) and key.startswith("BN_") and "\0" not in key and "=" not in key


def validated_record(raw: Any) -> dict[str, str] | None:
    """*raw* as a recorded environment, or None when it is not one.

    The registry is a file: anything may be in it. A record is usable only as
    a whole -- every key a ``BN_`` name, every value a string, no NUL (and no
    ``=`` in a name) -- because one bad entry reaching ``Popen(env=...)`` fails
    the spawn AFTER the old bridge is gone. A record that fails is no record,
    exactly like a registry written before the field existed."""
    if not isinstance(raw, dict):
        return None
    for key, value in raw.items():
        if not _well_formed_key(key) or not isinstance(value, str) or "\0" in value:
            return None
    return dict(raw)


def validated_withheld(raw: Any) -> list[str]:
    """The withheld-key names of a registry entry; anything malformed is
    dropped, since a name that cannot be trusted is not worth printing."""
    if not isinstance(raw, list):
        return []
    return sorted({key for key in raw if _well_formed_key(key)})


def restart_environment(
    recorded: Mapping[str, str] | None, caller: Mapping[str, str],
    withheld: Any = (),
) -> tuple[dict[str, str], dict[str, Any]]:
    """The environment a restarted bridge starts with, and what changed.

    Start from the ``BN_*`` set the old bridge recorded, then let every
    ``BN_*`` key the caller's environment holds override it -- explicitly
    empty included, since an export to "" is a decision, not an absence.
    Everything that is not a bridge setting is the caller's, as it always
    was. The report names each key by what happened to it: ``replayed``
    (record only), ``overridden`` (both, caller's value differs), ``added``
    (caller only); a key equal on both sides is not a change.
    ``not_recorded`` names each credential key the old bridge had set (its
    value was never stored, see `withheld_keys`) that the caller does not
    export: the one thing a restart cannot carry. A withheld key the caller
    does export is passed through and not reported -- there is nothing to
    compare it with. ``recorded`` is False for a registry written before the
    bridge published its environment, or whose record is malformed
    (`validated_record`), in which case nothing is replayed and the result is
    exactly the caller's environment.
    """
    env = dict(caller)
    report: dict[str, Any] = {
        "recorded": recorded is not None,
        "replayed": [],
        "overridden": [],
        "added": [],
        "not_recorded": [],
    }
    if recorded is None:
        return env, report
    withheld_names = set(validated_withheld(list(withheld)))
    for key in sorted(recorded):
        if not key.startswith("BN_") or key in _NEVER_REPLAYED:
            continue
        if key not in caller:
            env[key] = recorded[key]
            report["replayed"].append(key)
        elif caller[key] != recorded[key]:
            report["overridden"].append(key)
    report["added"] = sorted(
        key for key in caller
        if key.startswith("BN_") and key not in recorded
        and key not in _NEVER_REPLAYED and key not in withheld_names
    )
    report["not_recorded"] = sorted(
        key for key in withheld_names
        if key not in caller and key not in recorded and key not in _NEVER_REPLAYED
    )
    return env, report
