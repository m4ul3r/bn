"""Declarative op registry for the bridge dispatch.

Mirrors the CLI's @command/_COMMANDS registry (src/bn/cli.py). Each op is
declared once via @op; the read/write lock sets and the dispatch routing are
both DERIVED from this single source, replacing the triple-maintained list
(READ_LOCKED_OPS / WRITE_LOCKED_OPS / the _dispatch_on_main if-chain).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

Binder = Callable[[Any, dict[str, Any], "str | None"], Any]
Escalation = Callable[[dict[str, Any]], bool]

_LOCK_CLASSES = ("read", "write", "none")


@dataclass(frozen=True)
class OpSpec:
    name: str
    lock: str
    binder: Binder
    lock_escalation: Escalation | None = None


class OpRegistry:
    def __init__(self) -> None:
        self._ops: dict[str, OpSpec] = {}

    def op(self, name: str, *, lock: str, escalation: Escalation | None = None) -> Callable[[Binder], Binder]:
        if lock not in _LOCK_CLASSES:
            raise ValueError(f"invalid lock class {lock!r} for op {name!r}; expected one of {_LOCK_CLASSES}")

        def decorator(binder: Binder) -> Binder:
            if name in self._ops:
                raise ValueError(f"duplicate op registration: {name!r}")
            self._ops[name] = OpSpec(name=name, lock=lock, binder=binder, lock_escalation=escalation)
            return binder

        return decorator

    def clear(self) -> None:
        """Drop all registered ops.

        bridge.py is re-executed from scratch by the test harness's
        ``_load_bridge`` (``spec.loader.exec_module``) on every call, which
        re-runs the @op binder block against this same module-global registry.
        Clearing first makes that re-execution idempotent (identical final
        state) while keeping the duplicate-registration guard intact for
        genuine double-registration within one module execution.
        """
        self._ops.clear()

    def spec(self, name: str) -> OpSpec | None:
        return self._ops.get(name)

    def names(self) -> set[str]:
        return set(self._ops)

    def read_locked_ops(self) -> set[str]:
        return {n for n, s in self._ops.items() if s.lock == "read"}

    def write_locked_ops(self) -> set[str]:
        return {n for n, s in self._ops.items() if s.lock == "write"}


# The bridge's single global registry. bridge.py registers all ops against it.
REGISTRY = OpRegistry()
op = REGISTRY.op
