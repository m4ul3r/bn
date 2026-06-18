# C++ Object-Model Lens (`bn class`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `bn class <Name>` and `bn class list` — a C++ object-model lens that correlates demangled symbols, vtables, RTTI, and `operator new` sizes into a class-aware view, per issue #205.

**Architecture:** A new read-locked bridge module `read_class.py` builds a *class registry* from one scan over `bv.functions` + `bv.get_symbols()`, reusing existing evidence primitives (`_pointer_table_for_view`, `_normalize_code_pointer`, `il_format._display_name`). Two new ops (`class_list`, `class_show`) wired through facade shims; CLI commands in `cpp_class.py`; text renderers in `formatters.py`. No mutation, no new heavy analysis.

**Tech Stack:** Python 3.14, the existing bridge/CLI split, `pytest` with the `binaryninja` mock in `tests/test_bridge.py`.

**Spec:** `docs/superpowers/specs/2026-06-18-cpp-class-lens-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `plugin/bn_agent_bridge/read_class.py` | Class registry, name split, vtable layout, size, RTTI bases, instances. Free functions taking `ctx`. | Create |
| `plugin/bn_agent_bridge/bridge.py` | Facade shims `_class_list`/`_class_show`; `@op` binders. | Modify |
| `src/bn/commands/cpp_class.py` | `@command("class")` + `@command("class","list")` handlers. | Create |
| `src/bn/commands/__init__.py` | Import `cpp_class` so its decorators register. | Modify |
| `src/bn/formatters.py` | `_render_class_list_text`, `_render_class_show_text`. | Modify |
| `tests/test_read_class.py` | Unit tests for split helper, registry, vtable, size, RTTI, instances (mocked BN). | Create |
| `tests/test_cli.py` | `bn class` / `bn class list` argparse + renderer wiring. | Modify |

**Conventions to follow (verified in-repo):**
- Bridge free functions take `ctx` (the `BridgeContext` seam) as their first arg; never import `bridge`/`mutation_engine` from a `read_*` module.
- `ctx` provides: `_resolve_view(selector)`, `_find_function(bv, ident)`, `_pointer_size(bv)`, `_read_pointer_value(bv, addr, size=...)`, `_normalize_code_pointer(bv, value)`, `_address_context(bv, addr)`.
- Imports for shared helpers: `from ._shared import OperationFailure, _parse_address`.
- `@op("name", lock="read")` binders live in `bridge.py`; the bound method delegates to a facade shim that calls the `read_class` free function with `self.ctx`.
- Reads default to `--format text`; the CLI handler calls `_call(args, "<op>", params, require_target=True, allow_implicit_target=True, text_renderer=..., paged_spill=True, stem=...)`.

---

## Task 1: Depth-aware qualified-name split

**Files:**
- Create: `plugin/bn_agent_bridge/read_class.py`
- Test: `tests/test_read_class.py`

This is the highest-risk unit. It must split `Ns::Outer::method(args)` into `("Ns::Outer", "method(args)")` while ignoring `::` inside `<...>` (templates) and `(...)` (signatures), and handling ctor/dtor/operator forms.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_read_class.py
from __future__ import annotations

import importlib

read_class = importlib.import_module("bn_agent_bridge.read_class")
split = read_class._split_qualified_method


def test_split_plain_method():
    assert split("net::Session::onData(int)") == ("net::Session", "onData(int)")


def test_split_nested_class():
    assert split("a::b::Outer::Inner::run()") == ("a::b::Outer::Inner", "run()")


def test_split_namespaced_free_function():
    # Name-indistinguishable from a method; still clusters under the namespace.
    assert split("net::make_session(int)") == ("net", "make_session(int)")


def test_split_template_class_args_with_scope():
    assert split("std::map<int, std::string>::insert(int)") == (
        "std::map<int, std::string>",
        "insert(int)",
    )


def test_split_template_nested_angle():
    assert split("Vec<Pair<A, B>>::push(A)") == ("Vec<Pair<A, B>>", "push(A)")


def test_split_ctor_and_dtor():
    assert split("net::Session::Session(int)") == ("net::Session", "Session(int)")
    assert split("net::Session::~Session()") == ("net::Session", "~Session()")


def test_split_operator_call():
    assert split("net::Buf::operator()(int)") == ("net::Buf", "operator()(int)")


def test_split_operator_new():
    assert split("net::Pool::operator new(unsigned long)") == (
        "net::Pool",
        "operator new(unsigned long)",
    )


def test_split_no_scope_returns_none():
    assert split("memcpy") == (None, "memcpy")
    assert split("main") == (None, "main")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_read_class.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'bn_agent_bridge.read_class'`.

- [ ] **Step 3: Implement the split helper**

```python
# plugin/bn_agent_bridge/read_class.py
"""C++ object-model lens (#205): class registry, vtable layout, RTTI bases,
object size, and instance tracking, correlated from data Binary Ninja already
recovers (demangled symbols, RTTI data symbols, operator-new sizes).

All functions are read-only and take the BridgeContext seam (``ctx``); this
module never imports ``bridge`` or ``mutation_engine``."""
from __future__ import annotations

from typing import Any

from . import il_format
from ._shared import OperationFailure, _parse_address


def _strip_signature(name: str) -> str:
    """Return *name* with its trailing parameter-list ``(...)`` removed.

    Depth-aware over angle brackets so a '(' inside template args is ignored.
    The parameter list is the LAST top-level balanced paren group."""
    angle = 0
    paren = 0
    open_idx = None
    last_param_open = None
    for i, ch in enumerate(name):
        if ch == "<":
            angle += 1
        elif ch == ">" and angle:
            angle -= 1
        elif ch == "(" and angle == 0:
            if paren == 0:
                open_idx = i
            paren += 1
        elif ch == ")" and angle == 0 and paren:
            paren -= 1
            if paren == 0 and open_idx is not None:
                last_param_open = open_idx
    if last_param_open is not None:
        return name[:last_param_open]
    return name


def _toplevel_operator_index(head: str) -> int | None:
    """Index of a top-level ``operator`` keyword in *head*, else None. The
    method name begins here (it may itself contain '::', e.g. a conversion to a
    qualified type), so the class is whatever precedes the '::' before it."""
    angle = 0
    i = 0
    n = len(head)
    while i < n:
        ch = head[i]
        if ch == "<":
            angle += 1
        elif ch == ">" and angle:
            angle -= 1
        elif (
            angle == 0
            and head.startswith("operator", i)
            and (i == 0 or head[i - 1] in ":< ,(")
        ):
            return i
        i += 1
    return None


def _last_toplevel_scope(head: str) -> int | None:
    """Index of the last top-level ``::`` in *head* (angle-depth 0), else None."""
    angle = 0
    last = None
    i = 0
    n = len(head)
    while i < n:
        ch = head[i]
        if ch == "<":
            angle += 1
        elif ch == ">" and angle:
            angle -= 1
        elif ch == ":" and angle == 0 and i + 1 < n and head[i + 1] == ":":
            last = i
            i += 2
            continue
        i += 1
    return last


def _split_qualified_method(demangled: str) -> tuple[str | None, str]:
    """Split a demangled C++ name into ``(class_name, method)``.

    ``class_name`` is None for names with no scope qualifier. '::' inside
    ``<...>`` or ``(...)`` never splits. Handles ctor/dtor/operator forms."""
    name = (demangled or "").strip()
    if not name:
        return None, name
    head = _strip_signature(name)
    op_idx = _toplevel_operator_index(head)
    if op_idx is not None:
        cls = head[:op_idx].rstrip(": ")
        return (cls or None), name[op_idx:].strip()
    split = _last_toplevel_scope(head)
    if split is None:
        return None, name
    return head[:split], name[split + 2:].strip()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_read_class.py -x -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add plugin/bn_agent_bridge/read_class.py tests/test_read_class.py
git commit -m "feat(class): depth-aware C++ qualified-name split (#205)"
```

---

## Task 2: Class registry + confidence

**Files:**
- Modify: `plugin/bn_agent_bridge/read_class.py`
- Test: `tests/test_read_class.py`

Build the registry: cluster functions by class, attach vtable/typeinfo symbols, assign confidence (`rtti` / `ctor` / `name-only`).

- [ ] **Step 1: Write the failing tests**

Add a small fake-BN harness (functions carry a `.symbol.short_name`; data symbols carry `raw_name`/`short_name`/`address`).

```python
# append to tests/test_read_class.py
class _Sym:
    def __init__(self, raw_name, short_name, address):
        self.raw_name = raw_name
        self.short_name = short_name
        self.name = raw_name
        self.address = address


class _Fn:
    def __init__(self, start, mangled, demangled):
        self.start = start
        self.name = mangled
        self.raw_name = mangled
        self.symbol = _Sym(mangled, demangled, start)


class _RegistryBV:
    def __init__(self, functions, symbols):
        self.functions = list(functions)
        self._symbols = list(symbols)

    def get_symbols(self):
        return list(self._symbols)


def _make_registry_bv():
    fns = [
        _Fn(0x1000, "_ZN3net7SessionC1Eh", "net::Session::Session(unsigned char)"),
        _Fn(0x1100, "_ZN3net7SessionD1Ev", "net::Session::~Session()"),
        _Fn(0x1200, "_ZN3net7Session6onDataEi", "net::Session::onData(int)"),
        _Fn(0x1300, "_ZN3net4makeEv", "net::make()"),          # free fn -> name-only
        _Fn(0x1400, "_ZN3net4Pool5allocEv", "net::Pool::alloc()"),  # ctor-less; ti present
    ]
    syms = [
        _Sym("_ZTVN3net7SessionE", "vtable for net::Session", 0x9000),
        _Sym("_ZTIN3net7SessionE", "typeinfo for net::Session", 0x9100),
        _Sym("_ZTIN3net4PoolE", "typeinfo for net::Pool", 0x9200),
    ]
    return _RegistryBV(fns, syms)


def test_registry_clusters_methods():
    bv = _make_registry_bv()
    reg = read_class._build_class_registry(None, bv)
    assert set(reg) >= {"net::Session", "net::Pool", "net"}
    sess = reg["net::Session"]
    kinds = {m["demangled"]: m["kind"] for m in sess["methods"]}
    assert kinds["net::Session::Session(unsigned char)"] == "ctor"
    assert kinds["net::Session::~Session()"] == "dtor"
    assert kinds["net::Session::onData(int)"] == "method"


def test_registry_confidence_levels():
    reg = read_class._build_class_registry(None, _make_registry_bv())
    assert reg["net::Session"]["confidence"] == "rtti"   # has vtable+typeinfo
    assert reg["net::Pool"]["confidence"] == "rtti"       # has typeinfo only
    assert reg["net"]["confidence"] == "name-only"        # namespace-like


def test_registry_attaches_vtable_and_typeinfo():
    reg = read_class._build_class_registry(None, _make_registry_bv())
    assert reg["net::Session"]["vtable"]["address"] == "0x9000"
    assert reg["net::Session"]["typeinfo"]["address"] == "0x9100"
    assert reg["net::Pool"]["vtable"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_read_class.py -x -q -k registry`
Expected: FAIL — `_build_class_registry` not defined.

- [ ] **Step 3: Implement the registry**

```python
# append to plugin/bn_agent_bridge/read_class.py

# Itanium RTTI data-symbol tags. BN renders the demangled short_name with a
# leading "vtable for " / "typeinfo for " / "typeinfo name for " marker (older
# BN uses an underscored "_vtable_for_" form); strip either to get the class.
_RTTI_TAGS = (
    ("_ZTV", "vtable", ("vtable for ", "_vtable_for_")),
    ("_ZTI", "typeinfo", ("typeinfo for ", "_typeinfo_for_")),
    ("_ZTS", "typeinfo_name", ("typeinfo name for ", "_typeinfo_name_for_")),
)


def _class_of_rtti_symbol(sym, prefixes) -> str | None:
    """Class name for an RTTI symbol: strip the demangled marker, else None."""
    short = str(getattr(sym, "short_name", "") or "")
    for marker in prefixes:
        if short.startswith(marker):
            return short[len(marker):].strip()
    return None


def _rtti_symbol_maps(bv) -> dict[str, dict[str, Any]]:
    """{class_name: {"vtable": sym, "typeinfo": sym, "typeinfo_name": sym}}."""
    maps: dict[str, dict[str, Any]] = {}
    for sym in bv.get_symbols():
        raw = str(getattr(sym, "raw_name", "") or getattr(sym, "name", "") or "")
        for prefix, key, markers in _RTTI_TAGS:
            if not raw.startswith(prefix):
                continue
            cls = _class_of_rtti_symbol(sym, markers)
            if cls:
                maps.setdefault(cls, {}).setdefault(key, sym)
            break
    return maps


def _last_component(class_name: str) -> str:
    """Final ``::`` component (ignoring template args) — the ctor/dtor name."""
    head = _strip_signature(class_name)
    idx = _last_toplevel_scope(head)
    comp = head[idx + 2:] if idx is not None else head
    # Drop any template suffix on the component itself.
    angle = comp.find("<")
    return comp[:angle] if angle != -1 else comp


def _method_kind(class_name: str, method: str) -> str:
    """ctor / dtor / method, from the demangled method spelling."""
    last = _last_component(class_name)
    name = method.split("(", 1)[0].strip()
    if name == f"~{last}":
        return "dtor"
    if name == last:
        return "ctor"
    return "method"


def _sym_entry(sym) -> dict[str, Any] | None:
    if sym is None:
        return None
    return {
        "address": hex(int(getattr(sym, "address", 0))),
        "symbol": str(getattr(sym, "raw_name", "") or getattr(sym, "name", "")),
    }


def _build_class_registry(ctx, bv, *, query: str | None = None) -> dict[str, dict[str, Any]]:
    """One scan -> {class_name: ClassRecord}. Methods, RTTI symbols, confidence.
    Per-class drill-downs (vtable layout, size, bases, instances) are added by
    ``_class_show`` only for the requested class (too costly for every class)."""
    rtti = _rtti_symbol_maps(bv)
    registry: dict[str, dict[str, Any]] = {}
    needle = query.lower() if query else None

    for fn in bv.functions:
        demangled = il_format._display_name(fn)
        cls, method = _split_qualified_method(demangled)
        if cls is None:
            continue
        rec = registry.get(cls)
        if rec is None:
            rec = registry[cls] = {
                "name": cls,
                "methods": [],
                "vtable": None,
                "typeinfo": None,
                "typeinfo_name": None,
                "size": None,
                "bases": [],
                "instances": [],
                "confidence": "name-only",
            }
        rec["methods"].append({
            "address": hex(int(getattr(fn, "start", 0))),
            "mangled": str(getattr(fn, "name", "")),
            "demangled": demangled,
            "kind": _method_kind(cls, method),
        })

    # Ensure RTTI-only classes (no demangled methods clustered) still appear.
    for cls in rtti:
        registry.setdefault(cls, {
            "name": cls, "methods": [], "vtable": None, "typeinfo": None,
            "typeinfo_name": None, "size": None, "bases": [], "instances": [],
            "confidence": "name-only",
        })

    for cls, rec in registry.items():
        syms = rtti.get(cls, {})
        rec["vtable"] = _sym_entry(syms.get("vtable"))
        rec["typeinfo"] = _sym_entry(syms.get("typeinfo"))
        rec["typeinfo_name"] = _sym_entry(syms.get("typeinfo_name"))
        if syms.get("vtable") or syms.get("typeinfo") or syms.get("typeinfo_name"):
            rec["confidence"] = "rtti"
        elif any(m["kind"] in ("ctor", "dtor") for m in rec["methods"]):
            rec["confidence"] = "ctor"
        else:
            rec["confidence"] = "name-only"

    if needle:
        registry = {k: v for k, v in registry.items() if needle in k.lower()}
    return registry
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_read_class.py -x -q -k registry`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add plugin/bn_agent_bridge/read_class.py tests/test_read_class.py
git commit -m "feat(class): class registry + confidence from symbols/RTTI (#205)"
```

---

## Task 3: `class_list` op end-to-end (op + shim + CLI + formatter)

**Files:**
- Modify: `plugin/bn_agent_bridge/read_class.py` (the `_class_list` entry point)
- Modify: `plugin/bn_agent_bridge/bridge.py` (facade shim + `@op` binder)
- Create: `src/bn/commands/cpp_class.py`
- Modify: `src/bn/commands/__init__.py`
- Modify: `src/bn/formatters.py`
- Test: `tests/test_read_class.py`, `tests/test_cli.py`

- [ ] **Step 1: Write the failing bridge test**

```python
# append to tests/test_read_class.py
def test_class_list_envelope_filters_and_pages():
    bv = _make_registry_bv()

    class _Ctx:
        def _resolve_view(self, sel):
            return bv

    out = read_class._class_list(_Ctx(), None, include_all=True)
    names = [c["name"] for c in out["classes"]]
    assert "net::Session" in names and out["total"] == len(names)
    # Default (no --all) drops name-only clusters like the bare "net" namespace.
    confirmed = read_class._class_list(_Ctx(), None, include_all=False)
    assert "net" not in [c["name"] for c in confirmed["classes"]]
    assert "net::Session" in [c["name"] for c in confirmed["classes"]]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_read_class.py -x -q -k class_list`
Expected: FAIL — `_class_list` not defined.

- [ ] **Step 3: Implement `_class_list`**

```python
# append to plugin/bn_agent_bridge/read_class.py
def _list_row(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": rec["name"],
        "method_count": len(rec["methods"]),
        "has_vtable": rec["vtable"] is not None,
        "size": rec["size"],
        "bases": [b.get("name") for b in rec.get("bases", [])],
        "confidence": rec["confidence"],
    }


def _class_list(
    ctx,
    selector: str | None,
    *,
    query: str | None = None,
    include_all: bool = False,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    bv = ctx._resolve_view(selector)
    registry = _build_class_registry(ctx, bv, query=query)
    rows = [
        _list_row(rec)
        for rec in registry.values()
        if include_all or rec["confidence"] in ("rtti", "ctor")
    ]
    rows.sort(key=lambda r: r["name"])
    total = len(rows)
    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return {"classes": rows, "total": total, "offset": offset, "include_all": include_all}
```

- [ ] **Step 4: Add the facade shim + `@op` binder in `bridge.py`**

Add `from . import read_class` next to the existing `from . import read_evidence` import (top of `bridge.py`, ~line 27).

Add the facade shim near the other `read_evidence` shims (~line 1295), on `BinaryNinjaBridge`:

```python
    def _class_list(self, *a, **k):
        return read_class._class_list(self.ctx, *a, **k)

    def _class_show(self, *a, **k):
        return read_class._class_show(self.ctx, *a, **k)
```

Add the binders next to the other read `@op`s (~line 1760):

```python
@op("class_list", lock="read")
def _bind_class_list(bridge, params, target):
    return bridge._class_list(
        target,
        query=params.get("query"),
        include_all=_validate_bool(params.get("include_all"), label="include_all", default=False),
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
    )


@op("class_show", lock="read")
def _bind_class_show(bridge, params, target):
    return bridge._class_show(target, str(params["name"]))
```

- [ ] **Step 5: Add the CLI command + formatter, register the module**

```python
# src/bn/commands/cpp_class.py
from __future__ import annotations

import argparse
from typing import Any

from ..cli import _call, _effective_limit, arg, command
from ..formatters import _render_class_list_text, _render_class_show_text


@command("class", "list", help="List C++ classes recovered from symbols/RTTI",
         target=True, paged=True,
         args=[arg("--all", action="store_true", default=False, dest="all_clusters",
                   help="Include name-only clusters (possible namespaces), not just RTTI/ctor-confirmed classes"),
               arg("--query", help="Filter classes by name substring")])
def _class_list(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {"offset": args.offset}
    if args.query:
        params["query"] = args.query
    if args.all_clusters:
        params["include_all"] = True
    limit = _effective_limit(args)
    if limit is not None:
        params["limit"] = limit
    return _call(
        args, "class_list", params,
        require_target=True, allow_implicit_target=True,
        text_renderer=_render_class_list_text,
        paged_spill=True, page_label="classes", stem="class-list",
    )


@command("class", help="Show a C++ class: methods, vtable, size, bases, instances",
         target=True,
         args=[arg("name")])
def _class_show(args: argparse.Namespace) -> int:
    return _call(
        args, "class_show", {"name": args.name},
        require_target=True, allow_implicit_target=True,
        text_renderer=_render_class_show_text,
        stem="class-show",
    )
```

Register it — add to `src/bn/commands/__init__.py` after the `binary` import:

```python
from . import cpp_class  # noqa: F401
```

Add the list renderer to `src/bn/formatters.py` (the show renderer is added in Task 8 with a minimal stub now so the import resolves):

```python
def _render_class_list_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    rows = list(value.get("classes") or [])
    total = value.get("total", len(rows))
    lines = [f"classes: {len(rows)} shown of {total}"]
    for rec in rows:
        if not isinstance(rec, dict):
            lines.append(_render_fallback_text(rec))
            continue
        vt = "vtable" if rec.get("has_vtable") else "no-vtable"
        size = rec.get("size")
        size_s = size.get("value") if isinstance(size, dict) else size
        bases = ", ".join(b for b in (rec.get("bases") or []) if b)
        base_s = f"  : {bases}" if bases else ""
        lines.append(
            f"  {rec.get('name', '<unknown>')}  "
            f"methods={rec.get('method_count', 0)}  {vt}  "
            f"size={size_s if size_s is not None else '?'}  "
            f"[{rec.get('confidence', '?')}]{base_s}"
        )
    return "\n".join(lines)


def _render_class_show_text(value: Any) -> str:
    # Filled in by Task 8; fallback keeps the import valid meanwhile.
    return _render_fallback_text(value)
```

- [ ] **Step 6: Write the CLI wiring test**

```python
# tests/test_cli.py — add near other command-dispatch tests
def test_class_list_invokes_op(monkeypatch):
    captured = {}

    def fake_call(args, op, params, **kwargs):
        captured["op"] = op
        captured["params"] = params
        return 0

    import bn.commands.cpp_class as cpp_class
    monkeypatch.setattr(cpp_class, "_call", fake_call)
    from bn.cli import build_parser
    args = build_parser().parse_args(["class", "list", "--all", "--query", "Session"])
    assert args.func(args) == 0
    assert captured["op"] == "class_list"
    assert captured["params"]["include_all"] is True
    assert captured["params"]["query"] == "Session"


def test_class_show_invokes_op(monkeypatch):
    captured = {}

    def fake_call(args, op, params, **kwargs):
        captured["op"] = op
        captured["params"] = params
        return 0

    import bn.commands.cpp_class as cpp_class
    monkeypatch.setattr(cpp_class, "_call", fake_call)
    from bn.cli import build_parser
    args = build_parser().parse_args(["class", "net::Session"])
    assert args.func(args) == 0
    assert captured["op"] == "class_show"
    assert captured["params"]["name"] == "net::Session"
```

- [ ] **Step 7: Run to verify pass**

Run: `uv run pytest tests/test_read_class.py tests/test_cli.py -x -q -k "class"`
Expected: PASS. Also `uv run pytest tests/test_op_registry.py -q` (the op-registry consistency test must still pass with the two new ops).

- [ ] **Step 8: Commit**

```bash
git add plugin/bn_agent_bridge/read_class.py plugin/bn_agent_bridge/bridge.py \
        src/bn/commands/cpp_class.py src/bn/commands/__init__.py src/bn/formatters.py \
        tests/test_read_class.py tests/test_cli.py
git commit -m "feat(class): bn class list — discovery over the class registry (#205)"
```

---

## Task 4: Vtable layout

**Files:**
- Modify: `plugin/bn_agent_bridge/read_class.py`
- Test: `tests/test_read_class.py`

Read the class's vtable: skip the two Itanium header words (offset-to-top, typeinfo pointer), resolve each function slot, mark `__cxa_pure_virtual` / unnamed `sub_*`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_read_class.py
class _VtableCtx:
    """Minimal ctx exposing the pointer-table reader over a fake slot map."""
    def __init__(self, slots, pure_addr=0xDEAD):
        # slots: list of (value_addr, function_name_or_None)
        self._slots = slots
        self._pure = pure_addr

    def _pointer_size(self, bv):
        return 8

    def _pointer_table_layout(self, bv, start, *, entries, stride):
        rows = []
        for i, (val, fname) in enumerate(self._slots):
            target = {"status": "function", "normalized": hex(val),
                      "function": ({"name": fname, "address": hex(val)} if fname else None)}
            if val == self._pure:
                target = {"status": "function", "normalized": hex(val),
                          "function": {"name": "__cxa_pure_virtual", "address": hex(val)}}
            rows.append({"index": i, "entry_address": hex(start + i * stride),
                         "value": hex(val), "readable": True, "plausible": True, "target": target})
        return {"entries": rows}


def test_vtable_layout_skips_header_and_marks_slots():
    bv = object()
    slots = [(0x40e8b0, "onData"), (0x40e3d0, None), (0xDEAD, "__cxa_pure_virtual")]
    ctx = _VtableCtx(slots)
    layout = read_class._vtable_layout(ctx, bv, 0x9000)
    assert layout["address"] == "0x9000"
    # header words skipped: read starts at 0x9000 + 2*8
    s0, s1, s2 = layout["slots"]
    assert s0["index"] == 0 and s0["method"]["name"] == "onData"
    assert s1["unnamed"] is True            # sub_* / no symbol
    assert s2["pure_virtual"] is True
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_read_class.py -x -q -k vtable`
Expected: FAIL — `_vtable_layout` not defined.

- [ ] **Step 3: Implement `_vtable_layout`** (delegates to the existing pointer-table reader)

```python
# append to plugin/bn_agent_bridge/read_class.py
def _vtable_layout(ctx, bv, vtable_addr: int, *, max_slots: int = 64) -> dict[str, Any]:
    """Function slots of an Itanium vtable. Words [0] (offset-to-top) and [1]
    (typeinfo ptr) are header; slots start at +2*ptr_size. Reuses the
    Thumb-aware pointer-table reader."""
    ptr = ctx._pointer_size(bv)
    start = vtable_addr + 2 * ptr
    table = ctx._pointer_table_layout(bv, start, entries=max_slots, stride=ptr)
    slots: list[dict[str, Any]] = []
    for i, row in enumerate(table.get("entries") or []):
        target = row.get("target") or {}
        fn = target.get("function") if isinstance(target, dict) else None
        # A null/unmapped slot ends the vtable (next object / padding).
        if not row.get("readable") or target.get("status") not in ("function", "mapped"):
            break
        name = (fn or {}).get("name") if isinstance(fn, dict) else None
        slots.append({
            "index": i,
            "address": row.get("value"),
            "method": fn if isinstance(fn, dict) else None,
            "pure_virtual": name == "__cxa_pure_virtual",
            "unnamed": bool(fn) and isinstance(name, str) and name.startswith("sub_"),
        })
    return {"address": hex(int(vtable_addr)), "slots": slots}
```

Add the matching `ctx` helper name `_pointer_table_layout`. The seam already exposes `_pointer_table_for_view` via the bridge facade; expose a thin `ctx`-level alias so `read_class` does not import `read_evidence` directly. In `seam.py`, add to the `BridgeContext` protocol/impl:

```python
    def _pointer_table_layout(self, bv, start, *, entries, stride):
        from . import read_evidence
        return read_evidence._pointer_table_for_view(
            self, bv, start, entries=entries, stride_size=stride
        )
```

(Deferred import inside the method avoids the read-module import cycle the seam exists to break.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_read_class.py -x -q -k vtable`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugin/bn_agent_bridge/read_class.py plugin/bn_agent_bridge/seam.py tests/test_read_class.py
git commit -m "feat(class): vtable layout via Itanium header-skip + pointer reader (#205)"
```

---

## Task 5: Object size from `operator new` / BN type

**Files:**
- Modify: `plugin/bn_agent_bridge/read_class.py`
- Test: `tests/test_read_class.py`

Size with provenance: prefer a defined BN type's width; else the `operator new(N)` size at a construction site.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_read_class.py
class _SizeCtx:
    def __init__(self, type_width=None, new_size=None):
        self._type_width = type_width
        self._new_size = new_size

    def _find_type(self, bv, name):
        if self._type_width is None:
            return None
        class _T:
            width = self._type_width
        return name, _T()

    def _operator_new_size_at_ctor(self, bv, record):
        return self._new_size  # (size, addr) or None


def test_size_prefers_bn_type_width():
    rec = {"name": "net::Session", "methods": [], "vtable": None}
    out = read_class._object_size(_SizeCtx(type_width=0xD0), object(), rec)
    assert out == {"value": "0xd0", "source": "bn_type"}


def test_size_from_operator_new_when_no_type():
    rec = {"name": "net::Session", "methods": []}
    out = read_class._object_size(_SizeCtx(new_size=(0xD0, 0x443abc)), object(), rec)
    assert out["value"] == "0xd0" and out["source"] == "operator_new"
    assert out["at"] == "0x443abc"


def test_size_none_when_nothing_resolves():
    rec = {"name": "net::Session", "methods": []}
    assert read_class._object_size(_SizeCtx(), object(), rec) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_read_class.py -x -q -k size`
Expected: FAIL — `_object_size` not defined.

- [ ] **Step 3: Implement `_object_size` + the construction-site scan**

```python
# append to plugin/bn_agent_bridge/read_class.py
def _object_size(ctx, bv, record: dict[str, Any]) -> dict[str, Any] | None:
    """Object size with provenance. BN type width wins (authoritative when a
    type is defined); else the operator-new size at a construction site."""
    found = ctx._find_type(bv, record["name"])
    if found is not None:
        _, type_obj = found
        width = int(getattr(type_obj, "width", 0) or 0)
        if width > 0:
            return {"value": hex(width), "source": "bn_type"}
    new = ctx._operator_new_size_at_ctor(bv, record)
    if new is not None:
        size, at = new
        return {"value": hex(int(size)), "source": "operator_new", "at": hex(int(at))}
    return None
```

Add the `ctx` helper `_operator_new_size_at_ctor` in `seam.py`. It finds a ctor method, walks its inbound xrefs, and at each caller looks for a preceding `operator new(N)` call whose return flows into the ctor's `this`. Reuse the existing trace machinery and the canonical operator-new key set already in `taint_engine.py`:

```python
    def _operator_new_size_at_ctor(self, bv, record):
        """(size, addr) from an operator-new feeding a ctor call, else None.
        Best-effort: scans ctor xref sites for a nearby operator-new const arg."""
        from . import read_class  # for the canonical new-operator names
        ctors = [m for m in record.get("methods", []) if m.get("kind") == "ctor"]
        for ctor in ctors:
            for ref in self._inbound_code_refs(bv, ctor["address"]):
                hit = self._operator_new_const_near(bv, ref)
                if hit is not None:
                    return hit
        return None
```

> NOTE for the implementer: `_inbound_code_refs` and `_operator_new_const_near` are thin wrappers over the existing xref + MLIL-const machinery. If the underlying helper names differ in `seam.py`/`read_xrefs.py`, bind to the actual ones — do not invent new analysis. The unit test mocks `_operator_new_size_at_ctor` directly, so this wiring is validated by the Task 9 dogfood, not the unit test. Keep the function returning `None` (size unknown) rather than guessing when the const cannot be recovered.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_read_class.py -x -q -k size`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugin/bn_agent_bridge/read_class.py plugin/bn_agent_bridge/seam.py tests/test_read_class.py
git commit -m "feat(class): object size from BN type width / operator new (#205)"
```

---

## Task 6: RTTI base-class decode

**Files:**
- Modify: `plugin/bn_agent_bridge/read_class.py`
- Test: `tests/test_read_class.py`

Decode the `_ZTI` typeinfo object: `__class` (no base), `__si_class` (single base ptr), `__vmi_class` (count + base records). Resolve base typeinfo pointers back to class names via the registry's typeinfo symbol map. Structural fallback when the `__*_class_type_info` selector symbols are stripped.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_read_class.py
import struct


class _RttiCtx:
    """ctx reading little-endian 8-byte words from a fake memory map and
    resolving a typeinfo address back to a class name."""
    def __init__(self, words, ti_names):
        self._words = words            # {addr: int}
        self._ti_names = ti_names      # {addr: class_name}

    def _pointer_size(self, bv):
        return 8

    def _read_pointer_value(self, bv, addr, *, size=None):
        return self._words.get(addr)

    def _read_u32(self, bv, addr):
        return self._words.get(addr)

    def _typeinfo_name_at(self, bv, addr):
        return self._ti_names.get(addr)


def test_rtti_si_single_base():
    # __si_class_type_info: [vptr][name-ptr][base-ti-ptr]
    words = {0x9100: 0xAB00, 0x9108: 0x9300, 0x9110: 0x9200}
    ctx = _RttiCtx(words, {0x9200: "net::Endpoint"})
    bases = read_class._rtti_bases(ctx, object(), 0x9100, kind_hint="si")
    assert bases == [{"name": "net::Endpoint", "address": "0x9200", "kind": "public"}]


def test_rtti_vmi_multiple_bases():
    # __vmi_class_type_info: [vptr][name][flags][count][ (base-ti, off_flags) * count ]
    words = {
        0x9100: 0xAB10, 0x9108: 0x9300, 0x9110: 0x0, 0x9118: 2,
        0x9120: 0x9200, 0x9128: (2 << 8) | 0x2,   # public flag
        0x9130: 0x9240, 0x9138: (4 << 8) | 0x2,
    }
    ctx = _RttiCtx(words, {0x9200: "net::A", 0x9240: "net::B"})
    bases = read_class._rtti_bases(ctx, object(), 0x9100, kind_hint="vmi")
    assert [b["name"] for b in bases] == ["net::A", "net::B"]


def test_rtti_no_base():
    ctx = _RttiCtx({0x9100: 0xAB20, 0x9108: 0x9300}, {})
    assert read_class._rtti_bases(ctx, object(), 0x9100, kind_hint="base") == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_read_class.py -x -q -k rtti`
Expected: FAIL — `_rtti_bases` not defined.

- [ ] **Step 3: Implement `_rtti_bases`**

```python
# append to plugin/bn_agent_bridge/read_class.py
def _rtti_bases(ctx, bv, typeinfo_addr: int, *, kind_hint: str | None = None) -> list[dict[str, Any]]:
    """Base classes from an Itanium ``_ZTI`` object. ``kind_hint`` selects the
    layout: 'base' (no bases), 'si' (single), 'vmi' (multiple). When absent,
    infer structurally from the base-typeinfo pointers that resolve."""
    ptr = ctx._pointer_size(bv)
    name_field = typeinfo_addr + ptr          # word[1] = type-name ptr
    after_name = typeinfo_addr + 2 * ptr      # first layout-specific word

    def resolve(ti_addr: int, kind: str = "public") -> dict[str, Any]:
        return {
            "name": ctx._typeinfo_name_at(bv, ti_addr),
            "address": hex(int(ti_addr)),
            "kind": kind,
        }

    kind = kind_hint or _infer_rtti_kind(ctx, bv, typeinfo_addr, ptr)
    if kind == "base":
        return []
    if kind == "si":
        base = ctx._read_pointer_value(bv, after_name, size=ptr)
        return [resolve(base)] if base else []
    if kind == "vmi":
        # [flags:u32][base_count:u32] then base_count * (base-ti-ptr, off_flags)
        count = ctx._read_u32(bv, after_name + 4) or 0
        rec = after_name + 8
        out: list[dict[str, Any]] = []
        for _ in range(min(int(count), 64)):
            ti = ctx._read_pointer_value(bv, rec, size=ptr)
            off_flags = ctx._read_pointer_value(bv, rec + ptr, size=ptr) or 0
            rec += 2 * ptr
            if not ti:
                continue
            kind_s = "virtual" if (off_flags & 0x1) else "public"
            out.append(resolve(ti, kind_s))
        return out
    return []


def _infer_rtti_kind(ctx, bv, typeinfo_addr: int, ptr: int) -> str:
    """Best-effort layout inference when the __*_class_type_info selector
    symbol is unavailable: a resolvable single base ptr -> 'si'; a small
    plausible count followed by resolvable base ptrs -> 'vmi'; else 'base'."""
    after_name = typeinfo_addr + 2 * ptr
    candidate = ctx._read_pointer_value(bv, after_name, size=ptr)
    if candidate and ctx._typeinfo_name_at(bv, candidate):
        return "si"
    count = ctx._read_u32(bv, after_name + 4) or 0
    if 0 < count <= 16:
        first = ctx._read_pointer_value(bv, after_name + 8, size=ptr)
        if first and ctx._typeinfo_name_at(bv, first):
            return "vmi"
    return "base"
```

Add the `ctx` helpers `_read_u32` and `_typeinfo_name_at` in `seam.py`:

```python
    def _read_u32(self, bv, address):
        try:
            data = bytes(bv.read(address, 4))
        except Exception:
            return None
        return int.from_bytes(data, self._byteorder(bv), signed=False) if len(data) == 4 else None

    def _typeinfo_name_at(self, bv, address):
        """Class name for a _ZTI object address: prefer its data symbol's
        demangled 'typeinfo for X' marker; None if unresolved."""
        sym = bv.get_symbol_at(address) if hasattr(bv, "get_symbol_at") else None
        if sym is None:
            return None
        short = str(getattr(sym, "short_name", "") or "")
        for marker in ("typeinfo for ", "_typeinfo_for_"):
            if short.startswith(marker):
                return short[len(marker):].strip()
        return None
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_read_class.py -x -q -k rtti`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add plugin/bn_agent_bridge/read_class.py plugin/bn_agent_bridge/seam.py tests/test_read_class.py
git commit -m "feat(class): Itanium RTTI base-class decode (si/vmi + fallback) (#205)"
```

---

## Task 7: Instance tracking (best-effort)

**Files:**
- Modify: `plugin/bn_agent_bridge/read_class.py`
- Test: `tests/test_read_class.py`

Construction sites from ctor xrefs, classified new/stack/global; global slots storing the vtable address.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_read_class.py
class _InstCtx:
    def __init__(self, ctor_sites, vtable_data_refs):
        self._ctor_sites = ctor_sites          # list of dicts
        self._vtable_data_refs = vtable_data_refs

    def _ctor_construction_sites(self, bv, record):
        return list(self._ctor_sites)

    def _global_vtable_stores(self, bv, record):
        return list(self._vtable_data_refs)


def test_instances_collects_construction_and_global_stores():
    sites = [
        {"address": "0x443abc", "function": "net::open", "kind": "new", "size": "0xd0"},
        {"address": "0x4500a0", "function": "main", "kind": "stack", "size": None},
    ]
    globals_ = [{"symbol": "g_session", "address": "0x4cabcd"}]
    ctx = _InstCtx(sites, globals_)
    rec = {"name": "net::Session", "vtable": {"address": "0x9000"}, "methods": []}
    out = read_class._instances(ctx, object(), rec)
    assert out["construction_sites"][0]["kind"] == "new"
    assert out["stored_globals"][0]["symbol"] == "g_session"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_read_class.py -x -q -k instances`
Expected: FAIL — `_instances` not defined.

- [ ] **Step 3: Implement `_instances`**

```python
# append to plugin/bn_agent_bridge/read_class.py
def _instances(ctx, bv, record: dict[str, Any], *, cap: int = 128) -> dict[str, Any]:
    """Best-effort: where objects of this class are constructed and which
    globals hold one. Empty (not an error) when nothing is found."""
    sites = ctx._ctor_construction_sites(bv, record)[:cap]
    stored = ctx._global_vtable_stores(bv, record)[:cap] if record.get("vtable") else []
    return {"construction_sites": sites, "stored_globals": stored}
```

Add `ctx` helpers in `seam.py` (thin wrappers over existing xref/trace machinery; return `[]` when not recoverable — never fabricate):

```python
    def _ctor_construction_sites(self, bv, record):
        """For each ctor, its inbound call sites classified new/stack/global.
        Reuses the existing inbound-xref + operator-new const machinery."""
        sites = []
        for ctor in [m for m in record.get("methods", []) if m.get("kind") == "ctor"]:
            for ref in self._inbound_code_refs(bv, ctor["address"]):
                sites.append(self._classify_construction_site(bv, ref))
        return sites

    def _global_vtable_stores(self, bv, record):
        """Global data symbols whose stored value is this class's vtable addr."""
        vt = record.get("vtable")
        if not vt:
            return []
        addr = int(vt["address"], 16)
        out = []
        for ref in (bv.get_data_refs(addr) if hasattr(bv, "get_data_refs") else []):
            sym = bv.get_symbol_at(int(ref)) if hasattr(bv, "get_symbol_at") else None
            out.append({
                "symbol": str(getattr(sym, "short_name", "") or getattr(sym, "name", "")) if sym else None,
                "address": hex(int(ref)),
            })
        return out
```

> NOTE for the implementer: `_inbound_code_refs` and `_classify_construction_site` wrap the existing xref/trace helpers (`read_xrefs`, the `trace`/operator-new machinery). Bind to the real helper names in `seam.py`/`read_xrefs.py`; do not add new analysis. `_classify_construction_site` returns `{address, function, kind: new|stack|global, size}` — `kind="new"` with a `size` only when an operator-new const is recovered, else `stack`/`global` by the `this` storage, with `size: None`. The unit test mocks these directly; real behavior is validated in the Task 9 dogfood.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_read_class.py -x -q -k instances`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugin/bn_agent_bridge/read_class.py plugin/bn_agent_bridge/seam.py tests/test_read_class.py
git commit -m "feat(class): best-effort instance/construction-site tracking (#205)"
```

---

## Task 8: `class_show` assembly + text renderer

**Files:**
- Modify: `plugin/bn_agent_bridge/read_class.py`
- Modify: `src/bn/formatters.py`
- Test: `tests/test_read_class.py`, `tests/test_cli.py`

Assemble the full `ClassRecord` for one class (registry + vtable + size + bases + instances) and render the issue's text mock. Unknown class → `OperationFailure` with a discovery hint.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_read_class.py
def test_class_show_assembles_full_record():
    bv = _make_registry_bv()

    class _ShowCtx:
        def _resolve_view(self, sel):
            return bv
        def _pointer_size(self, b):
            return 8
        def _vtable_layout_for(self, b, addr):
            return {"address": hex(addr), "slots": [
                {"index": 0, "address": "0x40e8b0", "method": {"name": "onData"},
                 "pure_virtual": False, "unnamed": False}]}
        def _object_size_for(self, b, rec):
            return {"value": "0xd0", "source": "operator_new", "at": "0x443abc"}
        def _bases_for(self, b, rec):
            return [{"name": "net::Endpoint", "address": "0x9200", "kind": "public"}]
        def _instances_for(self, b, rec):
            return {"construction_sites": [], "stored_globals": []}

    out = read_class._class_show(_ShowCtx(), None, "net::Session")
    assert out["name"] == "net::Session"
    assert out["size"]["value"] == "0xd0"
    assert out["bases"][0]["name"] == "net::Endpoint"
    assert out["vtable"]["slots"][0]["method"]["name"] == "onData"


def test_class_show_unknown_name_errors_with_hint():
    bv = _make_registry_bv()

    class _Ctx:
        def _resolve_view(self, sel):
            return bv

    import pytest
    with pytest.raises(read_class.OperationFailure) as exc:
        read_class._class_show(_Ctx(), None, "net::Nope")
    assert "class list" in str(exc.value).lower()


def test_class_show_ambiguous_returns_all_matches():
    # Two classes share a leaf name across namespaces.
    fns = [
        _Fn(0x1000, "_ZN1a3FooC1Ev", "a::Foo::Foo()"),
        _Fn(0x2000, "_ZN1b3FooC1Ev", "b::Foo::Foo()"),
    ]
    bv = _RegistryBV(fns, [])

    class _Ctx:
        def _resolve_view(self, sel):
            return bv
        def _vtable_layout_for(self, b, a):
            return None
        def _object_size_for(self, b, r):
            return None
        def _bases_for(self, b, r):
            return []
        def _instances_for(self, b, r):
            return {"construction_sites": [], "stored_globals": []}

    out = read_class._class_show(_Ctx(), None, "Foo")
    assert out["ambiguous"] is True
    assert {m["name"] for m in out["matches"]} == {"a::Foo", "b::Foo"}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_read_class.py -x -q -k class_show`
Expected: FAIL — `_class_show` not defined.

- [ ] **Step 3: Implement `_class_show` (+ thin per-class ctx aliases)**

```python
# append to plugin/bn_agent_bridge/read_class.py
def _resolve_class_names(registry: dict[str, dict], name: str) -> list[str]:
    """Exact match, else all classes whose leaf component equals *name*."""
    if name in registry:
        return [name]
    leaf = _last_component(name) if "::" in name else name
    return sorted(k for k in registry if _last_component(k) == leaf)


def _enrich(ctx, bv, rec: dict[str, Any]) -> dict[str, Any]:
    if rec.get("vtable"):
        rec["vtable"] = ctx._vtable_layout_for(bv, int(rec["vtable"]["address"], 16)) or rec["vtable"]
    rec["size"] = ctx._object_size_for(bv, rec)
    rec["bases"] = ctx._bases_for(bv, rec)
    rec["instances"] = ctx._instances_for(bv, rec)
    return rec


def _class_show(ctx, selector: str | None, name: str) -> dict[str, Any]:
    bv = ctx._resolve_view(selector)
    registry = _build_class_registry(ctx, bv)
    matches = _resolve_class_names(registry, name)
    if not matches:
        raise OperationFailure(
            "unknown_class",
            f"No class named {name!r}. Run `bn class list` (add --all for "
            f"name-only clusters) to discover available classes.",
        )
    enriched = [_enrich(ctx, bv, registry[m]) for m in matches]
    if len(enriched) == 1:
        return enriched[0]
    return {"ambiguous": True, "query": name, "matches": enriched}
```

Add the per-class `ctx` aliases in `seam.py` (delegating to the `read_class` free functions; keeps `read_class` from importing `read_evidence`/itself through the bridge):

```python
    def _vtable_layout_for(self, bv, addr):
        from . import read_class
        return read_class._vtable_layout(self, bv, addr)

    def _object_size_for(self, bv, record):
        from . import read_class
        return read_class._object_size(self, bv, record)

    def _bases_for(self, bv, record):
        from . import read_class
        ti = record.get("typeinfo")
        if not ti:
            return []
        return read_class._rtti_bases(self, bv, int(ti["address"], 16))

    def _instances_for(self, bv, record):
        from . import read_class
        return read_class._instances(self, bv, record)
```

- [ ] **Step 4: Implement the `_render_class_show_text` renderer** (replace the Task 3 stub)

```python
# src/bn/formatters.py — replace the stub _render_class_show_text
def _render_class_show_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _render_fallback_text(value)
    if value.get("ambiguous"):
        out = [f"ambiguous class {value.get('query', '')!r}: {len(value.get('matches') or [])} matches"]
        for rec in value.get("matches") or []:
            out.append("")
            out.append(_render_one_class(rec))
        return "\n".join(out)
    return _render_one_class(value)


def _render_one_class(rec: Any) -> str:
    if not isinstance(rec, dict):
        return _render_fallback_text(rec)
    size = rec.get("size")
    size_s = size.get("value") if isinstance(size, dict) else None
    vt = rec.get("vtable") if isinstance(rec.get("vtable"), dict) else None
    vt_addr = vt.get("address") if vt else None
    bases = ", ".join(b.get("name") or "?" for b in (rec.get("bases") or []))
    head = f"class {rec.get('name', '<unknown>')}"
    bits = []
    if size_s:
        bits.append(f"size {size_s}")
    if vt_addr:
        bits.append(f"vtable @ {vt_addr}")
    if bases:
        bits.append(f"base: {bases}")
    if bits:
        head += "  (" + ", ".join(bits) + ")"
    lines = [head, f"  [{rec.get('confidence', '?')}]"]
    for m in rec.get("methods") or []:
        if m.get("kind") in ("ctor", "dtor"):
            lines.append(f"  {m['kind']:<6} {m.get('address', '?')}  {m.get('demangled', '')}")
    if vt and vt.get("slots"):
        for s in vt["slots"]:
            method = s.get("method") or {}
            label = (
                "__cxa_pure_virtual" if s.get("pure_virtual")
                else method.get("name") if isinstance(method, dict) and method.get("name")
                else "<unnamed>"
            )
            lines.append(f"  vtable [{s.get('index')}] {s.get('address', '?')}  {label}")
    inst = rec.get("instances") if isinstance(rec.get("instances"), dict) else {}
    parts = []
    for site in inst.get("construction_sites") or []:
        sz = f" (size {site['size']})" if site.get("size") else ""
        parts.append(f"{site.get('kind', '?')} @ {site.get('address', '?')}{sz}")
    for g in inst.get("stored_globals") or []:
        parts.append(f"stored -> {g.get('symbol') or '?'} @ {g.get('address', '?')}")
    if parts:
        lines.append("  instances: " + " ; ".join(parts))
    return "\n".join(lines)
```

- [ ] **Step 5: Write the renderer + bridge dispatch test**

```python
# tests/test_read_class.py — text rendering
def test_render_class_show_text_matches_mock_shape():
    from bn.formatters import _render_class_show_text
    rec = {
        "name": "net::Session", "confidence": "rtti",
        "size": {"value": "0xd0", "source": "operator_new"},
        "vtable": {"address": "0x4e0000", "slots": [
            {"index": 0, "address": "0x40e8b0", "method": {"name": "onData"}, "pure_virtual": False, "unnamed": False},
            {"index": 1, "address": "0x0", "method": None, "pure_virtual": True, "unnamed": False}]},
        "bases": [{"name": "net::Endpoint"}],
        "methods": [{"kind": "ctor", "address": "0x40abc0", "demangled": "net::Session::Session()"}],
        "instances": {"construction_sites": [{"kind": "new", "address": "0x443abc", "size": "0xd0"}],
                      "stored_globals": [{"symbol": "g_session", "address": "0x4cabcd"}]},
    }
    text = _render_class_show_text(rec)
    assert "class net::Session" in text and "size 0xd0" in text and "base: net::Endpoint" in text
    assert "vtable [0] 0x40e8b0  onData" in text
    assert "vtable [1] 0x0  __cxa_pure_virtual" in text
    assert "instances: new @ 0x443abc (size 0xd0) ; stored -> g_session @ 0x4cabcd" in text
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/test_read_class.py tests/test_cli.py -q -k "class"`
Expected: PASS.

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (no regressions; the prior count + the new tests).

- [ ] **Step 8: Commit**

```bash
git add plugin/bn_agent_bridge/read_class.py plugin/bn_agent_bridge/seam.py \
        src/bn/formatters.py tests/test_read_class.py
git commit -m "feat(class): assemble bn class <Name> full view + text renderer (#205)"
```

---

## Task 9: Sanitized real-BN dogfood + docs

**Files:**
- Modify: `plugin/bn_agent_bridge/` (only if dogfood reveals a real defect)
- Verify against a real BN install.

This validates the BN-API wiring the unit tests mock (operator-new size, construction-site classification, RTTI reads on real layouts).

- [ ] **Step 1: Restart any stale bridge so it runs current code**

The bridge must run the new module. For a headless instance loaded before this change, restart it:

```bash
uv run bn doctor --instance <id>      # check stale_plugin_code
uv run bn session stop <id>           # if stale; then re-start against the binary
```

- [ ] **Step 2: Run discovery against a real C++ target** (read-only; pass `-t`/`--instance` explicitly; never `instance use`)

```bash
uv run bn class list --instance <id> -t <view-id> --limit 20
uv run bn class list --instance <id> -t <view-id> --all --query <substr>
```

Expected: a list of classes with method counts, vtable flags, confidence. Confirm RTTI-confirmed classes appear without `--all` and namespace-like clusters only with `--all`.

- [ ] **Step 3: Drill into one class and verify each capability**

```bash
uv run bn class <ClassName> --instance <id> -t <view-id>
uv run bn class <ClassName> --instance <id> -t <view-id> --format json --out /tmp/cls.json
```

Verify against the binary: methods cluster correctly; the vtable layout's slot 0 == the first method after the 2 header words (cross-check with `bn evidence table <vtable-addr>`); RTTI bases resolve; `operator new` size is recovered at a construction site (cross-check with `bn decompile <ctor-caller>`). Fix any wiring defect in `seam.py` (the `_inbound_code_refs` / `_operator_new_const_near` / `_classify_construction_site` bindings) at its root — do not stub.

- [ ] **Step 4: Update the skill doc**

Add a short `bn class` / `bn class list` entry to `~/.claude/skills/bn/SKILL.md` (or the in-repo skill source if it lives in the repo) under the read-flow section, with one sanitized example. Keep it consistent with the evidence-family entries.

- [ ] **Step 5: Final full-suite run + commit**

```bash
uv run pytest -q
git add -A && git commit -m "docs(class): bn class skill entry; dogfood fixes (#205)"
```

> **Sanitization:** No real class/symbol names, addresses, or decompiled output go into any committed file, commit message, issue, or PR. Reproduce any finding with the invented names from the spec (`net::Session`, `net::Endpoint`, …). The secrets pre-commit hook will block the dogfood mount path — keep it out of committed text.

---

## Self-Review

**Spec coverage:**
- §3.1 methods grouped → Task 2 (clustering + kind) + Task 8 (render ctor/dtor/vtable groups). ✓
- §3.2 vtable layout → Task 4. ✓
- §3.3 object size + provenance → Task 5. ✓
- §3.4 base classes / RTTI (+ structural fallback §4.6) → Task 6 (`_infer_rtti_kind`). ✓
- §3.5 instances (best-effort) → Task 7. ✓
- §4.1 registry → Task 2. §4.2 depth-aware split → Task 1. §4.3 confidence + `--all` → Task 2 + Task 3. ✓
- §4.4 vtable map + header skip → Task 4. ✓
- §5 command surface (`bn class`, `bn class list`, dual-command pattern, JSON envelope, unknown-class exit) → Task 3 + Task 8. ✓
- §6 edge cases: ambiguous name → Task 8 (`_resolve_class_names`); no-C++ → empty list (Task 3); stripped RTTI → vtable still read, bases degrade (Task 6 fallback). ✓
- §7 testing: mocked-BN units (Tasks 1-8) + sanitized dogfood (Task 9). ✓

**Placeholder scan:** Two `NOTE for the implementer` blocks (Tasks 5, 7) intentionally bind to existing xref/operator-new helpers whose exact names must be confirmed in-repo rather than invented — this is a real instruction (do not add new analysis), not a placeholder, and the units are validated by the Task 9 dogfood since the unit tests mock those ctx methods. All code steps contain complete code.

**Type consistency:** `ClassRecord` keys (`name`, `methods`, `vtable`, `typeinfo`, `typeinfo_name`, `size`, `bases`, `instances`, `confidence`) are identical across Tasks 2/3/8. Method entry keys (`address`, `mangled`, `demangled`, `kind`) consistent. Vtable slot keys (`index`, `address`, `method`, `pure_virtual`, `unnamed`) consistent between Task 4 and Task 8 renderer. `_split_qualified_method`, `_build_class_registry`, `_class_list`, `_class_show`, `_vtable_layout`, `_object_size`, `_rtti_bases`, `_instances` names are stable across tasks and bridge shims.
