# Design: C++ object-model lens (`bn class`) — issue #205

Status: approved (design), pending implementation plan
Date: 2026-06-18
Issue: #205

> All examples use invented class/symbol names (`net::Session`, `net::Endpoint`, …)
> and fabricated addresses. No data from any real target appears in this document.

## 1. Problem

Reverse-engineering a stripped-but-symbolicated **C++** firmware stack, `bn`
presents a flat list of mangled symbols. The binary's real structure — classes,
vtables, object sizes, base classes, instances — has to be rebuilt by hand,
every session, for every class. The recurring friction in C++ RE is everything
one level up from a single function: the **object model**. Several already-filed
papercuts (#196 mangled display, #198 no `exports`, #200 namespaced types
unusable, #201 `xrefs <demangled>` → veneer) are symptoms of the tool having no
notion of "class."

This design adds a first-class **C++ object-model lens**: a `class` command
group that correlates data `bn` already recovers (demangled symbols, vtable data
symbols, RTTI, `operator new` sizes) into a class-aware view. It is mostly a
**correlation + presentation** layer over existing evidence primitives, not new
binary analysis.

## 2. Goals / non-goals

**Goals**
- `bn class <Name>` — a detailed single-class view delivering all five
  capabilities below.
- `bn class list` — discovery: enumerate classes in a freshly-loaded target
  before names are known.
- Reuse existing evidence primitives; add no new heavy analysis.
- Pure read-locked ops, spill-aware, `--format json` everywhere, text mode that
  mirrors the issue's mock.

**Non-goals**
- No mutation (the lens never renames, retypes, or defines types). Acting on its
  findings stays with the existing mutation commands.
- No demangler of our own — rely on Binary Ninja's demangled `short_name`.
- No re-mangling of a user-supplied class name (see §4.2 for why scanning
  beats re-mangling).
- Cross-binary class correlation is out of scope.

## 3. Capabilities (the five, per #205)

For a recovered/known C++ class `bn class <Name>` returns:

1. **Methods**, demangled and grouped (ctor / dtor / virtual / non-virtual),
   each with address + mangled + demangled name.
2. **Vtable layout** — slot index → method, resolved from the class's vtable
   data symbol (Thumb-normalized via the existing reader), with pure-virtual /
   thunk / unnamed-`sub_*` slots marked.
3. **Object size + layout** — size inferred from `operator new` / `new[]` at
   construction sites and/or a BN struct type of the same name; any recovered
   struct fields. Reported **with provenance**.
4. **Base classes / RTTI** — from the typeinfo object, when present.
5. **Instances** — where objects are constructed (`new` / stack / global) and
   where instance pointers are stored. Explicitly **best-effort**.

`bn class list` returns, per class: name, method count, `vtable?`, `size?`,
base classes, and a confidence marker (§4.3).

## 4. Architecture (Approach A — dedicated correlation layer)

```
CLI: src/bn/commands/cpp_class.py   (module named cpp_class to avoid the
                                      `class` keyword; commands are still
                                      `bn class` / `bn class list`)
        |  @command("class", ...)            -> op "class_show"
        |  @command("class", "list", ...)    -> op "class_list"
        v
  _call(...)  --(JSON over unix socket)-->  bridge
        ^
        |  formatters.py: _render_class_show_text / _render_class_list_text
        v
Bridge: plugin/bn_agent_bridge/read_class.py   (free functions taking ctx)
        |  facade shims on BinaryNinjaBridge
        |  @op("class_list",  lock="read")
        |  @op("class_show",  lock="read")
        v
  reuses: read_evidence._pointer_table_for_view  (vtable slots, Thumb-aware)
          read_evidence._RTTI_PREFIXES / _resolve_rtti_symbols / _symbol_by_any_name
          il_format._display_name                (demangled short_name)
          ctx._normalize_code_pointer / ctx._read_pointer_value / bv.read
          trace / operator-new detection         (object size at ctor sites)
```

Rationale for A over the alternatives:
- **B (fold into `bn evidence`)** — `read_evidence.py` is already the largest
  read module; discovery (`class list`) does not fit the `evidence <thing>`
  shape; #205 explicitly asks for `bn class`.
- **C (CLI-side stitching)** — the CLI has no BN dependency by design (no symbol
  enumeration, demangling, or RTTI reads), so this would need many round-trips
  and a re-implemented demangler. Rejected.

### 4.1 The class registry (core abstraction)

One pass over `bv.functions` + `bv.get_symbols()` produces, keyed by demangled
class name, a record:

```python
ClassRecord = {
  "name": "net::Session",          # demangled, fully-qualified
  "methods": [ {address, mangled, demangled, kind} ],  # kind: ctor|dtor|virtual|method
  "vtable":  {address, symbol} | None,
  "typeinfo": {address, symbol} | None,
  "typeinfo_name": {address, symbol} | None,
  "size": {value, source} | None,   # source: "operator_new" | "bn_type" | "rtti"
  "bases": [ {name, offset, kind} ],# kind: public|virtual (from RTTI)
  "instances": [...],               # see 4.7
  "confidence": "rtti" | "ctor" | "name-only",
}
```

The registry is built lazily per request and not cached across requests
(consistent with other read ops; correctness over speed). `class_list` returns
all records (filtered); `class_show <Name>` builds the registry, then resolves
`<Name>` and enriches that one record with vtable layout, sizes, RTTI bases, and
instances (the per-class drill-downs that are too expensive to run for every
class during a list).

### 4.2 Method clustering (depth-aware name split)

A function's demangled `short_name` (via `il_format._display_name`) looks like
`Ns::Outer::method(args)`. The class is the qualified prefix before the final
top-level `::<component>`. The split MUST be depth-aware:

- Ignore `::` nested inside `<…>` (templates: `Vec<Pair<A,B>>::push`).
- Ignore `::` and the signature inside `(…)` (the parameter list).
- Recognize ctor / dtor: trailing component equals the last class component
  (`Session::Session`) or `~` + it (`Session::~Session`).
- `operator` methods: the trailing component may contain `<`, `(`, `=`
  (`operator<<`, `operator()`, `operator new`); split on the last top-level `::`
  that precedes the signature, not on punctuation inside the method name.
- Free functions in a namespace (`ns::func(args)`) are name-indistinguishable
  from a method. They are clustered too but flagged `name-only` confidence (see
  §4.3); RTTI / ctor signals promote a cluster to a real class.

A single shared helper `split_qualified_method(demangled) -> (class, method)`
implements this and is unit-tested against the tricky cases above. Functions
with no `::` (C-style / global) are skipped.

### 4.3 Confidence and `class list` defaults

Each cluster gets a confidence:
- `rtti` — has a `_ZTV` (vtable) and/or `_ZTI`/`_ZTS` (typeinfo) symbol. A real,
  RTTI-emitting class.
- `ctor` — no RTTI symbol, but has a member whose demangled name is `C::C` or
  `C::~C`. A real class without RTTI (non-polymorphic, or RTTI stripped).
- `name-only` — neither; could be a namespace or a POD-ish class.

`bn class list` **defaults to `rtti` + `ctor`** clusters (the things that are
almost certainly classes). `--all` adds `name-only` clusters. `--query <substr>`
filters by name. `--format json` carries the confidence field regardless so a
caller can re-filter.

### 4.4 Vtable resolution and layout

- **Name → vtable symbol map:** scan data symbols whose raw name starts with
  `_ZTV`; the class name is the demangled `short_name` with the leading
  `vtable for ` / `_vtable_for_` marker stripped. (BN renders the demangled
  vtable symbol with that marker.) Build `{class_name: symbol}`. This avoids
  re-mangling the user's class name.
- **Layout:** the `_ZTV` object is the Itanium ABI vtable: word[0] =
  offset-to-top, word[1] = pointer to the class typeinfo, word[2…] = the virtual
  function pointers. Read slots via
  `_pointer_table_for_view(start = vtable_addr + 2*ptr_size, stride = ptr_size)`
  (already Thumb/ARM-aware and plausibility-tagging). Stop at the next vtable
  data symbol, the first null/non-function pointer run, or a slot cap.
- **Slot annotation:** resolve each slot to a function; mark
  `__cxa_pure_virtual`, thunks (BN thunk detection / `il_format`), and unnamed
  `sub_*` (vtable slots frequently recover method addresses BN never
  symbolized — surface these so the user can `bn function create` them).
- Word[1] (typeinfo pointer) feeds §4.6 base-class resolution.

A class with no `_ZTV` (non-polymorphic) simply has `vtable: null` and no
vtable section in the output — not an error.

### 4.5 Object size

Reported with provenance, first available wins but all found are listed:
- `operator_new` — at a construction site (xref to the ctor whose `this`/arg0
  originates from a recent `operator new(N)` / `operator new[](N)` return), read
  the size argument `N`. Reuses the existing trace / evidence argument
  machinery and the canonical operator-new key set already in `taint_engine.py`.
- `bn_type` — if a BN struct/class type named `<Name>` is defined, its
  `type.width`.
- `rtti` — not a direct size source in Itanium RTTI; included only if a future
  signal provides it. (Listed for completeness; may yield nothing.)

Size is `null` when nothing resolves — never fabricated.

### 4.6 Base classes / RTTI

Parse the `_ZTI<C>` typeinfo object pointed to by vtable word[1] (or found via
the `_ZTI` symbol map). Itanium ABI: the first word is a vptr into one of three
standard typeinfo vtables, which selects the layout:
- `__class_type_info` — no bases. Layout: [vptr][name-ptr].
- `__si_class_type_info` — single public base. Layout: [vptr][name-ptr]
  [base-typeinfo-ptr]. Resolve the base ptr → `_ZTI` symbol → class name via the
  map.
- `__vmi_class_type_info` — multiple / virtual inheritance. Layout: [vptr]
  [name-ptr][flags:uint32][base_count:uint32][ (base-typeinfo-ptr,
  offset_flags:long) * base_count ]. Decode each base ptr → name; the
  offset_flags low bits carry public/virtual flags and the high bits the offset.

The vptr is matched against the (demangled) names of the three
`__*_class_type_info` symbols when present; when those symbols are absent
(common in stripped images), fall back to **structural inference**: a non-null
base-typeinfo pointer at the `__si` offset that resolves to a known `_ZTI`
symbol indicates single inheritance; a plausible `base_count` followed by that
many resolvable typeinfo pointers indicates the `__vmi` shape. Bases that don't
resolve are reported as `{name: null, address}` rather than dropped. This step
is best-effort and clearly degrades to "no bases recovered."

### 4.7 Instances (best-effort)

- **Construction sites:** `bn xrefs` to the ctor symbol(s). For each caller,
  classify the object storage:
  - `new` — `this`/arg0 traces back to an `operator new(N)` return; record the
    site address, containing function, and size `N`.
  - `stack` — `this` is a stack slot (frame-relative).
  - `global` — `this` is a fixed data address; record the global data symbol.
- **Stored instances:** global data symbols whose stored value is the class's
  vtable address (object embeds the vptr) or the address of a `new`-constructed
  instance; report the data symbol (`stored -> g_session @ 0x…`).

Bounded by a slot/result cap and spill-aware. When nothing is found, the section
is empty (not an error). This is the least precise capability and is documented
as such in `--help` and output.

## 5. Command surface & output

```
bn class show <Name> [--format text|json] [--out PATH]
bn class list [--all] [--query SUBSTR] [--limit N] [--offset N]
              [--format text|json] [--out PATH]
```

> **Surface note (revised during implementation).** The original design proposed
> a bare `bn class <Name>` alongside `bn class list`, citing the `bn types` /
> `bn types show` dual pattern. That pattern only works because bare `bn types`
> has **no positional** — it has only optional flags. A bare `bn class <Name>`
> needs a positional `name` on the `class` parser, but `bn class list` requires a
> subparsers action on that same node; argparse treats both as positionals, so
> `bn class list` would bind `name="list"`. The show command is therefore
> registered as `bn class show <Name>` — matching the existing `struct show` /
> `types show` convention — keeping both user-selected commands as clean
> subcommands. (#205 offered `bn class <Name>` *and/or* `bn type show --class`;
> `bn class show <Name>` realizes the same intent.)

**Text — `bn class show <Name>` (mirrors the issue mock):**
```
class net::Session  (size 0xd0, vtable @ 0x4xxxxx, base: net::Endpoint)
  ctor   0x40abc0  net::Session::Session(uint8_t, Router*)
  dtor   0x40a860  net::Session::~Session()
  vtable [0] 0x40e8b0  onData(IoBuffer const&, uint8_t*, size_t)
         [1] 0x40e3d0  onConfig(...)
         [2] __cxa_pure_virtual
  instances: new @ 0x443abc (size 0xd0) ; stored -> g_session @ 0x4cxxxx
```

**Text — `bn class list`:** one row per class: `name  methods=N  vtable=Y/N
size=0x..  bases=...  [confidence]`.

**JSON:** the full `ClassRecord` (§4.1) for `class show`; a `{classes: [...],
total, ...}` envelope for `class list`, following the canonical `items`/`total`
paging envelope other list ops use. Both spill to disk past the token threshold
like every other read op.

**Exit codes:** standard read-command behavior — 0 on success; an unknown class
name to `bn class show <Name>` is a `BridgeError` (exit 2) with a message that
suggests `bn class list` (and `--all`) for discovery.

## 6. Edge cases

- **No C++ / no demangled names** — `class list` returns an empty list; not an
  error.
- **Namespace mistaken for a class** — mitigated by confidence (§4.3); a
  `name-only` cluster passed to `bn class show <Name>` still renders its methods
  but notes low confidence and absent vtable/RTTI.
- **Templates / nested classes / operator overloads** — handled by the
  depth-aware split (§4.2), which is the most heavily unit-tested unit.
- **Stripped RTTI** — vtable still readable; base classes degrade to
  best-effort/empty.
- **Thumb/ARM** — vtable slot reading inherits the existing reader's
  Thumb-normalization.
- **Oversized output** — spill envelope applies; `class show` of a class with a
  huge method set respects `--out` and the token threshold.
- **Ambiguous name (same class name in two namespaces)** — `class show` renders
  every matching record (each fully) rather than erroring, since the user cannot
  disambiguate without seeing them; the output names each match's fully-qualified
  form. `class list` shows each as its own row.

## 7. Testing

Mirrors the source layout; mocks the `binaryninja` module like the rest of the
suite (no BN license needed for unit tests).

- `tests/test_read_class.py` (bridge logic): the depth-aware split helper across
  templates / operators / ctor-dtor / nested / free-function cases; registry
  construction from a fake symbol+function set; vtable layout from a fake
  pointer table (header-word skipping, pure-virtual/thunk/`sub_*` marking); RTTI
  base decode for `__class` / `__si` / `__vmi` fakes; size-from-`operator new`
  with a fabricated construction site; confidence assignment.
- `tests/test_cli.py` additions: `bn class` / `bn class list` argparse wiring,
  text and JSON renderers, `--all` / `--query` filtering, unknown-class error
  and its discovery hint.
- `tests/test_formatters.py` (or the existing formatter tests): the text
  renderers against representative records.

**Dogfood (real BN, sanitized).** Validate against the firmware C++ stacks
already loaded as bridge instances (read-only mount). Per the parallel-agent
rule, pass `--instance`/`-t`
explicitly and never use `instance use` / `target use`. Confirm on a real class
that: methods cluster correctly, the vtable layout matches the Itanium
header-word offset, RTTI bases resolve, and `operator new` size is recovered.
**No real class/symbol names, addresses, or decompiled output go into any
committed artifact, issue, PR, or commit** — reproduce findings with the
invented names in this doc.

## 8. Out of scope / follow-ups

- Acting on the lens (auto-defining a struct from recovered layout, auto-naming
  vtable `sub_*` slots) — a natural next step, but a mutation, so separate.
- Caching the registry across requests for very large binaries.
- A `--by-class` flag on `bn function list` (the `class list` subcommand
  subsumes the discovery need #205 raised; revisit only if requested).

## 9. Implementation status (as shipped)

Recorded after implementation + a sanitized live dogfood against a real C++
target. All examples below use invented names.

**Fully working (validated live):**
- Method clustering + confidence (`class list`, `class show`).
- Vtable layout for vtables with absolute pointers (`.rodata`): slots resolved,
  `__cxa_pure_virtual` and unnamed `sub_*` slots marked.
- RTTI base classes (`__si`/`__vmi` decode + structural inference).
- Object size from a defined BN type (`source: bn_type`).
- Instances: construction sites (ctor inbound call sites) + globals storing the
  vtable.

**Dogfood-found fix (regression-tested):** BN demangles RTTI markers with
underscores and the vtable form carries a leading underscore (`_vtable_for_X`)
while typeinfo does not (`typeinfo_for_X`, `typeinfo_name_for_X`). The original
fixed marker list only matched the vtable form, so typeinfo — and therefore all
RTTI base classes — never resolved. Replaced with a regex matching both space
and underscore spellings.

**Best-effort gaps (honest, not fabricated; documented in `--help`/output):**
- **PIE vtables.** On position-independent targets, vtable function pointers
  live in `.data.rel.ro` and are zero in the static image (applied at load time
  via relocations BN does not surface here). Such a vtable resolves no slots;
  `class show` says so explicitly rather than rendering a class that looks like
  it has no virtuals.
- **`operator new` object size.** Recovering the allocation size that feeds a
  ctor's `this` needs MLIL def-use analysis that BN often does not expose at the
  call site. Not implemented; size falls back to a defined BN type when present,
  else `null` (never fabricated). `source: operator_new` is reserved for a
  follow-up.
- **Construction-site storage kind.** Sites are reported as `ctor-call` with the
  caller named; new/stack/global classification (and per-`new` size) is the same
  deferred MLIL-arg-recovery work as above.

These two MLIL-analysis gaps are the natural next follow-up (and pair well with
the §8 "acting on the lens" mutations).
