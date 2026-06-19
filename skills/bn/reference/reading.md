# bn reference — reading

Read-command catalog for the `bn` skill. Open when surveying/decompiling. See `../SKILL.md` for the map.
## 4. Read flow

```bash
bn target info
bn function list [--count] [--min-address 0x401000 --max-address 0x40ffff]
bn function search attachment [--count]
bn function search --regex 'attach|detach|follow'
bn function info <fn> [--verbose]
bn decompile <fn> [--addresses] [--lines 40:80] [--force-analysis]
bn il <fn> [--view {hlil|mlil|llil}] [--ssa]
bn disasm <fn>
bn xrefs <fn-or-addr> [--limit 20]
bn xrefs --field <Struct.field>
bn callsites <callee> --within <fn>
bn callsites <callee> --within-file <path>
bn evidence function <fn> [--context 2]              # per-call ABI args (LLIL/MLIL/HLIL + raw disasm) + thunk detection
bn evidence xrefs <fn-or-addr> [--limit N]           # inbound refs annotated with section/segment/symbol/disasm
bn evidence table <addr> [--entries N] [--stride N]  # interpret memory as a pointer/vtable table (Thumb-normalized)
bn evidence message <type-string> [--limit N]        # protobuf/RTTI type-name -> xrefs -> nearby metadata table windows
bn evidence init [--limit N]                         # .init_array/.ctors constructor-pointer summary
bn trace <fn> <addr> [--arg N] [--interprocedural]   # backward SSA slice: trace where a call argument originates
bn proto get <fn>
bn local list <fn>
bn read 0x... --length N [--encoding {hex|bytes}]   # address is positional; --address 0x... is an accepted alias
bn function create <address> [--preview]
bn types [--query <q>]
bn types show <name>
bn struct show <name>
bn class list [--all] [--no-stl] [--query <substr>]   # C++ classes from demangled symbols + RTTI
bn class show <Name>                                  # one class: methods, vtable, size, bases, instances
bn strings [--query <q>] [--regex] [--min-length 5] [--section .rodata] [--no-crt]
bn imports
bn sections [--query <q>]
bn comment list [--query <q>]
bn comment get   --address 0x... | --function <fn>
```

Notes:

- `bn function search` is case-insensitive substring; add `--regex` for regular expressions. `function list` and `function search` both accept `--min-address` / `--max-address`. Both also accept `--count`, which returns just the total (`function list --count` = whole-binary function count for fast sizing; `function search <q> --count` = number of matches for a query) instead of the listing.
- `bn xrefs` accepts a function name *or* a hex/decimal address. Text groups refs by caller (`code refs: 12 sites across 4 functions`); JSON adds `caller_function: {address, name}` so an `xrefs → --within-file` pipeline survives duplicate symbol names. Use `bn xrefs` for inbound references; reach for `bn callsites` when you need exact return-address recovery and local context.
- `bn decompile` renders Binary Ninja's **Pseudo C** (the same text the GUI shows), comments inline. It omits the address gutter by default — add `--addresses` when you need it (e.g. for `bn comment set --address`). For the underlying IL instead, use `bn il --view {hlil|mlil|llil}`.
- **Skipped (oversize) functions.** Binary Ninja skips analysis on functions that exceed its size/time limits and renders only a stub ("…taking too long to analyze… Loading…"). `bn decompile` detects this (`func.analysis_skipped`) and appends a `warning:` (and sets `analysis_skipped: true` in JSON) so you never mistake a stub for a real body. Pass `--force-analysis` to override the skip and reanalyze just that function before decompiling — it returns the full body and sets `analysis_forced: true`. It can be slow on very large functions and takes the **write lock** (it mutates analysis state), so avoid it on a bridge other agents are actively reading.
- **Width-sensitive reads — trust `bn disasm`, not the decompiler.** Pseudo C and HLIL share the same analysis and can hide the real access width. A byte compare (`cmp al, ...`) renders as a full-width equality, and a `zx.d` on a dereference does **not** imply a 4-byte load (it can be a 1-byte load zero-extended to 4). When the exact comparison width or memory-read size matters (off-by-one, OOB, signedness bugs), treat `bn disasm <fn>` as authoritative and confirm the operand size there before reasoning about the bug.
- `bn class` is the **C++ object-model lens** (#205): a correlation layer over data BN already recovers (demangled symbols, RTTI data symbols, vtables). `bn class list` clusters the binary's functions by demangled class and shows method count, vtable presence, and a confidence tag (`rtti` = has vtable/typeinfo, `ctor` = has a ctor/dtor, `name-only` = could be a namespace); it defaults to `rtti`+`ctor`-confirmed classes, with `--all` to include `name-only` clusters, `--query` to filter, and `--no-stl` to fold out standard-library / ABI-runtime classes (`std::`, `__gnu_cxx::`, `__cxxabiv1::`, reserved-id internals) so domain classes surface (it reports how many it hid). `bn class show <Name>` resolves one class (exact, or by leaf name across namespaces — ambiguous names list every match) and reports: methods grouped ctor/dtor + non-virtual members; the vtable layout (Itanium header-skipped, `__cxa_pure_virtual`/unnamed-`sub_*` slots marked); object size (from a defined BN type when present); RTTI base classes (Itanium `__si`/`__vmi` decode); and instances (ctor construction sites from code xrefs + globals that store the vtable). Best-effort gaps to know: vtables in `.data.rel.ro` on PIE targets read as empty (pointers applied at load time via relocations — the output says so), `operator new` object-size and per-site new/stack/global classification are not yet recovered (size falls back to a defined type; sites are reported as `ctor-call`). `--format json` carries the full record.
- `bn imports` JSON tags each entry with `kind` (`function`, `data`, `address`) and includes `library` + `raw_name`. Text marks data/address imports with `(data)` / `(address)`.
- `bn sections` exposes start/end, length, semantics, and segment-derived `r/w/x` permission flags. `--query` matches a substring of the section **name or its semantics label** (case-insensitive), so `--query code` finds executable sections (`.text` = `ReadOnlyCode`) and `--query data` finds the data sections, not just literal name matches.
- `bn strings`: `--no-crt` is a heuristic — drops single-character repetitions and strings sitting in `.text`. Combine with `--min-length` and `--section`. Add `--regex` to treat `--query` as a case-insensitive regex (use it for OR patterns like `'%s|%n|/bin/'` — plain `--query` is literal substring, so `\|` does **not** mean OR). A `--query` value that looks like a flag (e.g. `-h`) is preserved as the query, but a known sibling flag right after `--query` (e.g. `--regex`) still errors — put flags before `--query` or use `--query=<value>`. JSON is a paged `{items, total, ...}` envelope where each item is `{address, length, chars, type, value}` — the string text is in **`value`**, not `string`. Extract it with `bn strings --query foo --format json | jq -r '.items[].value'`. (A wall of `null` from `jq '.items[].string'` means the wrong field name, not "no match" — a true no-match returns an empty `items` array.)
- **JSON envelope contract (#275).** Every collection-returning read emits ONE shape:
  `{"kind": <discriminator>, "items": [...], "total": N, "offset": 0, "limit": L, "returned": R, "has_more": bool}`.
  `kind` names what the items are; `items` is **always** the data container (no per-command alias — the old `functions`/`classes`/`code_refs`/`data_refs` keys were dropped). `--count` mode returns `{"kind": <same>, "count": N, "total": N}` (no `items`). `kind` values: `functions` (list+search), `strings`, `imports`, `exports`, `sections`, `types`, `xrefs`, `field_xrefs` (`xrefs --field`), `symbol_presence` (`xrefs --any`), `callsites`, `classes`, `comments`, `pointer_table` (`evidence table`), `init_arrays` (`evidence init`), `messages` (`evidence message`), `imports_summary` (`imports --summary`, a keyed aggregate that keeps `namespaces`/`by_kind`, not a flat `items` list). Grouped reads keep their nesting inside `items` (e.g. `evidence init` items are sections, each retaining `entries`; `evidence message` items each retain `xrefs`).
  - **Nothing-found vs incomplete (don't confuse them):** `items: []` + `total: 0` means *clean — nothing found*. `has_more: true` means *more pages exist* (paging). `truncated: true` / a non-empty `assumptions` array (taint/trace/`evidence message`) means *the analysis hit a cap and is incomplete* — NOT an all-clear. An empty taint result with assumptions is "couldn't fully follow it," not "safe."
- **JSON list-command field map (the `.items[]` idiom and its exceptions).** Most list/search read commands share one paged envelope — `{kind, items, total, offset, limit, returned, has_more}` — so `jq '.items[]...'` is the near-universal extractor (`strings`, `imports`, `exports`, `sections`, `types`, `function list`/`search`, `callsites`, `xrefs`, `comment list`, `class list`). The catch: the *leaf* field holding the payload is **not always named after the command**, and selecting a wrong/missing key yields `null` (a wall of which reads like "no results"), never an error. Per-command leaves:
  - `strings` → `.items[].value` (the string text — **not** `.string`)
  - `types` → `.items[].decl` (the C declaration text) + `.items[].layout`, alongside `.name`/`.kind` (**not** `.definition`/`.type`)
  - `comment list` → `.items[].comment` (**not** `.text`); each item also carries `.address` and `.function`
  - `callsites` → the payload is **nested**, with no flat `.name`/`.address`: use `.items[].caller_static`, `.items[].callee.name`, `.items[].containing_function.name`
  - `function list`/`search`, `imports`, `exports` → `.items[].name` + `.items[].address` (plus `raw_name`/`display_name`, and `kind` on imports/exports)
  - `class list` → `.items[].name`, `.method_count`, `.has_vtable`, `.size`, `.bases`, `.confidence`
  - `sections` → `.items[].name` + `.start`/`.end` (hex strings) + `.length`, `.semantics` (e.g. `ReadOnlyCode`), and segment-derived `.readable`/`.writable`/`.executable` (present **only when the section has a backing segment** — absent, i.e. `null` under `jq`, otherwise)
  - `xrefs` → `.items[]`, each row carrying `.kind` (`code` | `data`), `.function` (containing-function name), and `.caller_function.{address,name}`. Filter by kind with `jq '.items[] | select(.kind=="code")'`; full-set totals are `.code_ref_count` / `.data_ref_count`. (Pre-#184 the op exposed top-level `code_refs`/`data_refs` arrays; those were dropped — `jq '.code_refs[]'` now returns empty and misreads as "no refs". The dual `code_refs`/`data_refs` shape survives only in `function info` / evidence embedding, not the `xrefs` command.)
  - **Exception — `local list` does NOT use `items`.** Its JSON is `{function, locals}`; extract with `jq '.locals[]'`. Each entry carries `.name`, `.type`, `.storage`, `.index`, and **`.local_id`** — a stable handle; prefer it over names, which drift across re-analysis.
- `bn evidence ...` is a read-locked family that surfaces the **raw material** behind a call/table/type so you don't have to hand-roll `py exec` — built for stripped C++/firmware. `evidence function <fn>` pairs each call's raw disasm with LLIL/MLIL/HLIL argument evidence and flags thunks (including PLT/import trampolines); `evidence xrefs` annotates each ref with section/segment/symbol + the referencing disassembly; `evidence table <addr>` reads a pointer/vtable table, is **ARM/Thumb-aware** (clears the T bit and resolves the even entry, marked `[thumb-adjusted]`), and tags each slot with `status` (`function`/`mapped`/`null`/`unmapped`) + `plausible` plus low-confidence `warnings` when the address doesn't look like a table; `evidence message <type-string>` is a protobuf/RTTI lens (type-name → xrefs → nearby metadata windows, e.g. the serializer slot); `evidence init` summarizes `.init_array`/`.ctors` constructor pointers. Large results spill to disk like other read ops.
- `bn trace <fn> <addr> [--arg N] [--interprocedural]` walks **MLIL SSA use-def chains backward** from a call-site argument to trace where it originates. It shows each intermediate SSA variable with its defining instruction and address, terminating at a function parameter, global, memory load, or call boundary. Add `--interprocedural` (with `--ip-depth N`, default 2) to follow **return values** across call boundaries into callee functions when the callee has a real MLIL body — this works best for self-contained code (static binaries, kernel modules). IP mode follows a value into a callee only when it is that callee's return value, and stops at the callee's own parameters (it does not map callee params back to caller args, nor walk up the caller chain — step up with `bn xrefs` + re-run for those). For dynamically-linked PLT/import calls, `--interprocedural` correctly falls through (no callee body to enter), producing identical output to intraprocedural mode. JSON output includes `interprocedural` and `ip_depth` fields. Example:
  ```bash
  bn trace main 0x27e1a --arg 1                              # intra: stops at call boundary
  bn trace main 0x27e1a --arg 1 --interprocedural            # IP: follows into callee if possible
  bn trace handle_l2cap_con_req 0x1c2bc --arg 2 --format json # structured JSON output
  ```
- `bn read --address <addr> --length <N>` returns raw bytes from the mapped view — the parallel-safe alternative to a `py exec` `bv.read(...)` (it runs under the shared read lock). Text mode prints an offset/hex/ASCII hexdump; JSON returns `{address, length, hex, ascii}`. A read that runs past mapped memory comes back **short** with `short_read: true`, `requested_length`, and a `note` (it does **not** error). An entirely unmapped address *does* error (`Address 0x... is not mapped`). Add `--encoding bytes` to stream the raw bytes to stdout (or to `--out <path>`) instead of a hexdump — useful for piping a blob into another tool or saving it to disk. Note that `--encoding bytes` emits only the raw stream, so the `short_read`/`note` marker is **not** visible in that mode — use the default hex mode (or JSON) when you need to know whether a read ran short.
- `bn function create <address> [--preview]` forces Binary Ninja to create and analyze a function at `<address>` when auto-analysis missed it. **When to reach for it:** a code entry point that BN never disassembled because it is reachable only through a **data pointer** (a vtable slot, a function-pointer table, a handler/dispatch array) and has no direct `call`. Point `bn read` / `bn xrefs` at the pointer table to recover the target address, then `bn function create` it so `decompile` / `xrefs` / `callsites` work on it. It is a verified mutation, so it honors the same `--preview` (create → verify → revert) and live-verify loop as the other mutations: it returns `noop` if a function already starts there, errors if the address is unmapped or not inside an executable segment, and rolls back with `verification_failed` if analysis produces no function. Save afterward (§6 step 4) to persist the new function.

## 5. Caller-static mapping

Prefer `bn callsites` over ad-hoc `py exec` whenever the task is "find the exact native return-address callers" or any direct-call mapping workflow.

`bn callsites` reports:

- `call_addr` — the native `call ...` instruction address.
- `caller_static` — the post-call return address (`call_addr + instruction_length`).
- `call_index` within the containing function, `within_query`, previous/next instructions, a local-or-null `hlil_statement`, and a best-effort `pre_branch_condition`.

```bash
bn callsites crt_rand --within bonus_pick_random_type --caller-static
bn callsites crt_rand --within fx_queue_add_random   --caller-static
bn callsites crt_rand --within-file /tmp/rng-functions.txt --format json
```

`--within-file` accepts one identifier (name or hex address) per non-empty line; lines beginning with `#` are ignored.

`hlil_statement` is intentionally local-or-null — when Binary Ninja only exposes a coarse enclosing region, expect `null` instead of a noisy whole-function blob. `pre_branch_condition` is the nearest enclosing pre-call HLIL condition when it can be recovered confidently; `null` is normal.

If you call `bn callsites <callee>` without `--within` / `--within-file`, the CLI prints a 3-option help block (`single caller`, `many callers`, `list callers`) instead of erroring.

