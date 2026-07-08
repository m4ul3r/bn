---
name: bn-re
description: "Reverse-engineering methodology for unknown binaries via the bn CLI. Highest-value moves over raw decompilation — a C++ class-lens triage that maps the type lattice (public API vs implementation, vtables, inheritance) before reading code, and a conditional hidden-surface sweep that recovers .init_array constructors, dispatch tables, and RTTI handlers only when the target's signatures warrant it (and deliberately skips them when they don't). Also covers function triage, iterative type/struct recovery, call-graph mapping, and naming."
---

# bn-re — Reverse Engineering Methodology

Use this skill when the user wants to understand, reverse engineer, or analyze a binary. This is a methodology guide — it tells you *what to do and why*. For command syntax, see the `bn` skill.

## Approaching an Unknown Binary

Start broad, then narrow:

> **One-shot triage (steps 1–3 in a single consistent read).** `bn evidence orient` composes target + analysis state, imports summary, a high-signal strings sample, function count, and sections into **one digest under a single read lock** — so nothing interleaves between the sub-reads the way separate commands would. Run it as the very first move on an unknown binary, then drill with the steps below. It samples higher-signal strings (min 6 chars) than `bn strings`, so its strings `total` is intentionally smaller.

1. **Orient** — get architecture, platform, and entry point:
   ```bash
   bn target info
   ```
   > **Cache restore ≠ clean slate.** If you loaded the target and it came back with a `note: restored cached database …`, the view was auto-restored from a prior `bn save` (a sibling `.bndb`, or the global `~/.cache/bn/bndb/<stem>.<hash>.bndb` cache used on read-only mounts) and **already carries earlier renames/comments** — so a multi-MB binary that loaded in seconds is a cache hit, not a cold analysis, and the names you see may be a previous run's. Before claiming a from-scratch recovery baseline, re-load with `--no-bndb` for pristine bytes. See "Global BNDB cache" in `reference/runtime.md`.

2. **Survey imports and strings** — these reveal libraries, APIs, and embedded literals that hint at functionality:
   ```bash
   bn imports
   bn strings
   ```
   Imports tell you what the binary *does* (network I/O, file ops, crypto, GUI). Strings reveal configuration keys, error messages, format strings, and embedded paths. On a symbol-rich C++ target, scope with `bn strings --section .rodata` — the unfiltered dump scans `.dynstr` and buries the real literals under mangled `_Z...` symbol names (`bn strings` prints a tip pointing this out when the dump is symbol-noisy).

3. **Scan the function list** — get a sense of scope:
   ```bash
   bn function list
   ```
   Note the total count, address range, and whether symbols are stripped. A stripped binary with 2000 functions requires different tactics than a symbolicated one with 50. Beware: `file(1)` says "stripped" whenever `.symtab` is gone, but a binary that kept its `.dynsym` (any shared object, and `-rdynamic`/default-visibility executables) still has real names for most functions — the tell is `bn function list` being *mostly named* (few `sub_XXXX`) and a non-empty `bn imports`, not `file`. Only a target with an empty import table **and** overwhelmingly `sub_XXXX` names is the truly-stripped case. For *vulnerability* work on that truly-stripped static target, see the "Stripped / static lane" in `bn-vr`, which inverts the import-first workflow (strings → string-xref → behavioral sink recovery).

4. **Map the C++ type lattice (RTTI / symbolicated C++ targets)** — when the symbols show C++ (mangled `_Z…` names, RTTI), lead with the class lens instead of grepping symbols by hand:
   ```bash
   bn class list --no-stl          # domain classes, folding std/ABI noise
   bn class show <ClassName>        # one class: methods, vtable slots, bases, construction sites
   ```
   `class list` clusters functions by class and separates the public API surface from the implementation/engine classes at a glance; `class show` resolves a class's inheritance, vtable layout, and where it is constructed. On a rich C++ binary this is the fastest orientation there is — start here, then drill with the sections below. (Plain-C / stripped targets have no classes to show — skip it.)

> **Quick-loaded target?** If the binary was opened with `bn load --quick` / `bn session start --quick` (fast, no analysis), `bn imports` and `bn sections` work, but `bn strings` **errors** until `bn refresh` (it refuses rather than return nothing) and `bn function list` is **partial** (only entry-point + symbol functions) — steps 2–3 would otherwise read as "no strings, almost no functions" and mislead the survey. Check `analysis_state` in `bn target info` (`"quick"` vs `"full"`) and `bn refresh` before surveying. See "Quick Load" in the `bn` skill (now `reference/runtime.md`).

## Function Triage

> **Pipe trap:** large `bn` read output (decompile, `function list`, etc.) spills to disk and stdout carries only an envelope. Piping that into `grep`/`jq`/`awk`/`c++filt` makes the filter see the envelope, **not** the data — a no-match then misreads as "absent" (e.g. concluding a name is mangled because `| grep _Z` matched nothing). Write to a file first and process it: `bn function list --out /tmp/fns.json && jq '.items|length' /tmp/fns.json`, or slice with `--limit`/`--lines` so it doesn't spill.

Not all functions matter equally. Prioritize:

- **Entry point and exports** — start with what the OS calls. `bn target info` gives the entry point; `bn function search main` or `bn function search start` may find the real main.
- **Large functions** — complex logic concentrates in big functions. Sort by size or instruction count.
- **High xref count** — functions called from many places are utilities or core abstractions:
  ```bash
  bn xrefs <function_name>
  ```
  Many inbound xrefs = widely used. Few xrefs + large body = likely a top-level handler.
- **String references** — functions containing interesting strings (error messages, protocol keywords, file paths) are high-value targets:
  ```bash
  bn strings --regex --query "error|fail|password|key|flag"
  ```
  Then use `bn xrefs` on the string address to find which functions reference it.
- **Import callers** — trace backward from interesting imports:
  ```bash
  bn xrefs malloc
  bn callsites recv --within <function>
  ```

## Hidden Code Surfaces

Binary Ninja's auto-analysis follows direct calls. Two important categories of code don't sit on that graph and will be invisible until you go looking for them.

> **One-shot sweep: `bn evidence surface`.** It composes this whole section — `.init_array`/`.ctors` constructors, candidate vtable/dispatch tables (runs of pointers-into-executable across `.data`/`.rodata`/`.data.rel.ro`), and the **missing-function candidates** (executable, data-referenced addresses with no BN function, each tagged `plausible`) — into one read-locked digest. On a fully-analyzed binary it's mostly `missing=0` (BN already found everything); its value is on stripped/optimized firmware, where it hands you the code BN's linear sweep didn't reach. It's **read-only** — confirm a candidate with `bn disasm <addr>` before `bn function create`. Use it to decide whether the manual steps below are even warranted, then drill in with them.

### Pre-main code (`.init_array`, constructors)

Functions tagged with `__attribute__((constructor))`, C++ static initializers, and any code the linker registers in `.init_array` run *before* `main`. They commonly stage globals, derive keys, or wire up dispatch tables — exactly the kind of setup that breaks an analysis built only from `main`'s call graph.

To find them:

1. `bn evidence init` finds every constructor/destructor section (`.init_array`, `.ctors`, `.fini_array`, …), walks each one, and resolves every slot to its function. It's **arch-aware** (uses the view's pointer size + endianness) and **ARM/Thumb-aware** (clears the T bit so an odd `0x…1` pointer resolves to the even function entry, marked `[thumb-adjusted]`) — don't hand-roll `bv.read` + `struct.unpack('<…Q')`, which hardcodes 8-byte little-endian and is silently wrong off x86-64. See the command in the bn skill (`reference/reading.md`).
2. Skip the toolchain stub `frame_dummy` — it's the first slot on most GCC builds and rarely interesting.
3. Decompile each remaining entry. Anything that writes to BSS / `.data` / `.bss` is staging state for `main` to read; rename it `stage1_<purpose>` (or similar) so the relationship is visible from later analysis.

ELF entry-flow review when nothing in `main` makes sense: `entry_point → __libc_start_main → main`, *but* `_start` and `__libc_start_main` invoke the `.init_array` callbacks before `main`. If a global "appears from nowhere" in `main`, the producer is almost certainly an `.init_array` entry.

### Data-only function references

If the binary has a dispatch table (an array of function pointers — common in VMs, FSAs, vtables, callback registries), Binary Ninja often *won't* identify the targets as functions because there's no direct `call` to them, only a data reference from the table.

Symptoms: `bn decompile <addr>` errors with `Function not found`; the bytes at `<addr>` look like a function prologue (`endbr64`, `push rbp`, `push {…,lr}` / `stp x29,x30` on ARM) on disasm, but it's marked as data.

Recover and create the targets:

1. `bn evidence table <table-addr> --entries N` reads the dispatch/vtable table and resolves each slot to a function — Thumb-normalized, with a `status`/`plausible` tag per entry and a warning when the address doesn't look like a table (so you can tell a real table from misread code). Slots that come back `status: mapped`/`unmapped` with no function are the ones BN missed.
2. `bn function create <target> --preview` creates and verifies a function at each missing slot (a previewed, revertible mutation — see the bn skill — `reference/reading.md` and `reference/mutating.md`). Save afterward to persist it.

After that, the normal `bn decompile` / `bn xrefs` flow works on the new function.

When this comes up most: VM opcode handler tables, FSA predicate tables, COM-style vtables, plugin registries. If you've recovered a struct of `(tag, fn_ptr)` rows and one of the `fn_ptr` targets is missing, this is almost always why.

### Stripped C++ / generated code (RTTI, vtables, protobuf)

Stripped C++ firmware leaks structure through RTTI type-name strings and vtables even when symbols are gone. To turn a type name into code:

- **First, if the binary still has demangled C++ symbols, use the class lens.** `bn class list --no-stl` clusters the recovered classes and `bn class show <Name>` gives one class's vtable + bases + construction sites — it correlates the symbols/RTTI/vtables BN already recovered, so reach for it before the lower-level evidence helpers below (which you still need on a *fully stripped* target where the class lens has no symbols to cluster).
- `bn evidence message <TypeName>` locates the type-name string (e.g. a mangled `N…E` typeinfo name or a `pkg.Message` proto string), lists its xrefs, and dumps the nearby metadata windows — the typeinfo table and the serializer/handler pointer slots sitting next to it. This is how you get from "I see the string `common.HeadUnitInfo`" to "its serializer is `sub_…`" without reading raw bytes by hand.
- `bn evidence table <vtable-addr>` lists a class vtable's methods (Thumb-normalized), so you can tell construction from dispatch from generated boilerplate.
- `bn evidence function <fn>` flags thunks (a `j_*` veneer or PLT/import trampoline → its target) and, for each call, shows the raw ABI argument evidence beside the pseudo-C — including the vtable offset for an indirect/virtual call (`(*(*this + 0xNN))(...)`). Reach for it when the decompiler's argument story is incomplete and you'd otherwise drop to MLIL/disassembly.

## Iterative Type Recovery

Type recovery is incremental. Don't try to get everything right at once.

### Phase 1: Rename functions
Start with the easiest wins — rename functions whose purpose is obvious from strings, imports, or call patterns:
```bash
bn symbol rename sub_401000 parse_config --preview
```
Always preview first. Renaming propagates through decompilation and makes surrounding code easier to read.

### Phase 2: Retype locals and parameters
Once a function's purpose is clear, fix the prototype and local types:
```bash
bn proto get parse_config
bn proto set parse_config "int32_t parse_config(char* buf, int32_t len)" --preview
bn local list parse_config
bn local retype parse_config arg1 "char*"
```
Correct prototypes propagate to all callers.

### Phase 3: Struct reconstruction
When you see repeated field accesses at fixed offsets from a pointer, that pointer is a struct. See the **Struct Reconstruction** section below.

### Batch mutations
When you have multiple renames or retypes queued up, use `bn batch apply` with a manifest instead of individual commands. This is faster and atomic. Pipe the manifest on stdin with a quoted heredoc — no temp file, and free-text comments need no escaping:

```bash
bn batch apply -t <selector> - <<'BN_EOF'
{"ops": [
  {"op": "rename_symbol", "identifier": "sub_401000", "new_name": "parse_header"},
  {"op": "set_comment", "address": "0x401040", "comment": "len isn't bounds-checked"}
]}
BN_EOF
```

Pass the target with `-t <selector>` — do **not** put `"target": "active"` in the manifest, since `active` does not resolve under multi-target headless. A file path is also accepted (`bn batch apply /tmp/manifest.json`); `--preview` (before `-`) diffs without committing.

## Call Graph Analysis

Understanding relationships between functions reveals architecture:

- **Trace callees** — what does a function depend on?
  ```bash
  bn decompile <function>
  ```
  Read the decompilation and note every function call.

- **Trace callers** — who calls this function?
  ```bash
  bn xrefs <function>
  ```

- **Detailed call context** — when you need to understand *how* a function is called (what arguments, under what conditions):
  ```bash
  bn callsites <callee> --within <caller>
  ```
  This gives you the exact call site with surrounding HLIL context.

- **Trace argument origins** — when you need to know where a specific call argument comes from (function parameter, global, heap allocation, previous call return):
  ```bash
  bn trace <caller> <call_address> --arg N
  bn trace <caller> <call_address> --arg N --interprocedural
  ```
  Each step shows the SSA variable, its defining instruction, and whether the chain terminates at a function parameter, memory load, or call boundary. Add `--interprocedural` to follow return values across internal call boundaries (works best on static/kernel binaries).

- **Build a mental call tree** — for key functions, trace both up and down 2-3 layers. This reveals the flow: entry -> dispatch -> handler -> utility.

## Commenting

Comment the *why* a name can't carry (assumptions, edge cases, cross-function relationships), and drop `TODO:` markers for deferred work so later passes resume via `bn comment list --query TODO`. If it fits in a name, rename instead.
```bash
bn comment set --address 0x401000 "len isn't bounds-checked; attacker-controlled"
bn comment set --address 0x402000 "TODO: arg2 looks like a callback — confirm signature"
```

## Struct Reconstruction

Repeated fixed-offset accesses off a pointer (`*(arg1 + 0x10)`, `*(arg1 + 0x18)`) mean it's a struct. Collect the offsets from decompilation, check for an existing type (`bn struct show <T>`), set fields at the observed offsets, retype the param, and re-decompile — iterate until it reads naturally. Complex structs: `bn types declare` or `bn py exec` + `StructureBuilder` (see the `bn` skill).
```bash
bn struct field set Player 0x10 health int32_t --preview
bn local retype <function> arg1 "Player*"
```
