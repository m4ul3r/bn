---
name: bn
description: "Use the local bn CLI for Binary Ninja reversing through the bn bridge, in a live GUI session or headless mode. Triggers include decompilation, function search, xrefs, callsites and exact caller_static mapping, IL/disassembly, type recovery, struct field edits, previewed mutations, stable local IDs, batch apply, BNDB save/load, evidence helpers (vtable/pointer-table and .init_array walking, protobuf/RTTI type lensing, raw-ABI call evidence), and inline BN Python."
---

# bn

Use this skill for reverse-engineering work against a Binary Ninja database via the local `bn` CLI: lifecycle, small reads, command discovery, verified mutations, and every concurrent fan-out agent. In a single-agent OMP session, list-shaped or locally filtered reads can use **`bn-kernel`** so complete results remain in retained Python variables instead of entering the transcript. The bridge runs as a GUI plugin or a headless process; the CLI auto-spawns a headless instance on first use.

> **Route to a methodology first — this skill is the HOW; the methodology is the WHAT.**
> - **Open-ended understanding / mapping a binary** → invoke the **`bn-re`** skill first. For a *long* survey spanning many functions, **dispatch the `bn-re` subagent** instead of running inline — it keeps the decompiler/xref token-flood out of your context and returns a distilled map (recovered state persists in the BNDB, read it back via `bn`).
> - **Finding bugs / security audit / attack surface** → invoke the **`bn-vr`** skill first; likewise **dispatch the `bn-vr` subagent** for a long audit spanning many sinks.
> - **List-shaped / multi-function / grep-like reads in a single-agent OMP session** → invoke **`bn-kernel`** first. Concurrent sibling task agents currently share eval globals, so they must use this direct CLI with explicit `-i/-t` instead.
> - Then come back here for command syntax.

## Reference (load on demand)

The full command catalog lives in three files **in this skill's directory** — open the one for what you're doing (resolve against the base dir shown when this skill loads; e.g. `<skill-dir>/reference/reading.md`):

- **`reference/reading.md`** — every read command (decompile, il, disasm, xrefs, callsites, the `evidence` family, trace, proto, types/struct/class, strings, imports, sections, comments) + the JSON `.items[]` field map.
- **`reference/mutating.md`** — the preview→verify→save mutation surface (rename, proto set, locals, struct fields, comments, batch apply) + bundles.
- **`reference/runtime.md`** — sessions/headless (load/save, quick-load boundary), output/spill envelopes, `py exec`, `doctor`, known quirks, skill install.

## Target selection

One open target: omit `-t`. Multiple open: pass `-t <selector>` (from `bn target list`; matches selector / target_id / view_id / filename / basename), before *or* after the subcommand.

> **Parallel / fan-out agents — HARD rule.** Sticky pins (`instance use` / `target use`) are one shared file per git repo, so concurrent agents clobber each other. Fan-out agents **MUST** pass **`-i/--instance` and `-t/--target`** explicitly on **every** command and **MUST NOT** call `instance use` / `target use` / `*clear`. Prefer one dedicated instance per agent, then use the short flags everywhere:
>
> OMP sibling task agents also share one retained eval namespace. They **MUST NOT**
> keep `bn_kernel` sessions or bulk rows in module globals during fan-out; another
> sibling can silently overwrite them between cells. Use the direct CLI in this
> section until the harness provides per-subagent Python namespaces.
>
> ```bash
> bn session start /path/to/bin --instance-id dogfood-1   # spawn name (not global -i)
> bn -i dogfood-1 -t <sel> decompile main
> bn -i dogfood-1 -t <sel> xrefs main
> ```
>
> Prefer **`-i <id>`** (short form of `--instance`) to keep fan-out command lines short. Env `BN_INSTANCE` is optional single-agent convenience only — **not** for multi-agent fan-out (shared env is still clobberable). (Detail → `reference/runtime.md`.)

## Two gotchas that cause wrong answers

- **Spill / pipe-trap.** Spilling is **opt-in**: only `BN_SPILL_TOKENS=<tokens>` writes an oversized body to disk, and only then does stdout carry a small envelope *instead of* the data — so a piped `grep`/`jq`/`rg` matches the envelope rather than the answer, and a no-match misreads as "absent" (`bn decompile f | grep memcpy` finding nothing ≠ no memcpy). By default the full payload goes to stdout and the *consuming* agent/Harness truncates it (head + tail, whole stream readable as an artifact), so bound anything you intend to parse — `--lines` / `--limit` / `--out`. Either way, read `bn`'s stderr: a payload big enough to matter prints a `note:` naming this command's own slicing flag. (Envelope keys → `reference/runtime.md`.)
- **HLIL can mislead beyond access width — trust `bn disasm`.** Pseudo-C / HLIL distort more than the operand size. Before concluding on a bound / overflow / off-by-one, confirm the shape in `bn disasm`:
  - **access width** — a byte compare can render full-width, and a `zx.d` on a deref need not mean a 4-byte load (the classic case);
  - **conditional-compare / `csel` / `ccmn` guards** — AArch64 flattens `cmp x,#hi; ccmp x,#0,#N,le; b.eq` (effectively `if (x >= lo && x <= hi)`) into a misleading ternary like `z = x-1 <= 0xfe ? x == 1 : true; if (!z) …` that reads as flag-munging with no real bound;
  - **loop-invariant bound pointers** — a hoisted fixed limit (`add x8, base, #0x100` at entry) can render relative to the *moving* write pointer (`if (&p[0x100] <= &p[k+1]) break;`), so a real total-size cap looks like it never fires → phantom overflow;
  - **accumulator / shift structure** — a size-parse loop that should be `size = (size << 4) | nibble` but *lacks* the `<< 4` renders as a normal accumulate, concealing that it only ever keeps the last nibble.
  Each of these produced a near-false-positive *critical* finding in dogfooding; the pseudo-C alone is not enough for a bounds claim.
  **ARM32 predication / Thumb `IT` guards** need a separate check: ARM conditional instructions and Thumb `it`/`itt`/`ite` blocks are the ARM32 counterpart, not AArch64 `csel`/`ccmp`. Disassembly names the `IT` and each covered physical instruction on separate rows, including mixed 16/32-bit instructions. Native per-instruction text may omit the condition inherited from `IT`: read the preceding IT mask with those operands, and cross-check LLIL/MLIL predicate and flag flow before calling a move/load/store unconditional. A window starting inside an IT block lacks that context; include the preceding IT. HLIL may express the resulting select as a useful ternary, but still verify its predicate and operands.

## Command index (what exists — flags live in reference)

- **Read** — `target info/list`, `function list/search/info/cfg/structured-il`, `decompile`, `il`, `disasm`, `xrefs`, `callsites`, `evidence function/xrefs/table/message/init/calls/orient/surface/virtual-call`, `trace`, `dataflow defuse/callgraph/values`, `taint forward/backward/models`, `go functions`, `proto get`, `local list`, `data vars/symbols`, `read`, `types [show]`, `struct show`, `class list/show`, `strings`, `imports`, `exports`, `sections`, `tag list/get/types`, `comment list/get` → **`reference/reading.md`**
- **Mutate** (verified; preview first) — `symbol rename`, `proto set`, `local rename/retype`, `comment set/delete`, `struct field set/rename/delete`, `types declare`, `data retype`, `function create`, `tag add/remove`, `tag type create/remove`, `go rename`, `batch apply`, `bundle function` → **`reference/mutating.md`**
- **Discover** — `capabilities` (no target; the registry-derived JSON index of every command + when to pick it — prefer it over guessing from this partial map) → **`reference/runtime.md`**
- **Session** — `load`, `save`, `close`, `refresh`, `session start/list/stop/restart/status`, `instance list/find/gc`, `target close` → **`reference/runtime.md`**
- **Tooling** (host-side, not analysis surface) — `doctor`, `help`, `plugin install`, `skill install` → **`reference/runtime.md`**
- **Escape hatch** — `py exec` (only when a built-in won't do; exclusive write lock) → **`reference/runtime.md`**

## Mutation safety (the loop)

Every mutation runs **preview → live-verify → readback → save**: `--preview` applies → diffs → reverts; live writes return `verified` / `noop` / `unsupported` / `verification_failed` (a failed batch fully reverts); read back with `proto get` / `struct show` / `decompile`; then **`bn save`** before close — annotations live in the `.bndb`. Full detail (statuses, `local_id`, batch manifest) → **`reference/mutating.md`**.

