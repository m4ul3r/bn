---
name: bn
description: Use the local bn CLI for Binary Ninja reversing through the bn bridge, in a live GUI session or headless mode. Triggers include decompilation, function search, xrefs, callsites and exact caller_static mapping, IL/disassembly, type recovery, struct field edits, previewed mutations, stable local IDs, batch apply, BNDB save/load, evidence helpers (vtable/pointer-table and .init_array walking, protobuf/RTTI type lensing, raw-ABI call evidence), and inline BN Python.
---

# bn

Use this skill for reverse-engineering work against a Binary Ninja database via the local `bn` CLI. The bridge runs as a GUI plugin or a headless process; the CLI auto-spawns a headless instance on first use.

> **Route to a methodology first — this skill is the HOW; the methodology is the WHAT.**
> - **Open-ended understanding / mapping a binary** → invoke the **`bn-re`** skill first. For a *long* survey spanning many functions, **dispatch the `bn-re` subagent** instead of running inline — it keeps the decompiler/xref token-flood out of your context and returns a distilled map (recovered state persists in the BNDB, read it back via `bn`).
> - **Finding bugs / security audit / attack surface** → invoke the **`bn-vr`** skill first; likewise **dispatch the `bn-vr` subagent** for a long audit spanning many sinks.
> - Then come back here for command syntax.

## Reference (load on demand)

The full command catalog lives in three files **in this skill's directory** — open the one for what you're doing (resolve against the base dir shown when this skill loads; e.g. `<skill-dir>/reference/reading.md`):

- **`reference/reading.md`** — every read command (decompile, il, disasm, xrefs, callsites, the `evidence` family, trace, proto, types/struct/class, strings, imports, sections, comments) + the JSON `.items[]` field map.
- **`reference/mutating.md`** — the preview→verify→save mutation surface (rename, proto set, locals, struct fields, comments, batch apply) + bundles.
- **`reference/runtime.md`** — sessions/headless (load/save, quick-load boundary), output/spill envelopes, `py exec`, `doctor`, known quirks, skill install.

## Target selection

One open target: omit `-t`. Multiple open: pass `-t <selector>` (from `bn target list`; matches selector / target_id / view_id / filename / basename), before *or* after the subcommand.

> **Parallel / fan-out agents — HARD rule.** Sticky pins (`instance use` / `target use`) are one shared file per git repo, so concurrent agents clobber each other. Fan-out agents **MUST** pass `--instance` and `-t` explicitly on every command and **MUST NOT** call `instance use` / `target use` / `*clear`. Prefer one dedicated instance per agent: `bn session start <bin> --instance-id <id>`, then thread `--instance <id>` everywhere. (Detail → `reference/runtime.md`.)

## Two gotchas that cause wrong answers

- **Spill / pipe-trap.** Output over ~10k tokens spills to disk; stdout carries only a small envelope. A piped `grep`/`jq`/`rg` then reads the *envelope*, not the data — a no-match misreads as "absent" (`bn decompile f | grep memcpy` finding nothing ≠ no memcpy). Write to a file first (`bn decompile f --out /tmp/f.txt && grep memcpy /tmp/f.txt`) or slice with `--lines` / `--limit`. (Envelope keys → `reference/runtime.md`.)
- **Width-sensitive reads — trust `bn disasm`.** Pseudo-C / HLIL hide the real access width: a byte compare can render full-width, and a `zx.d` on a deref need not mean a 4-byte load. For off-by-one / OOB / signedness, confirm the operand size in `bn disasm` before concluding.

## Command index (what exists — flags live in reference)

- **Read** — `target info/list`, `function list/search/info`, `decompile`, `il`, `disasm`, `xrefs`, `callsites`, `evidence function/xrefs/table/message/init`, `trace`, `proto get`, `local list`, `read`, `types [show]`, `struct show`, `class list/show`, `strings`, `imports`, `sections`, `comment list/get` → **`reference/reading.md`**
- **Mutate** (verified; preview first) — `symbol rename`, `proto set`, `local rename/retype`, `comment set/delete`, `struct field set/rename/delete`, `types declare`, `function create`, `batch apply`, `bundle` → **`reference/mutating.md`**
- **Session** — `load`, `save`, `close`, `refresh`, `session start/list/stop` → **`reference/runtime.md`**
- **Escape hatch** — `py exec` (only when a built-in won't do; exclusive write lock) → **`reference/runtime.md`**

## Mutation safety (the loop)

Every mutation runs **preview → live-verify → readback → save**: `--preview` applies → diffs → reverts; live writes return `verified` / `noop` / `unsupported` / `verification_failed` (a failed batch fully reverts); read back with `proto get` / `struct show` / `decompile`; then **`bn save`** before close — annotations live in the `.bndb`. Full detail (statuses, `local_id`, batch manifest) → **`reference/mutating.md`**.

