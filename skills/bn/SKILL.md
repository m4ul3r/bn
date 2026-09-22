---
name: bn
description: "Use the local bn CLI and Binary Ninja bridge for target selection, reads, verified mutations, and session lifecycle. Use bn-re for open-ended reversing, bn-vr for vulnerability research, and bn-kernel for large retained-Python reads when that runtime is available."
---

# bn

Use `bn` for Binary Ninja reads and writes through the bridge. Choose a methodology before a long investigation: `bn-re` for understanding a binary and `bn-vr` for finding security bugs. In an OMP session with a retained Python kernel, use `bn-kernel` for large collections that should stay out of the transcript. For one-off reads and lifecycle, use the CLI.

## Working rules

- Discover syntax with `bn help <group>` or `bn <command> --help`; `bn capabilities --format json` is the registry-derived command index. Open the reference that matches the current task: `reference/reading.md` for read commands and JSON shapes, `reference/mutating.md` for writes, or `reference/runtime.md` for routing, sessions, and output.
- Select the target shown by `bn target list`. One open target needs no `-t`; several require `-t <selector>` or a deliberate pin. When concurrent agents share a project or process environment, pass `-i <instance>` and `-t <selector>` explicitly on target commands. Do not change shared `instance use` / `target use` pins during fan-out. Name a new headless bridge with `bn session start <path> --instance-id <id>`; that command returns the selector to use.
- Check `analysis_state` before treating a short function or string list as complete. A quick-loaded view has partial function discovery, and analysis-dependent reads can refuse until `bn refresh`.
- Bound large reads with `--limit`, `--lines`, `--out`, or `--estimate-output`. `BN_SPILL_TOKENS` opt-in replaces oversized stdout with an artifact envelope, so a pipe may search the envelope rather than the data. Check stderr and the envelope before concluding that a search had no match.
- For a bounds, access-width, or branch-predicate claim, confirm the relevant instructions with `bn disasm`. Include a preceding Thumb `IT` instruction when its condition governs the window.

## Command index

- **Read** — `target info/list`, `function list/search/info/cfg/structured-il`, `decompile`, `il`, `disasm`, `xrefs`, `callsites`, `evidence function/xrefs/table/message/init/calls/orient/surface/virtual-call`, `trace`, `dataflow defuse/callgraph/values`, `taint forward/backward/models`, `go functions`, `proto get`, `local list`, `data vars/symbols`, `read`, `types [show]`, `struct show`, `class list/show`, `strings`, `imports`, `exports`, `sections`, `tag list/get/types`, `comment list/get`, `bundle function` → `reference/reading.md`
- **Mutate** — `symbol rename`, `proto set`, `local rename/retype`, `comment set/delete`, `struct field set/rename/delete`, `types declare`, `data retype`, `function create`, `tag add/remove`, `tag type create/remove`, `go rename`, `batch apply` → `reference/mutating.md`
- **Discover** — `capabilities` → `reference/runtime.md`
- **Session** — `load`, `save`, `close`, `refresh`, `session start/list/stop/restart/status`, `instance list/find/gc`, `target close` → `reference/runtime.md`
- **Tooling** — `doctor`, `help`, `plugin install`, `skill install`, `spill gc` → `reference/runtime.md`
- **Escape hatch** — `py exec` → `reference/runtime.md`

## Mutation loop

Preview when supported, inspect the result, apply, read back, then `bn save` before closing. A first `proto set` on a function without a user prototype cannot be previewed: Binary Ninja cannot clear the `has_user_type` flag that preview would set. Apply it live only when intended, then read back with `bn proto get`. A failed batch attempts rollback; `rollback_failed` means the view may still be dirty. Mutations print a compact text status line by default; request `--format json --summary` for a machine-readable status or `--verbose` for diffs. See `reference/mutating.md` for status and batch details.
