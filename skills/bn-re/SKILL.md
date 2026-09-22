---
name: bn-re
description: "Methodology for understanding an unknown binary through bn: orient, prioritize functions, map C++ classes or hidden dispatch surfaces when present, recover types, and preserve supported findings. Use bn for command syntax."
---

# bn-re

Use this methodology when the goal is to understand a binary. Use the `bn` skill for command syntax and mutation rules. Keep observations, inferences, and verified facts distinct. Respect a read-only request: analysis and proposed names do not require changing the BNDB.

## Orient, then choose a path

1. Start with `bn evidence orient` for architecture, analysis state, imports, sections, function count, and a high-signal string sample. Drill into `bn target info`, `bn imports`, `bn strings`, and `bn function list` only for details the digest did not answer. On a quick-loaded view, run `bn refresh` before treating function or string results as complete.
2. Check whether the loaded view came from a saved `.bndb` or cache. If the task requires a from-scratch baseline, use `--no-bndb` when loading the raw binary and inspect existing annotations before claiming names were newly recovered. Runtime details are in the `bn` skill's `reference/runtime.md`.
3. Choose pivots by evidence: entry point and exports, distinctive strings and their xrefs, large functions, callsites of relevant imports, and high-fan-in utilities. Imports are clues, not a complete inventory of internal behavior. An empty import list does not by itself establish static linking or lack of sinks.
   On ELF, `file(1)` can say "stripped" because `.symtab` is absent while `.dynsym` still supplies public names. `bn exports` lists BN's global/weak definitions from any symbol table, and BN may synthesize entry/init names; comparing it with imports and functions does not establish which symbol table survived. `bn sections` shows mapped sections, not the ELF section header table; use `readelf -S` for the `.symtab`/`.dynsym` distinction.
4. If demangled C++ symbols or RTTI are present, use `bn class list --no-stl` and `bn class show <Class>` before manually grepping methods. The class lens groups methods, bases, vtables, and construction sites. With RTTI on a stripped binary, try `class show` even if `class list` says `no-vtable`: it can recover vtable slots from typeinfo. If class names are absent, pivot from strings and data references instead.

## Check hidden code surfaces when indicated

Look for constructors, vtables, callback registrations, or data-referenced code when the target shows signatures for them: `.init_array` / `.ctors`, RTTI, pointer tables, or indirect dispatch. Use `bn evidence surface` to survey candidates, then `bn evidence init`, `bn evidence table <addr>`, or `bn evidence message <type>` for the relevant part. Its candidate-table scan needs at least three consecutive code pointers by default (minimum two), so find a single callback global with `bn xrefs <global-address>` and inspect its constructor writes. A plain C target with no such signatures may be better served by function triage.

A table entry tagged plausible is a lead, not a recovered function. Confirm its target with disassembly and xrefs. If BN missed a real function and writes are in scope, preview `bn function create <addr>`, apply, read back, and save. Constructor callbacks can initialize globals before `main`; trace their writes when later code reads unexplained state. For ARM/Thumb tables, use the bridge's pointer interpretation instead of assuming eight-byte little-endian entries.

## Recover names and types iteratively

- Rename a function only when strings, callers, data flow, or instruction evidence support its role. Re-decompile neighbors after a useful rename.
- Recover prototypes from callsites and the callee's own register use. A first `bn proto set` on an auto-typed function cannot be previewed; apply only when intended and verify with `bn proto get` and callers. Later changes to an existing user prototype can be previewed.
- Build structs from repeated fixed-offset accesses, check the existing type first, then retype one parameter and re-decompile. Keep uncertain fields provisional.
- Use `bn callsites` for exact call instructions and `caller_static`; use `bn trace` for a particular argument's origin. Interprocedural `trace` follows return values into callees, not parameters up through callers. Step up with xrefs and repeat the trace there.
- Confirm widths, guards, and bounds in `bn disasm` before relying on HLIL. When a Thumb window starts inside an `IT` block, include the preceding `IT` and its covered instructions.

When annotations are authorized, use `bn`'s preview and verification loop, read back, and save the BNDB. A name should carry purpose; comments should carry assumptions, cross-function relationships, or deferred checks. Report a compact map: entry and dispatch path, identified functions, key types, evidence for uncertain edges, and the next functions to inspect. State which hidden surfaces were checked and which were not.
