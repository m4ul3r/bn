---
name: bn-vr
description: "Methodology for vulnerability research with bn: map reachable input, enumerate modeled and unnamed sinks, trace source-to-sink paths, audit parser and destination bounds by hand, and report proof limits. Use bn for command syntax."
---

# bn-vr

Use this methodology for a binary security audit. Use the `bn` skill for commands and target handling. Define the audited entry points and sink population, then report coverage and unresolved frontiers. A short import list, a short taint result, or an empty result from a capped scan is never an all-clear.

## Map entry points and sinks

1. Check `bn target info` and ensure `analysis_state` is full before relying on function search or strings. Use `bn evidence orient` for the initial map. Inspect imports, exports, protocol and configuration strings, and inbound xrefs to find input handlers.
2. On named targets, use `bn class list --no-stl` / `class show` to locate C++ parse, receive, and dispatch methods. Use exact or anchored `function search` queries for known API names; substring hits are discovery only. Confirm actual call targets. If a registration function receives a stack-built descriptor, `bn evidence calls <reg-fn> --arg-struct N --field callback:ptr@OFF` can map commands to callbacks; confirm heuristic fields at their reported source instructions.
3. Enumerate sink callsites, not just sink names. `bn taint models --role sink --present --callsites` finds modeled sinks present in the target. Also review unmodeled internal copy, format, decode, execution, and allocation helpers. Named imports and unnamed internal sinks can coexist in one binary; choose pivots per function rather than declaring the whole target an import-first or stripped lane. On a mostly unnamed binary, use strings → xrefs → decompile and high-fan-in small helpers to recover behavior. Empty imports alone do not prove static linkage.
4. Inspect constructors or data-only dispatch surfaces when RTTI, `.init_array`, vtables, callback tables, or indirect handler registration indicate them. `bn evidence init`, `evidence surface`, `evidence table`, and `evidence message` expose leads; confirm reachability before adding a handler to the audited population.

## Trace each interesting sink

For each sink callsite, identify its caller, input provenance, destination capacity, and every check that constrains the write or execution. `bn callsites <sink> --within <caller>` gives local context; `bn trace <caller> <call-addr> --arg N` follows a particular argument within that function. `--interprocedural` follows a return value into an internal callee, but does not climb from a callee parameter into its callers. Use xrefs and another trace for that direction.

Use `bn taint forward -f <entry> --source param:0` or a modeled input call to find candidate sink flows; use `bn taint backward -f <handler> --sink arg:memcpy:2` to seek origins across callers. Forward source locators include `param:<n>`, `var:<selector>`, `ret:<callee>`, `arg:<callee>:<n>`, and `call:<callee>` (`model:<callee>` is an alias). `call:` seeds every output declared by a source model, including output buffers; `ret:` seeds only the return. Argument indices are zero-based. Backward sink locators accept `param:`, `var:`, and `arg:`. In JSON, forward findings are under `reached_sinks`; backward findings are under `slices` (`sinks` echoes the backward query).

Taint is a may-analysis and reports its limits under `leaves`, `assumptions`, and `soundness`. A global or object-field load can break a buffer's taint path; seed the parser parameter and state that assumption. An unmodeled getter or decoder is a frontier to audit manually, not proof of safety. For unresolved indirect calls, inspect value sets and dispatch tables; use `--resolve-map` only with supported targets. If an internal wrapper or external call lacks a model, add a scoped `--models` override and rerun. A `recv`/`read` wrapper with an attacker-controlled length writing to a fixed buffer needs a bounded-write sink model with `len_arg` and `buf_arg`; modeling it only as an input source misses that class.

## Manual checks taint cannot finish

- **Parser headers:** compare the loop guard with the full fixed-header width before each field load. Bound unknown-option skips and handler dispatch against the remaining bytes. Distinguish out-of-bounds reads from stale bytes in an oversized receive buffer.
- **Unbounded or repeated copies:** trace destination allocation and source maximum length. For `strcat` or loops, compare capacity with the initial length plus all appends; a clean tainted-length slice does not establish capacity safety. Audit every caller of a copy wrapper.
- **Raw decoders and output parameters:** establish how many bytes can be written, whether capacity is passed to the callee, and whether checks happen before the write. A coarse memory leaf does not answer that question.
- **Machine-code proof:** confirm the actual field-load widths, branch guards, register arguments, and stack allocation with `bn disasm`. HLIL can distort AArch64 conditional compares and hoisted bounds. On ARM32/Thumb, inspect conditional instructions and the preceding `IT` mask with all covered instructions before treating a load or store as unconditional.

For each confirmed finding, report location, trigger, root cause, input-to-sink path, capacity or invariant evidence, impact, and a reproducible input sketch when possible. Mark an unsupported hypothesis as suspected and name the missing proof. Summarize audited sinks, remaining frontiers, and whether a capped or partial read limited coverage. Follow the `bn` mutation loop only when annotations are within the task's scope.
