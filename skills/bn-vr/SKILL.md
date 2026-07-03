---
name: bn-vr
description: "Vulnerability-research methodology for finding security bugs in binaries via the bn CLI. Core discipline — exhaustive sink enumeration plus source→sink tracing — is what stops a fast false all-clear: when the import table looks empty or bounded, it forces the audit onto the binary's own copy/format/dispatch sinks and (on C++) the class lens to the directive/parse handlers, then taint/trace to prove or refute attacker control. Covers attack-surface mapping, the stripped/static lane, common bug patterns, the taint engine, and reporting."
---

# bn-vr — Vulnerability Research Methodology

Use this skill when the user wants to find vulnerabilities, audit for bugs, check security, or analyze attack surface in a binary. This is a methodology guide — it tells you *what to look for and why*. For command syntax, see the `bn` skill.

## Attack Surface Identification

Start by mapping what the binary does and where untrusted data enters:

> **No imports? (static / stripped firmware.)** If `bn imports` comes back empty or near-empty, the binary is statically linked and the import-first steps below won't bite — `bn xrefs strcpy` / `function search strcpy` return nothing when no function is *named* `strcpy`. Use the **Stripped / static lane** at the end of this section instead.
>
> **`file`=stripped ≠ static lane.** `file(1)` reports "stripped" whenever `.symtab` is gone, but a very common middle case still keeps a **fat `.dynsym`**: default-visibility / `-rdynamic` firmware executables (and *every* shared object) export hundreds of `GLOBAL DEFAULT FUNC` symbols plus named PLT imports. There the **import-first lane fully applies** — BN has real names for nearly every function. Pick the lane by the tell, not by `file`: `bn imports` **non-empty** *or* `bn function list` **mostly named** (only a minority of `sub_XXXX`) ⇒ import-first, even when `file` says stripped. Route to the stripped/static lane only when `bn imports` is empty/near-empty **and** `function list` is overwhelmingly `sub_XXXX`.

> **Quick-loaded target?** If the binary was opened with `bn load --quick` / `bn session start --quick`, the code is not analyzed yet: `bn imports` and `bn sections` work, but `bn strings` (step 3) **errors** until `bn refresh` (it refuses rather than return nothing) and `bn function list` / `bn function search` return only a **partial** set — a false "no dangerous strings, no sinks" all-clear. Confirm `analysis_state` is `"full"` (`bn target info`) and `bn refresh` before auditing. See "Quick Load" in the `bn` skill.

> **C++ / symbolicated target? Use the class lens to reach the handlers fast.** On a binary with demangled C++ symbols/RTTI, `bn class list --no-stl` surfaces the domain classes and `bn class show <Name>` lists a class's methods + vtable — the quickest way to locate the directive/parse/dispatch/`handle*`/`onReceive` entry points that take untrusted input, *before* you enumerate sinks. (On stripped/static firmware with no symbols, skip it and use the "Stripped / static lane" below.)

1. **Dangerous imports** — scan for functions with known vulnerability history:
   ```bash
   bn imports
   ```
   > **Shortcut:** `bash scripts/sink-sweep.sh -t <target>` (in this skill) enumerates the **copy / format-string / exec / input** sinks below **and** runs `bn xrefs` on each, printing every call site to trace back to a source — the sink-enumeration step that's easy to skip. Forward the usual `-i`/`-t` selectors. **Do not pass `--out` to it** — the script manages its own output; redirect stdout if you want a file (`... > sinks.txt`). (It skips the malloc family by design — heap bugs aren't found by xref'ing the allocator. Falls back cleanly: prints "no dangerous-sink imports" on a static/stripped target — then use the lane below.)

   Flag these categories:
   - **Unbounded copies**: `strcpy`, `strcat`, `sprintf`, `gets`, `scanf` (no length limit)
   - **Bounded but misusable**: `strncpy`, `snprintf`, `memcpy`, `memmove` (length may be attacker-controlled)
   - **Memory management**: `malloc`, `calloc`, `realloc`, `free` (UAF, double-free, heap overflow)
   - **Execution**: `system`, `exec*`, `popen`, `dlopen` (command/code injection)
   - **Format strings**: any `*printf` family where the format argument could be user-controlled

2. **Input sources** — identify where external data enters:
   ```bash
   bn imports
   ```
   Look for: `read`, `recv`, `recvfrom`, `fgets`, `fread`, `getenv`, `argv` access patterns. These are your taint sources.

3. **Interesting strings** — format strings, SQL fragments, shell commands, and paths hint at injection surfaces:
   ```bash
   bn strings --regex --query '%s|%x|%n|SELECT|INSERT|/bin/' --no-crt --min-length 4
   ```
   `--regex` makes `--query` a case-insensitive regex so the `|` actually means OR — without it `--query` is a literal substring and `\|` matches nothing. Use `--no-crt` to suppress locale/CRT noise and `--min-length` to skip short fragments; `--section .rodata` restricts to read-only data.

4. **Memory layout** — understand which regions are writable, executable, or both:
   ```bash
   bn sections
   ```
   Look for writable+executable sections (W+X) — these are high-value targets for code injection. Check section sizes and ranges to understand the binary's memory layout.

### Stripped / static lane (no imports, no symbols)

On stripped static firmware (busybox, embedded ARM/MIPS), `bn imports` is empty and `bn xrefs strcpy` / `function search strcpy` return nothing — there are no import names and no function names. Invert the workflow: enter from *data*, not from imported symbols.

1. **Confirm the case.**
   ```bash
   bn target info               # static? stripped? arch (ARM/Thumb/MIPS)?
   bn imports                   # empty / near-empty => statically linked
   bn function list --count     # thousands of sub_XXXX names = stripped
   ```
   A few-thousand-function count with `sub_XXXX` names and an empty import table is the signature.

2. **Enter from strings.** Strings are the surviving attack-surface map — config paths, format strings, command/applet names, protocol keywords. The first column of `bn strings` output is each string's address; pivot it to the code that uses it:
   ```bash
   bn strings --regex --query '/etc/|/bin/|%s|%n|password|http|telnet|login' --no-crt --min-length 4
   bn xrefs <string-address>    # who references this string
   bn decompile <referencing-fn>
   ```
   This `strings -> xrefs <addr> -> decompile` chain is the reliable spine when name/import search returns `none`.

3. **Recover the libc-like sinks by shape.** You can't `bn xrefs strcpy` if strcpy has no name — so identify the unnamed helpers behaviorally, then name them, which restores the source->sink tracing below. As you read the functions strings led you to, watch for a helper called from a copy/format pattern:
   - libc primitives (memcpy, strcpy, strlen, sprintf) are small, leaf/near-leaf, and called from *many* sites. Run `bn xrefs <sub_addr>` on a candidate — a high inbound count plus a tiny body is the tell.
   - Decompile the candidate, recognize the idiom (byte-copy loop, copy-until-NUL, scan-for-zero), then `bn symbol rename <addr> memcpy --preview`, verify, and apply.
   - Now `bn xrefs memcpy` works and the Pattern-based audit (below) is back in play.

4. **Walk constructor and dispatch tables.** Static firmware hides entry points the direct-call graph won't show (see bn-re's "Hidden Code Surfaces"):
   ```bash
   bn evidence init             # .init_array / constructors (Thumb-normalized)
   bn evidence table <addr>     # applet / dispatch / fuse_operations tables as fn pointers
   ```

5. **Confirm the bound in disasm — not just widths.** Stripped + ARM means the decompiler's story is frequently wrong for more than operand size: confirm field loads (`ldrb` vs `ldr`) **and** the guard/loop shape in `bn disasm` before concluding off-by-one / truncation / overflow. HLIL flattens `ccmp`/`csel` range guards into misleading ternaries, can render a hoisted loop-invariant limit relative to a *moving* pointer (so a real cap looks like it never fires), and hides a missing `<< 4` in a size accumulator — each has faked a critical finding. See the "HLIL can mislead beyond access width" note in the `bn` skill.

**Worked example — busybox.** BusyBox is an applet multiplexer: `main` dispatches on `argv[0]`/`argv[1]` to applet handlers, so its real attack surface is the applet table plus the strings that name applets:
```bash
bn strings --regex --query 'httpd|telnetd|login|/etc/(passwd|shadow)' --no-crt
bn xrefs <httpd-string-addr>            # -> the httpd applet handler
bn evidence table <applet-table-addr>  # enumerate every applet entry point
```
Then audit each reachable applet (httpd request parsing, telnetd, login) with the source->sink tracing below — now that the sinks have names from step 3.

## Input Tracing: Sources to Sinks

The core of VR is connecting *where data comes from* to *where it's used dangerously*.

### Forward tracing (from source)
Start at an input function and trace where its output flows:
```bash
bn xrefs read
bn callsites read --within <handler_function>
bn decompile <handler_function>
```
Read the decompilation: does the buffer from `read()` flow into `strcpy()`, `sprintf()`, or `system()` without validation?

### Backward tracing (from sink)
Start at a dangerous function and trace where its arguments come from:
```bash
bn xrefs strcpy
bn callsites strcpy --within <function>
```
For each callsite, examine: is the source argument bounded? Is the destination buffer large enough? Can the attacker control the source?

### Multi-hop tracing
Data often passes through several functions before reaching a sink. Follow it step by step:
1. Identify the sink callsite and its arguments
2. Trace each argument back through the caller's locals and parameters
3. Use `bn xrefs` on the caller to find *its* callers
4. Repeat until you reach an input source or lose the trail

### Sink-to-source tracing with `bn trace`

When you find a dangerous call (e.g. `memcpy(dst, src, len)`) and need to know where a specific argument originates, use `bn trace` to walk the SSA use-def chain backward:

```bash
bn trace <containing_function> <call_address> --arg N
```

This works within a single function (intraprocedural). For example, tracing which buffer flows into a `memcpy` destination reveals whether it's a stack local, a heap allocation, or a function parameter — letting you assess attacker control without reading every line of decompilation.

Add `--interprocedural` to cross call boundaries when the traced value is another function's **return value**:

```bash
bn trace handler 0x1234 --arg 0 --interprocedural        # follow through callee return
bn trace handler 0x1234 --arg 0 --interprocedural --ip-depth 3  # deeper recursion
```

Scope of `--interprocedural`: it follows a value *into* a callee only when that value is the callee's return value, then traces the callee's return-value origins and stops at the callee's own parameters. It does **not** map a callee's parameters back to the caller's argument expressions, and it does **not** walk *up* the caller chain. So for "this arg is the return of `foo()` — where does `foo` get it?" IP mode answers directly; for "this arg came from my own caller" or "trace up through every caller," step up manually with `bn xrefs` on the containing function and re-run `bn trace` in each caller.

The walk requires MLIL SSA, so use `--view mlil` (the default; `--view hlil` exists but often can't locate calls nested in assignment statements). IP mode works best on self-contained code (static binaries, kernel modules); for shared-library PLT/import calls the callee has no MLIL body, so IP mode correctly falls back to intraprocedural behavior. Use `--format json` to get structured step-by-step SSA variable information.

### When the trail is indirect or the args are unclear
Plain `bn xrefs`/`decompile` thin out when dispatch is indirect or the decompiler's argument story is incomplete (common in C++/IPC services). Three evidence helpers (syntax in the bn skill — `reference/reading.md`) pick up the trail:

- `bn evidence xrefs <sink-or-string>` — like `bn xrefs` but each ref carries section/segment/symbol + the referencing disassembly, so you can tell a real code caller from a vtable/RTTI/descriptor slot, and spot a sink that's reachable **only** through a vtable (no direct call).
- `bn evidence function <caller>` — shows the raw ABI arguments (registers + LLIL/MLIL/HLIL) next to the pseudo-C at each call, including the vtable offset for an indirect/virtual call. Use it to recover a sink's real arguments without dropping to disasm, and to see through `j_*`/PLT thunks to the true callee.
- `bn evidence message <TypeName>` — for protobuf/IPC message handlers, maps a message type-name string to its serializer/handler pointers, giving you the receive→parse→dispatch entry points to trace forward from.

Reminder: HLIL misleads beyond access/operand width — besides field-load size, it flattens `ccmp`/`csel` range guards into ternaries, aliases hoisted loop-invariant bounds to a moving pointer, and can drop a `<< 4` from a size accumulator. Confirm the bound in `bn disasm` before concluding on a truncation/off-by-one/overflow (see the "HLIL can mislead beyond access width" note in the `bn` skill).

## Systematic Audit Workflow

Go **pattern-based for breadth** — enumerate sinks (`bn xrefs strcpy`, …), trace each callsite's args backward, skip provably-safe ones, prioritize those fed by input sources — then **switch to line-by-line** on the high-value code (parsers, auth, crypto), tracking what's audited so you don't leave gaps.

## Taint Analysis (`bn taint`)

`bn taint` automates the propagation step over Binary Ninja's MLIL-SSA: it
follows def-use chains through assignments, arithmetic, phi joins, and a
built-in function-model DB (`recv`/`read`/`fgets`/`getenv` sources;
`memcpy`/`strcpy`/`sprintf`/`system`/… sinks). It is **interprocedural** — it
descends into in-binary callees (depth-bounded by `--max-depth`, cached per
function) so a sink inside a helper is reported against the entry function with
the full cross-boundary path, and it carries taint back through output-pointer
parameters (a helper that fills `*dst` taints the caller's buffer). Heap/pointer
flows (`*p = tainted; x = *p`) are recovered via memory-SSA store/load
correlation, not just stack buffers, and locally-built descriptor structs
populated from input and passed by address (a common protocol-stack pattern)
are tracked through their field stores. It stays honest about its limits: every
coarse-memory step, every unmodeled external call, and every unresolved indirect
call is reported under `assumptions` / `leaves`, and the result always carries a
`soundness` disclaimer. It is a may-analysis, not a proof.

**Forward — from a source to whatever sinks it reaches:**
```bash
# the buffer that recv()'s 2nd arg fills is the source:
bn taint forward -f <handler> --source arg:recv:1
# a function parameter is tainted on entry:
bn taint forward -f <handler> --source param:0
```
Reports each reached sink with its bug class (`overflow_len`,
`command_injection`, `format_string`, …) and the full SSA path.

> **Global / struct-field buffer source → seed the parser entry instead.** When the
> recv destination is a long-lived pointer in a global or daemon struct
> (`recvfrom(fd, G.pkt, …)`, then later `p = G.pkt; parse(p, n)`), seeding
> `--source arg:recvfrom:1` (or `arg:read:1`) can report **zero propagation** —
> the buffer-pointee taint isn't anchored across the global/struct-field pointer
> *load* that re-derives the parser's argument, so the recv store and the parser
> load aren't correlated. Seed the parser directly instead —
> `bn taint forward -f parse_packet --source param:0` — which surfaces the
> byte-copy / option-extraction loops honestly as `coarse_memory_store` leaves.
> The same indirect-load caveat is what bites whenever the recv destination is
> reached through a pointer load rather than a direct local buffer.

**Backward — slice a sink's argument back to its origin:**
```bash
bn taint backward -f <handler> --sink arg:memcpy:2   # where does the length come from?
bn taint backward -f <handler> --sink arg:strcpy:1
```
When a slice bottoms out at a parameter, it continues up into callers
(`caller_sites`, depth-bounded by `--max-depth`), mapping the parameter back to
each call's argument — so a length checked in a helper is traced to the `recv`
that produced it. Each result lists the `crosses:` chain and an `origin`.

`bn taint backward` and `bn trace` (see "Sink-to-source tracing" above) are
complementary backward slicers: `taint backward` seeds on a sink/var locator
and ascends into *callers* to find where a value originates, while `bn trace`
pins an exact call argument at a specific address and descends into *callees*
for return-value provenance. Reach for `trace` when interrogating a concrete
callsite, `taint backward` when hunting origins across the caller chain.

> **JSON: the findings live under a different key per direction.** `taint forward`
> puts the discovered sinks in **`reached_sinks`** (top-level array; `stats.sinks`
> is just its count). `taint backward` puts the discovered slices in **`slices`** —
> its top-level **`sinks`** key echoes the *input* sink locators you passed, **not**
> the findings. So extract with `jq '.reached_sinks[]'` (forward) vs `jq '.slices[]'`
> (backward); reading backward `.sinks` as results misreports a real slice as "0
> sinks".

**Source/sink locator grammar:** `param:<n>` · `var:<selector>` ·
`ret:<callee>` (forward only) · `arg:<callee>:<n>` (the buffer arg `n` points at).

**Underlying primitives** (useful on their own, and for auditing a taint result):
```bash
bn dataflow defuse <fn> --var <name#version>   # SSA def site + all uses
bn dataflow callgraph <fn> --direction callees # resolved callees; indirect via value-set
bn dataflow values <fn> --at <addr>            # value-set (possible values)
bn function structured-il <fn>                 # per-instruction op + vars_read/written
```

Indirect calls reached by taint are resolved automatically when value-set
analysis can pin the target(s) (e.g. const function-pointer tables), and taint
follows into each resolved target; only genuinely unresolved ones become
`leaves`. You can force resolution with `--resolve-map FILE`
(`{"0x4011f0": ["0x401176", "0x401195"]}`).

**When taint stops (the known-hard cases), fall back to the manual chain.** If a
flow hits a `leaves` entry — an `indirect_call_unresolved` (function pointer /
vtable VSA could not pin) or an un-modeled external — resolve it by hand and
continue:
- Try `bn dataflow callgraph <fn>` / `bn dataflow values <fn> --at <call-addr>`
  to pin an indirect target; feed it back via `--resolve-map`, or re-run taint
  inside that callee.
- For interprocedural flows, taint each function in turn and stitch the path:
  `bn taint backward -f <callee> --sink arg:<sink>:<n>` then map the callee's
  tainted parameter back to the caller's argument with
  `bn callsites <callee> --within <caller>` + `bn decompile <caller>`.
- For an un-modeled external, add a model to the override file
  (`~/.cache/bn/taint_models.json`, or `$BN_TAINT_MODELS`) and re-run.

Then assess exploitability as before: can the attacker control enough of the
input, are there length/sanitization checks in the path, and what is the memory
layout at the target?

## Reporting Findings

Per finding: location (fn + addr), bug class, trigger condition (what input reaches it), root cause, impact, the source→sink data-flow path, and a PoC sketch if constructible. Always state the `bn taint` `soundness` caveat — it's a may-analysis, not a proof.
