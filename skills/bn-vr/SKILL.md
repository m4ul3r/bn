---
name: bn-vr
description: Vulnerability research methodology for finding security bugs in binaries with the bn CLI. Covers attack surface identification, input tracing, common vulnerability patterns, systematic audit workflow, manual taint analysis, and reporting findings.
---

# bn-vr — Vulnerability Research Methodology

Use this skill when the user wants to find vulnerabilities, audit for bugs, check security, or analyze attack surface in a binary. This is a methodology guide — it tells you *what to look for and why*. For command syntax, see the `bn` skill.

## Attack Surface Identification

Start by mapping what the binary does and where untrusted data enters:

1. **Dangerous imports** — scan for functions with known vulnerability history:
   ```bash
   bn imports
   ```
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
Plain `bn xrefs`/`decompile` thin out when dispatch is indirect or the decompiler's argument story is incomplete (common in C++/IPC services). Three evidence helpers (syntax in the bn skill, §4) pick up the trail:

- `bn evidence xrefs <sink-or-string>` — like `bn xrefs` but each ref carries section/segment/symbol + the referencing disassembly, so you can tell a real code caller from a vtable/RTTI/descriptor slot, and spot a sink that's reachable **only** through a vtable (no direct call).
- `bn evidence function <caller>` — shows the raw ABI arguments (registers + LLIL/MLIL/HLIL) next to the pseudo-C at each call, including the vtable offset for an indirect/virtual call. Use it to recover a sink's real arguments without dropping to disasm, and to see through `j_*`/PLT thunks to the true callee.
- `bn evidence message <TypeName>` — for protobuf/IPC message handlers, maps a message type-name string to its serializer/handler pointers, giving you the receive→parse→dispatch entry points to trace forward from.

Reminder: HLIL can hide the real access/operand width — confirm the size in `bn disasm` before concluding on a truncation/off-by-one (see the width note in the bn skill, §4).

## Common Vulnerability Patterns

When reading decompiled output, watch for these patterns:

### Buffer overflows
- Fixed-size stack buffer (`char buf[64]`) with unbounded copy (`strcpy(buf, user_input)`)
- `memcpy` where the length comes from untrusted input without bounds checking
- Off-by-one in loop bounds writing to a buffer (`<= len` instead of `< len`)
- Integer truncation before allocation: `uint16_t len = attacker_controlled_uint32` then `malloc(len)`

### Format string vulnerabilities
- `printf(user_input)` or `sprintf(buf, user_input)` — the format argument must be a literal, never user-controlled
- Look for `%n` capability which allows arbitrary memory writes
- Check `syslog`, `fprintf`, `snprintf` — all `*printf` variants are affected

### Integer issues
- Arithmetic before allocation: `size = count * element_size` can overflow, leading to undersized allocation
- Signed/unsigned confusion: negative value passing a signed check but wrapping to large unsigned
- Truncation: 32-bit value stored in 16-bit variable before use as length

### Use-after-free
- `free(obj)` followed by continued use of `obj` — look for functions that free a structure member but the caller still holds a reference
- Event handlers that free an object while iteration over a list containing it continues
- Error paths that free resources but don't null the pointer, allowing reuse on retry

### Off-by-one
- Loop bounds: `for (i = 0; i <= len; i++)` writes one past the buffer
- Null terminator not accounted for: `malloc(strlen(s))` instead of `malloc(strlen(s) + 1)`
- Fence-post errors in index calculations

### Command/path injection
- User input concatenated into a string passed to `system()`, `popen()`, or `exec*()`
- Path traversal: user-supplied filename with `../` not sanitized before `open()`

## Systematic Audit Workflow

Choose an approach based on the binary:

### Pattern-based (faster, good for large binaries)
1. Search for dangerous sinks: `bn xrefs strcpy`, `bn xrefs sprintf`, etc.
2. For each callsite, trace the arguments backward to check for attacker control
3. Skip callsites where arguments are provably safe (constants, bounded copies)
4. Prioritize callsites where arguments come from input sources

### Function-by-function (thorough, good for small/critical binaries)
1. List all functions and triage by relevance (input handlers, parsers, protocol implementations)
2. Decompile each target function and read line by line
3. For each potential issue, trace forward and backward to confirm exploitability
4. Track which functions are audited to avoid gaps

### Hybrid approach
1. Start pattern-based to find low-hanging fruit
2. Switch to function-by-function for high-value code (parsers, auth, crypto)

## Manual Taint Analysis

When you need to rigorously track whether attacker-controlled data reaches a dangerous operation:

1. **Mark sources** — identify which function parameters or return values carry untrusted data:
   ```bash
   bn decompile <input_handler>
   ```

2. **Propagate taint** — trace through assignments, copies, and function calls:
   - Direct assignment: `dest = tainted_src` -> `dest` is tainted
   - Copy: `memcpy(dest, tainted_src, len)` -> `dest` is tainted
   - Function call: if a tainted value is passed as an argument, check whether the callee propagates it to its return value or to other outputs. Use `bn trace` to follow a sink argument back through SSA to its origin:
     ```bash
     bn trace <sink_function> <call_address> --arg N                    # intra: one function
     bn trace <sink_function> <call_address> --arg N --interprocedural  # IP: follow across calls
     ```
   ```bash
   bn callsites <function> --within <caller>
   bn decompile <callee>
   ```

3. **Check sinks** — when tainted data reaches a dangerous function, assess exploitability:
   - Can the attacker control enough of the input to trigger the bug?
   - Are there length checks, sanitization, or canaries in the path?
   - What is the memory layout at the target (stack vs heap, adjacent allocations)?

## Reporting Findings

For each potential vulnerability, capture:

- **Location**: function name and address
- **Bug class**: buffer overflow, format string, integer overflow, UAF, etc.
- **Trigger condition**: what input causes the vulnerable path to execute
- **Root cause**: why the code is wrong (missing bounds check, unchecked return value, etc.)
- **Impact**: what an attacker can achieve (crash, code execution, info leak, privilege escalation)
- **Data flow**: the path from input source to vulnerable sink, including intermediate functions
- **Proof of concept**: if possible, describe or construct an input that triggers the bug
