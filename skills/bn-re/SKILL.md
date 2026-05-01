---
name: bn-re
description: Reverse engineering methodology for analyzing unknown binaries with the bn CLI. Covers systematic triage, function identification, iterative type recovery, call graph analysis, naming conventions, and struct reconstruction.
---

# bn-re — Reverse Engineering Methodology

Use this skill when the user wants to understand, reverse engineer, or analyze a binary. This is a methodology guide — it tells you *what to do and why*. For command syntax, see the `bn` skill.

## Approaching an Unknown Binary

Start broad, then narrow:

1. **Orient** — get architecture, platform, and entry point:
   ```bash
   bn target info
   ```

2. **Survey imports and strings** — these reveal libraries, APIs, and embedded literals that hint at functionality:
   ```bash
   bn imports
   bn strings
   ```
   Imports tell you what the binary *does* (network I/O, file ops, crypto, GUI). Strings reveal configuration keys, error messages, format strings, and embedded paths.

3. **Scan the function list** — get a sense of scope:
   ```bash
   bn function list
   ```
   Note the total count, address range, and whether symbols are stripped. A stripped binary with 2000 functions requires different tactics than a symbolicated one with 50.

## Function Triage

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
  bn strings --query "error\|fail\|password\|key\|flag"
  ```
  Then use `bn xrefs` on the string address to find which functions reference it.
- **Import callers** — trace backward from interesting imports:
  ```bash
  bn xrefs malloc
  bn callsites recv --within <function>
  ```

## Hidden Code Surfaces

Binary Ninja's auto-analysis follows direct calls. Two important categories of code don't sit on that graph and will be invisible until you go looking for them.

### Pre-main code (`.init_array`, constructors)

Functions tagged with `__attribute__((constructor))`, C++ static initializers, and any code the linker registers in `.init_array` run *before* `main`. They commonly stage globals, derive keys, or wire up dispatch tables — exactly the kind of setup that breaks an analysis built only from `main`'s call graph.

To find them:

1. Look for `.init_array` in `bn sections` (it's a `ReadOnlyData` section near `.dynamic`).
2. Walk the array — each entry is one function pointer:

   ```bash
   bn py exec --stdin <<'PY'
   import struct
   start = 0x403c48      # .init_array start address from `bn sections`
   size  = 16            # .init_array size from `bn sections`
   for ptr in struct.unpack(f'<{size // 8}Q', bv.read(start, size)):
       print(hex(ptr))
   PY
   ```

3. Skip the toolchain stub `frame_dummy` — it's the first slot on most GCC builds and rarely interesting.
4. Decompile each remaining entry. Anything that writes to BSS / `.data` / `.bss` is staging state for `main` to read; rename it `stage1_<purpose>` (or similar) so the relationship is visible from later analysis.

ELF entry-flow review when nothing in `main` makes sense: `entry_point → __libc_start_main → main`, *but* `_start` and `__libc_start_main` invoke the `.init_array` callbacks before `main`. If a global "appears from nowhere" in `main`, the producer is almost certainly an `.init_array` entry.

### Data-only function references

If the binary has a dispatch table (an array of function pointers — common in VMs, FSAs, vtables, callback registries), Binary Ninja often *won't* identify the targets as functions because there's no direct `call` to them, only a data reference from the table.

Symptoms: `bn decompile <addr>` errors with `Function not found`; the bytes at `<addr>` look like a function prologue (`endbr64`, `push rbp`, …) on disasm, but it's marked as data.

Force-create the function and re-analyze:

```bash
bn py exec --stdin <<'PY'
target = 0x4014d0
if not bv.get_function_at(target):
    bv.create_user_function(target)
    bv.update_analysis_and_wait()
print("created" if bv.get_function_at(target) else "still missing")
PY
```

After that, the normal `bn decompile` / `bn xrefs` flow works on the new function.

When this comes up most: VM opcode handler tables, FSA predicate tables, COM-style vtables, plugin registries. If you've recovered a struct of `(tag, fn_ptr)` rows and one of the `fn_ptr` targets is missing, this is almost always why.

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
When you have multiple renames or retypes queued up, use `bn batch apply` with a manifest instead of individual commands. This is faster and atomic.

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

- **Build a mental call tree** — for key functions, trace both up and down 2-3 levels. This reveals the flow: entry -> dispatch -> handler -> utility.

## Naming Conventions

Consistency helps both you and the user:

- Use `snake_case` for functions and locals (matches C convention and Binary Ninja defaults).
- Prefix with module/subsystem when apparent: `net_send_packet`, `ui_draw_button`, `crypto_decrypt_block`.
- Name by *what it does*, not implementation: `validate_input` not `check_and_branch_if_less`.
- For unknown purpose, use descriptive placeholders: `process_buffer_0x4010a0` is better than `sub_4010a0`, but only rename when you're reasonably confident.
- Preview before committing:
  ```bash
  bn symbol rename sub_401000 process_buffer_0x4010a0 --preview
  ```

## Commenting

Good names carry most of the meaning, but comments fill the gaps:

- **Explain non-obvious behavior** — if a function's purpose or mechanism isn't clear from its name and types alone, add a comment at its entry address:
  ```bash
  bn decompile <function> --addresses
  bn comment set --address 0x401000 "Walks the linked list of active sessions and frees expired ones"
  ```

- **Use TODO comments for deferred work** — when you identify something that needs further analysis but isn't the current focus, leave a TODO so future passes can pick it up:
  ```bash
  bn comment set --address 0x402000 "TODO: second argument looks like a callback — trace callers to confirm signature"
  bn comment set --address 0x403000 "TODO: this allocation is never freed in the error path — possible leak"
  ```

- **When to comment vs. when to rename** — if you can express it in a name, rename instead. Comments are for *why* and *context* that a name can't carry: edge cases, assumptions, relationships to other functions, deferred questions.

- **Review outstanding TODOs** — on subsequent passes, check for deferred work:
  ```bash
  bn comment list --query TODO
  ```

## Struct Reconstruction

When decompiled code shows repeated offset accesses from a pointer (e.g., `*(arg1 + 0x10)`, `*(arg1 + 0x18)`), that pointer is likely a struct.

### Workflow

1. **Collect evidence** — decompile functions that use the pointer and note all accessed offsets and their apparent types:
   ```bash
   bn decompile <function> --addresses
   ```

2. **Check if a struct already exists** — it may be partially defined:
   ```bash
   bn struct show <TypeName>
   ```

3. **Create or extend the struct** — set fields at observed offsets:
   ```bash
   bn struct field set Player 0x0 vtable "void*" --preview
   bn struct field set Player 0x8 name "char*" --preview
   bn struct field set Player 0x10 health int32_t --preview
   ```

4. **Apply and verify** — retype the parameter to use the struct, then re-decompile to confirm the output improves:
   ```bash
   bn local retype <function> arg1 "Player*"
   bn decompile <function>
   ```

5. **Iterate** — new field names in decompilation often reveal more structure. Repeat until the code reads naturally.

For complex structs, use `bn types declare` or `bn py exec` with `StructureBuilder` (see the `bn` skill for examples).
