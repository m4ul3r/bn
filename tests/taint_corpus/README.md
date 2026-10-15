# Taint validation corpus

Small, self-contained targets with structural ground truth (`*.EXPECTED.json`)
for the `bn taint` / `bn dataflow` features. Sources are checked in; binaries
are compiled by `tests/test_taint_integration.py` at test time (the test is
skipped unless a real Binary Ninja is importable and a C/C++ compiler is on
PATH). Scoring is **structural** — we assert the expected (source, sink, class)
tuples and forbidden classes appear, not exact addresses, so the suite is not
brittle across compilers/optimisation.

| target                 | exercises                                              |
|------------------------|--------------------------------------------------------|
| `overflow.c`           | read() -> load -> arithmetic -> memcpy length (positive, fwd + bwd) |
| `read_bounded.c`       | `call:` source seeds the read() return; the read-bounded memcpy length is `bounded_len` (with source bound), not overflow (#157/#159) |
| `sanitized.c`          | constant memcpy length (negative — no false positive)  |
| `command_injection.c`  | tainted buffer -> system() pointer arg (external model)|
| `indirect_call.c`      | tainted data into a function-pointer call (leaf, not dropped) |
| `indirect_resolve.c`   | const function-pointer table resolved by value-set; taint follows into both targets |
| `conn_recv_vtable.c`   | connection-object vtable: recv SOURCE anchored at `conn->ops->recv` (an indirect call value-set can't pin) via `--resolve-map`; tainted length reaches a memcpy sink (#282) |
| `conn_send_vtable.c`   | backward mirror: send SINK anchored at `conn->ops->send` via `--resolve-map`; slice reaches the length parameter origin (#282) |
| `interproc.c`          | tainted buffer crosses a call boundary; sink lives in the callee (interprocedural descent) |
| `outparam.c`           | helper fills an output buffer through a pointer param; taint flows back to the caller |
| `multihop.c`           | input -> snprintf propagator -> system (command injection); mirrors DVRF socket_cmd |
| `bw_interproc.c`       | backward slice from a memcpy-length parameter crosses into the caller to reach the recv source |
| `heap_mem.c`           | tainted value stored to a heap buffer and loaded back reaches a memcpy length (memory-SSA correlation); parallel const store must not taint |
| `vtable.cpp`           | C++ virtual dispatch shows as an indirect callee (honest degradation) |
| `fortified.c`          | FORTIFY build: read -> `__memcpy_chk` length sink (`fortified_overflow`); needs `cflags` |
| `vararg_sprintf.c`     | a tainted *later* vararg into sprintf taints the dest buffer -> system (full vararg propagation, not just the first) |
| `global_buf.c`         | read() fills a global/static buffer, loaded back into a memcpy length (global buffer tracking) |
| `file_write.c`         | tainted data into fwrite: flagged only with `--sink-class file_write`, silent by default (positive + negative) |
| `heap_field.c`         | backward slice through a heap struct-field length: reaching-store recovery, and a `field_load_unresolved` leaf when the store is out of scope (no silent dead-end at malloc) |
| `bw_local_fill.c`      | backward slice from a `popen` command buffer continues through the modeled `snprintf` that filled the local buffer (origin: parameter → caller); an unmodeled in-binary writer handed `&buf` ends in an `address_of_local_unresolved` leaf and `safe_to_report_complete_slice: false` |
| `strtok_loop.c`        | `strtok(NULL, ...)` continuation: the second token carries the tainted first call's state into `system()`, disclosed as an assumption |
| `bw_terminate_append.c` | `fill; buf[n] = 0; append; popen` at -O1: the aliased-field terminator (a memory def with no `src_memory`) must not break the writer chain -- the parameter / `read` source copied in by the OLDER writer is the origin, not the literal append; a nested `strcpy(cmd, tmp)` source buffer is walked to its `read` |
| `bw_memory_paths.c`    | backward memory-path honesty: a conditional fill (memory phi with a writer-less arm) is an `address_of_local_unresolved` frontier; a source buffer overwritten AFTER it was copied is read at the copy's memory version; `&buf` escaping to a global makes a later unmodeled call an opaque writer and a later modeled call that reloads the stash (`strcpy(g_stash, host)`) a fill; a 1-byte terminator after a conditional `read` does not cover the bare arm; a `memset` arm is a literal fill that does |
| `strtok_escape.c`      | strtok state left tainted at return is disclosed (`tokenizer_state_escapes`) and withholds `safe_to_report_all_clear`; a clean non-NULL `strtok` that lies on EVERY path from the tainted call to the `strtok(NULL)` resets the continuation (negative); one under `if (mode)`, or a priming call before a loop that the back-edge bypasses (`tok_prime_loop`), does not, so the continuation stays a real `system()` flow (positive). `strtok_escape_o1.c` rebuilds it at -O1 |
| `bw_opt_shapes.c`      | -O1 optimiser shapes (`bw_opt_shapes_o2.c` rebuilds at -O2): a store through a phi'd / non-constant pointer is a writer whose value reaches the ascent (`buf[i] ^= key[i]`, `*p++ = *src++`), never a bare `source` with `complete_slice: true`; a byte-wise `*p++ = *src++` loop is a loop-only fill and stays a frontier leaf; `__builtin_memset` is a literal fill; `char buf[] = "lit"` lifted to `__builtin_strncpy` is modeled; a tainted `memcpy` length is a step and the fill outranks the terminator |
| `bw_global_operand.c`  | a `CONST_PTR` fill operand is a literal only in read-only storage: `sprintf(cmd, "%s", g_host)` from a `.bss` array is a frontier leaf with origin `global` (`complete_slice: false`); the `.rodata` literal twin is complete |
| `bw_offset_address.c`  | an address built on a buffer at a VARIABLE offset is that buffer: `popen(line + off)` and `strcpy(buf, &other[i])` walk the `read` that filled it beside the offset; `g_hosts[i]` (a `.bss` array element) is a `global` frontier leaf. `bw_offset_address_o1.c` rebuilds it at -O1 |
| `bw_byte_coverage.c`   | coverage by bytes, not by writer kind: a `read` through a pointer that may point at another local, a one-byte indexed terminator, an escape of `&buf` and a copy at a constant offset each leave the bare path a frontier (`complete_slice: false`); a copy at offset 0 covers. `bw_byte_coverage_o1.c` rebuilds it at -O1 |
| `bw_stash_epoch.c`     | stash identity is flow-sensitive (#1034 item 1): a pointer loaded before `&buf` escaped and copied through a second slot, an overwritten slot and a restored slot are not `&buf`, so their `strcpy(.., getenv(..))` is never a writer or the `source` origin of `buf`; the direct stash is |
| `stash_pointer_width.c` | an 8-byte stash reloaded at 4 bytes and zero-extended is not the pointer (#1034 item 2): writes through it cover nothing; the full-width reload is a definite writer. `stash_pointer_width_o1.c` rebuilds it at -O1, where `snprintf` folds into short stores |
| `stash_must_paths.c`   | a reload that may still hold the stash's previous target (first do-while trip, untaken branch) never certifies the buffer; an escape after a finished loop does not make the loop's call a writer. `stash_must_paths_o1.c` rebuilds it at -O1 |
| `loop_escape_index.c`  | a loop-carried escape makes the helper that runs before it opaque on the next trip; an indexed byte of a memset buffer walks the buffer's writers beside the index. `loop_escape_index_o1.c` rebuilds it at -O1 |
| `bw_loop_frontier.c`   | a memory-chain proof cut (300 unrelated calls) keeps the loop-carried escaped call `compose` named, like the uncut control (#1034 item 3) |
| `bw_wide_load.c`       | loads are walked through the writers of the loaded bytes, never the pointer (#1041 item 2): the upper half of a wider field store is the input it came from, not the caller's malloc; a field stored on one path only, or overwritten through a second pointer, is a `field_load_unresolved` frontier. `bw_wide_load_o2.c` rebuilds it at -O2 (`sx.q([m + 0xc].d)` operands) |

## `EXPECTED.json` schema

```jsonc
{
  "lang": "c" | "cpp",
  "cflags": ["-O2", "-D_FORTIFY_SOURCE=2"],   // optional: extra compiler flags (e.g. to emit __*_chk)
  "forward":  [{"function","source","sink_classes":[...],"sinks":[{"callee","class","arg"}],"leaves":[{"kind"}],"all_clear": false}],
  // "resolve_map": {"in_function": <fn>, "target": <callee>},  // optional: pin the lone indirect call in <fn> to <callee> via --resolve-map (addresses resolved at test time). <fn> must contain EXACTLY ONE indirect call so the harness can resolve it unambiguously.
  // "assumptions_contain": ["substr", ...],                    // optional: each substring must appear in an assumptions entry
  "backward": [{"function","sink","origin_kinds":[...],"forbid_origin_kinds":[...],"leaves":[{"kind"}],"forbid_leaves":[{"kind"}],"leaf_details_contain":["substr"],"forbid_leaf_details_contain":["substr"],"step_reasons_contain":["substr"],"forbid_step_reasons_contain":["substr"],"complete_slice": false,"expect_crossed": "<fn>"}],
  "negative": [{"function","source","sink_classes":[...],"forbid_sink_classes":[...],"all_clear": true}],
  "callgraph":[{"function","expect_indirect": true}]
}
```

`sink_classes` (optional, forward/negative) is passed through as `--sink-class`
to enable opt-in sink classes (e.g. `file_write`).

## Real-world validation runs (done; not checked in)

The engine has been exercised on real targets beyond the synthetic corpus:

- **DVRF firmware services** (`socket_bof.c`, `socket_cmd.c` from
  github.com/praetorian-inc/DVRF, compiled natively). Forward taint from
  `arg:read:1` correctly found `read -> sprintf` (overflow) and the multi-hop
  `read -> snprintf -> system` (command injection) with a full provenance path.
  `multihop.c` above is the regression distilled from this.
- **Statically-linked stripped MIPS busybox** (busybox.net prebuilt). Surfaced a
  real gap: with no symbols, the symbol-keyed model DB does not fire, so sinks
  go unrecognized. Workarounds: apply BN signature libraries, rename identified
  libc functions (then models match by name), or supply address-keyed targets
  via `--resolve-map` for indirect dispatch. The value-set indirect resolution
  and structured primitives still work on stripped code.

## Tier 3 — firmware-style targets (opt-in, not checked in)

Cross-architecture realism (MIPS/ARM, statically-linked libc, dispatch tables)
is validated against public firmware on demand, not in CI. These assert
*honest degradation* (unresolved leaves reported, assumptions flagged) rather
than exact precision/recall, since ground truth is fuzzy. Recommended recipe:

1. Pull a router `httpd`/`cgi` from a public firmware image (e.g. via
   `binwalk -e <image>`), or use a known-CVE build.
2. `bn session start ./httpd` then, for each request handler:
   `bn taint forward -f <handler> --source arg:recv:1`
   `bn taint backward -f <handler> --sink arg:strcpy:1`
3. Confirm tainted flows into `system`/`strcpy`/`sprintf` are reported, and
   that indirect dispatch-table calls appear under `leaves` with
   `indirect_call_unresolved` rather than being dropped.
