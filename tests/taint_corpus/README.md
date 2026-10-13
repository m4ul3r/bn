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
| `bw_opt_shapes.c`      | -O1 optimiser shapes (`bw_opt_shapes_o2.c` rebuilds at -O2): a store through a phi'd / non-constant pointer retains its input and caller ascent; byte-wise cursor copies remain frontiers where loop initialization cannot be proved; `__builtin_memset` is a literal fill; `char buf[] = "lit"` lifted to `__builtin_strncpy` has concrete constant-data NUL evidence; a tainted `memcpy` length remains an input rather than being hidden by the terminator |
| `bw_global_operand.c`  | a `CONST_PTR` fill operand is a literal only in read-only storage: `sprintf(cmd, "%s", g_host)` from a `.bss` array is a frontier leaf with origin `global` (`complete_slice: false`); the `.rodata` literal twin is complete |
| `bw_offset_address.c`  | an address built on a buffer at a VARIABLE offset is that buffer: `popen(line + off)` and `strcpy(buf, &other[i])` walk the `read` that filled it beside the offset; `g_hosts[i]` (a `.bss` array element) is a `global` frontier leaf. `bw_offset_address_o1.c` rebuilds it at -O1 |
| `bw_byte_coverage.c`   | coverage by bytes, not by writer kind: a `read` through a pointer that may point at another local, a one-byte indexed terminator, an escape of `&buf` and a copy at a constant offset each leave the bare path a frontier (`complete_slice: false`); a copy at offset 0 covers. `bw_byte_coverage_o1.c` rebuilds it at -O1 |
| `bw_regions.c`        | #931/#933: zero/short/full-request reads, suffix/indexed rows and append-only buffers remain incomplete; initialized struct-member strings and exact byte ranges are covered without requiring unrelated fields. `bw_regions_o1.c` / `bw_regions_o2.c` rebuild at -O1/-O2 |
| `strtok_fixpoint.c`   | #932: two-entry/two-back-edge tokenizer CFG with two unused-return initializers; the NULL continuation reaches `system` even when IL enumerates it first. O0/O1/O2 variants share the source |
| `loop_exit.c`        | #950: exact synthetic -O1 loop with filled/unwritten exits in both source orders; cyclic memory-PHI memoization must retain an unresolved entry path |
| `loop_escape_index.c` | loop-bottom address escape makes the next iteration's earlier helper an opaque writer; indexed scalar LOAD preserves parameter origin and caller ascent at O0. `loop_escape_index_o1.c` checks the escape at O1 |
| `stash_must_paths.c` | do-while and conditional stash reloads must not certify the unwritten first/alternate path; a stash after a finished loop leaves it covered. O0 pointer-copy byte LOAD is covered; O1 variant checks stash paths |
| `stash_pointer_width.c` | exact narrow/full pointer reload controls at O0/O1; partial reload plus zero-extension must not certify initialization. Copied LOAD epoch control is checked at O0; O1 may fold the scalar sink |
| `nested_escape_paths.c` | #951: native O0 nested call/escape arrangements require an opaque helper leaf; no-escape nests stay covered. Includes single-loop and differing-offset cache controls |

The region-coverage cases deliberately distinguish a may-source from a proven
initializer. Existing `read`-based offset/append/loop cases still require their
source steps, but now require an incomplete slice: the request length does not
prove the return count or NUL termination. A zero-filled branch cannot certify
the alternative short/error-read branch.

`bw_regions` keeps the modeled library-call boundary at every optimization
level with `-fno-builtin` and an explicit `_FORTIFY_SOURCE=0`. Some compilers
enable fortify by default; its header wrappers can still expand constant
copies into vector loads/stores despite `-fno-builtin`. General vector-load
provenance and multi-store interval unions are not claimed by this modeled-fill
fixture. The separate `fortified` and `bw_opt_shapes` cases retain their own
fortified/inlined-operation coverage.

`bw_opt_shapes.literal_init` is a covered-local control, not an obsolete
completion pin: C initializes all 64 bytes of `char buf[64] = "ping "`, and
`strcat` preserves its string. Optimized IL may instead copy into `buf + 5`;
that remains covered because the preceding five bytes were initialized.
The caller's unresolved `argv` field still withholds whole-slice completeness.
`read_or_memset` cannot claim the same proof: the `read` arm can leave a gap
before `buf[63] = 0`. The `ptr_copy` loop requires an induction proof the
walker does not provide, rather than whole-array initialization at loop entry.

`origin_kinds` is optional when a case checks region coverage rather than the
choice of one representative origin. A zero/disjoint write is not evidence for
the consumed bytes, and an indexed address also depends on its selector;
those cases retain explicit frontier and completeness assertions without
pinning one path. Overlapping nonzero reads and copy/member cases still check
their source/parameter origins. `string_not_full_bytes` must remain incomplete:
`snprintf`'s capacity is not the number of bytes it initializes.

`loop_exit` is compiled with `-O1 -fno-inline`, using the exact issue #950
source. It is analyzed only: do not execute its intentionally uninitialized
`popen` arguments. Compiler-dependent IL coverage is complemented by deterministic
memory-SSA tests for both exit-arm orders, `memset`/`strcpy` writers, initialized
entry controls, and shared-budget exhaustion. Cyclic results that consulted a
neutral backedge are not cached for other exit arms; revisits remain bounded
by the same proof allowance. This does not claim arbitrary loop induction.

`loop_escape_index` retains the review's invented probe source. Analyze it
only; its helper can copy an unbounded environment string. The O0 index case
checks parameter origin and ascent without certifying unknown-offset bytes.
At O1 the compiler can fold the LOAD away, so that variant checks only the
loop-carried escape. Deterministic memory-SSA controls cover both PHI orders,
unescaped initialized loops with unrelated unmodeled calls, and shared-budget
exhaustion on escape-state revisits.

`stash_must_paths` retains the round-two review's invented source and is
analyzed only: the do-while and conditional-stash paths intentionally consume
uninitialized bytes. O0/O1 cases distinguish must-reaching initialization from
may-reaching provenance, including inlined `strcpy` stores. Deterministic
tests cover both PHI orders, agreeing/disagreeing stash values, bare-entry,
opaque-call, partial/unknown-width arms, initialized loop entries, and bounded
must-query work. Retry evidence stays deduplicated and retains known parameter
sources even when the shared allowance is exhausted. Copied pointer LOAD
controls use nonzero offsets and preserve index origins without adding an
unrelated unknown-region demand.

`stash_pointer_width` retains the Sol review's exact invented source and
compiler flags (`-fno-builtin -fno-strict-aliasing -fno-inline`). Analyze only:
the narrow reload does not reconstruct the stack pointer. Full-width STORE/
LOAD controls remain covered; deterministic 32/64-bit, subfield, extension,
cast/operator and unknown-ABI controls require conservative pointer identity.
The copied-base epoch defect was reproduced synthetically, not natively:
O0 folds the base copy and O1 can make the scalar sink bounded/unseeded.
The deterministic test excludes post-load writer provenance.

`nested_escape_paths` retains the Opus review's exact invented source.
Analyze only: helper copies may overflow. Both O0 inner-call/outer-escape and
outer-call/inner-escape arrangements must name the opaque helper, while their
no-escape controls stay covered. Both mocked PHI orders complement the native
compiler shape. Overlapping cycles share escape state under the same allowance;
sequential loops remain separate. Moving inline/SSA stash values stop as unknown
without exhausting the allowance, and transient-operand cache tests check
object lifetime rather than relying on nondeterministic allocator id reuse.

## `EXPECTED.json` schema

```jsonc
{
  "lang": "c" | "cpp",
  "cflags": ["-O2", "-D_FORTIFY_SOURCE=2"],   // optional: extra compiler flags (e.g. to emit __*_chk)
  "forward":  [{"function","source","sink_classes":[...],"sinks":[{"callee","class","arg"}],"leaves":[{"kind"}],"all_clear": false}],
  // "resolve_map": {"in_function": <fn>, "target": <callee>},  // optional: pin the lone indirect call in <fn> to <callee> via --resolve-map (addresses resolved at test time). <fn> must contain EXACTLY ONE indirect call so the harness can resolve it unambiguously.
  // "assumptions_contain": ["substr", ...],                    // optional: each substring must appear in an assumptions entry
  "backward": [{"function","sink","origin_kinds":[...],"forbid_origin_kinds":[...],"leaves":[{"kind"}],"forbid_leaves":[{"kind"}],"leaf_details_contain":["substr"],"step_reasons_contain":["substr"],"complete_slice": false,"expect_crossed": "<fn>"}],
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
