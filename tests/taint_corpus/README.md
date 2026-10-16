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
| `bw_memory_paths.c`    | backward memory-path honesty: a conditional fill (memory phi with a writer-less arm) is an `address_of_local_unresolved` frontier; a source buffer overwritten AFTER it was copied is read at the copy's memory version; `&buf` escaping to a global makes a later unmodeled call an opaque writer and a later modeled call that reloads the stash (`strcpy(g_stash, host)`) a fill; a 1-byte terminator after a conditional `read` does not cover the bare arm; a `memset` arm is a literal fill, which since #931 covers no byte either |
| `strtok_escape.c`      | strtok state left tainted at return is disclosed (`tokenizer_state_escapes`) and withholds `safe_to_report_all_clear`; a clean non-NULL `strtok` that lies on EVERY path from the tainted call to the `strtok(NULL)` resets the continuation (negative); one under `if (mode)`, or a priming call before a loop that the back-edge bypasses (`tok_prime_loop`), does not, so the continuation stays a real `system()` flow (positive). `strtok_escape_o1.c` rebuilds it at -O1 |
| `bw_opt_shapes.c`      | -O1 optimiser shapes (`bw_opt_shapes_o2.c` rebuilds at -O2): a store through a phi'd / non-constant pointer is a writer whose value reaches the ascent (`buf[i] ^= key[i]`, `*p++ = *src++`), never a bare `source` with `complete_slice: true`; a byte-wise `*p++ = *src++` loop is a loop-only fill and stays a frontier leaf; `__builtin_memset` is a literal fill; `char buf[] = "lit"` lifted to `__builtin_strncpy` is modeled; a tainted `memcpy` length is a step and the fill outranks the terminator |
| `bw_global_operand.c`  | a `CONST_PTR` fill operand is a literal only in read-only storage: `sprintf(cmd, "%s", g_host)` from a `.bss` array is a frontier leaf with origin `global` (`complete_slice: false`); the `.rodata` literal twin has no global leaf (since #931 its `sprintf` fill still leaves the buffer unproven) |
| `bw_offset_address.c`  | an address built on a buffer at a VARIABLE offset is that buffer: `popen(line + off)` and `strcpy(buf, &other[i])` walk the `read` that filled it beside the offset; `g_hosts[i]` (a `.bss` array element) is a `global` frontier leaf. `bw_offset_address_o1.c` rebuilds it at -O1 |
| `bw_byte_coverage.c`   | coverage by bytes, not by writer kind: a `read` through a pointer that may point at another local, a one-byte indexed terminator, an escape of `&buf` and a copy at a constant offset each leave the bare path a frontier (`complete_slice: false`); since #931 a copy at offset 0 does not cover either -- no call proves a byte. `bw_byte_coverage_o1.c` rebuilds it at -O1 |
| `bw_stash_epoch.c`     | stash identity is flow-sensitive (#1034 item 1): a pointer loaded before `&buf` escaped and copied through a second slot, an overwritten slot and a restored slot are not `&buf`, so their `strcpy(.., getenv(..))` is never a writer or the `source` origin of `buf`; in `direct` the getenv call forgets the slot too, so its strcpy is a frontier |
| `stash_pointer_width.c` | an 8-byte stash reloaded at 4 bytes and zero-extended is not the pointer (#1034 item 2): writes through it cover nothing; the full-width reload is a definite writer (named in the steps; since #931 its snprintf proves no byte, so the slice is still incomplete). `stash_pointer_width_o1.c` rebuilds it at -O1, where `snprintf` folds into short stores |
| `stash_must_paths.c`   | a reload that may still hold the stash's previous target (first do-while trip, untaken branch) never certifies the buffer; the strcpy filling buf before a loop is itself an escape (it is handed `&buf`), so the loop's unmodeled call is a possible writer. `stash_must_paths_o1.c` rebuilds it at -O1 |
| `loop_escape_index.c`  | a loop-carried escape makes the helper that runs before it opaque on the next trip; an indexed byte of a memset buffer walks the buffer's writers beside the index. `loop_escape_index_o1.c` rebuilds it at -O1 |
| `bw_loop_frontier.c`   | a memory-chain proof cut (300 unrelated calls) keeps the loop-carried escaped call `compose` named, like the uncut control (#1034 item 3) |
| `bw_wide_load.c`       | loads are walked through the writers of the loaded bytes, never the pointer (#1041 item 2): the upper half of a wider field store is the input it came from, not the caller's malloc; a field stored on one path only, or overwritten through a second pointer, is a `field_load_unresolved` frontier. `bw_wide_load_o2.c` rebuilds it at -O2 (`sx.q([m + 0xc].d)` operands) |
| `bw_slot_identity.c`   | slot identity is exact (#1053 review 1): a store through the low 32 bits of `slot` is not a store to `[slot]`, so the strcpy through the full reload is a may-writer; the full-slot control names strcpy as a definite writer (incomplete since #931, which trusts no call's extent). Built at -O2 |
| `bw_fill_offset.c`     | recv is never certified as a field's producer (#1053 review 2): `recv(fd, p->hdr, 8)` before a `p->len` load leaves the length a `field_load_unresolved` frontier, with or without an earlier `p->len = n` store (base's verdict) |
| `bw_stash_overwrite.c` | `memcpy(&stash, &otherptr, 8)` forgets the remembered `&buf` (#1053 review 3): the later `strcpy(stash, ..)` is an opaque frontier, never a writer or `source` origin of `buf` |
| `bw_global_load.c`     | a fixed-address load of a writable global nothing in the function wrote is a `field_load_unresolved` frontier (#1053 review 4) |
| `bw_frame_outputs.c`   | no call is proof that a field was not written (#1053 rounds 1-3): recv (covering, disjoint, short, overlapping), recvmsg, readv, getline, stat, sscanf, asprintf, strtol and strncat each leave a `field_load_unresolved` frontier; a modeled source is named as a possible producer. `bw_frame_outputs_o2.c` rebuilds it at -O2 |
| `bw_frame_stash.c`     | remembered stash slots are forgotten at every call: strtol/asprintf/memmove/partial-memcpy rewrites of the slot are opaque frontiers, never a named writer of `buf`; the full-stash controls are conservatively incomplete too. `bw_frame_stash_o2.c` rebuilds it at -O2 |
| `bw_frame_append.c`    | strncat (appends at dst + strlen) and the strncpy control both leave a field inside the string a frontier: no call's write extent is trusted |
| `bw_frame_strtok.c`    | strtok rewrites the delimiter its model never names: a possible writer of the length field. Built at -O2 |
| `bw_call_planted.c`   | a call handed `&buf` may retain or plant a pointer into it (#1053 review 4): after `strtok_r(buf, ",", &g_save)` or `strtol(buf, &end, 10)` a reload of the saved pointer -- with or without an explicit stash -- is a may-alias, so the strcpy through it is a frontier. `bw_call_planted_o2.c` rebuilds it at -O2 |
| `bw_readonly_index.c`  | `malloc(readonly_sizes[i & 3])`: a bounded index into a read-only table is a `field_load_unresolved` frontier with `i` walked -- a precision cost since #1053 review 6 (base was complete at -O0 through its old pointer walk). `bw_readonly_index_o2.c` rebuilds it at -O2 |
| `bw_table_bounds.c`    | constant-table loads are frontiers whatever the index: an unchecked byte offset and unmasked `long`/`int` indices that may reach the mutable global after the table, and masked and compare-bounded indices too (#1053 reviews 5 and 6). `bw_table_bounds_o2.c` rebuilds it at -O2 |
| `bw_table_relro.c`     | a `.data.rel.ro` table in a PIE linked with `-z norelro` is in a writable segment, and a function writes its first entry: the masked load is a frontier. `bw_table_relro_o2.c` rebuilds it at -O2 |
| `bw_table_loader.c`    | loader-written table entries: `.rodata` tables in a PIE linked with `-z notext` whose entries the loader writes (a local IFUNC's `IRELATIVE` result, an external symbol's `R_X86_64_64`); at -O0 they sit in a non-writable segment under `DT_TEXTREL`, and the masked loads are frontiers. `bw_table_loader_o2.c` rebuilds it at -O2, where the toolchain here emits a writable `.rodata` in an RW segment with no `DT_TEXTREL` -- the layout is not enforced, and the -O0 build carries the non-writable case |
| `bw_call_planted_store_init.c` | the planting call isolated (N1): the buffer is filled by plain stores, strtol is the only call before the reload of its end pointer (global or local slot), and the strcpy through it is a frontier. `bw_call_planted_store_init_o2.c` rebuilds it at -O2 |
| `bw_store_regions.c`   | #931/#933: only plain stores prove the consumed region, from the sink's pointer to the end of the object. A 16-byte literal stored into the last member is complete (BN's `__builtin_strcpy`, whose LLIL is that store run); the same stores read through the whole struct, a member followed by another member, a 15-byte literal in a 16-byte buffer and a real `strncpy` call leave the unproven bytes in the leaf. `bw_store_regions_o2.c` rebuilds it at -O2, where the constant `memset` becomes a verified `__builtin_memset` and covers |
| `bw_alias_followups.c` | #1053 review follow-ups S1/S2 on a store-initialized buffer: a pointer carried in a struct field, the pointer `strsep` returns, and `strtok_r` continuing are possible writers (the strcpy through them is a frontier), never unrelated; the no-pointer control stays complete. `bw_alias_followups_o2.c` rebuilds it at -O2, where `si_struct_write` folds to a direct `strcpy(&u, host)` after the store and is complete with the parameter origin |
| `bw_regions.c`         | from #941, verdicts re-derived: zero, short, full and suffix `read`s, an indexed row, a lone `strcat`, member `snprintf`/`memcpy` fills read by `popen` or `write` -- every call fill is incomplete with `calls` and the `unproven` byte ranges of the consumed region. `bw_regions_o1.c`/`bw_regions_o2.c` rebuild it at -O1/-O2 |
| `scalar_member_regions.c` | from #941: a load of one member consumes only its 8 bytes -- the stored member is complete, the unwritten one is a frontier |
| `store_member_identity.c` | from #941: a `snprintf` through a phi of `&buf` and a member reload is never proof (all incomplete, the control included, since #931); a store through a truncated pointer writes no byte of buf |
| `external_member_regions.c` | from #941: field loads through a parameter pointer keep their store-matched origin or frontier (unchanged by the region model). `external_member_regions_o1.c` rebuilds it at -O1 |

Backward completeness since #931/#933: a local buffer is checked over the
region the sink consumes, and only plain stores (plus BN's `__builtin_*`
outlined store runs whose LLIL the walk checks) prove it initialised. No call
does -- so every target whose buffer a call fills (`snprintf`, `strcpy`,
`memcpy`, `read`, ...) expects `complete_slice: false`, with a leaf naming the
call (`calls`), the consumed `region` and the `unproven` ranges. That is the
deliberate cost; call-write extents are a follow-up.

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
