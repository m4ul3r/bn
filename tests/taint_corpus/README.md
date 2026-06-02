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
| `sanitized.c`          | constant memcpy length (negative — no false positive)  |
| `command_injection.c`  | tainted buffer -> system() pointer arg (external model)|
| `indirect_call.c`      | tainted data into a function-pointer call (leaf, not dropped) |
| `indirect_resolve.c`   | const function-pointer table resolved by value-set; taint follows into both targets |
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

## `EXPECTED.json` schema

```jsonc
{
  "lang": "c" | "cpp",
  "cflags": ["-O2", "-D_FORTIFY_SOURCE=2"],   // optional: extra compiler flags (e.g. to emit __*_chk)
  "forward":  [{"function","source","sink_classes":[...],"sinks":[{"callee","class","arg"}],"leaves":[{"kind"}]}],
  "backward": [{"function","sink","origin_kinds":[...]}],
  "negative": [{"function","source","sink_classes":[...],"forbid_sink_classes":[...]}],
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
