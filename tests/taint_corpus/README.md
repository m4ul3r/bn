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
| `interproc.c`          | tainted buffer crosses a call boundary; sink lives in the callee (interprocedural descent) |
| `vtable.cpp`           | C++ virtual dispatch shows as an indirect callee (honest degradation) |

## `EXPECTED.json` schema

```jsonc
{
  "lang": "c" | "cpp",
  "forward":  [{"function","source","sinks":[{"callee","class","arg"}],"leaves":[{"kind"}]}],
  "backward": [{"function","sink","origin_kinds":[...]}],
  "negative": [{"function","source","forbid_sink_classes":[...]}],
  "callgraph":[{"function","expect_indirect": true}]
}
```

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
