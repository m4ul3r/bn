# Taint/Trace Engine Audit

Date: 2026-06-15

> **Sanitized.** All target names, paths, instance IDs, symbol names, and addresses below are
> invented mock data that reproduce the observed engine behavior on a stand-alone example. They do
> not correspond to any real binary. Citations into the `bn` source tree are real.

Scope: audit the `bn taint` and `bn trace` engines, dogfood them against firmware, and report
usability and accuracy issues. The primary deep target was a stripped aarch64 protocol-parser daemon
(`parserd`); the expanded coverage pass sampled 10 additional binaries covering network daemons,
system services, object-transfer services, and proprietary services.

## Executive Summary

The taint engine is useful on real firmware. On `parserd`, forward taint found a non-trivial
interprocedural parser flow from `tlv_parse` into a `memcpy` length at `0x10f8c`, crossing
`msg_parse`, a helper tailcall, and `parse_oob_record`. On `devmgrd`, forward and backward taint
both connected a `read` return value to a `memcpy` length at `0x11514`. The newer frontier reporting
is also materially better than a bare all-clear: fresh sessions now emit `unmodeled_callee`,
`indirect_call_unresolved`, and `coarse_memory_store` leaves when taint reaches places the engine
cannot follow.

The biggest remaining accuracy gaps are around source modeling and heap/field provenance. A manual
`--source arg:recvfrom:1` run on `sub_10a40` reports no sinks, while `--source ret:recvfrom` on
the same function reaches a `memcpy` length sink at `0x10a5c`. Backward taint and trace also stop
at allocator returns for parser-struct fields, even when the relevant lengths are field loads from
records populated from attacker-controlled input.

The biggest usability gap is output trust and volume. Fresh sessions show the current attribution
and frontier features, but older live BN instances can keep stale in-process taint code while
`doctor` still reports `stale_plugin_code:false`. On large parser paths, forward taint is honest but
very noisy: useful sinks and frontiers are mixed with many framework fallback assumptions and coarse
memory-store leaves.

## Verification

Tests run:

```sh
PYTHONPATH=src pytest tests/test_taint_engine.py tests/test_dataflow.py -q
# 113 passed

/opt/bn/.venv/bin/python -m pytest tests/test_taint_integration.py -q
# 16 passed in 45.42s
```

Environment note: the integration tests must be run through the repo venv. The harness resolves the
CLI as a sibling of `sys.executable`; under system Python it looked for `/usr/bin/bn`, which is not
present in this environment.

Implementation notes checked:

- `plugin/bn_agent_bridge/taint_engine.py` implements interprocedural forward taint, per-callsite
  attribution, `unmodeled_callee` leaves, and coarse memory-store leaves.
- `src/bn/formatters.py` renders `by_source` and `unmodeled_callee`.
- `plugin/bn_agent_bridge/taint_models.json` models libc sources/sinks, but lacks common GLib
  models such as `g_try_malloc0`, `g_free`, `g_strndup`, `g_strdup_printf`, and `g_slist_*`.
- `FORWARD_TAINT_DESIGN.md` and the module header in `taint_engine.py` are stale relative to the
  current implementation.

## Firmware Targets

| Target | Role | Arch | Functions | Imports | Use |
| --- | --- | ---: | ---: | --- | --- |
| `parserd` | protocol/record parser daemon | aarch64 | ~940 | ~530 | Primary deep taint/trace target |
| `linksvcd` | link-management daemon | aarch64 | ~2300 | ~340 | Secondary import/source/sink sample |
| `proxyd` | socket proxy | aarch64 | ~80 | ~60 | Secondary small target sample |
| `xferd` | object-transfer daemon | aarch64 | ~1290 | ~310 | Object-transfer source and sink sample |
| `devmgrd` | device-manager system service | aarch64 | ~1180 | ~250 | System service taint and trace sample |
| `httpd` | web server | aarch64 | ~630 | ~155 | Network daemon source/frontier sample |
| `svc-alpha` | proprietary service | aarch64 | ~750 | ~220 | Proprietary source attribution sample |
| `cfgd` | network-config daemon | aarch64 | ~810 | ~350 | Config-daemon source smoke sample |
| `discoveryd` | service-discovery daemon | aarch64 | ~420 | ~260 | Discovery source/frontier sample |
| `svc-beta` | proprietary service | aarch64 | ~450 | ~100 | Proprietary read-source sample |
| `svc-gamma` | proprietary service | aarch64 | ~515 | ~205 | Proprietary read-source sample |

Representative import/callsite surface (illustrative):

- `parserd`: `recvfrom` has 1 code ref, `recv` has 15 refs across 7 functions, `memcpy` has 56 refs.
- `linksvcd`: `recv` has 3 refs across 2 functions, `memcpy` has 84 refs across 72 functions.
- `proxyd`: `splice` has 2 refs, `memcpy` has 1 ref.
- `xferd`: `recv` has 1 ref, `memcpy` has 26 refs across 23 functions, `sprintf` has 2 refs.
- `devmgrd`: `read` has 11 refs across 9 functions, `fread` has 2 refs, `memcpy` has 31 refs,
  and `strcpy` has 10 refs.
- `httpd`: `read` has 9 refs across 8 functions, `memcpy` has 25 refs, `memmove` has 6 refs.
- `svc-alpha`: `read` has 7 refs across 4 functions, `fread` has 2 refs, `memcpy` has 30 refs,
  and `strcpy` has 1 ref.
- `cfgd`: `read` has 3 refs across 2 functions, `memcpy` has 68 refs across 40 functions.
- `discoveryd`: `read` has 9 refs across 8 functions, `memcpy` has 3 refs across 2 functions.
- `svc-beta`: `read` has 1 ref in `sub_10d50`.
- `svc-gamma`: `read` has 6 refs across 6 functions.

This is a meaningful firmware dogfood sample, but it is still not a whole-image exhaustive
audit. The expanded pass is enough to exercise the taint/trace engine across multiple families and
failure modes; it is not a claim that every input-facing binary in the image was audited.

## Dogfood Results

### Flow A: TLV parser to parser-copy sink

Command:

```sh
bn --instance audit-parserd -t parserd taint forward \
  -f tlv_parse --source param:0 --max-depth 3 --format json
```

Result:

- Found 1 sink: `memcpy` at `0x10f8c`, `overflow_len`, tainted arg 2.
- Path begins with the TLV input pointer in `tlv_parse`, derives TLV length from bytes near
  `arg1`, calls `msg_parse` at `0x10d24`, then reaches an OOB-record parser helper and
  a length derived from `[x19 + 0x10].b - 1`.
- Stats: `functions_visited=13`, `max_depth=3`, `sinks=1`, `truncated=true`.
- Fresh session output included many honest frontiers: `coarse_memory_store` leaves and
  `unmodeled_callee` leaves for depth-bound calls such as `sub_11200`, `sub_10e10`,
  `log_debug`, and `log_error`.

Assessment: good recall for an interprocedural real-firmware flow, but high noise. The result is
actionable only after filtering the repeated coarse memory leaves and GLib fallback assumptions.

### Flow B: Backward taint from parser memcpy lengths

Command:

```sh
bn --instance audit-parserd -t parserd taint backward \
  -f msg_parse --sink arg:memcpy:2 --max-depth 2 --format json
```

Result:

- Seeded 3 `memcpy` length sites: `0x10408`, `0x10650`, `0x108e4`.
- All 3 slices ended at `g_try_malloc0` allocation returns.
- Example: `0x10650` traces `x2_28` to `[x3_10 + 4].b`, but then follows `x3_10` back to
  `g_try_malloc0(0x20)`.
- Leaves and assumptions were empty.

Assessment: accurate as a local SSA slice, but misleading for vulnerability triage. The user needs
to know that "origin: g_try_malloc0" really means "field on an allocated parser record; inspect
writes to that field", not that the allocator is the semantic origin of the length.

### Flow C: Exact trace on parser memcpy lengths

Commands:

```sh
bn --instance audit-parserd -t parserd trace msg_parse 0x10408 --arg 2 --format json
bn --instance audit-parserd -t parserd trace msg_parse 0x10650 --arg 2 --format json
```

Result:

- `0x10408`: `x2_2#94 = [x20_1#4 + 0x50].q`, then a phi over two record allocations.
- `0x10650`: `x2_28#23 = zx.q(x4_4#4)`, `x4_4#4 = zx.d([x3_10#11 + 4].b)`, then allocation.
- JSON uses verbose `"<SSAVariable: x version y>"` strings and `reason:null` for most normal
  definition steps.

Assessment: trace is good at exact local register provenance, but it needs field-load semantics and
cleaner JSON labels to be ergonomic during audit.

### Flow D: `recvfrom` source selection

Commands:

```sh
bn --instance audit-parserd -t parserd taint forward \
  -f sub_10a40 --source arg:recvfrom:1 --max-depth 3 --format json

bn --instance audit-parserd -t parserd taint forward \
  -f sub_10a40 --source ret:recvfrom --max-depth 3 --format json
```

Result:

- `arg:recvfrom:1` reached no sinks and emitted no assumptions or leaves.
- `ret:recvfrom` reached `memcpy` at `0x10a5c`, tainted arg 2, through `sub_10a10`.
- Exact trace of the `recvfrom` call showed:
  - arg 1 buffer: `buf#3 = [x19 + 0x18]`, where `x19 = arg1`
  - arg 2 length: `len#3 = sx.q([x19 + 0x10].d)`, where `x19 = arg1`

Assessment: this is the most important UX/accuracy problem. The model DB correctly knows `recvfrom`
has both `*arg:1` and `ret` sources, but the CLI makes the analyst pick one. Picking the buffer
source gives a clean-looking all-clear even though the return value from the same call drives a copy
length sink.

### Flow E: Per-callsite attribution

Command on a fresh session:

```sh
bn --instance audit-parserd -t parserd taint forward \
  -f sub_10c00 --source arg:recv:1 --max-depth 2 --format json
```

Result:

- Fresh output included `by_source` for `0x10c3c`, `0x10d14`, and `0x10ce8`.
- The assumptions state that 3 callsites were analyzed independently.

However, an older live `audit-parserd-stale` instance returned the previous behavior for the same
command:

```json
{
  "assumptions": ["3 callsites of recv; seeded from all"],
  "reached_sinks": [],
  "leaves": []
}
```

Both the older instance and the fresh instance reported `stale_plugin_code:false` through `doctor`.

Assessment: the engine feature works in a fresh process, but live-session staleness can invalidate
dogfood results while the health check looks clean.

### Flow F: Expanded firmware coverage pass

The first draft of this report only had one deep binary and two light samples. I expanded the pass
with fresh BN sessions and focused source/sink commands across additional binaries:

| Binary | Command focus | Result | Engine lesson |
| --- | --- | --- | --- |
| `xferd` | `sub_12300`, `arg:recv:1` and `ret:recv` | No sinks/leaves | Clean single-callsite source run; not every imported source produces useful frontier evidence. |
| `devmgrd` | `sub_11480`, `arg:read:1` | No sinks; 5 leaves, including `unmodeled_callee` and `coarse_memory_store` | Fresh frontier honesty works on non-GLib system code too. |
| `devmgrd` | `sub_11480`, `ret:read` | `memcpy` length sink at `0x11514`; backward slice origin is `read` | Source-to-sink works, but the engine does not classify the length as read-bounded by `0xfff`. |
| `httpd` | `resp_read`, `arg:read:1` | No sinks/leaves | Buffer-only source can be quiet even in a network parser. |
| `httpd` | `resp_read`, `ret:read` | No sinks; `coarse_memory_store` leaves and `indirect_call_unresolved` at `0x10da8` | Return-value source exposes frontiers that buffer-only source misses. |
| `svc-alpha` | `sub_10aa0`, `arg:read:1` and `ret:read` | No sinks; `by_source` for three read callsites | Per-callsite attribution works in a proprietary stripped service. |
| `cfgd` | `sub_10280`, `ret:read` | No sinks; unmodeled `log_fatal` fallback | Framework/local helper models are needed beyond libc. |
| `discoveryd` | `sub_10110`, `arg:read:1` and `ret:read` | No sinks; `lib_*` unmodeled assumptions and coarse global stores | Missing framework models create noise outside GLib too. |
| `svc-beta` | `sub_10d50`, `ret:read` | CLI error: return value discarded | Good user-facing error; suggests `arg:<n>` when ret cannot seed. |
| `svc-beta` | `sub_10d50`, `arg:read:1` | No sinks; C++ logging fallbacks | C++ local/framework functions need benign/noise-cut models. |
| `svc-gamma` | `sub_10b30`, `arg:read:1` and `ret:read` | No sinks/leaves | Clean smoke sample on another proprietary service. |

Assessment: the broader pass reinforces the main findings rather than changing them. The engine is
good at finding concrete source-to-sink flows and at surfacing stopped taint in fresh sessions. Its
practical weaknesses remain source presets, boundedness classification, structured-field
provenance, framework model coverage, and output volume.

## Findings

### High: Source selection can create false all-clears for receive APIs

Evidence: `sub_10a40` with `--source arg:recvfrom:1` reports no sinks/leaves, while
`--source ret:recvfrom` reaches `memcpy` length sink `0x10a5c` through `sub_10a10`.

Impact: analysts commonly think in terms of "data from recvfrom", not "only the buffer argument" or
"only the return value". A source locator that covers only one half of the API can produce a
dangerous false negative.

Recommendation:

- Add a source preset that expands a modeled source function into all source outputs, for example
  `--source call:recvfrom` or `--source model:recvfrom`.
- When a user chooses one source output for a modeled API that has more than one source output,
  print a hint such as: `recvfrom also models ret as source; consider --source ret:recvfrom`.
- Consider an audit mode that seeds all modeled source outputs for named input APIs by default.

### High: Backward taint and trace stop at allocation for structured parser fields

Evidence: backward slices from `msg_parse` memcpy lengths all report `origin:
g_try_malloc0`. Exact trace confirms field loads such as `[x20_1 + 0x50]`, `[x3_10 + 4]`, and
`[x3_10 + 0x10]` before terminating at allocation.

Impact: the slice is locally true but semantically incomplete. For parser structs, the interesting
origin is the write that populated the field from attacker input. Stopping at allocation hides that
link and can mislead triage.

Recommendation:

- Add backward support for field-load provenance: when tracing `load(base + const)`, search reaching
  stores to the same base/offset through memory SSA where available.
- If precise recovery is not possible, emit a leaf such as `field_load_unresolved` with base,
  offset, and a note to inspect writes to that field.
- Normalize this in both `bn trace` and `bn taint backward`, since both hit the same limitation.

### Medium: Read-bounded lengths are reported as generic overflow lengths

Evidence: in `devmgrd` `sub_11480`, `ret:read` reaches `memcpy` at `0x11514`. Backward taint
and exact trace both show the length is directly the return value of `read(fd, buf, 0xfff)`.

Impact: the flow is real, but the result class is too broad. A `read` return is attacker-influenced
but also bounded by the requested count and by the destination/source buffers around the callsite.
Reporting it only as `overflow_len` forces the analyst to manually distinguish a likely safe
bounded copy from a dangerous untrusted length.

Recommendation:

- Add a bounded-source length class for `read`/`recv` returns when the source call has a constant
  maximum count and the sink length is directly copied from that return.
- Include the source call's maximum count in the finding, for example `source_bound: 0xfff`.
- Where destination/source object sizes are recoverable, downgrade to a bounded or informational
  class when the copy is provably within the same buffer region.

### Medium: Forward taint is honest but too noisy on real parser code

Evidence: `tlv_parse --source param:0 --max-depth 3` found a real sink, but also emitted dozens
of assumptions and leaves. Repeated `coarse_memory_store` leaves and missing GLib models dominate
the result.

Impact: the engine is no longer silently dropping many frontiers, which is good, but the signal is
hard to scan. A user can miss the one sink among repetitive frontier records.

Recommendation:

- Group repeated leaves by kind and callee/address pattern in the text renderer.
- Add counts plus a short top-N view, with full JSON preserved.
- Model common GLib routines:
  - benign/no-return-taint: `g_free`, `g_slist_free_full`
  - allocation sinks without return taint: `g_try_malloc0`, `g_try_realloc`
  - string/list propagation: `g_strndup`, `g_strdup_printf`, `g_slist_append`, `g_slist_nth_data`
- Consider downgrading logging helpers such as `log_debug`/`log_error` to benign in per-target
  override models when they are local wrappers with no security sink behavior.

### Medium: Live-session staleness is not reliably detected by `doctor`

Evidence: an existing `audit-parserd-stale` session returned old multi-callsite
behavior (`seeded from all`) while a fresh `audit-parserd` session returned `by_source`.
`doctor` reported `stale_plugin_code:false` for both.

Impact: dogfood and audit results can be wrong if an instance keeps old Python modules loaded. This
is especially risky for taint/trace because small implementation changes alter result contracts.

Recommendation:

- Add an in-process taint-engine implementation fingerprint to `doctor`, not only plugin build id.
- Include loaded module path, mtime, and hash for `taint_engine.py` in `doctor --format json`.
- Provide or document a one-command `bn session restart <id>` workflow for plugin-code changes.

### Medium: Trace JSON is precise but not analyst-friendly

Evidence: trace output uses `"<SSAVariable: x version y>"`, many `reason:null` entries, and no
explicit `field_load` reason for the key parser-field loads.

Impact: machine consumers and humans both have to parse IL text to recover the actual SSA label and
field offset. This slows triage and makes reports harder to compare with taint paths.

Recommendation:

- Add a stable `ssa_label` field such as `x2_28#23` while preserving `ssa_var` for compatibility.
- Populate normal reasons such as `definition`, `phi_source`, `field_load`, and
  `call_or_jump_boundary`.
- For `field_load`, add `base`, `offset`, and `width` when recoverable.

### Low: JSON limiting for xrefs is intentionally rejected but inconvenient

Evidence: `xrefs --limit --format json` errors with `--limit only applies to --format text`.
Tests cover this behavior, so it is intentional.

Impact: firmware triage often wants bounded JSON for quick programmatic sampling. The current
behavior forces either text parsing or full JSON output.

Recommendation: support `--limit` for JSON xrefs as an output cap, or rename the current flag to
make its text-only nature obvious.

### Low: Documentation is stale relative to the engine

Evidence:

- `plugin/bn_agent_bridge/taint_engine.py` still describes an MVP intraprocedural/single-function
  engine with interprocedural stepping deferred.
- `FORWARD_TAINT_DESIGN.md` says `by_source`/`unmodeled_callee` have "No code yet", while code and
  tests now implement them.

Impact: maintainers and auditors can misread the engine's actual guarantees and result contract.

Recommendation: update the module header and move `FORWARD_TAINT_DESIGN.md` to an implemented or
historical-design status with current caveats.

## Scorecard

| Area | Rating | Notes |
| --- | --- | --- |
| Forward source-to-sink recall | Good | Found deep `parserd` parser flow and direct `devmgrd` `read` -> `memcpy` flow. |
| Boundedness classification | Medium/Weak | `read(fd, buf, 0xfff)` return into `memcpy` is reported as generic `overflow_len`. |
| Frontier honesty | Good in fresh sessions | `unmodeled_callee`, `indirect_call_unresolved`, and `coarse_memory_store` leaves surface stopped flows. |
| Per-callsite attribution | Good in fresh sessions | `by_source` works, but stale sessions can hide it. |
| Backward structured-data provenance | Weak | Field loads from allocated parser records stop at allocator origins. |
| Trace precision | Good locally | Exact SSA provenance works, but field semantics and JSON labels need improvement. |
| Model coverage | Medium | Libc coverage is useful; GLib, framework, C++ logging, and local helpers create noisy fallbacks. |
| Audit ergonomics | Medium | Powerful commands, but output volume, source-preset gaps, and stale sessions hurt trust. |

## Recommended Next Work

1. Add modeled-source presets and source hints for multi-output APIs such as `recv`, `recvfrom`, and
   `read`.
2. Add backward/trace field-load provenance leaves, then implement precise reaching-store recovery
   where memory SSA permits it.
3. Add boundedness classification for direct `read`/`recv` return-value lengths.
4. Add an in-process code fingerprint to `doctor` and document/recommend fresh sessions for taint
   dogfood.
5. Expand the bundled model DB for GLib, framework, C++ logging, and common local wrapper categories.
6. Improve the trace JSON contract with stable SSA labels and field-load metadata.
7. Update stale taint design docs and the `taint_engine.py` module header.

## Reproduction Commands

> Mock targets/paths/instances — substitute your own.

```sh
bn session start /fw/parserd --instance-id audit-parserd --format json

bn --instance audit-parserd -t parserd target info --format json
bn --instance audit-parserd -t parserd xrefs recvfrom --limit 20
bn --instance audit-parserd -t parserd xrefs recv --limit 20
bn --instance audit-parserd -t parserd xrefs memcpy --limit 40

bn --instance audit-parserd -t parserd taint forward \
  -f tlv_parse --source param:0 --max-depth 3 --format json

bn --instance audit-parserd -t parserd taint backward \
  -f msg_parse --sink arg:memcpy:2 --max-depth 2 --format json

bn --instance audit-parserd -t parserd trace msg_parse 0x10408 --arg 2 --format json
bn --instance audit-parserd -t parserd trace msg_parse 0x10650 --arg 2 --format json

bn --instance audit-parserd -t parserd taint forward \
  -f sub_10a40 --source arg:recvfrom:1 --max-depth 3 --format json

bn --instance audit-parserd -t parserd taint forward \
  -f sub_10a40 --source ret:recvfrom --max-depth 3 --format json

bn --instance audit-parserd -t parserd taint forward \
  -f sub_10c00 --source arg:recv:1 --max-depth 2 --format json

bn session start /fw/devmgrd --instance-id audit-devmgrd --format json
bn --instance audit-devmgrd -t devmgrd taint forward \
  -f sub_11480 --source ret:read --max-depth 3 --format json
bn --instance audit-devmgrd -t devmgrd taint backward \
  -f sub_11480 --sink arg:memcpy:2 --max-depth 2 --format json
bn --instance audit-devmgrd -t devmgrd trace sub_11480 0x11514 --arg 2 --format json

bn session start /fw/httpd --instance-id audit-httpd --format json
bn --instance audit-httpd -t httpd taint forward \
  -f resp_read --source ret:read --max-depth 3 --format json

bn session start /fw/svc-alpha --instance-id audit-svc-alpha --format json
bn --instance audit-svc-alpha -t svc-alpha taint forward \
  -f sub_10aa0 --source arg:read:1 --max-depth 3 --format json

bn session start /fw/discoveryd --instance-id audit-discoveryd --format json
bn --instance audit-discoveryd -t discoveryd taint forward \
  -f sub_10110 --source arg:read:1 --max-depth 3 --format json
```
