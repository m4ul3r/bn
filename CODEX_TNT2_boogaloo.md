# Taint/Trace Engine Audit

Date: 2026-06-15

Scope: audit the Binary Ninja taint and trace implementation in this repo, run the current engine against real firmware under `/mnt/fw/p1`, and evaluate usability plus accuracy. This is an engine audit, not a complete vulnerability assessment of the firmware.

## Executive Summary

The taint engine is no longer a toy MVP. It has useful interprocedural forward taint, per-callsite source attribution, modeled sink classes, some memory-SSA load/store correlation, coarse-memory frontier reporting, indirect-call frontier reporting, opt-in sink classes, and a focused corpus. On real AArch64 firmware it produced one plausible, multi-hop parser-to-`memcpy` finding that would be hard to assemble manually.

The biggest accuracy risk is context. The engine can correctly find attacker-controlled values reaching copy lengths, but it does not yet carry enough range/bounds evidence to decide whether the copy is actually unsafe. In the NFC parser sample, the reported `overflow_len` path is real taint propagation, but manual decompilation shows local allocation and length checks that make the security classification too strong without more boundedness reasoning.

The biggest usability risk is source semantics and result interpretation. `read`/`recv` sources force the user to pick buffer bytes vs return length. On simple stack-buffer code this works. On container-style code such as lighttpd buffers, `arg:read:1` can produce a clean-looking result even though the read bytes are semantically passed to parsers through buffer APIs the engine does not model. Negative results must be treated as "no modeled path found", not "no tainted data path exists".

Backward taint and exact trace are useful for local explanation, but their output is still too raw. The trace engine accurately reconstructs SSA slices, yet it emits verbose SSA object strings, mostly-null reasons, and unstructured memory load expressions. Backward taint can cross functions, but origins like `var_28#15` or `g_slist_nth_data` need field/record context to be analyst-friendly.

## What I Reviewed

Implementation areas:

- `plugin/bn_agent_bridge/taint_engine.py`
  - model loading and validation
  - source locator parsing
  - target resolution, import/thunk handling, and Thumb normalization
  - forward taint, backward taint, summaries, memory handling, sink classification
  - per-source attribution and assumptions/leaves
- `plugin/bn_agent_bridge/read_taint_slice.py`
  - exact call-argument SSA trace
  - backward slice behavior and trace result shape
- `src/bn/commands/dataflow.py`
  - CLI surface for `bn taint` and `bn trace`
- `src/bn/formatters.py`
  - human rendering of taint, trace, leaves, and per-source blocks
- `tests/test_taint_engine.py`, `tests/test_taint_integration.py`, `tests/test_dataflow.py`
  - synthetic unit coverage and BN corpus integration coverage

Verification run:

```sh
PYTHONPATH=src pytest tests/test_taint_engine.py tests/test_dataflow.py -q
/opt/bn/.venv/bin/python -m pytest tests/test_taint_integration.py -q
```

Results:

- `120 passed in 0.46s`
- `16 passed in 45.64s`

The integration corpus covers overflow, command injection, descriptors, opt-in file writes, fortified calls, global buffers, heap memory, indirect calls, resolved indirect calls, interprocedural flows, multihop flows, out-params, sanitized negatives, vararg `sprintf`, and vtable callgraph behavior.

## Firmware Targets

I scanned `/mnt/fw/p1` for network/parser-facing ELF targets with useful libc/import surfaces. Interesting candidates included:

- `/mnt/fw/p1/usr/tagl/bin/tskmgr`
- `/mnt/fw/p1/usr/tagl/bin/NS_BackupMgr`
- `/mnt/fw/p1/usr/tagl/bin/communication`
- `/mnt/fw/p1/usr/tagl/bin/SS_DeviceDetectionService`
- `/mnt/fw/p1/usr/libexec/nfc/neard`
- `/mnt/fw/p1/usr/libexec/bluetooth/bluetoothd`
- `/mnt/fw/p1/usr/libexec/bluetooth/obexd`
- `/mnt/fw/p1/usr/sbin/lighttpd`
- `/mnt/fw/p1/usr/sbin/dhcpd`
- `/mnt/fw/p1/usr/sbin/mosquitto`

I selected three fresh BN sessions for the dogfood pass:

| Instance | Firmware path | Reason |
| --- | --- | --- |
| `tnt2-neard` | `/mnt/fw/p1/usr/libexec/nfc/neard` | NFC/NDEF parsing, many parser functions, `recv`, `recvfrom`, `memcpy`, GLib allocation/list APIs |
| `tnt2-tskmgr` | `/mnt/fw/p1/usr/tagl/bin/tskmgr` | Mostly stripped TAGL service, simple `recv` and `memcpy` patterns |
| `tnt2-lighttpd` | `/mnt/fw/p1/usr/sbin/lighttpd` | Larger named network daemon with `read`, parser dispatch, buffer abstractions |

I used explicit `--instance` and `-t` for every BN command to avoid sticky-session ambiguity.

## Dogfood Results

### 1. `neard`: Interprocedural Parser Flow

Command:

```sh
bn --instance tnt2-neard -t neard taint forward \
  -f near_tlv_parse --source param:0 --max-depth 3 \
  --format json --out /tmp/tnt2-neard-tlv.json
```

Result:

- function: `near_tlv_parse` at `0x42de50`
- functions visited: `13`
- max depth reached: `3`
- truncated: `true`
- sinks: `1`
- sink: `memcpy` at `0x42f8bc`, class `overflow_len`, tainted arg `2`
- leaves: `36` `coarse_memory_store`, `6` `unmodeled_callee`
- assumptions: `49`

The useful part is the path quality. The engine connected:

```text
near_tlv_parse(param:0)
  -> near_ndef_parse_msg
  -> sub_429ea0
  -> __near_bluetooth_parse_oob_record
  -> memcpy(..., ..., attacker-derived length) at 0x42f8bc
```

This is real signal. The path tracks a byte field from the input object through phi nodes, integer narrowing/widening, a tailcall, and finally into `memcpy` length.

Manual check:

```sh
bn --instance tnt2-neard -t neard decompile \
  __near_bluetooth_parse_oob_record --addresses --lines 1:265
```

Relevant recovered code:

- `0x42f7c8`: reads `arg1[0x10]`
- `0x42f864`: computes `((uint8_t)x21_6 - 1)`
- `0x42f898`: allocates with `g_try_malloc0(x19_2 + 1)`
- `0x42f8bc`: copies with `memcpy(x0_29, &x27_1[2], x4_5)`

Assessment:

The taint path is accurate. The sink classification is too strong unless the report also carries the nearby allocation and bounds context. The existing bounded-length downgrade only handles relatively direct same-SSA allocation/length equivalence; this case uses `g_try_malloc0`, byte narrowing, `+1`, and loop/bounds structure. Treat this as a true positive taint path but an unconfirmed vulnerability classification.

### 2. `neard`: Backward Taint and Exact Trace at the Same Sink

Commands:

```sh
bn --instance tnt2-neard -t neard taint backward \
  -f __near_bluetooth_parse_oob_record --sink arg:memcpy:2 \
  --max-depth 3 --format json --out /tmp/tnt2-neard-oob-bw.json

bn --instance tnt2-neard -t neard trace \
  __near_bluetooth_parse_oob_record 0x42f8bc --arg 2 \
  --format json --out /tmp/tnt2-neard-oob-trace.json
```

Backward result:

- sink status: seeded `true`
- slices: `3`
- origins included:
  - call to `g_try_malloc0`
  - entry var `var_28#15`
  - call to `g_slist_nth_data`
- leaves: `0`
- assumptions: `0`

Trace result:

- step count: `14`
- truncated: `false`
- important steps:
  - `x2_5#14 = x4_5#6`
  - `x4_5#6 = zx.q(x19_2#3:0.b)`
  - `x19_2#3 = zx.d(x4_4#3)`
  - `x4_4#3 = x21_6#4:0.b - 1`
  - `x21_6#1 = zx.d([x19#1 + 0x10].b @ mem#7)`
  - `x19#1 = arg1#0`

Assessment:

Trace is precise and helpful for local explanation. Backward taint is less usable: it crosses functions, but the origins are not semantic enough to tell the analyst "this length comes from a byte field at `arg1 + 0x10` and loop-carried record parsing". The lack of leaves/assumptions also makes the backward result look more complete than it is.

### 3. `neard`: `recv` Source Selection Smoke Test

Commands:

```sh
bn --instance tnt2-neard -t neard taint forward \
  -f near_snep_core_read --source arg:recv:1 --max-depth 3 \
  --format json --out /tmp/tnt2-neard-snep-argrecv.json

bn --instance tnt2-neard -t neard taint forward \
  -f near_snep_core_read --source ret:recv --max-depth 3 \
  --format json --out /tmp/tnt2-neard-snep-retrecv.json
```

Results:

- `arg:recv:1`: 1 function visited, 0 sinks, 0 leaves, 0 assumptions
- `ret:recv`: 4 functions visited, 0 sinks, 0 leaves, 0 assumptions

Assessment:

This is not necessarily wrong, but it is a UX warning. A quiet result from a `read`/`recv` source should not read as proof that received bytes are irrelevant. The engine currently models the selected source form, not a higher-level "network input event" that expands to buffer bytes, return length, and API-specific state transitions.

### 4. `tskmgr`: Stripped Service With Two `recv` Calls

Target function:

```sh
bn --instance tnt2-tskmgr -t tskmgr decompile sub_40c910 --addresses --lines 1:220
```

Important recovered behavior:

- first `recv(arg1, &buf, 0x90, 0)` at `0x40c964`
- fixed-size `memcpy(arg2, &buf, 0x90)` at `0x40c9e4`
- if header fields are set, allocate `*(arg2 + 0x10)` bytes
- second `recv(arg1, allocated_buf, *(arg2 + 0x10), 0)` at `0x40ca98`

Forward buffer source:

```sh
bn --instance tnt2-tskmgr -t tskmgr taint forward \
  -f sub_40c910 --source arg:recv:1 --max-depth 3 \
  --format json --out /tmp/tnt2-tskmgr-recv-arg.json
```

Result:

- functions visited: `2`
- sinks: `0`
- leaves: `0`
- assumptions include:
  - seeded from `recv` at `0x40c964`
  - tainted buffer copied into destination of `memcpy` at `0x40c9e4`, propagated but not flagged as a sink
  - seeded from `recv` at `0x40ca98`
  - per-source attribution for 2 callsites
- `by_source` has independent entries for both callsites

Forward return source:

```sh
bn --instance tnt2-tskmgr -t tskmgr taint forward \
  -f sub_40c910 --source ret:recv --max-depth 3 \
  --format json --out /tmp/tnt2-tskmgr-recv-ret.json
```

Result:

- functions visited: `1`
- sinks: `0`
- leaves: `0`
- assumptions include per-source attribution and one unmodeled C++ logging function

Backward/trace check on fixed copy length:

```sh
bn --instance tnt2-tskmgr -t tskmgr taint backward \
  -f sub_40c910 --sink arg:memcpy:2 --max-depth 3 \
  --format json --out /tmp/tnt2-tskmgr-memcpy-bw.json

bn --instance tnt2-tskmgr -t tskmgr trace \
  sub_40c910 0x40c9e4 --arg 2 \
  --format json --out /tmp/tnt2-tskmgr-memcpy-len-trace.json
```

The backward command failed with a useful diagnostic:

```text
--sink arg 2 of memcpy reads no variable in the recovered IL (it is a constant or address expression) -- there is no def-chain to slice backward
```

The trace result had `step_count: 0`, which is correct for constant length `0x90`.

Assessment:

This is a good accuracy result. The engine avoided a false-positive overflow on a fixed-length copy while still recording that tainted bytes were copied to the destination. The per-source split is also useful on stripped firmware because both `recv` callsites are analyzed independently.

The CLI could still be friendlier: the backward constant case should ideally return a successful JSON result with a `constant_arg` terminal rather than a process error, so automated tooling can distinguish "constant and safe to slice no further" from actual failure.

### 5. `tskmgr`: Trace of Second `recv` Length

Command:

```sh
bn --instance tnt2-tskmgr -t tskmgr trace \
  sub_40c910 0x40ca98 --arg 2 \
  --format json --out /tmp/tnt2-tskmgr-second-recv-len-trace.json
```

Result:

- step count: `4`
- trace:
  - `x2_2#1 = x22_1#2`
  - `x22_1#2 = zx.q([x21#1 + 0x10].d @ mem#13)`
  - `x21#1 = arg2#0`
  - terminal: `arg2` function parameter

Assessment:

This is a compact, accurate local trace. It would be much more usable if memory loads were structured as fields:

```json
{
  "kind": "field_load",
  "base": "arg2#0",
  "offset": "0x10",
  "width": 4
}
```

Today the user has to parse this out of `il_text`.

### 6. `lighttpd`: Named Network Daemon and Buffer API Boundary

Target function:

```sh
bn --instance tnt2-lighttpd -t lighttpd decompile \
  http_response_read --addresses --lines 1:180
```

Important recovered behavior:

- one `read(arg5, buf, nbytes)` call at `0x42cd74`
- `buf` is derived from a lighttpd buffer object
- read bytes are committed through `buffer_commit`
- parsing continues through `http_response_parse_headers` or callback dispatch

Forward source commands:

```sh
bn --instance tnt2-lighttpd -t lighttpd taint forward \
  -f http_response_read --source arg:read:1 --max-depth 3 \
  --format json --out /tmp/tnt2-lighttpd-http-response-argread.json

bn --instance tnt2-lighttpd -t lighttpd taint forward \
  -f http_response_read --source ret:read --max-depth 3 \
  --format json --out /tmp/tnt2-lighttpd-http-response-retread.json
```

Results:

- `arg:read:1`: 1 function visited, 0 sinks, 0 leaves, 0 assumptions
- `ret:read`: 2 functions visited, 0 sinks, 3 leaves
- `ret:read` leaves:
  - two `coarse_memory_store` leaves in buffer-related code
  - one `indirect_call_unresolved` at callback dispatch `0x42cda8`

Trace commands:

```sh
bn --instance tnt2-lighttpd -t lighttpd trace \
  http_response_read 0x42cd74 --arg 2 \
  --format json --out /tmp/tnt2-lighttpd-read-nbytes-trace.json

bn --instance tnt2-lighttpd -t lighttpd trace \
  http_response_read 0x42cd74 --arg 1 \
  --format json --out /tmp/tnt2-lighttpd-read-buf-trace.json
```

Results:

- `nbytes` trace: 34 steps, not truncated
- `buf` trace: 9 steps, not truncated
- the `nbytes` trace correctly exposes constants such as `0x1000` and `0x40000`, phi joins, and a call boundary at `chunkqueue_length`
- the `buf` trace stops at `arg4` and loads from the lighttpd buffer object

Assessment:

This is the clearest false-negative risk in normal use. The buffer source returns a clean result, but the received bytes are still semantically passed through lighttpd's buffer object and parser APIs. The engine needs models for common buffer/container APIs, or the output needs a warning when a tainted output buffer immediately enters an unmodeled "commit/parse" abstraction.

The trace output is accurate but high-friction. A single read length produced 34 SSA steps, many with `reason: null`, and memory/field relationships are encoded only in raw IL text.

## Accuracy Findings

### A1. Forward interprocedural taint has real recall now

Evidence:

- `near_tlv_parse -> near_ndef_parse_msg -> sub_429ea0 -> __near_bluetooth_parse_oob_record -> memcpy`
- 13 functions visited
- sink path preserved calls, tailcall, phi nodes, load-derived values, and integer conversions

Impact:

This is already useful for firmware triage. It found a plausible length-controlled copy in a parser-heavy stripped-ish target with mixed named and unnamed functions.

Recommendation:

Keep investing in forward taint as the primary vulnerability discovery path. It is the strongest part of the engine.

### A2. Sink severity needs bounds/range context

Evidence:

- `neard` reports `overflow_len` at `0x42f8bc`
- manual decompilation shows allocation `g_try_malloc0(length + 1)` and copy `memcpy(..., length)`
- existing same-SSA bounded downgrade did not fire in this realistic case

Impact:

The engine can overstate severity. A reported `overflow_len` currently means "tainted length reached copy", not "overflow is feasible".

Recommendation:

Add a second-stage bounds classifier:

- model GLib allocation APIs such as `g_try_malloc`, `g_try_malloc0`, `g_malloc_n`, `g_try_malloc_n`
- normalize simple affine/equivalent expressions across casts and byte extensions
- carry allocation-size facts to destination pointers beyond exact same-SSA matches
- attach nearby guard facts to sinks, even if the engine cannot prove safety
- split classes such as `tainted_len`, `bounded_len`, `guarded_len`, `overflow_len_unproven`, `overflow_len_likely`

### A3. Negative results are too easy to over-read

Evidence:

- `lighttpd http_response_read --source arg:read:1` returns 0 sinks, 0 leaves, 0 assumptions
- recovered code commits read bytes into a buffer object and then parses/dispatches them
- `neard near_snep_core_read` also gives quiet `recv` source results
- simple stack-buffer `tskmgr` source tracking works, so the gap is abstraction/modeling rather than total source failure

Impact:

An analyst can mistake "no modeled sink" for "data is not attacker controlled downstream".

Recommendation:

Add source presets and abstraction warnings:

- `--source call:recv` or `--source model:network` should seed both output buffers and return values according to the model
- source models should identify out-buffer writes explicitly, not just argument values
- when an out-buffer is followed by calls like `buffer_commit`, `chunkqueue_*`, parser callbacks, or unresolved indirect calls, report a frontier leaf instead of a clean empty result

### A4. Backward taint is useful but under-explains provenance

Evidence:

- backward taint at `__near_bluetooth_parse_oob_record` produced 3 slices
- origins included `g_try_malloc0`, `var_28#15`, and `g_slist_nth_data`
- exact trace shows the key length is from `[arg1 + 0x10].b`, but backward origin summaries do not expose that semantic fact

Impact:

Backward results are not yet strong enough for analyst-facing root-cause summaries.

Recommendation:

Add structured origin annotations:

- `field_load(base=arg1, offset=0x10, width=1)`
- `loop_carried_phi`
- `caller_arg(function=sub_429ea0, arg=1)`
- `allocator_return(callee=g_try_malloc0)`
- `list_element_return(callee=g_slist_nth_data)`

Also report frontier leaves when backward traversal stops at unmodeled heap/list/container APIs.

### A5. Trace is accurate but not ergonomic

Evidence:

- `tskmgr` second `recv` length trace was concise and correct
- `lighttpd` `read` length trace was correct but 34 steps
- trace JSON uses strings like `<SSAVariable: x2_2 version 1>`
- many normal def-chain steps have `reason: null`
- memory loads are only parseable from `il_text`

Impact:

Trace is good for expert users but hard to consume programmatically and noisy for reports.

Recommendation:

Add normalized fields:

- `ssa_name`: `x2_2#1`
- `kind`: `copy`, `phi`, `constant`, `field_load`, `call_boundary`, `function_parameter`
- for loads: `base`, `offset`, `width`, `memory_version`
- for constants: `value`
- for calls: `callee`, `arg_index`, `return_value`

Then render a compact text trace by default and leave raw IL as supporting evidence.

### A6. Model coverage is the next major accuracy lever

Observed missing or weak areas:

- GLib allocation/list/string APIs: `g_try_malloc0`, `g_slist_nth_data`, `g_strdup`, `g_strdup_printf`
- lighttpd buffer/chunkqueue APIs: `buffer_commit`, `buffer_string_prepare_append`, `http_response_parse_headers`, `chunkqueue_*`
- network/input APIs beyond libc: `SSL_read`, `g_io_channel_read_chars`
- logging/assertion APIs that should usually be taint terminators or low-value leaves

Impact:

Real firmware is dominated by framework/container APIs. Without targeted models, the engine either stops too early or reports coarse leaves that are technically honest but not precise enough.

Recommendation:

Add model packs by ecosystem:

- `glib.json`
- `lighttpd.json`
- `openssl.json`
- `bluez.json`
- `posix-io.json`

Keep them data-driven and make missing-model reporting group by callee so one large run does not spam dozens of repeated assumptions.

### A7. Coarse leaves are honest but too noisy

Evidence:

- `neard near_tlv_parse` produced 36 `coarse_memory_store` leaves and 49 assumptions
- many assumptions repeat the same class of issue

Impact:

The engine is correctly disclosing unsoundness, but the current volume makes the result hard to rank.

Recommendation:

Group leaves by kind and callee/address family:

- show top 5 examples inline
- include counts by kind
- include full details in JSON
- add a severity/risk hint: `frontier_may_hide_sink`, `benign_logging`, `container_boundary`, `unknown`

### A8. Documentation is stale in places

Evidence:

- `plugin/bn_agent_bridge/taint_engine.py` still has header language describing an intraprocedural MVP and deferred interprocedural/memory-SSA work
- current implementation clearly has interprocedural summaries, per-source attribution, and memory-SSA correlation
- `FORWARD_TAINT_DESIGN.md` appears older than the implemented behavior

Impact:

Maintainers and users will underestimate current capabilities and misunderstand intended soundness boundaries.

Recommendation:

Update the header and design docs to match current behavior:

- may-analysis, depth-bounded, summary-based interprocedural forward taint
- exact places where memory is precise vs coarse
- source locator semantics
- interpretation of sinks, leaves, assumptions, and `by_source`

### A9. CLI papercuts matter during dogfood

Observed:

- `bn xrefs read --limit 40 --format json` fails because `--limit` only applies to text
- constant backward slices return process error instead of a structured terminal result
- `--source arg:recv:1` vs `--source ret:recv` is precise but not discoverable

Recommendation:

- allow `--limit` for JSON xref output, or ignore with a warning
- return JSON for constant/address-expression backward terminals
- add `bn taint sources FUNCTION` or `bn taint suggest-sources FUNCTION` to list likely source locators from imports/callsites

## Recommended Work Plan

Priority 1:

- Add structured trace nodes and field-load metadata.
- Add a `call:<callee>` or `model:<source-class>` source preset that expands multi-output APIs like `read` and `recv`.
- Add GLib allocator models, especially `g_try_malloc0`, and extend bounded-length equivalence across casts and simple `+/- 1` expressions.
- Make negative output explicitly say whether frontier coverage is complete, incomplete, or unknown.

Priority 2:

- Add framework/container model packs for GLib, lighttpd, OpenSSL, and BlueZ.
- Group assumptions and leaves in text output.
- Add structured JSON terminals for constants in backward taint and trace.
- Add sink evidence fields for nearby guards and allocation facts.

Priority 3:

- Update stale docs and module headers.
- Add firmware-derived regression fixtures from the three cases in this audit:
  - `neard` bounded-but-tainted `memcpy` length
  - `tskmgr` constant `memcpy` length with copied tainted bytes
  - `lighttpd` read buffer/container boundary
- Add a `bn taint explain` wrapper that runs forward, backward, and trace around a sink and emits one compact analyst report.

## Reproduction Artifacts

JSON artifacts generated during the audit:

- `/tmp/tnt2-neard-tlv.json`
- `/tmp/tnt2-neard-oob-bw.json`
- `/tmp/tnt2-neard-oob-trace.json`
- `/tmp/tnt2-neard-oob-src-trace.json`
- `/tmp/tnt2-neard-snep-argrecv.json`
- `/tmp/tnt2-neard-snep-retrecv.json`
- `/tmp/tnt2-tskmgr-recv-arg.json`
- `/tmp/tnt2-tskmgr-recv-ret.json`
- `/tmp/tnt2-tskmgr-memcpy-len-trace.json`
- `/tmp/tnt2-tskmgr-second-recv-len-trace.json`
- `/tmp/tnt2-tskmgr-evidence-40c910.json`
- `/tmp/tnt2-lighttpd-http-response-argread.json`
- `/tmp/tnt2-lighttpd-http-response-retread.json`
- `/tmp/tnt2-lighttpd-read-nbytes-trace.json`
- `/tmp/tnt2-lighttpd-read-buf-trace.json`

Main sessions:

```sh
bn session start /mnt/fw/p1/usr/libexec/nfc/neard --instance-id tnt2-neard --format json
bn session start /mnt/fw/p1/usr/tagl/bin/tskmgr --instance-id tnt2-tskmgr --format json
bn session start /mnt/fw/p1/usr/sbin/lighttpd --instance-id tnt2-lighttpd --format json
```

## Bottom Line

The engine is good enough to be useful on real firmware today, especially for forward discovery. It should not yet be treated as an oracle for exploitability or for absence of tainted flows. The next step is not a rewrite; it is targeted precision work: source presets, structured trace nodes, framework models, bounds/range evidence, and clearer negative-result semantics.
