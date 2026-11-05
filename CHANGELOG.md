# Changelog

All notable user-visible changes to `bn` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

The taint/trace hardening rounds R1–R5 (#1059–#1066) and the dogfood fix round
that followed them. Field and leaf names are documented in
`skills/bn/reference/reading.md`; `bn capabilities --format json` lists the
taint output vocabulary.

### Added

- `taint backward`: one slice per path with every origin of that path in
  `origins[]`; every followed caller site is kept, the caller cap follows
  address order and names each site it drops (#1059, #1061).
- `taint backward`: runtime start-up handoffs (an entry stub handing `main` to
  the C runtime) are a named `crt_entry` boundary, not an origin (#1060).
- `taint forward`: a `global_memory_continuation` frontier when tainted bytes
  are written into a writable global that code past a call boundary can read;
  one leaf per global, listing every call it stayed live across, its readers,
  and a runnable reseed (#1063).
- `taint forward`: `model_destination_unrecovered` when a reached model writes
  into an argument Binary Ninja did not recover, and `format_destination_unrecovered`
  / `scanf_arity_residual` for scanf-family formats that write more
  destinations than were seeded; the C23 `__isoc23_*` spellings resolve to the
  plain models (#1064, #1065).
- `taint forward`: `unread_callee_body` when tainted arguments enter an
  in-binary callee whose body was never analysed, whether or not a model covers
  the call (#1066).
- `taint forward`: `gated_sink_reached` when taint arms a sink gated behind an
  opt-in name (`recv_overflow`); `seed_unused` when the seed is never read.
- `taint forward`: `opt_in_sinks_reached` lists calls of an opt-in sink class
  the run did not enable (`file_write`, `net_write`) whose arming arguments
  taint reached; `diagnostics.opt_in_sink_classes_reached` names their classes
  and `diagnostics.exits` says whether taint leaves the function through its
  return value or out-parameters.
- `run_params` on every taint result echoes the `--resolve-map` path with its
  matched/unmatched pins and the model overlays actually merged; on a forward
  result it also echoes the enabled `--sink-class` list (`sink_classes`;
  backward has no `--sink-class`).
- `stats.truncation_cause` uses one documented vocabulary; every cause is
  named in text and none advertises a flag that cannot move it.
- `sink.width_narrowing` on `alloc_size`/`overflow_len` flows whose last
  arithmetic is computed in a narrower width than the sink's parameter.
- Builtin models: regex match APIs propagate the subject into the match
  offsets; fortified `__*_chk` destination sizes are value arguments; time
  formatting writes no caller buffer.
- `trace`: every terminal step is a named frontier kind with a roll-up, and
  interprocedural text keeps its call sites; `dataflow` lists value-set
  callers and accepts `--at ADDR:IL_INDEX`.
- `bn capabilities --format json` advertises the taint leaf kinds, truncation
  causes, origin kinds, sink statuses and `run_params` keys.
- Spilled output gets a private provenance sidecar (`<artifact>.meta.json`).
- `function cfg --limit/--offset/--address-window` slice the graph
  bridge-side; `read --encoding bytes` under `--format json` is a base64
  envelope and spills or writes `--out` like any read.
- `bn capabilities` rows state each command's output surface (formats,
  `--out`, `--estimate-output`, paging, slice flags); `target list` and
  every collection page state their row schema and paging.
- `CHANGELOG.md`.

### Changed

- `callee` on every taint row (reached sink, frontier leaf, backward origin,
  per-site slice and `sink_status` row) is the symbol the binary calls;
  the model key it was judged by is the separate `model` key.
- Every leaf's sentence is `detail`, and a leaf's `function` is an object
  `{name, display_name, address}`.
- `taint models`: a model file with an unknown key, a non-object model or an
  unknown sink class is refused naming the path; address-keyed models resolve
  against the target; overlays are labelled by the knob that supplied them.
- `--resolve-map` refuses scalar targets; a pin counts as matched only when it
  routed an indirect call or jump the walk could not resolve itself.
- Forward frontier text is one wrapped, structured block per fact, with the
  follow-up command printed as one paste-safe span.
- A forward run with no sink states the same claim in text and JSON: every
  verdict and `all_clear_reason` is scoped to the enabled sink classes and
  names the opt-in classes taint reached, and the verdict reads the
  `diagnostics.safe_to_report_all_clear` gate (an all-clear for those classes
  only, or "NOT an all-clear"); the next action no longer calls an
  unexplained empty result a local dead end.
- `taint`'s `--max-depth` is an analysis bound; size advice points at `--out`.
- `exports` lists one row per definition and counts the duplicate symbol
  records it folded (`duplicates_collapsed`); `exports` and `exports list`
  take the same flags.
- `taint_models.py`, `read_taint_models.py` and `taint_models.json` live under
  `src/bn/`, symlinked into the bridge package.

### Fixed

- Backward slices no longer accept a may-alias store, a write-back of the slot
  it loads, or a stack probe as a writer, and a modeled source behind a
  may-alias store is evidence rather than the origin.
- Backward caller ascent: each caller site is counted once, depth and cap cuts
  are recorded in `stats`, and a stack-only calling convention passes no
  register argument.
- Forward taint through globals: a tainted `g + i` write taints the object; an
  unproven out-of-object write keeps a blocking frontier; rebased fixed-address
  objects are one object; readers are found from referencing functions only;
  the global sweep is bounded, cached and recursion-safe.
- Forward taint follows a call once per field context and a veneer forwards
  every formal in order; a struct field store taints that field only.
- A fill's return bounds a copy only when its value set is proven
  non-negative.
- Cancelled or abandoned taint walks stop and say so; a timed-out request's
  error carries the cancel outcome.
- Session lifecycle: view ids never repeat across a restart, the idle reaper
  defers while a target holds unsaved work, and `doctor` identifies the `bn`
  on `PATH` by running it.
- Output envelopes: `ok` is the operation's outcome; a failed spill leaves no
  artifact behind.
- `taint models` presence text names the spelling the bridge selected.
- A negative address is an exit-2 refusal naming the accepted forms; an
  ambiguous `class show` name is the JSON refusal envelope.
