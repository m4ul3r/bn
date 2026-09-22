# bn reference — runtime

Use this reference for target routing, headless sessions, quick loads, output, and the Python escape hatch. Use `bn help <group>` for current flag grammar.

## Target and instance selection

```bash
bn target list
bn instance list
bn instance find <path-or-subname>
bn instance gc
bn target close <selector>
bn close -t <selector>
bn close --all
bn save
bn refresh
```

One open target can be selected implicitly. With several open targets, pass `-t <selector>` from `bn target list`; selectors include the returned selector, view or target id, and an unambiguous filename/path. `-i <instance>` selects a bridge. Put these flags before or after a command; explicit flags take precedence over ambient defaults.

Instance routing is `-i` → `BN_INSTANCE` → project sticky pin → sole/unique project-associated live instance → auto-spawn. Target routing is `-t` → `BN_TARGET` → project sticky pin → single-open auto-pick. `BN_TARGET` is an ambient default, not an explicit command-line selection. A bare destructive `bn close` ignores the ambient target and closes only the sole open view or refuses with a target list; use `-t`, a path, or `--all` deliberately. An empty `BN_TARGET` or empty sticky target is refused when a command would use it. A valid ambient target can affect a bare `bn save`, so pass `-t` for the database you intend to write.

```bash
bn instance use <id>
bn target use <selector>
bn instance clear
bn target clear
```

Sticky pins are shared by agents working under one project root. In concurrent work sharing a project or process environment, pass `-i` and `-t` explicitly and do not change shared pins. An agent with a genuinely private process environment can use its own `BN_INSTANCE` and `BN_TARGET`; they still lose to explicit flags. `bn session start` uses `--instance-id` to *name a new bridge*; global `-i` routes to an existing one.

## Sessions and analysis state

```bash
bn load /path/to/sample.bin
bn session start /path/to/sample.bin --instance-id analysis-1
bn session start /path/to/large.bndb --instance-id analysis-1 --detach
bn session status <job-id> -i analysis-1 --format json
bn session list
bn session restart analysis-1
bn session stop analysis-1
bn close
```

A headless start owns an instance that should be stopped when work ends. Set a positive `BN_IDLE_TIMEOUT` on an agent-owned start as a crash fallback (`BN_IDLE_TIMEOUT=3600` gives one hour); unset means no idle reaper. The command reports the loaded target selector. For long analysis, `--detach` registers a bridge and returns a load-job id. Poll that id: `terminal: false` means keep waiting; on terminal, `succeeded` is true or false. A no-id `session status` lists jobs, not one verdict. `status_command` is null when the bridge cannot name itself in a fresh CLI invocation. Stop an exact owned instance even if its target close fails; a timed-out start may have registered after the caller stopped waiting. Leave a pre-existing instance alone when start returns the exact duplicate-ID error.

A saved `.bndb` may be preferred over adjacent raw bytes. On a read-only source mount, `bn save` can use a content-hash cache under `BN_CACHE_DIR` (default `~/.cache/bn`); a later load can restore its annotations. Use `--no-bndb` when the task requires raw-byte analysis without prior names/comments. A save normally preserves the live target selector; if the response reports `rehomed: true`, get the new selector from `bn target list`.

`--quick` skips the full analysis pass. Sections and imports are useful immediately; function listings/searches are partial, strings can refuse, and xrefs, callsites, and taint need analysis. Check `analysis_state` and run `bn refresh` before using these reads for a complete survey. `decompile --force-analysis` can analyze one existing function; it does not replace full target analysis or create a missing function.

```bash
bn session start /path/to/sample.bin --instance-id triage-1 --quick
bn -i triage-1 target info
bn -i triage-1 refresh
```

## Output and artifacts

Read commands default to text. Mutations default to a compact **text status line**; use `--format json --summary` for parseable status, `--verbose` for detailed diffs, and `--out FILE` to write full detail away from stdout. Setup/export commands may default to JSON. Always parse the format requested, not the command's historical default.

`BN_SPILL_TOKENS` opt-in spills an oversized *read* to an artifact and puts an envelope on stdout. When a command is piped, the downstream `jq`/`rg`/`grep` may then see only the envelope. Without spill, a pipe receives full output, although a capturing wrapper may truncate it. Bound with `--limit`, `--lines`, or `--out`, and read stderr for slicing and spill notes. `--estimate-output` runs the read without emitting its body. Its envelope has `ok`, `estimated`, `format`, `bytes`, `tokens`, `tokenizer`, `summary`, and a command-specific `rerun` hint; `spill_token_limit` is present when a spill threshold is armed.

```bash
bn decompile parse_record --lines 20:45
bn function list --limit 100 --format json
bn strings --out /tmp/bn-strings.json
bn xrefs memcpy --estimate-output
bn spill gc --dry-run
```

`--out` writes an artifact and prints a small envelope. JSON envelopes identify `artifact_path`, `format`, `bytes`, `sha256`, and the source `target`/`instance`; check provenance before consuming a shared file. A text `xrefs` view may cap displayed caller groups even without a spill; use `--format json`, `--limit`, or `--out` for counts. `--lines START:END` is 1-indexed, inclusive, and text-only on decompile, IL, disasm, and structured IL. An out-of-range window errors rather than returning an empty answer. `bn spill gc` reclaims cached spill days; `--out` artifacts are caller-owned.

## Discovery and troubleshooting

```bash
bn help evidence
bn capabilities --format json
bn doctor
bn plugin install
bn skill install
```

`bn capabilities` is the registry-derived command catalog. Use `bn doctor` when discovery, installation, or bridge connectivity is wrong; stale plugin code requires restarting the bridge or GUI. `bn plugin install` and `bn skill install` install the local bridge and skill links. `bn instance gc` removes dead registry residue, not live bridges.

## Python escape hatch

```bash
bn py exec --stdin <<'PY'
print(bv.arch.name)
PY
```

Use `py exec` when a built-in read or verified mutation cannot express the task. It runs with an exclusive write lock and unsandboxed Binary Ninja/Python access, so even a read snippet blocks other clients and can mutate the view. Prefer `bn read`, `data vars`, and `data symbols` for common raw/data reads. Keep scripts bounded, verify any changes, and save intentionally.
