# bn reference — reading

Read-command catalog for the `bn` skill. Open when surveying/decompiling. See `../SKILL.md` for the map.
## Commands

```bash
bn target info
bn target list [--format json]                       # open targets + their selectors; `selector` is what -t takes
bn function list [--count] [--min-address 0x401000 --max-address 0x40ffff]
bn function list [--sort {address|size|name}] [--reverse] [--min-size 64] [--demangle]
bn function list [--named | --unnamed]               # meaningful names vs BN's auto sub_* (import thunks in neither)
bn function search attachment [--count]              # --count here = MATCHES, not the whole-binary total
bn function search --regex 'attach|detach|follow'
bn function search <q> [--sort … --reverse --min-size N --demangle]
bn function info <fn> [--verbose] [--blocks]         # --blocks: basic-block address ranges + edges
bn decompile <fn> [<fn> …] [--addresses] [--lines 40:80] [--force-analysis] [--include-annotations]
bn il <fn> [--view {hlil|mlil|llil}] [--ssa]          # --level is an accepted alias for --view
bn disasm <fn> [--lines 40:80 | --count 20]
bn disasm <addr> --linear [N]                        # up to N (default 32) units: one physical instruction or one undecodable .byte byte
bn xrefs <fn-or-addr> [--limit 20]
bn xrefs --field <Struct.field>
bn callsites <callee> --within <fn>
bn callsites <callee> --within-file <path>
bn evidence function <fn> [--context 2]              # per-call ABI args (LLIL/MLIL/HLIL + raw disasm) + thunk detection
bn evidence xrefs <fn-or-addr> [--limit N]           # inbound refs annotated with section/segment/symbol/disasm
bn evidence table <addr> [--entries N] [--stride N]  # interpret memory as a pointer/vtable table (Thumb-normalized)
bn evidence table <addr> --record-size N --ptr-fields o1,o2  # mixed-record mode: scalar + pointer fields per record
bn evidence table <addr> --record-size N --field cmd:u32@0 --field name:char[16]@8   # typed scalar/string fields
bn evidence calls <reg-fn> --arg-struct N --field type:u8@2 --field cb:ptr@16  # stack-descriptor fields per callsite
bn evidence message <type-string> [--limit N]        # protobuf/RTTI type-name -> xrefs -> nearby metadata table windows
bn evidence init [--limit N]                         # .init_array/.ctors constructor-pointer summary
bn evidence orient                                   # one-shot triage digest under a single read lock: target + analysis state, imports, strings sample, sections
bn evidence surface                                  # hidden code surface: init/ctor + vtable/dispatch tables + data-referenced code BN missed
bn evidence virtual-call --at <addr> [--providers <selector>]   # resolve an imported abstract/interface virtual call to the provider's vtable method
bn trace <fn> <addr> [--arg N] [--interprocedural]   # backward SSA slice: trace where a call argument originates
bn dataflow defuse <fn> --var <name|local_id|name#version>   # SSA def site + use sites of one variable
                                                    # a use that is argument set-up for a call whose
                                                    # recovered model dropped its stack-passed args is
                                                    # reported as a `hint: call ... call-model truncation`
bn dataflow callgraph <fn> [--direction {callees|callers|both}]   # resolved edges; indirect targets via value-set
bn dataflow values <fn> --at 0x401234                # value-set (possible values) at an instruction
bn taint models [--role {source|sink|propagator}] [--class overflow_len] [--present] [--callsites]   # known sources/sinks; --present needs a target
bn taint forward -f <fn> --source param:0 [--sink-class recv_overflow]   # untrusted data source→sink across calls (methodology → the `bn-vr` skill)
bn taint backward -f <fn> --sink arg:memcpy:2         # slice a sink's args back to their origin (`bounded: true` = provably bounded)
bn function cfg <fn> [--view {asm|mlil|hlil}]        # basic-block graph: blocks, their instructions, and the typed edges between them
bn function structured-il <fn>                       # flat per-instruction list: op plus vars_read/vars_written, no block partitioning
bn proto get <fn>
bn local list <fn>
bn data vars --start <addr> --end <addr>             # typed data variables in an address window (the window is mandatory)
bn data symbols [--limit N] [--offset N]             # named DataSymbols, internal ones included (paged: 100 by default)
bn read 0x... --length N [--encoding {hex|bytes}]   # address is positional; --address 0x... is an accepted alias
bn types [--query <q>]
bn types show <name>
bn struct show <name>
bn class list [--all] [--no-stl] [--query <substr>]   # C++ classes from demangled symbols + RTTI (+ your own declared class types under --all)
bn class show <Name>                                  # one class: methods, vtable, size, bases, instances; falls back to a class type you declared
bn strings [--query <q>] [--regex] [--min-length 5] [--section .rodata] [--no-crt]
bn imports
bn exports [--count]                                 # public exported symbols (contrast `imports`)
bn go functions [--summary | --count]                # recover Go names from .gopclntab (then `bn go rename`, a mutation)
bn sections [--query <q>]
bn tag list [--type Bookmarks --query <substr>]      # tags at all scopes; `--type Bookmarks` is the bookmarks tag
                                                     # UNFILTERED it walks EVERY function (function tags + address tags);
                                                     # narrow with --function <fn>, or --data for data tags alone
bn tag get 0x401000                                # tags at one address
bn tag get --function <fn>                          # whole-function tags
bn tag types                                         # tag types (built-in + custom)
bn comment list [--query <q>] [--scope {all|address|function}]   # `all` (default) includes function docs
bn comment get --address 0x...                     # address comment
bn comment get --function <fn>                     # function documentation comment
bn bundle function <fn> [--out bundle.json]          # export; does not mutate the BNDB
```

Notes:

- `function search` is a case-insensitive substring search. Use `--exact` for a known API, `--word` for an identifier token, or an anchored `--regex` for a family. A regex-shaped zero-hit literal query may be retried as a regex; `regex_fallback` discloses that choice. Confirm a source or sink against the import and actual call target, not its name alone.
- `function list --count` counts all functions; `function search <query> --count` counts matches. `--sort size --reverse --min-size N` is useful for large functions. `--named` / `--unnamed` separate meaningful names from auto `sub_*` names.
- **Duplicate function start addresses are disclosed (#757).** Binary Ninja can hold more than one Function record for one start address. Where an answer holds such an address it publishes `duplicate_starts_collapsed` / `duplicate_starts_unresolved`, and every text face that answer reaches discloses it too. This entry states no more than that on purpose: every summary of what the collapse does to a given answer has been false on some shape of it, so the answer carries that truth and this page does not restate it.
- `evidence orient` reports `existing_annotations` at the top level. A restored `.bndb` can carry earlier names and comments. `decompile` omits stored annotation bodies by default; `--include-annotations` opts in. Redaction covers matching standalone or inline `//` annotation lines, not arbitrary block comments or differently rendered text. Use `--addresses` when mapping a rendered line back to code.
- `bn decompile handle parse_file main` uses one bridge call and preserves the requested order. A missing identifier keeps its row and makes the command exit 2. One identifier retains the single-function `.text` shape; several return `{"kind": "decompile_batch", "requested": 3, "resolved": 2, "functions": [...]}`. Read `.functions[].decompiled.text`, not `.items[]`. An `analysis_skipped` stub is not a body; `--force-analysis` analyzes that function and can be slow.
- `disasm --lines START:END` and `--count N` count rendered function rows, not bytes. `disasm <addr> --linear N` walks physical instructions and undecodable `.byte` units outside recovered blocks. For width, branch, or bounds claims, inspect the instruction operands. On Thumb, include the preceding `IT` instruction and every covered instruction.
- `class list` groups demangled C++ methods and RTTI. Its default favors RTTI/constructor-supported classes; `--all` also shows name-only and user-declared class/struct/union types. A name-only cluster may be a namespace; a declared-only card does not prove methods or vtables are absent. `class show` reports ambiguity and scan caps rather than choosing a same-leaf class or treating a truncated vtable as complete.
- `evidence surface` surveys constructors, pointer tables, and executable data-referenced addresses BN may not have made functions. It is read-only; candidate `code_likely` and `decode_depth` are triage signals, not proof. Verify a candidate with disassembly before `function create`. `evidence init` handles pointer width, endianness, and ARM/Thumb normalization. `evidence table` distinguishes pointer arrays from mixed records; declare record stride and pointer or scalar fields only when the layout is known.
- `evidence calls <reg-fn> --arg-struct N --field name:type@offset` reads a stack-built descriptor at each registration call. `ptr` resolves a callback/data symbol when possible; unknown, computed, and merged values remain marked. Confirm heuristic values at `source_address`. `evidence function` pairs calls with raw ABI and IL evidence; its `argument_confidence` and arity warnings matter when the decompiler guesses arguments. `evidence virtual-call` may return several providers or a typed unresolved reason; a capped scan is not absence.
- `trace <caller> <call-addr> --arg N` follows MLIL SSA backward within the caller. `--interprocedural` follows return values into internal callees; it does not map callee parameters up into callers or follow output-pointer writes. The boundary leaf is `out_param_not_followed` locally or `interprocedural_out_param_not_followed` with `--interprocedural`. Use xrefs and another trace to climb callers.
- `function cfg` contains block edges. In asm view, edge `to` values are addresses; in MLIL/HLIL they are IL instruction indexes. `data vars` requires an address window; `data symbols` lists internal DataSymbols that exports omits. These read under the shared read lock, unlike arbitrary `py exec`.
- `bn read <addr> --length N` reads mapped bytes, with a hard ceiling of 100000 bytes per request. A short mapped read sets `short_read`; a request above the ceiling sets `capped`. Either is partial, with `requested_length` and a note. Under `--encoding bytes`, the note is visible on stderr while stdout carries the raw bytes. An unmapped start errors.

Before an unfamiliar large read, use `--estimate-output` to measure its output without emitting the body; then slice with `--limit`, `--lines`, or `--out`.

## JSON reads and coverage

Most collections use `{kind, items, total, offset, limit, returned, has_more}`. Inspect `kind` before selecting a container. Common leaves:

| Read | Rows or payload |
|---|---|
| `strings` | `.items[].value` |
| `function list` / `function search`, `imports`, `exports` | `.items[].name`, `.items[].address` |
| `sections` | `.items[].name`, `.start`, `.end`, `.length`, `.semantics` |
| `types` | `.items[].decl`, `.layout`, and aggregate `.members[]` |
| `local list` → `.items[]` | same list retained under `.locals`; rows carry `.local_id` |
| `xrefs` | `.items[]`, with `.kind` and `.caller_function` per reference |
| `callsites` | `.items[].caller_static`, `.callee`, `.containing_function` |
| `comment list` | `.items[].comment` and `.scope` (`address` or `function_doc`) |
| `tag list` | `.items[]`; `tag get` instead uses `.tags[]`; `tag types` uses `.tag_types[]` |
| `evidence function` | `.calls[]`, plus pagination metadata |

Addresses in JSON are hex STRING values such as `"0x401000"`; convert before arithmetic. `row_fields`, when present, lists available row keys. A zero-hit read may have no schema for a kind without a declared schema.

**Absence requires a complete read.** Check `has_more`, caps, and scan notes. In particular, an import xref scan may return `items: []` with `truncated: true` and `scan_note`; text prints `note: the caller scan was TRUNCATED`. That means unknown, not zero callers. For high-fan-in `callsites`, `total: null` is a lower bound: inspect `total_lower_bound`, `scan_truncated`, `caller_scan_truncated`, and `caller_scan_note`. Either truncation flag prevents a complete-caller claim. A determined integer total must remain stable across pages.

For taint JSON, forward flows are `reached_sinks[]`, each with `.sink` and `.path`; backward results are `slices[]`, while top-level `sinks` repeats the query. `sink_status[]` discloses whether a backward sink was seeded or proven `bounded`. A non-empty `leaves` list or `truncated: true` prevents an all-clear. Use `--models FILE` for a known wrapper; model entries are keyed by symbol and nest `sink` or `sources`, for example `{"app_recv":{"sink":{"class":"overflow_len","len_arg":1,"buf_arg":2}}}` for `app_recv(fd, len, buf)`. A bounded-write sink uses `len_arg` and `buf_arg` to examine the write length against its destination.

## Caller-static mapping

`bn callsites` returns native `call_addr` and post-call `caller_static` (the return address). Prefer it over hand-written `py exec` for exact caller mapping:

```bash
bn callsites parse_record --within process_message --caller-static
bn callsites parse_record --within-file /tmp/callers.txt --format json
```

`--within-file` accepts one function identifier per non-empty line, with `#` comments. A null `hlil_statement` means BN could not localize a safe statement; JSON may include a bounded `decompile_excerpt` for context. The excerpt is built for the requested page, so use `--limit` to bound its cost. Caller enumeration can stop at its budget; a short list with `caller_scan_truncated: true` is not proof that other callers do not exist.
