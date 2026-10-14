# Stress & command-surface harness

Local, CI-free checks that exercise the bn CLI/bridge against small synthetic
binaries. They are **not** part of `pytest` (they need a real Binary Ninja and
spawn live bridge sessions); run them by hand.

## Fixtures

`tests/fixtures/` holds license-clean **synthetic** x86_64 programs built from
`tests/fixtures/src/*.c` (plus one C++ program, `src/classes.cpp`) — no target/proprietary data. The built binaries are
git-ignored; both harnesses build them on demand. To (re)build manually:

```bash
make -C tests/fixtures        # produces *_x86_64 from src/*.c and src/*.cpp
make -C tests/fixtures clean
```

| fixture                | shape it exercises                              |
|------------------------|-------------------------------------------------|
| `hello_x86_64`         | strings + a helper function                     |
| `add_x86_64`           | named callable functions (`add`, `mul`)         |
| `crypto_x86_64`        | a toy XOR/rotate routine + a key string         |
| `statemachine_x86_64`  | a branchy switch state machine                  |
| `parser_x86_64`        | a length-prefixed record parser (source→sink)   |
| `dispatch_table_x86_64`| a const function-pointer dispatch table (indirect source→sink) |
| `classes_x86_64`       | a C++ PIE: single-inheritance vtables, an exception subclass of an imported class, a template class, an abstract class with zeroed destructor slots, one multiple-inheritance class (class lens, #922) |
| `classes_stripped_x86_64` | the same C++ PIE built with `-s`: synthesized slot-array vtable symbols must recover the same object heads and slot contents without phantom secondaries (#935) |
| `canary_x86_64`        | a stack-protected buffer: frame-map canary evidence and slot-to-canary distances |
| `vcalls_x86_64`        | a C++ binary whose virtual methods are reached only through vtable slots (one past the 64-slot layout window, one in a secondary vtable), and whose classes are heap-constructed by out-of-line ctors (`callsites` slot notes, `class` sizes, #998) |
| `sizes_x86_64`         | a C++ binary built at -O2 whose `operator new` allocations are NOT one object of the class whose ctor or vptr touches them (array new, members at offset 0, placement new, inlined derived ctors and final dtors, trailing storage, a header clearing its enclosing payload) and a class with a virtual base -- class sizes must be exact or absent (#998) |
| `slots_x86_64`         | a C++ binary built at -O2 with data shaped like a vtable slot (the word after a 70-slot vtable, a `{0, &typeid(X), fn}` registry) and a 5000-entry function-pointer table -- never a confirmed slot, never slow (#998) |

The paired class-lens regression checks slot addresses against each view's raw
words. Across the symbolized/stripped views, local and null slots, object heads,
and RTTI headers remain exact; only slots proven by BN to point to an
`ExternalSymbol` compare by raw symbol name and namespace instead of BN's
per-view synthesized address (for example, `__cxa_pure_virtual`).

## Multi-instance stress test

```bash
bash tests/stress/run_stress.sh          # builds fixtures if missing
bash tests/stress/run_stress.sh --keep   # leave sessions running for debugging
```

Exercises session lifecycle, `--instance` routing, auto-start, concurrent
mutations, backward compat and registry cleanup. Exit code = number of failed
assertions (0 = all passed).

> Note: the preflight stops **every** registered bn instance before running, so
> don't run it alongside live sessions you care about.

## Command-surface smoke audit

```bash
uv run python tests/stress/run_command_surface_audit.py
```

Enumerates every registered command (`bn.cli._COMMANDS`), runs `--help` for all
of them (argparse-wiring check), then runs each target *read* command against a
fixture with synthesized arguments, asserting no command crashes (a clean
handled error is fine). Mutations and global-state commands are help-only and
listed explicitly so coverage gaps are never silent. Exit 1 on any crash.
