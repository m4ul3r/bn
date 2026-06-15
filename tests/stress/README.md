# Stress & command-surface harness

Local, CI-free checks that exercise the bn CLI/bridge against small synthetic
binaries. They are **not** part of `pytest` (they need a real Binary Ninja and
spawn live bridge sessions); run them by hand.

## Fixtures

`tests/fixtures/` holds license-clean **synthetic** x86_64 programs built from
`tests/fixtures/src/*.c` — no target/proprietary data. The built binaries are
git-ignored; both harnesses build them on demand. To (re)build manually:

```bash
make -C tests/fixtures        # produces *_x86_64 from src/*.c
make -C tests/fixtures clean
```

| fixture                | shape it exercises                              |
|------------------------|-------------------------------------------------|
| `hello_x86_64`         | strings + a helper function                     |
| `add_x86_64`           | named callable functions (`add`, `mul`)         |
| `crypto_x86_64`        | a toy XOR/rotate routine + a key string         |
| `statemachine_x86_64`  | a branchy switch state machine                  |
| `parser_x86_64`        | a length-prefixed record parser (source→sink)   |

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
