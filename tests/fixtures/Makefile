# Synthetic, license-clean x86_64 fixtures for bn's stress + smoke harness.
#
# These are small, self-contained C programs with NO target/proprietary data.
# The built binaries are NOT committed (see .gitignore); regenerate them with:
#
#     make -C tests/fixtures
#
# run_stress.sh and run_command_surface_audit.py build them automatically when
# missing, so a clean checkout works without a manual step.

CC      ?= cc
# -O0 keeps every helper as a real, named function; -no-pie gives stable
# addresses; -fno-stack-protector keeps the decompiled output minimal.
CFLAGS  ?= -O0 -fno-stack-protector -no-pie
SRCDIR  := src
BINS    := hello_x86_64 add_x86_64 crypto_x86_64 statemachine_x86_64 parser_x86_64

all: $(BINS)

%_x86_64: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(BINS)

.PHONY: all clean
