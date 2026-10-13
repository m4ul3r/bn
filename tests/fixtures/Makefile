# Synthetic, license-clean x86_64 fixtures for bn's stress + smoke harness.
#
# These are small, self-contained C (and one C++) programs with NO
# target/proprietary data.
# The built binaries are NOT committed (see .gitignore); regenerate them with:
#
#     make -C tests/fixtures
#
# run_stress.sh and run_command_surface_audit.py build them automatically when
# missing, so a clean checkout works without a manual step.

CC      ?= cc
CXX     ?= c++
# -O0 keeps every helper as a real, named function; -no-pie gives stable
# addresses; -fno-stack-protector keeps the decompiled output minimal.
CFLAGS  ?= -O0 -fno-stack-protector -no-pie
# The C++ fixture stays a PIE on purpose: the `.data.rel.ro` layout of a PIE
# packs each vtable directly against the next class's RTTI record, which is
# the shape the class lens's guards exist for (#922).
CXXFLAGS ?= -O0 -fno-stack-protector
SRCDIR  := src
# Where the binaries land. Defaults to this directory (what the real-BN
# integration lane reads); tests that only exercise the *build* pass a tmp dir
# so a unit-only run leaves no working-tree side effects.
OUTDIR  ?= .
NAMES   := hello_x86_64 add_x86_64 crypto_x86_64 statemachine_x86_64 parser_x86_64 dispatch_table_x86_64 classes_x86_64 canary_x86_64
BINS    := $(addprefix $(OUTDIR)/,$(NAMES))
# The bridge prefers an adjacent <binary>.bndb over the binary it caches, so a
# sidecar left behind by an earlier build silently shadows the fresh one.
BNDBS   := $(addsuffix .bndb,$(BINS))

all: $(BINS)

$(OUTDIR)/%_x86_64: $(SRCDIR)/%.c
	@mkdir -p $(OUTDIR)
	$(CC) $(CFLAGS) -o $@ $<

$(OUTDIR)/%_x86_64: $(SRCDIR)/%.cpp
	@mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

# The one fixture that needs a stack canary: `local list --frame` must find it.
# Appended after -fno-stack-protector, so it wins.
$(OUTDIR)/canary_x86_64: CFLAGS += -fstack-protector-all

clean:
	rm -f $(BINS) $(BNDBS)

.PHONY: all clean
