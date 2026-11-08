# A fake extracted-firmware root filesystem for bn's real-BN project lane.
#
# Linux ELF built with a musl cross toolchain, so the tree looks like what a
# firmware unpacker leaves behind, independent of the host OS:
#
#   etc/ld.so.conf                 /usr/local/lib + `include /etc/ld.so.conf.d/*.conf`
#   etc/ld.so.conf.d/vendor.conf   /opt/vendor/lib
#   etc/profile                    exports LD_LIBRARY_PATH=/opt/extra/lib:... (dir absent)
#   lib/libc.so                    the toolchain's musl libc
#   lib/ld-musl-<arch>.so.1        -> libc.so (the loader alias a real rootfs has)
#   opt/vendor/lib/libgreet.so.1   SONAME libgreet.so.1; exports greet, vendor_sum, greet_count
#   opt/vendor/lib/libgreet.so     -> libgreet.so.1
#   usr/bin/app                    NEEDED libgreet.so.1 + libc.so, RPATH /opt/vendor/lib;
#                                  imports greet, vendor_sum, printf, strdup, strlen, free
#   usr/bin/libstray.so            a copy of libgreet on NO lookup path (must not be imported)
#
# The output tree is NOT committed (see .gitignore); regenerate it with:
#
#     make -C tests/fixtures/rootfs
#
# tests/conftest.py's `rootfs_fixture` builds it when a toolchain is found.

# Toolchain: CROSS_CC wins; else the first known Linux musl cross gcc on PATH
# or in Homebrew's musl-cross keg.
CROSS_CC ?=
_CANDIDATES := aarch64-linux-musl-gcc x86_64-linux-musl-gcc arm-linux-musleabihf-gcc musl-gcc
_SEARCH     := $(subst :, ,$(PATH)) /opt/homebrew/opt/musl-cross/bin
_FOUND      := $(firstword $(foreach c,$(_CANDIDATES),$(wildcard $(addsuffix /$(c),$(_SEARCH)))))
CC          := $(or $(CROSS_CC),$(_FOUND))

OUT    ?= out
SRCDIR := src

# A distro musl-gcc wrapper may have an empty sysroot and report glibc's
# generic compiler paths. Resolve the actual linked musl runtime instead.
PYTHON    ?= python3
LIBC_SRC  ?=
LDMUSL    ?=
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(CC),)
_RUNTIME := $(shell $(PYTHON) toolchain.py --cc "$(CC)" --libc "$(LIBC_SRC)" --loader "$(LDMUSL)")
endif
endif
_LIBC_SRC := $(word 1,$(_RUNTIME))
_LDMUSL   := $(word 2,$(_RUNTIME))

LIBGREET := $(OUT)/opt/vendor/lib/libgreet.so.1
APP      := $(OUT)/usr/bin/app
LIBC     := $(OUT)/lib/libc.so
STRAY    := $(OUT)/usr/bin/libstray.so
CONFS    := $(OUT)/etc/ld.so.conf $(OUT)/etc/ld.so.conf.d/vendor.conf $(OUT)/etc/profile

all: check-toolchain $(LIBGREET) $(OUT)/opt/vendor/lib/libgreet.so $(APP) $(LIBC) \
     $(OUT)/lib/$(_LDMUSL) $(STRAY) $(CONFS)

check-toolchain:
ifeq ($(CC),)
	$(error no Linux musl cross compiler found: set CROSS_CC=<path to *-linux-musl*-gcc> or install one of $(_CANDIDATES) (Homebrew: brew install musl-cross))
endif
ifeq ($(_LDMUSL),)
	$(error could not resolve $(CC)'s musl runtime; see diagnostics above, or set LIBC_SRC=<musl libc.so> and LDMUSL=ld-musl-<arch>.so.1)
endif

$(LIBGREET): $(SRCDIR)/greet.c
	@mkdir -p $(dir $@)
	$(CC) -shared -fPIC -Wl,-soname,libgreet.so.1 -o $@ $<

$(OUT)/opt/vendor/lib/libgreet.so: $(LIBGREET)
	ln -sfn libgreet.so.1 $@

# PIE keeps optional weak imports dynamically resolvable instead of allowing
# a non-PIE linker to fold their addresses to zero.
$(APP): $(SRCDIR)/app.c $(OUT)/opt/vendor/lib/libgreet.so Makefile
	@mkdir -p $(dir $@)
	$(CC) -fPIE -pie -o $@ $< -L$(dir $(LIBGREET)) -lgreet -Wl,-rpath,/opt/vendor/lib

$(LIBC): $(_LIBC_SRC)
	@mkdir -p $(dir $@)
	cp $< $@

$(OUT)/lib/$(_LDMUSL): $(LIBC)
	ln -sfn libc.so $@

$(STRAY): $(LIBGREET)
	@mkdir -p $(dir $@)
	cp $< $@

$(OUT)/etc/ld.so.conf:
	@mkdir -p $(dir $@)
	printf '/usr/local/lib\ninclude /etc/ld.so.conf.d/*.conf\n' > $@

$(OUT)/etc/ld.so.conf.d/vendor.conf:
	@mkdir -p $(dir $@)
	printf '# vendor\n/opt/vendor/lib\n' > $@

$(OUT)/etc/profile:
	@mkdir -p $(dir $@)
	printf 'export LD_LIBRARY_PATH=/opt/extra/lib:$$LD_LIBRARY_PATH\n' > $@

clean:
	rm -rf $(OUT)

.PHONY: all clean check-toolchain
