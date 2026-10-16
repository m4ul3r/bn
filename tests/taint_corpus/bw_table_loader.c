/* Loader-written bytes in a non-writable segment (#1053 review 6): both tables
 * sit in .rodata of a PIE linked with `-z notext`, so the loader writes every
 * entry before the program runs -- the resolver result of a local IFUNC (an
 * IRELATIVE entry) or the address of an external symbol (an R_X86_64_64 under
 * DT_TEXTREL). A masked index stays inside each table, yet the loaded value is
 * not the file's bytes, so each load is a frontier. Analyze only. */
#include <stdint.h>
#include <stdlib.h>

static uintptr_t impl_small(void) { return 16; }
static uintptr_t impl_large(void) { return 4096; }
static uintptr_t (*pick_impl(void))(void) { return getenv("SIZE_MODE") ? impl_large : impl_small; }
uintptr_t chosen_impl(void) __attribute__((ifunc("pick_impl")));

const uintptr_t resolver_sizes[4] __attribute__((section(".rodata"))) = {
    (uintptr_t)chosen_impl, (uintptr_t)chosen_impl, (uintptr_t)chosen_impl, (uintptr_t)chosen_impl};

const uintptr_t external_sizes[4] __attribute__((section(".rodata"))) = {
    (uintptr_t)&getenv, (uintptr_t)&getenv, (uintptr_t)&getenv, (uintptr_t)&getenv};

__attribute__((noinline)) void *resolver_entry(int i) { return malloc(resolver_sizes[i & 3]); }
__attribute__((noinline)) void *external_entry(int i) { return malloc(external_sizes[i & 3]); }

int main(void) { return 0; }
