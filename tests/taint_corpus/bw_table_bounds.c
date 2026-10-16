/* A constant table read at a variable index or byte offset is the binary's own
 * bytes only when BN bounds every address the load can take inside read-only
 * segments (#1053 review 5). A mutable global follows the table; the unmasked
 * index and the unchecked byte offset may reach it. Analyze only. */
#include <stdint.h>
#include <stdlib.h>

const uintptr_t readonly_sizes[4] = {1, 2, 3, 4};
uintptr_t mutable_after_table = 0x1234;

__attribute__((noinline)) void set_mutable(uintptr_t n) { mutable_after_table = n; }

/* An unchecked byte offset from the table base. */
__attribute__((noinline)) void *byte_offset(uintptr_t off) {
    return malloc(*(const uintptr_t *)((uintptr_t)readonly_sizes + off));
}

/* Unmasked indices of two widths. */
__attribute__((noinline)) void *unmasked_long(long i) { return malloc(readonly_sizes[i]); }
__attribute__((noinline)) void *unmasked_int(int i) { return malloc(readonly_sizes[i]); }

/* Controls: every possible address stays inside the table. */
__attribute__((noinline)) void *masked(int i) { return malloc(readonly_sizes[i & 3]); }
__attribute__((noinline)) void *cmp_bounded_u(unsigned i) {
    return i < 4 ? malloc(readonly_sizes[i]) : 0;
}
__attribute__((noinline)) void *cmp_bounded_s(int i) {
    return (i >= 0 && i < 4) ? malloc(readonly_sizes[i]) : 0;
}

int main(void) { return 0; }
