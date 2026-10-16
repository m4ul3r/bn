/* Section semantics do not make bytes immutable (#1053 review 5): this table
 * sits in .data.rel.ro, but linked as a PIE without RELRO it is in a writable
 * segment, and a function writes its first entry. Only a non-writable segment
 * makes the loaded entry the binary's own bytes. Analyze only. */
#include <stdint.h>
#include <stdlib.h>

uintptr_t target_value;
uintptr_t *relocated_sizes[4] __attribute__((section(".data.rel.ro"))) = {
    &target_value, &target_value, &target_value, &target_value};

__attribute__((noinline)) void set_first_entry(uintptr_t *p) { relocated_sizes[0] = p; }

__attribute__((noinline)) void *relocated(int i) {
    return malloc((uintptr_t)relocated_sizes[i & 3]);
}

int main(void) { return 0; }
