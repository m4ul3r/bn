/* A read-only table at a bounded index is the binary's own bytes (#1053 review 4):
 * the index is the slice's origin, even though `[(i << 3) + table]` keys as an exact
 * slot on the scaled index. Analyze only. */
#include <stdint.h>
#include <stdlib.h>
const uintptr_t readonly_sizes[4] = {1, 2, 3, 4};
__attribute__((noinline)) void *readonly_index(int i) {
    return malloc(readonly_sizes[i & 3]);
}
int main(void) { return 0; }
