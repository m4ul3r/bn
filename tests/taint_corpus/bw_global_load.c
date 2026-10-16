/* A fixed-address load of a writable global nothing here writes is a frontier
 * (#1053 review 4). Built at -O2. Analyze only. */
#include <stdint.h>
#include <stdlib.h>

uintptr_t mutable_length;
__attribute__((noinline)) void *global_and_n(uintptr_t n) {
    return malloc(n + mutable_length);
}
int main(void) { return 0; }
