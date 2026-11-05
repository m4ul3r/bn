/* Forward taint through a write into a global at a runtime index (D1-01).
 *
 * `stash` stores a tainted byte at `slots[i]` -- a signed index nothing
 * bounds -- and then runs the command in `command`, the 64-byte object laid
 * out right beside `slots` (both initialised, so both in .data). The write
 * may land outside `slots`, inside `command`: the walk must not answer with
 * an all-clear. `stash_masked` masks the index into `slots` (BN bounds the
 * address), so its write provably stays inside the object.
 */
#include <stdlib.h>

#define N __attribute__((noinline))

char slots[64] = "slot";
char command[64] = "true";

N void stash(const char *value, int i) {
    slots[i] = value[0];
    system(command);
}

N void stash_masked(const char *value, int i) {
    slots[i & 63] = value[0];
    system(command);
}

int main(int argc, char **argv) {
    if (argc > 2) {
        stash(argv[1], atoi(argv[2]));
        stash_masked(argv[1], atoi(argv[2]));
    }
    return 0;
}
