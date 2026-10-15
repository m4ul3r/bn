/* A proof cut must not drop a loop-carried escaped-call frontier it already
 * found (#1034 item 3): after 300 unrelated calls the memory-chain budget is
 * exhausted, yet `compose` -- which runs after &buf escaped on the loop's
 * second trip -- must still be named. `short_loop` is the uncut control.
 * Analyze only. */
#include <stdio.h>
#include <string.h>

char *stash;

__attribute__((noinline)) void clean(void) { __asm__ volatile(""); }
__attribute__((noinline)) void compose(void) { __asm__ volatile(""); }

#define CLEAN3 clean(); clean(); clean()
#define CLEAN30 CLEAN3; CLEAN3; CLEAN3; CLEAN3; CLEAN3; CLEAN3; CLEAN3; CLEAN3; CLEAN3; CLEAN3

__attribute__((noinline)) void long_loop(int n) {
    char buf[64];
    CLEAN30;
    CLEAN30;
    CLEAN30;
    CLEAN30;
    CLEAN30;
    CLEAN30;
    CLEAN30;
    CLEAN30;
    CLEAN30;
    CLEAN30;
    memset(buf, 0, sizeof buf);
    while (n--) {
        compose();
        stash = buf;
    }
    popen(buf, "r");
}

__attribute__((noinline)) void short_loop(int n) {
    char buf[64];
    memset(buf, 0, sizeof buf);
    while (n--) {
        compose();
        stash = buf;
    }
    popen(buf, "r");
}

int main(int argc, char **argv) {
    long_loop(argc);
    short_loop(argc);
    return 0;
}
