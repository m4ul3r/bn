/* Stash identity is flow-sensitive (#1034 item 1). `stash` starts out pointing
 * at another buffer; only `direct` really writes `buf` through it. The other
 * three write through a pointer that is NOT &buf, so their getenv bytes must
 * never become the origin of a complete slice of `buf`. In `direct` the getenv
 * call between the stash and the reload may write any memory, so the slot is
 * forgotten there too. Analyze only. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char g_other[64];
char *stash = g_other;
char *alias_slot;

/* A pointer loaded BEFORE &buf escaped, copied through a second slot. */
__attribute__((noinline)) void held_copy(void) {
    char buf[64];
    memset(buf, 0, sizeof buf);
    char *old = stash;
    stash = buf;
    alias_slot = old;
    strcpy(alias_slot, getenv("HOST"));
    popen(buf, "r");
}

/* The slot is overwritten after the stash. */
__attribute__((noinline)) void overwrite(void) {
    char buf[64];
    memset(buf, 0, sizeof buf);
    stash = buf;
    stash = g_other;
    strcpy(stash, getenv("HOST"));
    popen(buf, "r");
}

/* The slot is restored to its earlier value after the stash. */
__attribute__((noinline)) void restore(void) {
    char buf[64];
    memset(buf, 0, sizeof buf);
    char *saved = stash;
    stash = buf;
    stash = saved;
    strcpy(stash, getenv("HOST"));
    popen(buf, "r");
}

/* The reload IS &buf, but getenv runs before it and may rewrite the slot. */
__attribute__((noinline)) void direct(void) {
    char buf[64];
    memset(buf, 0, sizeof buf);
    stash = buf;
    strcpy(stash, getenv("HOST"));
    popen(buf, "r");
}

int main(void) {
    held_copy();
    overwrite();
    restore();
    direct();
    return 0;
}
