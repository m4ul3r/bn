/* Remembered stash slots survive only writes proven not to reach them (#1053 round 2):
 * a framed write through the proven &buf, or to another exact address. strtol and
 * asprintf (no frame), a store through an unpinned pointer, memmove over the slot and a
 * partial memcpy forget it. Analyze only. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char other[64];
char *stash;
char *const otherptr = other;
char **g_pp;

/* controls: genuinely complete full stashes */
__attribute__((noinline)) void ctl_direct(const char *host) {
    char buf[64];
    stash = buf;
    strcpy(stash, host);
    popen(buf, "r");
}
__attribute__((noinline)) void ctl_two_fills(const char *host, const char *pre) {
    char buf[64];
    stash = buf;
    strcpy(stash, pre);
    strcat(stash, host);
    popen(buf, "r");
}
__attribute__((noinline)) void ctl_snprintf(const char *host) {
    char buf[64];
    stash = buf;
    snprintf(stash, sizeof buf, "%s", host);
    popen(buf, "r");
}
__attribute__((noinline)) void ctl_unrelated_call(const char *host) {
    char buf[64];
    stash = buf;
    strcpy(other, "x");
    strcpy(stash, host);
    popen(buf, "r");
}

/* modeled call that rewrites the slot through an undeclared out-pointer */
__attribute__((noinline)) void kill_strtol(const char *host, const char *num) {
    char buf[64];
    memset(buf, 0, sizeof buf);
    stash = buf;
    strtol(num, &stash, 10);
    strcpy(stash, host);
    popen(buf, "r");
}
__attribute__((noinline)) void kill_asprintf(const char *host) {
    char buf[64];
    memset(buf, 0, sizeof buf);
    stash = buf;
    asprintf(&stash, "%s", "x");
    strcpy(stash, host);
    popen(buf, "r");
}
/* store through an unpinned pointer that may be &stash */
__attribute__((noinline)) void kill_ptr_store(const char *host, char **pp) {
    char buf[64];
    memset(buf, 0, sizeof buf);
    stash = buf;
    *pp = other;
    strcpy(stash, host);
    popen(buf, "r");
}
/* store through a pointer loaded from a global (may be &stash) */
__attribute__((noinline)) void kill_gload_store(const char *host) {
    char buf[64];
    memset(buf, 0, sizeof buf);
    stash = buf;
    *g_pp = other;
    strcpy(stash, host);
    popen(buf, "r");
}
/* memmove restore of a different pointer: definite overwrite */
__attribute__((noinline)) void kill_memmove(const char *host) {
    char buf[64];
    memset(buf, 0, sizeof buf);
    stash = buf;
    memmove(&stash, &otherptr, sizeof stash);
    strcpy(stash, host);
    popen(buf, "r");
}
/* partial overwrite: 4 bytes of the 8-byte slot */
__attribute__((noinline)) void kill_partial(const char *host) {
    char buf[64];
    memset(buf, 0, sizeof buf);
    stash = buf;
    memcpy((char *)&stash + 4, &otherptr, 4);
    strcpy(stash, host);
    popen(buf, "r");
}
/* overwrite of the byte just past the slot: slot survives (precision) */
char stash_pad[8];
__attribute__((noinline)) void keep_adjacent(const char *host) {
    char buf[64];
    memset(buf, 0, sizeof buf);
    stash = buf;
    memset(stash_pad, 0, sizeof stash_pad);
    strcpy(stash, host);
    popen(buf, "r");
}

/* A write through a reload PROVEN to be &buf cannot reach the stash slot. */
__attribute__((noinline)) void repeated_stash(const char *host) {
    char buf[64];
    stash = buf;
    memset(stash, 0, sizeof buf);
    strcpy(stash, host);
    popen(buf, "r");
}

int main(void) { return 0; }
