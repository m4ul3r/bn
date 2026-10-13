#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *g_ptr;
char *volatile g_vptr;

/* Writes attacker-controlled bytes through whatever the stash currently holds. */
__attribute__((noinline)) void helper(void) {
    if (g_ptr) strcpy(g_ptr, getenv("PROBE_X"));
    if (g_vptr) strcpy(g_vptr, getenv("PROBE_Y"));
}

/* Outer iteration 2: helper() in the inner loop runs after g_ptr = buf. */
__attribute__((noinline)) void nested_inner_call(int n, int m) {
    char buf[64];
    strcpy(buf, "id");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            helper();
        g_ptr = buf;
    }
    popen(buf, "r");
}

/* Outer iteration 2: helper() runs after the inner loop stashed buf. */
__attribute__((noinline)) void nested_outer_call(int n, int m) {
    char buf[64];
    strcpy(buf, "id");
    for (int i = 0; i < n; i++) {
        helper();
        for (int j = 0; j < m; j++)
            g_vptr = buf;
    }
    popen(buf, "r");
}

/* Controls: the same nests with no escape. */
__attribute__((noinline)) void nested_inner_call_ctl(int n, int m) {
    char buf[64];
    strcpy(buf, "id");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            helper();
        g_ptr = 0;
    }
    popen(buf, "r");
}

__attribute__((noinline)) void nested_outer_call_ctl(int n, int m) {
    char buf[64];
    strcpy(buf, "id");
    for (int i = 0; i < n; i++) {
        helper();
        for (int j = 0; j < m; j++)
            g_vptr = 0;
    }
    popen(buf, "r");
}

/* Single loop (#951 shape) control: must stay incomplete. */
__attribute__((noinline)) void single_loop(int n) {
    char buf[64];
    strcpy(buf, "id");
    for (int i = 0; i < n; i++) {
        helper();
        g_ptr = buf;
    }
    popen(buf, "r");
}

/* Offset-cache probes: a store at one offset, then fills at other offsets.
   buf[8..31] is never written, so popen may read uninitialized bytes. */
__attribute__((noinline)) void reuse_store_fill(void) {
    char buf[64];
    g_ptr = buf;
    *(unsigned long *)g_ptr = 0x4141414141414141UL;
    strcpy(buf + 32, "id");
    popen(buf, "r");
}

__attribute__((noinline)) void reuse_many(void) {
    char buf[64];
    g_ptr = buf;
    *(unsigned long *)g_ptr = 0x4141414141414141UL;
    *(unsigned long *)(g_ptr + 40) = 0x4242424242424242UL;
    memcpy(buf + 48, "AAAAAAAA", 8);
    strcpy(buf + 32, "id");
    popen(buf, "r");
}

int main(int argc, char **argv) {
    nested_inner_call(argc, argc);
    nested_outer_call(argc, argc);
    nested_inner_call_ctl(argc, argc);
    nested_outer_call_ctl(argc, argc);
    single_loop(argc);
    reuse_store_fill();
    reuse_many();
    (void)argv;
    return 0;
}
