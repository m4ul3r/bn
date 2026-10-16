#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *g_ptr;

/* Writes attacker-controlled bytes through whatever g_ptr currently holds. */
__attribute__((noinline)) void helper(void) {
    if (g_ptr) strcpy(g_ptr, getenv("PROBE_X"));
}

/* Iteration 1 stashes &buf; iteration 2 calls helper() after the escape. */
__attribute__((noinline)) void escape_loop(int n) {
    char buf[64];
    strcpy(buf, "id");
    for (volatile int k = 0; k < n; k++) {
        helper();
        g_ptr = buf;
    }
    popen(buf, "r");
}

/* A parameter-controlled index picks which byte sizes the allocation. */
__attribute__((noinline)) void idx_pick(int i) {
    char t[64];
    memset(t, 7, sizeof t);
    void *p = malloc((size_t)(unsigned char)t[i]);
    free(p);
}

int main(int argc, char **argv) {
    escape_loop(argc);
    idx_pick(argc);
    return 0;
}
