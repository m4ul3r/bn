#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *g_ptr;
char g_other[64];

/* do-while: iteration 1 writes through the OLD g_ptr; buf is uninitialized when n == 1. */
__attribute__((noinline)) void dowhile_stash(int n) {
    char buf[64];
    int k = 0;
    do {
        snprintf(g_ptr, 64, "%s", "id");
        g_ptr = buf;
    } while (++k < n);
    popen(buf, "r");
}

/* Same, with a strcpy the compiler may expand inline. */
__attribute__((noinline)) void dowhile_stash_strcpy(int n) {
    char buf[64];
    int k = 0;
    do {
        strcpy(g_ptr, "id");
        g_ptr = buf;
    } while (++k < n);
    popen(buf, "r");
}

/* No loop: the stash is conditional, so the reload may not be &buf. */
__attribute__((noinline)) void branch_stash(int c) {
    char buf[64];
    if (c)
        g_ptr = buf;
    snprintf(g_ptr, 64, "%s", "id");
    popen(buf, "r");
}

/* Constant-offset scalar read through a pointer copy of a fully filled local. */
__attribute__((noinline)) void ptr_copy_load(int unused) {
    char t[64];
    memset(t, 7, sizeof t);
    char *p = t;
    void *q = malloc((size_t)(unsigned char)p[3]);
    free(q);
}

/* Escape strictly after the loop; the loop's helper ran before it. */
__attribute__((noinline)) void touch(void) { g_other[0]++; }
__attribute__((noinline)) void escape_after_loop(int n) {
    char buf[64];
    strcpy(buf, "id");
    for (volatile int k = 0; k < n; k++)
        touch();
    g_ptr = buf;
    popen(buf, "r");
}

int main(int argc, char **argv) {
    g_ptr = g_other;
    dowhile_stash(argc);
    dowhile_stash_strcpy(argc);
    branch_stash(argc);
    ptr_copy_load(argc);
    escape_after_loop(argc);
    return 0;
}
