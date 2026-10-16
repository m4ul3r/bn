/* The planting call isolated (#1053 review 5, N1): the buffer is filled by
 * plain stores, so strtol -- handed the buffer and an end-pointer slot -- is the
 * only call before the reload. The pointer it leaves in the slot may point into
 * the buffer, so the strcpy through it is a frontier. Analyze only. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *g_end;

__attribute__((noinline)) void store_init_global_end(const char *host) {
    char buf[64];
    buf[0] = '1';
    buf[1] = '2';
    buf[2] = 0;
    strtol(buf, &g_end, 10);
    char *p = g_end;
    strcpy(p, host);
    popen(buf, "r");
}

__attribute__((noinline)) void store_init_local_end(const char *host) {
    char buf[64];
    char *end;
    buf[0] = '1';
    buf[1] = '2';
    buf[2] = 0;
    strtol(buf, &end, 10);
    char *p = end;
    strcpy(p, host);
    popen(buf, "r");
}

int main(void) { return 0; }
