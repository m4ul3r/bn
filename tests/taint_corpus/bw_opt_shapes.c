/* Shapes an OPTIMISING compiler produces for the backward local-buffer walk
 * (built at -O1; `bw_opt_shapes_o2.c` rebuilds the same source at -O2).
 * Every function here is a sink that reads a stack buffer, and every one used
 * to come back `safe_to_report_complete_slice: true` with a tainted input
 * missing from the slice -- the walk EXCLUDED what it could not key:
 *
 *  - a store through a pointer that is a phi / non-constant arithmetic
 *    (`rdx#2 = phi(&buf[1], rdx#3); [rdx#2 - 1].b = ...`) was not a writer;
 *  - a memset arm lifted to `__builtin_memset` (modeled `{}`) was "nothing
 *    writes the buffer";
 *  - a `char buf[] = "lit"` initialiser lifted to `__builtin_strncpy` /
 *    `__builtin_memcpy` had no model at all;
 *  - a tainted LENGTH of a literal fill decided the command and was named in
 *    no step, while the weak-origin tie went to the 1-byte terminator. */
#include <stdio.h>
#include <string.h>
#include <unistd.h>

void xor_loop(int fd, const char *key) {
    char buf[64];
    read(fd, buf, 63);
    buf[63] = 0;
    for (int i = 0; buf[i]; i++)
        buf[i] ^= key[i];           /* store through a phi'd pointer: `key` is an input */
    popen(buf, "r");                /* SINK */
}

void ptr_copy(const char *src) {
    char buf[64];
    char *p = buf;
    while (*src)
        *p++ = *src++;              /* `[rax#3 - 1].b = ...` through a phi'd cursor */
    *p = 0;
    popen(buf, "r");                /* SINK */
}

void read_or_memset(int fd, int flag) {
    char buf[64];
    if (flag)
        read(fd, buf, 63);
    else
        memset(buf, 0, sizeof buf);  /* -O1: `__builtin_memset`, a LITERAL fill */
    buf[63] = 0;
    popen(buf, "r");                /* SINK: short read arm remains uncertain */
}

void literal_init(const char *host) {
    char buf[64] = "ping ";         /* -O1: `__builtin_strncpy` / `__builtin_memcpy` */
    strcat(buf, host);
    popen(buf, "r");                /* SINK */
}

void literal_with_length(unsigned n) {
    char buf[64];
    /* The length is unbounded here (main passes argc) so the optimiser keeps
     * the memcpy a CALL instead of unrolling it into stores. */
    memcpy(buf, "echo hi; cat /etc/passwd", n);   /* `n` cuts the command */
    buf[n] = 0;
    popen(buf, "r");                /* SINK */
}

int main(int argc, char **argv) {
    xor_loop(0, argc > 1 ? argv[1] : "k");
    ptr_copy(argc > 1 ? argv[1] : "localhost");
    read_or_memset(0, argc);
    literal_init(argc > 1 ? argv[1] : "localhost");
    literal_with_length(argc);
    return 0;
}
