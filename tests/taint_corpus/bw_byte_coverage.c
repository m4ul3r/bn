/* Coverage is decided by the BYTES a writer provably writes, not by what kind
 * of writer it is. Each sink below reads a buffer that is never written on
 * the `flag == 0` path, and each used to be a complete slice because some
 * writer "covered" that path without writing the buffer: a read through a
 * pointer that may point at ANOTHER local, a one-byte indexed terminator, an
 * escape of the buffer's address, a copy into the buffer at a constant
 * offset. `fill_at_zero` is the control: a copy at offset 0 covers. */
#include <stdio.h>
#include <string.h>
#include <unistd.h>

char *g_stash;

void phi_pointer(int fd, int flag) {
    char a[64], b[64];
    char *p = flag ? a : b;
    read(fd, p, 63);                    /* writes `a` on ONE path only */
    popen(a, "r");                      /* SINK */
}

void indexed_terminator(int fd, int flag) {
    char buf[64];
    int n;
    if (flag)
        n = read(fd, buf, 63);
    else
        n = 5;
    buf[n] = 0;                         /* one byte, at a variable offset */
    popen(buf, "r");                    /* SINK */
}

void escape_after_fill(int fd, int flag) {
    char buf[64];
    if (flag)
        read(fd, buf, 63);
    g_stash = buf;                      /* stores the address; writes no byte */
    popen(buf, "r");                    /* SINK */
}

void offset_fill(int fd, int flag) {
    char buf[64];
    if (flag)
        read(fd, buf, 63);
    strcpy(buf + 60, "x");              /* writes bytes 60..61 only */
    popen(buf, "r");                    /* SINK */
}

void fill_at_zero(int fd, int flag, const char *host) {
    char buf[64];
    if (flag)
        read(fd, buf, 63);
    strcpy(buf, host);                  /* the string the sink reads, on every path */
    popen(buf, "r");                    /* SINK */
}

int main(int argc, char **argv) {
    phi_pointer(0, argc);
    indexed_terminator(0, argc);
    escape_after_fill(0, argc);
    offset_fill(0, argc);
    fill_at_zero(0, argc, argc > 1 ? argv[1] : "localhost");
    return 0;
}
