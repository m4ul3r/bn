/* A memcpy length read back from a local an in-binary helper wrote through an
 * out-pointer. At -O2 the read is lifted under a zero-extension
 * (`width_1 = zx.q(width @ mem#k)`); at -O1 it is a bare by-name read. Either
 * way the helper is an unmodeled writer of the local, so the slice is never a
 * complete one with origin `entry`. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__attribute__((noinline)) int parse_width(const char *s, int *out)
{
    int n = 0;
    while (*s >= '0' && *s <= '9') { n = n * 10 + (*s - '0'); s++; }
    *out = n;
    return *s == '/';
}

__attribute__((noinline)) int copy_field(const char *src, char *dst, size_t cap, int flag)
{
    int width = -1;
    unsigned long bits = 0x80;
    char tmp[16];
    memset(tmp, 0, sizeof tmp);
    if (flag) {
        if (parse_width(src, &width) <= 0)
            return -1;
        bits = (unsigned)width;
        if (bits == 0xffffffffUL)
            bits = 0x80;
    }
    size_t len = (bits + 7) >> 3;
    if (cap < len)
        return -2;
    memcpy(dst, tmp, len);
    return (int)len;
}

int main(int argc, char **argv)
{
    char out[64];
    if (argc > 1)
        copy_field(argv[1], out, sizeof out, argc > 2);
    return 0;
}
