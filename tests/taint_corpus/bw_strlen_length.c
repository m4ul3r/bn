/* A copy length computed by strlen() over the bytes it measures (IC06).
 *
 * `copy_label` copies `label` with length `strlen(label) + 1`: the length is
 * derived from the bytes `label` points at, so a backward slice of it must
 * demand those bytes, exactly as the copy's own source operand does.
 * `copy_read` measures a local buffer `read()` filled: the length's
 * provenance is the read. Built at -O0 with -g so the formals keep their
 * names. */
#include <string.h>
#include <unistd.h>

__attribute__((noinline)) void copy_label(char *dst, const char *label)
{
    memcpy(dst, label, strlen(label) + 1);
}

__attribute__((noinline)) void copy_read(char *dst, int fd)
{
    char tmp[64];
    read(fd, tmp, 63);
    tmp[63] = 0;
    memcpy(dst, "0123456789abcdef", strlen(tmp));
}

int main(int argc, char **argv)
{
    char out[80];
    copy_label(out, argv[argc - 1]);
    copy_read(out, 0);
    return out[0];
}
