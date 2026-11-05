/* Copy lengths computed by library calls over bytes or pointers (IC06).
 *
 * `copy_label` copies `label` with length `strlen(label) + 1`: the length is
 * computed from the bytes `label` points at, so a backward slice of it
 * demands those bytes -- its origin is that boundary at the strlen call,
 * never the formal `label` -- while the copy's own source operand IS the
 * pointer. `relay_label` hands its formal on, so the ascent reads the
 * caller's object the same way. `copy_field` copies up to the first comma:
 * `strchr` returns a POINTER, so the length `comma - line` follows the
 * pointer values. `copy_tag` measures a local whose NUL a store proves: the
 * demand is the three bytes up to it, not the 64-byte capacity. `copy_read`
 * measures a local `read()` filled: the length's provenance is the read.
 * Built at -O0 with -g so the formals and locals keep their names. */
#include <string.h>
#include <unistd.h>

__attribute__((noinline)) void copy_label(char *dst, const char *label)
{
    memcpy(dst, label, strlen(label) + 1);
}

__attribute__((noinline)) void relay_label(char *dst, const char *name)
{
    copy_label(dst, name);
}

__attribute__((noinline)) void copy_field(char *dst, const char *line)
{
    const char *comma = strchr(line, ',');
    memcpy(dst, line, comma - line);
}

__attribute__((noinline)) void copy_tag(char *dst, const char *src)
{
    char tag[64];
    tag[0] = 'o';
    tag[1] = 'k';
    tag[2] = 0;
    memcpy(dst, src, strlen(tag));
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
    relay_label(out, argv[0]);
    copy_field(out, argv[argc - 1]);
    copy_tag(out, argv[0]);
    copy_read(out, 0);
    return out[0];
}
