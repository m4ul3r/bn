/* A copy length counted by a loop, and one chosen between two literals (IC08).
 *
 * `count_fields` copies as many bytes as `list` has commas: `k` starts at 0
 * and grows on trips the loop's control picks, so its literal initializer is
 * where the value starts, not what it is -- the trip count (here decided by
 * the bytes of `list`) is control flow a value slice does not follow.
 * `fixed_header` copies one of two literal lengths: a true constant origin.
 * Built at -O0 with -g, where the comma test stays a branch. */
#include <stddef.h>
#include <string.h>

__attribute__((noinline)) void count_fields(char *dst, const char *src, const char *list)
{
    size_t k = 0;
    for (const char *p = list; *p; p++)
        if (*p == ',')
            k++;
    memcpy(dst, src, k);
}

__attribute__((noinline)) void fixed_header(char *dst, const char *src, int wide)
{
    size_t n = 8;
    if (wide)
        n = 16;
    memcpy(dst, src, n);
}

int main(int argc, char **argv)
{
    char out[64];
    count_fields(out, argv[0], argv[argc - 1]);
    fixed_header(out, argv[0], argc > 2);
    return out[0];
}
