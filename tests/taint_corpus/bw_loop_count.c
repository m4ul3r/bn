/* Copy lengths counted by loops, and one chosen between two literals (IC08).
 *
 * `count_fields` copies as many bytes as `list` has commas: `k` starts at 0
 * and grows on trips the loop's control picks, so its literal initializer is
 * where the value starts, not what it is -- the trip count (here decided by
 * the bytes of `list`) is control flow a value slice does not follow.
 * `count_cells` counts inside a nested loop, `scale_rounds` updates its
 * length through a call (`k = next_round(k)`), and `count_or_reset` lets a
 * call reset the count on another back edge: each is still a recurrence.
 * `fixed_header` copies one of two literal lengths: a true constant origin.
 * Built at -O0 with -g, where the tests stay branches. */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

__attribute__((noinline)) void count_fields(char *dst, const char *src, const char *list)
{
    size_t k = 0;
    for (const char *p = list; *p; p++)
        if (*p == ',')
            k++;
    memcpy(dst, src, k);
}

__attribute__((noinline)) void count_cells(char *dst, const char *src, int rows, int cols)
{
    size_t k = 0;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            if (src[r * cols + c] == ',')
                k++;
    memcpy(dst, src, k);
}

__attribute__((noinline)) size_t next_round(size_t k)
{
    return k * 3 + 1;
}

__attribute__((noinline)) void scale_rounds(char *dst, const char *src, int rounds)
{
    size_t k = 0;
    for (int i = 0; i < rounds; i++)
        k = next_round(k);
    memcpy(dst, src, k);
}

__attribute__((noinline)) size_t restart_count(void)
{
    return (size_t)(rand() & 7);
}

__attribute__((noinline)) void count_or_reset(char *dst, const char *src, int n)
{
    size_t k = 0;
    for (int i = 0; i < n; i++) {
        if (src[i] == ',')
            k++;
        else if (src[i] == ';')
            k = restart_count();
    }
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
    count_cells(out, argv[0], argc, argc);
    scale_rounds(out, argv[0], argc);
    count_or_reset(out, argv[0], argc);
    fixed_header(out, argv[0], argc > 2);
    return out[0];
}
