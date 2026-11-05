/* A fortified copy of a static buffer that a helper fills at a runtime
 * offset and returns, reached through a recursive walk (V5-01).
 *
 * Built with -O2 -fno-inline -D_FORTIFY_SOURCE=2. `format_name` appends its
 * argument to `g_name` at `strlen(g_name)` -- a write through `g_name + n` --
 * and returns `g_name`; `walk` recurses, then copies what `format_name`
 * returned into the fixed-size `g_out` with `__strcpy_chk` (the counter after
 * it keeps that a call, not a tail jump). A `--max-depth` that stops at
 * `format_name` assumes its return is tainted and reaches that copy; every
 * deeper budget descends `format_name` and must reach it too. */
#include <stdio.h>
#include <string.h>

static char g_name[256];
char g_out[64];
int g_walks;

__attribute__((noinline)) char *format_name(const char *label, int skip_dot)
{
    size_t n = strlen(g_name);
    strcpy(g_name + n, label);
    return skip_dot ? g_name + 1 : g_name;
}

__attribute__((noinline)) void walk(const char *label, int depth)
{
    if (depth > 0)
        walk(label, depth - 1);
    strcpy(g_out, format_name(label, depth & 1));
    g_walks++;
}

int main(int argc, char **argv)
{
    if (argc > 1)
        walk(argv[1], argc);
    puts(g_out);
    return 0;
}
