/* Single-block -O2 bodies Binary Ninja flags `is_thunk` because they END in a
 * tail call.
 *
 * Built with -O2 -fno-inline-small-functions -D_FORTIFY_SOURCE=2: the copies
 * become __*_chk tail calls with arguments the wrapper rearranged, so none of
 * them is a pure veneer, and `dispatch_frame` makes calls of its own before
 * tail-calling `store_tail`. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

char g_frame[64];
int g_count;

__attribute__((noinline)) void store_tail(const char *p, size_t n)
{
    memcpy(g_frame, p, n);
}

__attribute__((noinline)) void move_op(const char *p, size_t n) { memmove(g_frame, p, n); }
__attribute__((noinline)) void count_op(const char *p, size_t n) { g_count = (int)n; }
typedef void (*frame_op)(const char *, size_t);
static frame_op g_ops[2] = { move_op, count_op };

__attribute__((noinline)) void copy_name(const char *p, size_t n)
{
    strncpy(g_frame, p, n);
}

__attribute__((noinline)) size_t frame_len(const char *p)
{
    return strlen(p);
}

__attribute__((noinline)) void dispatch_frame(const char *buf, size_t n, int sel)
{
    size_t m = frame_len(buf);
    g_ops[sel & 1](buf, n);
    copy_name(buf, n);
    g_count += (int)m;
    store_tail(buf, n);
}

int main(int argc, char **argv)
{
    char buf[256];
    ssize_t n = read(0, buf, sizeof buf);
    if (n <= 0)
        return 1;
    dispatch_frame(buf, (size_t)n, argc);
    return 0;
}
