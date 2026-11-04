/* A tainted scalar stored into a global by a writer that makes no call after
 * the write, read by ANOTHER function -- and a control that reads only the
 * global declared right before it.
 *
 * `fill` must keep a blocking frontier at its store: the walk never ascends
 * past the seeded function, so `use`'s `memcpy(dst, g_buf, g_len)` is code it
 * does not follow, and `safe_to_report_all_clear` must stay false.
 *
 * `fill_then_log` calls `log_banner`, whose only global is `g_banner`, the
 * object laid out immediately before `g_len`. Its string read has no stated
 * length, but `g_banner` DECLARES 16 bytes, so that read is of its own
 * storage: it must never be cited as a reader of `g_len`.
 *
 * All three globals are initialised so they share `.data` in definition
 * order -- the neighbour really abuts the tainted scalar. */
#include <stdio.h>
#include <string.h>
#include <unistd.h>

char g_banner[16] = "ready";
int g_len = 1;
char g_buf[256] = {1};

__attribute__((noinline)) void log_banner(void) {
    fputs(g_banner, stdout);                     /* the neighbour, unbounded */
}

__attribute__((noinline)) void fill(void) {
    g_len = (int)read(0, g_buf, sizeof g_buf);   /* no call after the write */
}

__attribute__((noinline)) void fill_then_log(void) {
    g_len = (int)read(0, g_buf, sizeof g_buf);
    log_banner();                                /* control: not a reader */
}

__attribute__((noinline)) void use(void) {
    char dst[64];
    memcpy(dst, g_buf, g_len);                   /* reads both elsewhere */
    puts(dst);
}

int main(void) {
    fill();
    fill_then_log();
    use();
    return 0;
}
