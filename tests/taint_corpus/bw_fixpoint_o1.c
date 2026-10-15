#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
char *g_ptr;
__attribute__((noinline)) int c1(void) { return getchar() == 'a'; }
__attribute__((noinline)) int c2(void) { return getchar() == 'b'; }
__attribute__((noinline)) void helper(void) {
    if (g_ptr) strcpy(g_ptr, getenv("PROBE_X"));
}
__attribute__((noinline)) void fill_first(void) {
    char buf[64], other[64];
    for (;;) {
        if (c1()) { memset(buf, 0, sizeof buf); break; }
        strcpy(other, "x"); puts(other);
        if (c2()) break;
    }
    popen(buf, "r");
}
__attribute__((noinline)) void fill_second(void) {
    char buf[64], other[64];
    for (;;) {
        strcpy(other, "x"); puts(other);
        if (c2()) break;
        if (c1()) { memset(buf, 0, sizeof buf); break; }
    }
    popen(buf, "r");
}
__attribute__((noinline)) void escape_loop(int n) {
    char buf[64]; strcpy(buf, "id");
    for (volatile int k = 0; k < n; k++) { helper(); g_ptr = buf; }
    popen(buf, "r");
}
__attribute__((noinline)) void zero_fill(void) {
    char buf[64]; read(0, buf, 0); popen(buf, "r");
}
__attribute__((noinline)) void short_fill(void) {
    char buf[64]; memset(buf, 0, 1); popen(buf, "r");
}
__attribute__((noinline)) void dowhile_stash(int n) {
    char buf[64]; int k = 0;
    do { snprintf(g_ptr, 64, "%s", "id"); g_ptr = buf; } while (++k < n);
    popen(buf, "r");
}
int main(int argc, char **argv) {
    fill_first(); fill_second(); escape_loop(argc); zero_fill(); short_fill();
    dowhile_stash(argc); return 0;
}
