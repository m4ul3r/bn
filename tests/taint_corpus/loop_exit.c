#include <stdio.h>
#include <stdlib.h>
#include <string.h>
__attribute__((noinline)) int c1(void) { return getchar() == 'a'; }
__attribute__((noinline)) int c2(void) { return getchar() == 'b'; }

__attribute__((noinline)) void fill_first(void) {
    char buf[64];
    char other[64];
    for (;;) {
        if (c1()) { memset(buf, 0, sizeof buf); break; }
        strcpy(other, "x");
        puts(other);
        if (c2()) break;          /* buf never written on this exit */
    }
    popen(buf, "r");
}

__attribute__((noinline)) void fill_second(void) {
    char buf[64];
    char other[64];
    for (;;) {
        strcpy(other, "x");
        puts(other);
        if (c2()) break;          /* buf never written on this exit */
        if (c1()) { memset(buf, 0, sizeof buf); break; }
    }
    popen(buf, "r");
}

int main(void) { fill_first(); fill_second(); return 0; }
