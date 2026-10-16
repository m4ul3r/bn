/* #1053 review follow-ups S1/S2: a pointer into a store-initialized buffer
 * carried in a non-address-taken struct field (S1), or returned by a
 * `*arg:N -> ret` model whose argument points AT the pointer (strsep,
 * strtok_r continuing; S2), is a possible writer of the buffer -- never
 * unrelated. The buffer is filled by one whole-width store (no call), so a
 * dropped writer would read as a complete slice. Analyze only. */
#define _DEFAULT_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *g_end;
char *g_save;
struct in { char *s; long base; };
union u8 { char b[8]; uint64_t q; };

__attribute__((noinline)) void si_struct_write(const char *host) {
    union u8 u;
    struct in c;
    u.q = 0x3231;
    c.s = u.b;
    c.base = 10;
    strcpy(c.s, host);
    popen(u.b, "r");
}
__attribute__((noinline)) void si_struct_in(const char *host) {
    union u8 u;
    struct in c;
    u.q = 0x3231;
    c.s = u.b;
    c.base = 10;
    strtol(c.s, &g_end, (int)c.base);
    char *p = g_end;
    strcpy(p, host);
    popen(u.b, "r");
}
__attribute__((noinline)) void si_struct_in_strtok(const char *host) {
    union u8 u;
    struct in c;
    u.q = 0x6261;
    c.s = u.b;
    c.base = 0;
    strtok_r(c.s, ",", &g_save);
    char *p = g_save;
    strcpy(p, host);
    popen(u.b, "r");
}
__attribute__((noinline)) void si_strsep(const char *host) {
    union u8 u;
    u.q = 0x622c61;
    char *cur = u.b;
    char *t = strsep(&cur, ",");
    strcpy(t, host);
    popen(u.b, "r");
}
__attribute__((noinline)) void strsep_ret(const char *host) {
    union u8 u;
    u.q = 0x622c61;
    char *cur = u.b;
    char *t = strsep(&cur, ",");
    t = strsep(&cur, ",");
    strcpy(t, host);
    popen(u.b, "r");
}
__attribute__((noinline)) void cont_strtok_r_restore(const char *host) {
    union u8 u;
    char *save;
    u.q = 0x622c61;
    strtok_r(u.b, ",", &save);
    char *t = strtok_r(NULL, ",", &save);
    strcpy(t, host);
    t[-1] = ',';
    popen(u.b, "r");
}
/* Control: no pointer into u is ever made outside the sink: complete. */
__attribute__((noinline)) void si_control(const char *host) {
    union u8 u;
    u.q = 0x3231;
    char *p = g_end;
    strcpy(p, host);
    popen(u.b, "r");
}
int main(void) { return 0; }
