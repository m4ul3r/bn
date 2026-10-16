/* A call handed &buf may retain or plant a pointer into it (#1053 review 4):
 * strtok_r's save pointer and strtol's end pointer hold buf + 2 afterwards, so a
 * later reload -- before, after or without an explicit stash -- may alias buf and
 * the strcpy through it is a frontier, never an unrelated pointer. Analyze only. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
char *g_end;
char *g_save;
char *stash;
/* glibc strtok_r with no delimiter: returns buf, *save = buf + strlen(buf) */
__attribute__((noinline)) void epoch_strtok_r(const char *host) {
    char buf[64];
    strcpy(buf, "ab");
    strtok_r(buf, ",", &g_save);  /* g_save = buf + 2 */
    char *p = g_save;             /* loaded before &buf escapes */
    stash = buf;                  /* first store escape of &buf */
    strcpy(p, host);              /* buf = "ab" + host */
    popen(buf, "r");
}
__attribute__((noinline)) void noescape_strtok_r(const char *host) {
    char buf[64];
    strcpy(buf, "ab");
    strtok_r(buf, ",", &g_save);
    char *p = g_save;
    strcpy(p, host);
    popen(buf, "r");
}
__attribute__((noinline)) void late_load_strtok_r(const char *host) {
    char buf[64];
    strcpy(buf, "ab");
    strtok_r(buf, ",", &g_save);
    stash = buf;
    char *p = g_save;             /* loaded after the escape: control */
    strcpy(p, host);
    popen(buf, "r");
}
/* strtol(buf, &g_end): g_end = buf + 2 for "12" */
__attribute__((noinline)) void epoch_strtol(const char *host) {
    char buf[64];
    strcpy(buf, "12");
    strtol(buf, &g_end, 10);
    char *p = g_end;
    stash = buf;
    strcpy(p, host);
    popen(buf, "r");
}
__attribute__((noinline)) void noescape_strtol(const char *host) {
    char buf[64];
    strcpy(buf, "12");
    strtol(buf, &g_end, 10);
    char *p = g_end;
    strcpy(p, host);
    popen(buf, "r");
}
__attribute__((noinline)) void epoch_strtol_local(const char *host) {
    char buf[64];
    char *end;
    strcpy(buf, "12");
    strtol(buf, &end, 10);
    char *p = end;
    stash = buf;
    strcpy(p, host);
    popen(buf, "r");
}
int main(void) { return 0; }
