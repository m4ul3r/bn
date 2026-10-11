/* strtok's static state outlives the function that armed it. A tainted
 * `strtok(line, ...)` followed by a call into a helper that continues with
 * `strtok(NULL, ...)` is a flow the per-function continuation model cannot
 * follow -- so the escape must be disclosed and the all-clear withheld. A
 * clean non-NULL `strtok(other, ...)` in between is a precise reset: the
 * continuation after it tokenizes `other`, not the tainted line. */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void helper_next(void) {
    char *n = strtok(NULL, " ");    /* continues whatever state the caller left */
    if (n)
        system(n);
}

void tok_escape(int fd) {
    char line[128];
    read(fd, line, 127);            /* line tainted */
    line[127] = 0;
    strtok(line, " ");              /* arms libc's state with tainted bytes */
    helper_next();                  /* ... which escapes into the helper */
}

void tok_reset(int fd) {
    char line[128];
    char other[32] = "a b c";
    read(fd, line, 127);
    line[127] = 0;
    strtok(line, " ");              /* tainted */
    strtok(other, " ");             /* clean reset of the static state */
    char *x = strtok(NULL, " ");    /* continues `other` */
    if (x)
        system(x);                  /* NOT a tainted sink */
}

int main(void) {
    tok_escape(0);
    tok_reset(0);
    return 0;
}
