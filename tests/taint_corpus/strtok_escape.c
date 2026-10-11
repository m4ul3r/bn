/* strtok's static state outlives the function that armed it. A tainted
 * `strtok(line, ...)` followed by a call into a helper that continues with
 * `strtok(NULL, ...)` is a flow the per-function continuation model cannot
 * follow -- so the escape must be disclosed and the all-clear withheld. A
 * clean non-NULL `strtok(other, ...)` in between is a precise reset ONLY when
 * it lies on EVERY path from the tainted call to the `strtok(NULL)`. A clean
 * call under `if (mode)` leaves the tainted line on the `mode == 0` path, and
 * a priming call before a loop dominates the continuation yet the tainted call
 * at the bottom of the body reaches it around the back-edge -- so in both the
 * continuation stays a real flow. */
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

void tok_reset_conditional(int fd, int mode) {
    char line[128];
    char other[32] = "a b c";
    read(fd, line, 127);
    line[127] = 0;
    strtok(line, " ");              /* tainted */
    if (mode)
        strtok(other, " ");         /* clean reset on SOME paths only */
    char *x = strtok(NULL, " ");    /* mode == 0: continues the tainted line */
    if (x)
        system(x);                  /* a real command injection */
}

void tok_prime_loop(int fd, int n) {
    char line[128];
    char other[32] = "a b c";
    read(fd, line, 127);
    line[127] = 0;
    strtok(other, " ");             /* clean priming call BEFORE the loop */
    for (int i = 0; i < n; i++) {
        char *x = strtok(NULL, " ");  /* iteration 2+: continues `line` */
        if (x)
            system(x);              /* a real command injection */
        strtok(line, " ");          /* tainted, at the bottom of the body */
    }
}

int main(int argc, char **argv) {
    (void)argv;
    tok_escape(0);
    tok_reset(0);
    tok_reset_conditional(0, argc);
    tok_prime_loop(0, argc);
    return 0;
}
