/* Attacker bytes are copied into a writable GLOBAL by a bounded copy (no sink
 * of its own), and a later call with NO tainted argument hands those bytes to
 * a modeled sink. Forward taint keeps its taint set per function and passes
 * nothing but arguments across a call, so the consumer used to be reached by
 * nothing at all: zero sinks, zero leaves and a true all-clear.
 *
 * `run_hook` must raise a global_memory_continuation frontier naming g_dir;
 * `no_global`, which references no global, must stay silent (the frontier is
 * evidence-gated, not a marker on every call after a global write).
 * Built with -fno-builtin so the bounded copy stays a modeled call. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char g_dir[256];

__attribute__((noinline)) void run_hook(void) {
    char cmd[300];
    snprintf(cmd, sizeof cmd, "%s/hook.sh", g_dir);  /* reads the tainted global */
    system(cmd);                                     /* modeled sink downstream */
}

__attribute__((noinline)) void no_global(void) {
    putchar('.');                                    /* touches no global */
}

int main(int argc, char **argv) {
    if (argc < 2)
        return 1;
    strncpy(g_dir, argv[1], sizeof g_dir - 1);       /* bounded: not a sink */
    no_global();                                     /* control: no frontier */
    run_hook();                                      /* the continuation */
    return 0;
}
