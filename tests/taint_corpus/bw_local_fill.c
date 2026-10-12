/* Backward through a local buffer's CONTENTS. The sink reads the bytes of a
 * stack buffer it is handed by address; the slice must continue through the
 * call that wrote them (a modeled snprintf) to that call's operands, or -- when
 * the writer is an unmodeled in-binary helper -- stop with an honest
 * `address_of_local_unresolved` leaf instead of claiming a complete slice. */
#include <stdio.h>
#include <string.h>

static void compose(char *dst, unsigned n, const char *host) {
    strncpy(dst, "ping ", n);
    strncat(dst, host, n - 6);
}

void run_modeled(const char *host) {
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "ping %s", host);   /* modeled fill from a parameter */
    popen(cmd, "r");                                /* SINK: contents of cmd */
}

void run_opaque(const char *host) {
    char cmd[128];
    compose(cmd, sizeof(cmd), host);                /* unmodeled writer handed &cmd */
    popen(cmd, "r");                                /* SINK: contents of cmd */
}

int main(int argc, char **argv) {
    run_modeled(argc > 1 ? argv[1] : "localhost");
    run_opaque(argc > 1 ? argv[1] : "localhost");
    return 0;
}
