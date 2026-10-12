/* A writer input that is the ADDRESS of a mutable global. Both `"ping %s"` and
 * `g_host` reach the modeled fill as `MLIL_CONST_PTR` operands, but only the
 * former is a literal: it points into read-only data. `g_host` is a `.bss`
 * array any other function may have filled (`read(fd, g_host, ...)`), so the
 * bytes it supplies are unaccounted for -- a frontier leaf that names the
 * global's address and withholds the complete-slice claim, never "complete". */
#include <stdio.h>
#include <unistd.h>

char g_host[64];

void set_host(int fd) {
    read(fd, g_host, 63);
    g_host[63] = 0;
}

void ping_global_array(void) {
    char cmd[128];
    sprintf(cmd, "ping %s", g_host);     /* `g_host` is an address into .bss */
    popen(cmd, "r");                      /* SINK: runs whatever set_host read */
}

void ping_literal(void) {
    char cmd[128];
    sprintf(cmd, "ping %s", "localhost"); /* both operands point into .rodata */
    popen(cmd, "r");                      /* SINK: a complete, literal slice */
}

int main(void) {
    set_host(0);
    ping_global_array();
    ping_literal();
    return 0;
}
