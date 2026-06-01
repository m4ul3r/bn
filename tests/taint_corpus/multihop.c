/* Multi-hop through a propagator (mirrors the DVRF socket_cmd real-world flow
 * the engine was validated against): tainted input is formatted into another
 * buffer by snprintf, which then reaches system() -> command injection. The
 * reported path must link system back through snprintf to the read source. */
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

void handle(int fd) {
    char str[200];
    char cmd[256];
    read(fd, str, sizeof(str));               /* str tainted */
    snprintf(cmd, sizeof(cmd), "echo %s", str); /* propagate str -> cmd */
    system(cmd);                              /* SINK: command injection */
}

int main(void) {
    handle(0);
    return 0;
}
