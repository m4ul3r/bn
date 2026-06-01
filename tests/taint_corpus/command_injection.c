/* Positive (external model): tainted buffer flows as the pointer argument to
 * system(), exercising pointer-arg sink detection + the function-model DB. */
#include <unistd.h>
#include <stdlib.h>

void handle(int fd) {
    char cmd[128];
    read(fd, cmd, sizeof(cmd));   /* cmd tainted */
    system(cmd);                  /* SINK: tainted command (pointer arg) */
}

int main(void) {
    handle(0);
    return 0;
}
