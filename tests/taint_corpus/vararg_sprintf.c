/* The tainted value is the SECOND vararg to sprintf (a constant precedes it), so
 * the old "first vararg only" rule missed it. Full vararg propagation taints the
 * dest buffer from arg index 3, which then reaches system(). */
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

void handle(int fd) {
    char name[64];
    char cmd[160];
    read(fd, name, sizeof(name));          /* name tainted */
    sprintf(cmd, "%s %s", "ls", name);     /* name is the 2nd vararg (arg 3) */
    system(cmd);                           /* command injection via cmd */
}

int main(void) {
    handle(0);
    return 0;
}
