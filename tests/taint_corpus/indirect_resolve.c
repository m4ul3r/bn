/* Hard case resolved (Phase 3A): the sink is reachable only through a const
 * function-pointer table. Value-set analysis pins the table targets, so taint
 * is followed into BOTH handlers (run_cmd -> system, run_copy -> strcpy). */
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

static void run_cmd(char *p) { system(p); }
static void run_copy(char *p) { char b[16]; strcpy(b, p); }

static void (*const table[2])(char *) = { run_cmd, run_copy };

void dispatch(int sel, char *buf) {
    table[sel & 1](buf);   /* indirect call via const table -> VSA-resolvable */
}

void handler(int fd) {
    char buf[128];
    read(fd, buf, sizeof(buf));
    dispatch(1, buf);
}

int main(void) {
    handler(0);
    return 0;
}
