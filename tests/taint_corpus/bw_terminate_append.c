/* The `fill; terminate; append; sink` idiom. The terminator `buf[n] = 0` lifts
 * to MLIL_SET_VAR_ALIASED_FIELD, a memory definition with no `src_memory`; the
 * backward local-buffer walk must step through it (via `ssa_memory_version`)
 * so the OLDER writer -- the one that copied the parameter / received the
 * input -- is not dropped and the later literal append cannot pose as a
 * complete origin. Built at -O1 (FORTIFY `_chk` callees), the shape a reviewer
 * reproduced; the slice must bottom out at the parameter (no caller: `main`
 * does not call it) / the `read` source, with no unresolved leaf. */
#include <stdio.h>
#include <string.h>
#include <unistd.h>

void fill_terminate_append(const char *host) {
    char cmd[128];
    strcpy(cmd, host);          /* OLDER writer: the parameter */
    cmd[100] = 0;               /* aliased-field terminator (no src_memory) */
    strcat(cmd, " -c 1");       /* NEWER writer: a literal */
    popen(cmd, "r");            /* SINK: contents of cmd */
}

void read_terminate_append(int fd) {
    char cmd[128];
    read(fd, cmd, 100);         /* OLDER writer: a receive source */
    cmd[100] = 0;
    strcat(cmd, " -c 1");
    popen(cmd, "r");            /* SINK */
}

void copy_from_tmp(int fd) {
    char tmp[64];
    char cmd[128];
    read(fd, tmp, 63);          /* the nested buffer's source */
    tmp[63] = 0;
    strcpy(cmd, tmp);           /* `&tmp` operand: a buffer, not a literal */
    popen(cmd, "r");            /* SINK */
}

int main(void) {
    read_terminate_append(0);
    copy_from_tmp(0);
    return 0;
}
