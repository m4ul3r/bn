/* Memory-path honesty for the backward local-buffer walk. A conditional fill
 * leaves the sink reading a memory phi with a bare entry arm (no writer on
 * one path); a source buffer overwritten AFTER it was copied cannot reach the
 * copy; a buffer whose address escaped to a global may be written by any
 * unmodeled call that runs after the escape -- and by a MODELED call handed
 * the stashed pointer back (`strcpy(g_stash, host)`), which is a fill of the
 * buffer, not a call to trust to its model; a 1-byte terminator after a
 * conditional fill does not cover the bare arm; a memset arm is a literal
 * fill, but a short read on the other arm does not prove whole-object coverage. */
#include <stdio.h>
#include <string.h>
#include <unistd.h>

char *g_escaped;

void fill_global(const char *host) {
    strcpy(g_escaped, host);
}

void conditional_fill(int flag) {
    char dst[128];
    if (flag)
        snprintf(dst, sizeof dst, "fixed");  /* one arm writes dst ... */
    popen(dst, "r");                         /* SINK: ... the other reads the stack */
}

void chained_fill(const char *input) {
    char src[64];
    char dst[64];
    snprintf(src, sizeof src, "fixed");      /* what strcpy actually copied */
    strcpy(dst, src);
    snprintf(src, sizeof src, "%s", input);  /* AFTER the copy: cannot reach dst */
    popen(dst, "r");                         /* SINK: contents of dst */
}

void escape_then_call(const char *host) {
    char cmd[128];
    cmd[0] = 0;
    g_escaped = cmd;                         /* &cmd escapes to a global */
    fill_global(host);                       /* unmodeled; may write through it */
    popen(cmd, "r");                         /* SINK */
}

void escape_then_modeled(const char *host) {
    char cmd[128];
    cmd[0] = 0;
    g_escaped = cmd;                         /* &cmd escapes to a global */
    strcpy(g_escaped, host);                 /* modeled, reloads the stash: a FILL of cmd */
    popen(cmd, "r");                         /* SINK: runs `host` */
}

void terminated_conditional_fill(int fd, int flag) {
    char buf[64];
    if (flag)
        read(fd, buf, 63);                   /* one arm writes buf ... */
    buf[63] = 0;                             /* ... a 1-byte store covers NO path */
    popen(buf, "r");                         /* SINK: on flag == 0 reads the stack */
}

void read_or_memset(int fd, int flag) {
    char buf[64];
    if (flag)
        read(fd, buf, 63);
    else
        memset(buf, 0, sizeof buf);          /* a literal fill: this arm IS written */
    buf[63] = 0;
    popen(buf, "r");                         /* SINK: short read arm remains uncertain */
}

int main(int argc, char **argv) {
    const char *h = argc > 1 ? argv[1] : "localhost";
    conditional_fill(argc);
    chained_fill(h);
    escape_then_call(h);
    escape_then_modeled(h);
    terminated_conditional_fill(0, argc);
    read_or_memset(0, argc);
    return 0;
}
