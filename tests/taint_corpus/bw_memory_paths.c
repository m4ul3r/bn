/* Memory-path honesty for the backward local-buffer walk. A conditional fill
 * leaves the sink reading a memory phi with a bare entry arm (no writer on
 * one path); a source buffer overwritten AFTER it was copied cannot reach the
 * copy; a buffer whose address escaped to a global may be written by any
 * unmodeled call that runs after the escape. */
#include <stdio.h>
#include <string.h>

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

int main(int argc, char **argv) {
    const char *h = argc > 1 ? argv[1] : "localhost";
    conditional_fill(argc);
    chained_fill(h);
    escape_then_call(h);
    return 0;
}
