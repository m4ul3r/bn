/* Memory-SSA precision (Phase 3C): a tainted value is stored into a heap buffer
 * (*p) and later loaded back, then used as a memcpy length. The AddressOf-only
 * rule misses this; memory-SSA store/load correlation recovers it. The parallel
 * store to *q (a constant) must NOT taint q's load. */
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

void handle(int fd) {
    char *p = malloc(64);
    char *q = malloc(64);
    long t;
    read(fd, &t, sizeof(t));   /* t tainted */
    p[0] = (char)t;            /* store tainted -> *p */
    q[0] = 'x';                /* store constant -> *q (must not taint) */
    char c = p[0];             /* load tainted from *p (via mem-SSA) */
    char d = q[0];             /* load constant from *q */
    char buf[8];
    memcpy(buf, p, c);         /* SINK: tainted length c */
    (void)d;
}

int main(void) {
    handle(0);
    return 0;
}
