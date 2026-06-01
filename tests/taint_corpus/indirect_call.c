/* Hard case (indirect call): a tainted buffer flows into a call through a
 * function pointer. The engine must surface this as an unresolved leaf rather
 * than silently dropping the edge. */
#include <unistd.h>

typedef void (*cb_t)(char *);

void invoke(int fd, cb_t f) {
    char buf[64];
    read(fd, buf, sizeof(buf));   /* buf tainted */
    f(buf);                       /* INDIRECT call reached by taint */
}

int main(void) {
    invoke(0, 0);
    return 0;
}
