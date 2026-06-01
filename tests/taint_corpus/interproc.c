/* Interprocedural: the tainted buffer is passed to a helper, and the sink
 * (memcpy with a tainted length) lives inside the helper. Forward taint must
 * descend into copy_it and bubble the finding back up to handler. */
#include <unistd.h>
#include <string.h>

static void copy_it(char *src, char *dst) {
    memcpy(dst, src, (size_t)(unsigned char)src[0]);  /* SINK inside callee */
}

void handler(int fd) {
    char buf[64];
    char out[16];
    read(fd, buf, sizeof(buf));   /* buf tainted */
    copy_it(buf, out);            /* tainted buffer crosses the call boundary */
}

int main(void) {
    handler(0);
    return 0;
}
