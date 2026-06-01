/* Backward interprocedural: the memcpy length is a parameter of the callee.
 * A backward slice from the sink must continue up into the caller and reach the
 * recv() that produced the length (the source). */
#include <sys/socket.h>
#include <string.h>
#include <unistd.h>

static void use_len(char *dst, char *src, int n) {
    memcpy(dst, src, n);    /* SINK: length n arrives as a parameter */
}

void handler(int fd) {
    char buf[64];
    char out[64];
    int n = recv(fd, buf, sizeof(buf), 0);   /* recv return -> n tainted (source) */
    use_len(out, buf, n);                     /* n crosses the call boundary */
}

int main(void) {
    handler(0);
    return 0;
}
