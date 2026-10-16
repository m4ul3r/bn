/* #1041 item 1: recv fills a buffer reached through a struct field plus an
 * offset (not keyable), and the same pointer is then handed to a callback. The
 * seed is disclosed as misanchored; the callback is not followed and the leaf
 * says so. Synthetic. */
#include <sys/socket.h>

typedef void (*cb_t)(const unsigned char *);
cb_t g_cb;
struct conn { int fd; unsigned char *buf; };

__attribute__((noinline)) int pump_cb(struct conn *c, long have) {
    unsigned char *p = c->buf + have;
    recv(c->fd, p, 64, 0);
    g_cb(p);
    return 0;
}

int main(int argc, char **argv) {
    struct conn c = {argc, (unsigned char *)argv[0]};
    return pump_cb(&c, argc);
}
