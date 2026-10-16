/* #1050 review round 2: an unkeyed receive buffer (struct field plus offset)
 * whose pointer is also used elsewhere. A use that runs only BEFORE the
 * receive cannot see the received bytes; one after it, or around a loop
 * back-edge, can. Synthetic. */
#include <stdlib.h>
#include <sys/socket.h>

typedef void (*cb_t)(const char *);
cb_t g_cb;
struct conn { int fd; int have; char *buf; };

__attribute__((noinline)) int sys_before(struct conn *c) {
    char *p = c->buf + c->have;
    int r = system(p);
    recv(c->fd, p, 64, 0);
    return r;
}

__attribute__((noinline)) int cb_loop(struct conn *c, int n) {
    char *p = c->buf + c->have;
    while (n-- > 0) {
        g_cb(p);
        recv(c->fd, p, 64, 0);
    }
    return 0;
}

__attribute__((noinline)) int sys_after(struct conn *c) {
    char *p = c->buf + c->have;
    recv(c->fd, p, 64, 0);
    return system(p);
}

int main(int argc, char **argv) {
    struct conn c = {argc, 0, argv[0]};
    return sys_before(&c) + cb_loop(&c, argc) + sys_after(&c);
}
