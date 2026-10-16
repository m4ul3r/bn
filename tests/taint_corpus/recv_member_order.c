/* #1050 review round 2: system(p) runs BEFORE recv(fd, p, ..) fills an
 * unkeyed receive buffer (struct field plus offset), so the received bytes
 * cannot reach it. Synthetic. */
#include <stdlib.h>
#include <sys/socket.h>

struct conn { int fd; int have; char *buf; };

__attribute__((noinline)) int sys_before(struct conn *c) {
    char *p = c->buf + c->have;
    int r = system(p);
    recv(c->fd, p, 64, 0);
    return r;
}

int main(int argc, char **argv) {
    struct conn c = {argc, 0, argv[0]};
    return sys_before(&c);
}
