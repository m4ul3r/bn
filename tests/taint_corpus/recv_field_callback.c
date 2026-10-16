/* #1050 review round 4: the receive buffer pointer is loaded from a connection
 * field into a local; the field is reloaded for a harmless consumer (the reload
 * correlates with the receive's slot); the saved local then goes to a callback
 * bn cannot resolve. The callback is an unchecked consumer of the received
 * bytes, so the run is not an all-clear. The control has no callback.
 * Synthetic. */
#include <stdio.h>
#include <sys/socket.h>

typedef void (*cb_t)(const char *);
cb_t g_cb;
struct conn { int fd; char *buf; };

__attribute__((noinline)) int field_cb_copy(struct conn *c) {
    char *p = c->buf;
    recv(c->fd, p, 64, 0);
    puts(c->buf);
    g_cb(p);
    return 0;
}

__attribute__((noinline)) int field_no_cb(struct conn *c) {
    char *p = c->buf;
    recv(c->fd, p, 64, 0);
    puts(c->buf);
    return 0;
}

int main(int argc, char **argv) {
    struct conn c = {argc, argv[0]};
    return field_cb_copy(&c) + field_no_cb(&c);
}
