/* Synthetic, license-clean fixture for bn. No target/proprietary data.
 *
 * The backward mirror of conn_recv_vtable.c: an attacker-derived length flows
 * into a write/emit primitive dispatched through the connection vtable
 * (`conn->ops->send(conn, dst, n)`). `conn` is a parameter, so value-set cannot
 * pin the slot -- a `--resolve-map` pin is required to anchor the SINK. Backward
 * taint regression for indirect sink anchoring (#282):
 *   taint backward -f emit --sink arg:sock_send:2 --resolve-map <map>
 * must anchor at the indirect call and slice the length back to the `n`
 * parameter origin. Analyzed only; never executed on input.
 */
#include <unistd.h>
#include <string.h>

struct conn;
typedef long (*io_fn)(struct conn *c, char *buf, unsigned n);
struct conn_ops { io_fn recv; io_fn send; };
struct conn { const struct conn_ops *ops; int fd; };

__attribute__((noinline, used))
static long sock_recv(struct conn *c, char *buf, unsigned n) { return read(c->fd, buf, n); }
__attribute__((noinline, used))
static long sock_send(struct conn *c, char *buf, unsigned n) { return write(c->fd, buf, n); }

static const struct conn_ops tcp_ops = { sock_recv, sock_send };

__attribute__((noinline, used))
void emit(struct conn *c, char *src, unsigned n) {
    char dst[32];
    if (n > sizeof dst) n = sizeof dst;
    memcpy(dst, src, n);
    c->ops->send(c, dst, n);               /* indirect SINK site (vtable) */
}

int main(int argc, char **argv) {
    static struct conn c = { &tcp_ops, 0 };
    emit(&c, argv[0], (unsigned)argc);
    return 0;
}
