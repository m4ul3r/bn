/* Synthetic, license-clean fixture for bn. No target/proprietary data.
 *
 * A realistic connection object: a struct holds a const vtable of I/O function
 * pointers (`conn->ops->recv`). Because `conn` is a parameter, `conn->ops` is a
 * runtime load that Binary Ninja's value-set CANNOT pin -- so anchoring the recv
 * SOURCE at the indirect call requires an agent `--resolve-map` pin (the
 * dominant real-server I/O shape, e.g. a socket read routed through a transport
 * vtable). Forward taint regression for indirect recv-source anchoring (#282):
 *   taint forward -f handle --source arg:sock_recv:1 --resolve-map <map>
 * must anchor at `conn->ops->recv(conn, buf, n)` and propagate the attacker
 * bytes into the memcpy length sink. Analyzed only; never executed on input.
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

static char g_out[64];

__attribute__((noinline, used))
void handle(struct conn *c) {
    char buf[64];
    c->ops->recv(c, buf, sizeof buf);      /* indirect SOURCE site (vtable) */
    unsigned len = (unsigned char)buf[0];  /* attacker-controlled length     */
    memcpy(g_out, buf, len);               /* overflow_len sink              */
}

int main(int argc, char **argv) {
    static struct conn c = { &tcp_ops, 0 };
    (void)argv;
    handle(&c);
    return argc;
}
