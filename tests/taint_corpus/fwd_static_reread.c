/* A function-static global the SAME function reads back on its next call.
 *
 * `next_index` loads its static `cursor` to index a table, then stores this
 * call's input into it and returns: the tainted store reaches the load only
 * on the next activation, which the walk never sees. Its exit must keep a
 * blocking `coarse_memory_store` naming `next_index`'s own load.
 *
 * `latest_index` is the control: it stores first and loads after, so every
 * read sees this activation's own value -- the one the walk followed -- and
 * nothing else references the static. */
#include <unistd.h>

static unsigned char table[256];

__attribute__((noinline)) int next_index(int fd) {
    static int cursor;
    unsigned char c = table[cursor & 0xff];   /* the previous call's input */
    int n;
    if (read(fd, &n, sizeof n) != (ssize_t)sizeof n) return -1;
    cursor = n;                               /* tainted; no later use here */
    return c;
}

__attribute__((noinline)) int latest_index(int fd) {
    static int newest;
    int n;
    if (read(fd, &n, sizeof n) != (ssize_t)sizeof n) return -1;
    newest = n;
    return table[newest & 0xff];
}

int main(int argc, char **argv) {
    (void)argv;
    table[argc & 0xff] = (unsigned char)argc;
    return next_index(0) + next_index(0) + latest_index(argc) + latest_index(0);
}
