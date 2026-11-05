/* The forward all-clear's scope (IC07): a seed reaching a net_write sink, a
 * class off by default, and a seed that only reaches the return value. */
#include <string.h>
#include <sys/socket.h>

__attribute__((noinline)) unsigned scale_count(unsigned n) {
    return n * 4u + 1u;               /* the seed reaches only the return */
}

__attribute__((noinline)) int push_label(int fd, const char *label) {
    return (int)send(fd, label, strlen(label), 0);   /* net_write: opt-in */
}

int main(int argc, char **argv) {
    return (int)scale_count((unsigned)argc) + push_label(0, argv[0]);
}
