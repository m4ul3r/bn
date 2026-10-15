/* Width-less scanf string conversions (#1041 item 4). Synthetic. */
#include <stdio.h>
#include <string.h>
#include <unistd.h>

extern int consume(const char *);

/* fscanf with a width-less scanset into a fixed stack buffer: unbounded. */
__attribute__((noinline)) int load_line(FILE *f) {
    char line[64];
    if (fscanf(f, "%[^\n]", line) != 1)
        return -1;
    return consume(line);
}

/* Same read with a width: bounded, never a sink. */
__attribute__((noinline)) int load_line_bounded(FILE *f) {
    char line[64];
    if (fscanf(f, "%63[^\n]", line) != 1)
        return -1;
    return consume(line);
}

/* sscanf from a received buffer: %d is fine, the second %s is not. */
__attribute__((noinline)) int parse_reply(int fd) {
    char msg[128], name[16];
    int code;
    long n = read(fd, msg, sizeof msg - 1);
    if (n <= 0)
        return -1;
    msg[n] = 0;
    if (sscanf(msg, "%d %s", &code, name) != 2)
        return -1;
    return code + consume(name);
}

/* sscanf of a constant string beside a tainted read: the read never reaches it. */
__attribute__((noinline)) int parse_constant(int fd) {
    char msg[32], name[16];
    int code;
    if (read(fd, msg, sizeof msg) <= 0)
        return -1;
    sscanf("200 ok", "%d %s", &code, name);
    return code + consume(name) + msg[0];
}

int consume(const char *s) { return (int)strlen(s); }

int main(int argc, char **argv) {
    (void)argv;
    return load_line(stdin) + load_line_bounded(stdin) + parse_reply(argc) + parse_constant(argc);
}
