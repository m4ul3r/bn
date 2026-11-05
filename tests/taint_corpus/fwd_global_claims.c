/* Forward taint claim gaps around globals (P2-11), a struct-field store
 * (P2-08) and a gated receive length (P2-10), one function per shape.
 *
 *   exit_case     read() fills g_exit and returns; use_exit() reads it
 *                 elsewhere -> a blocking frontier at the write, never an
 *                 all-clear.
 *   cleared_case  the same fill, then memset() clears every tainted byte
 *                 before the return -> nothing is live; all-clear.
 *   alias_case    read() fills g_alias, then use_alias() copies from it
 *                 through `g_alias_ptr`, a pointer global initialised to it
 *                 -> a continuation naming use_alias.
 *   param_case    copy_name(g_name, arg) fills a global through a pointer
 *                 parameter -> the global is keyed from the caller's
 *                 argument and the later use_name() call reads it.
 *   heap_case     the block g_heap points at is filled, then reloaded
 *                 through the global in the same function and measured
 *                 into a malloc() length -> the alloc_size sink is reached.
 *   field_case    a tainted number goes into g_cfg.port; a DIFFERENT
 *                 field's strlen() sizes a memcpy -> no flow.
 *   recv_case     a tainted length reaches read() -> with the default sink
 *                 classes a gated-sink frontier; with recv_overflow a sink.
 *   recv_loop     the same length drives a fill loop (`n - got` per read)
 *                 -> the same gated-sink frontier, never an all-clear.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define N __attribute__((noinline))

char g_exit[64];
char g_alias[64];
char *g_alias_ptr = g_alias;
char g_name[64];
char *g_heap;

struct config {
    char *name;
    long pad;
    long port;
};
struct config g_cfg = {"default", 0, 0};

N void use_exit(void) { char d[16]; strcpy(d, g_exit); puts(d); }
N void exit_case(void) { read(0, g_exit, sizeof g_exit); }
N void cleared_case(void) {
    read(0, g_exit, sizeof g_exit);
    memset(g_exit, 0, sizeof g_exit);
}

N void use_alias(void) { char d[16]; strcpy(d, g_alias_ptr); puts(d); }
N void alias_case(void) { read(0, g_alias, sizeof g_alias); use_alias(); }

N void copy_name(char *to, const char *from) { strncpy(to, from, sizeof g_name - 1); }
N void use_name(void) { system(g_name); }
N void param_case(char *arg) { copy_name(g_name, arg); use_name(); }

N void heap_case(char *arg, size_t n) {
    g_heap = malloc(n);
    memcpy(g_heap, arg, n);
    char *p = g_heap;
    char *q = malloc(strlen(p) + 1);
    puts(q);
}

N void field_case(char *arg) {
    g_cfg.port = strtol(arg, NULL, 10);
    size_t n = strlen(g_cfg.name) + 1;
    char *copy = malloc(64);
    memcpy(copy, g_cfg.name, n);
    puts(copy);
}

N int recv_case(int fd, char *buf, int n) { return (int)read(fd, buf, n); }

N int recv_loop(int fd, char *buf, int n) {
    int got = 0;
    while (got < n) {
        int r = (int)read(fd, buf + got, (size_t)(n - got));
        if (r <= 0)
            break;
        got += r;
    }
    return got;
}

int main(int argc, char **argv) {
    char b[64];
    switch (argc) {
    case 1: exit_case(); use_exit(); break;
    case 2: cleared_case(); use_exit(); break;
    case 3: alias_case(); break;
    case 4: param_case(argv[1]); break;
    case 5: heap_case(argv[1], (size_t)argc); break;
    case 6: field_case(argv[1]); break;
    case 7: recv_loop(0, b, argc); break;
    default: recv_case(0, b, argc); break;
    }
    return 0;
}
