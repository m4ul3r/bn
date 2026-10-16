/* #931/#933 review round 3: a region proof from stores is a fact, never
 * certification -- a stack array BN splits into fragments (with or without a
 * later rename), and a stored array whose upper half `read` overwrites, are
 * incomplete -- and a pointer `fgets` returns over a writable global buffer or
 * global struct member is the same `global` frontier as the global itself.
 * Analyze only; never execute. */
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#define NI __attribute__((noinline))
char g_buf[64];
struct req { int id; char cmd[60]; } g_req;
NI void rename_fragment(void) { char b[32]; __builtin_memcpy(b, "abcdefghijklmnop", 16); popen(b, "r"); }
NI void upper_half_read(int fd) { char b[16]; __builtin_memcpy(b, "abcdefghijklmno", 16); read(fd, b + 8, 8); popen(b, "r"); }
NI void returned_global(FILE *f) { char *p = fgets(g_buf, 64, f); if (p) write(1, p, 64); }
NI void returned_global_member(FILE *f) { char *p = fgets(g_req.cmd, 60, f); if (p) write(1, p, 60); }
int main(void) { return 0; }
