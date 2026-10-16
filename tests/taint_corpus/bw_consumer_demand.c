/* #931/#933 review round 1: the consumed region is the CONSUMER's demand,
 * carried back through copies, arithmetic and phis (widened when the offset
 * is not one constant), and "no call certifies" holds on every consumer path:
 * a pointer a call returned into a local, a heap object and a pointer
 * parameter are never complete. Analyze only; never execute. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
volatile int review_flag;
#define NI __attribute__((noinline))
struct pre { char p[8]; char cmd[8]; };
struct tri { char a[8]; char cmd[8]; char cmd2[8]; };
struct req { int id; char cmd[16]; };
NI void widened_array(void) { char b[16]; __builtin_memcpy(b + 8, "abcdefg", 8); char *p = b + 8; popen(p - 8, "r"); }
NI void loop_back(void) { struct pre s; __builtin_memcpy(s.cmd, "abcdefg", 8); char *p = s.cmd; while (review_flag && p > s.p) p--; popen(p, "r"); }
NI void phi_minus(int k) { struct tri s; __builtin_memcpy(s.cmd, "abcdefg", 8); __builtin_memcpy(s.cmd2, "hijklmn", 8); char *p = k ? s.cmd : s.cmd2; if (review_flag) p -= 4; popen(p, "r"); }
NI void narrowed(void) { struct req s; __builtin_memcpy(s.cmd, "ls -la /tmp/xyz", 16); char *p = (char *)&s; popen(p + 4, "r"); }
NI void direct_member(void) { struct req s; __builtin_memcpy(s.cmd, "ls -la /tmp/xyz", 16); popen(s.cmd, "r"); }
NI void fgets_ret_write(FILE *f) { char c[64]; char *p = fgets(c, 64, f); if (p) write(1, p, 64); }
NI void fgets_ret_popen(FILE *f) { char c[64]; char *p = fgets(c, 64, f); if (p) popen(p, "r"); }
NI void heap_fill(const char *host) { char *b = malloc(64); strcpy(b, host); popen(b, "r"); }
NI void param_fill(char *b, const char *host) { strcpy(b, host); popen(b, "r"); }
NI void nonul_memset(int v) { char c[64]; memset(c, v, 64); popen(c, "r"); }
int main(void) { return 0; }
