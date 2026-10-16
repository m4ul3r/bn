/* #931/#933: only plain stores prove the consumed region initialised, and the
 * region runs from the sink's pointer to the end of the object. gcc expands a
 * short literal strcpy and a constant memset into inline stores, which BN
 * outlines as `__builtin_*` calls; their extent counts only after the walk
 * checks their LLIL is that store run. Analyze only. */
#include <stdio.h>
#include <string.h>

struct tail_cmd { int id; char cmd[16]; };
struct head_cmd { char cmd[16]; int flags; };

/* cmd is the last member and its 16 bytes are stored: complete. */
__attribute__((noinline)) void member_stored(void) {
    struct tail_cmd s;
    strcpy(s.cmd, "ls -la /tmp/xyz");
    popen(s.cmd, "r");
}
/* The same stores, but the sink reads the whole struct: id is never written. */
__attribute__((noinline)) void whole_struct(void) {
    struct tail_cmd s;
    strcpy(s.cmd, "ls -la /tmp/xyz");
    popen((char *)&s, "r");
}
/* cmd is followed by flags, which a forward reader may reach. */
__attribute__((noinline)) void member_then_tail(void) {
    struct head_cmd s;
    strcpy(s.cmd, "ls -la /tmp/xyz");
    popen(s.cmd, "r");
}
/* 15 bytes stored into a 16-byte buffer: the last byte is unproven. */
__attribute__((noinline)) void short_literal(void) {
    char cmd[16];
    strcpy(cmd, "ls -la /tmp/xy");
    popen(cmd, "r");
}
/* A constant memset of the whole buffer, then a 1-byte store. */
__attribute__((noinline)) void zeroed(int c) {
    char cmd[64];
    memset(cmd, 0, sizeof cmd);
    if (c)
        cmd[0] = 'x';
    popen(cmd, "r");
}
/* A real call with the same arguments proves nothing. */
__attribute__((noinline)) void call_fill(const char *x) {
    char cmd[16];
    strncpy(cmd, x, sizeof cmd);
    popen(cmd, "r");
}
int main(int argc, char **argv) {
    member_stored(); whole_struct(); member_then_tail(); short_literal();
    zeroed(argc); call_fill(argv[0]);
    return 0;
}
