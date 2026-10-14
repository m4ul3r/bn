/* #931/#933: may-writes, definite byte extents and consumed members. */
#include <stdio.h>
#include <string.h>
#include <unistd.h>

struct request { int id; char cmd[64]; int flags; };

__attribute__((noinline)) void zero_read(int fd) {
    char buf[64]; read(fd, buf, 0); popen(buf, "r");
}
__attribute__((noinline)) void short_read(int fd) {
    char buf[64]; read(fd, buf, 1); popen(buf, "r");
}
__attribute__((noinline)) void full_request(int fd) {
    char buf[64]; read(fd, buf, sizeof buf); popen(buf, "r");
}
__attribute__((noinline)) void suffix_read(int fd) {
    char buf[64]; read(fd, buf, 1); popen(buf + 32, "r");
}
__attribute__((noinline)) void indexed_row(int fd, unsigned i) {
    char rows[4][32]; read(fd, rows[0], 32); popen(rows[i & 3], "r");
}
__attribute__((noinline)) void append_only(const char *input) {
    char buf[64]; strcat(buf, input); popen(buf, "r");
}
__attribute__((noinline)) void member_string(const char *input) {
    struct request r; snprintf(r.cmd, sizeof r.cmd, "%s", input); popen(r.cmd, "r");
}
__attribute__((noinline)) void member_copy(const char *input) {
    struct request r; char out[64];
    snprintf(r.cmd, sizeof r.cmd, "%s", input); strcpy(out, r.cmd); popen(out, "r");
}
__attribute__((noinline)) void member_bytes(int fd, const char *input) {
    struct request r; memcpy(r.cmd, input, sizeof r.cmd); write(fd, r.cmd, sizeof r.cmd);
}
__attribute__((noinline)) void partial_member(int fd, const char *input) {
    struct request r; memcpy(r.cmd, input, 8); write(fd, r.cmd, sizeof r.cmd);
}
__attribute__((noinline)) void unknown_member(int fd, const char *input, unsigned n) {
    struct request r; memcpy(r.cmd, input, n); write(fd, r.cmd, sizeof r.cmd);
}
__attribute__((noinline)) void string_not_full_bytes(int fd, const char *input) {
    struct request r; snprintf(r.cmd, sizeof r.cmd, "%s", input); write(fd, r.cmd, sizeof r.cmd);
}
__attribute__((noinline)) void byte_suffix(int fd, const char *input) {
    char buf[64]; memcpy(buf, input, sizeof buf); write(fd, buf + 32, 16);
}
__attribute__((noinline)) void byte_suffix_gap(int fd, const char *input) {
    char buf[64]; memcpy(buf, input, 16); write(fd, buf + 32, 16);
}
int main(void) { return 0; }
