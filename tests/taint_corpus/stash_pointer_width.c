#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
volatile uintptr_t stash;
__attribute__((noinline)) void truncated_stash(void) {
    char buf[64];
    stash = (uintptr_t)buf;
    char *p = (char *)(uintptr_t)*(volatile uint32_t *)&stash;
    snprintf(p, 64, "%s", "id");
    popen(buf, "r");
}
__attribute__((noinline)) void full_stash(void) {
    char buf[64];
    stash = (uintptr_t)buf;
    char *p = (char *)stash;
    snprintf(p, 64, "%s", "id");
    popen(buf, "r");
}
__attribute__((noinline)) void copied_epoch(int fd) {
    unsigned char buf[64];
    memset(buf, 7, sizeof buf);
    unsigned char *p = buf + 2;
    size_t n = p[3];
    read(fd, buf, 64);
    free(malloc(n));
}
int main(void) { return 0; }
