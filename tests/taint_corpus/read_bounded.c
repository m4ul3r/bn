/* Bounded length: the memcpy length is the read() return value, which is
 * provably <= the constant count argument (sizeof buf). Forward taint must
 * report this as a bounded copy (bounded_len with the source bound), NOT an
 * unbounded/over-stated overflow. Sourcing via call:read seeds read's return
 * (and its output buffer) so the length is tainted in the first place. */
#include <unistd.h>
#include <string.h>
#include <stdio.h>

void process_bounded(int fd) {
    char buf[64];
    char dst[64];
    long n = read(fd, buf, sizeof(buf));   /* n is tainted; n <= 64 (the count) */
    if (n > 0) {
        memcpy(dst, buf, n);               /* SINK: length is read-bounded, not overflow */
    }
    printf("%ld\n", n);
}

int main(void) {
    process_bounded(0);
    return 0;
}
