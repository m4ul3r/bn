/* Negative: input is read but the copy length is a compile-time constant,
 * so no tainted value reaches the memcpy length. Guards against over-tainting. */
#include <unistd.h>
#include <string.h>

void safe_copy(int fd) {
    char buf[64];
    char dst[64];
    read(fd, buf, sizeof(buf));    /* buf tainted */
    memcpy(dst, buf, sizeof(dst)); /* length is constant -> NOT a finding */
}

int main(void) {
    safe_copy(0);
    return 0;
}
