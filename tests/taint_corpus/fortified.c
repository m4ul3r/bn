/* FORTIFY_SOURCE build: read() taints a stack buffer, a byte from it becomes a
 * length, and the copy lowers to __memcpy_chk (not plain memcpy). Exercises the
 * fortified _chk model + lookup (underscore-stripped key). Compiled with
 * -O2 -D_FORTIFY_SOURCE=2 (see cflags in the EXPECTED file) so the _chk call is
 * actually emitted. */
#include <unistd.h>
#include <string.h>

void process(int fd) {
    char buf[64];
    char dst[16];
    read(fd, buf, sizeof(buf));            /* buf tainted */
    unsigned char n = (unsigned char)buf[0];
    memcpy(dst, buf, n);                    /* -> __memcpy_chk(dst, buf, n, 16) */
    write(1, dst, sizeof(dst));             /* keep dst live (write sink off by default) */
}

int main(void) {
    process(0);
    return 0;
}
