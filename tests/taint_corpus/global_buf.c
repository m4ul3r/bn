/* read() fills a GLOBAL/static buffer (referenced by absolute address, not a
 * stack &local), a byte is loaded back out of it and used as a memcpy length.
 * Exercises global-buffer taint tracking (seed + load-from-global + length sink).
 * Built -no-pie (default) so the global has an absolute address. */
#include <unistd.h>
#include <string.h>

char g[64];                                 /* global buffer */

void g_handle(int fd) {
    read(fd, g, sizeof(g));                 /* taint the global */
    unsigned char n = (unsigned char)g[0];  /* load back out of the tainted global */
    char dst[16];
    memcpy(dst, g, n);                      /* tainted length -> overflow_len */
    write(1, dst, sizeof(dst));             /* keep dst live */
}

int main(void) {
    g_handle(0);
    return 0;
}
