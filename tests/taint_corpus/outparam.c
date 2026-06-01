/* Out-parameter propagation: a helper writes tainted data through a pointer
 * parameter (its output buffer). Taint must flow back into the caller's buffer
 * so the downstream sink (system on the filled buffer) is reported. */
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

static void fill(char *src, char *dst) {
    strcpy(dst, src);   /* writes tainted src through the dst out-parameter */
}

void handler(int fd) {
    char buf[64];
    char out[64];
    read(fd, buf, sizeof(buf));   /* buf tainted */
    fill(buf, out);               /* out becomes tainted via the out-parameter */
    system(out);                  /* SINK reachable only through the out-parameter */
}

int main(void) {
    handler(0);
    return 0;
}
