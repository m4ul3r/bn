/* Positive: attacker-controlled length from read() reaches memcpy. */
#include <unistd.h>
#include <string.h>
#include <stdio.h>

void process(int fd) {
    char buf[64];
    char dst[16];
    long n = read(fd, buf, sizeof(buf));   /* buf is tainted */
    int len = (int)buf[0] + 4;             /* load + arithmetic */
    if (n > 0) {
        memcpy(dst, buf, len);             /* SINK: tainted length */
    }
    printf("%ld\n", n);
}

int main(void) {
    process(0);
    return 0;
}
