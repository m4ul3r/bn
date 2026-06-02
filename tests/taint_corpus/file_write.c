/* Tainted input is written straight out to a file via fwrite. This is only a
 * finding when the opt-in file_write sink class is enabled (--sink-class
 * file_write); by default it must stay silent. */
#include <unistd.h>
#include <stdio.h>

void persist(int fd) {
    char buf[64];
    read(fd, buf, sizeof(buf));            /* buf tainted */
    FILE *f = fopen("/tmp/persist.bin", "wb");
    if (f) {
        fwrite(buf, 1, sizeof(buf), f);    /* tainted data -> fwrite (file_write) */
        fclose(f);
    }
}

int main(void) {
    persist(0);
    return 0;
}
