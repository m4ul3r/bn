/* recv is never certified as a field's producer (#1053 review 2): recv writes p->hdr,
 * bytes [0,8), yet no call's write extent is trusted, so p->len at [8,12) stays a
 * frontier with or without an earlier store. Built at -O2. Analyze only. */
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
struct pkt { char hdr[8]; int len; };
__attribute__((noinline)) void fill_other_field(int fd, struct pkt *p,
        char *d, const char *s, int n) {
    p->len = n;
    recv(fd, p->hdr, sizeof p->hdr, 0);
    memcpy(d, s, ((volatile struct pkt *)p)->len);
}
__attribute__((noinline)) void fill_other_field_nostore(int fd, struct pkt *p,
        char *d, const char *s) {
    recv(fd, p->hdr, sizeof p->hdr, 0);
    memcpy(d, s, ((volatile struct pkt *)p)->len);
}
int main(int argc, char **argv) {
    struct pkt p;
    char d[64];
    fill_other_field(argc, &p, d, argv[0], argc);
    fill_other_field_nostore(argc, &p, d, argv[0]);
    return d[0];
}
