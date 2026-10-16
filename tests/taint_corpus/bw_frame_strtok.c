/* strtok writes the delimiter it finds, a memory write its model never names: a
 * possible writer of the length field (#1053 round 2). Built at -O2. Analyze only. */
#include <string.h>
#include <stdint.h>
struct packet { char hdr[8]; uint32_t len; };
__attribute__((noinline)) void tokenized_length(struct packet *p,char *d,const char *s) {
    memcpy(p->hdr,"abcdefgh",8);
    p->len=44; /* little endian: ',' then NUL; entire object is a terminated string */
    strtok((char *)p,","); /* this replaces the delimiter at byte 8 by NUL */
    memcpy(d,s,((volatile struct packet *)p)->len);
}
int main(void) {return 0;}
