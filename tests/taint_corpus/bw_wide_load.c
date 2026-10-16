/* A length read back as the upper half of a wider field (#1041 item 2): one
 * 8-byte store of input bytes, then two 4-byte sign-extending loads of its
 * upper half. The lengths come from `in`, never from the caller's malloc of
 * the message object. Analyze only. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct msg { uint32_t kind; uint32_t pad; int64_t lens; char body[64]; };

__attribute__((noinline)) void deser(struct msg *m, const unsigned char *in, char *d1, char *d2) {
    int64_t raw;
    memcpy(&raw, in, 8);
    m->lens = raw;
    int32_t a = ((volatile int32_t *)&m->lens)[1];
    int32_t b = ((volatile int32_t *)&m->lens)[1];
    memcpy(d1, m->body, a);
    memcpy(d2, m->body, b);
}

/* A field written on one path only: the other path reads what the caller left. */
__attribute__((noinline)) void half_set(struct msg *m, const unsigned char *in, char *d, int c) {
    if (c)
        m->kind = in[0];
    memcpy(d, m->body, m->kind);
}

/* A store through a second pointer may overwrite the field before it is read. */
__attribute__((noinline)) void two_ptrs(struct msg *m, struct msg *n, char *d) {
    m->kind = 4;
    n->kind = (uint32_t)(uintptr_t)d;
    memcpy(d, m->body, ((volatile struct msg *)m)->kind);
}

int main(int argc, char **argv) {
    struct msg *m = malloc(sizeof *m);
    char d1[64], d2[64];
    deser(m, (const unsigned char *)argv[1], d1, d2);
    half_set(m, (const unsigned char *)argv[1], d1, argc);
    two_ptrs(m, m, d1);
    return d1[0] + d2[0];
}
