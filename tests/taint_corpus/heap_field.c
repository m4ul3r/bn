/* Backward slice through a heap struct field (#158).
 *
 * A parsed length is stored into a heap-allocated struct field and later used as
 * a copy size. The backward slice from the memcpy length must NOT dead-end at
 * the allocation (malloc), which reads as "locally allocated / clean":
 *
 *  - handle_store: the field is filled by an in-function store, so the slice
 *    recovers the reaching store and continues to the parse of the input.
 *  - handle_extern: the field is filled by a helper (out of this function's
 *    memory scope), so the slice surfaces a `field_load_unresolved` leaf. */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct hdr { unsigned int len; char body[64]; };

static unsigned int parse_len(const char *p) {
    return (unsigned)(unsigned char)p[0] | ((unsigned)(unsigned char)p[1] << 8);
}

void handle_store(int fd) {
    char in[64];
    read(fd, in, sizeof(in));            /* in tainted */
    struct hdr *h = malloc(sizeof(*h));
    h->len = parse_len(in);              /* tainted value stored to heap field */
    char dst[16];
    memcpy(dst, h->body, h->len);        /* SINK: length is a heap field */
}

void fill_len(struct hdr *h, int fd) {
    read(fd, &h->len, sizeof(h->len));
}

void handle_extern(int fd) {
    struct hdr *h = malloc(sizeof(struct hdr));
    fill_len(h, fd);                     /* field filled out of memory scope */
    char dst[16];
    memcpy(dst, h->body, h->len);        /* SINK: heap field, store not in scope */
}

int main(void) {
    handle_store(0);
    handle_extern(0);
    return 0;
}
