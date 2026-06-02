/* Typed-struct field stores (MLIL_SET_VAR_FIELD / _ALIASED_FIELD): a local
 * descriptor struct is populated from attacker input and passed BY ADDRESS to a
 * helper that reads its fields. These field-set ops expose no vars_written, so
 * the engine must taint the struct via the op's .dest. */
#include <unistd.h>
#include <string.h>

struct desc { char *data; int len; };

static void emit(struct desc *d) {
    char out[16];
    memcpy(out, d->data, d->len);   /* over-read: attacker-controlled len */
}

void handle(int fd) {
    char buf[64];
    struct desc d;
    read(fd, buf, sizeof(buf));     /* buf tainted */
    d.data = buf;
    d.len  = buf[0];                /* tainted fields of a local struct */
    emit(&d);                        /* descriptor passed by address */
}

int main(void) {
    handle(0);
    return 0;
}
