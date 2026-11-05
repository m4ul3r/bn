/* Inline string expansions at -O2 (V8-03..V8-06): gcc expands a constant-size
 * memset/memcpy of a 0x140-byte record into head/tail stores plus a
 * `rep stos`/`rep movs` whose count is the constant total corrected by the
 * destination's alignment, and a byte-length copy into the same shape with the
 * length added in. BN lifts each `rep` to an intrinsic that MLIL renders as two
 * `__builtin_memset`/`__builtin_memcpy` calls at one address. */
#include <string.h>
#include <unistd.h>

struct record { long head; char body[0x130]; long tail; };

__attribute__((noinline)) struct record *record_reset(struct record *r)
{
    memset(r, 0, sizeof *r);
    return r;
}

__attribute__((noinline)) void record_copy(struct record *dst, const struct record *src)
{
    memcpy(dst, src, sizeof *dst);
}

/* The length byte comes from the packet: a tainted total. */
__attribute__((noinline)) int copy_label(const unsigned char *pkt, char *out)
{
    unsigned n = pkt[0];
    memcpy(out, pkt + 1, n);
    return (int)n;
}

int main(void)
{
    struct record a, b;
    char name[64];
    unsigned char pkt[256];
    if (read(0, pkt, sizeof pkt) <= 0)
        return 1;
    record_reset(&a);
    record_copy(&b, &a);
    copy_label(pkt, name);
    return name[0] + b.body[3];
}
