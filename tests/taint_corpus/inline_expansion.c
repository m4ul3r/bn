/* Inline string expansions at -O2 (V8-03..V8-06, G4c): gcc expands a
 * constant-size memset/memcpy of a 0x140-byte record into head/tail stores plus
 * a `rep stos`/`rep movs` whose count is the constant total corrected by the
 * destination's alignment, and a variable-size memset/memcpy into the same
 * shape with the length added in. BN lifts each `rep` to an intrinsic that MLIL
 * renders as two `__builtin_memset`/`__builtin_memcpy` calls at one address. */
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

/* A tainted-size fill: the size byte comes from the packet. */
__attribute__((noinline)) void clear_tail(char *out, const unsigned char *pkt)
{
    memset(out, 0, pkt[0]);
}

/* A masked tainted size: at most 0x40f bytes, but the packet picks how many. */
__attribute__((noinline)) void clear_masked(char *out, const unsigned short *pkt)
{
    memset(out, 0, (pkt[0] & 0x3ff) + 16);
}

int main(void)
{
    struct record a, b;
    char name[64];
    char tail[0x410];
    unsigned char pkt[256];
    if (read(0, pkt, sizeof pkt) <= 0)
        return 1;
    record_reset(&a);
    record_copy(&b, &a);
    copy_label(pkt, name);
    clear_tail(tail, pkt);
    clear_masked(tail, (const unsigned short *)pkt);
    return name[0] + b.body[3] + tail[7];
}
