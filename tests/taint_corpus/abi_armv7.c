/* armv7 (Thumb, EABI hard-float) argument-mapping shapes, built with
 * arm-linux-gnueabihf-gcc -O2 -mthumb.
 *
 * `put_wide` takes a `long long` that EABI passes in r2:r3 (r1 is padding);
 * Binary Ninja binds it to a stack slot and the next formal to r1, so the
 * body reads `entry_r2` with no formal bound to it. `exit_usage` is a
 * no-argument callee whose prologue is `push {r3, lr}` -- r3 is stack
 * alignment, not an argument. `flush_window` tail-calls glibc's `memcpy`,
 * whose PLT stub BN types with the IFUNC resolver's one-argument prototype,
 * with a length the Thumb `itee` block computes on two paths: the length
 * register is a phi of two definitions. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__attribute__((noinline)) void put_wide(char *dst, unsigned long long size, int n)
{
    char *p = malloc((size_t)size);
    if (!p)
        return;
    memset(p, 0x41, n);
    memcpy(dst, p, n);
    free(p);
}

__attribute__((noinline)) int pass_wide(char *dst, unsigned int len)
{
    put_wide(dst, (unsigned long long)len, 2);
    return 0;
}

__attribute__((noinline)) void exit_usage(void)
{
    puts("usage: abi_armv7 N");
    exit(1);
}

__attribute__((noinline)) int check_args(int a, int b, int c, int d)
{
    if (d > 7)
        exit_usage();
    return a + b + c + d;
}

struct window {
    unsigned char *buf;
    unsigned size;
    unsigned have;
    unsigned next;
};

__attribute__((noinline)) void flush_window(struct window *w, unsigned char *out,
                                            unsigned copy)
{
    unsigned n = w->next;
    if (copy > n)
        n = copy - n;
    else
        n = w->have + copy;
    memcpy(out, w->buf, n);
}

int main(int argc, char **argv)
{
    char buf[64];
    struct window win = {(unsigned char *)argv[0], 64, (unsigned)argc, 3};
    flush_window(&win, (unsigned char *)buf, (unsigned)argc);
    unsigned int len = argc > 1 ? (unsigned int)strtoul(argv[1], NULL, 0) : 4;
    pass_wide(buf, len);
    return check_args(argc, argc, argc, (int)len);
}
