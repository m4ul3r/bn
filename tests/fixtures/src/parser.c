/* Synthetic, license-clean stress fixture for bn. No target data.
 * A tiny length-prefixed record parser (a source->sink shape for taint demos,
 * exercised by the stress harness only for analysis, never executed on input). */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

struct record {
    uint8_t  type;
    uint8_t  len;
    char     data[64];
};

__attribute__((noinline, used)) int parse_record(const uint8_t *in, size_t n, struct record *out) {
    if (n < 2)
        return -1;
    out->type = in[0];
    out->len = in[1];
    size_t copy = out->len;            /* attacker-controlled length */
    if (copy > n - 2)
        copy = n - 2;
    memcpy(out->data, in + 2, copy);   /* bounded copy */
    return (int)copy;
}

int main(int argc, char **argv) {
    uint8_t in[8] = {1, 4, 'a', 'b', 'c', 'd', 0, 0};
    struct record r;
    int k = parse_record(in, sizeof(in), &r);
    printf("parsed %d bytes, type=%u\n", k, r.type);
    return 0;
}
