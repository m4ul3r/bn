/* Synthetic, license-clean stress fixture for bn. No target data.
 * A toy reversible "cipher" (XOR + rotate); not real cryptography. */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

static const char KEY[] = "synthetic-fixture-key";

__attribute__((noinline, used)) uint8_t rotl8(uint8_t v, int n) {
    return (uint8_t)((v << n) | (v >> (8 - n)));
}

__attribute__((noinline, used)) void encrypt(uint8_t *buf, size_t len) {
    size_t klen = strlen(KEY);
    for (size_t i = 0; i < len; i++)
        buf[i] = rotl8((uint8_t)(buf[i] ^ KEY[i % klen]), 3);
}

int main(int argc, char **argv) {
    uint8_t data[32];
    memset(data, 0x41, sizeof(data));
    encrypt(data, sizeof(data));
    printf("encrypted first byte: %02x\n", data[0]);
    return 0;
}
