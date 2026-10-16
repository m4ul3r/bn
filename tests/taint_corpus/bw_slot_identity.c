/* Slot identity is exact (#1053 review 1): `truncated_slot` stores &buf through the
 * low 32 bits of `slot` and reloads the full pointer, so strcpy need not write buf;
 * `full_slot` is the control. Built at -O2. Analyze only. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

__attribute__((noinline)) void truncated_slot(char **slot, const char *host) {
    char buf[64];
    *(char **)(uintptr_t)(uint32_t)(uintptr_t)slot = buf;
    strcpy(*slot, host);
    popen(buf, "r");
}

__attribute__((noinline)) void full_slot(char **slot, const char *host) {
    char buf[64];
    *slot = buf;
    strcpy(*slot, host);
    popen(buf, "r");
}

int main(void) { return 0; }
