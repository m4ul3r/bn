/* Synthetic, license-clean stack-frame fixture for bn. No target data.
 * Built with -fstack-protector-all (see the Makefile) so `local list --frame`
 * has a real canary slot to find above a fixed-size stack buffer. */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

__attribute__((noinline, used)) int copy_label(const char *src) {
    char label[44];
    memset(label, 0, sizeof(label));
    strcpy(label, src);
    return puts(label);
}

/* A normal argument-relative load, not a TLS guard, even if its argument is
 * renamed to look like an architectural TLS register in the analysis. */
__attribute__((noinline, used, no_stack_protector))
uintptr_t read_context_word(const uintptr_t *context) {
    volatile uintptr_t word = context[5];
    return word;
}

int main(int argc, char **argv) {
    return copy_label(argc > 1 ? argv[1] : "label");
}
