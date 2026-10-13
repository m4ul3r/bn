/* Synthetic, license-clean stack-frame fixture for bn. No target data.
 * Built with -fstack-protector-all (see the Makefile) so `local list --frame`
 * has a real canary slot to find above a fixed-size stack buffer. */
#include <stdio.h>
#include <string.h>

__attribute__((noinline, used)) int copy_label(const char *src) {
    char label[44];
    memset(label, 0, sizeof(label));
    strcpy(label, src);
    return puts(label);
}

int main(int argc, char **argv) {
    return copy_label(argc > 1 ? argv[1] : "label");
}
