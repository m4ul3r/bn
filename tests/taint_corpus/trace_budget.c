/* `trace` step budget and quick-view disclosure (BN6-C010, BN6-C011): the
 * memcpy source argument of publish_banner walks back four steps to its only
 * parameter at -O0 -fno-builtin, so a budget of exactly four drains the walk. */
#include <stdlib.h>
#include <string.h>

char g_banner[64];

__attribute__((noinline)) int render_banner(void) { return system(g_banner); }

__attribute__((noinline)) int publish_banner(const char *text) {
    memcpy(g_banner, text, 8);
    return render_banner();
}

int main(int argc, char **argv) {
    if (argc < 2)
        return 0;
    return publish_banner(argv[1]);
}
