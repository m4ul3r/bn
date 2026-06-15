/* Synthetic, license-clean stress fixture for bn. No target data.
 * A small switch-driven state machine (branchy, decompiles cleanly). */
#include <stdio.h>
#include <string.h>

enum state { S_IDLE, S_RUN, S_PAUSE, S_STOP };

__attribute__((noinline, used)) int step(int st, char ev) {
    switch (st) {
        case S_IDLE:  return ev == 'g' ? S_RUN : S_IDLE;
        case S_RUN:   return ev == 'p' ? S_PAUSE : (ev == 'x' ? S_STOP : S_RUN);
        case S_PAUSE: return ev == 'g' ? S_RUN : (ev == 'x' ? S_STOP : S_PAUSE);
        default:      return S_STOP;
    }
}

int main(int argc, char **argv) {
    const char *evs = argc > 1 ? argv[1] : "gpgx";
    int st = S_IDLE;
    for (size_t i = 0; i < strlen(evs); i++)
        st = step(st, evs[i]);
    printf("final state: %d\n", st);
    return 0;
}
