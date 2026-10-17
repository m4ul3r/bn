#include <stdint.h>
#include <stdio.h>

/* Invented reviewer probe: struct-member stash reached through a pointer PHI. */
struct pair { char *a; char *b; };
struct pair g_pair;

/* On the c == 0 path x is g_pair.a (other), so buf is never written. */
__attribute__((noinline)) void member_phi(char *other, int c) {
    char buf[64];
    struct pair *pp = &g_pair;
    pp->a = other;
    pp->b = buf;
    char *x = c ? buf : pp->a;
    snprintf(x, 64, "%s", "id");
    popen(buf, "r");
}

/* Mirror: member a holds buf, the PHI arm reloads member b. */
__attribute__((noinline)) void member_phi_rev(char *other, int c) {
    char buf[64];
    struct pair *pp = &g_pair;
    pp->b = other;
    pp->a = buf;
    char *x = c ? buf : pp->b;
    snprintf(x, 64, "%s", "id");
    popen(buf, "r");
}

/* Control: both PHI arms really are buf. */
__attribute__((noinline)) void member_phi_ctl(char *other, int c) {
    char buf[64];
    struct pair *pp = &g_pair;
    pp->a = other;
    pp->b = buf;
    char *x = c ? buf : pp->b;
    snprintf(x, 64, "%s", "id");
    popen(buf, "r");
}

/* Truncated pointer terminator: not a write into buf. */
__attribute__((noinline)) void low_part_terminator(void) {
    char buf[64];
    char *p = buf;
    *(char *)(uintptr_t)(uint32_t)(uintptr_t)p = 0;
    popen(buf, "r");
}

int main(int argc, char **argv) {
    member_phi(argv[0], argc);
    member_phi_rev(argv[0], argc);
    member_phi_ctl(argv[0], argc);
    if (argc > 5) low_part_terminator();
    return 0;
}
