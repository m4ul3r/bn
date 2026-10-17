/* #1050 review finding 2: the tainted strtok is followed by a register-indirect
 * jump BN cannot resolve (no enumerated edges, undetermined successors). It
 * may resume at the continuation, so the continuation stays armed. x86-64. */
#include <stdlib.h>
#include <string.h>

__attribute__((noinline)) int tok_jump(char *p, void *target, int first) {
    if (first)
        goto resume;
    strtok(p, " ");
    __asm__ volatile("jmp *%0" :: "r"(target));
    __builtin_unreachable();
resume:
    system(strtok(NULL, " "));
    return 1;
}

int main(int argc, char **argv) { return tok_jump(argv[0], argv[1], argc); }
