/* Synthetic, license-clean stress fixture for bn. No target data. */
#include <stdio.h>
#include <string.h>

__attribute__((noinline)) int greet(const char *who) {
    char buf[64];
    snprintf(buf, sizeof(buf), "hello, %s", who);
    return (int)strlen(buf);
}

int main(int argc, char **argv) {
    const char *who = argc > 1 ? argv[1] : "world";
    printf("%s\n", "starting up");
    int n = greet(who);
    printf("greeting length: %d\n", n);
    return 0;
}
