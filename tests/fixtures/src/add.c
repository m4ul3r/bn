/* Synthetic, license-clean stress fixture for bn. No target data. */
#include <stdio.h>
#include <stdlib.h>

__attribute__((noinline, used)) int add(int a, int b) {
    return a + b;
}

__attribute__((noinline, used)) int mul(int a, int b) {
    int acc = 0;
    for (int i = 0; i < b; i++)
        acc = add(acc, a);
    return acc;
}

int main(int argc, char **argv) {
    int a = argc > 1 ? atoi(argv[1]) : 2;
    int b = argc > 2 ? atoi(argv[2]) : 3;
    printf("add=%d mul=%d\n", add(a, b), mul(a, b));
    return 0;
}
