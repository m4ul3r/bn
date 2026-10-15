/* Analyze only: synthetic unsafe consumers used to check slice honesty. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

char old_target[64];
char *review_stash = old_target;

__attribute__((noinline)) void future_stash(void) {
    char buf[64];
    char *host = getenv("REVIEW_HOST");
    memset(buf, 0, sizeof buf);
    strcpy(review_stash, host); /* writes old_target, never buf */
    review_stash = buf;
    popen(buf, "r");
}

__attribute__((noinline)) void wide_full(const wchar_t *input) {
    wchar_t buf[16];
    wmemcpy(buf, input, 16);
    popen((char *)buf, "r");
}

__attribute__((noinline)) void wide_short(const wchar_t *input) {
    wchar_t buf[16];
    wmemcpy(buf, input, 8);
    popen((char *)buf, "r");
}

__attribute__((noinline)) void clean_helper(void) { puts("tick"); }
__attribute__((noinline)) void strcpy_literal(void) {
    char buf[64]; strcpy(buf, "id"); popen(buf, "r");
}
__attribute__((noinline)) void loop_no_escape(int n) {
    char buf[64]; strcpy(buf, "id");
    for (volatile int k = 0; k < n; k++) clean_helper();
    popen(buf, "r");
}

/* Not covered by the sink-metadata guard; smoke output documents the limit. */
__attribute__((noinline)) void fgets_one(void) {
    char buf[64]; fgets(buf, 1, stdin); popen(buf, "r");
}
__attribute__((noinline)) void fread_zero(void) {
    char buf[64]; fread(buf, 1, 0, stdin); popen(buf, "r");
}
/* 300 earlier definitions force a discovery cut before the escape solve. */
#define NOISE4 (clean_helper(), clean_helper(), clean_helper(), clean_helper())
#define NOISE16 (NOISE4, NOISE4, NOISE4, NOISE4)
#define NOISE64 (NOISE16, NOISE16, NOISE16, NOISE16)
__attribute__((noinline)) void budget_known_callee(void) {
    char buf[64];
    NOISE64; NOISE64; NOISE64; NOISE64; NOISE16; NOISE16; NOISE4; NOISE4; NOISE4;
    memset(buf, 0, sizeof buf);
    review_stash = buf;
    clean_helper();
    popen(buf, "r");
}
#undef NOISE64
#undef NOISE16
#undef NOISE4
int main(void) { return 0; }
