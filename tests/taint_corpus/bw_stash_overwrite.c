/* A modeled copy that overwrites the stash slot forgets it (#1053 review 3): after
 * memcpy(&stash, &otherptr, 8) the strcpy writes `other`, not buf. Analyze only. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char other[64];
char *stash;
char *const otherptr = other;

__attribute__((noinline)) void overwritten_stash(const char *host) {
    char buf[64];
    memset(buf, 0, sizeof buf);
    stash = buf;
    memcpy(&stash, &otherptr, sizeof stash);
    strcpy(stash, host);
    popen(buf, "r");
}

int main(void) {
    overwritten_stash(getenv("HOST"));
    return 0;
}
