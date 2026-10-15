/* #932: two entries and two back-edges, unused-return initializers.
 * The O2 lift may enumerate the NULL continuation before either initializer.
 */
#include <stdlib.h>
#include <string.h>

__attribute__((noinline)) void token_cycle(char *line, unsigned mode, int count) {
    char *token;
    if (mode & 1) goto left;
    goto right;
left:
    if (--count <= 0) return;
    token = strtok(NULL, " ");
    if (token) system(token);
    if (mode & 2) goto right;
    (void)strtok(line, " ");
    goto left;
right:
    (void)strtok(line, " ");
    if (mode & 1) goto left;
    if (--count > 0) goto right;
}
int main(void) { return 0; }
