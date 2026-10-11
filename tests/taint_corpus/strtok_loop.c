/* strtok continuation: only the FIRST call names the buffer; every later token
 * comes from `strtok(NULL, ...)`, which reads libc's static state rather than an
 * argument. The second token must still carry the input's taint into system(). */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void run_second_word(int fd) {
    char line[128];
    read(fd, line, sizeof(line) - 1);   /* line tainted */
    line[127] = 0;
    char *verb = strtok(line, " ");     /* first token: names the buffer */
    char *arg = strtok(NULL, " ");      /* continuation: no argument carries taint */
    if (verb && arg)
        system(arg);                     /* SINK: second token */
}

int main(void) {
    run_second_word(0);
    return 0;
}
