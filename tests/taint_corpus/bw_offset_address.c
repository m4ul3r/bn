/* An address built on a buffer at a VARIABLE offset is still that buffer.
 * `popen(line + off)` lifts to `rdi = &line + off`; a walk that followed only
 * the offset's reads named the parameter `off` as the whole origin, dropped
 * the `read` that filled `line`, and claimed a complete slice. The same holds
 * on the input side of a fill (`strcpy(buf, &other[i])`) and for an element
 * of a writable global array (`g_hosts[i]`, a `.bss` address plus an index),
 * which is bytes any function may have set -- a frontier leaf, not an index. */
#include <stdio.h>
#include <string.h>
#include <unistd.h>

char g_hosts[4][64];

void sink_offset(int fd, int off) {
    char line[128];
    read(fd, line, 127);
    line[127] = 0;
    popen(line + off, "r");               /* SINK: the bytes come from `read` */
}

void input_offset(int fd, int i) {
    char other[64];
    char buf[128];
    read(fd, other, 63);
    other[63] = 0;
    strcpy(buf, &other[i]);               /* the copy's input is `other`, not `i` */
    popen(buf, "r");                      /* SINK */
}

void global_element(int i) {
    char buf[128];
    sprintf(buf, "ping %s", g_hosts[i]);  /* an element of a .bss array */
    popen(buf, "r");                      /* SINK: unaccounted global bytes */
}

int main(int argc, char **argv) {
    (void)argv;
    sink_offset(0, argc);
    input_offset(0, argc);
    global_element(argc & 3);
    return 0;
}
