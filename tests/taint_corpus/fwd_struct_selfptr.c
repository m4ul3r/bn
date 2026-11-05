/* Forward taint through a struct member that points into ANOTHER member of
 * the same, partially tainted struct (V1-01 / V2-01).
 *
 * `main` taints only `data.slots[0].path`; `data.active` points at
 * `data.slots[0]`. `attach_active` reads `s->active->path` -- a load through
 * the self-pointing member -- and opens it: a TRUE flow into fopen. The
 * `touch_log` call opens the clean `log` member: field-only struct taint must
 * keep that one silent.
 *
 * `indexed_entry` points `active` at a slot chosen at run time, so the member
 * the load reaches cannot be resolved: that is a `field_load_unresolved`
 * frontier, never silence.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N __attribute__((noinline))

struct slot {
    void *handle;
    const char *path;
    char *spare;
};

struct session {
    int flags;
    FILE *log;
    struct slot *active;
    long mode;
    struct slot slots[4];
};

N FILE *open_path(const char *path) { return fopen(path, "rb"); }

N void attach_active(struct session *s) {
    const char *path = s->active->path;
    if (path)
        s->log = open_path(path);
}

N void touch_log(struct session *s) {
    FILE *f = fopen((const char *)s->log, "r");
    if (f)
        fclose(f);
}

N int indexed_entry(int k, char **argv) {
    struct session data;
    memset(&data, 0, sizeof data);
    data.active = &data.slots[k & 3];
    data.slots[0].path = argv[1];
    attach_active(&data);
    return data.log != NULL;
}

int main(int argc, char **argv) {
    struct session data;
    memset(&data, 0, sizeof data);
    data.active = &data.slots[0];
    if (argc > 1)
        data.slots[0].path = argv[1];
    attach_active(&data);
    touch_log(&data);
    return indexed_entry(argc, argv);
}
