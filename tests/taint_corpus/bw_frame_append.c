/* strncat appends at dst + strlen(dst): its count is not [dst, dst+n) geometry (#1053
 * round 2). The strncpy control writes exactly [0,3). Analyze only. */
#include <stdint.h>
#include <string.h>
struct cat {
    union {
        char name[32];
        struct { char pad[20]; uint32_t len; } fields;
    } u;
};
__attribute__((noinline)) void append_length(struct cat *c, char *d, const char *s) {
    memset(c->u.name, 'A', 20);
    c->u.fields.len = 7;
    strncat(c->u.name, "ABC", 3);
    memcpy(d, s, ((volatile struct cat *)c)->u.fields.len);
}
__attribute__((noinline)) void copy_length(struct cat *c, char *d, const char *s) {
    memset(c->u.name, 'A', 20);
    c->u.fields.len = 7;
    strncpy(c->u.name, "ABC", 3);
    memcpy(d, s, ((volatile struct cat *)c)->u.fields.len);
}
int main(void) { return 0; }
