/* strncat appends at dst + strlen(dst), so its count is no destination geometry; no
 * call's write extent is trusted, so the strncpy control is a possible writer as well
 * (#1053 rounds 2-3). Analyze only. */
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
