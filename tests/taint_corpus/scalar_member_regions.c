/* Invented reviewer probe for PR #941 round 5: scalar member LOAD demand. */
#include <stdlib.h>

struct pairl { long a; long b; };

/* Only member a is initialized; the sink consumes member b (uninitialized). */
void scalar_member_wrong(void) {
    struct pairl r;
    struct pairl *p = &r;
    r.a = 1;
    malloc(p->b);
}

/* Control: the consumed member b is initialized. */
void scalar_member_ctl(void) {
    struct pairl r;
    struct pairl *p = &r;
    r.b = 1;
    malloc(p->b);
}

/* Mirror: b initialized, a consumed (uninitialized). */
void scalar_member_mirror(void) {
    struct pairl r;
    struct pairl *p = &r;
    r.b = 1;
    malloc(p->a);
}

/* Through a pointer store rather than a named store. */
void scalar_member_ptr_store(void) {
    struct pairl r;
    struct pairl *p = &r;
    p->a = 1;
    malloc(p->b);
}

int main(void) {
    scalar_member_wrong(); scalar_member_ctl(); scalar_member_mirror();
    scalar_member_ptr_store();
    return 0;
}
