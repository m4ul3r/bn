#include <stdlib.h>
struct pair { long a; long b; };
__attribute__((noinline)) void external_wrong(struct pair *p) { p->b=1; free(malloc(p->a)); }
__attribute__((noinline)) void external_control(struct pair *p) { p->b=1; free(malloc(p->b)); }
__attribute__((noinline)) void external_source(struct pair *p,long source) { p->a=source; p->b=1; free(malloc(p->a)); }
int main(void) { return 0; }
