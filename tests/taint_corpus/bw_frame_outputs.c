/* A modeled call proves it did NOT write a field only from a declared `write_frame`
 * whose pointers all appear in the call (#1053 round 2). recv's frame bounds its write
 * to [buf, buf+len); recvmsg, readv, getline, stat, sscanf, asprintf and strtol declare
 * no frame, and strncat's frame is open-ended from dst (it appends at dst + strlen).
 * Analyze only. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>

struct pkt { char hdr[8]; int len; };
union upkt { char raw[12]; struct { char hdr[8]; int len; } s; };

/* variable length: recv may reach len -> must stay incomplete */
__attribute__((noinline)) void fill_varlen(int fd, struct pkt *p, size_t n, char *d, const char *s) {
    p->len = 4;
    recv(fd, p->hdr, n, 0);
    memcpy(d, s, ((volatile struct pkt *)p)->len);
}

/* union, recv covers the overlaid len: origin recv is right */
__attribute__((noinline)) void fill_union_cover(int fd, union upkt *u, char *d, const char *s) {
    u->s.len = 4;
    recv(fd, u->raw, 12, 0);
    memcpy(d, s, ((volatile union upkt *)u)->s.len);
}

/* union, recv writes only [0,8): len is the earlier constant 4 */
__attribute__((noinline)) void fill_union_short(int fd, union upkt *u, char *d, const char *s) {
    u->s.len = 4;
    recv(fd, u->raw, 8, 0);
    memcpy(d, s, ((volatile union upkt *)u)->s.len);
}

/* overlapping: recv [0,10) overlaps len [8,12): recv AND n are origins */
__attribute__((noinline)) void fill_overlap(int fd, struct pkt *p, int n, char *d, const char *s) {
    p->len = n;
    recv(fd, p->hdr, 10, 0);
    memcpy(d, s, ((volatile struct pkt *)p)->len);
}

/* overlapping fields through a second view of the same object */
__attribute__((noinline)) void fill_overlap_tail(int fd, struct pkt *p, int n, char *d, const char *s) {
    p->len = n;
    recv(fd, (char *)p + 10, 4, 0);   /* [10,14) overlaps len's upper half */
    memcpy(d, s, ((volatile struct pkt *)p)->len);
}

/* ---- omitted modeled out-pointers (model declares fewer writes) ---- */
struct hdr { int len; int kind; };

/* recvmsg model: sources *iovec:1 (no *arg:N) */
__attribute__((noinline)) void om_recvmsg(int fd, struct hdr *h, char *d, const char *s) {
    struct iovec iov;
    struct msghdr msg;
    memset(&msg, 0, sizeof msg);
    iov.iov_base = h;
    iov.iov_len = sizeof *h;
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    h->len = 0;
    recvmsg(fd, &msg, 0);
    memcpy(d, s, ((volatile struct hdr *)h)->len);
}

/* readv model: sources ret only */
__attribute__((noinline)) void om_readv(int fd, struct hdr *h, char *d, const char *s) {
    struct iovec iov;
    iov.iov_base = h;
    iov.iov_len = sizeof *h;
    h->len = 0;
    readv(fd, &iov, 1);
    memcpy(d, s, ((volatile struct hdr *)h)->len);
}

/* getline model: sources *arg:0 and ret, but it also writes *n (arg1) */
struct gl { size_t n; char *line; };
__attribute__((noinline)) void om_getline(struct gl *g, FILE *fp, char *d, const char *s) {
    g->n = 0;
    g->line = NULL;
    getline(&g->line, &g->n, fp);
    memcpy(d, s, ((volatile struct gl *)g)->n);
}

/* stat model: sink-only (path), but it fills *arg:1 */
__attribute__((noinline)) void om_stat(const char *path, struct stat *st, char *d, const char *s) {
    st->st_size = 0;
    stat(path, st);
    memcpy(d, s, ((volatile struct stat *)st)->st_size);
}

/* sscanf model: arity_capped at *arg:5; the 5th conversion is arg6 */
struct pkt g_pkt;
__attribute__((noinline)) void om_sscanf(const char *line, char *d, const char *s) {
    int a, b, c, e;
    g_pkt.len = 4;
    sscanf(line, "%d %d %d %d %d", &a, &b, &c, &e, &g_pkt.len);
    memcpy(d, s, *(volatile int *)&g_pkt.len);
}

/* strncat model: length n for *arg:0, but strncat appends at strlen(dst) */
struct cat { char name[32]; };
__attribute__((noinline)) void om_strncat(struct cat *c, const char *src, char *d, const char *s) {
    *(int *)(c->name + 20) = 7;
    strncat(c->name, src, 4);
    memcpy(d, s, *(volatile int *)(c->name + 20));
}

/* asprintf model: sink-only, but it writes *arg:0 */
struct ap { char *s; };
__attribute__((noinline)) void om_asprintf(struct ap *a, const char *host) {
    a->s = "true";
    asprintf(&a->s, "%s", host);
    popen(((volatile struct ap *)a)->s, "r");
}

/* strtol model: propagates *arg:0 -> ret only, but it writes *endptr */
struct ep { long v; char *end; };
__attribute__((noinline)) void om_strtol(struct ep *e, const char *num) {
    e->end = "true";
    e->v = strtol(num, &e->end, 10);
    popen(((volatile struct ep *)e)->end, "r");
}

int main(void) { return 0; }
