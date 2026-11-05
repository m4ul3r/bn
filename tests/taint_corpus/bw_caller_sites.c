/* Backward caller ascent accounting. `quote_arg` copies its argument with
 * strcpy and has ten call sites: `edit_entry` hands one `pick_name()` result to
 * it directly AND through `open_alt`, so two sites reach one origin; eight more
 * pass their own parameter. Every site is either followed or named.
 *
 * `put_word` has twenty call sites, past the caller cap: the followed sixteen
 * are the lowest addresses, and the rest are named in address order. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char g_quoted[512];

__attribute__((noinline)) char *quote_arg(const char *s)
{
    strcpy(g_quoted, s);
    return g_quoted;
}

__attribute__((noinline)) char *pick_name(int alt)
{
    char *p = getenv(alt ? "ALT_ENTRY" : "ENTRY");
    return p ? p : "none";
}

__attribute__((noinline)) void open_alt(const char *name)
{
    puts(quote_arg(name));
}

__attribute__((noinline)) void edit_entry(int alt)
{
    char *name = pick_name(alt);
    puts(quote_arg(name));
    open_alt(name);
}

#define SITE(n) \
    __attribute__((noinline)) void show_##n(const char *p) { puts(quote_arg(p)); }
SITE(0) SITE(1) SITE(2) SITE(3) SITE(4) SITE(5) SITE(6) SITE(7)

static char g_words[4096];

__attribute__((noinline)) void put_word(const char *w, size_t n)
{
    memcpy(g_words, w, n);
}

#define WORD(n) \
    __attribute__((noinline)) void word_##n(const char *w) { put_word(w, (n) + 1); }
WORD(0) WORD(1) WORD(2) WORD(3) WORD(4) WORD(5) WORD(6) WORD(7) WORD(8) WORD(9)
WORD(10) WORD(11) WORD(12) WORD(13) WORD(14) WORD(15) WORD(16) WORD(17) WORD(18) WORD(19)

int main(int argc, char **argv)
{
    const char *a = argc > 1 ? argv[1] : "";
    edit_entry(argc > 2);
    show_0(a); show_1(a); show_2(a); show_3(a); show_4(a); show_5(a); show_6(a); show_7(a);
    word_0(a); word_1(a); word_2(a); word_3(a); word_4(a); word_5(a); word_6(a); word_7(a);
    word_8(a); word_9(a); word_10(a); word_11(a); word_12(a); word_13(a); word_14(a);
    word_15(a); word_16(a); word_17(a); word_18(a); word_19(a);
    return 0;
}
