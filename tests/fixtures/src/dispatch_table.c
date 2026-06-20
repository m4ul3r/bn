/* Synthetic, license-clean fixture for bn. No target/proprietary data.
 *
 * A C-style const function-pointer dispatch table: `table[cmd](buf, n)` with a
 * bounded index into a `static const` array in .rodata. Built non-PIE, this is
 * the shape Binary Ninja's value-set CAN pin (LookupTableValue), unlike a C++
 * vtable (relocation-filled) or a data-indexed table (symbolic index). It is the
 * real-BN regression fixture for value-set indirect-source anchoring (#282):
 * `taint forward --source arg:h_copy:1` must anchor at the indirect call via
 * value-set and propagate the attacker length into h_copy's copy sink.
 * Analyzed only; never executed on input.
 */
#include <string.h>

typedef void (*handler_t)(char *buf, unsigned n);

static char g_out[64];

/* h_copy: a copy sink -- memcpy(dst, buf, n) with an attacker-controlled n. */
__attribute__((noinline, used)) static void h_copy(char *buf, unsigned n) { memcpy(g_out, buf, n); }
__attribute__((noinline, used)) static void h_noop(char *buf, unsigned n) { (void)buf; (void)n; }
__attribute__((noinline, used)) static void h_log (char *buf, unsigned n) { (void)buf; g_out[0] = (char)n; }

/* const function-pointer dispatch table in .rodata */
static const handler_t table[3] = { h_copy, h_noop, h_log };

__attribute__((noinline, used)) void dispatch(unsigned cmd, char *buf, unsigned n) {
    if (cmd < 3)
        table[cmd](buf, n);            /* bounded index into a const table */
}

int main(int argc, char **argv) {
    dispatch((unsigned)argc & 1u, argv[0], (unsigned)argc);
    return 0;
}
