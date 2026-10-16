/* bw_table_loader at -O2. The toolchain may emit a writable .rodata in an RW
 * segment with no DT_TEXTREL here; the -O0 build carries the non-writable case. */
#include "bw_table_loader.c"
