/* Forward taint into a two-field global record (V6-01).
 *
 * `stamp` writes one tainted byte into `rec.tag` at an index nothing bounds,
 * so the record is tainted at an unknown offset. `object_wide` then copies
 * `rec.note` -- a field no tainted byte was written to at a known offset --
 * and `note_of` hands the same field back as a pointer: both copies are
 * reached only through the object-wide taint. `field_precise` copies the
 * tainted string into `rec.note` itself and back out.
 */
#include <stdlib.h>
#include <string.h>

#define N __attribute__((noinline))

struct record {
    char tag[32];
    char note[32];
};

struct record rec = {"tag", "note"};
char out[64];

N void stamp(const char *value, int i) {
    rec.tag[i] = value[0];
}

N char *note_of(const char *value, int i) {
    stamp(value, i);
    return rec.note;
}

N void object_wide(const char *value, int i) {
    stamp(value, i);
    strcpy(out, rec.note);
}

N void object_wide_returned(const char *value, int i) {
    strcpy(out, note_of(value, i));
}

N void field_precise(const char *value) {
    strcpy(rec.note, value);
    strcpy(out, rec.note);
}

int main(int argc, char **argv) {
    if (argc > 2) {
        object_wide(argv[1], atoi(argv[2]));
        object_wide_returned(argv[1], atoi(argv[2]));
        field_precise(argv[1]);
    }
    return 0;
}
