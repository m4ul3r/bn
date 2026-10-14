// Synthetic C++ fixture for #998 item 1, built at -O2: data that LOOKS like a
// vtable slot. `callsites`/`xrefs` may leave these undecided; they must never
// confirm them as slots, and a long function-pointer table must not make
// either command slow.
//
// GCC lays `.data.rel.ro` out with the key-function vtable first and the
// file-scope objects after it in reverse definition order, so the layout is:
// `vtable for Short` (70 slots), `adjacent_table`, `regs`, `long_table`.
#include <typeinfo>

#define NOINLINE __attribute__((noinline))
#define SL_V(n) virtual int m##n();
#define SL_V10(t) SL_V(t##0) SL_V(t##1) SL_V(t##2) SL_V(t##3) SL_V(t##4) \
                  SL_V(t##5) SL_V(t##6) SL_V(t##7) SL_V(t##8) SL_V(t##9)
#define SL_D(n) int Short::m##n() { return n; }
#define SL_D10(t) SL_D(t##0) SL_D(t##1) SL_D(t##2) SL_D(t##3) SL_D(t##4) \
                  SL_D(t##5) SL_D(t##6) SL_D(t##7) SL_D(t##8) SL_D(t##9)

struct Short {
    SL_V(0) SL_V(1) SL_V(2) SL_V(3) SL_V(4) SL_V(5) SL_V(6) SL_V(7) SL_V(8) SL_V(9)
    SL_V10(1) SL_V10(2) SL_V10(3) SL_V10(4) SL_V10(5) SL_V10(6)
};
SL_D(0) SL_D(1) SL_D(2) SL_D(3) SL_D(4) SL_D(5) SL_D(6) SL_D(7) SL_D(8) SL_D(9)
SL_D10(1) SL_D10(2) SL_D10(3) SL_D10(4) SL_D10(5) SL_D10(6)

struct X {
    virtual ~X() {}
    virtual int v() { return 1; }
};

NOINLINE int fx(int x) { return x * 5; }
NOINLINE int adjacent(int x) { return x + 7; }
NOINLINE int hdef(int x) { return x - 1; }

// A long function-pointer table: 5000 entries of `hdef`.
#define SL_H1 hdef,
#define SL_H10 SL_H1 SL_H1 SL_H1 SL_H1 SL_H1 SL_H1 SL_H1 SL_H1 SL_H1 SL_H1
#define SL_H100 SL_H10 SL_H10 SL_H10 SL_H10 SL_H10 SL_H10 SL_H10 SL_H10 SL_H10 SL_H10
#define SL_H1000 SL_H100 SL_H100 SL_H100 SL_H100 SL_H100 SL_H100 SL_H100 SL_H100 SL_H100 SL_H100
extern int (*const long_table[])(int);
int (*const long_table[])(int) = {SL_H1000 SL_H1000 SL_H1000 SL_H1000 SL_H1000};

// A registry entry shaped like a vtable header: {offset-to-top 0, &typeid(X), fn}.
struct Reg {
    long zero;
    const std::type_info *ti;
    int (*fn)(int);
};
extern const Reg regs[];
const Reg regs[] = {{0, &typeid(X), fx}};

// One function pointer right after Short's 70 slots.
extern int (*const adjacent_table[])(int);
int (*const adjacent_table[])(int) = {adjacent};

int main(int argc, char **) {
    Short s;
    X x;
    int r = adjacent_table[0](argc) + regs[0].fn(argc) + s.m3() + x.v();
    r += long_table[argc % 5000](argc) + static_cast<int>(regs[0].zero);
    return r;
}
