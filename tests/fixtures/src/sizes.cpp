// Synthetic C++ fixture for #998 item 4, built at -O2: the shapes where an
// `operator new(N)` allocation is NOT one object of the class whose vptr or
// constructor touches it first. The class lens may leave any of these sizes
// unknown; it must never report a wrong one. Each real size is pinned below.
#include <cstdio>
#include <cstring>
#include <new>

#define NOINLINE __attribute__((noinline))

namespace sz {

// Out-of-line ctor that initializes every byte, plus a virtual destructor:
// both the deleting destructor and the constructor's extent prove 0x28.
struct Plain {
    long a[4];
    NOINLINE Plain();
    virtual ~Plain();
    virtual int f();
};
Plain::Plain() { for (int i = 0; i < 4; i++) a[i] = i; }
Plain::~Plain() {}
int Plain::f() { return static_cast<int>(a[1]); }

// Array new of a class with an inlined ctor and no virtual destructor.
struct Inl {
    int v = 1;
    virtual int g() { return v; }
};

// A polymorphic member at offset 0 of a non-polymorphic holder, ctor inlined.
struct Elem {
    int x = 2;
    virtual int h() { return x; }
};
struct Holder {
    Elem e;
    long pad[8] = {};
};

// The same with the member's ctor out of line.
struct Inner {
    int x;
    NOINLINE Inner();
    virtual int k();
};
Inner::Inner() : x(5) {}
int Inner::k() { return x; }
struct Outer {
    Inner in;
    long pad[8];
    Outer() { pad[0] = 6; }
};

// Placement new into a larger raw allocation.
struct Gadget {
    long q;
    NOINLINE Gadget();
    virtual int m();
};
Gadget::Gadget() : q(7) {}
int Gadget::m() { return static_cast<int>(q); }

// A non-polymorphic base with an out-of-line ctor under an inlined derived ctor.
struct PBase {
    long b;
    NOINLINE PBase();
};
PBase::PBase() : b(8) {}
struct PDer : PBase {
    long d[10];
    PDer() { d[0] = 1; }
};

// A `final` class whose inlined destructor re-stamps the base vptr before
// `operator delete`.
struct Base {
    long b = 0;
    virtual ~Base() { std::puts("base"); }
    virtual int r() { return 1; }
};
struct Leaf final : Base {
    long l[6];
    Leaf() { l[0] = 3; }
    ~Leaf() override { l[0] = 0; }
    int r() override { return 2; }
};

// An out-of-line base ctor under an inlined polymorphic derived ctor.
struct PB {
    long p;
    NOINLINE PB();
    virtual ~PB();
    virtual int s();
};
PB::PB() : p(9) {}
PB::~PB() {}
int PB::s() { return static_cast<int>(p); }
struct PD : PB {
    long more[8];
    PD() { more[0] = 4; }
    int s() override { return static_cast<int>(more[0]); }
};

// No vtable, so no deleting destructor; its ctor writes every byte of the
// `operator new(0x18)` it runs on -- which still proves nothing (see below):
// an allocation is only ever an upper bound.
struct Rec {
    long a, b, c;
    NOINLINE Rec();
};
Rec::Rec() : a(1), b(2), c(3) {}

// Trailing storage: the ctor fills bytes PAST its own object, in a larger
// allocation, so its writes reach the allocation's end (#998 review round 2).
struct Trailing {
    long a, b;
    NOINLINE Trailing();
};
Trailing::Trailing() : a(1), b(2) { std::memset(reinterpret_cast<char *>(this) + 16, 0, 64); }
struct Trail {
    long n;
    NOINLINE Trail();
};
Trail::Trail() : n(1) { reinterpret_cast<long *>(this)[1] = 0; }

// A header whose ctor clears its enclosing (standard-layout) object's payload.
struct Header {
    long tag;
    NOINLINE Header();
};
struct Storage {
    Header head;
    char payload[64];
};
Header::Header() : tag(3) { std::memset(reinterpret_cast<Storage *>(this)->payload, 0, 64); }

// A virtual base: a BN type that covers only the non-virtual part (what a
// type library gives `std::istream`) is not the class size.
struct VB0 {
    long x[20];
};
struct VD0 : virtual VB0 {
    long d;
    NOINLINE VD0();
};
VD0::VD0() : d(5) { x[0] = 6; }

static_assert(sizeof(Plain) == 0x28, "Plain");
static_assert(sizeof(Inl) == 0x10, "Inl");
static_assert(sizeof(Elem) == 0x10 && sizeof(Holder) == 0x50, "Holder");
static_assert(sizeof(Inner) == 0x10 && sizeof(Outer) == 0x50, "Outer");
static_assert(sizeof(Gadget) == 0x10, "Gadget");
static_assert(sizeof(PBase) == 0x8 && sizeof(PDer) == 0x58, "PDer");
static_assert(sizeof(Base) == 0x10 && sizeof(Leaf) == 0x40, "Leaf");
static_assert(sizeof(PB) == 0x10 && sizeof(PD) == 0x50, "PD");
static_assert(sizeof(Rec) == 0x18, "Rec");
static_assert(sizeof(Trailing) == 0x10 && sizeof(Trail) == 0x8, "Trailing");
static_assert(sizeof(Header) == 0x8 && sizeof(Storage) == 0x48, "Storage");
static_assert(sizeof(VD0) == 0xb0, "VD0");

NOINLINE Plain *make_plain() { return new Plain(); }
NOINLINE Inl *make_inl() { return new Inl[3]; }
NOINLINE Holder *make_holder() { return new Holder(); }
NOINLINE Rec *make_rec() { return new Rec(); }
NOINLINE Trailing *make_trailing() { return new (::operator new(sizeof(Trailing) + 64)) Trailing(); }
NOINLINE Trail *make_trail() { return new (::operator new(sizeof(Trail) + 8)) Trail(); }
NOINLINE Storage *make_storage() { return new Storage(); }
NOINLINE VD0 *make_vd0() { return new VD0(); }
NOINLINE Outer *make_outer() { return new Outer(); }
NOINLINE Gadget *make_gadget() { return new (::operator new(0x100)) Gadget(); }
NOINLINE PDer *make_pder() { return new PDer(); }
NOINLINE int churn_leaf() {
    Leaf *leaf = new Leaf();
    int r = leaf->r();
    delete leaf;
    return r;
}
NOINLINE PB *make_pd() { return new PD(); }

} // namespace sz

int main() {
    int r = sz::make_plain()->f();
    r += sz::make_inl()[1].g();
    r += sz::make_holder()->e.h();
    r += sz::make_outer()->in.k();
    r += sz::make_gadget()->m();
    r += static_cast<int>(sz::make_pder()->b);
    r += sz::churn_leaf();
    r += sz::make_pd()->s();
    r += static_cast<int>(sz::make_rec()->c);
    r += static_cast<int>(sz::make_trailing()->b + sz::make_trail()->n);
    r += static_cast<int>(sz::make_storage()->head.tag + sz::make_vd0()->d);
    std::printf("%d\n", r);
    return 0;
}
