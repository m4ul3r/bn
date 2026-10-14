// Synthetic C++ fixture for #998: a virtual method reached only through its
// vtable slot (`callsites` must say so instead of "no callsites"), classes
// constructed on the heap through an out-of-line constructor, a derived class
// sharing the base's slot for a method it does not override, a class whose
// vtable runs past the class lens's 64-slot window, and a multiple-inheritance
// class whose second base's slots live in a secondary vtable.
#include <cstdio>

namespace vc {

struct Dispatcher {
    int id = 0;
    long data[6] = {};
    Dispatcher();
    virtual ~Dispatcher() {}
    virtual int dispatch(int msg, int a, int b);
    virtual int reg(int msg, void *h);
};

struct AudioDisp : Dispatcher {
    char buf[64] = {};
    AudioDisp();
    int dispatch(int msg, int a, int b) override;
};

Dispatcher::Dispatcher() { id = 7; }
AudioDisp::AudioDisp() { buf[0] = 1; }

int Dispatcher::dispatch(int msg, int a, int b) { return msg + a + b + id; }
int Dispatcher::reg(int msg, void *h) { data[msg % 6] = reinterpret_cast<long>(h); return msg; }
int AudioDisp::dispatch(int msg, int a, int b) { return buf[msg & 63] + a - b; }

int run_table(int x) { return x * 3; }

Dispatcher *make(int kind) {
    if (kind > 1) {
        return new AudioDisp();
    }
    return new Dispatcher();
}

void register_all(Dispatcher *d) {
    for (int i = 0; i < 4; i++) {
        d->reg(i, reinterpret_cast<void *>(&run_table));
    }
}

int pump(Dispatcher *d, int n) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        s += d->dispatch(i, n, s);
    }
    return s;
}

// 70 virtuals: slots 0..69, so `m68` sits past the 64-slot layout window.
#define VC_V(n) virtual int m##n() { return n; }
#define VC_V10(t) VC_V(t##0) VC_V(t##1) VC_V(t##2) VC_V(t##3) VC_V(t##4) \
                  VC_V(t##5) VC_V(t##6) VC_V(t##7) VC_V(t##8) VC_V(t##9)
struct Big {
    VC_V(0) VC_V(1) VC_V(2) VC_V(3) VC_V(4) VC_V(5) VC_V(6) VC_V(7) VC_V(8) VC_V(9)
    VC_V10(1) VC_V10(2) VC_V10(3) VC_V10(4) VC_V10(5) VC_V10(6)
};

struct Left {
    virtual int fa() { return 1; }
};
struct Right {
    virtual int fb() { return 2; }
};
struct Both : Left, Right {
    long tag = 3;
    int fa() override { return 10; }
    int fb() override { return static_cast<int>(tag); }
};

int poke(Big *big, Right *right) { return big->m3() + right->fb(); }

} // namespace vc

int main(int argc, char **) {
    vc::Dispatcher *d = vc::make(argc);
    vc::register_all(d);
    int r = vc::pump(d, argc * 4);
    std::printf("%d\n", r);
    vc::Big big;
    vc::Both both;
    std::printf("%d %d\n", vc::poke(&big, &both), both.fa());
    delete d;
    return 0;
}
