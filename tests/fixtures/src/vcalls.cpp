// Synthetic C++ fixture for #998: a virtual method reached only through its
// vtable slot (`callsites` must say so instead of "no callsites"), classes
// constructed on the heap through an out-of-line constructor (`class show`
// recovers the size from `operator new(N)`), and a derived class sharing the
// base's slot for a method it does not override.
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

} // namespace vc

int main(int argc, char **) {
    vc::Dispatcher *d = vc::make(argc);
    vc::register_all(d);
    int r = vc::pump(d, argc * 4);
    std::printf("%d\n", r);
    delete d;
    return 0;
}
