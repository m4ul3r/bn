// Hard case (C++ vtable): a virtual call dispatches through a vtable slot.
// MVP behaviour is honest degradation -- the call surfaces as an indirect
// callee (resolved by value-set only when the dynamic type is pinned).
// Type-informed devirtualization is a Phase-3 item.
#include <unistd.h>

struct Handler {
    virtual void run(char *p) = 0;
    virtual ~Handler() {}
};

struct Real : Handler {
    void run(char *p) override;
};

void Real::run(char *p) { (void)p; }

void dispatch(Handler *h, char *buf) {
    h->run(buf);   // virtual call -> indirect through vtable
}

int main() {
    return 0;
}
