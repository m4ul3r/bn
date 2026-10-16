// A static member function: Itanium mangling does not mark it static, so BN
// gives `reg::Registry::get` a leading object parameter its callers never set
// (#1013). `set_handler` is a real member called on the returned object.
#include <cstdio>

namespace reg {
class Registry {
public:
    static Registry *get();
    void set_handler(void (*h)(const char *, int));
    void (*h_)(const char *, int);
};
}  // namespace reg

static reg::Registry g_registry;

__attribute__((noinline)) reg::Registry *reg::Registry::get() { return &g_registry; }
__attribute__((noinline)) void reg::Registry::set_handler(void (*h)(const char *, int)) { h_ = h; }

__attribute__((noinline)) void on_packet(const char *b, int n) { std::printf("%.*s", n, b); }
__attribute__((noinline)) void start_listener() { reg::Registry::get()->set_handler(on_packet); }
__attribute__((noinline)) void stop_listener() { reg::Registry::get()->set_handler(nullptr); }

int main() {
    start_listener();
    stop_listener();
    return 0;
}
