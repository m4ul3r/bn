// Synthetic C++ fixture for the class lens (#922): a single-inheritance chain
// (Handler -> EchoHandler -> AuditHandler) with NO multiple inheritance, an
// exception subclass of an IMPORTED class (std::runtime_error, whose typeinfo
// is a GOT/import slot in this image), and a template class whose demangled
// spelling differs between its method symbols and its RTTI symbols.
//
// Built as a PIE (see the Makefile): the `.data.rel.ro` layout packs each
// class's vtable directly against the next class's `__si_class_type_info`
// record, which is the shape that produced phantom slots and phantom
// "secondary vtables" before the guards.
#include <cstdio>
#include <stdexcept>
#include <string>

namespace svc {

struct Handler {
    virtual ~Handler() {}
    virtual int handle(int code) = 0;
    virtual const char *name() const { return "handler"; }
};

struct EchoHandler : Handler {
    int handle(int code) override { return code; }
    const char *name() const override { return "echo"; }
};

struct AuditHandler : EchoHandler {
    int handle(int code) override {
        std::printf("audit %d\n", code);
        return EchoHandler::handle(code);
    }
    virtual void flush() { std::fflush(stdout); }
};

template <class T>
struct Box : Handler {
    T value;
    explicit Box(T v) : value(v) {}
    int handle(int code) override { return code + static_cast<int>(value.size()); }
    const char *name() const override { return value.c_str(); }
};

struct ParseError : std::runtime_error {
    explicit ParseError(const char *what) : std::runtime_error(what) {}
};

int dispatch(Handler &h, int code) {
    if (code < 0) {
        throw ParseError("negative code");
    }
    return h.handle(code);
}

} // namespace svc

int main(int argc, char **argv) {
    svc::AuditHandler audit;
    svc::EchoHandler echo;
    svc::Box<std::string> box(std::string(argc > 2 ? argv[2] : "box"));
    svc::Handler *h = argc > 1 ? static_cast<svc::Handler *>(&audit) : &echo;
    if (argc > 2) {
        h = &box;
    }
    try {
        return svc::dispatch(*h, argc - 1);
    } catch (const svc::ParseError &ex) {
        std::puts(ex.what());
        return 2;
    }
}
