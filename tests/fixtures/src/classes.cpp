// Synthetic C++ fixture for the class lens (#922): a single-inheritance chain
// (Handler -> EchoHandler -> AuditHandler) with NO multiple inheritance, an
// exception subclass of an IMPORTED class (std::runtime_error, whose typeinfo
// is a GOT/import slot in this image), a template class whose demangled
// spelling differs between its method symbols and its RTTI symbols, an
// abstract class whose destructor slots are zero words in the middle of its
// vtable, and one multiple-inheritance class with a genuine secondary vtable.
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

// An abstract class whose destructor sits BETWEEN other virtuals: GCC emits
// the abstract class's two destructor entries as zero words, so BN's own
// `Codec::VTable` struct ends at the first null (one slot) while the table
// goes on with `__cxa_pure_virtual` and `version()`. A slot walk bounded by
// BN's struct width hid those slots and called the table complete.
struct Codec {
    virtual int probe() { return 1; }
    virtual ~Codec() {}
    virtual int decode(int unit) = 0;
    virtual int version() { return 3; }
};

struct HexCodec : Codec {
    int decode(int unit) override { return unit * 16; }
};

// Multiple inheritance: Pipe's secondary (Source) vtable sits inside the
// `vtable for svc::Pipe` symbol's group at offset-to-top -8, and its slots are
// `_ZThn8_` thunks into Pipe's own methods -- the ABI's layout, not a guess.
struct Sink {
    virtual ~Sink() {}
    virtual int put(int unit) { return unit; }
};

struct Source {
    virtual ~Source() {}
    virtual int get() { return 0; }
};

struct Pipe : Sink, Source {
    int last = 0;
    int put(int unit) override { last = unit; return unit; }
    int get() override { return last; }
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
    svc::HexCodec hex;
    svc::Pipe pipe;
    svc::Handler *h = argc > 1 ? static_cast<svc::Handler *>(&audit) : &echo;
    svc::Codec *codec = &hex;
    svc::Source *source = &pipe;
    if (argc > 2) {
        h = &box;
    }
    pipe.put(codec->probe() + codec->decode(argc) + codec->version());
    try {
        return svc::dispatch(*h, argc - 1) + source->get();
    } catch (const svc::ParseError &ex) {
        std::puts(ex.what());
        return 2;
    }
}
