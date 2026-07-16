try:
    from .bridge import start_bridge, start_headless, ui
except ModuleNotFoundError as exc:
    # The bridge (and its helpers) hard-import `binaryninja`, which only exists
    # inside the Binary Ninja Python environment. Importing this package from a
    # plain interpreter -- unit tests that pull in a pure submodule such as
    # `taint_engine`, or headless tooling -- must not crash on the missing
    # module. Re-raise anything that is NOT the absence of Binary Ninja.
    root = (exc.name or "").split(".", 1)[0]
    if root not in ("binaryninja", "binaryninjaui"):
        raise

    def _bn_unavailable(*_args, **_kwargs):
        # Bound in place of the real entry points when Binary Ninja is absent, so a
        # stray package-level `start_bridge()`/`start_headless()` raises a clear
        # message instead of a bare `NoneType is not callable`. (Real headless
        # callers import from `.bridge` directly and get the import error there.)
        raise RuntimeError(
            "Binary Ninja is not available in this interpreter, so the bn bridge "
            "cannot start. Run inside Binary Ninja's Python (the GUI plugin) or via "
            "`bn-agent` with a licensed headless install.")

    start_bridge = start_headless = _bn_unavailable  # type: ignore[assignment]
    ui = None
else:
    # Auto-start only when loaded as a Binary Ninja GUI plugin.
    # Headless callers use start_headless() directly.
    if ui is not None:
        start_bridge()
