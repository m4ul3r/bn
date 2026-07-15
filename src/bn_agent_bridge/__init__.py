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
    start_bridge = start_headless = None  # type: ignore[assignment]
    ui = None
else:
    # Auto-start only when loaded as a Binary Ninja GUI plugin.
    # Headless callers use start_headless() directly.
    if ui is not None:
        start_bridge()
