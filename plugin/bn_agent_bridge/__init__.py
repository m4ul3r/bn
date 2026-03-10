from .bridge import start_bridge, start_headless, ui

# Auto-start only when loaded as a Binary Ninja GUI plugin.
# Headless callers use start_headless() directly.
if ui is not None:
    start_bridge()
