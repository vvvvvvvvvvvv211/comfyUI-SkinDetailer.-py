"""ComfyUI-SkinDetailer -- automatic skin-only detail restoration.

Exposes one node: ``Skin Detailer (SAM3 • Texture • Refine)``.
"""

from .node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

#: Served to the frontend so ``web/skin_detailer.js`` is loaded with the node.
WEB_DIRECTORY = "./web"

__version__ = "1.5.0"
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "__version__",
]
