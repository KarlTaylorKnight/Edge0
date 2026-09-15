"""Streaming expert offload: mmap stores, shared LRU caches, staged
decode slots, whole-layer prefill and hot-expert pins.

``StreamingSwitchGLU`` and ``install_streaming_experts`` bind to the
active backend at import; they are exposed lazily (PEP 562) so the
backend-free modules (``cache``, ``options``, ``budget``, ``mmap``)
stay importable on hosts with no backend installed — the Task 4 budget
tests run everywhere.
"""

from edge0.streaming.cache import PrefetchBuffer, SharedExpertCache
from edge0.streaming.mmap import SafetensorsMmap
from edge0.streaming.options import LayerOptions

__all__ = [
    "SafetensorsMmap",
    "SharedExpertCache",
    "PrefetchBuffer",
    "StreamingSwitchGLU",
    "LayerOptions",
    "install_streaming_experts",
]

_LAZY = {
    "StreamingSwitchGLU": ("edge0.streaming.layer", "StreamingSwitchGLU"),
    "install_streaming_experts": ("edge0.streaming.install",
                                  "install_streaming_experts"),
}


def __getattr__(name):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None
    import importlib
    return getattr(importlib.import_module(module_name), attr)
