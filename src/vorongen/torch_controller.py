"""Backward-compatible wrapper for ``vorongen.controllers.torch``."""

from ._deprecation import warn_flat_module
from .controllers import torch as _impl

warn_flat_module("vorongen.torch_controller", "vorongen.controllers.torch")

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})
