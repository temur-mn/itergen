"""Backward-compatible wrapper for ``vorongen.controllers.classic``."""

from ._deprecation import warn_flat_module
from .controllers import classic as _impl

warn_flat_module("vorongen.nn_controller", "vorongen.controllers.classic")

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})
