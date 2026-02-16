"""Backward-compatible wrapper for ``vorongen.runtime.rng``."""

from ._deprecation import warn_flat_module
from .runtime import rng as _impl

warn_flat_module("vorongen.rng", "vorongen.runtime.rng")

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})
