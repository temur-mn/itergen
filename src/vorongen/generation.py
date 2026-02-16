"""Backward-compatible wrapper for ``vorongen.engine.generation``."""

from ._deprecation import warn_flat_module
from .engine import generation as _impl

warn_flat_module("vorongen.generation", "vorongen.engine.generation")

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})
