"""Backward-compatible wrapper for ``vorongen.engine.initial``."""

from ._deprecation import warn_flat_module
from .engine import initial as _impl

warn_flat_module("vorongen.initial", "vorongen.engine.initial")

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})
