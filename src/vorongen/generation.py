"""Backward-compatible wrapper for ``vorongen.engine.generation``."""

from .engine import generation as _impl

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})
