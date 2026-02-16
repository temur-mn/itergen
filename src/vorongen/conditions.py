"""Backward-compatible wrapper for ``vorongen.scoring.conditions``."""

from ._deprecation import warn_flat_module
from .scoring import conditions as _impl

warn_flat_module("vorongen.conditions", "vorongen.scoring.conditions")

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})
