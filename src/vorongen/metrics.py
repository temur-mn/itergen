"""Backward-compatible wrapper for ``vorongen.scoring.metrics``."""

from ._deprecation import warn_flat_module
from .scoring import metrics as _impl

warn_flat_module("vorongen.metrics", "vorongen.scoring.metrics")

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})
