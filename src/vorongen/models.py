"""Backward-compatible wrapper for ``vorongen.api.models``."""

from ._deprecation import warn_flat_module
from .api import models as _impl

warn_flat_module("vorongen.models", "vorongen.api.models")

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})
