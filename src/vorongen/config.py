"""Backward-compatible wrapper for ``vorongen.schema.config``."""

from ._deprecation import warn_flat_module
from .schema import config as _impl

warn_flat_module("vorongen.config", "vorongen.schema.config")

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})
