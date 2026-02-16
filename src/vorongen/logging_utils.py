"""Backward-compatible wrapper for ``vorongen.runtime.logging_utils``."""

from ._deprecation import warn_flat_module
from .runtime import logging_utils as _impl

warn_flat_module("vorongen.logging_utils", "vorongen.runtime.logging_utils")

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})
