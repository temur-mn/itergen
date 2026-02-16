"""Backward-compatible wrapper for ``vorongen.schema.samples``."""

from ._deprecation import warn_flat_module
from .schema import samples as _impl

warn_flat_module("vorongen.sample_configs", "vorongen.schema.samples")

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})
