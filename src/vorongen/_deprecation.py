"""Deprecation helpers for compatibility wrappers."""

from __future__ import annotations

import warnings


def warn_flat_module(old_module: str, new_module: str) -> None:
    """Emit a deprecation warning for a legacy flat-module import path."""

    warnings.warn(
        f"`{old_module}` is deprecated and will be removed in a future release. "
        f"Use `{new_module}` instead.",
        DeprecationWarning,
        stacklevel=3,
    )
