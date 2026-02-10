"""Sample config helpers for quick package onboarding."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

import yaml
from . import sample_configs


_SAMPLE_MAP = {
    "binary": sample_configs.CONFIG_BINARY,
    "categorical": sample_configs.CONFIG_CATEGORICAL,
    "continuous": sample_configs.CONFIG_CONTINUOUS,
    "mixed": sample_configs.CONFIG_MIXED,
    "mixed_large": sample_configs.CONFIG_MIXED_LARGE,
    "binary_categorical_large": sample_configs.CONFIG_BINARY_CATEGORICAL_LARGE,
    "continuous_parent_bins": sample_configs.CONFIG_CONTINUOUS_PARENT_BINS,
}


def available_sample_configs() -> List[str]:
    """Return supported sample config names."""
    return sorted(_SAMPLE_MAP.keys())


def get_sample_config(name: str) -> Dict[str, Any]:
    """Load one of the built-in sample configs by name."""
    key = str(name or "").strip().lower()
    if key not in _SAMPLE_MAP:
        available = ", ".join(available_sample_configs())
        raise KeyError(f"Unknown sample config '{name}'. Available: {available}")

    parsed = yaml.safe_load(_SAMPLE_MAP[key])
    if not isinstance(parsed, dict):
        raise ValueError(f"Sample config '{key}' did not parse to a dictionary")
    return deepcopy(parsed)


def get_sample_yaml(name: str) -> str:
    """Return the raw YAML text for a built-in sample config."""
    key = str(name or "").strip().lower()
    if key not in _SAMPLE_MAP:
        available = ", ".join(available_sample_configs())
        raise KeyError(f"Unknown sample config '{name}'. Available: {available}")
    return _SAMPLE_MAP[key]
