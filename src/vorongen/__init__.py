"""Voronogeen package: import-first API over the vorongen core engine."""

from .pipeline import (
    VorongenSynthesizer,
    compare_torch_vs_classic,
    generate,
    load_config,
)
from .samples import available_sample_configs, get_sample_config, get_sample_yaml
from .settings import RunConfig, SynthesisResult, TorchControllerConfig
from .torch_controller import TorchPenaltyController, is_torch_available

__all__ = [
    "RunConfig",
    "SynthesisResult",
    "TorchControllerConfig",
    "TorchPenaltyController",
    "VoronogeenSynthesizer",
    "available_sample_configs",
    "compare_torch_vs_classic",
    "generate",
    "get_sample_config",
    "get_sample_yaml",
    "is_torch_available",
    "load_config",
]
