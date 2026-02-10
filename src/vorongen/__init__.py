"""Public import-first API for the ``vorongen`` package."""

from .pipeline import (
    VorongenSynthesizer,
    VoronogeenSynthesizer,
    compare_torch_vs_classic,
    generate,
    load_config,
)
from .samples import available_sample_configs, get_sample_config, get_sample_yaml
from .settings import RunConfig, SynthesisResult, TorchControllerConfig
from .torch_controller import TorchPenaltyController, is_torch_available

__version__ = "0.1.0"

__all__ = [
    "RunConfig",
    "SynthesisResult",
    "TorchControllerConfig",
    "TorchPenaltyController",
    "VorongenSynthesizer",
    "VoronogeenSynthesizer",
    "available_sample_configs",
    "compare_torch_vs_classic",
    "generate",
    "get_sample_config",
    "get_sample_yaml",
    "is_torch_available",
    "load_config",
]
