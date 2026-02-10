"""Public package interface for vorongen."""

from importlib.metadata import PackageNotFoundError, version

from .models import GenerateResult, RunConfig, TorchControllerConfig
from .sample_configs import (
    available_sample_configs,
    get_sample_config,
    load_config,
)
from .synthesizer import (
    VorongenSynthesizer,
    compare_torch_vs_classic,
    generate,
    is_torch_available,
)

try:
    __version__ = version("vorongen")
except PackageNotFoundError:
    __version__ = "0.1.0"


__all__ = [
    "GenerateResult",
    "RunConfig",
    "TorchControllerConfig",
    "VorongenSynthesizer",
    "available_sample_configs",
    "compare_torch_vs_classic",
    "generate",
    "get_sample_config",
    "is_torch_available",
    "load_config",
]
