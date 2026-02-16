"""Public package interface for itergen."""

from importlib.metadata import PackageNotFoundError, version

from .api.models import GenerateResult, RunConfig, TorchControllerConfig
from .api.synthesizer import (
    ItergenSynthesizer,
    compare_torch_vs_classic,
    generate,
    is_torch_available,
)
from .schema.samples import (
    available_sample_configs,
    get_sample_config,
    load_config,
)

try:
    __version__ = version("itergen")
except PackageNotFoundError:
    __version__ = "0.1.0"


__all__ = [
    "GenerateResult",
    "RunConfig",
    "TorchControllerConfig",
    "ItergenSynthesizer",
    "available_sample_configs",
    "compare_torch_vs_classic",
    "generate",
    "get_sample_config",
    "is_torch_available",
    "load_config",
]
