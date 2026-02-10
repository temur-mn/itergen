"""Typed configuration and result objects for the package API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class TorchControllerConfig:
    """Configuration for the PyTorch adaptive penalty controller."""

    enabled: bool = True
    lr: float = 3e-3
    hidden_dim: int = 48
    min_multiplier: float = 0.02
    max_multiplier: float = 0.10
    l2: float = 1e-4
    ema_alpha: float = 0.65
    trend_scale: float = 0.8
    base_weight: float = 0.10
    device: str = "cpu"


@dataclass
class RunConfig:
    """Runtime settings for synthesis runs."""

    n_rows: int = 5000
    seed: int = 42
    tolerance: float = 0.03
    max_attempts: int = 3
    log_level: str = "quiet"
    missing_columns_mode: str = "skip"
    proposal_scoring_mode: str = "incremental"
    collect_history: bool = True
    decode_categorical: bool = True
    quality_top_n: int = 7
    use_torch_controller: bool = True
    torch_required: bool = False
    optimize_overrides: Dict[str, Any] = field(default_factory=dict)
    equilibrium_rule_overrides: Dict[str, float] = field(default_factory=dict)
    torch_controller: TorchControllerConfig = field(
        default_factory=TorchControllerConfig
    )


@dataclass
class SynthesisResult:
    """Structured output returned by the package API."""

    dataframe: pd.DataFrame
    encoded_dataframe: pd.DataFrame
    metrics: Dict[str, Any]
    quality_report: Dict[str, Any]
    success: bool
    attempts: int
    history: Optional[List[Dict[str, Any]]]
    initial_dataframe: Optional[pd.DataFrame]
    resolved_config: Dict[str, Any]
    validation_warnings: List[str]
    feasibility_warnings: List[str]
    runtime_notes: List[str]

    def objective(self) -> float:
        """Return the optimization objective for quick comparisons."""
        return float(self.metrics.get("objective", 0.0))
