"""Public runtime models for the import-first API."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class TorchControllerConfig:
    """Configuration for the torch-backed step-size controller."""

    lr: float = 2e-3
    hidden_dim: int = 48
    weight_decay: float = 0.0
    min_mult: float = 0.025
    max_mult: float = 0.075
    trend_scale: float = 0.7
    ema_alpha: float = 0.6
    base_weight: float = 0.01
    device: str = "cpu"


@dataclass
class RunConfig:
    """Top-level runtime options for synthetic generation."""

    n_rows: Optional[int] = None
    seed: Optional[int] = None
    tolerance: Optional[float] = None
    max_attempts: Optional[int] = None
    log_level: Optional[str] = None
    output_path: Optional[str] = None
    attempt_workers: Optional[int] = None
    proposal_scoring_mode: Optional[str] = None
    missing_columns_mode: Optional[str] = None
    small_group_mode: Optional[str] = None
    collect_history: bool = False
    optimize_overrides: Dict[str, Any] = field(default_factory=dict)
    rules_overrides: Dict[str, Any] = field(default_factory=dict)
    use_torch_controller: bool = False
    torch_required: bool = False
    torch_controller: Optional[TorchControllerConfig] = None


@dataclass
class GenerateResult:
    """Result payload returned by high-level generation APIs."""

    dataframe: pd.DataFrame
    metrics: Dict[str, Any]
    quality_report: Dict[str, Any]
    success: bool
    attempts: int
    output_path: Path
    log_path: Path
    runtime_notes: List[str] = field(default_factory=list)
    history: Optional[List[Dict[str, Any]]] = None
    initial_dataframe: Optional[pd.DataFrame] = None

    def objective(self) -> float:
        """Return the objective score from the final metrics payload."""

        return float(self.metrics.get("objective", 0.0))
