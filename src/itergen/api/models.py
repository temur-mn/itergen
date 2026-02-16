"""Public runtime models for the import-first API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

    n_rows: int | None = None
    seed: int | None = None
    tolerance: float | None = None
    max_attempts: int | None = None
    log_level: str | None = None
    log_dir: str | None = None
    output_path: str | None = None
    attempt_workers: int | None = None
    proposal_scoring_mode: str | None = None
    missing_columns_mode: str | None = None
    small_group_mode: str | None = None
    collect_history: bool = False
    optimize_overrides: dict[str, Any] = field(default_factory=dict)
    rules_overrides: dict[str, Any] = field(default_factory=dict)
    use_torch_controller: bool = False
    torch_required: bool = False
    torch_controller: TorchControllerConfig | None = None


@dataclass
class GenerateResult:
    """Result payload returned by high-level generation APIs."""

    dataframe: pd.DataFrame
    metrics: dict[str, Any]
    quality_report: dict[str, Any]
    success: bool
    attempts: int
    output_path: Path
    log_path: Path
    runtime_notes: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] | None = None
    initial_dataframe: pd.DataFrame | None = None

    def objective(self) -> float:
        """Return the objective score from the final metrics payload."""

        return float(self.metrics.get("objective", 0.0))
