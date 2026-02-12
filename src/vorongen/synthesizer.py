"""High-level import-first runtime API for synthetic dataset generation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from . import defaults
from .config import build_column_specs, resolve_missing_columns, validate_config
from .generation import generate_until_valid
from .logging_utils import setup_run_logger
from .metrics import build_quality_report, default_equilibrium_rules
from .models import GenerateResult, RunConfig, TorchControllerConfig
from .sample_configs import load_config

_VALID_LOG_LEVELS = {"info", "quiet"}
_VALID_MISSING_MODES = {"prompt", "skip", "error"}
_VALID_SCORING_MODES = {"incremental", "full"}


def is_torch_available() -> bool:
    """Return True when `torch` can be imported in the runtime environment."""

    return find_spec("torch") is not None


def _coerce_int(value: Any, fallback: int, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(fallback)
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def _coerce_float(value: Any, fallback: float, minimum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(fallback)
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def _normalize_choice(value: Any, allowed: set[str], fallback: str) -> str:
    text = str(value).strip().lower() if value is not None else ""
    if text in allowed:
        return text
    return fallback


def _controller_config_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    return {}


def _timestamped_output_name() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{stamp}_vorongen.xlsx"


def _resolve_output_path(output_path: str | None) -> Path:
    raw = output_path if output_path is not None else defaults.DEFAULT_OUTPUT_PATH
    text = str(raw).strip()
    if not text:
        text = defaults.DEFAULT_OUTPUT_PATH

    path = Path(text).expanduser()
    is_dir = (
        text.endswith("/")
        or text.endswith("\\")
        or path.suffix == ""
        or (path.exists() and path.is_dir())
    )
    if is_dir:
        path = path / _timestamped_output_name()
    return path


def _apply_numeric_rule_override(
    rules: dict[str, float], target_key: str, label: str, value: Any, logger
) -> None:
    if value is None:
        return
    try:
        rules[target_key] = float(value)
    except (TypeError, ValueError):
        logger.warning(f"{label} must be numeric; ignoring override")


def _build_equilibrium_rules(
    metadata: Mapping[str, Any], tolerance: float, overrides: Mapping[str, Any], logger
) -> dict[str, float]:
    rules = default_equilibrium_rules(tolerance)
    known_keys = (
        "objective_max",
        "max_error_max",
        "max_column_deviation_max",
        "continuous_bin_mean_error_max",
        "continuous_bin_max_error_max",
        "continuous_violation_rate_max",
        "continuous_mean_violation_max",
        "continuous_max_violation_max",
    )
    for key in known_keys:
        _apply_numeric_rule_override(
            rules,
            key,
            f"metadata.{key}",
            metadata.get(key),
            logger,
        )

    for key, value in (overrides or {}).items():
        _apply_numeric_rule_override(
            rules,
            key,
            f"run_config.rules_overrides.{key}",
            value,
            logger,
        )

    return rules


def _decode_output_dataframe(df, column_specs):
    display_df = df.copy()
    for col_id, spec in column_specs.items():
        if spec.get("kind") != "categorical":
            continue
        code_to_cat = spec.get("code_to_cat")
        if not code_to_cat or col_id not in display_df.columns:
            continue
        mapped = display_df[col_id].map(code_to_cat)
        display_df[col_id] = mapped.where(mapped.notna(), display_df[col_id])
    return display_df


class VorongenSynthesizer:
    """High-level facade for running config-driven synthetic generation."""

    def __init__(self, config: Any, run_config: RunConfig | None = None):
        self._config = load_config(config)
        self.run_config = run_config or RunConfig()

    def generate(self) -> GenerateResult:
        config = copy.deepcopy(self._config)
        metadata = config.get("metadata", {}) if isinstance(config, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}

        logger, log_path = setup_run_logger(name="vorongen")
        runtime_notes = []

        torch_available = is_torch_available()
        controller_backend = "classic"
        if self.run_config.use_torch_controller:
            if torch_available:
                controller_backend = "torch"
            elif self.run_config.torch_required:
                raise RuntimeError(
                    "Torch controller was requested as required, "
                    "but torch is not installed"
                )
            else:
                runtime_notes.append(
                    "Torch controller requested but torch is unavailable; "
                    "falling back to classic controller"
                )

        controller_config = _controller_config_payload(self.run_config.torch_controller)
        if controller_backend == "torch" and not controller_config:
            controller_config = _controller_config_payload(TorchControllerConfig())
        runtime_notes.append(f"Controller backend: {controller_backend}")

        missing_mode = _normalize_choice(
            self.run_config.missing_columns_mode
            if self.run_config.missing_columns_mode is not None
            else metadata.get(
                "missing_columns_mode", defaults.DEFAULT_MISSING_COLUMNS_MODE
            ),
            _VALID_MISSING_MODES,
            defaults.DEFAULT_MISSING_COLUMNS_MODE,
        )
        log_level = _normalize_choice(
            self.run_config.log_level
            if self.run_config.log_level is not None
            else metadata.get("log_level", defaults.DEFAULT_LOG_LEVEL),
            _VALID_LOG_LEVELS,
            defaults.DEFAULT_LOG_LEVEL,
        )
        scoring_mode = _normalize_choice(
            self.run_config.proposal_scoring_mode
            if self.run_config.proposal_scoring_mode is not None
            else metadata.get("proposal_scoring_mode", "incremental"),
            _VALID_SCORING_MODES,
            "incremental",
        )

        n_rows = _coerce_int(
            self.run_config.n_rows,
            metadata.get("n_rows", defaults.DEFAULT_ROWS),
            minimum=1,
        )
        base_seed = _coerce_int(
            self.run_config.seed,
            metadata.get("seed", defaults.DEFAULT_SEED),
        )
        tolerance = _coerce_float(
            self.run_config.tolerance,
            metadata.get("tolerance", defaults.DEFAULT_TOLERANCE),
            minimum=1e-6,
        )
        max_attempts = _coerce_int(
            self.run_config.max_attempts,
            metadata.get("max_attempts", defaults.DEFAULT_MAX_ATTEMPTS),
            minimum=1,
        )

        output_raw = (
            self.run_config.output_path
            if self.run_config.output_path is not None
            else metadata.get("output_path", defaults.DEFAULT_OUTPUT_PATH)
        )
        output_file = _resolve_output_path(output_raw)

        config = resolve_missing_columns(config, mode=missing_mode)
        warnings = validate_config(config)
        if warnings and log_level != "quiet":
            logger.warning("[CONFIG WARNINGS]")
            for warning in warnings:
                logger.warning(f"  - {warning}")

        settings = defaults.derive_settings(n_rows, tolerance)
        small_group_mode = (
            self.run_config.small_group_mode
            if self.run_config.small_group_mode is not None
            else metadata.get("small_group_mode", defaults.DEFAULT_SMALL_GROUP_MODE)
        )
        settings["small_group_mode"] = small_group_mode

        attempt_workers = metadata.get("attempt_workers", 1)
        advanced = config.get("advanced", {})
        advanced_enabled = isinstance(advanced, dict) and bool(advanced.get("enabled"))
        if advanced_enabled:
            for key, value in advanced.items():
                if key in settings:
                    settings[key] = value
            if "attempt_workers" in advanced:
                attempt_workers = advanced.get("attempt_workers")

        if self.run_config.attempt_workers is not None:
            attempt_workers = self.run_config.attempt_workers
        attempt_workers = _coerce_int(attempt_workers, 1, minimum=1)

        rules = _build_equilibrium_rules(
            metadata,
            tolerance,
            self.run_config.rules_overrides,
            logger,
        )

        optimize_kwargs = {
            **settings,
            "log_level": log_level,
            "weight_marginal": defaults.DEFAULT_WEIGHT_MARGINAL,
            "weight_conditional": defaults.DEFAULT_WEIGHT_CONDITIONAL,
            "flip_mode": defaults.DEFAULT_FLIP_MODE,
            "proposal_scoring_mode": scoring_mode,
            "controller_backend": controller_backend,
            "controller_config": controller_config,
        }
        if advanced_enabled:
            for key in (
                "weight_marginal",
                "weight_conditional",
                "flip_mode",
                "small_group_mode",
            ):
                if key in advanced:
                    optimize_kwargs[key] = advanced[key]
        if self.run_config.optimize_overrides:
            optimize_kwargs.update(self.run_config.optimize_overrides)

        df, metrics, ok, attempts, history, initial_df = generate_until_valid(
            config,
            n_rows=n_rows,
            base_seed=base_seed,
            max_attempts=max_attempts,
            attempt_workers=attempt_workers,
            tolerance=tolerance,
            rules=rules,
            optimize_kwargs=optimize_kwargs,
            log_level=log_level,
            collect_history=bool(self.run_config.collect_history),
            logger=logger,
        )

        if metrics is None or df is None:
            raise ValueError("No dataset or metrics available")

        column_specs = build_column_specs(config)
        display_df = _decode_output_dataframe(df, column_specs)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            display_df.to_excel(output_file)
        except ModuleNotFoundError as exc:
            if getattr(exc, "name", "") == "openpyxl":
                raise RuntimeError(
                    "Saving Excel output requires openpyxl. "
                    "Install with `pip install openpyxl`."
                ) from exc
            raise

        quality = build_quality_report(
            df,
            column_specs,
            min_group_size=optimize_kwargs.get(
                "min_group_size", settings["min_group_size"]
            ),
            small_group_mode=optimize_kwargs.get("small_group_mode", "ignore"),
            top_n=5,
        )

        if not ok:
            runtime_notes.append(
                "Rules not fully satisfied; returning best-effort dataset from attempts"
            )

        if log_level != "quiet":
            status = "OK" if ok else "BEST_EFFORT"
            logger.info(
                f"[FINAL METRICS] status={status} attempts={attempts} "
                f"objective={metrics['objective']:.6f} "
                f"max_error={metrics['max_error']:.6f} output={output_file}"
            )

        return GenerateResult(
            dataframe=display_df,
            metrics=metrics,
            quality_report=quality,
            success=bool(ok),
            attempts=int(attempts),
            output_path=output_file,
            log_path=Path(log_path),
            runtime_notes=runtime_notes,
            history=history,
            initial_dataframe=initial_df,
        )


def generate(config: Any, run_config: RunConfig | None = None) -> GenerateResult:
    """Convenience function for one-off generation calls."""

    return VorongenSynthesizer(config, run_config=run_config).generate()


def compare_torch_vs_classic(
    config: Any, run_config: RunConfig | None = None
) -> dict[str, GenerateResult]:
    """Run classic and torch-requested flows for quick side-by-side comparison."""

    base_cfg = run_config or RunConfig()
    classic_cfg = replace(base_cfg, use_torch_controller=False, torch_required=False)
    torch_cfg = replace(base_cfg, use_torch_controller=True, torch_required=False)
    return {
        "classic": generate(config, classic_cfg),
        "torch": generate(config, torch_cfg),
    }
