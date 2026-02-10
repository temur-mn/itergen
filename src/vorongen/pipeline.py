"""High-level package API for stateful synthetic-data generation."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import yaml

from . import defaults
from .settings import RunConfig, SynthesisResult
from .torch_controller import build_configured_controller_class, is_torch_available
from .config import (
    build_column_specs,
    check_feasibility,
    resolve_missing_columns,
    validate_config,
)
from .generation import generate_until_valid
from .metrics import build_quality_report, default_equilibrium_rules
from . import optimizer as optimizer_module


ConfigSource = Union[Dict[str, Any], str, Path]


def load_config(config_source: ConfigSource) -> Dict[str, Any]:
    """Load config from a dict, YAML string, or YAML file path."""
    if isinstance(config_source, dict):
        return deepcopy(config_source)

    if isinstance(config_source, Path):
        text = config_source.expanduser().read_text(encoding="utf-8")
        parsed = yaml.safe_load(text)
    elif isinstance(config_source, str):
        stripped = config_source.strip()
        candidate = Path(stripped).expanduser()
        is_file = False
        if "\n" not in stripped:
            try:
                is_file = candidate.is_file()
            except OSError:
                is_file = False

        if is_file:
            text = candidate.read_text(encoding="utf-8")
            parsed = yaml.safe_load(text)
        else:
            parsed = yaml.safe_load(stripped)
    else:
        raise TypeError("config_source must be dict, YAML string, or path")

    if not isinstance(parsed, dict):
        raise ValueError("Config did not parse into a dictionary")
    return parsed


@contextmanager
def _patched_penalty_controller(controller_cls) -> Iterator[None]:
    original_controller = optimizer_module.PenaltyController
    optimizer_module.PenaltyController = controller_cls
    try:
        yield
    finally:
        optimizer_module.PenaltyController = original_controller


def _decode_categorical_values(df, column_specs):
    out = df.copy()
    for col_id, spec in column_specs.items():
        code_to_cat = spec.get("code_to_cat")
        if not code_to_cat:
            continue
        if col_id not in out.columns:
            continue
        mapped = out[col_id].map(code_to_cat)
        out[col_id] = mapped.where(mapped.notna(), out[col_id])
    return out


class VorongenSynthesizer:
    """Stateful generation pipeline with optional PyTorch adaptation."""

    def __init__(
        self, config_source: ConfigSource, run_config: Optional[RunConfig] = None
    ):
        self.raw_config = load_config(config_source)
        self.run_config = run_config or RunConfig()

        self._resolved_config: Optional[Dict[str, Any]] = None
        self._validation_warnings = []
        self._feasibility_warnings = []
        self.last_result: Optional[SynthesisResult] = None

    def _build_rules(self) -> Dict[str, float]:
        rules = default_equilibrium_rules(self.run_config.tolerance)
        for key, value in self.run_config.equilibrium_rule_overrides.items():
            rules[key] = float(value)
        return rules

    def _build_optimize_kwargs(self) -> Dict[str, Any]:
        kwargs = defaults.derive_settings(
            n_rows=self.run_config.n_rows,
            tolerance=self.run_config.tolerance,
        )
        kwargs.update(
            {
                "log_level": self.run_config.log_level,
                "weight_marginal": defaults.DEFAULT_WEIGHT_MARGINAL,
                "weight_conditional": defaults.DEFAULT_WEIGHT_CONDITIONAL,
                "flip_mode": defaults.DEFAULT_FLIP_MODE,
                "proposal_scoring_mode": self.run_config.proposal_scoring_mode,
            }
        )
        kwargs.update(self.run_config.optimize_overrides)
        return kwargs

    def _controller_context(self, runtime_notes):
        torch_cfg = self.run_config.torch_controller
        if not self.run_config.use_torch_controller or not torch_cfg.enabled:
            return nullcontext()

        if not is_torch_available():
            message = (
                "PyTorch is not installed; falling back to classic controller. "
                "Install `torch` to enable adaptive neural controller mode."
            )
            if self.run_config.torch_required:
                raise RuntimeError(message)
            runtime_notes.append(message)
            return nullcontext()

        runtime_notes.append("PyTorch adaptive controller enabled.")
        controller_cls = build_configured_controller_class(torch_cfg)
        return _patched_penalty_controller(controller_cls)

    def prepare(self) -> "VorongenSynthesizer":
        resolved = resolve_missing_columns(
            self.raw_config,
            mode=self.run_config.missing_columns_mode,
        )
        warnings = validate_config(resolved)

        specs = build_column_specs(resolved)
        optimize_kwargs = self._build_optimize_kwargs()
        min_group_size = int(optimize_kwargs.get("min_group_size", 25))
        feasibility_warnings, feasibility_errors = check_feasibility(
            resolved,
            specs,
            n_rows=self.run_config.n_rows,
            min_group_size=min_group_size,
        )
        if feasibility_errors:
            raise ValueError("; ".join(feasibility_errors))

        self._resolved_config = resolved
        self._validation_warnings = list(warnings)
        self._feasibility_warnings = list(feasibility_warnings)
        return self

    def generate(self) -> SynthesisResult:
        if self._resolved_config is None:
            self.prepare()
        assert self._resolved_config is not None

        optimize_kwargs = self._build_optimize_kwargs()
        rules = self._build_rules()
        runtime_notes = []

        with self._controller_context(runtime_notes):
            df, metrics, ok, attempts, history, initial_df = generate_until_valid(
                self._resolved_config,
                n_rows=self.run_config.n_rows,
                base_seed=self.run_config.seed,
                max_attempts=self.run_config.max_attempts,
                tolerance=self.run_config.tolerance,
                rules=rules,
                optimize_kwargs=optimize_kwargs,
                log_level=self.run_config.log_level,
                collect_history=self.run_config.collect_history,
                logger=None,
            )

        if df is None or metrics is None:
            raise RuntimeError("Generation did not return a dataset and metrics")

        column_specs = build_column_specs(self._resolved_config)
        min_group_size = int(optimize_kwargs.get("min_group_size", 25))
        small_group_mode = str(
            optimize_kwargs.get("small_group_mode", defaults.DEFAULT_SMALL_GROUP_MODE)
        )

        quality = build_quality_report(
            df,
            column_specs,
            min_group_size=min_group_size,
            small_group_mode=small_group_mode,
            top_n=self.run_config.quality_top_n,
        )

        encoded_df = df.copy()
        if self.run_config.decode_categorical:
            display_df = _decode_categorical_values(df, column_specs)
            initial_display = (
                _decode_categorical_values(initial_df, column_specs)
                if initial_df is not None
                else None
            )
        else:
            display_df = df.copy()
            initial_display = initial_df.copy() if initial_df is not None else None

        result = SynthesisResult(
            dataframe=display_df,
            encoded_dataframe=encoded_df,
            metrics=metrics,
            quality_report=quality,
            success=bool(ok),
            attempts=int(attempts),
            history=history,
            initial_dataframe=initial_display,
            resolved_config=deepcopy(self._resolved_config),
            validation_warnings=list(self._validation_warnings),
            feasibility_warnings=list(self._feasibility_warnings),
            runtime_notes=runtime_notes,
        )
        self.last_result = result
        return result

    def fit(self) -> SynthesisResult:
        """Alias for ``generate`` to align with ML pipeline vocabulary."""
        return self.generate()


def generate(
    config_source: ConfigSource, run_config: Optional[RunConfig] = None
) -> SynthesisResult:
    """Functional API to run synthesis in one call."""
    return VorongenSynthesizer(
        config_source=config_source, run_config=run_config
    ).generate()


def compare_torch_vs_classic(
    config_source: ConfigSource,
    run_config: Optional[RunConfig] = None,
) -> Dict[str, SynthesisResult]:
    """Run the same configuration with and without the PyTorch controller."""
    base = run_config or RunConfig()
    classic_cfg = replace(base, use_torch_controller=False)
    torch_cfg = replace(base, use_torch_controller=True)

    classic = generate(config_source, classic_cfg)
    torch_result = generate(config_source, torch_cfg)
    return {"classic": classic, "torch": torch_result}


# Backwards-compatible alias retained for older notebook text.
VoronogeenSynthesizer = VorongenSynthesizer
