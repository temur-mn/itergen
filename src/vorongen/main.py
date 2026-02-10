"""
Entry point for generating synthetic binary data from a sample config.
"""

from pathlib import Path

import yaml

from . import defaults
from .config import build_column_specs, resolve_missing_columns, validate_config
from .generation import generate_until_valid
from .logging_utils import setup_run_logger
from .metrics import build_quality_report, default_equilibrium_rules
from .sample_configs import CONFIG_MIXED_LARGE


def _build_equilibrium_rules(metadata, tolerance, logger):
    rules = default_equilibrium_rules(tolerance)
    for key in (
        "objective_max",
        "max_error_max",
        "max_column_deviation_max",
        "continuous_bin_mean_error_max",
        "continuous_bin_max_error_max",
        "continuous_violation_rate_max",
        "continuous_mean_violation_max",
        "continuous_max_violation_max",
    ):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            rules[key] = float(value)
        except (TypeError, ValueError):
            logger.warning(f"metadata.{key} must be numeric; ignoring override")
    return rules


def main():
    logger, log_path = setup_run_logger()

    raw_config = yaml.safe_load(CONFIG_MIXED_LARGE)
    config = raw_config
    metadata = config.get("metadata", {})

    missing_columns_mode = metadata.get(
        "missing_columns_mode", defaults.DEFAULT_MISSING_COLUMNS_MODE
    )
    if missing_columns_mode not in ("prompt", "skip", "error"):
        missing_columns_mode = defaults.DEFAULT_MISSING_COLUMNS_MODE

    run_log_level = metadata.get("log_level", defaults.DEFAULT_LOG_LEVEL)
    if run_log_level not in ("info", "quiet"):
        run_log_level = defaults.DEFAULT_LOG_LEVEL

    proposal_scoring_mode = metadata.get("proposal_scoring_mode", "incremental")
    if proposal_scoring_mode not in ("incremental", "full"):
        proposal_scoring_mode = "incremental"

    output_path = metadata.get("output_path", defaults.DEFAULT_OUTPUT_PATH)
    if not isinstance(output_path, str) or not output_path.strip():
        output_path = defaults.DEFAULT_OUTPUT_PATH
    output_path = output_path.strip()

    config = resolve_missing_columns(config, mode=missing_columns_mode)
    warnings = validate_config(config)
    if warnings:
        logger.warning("[CONFIG WARNINGS]")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    n_rows = metadata.get("n_rows", defaults.DEFAULT_ROWS)
    base_seed = metadata.get("seed", defaults.DEFAULT_SEED)
    tolerance = metadata.get("tolerance", defaults.DEFAULT_TOLERANCE)
    max_attempts = metadata.get("max_attempts", defaults.DEFAULT_MAX_ATTEMPTS)
    rules = _build_equilibrium_rules(metadata, tolerance, logger)
    small_group_mode = metadata.get(
        "small_group_mode", defaults.DEFAULT_SMALL_GROUP_MODE
    )

    settings = defaults.derive_settings(n_rows, tolerance)
    settings["small_group_mode"] = small_group_mode

    advanced = config.get("advanced", {})
    advanced_enabled = bool(advanced.get("enabled"))
    if advanced_enabled:
        for key, value in advanced.items():
            if key in settings:
                settings[key] = value
    optimize_kwargs = {
        **settings,
        "log_level": run_log_level,
        "weight_marginal": defaults.DEFAULT_WEIGHT_MARGINAL,
        "weight_conditional": defaults.DEFAULT_WEIGHT_CONDITIONAL,
        "flip_mode": defaults.DEFAULT_FLIP_MODE,
        "proposal_scoring_mode": proposal_scoring_mode,
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

    df, eq, ok, attempts, _history, _initial_df = generate_until_valid(
        config,
        n_rows=n_rows,
        base_seed=base_seed,
        max_attempts=max_attempts,
        tolerance=tolerance,
        rules=rules,
        optimize_kwargs=optimize_kwargs,
        log_level=run_log_level,
        collect_history=False,
        logger=logger,
    )

    if eq is None or df is None:
        raise ValueError("No dataset or metrics available")
    assert eq is not None
    assert df is not None

    column_specs = build_column_specs(config)
    display_df = df.copy()
    for col_id, spec in column_specs.items():
        if spec.get("kind") != "categorical":
            continue
        code_to_cat = spec.get("code_to_cat")
        if not code_to_cat or col_id not in display_df.columns:
            continue
        mapped = display_df[col_id].map(code_to_cat)
        display_df[col_id] = mapped.where(mapped.notna(), display_df[col_id])

    status = "OK" if ok else "BEST_EFFORT"
    if run_log_level != "quiet":
        logger.info(
            f"[FINAL METRICS] status={status} attempts={attempts} "
            f"objective={eq['objective']:.6f} mean_marginal={eq['mean_marginal']:.6f} "
            f"mean_conditional={eq['mean_conditional']:.6f} max_error={eq['max_error']:.6f} "
            f"max_column_deviation={eq['max_column_deviation']:.6f} "
            f"continuous_bin_max_error={eq['continuous_bin_max_error']:.6f}"
        )

    if run_log_level != "quiet":
        logger.info("\n[FINAL STATS]")
        for col in raw_config["columns"]:
            col_id = col["column_id"]
            if col_id in display_df.columns:
                spec = column_specs.get(col_id)
                if spec and spec.get("kind") == "continuous":
                    series = display_df[col_id]
                    stats = {
                        "count": int(series.count()),
                        "mean": float(series.mean()),
                        "std": float(series.std(ddof=0)),
                        "min": float(series.min()),
                        "max": float(series.max()),
                    }
                    logger.info(f"  {col_id}: {stats}")
                else:
                    logger.info(
                        f"  {col_id}: {display_df[col_id].value_counts().to_dict()}"
                    )

    output_file = Path(output_path).expanduser()
    if output_file.parent != Path("."):
        output_file.parent.mkdir(parents=True, exist_ok=True)
    display_df.to_excel(output_file)
    if run_log_level != "quiet":
        logger.info(f"\n [DATASET IS SAVED] path={output_file}")

    quality = build_quality_report(
        df,
        column_specs,
        min_group_size=settings["min_group_size"],
        small_group_mode=settings.get("small_group_mode", "ignore"),
        top_n=5,
    )
    if run_log_level != "quiet":
        logger.info("\n[QUALITY REPORT]")
        logger.info(
            f"  confidence={quality['confidence']:.3f} "
            f"objective={quality['objective']:.6f} "
            f"mean_marginal={quality['mean_marginal']:.6f} "
            f"mean_conditional={quality['mean_conditional']:.6f} "
            f"max_error={quality['max_error']:.6f} "
            f"max_column_deviation={quality['max_column_deviation']:.6f} "
            f"continuous_bin_mean_error={quality['continuous_bin_mean_error']:.6f} "
            f"continuous_bin_max_error={quality['continuous_bin_max_error']:.6f} "
            f"continuous_violation_rate={quality['continuous_violation_rate']:.6f} "
            f"continuous_mean_violation={quality['continuous_mean_violation']:.6f} "
            f"continuous_max_violation={quality['continuous_max_violation']:.6f}"
        )
        logger.info("  per_column:")
        for row in quality["per_column"]:
            logger.info(
                f"    - {row['column_id']}: marginal_error={row['marginal_error']:.6f} "
                f"conditional_error={row['conditional_error']:.6f}"
            )
        logger.info("  worst_conditionals:")
        for row in quality["worst_conditionals"]:
            ignored = " (ignored)" if row.get("ignored") else ""
            logger.info(
                f"    - {row['column_id']} | {row['condition']} | "
                f"error={row['error']:.6f} | group_size={row['group_size']}{ignored}"
            )
        if quality["small_groups"]:
            logger.info("  small_groups:")
            for row in quality["small_groups"][:10]:
                logger.info(
                    f"    - {row['column_id']} | {row['condition']} | n={row['group_size']}"
                )
        if quality["worst_continuous_bounds"]:
            logger.info("  worst_continuous_bounds:")
            for row in quality["worst_continuous_bounds"]:
                condition = row.get("condition") or "<marginal>"
                logger.info(
                    f"    - {row['column_id']} | {condition} | "
                    f"violation_rate={row['violation_rate']:.6f} "
                    f"mean_violation={row['mean_violation']:.6f} "
                    f"max_violation={row['max_violation']:.6f}"
                )
        if quality["worst_continuous_bins"]:
            logger.info("  worst_continuous_bins:")
            for row in quality["worst_continuous_bins"]:
                condition = row.get("condition") or "<marginal>"
                logger.info(
                    f"    - {row['column_id']} | {condition} | "
                    f"bin_error={row['error']:.6f} | group_size={row['group_size']}"
                )

    print(
        f"[FINAL SUMMARY] status={status} attempts={attempts} "
        f"confidence={quality['confidence']:.3f} "
        f"objective={quality['objective']:.6f} output={output_file} log={log_path}"
    )


if __name__ == "__main__":
    main()
