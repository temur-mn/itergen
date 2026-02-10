"""
Generation loop that retries until equilibrium rules are satisfied.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import build_column_specs, check_feasibility
from .initial import generate_initial
from .metrics import (
    check_equilibrium_rules,
    compute_equilibrium_metrics,
    default_equilibrium_rules,
)
from .optimizer import optimize
from .rng import RNG


def _print_stats(df, columns, column_specs, label, logger=None):
    if logger is None:
        return
    logger.info(label)
    for col in columns:
        col_id = col["column_id"]
        spec = column_specs.get(col_id)
        if spec and spec.get("kind") == "continuous":
            series = df[col_id]
            stats = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
            logger.info(f"  {col_id}: {stats}")
        else:
            logger.info(f"  {col_id}: {df[col_id].value_counts().to_dict()}")


def generate_until_valid(
    config,
    n_rows,
    base_seed,
    max_attempts=None,
    tolerance=0.02,
    rules=None,
    optimize_kwargs=None,
    log_level="info",
    collect_history=False,
    logger=None,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[Dict[str, Any]],
    bool,
    int,
    Optional[List[Dict[str, Any]]],
    Optional[pd.DataFrame],
]:
    if optimize_kwargs is None:
        optimize_kwargs = {}
    else:
        optimize_kwargs = dict(optimize_kwargs)
    if rules is None:
        rules = default_equilibrium_rules(tolerance)

    per_column_limit = rules.get("max_column_deviation_max", float(tolerance) * 1.25)
    try:
        per_column_limit = float(per_column_limit)
    except (TypeError, ValueError):
        per_column_limit = float(tolerance) * 1.25
    optimize_kwargs.setdefault("max_column_deviation_limit", per_column_limit)

    min_group_size = optimize_kwargs.get("min_group_size", 25)
    weight_marginal = optimize_kwargs.get("weight_marginal", 1.0)
    weight_conditional = optimize_kwargs.get("weight_conditional", 0.6)
    small_group_mode = optimize_kwargs.get("small_group_mode", "ignore")

    column_specs = build_column_specs(config)
    feas_warnings, feas_errors = check_feasibility(
        config,
        column_specs,
        n_rows=n_rows,
        min_group_size=min_group_size,
    )
    if feas_warnings and logger is not None:
        logger.warning("[FEASIBILITY WARNINGS]")
        for warning in feas_warnings:
            logger.warning(f"  - {warning}")
    if feas_errors:
        raise ValueError("; ".join(feas_errors))
    best_df = None
    best_metrics = None
    best_history = None
    best_initial = None

    attempt = 0
    while True:
        if max_attempts is not None and attempt >= max_attempts:
            break
        attempt_seed = RNG.derive_seed(base_seed, "attempt", attempt)
        if logger is not None and log_level != "quiet":
            total = max_attempts if max_attempts is not None else "∞"
            logger.info(f"[ATTEMPT {attempt + 1}/{total}] seed={attempt_seed}")

        initial_df = generate_initial(
            n_rows, config, seed=attempt_seed, column_specs=column_specs
        )
        df = initial_df.copy()
        if log_level != "quiet":
            _print_stats(
                df,
                config["columns"],
                column_specs,
                "[INITIAL STATS]",
                logger=logger,
            )

        history = [] if collect_history else None

        df = optimize(
            df,
            config,
            column_specs,
            seed=attempt_seed,
            tolerance=tolerance,
            history=history,
            logger=logger,
            **optimize_kwargs,
        )

        metrics = compute_equilibrium_metrics(
            df,
            column_specs,
            min_group_size=min_group_size,
            weight_marginal=weight_marginal,
            weight_conditional=weight_conditional,
            small_group_mode=small_group_mode,
            include_continuous_bounds=True,
        )

        ok, violations = check_equilibrium_rules(metrics, rules)
        if ok:
            return df, metrics, True, attempt + 1, history, initial_df

        if best_metrics is None or metrics["objective"] < best_metrics["objective"]:
            best_df = df
            best_metrics = metrics
            best_history = history
            best_initial = initial_df

        if logger is not None and log_level != "quiet":
            details = ", ".join(violations) if violations else "no violations listed"
            logger.info(f"[RETRY] rules_not_met={details}")

        attempt += 1

    return best_df, best_metrics, False, attempt, best_history, best_initial
