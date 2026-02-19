"""
Generation loop that retries until equilibrium rules are satisfied.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import pandas as pd

from ..runtime.rng import RNG
from ..schema.config import build_column_specs, check_feasibility
from ..scoring.metrics import (
    check_equilibrium_rules,
    compute_equilibrium_metrics,
    default_equilibrium_rules,
)
from .initial import generate_initial
from .optimizer import optimize


def _run_single_attempt(
    attempt,
    config,
    column_specs,
    n_rows,
    base_seed,
    tolerance,
    rules,
    optimize_kwargs,
    min_group_size,
    weight_marginal,
    weight_conditional,
    small_group_mode,
    collect_history,
):
    attempt_seed = RNG.derive_seed(base_seed, "attempt", attempt)
    initial_df = generate_initial(
        n_rows, config, seed=attempt_seed, column_specs=column_specs
    )
    df = initial_df.copy()
    history = [] if collect_history else None

    df = optimize(
        df,
        config,
        column_specs,
        seed=attempt_seed,
        tolerance=tolerance,
        history=history,
        logger=None,
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
    return {
        "attempt": int(attempt),
        "attempt_seed": int(attempt_seed),
        "df": df,
        "metrics": metrics,
        "ok": bool(ok),
        "violations": list(violations),
        "history": history,
        "initial_df": initial_df,
    }


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
    attempt_workers=1,
) -> tuple[
    pd.DataFrame | None,
    dict[str, Any] | None,
    bool,
    int,
    list[dict[str, Any]] | None,
    pd.DataFrame | None,
]:
    if optimize_kwargs is None:
        optimize_kwargs = {}
    else:
        optimize_kwargs = dict(optimize_kwargs)
    if rules is None:
        rules = default_equilibrium_rules(tolerance)

    try:
        attempt_workers = int(attempt_workers)
    except (TypeError, ValueError):
        attempt_workers = 1
    attempt_workers = max(1, attempt_workers)

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

    if attempt_workers > 1 and max_attempts is not None and int(max_attempts) > 1:
        total_attempts = int(max_attempts)
        worker_count = min(attempt_workers, total_attempts)

        if logger is not None and log_level != "quiet":
            logger.info(
                "[ATTEMPT MODE] "
                f"parallel workers={worker_count} total_attempts={total_attempts} "
                "selection=deterministic_attempt_order"
            )

        try:
            first_success_attempt = None
            first_success_result = None
            best_failure_result = None
            failed_violations = {}
            next_to_resolve = 0
            next_to_submit = 0
            pending_results = {}

            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_to_attempt = {}

                def _submit_attempt(attempt_index):
                    future = executor.submit(
                        _run_single_attempt,
                        attempt_index,
                        config,
                        column_specs,
                        n_rows,
                        base_seed,
                        tolerance,
                        rules,
                        optimize_kwargs,
                        min_group_size,
                        weight_marginal,
                        weight_conditional,
                        small_group_mode,
                        collect_history,
                    )
                    future_to_attempt[future] = attempt_index

                while (
                    len(future_to_attempt) < worker_count
                    and next_to_submit < total_attempts
                ):
                    _submit_attempt(next_to_submit)
                    next_to_submit += 1

                completed_attempts = 0
                while future_to_attempt:
                    future = next(as_completed(tuple(future_to_attempt)))
                    future_to_attempt.pop(future)
                    completed_attempts += 1
                    result = future.result()
                    attempt = int(result["attempt"])
                    pending_results[attempt] = result

                    if logger is not None and log_level != "quiet":
                        metrics = result.get("metrics", {})
                        objective = metrics.get("objective")
                        max_error = metrics.get("max_error")
                        try:
                            objective_text = f"{float(objective):.6f}"
                        except (TypeError, ValueError):
                            objective_text = "n/a"
                        try:
                            max_error_text = f"{float(max_error):.6f}"
                        except (TypeError, ValueError):
                            max_error_text = "n/a"

                        status_text = "OK" if result.get("ok") else "RETRY"
                        logger.info(
                            "[ATTEMPT COMPLETE] "
                            f"completed={completed_attempts}/{total_attempts} "
                            f"attempt={attempt + 1}/{total_attempts} "
                            f"status={status_text} "
                            f"objective={objective_text} max_error={max_error_text}"
                        )

                    while next_to_resolve in pending_results:
                        ordered = pending_results.pop(next_to_resolve)
                        if ordered["ok"]:
                            first_success_attempt = next_to_resolve
                            first_success_result = ordered
                            break

                        failed_violations[next_to_resolve] = list(
                            ordered.get("violations", [])
                        )
                        if (
                            best_failure_result is None
                            or ordered["metrics"]["objective"]
                            < best_failure_result["metrics"]["objective"]
                        ):
                            best_failure_result = ordered
                        next_to_resolve += 1

                    if first_success_result is not None:
                        for pending_future in tuple(future_to_attempt):
                            pending_future.cancel()
                        break

                    while (
                        len(future_to_attempt) < worker_count
                        and next_to_submit < total_attempts
                    ):
                        _submit_attempt(next_to_submit)
                        next_to_submit += 1

            if first_success_result is not None and first_success_attempt is not None:
                attempts_used = first_success_attempt + 1
                if logger is not None and log_level != "quiet":
                    for attempt in range(attempts_used):
                        if attempt not in failed_violations:
                            continue
                        details = ", ".join(failed_violations[attempt])
                        logger.info(f"[RETRY] rules_not_met={details}")
                return (
                    first_success_result["df"],
                    first_success_result["metrics"],
                    True,
                    attempts_used,
                    first_success_result["history"],
                    first_success_result["initial_df"],
                )

            if best_failure_result is not None:
                if logger is not None and log_level != "quiet":
                    for attempt in range(total_attempts):
                        violations = failed_violations.get(attempt)
                        if not violations:
                            continue
                        details = ", ".join(violations)
                        logger.info(f"[RETRY] rules_not_met={details}")
                return (
                    best_failure_result["df"],
                    best_failure_result["metrics"],
                    False,
                    total_attempts,
                    best_failure_result["history"],
                    best_failure_result["initial_df"],
                )

            return None, None, False, total_attempts, None, None
        except Exception as exc:
            if logger is not None and log_level != "quiet":
                logger.warning(
                    "[ATTEMPT MODE] "
                    f"parallel execution failed ({exc}); falling back to sequential"
                )

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

        history: list[dict[str, Any]] | None = [] if collect_history else None

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
