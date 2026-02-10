"""
Benchmark optimizer proposal scoring modes.

This compares:
- proposal_scoring_mode="full" (full objective recompute per proposal)
- proposal_scoring_mode="incremental" (dependency-scoped preview path)

Run:
    python -m project.benchmarks.benchmark_optimizer_scoring

Optional env overrides:
    BENCH_ROWS, BENCH_REPEATS, BENCH_MAX_ITERS, BENCH_BATCH_SIZE,
    BENCH_PATIENCE, BENCH_TOLERANCE, BENCH_BASE_SEED
"""

import os
import statistics
import time

import yaml

from project import defaults
from project.config import resolve_missing_columns, validate_config
from project.generation import generate_until_valid
from project.sample_configs import CONFIG_MIXED_LARGE


def _env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


BENCH_ROWS = max(200, _env_int("BENCH_ROWS", 4000))
BENCH_REPEATS = max(1, _env_int("BENCH_REPEATS", 3))
BENCH_MAX_ITERS = max(5, _env_int("BENCH_MAX_ITERS", 80))
BENCH_BATCH_SIZE = max(64, _env_int("BENCH_BATCH_SIZE", 512))
BENCH_PATIENCE = max(2, _env_int("BENCH_PATIENCE", 14))
BENCH_TOLERANCE = max(1e-6, _env_float("BENCH_TOLERANCE", 0.02))
BENCH_BASE_SEED = _env_int("BENCH_BASE_SEED", defaults.DEFAULT_SEED)


def _prepare_config():
    config = yaml.safe_load(CONFIG_MIXED_LARGE)
    config = resolve_missing_columns(config, mode="error")
    validate_config(config)
    return config


def _build_optimize_kwargs(mode):
    settings = defaults.derive_settings(BENCH_ROWS, BENCH_TOLERANCE)
    settings["max_iters"] = BENCH_MAX_ITERS
    settings["patience"] = BENCH_PATIENCE
    settings["batch_size"] = BENCH_BATCH_SIZE
    return {
        **settings,
        "log_level": "quiet",
        "weight_marginal": defaults.DEFAULT_WEIGHT_MARGINAL,
        "weight_conditional": defaults.DEFAULT_WEIGHT_CONDITIONAL,
        "flip_mode": defaults.DEFAULT_FLIP_MODE,
        "proposal_scoring_mode": mode,
    }


def _run_once(mode, seed):
    config = _prepare_config()
    optimize_kwargs = _build_optimize_kwargs(mode)

    t0 = time.perf_counter()
    _df, eq, ok, attempts, _history, _initial_df = generate_until_valid(
        config,
        n_rows=BENCH_ROWS,
        base_seed=seed,
        max_attempts=1,
        tolerance=BENCH_TOLERANCE,
        optimize_kwargs=optimize_kwargs,
        log_level="quiet",
        collect_history=False,
        logger=None,
    )
    elapsed = time.perf_counter() - t0

    if eq is None:
        raise RuntimeError(f"No equilibrium metrics produced for mode={mode}")

    return {
        "elapsed_sec": elapsed,
        "objective": float(eq.get("objective", 0.0)),
        "ok": bool(ok),
        "attempts": int(attempts),
    }


def _summary_rows(results_by_seed):
    rows = {}
    for mode in ("full", "incremental"):
        runs = [payload[mode] for payload in results_by_seed.values()]
        times = [run["elapsed_sec"] for run in runs]
        objectives = [run["objective"] for run in runs]
        rows[mode] = {
            "count": len(runs),
            "avg_sec": statistics.mean(times),
            "median_sec": statistics.median(times),
            "min_sec": min(times),
            "max_sec": max(times),
            "std_sec": statistics.pstdev(times) if len(times) > 1 else 0.0,
            "avg_objective": statistics.mean(objectives),
            "ok_runs": sum(1 for run in runs if run["ok"]),
        }
    return rows


def main():
    seeds = [BENCH_BASE_SEED + idx for idx in range(BENCH_REPEATS)]
    results_by_seed = {}

    print("[BENCHMARK] optimizer proposal scoring")
    print(
        f"rows={BENCH_ROWS} repeats={BENCH_REPEATS} max_iters={BENCH_MAX_ITERS} "
        f"batch_size={BENCH_BATCH_SIZE} patience={BENCH_PATIENCE} tol={BENCH_TOLERANCE}"
    )

    for idx, seed in enumerate(seeds):
        order = ("full", "incremental") if idx % 2 == 0 else ("incremental", "full")
        results_by_seed[seed] = {}
        for mode in order:
            run = _run_once(mode, seed)
            results_by_seed[seed][mode] = run
            print(
                f"  seed={seed} mode={mode:<11} time={run['elapsed_sec']:.3f}s "
                f"objective={run['objective']:.6f} ok={run['ok']}"
            )

    summary = _summary_rows(results_by_seed)
    full_avg = summary["full"]["avg_sec"]
    incremental_avg = summary["incremental"]["avg_sec"]
    speedup = full_avg / incremental_avg if incremental_avg > 0 else float("inf")

    objective_diffs = []
    for seed in seeds:
        full_obj = results_by_seed[seed]["full"]["objective"]
        inc_obj = results_by_seed[seed]["incremental"]["objective"]
        objective_diffs.append(abs(full_obj - inc_obj))

    print("\n[SUMMARY]")
    for mode in ("full", "incremental"):
        row = summary[mode]
        print(
            f"  {mode:<11} avg={row['avg_sec']:.3f}s median={row['median_sec']:.3f}s "
            f"std={row['std_sec']:.3f}s min={row['min_sec']:.3f}s max={row['max_sec']:.3f}s "
            f"avg_objective={row['avg_objective']:.6f} ok={row['ok_runs']}/{row['count']}"
        )

    print(f"  speedup_full_over_incremental={speedup:.2f}x")
    print(
        f"  objective_abs_diff_max={max(objective_diffs):.10f} "
        f"objective_abs_diff_avg={statistics.mean(objective_diffs):.10f}"
    )


if __name__ == "__main__":
    main()
