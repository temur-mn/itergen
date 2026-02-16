"""
Benchmark attempt-level parallel workers.

This compares wall time of `generate_until_valid` for different `attempt_workers`
values while keeping all optimizer settings fixed.

Run:
    python -m itergen.benchmarks.benchmark_attempt_workers

Optional env overrides:
    BENCH_ROWS, BENCH_REPEATS, BENCH_MAX_ATTEMPTS, BENCH_MAX_ITERS,
    BENCH_BATCH_SIZE, BENCH_PATIENCE, BENCH_TOLERANCE, BENCH_BASE_SEED,
    BENCH_WORKERS

Examples:
    BENCH_WORKERS=1,2 python -m itergen.benchmarks.benchmark_attempt_workers
"""

import os
import statistics
import time

import yaml

from itergen.engine.generation import generate_until_valid
from itergen.schema import defaults
from itergen.schema.config import resolve_missing_columns, validate_config
from itergen.schema.samples import CONFIG_MIXED_LARGE


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


def _env_workers(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    out = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            parsed = int(token)
        except ValueError:
            continue
        if parsed >= 1:
            out.append(parsed)
    return out or default


BENCH_ROWS = max(200, _env_int("BENCH_ROWS", 3000))
BENCH_REPEATS = max(1, _env_int("BENCH_REPEATS", 2))
BENCH_MAX_ATTEMPTS = max(1, _env_int("BENCH_MAX_ATTEMPTS", 3))
BENCH_MAX_ITERS = max(2, _env_int("BENCH_MAX_ITERS", 20))
BENCH_BATCH_SIZE = max(64, _env_int("BENCH_BATCH_SIZE", 512))
BENCH_PATIENCE = max(1, _env_int("BENCH_PATIENCE", 5))
BENCH_TOLERANCE = max(1e-6, _env_float("BENCH_TOLERANCE", 0.02))
BENCH_BASE_SEED = _env_int("BENCH_BASE_SEED", defaults.DEFAULT_SEED)
BENCH_WORKERS = _env_workers("BENCH_WORKERS", [1, 2])


def _prepare_config():
    config = yaml.safe_load(CONFIG_MIXED_LARGE)
    config = resolve_missing_columns(config, mode="error")
    validate_config(config)
    return config


def _build_optimize_kwargs():
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
        "proposal_scoring_mode": "incremental",
    }


def _run_once(workers, seed):
    config = _prepare_config()
    optimize_kwargs = _build_optimize_kwargs()

    t0 = time.perf_counter()
    _df, eq, ok, attempts, _history, _initial_df = generate_until_valid(
        config,
        n_rows=BENCH_ROWS,
        base_seed=seed,
        max_attempts=BENCH_MAX_ATTEMPTS,
        attempt_workers=workers,
        tolerance=BENCH_TOLERANCE,
        optimize_kwargs=optimize_kwargs,
        log_level="quiet",
        collect_history=False,
        logger=None,
    )
    elapsed = time.perf_counter() - t0

    if eq is None:
        raise RuntimeError(f"No equilibrium metrics produced for workers={workers}")

    return {
        "elapsed_sec": elapsed,
        "objective": float(eq.get("objective", 0.0)),
        "ok": bool(ok),
        "attempts": int(attempts),
        "workers": int(workers),
    }


def main():
    seeds = [BENCH_BASE_SEED + idx for idx in range(BENCH_REPEATS)]
    by_workers = {workers: [] for workers in BENCH_WORKERS}

    print("[BENCHMARK] attempt-level workers")
    print(
        f"rows={BENCH_ROWS} repeats={BENCH_REPEATS} max_attempts={BENCH_MAX_ATTEMPTS} "
        f"max_iters={BENCH_MAX_ITERS} batch_size={BENCH_BATCH_SIZE} "
        f"patience={BENCH_PATIENCE} tol={BENCH_TOLERANCE} workers={BENCH_WORKERS}"
    )

    for seed_idx, seed in enumerate(seeds):
        order = BENCH_WORKERS if seed_idx % 2 == 0 else list(reversed(BENCH_WORKERS))
        for workers in order:
            result = _run_once(workers, seed)
            by_workers[workers].append(result)
            print(
                f"  seed={seed} workers={workers:<2} "
                f"time={result['elapsed_sec']:.3f}s attempts={result['attempts']} "
                f"objective={result['objective']:.6f} ok={result['ok']}"
            )

    print("\n[SUMMARY]")
    summary = {}
    for workers in BENCH_WORKERS:
        rows = by_workers[workers]
        times = [row["elapsed_sec"] for row in rows]
        objectives = [row["objective"] for row in rows]
        attempts = [row["attempts"] for row in rows]
        summary[workers] = {
            "avg_sec": statistics.mean(times),
            "median_sec": statistics.median(times),
            "min_sec": min(times),
            "max_sec": max(times),
            "avg_objective": statistics.mean(objectives),
            "avg_attempts": statistics.mean(attempts),
        }
        print(
            f"  workers={workers:<2} avg={summary[workers]['avg_sec']:.3f}s "
            f"median={summary[workers]['median_sec']:.3f}s "
            f"min={summary[workers]['min_sec']:.3f}s "
            f"max={summary[workers]['max_sec']:.3f}s "
            f"avg_attempts={summary[workers]['avg_attempts']:.2f} "
            f"avg_objective={summary[workers]['avg_objective']:.6f}"
        )

    if 1 in summary and len(summary) > 1:
        base = summary[1]["avg_sec"]
        for workers in BENCH_WORKERS:
            if workers == 1:
                continue
            speedup = (
                base / summary[workers]["avg_sec"]
                if summary[workers]["avg_sec"] > 0
                else float("inf")
            )
            print(f"  speedup_workers_{workers}_vs_1={speedup:.2f}x")


if __name__ == "__main__":
    main()
