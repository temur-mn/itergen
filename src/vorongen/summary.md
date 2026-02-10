# Module Summary and Current Workflow

This document describes how `project` works today, with emphasis on config integrity,
missing-column handling, feasibility checks, and continuous-variable diagnostics.

Note: `project/` now also serves as the legacy engine backend for the new
publishable package namespace `declarix/`.

## Runtime entrypoint

- `uv run python -m project` executes `project/__main__.py`, which calls `project.main.main()`.
- `project/main.py` currently loads `CONFIG_MIXED_LARGE` from
  `project/sample_configs.py`.
- Runtime behavior is driven by `metadata` defaults in `project/defaults.py`:
  - `missing_columns_mode`: `prompt`
  - `output_path`: `draft_1.xlsx`
  - `log_level`: `info`

## End-to-end flow

1. Load YAML config string.
2. Resolve missing referenced columns (`config.resolve_missing_columns`).
3. Validate config structure (`config.validate_config`).
4. Build typed/normalized specs (`config.build_column_specs`).
5. Run feasibility checks (`config.check_feasibility`).
6. Generate initial sample (`initial.generate_initial`).
7. Optimize with stochastic proposals (`optimizer.optimize`).
   - both guided and random paths use shared condition matching.
8. Evaluate metrics/rules (`metrics.compute_equilibrium_metrics`,
   `metrics.check_equilibrium_rules`).
9. Map categorical codes back to labels, save output, and log quality report.

## Config integrity and missing-column resolution

### Reference model

Reference tracking is built from:

- `distribution.depend_on`
- condition keys inside `conditional_probs`
- condition keys inside `conditional_targets`

Any referenced column not present in `columns` is treated as missing.

### Modes (`metadata.missing_columns_mode`)

- `error`:
  - fail immediately with missing ids and source references.
  - recommended for CI or strict pipelines.

- `skip`:
  - removes all dependent columns **transitively** (across binary/categorical/continuous).
  - dependency graph includes both `depend_on` edges and condition-key references.
  - after pruning, integrity is re-validated; unresolved references fail fast.

- `prompt`:
  - interactive resolver for local debugging.
  - for each missing id, user can:
    1) add Bernoulli stub column,
    2) prune all transitive dependents,
    3) abort.
  - before pruning, a preview shows impacted column count and sample ids.
  - after each decision, references are recomputed until none remain.
  - in non-interactive environments, prompt mode fails with guidance to use
    `error` or `skip`.

### Integrity guarantees after resolution

After `resolve_missing_columns` returns successfully:

- no unresolved references remain in `depend_on` or condition keys,
- pruning is transitive and graph-consistent,
- result is returned as a resolved copy (source object intent remains intact).

## Optimizer and objective model

- Discrete/binary/categorical columns are optimized via guided + random flips.
- Objective uses:
  - `weight_marginal * mean_marginal`
  - `+ weight_conditional * mean_conditional`
- `max_error` and `max_column_deviation` are tracked for rule checks but not part of objective.
- Proposal scoring supports two modes:
  - `incremental` (default): recalculates objective only for impacted dependency slice.
  - `full`: recomputes full objective on each proposal (baseline / benchmarking).

## Continuous variables: current behavior and diagnostics

### Continuous-parent interactions (explicit bins only)

Continuous columns can now be used as dependency parents for binary/categorical/continuous
children, but only through explicit `conditioning_bins` labels.

Example:

```yaml
- column_id: "risk_score"
  distribution:
    type: "continuous"
    targets: { mean: 50.0, std: 15.0, min: 0.0, max: 100.0 }
    conditioning_bins:
      edges: [0.0, 35.0, 65.0, 100.0]
      labels: ["low", "mid", "high"]

- column_id: "retained"
  values: { true_value: 1, false_value: 0 }
  distribution:
    type: "conditional"
    depend_on: ["risk_score"]
    conditional_probs:
      "risk_score=low": { true_prob: 0.8, false_prob: 0.2 }
      "risk_score=mid": { true_prob: 0.5, false_prob: 0.5 }
      "risk_score=high": { true_prob: 0.2, false_prob: 0.8 }
```

Validation is strict:

- if a continuous column is referenced as a dependency parent without valid
  `conditioning_bins`, config validation fails,
- condition values must be in the parent domain (including bin labels),
- unresolved/unsupported condition values are no longer silently dropped during spec build.

Bin interval semantics are:

- `[lower, upper)` for all non-last bins,
- `[lower, upper]` for the last bin.

### Shared condition engine

`project/conditions.py` is now the single condition-matching path used by:

- `initial.py` for conditional initialization,
- `adjustments.py` for guided/random proposal targeting,
- `metrics.py` for conditional scoring and diagnostics,
- `optimizer.py` for small-group lock masks.

This keeps condition semantics identical across generation, optimization, and evaluation.

### What is optimized today

- Continuous columns use a hybrid flow:
  - bin-first distribution control (`conditioning_bins` + `bin_probs` /
    `conditional_bin_probs`),
  - near-bin-preserving numeric updates with bounded noise.
- Continuous objective/error includes bin-distribution divergence (JS) and optional
  mean/std alignment where provided.
- `min`/`max` remain hard bounds for initialization and updates.
- Continuous conditional targets and conditional bin-probs share the same
  `hard`/`soft` blending semantics.

### Added diagnostics

Quality/metrics now report continuous bound compliance separately:

- `continuous_bin_mean_error`: weighted mean JS divergence between observed and
  target bin distributions.
- `continuous_bin_max_error`: worst JS divergence across marginal/conditional
  bin scopes.

- `continuous_violation_rate`: weighted fraction of rows outside bounds.
- `continuous_mean_violation`: weighted mean out-of-bound distance.
- `continuous_max_violation`: worst out-of-bound distance observed.
- `worst_continuous_bounds`: top rows with largest violations (marginal/conditional scope).
- `worst_continuous_bins`: top marginal/conditional scopes with largest bin divergence.

These diagnostics are visibility-first and do not change the optimization objective.

### Optional equilibrium rule keys

Rules can optionally enforce continuous bounds:

- `continuous_violation_rate_max`
- `continuous_mean_violation_max`
- `continuous_max_violation_max`
- `continuous_bin_mean_error_max`
- `continuous_bin_max_error_max`

If omitted, behavior is unchanged from previous defaults.

## Feasibility checks

`check_feasibility` performs two layers:

1. Structural feasibility
   - missing conditional combinations vs dependency-domain cartesian product,
   - conditional mode handling (`hard` can error, `fallback` can disable conditionals).

2. Expected support feasibility
   - estimates conditional group support as:
     - `expected_rows ~= n_rows * P(condition)`
   - `P(condition)` is approximated from parent marginals (or fallback averages).
   - warns when expected rows are near/below `min_group_size`.
   - can escalate to error in `hard` mode when support is clearly too small.

Note: support estimates are approximate and intended as early risk signals.

## Metadata runtime controls

- `missing_columns_mode`: `prompt`, `skip`, `error`.
- `output_path`: output Excel path.
- `log_level`: `info` or `quiet`.
- `proposal_scoring_mode`: `incremental` (default) or `full`.
- `objective_max`, `max_error_max`: optional overrides for base equilibrium checks.
- `max_column_deviation_max`: optional per-column cap (default `tolerance * 1.25`).
- `continuous_violation_rate_max`, `continuous_mean_violation_max`,
  `continuous_max_violation_max`: optional continuous-bound rule thresholds.

Recommended usage:

- local exploratory runs: `prompt`
- CI/production validation: `error`
- automated best-effort pruning pipelines: `skip`

## Latest full-size run snapshot

Fresh production-sized rerun completed with:

- command: `python -m project`
- config: `CONFIG_MIXED_LARGE`
- rows: 30,000
- log: `project/logs/run_20260209_101952.log`
- output: `draft_1.xlsx`

Final status (from log):

- `status=OK`, `attempts=3`
- `objective=0.003756`
- `mean_marginal=0.002859`
- `mean_conditional=0.001121`
- `max_error=0.015280`
- `max_column_deviation=0.009486`
- `confidence=0.996`
- `continuous_violation_rate=0.000058`
- `continuous_mean_violation=0.082960`
- `continuous_max_violation=2.000000`

Observed bottlenecks in this run:

- Retry overhead: first two attempts retried due to `max_error>0.02` before final pass met rules.
- Proposal selectivity is low: acceptance rate remained near-saturated (~99.8%), so many proposals are accepted.
- Per-batch gain is small near convergence: median best `objective_delta` about `-1.6e-05`.
- Throughput hotspot: ~590 batches total with average cadence around ~0.455s/batch.

Most problematic slices in final quality report:

- Worst conditional: `risk_score | loyalty=0, discount=0 | error=0.015280 | group_size=12228`.
- Worst continuous bound slice: `support_cost | support_channel=2, priority_support=0 | max_violation=2.000000`.

## Benchmark helper

Use `project/benchmarks/benchmark_optimizer_scoring.py` to compare `proposal_scoring_mode`
runtime on `CONFIG_MIXED_LARGE`.

Example:

- `python -m project.benchmarks.benchmark_optimizer_scoring`
- Optional env overrides: `BENCH_ROWS`, `BENCH_REPEATS`, `BENCH_MAX_ITERS`,
  `BENCH_BATCH_SIZE`, `BENCH_PATIENCE`, `BENCH_TOLERANCE`, `BENCH_BASE_SEED`.
