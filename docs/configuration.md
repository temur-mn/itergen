# Configuration

`vorongen` accepts configuration as:

- a Python mapping (`dict`)
- YAML text
- a YAML file loaded and parsed in Python (`load_config(...)`)

Use `get_sample_config(<name>)` to start from known-good templates and adapt safely.

## Configuration Shape

Typical structure:

```yaml
metadata:
  n_rows: 3000
  seed: 42
  tolerance: 0.02
  max_attempts: 3
  attempt_workers: 2

advanced:
  enabled: true
  max_iters: 200
  proposals_per_batch: 32

columns:
  - column_id: "loyalty"
    values: { true_value: 1, false_value: 0 }
    distribution:
      type: "bernoulli"
      probabilities: { true_prob: 0.45, false_prob: 0.55 }
```

Core sections:

- `metadata`: runtime defaults, rule thresholds, behavior flags
- `columns`: schema and target distributions
- `advanced` (optional): optimizer-level tuning controls

## Runtime Metadata Reference

`metadata` supports these runtime-focused keys:

| Key | Purpose |
| --- | --- |
| `n_rows` | Default row count when `RunConfig.n_rows` is not provided |
| `seed` | Base deterministic seed |
| `tolerance` | Default objective target level |
| `max_attempts` | Attempt retry budget |
| `attempt_workers` | Process-level parallel attempt workers |
| `log_level` | `info` or `quiet` |
| `log_dir` | Directory for per-run log files |
| `output_path` | Output file path or directory hint |
| `proposal_scoring_mode` | `incremental` or `full` proposal scoring |
| `missing_columns_mode` | `prompt`, `skip`, or `error` |
| `small_group_mode` | Small-group handling strategy during optimization/scoring |
| `conditional_mode` | Global conditional behavior: `hard`, `soft`, `fallback` |
| `continuous_bin_conflict_mode` | Continuous bin conflict behavior: `infer`, `warn`, `error` |

Rule-threshold keys (optional) are also read from metadata:

- `objective_max`
- `max_error_max`
- `max_column_deviation_max`
- `continuous_bin_mean_error_max`
- `continuous_bin_max_error_max`
- `continuous_violation_rate_max`
- `continuous_mean_violation_max`
- `continuous_max_violation_max`

## Column Feature Types

### 1) Binary (`bernoulli`)

Independent binary column with explicit probabilities.

```yaml
- column_id: "loyalty"
  values: { true_value: 1, false_value: 0 }
  distribution:
    type: "bernoulli"
    probabilities: { true_prob: 0.4, false_prob: 0.6 }
```

### 2) Binary Conditional (`conditional`)

Binary target controlled by parent columns.

```yaml
- column_id: "discount"
  values: { true_value: 1, false_value: 0 }
  distribution:
    type: "conditional"
    depend_on: ["loyalty"]
    conditional_probs:
      "loyalty=1": { true_prob: 0.55, false_prob: 0.45 }
      "loyalty=0": { true_prob: 0.25, false_prob: 0.75 }
```

### 3) Categorical

Finite category domain with optional marginal and conditional targets.

```yaml
- column_id: "device"
  values:
    categories: ["mobile", "desktop", "tablet"]
  distribution:
    type: "categorical"
    probabilities: { mobile: 0.50, desktop: 0.35, tablet: 0.15 }
    depend_on: ["loyalty"]
    conditional_probs:
      "loyalty=1": { mobile: 0.60, desktop: 0.30, tablet: 0.10 }
      "loyalty=0": { mobile: 0.40, desktop: 0.40, tablet: 0.20 }
```

### 4) Continuous

Continuous target with moment/range constraints and optional conditional targets.

```yaml
- column_id: "support_cost"
  distribution:
    type: "continuous"
    depend_on: ["support_channel", "priority_support"]
    targets: { mean: 18.0, std: 5.0, min: 2.0, max: 60.0 }
    conditioning_bins:
      edges: [1.0, 8.0, 14.0, 20.0, 30.0, 45.0, 90.0]
      labels: ["c1", "c2", "c3", "c4", "c5", "c6"]
    conditional_targets:
      "support_channel=chat, priority_support=1":
        { mean: 18.0, std: 5.0, min: 2.0, max: 60.0 }
```

Continuous columns additionally support:

- `bin_probs` for global bin targets
- `conditional_bin_probs` for condition-specific bin targets

## Advanced Optimizer Features

Enable with `advanced.enabled: true`. Supported keys include:

- Search and schedule:
  `batch_size`, `proposals_per_batch`, `max_iters`, `patience`,
  `temperature_init`, `temperature_decay`
- Step magnitudes:
  `step_size_marginal`, `step_size_conditional`,
  `step_size_continuous_marginal`, `step_size_continuous_conditional`,
  `max_flip_frac`, `random_flip_frac`
- Weighting and behavior:
  `weight_marginal`, `weight_conditional`, `flip_mode`, `small_group_mode`
- Group/category controls:
  `min_group_size`, `large_category_threshold`, `target_column_pool_size`
- Continuous-specialized controls:
  `continuous_dependency_gain`, `continuous_magnifier_min`,
  `continuous_magnifier_max`, `continuous_noise_frac`,
  `continuous_edge_guard_frac`
- Attempt-level parallel override:
  `attempt_workers`

Deprecated advanced keys (`hybrid_ratio`, `weight_max`) are ignored with warnings.

## Precedence: `RunConfig` Vs Metadata

When both are present, `RunConfig` wins over `metadata` for overlapping runtime
options. This allows stable defaults in YAML with explicit per-run overrides.

## Missing Dependency Handling

`missing_columns_mode` controls unresolved references:

- `error`: strict fail-fast (recommended for CI and production)
- `skip`: auto-prune unresolved dependency branches
- `prompt`: interactive remediation for local/manual usage

If `prompt` is used in non-interactive environments, generation fails with a
clear error explaining how to switch modes.

## Validation And Feasibility Workflow

Before expensive runs, validate config and check feasibility:

```python
from vorongen.schema.config import build_column_specs, check_feasibility, validate_config

warnings = validate_config(config)
specs = build_column_specs(config)
feas_warnings, feas_errors = check_feasibility(config, specs, n_rows=1000)
```

What this catches:

- schema issues (unknown distributions, bad condition keys, invalid probabilities)
- dependency integrity issues (missing domains/references)
- likely infeasible conditional group sizes versus `min_group_size`
- continuous bin/target conflicts based on selected conflict policy

## Practical Tuning Sequence

1. Start from a built-in sample (`mixed`, `continuous_parent_bins`, `mixed_large`).
2. Validate and fix warnings that affect dependencies or probability mass.
3. Tune with smaller `n_rows` first, then scale up.
4. Adjust rule thresholds only after reviewing `quality_report` worst offenders.
5. For deterministic automation, keep `missing_columns_mode=error` and fixed seeds.
