# Feature Guide

`itergen` is a config-driven synthetic tabular data engine with explicit support for
dependency-aware binary, categorical, and continuous generation.

This guide describes each current feature type and how it fits into the runtime.

## Feature Types At A Glance

| Feature type | What it provides | Where to use it |
| --- | --- | --- |
| Schema and distributions | Binary, categorical, and continuous targets with dependency conditions | `columns[*].distribution` |
| Dependency management | Ordered dependency execution, condition key parsing, missing-reference strategies | `depend_on`, `missing_columns_mode` |
| Runtime orchestration | Retry loops, attempt-level parallelism, deterministic seeds | `RunConfig`, `metadata` |
| Optimization controls | Guided plus random proposals, annealing, plateau stop, per-column step multipliers | `advanced`, `RunConfig.proposal_scoring_mode` |
| Controller backends | Classic controller by default, optional torch backend | `RunConfig(use_torch_controller=...)` |
| Quality and rule checks | Objective, max error, column deviation, continuous bin and bounds validation | `quality_report`, `rules_overrides`, metadata rule keys |
| Observability and artifacts | Timestamped workbook output, per-run logs, runtime notes, optional history | `GenerateResult` |

## 1. Schema And Distribution Features

### Binary: `bernoulli`

- Independent binary column with explicit `true_prob` and `false_prob`.
- Best for root features with no parent dependencies.

### Binary: `conditional`

- Binary target controlled by one or more parents declared in `depend_on`.
- Each `conditional_probs` key uses `col=value` pairs (for example
  `"device=mobile, loyalty=1"`).
- Supports conditional behavior modes (`hard`, `soft`, `fallback`) when advanced
  mode is enabled.

### Categorical

- Supports finite category domains via `values.categories`.
- Supports optional marginal probabilities plus full conditional probability maps.
- Internally encoded for optimization and decoded in final output.

### Continuous

- Supports moment and range targets via `targets: {mean, std, min, max}`.
- Supports `conditioning_bins` to make continuous columns dependency-safe.
- Supports global `bin_probs`, plus conditional moments and conditional bin
  probabilities.
- Supports bin-target conflict policy through
  `metadata.continuous_bin_conflict_mode` (`infer`, `warn`, `error`).

## 2. Dependency And Integrity Features

- Dependency graph ordering is resolved automatically before generation.
- Condition keys are validated against declared domains and dependency parents.
- Missing references are handled via `missing_columns_mode`:
  - `error`: strict fail-fast.
  - `skip`: prune dependent branches automatically.
  - `prompt`: interactive remediation for local/manual runs.
- Feasibility checks estimate expected support for each conditional group and warn
  when `min_group_size` is likely unrealistic.

## 3. Runtime And Execution Features

- Unified high-level API through `ItergenSynthesizer` and `RunConfig`.
- Runtime precedence model: `RunConfig` overrides metadata values.
- Attempt-based generation with best-effort fallback if strict rules are not met.
- Parallel full attempts via `attempt_workers` (process-level), with deterministic attempt-order selection.
- Deterministic randomness via seed derivation for reproducibility.

## 4. Optimization Features

- Mixed strategy search: guided flips + random flips per batch.
- Annealed acceptance schedule (`temperature_init`, `temperature_decay`).
- Two scoring modes:
  - `incremental`: patch-based objective updates for speed.
  - `full`: full metric recomputation for each proposal.
- Convergence controls via tolerance, `max_iters`, and plateau `patience`.
- Small-group handling modes:
  - `ignore`: skip tiny groups for conditional scoring.
  - `downweight`: keep groups with reduced impact.
  - `lock`: protect small-group rows during proposal flips.

## 5. Controller Features

### Classic Controller (default)

- Lightweight controller with no torch dependency.
- Learns per-column multipliers from marginal/conditional error trends.

### Torch Controller (optional)

- Neural controller with embedding plus MLP predictor for multipliers.
- Enable with `RunConfig(use_torch_controller=True)`.
- Optional strict mode via `RunConfig(torch_required=True)`.
- Safe fallback to classic mode when torch is unavailable and strict mode is off.

## 6. Quality, Rules, And Diagnostics

- Final metrics include:
  - objective, mean_marginal, mean_conditional
  - max_error, max_column_deviation
  - continuous bin errors and continuous bounds violations
- `quality_report` includes:
  - confidence score
  - per-column marginal/conditional error breakdown
  - worst conditionals, worst continuous bins, worst continuous bounds
  - detected small groups
- Rule thresholds can be set from tolerance defaults, metadata overrides, or
  `RunConfig.rules_overrides`.

## 7. Output, Logs, And Runtime Artifacts

- Output workbook write can be disabled via `save_output=false`
  (`metadata.save_output` or `RunConfig.save_output`).
- When output saving is enabled, workbook is written to a timestamped path if a
  filename is not provided.
- Each run writes a timestamped log file under the configured `log_dir`
  (`metadata.log_dir` or `RunConfig.log_dir`; default `src/itergen/logs/`).
- `GenerateResult` returns:
  - `dataframe`, `metrics`, `quality_report`
  - `success`, `attempts`, `output_path` (or `None`), `log_path`
  - `runtime_notes`, plus optional `history` and `initial_dataframe`

## 8. Built-In Feature Scenarios

Built-in sample configs are designed as feature templates:

- `binary`, `categorical`, `continuous`
- `mixed`, `mixed_large`
- `binary_categorical_large`, `continuous_parent_bins`

Use `available_sample_configs()` to list them and `get_sample_config(name)` to
load a deep-copy config payload for adaptation.

## Related Docs

- Runtime setup and examples: `quickstart.md`
- Full config reference and examples: `configuration.md`
- Public API surface: `api.md`
- System internals: `architecture.md`
