# Configuration

`vorongen` accepts config input as either:

- a Python mapping (`dict`)
- YAML text
- a YAML file loaded via CLI (`--config`)

Use `vorongen.get_sample_config(<name>)` to start from known-good examples.

## Core sections

Each config typically includes:

- `metadata`: run defaults and behavior flags
- `columns`: column definitions with distributions and dependencies
- `advanced` (optional): optimizer tuning overrides

## Distribution types

- `bernoulli`: independent binary column (`true_prob`, `false_prob`)
- `conditional`: binary column conditional on parent columns
- `categorical`: finite-category column with marginal/conditional probabilities
- `continuous`: continuous column with targets and optional conditional targets

## Runtime overrides

Runtime settings can come from config metadata and/or `RunConfig`:

- `n_rows`
- `seed`
- `tolerance`
- `max_attempts`
- `log_level`
- `output_path`
- `attempt_workers`
- `missing_columns_mode`
- `proposal_scoring_mode`
- `small_group_mode`
- `use_torch_controller`
- `torch_required`
- `torch_controller`

CLI flags map to the same runtime controls, and CLI flags take precedence over
config metadata.

Use CLI validation mode to check config parse, schema, and feasibility without
running generation:

`python -m vorongen --config path/to/config.yaml --validate-config`

Validation mode is the recommended first step before long runs and CI jobs.

## Missing dependencies behavior

`missing_columns_mode` controls how unresolved dependencies are handled:

- `error`: fail fast
- `skip`: prune invalid branches
- `prompt`: interactive guidance path

For non-interactive environments (CI/scripts), prefer `error`.

## Controller backend behavior

- default backend is `classic`
- set `--use-torch-controller` to request `torch`
- set `--torch-required` to fail fast if torch is unavailable
- when torch is optional and unavailable, the runtime falls back to classic

## Output behavior

When output filename is omitted, vorongen writes a timestamped workbook to
`output/`, for example:

`output/20260210_151200_123456_vorongen.xlsx`
