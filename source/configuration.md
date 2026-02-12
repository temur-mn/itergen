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

CLI flags map to the same runtime controls.

Use CLI validation mode to check config parse, schema, and feasibility without
running generation:

`python -m vorongen --config path/to/config.yaml --validate-config`

## Missing dependencies behavior

`missing_columns_mode` controls how unresolved dependencies are handled:

- `error`: fail fast
- `skip`: prune invalid branches
- `prompt`: interactive guidance path

For non-interactive environments (CI/scripts), prefer `error`.

## Output behavior

When output filename is omitted, vorongen writes a timestamped workbook to
`output/`, for example:

`output/20260210_151200_123456_vorongen.xlsx`
