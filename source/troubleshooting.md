# Troubleshooting

## `openpyxl` missing when saving output

If you see:

`Saving Excel output requires openpyxl.`

Install dependencies:

```bash
pip install vorongen
```

or

```bash
pip install openpyxl
```

## `python -m vorongen` exits immediately

This is expected. `vorongen` is API-only and no longer provides a CLI.

Use one of:

```bash
python sample_run.py
```

or run from Python:

```python
from vorongen import RunConfig, VorongenSynthesizer, get_sample_config

result = VorongenSynthesizer(get_sample_config("mixed"), RunConfig(n_rows=1000)).generate()
```

## Validate config before generation

Use programmatic validation to surface schema or feasibility issues early:

```python
from vorongen.schema.config import build_column_specs, check_feasibility, validate_config

warnings = validate_config(config)
specs = build_column_specs(config)
feas_warnings, feas_errors = check_feasibility(config, specs, n_rows=1000)
```

## Config dependency errors

If validation fails with missing parent/column references:

- verify `depend_on` values match existing `column_id`s
- use `missing_columns_mode="error"` for deterministic failure
- start from a built-in sample and incrementally adapt

## Torch controller requested but torch unavailable

If torch controller mode is requested and torch is not installed:

- with `RunConfig(torch_required=True)`, the run fails fast
- with `RunConfig(torch_required=False)`, vorongen falls back to classic controller

Install torch extras when you want strict torch-backed runs:

`pip install -e .[torch]`

## Docs build fails

Install docs extras first:

`pip install -e .[docs]`

Then run:

`python -m sphinx -W -b html source build/html`

## Slow convergence

- reduce row count during tuning iterations
- simplify conditional branches first, then add complexity
- inspect `quality_report` and focus on worst columns/conditions
