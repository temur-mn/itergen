# Troubleshooting

## `openpyxl` missing when saving output

If you see:

`Saving Excel output requires openpyxl.`

Install dependencies:

```bash
pip install -e .
```

or

```bash
pip install openpyxl
```

## CLI with no arguments does not generate data

This is expected. No-arg mode prints guidance only. Use one of:

```bash
python -m vorongen --sample mixed --rows 1000
python -m vorongen --config path/to/config.yaml --rows 1000
python sample_run.py
```

## Config dependency errors

If validation fails with missing parent/column references:

- verify `depend_on` values match existing `column_id`s
- run with `missing_columns_mode=error` for deterministic failure
- start from a built-in sample and incrementally adapt

You can validate config and feasibility without generating data:

`python -m vorongen --config path/to/config.yaml --validate-config`

## Torch controller requested but torch unavailable

If `--use-torch-controller` is set and torch is not installed:

- with `--torch-required`, the run fails fast
- without `--torch-required`, vorongen falls back to classic controller

Install torch extras when you want strict torch-backed runs:

`pip install -e .[torch]`

## Slow convergence

- reduce row count during tuning iterations
- simplify conditional branches first, then add complexity
- inspect `quality_report` and focus on worst columns/conditions
