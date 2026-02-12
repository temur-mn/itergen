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

## CLI with no arguments does not generate data

This is expected. No-arg mode prints guidance only. Use one of:

```bash
python -m vorongen --sample mixed --rows 1000
python -m vorongen --config path/to/config.yaml --rows 1000
python sample_run.py
```

## Validate config before generation

Use validate-only mode to surface schema or feasibility issues early:

`python -m vorongen --config path/to/config.yaml --validate-config`

## Config dependency errors

If validation fails with missing parent/column references:

- verify `depend_on` values match existing `column_id`s
- run with `--missing-columns-mode error` for deterministic failure
- start from a built-in sample and incrementally adapt

## Torch controller requested but torch unavailable

If `--use-torch-controller` is set and torch is not installed:

- with `--torch-required`, the run fails fast
- without `--torch-required`, vorongen falls back to classic controller

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
