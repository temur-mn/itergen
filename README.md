# vorongen

`vorongen` generates synthetic tabular datasets from declarative YAML-like configs
with support for binary, categorical, and continuous dependencies.

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
# Development tooling
pip install -e .[dev]

# Documentation tooling
pip install -e .[docs]

# Optional torch-backed controller features
pip install -e .[torch]
```

## Documentation

Build the Sphinx docs locally:

```bash
python -m sphinx -W -b html source build/html
```

Main docs pages:

- `source/quickstart.md`
- `source/configuration.md`
- `source/api.md`
- `source/architecture.md`
- `source/troubleshooting.md`

## Quickstart (CLI)

Run the packaged module entrypoint:

```bash
python -m vorongen
```

With no arguments, the CLI prints guided next steps (notebook + sample script +
explicit commands).

To generate data directly from CLI:

```bash
python -m vorongen --list-samples
python -m vorongen --sample mixed --rows 1200 --log-level quiet
python -m vorongen --config path/to/config.yaml --rows 1200
python -m vorongen --config path/to/config.yaml --validate-config
```

Default controller backend is classic (no torch dependency required).
Use torch-backed controller when available:

```bash
python -m vorongen --sample mixed --use-torch-controller --torch-device auto
```

If no explicit output filename is provided, generated data is written to
`output/<timestamp>_vorongen.xlsx`.

## Quickstart (Script + Notebook)

Use the suggested script for a local smoke run:

```bash
python sample_run.py
```

Or run interactively in notebook:

- `notebooks/testing_new_tools.ipynb`

## Quickstart (Programmatic)

```python
from vorongen import RunConfig, VorongenSynthesizer, get_sample_config

config = get_sample_config("mixed")
run_cfg = RunConfig(
    n_rows=3000,
    seed=101,
    tolerance=0.04,
    max_attempts=2,
    log_level="quiet",
)

result = VorongenSynthesizer(config, run_cfg).generate()

print(result.success, result.objective())
print(result.output_path)
print(result.dataframe.head())
```

## Sample Configurations

Built-in sample config strings are available in `vorongen.sample_configs`,
including:

- `CONFIG_BINARY`
- `CONFIG_CATEGORICAL`
- `CONFIG_CONTINUOUS`
- `CONFIG_MIXED`
- `CONFIG_MIXED_LARGE`

## Notebooks

- `notebooks/testing_new_tools.ipynb`
- `notebooks/notes.ipynb`

## Project standards

- Contribution guide: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Changelog: `CHANGELOG.md`

## Quality Checks

```bash
pip install -e .[dev]
ruff check .
ruff format --check .
mypy src/vorongen
PYTHONPATH=src python -m unittest discover -s tests -p "test*.py" -q
coverage run -m unittest discover -s tests -p "test*.py"
coverage report --fail-under=85
python -m build
```
