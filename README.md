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

## Quickstart (CLI)

Run the packaged module entrypoint:

```bash
python -m vorongen
```

By default, if no filename is provided, generated data is written to
`output/<timestamp>_vorongen.xlsx`.

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
