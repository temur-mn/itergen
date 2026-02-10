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

Current default behavior runs the built-in `CONFIG_MIXED_LARGE` sample and saves
an Excel output (for example `draft_1.xlsx`) plus a run log in
`src/vorongen/logs/`.

## Quickstart (Programmatic)

```python
import yaml

from vorongen import defaults
from vorongen.config import resolve_missing_columns, validate_config
from vorongen.generation import generate_until_valid
from vorongen.sample_configs import CONFIG_MIXED

config = yaml.safe_load(CONFIG_MIXED)
config = resolve_missing_columns(config, mode="error")
validate_config(config)

n_rows = 1000
tolerance = 0.05
settings = defaults.derive_settings(n_rows, tolerance)

optimize_kwargs = {
    **settings,
    "log_level": "quiet",
    "weight_marginal": defaults.DEFAULT_WEIGHT_MARGINAL,
    "weight_conditional": defaults.DEFAULT_WEIGHT_CONDITIONAL,
    "flip_mode": defaults.DEFAULT_FLIP_MODE,
    "proposal_scoring_mode": "incremental",
}

df, metrics, ok, attempts, _history, _initial_df = generate_until_valid(
    config,
    n_rows=n_rows,
    base_seed=defaults.DEFAULT_SEED,
    max_attempts=3,
    tolerance=tolerance,
    optimize_kwargs=optimize_kwargs,
    log_level="quiet",
    collect_history=False,
    logger=None,
)

print("ok:", ok, "attempts:", attempts)
print("objective:", metrics["objective"])
print(df.head())
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
