# itergen

<div align="center">
  <img src="./logo/logo.svg" alt="itergen logo" width="200">
</div>

<p align="center">
  <b>Documentation</b>:
  <a href="https://temur-mn.github.io/itergen/">See</a>
</p>

`itergen` is a config-driven synthetic tabular data generator for dependency-aware
workflows with `binary`, `categorical`, and `continuous` columns.

It supports:

- YAML or Python-dict configurations
- dependency-aware conditional distributions
- quality metrics and best-effort retry loops
- optional torch-backed controller optimization
- import-first Python API

## Installation

From PyPI:

```bash
pip install itergen
```

From source:

```bash
git clone <repo-url>
cd itergen
pip install -e .
```

Optional source extras:

```bash
pip install -e .[dev]    # lint, type-checking, tests
pip install -e .[docs]   # Sphinx docs tooling
pip install -e .[torch]  # torch controller backend
```

## Runtime mode

`itergen` is API-only. No CLI entrypoint is shipped.

Use `python sample_run.py` for a quick local smoke run.

## Quick Python API usage

```python
from itergen import RunConfig, ItergenSynthesizer, get_sample_config

config = get_sample_config("mixed")
run_cfg = RunConfig(
    n_rows=3000,
    seed=101,
    tolerance=0.04,
    max_attempts=2,
    log_level="quiet",
)

result = ItergenSynthesizer(config, run_cfg).generate()
print(result.success, result.objective())
print(result.output_path)
```

For in-memory only runs (no Excel write), set `save_output=False`:

```python
result = ItergenSynthesizer(config, RunConfig(n_rows=3000, save_output=False)).generate()
print(result.dataframe.shape, result.output_path)  # output_path is None
```

Default output path is `output/<timestamp>_itergen.xlsx`.


## Quality gate

```bash
ruff check .
ruff format --check .
mypy src/itergen
PYTHONPATH=src python -m unittest discover -s tests -p "test*.py" -q
python -m build
python -m twine check dist/*
```

## Project standards

- Contribution guide: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Changelog: `CHANGELOG.md`
