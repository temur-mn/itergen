# vorongen

`vorongen` is a config-driven synthetic tabular data generator for dependency-aware
workflows with binary, categorical, and continuous columns.

It supports:

- YAML or Python-dict configurations
- dependency-aware conditional distributions
- quality metrics and best-effort retry loops
- optional torch-backed controller optimization
- import-first Python API

## Installation

From PyPI:

```bash
pip install vorongen
```

From source:

```bash
git clone <repo-url>
cd vorongen
pip install -e .
```

Optional source extras:

```bash
pip install -e .[dev]    # lint, type-checking, tests
pip install -e .[docs]   # Sphinx docs tooling
pip install -e .[torch]  # torch controller backend
```

## Runtime mode

`vorongen` is API-only. No CLI entrypoint is shipped.

Use `python sample_run.py` for a quick local smoke run.

## Quick Python API usage

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
```

Default output path is `output/<timestamp>_vorongen.xlsx`.

## Documentation (Sphinx)

Primary docs are maintained in Sphinx under `source/`.

- Quickstart: `source/quickstart.md`
- Configuration: `source/configuration.md`
- API reference: `source/api.md`
- Architecture: `source/architecture.md`
- Editable architecture diagram: `source/diagrams/vorongen-architecture.drawio`
- Troubleshooting: `source/troubleshooting.md`
- Release guide (Git + PyPI): `source/release.md`

Build docs locally:

```bash
python -m sphinx -W -b html source build/html
```

## Quality gate

```bash
ruff check .
ruff format --check .
mypy src/vorongen
PYTHONPATH=src python -m unittest discover -s tests -p "test*.py" -q
coverage run -m unittest discover -s tests -p "test*.py"
coverage report --fail-under=85
python -m build
python -m twine check dist/*
```

## Project standards

- Contribution guide: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Changelog: `CHANGELOG.md`
