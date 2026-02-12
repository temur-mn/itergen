# Quickstart

## Installation

```bash
pip install vorongen
```

From source (development):

```bash
git clone <repo-url>
cd vorongen
pip install -e .
```

Optional extras from source:

```bash
pip install -e .[dev]
pip install -e .[docs]
pip install -e .[torch]
```

## CLI usage

With no arguments, `python -m vorongen` prints guided next steps instead of
running generation directly.

Run generation from a built-in sample:

```bash
python -m vorongen --sample mixed --rows 1200 --log-level quiet
```

Run generation from a YAML file:

```bash
python -m vorongen --config path/to/config.yaml --rows 1200
```

Validate config and feasibility only (no generation):

```bash
python -m vorongen --config path/to/config.yaml --validate-config
```

Use torch backend when available:

```bash
python -m vorongen --sample mixed --use-torch-controller --torch-device auto
```

List sample configurations:

```bash
python -m vorongen --list-samples
```

Default output path is `output/<timestamp>_vorongen.xlsx`.

## Quick local smoke run

Use the project helper script:

```bash
python sample_run.py
```

## Programmatic usage

```python
from vorongen import RunConfig, VorongenSynthesizer, get_sample_config

config = get_sample_config("mixed")
run_cfg = RunConfig(n_rows=3000, seed=101, tolerance=0.04, max_attempts=2)

result = VorongenSynthesizer(config, run_cfg).generate()
print(result.success, result.objective())
print(result.output_path)
```

## Next docs

- Configuration details: `configuration.md`
- API reference: `api.md`
- Troubleshooting: `troubleshooting.md`
