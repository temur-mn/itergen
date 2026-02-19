# Quickstart

## Installation

```bash
pip install itergen
```

From source (development):

```bash
git clone <repo-url>
cd itergen
pip install -e .
```

Optional extras from source:

```bash
pip install -e .[dev]
pip install -e .[docs]
pip install -e .[torch]
```

## Programmatic usage

Use built-in sample configs:

```python
from itergen import RunConfig, ItergenSynthesizer, get_sample_config

config = get_sample_config("mixed")
run_cfg = RunConfig(n_rows=1200, seed=101, tolerance=0.04, max_attempts=2)

result = ItergenSynthesizer(config, run_cfg).generate()
print(result.success, result.output_path)
```

To run in-memory only (skip workbook write):

```python
result = ItergenSynthesizer(
    config,
    RunConfig(n_rows=1200, save_output=False),
).generate()
print(result.output_path)  # None
```

Load from YAML text:

```python
from pathlib import Path

from itergen import RunConfig, ItergenSynthesizer, load_config

yaml_text = Path("path/to/config.yaml").read_text(encoding="utf-8")
config = load_config(yaml_text)
result = ItergenSynthesizer(config, RunConfig(n_rows=1200)).generate()
print(result.metrics["objective"])
```

Default output path is `output/<timestamp>_itergen.xlsx`.

## Quick local smoke run

Use the project helper script:

```bash
python sample_run.py
```

## Next docs

- Feature guide: `features.md`
- Configuration details: `configuration.md`
- API reference: `api.md`
- Troubleshooting: `troubleshooting.md`
