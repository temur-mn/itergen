# vorongen

`vorongen` is an import-first synthetic data generation package with configurable
binary, categorical, and continuous dependency modeling.

## Install

```bash
pip install -e .
```

Install with PyTorch support for adaptive optimization control:

```bash
pip install -e .[torch]
```

## Minimal usage

```python
from vorongen import RunConfig, VorongenSynthesizer, get_sample_config

config = get_sample_config("mixed")
run_cfg = RunConfig(n_rows=3000, tolerance=0.04, log_level="quiet")

result = VorongenSynthesizer(config, run_cfg).generate()
print(result.success, result.objective())
print(result.dataframe.head())
```

## Notebooks

- `notebooks/testing_new_tools.ipynb`
- `notebooks/notes.ipynb`
