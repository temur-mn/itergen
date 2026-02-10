# voronogeen_package

Import-first package API for the `vorongen` synthetic data engine.

## Why this package

- Provides a reusable, class-based interface for colleagues.
- Supports a PyTorch adaptive controller for optimization strength per column.
- Keeps workflow notebook-friendly (no CLI dependency).

## Quick start

Install in editable mode from the repository root:

```bash
pip install -e .
```

Install with PyTorch extras:

```bash
pip install -e .[torch]
```

```python
from voronogeen_package import (
    RunConfig,
    TorchControllerConfig,
    VoronogeenSynthesizer,
    get_sample_config,
)

config = get_sample_config("mixed")
run = RunConfig(
    n_rows=4000,
    tolerance=0.04,
    use_torch_controller=True,
    torch_controller=TorchControllerConfig(lr=2e-3, hidden_dim=48),
)

engine = VoronogeenSynthesizer(config, run)
result = engine.generate()

print(result.success, result.objective())
print(result.dataframe.head())
```

## Notes

- If `torch` is unavailable, the system falls back to the classic controller unless `RunConfig(torch_required=True)` is set.
- Notebook example is available at `voronogeen_package/notebooks/state_of_art_workflow.ipynb`.
