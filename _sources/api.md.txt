# API Reference

## Top-level convenience imports

The package root re-exports the most common runtime entry points:

- `RunConfig`
- `GenerateResult`
- `TorchControllerConfig`
- `ItergenSynthesizer`
- `generate`
- `compare_torch_vs_classic`
- `get_sample_config`
- `available_sample_configs`

## Runtime models

```{eval-rst}
.. automodule:: itergen.api.models
   :members:
```

## Runtime orchestration

```{eval-rst}
.. automodule:: itergen.api.synthesizer
   :members:
```

## Built-in sample configs

```{eval-rst}
.. automodule:: itergen.schema.samples
   :members:
```

Legacy flat-module imports (for example `itergen.models` and
`itergen.sample_configs`) have been removed. Use canonical module paths such as
`itergen.api.models` and `itergen.schema.samples`.
