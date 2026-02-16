# API Reference

## Top-level convenience imports

The package root re-exports the most common runtime entry points:

- `RunConfig`
- `GenerateResult`
- `TorchControllerConfig`
- `VorongenSynthesizer`
- `generate`
- `compare_torch_vs_classic`
- `get_sample_config`
- `available_sample_configs`

## Runtime models

```{eval-rst}
.. automodule:: vorongen.api.models
   :members:
```

## Runtime orchestration

```{eval-rst}
.. automodule:: vorongen.api.synthesizer
   :members:
```

## Built-in sample configs

```{eval-rst}
.. automodule:: vorongen.schema.samples
   :members:
```

Legacy flat-module imports (for example `vorongen.models` and
`vorongen.sample_configs`) have been removed. Use canonical module paths such as
`vorongen.api.models` and `vorongen.schema.samples`.
