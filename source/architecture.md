# Architecture

## Editable diagram

- Draw.io source: [vorongen-architecture.drawio](diagrams/vorongen-architecture.drawio)

## Runtime flow

1. Load config (`dict`, YAML text, or YAML file).
2. Resolve missing references (`error`, `skip`, or interactive `prompt`).
3. Validate schema + feasibility before generation.
4. Generate initial dataframe proposal.
5. Run iterative optimization batches.
6. Evaluate equilibrium rules and retry attempts when needed.
7. Save output, metrics, and quality report.

## Main packages

- `vorongen.api`: CLI entrypoints (`cli`) plus high-level runtime facade (`synthesizer`) and public dataclasses (`models`).
- `vorongen.schema`: config parsing/validation (`config`), defaults (`defaults`), and built-in samples (`samples`).
- `vorongen.engine`: initial generation (`initial`), retry loop (`generation`), optimizer (`optimizer`), and proposal helpers (`adjustments`).
- `vorongen.scoring`: condition matching (`conditions`) and equilibrium/objective reporting (`metrics`).
- `vorongen.controllers`: classic controller (`classic`) and torch-backed controller (`torch`).
- `vorongen.runtime`: deterministic RNG helpers (`rng`) and run logging (`logging_utils`).
- Legacy flat-module imports (for example `vorongen.cli`, `vorongen.config`) have
  been removed; import from canonical packages such as `vorongen.api` and
  `vorongen.schema`.

## Controller backends

- `classic` (default): lightweight penalty controller with no torch dependency.
- `torch` (opt-in): torch-backed adaptive controller for multiplier updates.
- When torch is requested but unavailable and not required, runtime falls back to
  classic mode and records a runtime note.
