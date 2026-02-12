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

## Main modules

- `vorongen.cli`: CLI argument parsing, validate-only mode, and run orchestration.
- `vorongen.synthesizer`: high-level runtime facade used by CLI and API.
- `vorongen.config`: config validation, dependency resolution, and column specs.
- `vorongen.generation`: attempt loop (`generate_until_valid`) with retry workers.
- `vorongen.optimizer`: proposal/acceptance loop and controller integration.
- `vorongen.metrics`: objective/error computation and quality report output.

## Controller backends

- `classic` (default): lightweight penalty controller with no torch dependency.
- `torch` (opt-in): torch-backed adaptive controller for multiplier updates.
- When torch is requested but unavailable and not required, runtime falls back to
  classic mode and records a runtime note.
