# Architecture

## Generation pipeline

1. Parse and normalize config (`load_config`).
2. Resolve missing dependency references (`resolve_missing_columns`).
3. Validate configuration (`validate_config`).
4. Build initial dataset proposal.
5. Optimize toward marginal/conditional equilibrium targets.
6. Evaluate metrics and rules.
7. Save decoded output and emit quality report.

## Main runtime layers

- `vorongen.synthesizer`: high-level public orchestration.
- `vorongen.generation`: attempt-level retry loop.
- `vorongen.optimizer`: iterative proposal/acceptance process.
- `vorongen.metrics`: objective, equilibrium, and diagnostics.
- `vorongen.config`: schema validation and spec construction.

## Design notes

- Public API stays import-first (`RunConfig`, `VorongenSynthesizer`).
- CLI and script paths are thin wrappers over the same runtime layer.
- Sample configs provide reproducible starting points.
