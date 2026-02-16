# Architecture

## Runtime flow

1. Load config (`dict`, YAML text, or YAML file).
2. Resolve missing references (`error`, `skip`, or interactive `prompt`).
3. Validate schema + feasibility before generation.
4. Generate initial dataframe proposal.
5. Run iterative optimization batches.
6. Evaluate equilibrium rules and retry attempts when needed.
7. Save output, metrics, and quality report.

## Feature lanes

- **Schema lane**: parse conditions, normalize targets, build column specs and domains.
- **Generation lane**: initialize rows, optimize proposals, retry across attempts.
- **Scoring lane**: evaluate marginal, conditional, continuous bin, and bounds metrics.
- **Control lane**: adapt per-column multipliers through classic or torch controller.
- **Observability lane**: emit run logs, final artifacts, and runtime notes.

## Main packages

- `vorongen.api`: high-level runtime facade (`synthesizer`) and public dataclasses (`models`).
- `vorongen.schema`: config parsing/validation (`config`), defaults (`defaults`), and built-in samples (`samples`).
- `vorongen.engine`: initial generation (`initial`), retry loop (`generation`), optimizer (`optimizer`), and proposal helpers (`adjustments`).
- `vorongen.scoring`: condition matching (`conditions`) and equilibrium/objective reporting (`metrics`).
- `vorongen.controllers`: classic controller (`classic`) and torch-backed controller (`torch`).
- `vorongen.runtime`: deterministic RNG helpers (`rng`) and run logging (`logging_utils`).
- Legacy flat-module imports (for example `vorongen.config`, `vorongen.metrics`)
  have been removed; import from canonical packages such as `vorongen.api`,
  `vorongen.schema`, `vorongen.engine`, and `vorongen.scoring`.

## Execution modes

- Attempt loop can run sequentially or in process-level parallel mode.
- Optimizer proposal scoring can run in `incremental` mode (patch-based objective
  updates) or `full` mode (complete recomputation per proposal).
- Small-group behavior can be switched to ignore, downweight, or lock-sensitive
  policies based on workload requirements.

## Controller backends

- `classic` (default): lightweight penalty controller with no torch dependency.
- `torch` (opt-in): torch-backed adaptive controller for multiplier updates.
- When torch is requested but unavailable and not required, runtime falls back to
  classic mode and records a runtime note.

## Output artifacts

- Generated dataset workbook (`output/<timestamp>_vorongen.xlsx` by default).
- Per-run log file under configured `log_dir` (default `src/itergen/logs/`).
- Structured result payload containing metrics, quality report, and optional
  iteration history for diagnostics.
