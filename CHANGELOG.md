# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog and this project follows Semantic
Versioning principles.

## [Unreleased]

### Added

- Guidance-first CLI entrypoint with explicit sample/config run modes.
- `sample_run.py` helper for quick local smoke execution.
- Sphinx docs structure with quickstart, configuration, API, architecture, and
  troubleshooting pages.
- CI docs gate and repository governance templates/policies.
- Config-only CLI validation mode (`--validate-config`) with feasibility checks.
- Torch controller tuning CLI flags (`--torch-lr`, `--torch-hidden-dim`,
  `--torch-weight-decay`, `--torch-device`).
- Release guide for synchronized GitHub + PyPI publishing (`source/release.md`).
- Draw.io architecture diagram (`source/diagrams/vorongen-architecture.drawio`).

### Changed

- Module and console entrypoints now live under `vorongen.api` (`vorongen.api.cli`,
  `vorongen.api.synthesizer`) with compatibility wrappers kept for legacy imports.
- Documentation build setup upgraded to modern Sphinx extensions and theme.
- Torch-requested runs now use a real torch-backed controller backend when
  available, with classic fallback when optional.
- API docs now use direct module autodoc pages instead of generated autosummary
  files tracked in git.
- Repository cleanup for release readiness (removed scratch notebooks, legacy
  root wrapper file, and low-value note stubs).

## [0.1.0] - 2026-02-10

### Added

- Initial import-first runtime API.
- Config-driven synthetic generation engine and benchmark scripts.
