# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog and this project follows Semantic
Versioning principles.

## [Unreleased]

### Added

- `sample_run.py` helper for quick local smoke execution.
- Sphinx docs structure with quickstart, configuration, API, architecture, and
  troubleshooting pages.
- CI docs gate and repository governance templates/policies.
- Release guide for synchronized GitHub + PyPI publishing (`source/release.md`).
- Draw.io architecture diagram (`source/diagrams/vorongen-architecture.drawio`).

### Changed

- Package runtime is API-only; CLI entrypoints and `vorongen.api.cli` are removed.
- Removed legacy flat-module imports (for example `vorongen.config`,
  `vorongen.metrics`) in favor of canonical package paths.
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
