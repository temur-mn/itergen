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

### Changed

- Module and console entrypoints now route through `vorongen.cli`.
- Documentation build setup upgraded to modern Sphinx extensions and theme.

## [0.1.0] - 2026-02-10

### Added

- Initial import-first runtime API.
- Config-driven synthetic generation engine and benchmark scripts.
