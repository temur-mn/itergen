# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog and this project follows Semantic
Versioning principles.

## [Unreleased]


### Added

- Torch-backed controller execution path with classic-controller fallback handling.
- Expanded Sphinx documentation set (quickstart, features, configuration, API, architecture, troubleshooting, release).
- Repository governance and maintenance templates (contribution, security, issue/PR templates, dependabot).
- Release-artifact workflow for building and validating distributions on tagged releases.

### Changed

- Standardized project references and quality-gate commands around `itergen` (`src/itergen`, `docs/source`).
- Aligned CI/docs workflow build paths and docs deployment output for GitHub Actions.
- Refined README presentation with project logo and documentation link.
- Improved attempt-worker scheduling with bounded in-flight submission while preserving deterministic attempt-order selection.
- Clarified worker/parallelism behavior in configuration, features, and troubleshooting docs.

### Fixed

- Corrected GitHub issue-template security advisory contact URL.
- Removed stale placeholder entries and documented current release status in this changelog.

## [0.1.0] - 2026-02-16

### Added

- Initial import-first Python API (`RunConfig`, synthesizer facade, generate helpers).
- Config-driven synthetic tabular generation for binary, categorical, and continuous columns.
- Core optimization and scoring loop with retry attempts and quality reporting.
- Built-in sample configuration templates and local sample-run workflow.
