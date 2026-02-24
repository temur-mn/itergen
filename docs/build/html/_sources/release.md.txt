# Release Guide

This page is the canonical release checklist for both GitHub and PyPI.

## 1) Prepare release changes

1. Update version in `pyproject.toml`.
2. Move user-facing notes from `## [Unreleased]` in `CHANGELOG.md` into the new
   version section.
3. Ensure docs reflect any API/config behavior changes.

## 2) Run full quality gate

```bash
pip install -e .[dev,torch]
ruff check .
ruff format --check .
mypy src/itergen
python -m build
python -m twine check dist/*
```

## 3) Publish to GitHub

1. Commit and merge release changes.
2. Create a tag: `v<version>`.
3. Push tag to trigger release artifact workflow.
4. Create a GitHub Release and paste changelog notes.

## 4) Publish to PyPI

Use trusted publishing if configured, or upload manually:

```bash
python -m twine upload dist/*
```

After publishing, verify:

- package installs from PyPI (`pip install itergen==<version>`)
- import-first runtime works (`python sample_run.py`)
- docs and changelog links in `pyproject.toml` are correct
