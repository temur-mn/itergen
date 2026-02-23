# Contributing to itergen

Thanks for contributing.

## Development setup

```bash
git clone <repo-url>
cd itergen
pip install -e .[dev,docs]
```

## Local checks

Run these before opening a PR:

```bash
ruff check .
ruff format --check .
mypy src/itergen
python -m sphinx -W -b html docs/source docs/build/html
python -m build
python -m twine check dist/*
```

Release process is documented in `docs/source/release.md`.

## Branch and PR guidelines

- Keep PRs focused and reviewable.
- Use descriptive commit messages.
- Update docs with behavior changes.
- Avoid unrelated formatting churn.

## Pull request checklist

- [ ] Documentation updated where relevant
- [ ] CI passes on all required jobs
- [ ] Changelog updated for user-facing changes
