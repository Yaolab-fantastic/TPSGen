# Contributing to TomatoPromoterDesigner

## Scope

This repository is meant to evolve from a research aggregation workspace into a reproducible software package for tissue-biased tomato promoter design. Contributions should move the codebase toward one of these goals:

- clearer and more stable package interfaces
- reproducible data preprocessing
- better model integration boundaries
- stronger tests, examples, and documentation
- release readiness for a future `Bioinformatics` `Application Note`

## Working principles

- keep the installable package under `src/tomato_promoter_designer/`
- prefer small, reviewable pull requests
- avoid committing raw large datasets or opaque binary artifacts without documentation
- update `models/weights_manifest.json` whenever model files or checkpoint locations change
- update `docs/reproducibility.md` when workflow assumptions change

## Suggested development workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make test
make demo
```

## Repository conventions

- `io/`: file contracts, parsers, schema validation
- `preprocessing/`: data extraction and feature preparation
- `models/`: trained-model wrappers and deterministic placeholders
- `pipeline/`: user-facing composition logic
- `evaluation/`: offline metrics and comparison utilities
- `docs/`: paper-facing and release-facing documentation

## Before opening a release

- all tests pass with `make test`
- CLI examples run with `make demo`
- `README.md` reflects the current command set
- `CITATION.cff` reflects the current software and author metadata
- `docs/release_checklist.md` is reviewed end to end
