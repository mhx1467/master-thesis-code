# Development Guide

This repository is thesis research code, but it should stay easy to change by hand.
Prefer small modules, explicit experiment configs, and reusable helpers over copying
logic between scripts or notebooks.

## Daily Workflow

```bash
python3 -m venv .venv
source .venv/bin/activate
make install-dev
make quality
```

Use `make lint` for a fast style check and `make test` when changing behavior.

## Commit Messages

Use the repository convention:

```text
type(scope): short imperative summary
```

Common types are `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, and `perf`.
Keep scopes short and concrete, for example `mamba`, `lossless`, `hyperview2`,
`notebook`, `config`, or `tooling`.

## Code Organization

- Put reusable model I/O behavior in `hsi_compression.engine.model_io`.
- Put dataset and benchmark protocol logic in package modules, not notebooks.
- Keep scripts as thin command-line entry points around package functions.
- Keep generated outputs under `artifacts/`, `summaries/`, `wandb/`, or a local data
  directory that is ignored by Git.
- Add or update tests when a helper becomes shared by more than one script.

## Notebook Rules

Colab notebooks should be orchestration layers, not the only implementation of an
experiment. Shared setup, preprocessing, model loading, metric calculation, and
serialization should live in package modules or small helper scripts. A notebook cell
should usually configure paths or call one clearly named function.
