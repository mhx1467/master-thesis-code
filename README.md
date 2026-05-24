# HSI Compression

PyTorch research codebase for hyperspectral image compression experiments.
The repository provides data loading, model definitions, training, evaluation,
and utility scripts used to run experiments on HySpecNet-11k and related HSI
workflows.

## Repository Layout

```text
src/hsi_compression/     package code: data, models, metrics, training
configs/                 YAML experiment configurations
scripts/                 core training, evaluation, conversion, and analysis scripts
tests/                   tests for data, models, metrics, and protocols
docs/                    technical notes and protocol documentation
summaries/               experiment summaries
artifacts/               local outputs, checkpoints, and analyses
```

`artifacts/`, `dataset/`, and `wandb/` are local runtime directories and should
not be treated as source files.

## Requirements

- Python `3.10+`
- PyTorch
- HySpecNet-11k for the main benchmark workflow

Optional dependency groups:

- `mamba` for Mamba-based models,
- `lossless` for lossless codec experiments,
- `dev` for formatting and linting tools.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

Install optional extras when needed:

```bash
pip install -e '.[mamba,dev]'
pip install -e '.[lossless,dev]'
```

## Data

Set the dataset root:

```bash
export DATASET_ROOT=/path/to/hyspecnet-11k
```

Expected HySpecNet-11k layout:

```text
<DATASET_ROOT>/
  patches/
  splits/
    easy/
      train.csv
      val.csv
      test.csv
    hard/
      train.csv
      val.csv
      test.csv
```

Benchmark runs expect `patches/...-DATA.npy` files with shape
`(202, 128, 128)`, dtype `float32`, and values normalized to `[0, 1]`.

If the dataset contains only `*-SPECTRAL_IMAGE.TIF`, convert it first:

```bash
python scripts/convert_tif_to_npy.py "$DATASET_ROOT" --workers 8 --verify
```

## Training

```bash
python scripts/train.py \
  --config configs/mamba/hierarchical_spectral_mamba_ae_latent96.yaml \
  --dataset-root "$DATASET_ROOT"
```

Useful options:

```bash
--disable-wandb
--resume
--pretrained <checkpoint>
--override-rd-lambda <value>
--override-lr <value>
--override-epochs <value>
--override-experiment-name <name>
```

## Evaluation

```bash
python scripts/evaluate.py \
  artifacts/checkpoints/<checkpoint>.pt \
  "$DATASET_ROOT" \
  --split test \
  --difficulty easy \
  --save-json
```

Debug on a small subset:

```bash
python scripts/evaluate.py \
  artifacts/checkpoints/<checkpoint>.pt \
  "$DATASET_ROOT" \
  --split val \
  --difficulty easy \
  --subset-size 32 \
  --disable-wandb
```

## Common Scripts

```text
scripts/train.py                                  train compression models
scripts/evaluate.py                               evaluate checkpoints
scripts/convert_tif_to_npy.py                     prepare HySpecNet DATA.npy files
scripts/export_reconstruction_qualitative.py      export qualitative reconstructions
scripts/evaluate_lossless_codecs.py               evaluate lossless codecs
scripts/audit_lossless_tcn_protocol.py            audit lossless TCN protocol
```

## Code Quality

```bash
make lint
make check-format
pytest
```

Apply formatting:

```bash
make format
```

## Notes

- Keep benchmark runs on the official HySpecNet-11k split files.
- Do not mix generated artifacts, checkpoints, datasets, or W&B outputs into
  ordinary code changes.
- Record the config, checkpoint, dataset split, and git commit for reported
  experiment results.
