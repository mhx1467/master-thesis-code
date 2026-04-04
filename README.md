# HSI Compression

Master thesis codebase for hyperspectral image compression experiments on HySpecNet-11k.

## Setup

```bash
pip install -e .
```

## Example usage

```bash
python scripts/train.py --config configs/baseline_2d/baseline_2d_ae.yaml
```

Training and evaluation use a single benchmark-aligned HySpecNet-11k pipeline:
- split files are consumed as published by HySpecNet-11k
- benchmark train/eval inputs are loaded only from `patches/...-DATA.npy`
- if you only have `*-SPECTRAL_IMAGE.TIF`, convert them to `*-DATA.npy` before any benchmark run

Evaluation reports:
- reference metrics for comparison with HySpecNet: `PSNR`, `SSIM`, `SA`, `bpppc`
- additional diagnostics for local analysis: masked metrics and actual bitstream metrics

## Dataset Preparation

After downloading and extracting HySpecNet-11k, point the project to the dataset root that contains at least:

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

Expected split entries are relative paths inside `patches/`, exactly as provided by HySpecNet-11k, for example
`TILE/PATCH/PATCH-DATA.npy`.

Set the dataset path:

```bash
export DATASET_ROOT=/path/to/hyspecnet-11k-full
```

If the downloaded dataset already contains `patches/...-DATA.npy`, no extra preprocessing is needed.

If the dataset contains only `...-SPECTRAL_IMAGE.TIF`, generate benchmark-compatible `DATA.npy` files:

```bash
python scripts/convert_tif_to_npy.py $DATASET_ROOT --workers 8 --verify
```

This conversion applies the HySpecNet reference preprocessing:
- remove the 22 invalid water-vapor bands
- clip values to `0..10000`
- normalize to `[0,1]`
- save `float32` arrays in `(202, 128, 128)` format

Quick checks:

```bash
find $DATASET_ROOT/patches -type f -name '*-DATA.npy' | head
find $DATASET_ROOT/splits/easy -maxdepth 1 -type f
```

You can then train or evaluate directly:

```bash
python scripts/train.py --config configs/baseline_2d/baseline_2d_ae.yaml --dataset-root $DATASET_ROOT
python scripts/evaluate.py artifacts/checkpoints/<checkpoint>.pt $DATASET_ROOT --split test --save-json
```

## VM Management

Create/update VM definitions in `configs/misc/vms.yaml`.

Each VM entry supports:
- `name`
- `host`
- `ssh_key_path`
- `port` (optional, defaults to `22`)
- `user` (optional)
- `dataset_path` (optional, used by `copy-dataset`)
- `remote_project_dir` (optional, used by `prepare-environment`)

Run commands via the CLI:

```bash
hsi-vm list
hsi-vm show gpu-a100
hsi-vm ssh gpu-a100
hsi-vm run gpu-a100 copy-dataset
hsi-vm run gpu-a100 prepare-environment --python-version 3.10
```

The predefined command scripts are now under `scripts/commands/`:
- `scripts/commands/copy-data-set-to-vm.sh`
- `scripts/commands/prepare-environment-on-vm.sh`

Compatibility wrappers remain at:
- `scripts/copy-data-set-to-vm.sh`
- `scripts/prepare-environment-on-vm.sh`

##### Evaluation `val`
```bash
python scripts/evaluate.py artifacts/checkpoints/baseline_2d_ae_easy_train256_val64_latent16_masked_mse.pt \
  $DATASET_ROOT \
  --split val \
  --save-json
```

##### Evaluation `test`
```bash
python scripts/evaluate.py artifacts/checkpoints/baseline_2d_ae_easy_train256_val64_latent16_masked_mse.pt \
  $DATASET_ROOT \
  --split test \
  --save-json
```

##### Evaluation debug
```bash
python scripts/evaluate.py artifacts/checkpoints/baseline_2d_ae_easy_train256_val64_latent16_masked_mse.pt \
  $DATASET_ROOT \
  --split val \
  --subset-size 32 \
  --disable-wandb
```
