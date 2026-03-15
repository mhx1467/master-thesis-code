# HSI Compression

Master thesis codebase for hyperspectral image compression experiments on HySpecNet-11k.

## Setup

```bash
pip install -e .
```

## Example usage

```bash
python scripts/train.py --config configs/baseline_2d_ae.yaml
```

## VM Management

Create/update VM definitions in `configs/vms.yaml`.

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