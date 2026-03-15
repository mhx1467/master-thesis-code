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