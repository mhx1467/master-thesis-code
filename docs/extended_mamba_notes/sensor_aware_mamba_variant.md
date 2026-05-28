# Sensor-Aware Hierarchical Mamba Variant

## Motivation

The original `hierarchical_spectral_mamba_ae` is strong on the HySpecNet-11k protocol, but its
parameters are tied to the training sensor grid in two important places:

- the spectral position embedding is tied to the grouped 202-band sequence;
- the spatial branch and final decoder contain channel-count-dependent weights.

This is acceptable for reference-comparable HySpecNet experiments, but it is a weak foundation
for zero-shot or light-adaptation transfer to a different sensor such as HYPERVIEW2/PRISMA. The
downstream diagnostics showed systematic spectral shift and band-wise bias rather than only
random reconstruction noise.

## Implemented Variant

The new registry key is:

```text
hierarchical_spectral_mamba_sensor_aware
```

It is implemented next to the old model, so previous checkpoints and results remain
reproducible.

The variant adds:

- wavelength embeddings for every spectral band;
- no learned parameters whose tensor shape depends on the number of input/output bands;
- a wavelength-conditioned sensor adapter initialized as identity;
- a channel-agnostic spatial conditioning path based on per-pixel spectral statistics;
- a dynamic wavelength-conditioned band decoder;
- optional encoder-side spectral augmentation for HySpecNet pretraining.

The intended research claim is not that zero-shot transfer must fully solve HYPERVIEW2. The
claim to test is narrower: HySpecNet pretraining should learn a more transferable HSI prior
when the model is wavelength-aware, and target-domain adaptation should require fewer trainable
parameters than a full retrain.

## Training Configs

Pretraining on HySpecNet:

```bash
python scripts/train.py \
  --config configs/mamba/hierarchical_spectral_mamba_sensor_aware_latent96.yaml \
  --dataset-root "$DATASET_ROOT"
```

RD fine-tuning from the pretraining checkpoint:

```bash
python scripts/train.py \
  --config configs/mamba/hierarchical_spectral_mamba_sensor_aware_rd_lambda_0_0003.yaml \
  --dataset-root "$DATASET_ROOT" \
  --pretrained artifacts/checkpoints/hierarchical_spectral_mamba_sensor_aware_recon_latent96_best.pt
```

## Evaluation Discipline

For HySpecNet-11k, compare against the existing `K=4` Mamba at the same dataset split,
normalization, and actual bitstream protocol.

For HYPERVIEW2, keep the results diagnostic rather than reference-comparable:

- original data;
- lossless passthrough;
- spectral-resample-only control;
- old HySpecNet Mamba zero-shot/fine-tune;
- new sensor-aware Mamba zero-shot/fine-tune;
- adapter-only fine-tune if the notebook/training script supports freezing by parameter name.

The important failure criterion is whether the new model remains substantially worse than the
spectral-resample-only control in the original-trained downstream regressor setting. If that
happens, wavelength awareness alone did not preserve the task-relevant spectral information.
