# Hyperprior Preflight Review

Date: 2026-05-26

## Verdict

The `hierarchical_spectral_mamba_hyperprior` variant is code-ready, but it should be trained with
two safeguards before treating it as a comparable experiment:

- CompressAI entropy bottleneck `*.quantiles` must be optimized only by the auxiliary optimizer.
- Hyperprior runs should be warm-started from the current K=4 spatial Mamba checkpoint using
  compatible-key loading, because the research question is the entropy model change, not a full
  retrain from random initialization.

## Findings

### Architecture

The hyperprior model keeps the existing hierarchical spectral Mamba encoder, CNN spatial
conditioning, and decoder unchanged. It replaces the flat latent `EntropyBottleneck` with:

- `hyper_encoder(z) -> y`,
- `EntropyBottleneck(y)`,
- `hyper_decoder(y_hat) -> means/scales`,
- `GaussianConditional(z | means, scales)`.

This matches the intended ablation: current Mamba + better entropy model.

### Rate reporting

The model returns combined likelihoods from the main latent and hyperlatent. The current
rate-distortion loss and evaluator compute likelihood bpppc by summing `-log2(likelihoods)` and
dividing by the original cube values. This is appropriate for comparing likelihood bpppc against the
existing entropy-bottleneck Mamba.

Actual bitstream metrics still need checkpoint evaluation after training.

### Required adjustment

The previous training script put entropy bottleneck quantile parameters into both the main optimizer
and the aux optimizer. This is not the standard CompressAI setup and can perturb entropy-model CDF
calibration. The training script now splits main and aux parameters into disjoint groups.

### Experiment alignment

The original hyperprior config only covered RD `lambda=0.01`. Recent useful Mamba comparisons also
use the `lambda=0.001` region around the ~0.03 actual bpppc point, so a matching hyperprior config
was added.

## Recommended first run

Warm-start hyperprior from the current K=4 spatial Mamba checkpoint:

```bash
python3 scripts/train.py \
  --config configs/mamba/hierarchical_spectral_mamba_hyperprior_rd_lambda_0_001.yaml \
  --dataset-root "$DATASET_ROOT" \
  --pretrained artifacts/checkpoints/hierarchical_spectral_mamba_ae_k4_spatial_rd_lambda_0_001_ft_best.pt \
  --pretrained-compatible
```

If the goal is a pure entropy-model ablation without spectral-feature loss influence, use the
closest masked-MSE K=4 spatial checkpoint instead.

## Decision Criteria

Keep the hyperprior variant only if it improves at least one of:

- actual bpppc at similar PSNR/SAM,
- likelihood bpppc calibration against actual bpppc,
- actual compression ratio without downstream regression degradation.
