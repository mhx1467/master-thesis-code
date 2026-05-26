# HYPERVIEW2 Downstream Diagnostics For Mamba

Date: 2026-05-26

## Scope

This note summarizes the current downstream-regression diagnostics for reconstructed HYPERVIEW2
cubes produced by the HySpecNet-trained Mamba codec.

This is a diagnostic transfer experiment, not a HySpecNet-11k reference-comparable compression
result. The goal is to understand whether reconstructed hyperspectral cubes preserve information
useful for downstream chemical-property regression.

## Inputs

Diagnostic CSV files:

```text
artifacts/analysis/hyperview2_diagnostics_drive/
```

Main corrected reconstruction source:

```text
hyperview2_mamba_latent48_input_percentile
```

The older source without the `_input_percentile` suffix is retained only as a preprocessing-control
artifact. It showed a large scale mismatch and should not be used for the main downstream
conclusions.

## Evaluation Modes

```text
original_train_to_original_val
```

Train the downstream regressor on original HYPERVIEW2 train features and validate on original
validation features. This is the downstream baseline.

```text
original_train_to_recon_val
```

Train the regressor on original features and validate on reconstructed validation features. This
measures deployment-time domain shift caused by compression.

```text
recon_train_to_recon_val
```

Train and validate the regressor on reconstructed features. This shows how much of the loss can be
recovered if downstream models are retrained on compressed-domain data.

## Best Aggregate Scores

Lower Hyperview score is better.

| source | mode | best model | score | mean MSE | mean MAE |
|---|---|---|---:|---:|---:|
| original | original_train_to_original_val | hist_gradient_boosting | 0.536170 | 341.463226 | 8.284453 |
| Mamba corrected | original_train_to_recon_val | extra_trees | 0.930558 | 596.633301 | 11.494325 |
| Mamba corrected | recon_train_to_recon_val | random_forest | 0.791081 | 488.764008 | 9.952694 |

Interpretation:

- compression causes a large downstream-domain shift: `0.536 -> 0.931`;
- retraining the regressor on reconstructed data recovers part of the gap: `0.931 -> 0.791`;
- a substantial gap remains versus original data, so the issue is not only regressor-domain mismatch.

## Per-Target Degradation

Best relative MSE per target, using the best available regressor in each mode.

| target | original rel. MSE | original->recon rel. MSE | recon->recon rel. MSE | original->recon gap | recon->recon gap |
|---|---:|---:|---:|---:|---:|
| B | 0.378019 | 0.956798 | 0.725202 | +0.578779 | +0.347182 |
| Cu | 0.617004 | 1.014401 | 0.952973 | +0.397397 | +0.335969 |
| Fe | 0.497747 | 0.916446 | 0.733860 | +0.418699 | +0.236113 |
| Mn | 0.568022 | 0.839933 | 0.709630 | +0.271912 | +0.141609 |
| S | 0.613229 | 0.892444 | 0.795402 | +0.279215 | +0.182173 |
| Zn | 0.526217 | 0.964517 | 0.811395 | +0.438300 | +0.285178 |

The most affected targets are `B`, `Zn`, `Fe`, and `Cu`. `Mn` is the least affected, although it
still degrades.

## Prediction Shift

For `extra_trees` in `original_train_to_recon_val`, correlation between predictions on original
features and reconstructed features:

| target | prediction corr. original vs reconstructed |
|---|---:|
| Cu | 0.145647 |
| B | 0.400761 |
| Zn | 0.543663 |
| S | 0.623844 |
| Fe | 0.643348 |
| Mn | 0.648740 |

`Cu` is nearly decorrelated, and `B` is weak. This is stronger evidence than aggregate MSE that
the reconstruction changes downstream-relevant spectral features, not merely pixel-level noise.

## Spectral Reconstruction Diagnostics

Corrected Mamba reconstruction, per-band statistics:

| statistic | MAE | RMSE | bias |
|---|---:|---:|---:|
| mean | 0.038482 | 0.054138 | -0.006806 |
| median | 0.039346 | 0.055035 | -0.006236 |
| 95th percentile | 0.060853 | 0.084728 | 0.009808 |
| max | 0.096159 | 0.140085 | 0.029398 |

Worst bands by RMSE are concentrated around band indices `160-180`. The worst observed band is:

| band | original mean | reconstruction mean | bias | MAE | RMSE |
|---:|---:|---:|---:|---:|---:|
| 162 | 0.649447 | 0.601934 | -0.047513 | 0.096159 | 0.140085 |

Band-range summary:

| band range | mean MAE | mean RMSE | mean bias |
|---|---:|---:|---:|
| 0-9 | 0.049474 | 0.070462 | +0.007199 |
| 10-29 | 0.041455 | 0.060802 | +0.000893 |
| 30-49 | 0.035933 | 0.049769 | +0.002537 |
| 50-79 | 0.031060 | 0.042952 | +0.004308 |
| 80-109 | 0.025472 | 0.038269 | +0.000389 |
| 110-139 | 0.041617 | 0.057694 | -0.012723 |
| 140-169 | 0.036308 | 0.052528 | -0.015986 |
| 170-199 | 0.050262 | 0.069550 | -0.016503 |
| 200-229 | 0.042229 | 0.056860 | -0.016353 |

The model is systematically low-biased in much of the higher spectral range. This is likely more
important for downstream regression than the global PSNR alone suggests.

## Feature Drift

Corrected Mamba feature drift versus the older preprocessing-control run:

| source | mean feature MAE | median feature MAE | mean feature RMSE | median feature RMSE | mean max abs |
|---|---:|---:|---:|---:|---:|
| old Mamba preprocessing control | 0.071103 | 0.080000 | 0.148427 | 0.171217 | 0.690391 |
| corrected input-percentile Mamba | 0.014266 | 0.011760 | 0.025295 | 0.020744 | 0.144989 |

The corrected preprocessing reduced feature drift by roughly `5x-6x`, but downstream degradation
remains. This supports the conclusion that the issue is not just a normalization bug.

## Sample-Level Relationship

Spearman correlation between sample-level downstream error and reconstruction/feature error:

| downstream model | abs downstream error vs feature MAE | abs downstream error vs cube RMSE |
|---|---:|---:|
| extra_trees | 0.1613 | 0.1666 |
| random_forest | 0.2230 | 0.2293 |
| hist_gradient_boosting | 0.1718 | 0.1748 |
| ridge | 0.4942 | 0.4738 |

For tree models, the relationship is weak. This suggests that average reconstruction error per
sample is not sufficient to explain downstream failure. The reconstruction likely perturbs
specific spectral cues used by the regressors.

## Current Interpretation

The corrected Mamba codec preserves enough information to maintain high compression metrics on the
source compression task, but it does not yet preserve downstream-critical spectral shape on
HYPERVIEW2.

The dominant failure mode appears to be spectral-feature drift:

- significant prediction shift for `B`, `Cu`, `Zn`, and `Fe`;
- low prediction correlation for `Cu` and `B`;
- systematic per-band bias in high spectral bands;
- weak relationship between global sample RMSE and downstream error for tree regressors.

This means improving only PSNR is unlikely to solve the downstream problem.

## Recommended Model Direction

The next Mamba improvement should prioritize spectral/feature preservation in the training
objective. The current spectral-feature RD fine-tuning experiment is aligned with this diagnosis.

Primary metrics to compare after the new checkpoint finishes:

- `original_train_to_recon_val` Hyperview score;
- `recon_train_to_recon_val` Hyperview score;
- relative MSE for `B`, `Cu`, `Zn`, and `Fe`;
- prediction correlation original vs reconstructed, especially for `Cu` and `B`;
- per-band RMSE and bias around bands `160-180`;
- SAM / spectral-angle metrics on HySpecNet-11k to ensure compression-side spectral fidelity also
  improves.

## Caveats

- These diagnostics use HYPERVIEW2 transfer from a HySpecNet-trained codec and are not
  reference-comparable HySpecNet-11k compression results.
- The analysis depends on the current HYPERVIEW2 feature extraction and masking path.
- Four samples in `feature_drift_by_sample.csv` have zero feature drift while their cube-level
  sample errors are non-zero. This should be audited before making strong sample-level claims.
- The older non-`input_percentile` Mamba source should remain a preprocessing-control artifact only.
