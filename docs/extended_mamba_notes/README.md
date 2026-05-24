# Extended Mamba Notes

Research notes for extending the current hierarchical spectral Mamba codec toward a fuller
spectral-spatial Mamba + hyperprior design.

This folder is intended for decision records, literature checks, ablation plans, and experiment
summaries. Keep benchmark protocol changes out of these notes unless they are explicitly marked
as non-reference-comparable.

## Current Scope

- Lossy learned HSI compression on HySpecNet-11k.
- Baseline: current hierarchical spectral Mamba codec with CNN spatial conditioning and
  `EntropyBottleneck`.
- Candidate extensions: hyperprior, spatial/window Mamba, fusion variants, and Mamba-based
  decoder refinement.

## Notes

- `spatial_mamba_literature_decision.md`: literature-backed decision on whether to add spatial
  Mamba immediately.
- `roadmap.md`: staged plan for extending the current hierarchical Mamba codec.
