# Spatial Mamba Literature Decision

Date: 2026-05-24

## Decision

Do not implement a full spatial/window Mamba branch as the first extension of the current Mamba
codec.

Proceed first with a hyperprior extension. Keep spatial Mamba as a small, budget-capped pilot only
after the hyperprior baseline is measured. If the pilot does not improve rate-distortion under the
same HySpecNet-11k protocol, reject the spatial Mamba branch and document the result here.

## Context

The target architecture description proposes:

```text
spectral Mamba path + spatial/window Mamba path -> fused latent -> hyperprior -> entropy coding
```

Our current lossy Mamba codec is closer to:

```text
spectral Mamba encoder + CNN spatial conditioning -> EntropyBottleneck -> CNN decoder
```

The main missing components are:

- hyperprior,
- spatial/window Mamba,
- stronger fusion ablations,
- Mamba-based decoder/refinement.

The immediate engineering constraint is a single RTX 4090. Spatial/window Mamba can increase memory
and implementation complexity before we know whether the simpler entropy-model upgrade already
captures the largest gain.

## Literature Findings

### Mamba for generic image classification is disputed

MambaOut argues that Mamba is best suited to long-sequence and autoregressive settings, while
ImageNet-style classification is neither. Its authors report that removing the SSM mixer and using a
gated CNN-style block can outperform visual Mamba models on ImageNet classification, while Mamba
remains more defensible for detection and segmentation where sequences are longer.

Source: Yu and Wang, "MambaOut: Do We Really Need Mamba for Vision?", CVPR 2025.
https://arxiv.org/abs/2405.07992
https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_MambaOut_Do_We_Really_Need_Mamba_for_Vision_CVPR_2025_paper.pdf

Implication for us: HSI compression is not ImageNet classification, but a spatial branch over
128x128 patches is also not inherently autoregressive. A spatial Mamba branch should not be assumed
useful just because it matches the architecture description.

### Naive 1D flattening loses spatial structure

2DMamba explicitly identifies a limitation of visual Mamba variants that flatten 2D images into 1D
sequences: spatial structure is lost in at least one direction, producing what the paper calls a
spatial discrepancy. The paper proposes an intrinsic 2D selective scan to preserve 2D continuity,
which is more complex than simply applying our existing sequence block over rasterized windows.

Source: Zhang et al., "2DMamba: Efficient State Space Model for Image Representation with
Applications on Giga-Pixel Whole Slide Image Classification", CVPR 2025.
https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_2DMamba_Efficient_State_Space_Model_for_Image_Representation_with_Applications_CVPR_2025_paper.pdf

Implication for us: a simple raster-scan spatial Mamba would be a weak implementation of the idea.
If we test spatial Mamba, it should use local/window or 2D-aware scanning, not one global flattened
sequence.

### Local/window scan is a known fix, but not free

LocalMamba states that earlier vision Mamba approaches did not clearly surpass CNNs or ViTs and
that flattening spatial tokens can overlook local 2D dependencies. It proposes windowed selective
scan to preserve local dependencies and reports ImageNet gains over Vim at matched FLOPs.

Source: Huang et al., "LocalMamba: Visual State Space Model with Windowed Selective Scan".
https://arxiv.org/abs/2403.09338

Implication for us: if we do any spatial Mamba experiment, the smallest defensible variant is a
local/window scan at the latent grid, not a global full-resolution scan.

### HSI papers often propose spatial-spectral Mamba, but they are classification papers

S2Mamba proposes separate spatial and spectral SSM paths plus a mixture gate for HSI
classification. This supports the conceptual plausibility of spectral-spatial Mamba in HSI, but the
task is land-cover classification rather than compression.

Source: Wang et al., "S2Mamba: A Spatial-spectral State Space Model for Hyperspectral Image
Classification".
https://arxiv.org/abs/2404.18213

MambaMoE notes that Mamba's sequential modeling can disrupt the original spatial structure of HSI
and cause loss of spatial coherence. It addresses this with directional experts and routing.

Source: "MambaMoE: Mixture-of-spectral-spatial-experts state space model for hyperspectral image
classification".
https://www.sciencedirect.com/science/article/abs/pii/S1566253525008735

Implication for us: HSI classification literature does not justify directly spending the RTX 4090
budget on a large spatial Mamba compression branch. It justifies a controlled ablation if simpler
codec improvements are exhausted.

### Newer spatial Mamba variants can work, but cost matters

Spatial-Mamba reports strong ImageNet, COCO, and ADE20K results and introduces a spatial structure
fusion mechanism. Its ablations also show that more flexible fusion can improve accuracy while
reducing throughput substantially.

Source: "Spatial-Mamba: Effective Visual State Space Models via Structure-Aware State Fusion",
ICLR 2025.
https://openreview.net/pdf/8219b6919d5e23d90a43ddcdec43ba242eac39c4.pdf

Implication for us: spatial Mamba is not categorically bad, but the useful versions are carefully
2D/local-structure-aware. A naive branch is likely to be a poor tradeoff.

## Research Hypothesis

The current Mamba codec is more likely bottlenecked by its simple entropy model than by the lack of
a spatial Mamba branch. A hyperprior should improve bitrate more directly than adding spatial Mamba,
because compression performance depends strongly on the probability model used to code latents.

## Recommended Implementation Order

1. Add a hyperprior variant to the current hierarchical spectral Mamba codec.
2. Run the same RD sweep against the current `EntropyBottleneck` baseline.
3. If hyperprior improves `actual_bpppc` or `likelihood_bpppc`, keep it as the new baseline.
4. Only then test a small spatial/window Mamba pilot at the latent or 64x64 feature grid.
5. Reject spatial Mamba if it fails to improve RD metrics under the same compute budget.

## Spatial Mamba Pilot Constraints

If tested, the pilot must be deliberately small:

- operate at latent-grid or pre-latent resolution, not full 128x128 raw pixels;
- use local windows such as 4x4 or 8x8;
- compare against the existing CNN spatial conditioning path with matched latent channels;
- keep batch size feasible on RTX 4090;
- stop if wall time or memory pressure prevents fair RD sweeps.

## Required Ablations

- Current Mamba + `EntropyBottleneck`.
- Current Mamba + hyperprior.
- Hyperprior + CNN spatial conditioning.
- Hyperprior + small window spatial Mamba.
- Fusion: affine conditioning vs concat + 1x1 projection vs gated fusion.

## Metrics

Reference-comparable:

- PSNR,
- SSIM,
- SA,
- bpppc.

Diagnostics:

- likelihood_bpppc,
- actual_bpppc,
- actual_compression_ratio,
- encode/decode time,
- peak VRAM,
- train throughput.

## Rejection Criteria

Reject the spatial Mamba branch if any of these hold:

- no RD improvement over CNN spatial conditioning at comparable bitrate;
- improvement is smaller than run-to-run noise;
- memory forces much smaller batch/subset settings than the baseline;
- encode/decode latency becomes impractical;
- gains appear only under non-reference-comparable protocol changes.

## Protocol Constraints

All comparisons must preserve:

- HySpecNet-11k official split files;
- normalized `DATA.npy` benchmark inputs unless explicitly running a non-reference pilot;
- identical metric definitions and aggregation;
- same train/val/test difficulty;
- same subset sizes for pilot comparisons;
- same checkpoint selection rule.

## Current Recommendation

Implement hyperprior first. Do not spend full implementation effort on spatial Mamba until the
hyperprior result is known.
