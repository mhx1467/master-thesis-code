# Extended Mamba Roadmap

Date: 2026-05-24

## Goal

Move from the current hierarchical spectral Mamba codec toward the fuller architecture idea:

```text
spectral Mamba path
+ spatial/window Mamba path
-> fused latent
-> hyperprior
-> entropy coding
-> Mamba-aware decoder/refinement
```

Do this incrementally. Each component must earn its place through an ablation against the current
HySpecNet-11k protocol.

## Why We Started With Hierarchical Spectral Mamba

The full target model has too many moving parts to implement first:

- spectral Mamba,
- spatial Mamba,
- fusion,
- hyperprior,
- entropy coding,
- symmetric decoder.

Starting with the hierarchical spectral Mamba isolated the most HSI-specific hypothesis: spectral
bands behave like a structured sequence, and modeling this sequence should improve learned
compression. This gave us a trainable baseline with benchmark-compatible data loading, metrics,
checkpointing, and evaluation before adding more expensive components.

## Current Baseline

Current lossy Mamba codec:

```text
hierarchical spectral Mamba encoder
+ CNN spatial conditioning
-> EntropyBottleneck
-> CNN decoder
```

Strengths:

- explicit spectral sequence modeling;
- learned multi-token spectral summaries;
- mask-aware spatial conditioning;
- real CompressAI entropy coding path;
- benchmark-compatible evaluation.

Known gaps relative to the target description:

- no hyperprior;
- no spatial/window Mamba branch;
- fusion is affine conditioning, not a full two-path fusion study;
- decoder is mostly CNN, not symmetric Mamba.

## Step 0: Freeze Baseline

Record the current hierarchical Mamba results as the baseline.

Required controls:

- same HySpecNet-11k split files;
- same difficulty;
- same normalized input protocol;
- same train/val/test subset policy;
- same metrics and aggregation;
- same checkpoint selection rule.

Metrics:

- PSNR;
- SSIM;
- SA;
- proxy_bpppc;
- likelihood_bpppc;
- actual_bpppc when available;
- encode/decode time;
- peak VRAM.

Deliverable:

- baseline result JSONs and a short note summarizing RD position versus current baselines.

## Step 1: Add Hyperprior

Implement:

```text
z -> hyper_encoder -> z_hyper
z_hyper -> EntropyBottleneck
z_hyper_hat -> hyper_decoder -> distribution parameters for z
z -> conditional entropy model
```

Hypothesis:

The current model is likely bottlenecked by the simple latent entropy model. A hyperprior should
improve bitrate estimation and actual coding efficiency more directly than spatial Mamba.

Variants:

- current `EntropyBottleneck`;
- small hyperprior;
- medium hyperprior.

Success criterion:

- lower `likelihood_bpppc` and/or `actual_bpppc` at comparable PSNR/SSIM/SA;
- no unacceptable encode/decode or VRAM increase.

Failure criterion:

- no RD improvement over the current entropy bottleneck;
- unstable training;
- actual coding does not match likelihood improvements.

## Step 2: Hyperprior RD Sweep

Run a controlled RD sweep after hyperprior implementation.

Keep fixed:

- architecture except entropy model;
- dataset protocol;
- split and subset sizes;
- optimizer schedule where possible.

Compare:

- RD curves, not a single checkpoint;
- likelihood bitrate versus actual bitrate;
- runtime and VRAM.

Decision:

- If hyperprior improves RD, promote it to the new baseline.
- If not, keep the simpler entropy bottleneck and document rejection.

## Step 3: Small Spatial/Window Mamba Pilot

Only run this after the hyperprior result is known.

Rationale:

Literature suggests naive spatial Mamba is risky for image-like data, especially if 2D structure is
flattened into a simple 1D raster sequence. A spatial branch must be local/windowed or 2D-aware to
be defensible.

Pilot constraints:

- operate on latent or pre-latent features, not full-resolution raw pixels;
- use local windows such as 4x4 or 8x8;
- keep RTX 4090 memory use realistic;
- compare against the existing CNN spatial conditioning branch;
- cap training time and stop early if no signal appears.

Variants:

- CNN spatial conditioning baseline;
- local/window spatial Mamba with additive fusion;
- local/window spatial Mamba with concat + 1x1 projection;
- local/window spatial Mamba with affine conditioning.

Success criterion:

- measurable RD improvement over CNN spatial conditioning at comparable compute;
- stable memory and throughput.

Rejection criterion:

- no RD gain;
- improvement smaller than run-to-run noise;
- much higher VRAM or slower training;
- only works under non-reference-comparable protocol changes.

## Step 4: Fusion Ablation

If spatial Mamba survives the pilot, run fusion ablations.

Fusion variants:

```text
affine: spec * (1 + gamma) + beta
add:    spec + spatial
concat: conv1x1([spec, spatial])
gated:  gate * spec + (1 - gate) * spatial
```

Hypothesis:

Better fusion may let the latent separate spectral and spatial information more cleanly, improving
RD performance.

Success criterion:

- RD improvement without large compute penalty;
- no degradation in SA/SAM.

## Step 5: Decide On Spatial Mamba

After the pilot and fusion ablation:

- keep spatial Mamba only if it improves RD under the same protocol;
- otherwise reject it and keep the literature-backed decision note.

This is acceptable even if the original target description included spatial Mamba. The thesis can
state that the component was considered, supported by literature, tested under a constrained pilot,
and rejected because it did not improve compression enough for the compute cost.

## Step 6: Mamba-Aware Decoder Or Refinement

The target description also implies a symmetric decoder. Our decoder is currently mostly CNN.

Do not jump straight to a fully symmetric decoder. Start with a small Mamba-aware refinement:

```text
z_hat -> CNN upsample -> preliminary x_hat
preliminary x_hat -> per-pixel spectral Mamba refinement -> final x_hat
```

Variants:

- CNN decoder baseline;
- CNN decoder + Conv1D spectral refinement;
- CNN decoder + Mamba spectral refinement;
- fuller Mamba decoder only if refinement is promising.

Success criterion:

- improved PSNR/SSIM/SA at comparable bitrate;
- decode latency remains acceptable.

Failure criterion:

- better reconstruction but worse RD due to larger latent pressure;
- significant decode slowdown;
- unstable refinement.

## Final Candidate Model

Only compose the final model from components that passed their ablations:

```text
spectral Mamba encoder
+ optional spatial/window Mamba branch
+ selected fusion module
+ hyperprior if beneficial
+ Mamba-aware decoder/refinement if beneficial
```

The final thesis contribution should be framed as an evidence-backed architecture, not as a direct
implementation of every initially proposed component.

## Mandatory Baselines

- baseline 2D autoencoder;
- baseline 3D patch autoencoder;
- current hierarchical spectral Mamba;
- hierarchical spectral Mamba + hyperprior;
- optional spatial Mamba pilot variant;
- final selected model.

## Reviewer Objections To Preempt

- "Is the improvement due to Mamba or just a better entropy model?"
  - Answer with EntropyBottleneck vs hyperprior ablation.

- "Does spatial Mamba actually help compression?"
  - Answer with CNN spatial conditioning vs spatial/window Mamba ablation.

- "Are metrics comparable?"
  - Preserve split, normalization, masking, and aggregation semantics.

- "Is actual bitrate measured?"
  - Report both likelihood/proxy and actual bitstream metrics where supported.

- "Is the model too expensive for the gain?"
  - Report peak VRAM, train throughput, encode time, and decode time.

## Immediate Next Step

Implement `hierarchical_spectral_mamba_hyperprior` as the next model variant. Keep all other
architecture components unchanged so the first result isolates the hyperprior effect.
