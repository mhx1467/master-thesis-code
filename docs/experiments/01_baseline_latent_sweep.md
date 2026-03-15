# Baseline Experiment: Latent Size Sweep for 2D Autoencoder

## Goal

The goal of this experiment was to analyze how the size of the latent representation influences the reconstruction quality of a hyperspectral image compression model.

Specifically, I tested whether increasing the number of latent channels improves reconstruction quality, and how it affects the effective compression ratio.

The experiment used the **Baseline 2D Autoencoder** architecture and varied only the **latent bottleneck size**.

---

# Research Question

**How does the latent bottleneck size affect reconstruction quality and compression ratio for a 2D convolutional autoencoder applied to hyperspectral image compression?**

More specifically:

> Does increasing the latent dimensionality improve reconstruction quality, and what is the trade-off between reconstruction fidelity and compression ratio?

---

# Experimental Setup

### Dataset

* Dataset: **HyspecNet-11k**
* Difficulty split: **easy**
* Training samples: **8037**
* Validation samples: **2295**
* Test samples: **1148**

### Input

Hyperspectral patches:

```
224 spectral bands
128 × 128 spatial resolution
```

### Model

Architecture:

```
Baseline2DAutoencoder
```

Structure:

```
Input (224 × 128 × 128)

Conv Encoder

Latent representation (C × 32 × 32)

Conv Decoder

Reconstructed HSI cube
```

### Training Settings

* optimizer: **Adam**
* learning rate: **1e-4**
* loss: **masked MSE**
* epochs: **max 20**
* early stopping: **patience = 4**
* normalization: **per-band standardization**
* nodata pixels masked during loss computation

---

# Tested Variants

Three bottleneck sizes were evaluated:

| Variant | Latent Channels | Latent Shape |
| ------- | --------------- | ------------ |
| Model A | 8               | (8, 32, 32)  |
| Model B | 16              | (16, 32, 32) |
| Model C | 32              | (32, 32, 32) |

---

# Compression Ratio (Proxy)

The proxy compression ratio is estimated as:

```
input size / latent size
```

Input size:

```
224 × 128 × 128
```

Latent size:

```
C × 32 × 32
```

| Latent Channels | Compression Ratio (proxy) |
| --------------- | ------------------------- |
| 8               | ~448×                     |
| 16              | ~224×                     |
| 32              | ~112×                     |

---

# Results

## Validation Results

| Latent Channels | Loss         | RMSE         | PSNR         | SAM (deg) |
| --------------- | ------------ | ------------ | ------------ | --------- |
| 8               | **0.012320** | **0.105787** | **19.99 dB** | **8.66°** |
| 16              | 0.019208     | 0.132232     | 18.04 dB     | 10.55°    |
| 32              | 0.015129     | 0.117332     | 19.08 dB     | 9.42°     |

---

## Test Results

| Latent Channels | Loss         | RMSE         | PSNR         | SAM (deg) | Ratio    |
| --------------- | ------------ | ------------ | ------------ | --------- | -------- |
| 8               | **0.011780** | **0.103457** | **20.19 dB** | **8.61°** | **448×** |
| 16              | 0.018243     | 0.128923     | 18.27 dB     | 10.42°    | 224×     |
| 32              | 0.014429     | 0.114442     | 19.33 dB     | 9.31°     | 112×     |

---

# Observations

1. The **latent size strongly affects reconstruction quality**.

2. Contrary to the common expectation, the **smallest latent representation (8 channels) achieved the best reconstruction quality**.

3. Increasing latent dimensionality from 8 → 16 degraded performance.

4. Increasing latent size further to 32 improved performance relative to 16 but still did not outperform the 8-channel model.

5. The **latent-8 model achieved both the best reconstruction quality and the highest compression ratio**.

This suggests that a stronger bottleneck may act as an implicit regularizer and help the model learn more efficient representations.

---

# Answer to Research Question

**Result:**

> Increasing the latent dimensionality did not improve reconstruction quality for the tested 2D autoencoder architecture.

Instead:

* the **smallest latent representation (8 channels)** achieved the **best reconstruction metrics**,
* while simultaneously providing the **highest compression ratio (~448×)**.

This indicates that for the tested architecture and training setup, **a stronger bottleneck leads to better generalization and more efficient feature representation**.

---

# Conclusion

The best performing configuration of the Baseline 2D Autoencoder is:

```
latent_channels = 8
```

This configuration will be used as the **baseline reference model** for further experiments.
