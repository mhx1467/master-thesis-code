# Spectral Modeling Experiment: TCN-based HSI Autoencoder

## Goal

The goal of this experiment was to evaluate whether explicitly modeling spectral dependencies in hyperspectral images improves compression quality compared to a standard 2D convolutional autoencoder.

I've investigated the use of **Temporal Convolutional Networks (TCN)** to model spectral sequences across wavelength bands.

---

# Research Question

**Does explicit spectral sequence modeling using TCN improve reconstruction quality in hyperspectral image compression compared to a baseline 2D autoencoder?**

More specifically:

> Can modeling inter-band dependencies using 1D temporal convolutions over the spectral dimension improve reconstruction metrics (RMSE, PSNR, SAM)?

---

# Motivation

Hyperspectral images exhibit strong correlations across adjacent spectral bands.

The baseline 2D autoencoder:

* processes spectral channels jointly using 2D convolutions,
* but does not explicitly model spectral ordering.

TCN was introduced to:

* treat each pixel spectrum as a sequence,
* explicitly model long-range dependencies across wavelengths.

---

# Tested Architectures

Two variants of TCN-based autoencoders were implemented.

---

## TCN v1: Latent-Space Spectral Modeling

### Architecture

```text
Input (N, 224, 128, 128)

Spatial Encoder (Conv2d)

Latent (C_latent, 32, 32)

TCN over latent channels

Spatial Decoder

Reconstruction
```

### Key Property

* TCN operates on latent channels (e.g. length = 8)
* Does NOT see full spectral resolution (224 bands)

### Limitation

* Spectral modeling happens after compression
* Sequence length too short, weak spectral modeling

---

## TCN v2: Spectral-First Modeling

### Architecture

```text
Input (N, 224, 128, 128)

TCN over full spectral sequence (per pixel)

Spectrally refined features

Spatial Encoder

Latent (8, 32, 32)

Spatial Decoder

Reconstruction
```

### Key Property

* TCN operates on full spectral sequence (length = 224)
* Each pixel spectrum is treated as a sequence

### Improvement over v1

* Proper spectral modeling before compression
* Better alignment with HSI structure

---

# Experimental Setup

### Dataset

* Dataset: **HyspecNet-11k**
* Split: **easy**
* Subset used (mid-scale):

  * train: 256
  * val: 64

### Training

* optimizer: Adam
* learning rate: 1e-4
* loss: masked MSE
* epochs: 20
* early stopping: patience = 4
* normalization: per-band standardization

---

# Results

## TCN v1 (mid-scale)

| Metric   | Value  |
| -------- | ------ |
| val_loss | ~0.056 |
| RMSE     | ~0.222 |
| SAM      | ~11.1° |

---

## TCN v2 (mid-scale)

| Metric   | Value      |
| -------- | ---------- |
| val_loss | **0.0522** |
| RMSE     | **0.2181** |
| SAM      | **11.35°** |

---

## Baseline Reference (mid-scale)

For comparison:

| Model          | val_loss | RMSE  | SAM  |
| -------------- | -------- | ----- | ---- |
| Baseline 2D AE | ~0.055   | ~0.22 | ~11° |

---

# Observations

### 1. TCN v2 improves over TCN v1

* Lower loss
* Lower RMSE
* More stable training

This confirms that:

> Modeling the full spectral sequence is more effective than operating on compressed latent channels.

---

### 2. No improvement over baseline

Despite architectural improvements:

* TCN v2 does **not outperform baseline 2D AE**
* Performance remains comparable or slightly worse

---

### 3. Training instability

TCN models show:

* higher variance in validation loss,
* occasional spikes,
* more sensitivity to optimization.

This suggests:

* more complex optimization landscape,
* higher difficulty compared to simple Conv2D models.

---

### 4. Strong baseline effect

The baseline 2D AE:

* already captures spectral correlations implicitly,
* is simpler and more stable,
* achieves competitive or better results.

---

# Answer to Research Question

**Result:**

> Explicit spectral modeling using TCN does not improve reconstruction quality over a standard 2D autoencoder in the tested configurations.

While TCN successfully models spectral dependencies, it does not translate into improved compression performance under the current architecture and training setup.

---

# Conclusion

* TCN-based models are **valid and trainable**
* Proper spectral modeling (v2) improves over naive approaches (v1)
* However, no performance gain over baseline was observed

---

# Implications

This suggests that:

* simple convolutional architectures are already strong baselines for HSI compression
* adding spectral sequence modeling alone is insufficient
* better integration of spatial and spectral modeling is required
