Next, move from **data plumbing** to a **real experiment scaffold**.

You now have the hard foundation done. The next steps should be:

## 1. Freeze preprocessing decisions

Write down, in one place:

* split used: `easy`
* input source: `SPECTRAL_IMAGE.TIF`
* nodata value: `-32768`
* invalid channels: the 22-channel list
* whether you keep or drop invalid channels
* normalization: per-band mean/std from train split, valid pixels only

This becomes your preprocessing contract. Do not casually change it later.

---

## 2. Build reusable train/val/test data helpers

Right now you have scripts. Next you want functions like:

* `get_split_paths(split_name="train", difficulty="easy")`
* `build_dataset(split_name, training=False)`
* `build_dataloader(split_name, batch_size, shuffle)`

That will stop the copy-paste drift.

A good next file is `data.py` with:

* constants
* path resolution
* stats loading
* dataset construction
* dataloader construction

---

## 3. Build a real validation loop

Before a better model, build the training loop structure.

You want:

* training epoch
* validation epoch
* logging of:

  * train loss
  * val masked MSE
  * val masked RMSE
  * val masked SAM
* best-checkpoint saving

This is more important than model complexity.

---

## 4. Implement the first real baseline model

Your current tiny AE is a sanity-check model, not yet a real compression baseline.

The first proper baseline should be:

### Small 2D convolutional autoencoder with downsampling

Input:

* `(B, H, W)` = `(224, 128, 128)` or `(202, 128, 128)`

Architecture idea:

* encoder:

  * Conv block
  * stride-2 downsample
  * Conv block
  * stride-2 downsample
  * latent bottleneck
* decoder:

  * transpose conv / upsample
  * reconstruct to original size

This gives you an actual compressed latent representation.

The first baseline does not need entropy coding yet. For now, compression can be represented by:

* latent channel count
* latent spatial size
* total latent elements vs input elements

---

## 5. Define your first evaluation protocol

Keep it small and controlled.

I would start with:

* split: `easy`
* train on full train split
* validate on val split
* test only after you are satisfied

Metrics:

* masked MSE
* masked RMSE
* masked SAM (degrees)
* later PSNR

Train only one baseline first.

---

## 6. Define the compression proxy clearly

Since you are not yet doing full learned entropy coding, define a simple rate proxy:

[
\text{compression ratio proxy} = \frac{\text{input tensor elements}}{\text{latent tensor elements}}
]

For example, if input is:

* `224 x 128 x 128`

and latent is:

* `16 x 32 x 32`

then the proxy ratio is:

[
\frac{224 \cdot 128 \cdot 128}{16 \cdot 32 \cdot 32}
]

This is not full codec bitrate, but it is a valid thesis-stage comparison if stated clearly.

---

## 7. Add checkpointing now

Save:

* `last.pt`
* `best_val_loss.pt`
* maybe `best_val_sam.pt`

Each checkpoint should store:

* model state
* optimizer state
* epoch
* config dict

Without this, later experiments become painful.

---

## 8. Add a config object

Do not hardcode everything inside scripts forever.

Use either:

* a Python dict
* a dataclass
* YAML later if you want

At minimum:

```python
config = {
    "difficulty": "easy",
    "batch_size": 4,
    "lr": 1e-3,
    "epochs": 30,
    "latent_channels": 16,
    "drop_invalid_channels": False,
}
```

This makes experiments repeatable.

---

## 9. Add visualization utilities

Before large training, make two small debug plots:

* pseudo-RGB image from selected bands
* spectral curve for a few pixels

This helps you inspect:

* raw data
* normalized data
* reconstructed data

For HSI, this is extremely useful and thesis-friendly.

---

## 10. Only then start the real experiment ladder

Your sequence should be:

### Phase A

One proper 2D baseline:

* one latent size
* short training
* validate metrics

### Phase B

Same model at 2–3 latent sizes

### Phase C

Add second model family:

* maybe 3D CNN AE

### Phase D

Add one stronger model

That keeps the project controlled.

---

# What I would do immediately next

The very next deliverable should be:

## `train_baseline.py`

with:

* reusable dataloaders
* one real 2D convolutional AE
* train loop
* val loop
* checkpoint saving
* metric printing

That is the correct next milestone.

A clean order for the next few tasks is:

1. `constants.py`
2. `data.py`
3. `baseline_ae.py`
4. `engine.py` for train/validate
5. `train_baseline.py`

If you want, next I’ll help you build exactly that starting from `constants.py` and `data.py`.
