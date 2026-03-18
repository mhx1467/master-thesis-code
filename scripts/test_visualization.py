import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from hsi_compression.data import build_dataset
from hsi_compression.engine.checkpointing import load_checkpoint
from hsi_compression.models.registry import build_model
from hsi_compression.paths import checkpoints_dir
from hsi_compression.utils.env import load_project_env
from hsi_compression.visualization import (
    choose_evenly_spaced_rgb_bands,
    hsi_to_rgb,
    plot_mean_spectrum_comparison,
    plot_random_spectra,
)


def main():
    load_project_env()

    if len(sys.argv) < 2:
        print("Usage: python test_visualization.py <dataset_root_path> [checkpoint_path]")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])
    if not dataset_root.exists():
        print(f"Error: Provided dataset root path does not exist: {dataset_root}")
        sys.exit(1)

    checkpoint_path = (
        Path(sys.argv[2]) if len(sys.argv) >= 3 else checkpoints_dir() / "baseline_2d_ae_full.pt"
    )
    if not checkpoint_path.exists():
        print(f"Error: checkpoint does not exist: {checkpoint_path}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = build_dataset(
        dataset_root=dataset_root,
        split_name="test",
        difficulty="easy",
        normalized=True,
        return_mask=True,
        drop_invalid_channels=False,
    )

    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]

    if isinstance(sample, dict):
        x = sample["x"]  # [C, H, W]
        mask = sample.get("mask", None)
    else:
        x = sample
        mask = None

    in_channels = x.shape[0]
    x = x.unsqueeze(0).to(device)  # [1, C, H, W]

    print("Selected sample:", idx)
    print("Input shape:", x.shape)
    print("In channels:", in_channels)

    checkpoint_raw = torch.load(checkpoint_path, map_location="cpu")
    ckpt_config = checkpoint_raw.get("config", {})
    model_name = ckpt_config.get("model_name", "baseline_2d_ae")
    model_kwargs = ckpt_config.get("model_kwargs", {})
    if not model_kwargs:
        model_kwargs = {
            "latent_channels": ckpt_config.get("latent_channels", 16),
            "hidden_channels": tuple(ckpt_config.get("hidden_channels", (128, 64))),
        }

    model = build_model(
        model_name=model_name,
        in_channels=in_channels,
        **model_kwargs,
    )

    model.to(device)
    load_checkpoint(path=checkpoint_path, model=model, map_location=device)
    model.eval()

    with torch.no_grad():
        output = model(x)

    x_hat = output["x_hat"] if isinstance(output, dict) else output

    x = x[0].cpu()
    x_hat = x_hat[0].cpu()

    bands = choose_evenly_spaced_rgb_bands(x.shape[0])

    rgb_input = hsi_to_rgb(x, bands=bands, mask=mask)
    rgb_recon = hsi_to_rgb(x_hat, bands=bands, mask=mask)
    rgb_error = (rgb_input - rgb_recon) ** 2

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].imshow(rgb_input)
    axes[0, 0].set_title("Input RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb_recon)
    axes[0, 1].set_title("Reconstruction RGB")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(rgb_error)
    axes[0, 2].set_title("RGB Squared Error")
    axes[0, 2].axis("off")

    plot_random_spectra(
        x,
        n=5,
        mask=mask,
        title="Random input spectra",
        ax=axes[1, 0],
        show=False,
    )

    plot_random_spectra(
        x_hat,
        n=5,
        mask=mask,
        title="Random reconstruction spectra",
        ax=axes[1, 1],
        show=False,
    )

    plot_mean_spectrum_comparison(
        x,
        x_hat,
        mask=mask,
        title="Mean spectrum comparison",
        ax=axes[1, 2],
        show=False,
    )

    fig.suptitle("HSI Visualization (Single Canvas)", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
