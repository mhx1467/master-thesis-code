import random
import sys
from pathlib import Path

import torch

from hsi_compression.data import build_dataset
from hsi_compression.models.registry import build_model
from hsi_compression.utils.env import load_project_env
from hsi_compression.visualization import (
    plot_rgb,
    plot_rgb_comparison,
    choose_evenly_spaced_rgb_bands,
    plot_random_spectra,
    plot_mean_spectrum_comparison,
)


def main():
    load_project_env()

    if len(sys.argv) < 2:
        print("Usage: python test_visualization.py <dataset_root_path>")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])
    if not dataset_root.exists():
        print(f"Error: Provided dataset root path does not exist: {dataset_root}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = build_dataset(
        dataset_root=dataset_root,
        split_name="train",
        difficulty="easy",
        normalized=True,
        return_mask=True,
        drop_invalid_channels=False,
    )

    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]

    if isinstance(sample, dict):
        x = sample["x"]          # [C, H, W]
        mask = sample.get("mask", None)
    else:
        x = sample
        mask = None

    in_channels = x.shape[0]
    x = x.unsqueeze(0).to(device)   # [1, C, H, W]

    print("Selected sample:", idx)
    print("Input shape:", x.shape)
    print("In channels:", in_channels)

    model = build_model(
        model_name="baseline_2d_ae",
        in_channels=in_channels,
        model_kwargs=dict(
            latent_channels=16,
            hidden_channels=[128, 64],
        ),
    )

    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(x)

    if isinstance(output, dict):
        x_hat = output["x_hat"]
    else:
        x_hat = output

    x = x[0].cpu()
    x_hat = x_hat[0].cpu()

    bands = choose_evenly_spaced_rgb_bands(x.shape[0])

    plot_rgb(
        x,
        bands=bands,
        mask=mask,
        title="Input RGB",
    )

    plot_rgb_comparison(
        x,
        x_hat,
        bands=bands,
        mask=mask,
    )

    plot_random_spectra(
        x,
        n=5,
        mask=mask,
        title="Random input spectra",
    )

    plot_mean_spectrum_comparison(
        x,
        x_hat,
        mask=mask,
        title="Mean spectrum comparison",
    )


if __name__ == "__main__":
    main()