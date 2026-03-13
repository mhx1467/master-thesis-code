from pathlib import Path
import sys
import torch
from torch.utils.data import Subset

from hsi_compression.data.datamodule import build_dataset, build_dataloader
from hsi_compression.models import TinyHSIAutoencoder
from hsi_compression.metrics import masked_mse


def main():
    device = torch.device("cpu")
    if len(sys.argv) < 2:
        print("Usage: python overfit_test.py <dataset_root_path>")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])
    if not dataset_root.exists():
        print(f"Error: Provided dataset root path does not exist: {dataset_root}")
        print("Example usage: python overfit_test.py /path/to/dataset")
        sys.exit(1)
    ds = build_dataset(
        dataset_root=dataset_root,
        split_name="train",
        difficulty="easy",
        normalized=True,
        return_mask=True,
        drop_invalid_channels=False,
    )

    small_ds = Subset(ds, list(range(8)))
    loader = build_dataloader(small_ds, batch_size=2, shuffle=True, num_workers=0)

    model = TinyHSIAutoencoder(bands=224, latent_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(30):
        model.train()
        total_loss = 0.0

        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["valid_mask"].to(device)

            optimizer.zero_grad()
            x_hat = model(x)
            loss = masked_mse(x_hat, x, mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1:03d} | masked_loss={total_loss / len(loader):.6f}")


if __name__ == "__main__":
    main()