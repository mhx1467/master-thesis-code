from pathlib import Path
import sys

import torch
import wandb
from torch.utils.data import Subset

from hsi_compression.data.datamodule import build_dataset, build_dataloader
from hsi_compression.metrics import masked_mse
from hsi_compression.models import TinyHSIAutoencoder
from hsi_compression.utils import load_project_env, set_seed
from hsi_compression.utils.wandb_utils import init_wandb


def main():
    load_project_env()

    if len(sys.argv) < 2:
        print("Usage: python scripts/overfit_test.py <dataset_root_path>")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])
    if not dataset_root.exists():
        print(f"Error: Provided dataset root path does not exist: {dataset_root}")
        print("Example usage: python scripts/overfit_test.py /path/to/dataset")
        sys.exit(1)

    config = {
        "seed": 42,
        "dataset_root": str(dataset_root),
        "difficulty": "easy",
        "subset_size": 8,
        "batch_size": 2,
        "num_workers": 0,
        "epochs": 30,
        "lr": 1e-3,
        "drop_invalid_channels": False,
        "model_name": "tiny_ae",
        "latent_channels": 16,
        "loss_name": "masked_mse",
    }

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds = build_dataset(
        dataset_root=dataset_root,
        split_name="train",
        difficulty=config["difficulty"],
        normalized=True,
        return_mask=True,
        drop_invalid_channels=config["drop_invalid_channels"],
    )

    small_ds = Subset(ds, list(range(config["subset_size"])))
    loader = build_dataloader(
        small_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    sample = ds[0]
    in_channels = sample["x"].shape[0]

    model = TinyHSIAutoencoder(
        bands=in_channels,
        latent_channels=config["latent_channels"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    config["device"] = str(device)
    config["in_channels"] = in_channels
    config["num_params"] = num_params
    config["input_shape"] = tuple(sample["x"].shape)

    with init_wandb(
        project="hsi-compression",
        run_name=f"overfit_{config['model_name']}_{config['difficulty']}_latent{config['latent_channels']}",
        config=config,
    ) as run:
        best_train_loss = float("inf")
        best_epoch = None

        for epoch in range(1, config["epochs"] + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0
            latent_shape = None

            for batch in loader:
                x = batch["x"].to(device)
                mask = batch["valid_mask"].to(device)

                optimizer.zero_grad()

                outputs = model(x)
                x_hat = outputs["x_hat"]
                z = outputs.get("z")

                loss = masked_mse(x_hat, x, mask)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if latent_shape is None and z is not None:
                    latent_shape = tuple(z.shape[1:])

            avg_loss = total_loss / max(num_batches, 1)

            log_dict = {
                "epoch": epoch,
                "train/loss": avg_loss,
            }

            if latent_shape is not None:
                log_dict["model/latent_c"] = latent_shape[0]
                log_dict["model/latent_h"] = latent_shape[1]
                log_dict["model/latent_w"] = latent_shape[2]

            run.log(log_dict, step=epoch)

            print(f"Epoch {epoch:03d} | masked_loss={avg_loss:.6f}")

            if avg_loss < best_train_loss:
                best_train_loss = avg_loss
                best_epoch = epoch
                run.summary["best_train_loss"] = best_train_loss
                run.summary["best_epoch"] = best_epoch

        print("Overfit test complete.")
        print(f"Best train loss: {best_train_loss:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()