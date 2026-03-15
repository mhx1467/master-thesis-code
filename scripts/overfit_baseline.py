from pathlib import Path
import os
import sys

import torch
import wandb
from torch.utils.data import Subset

from hsi_compression.data import build_dataset, build_dataloader
from hsi_compression.engine.checkpointing import save_checkpoint
from hsi_compression.metrics import masked_mse, masked_rmse
from hsi_compression.models import Baseline2DAutoencoder
from hsi_compression.paths import ensure_artifact_dirs, checkpoints_dir
from hsi_compression.utils import load_project_env, set_seed
from hsi_compression.utils.wandb_utils import init_wandb


def tensor_stats(x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    if mask is not None:
        vals = x[mask]
    else:
        vals = x.reshape(-1)

    if vals.numel() == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "mean": vals.mean().item(),
        "std": vals.std().item(),
        "min": vals.min().item(),
        "max": vals.max().item(),
    }


def main():
    load_project_env()

    if len(sys.argv) >= 2:
        dataset_root = Path(sys.argv[1])
    else:
        dataset_root = Path(
            os.environ.get(
                "DATASET_ROOT",
                "/home/brwsx/hsi-compression/pipelines/pull-dataset/hyspectnet-11k/hyspecnet-11k-full",
            )
        )

    if not dataset_root.exists():
        print(f"Error: dataset root does not exist: {dataset_root}")
        sys.exit(1)

    config = {
        "seed": 42,
        "dataset_root": str(dataset_root),
        "difficulty": "easy",
        "subset_size": 8,
        "batch_size": 2,
        "num_workers": 0,
        "epochs": 100,
        "lr": 1e-4,
        "drop_invalid_channels": False,
        "hidden_channels": (128, 64),
        "latent_channels": 16,
        "model_name": "baseline_2d_ae_overfit",
        "loss_name": "masked_mse",
    }

    set_seed(config["seed"])
    ensure_artifact_dirs()

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

    subset_indices = list(range(config["subset_size"]))
    small_ds = Subset(ds, subset_indices)

    loader = build_dataloader(
        small_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    sample = ds[0]
    in_channels = sample["x"].shape[0]

    model = Baseline2DAutoencoder(
        in_channels=in_channels,
        hidden_channels=config["hidden_channels"],
        latent_channels=config["latent_channels"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config["device"] = str(device)
    config["in_channels"] = in_channels
    config["num_params"] = num_params
    config["input_shape"] = (in_channels, 128, 128)

    print(f"Trainable parameters: {num_params:,}")

    checkpoint_path = checkpoints_dir() / "baseline_2d_ae_overfit_best.pt"
    best_loss = float("inf")
    best_epoch = None

    with init_wandb(
        project="hsi-compression",
        run_name=f"{config['model_name']}_{config['difficulty']}_latent{config['latent_channels']}",
        config=config,
    ) as run:
        for epoch in range(1, config["epochs"] + 1):
            model.train()

            total_loss = 0.0
            total_rmse = 0.0
            num_batches = 0

            last_x_stats = None
            last_xhat_stats = None
            last_latent_shape = None

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

                rmse = masked_rmse(x_hat.detach(), x, mask)

                total_loss += loss.item()
                total_rmse += rmse.item()
                num_batches += 1

                last_x_stats = tensor_stats(x.detach(), mask)
                last_xhat_stats = tensor_stats(x_hat.detach(), mask)
                if z is not None:
                    last_latent_shape = tuple(z.shape[1:])

            avg_loss = total_loss / max(num_batches, 1)
            avg_rmse = total_rmse / max(num_batches, 1)

            log_dict = {
                "epoch": epoch,
                "train/loss": avg_loss,
                "train/rmse": avg_rmse,
                "debug/x_mean": last_x_stats["mean"],
                "debug/x_std": last_x_stats["std"],
                "debug/x_hat_mean": last_xhat_stats["mean"],
                "debug/x_hat_std": last_xhat_stats["std"],
                "debug/x_hat_min": last_xhat_stats["min"],
                "debug/x_hat_max": last_xhat_stats["max"],
            }

            if last_latent_shape is not None:
                log_dict["model/latent_c"] = last_latent_shape[0]
                log_dict["model/latent_h"] = last_latent_shape[1]
                log_dict["model/latent_w"] = last_latent_shape[2]

                ratio = model.compression_ratio_proxy(
                    input_shape=config["input_shape"],
                    latent_shape=last_latent_shape,
                )
                log_dict["model/compression_ratio_proxy"] = ratio

            run.log(log_dict, step=epoch)

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={avg_loss:.6f} | "
                f"train_rmse={avg_rmse:.6f} | "
                f"x_mean={last_x_stats['mean']:.4f} | "
                f"x_std={last_x_stats['std']:.4f} | "
                f"x_hat_mean={last_xhat_stats['mean']:.4f} | "
                f"x_hat_std={last_xhat_stats['std']:.4f}"
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch

                save_checkpoint(
                    path=checkpoint_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    best_val_loss=best_loss,
                    extra={
                        "latent_shape": last_latent_shape,
                        "subset_indices": subset_indices,
                    },
                )

                run.summary["best_train_loss"] = best_loss
                run.summary["best_epoch"] = best_epoch
                run.summary["best_checkpoint_path"] = str(checkpoint_path)

        print("Overfit baseline complete.")
        print(f"Best train loss: {best_loss:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()