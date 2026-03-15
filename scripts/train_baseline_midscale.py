from pathlib import Path
import os
import sys

import torch
import wandb
from torch.utils.data import Subset

from hsi_compression.data import build_dataset, build_dataloader
from hsi_compression.engine import fit
from hsi_compression.metrics import masked_mse
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
        "train_subset_size": 256,
        "val_subset_size": 64,
        "batch_size": 4,
        "num_workers": 0,
        "epochs": 20,
        "lr": 1e-4,
        "drop_invalid_channels": False,
        "hidden_channels": (128, 64),
        "latent_channels": 16,
        "model_name": "baseline_2d_ae_midscale",
        "loss_name": "masked_mse",
        "debug_reconstruction_stats": True,
    }

    set_seed(config["seed"])
    ensure_artifact_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_train_ds = build_dataset(
        dataset_root=dataset_root,
        split_name="train",
        difficulty=config["difficulty"],
        normalized=True,
        return_mask=True,
        drop_invalid_channels=config["drop_invalid_channels"],
    )

    full_val_ds = build_dataset(
        dataset_root=dataset_root,
        split_name="val",
        difficulty=config["difficulty"],
        normalized=True,
        return_mask=True,
        drop_invalid_channels=config["drop_invalid_channels"],
    )

    train_indices = list(range(min(config["train_subset_size"], len(full_train_ds))))
    val_indices = list(range(min(config["val_subset_size"], len(full_val_ds))))

    train_ds = Subset(full_train_ds, train_indices)
    val_ds = Subset(full_val_ds, val_indices)

    train_loader = build_dataloader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    val_loader = build_dataloader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    sample = full_train_ds[0]
    in_channels = sample["x"].shape[0]

    model = Baseline2DAutoencoder(
        in_channels=in_channels,
        hidden_channels=config["hidden_channels"],
        latent_channels=config["latent_channels"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = masked_mse

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    config["device"] = str(device)
    config["in_channels"] = in_channels
    config["num_params"] = num_params
    config["input_shape"] = (in_channels, 128, 128)

    checkpoint_path = checkpoints_dir() / "baseline_2d_ae_midscale_best.pt"

    with init_wandb(
        project="hsi-compression",
        run_name=(
            f"{config['model_name']}_"
            f"{config['difficulty']}_"
            f"train{config['train_subset_size']}_"
            f"val{config['val_subset_size']}_"
            f"latent{config['latent_channels']}"
        ),
        config=config,
    ) as run:
        if config["debug_reconstruction_stats"]:
            debug_batch = next(iter(train_loader))
            x_dbg = debug_batch["x"].to(device)
            m_dbg = debug_batch["valid_mask"].to(device)

            model.eval()
            with torch.no_grad():
                outputs_dbg = model(x_dbg)
                xhat_dbg = outputs_dbg["x_hat"]

            x_stats = tensor_stats(x_dbg, m_dbg)
            xhat_stats = tensor_stats(xhat_dbg, m_dbg)

            print("DEBUG INPUT/OUTPUT STATS BEFORE TRAINING")
            print(
                f"x     mean={x_stats['mean']:.4f} std={x_stats['std']:.4f} "
                f"min={x_stats['min']:.4f} max={x_stats['max']:.4f}"
            )
            print(
                f"x_hat mean={xhat_stats['mean']:.4f} std={xhat_stats['std']:.4f} "
                f"min={xhat_stats['min']:.4f} max={xhat_stats['max']:.4f}"
            )

            run.log({
                "debug_init/x_mean": x_stats["mean"],
                "debug_init/x_std": x_stats["std"],
                "debug_init/x_min": x_stats["min"],
                "debug_init/x_max": x_stats["max"],
                "debug_init/x_hat_mean": xhat_stats["mean"],
                "debug_init/x_hat_std": xhat_stats["std"],
                "debug_init/x_hat_min": xhat_stats["min"],
                "debug_init/x_hat_max": xhat_stats["max"],
            }, step=0)

        result = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epochs=config["epochs"],
            checkpoint_path=checkpoint_path,
            config=config,
            logger=run,
            scheduler=None,
            show_progress=True,
        )

        artifact = wandb.Artifact(
            name=f"{config['model_name']}-{run.id}",
            type="model",
            metadata={
                "best_val_loss": result["best_val_loss"],
                "difficulty": config["difficulty"],
                "train_subset_size": config["train_subset_size"],
                "val_subset_size": config["val_subset_size"],
                "latent_channels": config["latent_channels"],
            },
        )
        artifact.add_file(str(checkpoint_path))
        run.log_artifact(artifact)

        print("Mid-scale training complete.")
        print(f"Best val loss: {result['best_val_loss']:.6f}")


if __name__ == "__main__":
    main()