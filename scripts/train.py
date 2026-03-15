from pathlib import Path
import os
import sys
import argparse

import torch
import wandb
from torch.utils.data import Subset

from hsi_compression.data import build_dataset, build_dataloader
from hsi_compression.engine import fit
from hsi_compression.losses import build_loss
from hsi_compression.models.registry import build_model
from hsi_compression.paths import ensure_artifact_dirs, checkpoints_dir
from hsi_compression.utils import (
    load_config,
    load_project_env,
    set_seed,
    get_git_commit_hash,
    get_git_short_hash,
    is_git_dirty,
)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Generic training script driven by YAML config")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Optional dataset root override. If omitted, DATASET_ROOT env var will be used.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional W&B run name override",
    )

    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable W&B logging for this run",
    )

    return parser.parse_args()


def main():
    load_project_env()
    args = parse_args()

    cfg = load_config(args.config)

    experiment_cfg = cfg.get("experiment", {})
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    logging_cfg = cfg.get("logging", {})

    dataset_root_str = (
        args.dataset_root
        or os.environ.get("DATASET_ROOT")
    )

    dataset_root = Path(dataset_root_str)
    if not dataset_root.exists():
        print(f"Error: dataset root does not exist: {dataset_root}")
        sys.exit(1)

    seed = experiment_cfg.get("seed", 42)
    set_seed(seed)
    ensure_artifact_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    difficulty = data_cfg.get("difficulty", "easy")
    batch_size = data_cfg.get("batch_size", 4)
    num_workers = data_cfg.get("num_workers", 0)
    train_subset_size = data_cfg.get("train_subset_size", None)
    val_subset_size = data_cfg.get("val_subset_size", None)
    drop_invalid_channels = data_cfg.get("drop_invalid_channels", False)

    epochs = training_cfg.get("epochs", 20)
    lr = training_cfg.get("lr", 1e-4)
    loss_name = training_cfg.get("loss_name", "masked_mse")
    debug_reconstruction_stats = training_cfg.get("debug_reconstruction_stats", False)

    model_name = model_cfg["model_name"]
    model_kwargs = model_cfg.get("model_kwargs", {})

    use_wandb = logging_cfg.get("use_wandb", True) and not args.disable_wandb
    wandb_project = logging_cfg.get("project", "hsi-compression")

    full_train_ds = build_dataset(
        dataset_root=dataset_root,
        split_name="train",
        difficulty=difficulty,
        normalized=True,
        return_mask=True,
        drop_invalid_channels=drop_invalid_channels,
    )

    full_val_ds = build_dataset(
        dataset_root=dataset_root,
        split_name="val",
        difficulty=difficulty,
        normalized=True,
        return_mask=True,
        drop_invalid_channels=drop_invalid_channels,
    )

    if train_subset_size is not None:
        train_size = min(train_subset_size, len(full_train_ds))
        train_ds = Subset(full_train_ds, list(range(train_size)))
    else:
        train_ds = full_train_ds

    if val_subset_size is not None:
        val_size = min(val_subset_size, len(full_val_ds))
        val_ds = Subset(full_val_ds, list(range(val_size)))
    else:
        val_ds = full_val_ds

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    train_loader = build_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = build_dataloader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    sample = full_train_ds[0]
    in_channels = sample["x"].shape[0]

    model = build_model(
        model_name=model_name,
        in_channels=in_channels,
        **model_kwargs,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = build_loss(loss_name)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    run_config = {
        "config_path": str(Path(args.config).resolve()),
        "dataset_root": str(dataset_root),
        "device": str(device),
        "experiment": experiment_cfg,
        "data": data_cfg,
        "training": training_cfg,
        "model": model_cfg,
        "logging": logging_cfg,
        "in_channels": in_channels,
        "num_params": num_params,
        "input_shape": (in_channels, 128, 128),
        "git_commit": get_git_commit_hash(),
        "git_short_commit": get_git_short_hash(),
        "git_dirty": is_git_dirty(),
    }

    ckpt_name = experiment_cfg.get("name", model_name)
    ckpt_name = f"{ckpt_name}.pt"
    checkpoint_path = checkpoints_dir() / ckpt_name

    run_name = args.run_name or experiment_cfg.get("name") or model_name

    logger = None
    wandb_run = None

    if use_wandb:
        wandb_run = init_wandb(
            project=wandb_project,
            run_name=run_name,
            config=run_config,
        )
        logger = wandb_run

    try:
        if debug_reconstruction_stats:
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

            if logger is not None:
                logger.log({
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
            epochs=epochs,
            checkpoint_path=checkpoint_path,
            config=run_config,
            logger=logger,
            scheduler=None,
            show_progress=True,
        )

        if logger is not None:
            artifact = wandb.Artifact(
                name=f"{model_name}-{wandb_run.id}",
                type="model",
                metadata={
                    "best_val_loss": result["best_val_loss"],
                    "difficulty": difficulty,
                    "train_subset_size": train_subset_size,
                    "val_subset_size": val_subset_size,
                    "loss_name": loss_name,
                    "git_commit": run_config["git_commit"],
                },
            )
            artifact.add_file(str(checkpoint_path))
            logger.log_artifact(artifact)

        print("Training complete.")
        print(f"Best val loss: {result['best_val_loss']:.6f}")
        print(f"Checkpoint: {checkpoint_path}")

    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()