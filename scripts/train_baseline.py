from pathlib import Path
import sys
import torch

from hsi_compression.data import build_dataset, build_dataloader
from hsi_compression.engine import train_one_epoch, validate
from hsi_compression.models import Baseline2DAutoencoder
from hsi_compression.paths import ensure_artifact_dirs, checkpoints_dir
from hsi_compression.utils import set_seed


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_baseline.py <dataset_root_path>")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])
    if not dataset_root.exists():
        print(f"Error: Provided dataset root path does not exist: {dataset_root}")
        print("Example usage: python train_baseline.py /path/to/dataset")
        sys.exit(1)


    config = {
        "seed": 42,
        "dataset_root": dataset_root,
        "difficulty": "easy",
        "batch_size": 4,
        "num_workers": 0,
        "epochs": 20,
        "lr": 1e-3,
        "drop_invalid_channels": False,
        "hidden_channels": (128, 64),
        "latent_channels": 16,
    }

    set_seed(config["seed"])
    ensure_artifact_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_root = Path(config["dataset_root"])

    train_ds = build_dataset(
        dataset_root=dataset_root,
        split_name="train",
        difficulty=config["difficulty"],
        normalized=True,
        return_mask=True,
        drop_invalid_channels=config["drop_invalid_channels"],
    )

    val_ds = build_dataset(
        dataset_root=dataset_root,
        split_name="val",
        difficulty=config["difficulty"],
        normalized=True,
        return_mask=True,
        drop_invalid_channels=config["drop_invalid_channels"],
    )

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

    sample = train_ds[0]
    in_channels = sample["x"].shape[0]

    model = Baseline2DAutoencoder(
        in_channels=in_channels,
        hidden_channels=config["hidden_channels"],
        latent_channels=config["latent_channels"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    best_val_loss = float("inf")
    best_ckpt_path = checkpoints_dir() / "baseline_2d_ae_easy_best.pt"

    for epoch in range(1, config["epochs"] + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
        )

        latent_shape = val_metrics["latent_shape"]
        ratio = model.compression_ratio_proxy(
            input_shape=(in_channels, 128, 128),
            latent_shape=latent_shape,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['train_loss']:.6f} | "
            f"train_rmse={train_metrics['train_rmse']:.6f} | "
            f"val_loss={val_metrics['val_loss']:.6f} | "
            f"val_rmse={val_metrics['val_rmse']:.6f} | "
            f"val_sam_deg={val_metrics['val_sam_deg']:.6f} | "
            f"latent={latent_shape} | "
            f"ratio≈{ratio:.2f}"
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "best_val_loss": best_val_loss,
                    "latent_shape": latent_shape,
                    "compression_ratio_proxy": ratio,
                },
                best_ckpt_path,
            )
            print(f"  Saved best checkpoint to: {best_ckpt_path}")

    print("Training complete.")
    print(f"Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()