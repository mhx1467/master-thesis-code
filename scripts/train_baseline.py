from pathlib import Path
import os

import torch
import wandb

from hsi_compression.data import build_dataset, build_dataloader
from hsi_compression.engine import fit
from hsi_compression.metrics import masked_mse
from hsi_compression.models import Baseline2DAutoencoder
from hsi_compression.paths import ensure_artifact_dirs, checkpoints_dir
from hsi_compression.utils import set_seed


def main():
    config = {
        "seed": 42,
        "dataset_root": os.environ.get(
            "DATASET_ROOT",
            "/home/brwsx/hsi-compression/pipelines/pull-dataset/hyspectnet-11k/hyspecnet-11k-full",
        ),
        "difficulty": "easy",
        "batch_size": 4,
        "num_workers": 4,
        "epochs": 20,
        "lr": 1e-3,
        "drop_invalid_channels": False,
        "hidden_channels": (128, 64),
        "latent_channels": 16,
        "model_name": "baseline_2d_ae",
        "loss_name": "masked_mse",
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
    loss_fn = masked_mse

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    config["device"] = str(device)
    config["in_channels"] = in_channels
    config["num_params"] = num_params
    config["input_shape"] = (in_channels, 128, 128)

    checkpoint_path = checkpoints_dir() / "baseline_2d_ae_easy_best.pt"

    with wandb.init(
        project="hsi-compression",
        name=f"{config['model_name']}_{config['difficulty']}_latent{config['latent_channels']}",
        config=config,
    ) as run:
        # Optional. Disable if it slows training too much.
        # run.watch(model, log="gradients", log_freq=100)

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
        )

        # Upload best checkpoint as W&B artifact
        artifact = wandb.Artifact(
            name=f"{config['model_name']}-{run.id}",
            type="model",
            metadata={
                "best_val_loss": result["best_val_loss"],
                "difficulty": config["difficulty"],
                "latent_channels": config["latent_channels"],
            },
        )
        artifact.add_file(str(checkpoint_path))
        run.log_artifact(artifact)

        print("Training complete.")
        print(f"Best val loss: {result['best_val_loss']:.6f}")


if __name__ == "__main__":
    main()