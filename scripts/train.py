import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

from hsi_compression.data import build_dataloader, build_dataset
from hsi_compression.engine import fit
from hsi_compression.losses import build_loss
from hsi_compression.models.registry import build_model
from hsi_compression.paths import checkpoints_dir, ensure_artifact_dirs
from hsi_compression.utils import (
    get_git_short_hash,
    is_git_dirty,
    load_config,
    load_project_env,
    set_seed,
)
from hsi_compression.utils.wandb_utils import init_wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Train HSI compression model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
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

    dataset_root = Path(
        args.dataset_root or os.environ.get("DATASET_ROOT") or "/data/hyspecnet-11k"
    )
    if not dataset_root.exists():
        print(f"Error: dataset_root does not exist: {dataset_root}")
        sys.exit(1)

    set_seed(experiment_cfg.get("seed", 42))
    ensure_artifact_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    difficulty = data_cfg.get("difficulty", "easy")
    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 4)
    drop_invalid = data_cfg.get("drop_invalid_channels", True)
    train_subset = data_cfg.get("train_subset_size", None)
    val_subset = data_cfg.get("val_subset_size", None)

    print(
        f"\nDataset: {difficulty} split | drop_invalid_channels={drop_invalid} | normalization: global min-max [0,1]"
    )

    train_ds = build_dataset(
        dataset_root=dataset_root,
        split_name="train",
        difficulty=difficulty,
        normalized=True,
        return_mask=True,
        drop_invalid_channels=drop_invalid,
    )
    val_ds = build_dataset(
        dataset_root=dataset_root,
        split_name="val",
        difficulty=difficulty,
        normalized=True,
        return_mask=True,
        drop_invalid_channels=drop_invalid,
    )

    if train_subset:
        train_ds = Subset(train_ds, list(range(min(train_subset, len(train_ds)))))
    if val_subset:
        val_ds = Subset(val_ds, list(range(min(val_subset, len(val_ds)))))

    sample = train_ds[0] if not train_subset else train_ds.dataset[0]
    num_input_bands = sample["x"].shape[0]
    print(f"Input bands: {num_input_bands} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = build_dataloader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = build_dataloader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model_name = model_cfg.get("model_name")
    model_kwargs = {
        k: v for k, v in model_cfg.get("model_kwargs", {}).items() if k != "in_channels"
    }

    model = build_model(model_name=model_name, in_channels=num_input_bands, **model_kwargs).to(
        device
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name} | Parameters: {n_params:,}")

    epochs = training_cfg.get("epochs", 500)
    lr = training_cfg.get("lr", 1e-4)
    loss_name = training_cfg.get("loss_name", "mse")
    grad_clip = training_cfg.get("grad_clip_max_norm", 1.0)
    quant_bits = training_cfg.get("quantization_bits", 8)
    sam_every = training_cfg.get("sam_every_n_epochs", 10)
    scheduler_cfg = training_cfg.get("scheduler", {})

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = None
    if scheduler_cfg.get("enabled", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("T_max", epochs),
            eta_min=scheduler_cfg.get("eta_min", 1e-6),
        )

    loss_fn = build_loss(loss_name)
    exp_name = experiment_cfg.get("name", "experiment")
    ckpt_path = checkpoints_dir() / f"{exp_name}_best.pt"

    use_wandb = logging_cfg.get("use_wandb", False) and not args.disable_wandb
    run_cfg = {
        **cfg,
        "num_input_bands": num_input_bands,
        "git_hash": get_git_short_hash(),
        "git_dirty": is_git_dirty(),
    }

    def _run(logger=None):
        return fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epochs=epochs,
            checkpoint_path=ckpt_path,
            config=cfg,
            logger=logger,
            scheduler=scheduler,
            show_progress=True,
            train_sampler=None,
            grad_clip_max_norm=grad_clip,
            num_input_bands=num_input_bands,
            quantization_bits=quant_bits,
            sam_every_n_epochs=sam_every,
            resume=args.resume,
        )

    if use_wandb:
        with init_wandb(
            project=logging_cfg.get("project", "hsi-compression"),
            run_name=args.run_name or exp_name,
            config=run_cfg,
        ) as run:
            result = _run(logger=run)
    else:
        result = _run()

    print(f"\n{'=' * 55}")
    print("Training completed.")
    print(f"Best val/psnr: {result['best_val_psnr']:.2f} dB")
    print(f"Checkpoint:    {ckpt_path}")


if __name__ == "__main__":
    main()
