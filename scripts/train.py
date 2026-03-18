import argparse
import os
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

from hsi_compression.data import build_dataloader, build_dataset
from hsi_compression.engine import fit
from hsi_compression.losses import build_loss
from hsi_compression.models.registry import build_model
from hsi_compression.paths import checkpoints_dir, ensure_artifact_dirs
from hsi_compression.utils import (
    cleanup_distributed,
    get_git_short_hash,
    is_git_dirty,
    is_main_process,
    load_config,
    load_project_env,
    set_seed,
    setup_distributed,
)
from hsi_compression.utils.wandb_utils import init_wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Train HSI compression model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def main():
    load_project_env()
    args = parse_args()

    distributed, rank, world_size, local_rank = setup_distributed()

    try:
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
            if is_main_process():
                print(f"Error: dataset_root does not exist: {dataset_root}")
            sys.exit(1)

        seed = experiment_cfg.get("seed", 42)
        set_seed(seed + rank)
        ensure_artifact_dirs()

        device = (
            torch.device(f"cuda:{local_rank}")
            if distributed
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )

        if is_main_process():
            print(f"Device: {device} | World size: {world_size}")

        difficulty = data_cfg.get("difficulty", "easy")
        batch_size = data_cfg.get("batch_size", 16)
        num_workers = data_cfg.get("num_workers", 4)
        drop_invalid = data_cfg.get("drop_invalid_channels", True)
        train_subset = data_cfg.get("train_subset_size", None)
        val_subset = data_cfg.get("val_subset_size", None)

        epochs = training_cfg.get("epochs", 500)
        lr = training_cfg.get("lr", 1e-4)
        loss_name = training_cfg.get("loss_name", "mse")
        grad_clip = training_cfg.get("grad_clip_max_norm", 1.0)
        quantization_bits = training_cfg.get("quantization_bits", 8)

        scheduler_cfg = training_cfg.get("scheduler", {})

        if is_main_process():
            print(
                f"\nDataset: {difficulty} split | "
                f"drop_invalid_channels={drop_invalid} | "
                f"normalization: global min-max [0,1]"
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

        sample_x = (train_ds[0] if not train_subset else train_ds.dataset[0])["x"]
        num_input_bands = sample_x.shape[0]

        if is_main_process():
            print(f"Input bands: {num_input_bands} | Train: {len(train_ds)} | Val: {len(val_ds)}")

        train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None

        train_loader = build_dataloader(
            train_ds,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            sampler=train_sampler,
        )
        val_loader = build_dataloader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        model_name = model_cfg.get("model_name")
        model_kwargs = model_cfg.get("model_kwargs", {})

        model_kwargs["in_channels"] = num_input_bands

        model = build_model(
            model_name=model_name,
            in_channels=num_input_bands,
            **{k: v for k, v in model_kwargs.items() if k != "in_channels"},
        ).to(device)

        if distributed:
            model = DDP(model, device_ids=[local_rank])

        if is_main_process():
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model: {model_name} | Parameters: {n_params:,}")

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

        def _run_training(logger=None):
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
                show_progress=is_main_process(),
                train_sampler=train_sampler,
                grad_clip_max_norm=grad_clip,
                num_input_bands=num_input_bands,
                quantization_bits=quantization_bits,
            )

        if use_wandb and is_main_process():
            with init_wandb(
                project=logging_cfg.get("project", "hsi-compression"),
                run_name=args.run_name or exp_name,
                config=run_cfg,
            ) as run:
                result = _run_training(logger=run)
        else:
            result = _run_training(logger=None)

        if is_main_process():
            print(f"\n{'=' * 60}")
            print("Training completed.")
            print(f"Best val/psnr: {result['best_val_psnr']:.2f} dB")
            print(f"Checkpoint: {ckpt_path}")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
