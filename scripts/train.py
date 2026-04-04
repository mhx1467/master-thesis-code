import argparse
import os
import sys
import time
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
    prefer_npy = data_cfg.get("prefer_npy", True)
    npy_mmap = data_cfg.get("npy_mmap", False)
    pin_memory = data_cfg.get("pin_memory", True)
    persistent_workers = data_cfg.get("persistent_workers", None)
    prefetch_factor = data_cfg.get("prefetch_factor", 2)
    train_subset = data_cfg.get("train_subset_size", None)
    val_subset = data_cfg.get("val_subset_size", None)
    use_amp = training_cfg.get("use_amp", True) and device.type == "cuda"

    print(
        f"\nDataset: {difficulty} split | drop_invalid_channels={drop_invalid} | normalization: global min-max [0,1]"
    )
    print(f"AMP enabled: {use_amp}")

    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(
            f"GPU memory before data/model: free={free_bytes / 1024**3:.2f} GiB / total={total_bytes / 1024**3:.2f} GiB"
        )

    train_ds = build_dataset(
        dataset_root=dataset_root,
        split_name="train",
        difficulty=difficulty,
        normalized=True,
        return_mask=True,
        drop_invalid_channels=drop_invalid,
        prefer_npy=prefer_npy,
        npy_mmap=npy_mmap,
    )
    val_ds = build_dataset(
        dataset_root=dataset_root,
        split_name="val",
        difficulty=difficulty,
        normalized=True,
        return_mask=True,
        drop_invalid_channels=drop_invalid,
        prefer_npy=prefer_npy,
        npy_mmap=npy_mmap,
    )

    if train_subset:
        train_ds = Subset(train_ds, list(range(min(train_subset, len(train_ds)))))
    if val_subset:
        val_ds = Subset(val_ds, list(range(min(val_subset, len(val_ds)))))

    sample = train_ds[0]
    sample_tensor = sample["x"] if isinstance(sample, dict) else torch.as_tensor(sample)
    num_input_bands = int(sample_tensor.shape[0])
    print(f"Input bands: {num_input_bands} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_base_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    val_base_ds = val_ds.dataset if isinstance(val_ds, Subset) else val_ds
    if hasattr(train_base_ds, "using_npy"):
        source = ".npy" if bool(getattr(train_base_ds, "using_npy", False)) else ".TIF"
        print(f"Train source: {source} | npy_mmap={npy_mmap}")
    if hasattr(val_base_ds, "using_npy"):
        source = ".npy" if bool(getattr(val_base_ds, "using_npy", False)) else ".TIF"
        print(f"Val source:   {source} | npy_mmap={npy_mmap}")

    debug_loader_timing = data_cfg.get("debug_loader_timing", False)
    if debug_loader_timing:
        warmup_batches = data_cfg.get("debug_loader_warmup_batches", 4)
        timed_batches = data_cfg.get("debug_loader_timed_batches", 16)
        total_batches = max(1, warmup_batches + timed_batches)
        timing_loader = build_dataloader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        iterator = iter(timing_loader)
        load_times = []
        sync_times = []
        for step in range(total_batches):
            t0 = time.perf_counter()
            batch = next(iterator)
            t1 = time.perf_counter()
            x = batch["x"] if isinstance(batch, dict) else batch
            if device.type == "cuda":
                s0 = time.perf_counter()
                _ = x.to(device, non_blocking=True)
                torch.cuda.synchronize(device)
                s1 = time.perf_counter()
                sync_times.append(s1 - s0)
            if step >= warmup_batches:
                load_times.append(t1 - t0)

        if load_times:
            mean_load = sum(load_times) / len(load_times)
            p95_idx = max(0, min(len(load_times) - 1, int(0.95 * len(load_times)) - 1))
            p95_load = sorted(load_times)[p95_idx]
            print(
                f"Loader timing ({len(load_times)} batches): mean={mean_load * 1000:.1f}ms, p95={p95_load * 1000:.1f}ms"
            )
        if sync_times:
            mean_sync = sum(sync_times) / len(sync_times)
            print(f"Host->GPU copy timing: mean={mean_sync * 1000:.1f}ms")

    train_loader = build_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
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
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(
            f"GPU memory after model: allocated={allocated:.2f} GiB | reserved={reserved:.2f} GiB"
        )

    epochs = training_cfg.get("epochs", 500)
    lr = training_cfg.get("lr", 1e-4)
    loss_name = training_cfg.get("loss_name", "masked_mse")
    grad_clip = training_cfg.get("grad_clip_max_norm", 1.0)
    sam_every = training_cfg.get("sam_every_n_epochs", 10)
    scheduler_cfg = training_cfg.get("scheduler", {})

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    aux_parameters = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]
    aux_optimizer = torch.optim.Adam(aux_parameters, lr=1e-3) if aux_parameters else None
    scheduler = None
    if scheduler_cfg.get("enabled", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("T_max", epochs),
            eta_min=scheduler_cfg.get("eta_min", 1e-6),
        )

    if loss_name == "rate_distortion":
        rd_lambda = training_cfg.get("rd_lambda", 0.01)
        distortion_metric = training_cfg.get("distortion_metric", "masked_mse")
        loss_fn = build_loss(
            "rate_distortion", lmbda=rd_lambda, distortion_metric=distortion_metric
        )
        print(f"Loss: Rate-Distortion (lambda={rd_lambda}, D={distortion_metric})")
    else:
        loss_fn = build_loss(loss_name)
    exp_name = experiment_cfg.get("name", "experiment")
    ckpt_path = checkpoints_dir() / f"{exp_name}_best.pt"

    use_wandb = logging_cfg.get("use_wandb", False) and not args.disable_wandb
    run_cfg = {
        **cfg,
        "num_input_bands": num_input_bands,
        "git_hash": get_git_short_hash(),
        "git_dirty": is_git_dirty(),
        "model_num_params": n_params,
        "amp_enabled": use_amp,
    }

    def _run(logger=None):
        return fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            aux_optimizer=aux_optimizer,
            device=device,
            epochs=epochs,
            checkpoint_path=ckpt_path,
            config=cfg,
            logger=logger,
            scheduler=scheduler,
            show_progress=True,
            train_sampler=None,
            grad_clip_max_norm=grad_clip,
            sam_every_n_epochs=sam_every,
            resume=args.resume,
            use_amp=use_amp,
        )

    if use_wandb:
        project = logging_cfg.get("project", "hsi-compression-paper")
        run_name = args.run_name or exp_name
        with init_wandb(project=project, run_name=run_name, config=run_cfg) as run:
            _run(logger=run)
    else:
        _run(logger=None)


if __name__ == "__main__":
    main()
