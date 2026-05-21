#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from hsi_compression.downstream import (
    HYPERVIEW2_TARGET_COLUMNS,
    Hyperview2CompressionDataset,
    build_hyperview2_samples,
    collate_compression_batch,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a HYPERVIEW2-compatible lossy compressor on the same train/val sample IDs "
            "as a downstream regressor checkpoint."
        )
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--downstream-checkpoint", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--id-column", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument(
        "--wandb-resume",
        type=str,
        default=None,
        choices=("allow", "must", "never", "auto"),
    )
    parser.add_argument("--override-lr", type=float, default=None)
    parser.add_argument("--override-epochs", type=int, default=None)
    parser.add_argument("--override-experiment-name", type=str, default=None)
    return parser.parse_args()


def _load_downstream_split(path: Path) -> tuple[dict[str, Any], set[str], set[str]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    train_ids = {str(value) for value in checkpoint.get("train_ids", [])}
    val_ids = {str(value) for value in checkpoint.get("val_ids", [])}
    if not train_ids or not val_ids:
        raise ValueError(
            "Downstream checkpoint must contain non-empty train_ids and val_ids. "
            "This script uses them to avoid training the compressor on downstream validation data."
        )
    return checkpoint, train_ids, val_ids


def _select_samples(samples, wanted: set[str], split_name: str):
    selected = [sample for sample in samples if sample.sample_id in wanted]
    missing = sorted(wanted - {sample.sample_id for sample in selected})
    if not selected:
        preview = ", ".join(missing[:8])
        raise ValueError(f"No HYPERVIEW2 samples matched {split_name} ids. Missing: {preview}")
    if missing:
        preview = ", ".join(missing[:8])
        print(f"Warning: {len(missing)} {split_name} ids were not paired with arrays: {preview}")
    return selected


def _build_loader(
    dataset: Hyperview2CompressionDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    pad_multiple: int,
    min_spatial_size: int,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": partial(
            collate_compression_batch,
            pad_multiple=pad_multiple,
            min_spatial_size=min_spatial_size,
        ),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def main() -> int:
    load_project_env()
    args = parse_args()
    cfg = load_config(args.config)
    experiment_cfg = cfg.get("experiment", {})
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    logging_cfg = cfg.get("logging", {})

    if args.override_experiment_name is not None:
        experiment_cfg["name"] = args.override_experiment_name
    if args.override_lr is not None:
        training_cfg["lr"] = args.override_lr
    if args.override_epochs is not None:
        training_cfg["epochs"] = args.override_epochs
    cfg["experiment"] = experiment_cfg
    cfg["training"] = training_cfg

    downstream_ckpt, train_ids, val_ids = _load_downstream_split(args.downstream_checkpoint)
    downstream_cfg = downstream_ckpt.get("config", {})
    target_columns = tuple(downstream_cfg.get("target_columns", HYPERVIEW2_TARGET_COLUMNS))
    modality = data_cfg.get("modality", "from_downstream")
    if modality == "from_downstream":
        modality = downstream_cfg.get("modality", "prisma")
    normalization = data_cfg.get("normalization", "from_downstream")
    if normalization == "from_downstream":
        normalization = downstream_cfg.get("normalization", "percentile")
    data_cfg["modality"] = modality
    data_cfg["normalization"] = normalization
    cfg["data"] = data_cfg

    dataset_root = Path(args.dataset_root or os.environ.get("HV2_ROOT") or ".").expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(f"HYPERVIEW2 dataset root does not exist: {dataset_root}")

    seed = int(experiment_cfg.get("seed", 42))
    set_seed(seed)
    ensure_artifact_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(training_cfg.get("use_amp", True) and device.type == "cuda")
    print(f"Device: {device}")
    print(
        "\nDataset: HYPERVIEW2 downstream-aligned compression protocol | "
        f"modality={modality} | normalization={normalization}"
    )
    print(
        f"Split source: {args.downstream_checkpoint} | train_ids={len(train_ids)} | "
        f"val_ids={len(val_ids)}"
    )
    print(f"AMP enabled: {use_amp}")

    samples = build_hyperview2_samples(
        dataset_root=dataset_root,
        modality=modality,
        labels_csv=args.labels_csv,
        id_column=args.id_column,
        target_columns=target_columns,
        max_samples=None,
    )
    train_samples = _select_samples(samples, train_ids, "train")
    val_samples = _select_samples(samples, val_ids, "val")
    train_subset = data_cfg.get("train_subset_size")
    val_subset = data_cfg.get("val_subset_size")
    if train_subset:
        train_samples = train_samples[: int(train_subset)]
    if val_subset:
        val_samples = val_samples[: int(val_subset)]

    train_ds = Hyperview2CompressionDataset(
        train_samples,
        modality=modality,
        normalization=normalization,
    )
    val_ds = Hyperview2CompressionDataset(
        val_samples,
        modality=modality,
        normalization=normalization,
    )
    sample = train_ds[0]
    num_input_bands = int(sample["x"].shape[0])
    print(f"Input bands: {num_input_bands} | Train: {len(train_ds)} | Val: {len(val_ds)}")
    print("Variable spatial shapes are padded only inside each batch.")

    batch_size = int(data_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    persistent_workers = bool(data_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = data_cfg.get("prefetch_factor", 2)
    pad_multiple = int(data_cfg.get("pad_multiple", 4))
    min_spatial_size = int(data_cfg.get("min_spatial_size", 4))

    train_loader = _build_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pad_multiple=pad_multiple,
        min_spatial_size=min_spatial_size,
    )
    val_loader = _build_loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pad_multiple=pad_multiple,
        min_spatial_size=min_spatial_size,
    )

    model_name = model_cfg.get("model_name")
    model_kwargs = {
        key: value
        for key, value in model_cfg.get("model_kwargs", {}).items()
        if key != "in_channels"
    }
    model = build_model(model_name=model_name, in_channels=num_input_bands, **model_kwargs).to(
        device
    )
    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Model: {model_name} | Parameters: {n_params:,}")

    epochs = int(training_cfg.get("epochs", 100))
    lr = float(training_cfg.get("lr", 1e-4))
    loss_name = training_cfg.get("loss_name", "masked_mse")
    loss_kwargs = training_cfg.get("loss_kwargs", {})
    grad_clip = float(training_cfg.get("grad_clip_max_norm", 1.0))
    sam_every = int(training_cfg.get("sam_every_n_epochs", 10))
    fast_train_metrics = bool(training_cfg.get("fast_train_metrics", True))
    scheduler_cfg = training_cfg.get("scheduler", {})

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    aux_parameters = [
        param for name, param in model.named_parameters() if name.endswith(".quantiles")
    ]
    aux_optimizer = torch.optim.Adam(aux_parameters, lr=1e-3) if aux_parameters else None
    scheduler = None
    if scheduler_cfg.get("enabled", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.get("T_max", epochs)),
            eta_min=float(scheduler_cfg.get("eta_min", 1e-6)),
        )

    if loss_name == "rate_distortion":
        rd_lambda = float(training_cfg.get("rd_lambda", 0.01))
        distortion_metric = training_cfg.get("distortion_metric", "masked_mse")
        loss_fn = build_loss(
            "rate_distortion",
            lmbda=rd_lambda,
            distortion_metric=distortion_metric,
        )
        print(f"Loss: Rate-Distortion (lambda={rd_lambda}, D={distortion_metric})")
    else:
        loss_fn = build_loss(loss_name, **loss_kwargs)
        print(f"Loss: {loss_name}")
    # HYPERVIEW2 batches are padded to support variable tiny spatial shapes. Checkpoint
    # selection must follow the masked validation objective, not unmasked padded PSNR.
    loss_fn.select_by_loss = True

    exp_name = experiment_cfg.get("name", "hyperview2_compressor")
    ckpt_path = checkpoints_dir() / f"{exp_name}_best.pt"
    run_cfg = {
        **cfg,
        "num_input_bands": num_input_bands,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "downstream_checkpoint": str(args.downstream_checkpoint),
        "git_hash": get_git_short_hash(),
        "git_dirty": is_git_dirty(),
        "model_num_params": n_params,
        "amp_enabled": use_amp,
        "protocol": "hyperview2_downstream_aligned_unsupervised_compressor",
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
            config={**cfg, "downstream_checkpoint": str(args.downstream_checkpoint)},
            logger=logger,
            scheduler=scheduler,
            show_progress=True,
            train_sampler=None,
            grad_clip_max_norm=grad_clip,
            sam_every_n_epochs=sam_every,
            resume=args.resume,
            use_amp=use_amp,
            fast_train_metrics=fast_train_metrics,
        )

    use_wandb = bool(logging_cfg.get("use_wandb", False) and not args.disable_wandb)
    if use_wandb:
        with init_wandb(
            project=logging_cfg.get("project", "hsi-compression-paper"),
            run_name=args.run_name or exp_name,
            config=run_cfg,
            run_id=args.wandb_run_id,
            resume=args.wandb_resume,
        ) as run:
            _run(logger=run)
    else:
        _run(logger=None)

    print(f"Best checkpoint path: {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
