import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hsi_compression.downstream.hyperview2 import (
    Hyperview2CompressionDataset,
    build_hyperview2_samples,
    collate_compression_batch,
    resolve_hyperview2_root,
    split_samples,
)
from hsi_compression.downstream.hyperview2_compression_eval import (
    apply_input_spectral_mapping,
    build_spectral_mapping,
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


def _split_main_aux_parameters(model):
    main_parameters = []
    aux_parameters = []
    seen_main = set()
    seen_aux = set()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".quantiles"):
            aux_parameters.append(parameter)
            seen_aux.add(id(parameter))
        else:
            main_parameters.append(parameter)
            seen_main.add(id(parameter))

    overlap = seen_main & seen_aux
    if overlap:
        raise RuntimeError("Main and aux optimizer parameter groups must be disjoint")
    return main_parameters, aux_parameters


def _load_pretrained_weights(model, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])


def _build_hyperview2_collate(
    spectral_mapping: dict | None,
    pad_multiple: int,
    min_spatial_size: int,
):
    def _collate(batch):
        collated = collate_compression_batch(
            batch,
            pad_multiple=pad_multiple,
            min_spatial_size=min_spatial_size,
        )
        if spectral_mapping is None:
            return collated
        x, mask = apply_input_spectral_mapping(
            collated["x"],
            collated["valid_mask"],
            spectral_mapping,
        )
        collated["x"] = x
        collated["valid_mask"] = mask
        collated["original_shape"] = [
            (int(x.shape[1]), int(shape[-2]), int(shape[-1]))
            for shape in collated["original_shape"]
        ]
        return collated

    return _collate


def _build_hyperview2_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool | None,
    prefetch_factor: int | None,
    seed: int,
    collate_fn,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "generator": generator,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = (
            persistent_workers if persistent_workers is not None else True
        )
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune an HSI compressor on the HYPERVIEW2 PRISMA train split."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument(
        "--wandb-resume",
        type=str,
        choices=("allow", "must", "never", "auto"),
        default=None,
    )
    parser.add_argument("--override-rd-lambda", type=float, default=None)
    parser.add_argument("--override-lr", type=float, default=None)
    parser.add_argument("--override-epochs", type=int, default=None)
    parser.add_argument("--override-experiment-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
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
    if args.override_rd_lambda is not None:
        training_cfg["rd_lambda"] = args.override_rd_lambda
    if args.override_lr is not None:
        training_cfg["lr"] = args.override_lr
    if args.override_epochs is not None:
        training_cfg["epochs"] = args.override_epochs
    cfg["experiment"] = experiment_cfg
    cfg["training"] = training_cfg

    seed = int(experiment_cfg.get("seed", 42))
    set_seed(seed)
    ensure_artifact_dirs()

    dataset_root = Path(
        args.dataset_root
        or os.environ.get("HYPERVIEW2_ROOT")
        or os.environ.get("DATASET_ROOT")
        or "data/hyperview2/HYPERVIEW2"
    )
    hv2_root = resolve_hyperview2_root(dataset_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(training_cfg.get("use_amp", True)) and device.type == "cuda"

    modality = data_cfg.get("modality", "prisma")
    normalization = data_cfg.get("normalization", "reflectance_0_1")
    spectral_mapping_name = data_cfg.get("spectral_mapping", None)
    val_fraction = float(data_cfg.get("val_fraction", 0.2))
    batch_size = int(data_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    persistent_workers = data_cfg.get("persistent_workers", None)
    prefetch_factor = data_cfg.get("prefetch_factor", 2)
    pad_multiple = int(data_cfg.get("pad_multiple", 4))
    min_spatial_size = int(data_cfg.get("min_spatial_size", 4))
    train_subset = data_cfg.get("train_subset_size", None)
    val_subset = data_cfg.get("val_subset_size", None)

    print(f"Device: {device}")
    print("Dataset: HYPERVIEW2 downstream fine-tuning protocol, not HySpecNet reference")
    print(f"HYPERVIEW2 root: {hv2_root}")
    print(
        f"Input modality={modality} | normalization={normalization} | "
        f"spectral_mapping={spectral_mapping_name or 'none'}"
    )
    print(f"AMP enabled: {use_amp}")

    samples = build_hyperview2_samples(hv2_root, modality=modality, split="train")
    train_samples, val_samples = split_samples(samples, val_fraction=val_fraction, seed=seed)
    if train_subset:
        train_samples = train_samples[: min(int(train_subset), len(train_samples))]
    if val_subset:
        val_samples = val_samples[: min(int(val_subset), len(val_samples))]

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
    first = train_ds[0]
    input_channels = int(first["x"].shape[0])
    spectral_mapping = build_spectral_mapping(
        spectral_mapping_name,
        source_root=hv2_root,
        modality=modality,
        input_channels=input_channels,
    )
    model_input_channels = (
        int(spectral_mapping["model_input_channels"])
        if spectral_mapping is not None
        else input_channels
    )
    print(
        f"Input bands: {input_channels} -> model bands: {model_input_channels} | "
        f"Train: {len(train_ds)} | Val: {len(val_ds)}"
    )

    collate_fn = _build_hyperview2_collate(
        spectral_mapping=spectral_mapping,
        pad_multiple=pad_multiple,
        min_spatial_size=min_spatial_size,
    )
    train_loader = _build_hyperview2_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
        collate_fn=collate_fn,
    )
    val_loader = _build_hyperview2_loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
        collate_fn=collate_fn,
    )

    model_name = model_cfg.get("model_name")
    model_kwargs = {
        key: value
        for key, value in model_cfg.get("model_kwargs", {}).items()
        if key != "in_channels"
    }
    model = build_model(
        model_name=model_name,
        in_channels=model_input_channels,
        **model_kwargs,
    ).to(device)

    if args.pretrained:
        pretrained_path = Path(args.pretrained)
        if not pretrained_path.exists():
            print(f"Error: pretrained checkpoint does not exist: {pretrained_path}")
            sys.exit(1)
        print(f"Loading pretrained weights strictly from: {pretrained_path}")
        _load_pretrained_weights(model, pretrained_path, device=device)

    if hasattr(model, "update"):
        model.update(force=True)

    n_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Model: {model_name} | Parameters: {n_params:,}")
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(
            f"GPU memory after model: allocated={allocated:.2f} GiB | reserved={reserved:.2f} GiB"
        )

    loss_name = training_cfg.get("loss_name", "masked_mse")
    loss_kwargs = training_cfg.get("loss_kwargs", {})
    if loss_name == "rate_distortion":
        rd_lambda = float(training_cfg.get("rd_lambda", 0.01))
        distortion_metric = training_cfg.get("distortion_metric", "masked_mse")
        distortion_kwargs = training_cfg.get(
            "distortion_kwargs",
            loss_kwargs.get("distortion_kwargs", {}),
        )
        loss_fn = build_loss(
            "rate_distortion",
            lmbda=rd_lambda,
            distortion_metric=distortion_metric,
            distortion_kwargs=distortion_kwargs,
        )
        print(
            f"Loss: Rate-Distortion (lambda={rd_lambda}, D={distortion_metric}, "
            f"D_kwargs={distortion_kwargs})"
        )
    else:
        loss_fn = build_loss(loss_name, **loss_kwargs)
        print(f"Loss: {loss_name}")

    lr = float(training_cfg.get("lr", 1e-4))
    main_parameters, aux_parameters = _split_main_aux_parameters(model)
    optimizer = torch.optim.Adam(main_parameters, lr=lr)
    aux_optimizer = torch.optim.Adam(aux_parameters, lr=1e-3) if aux_parameters else None
    epochs = int(training_cfg.get("epochs", 100))
    scheduler_cfg = training_cfg.get("scheduler", {})
    scheduler = None
    if scheduler_cfg.get("enabled", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.get("T_max", epochs)),
            eta_min=float(scheduler_cfg.get("eta_min", 1e-6)),
        )

    exp_name = experiment_cfg.get("name", "hyperview2_compressor")
    ckpt_path = checkpoints_dir() / f"{exp_name}_best.pt"
    cfg["data"] = {
        **data_cfg,
        "dataset": "hyperview2",
        "dataset_root": str(hv2_root),
        "split_source": "train_gt.csv fixed internal train/validation split",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "input_channels": input_channels,
        "model_input_channels": model_input_channels,
        "spectral_mapping_payload": spectral_mapping,
    }
    run_cfg = {
        **cfg,
        "num_input_bands": model_input_channels,
        "git_hash": get_git_short_hash(),
        "git_dirty": is_git_dirty(),
        "model_num_params": n_params,
        "amp_enabled": use_amp,
        "pretrained": str(args.pretrained) if args.pretrained else None,
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
            grad_clip_max_norm=float(training_cfg.get("grad_clip_max_norm", 1.0)),
            resume=args.resume,
            sam_every_n_epochs=int(training_cfg.get("sam_every_n_epochs", 10)),
            use_amp=use_amp,
            fast_train_metrics=bool(training_cfg.get("fast_train_metrics", False)),
        )

    use_wandb = bool(logging_cfg.get("use_wandb", False)) and not args.disable_wandb
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


if __name__ == "__main__":
    main()
