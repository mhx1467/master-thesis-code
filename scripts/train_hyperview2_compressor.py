import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
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
    _load_state_dict_allowing_entropy_runtime_buffers,
    apply_input_spectral_mapping,
    build_spectral_mapping,
    model_input_wavelengths_for_mapping,
)
from hsi_compression.engine import fit
from hsi_compression.losses import build_loss
from hsi_compression.models.registry import build_model
from hsi_compression.paths import checkpoints_dir, ensure_artifact_dirs, logs_dir
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


def _load_pretrained_weights(model, checkpoint_path: Path, device: torch.device) -> list[str]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return _load_state_dict_allowing_entropy_runtime_buffers(model, checkpoint["model_state_dict"])


def _limit_samples(samples, subset_size: int | None, seed: int):
    samples = list(samples)
    if subset_size is None:
        return samples
    subset_size = int(subset_size)
    if subset_size <= 0:
        raise ValueError("subset_size must be positive when provided")
    if subset_size >= len(samples):
        return samples
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    selected = sorted(indices[:subset_size])
    return [samples[index] for index in selected]


def _compile_regex_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern {pattern!r}: {exc}") from exc
    return compiled


def _configure_trainable_parameters(
    model: torch.nn.Module,
    *,
    trainable_regex: list[str],
    freeze_regex: list[str],
) -> dict:
    trainable_patterns = _compile_regex_patterns(trainable_regex)
    freeze_patterns = _compile_regex_patterns(freeze_regex)
    trainable_names = []
    frozen_names = []
    total_parameters = 0
    trainable_parameters = 0

    for name, parameter in model.named_parameters():
        total_parameters += parameter.numel()
        if trainable_patterns:
            parameter.requires_grad = any(pattern.search(name) for pattern in trainable_patterns)
        if freeze_patterns and any(pattern.search(name) for pattern in freeze_patterns):
            parameter.requires_grad = False
        if parameter.requires_grad:
            trainable_names.append(name)
            trainable_parameters += parameter.numel()
        else:
            frozen_names.append(name)

    if not trainable_names:
        raise ValueError(
            "No trainable parameters left after applying --trainable-regex/--freeze-regex."
        )

    return {
        "trainable_regex": list(trainable_regex),
        "freeze_regex": list(freeze_regex),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": total_parameters - trainable_parameters,
        "trainable_tensors": len(trainable_names),
        "frozen_tensors": len(frozen_names),
        "trainable_names_preview": trainable_names[:50],
        "frozen_names_preview": frozen_names[:50],
    }


def _build_hyperview2_collate(
    spectral_mapping: dict | None,
    pad_multiple: int,
    min_spatial_size: int,
    model_wavelengths: list[float] | None = None,
):
    def _collate(batch):
        collated = collate_compression_batch(
            batch,
            pad_multiple=pad_multiple,
            min_spatial_size=min_spatial_size,
        )
        if spectral_mapping is not None:
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
        if model_wavelengths is not None:
            wavelengths = torch.as_tensor(model_wavelengths, dtype=torch.float32)
            collated["wavelengths"] = wavelengths
            collated["output_wavelengths"] = wavelengths
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
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail instead of silently falling back to CPU when CUDA is unavailable.",
    )
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
    parser.add_argument("--override-train-subset-size", type=int, default=None)
    parser.add_argument("--override-val-subset-size", type=int, default=None)
    parser.add_argument(
        "--trainable-regex",
        action="append",
        default=[],
        help=(
            "Regex selecting parameters that remain trainable. Repeatable. "
            "When omitted, all parameters stay trainable unless --freeze-regex matches."
        ),
    )
    parser.add_argument(
        "--freeze-regex",
        action="append",
        default=[],
        help="Regex selecting parameters to freeze after optional --trainable-regex filtering.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    load_project_env()
    args = parse_args()

    cfg = load_config(args.config)
    experiment_cfg = cfg.get("experiment", {})
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    logging_cfg = cfg.get("logging", {})
    model_name = model_cfg.get("model_name")

    if args.override_experiment_name is not None:
        experiment_cfg["name"] = args.override_experiment_name
    if args.override_rd_lambda is not None:
        training_cfg["rd_lambda"] = args.override_rd_lambda
    if args.override_lr is not None:
        training_cfg["lr"] = args.override_lr
    if args.override_epochs is not None:
        training_cfg["epochs"] = args.override_epochs
    if args.override_train_subset_size is not None:
        data_cfg["train_subset_size"] = args.override_train_subset_size
    if args.override_val_subset_size is not None:
        data_cfg["val_subset_size"] = args.override_val_subset_size
    cfg["experiment"] = experiment_cfg
    cfg["data"] = data_cfg
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
    cuda_available = torch.cuda.is_available()
    if args.require_cuda and not cuda_available:
        print("Error: CUDA is required for this run, but torch.cuda.is_available() is False.")
        sys.exit(1)
    device = torch.device("cuda" if cuda_available else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
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
    subset_seed = int(data_cfg.get("subset_seed", seed))

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    else:
        print("Warning: running on CPU. This is intended only for small smoke tests.")
    print("Dataset: HYPERVIEW2 downstream fine-tuning protocol, not HySpecNet reference")
    print(f"HYPERVIEW2 root: {hv2_root}")
    print(
        f"Input modality={modality} | normalization={normalization} | "
        f"spectral_mapping={spectral_mapping_name or 'none'}"
    )
    print(f"AMP enabled: {use_amp}")

    samples = build_hyperview2_samples(hv2_root, modality=modality, split="train")
    train_samples, val_samples = split_samples(samples, val_fraction=val_fraction, seed=seed)
    train_samples = _limit_samples(train_samples, train_subset, seed=subset_seed)
    val_samples = _limit_samples(val_samples, val_subset, seed=subset_seed + 1)

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
    model_input_wavelengths = (
        model_input_wavelengths_for_mapping(
            spectral_mapping,
            source_root=hv2_root,
            modality=modality,
            input_channels=input_channels,
        )
        if model_name == "hierarchical_spectral_mamba_sensor_aware"
        else None
    )
    print(
        f"Input bands: {input_channels} -> model bands: {model_input_channels} | "
        f"Train: {len(train_ds)} | Val: {len(val_ds)}"
    )
    if model_input_wavelengths is not None:
        print(
            "Model wavelengths: "
            f"{len(model_input_wavelengths)} values from {min(model_input_wavelengths):.2f} "
            f"to {max(model_input_wavelengths):.2f}"
        )

    collate_fn = _build_hyperview2_collate(
        spectral_mapping=spectral_mapping,
        pad_multiple=pad_multiple,
        min_spatial_size=min_spatial_size,
        model_wavelengths=model_input_wavelengths,
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
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

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
        print(f"Loading pretrained weights from: {pretrained_path}")
        skipped = _load_pretrained_weights(model, pretrained_path, device=device)
        if skipped:
            print("Ignored entropy runtime buffers restored by model.update():")
            for name in skipped:
                print(f"  {name}")

    if hasattr(model, "update"):
        model.update(force=True)

    trainability = _configure_trainable_parameters(
        model,
        trainable_regex=args.trainable_regex,
        freeze_regex=args.freeze_regex,
    )
    n_params = trainability["trainable_parameters"]
    print(
        f"Model: {model_name} | Trainable parameters: {n_params:,} / "
        f"{trainability['total_parameters']:,}"
    )
    if args.trainable_regex or args.freeze_regex:
        print(
            "Trainability filter: "
            f"trainable_regex={args.trainable_regex or 'all'} | "
            f"freeze_regex={args.freeze_regex or 'none'} | "
            f"trainable_tensors={trainability['trainable_tensors']} | "
            f"frozen_tensors={trainability['frozen_tensors']}"
        )
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
        "model_input_wavelengths": model_input_wavelengths,
        "spectral_mapping_payload": spectral_mapping,
    }
    run_cfg = {
        **cfg,
        "num_input_bands": model_input_channels,
        "git_hash": get_git_short_hash(),
        "git_dirty": is_git_dirty(),
        "model_num_params": n_params,
        "model_total_params": trainability["total_parameters"],
        "trainability": trainability,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
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
            result = _run(logger=run)
    else:
        result = _run(logger=None)

    summary_path = logs_dir() / f"{exp_name}_train_summary.json"
    summary_payload = {
        "experiment_name": exp_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_best": str(ckpt_path),
        "checkpoint_last": str(ckpt_path.parent / f"{exp_name}_last.pt"),
        "run_config": run_cfg,
        "best_val_loss": result.get("best_val_loss"),
        "best_val_ref_psnr": result.get("best_val_ref_psnr"),
        "history": result.get("history", []),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"Saved training summary: {summary_path}")


if __name__ == "__main__":
    main()
