import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hsi_compression.data import build_dataset
from hsi_compression.engine.checkpointing import load_checkpoint
from hsi_compression.engine.model_io import call_model_forward
from hsi_compression.metrics import (
    masked_mae,
    masked_mse,
    masked_psnr,
    masked_sam_deg,
    masked_sid,
)
from hsi_compression.models.registry import build_model
from hsi_compression.utils import load_project_env
from hsi_compression.visualization import choose_evenly_spaced_rgb_bands

DEFAULT_CHECKPOINTS = (
    ("MSE", "artifacts/checkpoints/hierarchical_spectral_mamba_ae_recon_latent96_best.pt"),
    (
        "RD_0.0003",
        "artifacts/checkpoints/hierarchical_spectral_mamba_ae_k4_spatial_rd_lambda_0_01_best.pt",
    ),
    (
        "RD_0.0005",
        "artifacts/checkpoints/hierarchical_spectral_mamba_ae_k4_spatial_rd_lambda_0_0005_best.pt",
    ),
)


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path


@dataclass
class LoadedModel:
    label: str
    path: Path
    model: torch.nn.Module
    config: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export qualitative HSI reconstruction comparisons for fixed checkpoints: "
            "pseudo-RGB, error maps, spectra, and per-patch metrics."
        )
    )
    parser.add_argument("dataset_root", nargs="?", default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--selection-mode", choices=("fixed", "error_quantiles"), default="fixed")
    parser.add_argument(
        "--sample-indices",
        default=None,
        help="Comma-separated dataset indices. If omitted in fixed mode, use 0, mid, last.",
    )
    parser.add_argument(
        "--selection-pool-size",
        type=int,
        default=128,
        help="Number of leading split samples scanned in error_quantiles mode.",
    )
    parser.add_argument(
        "--selection-quantiles",
        default="0.5,0.9,1.0",
        help="Comma-separated error quantiles for error_quantiles mode.",
    )
    parser.add_argument(
        "--selection-checkpoint-label",
        default="RD_0.0005",
        help="Checkpoint label used to rank samples in error_quantiles mode.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Checkpoint as LABEL=PATH. Repeat to override the default MSE/RD set.",
    )
    parser.add_argument(
        "--rgb-bands",
        default=None,
        help="Comma-separated RGB band indices, e.g. 150,100,50. Default: evenly spaced.",
    )
    parser.add_argument("--num-spectra", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--save-pdf", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _parse_csv_ints(value: str) -> list[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("Expected at least one integer")
    return out


def _parse_csv_floats(value: str) -> list[float]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise ValueError("Expected at least one float")
    return out


def _parse_checkpoints(values: list[str] | None) -> list[CheckpointSpec]:
    raw = values or [f"{label}={path}" for label, path in DEFAULT_CHECKPOINTS]
    specs = []
    seen = set()
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Checkpoint must use LABEL=PATH format, got: {item}")
        label, path = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty checkpoint label in: {item}")
        if label in seen:
            raise ValueError(f"Duplicate checkpoint label: {label}")
        seen.add(label)
        specs.append(CheckpointSpec(label=label, path=Path(path.strip())))
    return specs


def _call_model_forward(model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor | None) -> dict:
    outputs = call_model_forward(model, x, mask)
    if not isinstance(outputs, dict) or "x_hat" not in outputs:
        raise RuntimeError("Model output must be a dict containing 'x_hat'.")
    return outputs


def _load_model(spec: CheckpointSpec, in_channels: int, device: torch.device) -> LoadedModel:
    if not spec.path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {spec.path}")
    raw = torch.load(spec.path, map_location="cpu", weights_only=False)
    cfg = raw.get("config", {})
    model_section = cfg.get("model", {})
    model_name = model_section.get("model_name")
    if model_name is None:
        raise RuntimeError(f"Checkpoint {spec.path} does not contain config.model.model_name")
    model_kwargs = {
        k: v for k, v in model_section.get("model_kwargs", {}).items() if k != "in_channels"
    }
    model = build_model(model_name=model_name, in_channels=in_channels, **model_kwargs).to(device)
    load_checkpoint(path=spec.path, model=model, map_location=device)
    if hasattr(model, "update"):
        model.update(force=True)
    model.eval()
    return LoadedModel(label=spec.label, path=spec.path, model=model, config=cfg)


def _safe_name(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    out = "".join(ch if ch in allowed else "_" for ch in value)
    return out.strip("._-") or "sample"


def _valid_pixel_mask(mask: torch.Tensor | None, h: int, w: int) -> np.ndarray:
    if mask is None:
        return np.ones((h, w), dtype=bool)
    mask_np = mask.detach().cpu().bool().numpy()
    if mask_np.ndim == 3:
        return mask_np.any(axis=0)
    if mask_np.ndim == 2:
        return mask_np
    raise ValueError(f"Unexpected mask shape: {mask_np.shape}")


def _reference_rgb_params(
    x_chw: np.ndarray,
    bands: tuple[int, int, int],
    valid_pixels: np.ndarray,
    percentile_stretch: tuple[float, float] = (2.0, 98.0),
) -> list[tuple[float, float]]:
    p_low, p_high = percentile_stretch
    params = []
    for band in bands:
        channel = x_chw[band]
        valid = channel[valid_pixels]
        if valid.size == 0:
            params.append((0.0, 1.0))
            continue
        lo = float(np.percentile(valid, p_low))
        hi = float(np.percentile(valid, p_high))
        if hi <= lo:
            hi = lo + 1e-8
        params.append((lo, hi))
    return params


def _rgb_with_params(
    x_chw: np.ndarray,
    bands: tuple[int, int, int],
    params: list[tuple[float, float]],
    valid_pixels: np.ndarray,
) -> np.ndarray:
    channels = []
    for band, (lo, hi) in zip(bands, params, strict=True):
        channel = np.clip((x_chw[band] - lo) / (hi - lo), 0.0, 1.0)
        channels.append(channel.astype(np.float32))
    rgb = np.stack(channels, axis=-1)
    rgb[~valid_pixels] = 0.0
    return rgb


def _error_map(x_hat: np.ndarray, x: np.ndarray, valid_pixels: np.ndarray) -> np.ndarray:
    err = np.mean(np.abs(x_hat - x), axis=0)
    err = err.astype(np.float32)
    err[~valid_pixels] = np.nan
    return err


def _sample_metrics(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor | None) -> dict:
    x_b = x.unsqueeze(0)
    x_hat_b = x_hat.unsqueeze(0)
    mask_b = torch.ones_like(x_b, dtype=torch.bool) if mask is None else mask.unsqueeze(0)
    return {
        "masked_mse": float(masked_mse(x_hat_b, x_b, mask_b).item()),
        "masked_mae": float(masked_mae(x_hat_b, x_b, mask_b).item()),
        "masked_psnr": float(masked_psnr(x_hat_b, x_b, mask_b, data_range=1.0).item()),
        "masked_sam_deg": float(masked_sam_deg(x_hat_b, x_b, mask_b).item()),
        "masked_sid": float(masked_sid(x_hat_b, x_b, mask_b).item()),
    }


def _choose_spectrum_coords(
    valid_pixels: np.ndarray,
    anchor_error: np.ndarray,
    n: int,
    seed: int,
) -> list[tuple[int, int]]:
    ys, xs = np.where(valid_pixels)
    if len(ys) == 0:
        raise ValueError("No valid pixels available for spectrum selection")

    coords: list[tuple[int, int]] = []

    center = np.array([(valid_pixels.shape[0] - 1) / 2.0, (valid_pixels.shape[1] - 1) / 2.0])
    distances = (ys - center[0]) ** 2 + (xs - center[1]) ** 2
    center_idx = int(np.argmin(distances))
    coords.append((int(ys[center_idx]), int(xs[center_idx])))

    finite_error = np.where(np.isfinite(anchor_error), anchor_error, -np.inf)
    high_idx = np.unravel_index(int(np.argmax(finite_error)), finite_error.shape)
    coords.append((int(high_idx[0]), int(high_idx[1])))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ys))
    for idx in order:
        coords.append((int(ys[idx]), int(xs[idx])))
        if len(dict.fromkeys(coords)) >= n:
            break

    return list(dict.fromkeys(coords))[:n]


def _select_fixed_indices(dataset_len: int, value: str | None) -> list[int]:
    if value is not None:
        indices = _parse_csv_ints(value)
    else:
        indices = [0, dataset_len // 2, dataset_len - 1]
    for idx in indices:
        if idx < 0 or idx >= dataset_len:
            raise IndexError(f"Sample index {idx} out of range for dataset length {dataset_len}")
    return list(dict.fromkeys(indices))


@torch.no_grad()
def _select_error_quantile_indices(
    dataset,
    model: LoadedModel,
    device: torch.device,
    pool_size: int,
    quantiles: list[float],
    use_amp: bool,
    show_progress: bool,
) -> list[int]:
    pool_size = min(pool_size, len(dataset))
    if pool_size <= 0:
        raise ValueError("selection_pool_size must be positive")
    records = []
    iterator = range(pool_size)
    if show_progress:
        print(f"Scanning first {pool_size} samples with {model.label} for error quantiles")
    for idx in iterator:
        sample = dataset[idx]
        x = sample["x"].unsqueeze(0).to(device)
        mask = sample.get("valid_mask")
        mask = mask.unsqueeze(0).to(device) if mask is not None else None
        with torch.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda",
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        ):
            outputs = _call_model_forward(model.model, x, mask)
        x_hat = outputs["x_hat"].float()
        metric_mask = mask if mask is not None else torch.ones_like(x, dtype=torch.bool)
        score = float(masked_mse(x_hat, x, metric_mask).item())
        records.append((idx, score))

    records.sort(key=lambda item: item[1])
    selected = []
    for q in quantiles:
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"Quantile must be in [0, 1], got {q}")
        pos = int(round(q * (len(records) - 1)))
        selected.append(records[pos][0])
    return list(dict.fromkeys(selected))


@torch.no_grad()
def _reconstruct_sample(
    models: list[LoadedModel],
    sample: dict,
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    x = sample["x"].unsqueeze(0).to(device)
    mask = sample.get("valid_mask")
    mask = mask.unsqueeze(0).to(device) if mask is not None else None
    recons = {}
    for loaded in models:
        # every checkpoint reconstructs the exact same input sample for fair visual comparison.
        with torch.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda",
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        ):
            outputs = _call_model_forward(loaded.model, x, mask)
        recons[loaded.label] = outputs["x_hat"][0].detach().float().cpu().clamp(0.0, 1.0)
    return x[0].detach().float().cpu(), None if mask is None else mask[0].detach().cpu(), recons


def _save_rgb_error_grid(
    output_base: Path,
    x_np: np.ndarray,
    recons_np: dict[str, np.ndarray],
    bands: tuple[int, int, int],
    valid_pixels: np.ndarray,
    title: str,
    dpi: int,
    save_pdf: bool,
) -> None:
    # use one rgb stretch from the original image so reconstructions are visually comparable.
    rgb_params = _reference_rgb_params(x_np, bands=bands, valid_pixels=valid_pixels)
    original_rgb = _rgb_with_params(x_np, bands=bands, params=rgb_params, valid_pixels=valid_pixels)
    error_maps = {
        # error maps show mean absolute spectral error at each spatial location.
        label: _error_map(x_hat, x_np, valid_pixels=valid_pixels)
        for label, x_hat in recons_np.items()
    }
    finite_values = np.concatenate(
        [err[np.isfinite(err)].reshape(-1) for err in error_maps.values()]
    )
    err_vmax = float(np.percentile(finite_values, 99.0)) if finite_values.size else 1.0
    # percentile cap prevents one outlier pixel from hiding the rest of the error structure.
    err_vmax = max(err_vmax, 1e-8)

    labels = list(recons_np)
    fig, axes = plt.subplots(2, len(labels) + 1, figsize=(4.0 * (len(labels) + 1), 7.0))
    axes = np.asarray(axes)
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    axes[1, 0].text(0.02, 0.5, f"RGB bands: {bands}\nError: mean |x-x_hat| over bands", va="center")

    for col, label in enumerate(labels, start=1):
        # top row shows rgb reconstruction, bottom row shows spatial error.
        recon_rgb = _rgb_with_params(
            recons_np[label], bands=bands, params=rgb_params, valid_pixels=valid_pixels
        )
        axes[0, col].imshow(recon_rgb)
        axes[0, col].set_title(label)
        axes[0, col].axis("off")

        im = axes[1, col].imshow(error_maps[label], cmap="magma", vmin=0.0, vmax=err_vmax)
        axes[1, col].set_title(f"{label} error")
        axes[1, col].axis("off")
        fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _save_spectra_plot(
    output_base: Path,
    x_np: np.ndarray,
    recons_np: dict[str, np.ndarray],
    coords: list[tuple[int, int]],
    title: str,
    dpi: int,
    save_pdf: bool,
) -> None:
    x_axis = np.arange(x_np.shape[0])
    fig, axes = plt.subplots(1, len(coords), figsize=(5.2 * len(coords), 4.2), squeeze=False)
    for ax, (row, col) in zip(axes[0], coords, strict=True):
        # spectra plots expose band-level errors that rgb images can hide.
        ax.plot(x_axis, x_np[:, row, col], label="Original", linewidth=2.2, color="black")
        for label, x_hat in recons_np.items():
            ax.plot(x_axis, x_hat[:, row, col], label=label, linewidth=1.4)
        ax.set_title(f"Pixel ({row}, {col})")
        ax.set_xlabel("Band")
        ax.set_ylabel("Normalized reflectance")
        ax.grid(True, alpha=0.3)
    axes[0, -1].legend(loc="best", fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _save_mean_spectrum_plot(
    output_base: Path,
    x_np: np.ndarray,
    recons_np: dict[str, np.ndarray],
    valid_pixels: np.ndarray,
    title: str,
    dpi: int,
    save_pdf: bool,
) -> None:
    x_axis = np.arange(x_np.shape[0])
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    # mean spectrum gives one compact summary of reconstruction bias across valid pixels.
    ax.plot(
        x_axis, x_np[:, valid_pixels].mean(axis=1), label="Original", linewidth=2.2, color="black"
    )
    for label, x_hat in recons_np.items():
        ax.plot(x_axis, x_hat[:, valid_pixels].mean(axis=1), label=label, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Band")
    ax.set_ylabel("Mean normalized reflectance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_summary_markdown(
    output_dir: Path, sample_records: list[dict], models: list[LoadedModel]
) -> None:
    # markdown summary links figures and metrics into one human-readable artifact.
    lines = [
        "# Qualitative Reconstruction Export",
        "",
        "All reconstructions use the same HySpecNet split sample and the same RGB stretch per patch.",
        "",
        "## Checkpoints",
        "",
        "| label | checkpoint |",
        "| --- | --- |",
    ]
    for loaded in models:
        lines.append(f"| {loaded.label} | `{loaded.path}` |")
    lines.extend(
        [
            "",
            "## Samples",
            "",
            "| index | patch_id | path |",
            "| ---: | --- | --- |",
        ]
    )
    for record in sample_records:
        lines.append(f"| {record['index']} | `{record['patch_id']}` | `{record['path']}` |")
    lines.extend(
        [
            "",
            "## Per-Sample Metrics",
            "",
            "| index | patch_id | model | PSNR dB | SAM deg | MAE | SID |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in sample_records:
        for label, metrics in record["metrics"].items():
            lines.append(
                f"| {record['index']} | `{record['patch_id']}` | {label} | "
                f"{metrics['masked_psnr']:.4f} | {metrics['masked_sam_deg']:.4f} | "
                f"{metrics['masked_mae']:.6f} | {metrics['masked_sid']:.6f} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    load_project_env()
    args = parse_args()

    dataset_root = Path(
        args.dataset_root
        or os.environ.get("DATASET_ROOT")
        or "/workspace/data/hyspectnet-11k/hyspecnet-11k-full"
    )
    if not dataset_root.exists():
        print(f"Error: dataset_root does not exist: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(
        args.output_dir or "artifacts/analysis/hierarchical_mamba_rd_20260509/qualitative"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(
        # qualitative export uses the reference preprocessed benchmark artifacts.
        dataset_root=dataset_root,
        split_name=args.split,
        difficulty=args.difficulty,
        normalized=True,
        return_mask=True,
        drop_invalid_channels=True,
        prefer_npy=True,
        npy_mmap=False,
    )
    first = dataset[0]
    in_channels = int(first["x"].shape[0])
    # default rgb bands are chosen from available spectral bands when not specified.
    rgb_bands = (
        tuple(_parse_csv_ints(args.rgb_bands))
        if args.rgb_bands is not None
        else choose_evenly_spaced_rgb_bands(in_channels)
    )
    if len(rgb_bands) != 3:
        raise ValueError("--rgb-bands must contain exactly three indices")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.disable_amp and device.type == "cuda"
    print(f"Device: {device}")
    print(f"Dataset: {args.difficulty}/{args.split} | samples={len(dataset)} | bands={in_channels}")
    print(f"RGB bands: {rgb_bands}")

    specs = _parse_checkpoints(args.checkpoint)
    # load all checkpoints once so every selected sample uses the same models.
    models = [_load_model(spec, in_channels=in_channels, device=device) for spec in specs]
    print("Loaded checkpoints:")
    for loaded in models:
        print(f"  {loaded.label}: {loaded.path}")

    if args.selection_mode == "fixed":
        # fixed indices are useful when comparing the same examples across runs.
        sample_indices = _select_fixed_indices(len(dataset), args.sample_indices)
    else:
        selection_model = next(
            (loaded for loaded in models if loaded.label == args.selection_checkpoint_label),
            None,
        )
        if selection_model is None:
            raise ValueError(
                f"--selection-checkpoint-label={args.selection_checkpoint_label!r} "
                f"not found in loaded labels {[m.label for m in models]}"
            )
        # error quantiles choose easy, medium, and hard examples from a model's errors.
        sample_indices = _select_error_quantile_indices(
            dataset=dataset,
            model=selection_model,
            device=device,
            pool_size=args.selection_pool_size,
            quantiles=_parse_csv_floats(args.selection_quantiles),
            use_amp=use_amp,
            show_progress=not args.no_progress,
        )

    print(f"Selected sample indices: {sample_indices}")

    sample_records = []
    for sample_index in sample_indices:
        sample = dataset[sample_index]
        patch_id = sample.get("patch_id", f"idx_{sample_index}")
        sample_dir = output_dir / f"sample_{sample_index:04d}_{_safe_name(str(patch_id))}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        x_cpu, mask_cpu, recons = _reconstruct_sample(
            models=models,
            sample=sample,
            device=device,
            use_amp=use_amp,
        )
        x_np = x_cpu.numpy()
        recons_np = {label: tensor.numpy() for label, tensor in recons.items()}
        valid_pixels = _valid_pixel_mask(mask_cpu, h=x_np.shape[1], w=x_np.shape[2])
        anchor_label = models[-1].label
        # spectrum coordinates are selected from the last model's error map.
        anchor_error = _error_map(recons_np[anchor_label], x_np, valid_pixels)
        coords = _choose_spectrum_coords(
            valid_pixels=valid_pixels,
            anchor_error=anchor_error,
            n=args.num_spectra,
            seed=args.seed + sample_index,
        )

        metrics = {
            # store numerical metrics next to figures so examples are not judged only visually.
            label: _sample_metrics(tensor, x_cpu, mask_cpu)
            for label, tensor in recons.items()
        }

        title = f"{args.difficulty}/{args.split} index={sample_index} patch={patch_id}"
        _save_rgb_error_grid(
            output_base=sample_dir / "rgb_reconstruction_error",
            x_np=x_np,
            recons_np=recons_np,
            bands=rgb_bands,
            valid_pixels=valid_pixels,
            title=title,
            dpi=args.dpi,
            save_pdf=args.save_pdf,
        )
        _save_spectra_plot(
            output_base=sample_dir / "pixel_spectra",
            x_np=x_np,
            recons_np=recons_np,
            coords=coords,
            title=f"Pixel spectra | {title}",
            dpi=args.dpi,
            save_pdf=args.save_pdf,
        )
        _save_mean_spectrum_plot(
            output_base=sample_dir / "mean_spectrum",
            x_np=x_np,
            recons_np=recons_np,
            valid_pixels=valid_pixels,
            title=f"Mean spectrum | {title}",
            dpi=args.dpi,
            save_pdf=args.save_pdf,
        )

        record = {
            "index": sample_index,
            "patch_id": patch_id,
            "path": sample.get("path"),
            "rgb_bands": list(rgb_bands),
            "spectrum_coords": coords,
            "metrics": metrics,
            "files": {
                "rgb_reconstruction_error": str(sample_dir / "rgb_reconstruction_error.png"),
                "pixel_spectra": str(sample_dir / "pixel_spectra.png"),
                "mean_spectrum": str(sample_dir / "mean_spectrum.png"),
            },
        }
        (sample_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
        sample_records.append(record)
        print(f"Saved qualitative sample: {sample_dir}")

    manifest = {
        # manifest records all inputs needed to recreate the qualitative export.
        "dataset_root": str(dataset_root),
        "split": args.split,
        "difficulty": args.difficulty,
        "num_samples_in_split": len(dataset),
        "selected_indices": sample_indices,
        "rgb_bands": list(rgb_bands),
        "use_amp": use_amp,
        "checkpoints": [
            {
                "label": loaded.label,
                "path": str(loaded.path),
                "experiment": loaded.config.get("experiment", {}).get("name"),
                "loss": loaded.config.get("training", {}).get("loss_name"),
                "rd_lambda": loaded.config.get("training", {}).get("rd_lambda"),
            }
            for loaded in models
        ],
        "samples": sample_records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_summary_markdown(output_dir, sample_records=sample_records, models=models)
    print(f"Saved qualitative export: {output_dir}")


if __name__ == "__main__":
    main()
