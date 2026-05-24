import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hsi_compression.data import build_dataset
from hsi_compression.engine.checkpointing import load_checkpoint
from hsi_compression.metrics import masked_mae, masked_psnr, masked_sam_deg, ref_ssim
from hsi_compression.models.registry import build_model
from hsi_compression.utils import load_project_env
from hsi_compression.visualization import choose_evenly_spaced_rgb_bands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a compact qualitative panel for one Mamba checkpoint: original, "
            "reconstruction, and original-vs-reconstructed spectrum per selected sample."
        )
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("dataset_root", nargs="?", default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument(
        "--sample-indices",
        default=None,
        help="Comma-separated dataset indices. If omitted, use 0, middle, and last.",
    )
    parser.add_argument(
        "--pixel-mode",
        choices=("center", "max_error", "random"),
        default="max_error",
        help="Pixel used for the spectral curve in each sample.",
    )
    parser.add_argument("--rgb-bands", default=None, help="Comma-separated RGB band indices.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-pdf", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument(
        "--prefer-tif",
        action="store_true",
        help=(
            "Resolve split entries to matching *-SPECTRAL_IMAGE.TIF files instead of "
            "benchmark *-DATA.npy artifacts. Use only for local qualitative exports when "
            "DATA.npy files are unavailable."
        ),
    )
    return parser.parse_args()


def _parse_csv_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def _selected_indices(dataset_len: int, value: str | None) -> list[int]:
    indices = (
        _parse_csv_ints(value) if value is not None else [0, dataset_len // 2, dataset_len - 1]
    )
    for idx in indices:
        if idx < 0 or idx >= dataset_len:
            raise IndexError(f"Sample index {idx} out of range for dataset length {dataset_len}")
    return list(dict.fromkeys(indices))


def _call_model_forward(model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor | None) -> dict:
    try:
        outputs = model(x, valid_mask=mask)
    except TypeError:
        outputs = model(x)
    if not isinstance(outputs, dict) or "x_hat" not in outputs:
        raise RuntimeError("Model output must be a dict containing 'x_hat'.")
    return outputs


def _load_model(checkpoint_path: Path, in_channels: int, device: torch.device) -> tuple:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = raw.get("config", {})
    model_section = cfg.get("model", {})
    model_name = model_section.get("model_name")
    if model_name is None:
        raise RuntimeError(f"Checkpoint {checkpoint_path} has no config.model.model_name")
    model_kwargs = {
        key: value
        for key, value in model_section.get("model_kwargs", {}).items()
        if key != "in_channels"
    }
    model = build_model(model_name=model_name, in_channels=in_channels, **model_kwargs).to(device)
    load_checkpoint(path=checkpoint_path, model=model, optimizer=None, map_location=device)
    if hasattr(model, "update"):
        model.update(force=True)
    model.eval()
    return model, cfg


def _valid_pixel_mask(mask: torch.Tensor | None, h: int, w: int) -> np.ndarray:
    if mask is None:
        return np.ones((h, w), dtype=bool)
    mask_np = mask.detach().cpu().bool().numpy()
    if mask_np.ndim == 3:
        return mask_np.any(axis=0)
    if mask_np.ndim == 2:
        return mask_np
    raise ValueError(f"Unexpected mask shape: {mask_np.shape}")


def _rgb_params(
    x_chw: np.ndarray,
    bands: tuple[int, int, int],
    valid_pixels: np.ndarray,
) -> list[tuple[float, float]]:
    params = []
    for band in bands:
        values = x_chw[band][valid_pixels]
        if values.size == 0:
            params.append((0.0, 1.0))
            continue
        lo = float(np.percentile(values, 2.0))
        hi = float(np.percentile(values, 98.0))
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
    rgb_channels = []
    for band, (lo, hi) in zip(bands, params, strict=True):
        channel = np.clip((x_chw[band] - lo) / (hi - lo), 0.0, 1.0)
        rgb_channels.append(channel.astype(np.float32))
    rgb = np.stack(rgb_channels, axis=-1)
    rgb[~valid_pixels] = 0.0
    return rgb


def _choose_spectrum_pixel(
    x_np: np.ndarray,
    x_hat_np: np.ndarray,
    valid_pixels: np.ndarray,
    mode: str,
    seed: int,
) -> tuple[int, int]:
    ys, xs = np.where(valid_pixels)
    if len(ys) == 0:
        raise ValueError("No valid pixels available")

    if mode == "center":
        center = np.array([(valid_pixels.shape[0] - 1) / 2.0, (valid_pixels.shape[1] - 1) / 2.0])
        distances = (ys - center[0]) ** 2 + (xs - center[1]) ** 2
        idx = int(np.argmin(distances))
        return int(ys[idx]), int(xs[idx])

    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = int(rng.integers(0, len(ys)))
        return int(ys[idx]), int(xs[idx])

    error = np.mean(np.abs(x_hat_np - x_np), axis=0)
    error = np.where(valid_pixels, error, -np.inf)
    row, col = np.unravel_index(int(np.argmax(error)), error.shape)
    return int(row), int(col)


def _sample_metrics(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor | None,
) -> dict[str, float]:
    x_b = x.unsqueeze(0)
    x_hat_b = x_hat.unsqueeze(0)
    mask_b = torch.ones_like(x_b, dtype=torch.bool) if mask is None else mask.unsqueeze(0)
    return {
        "masked_psnr": float(masked_psnr(x_hat_b, x_b, mask_b, data_range=1.0).item()),
        "ssim": float(ref_ssim(x_hat_b, x_b, data_range=1.0, channels=x_b.shape[1]).item()),
        "masked_sam_deg": float(masked_sam_deg(x_hat_b, x_b, mask_b).item()),
        "masked_mae": float(masked_mae(x_hat_b, x_b, mask_b).item()),
    }


@torch.no_grad()
def _reconstruct(
    model: torch.nn.Module,
    sample: dict,
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    x = sample["x"].unsqueeze(0).to(device)
    mask = sample.get("valid_mask")
    mask = mask.unsqueeze(0).to(device) if mask is not None else None
    with torch.autocast(
        device_type=device.type,
        enabled=use_amp and device.type == "cuda",
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
    ):
        outputs = _call_model_forward(model, x, mask)
    x_hat = outputs["x_hat"].float().clamp(0.0, 1.0)
    return (
        x[0].detach().cpu().float(),
        None if mask is None else mask[0].detach().cpu(),
        x_hat[0].detach().cpu(),
    )


def _safe_stem(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    stem = "".join(ch if ch in allowed else "_" for ch in value).strip("._-")
    return stem or "mamba_reconstruction_panel"


def _save_panel(
    output_path: Path,
    rows: list[dict],
    rgb_bands: tuple[int, int, int],
    title: str,
    dpi: int,
    save_pdf: bool,
) -> None:
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(13.8, 4.0 * len(rows)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.35]},
    )
    band_axis = np.arange(rows[0]["x_np"].shape[0])

    for row_idx, row in enumerate(rows):
        x_np = row["x_np"]
        x_hat_np = row["x_hat_np"]
        valid_pixels = row["valid_pixels"]
        rgb_params = _rgb_params(x_np, rgb_bands, valid_pixels)
        original_rgb = _rgb_with_params(x_np, rgb_bands, rgb_params, valid_pixels)
        recon_rgb = _rgb_with_params(x_hat_np, rgb_bands, rgb_params, valid_pixels)
        spec_row, spec_col = row["spectrum_coord"]

        ax_orig, ax_recon, ax_spec = axes[row_idx]
        ax_orig.imshow(original_rgb)
        ax_orig.set_axis_off()
        ax_orig.set_title(f"Original\nidx={row['index']} | {row['patch_id']}", fontsize=10)

        metrics = row["metrics"]
        ax_recon.imshow(recon_rgb)
        ax_recon.set_axis_off()
        ax_recon.set_title(
            "Reconstruction\n"
            f"PSNR={metrics['masked_psnr']:.3f} dB | SSIM={metrics['ssim']:.4f}\n"
            f"SA={metrics['masked_sam_deg']:.3f} deg | MAE={metrics['masked_mae']:.5f}",
            fontsize=10,
        )

        ax_spec.plot(
            band_axis,
            x_np[:, spec_row, spec_col],
            color="black",
            linewidth=2.0,
            label="Original",
        )
        ax_spec.plot(
            band_axis,
            x_hat_np[:, spec_row, spec_col],
            color="#d55e00",
            linewidth=1.5,
            label="Reconstruction",
        )
        ax_spec.set_title(f"Spectrum at pixel ({spec_row}, {spec_col})", fontsize=10)
        ax_spec.set_xlabel("Band")
        ax_spec.set_ylabel("Normalized reflectance")
        ax_spec.grid(True, alpha=0.25)
        ax_spec.legend(loc="best", fontsize=8)

    fig.suptitle(f"{title} | RGB bands={rgb_bands}", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    load_project_env()
    args = parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: checkpoint does not exist: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    dataset_root = Path(
        args.dataset_root
        or os.environ.get("DATASET_ROOT")
        or "/workspace/data/hyspectnet-11k/hyspecnet-11k-full"
    )
    if not dataset_root.exists():
        print(f"Error: dataset_root does not exist: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    dataset = build_dataset(
        dataset_root=dataset_root,
        split_name=args.split,
        difficulty=args.difficulty,
        normalized=True,
        return_mask=True,
        drop_invalid_channels=True,
        prefer_npy=not args.prefer_tif,
        npy_mmap=False,
    )
    first = dataset[0]
    in_channels = int(first["x"].shape[0])
    rgb_bands = (
        tuple(_parse_csv_ints(args.rgb_bands))
        if args.rgb_bands is not None
        else choose_evenly_spaced_rgb_bands(in_channels)
    )
    if len(rgb_bands) != 3:
        raise ValueError("--rgb-bands must contain exactly three band indices")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.disable_amp and device.type == "cuda"
    model, cfg = _load_model(checkpoint_path, in_channels=in_channels, device=device)

    rows = []
    sample_indices = _selected_indices(len(dataset), args.sample_indices)
    for sample_index in sample_indices:
        sample = dataset[sample_index]
        x_cpu, mask_cpu, x_hat_cpu = _reconstruct(
            model=model,
            sample=sample,
            device=device,
            use_amp=use_amp,
        )
        x_np = x_cpu.numpy()
        x_hat_np = x_hat_cpu.numpy()
        valid_pixels = _valid_pixel_mask(mask_cpu, h=x_np.shape[1], w=x_np.shape[2])
        spectrum_coord = _choose_spectrum_pixel(
            x_np=x_np,
            x_hat_np=x_hat_np,
            valid_pixels=valid_pixels,
            mode=args.pixel_mode,
            seed=args.seed + sample_index,
        )
        rows.append(
            {
                "index": sample_index,
                "patch_id": sample.get("patch_id", f"idx_{sample_index}"),
                "path": sample.get("path"),
                "x_np": x_np,
                "x_hat_np": x_hat_np,
                "valid_pixels": valid_pixels,
                "spectrum_coord": spectrum_coord,
                "metrics": _sample_metrics(x_hat_cpu, x_cpu, mask_cpu),
            }
        )

    experiment_name = cfg.get("experiment", {}).get("name", checkpoint_path.stem)
    output_path = Path(
        args.output
        or f"artifacts/analysis/mamba_k4_reconstruction_panel/{_safe_stem(experiment_name)}_{args.split}.png"
    )
    _save_panel(
        output_path=output_path,
        rows=rows,
        rgb_bands=rgb_bands,
        title=f"{experiment_name} | {args.difficulty}/{args.split}",
        dpi=args.dpi,
        save_pdf=args.save_pdf,
    )

    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "experiment": experiment_name,
        "model_name": cfg.get("model", {}).get("model_name"),
        "dataset_root": str(dataset_root),
        "dataset_source": "tif" if args.prefer_tif else "data_npy",
        "difficulty": args.difficulty,
        "split": args.split,
        "rgb_bands": list(rgb_bands),
        "pixel_mode": args.pixel_mode,
        "output": str(output_path),
        "samples": [
            {
                "index": row["index"],
                "patch_id": row["patch_id"],
                "path": row["path"],
                "spectrum_coord": list(row["spectrum_coord"]),
                "metrics": row["metrics"],
            }
            for row in rows
        ],
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved panel: {output_path}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
