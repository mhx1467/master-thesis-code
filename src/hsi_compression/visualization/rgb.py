from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

ArrayLike = np.ndarray | torch.Tensor


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x)


def _ensure_chw(x: np.ndarray, channel_first: bool | None = None) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={x.shape}")

    if channel_first is True:
        return x

    if channel_first is False:
        return np.transpose(x, (2, 0, 1))

    # Auto-detect
    # Heuristic: if first dim is relatively small, assume CHW
    if x.shape[0] <= 256 and x.shape[1] > 16 and x.shape[2] > 16:
        return x

    # Otherwise assume HWC
    return np.transpose(x, (2, 0, 1))


def _normalize_channel(
    channel: np.ndarray,
    mask: np.ndarray | None = None,
    p_low: float = 2.0,
    p_high: float = 98.0,
    eps: float = 1e-8,
) -> np.ndarray:
    if mask is not None:
        valid = channel[mask]
        if valid.size == 0:
            return np.zeros_like(channel, dtype=np.float32)
    else:
        valid = channel.reshape(-1)

    lo = np.percentile(valid, p_low)
    hi = np.percentile(valid, p_high)

    if hi - lo < eps:
        return np.zeros_like(channel, dtype=np.float32)

    out = (channel - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def _apply_gamma(rgb: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if gamma == 1.0:
        return rgb
    return np.power(np.clip(rgb, 0.0, 1.0), 1.0 / gamma)


def hsi_to_rgb(
    x: ArrayLike,
    bands: tuple[int, int, int] = (30, 20, 10),
    channel_first: bool | None = None,
    mask: ArrayLike | None = None,
    percentile_stretch: tuple[float, float] = (2.0, 98.0),
    gamma: float = 1.0,
) -> np.ndarray:
    x_np = _to_numpy(x)
    x_chw = _ensure_chw(x_np, channel_first=channel_first)

    c, h, w = x_chw.shape
    r_idx, g_idx, b_idx = bands

    if max(bands) >= c or min(bands) < 0:
        raise IndexError(f"Band indices {bands} out of range for C={c}")

    mask_np = None if mask is None else _to_numpy(mask).astype(bool)
    if mask_np is not None and mask_np.shape != (h, w):
        raise ValueError(f"Mask shape must be {(h, w)}, got {mask_np.shape}")

    p_low, p_high = percentile_stretch
    r = _normalize_channel(x_chw[r_idx], mask=mask_np, p_low=p_low, p_high=p_high)
    g = _normalize_channel(x_chw[g_idx], mask=mask_np, p_low=p_low, p_high=p_high)
    b = _normalize_channel(x_chw[b_idx], mask=mask_np, p_low=p_low, p_high=p_high)

    rgb = np.stack([r, g, b], axis=-1)
    rgb = _apply_gamma(rgb, gamma=gamma)

    if mask_np is not None:
        rgb = rgb.copy()
        rgb[~mask_np] = 0.0

    return rgb


def plot_rgb(
    x: ArrayLike,
    bands: tuple[int, int, int] = (30, 20, 10),
    channel_first: bool | None = None,
    mask: ArrayLike | None = None,
    percentile_stretch: tuple[float, float] = (2.0, 98.0),
    gamma: float = 1.0,
    title: str | None = None,
    figsize: tuple[int, int] = (6, 6),
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    rgb = hsi_to_rgb(
        x=x,
        bands=bands,
        channel_first=channel_first,
        mask=mask,
        percentile_stretch=percentile_stretch,
        gamma=gamma,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(rgb)
    ax.axis("off")

    if title is None:
        title = f"Pseudo-RGB | bands={bands}"
    ax.set_title(title)

    if show and ax.figure:
        plt.tight_layout()
        plt.show()

    return ax


def plot_rgb_comparison(
    x_true: ArrayLike,
    x_pred: ArrayLike,
    bands: tuple[int, int, int] = (30, 20, 10),
    channel_first: bool | None = None,
    mask: ArrayLike | None = None,
    percentile_stretch: tuple[float, float] = (2.0, 98.0),
    gamma: float = 1.0,
    titles: tuple[str, str] = ("Input", "Reconstruction"),
    figsize: tuple[int, int] = (12, 5),
    show: bool = True,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    rgb_true = hsi_to_rgb(
        x_true,
        bands=bands,
        channel_first=channel_first,
        mask=mask,
        percentile_stretch=percentile_stretch,
        gamma=gamma,
    )
    rgb_pred = hsi_to_rgb(
        x_pred,
        bands=bands,
        channel_first=channel_first,
        mask=mask,
        percentile_stretch=percentile_stretch,
        gamma=gamma,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(rgb_true)
    axes[0].set_title(titles[0])
    axes[0].axis("off")

    axes[1].imshow(rgb_pred)
    axes[1].set_title(titles[1])
    axes[1].axis("off")

    if show:
        plt.tight_layout()
        plt.show()

    return fig, axes


def save_rgb(
    path: str,
    x: ArrayLike,
    bands: tuple[int, int, int] = (30, 20, 10),
    channel_first: bool | None = None,
    mask: ArrayLike | None = None,
    percentile_stretch: tuple[float, float] = (2.0, 98.0),
    gamma: float = 1.0,
    dpi: int = 150,
) -> None:
    rgb = hsi_to_rgb(
        x=x,
        bands=bands,
        channel_first=channel_first,
        mask=mask,
        percentile_stretch=percentile_stretch,
        gamma=gamma,
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def choose_evenly_spaced_rgb_bands(num_bands: int) -> tuple[int, int, int]:
    if num_bands < 3:
        raise ValueError("num_bands must be >= 3")

    b = int(round(0.25 * (num_bands - 1)))
    g = int(round(0.50 * (num_bands - 1)))
    r = int(round(0.75 * (num_bands - 1)))
    return (r, g, b)
