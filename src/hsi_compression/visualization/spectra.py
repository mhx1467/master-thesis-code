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

    if x.shape[0] <= 256 and x.shape[1] > 16 and x.shape[2] > 16:
        return x

    return np.transpose(x, (2, 0, 1))


def _default_wavelengths(num_bands: int) -> np.ndarray:
    return np.arange(num_bands)


def _validate_mask(mask: ArrayLike | None, hw: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    mask_np = _to_numpy(mask).astype(bool)
    if mask_np.shape != hw:
        raise ValueError(f"Mask shape must be {hw}, got {mask_np.shape}")
    return mask_np


def extract_spectrum(
    x: ArrayLike,
    row: int,
    col: int,
    channel_first: bool | None = None,
) -> np.ndarray:
    x_np = _to_numpy(x)
    x_chw = _ensure_chw(x_np, channel_first=channel_first)
    _, h, w = x_chw.shape

    if not (0 <= row < h and 0 <= col < w):
        raise IndexError(f"Pixel ({row}, {col}) out of bounds for H={h}, W={w}")

    return x_chw[:, row, col].astype(np.float32)


def mean_spectrum(
    x: ArrayLike,
    mask: ArrayLike | None = None,
    channel_first: bool | None = None,
) -> np.ndarray:
    x_np = _to_numpy(x)
    x_chw = _ensure_chw(x_np, channel_first=channel_first)
    c, h, w = x_chw.shape

    mask_np = _validate_mask(mask, (h, w))
    flat = x_chw.reshape(c, -1)

    if mask_np is None:
        return flat.mean(axis=1).astype(np.float32)

    valid = mask_np.reshape(-1)
    if valid.sum() == 0:
        return np.zeros(c, dtype=np.float32)

    return flat[:, valid].mean(axis=1).astype(np.float32)


def sample_valid_pixels(
    mask: ArrayLike,
    n: int = 5,
    seed: int | None = 42,
) -> list[tuple[int, int]]:
    mask_np = _to_numpy(mask).astype(bool)
    ys, xs = np.where(mask_np)

    if len(ys) == 0:
        raise ValueError("Mask contains no valid pixels")

    n = min(n, len(ys))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ys), size=n, replace=False)

    return [(int(ys[i]), int(xs[i])) for i in idx]


def plot_spectrum(
    spectrum: ArrayLike,
    wavelengths: ArrayLike | None = None,
    title: str = "Spectrum",
    xlabel: str = "Band",
    ylabel: str = "Reflectance / Intensity",
    figsize: tuple[int, int] = (8, 4),
    ax: plt.Axes | None = None,
    show: bool = True,
    label: str | None = None,
) -> plt.Axes:
    spec = _to_numpy(spectrum).reshape(-1)
    x_axis = (
        _default_wavelengths(len(spec))
        if wavelengths is None
        else _to_numpy(wavelengths).reshape(-1)
    )

    if len(x_axis) != len(spec):
        raise ValueError("wavelengths and spectrum must have the same length")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x_axis, spec, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if label is not None:
        ax.legend()

    if show and ax.figure:
        plt.tight_layout()
        plt.show()

    return ax


def plot_pixel_spectra(
    x: ArrayLike,
    coords: Sequence[tuple[int, int]],
    wavelengths: ArrayLike | None = None,
    channel_first: bool | None = None,
    title: str = "Pixel spectra",
    xlabel: str = "Band",
    ylabel: str = "Reflectance / Intensity",
    figsize: tuple[int, int] = (9, 5),
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    x_np = _to_numpy(x)
    x_chw = _ensure_chw(x_np, channel_first=channel_first)
    c, _, _ = x_chw.shape

    x_axis = _default_wavelengths(c) if wavelengths is None else _to_numpy(wavelengths).reshape(-1)
    if len(x_axis) != c:
        raise ValueError("wavelengths length must match number of spectral bands")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for row, col in coords:
        spec = extract_spectrum(x_chw, row=row, col=col, channel_first=True)
        ax.plot(x_axis, spec, label=f"({row}, {col})")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if show and ax.figure:
        plt.tight_layout()
        plt.show()

    return ax


def plot_random_spectra(
    x: ArrayLike,
    n: int = 5,
    mask: ArrayLike | None = None,
    wavelengths: ArrayLike | None = None,
    channel_first: bool | None = None,
    seed: int | None = 42,
    title: str = "Random pixel spectra",
    figsize: tuple[int, int] = (9, 5),
    ax: plt.Axes | None = None,
    show: bool = True,
) -> tuple[plt.Axes, list[tuple[int, int]]]:
    x_np = _to_numpy(x)
    x_chw = _ensure_chw(x_np, channel_first=channel_first)
    _, h, w = x_chw.shape

    if mask is not None:
        coords = sample_valid_pixels(mask=mask, n=n, seed=seed)
    else:
        rng = np.random.default_rng(seed)
        coords = [(int(rng.integers(0, h)), int(rng.integers(0, w))) for _ in range(n)]

    ax = plot_pixel_spectra(
        x=x_chw,
        coords=coords,
        wavelengths=wavelengths,
        channel_first=True,
        title=title,
        figsize=figsize,
        ax=ax,
        show=show,
    )
    return ax, coords


def plot_mean_spectrum(
    x: ArrayLike,
    mask: ArrayLike | None = None,
    wavelengths: ArrayLike | None = None,
    channel_first: bool | None = None,
    title: str = "Mean spectrum",
    figsize: tuple[int, int] = (8, 4),
    ax: plt.Axes | None = None,
    show: bool = True,
    label: str = "mean",
) -> plt.Axes:
    x_np = _to_numpy(x)
    x_chw = _ensure_chw(x_np, channel_first=channel_first)
    spec = mean_spectrum(x_chw, mask=mask, channel_first=True)

    return plot_spectrum(
        spectrum=spec,
        wavelengths=wavelengths,
        title=title,
        figsize=figsize,
        ax=ax,
        show=show,
        label=label,
    )


def plot_spectrum_comparison(
    x_true: ArrayLike,
    x_pred: ArrayLike,
    row: int,
    col: int,
    wavelengths: ArrayLike | None = None,
    channel_first: bool | None = None,
    title: str | None = None,
    xlabel: str = "Band",
    ylabel: str = "Reflectance / Intensity",
    figsize: tuple[int, int] = (9, 5),
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    true_spec = extract_spectrum(x_true, row=row, col=col, channel_first=channel_first)
    pred_spec = extract_spectrum(x_pred, row=row, col=col, channel_first=channel_first)

    x_axis = (
        _default_wavelengths(len(true_spec))
        if wavelengths is None
        else _to_numpy(wavelengths).reshape(-1)
    )
    if len(x_axis) != len(true_spec):
        raise ValueError("wavelengths length must match number of spectral bands")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x_axis, true_spec, label="input")
    ax.plot(x_axis, pred_spec, label="reconstruction")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Spectrum comparison at pixel ({row}, {col})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if show and ax.figure:
        plt.tight_layout()
        plt.show()

    return ax


def plot_mean_spectrum_comparison(
    x_true: ArrayLike,
    x_pred: ArrayLike,
    mask: ArrayLike | None = None,
    wavelengths: ArrayLike | None = None,
    channel_first: bool | None = None,
    title: str = "Mean spectrum comparison",
    xlabel: str = "Band",
    ylabel: str = "Reflectance / Intensity",
    figsize: tuple[int, int] = (9, 5),
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    true_mean = mean_spectrum(x_true, mask=mask, channel_first=channel_first)
    pred_mean = mean_spectrum(x_pred, mask=mask, channel_first=channel_first)

    x_axis = (
        _default_wavelengths(len(true_mean))
        if wavelengths is None
        else _to_numpy(wavelengths).reshape(-1)
    )
    if len(x_axis) != len(true_mean):
        raise ValueError("wavelengths length must match number of spectral bands")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x_axis, true_mean, label="input mean")
    ax.plot(x_axis, pred_mean, label="reconstruction mean")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if show and ax.figure:
        plt.tight_layout()
        plt.show()

    return ax
