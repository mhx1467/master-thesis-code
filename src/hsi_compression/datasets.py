from pathlib import Path

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset

from hsi_compression.constants import NODATA_VALUE


class HSITiffDataset(Dataset):
    def __init__(
        self,
        paths: list[str | Path],
        nodata_value: int = NODATA_VALUE,
        replace_nodata_with: float = 0.0,
        transform=None,
        return_mask: bool = True,
        invalid_channels: list[int] | None = None,
        drop_invalid_channels: bool = True,
        prefer_npy: bool = True,
    ):
        self.paths = [Path(p) for p in paths]
        self.nodata_value = nodata_value
        self.replace_nodata_with = replace_nodata_with
        self.transform = transform
        self.return_mask = return_mask
        self.invalid_channels = sorted(invalid_channels or [])
        self.drop_invalid_channels = drop_invalid_channels
        self.prefer_npy = prefer_npy

        if len(self.paths) == 0:
            raise ValueError("Empty dataset: no paths provided.")

        if self.prefer_npy:
            npy_path = self._tif_to_npy_path(self.paths[0])
            self._use_npy = npy_path.exists()
            if self._use_npy:
                sample = np.load(str(npy_path))
                self._npy_shape = sample.shape  # (H, W, C) or (C, H, W)
            else:
                self._use_npy = False
        else:
            self._use_npy = False

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]

        if self._use_npy:
            x, valid_mask = self._load_npy(path)
        else:
            x, valid_mask = self._load_tif(path)

        x = torch.from_numpy(x)                          # (C, H, W) float32
        valid_mask = torch.from_numpy(valid_mask)         # (C, H, W) bool

        if self.transform is not None:
            x = self.transform(x, valid_mask)

        patch_id = path.stem.replace("-SPECTRAL_IMAGE", "")

        if self.return_mask:
            return {"x": x, "valid_mask": valid_mask, "path": str(path), "patch_id": patch_id}
        return x

    def _load_npy(self, tif_path: Path):
        npy_path = self._tif_to_npy_path(tif_path)
        data = np.load(str(npy_path))   # (H, W, C) or (C, H, W)

        if data.shape[-1] < data.shape[0]:
            # (H, W, C) to (C, H, W)
            data = data.transpose(2, 0, 1)

        data = data.astype(np.float32)

        valid_mask = np.ones_like(data, dtype=bool)

        return data, valid_mask

    def _load_tif(self, path: Path):
        x = tiff.imread(str(path))   # (C, H, W) int16

        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got {x.shape} for {path}")

        valid_mask = (x != self.nodata_value)

        if self.invalid_channels:
            valid_mask[self.invalid_channels, :, :] = False

        x = x.astype(np.float32)
        x[~valid_mask] = self.replace_nodata_with

        if self.drop_invalid_channels and self.invalid_channels:
            keep = np.ones(x.shape[0], dtype=bool)
            keep[self.invalid_channels] = False
            x = x[keep]
            valid_mask = valid_mask[keep]

        return x, valid_mask

    @staticmethod
    def _tif_to_npy_path(tif_path: Path) -> Path:
        stem = tif_path.stem.replace("-SPECTRAL_IMAGE", "")
        return tif_path.parent / f"{stem}-DATA.npy"

    @property
    def using_npy(self) -> bool:
        return self._use_npy
