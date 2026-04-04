from pathlib import Path

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset

from hsi_compression.constants import GLOBAL_MAX, GLOBAL_MIN, NODATA_VALUE

_NPY_SHAPE_HWC = (128, 128, 202)
_NPY_SHAPE_CHW = (202, 128, 128)


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
        npy_mmap: bool = False,
    ):
        self.paths = [Path(p) for p in paths]
        self.nodata_value = nodata_value
        self.replace_nodata_with = replace_nodata_with
        self.transform = transform
        self.return_mask = return_mask
        self.invalid_channels = sorted(invalid_channels or [])
        self.drop_invalid_channels = drop_invalid_channels
        self.npy_mmap = npy_mmap
        self._npy_is_chw = False

        if not self.paths:
            raise ValueError("Empty dataset: no paths provided.")

        first_path = self.paths[0]
        self._use_npy = first_path.suffix.lower() == ".npy"
        if self._use_npy:
            sample = np.load(str(first_path), mmap_mode="r")
            if sample.shape == _NPY_SHAPE_CHW:
                self._npy_is_chw = True
            elif sample.shape == _NPY_SHAPE_HWC:
                self._npy_is_chw = False
            else:
                raise ValueError(
                    f"Unexpected .npy shape {sample.shape} for {first_path}. "
                    "Expected (202, 128, 128) or (128, 128, 202)."
                )
        elif prefer_npy:
            raise ValueError(
                "Benchmark dataset path resolved to TIF files. "
                "Train/val/test benchmark runs must use preprocessed '*-DATA.npy' artifacts."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        patch_id = path.stem.replace("-SPECTRAL_IMAGE", "").replace("-DATA", "")

        if self._use_npy:
            x = self._load_npy(path)
            if not x.flags.writeable:
                x = np.array(x, copy=True)
            if x.dtype != np.float32 or not x.flags.c_contiguous:
                x = np.ascontiguousarray(x, dtype=np.float32)
            x_tensor = torch.from_numpy(x)
            mask_tensor = self._build_mask_for_npy(x_tensor)
            if self.transform is not None:
                x_tensor = self.transform(x_tensor, mask_tensor)
            if self.return_mask:
                return {
                    "x": x_tensor,
                    "valid_mask": mask_tensor,
                    "path": str(path),
                    "patch_id": patch_id,
                }
            return x_tensor

        x, valid_mask = self._load_tif(path)
        if x.dtype != np.float32 or not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float32)
        if not valid_mask.flags.c_contiguous:
            valid_mask = np.ascontiguousarray(valid_mask)
        x_tensor = torch.from_numpy(x)
        mask_tensor = torch.from_numpy(valid_mask)
        if self.transform is not None:
            x_tensor = self.transform(x_tensor, mask_tensor)

        if self.return_mask:
            return {
                "x": x_tensor,
                "valid_mask": mask_tensor,
                "path": str(path),
                "patch_id": patch_id,
            }
        return x_tensor

    def _build_mask_for_npy(self, x_tensor: torch.Tensor) -> torch.Tensor:
        # HySpecNet benchmark DATA.npy artifacts are already preprocessed to 202 valid bands.
        return torch.ones_like(x_tensor, dtype=torch.bool)

    def _load_npy(self, npy_path: Path):
        data = np.load(str(npy_path), mmap_mode="r" if self.npy_mmap else None)
        return data if self._npy_is_chw else data.transpose(2, 0, 1)

    def _load_tif(self, path: Path):
        x = tiff.imread(str(path))
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got {x.shape} for {path}")

        valid_mask = x != self.nodata_value
        if self.invalid_channels:
            valid_mask[self.invalid_channels] = False

        x = x.astype(np.float32)
        x[~valid_mask] = self.replace_nodata_with

        if self.drop_invalid_channels and self.invalid_channels:
            keep = [i for i in range(x.shape[0]) if i not in self.invalid_channels]
            x = x[keep]
            valid_mask = valid_mask[keep]

        x = np.clip(x, GLOBAL_MIN, GLOBAL_MAX)
        x = x / (GLOBAL_MAX - GLOBAL_MIN)

        return x, valid_mask

    @property
    def using_npy(self) -> bool:
        return self._use_npy
