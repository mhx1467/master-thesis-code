from pathlib import Path

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset

from hsi_compression.constants import NODATA_VALUE

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

        self._use_npy = False
        if prefer_npy:
            npy_path = self._tif_to_npy_path(self.paths[0])
            if npy_path.exists():
                sample = np.load(str(npy_path), mmap_mode="r")
                if sample.shape == _NPY_SHAPE_CHW:
                    self._use_npy = True
                    self._npy_is_chw = True
                elif sample.shape == _NPY_SHAPE_HWC:
                    self._use_npy = True
                    self._npy_is_chw = False
                else:
                    import warnings

                    warnings.warn(
                        f"Unexpected .npy shape {sample.shape}. Fallback to .TIF.",
                        UserWarning,
                    )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]

        if self._use_npy:
            x = self._load_npy(path)
            if not x.flags.writeable:
                x = np.array(x, copy=True)
            if x.dtype != np.float32 or not x.flags.c_contiguous:
                x = np.ascontiguousarray(x, dtype=np.float32)
            x_tensor = torch.from_numpy(x)
            if self.transform is not None:
                x_tensor = self.transform(x_tensor)
        else:
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
                patch_id = path.stem.replace("-SPECTRAL_IMAGE", "")
                return {
                    "x": x_tensor,
                    "valid_mask": mask_tensor,
                    "path": str(path),
                    "patch_id": patch_id,
                }

        return x_tensor

    def _load_npy(self, tif_path: Path):
        npy_path = self._tif_to_npy_path(tif_path)
        data = np.load(str(npy_path), mmap_mode="r" if self.npy_mmap else None)
        out = data if self._npy_is_chw else data.transpose(2, 0, 1)
        return out

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

        return x, valid_mask

    @staticmethod
    def _tif_to_npy_path(tif_path: Path) -> Path:
        stem = tif_path.stem.replace("-SPECTRAL_IMAGE", "")
        return tif_path.parent / f"{stem}-DATA.npy"

    @property
    def using_npy(self) -> bool:
        return self._use_npy
