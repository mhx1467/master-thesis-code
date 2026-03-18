from pathlib import Path

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset


class HSITiffDataset(Dataset):
    def __init__(
        self,
        paths: list[str | Path],
        nodata_value: int = -32768,
        replace_nodata_with: float = 0.0,
        transform=None,
        return_mask: bool = False,
        invalid_channels: list[int] | None = None,
        drop_invalid_channels: bool = False,
    ):
        self.paths = [Path(p) for p in paths]
        self.nodata_value = nodata_value
        self.replace_nodata_with = replace_nodata_with
        self.transform = transform
        self.return_mask = return_mask
        self.invalid_channels = sorted(invalid_channels or [])
        self.drop_invalid_channels = drop_invalid_channels

        if len(self.paths) == 0:
            raise ValueError("Empty dataset: no paths provided.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]

        x = tiff.imread(path)  # expected shape: (B, H, W)
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got shape {x.shape} for {path}")

        valid_mask = x != self.nodata_value

        # Known dataset-invalid bands
        if self.invalid_channels:
            valid_mask[self.invalid_channels, :, :] = False

        x = x.astype(np.float32)
        x[~valid_mask] = self.replace_nodata_with

        if self.drop_invalid_channels and self.invalid_channels:
            keep = np.ones(x.shape[0], dtype=bool)
            keep[self.invalid_channels] = False
            x = x[keep]
            valid_mask = valid_mask[keep]

        x = torch.from_numpy(x)
        valid_mask = torch.from_numpy(valid_mask.astype(np.bool_))

        if self.transform is not None:
            x = self.transform(x, valid_mask)

        patch_id = path.stem.replace("-SPECTRAL_IMAGE", "")

        if self.return_mask:
            return {
                "x": x,
                "valid_mask": valid_mask,
                "path": str(path),
                "patch_id": patch_id,
            }

        return x
