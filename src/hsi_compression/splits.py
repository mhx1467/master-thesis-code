from pathlib import Path

import pandas as pd


def load_split_csv(csv_path: str | Path) -> list[str]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if df.shape[1] != 1:
        raise ValueError(f"Expected 1-column CSV, got shape={df.shape} for {csv_path}")

    return df.iloc[:, 0].astype(str).tolist()


def csv_entry_to_spectral_tif(dataset_root: str | Path, rel_path: str) -> Path:
    """
    Convert a split CSV entry like:
      SCENE/PATCH/PATCH-DATA.npy
    into:
      <dataset_root>/patches/SCENE/PATCH/PATCH-SPECTRAL_IMAGE.TIF
    """
    dataset_root = Path(dataset_root)
    rel = Path(rel_path)

    filename = rel.name
    if not filename.endswith("-DATA.npy"):
        raise ValueError(f"Unexpected split filename format: {rel_path}")

    spectral_name = filename.replace("-DATA.npy", "-SPECTRAL_IMAGE.TIF")
    spectral_rel = rel.parent / spectral_name

    return dataset_root / "patches" / spectral_rel


def split_csv_path(dataset_root: str | Path, split_name: str, difficulty: str) -> Path:
    dataset_root = Path(dataset_root)
    return dataset_root / "splits" / difficulty / f"{split_name}.csv"


def resolve_split_paths(
    dataset_root: str | Path,
    csv_path: str | Path,
) -> list[Path]:
    rel_paths = load_split_csv(csv_path)
    paths = [csv_entry_to_spectral_tif(dataset_root, p) for p in rel_paths]

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} resolved spectral files do not exist. First missing: {missing[0]}"
        )

    return paths
