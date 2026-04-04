import csv
from pathlib import Path


def load_split_csv(csv_path: str | Path) -> list[str]:
    csv_path = Path(csv_path)
    entries: list[str] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) != 1:
                raise ValueError(f"Expected 1-column CSV row, got {row!r} for {csv_path}")
            entries.append(row[0].strip())

    if not entries:
        raise ValueError(f"Empty split CSV: {csv_path}")

    return entries


def csv_entry_to_patch_path(dataset_root: str | Path, rel_path: str) -> Path:
    dataset_root = Path(dataset_root)
    rel = Path(rel_path)
    return dataset_root / "patches" / rel


def split_csv_path(dataset_root: str | Path, split_name: str, difficulty: str) -> Path:
    dataset_root = Path(dataset_root)
    return dataset_root / "splits" / difficulty / f"{split_name}.csv"


def resolve_split_paths(
    dataset_root: str | Path,
    csv_path: str | Path,
) -> list[Path]:
    rel_paths = load_split_csv(csv_path)
    paths = [csv_entry_to_patch_path(dataset_root, p) for p in rel_paths]

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} resolved patch files do not exist. First missing: {missing[0]}"
        )

    return paths
