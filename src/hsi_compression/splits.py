import csv
from pathlib import Path


def _validate_reference_split_entry(rel_path: str) -> None:
    rel = Path(rel_path)

    if rel.is_absolute():
        raise ValueError(f"Split entry must be a relative path, got absolute path: {rel_path}")
    if rel.parts and rel.parts[0] == "patches":
        raise ValueError(
            "Unexpected split entry with leading 'patches/'. "
            "HySpecNet-11k reference CSV entries are relative to the patches directory, "
            "e.g. 'TILE/PATCH/PATCH-DATA.npy'."
        )
    if any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError(f"Invalid split entry path components: {rel_path}")
    if len(rel.parts) != 3:
        raise ValueError(
            "Unexpected split entry format. "
            "HySpecNet-11k reference CSV entries must have exactly 3 path components: "
            "'TILE/PATCH/PATCH-DATA.npy'."
        )
    if rel.suffix.lower() != ".npy" or not rel.name.endswith("-DATA.npy"):
        raise ValueError(
            "Unexpected split entry filename. "
            "HySpecNet-11k benchmark CSV entries must point to '*-DATA.npy' artifacts."
        )
    if rel.parts[1] != rel.stem.removesuffix("-DATA"):
        raise ValueError(
            "Unexpected split entry patch naming. "
            "Expected 'TILE/PATCH/PATCH-DATA.npy' with matching patch directory and filename."
        )


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
    _validate_reference_split_entry(rel_path)
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
