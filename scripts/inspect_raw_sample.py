import sys
from pathlib import Path

from hsi_compression.data.datamodule import build_dataset
from hsi_compression.utils import load_project_env


def main():
    load_project_env()

    if len(sys.argv) < 2:
        print("Usage: python inspect_raw_sample.py <dataset_root_path>")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])
    if not dataset_root.exists():
        print(f"Error: Provided dataset root path does not exist: {dataset_root}")
        print("Example usage: python inspect_raw_sample.py /path/to/dataset")
        sys.exit(1)

    ds = build_dataset(
        dataset_root=dataset_root,
        split_name="train",
        difficulty="easy",
        normalized=False,
        return_mask=True,
        drop_invalid_channels=False,
    )

    sample = ds[0]
    x = sample["x"]
    m = sample["valid_mask"]

    print("x shape:", tuple(x.shape))
    print("x dtype:", x.dtype)
    print("valid ratio:", m.float().mean().item())
    print("x min:", x.min().item())
    print("x max:", x.max().item())
    print("x mean:", x.mean().item())
    print("x std:", x.std().item())


if __name__ == "__main__":
    main()
