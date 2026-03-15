from pathlib import Path
import random
import sys
from torch.utils.data import Subset

from hsi_compression.data import build_dataset, build_dataloader
from hsi_compression.utils import load_project_env


def main():
    load_project_env()

    if len(sys.argv) < 2:
        print("Usage: python check_normalization_subset.py <dataset_root_path>")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])
    if not dataset_root.exists():
        print(f"Error: Provided dataset root path does not exist: {dataset_root}")
        print("Example usage: python check_normalization_subset.py /path/to/dataset")
        sys.exit(1)

    ds = build_dataset(
        dataset_root=dataset_root,
        split_name="train",
        difficulty="easy",
        normalized=True,
        return_mask=True,
        drop_invalid_channels=False,
    )

    rng = random.Random(42)
    indices = rng.sample(range(len(ds)), 500)

    subset = Subset(ds, indices)
    loader = build_dataloader(subset, batch_size=4, shuffle=False, num_workers=0)

    all_sum = 0.0
    all_sumsq = 0.0
    all_count = 0

    for batch in loader:
        x = batch["x"]
        m = batch["valid_mask"]

        vals = x[m]
        all_sum += vals.sum().item()
        all_sumsq += (vals ** 2).sum().item()
        all_count += vals.numel()

    mean = all_sum / all_count
    var = all_sumsq / all_count - mean ** 2
    std = var ** 0.5

    print("normalized valid mean:", mean)
    print("normalized valid std:", std)


if __name__ == "__main__":
    main()