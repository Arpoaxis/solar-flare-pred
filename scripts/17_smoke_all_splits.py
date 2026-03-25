"""
Run a quick DataLoader sanity check for the train, validation, and test splits.

This script loops over the saved split CSV files, instantiates the custom
SDOBenchmarkDataset for each split, wraps each dataset in a PyTorch
DataLoader, and retrieves one batch to confirm that loading works across
all partitions.

It is intended as a lightweight integration check between:
    - the saved split CSV files
    - the curated index
    - the custom dataset class
    - the PyTorch DataLoader

Inputs:
    - data/interim/sdobenchmark/index.parquet
    - data/interim/sdobenchmark/splits/train.csv
    - data/interim/sdobenchmark/splits/val.csv
    - data/interim/sdobenchmark/splits/test.csv

Output:
    - Printed dataset length for each split
    - Printed one batch shape and labels for each split
"""

from pathlib import Path

from torch.utils.data import DataLoader

from src.data.sdobenchmark_dataset import SDOBenchmarkDataset


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
# Resolve the project root relative to this script so it works
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[1]

# Curated index and split directory used to build datasets.
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
SPLITS = ROOT / "data" / "interim" / "sdobenchmark" / "splits"

# Evaluate all saved row-level splits.
SPLIT_NAMES = ["train", "val", "test"]

# Keep the batch small so the check stays fast and easy to inspect.
BATCH_SIZE = 2
SHUFFLE = False
NUM_WORKERS = 0


def build_dataloader(csv_path: Path) -> tuple[SDOBenchmarkDataset, DataLoader]:
    """
    Build a dataset and DataLoader for one split CSV.

    Args:
        csv_path (Path): Path to the split CSV file.

    Returns:
        tuple[SDOBenchmarkDataset, DataLoader]:
            - Instantiated dataset
            - DataLoader wrapping that dataset
    """
    dataset = SDOBenchmarkDataset(csv_path, INDEX)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
    )
    return dataset, dataloader


def inspect_split(split_name: str) -> None:
    """
    Load one batch from a dataset split and print its basic properties.

    Args:
        split_name (str): Split name, such as 'train', 'val', or 'test'.
    """
    csv_path = SPLITS / f"{split_name}.csv"
    dataset, dataloader = build_dataloader(csv_path)

    x, y = next(iter(dataloader))

    print(
        split_name,
        "len:", len(dataset),
        "batch x:", tuple(x.shape),
        "batch y:", y.tolist(),
    )


def main() -> None:
    """
    Run a batch-loading sanity check across all saved dataset splits.
    """
    for split_name in SPLIT_NAMES:
        inspect_split(split_name)


if __name__ == "__main__":
    main()