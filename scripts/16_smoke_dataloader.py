"""
Run a quick DataLoader sanity check for the SDOBenchmark dataset.

This script instantiates the custom SDOBenchmarkDataset using the saved
training split and curated index, wraps it in a PyTorch DataLoader, and
retrieves one batch to confirm that dataset loading works as expected.

It is intended as a lightweight integration check between:
    - the split CSV
    - the curated index
    - the custom dataset class
    - the PyTorch DataLoader

Inputs:
    - data/interim/sdobenchmark/index.parquet
    - data/interim/sdobenchmark/splits/train.csv

Output:
    - Printed batch tensor shape and dtype
    - Printed batch labels
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

# Curated index and training split used to build the dataset.
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
TRAIN = ROOT / "data" / "interim" / "sdobenchmark" / "splits" / "train.csv"

# Small batch size keeps this script fast and easy to inspect.
BATCH_SIZE = 2

# Use single-process loading for a simple first-pass sanity check.
NUM_WORKERS = 0

# Pin memory can help host-to-device transfer performance when training on CUDA.
PIN_MEMORY = True

# Shuffle is enabled here to mimic the typical training-loader setup.
SHUFFLE = True


def build_dataloader() -> DataLoader:
    """
    Create a DataLoader for a quick training-split sanity check.

    Returns:
        DataLoader: DataLoader wrapping the SDOBenchmark training dataset.
    """
    dataset = SDOBenchmarkDataset(TRAIN, INDEX)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def main() -> None:
    """
    Load one batch from the training DataLoader and print its basic properties.
    """
    dataloader = build_dataloader()

    # Pull one batch to confirm that dataset indexing, collation, and tensor
    # shapes behave as expected.
    x, y = next(iter(dataloader))

    print("batch x:", tuple(x.shape), x.dtype)
    print("batch y:", y)


if __name__ == "__main__":
    main()