"""
Load and sanity-check one JPG image from the training split.

This script selects the first sample ID from the training split, resolves
its sample directory using the curated index, loads one JPG image from that
directory, and verifies that the image can be converted into the expected
model-ready tensor format.

The final check confirms that the image is represented as a single-channel
256×256 float tensor scaled to the [0, 1] range.

Inputs:
    - data/interim/sdobenchmark/index.parquet
    - data/interim/sdobenchmark/splits/train.csv

Output:
    - Printed sample ID and JPG filename
    - Printed raw image shape and dtype
    - Printed processed image shape, dtype, and value range
"""

from pathlib import Path

import pandas as pd
import torch
from torchvision.io import read_image


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
# Resolve the project root relative to this script so it works
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[1]

# Curated sample index and training split file.
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
SPLIT = ROOT / "data" / "interim" / "sdobenchmark" / "splits" / "train.csv"

# Expected image pattern and output tensor shape.
IMAGE_GLOB = "*.jpg"
EXPECTED_SHAPE = (1, 256, 256)


def get_first_training_sample_id(split_path: Path) -> str:
    """
    Return the first sample ID from the training split CSV.

    Args:
        split_path (Path): Path to the training split CSV.

    Returns:
        str: First sample ID in the split file.
    """
    split_df = pd.read_csv(split_path)
    return str(split_df.iloc[0]["id"])


def get_sample_dir(index_path: Path, sample_id: str) -> Path:
    """
    Look up the sample directory for a given sample ID in the curated index.

    Args:
        index_path (Path): Path to the curated index parquet file.
        sample_id (str): Sample identifier to search for.

    Returns:
        Path: Filesystem path to the sample directory.
    """
    index_df = pd.read_parquet(index_path)
    row = index_df[index_df["id"].astype(str) == sample_id]

    if row.empty:
        raise KeyError(f"Sample ID not found in index: {sample_id}")

    return Path(row.iloc[0]["sample_dir"])


def get_first_jpg(sample_dir: Path) -> Path:
    """
    Return the first JPG file found in a sample directory.

    Args:
        sample_dir (Path): Directory containing sample image files.

    Returns:
        Path: First JPG path in sorted order.
    """
    jpg_files = sorted(sample_dir.glob(IMAGE_GLOB))

    if not jpg_files:
        raise FileNotFoundError(f"No JPG files found in sample directory: {sample_dir}")

    return jpg_files[0]


def preprocess_image(img: torch.Tensor) -> torch.Tensor:
    """
    Convert a loaded image tensor into the expected model-ready format.

    The image is converted to a single-channel float tensor in the [0, 1]
    range. If the JPG is stored as RGB, only one channel is kept because
    the channels are typically identical for this dataset.

    Args:
        img (torch.Tensor): Raw image tensor with shape [C, H, W].

    Returns:
        torch.Tensor: Processed float tensor scaled to [0, 1].
    """
    if img.shape[0] == 3:
        img = img[0:1]

    img = img.float() / 255.0
    return img


def main() -> None:
    """
    Load one sample image and verify its basic tensor properties.
    """
    # Use the first sample from the training split as a quick image-loading check.
    sample_id = get_first_training_sample_id(SPLIT)
    sample_dir = get_sample_dir(INDEX, sample_id)

    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    # Select one JPG image from the sample directory and load it as a tensor.
    jpg_path = get_first_jpg(sample_dir)
    img = read_image(str(jpg_path))  # Raw tensor is typically uint8 with shape [C, H, W].

    print("id:", sample_id)
    print("jpg:", jpg_path.name)
    print("shape:", tuple(img.shape), "dtype:", img.dtype)

    # Convert the raw image to the expected model input format.
    img = preprocess_image(img)

    print(
        "after:", tuple(img.shape),
        "dtype:", img.dtype,
        "min/max:", float(img.min()), float(img.max()),
    )

    assert img.shape == EXPECTED_SHAPE, f"unexpected shape {tuple(img.shape)}"


if __name__ == "__main__":
    main()