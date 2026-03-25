"""
Load one sample into a [T, C, H, W] tensor and verify its structure.

This script selects the first sample from the training split, resolves its
sample directory through the curated index, loads the expected JPG files for
all timestamps and channels, and stacks them into a spatiotemporal tensor.

The final tensor has shape:
    [T, C, H, W] = [4, 10, 256, 256]

This serves as a sanity check for downstream dataset loading and confirms
that a complete sample can be assembled into the expected model input format.

Inputs:
    - data/interim/sdobenchmark/index.parquet
    - data/interim/sdobenchmark/splits/train.csv

Output:
    - Printed sample ID and label
    - Printed missing-file summary
    - Printed per-timestep mean values
    - Printed final tensor shape, dtype, and value range
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

# Expected channel order for one sample tensor.
CHANNELS = [
    "94",
    "131",
    "171",
    "193",
    "211",
    "304",
    "335",
    "1700",
    "continuum",
    "magnetogram",
]

EXPECTED_TIMESTEPS = 4
EXPECTED_CHANNELS = 10
EXPECTED_IMAGE_SHAPE = (1, 256, 256)
MAX_MISSING_PRINT = 10
IMAGE_GLOB = "*.jpg"


def get_first_training_sample(split_path: Path) -> tuple[str, int]:
    """
    Return the first sample ID and label from the training split CSV.

    Args:
        split_path (Path): Path to the training split CSV.

    Returns:
        tuple[str, int]:
            - Sample ID
            - Binary label for that sample
    """
    split_df = pd.read_csv(split_path)
    row = split_df.iloc[0]
    return str(row["id"]), int(row["label_m1p"])


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
    match = index_df[index_df["id"].astype(str) == sample_id]

    if match.empty:
        raise KeyError(f"Sample ID not found in index: {sample_id}")

    return Path(match.iloc[0]["sample_dir"])


def get_sample_timestamps(sample_dir: Path) -> list[str]:
    """
    Collect sorted timestamps from JPG filenames in a sample directory.

    Filenames are assumed to follow the pattern:
        <timestamp>__<channel>.jpg

    Lexical sorting is valid because timestamps use a sortable ISO-like format.

    Args:
        sample_dir (Path): Directory containing sample image files.

    Returns:
        list[str]: Sorted unique timestamps found in the sample directory.
    """
    files = sorted(sample_dir.glob(IMAGE_GLOB))
    timestamps = sorted({path.stem.split("__")[0] for path in files})
    return timestamps


def load_channel_image(image_path: Path) -> torch.Tensor:
    """
    Load one JPG file and convert it to a float tensor in the [0, 1] range.

    If the image is stored with three channels, only one channel is kept
    because these dataset JPGs are typically grayscale content saved in
    redundant RGB format.

    Args:
        image_path (Path): Path to the JPG file.

    Returns:
        torch.Tensor: Single-channel float tensor with shape [1, 256, 256].
    """
    img = read_image(str(image_path))

    if img.shape[0] == 3:
        img = img[0:1]

    img = img.float() / 255.0

    if tuple(img.shape) != EXPECTED_IMAGE_SHAPE:
        raise ValueError(
            f"Unexpected image shape for {image_path.name}: {tuple(img.shape)}"
        )

    return img


def load_sample_tensor(
    sample_dir: Path,
    timestamps: list[str],
    channels: list[str],
) -> tuple[torch.Tensor, list[str]]:
    """
    Load a complete sample into a [T, C, H, W] tensor.

    For each timestamp, images are loaded in the provided channel order and
    concatenated into a [C, H, W] frame. Frames are then stacked over time.

    Args:
        sample_dir (Path): Directory containing the sample JPG files.
        timestamps (list[str]): Sorted timestamps to load.
        channels (list[str]): Expected channel order.

    Returns:
        tuple[torch.Tensor, list[str]]:
            - Sample tensor with shape [T, C, H, W]
            - List of any missing filenames encountered

    Raises:
        FileNotFoundError: If any expected channel file is missing at a timestamp.
    """
    frames = []
    missing_files = []

    for timestamp in timestamps:
        channel_tensors = []

        for channel in channels:
            image_path = sample_dir / f"{timestamp}__{channel}.jpg"

            if not image_path.exists():
                missing_files.append(image_path.name)
                continue

            img = load_channel_image(image_path)
            channel_tensors.append(img)

        if len(channel_tensors) != len(channels):
            missing_count = len(channels) - len(channel_tensors)
            raise FileNotFoundError(
                f"Missing {missing_count} files at timestamp {timestamp}: "
                f"{missing_files[:MAX_MISSING_PRINT]}"
            )

        frame = torch.cat(channel_tensors, dim=0)  # [C, H, W]
        frames.append(frame)

    sample_tensor = torch.stack(frames, dim=0)  # [T, C, H, W]
    return sample_tensor, missing_files


def find_missing_files(
    sample_dir: Path,
    timestamps: list[str],
    channels: list[str],
) -> list[str]:
    """
    Re-scan the sample directory for any missing expected files.

    This provides a simple explicit report after the main tensor-loading step.

    Args:
        sample_dir (Path): Directory containing the sample JPG files.
        timestamps (list[str]): Expected timestamps.
        channels (list[str]): Expected channels.

    Returns:
        list[str]: Missing filenames.
    """
    missing_files = []

    for timestamp in timestamps:
        for channel in channels:
            image_path = sample_dir / f"{timestamp}__{channel}.jpg"
            if not image_path.exists():
                missing_files.append(image_path.name)

    return missing_files


def main() -> None:
    """
    Load one sample into a [T, C, H, W] tensor and verify its structure.
    """
    # Use the first sample from the training split as a quick integrity check.
    sample_id, label = get_first_training_sample(SPLIT)
    sample_dir = get_sample_dir(INDEX, sample_id)

    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    # Collect timestamps from filenames and validate the expected sample layout.
    timestamps = get_sample_timestamps(sample_dir)

    assert len(timestamps) == EXPECTED_TIMESTEPS, (
        f"expected {EXPECTED_TIMESTEPS} timestamps, got {len(timestamps)}"
    )
    assert len(CHANNELS) == EXPECTED_CHANNELS, (
        f"expected {EXPECTED_CHANNELS} channels, got {len(CHANNELS)}"
    )

    # Load the complete sample in [T, C, H, W] format.
    x, _ = load_sample_tensor(sample_dir, timestamps, CHANNELS)

    # Re-scan for missing files to print an explicit completeness summary.
    missing = find_missing_files(sample_dir, timestamps, CHANNELS)

    print("missing count:", len(missing))
    if missing:
        print("missing examples:", missing[:MAX_MISSING_PRINT])

    assert len(missing) == 0

    # Compute a simple per-timestep summary to confirm the tensor contains
    # nontrivial image content across all four time steps.
    means = [float(x[t].mean()) for t in range(x.shape[0])]
    print("per-timestep mean:", means)

    print("id:", sample_id)
    print("y:", label)
    print("x shape:", tuple(x.shape), "dtype:", x.dtype)
    print("x min/max:", float(x.min()), float(x.max()))


if __name__ == "__main__":
    main()