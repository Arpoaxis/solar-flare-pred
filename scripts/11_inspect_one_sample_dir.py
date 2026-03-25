"""
Inspect the timestamp and channel structure of one training sample.

This script selects the first sample ID from the training split, looks up
its sample directory in the curated index, lists the JPG files in that
directory, and summarizes the number of unique timestamps and channels
present in the sample.

It also checks for unexpected filename patterns and reports any files
that do not match the expected:
    <timestamp>__<channel>.jpg

Inputs:
    - data/interim/sdobenchmark/index.parquet
    - data/interim/sdobenchmark/splits/train.csv

Output:
    - Printed sample directory summary
    - Printed unique timestamps and channels
    - Printed file counts per timestamp
    - Printed unexpected filenames, if any
"""

from collections import Counter
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
# Resolve the project root relative to this script so it works
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[1]

# Curated sample index and training split file.
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
SPLIT = ROOT / "data" / "interim" / "sdobenchmark" / "splits" / "train.csv"

# Expected file extension for per-sample images.
IMAGE_GLOB = "*.jpg"

# Limit how many malformed filenames are previewed.
MAX_BAD_FILENAMES_PRINT = 10


def get_first_training_sample_id(split_path):
    """
    Return the first sample ID from the training split CSV.

    Args:
        split_path (Path): Path to the training split CSV.

    Returns:
        str: First sample ID in the split file.
    """
    split_df = pd.read_csv(split_path)
    return str(split_df.iloc[0]["id"])


def get_sample_dir(index_path, sample_id):
    """
    Look up the sample directory for a given sample ID in the curated index.

    Args:
        index_path (Path): Path to the curated index parquet file.
        sample_id (str): Sample identifier to search for.

    Returns:
        Path: Filesystem path to the sample directory.
    """
    index_df = pd.read_parquet(index_path)
    row = index_df[index_df["id"].astype(str) == sample_id].iloc[0]
    return Path(row["sample_dir"])


def parse_timestamp_channel_pairs(files):
    """
    Parse timestamp and channel pairs from sample image filenames.

    Filenames are expected to follow the pattern:
        <timestamp>__<channel>.jpg

    The channel is kept as a string because some modalities are numeric
    (for example, '131') while others are named text labels
    (for example, 'continuum').

    Args:
        files (list[Path]): JPG files in the sample directory.

    Returns:
        tuple[list[tuple[str, str]], list[str]]:
            - Parsed (timestamp, channel) pairs
            - Filenames that do not match the expected pattern
    """
    pairs = []
    bad_filenames = []

    for path in files:
        parts = path.stem.split("__")

        if len(parts) != 2:
            bad_filenames.append(path.name)
            continue

        timestamp, channel = parts[0], parts[1]
        pairs.append((timestamp, channel))

    return pairs, bad_filenames


def main():
    """
    Inspect the timestamp-channel structure of one training sample.
    """
    # Use the first sample from the training split as a quick integrity check.
    sample_id = get_first_training_sample_id(SPLIT)
    sample_dir = get_sample_dir(INDEX, sample_id)

    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    # List all JPG files for the selected sample.
    files = sorted(sample_dir.glob(IMAGE_GLOB))

    print("id:", sample_id)
    print("sample_dir:", sample_dir)
    print("num_files:", len(files))

    # Parse filename components to verify timestamp and channel coverage
    # and to detect any malformed filenames.
    pairs, bad_filenames = parse_timestamp_channel_pairs(files)

    timestamps = sorted({timestamp for timestamp, _ in pairs})
    channels = sorted({channel for _, channel in pairs})

    print("unique timestamps:", len(timestamps), timestamps)
    print("unique channels:", len(channels), channels)

    # Count how many files appear for each timestamp to confirm the
    # expected per-timestamp modality coverage.
    counts = Counter(timestamp for timestamp, _ in pairs)
    print("files per timestamp:", dict(counts))

    if bad_filenames:
        print("unexpected filenames:", bad_filenames[:MAX_BAD_FILENAMES_PRINT])


if __name__ == "__main__":
    main()