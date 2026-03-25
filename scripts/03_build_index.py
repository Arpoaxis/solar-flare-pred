"""
Build a filtered index of complete SDOBenchmark samples.

This script reads the SDOBenchmark metadata for the training and test
partitions, combines them into a single dataframe, derives a binary
M1+ flare label from peak flux, parses active-region and sample identifiers,
and records the filesystem location for each sample directory.

Only samples containing the full expected set of 40 JPG images
(4 timestamps × 10 modalities) are retained in the exported index.

Inputs:
    - training/meta_data.csv
    - test/meta_data.csv

Output:
    - data/interim/sdobenchmark/index.parquet
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
# Resolve the project root relative to this script so the code works
# regardless of the current working directory used to launch it.
ROOT = Path(__file__).resolve().parents[1]

# SDOBenchmark raw dataset root.
DATA_ROOT = ROOT / "data" / "external" / "sdobenchmark" / "raw" / "SDOBenchmark-data-full"

# Source metadata files.
TRAIN_CSV = DATA_ROOT / "training" / "meta_data.csv"
TEST_CSV = DATA_ROOT / "test" / "meta_data.csv"

# Output location for the curated index.
OUT_DIR = ROOT / "data" / "interim" / "sdobenchmark"
OUT_PATH = OUT_DIR / "index.parquet"

# GOES threshold for an M1.0 flare in W/m^2.
M1_THRESHOLD = 1e-5

# Each complete sample should contain 4 timestamps × 10 modalities.
EXPECTED_IMAGE_COUNT = 40


def load_split_metadata(csv_path, split_name):
    """
    Load metadata for one official dataset split and annotate its split label.

    Args:
        csv_path (Path): Path to the metadata CSV file.
        split_name (str): Split name to assign, such as 'training' or 'test'.

    Returns:
        pd.DataFrame: Loaded metadata with an added split column.
    """
    df = pd.read_csv(csv_path)
    df["split"] = split_name
    return df


def build_sample_dir(row):
    """
    Construct the expected filesystem path for a sample directory.

    Args:
        row (pd.Series): Dataframe row containing split, ar, and sample.

    Returns:
        Path: Path to the sample directory on disk.
    """
    return DATA_ROOT / row["split"] / row["ar"] / row["sample"]


def count_jpg_files(path):
    """
    Count JPG images in a sample directory.

    Args:
        path (Path): Directory path for one sample.

    Returns:
        int: Number of JPG files found in the directory.
    """
    return len(list(path.glob("*.jpg")))


def main():
    """
    Build and save a filtered index containing only complete samples.
    """
    print("Dataset root exists:", DATA_ROOT.exists())

    # Load metadata for the official training and test partitions.
    train_df = load_split_metadata(TRAIN_CSV, "training")
    test_df = load_split_metadata(TEST_CSV, "test")

    print("train rows:", len(train_df))
    print("test rows:", len(test_df))
    print("train columns:", list(train_df.columns))

    # Combine both official partitions into one index while preserving
    # the original split label for later path reconstruction.
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Create the binary target:
    # 1 = at least M1.0 flare, 0 = below M1.0.
    df["label_m1p"] = (df["peak_flux"] >= M1_THRESHOLD).astype(int)

    print("total rows:", len(df))
    print("M1+ positives:", int(df["label_m1p"].sum()))
    print("M1+ rate:", float(df["label_m1p"].mean()))

    # The dataset ID is assumed to follow the pattern:
    #     <active_region>_<sample_identifier>
    # Split that value into separate fields for easier downstream use.
    df["ar"] = df["id"].str.split("_").str[0]
    df["sample"] = df["id"].str.split("_", n=1).str[1]

    print(df[["id", "ar", "sample"]].head())

    # Build the expected path to each sample directory.
    df["sample_dir"] = df.apply(build_sample_dir, axis=1)

    # Each valid sample is expected to contain exactly 40 JPG images:
    # 4 timestamps × 10 modalities.
    df["num_images"] = df["sample_dir"].apply(count_jpg_files)
    df["is_complete_40"] = df["num_images"] == EXPECTED_IMAGE_COUNT

    print(df["num_images"].value_counts().head(10))
    print("complete_40:", int(df["is_complete_40"].sum()), "/", len(df))

    # Retain only complete samples for downstream modeling.
    index_df = df[df["is_complete_40"]].copy()

    # Convert Path objects to strings for parquet compatibility.
    index_df["sample_dir"] = index_df["sample_dir"].astype(str)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_df.to_parquet(OUT_PATH, index=False)

    print("Wrote:", OUT_PATH)
    print("Rows in index:", len(index_df))


if __name__ == "__main__":
    main()