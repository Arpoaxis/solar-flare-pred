"""
Inspect the training split CSV used for model development.

This script performs a lightweight schema check on the training split file.
It reports basic dataset shape information, identifies likely label columns,
searches for columns that may contain file or image-path data, and prints
selected values from the first row for quick inspection.

Input:
    - data/interim/sdobenchmark/splits/train.csv

Output:
    - Printed schema and sample-row diagnostics
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
# Resolve the project root relative to this script so it works
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[1]

# Training split CSV to inspect.
CSV_PATH = ROOT / "data" / "interim" / "sdobenchmark" / "splits" / "train.csv"

# Common names that may indicate a target column.
LABEL_CANDIDATES = ("y", "label", "target", "label_m1p", "label_m1", "flare")

# Common substrings that may indicate a path-like or image-like column.
PATH_KEYWORDS = (
    "path",
    "file",
    "img",
    "image",
    "png",
    "npy",
    "npz",
    "t0",
    "t1",
    "t2",
    "t3",
    "c0",
    "c1",
    "channel",
)

# Limit how many first-row fields are printed during inspection.
MAX_PREVIEW_FIELDS = 25


def find_label_columns(columns):
    """
    Identify columns whose names suggest they may contain labels or targets.

    Args:
        columns (list[str]): Dataframe column names.

    Returns:
        list[str]: Matching label-like column names.
    """
    return [col for col in columns if col.lower() in LABEL_CANDIDATES]


def find_path_like_columns(columns):
    """
    Identify columns whose names suggest they may contain file or image data.

    Args:
        columns (list[str]): Dataframe column names.

    Returns:
        list[str]: Matching path-like column names.
    """
    return [
        col
        for col in columns
        if any(keyword in col.lower() for keyword in PATH_KEYWORDS)
    ]


def print_first_row_subset(df, selected_columns):
    """
    Print selected fields from the first row for quick manual inspection.

    Args:
        df (pd.DataFrame): Dataframe to inspect.
        selected_columns (list[str]): Columns to print from the first row.
    """
    if df.empty:
        print("\nDataset is empty; no row preview available.")
        return

    print("\nfirst row (subset):")
    row = df.iloc[0].to_dict()

    for col in selected_columns[:MAX_PREVIEW_FIELDS]:
        print(f"{col} = {row.get(col)}")


def main():
    """
    Run a quick schema and sample-value inspection on the training split CSV.
    """
    print("exists:", CSV_PATH.exists(), "| path:", CSV_PATH)

    # Load the split file for inspection.
    df = pd.read_csv(CSV_PATH)

    print("rows:", len(df))
    print("num_cols:", len(df.columns))
    print("columns:", df.columns.tolist())

    # Search for columns that may represent labels or targets.
    label_cols = find_label_columns(df.columns.tolist())
    print("label candidates:", label_cols)

    # Search for columns that may contain file paths, image references,
    # per-timestep fields, or channel-specific inputs.
    path_like_cols = find_path_like_columns(df.columns.tolist())
    print("path-like cols:", len(path_like_cols))
    print("path-like cols (first 20):", path_like_cols[:20])

    # Preview selected first-row values from the most relevant columns.
    preview_cols = label_cols + path_like_cols
    print_first_row_subset(df, preview_cols)


if __name__ == "__main__":
    main()