"""
Inspect a specific sample in the curated SDOBenchmark index.

This script locates the curated index parquet file, loads it into a
dataframe, searches for a known sample ID, identifies path-like columns,
and prints selected path-related fields for that sample if found.

Input:
    - data/interim/sdobenchmark/index.parquet
      or
    - data/interim/sdobenchmark/index/index.parquet

Output:
    - Printed index path used
    - Printed schema summary
    - Printed path-like field values for the selected sample
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
# Resolve the project root relative to this script so it works
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[1]

# Possible locations for the curated index file.
INDEX_CANDIDATES = [
    ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet",
    ROOT / "data" / "interim" / "sdobenchmark" / "index" / "index.parquet",
]

# Sample ID to inspect in the index.
SAMPLE_ID = "11390_2012_01_05_17_19_01_1"

# Keywords used to identify columns that may contain file or path data.
PATH_KEYWORDS = ("path", "file", "img", "image", "png", "npy", "npz")

# Limit how many matching columns are displayed.
MAX_PATH_COLS_PRINT = 30
MAX_ROW_FIELDS_PRINT = 20


def find_existing_index_path(candidates):
    """
    Return the first existing index path from a list of candidates.

    Args:
        candidates (list[Path]): Candidate parquet file locations.

    Returns:
        Path | None: First existing path, or None if none are found.
    """
    return next((path for path in candidates if path.exists()), None)


def find_path_like_columns(columns):
    """
    Identify columns whose names suggest file or path-related content.

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


def print_sample_path_fields(row_df, path_columns):
    """
    Print selected path-related fields from a single matching row.

    Args:
        row_df (pd.DataFrame): Single-row dataframe containing the sample.
        path_columns (list[str]): Path-like columns to inspect.
    """
    if len(row_df) != 1 or not path_columns:
        return

    print("\nSample row path fields (first 20):")
    row = row_df.iloc[0].to_dict()

    for col in path_columns[:MAX_ROW_FIELDS_PRINT]:
        print(f"{col} = {row.get(col)}")


def main():
    """
    Locate the curated index and inspect one sample's path-related fields.
    """
    # Find the first available curated index path.
    index_path = find_existing_index_path(INDEX_CANDIDATES)
    print("index_path:", index_path)

    if index_path is None:
        raise FileNotFoundError("Could not find index.parquet in expected locations.")

    # Load the curated index for inspection.
    df = pd.read_parquet(index_path)

    # Look up the requested sample ID.
    row = df[df["id"].astype(str) == SAMPLE_ID].head(1)

    print("found rows:", len(row))
    print("columns:", df.columns.tolist())

    # Identify columns that may contain file or path information.
    path_like_cols = find_path_like_columns(df.columns.tolist())
    print("path-like columns:", len(path_like_cols))
    print("path-like columns (first 30):", path_like_cols[:MAX_PATH_COLS_PRINT])

    # Print the selected sample's path-related fields if available.
    print_sample_path_fields(row, path_like_cols)


if __name__ == "__main__":
    main()