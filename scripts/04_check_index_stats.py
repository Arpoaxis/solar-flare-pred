"""
Summarize the filtered SDOBenchmark index by official dataset split.

This script loads the curated index of complete samples, converts the
sample start timestamp to datetime format, and reports split-level
summary statistics for dataset size, active-region coverage, class
balance, and temporal range.

Input:
    - data/interim/sdobenchmark/index.parquet

Output:
    - Printed summary table by split
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
# Resolve the project root relative to this script so it works
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"


def main():
    """
    Run the split-level summary workflow for the curated dataset index.
    """
    # Load the filtered index containing complete samples only.
    df = pd.read_parquet(INDEX)

    # Convert the raw start timestamp to datetime so temporal summaries
    # can be computed reliably.
    df["start_dt"] = pd.to_datetime(df["start"])

    # Summarize each official split by sample count, unique active regions,
    # class balance, and observed time range.
    summary = df.groupby("split").agg(
        rows=("id", "count"),
        ars=("ar", "nunique"),
        positives=("label_m1p", "sum"),
        pos_rate=("label_m1p", "mean"),
        start_min=("start_dt", "min"),
        start_max=("start_dt", "max"),
    )

    print(summary)


if __name__ == "__main__":
    main()