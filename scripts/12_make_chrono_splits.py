"""
Create leakage-safe chronological train/validation/test splits.

This script loads the curated SDOBenchmark index, keeps only complete
samples, and creates chronological splits using active-region ownership
based on each AR's earliest observed timestamp.

Split policy:
    - An AR belongs to the left side of a cutoff if its earliest sample
      timestamp is earlier than the cutoff.
    - An AR belongs to the right side of a cutoff if its earliest sample
      timestamp is on or after the cutoff.
    - After AR ownership is assigned, each split is made temporally pure by
      purging any rows that fall on the wrong side of the cutoff.

The workflow selects:
    1. A chronological test cutoff from the later portion of the dataset
    2. A chronological validation cutoff within the remaining pre-test era

Candidate cutoffs are evaluated against minimum row-count and positive-count
constraints so the resulting splits remain usable for model development.

Inputs:
    - data/interim/sdobenchmark/index.parquet

Outputs:
    - data/interim/sdobenchmark/splits/chrono-train.csv
    - data/interim/sdobenchmark/splits/chrono-val.csv
    - data/interim/sdobenchmark/splits/chrono-test.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
SPLITS_DIR = ROOT / "data" / "interim" / "sdobenchmark" / "splits"

CHRONO_TRAIN = SPLITS_DIR / "chrono-train.csv"
CHRONO_VAL = SPLITS_DIR / "chrono-val.csv"
CHRONO_TEST = SPLITS_DIR / "chrono-test.csv"

LABEL_COL = "label_m1p"
AR_COL = "ar"
TIME_COL = "start"
PARSED_TIME_COL = "start_dt"
COMPLETE_COL = "is_complete_40"

OUTPUT_COLUMNS = [
    "id",
    "ar",
    "start",
    "end",
    "peak_flux",
    "label_m1p",
    "sample_dir",
    "num_images",
    "is_complete_40",
    "split",
]


def summarize_split(df: pd.DataFrame, name: str) -> None:
    """
    Print summary statistics for a dataset split.

    Args:
        df (pd.DataFrame): Split dataframe.
        name (str): Split name to display.
    """
    rows = len(df)
    pos = int(df[LABEL_COL].sum())
    rate = (pos / rows) if rows else 0.0
    ars = df[AR_COL].nunique() if rows else 0
    tmin = df[PARSED_TIME_COL].min() if rows else pd.NaT
    tmax = df[PARSED_TIME_COL].max() if rows else pd.NaT

    print(
        f"{name}: rows={rows}, pos={pos}, pos_rate={rate:.4f}, "
        f"ARs={ars}, start_min={tmin}, start_max={tmax}"
    )


def print_overlap(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> None:
    """
    Print the number of overlapping active regions between two splits.

    Args:
        left_df (pd.DataFrame): First split dataframe.
        right_df (pd.DataFrame): Second split dataframe.
        left_name (str): Name of the first split.
        right_name (str): Name of the second split.
    """
    overlap = set(left_df[AR_COL].unique()) & set(right_df[AR_COL].unique())
    print(f"AR overlap {left_name}∩{right_name}: {len(overlap)}")


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """
    Create a monthly cutoff grid between two timestamps, inclusive.

    Args:
        start (pd.Timestamp): Start of the candidate range.
        end (pd.Timestamp): End of the candidate range.

    Returns:
        pd.DatetimeIndex: Month-start timestamps spanning the range.
    """
    start = pd.Timestamp(start).to_period("M").to_timestamp()
    end = pd.Timestamp(end).to_period("M").to_timestamp()
    return pd.date_range(start, end, freq="MS")


def split_by_ar_ownership_and_time(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    ar_earliest: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into left/right partitions using AR ownership and
    strict temporal purity.

    AR ownership rule:
        - AR belongs to the left side if earliest sample timestamp < cutoff
        - AR belongs to the right side if earliest sample timestamp >= cutoff

    Temporal purity rule:
        - Left split keeps only rows with start_dt < cutoff
        - Right split keeps only rows with start_dt >= cutoff

    Any rows that would violate the temporal side after AR assignment are
    purged from the final splits.

    Args:
        df (pd.DataFrame): Dataset to split.
        cutoff (pd.Timestamp): Candidate cutoff timestamp.
        ar_earliest (pd.Series): Earliest timestamp per AR.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - Left split
            - Right split
    """
    left_ars = set(ar_earliest[ar_earliest < cutoff].index)
    right_ars = set(ar_earliest[ar_earliest >= cutoff].index)

    left_df = df[df[AR_COL].isin(left_ars) & (df[PARSED_TIME_COL] < cutoff)].copy()
    right_df = df[df[AR_COL].isin(right_ars) & (df[PARSED_TIME_COL] >= cutoff)].copy()

    return left_df, right_df


def build_candidate_table(
    df: pd.DataFrame,
    ar_earliest: pd.Series,
    candidate_cutoffs: pd.DatetimeIndex,
    min_left_rows: int,
    min_right_rows: int,
    min_left_pos: int,
    min_right_pos: int,
) -> pd.DataFrame:
    """
    Evaluate candidate cutoffs against size and class-balance constraints.

    Args:
        df (pd.DataFrame): Dataset to split.
        ar_earliest (pd.Series): Earliest timestamp per AR.
        candidate_cutoffs (pd.DatetimeIndex): Candidate cutoff timestamps.
        min_left_rows (int): Minimum allowed rows on the left side.
        min_right_rows (int): Minimum allowed rows on the right side.
        min_left_pos (int): Minimum allowed positives on the left side.
        min_right_pos (int): Minimum allowed positives on the right side.

    Returns:
        pd.DataFrame: Candidate summary table with counts, rates, and validity.
    """
    rows = []

    for cutoff in candidate_cutoffs:
        left_df, right_df = split_by_ar_ownership_and_time(df, cutoff, ar_earliest)

        left_rows = len(left_df)
        right_rows = len(right_df)
        left_pos = int(left_df[LABEL_COL].sum())
        right_pos = int(right_df[LABEL_COL].sum())

        rows.append(
            {
                "cutoff": cutoff,
                "left_rows": left_rows,
                "right_rows": right_rows,
                "left_pos": left_pos,
                "right_pos": right_pos,
                "left_rate": (left_pos / left_rows) if left_rows else np.nan,
                "right_rate": (right_pos / right_rows) if right_rows else np.nan,
                "ok": (
                    left_rows >= min_left_rows
                    and right_rows >= min_right_rows
                    and left_pos >= min_left_pos
                    and right_pos >= min_right_pos
                ),
            }
        )

    return pd.DataFrame(rows)


def choose_latest_valid_cutoff(candidate_table: pd.DataFrame) -> pd.Timestamp:
    """
    Return the latest candidate cutoff that satisfies all constraints.

    Args:
        candidate_table (pd.DataFrame): Candidate cutoff summary table.

    Returns:
        pd.Timestamp: Latest valid cutoff.

    Raises:
        RuntimeError: If no candidate satisfies the constraints.
    """
    valid = candidate_table[candidate_table["ok"]].sort_values("cutoff")

    if valid.empty:
        raise RuntimeError(
            "No valid cutoff found. Relax the minimum row/positive constraints."
        )

    return pd.Timestamp(valid.iloc[-1]["cutoff"])


def save_split(df: pd.DataFrame, path: Path, split_name: str) -> None:
    """
    Save one chronological split to CSV.

    Args:
        df (pd.DataFrame): Split dataframe to save.
        path (Path): Output CSV path.
        split_name (str): Split name to write into the split column.
    """
    out = df.copy()
    out["split"] = split_name

    cols = [col for col in OUTPUT_COLUMNS if col in out.columns]
    out[cols].to_csv(path, index=False)


def main() -> None:
    """
    Build and save chronological train/validation/test splits.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the curated index and prepare timestamps for temporal splitting.
    df = pd.read_parquet(INDEX).copy()
    df[PARSED_TIME_COL] = pd.to_datetime(df[TIME_COL], errors="raise")

    # Keep only complete samples so chronological statistics and outputs are
    # based on the same modeling-ready dataset used elsewhere.
    df = df[df[COMPLETE_COL] == True].copy()

    print("=== Overall filtered dataset (complete samples only) ===")
    summarize_split(df, "all_complete")
    print(f"overall positives: {int(df[LABEL_COL].sum())}")
    print()
    print(f"AR count: {df[AR_COL].nunique()}")
    print(f"overall date range: {df[PARSED_TIME_COL].min()} -> {df[PARSED_TIME_COL].max()}")
    print()

    total_rows = len(df)
    total_pos = int(df[LABEL_COL].sum())

    # AR ownership is determined by each active region's earliest appearance.
    ar_earliest = df.groupby(AR_COL)[PARSED_TIME_COL].min().sort_values()

    # -----------------------------------------------------------------
    # Step 1: choose the chronological test cutoff
    # -----------------------------------------------------------------
    min_test_rows = max(200, int(round(total_rows * 0.08)))
    min_trainval_rows = max(500, int(round(total_rows * 0.50)))
    min_test_pos = max(50, min(150, int(round(total_pos * 0.20))))
    min_trainval_pos = max(100, int(round(total_pos * 0.50)))

    # Search for test cutoffs within the later portion of the timeline so
    # the test split resembles a future holdout rather than a random slice.
    q60 = df[PARSED_TIME_COL].quantile(0.60)
    q90 = df[PARSED_TIME_COL].quantile(0.90)
    test_candidates = month_starts(q60, q90)

    test_table = build_candidate_table(
        df=df,
        ar_earliest=ar_earliest,
        candidate_cutoffs=test_candidates,
        min_left_rows=min_trainval_rows,
        min_right_rows=min_test_rows,
        min_left_pos=min_trainval_pos,
        min_right_pos=min_test_pos,
    )

    print("=== Test cutoff candidates ===")
    print(test_table.to_string(index=False))
    print()

    test_cutoff = choose_latest_valid_cutoff(test_table)
    chrono_trainval, chrono_test = split_by_ar_ownership_and_time(
        df,
        test_cutoff,
        ar_earliest,
    )

    print(f"chosen test cutoff: {test_cutoff}")
    summarize_split(chrono_trainval, "chrono_trainval")
    summarize_split(chrono_test, "chrono_test")
    print()

    # -----------------------------------------------------------------
    # Step 2: choose the chronological validation cutoff inside the
    # remaining pre-test era
    # -----------------------------------------------------------------
    pre_rows = len(chrono_trainval)
    pre_pos = int(chrono_trainval[LABEL_COL].sum())
    pre_ar_earliest = chrono_trainval.groupby(AR_COL)[PARSED_TIME_COL].min().sort_values()

    min_val_rows = max(100, int(round(pre_rows * 0.08)))
    min_core_train_rows = max(500, int(round(pre_rows * 0.65)))
    min_val_pos = max(30, min(80, int(round(pre_pos * 0.12))))
    min_core_train_pos = max(80, int(round(pre_pos * 0.60)))

    # Search for validation cutoffs near the later part of the pre-test era
    # so validation remains more forward-looking than early historical data.
    q70_pre = chrono_trainval[PARSED_TIME_COL].quantile(0.70)
    q95_pre = chrono_trainval[PARSED_TIME_COL].quantile(0.95)
    val_candidates = month_starts(q70_pre, q95_pre)

    val_table = build_candidate_table(
        df=chrono_trainval,
        ar_earliest=pre_ar_earliest,
        candidate_cutoffs=val_candidates,
        min_left_rows=min_core_train_rows,
        min_right_rows=min_val_rows,
        min_left_pos=min_core_train_pos,
        min_right_pos=min_val_pos,
    )

    print("=== Validation cutoff candidates ===")
    print(val_table.to_string(index=False))
    print()

    val_cutoff = choose_latest_valid_cutoff(val_table)
    chrono_train, chrono_val = split_by_ar_ownership_and_time(
        chrono_trainval,
        val_cutoff,
        pre_ar_earliest,
    )

    print(f"chosen val cutoff: {val_cutoff}")
    print()

    print("=== Final chronological splits ===")
    summarize_split(chrono_train, "chrono_train")
    summarize_split(chrono_val, "chrono_val")
    summarize_split(chrono_test, "chrono_test")
    print()

    print("=== AR overlap checks ===")
    print_overlap(chrono_train, chrono_val, "train", "val")
    print_overlap(chrono_train, chrono_test, "train", "test")
    print_overlap(chrono_val, chrono_test, "val", "test")
    print()

    kept_rows = len(chrono_train) + len(chrono_val) + len(chrono_test)
    purged_rows = total_rows - kept_rows

    print("=== Purge summary ===")
    print(f"total complete rows: {total_rows}")
    print(f"rows kept in chrono splits: {kept_rows}")
    print(f"rows purged at cutoff boundaries: {purged_rows}")
    print()

    save_split(chrono_train, CHRONO_TRAIN, "chrono_train")
    save_split(chrono_val, CHRONO_VAL, "chrono_val")
    save_split(chrono_test, CHRONO_TEST, "chrono_test")

    print("Wrote:")
    print(CHRONO_TRAIN)
    print(CHRONO_VAL)
    print(CHRONO_TEST)


if __name__ == "__main__":
    main()