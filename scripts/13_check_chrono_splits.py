"""
Validate the saved chronological train/validation/test splits.

This script loads the previously generated chronological split CSV files,
summarizes each split, checks for active-region (AR) overlap, and verifies
that the splits are strictly ordered in time.

The checks confirm that the saved chronological splits are:
    - AR-disjoint
    - strictly time ordered
    - ready for downstream model development and evaluation

Inputs:
    - data/interim/sdobenchmark/splits/chrono-train.csv
    - data/interim/sdobenchmark/splits/chrono-val.csv
    - data/interim/sdobenchmark/splits/chrono-test.csv

Output:
    - Printed split summaries
    - Printed AR overlap checks
    - Printed strict time-order checks
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
# Resolve the project root relative to this script so it works
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "interim" / "sdobenchmark" / "splits"

CHRONO_TRAIN = SPLITS_DIR / "chrono-train.csv"
CHRONO_VAL = SPLITS_DIR / "chrono-val.csv"
CHRONO_TEST = SPLITS_DIR / "chrono-test.csv"

LABEL_COL = "label_m1p"
AR_COL = "ar"
TIME_COL = "start"
PARSED_TIME_COL = "start_dt"


def load_split(path: Path) -> pd.DataFrame:
    """
    Load one chronological split CSV and parse its start timestamp.

    Args:
        path (Path): Path to the split CSV file.

    Returns:
        pd.DataFrame: Loaded split dataframe with parsed datetime column.
    """
    df = pd.read_csv(path).copy()
    df[PARSED_TIME_COL] = pd.to_datetime(df[TIME_COL], errors="raise")
    return df


def summarize_split(df: pd.DataFrame, name: str) -> None:
    """
    Print summary statistics for a chronological split.

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


def count_ar_overlap(left_df: pd.DataFrame, right_df: pd.DataFrame) -> int:
    """
    Count overlapping active regions between two splits.

    Args:
        left_df (pd.DataFrame): First split dataframe.
        right_df (pd.DataFrame): Second split dataframe.

    Returns:
        int: Number of overlapping active regions.
    """
    return len(set(left_df[AR_COL].unique()) & set(right_df[AR_COL].unique()))


def print_overlap(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> int:
    """
    Print and return AR overlap count between two splits.

    Args:
        left_df (pd.DataFrame): First split dataframe.
        right_df (pd.DataFrame): Second split dataframe.
        left_name (str): Name of the first split.
        right_name (str): Name of the second split.

    Returns:
        int: Number of overlapping active regions.
    """
    overlap = count_ar_overlap(left_df, right_df)
    print(f"AR overlap {left_name}∩{right_name}: {overlap}")
    return overlap


def check_strict_time_order(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[bool, bool, bool]:
    """
    Check strict chronological ordering across train, validation, and test.

    Args:
        train_df (pd.DataFrame): Training split.
        val_df (pd.DataFrame): Validation split.
        test_df (pd.DataFrame): Test split.

    Returns:
        tuple[bool, bool, bool]:
            - Whether train ends before validation begins
            - Whether validation ends before test begins
            - Whether train ends before test begins
    """
    train_max = train_df[PARSED_TIME_COL].max()
    val_min = val_df[PARSED_TIME_COL].min()
    val_max = val_df[PARSED_TIME_COL].max()
    test_min = test_df[PARSED_TIME_COL].min()

    print("=== Strict time-order checks ===")
    print(f"train max start: {train_max}")
    print(f"val min start:   {val_min}")
    print(f"val max start:   {val_max}")
    print(f"test min start:  {test_min}")
    print()

    ok_train_val = train_max < val_min
    ok_val_test = val_max < test_min
    ok_train_test = train_max < test_min

    print(f"train before val:  {ok_train_val}")
    print(f"val before test:   {ok_val_test}")
    print(f"train before test: {ok_train_test}")
    print()

    return ok_train_val, ok_val_test, ok_train_test


def main() -> None:
    """
    Validate the saved chronological split files.
    """
    # Load all previously saved chronological splits.
    train_df = load_split(CHRONO_TRAIN)
    val_df = load_split(CHRONO_VAL)
    test_df = load_split(CHRONO_TEST)

    # Report split size, class balance, AR count, and temporal coverage.
    print("=== Chronological split summary ===")
    summarize_split(train_df, "chrono_train")
    summarize_split(val_df, "chrono_val")
    summarize_split(test_df, "chrono_test")
    print()

    # Check that active regions do not appear in more than one split.
    print("=== AR overlap checks ===")
    overlap_train_val = print_overlap(train_df, val_df, "train", "val")
    overlap_train_test = print_overlap(train_df, test_df, "train", "test")
    overlap_val_test = print_overlap(val_df, test_df, "val", "test")
    print()

    # Check that split time ranges are strictly ordered.
    ok_train_val, ok_val_test, ok_train_test = check_strict_time_order(
        train_df,
        val_df,
        test_df,
    )

    if not (ok_train_val and ok_val_test and ok_train_test):
        raise SystemExit("Chronological order check FAILED.")

    if overlap_train_val != 0 or overlap_train_test != 0 or overlap_val_test != 0:
        raise SystemExit("AR overlap check FAILED.")

    print("All chronological split checks passed.")


if __name__ == "__main__":
    main()