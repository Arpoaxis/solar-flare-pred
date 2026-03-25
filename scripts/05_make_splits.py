"""
Create AR-disjoint train/validation/test splits from the curated index.

This script builds leakage-safe dataset splits for solar flare prediction.
It uses the official SDOBenchmark training partition as the source for
train/validation selection and preserves the official test partition unchanged.

Validation active regions (ARs) are selected from the most recent portion
of the training data. The candidate pool is expanded backward in time until
it contains both:
    - at least the target number of validation ARs
    - at least the target number of positive samples

This produces an AR-disjoint validation set with a more useful number of
positive examples for model development.

Input:
    - data/interim/sdobenchmark/index.parquet

Outputs:
    - data/interim/sdobenchmark/splits/train.csv
    - data/interim/sdobenchmark/splits/val.csv
    - data/interim/sdobenchmark/splits/test.csv
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
SPLIT_DIR = ROOT / "data" / "interim" / "sdobenchmark" / "splits"

VAL_FRACTION = 0.10
TARGET_POS = 40

OUTPUT_COLUMNS = ["id", "ar", "start", "end", "peak_flux", "label_m1p"]


def print_split_stats(name, df):
    """
    Print basic summary statistics for a dataset split.

    Args:
        name (str): Split name to display.
        df (pd.DataFrame): Dataframe for the split.
    """
    positives = int(df["label_m1p"].sum())
    rows = len(df)
    pos_rate = positives / rows if rows > 0 else 0.0
    ars = df["ar"].nunique()

    print(
        f"{name}: rows = {rows}, positives = {positives}, "
        f"pos_rate = {pos_rate:.4f}, ARs = {ars}"
    )


def get_ar_time_and_positive_stats(train_df):
    """
    Compute per-AR recency and positive-label counts.

    Args:
        train_df (pd.DataFrame): Official training partition.

    Returns:
        tuple[pd.Series, pd.Series]:
            - Latest observed timestamp per AR, sorted oldest to newest
            - Positive sample count per AR
    """
    ar_last_time = train_df.groupby("ar")["start_dt"].max().sort_values()
    ar_pos = train_df.groupby("ar")["label_m1p"].sum()

    return ar_last_time, ar_pos


def build_candidate_pool(ar_last_time, ar_pos, n_val, target_pos):
    """
    Expand a recent-AR candidate pool backward in time until it contains
    enough ARs and enough positives for validation selection.

    Args:
        ar_last_time (pd.Series): Latest observed time per AR, sorted oldest to newest.
        ar_pos (pd.Series): Positive sample count per AR.
        n_val (int): Target number of validation ARs.
        target_pos (int): Minimum target number of positive samples.

    Returns:
        tuple[list[str], int]:
            - Candidate AR list in newest-to-oldest selection order
            - Total positives contained in the candidate pool
    """
    ordered_ars = list(ar_last_time.index)  # oldest -> newest
    candidate_ars = []
    candidate_pos = 0

    for ar in reversed(ordered_ars):  # newest -> oldest
        candidate_ars.append(ar)
        candidate_pos += int(ar_pos.get(ar, 0))

        if len(candidate_ars) >= n_val and candidate_pos >= target_pos:
            break

    return candidate_ars, candidate_pos


def choose_validation_ars(candidate_ars, ar_pos, n_val, target_pos):
    """
    Select validation ARs from the candidate pool.

    Positive ARs are chosen first, prioritizing the newest available regions.
    If additional slots remain, they are filled using the newest negative-only ARs.

    Args:
        candidate_ars (list[str]): Candidate ARs in newest-to-oldest order.
        ar_pos (pd.Series): Positive sample count per AR.
        n_val (int): Target number of validation ARs.
        target_pos (int): Minimum target number of positive samples.

    Returns:
        list[str]: Selected validation ARs.
    """
    positive_ars = [ar for ar in candidate_ars if int(ar_pos.get(ar, 0)) > 0]
    negative_ars = [ar for ar in candidate_ars if int(ar_pos.get(ar, 0)) == 0]

    selected_ars = []
    selected_pos = 0

    for ar in positive_ars:
        if len(selected_ars) >= n_val:
            break

        selected_ars.append(ar)
        selected_pos += int(ar_pos.get(ar, 0))

        if selected_pos >= target_pos:
            break

    for ar in negative_ars:
        if len(selected_ars) >= n_val:
            break
        if ar not in selected_ars:
            selected_ars.append(ar)

    return selected_ars


def report_overlap_checks(train_rows, val_rows, test_rows):
    """
    Print leakage checks for AR and sample ID overlap across splits.

    Args:
        train_rows (pd.DataFrame): Training split rows.
        val_rows (pd.DataFrame): Validation split rows.
        test_rows (pd.DataFrame): Test split rows.
    """
    ar_overlap = len(set(train_rows["ar"]).intersection(set(val_rows["ar"])))
    print("AR overlap train∩val:", ar_overlap)

    id_overlaps = {
        "train∩val": len(set(train_rows["id"]) & set(val_rows["id"])),
        "train∩test": len(set(train_rows["id"]) & set(test_rows["id"])),
        "val∩test": len(set(val_rows["id"]) & set(test_rows["id"])),
    }
    print("ID overlaps:", id_overlaps)


def write_split_csvs(train_rows, val_rows, test_rows):
    """
    Save split CSV files for downstream data loading.

    Args:
        train_rows (pd.DataFrame): Training split rows.
        val_rows (pd.DataFrame): Validation split rows.
        test_rows (pd.DataFrame): Test split rows.
    """
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train_rows[OUTPUT_COLUMNS].to_csv(SPLIT_DIR / "train.csv", index=False)
    val_rows[OUTPUT_COLUMNS].to_csv(SPLIT_DIR / "val.csv", index=False)
    test_rows[OUTPUT_COLUMNS].to_csv(SPLIT_DIR / "test.csv", index=False)

    print("wrote:", SPLIT_DIR)
    print(
        "train rows:", len(train_rows),
        "val rows:", len(val_rows),
        "test rows:", len(test_rows),
    )


def main():
    """
    Build AR-disjoint train/validation/test splits and save them to CSV.
    """
    # Load the curated index containing complete samples only.
    df = pd.read_parquet(INDEX)

    # Use the official training partition as the source for train/validation
    # selection and preserve the official test partition unchanged.
    train_df = df[df["split"] == "training"].copy()
    test_rows = df[df["split"] == "test"].copy()

    # Convert timestamps so AR recency can be computed reliably.
    train_df["start_dt"] = pd.to_datetime(train_df["start"])

    all_train_ars = sorted(train_df["ar"].unique())
    n_val = int(round(VAL_FRACTION * len(all_train_ars)))

    print("training ARs:", len(all_train_ars))
    print("val ARs (10%):", n_val)

    ar_last_time, ar_pos = get_ar_time_and_positive_stats(train_df)

    candidate_ars, candidate_pos = build_candidate_pool(
        ar_last_time=ar_last_time,
        ar_pos=ar_pos,
        n_val=n_val,
        target_pos=TARGET_POS,
    )

    val_ar_list = choose_validation_ars(
        candidate_ars=candidate_ars,
        ar_pos=ar_pos,
        n_val=n_val,
        target_pos=TARGET_POS,
    )

    val_ars = set(val_ar_list)
    train_ars = set(all_train_ars) - val_ars

    val_positive_count = int(train_df[train_df["ar"].isin(val_ars)]["label_m1p"].sum())

    print("candidate ARs:", len(candidate_ars), "candidate positives:", candidate_pos)
    print("val ARs:", len(val_ars), "val positives:", val_positive_count)
    print("earliest AR last-time:", ar_last_time.iloc[0])
    print("latest AR last-time:", ar_last_time.iloc[-1])

    # Materialize row-level splits from the selected AR assignments.
    train_rows = train_df[train_df["ar"].isin(train_ars)].copy()
    val_rows = train_df[train_df["ar"].isin(val_ars)].copy()

    write_split_csvs(train_rows, val_rows, test_rows)
    report_overlap_checks(train_rows, val_rows, test_rows)

    print_split_stats("train", train_rows)
    print_split_stats("val", val_rows)
    print_split_stats("test", test_rows)

    ar_pos_sorted = train_df.groupby("ar")["label_m1p"].sum().sort_values(ascending=False)
    print("ARs with ≥1 positive:", int((ar_pos_sorted >= 1).sum()))
    print("Top 10 AR positive counts:\n", ar_pos_sorted.head(10))

    print(
        "SPLIT_POLICY: AR-disjoint; time-aware candidate pool expanded "
        "until >= TARGET_POS positives; TARGET_POS =",
        TARGET_POS,
    )


if __name__ == "__main__":
    main()