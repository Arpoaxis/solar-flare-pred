from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
SPLITS_DIR = ROOT / "data" / "interim" / "sdobenchmark" / "splits"

CHRONO_TRAIN = SPLITS_DIR / "chrono-train.csv"
CHRONO_VAL = SPLITS_DIR / "chrono-val.csv"
CHRONO_TEST = SPLITS_DIR / "chrono-test.csv"


LABEL_COL = "label_m1p"
AR_COL = "ar"
TIME_COL = "start"
COMPLETE_COL = "is_complete_40"


def summarize_split(df: pd.DataFrame, name: str) -> None:
    rows = len(df)
    pos = int(df[LABEL_COL].sum())
    rate = (pos / rows) if rows else 0.0
    ars = df[AR_COL].nunique() if rows else 0
    tmin = df["start_dt"].min() if rows else pd.NaT
    tmax = df["start_dt"].max() if rows else pd.NaT

    print(
        f"{name}: rows={rows}, pos={pos}, pos_rate={rate:.4f}, "
        f"ARs={ars}, start_min={tmin}, start_max={tmax}"
    )


def print_overlap(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> None:
    overlap = set(a[AR_COL].unique()) & set(b[AR_COL].unique())
    print(f"AR overlap {name_a}∩{name_b}: {len(overlap)}")


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    start = pd.Timestamp(start).to_period("M").to_timestamp()
    end = pd.Timestamp(end).to_period("M").to_timestamp()
    return pd.date_range(start, end, freq="MS")


def ownership_split_strict(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    ar_earliest: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    AR ownership rule:
      - AR belongs to left side if earliest sample timestamp < cutoff
      - AR belongs to right side if earliest sample timestamp >= cutoff

    Strict temporal purity:
      - left split keeps only rows with start_dt < cutoff
      - right split keeps only rows with start_dt >= cutoff

    Any rows that would violate the temporal side after AR assignment are purged.
    """
    left_ars = set(ar_earliest[ar_earliest < cutoff].index)
    right_ars = set(ar_earliest[ar_earliest >= cutoff].index)

    left = df[df[AR_COL].isin(left_ars) & (df["start_dt"] < cutoff)].copy()
    right = df[df[AR_COL].isin(right_ars) & (df["start_dt"] >= cutoff)].copy()
    return left, right


def build_candidate_table(
    df: pd.DataFrame,
    ar_earliest: pd.Series,
    candidate_cutoffs: pd.DatetimeIndex,
    min_left_rows: int,
    min_right_rows: int,
    min_left_pos: int,
    min_right_pos: int,
) -> pd.DataFrame:
    rows = []

    for cutoff in candidate_cutoffs:
        left, right = ownership_split_strict(df, cutoff, ar_earliest)

        left_rows = len(left)
        right_rows = len(right)
        left_pos = int(left[LABEL_COL].sum())
        right_pos = int(right[LABEL_COL].sum())

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


def choose_latest_valid_cutoff(candidates: pd.DataFrame) -> pd.Timestamp:
    valid = candidates[candidates["ok"]].sort_values("cutoff")
    if valid.empty:
        raise RuntimeError(
            "No valid cutoff found. Relax the minimum row/positive constraints."
        )
    return pd.Timestamp(valid.iloc[-1]["cutoff"])


def save_split(df: pd.DataFrame, path: Path, split_name: str) -> None:
    out = df.copy()
    out["split"] = split_name

    preferred_cols = [
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
    cols = [c for c in preferred_cols if c in out.columns]
    out[cols].to_csv(path, index=False)


def main() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INDEX).copy()
    df["start_dt"] = pd.to_datetime(df[TIME_COL], errors="raise")
    df = df[df[COMPLETE_COL] == True].copy()

    print("=== Overall filtered dataset (complete samples only) ===")
    summarize_split(df, "all_complete")
    print(f"overall positives: {int(df[LABEL_COL].sum())}")
    print()
    print(f"AR count: {df[AR_COL].nunique()}")
    print(f"overall date range: {df['start_dt'].min()} -> {df['start_dt'].max()}")
    print()

    total_rows = len(df)
    total_pos = int(df[LABEL_COL].sum())
    ar_earliest = df.groupby(AR_COL)["start_dt"].min().sort_values()

    # -----------------------------
    # Step 1: choose chrono test cutoff
    # -----------------------------
    min_test_rows = max(200, int(round(total_rows * 0.08)))
    min_trainval_rows = max(500, int(round(total_rows * 0.50)))
    min_test_pos = max(50, min(150, int(round(total_pos * 0.20))))
    min_trainval_pos = max(100, int(round(total_pos * 0.50)))

    q60 = df["start_dt"].quantile(0.60)
    q90 = df["start_dt"].quantile(0.90)
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
    chrono_trainval, chrono_test = ownership_split_strict(df, test_cutoff, ar_earliest)

    print(f"chosen test cutoff: {test_cutoff}")
    summarize_split(chrono_trainval, "chrono_trainval")
    summarize_split(chrono_test, "chrono_test")
    print()

    # -----------------------------
    # Step 2: choose chrono val cutoff inside pre-test era
    # -----------------------------
    pre_rows = len(chrono_trainval)
    pre_pos = int(chrono_trainval[LABEL_COL].sum())
    pre_ar_earliest = chrono_trainval.groupby(AR_COL)["start_dt"].min().sort_values()

    min_val_rows = max(100, int(round(pre_rows * 0.08)))
    min_core_train_rows = max(500, int(round(pre_rows * 0.65)))
    min_val_pos = max(30, min(80, int(round(pre_pos * 0.12))))
    min_core_train_pos = max(80, int(round(pre_pos * 0.60)))

    q70_pre = chrono_trainval["start_dt"].quantile(0.70)
    q95_pre = chrono_trainval["start_dt"].quantile(0.95)
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
    chrono_train, chrono_val = ownership_split_strict(
        chrono_trainval, val_cutoff, pre_ar_earliest
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