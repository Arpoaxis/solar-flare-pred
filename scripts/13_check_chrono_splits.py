from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "interim" / "sdobenchmark" / "splits"

CHRONO_TRAIN = SPLITS_DIR / "chrono-train.csv"
CHRONO_VAL = SPLITS_DIR / "chrono-val.csv"
CHRONO_TEST = SPLITS_DIR / "chrono-test.csv"

LABEL_COL = "label_m1p"
AR_COL = "ar"
TIME_COL = "start"


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["start_dt"] = pd.to_datetime(df[TIME_COL], errors="raise")
    return df


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


def main() -> None:
    train = load_split(CHRONO_TRAIN)
    val = load_split(CHRONO_VAL)
    test = load_split(CHRONO_TEST)

    print("=== Chronological split summary ===")
    summarize_split(train, "chrono_train")
    summarize_split(val, "chrono_val")
    summarize_split(test, "chrono_test")
    print()

    print("=== AR overlap checks ===")
    print_overlap(train, val, "train", "val")
    print_overlap(train, test, "train", "test")
    print_overlap(val, test, "val", "test")
    print()

    print("=== Strict time-order checks ===")
    train_max = train["start_dt"].max()
    val_min = val["start_dt"].min()
    val_max = val["start_dt"].max()
    test_min = test["start_dt"].min()

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

    if not (ok_train_val and ok_val_test and ok_train_test):
        raise SystemExit("Chronological order check FAILED.")

    overlap_train_val = len(set(train[AR_COL].unique()) & set(val[AR_COL].unique()))
    overlap_train_test = len(set(train[AR_COL].unique()) & set(test[AR_COL].unique()))
    overlap_val_test = len(set(val[AR_COL].unique()) & set(test[AR_COL].unique()))

    if overlap_train_val != 0 or overlap_train_test != 0 or overlap_val_test != 0:
        raise SystemExit("AR overlap check FAILED.")

    print("All chronological split checks passed.")


if __name__ == "__main__":
    main()