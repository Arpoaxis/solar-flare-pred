import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
df = pd.read_parquet(INDEX)

# print("rows:", len(df))
# print("positives:", int(df["label_m1p"].sum()))
# print("positive rate:", float(df["label_m1p"].mean()))
# print("by split:\n", df.groupby("split")["label_m1p"].mean())

df["start_dt"] = pd.to_datetime(df["start"])

summary = df.groupby("split").agg(
    rows = ("id", "count"),
    ars = ("ar", "nunique"),
    positives = ("label_m1p", "sum"),
    pos_rate = ("label_m1p", "mean"),
    start_min = ("start_dt", "min"),
    start_max = ("start_dt", "max"),
)

print(summary)