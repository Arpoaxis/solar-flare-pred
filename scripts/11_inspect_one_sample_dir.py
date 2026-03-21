from pathlib import Path
from collections import Counter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
SPLIT = ROOT / "data" / "interim" / "sdobenchmark" / "splits" / "train.csv"

split_df = pd.read_csv(SPLIT)
sample_id = str(split_df.iloc[0]["id"])

idx = pd.read_parquet(INDEX)
row = idx[idx["id"].astype(str) == sample_id].iloc[0]
sample_dir = Path(row["sample_dir"])

files = sorted(sample_dir.glob("*.jpg"))
print("id:", sample_id)
print("sample_dir:", sample_dir)
print("num_files:", len(files))

pairs = []
bad = []
for p in files:
    parts = p.stem.split("__")
    if len(parts) != 2:
        bad.append(p.name)
        continue
    ts, ch = parts[0], parts[1]   # ch can be '131' or 'continuum'
    pairs.append((ts, ch))

ts_set = sorted({t for t, _ in pairs})
ch_set = sorted({c for _, c in pairs})
print("unique timestamps:", len(ts_set), ts_set)
print("unique channels:", len(ch_set), ch_set)

counts = Counter(t for t, _ in pairs)
print("files per timestamp:", dict(counts))

if bad:
    print("unexpected filenames:", bad[:10])