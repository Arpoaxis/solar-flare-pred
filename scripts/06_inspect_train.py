from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # scripts/ -> project root
csv_path = ROOT / "data" / "interim" / "sdobenchmark" / "splits" / "train.csv"

print("exists:", csv_path.exists(), "| path:", csv_path)

df = pd.read_csv(csv_path)

print("rows:", len(df))
print("num_cols:", len(df.columns))
print("columns:", df.columns.tolist())

label_cols = [c for c in df.columns if c.lower() in ("y","label","target","label_m1p","label_m1","flare")]
print("label candidates:", label_cols)

pathish = [c for c in df.columns if any(k in c.lower() for k in
                                        ("path","file","img","image","png","npy","npz","t0","t1","t2","t3","c0","c1","channel"))]
print("path-like cols:", len(pathish))
print("path-like cols (first 20):", pathish[:20])

print("\nfirst row (subset):")
row = df.iloc[0].to_dict()
for k in (label_cols + pathish)[:25]:
    print(f"{k} = {row.get(k)}")