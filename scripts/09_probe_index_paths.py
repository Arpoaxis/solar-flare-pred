from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# try the most likely index locations
candidates = [
    ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet",
    ROOT / "data" / "interim" / "sdobenchmark" / "index" / "index.parquet",
]
index_path = next((p for p in candidates if p.exists()), None)
print("index_path:", index_path)

df = pd.read_parquet(index_path)

sample_id = "11390_2012_01_05_17_19_01_1"
row = df[df["id"].astype(str) == sample_id].head(1)

print("found rows:", len(row))
print("columns:", df.columns.tolist())

pathish = [c for c in df.columns if any(k in c.lower() for k in ("path","file","img","image","png","npy","npz"))]
print("path-like columns:", len(pathish))
print("path-like columns (first 30):", pathish[:30])

if len(row) == 1 and pathish:
    print("\nSample row path fields (first 20):")
    r = row.iloc[0].to_dict()
    for k in pathish[:20]:
        print(k, "=", r.get(k))