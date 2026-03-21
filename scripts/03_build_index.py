import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "external" / "sdobenchmark" / "raw" / "SDOBenchmark-data-full"

print("Dataset root exists:", DATA_ROOT.exists())


train_csv = DATA_ROOT / "training" / "meta_data.csv"
test_csv = DATA_ROOT / "test" / "meta_data.csv"

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

print("train rows:", len(train_df))
print("test rows:", len(test_df))
print("train columns:", list(train_df.columns))

M1_THRESHOLD = 1e-5 # GOES M1.0 = 1e-5 W/m^2

train_df["split"] = "training"
test_df["split"] = "test"

df = pd.concat([train_df, test_df], ignore_index=True)
df["label_m1p"] = (df["peak_flux"] >= M1_THRESHOLD).astype(int)

print("total rows:", len(df))
print("M1+ positives:", int(df["label_m1p"].sum()))
print("M1+ rate:", float(df["label_m1p"].mean()))

df["ar"] = df["id"].str.split("_").str[0]
df["sample"] = df["id"].str.split("_", n = 1).str[1]

print(df[["id", "ar", "sample"]].head())

def sample_dir(row):
    return DATA_ROOT / row["split"] / row["ar"] / row["sample"]

df["sample_dir"] = df.apply(sample_dir, axis = 1)

df["num_images"] = df["sample_dir"].apply(lambda p: len(list(p.glob("*.jpg"))))
df["is_complete_40"] = df["num_images"] == 40

print(df["num_images"].value_counts().head(10))
print("complete_40:", int(df["is_complete_40"].sum()), "/", len(df))

OUT_DIR = ROOT / "data" / "interim" / "sdobenchmark"
OUT_DIR.mkdir(parents = True, exist_ok = True)

index_df = df[df["is_complete_40"]].copy()

out_path = OUT_DIR / "index.parquet"
index_df["sample_dir"] = index_df["sample_dir"].astype(str)
index_df.to_parquet(out_path, index = False)

print("Wrote:", out_path)
print("Rows in index:", len(index_df))