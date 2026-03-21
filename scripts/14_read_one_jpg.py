from pathlib import Path
import pandas as pd
import torch
from torchvision.io import read_image

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
SPLIT = ROOT / "data" / "interim" / "sdobenchmark" / "splits" / "train.csv"

# 1) grab one id from train
split_df = pd.read_csv(SPLIT)
sample_id = str(split_df.iloc[0]["id"])

# 2) map id -> sample_dir (from your clean index)
idx = pd.read_parquet(INDEX)
sample_dir = Path(idx[idx["id"].astype(str) == sample_id].iloc[0]["sample_dir"])

# 3) pick one jpg inside that folder and load it
jpg_path = sorted(sample_dir.glob("*.jpg"))[0]
img = read_image(str(jpg_path))  # uint8 tensor, shape [C,H,W]

print("id:", sample_id)
print("jpg:", jpg_path.name)
print("shape:", tuple(img.shape), "dtype:", img.dtype)

# 4) convert to 1x256x256 float in [0,1]
if img.shape[0] == 3:            # RGB jpg
    img = img[0:1]               # take 1 channel (they’re usually identical)
img = img.float() / 255.0

print("after:", tuple(img.shape), "dtype:", img.dtype,
      "min/max:", float(img.min()), float(img.max()))

assert img.shape == (1, 256, 256), f"unexpected shape {tuple(img.shape)}"