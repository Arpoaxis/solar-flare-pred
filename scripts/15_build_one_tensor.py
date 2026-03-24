from pathlib import Path
import pandas as pd
import torch
from torchvision.io import read_image

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
SPLIT = ROOT / "data" / "interim" / "sdobenchmark" / "splits" / "train.csv"

CHANNELS = ["94", "131", "171", "193", "211", "304", "335", "1700", "continuum", "magnetogram"]

split_df = pd.read_csv(SPLIT)
sample_id = str(split_df.iloc[0]["id"])
y = int(split_df.iloc[0]["label_m1p"])

idx = pd.read_parquet(INDEX)
sample_dir = Path(idx[idx["id"].astype(str) == sample_id].iloc[0]["sample_dir"])

# collect and sort timestamps (lexical sort works with YYYY-MM-DDTHHMMSS format)
files = sorted(sample_dir.glob("*.jpg"))
timestamps = sorted({p.stem.split("__")[0] for p in files})

assert len(timestamps) == 4, f"expected 4 timestamps, got {len(timestamps)}"
assert len(CHANNELS) == 10, f"expected 10 channels, got {len(CHANNELS)}"

# load into [T,C,256,256]
frames = []
missing = []
for ts in timestamps:
    chans = []
    for ch in CHANNELS:
        p = sample_dir / f"{ts}__{ch}.jpg"
        if not p.exists():
            missing.append(p.name)
            continue
        img = read_image(str(p))          # [1,256,256] uint8
        img = img.float() / 255.0         # [1,256,256] float32
        chans.append(img)
    if len(chans) != 10:
        raise FileNotFoundError(f"missing {10-len(chans)} files at timestamp {ts}: {missing[:10]}")
    frame = torch.cat(chans, dim=0)       # [10,256,256]
    frames.append(frame)

missing = []
for ts in timestamps:
    for ch in CHANNELS:
        p = sample_dir / f"{ts}__{ch}.jpg"
        if not p.exists():
            missing.append(p.name)

print("missing count:", len(missing))
if missing:
    print("missing examples:", missing[:10])

assert len(missing) == 0

x = torch.stack(frames, dim=0)            # [4,10,256,256]

means = [float(x[t].mean()) for t in range(x.shape[0])]
print("per-timestep mean:", means)

print("id:", sample_id)
print("y:", y)
print("x shape:", tuple(x.shape), "dtype:", x.dtype)
print("x min/max:", float(x.min()), float(x.max()))