from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

CHANNELS = ["94", "131", "171", "193", "211", "304", "335", "1700", "continuum", "magnetogram"]

class SDOBenchmarkDataset(Dataset):
    def __init__(self, split_csv: Path, index_parquet: Path ):
        self.df = pd.read_csv(split_csv)

        idx = pd.read_parquet(index_parquet)[["id", "sample_dir"]].copy()

        # map: id (string) -> sample_dir (string)
        idx["id"] = idx["id"].astype(str)
        self.id_to_dir = dict(zip(idx["id"].values, idx["sample_dir"].values))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        sample_id = str(row["id"])
        y = int(row["label_m1p"])

        sample_dir = Path(self.id_to_dir[sample_id])

        files = sorted(sample_dir.glob("*.jpg"))
        timestamps = sorted({p.stem.split("__")[0] for p in files})
        assert len(timestamps) == 4, f"{sample_id}: expected 4 timestamps, got {len(timestamps)}"

        frames = []
        for ts in timestamps:
            chans = []
            for ch in CHANNELS:
                p = sample_dir / f"{ts}__{ch}.jpg"
                img = read_image(str(p)).float() / 255.0  # [1,256,256]
                chans.append(img)
            frames.append(torch.cat(chans, dim=0))  # [10,256,256]

        x = torch.stack(frames, dim=0)  # [4,10,256,256]

        assert x.shape == (4, 10, 256, 256), x.shape
        return x, y