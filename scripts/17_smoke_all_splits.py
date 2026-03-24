from pathlib import Path
from torch.utils.data import DataLoader
from src.data.sdobenchmark_dataset import SDOBenchmarkDataset

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
SPLITS = ROOT / "data" / "interim" / "sdobenchmark" / "splits"

for name in ["train", "val", "test"]:
    csv_path = SPLITS / f"{name}.csv"
    if __name__ == "__main__":
        ds = SDOBenchmarkDataset(csv_path, INDEX)
        dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        x, y = next(iter(dl))
        print(name, "len:", len(ds), "batch x:", tuple(x.shape), "batch y:", y.tolist())