from pathlib import Path
from torch.utils.data import DataLoader
from src.data.sdobenchmark_dataset import SDOBenchmarkDataset

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
TRAIN = ROOT / "data" / "interim" / "sdobenchmark" / "splits" / "train.csv"

ds = SDOBenchmarkDataset(TRAIN, INDEX)
dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)

if __name__ == "__main__":
    x, y = next(iter(dl))
    print("batch x:", tuple(x.shape), x.dtype)
    print("batch y:", y)