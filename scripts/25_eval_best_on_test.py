"""
Evaluate the best saved CNN baseline checkpoint on the test split.

This script loads the saved best model checkpoint, rebuilds the CNN
baseline architecture, evaluates it on the held-out test split, and
reports the average BCE-with-logits loss.

The CNN baseline uses only the last timestep from each sample:
    [B, 4, 10, 256, 256] -> [B, 10, 256, 256]

Inputs:
    - data/interim/sdobenchmark/index.parquet
    - data/interim/sdobenchmark/splits/test.csv
    - runs/checkpoints/cnn_baseline_best.pt

Output:
    - Printed device selection
    - Printed checkpoint path and existence check
    - Printed loaded checkpoint epoch
    - Printed average test loss
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.sdobenchmark_dataset import SDOBenchmarkDataset
from src.models.cnn_baseline import CNNBaseline


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
# Resolve the project root relative to this script so it works
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[1]

# Curated index, split directory, and saved checkpoint location.
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
SPLITS = ROOT / "data" / "interim" / "sdobenchmark" / "splits"
TEST_CSV = SPLITS / "test.csv"

CKPT = ROOT / "runs" / "checkpoints" / "cnn_baseline_best.pt"

# Evaluation loader settings.
BATCH_SIZE = 8
SHUFFLE = False
NUM_WORKERS = 0
PIN_MEMORY = True


def build_test_dataloader() -> DataLoader:
    """
    Create the test DataLoader for checkpoint evaluation.

    Returns:
        DataLoader: DataLoader wrapping the test split dataset.
    """
    dataset = SDOBenchmarkDataset(TEST_CSV, INDEX)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def build_model(device: torch.device) -> nn.Module:
    """
    Build the CNN baseline model for evaluation.

    Args:
        device (torch.device): Target device for model execution.

    Returns:
        nn.Module: CNN baseline model moved to the requested device.
    """
    model = CNNBaseline(in_channels=10)
    return model.to(device)


@torch.no_grad()
def evaluate_loss(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """
    Evaluate average BCE-with-logits loss over the test DataLoader.

    The CNN baseline uses only the most recent timestep from each sample.

    Args:
        model (nn.Module): Trained model in evaluation mode.
        dataloader (DataLoader): Test DataLoader.
        device (torch.device): Execution device.

    Returns:
        float: Average loss across processed batches.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    batch_count = 0

    for x, y in dataloader:
        # The CNN baseline consumes only the last frame:
        # [B, 4, 10, 256, 256] -> [B, 10, 256, 256]
        x = x[:, -1].to(device, non_blocking=True)
        y = y.float().to(device).view(-1, 1)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.detach().cpu())
        batch_count += 1

    return total_loss / max(batch_count, 1)


def load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> dict:
    """
    Load model weights from a saved checkpoint.

    Args:
        model (nn.Module): Model instance to populate.
        ckpt_path (Path): Path to the saved checkpoint.
        device (torch.device): Device used for checkpoint loading.

    Returns:
        dict: Loaded checkpoint dictionary.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def main() -> None:
    """
    Load the best checkpoint and evaluate it on the test split.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("ckpt:", CKPT, "| exists:", CKPT.exists())

    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    test_loader = build_test_dataloader()
    model = build_model(device)

    checkpoint = load_checkpoint(model, CKPT, device)
    print("loaded epoch:", checkpoint.get("epoch"))

    test_loss = evaluate_loss(model, test_loader, device)
    print("test_loss:", test_loss)


if __name__ == "__main__":
    main()