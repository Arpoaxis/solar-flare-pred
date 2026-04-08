"""
Run a quick forward-pass sanity check for the CNN-GRU model.

This script loads one batch from the training split, prints the input and
label shapes, instantiates the CNN-GRU model, performs a forward pass on the
selected device, and prints the output logit shape and dtype.

It is intended as a lightweight integration check between:
    - the saved training split
    - the curated sample index
    - the custom dataset class
    - the CNN-GRU model
    - device transfer and forward execution

Inputs:
    - data/interim/sdobenchmark/index.parquet
    - data/interim/sdobenchmark/splits/train.csv

Output:
    - Printed device selection
    - Printed batch input shape and labels
    - Printed output logit shape and dtype
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.sdobenchmark_dataset import SDOBenchmarkDataset
from src.models.cnn_gru import CNNGRU


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
# Resolve the project root relative to this script so it works
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[1]

# Curated index and split directory.
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
SPLITS = ROOT / "data" / "interim" / "sdobenchmark" / "splits"
TRAIN_CSV = SPLITS / "train.csv"

# DataLoader settings for a lightweight forward-pass check.
BATCH_SIZE = 2
SHUFFLE = True
NUM_WORKERS = 0
PIN_MEMORY = True

# CNN-GRU model configuration.
IN_CHANNELS = 10
FEATURE_DIM = 128
HIDDEN_DIM = 128


def build_dataloader() -> DataLoader:
    """
    Create a DataLoader for one-batch CNN-GRU sanity checking.

    Returns:
        DataLoader: DataLoader wrapping the training split dataset.
    """
    dataset = SDOBenchmarkDataset(TRAIN_CSV, INDEX)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def build_model(device: torch.device) -> CNNGRU:
    """
    Build the CNN-GRU model for a forward-pass check.

    Args:
        device (torch.device): Target device for model execution.

    Returns:
        CNNGRU: Model moved to the requested device.
    """
    model = CNNGRU(
        in_channels=IN_CHANNELS,
        feat_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
    )
    return model.to(device)


def main() -> None:
    """
    Load one batch and run a forward pass through the CNN-GRU model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Build the DataLoader and retrieve one training batch.
    dataloader = build_dataloader()
    x, y = next(iter(dataloader))

    print("x:", tuple(x.shape), x.dtype, "| y:", y.tolist())

    # Build the model and run a forward pass on the selected device.
    model = build_model(device)
    x = x.to(device, non_blocking=True)

    logits = model(x)

    print("logits:", tuple(logits.shape), logits.dtype)


if __name__ == "__main__":
    main()