"""
Run a one-batch training-step sanity check for the SDOBenchmark pipeline.

This script builds the custom training dataset and DataLoader, creates a
minimal baseline model, runs one forward/backward optimization step, and
prints the resulting tensor shapes and loss.

It is intended as an end-to-end smoke test for:
    - dataset loading
    - batch collation
    - device transfer
    - model forward pass
    - loss computation
    - backward pass
    - optimizer step

Inputs:
    - data/interim/sdobenchmark/index.parquet
    - data/interim/sdobenchmark/splits/train.csv

Output:
    - Printed device selection
    - Printed batch tensor shapes
    - Printed one training loss value
    - Printed confirmation that one optimizer step completed
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.sdobenchmark_dataset import SDOBenchmarkDataset


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
# Resolve the project root relative to this script so it works
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[1]

# Curated index and training split used to build the dataset.
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
TRAIN = ROOT / "data" / "interim" / "sdobenchmark" / "splits" / "train.csv"

# Small batch size keeps this smoke test lightweight.
BATCH_SIZE = 2
SHUFFLE = True
NUM_WORKERS = 0
PIN_MEMORY = True

# Expected sample tensor shape:
# [T, C, H, W] = [4, 10, 256, 256]
INPUT_DIM = 4 * 10 * 256 * 256

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2


def build_dataloader() -> DataLoader:
    """
    Create a training DataLoader for a one-batch sanity check.

    Returns:
        DataLoader: DataLoader wrapping the SDOBenchmark training dataset.
    """
    dataset = SDOBenchmarkDataset(TRAIN, INDEX)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def build_model(device: torch.device) -> nn.Module:
    """
    Build a minimal baseline model for one-step training verification.

    The model simply flattens the full sample tensor and applies a single
    linear layer to produce one logit for binary classification.

    Args:
        device (torch.device): Target device for model execution.

    Returns:
        nn.Module: Model moved to the requested device.
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(INPUT_DIM, 1),
    )
    return model.to(device)


def main() -> None:
    """
    Run one end-to-end training step and verify that all components work.
    """
    # Use CUDA when available so this check matches the intended training setup.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Build the training loader and minimal test model.
    dataloader = build_dataloader()
    model = build_model(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Pull one batch to verify dataset loading, collation, and device transfer.
    x, y = next(iter(dataloader))
    x = x.to(device, non_blocking=True)
    y = y.float().to(device).view(-1, 1)

    # Run one forward/backward pass and update the model once.
    logits = model(x)
    loss = criterion(logits, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print("x shape:", tuple(x.size()))
    print("y shape:", tuple(y.size()))
    print("loss:", float(loss.detach().cpu()))
    print("step ok")


if __name__ == "__main__":
    main()