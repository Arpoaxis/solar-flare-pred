"""
Train the CNN-GRU flare forecasting model with early stopping.

This script trains the CNN-GRU model on the AR-disjoint training split,
evaluates validation loss after each epoch, logs epoch losses to CSV, and
saves both latest and best checkpoints.

Early stopping is applied based on validation loss.

Inputs:
    - data/interim/sdobenchmark/index.parquet
    - data/interim/sdobenchmark/splits/train.csv
    - data/interim/sdobenchmark/splits/val.csv

Outputs:
    - runs/checkpoints/<RUN_NAME>_latest.pt
    - runs/checkpoints/<RUN_NAME>_best.pt
    - runs/logs/<RUN_NAME>_log.csv
"""

from pathlib import Path
import csv

import torch
import torch.nn as nn
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
VAL_CSV = SPLITS / "val.csv"

# Training configuration.
EPOCHS = 10
PATIENCE = 2
MIN_DELTA = 0.0

BATCH_SIZE = 4
NUM_WORKERS = 0

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2

# CNN-GRU model configuration.
IN_CHANNELS = 10
FEATURE_DIM = 128
HIDDEN_DIM = 128

# Output naming.
# Example alternative:
# RUN_NAME = "cnn_gru"
RUN_NAME = "cnn_gru_es"


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    """
    Save a training checkpoint to disk.

    Args:
        path (Path): Output checkpoint path.
        model (nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer state to save.
        epoch (int): Epoch number associated with the checkpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
        },
        path,
    )


def build_dataloader(split_csv: Path, shuffle: bool) -> DataLoader:
    """
    Build a DataLoader for one dataset split.

    Args:
        split_csv (Path): Path to the split CSV file.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: DataLoader for the requested split.
    """
    dataset = SDOBenchmarkDataset(split_csv, INDEX)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


def build_model(device: torch.device) -> CNNGRU:
    """
    Build the CNN-GRU model for training.

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


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """
    Run one training or validation epoch and return average loss.

    When an optimizer is provided, the model runs in training mode and
    parameters are updated. Otherwise, the epoch runs in evaluation mode.

    Args:
        model (nn.Module): Model to train or evaluate.
        dataloader (DataLoader): DataLoader for the current split.
        device (torch.device): Execution device.
        optimizer (torch.optim.Optimizer | None): Optimizer for training mode.

    Returns:
        float: Average BCE-with-logits loss over processed batches.
    """
    train_mode = optimizer is not None
    model.train(train_mode)

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    batch_count = 0

    for step, (x, y) in enumerate(dataloader):
        # Full sample shape is [B, 4, 10, 256, 256]. The CNN-GRU model
        # uses all four timesteps and learns temporal structure explicitly.
        x = x.to(device, non_blocking=True)
        y = y.float().to(device).view(-1, 1)

        logits = model(x)
        loss = criterion(logits, y)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        batch_count += 1

        if step == 0:
            print("batch x:", tuple(x.shape), x.dtype, "| logits:", tuple(logits.shape))

    return total_loss / max(batch_count, 1)


def update_early_stopping(
    val_loss: float,
    best_val: float,
    bad_epochs: int,
) -> tuple[bool, float, int, bool]:
    """
    Update early-stopping state based on the current validation loss.

    Args:
        val_loss (float): Current epoch validation loss.
        best_val (float): Best validation loss seen so far.
        bad_epochs (int): Number of consecutive non-improving epochs.

    Returns:
        tuple[bool, float, int, bool]:
            - Whether training should stop
            - Updated best validation loss
            - Updated bad epoch count
            - Whether this epoch produced a new best model
    """
    if val_loss < best_val - MIN_DELTA:
        return False, val_loss, 0, True

    bad_epochs += 1
    stop = bad_epochs >= PATIENCE
    return stop, best_val, bad_epochs, False


def main() -> None:
    """
    Train the CNN-GRU model and apply early stopping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader = build_dataloader(TRAIN_CSV, shuffle=True)
    val_loader = build_dataloader(VAL_CSV, shuffle=False)

    model = build_model(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    checkpoint_dir = ROOT / "runs" / "checkpoints"
    log_path = ROOT / "runs" / "logs" / f"{RUN_NAME}_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    bad_epochs = 0

    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "val_loss"])

        for epoch in range(1, EPOCHS + 1):
            train_loss = run_one_epoch(
                model,
                train_loader,
                device,
                optimizer=optimizer,
            )
            val_loss = run_one_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
            )

            print(f"epoch {epoch} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

            writer.writerow([epoch, train_loss, val_loss])
            log_file.flush()

            # Always save the most recent model state.
            save_checkpoint(
                checkpoint_dir / f"{RUN_NAME}_latest.pt",
                model,
                optimizer,
                epoch,
            )

            stop, best_val, bad_epochs, is_best = update_early_stopping(
                val_loss=val_loss,
                best_val=best_val,
                bad_epochs=bad_epochs,
            )

            if is_best:
                save_checkpoint(
                    checkpoint_dir / f"{RUN_NAME}_best.pt",
                    model,
                    optimizer,
                    epoch,
                )

            if stop:
                print(f"early stopping at epoch {epoch} (best_val={best_val:.4f})")
                break


if __name__ == "__main__":
    main()