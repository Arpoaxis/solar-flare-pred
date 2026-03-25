"""
Train the CNN baseline on the AR-disjoint train/validation split.

This script trains the CNN-only baseline using the saved train and
validation CSV splits, logs epoch-level loss values to CSV, and saves
both the latest checkpoint and the best checkpoint based on validation loss.

The model uses only the last timestep from each sample:
    [B, 4, 10, 256, 256] -> [B, 10, 256, 256]

Inputs:
    - data/interim/sdobenchmark/index.parquet
    - data/interim/sdobenchmark/splits/train.csv
    - data/interim/sdobenchmark/splits/val.csv

Outputs:
    - runs/checkpoints/cnn_baseline_latest.pt
    - runs/checkpoints/cnn_baseline_best.pt
    - runs/logs/cnn_baseline_log.csv
"""

from pathlib import Path
import csv

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
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
SPLITS = ROOT / "data" / "interim" / "sdobenchmark" / "splits"

TRAIN_CSV = SPLITS / "train.csv"
VAL_CSV = SPLITS / "val.csv"

CKPT_DIR = ROOT / "runs" / "checkpoints"
LOG_PATH = ROOT / "runs" / "logs" / "cnn_baseline_log.csv"

# Training configuration.
EPOCHS = 3
BATCH_SIZE = 4
NUM_WORKERS = 0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2

# Debug / monitoring controls.
PRINT_FIRST_BATCH_SHAPES = True
PRINT_POS_RATE = True
PRINT_VAL_CONFUSION_AT_05 = True

# Threshold used for simple confusion-count reporting.
DEFAULT_THRESHOLD = 0.5


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    """
    Save a model checkpoint to disk.

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


def build_dataloader(csv_path: Path, shuffle: bool) -> DataLoader:
    """
    Build a DataLoader for one dataset split.

    Args:
        csv_path (Path): Path to the split CSV file.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: DataLoader for the requested split.
    """
    dataset = SDOBenchmarkDataset(csv_path, INDEX)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


@torch.no_grad()
def binary_stats_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[int, int, int, int]:
    """
    Compute basic binary confusion counts from logits.

    Args:
        logits (torch.Tensor): Model logits with shape [B, 1].
        targets (torch.Tensor): Binary targets with shape [B, 1].
        threshold (float): Probability threshold used for classification.

    Returns:
        tuple[int, int, int, int]:
            - True positives
            - False positives
            - True negatives
            - False negatives
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(torch.int32)
    targets_int = targets.to(torch.int32)

    tp = int(((preds == 1) & (targets_int == 1)).sum().item())
    fp = int(((preds == 1) & (targets_int == 0)).sum().item())
    tn = int(((preds == 0) & (targets_int == 0)).sum().item())
    fn = int(((preds == 0) & (targets_int == 1)).sum().item())

    return tp, fp, tn, fn


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> float:
    """
    Run one full training or validation epoch.

    When an optimizer is provided, the model runs in training mode and
    parameters are updated. Otherwise, the epoch runs in evaluation mode.

    Args:
        model (nn.Module): Model to train or evaluate.
        dataloader (DataLoader): DataLoader for the current split.
        device (torch.device): Execution device.
        optimizer (torch.optim.Optimizer | None): Optimizer for training mode.
        max_batches (int | None): Optional cap on batches processed.

    Returns:
        float: Average BCE-with-logits loss over processed batches.
    """
    train_mode = optimizer is not None
    model.train(train_mode)

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    batch_count = 0

    # Track observed class balance over processed batches.
    positives = 0
    total_labels = 0

    # Track simple confusion counts for thresholded validation output.
    tp = fp = tn = fn = 0

    for step, (x, y) in enumerate(dataloader):
        if (max_batches is not None) and (step >= max_batches):
            break

        # The CNN baseline uses only the most recent frame.
        # Input shape: [B, 4, 10, 256, 256] -> [B, 10, 256, 256]
        x = x[:, -1].to(device, non_blocking=True)
        y = y.float().to(device).view(-1, 1)

        positives += int(y.sum().item())
        total_labels += y.numel()

        logits = model(x)
        loss = criterion(logits, y)

        # Compute simple threshold-based stats for inspection.
        a, b, c, d = binary_stats_from_logits(
            logits.detach(),
            y.detach(),
            threshold=DEFAULT_THRESHOLD,
        )
        tp += a
        fp += b
        tn += c
        fn += d

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        batch_count += 1

        if PRINT_FIRST_BATCH_SHAPES and step == 0:
            print("batch x:", tuple(x.shape), x.dtype, "| logits:", tuple(logits.shape))

    avg_loss = total_loss / max(batch_count, 1)

    if PRINT_POS_RATE and total_labels > 0:
        tag = "train" if train_mode else "val"
        print(f"{tag} pos rate over seen batches: {positives}/{total_labels} = {positives / total_labels:.4f}")

    if (not train_mode) and PRINT_VAL_CONFUSION_AT_05:
        print(f"val @0.5: TP={tp} FP={fp} TN={tn} FN={fn}")

    return avg_loss


def main() -> None:
    """
    Train the CNN baseline and save logs and checkpoints.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader = build_dataloader(TRAIN_CSV, shuffle=True)
    val_loader = build_dataloader(VAL_CSV, shuffle=False)

    model = CNNBaseline(in_channels=10).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    # Open the log file once and append one row per epoch.
    with open(LOG_PATH, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "val_loss"])

        for epoch in range(1, EPOCHS + 1):
            train_loss = run_one_epoch(
                model,
                train_loader,
                device,
                optimizer=optimizer,
                max_batches=None,
            )
            val_loss = run_one_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
                max_batches=None,
            )

            print(f"epoch {epoch} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

            writer.writerow([epoch, train_loss, val_loss])
            log_file.flush()

            # Always save the most recent model state.
            save_checkpoint(CKPT_DIR / "cnn_baseline_latest.pt", model, optimizer, epoch)

            # Update the best checkpoint when validation loss improves.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(CKPT_DIR / "cnn_baseline_best.pt", model, optimizer, epoch)


if __name__ == "__main__":
    main()