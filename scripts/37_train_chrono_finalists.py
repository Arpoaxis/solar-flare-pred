"""
Train finalist flare forecasting models on the chronological splits.

This script trains a configured set of finalist models using the chronological
training and validation splits. It supports both single-timestep models and
full-sequence models, optional positive-class weighting, and optional early
stopping based on validation loss.

For each enabled model, the script:
    - builds the model
    - trains on the chronological training split
    - evaluates on the chronological validation split
    - saves latest and best checkpoints
    - appends epoch-level losses to a shared CSV log

Model input modes:
    - "last": uses only the final timestep
        [B, 4, 10, 256, 256] -> [B, 10, 256, 256]
    - "full": uses the full spatiotemporal tensor
        [B, 4, 10, 256, 256]

Inputs:
    - data/interim/sdobenchmark/index.parquet
    - data/interim/sdobenchmark/splits/chrono_train.csv or chrono-train.csv
    - data/interim/sdobenchmark/splits/chrono_val.csv or chrono-val.csv

Outputs:
    - runs/checkpoints/chrono_<model_name>_latest.pt
    - runs/checkpoints/chrono_<model_name>_best.pt
    - runs/logs/chrono_training_log.csv
"""

from pathlib import Path
import csv
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.sdobenchmark_dataset import SDOBenchmarkDataset
from src.models.cnn_baseline import CNNBaseline
from src.models.cnn_timepool import CNNTimeMeanPool
from src.models.cnn_gru import CNNGRU


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
# Resolve the project root relative to this script so it works
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[1]

INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
SPLITS = ROOT / "data" / "interim" / "sdobenchmark" / "splits"
CHECKPOINT_DIR = ROOT / "runs" / "checkpoints"
LOG_PATH = ROOT / "runs" / "logs" / "chrono_training_log.csv"

# Training configuration shared by all finalist runs.
BATCH_SIZE = 4
NUM_WORKERS = 0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2

# Set to one model name, such as "cnn_timepool_es", to run a single model.
ONLY: str | None = None

# Finalist model specifications.
# "mode" controls how the model consumes x:
#   - "last": use only x[:, -1] -> [B, 10, 256, 256]
#   - "full": use all timesteps -> [B, 4, 10, 256, 256]
FINALISTS: list[dict[str, Any]] = [
    {
        "name": "cnn_baseline",
        "mode": "last",
        "posw": False,
        "max_epochs": 3,
        "patience": None,
        "builder": lambda device: CNNBaseline(in_channels=10).to(device),
    },
    {
        "name": "cnn_baseline_posw",
        "mode": "last",
        "posw": True,
        "max_epochs": 3,
        "patience": None,
        "builder": lambda device: CNNBaseline(in_channels=10).to(device),
    },
    {
        "name": "cnn_timepool",
        "mode": "full",
        "posw": False,
        "max_epochs": 3,
        "patience": None,
        "builder": lambda device: CNNTimeMeanPool(in_channels=10).to(device),
    },
    {
        "name": "cnn_timepool_es",
        "mode": "full",
        "posw": False,
        "max_epochs": 10,
        "patience": 2,
        "builder": lambda device: CNNTimeMeanPool(in_channels=10).to(device),
    },
    {
        "name": "cnn_gru",
        "mode": "full",
        "posw": False,
        "max_epochs": 3,
        "patience": None,
        "builder": lambda device: CNNGRU(
            in_channels=10,
            feat_dim=128,
            hidden_dim=128,
        ).to(device),
    },
    {
        "name": "cnn_gru_es",
        "mode": "full",
        "posw": False,
        "max_epochs": 10,
        "patience": 2,
        "builder": lambda device: CNNGRU(
            in_channels=10,
            feat_dim=128,
            hidden_dim=128,
        ).to(device),
    },
    {
        "name": "cnn_gru_posw",
        "mode": "full",
        "posw": True,
        "max_epochs": 3,
        "patience": None,
        "builder": lambda device: CNNGRU(
            in_channels=10,
            feat_dim=128,
            hidden_dim=128,
        ).to(device),
    },
    {
        "name": "cnn_gru_posw_es",
        "mode": "full",
        "posw": True,
        "max_epochs": 10,
        "patience": 2,
        "builder": lambda device: CNNGRU(
            in_channels=10,
            feat_dim=128,
            hidden_dim=128,
        ).to(device),
    },
]


def find_chrono_csv(splits_dir: Path, split_name: str) -> Path:
    """
    Find a chronological split CSV using either underscore or hyphen naming.

    Supported filename patterns:
        - chrono_train.csv / chrono_val.csv
        - chrono-train.csv / chrono-val.csv

    Args:
        splits_dir (Path): Directory containing split CSV files.
        split_name (str): Split suffix, such as "train" or "val".

    Returns:
        Path: Path to the requested chronological split CSV.
    """
    candidates = [
        splits_dir / f"chrono_{split_name}.csv",
        splits_dir / f"chrono-{split_name}.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find chrono {split_name} split in {splits_dir}. "
        f"Tried: {[p.name for p in candidates]}"
    )


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


def build_dataloader(dataset: SDOBenchmarkDataset, shuffle: bool) -> DataLoader:
    """
    Build a DataLoader for one dataset split.

    Args:
        dataset (SDOBenchmarkDataset): Dataset instance to wrap.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: DataLoader for the requested split.
    """
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


def compute_pos_weight(
    train_dataset: SDOBenchmarkDataset,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the positive-class weight from the chronological training split.

    The weight is defined as:
        pos_weight = number_of_negatives / number_of_positives

    Args:
        train_dataset (SDOBenchmarkDataset): Chronological training dataset.
        device (torch.device): Target device for the resulting tensor.

    Returns:
        torch.Tensor: Single-value tensor used by BCEWithLogitsLoss.
    """
    positives = int(train_dataset.df["label_m1p"].sum())
    negatives = int(len(train_dataset.df) - positives)

    weight_value = negatives / max(positives, 1)
    print(f"chrono pos_weight = {negatives}/{positives} = {weight_value:.4f}")

    return torch.tensor([weight_value], device=device)


def forward_logits(
    spec: dict[str, Any],
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Run the model forward according to its input mode.

    Args:
        spec (dict[str, Any]): Model specification dictionary.
        model (nn.Module): Model to run.
        x (torch.Tensor): Input tensor from the dataset.
        device (torch.device): Execution device.

    Returns:
        torch.Tensor: Output logits with shape [B, 1].
    """
    if spec["mode"] == "last":
        # Last-timestep models use only the most recent frame.
        x_input = x[:, -1].to(device, non_blocking=True)   # [B, 10, 256, 256]
    else:
        # Full-sequence models consume all timesteps directly.
        x_input = x.to(device, non_blocking=True)          # [B, 4, 10, 256, 256]

    return model(x_input)


def run_one_epoch(
    spec: dict[str, Any],
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    pos_weight: torch.Tensor | None = None,
) -> float:
    """
    Run one training or validation epoch and return average loss.

    When an optimizer is provided, the model runs in training mode and
    parameters are updated. Otherwise, the epoch runs in evaluation mode.

    Args:
        spec (dict[str, Any]): Model specification dictionary.
        model (nn.Module): Model to train or evaluate.
        dataloader (DataLoader): DataLoader for the current split.
        device (torch.device): Execution device.
        optimizer (torch.optim.Optimizer | None): Optimizer for training mode.
        pos_weight (torch.Tensor | None): Positive-class weight for BCE loss.

    Returns:
        float: Average BCE-with-logits loss over processed batches.
    """
    train_mode = optimizer is not None
    model.train(train_mode)

    criterion = (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if pos_weight is not None
        else nn.BCEWithLogitsLoss()
    )

    total_loss = 0.0
    batch_count = 0

    for step, (x, y) in enumerate(dataloader):
        y = y.float().to(device).view(-1, 1)

        logits = forward_logits(spec, model, x, device)
        loss = criterion(logits, y)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        batch_count += 1

        if step == 0:
            # Print one batch shape per epoch per split for sanity checking.
            print("batch logits:", tuple(logits.shape), logits.dtype)

    return total_loss / max(batch_count, 1)


def train_one_model(
    spec: dict[str, Any],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
    writer: csv.writer,
    log_file,
) -> None:
    """
    Train one finalist model on the chronological splits.

    Args:
        spec (dict[str, Any]): Model specification dictionary.
        train_dataloader (DataLoader): Chronological training DataLoader.
        val_dataloader (DataLoader): Chronological validation DataLoader.
        device (torch.device): Execution device.
        pos_weight (torch.Tensor): Positive-class weight from chrono-train.
        writer (csv.writer): CSV writer for logging epoch results.
        log_file: Open file handle used for immediate flushes.
    """
    run_name = f"chrono_{spec['name']}"
    best_checkpoint = CHECKPOINT_DIR / f"{run_name}_best.pt"
    latest_checkpoint = CHECKPOINT_DIR / f"{run_name}_latest.pt"

    print(f"\n=== TRAIN: {run_name} ===")

    model = spec["builder"](device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    bad_epochs = 0

    for epoch in range(1, spec["max_epochs"] + 1):
        weight_tensor = pos_weight if spec["posw"] else None

        train_loss = run_one_epoch(
            spec=spec,
            model=model,
            dataloader=train_dataloader,
            device=device,
            optimizer=optimizer,
            pos_weight=weight_tensor,
        )
        val_loss = run_one_epoch(
            spec=spec,
            model=model,
            dataloader=val_dataloader,
            device=device,
            optimizer=None,
            pos_weight=weight_tensor,
        )

        print(f"epoch {epoch} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

        # Always save the most recent model state.
        save_checkpoint(latest_checkpoint, model, optimizer, epoch)

        is_best = 0
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            save_checkpoint(best_checkpoint, model, optimizer, epoch)
            is_best = 1
        else:
            if spec["patience"] is not None:
                bad_epochs += 1
                if bad_epochs >= spec["patience"]:
                    print(f"early stopping at epoch {epoch} (best_val={best_val_loss:.4f})")
                    writer.writerow(
                        [run_name, epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", is_best]
                    )
                    log_file.flush()
                    break

        writer.writerow([run_name, epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", is_best])
        log_file.flush()


def main() -> None:
    """
    Train all enabled finalist models on the chronological splits.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    chrono_train_csv = find_chrono_csv(SPLITS, "train")
    chrono_val_csv = find_chrono_csv(SPLITS, "val")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("chrono_train:", chrono_train_csv)
    print("chrono_val:", chrono_val_csv)

    train_dataset = SDOBenchmarkDataset(chrono_train_csv, INDEX)
    val_dataset = SDOBenchmarkDataset(chrono_val_csv, INDEX)

    pos_weight = compute_pos_weight(train_dataset, device)

    train_dataloader = build_dataloader(train_dataset, shuffle=True)
    val_dataloader = build_dataloader(val_dataset, shuffle=False)

    with open(LOG_PATH, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["model", "epoch", "train_loss", "val_loss", "best"])

        for spec in FINALISTS:
            if ONLY is not None and spec["name"] != ONLY:
                continue

            train_one_model(
                spec=spec,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                device=device,
                pos_weight=pos_weight,
                writer=writer,
                log_file=log_file,
            )

    print("\nsaved training log:", LOG_PATH)


if __name__ == "__main__":
    main()