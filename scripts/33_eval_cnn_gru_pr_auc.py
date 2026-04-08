"""
Evaluate the saved CNN-GRU checkpoint on validation and test splits.

This script loads a saved CNN-GRU checkpoint, rebuilds the model
architecture, evaluates it on the validation and test splits, and reports
average BCE-with-logits loss, Average Precision (PR-AUC), and Brier score.

The CNN-GRU model consumes the full sample tensor:
    [B, 4, 10, 256, 256]

Inputs:
    - data/interim/sdobenchmark/index.parquet
    - data/interim/sdobenchmark/splits/val.csv
    - data/interim/sdobenchmark/splits/test.csv
    - runs/checkpoints/<checkpoint>.pt

Output:
    - Printed device selection
    - Printed checkpoint existence check
    - Printed loaded checkpoint epoch
    - Printed split-level loss and metrics
"""

from pathlib import Path

import numpy as np
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

# Curated index, split directory, and checkpoint location.
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
SPLITS = ROOT / "data" / "interim" / "sdobenchmark" / "splits"

# Choose the checkpoint to evaluate.
# Example alternative:
# CKPT = ROOT / "runs" / "checkpoints" / "cnn_gru_best.pt"
CKPT = ROOT / "runs" / "checkpoints" / "cnn_gru_posw_best.pt"

# Evaluate both validation and test splits for comparison.
SPLIT_NAMES = ["val", "test"]

# Evaluation loader settings.
BATCH_SIZE = 8
SHUFFLE = False
NUM_WORKERS = 0
PIN_MEMORY = True

# CNN-GRU model configuration.
IN_CHANNELS = 10
FEATURE_DIM = 128
HIDDEN_DIM = 128


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Average Precision (PR-AUC) using a ranking-based formulation.

    The implementation sorts samples by descending predicted score and
    averages precision at the ranks where positive examples appear.

    Args:
        y_true (np.ndarray): Binary ground-truth labels.
        y_score (np.ndarray): Predicted probabilities or scores.

    Returns:
        float: Average Precision value.
    """
    y_true = y_true.astype(np.int32)

    # Rank samples from highest predicted score to lowest.
    order = np.argsort(-y_score)
    y_true = y_true[order]

    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0

    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / (tp + fp)

    return float(precision[y_true == 1].sum() / n_pos)


def build_dataloader(split_name: str) -> DataLoader:
    """
    Create a DataLoader for one saved dataset split.

    Args:
        split_name (str): Split name, such as 'val' or 'test'.

    Returns:
        DataLoader: DataLoader wrapping the requested split dataset.
    """
    dataset = SDOBenchmarkDataset(SPLITS / f"{split_name}.csv", INDEX)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def build_model(device: torch.device) -> CNNGRU:
    """
    Build the CNN-GRU model for evaluation.

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


def load_checkpoint(
    model: nn.Module,
    ckpt_path: Path,
    device: torch.device,
) -> dict:
    """
    Load model weights from a saved checkpoint.

    Args:
        model (nn.Module): Model instance to populate.
        ckpt_path (Path): Path to the saved checkpoint file.
        device (torch.device): Device used for checkpoint loading.

    Returns:
        dict: Loaded checkpoint dictionary.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate one split and compute basic classification metrics.

    The CNN-GRU model consumes the full input tensor:
        [B, 4, 10, 256, 256]

    Metrics reported:
        - average BCE-with-logits loss
        - Average Precision (used here as PR-AUC)
        - Brier score
        - class balance summary

    Args:
        model (nn.Module): Trained model in evaluation mode.
        dataloader (DataLoader): Split DataLoader.
        device (torch.device): Execution device.

    Returns:
        dict: Evaluation summary for the split.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_probs = []
    all_targets = []

    total_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)      # [B, 4, 10, 256, 256]
        y = y.float().to(device).view(-1, 1)     # [B, 1]

        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        y_cpu = y.cpu().numpy().reshape(-1)

        all_probs.append(probs)
        all_targets.append(y_cpu)

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    y_true = np.concatenate(all_targets)
    y_score = np.concatenate(all_probs)

    avg_loss = total_loss / max(n_batches, 1)
    ap = average_precision(y_true, y_score)
    brier = float(np.mean((y_score - y_true) ** 2))

    pos = int(y_true.sum())
    n = int(len(y_true))
    pos_rate = pos / n if n > 0 else 0.0

    return {
        "n": n,
        "pos": pos,
        "pos_rate": pos_rate,
        "bce_loss": avg_loss,
        "pr_auc_ap": ap,
        "brier": brier,
    }


def print_split_metrics(split_name: str, stats: dict) -> None:
    """
    Print a compact one-line summary of evaluation metrics for a split.

    Args:
        split_name (str): Split name being reported.
        stats (dict): Metrics dictionary returned by evaluate_split().
    """
    print(
        f"{split_name}: "
        f"n={stats['n']} pos={stats['pos']} rate={stats['pos_rate']:.4f} | "
        f"loss={stats['bce_loss']:.4f} | "
        f"PR-AUC(AP)={stats['pr_auc_ap']:.4f} | "
        f"Brier={stats['brier']:.4f}"
    )


def main() -> None:
    """
    Load the saved checkpoint and evaluate it on validation and test.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("ckpt exists:", CKPT.exists(), "|", CKPT)

    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    model = build_model(device)
    checkpoint = load_checkpoint(model, CKPT, device)
    print("loaded epoch:", checkpoint.get("epoch"))

    for split_name in SPLIT_NAMES:
        dataloader = build_dataloader(split_name)
        stats = evaluate_split(model, dataloader, device)
        print_split_metrics(split_name, stats)


if __name__ == "__main__":
    main()