"""
Select a TSS-maximizing threshold on validation and apply it to test.

This script loads the best saved checkpoint for the CNN time-mean-pooling
baseline, collects predicted probabilities for the validation and test splits,
selects the decision threshold that maximizes True Skill Statistic (TSS) on
the validation set, and then applies that same threshold to the test set.

The time-pooled model consumes the full sample tensor:
    [B, 4, 10, 256, 256]

Inputs:
    - data/interim/sdobenchmark/index.parquet
    - data/interim/sdobenchmark/splits/val.csv
    - data/interim/sdobenchmark/splits/test.csv
    - runs/checkpoints/cnn_timepool_best.pt

Output:
    - Printed device selection
    - Printed checkpoint existence check
    - Printed loaded checkpoint epoch
    - Printed validation-selected threshold
    - Printed validation and test TSS with confusion counts
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.sdobenchmark_dataset import SDOBenchmarkDataset
from src.models.cnn_timepool import CNNTimeMeanPool


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
# Resolve the project root relative to this script so it works
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[1]

# Curated index, split directory, and saved checkpoint location.
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
SPLITS = ROOT / "data" / "interim" / "sdobenchmark" / "splits"
VAL_CSV = SPLITS / "val.csv"
TEST_CSV = SPLITS / "test.csv"

CKPT = ROOT / "runs" / "checkpoints" / "cnn_timepool_best.pt"

# Evaluation loader settings.
BATCH_SIZE = 8
SHUFFLE = False
NUM_WORKERS = 0
PIN_MEMORY = True


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------
def confusion_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> tuple[int, int, int, int]:
    """
    Convert probabilities to hard predictions and compute confusion counts.

    Args:
        y_true (np.ndarray): Binary ground-truth labels.
        y_prob (np.ndarray): Predicted probabilities.
        threshold (float): Probability threshold for positive predictions.

    Returns:
        tuple[int, int, int, int]:
            - True positives
            - False positives
            - True negatives
            - False negatives
    """
    y = y_true.astype(np.int32)
    pred = (y_prob >= threshold).astype(np.int32)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    return tp, fp, tn, fn


def tss_from_confusion(tp: int, fp: int, tn: int, fn: int) -> float:
    """
    Compute the True Skill Statistic (TSS) from confusion counts.

    TSS is defined as:
        TSS = TPR - FPR

    Args:
        tp (int): True positives.
        fp (int): False positives.
        tn (int): True negatives.
        fn (int): False negatives.

    Returns:
        float: True Skill Statistic.
    """
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return float(tpr - fpr)


def best_tss_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """
    Find the probability threshold that maximizes TSS.

    The search is performed by sorting predicted probabilities in descending
    order and sweeping through the candidate cutoffs efficiently.

    Args:
        y_true (np.ndarray): Binary ground-truth labels.
        y_prob (np.ndarray): Predicted probabilities.

    Returns:
        tuple[float, float]:
            - Threshold that maximizes TSS
            - Best TSS achieved at that threshold
    """
    y = y_true.astype(np.int32)
    p = y_prob.astype(np.float64)

    order = np.argsort(-p)
    y_sorted = y[order]
    p_sorted = p[order]

    pos_total = int(y_sorted.sum())
    neg_total = int(len(y_sorted) - pos_total)

    if pos_total == 0 or neg_total == 0:
        return 0.5, 0.0

    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)

    tpr = tp_cum / pos_total
    fpr = fp_cum / neg_total
    tss = tpr - fpr

    best_index = int(np.argmax(tss))
    best_threshold = float(p_sorted[best_index])

    # Recompute confusion at the selected threshold so ties are handled cleanly.
    tp, fp, tn, fn = confusion_at_threshold(y, p, best_threshold)
    best_tss = tss_from_confusion(tp, fp, tn, fn)

    # Guardrail: TSS should remain within the valid range.
    assert -1.0001 <= best_tss <= 1.0001, f"TSS out of range: {best_tss}"

    return best_threshold, best_tss


# ---------------------------------------------------------------------
# Model output collection
# ---------------------------------------------------------------------
@torch.no_grad()
def collect_probs_and_labels(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model over a dataloader and collect labels and probabilities.

    The time-pooled CNN consumes the full input tensor:
        [B, 4, 10, 256, 256]

    Args:
        model (torch.nn.Module): Model in evaluation mode.
        dataloader (DataLoader): Split DataLoader.
        device (torch.device): Execution device.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Ground-truth labels with shape [N]
            - Predicted probabilities with shape [N]
    """
    model.eval()
    probs_all = []
    labels_all = []

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        logits = model(x)

        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        probs_all.append(probs)
        labels_all.append(y.numpy().reshape(-1))

    return np.concatenate(labels_all), np.concatenate(probs_all)


# ---------------------------------------------------------------------
# Builders and reporting helpers
# ---------------------------------------------------------------------
def build_dataloader(split_csv: Path) -> DataLoader:
    """
    Create a DataLoader for one saved dataset split.

    Args:
        split_csv (Path): Path to the split CSV file.

    Returns:
        DataLoader: DataLoader for the requested split.
    """
    dataset = SDOBenchmarkDataset(split_csv, INDEX)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def build_model(device: torch.device) -> torch.nn.Module:
    """
    Build the time-pooled CNN model for evaluation.

    Args:
        device (torch.device): Target device for model execution.

    Returns:
        torch.nn.Module: CNN time-mean-pooling model moved to the requested device.
    """
    model = CNNTimeMeanPool(in_channels=10)
    return model.to(device)


def load_checkpoint(
    model: torch.nn.Module,
    ckpt_path: Path,
    device: torch.device,
) -> dict:
    """
    Load model weights from a saved checkpoint.

    Args:
        model (torch.nn.Module): Model instance to populate.
        ckpt_path (Path): Path to the saved checkpoint file.
        device (torch.device): Device used for checkpoint loading.

    Returns:
        dict: Loaded checkpoint dictionary.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def print_tss_report(
    split_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> float:
    """
    Print TSS and confusion counts for a split at a fixed threshold.

    Args:
        split_name (str): Split name to display.
        y_true (np.ndarray): Ground-truth labels.
        y_prob (np.ndarray): Predicted probabilities.
        threshold (float): Decision threshold.

    Returns:
        float: TSS at the given threshold.
    """
    tp, fp, tn, fn = confusion_at_threshold(y_true, y_prob, threshold)
    tss = tss_from_confusion(tp, fp, tn, fn)

    print(
        f"{split_name}: {tss:.4f} | "
        f"TP={tp} FP={fp} TN={tn} FN={fn}"
    )
    return tss


def main() -> None:
    """
    Select a validation threshold by maximizing TSS and apply it to test.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("ckpt exists:", CKPT.exists(), "|", CKPT)

    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    # Load the trained model checkpoint.
    model = build_model(device)
    checkpoint = load_checkpoint(model, CKPT, device)
    print("loaded epoch:", checkpoint.get("epoch"))

    # Build validation and test loaders.
    val_loader = build_dataloader(VAL_CSV)
    test_loader = build_dataloader(TEST_CSV)

    # Collect probabilities and labels for both splits.
    y_val, p_val = collect_probs_and_labels(model, val_loader, device)
    y_test, p_test = collect_probs_and_labels(model, test_loader, device)

    # Choose a threshold on the validation set by maximizing TSS.
    threshold, val_tss = best_tss_threshold(y_val, p_val)

    print(f"\nChosen threshold (max TSS on val): {threshold:.6f}")
    print_tss_report("val TSS", y_val, p_val, threshold)
    print_tss_report("test TSS (using val threshold)", y_test, p_test, threshold)


if __name__ == "__main__":
    main()