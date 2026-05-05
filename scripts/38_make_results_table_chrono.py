"""
Summarize validation and test results across chronological-holdout models.

This script evaluates saved checkpoints trained on the chronological splits,
computes threshold-free and threshold-based metrics, and writes a single CSV
summary suitable for comparison tables in the report or paper.

Metrics reported:
    - Average Precision (used here as PR-AUC)
    - Brier score
    - average BCE-with-logits loss
    - True Skill Statistic (TSS), where the decision threshold is chosen on
      the chronological validation split and then applied unchanged to the
      chronological test split

Model input modes:
    - "last": uses only the final timestep
        [B, 4, 10, 256, 256] -> [B, 10, 256, 256]
    - "full": uses the full spatiotemporal tensor
        [B, 4, 10, 256, 256]

Inputs:
    - data/interim/sdobenchmark/index.parquet
    - data/interim/sdobenchmark/splits/chrono_val.csv or chrono-val.csv
    - data/interim/sdobenchmark/splits/chrono_test.csv or chrono-test.csv
    - runs/checkpoints/chrono_*.pt

Output:
    - runs/logs/results_summary_chrono.csv
"""

from pathlib import Path
import csv
from typing import Any

import numpy as np
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
OUT_CSV = ROOT / "runs" / "logs" / "results_summary_chrono.csv"

# Evaluation loader settings.
BATCH_SIZE = 8
SHUFFLE = False
NUM_WORKERS = 0
PIN_MEMORY = True

# Model specifications for the chronological comparison table.
MODEL_SPECS: list[dict[str, Any]] = [
    {
        "name": "cnn_baseline",
        "mode": "last",
        "checkpoint": "chrono_cnn_baseline_best.pt",
        "builder": lambda device: CNNBaseline(in_channels=10).to(device),
    },
    {
        "name": "cnn_baseline_posw",
        "mode": "last",
        "checkpoint": "chrono_cnn_baseline_posw_best.pt",
        "builder": lambda device: CNNBaseline(in_channels=10).to(device),
    },
    {
        "name": "cnn_timepool",
        "mode": "full",
        "checkpoint": "chrono_cnn_timepool_best.pt",
        "builder": lambda device: CNNTimeMeanPool(in_channels=10).to(device),
    },
    {
        "name": "cnn_timepool_es",
        "mode": "full",
        "checkpoint": "chrono_cnn_timepool_es_best.pt",
        "builder": lambda device: CNNTimeMeanPool(in_channels=10).to(device),
    },
    {
        "name": "cnn_gru",
        "mode": "full",
        "checkpoint": "chrono_cnn_gru_best.pt",
        "builder": lambda device: CNNGRU(
            in_channels=10,
            feat_dim=128,
            hidden_dim=128,
        ).to(device),
    },
    {
        "name": "cnn_gru_es",
        "mode": "full",
        "checkpoint": "chrono_cnn_gru_es_best.pt",
        "builder": lambda device: CNNGRU(
            in_channels=10,
            feat_dim=128,
            hidden_dim=128,
        ).to(device),
    },
    {
        "name": "cnn_gru_posw",
        "mode": "full",
        "checkpoint": "chrono_cnn_gru_posw_best.pt",
        "builder": lambda device: CNNGRU(
            in_channels=10,
            feat_dim=128,
            hidden_dim=128,
        ).to(device),
    },
    {
        "name": "cnn_gru_posw_es",
        "mode": "full",
        "checkpoint": "chrono_cnn_gru_posw_es_best.pt",
        "builder": lambda device: CNNGRU(
            in_channels=10,
            feat_dim=128,
            hidden_dim=128,
        ).to(device),
    },
]


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------
def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Average Precision (PR-AUC) using a ranking-based formulation.

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

    The threshold is selected on the chronological validation split and
    then applied unchanged to the chronological test split.

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
    threshold = float(p_sorted[best_index])

    # Recompute confusion at the chosen threshold to handle ties cleanly.
    tp, fp, tn, fn = confusion_at_threshold(y, p, threshold)
    best_tss = tss_from_confusion(tp, fp, tn, fn)

    assert -1.0001 <= best_tss <= 1.0001, f"TSS out of range: {best_tss}"
    return threshold, best_tss


# ---------------------------------------------------------------------
# Data and model helpers
# ---------------------------------------------------------------------
def find_chrono_csv(splits_dir: Path, split_name: str) -> Path:
    """
    Find a chronological split CSV using either underscore or hyphen naming.

    Supported filename patterns:
        - chrono_val.csv / chrono_test.csv
        - chrono-val.csv / chrono-test.csv

    Args:
        splits_dir (Path): Directory containing split CSV files.
        split_name (str): Split suffix, such as "val" or "test".

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


def build_dataloader(split_csv: Path) -> DataLoader:
    """
    Create a DataLoader for one saved split.

    Args:
        split_csv (Path): Path to the split CSV file.

    Returns:
        DataLoader: DataLoader for the split.
    """
    dataset = SDOBenchmarkDataset(split_csv, INDEX)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    """
    Load model weights from a saved checkpoint.

    Args:
        model (nn.Module): Model instance to populate.
        checkpoint_path (Path): Path to the saved checkpoint file.
        device (torch.device): Device used for checkpoint loading.

    Returns:
        dict: Loaded checkpoint dictionary.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


@torch.no_grad()
def collect_probs_and_loss(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run a model over one split and collect labels, probabilities, and loss.

    BCE loss here is intentionally unweighted for comparability across
    models, including those trained with positive-class weighting.

    Args:
        model (nn.Module): Model in evaluation mode.
        dataloader (DataLoader): Split DataLoader.
        device (torch.device): Execution device.
        mode (str): Input mode, either "last" or "full".

    Returns:
        tuple[np.ndarray, np.ndarray, float]:
            - Ground-truth labels with shape [N]
            - Predicted probabilities with shape [N]
            - Average BCE-with-logits loss
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    probs_all = []
    labels_all = []

    total_loss = 0.0
    batch_count = 0

    for x, y in dataloader:
        y_float = y.float().to(device).view(-1, 1)

        if mode == "last":
            # Use only the final timestep for last-frame baselines.
            x_input = x[:, -1].to(device, non_blocking=True)   # [B, 10, 256, 256]
        else:
            # Use the full spatiotemporal tensor for time-aware models.
            x_input = x.to(device, non_blocking=True)          # [B, 4, 10, 256, 256]

        logits = model(x_input)
        loss = criterion(logits, y_float)

        probs_all.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
        labels_all.append(y.numpy().reshape(-1))

        total_loss += float(loss.detach().cpu())
        batch_count += 1

    y_true = np.concatenate(labels_all)
    y_prob = np.concatenate(probs_all)
    avg_loss = total_loss / max(batch_count, 1)

    return y_true, y_prob, avg_loss


def summarize_model_results(
    model: nn.Module,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    device: torch.device,
    mode: str,
) -> dict[str, float]:
    """
    Evaluate one model on chrono-val and chrono-test and return summary metrics.

    The decision threshold is chosen on chrono-val by maximizing TSS and then
    applied unchanged to chrono-test.

    Args:
        model (nn.Module): Loaded model for evaluation.
        val_dataloader (DataLoader): Chronological validation DataLoader.
        test_dataloader (DataLoader): Chronological test DataLoader.
        device (torch.device): Execution device.
        mode (str): Input mode, either "last" or "full".

    Returns:
        dict[str, float]: Summary metrics for one model.
    """
    y_val, p_val, val_loss = collect_probs_and_loss(model, val_dataloader, device, mode)
    y_test, p_test, test_loss = collect_probs_and_loss(model, test_dataloader, device, mode)

    val_ap = average_precision(y_val, p_val)
    test_ap = average_precision(y_test, p_test)

    val_brier = float(np.mean((p_val - y_val) ** 2))
    test_brier = float(np.mean((p_test - y_test) ** 2))

    threshold, _ = best_tss_threshold(y_val, p_val)

    tp, fp, tn, fn = confusion_at_threshold(y_val, p_val, threshold)
    val_tss = tss_from_confusion(tp, fp, tn, fn)

    tp, fp, tn, fn = confusion_at_threshold(y_test, p_test, threshold)
    test_tss = tss_from_confusion(tp, fp, tn, fn)

    return {
        "val_ap": val_ap,
        "test_ap": test_ap,
        "val_brier": val_brier,
        "test_brier": test_brier,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "threshold": threshold,
        "val_tss": val_tss,
        "test_tss": test_tss,
    }


def main() -> None:
    """
    Evaluate all configured chronological models and write a summary CSV.
    """
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    chrono_val_csv = find_chrono_csv(SPLITS, "val")
    chrono_test_csv = find_chrono_csv(SPLITS, "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("chrono_val:", chrono_val_csv)
    print("chrono_test:", chrono_test_csv)

    # Build the shared validation and test loaders once for all models.
    val_dataloader = build_dataloader(chrono_val_csv)
    test_dataloader = build_dataloader(chrono_test_csv)

    with open(OUT_CSV, "w", newline="") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(
            [
                "model",
                "ckpt_epoch",
                "val_ap",
                "test_ap",
                "val_brier",
                "test_brier",
                "val_loss",
                "test_loss",
                "tss_thresh_val",
                "val_tss",
                "test_tss",
            ]
        )

        for spec in MODEL_SPECS:
            checkpoint_path = CHECKPOINT_DIR / spec["checkpoint"]

            if not checkpoint_path.exists():
                print("SKIP (missing):", spec["name"], checkpoint_path.name)
                continue

            model = spec["builder"](device)
            checkpoint = load_checkpoint(model, checkpoint_path, device)
            epoch = checkpoint.get("epoch")

            results = summarize_model_results(
                model=model,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                device=device,
                mode=spec["mode"],
            )

            writer.writerow(
                [
                    spec["name"],
                    epoch,
                    f"{results['val_ap']:.4f}",
                    f"{results['test_ap']:.4f}",
                    f"{results['val_brier']:.4f}",
                    f"{results['test_brier']:.4f}",
                    f"{results['val_loss']:.4f}",
                    f"{results['test_loss']:.4f}",
                    f"{results['threshold']:.6f}",
                    f"{results['val_tss']:.4f}",
                    f"{results['test_tss']:.4f}",
                ]
            )
            out_file.flush()

            print(f"WROTE: {spec['name']} (epoch {epoch})")

    print("saved:", OUT_CSV)


if __name__ == "__main__":
    main()