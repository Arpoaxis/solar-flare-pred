"""
PyTorch dataset for loading SDOBenchmark samples from saved split files.

This dataset reads sample IDs and labels from a split CSV, resolves each
sample directory through the curated index parquet file, and loads up to
four complete timestamps per sample using a fixed channel order.

For each selected timestamp, the dataset loads all ten modality images and
stacks them into a frame of shape:
    [C, H, W] = [10, 256, 256]

Frames are then stacked over time to produce the final sample tensor:
    [T, C, H, W] = [4, 10, 256, 256]

If fewer than four complete timestamps are available, the oldest complete
timestamp is duplicated at the front until four frames are present.
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
# Fixed channel order used when assembling one sample tensor.
CHANNELS = [
    "94",
    "131",
    "171",
    "193",
    "211",
    "304",
    "335",
    "1700",
    "continuum",
    "magnetogram",
]

EXPECTED_TIMESTEPS = 4
EXPECTED_CHANNELS = 10
EXPECTED_SHAPE = (4, 10, 256, 256)
IMAGE_SUFFIX = ".jpg"


class SDOBenchmarkDataset(Dataset):
    """
    Dataset for loading SDOBenchmark samples as [T, C, H, W] tensors.

    Args:
        split_csv (Path): CSV file containing sample IDs and labels.
        index_parquet (Path): Parquet file mapping sample IDs to sample directories.
    """

    def __init__(self, split_csv: Path, index_parquet: Path) -> None:
        self.df = pd.read_csv(split_csv)

        # Load only the fields needed to map sample IDs to their directories.
        index_df = pd.read_parquet(index_parquet)[["id", "sample_dir"]].copy()

        # Store IDs as strings so lookup stays consistent across CSV/parquet reads.
        index_df["id"] = index_df["id"].astype(str)
        self.id_to_dir = dict(zip(index_df["id"].values, index_df["sample_dir"].values))

    def __len__(self) -> int:
        """
        Return the number of samples in the split.
        """
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        """
        Load one sample and its binary label.

        Args:
            i (int): Row index in the split CSV.

        Returns:
            tuple[torch.Tensor, int]:
                - Sample tensor with shape [4, 10, 256, 256]
                - Binary flare label
        """
        row = self.df.iloc[i]
        sample_id = str(row["id"])
        label = int(row["label_m1p"])

        sample_dir = Path(self.id_to_dir[sample_id])

        # Discover available timestamps from the files present in the sample directory.
        files = sorted(sample_dir.glob(f"*{IMAGE_SUFFIX}"))
        timestamps = sorted({path.stem.split("__")[0] for path in files})

        # Cache existing filenames once to avoid repeated filesystem exists() calls.
        existing_files = {path.name for path in files}

        # Keep only timestamps that contain all required channel files.
        complete_timestamps = self._find_complete_timestamps(timestamps, existing_files)

        if not complete_timestamps:
            raise RuntimeError(f"{sample_id}: no complete timestamps found")

        # Use the most recent four complete timestamps when available.
        # If fewer than four exist, pad by duplicating the oldest complete timestamp.
        selected_timestamps = self._select_timestamps(complete_timestamps)

        # Load the selected timestamps in [T, C, H, W] format.
        x = self._load_sample_tensor(sample_dir, selected_timestamps)

        assert tuple(x.shape) == EXPECTED_SHAPE, x.shape
        return x, label

    def _find_complete_timestamps(
        self,
        timestamps: list[str],
        existing_files: set[str],
    ) -> list[str]:
        """
        Return timestamps that contain all required channel files.

        Args:
            timestamps (list[str]): Candidate timestamps found in the sample directory.
            existing_files (set[str]): Filenames already present in the directory.

        Returns:
            list[str]: Timestamps with complete channel coverage.
        """
        complete_timestamps = []

        for timestamp in timestamps:
            if all(f"{timestamp}__{channel}{IMAGE_SUFFIX}" in existing_files for channel in CHANNELS):
                complete_timestamps.append(timestamp)

        return complete_timestamps

    def _select_timestamps(self, complete_timestamps: list[str]) -> list[str]:
        """
        Select exactly four timestamps for one sample.

        Args:
            complete_timestamps (list[str]): Available complete timestamps in sorted order.

        Returns:
            list[str]: Exactly four timestamps.
        """
        if len(complete_timestamps) >= EXPECTED_TIMESTEPS:
            return complete_timestamps[-EXPECTED_TIMESTEPS:]

        selected = complete_timestamps[:]
        while len(selected) < EXPECTED_TIMESTEPS:
            selected = [complete_timestamps[0]] + selected

        return selected

    def _load_sample_tensor(
        self,
        sample_dir: Path,
        timestamps: list[str],
    ) -> torch.Tensor:
        """
        Load one sample as a [T, C, H, W] tensor.

        Args:
            sample_dir (Path): Directory containing the sample image files.
            timestamps (list[str]): Selected timestamps to load.

        Returns:
            torch.Tensor: Sample tensor with shape [4, 10, 256, 256].
        """
        frames = []

        for timestamp in timestamps:
            channel_tensors = []

            for channel in CHANNELS:
                image_path = sample_dir / f"{timestamp}__{channel}{IMAGE_SUFFIX}"
                img = read_image(str(image_path)).float() / 255.0
                channel_tensors.append(img)

            frame = torch.cat(channel_tensors, dim=0)  # [10, 256, 256]
            frames.append(frame)

        return torch.stack(frames, dim=0)  # [4, 10, 256, 256]