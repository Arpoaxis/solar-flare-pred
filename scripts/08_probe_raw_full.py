"""
Inspect file types and example paths within the full SDOBenchmark raw dataset.

This script checks that the full raw dataset directory exists, previews its
top-level contents, samples up to a fixed number of filesystem entries from
the dataset tree, counts file extensions, and prints a small set of example
file paths for quick validation.

Input:
    - data/external/sdobenchmark/raw/SDOBenchmark-data-full/

Output:
    - Printed directory preview
    - Printed file-extension counts
    - Printed example file paths
"""

from collections import Counter
from pathlib import Path
import itertools


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
# Resolve the project root relative to this script so it works
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[1]

# Full SDOBenchmark raw dataset root.
RAW_FULL = ROOT / "data" / "external" / "sdobenchmark" / "raw" / "SDOBenchmark-data-full"

# Limit how many top-level entries are previewed.
MAX_TOP_PREVIEW = 30

# Limit how many filesystem entries are sampled during recursive scanning.
MAX_SCAN_ITEMS = 5000

# Limit how many example file paths are retained for display.
MAX_SAMPLE_PATHS = 20

# Limit how many example file paths are printed.
MAX_PRINT_PATHS = 10

# Limit how many extension counts are displayed.
MAX_EXTENSION_COUNTS = 10


def print_top_level_entries(root_path, limit):
    """
    Print a short preview of the dataset root contents.

    Args:
        root_path (Path): Dataset root to inspect.
        limit (int): Maximum number of entries to print.
    """
    top_level_items = sorted(root_path.iterdir())

    print("\nTop-level entries (first 30):")
    for path in top_level_items[:limit]:
        kind = "DIR " if path.is_dir() else "FILE"
        print(kind, path.name)


def scan_file_extensions(root_path, max_scan_items, max_sample_paths):
    """
    Sample filesystem entries under the dataset root and summarize file types.

    Args:
        root_path (Path): Dataset root to scan recursively.
        max_scan_items (int): Maximum number of paths to inspect.
        max_sample_paths (int): Maximum number of example file paths to retain.

    Returns:
        tuple[Counter, list[Path]]:
            - Counter of file extensions
            - Example file paths collected during scanning
    """
    extension_counts = Counter()
    sample_paths = []

    for path in itertools.islice(root_path.rglob("*"), max_scan_items):
        if not path.is_file():
            continue

        extension_counts[path.suffix.lower()] += 1

        if len(sample_paths) < max_sample_paths:
            sample_paths.append(path)

    return extension_counts, sample_paths


def main():
    """
    Inspect the raw dataset structure and summarize sampled file types.
    """
    print("raw_full exists:", RAW_FULL.exists(), "|", RAW_FULL)

    # Preview the immediate contents of the dataset root to confirm the
    # expected top-level directory layout.
    print_top_level_entries(RAW_FULL, MAX_TOP_PREVIEW)

    # Sample the dataset tree to get a quick view of file types and paths
    # without scanning the entire directory structure.
    extension_counts, sample_paths = scan_file_extensions(
        root_path=RAW_FULL,
        max_scan_items=MAX_SCAN_ITEMS,
        max_sample_paths=MAX_SAMPLE_PATHS,
    )

    print("\nMost common extensions (top 10):", extension_counts.most_common(MAX_EXTENSION_COUNTS))

    print("\nExample file paths (first 10):")
    for path in sample_paths[:MAX_PRINT_PATHS]:
        print(" -", path)


if __name__ == "__main__":
    main()