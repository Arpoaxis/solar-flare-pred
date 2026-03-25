"""
Inspect the raw SDOBenchmark data directory layout.

This script lists the immediate contents of the raw data directory,
reports how many items and subdirectories are present, and highlights
the single directory candidate when the raw folder appears to contain
one top-level dataset root.

Input:
    - data/external/sdobenchmark/raw/

Output:
    - Printed directory layout summary
"""

from pathlib import Path


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
# Resolve the project root relative to this script so it works
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[1]

# Raw SDOBenchmark parent directory to inspect.
RAW_ROOT = ROOT / "data" / "external" / "sdobenchmark" / "raw"

# Limit how many entries are previewed in the console.
MAX_PREVIEW_ITEMS = 30


def print_directory_preview(items, limit):
    """
    Print a short preview of directory contents.

    Args:
        items (list[Path]): Filesystem entries to preview.
        limit (int): Maximum number of entries to print.
    """
    for path in items[:limit]:
        kind = "DIR " if path.is_dir() else "FILE"
        print(kind, path.name)


def find_subdirectories(items):
    """
    Return only the directory entries from a list of filesystem items.

    Args:
        items (list[Path]): Filesystem entries under the raw root.

    Returns:
        list[Path]: Subdirectories found in the provided items.
    """
    return [path for path in items if path.is_dir()]


def main():
    """
    Inspect the top-level contents of the raw dataset directory.
    """
    # Read the immediate contents of the raw dataset directory.
    items = list(RAW_ROOT.iterdir())

    print("raw_root:", RAW_ROOT)
    print("num items:", len(items))

    # Show a short preview of the first entries to confirm the directory layout.
    print_directory_preview(items, MAX_PREVIEW_ITEMS)

    # If only one subdirectory exists, it is likely the actual dataset root.
    dirs = find_subdirectories(items)
    print("num dirs:", len(dirs))

    if len(dirs) == 1:
        print("single dir candidate:", dirs[0])


if __name__ == "__main__":
    main()