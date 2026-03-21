from pathlib import Path
from collections import Counter
import itertools

ROOT = Path(__file__).resolve().parents[1]
raw_full = ROOT / "data" / "external" / "sdobenchmark" / "raw" / "SDOBenchmark-data-full"

print("raw_full exists:", raw_full.exists(), "|", raw_full)

# show top-level entries
top = sorted(raw_full.iterdir())
print("\nTop-level entries (first 30):")
for p in top[:30]:
    print(("DIR " if p.is_dir() else "FILE"), p.name)

# sample some files and count extensions
ext = Counter()
sample_paths = []

for p in itertools.islice(raw_full.rglob("*"), 5000):
    if p.is_file():
        ext[p.suffix.lower()] += 1
        if len(sample_paths) < 20:
            sample_paths.append(p)

print("\nMost common extensions (top 10):", ext.most_common(10))
print("\nExample file paths (first 10):")
for p in sample_paths[:10]:
    print(" -", p)