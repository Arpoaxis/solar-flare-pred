from pathlib import Path
import itertools

ROOT = Path(__file__).resolve().parents[1]
raw_root = ROOT / "data" / "external" / "sdobenchmark" / "raw"

items = list(raw_root.iterdir())
print("raw_root:", raw_root)
print("num items:", len(items))

# show first 30 entries
for p in items[:30]:
    print(("DIR " if p.is_dir() else "FILE"), p.name)

# if there's exactly one folder, that's probably the real data root
dirs = [p for p in items if p.is_dir()]
print("num dirs:", len(dirs))
if len(dirs) == 1:
    print("single dir candidate:", dirs[0])