import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "interim" / "sdobenchmark" / "index.parquet"
df = pd.read_parquet(INDEX)

train = df[df["split"] == "training"].copy()
ars = sorted(train["ar"].unique())
n_val = int(round(0.1 * len(ars)))

print("training ARs: ", len(ars))
print("val ARs (10%): ", n_val)


train["start_dt"] = pd.to_datetime(train["start"])
# For each AR: get the latest observed start time
ar_last_time = train.groupby("ar")["start_dt"].max().sort_values()

TARGET_POS = 40  #Target for the number of positive ARs for the validation set

# positives per AR
ar_pos = train.groupby("ar")["label_m1p"].sum()

# expand candidate pool backwards in time until it contains enough positives
ordered_ars = list(ar_last_time.index) #oldest -> newest
cand = []
cand_pos = 0
for ar in reversed(ordered_ars): # newest -> oldest
    cand.append(ar)
    cand_pos += int(ar_pos.get(ar, 0))
    if cand_pos >= TARGET_POS and len(cand) >= n_val:
        break

cand_set = set(cand)

# choose validation ARs: positives first (latest-first), then fill with latest remaining
pos_ars = [ar for ar in reversed(cand) if int(ar_pos.get(ar, 0)) > 0 ]
neg_ars = [ar for ar in reversed(cand) if int(ar_pos.get(ar, 0)) == 0 ]

val_list = []
pos_count = 0
for ar in pos_ars:
    if len(val_list) >= n_val:
        break
    val_list.append(ar)
    pos_count += int(ar_pos.get(ar, 0))
    if pos_count >= TARGET_POS:
        break

for ar in neg_ars:
    if len(val_list) >= n_val:
        break
    if ar not in val_list:
        val_list.append(ar)

val_ars = set(val_list)
print("candidate ARs: ", len(cand_set), "candidate positives: ", cand_pos)
print("val ARs:", len(val_ars), "val positives:", int(train[train["ar"].isin(val_ars)]["label_m1p"].sum()))


print("earliest AR last-time:", ar_last_time.iloc[0])
print("latest AR last-time:", ar_last_time.iloc[-1])

train_ars = set(ars) - val_ars

SPLIT_DIR = Path(r"E:\solar-flare-pred\data\interim\sdobenchmark\splits")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

train_rows = train[train["ar"].isin(train_ars)].copy()
val_rows = train[train["ar"].isin(val_ars)].copy()
test_rows = df[df["split"] == "test"].copy()

cols = ["id", "ar", "start", "end", "peak_flux", "label_m1p"]

train_rows[cols].to_csv(SPLIT_DIR / "train.csv", index=False)
val_rows[cols].to_csv(SPLIT_DIR / "val.csv", index=False)
test_rows[cols].to_csv(SPLIT_DIR / "test.csv", index=False)

print("wrote:", SPLIT_DIR)
print("train rows:", len(train_rows), "val rows:", len(val_rows), "test rows:", len(test_rows))

# quick AR-leakage check
print("AR overlap train∩val:", len(set(train_rows["ar"]).intersection(set(val_rows["ar"]))))

overlaps = {
    "train∩val": len(set(train_rows["id"]) & set(val_rows["id"])),
    "train∩test": len(set(train_rows["id"]) & set(test_rows["id"])),
    "val∩test": len(set(val_rows["id"]) & set(test_rows["id"])),
}
print("ID overlaps:", overlaps)

def stats(name, d):
    pos = int(d["label_m1p"].sum())
    n = len(d)
    print(f"{name}: rows = {n}, positives = {pos}, pos_rate ={pos/n:.4f}, ARs = {d['ar'].nunique()}")

stats("train", train_rows)
stats("val", val_rows)
stats("test", test_rows)

ar_pos = train.groupby("ar")["label_m1p"].sum().sort_values(ascending=False)
print("ARs with ≥1 positive:", int((ar_pos >= 1).sum()))
print("Top 10 AR positive counts:\n", ar_pos.head(10))

print("SPLIT_POLICY: AR-disjoint; time-aware candidate pool expanded until >=TARGET_POS positives; TARGET_POS=", TARGET_POS)