"""Ανεξάρτητο dev slice = 30 (5/κελί × 6), φυσικό mix, από train.
Disjoint από d_test, d_calib(90) [άρα και d_tune⊂calib], d_tune. Χωρίς API."""
import json, random
import config as C
from src.dataset import load_split

PER_CELL = 5
rng = random.Random(C.SEED + 7)  # διαφορετικό stream· ούτως ή άλλως αποκλείουμε calib ids

def ids(name):
    try: return {r["id"] for r in json.load(open(C.SAMPLE_DIR/f"{name}.json", encoding="utf-8"))}
    except FileNotFoundError: return set()

exclude = ids("d_test") | ids("d_calib") | ids("d_tune")
pool = [r for r in load_split(C.TUNE_SPLIT) if r["id"] not in exclude]
by_cell = {}
for r in pool:
    by_cell.setdefault((r["source"], r["complexity"]), []).append(r)

picked = []
for s in C.STRATA_SOURCE:
    for c in C.STRATA_COMPLEXITY:
        cand = list(by_cell.get((s, c), [])); rng.shuffle(cand)
        if len(cand) < PER_CELL: raise ValueError(f"Λίγα για {(s,c)}: {len(cand)}")
        picked.extend(cand[:PER_CELL])

assert not ({r["id"] for r in picked} & exclude), "LEAKAGE στο dev!"
json.dump(picked, open(C.SAMPLE_DIR/"d_dev.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
from collections import Counter
print("d_dev:", len(picked))
for k in sorted(Counter((r["source"],r["complexity"]) for r in picked)): print("  ",k)
