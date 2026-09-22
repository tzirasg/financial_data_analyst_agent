"""Χτίζει calibration pool 90 (15/κελί × 6) από το train split,
disjoint από D_test & το ΤΩΡΙΝΟ D_tune. Χωρίς API — μόνο επιλογή.
Seed=42 για αναπαραγωγιμότητα."""
import json, random
import config as C
from src.dataset import load_split

PER_CELL = 15
rng = random.Random(C.SEED)

d_test = json.load(open(C.SAMPLE_DIR/"d_test.json", encoding="utf-8"))
d_tune = json.load(open(C.SAMPLE_DIR/"d_tune.json", encoding="utf-8"))
exclude = {r["id"] for r in d_test} | {r["id"] for r in d_tune}

pool = [r for r in load_split(C.TUNE_SPLIT) if r["id"] not in exclude]
by_cell = {}
for r in pool:
    by_cell.setdefault((r["source"], r["complexity"]), []).append(r)

picked = []
for s in C.STRATA_SOURCE:
    for c in C.STRATA_COMPLEXITY:
        cand = list(by_cell.get((s, c), []))
        rng.shuffle(cand)
        if len(cand) < PER_CELL:
            raise ValueError(f"Λίγα για {(s,c)}: {len(cand)}")
        picked.extend(cand[:PER_CELL])

# no-leakage έλεγχος
assert not ({r["id"] for r in picked} & exclude), "LEAKAGE στο calib!"
json.dump(picked, open(C.SAMPLE_DIR/"d_calib.json","w",encoding="utf-8"),
          ensure_ascii=False, indent=2)
from collections import Counter
dist = Counter((r["source"], r["complexity"]) for r in picked)
print("d_calib:", len(picked), "instances")
for k in sorted(dist): print(" ", k, dist[k])
