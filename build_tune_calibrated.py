"""
Δρόμος Α — Αντιπροσωπευτική επιλογή D_tune.

1) Τρέχει baseline (low, κενό config) στο calibration pool (d_calib, 90).
2) Ταξινομεί τα σφάλματα -> ΓΕΝΙΚΗ κατανομή λαθών FinQA (σε αυτό το μοντέλο).
3) Επιλέγει 20 failures αναλογικά με τη γενική κατανομή, με floor >=1 ανά
   παρατηρούμενη κλάση (εγγυάται εκπροσώπηση scale_unit κ.λπ.).
4) Γράφει data/sample/d_tune.json (backup του παλιού) + calibration.json (provenance).

Το D_test ΔΕΝ αγγίζεται. Η επιλογή οδηγείται από τη γενική κατανομή, ΟΧΙ από το D_test.
Τρέχει στο cloud (API). Resumable μέσω evaluate_checkpointed.
"""
import json, shutil
from collections import Counter, defaultdict
import config as C
from src.dataset import load_sample
from src.logging_util import RunLogger
from src.loop import ConfigState
from run import make_answer_fn, evaluate_checkpointed

N_SELECT = 20

def largest_remainder(dist: dict, classes: list, remaining: int) -> dict:
    total = sum(dist[c] for c in classes) or 1
    exact = {c: remaining * dist[c] / total for c in classes}
    base = {c: int(exact[c]) for c in classes}
    used = sum(base.values())
    # μοίρασε τα υπόλοιπα με φθίνον remainder (tie-break: μεγαλύτερο dist, μετά αλφαβητικά)
    order = sorted(classes, key=lambda c: (-(exact[c] - base[c]), -dist[c], c))
    for c in order[: remaining - used]:
        base[c] += 1
    return base

def main():
    logger = RunLogger(tag="calib")
    d_calib = load_sample("d_calib")
    answer_fn = make_answer_fn(logger=logger)
    logger.line(f"=== CALIBRATION | {C.summary()} | model={C.MODEL_ID} ===")
    logger.line(f"d_calib = {len(d_calib)} (baseline low, κενό config)\n")

    results = evaluate_checkpointed(d_calib, C.EFFORT_SCHEDULE[0], ConfigState(),
                                    answer_fn, logger, "calib_baseline")

    valid = [r for r in results if not r.get("api_error")]
    failures = [r for r in valid if not r.get("correct")
                and r.get("error_class") not in (None, C.CORRECT_LABEL)]
    by_class = defaultdict(list)
    for r in failures:
        by_class[r["error_class"]].append(r)
    dist = {ec: len(rs) for ec, rs in by_class.items()}
    classes = sorted(dist, key=lambda c: (-dist[c], c))

    logger.line(f"\n--- ΓΕΝΙΚΗ κατανομή λαθών (n_valid={len(valid)}, "
                f"n_fail={len(failures)}, acc={1-len(failures)/max(len(valid),1):.3f}) ---")
    for c in classes:
        logger.line(f"  {c:24s} {dist[c]:2d}  ({dist[c]/max(len(failures),1)*100:4.1f}%)")

    # ── Επιλογή 20 ──
    fill_note = ""
    if len(failures) <= N_SELECT:
        selected = list(failures)
        # γέμισε ως τα 20 με 'hard correct' (3plus_step) ώστε N=20 σταθερό
        if len(selected) < N_SELECT:
            rest = [r for r in valid if r not in failures]
            rest.sort(key=lambda r: ({"3plus_step":0,"2_step":1,"1_step":2}.get(r["complexity"],3), str(r["id"])))
            need = N_SELECT - len(selected)
            selected += rest[:need]
            fill_note = f" (+{need} hard-correct fill: λιγότερα από {N_SELECT} failures)"
    else:
        alloc = {c: 1 for c in classes}               # floor
        base = largest_remainder(dist, classes, N_SELECT - len(classes))
        for c in classes:
            alloc[c] += base[c]
        # cap στη διαθεσιμότητα + redistribute τυχόν overflow
        overflow = 0
        for c in classes:
            if alloc[c] > dist[c]:
                overflow += alloc[c] - dist[c]; alloc[c] = dist[c]
        for c in sorted(classes, key=lambda c: (-dist[c], c)):
            while overflow > 0 and alloc[c] < dist[c]:
                alloc[c] += 1; overflow -= 1
        selected = []
        for c in classes:
            picks = sorted(by_class[c], key=lambda r: str(r["id"]))[: alloc[c]]
            selected.extend(picks)

    sel_dist = Counter(r.get("error_class") for r in selected)
    logger.line(f"\n--- ΕΠΙΛΕΓΜΕΝΟ D_tune = {len(selected)}{fill_note} ---")
    for c in sorted(sel_dist, key=lambda c:(-sel_dist[c], str(c))):
        logger.line(f"  {str(c):24s} {sel_dist[c]}")

    # απογύμνωσε στα raw instance fields (όπως τα περιμένει το run.py)
    id2inst = {i["id"]: i for i in d_calib}
    d_tune_new = [id2inst[r["id"]] for r in selected]

    # backup + γράψιμο
    old = C.SAMPLE_DIR / "d_tune.json"
    if old.exists():
        shutil.copy(old, C.SAMPLE_DIR / "d_tune_v1_baseline10.json")
    json.dump(d_tune_new, open(old, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    prov = {
        "method": "Δρόμος Α — representative selection (general FinQA error distribution)",
        "calib_pool_size": len(d_calib), "n_valid": len(valid), "n_failures": len(failures),
        "baseline_calib_accuracy": round(1 - len(failures)/max(len(valid),1), 4),
        "general_error_distribution": dist,
        "selected_tune_size": len(d_tune_new),
        "selected_error_distribution": dict(sel_dist),
        "selected_ids": [i["id"] for i in d_tune_new],
        "seed": C.SEED, "note": "D_test άθικτο· επιλογή από γενική κατανομή, όχι από D_test.",
    }
    json.dump(prov, open(logger.dir / "calibration.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    shutil.copy(logger.dir / "calibration.json", C.SAMPLE_DIR / "calibration.json")
    logger.line(f"\n>>> Νέο d_tune.json ({len(d_tune_new)}) γράφτηκε. "
                f"Provenance: {logger.dir/'calibration.json'}")

if __name__ == "__main__":
    main()
