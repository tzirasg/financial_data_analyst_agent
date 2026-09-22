"""
src/dataset.py — Κατασκευή των δειγμάτων D_test και D_tune από το FinQA.

- D_test = 50 held-out (από FinQA test.json), στρωματοποιημένα σε 6 κελιά
  (πηγή {table,text} × πολυπλοκότητα {1,2,3+ βήματα}), ισοκατανεμημένα.
  Χρησιμοποιείται ΜΟΝΟ στην τελική αξιολόγηση.
- D_tune = 10 «μέτριου» επιπέδου (από FinQA train.json), 2-step-centered ώστε ο
  agent να κάνει μαθήσιμα λάθη με χώρο βελτίωσης. Χρησιμοποιείται ΜΟΝΟ μέσα στον βρόχο.
- Τα δύο σύνολα προέρχονται από διαφορετικά splits -> δομικά ΜΗΔΕΝ leakage.

Έξοδος: data/sample/d_test.json , data/sample/d_tune.json (κανονικοποιημένες εγγραφές).
Τρέξιμο:  python -m src.dataset
"""
from __future__ import annotations
import json, re, random
from pathlib import Path
import config as C

_OPS = r'(add|subtract|multiply|divide|exp|greater|table_sum|table_average|table_max|table_min)\('


# ── Βοηθητικά χαρακτηρισμού στιγμιότυπων ──────────────────────────────
def n_steps(program: str) -> int:
    """Αριθμός πράξεων στο gold program -> proxy πολυπλοκότητας."""
    return len(re.findall(_OPS, program or ""))


def complexity_bin(program: str) -> str | None:
    s = n_steps(program)
    if s == 1: return "1_step"
    if s == 2: return "2_step"
    if s >= 3: return "3plus_step"
    return None  # 0 πράξεις (π.χ. constant-only) -> εκτός δείγματος


def source_of(gold_inds: dict) -> str:
    """table vs text από τα gold supporting facts (πλειοψηφία)."""
    keys = list((gold_inds or {}).keys())
    t = sum(k.startswith("table") for k in keys)
    x = sum(k.startswith("text") for k in keys)
    if t == 0 and x == 0: return "unknown"
    return "table" if t >= x else "text"


def is_numeric(v) -> bool:
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


def render_table(table) -> str:
    if not table: return "(no table)"
    return "\n".join(" | ".join(str(c) for c in row) for row in table)


def build_context(rec: dict) -> str:
    """Κανονικοποιημένο κείμενο συμφραζομένων (pre-text + πίνακας + post-text)."""
    pre = " ".join(rec.get("pre_text", []) or []).strip()
    post = " ".join(rec.get("post_text", []) or []).strip()
    table = render_table(rec.get("table", []))
    parts = []
    if pre:  parts.append("PRE-TEXT:\n" + pre)
    parts.append("TABLE:\n" + table)
    if post: parts.append("POST-TEXT:\n" + post)
    return "\n\n".join(parts)


def normalize(rec: dict, split: str) -> dict | None:
    qa = rec.get("qa", {})
    prog = qa.get("program", "")
    cbin = complexity_bin(prog)
    src = source_of(qa.get("gold_inds"))
    exe = qa.get("exe_ans")
    if cbin is None or src == "unknown" or not is_numeric(exe):
        return None
    return {
        "id": rec.get("id"),
        "question": qa.get("question", "").strip(),
        "context": build_context(rec),
        "gold_program": prog,
        "gold_answer": float(exe),      # αριθμητική αλήθεια (execution accuracy)
        "answer_str": str(qa.get("answer", "")).strip(),
        "source": src,
        "complexity": cbin,
        "split": split,
    }


def load_split(split: str) -> list[dict]:
    raw = json.load(open(C.RAW_DIR / f"{split}.json", encoding="utf-8"))
    out = []
    for r in raw:
        n = normalize(r, split)
        if n:
            out.append(n)
    return out


# ── Στρωματοποιημένη δειγματοληψία ────────────────────────────────────
def test_allocation() -> dict:
    """50 όσο πιο ισοκατανεμημένα γίνεται στα 6 κελιά."""
    cells = [(s, c) for s in C.STRATA_SOURCE for c in C.STRATA_COMPLEXITY]
    base, rem = divmod(C.N_TEST, len(cells))
    return {cell: base + (1 if i < rem else 0) for i, cell in enumerate(cells)}


def stratified_pick(pool: list[dict], allocation: dict, rng: random.Random) -> list[dict]:
    by_cell: dict[tuple, list] = {}
    for r in pool:
        by_cell.setdefault((r["source"], r["complexity"]), []).append(r)
    picked = []
    for cell, k in allocation.items():
        cand = list(by_cell.get(cell, []))
        rng.shuffle(cand)
        if len(cand) < k:
            raise ValueError(f"Λίγα στιγμιότυπα για {cell}: {len(cand)} < {k}")
        picked.extend(cand[:k])
    return picked


def build() -> tuple[list, list]:
    rng = random.Random(C.SEED)
    test_pool = load_split(C.TEST_SPLIT)
    tune_pool = load_split(C.TUNE_SPLIT)

    d_test = stratified_pick(test_pool, test_allocation(), rng)
    d_tune = stratified_pick(tune_pool, dict(C.TUNE_ALLOCATION), rng)

    # Έλεγχος no-leakage (διαφορετικά splits, αλλά επιβεβαιώνουμε και με id)
    overlap = {r["id"] for r in d_test} & {r["id"] for r in d_tune}
    assert not overlap, f"LEAKAGE! κοινά ids: {overlap}"

    C.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(d_test, open(C.SAMPLE_DIR / "d_test.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(d_tune, open(C.SAMPLE_DIR / "d_tune.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return d_test, d_tune


def _dist(rows: list[dict]) -> dict:
    from collections import Counter
    c = Counter((r["source"], r["complexity"]) for r in rows)
    return {f"{a}/{b}": n for (a, b), n in sorted(c.items())}


def load_sample(name: str) -> list[dict]:
    """name ∈ {'d_test','d_tune'} — φόρτωση έτοιμου δείγματος."""
    return json.load(open(C.SAMPLE_DIR / f"{name}.json", encoding="utf-8"))


if __name__ == "__main__":
    d_test, d_tune = build()
    print(f"D_test: {len(d_test)}  ->  {_dist(d_test)}")
    print(f"D_tune: {len(d_tune)}  ->  {_dist(d_tune)}")
    print(f"seed={C.SEED}  |  saved to {C.SAMPLE_DIR}")
    ex = d_tune[0]
    print("\n--- Δείγμα D_tune[0] ---")
    print("id:", ex["id"], "| source:", ex["source"], "| complexity:", ex["complexity"])
    print("Q:", ex["question"])
    print("gold_program:", ex["gold_program"], "| gold_answer:", ex["gold_answer"])
    print("context (πρώτοι 300 χαρ):", ex["context"][:300].replace("\n", " "))
