"""
src/metrics.py — Μετρικές αξιολόγησης (FinQA).

Πρωτεύουσες:
  - execution accuracy: |pred - gold| <= 1% * |gold| (ανοχή 1%, σχετική).
  - program accuracy: αντιστοιχία ακολουθίας πράξεων+τελεστέων (proxy — βλ. σημείωση).
Δευτερεύοντα (στο aggregate): ανά-κελί ακρίβεια, ανά-κλάση σφάλματος, tokens.

Σημείωση program accuracy: ο agent παράγει «ελεύθερο» πρόγραμμα, οπότε συγκρίνουμε
κανονικοποιημένη ακολουθία (op, αριθμητικοί τελεστέοι). Είναι proxy του επίσημου
FinQA program match — καταγράφεται ως δευτερεύουσα μετρική (Threats to Validity).
"""
from __future__ import annotations
import re
import config as C

_NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")
_OP = re.compile(
    r"(add|subtract|multiply|divide|exp|greater|table_sum|table_average|table_max|table_min)"
    r"\s*\(([^)]*)\)"
)


# ── Execution accuracy ────────────────────────────────────────────────
def execution_correct(pred, gold, tol: float = None) -> bool:
    tol = C.EXEC_TOLERANCE if tol is None else tol
    if pred is None or gold is None:
        return False
    try:
        pred = float(pred); gold = float(gold)
    except (TypeError, ValueError):
        return False
    if abs(gold) < 1e-9:
        return abs(pred) <= tol
    return abs(pred - gold) <= tol * abs(gold)


# ── Program accuracy (proxy) ──────────────────────────────────────────
def numbers_in(text: str) -> list[float]:
    out = []
    for tok in _NUM.findall(text or ""):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return out


def normalize_program(prog: str) -> tuple:
    """(op, στρογγυλεμένοι αριθμητικοί τελεστέοι) ανά πράξη, με σειρά."""
    ops = []
    for m in _OP.finditer(prog or ""):
        op = m.group(1)
        nums = tuple(round(n, 4) for n in numbers_in(m.group(2)))
        ops.append((op, nums))
    return tuple(ops)


def program_correct(pred_program: str, gold_program: str) -> bool:
    gp = normalize_program(gold_program)
    if not gp:
        return False
    return normalize_program(pred_program) == gp


# ── Βαθμολόγηση ενός αποτελέσματος ────────────────────────────────────
def grade(result: dict) -> dict:
    result = dict(result)
    result["correct"] = execution_correct(result.get("pred_answer"), result.get("gold_answer"))
    result["program_ok"] = program_correct(result.get("pred_program"), result.get("gold_program"))
    return result


# ── Συγκεντρωτικά ─────────────────────────────────────────────────────
def aggregate(results: list[dict]) -> dict:
    n = len(results)
    graded = [g if "correct" in g else grade(g) for g in results]
    exec_acc = sum(g["correct"] for g in graded) / n if n else 0.0
    prog_acc = sum(g.get("program_ok", False) for g in graded) / n if n else 0.0

    per_cell: dict[str, dict] = {}
    for g in graded:
        cell = f"{g.get('source','?')}/{g.get('complexity','?')}"
        d = per_cell.setdefault(cell, {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += int(g["correct"])
    for d in per_cell.values():
        d["acc"] = d["correct"] / d["n"] if d["n"] else 0.0

    err_classes = {}
    for g in graded:
        ec = g.get("error_class")
        if ec and ec != C.CORRECT_LABEL:
            err_classes[ec] = err_classes.get(ec, 0) + 1

    thoughts = sum(g.get("usage", {}).get("thought_tokens", 0) for g in graded)
    return {
        "n": n,
        "execution_accuracy": round(exec_acc, 4),
        "program_accuracy": round(prog_acc, 4),
        "n_correct": sum(g["correct"] for g in graded),
        "per_cell": per_cell,
        "error_classes": dict(sorted(err_classes.items(), key=lambda kv: -kv[1])),
        "thought_tokens_total": thoughts,
    }


if __name__ == "__main__":
    assert execution_correct(0.0990, 0.0986) is True   # εντός 1%
    assert execution_correct(9.9, 0.099) is False       # scale mismatch
    assert execution_correct(-35.0, -35.0) is True
    assert program_correct("subtract(22471, 20624)", "subtract(22471, 20624)") is True
    assert program_correct("add(1,2)", "subtract(1,2)") is False
    print("metrics self-test OK")
