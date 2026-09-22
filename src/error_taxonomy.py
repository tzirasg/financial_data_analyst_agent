"""
src/error_taxonomy.py — Ταξινόμηση σφαλμάτων FinQA στις 7 κλάσεις του σχεδίου.

Κλάσεις: scale_unit, wrong_cell, wrong_operation, sign_error,
         multi_step_order, hallucinated_constant, format_error  (+ 'correct').

Default: ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΟΙ ΚΑΝΟΝΕΣ (αναπαραγώγιμο, χωρίς επιπλέον κλήσεις API).
Η λογική:
  1) σωστό (εντός 1%)                              -> correct
  2) δεν έβγαλε/δεν parse-άρεται αριθμός           -> format_error
  3) pred ≈ -gold                                  -> sign_error
  4) pred ≈ gold × 10^k (π.χ. ποσοστό ως %, χιλ.)  -> scale_unit
  5) χρήση αριθμού που ΔΕΝ υπάρχει στο context      -> hallucinated_constant
  6) διαφορετικές πράξεις από το gold              -> wrong_operation
  7) ίδιες πράξεις, διαφορετικοί τελεστέοι         -> wrong_cell
  8) ίδιες πράξεις & τελεστέοι, λάθος αποτέλεσμα   -> multi_step_order
  fallback                                         -> wrong_cell

Προαιρετικά JUDGE_MODE="llm": LLM-as-judge για τις κλάσεις 5–8 (σημασιολογικές).
"""
from __future__ import annotations
import config as C
from src.metrics import execution_correct, normalize_program, numbers_in

_SCALES = [10, 100, 1000, 10000, 0.1, 0.01, 0.001, 0.0001]
_REL = 0.02  # ανοχή αντιστοίχισης για sign/scale/context (2%)


def _approx(a, b, rel=_REL) -> bool:
    if a is None or b is None:
        return False
    if abs(b) < 1e-9:
        return abs(a) < 1e-9
    return abs(a - b) <= rel * abs(b)


def _context_numbers(instance) -> list[float]:
    return numbers_in(instance.get("context", "")) if instance else []


def _operands(prog: str) -> list[float]:
    nums = []
    for _op, args in normalize_program(prog):
        nums.extend(args)
    return nums


def _op_names(prog: str) -> list[str]:
    return [op for op, _ in normalize_program(prog)]


def classify(result: dict, instance: dict | None = None) -> str:
    if result.get("api_error"):
        return "api_error"
    gold = result.get("gold_answer")
    pred = result.get("pred_answer")

    # 1) σωστό
    if execution_correct(pred, gold):
        return C.CORRECT_LABEL
    # 2) format
    if pred is None or not result.get("parsed_ok", True):
        return "format_error"
    # 3) sign
    if gold is not None and _approx(pred, -gold):
        return "sign_error"
    # 4) scale/unit (π.χ. έγραψε 9.9 αντί 0.099, ή millions vs thousands)
    if gold not in (None, 0):
        for k in _SCALES:
            if _approx(pred, gold * k):
                return "scale_unit"

    # ── program-based (κλάσεις 5–8) ──
    gold_ops = _op_names(result.get("gold_program", ""))
    pred_ops = _op_names(result.get("pred_program", ""))
    gold_nums = _operands(result.get("gold_program", ""))
    pred_nums = _operands(result.get("pred_program", ""))
    ctx = _context_numbers(instance)

    # 5) hallucinated constant: τελεστέος του agent που δεν υπάρχει στο context
    if pred_nums and ctx:
        for n in pred_nums:
            if not any(_approx(n, c) for c in ctx):
                return "hallucinated_constant"

    if gold_ops and pred_ops:
        # 6) διαφορετικό σύνολο πράξεων
        if sorted(pred_ops) != sorted(gold_ops):
            return "wrong_operation"
        # 7) ίδιες πράξεις, διαφορετικοί τελεστέοι -> λάθος κελί/τιμή
        if sorted(round(x, 2) for x in pred_nums) != sorted(round(x, 2) for x in gold_nums):
            return "wrong_cell"
        # 8) ίδια πράξεις & τελεστέοι αλλά λάθος αποτέλεσμα -> σειρά βημάτων
        return "multi_step_order"

    # fallback: αν ξέρουμε ότι οι πράξεις διαφέρουν
    if gold_ops and pred_ops and sorted(pred_ops) != sorted(gold_ops):
        return "wrong_operation"
    return "wrong_cell"


# ── Προαιρετικός LLM judge (JUDGE_MODE='llm') ─────────────────────────
_JUDGE_PROMPT = """You are grading a financial QA answer. Classify the SINGLE most likely error.
Return ONLY one label from: {labels}.

Question: {q}
Gold program: {gp}
Gold answer: {ga}
Model program: {pp}
Model answer: {pa}
Label:"""


def llm_classify(result: dict, instance: dict | None = None) -> str:
    from src.genai_client import generate
    if execution_correct(result.get("pred_answer"), result.get("gold_answer")):
        return C.CORRECT_LABEL
    prompt = _JUDGE_PROMPT.format(
        labels=", ".join(C.ERROR_CLASSES),
        q=result.get("question", ""), gp=result.get("gold_program", ""),
        ga=result.get("gold_answer"), pp=result.get("pred_program", ""),
        pa=result.get("pred_answer"),
    )
    out = generate(prompt, thinking_level="low").get("text", "").lower()
    for c in C.ERROR_CLASSES:
        if c in out:
            return c
    return classify(result, instance)  # πέσε πίσω στους κανόνες


def classify_result(result: dict, instance: dict | None = None) -> str:
    if C.JUDGE_MODE == "llm":
        return llm_classify(result, instance)
    return classify(result, instance)


if __name__ == "__main__":
    inst = {"context": "revenue 22471 and 20624 in millions ... ratio 8.1 of 56.0"}
    cases = [
        ({"gold_answer": 1847.0, "pred_answer": 1847.0}, "correct"),
        ({"gold_answer": 1847.0, "pred_answer": None, "parsed_ok": False}, "format_error"),
        ({"gold_answer": 35.0, "pred_answer": -35.0}, "sign_error"),
        ({"gold_answer": 0.099, "pred_answer": 9.9}, "scale_unit"),
        ({"gold_answer": 5.0, "pred_answer": 999999.0,
          "gold_program": "divide(8.1, 56.0)", "pred_program": "divide(8.1, 42.0)"},
         "hallucinated_constant"),
        ({"gold_answer": 5.0, "pred_answer": 3.0,
          "gold_program": "subtract(22471, 20624)", "pred_program": "add(22471, 20624)"},
         "wrong_operation"),
        ({"gold_answer": 5.0, "pred_answer": 3.0,
          "gold_program": "subtract(22471, 20624)", "pred_program": "subtract(22471, 8.1)"},
         "wrong_cell"),
    ]
    ok = True
    for res, exp in cases:
        res.setdefault("parsed_ok", True)
        got = classify(res, inst)
        flag = "OK" if got == exp else "XX"
        if got != exp: ok = False
        print(f"  {flag} expected={exp:22s} got={got}")
    print("taxonomy self-test", "PASSED" if ok else "FAILED")
