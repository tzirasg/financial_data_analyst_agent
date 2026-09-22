"""
src/planner.py — Το βήμα PLAN του βρόχου.

Δεδομένων των ταξινομημένων σφαλμάτων σε ένα config:
  1) βρες την top-1 κλάση σφάλματος,
  2) διάλεξε ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΑ ένα πραγματικό failing στιγμιότυπο αυτής της κλάσης,
  3) φτιάξε ένα patch = (γενικός κανόνας) + (διορθωτικό exemplar από το gold).

Ο κανόνας παράγεται με reflection από το ίδιο το μοντέλο (grounded στην αποτυχία),
με ντετερμινιστικό template fallback ανά κλάση αν αποτύχει η κλήση.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
import config as C


@dataclass
class Patch:
    loop: int
    error_class: str
    rule_id: str
    rule_text: str
    exemplar: dict
    note: str
    source_failure_id: str = ""


# Ντετερμινιστικά templates ανά κλάση (fallback + βάση για το reflection)
RULE_TEMPLATES = {
    "scale_unit": "Express ratios and percentages as decimal fractions (e.g. 9.9% -> 0.099) and keep units consistent with the table (watch millions vs thousands).",
    "wrong_cell": "Locate the exact row and column the question refers to before computing; verify each number against the table label.",
    "wrong_operation": "Match the operation to the wording: 'change/difference' -> subtract, 'growth/percent change' -> subtract then divide by the base, 'total' -> add.",
    "sign_error": "Keep the sign: for a change use later value minus earlier value; a decrease must stay negative.",
    "multi_step_order": "For multi-step questions compute intermediate results in order and reuse them; do not reorder or merge the operands.",
    "hallucinated_constant": "Use ONLY numbers that literally appear in the CONTEXT; never introduce a constant that is not given.",
    "format_error": "Always finish with the FINAL_ANSWER block containing a single numeric answer and the program.",
}

_REFLECT = """A financial-QA model made an error of type "{ec}".

Question: {q}
CONTEXT (excerpt): {ctx}
Correct program: {gp}
Correct answer: {ga}
Model's wrong program: {pp}
Model's wrong answer: {pa}

Write ONE short, general, imperative rule (max 25 words) that would prevent this
class of error on similar questions. Do not mention this specific example.
Return only the rule text."""


def top_error_class(graded: list[dict]) -> tuple[str | None, Counter]:
    counts = Counter(r.get("error_class") for r in graded
                     if r.get("error_class") not in (None, C.CORRECT_LABEL, "api_error"))
    if not counts:
        return None, counts
    # ντετερμινιστικό tie-break: μεγαλύτερο count, μετά αλφαβητικά
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return top, counts


def pick_failure(graded: list[dict], error_class: str) -> dict | None:
    fails = [r for r in graded if r.get("error_class") == error_class]
    fails.sort(key=lambda r: str(r.get("id")))
    return fails[0] if fails else None


def _reflect_rule(error_class: str, failure: dict, instance: dict, use_llm: bool) -> str:
    if not use_llm:
        return RULE_TEMPLATES.get(error_class, "Answer carefully and show the program.")
    try:
        from src.genai_client import generate
        ctx = (instance.get("context", "") if instance else "")[:600]
        prompt = _REFLECT.format(
            ec=error_class, q=failure.get("question", ""), ctx=ctx,
            gp=failure.get("gold_program", ""), ga=failure.get("gold_answer"),
            pp=failure.get("pred_program", ""), pa=failure.get("pred_answer"),
        )
        txt = generate(prompt, thinking_level="low").get("text", "").strip()
        # καθάρισε εισαγωγικά/newlines· κράτα λογικό μήκος
        txt = txt.strip().strip('"').replace("\n", " ").strip()
        if 8 <= len(txt) <= 300:
            return txt
    except Exception:
        pass
    return RULE_TEMPLATES.get(error_class, "Answer carefully and show the program.")


def ranked_error_classes(graded: list[dict]) -> list[str]:
    """Όλες οι κλάσεις σφάλματος, ταξινομημένες: count↓, μετά αλφαβητικά.
    Ίδιο tie-break με το top_error_class -> ο #1 εδώ == top_error_class."""
    counts = Counter(r.get("error_class") for r in graded
                     if r.get("error_class") not in (None, C.CORRECT_LABEL, "api_error"))
    return [ec for ec, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def plan_patch_for_class(loop_idx: int, graded: list[dict], error_class: str,
                         instances_by_id: dict[str, dict],
                         use_llm: bool = True) -> Patch | None:
    """Όπως το plan_patch αλλά για ΣΥΓΚΕΚΡΙΜΕΝΗ κλάση (για ranked fallback)."""
    failure = pick_failure(graded, error_class)
    if failure is None:
        return None
    inst = instances_by_id.get(failure["id"])
    ec = error_class
    rule_text = _reflect_rule(ec, failure, inst, use_llm)
    exemplar = {
        "id": f"E{loop_idx+1}",
        "error_class": ec,
        "question": failure.get("question", ""),
        "correct_program": failure.get("gold_program", ""),
        "correct_answer": failure.get("gold_answer"),
        "note": f"Earlier the model answered {failure.get('pred_answer')} "
                f"(type: {ec}); the correct result is {failure.get('gold_answer')}.",
        "added_loop": loop_idx + 1,
    }
    return Patch(
        loop=loop_idx + 1, error_class=ec, rule_id=f"R{loop_idx+1}",
        rule_text=rule_text, exemplar=exemplar, note=exemplar["note"],
        source_failure_id=failure.get("id", ""),
    )


def plan_patch(loop_idx: int, graded: list[dict],
               instances_by_id: dict[str, dict],
               use_llm: bool = True) -> Patch | None:
    """Φτιάχνει το patch για αυτόν τον κύκλο. None αν δεν υπάρχουν σφάλματα."""
    ec, counts = top_error_class(graded)
    if ec is None:
        return None  # τίποτα να διορθώσεις -> σήμα για early stop
    failure = pick_failure(graded, ec)
    inst = instances_by_id.get(failure["id"]) if failure else None

    rule_text = _reflect_rule(ec, failure, inst, use_llm)
    exemplar = {
        "id": f"E{loop_idx+1}",
        "error_class": ec,
        "question": failure.get("question", "") if failure else "",
        "correct_program": failure.get("gold_program", "") if failure else "",
        "correct_answer": failure.get("gold_answer") if failure else None,
        "note": f"Earlier the model answered {failure.get('pred_answer') if failure else '?'} "
                f"(type: {ec}); the correct result is {failure.get('gold_answer') if failure else '?'}.",
        "added_loop": loop_idx + 1,
    }
    return Patch(
        loop=loop_idx + 1,
        error_class=ec,
        rule_id=f"R{loop_idx+1}",
        rule_text=rule_text,
        exemplar=exemplar,
        note=exemplar["note"],
        source_failure_id=failure.get("id", "") if failure else "",
    )


if __name__ == "__main__":
    graded = [
        {"id": "a", "error_class": "scale_unit", "question": "pct change?",
         "gold_program": "divide(1,2)", "gold_answer": 0.5, "pred_program": "divide(1,2)", "pred_answer": 50.0},
        {"id": "b", "error_class": "scale_unit", "question": "ratio?",
         "gold_program": "divide(3,4)", "gold_answer": 0.75, "pred_program": "divide(3,4)", "pred_answer": 75.0},
        {"id": "c", "error_class": "wrong_cell", "question": "x?",
         "gold_program": "subtract(9,4)", "gold_answer": 5, "pred_program": "subtract(9,1)", "pred_answer": 8},
        {"id": "d", "error_class": C.CORRECT_LABEL},
    ]
    inst = {"a": {"context": "values 1 and 2"}, "b": {}, "c": {}}
    p = plan_patch(0, graded, inst, use_llm=False)
    print("top class:", top_error_class(graded)[0], "| counts:", dict(top_error_class(graded)[1]))
    print("patch:", p.rule_id, p.error_class, "->", p.rule_text[:60])
    print("exemplar note:", p.exemplar["note"])
