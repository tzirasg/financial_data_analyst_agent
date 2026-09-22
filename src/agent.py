"""
src/agent.py — Ο agent που απαντά ερωτήσεις FinQA.

Το prompt συντίθεται από τρία κομμάτια του action space:
  1. base_prompt.txt        (σταθερό)
  2. guidance_rules         (κανόνες που πρόσθεσε ο βρόχος)  -> μοχλός (α)
  3. exemplar_bank          (διορθωτικά παραδείγματα)        -> μοχλός (α)
Ο δεύτερος μοχλός (β) είναι το thinking_level, που περνά στην κλήση.

Τα registries μπορούν να δοθούν είτε ως in-memory λίστες (ο βρόχος κρατά
τρέχουσα κατάσταση) είτε να φορτωθούν από τα αρχεία prompts/.
"""
from __future__ import annotations
import re, json
from pathlib import Path
import yaml
import config as C
from src.genai_client import generate

# ── Φόρτωση registries ────────────────────────────────────────────────
def load_base_prompt() -> str:
    return (C.PROMPTS_DIR / "base_prompt.txt").read_text(encoding="utf-8").strip()


def load_rules() -> list[dict]:
    data = yaml.safe_load((C.PROMPTS_DIR / "guidance_rules.yaml").read_text(encoding="utf-8"))
    return (data or {}).get("rules", []) or []


def load_exemplars() -> list[dict]:
    data = json.loads((C.PROMPTS_DIR / "exemplar_bank.json").read_text(encoding="utf-8"))
    return (data or {}).get("exemplars", []) or []


# ── Σύνθεση prompt ────────────────────────────────────────────────────
def render_rules(rules: list[dict]) -> str:
    if not rules:
        return ""
    lines = ["RULES TO FOLLOW (learned from earlier mistakes):"]
    for i, r in enumerate(rules, 1):
        lines.append(f"{i}. [{r.get('error_class','')}] {r.get('rule_text','').strip()}")
    return "\n".join(lines)


def render_exemplars(exemplars: list[dict]) -> str:
    if not exemplars:
        return ""
    blocks = ["WORKED EXAMPLES (study the correct approach):"]
    for ex in exemplars:
        blocks.append(
            f"- Question: {ex.get('question','').strip()}\n"
            f"  Correct program: {ex.get('correct_program','').strip()}\n"
            f"  Correct answer: {ex.get('correct_answer','')}\n"
            f"  Why: {ex.get('note','').strip()}"
        )
    return "\n".join(blocks)


def assemble_prompt(instance: dict, rules: list[dict], exemplars: list[dict]) -> str:
    parts = [load_base_prompt()]
    rr = render_rules(rules)
    if rr:
        parts.append(rr)
    re_ = render_exemplars(exemplars)
    if re_:
        parts.append(re_)
    parts.append(
        "Now answer this one.\n\n"
        f"CONTEXT:\n{instance['context']}\n\n"
        f"QUESTION: {instance['question']}"
    )
    return "\n\n".join(parts)


# ── Parsing απάντησης ─────────────────────────────────────────────────
_ANSWER_RE = re.compile(r"answer\s*[:=]\s*([^\n\r]+)", re.IGNORECASE)
_PROGRAM_RE = re.compile(r"program\s*[:=]\s*([^\n\r]+)", re.IGNORECASE)
_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


def parse_number(s: str):
    """Παίρνει την ΚΥΡΙΟΛΕΚΤΙΚΗ αριθμητική τιμή που δηλώνει ο agent.
    ΔΕΝ μετατρέπει '%': αν ο agent γράψει '9.9%' -> 9.9 (και θα φανεί ως λάθος
    κλίμακας/format έναντι του gold 0.099· αυτό είναι μαθήσιμο σφάλμα, όχι bug)."""
    if s is None:
        return None
    # αφαιρούμε κόμματα χιλιάδων πριν το match
    m = _NUM_RE.search(s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_response(text: str) -> dict:
    """Βγάζει (pred_answer, pred_program) από το FINAL_ANSWER block (τελευταίο)."""
    ans_matches = list(_ANSWER_RE.finditer(text or ""))
    prog_matches = list(_PROGRAM_RE.finditer(text or ""))
    ans_raw = ans_matches[-1].group(1).strip() if ans_matches else None
    prog_raw = prog_matches[-1].group(1).strip() if prog_matches else None
    return {
        "pred_answer": parse_number(ans_raw),
        "pred_answer_raw": ans_raw,
        "pred_program": prog_raw,
        "parsed_ok": ans_raw is not None,
    }


# ── Κύρια συνάρτηση: απάντησε ένα στιγμιότυπο ─────────────────────────
def answer_instance(instance: dict, thinking_level: str,
                    rules: list[dict] | None = None,
                    exemplars: list[dict] | None = None,
                    model: str | None = None) -> dict:
    rules = load_rules() if rules is None else rules
    exemplars = load_exemplars() if exemplars is None else exemplars
    prompt = assemble_prompt(instance, rules, exemplars)
    res = generate(prompt, thinking_level=thinking_level, model=model)
    parsed = parse_response(res["text"])
    return {
        "id": instance["id"],
        "source": instance["source"],
        "complexity": instance["complexity"],
        "question": instance["question"],
        "gold_answer": instance["gold_answer"],
        "gold_program": instance["gold_program"],
        "pred_answer": parsed["pred_answer"],
        "pred_answer_raw": parsed["pred_answer_raw"],
        "pred_program": parsed["pred_program"],
        "parsed_ok": parsed["parsed_ok"],
        "thinking_level": thinking_level,
        "raw_text": res["text"],
        "usage": {
            "prompt_tokens": res["prompt_tokens"],
            "thought_tokens": res["thought_tokens"],
            "output_tokens": res["output_tokens"],
            "attempts": res["attempts"],
        },
    }


if __name__ == "__main__":
    # γρήγορος έλεγχος parsing χωρίς κλήση API
    demo = "reasoning...\nFINAL_ANSWER\nprogram: subtract(5829, 5735)\nanswer: 94"
    print(parse_response(demo))
