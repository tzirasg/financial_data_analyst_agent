"""
src/gate.py — Governance gate (η «πύλη» απόφασης του βρόχου).

Ένα υποψήφιο patch γίνεται ACCEPT μόνο αν ισχύουν ΚΑΙ ΤΑ ΤΡΙΑ:
  (1) ΔAccuracy ≥ MIN_DELTA_ACCURACY           (κέρδος στο D_tune, no-regression στο σύνολο)
  (2) compliance OK                            (δεν βγαίνει εκτός locked action space)
  (3) program-equivalence OK                   (δεν χαλάει ό,τι ήταν ήδη σωστό)
Αλλιώς REJECT και το config μένει ως είχε. Κάθε απόφαση καταγράφεται (auditability).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import config as C

# Λέξεις-κλειδιά εκτός action space -> non-compliant patch (blocked)
FORBIDDEN_KEYWORDS = [
    "temperature", "top_p", "top-p", "top_k", "top-k",
    "retriev", "rag", "fine-tune", "finetune", "fine tune", "re-train", "retrain",
    "web search", "browse", "tool call", "function call", "api call",
    "decompos", "change model", "switch model", "gpt", "claude", "different model",
]


@dataclass
class GateDecision:
    accepted: bool
    delta_accuracy: float
    compliance_ok: bool
    no_regression_ok: bool
    n_regressions: int
    n_fixed: int
    reason: str
    blocked_reason: str = ""
    # ── dev-guard πεδία (πείραμα a+b· ουδέτερα όταν δεν χρησιμοποιείται) ──
    dev_ok: bool = True
    dev_prev_acc: float = 0.0
    dev_cand_acc: float = 0.0
    dev_delta: float = 0.0


def check_compliance(patch) -> tuple[bool, str]:
    """Σκανάρει το κείμενο του patch για αναφορές εκτός των 2 επιτρεπτών μοχλών."""
    text = f"{getattr(patch,'rule_text','')} {getattr(patch,'note','')}".lower()
    for kw in FORBIDDEN_KEYWORDS:
        if kw in text:
            return False, f"non-compliant keyword: '{kw}'"
    return True, "compliant"


def regression_report(prev_results: list[dict], cand_results: list[dict]) -> tuple[int, int]:
    """Επιστρέφει (n_regressions, n_fixed) συγκρίνοντας ανά id (χρειάζεται 'correct')."""
    prev = {r["id"]: bool(r.get("correct")) for r in prev_results}
    n_regr = n_fixed = 0
    for r in cand_results:
        was = prev.get(r["id"])
        now = bool(r.get("correct"))
        if was is True and not now:
            n_regr += 1
        elif was is False and now:
            n_fixed += 1
    return n_regr, n_fixed


def evaluate(prev_acc: float, cand_acc: float,
             prev_results: list[dict], cand_results: list[dict],
             patch) -> GateDecision:
    delta = round(cand_acc - prev_acc, 6)
    comp_ok, comp_reason = check_compliance(patch)
    n_regr, n_fixed = regression_report(prev_results, cand_results)
    noreg_ok = n_regr <= C.GATE_MAX_REGRESSIONS
    gain_ok = delta >= C.MIN_DELTA_ACCURACY

    accepted = gain_ok and comp_ok and noreg_ok
    if accepted:
        reason = f"ACCEPT: Δacc={delta:+.3f}, fixed={n_fixed}, regressions={n_regr}, {comp_reason}"
    else:
        why = []
        if not gain_ok: why.append(f"Δacc={delta:+.3f} < {C.MIN_DELTA_ACCURACY}")
        if not comp_ok: why.append(comp_reason)
        if not noreg_ok: why.append(f"regressions={n_regr} > {C.GATE_MAX_REGRESSIONS}")
        reason = "REJECT: " + "; ".join(why)
    return GateDecision(
        accepted=accepted, delta_accuracy=delta,
        compliance_ok=comp_ok, no_regression_ok=noreg_ok,
        n_regressions=n_regr, n_fixed=n_fixed, reason=reason,
        blocked_reason="" if comp_ok else comp_reason,
    )


def evaluate_with_dev(prev_acc: float, cand_acc: float,
                      prev_results: list[dict], cand_results: list[dict], patch,
                      dev_prev_acc: float, dev_cand_acc: float) -> GateDecision:
    """Validation-guarded gate (πείραμα a+b):
    ίδιο tune-gate + επιπλέον όρος: το candidate ΔΕΝ ρίχνει το D_dev exec accuracy
    πάνω από C.DEV_MAX_ACC_DROP. ACCEPT μόνο αν περνούν ΚΑΙ ΤΑ ΔΥΟ."""
    d = evaluate(prev_acc, cand_acc, prev_results, cand_results, patch)
    dev_delta = round(dev_cand_acc - dev_prev_acc, 6)
    dev_ok = dev_delta >= -C.DEV_MAX_ACC_DROP
    d.dev_ok = dev_ok
    d.dev_prev_acc = round(dev_prev_acc, 4)
    d.dev_cand_acc = round(dev_cand_acc, 4)
    d.dev_delta = dev_delta
    if d.accepted and dev_ok:
        d.reason = (f"ACCEPT: Δacc={d.delta_accuracy:+.3f} tune (fixed={d.n_fixed}, "
                    f"regr={d.n_regressions}); dev {dev_prev_acc:.3f}->{dev_cand_acc:.3f} "
                    f"(Δ{dev_delta:+.3f}) OK; compliant")
    elif d.accepted and not dev_ok:
        # πέρασε το tune αλλά ΚΟΒΕΤΑΙ από dev-regression
        d.accepted = False
        d.reason = (f"REJECT(dev): tune Δacc={d.delta_accuracy:+.3f} OK αλλά dev "
                    f"{dev_prev_acc:.3f}->{dev_cand_acc:.3f} (Δ{dev_delta:+.3f}) < -{C.DEV_MAX_ACC_DROP}")
    else:
        d.reason = "REJECT(tune): " + d.reason.replace("REJECT: ", "")
    return d
